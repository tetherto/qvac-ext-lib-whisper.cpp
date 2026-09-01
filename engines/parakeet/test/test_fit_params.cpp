// Fit-projection parity tests (include/parakeet/fit.h): assert that the
// metadata-only memory projection matches what a REAL load actually
// allocates, byte for byte where the projection is exact by construction:
//
//   1. fit_params returns a projection (never Error) for a readable GGUF,
//      with non-zero weight and encoder-compute figures and a report;
//   2. weights parity -- the projected weight bytes equal the allocated
//      weight buffer bytes of a real load_from_gguf on the same backend
//      (both sides run the same buffer-type sizing, so this is exact);
//   3. encoder-compute parity -- the projected worst-case window graph equals
//      the gallocr buffer a real run_encoder reserves at the same mel frame
//      count (ggml_gallocr_reserve_n_size is reserve's size-only twin);
//   4. device-side saturation -- once the audio exceeds one long-form window,
//      the projected device bytes stop growing (only host extras keep
//      growing), which is the property callers rely on to fit long audio --
//      and the windowed projection prices the RESIDENT graph cache
//      (run_encoder keeps up to 3 window graphs alive), not a single graph;
//   5. a missing model file is Error/"model-unreadable", never Success;
//   6. Sortformer only: the projected head compute equals the scheduler
//      buffer a real diarize actually reserves (the head allocates through
//      the shared sched, and the projection uses the sched's size-only twin).
//
// Usage: test-fit-params <model.gguf> [n_gpu_layers]
// (CMake registers the CPU form, n_gpu_layers omitted = 0 -- the only backend
// every CI lane has; pass e.g. 99 manually to check parity on a GPU backend.)
// Exit 0 on success; non-zero with a FAIL line per broken invariant.

#include "parakeet/fit.h"
#include "long_form.h"
#include "parakeet_ctc.h"
#include "parakeet_sortformer.h"

#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

// Same three-stride-2-conv recurrence run_encoder uses (see
// parakeet_fit.cpp::encoder_out_frames).
int encoder_out_frames(const parakeet::EncoderConfig & enc, long long n_mel_frames) {
    auto next = [&](long long len) {
        return enc.causal_downsampling ? (len / 2 + 1) : ((len - 1) / 2 + 1);
    };
    long long t = next(next(next(n_mel_frames)));
    return t < 1 ? 1 : (int) t;
}

}  // namespace

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    const std::string model_path   = argv[1];
    const int         n_gpu_layers = argc > 2 ? std::atoi(argv[2]) : 0;

    constexpr float kAudioSeconds = 10.0f;

    parakeet::FitOptions fopts;
    fopts.model_gguf_path = model_path;
    fopts.audio_seconds   = kAudioSeconds;
    fopts.n_gpu_layers    = n_gpu_layers;

    // 1. A readable GGUF always yields a projection.
    const parakeet::FitResult fit = parakeet::fit_params(fopts);
    expect(fit.status != parakeet::FitStatus::Error,
           "fit_params returned Error (" + fit.reason + ") for a readable model");
    expect(fit.device.weights_bytes > 0,        "projected weights_bytes == 0");
    expect(fit.device.encoder_compute_bytes > 0,"projected encoder_compute_bytes == 0");
    expect(fit.device.total_bytes >= fit.device.weights_bytes + fit.device.encoder_compute_bytes,
           "projected total < weights + encoder compute");
    expect(!fit.report.empty(),                 "empty report");
    expect(!fit.model_type.empty(),             "empty model_type");
    expect(fit.device_total_bytes > 0,          "device_total_bytes == 0");
    if (g_failures) {
        return g_failures;  // nothing below is meaningful without a projection
    }
    std::printf("%s", fit.report.c_str());

    // 2 + 3. Parity against a real load on the same (CPU) backend.
    {
        parakeet::ParakeetCtcModel model;
        if (parakeet::load_from_gguf(model_path, model, /*n_threads=*/0,
                                     n_gpu_layers, /*verbose=*/false) != 0) {
            fail("real load_from_gguf failed");
            return g_failures;
        }

        const size_t real_weights = parakeet::model_weights_buffer_bytes(model);
        expect(real_weights == fit.device.weights_bytes,
               "weights parity: projected " + std::to_string(fit.device.weights_bytes) +
               " != allocated " + std::to_string(real_weights));

        // Reproduce fit_params' workload arithmetic: 10 s fits one window on
        // every shipped model, so the projected graph is the single-pass graph
        // at ceil(seconds * sample_rate / hop) mel frames.
        const int n_mels = model.mel_cfg.n_mels;
        const int n_mel_frames = (int) std::ceil(
            (double) kAudioSeconds * model.mel_cfg.sample_rate / model.mel_cfg.hop_length);
        std::vector<float> zero_mel((size_t) n_mel_frames * n_mels, 0.0f);
        parakeet::EncoderOutputs enc_out;
        if (parakeet::run_encoder(model, zero_mel.data(), n_mel_frames, n_mels,
                                  enc_out, /*max_layers=*/-1,
                                  /*capture_intermediates=*/false) != 0) {
            fail("real run_encoder failed");
            return g_failures;
        }
        const size_t real_compute = parakeet::model_encoder_compute_buffer_bytes(model);
        expect(real_compute == fit.device.encoder_compute_bytes,
               "encoder compute parity: projected " +
               std::to_string(fit.device.encoder_compute_bytes) +
               " != reserved " + std::to_string(real_compute));

        // 4b. The windowed projection prices run_encoder's RESIDENT graph
        //     cache (first + middle + last window graphs stay alive in the
        //     3-slot LRU), so at >= 3 windows it must exceed the single
        //     worst-window measurement.
        if (fit.model_type != "sortformer") {
            const long long total_mel_600 = (long long) std::ceil(
                600.0 * model.mel_cfg.sample_rate / model.mel_cfg.hop_length);
            const parakeet::LongFormPlan plan = parakeet::resolve_long_form_plan_frames(
                0, 0, model.encoder_cfg.pos_emb_max_len,
                model.encoder_cfg.subsampling_factor, total_mel_600);
            if (plan.enabled) {
                const int middle_mel =
                    (plan.center_frames + 2 * plan.context_frames) * plan.sub;
                size_t single_window = 0;
                if (parakeet::measure_encoder_compute(model, middle_mel, n_mels,
                                                      single_window) != 0) {
                    fail("measure_encoder_compute failed at the middle window length");
                } else {
                    parakeet::FitOptions w = fopts;
                    w.audio_seconds = 600.0f;
                    const parakeet::FitResult fw = parakeet::fit_params(w);
                    expect(fw.status != parakeet::FitStatus::Error,
                           "600 s windowed projection errored");
                    expect(fw.device.encoder_compute_bytes > single_window,
                           "windowed projection does not price the resident graph cache: " +
                           std::to_string(fw.device.encoder_compute_bytes) +
                           " <= single worst window " + std::to_string(single_window));
                }
            }
        }

        // 6. Sortformer head parity: the projection prices the head with the
        //    scheduler's size-only reserve; a real diarize must land on the
        //    same scheduler buffer, byte for byte. (Single-backend runs only:
        //    with a GPU + CPU sched the projection splits the CPU-fallback
        //    portion into host_bytes, which this sum cannot isolate.)
        if (fit.model_type == "sortformer" && n_gpu_layers == 0) {
            const long long total_mel = (long long) std::ceil(
                (double) kAudioSeconds * model.mel_cfg.sample_rate / model.mel_cfg.hop_length);
            const int T = encoder_out_frames(model.encoder_cfg, total_mel);
            const int D = model.encoder_cfg.sortformer_fc_d_model;
            std::vector<float> zero_enc((size_t) T * D, 0.0f);
            parakeet::SortformerDiarizationOptions sopts;
            parakeet::SortformerDiarizationResult  sres;
            if (parakeet::sortformer_diarize_ggml(model, zero_enc.data(), T, D,
                                                  sopts, sres) != 0) {
                fail("real sortformer_diarize_ggml failed");
            } else {
                ggml_backend_sched_t sched = parakeet::model_sched(model);
                size_t real_head = 0;
                const int nb = sched ? ggml_backend_sched_get_n_backends(sched) : 0;
                for (int i = 0; i < nb; ++i) {
                    real_head += ggml_backend_sched_get_buffer_size(
                        sched, ggml_backend_sched_get_backend(sched, i));
                }
                expect(real_head == fit.device.decoder_compute_bytes,
                       "sortformer head parity: projected " +
                       std::to_string(fit.device.decoder_compute_bytes) +
                       " != sched reserved " + std::to_string(real_head));
            }
        }
    }

    // 4. Device-side projection saturates at the long-form window; host extras
    //    keep growing with the audio. (Sortformer diarization does not window
    //    -- its device bytes DO grow -- but the CI fixtures here are
    //    transcription models.)
    if (fit.model_type != "sortformer") {
        parakeet::FitOptions a = fopts;  a.audio_seconds = 600.0f;
        parakeet::FitOptions b = fopts;  b.audio_seconds = 7200.0f;
        const parakeet::FitResult fa = parakeet::fit_params(a);
        const parakeet::FitResult fb = parakeet::fit_params(b);
        expect(fa.status != parakeet::FitStatus::Error, "600 s projection errored");
        expect(fb.status != parakeet::FitStatus::Error, "7200 s projection errored");
        expect(fa.device.total_bytes == fb.device.total_bytes,
               "device projection did not saturate at the long-form window: 600 s -> " +
               std::to_string(fa.device.total_bytes) + ", 7200 s -> " +
               std::to_string(fb.device.total_bytes));
        expect(fb.host_bytes > fa.host_bytes,
               "host extras did not grow with audio length");
    }

    // 5. Errors are reported as Error, never Success.
    {
        parakeet::FitOptions bad = fopts;
        bad.model_gguf_path = model_path + ".does-not-exist";
        const parakeet::FitResult fr = parakeet::fit_params(bad);
        expect(fr.status == parakeet::FitStatus::Error, "missing model was not Error");
        expect(fr.reason == "model-unreadable",
               "missing model reason was '" + fr.reason + "'");
        expect(!fr.fits, "missing model reported fits");
    }

    if (g_failures == 0) {
        std::printf("test-fit-params: all checks passed\n");
    }
    return g_failures;
}
