// QVAC (see PATCHES.md): fit-projection parity tests for whisper_fit_params
// (include/whisper.h) -- assert that the metadata-only memory projection
// matches what a REAL load actually allocates, byte for byte, where the
// projection is exact by construction:
//
//   1. whisper_fit_params returns a projection (never Error) for a readable
//      model, with non-zero weight/kv/compute figures and a report;
//   2. byte parity -- a projection at n_decoders = 1 must equal
//      whisper_fit_actual's measurement of a real
//      whisper_init_from_file_with_params + whisper_init_state on every
//      component: weights (same buffer-type sizing loop), kv caches (same
//      from_buft sizing), the four schedulers' compute buffers
//      (ggml_backend_sched_reserve_size vs the buffers
//      ggml_backend_sched_alloc_graph committed), and the host-overflow
//      split;
//   3. n_decoders worst case -- a projection at the whisper_full default
//      (5 decoders) must price a strictly larger self-attention KV cache
//      (the (n+2)x recreation) and never a smaller decode graph;
//   4. workload scaling -- longer audio grows ONLY host_bytes; the device
//      projection is init-time-fixed (whisper allocates its schedulers at
//      the model's full n_audio_ctx regardless of input length);
//   5. VAD -- with a VAD model the projection gains a non-zero vad component
//      that also matches the real VAD context byte for byte.
//
// Usage: test-whisper-fit-params [model.bin] [vad-model.bin] [gpu]
// (CMake registers the CPU form against the committed for-tests fixtures, so
// this runs model-free-download in CI; pass "gpu" manually to check parity on
// a GPU backend.)
// Exit 0 on success; non-zero with a FAIL line per broken invariant.

#include "whisper.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

void expect_eq(uint64_t projected, uint64_t actual, const std::string & what) {
    if (projected != actual) {
        fail(what + ": projected " + std::to_string(projected) +
             " != actual " + std::to_string(actual));
    }
}

void log_callback_null(ggml_log_level, const char *, void *) {}

}  // namespace

int main(int argc, char ** argv) {
#ifndef WHISPER_FIT_TEST_MODEL
#define WHISPER_FIT_TEST_MODEL ""
#endif
#ifndef WHISPER_FIT_TEST_VAD_MODEL
#define WHISPER_FIT_TEST_VAD_MODEL ""
#endif
    std::string model_path     = WHISPER_FIT_TEST_MODEL;
    std::string vad_model_path = WHISPER_FIT_TEST_VAD_MODEL;
    bool        use_gpu        = false;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) vad_model_path = argv[2];
    if (argc > 3 && std::strcmp(argv[3], "gpu") == 0) use_gpu = true;

    if (model_path.empty()) {
        std::fprintf(stderr, "usage: %s <model.bin> [vad-model.bin] [gpu]\n", argv[0]);
        return 2;
    }

    whisper_log_set(log_callback_null, nullptr);

    whisper_fit_options opts = whisper_fit_default_options();
    opts.model_path    = model_path.c_str();
    opts.use_gpu       = use_gpu;
    opts.n_decoders    = 1; // match whisper_fit_actual's post-init resident set
    opts.audio_seconds = 30.0f;

    // 1. A readable model always yields a projection.
    whisper_fit_result fit;
    const int rc = whisper_fit_params(&opts, &fit);
    expect(rc == (int) fit.status, "return value == status (exit-code contract)");
    expect(fit.status != WHISPER_FIT_ERROR,
           std::string("whisper_fit_params returned Error (") + fit.reason + ") for a readable model");
    expect(fit.device.weights_bytes > 0, "projected weights_bytes == 0");
    expect(fit.device.kv_bytes > 0,      "projected kv_bytes == 0");
    expect(fit.device.compute_bytes > 0, "projected compute_bytes == 0");
    expect(fit.device.vad_bytes == 0,    "projected vad_bytes != 0 without a VAD model");
    expect(fit.device.total_bytes >= fit.device.weights_bytes + fit.device.kv_bytes,
           "projected total < weights + kv");
    expect(fit.host_bytes > 0,           "projected host_bytes == 0");
    expect(fit.report[0] != '\0',        "empty report");
    expect(fit.model_type[0] != '\0',    "empty model_type");
    expect(fit.device_total_bytes > 0,   "device_total_bytes == 0");
    if (g_failures) {
        return g_failures; // nothing below is meaningful without a projection
    }
    std::printf("%s", fit.report);

    // 2. Byte parity against a real load + state init.
    {
        whisper_fit_breakdown actual;
        if (whisper_fit_actual(&opts, &actual) != 0) {
            fail("whisper_fit_actual (real load) failed");
            return g_failures;
        }
        expect_eq(fit.device.weights_bytes, actual.weights_bytes, "weights parity");
        expect_eq(fit.device.kv_bytes,      actual.kv_bytes,      "kv parity");
        expect_eq(fit.device.compute_bytes, actual.compute_bytes, "compute parity");
        expect_eq(fit.device.total_bytes,   actual.total_bytes,   "total parity");
        expect_eq(fit.device.host_overflow_bytes, actual.host_overflow_bytes,
                  "host-overflow parity");
    }

    // 3. Worst-case decoders: the (n+2)x KV recreation must be priced in.
    {
        whisper_fit_options wopts = opts;
        wopts.n_decoders = 5; // whisper_full default best_of/beam_size
        whisper_fit_result worst;
        whisper_fit_params(&wopts, &worst);
        expect(worst.status != WHISPER_FIT_ERROR, "n_decoders=5 projection is Error");
        expect(worst.device.kv_bytes > fit.device.kv_bytes,
               "n_decoders=5 must project a larger self-attention KV cache");
        expect(worst.device.compute_bytes >= fit.device.compute_bytes,
               "n_decoders=5 must never project a smaller decode graph");
        expect(worst.device.weights_bytes == fit.device.weights_bytes,
               "n_decoders must not change the weights projection");
    }

    // 4. Longer audio grows only the host side; the device set is fixed.
    {
        whisper_fit_options lopts = opts;
        lopts.audio_seconds = 3600.0f;
        whisper_fit_result longer;
        whisper_fit_params(&lopts, &longer);
        expect(longer.status != WHISPER_FIT_ERROR, "long-audio projection is Error");
        expect(longer.device.total_bytes == fit.device.total_bytes,
               "device projection must not grow with audio_seconds");
        expect(longer.host_bytes > fit.host_bytes,
               "host_bytes must grow with audio_seconds");
    }

    // 5. VAD component, with its own byte-parity gate.
    if (!vad_model_path.empty()) {
        whisper_fit_options vopts = opts;
        vopts.vad_model_path = vad_model_path.c_str();
        whisper_fit_result vfit;
        whisper_fit_params(&vopts, &vfit);
        expect(vfit.status != WHISPER_FIT_ERROR, "VAD projection is Error");
        expect(vfit.device.vad_bytes + vfit.device.host_overflow_bytes >
               fit.device.vad_bytes + fit.device.host_overflow_bytes,
               "VAD projection added no bytes anywhere");
        expect(vfit.device.weights_bytes == fit.device.weights_bytes,
               "VAD must not change the whisper weights projection");

        whisper_fit_breakdown vactual;
        if (whisper_fit_actual(&vopts, &vactual) != 0) {
            fail("whisper_fit_actual with VAD failed");
        } else {
            expect_eq(vfit.device.vad_bytes, vactual.vad_bytes, "vad parity");
            expect_eq(vfit.device.host_overflow_bytes, vactual.host_overflow_bytes,
                      "vad host-overflow parity");
        }

        // an unreadable VAD model must be Error, never a silent no-VAD fit
        whisper_fit_options bopts = opts;
        bopts.vad_model_path = "/nonexistent/vad.bin";
        whisper_fit_result bad;
        expect(whisper_fit_params(&bopts, &bad) == (int) WHISPER_FIT_ERROR &&
               std::strcmp(bad.reason, "vad-model-unreadable") == 0,
               "missing VAD model must be Error/vad-model-unreadable");
    }

    if (g_failures == 0) {
        std::printf("test-whisper-fit-params: all checks passed\n");
    }
    return g_failures;
}
