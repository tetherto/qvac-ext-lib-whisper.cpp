// Memory-fit preflight (include/parakeet/fit.h): project the model + workload
// against the device memory available right now, without reading weight data.
//
// The projection mirrors the real runtime path by construction rather than by
// formula: it drives the same loader (metadata-only), the same graph builders,
// and the same allocation policies through ggml's size-only APIs
// (ggml_backend_alloc_ctx_tensors_from_buft_size / ggml_gallocr_reserve_n_size),
// so it tracks runtime changes instead of drifting from them.

#include "parakeet/fit.h"

#include "long_form.h"
#include "parakeet_ctc.h"
#include "parakeet_eou.h"
#include "parakeet_log.h"
#include "parakeet_sortformer.h"
#include "parakeet_tdt.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

namespace parakeet {

const char * fit_status_name(FitStatus status) {
    switch (status) {
        case FitStatus::Success: return "success";
        case FitStatus::Failure: return "failure";
        case FitStatus::Error:   return "error";
    }
    return "error";
}

namespace {

std::string fmt_mib(uint64_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f MiB", (double) bytes / (1024.0 * 1024.0));
    return buf;
}

// Post-subsampling encoder frame count: the same three stride-2 conv
// recurrence run_encoder / coreml_encoder_out_frames use to size the graph.
int encoder_out_frames(const EncoderConfig & enc, long long n_mel_frames) {
    auto next = [&](long long len) {
        // _conv_out_len(len, k=3, s=2, p=1) == (len - 1) / 2 + 1
        return enc.causal_downsampling ? (len / 2 + 1) : ((len - 1) / 2 + 1);
    };
    long long t = next(next(next(n_mel_frames)));
    if (t < 1) t = 1;
    if (t > std::numeric_limits<int>::max()) t = std::numeric_limits<int>::max();
    return (int) t;
}

}  // namespace

FitResult fit_params(const FitOptions & opts) {
    FitResult r;

    if (opts.model_gguf_path.empty() || !(opts.audio_seconds > 0.0f)) {
        r.reason = "invalid-arguments";
        return r;
    }

    if (!opts.backends_dir.empty()) {
        set_backends_directory(opts.backends_dir);
    }

    // ── Metadata-only load: backend resolution + tensor wiring + weight sizing
    ParakeetCtcModel model;
    GgufLoadMeasure  lm;
    {
        const int rc = load_from_gguf_metadata_only(
            opts.model_gguf_path, model, opts.n_threads, opts.n_gpu_layers,
            opts.verbose, lm);
        if (rc == 10) {  // CPU backend init failed: nothing registered at all
            r.reason = "no-backend-device";
            return r;
        }
        if (rc != 0) {
            r.reason = "model-unreadable";
            return r;
        }
    }

    r.model_type    = model_type_name(model.model_type);
    r.model_variant = model.model_variant;
    r.device_name   = model_active_backend_name(model);

    ggml_backend_t backend = model_active_backend(model);
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    if (!dev) {
        r.reason = "no-backend-device";
        return r;
    }
    r.device_is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    {
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);
        r.device_free_bytes  = free_b;
        r.device_total_bytes = total_b;
    }

    // ── Workload → worst-case single-encode window ─────────────────────────
    const MelConfig & mel = model.mel_cfg;
    const long long total_mel = std::max<long long>(
        1, (long long) std::ceil((double) opts.audio_seconds *
                                 (double) mel.sample_rate / (double) mel.hop_length));

    // The transcribe paths bound device memory by sliding the encoder over
    // long inputs (resolve_long_form_plan + run_encoder_windowed). Offline
    // diarize (Sortformer) runs the encoder AND the O(T^2) head over the full
    // input in one graph, so its projection must use the full length.
    long long worst_window_mel = total_mel;
    if (model.model_type != ParakeetModelType::SORTFORMER) {
        const LongFormPlan plan = resolve_long_form_plan_frames(
            opts.long_form_window_frames, opts.long_form_context_frames,
            model.encoder_cfg.pos_emb_max_len, model.encoder_cfg.subsampling_factor,
            total_mel);
        if (plan.enabled) {
            const int center_mel = plan.center_frames  * plan.sub;
            const int ctx_mel    = plan.context_frames * plan.sub;
            const int n_units    = (int) std::min<long long>(
                total_mel, std::numeric_limits<int>::max());
            worst_window_mel = 0;
            for (const LongFormWindow & w :
                 plan_long_form_windows(n_units, center_mel, ctx_mel)) {
                worst_window_mel = std::max<long long>(worst_window_mel, w.window_len);
            }
            if (worst_window_mel <= 0) worst_window_mel = total_mel;
        }
    }
    if (worst_window_mel > std::numeric_limits<int>::max()) {
        r.reason = "invalid-arguments";  // absurd single-window projection
        return r;
    }
    const int window_mel      = (int) worst_window_mel;
    const int window_enc      = encoder_out_frames(model.encoder_cfg, window_mel);
    const int total_enc       = encoder_out_frames(model.encoder_cfg, total_mel);

    // ── Encoder compute: build + size the real worst-case window graph ─────
    size_t enc_compute = 0;
    if (measure_encoder_compute(model, window_mel, mel.n_mels, enc_compute) != 0) {
        r.reason = "measurement-failed";
        return r;
    }

    // ── Decoder-side buffers, per model type ───────────────────────────────
    DecoderFitMeasure dm;
    switch (model.model_type) {
        case ParakeetModelType::CTC:
            break;  // the CTC head lives inside the encoder graph, already measured
        case ParakeetModelType::RNNT:
        case ParakeetModelType::TDT:
            if (tdt_measure_runtime(model, window_enc, dm) != 0) {
                r.reason = "measurement-failed";
                return r;
            }
            break;
        case ParakeetModelType::EOU:
            if (eou_measure_runtime(model, window_enc, dm) != 0) {
                r.reason = "measurement-failed";
                return r;
            }
            break;
        case ParakeetModelType::SORTFORMER: {
            size_t head_bytes = 0;
            if (sortformer_measure_head(model, window_enc, head_bytes) != 0) {
                r.reason = "measurement-failed";
                return r;
            }
            dm.device_compute_bytes = head_bytes;
            break;
        }
    }

    // ── Host-side extras (scale with the FULL input, unlike the windowed
    //    device buffers): input samples, mel slab, stitched encoder output,
    //    CTC logits, encoder-graph host mirrors, CPU-path decoder weights. ──
    {
        uint64_t host = 0;
        host += (uint64_t) std::llround((double) opts.audio_seconds * mel.sample_rate)
              * sizeof(float);                                        // input samples
        host += (uint64_t) total_mel * mel.n_mels * sizeof(float);    // mel spectrogram
        host += (uint64_t) total_enc * model.encoder_cfg.d_model * sizeof(float);  // encoder_out
        if (model.model_type == ParakeetModelType::CTC) {
            host += (uint64_t) total_enc * model.vocab_size * sizeof(float);       // logits
        }
        // EncoderGraph host mirrors at the worst window: relative positional
        // encoding and (chunked-attention models) the T x T attention mask.
        host += (uint64_t) (2 * (long long) window_enc - 1) *
                model.encoder_cfg.d_model * sizeof(float);
        if (model.encoder_cfg.att_chunked_limited &&
            model.encoder_cfg.att_context_left  >= 0 &&
            model.encoder_cfg.att_context_right >= 0) {
            host += (uint64_t) window_enc * window_enc * sizeof(float);
        }
        host += dm.host_bytes;           // CPU decode path: dequantised f32 weights
        host += lm.sortformer_cpu_bytes; // Mali-Vulkan CPU head copy (host RAM)
        r.host_bytes = host;
    }

    r.device.weights_bytes         = lm.weights_bytes + lm.repack_bytes;
    r.device.encoder_compute_bytes = enc_compute;
    r.device.decoder_state_bytes   = dm.device_state_bytes;
    r.device.decoder_compute_bytes = dm.device_compute_bytes;
    r.device.total_bytes = r.device.weights_bytes + r.device.encoder_compute_bytes +
                           r.device.decoder_state_bytes + r.device.decoder_compute_bytes;

    // ── Verdict ────────────────────────────────────────────────────────────
    // On a CPU device the "device" buffers and the host extras compete for
    // the same physical RAM, so both count against the free figure.
    uint64_t required = r.device.total_bytes + opts.margin_bytes;
    if (r.device_is_cpu) {
        required += r.host_bytes;
    }
    r.fits   = required <= r.device_free_bytes;
    r.status = r.fits ? FitStatus::Success : FitStatus::Failure;
    r.reason = r.fits ? "fits" : "does-not-fit";

    // ── Report ─────────────────────────────────────────────────────────────
    {
        std::string s;
        char line[256];
        std::snprintf(line, sizeof(line), "model:    %s%s%s%s\n",
                      r.model_type.c_str(),
                      r.model_variant.empty() ? "" : " (",
                      r.model_variant.c_str(),
                      r.model_variant.empty() ? "" : ")");
        s += line;
        std::snprintf(line, sizeof(line), "device:   %s (%s), free %s / total %s\n",
                      r.device_name.c_str(), r.device_is_cpu ? "CPU" : "GPU",
                      fmt_mib(r.device_free_bytes).c_str(),
                      fmt_mib(r.device_total_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line),
                      "workload: %.1f s audio -> worst window %d mel frames (%d encoder frames)\n",
                      (double) opts.audio_seconds, window_mel, window_enc);
        s += line;
        s += "device projection:\n";
        std::snprintf(line, sizeof(line), "  weights:         %14s\n",
                      fmt_mib(r.device.weights_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  encoder compute: %14s\n",
                      fmt_mib(r.device.encoder_compute_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  decoder state:   %14s\n",
                      fmt_mib(r.device.decoder_state_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  decoder compute: %14s\n",
                      fmt_mib(r.device.decoder_compute_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "  total:           %14s\n",
                      fmt_mib(r.device.total_bytes).c_str());
        s += line;
        std::snprintf(line, sizeof(line), "host extras:       %14s%s\n",
                      fmt_mib(r.host_bytes).c_str(),
                      r.device_is_cpu ? " (same RAM pool as the device projection)" : "");
        s += line;
        std::snprintf(line, sizeof(line), "margin:            %14s\n",
                      fmt_mib(opts.margin_bytes).c_str());
        s += line;
        if (r.fits) {
            std::snprintf(line, sizeof(line), "verdict: FITS (headroom %s)\n",
                          fmt_mib(r.device_free_bytes - required).c_str());
        } else {
            std::snprintf(line, sizeof(line), "verdict: DOES NOT FIT (short by %s)\n",
                          fmt_mib(required - r.device_free_bytes).c_str());
        }
        s += line;
        r.report = std::move(s);
    }

    return r;
}

}  // namespace parakeet
