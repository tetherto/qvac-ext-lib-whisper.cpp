// Parity/replay driver for the MiniMax-Music3 engine.
//
// Modes:
//   replay: force a recorded prompt (token ids), semantic/acoustic codes and
//           per-window initial noise through the native pipeline, dumping the
//           per-window final latents and the stitched audio. Feeding the
//           official Diffusers run's tokens/codes/noise lets the native
//           synthesis stack be compared 1:1 against the reference.
//   full:   run the native pipeline end-to-end from --caption/--lyrics.
//
// Outputs (under --out):
//   audio.wav            stitched stereo 16-bit WAV
//   window-<k>.f32       per-window final latents (128 x L, channel-major)
//   frame-hiddens.f32    AR conditioning hiddens (frames x 8 x 4096)
//   semantic.i32 / acoustic.i32   emitted codes

#include "minimax/backend.h"
#include "minimax/logic.h"
#include "minimax/mm3-pipeline.h"
#include "minimax/request-utils.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_raw(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    std::vector<T> data((size_t) bytes / sizeof(T));
    f.read(reinterpret_cast<char *>(data.data()), (std::streamsize) (data.size() * sizeof(T)));
    return data;
}

template <typename T>
void write_raw(const std::string & path, const T * data, size_t count) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data), (std::streamsize) (count * sizeof(T)));
}

void write_wav(const std::string & path, const std::vector<float> & planar, int64_t samples, int rate) {
    std::vector<int16_t> pcm((size_t) samples * 2);
    for (int64_t i = 0; i < samples; ++i) {
        const float l = planar[(size_t) i];
        const float r = planar[(size_t) (samples + i)];
        pcm[(size_t) i * 2]     = (int16_t) (std::max(-1.0f, std::min(1.0f, l)) * 32767.0f);
        pcm[(size_t) i * 2 + 1] = (int16_t) (std::max(-1.0f, std::min(1.0f, r)) * 32767.0f);
    }
    const uint32_t data_bytes = (uint32_t) (pcm.size() * sizeof(int16_t));
    const uint32_t byte_rate = (uint32_t) rate * 2 * 2;
    std::ofstream f(path, std::ios::binary);
    uint32_t u32;
    uint16_t u16;
    f.write("RIFF", 4);
    u32 = 36 + data_bytes;      f.write((char *) &u32, 4);
    f.write("WAVEfmt ", 8);
    u32 = 16;                   f.write((char *) &u32, 4);
    u16 = 1;                    f.write((char *) &u16, 2);
    u16 = 2;                    f.write((char *) &u16, 2);
    u32 = (uint32_t) rate;      f.write((char *) &u32, 4);
    u32 = byte_rate;            f.write((char *) &u32, 4);
    u16 = 4;                    f.write((char *) &u16, 2);
    u16 = 16;                   f.write((char *) &u16, 2);
    f.write("data", 4);
    f.write((char *) &data_bytes, 4);
    f.write((const char *) pcm.data(), data_bytes);
}

const char * arg_value(int argc, char ** argv, const char * name, const char * fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return fallback;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string models  = arg_value(argc, argv, "--models", "");
    const std::string out_dir = arg_value(argc, argv, "--out", ".");
    const std::string mode    = arg_value(argc, argv, "--mode", "full");
    if (models.empty()) {
        fprintf(stderr,
                "usage: mm3-replay --models <dir> --out <dir> --mode replay|full\n"
                "  replay: --tokens <i32 file> --semantic <i32 file> --acoustic <i32 file>\n"
                "          [--noise <f32 file>]... (per window)\n"
                "  full:   --caption <text> [--lyrics <text>]\n"
                "  common: [--seed N] [--steps N] [--max-frames N] [--threads N] [--device cpu|gpu|auto]\n");
        return 1;
    }

    const int threads = atoi(arg_value(argc, argv, "--threads", "0"));
    backend_configure_cpu(threads > 0 ? threads : (int) std::thread::hardware_concurrency(), "");
    backend_configure_device(arg_value(argc, argv, "--device", ""));

    MM3Model model;
    const tts_cpp::minimax::detail::ModelPair pair =
        tts_cpp::minimax::detail::resolve_model_pair(models, "", "");
    model.models_dir = models;
    mm3_probe_file(pair.lm, &model.lm_file, &model.lm_cfg, nullptr, &model.meta_errors);
    mm3_probe_file(pair.synth, &model.synth_file, nullptr, &model.synth_cfg, &model.meta_errors);
    if (!model.meta_errors.empty()) {
        fprintf(stderr, "probe error: %s\n", model.meta_errors.front().c_str());
        return 1;
    }
    std::string error;
    if (!mm3_load(&model, &error)) {
        fprintf(stderr, "load error: %s\n", error.c_str());
        return 1;
    }

    if (mode == "condcheck") {
        // Does the DiT's resident condition tensor survive repeated graph
        // computes? Upload cond once (gate 1), interleave an uncond compute
        // (gate 0), then recompute with gate 1 and cond=nullptr: if the
        // scheduler's allocator recycled the input block, run 3 diverges
        // from run 1.
        const int64_t L = 128;
        const int64_t N = (int64_t) model.synth_cfg.dit.in_channels * L;
        const int64_t CN = (int64_t) model.synth_cfg.dit.condition_dim * L;
        std::vector<float> lat((size_t) N), cond((size_t) CN);
        uint64_t state = 1234;
        tts_cpp::minimax::detail::fill_noise(state, 0, lat, N);
        tts_cpp::minimax::detail::fill_noise(state, 1, cond, CN);
        std::vector<float> out1((size_t) N), out_u((size_t) N), out3((size_t) N);
        if (!mm3_dit_prepare(model, &g_mm3_dit, &error) ||
            !mm3_dit_run(model, &g_mm3_dit, lat.data(), cond.data(), 1.0f, 0.5f, L, out1.data(), &error) ||
            !mm3_dit_run(model, &g_mm3_dit, lat.data(), cond.data(), 0.0f, 0.5f, L, out_u.data(), &error) ||
            !mm3_dit_run(model, &g_mm3_dit, lat.data(), cond.data(), 1.0f, 0.5f, L, out3.data(), &error)) {
            fprintf(stderr, "condcheck error: %s\n", error.c_str());
            return 1;
        }
        double dot = 0, n1 = 0, n3 = 0, max_abs = 0;
        for (int64_t i = 0; i < N; ++i) {
            dot += (double) out1[(size_t) i] * out3[(size_t) i];
            n1 += (double) out1[(size_t) i] * out1[(size_t) i];
            n3 += (double) out3[(size_t) i] * out3[(size_t) i];
            max_abs = std::max(max_abs, (double) std::fabs(out1[(size_t) i] - out3[(size_t) i]));
        }
        fprintf(stderr, "[condcheck] cos(out1,out3)=%.9f max_abs_diff=%.6g\n",
                dot / std::sqrt(n1 * n3), max_abs);
        return max_abs < 1e-4 ? 0 : 2;
    }

    MM3GenRequest request;
    request.seed       = (uint64_t) atoll(arg_value(argc, argv, "--seed", "42"));
    request.steps      = atoi(arg_value(argc, argv, "--steps", "30"));
    request.max_frames = atoll(arg_value(argc, argv, "--max-frames", "300"));
    request.cfg_flow   = model.synth_cfg.flow.cfg_scale > 0 ? model.synth_cfg.flow.cfg_scale : 1.7f;
    request.keep_window_latents = true;

    if (mode == "replay") {
        request.ids_cond = read_raw<int32_t>(arg_value(argc, argv, "--tokens", ""));
        request.forced_semantic = read_raw<int32_t>(arg_value(argc, argv, "--semantic", ""));
        request.forced_acoustic = read_raw<int32_t>(arg_value(argc, argv, "--acoustic", ""));
        request.max_frames = (int64_t) request.forced_semantic.size();
        for (int i = 1; i + 1 < argc; ++i) {
            if (strcmp(argv[i], "--noise") == 0) {
                request.forced_noise.push_back(read_raw<float>(argv[i + 1]));
            }
        }
        fprintf(stderr, "[replay] %zu prompt tokens, %zu frames, %zu noise windows\n",
                request.ids_cond.size(), request.forced_semantic.size(), request.forced_noise.size());
    } else {
        const std::string caption = arg_value(argc, argv, "--caption", "");
        const std::string lyrics  = arg_value(argc, argv, "--lyrics", "");
        request.prompt = tts_cpp::minimax::detail::build_prompt(caption, lyrics);
        fprintf(stderr, "[full] prompt: %s\n", request.prompt.c_str());
    }

    MM3Tokenizer tokenizer;
    MM3GenResult result;
    const auto progress = [](const MM3GenProgress & p) {
        fprintf(stderr, "[progress] %s %lld/%lld window %lld/%lld\n", p.stage,
                (long long) p.step, (long long) p.n_steps, (long long) p.window,
                (long long) p.n_windows);
    };
    if (!mm3_generate(model, request, &tokenizer, progress, &result, &error)) {
        fprintf(stderr, "generate error: %s\n", error.c_str());
        return 1;
    }

    write_wav(out_dir + "/audio.wav", result.audio, result.n_samples, result.sample_rate);
    for (size_t w = 0; w < result.window_latents.size(); ++w) {
        write_raw(out_dir + "/window-" + std::to_string(w) + ".f32",
                  result.window_latents[w].data(), result.window_latents[w].size());
    }
    write_raw(out_dir + "/frame-hiddens.f32", result.ar.frame_hiddens.data(),
              result.ar.frame_hiddens.size());
    write_raw(out_dir + "/semantic.i32", result.ar.semantic_all.data(), result.ar.semantic_all.size());
    write_raw(out_dir + "/acoustic.i32", result.ar.acoustic_all.data(), result.ar.acoustic_all.size());

    fprintf(stderr,
            "[done] frames=%lld windows=%lld samples=%lld peak=%.3f rms=%.4f "
            "ar=%.0fms cond=%.0fms flow=%.0fms voc=%.0fms total=%.0fms\n",
            (long long) result.frames, (long long) result.n_windows, (long long) result.n_samples,
            result.peak, result.rms, result.ar_ms, result.cond_ms, result.flow_ms, result.voc_ms,
            result.total_ms);
    return 0;
}
