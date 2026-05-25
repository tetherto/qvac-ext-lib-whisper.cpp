#include "qwen/cli.h"
#include "qwen/engine.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

const char * kQwenVersion =
#ifdef QWEN_VERSION
    QWEN_VERSION
#else
    "0.0.0"
#endif
;

qwen::Backend parse_backend(const char * s) {
    if (std::strcmp(s, "gguf") == 0)        return qwen::Backend::GGUF;
    if (std::strcmp(s, "safetensors") == 0) return qwen::Backend::Safetensors;
    throw std::runtime_error(std::string("unknown backend '") + s + "' (expected: safetensors | gguf)");
}

void print_help() {
    std::printf(
        "qwen-asr %s -- Qwen3-ASR speech-to-text\n"
        "\n"
        "Usage:\n"
        "  qwen-asr --version\n"
        "  qwen-asr --help\n"
        "  qwen-asr transcribe --model <path> --wav <path.wav> [options]\n"
        "\n"
        "transcribe options:\n"
        "  --backend <name>       Inference backend: safetensors (default) | gguf (v0.2)\n"
        "  --model <path>         Path to the model: directory with model.safetensors\n"
        "                         (safetensors backend) or .gguf file (gguf backend)\n"
        "  --threads <n>          Number of inference threads (default: autodetect)\n"
        "  --language <name>      Force output language (default: auto-detect)\n"
        "                         Examples: English, Chinese, Cantonese, Spanish, ...\n"
        "  --prompt <text>        System prompt for contextual biasing\n"
        "                         (e.g. \"Preserve spelling: CPU, CUDA, PostgreSQL\")\n"
        "  --verbose              Print status messages on stderr\n"
        "  --debug                Print per-layer debug information\n"
        "\n"
        "Safetensors backend (v0.1, default) expects a HuggingFace checkpoint dir with:\n"
        "  model.safetensors  config.json  vocab.json  merges.txt\n"
        "  tokenizer_config.json  preprocessor_config.json\n"
        "\n"
        "Get one with:\n"
        "  huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ./Qwen3-ASR-0.6B\n",
        kQwenVersion);
}

int run_transcribe(int argc, char ** argv) {
    qwen::EngineOptions opts;
    std::string wav;

    for (int i = 0; i < argc; ++i) {
        const char * a = argv[i];
        if ((std::strcmp(a, "--model") == 0 || std::strcmp(a, "--model-dir") == 0) && i + 1 < argc) {
            opts.model_path = argv[++i];
        } else if (std::strcmp(a, "--backend") == 0 && i + 1 < argc) {
            try {
                opts.backend = parse_backend(argv[++i]);
            } catch (const std::exception & e) {
                std::fprintf(stderr, "qwen-asr transcribe: %s\n", e.what());
                return 2;
            }
        } else if (std::strcmp(a, "--wav") == 0 && i + 1 < argc) {
            wav = argv[++i];
        } else if (std::strcmp(a, "--threads") == 0 && i + 1 < argc) {
            opts.n_threads = std::atoi(argv[++i]);
        } else if (std::strcmp(a, "--language") == 0 && i + 1 < argc) {
            opts.language = argv[++i];
        } else if (std::strcmp(a, "--prompt") == 0 && i + 1 < argc) {
            opts.system_prompt = argv[++i];
        } else if (std::strcmp(a, "--verbose") == 0) {
            opts.verbose = 1;
        } else if (std::strcmp(a, "--debug") == 0) {
            opts.verbose = 2;
        } else if (std::strcmp(a, "--max-new-tokens") == 0 && i + 1 < argc) {
            opts.max_new_tokens = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "qwen-asr transcribe: unknown arg '%s'\n", a);
            return 2;
        }
    }
    if (opts.model_path.empty() || wav.empty()) {
        std::fprintf(stderr, "qwen-asr transcribe: --model and --wav are required\n");
        return 2;
    }

    try {
        qwen::Engine engine(opts);
        auto result = engine.transcribe(wav);
        std::puts(result.text.c_str());
        if (opts.verbose > 0) {
            std::fprintf(stderr,
                "Inference: %.0f ms, %d text tokens (encoder %.0f ms, decoder %.0f ms)\n",
                result.total_ms, result.text_tokens, result.encode_ms, result.decode_ms);
            if (result.audio_ms > 0.0 && result.total_ms > 0.0) {
                double audio_s    = result.audio_ms / 1000.0;
                double infer_s    = result.total_ms / 1000.0;
                double realtime_x = audio_s / infer_s;
                std::fprintf(stderr, "Audio: %.1f s processed in %.1f s (%.2fx realtime)\n",
                    audio_s, infer_s, realtime_x);
            }
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "qwen-asr: %s\n", e.what());
        return 1;
    }
}

}

extern "C" int qwen_cli_main(int argc, char ** argv) {
    if (argc < 2) {
        print_help();
        return 0;
    }
    const char * cmd = argv[1];
    if (std::strcmp(cmd, "--version") == 0 || std::strcmp(cmd, "-v") == 0) {
        std::printf("qwen-asr %s\n", kQwenVersion);
        return 0;
    }
    if (std::strcmp(cmd, "--help") == 0 || std::strcmp(cmd, "-h") == 0) {
        print_help();
        return 0;
    }
    if (std::strcmp(cmd, "transcribe") == 0) {
        return run_transcribe(argc - 2, argv + 2);
    }
    std::fprintf(stderr, "qwen-asr: unknown command '%s' (try --help)\n", cmd);
    return 2;
}
