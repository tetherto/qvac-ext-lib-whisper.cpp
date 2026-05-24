#pragma once

// One-shot transcription of 16 kHz mono audio with Qwen3-ASR-0.6B / 1.7B.
//
// Usage:
//
//     #include <qwen/engine.h>
//     using qwen::Engine;
//     using qwen::EngineOptions;
//
//     EngineOptions opts;
//     opts.model_dir = "models/hf/0.6b";   // dir with model.safetensors + vocab.json + merges.txt
//     opts.n_threads = 8;                   // 0 = autodetect (one per CPU)
//
//     Engine engine(opts);
//     auto result = engine.transcribe("path/to/audio.wav");
//     std::puts(result.text.c_str());
//
// Threading model:
//   - Concurrent transcribe() calls on the same Engine are NOT supported (the
//     decoder reuses internal buffers + KV cache). Wrap in a mutex or hold
//     one Engine per worker.
//   - cancel() is currently a no-op (deferred to v0.2 alongside streaming).
//   - ~Engine() does NOT wait for in-flight calls; destroying an Engine
//     while another thread is inside transcribe() is undefined behaviour.

#include "export.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace qwen {

struct EngineOptions {
    // Directory holding the model files (model.safetensors + vocab.json
    // + merges.txt + config.json + tokenizer_config.json). Get one with:
    //   huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir <dir>
    std::string model_dir;

    // 0 = autodetect (one thread per logical CPU).
    int n_threads = 0;

    // 0 = greedy single-sample decode (the default; matches the upstream
    // qwen-asr inference settings). Reserved for future sampling support.
    int  max_new_tokens     = 448;
    bool greedy_sampling    = true;
    float temperature       = 0.0f;
    float repetition_penalty = 1.0f;

    // Optional language hint forwarded into the decoder prompt. "" or "auto"
    // lets the model auto-detect (Qwen3-ASR's intended default). When set to
    // a recognised language name ("English", "Chinese", "Cantonese", ...)
    // the prompt is biased toward that language.
    std::string language;

    // Optional system prompt for contextual biasing (e.g. "Preserve
    // spelling: CPU, CUDA, PostgreSQL, Redis"). Empty = no prompt.
    std::string system_prompt;

    // Verbosity:  0 = silent, 1 = status messages on stderr, 2 = debug.
    int verbose = 0;
};

struct EngineResult {
    std::string text;

    double encode_ms   = 0.0;
    double decode_ms   = 0.0;
    double total_ms    = 0.0;

    int text_tokens    = 0;
    double audio_ms    = 0.0;
};

using TokenCallback = std::function<void(const std::string & piece)>;

class QWEN_API Engine {
public:
    explicit Engine(const EngineOptions & opts);
    ~Engine();

    Engine(const Engine &)             = delete;
    Engine & operator=(const Engine &) = delete;
    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;

    EngineResult transcribe(const std::string & wav_path);

    EngineResult transcribe_samples(const float * samples,
                                    int n_samples);

    // Optional per-token callback invoked during autoregressive decoding;
    // pieces are concatenated to form the final EngineResult::text.
    void set_token_callback(TokenCallback cb);

    void cancel();

    const EngineOptions & options() const;

    struct Impl;

private:
    std::unique_ptr<Impl> pimpl_;
};

}
