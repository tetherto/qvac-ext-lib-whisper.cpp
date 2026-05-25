#pragma once

#include "backend.h"
#include "export.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace qwen {

struct EngineOptions {
    Backend backend = Backend::Safetensors;

    std::string model_path;

    int n_threads = 0;

    int  max_new_tokens     = 448;
    bool greedy_sampling    = true;
    float temperature       = 0.0f;
    float repetition_penalty = 1.0f;

    std::string language;

    std::string system_prompt;

    int verbose = 0;
};

struct EngineResult {
    std::string text;

    double encode_ms   = 0.0;
    double decode_ms   = 0.0;
    double total_ms    = 0.0;

    int    text_tokens = 0;
    double audio_ms    = 0.0;
};

using TokenCallback = std::function<void(const std::string & piece)>;

class QWEN_API IEngine {
public:
    virtual ~IEngine() = default;

    virtual EngineResult transcribe(const std::string & wav_path) = 0;

    virtual EngineResult transcribe_samples(const float * samples,
                                            int n_samples) = 0;

    virtual void set_token_callback(TokenCallback cb) = 0;

    virtual void cancel() = 0;

    virtual const EngineOptions & options() const = 0;

    virtual Backend backend() const = 0;
};

QWEN_API std::unique_ptr<IEngine> create_engine(const EngineOptions & opts);

class QWEN_API Engine {
public:
    explicit Engine(const EngineOptions & opts);
    ~Engine();

    Engine(const Engine &)             = delete;
    Engine & operator=(const Engine &) = delete;
    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;

    EngineResult transcribe(const std::string & wav_path);

    EngineResult transcribe_samples(const float * samples, int n_samples);

    void set_token_callback(TokenCallback cb);

    void cancel();

    const EngineOptions & options() const;

    Backend backend() const;

    IEngine * native() noexcept;

private:
    std::unique_ptr<IEngine> impl_;
};

}
