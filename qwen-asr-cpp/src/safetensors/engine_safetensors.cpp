#include "engine_safetensors.h"

extern "C" {
#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
}

#include <atomic>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace qwen::safetensors {

namespace {

void apply_threads(int requested) {
    int n = requested > 0 ? requested : qwen_get_num_cpus();
    qwen_set_threads(n);
}

void apply_verbosity(int level) {
    qwen_verbose = level;
}

EngineResult collect_perf(qwen_ctx_t * ctx, std::string text) {
    EngineResult r;
    r.text         = std::move(text);
    r.encode_ms    = ctx->perf_encode_ms;
    r.decode_ms    = ctx->perf_decode_ms;
    r.total_ms     = ctx->perf_total_ms;
    r.text_tokens  = ctx->perf_text_tokens;
    r.audio_ms     = ctx->perf_audio_ms;
    return r;
}

class Engine final : public IEngine {
public:
    explicit Engine(const EngineOptions & opts);
    ~Engine() override;

    Engine(const Engine &)             = delete;
    Engine & operator=(const Engine &) = delete;

    EngineResult transcribe(const std::string & wav_path) override;

    EngineResult transcribe_samples(const float * samples, int n_samples) override;

    void set_token_callback(TokenCallback cb) override;

    void cancel() override;

    const EngineOptions & options() const override { return opts_; }

    Backend backend() const override { return Backend::Safetensors; }

    void on_token(const std::string & piece);

private:
    void apply_prompt_and_language();
    void prime_token_callback();

    EngineOptions    opts_;
    qwen_ctx_t *     ctx_ = nullptr;
    TokenCallback    token_cb_;
    std::atomic_bool cancelled_{false};
};

extern "C" void token_cb_trampoline(const char * piece, void * userdata) {
    if (!userdata || !piece) {
        return;
    }
    auto * engine = static_cast<Engine *>(userdata);
    engine->on_token(piece);
}

Engine::Engine(const EngineOptions & opts) : opts_(opts) {
    if (opts_.model_path.empty()) {
        throw std::runtime_error(
            "qwen::safetensors::Engine: EngineOptions.model_path is required (point it to "
            "a HuggingFace checkpoint directory with model.safetensors)");
    }

    apply_verbosity(opts_.verbose);
    apply_threads(opts_.n_threads);

    ctx_ = qwen_load(opts_.model_path.c_str());
    if (!ctx_) {
        throw std::runtime_error(
            "qwen::safetensors::Engine: failed to load model from '" + opts_.model_path + "'");
    }

    apply_prompt_and_language();
}

Engine::~Engine() {
    if (ctx_) {
        qwen_free(ctx_);
        ctx_ = nullptr;
    }
}

void Engine::apply_prompt_and_language() {
    if (!opts_.system_prompt.empty()) {
        if (qwen_set_prompt(ctx_, opts_.system_prompt.c_str()) != 0) {
            qwen_free(ctx_);
            ctx_ = nullptr;
            throw std::runtime_error("qwen::safetensors::Engine: failed to apply system prompt");
        }
    }
    if (!opts_.language.empty() && opts_.language != "auto") {
        if (qwen_set_force_language(ctx_, opts_.language.c_str()) != 0) {
            const std::string supported = qwen_supported_languages_csv();
            qwen_free(ctx_);
            ctx_ = nullptr;
            throw std::runtime_error(
                "qwen::safetensors::Engine: unsupported language '" + opts_.language +
                "'. Supported: " + supported);
        }
    }
}

void Engine::prime_token_callback() {
    if (token_cb_) {
        qwen_set_token_callback(ctx_, token_cb_trampoline, this);
    } else {
        qwen_set_token_callback(ctx_, nullptr, nullptr);
    }
}

void Engine::on_token(const std::string & piece) {
    if (token_cb_) {
        token_cb_(piece);
    }
}

EngineResult Engine::transcribe(const std::string & wav_path) {
    cancelled_.store(false, std::memory_order_relaxed);
    prime_token_callback();

    char * raw = qwen_transcribe(ctx_, wav_path.c_str());
    if (!raw) {
        throw std::runtime_error(
            "qwen::safetensors::Engine::transcribe: transcription failed for '" + wav_path + "'");
    }
    std::string text(raw);
    free(raw);
    return collect_perf(ctx_, std::move(text));
}

EngineResult Engine::transcribe_samples(const float * samples, int n_samples) {
    if (!samples || n_samples <= 0) {
        throw std::runtime_error("qwen::safetensors::Engine::transcribe_samples: empty input");
    }
    cancelled_.store(false, std::memory_order_relaxed);
    prime_token_callback();

    char * raw = qwen_transcribe_audio(ctx_, samples, n_samples);
    if (!raw) {
        throw std::runtime_error("qwen::safetensors::Engine::transcribe_samples: transcription failed");
    }
    std::string text(raw);
    free(raw);
    return collect_perf(ctx_, std::move(text));
}

void Engine::set_token_callback(TokenCallback cb) {
    token_cb_ = std::move(cb);
}

void Engine::cancel() {
    cancelled_.store(true, std::memory_order_relaxed);
}

}

std::unique_ptr<IEngine> make_engine(const EngineOptions & opts) {
    return std::unique_ptr<IEngine>(new Engine(opts));
}

}
