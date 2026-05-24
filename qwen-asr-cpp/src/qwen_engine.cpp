#include "qwen/engine.h"

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

namespace qwen {

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

}

struct Engine::Impl {
    EngineOptions opts;
    qwen_ctx_t *  ctx = nullptr;
    TokenCallback token_cb;
    std::atomic_bool cancelled{false};
};

namespace {

extern "C" void token_cb_trampoline(const char * piece, void * userdata) {
    auto * impl = static_cast<Engine::Impl *>(userdata);
    if (impl && impl->token_cb && piece) {
        impl->token_cb(std::string(piece));
    }
}

}

Engine::Engine(const EngineOptions & opts)
    : pimpl_(std::make_unique<Impl>()) {
    pimpl_->opts = opts;
    if (opts.model_dir.empty()) {
        throw std::runtime_error("qwen::Engine: EngineOptions.model_dir is required");
    }

    apply_verbosity(opts.verbose);
    apply_threads(opts.n_threads);

    pimpl_->ctx = qwen_load(opts.model_dir.c_str());
    if (!pimpl_->ctx) {
        throw std::runtime_error("qwen::Engine: failed to load model from '" + opts.model_dir + "'");
    }

    if (!opts.system_prompt.empty()) {
        if (qwen_set_prompt(pimpl_->ctx, opts.system_prompt.c_str()) != 0) {
            qwen_free(pimpl_->ctx);
            pimpl_->ctx = nullptr;
            throw std::runtime_error("qwen::Engine: failed to apply system prompt");
        }
    }
    if (!opts.language.empty() && opts.language != "auto") {
        if (qwen_set_force_language(pimpl_->ctx, opts.language.c_str()) != 0) {
            const std::string supported = qwen_supported_languages_csv();
            qwen_free(pimpl_->ctx);
            pimpl_->ctx = nullptr;
            throw std::runtime_error(
                "qwen::Engine: unsupported language '" + opts.language +
                "'. Supported: " + supported);
        }
    }
}

Engine::~Engine() {
    if (pimpl_ && pimpl_->ctx) {
        qwen_free(pimpl_->ctx);
        pimpl_->ctx = nullptr;
    }
}

Engine::Engine(Engine &&) noexcept             = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

EngineResult Engine::transcribe(const std::string & wav_path) {
    pimpl_->cancelled.store(false, std::memory_order_relaxed);
    if (pimpl_->token_cb) {
        qwen_set_token_callback(pimpl_->ctx, token_cb_trampoline, pimpl_.get());
    } else {
        qwen_set_token_callback(pimpl_->ctx, nullptr, nullptr);
    }

    char * raw = qwen_transcribe(pimpl_->ctx, wav_path.c_str());
    if (!raw) {
        throw std::runtime_error("qwen::Engine::transcribe: transcription failed for '" + wav_path + "'");
    }
    std::string text(raw);
    free(raw);
    return collect_perf(pimpl_->ctx, std::move(text));
}

EngineResult Engine::transcribe_samples(const float * samples, int n_samples) {
    if (!samples || n_samples <= 0) {
        throw std::runtime_error("qwen::Engine::transcribe_samples: empty input");
    }
    pimpl_->cancelled.store(false, std::memory_order_relaxed);
    if (pimpl_->token_cb) {
        qwen_set_token_callback(pimpl_->ctx, token_cb_trampoline, pimpl_.get());
    } else {
        qwen_set_token_callback(pimpl_->ctx, nullptr, nullptr);
    }

    char * raw = qwen_transcribe_audio(pimpl_->ctx, samples, n_samples);
    if (!raw) {
        throw std::runtime_error("qwen::Engine::transcribe_samples: transcription failed");
    }
    std::string text(raw);
    free(raw);
    return collect_perf(pimpl_->ctx, std::move(text));
}

void Engine::set_token_callback(TokenCallback cb) {
    pimpl_->token_cb = std::move(cb);
}

void Engine::cancel() {
    pimpl_->cancelled.store(true, std::memory_order_relaxed);
}

const EngineOptions & Engine::options() const {
    return pimpl_->opts;
}

}
