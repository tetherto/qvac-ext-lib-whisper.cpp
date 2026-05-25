#include "qwen/engine.h"

#include <utility>

namespace qwen {

Engine::Engine(const EngineOptions & opts) : impl_(create_engine(opts)) {}

Engine::~Engine() = default;

Engine::Engine(Engine &&) noexcept             = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

EngineResult Engine::transcribe(const std::string & wav_path) {
    return impl_->transcribe(wav_path);
}

EngineResult Engine::transcribe_samples(const float * samples, int n_samples) {
    return impl_->transcribe_samples(samples, n_samples);
}

void Engine::set_token_callback(TokenCallback cb) {
    impl_->set_token_callback(std::move(cb));
}

void Engine::cancel() {
    impl_->cancel();
}

const EngineOptions & Engine::options() const {
    return impl_->options();
}

Backend Engine::backend() const {
    return impl_->backend();
}

IEngine * Engine::native() noexcept {
    return impl_.get();
}

}
