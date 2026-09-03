#include "audiogen-cpp/acestep/vae.h"

#include "vae_ggml.h"              // internal: VaeModel + vae_model_*
#include "vae_encode_windows.h"

#include "acestep/backend_registry.h"
#include "acestep/engine_backends.h"

#include "ggml-backend.h"

#include <cstdio>
#include <stdexcept>

namespace tts_cpp::acestep {

struct Vae::Impl {
    ggml_backend_t backend = nullptr;  // owned
    VaeModel *     model   = nullptr;
    std::string    backend_name = "CPU";

    ~Impl() {
        if (model) vae_model_free(model);
        if (backend) ggml_backend_free(backend);
    }
};

Vae::Vae() : impl_(std::make_unique<Impl>()) {}
Vae::~Vae() = default;

std::unique_ptr<Vae> Vae::load(const std::string & gguf_path, const VaeOptions & opts) {
    std::unique_ptr<Vae> v(new Vae());

    // The two custom ops (col2im_1d, snake) have CPU, Metal and Vulkan kernels
    // in the ggml-speech fork, so the decode/encode graph can run on a GPU
    // backend. n_gpu_layers > 0 opts in (Metal on Apple, Vulkan elsewhere);
    // falls back to CPU when no GPU backend is registered/available. Backends are
    // acquired through the ggml registry (see backend_registry.h) so the CPU path
    // resolves on arm64 dlopen (GGML_CPU_ALL_VARIANTS) builds too. When Vae::load
    // runs standalone (not via Engine), backends_dir loads the modules first.
    load_backends(opts.backends_dir);

    // Resolution shared with the engine and the memory-fit projection
    // (engine_backends.h), so all three agree by construction.
    ggml_backend_t backend = resolve_vae_backend(opts.n_gpu_layers, opts.n_threads, opts.verbose);
    if (!backend) throw std::runtime_error("acestep-vae: failed to init CPU backend");

    VaeModel * model = vae_model_load(gguf_path, backend, opts.with_encoder, opts.verbose);
    if (!model) {
        ggml_backend_free(backend);
        throw std::runtime_error("acestep-vae: failed to load VAE GGUF: " + gguf_path);
    }

    v->impl_->backend = backend;
    v->impl_->model   = model;
    const char * bn   = ggml_backend_name(backend);
    v->impl_->backend_name = bn ? bn : "CPU";
    return v;
}

std::vector<float> Vae::decode(const std::vector<float> & latent, int T_latent,
                               const ProgressCb & on_progress) const {
    std::vector<float> pcm;
    if (T_latent <= 0 || (int) latent.size() < T_latent * 64) return {};
    int T_audio = vae_model_decode(impl_->model, latent.data(), T_latent, pcm, on_progress);
    if (T_audio < 0) return {};
    return pcm;
}

std::vector<float> Vae::encode(const std::vector<float> & pcm_interleaved, int frames,
                               int * T_latent_out,
                               const ProgressCb & on_progress) const {
    const int window_count = vae_encode_window_count(frames);
    int window_index = 0;
    const VaeWindowEncoder encode = [this, &on_progress, &window_index, window_count](
                                        const float * pcm, int window_frames,
                                        std::vector<float> & latent) {
        ProgressCb window_progress;
        if (on_progress) {
            window_progress = [&on_progress, &window_index, window_count](
                                  int done, int total) {
                return on_progress(window_index * total + done,
                                   window_count * total);
            };
        }
        const int encoded = vae_model_encode(
            impl_->model, pcm, window_frames, latent, window_progress);
        ++window_index;
        return encoded;
    };
    return encode_vae_pcm_bounded(pcm_interleaved, frames, encode, T_latent_out);
}

bool        Vae::has_encoder() const   { return vae_model_has_encoder(impl_->model); }
int         Vae::sample_rate() const   { return 48000; }
int         Vae::upsample_factor() const { return 1920; }
std::string Vae::backend_name() const  { return impl_->backend_name; }

} // namespace tts_cpp::acestep
