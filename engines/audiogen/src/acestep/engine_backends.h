#pragma once

// Backend resolution shared by Engine::create (engine.cpp) and the memory-fit
// preflight (fit.cpp): one primary backend (GPU when requested and available,
// CPU otherwise), a CPU backend for the stages the placement policy pins off
// the GPU, and the per-stage assignments from stage_placement.h. Extracted
// verbatim from Engine::create so the preflight resolves the same devices a
// real load would by construction -- change it here and both agree.

#include "audiogen-cpp/gpu_fallback.h"

#include "acestep/backend_registry.h"
#include "acestep/stage_placement.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <thread>

namespace tts_cpp::acestep {

struct AcestepBackends {
    ggml_backend_t backend     = nullptr;  // primary (GPU or CPU); owned
    ggml_backend_t backend_cpu = nullptr;  // owned when != backend
    // Per-stage assignments (aliases of the two above, never owned):
    // textenc + cond, the autoregressive LM, and the FSQ detokenizer. The DiT
    // uses `backend` directly and the VAE creates its own backend (Vae::load).
    ggml_backend_t enc   = nullptr;
    ggml_backend_t lm    = nullptr;
    ggml_backend_t detok = nullptr;

    bool              on_gpu              = false;
    GpuFallbackReason gpu_fallback_reason = GpuFallbackReason::not_requested;
    int               nth                 = 4;
};

// Resolve the backends for `n_gpu_layers` / `n_threads` exactly as
// Engine::create does. load_backends(backends_dir) must have run first.
// Returns false when no backend can be initialised at all.
inline bool resolve_acestep_backends(int n_gpu_layers, int n_threads, bool verbose, AcestepBackends & out) {
    out = AcestepBackends{};

    int nth = n_threads > 0 ? n_threads : (int) std::thread::hardware_concurrency();
    if (nth < 1) nth = 4;
    out.nth = nth;

    // Backend for the ggml stages (text-encoder, LM, cond/detok, DiT). These use
    // standard ggml ops, so Metal, CUDA, Vulkan, and validated Adreno 700+
    // OpenCL can run them; n_gpu_layers > 0 opts in. Falls back to CPU when no
    // GPU backend is registered/available.
    bool on_gpu = false;
    if (n_gpu_layers > 0) {
        out.backend = backend_gpu_init(&out.gpu_fallback_reason);
        on_gpu      = (out.backend != nullptr);
        if (!on_gpu && verbose) {
            fprintf(stderr, "[acestep-engine] GPU requested but no GPU backend available (%s); using CPU\n",
                    gpu_fallback_reason_name(out.gpu_fallback_reason));
        }
    }
    if (!out.backend) out.backend = backend_cpu_init();
    if (!out.backend) return false;
    if (on_gpu) {
        if (verbose) fprintf(stderr, "[acestep-engine] DiT/VAE on GPU backend: %s\n", ggml_backend_name(out.backend));
        // Dedicated CPU backend for whichever stages are pinned off the GPU below.
        out.backend_cpu = backend_cpu_init();
        if (!out.backend_cpu) {
            ggml_backend_free(out.backend);
            out.backend = nullptr;
            return false;
        }
        backend_set_n_threads(out.backend_cpu, nth);
    } else {
        backend_set_n_threads(out.backend, nth);
        out.backend_cpu = out.backend;  // single CPU backend serves every stage
    }
    out.on_gpu = on_gpu;

    // Stage placement when a GPU is active (see stage_placement.h): the DiT,
    // the VAE and the one-shot text/cond encoders always run on it; the LM and
    // the FSQ detokenizer are allowlisted per backend, with the ACESTEP_*
    // environment escape hatches applied after the allowlist.
    ggml_backend_t enc_backend   = out.backend;
    ggml_backend_t lm_backend    = out.backend;
    ggml_backend_t detok_backend = out.backend;
    if (on_gpu) {
        const StagePlacement place =
            resolve_stage_placement(backend_reg_name(out.backend), backend_dev_description(out.backend),
                                    placement_overrides_from_env());
        if (!place.enc_on_gpu)   enc_backend   = out.backend_cpu;
        if (!place.lm_on_gpu)    lm_backend    = out.backend_cpu;
        if (!place.detok_on_gpu) detok_backend = out.backend_cpu;
        if (verbose) fprintf(stderr, "[acestep-engine] backends: enc=%s lm=%s detok=%s dit/vae=%s\n",
                             ggml_backend_name(enc_backend), ggml_backend_name(lm_backend),
                             ggml_backend_name(detok_backend), ggml_backend_name(out.backend));
    }
    out.enc   = enc_backend;
    out.lm    = lm_backend;
    out.detok = detok_backend;
    return true;
}

// Free the owned backends (safe on a default-constructed struct).
inline void free_acestep_backends(AcestepBackends & b) {
    if (b.backend_cpu && b.backend_cpu != b.backend) ggml_backend_free(b.backend_cpu);
    if (b.backend) ggml_backend_free(b.backend);
    b = AcestepBackends{};
}

// ── VAE backend ──────────────────────────────────────────────────────────────
// The VAE acquires its own backend (Vae::load) rather than borrowing the
// primary one; the engine feeds it EngineOptions::n_gpu_layers with the
// ACESTEP_VAE_GPU diagnostic override. Both pieces live here so Vae::load, the
// engine, and the memory-fit projection resolve the same device by
// construction -- a future change lands in all three at once.

// ACESTEP_VAE_GPU forces the VAE backend independently of the other stages so
// a decode can be compared CPU-vs-GPU on an identical latent (=1 -> GPU,
// =0 -> CPU); leaves the LM/DiT backend untouched.
inline int vae_gpu_layers_from_env(int n_gpu_layers) {
    if (const char * e = std::getenv("ACESTEP_VAE_GPU")) {
        return (e[0] == '1') ? 99 : 0;
    }
    return n_gpu_layers;
}

// GPU when requested and available (the two custom VAE ops have Metal/Vulkan/
// validated-OpenCL kernels in the ggml-speech fork), CPU fallback otherwise,
// with the CPU thread count applied. Returns null only when no CPU backend can
// be initialised; the returned backend is owned by the caller.
inline ggml_backend_t resolve_vae_backend(int n_gpu_layers, int n_threads, bool verbose) {
    ggml_backend_t backend = nullptr;
    if (n_gpu_layers > 0) {
        backend = backend_gpu_init();
        if (!backend && verbose) {
            fprintf(stderr, "[acestep-vae] GPU requested but no GPU backend available; using CPU\n");
        }
    }
    if (!backend) {
        backend = backend_cpu_init();
        if (!backend) return nullptr;
        int nthreads = n_threads > 0 ? n_threads : (int) std::thread::hardware_concurrency();
        if (nthreads <= 0) nthreads = 4;
        backend_set_n_threads(backend, nthreads);
    }
    return backend;
}

}  // namespace tts_cpp::acestep
