#pragma once

// GPU-picker helper for callers of the vendored whisper.cpp
// (third_party/whisper.cpp). Returns the value the caller should pass as
// `whisper_context_params.gpu_device`.
//
// WHY THIS EXISTS
// ---------------
// whisper's own `whisper_backend_init_gpu` walks the ggml-backend registry
// linearly and picks the Nth GPU-or-IGPU device (where N is
// `params.gpu_device`, default 0). That default silently loses on two setups
// speech callers care about:
//
//   1. NVIDIA host with both CUDA and Vulkan compiled in — ggml usually
//      enumerates Vulkan first, so `gpu_device=0` picks the Vulkan adapter
//      on the same NVIDIA card. CUDA is vendor-native and measurably faster;
//      that is what llm-llamacpp / diffusion / tts / parakeet / audiogen all
//      prefer, so whisper callers should too.
//   2. Hybrid dGPU + iGPU host (e.g. NVIDIA discrete + Intel integrated) —
//      ggml enumeration order is not stable across driver installs, and
//      `gpu_device=0` may land on the iGPU. All other QVAC speech engines
//      prefer discrete over integrated.
//
// The vendored whisper subtree is gated by third_party/whisper.cpp/PATCHES.md
// against direct edits, so the selection lives here on the caller side.
// Consumers in the qvac monorepo (packages/asr-ggml) wire the return value
// into whisper_context_params.gpu_device before whisper_init_from_file_*.
//
// USAGE
// -----
//     #include "whisper-gpu-picker.h"
//     whisper_context_params p = whisper_context_default_params();
//     p.use_gpu    = true;
//     p.gpu_device = qvac::speech::pick_whisper_gpu_device();
//     ctx = whisper_init_from_file_with_params(path, p);
//
// The helper never calls into whisper.cpp itself — it only reads the shared
// ggml-backend registry, so it links cleanly under both GGML_BACKEND_DL=ON
// (dlopened per-arch backends) and OFF (static-linked) builds.

#include "ggml-backend.h"

#include <cstring>

namespace qvac::speech {

// Return the index-among-GPU-devices — using whisper.cpp's exact enumeration
// order (GPU + IGPU, in ggml registry order) — that the QVAC speech tier
// policy prefers (CUDA-first on NVIDIA, discrete-first over integrated).
//
//   1. First CUDA device (dGPU tier before iGPU tier — CUDA-on-tegra shows up
//      as IGPU on some drivers, so we still fall through the iGPU tier if
//      no discrete CUDA card is visible).
//   2. Otherwise, first discrete (GPU) non-CUDA device — Vulkan/Metal on a
//      dGPU, etc.
//   3. Otherwise, first integrated (IGPU) device — UMA Vulkan, Mali iGPU,
//      Intel iGPU, Apple integrated, etc.
//   4. Otherwise 0 (whisper's own default; whisper_backend_init_gpu will
//      report "no GPU found" and return nullptr, and the caller falls back
//      to CPU as before).
//
// Returns 0 when the registry has not yet been loaded (no
// `ggml_backend_load_all*` call has happened): with an empty device list
// there is nothing to prefer, and the default preserves whisper's own
// no-GPU-found path.
inline int pick_whisper_gpu_device() {
    const size_t n_dev = ggml_backend_dev_count();

    // Whisper counts GPU-or-IGPU devices linearly; we mirror that counting
    // so the index we return means the same thing on the whisper side.
    // Track the first hit in each tier as we walk.
    int idx_cuda_discrete   = -1;
    int idx_cuda_integrated = -1;
    int idx_discrete        = -1;
    int idx_integrated      = -1;

    int cnt = 0;
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        ggml_backend_reg_t reg      = ggml_backend_dev_backend_reg(dev);
        const char *       reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
        const bool         is_cuda  = reg_name && std::strcmp(reg_name, "CUDA") == 0;
        const bool         is_integrated = (type == GGML_BACKEND_DEVICE_TYPE_IGPU);

        if (is_cuda && !is_integrated && idx_cuda_discrete   < 0) idx_cuda_discrete   = cnt;
        if (is_cuda &&  is_integrated && idx_cuda_integrated < 0) idx_cuda_integrated = cnt;
        if (!is_cuda && !is_integrated && idx_discrete       < 0) idx_discrete        = cnt;
        if (!is_cuda &&  is_integrated && idx_integrated     < 0) idx_integrated      = cnt;

        ++cnt;
    }

    if (idx_cuda_discrete   >= 0) return idx_cuda_discrete;
    if (idx_cuda_integrated >= 0) return idx_cuda_integrated;
    if (idx_discrete        >= 0) return idx_discrete;
    if (idx_integrated      >= 0) return idx_integrated;
    return 0;
}

// Testable form of the tier policy: takes an explicit device list (registry
// name, is-integrated) and returns the index the policy prefers. Kept
// alongside `pick_whisper_gpu_device` so both call the same logic; the
// process-scoped version above is a thin wrapper that reads the ggml
// registry and delegates here. Exposed for unit tests that simulate a
// fake device topology without depending on the host's real GPU set.
struct PickerDevice {
    const char * reg_name;      // registry name, e.g. "CUDA" / "Vulkan" / "Metal" / "OpenCL"
    bool         is_integrated; // true for GGML_BACKEND_DEVICE_TYPE_IGPU
};

inline int pick_whisper_gpu_device_from(const PickerDevice * devs, size_t n) {
    int idx_cuda_discrete   = -1;
    int idx_cuda_integrated = -1;
    int idx_discrete        = -1;
    int idx_integrated      = -1;

    for (size_t i = 0; i < n; ++i) {
        const PickerDevice & d = devs[i];
        const bool is_cuda = d.reg_name && std::strcmp(d.reg_name, "CUDA") == 0;
        const int  idx     = static_cast<int>(i);
        if (is_cuda && !d.is_integrated && idx_cuda_discrete   < 0) idx_cuda_discrete   = idx;
        if (is_cuda &&  d.is_integrated && idx_cuda_integrated < 0) idx_cuda_integrated = idx;
        if (!is_cuda && !d.is_integrated && idx_discrete       < 0) idx_discrete        = idx;
        if (!is_cuda &&  d.is_integrated && idx_integrated     < 0) idx_integrated      = idx;
    }

    if (idx_cuda_discrete   >= 0) return idx_cuda_discrete;
    if (idx_cuda_integrated >= 0) return idx_cuda_integrated;
    if (idx_discrete        >= 0) return idx_discrete;
    if (idx_integrated      >= 0) return idx_integrated;
    return 0;
}

}  // namespace qvac::speech
