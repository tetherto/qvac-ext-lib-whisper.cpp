#include "backend_selection.h"

#include "test_env_portable.h"

#include <cstdio>
#include <stdexcept>

namespace {

constexpr const char * VULKAN_BACKEND = "Vulkan";
constexpr const char * CUDA_BACKEND = "CUDA";

int failures = 0;

void check(bool condition, const char * message) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

// A GPU arm names its backend here, so an unusable value has to be rejected
// rather than skipping every device and leaving the arm to pass on the CPU.
bool selection_rejects(const char * forced) {
    if (forced) {
        setenv("TTS_CPP_GPU_BACKEND", forced, 1);
    } else {
        unsetenv("TTS_CPP_GPU_BACKEND");
    }
    try {
        tts_cpp::detail::init_gpu_backend(/*n_gpu_layers=*/1, /*verbose=*/false, "test");
    } catch (const std::exception &) {
        unsetenv("TTS_CPP_GPU_BACKEND");
        return true;
    }
    unsetenv("TTS_CPP_GPU_BACKEND");
    return false;
}

}

int main() {
    using tts_cpp::detail::GpuBackendRequirement;
    using tts_cpp::detail::gpu_backend_satisfies_requirement;

    check(gpu_backend_satisfies_requirement(nullptr, GpuBackendRequirement::Any),
          "unrestricted selection must accept an unnamed backend");
    check(gpu_backend_satisfies_requirement(CUDA_BACKEND, GpuBackendRequirement::Any),
          "unrestricted selection must accept CUDA");
    check(gpu_backend_satisfies_requirement(VULKAN_BACKEND,
                                            GpuBackendRequirement::Vulkan),
          "Vulkan selection must accept Vulkan");
    check(!gpu_backend_satisfies_requirement(CUDA_BACKEND,
                                             GpuBackendRequirement::Vulkan),
          "Vulkan selection must reject CUDA");
    check(!gpu_backend_satisfies_requirement(nullptr, GpuBackendRequirement::Vulkan),
          "Vulkan selection must reject an unnamed backend");
    const auto vkmtlcl = GpuBackendRequirement::Metal | GpuBackendRequirement::OpenCL |
                         GpuBackendRequirement::Vulkan;
    check(gpu_backend_satisfies_requirement(VULKAN_BACKEND, vkmtlcl),
          "Metal|OpenCL|Vulkan selection must accept Vulkan");
    check(!gpu_backend_satisfies_requirement(CUDA_BACKEND, vkmtlcl),
          "Metal|OpenCL|Vulkan selection must reject CUDA");

    check(selection_rejects("bogus"),
          "an unknown TTS_CPP_GPU_BACKEND must be rejected, not silently ignored");
    check(!selection_rejects("cuda"),
          "a known TTS_CPP_GPU_BACKEND must be accepted whether or not that device exists");
    check(!selection_rejects(nullptr),
          "an unset TTS_CPP_GPU_BACKEND must leave selection alone");
    return failures == 0 ? 0 : 1;
}
