#include "backend_selection.h"

#include <cstdio>

namespace {

constexpr const char * VULKAN_BACKEND = "Vulkan";
constexpr const char * CUDA_BACKEND = "CUDA";

int failures = 0;

void check(bool condition, const char * message) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
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
    check(gpu_backend_satisfies_requirement(VULKAN_BACKEND,
                                            GpuBackendRequirement::MetalOrOpenCLOrVulkan),
          "MetalOrOpenCLOrVulkan selection must accept Vulkan");
    check(!gpu_backend_satisfies_requirement(CUDA_BACKEND,
                                             GpuBackendRequirement::MetalOrOpenCLOrVulkan),
          "MetalOrOpenCLOrVulkan selection must reject CUDA");
    return failures == 0 ? 0 : 1;
}
