// Policy test for GpuBackendRequirement filtering in the backend-selection
// walk. Runs devicelessly: gpu_backend_satisfies_requirement() is the exact
// predicate init_gpu_backend() applies per registered device, so pinning it
// pins which devices each requirement may consider. The multi-backend case
// this guards: with MetalOrOpenCL, a Vulkan device that sorts first must be
// skipped at the walk (leaving Metal selectable) instead of winning selection
// and being rejected afterwards, which would drop a Metal-capable host to CPU.

#include "backend_selection.h"

#include <cstdio>

using tts_cpp::detail::GpuBackendRequirement;
using tts_cpp::detail::gpu_backend_satisfies_requirement;

namespace {

int g_failures = 0;

void expect(bool cond, const char * what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

} // namespace

int main() {
    const auto any      = GpuBackendRequirement::Any;
    const auto vulkan   = GpuBackendRequirement::Vulkan;
    const auto vkmtl    = GpuBackendRequirement::VulkanOrMetal;
    const auto mtlcl    = GpuBackendRequirement::MetalOrOpenCL;
    const auto mtlclvk  = GpuBackendRequirement::MetalOrOpenCLOrVulkan;

    expect(gpu_backend_satisfies_requirement("Metal", any), "Any accepts Metal");
    expect(gpu_backend_satisfies_requirement("Vulkan", any), "Any accepts Vulkan");
    expect(gpu_backend_satisfies_requirement("CUDA", any), "Any accepts CUDA");
    expect(gpu_backend_satisfies_requirement(nullptr, any), "Any accepts unnamed");

    expect(gpu_backend_satisfies_requirement("Vulkan", vulkan), "Vulkan accepts Vulkan");
    expect(!gpu_backend_satisfies_requirement("Metal", vulkan), "Vulkan rejects Metal");
    expect(!gpu_backend_satisfies_requirement(nullptr, vulkan), "Vulkan rejects unnamed");

    expect(gpu_backend_satisfies_requirement("Vulkan", vkmtl), "VulkanOrMetal accepts Vulkan");
    expect(gpu_backend_satisfies_requirement("MTL", vkmtl), "VulkanOrMetal accepts MTL");
    expect(gpu_backend_satisfies_requirement("Metal", vkmtl), "VulkanOrMetal accepts Metal under its former name");
    expect(!gpu_backend_satisfies_requirement("OpenCL", vkmtl), "VulkanOrMetal rejects OpenCL");
    expect(!gpu_backend_satisfies_requirement("CUDA", vkmtl), "VulkanOrMetal rejects CUDA");
    expect(!gpu_backend_satisfies_requirement(nullptr, vkmtl), "VulkanOrMetal rejects unnamed");

    expect(gpu_backend_satisfies_requirement("Metal", mtlcl), "MetalOrOpenCL accepts Metal");
    expect(gpu_backend_satisfies_requirement("MTL", mtlcl), "MetalOrOpenCL accepts MTL");
    expect(gpu_backend_satisfies_requirement("OpenCL", mtlcl), "MetalOrOpenCL accepts OpenCL");
    expect(!gpu_backend_satisfies_requirement("Vulkan", mtlcl), "MetalOrOpenCL rejects Vulkan");
    expect(!gpu_backend_satisfies_requirement("CUDA", mtlcl), "MetalOrOpenCL rejects CUDA");
    expect(!gpu_backend_satisfies_requirement(nullptr, mtlcl), "MetalOrOpenCL rejects unnamed");

    expect(gpu_backend_satisfies_requirement("Metal", mtlclvk), "MetalOrOpenCLOrVulkan accepts Metal");
    expect(gpu_backend_satisfies_requirement("MTL", mtlclvk), "MetalOrOpenCLOrVulkan accepts MTL");
    expect(gpu_backend_satisfies_requirement("OpenCL", mtlclvk), "MetalOrOpenCLOrVulkan accepts OpenCL");
    expect(gpu_backend_satisfies_requirement("Vulkan", mtlclvk), "MetalOrOpenCLOrVulkan accepts Vulkan");
    expect(!gpu_backend_satisfies_requirement("CUDA", mtlclvk), "MetalOrOpenCLOrVulkan rejects CUDA");
    expect(!gpu_backend_satisfies_requirement(nullptr, mtlclvk), "MetalOrOpenCLOrVulkan rejects unnamed");

    if (g_failures) {
        fprintf(stderr, "%d requirement checks failed\n", g_failures);
        return 1;
    }
    fprintf(stderr, "PASS\n");
    return 0;
}
