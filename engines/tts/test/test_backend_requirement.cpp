// Policy test for GpuBackendRequirement filtering in the backend-selection
// walk. Runs devicelessly: gpu_backend_satisfies_requirement() is the exact
// predicate init_gpu_backend() applies per registered device, so pinning it
// pins which devices each requirement may consider. The multi-backend case
// this guards: with Metal|OpenCL, a Vulkan device that sorts first must be
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
    const auto any     = GpuBackendRequirement::Any;
    const auto vulkan  = GpuBackendRequirement::Vulkan;
    const auto opencl  = GpuBackendRequirement::OpenCL;
    const auto cuda    = GpuBackendRequirement::CUDA;
    const auto vkmtl   = GpuBackendRequirement::Vulkan | GpuBackendRequirement::Metal;
    const auto mtlcl   = GpuBackendRequirement::Metal  | GpuBackendRequirement::OpenCL;
    const auto vkmtlcl = GpuBackendRequirement::Vulkan | GpuBackendRequirement::Metal |
                         GpuBackendRequirement::OpenCL;
    // The set both engines admit: Audio8 asks for it on every platform,
    // CosyVoice3 on desktop (cosyvoice_gpu_requirement()).
    const auto vkmtlclcu = vkmtlcl | GpuBackendRequirement::CUDA;

    expect(gpu_backend_satisfies_requirement("Metal", any), "Any accepts Metal");
    expect(gpu_backend_satisfies_requirement("Vulkan", any), "Any accepts Vulkan");
    expect(gpu_backend_satisfies_requirement("CUDA", any), "Any accepts CUDA");
    expect(gpu_backend_satisfies_requirement(nullptr, any), "Any accepts unnamed");

    expect(gpu_backend_satisfies_requirement("Vulkan", vulkan), "Vulkan accepts Vulkan");
    expect(!gpu_backend_satisfies_requirement("Metal", vulkan), "Vulkan rejects Metal");
    expect(!gpu_backend_satisfies_requirement(nullptr, vulkan), "Vulkan rejects unnamed");

    expect(gpu_backend_satisfies_requirement("OpenCL", opencl), "OpenCL accepts OpenCL");
    expect(!gpu_backend_satisfies_requirement("Vulkan", opencl), "OpenCL rejects Vulkan");
    expect(!gpu_backend_satisfies_requirement("MTL", opencl), "OpenCL rejects MTL");
    expect(!gpu_backend_satisfies_requirement(nullptr, opencl), "OpenCL rejects unnamed");

    expect(gpu_backend_satisfies_requirement("Vulkan", vkmtl), "Vulkan|Metal accepts Vulkan");
    expect(gpu_backend_satisfies_requirement("MTL", vkmtl), "Vulkan|Metal accepts MTL");
    expect(gpu_backend_satisfies_requirement("Metal", vkmtl), "Vulkan|Metal accepts Metal under its former name");
    expect(!gpu_backend_satisfies_requirement("OpenCL", vkmtl), "Vulkan|Metal rejects OpenCL");
    expect(!gpu_backend_satisfies_requirement("CUDA", vkmtl), "Vulkan|Metal rejects CUDA");
    expect(!gpu_backend_satisfies_requirement(nullptr, vkmtl), "Vulkan|Metal rejects unnamed");

    expect(gpu_backend_satisfies_requirement("Metal", mtlcl), "Metal|OpenCL accepts Metal");
    expect(gpu_backend_satisfies_requirement("MTL", mtlcl), "Metal|OpenCL accepts MTL");
    expect(gpu_backend_satisfies_requirement("OpenCL", mtlcl), "Metal|OpenCL accepts OpenCL");
    expect(!gpu_backend_satisfies_requirement("Vulkan", mtlcl), "Metal|OpenCL rejects Vulkan");
    expect(!gpu_backend_satisfies_requirement("CUDA", mtlcl), "Metal|OpenCL rejects CUDA");
    expect(!gpu_backend_satisfies_requirement(nullptr, mtlcl), "Metal|OpenCL rejects unnamed");

    // The three-backend set admits everything it names and still excludes CUDA,
    // so widening it for OpenCL did not silently turn into Any.
    expect(gpu_backend_satisfies_requirement("Vulkan", vkmtlcl), "Vulkan|Metal|OpenCL accepts Vulkan");
    expect(gpu_backend_satisfies_requirement("MTL", vkmtlcl), "Vulkan|Metal|OpenCL accepts MTL");
    expect(gpu_backend_satisfies_requirement("Metal", vkmtlcl), "Vulkan|Metal|OpenCL accepts Metal under its former name");
    expect(gpu_backend_satisfies_requirement("OpenCL", vkmtlcl), "Vulkan|Metal|OpenCL accepts OpenCL");
    expect(!gpu_backend_satisfies_requirement("CUDA", vkmtlcl), "Vulkan|Metal|OpenCL rejects CUDA");
    expect(!gpu_backend_satisfies_requirement(nullptr, vkmtlcl), "Vulkan|Metal|OpenCL rejects unnamed");

    expect(gpu_backend_satisfies_requirement("CUDA", cuda), "CUDA accepts CUDA");
    expect(!gpu_backend_satisfies_requirement("Vulkan", cuda), "CUDA rejects Vulkan");
    expect(!gpu_backend_satisfies_requirement("MTL", cuda), "CUDA rejects MTL");
    expect(!gpu_backend_satisfies_requirement("OpenCL", cuda), "CUDA rejects OpenCL");
    expect(!gpu_backend_satisfies_requirement(nullptr, cuda), "CUDA rejects unnamed");

    // The set both engines request once CUDA is admitted: it names four
    // backends and must still reject an unnamed device, so adding the CUDA
    // flag did not collapse the set into Any.
    expect(gpu_backend_satisfies_requirement("Vulkan", vkmtlclcu), "Vulkan|Metal|OpenCL|CUDA accepts Vulkan");
    expect(gpu_backend_satisfies_requirement("MTL", vkmtlclcu), "Vulkan|Metal|OpenCL|CUDA accepts MTL");
    expect(gpu_backend_satisfies_requirement("Metal", vkmtlclcu), "Vulkan|Metal|OpenCL|CUDA accepts Metal under its former name");
    expect(gpu_backend_satisfies_requirement("OpenCL", vkmtlclcu), "Vulkan|Metal|OpenCL|CUDA accepts OpenCL");
    expect(gpu_backend_satisfies_requirement("CUDA", vkmtlclcu), "Vulkan|Metal|OpenCL|CUDA accepts CUDA");
    expect(!gpu_backend_satisfies_requirement("SomeFutureGPU", vkmtlclcu), "Vulkan|Metal|OpenCL|CUDA rejects an unnamed backend");
    expect(!gpu_backend_satisfies_requirement(nullptr, vkmtlclcu), "Vulkan|Metal|OpenCL|CUDA rejects unnamed");

    if (g_failures) {
        fprintf(stderr, "%d requirement checks failed\n", g_failures);
        return 1;
    }
    fprintf(stderr, "PASS\n");
    return 0;
}
