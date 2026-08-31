// Model-free unit tests for parakeet's GPU-tier selection policy
// (parakeet::gpu_tier_for).
//
// parakeet's init_gpu_backend must prefer CUDA over Vulkan on NVIDIA, and
// prefer discrete GPUs over integrated ones, matching the existing
// llm-llamacpp / diffusion behavior. The tier ranking is a pure function so
// a synthesised device topology can be scored without touching the live
// ggml-backend registry (which needs real drivers to populate). If someone
// edits init_gpu_backend's bucket order without touching the ranking here,
// the test still fails because the ordering constants encode the same
// policy.
//
// Exit 0 on success; non-zero with FAIL lines otherwise.

#include "backend_util.h"

#include "ggml-backend.h"

#include <cstdio>
#include <string>

using parakeet::GpuTier;
using parakeet::gpu_tier_for;

namespace {

int g_checks   = 0;
int g_failures = 0;

void expect(bool cond, const char * what) {
    ++g_checks;
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

#define CHECK(cond) expect((cond), #cond)

void test_gpu_tier_policy() {
    // Adreno 700+ OpenCL wins the walk — Snapdragon 8 Gen 2+ ships validated
    // OpenCL kernels that outperform Vulkan.
    CHECK(gpu_tier_for("OpenCL", GGML_BACKEND_DEVICE_TYPE_GPU,  740) == GpuTier::AdrenoOpenCL700Plus);
    CHECK(gpu_tier_for("OpenCL", GGML_BACKEND_DEVICE_TYPE_IGPU, 830) == GpuTier::AdrenoOpenCL700Plus);
    // Adreno 6xx / non-Adreno OpenCL lands in the last-resort tier — its
    // Adreno-6xx-broken skip lives inside init_gpu_backend (enumeration
    // side, not ranking side).
    CHECK(gpu_tier_for("OpenCL", GGML_BACKEND_DEVICE_TYPE_GPU,  640) == GpuTier::OpenCLOther);
    CHECK(gpu_tier_for("OpenCL", GGML_BACKEND_DEVICE_TYPE_GPU,  -1)  == GpuTier::OpenCLOther);

    // CUDA outranks Vulkan on the same NVIDIA card.
    CHECK(gpu_tier_for("CUDA",   GGML_BACKEND_DEVICE_TYPE_GPU,  -1)  == GpuTier::CudaDiscrete);
    CHECK(gpu_tier_for("Vulkan", GGML_BACKEND_DEVICE_TYPE_GPU,  -1)  == GpuTier::OtherDiscrete);
    CHECK(static_cast<int>(GpuTier::CudaDiscrete) <
          static_cast<int>(GpuTier::OtherDiscrete));

    // Tegra / Jetson report CUDA as IGPU on some drivers — CUDA-first still
    // wins over a discrete Vulkan adapter.
    CHECK(gpu_tier_for("CUDA", GGML_BACKEND_DEVICE_TYPE_IGPU, -1) == GpuTier::CudaIntegrated);
    CHECK(static_cast<int>(GpuTier::CudaIntegrated) <
          static_cast<int>(GpuTier::OtherDiscrete));

    // Discrete outranks integrated at every tier that has both — matches the
    // llm-llamacpp / diffusion dGPU-first preference.
    CHECK(static_cast<int>(GpuTier::CudaDiscrete)  < static_cast<int>(GpuTier::CudaIntegrated));
    CHECK(static_cast<int>(GpuTier::OtherDiscrete) < static_cast<int>(GpuTier::OtherIntegrated));

    // Metal is on the same tier as Vulkan (both non-OpenCL, non-CUDA GPUs).
    CHECK(gpu_tier_for("Metal", GGML_BACKEND_DEVICE_TYPE_GPU,  -1) == GpuTier::OtherDiscrete);
    CHECK(gpu_tier_for("MTL",   GGML_BACKEND_DEVICE_TYPE_IGPU, -1) == GpuTier::OtherIntegrated);

    // Non-GPU / non-IGPU devices are not selectable — CPU, ACCEL, unknown
    // types must never be picked into a GPU bucket.
    CHECK(gpu_tier_for("Vulkan", GGML_BACKEND_DEVICE_TYPE_CPU,   -1) == GpuTier::NotSelectable);
    CHECK(gpu_tier_for("BLAS",   GGML_BACKEND_DEVICE_TYPE_ACCEL, -1) == GpuTier::NotSelectable);
    CHECK(gpu_tier_for(nullptr,  GGML_BACKEND_DEVICE_TYPE_CPU,   -1) == GpuTier::NotSelectable);

    // A null / empty registry name on a GPU lands in the OtherDiscrete/Integrated
    // bucket (not OpenCL, not CUDA) — same fallback used for un-recognised drivers.
    CHECK(gpu_tier_for(nullptr, GGML_BACKEND_DEVICE_TYPE_GPU,  -1) == GpuTier::OtherDiscrete);
    CHECK(gpu_tier_for("",      GGML_BACKEND_DEVICE_TYPE_IGPU, -1) == GpuTier::OtherIntegrated);
}

}  // namespace

int main() {
    test_gpu_tier_policy();

    std::fprintf(stderr, "[test-parakeet-backend-selection] %d/%d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
