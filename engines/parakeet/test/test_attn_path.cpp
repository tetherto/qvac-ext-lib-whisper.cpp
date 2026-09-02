// The fused attention is a build option, but the CPU path must keep the
// unfused graph in every build so CPU output does not depend on which GPU
// backend the binary carries. A GPU backend, when one initialises, follows
// the build option.
#include "backend_util.h"

#include "ggml-backend.h"

#include <cstdio>

namespace {

int check(bool ok, const char * what) {
    std::printf("[attn-path] %s: %s\n", what, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

ggml_backend_t init_first_gpu() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        if (ggml_backend_t b = ggml_backend_dev_init(dev, nullptr)) return b;
    }
    return nullptr;
}

} // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        std::printf("[attn-path] no CPU backend registered\n");
        return 1;
    }
    int failures = 0;
    failures += check(!parakeet::flash_attn_allowed(true, cpu),  "CPU keeps the unfused graph when flash-attn is compiled in");
    failures += check(!parakeet::flash_attn_allowed(false, cpu), "CPU keeps the unfused graph when flash-attn is compiled out");
    failures += check(!parakeet::flash_attn_allowed(true, nullptr), "no backend means no fused graph");
    ggml_backend_free(cpu);

    if (ggml_backend_t gpu = init_first_gpu()) {
        failures += check(parakeet::flash_attn_allowed(true, gpu),   "GPU follows the build option when compiled in");
        failures += check(!parakeet::flash_attn_allowed(false, gpu), "GPU follows the build option when compiled out");
        ggml_backend_free(gpu);
    } else {
        std::printf("[attn-path] no GPU backend on this host, GPU cases skipped\n");
    }
    std::printf("[attn-path] %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
