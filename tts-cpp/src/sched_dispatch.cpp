#include "sched_dispatch.h"

#include "backend_selection.h"
#include "backend_util.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace tts_cpp::detail {

bool graph_fully_supported(ggml_backend_t backend, const ggml_cgraph * gf) {
    if (!backend || !gf) return false;
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(gf));
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(gf), i);
        if (!ggml_backend_supports_op(backend, node)) {
            return false;
        }
    }
    return true;
}

bool sched_force_enabled() {
    const char * e = getenv("TTS_CPP_T3_FORCE_SCHED");
    return e && e[0] && e[0] != '0';
}

bool graph_has_unsupported_preallocated_op(ggml_backend_t primary, const ggml_cgraph * gf) {
    if (!primary || !gf) return false;
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(gf));
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(gf), i);
        ggml_backend_buffer_t buffer = node->view_src ? node->view_src->buffer : node->buffer;
        if (!buffer) continue;
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer);
        // Mirror ggml_backend_sched_backend_from_buffer: a backend qualifies
        // only when it supports BOTH the buffer type and the op.
        if (ggml_backend_supports_buft(primary, buft) &&
            ggml_backend_supports_op(primary, node)) {
            continue;
        }
        if (cpu_dev &&
            ggml_backend_dev_supports_buft(cpu_dev, buft) &&
            ggml_backend_dev_supports_op(cpu_dev, node)) {
            continue;
        }
        fprintf(stderr, "sched_dispatch: op %s (%s) writes pre-allocated buffer %s "
                        "but no backend can run it there (primary buft=%d op=%d; cpu buft=%d op=%d)\n",
                ggml_op_name(node->op), node->name, ggml_backend_buffer_name(buffer),
                (int)ggml_backend_supports_buft(primary, buft),
                (int)ggml_backend_supports_op(primary, node),
                (int)(cpu_dev && ggml_backend_dev_supports_buft(cpu_dev, buft)),
                (int)(cpu_dev && ggml_backend_dev_supports_op(cpu_dev, node)));
        return true;
    }
    return false;
}

bool sched_fallback_ensure(sched_fallback & fb, ggml_backend_t primary,
                           size_t graph_size,
                           const std::vector<ggml_backend_buffer_t> & weight_buffers) {
    if (!primary) return false;
    if (fb.sched) return true;
    try {
        // call_once re-runs the body if it throws, matching s3gen_sched_alloc's
        // retry-on-failure behaviour; a successful run is executed exactly once
        // even under concurrent callers.
        std::call_once(*fb.once, [&] {
            // Mark weights USAGE_WEIGHTS so sched copies a primary-resident
            // weight to CPU when a CPU-routed op consumes it.
            for (ggml_backend_buffer_t buf : weight_buffers) {
                if (buf) ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
            ggml_backend_t sched_backends[2] = { primary, nullptr };
            int n_sched_backends = 1;
            if (!backend_is_cpu(primary)) {
                fb.cpu_backend = init_cpu_backend();
                if (!fb.cpu_backend) {
                    throw std::runtime_error("init CPU backend for scheduler failed");
                }
                sched_backends[1] = fb.cpu_backend;
                n_sched_backends = 2;
            }
            fb.sched = ggml_backend_sched_new(sched_backends, /*bufts=*/nullptr,
                                              n_sched_backends, graph_size,
                                              /*parallel=*/false, /*op_offload=*/false);
            if (!fb.sched) {
                if (fb.cpu_backend) { ggml_backend_free(fb.cpu_backend); fb.cpu_backend = nullptr; }
                throw std::runtime_error("ggml_backend_sched_new failed");
            }
        });
    } catch (const std::exception & e) {
        fprintf(stderr, "sched_dispatch: %s\n", e.what());
        return false;
    }
    return fb.sched != nullptr;
}

bool sched_fallback_alloc(sched_fallback & fb, ggml_cgraph * gf) {
    if (!fb.sched || !gf) return false;
    ggml_backend_sched_reset(fb.sched);
    return ggml_backend_sched_alloc_graph(fb.sched, gf);
}

ggml_status sched_fallback_compute(sched_fallback & fb, ggml_backend_t primary,
                                   ggml_cgraph * gf, int n_threads) {
    if (!fb.sched || !gf) return GGML_STATUS_FAILED;
    // CPU work inside the sched runs on cpu_backend (GPU primary) or the
    // primary itself (CPU-only model).  Set its thread count per call.
    backend_set_n_threads(fb.cpu_backend ? fb.cpu_backend : primary, n_threads);
    return ggml_backend_sched_graph_compute(fb.sched, gf);
}

ggml_status direct_compute(ggml_backend_t backend, ggml_cgraph * gf, int n_threads) {
    if (!backend || !gf) return GGML_STATUS_FAILED;
    backend_set_n_threads(backend, n_threads);
    return ggml_backend_graph_compute(backend, gf);
}

void sched_fallback_free(sched_fallback & fb) {
    if (fb.sched)       { ggml_backend_sched_free(fb.sched); fb.sched = nullptr; }
    if (fb.cpu_backend) { ggml_backend_free(fb.cpu_backend); fb.cpu_backend = nullptr; }
    // Re-arm so a later ensure() can rebuild (e.g. tests reusing one model).
    fb.once = std::make_unique<std::once_flag>();
}

} // namespace tts_cpp::detail
