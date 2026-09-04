#pragma once

// Size-only graph pricing that mirrors the engines' dual-path dispatch
// (src/sched_dispatch.h) by construction: a graph the primary backend fully
// supports is priced with the gallocr the direct path would reserve
// (ggml_gallocr_reserve_n_size); a graph it does not is priced through the
// same [primary, CPU-last] scheduler shape the fallback path allocates with
// (ggml_backend_sched_reserve_size). Both APIs are the size-only twins of the
// real allocators in the QVAC ggml-speech pin -- nothing is allocated on any
// device and nothing runs.
//
// Shared by the audio8 and chatterbox fit projectors.

#include "backend_selection.h"
#include "backend_util.h"
#include "sched_dispatch.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>

namespace tts_cpp::detail {

struct fit_graph_price {
    uint64_t device_bytes = 0;  // the primary backend's portion
    uint64_t host_bytes   = 0;  // the CPU-fallback portion on sched-path graphs
    bool     used_sched   = false;
};

// Price one freshly built graph. The graph's weight/state leafs must already
// be marked externally allocated (non-null data) so only true compute scratch
// is counted. Returns false when no pricer could be constructed.
inline bool fit_price_graph(ggml_backend_t backend, ggml_cgraph * graph,
                            size_t sched_graph_size, fit_graph_price & out) {
    out = fit_graph_price{};
    const bool use_sched =
        sched_force_enabled() || !graph_fully_supported(backend, graph);
    if (!use_sched) {
        ggml_gallocr_t pricer =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!pricer) return false;
        size_t size = 0;
        ggml_gallocr_reserve_n_size(pricer, graph, nullptr, nullptr, &size);
        ggml_gallocr_free(pricer);
        out.device_bytes = size;
        return true;
    }
    // Same shape as sched_fallback_ensure: [primary, CPU-last], CPU skipped
    // when the primary IS the CPU backend.
    ggml_backend_t cpu = backend_is_cpu(backend) ? nullptr : init_cpu_backend();
    ggml_backend_t backends[2];
    int n = 0;
    backends[n++] = backend;
    if (cpu) backends[n++] = cpu;
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, /*bufts=*/nullptr, n, sched_graph_size,
        /*parallel=*/false, /*op_offload=*/false);
    if (!sched) {
        if (cpu) ggml_backend_free(cpu);
        return false;
    }
    size_t sizes[2] = {0, 0};
    ggml_backend_sched_reserve_size(sched, graph, sizes);
    ggml_backend_sched_free(sched);
    if (cpu) ggml_backend_free(cpu);
    out.device_bytes = sizes[0];
    out.host_bytes   = cpu ? sizes[1] : 0;
    out.used_sched   = true;
    return true;
}

}  // namespace tts_cpp::detail
