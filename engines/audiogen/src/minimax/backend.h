#pragma once

#include "acestep/backend_registry.h"
#include "ggml-backend.h"
#include "logic.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

struct BackendPair {
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    bool has_gpu = false;
};

inline BackendPair g_backend_cache = {};
inline int g_backend_refs = 0;
inline int g_backend_threads = 1;
inline std::string g_backend_modules_dir;
inline std::string g_backend_device = "cpu";

static void backend_configure_cpu(int n_threads, const std::string & modules_dir) {
    const int requested_threads = n_threads > 0 ? n_threads : 1;
    if (!tts_cpp::minimax::detail::backend_configuration_matches(
            g_backend_refs, g_backend_threads, g_backend_modules_dir, requested_threads, modules_dir)) {
        throw std::runtime_error(
            "minimax engine: active CPU backend uses different thread or backend directory options");
    }
    g_backend_threads = requested_threads;
    g_backend_modules_dir = modules_dir;
}

// Resolve the requested compute device: an explicit option wins, then the
// MM3_DEVICE environment variable, then "cpu". Values: cpu | gpu | auto.
static void backend_configure_device(const std::string & device) {
    std::string requested = device;
    if (requested.empty()) {
        const char * env = std::getenv("MM3_DEVICE");
        requested = env && *env ? env : "cpu";
    }
    if (requested != "cpu" && requested != "gpu" && requested != "auto") {
        throw std::runtime_error("minimax engine: device must be cpu, gpu or auto, got '" + requested + "'");
    }
    if (g_backend_refs > 0 && requested != g_backend_device) {
        throw std::runtime_error("minimax engine: active backend already initialised with device '" +
                                 g_backend_device + "'");
    }
    g_backend_device = requested;
}

static void backend_set_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_reg_t registry = device ? ggml_backend_dev_backend_reg(device) : nullptr;
    if (!registry) {
        return;
    }
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(registry, "ggml_backend_set_n_threads"));
    if (set_threads) {
        set_threads(backend, n_threads);
    }
}

static void backend_set_abort_handler(ggml_backend_t backend, ggml_abort_callback callback, void * user_data) {
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_reg_t registry = device ? ggml_backend_dev_backend_reg(device) : nullptr;
    if (!registry) {
        return;
    }
    auto set_abort = reinterpret_cast<ggml_backend_set_abort_callback_t>(
        ggml_backend_reg_get_proc_address(registry, "ggml_backend_set_abort_callback"));
    if (set_abort) {
        set_abort(backend, callback, user_data);
    }
}

static ggml_backend_t backend_create_cpu() {
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t backend = device ? ggml_backend_dev_init(device, nullptr) : nullptr;
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (backend) {
        backend_set_threads(backend, g_backend_threads);
    }
    return backend;
}

static BackendPair backend_init(const char * tag) {
    if (g_backend_refs > 0) {
        ++g_backend_refs;
        return g_backend_cache;
    }
    if (!g_backend_modules_dir.empty()) {
        ggml_backend_load_all_from_path(g_backend_modules_dir.c_str());
    } else {
        ggml_backend_load_all();
    }
    BackendPair pair;
    pair.cpu_backend = backend_create_cpu();
    pair.backend = pair.cpu_backend;
    if (!pair.cpu_backend) {
        throw std::runtime_error("minimax engine: CPU backend initialization failed");
    }
    if (g_backend_device == "gpu" || g_backend_device == "auto") {
        if (ggml_backend_t gpu = tts_cpp::acestep::backend_gpu_init()) {
            pair.backend = gpu;
            pair.has_gpu = true;
            fprintf(stderr, "[%s] Using GPU backend %s (%s); CPU handles unsupported ops\n", tag,
                    tts_cpp::acestep::backend_reg_name(gpu), ggml_backend_name(gpu));
        } else if (g_backend_device == "gpu") {
            // An explicit GPU request must not silently degrade into a run that
            // is orders of magnitude slower; only device=auto may fall back.
            ggml_backend_free(pair.cpu_backend);
            throw std::runtime_error(
                "minimax engine: device=gpu but no usable GPU backend was found; "
                "use device=auto for CPU fallback");
        } else {
            fprintf(stderr, "[%s] device=auto found no usable GPU backend; using CPU\n", tag);
        }
    }
    g_backend_cache = pair;
    g_backend_refs = 1;
    return pair;
}

static void backend_release(ggml_backend_t backend, ggml_backend_t cpu_backend) {
    if (g_backend_refs <= 0) {
        return;
    }
    --g_backend_refs;
    if (g_backend_refs != 0) {
        return;
    }
    if (backend && backend != cpu_backend) {
        ggml_backend_free(backend);
    }
    if (cpu_backend) {
        ggml_backend_free(cpu_backend);
    }
    g_backend_cache = {};
}

static ggml_backend_sched_t backend_sched_new(BackendPair pair, int max_nodes) {
    ggml_backend_t backends[2] = {pair.backend, pair.cpu_backend};
    ggml_backend_buffer_type_t buffer_types[2] = {ggml_backend_get_default_buffer_type(pair.backend),
                                                  ggml_backend_get_default_buffer_type(pair.cpu_backend)};
    const int n_backends = pair.has_gpu ? 2 : 1;
    ggml_backend_sched_t scheduler =
        ggml_backend_sched_new(backends, buffer_types, n_backends, max_nodes, false, true);
    if (!scheduler) {
        throw std::runtime_error("minimax engine: scheduler initialization failed");
    }
    return scheduler;
}
