#include "qwen/engine.h"

#include <stdexcept>
#include <string>

#ifdef QWEN_HAVE_BACKEND_SAFETENSORS
#include "safetensors/engine_safetensors.h"
#endif

#ifdef QWEN_HAVE_BACKEND_GGUF
#include "gguf/engine_gguf.h"
#endif

namespace qwen {

namespace {

[[noreturn]] void throw_backend_unavailable(Backend b) {
    throw std::runtime_error(
        std::string("qwen::create_engine: backend '") + backend_name(b) +
        "' is not compiled in this build");
}

}

std::unique_ptr<IEngine> create_engine(const EngineOptions & opts) {
    switch (opts.backend) {
        case Backend::Safetensors:
#ifdef QWEN_HAVE_BACKEND_SAFETENSORS
            return safetensors::make_engine(opts);
#else
            throw_backend_unavailable(opts.backend);
#endif
        case Backend::GGUF:
#ifdef QWEN_HAVE_BACKEND_GGUF
            return gguf::make_engine(opts);
#else
            throw_backend_unavailable(opts.backend);
#endif
    }
    throw std::runtime_error("qwen::create_engine: unknown backend");
}

}
