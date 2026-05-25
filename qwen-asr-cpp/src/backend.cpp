#include "qwen/backend.h"

namespace qwen {

const char * backend_name(Backend b) {
    switch (b) {
        case Backend::Safetensors: return "safetensors";
        case Backend::GGUF:        return "gguf";
    }
    return "unknown";
}

bool backend_compiled_in(Backend b) {
    switch (b) {
        case Backend::Safetensors:
#ifdef QWEN_HAVE_BACKEND_SAFETENSORS
            return true;
#else
            return false;
#endif
        case Backend::GGUF:
#ifdef QWEN_HAVE_BACKEND_GGUF
            return true;
#else
            return false;
#endif
    }
    return false;
}

}
