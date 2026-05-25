#pragma once

#include "export.h"

namespace qwen {

enum class Backend {
    Safetensors = 0,
    GGUF        = 1,
};

QWEN_API const char * backend_name(Backend b);

QWEN_API bool backend_compiled_in(Backend b);

}
