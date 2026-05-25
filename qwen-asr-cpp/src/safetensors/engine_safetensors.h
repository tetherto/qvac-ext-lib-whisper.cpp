#pragma once

#include "qwen/engine.h"

#include <memory>

namespace qwen::safetensors {

std::unique_ptr<IEngine> make_engine(const EngineOptions & opts);

}
