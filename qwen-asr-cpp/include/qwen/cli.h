#pragma once

// Library-side entry point for the qwen-asr CLI. The qwen-asr-cli executable
// is a thin shim that forwards argv straight into this function, so embedded
// hosts that want CLI semantics from inside a larger process can link
// libqwen-asr and call qwen_cli_main directly.

#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

QWEN_API int qwen_cli_main(int argc, char ** argv);

#ifdef __cplusplus
}
#endif
