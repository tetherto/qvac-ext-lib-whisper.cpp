#ifndef QWEN_C_API_H
#define QWEN_C_API_H

#include "export.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QWEN_BACKEND_SAFETENSORS = 0,
    QWEN_BACKEND_GGUF        = 1,
} qwen_c_backend;

typedef struct qwen_engine qwen_engine;

typedef struct {
    qwen_c_backend backend;
    int            n_threads;
    int            verbose;
    const char *   language;
    const char *   system_prompt;
} qwen_c_options;

typedef struct {
    char * text;
    double encode_ms;
    double decode_ms;
    double total_ms;
    double audio_ms;
    int    text_tokens;
} qwen_c_result;

QWEN_API qwen_c_options qwen_c_options_default(void);

QWEN_API int qwen_c_backend_available(qwen_c_backend backend);

QWEN_API const char * qwen_c_backend_name(qwen_c_backend backend);

QWEN_API qwen_engine * qwen_c_engine_create(const char *   model_path,
                                            qwen_c_options options,
                                            char **        error_out);

QWEN_API void qwen_c_engine_destroy(qwen_engine * engine);

QWEN_API qwen_c_backend qwen_c_engine_backend(const qwen_engine * engine);

QWEN_API int qwen_c_engine_transcribe(qwen_engine *   engine,
                                      const char *    wav_path,
                                      qwen_c_result * result_out,
                                      char **         error_out);

QWEN_API int qwen_c_engine_transcribe_samples(qwen_engine *   engine,
                                              const float *   samples,
                                              int             n_samples,
                                              qwen_c_result * result_out,
                                              char **         error_out);

QWEN_API void qwen_c_result_free(qwen_c_result * result);

QWEN_API void qwen_c_string_free(char * s);

#ifdef __cplusplus
}
#endif

#endif
