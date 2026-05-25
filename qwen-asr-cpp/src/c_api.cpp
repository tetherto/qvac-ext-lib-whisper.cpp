#include "qwen/c_api.h"
#include "qwen/engine.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <string>

namespace {

char * dup_c_str(const std::string & s) {
    char * out = static_cast<char *>(std::malloc(s.size() + 1));
    if (out == nullptr) return nullptr;
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    return out;
}

void set_error(char ** error_out, const std::string & msg) {
    if (error_out != nullptr) {
        *error_out = dup_c_str(msg);
    }
}

void fill_result(qwen_c_result * out, const qwen::EngineResult & src) {
    if (out == nullptr) return;
    out->text        = dup_c_str(src.text);
    out->encode_ms   = src.encode_ms;
    out->decode_ms   = src.decode_ms;
    out->total_ms    = src.total_ms;
    out->audio_ms    = src.audio_ms;
    out->text_tokens = src.text_tokens;
}

qwen::Engine * as_engine(qwen_engine * e) { return reinterpret_cast<qwen::Engine *>(e); }
const qwen::Engine * as_engine(const qwen_engine * e) { return reinterpret_cast<const qwen::Engine *>(e); }

qwen::Backend from_c_backend(qwen_c_backend b) {
    switch (b) {
        case QWEN_BACKEND_GGUF:        return qwen::Backend::GGUF;
        case QWEN_BACKEND_SAFETENSORS:
        default:                       return qwen::Backend::Safetensors;
    }
}

qwen_c_backend to_c_backend(qwen::Backend b) {
    switch (b) {
        case qwen::Backend::GGUF:        return QWEN_BACKEND_GGUF;
        case qwen::Backend::Safetensors: return QWEN_BACKEND_SAFETENSORS;
    }
    return QWEN_BACKEND_SAFETENSORS;
}

}

extern "C" {

qwen_c_options qwen_c_options_default(void) {
    qwen_c_options o{};
    o.backend       = QWEN_BACKEND_SAFETENSORS;
    o.n_threads     = 0;
    o.verbose       = 0;
    o.language      = nullptr;
    o.system_prompt = nullptr;
    return o;
}

int qwen_c_backend_available(qwen_c_backend backend) {
    return qwen::backend_compiled_in(from_c_backend(backend)) ? 1 : 0;
}

const char * qwen_c_backend_name(qwen_c_backend backend) {
    return qwen::backend_name(from_c_backend(backend));
}

qwen_engine * qwen_c_engine_create(const char *   model_path,
                                   qwen_c_options options,
                                   char **        error_out) {
    if (model_path == nullptr || model_path[0] == '\0') {
        set_error(error_out, "qwen_c_engine_create: model_path is required");
        return nullptr;
    }
    try {
        qwen::EngineOptions opts;
        opts.backend    = from_c_backend(options.backend);
        opts.model_path = model_path;
        opts.n_threads  = options.n_threads;
        opts.verbose    = options.verbose;
        if (options.language != nullptr)      opts.language      = options.language;
        if (options.system_prompt != nullptr) opts.system_prompt = options.system_prompt;
        auto * engine = new qwen::Engine(opts);
        return reinterpret_cast<qwen_engine *>(engine);
    } catch (const std::exception & ex) {
        set_error(error_out, ex.what());
        return nullptr;
    } catch (...) {
        set_error(error_out, "qwen_c_engine_create: unknown error");
        return nullptr;
    }
}

void qwen_c_engine_destroy(qwen_engine * engine) {
    delete as_engine(engine);
}

qwen_c_backend qwen_c_engine_backend(const qwen_engine * engine) {
    if (engine == nullptr) return QWEN_BACKEND_SAFETENSORS;
    return to_c_backend(as_engine(engine)->backend());
}

int qwen_c_engine_transcribe(qwen_engine *   engine,
                             const char *    wav_path,
                             qwen_c_result * result_out,
                             char **         error_out) {
    if (engine == nullptr || wav_path == nullptr || result_out == nullptr) {
        set_error(error_out, "qwen_c_engine_transcribe: null argument");
        return -1;
    }
    try {
        const auto r = as_engine(engine)->transcribe(wav_path);
        fill_result(result_out, r);
        return 0;
    } catch (const std::exception & ex) {
        set_error(error_out, ex.what());
        return -1;
    } catch (...) {
        set_error(error_out, "qwen_c_engine_transcribe: unknown error");
        return -1;
    }
}

int qwen_c_engine_transcribe_samples(qwen_engine *   engine,
                                     const float *   samples,
                                     int             n_samples,
                                     qwen_c_result * result_out,
                                     char **         error_out) {
    if (engine == nullptr || samples == nullptr || result_out == nullptr || n_samples <= 0) {
        set_error(error_out, "qwen_c_engine_transcribe_samples: invalid argument");
        return -1;
    }
    try {
        const auto r = as_engine(engine)->transcribe_samples(samples, n_samples);
        fill_result(result_out, r);
        return 0;
    } catch (const std::exception & ex) {
        set_error(error_out, ex.what());
        return -1;
    } catch (...) {
        set_error(error_out, "qwen_c_engine_transcribe_samples: unknown error");
        return -1;
    }
}

void qwen_c_result_free(qwen_c_result * result) {
    if (result == nullptr) return;
    if (result->text != nullptr) {
        std::free(result->text);
        result->text = nullptr;
    }
}

void qwen_c_string_free(char * s) {
    if (s != nullptr) std::free(s);
}

}
