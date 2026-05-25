#include "qwen/engine.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

int fail(const char * msg) {
    std::fprintf(stderr, "test-construct: FAIL -- %s\n", msg);
    return 1;
}

int check_empty_model_path_rejected() {
    qwen::EngineOptions opts;
    try {
        qwen::Engine engine(opts);
        return fail("expected std::runtime_error for empty model_path");
    } catch (const std::runtime_error &) {
        return 0;
    } catch (...) {
        return fail("expected std::runtime_error, got a different exception type");
    }
}

int check_missing_model_path_surfaces_load_error() {
    qwen::EngineOptions opts;
    opts.model_path = "/nonexistent/qwen-asr-test-model-dir-xyz";
    try {
        qwen::Engine engine(opts);
        return fail("expected std::runtime_error when model_path does not exist");
    } catch (const std::runtime_error & e) {
        const std::string what = e.what();
        if (what.find("failed to load model") == std::string::npos) {
            return fail("expected 'failed to load model' in exception message");
        }
        return 0;
    } catch (...) {
        return fail("expected std::runtime_error from qwen_load failure");
    }
}

int check_default_backend_is_safetensors() {
    qwen::EngineOptions opts;
    if (opts.backend != qwen::Backend::Safetensors) {
        return fail("expected default backend Safetensors");
    }
    if (std::string(qwen::backend_name(opts.backend)) != "safetensors") {
        return fail("expected backend_name 'safetensors'");
    }
    return 0;
}

int check_gguf_backend_reports_unimplemented_or_missing() {
    qwen::EngineOptions opts;
    opts.backend    = qwen::Backend::GGUF;
    opts.model_path = "/nonexistent/qwen-asr-test-model.gguf";
    try {
        qwen::Engine engine(opts);
        return fail("expected std::runtime_error when constructing GGUF engine");
    } catch (const std::runtime_error &) {
        return 0;
    } catch (...) {
        return fail("expected std::runtime_error from GGUF backend");
    }
}

}

int main() {
    int rc = 0;
    rc |= check_empty_model_path_rejected();
    rc |= check_missing_model_path_surfaces_load_error();
    rc |= check_default_backend_is_safetensors();
    rc |= check_gguf_backend_reports_unimplemented_or_missing();
    if (rc == 0) {
        std::puts("test-construct: OK");
    }
    return rc;
}
