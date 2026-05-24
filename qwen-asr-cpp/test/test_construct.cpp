#include "qwen/engine.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

int fail(const char * msg) {
    std::fprintf(stderr, "test-construct: FAIL -- %s\n", msg);
    return 1;
}

int check_empty_model_dir_rejected() {
    qwen::EngineOptions opts;
    try {
        qwen::Engine engine(opts);
        return fail("expected std::runtime_error for empty model_dir");
    } catch (const std::runtime_error &) {
        return 0;
    } catch (...) {
        return fail("expected std::runtime_error, got a different exception type");
    }
}

int check_missing_model_dir_surfaces_load_error() {
    qwen::EngineOptions opts;
    opts.model_dir = "/nonexistent/qwen-asr-test-model-dir-xyz";
    try {
        qwen::Engine engine(opts);
        return fail("expected std::runtime_error when model_dir does not exist");
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

}

int main() {
    int rc = 0;
    rc |= check_empty_model_dir_rejected();
    rc |= check_missing_model_dir_surfaces_load_error();
    if (rc == 0) {
        std::puts("test-construct: OK");
    }
    return rc;
}
