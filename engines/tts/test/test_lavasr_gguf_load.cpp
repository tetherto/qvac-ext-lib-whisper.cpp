// Fail-closed GGUF loading for the LavaSR denoiser and enhancer
// (src/lavasr/{denoiser,enhancer}_gguf.cpp).
//
// No model fixture: synthesizes tiny metadata-only GGUFs in the system temp
// directory and asserts both loaders reject them through their documented
// contract (return false, set *err) instead of half-loading: an empty file, a
// truncated file, a missing general.architecture key, the other engine's
// architecture, and a correct architecture with no weight tensors.
//
// Mistyped metadata is deliberately not exercised: the u32/f32 metadata
// readers call gguf_get_val_u32 without a type check, so a mistyped key
// GGML_ABORTs the process rather than failing closed. Same hazard the parler
// loader has; see test_gguf_load.cpp in test/parler.

#include "lavasr/denoiser_gguf.h"
#include "lavasr/enhancer_gguf.h"
#include "test_env_portable.h"

#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

#define CHECK(cond, ...) do {                                  \
    if (!(cond)) {                                             \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);   \
        fprintf(stderr, __VA_ARGS__);                          \
        fprintf(stderr, "\n");                                 \
        ++g_failures;                                          \
    }                                                          \
} while (0)

std::string fixture_path(const char * stem) {
    return (fs::temp_directory_path() /
            ("tts-lavasr-gguf-" + std::string(stem) + ".gguf")).string();
}

void write_metadata_gguf(const std::string & path, const char * architecture) {
    gguf_context * g = gguf_init_empty();
    if (architecture) {
        gguf_set_val_str(g, "general.architecture", architecture);
    }
    if (!gguf_write_to_file(g, path.c_str(), /*only_meta=*/ false)) {
        fprintf(stderr, "FATAL: cannot write fixture %s\n", path.c_str());
        exit(1);
    }
    gguf_free(g);
}

void write_raw(const std::string & path, const char * bytes, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "FATAL: cannot write fixture %s\n", path.c_str());
        exit(1);
    }
    if (n) fwrite(bytes, 1, n, f);
    fclose(f);
}

void expect_denoiser_fails(const std::string & path, const char * expected) {
    tts_cpp::lavasr::DenoiserWeights weights;
    std::string err;
    const bool ok = tts_cpp::lavasr::load_denoiser_gguf(path, weights, &err);
    CHECK(!ok, "loading %s as a denoiser must fail", path.c_str());
    CHECK(err.find(expected) != std::string::npos,
          "denoiser error for %s should mention '%s', got '%s'",
          path.c_str(), expected, err.c_str());
}

void expect_enhancer_fails(const std::string & path, const char * expected) {
    tts_cpp::lavasr::EnhancerWeights weights;
    std::string err;
    const bool ok = tts_cpp::lavasr::load_enhancer_gguf(path, weights, &err);
    CHECK(!ok, "loading %s as an enhancer must fail", path.c_str());
    CHECK(err.find(expected) != std::string::npos,
          "enhancer error for %s should mention '%s', got '%s'",
          path.c_str(), expected, err.c_str());
}

void test_missing_file() {
    const std::string path = fixture_path("absent");
    fs::remove(path);
    expect_denoiser_fails(path, "failed to open GGUF");
    expect_enhancer_fails(path, "failed to open GGUF");
}

void test_empty_and_truncated_file() {
    const std::string empty = fixture_path("empty");
    write_raw(empty, nullptr, 0);
    expect_denoiser_fails(empty, "failed to open GGUF");
    expect_enhancer_fails(empty, "failed to open GGUF");
    fs::remove(empty);

    const std::string truncated = fixture_path("truncated");
    const char header[] = { 'G', 'G', 'U', 'F' };
    write_raw(truncated, header, sizeof(header));
    expect_denoiser_fails(truncated, "failed to open GGUF");
    expect_enhancer_fails(truncated, "failed to open GGUF");
    fs::remove(truncated);
}

void test_missing_architecture_key() {
    const std::string path = fixture_path("no-arch");
    write_metadata_gguf(path, nullptr);
    expect_denoiser_fails(path, "not a lavasr-denoiser GGUF");
    expect_enhancer_fails(path, "not a lavasr-enhancer GGUF");
    fs::remove(path);
}

// The two GGUFs are structurally identical apart from this key, so each loader
// has to refuse the other's file rather than reading it as its own.
void test_architecture_is_not_interchangeable() {
    const std::string denoiser = fixture_path("arch-denoiser");
    write_metadata_gguf(denoiser, "lavasr-denoiser");
    expect_enhancer_fails(denoiser, "not a lavasr-enhancer GGUF");
    fs::remove(denoiser);

    const std::string enhancer = fixture_path("arch-enhancer");
    write_metadata_gguf(enhancer, "lavasr-enhancer");
    expect_denoiser_fails(enhancer, "not a lavasr-denoiser GGUF");
    fs::remove(enhancer);
}

void test_right_architecture_without_tensors() {
    const std::string denoiser = fixture_path("denoiser-no-tensors");
    write_metadata_gguf(denoiser, "lavasr-denoiser");
    tts_cpp::lavasr::DenoiserWeights dw;
    std::string derr;
    CHECK(!tts_cpp::lavasr::load_denoiser_gguf(denoiser, dw, &derr),
          "a denoiser GGUF with no tensors must not load");
    CHECK(!derr.empty(), "a rejected denoiser load must set an error message");
    fs::remove(denoiser);

    const std::string enhancer = fixture_path("enhancer-no-tensors");
    write_metadata_gguf(enhancer, "lavasr-enhancer");
    tts_cpp::lavasr::EnhancerWeights ew;
    std::string eerr;
    CHECK(!tts_cpp::lavasr::load_enhancer_gguf(enhancer, ew, &eerr),
          "an enhancer GGUF with no tensors must not load");
    CHECK(!eerr.empty(), "a rejected enhancer load must set an error message");
    fs::remove(enhancer);
}

void test_null_error_pointer_is_allowed() {
    const std::string path = fixture_path("null-err");
    write_metadata_gguf(path, "lavasr-denoiser");
    tts_cpp::lavasr::DenoiserWeights weights;
    CHECK(!tts_cpp::lavasr::load_denoiser_gguf(path, weights, nullptr),
          "the default err argument must still reject a tensorless GGUF");
    fs::remove(path);
}

}

int main() {
    test_missing_file();
    test_empty_and_truncated_file();
    test_missing_architecture_key();
    test_architecture_is_not_interchangeable();
    test_right_architecture_without_tensors();
    test_null_error_pointer_is_allowed();

    if (g_failures) {
        fprintf(stderr, "test-lavasr-gguf-load: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("test-lavasr-gguf-load: all checks passed\n");
    return 0;
}
