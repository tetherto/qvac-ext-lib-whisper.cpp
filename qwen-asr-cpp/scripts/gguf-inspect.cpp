#include <ggml.h>
#include <gguf.h>

#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: gguf-inspect <file.gguf> [tensor-name-substring]\n");
        return 2;
    }
    const char * path = argv[1];

    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params p { .no_alloc = true, .ctx = &meta_ctx };
    struct gguf_context * gctx = gguf_init_from_file(path, p);
    if (gctx == nullptr) {
        std::fprintf(stderr, "gguf-inspect: failed to open %s\n", path);
        return 1;
    }

    const int64_t n_kv      = gguf_get_n_kv(gctx);
    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    std::printf("file:        %s\n", path);
    std::printf("kv pairs:    %lld\n", (long long) n_kv);
    std::printf("tensors:     %lld\n", (long long) n_tensors);
    std::printf("alignment:   %zu\n", gguf_get_alignment(gctx));

    std::printf("\nmetadata keys:\n");
    for (int64_t i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(gctx, i);
        std::printf("  %s\n", key);
    }

    const char * filter = argc >= 3 ? argv[2] : nullptr;
    std::printf("\ntensors%s%s:\n",
        filter ? " matching '" : "",
        filter ? filter        : "");
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        if (filter && std::strstr(name, filter) == nullptr) continue;
        ggml_tensor * t = ggml_get_tensor(meta_ctx, name);
        const enum ggml_type type = gguf_get_tensor_type(gctx, i);
        std::printf("  %-55s type=%-6s shape=[%lld,%lld,%lld,%lld] dims=%d\n",
            name, ggml_type_name(type),
            (long long) (t ? t->ne[0] : 0), (long long) (t ? t->ne[1] : 0),
            (long long) (t ? t->ne[2] : 0), (long long) (t ? t->ne[3] : 0),
            t ? ggml_n_dims(t) : 0);
    }

    gguf_free(gctx);
    ggml_free(meta_ctx);
    return 0;
}
