#include "vae_gguf.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

namespace tts_cpp::acestep {

bool vae_gguf_open(VaeGGUF & g, const std::string & path) {
    if (!mapped_file_open(g.file, path, "acestep-vae")) return false;
    struct gguf_init_params p = { /*no_alloc=*/true, /*ctx=*/&g.meta };
    g.ctx = gguf_init_from_file(path.c_str(), p);
    if (!g.ctx) { fprintf(stderr, "[acestep-vae] failed to parse %s\n", path.c_str()); return false; }
    g.data_off = gguf_get_data_offset(g.ctx);
    return true;
}

void vae_gguf_close(VaeGGUF & g) {
    if (g.ctx) gguf_free(g.ctx);
    if (g.meta) ggml_free(g.meta);
    mapped_file_close(g.file);
    g.ctx      = nullptr;
    g.meta     = nullptr;
    g.data_off = 0;
}

const void * vae_gdata(const VaeGGUF & g, const std::string & name) {
    int64_t idx = gguf_find_tensor(g.ctx, name.c_str());
    if (idx < 0) return nullptr;
    return g.file.data + g.data_off + gguf_get_tensor_offset(g.ctx, idx);
}

ggml_tensor * vae_gmeta(const VaeGGUF & g, const std::string & name) {
    return ggml_get_tensor(g.meta, name.c_str());
}

bool vae_gguf_has(const VaeGGUF & g, const std::string & name) {
    return gguf_find_tensor(g.ctx, name.c_str()) >= 0;
}

static float bf16_to_f32(uint16_t v) {
    ggml_bf16_t b; b.bits = v; return ggml_bf16_to_fp32(b);
}

// Weight-norm resolution and f16 conversion walk every conv weight on the
// host at load time; single-threaded they cost ~1.2 s per VAE load. The rows
// are independent, so split [0, n) into per-thread ranges (bit-identical:
// per-row arithmetic order is unchanged).
template <typename F>
static void parallel_rows(int n, F && fn) {
    const unsigned hw        = std::thread::hardware_concurrency();
    const int      n_threads = (int) std::min<unsigned>(hw ? hw : 1, 16);
    if (n < 128 || n_threads <= 1) {
        fn(0, n);
        return;
    }
    const int chunk = (n + n_threads - 1) / n_threads;
    std::vector<std::thread> workers;
    for (int t = 0; t < n_threads; t++) {
        const int begin = t * chunk;
        const int end   = std::min(n, begin + chunk);
        if (begin >= end) break;
        workers.emplace_back([&fn, begin, end] { fn(begin, end); });
    }
    for (auto & w : workers) w.join();
}

static void upload_f32_as(ggml_tensor * dst, const std::vector<float> & w) {
    if (dst->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(w.size());
        parallel_rows((int) (w.size() / 4096) + 1, [&](int begin, int end) {
            const size_t lo = (size_t) begin * 4096;
            const size_t hi = std::min(w.size(), (size_t) end * 4096);
            if (lo < hi) ggml_fp32_to_fp16_row(w.data() + lo, h.data() + lo, (int) (hi - lo));
        });
        ggml_backend_tensor_set(dst, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(dst, w.data(), 0, w.size() * sizeof(float));
    }
}

void vae_fuse_wn(ggml_tensor * dst, const VaeGGUF & g, const std::string & pfx) {
    ggml_tensor *    mv   = vae_gmeta(g, pfx + ".weight_v");
    const uint16_t * gp   = (const uint16_t *) vae_gdata(g, pfx + ".weight_g");
    const uint16_t * vp   = (const uint16_t *) vae_gdata(g, pfx + ".weight_v");
    const int        nd   = ggml_n_dims(mv);
    const int        dim0 = (int) mv->ne[nd - 1];
    const int        fan  = (int) (ggml_nelements(mv) / dim0);
    std::vector<float> w((size_t) dim0 * fan);
    parallel_rows(dim0, [&](int begin, int end) {
        for (int d = begin; d < end; d++) {
            float gv = bf16_to_f32(gp[d]), nsq = 0.0f;
            for (int i = 0; i < fan; i++) { float vv = bf16_to_f32(vp[(size_t) d * fan + i]); nsq += vv * vv; }
            float s = gv / (sqrtf(nsq) + 1e-12f);
            for (int i = 0; i < fan; i++) w[(size_t) d * fan + i] = bf16_to_f32(vp[(size_t) d * fan + i]) * s;
        }
    });
    upload_f32_as(dst, w);
}

void vae_fuse_wn_ct(ggml_tensor * dst, const VaeGGUF & g, const std::string & pfx) {
    ggml_tensor *    mv   = vae_gmeta(g, pfx + ".weight_v");
    const uint16_t * gp   = (const uint16_t *) vae_gdata(g, pfx + ".weight_g");
    const uint16_t * vp   = (const uint16_t *) vae_gdata(g, pfx + ".weight_v");
    const int        nd   = ggml_n_dims(mv);
    const int        dim0 = (int) mv->ne[nd - 1];              // IC
    const int        fan  = (int) (ggml_nelements(mv) / dim0); // K*OC
    std::vector<float> w((size_t) dim0 * fan);
    parallel_rows(dim0, [&](int begin, int end) {
        for (int d = begin; d < end; d++) {
            float gv = bf16_to_f32(gp[d]), nsq = 0.0f;
            for (int i = 0; i < fan; i++) { float vv = bf16_to_f32(vp[(size_t) d * fan + i]); nsq += vv * vv; }
            float s = gv / (sqrtf(nsq) + 1e-12f);
            for (int i = 0; i < fan; i++) w[(size_t) i * dim0 + d] = bf16_to_f32(vp[(size_t) d * fan + i]) * s;  // transpose
        }
    });
    upload_f32_as(dst, w);
}

void vae_load_snake(ggml_tensor * dst, const VaeGGUF & g, const std::string & name, bool inv) {
    ggml_tensor *    mt  = vae_gmeta(g, name);
    const int        C   = (int) mt->ne[1];
    const uint16_t * raw = (const uint16_t *) vae_gdata(g, name);
    std::vector<float> d(C);
    for (int i = 0; i < C; i++) { float e = expf(bf16_to_f32(raw[i])); d[i] = inv ? 1.0f / e : e; }
    ggml_backend_tensor_set(dst, d.data(), 0, C * sizeof(float));
}

void vae_load_bias(ggml_tensor * dst, const VaeGGUF & g, const std::string & name) {
    ggml_tensor *    mt  = vae_gmeta(g, name);
    const int        C   = (int) mt->ne[0];
    const uint16_t * raw = (const uint16_t *) vae_gdata(g, name);
    std::vector<float> d(C);
    for (int i = 0; i < C; i++) d[i] = bf16_to_f32(raw[i]);
    ggml_backend_tensor_set(dst, d.data(), 0, C * sizeof(float));
}

} // namespace tts_cpp::acestep
