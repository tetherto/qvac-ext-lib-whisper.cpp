#pragma once

#include "mm3-model.h"
#include "mm3-sample.h"

#include "backend.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#define MM3_DEPTH_MAX_NODES 512

#define MM3_DEPTH_MAX_STEPS 16

struct MM3DepthStep {
    ggml_backend_sched_t sched = nullptr;
    ggml_context *       gctx  = nullptr;
    uint8_t *            gbuf  = nullptr;
    ggml_cgraph *        graph = nullptr;

    ggml_tensor * in_hidden = nullptr;
    ggml_tensor * in_sem    = nullptr;
    ggml_tensor * in_ac     = nullptr;
    ggml_tensor * out_fused = nullptr;

    ggml_tensor * mask = nullptr;
    int64_t       S    = 0;
    size_t        compute_bytes = 0;
    int           n_nodes       = 0;
};

struct MM3DepthGraph {
    ggml_backend_t backend     = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    bool           backend_ref = false;
    WeightCtx      prep        = {};

    const void * synth_token = nullptr;
    const void * lm_token    = nullptr;

    MM3DepthStep step[MM3_DEPTH_MAX_STEPS];
    int          n_steps = 0;
};

struct MM3DepthFrame {
    int32_t            codes[MM3_DEPTH_MAX_STEPS] = { 0 };
    std::vector<float> hiddens;
    std::vector<float> logits_cond;
    std::vector<float> logits_uncond;
    int                n_codes = 0;
    double             ms      = 0.0;
};

static void mm3_depth_free_step(MM3DepthStep * s) {
    if (s->gctx) {
        if (s->sched) {
            ggml_backend_sched_reset(s->sched);
        }
        ggml_free(s->gctx);
        free(s->gbuf);
    }
    if (s->sched) {
        ggml_backend_sched_free(s->sched);
    }
    *s = MM3DepthStep{};
}

static void mm3_depth_free(MM3DepthGraph * g) {
    for (int i = 0; i < MM3_DEPTH_MAX_STEPS; i++) {
        mm3_depth_free_step(&g->step[i]);
    }
    wctx_free(&g->prep);
    g->n_steps     = 0;
    g->synth_token = nullptr;
    g->lm_token    = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

static ggml_tensor * mm3_depth_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

static ggml_tensor * mm3_depth_block(ggml_context * ctx, const MM3DepthConfig & c, const MM3DepthLayer & w,
                                     ggml_tensor * h, ggml_tensor * mask) {
    const int64_t H  = (int64_t) c.embedding_length;
    const int64_t D  = (int64_t) c.head_dim;
    const int64_t Nh = (int64_t) c.head_count;
    const int64_t S  = h->ne[1];
    const int64_t B  = h->ne[2];

    ggml_tensor * n = mm3_depth_norm(ctx, h, w.attn_norm, c.rms_eps);

    ggml_tensor * q = ggml_mul_mat(ctx, w.attn_q, n);
    ggml_tensor * k = ggml_mul_mat(ctx, w.attn_k, n);
    ggml_tensor * v = ggml_mul_mat(ctx, w.attn_v, n);

    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, D, Nh, S, B), 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, D, Nh, S, B), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, D, Nh, S, B), 0, 2, 1, 3));

    const float   scale  = 1.0f / sqrtf((float) D);
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores               = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);

    ggml_tensor * vt   = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * attn = ggml_mul_mat(ctx, vt, scores);
    attn               = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn               = ggml_reshape_3d(ctx, attn, H, S, B);

    h = ggml_add(ctx, h, ggml_mul_mat(ctx, w.attn_output, attn));

    ggml_tensor * n2   = mm3_depth_norm(ctx, h, w.ffn_norm, c.rms_eps);
    ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, w.ffn_gate, n2));
    ggml_tensor * up   = ggml_mul_mat(ctx, w.ffn_up, n2);
    ggml_tensor * y    = ggml_mul_mat(ctx, w.ffn_down, ggml_mul(ctx, gate, up));
    return ggml_add(ctx, h, y);
}

static bool mm3_depth_build_step(const MM3Model & m, MM3DepthGraph * g, int cb, std::string * err) {
    const MM3DepthConfig &  c  = m.synth_cfg.depth;
    const MM3DepthWeights & w  = m.synth.depth;
    const int64_t           H  = (int64_t) c.embedding_length;
    const int64_t           S  = cb + 1;
    MM3DepthStep *          s  = &g->step[cb - 1];

    const size_t ctx_bytes =
        ggml_tensor_overhead() * (MM3_DEPTH_MAX_NODES + 64) + ggml_graph_overhead_custom(MM3_DEPTH_MAX_NODES, false);
    s->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!s->gbuf) {
        if (err) {
            *err = "out of host memory allocating the depth graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, s->gbuf,  true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(s->gbuf);
        s->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the depth graph context";
        }
        return false;
    }

    s->in_hidden = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, 1, 2);
    ggml_set_name(s->in_hidden, "mm3_depth_lm_hidden");
    ggml_set_input(s->in_hidden);

    s->in_sem = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(s->in_sem, "mm3_depth_semantic_id");
    ggml_set_input(s->in_sem);

    if (cb > 1) {
        s->in_ac = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, cb - 1);
        ggml_set_name(s->in_ac, "mm3_depth_acoustic_rows");
        ggml_set_input(s->in_ac);
    }

    ggml_tensor * tok0 = ggml_mul_mat(ctx, w.proj, s->in_hidden);

    ggml_tensor * shared = ggml_get_rows(ctx, m.lm.token_embd, s->in_sem);
    if (cb > 1) {
        ggml_tensor * ac = ggml_get_rows(ctx, w.audio_embd, s->in_ac);
        shared           = ggml_concat(ctx, shared, ac, 1);
    }
    shared = ggml_mul_mat(ctx, w.proj, shared);
    shared = ggml_concat(ctx, shared, shared, 2);
    ggml_tensor * seq = ggml_concat(ctx, tok0, shared, 1);

    ggml_tensor * pos = ggml_view_2d(ctx, w.pos_embd, H, S, w.pos_embd->nb[1], 0);
    ggml_tensor * h   = ggml_add(ctx, seq, pos);

    for (size_t i = 0; i < w.blk.size(); i++) {
        h = mm3_depth_block(ctx, c, w.blk[i], h, s->mask);
    }
    h = mm3_depth_norm(ctx, h, w.output_norm, c.rms_eps);

    ggml_tensor * last = ggml_cont(
        ctx, ggml_view_3d(ctx, h, H, 1, 2, h->nb[1], h->nb[2], (size_t) (S - 1) * h->nb[1]));

    ggml_tensor * logits = ggml_mul_mat(ctx, w.head[(size_t) (cb - 1)], last);
    ggml_set_name(last, "mm3_depth_hidden");
    ggml_set_name(logits, "mm3_depth_logits");

    // One fused output per step: [hidden row0, hidden row1, logits row0,
    // logits row1], so the host needs a single readback instead of two.
    ggml_tensor * hid_flat   = ggml_reshape_1d(ctx, last, ggml_nelements(last));
    ggml_tensor * logit_flat = ggml_reshape_1d(ctx, logits, ggml_nelements(logits));
    s->out_fused             = ggml_concat(ctx, hid_flat, logit_flat, 0);
    ggml_set_name(s->out_fused, "mm3_depth_fused");
    ggml_set_output(s->out_fused);

    s->graph = ggml_new_graph_custom(ctx, MM3_DEPTH_MAX_NODES, false);
    ggml_build_forward_expand(s->graph, s->out_fused);

    ggml_backend_sched_reset(s->sched);
    if (!ggml_backend_sched_alloc_graph(s->sched, s->graph)) {
        ggml_free(ctx);
        free(s->gbuf);
        s->gbuf  = nullptr;
        s->graph = nullptr;
        if (err) {
            *err = "depth graph allocation failed (out of VRAM?) for codebook " + std::to_string(cb);
        }
        return false;
    }

    s->gctx          = ctx;
    s->S             = S;
    s->n_nodes       = ggml_graph_n_nodes(s->graph);
    s->compute_bytes = ggml_backend_sched_get_buffer_size(s->sched, g->backend);
    return true;
}

static bool mm3_depth_prepare(const MM3Model & m, MM3DepthGraph * g, std::string * err) {
    if (!m.loaded) {
        if (err) {
            *err = "MiniMax-Music3 is not warm (POST /mm3/warm first)";
        }
        return false;
    }
    const void * st = (const void *) m.wctx_synth.buffer;
    const void * lt = (const void *) m.wctx_lm.buffer;
    if (g->synth_token == st && g->lm_token == lt && g->n_steps > 0) {
        return true;
    }
    mm3_depth_free(g);

    const MM3DepthConfig & c = m.synth_cfg.depth;
    if (c.block_count == 0 || c.embedding_length == 0 || c.head_count == 0 || c.head_dim == 0) {
        if (err) {
            *err = "depth config is empty — mm3.depth.* KVs missing from the synth GGUF";
        }
        return false;
    }
    if (c.rope) {
        if (err) {
            *err = "mm3.depth.rope is true; this port implements the documented learned-absolute-position variant only";
        }
        return false;
    }
    if (!c.causal) {
        if (err) {
            *err = "mm3.depth.causal is false; this port implements the documented causal variant only";
        }
        return false;
    }
    const int NC = (int) c.num_codebooks - 1;
    if (NC < 1 || NC >= MM3_DEPTH_MAX_STEPS) {
        if (err) {
            *err = "mm3.depth.num_codebooks - 1 = " + std::to_string(NC) + " is outside 1.." +
                   std::to_string(MM3_DEPTH_MAX_STEPS - 1);
        }
        return false;
    }
    if ((int64_t) c.max_position < (int64_t) NC + 1) {
        if (err) {
            *err = "depth.pos_embd has " + std::to_string(c.max_position) + " rows, need " + std::to_string(NC + 1);
        }
        return false;
    }
    if (!m.lm.token_embd) {
        if (err) {
            *err = "the LM token embedding is not resident; the depth decoder needs it for the semantic code";
        }
        return false;
    }

    if (!m.synth.depth.pos_embd || m.synth.depth.pos_embd->type != GGML_TYPE_F32) {
        if (err) {
            *err = "depth.pos_embd.weight is not F32; the layout contract pins it to F32";
        }
        return false;
    }

    BackendPair bp = backend_init("MM3-Depth");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;

    wctx_init(&g->prep, NC);
    for (int cb = 1; cb <= NC; cb++) {
        const int64_t S = cb + 1;
        auto          d = std::make_unique<float[]>((size_t) (S * S));
        for (int64_t q = 0; q < S; q++) {
            for (int64_t k = 0; k < S; k++) {
                d[(size_t) (k + q * S)] = k <= q ? 0.0f : -INFINITY;
            }
        }
        ggml_tensor * t = ggml_new_tensor_2d(g->prep.ctx, GGML_TYPE_F32, S, S);
        char          nm[64];
        snprintf(nm, sizeof(nm), "mm3.depth.causal_mask.%d", cb);
        ggml_set_name(t, nm);
        g->prep.pending.push_back({ t, d.get(), (size_t) (S * S) * sizeof(float), 0 });
        g->prep.staging.push_back(std::move(d));
        g->step[cb - 1].mask = t;
    }
    if (!wctx_alloc(&g->prep, g->backend)) {
        if (err) {
            *err = "backend buffer allocation failed for the depth causal masks";
        }
        mm3_depth_free(g);
        return false;
    }

    for (int cb = 1; cb <= NC; cb++) {
        g->step[cb - 1].sched = backend_sched_new(bp, MM3_DEPTH_MAX_NODES * 2);
        if (!mm3_depth_build_step(m, g, cb, err)) {
            mm3_depth_free(g);
            return false;
        }
    }
    g->n_steps     = NC;
    g->synth_token = st;
    g->lm_token    = lt;

    size_t total = 0;
    for (int i = 0; i < NC; i++) {
        total += g->step[i].compute_bytes;
    }
    fprintf(stderr,
            "[MM3-Depth] Prepared: %u blocks, %u heads x %u, %d step graphs (S=2..%d, %d nodes each), "
            "compute buffers %.1f MB\n",
            c.block_count, c.head_count, c.head_dim, NC, NC + 1, g->step[NC - 1].n_nodes,
            (double) total / (1024.0 * 1024.0));
    return true;
}

static MM3DepthGraph g_mm3_depth;

static bool mm3_depth_decode_frame(const MM3Model & m, const float * lm_hidden_cond, const float * lm_hidden_uncond,
                                   int32_t semantic_code, const int32_t * forced_codes, MM3DepthFrame * out,
                                   std::string * err = nullptr, std::mt19937_64 * rng = nullptr, int top_k = 0) {
    if (!mm3_depth_prepare(m, &g_mm3_depth, err)) {
        return false;
    }
    const MM3DepthConfig & c  = m.synth_cfg.depth;
    const MM3LmConfig &    lc = m.lm_cfg;
    const int64_t          H  = (int64_t) c.embedding_length;
    const int64_t          V  = (int64_t) c.audio_vocab_size;
    const int              NC = g_mm3_depth.n_steps;

    if (semantic_code < 0 || (uint32_t) semantic_code >= lc.semantic_vocab_size) {
        if (err) {
            *err = "semantic code " + std::to_string(semantic_code) + " is outside [0, " +
                   std::to_string(lc.semantic_vocab_size) + ")";
        }
        return false;
    }
    if (forced_codes) {
        for (int i = 0; i < NC; i++) {
            if (forced_codes[i] < 0 || (int64_t) forced_codes[i] >= V) {
                if (err) {
                    *err = "forced code " + std::to_string(forced_codes[i]) + " for codebook " + std::to_string(i + 1) +
                           " is outside [0, " + std::to_string(V) + ")";
                }
                return false;
            }
        }
    }

    out->n_codes = NC;
    out->hiddens.assign((size_t) (NC * H), 0.0f);
    out->logits_cond.assign((size_t) (NC * V), 0.0f);
    out->logits_uncond.assign((size_t) (NC * V), 0.0f);

    const int32_t      sem_id  = semantic_code + (int32_t) lc.semantic_vocab_offset;
    const float        cfg     = lc.ar_cfg_scale > 0.0f ? lc.ar_cfg_scale : 1.5f;
    std::vector<int32_t> ac_rows((size_t) NC, 0);
    std::vector<float>   fused_buf((size_t) (H * 2 + V * 2));
    std::vector<float>   samp_scratch;

    const auto t0 = std::chrono::steady_clock::now();
    for (int cb = 1; cb <= NC; cb++) {
        MM3DepthStep * s = &g_mm3_depth.step[cb - 1];

        ggml_backend_tensor_set(s->in_hidden, lm_hidden_cond, 0, (size_t) H * sizeof(float));
        ggml_backend_tensor_set(s->in_hidden, lm_hidden_uncond, (size_t) H * sizeof(float),
                                (size_t) H * sizeof(float));
        ggml_backend_tensor_set(s->in_sem, &sem_id, 0, sizeof(int32_t));
        if (s->in_ac) {
            ggml_backend_tensor_set(s->in_ac, ac_rows.data(), 0, (size_t) (cb - 1) * sizeof(int32_t));
        }

        if (ggml_backend_sched_graph_compute(s->sched, s->graph) != GGML_STATUS_SUCCESS) {
            if (err) {
                *err = "depth graph compute failed at codebook " + std::to_string(cb);
            }
            return false;
        }

        ggml_backend_tensor_get(s->out_fused, fused_buf.data(), 0, fused_buf.size() * sizeof(float));

        // Offsets follow the fused layout built in mm3_depth_build_step.
        float * hid_buf   = fused_buf.data();
        float * logit_buf = fused_buf.data() + (size_t) (H * 2);

        float * lc_row = out->logits_cond.data() + (size_t) ((cb - 1) * V);
        float * lu_row = out->logits_uncond.data() + (size_t) ((cb - 1) * V);
        memcpy(lc_row, logit_buf, (size_t) V * sizeof(float));
        memcpy(lu_row, logit_buf + V, (size_t) V * sizeof(float));

        memcpy(out->hiddens.data() + (size_t) ((cb - 1) * H), hid_buf, (size_t) H * sizeof(float));

        int32_t code;
        if (forced_codes) {
            code = forced_codes[cb - 1];
        } else {

            float * guided = logit_buf + V;
            for (int64_t i = 0; i < V; i++) {
                const float u = lu_row[i];
                guided[i]     = u + cfg * (lc_row[i] - u);
            }
            if (rng) {
                code = (int32_t) mm3_sample_top_k(guided, V, top_k, *rng, &samp_scratch);
            } else {
                int32_t best_i = 0;
                float   best_v = -INFINITY;
                for (int64_t i = 0; i < V; i++) {
                    if (guided[i] > best_v) {
                        best_v = guided[i];
                        best_i = (int32_t) i;
                    }
                }
                code = best_i;
            }
        }
        out->codes[cb - 1] = code;

        ac_rows[(size_t) (cb - 1)] = code + (int32_t) ((cb - 1) * V);
    }
    out->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}
