#include "model.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace qwen::gguf {

namespace {

constexpr int   ENCODER_CHUNK_SIZE     = 100;
constexpr int   ENCODER_N_WINDOW_INFER = 800;
constexpr int   CONV_KERNEL            = 3;
constexpr int   CONV_STRIDE            = 2;
constexpr int   CONV_PAD               = 1;
constexpr float ENC_LAYERNORM_EPS      = 1e-5f;

int conv_out_dim(int x) {
    return (x + 2 * CONV_PAD - CONV_KERNEL) / CONV_STRIDE + 1;
}

int tokens_for_chunk(int chunk_w) {
    const int w1 = conv_out_dim(chunk_w);
    const int w2 = conv_out_dim(w1);
    const int w3 = conv_out_dim(w2);
    return w3;
}

ggml_tensor * to_f32(ggml_context * ctx, ggml_tensor * t) {
    if (t == nullptr || t->type == GGML_TYPE_F32) return t;
    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

ggml_tensor * layer_norm(ggml_context * ctx,
                         ggml_tensor * x,
                         ggml_tensor * w,
                         ggml_tensor * b,
                         float eps) {
    ggml_tensor * h = ggml_norm(ctx, x, eps);
    h = ggml_mul(ctx, h, to_f32(ctx, w));
    if (b != nullptr) h = ggml_add(ctx, h, to_f32(ctx, b));
    return h;
}

ggml_tensor * linear(ggml_context * ctx,
                     ggml_tensor * x,
                     ggml_tensor * w,
                     ggml_tensor * b) {
    ggml_tensor * h = ggml_mul_mat(ctx, w, x);
    if (b != nullptr) h = ggml_add(ctx, h, to_f32(ctx, b));
    return h;
}

ggml_tensor * rms_norm(ggml_context * ctx,
                       ggml_tensor * x,
                       ggml_tensor * w,
                       float eps) {
    ggml_tensor * h = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, h, to_f32(ctx, w));
}

ggml_tensor * gelu_exact_or_tanh(ggml_context * ctx, ggml_tensor * x) {
    return ggml_gelu(ctx, x);
}

void sinusoidal_pe(std::vector<float> & out, int n_tokens, int dim) {
    out.assign(n_tokens * dim, 0.0f);
    const int   half = dim / 2;
    const float log_timescale = std::log(10000.0f) / static_cast<float>(half - 1);
    for (int t = 0; t < n_tokens; ++t) {
        for (int i = 0; i < half; ++i) {
            const float inv_freq = std::exp(-static_cast<float>(i) * log_timescale);
            const float angle    = static_cast<float>(t) * inv_freq;
            out[t * dim + i]        = std::sin(angle);
            out[t * dim + half + i] = std::cos(angle);
        }
    }
}

ggml_tensor * bidirectional_attention(ggml_context * ctx,
                                      ggml_tensor * q,
                                      ggml_tensor * k,
                                      ggml_tensor * v,
                                      int n_heads,
                                      int head_dim,
                                      int n_seq) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_seq);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, n_seq);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, n_seq);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
    kq = ggml_scale(ctx, kq, scale);
    kq = ggml_soft_max(ctx, kq);
    ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * out = ggml_mul_mat(ctx, v_t, kq);
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    out = ggml_reshape_2d(ctx, out, n_heads * head_dim, n_seq);
    return out;
}

ggml_tensor * encoder_layer(ggml_context * ctx,
                            ggml_tensor * x,
                            const EncoderBlock & blk,
                            int n_heads,
                            int head_dim,
                            int n_seq) {
    ggml_tensor * h = layer_norm(ctx, x, blk.attn_norm_w, blk.attn_norm_b, ENC_LAYERNORM_EPS);
    ggml_tensor * q = linear(ctx, h, blk.attn_q_w, blk.attn_q_b);
    ggml_tensor * k = linear(ctx, h, blk.attn_k_w, blk.attn_k_b);
    ggml_tensor * v = linear(ctx, h, blk.attn_v_w, blk.attn_v_b);
    ggml_tensor * a = bidirectional_attention(ctx, q, k, v, n_heads, head_dim, n_seq);
    a = linear(ctx, a, blk.attn_o_w, blk.attn_o_b);
    x = ggml_add(ctx, x, a);
    h = layer_norm(ctx, x, blk.ffn_norm_w, blk.ffn_norm_b, ENC_LAYERNORM_EPS);
    ggml_tensor * f = linear(ctx, h, blk.ffn_gate_w, blk.ffn_gate_b);
    f = gelu_exact_or_tanh(ctx, f);
    f = linear(ctx, f, blk.ffn_down_w, blk.ffn_down_b);
    x = ggml_add(ctx, x, f);
    return x;
}

ggml_tensor * add_channel_bias(ggml_context * ctx, ggml_tensor * x, ggml_tensor * b) {
    ggml_tensor * b4 = ggml_reshape_4d(ctx, to_f32(ctx, b), 1, 1, b->ne[0], 1);
    return ggml_add(ctx, x, b4);
}

ggml_tensor * encoder_conv_stem(ggml_context * ctx,
                                ggml_tensor * mel,
                                const Model & model) {
    ggml_tensor * x = mel;
    x = ggml_conv_2d(ctx, model.conv1_w, x, CONV_STRIDE, CONV_STRIDE, CONV_PAD, CONV_PAD, 1, 1);
    x = add_channel_bias(ctx, x, model.conv1_b);
    x = ggml_gelu(ctx, x);
    x = ggml_conv_2d(ctx, model.conv2_w, x, CONV_STRIDE, CONV_STRIDE, CONV_PAD, CONV_PAD, 1, 1);
    x = add_channel_bias(ctx, x, model.conv2_b);
    x = ggml_gelu(ctx, x);
    x = ggml_conv_2d(ctx, model.conv3_w, x, CONV_STRIDE, CONV_STRIDE, CONV_PAD, CONV_PAD, 1, 1);
    x = add_channel_bias(ctx, x, model.conv3_b);
    x = ggml_gelu(ctx, x);
    return x;
}

ggml_tensor * encoder_stem_forward(ggml_context * ctx,
                                   const Model & model,
                                   ggml_tensor * mel_chunk,
                                   ggml_tensor * pe_input,
                                   int n_tokens) {
    ggml_tensor * x = encoder_conv_stem(ctx, mel_chunk, model);
    const int h = static_cast<int>(x->ne[1]);
    const int c = static_cast<int>(x->ne[2]);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
    x = ggml_reshape_2d(ctx, x, h * c, n_tokens);
    x = ggml_mul_mat(ctx, model.conv_out_w, x);
    x = ggml_add(ctx, x, pe_input);
    return x;
}

ggml_tensor * make_input_mel_chunk(ggml_context * ctx,
                                   const float * mel_data,
                                   int n_mels,
                                   int n_frames_full,
                                   int chunk_start,
                                   int chunk_w) {
    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, chunk_w, n_mels, 1, 1);
    auto * dst = static_cast<float *>(t->data);
    for (int m = 0; m < n_mels; ++m) {
        std::memcpy(dst + m * chunk_w,
                    mel_data + m * n_frames_full + chunk_start,
                    chunk_w * sizeof(float));
    }
    return t;
}

ggml_tensor * make_pe_tensor(ggml_context * ctx, int n_tokens, int dim) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n_tokens);
    std::vector<float> pe;
    sinusoidal_pe(pe, n_tokens, dim);
    std::memcpy(t->data, pe.data(), pe.size() * sizeof(float));
    return t;
}

std::vector<float> encode_chunk_stem(const Model & model,
                                     const float * mel_data,
                                     int n_frames_full,
                                     int chunk_start,
                                     int chunk_w,
                                     int n_threads,
                                     int chunk_idx) {
    (void) chunk_idx;
    const int n_tokens = tokens_for_chunk(chunk_w);
    const int n_mels   = static_cast<int>(model.hparams.enc_n_mels);
    const int enc_dim  = static_cast<int>(model.hparams.enc_dim);

    const size_t mem_size = 256ULL * 1024ULL * 1024ULL;
    struct ggml_init_params p{};
    p.mem_size   = mem_size;
    p.mem_buffer = nullptr;
    p.no_alloc   = false;
    ggml_context * ctx = ggml_init(p);
    if (ctx == nullptr) throw std::runtime_error("qwen::gguf::encode_chunk_stem: ggml_init failed");

    ggml_tensor * mel_in = make_input_mel_chunk(ctx, mel_data, n_mels, n_frames_full, chunk_start, chunk_w);
    ggml_tensor * pe_in  = make_pe_tensor(ctx, n_tokens, enc_dim);
    ggml_tensor * out    = encoder_stem_forward(ctx, model, mel_in, pe_in, n_tokens);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, out);
    ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    ggml_graph_compute(gf, &plan);

    std::vector<float> result(static_cast<size_t>(n_tokens) * enc_dim);
    std::memcpy(result.data(), out->data, result.size() * sizeof(float));
    ggml_free(ctx);
    return result;
}

EncoderOutput encode_transformer_and_proj(const Model & model,
                                          const std::vector<float> & stem,
                                          int total_tokens,
                                          int n_threads,
                                          int verbose) {
    const uint32_t enc_dim   = model.hparams.enc_dim;
    const uint32_t n_heads   = model.hparams.enc_heads;
    const uint32_t head_dim  = enc_dim / n_heads;
    const int      out_dim   = static_cast<int>(model.hparams.text_dim);

    const size_t base_mb       = 256ULL;
    const size_t per_token_mb  = 2ULL;
    const size_t adaptive_mb   = base_mb + per_token_mb * static_cast<size_t>(total_tokens);
    const size_t max_mb        = 1280ULL;
    const size_t pick_mb       = adaptive_mb < max_mb ? adaptive_mb : max_mb;
    const size_t mem_size = pick_mb * 1024ULL * 1024ULL;
    struct ggml_init_params p{};
    p.mem_size   = mem_size;
    p.mem_buffer = nullptr;
    p.no_alloc   = false;
    ggml_context * ctx = ggml_init(p);
    if (ctx == nullptr) throw std::runtime_error("qwen::gguf::encode_transformer: ggml_init failed");

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc_dim, total_tokens);
    std::memcpy(x->data, stem.data(), stem.size() * sizeof(float));

    for (uint32_t i = 0; i < model.hparams.enc_layers; ++i) {
        x = encoder_layer(ctx, x, model.enc[i],
                          static_cast<int>(n_heads),
                          static_cast<int>(head_dim),
                          total_tokens);
    }
    x = layer_norm(ctx, x, model.ln_post_w, model.ln_post_b, ENC_LAYERNORM_EPS);
    x = linear(ctx, x, model.proj1_w, model.proj1_b);
    x = ggml_gelu(ctx, x);
    x = linear(ctx, x, model.proj2_w, model.proj2_b);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, x);
    ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();

    if (verbose > 0) {
        std::fprintf(stderr, "qwen-asr gguf encoder transformer: tokens=%d work_size=%zu MiB\n",
                     total_tokens, plan.work_size / (1024 * 1024));
    }
    ggml_graph_compute(gf, &plan);

    EncoderOutput eo;
    eo.seq_len = total_tokens;
    eo.dim     = out_dim;
    const size_t total = static_cast<size_t>(out_dim) * total_tokens;
    eo.data.assign(total, 0.0f);
    std::memcpy(eo.data.data(), x->data, total * sizeof(float));

    ggml_free(ctx);
    return eo;
}

}

Model::Model(ModelLoader & loader_) : hparams(loader_.hparams()), loader(loader_) {
    auto T = [&](const std::string & name) { return loader.tensor(name); };
    conv1_w    = T("enc.conv2d1.weight");
    conv1_b    = T("enc.conv2d1.bias");
    conv2_w    = T("enc.conv2d2.weight");
    conv2_b    = T("enc.conv2d2.bias");
    conv3_w    = T("enc.conv2d3.weight");
    conv3_b    = T("enc.conv2d3.bias");
    conv_out_w = T("enc.conv_out.weight");
    ln_post_w  = T("enc.ln_post.weight");
    ln_post_b  = T("enc.ln_post.bias");
    proj1_w    = T("enc.proj1.weight");
    proj1_b    = T("enc.proj1.bias");
    proj2_w    = T("enc.proj2.weight");
    proj2_b    = T("enc.proj2.bias");
    enc.resize(hparams.enc_layers);
    for (uint32_t i = 0; i < hparams.enc_layers; ++i) {
        const std::string p = "enc.blk." + std::to_string(i) + ".";
        EncoderBlock & b   = enc[i];
        b.attn_q_w     = T(p + "attn_q.weight");
        b.attn_q_b     = T(p + "attn_q.bias");
        b.attn_k_w     = T(p + "attn_k.weight");
        b.attn_k_b     = T(p + "attn_k.bias");
        b.attn_v_w     = T(p + "attn_v.weight");
        b.attn_v_b     = T(p + "attn_v.bias");
        b.attn_o_w     = T(p + "attn_output.weight");
        b.attn_o_b     = T(p + "attn_output.bias");
        b.attn_norm_w  = T(p + "attn_norm.weight");
        b.attn_norm_b  = T(p + "attn_norm.bias");
        b.ffn_gate_w   = T(p + "ffn_gate.weight");
        b.ffn_gate_b   = T(p + "ffn_gate.bias");
        b.ffn_down_w   = T(p + "ffn_down.weight");
        b.ffn_down_b   = T(p + "ffn_down.bias");
        b.ffn_norm_w   = T(p + "ffn_norm.weight");
        b.ffn_norm_b   = T(p + "ffn_norm.bias");
    }
    tok_embd_w    = T("token_embd.weight");
    output_norm_w = T("output_norm.weight");
    output_w      = loader.find("output.weight") ? T("output.weight") : tok_embd_w;
    dec.resize(hparams.text_layers);
    for (uint32_t i = 0; i < hparams.text_layers; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        DecoderBlock & d   = dec[i];
        d.attn_q_w      = T(p + "attn_q.weight");
        d.attn_k_w      = T(p + "attn_k.weight");
        d.attn_v_w      = T(p + "attn_v.weight");
        d.attn_o_w      = T(p + "attn_output.weight");
        d.attn_q_norm_w = T(p + "attn_q_norm.weight");
        d.attn_k_norm_w = T(p + "attn_k_norm.weight");
        d.attn_norm_w   = T(p + "attn_norm.weight");
        d.ffn_gate_w    = T(p + "ffn_gate.weight");
        d.ffn_up_w      = T(p + "ffn_up.weight");
        d.ffn_down_w    = T(p + "ffn_down.weight");
        d.ffn_norm_w    = T(p + "ffn_norm.weight");
    }
}

namespace {

float fp16_to_f32(uint16_t h) {
    const uint32_t sign = (h >> 15) & 0x1;
    const uint32_t exp  = (h >> 10) & 0x1f;
    const uint32_t frac =  h        & 0x3ff;
    uint32_t out;
    if (exp == 0) {
        if (frac == 0) {
            out = sign << 31;
        } else {
            uint32_t e = 1;
            uint32_t f = frac;
            while ((f & 0x400) == 0) { f <<= 1; e += 1; }
            f &= 0x3ff;
            out = (sign << 31) | ((127 - 15 - e + 1) << 23) | (f << 13);
        }
    } else if (exp == 0x1f) {
        out = (sign << 31) | 0x7f800000 | (frac << 13);
    } else {
        out = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

void embed_lookup_with_audio(const Model & model,
                             const std::vector<int32_t> & input_tokens,
                             const EncoderOutput & enc_out,
                             int audio_token_id,
                             std::vector<float> & out) {
    const int dim = static_cast<int>(model.hparams.text_dim);
    const auto * weights = static_cast<const uint16_t *>(model.tok_embd_w->data);
    const int   vocab    = static_cast<int>(model.hparams.text_vocab);
    out.assign(input_tokens.size() * dim, 0.0f);
    int audio_seen = 0;
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        const int id = input_tokens[i];
        if (id == audio_token_id) {
            if (audio_seen < enc_out.seq_len) {
                std::memcpy(out.data() + i * dim,
                            enc_out.data.data() + audio_seen * enc_out.dim,
                            dim * sizeof(float));
            }
            audio_seen += 1;
            continue;
        }
        if (id < 0 || id >= vocab) continue;
        const uint16_t * row = weights + static_cast<size_t>(id) * dim;
        for (int k = 0; k < dim; ++k) {
            out[i * dim + k] = fp16_to_f32(row[k]);
        }
    }
}

ggml_tensor * rms_norm_per_head(ggml_context * ctx,
                                ggml_tensor * x,
                                ggml_tensor * w,
                                float eps) {
    ggml_tensor * h = ggml_rms_norm(ctx, x, eps);
    ggml_tensor * wf = to_f32(ctx, w);
    return ggml_mul(ctx, h, wf);
}

ggml_tensor * apply_rope(ggml_context * ctx,
                         ggml_tensor * x,
                         ggml_tensor * pos,
                         int head_dim,
                         float rope_base) {
    return ggml_rope_ext(ctx, x, pos, nullptr,
                         head_dim,
                         GGML_ROPE_TYPE_NEOX,
                         0,
                         rope_base,
                         1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

ggml_tensor * build_causal_mask(ggml_context * ctx,
                                int n_new, int n_kv,
                                const std::vector<int32_t> & positions) {
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_new);
    auto * dst = static_cast<float *>(mask->data);
    const float neg_inf = -INFINITY;
    for (int i = 0; i < n_new; ++i) {
        const int p_i = positions[i];
        for (int j = 0; j < n_kv; ++j) {
            dst[i * n_kv + j] = (j <= p_i) ? 0.0f : neg_inf;
        }
    }
    return mask;
}

ggml_tensor * decoder_ffn(ggml_context * ctx, ggml_tensor * x, const DecoderBlock & blk) {
    ggml_tensor * g = ggml_mul_mat(ctx, blk.ffn_gate_w, x);
    g = ggml_silu(ctx, g);
    ggml_tensor * u = ggml_mul_mat(ctx, blk.ffn_up_w, x);
    ggml_tensor * h = ggml_mul(ctx, g, u);
    return ggml_mul_mat(ctx, blk.ffn_down_w, h);
}

}

void init_decoder_state(DecoderState & st, const Model & model, int ctx_max) {
    const int head_dim   = static_cast<int>(model.hparams.text_head_dim);
    const int n_kv_heads = static_cast<int>(model.hparams.text_kv_heads);
    const int n_layers   = static_cast<int>(model.hparams.text_layers);
    st.kv_dim   = head_dim * n_kv_heads;
    st.n_layers = n_layers;
    st.ctx_max  = ctx_max;
    st.pos      = 0;
    st.n_kv     = 0;
    const size_t per_layer = static_cast<size_t>(ctx_max) * st.kv_dim;
    st.k_cache.assign(per_layer * n_layers, 0.0f);
    st.v_cache.assign(per_layer * n_layers, 0.0f);
}

namespace {

void run_decoder_layer(const Model & model,
                       int layer_idx,
                       DecoderState & st,
                       const std::vector<int32_t> & positions,
                       int n_total_kv,
                       std::vector<float> & x_state,
                       int n_threads) {
    const DecoderBlock & blk = model.dec[layer_idx];
    const int n_seq      = static_cast<int>(positions.size());
    const int dim        = static_cast<int>(model.hparams.text_dim);
    const int head_dim   = static_cast<int>(model.hparams.text_head_dim);
    const int n_heads    = static_cast<int>(model.hparams.text_heads);
    const int n_kv_heads = static_cast<int>(model.hparams.text_kv_heads);
    const float rope_base = model.hparams.text_rope_base;
    const float rms_eps   = model.hparams.text_rms_eps;

    const size_t per_pos_kv = static_cast<size_t>(n_kv_heads) * head_dim;
    const size_t layer_off  = static_cast<size_t>(layer_idx) * st.ctx_max * per_pos_kv;

    const size_t mem_size = 64ULL * 1024ULL * 1024ULL;
    struct ggml_init_params p{};
    p.mem_size   = mem_size;
    p.mem_buffer = nullptr;
    p.no_alloc   = false;
    ggml_context * ctx = ggml_init(p);
    if (ctx == nullptr) throw std::runtime_error("qwen::gguf::run_decoder_layer: ggml_init failed");

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n_seq);
    std::memcpy(x->data, x_state.data(), x_state.size() * sizeof(float));

    ggml_tensor * pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seq);
    std::memcpy(pos_t->data, positions.data(), n_seq * sizeof(int32_t));

    ggml_tensor * x_norm = rms_norm(ctx, x, blk.attn_norm_w, rms_eps);

    ggml_tensor * q = ggml_mul_mat(ctx, blk.attn_q_w, x_norm);
    ggml_tensor * k = ggml_mul_mat(ctx, blk.attn_k_w, x_norm);
    ggml_tensor * v = ggml_mul_mat(ctx, blk.attn_v_w, x_norm);

    q = ggml_reshape_3d(ctx, q, head_dim, n_heads,    n_seq);
    k = ggml_reshape_3d(ctx, k, head_dim, n_kv_heads, n_seq);
    v = ggml_reshape_3d(ctx, v, head_dim, n_kv_heads, n_seq);

    q = rms_norm_per_head(ctx, q, blk.attn_q_norm_w, rms_eps);
    k = rms_norm_per_head(ctx, k, blk.attn_k_norm_w, rms_eps);

    q = apply_rope(ctx, q, pos_t, head_dim, rope_base);
    k = apply_rope(ctx, k, pos_t, head_dim, rope_base);

    {
        ggml_cgraph * gtmp = ggml_new_graph(ctx);
        ggml_build_forward_expand(gtmp, k);
        ggml_build_forward_expand(gtmp, v);
        ggml_cplan plan = ggml_graph_plan(gtmp, n_threads, nullptr);
        std::vector<uint8_t> work(plan.work_size);
        plan.work_data = work.data();
        ggml_graph_compute(gtmp, &plan);
    }

    {
        const float * k_data = static_cast<const float *>(k->data);
        const float * v_data = static_cast<const float *>(v->data);
        for (int i = 0; i < n_seq; ++i) {
            const size_t dst = layer_off + static_cast<size_t>(positions[i]) * per_pos_kv;
            std::memcpy(st.k_cache.data() + dst, k_data + i * per_pos_kv, per_pos_kv * sizeof(float));
            std::memcpy(st.v_cache.data() + dst, v_data + i * per_pos_kv, per_pos_kv * sizeof(float));
        }
    }

    ggml_tensor * k_all = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_kv_heads, n_total_kv);
    ggml_tensor * v_all = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_kv_heads, n_total_kv);
    std::memcpy(k_all->data, st.k_cache.data() + layer_off,
                static_cast<size_t>(n_total_kv) * per_pos_kv * sizeof(float));
    std::memcpy(v_all->data, st.v_cache.data() + layer_off,
                static_cast<size_t>(n_total_kv) * per_pos_kv * sizeof(float));

    ggml_tensor * mask = build_causal_mask(ctx, n_seq, n_total_kv, positions);

    const float scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int   repeat = n_heads / n_kv_heads;

    ggml_tensor * q_perm = ggml_cont(ctx, ggml_permute(ctx, q,     0, 2, 1, 3));
    ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k_all, 0, 2, 1, 3));
    ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v_all, 0, 2, 1, 3));

    ggml_tensor * q4 = ggml_reshape_4d(ctx, q_perm, head_dim, n_seq,      repeat,      n_kv_heads);
    ggml_tensor * k4 = ggml_reshape_4d(ctx, k_perm, head_dim, n_total_kv, 1,           n_kv_heads);

    ggml_tensor * kq = ggml_mul_mat(ctx, k4, q4);
    kq = ggml_scale(ctx, kq, scale);

    ggml_tensor * mask4 = ggml_reshape_4d(ctx, mask, n_total_kv, n_seq, 1, 1);
    kq = ggml_add(ctx, kq, mask4);
    kq = ggml_soft_max(ctx, kq);

    ggml_tensor * v_t   = ggml_cont(ctx, ggml_transpose(ctx, v_perm));
    ggml_tensor * v4    = ggml_reshape_4d(ctx, v_t, n_total_kv, head_dim, 1, n_kv_heads);
    ggml_tensor * attn  = ggml_mul_mat(ctx, v4, kq);

    attn = ggml_reshape_3d(ctx, attn, head_dim, n_seq, n_heads);
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(ctx, attn, n_heads * head_dim, n_seq);
    ggml_tensor * o = ggml_mul_mat(ctx, blk.attn_o_w, attn);

    x = ggml_add(ctx, x, o);

    ggml_tensor * ff_norm = rms_norm(ctx, x, blk.ffn_norm_w, rms_eps);
    ggml_tensor * ff_out  = decoder_ffn(ctx, ff_norm, blk);
    x = ggml_add(ctx, x, ff_out);

    ggml_cgraph * gfinal = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gfinal, x);
    ggml_cplan plan2 = ggml_graph_plan(gfinal, n_threads, nullptr);
    std::vector<uint8_t> work2(plan2.work_size);
    plan2.work_data = work2.data();
    ggml_graph_compute(gfinal, &plan2);

    std::memcpy(x_state.data(), x->data, x_state.size() * sizeof(float));

    ggml_free(ctx);
}

void compute_final_logits(const Model & model,
                          const std::vector<float> & x_state,
                          int n_seq,
                          bool want_logits_last_only,
                          int n_threads,
                          DecoderStepResult & out) {
    const int dim   = static_cast<int>(model.hparams.text_dim);
    const int vocab = static_cast<int>(model.hparams.text_vocab);
    const float rms_eps = model.hparams.text_rms_eps;

    const size_t mem_size = static_cast<size_t>(vocab) * sizeof(float) * 8ULL
                          + 64ULL * 1024ULL * 1024ULL;
    struct ggml_init_params p{};
    p.mem_size   = mem_size;
    p.mem_buffer = nullptr;
    p.no_alloc   = false;
    ggml_context * ctx = ggml_init(p);

    const int take = want_logits_last_only ? 1 : n_seq;
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, take);
    const float * src = x_state.data() + (want_logits_last_only ? static_cast<size_t>(n_seq - 1) * dim : 0);
    std::memcpy(x->data, src, static_cast<size_t>(take) * dim * sizeof(float));

    x = rms_norm(ctx, x, model.output_norm_w, rms_eps);
    ggml_tensor * logits = ggml_mul_mat(ctx, model.output_w, x);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);
    ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    ggml_graph_compute(gf, &plan);

    out.vocab_size = vocab;
    out.logits.assign(static_cast<size_t>(vocab) * take, 0.0f);
    std::memcpy(out.logits.data(), logits->data, out.logits.size() * sizeof(float));

    ggml_free(ctx);
}

}

DecoderStepResult decoder_step(const Model & model,
                               DecoderState & st,
                               const std::vector<int32_t> & input_tokens,
                               const std::vector<int32_t> & positions,
                               const EncoderOutput & enc_out,
                               int audio_token_id,
                               int n_threads,
                               bool want_logits_last_only) {
    const int n_seq    = static_cast<int>(input_tokens.size());
    const int dim      = static_cast<int>(model.hparams.text_dim);
    const int n_layers = static_cast<int>(model.hparams.text_layers);

    int max_pos = -1;
    for (int p : positions) max_pos = std::max(max_pos, p);
    const int n_total_kv = max_pos + 1;
    if (n_total_kv > st.ctx_max) {
        throw std::runtime_error("qwen::gguf::decoder_step: KV cache exceeded ctx_max");
    }

    std::vector<float> x_state;
    embed_lookup_with_audio(model, input_tokens, enc_out, audio_token_id, x_state);

    for (int l = 0; l < n_layers; ++l) {
        run_decoder_layer(model, l, st, positions, n_total_kv, x_state, n_threads);
    }

    DecoderStepResult result;
    compute_final_logits(model, x_state, n_seq, want_logits_last_only, n_threads, result);

    st.pos  = std::max(st.pos, max_pos + 1);
    st.n_kv = std::max(st.n_kv, n_total_kv);
    (void) dim;
    return result;
}

int32_t greedy_argmax(const float * logits, int vocab_size) {
    int32_t best = 0;
    float   bv   = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}

EncoderOutput encode_audio(const Model & model,
                           const MelSpectrogram & mel,
                           int n_threads,
                           int verbose) {
    if (mel.n_frames <= 0 || mel.n_mels != static_cast<int>(model.hparams.enc_n_mels)) {
        throw std::runtime_error("qwen::gguf::encode_audio: mel input is invalid");
    }
    const int chunk_size = ENCODER_CHUNK_SIZE;
    const int n_chunks   = (mel.n_frames + chunk_size - 1) / chunk_size;
    const int out_dim    = static_cast<int>(model.hparams.text_dim);

    EncoderOutput full;
    full.dim = out_dim;
    (void) out_dim;
    for (int c = 0; c < n_chunks; ++c) {
        const int start = c * chunk_size;
        const int width = std::min(chunk_size, mel.n_frames - start);
        std::vector<float> stem = encode_chunk_stem(model, mel.data.data(), mel.n_frames,
                                                    start, width, n_threads, c);
        const int piece_tokens = tokens_for_chunk(width);
        EncoderOutput piece = encode_transformer_and_proj(model, stem, piece_tokens, n_threads, 0);
        full.dim = piece.dim;
        full.data.insert(full.data.end(), piece.data.begin(), piece.data.end());
        full.seq_len += piece.seq_len;
        if (verbose > 0) {
            std::fprintf(stderr,
                "qwen-asr gguf encoder: chunk %d/%d width=%d tokens=%d total=%d dim=%d\n",
                c + 1, n_chunks, width, piece_tokens, full.seq_len, full.dim);
        }
    }
    return full;
}

}
