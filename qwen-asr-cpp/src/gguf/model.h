#pragma once

#include "audio.h"
#include "hparams.h"
#include "model_loader.h"
#include "tokenizer.h"

#include <ggml.h>

#include <memory>
#include <string>
#include <vector>

namespace qwen::gguf {

struct EncoderBlock {
    ggml_tensor * attn_q_w     = nullptr;
    ggml_tensor * attn_q_b     = nullptr;
    ggml_tensor * attn_k_w     = nullptr;
    ggml_tensor * attn_k_b     = nullptr;
    ggml_tensor * attn_v_w     = nullptr;
    ggml_tensor * attn_v_b     = nullptr;
    ggml_tensor * attn_o_w     = nullptr;
    ggml_tensor * attn_o_b     = nullptr;
    ggml_tensor * attn_norm_w  = nullptr;
    ggml_tensor * attn_norm_b  = nullptr;
    ggml_tensor * ffn_gate_w   = nullptr;
    ggml_tensor * ffn_gate_b   = nullptr;
    ggml_tensor * ffn_down_w   = nullptr;
    ggml_tensor * ffn_down_b   = nullptr;
    ggml_tensor * ffn_norm_w   = nullptr;
    ggml_tensor * ffn_norm_b   = nullptr;
};

struct DecoderBlock {
    ggml_tensor * attn_q_w      = nullptr;
    ggml_tensor * attn_k_w      = nullptr;
    ggml_tensor * attn_v_w      = nullptr;
    ggml_tensor * attn_o_w      = nullptr;
    ggml_tensor * attn_q_norm_w = nullptr;
    ggml_tensor * attn_k_norm_w = nullptr;
    ggml_tensor * attn_norm_w   = nullptr;
    ggml_tensor * ffn_gate_w    = nullptr;
    ggml_tensor * ffn_up_w      = nullptr;
    ggml_tensor * ffn_down_w    = nullptr;
    ggml_tensor * ffn_norm_w    = nullptr;
};

struct Model {
    explicit Model(ModelLoader & loader);

    const Hparams & hparams;
    ModelLoader   & loader;

    ggml_tensor * conv1_w      = nullptr;
    ggml_tensor * conv1_b      = nullptr;
    ggml_tensor * conv2_w      = nullptr;
    ggml_tensor * conv2_b      = nullptr;
    ggml_tensor * conv3_w      = nullptr;
    ggml_tensor * conv3_b      = nullptr;
    ggml_tensor * conv_out_w   = nullptr;
    ggml_tensor * ln_post_w    = nullptr;
    ggml_tensor * ln_post_b    = nullptr;
    ggml_tensor * proj1_w      = nullptr;
    ggml_tensor * proj1_b      = nullptr;
    ggml_tensor * proj2_w      = nullptr;
    ggml_tensor * proj2_b      = nullptr;
    std::vector<EncoderBlock> enc;

    ggml_tensor * tok_embd_w    = nullptr;
    ggml_tensor * output_norm_w = nullptr;
    ggml_tensor * output_w      = nullptr;
    std::vector<DecoderBlock> dec;
};

struct EncoderOutput {
    std::vector<float> data;
    int                seq_len = 0;
    int                dim     = 0;
};

EncoderOutput encode_audio(const Model & model,
                           const MelSpectrogram & mel,
                           int n_threads,
                           int verbose);

struct DecoderState {
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    int                n_kv     = 0;
    int                pos      = 0;
    int                kv_dim   = 0;
    int                n_layers = 0;
    int                ctx_max  = 0;
};

void init_decoder_state(DecoderState & st, const Model & model, int ctx_max);

struct DecoderStepResult {
    std::vector<float> logits;
    int                vocab_size = 0;
};

DecoderStepResult decoder_step(const Model & model,
                               DecoderState & st,
                               const std::vector<int32_t> & input_tokens,
                               const std::vector<int32_t> & positions,
                               const EncoderOutput & enc_out,
                               int audio_token_id,
                               int n_threads,
                               bool want_logits_last_only);

int32_t greedy_argmax(const float * logits, int vocab_size);

}
