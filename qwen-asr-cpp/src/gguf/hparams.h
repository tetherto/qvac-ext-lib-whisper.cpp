#pragma once

#include <cstdint>

namespace qwen::gguf {

struct Hparams {
    uint32_t text_ctx           = 0;
    uint32_t text_dim           = 0;
    uint32_t text_ff            = 0;
    uint32_t text_layers        = 0;
    uint32_t text_heads         = 0;
    uint32_t text_kv_heads      = 0;
    float    text_rms_eps       = 1e-6f;
    float    text_rope_base     = 1000000.0f;
    uint32_t text_vocab         = 0;
    uint32_t text_head_dim      = 0;

    uint32_t enc_dim            = 0;
    uint32_t enc_ff             = 0;
    uint32_t enc_layers         = 0;
    uint32_t enc_heads          = 0;
    uint32_t enc_n_mels         = 128;

    uint32_t bos_token_id       = 151643;
    uint32_t eos_token_id       = 151645;
    uint32_t pad_token_id       = 151643;
    uint32_t audio_token_id     = 151676;
    uint32_t audio_start_id     = 151669;
    uint32_t audio_end_id       = 151670;
    uint32_t im_start_id        = 151644;
    uint32_t im_end_id          = 151645;
    uint32_t asr_text_id        = 151677;
};

}
