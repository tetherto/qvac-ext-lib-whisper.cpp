#include "model_loader.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace qwen::gguf {

namespace {

constexpr const char * KEY_TEXT_CTX        = "qwen3-asr.text.context_length";
constexpr const char * KEY_TEXT_DIM        = "qwen3-asr.text.embedding_length";
constexpr const char * KEY_TEXT_FF         = "qwen3-asr.text.feed_forward_length";
constexpr const char * KEY_TEXT_LAYERS     = "qwen3-asr.text.block_count";
constexpr const char * KEY_TEXT_HEADS      = "qwen3-asr.text.attention.head_count";
constexpr const char * KEY_TEXT_KV_HEADS   = "qwen3-asr.text.attention.head_count_kv";
constexpr const char * KEY_TEXT_HEAD_DIM   = "qwen3-asr.text.attention.head_dim";
constexpr const char * KEY_TEXT_VOCAB      = "qwen3-asr.text.vocab_size";
constexpr const char * KEY_TEXT_RMS_EPS    = "qwen3-asr.text.attention.layer_norm_rms_epsilon";
constexpr const char * KEY_TEXT_ROPE_BASE  = "qwen3-asr.text.rope.freq_base";
constexpr const char * KEY_ENC_DIM         = "qwen3-asr.encoder.embedding_length";
constexpr const char * KEY_ENC_FF          = "qwen3-asr.encoder.feed_forward_length";
constexpr const char * KEY_ENC_LAYERS      = "qwen3-asr.encoder.block_count";
constexpr const char * KEY_ENC_HEADS       = "qwen3-asr.encoder.attention.head_count";
constexpr const char * KEY_ENC_OUT_DIM     = "qwen3-asr.encoder.output_dim";
constexpr const char * KEY_ENC_N_MELS      = "qwen3-asr.encoder.num_mel_bins";
constexpr const char * KEY_AUDIO_TOKEN     = "qwen3-asr.audio.token_id";
constexpr const char * KEY_AUDIO_START     = "qwen3-asr.audio.start_token_id";
constexpr const char * KEY_AUDIO_END       = "qwen3-asr.audio.end_token_id";

constexpr const char * KEY_VOCAB_TOKENS    = "tokenizer.ggml.tokens";
constexpr const char * KEY_VOCAB_MERGES    = "tokenizer.ggml.merges";
constexpr const char * KEY_BOS_ID          = "tokenizer.ggml.bos_token_id";
constexpr const char * KEY_EOS_ID          = "tokenizer.ggml.eos_token_id";
constexpr const char * KEY_PAD_ID          = "tokenizer.ggml.padding_token_id";

uint32_t get_u32_or(gguf_context * g, const char * key, uint32_t fallback) {
    const int64_t i = gguf_find_key(g, key);
    if (i < 0) return fallback;
    return gguf_get_val_u32(g, i);
}

float get_f32_or(gguf_context * g, const char * key, float fallback) {
    const int64_t i = gguf_find_key(g, key);
    if (i < 0) return fallback;
    return gguf_get_val_f32(g, i);
}

std::vector<std::string> read_string_array(gguf_context * g, const char * key) {
    std::vector<std::string> out;
    const int64_t i = gguf_find_key(g, key);
    if (i < 0) return out;
    const int64_t n = gguf_get_arr_n(g, i);
    out.reserve(n);
    for (int64_t k = 0; k < n; ++k) {
        out.emplace_back(gguf_get_arr_str(g, i, k));
    }
    return out;
}

}

ModelLoader::ModelLoader(const std::string & path) : path_(path) {
    struct gguf_init_params p{};
    p.no_alloc = true;
    p.ctx      = &ctx_;
    gctx_ = gguf_init_from_file(path_.c_str(), p);
    if (gctx_ == nullptr) {
        throw std::runtime_error("qwen::gguf::ModelLoader: failed to open '" + path_ + "'");
    }
    data_start_ = gguf_get_data_offset(gctx_);
    load_metadata();
    load_vocab();
    load_tensor_table();
    map_weights();
}

ModelLoader::~ModelLoader() {
    if (gctx_ != nullptr) gguf_free(gctx_);
    if (ctx_  != nullptr) ggml_free(ctx_);
    if (file_mmap_ != nullptr) munmap(file_mmap_, file_size_);
    if (file_fd_   >= 0)       close(file_fd_);
}

void ModelLoader::map_weights() {
    file_fd_ = open(path_.c_str(), O_RDONLY);
    if (file_fd_ < 0) {
        throw std::runtime_error("qwen::gguf::ModelLoader: cannot open '" + path_ + "' for mmap");
    }
    struct stat st{};
    if (fstat(file_fd_, &st) != 0) {
        throw std::runtime_error("qwen::gguf::ModelLoader: fstat failed for '" + path_ + "'");
    }
    file_size_ = static_cast<size_t>(st.st_size);
    file_mmap_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, file_fd_, 0);
    if (file_mmap_ == MAP_FAILED) {
        file_mmap_ = nullptr;
        throw std::runtime_error("qwen::gguf::ModelLoader: mmap failed for '" + path_ + "'");
    }
    uint8_t * base = static_cast<uint8_t *>(file_mmap_);
    for (auto & kv : tensors_) {
        ggml_tensor * t = kv.second.tensor;
        t->data = base + kv.second.offset;
    }
}

void ModelLoader::load_metadata() {
    hparams_.text_ctx        = get_u32_or(gctx_, KEY_TEXT_CTX,       8192);
    hparams_.text_dim        = get_u32_or(gctx_, KEY_TEXT_DIM,       1024);
    hparams_.text_ff         = get_u32_or(gctx_, KEY_TEXT_FF,        3072);
    hparams_.text_layers     = get_u32_or(gctx_, KEY_TEXT_LAYERS,    28);
    hparams_.text_heads      = get_u32_or(gctx_, KEY_TEXT_HEADS,     16);
    hparams_.text_kv_heads   = get_u32_or(gctx_, KEY_TEXT_KV_HEADS,  8);
    hparams_.text_rms_eps    = get_f32_or(gctx_, KEY_TEXT_RMS_EPS,   1e-6f);
    hparams_.text_rope_base  = get_f32_or(gctx_, KEY_TEXT_ROPE_BASE, 1000000.0f);
    hparams_.enc_dim         = get_u32_or(gctx_, KEY_ENC_DIM,        896);
    hparams_.enc_ff          = get_u32_or(gctx_, KEY_ENC_FF,         3584);
    hparams_.enc_layers      = get_u32_or(gctx_, KEY_ENC_LAYERS,     18);
    hparams_.enc_heads       = get_u32_or(gctx_, KEY_ENC_HEADS,      14);
    hparams_.enc_n_mels      = get_u32_or(gctx_, KEY_ENC_N_MELS,     128);
    const uint32_t default_head_dim =
        (hparams_.text_heads > 0) ? (hparams_.text_dim / hparams_.text_heads) : 0;
    hparams_.text_head_dim   = get_u32_or(gctx_, KEY_TEXT_HEAD_DIM, default_head_dim);
    hparams_.bos_token_id    = get_u32_or(gctx_, KEY_BOS_ID,        hparams_.bos_token_id);
    hparams_.eos_token_id    = get_u32_or(gctx_, KEY_EOS_ID,        hparams_.eos_token_id);
    hparams_.pad_token_id    = get_u32_or(gctx_, KEY_PAD_ID,        hparams_.pad_token_id);
    hparams_.audio_token_id  = get_u32_or(gctx_, KEY_AUDIO_TOKEN,   hparams_.audio_token_id);
    hparams_.audio_start_id  = get_u32_or(gctx_, KEY_AUDIO_START,   hparams_.audio_start_id);
    hparams_.audio_end_id    = get_u32_or(gctx_, KEY_AUDIO_END,     hparams_.audio_end_id);
}

void ModelLoader::load_vocab() {
    vocab_  = read_string_array(gctx_, KEY_VOCAB_TOKENS);
    merges_ = read_string_array(gctx_, KEY_VOCAB_MERGES);
    if (vocab_.empty()) {
        throw std::runtime_error("qwen::gguf::ModelLoader: vocab is empty (missing tokenizer.ggml.tokens)");
    }
    const uint32_t declared = get_u32_or(gctx_, "qwen3-asr.text.vocab_size", 0);
    hparams_.text_vocab = declared > 0 ? declared : static_cast<uint32_t>(vocab_.size());
}

void ModelLoader::load_tensor_table() {
    const int64_t n = gguf_get_n_tensors(gctx_);
    tensors_.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(gctx_, i);
        ggml_tensor * t = ggml_get_tensor(ctx_, name);
        if (t == nullptr) continue;
        LoadedTensor lt;
        lt.tensor = t;
        lt.offset = data_start_ + gguf_get_tensor_offset(gctx_, i);
        lt.nbytes = ggml_nbytes(t);
        total_bytes_ += lt.nbytes;
        tensors_.emplace(std::string(name), lt);
    }
}

const LoadedTensor * ModelLoader::find(const std::string & name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return &it->second;
}

ggml_tensor * ModelLoader::tensor(const std::string & name) const {
    const auto * lt = find(name);
    if (lt == nullptr) {
        throw std::runtime_error("qwen::gguf::ModelLoader: tensor not found: '" + name + "'");
    }
    return lt->tensor;
}

void ModelLoader::read_into(const LoadedTensor & lt, void * dst) const {
    std::ifstream in(path_, std::ios::binary);
    if (!in) throw std::runtime_error("qwen::gguf::ModelLoader: cannot reopen '" + path_ + "'");
    in.seekg(static_cast<std::streamoff>(lt.offset), std::ios::beg);
    in.read(static_cast<char *>(dst), static_cast<std::streamsize>(lt.nbytes));
    if (!in) throw std::runtime_error("qwen::gguf::ModelLoader: short read for tensor");
}

std::vector<std::string> ModelLoader::missing(const std::vector<std::string> & required) const {
    std::vector<std::string> out;
    for (const auto & name : required) {
        if (!has(name)) out.push_back(name);
    }
    return out;
}

namespace {

void emit_encoder_layers(uint32_t n, std::vector<std::string> & out) {
    for (uint32_t i = 0; i < n; ++i) {
        const std::string p = "enc.blk." + std::to_string(i) + ".";
        const char * suffixes[] = {
            "attn_q.weight", "attn_q.bias",
            "attn_k.weight", "attn_k.bias",
            "attn_v.weight", "attn_v.bias",
            "attn_output.weight", "attn_output.bias",
            "attn_norm.weight", "attn_norm.bias",
            "ffn_gate.weight", "ffn_gate.bias",
            "ffn_down.weight", "ffn_down.bias",
            "ffn_norm.weight", "ffn_norm.bias",
        };
        for (const char * s : suffixes) out.push_back(p + s);
    }
}

void emit_decoder_layers(uint32_t n, std::vector<std::string> & out) {
    for (uint32_t i = 0; i < n; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        const char * suffixes[] = {
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "attn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "ffn_norm.weight",
        };
        for (const char * s : suffixes) out.push_back(p + s);
    }
}

}

std::vector<std::string> expected_tensor_names(const Hparams & h) {
    std::vector<std::string> out;
    out.push_back("token_embd.weight");
    out.push_back("output_norm.weight");
    out.push_back("enc.conv2d1.weight");
    out.push_back("enc.conv2d1.bias");
    out.push_back("enc.conv2d2.weight");
    out.push_back("enc.conv2d2.bias");
    out.push_back("enc.conv2d3.weight");
    out.push_back("enc.conv2d3.bias");
    out.push_back("enc.conv_out.weight");
    out.push_back("enc.ln_post.weight");
    out.push_back("enc.ln_post.bias");
    out.push_back("enc.proj1.weight");
    out.push_back("enc.proj1.bias");
    out.push_back("enc.proj2.weight");
    out.push_back("enc.proj2.bias");
    emit_encoder_layers(h.enc_layers, out);
    emit_decoder_layers(h.text_layers, out);
    return out;
}

}
