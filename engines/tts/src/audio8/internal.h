#pragma once
// Audio8 engine internals: hyperparameters, weights, stage entry points.
//
// Audio8 is a DualAR model. A 24-layer "slow" transformer reads the ChatML
// prompt one frame at a time and emits a semantic token; a 4-layer "fast"
// transformer then expands that token into the remaining nine codebook values
// for the same frame. A separate DAC-style codec turns the ten codebooks into
// audio, and its encoder turns reference audio back into codes for cloning.
//
// The LM and the two codec halves are separate GGUFs, so a text-only build
// can skip the encoder.

#include "ggml.h"
#include "ggml-backend.h"
#include "sched_dispatch.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "audio8/tokenizer.h"

namespace tts_cpp {
namespace audio8 {
namespace detail {

constexpr int AUDIO8_MAX_NODES = 8192;

// Adds the wall time of its scope to `sink`, in milliseconds. Stages are timed
// by accumulation because most of them are entered once per generated frame.
class stage_timer {
public:
    explicit stage_timer(double & sink)
        : sink_(sink), start_(std::chrono::steady_clock::now()) {}

    ~stage_timer() {
        const std::chrono::duration<double, std::milli> elapsed =
            std::chrono::steady_clock::now() - start_;
        sink_ += elapsed.count();
    }

    stage_timer(const stage_timer &) = delete;
    stage_timer & operator=(const stage_timer &) = delete;

private:
    double & sink_;
    std::chrono::steady_clock::time_point start_;
};

struct lm_hparams {
    int depth = 0;
    int hidden = 0;
    int n_head = 0;
    int n_kv = 0;
    int head_dim = 0;
    int inter = 0;
    int vocab = 0;
    int fast_depth = 0;
    int fast_hidden = 0;
    int fast_n_head = 0;
    int fast_n_kv = 0;
    int fast_head_dim = 0;
    int fast_inter = 0;
    int num_codebooks = 0;
    int codebook_size = 0;
    int semantic_begin = 0;
    int semantic_end = 0;
    int eos = 0;
    int pad = 0;
    int max_seq_len = 0;
    int ras_window = 0;
    float rope_theta = 0.0f;
    float rms_eps = 0.0f;
    float ras_top_p = 0.0f;
    float ras_temperature = 0.0f;
    bool norm_fast_input = true;
    bool qkv_bias = true;
    bool fast_qkv_bias = false;
};

struct attention_weights {
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    ggml_tensor * wv = nullptr;
    ggml_tensor * wo = nullptr;
    ggml_tensor * wq_b = nullptr;
    ggml_tensor * wk_b = nullptr;
    ggml_tensor * wv_b = nullptr;
    ggml_tensor * attn_norm = nullptr;
};

struct block_weights {
    attention_weights attn;
    ggml_tensor * w1 = nullptr;
    ggml_tensor * w2 = nullptr;
    ggml_tensor * w3 = nullptr;
    ggml_tensor * ffn_norm = nullptr;
};

// One transformer's KV slab: [head_dim * n_kv, capacity] per layer, stacked
// layer-major in a single tensor so a step writes one view and reads one view.
struct kv_cache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    int capacity = 0;
    int stride = 0;
};

struct lm_model {
    lm_hparams hp;

    ggml_backend_t backend = nullptr;
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr;
    ::tts_cpp::detail::sched_fallback sched;

    ggml_tensor * tok_emb = nullptr;
    ggml_tensor * codebook_emb = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * sem_head = nullptr;
    ggml_tensor * rope_cos = nullptr;
    ggml_tensor * rope_sin = nullptr;
    std::vector<block_weights> blocks;

    ggml_tensor * fast_emb = nullptr;
    ggml_tensor * fast_norm = nullptr;
    ggml_tensor * fast_out = nullptr;
    ggml_tensor * fast_rope_cos = nullptr;
    ggml_tensor * fast_rope_sin = nullptr;
    std::vector<block_weights> fast_blocks;

    kv_cache slow_kv;
    kv_cache fast_kv;
    // Separate arenas: the two graphs have different shapes, and alternating
    // them through one allocator would resize it on every call.
    ggml_gallocr_t slow_allocr = nullptr;
    ggml_gallocr_t fast_allocr = nullptr;
    ggml_gallocr_t frame_allocr = nullptr;
    // Whether this backend can pick codes itself, decided once at load time.
    bool picks_codes = false;
    bool precise_outputs = false;

    TokenizerData tokenizer;
};

struct transformer_spec {
    int n_layer = 0;
    int n_head = 0;
    int n_kv = 0;
    int dim = 0;
    int inter = 0;
    int window = 0;
    int head_dim = 0;
    float rope_theta = 0.0f;
    float norm_eps = 0.0f;
};

// Pre-norm block with the learned residual scales the codec's window
// transformers use in place of Audio8's plain residual add.
struct scaled_block_weights {
    attention_weights attn;
    ggml_tensor * attn_scale = nullptr;
    ggml_tensor * w1 = nullptr;
    ggml_tensor * w2 = nullptr;
    ggml_tensor * w3 = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_scale = nullptr;
};

struct window_transformer {
    transformer_spec spec;
    std::vector<scaled_block_weights> blocks;
    ggml_tensor * norm = nullptr;
    ggml_tensor * rope_cos = nullptr;
    ggml_tensor * rope_sin = nullptr;
};

// A causal conv1d and its bias. Weight norm is already folded in, so the
// weight is the materialised kernel.
struct conv_weights {
    ggml_tensor * w = nullptr;
    ggml_tensor * b = nullptr;
};

struct convnext_weights {
    conv_weights dwconv;
    ggml_tensor * norm_w = nullptr;
    ggml_tensor * norm_b = nullptr;
    conv_weights pwconv1;
    conv_weights pwconv2;
    ggml_tensor * gamma = nullptr;
};

struct resample_stage {
    conv_weights conv;
    convnext_weights convnext;
    int stride = 0;
};

// Both halves carry the codebook, but only the encoder needs the projection
// into it and only the decoder the projection back out.
struct quantizer_weights {
    ggml_tensor * codebook = nullptr;
    conv_weights in_proj;
    conv_weights out_proj;
};

struct residual_unit {
    ggml_tensor * alpha1 = nullptr;
    conv_weights conv1;
    ggml_tensor * alpha2 = nullptr;
    conv_weights conv2;
};

// Encoder: three residual units, Snake, strided conv, and on the last stage a
// window transformer. Decoder: Snake, transposed conv, three residual units.
struct dac_stage {
    ggml_tensor * alpha = nullptr;
    conv_weights conv;
    int stride = 0;
    std::vector<residual_unit> units;
    window_transformer transformer;
    bool has_transformer = false;
};

struct codec_hparams {
    int sample_rate = 0;
    int frame_size = 0;
    int num_codebooks = 0;
    int latent_dim = 0;
    int codebook_dim = 0;
    int max_frames = 0;
    int semantic_codebook_size = 0;
    int residual_codebook_size = 0;
    int residual_codebooks = 0;
    float convnext_norm_eps = 0.0f;
    float snake_epsilon = 0.0f;
    std::vector<int> encoder_strides;
    std::vector<int> decoder_strides;
    std::vector<int> residual_dilations;
};

struct codec_model {
    codec_hparams hp;
    bool has_encoder = false;
    bool has_decoder = false;

    ggml_backend_t backend = nullptr;
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr;
    ::tts_cpp::detail::sched_fallback sched;
    bool precise_outputs = false;
    // Both directions run in two graphs: one over the whole sequence at one
    // column per frame, and one over blocks of the sample-rate stack. They get
    // an arena each, because alternating them through one would resize it on
    // every switch.
    ggml_gallocr_t allocr = nullptr;
    ggml_gallocr_t block_allocr = nullptr;

    std::vector<quantizer_weights> semantic_quantizers;
    std::vector<quantizer_weights> residual_quantizers;
    window_transformer post;
    std::vector<resample_stage> upsample;

    conv_weights dec_in;
    std::vector<dac_stage> dec_stages;
    ggml_tensor * dec_out_alpha = nullptr;
    conv_weights dec_out;

    // How much of the sample-rate stack each block covers: frames of output
    // when synthesising, columns of the encoder's 512-fold reduction when
    // analysing. Smaller blocks cost less scratch and repeat more work on each
    // block's context.
    //
    // Synthesis takes zero to mean the widest block whose scratch fits
    // synthesis_scratch_budget, which is the fastest width available at that
    // budget; a positive value pins it instead. A zero budget takes a share of
    // what the backend reports free.
    int synthesis_block_frames = 0;
    size_t synthesis_scratch_budget = 0;
    int analysis_block_columns = 128;

    conv_weights enc_in;
    std::vector<dac_stage> enc_stages;
    ggml_tensor * enc_out_alpha = nullptr;
    conv_weights enc_out;
    window_transformer pre;
    std::vector<resample_stage> downsample;
};

bool load_lm(const std::string & path, int n_gpu_layers, lm_model & model,
             std::string * error);
void free_lm(lm_model & model);

// What a codec GGUF says about itself: which half it holds and the shapes it
// was converted with.
struct codec_header {
    codec_hparams hp;
    std::string part;  // "encoder" or "decoder"
};

// Reads that header and nothing else, so a set of GGUFs can be checked for
// belonging together before any of the weights are pulled off disk.
bool peek_codec_header(const std::string & path, codec_header & header,
                       std::string * error);

bool load_codec(const std::string & path, int n_gpu_layers, codec_model & model,
                std::string * error);
void free_codec(codec_model & model);

// Prompt frames, laid out the way the model reads them: row 0 is the semantic
// or text token, rows 1..num_codebooks the reference codebooks (zero outside a
// cloning prompt). Column-major, one column per position.
struct prompt_frames {
    std::vector<int32_t> values;
    int width = 0;
    int rows = 0;
};

prompt_frames build_frames(const lm_hparams & hp, const PromptSegments & segments,
                           const std::vector<int32_t> & reference_codes, int reference_len);

// Runs the slow transformer over `width` frames ending at position `n_past`,
// returning the semantic logits for the last frame ([codebook_size + 1], the
// codebook rows followed by EOS) and the hidden state the fast head consumes.
// The codec analysis front end (raw audio through the causal conv stack),
// exposed for the same two-backend node comparison as the LM graph.
ggml_tensor * build_conv_stack(ggml_context * ctx, const codec_model & model,
                               int n_samples);

// The slow transformer's prefill/step graph, exposed so a harness can compute
// the identical graph on two backends and compare it node by node.
struct scratch;
struct slow_graph_outputs {
    ggml_tensor * logits  = nullptr;
    ggml_tensor * carried = nullptr;
};
void build_slow_graph(lm_model & model, scratch & build, int width, int n_past,
                      slow_graph_outputs & outs);
void set_slow_graph_inputs(const lm_model & model, ggml_cgraph * graph,
                           const int32_t * frames, int width, int n_past);

bool slow_step(lm_model & model, const int32_t * frames, int width, int n_past,
               int n_threads, std::vector<float> & sem_logits,
               std::vector<float> & fast_input, std::string * error);

int argmax_of(const std::vector<float> & values);

// Turns one position's fast-head logits into a codebook value.
using code_picker = std::function<int(const std::vector<float> & logits, int position)>;

// Expands one semantic token into num_codebooks values. codes_out is sized
// num_codebooks; entry 0 is derived from `semantic` without a forward pass.
bool fast_step(lm_model & model, const std::vector<float> & fast_input, int semantic,
               int n_threads, const code_picker & pick, std::vector<int32_t> & codes_out,
               std::string * error);

// fast_step's greedy equivalent in one graph instead of num_codebooks graphs,
// picking each code on the backend. One submission rather than ten: worth roughly
// a quarter of a millisecond per position on Metal, where a submission costs far
// more than the kernels it carries. Only call it when `model.picks_codes`.
//
// Picking on the device means ggml's argmax settles a tie between two equal
// logits differently from argmax_of, which keeps the first. That is why
// picks_codes excludes the CPU backend, whose per-position path is the reference
// for the tie-break and which has no submissions to collapse anyway.
bool fast_frame(lm_model & model, const std::vector<float> & fast_input, int semantic,
                int n_threads, std::vector<int32_t> & codes_out, std::string * error);

// Stage boundaries the parity test compares one at a time, so a divergence
// names the module that caused it instead of only showing a wrong waveform.
// Each is channels-inner: [latent_dim, frames], and [latent_dim, 4 * frames]
// for the upsampled latent.
struct decode_taps {
    std::vector<float> semantic;
    std::vector<float> residual;
    std::vector<float> post;
    std::vector<float> latent;
};

// Asked between blocks of the sample-rate stack, the one place a codec pass
// can stop: once ggml has a graph it runs to completion. Returning true ends
// the pass with CANCELLED as the error. An empty hook never cancels.
using cancel_hook = std::function<bool()>;

constexpr const char * CANCELLED = "audio8: synthesis cancelled";

// The decode pass runs as two graphs with very different shapes, so they are
// timed apart: the post transformer works one column per frame, the sample-rate
// stack expands each column into frame_size samples.
struct decode_timing {
    double latent_ms = 0.0;
    double synthesis_ms = 0.0;
    // The block width synthesis settled on and what the allocator priced it at,
    // which is the only view of a width chosen from a memory budget.
    int block_frames = 0;
    size_t block_scratch = 0;
};

// What synthesis_block_frames == 0 resolves the scratch budget to, given a
// configured budget and what the backend reports for the device. Separate from
// the query so the arithmetic can be checked against figures a real device will
// not produce on demand.
size_t synthesis_scratch_budget(size_t configured, size_t free_bytes, size_t total_bytes);

// codes: [num_codebooks, n_frames] row-major. Writes n_frames * frame_size
// samples at the codec sample rate.
bool decode_codes(codec_model & model, const int32_t * codes, int n_frames,
                  int n_threads, const cancel_hook & cancel,
                  std::vector<float> & pcm_out, std::string * error,
                  decode_taps * taps = nullptr, decode_timing * timing = nullptr);

// The convolutional encoder's output, [latent_dim, 4 * frames], and what the
// downsampling stages and the pre-module make of it, [latent_dim, frames].
struct encode_taps {
    std::vector<float> encoded;
    std::vector<float> downsampled;
};

// pcm: mono at the codec sample rate. Writes [num_codebooks, n_frames].
bool encode_audio(codec_model & model, const float * pcm, int n_samples,
                  int n_threads, const cancel_hook & cancel,
                  std::vector<int32_t> & codes_out, int & n_frames,
                  std::string * error, encode_taps * taps = nullptr);

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
