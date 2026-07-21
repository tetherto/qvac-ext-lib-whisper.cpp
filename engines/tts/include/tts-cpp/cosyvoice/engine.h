#pragma once

// Persistent CosyVoice3 engine (Fun-CosyVoice3-0.5B / 1.5B).
//
// STATUS — ITERATION 1 (SCAFFOLD):
//   This is the wiring skeleton for the CosyVoice3 sibling engine.  The
//   public API, option plumbing, streaming contract and backend reporting
//   are in place and match the Chatterbox / Supertonic engines so the JS
//   addon + @qvac/sdk flow works end-to-end.  The real CPU inference graphs
//   (Qwen2 LM -> supervised S3 speech tokenizer -> DiT flow-matching -> HiFT
//   vocoder) are NOT implemented yet: synthesize() returns a clearly-marked
//   placeholder waveform (a short marker tone + silence).  Do not mistake
//   its output for real speech.  See PROGRESS_COSYVOICE.md for the staged
//   bring-up plan.
//
// Pipeline (once implemented, CPU path):
//   text --(wetext frontend)--> Qwen2.5 LM --> speech tokens [0, 6561)
//        --> DiT conditional-flow-matching (Euler ODE) --> mel
//        --> CausalHiFT vocoder --> 24 kHz mono PCM
//   Zero-shot voice cloning adds: reference audio --> S3 tokenizer (prompt
//   tokens) + CAM++ (192-d speaker embedding).
//
// CosyVoice3 ships as a small set of GGUFs (llm / flow / hift / s3tok /
// campplus / voices).  Point `model_dir` at the folder that holds them, or
// set the per-component paths explicitly.
//
// Usage:
//
//     using tts_cpp::cosyvoice::Engine;
//     using tts_cpp::cosyvoice::EngineOptions;
//
//     EngineOptions opts;
//     opts.model_dir     = "models/cosyvoice3-0.5b";
//     opts.n_gpu_layers  = 0;                 // 0 = CPU (iteration 1 is CPU-only)
//
//     Engine engine(opts);
//     auto result = engine.synthesize("Hello world.");
//     write_wav(result.pcm, result.sample_rate);   // 24 kHz mono float32
//
// Threading model (mirrors the sibling engines):
//   - synthesize() on the same Engine instance is NOT safe to call
//     concurrently.
//   - synthesize() on different Engine instances from different threads is
//     safe.
//   - cancel() is safe from any thread.
//
// Implemented in src/cosyvoice_engine.cpp.

#include "tts-cpp/backend.h"
#include "tts-cpp/export.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::cosyvoice {

struct EngineOptions {
    // Directory holding the standard CosyVoice3 GGUFs
    // (cosyvoice3-{llm,flow,hift,s3tok,campplus,voices}-*.gguf).  Either set
    // this, or set the per-component paths below explicitly (explicit paths
    // win over the ones discovered under model_dir).
    std::string model_dir;

    std::string llm_gguf_path;       // Qwen2.5 LM (text -> speech tokens)
    std::string flow_gguf_path;      // DiT conditional-flow-matching (tokens -> mel)
    std::string hift_gguf_path;      // CausalHiFT vocoder (mel -> 24 kHz PCM)
    std::string s3tok_gguf_path;     // supervised S3 speech tokenizer (zero-shot prompt)
    std::string campplus_gguf_path;  // CAM++ speaker encoder (zero-shot)

    // Qwen2 byte-level BPE text frontend (vocab.json + merges.txt) and the
    // baked default-voice GGUF (voice/prompt_stok, prompt_token, prompt_feat,
    // embedding — see scripts/bake-cosyvoice3-voice.py).  Resolved under
    // model_dir when left empty (vocab.json / merges.txt / voice.gguf).
    std::string vocab_path;
    std::string merges_path;
    std::string voice_gguf_path;

    // ---- Zero-shot voice cloning (optional) ----
    // Reference audio (a few seconds of clean speech) + its transcript.
    // CosyVoice3 conditions the LM on the prompt transcript and the flow /
    // vocoder on the prompt speech tokens + speaker embedding.
    std::string reference_audio;
    std::string prompt_text;

    // Instruct mode (CosyVoice3 instruct2): a natural-language instruction that
    // controls dialect / accent / emotion / speed / volume, e.g.
    // "请用广东话表达。" (speak in Cantonese) or "speak slowly and cheerfully".
    // When non-empty the LM is conditioned on this instruction and drops its
    // prompt speech tokens; the baked/selected voice still supplies the timbre.
    // Empty (default) = zero-shot (prompt_text / baked-voice transcript path).
    std::string instruct_text;
    // Or select a voice baked into voices.gguf (mutually exclusive with the
    // reference-audio path; reference audio wins when both are set).
    std::string voice;

    // Language hint (CosyVoice3 is multilingual; used by the text frontend).
    std::string language = "en";

    int seed         = 42;
    int n_threads    = 0;   // 0 = library default (hardware concurrency)
    int n_gpu_layers = 0;   // 0 = CPU.  Iteration 1 is CPU-only; >0 reserved.

    // Desired output sample rate in Hz.  CosyVoice3 is natively 24 kHz; a
    // positive value other than 24000 resamples the final PCM and is reported
    // on SynthesisResult::sample_rate.  0 keeps the native rate.
    int output_sample_rate = 0;

    // Flow-matching Euler step count.  0 = model default (CosyVoice3 uses 10).
    int cfm_steps = 0;

    // ---------------- Streaming synthesis ----------------------------
    //
    // CosyVoice3 is natively a chunked / token-by-token streaming model: the
    // Qwen2 LM emits speech tokens autoregressively and token2wav consumes
    // them in hops (the DiT flow + CausalHiFT are causal), stitching chunks
    // with mel overlap.  When `stream_chunk_tokens > 0` AND a non-empty
    // callback is passed to synthesize(), the engine runs that chunked loop
    // and invokes the callback with each chunk's 24 kHz PCM as it is produced.
    // The returned SynthesisResult.pcm still contains the concatenated audio
    // (the callback is an *addition*, not a replacement), so the documented
    // `result.pcm == concat(callback chunks)` invariant holds.  Streaming is
    // disabled (batch path) when stream_chunk_tokens == 0 OR the callback is
    // empty.
    //
    //   stream_chunk_tokens        Speech tokens per token2wav hop.
    //                              0 = non-streaming (batch).
    //   stream_first_chunk_tokens  Override for the *first* chunk so first
    //                              audio lands early; 0 = same as
    //                              stream_chunk_tokens.
    //   stream_left_context_tokens Left context carried into each chunk to
    //                              bound per-chunk cost (avoids O(N^2)).
    int stream_chunk_tokens        = 0;
    int stream_first_chunk_tokens  = 0;
    int stream_left_context_tokens = 0;

    // Directory to scan for dynamically-loaded ggml backends.  Forwarded to
    // ggml_backend_load_all_from_path() on first construction; empty falls
    // back to ggml's default search path.  (No-op on the iteration-1 stub
    // CPU path, but plumbed so the option surface is stable.)
    std::string backends_dir;
};

// Per-chunk PCM callback for streaming synthesis.  Receives `samples`
// consecutive float32 mono samples at SynthesisResult::sample_rate (24 kHz by
// default).  The buffer is owned by the engine and must not be retained past
// the callback; copy out if you need the data.
//   `chunk_index`  0-based; increments by exactly 1 per chunk.
//   `is_last`      true only on the final chunk.
// Throwing from this callback aborts synthesis (the exception propagates out
// of synthesize()).
using StreamCallback = std::function<void(
    const float * pcm, std::size_t samples, int chunk_index, bool is_last)>;

struct SynthesisResult {
    std::vector<float> pcm;
    int   sample_rate = 24000;
    float duration_s  = 0.0f;
};

// Persistent engine.  Loads the model once at construction; subsequent
// synthesize() calls reuse the resident model.
class TTS_CPP_API Engine {
public:
    // Validates options and prepares the engine.  Throws std::runtime_error
    // on a hard failure (e.g. model_dir set but missing).  In the
    // iteration-1 scaffold the heavy GGUF weights are not required to exist:
    // a warning is logged and synthesize() returns placeholder audio so the
    // end-to-end SDK flow is exercisable before the converters land.
    explicit Engine(const EngineOptions & opts);
    ~Engine();

    Engine(const Engine &)             = delete;
    Engine & operator=(const Engine &) = delete;
    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;

    // Synthesize `text` into PCM (24 kHz mono float32 by default).  Throws
    // std::runtime_error on failure.  Empty `text` is rejected.
    // Not safe to call concurrently on the same Engine instance.
    SynthesisResult synthesize(const std::string & text);

    // Streaming variant — see EngineOptions streaming block.  Falls through
    // to the batch path when streaming is disabled.
    SynthesisResult synthesize(const std::string & text,
                               const StreamCallback & on_chunk);

    // Best-effort cancel of an in-flight synthesize() on another thread.
    void cancel();

    const EngineOptions & options() const;

    // Registered name of the resolved backend ("CPU", "Metal", ...).
    std::string backend_name() const;

    // Resolved compute device.  CPU on the iteration-1 stub.
    BackendDevice backend_device() const;

    // True when a GPU device was present but unusable (fell back to CPU).
    bool gpu_unsupported() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Convenience one-shot wrapper.  Equivalent to:
//   Engine e(opts); return e.synthesize(text);
TTS_CPP_API SynthesisResult synthesize(const EngineOptions & opts, const std::string & text);

} // namespace tts_cpp::cosyvoice
