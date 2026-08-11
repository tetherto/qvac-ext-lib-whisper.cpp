#include "audiogen-cpp/acestep/engine.h"

#include "audiogen-cpp/acestep/vae.h"

#include "acestep/bpe_tokenizer.h"
#include "acestep/cond_ggml.h"
#include "acestep/detok_ggml.h"
#include "acestep/dit_ggml.h"
#include "acestep/dit_gguf.h"  // DitGGUF: read DiT config + validate GGUF headers at create()
#include "acestep/lm_ggml.h"
#include "acestep/lm_pipeline.h"
#include "acestep/philox.h"
#include "acestep/textenc_ggml.h"

#include "acestep/backend_registry.h"
#include "acestep/cover_noise.h"
#include "acestep/generate_task.h"
#include "acestep/generation_conditioning.h"
#include "acestep/generation_plan.h"
#include "acestep/stage_placement.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>

// ACE-Step end-to-end music engine. Wires the ported stages behind
// tts_cpp::acestep::Engine::generate():
//   caption+lyrics -> LM (audio codes) -> FSQ detok (context latents)
//                  -> text-enc + cond-enc (cross-attn states)
//                  -> DiT flow-matching sample -> VAE decode -> stereo 48 kHz.
//
// Turbo text2music path: single sequence, no CFG, Phase-2 codes only (Phase-1
// CoT/metadata generation + metadata FSM are a follow-up). Matches acestep.cpp's
// synth wiring: because LM audio codes are present, the DiT uses the COVER
// instruction and detokenized codes as the conditioning context.

namespace tts_cpp::acestep {

// --- synth-parity debug hooks (compiled out by default) -----------
// Isolate DiT vs VAE by injecting acestep.cpp --dump tensors and dumping ours.
// Dump .bin format: 3x int32 header [ndim, d0, d1] + float32 payload.
// Build with -DACESTEP_PARITY_DEBUG to enable; otherwise these are absent from
// the production generate() path.
#ifdef ACESTEP_PARITY_DEBUG
namespace {
bool dbg_load_dump(const char * path, std::vector<float> & dst) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[acestep-dbg] cannot open %s\n", path); return false; }
    int32_t hdr[3] = {0, 0, 0};
    if (fread(hdr, sizeof(int32_t), 3, f) != 3) { fclose(f); return false; }
    const int    ndim = hdr[0];
    const size_t n    = (size_t) hdr[1] * (size_t) ((ndim >= 2) ? hdr[2] : 1);
    if (n != dst.size()) {
        fprintf(stderr, "[acestep-dbg] size mismatch %s: dump=%zu ours=%zu\n", path, n, dst.size());
        fclose(f);
        return false;
    }
    const size_t got = fread(dst.data(), sizeof(float), n, f);
    fclose(f);
    fprintf(stderr, "[acestep-dbg] injected %s (%zu floats)\n", path, got);
    return got == n;
}

void dbg_write_dump(const std::string & path, const std::vector<float> & src, int d0, int d1) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "[acestep-dbg] cannot write %s\n", path.c_str()); return; }
    int32_t hdr[3] = {2, d0, d1};
    fwrite(hdr, sizeof(int32_t), 3, f);
    fwrite(src.data(), sizeof(float), src.size(), f);
    fclose(f);
    fprintf(stderr, "[acestep-dbg] wrote %s ([%d, %d], %zu floats)\n", path.c_str(), d0, d1, src.size());
}

const char * dbg_env(const char * k) {
    const char * v = getenv(k);
    return (v && *v) ? v : nullptr;
}
}  // namespace
#endif  // ACESTEP_PARITY_DEBUG

// DiT instruction headers (task-types.h). COVER is used whenever LM codes exist.
static const char * DIT_INSTR_COVER = "Generate audio semantic tokens based on the given conditions:";

namespace fs = std::filesystem;

struct Engine::Impl {
    EngineOptions opts;

    ggml_backend_t backend       = nullptr;  // primary backend (GPU or CPU) for textenc/cond/DiT
    ggml_backend_t backend_cpu   = nullptr;  // CPU backend for the stages pinned off the GPU; == backend when primary is CPU
    ggml_backend_t backend_lm    = nullptr;  // backend the LM loads on (see create)
    ggml_backend_t backend_enc   = nullptr;  // backend for textenc + cond (see create)
    ggml_backend_t backend_detok = nullptr;  // backend the FSQ detok loads on (CPU off-GPU, GPU on Vulkan; see create)
    TextEncModel * textenc = nullptr;
    LMModel *      lm      = nullptr;
    CondModel *    cond    = nullptr;
    DetokModel *   detok   = nullptr;
    DitModel *     dit     = nullptr;

    std::unique_ptr<Vae> vae;

    BpeTokenizer bpe_lm;    // LM vocab (+ audio codes) — Phase-2 prompt
    BpeTokenizer bpe_text;  // text-encoder vocab — DiT prompt / lyric lookup

    mutable std::atomic<bool> cancel_flag{ false };

    // Sequential-loading state (see create()/generate()). In the default low-mem
    // mode no stage is resident after create(); generate() loads each stage right
    // before its step and frees it right after, so the peak resident set is one
    // stage — not all six at once. dit_cfg is read from the DiT GGUF metadata at
    // create() so the context-build step needs no resident DiT; sr lets
    // sample_rate() answer before the VAE is loaded.
    DitConfig  dit_cfg{};
    int        sr          = 48000;
    int        nth         = 4;
    bool       keep_stages = false;  // ACESTEP_KEEP_STAGES: eager-load + never free
    VaeOptions vae_opts{};           // saved so the VAE can be (re)loaded lazily

    ~Impl() {
        free_dit();
        free_detok();
        free_cond();
        free_lm();
        free_textenc();
        free_vae();
        if (backend_cpu && backend_cpu != backend) ggml_backend_free(backend_cpu);
        if (backend) ggml_backend_free(backend);
    }

    // --- per-stage lazy load / free -----------------------------------------
    // ensure_*() loads a stage if it is not already resident (idempotent, so it
    // serves both the eager keep-stages path and the lazy per-generate reload);
    // free_*() releases it. All throw on a genuine load failure.
    void ensure_textenc() {
        if (textenc) return;
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading text-encoder\n");
        textenc = textenc_model_load(opts.text_enc_model_path, backend_enc, opts.verbose);
        if (!textenc) throw std::runtime_error("acestep engine: text-encoder load failed");
    }
    void ensure_lm() {
        if (lm) return;
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading LM\n");
        lm = lm_model_load(opts.lm_model_path, backend_lm, /*max_seq_len=*/2048, opts.verbose, /*n_kv_sets=*/2);
        if (!lm) throw std::runtime_error("acestep engine: LM load failed");
    }
    void ensure_cond() {
        if (cond) return;
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading cond-encoder\n");
        cond = cond_model_load(opts.dit_model_path, backend_enc, opts.verbose);
        if (!cond) throw std::runtime_error("acestep engine: cond-encoder load failed");
    }
    void ensure_detok() {
        if (detok) return;
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading FSQ detokenizer\n");
        detok = detok_model_load(opts.dit_model_path, backend_detok, opts.verbose);
        if (!detok) throw std::runtime_error("acestep engine: FSQ detokenizer load failed");
    }
    void ensure_dit() {
        if (dit) return;
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading DiT\n");
        dit = dit_model_load(opts.dit_model_path, backend, opts.verbose);
        if (!dit) throw std::runtime_error("acestep engine: DiT load failed");
    }
    void ensure_vae(bool with_encoder = false) {
        if (vae && (!with_encoder || vae->has_encoder())) return;
        if (vae) free_vae();
        if (opts.verbose) fprintf(stderr, "[acestep-engine] loading VAE\n");
        VaeOptions load_opts = vae_opts;
        load_opts.with_encoder = with_encoder;
        vae = Vae::load(opts.vae_model_path, load_opts);
        if (!vae) throw std::runtime_error("acestep engine: VAE load failed");
        sr = vae->sample_rate();
    }
    void free_textenc() { if (textenc) { textenc_model_free(textenc); textenc = nullptr; } }
    void free_lm()      { if (lm)      { lm_model_free(lm);           lm      = nullptr; } }
    void free_cond()    { if (cond)    { cond_model_free(cond);       cond    = nullptr; } }
    void free_detok()   { if (detok)   { detok_model_free(detok);     detok   = nullptr; } }
    void free_dit()     { if (dit)     { dit_model_free(dit);         dit     = nullptr; } }
    void free_vae()     { vae.reset(); }
};

// ------------------------------------------------------------ path resolution
static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

// Classify the four ACE-Step GGUFs in models_dir by anchoring on their known
// filename stems (Qwen3-Embedding / acestep-*-lm / vae / acestep-v15-{turbo,sft}).
// Explicit paths in EngineOptions always win over the scan. The stems are chosen
// so no bare short token (like "lm") can match an unrelated file; the most
// specific stages (embedding, vae) are tested before the shorter lm/dit stems.
static void resolve_paths(EngineOptions & o) {
    if (o.models_dir.empty()) return;
    std::error_code ec;
    for (auto & e : fs::directory_iterator(o.models_dir, ec)) {
        if (!e.is_regular_file()) continue;
        std::string path = e.path().string();
        std::string name = to_lower(e.path().filename().string());
        if (name.size() < 5 || name.substr(name.size() - 5) != ".gguf") continue;
        auto has = [&](const char * s) { return name.find(s) != std::string::npos; };
        if (has("embedding") || has("text-enc") || has("textenc")) {
            if (o.text_enc_model_path.empty()) o.text_enc_model_path = path;
        } else if (has("vae")) {
            if (o.vae_model_path.empty()) o.vae_model_path = path;
        } else if (has("-lm") || has("lm-") || has("_lm") || has("ace-lm") || has("5hz-lm")) {
            if (o.lm_model_path.empty()) o.lm_model_path = path;
        } else if (has("turbo") || has("dit") || has("v15") || has("sft")) {
            if (o.dit_model_path.empty()) o.dit_model_path = path;
        }
    }
}

// ------------------------------------------------------------ construction
Engine::Engine() : impl_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::create(const EngineOptions & opts_in) {
    EngineOptions opts = opts_in;
    resolve_paths(opts);

    auto need = [&](const std::string & p, const char * what) {
        if (p.empty()) throw std::runtime_error(std::string("acestep engine: missing ") + what + " GGUF");
    };
    need(opts.text_enc_model_path, "text-encoder");
    need(opts.lm_model_path, "LM");
    need(opts.dit_model_path, "DiT");
    need(opts.vae_model_path, "VAE");

    std::unique_ptr<Engine> eng(new Engine());
    Impl *                  m = eng->impl_.get();
    m->opts                   = opts;

    const bool v = opts.verbose;

    // Load the dlopen'd ggml backend modules (per-microarch CPU variants on
    // arm64, plus any GPU MODULE .so) the addon staged next to its `.bare`, so
    // the registry-based backend init below can find a CPU/GPU device. No-op on
    // static-linked desktop / Apple builds. Must run before any backend init.
    load_backends(opts.backends_dir);

    int nth = opts.n_threads > 0 ? opts.n_threads : (int) std::thread::hardware_concurrency();
    if (nth < 1) nth = 4;

    // Backend for the ggml stages (text-encoder, LM, cond/detok, DiT). These use
    // standard ggml ops, so Metal, CUDA, Vulkan, and validated Adreno 700+
    // OpenCL can run them; opts.n_gpu_layers > 0 opts in. The VAE gets its own
    // dedicated backend (see Vae::load) and also follows n_gpu_layers now that
    // its snake / col2im_1d ops are validated on Metal, Vulkan, and Adreno
    // OpenCL in the ggml-speech fork.
    // Falls back to CPU when no GPU backend is registered/available.
    bool on_gpu = false;
    if (opts.n_gpu_layers > 0) {
        m->backend = backend_gpu_init();
        on_gpu     = (m->backend != nullptr);
        if (!on_gpu && v) fprintf(stderr, "[acestep-engine] GPU requested but no GPU backend available; using CPU\n");
    }
    if (!m->backend) m->backend = backend_cpu_init();
    if (!m->backend) throw std::runtime_error("acestep engine: backend init failed");
    if (on_gpu) {
        if (v) fprintf(stderr, "[acestep-engine] DiT/VAE on GPU backend: %s\n", ggml_backend_name(m->backend));
        // Dedicated CPU backend for whichever stages are pinned off the GPU below.
        m->backend_cpu = backend_cpu_init();
        if (!m->backend_cpu) throw std::runtime_error("acestep engine: CPU backend init failed");
        backend_set_n_threads(m->backend_cpu, nth);
    } else {
        backend_set_n_threads(m->backend, nth);
        m->backend_cpu = m->backend;  // single CPU backend serves every stage
    }

    // Stage placement when a GPU is active. The DiT, the VAE and the one-shot
    // text/cond encoders always run on it. The autoregressive LM and the FSQ
    // detokenizer are allowlisted per backend, so a backend nobody has measured
    // keeps the CPU placement and cannot silently regress generated audio.
    // ACESTEP_LM_GPU / ACESTEP_DETOK_GPU take that measurement without a rebuild.
    // The policy itself lives in stage_placement.h so it can be unit tested
    // without a GPU (test/test_acestep_units.cpp); this only applies its answer.
    //
    // Env escape hatches (applied after the allowlist; CPU wins if both are set):
    //   ACESTEP_LM_GPU=1        -> LM on the GPU, whatever the backend
    //   ACESTEP_LM_CPU=1        -> LM on the CPU, whatever the backend
    //   ACESTEP_DETOK_GPU=1     -> detokenizer on the GPU, whatever the backend
    //   ACESTEP_DETOK_CPU=1     -> detokenizer on the CPU, whatever the backend
    //   ACESTEP_ENCODERS_CPU=1  -> move the encoders to the CPU (trim wired mem)
    ggml_backend_t enc_backend   = m->backend;
    ggml_backend_t lm_backend    = m->backend;
    ggml_backend_t detok_backend = m->backend;
    if (on_gpu) {
        const StagePlacement place =
            resolve_stage_placement(backend_reg_name(m->backend), placement_overrides_from_env());
        if (!place.enc_on_gpu)   enc_backend   = m->backend_cpu;
        if (!place.lm_on_gpu)    lm_backend    = m->backend_cpu;
        if (!place.detok_on_gpu) detok_backend = m->backend_cpu;
        if (v) fprintf(stderr, "[acestep-engine] backends: enc=%s lm=%s detok=%s dit/vae=%s\n",
                       ggml_backend_name(enc_backend), ggml_backend_name(lm_backend),
                       ggml_backend_name(detok_backend), ggml_backend_name(m->backend));
    }
    m->backend_enc   = enc_backend;
    m->backend_lm    = lm_backend;
    m->backend_detok = detok_backend;  // GPU on the allowlisted backends, CPU otherwise
    m->nth           = nth;

    // ACE-Step loads six weight sets (text-enc, LM, cond, detok, DiT, VAE). Held
    // resident at once they peak well past a non-entitled iOS app's memory ceiling
    // and the OS jetsam-kills the process mid-load. So by default the engine loads
    // each stage lazily inside generate() — right before its step, freed right
    // after — bounding the peak to a single stage (QVAC-22955). Opt out with
    // ACESTEP_KEEP_STAGES=1 (servers that generate back-to-back and prefer to pay
    // the load cost once and keep everything resident).
    m->keep_stages = [] {
        const char * e = std::getenv("ACESTEP_KEEP_STAGES");
        return e && (e[0] == '1' || e[0] == 't' || e[0] == 'T' || e[0] == 'y' || e[0] == 'Y');
    }();

    // VAE load options, saved so generate() can (re)load the VAE lazily. Decode
    // only (no encoder). ACESTEP_VAE_GPU forces the VAE backend independently of
    // the other stages so a decode can be compared CPU-vs-GPU on an identical
    // latent (=1 -> GPU, =0 -> CPU); leaves the LM/DiT backend untouched.
    VaeOptions vo;
    vo.verbose      = v;
    vo.with_encoder = false;
    vo.n_threads    = nth;
    vo.n_gpu_layers = opts.n_gpu_layers;  // validated on Metal, Vulkan, and Adreno OpenCL
    if (const char * e = std::getenv("ACESTEP_VAE_GPU")) {
        vo.n_gpu_layers = (e[0] == '1') ? 99 : 0;
    }
    m->vae_opts = vo;

    // Read the DiT config from GGUF metadata up front so the context-build step in
    // generate() needs no resident DiT (the full DiT is loaded lazily just before
    // the diffusion step). This also validates the DiT GGUF header at create().
    {
        DitGGUF g;
        if (!dit_gguf_open(g, opts.dit_model_path))
            throw std::runtime_error("acestep engine: cannot open DiT GGUF: " + opts.dit_model_path);
        const bool cfg_ok = dit_gguf_read_config(g, m->dit_cfg);
        dit_gguf_close(g);
        if (!cfg_ok) throw std::runtime_error("acestep engine: bad DiT config in " + opts.dit_model_path);
    }

    // Tokenizers: LM prompt uses the LM vocab; DiT text prompt + lyric lookup use
    // the text-encoder vocab. Fall back to the LM tokenizer if the text-encoder
    // GGUF has no tokenizer KV (same Qwen text vocab in the shared range). Loading
    // them here also validates the LM + text-encoder GGUF headers at create().
    if (!bpe_load_from_gguf(m->bpe_lm, opts.lm_model_path))
        throw std::runtime_error("acestep engine: LM tokenizer load failed");
    if (!bpe_load_from_gguf(m->bpe_text, opts.text_enc_model_path)) {
        if (v) fprintf(stderr, "[acestep-engine] text-encoder has no tokenizer KV; reusing LM tokenizer\n");
        m->bpe_text = m->bpe_lm;
    }

    // Validate the VAE GGUF header too (a no_alloc metadata parse), so create()
    // still fails fast on a missing/corrupt file instead of only at first
    // generate(). The other three GGUFs were validated above.
    {
        DitGGUF g;
        if (!dit_gguf_open(g, opts.vae_model_path))
            throw std::runtime_error("acestep engine: cannot open VAE GGUF: " + opts.vae_model_path);
        dit_gguf_close(g);
    }

    // ACESTEP_KEEP_STAGES: eager-load every stage now and keep it resident (the
    // pre-QVAC-22955 behaviour). Default (lazy) leaves all stages unloaded here;
    // generate() loads/frees them per step.
    if (m->keep_stages) {
        m->ensure_textenc();
        m->ensure_lm();
        m->ensure_cond();
        m->ensure_detok();
        m->ensure_dit();
        m->ensure_vae();
    }

    if (v) fprintf(stderr, "[acestep-engine] ready (threads=%d, %s)\n", nth,
                   m->keep_stages ? "stages resident" : "stages lazy/low-mem");
    return eng;
}

// ------------------------------------------------------------ stage timing
// The ProgressFn only carries cumulative wall-clock at stage boundaries, which
// leaves detok + text-encoder + cond-encoder inside a single unmeasured gap.
// Recording each stage explicitly is what lets the "speed up each stage" vs
// "overlap the stages" question be answered with measurements. Printed when
// EngineOptions::verbose is set.
namespace {
class StageTimes {
public:
    void mark(const char * name) {
        const auto now = std::chrono::steady_clock::now();
        entries_.emplace_back(name, std::chrono::duration<double, std::milli>(now - last_).count());
        last_ = now;
    }

    void dump(FILE * f) const {
        double total = 0.0;
        for (const auto & e : entries_) total += e.second;
        fprintf(f, "[acestep-timing] per-stage wall clock (total %.0f ms)\n", total);
        for (const auto & e : entries_)
            fprintf(f, "[acestep-timing]   %-12s %9.1f ms  %5.1f%%\n", e.first, e.second,
                    total > 0.0 ? 100.0 * e.second / total : 0.0);
    }

private:
    std::chrono::steady_clock::time_point       last_ = std::chrono::steady_clock::now();
    std::vector<std::pair<const char *, double>> entries_;
};

// Per-stage tensor dumps for cross-backend parity work. Inactive (and free)
// unless EngineOptions::dump_stages_dir is set, so this lives in the normal
// build instead of behind a compile flag: localising a backend divergence
// should not require rebuilding the engine.
class StageDump {
public:
    explicit StageDump(std::string dir, bool verbose) : dir_(std::move(dir)), verbose_(verbose) {}

    bool active() const { return !dir_.empty(); }

    // d0 = row count (slow axis), d1 = row length (fastest-varying axis), which
    // is the convention dbg_write_dump and acestep.cpp's --dump already use.
    void write(const char * name, const float * data, size_t n, int d0, int d1) const {
        if (!active()) return;
        const std::string path = dir_ + "/" + name + ".bin";
        FILE *            f    = fopen(path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "[acestep-dump] cannot write %s\n", path.c_str());
            return;
        }
        const int32_t hdr[3] = {2, d0, d1};
        fwrite(hdr, sizeof(int32_t), 3, f);
        fwrite(data, sizeof(float), n, f);
        fclose(f);
        if (verbose_) fprintf(stderr, "[acestep-dump] %s [%d, %d] %zu floats\n", path.c_str(), d0, d1, n);
    }

    void write(const char * name, const std::vector<float> & v, int d0, int d1) const {
        write(name, v.data(), v.size(), d0, d1);
    }

    // Integer stages (LM codes) are stored as float32 so one reader handles every
    // dump; code ids are well inside the exactly-representable range.
    void write_ints(const char * name, const std::vector<int> & v) const {
        if (!active()) return;
        std::vector<float> f(v.begin(), v.end());
        write(name, f, (int) f.size(), 1);
    }

private:
    std::string dir_;
    bool        verbose_ = false;
};
}  // namespace

// ------------------------------------------------------------ generate
static std::string build_metas(int bpm, const std::string & timesig, const std::string & keyscale, float dur) {
    char bpm_b[16] = "N/A";
    if (bpm > 0) snprintf(bpm_b, sizeof(bpm_b), "%d", bpm);
    const char * ts = timesig.empty() ? "N/A" : timesig.c_str();
    const char * ks = keyscale.empty() ? "N/A" : keyscale.c_str();
    char         buf[512];
    snprintf(buf, sizeof(buf), "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n", bpm_b, ts,
             ks, (int) dur);
    return buf;
}

static constexpr int AUDIO_CHANNELS         = 2;
static constexpr int AUDIO_SAMPLE_RATE      = 48000;
static constexpr int AUDIO_LATENT_RATE      = 25;
static constexpr int AUDIO_LATENT_CHANNELS  = 64;
static constexpr int AUDIO_CODE_FRAME_RATIO = 5;
static constexpr int TURBO_STEPS            = 8;
static constexpr int STANDARD_STEPS         = 50;
static constexpr float TURBO_SHIFT          = 3.0f;
static constexpr float STANDARD_SHIFT       = 1.0f;

using StageReporter = std::function<bool(const char *, int, int)>;

static long long resolve_seed(long long seed) {
    if (seed >= 0) return seed;
    std::random_device random;
    return (long long) random();
}

static AcePrompt make_prompt(const GenerateParams & params, const std::string & language) {
    AcePrompt prompt;
    prompt.caption        = params.caption;
    prompt.lyrics         = params.lyrics.empty() ? "[Instrumental]" : params.lyrics;
    prompt.duration       = params.duration;
    prompt.bpm            = params.bpm;
    prompt.keyscale       = params.keyscale;
    prompt.timesignature  = params.timesignature;
    prompt.vocal_language = language;
    return prompt;
}

static bool encode_audio(Vae & vae, const std::vector<float> & pcm, const char * stage,
                         const char * dump_name, const StageReporter & report, bool verbose,
                         StageDump & dump, StageTimes & timing, EncodedAudio & output) {
    if (!report(stage, 0, 1)) return false;

    const int pcm_frames = (int) (pcm.size() / AUDIO_CHANNELS);
    output.latent = vae.encode(pcm, pcm_frames, &output.frames);
    if (output.latent.empty() || output.frames <= 0) {
        throw std::runtime_error(std::string("acestep engine: ") + stage + " audio VAE encode failed");
    }

    if (verbose) {
        fprintf(stderr, "[acestep-engine] %s audio: %.2fs -> %d latent frames\n",
                stage, (float) pcm_frames / AUDIO_SAMPLE_RATE, output.frames);
    }
    dump.write(dump_name, output.latent, output.frames, AUDIO_LATENT_CHANNELS);
    timing.mark(stage);
    return report(stage, 1, 1);
}

static const char * resolve_conditioning_dump_name(const char * stage) {
    return strcmp(stage, "source") == 0 ? "00_source_latent" : "00_reference_latent";
}

static void fill_dit_context_mask(float * frame, int output_channels) {
    for (int channel = 0; channel < output_channels; ++channel) {
        frame[output_channels + channel] = 1.0f;
    }
}

static void fill_dit_context_frames(std::vector<float> & context, const std::vector<float> & latent,
                                    const float * silence, int frames, int latent_frames,
                                    int context_channels, int output_channels) {
    for (int frame_index = 0; frame_index < frames; ++frame_index) {
        float * frame = context.data() + (size_t) frame_index * context_channels;
        if (frame_index < latent_frames) {
            memcpy(frame, latent.data() + (size_t) frame_index * output_channels,
                   (size_t) output_channels * sizeof(float));
        } else if (silence) {
            memcpy(frame, silence, (size_t) output_channels * sizeof(float));
        }
        fill_dit_context_mask(frame, output_channels);
    }
}

static std::vector<float> make_dit_context(const std::vector<float> & latent,
                                           const std::vector<float> & silence, int frames,
                                           int latent_frames, int context_channels,
                                           int output_channels) {
    std::vector<float> context((size_t) context_channels * frames, 0.0f);
    fill_dit_context_frames(context, latent, silence.empty() ? nullptr : silence.data(),
                            frames, latent_frames, context_channels, output_channels);
    return context;
}

struct GenerationState {
    GenerateTask task;
    GenerationPlan plan;
    long long seed = 0;
    std::string language;
    AcePrompt prompt;
    GenerationConditioning conditioning;
    std::vector<float> context_latents;
    int code_frames = 0;
    int latent_frames = 0;
    bool low_memory = false;
};

struct PromptEncoding {
    std::vector<float> text_hidden;
    std::vector<float> lyric_embedding;
    int text_tokens = 0;
    int lyric_tokens = 0;
};

struct EncoderConditioning {
    std::vector<float> context;
    std::vector<float> hidden;
    int frames = 0;
    int context_channels = 0;
    int sequence = 0;
    int hidden_size = 0;
};

struct NoiseSchedule {
    std::vector<float> noise;
    std::vector<float> schedule;
    int steps = 0;
    float shift = 0.0f;
};

static GenerationState make_generation_state(const GenerateParams & params, bool keep_stages) {
    GenerationState state;
    if (const std::string error = resolve_generate_task(params, state.task); !error.empty()) {
        throw std::invalid_argument(error);
    }
    state.plan = make_generation_plan(params, state.task);
    state.seed = resolve_seed(params.seed);
    state.language = params.vocal_language.empty() ? "en" : params.vocal_language;
    state.prompt = make_prompt(params, state.language);
    state.low_memory = !keep_stages;
    return state;
}

template <typename EngineImpl>
static bool prepare_audio_conditioning(EngineImpl & engine, const GenerateParams & params,
                                       GenerationState & state, const StageReporter & report,
                                       StageDump & dump, StageTimes & timing) {
    if (!state.plan.encode_source && !state.plan.encode_reference) return true;

    engine.ensure_vae(true);
    const AudioEncoder encode =
        [&](const std::vector<float> & pcm, const char * stage, EncodedAudio & output) {
            return encode_audio(*engine.vae, pcm, stage, resolve_conditioning_dump_name(stage),
                                report, engine.opts.verbose, dump, timing, output);
        };
    if (!prepare_generation_conditioning(params, state.plan, encode, state.conditioning)) return false;

    if (state.plan.encode_source) {
        state.context_latents = std::move(state.conditioning.source.latent);
        state.latent_frames = state.conditioning.source.frames;
        state.prompt.duration = (float) state.latent_frames / AUDIO_LATENT_RATE;
    }
    if (state.low_memory) engine.free_vae();
    return true;
}

static bool has_complete_metadata(const AcePrompt & prompt) {
    return prompt.bpm > 0 && prompt.duration > 0 &&
           !prompt.keyscale.empty() && !prompt.timesignature.empty();
}

static LmSampleParams make_lm_sample_params(const GenerateParams & params, long long seed,
                                            bool verbose, const StageReporter & report) {
    LmSampleParams sample;
    sample.temperature = params.lm_temperature;
    sample.top_p = params.lm_top_p;
    sample.top_k = params.lm_top_k;
    sample.cfg_scale = params.lm_cfg_scale;
    sample.seed = (uint32_t) seed;
    sample.verbose = verbose;
    sample.on_step = [&](int current, int total) { report("lm", current, total); };
    return sample;
}

template <typename EngineImpl>
static void generate_audio_codes(EngineImpl & engine, const GenerateParams & params,
                                 GenerationState & state, const StageReporter & report,
                                 std::vector<int> & codes) {
    if (!params.audio_codes.empty()) {
        codes = params.audio_codes;
        if (engine.opts.verbose) {
            fprintf(stderr, "[acestep-engine] using %zu pre-supplied codes (LM skipped)\n", codes.size());
        }
        return;
    }

    engine.ensure_lm();
    const LmSampleParams sample =
        make_lm_sample_params(params, state.seed, engine.opts.verbose, report);
    if (params.lm_phase1 && !has_complete_metadata(state.prompt)) {
        LmSampleParams phase_one = sample;
        phase_one.max_new_tokens = 0;
        if (!lm_generate_phase1(engine.lm, engine.bpe_lm, state.prompt, phase_one)) {
            fprintf(stderr, "[acestep-engine] Phase 1 failed; falling back to provided/default metas\n");
        }
    }
    if (!lm_generate_codes(engine.lm, engine.bpe_lm, state.prompt, sample, codes) || codes.empty()) {
        throw std::runtime_error("acestep engine: LM produced no audio codes");
    }
}

template <typename EngineImpl>
static bool run_lm_stage(EngineImpl & engine, const GenerateParams & params,
                         GenerationState & state, const StageReporter & report,
                         StageDump & dump, StageTimes & timing, std::vector<int> & codes) {
    if (!report("lm", 0, 1)) return false;
    generate_audio_codes(engine, params, state, report, codes);
    if (!report("lm", 1, 1)) return false;

    if (state.low_memory) {
        if (engine.lm && engine.opts.verbose) fprintf(stderr, "[acestep-engine] freeing LM (low-mem)\n");
        engine.free_lm();
    }
    timing.mark("lm");
    dump.write_ints("01_lm_codes", codes);
    return true;
}

template <typename EngineImpl>
static void run_detokenizer_stage(EngineImpl & engine, const std::vector<int> & codes,
                                  GenerationState & state, StageDump & dump, StageTimes & timing) {
    engine.ensure_detok();
    state.code_frames = (int) codes.size();
    state.latent_frames = state.code_frames * AUDIO_CODE_FRAME_RATIO;
    state.context_latents.assign((size_t) AUDIO_LATENT_CHANNELS * state.latent_frames, 0.0f);
    if (detok_model_decode(engine.detok, codes.data(), state.code_frames,
                           state.context_latents.data()) != state.latent_frames) {
        throw std::runtime_error("acestep engine: FSQ detokenizer failed");
    }
    timing.mark("detok");
    dump.write("02_detok_latent", state.context_latents, state.latent_frames, AUDIO_LATENT_CHANNELS);

    if (state.low_memory) {
        if (engine.detok && engine.opts.verbose) {
            fprintf(stderr, "[acestep-engine] freeing FSQ detokenizer (low-mem)\n");
        }
        engine.free_detok();
    }
}

template <typename EngineImpl>
static bool prepare_context_latents(EngineImpl & engine, const GenerateParams & params,
                                    GenerationState & state, const StageReporter & report,
                                    StageDump & dump, StageTimes & timing) {
    if (!state.plan.run_lm || !state.plan.run_detokenizer) {
        timing.mark("lm");
        timing.mark("detok");
        return true;
    }

    std::vector<int> codes;
    if (!run_lm_stage(engine, params, state, report, dump, timing, codes)) return false;
    run_detokenizer_stage(engine, codes, state, dump, timing);
    return true;
}

struct PromptTokens {
    std::vector<int32_t> text;
    std::vector<int32_t> lyrics;
};

#ifdef ACESTEP_PARITY_DEBUG
static void print_token_range(const std::vector<int> & tokens, int first, int last) {
    for (int index = first; index < last; ++index) fprintf(stderr, " %d", tokens[(size_t) index]);
}

static void print_prompt_tokens(const std::vector<int> & text, const std::vector<int> & lyrics) {
    fprintf(stderr, "[acestep-dbg] S_text=%zu S_lyric=%zu (enc_S=%zu)\n",
            text.size(), lyrics.size(), text.size() + lyrics.size());
    fprintf(stderr, "[acestep-dbg] text_ids[0..8]:");
    print_token_range(text, 0, std::min(8, (int) text.size()));
    fprintf(stderr, " ... tail:");
    print_token_range(text, std::max(0, (int) text.size() - 4), (int) text.size());
    fprintf(stderr, "\n[acestep-dbg] lyric_ids[0..8]:");
    print_token_range(lyrics, 0, std::min(8, (int) lyrics.size()));
    fprintf(stderr, " ... tail:");
    print_token_range(lyrics, std::max(0, (int) lyrics.size() - 4), (int) lyrics.size());
    fprintf(stderr, "\n");
}
#endif

static PromptTokens tokenize_prompt(const BpeTokenizer & tokenizer, const AcePrompt & prompt,
                                    const std::string & language) {
    const std::string metadata =
        build_metas(prompt.bpm, prompt.timesignature, prompt.keyscale, prompt.duration);
    const std::string text = std::string("# Instruction\n") + DIT_INSTR_COVER +
                             "\n\n# Caption\n" + prompt.caption + "\n\n# Metas\n" +
                             metadata + "<|endoftext|>\n";
    const std::string lyrics = std::string("# Languages\n") + language +
                               "\n\n# Lyric\n" + prompt.lyrics + "<|endoftext|>";
    const std::vector<int> text_ids = bpe_encode(tokenizer, text, true);
    const std::vector<int> lyric_ids = bpe_encode(tokenizer, lyrics, true);

#ifdef ACESTEP_PARITY_DEBUG
    if (dbg_env("ACESTEP_DUMP_DIR")) print_prompt_tokens(text_ids, lyric_ids);
#endif

    return {
        std::vector<int32_t>(text_ids.begin(), text_ids.end()),
        std::vector<int32_t>(lyric_ids.begin(), lyric_ids.end())
    };
}

template <typename EngineImpl>
static PromptEncoding encode_prompt(EngineImpl & engine, const PromptTokens & tokens,
                                    GenerationState & state, StageDump & dump, StageTimes & timing) {
    engine.ensure_textenc();
    PromptEncoding encoding;
    encoding.text_tokens = (int) tokens.text.size();
    encoding.lyric_tokens = (int) tokens.lyrics.size();
    if (!textenc_model_forward(engine.textenc, tokens.text.data(), encoding.text_tokens,
                               encoding.text_hidden)) {
        throw std::runtime_error("acestep engine: text-encoder forward failed");
    }
    if (!textenc_model_embed_lookup(engine.textenc, tokens.lyrics.data(), encoding.lyric_tokens,
                                    encoding.lyric_embedding)) {
        throw std::runtime_error("acestep engine: lyric embed lookup failed");
    }
    timing.mark("textenc");
    dump.write("04_text_hidden", encoding.text_hidden, encoding.text_tokens,
               (int) (encoding.text_hidden.size() / (size_t) encoding.text_tokens));
    dump.write("05_lyric_embed", encoding.lyric_embedding, encoding.lyric_tokens,
               (int) (encoding.lyric_embedding.size() / (size_t) encoding.lyric_tokens));

    if (state.low_memory) {
        if (engine.textenc && engine.opts.verbose) {
            fprintf(stderr, "[acestep-engine] freeing text-encoder (low-mem)\n");
        }
        engine.free_textenc();
    }
    return encoding;
}

template <typename EngineImpl>
static void encode_cross_attention(EngineImpl & engine, const PromptEncoding & prompt,
                                   const GenerationState & state, EncoderConditioning & output,
                                   StageDump & dump, StageTimes & timing) {
    const TimbreInput timbre = resolve_timbre_input(
        state.plan, state.conditioning.reference, state.context_latents,
        state.latent_frames, cond_model_silence_frame(engine.cond));
    if (!cond_model_forward(engine.cond, prompt.text_hidden.data(), prompt.text_tokens,
                            prompt.lyric_embedding.data(), prompt.lyric_tokens,
                            timbre.data, timbre.frames, output.hidden, &output.sequence)) {
        throw std::runtime_error("acestep engine: cond-encoder forward failed");
    }
    output.hidden_size = (int) (output.hidden.size() / (size_t) output.sequence);

    if (state.low_memory) {
        if (engine.cond && engine.opts.verbose) {
            fprintf(stderr, "[acestep-engine] freeing cond-encoder (low-mem)\n");
        }
        engine.free_cond();
    }
    timing.mark("cond");
    dump.write("03_context", output.context, output.frames, output.context_channels);
    dump.write("06_enc_hidden", output.hidden, output.sequence, output.hidden_size);
}

template <typename EngineImpl>
static EncoderConditioning prepare_encoder_conditioning(EngineImpl & engine,
                                                         GenerationState & state,
                                                         StageDump & dump, StageTimes & timing) {
    const DitConfig & config = engine.dit_cfg;
    const int patch = config.patch_size;
    EncoderConditioning output;
    output.frames = ((state.latent_frames + patch - 1) / patch) * patch;
    output.context_channels = config.in_channels - config.out_channels;

    engine.ensure_cond();
    output.context = make_dit_context(
        state.context_latents, cond_model_silence_frame(engine.cond), output.frames,
        state.latent_frames, output.context_channels, config.out_channels);
    const PromptTokens tokens = tokenize_prompt(engine.bpe_text, state.prompt, state.language);
    const PromptEncoding prompt = encode_prompt(engine, tokens, state, dump, timing);
    encode_cross_attention(engine, prompt, state, output, dump, timing);
    return output;
}

static NoiseSchedule prepare_noise(const GenerateParams & params, const GenerationState & state,
                                   const DitConfig & config, const EncoderConditioning & conditioning,
                                   bool verbose, StageDump & dump, StageTimes & timing) {
    NoiseSchedule output;
    output.noise.resize((size_t) config.out_channels * conditioning.frames);
    philox_randn(state.seed, output.noise.data(), (int) output.noise.size(), true);
    output.steps = params.inference_steps > 0
                       ? params.inference_steps
                       : (config.is_turbo ? TURBO_STEPS : STANDARD_STEPS);
    output.shift = params.shift > 0.0f
                       ? params.shift
                       : (config.is_turbo ? TURBO_SHIFT : STANDARD_SHIFT);
    dit_build_schedule(output.shift, output.steps, output.schedule);

    if (state.plan.blend_cover_noise) {
        const CoverNoiseResult adjustment = apply_cover_noise(
            output.noise, state.context_latents, conditioning.frames, state.latent_frames,
            config.out_channels, state.task.cover_noise_strength, output.schedule);
        output.steps = adjustment.remaining_steps;
        if (verbose) {
            fprintf(stderr,
                    "[acestep-engine] cover_noise_strength=%.2f -> nearest_t=%.4f remaining_steps=%d\n",
                    state.task.cover_noise_strength, adjustment.nearest_time, output.steps);
        }
    }
    timing.mark("noise");
    dump.write("07_noise", output.noise, conditioning.frames, config.out_channels);
    return output;
}

#ifdef ACESTEP_PARITY_DEBUG
static void inject_parity_inputs(NoiseSchedule & noise, EncoderConditioning & conditioning) {
    if (const char * path = dbg_env("ACESTEP_INJECT_NOISE")) dbg_load_dump(path, noise.noise);
    if (const char * path = dbg_env("ACESTEP_INJECT_CONTEXT")) dbg_load_dump(path, conditioning.context);
    if (const char * path = dbg_env("ACESTEP_INJECT_ENC")) dbg_load_dump(path, conditioning.hidden);
}

static void dump_parity_inputs(const std::vector<float> & latent, const NoiseSchedule & noise,
                               const EncoderConditioning & conditioning, int output_channels) {
    const char * directory = dbg_env("ACESTEP_DUMP_DIR");
    if (!directory) return;
    const std::string path(directory);
    dbg_write_dump(path + "/our_dit_output.bin", latent, conditioning.frames, output_channels);
    dbg_write_dump(path + "/our_noise.bin", noise.noise, conditioning.frames, output_channels);
    dbg_write_dump(path + "/our_context.bin", conditioning.context, conditioning.frames,
                   conditioning.context_channels);
    dbg_write_dump(path + "/our_enc_hidden.bin", conditioning.hidden, conditioning.sequence,
                   conditioning.hidden_size);
}
#endif

template <typename EngineImpl>
static bool sample_dit_latent(EngineImpl & engine, const GenerateParams & params,
                              const GenerationState & state, EncoderConditioning & conditioning,
                              NoiseSchedule & noise, const StageReporter & report,
                              StageDump & dump, StageTimes & timing, std::vector<float> & latent) {
    const DitConfig & config = engine.dit_cfg;
    if (engine.opts.verbose) {
        fprintf(stderr, "[acestep-engine] DiT: turbo=%d steps=%d shift=%.2f T=%d task=%s\n",
                (int) config.is_turbo, noise.steps, noise.shift,
                conditioning.frames, state.task.type.c_str());
    }
    engine.ensure_dit();
    if (!report("dit", 0, noise.steps)) return false;

    DitSampleParams sample;
    sample.noise = noise.noise.data();
    sample.context_latents = conditioning.context.data();
    sample.enc_hidden = conditioning.hidden.data();
    sample.enc_S = conditioning.sequence;
    sample.H_enc = conditioning.hidden_size;
    sample.T = conditioning.frames;
    sample.N = 1;
    sample.schedule = noise.schedule.data();
    sample.num_steps = noise.steps;
    sample.real_enc_S = &conditioning.sequence;
    sample.dcw_enabled = params.dcw_enabled;
    sample.dcw_scaler = params.dcw_scaler;
    sample.dcw_high_scaler = params.dcw_high_scaler;
    sample.on_step = [&](int step, int total) { return report("dit", step, total); };

    if (!dit_sample(engine.dit, sample, latent)) {
        if (engine.cancel_flag.load()) return false;
        throw std::runtime_error("acestep engine: DiT sample failed");
    }
    if (!report("dit", noise.steps, noise.steps)) return false;
    timing.mark("dit");
    dump.write("08_dit_latent", latent, conditioning.frames, config.out_channels);
    if (state.low_memory) {
        if (engine.dit && engine.opts.verbose) fprintf(stderr, "[acestep-engine] freeing DiT (low-mem)\n");
        engine.free_dit();
    }

#ifdef ACESTEP_PARITY_DEBUG
    dump_parity_inputs(latent, noise, conditioning, config.out_channels);
#endif
    return true;
}

template <typename EngineImpl>
static bool decode_audio(EngineImpl & engine, const std::vector<float> & latent,
                         const EncoderConditioning & conditioning, bool low_memory,
                         const StageReporter & report, StageDump & dump,
                         StageTimes & timing, GenerateResult & result) {
    engine.ensure_vae();
    if (!report("vae", 0, 1)) return false;

    bool completed = true;
    result.pcm = engine.vae->decode(latent, conditioning.frames, [&](int done, int total) {
        completed = report("vae", done, total);
        return completed;
    });
    if (!completed) return false;
    if (result.pcm.empty()) throw std::runtime_error("acestep engine: VAE decode failed");
    report("vae", 1, 1);
    timing.mark("vae");
    dump.write("09_vae_pcm", result.pcm, (int) (result.pcm.size() / AUDIO_CHANNELS), AUDIO_CHANNELS);
    if (engine.opts.verbose) timing.dump(stderr);

    if (low_memory) {
        if (engine.vae && engine.opts.verbose) fprintf(stderr, "[acestep-engine] freeing VAE (low-mem)\n");
        engine.free_vae();
    }
    return true;
}

static void populate_metadata(const GenerationState & state, GenerateResult & result) {
    result.metadata.caption = state.prompt.caption;
    result.metadata.lyrics = state.prompt.lyrics;
    result.metadata.keyscale = state.prompt.keyscale;
    result.metadata.vocal_language = state.prompt.vocal_language;
    result.metadata.bpm = state.prompt.bpm;
    result.metadata.timesignature =
        state.prompt.timesignature.empty() ? 0 : atoi(state.prompt.timesignature.c_str());
    result.metadata.seed = state.seed;
    result.metadata.n_codes = state.code_frames;
}

GenerateResult Engine::generate(const GenerateParams & params, const ProgressFn & progress) const {
    Impl & engine = *impl_;
    engine.cancel_flag.store(false);

    GenerateResult result;
    result.sample_rate = engine.sr;
    result.channels = AUDIO_CHANNELS;

    GenerationState state = make_generation_state(params, engine.keep_stages);
    const StageReporter report = [&](const char * stage, int step, int total) {
        if (progress && !progress(stage, step, total)) {
            engine.cancel_flag.store(true);
            return false;
        }
        return !engine.cancel_flag.load();
    };
    StageTimes timing;
    StageDump dump(engine.opts.dump_stages_dir, engine.opts.verbose);

    if (!prepare_audio_conditioning(engine, params, state, report, dump, timing)) return result;
    if (!prepare_context_latents(engine, params, state, report, dump, timing)) return result;

    EncoderConditioning conditioning = prepare_encoder_conditioning(engine, state, dump, timing);
    NoiseSchedule noise = prepare_noise(
        params, state, engine.dit_cfg, conditioning, engine.opts.verbose, dump, timing);

#ifdef ACESTEP_PARITY_DEBUG
    inject_parity_inputs(noise, conditioning);
#endif

    std::vector<float> latent;
    if (!sample_dit_latent(engine, params, state, conditioning, noise,
                           report, dump, timing, latent)) {
        return result;
    }
    if (!decode_audio(engine, latent, conditioning, state.low_memory,
                      report, dump, timing, result)) {
        return result;
    }
    populate_metadata(state, result);
    return result;
}

void        Engine::cancel() const { impl_->cancel_flag.store(true); }
int         Engine::sample_rate() const { return impl_->sr; }  // cached; VAE loaded lazily
std::string Engine::backend_name() const { return impl_->backend ? ggml_backend_name(impl_->backend) : "cpu"; }

} // namespace tts_cpp::acestep
