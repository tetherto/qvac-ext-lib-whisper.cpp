// CPU-only, weight-free unit tests for the ACE-Step pipeline's pure logic.
//
// None of these need a GGUF fixture — they exercise the deterministic math
// that the rest of the pipeline is built on, so a refactor that drifts from
// the acestep.cpp reference breaks here (fast, on a fresh checkout under
// `ctest -L unit`) before the fixture-bound integration tests get a chance to.
//
// Coverage:
//   1. dit_build_schedule  — flow-matching time schedule (shift/steps).
//   2. dit_apply_haar_dcw  — official sampler-side low/high wavelet correction.
//   3. philox_randn        — Philox4x32-10 + Box-Muller (torch.randn parity).
//   4. fsq_decode_index    — FSQ index -> 6 normalized dims (strides 8/8/8/5/5/5).
//   5. sample_top_k_p      — top-k/top-p LM sampler (determinism + argmax).
//   6. vae_progress_pct    — VAE decode progress clamp/monotonicity.
//   6b. vae_shrink_window_core — chunked-decode window vs the backend alloc cap.
//   7. GPU device types    — discrete and integrated GPUs are selectable.
//   8. stage placement     — which backend the LM / detokenizer / encoders run on.

#include "backend_registry.h"
#include "cover_noise.h"
#include "dit_ggml.h"
#include "detok_ggml.h"
#include "generate_task.h"
#include "generation_conditioning.h"
#include "generation_plan.h"
#include "lm_pipeline.h"
#include "philox.h"
#include "stage_placement.h"
#include "vae_encode_windows.h"
#include "vae_ggml.h"
#include "wav_reader.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++g_checks;                                                                   \
        if (!(cond)) {                                                                \
            ++g_failures;                                                             \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        }                                                                             \
    } while (0)

bool approx(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }

// 1. dit_build_schedule ------------------------------------------------------
void test_schedule() {
    using tts_cpp::acestep::dit_build_schedule;
    std::vector<float> sched;

    // shift == 1.0 collapses to a linear ramp: schedule[i] = 1 - i/N.
    dit_build_schedule(1.0f, 8, sched);
    CHECK(sched.size() == 8);
    CHECK(approx(sched[0], 1.0f));                 // first step starts at pure noise
    CHECK(approx(sched[4], 1.0f - 4.0f / 8.0f));   // == 0.5
    for (size_t i = 1; i < sched.size(); ++i) CHECK(sched[i] < sched[i - 1]);  // strictly decreasing

    // shift > 1 (turbo uses 3.0) still starts at 1.0 and stays monotone.
    dit_build_schedule(3.0f, 8, sched);
    CHECK(sched.size() == 8);
    CHECK(approx(sched[0], 1.0f));
    for (size_t i = 1; i < sched.size(); ++i) CHECK(sched[i] < sched[i - 1]);
    for (float v : sched) CHECK(v > 0.0f && v <= 1.0f);
}

// 2. Haar DCW ----------------------------------------------------------------
void test_haar_dcw() {
    using tts_cpp::acestep::dit_apply_haar_dcw;

    // Low-band correction moves both samples in a pair together.
    std::vector<float> x = { 2.0f, 4.0f };
    const std::vector<float> y_low = { 1.0f, 3.0f };
    dit_apply_haar_dcw(x, y_low, /*T=*/2, /*C=*/1, /*N=*/1, 0.1f, 0.0f);
    CHECK(approx(x[0], 2.1f));
    CHECK(approx(x[1], 4.1f));

    // High-band correction changes the contrast inside the temporal pair.
    x = { 2.0f, 4.0f };
    const std::vector<float> y_high = { 1.0f, 5.0f };
    dit_apply_haar_dcw(x, y_high, 2, 1, 1, 0.0f, 0.2f);
    CHECK(approx(x[0], 2.2f));
    CHECK(approx(x[1], 3.8f));

    // Odd temporal lengths use a zero-padded partner and discard it after IDWT.
    x = { 2.0f };
    const std::vector<float> y_odd = { 1.0f };
    dit_apply_haar_dcw(x, y_odd, 1, 1, 1, 0.1f, 0.2f);
    CHECK(approx(x[0], 2.15f));

    // Disabled coefficients are an exact no-op.
    x = { -3.0f, 7.0f };
    const std::vector<float> before = x;
    dit_apply_haar_dcw(x, y_low, 2, 1, 1, 0.0f, 0.0f);
    CHECK(x == before);
}

// 3. philox_randn ------------------------------------------------------------
void test_philox() {
    using tts_cpp::acestep::philox_randn;

    // Golden vector for seed 42 (bf16-rounded, the mode the engine uses for
    // the DiT initial noise). These values were validated corr=1.0 against
    // torch.randn() on CUDA via acestep.cpp's --dump; they lock the port.
    const float golden[8] = { 0.194335938f,  2.156250000f,  -0.171875000f, 0.847656250f,
                              -1.921875000f, 0.652343750f,  -0.648437500f, -0.816406250f };
    float out[8];
    philox_randn(42, out, 8, /*bf16_round=*/true);
    for (int i = 0; i < 8; ++i) CHECK(approx(out[i], golden[i], 1e-6f));

    // Determinism: same seed -> identical stream.
    float a[16], b[16];
    philox_randn(1234, a, 16, true);
    philox_randn(1234, b, 16, true);
    for (int i = 0; i < 16; ++i) CHECK(a[i] == b[i]);

    // Different seeds -> different stream (extremely unlikely to collide).
    float c[16];
    philox_randn(4321, c, 16, true);
    bool differs = false;
    for (int i = 0; i < 16; ++i) differs |= (a[i] != c[i]);
    CHECK(differs);

    // Statistical sanity over a large draw: mean ~ 0, std ~ 1.
    const int          N = 8192;
    std::vector<float> big(N);
    philox_randn(7, big.data(), N, false);
    double mean = 0.0;
    for (float v : big) mean += v;
    mean /= N;
    double var = 0.0;
    for (float v : big) var += (v - mean) * (v - mean);
    var /= N;
    CHECK(std::fabs(mean) < 0.05);
    CHECK(std::fabs(std::sqrt(var) - 1.0) < 0.05);
}

// 3. fsq_decode_index --------------------------------------------------------
void test_fsq() {
    using tts_cpp::acestep::fsq_decode_index;
    float out[6];

    // index 0 -> all digits 0 -> every dim at the low end (-1).
    fsq_decode_index(0, out);
    for (int d = 0; d < 6; ++d) CHECK(approx(out[d], -1.0f));

    // digit 1 in dim0 (L=8, half=3.5): 1/3.5 - 1.
    fsq_decode_index(1, out);
    CHECK(approx(out[0], 1.0f / 3.5f - 1.0f));
    for (int d = 1; d < 6; ++d) CHECK(approx(out[d], -1.0f));

    // Max index across all dims -> every dim at the high end (+1).
    int max_index = 8 * 8 * 8 * 5 * 5 * 5 - 1;
    fsq_decode_index(max_index, out);
    for (int d = 0; d < 6; ++d) CHECK(approx(out[d], 1.0f));

    // Output is always in [-1, 1] (small FP slack at the boundaries).
    const float slack = 1e-4f;
    for (int idx = 0; idx < 500; idx += 37) {
        fsq_decode_index(idx, out);
        for (int d = 0; d < 6; ++d) CHECK(out[d] >= -1.0f - slack && out[d] <= 1.0f + slack);
    }
}

// 4. sample_top_k_p ----------------------------------------------------------
void test_sampler() {
    using tts_cpp::acestep::sample_top_k_p;

    const int         V = 32;
    std::vector<float> logits(V, 0.0f);
    logits[17] = 10.0f;  // clearly dominant token

    // top_k = 1 keeps only the argmax, so the pick is deterministic regardless
    // of the RNG draw.
    for (uint32_t seed = 0; seed < 8; ++seed) {
        std::vector<float> l = logits;
        std::mt19937       rng(seed);
        CHECK(sample_top_k_p(l.data(), V, 1.0f, 1.0f, /*top_k=*/1, rng) == 17);
    }

    // Same RNG seed -> identical token sequence (reproducible generation).
    std::vector<float> l1 = logits, l2 = logits;
    std::mt19937       r1(99), r2(99);
    for (int step = 0; step < 5; ++step) {
        std::vector<float> a = l1, b = l2;
        int ta = sample_top_k_p(a.data(), V, 0.85f, 0.9f, 0, r1);
        int tb = sample_top_k_p(b.data(), V, 0.85f, 0.9f, 0, r2);
        CHECK(ta == tb);
        CHECK(ta >= 0 && ta < V);
    }
}

// 5. vae_progress_pct --------------------------------------------------------
// The VAE decode reports progress per computed graph node. A GPU+CPU scheduler
// can insert extra copy/split nodes, so the callback may fire MORE than
// ggml_graph_n_nodes(gf) times; the percentage must stay monotone and bounded
// to [0, 100] (no progress-bar overshoot). This locks that clamp/throttle math.
void test_vae_progress() {
    using tts_cpp::acestep::vae_progress_pct;

    // Endpoints and a midpoint.
    CHECK(vae_progress_pct(0, 200) == 0);
    CHECK(vae_progress_pct(100, 200) == 50);
    CHECK(vae_progress_pct(200, 200) == 100);

    // Overshoot: the scheduler fires past `total` -> clamped to 100, never more.
    CHECK(vae_progress_pct(201, 200) == 100);
    CHECK(vae_progress_pct(10000, 200) == 100);

    // Degenerate inputs are safe (no div-by-zero, no negative pct).
    CHECK(vae_progress_pct(5, 0) == 0);
    CHECK(vae_progress_pct(5, -1) == 0);
    CHECK(vae_progress_pct(-3, 200) == 0);

    // Monotone non-decreasing and bounded across a full sweep that overshoots,
    // mirroring the eval callback incrementing `done` once per fired node.
    const int total = 137;  // odd, to exercise integer division
    int       prev  = -1;
    int       last_emitted = -1;
    for (int done = 1; done <= total + 25; ++done) {  // +25 => simulate extra copy/split nodes
        int pct = vae_progress_pct(done, total);
        CHECK(pct >= 0 && pct <= 100);
        CHECK(pct >= prev);  // never goes backwards
        prev = pct;
        // Throttle: only distinct percentages would be surfaced to the user cb.
        if (pct != last_emitted) last_emitted = pct;
    }
    CHECK(prev == 100);          // ends exactly at 100
    CHECK(last_emitted == 100);  // the final surfaced value is 100
}

// 5b. VAE window sizing ------------------------------------------------------
// The decoder's im2col node grows linearly with the window and cannot be split
// across allocations, so a backend that caps allocation size caps the window.
// Adreno 740 reports 1024 MB and the 256+2*48 window needs 1155 MiB, which is
// what aborted a 30 s GPU decode; these are those measured numbers.
void test_vae_window_core() {
    using tts_cpp::acestep::vae_shrink_window_core;

    const int    core = 256, ov = 48, core_min = 64;
    const size_t adreno_cap = (size_t) 1024 * 1024 * 1024;
    const size_t peak_352   = (size_t) 1155 * 1024 * 1024;  // measured at 256 + 2*48 frames

    // A backend that does not cap allocations (CPU reports SIZE_MAX) keeps 256,
    // so CPU / Metal / iOS behaviour is untouched.
    CHECK(vae_shrink_window_core(core, ov, peak_352, SIZE_MAX, core_min) == core);
    CHECK(vae_shrink_window_core(core, ov, (size_t) 900 * 1024 * 1024, adreno_cap, core_min) == core);

    // Adreno: must shrink, and the result must actually fit once rescaled.
    const int fitted = vae_shrink_window_core(core, ov, peak_352, adreno_cap, core_min);
    CHECK(fitted < core);
    CHECK(fitted >= core_min);
    const double bytes_per_frame = (double) peak_352 / (double) (core + 2 * ov);
    CHECK((size_t) (bytes_per_frame * (fitted + 2 * ov)) <= adreno_cap);

    // Never below the floor, however small the cap.
    CHECK(vae_shrink_window_core(core, ov, peak_352, 1024 * 1024, core_min) == core_min);
    CHECK(vae_shrink_window_core(core_min, ov, peak_352, 1024 * 1024, core_min) == core_min);

    // Converges: re-applying with the rescaled peak is a fixed point.
    const size_t peak_fitted = (size_t) (bytes_per_frame * (fitted + 2 * ov));
    CHECK(vae_shrink_window_core(fitted, ov, peak_fitted, adreno_cap, core_min) == fitted);

    // Always makes progress while it does not fit, so the caller's loop ends.
    int c = core;
    for (int i = 0; i < 16 && c > core_min; ++i) {
        const int next = vae_shrink_window_core(c, ov, peak_352, adreno_cap, core_min);
        if (next == c) break;
        CHECK(next < c);
        c = next;
    }
}

// 6. GPU device types --------------------------------------------------------
// Vulkan classifies UMA adapters such as Android Mali as IGPU. AceStep must
// accept both device classes while rejecting CPU and non-GPU accelerators.
void test_backend_device_types() {
    using tts_cpp::acestep::backend_device_type_is_gpu;
    using tts_cpp::acestep::backend_reg_name_is_validated_gpu;
    using tts_cpp::acestep::parse_adreno_version;

    CHECK(backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_IGPU));
    CHECK(!backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_CPU));
    CHECK(!backend_device_type_is_gpu(GGML_BACKEND_DEVICE_TYPE_ACCEL));

    CHECK(backend_reg_name_is_validated_gpu("Vulkan"));
    CHECK(backend_reg_name_is_validated_gpu("MTL"));
    CHECK(backend_reg_name_is_validated_gpu("Metal"));
    // OpenCL is deliberately absent: it is reached by its own Adreno pass in
    // backend_gpu_init(), not by this Vulkan/Metal preference.
    CHECK(!backend_reg_name_is_validated_gpu("OpenCL"));
    CHECK(!backend_reg_name_is_validated_gpu("CUDA"));
    CHECK(!backend_reg_name_is_validated_gpu(nullptr));

    // That Adreno pass gates on the generation parsed out of the device
    // name/description, so the parse is what decides OpenCL-over-Vulkan.
    CHECK(parse_adreno_version("Adreno (TM) 740") == 740);
    CHECK(parse_adreno_version("QUALCOMM Adreno(TM) 830") == 830);
    CHECK(parse_adreno_version("Adreno X1-85") == 800);       // Snapdragon-X naming
    CHECK(parse_adreno_version("Adreno (TM) 640") == 640);    // parsed, but below the 700 gate
    // An embedded "OpenCL 3.0" must not be mistaken for the model number, and a
    // non-Adreno device must not be claimed at all.
    CHECK(parse_adreno_version("Adreno (TM) 750 (OpenCL 3.0)") == 750);
    CHECK(parse_adreno_version("Mali-G715") == -1);
    CHECK(parse_adreno_version("Apple M1 Pro") == -1);
    CHECK(parse_adreno_version("") == -1);
    CHECK(parse_adreno_version(nullptr) == -1);
}

// 7. stage placement ---------------------------------------------------------
// Which backend each stage runs on decides which numerical path the generated
// audio takes, so the policy is locked here rather than only observed on a
// device lane. Mirrors the three branches Engine::create() relies on: the
// backend allowlist, the CPU fallback for everything else, and the env
// overrides layered on top.
void test_stage_placement() {
    using tts_cpp::acestep::backend_name_is_metal;
    using tts_cpp::acestep::backend_name_is_opencl;
    using tts_cpp::acestep::backend_name_is_vulkan;
    using tts_cpp::acestep::PlacementOverrides;
    using tts_cpp::acestep::resolve_stage_placement;
    using tts_cpp::acestep::StagePlacement;

    // -- backend name matching ------------------------------------------------
    // ggml-metal registers as "MTL"; older ggml reported "Metal". Both must
    // match or the allowlist is silently dead on one of them.
    CHECK(backend_name_is_metal("MTL"));
    CHECK(backend_name_is_metal("Metal"));
    CHECK(backend_name_is_vulkan("Vulkan"));
    // ggml-opencl's REGISTRY name; its device reports as "GPUOpenCL", which is
    // not what reaches here.
    CHECK(backend_name_is_opencl("OpenCL"));
    CHECK(!backend_name_is_opencl("GPUOpenCL"));

    // The input is the REGISTRY name, which carries no device-index suffix.
    // ggml_backend_name() would hand over "MTL0" / "Vulkan0" and match nothing.
    CHECK(!backend_name_is_metal("MTL0"));
    CHECK(!backend_name_is_vulkan("Vulkan0"));
    CHECK(!backend_name_is_opencl("OpenCL0"));

    // Exact compare: no case folding, no substring match, and null/empty safe.
    CHECK(!backend_name_is_metal("mtl"));
    CHECK(!backend_name_is_metal("metal"));
    CHECK(!backend_name_is_metal("MTLX"));
    CHECK(!backend_name_is_vulkan("vulkan"));
    CHECK(!backend_name_is_opencl("opencl"));
    CHECK(!backend_name_is_metal("CUDA"));
    CHECK(!backend_name_is_vulkan("MTL"));
    CHECK(!backend_name_is_metal("Vulkan"));
    CHECK(!backend_name_is_opencl("Vulkan"));
    CHECK(!backend_name_is_metal(""));
    CHECK(!backend_name_is_vulkan(""));
    CHECK(!backend_name_is_opencl(""));
    CHECK(!backend_name_is_metal(nullptr));
    CHECK(!backend_name_is_vulkan(nullptr));
    CHECK(!backend_name_is_opencl(nullptr));

    const PlacementOverrides none;

    // -- allowlist: Metal and OpenCL keep LM + detokenizer on GPU ---------------
    for (const char * allowed : { "MTL", "Metal", "OpenCL" }) {
        StagePlacement p = resolve_stage_placement(allowed, none);
        CHECK(p.lm_on_gpu);
        CHECK(p.detok_on_gpu);
        CHECK(p.enc_on_gpu);  // encoders follow the GPU on every backend
    }

    // Vulkan is validated for every stage except the autoregressive LM. On
    // Mali-G715, GPU LM logits collapse to repeated codes and truncate songs.
    {
        StagePlacement p = resolve_stage_placement("Vulkan", none);
        CHECK(!p.lm_on_gpu);
        CHECK(p.detok_on_gpu);
        CHECK(p.enc_on_gpu);
    }

    // -- fallback: everything else keeps the shipping CPU placement -----------
    // Unmeasured backends must not silently pick up the GPU path. "MTL0" is in
    // this list on purpose: a suffixed name is NOT the allowlisted one.
    const char * const others[] = { "CUDA",     "SYCL",     "BLAS", "CPU",  "MTL0",
                                    "Vulkan0",  "OpenCL0",  "",     nullptr };
    for (const char * other : others) {
        StagePlacement p = resolve_stage_placement(other, none);
        CHECK(!p.lm_on_gpu);
        CHECK(!p.detok_on_gpu);
        CHECK(p.enc_on_gpu);  // only the LM and the detokenizer are allowlisted
    }

    // -- env overrides: applied after the allowlist ---------------------------
    // GPU hatch lifts a non-allowlisted backend (this is how a new backend gets
    // measured without a rebuild).
    {
        PlacementOverrides ov;
        ov.lm_gpu        = true;
        StagePlacement p = resolve_stage_placement("CUDA", ov);
        CHECK(p.lm_on_gpu);
        CHECK(!p.detok_on_gpu);  // the LM hatch must not move the detokenizer
    }
    {
        PlacementOverrides ov;
        ov.detok_gpu     = true;
        StagePlacement p = resolve_stage_placement("CUDA", ov);
        CHECK(p.detok_on_gpu);
        CHECK(!p.lm_on_gpu);
    }

    // CPU hatch demotes an allowlisted backend.
    {
        PlacementOverrides ov;
        ov.lm_cpu        = true;
        StagePlacement p = resolve_stage_placement("MTL", ov);
        CHECK(!p.lm_on_gpu);
        CHECK(p.detok_on_gpu);
    }
    {
        PlacementOverrides ov;
        ov.detok_cpu     = true;
        StagePlacement p = resolve_stage_placement("Vulkan", ov);
        CHECK(!p.detok_on_gpu);
        CHECK(!p.lm_on_gpu);
    }

    // Precedence: CPU wins when both hatches are set for the same stage, on an
    // allowlisted backend and on a fallback one alike.
    for (const char * name : { "MTL", "Metal", "Vulkan", "OpenCL", "CUDA" }) {
        PlacementOverrides ov;
        ov.lm_gpu        = true;
        ov.lm_cpu        = true;
        ov.detok_gpu     = true;
        ov.detok_cpu     = true;
        StagePlacement p = resolve_stage_placement(name, ov);
        CHECK(!p.lm_on_gpu);
        CHECK(!p.detok_on_gpu);
    }

    // The encoder hatch is independent of the LM/detokenizer allowlist.
    for (const char * name : { "MTL", "Vulkan", "OpenCL", "CUDA" }) {
        PlacementOverrides ov;
        ov.encoders_cpu  = true;
        StagePlacement p = resolve_stage_placement(name, ov);
        CHECK(!p.enc_on_gpu);
        CHECK(p.lm_on_gpu == (backend_name_is_metal(name) || backend_name_is_opencl(name)));
    }
}

// 7b. env -> overrides -------------------------------------------------------
// Locks which variable drives which stage, and that PRESENCE is what counts:
// ACESTEP_LM_CPU=0 still forces the LM to the CPU (the getenv() semantics this
// policy inherited). Uses "0" as the "set" value so the assertion is identical
// on POSIX and Windows, where _putenv_s(k, "") removes the variable instead.
void set_env(const char * key, const char * value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value) setenv(key, value, 1);
    else       unsetenv(key);
#endif
}

void test_placement_env() {
    using tts_cpp::acestep::placement_overrides_from_env;
    using tts_cpp::acestep::PlacementOverrides;

    const char * const vars[] = { "ACESTEP_LM_GPU",    "ACESTEP_LM_CPU", "ACESTEP_DETOK_GPU",
                                  "ACESTEP_DETOK_CPU", "ACESTEP_ENCODERS_CPU" };
    auto clear_all = [&] {
        for (const char * k : vars) set_env(k, nullptr);
    };

    // Nothing set -> no override.
    clear_all();
    {
        PlacementOverrides ov = placement_overrides_from_env();
        CHECK(!ov.lm_gpu);
        CHECK(!ov.lm_cpu);
        CHECK(!ov.detok_gpu);
        CHECK(!ov.detok_cpu);
        CHECK(!ov.encoders_cpu);
    }

    // One variable at a time -> exactly its own flag, nothing else.
    for (size_t i = 0; i < sizeof(vars) / sizeof(vars[0]); ++i) {
        clear_all();
        set_env(vars[i], "0");  // presence, not value
        PlacementOverrides ov  = placement_overrides_from_env();
        const bool         set[] = { ov.lm_gpu, ov.lm_cpu, ov.detok_gpu, ov.detok_cpu, ov.encoders_cpu };
        for (size_t j = 0; j < sizeof(set) / sizeof(set[0]); ++j) CHECK(set[j] == (i == j));
    }

    clear_all();  // leave the environment as found
}

void test_generate_task_kinds() {
    using tts_cpp::acestep::TASK_COVER;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::TASK_TEXT2MUSIC;
    using tts_cpp::acestep::is_cover_task;

    CHECK(is_cover_task(TASK_COVER));
    CHECK(is_cover_task(TASK_COVER_NOFSQ));
    CHECK(!is_cover_task(TASK_TEXT2MUSIC));
    CHECK(!is_cover_task(""));
}

void test_generate_task_defaults() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::TASK_TEXT2MUSIC;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    params.source_audio.assign(8, 0.25f);
    params.reference_audio.assign(8, 0.5f);
    const float * source_data = params.source_audio.data();
    const float * reference_data = params.reference_audio.data();
    GenerateTask task;
    CHECK(resolve_generate_task(params, task).empty());
    CHECK(task.type == TASK_TEXT2MUSIC);
    CHECK(approx(task.audio_cover_strength, 1.0f));
    CHECK(approx(task.cover_noise_strength, 0.0f));
    CHECK(params.source_audio.data() == source_data);
    CHECK(params.reference_audio.data() == reference_data);
}

void test_generate_task_audio_layout() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;

    params.task_type = TASK_COVER_NOFSQ;
    CHECK(resolve_generate_task(params, task).find("requires source_audio") != std::string::npos);

    params.source_audio.assign(3, 0.0f);
    CHECK(resolve_generate_task(params, task).find("source_audio must be interleaved stereo") != std::string::npos);

    params.source_audio.assign(4, 0.0f);
    params.reference_audio.assign(5, 0.0f);
    CHECK(resolve_generate_task(params, task).find("reference_audio must be interleaved stereo") != std::string::npos);

    params.reference_audio.assign(6, 0.0f);
    CHECK(resolve_generate_task(params, task).empty());
}

void test_generate_task_errors() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::TASK_COVER;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;

    params.task_type = "repaint";
    CHECK(resolve_generate_task(params, task).find("unsupported task_type") != std::string::npos);

    params.task_type = TASK_COVER;
    params.source_audio.assign(4, 0.0f);
    CHECK(resolve_generate_task(params, task).find("not implemented") != std::string::npos);

    params.task_type = TASK_COVER_NOFSQ;
    params.audio_cover_strength = 0.5f;
    CHECK(resolve_generate_task(params, task).find("audio_cover_strength") != std::string::npos);
}

void test_generate_task_strengths() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;

    params.audio_cover_strength = 2.0f;
    params.cover_noise_strength = -0.5f;
    CHECK(resolve_generate_task(params, task).empty());
    CHECK(approx(task.audio_cover_strength, 1.0f));
    CHECK(approx(task.cover_noise_strength, 0.0f));

    params.audio_cover_strength = std::numeric_limits<float>::quiet_NaN();
    CHECK(resolve_generate_task(params, task).find("must be finite") != std::string::npos);

    params.audio_cover_strength = std::numeric_limits<float>::infinity();
    CHECK(resolve_generate_task(params, task).find("must be finite") != std::string::npos);

    params.audio_cover_strength = 1.0f;
    params.cover_noise_strength = std::numeric_limits<float>::quiet_NaN();
    CHECK(resolve_generate_task(params, task).find("must be finite") != std::string::npos);

    params.cover_noise_strength = std::numeric_limits<float>::infinity();
    CHECK(resolve_generate_task(params, task).find("must be finite") != std::string::npos);
}

void test_generation_plans() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::GenerationPlan;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::make_generation_plan;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;
    CHECK(resolve_generate_task(params, task).empty());

    GenerationPlan plan = make_generation_plan(params, task);
    CHECK(!plan.encode_source);
    CHECK(!plan.encode_reference);
    CHECK(plan.run_lm);
    CHECK(plan.run_detokenizer);

    params.reference_audio.assign(4, 0.0f);
    CHECK(resolve_generate_task(params, task).empty());
    plan = make_generation_plan(params, task);
    CHECK(plan.encode_reference);
    CHECK(plan.run_lm);

    params.task_type = TASK_COVER_NOFSQ;
    params.source_audio.assign(4, 0.0f);
    params.reference_audio.clear();
    CHECK(resolve_generate_task(params, task).empty());
    plan = make_generation_plan(params, task);
    CHECK(plan.encode_source);
    CHECK(plan.reuse_source_reference);
    CHECK(!plan.run_lm);
    CHECK(!plan.run_detokenizer);
    CHECK(!plan.blend_cover_noise);

    params.reference_audio.assign(4, 0.0f);
    params.cover_noise_strength = 0.5f;
    CHECK(resolve_generate_task(params, task).empty());
    plan = make_generation_plan(params, task);
    CHECK(plan.encode_reference);
    CHECK(!plan.reuse_source_reference);
    CHECK(plan.blend_cover_noise);
}

void test_generation_conditioning() {
    using tts_cpp::acestep::AudioEncoder;
    using tts_cpp::acestep::EncodedAudio;
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::GenerationConditioning;
    using tts_cpp::acestep::GenerationPlan;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::TimbreInput;
    using tts_cpp::acestep::make_generation_plan;
    using tts_cpp::acestep::prepare_generation_conditioning;
    using tts_cpp::acestep::resolve_generate_task;
    using tts_cpp::acestep::resolve_timbre_input;

    GenerateParams params;
    params.task_type = TASK_COVER_NOFSQ;
    params.source_audio = { 1.0f, 2.0f, 3.0f, 4.0f };
    params.reference_audio = { 5.0f, 6.0f };

    GenerateTask task;
    CHECK(resolve_generate_task(params, task).empty());
    const GenerationPlan plan = make_generation_plan(params, task);

    int source_calls = 0;
    int reference_calls = 0;
    const AudioEncoder encode = [&](const std::vector<float> & pcm, const char * stage, EncodedAudio & output) {
        if (std::string(stage) == "source") ++source_calls;
        if (std::string(stage) == "reference") ++reference_calls;
        output.latent = { pcm.front(), pcm.back() };
        output.frames = (int) (pcm.size() / 2);
        return true;
    };

    GenerationConditioning conditioning;
    CHECK(prepare_generation_conditioning(params, plan, encode, conditioning));
    CHECK(source_calls == 1);
    CHECK(reference_calls == 1);
    CHECK(conditioning.source.latent.front() == params.source_audio.front());
    CHECK(conditioning.reference.latent.front() == params.reference_audio.front());

    const std::vector<float> silence = { -1.0f, -1.0f };
    const TimbreInput explicit_reference =
        resolve_timbre_input(plan, conditioning.reference, conditioning.source.latent,
                             conditioning.source.frames, silence);
    CHECK(explicit_reference.data == conditioning.reference.latent.data());
    CHECK(explicit_reference.frames == conditioning.reference.frames);

    params.reference_audio.clear();
    CHECK(resolve_generate_task(params, task).empty());
    const GenerationPlan reuse_plan = make_generation_plan(params, task);
    GenerationConditioning reuse_conditioning;
    CHECK(prepare_generation_conditioning(params, reuse_plan, encode, reuse_conditioning));
    CHECK(source_calls == 2);
    CHECK(reference_calls == 1);
    const TimbreInput reused_source =
        resolve_timbre_input(reuse_plan, reuse_conditioning.reference, reuse_conditioning.source.latent,
                             reuse_conditioning.source.frames, silence);
    CHECK(reused_source.data == reuse_conditioning.source.latent.data());
    CHECK(reused_source.frames == reuse_conditioning.source.frames);
}

void test_cover_noise_blending() {
    using tts_cpp::acestep::CoverNoiseResult;
    using tts_cpp::acestep::apply_cover_noise;

    std::vector<float> noise = { 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f };
    const std::vector<float> source = { 5.0f, 7.0f, 11.0f, 13.0f };
    std::vector<float> schedule = { 1.0f, 0.75f, 0.5f, 0.25f };

    const CoverNoiseResult result = apply_cover_noise(noise, source, 3, 2, 2, 0.5f, schedule);
    CHECK(approx(result.nearest_time, 0.5f));
    CHECK(result.remaining_steps == 2);
    CHECK(schedule.size() == 2);
    CHECK(approx(noise[0], 3.0f));
    CHECK(approx(noise[1], 4.0f));
    CHECK(approx(noise[2], 6.5f));
    CHECK(approx(noise[3], 7.5f));
    CHECK(approx(noise[4], 7.0f));
    CHECK(approx(noise[5], 8.0f));
}

void fill_fake_pcm(std::vector<float> & pcm, int frames) {
    for (int frame = 0; frame < frames; ++frame) {
        pcm[(size_t) frame * 2] = (float) frame;
        pcm[(size_t) frame * 2 + 1] = (float) -frame;
    }
}

void fill_fake_latent_frame(std::vector<float> & latent, int frame, float value) {
    using tts_cpp::acestep::VAE_LATENT_CHANNELS;
    for (int channel = 0; channel < VAE_LATENT_CHANNELS; ++channel) {
        latent[(size_t) frame * VAE_LATENT_CHANNELS + channel] = value + (float) channel;
    }
}

int encode_fake_vae(const float * pcm, int frames, std::vector<float> & latent) {
    using tts_cpp::acestep::VAE_ENCODER_UPSAMPLE;
    using tts_cpp::acestep::VAE_LATENT_CHANNELS;

    const int latent_frames = frames / VAE_ENCODER_UPSAMPLE;
    latent.resize((size_t) latent_frames * VAE_LATENT_CHANNELS);
    for (int frame = 0; frame < latent_frames; ++frame) {
        const float value = pcm[(size_t) frame * VAE_ENCODER_UPSAMPLE * 2];
        fill_fake_latent_frame(latent, frame, value);
    }
    return latent_frames;
}

void test_vae_encode_window_boundaries() {
    using tts_cpp::acestep::VAE_AUDIO_CHUNK_FRAMES;
    using tts_cpp::acestep::VAE_AUDIO_OVERLAP_FRAMES;
    using tts_cpp::acestep::make_vae_encode_window;
    using tts_cpp::acestep::vae_encode_window_count;

    CHECK(vae_encode_window_count(VAE_AUDIO_CHUNK_FRAMES) == 1);
    CHECK(vae_encode_window_count(VAE_AUDIO_CHUNK_FRAMES + 1) == 2);

    const auto first = make_vae_encode_window(0, VAE_AUDIO_CHUNK_FRAMES + 1);
    const auto second = make_vae_encode_window(1, VAE_AUDIO_CHUNK_FRAMES + 1);
    CHECK(first.window_start == 0);
    CHECK(second.window_start == second.core_start - VAE_AUDIO_OVERLAP_FRAMES);
    CHECK(first.core_end == second.core_start);
}

void test_vae_encode_window_parity() {
    using tts_cpp::acestep::VAE_AUDIO_CHUNK_FRAMES;
    using tts_cpp::acestep::VAE_AUDIO_STRIDE_FRAMES;
    using tts_cpp::acestep::encode_vae_pcm_bounded;

    const int frames = VAE_AUDIO_CHUNK_FRAMES + VAE_AUDIO_STRIDE_FRAMES;
    std::vector<float> pcm((size_t) frames * 2);
    fill_fake_pcm(pcm, frames);

    std::vector<float> expected;
    const int expected_frames = encode_fake_vae(pcm.data(), frames, expected);
    int actual_frames = 0;
    const std::vector<float> actual =
        encode_vae_pcm_bounded(pcm, frames, encode_fake_vae, &actual_frames);

    CHECK(actual_frames == expected_frames);
    CHECK(actual == expected);
}

template <typename T>
void write_test_value(FILE * file, T value) {
    fwrite(&value, sizeof(T), 1, file);
}

void write_test_wav(FILE * file, uint16_t channels, const std::vector<int16_t> & samples) {
    constexpr uint16_t PCM_FORMAT = 1;
    constexpr uint16_t BITS = 16;
    constexpr uint32_t RATE = 48000;
    constexpr uint32_t FORMAT_SIZE = 16;

    const uint16_t block_align = channels * (BITS / 8);
    const uint32_t byte_rate = RATE * block_align;
    const uint32_t data_size = (uint32_t) (samples.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_size;

    fwrite("RIFF", 1, 4, file);
    write_test_value(file, riff_size);
    fwrite("WAVE", 1, 4, file);
    fwrite("fmt ", 1, 4, file);
    write_test_value(file, FORMAT_SIZE);
    write_test_value(file, PCM_FORMAT);
    write_test_value(file, channels);
    write_test_value(file, RATE);
    write_test_value(file, byte_rate);
    write_test_value(file, block_align);
    write_test_value(file, BITS);
    fwrite("data", 1, 4, file);
    write_test_value(file, data_size);
    fwrite(samples.data(), sizeof(int16_t), samples.size(), file);
}

FILE * open_test_file() {
    FILE * file = tmpfile();
    CHECK(file != nullptr);
    return file;
}

void test_wav_reader_mono_and_stereo() {
    using tts_cpp::acestep::WavReadResult;
    using tts_cpp::acestep::read_pcm16_wav;

    FILE * mono = open_test_file();
    if (!mono) return;
    write_test_wav(mono, 1, { 16384, -16384 });
    const WavReadResult mono_result = read_pcm16_wav(mono);
    fclose(mono);
    CHECK(mono_result.error.empty());
    CHECK(mono_result.frames == 2);
    CHECK(approx(mono_result.pcm[0], 0.5f));
    CHECK(approx(mono_result.pcm[1], 0.5f));

    FILE * stereo = open_test_file();
    if (!stereo) return;
    write_test_wav(stereo, 2, { 16384, -16384 });
    const WavReadResult stereo_result = read_pcm16_wav(stereo);
    fclose(stereo);
    CHECK(stereo_result.error.empty());
    CHECK(stereo_result.frames == 1);
    CHECK(approx(stereo_result.pcm[0], 0.5f));
    CHECK(approx(stereo_result.pcm[1], -0.5f));
}

void test_wav_reader_rejects_multichannel() {
    using tts_cpp::acestep::WavReadResult;
    using tts_cpp::acestep::read_pcm16_wav;

    FILE * file = open_test_file();
    if (!file) return;
    write_test_wav(file, 3, { 1, 2, 3 });
    const WavReadResult result = read_pcm16_wav(file);
    fclose(file);
    CHECK(!result.error.empty());
    CHECK(result.pcm.empty());
}

}  // namespace

int main() {
    test_schedule();
    test_haar_dcw();
    test_philox();
    test_fsq();
    test_sampler();
    test_vae_progress();
    test_vae_window_core();
    test_backend_device_types();
    test_stage_placement();
    test_placement_env();
    test_generate_task_kinds();
    test_generate_task_defaults();
    test_generate_task_audio_layout();
    test_generate_task_errors();
    test_generate_task_strengths();
    test_generation_plans();
    test_generation_conditioning();
    test_cover_noise_blending();
    test_vae_encode_window_boundaries();
    test_vae_encode_window_parity();
    test_wav_reader_mono_and_stereo();
    test_wav_reader_rejects_multichannel();

    std::fprintf(stderr, "[test-acestep-units] %d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
