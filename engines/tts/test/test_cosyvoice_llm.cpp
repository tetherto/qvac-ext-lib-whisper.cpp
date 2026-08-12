// Parity test — CosyVoice3 LLM (stage 5): text ids -> speech tokens.
//
// Forces greedy (argmax) decoding and asserts the generated speech-token
// trajectory matches the PyTorch reference (gen_tokens.npy) exactly. Greedy
// makes the LM deterministic so the C++ decode can be validated bit-for-bit
// (the shipped RAS sampling is stochastic). Exits non-zero on failure;
// registered as a ctest, auto-disabled when fixtures are absent.
//
// Fixtures (generate with scripts/dump-cosyvoice3-llm-reference.py):
//   <in-dir>/{text_ids,prompt_stok,gen_tokens}.npy
//
// Usage:
//   test-cosyvoice-llm --llm-gguf LLM.gguf --in-dir DIR [--min-match 0.99]
//
// Set COSYVOICE_TEST_GPU=1 to run the same parity check on the selected GPU
// backend (mirrors PARLER_TEST_GPU); fails if no GPU backend initializes.
// The GPU run additionally generates on CPU in-process and requires the GPU
// trajectory (tokens and stop) to be identical -- unconditionally on Metal
// and OpenCL, and by default everywhere (--xb-min-match / --xb-max-len-diff
// can relax the non-exact backends from measured data). EOS-position parity
// is enforceable only cross-backend: even the CPU leg does not reproduce the
// PyTorch reference's stop step (near-tie logits there), so the reference
// gate stays a prefix-match while the cross-backend gate catches a backend
// that keeps generating past the stop the CPU implementation produces.

#include "npy.h"
#include "backend_selection.h"
#include "backend_util.h"
#include "cosyvoice_pipeline.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Speech-token vocab is [0, 6561); anything >= this is an EOS / special marker
// that the C++ generator consumes as the stop signal (and does not emit).
static const int SPEECH_TOKEN_MAX = 6561;

int main(int argc, char ** argv) {
    std::string gguf, in_dir;
    double min_match = 0.99;
    double xb_min_match = 1.0;
    int xb_max_len_diff = 0;
    int seed = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--llm-gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--in-dir" && i + 1 < argc) in_dir = argv[++i];
        else if (a == "--min-match" && i + 1 < argc) min_match = std::atof(argv[++i]);
        else if (a == "--xb-min-match" && i + 1 < argc) xb_min_match = std::atof(argv[++i]);
        else if (a == "--xb-max-len-diff" && i + 1 < argc) xb_max_len_diff = std::atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s --llm-gguf LLM.gguf --in-dir DIR [--min-match 0.99]\n"
                               "          [--xb-min-match 1.0] [--xb-max-len-diff 0]\n", argv[0]); return 2; }
    }
    if (gguf.empty() || in_dir.empty()) { fprintf(stderr, "missing --llm-gguf / --in-dir\n"); return 2; }

    ggml_backend_t backend = nullptr;
    if (std::getenv("COSYVOICE_TEST_GPU")) {
        // Same requirement as the production engine (cosyvoice_engine.cpp), so
        // on a mixed-backend host the harness validates a backend the engine
        // would actually select instead of whichever device sorts first.
        backend = ::tts_cpp::detail::init_gpu_backend(
            99, /*verbose=*/false, "test-cosyvoice", /*vulkan_device=*/0,
            /*allow_arm_mali=*/false, /*out_gpu_present_but_unused=*/nullptr,
            cosyvoice_gpu_requirement());
        if (!backend) { fprintf(stderr, "FAIL: COSYVOICE_TEST_GPU set but no GPU backend\n"); return 1; }
    }
    model_ctx m = cosyvoice_load_gguf(gguf, backend);
    qwen_hp hp = cosyvoice_qwen_hp(m);  // exercise the GGUF-KV hp path

    npy_array tid_a = npy_load(in_dir + "/text_ids.npy");
    npy_array ps_a  = npy_load(in_dir + "/prompt_stok.npy");
    npy_array ref_a = npy_load(in_dir + "/gen_tokens.npy");

    std::vector<int> text_ids((int)tid_a.shape[0]);
    { const int32_t * p = npy_as_i32(tid_a); for (size_t i = 0; i < text_ids.size(); ++i) text_ids[i] = p[i]; }
    std::vector<int> prompt_stok(ps_a.shape.empty() ? 0 : (int)ps_a.shape[0]);
    { const int32_t * p = npy_as_i32(ps_a); for (size_t i = 0; i < prompt_stok.size(); ++i) prompt_stok[i] = p[i]; }

    // Reference greedy tokens, truncated at the first EOS/special marker.
    const int32_t * rp = npy_as_i32(ref_a);
    std::vector<int> ref;
    for (size_t i = 0; i < (size_t)ref_a.shape[0]; ++i) {
        if (rp[i] >= SPEECH_TOKEN_MAX) break;
        ref.push_back(rp[i]);
    }
    if (ref.empty()) { fprintf(stderr, "FAIL: reference has no speech tokens\n"); return 1; }

    const int max_steps = (int)ref.size() + 16;
    std::vector<int> got = cosyvoice_llm_generate(
        m, hp, text_ids, prompt_stok, max_steps, /*greedy=*/true, seed, /*min_len=*/0);

    if (backend) {
        // Cross-backend parity against an in-process CPU run. Guards the case
        // the reference prefix gate cannot see: a GPU backend that reproduces
        // the prefix but stops differently (e.g. never emits EOS where the
        // CPU implementation does). Metal and OpenCL are gated on exact
        // equality unconditionally: their greedy trajectories measured
        // bit-identical to CPU (tokens and stop), so any difference is a
        // regression, not tolerance-worthy noise. Backends whose reduction
        // order may legitimately flip a near-tie argmax take the
        // --xb-min-match / --xb-max-len-diff knobs instead -- the defaults
        // (1.0 / 0) still demand exact equality, so relaxation is an explicit
        // per-invocation decision backed by measured data, never a baked-in
        // tolerance. The length bound keeps the gate's original purpose
        // (catching EOS runaway) intact under any relaxation.
        model_ctx m_cpu = cosyvoice_load_gguf(gguf);
        qwen_hp hp_cpu = cosyvoice_qwen_hp(m_cpu);
        std::vector<int> got_cpu = cosyvoice_llm_generate(
            m_cpu, hp_cpu, text_ids, prompt_stok, max_steps, /*greedy=*/true, seed, /*min_len=*/0);
        size_t first_diff = 0;
        while (first_diff < std::min(got.size(), got_cpu.size()) &&
               got[first_diff] == got_cpu[first_diff]) {
            ++first_diff;
        }
        const size_t n_xb = std::min(got.size(), got_cpu.size());
        size_t match_xb = 0;
        for (size_t i = 0; i < n_xb; ++i) if (got[i] == got_cpu[i]) ++match_xb;
        const double ratio_xb = n_xb ? (double)match_xb / (double)n_xb : 0.0;
        const size_t len_diff = std::max(got.size(), got_cpu.size()) - n_xb;
        const bool exact_required =
            ::tts_cpp::detail::backend_is_metal(backend) ||
            ::tts_cpp::detail::backend_is_opencl(backend);
        fprintf(stderr, "gpu-vs-cpu: gpu=%zu cpu=%zu tokens  match=%zu/%zu (%.4f)  len-diff=%zu  gate=%s\n",
                got.size(), got_cpu.size(), match_xb, n_xb, ratio_xb, len_diff,
                exact_required ? "exact" : "parameterized");
        const bool xb_fail = exact_required
            ? (got != got_cpu)
            : (!(ratio_xb >= xb_min_match) || len_diff > (size_t)xb_max_len_diff);
        if (xb_fail) {
            fprintf(stderr,
                    "FAIL: GPU greedy trajectory diverges from CPU "
                    "(first difference at index %zu)\n",
                    first_diff);
            return 1;
        }
    }

    size_t n = std::min(got.size(), ref.size());
    size_t match = 0;
    for (size_t i = 0; i < n; ++i) if (got[i] == ref[i]) ++match;
    double ratio = n ? (double)match / (double)n : 0.0;

    fprintf(stderr, "greedy tokens: got=%zu ref=%zu  match=%zu/%zu (%.4f, threshold %.4f)\n",
            got.size(), ref.size(), match, n, ratio, min_match);
    if (got.size() < ref.size())
        fprintf(stderr, "  note: generated fewer tokens than reference\n");
    if (!(ratio >= min_match) || n < ref.size() * min_match) {
        fprintf(stderr, "FAIL: greedy token trajectory diverges from reference\n");
        return 1;
    }
    fprintf(stderr, "PASS\n");
    return 0;
}
