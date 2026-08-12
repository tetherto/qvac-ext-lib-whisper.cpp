// Parity test — CosyVoice3 flow (stage 4): speech tokens -> mel.
//
// Runs the shared cosyvoice_flow_run() on reference prompt/token inputs and
// asserts the produced mel matches the PyTorch reference (flow_mel.npy) at
// cosine >= threshold. Exits non-zero on failure (registered as a ctest,
// auto-disabled when the fixtures below are absent).
//
// Fixtures (generate with scripts/dump-cosyvoice3-reference.py):
//   <in-dir>/{prompt_token,speech_tokens,prompt_feat,embedding,flow_mel}.npy
//
// Usage:
//   test-cosyvoice-flow --flow-gguf FLOW.gguf --in-dir DIR [--min-cosine 0.99]
//
// Set COSYVOICE_TEST_GPU=1 to run the same parity check on the selected GPU
// backend (mirrors PARLER_TEST_GPU); fails if no GPU backend initializes.
// The GPU run additionally computes the mel on CPU in-process and gates the
// two against each other: cosine >= --xb-min-cosine AND max absolute
// difference <= --xb-max-abs (log-mel units). The absolute bound exists
// because cosine is scale-invariant -- it would accept a uniformly scaled
// mel that shifts the vocoder's output level. Same-implementation legs sit
// far tighter than the PyTorch-reference cosine, so this is the gate that
// actually catches a backend numerics regression (e.g. an f16-accumulating
// matmul rewrite) in the matmul-heavy DiT.

#include "npy.h"
#include "backend_selection.h"
#include "cosyvoice_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string gguf, in_dir;
    double min_cosine = 0.99;
    double xb_min_cosine = 0.9995;
    double xb_max_abs = 0.5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--flow-gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--in-dir" && i + 1 < argc) in_dir = argv[++i];
        else if (a == "--min-cosine" && i + 1 < argc) min_cosine = std::atof(argv[++i]);
        else if (a == "--xb-min-cosine" && i + 1 < argc) xb_min_cosine = std::atof(argv[++i]);
        else if (a == "--xb-max-abs" && i + 1 < argc) xb_max_abs = std::atof(argv[++i]);
        else { fprintf(stderr, "usage: %s --flow-gguf FLOW.gguf --in-dir DIR [--min-cosine 0.99]\n"
                               "          [--xb-min-cosine 0.9995] [--xb-max-abs 0.5]\n", argv[0]); return 2; }
    }
    if (gguf.empty() || in_dir.empty()) { fprintf(stderr, "missing --flow-gguf / --in-dir\n"); return 2; }

    const int MEL = 80;
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

    npy_array ptok_a  = npy_load(in_dir + "/prompt_token.npy");
    npy_array stok_a  = npy_load(in_dir + "/speech_tokens.npy");
    npy_array pfeat_a = npy_load(in_dir + "/prompt_feat.npy");
    npy_array emb_a   = npy_load(in_dir + "/embedding.npy");
    npy_array ref_a   = npy_load(in_dir + "/flow_mel.npy");

    int T_ptok = (int)ptok_a.shape[0], T_stok = (int)stok_a.shape[0];
    int mel_len1 = (int)pfeat_a.shape[0];
    std::vector<int> ptok(T_ptok), stok(T_stok);
    { const int32_t * p = npy_as_i32(ptok_a); for (int i = 0; i < T_ptok; ++i) ptok[i] = p[i]; }
    { const int32_t * p = npy_as_i32(stok_a); for (int i = 0; i < T_stok; ++i) stok[i] = p[i]; }
    std::vector<float> pfeat(npy_as_f32(pfeat_a), npy_as_f32(pfeat_a) + (size_t)mel_len1 * MEL);
    std::vector<float> emb(npy_as_f32(emb_a), npy_as_f32(emb_a) + (size_t)emb_a.shape[0]);

    int mel_len2 = 0;
    std::vector<float> mel = cosyvoice_flow_run(m, ptok, stok, pfeat, mel_len1, emb, mel_len2);

    if (backend) {
        // Cross-backend leg: same implementation on CPU, so the two mels must
        // agree far more tightly than either agrees with the PyTorch fixture.
        model_ctx m_cpu = cosyvoice_load_gguf(gguf);
        int mel_len2_cpu = 0;
        std::vector<float> mel_cpu =
            cosyvoice_flow_run(m_cpu, ptok, stok, pfeat, mel_len1, emb, mel_len2_cpu);
        if (mel_len2_cpu != mel_len2 || mel_cpu.size() != mel.size()) {
            fprintf(stderr, "FAIL: GPU mel shape differs from CPU (gpu %d frames, cpu %d frames)\n",
                    mel_len2, mel_len2_cpu);
            return 1;
        }
        double dot = 0, na = 0, nb = 0, max_abs = 0;
        for (size_t i = 0; i < mel.size(); ++i) {
            const double a = mel[i], b = mel_cpu[i];
            dot += a * b; na += a * a; nb += b * b;
            max_abs = std::max(max_abs, std::fabs(a - b));
        }
        const double xb_cosine = dot / (std::sqrt(na) * std::sqrt(nb));
        fprintf(stderr, "gpu-vs-cpu: cosine = %.6f (threshold %.4f)  max|diff| = %.4f (bound %.4f)\n",
                xb_cosine, xb_min_cosine, max_abs, xb_max_abs);
        if (!(xb_cosine >= xb_min_cosine) || !(max_abs <= xb_max_abs)) {
            fprintf(stderr, "FAIL: GPU mel diverges from CPU\n");
            return 1;
        }
    }

    if (mel.size() != (size_t)MEL * mel_len2 || (int)ref_a.shape[0] != MEL) {
        fprintf(stderr, "FAIL: shape mismatch (got %zu, ref [%lld,%lld])\n",
                mel.size(), (long long)ref_a.shape[0], (long long)ref_a.shape[1]);
        return 1;
    }
    size_t n = std::min(mel.size(), (size_t)MEL * (size_t)ref_a.shape[1]);
    const float * e = npy_as_f32(ref_a);
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) { const double a = mel[i], b = e[i]; dot += a * b; na += a * a; nb += b * b; }
    double cosine = dot / (std::sqrt(na) * std::sqrt(nb));

    fprintf(stderr, "flow_mel cosine = %.6f  (threshold %.4f, %d frames)\n", cosine, min_cosine, mel_len2);
    if (!(cosine >= min_cosine)) { fprintf(stderr, "FAIL: flow_mel cosine below threshold\n"); return 1; }
    fprintf(stderr, "PASS\n");
    return 0;
}
