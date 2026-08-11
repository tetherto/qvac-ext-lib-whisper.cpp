// Parity test — CosyVoice3 CausalHiFT vocoder: mel -> 24 kHz waveform.
//
// Runs cosyvoice_hift_synth() on a reference mel and asserts the waveform
// correlates with the PyTorch reference (hift_wav.npy) above a threshold.
// HiFT's NSF source excitation is partly stochastic (unvoiced noise, different
// RNG than the reference), so an exact match is not expected — the gate is a
// Pearson correlation, which caught the earlier metallic phase bug
// (0.136 -> 0.996 after the fix). Exits non-zero on failure; registered as a
// ctest, auto-disabled when fixtures are absent.
//
// Fixtures (generate with scripts/dump-cosyvoice3-reference.py):
//   <in-dir>/{hift_mel_in,hift_wav}.npy
//
// Usage:
//   test-cosyvoice-hift --hift-gguf HIFT.gguf --in-dir DIR [--min-corr 0.90]
//
// Set COSYVOICE_TEST_GPU=1 to run the check on the selected GPU backend
// (mirrors PARLER_TEST_GPU).  The GPU run gates in two steps: SineGen2
// integrates f0 into sine phases, so sub-Hz cross-backend f0 noise
// decorrelates the raw waveform with no audible effect.  (1) GPU f0 must
// match CPU f0 at cosine >= --min-f0-cosine; (2) the GPU synth runs with
// the CPU f0 pinned and faces the same --min-corr waveform gate.

#include "npy.h"
#include "backend_selection.h"
#include "cosyvoice_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double pearson(const float * a, const float * b, size_t n) {
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= (double)n; mb /= (double)n;
    double cov = 0, va = 0, vb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        cov += da * db; va += da * da; vb += db * db;
    }
    return cov / (std::sqrt(va) * std::sqrt(vb));
}

static double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

int main(int argc, char ** argv) {
    std::string gguf, in_dir;
    double min_corr = 0.90;
    double min_f0_cosine = 0.9999;
    int seed = 42;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--hift-gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--in-dir" && i + 1 < argc) in_dir = argv[++i];
        else if (a == "--min-corr" && i + 1 < argc) min_corr = std::atof(argv[++i]);
        else if (a == "--min-f0-cosine" && i + 1 < argc) min_f0_cosine = std::atof(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s --hift-gguf HIFT.gguf --in-dir DIR [--min-corr 0.90] [--min-f0-cosine 0.9999]\n", argv[0]); return 2; }
    }
    if (gguf.empty() || in_dir.empty()) { fprintf(stderr, "missing --hift-gguf / --in-dir\n"); return 2; }

    const int MEL = 80;
    ggml_backend_t backend = nullptr;
    if (std::getenv("COSYVOICE_TEST_GPU")) {
        backend = ::tts_cpp::detail::init_gpu_backend(99, /*verbose=*/false, "test-cosyvoice");
        if (!backend) { fprintf(stderr, "FAIL: COSYVOICE_TEST_GPU set but no GPU backend\n"); return 1; }
    }
    model_ctx m = cosyvoice_load_gguf(gguf, backend);

    // mel npy shape (80, T) -> channel-major flat[ch*T + t].
    npy_array mel_a = npy_load(in_dir + "/hift_mel_in.npy");
    int T_mel = (int)mel_a.shape[1];
    std::vector<float> mel((size_t)T_mel * MEL);
    std::memcpy(mel.data(), npy_as_f32(mel_a), mel.size() * sizeof(float));

    npy_array ref_a = npy_load(in_dir + "/hift_wav.npy");
    size_t ref_n = 1;
    for (size_t d = 0; d < ref_a.shape.size(); ++d) ref_n *= (size_t)ref_a.shape[d];

    std::vector<float> wav;
    if (backend) {
        model_ctx m_cpu = cosyvoice_load_gguf(gguf);
        std::vector<float> f0_cpu = cosyvoice_hift_f0(m_cpu, mel, T_mel);
        std::vector<float> f0_gpu = cosyvoice_hift_f0(m, mel, T_mel);
        double f0_cos = cosine(f0_cpu, f0_gpu);
        fprintf(stderr, "f0 cpu-vs-gpu cosine = %.6f  (threshold %.4f, %d frames)\n",
                f0_cos, min_f0_cosine, T_mel);
        if (!(f0_cos >= min_f0_cosine)) {
            fprintf(stderr, "FAIL: GPU f0 predictor diverged from CPU\n");
            return 1;
        }
        wav = cosyvoice_hift_synth(m, mel, T_mel, seed, nullptr, &f0_cpu);
    } else {
        wav = cosyvoice_hift_synth(m, mel, T_mel, seed);
    }

    size_t n = std::min(wav.size(), ref_n);
    if (n == 0) { fprintf(stderr, "FAIL: empty waveform\n"); return 1; }
    double corr = pearson(wav.data(), npy_as_f32(ref_a), n);

    fprintf(stderr, "hift waveform corr = %.6f  (threshold %.4f, got=%zu ref=%zu)\n",
            corr, min_corr, wav.size(), ref_n);
    if (!(corr >= min_corr)) { fprintf(stderr, "FAIL: hift waveform correlation below threshold\n"); return 1; }
    fprintf(stderr, "PASS\n");
    return 0;
}
