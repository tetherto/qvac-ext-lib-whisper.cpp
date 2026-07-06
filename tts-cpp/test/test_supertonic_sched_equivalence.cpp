// Pinning test for the Supertonic dispatch after its migration onto
// src/sched_dispatch.{h,cpp}.
//
// Supertonic's dispatch gate (supertonic_use_sched) deliberately does NOT
// honor TTS_CPP_FORCE_SCHED: Supertonic reuses its sched-routed cached
// graphs across calls, and forcing the scheduler onto a walk-passing
// backend was measured to corrupt the second use of a freshly-built
// cached graph on CPU (first use bit-identical, every reuse diverges,
// deterministic) and to crash on Metal (the f16-KV flash-attn graph built
// for Metal lands partially on the CPU backend).  T3/HiFT are force-safe
// only because they rebuild the graph per sched pass.
//
// This test pins BOTH properties that keep that decision safe:
//   A  — direct synth
//   B  — TTS_CPP_FORCE_SCHED=1: Supertonic must IGNORE the flag; output
//        must stay bit-identical to A.  If someone wires the flag into
//        supertonic_use_sched without first making reuse-through-sched
//        safe, this fails (or crashes) immediately.
//   A' — direct again: determinism / no state poisoning across engines.
// A fresh Engine per phase (generation_id bump rebuilds every
// thread_local cache).  PCM compared byte-for-byte (memcmp); bit-exactness
// is the bar — do NOT relax this to a numeric tolerance.
//
// usage: test-supertonic-sched-equivalence MODEL.gguf [n_gpu_layers]

#include "tts-cpp/supertonic/engine.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK_MSG(cond, ...) do {                                     \
    ++g_checks;                                                       \
    if (!(cond)) {                                                    \
        ++g_failures;                                                 \
        std::fprintf(stderr, "FAIL %s:%d  %s  ", __FILE__, __LINE__, #cond); \
        std::fprintf(stderr, __VA_ARGS__);                            \
        std::fprintf(stderr, "\n");                                   \
    }                                                                 \
} while (0)

const char * kText = "The quick brown fox jumps over the lazy dog.";

bool run_phase(const std::string & gguf, int n_gpu_layers, std::vector<float> & pcm,
               const char * label) {
    try {
        tts_cpp::supertonic::EngineOptions opts;
        opts.model_gguf_path = gguf;
        opts.seed            = 42;
        opts.n_threads       = 2;
        opts.n_gpu_layers    = n_gpu_layers;
        tts_cpp::supertonic::Engine engine(opts);
        pcm = engine.synthesize(kText).pcm;
        return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "phase %s: %s\n", label, e.what());
        return false;
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [n_gpu_layers]\n", argv[0]);
        return 2;
    }
    const std::string gguf = argv[1];
    const int n_gpu_layers = (argc > 2) ? std::atoi(argv[2]) : 0;

    std::vector<float> pcm_a, pcm_b, pcm_a2;

    unsetenv("TTS_CPP_FORCE_SCHED");
    CHECK_MSG(run_phase(gguf, n_gpu_layers, pcm_a, "A (direct)"), "phase A failed");

    setenv("TTS_CPP_FORCE_SCHED", "1", 1);
    CHECK_MSG(run_phase(gguf, n_gpu_layers, pcm_b, "B (sched)"), "phase B failed");
    unsetenv("TTS_CPP_FORCE_SCHED");

    CHECK_MSG(run_phase(gguf, n_gpu_layers, pcm_a2, "A' (direct)"), "phase A' failed");

    if (g_failures == 0) {
        bool any_signal = false, any_nan = false;
        for (float v : pcm_a) {
            if (std::fabs(v) > 1e-4f) any_signal = true;
            if (std::isnan(v))        any_nan = true;
        }
        CHECK_MSG(!pcm_a.empty(), "phase A produced no PCM");
        CHECK_MSG(any_signal, "phase A PCM is all-zero/near-silent");
        CHECK_MSG(!any_nan, "phase A PCM contains NaN");

        CHECK_MSG(pcm_a.size() == pcm_b.size(),
                  "A vs B sample count %zu != %zu", pcm_a.size(), pcm_b.size());
        CHECK_MSG(pcm_a.size() == pcm_a2.size(),
                  "A vs A' sample count %zu != %zu", pcm_a.size(), pcm_a2.size());
        if (pcm_a.size() == pcm_b.size() && !pcm_a.empty()) {
            const bool ab_same = std::memcmp(pcm_a.data(), pcm_b.data(),
                                             pcm_a.size() * sizeof(float)) == 0;
            CHECK_MSG(ab_same, "TTS_CPP_FORCE_SCHED leaked into Supertonic dispatch "
                               "(output changed; reuse-through-sched is UNSAFE here)");
            if (!ab_same) {
                size_t first = pcm_a.size();
                size_t n_diff = 0;
                float  max_abs = 0.0f;
                for (size_t i = 0; i < pcm_a.size(); ++i) {
                    if (pcm_a[i] != pcm_b[i]) {
                        if (first == pcm_a.size()) first = i;
                        ++n_diff;
                        const float d = std::fabs(pcm_a[i] - pcm_b[i]);
                        if (d > max_abs) max_abs = d;
                    }
                }
                std::fprintf(stderr,
                             "  A/B divergence: first at sample %zu/%zu, %zu samples differ, max |d|=%g\n",
                             first, pcm_a.size(), n_diff, (double) max_abs);
            }
        }
        if (pcm_a.size() == pcm_a2.size() && !pcm_a.empty()) {
            CHECK_MSG(std::memcmp(pcm_a.data(), pcm_a2.data(),
                                  pcm_a.size() * sizeof(float)) == 0,
                      "direct pre- vs post-sched PCM NOT bit-identical");
        }
    }

    std::fprintf(stderr, "%s: %d/%d checks passed\n",
                 argv[0], g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
