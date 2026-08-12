// Pins cosyvoice_conv1d_f32's non-contiguous-input guard. The builder must
// ggml_cont a strided view before im2col: ggml-vulkan's IM2COL supports_op
// requires a contiguous signal, so without the guard a view input demotes
// the whole stage to the sched-fallback path. Deviceless on CPU (a permuted
// view must produce bit-identical output to a pre-contiguous input); with
// COSYVOICE_TEST_GPU=1 it additionally requires the built graph to be fully
// supported by the selected GPU backend (no scheduler fallback) and the GPU
// output to match the CPU reference.

#include "backend_selection.h"
#include "cosyvoice_pipeline.h"
#include "sched_dispatch.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kKernel = 3, kCin = 4, kCout = 2, kLen = 17;

// Deterministic fills so the two legs and both backends see identical bytes.
std::vector<float> make_signal() {
    std::vector<float> v((size_t)kCin * kLen);
    for (size_t i = 0; i < v.size(); ++i) v[i] = std::sin(0.37f * (float)i) + 0.11f * (float)(i % 5);
    return v;
}
std::vector<float> make_weights() {
    std::vector<float> v((size_t)kKernel * kCin * kCout);
    for (size_t i = 0; i < v.size(); ++i) v[i] = std::cos(0.23f * (float)i) - 0.07f * (float)(i % 3);
    return v;
}

bool close_enough(float a, float b) {
    return std::fabs(a - b) <= 1e-4f + 1e-4f * std::fabs(b);
}

// Builds conv over either a contiguous copy of the signal or a strided
// permute view of it, runs on `backend`, and reports whether the graph was
// fully supported there (i.e. needed no sched fallback).
std::vector<float> run_conv(ggml_backend_t backend, bool use_view, bool & fully_supported) {
    static const size_t buf_size = 4 * 1024 * 1024;
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    // Signal staged transposed [Cin, Len]; the permute view flips it to the
    // conv layout [Len, Cin, 1] WITHOUT making it contiguous.
    ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kCin, kLen);
    ggml_set_name(xt, "xt"); ggml_set_input(xt);
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kKernel, kCin, kCout);
    ggml_set_name(w, "w"); ggml_set_input(w);

    ggml_tensor * x = ggml_permute(ctx, xt, 1, 0, 2, 3);          // [Len, Cin, 1] strided view
    if (!use_view) x = ggml_cont(ctx, x);                          // pre-contiguous reference leg
    ggml_tensor * y = cosyvoice_conv1d_f32(ctx, w, x, 1, 0, 1);
    ggml_set_name(y, "y"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    fully_supported = ::tts_cpp::detail::graph_fully_supported(backend, gf);

    auto * allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);
    const auto sig = make_signal();
    const auto wts = make_weights();
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "xt"), sig.data(), 0, sig.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w"),  wts.data(), 0, wts.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);
    ggml_tensor * out = ggml_graph_get_tensor(gf, "y");
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return res;
}

int compare(const std::vector<float> & got, const std::vector<float> & ref, const char * what) {
    if (got.size() != ref.size()) {
        fprintf(stderr, "FAIL: %s size mismatch (%zu vs %zu)\n", what, got.size(), ref.size());
        return 1;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!close_enough(got[i], ref[i])) {
            fprintf(stderr, "FAIL: %s mismatch @ %zu: got=%.6g ref=%.6g\n", what, i, got[i], ref[i]);
            return 1;
        }
    }
    return 0;
}

} // namespace

int main() {
    ggml_backend_t cpu = ::tts_cpp::detail::init_cpu_backend();
    if (!cpu) { fprintf(stderr, "FAIL: no CPU backend\n"); return 1; }

    bool supported = false;
    const auto ref_cont = run_conv(cpu, /*use_view=*/false, supported);
    const auto cpu_view = run_conv(cpu, /*use_view=*/true, supported);
    // Same dot products in the same order once the guard conts the view, so
    // the CPU comparison is exact.
    if (cpu_view != ref_cont) {
        fprintf(stderr, "FAIL: CPU view leg differs from pre-contiguous leg\n");
        return 1;
    }

    if (std::getenv("COSYVOICE_TEST_GPU")) {
        ggml_backend_t gpu = ::tts_cpp::detail::init_gpu_backend(
            99, /*verbose=*/false, "test-cosyvoice", /*vulkan_device=*/0,
            /*allow_arm_mali=*/false, /*out_gpu_present_but_unused=*/nullptr,
            cosyvoice_gpu_requirement());
        if (!gpu) { fprintf(stderr, "FAIL: COSYVOICE_TEST_GPU set but no GPU backend\n"); return 1; }
        bool gpu_supported = false;
        const auto gpu_view = run_conv(gpu, /*use_view=*/true, gpu_supported);
        if (!gpu_supported) {
            fprintf(stderr, "FAIL: view-input conv graph not fully supported on %s "
                            "(guard regressed; stage would demote to sched fallback)\n",
                    ggml_backend_name(gpu));
            ggml_backend_free(gpu);
            ggml_backend_free(cpu);
            return 1;
        }
        const int rc = compare(gpu_view, ref_cont, "GPU view leg vs CPU");
        ggml_backend_free(gpu);
        if (rc) { ggml_backend_free(cpu); return 1; }
    }

    ggml_backend_free(cpu);
    fprintf(stderr, "PASS\n");
    return 0;
}
