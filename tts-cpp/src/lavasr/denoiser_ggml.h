#pragma once

// LavaSR denoiser neural net as a ggml compute graph (OpenCL / CPU).
//
// GPU twin of the scalar denoiser_net_forward (denoiser_core.cpp): the UL-UNAS
// U-Net (ERB -> conv encoder -> 2x DPGRNN -> conv decoder -> ratio mask), as a
// ggml graph so it can run on the Adreno OpenCL backend.  The STFT/resample +
// chunked overlap-add stay CPU DSP (denoiser.cpp) — only the neural core moves
// here.  The recurrence is BATCHED over the many independent GRU sweeps (see the
// gru_batched note) so each dispatch is a real GEMM, not a tiny mat-vec.

#include "denoiser_core.h"

#include "ggml.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::lavasr {

namespace detail {

// Batched PyTorch GRU (gate order r,z,n; zero init state) over B independent
// sequences that SHARE weights.  Layouts (ggml, ne0 = fastest):
//   x:   [I, B, L]  (I input-feature, B batch of sweeps, L sequence length)
//   Wih: [I, 3H]    (PyTorch weight_ih_l0 [3H, I] loaded verbatim -> ne0=I)
//   Whh: [H, 3H]    (PyTorch weight_hh_l0 [3H, H] -> ne0=H)
//   Bih, Bhh: [3H]
// `reverse` processes the sequence backward but writes each step's hidden at its
// ORIGINAL position, so a BiGRU is concat(forward, reverse) along ne0.
// Returns y: [H, B, L], y[:, :, t] = hidden after step t.
ggml_tensor * gru_batched(ggml_context * ctx, ggml_tensor * x, ggml_tensor * Wih,
                          ggml_tensor * Whh, ggml_tensor * Bih, ggml_tensor * Bhh,
                          bool reverse);

// Grouped/depthwise 2-D conv, causal in time (pad top kt-1, none bottom),
// symmetric freq pad, freq stride only (time stride 1).  Mirrors the scalar
// Runner::conv2d.  Layouts (ggml, ne0 fastest):
//   x: [F, T, Cin]         (T3-space: ne0=freq, ne1=time, ne2=channel)
//   W: [kf, kt, Cin/g, Cout]  (PyTorch [Cout,Cin/g,kt,kf] loaded verbatim)
//   bias: [Cout] or null
// returns [Fout, T, Cout], Fout = (F + 2*pad_f - kf)/stride_f + 1.
ggml_tensor * conv2d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * W,
                     ggml_tensor * bias, int stride_f, int pad_f, int groups);

// Transposed 2-D conv (decoder freq-upsampler), causal time.  Decomposed as
// zero-insert freq-upsample by stride_f then the verified conv2d with the causal
// kernel: exact to the scalar Runner::conv_transpose2d.  Wc is the PRE-REINDEXED
// regular-conv kernel [kf, kt, Cin/g, Cout] (host IC<->OC swap + kt/kf flip of the
// PyTorch transpose kernel [Cin,Cout/g,kt,kf]).  pad_f = the scalar transpose pad
// (kf/2).  x:[F,T,Cin] -> [(F-1)*stride_f + kf - 2*pad_f, T, Cout].
ggml_tensor * conv_transpose2d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * Wc,
                               ggml_tensor * bias, int stride_f, int pad_f, int groups);

// Resolves a weight tensor by name; returns nullptr if absent (mirrors Runner::Wopt).
using WResolver = std::function<ggml_tensor *(const std::string &)>;

// Dual-path grouped RNN bottleneck (denoiser_core.cpp dpgrnn).  x:[F,T,C] ->
// [F,T,C].  intra = grouped BiGRU over freq (batch=time) + FC + LayerNorm-(F,C) +
// residual; inter = grouped uni-GRU over time (batch=freq) + FC + LN + residual.
// The many independent sweeps batch into the GRU's B dim (see gru_batched).
ggml_tensor * dpgrnn(ggml_context * ctx, ggml_tensor * x, const std::string & prefix,
                     const WResolver & W, float ln_eps);

// Causal time-frequency attention (denoiser_core.cpp ctfa).  x:[F,T,C] -> [F,T,C].
// TA: energy-mean over freq -> GRU over time -> FC -> sigmoid (temporal gate);
// FA: energy-mean over chan -> fold freq by r -> BiGRU -> FC -> sigmoid (freq gate);
// out = at(temporal) * x * af(frequency).
ggml_tensor * ctfa(ggml_context * ctx, ggml_tensor * x, const std::string & prefix,
                   const WResolver & W, int freq_comp_ratio);

} // namespace detail

class DenoiserGgml {
public:
    // Build the backend + upload weights.  n_gpu_layers: >0 = GPU (Adreno OpenCL,
    // CPU fallback), <0 = ggml-CPU (the CPU counterpart of the GPU graph).  Throws.
    static std::unique_ptr<DenoiserGgml> create(const DenoiserWeights & w,
                                                int n_gpu_layers, bool verbose = false);
    ~DenoiserGgml();
    DenoiserGgml(const DenoiserGgml &)             = delete;
    DenoiserGgml & operator=(const DenoiserGgml &) = delete;

    // Same contract as denoiser_net_forward: real/imag [L*spec_bins] (t*F+f) in
    // and out, one zero-state chunk of L frames.
    void chunk_forward(const std::vector<float> & real_in, const std::vector<float> & imag_in,
                       int L, std::vector<float> & real_out, std::vector<float> & imag_out);

    // Batched form: n_chunks independent zero-state chunks stacked as [n_chunks][L*F]
    // (chunk c at offset c*L*F), run in ONE graph (chunk-batch in ne3) so the GPU
    // dispatch cost is paid once, not per chunk.  Output same layout.
    void batch_forward(const std::vector<float> & real_in, const std::vector<float> & imag_in,
                       int L, int n_chunks, std::vector<float> & real_out, std::vector<float> & imag_out);

    bool         is_gpu() const;
    const char * backend_name() const;

private:
    DenoiserGgml();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::lavasr
