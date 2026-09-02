// The fused LSTM cell op must reproduce the decomposed per-gate decoder graph exactly:
// decode the same utterance with both runtimes on the requested backend and compare tokens.
//   test-tdt-lstm-parity <tdt.gguf> <wav> [--n-gpu-layers N] [--backends-dir DIR]
#include "parakeet_ctc.h"
#include "parakeet_tdt.h"
#include "mel_preprocess.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int decode_with(const parakeet::ParakeetCtcModel & model,
                const parakeet::EncoderOutputs & enc,
                bool allow_fused,
                std::vector<int32_t> & out_tokens,
                bool & fused_used) {
    using namespace parakeet;
    TdtRuntimeWeights rt;
    if (int rc = tdt_prepare_runtime(model, rt, allow_fused); rc != 0) return 150 + rc;
    fused_used = rt.fused_lstm_cell;
    TdtDecodeOptions dopts;
    TdtDecodeResult  dres;
    if (int rc = tdt_greedy_decode(model, rt, enc.encoder_out.data(),
                                   enc.n_enc_frames, enc.d_model, dopts, dres); rc != 0) return 160 + rc;
    out_tokens = std::move(dres.token_ids);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    using namespace parakeet;
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <tdt.gguf> <wav> [--n-gpu-layers N]\n", argv[0]);
        return 2;
    }
    int n_gpu_layers = 0;
    for (int i = 3; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--n-gpu-layers") == 0) n_gpu_layers = std::atoi(argv[i + 1]);
        if (std::strcmp(argv[i], "--backends-dir") == 0) set_backends_directory(argv[i + 1]);
    }

    ParakeetCtcModel model;
    if (int rc = load_from_gguf(argv[1], model, /*n_threads=*/0, n_gpu_layers, /*verbose=*/false); rc != 0) {
        std::fprintf(stderr, "[tdt-lstm-parity] load_from_gguf rc=%d\n", rc);
        return 1;
    }
    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(argv[2], samples, sr); rc != 0) {
        std::fprintf(stderr, "[tdt-lstm-parity] load_wav rc=%d\n", rc);
        return 1;
    }
    std::vector<float> mel;
    int n_frames = 0;
    if (int rc = compute_log_mel(samples.data(), (int) samples.size(), model.mel_cfg, mel, n_frames); rc != 0) {
        std::fprintf(stderr, "[tdt-lstm-parity] compute_log_mel rc=%d\n", rc);
        return 1;
    }
    EncoderOutputs enc;
    if (int rc = run_encoder(model, mel.data(), n_frames, model.mel_cfg.n_mels, enc); rc != 0) {
        std::fprintf(stderr, "[tdt-lstm-parity] run_encoder rc=%d\n", rc);
        return 1;
    }

    std::vector<int32_t> plain, fused;
    bool fused_used = false, plain_used = false;
    if (int rc = decode_with(model, enc, false, plain, plain_used); rc != 0) return rc;
    if (int rc = decode_with(model, enc, true, fused, fused_used); rc != 0) return rc;
    if (!fused_used) {
        std::fprintf(stderr, "[tdt-lstm-parity] SKIP: backend has no fused LSTM cell (n_gpu_layers=%d)\n", n_gpu_layers);
        return 3;
    }
    if (plain != fused) {
        std::fprintf(stderr, "[tdt-lstm-parity] FAIL: %zu tokens (decomposed) vs %zu tokens (fused) differ\n",
                     plain.size(), fused.size());
        return 1;
    }
    std::printf("[tdt-lstm-parity] PASS: %zu tokens identical with and without the fused LSTM cell\n", fused.size());
    return 0;
}
