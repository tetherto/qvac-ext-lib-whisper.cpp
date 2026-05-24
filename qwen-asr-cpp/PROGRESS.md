# qwen-asr-cpp — progress

Two release tracks:

- **v0.1** wraps the pure-C `antirez/qwen-asr` engine (vendored under
  `vendor/qwen-asr-c/`) behind a C++ `qwen::Engine`. CPU only (NEON / AVX
  / AVX-512 SIMD + optional BLAS). Goal: usable transcription today.
- **v0.2** ports the inference path onto GGML for multi-backend GPU
  support (Metal / CUDA / Vulkan / OpenCL), keeping the same public API.

Legend: `done` / `in-flight` / `next` / `pending`.

---

## v0.1 — vendor + C++ wrapper

### Phase 0 — Discovery & validation (done)

| Item | Status |
| --- | --- |
| Confirm `Qwen/Qwen3-ASR-0.6B` config (d_model, n_layers, vocab, mel bins) against published card | done |
| Cache reference materials under `vendor/qwen-asr-c/MODEL.md` | done |
| Identify `antirez/qwen-asr` as an MIT-compatible working implementation | done |
| Decide vendor-vs-port for v0.1 | done (vendor) |

### Phase 1 — Bootstrap (done)

| Item | Status |
| --- | --- |
| Directory + CMake skeleton | done |
| Public headers (`engine.h`, `cli.h`, `export.h`, umbrella `qwen.h`) | done |
| `Engine` pimpl + CLI wiring + `--version` / `--help` / `transcribe` | done |
| Install rules (`qwen-asr-cppConfig.cmake`, exported targets, public headers) | done |
| `scripts/setup-ggml.sh`, `scripts/download-models.sh` | done |
| Local desktop build green (`cmake --build build -j && ctest`) | done |

### Phase 2 — Vendor integration (done)

| Item | Status |
| --- | --- |
| Vendor `antirez/qwen-asr` under `vendor/qwen-asr-c/` (verbatim, MIT, LICENSE + MODEL.md preserved) | done |
| `qwen-asr-vendor` static library target with `-O3 -ffast-math -march=native` | done |
| Apple Accelerate / OpenBLAS detection + linking | done |
| `qwen::Engine` C++ wrapper over `qwen_ctx_t*` | done |
| Token-streaming callback bridge (`set_token_callback`) | done |
| `qwen-asr-cli` binary with `transcribe`, `--threads`, `--language`, `--prompt`, `--verbose`, `--debug` | done |
| `test-construct` unit test (empty / missing `model_dir` rejected) | done |
| `test-transcribe-jfk` end-to-end test (loads 0.6B, transcribes `jfk.wav`, asserts canonical phrase) | done |
| First green transcription on Apple Silicon | done (4.63x realtime) |

### Phase 3 — Polish (next)

| Item | Status |
| --- | --- |
| Linux smoke build verification (Ubuntu OpenBLAS) | pending |
| iOS xcframework target | pending |
| Android NDK build target | pending |
| Windows clang-cl smoke build | pending |
| Streaming API surface (`Engine::transcribe_stream`, `Engine::stream_start`) — antirez already supports this | pending |
| GitHub Actions CI matrix | pending |
| `qvac-registry-vcpkg` port for downstream consumption | pending |

---

## v0.2 — GGML port (planned)

Replace the vendored pure-C inference with a GGML-backed implementation
to enable Metal / CUDA / Vulkan / OpenCL backends, keeping `qwen::Engine`
binary-compatible with v0.1.

Reference materials already cached:

- `vendor/qwen-asr-c/MODEL.md` — authoritative architecture spec.
- `vendor/qwen-asr-c/qwen_asr_encoder.c` — encoder forward pass in C.
- `vendor/qwen-asr-c/qwen_asr_decoder.c` — decoder + prefill + sampling.
- `vendor/qwen-asr-c/qwen_asr_audio.c` — mel preprocessing.
- `vendor/qwen-asr-c/qwen_asr_tokenizer.c` — BPE tokenizer.
- `examples/talk-llama/models/qwen3.cpp` (parent repo) — Qwen3 GGML builder.

### Phase A — GGML graph builders

| Item | Status |
| --- | --- |
| Mel preprocessing on GGML (Whisper-style, port from `qwen_asr_audio.c`) | pending |
| AuT encoder graph (conv stem + per-chunk transformer + projector) | pending |
| Qwen3 decoder graph (Q/K RMSNorm + RoPE + GQA + SwiGLU + KV cache) | pending |
| Audio-pad replacement splicing in input embedding | pending |
| Greedy sampling + `<asr_text>` parsing | pending |

### Phase B — Cross-backend validation

| Item | Status |
| --- | --- |
| CPU parity vs v0.1 (`jfk.wav` token-id-exact) | pending |
| Metal parity sweep (M-series) | pending |
| CUDA parity sweep (RTX class) | pending |
| Vulkan parity sweep | pending |
| OpenCL parity sweep (Adreno) | pending |

### Phase C — Quantization

| Item | Status |
| --- | --- |
| GGUF converter `scripts/convert-hf-to-gguf.py` (q8_0 / q5_k_m / q4_k_m) | pending |
| WER drift vs FP16 reference on a small held-out fixture set | pending |
| Per-quant `test-perf-regression` | pending |
