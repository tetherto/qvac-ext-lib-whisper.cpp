# qwen-asr-cpp

A C++ inference library for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) (Alibaba Cloud / Qwen Team) on-device automatic speech recognition.

Sibling of `parakeet-cpp` inside `qvac-ext-lib-whisper.cpp`. Exposes a `qwen::Engine` API analogous to `parakeet::Engine`, packaged the same way (CMake target `qwen-asr::qwen-asr`, install rules, `cli` binary).

## Status

**v0.1 — usable.** The library transcribes WAV files end-to-end. Verified locally on Apple Silicon (M-series) with CPU + Accelerate BLAS at ~4.6x realtime on `jfk.wav` (11 s of audio in 2.4 s, 0.6B model). Two `ctest` targets cover construction errors + an end-to-end transcription against a ground-truth fixture.

**Inference backend.** v0.1 vendors the pure-C [`antirez/qwen-asr`](https://github.com/antirez/qwen-asr) (MIT) engine under `vendor/qwen-asr-c/` and wraps it behind a C++ `Engine`. This gives us a working transcriber today; the trade-off is CPU-only (NEON on ARM, AVX/AVX-512 on x86, optional BLAS for matmul on macOS/Linux).

**v0.2 — planned.** Port the inference path onto GGML to get Metal / CUDA / Vulkan / OpenCL backends (same shape as `parakeet-cpp`). The `vendor/qwen-asr-c/MODEL.md` spec + the existing C implementation are the references for that port.

## Quick start (Linux / macOS, desktop)

```bash
cd qwen-asr-cpp

# 1) Download the 0.6B checkpoint from HuggingFace (~1.9 GB).
./scripts/download-models.sh --type 0.6b

# 2) Configure + build (no ggml in v0.1; pure C vendor + C++ wrapper).
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3) Transcribe.
./build/qwen-asr transcribe \
    --model-dir models/hf/0.6b \
    --wav test/samples/jfk.wav \
    --language English \
    --verbose
```

Expected output (verified on M-series):

```
And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.
Inference: 2375 ms, 26 text tokens (encoder 429 ms, decoder 1943 ms)
Audio: 11.0 s processed in 2.4 s (4.63x realtime)
```

## C++ API

```cpp
#include <qwen/engine.h>

qwen::EngineOptions opts;
opts.model_dir = "models/hf/0.6b";
opts.language  = "English";   // or "" for auto-detect
opts.n_threads = 0;            // 0 = autodetect

qwen::Engine engine(opts);
auto result = engine.transcribe("audio.wav");
std::puts(result.text.c_str());
```

Optional per-token streaming callback (for live UI):

```cpp
engine.set_token_callback([](const std::string & piece) {
    std::fputs(piece.c_str(), stdout);
    std::fflush(stdout);
});
engine.transcribe("audio.wav");
```

## Build options

| Option | Default | Purpose |
| --- | --- | --- |
| `QWEN_BUILD_LIBRARY`     | `ON`  | Build `libqwen-asr` |
| `QWEN_BUILD_EXECUTABLES` | `ON`* | Build the `qwen-asr` CLI |
| `QWEN_BUILD_TESTS`       | `ON`* | Build the `ctest` targets |
| `QWEN_USE_BLAS`          | `ON`  | Link Apple Accelerate (macOS) or OpenBLAS (Linux) for matmul speedup |
| `QWEN_INSTALL`           | `ON`  | Generate `install(...)` rules |
| `QWEN_CCACHE`            | `ON`  | Use `ccache` if found |

`*` = on when configured standalone; off when added via `add_subdirectory()`.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

| Target | Label | Runs |
| --- | --- | --- |
| `test-construct`      | `unit` | Empty `model_dir` rejected; non-existent dir surfaces load error. Fast, no model required. |
| `test-transcribe-jfk` | `e2e`  | Loads the 0.6B model and transcribes `test/samples/jfk.wav`, asserts the hypothesis contains "ask not" + "country". Auto-disabled when `models/hf/0.6b/model.safetensors` is missing. |

## Cross-platform support

| Platform | Status |
| --- | --- |
| macOS arm64 (Apple Silicon, NEON + Accelerate) | **verified end-to-end** |
| macOS x64 (AVX/AVX-512 + Accelerate) | should work (untested) |
| Linux x64 (AVX/AVX-512 + OpenBLAS) | should work (untested) |
| Linux ARM (NEON + OpenBLAS) | should work (untested) |
| iOS arm64 | CMake should work; not yet wrapped in an xcframework |
| Android arm64-v8a | CMake should work; not yet packaged |
| Windows x64 | CMake configured; not yet tested |

For full Vulkan / OpenCL / Metal / CUDA backend coverage, see the v0.2 GGML-port plan in `PROGRESS.md`.

## License

MIT. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for third-party attributions.
