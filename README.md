# qvac-ext-lib-whisper.cpp — QVAC speech stack

C++ speech engines on [ggml](https://github.com/tetherto/qvac-ext-ggml),
organized as a symmetric multi-engine repo (see `docs/` and QIP #94):

```
CMakeLists.txt              # feature-gated umbrella superbuild (ours)
third_party/whisper.cpp/    # upstream whisper.cpp, vendored as a git subtree
                            #   pinned @ v1.9.1; minimally divergent — every
                            #   QVAC delta is listed in its PATCHES.md
engines/
  parakeet/                 # NVIDIA Parakeet ASR family (CTC/TDT/EOU/Sortformer)
  tts/                      # chatterbox + supertonic TTS, LavaSR enhancement
```

## Building

All components consume **one system ggml** — the `ggml-speech` vcpkg port,
built from [`qvac-ext-ggml`](https://github.com/tetherto/qvac-ext-ggml)
(`speech` branch). The `ggml/` tree inside the whisper subtree is never
compiled. Quick start against a local ggml install:

```sh
git clone --depth 1 --branch speech https://github.com/tetherto/qvac-ext-ggml ggml-src
cmake -S ggml-src -B ggml-src/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_PREFIX=$PWD/ggml-install
cmake --build ggml-src/build -j && cmake --install ggml-src/build

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$PWD/ggml-install
cmake --build build -j
```

Feature gates: `-DSPEECH_BUILD_{WHISPER,PARAKEET,TTS}=ON/OFF`,
`-DSPEECH_BUILD_TESTS=ON` (then `ctest --test-dir build -LE 'gpu|perf'`),
`-DSPEECH_BUILD_EXECUTABLES=OFF` for library-only builds.

Each engine still configures standalone (`cmake -S engines/parakeet`, …) —
that's what the per-engine vcpkg ports and the CI lanes use; see each
engine's README for models, tools, and test details.

## Touching `third_party/whisper.cpp`

Don't — unless you're doing an upstream sync (`docs/UPSTREAM-SYNC.md`) or
also updating `third_party/whisper.cpp/PATCHES.md` in the same PR. CI
enforces that the subtree matches the pinned upstream tag outside the
declared patch list. ggml patches never go here; they go to
`qvac-ext-ggml@speech`.
