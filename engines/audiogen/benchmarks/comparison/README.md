# ACE-Step engine-to-engine comparison

Direct comparison of QVAC `engines/audiogen` (`music-cli`) and
[`ServeurpersoCom/acestep.cpp`](https://github.com/ServeurpersoCom/acestep.cpp)
(`ace-lm` + `ace-synth`). The measured path is the two C++ CLIs only. It does
not use `packages/audiogen-ggml`, an addon, Bare, the SDK, RPC, or a model
downloader inside the timed interval.

This is an **implementation-level** comparison. The engines share ACE-Step 1.5
GGUF layout and sampling defaults, but they do not share a ggml pin, a process
shape, or identical default hyperparameters.

Read [`architecture.md`](architecture.md) before quoting a winner.

## Layout

```text
engines/audiogen/benchmarks/comparison/
  config.json                 # shared run configuration
  prompts/manifest.json       # deterministic prompts, lyrics, seeds, durations
  adapters/qvac.js            # music-cli adapter
  adapters/acestep.js         # ace-lm + ace-synth adapter
  lib/                        # parse, backend, audio, aggregate, report
  run-comparison.js
  generate-report.js
  capture-environment.js
  tests/
  reports/                    # reviewed platform reports
  models/                     # gitignored GGUFs
  vendor/acestep.cpp          # gitignored external checkout
  out/                        # gitignored generated results
```

## Dependencies

Install through the host package manager, then review local source before
building:

- CMake >= 3.20, a C++17 compiler, git, Node.js 20+
- Python is optional (CLAP). Fréchet Audio Distance is not run by default
  because this harness does not ship a licensed reference corpus.

Do not pipe remote installers into a shell.

## Model acquisition

Download the same four Serveurperso GGUFs into `models/`. These files are data,
not executables. Record SHA-256 hashes in the report.

| Role | File | Quant |
|---|---|---|
| Text encoder | `Qwen3-Embedding-0.6B-Q8_0.gguf` | q8_0 |
| LM | `acestep-5Hz-lm-0.6B-Q8_0.gguf` | q8_0 |
| DiT | `acestep-v15-turbo-Q8_0.gguf` | q8_0 |
| VAE | `vae-BF16.gguf` | bf16 |

Source: https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF

The 0.6B LM is used instead of the acestep.cpp README default 4B LM so both
engines run the QVAC-validated combination. Put all four files in one
directory and point both CLIs at it.

Conversion from safetensors, if you must rebuild, uses `convert.py` from
acestep.cpp. QVAC does not ship a converter; it loads that GGUF layout.

## Build QVAC music-cli

From the whisper.cpp / speech-stack repository root. Review
`https://github.com/tetherto/qvac-ext-ggml` (`speech` branch) before compiling;
the VAE needs `ggml_snake` and `ggml_col2im_1d`.

CPU (Metal disabled):

```sh
git clone --depth 1 --branch speech https://github.com/tetherto/qvac-ext-ggml ggml-src-cpu
cmake -S ggml-src-cpu -B ggml-src-cpu/build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_METAL=OFF \
  -DCMAKE_INSTALL_PREFIX=$PWD/ggml-install-cpu
cmake --build ggml-src-cpu/build -j
cmake --install ggml-src-cpu/build
cmake -S engines/audiogen -B engines/audiogen/build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$PWD/ggml-install-cpu -DAUDIOGEN_BUILD_TESTS=ON
cmake --build engines/audiogen/build-cpu --target music-cli -j
```

Metal:

```sh
git clone --depth 1 --branch speech https://github.com/tetherto/qvac-ext-ggml ggml-src-metal
cmake -S ggml-src-metal -B ggml-src-metal/build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DCMAKE_INSTALL_PREFIX=$PWD/ggml-install-metal
cmake --build ggml-src-metal/build -j
cmake --install ggml-src-metal/build
cmake -S engines/audiogen -B engines/audiogen/build-metal -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$PWD/ggml-install-metal
cmake --build engines/audiogen/build-metal --target music-cli -j
```

Set `ACESTEP_QVAC_CLI` to the `music-cli` you intend to measure.

## Build acestep.cpp

Checkout next to this comparison directory (`vendor/acestep.cpp`) or set
`ACESTEP_CPP_DIR`. Initialise its ggml submodule. Review the source, then
build. On macOS a default build enables Metal; a CPU comparison requires a
separate tree with Metal off.

```sh
git clone https://github.com/ServeurpersoCom/acestep.cpp.git vendor/acestep.cpp
git -C vendor/acestep.cpp submodule update --init
cmake -S vendor/acestep.cpp -B vendor/acestep.cpp/build-cpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
cmake --build vendor/acestep.cpp/build-cpu --target ace-lm ace-synth -j
cmake -S vendor/acestep.cpp -B vendor/acestep.cpp/build-metal \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build vendor/acestep.cpp/build-metal --target ace-lm ace-synth -j
```

Set `ACESTEP_CPP_LM` and `ACESTEP_CPP_SYNTH`.

## Run

```sh
cd engines/audiogen/benchmarks/comparison
node tests/backend.test.js tests/timing.test.js tests/aggregate.test.js \
  tests/report.test.js tests/audio.test.js
node capture-environment.js
node run-comparison.js --dry-run --backend cpu
ACESTEP_QVAC_CLI=... ACESTEP_CPP_LM=... ACESTEP_CPP_SYNTH=... \
  node run-comparison.js --backend cpu
ACESTEP_QVAC_CLI=... ACESTEP_CPP_LM=... ACESTEP_CPP_SYNTH=... \
  node run-comparison.js --backend metal
```

Interrupted runs resume from `out/rounds/`. `--force` rebuilds every round.
`--prompts instrumental-folk,short-drone` selects a subset.
`ACESTEP_COMPARE_ORDER=random|alternate|qvac-first|acestep-first` and
`ACESTEP_COMPARE_COOLDOWN_MS` control order and pauses.

`node generate-report.js out/cpu.json` regenerates Markdown from JSON.

Copy reviewed JSON/Markdown into `reports/mac-arm64/` and write
`verification-report.md`. Keep large WAVs out of git; record checksums.

## Interpreting metrics

- **generation_ms**: QVAC `[acestep-timing]` total when present; otherwise
  process wall. For acestep.cpp, LM wall plus synth wall (or parsed load/gen
  lines when the CLIs print them).
- **e2e_ms**: process wall, including WAV write. QVAC is one process;
  acestep.cpp is two.
- **RTF**: generation seconds / generated audio seconds. Lower is faster.
- **init_ms**: QVAC is lazy-load, so init is folded into the first stage of
  `generate()`. acestep.cpp load lines are summed when logged.
- Quality metrics (silence, clipping, duration error, optional CLAP) use the
  WAV after generation and are not part of inference time.

Medians and distributions are reported, not only the fastest run.

## Adding another platform

1. Capture environment with `capture-environment.js`.
2. Build CPU and the platform GPU backend for **both** engines.
3. Run `--backend cpu` and the GPU name only if logs prove both used that GPU.
4. Store reports under `reports/<target>/`.
