# audiogen-cpp

Native [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) text-to-music in pure C++ on [ggml](https://github.com/tetherto/qvac-ext-ggml): caption and lyrics in, 48 kHz stereo audio out, no Python or PyTorch at inference time.

| Property | Value |
|---|---|
| CMake project | `audiogen-cpp` v0.1.0 |
| Public API | `tts_cpp::acestep::Engine` (`include/audiogen-cpp/acestep/engine.h`) |
| Output | interleaved stereo PCM, 48 kHz, `pcm[t * 2 + ch]` |
| Backends | CPU, Vulkan (including Android Mali iGPUs), Metal, OpenCL (validated on Adreno 700+) |
| ggml | requires the `ggml-speech` port for the custom `ggml_snake` and `ggml_col2im_1d` ops |
| Consumed by | the `@qvac/audiogen-ggml` addon in [QVAC](https://github.com/tetherto/qvac) |

## Pipeline

```
caption + lyrics (+ bpm, key, time signature, language)
   |
   v
LM (acestep-5Hz-lm, Qwen3 causal) -> metadata + acoustic codes
FSQ detokenizer                    -> DiT context latents
text encoder (Qwen3-Embedding)     -> prompt embeddings
condition encoder                  -> cross-attention states
DiT (flow matching, Euler)         -> 64-channel acoustic latent
VAE (AutoencoderOobleck)           -> 48 kHz stereo PCM
```

The VAE upsamples by 1920, so `T_audio = T_latent * 1920`. Latents are time-major, `latent[t * 64 + c]`.

## Model stages

Four GGUF files provide six runtime weight sets. The DiT GGUF contains the
condition encoder and FSQ detokenizer weights as well as the DiT weights; the
other files provide the text encoder, LM, and VAE. Point `--models <dir>` at a
directory and the engine classifies files by filename stem, or pass explicit
per-file paths, which always win over the scan.

| Stage | Upstream model | Filename stems matched | Flag |
|---|---|---|---|
| Text encoder | Qwen3-Embedding | `embedding`, `text-enc`, `textenc` | `--text` |
| LM | ACE-Step 5 Hz LM (Qwen3 causal) | `-lm`, `lm-`, `_lm`, `ace-lm`, `5hz-lm` | `--lm` |
| DiT + condition encoder + FSQ detokenizer | ACE-Step v1.5 diffusion transformer | `turbo`, `dit`, `v15`, `sft` | `--dit` |
| VAE | AutoencoderOobleck | `vae` | `--vae` |

The most specific stems (`embedding`, `vae`) are tested first so no short token such as `lm` can claim an unrelated file.

### DiT variants

Detected from the `acestep.is_turbo` GGUF key; absent means base or sft.

| Variant | Default steps | Default shift | Notes |
|---|--:|--:|---|
| turbo | 8 | 3.0 | fastest, no CFG on the DiT |
| base / sft | 50 | 1.0 | DiT CFG and APG (`guidance > 1`) are deferred |

Weights load quantized. `f32`, `f16`, and `bf16` are handled for norms and biases, and the detokenizer's `special_tokens` may be `q8_0`. This tree ships no converter: the stage GGUFs come from `convert.py` in [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp), the upstream C++/ggml implementation this port follows, which also publishes pre-quantized GGUFs at [Serveurperso/ACE-Step-1.5-GGUF](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF).

### Model setup

The QVAC registry publishes three validated four-file combinations. All use
the same fixed text encoder, LM, and VAE; select one DiT:

| Role | Filename | Quantization | Approximate size | Variants |
|---|---|---|---:|---|
| Text encoder | `Qwen3-Embedding-0.6B-Q8_0.gguf` | `q8_0` | 748 MB | all |
| LM | `acestep-5Hz-lm-0.6B-Q8_0.gguf` | `q8_0` | 677 MB | all |
| VAE | `vae-BF16.gguf` | `bf16` | 322 MB | all |
| DiT | `acestep-v15-turbo-Q4_K_M.gguf` | `q4_k_m` | 1.35 GB | `turbo-q4` |
| DiT | `acestep-v15-turbo-Q8_0.gguf` | `q8_0` | 2.37 GB | `turbo-q8` |
| DiT | `acestep-v15-sft-Q8_0.gguf` | `q8_0` | 2.37 GB | `sft` |

From a [QVAC](https://github.com/tetherto/qvac) checkout with package
dependencies installed, use its registry client rather than constructing
storage URLs:

```sh
npm --prefix packages/audiogen-ggml run download-models:registry -- \
  --output models/acestep --variant turbo-q4
```

Replace `turbo-q4` with `turbo-q8` or `sft`. The `all` variant fetches all
three DiTs, but do not then rely on `--models` alone: directory iteration does
not define which matching DiT is selected. Store each variant in a separate
directory, or pass the intended file explicitly with `--dit` /
`EngineOptions::dit_model_path`. The registry entries record the upstream
Hugging Face model-card links, but the registry GGUFs are QVAC-built artifacts.
The external
`Serveurperso/ACE-Step-1.5-GGUF` files above have separate provenance and are
not the combinations validated by QVAC.

## Backends

`n_gpu_layers > 0` (`--gpu` on the CLI) selects a GPU backend through the ggml
registry. Selection tries Adreno 700+ OpenCL first; validated Vulkan/Metal
discrete devices, then validated integrated devices; and finally other
discrete, then integrated GPU backends. Integrated-device support is required
because Vulkan reports Android UMA adapters such as Pixel's Mali-G715 as
`IGPU`.

| Stage | Placement when a GPU is selected |
|---|---|
| DiT, VAE | GPU |
| Text encoder, condition encoder | GPU, unless `ACESTEP_ENCODERS_CPU` is present |
| LM | GPU on Metal and OpenCL; CPU on Vulkan and every unmeasured backend |
| FSQ detokenizer | GPU on Vulkan, Metal, and OpenCL; CPU on every unmeasured backend |

The LM and detokenizer use an allowlist rather than a denylist: a backend keeps the CPU placement until the stage has been measured on it, so adding one cannot silently regress generated audio. Metal and OpenCL are validated for both stages; the recorded OpenCL validation used an Adreno 740. Vulkan is validated for the detokenizer, but the autoregressive LM remains on CPU because Mali-G715 testing showed code collapse and early termination on Vulkan.

Measurement is against an F32-dequantized reference (`scripts/dequant_gguf.py`), not against CPU. CPU is not automatically ground truth for a quantized model — ggml's CPU matmul quantizes activations to Q8_1 internally. On Metal the LM reproduces the F32 argmax trajectory exactly where CPU Q8_0 diverges at the first token. On Mali Vulkan, however, the LM produces repeated semantic codes and can terminate at roughly half the requested duration, while the same device produces a diverse full-length sequence with the LM on CPU.

On a GPU that supports F32 flash attention, Phase 2 also decodes the conditional and unconditional CFG paths in one batched graph. This matches the reference `acestep.cpp` LM path: the same prompt, model, sampler settings, and seed produce the same semantic-code sequence. Unsupported backends keep the separate F32 manual-attention path.

`lm-smoke --gpu --quantized-batch-cfg-regression --model <Q4-or-Q8-LM.gguf>` compares the compact batched head against two full-vocabulary decode streams. It requires quantized tied embeddings and fails if either stream changes argmax or drops below `0.99999` logit cosine.

The policy itself lives in [`src/acestep/stage_placement.h`](src/acestep/stage_placement.h), separate from the engine, so it is unit tested without a GPU.

### Environment overrides

Applied after the allowlist. The LM, detokenizer, and encoder overrides use
presence semantics: even a value of `0` activates them, and CPU wins when both
CPU and GPU are present for one stage. `ACESTEP_VAE_GPU` checks its first
character: `1` selects GPU and every other present value selects CPU.

| Variable | Effect |
|---|---|
| `ACESTEP_LM_GPU` / `ACESTEP_LM_CPU` | presence forces the LM onto the GPU or CPU |
| `ACESTEP_DETOK_GPU` / `ACESTEP_DETOK_CPU` | presence forces the detokenizer onto the GPU or CPU |
| `ACESTEP_ENCODERS_CPU` | presence moves the encoders to the CPU to trim wired memory |
| `ACESTEP_VAE_GPU` | a value beginning with `1` forces GPU; any other present value forces CPU |
| `ACESTEP_KEEP_STAGES` | values beginning with `1`, `t`, `T`, `y`, or `Y` eagerly load and keep every stage resident |
| `ACESTEP_LM_DUMP_LAYERS` | write the LM's per-layer prefill hidden states to this file path |
| `ACESTEP_LM_DUMP_TOKENS` | write the Phase-2 prompt token IDs as CSV |
| `ACESTEP_LM_DUMP_LOGITS` | write the conditional Phase-2 prefill logits as raw F32 |
| `ACESTEP_VAE_PROFILE` | print the VAE per-op-type time inventory |
| `ACESTEP_VAE_WIN_CORE` | diagnostic-only positive integer that pins the decode window core; not a supported tuning API |

Use `ACESTEP_LM_GPU` or `ACESTEP_DETOK_GPU` to take the measurement that would widen the allowlist for a new backend, without a rebuild.

### Parity debug hooks

`ACESTEP_PARITY_DEBUG` is a compile-time macro, not an environment variable, and there is no CMake option for it: configure with `-DCMAKE_CXX_FLAGS=-DACESTEP_PARITY_DEBUG`. The hooks below are `#ifdef`-ed out of `generate()` without it, so setting them on a default build does nothing.

| Variable | Effect |
|---|---|
| `ACESTEP_DUMP_DIR` | log the tokenizer shapes and write `our_dit_output.bin`, `our_noise.bin`, `our_context.bin`, `our_enc_hidden.bin` here for tensor-level comparison against acestep.cpp |
| `ACESTEP_INJECT_NOISE`, `ACESTEP_INJECT_ENC`, `ACESTEP_INJECT_CONTEXT` | replace the noise, encoder hidden states, or DiT context with an acestep.cpp `--dump` tensor to isolate a diverging stage |

Dumps use a flat `[ndim, d0, d1]` int32 header followed by the `f32` payload, the same format the injection hooks read.

### Memory

By default no stage stays resident after `create()`: `generate()` loads each stage immediately before its step and frees it right after, so only one stage is resident for the LM, detokenizer, DiT, and VAE steps rather than all six at once, which is what keeps a non-entitled iOS app inside its memory budget. The one overlap is the condition encoder: it supplies the silence frame that pads the DiT context before the text encoder runs and its own forward comes after text encoding, so it is loaded first and the two encoders are co-resident until the text encoder is freed. That pair is the only two-stage overlap, not necessarily the largest memory footprint. A truthy `ACESTEP_KEEP_STAGES` opts out, eagerly loads every stage in `Engine::create()`, and keeps them resident.

Long VAE decodes are split into overlapping windows. The engine probes the
active backend's maximum allocation size against the real decode graph and
shrinks the core window when needed; short inputs remain a single graph.
`ACESTEP_VAE_WIN_CORE` can pin the core only for diagnostics. VAE encode is not
windowed and still allocates one full graph.

## Build

Standalone from the repository root, against an installed `ggml-speech`:

```sh
cmake -S engines/audiogen -B build/audiogen -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-install
cmake --build build/audiogen -j
./build/audiogen/music-cli --help
```

The umbrella build uses `SPEECH_BUILD_AUDIOGEN=ON` (the default):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-install -DSPEECH_BUILD_AUDIOGEN=ON
cmake --build build -j
./build/engines/audiogen/music-cli --help
```

Visual Studio, Xcode, and other multi-config generators add a configuration
directory such as `Release/` beneath the executable directory. See the
[top-level README](../../README.md) for the shared ggml build.

| Option | Default | Effect |
|---|---|---|
| `AUDIOGEN_BUILD_LIBRARY` | `ON` | build the library; linkage follows `BUILD_SHARED_LIBS` |
| `AUDIOGEN_BUILD_EXECUTABLES` | `ON` standalone, `OFF` as a subdirectory | CLIs and per-stage smoke harnesses |
| `AUDIOGEN_BUILD_TESTS` | `ON` standalone, `OFF` as a subdirectory | CPU-only unit tests |
| `AUDIOGEN_INSTALL` | `ON` | generate install rules |
| `AUDIOGEN_USE_SYSTEM_GGML` | `ON` | `find_package(ggml)`; required, there is no supported vendored ggml in this tree |
| `AUDIOGEN_CCACHE` | `ON` | use ccache when available |

Consume the installed package with:

```cmake
find_package(audiogen-cpp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE audiogen-cpp::audiogen-cpp)
```

Minimal C++ usage:

```cpp
#include <audiogen-cpp/acestep/engine.h>

#include <exception>
#include <iostream>

int main() {
    try {
        tts_cpp::acestep::EngineOptions options;
        options.models_dir = "models/acestep";
        options.n_gpu_layers = 99;

        auto engine = tts_cpp::acestep::Engine::create(options);
        tts_cpp::acestep::GenerateParams params;
        params.caption = "Driving synth pop with bright analog leads";
        params.duration = 8.0f;

        bool cancel_requested = false;
        auto result = engine->generate(params, [&cancel_requested](const std::string &, int, int) {
            return !cancel_requested; // return false to cancel cooperatively
        });
        std::cout << result.pcm.size() / 2 << " stereo frames at "
                  << result.sample_rate << " Hz\n";
        // result.pcm is interleaved: pcm[frame * 2 + channel].
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
```

## Command line tools

| Binary | Purpose |
|---|---|
| `music-cli` | end-to-end text-to-music |
| `acestep-cli` | VAE decode and reconstruction roundtrip harness |
| `textenc-smoke`, `lm-smoke`, `lmgen-smoke`, `bpe-smoke`, `detok-smoke`, `cond-smoke`, `dit-smoke` | per-stage smoke harnesses |

### music-cli

```sh
# all four GGUFs in one directory
./build/audiogen/music-cli --models models/acestep --out song.wav --dur 8 --seed 42

# explicit per-stage paths, GPU, custom prompt
./build/audiogen/music-cli --dit dit.gguf --lm lm.gguf --text emb.gguf --vae vae.gguf \
                  --caption "driving synth pop, bright analog leads" \
                  --lyrics "[Instrumental]" --bpm 128 --key "C major" --tsig 4/4 \
                  --lang en --steps 8 --shift 3.0 --gpu --out song.wav
```

| Flag | Default | Meaning |
|---|---|---|
| `--models DIR` | | directory holding the four stage GGUFs |
| `--dit`, `--lm`, `--text`, `--vae` | | explicit per-stage GGUF paths |
| `--caption TEXT` | built-in pop-rock prompt | text prompt |
| `--lyrics TEXT` | `[Instrumental]` | lyrics |
| `--dur SECONDS` | `8` | target length, drives the LM code count |
| `--seed N` | `42` | negative means random |
| `--bpm N`, `--key STR`, `--tsig STR`, `--lang CODE` | inferred | optional metadata hints |
| `--steps N`, `--shift F` | per variant | sampler overrides |
| `--no-dcw` | DCW enabled | disable the official Haar low/high correction applied after each DiT step |
| `--temp F`, `--topp F`, `--topk N`, `--cfg F` | `0.85`, `0.9`, off, `2.0` | LM sampling for the audio codes |
| `--no-phase1` | off | skip the LM metadata auto-fill pass |
| `--req FILE` | | request JSON; pre-supplied `audio_codes` skip the LM stage |
| `--gpu`, `--threads N` | CPU, hardware concurrency | compute placement |
| `--backends-dir DIR` | | directory containing staged dynamic ggml backend modules; required by Android and Linux arm64 dynamic-backend builds |
| `--dump-stages DIR` | | write one `.bin` per stage into an existing directory |
| `--out PATH` | `music_out.wav` | output WAV path |

`--req` accepts a flat JSON object. Request values override overlapping CLI
values after CLI parsing:

| Field | Accepted value |
|---|---|
| `caption`, `lyrics`, `keyscale`, `timesignature`, `vocal_language` | string |
| `bpm`, `inference_steps`, `seed` | number |
| `duration`, `shift`, `dcw_scaler`, `dcw_high_scaler` | number |
| `dcw_enabled` | boolean, or `0` / `1` |
| `audio_codes` | quoted comma-separated string, for example `"12,34,56"` |

`audio_codes` is currently not parsed as a JSON numeric array. A non-empty
value bypasses the LM and feeds the codes directly to the FSQ detokenizer.

The sampler enables ACE-Step's single-level Haar DCW `double` mode by default.
At timestep `t`, the low band uses `t * 0.05` and the high band uses
`(1 - t) * 0.02`, matching the official Python defaults. The correction runs
on the host latent between DiT steps, so it is backend-independent and adds
negligible work compared with a transformer forward.

### acestep-cli

```sh
./build/audiogen/acestep-cli --model vae.gguf --t-latent 32 --out out.wav
./build/audiogen/acestep-cli --model vae.gguf --roundtrip --in in.wav --seconds 2.56 --out out.wav
```

The first form is the default mode: it decodes a synthetic latent, which checks that real weights load and that the decode graph (`ggml_col2im_1d` + `ggml_snake`) runs on the selected backend. `--roundtrip` encodes a real WAV and prints the per-channel reconstruction correlation, the audible end-to-end VAE check. Both forms take `--gpu`.

## Tests and parity

```sh
cmake -S engines/audiogen -B build/audiogen -DAUDIOGEN_BUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-install
cmake --build build/audiogen -j
ctest --test-dir build/audiogen
```

`test-acestep-units` covers the weight-free CPU logic and needs no GGUFs. That includes the stage-placement policy above: the backend allowlist (both the `MTL` and `Metal` registry names), the CPU fallback for every unmeasured backend, and the environment override precedence.

Stage dumps are the tool for localising a backend divergence. Run the same prompt twice with `--dump-stages`, then compare:

| Script | Purpose |
|---|---|
| `scripts/stage_cos.py` | per-stage cosine and rel_l2 between two dump directories; fails under a worst-stage cosine of 0.999, and also on a missing, truncated, or mismatched counterpart |
| `scripts/dequant_gguf.py` | rewrite a GGUF with every quantized tensor dequantized to F32, giving a ground-truth trajectory to compare a quantized run against |

Both need `numpy`; `dequant_gguf.py` also needs `gguf`.

An informal local comparison against upstream `acestep.cpp` measured 0.98 to
0.99 end-to-end correlation on identical codes, and LM greedy decoding matched
upstream argmax. This is not reproducible benchmark evidence: the repository
does not pin the model set, upstream revision, command, correlation definition,
or result artifact. Use stage dumps and the scripts above for a recorded
comparison.

## Performance

There is no CI benchmark suite for this engine yet. `music-cli` always enables
verbose engine output and prints per-stage wall clock to stderr (`[music-cli]`,
`[acestep-timing]`). Direct library use prints `[acestep-timing]` only when
`EngineOptions::verbose` is enabled; it defaults to `false`.

## License

`audiogen-cpp` is MIT licensed; see [LICENSE](LICENSE). Model weights and upstream model code carry their own terms, listed in [NOTICE](NOTICE).
