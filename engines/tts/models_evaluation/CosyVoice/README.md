# CosyVoice3 bring-up sandbox

Reference-capture + conversion sandbox for the `tts_cpp::cosyvoice` engine
(Fun-CosyVoice3-0.5B / 1.5B). This directory is the numerical source of truth
for the staged CPU bring-up: run the upstream PyTorch model once, dump per-stage
tensors as `.npy`, then validate each ggml graph against them (cosine / relative
error), exactly like the Chatterbox (`PROGRESS.md`) and Supertonic
(`PROGRESS_SUPERTONIC.md`) bring-ups.

> Status: **iteration 1**. The C++ engine is still the scaffold
> (`src/cosyvoice_engine.cpp` returns placeholder audio). This sandbox produces
> the reference tensors + GGUFs that the real graphs are validated against.
> See `../../PROGRESS_COSYVOICE.md` for the staged plan and current state.

Everything here is **Python / offline tooling** — it needs PyTorch + the
upstream CosyVoice repo + the model weights, none of which the C++ inference
binaries depend on. Run it on the machine that has torch + the weights (e.g. the
RTX 3090 box); CPU-only torch works too (slower).

---

## 0. Prerequisites

```bash
# A clean venv is strongly recommended (CosyVoice pins older deps).
python3 -m venv .venv && source .venv/bin/activate

# Upstream CosyVoice (Apache-2.0) — provides the model classes we hook.
git clone --recursive https://github.com/FunAudioLLM/CosyVoice
pip install -r CosyVoice/requirements.txt
# Matcha-TTS submodule is on the path CosyVoice expects:
export PYTHONPATH="$PWD/CosyVoice:$PWD/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH"

# Conversion tooling (same package the other tts-cpp converters use).
pip install gguf numpy torch soundfile
```

## 1. Download the weights (Apache-2.0)

```bash
pip install "huggingface_hub[cli]"
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir models/Fun-CosyVoice3-0.5B
# ModelScope mirror (no HF account): iic/CosyVoice3-0.5B
```

The model dir contains (names may vary slightly per release — confirm on first
load): `llm.pt`, `flow.pt`, `hift.pt`, `campplus.onnx`,
`speech_tokenizer_v*.onnx`, `cosyvoice.yaml`, plus baked `spk2info.pt` voices.

## 2. Dump per-stage reference tensors

```bash
python3 dump_cosyvoice3_reference.py \
    --model-dir models/Fun-CosyVoice3-0.5B \
    --tts-text "The quick brown fox jumps over the lazy dog." \
    --prompt-audio CosyVoice/asset/zero_shot_prompt.wav \
    --prompt-text "希望你以后能够做的比我还好呦。" \
    --out-dir artifacts/cv3-ref
```

Writes to `artifacts/cv3-ref/` (float32, C-contiguous `.npy`, loadable by
`include/tts-cpp/npy.h`):

| file | stage | shape |
|---|---|---|
| `speech_tokens.npy` | LLM output | `[T_tok]` int32, in `[0, 6561)` |
| `prompt_token.npy` | flow input (prompt) | `[T_ptok]` int32 |
| `embedding.npy` | CAM++ speaker emb | `[192]` |
| `prompt_feat.npy` | flow input (prompt mel) | `[T_pmel, 80]` |
| `flow_mel.npy` | **flow output** (vocoder input) | `[80, T_mel]` |
| `hift_wav.npy` | **hift output** (reference audio) | `[N]` @ 24 kHz |
| `reference.wav` | same, as WAV for listening | — |

`flow_mel.npy` → `hift_wav.npy` is the pair the **HiFT vocoder** bring-up
(step 4, first) is validated against.

### Emotional / instruction prompting

Pass `--instruct-text` to use `inference_instruct2` (emotion / speed / volume /
dialect control via natural language) instead of zero-shot. The instruction
conditions the **LLM** (which our native stack builds at stage 5); the flow +
HiFT stages are the same, so the captured references also cover the emotional
path. Example — same sentence, different emotions:

```bash
for emo in "Speak happily" "Speak sadly" "Speak angrily" "Speak in a whisper"; do
  python3 dump_cosyvoice3_reference.py --model-dir models/Fun-CosyVoice3-0.5B \
    --tts-text "I can't believe it's already the weekend." \
    --prompt-audio CosyVoice/asset/zero_shot_prompt.wav \
    --instruct-text "$emo" --out-dir "artifacts/emo-$(echo $emo | tr ' A-Z' '_a-z')"
done
```

Listen to each `artifacts/emo-*/reference.wav` to confirm emotional prompting
works. Chinese instructions (e.g. `用开心的语气说`) also work.

## 3. Convert sub-models to GGUF

Start with the vocoder (smallest, back-of-pipeline → validated first):

```bash
python3 ../../scripts/convert-cosyvoice3-hift-to-gguf.py \
    --hift models/Fun-CosyVoice3-0.5B/hift.pt \
    --config models/Fun-CosyVoice3-0.5B/cosyvoice.yaml \
    --outfile artifacts/cosyvoice3-hift-f32.gguf --dtype f32
```

(Flow/DiT and LLM converters land next — see `PROGRESS_COSYVOICE.md`.)

## 4. Bring-up order (back-to-front, each parity-gated)

1. **HiFT vocoder** — feed `flow_mel.npy`, compare ggml wav vs `hift_wav.npy`
   (rel err target ~1e-4 on audio). Proves the vocoder + gives audible output.
2. **DiT flow** — feed `speech_tokens` + prompt tensors, compare mel vs
   `flow_mel.npy` (the net-new estimator).
3. **Qwen2 LM** — compare speech tokens vs `speech_tokens.npy`.
4. **End-to-end** — chain all three; WER / speaker-sim sanity on a small set.

Each stage gets a `test/test_cosyvoice_*.cpp` harness that loads the `.npy`
references via `npy.h` and asserts the tolerance, mirroring
`test/test_supertonic_*.cpp`.
