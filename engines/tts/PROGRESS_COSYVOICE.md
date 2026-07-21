# CosyVoice3 bring-up progress

Staged bring-up of the `tts_cpp::cosyvoice` engine (Fun-CosyVoice3-0.5B / 1.5B),
CPU-first, validated numerically against the upstream PyTorch reference — same
playbook as `PROGRESS.md` (Chatterbox) and `PROGRESS_SUPERTONIC.md` (Supertonic).

## Pipeline

```
text --(wetext + Qwen2 BPE)--> Qwen2.5-0.5B LM --> speech tokens [0,6561)
     --> DiT CausalConditionalCFM (10-step Euler) --> mel [80, T]
     --> CausalHiFTGenerator (NSF + iSTFT n_fft=16/hop=4) --> 24 kHz PCM
zero-shot prompt: reference wav --> S3 tokenizer (prompt tokens) + CAM++ (192-d emb)
```

## Status

| # | Stage | State | Reuse | Notes |
|---|---|---|---|---|
| 0 | Public engine + JS/SDK/addon wiring | ✅ **engine live** (QVAC-21928) | shared `cosyvoice_pipeline.{h,cpp}` | `Engine::synthesize()` runs the real LM→flow→HiFT chain; **Whisper-verified** ("The quick brown fox…"). Addon/SDK route still to wire |
| 1 | Reference sandbox + dump script | ✅ **ran on 3090** | `dump-supertonic-reference.py` pattern | refs in `models_evaluation/CosyVoice/artifacts/cv3-ref/`; reference.wav Whisper-verified intelligible |
| 2 | HiFT GGUF converter | ✅ ran; names fixed | `convert-*-to-gguf.py` | emits `hift/<path>` names to match `run_hift_decode` (246 tensors, f32) |
| 3 | **HiFT ggml graph** | ✅ **done** (`src/cosyvoice_hift.cpp`) | adapted `run_hift_decode` + causal + SineGen2 | **0.989 log-mel / 0.999 energy corr** vs `hift_wav.npy` |
| 4 | **Flow GGUF converter + DiT ggml graph** | ✅ **done** (`src/cosyvoice_flow.cpp`, `scripts/convert-cosyvoice3-flow-to-gguf.py`) | net-new 22-layer DiT + CFM Euler | **flow_mel cosine 1.000000** (rel 6.7e-4) vs `flow_mel.npy` |
| 5 | **LLM GGUF converter + Qwen2 ggml graph** | ✅ **done** (`src/cosyvoice_llm.cpp`, `scripts/convert-cosyvoice3-llm-to-gguf.py`) | Qwen2.5-0.5B (GQA, QKV bias, SwiGLU, RoPE θ=1e6) + autoregressive decode + RAS sampling | prefill logits **cosine 1.000000**; generates speech tokens to natural EOS |
| 6 | Zero-shot prompt path | ~ partial | `s3tokenizer.cpp`, `campplus.cpp` | prompt tensors (S3 tokens, CAM++ emb, prompt mel) currently taken from PyTorch dump; native S3/CAM++ still to port |
| 7 | **Text frontend (Qwen2 BPE)** | ✅ **done** (`src/qwen_tokenizer.h`) | byte-level BPE + merges | **4/4 reference tokenizations exact** (EN+ZH). wetext normalization not yet ported |
| 8 | **End-to-end (text → audio, C++)** | ✅ **works** | LM → flow → HiFT chained | typed English → speech, **Whisper-verified intelligible**, zero PyTorch |

### Validation bugs found & fixed (per-stage bisection vs PyTorch)
- corrupted reference (`.numpy()` aliasing a reused Euler buffer → `.clone()`)
- head-0-only RoPE in the DiT (x_transformers ropes only the first head)
- `conv1d_f32` mul_mat operand order (im2col first, kernel second)
- gallocr clobbering conditioning inputs across Euler steps (re-set every step)
- GQA head-recombination permute in the Qwen2 attention (`0,3,1,2`)

### Library refactor (QVAC-21928, done)
The three graphs now live in one shared TU, `src/cosyvoice_pipeline.{h,cpp}`
(in-memory, ggml-only, no npy/file I/O), which is the single source of truth for:
- the parity CLIs `cosyvoice-{llm,flow,hift}` (now thin harnesses; gates re-verified:
  LM prefill cosine 1.000000, flow_mel cosine 1.000000, dit_out cosine 1.000000), and
- the public `Engine` (`src/cosyvoice_engine.cpp`), whose `synthesize()` chains
  `cosyvoice_llm_generate → cosyvoice_flow_run → cosyvoice_hift_synth`.
Default voice is baked into a small `voice.gguf` (`scripts/bake-cosyvoice3-voice.py`:
`voice/prompt_stok`, `prompt_token`, `prompt_feat`, `embedding`), so the engine
needs **no PyTorch** and no live S3/CAM++.  `cosyvoice_pipeline.cpp` is in the
`tts-cpp` library sources + all three CLI targets (CMakeLists).

### LM KV cache (done)
`cosyvoice_llm_generate` now prefills the prompt once and decodes one token at a
time against a per-layer post-rope K/V cache (`qwen_step_kv`) — O(L²) instead of
O(L³).  Bit-identical output to the old re-prefill path (same tokens, same WAV
md5); LM on "The quick brown fox…" dropped **63 s → ~2.5 s** decode (~25×/token),
full engine end-to-end **~98 s → ~25 s**.  Bring-up bug: the prefill V-cache
output was a reshape *view* of the v_proj tensor, whose memory gallocr recycled
as scratch → garbage on read-back; fixed by `ggml_cont`-materialising the K/V
cache outputs before marking them graph outputs.

### Remaining (engineering, not numerics)
- Wire the `tts-ggml` addon (`CosyvoiceModel`) + `@qvac/sdk` cosyvoice3 route to `Engine`
- Native S3 tokenizer + CAM++ (stage 6) so zero-shot from arbitrary reference audio works
- wetext text normalization
- (opt) speed: flow now dominates (10 Euler × 22-layer DiT, batch-2) — DiT is the next lever

## Minimum path to first *real* audio

EN-only, greedy LM, f32, non-streaming, **baked voice** (from `spk2info` /
voices.gguf — skips live S3 tokenizer + CAM++): stages 2→3→4→5 only. Bring up
back-to-front (HiFT first) so each stage validates independently against the
dumped `.npy` references.

## HiFT (stage 3) — confirmed config + reuse plan

CosyVoice3 vocoder is `CausalHiFTGenerator` (from `cosyvoice3.yaml`):
in_channels 80, base_channels 512, nb_harmonics 8, sample_rate 24000,
upsample_rates [8,5,3], upsample_kernel_sizes [16,11,7],
resblock_kernel_sizes [3,7,11], resblock_dilations [[1,3,5]×3],
source_resblock_kernel_sizes [7,7,11], istft n_fft 16 / hop 4,
nsf_alpha 0.1 / nsf_sigma 0.003 / nsf_voiced_threshold 10,
lrelu_slope 0.1, **conv_pre_look_right 4** (causal), audio_limit 0.99,
f0_predictor = CausalConvRNNF0Predictor (5-conv condnet + classifier).
weight_norm layout = new `parametrizations.weight.original0/1`.

The existing `run_hift_decode` (`src/chatterbox_tts.cpp:2053`) already hardcodes
this exact config, so stage 3 = reuse it with two changes:
1. tensor names — converter now emits `hift/<path>` (done).
2. **causal padding** — replace symmetric conv padding with left-only zero-pad
   (`(K-1)·d` left; conv_pre look-right 4 → left `(K-1)-4`, right 4) via the
   existing `zero_pad_dim0` helper; adjust STFT/iSTFT edge trims. This is the
   only structurally-new work.
Verify against refs (don't assume identical to Chatterbox): src_rb kernels,
the i==2 channel-concat trick, f0_predictor tail activation.

## HiFT bring-up result (stage 3, done)

`src/cosyvoice_hift.cpp` (standalone `cosyvoice-hift` CLI, mel -> 24 kHz wav)
reproduces the CosyVoice3 CausalHiFTGenerator on CPU. Fed the reference
`hift_mel_in.npy`, its output matches `hift_wav.npy` at **0.989 log-mel
spectrogram corr** and **0.999 energy-envelope corr**; A/B against
`reference.wav` is perceptually identical. The residual (waveform corr ~0) is
the random unvoiced excitation noise (`randn`), which can't match the
reference's RNG draw and is perceptually irrelevant.

Key correctness fix during bring-up: the f0-predictor's `condnet[0]` is
`causal_type='right'` (right-pad), not left — the wrong direction produced a
garbage pitch contour (max 12 kHz) and an audibly wrong excitation.

Built locally on macOS against the in-tree root `ggml` (no vcpkg needed —
`cosyvoice_hift.cpp` uses only standard ggml ops).

## Next action

DiT flow (stage 4): author `scripts/convert-cosyvoice3-flow-to-gguf.py` + the
DiT ggml graph (the net-new long pole), parity `speech_tokens.npy` + prompt
tensors -> mel vs `flow_mel.npy`. Then the Qwen2 LM (stage 5).
Optional follow-up: promote `cosyvoice_hift.cpp` from a standalone CLI into a
library function called by `cosyvoice_engine.cpp`, and add a
`test/test_cosyvoice_hift.cpp` npy-parity harness (assert log-mel corr ≥ 0.98).
