#!/usr/bin/env python3
"""Dump HuggingFace Qwen3-ASR encoder + decoder activations for a single audio file.

Used as a reference oracle for the from-scratch ggml C++ port. Writes raw
float32 tensors to /tmp/qwen_ref_*.bin and prints shapes/stats so the C++
implementation can be diff-ed layer by layer.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)


MODEL_DIR = Path("models/hf/0.6b")
WAV_PATH = Path("test/samples/jfk.wav")
OUT_DIR = Path("/tmp")


def load_audio_mel(wav_path: Path) -> torch.Tensor:
    """Compute log-mel spectrogram identical to Qwen3-ASR preprocessor.

    Returns a (1, 128, n_frames) tensor of float32.
    """
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"jfk.wav must be 16 kHz, got {sr}")
    audio = audio.astype(np.float32)
    # Match preprocessor_config.json (Qwen2/3 audio preprocessor).
    n_fft = 400
    hop = 160
    win = torch.hann_window(n_fft, periodic=False, dtype=torch.float32)
    audio_t = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
    audio_t = F.pad(audio_t, (n_fft // 2, n_fft // 2), mode="reflect").squeeze(0).squeeze(0)
    stft = torch.stft(
        audio_t,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=False,
        return_complex=True,
    )
    magnitudes = stft.abs() ** 2  # (n_bins, n_frames)

    # Build mel filter bank matching HF preprocessor.
    n_mels = 128
    mel_filters = build_mel_filters(sr, n_fft, n_mels)
    mel_spec = mel_filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.unsqueeze(0)


def build_mel_filters(sr: int, n_fft: int, n_mels: int) -> torch.Tensor:
    """Slaney-normalised mel filter bank (HF default for audio encoders)."""
    f_min, f_max = 0.0, sr / 2
    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0, sr / 2, n_freqs, dtype=torch.float64)
    m_min = hz_to_mel(f_min)
    m_max = hz_to_mel(f_max)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2, dtype=torch.float64)
    f_pts = mel_to_hz(m_pts)
    filters = torch.zeros((n_mels, n_freqs), dtype=torch.float64)
    for m in range(n_mels):
        left, center, right = f_pts[m], f_pts[m + 1], f_pts[m + 2]
        lower = (fft_freqs - left) / (center - left)
        upper = (right - fft_freqs) / (right - center)
        filters[m] = torch.clamp(torch.minimum(lower, upper), min=0.0)
        enorm = 2.0 / (right - left)
        filters[m] *= enorm
    return filters.float()


def hz_to_mel(hz: float) -> float:
    return 1127.0 * np.log(1.0 + hz / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.exp(mel / 1127.0) - 1.0)


def build_encoder() -> Qwen3OmniMoeAudioEncoder:
    cfg_full = json.loads((MODEL_DIR / "config.json").read_text())
    ac = cfg_full["thinker_config"]["audio_config"]
    cfg = Qwen3OmniMoeAudioEncoderConfig(
        d_model=ac["d_model"],
        encoder_attention_heads=ac["encoder_attention_heads"],
        encoder_ffn_dim=ac["encoder_ffn_dim"],
        encoder_layers=ac["encoder_layers"],
        num_mel_bins=ac["num_mel_bins"],
        max_source_positions=ac["max_source_positions"],
        downsample_hidden_size=ac["downsample_hidden_size"],
        output_dim=ac["output_dim"],
        n_window=ac["n_window"],
        n_window_infer=ac["n_window_infer"],
        conv_chunksize=ac["conv_chunksize"],
        activation_function=ac["activation_function"],
        attention_dropout=0.0,
        dropout=0.0,
        scale_embedding=ac["scale_embedding"],
    )
    encoder = Qwen3OmniMoeAudioEncoder(cfg)
    return encoder


def load_audio_weights(encoder: Qwen3OmniMoeAudioEncoder) -> None:
    state = load_file(str(MODEL_DIR / "model.safetensors"))
    new_state: dict[str, torch.Tensor] = {}
    prefix = "thinker.audio_tower."
    for k, v in state.items():
        if k.startswith(prefix):
            new_state[k[len(prefix):]] = v.to(torch.float32)
    missing, unexpected = encoder.load_state_dict(new_state, strict=False)
    print(f"audio encoder load: missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)
    if missing:
        for m in missing[:5]:
            print(f"  missing: {m}", file=sys.stderr)
    if unexpected:
        for u in unexpected[:5]:
            print(f"  unexpected: {u}", file=sys.stderr)


def dump_f32(path: Path, tensor: torch.Tensor) -> None:
    arr = tensor.detach().contiguous().to(torch.float32).cpu().numpy()
    with path.open("wb") as f:
        f.write(struct.pack("<I", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<i", d))
        f.write(arr.tobytes())
    flat = arr.flatten()
    print(
        f"{path.name}: shape={list(arr.shape)} min={flat.min():.4f} max={flat.max():.4f} "
        f"mean={flat.mean():.4f} std={flat.std():.4f}",
        file=sys.stderr,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mel = load_audio_mel(WAV_PATH).squeeze(0)
    dump_f32(OUT_DIR / "qwen_ref_mel.bin", mel)

    encoder = build_encoder()
    encoder.eval()
    load_audio_weights(encoder)

    activations: dict[str, torch.Tensor] = {}

    def save(name: str):
        def hook(_module, _inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            activations[name] = t.detach().clone()
        return hook

    encoder.conv2d1.register_forward_hook(save("conv2d1"))
    encoder.conv2d2.register_forward_hook(save("conv2d2"))
    encoder.conv2d3.register_forward_hook(save("conv2d3"))
    encoder.conv_out.register_forward_hook(save("conv_out"))
    for idx, layer in enumerate(encoder.layers):
        if idx in (0, 1, 5, 17):
            layer.register_forward_hook(save(f"layer_{idx:02d}"))
    encoder.ln_post.register_forward_hook(save("ln_post"))
    encoder.proj1.register_forward_hook(save("proj1"))
    encoder.proj2.register_forward_hook(save("proj2"))

    feature_lens = torch.tensor([mel.shape[-1]], dtype=torch.long)
    with torch.no_grad():
        out = encoder(input_features=mel, feature_lens=feature_lens)
    enc_out = out.last_hidden_state
    dump_f32(OUT_DIR / "qwen_ref_encoder_out.bin", enc_out)

    for name, t in activations.items():
        dump_f32(OUT_DIR / f"qwen_ref_enc_{name}.bin", t)


if __name__ == "__main__":
    main()
