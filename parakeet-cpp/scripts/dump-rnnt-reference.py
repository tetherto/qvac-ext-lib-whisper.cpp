#!/usr/bin/env python3
"""Dump NeMo greedy reference for plain RNN-T parity (token IDs + transcript).

Adapted from dump-tdt-reference.py for a hybrid EncDecHybridRNNTCTCBPEModel
(e.g. nvidia/stt_ka_fastconformer_hybrid_large_pc): forces the RNN-T decoder
(not the CTC aux head) and dumps the greedy token stream so the C++
test-rnnt-decoder-parity can assert bit-identical greedy decoding.

    <out>/
        token_ids.npy   (N,)              NeMo greedy RNN-T token IDs
        transcript.txt                    NeMo greedy transcript
        encoder_out.npy  (T_enc, d_model) NeMo encoder output (optional parity)
        mel.npy          (n_mels, T_mel)  post-preprocessor log-mel (optional)

Greedy decoding is deterministic; the C++ side must reproduce token_ids exactly.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav", type=Path, required=True, help="Input mono 16 kHz wav")
    p.add_argument("--out", type=Path, default=Path("artifacts/rnnt-ref"), help="Output directory for dumps")
    p.add_argument("--nemo-model", type=Path,
                   default=Path("models/stt_ka_fastconformer_hybrid_large_pc.nemo"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-encoder-dump", action="store_true",
                   help="Skip mel/encoder_out dumps (token parity only)")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    import nemo.collections.asr as nemo_asr

    print(f"[rnnt-ref] restoring {args.nemo_model}", file=sys.stderr)
    model = nemo_asr.models.ASRModel.restore_from(str(args.nemo_model), map_location=args.device)
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    # Hybrid RNN-T/CTC: force the RNN-T decoder (keeps the model's default greedy
    # decoding strategy -- the configuration NeMo's published WER is measured at).
    if hasattr(model, "cur_decoder"):
        try:
            model.change_decoding_strategy(decoder_type="rnnt")
            print("[rnnt-ref] forced decoder_type=rnnt (hybrid model)", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[rnnt-ref] WARN: change_decoding_strategy failed ({e}); using model default",
                  file=sys.stderr)

    try:
        gd = model.cfg.decoding.greedy
        print(f"[rnnt-ref] decoding.greedy.max_symbols={gd.get('max_symbols', None)}", file=sys.stderr)
    except Exception:
        pass

    if not args.no_encoder_dump:
        import soundfile as sf
        wav, sr = sf.read(str(args.wav), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
            sr = 16000
        wav_t = torch.from_numpy(wav).unsqueeze(0).to(args.device)
        length_t = torch.tensor([len(wav)], dtype=torch.long, device=args.device)
        with torch.inference_mode():
            mel, mel_len = model.preprocessor(input_signal=wav_t, length=length_t)
            np.save(args.out / "mel.npy", mel[0].detach().cpu().numpy().astype(np.float32))
            enc_out, enc_len = model.encoder(audio_signal=mel, length=mel_len)
            enc_np = enc_out[0].permute(1, 0).detach().cpu().numpy().astype(np.float32)
            np.save(args.out / "encoder_out.npy", enc_np)
            print(f"[rnnt-ref] encoder_out: {enc_np.shape} (T_enc, d_model)", file=sys.stderr)

    print(f"[rnnt-ref] transcribing {args.wav} (NeMo greedy RNN-T)...", file=sys.stderr)
    hyps = model.transcribe([str(args.wav)], batch_size=1)
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    h0 = hyps[0] if isinstance(hyps, list) else hyps

    text = h0.text if hasattr(h0, "text") else str(h0)
    (args.out / "transcript.txt").write_text(text + "\n", encoding="utf-8")
    print(f"[rnnt-ref] transcript: {text!r}", file=sys.stderr)

    token_ids = None
    if hasattr(h0, "y_sequence"):
        ts = h0.y_sequence
        token_ids = (ts.detach().cpu().numpy() if hasattr(ts, "detach")
                     else np.asarray(ts)).astype(np.int32)
    if token_ids is not None:
        np.save(args.out / "token_ids.npy", token_ids)
        print(f"[rnnt-ref] token_ids: {token_ids.shape} -> token_ids.npy "
              f"(first 24: {token_ids[:24].tolist()})", file=sys.stderr)
    else:
        print("[rnnt-ref] WARN: hypothesis has no y_sequence; no token_ids.npy", file=sys.stderr)

    print(f"[rnnt-ref] done -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
