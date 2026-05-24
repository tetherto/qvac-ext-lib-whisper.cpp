"""Download one short clip per language from Google FLEURS via Hugging Face.

Bypasses the ``datasets`` audio decoder (which now pulls torchcodec / ffmpeg)
by downloading a single parquet shard per locale with ``huggingface_hub`` and
decoding the embedded WAV bytes with ``soundfile``.

Output: examples/ios-simulator-demo/QwenAsrDemo/samples/<lang>.wav (+ .txt).
"""

from __future__ import annotations

import io
import struct
import sys
from pathlib import Path
from typing import List, Tuple

LANGUAGES: List[Tuple[str, str, str]] = [
    ("zh", "cmn_hans_cn", "Chinese (Mandarin)"),
    ("ja", "ja_jp",       "Japanese"),
    ("ko", "ko_kr",       "Korean"),
    ("hi", "hi_in",       "Hindi"),
    ("ar", "ar_eg",       "Arabic (Egyptian)"),
    ("ru", "ru_ru",       "Russian"),
    ("es", "es_419",      "Spanish (LatAm)"),
    ("it", "it_it",       "Italian"),
]

TARGET_SR        = 16000
DESIRED_MIN_SECS = 5.0
DESIRED_MAX_SECS = 18.0


def write_wav_pcm16(path: Path, samples, sample_rate: int) -> None:
    import numpy as np
    s   = np.asarray(samples, dtype=np.float32)
    s   = np.clip(s, -1.0, 1.0)
    pcm = (s * 32767.0).astype(np.int16)
    data_bytes  = pcm.tobytes()
    n_channels  = 1
    bps         = 16
    byte_rate   = sample_rate * n_channels * bps // 8
    block_align = n_channels * bps // 8
    riff_size   = 36 + len(data_bytes)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", n_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bps))
        f.write(b"data")
        f.write(struct.pack("<I", len(data_bytes)))
        f.write(data_bytes)


def resample_to_16k(samples, src_sr: int):
    import numpy as np
    if src_sr == TARGET_SR:
        return np.asarray(samples, dtype=np.float32)
    try:
        import librosa
        return librosa.resample(np.asarray(samples, dtype=np.float32),
                                orig_sr=src_sr, target_sr=TARGET_SR)
    except ImportError:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(src_sr, TARGET_SR)
        return resample_poly(np.asarray(samples, dtype=np.float32),
                             up=TARGET_SR // g, down=src_sr // g).astype("float32")


def list_first_test_shard(repo_id: str, locale: str) -> str:
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    candidates = [f for f in files
                  if f.startswith(f"parquet-data/{locale}/test-") and f.endswith(".parquet")]
    if not candidates:
        raise RuntimeError(f"no test parquet shard for {locale}")
    return sorted(candidates)[0]


def decode_audio_bytes(blob: bytes):
    import soundfile as sf
    samples, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    if hasattr(samples, "ndim") and samples.ndim == 2:
        samples = samples.mean(axis=1)
    return samples, sr


def pick_first_in_range(table, min_secs: float, max_secs: float):
    audio_col = table["audio"]
    text_col  = (table["transcription"] if "transcription" in table.column_names
                 else table["raw_transcription"] if "raw_transcription" in table.column_names
                 else None)
    fallback = None
    for i, raw in enumerate(audio_col):
        d = raw.as_py()
        if not isinstance(d, dict) or "bytes" not in d or d["bytes"] is None:
            continue
        try:
            samples, sr = decode_audio_bytes(d["bytes"])
        except Exception:
            continue
        dur = len(samples) / float(sr)
        text = text_col[i].as_py() if text_col is not None else ""
        item = (samples, sr, text, dur)
        if min_secs <= dur <= max_secs:
            return item
        if fallback is None:
            fallback = item
    return fallback


def download_language(code: str, locale: str, label: str, out_dir: Path) -> None:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    shard_path = list_first_test_shard("google/fleurs", locale)
    print(f"[{code}] {label} ({locale}) -> {shard_path}")
    local_parquet = hf_hub_download("google/fleurs", shard_path, repo_type="dataset")
    table = pq.read_table(local_parquet)
    chosen = pick_first_in_range(table, DESIRED_MIN_SECS, DESIRED_MAX_SECS)
    if chosen is None:
        sys.stderr.write(f"[{code}] no decodable rows in {shard_path}\n")
        return
    samples, sr, text, dur = chosen
    samples_16k = resample_to_16k(samples, sr)
    wav_path = out_dir / f"{code}.wav"
    txt_path = out_dir / f"{code}.txt"
    write_wav_pcm16(wav_path, samples_16k, TARGET_SR)
    txt_path.write_text((text or "").strip() + "\n", encoding="utf-8")
    print(f"[{code}]   {dur:.1f}s @ {sr} Hz -> 16 kHz; wrote {wav_path.stat().st_size//1024} KiB")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    out_dir      = project_root / "examples" / "ios-simulator-demo" / "QwenAsrDemo" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for code, locale, label in LANGUAGES:
        try:
            download_language(code, locale, label, out_dir)
        except Exception as ex:
            sys.stderr.write(f"[{code}] FAILED: {ex}\n")
            failures.append(code)
    print()
    print(f"summary: ok={len(LANGUAGES) - len(failures)} fail={len(failures)} ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
