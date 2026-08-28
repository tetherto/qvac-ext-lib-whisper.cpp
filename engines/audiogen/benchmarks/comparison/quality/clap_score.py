#!/usr/bin/env python3
"""Score generated WAVs against prompt text with LAION CLAP. Prints one JSON object to stdout."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

DEFAULT_MODEL = 'laion/larger_clap_music_and_speech'


def fail (message: str, code: int = 1) -> None:
    json.dump({'ok': False, 'error': message}, sys.stdout)
    sys.stdout.write('\n')
    raise SystemExit(code)


def mix_to_mono (waveform: Any) -> Any:
    import numpy as np
    if waveform.ndim == 1:
        return waveform.astype(np.float32)
    return waveform.mean(axis=1).astype(np.float32)


def load_mono_wav (path: str) -> tuple[Any, int]:
    import soundfile as sf
    waveform, sample_rate = sf.read(path, always_2d=True)
    return mix_to_mono(waveform), int(sample_rate)


def load_clap (model_id: str, revision: str | None, device: str):
    import torch
    from transformers import ClapModel, ClapProcessor

    kwargs = {}
    if revision:
        kwargs['revision'] = revision
    processor = ClapProcessor.from_pretrained(model_id, **kwargs)
    model = ClapModel.from_pretrained(model_id, **kwargs)
    model.eval()
    model.to(device)
    return model, processor


def clap_sampling_rate(processor: Any) -> int:
    extractor = getattr(processor, 'feature_extractor', None)
    rate = getattr(extractor, 'sampling_rate', None)
    if not rate:
        raise RuntimeError('CLAP processor did not report a sampling_rate')
    return int(rate)


def score_waveform (model: Any, processor: Any, waveform: Any, sample_rate: int, text: str, device: str) -> float:
    import torch

    text_inputs = processor(text=[text], return_tensors='pt', padding=True)
    audio_inputs = processor(
        audios=[waveform],
        sampling_rate=sample_rate,
        return_tensors='pt',
        padding=True
    )
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    audio_inputs = {key: value.to(device) for key, value in audio_inputs.items()}
    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)
        audio_emb = model.get_audio_features(**audio_inputs)
        text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
        audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)
        score = (text_emb * audio_emb).sum(dim=-1)
    return float(score.item())


def score_item (model: Any, processor: Any, item: dict[str, Any], device: str) -> dict[str, Any]:
    started = time.perf_counter()
    wav_path = item.get('wav')
    text = item.get('text')
    item_id = item.get('id')
    if not wav_path or not text:
        return {'id': item_id, 'ok': False, 'score': None, 'error': 'item needs wav and text', 'elapsedMs': 0}
    try:
        waveform, sample_rate = load_mono_wav(wav_path)
        score = score_waveform(model, processor, waveform, sample_rate, text, device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {'id': item_id, 'ok': True, 'score': score, 'error': None, 'elapsedMs': elapsed_ms}
    except Exception as error:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {'id': item_id, 'ok': False, 'score': None, 'error': str(error), 'elapsedMs': elapsed_ms}


def score_items (model: Any, processor: Any, items: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    return [score_item(model, processor, item, device) for item in items]


def resolve_device (requested: str) -> str:
    import torch
    if requested == 'cpu':
        return 'cpu'
    if requested == 'mps':
        if not torch.backends.mps.is_available():
            fail('ACESTEP_CLAP_DEVICE=mps but MPS is not available')
        return 'mps'
    fail(f'unsupported CLAP device: {requested}')
    return 'cpu'


def load_batch (path: str) -> dict[str, Any]:
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def build_items_from_args (args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.batch:
        payload = load_batch(args.batch)
        items = payload.get('items')
        if not isinstance(items, list) or not items:
            fail('CLAP batch file must contain a non-empty items array')
        return items
    if not args.wav or args.text is None:
        fail('pass --batch or both --wav and --text')
    return [{'id': args.wav, 'wav': args.wav, 'text': args.text}]


def parse_args (argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Score WAVs with LAION CLAP; JSON on stdout only')
    parser.add_argument('--batch', help='JSON file with items[{id,wav,text}]')
    parser.add_argument('--wav', help='single WAV path')
    parser.add_argument('--text', help='single caption (or caption+lyrics) string')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--revision', default=None)
    parser.add_argument('--device', default='cpu')
    return parser.parse_args(argv)


def main (argv: list[str]) -> None:
    args = parse_args(argv)
    try:
        import numpy  # noqa: F401
        import soundfile  # noqa: F401
        import torch  # noqa: F401
        from transformers import ClapModel, ClapProcessor  # noqa: F401
    except ImportError as error:
        fail(
            'CLAP Python dependencies missing. Review quality/requirements.txt, then '
            f'run: python3 -m pip install -r quality/requirements.txt ({error})'
        )
    device = resolve_device(args.device)
    items = build_items_from_args(args)
    try:
        model, processor = load_clap(args.model, args.revision, device)
    except Exception as error:
        fail(f'failed to load CLAP model {args.model}: {error}')
    scores = score_items(model, processor, items, device)
    resolved_revision = args.revision
    if hasattr(model, 'config'):
        resolved_revision = getattr(model.config, '_commit_hash', args.revision) or args.revision
    result = {
        'ok': True,
        'model': args.model,
        'revision': resolved_revision,
        'samplingRate': clap_sampling_rate(processor),
        'device': device,
        'scores': scores
    }
    json.dump(result, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main(sys.argv[1:])
