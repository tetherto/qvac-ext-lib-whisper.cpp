#!/usr/bin/env python3
"""Compare two raw f32 tensor dumps written by dump_hf_reference.py / C++ engine.

Each file starts with: u32 ndim; i32 dims[ndim]; then f32 data.
"""

import struct
import sys
from pathlib import Path

import numpy as np


def load(path: Path):
    with path.open("rb") as f:
        ndim = struct.unpack("<I", f.read(4))[0]
        dims = list(struct.unpack(f"<{ndim}i", f.read(4 * ndim)))
        nelem = int(np.prod(dims))
        data = np.frombuffer(f.read(nelem * 4), dtype=np.float32).reshape(dims)
    return data


def main():
    a = load(Path(sys.argv[1]))
    b = load(Path(sys.argv[2]))
    print(f"A: shape={list(a.shape)} mean={a.mean():.4f} std={a.std():.4f} range=[{a.min():.4f},{a.max():.4f}]")
    print(f"B: shape={list(b.shape)} mean={b.mean():.4f} std={b.std():.4f} range=[{b.min():.4f},{b.max():.4f}]")

    if a.shape != b.shape:
        print("DIFFERENT SHAPES: trimming to common")
        if a.ndim == 2:
            r = min(a.shape[0], b.shape[0])
            c = min(a.shape[1], b.shape[1])
            a = a[:r, :c]
            b = b[:r, :c]
        else:
            n = min(a.size, b.size)
            a = a.flatten()[:n]
            b = b.flatten()[:n]

    diff = (a - b).astype(np.float64)
    max_abs = np.abs(diff).max()
    rms = np.sqrt((diff ** 2).mean())
    cos = float((a.flatten() @ b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    print(f"max_abs_diff={max_abs:.4f} rms={rms:.4f} cosine={cos:.6f}")

    if a.ndim == 2:
        per_row_cos = []
        for i in range(a.shape[0]):
            x, y = a[i], b[i]
            d = float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
            per_row_cos.append(d)
        per_row_cos = np.array(per_row_cos)
        print(
            f"per-row cosine: mean={per_row_cos.mean():.4f} min={per_row_cos.min():.4f} "
            f"max={per_row_cos.max():.4f}"
        )
        print("first 8 rows cosine:", [f"{x:.4f}" for x in per_row_cos[:8]])


if __name__ == "__main__":
    main()
