#!/usr/bin/env python3
"""Per-stage cosine similarity / rel_l2 between two --dump-stages directories.

Stage dumps are int32 {ndim=2, d0, d1} followed by d0*d1 float32 (StageDump::write).
"""
import pathlib
import struct
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        _ndim, d0, d1 = struct.unpack("iii", f.read(12))
        data = np.fromfile(f, dtype=np.float32)
    return data, [d0, d1]


def main():
    a_dir, b_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    stages = sorted(p.name for p in a_dir.glob("*.bin"))
    print(f"{'stage':<20}{'shape':>18}{'cosine':>12}{'rel_l2':>12}{'max_abs':>12}")
    worst = 1.0
    for s in stages:
        b_path = b_dir / s
        if not b_path.exists():
            continue
        a, dims = load(a_dir / s)
        b, _ = load(b_path)
        n = min(a.size, b.size)
        a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else 1.0
        rel = float(np.linalg.norm(a - b) / (na + 1e-30))
        print(f"{s:<20}{str(dims):>18}{cos:12.6f}{rel:12.6f}{np.abs(a - b).max():12.4g}")
        worst = min(worst, cos)
    print(f"\nworst-stage cosine: {worst:.6f}  ({'PASS' if worst >= 0.999 else 'FAIL'} at >= 0.999)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
