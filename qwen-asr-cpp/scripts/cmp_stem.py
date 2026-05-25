#!/usr/bin/env python3
"""Compare conv stem outputs: HF vs C++ for chunk 0."""

import struct
import sys
from pathlib import Path
import numpy as np


def load(path, is_ggml=False):
    """Load a raw f32 dump. is_ggml=True reverses dims (ggml ne[0] is fastest)."""
    with open(path, "rb") as f:
        ndim = struct.unpack("<I", f.read(4))[0]
        dims = list(struct.unpack(f"<{ndim}i", f.read(4 * ndim)))
        n = int(np.prod(dims))
        arr = np.frombuffer(f.read(n * 4), dtype=np.float32)
        if is_ggml:
            return arr.reshape(list(reversed(dims)))
        return arr.reshape(dims)


def diff(name, a, b):
    a = a.flatten()
    b = b.flatten()
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    cos = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    print(f"{name}: A.size={a.size} B.size={b.size} cos={cos:.6f} max_abs={np.abs(a-b).max():.4f}")


def main():
    hf_conv3 = load("/tmp/qwen_ref_enc_conv2d3.bin")
    cpp_conv3 = load("/tmp/qwen_cpp_conv3.bin", is_ggml=True)
    print(f"HF conv2d3 shape={list(hf_conv3.shape)}")
    print(f"CPP conv3 (ggml-reversed) shape={list(cpp_conv3.shape)}")

    import torch
    import torch.nn.functional as F
    hf_chunk0 = hf_conv3[0]
    hf_chunk0_gelu = F.gelu(torch.from_numpy(hf_chunk0.copy())).numpy()
    cpp_chw = cpp_conv3.reshape(480, 16, 13)
    diff("conv3 raw (CPP vs HF raw)",  cpp_chw, hf_chunk0)
    diff("conv3 raw (CPP vs HF GELU)", cpp_chw, hf_chunk0_gelu)

    hf_convout = load("/tmp/qwen_ref_enc_conv_out.bin")
    cpp_convout = load("/tmp/qwen_cpp_convout.bin", is_ggml=True)
    hf_chunk0_co = hf_convout[0]
    print(f"HF conv_out chunk0 shape={hf_chunk0_co.shape}")
    print(f"CPP conv_out shape={cpp_convout.shape}")
    diff("conv_out", cpp_convout, hf_chunk0_co)

    cpp_flat = load("/tmp/qwen_cpp_flat.bin", is_ggml=True)
    print(f"CPP flat shape={cpp_flat.shape}")
    # HF order: (c=480 outer, f=16 inner). hf_chunk0_gelu: (C=480, H=16, W=13).
    # Want (T=13, C=480, F=16) → flatten → (T, C*F).
    hf_flat = hf_chunk0_gelu.transpose(2, 0, 1).reshape(13, -1)
    diff("flat c-outer-f-inner (CPP vs HF GELU)", cpp_flat, hf_flat)
    # Try f-outer-c-inner ordering
    hf_flat2 = hf_chunk0_gelu.transpose(2, 1, 0).reshape(13, -1)
    diff("flat f-outer-c-inner", cpp_flat, hf_flat2)


if __name__ == "__main__":
    main()
