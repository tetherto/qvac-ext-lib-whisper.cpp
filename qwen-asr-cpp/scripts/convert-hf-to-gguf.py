#!/usr/bin/env python3
"""
convert-hf-to-gguf.py
=====================

Convert a Qwen3-ASR HuggingFace checkpoint (config.json + model.safetensors +
vocab.json + merges.txt) into a single .gguf file consumable by the v0.2 GGUF
backend (`qwen::gguf::Engine`).

This script is hand-written for the Qwen3-ASR layout. It does NOT import or
copy code from any closed-license / unlicensed third-party Qwen3-ASR port.
The GGUF write pipeline mirrors the public conventions used by llama.cpp's
own `convert_hf_to_gguf.py` (MIT) and the GGUF spec at
https://github.com/ggerganov/ggml/blob/master/docs/gguf.md.

Tensor name mapping
-------------------

We keep the HuggingFace prefixes verbatim so the loader can map back to the
mathematical reference in `vendor/qwen-asr-c/`:

    thinker.audio_tower.conv2d1.weight                  -> enc.conv2d1.weight
    thinker.audio_tower.layers.{i}.self_attn.q_proj.*   -> enc.blk.{i}.attn_q.*
    thinker.audio_tower.layers.{i}.self_attn_layer_norm.* -> enc.blk.{i}.attn_norm.*
    thinker.audio_tower.layers.{i}.fc1.*                -> enc.blk.{i}.ffn_gate.*
    thinker.audio_tower.layers.{i}.fc2.*                -> enc.blk.{i}.ffn_down.*
    thinker.audio_tower.layers.{i}.final_layer_norm.*   -> enc.blk.{i}.ffn_norm.*
    thinker.audio_tower.ln_post.*                       -> enc.ln_post.*
    thinker.audio_tower.proj1.*                         -> enc.proj1.*
    thinker.audio_tower.proj2.*                         -> enc.proj2.*
    thinker.model.embed_tokens.weight                   -> token_embd.weight
    thinker.lm_head.weight                              -> output.weight
    thinker.model.norm.weight                           -> output_norm.weight
    thinker.model.layers.{i}.self_attn.q_proj.weight    -> blk.{i}.attn_q.weight
      (and same scheme for k, v, o, q_norm, k_norm)
    thinker.model.layers.{i}.input_layernorm.*          -> blk.{i}.attn_norm.*
    thinker.model.layers.{i}.post_attention_layernorm.* -> blk.{i}.ffn_norm.*
    thinker.model.layers.{i}.mlp.gate_proj.*            -> blk.{i}.ffn_gate.*
    thinker.model.layers.{i}.mlp.up_proj.*              -> blk.{i}.ffn_up.*
    thinker.model.layers.{i}.mlp.down_proj.*            -> blk.{i}.ffn_down.*

GGUF metadata keys
------------------

We use the `qwen3-asr` namespace for ASR-specific keys and reuse the
`qwen3` namespace for the decoder LLM (matches llama.cpp's Qwen3 family).
"""

from __future__ import annotations

import argparse
import json
import mmap
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("convert-hf-to-gguf.py: numpy is required (pip install numpy)")


GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_BF16 = 30
GGML_TYPE_Q8_0 = 8


def map_tensor_name(hf_name: str) -> str | None:
    encoder_prefix = "thinker.audio_tower."
    decoder_prefix = "thinker.model.layers."
    if hf_name == "thinker.model.embed_tokens.weight":
        return "token_embd.weight"
    if hf_name == "thinker.lm_head.weight":
        return "output.weight"
    if hf_name == "thinker.model.norm.weight":
        return "output_norm.weight"
    if hf_name.startswith(encoder_prefix):
        rest = hf_name[len(encoder_prefix):]
        return map_encoder_name(rest)
    if hf_name.startswith(decoder_prefix):
        rest = hf_name[len(decoder_prefix):]
        return map_decoder_name(rest)
    return None


def map_encoder_name(rest: str) -> str | None:
    if rest.startswith("layers."):
        return map_encoder_layer(rest)
    if rest in ("conv2d1.weight", "conv2d1.bias",
                "conv2d2.weight", "conv2d2.bias",
                "conv2d3.weight", "conv2d3.bias",
                "conv_out.weight",
                "ln_post.weight", "ln_post.bias",
                "proj1.weight", "proj1.bias",
                "proj2.weight", "proj2.bias"):
        return f"enc.{rest}"
    return None


def map_encoder_layer(rest: str) -> str | None:
    parts = rest.split(".")
    if len(parts) < 3 or parts[0] != "layers":
        return None
    idx = parts[1]
    tail = ".".join(parts[2:])
    mapping = {
        "self_attn.q_proj":       "attn_q",
        "self_attn.k_proj":       "attn_k",
        "self_attn.v_proj":       "attn_v",
        "self_attn.out_proj":     "attn_output",
        "self_attn_layer_norm":   "attn_norm",
        "fc1":                    "ffn_gate",
        "fc2":                    "ffn_down",
        "final_layer_norm":       "ffn_norm",
    }
    for k, v in mapping.items():
        if tail.startswith(k + "."):
            suffix = tail[len(k) + 1:]
            return f"enc.blk.{idx}.{v}.{suffix}"
    return None


def map_decoder_name(rest: str) -> str | None:
    parts = rest.split(".")
    if len(parts) < 2:
        return None
    idx = parts[0]
    tail = ".".join(parts[1:])
    mapping = {
        "self_attn.q_proj":             "attn_q",
        "self_attn.k_proj":             "attn_k",
        "self_attn.v_proj":             "attn_v",
        "self_attn.o_proj":             "attn_output",
        "self_attn.q_norm":             "attn_q_norm",
        "self_attn.k_norm":             "attn_k_norm",
        "input_layernorm":              "attn_norm",
        "post_attention_layernorm":     "ffn_norm",
        "mlp.gate_proj":                "ffn_gate",
        "mlp.up_proj":                  "ffn_up",
        "mlp.down_proj":                "ffn_down",
    }
    for k, v in mapping.items():
        if tail.startswith(k + "."):
            suffix = tail[len(k) + 1:]
            return f"blk.{idx}.{v}.{suffix}"
    return None


def pack_string(s: str) -> bytes:
    data = s.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def pack_kv_uint32(key: str, value: int) -> bytes:
    return pack_string(key) + struct.pack("<II", GGUF_TYPE_UINT32, value)


def pack_kv_int32(key: str, value: int) -> bytes:
    return pack_string(key) + struct.pack("<Ii", GGUF_TYPE_INT32, value)


def pack_kv_float32(key: str, value: float) -> bytes:
    return pack_string(key) + struct.pack("<If", GGUF_TYPE_FLOAT32, value)


def pack_kv_string(key: str, value: str) -> bytes:
    return pack_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + pack_string(value)


def pack_kv_array_string(key: str, values: list[str]) -> bytes:
    head = pack_string(key) + struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_STRING, len(values))
    body = b"".join(pack_string(v) for v in values)
    return head + body


def pack_kv_array_int32(key: str, values: list[int]) -> bytes:
    head = pack_string(key) + struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_INT32, len(values))
    body = struct.pack(f"<{len(values)}i", *values)
    return head + body


def align_up(offset: int, alignment: int) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def bf16_to_f16_bytes(raw: bytes) -> bytes:
    arr = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    fp32 = (arr << 16).view(np.float32)
    return fp32.astype(np.float16).tobytes()


def convert_raw(raw: bytes, src_dtype: str, keep_bf16: bool) -> tuple[bytes, int]:
    if src_dtype == "BF16":
        if keep_bf16:
            return raw, GGML_TYPE_BF16
        return bf16_to_f16_bytes(raw), GGML_TYPE_F16
    if src_dtype == "F16":
        return raw, GGML_TYPE_F16
    if src_dtype == "F32":
        return raw, GGML_TYPE_F32
    raise SystemExit(f"convert: unsupported source dtype '{src_dtype}'")


def load_vocab(model_dir: Path) -> tuple[list[str], list[int]]:
    vocab_file = model_dir / "vocab.json"
    if not vocab_file.exists():
        raise SystemExit(f"convert: vocab.json not found at {vocab_file}")
    with vocab_file.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    added_ids: dict[int, str] = {}
    added_specials: set[int] = set()
    tok_cfg = model_dir / "tokenizer_config.json"
    if tok_cfg.exists():
        cfg = json.loads(tok_cfg.read_text(encoding="utf-8"))
        for k, v in (cfg.get("added_tokens_decoder") or {}).items():
            tid = int(k)
            content = v.get("content") if isinstance(v, dict) else str(v)
            if content is not None:
                added_ids[tid] = content
                if isinstance(v, dict) and v.get("special"):
                    added_specials.add(tid)
    tj = model_dir / "tokenizer.json"
    if tj.exists():
        try:
            data = json.loads(tj.read_text(encoding="utf-8"))
            for entry in (data.get("added_tokens") or []):
                tid = int(entry.get("id", -1))
                content = entry.get("content")
                if tid >= 0 and content is not None and tid not in added_ids:
                    added_ids[tid] = content
                    if entry.get("special"):
                        added_specials.add(tid)
        except Exception:
            pass
    max_id = max([v for v in vocab.values()] + list(added_ids.keys()) + [-1])
    tokens: list[str] = [""] * (max_id + 1)
    types:  list[int] = [1] * (max_id + 1)
    for tok, tid in vocab.items():
        tokens[tid] = tok
    for tid, content in added_ids.items():
        tokens[tid] = content
        types[tid]  = 3 if tid in added_specials else 4
    for i, t in enumerate(tokens):
        if not t:
            tokens[i] = f"<unused_{i}>"
            types[i]  = 5
    return tokens, types


def load_merges(model_dir: Path) -> list[str]:
    merges_file = model_dir / "merges.txt"
    if not merges_file.exists():
        return []
    out = []
    with merges_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if i == 0 and line.startswith("#"):
                continue
            if line:
                out.append(line)
    return out


class SafetensorsShard:
    def __init__(self, path: Path):
        self.path = path
        self.fh   = path.open("rb")
        header_len = struct.unpack("<Q", self.fh.read(8))[0]
        header     = json.loads(self.fh.read(header_len).decode("utf-8"))
        self.metadata     = header.pop("__metadata__", {})
        self.tensors_meta = header
        self.data_start   = 8 + header_len
        self.fh.seek(0, 2)
        self.file_size = self.fh.tell()
        self.mm = mmap.mmap(self.fh.fileno(), 0, access=mmap.ACCESS_READ)

    def keys(self):
        return list(self.tensors_meta.keys())

    def info(self, name):
        meta = self.tensors_meta[name]
        dtype = meta["dtype"]
        shape = meta["shape"]
        beg, end = meta["data_offsets"]
        return dtype, shape, self.data_start + beg, self.data_start + end

    def raw(self, name):
        _, _, beg, end = self.info(name)
        return bytes(self.mm[beg:end])

    def close(self):
        self.mm.close()
        self.fh.close()


def open_safetensors(model_dir: Path):
    single = model_dir / "model.safetensors"
    if single.exists():
        return [SafetensorsShard(single)]
    index = model_dir / "model.safetensors.index.json"
    if not index.exists():
        raise SystemExit(f"convert: no model.safetensors[.index.json] in {model_dir}")
    meta = json.loads(index.read_text(encoding="utf-8"))
    weight_map = meta["weight_map"]
    shards = sorted(set(weight_map.values()))
    return [SafetensorsShard(model_dir / s) for s in shards]


def iter_tensors(shards):
    for sh in shards:
        for name in sh.keys():
            yield name, sh


def emit(model_dir: Path, out_path: Path, keep_bf16: bool) -> None:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    audio_cfg = config.get("audio_config", config.get("thinker_config", {}).get("audio_config", {}))
    text_cfg  = config.get("text_config",  config.get("thinker_config", {}).get("text_config", {}))
    hidden_size       = int(text_cfg.get("hidden_size",       1024))
    n_layers          = int(text_cfg.get("num_hidden_layers", 28))
    n_heads           = int(text_cfg.get("num_attention_heads", 16))
    n_kv_heads        = int(text_cfg.get("num_key_value_heads", 8))
    head_dim          = int(text_cfg.get("head_dim", hidden_size // max(n_heads, 1)))
    intermediate_size = int(text_cfg.get("intermediate_size", 3072))
    rope_base         = float(text_cfg.get("rope_theta", 1000000.0))
    rms_eps           = float(text_cfg.get("rms_norm_eps", 1e-6))
    vocab_size        = int(text_cfg.get("vocab_size", 151936))
    enc_d_model       = int(audio_cfg.get("d_model", audio_cfg.get("hidden_size", 1024)))
    enc_n_heads       = int(audio_cfg.get("encoder_attention_heads", audio_cfg.get("num_attention_heads", 16)))
    enc_n_layers      = int(audio_cfg.get("encoder_layers", audio_cfg.get("num_hidden_layers", 18)))
    enc_ffn_dim       = int(audio_cfg.get("encoder_ffn_dim", audio_cfg.get("intermediate_size", enc_d_model * 4)))
    enc_output_dim    = int(audio_cfg.get("output_dim", hidden_size))
    enc_n_mels        = int(audio_cfg.get("num_mel_bins", 128))
    audio_token_id    = int(config.get("thinker_config", {}).get("audio_token_id", 151676))
    audio_start_id    = int(config.get("thinker_config", {}).get("audio_start_token_id", 151669))
    audio_end_id      = int(config.get("thinker_config", {}).get("audio_end_token_id", 151670))

    shards = open_safetensors(model_dir)

    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []
    skipped = 0
    for hf_name, sh in iter_tensors(shards):
        new_name = map_tensor_name(hf_name)
        if new_name is None:
            skipped += 1
            continue
        dtype, shape, _, _ = sh.info(hf_name)
        raw_src = sh.raw(hf_name)
        try:
            raw_out, ggml_type = convert_raw(raw_src, dtype, keep_bf16)
        except SystemExit:
            print(f"  skip ({dtype}): {hf_name}", file=sys.stderr)
            skipped += 1
            continue
        tensors.append((new_name, tuple(shape), ggml_type, raw_out))
    print(f"  mapped {len(tensors)} tensors, skipped {skipped}", file=sys.stderr)

    tokens, token_types = load_vocab(model_dir)
    merges = load_merges(model_dir)

    metadata = bytearray()
    metadata += pack_kv_string("general.architecture", "qwen3-asr")
    metadata += pack_kv_string("general.name", "qwen3-asr-0.6b")
    metadata += pack_kv_uint32("qwen3-asr.text.context_length", int(text_cfg.get("max_position_embeddings", 8192)))
    metadata += pack_kv_uint32("qwen3-asr.text.embedding_length", hidden_size)
    metadata += pack_kv_uint32("qwen3-asr.text.feed_forward_length", intermediate_size)
    metadata += pack_kv_uint32("qwen3-asr.text.block_count", n_layers)
    metadata += pack_kv_uint32("qwen3-asr.text.attention.head_count", n_heads)
    metadata += pack_kv_uint32("qwen3-asr.text.attention.head_count_kv", n_kv_heads)
    metadata += pack_kv_uint32("qwen3-asr.text.attention.head_dim", head_dim)
    metadata += pack_kv_uint32("qwen3-asr.text.vocab_size", vocab_size)
    metadata += pack_kv_float32("qwen3-asr.text.attention.layer_norm_rms_epsilon", rms_eps)
    metadata += pack_kv_float32("qwen3-asr.text.rope.freq_base", rope_base)
    metadata += pack_kv_uint32("qwen3-asr.encoder.embedding_length", enc_d_model)
    metadata += pack_kv_uint32("qwen3-asr.encoder.feed_forward_length", enc_ffn_dim)
    metadata += pack_kv_uint32("qwen3-asr.encoder.block_count", enc_n_layers)
    metadata += pack_kv_uint32("qwen3-asr.encoder.attention.head_count", enc_n_heads)
    metadata += pack_kv_uint32("qwen3-asr.encoder.output_dim", enc_output_dim)
    metadata += pack_kv_uint32("qwen3-asr.encoder.num_mel_bins", enc_n_mels)
    metadata += pack_kv_uint32("qwen3-asr.audio.token_id",       audio_token_id)
    metadata += pack_kv_uint32("qwen3-asr.audio.start_token_id", audio_start_id)
    metadata += pack_kv_uint32("qwen3-asr.audio.end_token_id",   audio_end_id)
    metadata += pack_kv_string("tokenizer.ggml.model", "gpt2")
    metadata += pack_kv_array_string("tokenizer.ggml.tokens", tokens)
    metadata += pack_kv_array_int32("tokenizer.ggml.token_type", token_types)
    if merges:
        metadata += pack_kv_array_string("tokenizer.ggml.merges", merges)
    metadata += pack_kv_uint32("tokenizer.ggml.bos_token_id", 151643)
    metadata += pack_kv_uint32("tokenizer.ggml.eos_token_id", 151645)
    metadata += pack_kv_uint32("tokenizer.ggml.padding_token_id", 151643)

    n_kv_pairs = count_kv_pairs(metadata)

    alignment = 32
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensors), n_kv_pairs))
        f.write(metadata)
        for name, shape, ggml_type, _ in tensors:
            f.write(pack_string(name))
            f.write(struct.pack("<I", len(shape)))
            for d in reversed(shape):
                f.write(struct.pack("<Q", int(d)))
            f.write(struct.pack("<I", ggml_type))
            f.write(struct.pack("<Q", 0))
        data_start = align_up(f.tell(), alignment)
        f.write(b"\x00" * (data_start - f.tell()))
        offsets = []
        for _, _, _, raw in tensors:
            offsets.append(f.tell() - data_start)
            f.write(raw)
            pad = align_up(f.tell(), alignment) - f.tell()
            f.write(b"\x00" * pad)
        patch_tensor_offsets(out_path, tensors, offsets, n_kv_pairs, len(metadata))


def count_kv_pairs(buf: bytes) -> int:
    count = 0
    pos = 0
    while pos < len(buf):
        klen = struct.unpack_from("<Q", buf, pos)[0]
        pos += 8 + klen
        vtype = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        pos = skip_value(buf, pos, vtype)
        count += 1
    return count


def skip_value(buf: bytes, pos: int, vtype: int) -> int:
    if vtype in (GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL):  return pos + 1
    if vtype in (GGUF_TYPE_UINT16, GGUF_TYPE_INT16):                return pos + 2
    if vtype in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32): return pos + 4
    if vtype in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64): return pos + 8
    if vtype == GGUF_TYPE_STRING:
        klen = struct.unpack_from("<Q", buf, pos)[0]
        return pos + 8 + klen
    if vtype == GGUF_TYPE_ARRAY:
        sub = struct.unpack_from("<I", buf, pos)[0]; pos += 4
        n   = struct.unpack_from("<Q", buf, pos)[0]; pos += 8
        for _ in range(n):
            pos = skip_value(buf, pos, sub)
        return pos
    raise SystemExit(f"convert: unknown gguf value type {vtype}")


def patch_tensor_offsets(out_path: Path,
                         tensors: list,
                         offsets: list[int],
                         n_kv_pairs: int,
                         metadata_len: int) -> None:
    header_len = 4 + 4 + 8 + 8 + metadata_len
    with out_path.open("r+b") as f:
        pos = header_len
        for (name, shape, _, _), off in zip(tensors, offsets):
            f.seek(pos)
            nlen = struct.unpack("<Q", f.read(8))[0]
            f.seek(nlen, 1)
            f.seek(4, 1)
            f.seek(8 * len(shape), 1)
            f.seek(4, 1)
            f.write(struct.pack("<Q", off))
            pos = f.tell()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, help="HF checkpoint directory (with model.safetensors etc.)")
    p.add_argument("--out",       required=True, help="Output .gguf file")
    p.add_argument("--keep-bf16", action="store_true", help="Keep BF16 weights (default: cast to F16)")
    args = p.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    out_path  = Path(args.out).expanduser().resolve()
    if not model_dir.is_dir():
        sys.exit(f"convert: model dir not found: {model_dir}")
    emit(model_dir, out_path, keep_bf16=args.keep_bf16)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"convert: wrote {out_path} ({size_mb:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
