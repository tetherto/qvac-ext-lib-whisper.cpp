#!/usr/bin/env python3
"""Convert a Parler-TTS checkpoint (mini-v1 / large-v1) to a single tts-cpp GGUF.

Contains: T5 text encoder, Parler decoder LM, DAC decoder (decode path only),
T5 unigram tokenizer (tokens + scores). DAC/quantizer weight-norm (weight_g /
weight_v) is folded into plain weights at convert time. The DAC encoder and the
quantizer in_proj tensors are intentionally dropped (encode path unused).

Two-sided completeness: every source tensor must be either consumed or on the
explicit ignore list, and every expected destination must be produced —
otherwise the converter aborts.

Usage:
  python3 convert-parler-to-gguf.py --model-id parler-tts/parler-tts-mini-v1 \
      --dtype f32 --out tts-cpp/models/parler-mini-v1-f32.gguf
"""

import argparse
import json
import os
import sys

import numpy as np


def log(msg):
    print(f"[convert-parler] {msg}", flush=True)


def fold_weight_norm(g, v):
    """weight_norm dim=0: w[i] = v[i] * g[i] / ||v[i]||_2  (norm over all dims but 0)."""
    v = np.asarray(v, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32).reshape(v.shape[0], *([1] * (v.ndim - 1)))
    norm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=tuple(range(1, v.ndim)), keepdims=True))
    return (v * (g / norm.astype(np.float32))).astype(np.float32)


class Converter:
    def __init__(self, get, src_names):
        self.get = get                      # name -> np.ndarray
        self.src_names = set(src_names)
        self.consumed = set()
        self.out = {}                       # dest name -> np.ndarray (f32)

    def take(self, src):
        if src not in self.src_names:
            raise KeyError(f"source tensor missing: {src}")
        self.consumed.add(src)
        return np.asarray(self.get(src), dtype=np.float32)

    def has(self, src):
        return src in self.src_names

    def emit(self, dest, arr):
        assert dest not in self.out, f"duplicate dest {dest}"
        self.out[dest] = np.ascontiguousarray(arr, dtype=np.float32)

    def move(self, dest, src):
        self.emit(dest, self.take(src))

    def move_folded(self, dest, src_prefix):
        g = self.take(src_prefix + ".weight_g")
        v = self.take(src_prefix + ".weight_v")
        self.emit(dest + ".weight", fold_weight_norm(g, v))
        self.move(dest + ".bias", src_prefix + ".bias")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="parler-tts/parler-tts-mini-v1")
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import gguf

    snap = snapshot_download(args.model_id)
    log(f"snapshot: {snap}")

    with open(os.path.join(snap, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(snap, "generation_config.json")) as f:
        gen_cfg = json.load(f)
    with open(os.path.join(snap, "tokenizer.json")) as f:
        tok_json = json.load(f)

    t5 = cfg["text_encoder"]
    dec = cfg["decoder"]
    assert cfg["model_type"] == "parler_tts"
    assert dec.get("rope_embeddings", False) is False, "v1 models use sinusoidal positions"
    assert cfg.get("prompt_cross_attention", False) is False, "v1 models prepend the prompt"

    # ---- gather source tensors (single or sharded safetensors) ----
    files = [f for f in os.listdir(snap) if f.endswith(".safetensors")]
    handles = [safe_open(os.path.join(snap, f), framework="np") for f in sorted(files)]
    src_of = {}
    for h in handles:
        for n in h.keys():
            src_of[n] = h
    log(f"{len(src_of)} source tensors across {len(files)} file(s)")

    cv = Converter(lambda n: src_of[n].get_tensor(n), src_of.keys())

    # ---------------- T5 encoder ----------------
    n_t5 = t5["num_layers"]
    cv.move("t5.embed_tokens.weight", "text_encoder.shared.weight")
    for i in range(n_t5):
        b = f"text_encoder.encoder.block.{i}.layer"
        d = f"t5.blk.{i}"
        cv.move(f"{d}.attn_norm.weight", f"{b}.0.layer_norm.weight")
        for s, t in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_o")):
            cv.move(f"{d}.{t}.weight", f"{b}.0.SelfAttention.{s}.weight")
        cv.move(f"{d}.ffn_norm.weight", f"{b}.1.layer_norm.weight")
        cv.move(f"{d}.ffn_gate.weight", f"{b}.1.DenseReluDense.wi_0.weight")
        cv.move(f"{d}.ffn_up.weight", f"{b}.1.DenseReluDense.wi_1.weight")
        cv.move(f"{d}.ffn_down.weight", f"{b}.1.DenseReluDense.wo.weight")
    cv.move("t5.blk.0.attn_rel_b.weight",
            "text_encoder.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
    cv.move("t5.output_norm.weight", "text_encoder.encoder.final_layer_norm.weight")

    # ---------------- glue ----------------
    # enc_to_dec_proj exists (and is applied) only when t5.d_model != dec.hidden_size
    enc_to_dec = cv.has("enc_to_dec_proj.weight")
    expect_proj = t5["d_model"] != dec["hidden_size"]
    assert enc_to_dec == expect_proj, \
        f"enc_to_dec_proj presence ({enc_to_dec}) contradicts dims (t5 {t5['d_model']} vs dec {dec['hidden_size']})"
    if enc_to_dec:
        cv.move("enc_to_dec.weight", "enc_to_dec_proj.weight")
        cv.move("enc_to_dec.bias", "enc_to_dec_proj.bias")
    cv.move("dec.embed_prompts.weight", "embed_prompts.weight")

    # ---------------- decoder LM ----------------
    n_dec = dec["num_hidden_layers"]
    n_cb = dec["num_codebooks"]
    cv.move("dec.embed_positions.weight", "decoder.model.decoder.embed_positions.weights")
    for k in range(n_cb):
        cv.move(f"dec.embed_tokens.{k}.weight", f"decoder.model.decoder.embed_tokens.{k}.weight")
        cv.move(f"dec.lm_heads.{k}.weight", f"decoder.lm_heads.{k}.weight")
    for i in range(n_dec):
        b = f"decoder.model.decoder.layers.{i}"
        d = f"dec.blk.{i}"
        cv.move(f"{d}.attn_norm.weight", f"{b}.self_attn_layer_norm.weight")
        cv.move(f"{d}.attn_norm.bias", f"{b}.self_attn_layer_norm.bias")
        for s, t in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("out_proj", "attn_o")):
            cv.move(f"{d}.{t}.weight", f"{b}.self_attn.{s}.weight")
        cv.move(f"{d}.cross_norm.weight", f"{b}.encoder_attn_layer_norm.weight")
        cv.move(f"{d}.cross_norm.bias", f"{b}.encoder_attn_layer_norm.bias")
        for s, t in (("q_proj", "cross_q"), ("k_proj", "cross_k"), ("v_proj", "cross_v"), ("out_proj", "cross_o")):
            cv.move(f"{d}.{t}.weight", f"{b}.encoder_attn.{s}.weight")
        cv.move(f"{d}.ffn_norm.weight", f"{b}.final_layer_norm.weight")
        cv.move(f"{d}.ffn_norm.bias", f"{b}.final_layer_norm.bias")
        cv.move(f"{d}.ffn_up.weight", f"{b}.fc1.weight")
        cv.move(f"{d}.ffn_down.weight", f"{b}.fc2.weight")
    cv.move("dec.output_norm.weight", "decoder.model.decoder.layer_norm.weight")
    cv.move("dec.output_norm.bias", "decoder.model.decoder.layer_norm.bias")

    # ---------------- DAC (decode path) ----------------
    n_q = cfg["audio_encoder"]["num_codebooks"]
    for k in range(n_q):
        q = f"audio_encoder.model.quantizer.quantizers.{k}"
        cv.move(f"dac.quant.{k}.codebook.weight", f"{q}.codebook.weight")
        cv.move_folded(f"dac.quant.{k}.out_proj", f"{q}.out_proj")

    dacm = "audio_encoder.model.decoder.model"
    cv.move_folded("dac.dec.conv_in", f"{dacm}.0")
    conv_in_w = cv.out["dac.dec.conv_in.weight"]
    dac_latent, dac_decoder_dim = conv_in_w.shape[1], conv_in_w.shape[0]

    rates = []
    n_blocks = 0
    while cv.has(f"{dacm}.{1 + n_blocks}.block.1.weight_v"):
        n_blocks += 1
    for i in range(n_blocks):
        blk = f"{dacm}.{1 + i}"
        d = f"dac.dec.blk.{i}"
        cv.move(f"{d}.snake.alpha", f"{blk}.block.0.alpha")
        cv.move_folded(f"{d}.convt", f"{blk}.block.1")
        k = cv.out[f"{d}.convt.weight"].shape[2]
        assert k % 2 == 0, f"convT kernel {k} not 2*stride"
        rates.append(k // 2)
        for j in range(3):
            ru = f"{blk}.block.{2 + j}.block"
            cv.move(f"{d}.res.{j}.snake1.alpha", f"{ru}.0.alpha")
            cv.move_folded(f"{d}.res.{j}.conv1", f"{ru}.1")
            cv.move(f"{d}.res.{j}.snake2.alpha", f"{ru}.2.alpha")
            cv.move_folded(f"{d}.res.{j}.conv2", f"{ru}.3")
    cv.move(f"dac.dec.snake_out.alpha", f"{dacm}.{1 + n_blocks}.alpha")
    cv.move_folded("dac.dec.conv_out", f"{dacm}.{2 + n_blocks}")
    hop = int(np.prod(rates))
    log(f"DAC: latent={dac_latent} decoder_dim={dac_decoder_dim} rates={rates} hop={hop}")
    assert cv.out["dac.dec.conv_out.weight"].shape[0] == 1, "conv_out must emit mono"

    # ---------------- completeness check ----------------
    IGNORE_PREFIXES = (
        "audio_encoder.model.encoder.",          # encode path unused
    )
    IGNORE_SUBSTR = (
        ".in_proj.",                             # quantizer encode-side proj
    )
    leftovers = [n for n in cv.src_names - cv.consumed
                 if not n.startswith(IGNORE_PREFIXES) and not any(s in n for s in IGNORE_SUBSTR)]
    if leftovers:
        for n in sorted(leftovers)[:40]:
            log(f"UNCONSUMED: {n}")
        raise SystemExit(f"{len(leftovers)} unconsumed source tensors — refusing to write GGUF")
    log(f"consumed {len(cv.consumed)}/{len(cv.src_names)} source tensors "
        f"({len(cv.src_names) - len(cv.consumed)} intentionally ignored), {len(cv.out)} dest tensors")

    # ---------------- tokenizer (fast unigram from tokenizer.json) ----------------
    assert tok_json["model"]["type"] == "Unigram", tok_json["model"]["type"]
    norm = tok_json.get("normalizer") or {}
    norms = norm.get("normalizers", [norm]) if norm else []
    charsmap = b""
    for nn in norms:
        if nn.get("type") == "Precompiled":
            import base64
            charsmap = base64.b64decode(nn["precompiled_charsmap"])
    vocab = tok_json["model"]["vocab"]          # [[piece, score], ...]
    unk_id = tok_json["model"]["unk_id"]
    tokens = [p for p, _ in vocab]
    scores = [float(s) for _, s in vocab]
    for at in tok_json.get("added_tokens", []):
        if at["id"] >= len(tokens):
            tokens.extend([""] * (at["id"] + 1 - len(tokens)))
            scores.extend([-1e9] * (at["id"] + 1 - len(scores)))
        tokens[at["id"]] = at["content"]
    log(f"tokenizer: {len(tokens)} pieces, unk_id={unk_id}")

    # ---------------- write GGUF ----------------
    w = gguf.GGUFWriter(args.out, "parler")
    w.add_string("parler.arch", "parler-tts")
    variant = args.model_id.rsplit("/", 1)[-1]
    w.add_string("parler.variant", variant)
    w.add_string("parler.reference_repo", args.model_id)
    w.add_string("parler.ftype", args.dtype)

    w.add_uint32("parler.t5.n_layer", t5["num_layers"])
    w.add_uint32("parler.t5.d_model", t5["d_model"])
    w.add_uint32("parler.t5.d_ff", t5["d_ff"])
    w.add_uint32("parler.t5.n_head", t5["num_heads"])
    w.add_uint32("parler.t5.d_kv", t5["d_kv"])
    w.add_uint32("parler.t5.rel_buckets", t5["relative_attention_num_buckets"])
    w.add_uint32("parler.t5.rel_max_dist", t5["relative_attention_max_distance"])
    w.add_float32("parler.t5.rms_eps", t5["layer_norm_epsilon"])
    w.add_uint32("parler.t5.vocab_size", t5["vocab_size"])

    w.add_uint32("parler.dec.n_layer", dec["num_hidden_layers"])
    w.add_uint32("parler.dec.d_model", dec["hidden_size"])
    w.add_uint32("parler.dec.n_head", dec["num_attention_heads"])
    w.add_uint32("parler.dec.d_ff", dec["ffn_dim"])
    w.add_uint32("parler.dec.n_codebooks", dec["num_codebooks"])
    w.add_uint32("parler.dec.vocab_size", dec["vocab_size"])
    w.add_float32("parler.dec.ln_eps", 1e-5)   # nn.LayerNorm default, verified vs live model
    w.add_uint32("parler.dec.bos_token_id", gen_cfg["bos_token_id"])
    w.add_uint32("parler.dec.eos_token_id", gen_cfg["eos_token_id"])
    w.add_uint32("parler.dec.pad_token_id", gen_cfg["pad_token_id"])
    w.add_uint32("parler.dec.decoder_start_token_id", gen_cfg["decoder_start_token_id"])
    w.add_uint32("parler.dec.max_position", dec["max_position_embeddings"])
    w.add_bool("parler.enc_to_dec", enc_to_dec)

    w.add_uint32("parler.gen.max_length", gen_cfg["max_length"])
    # 0 = disabled. NOT defaulted: modeling_parler_tts.py has no code-side
    # min_new_tokens fallback; only mini-v1's generation_config.json sets it.
    w.add_uint32("parler.gen.min_new_tokens", gen_cfg.get("min_new_tokens", 0))
    w.add_bool("parler.gen.do_sample", gen_cfg.get("do_sample", False))
    w.add_float32("parler.gen.temperature", gen_cfg.get("temperature", 1.0))
    w.add_uint32("parler.gen.top_k", gen_cfg.get("top_k", 50))

    w.add_uint32("parler.dac.sample_rate", cfg["audio_encoder"]["sampling_rate"])
    w.add_uint32("parler.dac.n_quantizers", n_q)
    w.add_uint32("parler.dac.codebook_size", cfg["audio_encoder"]["codebook_size"])
    w.add_uint32("parler.dac.latent_dim", dac_latent)
    w.add_uint32("parler.dac.decoder_dim", dac_decoder_dim)
    w.add_uint32("parler.dac.hop", hop)
    w.add_array("parler.dac.rates", rates)

    w.add_string("tokenizer.ggml.model", "t5")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    if charsmap:
        w.add_precompiled_charsmap(charsmap)
        log(f"tokenizer: precompiled charsmap {len(charsmap)} bytes")
    w.add_uint32("tokenizer.ggml.unknown_token_id", unk_id)
    w.add_uint32("tokenizer.ggml.eos_token_id", 1)
    w.add_uint32("tokenizer.ggml.padding_token_id", 0)
    w.add_bool("parler.tokenizer.add_eos", True)  # T5 appends </s>; verified in reference dump

    # dtype policy: f16 only for decoder matmul weights / embeddings.
    # Numerically delicate tensors stay f32: norms, biases, alphas, rel-pos
    # bias, positional table, all DAC tensors — and the ENTIRE T5 encoder:
    # flan-T5 activations overflow the f16 range (the well-known T5 fp16
    # trap), and ggml's f16 mul_mat converts activation rows to f16 for the
    # dot product, which turns the encoder output into NaN.
    def keep_f32(name):
        return (name.endswith(".bias") or ".alpha" in name or "norm" in name
                or name.startswith("t5.")
                or name == "dec.embed_positions.weight"
                or name.startswith("dac."))

    n_f16 = 0
    for name in sorted(cv.out):
        arr = cv.out[name]
        if args.dtype == "f16" and not keep_f32(name):
            w.add_tensor(name, arr.astype(np.float16))
            n_f16 += 1
        else:
            w.add_tensor(name, arr)
    log(f"tensors: {len(cv.out)} total, {n_f16} cast to f16")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log(f"wrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    sys.exit(main())
