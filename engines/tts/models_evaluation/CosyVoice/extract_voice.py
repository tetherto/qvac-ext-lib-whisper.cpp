#!/usr/bin/env python3
"""Extract CosyVoice3 zero-shot prompt tensors for a new baked voice.

Runs the PyTorch frontend on a reference clip + transcript and dumps the four
tensors the native engine bakes into voice.gguf (see bake-cosyvoice3-voice.py):

  prompt_stok.npy   [P] int32   llm_prompt_speech_token  (LM prompt speech tokens)
  prompt_token.npy  [P] int32   flow_prompt_speech_token (flow prompt speech tokens)
  prompt_feat.npy   [T,80] f32  prompt_speech_feat       (prompt mel)
  embedding.npy     [192] f32   flow_embedding           (CAM++ speaker embedding)

Usage (venv active, from models_evaluation/CosyVoice):
  PYTHONPATH=CosyVoice:CosyVoice/third_party/Matcha-TTS \\
  python3 extract_voice.py --model-dir models/Fun-CosyVoice3-0.5B \\
      --prompt-audio /path/ref.wav --prompt-text "transcript" --out-dir artifacts/voice-en
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-audio", required=True)
    ap.add_argument("--prompt-text", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from cosyvoice.cli.cosyvoice import CosyVoice3
    os.makedirs(args.out_dir, exist_ok=True)
    cv = CosyVoice3(args.model_dir, load_trt=False, load_vllm=False, fp16=False)

    prompt_text = "You are a helpful assistant.<|endofprompt|>" + args.prompt_text
    mi = cv.frontend.frontend_zero_shot(
        "placeholder tts text.", prompt_text, args.prompt_audio, cv.sample_rate, "")
    print("model_input keys:", sorted(mi.keys()))

    def np1(x):
        return np.ascontiguousarray(x[0].detach().cpu().numpy())

    pstok = np1(mi["llm_prompt_speech_token"]).astype(np.int32)
    ptok = np1(mi["flow_prompt_speech_token"]).astype(np.int32)
    pfeat = np1(mi["prompt_speech_feat"]).astype(np.float32)   # [T, 80]
    emb = np1(mi["flow_embedding"]).astype(np.float32)         # [192]

    np.save(os.path.join(args.out_dir, "prompt_stok.npy"), pstok)
    np.save(os.path.join(args.out_dir, "prompt_token.npy"), ptok)
    np.save(os.path.join(args.out_dir, "prompt_feat.npy"), pfeat)
    np.save(os.path.join(args.out_dir, "embedding.npy"), emb)

    with open(os.path.join(args.out_dir, "prompt_text.txt"), "w") as f:
        f.write(args.prompt_text)

    print(f"prompt_stok {pstok.shape}  prompt_token {ptok.shape}  "
          f"prompt_feat {pfeat.shape}  embedding {emb.shape}")
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
