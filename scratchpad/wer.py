#!/usr/bin/env python3
"""Dependency-free WER for QVAC-21623 GPU-vs-CPU quality gate.

Reference = CPU (-ng) transcript; hypothesis = GPU transcript. WER measures the
*quality degradation from the GPU optimizations* (Master's "difference <= 1%").
Also supports run-to-run consistency: max pairwise WER among N GPU runs.

Usage:
  wer.py REF HYP                      -> WER(HYP vs REF)
  wer.py --consistency F1 F2 [F3...]  -> max pairwise WER among files (+ all pairs)
Files may be raw whisper-cli output (timestamped segment lines are auto-extracted)
or plain text. Use '-' for stdin.
"""
import re
import sys

TS = re.compile(r'^\s*\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*(.*)$')


def read_text(path):
    if path == '-':
        raw = sys.stdin.read()
    else:
        with open(path, 'r', errors='replace') as f:
            raw = f.read()
    # Extract whisper-cli segment text if timestamped lines are present.
    segs = [m.group(1) for m in (TS.match(ln) for ln in raw.splitlines()) if m]
    return ' '.join(segs) if segs else raw


def normalize(text):
    text = text.lower()
    # keep apostrophes inside words, drop other punctuation
    text = re.sub(r"[^\w'\s]", ' ', text)
    return text.split()


def edit_distance(a, b):
    # Levenshtein over word lists (O(len(a)*len(b)) time, O(len(b)) space)
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, wb in enumerate(b, 1):
            cost = 0 if wa == wb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def wer(ref_words, hyp_words):
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def main():
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == '--consistency':
        files = args[1:]
        texts = [normalize(read_text(f)) for f in files]
        worst = 0.0
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                w = wer(texts[i], texts[j])
                print(f"  {files[i]} vs {files[j]}: WER={w*100:.2f}%")
                worst = max(worst, w)
        print(f"MAX_PAIRWISE_WER={worst*100:.2f}%")
    elif len(args) == 2:
        ref = normalize(read_text(args[0]))
        hyp = normalize(read_text(args[1]))
        w = wer(ref, hyp)
        d = edit_distance(ref, hyp)
        print(f"ref_words={len(ref)} hyp_words={len(hyp)} edits={d} WER={w*100:.2f}%")
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == '__main__':
    main()
