#!/usr/bin/env python3
"""On-device whisper bench for QVAC-21623. Thermal-gated, N runs, parses timings,
saves per-run transcript for WER. Reference config: fa OFF (default), 4 threads.

  bench.py --bin whisper-cli-p0 --model ggml-tiny-q8_0.bin --clip jfk.wav \
           --mode gpu --runs 3 --tag t-tiny-jfk-gpu [--extra "-fa"]

Emits median total/encode ms + RTF and writes logs+transcripts to scratchpad/p0-logs/.
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
import time

SERIAL = "10BD1C1LEF0001R"
DEV_DIR = "/data/local/tmp/w21623"
ZONE = "/sys/class/thermal/thermal_zone63/temp"
COOL_C = 44.0
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p0-logs")

TS = re.compile(r'^\s*\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->')


def adb(*args, timeout=300):
    return subprocess.run(["adb", "-s", SERIAL, *args], capture_output=True,
                          text=True, timeout=timeout)


def temp_c():
    r = adb("shell", "cat", ZONE, timeout=15)
    try:
        return int(r.stdout.strip()) / 1000.0
    except ValueError:
        return -1.0


def cool_wait(maxwait=180):
    t0 = time.time()
    while time.time() - t0 < maxwait:
        t = temp_c()
        if 0 < t < COOL_C:
            return t
        time.sleep(6)
    return temp_c()


def parse(out):
    d = {}
    for key, pat in (("total", r"total time =\s*([\d.]+)"),
                     ("encode", r"encode time =\s*([\d.]+)"),
                     ("decode", r"decode time =\s*([\d.]+)"),
                     ("batchd", r"batchd time =\s*([\d.]+)"),
                     ("sec", r"\(\d+ samples,\s*([\d.]+) sec\)")):
        m = re.search(pat, out)
        d[key] = float(m.group(1)) if m else None
    d["transcript"] = " ".join(ln.split("]", 1)[1].strip()
                               for ln in out.splitlines() if TS.match(ln))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--mode", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--env", default="")  # e.g. "GGML_OPENCL_FA_ADRENO=1"
    a = ap.parse_args()
    os.makedirs(LOGDIR, exist_ok=True)

    ng = "-ng" if a.mode == "cpu" else ""
    env = (a.env + " ") if a.env else ""
    cmd = f"cd {DEV_DIR} && {env}./{a.bin} -m {a.model} -f {a.clip} {ng} {a.extra} 2>&1"

    totals, encodes = [], []
    for i in range(a.runs):
        t = cool_wait()
        r = adb("shell", cmd, timeout=600)
        d = parse(r.stdout)
        log = os.path.join(LOGDIR, f"{a.tag}-r{i}.log")
        txt = os.path.join(LOGDIR, f"{a.tag}-r{i}.txt")
        with open(log, "w") as f:
            f.write(r.stdout)
        with open(txt, "w") as f:
            f.write(d["transcript"] + "\n")
        rtf = (d["total"] / (d["sec"] * 1000)) if d["total"] and d["sec"] else None
        totals.append(d["total"])
        encodes.append(d["encode"])
        print(f"  r{i} T={temp_c():.0f}C total={d['total']:.0f} enc={d['encode']:.0f} "
              f"dec={d['decode']:.0f} bat={d['batchd']:.0f} RTF={rtf:.4f} :: {d['transcript'][:70]}")

    tot = [x for x in totals if x]
    enc = [x for x in encodes if x]
    sec = d["sec"]
    med_tot = statistics.median(tot) if tot else 0
    med_enc = statistics.median(enc) if enc else 0
    print(f"[{a.tag}] median total={med_tot:.0f}ms encode={med_enc:.0f}ms "
          f"RTF={med_tot/(sec*1000):.4f} (audio {sec}s, n={len(tot)})")


if __name__ == "__main__":
    main()
