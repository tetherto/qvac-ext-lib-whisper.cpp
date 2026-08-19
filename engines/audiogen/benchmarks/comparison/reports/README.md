# ACE-Step comparison reports

Reviewed, committed results live under `reports/<target>/`.

```text
reports/
  README.md
  mac-arm64/
    verification-report.md
    cpu.json
    cpu.md
    metal.json
    metal.md
    manifest.json
    logs/          # optional; gitignored if large
    samples/       # optional WAVs; prefer checksums if large
```

Copy files from `out/` after a completed run. Do not commit GGUFs or full
sample trees unless they are small enough for git.

Each target needs a human-authored `verification-report.md` with the thirteen
sections listed in the comparison README.
