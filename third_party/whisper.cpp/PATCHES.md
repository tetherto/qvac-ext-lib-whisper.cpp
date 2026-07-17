# QVAC patches on vendored whisper.cpp

This directory is a **git subtree** of [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp),
currently pinned at **v1.9.1** (`f049fff9`). It is *upstream-tracking,
minimally divergent*: the only files that may differ from the pinned upstream
tag are the ones listed below. CI enforces this (divergence-guard job) —
if you change anything in this directory, either it's an upstream sync
(`git subtree pull`, see `docs/UPSTREAM-SYNC.md`) or this manifest must be
updated in the same PR.

**ggml is special:** `ggml/` in this subtree must stay byte-identical to
upstream. All QVAC ggml patches live in
[`qvac-ext-ggml`](https://github.com/tetherto/qvac-ext-ggml) (`speech` branch),
which ships as the `ggml-speech` vcpkg port; the umbrella build forces
`WHISPER_USE_SYSTEM_GGML=ON`, so the vendored `ggml/` is never compiled.

## Manifest

| File | Change | Why |
|---|---|---|
| `src/whisper.cpp` | vocab-logits slice + decoder QKV matmul fusion | QVAC-21623: Adreno q8 decode perf; slices the vocab logits matmul and fuses the decoder QKV matmuls |
| `src/whisper-logits-slice.h` | new file | QVAC-21623: logits-slice helper |
| `include/whisper.h` | API additions | QVAC-21623: expose logits-slice controls |
| `src/CMakeLists.txt` | build glue | wire whisper-logits-slice.h; install/export adjustments for the vcpkg port |
| `CMakeLists.txt` | install/export + robustness | export `whisper-targets`, headers under `include/whisper/`; guard `git-vars`/js-bindings config steps so source-tarball (vcpkg) builds work |
| `cmake/git-vars.cmake` | robustness | tolerate non-git source trees (vcpkg tarballs) |
| `cmake/whisper-config.cmake.in` | config fixes | correct find_package config for system-ggml consumers |

Marker convention: new QVAC code blocks carry a `QVAC` reference in a nearby
comment where practical.

## Updating this pin

See `docs/UPSTREAM-SYNC.md`. Short version: `git subtree pull
--prefix third_party/whisper.cpp upstream <tag> --squash`, resolve conflicts
(expected only in the files above), re-verify this manifest, run the full CI
matrix.
