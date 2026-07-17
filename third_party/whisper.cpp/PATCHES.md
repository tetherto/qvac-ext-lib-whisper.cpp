# QVAC patches on vendored whisper.cpp

This directory is a **git subtree** of [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp),
currently pinned at **v1.9.1** (`f049fff9`; machine-readable pin in
`UPSTREAM_PIN`, consumed by the divergence-guard CI). It is *upstream-tracking,
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

Each row must be self-contained: state what the change does and why it
exists, so the manifest is meaningful without access to any tracker.

| File | Change | Why |
|---|---|---|
| `src/whisper.cpp` | vocab-logits slice + decoder QKV matmul fusion | decode-time perf on mobile GPUs (quantized decode on Adreno-class hardware): the vocab-logits matmul is sliced to only the rows sampling actually needs instead of the full vocab, and the decoder's three Q/K/V matmuls are fused into one — upstream does neither |
| `src/whisper-logits-slice.h` | new file | helper implementing the vocab-logits slice above |
| `include/whisper.h` | API additions | public controls for the logits-slice behavior so bindings/addons can opt in per-decode |
| `src/CMakeLists.txt` | build glue | wire whisper-logits-slice.h; install/export adjustments for the vcpkg port |
| `CMakeLists.txt` | install/export + robustness | export `whisper-targets`, headers under `include/whisper/`; guard `git-vars`/js-bindings config steps so source-tarball (vcpkg) builds work |
| `cmake/git-vars.cmake` | robustness | tolerate non-git source trees (vcpkg tarballs) |
| `cmake/whisper-config.cmake.in` | config fixes | correct find_package config for system-ggml consumers |
| `CMakeLists.txt`, `src/CMakeLists.txt`, `examples/CMakeLists.txt` | `WHISPER_BUILD_PARAKEET` option (default ON) | gate upstream's bundled parakeet: its `parakeet`/`parakeet-cli` target names collide with `engines/parakeet`; the umbrella sets it OFF. Upstreaming candidate |
| `CMakeLists.txt` | `include(GNUInstallDirs)` moved before `add_subdirectory(src)` | the whisper INSTALL_INTERFACE expands `${CMAKE_INSTALL_INCLUDEDIR}`; unset it exports a bogus `/whisper` path breaking install-tree `find_package(whisper)`. Absorbed from the registry port's patch. Upstreaming candidate |

Marker convention: new QVAC code blocks carry a `QVAC` reference in a nearby
comment where practical.

## Updating this pin

See `docs/UPSTREAM-SYNC.md`. Short version: `git subtree pull
--prefix third_party/whisper.cpp upstream <tag> --squash`, resolve conflicts
(expected only in the files above), re-verify this manifest, run the full CI
matrix.
