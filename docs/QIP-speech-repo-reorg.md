# QIP: Reorganize `qvac-ext-lib-whisper.cpp` into a symmetric multi-engine speech repo

| | |
|---|---|
| **Status** | Draft — for approval |
| **Owner** | Omar Gad |
| **Date** | 2026-07-17 |
| **Repo** | `tetherto/qvac-ext-lib-whisper.cpp` |
| **Related** | Ticket 2 (`transcription-ggml` NPM pkg), Ticket 4 (umbrella vcpkg port), repo rename (deferred) |

## Summary

Restructure the repo into a symmetric layout: upstream `whisper.cpp` vendored under `third_party/whisper.cpp/` via **git subtree**, our engines under `engines/{whisper,parakeet,tts}` (future: `engines/qwen-asr`), and a feature-gated umbrella `CMakeLists.txt` at the root that we own. Alongside the reorg, stand up **in-repo CI** that builds each engine and runs its existing `ctest` suites, so the team stops depending on the 3-PR downstream chain (`repo → registry → qvac`) for basic verification.

One material amendment to the original plan: the vendored tree cannot be "pristine, never hand-edited" — **we already carry ~374 lines of patches on upstream files** (see [Audit findings](#audit-findings)). The policy is amended to *upstream-tracking, minimally divergent*, with a committed patch manifest and a CI guard that enforces the divergence stays declared and minimal.

## Problem

Our C++ speech engines live in asymmetric, hard-to-extend places:

- Upstream `whisper.cpp` sits at the repo **root**, interleaved with our build glue, org workflows, and patches — there is no boundary between "upstream" and "ours".
- `parakeet-cpp/` and `tts-cpp/` are isolated sibling folders with their own conventions, not built or tested by anything in this repo's CI.
- Upstream now ships **its own `parakeet`** (`src/parakeet.cpp`, `include/parakeet.h`, C API `parakeet_*`) as of upstream PR #3735 / v1.9.1 — the root tree is no longer even purely "whisper", and it name-collides conceptually with our `parakeet-cpp` (C++ `namespace parakeet`).
- Upstream syncs are a manual squashed re-apply of the whole tree (commit `cb91a378`, single parent). Every sync re-resolves *all* of our divergence by hand instead of only the files we actually touch.
- There is no clean home for future engines (`qwen-asr`).
- **No CI in this repo exercises our engines.** The active workflows are upstream's build matrix (partially adapted in QVAC-19386) plus org-wide `check-approvals` / `security-baseline`. Real verification of parakeet/tts happens three repos downstream in `tetherto/qvac` (`on-pr-tts-ggml`), behind a registry pin bump and a `verified` label — a multi-day loop for changes that a `ctest` run would catch in minutes.

## Audit findings

Facts verified against the repo on 2026-07-17 that correct assumptions in the original draft:

1. **The root tree is a fork, not a vendored copy.** Diff vs the upstream `v1.9.1` tag on upstream-owned paths: `src/whisper.cpp` +205 lines (vocab-logits slice, decoder QKV fusion — QVAC-21623), `include/whisper.h` +4, `src/whisper-logits-slice.h` (new), `ggml/src/ggml-backend-reg.cpp` +91, `ggml/src/ggml-vulkan/ggml-vulkan.cpp` +44, plus CMake glue — ~374 lines total. The v1.9.1 sync commit itself (`cb91a378`) carried ~278 of these forward. Any "never edit `third_party/`" rule is dead on arrival unless these patches get an explicit home.
2. **Canonical ggml is not in this repo.** Production builds consume the `ggml-speech` vcpkg port, built from the `qvac-ext-ggml` repo's `speech` branch. `tts-cpp` already *requires* system ggml in-tree (`TTS_CPP_USE_SYSTEM_GGML=ON`, bundled path rejected at configure time). The in-repo `ggml/` patches therefore only affect standalone dev builds — they duplicate (and can skew from) `qvac-ext-ggml/speech`.
3. **`git subtree add` cannot be applied to the existing tree directly** — the upstream files already exist at root with local modifications, and `git subtree pull` requires prior subtree metadata. The migration needs an explicit conversion step (recipe below).
4. **Both engines already have real, CI-ready test suites**: `parakeet-cpp/test` (19 harnesses: parity, streaming, determinism, perf regression) and `tts-cpp/test` (17+ harnesses), wired into `ctest` with **labels separating GPU and perf tests from the rest** (`ctest -LE 'gpu|perf'` is the hosted-runner lane). Fixtures are small (13 MB + 40 KB); tests whose model/reference files are absent auto-register as `DISABLED`, so a model-free run is still green.
5. Upstream's `.github/workflows` currently run on our repo. Moving the upstream tree under `third_party/` makes them inert automatically (GitHub only reads the root `.github/`) — a benefit, but it means we **must** ship replacement CI in the same change, or we lose the whisper build coverage `build.yml` provides today.
6. Minor: an `upstream` remote *is* configured in working clones; what's missing is scripted/documented sync, not the remote.

## Decision

Adopt **Model B (symmetric layout, upstream as git subtree)** with the amendments below.

### Target layout

```
qvac-ext-lib-whisper.cpp/           (rename deferred — see Out of scope)
  CMakeLists.txt                    # umbrella superbuild, feature-gated; ours
  cmake/                            # shared toolchain/helper modules; ours
  .github/workflows/                # OUR CI only (upstream's becomes inert under third_party/)
  docs/
  third_party/
    whisper.cpp/                    # upstream via git subtree; minimally divergent (see policy)
      PATCHES.md                    #   manifest of every QVAC delta vs the pinned upstream tag
  engines/
    whisper/                        # thin adapter/glue over third_party (see Open decisions)
    parakeet/                       # git mv from parakeet-cpp/
    tts/                            # git mv from tts-cpp/   (chatterbox + supertonic + lavasr)
    qwen-asr/                       # future
```

### Divergence policy for `third_party/whisper.cpp` (amendment)

*Upstream-tracking, minimally divergent* — not "pristine":

- Upstream lands via `git subtree pull --prefix third_party/whisper.cpp upstream <tag> --squash`. Subtree does a real 3-way merge, so conflicts occur **only on the handful of files we touch** (today: `src/whisper.cpp`, `src/CMakeLists.txt`, `include/whisper.h`), instead of today's whole-tree re-apply.
- Every QVAC change inside the prefix must be listed in `third_party/whisper.cpp/PATCHES.md` (file, ticket, rationale) and marked in code with `// QVAC:` comments where practical.
- **A CI guard job diffs the subtree against the pinned upstream tag and fails if any file outside the manifest differs.** This is what actually keeps divergence honest — not a convention in a doc.
- **ggml divergence is banned in this repo.** All ggml patches go to `qvac-ext-ggml` (`speech` branch), which feeds the `ggml-speech` port. The umbrella build forces system ggml (`WHISPER_USE_SYSTEM_GGML=ON` and the engines' equivalents), so the vendored `ggml/` is never compiled in our builds. The existing in-tree `ggml-backend-reg.cpp` / `ggml-vulkan.cpp` deltas must be confirmed present in `qvac-ext-ggml/speech` (or ported there) during migration, then dropped from the subtree.
- Whisper-core patches should be considered for upstreaming when generic; QVAC-specific ones (logits slice, QKV fusion) stay in the manifest.
- Upstream's `parakeet` (C API `parakeet_*`) is **not built** by the umbrella (feature-gated off) to avoid confusion with `engines/parakeet` (C++ `namespace parakeet`). Documented in `PATCHES.md` prose.

### Umbrella build

Root `CMakeLists.txt` (ours, from scratch — the current root one is upstream's with edits):

```cmake
option(SPEECH_BUILD_WHISPER  "Build the whisper engine"  ON)
option(SPEECH_BUILD_PARAKEET "Build the parakeet engine" ON)
option(SPEECH_BUILD_TTS      "Build the TTS engines"     ON)
option(SPEECH_BUILD_TESTS    "Build engine test harnesses" OFF)
```

- One `find_package(ggml CONFIG REQUIRED)` resolved from the `ggml-speech` port; all engines link the same ggml (already validated for whisper + parakeet).
- Each engine remains independently configurable standalone (their own `CMakeLists.txt` keep working) — the umbrella is additive, so per-engine vcpkg ports keep functioning until the umbrella port (Ticket 4) replaces them.

## Migration plan

Ordered to de-risk: CI first (so the moves are protected), then pure renames (reviewable as renames), then the vendor conversion.

**PR 0 — CI scaffold (do now, before any reorg).** See [CI & testing](#ci--testing). Lands against current paths (`parakeet-cpp/`, `tts-cpp/`); the path updates in later PRs are one-line changes.

**PR 1 — engine moves (rename-only).**
- `git mv parakeet-cpp engines/parakeet` and `git mv tts-cpp engines/tts` in a commit containing *no content changes*, so Git records pure renames and `git log --follow` / blame survive. Any path-string fixups (docs, scripts, CI paths) go in a separate follow-up commit in the same PR.
- Coordinate with the registry: the `parakeet-cpp` / `tts-cpp` vcpkg ports reference these paths, so the next pin bump after this PR must carry the port-side path updates (normal 3-PR propagation).

**PR 2 — vendor conversion.** `git subtree add` can't target a prefix whose files already exist, so the conversion is:
1. `git rm -r` the upstream-owned root paths (`src/ include/ ggml/ examples/ bindings/ ci/ models/ samples/ grammars/ tests/ scripts/ Makefile …`).
2. `git subtree add --prefix third_party/whisper.cpp upstream v1.9.1 --squash` — creates proper subtree metadata so future `git subtree pull` works.
3. Re-apply our whisper delta as **one commit** ("QVAC patches on vendored whisper.cpp", cherry-picked from the pre-conversion tree) + commit `PATCHES.md`. One auditable divergence commit; the ggml deltas are *not* re-applied (routed to `qvac-ext-ggml` instead, verified first).
4. Add the new root umbrella `CMakeLists.txt` + prune root `.github/workflows` down to our own.
- Trade-off, stated plainly: blame *inside* the vendored tree points at the squash commit rather than per-line upstream history (acceptable for vendored code — upstream history lives upstream); our delta stays visible as its own commit + manifest.

**PR 3 — sync docs + guard.** `docs/UPSTREAM-SYNC.md` (the subtree pull runbook: remote setup, `git subtree pull … --squash`, expected conflict files, post-pull checklist incl. re-verifying `PATCHES.md`), plus the CI divergence-guard job.

**Freeze window:** PRs 1–2 rewrite most paths; land them back-to-back with open feature branches rebased immediately after (renames make `git rebase` handle it mostly automatically, but coordinate in the team channel).

## CI & testing

### What we can stand up now (PR 0, pre-reorg)

Everything below runs against the engines as they exist today; the reorg only changes path filters.

**`parakeet-ci.yml` / `tts-ci.yml`** — per-PR, path-filtered, no third-party actions beyond `actions/checkout` + `actions/cache`. *Validated end-to-end locally (macOS arm64): both engines build against a CPU-only system ggml and the test lanes pass — tts 68/68, parakeet 1 runnable (rest auto-`DISABLED`, see below).*

| Job | Runner | What it does |
|---|---|---|
| `parakeet build-test` | `ubuntu-24.04`, `macos-14` | Build `qvac-ext-ggml@speech` (CPU-only, shared cache keyed by branch SHA) → configure `PARAKEET_USE_SYSTEM_GGML=ON` + `PARAKEET_BUILD_TESTS=ON` → build → `ctest -LE 'gpu\|perf'` |
| `tts build-test` | same | Same ggml install → `TTS_CPP_BUILD_TESTS=ON` (system ggml already mandatory in-tree) → `ctest -LE 'gpu\|perf'` — 68 tests pass model-free today |
| `gpu` (both files) | `[self-hosted, gpu]` | Stub behind `workflow_dispatch` input until runners exist; runs `ctest -L gpu` |

- **Test filter:** the suites label tests `unit`/`fixture`/`cpu`/`gpu`/`perf`/`mtl-*`; the CI lane runs `-LE 'gpu|perf'` (GPU absent on hosted runners, perf gates meaningless on shared ones).
- **Path filters:** `parakeet-cpp/**` and `tts-cpp/**` respectively (post-reorg: `engines/parakeet/**`, `engines/tts/**`; `third_party/**` → all).
- **ggml provisioning:** both repos are public, so CI builds `qvac-ext-ggml@speech` directly (shallow clone, `actions/cache` on the install dir keyed by the branch tip SHA) — no vcpkg/registry auth in the loop. This *is* the production ggml source (`ggml-speech` port ships the same branch). The vcpkg overlay-pin variant moves to the port-smoke follow-up.
- **Models/fixtures:** committed fixtures are tiny (13 MB parakeet, 40 KB tts) and drive the lane above. The parity suites need converted GGUFs — `download-all-models.sh` pulls **~14.5 GiB of `.nemo` checkpoints requiring Python conversion**, which is not CI-viable. Both harnesses already accept pre-staged fixture roots (`PARAKEET_TEST_{MODEL,REF}_DIR` cache paths) and auto-`DISABLE` (→ "Not Run") when files are absent, so the enablement path is clear: host the small q8_0 GGUFs + `.npy` reference dumps on HF/S3 and restore them via `actions/cache` (follow-up, not PR 0).

**`port-smoke.yml`** (follow-up to PR 0 — needs the registry portfiles as input) — packages repo HEAD as a vcpkg source tarball (codeload-style) into overlay ports for `parakeet-cpp`/`tts-cpp` and builds a minimal consumer. Catches "the port will break" *before* the registry PR, removing the most common failure of the 3-PR chain.

**Whisper build job** — deferred to PR 2: upstream's (adapted) `build.yml` still runs at root today, so adding another whisper build now would duplicate it. PR 2 replaces it with a targeted build + `whisper-cli` smoke job when the upstream workflows go inert.

**GPU lane (deferred, documented):** `ctest -L gpu` (`test_vk_vs_cpu`, `test_metal_ops`, Adreno/Mali suites) needs self-hosted runners with real GPUs. Scaffold the job behind a `workflow_dispatch` + `runs-on: [self-hosted, gpu]` label now, leave it manual until runners exist. Hosted `macos-14` runners do expose Metal, so `ctest -L gpu` on the macOS jobs can be trialled cheaply first.

### What the reorg adds (PR 2/3)

- **Umbrella job:** configure the root superbuild with all engines ON — proves the "one ggml, all engines" invariant on every PR that touches shared surface.
- **Divergence guard:** checks out the pinned upstream tag, diffs against `third_party/whisper.cpp`, fails if files outside `PATCHES.md` differ. Runs only when `third_party/**` changes.
- **Upstream CI noise gone:** upstream's workflows move under `third_party/` and stop executing; root `.github/workflows` contains only ours + org baselines.

### What stays downstream

`tetherto/qvac` e2e (`on-pr-tts-ggml`) remains the integration gate — this QIP doesn't replace it, it front-loads the failures that today only surface there.

## Alternatives considered

1. **Keep separate folders/repos + per-engine ports** — perpetuates version skew (whisper@one commit vs parakeet@another) and the asymmetry this QIP exists to remove.
2. **`git submodule` for upstream** — incompatible with vcpkg (`vcpkg_from_github` uses source tarballs, which exclude submodule content). Subtree content is baked into the tarball.
3. **Plain `git merge -X subtree=…` of upstream into the prefix** — works mechanically but leaves no subtree metadata and makes the pin implicit; `git subtree --squash` gives an explicit, greppable pin commit per sync.
4. **Truly pristine subtree + patch files applied at build time** (`engines/whisper/patches/*.patch`) — keeps the vendored tree byte-identical, but wrecks day-to-day DX (IDE navigation, debugging, blame all see un-patched sources), complicates the vcpkg port, and turns every upstream sync into patch-fuzz whack-a-mole. Rejected in favor of the minimally-divergent policy + CI guard.
5. **Pristine subtree + all our logic in an adapter layer** — impossible for the existing patches: QKV fusion and logits-slice modify whisper's internal graph construction; there is no seam an adapter could hook.

## Consequences

1. Symmetric engine layout; upstream pulls become 3-way merges touching ~3 known files instead of whole-tree re-applies; one in-repo upstream pin shared by all engines.
2. Engine regressions surface on the PR in minutes (`ctest -L cpu`) instead of days later in `qvac` e2e.
3. Divergence from upstream is explicit (manifest + guard) instead of implicit (scattered through a root tree that looks upstream-owned).
4. ggml patching gets a single home (`qvac-ext-ggml/speech`), removing today's silent two-copy skew.
5. Costs: blame inside the vendored tree coarsens to the squash/sync commits; a one-time freeze window for the moves; registry ports need a coordinated path bump; the team must learn one rule — *"don't touch `third_party/` outside a sync or manifest PR"* — with CI enforcing it.

## Out of scope

1. Repo rename to `qvac-fabric-speech.cpp` (tracked separately).
2. The umbrella vcpkg port (Ticket 4) — enabled by this reorg, delivered separately.
3. `transcription-ggml` NPM package + SDK changes (Ticket 2).
4. GPU self-hosted runner procurement (scaffolded, not delivered).

## Open decisions

1. **`engines/whisper/` adapter vs consuming `third_party/` directly.** Recommendation: create `engines/whisper/` now but keep it *thin* — the umbrella-facing CMake target, QVAC-side glue/headers, and the natural future home for whisper-adjacent code that would otherwise grow the `PATCHES.md`. Core patches stay in the subtree.
2. **`engines/qwen-asr/` scaffolding now vs at implementation time.** Recommendation: a README stub only; empty scaffolds rot.

## Approvals

| Reviewer | Role | Status |
|---|---|---|
| _TBD_ | Speech team lead | ☐ |
| _TBD_ | Infra/registry owner | ☐ |
| _TBD_ | Downstream (qvac) owner | ☐ |
