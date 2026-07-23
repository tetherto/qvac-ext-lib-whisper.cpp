# QIP: Unify the ASR NPM packages into `@qvac/transcription-ggml`

| | |
|---|---|
| **Status** | Draft — for approval |
| **Owner** | Omar Gad |
| **Date** | 2026-07-23 |
| **Target repo** | `tetherto/qvac` (`packages/`) — this doc lives here alongside the sibling repo-reorg QIP |
| **Related** | Ticket 1 (repo reorg, [QIP #94](https://github.com/tetherto/qvac-ext-lib-whisper.cpp/pull/94) — PRs 0–3 landed), Ticket 4 (umbrella vcpkg port — **not started**), Ticket A (SDK repoint — separate) |

## Summary

Replace `@qvac/transcription-whispercpp` (0.12.1) and `@qvac/transcription-parakeet` (0.10.1) with one multi-engine `@qvac/transcription-ggml` package, following the proven `@qvac/tts-ggml` shape: one public class with static engine constants, one Bare addon dispatching on `engineType`, one CI/benchmark family instead of two. Migration preserves git history (`git mv`), keeps the SDK green throughout (addon-only merge — the SDK's two plugins and two `modelType`s survive unchanged until the separate repoint ticket), and **does not block on the umbrella vcpkg port**: the addon consumes today's `whisper-cpp` + `parakeet-cpp` ports side-by-side (they already share one `ggml-speech`), and swaps to `speech-cpp[whisper,parakeet]` later as an invisible dependency change.

Three material amendments to the original plan, from auditing the packages as they exist today (2026-07-23, `qvac@6ae29cb7`):

1. **`tts-ggml` is not the driver-table design the plan implies** — it is one monolithic orchestrator class with inline `engine ===` branches and a *flat, non-discriminated* options object. We adopt its **native** pattern verbatim (`createInstance` dispatch + `dynamic_cast` in `runJob`) but deliberately improve on its **JS** pattern with small internal per-engine drivers, because the two ASR surfaces diverge far more than chatterbox/supertonic do.
2. **Do not invent a third config vocabulary.** The public config is discriminated by `engine` at the top level, but each branch keeps its engine's existing, allow-listed field set (`whisperConfig` snake_case incl. `configChecker` rules; `parakeetConfig` camelCase incl. the `streaming*`/`spkCache*` surface) verbatim. Normalization applies to the *method surface, streaming events, and audio input* — not to per-engine tuning knobs. This is what makes the merge genuinely addon-only for the SDK.
3. **The CI/benchmark port is the largest single work item** — ~20 workflow files retired, ~10 created, 7 central registries edited, two Python WER/CER harnesses and two RTF aggregators merged, plus Device Farm secrets provisioning. It is planned explicitly below, not left as an afterthought.

## Problem

Two ASR packages duplicate the same stack — Bare addon on `qvac-lib-inference-addon-cpp` (vcpkg) ≥1.2.4, `cmake-bare`/`cmake-vcpkg`, TypeScript two-layer JS (orchestrator + native interface), brittle tests, Python WER/CER benchmark harness, and a 10-workflow CI family each — while diverging in API shape:

- Constructor: `new TranscriptionParakeet({ files, config, logger, exclusiveRun })` vs whisper's two-positional-arg `new TranscriptionWhispercpp({ files, logger, exclusiveRun, opts }, config)`.
- Config: `config.parakeetConfig` camelCase vs `config.whisperConfig` snake_case, with a fragile JS↔native key-translation layer (`NON_ADDON_WHISPER_KEYS`, `max_seconds`→`duration_ms`) on the whisper side.
- Streaming: whisper emits typed `{type:'vad'}` / `{type:'endOfTurn'}` events (Silero VAD, `files.vadModel`); parakeet has no typed events — EOU is a boolean `isEndOfTurn` on segments, VAD an `emitEnergyVad` flag.
- Duplicated-and-skewing: the `BackendId` enum, the `s16le`→f32 audio conversion, mock-binding test scaffolding, mobile test generators, benchmark servers/clients, and the entire workflow family.

The SDK dispatches by `modelType` through a plugin registry; package names appear only in two plugin files at load time. `@qvac/tts-ggml` already proves the one-package/multi-engine model (and the SDK already absorbed one package consolidation, `onnx-tts` → `tts-ggml`, via alias mapping in `engine-addon-map.ts`).

## Audit findings

Facts verified against `tetherto/qvac@6ae29cb7` that correct the original draft:

1. **Versions and language moved on**: whispercpp is 0.12.1 (not 0.12.0), parakeet 0.10.1, tts-ggml 0.5.1. All three are **TypeScript** (`src/*.ts` → committed generated JS + `.d.ts`, enforced by `check:generated`) since [qvac#3336](https://github.com/tetherto/qvac/pull/3336). The merged package is TS from day one — "two-layer JS" is now "two-layer TS".
2. **`tts-ggml`'s real anatomy**: one ~1600-line `TTSGgml` class, flat `TTSGgmlOptions` (both engines' fields co-mingled), engine auto-detect from `files`, dispatch via `_buildChatterboxParams`/`_buildSupertonicParams` ternaries. Native side: `JSAdapter::readEngineType()` → `make_unique<{Chatterbox,Supertonic}Model>` in `createInstance`; `dynamic_cast` in `runJob`/`reload`; both models implement `IModel`/`IModelCancel`/`IModelAsyncLoad`. Its exports map is **not** a single `"."` — it carries utility subpaths (`./text-chunker`, `./addonLogging`, `./binding.js`). The rule worth copying is *no per-engine subpaths*, not *no subpaths*.
3. **The C++ layers are already near-twins** — merge-friendly: same framework interfaces, same binding verbs (`createInstance/runJob/startStreaming/appendStreamingAudio/endStreaming/cancel/destroyInstance/setLogger`), same worker-thread StreamingProcessor design with a global session map keyed by `AddonJs*`, same dynamic ggml-backend staging (`libqvac-speech-ggml-*.so` → `prebuilds/<subdir>`, `BACKENDS_SUBDIR`). Deltas: parakeet exports `getBackendInfo` (whisper folds backend info into `RuntimeStats`); whisper has a **native** `reload` (context rebuilt only when context-affecting params change) while parakeet reloads by destroy+recreate in JS — the draft's `supportsReload` driver hook is grounded.
4. **The JS layers are where the divergence lives** (constructor shape, config vocabularies, event models, audio formats above) — which is exactly why the unified orchestrator needs per-engine drivers rather than tts-ggml's inline branches.
5. **SDK reality check**: dispatch is by `modelType` (registry `Map`), and only `packages/sdk/server/bare/plugins/{whispercpp,parakeet}-transcription/plugin.ts` import the packages. But the SDK's zod discrimination for transcription is keyed by **`modelType`** (two separate schema branches in `load-model.ts`), *not* by an `engine` field — the engine-keyed discriminated union is the TTS pattern. The original claim "matching the SDK's zod discrimination" holds only if the SDK later consolidates to one modelType (Ticket A option, not required). Note also the whisper plugin already special-cases `endOfTurn.source === 'parakeet'` — a `source` field on end-of-turn events is precedented in SDK code.
6. **Ticket 4 has not started**: the registry has no umbrella `speech-cpp` port — only `whisper-cpp` (1.9.1#4), `parakeet-cpp` (date-pinned), `tts-cpp`, `ggml-speech`. Both ASR ports depend on the same `ggml-speech` (≥2026-07-13/-21), and the repo-reorg umbrella build has already validated whisper+parakeet linking one ggml. Serializing this package behind Ticket 4 is unnecessary.
7. **Beyond the SDK, two more consumers/reference points**: `@qvac/tts-ggml` devDepends on `@qvac/transcription-whispercpp` (its WER/CER quality benchmark runs Whisper over synthesized audio, QVAC-22636), and `on-pr-bare-sdk-e2e.yml` npm-installs the whisper package. Both need repointing at retirement.
8. **Naming hazard that the new name fixes**: `bci-whispercpp` is a *separate, surviving* package whose workflow files match the `*whispercpp*.yml` path globs used by the whisper family today. `transcription-ggml` has no substring collision with anything.
9. **In-flight PRs against the old packages** ([#3314](https://github.com/tetherto/qvac/pull/3314) Core ML backend reporting, [#3228](https://github.com/tetherto/qvac/pull/3228) RNN-T support, [#3126](https://github.com/tetherto/qvac/pull/3126)/[#3130](https://github.com/tetherto/qvac/pull/3130) lint chores) — the migration needs a short freeze/rebase window coordinated with their authors.
10. **Cleanup to take with the move, not replicate**: whisper's `addon/CMakeLists.txt` is dead (references nonexistent sources; sole consumer of the unused `mel_filters.bin`), and its `package.json` exports a `./transcription-addon` path whose directory doesn't exist in-tree. Neither survives the merge.

## Decision

Create `@qvac/transcription-ggml` in `tetherto/qvac/packages/`, mirroring `tts-ggml` where it is strong (native dispatch, one addon, one CI family, exports discipline) and deliberately improving where ASR needs more (internal JS drivers, engine-discriminated config, normalized streaming events).

### Public API

```ts
class TranscriptionGgml {
  static readonly ENGINE_WHISPER = 'whisper'
  static readonly ENGINE_PARAKEET = 'parakeet'
  // future: ENGINE_QWEN_ASR = 'qwen-asr'

  constructor(options: {
    engine?: 'whisper' | 'parakeet'   // else auto-detected from files (GGUF metadata / known filenames — parakeet already self-detects CTC/TDT/EOU/Sortformer)
    files: { model: string, vadModel?: string /* whisper streaming */ }
    config?: WhisperEngineConfig | ParakeetEngineConfig   // discriminated by `engine`; per-engine vocabularies preserved verbatim
    logger?: LoggerInterface
    exclusiveRun?: boolean            // default true, one shared queue implementation
  })

  load(); unload(); destroy(); reload(newConfig?); cancel(jobId?)
  getState(); status(); getEngineType(); getBackendInfo()   // getBackendInfo implemented for both (whisper's data already exists in RuntimeStats)
  pause(); unpause()                                        // per-engine support documented; unsupported → structured error (as today)
  run(audio): Promise<QvacResponse<TranscriptionSegment[]>>
  runStreaming(audio, opts?): Promise<QvacResponse<StreamEvent>>
}
```

- **Exports**: `"."` + utility subpaths only (`./addonLogging`, `./binding.js`, `./package`) — the SDK imports `/addonLogging` today. The per-engine subpaths (`./parakeet`, `./transcription-addon`) are dropped.
- **Config**: top-level discriminant `engine`; each branch carries the engine's existing allow-listed fields unchanged (whisper's `configChecker` moves into the whisper driver's `validateConfig`). No third vocabulary, no silent key remapping beyond what each package already does internally.
- **Audio**: accepted input set is the union both engines already handle — `s16le` (default) and `f32le`, as typed arrays, chunk arrays, or (async) iterables, 16 kHz mono. Conversion is per-driver (`normalizeAudio`); whisper's `audio_format: 'decoded'` stays available inside its config branch.
- **Streaming events** — normalized union with a `source` field: `segment` (common core `{text, start, end, id}` + engine extras like `startsWord`), `vad` (`{speaking, probability, source: 'silero'|'energy'}`), `endOfTurn` (`{silenceDurationMs?, source: 'vad-silence'|'model-eou'}`). The whisper driver passes its typed events through; the parakeet driver synthesizes `endOfTurn` from `isEndOfTurn` segments. One `BackendId` enum, one `RuntimeStats` envelope with engine-specific timing extras.

### Internal structure

- **JS drivers** (internal, not exported): `src/engines/{whisper,parakeet}/` each implementing `{ buildNativeConfig, validateConfig, normalizeAudio, buildStreamingOpts, mapSegment, mapStreamEvent, supportsReload }`. The orchestrator holds a driver table — no `if engine ===` chains. This is a stated departure from tts-ggml's inline branches, justified by the ASR surface divergence (finding 4); if tts-ggml later wants the same structure, that's its own refactor.
- **Native addon** — the tts-ggml pattern verbatim: one `BARE_MODULE`, `JSAdapter::readEngineType()` → `make_unique<{Whisper,Parakeet}Model>` in `createInstance`; `dynamic_cast` dispatch in `runJob`/`reload`; binding surface is the union of today's verbs (`reload` native for whisper, JS-recreate for parakeet behind `supportsReload`; `getBackendInfo` for both). Both models already implement `IModel`/`IModelCancel`/`IModelAsyncLoad`, so they slot in unmodified apart from namespace unification. The two StreamingProcessors keep their engine-specific internals behind one session registry. `BACKENDS_SUBDIR` must be defined on **every** target that compiles backend-path code (a latent bug of this kind exists in tts-ggml's CMake), and no direct ggml-cpu symbol calls (GGML_BACKEND_DL, the June Android regression).

### Dependency strategy (de-serializing Ticket 4)

- **Phase A (this QIP)**: `vcpkg.json` depends on **both** `whisper-cpp` and `parakeet-cpp` with today's per-platform backend features. Both resolve the same `ggml-speech`, and whisper+parakeet coexisting on one ggml is already proven by the repo-reorg umbrella build. Risk to verify in PR 1/2 CI: duplicate-symbol or CMake-config clashes when both ports link into one Bare module (none expected — distinct libs, shared ggml).
- **Phase B (when Ticket 4 lands)**: swap to `speech-cpp[whisper,parakeet]` + platform backend features in one dependency-only PR. Invisible to every consumer; also the natural point to drop the two per-engine ports.

## Migration plan

All PRs in `tetherto/qvac`. Ordered so the tree, npm, and the SDK stay green at every step; per the packages rule, version bumps ride in their own PRs.

**PR 1 — whisper becomes the unified package (history-preserving).**
`git mv packages/transcription-whispercpp packages/transcription-ggml` as a pure-rename commit, then reshape commits in the same PR: package rename to `@qvac/transcription-ggml`, `src/index.ts` → orchestrator + `src/engines/whisper/` driver, drop dead `addon/CMakeLists.txt` + `mel_filters.bin` + `./transcription-addon` export, native namespace unification. Ship the full CI family for `transcription-ggml` (see table below) **in the same PR** (with `pull_request_target`, on-pr workflows execute from the base branch, so the family must be on `main` before package PRs get CI — PR 1 itself is validated via `workflow_dispatch`/label runs), retire the whisper family, and edit the central registries (add `transcription-ggml`, drop `transcription-whispercpp`). The last published `@qvac/transcription-whispercpp@0.12.x` keeps the SDK green.

**PR 2 — parakeet merges in (history-preserving).**
`git mv` parakeet's substantive code into place: `addon/src/model-interface/{ParakeetStreamingProcessor,ParakeetTypes,parakeet/}`, `src/parakeet.ts` (as the driver's native interface), tests, examples, benchmark server/client configs, model manifests and mobile generators. Add the parakeet driver + native `createInstance`/`runJob` branches + the dual-port `vcpkg.json`. Delete `packages/transcription-parakeet`, retire its CI family, prune central registries. Merge the two Python WER/CER harnesses and the two RTF aggregators (below).

**PR 3 — SDK repoint (Ticket A, tracked separately but sequenced here).**
The two plugin files import `TranscriptionGgml` + `ENGINE_*` constants; update `ADDON_*`/`PLUGIN_*` constants, `engine-addon-map.ts` legacy aliases (`@qvac/transcription-whispercpp` / `@qvac/transcription-parakeet` → canonical, the `onnx-tts` precedent), `bare-sdk` `plugin-addons.mjs`, `sdk/package.json` deps + `ensureprebuilds` script, tts-ggml's benchmark devDep, `on-pr-bare-sdk-e2e.yml` install line. **Keep both `modelType`s and both plugins** — public SDK API (`transcribe`/`transcribeStream`/`loadModel({modelType})`) unchanged; consolidating to one modelType with an `engine` discriminator (full TTS parity) is an optional later step behind the same alias machinery.

**Freeze window**: between PR 1 and PR 3, changes to the old packages have no home — coordinate with in-flight PRs #3314/#3228/#3126/#3130 (rebase onto `transcription-ggml` paths or land before PR 1).

**Validation gate per PR**: the standard downstream flow — label-gated prebuilds + desktop + mobile lanes on the PR itself (the family ships in PR 1), plus `coload-smoke-ggml` since the addon list changes.

## Workflows & benchmarks porting

Inventory from auditing `.github/workflows/` (the package families are strictly parallel; tts-ggml is the copy template):

**Create (10 files + 1 script):** `on-pr-`, `on-merge-`, `on-pr-close-`, `prebuilds-` (one artifact prefix, `transcription-ggml-`), `integration-test-` (RTF matrix inputs), `integration-mobile-test-` (engine axis in the matrix, like tts-ggml's chatterbox/supertonic axes), `cpp-test-coverage-`, `benchmark-transcription-ggml.yml` (accuracy), `benchmark-performance-transcription-ggml.yml` (RTF orchestrator), optional `repository-dispatch-`; plus `scripts/perf-report/aggregate-transcription-ggml-rtf.js` replacing `aggregate-whisper-rtf.js` + `aggregate-parakeet-rtf.js`.

**Edit (central registries):** `pr-gate-merge.yml` paths-filter, `prebuilds-caller.yml` job, `packages/ggml-coload-smoke/addons.js` (canonical addon inventory, `stack: 'speech'` — documented as sync-coupled with the SDK `ADDON_*` constants), `gate-selftest.yml` hardcoded addon string, `coload-smoke-ggml.yml` + `coload-smoke-mobile-ggml.yml` trigger paths, `perf-report.yml` (weekly cron currently covers parakeet only — point it at the merged package; whisper mobile perf becomes cron-covered for free).

**Retire (20 files):** both old families. **Do not touch** the addon-agnostic core (`ci-router`, `label-gate`, `reusable-prebuilds.yml`, `run-mobile-integration-tests`) — it needs no per-addon data. **Leave `bci-whispercpp` alone** — every `*whispercpp*` match must be checked against it before deletion.

**Benchmarks specifically:**
- *Accuracy (ASR-only track)*: merge the two `benchmarks/{server,client}` Python harnesses into one, keeping both matrices — whisper's datasets (incl. Arabic Common Voice from S3, `AraDiaWER`) and HF→GGML conversion, parakeet's model-type axis (tdt/ctc/eou/sortformer × librispeech/fleurs, GPU/streaming combos) — behind an `engine` matrix dimension in one `benchmark-transcription-ggml.yml`.
- *RTF/perf*: one desktop `benchmark_matrix_json` and one `run-rtf-benchmark-matrix.js` (the two are near-duplicates); unified report artifact replaces `whisper-unified-performance-report`/`parakeet-unified-performance-report`; `benchmarks/manual-results/` baselines migrate so `--manual-dir` aggregation keeps working.
- *Cross-package*: tts-ggml's quality benchmark (runs Whisper on synthesized audio) repoints its devDep to `@qvac/transcription-ggml` in PR 3.

**Infra (outside the repo):** Device Farm needs a project/pool ARN secret pair for the merged mobile lane. Recommendation: repoint the parakeet project's secrets (whisper's project is shared with BCI and must survive independently); alternatively provision `AWS_DEVICE_FARM_PROJECT_ARN_TRANSCRIPTION_GGML` fresh. Needs the infra owner before PR 1's mobile lane can run.

## Alternatives considered

1. **Keep two packages** — continued duplication and skew; every finding in the Problem section persists, and qwen-asr would become a third copy.
2. **Per-engine subpath exports** (`transcription-ggml/whisper`) — larger public surface, diverges from tts-ggml, and defeats the point (consumers coupling to engines by import path instead of by config).
3. **Fully normalized third config vocabulary** (the original draft's reading) — maximal API elegance, but every field remap is a silent-breakage risk (whisper's key-translation layer is already the fragile spot), it forces nontrivial SDK plugin translation (no longer addon-only), and it throws away whisper's battle-tested `configChecker` allow-lists. Rejected in favor of engine-discriminated config with preserved vocabularies; a normalized *common* layer can grow additively later.
4. **Block on `speech-cpp` (Ticket 4) as the dependency** — serializes this QIP behind an unstarted port for zero consumer-visible benefit; the two-port Phase A is equivalent at the ABI level (same `ggml-speech`) and the swap is one manifest PR later.
5. **Mirror tts-ggml's monolithic orchestrator exactly** — simpler to review against the template, but the ASR engines' JS surfaces are far more divergent than the TTS engines'; the inline-branch pattern is already strained at ~1600 lines with two *similar* engines. Drivers keep qwen-asr additive.
6. **New package assembled from scratch (no `git mv`)** — loses blame/`git log --follow` across the migration for two heavily-maintained packages. Rejected; rename-only commits first, reshape commits second, per the repo-reorg QIP's own convention.

## Consequences

1. One ASR package, one addon, one CI family (−20/+10 workflow files), one benchmark harness, one Device Farm lane, one `BackendId`/stats vocabulary — full parity with `tts-ggml`, and a clean additive home for qwen-asr (`ENGINE_QWEN_ASR` + one driver + one native model class).
2. SDK breakage near zero by construction: two plugins keep their `modelType`s and per-engine zod schemas; the repoint (PR 3) swaps imports and constants only. The `onnx-tts` → `tts-ggml` alias precedent covers stragglers referencing old package names.
3. Apps loading both engines ship one prebuild instead of two (shared framework/ggml glue deduplicated); apps loading one engine carry the other's engine lib in the addon binary — measured at PR 2, expected small relative to models, and Phase B's feature-gated umbrella port is the lever if it isn't.
4. Costs: one-time merge effort concentrated in the JS orchestrator + event normalization; a freeze/rebase window for four in-flight PRs; Device Farm secret provisioning; reviewers must hold the line on "no per-engine subpaths, no third config vocabulary" as engines accrete.

## Out of scope

1. The umbrella vcpkg port itself (Ticket 4) — only the Phase B swap point is defined here.
2. The SDK repoint lands as its own ticket/PR (Ticket A; sequenced above as PR 3 so this QIP owns the ordering, not the implementation).
3. SDK modelType consolidation to a single `transcription-ggml` type (optional follow-up behind alias machinery).
4. Version bumps (own PRs per `qvac/packages` rule). Retirement/deprecation of the old npm package names on GPR/npm follows org policy after PR 3.
5. qwen-asr engine implementation.

## Open decisions

1. **Device Farm secrets**: repoint parakeet's project vs provision a fresh `TRANSCRIPTION_GGML` pair (recommendation: repoint parakeet's — whisper's is entangled with BCI).
2. **Initial version**: recommendation `0.1.0` (new name, no continuity constraint; tts-ggml precedent).
3. **PR 1 API scope**: land the unified constructor/API with whisper-only support immediately (recommendation — keeps PR 2 a pure engine addition), vs keeping the old whisper API until PR 2.
4. **`pause`/`unpause` in the public surface**: keep with per-engine structured "not supported" errors (recommendation, status quo), vs dropping until both engines support them.

## Approvals

| Reviewer | Role | Status |
|---|---|---|
| _TBD_ | Speech team lead | ☐ |
| _TBD_ | SDK owner | ☐ |
| _TBD_ | Infra/CI owner (Device Farm, registries) | ☐ |
