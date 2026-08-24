# AceSTEP on AMD Strix Halo — optimization campaign ledger

Goal: 2x faster end-to-end generation vs baseline on the Strix Halo Vulkan backend
(turbo-Q4_K_M and turbo-Q8_0, 30 s, warm), quality >= 98% (bit-correct preferred),
no regressions on other backends/models. sft-Q8_0 tracked at milestones.

Machine: AMD Ryzen AI MAX+ 395, Radeon 8060S iGPU (RADV GFX1151, RDNA3.5, 40 CU),
Mesa RADV 25.2.8, 2 GiB VRAM carve + 112 GiB GTT (UMA), 121 GB RAM, 32 threads.
Runtime device line: `uma: 1 | fp16: 1 | bf16: 0 | warp size: 64 | int dot: 0 |
matrix cores: KHR_coopmat`.

Method: every iteration = hypothesis -> cheapest verification first -> change ->
gate -> regression net -> verdict. Verdict has two axes (observation reproduced /
mechanism confirmed); only mechanism-confirmed counts as CORRECT. Three WRONG in a
row => step back + independent review.

Benchmark protocol: `~/qvac-campaigns/QVAC-24013/bin/bench.sh` — pinned prompts P1
(pop, clean English lyrics) / P2 (rock), seeds 42/1337, 30 s, 1 discarded cold run +
3 warm runs, median warm wall = the metric. Results: benchmarks/results.jsonl.

## Status

Phase 0 (infrastructure) complete; Phase 1 baseline matrix running.

Phase 0 verified facts:
- Builds: ggml@speech Vulkan (ninja+ccache, install ~/ggml-install/vk) and audiogen
  (build/audiogen) green; all 5 unit tests pass (LD_LIBRARY_PATH must include the
  ggml install lib dir).
- Regression reference: per-op test-backend-ops sweep on Vulkan0 — 137/137 ops
  covered, all pass except pre-existing IM2COL_3D crash
  (GGML_ASSERT ggml-vulkan.cpp:7123 workgroup-count overflow; not in AceSTEP graphs;
  candidate generic fix). Full-sweep-single-process log truncates at that crash, so
  per-op sweeps (campaign bin/sweep_ops.sh) are the reference method.
- Coopmat ground truth (probe): 16x16x16 f16xf16 with f16 AND f32 accumulators
  (FA_COOPMAT1 with GGML_PREC_F32 is possible); full int8 coopmat set present but
  unused by current shaders. No bf16 coopmat. VK_EXT_external_memory_host present
  (H3c viable). System glslc lacks GL_EXT_integer_dot_product -> int dot: 0 (H11).
- Smoke run observation (turbo-q4, requested --dur 5, seed 42, GPU): LM 8460 ms
  (56%), VAE 4687 ms (31%), DiT 1014 ms (7%); total 15.3 s. LM runs on CPU by
  default policy. NOTE: --dur forces duration metadata but the LM generated codes
  to the max_tokens cap (dur*5+100 = 125 codes -> 25.2 s actual audio). Benchmarks
  record actual WAV seconds; RTF computed against actual.

## Baseline (Phase 1) — TBD

| Config | Median warm wall (s) | RTF | Notes |
|---|---|---|---|
| turbo-q4 / P1 / s42 | | | |
| turbo-q4 / P2 / s42 | | | |
| turbo-q8 / P1 / s42 | | | |
| turbo-q8 / P2 / s42 | | | |
| sft-q8 / P1 / s42 | | | tracking only |

## Hypothesis ledger

(entries appended below; format: Hn — status — hypothesis — why — test — changes —
outcome — verdict (observation / mechanism) — learning — next)

### Pre-registered backlog (from plan, ordered; re-rank after Phase 2 data)

- H4 LM placement: LM-on-Vulkan may beat LM-on-CPU on Strix (policy is a Mali
  carry-over). Pre-verify: ACESTEP_LM_GPU=1 A/B.
- H1 DiT graph+gallocr reuse across Euler steps. Pre-verify: perf-logger gap
  (stage wall - GPU op time).
- H2 Euler+DCW in-graph, latent resident on device. Pre-verify: transfer ops in
  perf-logger.
- H3 weight residency: (a) KEEP_STAGES policy, (b) UMA prefer_host_memory A/B
  (rebuild), (c) zero-copy weights via VK_EXT_external_memory_host.
- H5 LM per-token gallocr churn.
- H7 RDNA3 subgroup pinning entry + RDNA3.5 shmem occupancy tuning.
- H8 FA path: confirm f32acc coopmat / FA actually running; tune or re-precision.
- H6 shared GPU backend instance (engine + VAE).
- H10 VAE kernel/fusion work (only if VAE share stays large).
- H11 integer-dot shaders: system glslc lacks GL_EXT_integer_dot_product so `int
  dot: 0` at runtime; newer glslc may unlock int8-dot paths for q8/q4 on RDNA3.
