---
id: "gp-structural-dedup"
title: "One owner per behaviour: shape_product, training policy out of ojas-cpu, head-dim constant, Metal/wgpu conformance suite, retire the tiny Metal step, one GDN reference, shared test support"
status: backlog
priority: 2
severity: medium
type: refactor
owner: "unassigned"
due: "none"
labels:
  - "refactor"
  - "cleanup"
  - "tests"
  - "architecture"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/limits.rs"
  - "ojas-cpu/src/validate.rs"
  - "ojas-cpu/src/train.rs"
  - "ojas-cpu/src/schedule.rs"
  - "ojas-autograd/src/tape.rs"
  - "ojas-autograd/Cargo.toml"
  - "ojas-model/src/trainer.rs"
  - "ojas-model/src/spec.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-metal/src/gpu.rs"
  - "ojas-metal/build.rs"
  - "ojas-core/src/backend.rs"
  - "ojas-kernels/src/geometry.rs"
  - "ojas-metal/tests/"
  - "ojas-wgpu/tests/"
  - "ojas-cuda/src/bf16.rs"
  - "ojas-cuda/src/json.rs"
  - "ojas-simd/src/"
acceptance_criteria:
  - "One shape_product (ojas-core/src/limits.rs) with one zero-axis rule; the copies in ojas-cpu/src/validate.rs, ojas-cpu/src/train.rs and ojas-autograd/src/tape.rs delegate or are deleted; a test pins the chosen zero-axis behaviour at every former call site"
  - "LrSchedule, optim_group, MUON_* and ADAM_HYBRID_WEIGHT_DECAY move to a device-neutral home; Trainer<B> builds step configs through the same function HybridOptimizer uses (no second copy at trainer.rs:777); ojas-model no longer needs ojas-cpu for a Metal-only build"
  - "HybridOptimizer/GradAccumulator/mean_micrograds are either used by the trainer or moved to ojas-autograd's tiny path, and ojas-autograd's normal ojas-cpu dependency is justified or dropped"
  - "One attention head-dim limit: Metal's private ATTN_MAX_HEAD_DIM, ojas_core::METAL_MAX_HEAD_DIM and ojas_kernels::ATTENTION_MAX_HEAD_DIM are one constant, or a compile-time/test assertion keeps them equal"
  - "A generic conformance suite over B: Backend replaces the paired Metal/wgpu test files (shape_first, kv_cache, permute, linear_ce, attention_window, gate_saved, accumulate_grad, cast_bf16); count asymmetries (linear_ce 6 vs 11, kv_cache 10 vs 15) are resolved so both backends run every case"
  - "The legacy tiny Metal step (ojas-metal/src/gpu.rs, TinyShape, causal_attn_bwd.metal) is retired after a two-signal caller check, keeping per_head_gate.metal and gate_dbias_threads that device.rs uses; README mermaid and op-coverage.md:116 updated"
  - "ojas-cuda/src/bf16.rs delegates to ojas_core::f32_to_bf16/bf16_to_f32; ojas-cuda/src/json.rs and ojas-model/src/json.rs share one writer or the split is justified"
  - "Exports with no caller outside their crate (ojas-simd sgemm_tile_with, vdsp_vmul, vdsp_vadd, vdsp_vmul_append, vdsp_mmov, backend_name; Tensor::element_byte_offset, write_ne_bytes) are confirmed with two independent signals and removed or made private; the orphaned doc comment at tape.rs:1335-1337 is fixed"
  - "ojas-model validate_for_training (now a pass-through, spec.rs:213-215) is inlined or given a real check"
  - "Report lines removed alongside lines added"
  - "ojas-device's GemmBlocks and CacheBudget (tuning.rs, plan.rs; listed in README:349) are either consumed by ojas-cpu's GEMM blocking, which today uses the fixed blocking(exec.numerics) (ojas-cpu/src/gemm.rs:385, :599), or removed after a two-signal check that their only caller outside ojas-device is examples/profile.rs:56; docs/adaptive-resources.md:160 already claims no speedup from them"
  - "ojas owns one f64 Gated DeltaNet reference: ojas-oracle/src/gdn.rs and ojas-cuda/tests/reference/gdn_published.rs (DevMap clones: identical Inputs struct, plus the oracle tests' builders) converge, or the reason for two is written in both headers; the tessl fixture copy (ojas-cuda/tests/fixtures/tessl/gdn_train_published.rs) stays a deliberate unmodified cross-check"
  - "Test helpers duplicated outside ojas-autograd (gp-ci-and-repo-hygiene covers that crate only) move to one shared test-support module per concern: ojas-model/tests/common/mod.rs repeats the linear/mul/residual/cross-entropy backward references of ojas-autograd/tests/common and device_tape.rs; the counting realloc allocator is copied in four ojas-cpu tests (attention_fast, framework_ce_heap, redteam_linear_heap, redteam_ops_heap); ref_forward/ref_backward and assert_bits are exact copies between kernel_harden.rs, linear_kernel.rs and ops.rs; ojas-cuda's assert_all_pass is copied in four device tests and assert_not_wired in src/step.rs and tests/step_refuses.rs; poll_until is near-identical in ojas-cuda/src/lib.rs and ojas-hip/src/lib.rs"
---

# Task brief v1

## Title
One owner per behaviour: shape_product, training policy out of ojas-cpu, head-dim constant, Metal/wgpu conformance suite, retire the tiny Metal step, one GDN reference, shared test support

Task: gp-structural-dedup
Type: refactor
Status: backlog
Priority: 2 (Normal)
Severity: medium
Labels: refactor, cleanup, tests, architecture

## Repositories
- ojas

## Description
Gap audit 2026-10-07. The audit agents did not consult DevMap: GitPulse's brief reported the graph unavailable (its bundled binary reads schema 23, the store is 24), but the devmap plugin itself was healthy (generation 6811). So every "no caller" below is an rg text search and must be re-checked with DevMap (`devmap_dead_symbols`, `devmap_impact`) before deleting. One spot check: `devmap_impact ojas-metal/src/gpu.rs` (depth 2, walk_incomplete) shows `ojas-metal/src/device.rs` importing gpu.rs for `gate_dbias_threads`, consistent with keeping that piece. V = verified, I = inferred.

- **shape_product x4 with three behaviours [V]:** `ojas-core/src/limits.rs:30` (zero axis gives 0), `ojas-cpu/src/validate.rs:314` (no zero rule), `ojas-cpu/src/train.rs:400` and `ojas-autograd/src/tape.rs:1338` (zero axis is a Shape error). `limits.rs:3-5` claims single ownership.
- **Training policy in the CPU crate [V]:** `ojas-model/src/trainer.rs:43-45` imports `scaled_lr, LrSchedule, OptimGroup, ADAM_HYBRID_WEIGHT_DECAY, MUON_*` from ojas-cpu; `trainer.rs:777` rebuilds step configs "as HybridOptimizer::step forms it".
- **Three head-dim constants [V]:** `ojas-metal/src/device.rs:88`, `ojas-core/src/backend.rs:64`, `ojas-kernels/src/geometry.rs:166`, all 256 today, nothing enforcing agreement.
- **Near-duplicate Metal/wgpu suites [V]:** eight paired files, each crate with its own `common/mod.rs`; neither uses `ojas_kernels::{max_abs, linear_close, splitmix_f32}`. (`gp-ci-and-repo-hygiene` dedups autograd helpers only.)
- **Tiny Metal step [I, rg only]:** no crate outside ojas-metal references `ojas_metal::gpu`, `tiny_train_step` or `TinyShape`; it keeps head dim 64 / seq <= 16 limits (`gpu.rs:22-32`) and is the only user of `causal_attn_bwd.metal` (`build.rs:14`).
- **ojas-cuda duplicates [V]:** `bf16.rs:11-29` matches `ojas_core` `autocast.rs:48-66` bit for bit.

Chesterton's fence applies to every removal: find why it exists first.

**Added by the second gap audit 2026-10-07** (DevMap was consulted this time: `devmap_dead_symbols` returned 42 rows with `walk_incomplete` (62,512 unresolved attribution sites), so its "dead" rows are candidates only; `devmap_clones` (min 80 nodes) found 150 groups and showed 106, truncated. [V] re-read by the auditor; [A] read by an audit subagent, not re-read):
- **GemmBlocks/CacheBudget computed, not consumed [A; V that they live in ojas-device/src/tuning.rs and plan.rs].**
- **Two ojas-owned GDN references [V via DevMap clones]:** exact-clone group on `Inputs` across ojas-oracle/src/gdn.rs, ojas-cuda/tests/reference/gdn_published.rs and the tessl fixture copy.
- **Test-helper clone groups [V via DevMap clones]:** the groups named in the criterion. docs/ vs site/ JS duplicates are left to gp-docs-pipeline-single-source.
- **Checked and dropped:** three JSON parsers with different grammars. ojas-data/src/bpe.rs:484-485 documents why it does not use ojas-io's parser (vocab keys must reject every surrogate), so the split is deliberate. DevMap's high-confidence dead `CeCase` (ojas-cuda/src/small_smoke.rs) is used at :669-696, so it is not dead.
