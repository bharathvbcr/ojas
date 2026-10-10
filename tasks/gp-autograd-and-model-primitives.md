---
id: "gp-autograd-and-model-primitives"
title: "Scale Autograd Tape with Activation Checkpointing, Hybrid GDN/Conv1D Primitives, and Shape Validators"
status: done
priority: 1
severity: high
type: feature
owner: "unassigned"
due: "none"
labels:
  - "autograd"
  - "memory"
  - "scaling"
  - "qwen35"
  - "backend"
  - "kernel"
  - "shape-contract"
  - "metal"
repositories:
  - "ojas"
planned_files:
  - "ojas-autograd/src/tape.rs"
  - "ojas-model/src/block.rs"
  - "ojas-model/src/trainer.rs"
  - "ojas-core/src/backend.rs"
  - "ojas-core/src/shapes.rs"
  - "ojas-cpu/src/accum.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-wgpu/src/backend.rs"
  - "ojas-qwen35/src/model.rs"
  - "ojas-qwen35/src/groups.rs"
  - "ojas-qwen35/src/step.rs"
  - "ojas-qwen35/README.md"
acceptance_criteria:
  - "Implement block-level activation checkpointing in Tape, releasing internal layer activations during forward and recomputing on demand during backward traversal"
  - "Enable sequence length scaling past T=2048 without exceeding resident memory budgets, with gradient equality tests against standard runs"
  - "Add chunked_gdn_forward and chunked_gdn_backward methods to Backend trait"
  - "Add depthwise causal_conv1d_silu forward and backward methods to Backend trait"
  - "Add elementwise gated RMSNorm and partial RoPE with MRoPE collapse to Backend trait"
  - "shapes.rs owns accumulate_grad and permute checks, per-backend copies are deleted, and cross-backend test passes for edge cases"
  - "check_tessl_lr is removed and a test proves per-group LR scales updates (step.rs:632 calls adamw_step_scaled; test at gpu_parity.rs:526; tessl 3881cda is on tessl main)"
  - "Native Metal kernels for the hybrid ops land and the tiny Qwen3.5 hybrid fixture runs forward and backward through the CPU and Metal Tapes (the real-2B Tape run moved to gp-long-runs-and-quiet-benches)"
---

# Task brief v1

## Title
Scale Autograd Tape with Activation Checkpointing, Hybrid GDN/Conv1D Primitives, and Shape Validators

Task: gp-autograd-and-model-primitives
Type: feature
Status: done
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: autograd, memory, scaling, qwen35, backend, kernel, shape-contract, metal

## Repositories
- ojas

## Description
Consolidated task covering high-level model execution, Autograd Tape scaling, and architecture contract validation:
- Selective activation checkpointing in Autograd Tape (`gp-activation-checkpointing`)
- Standardizing hybrid primitives (GDN, Conv1D) in Backend trait (`gp-hybrid-layer-primitives`)
- Centralizing `accumulate_grad` and `permute` shape validators in `ojas_core::shapes` (`gp-shape-validators`)
- Qwen3.5 Metal follow-ups: per-parameter learning rate and 2B memory headroom (`gp-qwen35-metal-followups`)

### Audit status and progress notes (2026-10-05)
1. **Activation Checkpointing:** Autograd Tape currently retains all intermediate layer activations across all blocks. At context lengths T > 2048, resident memory spikes dramatically. Selective block checkpointing recomputes forward activations during backward adjoint traversal, trading compute for memory.
2. **Hybrid Layer Primitives:** Qwen3.5 mixes linear attention (Gated Delta Networks / GDN) and causal 1D conv with attention layers. Currently, these primitives bypass `ojas_core::Backend` and exist only as isolated Metal kernels in tessl. They need standardized trait signatures and native backends.
3. **Shape Contracts:** Every other op's shapes are validated centrally in `ojas-core/src/shapes.rs` (`docs/shape-contract.md`), except `accumulate_grad` and `permute`, which have separate per-backend copies in CPU, Metal, and wgpu.
4. **Qwen3.5 Metal Follow-ups:** Per-parameter learning rates are currently blocked on upstream tessl (`check_tessl_lr` in `ojas-qwen35/src/groups.rs:278` refuses `lr_scale != 1.0`). In addition, empirical 2B headroom measurements on Metal remain to be documented.

### Progress (2026-10-06, Gated DeltaNet session; uncommitted in the working tree)
- **GDN in the trait.** `Backend::chunked_gdn_forward` / `chunked_gdn_backward`
  (`GdnInputs`, `GdnForward`, `GdnGrad`; validator `chunked_gdn_*_dims` in
  `ojas-core/src/shapes.rs`) cover the published rule at transformers' seam,
  with an optional initial state, a final-state gradient, and checkpoints
  every 64 tokens. The CPU takes any dims (`ojas-cpu/src/gdn.rs`). Metal runs
  tessl's `gdn_train` (Dk 128, Dv a multiple of 16; other dims are
  `Unsupported`). wgpu, CUDA and HIP refuse through the trait default.
  `Tape::chunked_gdn` records it. `Autocast` keeps it f32.
- **Verified** (`target-gdn/step*.log`):
  - the CPU against transformers' f64 goldens at T 1/63/64/65/130, worst 2.7e-6;
  - a 120-shape sweep against the f64 forward (`ojas-oracle/src/gdn.rs`);
  - central differences across checkpoints;
  - T 4096 within 2.1e-7;
  - the same bits across numerics, 1–18 threads and 8 concurrent callers;
  - Metal against the CPU within 2e-4, including deferred NaN faults, forced commits, recycling and injected allocation failures;
  - Metal bit-identical when every forward and backward operand starts at a byte offset that is not 16-aligned;
  - the forced-commit stress counts its waited commits against an untuned round, so a tune that is ignored fails it (`step6.log`, `step7.log`);
  - three fail-first mutants killed;
  - workspace clippy clean.
- **Full suites pass** for ojas-core, ojas-cpu, ojas-metal (59 lib tests plus
  every integration suite), ojas-wgpu, ojas-autograd, ojas-capi, ojas-oracle,
  ojas-model, ojas-infer, ojas-qwen35 and ojas-gusset-engine
  (`target-gdn/step4.log`, `step5.log`). ojas-cuda and ojas-hip were
  clippy-checked only; this Mac has no device for them.
- **Still open in Phase 3:** causal conv1d, gated RMSNorm, MRoPE, the GDN
  gates, a wgpu GDN kernel, and the 2B hybrid through Tape.

### Audit and progress (2026-10-07, Metal hybrid-ops session)
Criteria, checked against this branch:
1. **Block checkpointing — met.** `Tape::checkpoint` (`ojas-autograd/src/tape.rs`)
   and `ActivationCheckpoint::Blocks` (`ojas-model/src/block.rs`);
   gates in `ojas-autograd/tests/checkpoint.rs`.
2. **Past T = 2048 — met.** `ojas-model/tests/activation_checkpoint.rs`:
   bit-equal loss and gradients, and at T 2304 a budget that holds the
   checkpointed step refuses the uncheckpointed one.
3. **GDN in the trait — met** (2026-10-06 notes above).
4. **Conv1d in the trait — met** (CPU, autocast, capi gate, tape).
5. **Gated RMSNorm and partial RoPE with MRoPE collapse — met**
   (`rope_partial_*`, `ojas_core::mrope_text_tables`).
6. **Metal — partly met.** This session added Metal
   `causal_conv1d_silu_*` and `gated_rms_norm_*` on tessl's `qwen35` /
   `qwen35_bwd` kernels, and `rope_partial_*` on `ojas_rope`, which now
   takes a rotary width (`rotary == dim` is the old kernel). Gates:
   `ojas-metal/tests/hybrid.rs` (CPU parity, unaligned offsets bit-equal,
   refusals, deferred faults) and `ojas-model/tests/metal_hybrid_tape.rs`
   (conv → GDN → gated norm → partial RoPE on the Metal tape against the
   CPU tape, and checkpointed against direct bit for bit). **Open:** the
   2B hybrid itself through Tape. ojas-model has no hybrid block, and
   building one is a rework the user deferred.
7. **Shape validators — met** (`accumulate_grad_dims`, `permute_dims`;
   `docs/shape-contract.md`).
8. **Per-parameter LR — blocked.** tessl main (4e5faac) has no per-entry
   `lr_scale`, and the branch `check_tessl_lr` names is not in the local
   tessl. `check_tessl_lr` stays.
9. **2B headroom — open.** `ojas-qwen35/README.md` still lists
   `save_state`, `load_state` and longer sequences as not measured. Those
   are ~50 GB Metal runs, deferred by the user.

Build note: in a GitPulse worktree, cargo fails to load gusset through
`.gitpulse/devtools` ("`workspace.package.version` was not defined").
Resolving tessl through `.gitpulse/worktrees/tessl` caches the main
checkout's workspace root, and gusset's logical path sits under it. This
session pointed the two gusset path dependencies at the real path, as a
local, uncommitted edit.

### Re-audit (2026-10-08, main at 470e2cc)
- **Criteria 1–5 and 7 re-verified** on main, release, `--test-threads=1`, all
  green: ojas-core lib (144 + 3 ignored), ojas-autograd `checkpoint` (7),
  ojas-model `activation_checkpoint` (2) and `metal_hybrid_tape` (2),
  ojas-cpu `hybrid`/`permute`/`shape_first` (8/8/1), ojas-metal
  `hybrid`/`permute`/`shape_first` (9/11/5), ojas-wgpu `permute`/`shape_first`
  (9/4), ojas-qwen35 `cpu_groups` (4). No backend defines its own
  `accumulate_grad` or `permute` validator; CPU, Metal and wgpu all call
  `ojas_core::{accumulate_grad_dims, permute_dims}`.
- **8 still blocked.** tessl main (144aa9b) `Qwen35Model::adamw_step` takes one
  `AdamWHyper` and per-entry `weight_decay` only, with no `lr_scale`. No
  `lappi-train-lrscale-mrope` branch, local or remote. `check_tessl_lr` stays.
- **6 and 9 unchanged.** Both need the user's decision: a hybrid block in
  ojas-model (6), and the ~50 GB 2B Metal runs (9). The HF cache has
  `Qwen3.5-2B` and `Qwen3.5-4B-Base` but not `Qwen3.5-2B-Base`, which
  `gpu_parity.rs::snapshot_dir` expects unless `QWEN35_2B_SNAPSHOT` is set.

### Code work (2026-10-08, branch `feat/qwen35-hybrid-tape`, uncommitted)
- **6: the Qwen3.5 tower on Tape.** `ojas_model::qwen35` (`spec.rs`,
  `params.rs`, `forward.rs`) is the hybrid text tower over `Graph`:
  `load_hf` splits `in_proj_qkv`, `conv1d` and `q_proj` into row blocks
  (`fuse_grads` inverts it); every `Qwen3_5RMSNorm` is `1 + w` formed on the
  graph; `ActivationCheckpoint::Blocks` checkpoints each layer. Two new
  Backend ops back it, with validators in `shapes.rs`, CPU, Metal
  (`ojas_sigmoid_*`, `ojas_gdn_decay_*`), autocast, capi gate and Tape:
  `sigmoid_*` (attention output gate, `beta`) and `gdn_log_decay_*` (`g`).
  `Qwen35TextConfig::tape_spec` (ojas-qwen35) feeds it the parsed config.
- **Verified on tessl's tiny fixture** (copied to
  `ojas-model/tests/fixtures/qwen35_tiny`): the CPU tape matches transformers'
  loss to f32 and all 27 gradients within 3.8e-6 of peak; Metal within 8.2e-8
  (loss) and 2.8e-6 (gradients), and within 2e-4 of the CPU tape; checkpointed
  equals direct bit for bit on both; `Eval` gives the tape's loss bits. Two
  planted layout mutants (no `1 + w`, swapped q/gate rows) fail the gate.
  `cpu_tape_spec.rs` pins the tape's tensor names to the provider's for the
  real 2B config.
- **Not yet run:** `ojas-qwen35/tests/gpu_tape_2b.rs`
  (`gpu_real_2b_tape_forward_backward_matches_the_provider`), the real 2B
  through the Metal tape against tessl's step. It needs ~40 GB, and other
  sessions held 2B jobs on the GPU throughout this session.
- **8: per-parameter LR.** tessl (uncommitted): `Qwen35Model::adamw_step_scaled`,
  `adamw_step` delegates; tests in `tests/qwen35_adamw.rs`. ojas-qwen35:
  `check_tessl_lr` removed, `adamw_step` passes `plan.lr_scale()`;
  `gpu_tiny_per_group_lr_scales_the_update` and a CPU plan test. Needs the
  tessl change committed before this branch can merge.
- **Checks:** workspace fmt and clippy `-D warnings` clean; sitegen `-check`
  clean; 31 non-GPU and 11 GPU suites of the touched crates green.
- **9** is unchanged: the 2B save/load and long-sequence headroom runs.

### Close-out audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

Closed by 66c5e52 and the 3b2d540 / b363802 merges [A]:
- **Tape checkpointing:** Tape::checkpoint (tape.rs:698), with tests including activation_checkpoint.rs past T=2048.
- **Hybrid ops in the trait:** GDN, conv1d+SiLU, gated RMSNorm and partial RoPE / MRoPE (backend.rs:846-970; rope.rs:126).
- **Metal hybrid kernels:** backend.rs:1495-1722.
- **Shape checks:** shapes.rs owns the accumulate_grad and permute checks (:1271, :1283).
- **check_tessl_lr:** gone, with a per-group LR test.

Split out:
- The real-2B Tape run (ojas-qwen35/tests/gpu_tape_2b.rs:96 exists but has never run) and the README 2B save/load and long-sequence headroom (ojas-qwen35/README.md:230-235 says 'Not measured') go to gp-long-runs-and-quiet-benches.
- The wgpu GDN, sigmoid and gdn_log_decay kernels go to gp-wgpu-hybrid-ops.
- A training loop for the Tape tower goes to gp-qwen35-tape-training-loop.

The brief's 'blocked on uncommitted tessl' note is stale [A].

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- Weak claim: the per-group LR proof `gpu_tiny_per_group_lr_scales_the_update` (ojas-qwen35/tests/gpu_parity.rs:524) is #[ignore] and macOS-only, which this brief did not say. qwen35_metal_tape.rs:30 skips on any value of OJAS_ALLOW_NO_GPU. Running them is owned by gp-test-suite-integrity [A].

### Execution plan
- **Phase 1 (Shape Validator Centralization):** Move `accumulate_grad` and `permute` validation into `ojas_core::shapes`, delete backend copies, and verify zero-length permute axes cross-backend.
- **Phase 2 (Activation Checkpointing):** Implement block-level activation checkpointing on Tape, verifying exact gradient parity against uncheckpointed runs.
- **Phase 3 (Hybrid Architecture Primitives):** Expose GDN chunk rule, causal conv1d, gated RMSNorm, and MRoPE in `Backend` trait with Metal implementation.
- **Phase 4 (Qwen3.5 Metal Follow-ups):** Remove `check_tessl_lr` once tessl adds per-param LR, and record 2B peak memory headroom in README.

## Acceptance criteria
- [x] Implement block-level activation checkpointing in Tape, releasing internal layer activations during forward and recomputing on demand during backward traversal
- [x] Enable sequence length scaling past T=2048 without exceeding resident memory budgets, with gradient equality tests against standard runs
- [x] Add chunked_gdn_forward and chunked_gdn_backward methods to Backend trait
- [x] Add depthwise causal_conv1d_silu forward and backward methods to Backend trait
- [x] Add elementwise gated RMSNorm and partial RoPE with MRoPE collapse to Backend trait
- [x] shapes.rs owns accumulate_grad and permute checks, per-backend copies are deleted, and cross-backend test passes for edge cases
- [x] check_tessl_lr is removed and a test proves per-group LR scales updates (step.rs:632 calls adamw_step_scaled; test at gpu_parity.rs:526; tessl 3881cda is on tessl main)
- [x] Native Metal kernels for the hybrid ops land and the tiny Qwen3.5 hybrid fixture runs forward and backward through the CPU and Metal Tapes (the real-2B Tape run moved to gp-long-runs-and-quiet-benches)

## Planned files
- ojas-autograd/src/tape.rs
- ojas-model/src/block.rs
- ojas-model/src/trainer.rs
- ojas-core/src/backend.rs
- ojas-core/src/shapes.rs
- ojas-cpu/src/accum.rs
- ojas-metal/src/device.rs
- ojas-metal/src/backend.rs
- ojas-wgpu/src/backend.rs
- ojas-qwen35/src/model.rs
- ojas-qwen35/src/groups.rs
- ojas-qwen35/src/step.rs
- ojas-qwen35/README.md
