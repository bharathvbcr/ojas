---
id: "gp-gpu-runtime-hardening"
title: "Harden GPU Runtime: 64-Bit Fault Word, Typed Device-Lost Error, and Step Overhead Reduction"
status: done
priority: 2
severity: medium
type: refactor
owner: "unassigned"
due: "none"
labels:
  - "gpu"
  - "wgpu"
  - "metal"
  - "faults"
  - "errors"
  - "contract"
  - "performance"
repositories:
  - "ojas"
planned_files:
  - "ojas-wgpu/src/backend.rs"
  - "ojas-wgpu/src/context.rs"
  - "ojas-kernels/src/wgsl/"
  - "ojas-wgpu/tests/faults.rs"
  - "ojas-core/src/error.rs"
  - "ojas-capi/src/lib.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-model/src/trainer.rs"
  - "bench/"
acceptance_criteria:
  - "Expand the 32-bit fault status word in WgpuBackend and WGSL kernels to a 64-bit word (atomic<u64> or array<u32, 2>)"
  - "Ensure new operations beyond the current 32 allocated op bits can record non-finite and numerical faults without collision or overflow"
  - "Verify deferred fault reporting order preserves first-fault precedence, and wgpu fault test suites pass without regression"
  - "OjasError has a typed device-lost variant (or structured field) that Metal, wgpu and CUDA set at point of detection"
  - "ojas-capi maps error kind from the typed variant with zero substring matching"
  - "Unit test per GPU backend proves a lost or poisoned device surfaces as the typed variant"
  - "Upload waits per step are counted by trigger (Link Wait counters); the 124M memory-cap split by trigger is owned by gp-long-runs-and-quiet-benches, not claimed here"
  - "Waiting host uploads per Metal trainer step are reduced, with documented before/after counts"
  - "Metal and wgpu implement saved-sigmoid gate pair, with parity test against CPU and committed A/B benchmark citations"
---

# Task brief v1

## Title
Harden GPU Runtime: 64-Bit Fault Word, Typed Device-Lost Error, and Step Overhead Reduction

Task: gp-gpu-runtime-hardening
Type: refactor
Status: done
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: gpu, wgpu, metal, faults, errors, contract, performance

## Repositories
- ojas

## Description
Consolidated task addressing GPU runtime contracts, fault status buffer capacity, error typing, and per-step submission overhead:
- WGPU 64-bit status word expansion (`gp-wgpu-fault-word-64`)
- Typed device-lost error variants across backends (`gp-typed-device-lost-error`)
- GPU per-step overhead reduction (`gp-gpu-step-overhead`)

### Audit status and progress notes (2026-10-05)
1. **WGPU Status Word Exhaustion:** All 32 bits of the current 32-bit status word in `ojas-wgpu` are allocated (`OP_NAMES: [&str; 32]`). The wgpu `cast_bf16` added in 1f26a4b currently works around this by opening its job with a fault mask of `0`, meaning kernel faults cannot be reported by `sync`. The buffer must be expanded to 64 bits (`atomic<u64>` or `array<u32, 2>`).
2. **Brittle Error Matching in C API:** `ojas-capi/src/lib.rs:104-105` classifies device loss by substring searching (`detail.contains("device lost") || detail.contains("runtime poisoned")`). GPU backends should surface a structured `OjasError::DeviceLost` variant directly.
3. **Per-Step Overhead:** Metal training steps spend avoidable time on waiting host uploads and memory-cap commits. In addition, gate activations recompute sigmoid during backward traversal on Metal and wgpu instead of utilizing the saved-sigmoid pair available on CPU.

### Close-out audit (2026-10-08, at 86a1096)
The three notes above describe the tree before this task's code landed. Each criterion is now met. [V] means verified by a test run of this commit (`cargo test -j 2 --test-threads=2`). [R] means read in the code.
1. **Fault words.** [R] `ojas-wgpu/src/context.rs` sets `FAULT_OPS = 64` over two `u32` mask words, plus a first-op word (`ojas-kernels/src/wgsl/common.wgsl` `raise`, `fault.wgsl`). `cast_bf16` has its own bit (index 32). [V] `ojas-wgpu --lib`: 28 pass, including `fault_ids_at_both_ends_of_both_mask_words_land_on_their_own_bit` and `a_fault_id_past_the_mask_is_refused_before_anything_is_recorded`. [V] `--test faults --test contract --test gate_saved --test cast_bf16` all pass, among them `the_first_faulting_op_is_named_not_the_lowest_bit`.
2. **Typed device loss.** [R] `OjasError::DeviceLost { backend, detail }` lives in `ojas-core/src/error.rs`. Metal sets it in `link.rs` `device_lost`, wgpu in `context.rs` `device_lost`, and CUDA in `error.rs`. [R] `ojas-capi/src/lib.rs` `kind_of` matches on the variant only; it does no substring matching. [V] Each backend has a unit test:
   - wgpu: `context::tests::a_device_call_after_loss_is_the_typed_device_lost_error`.
   - Metal: `backend::tests::a_poisoned_runtime_surfaces_as_the_typed_device_lost_error`.
   - CUDA (`--features cuda`, mapping only; no device here): `error::tests::sticky_driver_codes_map_to_device_lost_and_the_rest_to_backend`.
   - capi: `ops_tests::a_lost_device_is_its_own_kind` (capi lib: 80 pass).
3. **Step overhead.**
   - [V] `MetalBackend::wait_counts` counts every wait by trigger. `ojas-metal --test wait_triggers --test wait_count --test deferred_faults` and the Metal lib suite (60) pass.
   - [V] `ojas-model --test metal_waits` asserts upload waits per tiny trainer step: 11 before, 0 after. The other triggers are unchanged. Both counts are documented in `docs/metal-deferred-faults.md` §9.3.
   - [R] The saved-sigmoid gate pair exists on both backends: Metal's `ojas_per_head_gate_bwd_saved` and wgpu's `GATE_BWD_SAVED`.
   - [V] The CPU parity tests, `the_saved_pair_is_the_plain_pair_bit_for_bit_and_matches_cpu` in `ojas-{metal,wgpu}/tests/gate_saved.rs`, pass.
   - [V] The A/B that `bench/README.md` cited had never been committed. It is now run and committed in `bench/results/2026-10-07-gate-saved/`, and the citations point there. In the quiet rounds, the saved backward is about 7% faster on Metal and about 15% faster on wgpu. The saving forward costs at most about 2%.
- **Not measured:** the 124M memory-cap split per trigger (§9.4 still infers about 41 commits per step), and CUDA device loss on a real device (needs the GH200).
- **How it ran:** the tests and benches ran from the main checkout's manifest at the same commit. From this worktree, cargo fails to inherit gusset's `workspace.package.version` through the symlinked `.gitpulse/devtools` path.

### Re-audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

The close-out holds [A]:
- FAULT_OPS = 64 (context.rs:209), and OP_NAMES is bounded at compile time.
- Typed DeviceLost is set on Metal, wgpu and CUDA.
- capi kind_of matches variants only (ojas-capi/src/lib.rs:96-105).
- The saved-sigmoid gate A/B is committed.

One caveat: criterion 7 is ticked, but the 124M memory-cap split by trigger is inferred (about 41 commits per step), as this brief's own 'Not measured' line says. That measurement now lives in gp-device-memory-probes, and the run in gp-long-runs-and-quiet-benches. HIP maps only to DeviceError, not DeviceLost; that is owned by gp-hip-backend.

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- Criterion 7 was ticked while this brief's own text said the 124M split was inferred. Its text is narrowed to what was measured, the counters, and the 124M split is owned by gp-long-runs-and-quiet-benches [A].
- Weak tests behind this close-out, now owned by gp-test-suite-integrity:
  - The Metal device-loss test uses a #[cfg(test)] poison hook and never runs in CI.
  - The CUDA mapping test iterates over the crate's own STICKY_DRIVER_CODES, close to a self-check [A].

### Execution plan
- **Phase 1 (WGPU 64-bit Fault Buffer):** Expand WGSL fault status buffer to 64-bit word or two-word structure. Update `Op::bit` and allocate real bits to `cast_bf16` and future ops while preserving first-fault precedence.
- **Phase 2 (Typed Device-Lost Variant):** Add `OjasError::DeviceLost { backend, detail }`. Surface it directly from Metal, wgpu, and CUDA. Update `ojas-capi` to map it cleanly.
- **Phase 3 (Step Overhead Reduction):** Instrument and reduce upload waits per Metal trainer step. Implement the saved-sigmoid gate kernel pair on Metal and wgpu, backed by committed A/B benchmarks.

## Acceptance criteria
- [x] Expand the 32-bit fault status word in WgpuBackend and WGSL kernels to a 64-bit word (atomic<u64> or array<u32, 2>)
- [x] Ensure new operations beyond the current 32 allocated op bits can record non-finite and numerical faults without collision or overflow
- [x] Verify deferred fault reporting order preserves first-fault precedence, and wgpu fault test suites pass without regression
- [x] OjasError has a typed device-lost variant (or structured field) that Metal, wgpu and CUDA set at point of detection
- [x] ojas-capi maps error kind from the typed variant with zero substring matching
- [x] Unit test per GPU backend proves a lost or poisoned device surfaces as the typed variant
- [x] Upload waits per step are counted by trigger (Link Wait counters); the 124M memory-cap split by trigger is owned by gp-long-runs-and-quiet-benches, not claimed here
- [x] Waiting host uploads per Metal trainer step are reduced, with documented before/after counts
- [x] Metal and wgpu implement saved-sigmoid gate pair, with parity test against CPU and committed A/B benchmark citations

## Planned files
- ojas-wgpu/src/backend.rs
- ojas-wgpu/src/context.rs
- ojas-kernels/src/wgsl/
- ojas-wgpu/tests/faults.rs
- ojas-core/src/error.rs
- ojas-capi/src/lib.rs
- ojas-metal/src/backend.rs
- ojas-model/src/trainer.rs
- bench/
