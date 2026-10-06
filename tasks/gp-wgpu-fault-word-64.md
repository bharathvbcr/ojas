---
id: "gp-wgpu-fault-word-64"
title: "Expand WGPU Fault Status Buffer to 64-Bit Word or Two-Word Structure"
status: ready
priority: 2
severity: medium
type: refactor
owner: "unassigned"
due: "none"
labels:
  - "wgpu"
  - "gpu"
  - "faults"
  - "contract"
repositories:
  - "ojas"
planned_files:
  - "ojas-wgpu/src/backend.rs"
  - "ojas-wgpu/src/context.rs"
  - "ojas-kernels/src/wgsl/"
  - "ojas-wgpu/tests/faults.rs"
acceptance_criteria:
  - "Expand the 32-bit fault status word in WgpuBackend and WGSL kernels to a 64-bit word (atomic<u64> or array<u32, 2>)"
  - "Ensure new operations beyond the current 32 allocated op bits can record non-finite and numerical faults without collision or overflow"
  - "Verify deferred fault reporting order preserves first-fault precedence"
  - "All wgpu fault test suites pass without regression"
---

# Task brief v1

## Title
Expand WGPU Fault Status Buffer to 64-Bit Word or Two-Word Structure

Task: gp-wgpu-fault-word-64
Type: refactor
Status: ready
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: wgpu, gpu, faults, contract

## Repositories
- ojas

## Description
In `ojas-wgpu`, kernel status and non-finite fault tracking is recorded in a shared device status buffer. As recorded in `docs/pytorch-parity-plan.md` (§4 open residue), all 32 bits of the current 32-bit status word are fully allocated. The last four bits were assigned to `accumulate_grad`, `linear_cross_entropy_mean`, `cached_attention_forward`, and `kv_cache_write`. Adding any 33rd op that reports faults through the deferred fault protocol requires expanding the status buffer capacity.

### Progress (audit of 9668bfa, 2026-10-05)
Not started: `OP_NAMES: [&str; 32]` (`ojas-wgpu/src/backend.rs:85`), `Op::bit` is `1u32 << n` (:128-130), and `fault.wgsl` still uses `array<atomic<u32>>`.

**The limit is already being worked around (verified):** the wgpu `cast_bf16` added in 1f26a4b opens its job with a fault mask of `0` (`ojas-wgpu/src/backend.rs:1827`), while every other op passes `op.bit()` (:527). Inferred: a fault raised inside the `cast_bf16` kernel sets no bit, so `sync` cannot report it. Give it a real bit once the word is widened.

## Acceptance criteria
- [ ] Expand the 32-bit fault status word in WgpuBackend and WGSL kernels to a 64-bit word (atomic<u64> or array<u32, 2>)
- [ ] Ensure new operations beyond the current 32 allocated op bits can record non-finite and numerical faults without collision or overflow
- [ ] Verify deferred fault reporting order preserves first-fault precedence
- [ ] All wgpu fault test suites pass without regression

## Planned files
- ojas-wgpu/src/backend.rs
- ojas-wgpu/src/context.rs
- ojas-kernels/src/wgsl/
- ojas-wgpu/tests/faults.rs
