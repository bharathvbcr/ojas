---
id: "gp-gpu-runtime-hardening"
title: "Harden GPU Runtime: 64-Bit Fault Word, Typed Device-Lost Error, and Step Overhead Reduction"
status: ready
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
  - "Upload waits and memory-cap commits per step are measured and counted by trigger, not estimated"
  - "Waiting host uploads per Metal trainer step are reduced, with documented before/after counts"
  - "Metal and wgpu implement saved-sigmoid gate pair, with parity test against CPU and committed A/B benchmark citations"
---

# Task brief v1

## Title
Harden GPU Runtime: 64-Bit Fault Word, Typed Device-Lost Error, and Step Overhead Reduction

Task: gp-gpu-runtime-hardening
Type: refactor
Status: ready
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

### Execution plan
- **Phase 1 (WGPU 64-bit Fault Buffer):** Expand WGSL fault status buffer to 64-bit word or two-word structure. Update `Op::bit` and allocate real bits to `cast_bf16` and future ops while preserving first-fault precedence.
- **Phase 2 (Typed Device-Lost Variant):** Add `OjasError::DeviceLost { backend, detail }`. Surface it directly from Metal, wgpu, and CUDA. Update `ojas-capi` to map it cleanly.
- **Phase 3 (Step Overhead Reduction):** Instrument and reduce upload waits per Metal trainer step. Implement the saved-sigmoid gate kernel pair on Metal and wgpu, backed by committed A/B benchmarks.

## Acceptance criteria
- [ ] Expand the 32-bit fault status word in WgpuBackend and WGSL kernels to a 64-bit word (atomic<u64> or array<u32, 2>)
- [ ] Ensure new operations beyond the current 32 allocated op bits can record non-finite and numerical faults without collision or overflow
- [ ] Verify deferred fault reporting order preserves first-fault precedence, and wgpu fault test suites pass without regression
- [ ] OjasError has a typed device-lost variant (or structured field) that Metal, wgpu and CUDA set at point of detection
- [ ] ojas-capi maps error kind from the typed variant with zero substring matching
- [ ] Unit test per GPU backend proves a lost or poisoned device surfaces as the typed variant
- [ ] Upload waits and memory-cap commits per step are measured and counted by trigger, not estimated
- [ ] Waiting host uploads per Metal trainer step are reduced, with documented before/after counts
- [ ] Metal and wgpu implement saved-sigmoid gate pair, with parity test against CPU and committed A/B benchmark citations

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
