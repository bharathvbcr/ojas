---
id: "gp-bf16-compute-tier"
title: "Implement BF16 Operand / FP32 Accumulation Compute Tier"
status: ready
priority: 0
severity: critical
type: feature
owner: "unassigned"
due: "none"
labels:
  - "kernel"
  - "precision"
  - "performance"
  - "memory"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/backend.rs"
  - "ojas-core/src/dtype.rs"
  - "ojas-core/src/autocast.rs"
  - "ojas-cpu/src/backend.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-metal/kernels/ojas_backend.metal"
  - "ojas-wgpu/src/backend.rs"
acceptance_criteria:
  - "Add BF16 tensor storage and kernel operands across Backend trait methods"
  - "Implement BF16 GEMM with FP32 accumulation on CPU (Accelerate / NEON), Metal (Apple matrix units), and wgpu"
  - "Implement BF16 attention forward and backward kernels with numerical stability guards"
  - "Verify that batch size B >= 8 micro-batches fit in Apple Silicon unified memory without paging or thrashing"
  - "Parity and loss divergence tests pass against FP32 reference, with a stated divergence bound (not only finite-and-different)"
---

# Task brief v1

## Title
Implement BF16 Operand / FP32 Accumulation Compute Tier

Task: gp-bf16-compute-tier
Type: feature
Status: ready
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: none
Labels: kernel, precision, performance, memory

## Repositories
- ojas

## Description
Ojas compute across all backends (CPU, Metal, wgpu) is currently strictly f32. At realistic training batch sizes (B >= 8), FP32 thrashes unified memory on Apple Silicon, whereas BF16 fits comfortably. Furthermore, running FP32 leaves hardware Tensor Cores and Apple matrix units running at half or quarter potential FLOPS.

This task introduces a BF16-operand / FP32-accumulation compute tier across the Backend trait, enabling mixed-precision GEMM and attention kernels.

### Progress (audit of 9668bfa, 2026-10-05, code inspection only; tests not executed)
Commit 1f26a4b's message says it added a "BF16 autocast tier". What actually landed is **bf16 emulated in f32 storage**, not this task:
- `Autocast<B>` (`ojas-core/src/autocast.rs`) rounds f32 values to bf16 precision, keeps them in f32 buffers and tags them as rounded (`ojas-core/src/tensor.rs:126-141`). The only new Backend op is `cast_bf16` (`ojas-core/src/backend.rs:816`), which takes f32 and returns f32. `docs/dtype-policy.md` says so: training storage stays F32, autocast is off by default.
- Linear and attention under autocast round their inputs, then call the existing f32 kernels (`ojas-cpu/tests/autocast_linear.rs`, `autocast.rs:516`). No bf16 GEMM or attention kernel exists on any backend.
- Storage stays f32, so autocast saves no memory, and criterion 4 cannot be met by it.
- Tests: `autocast_off_matches_a_raw_backend_bit_for_bit`, plus `cast_bf16` host-equivalence on Metal and wgpu. `autocast_bf16_is_finite_and_moves_a_parameter` only checks that values are finite and differ from f32. Nothing bounds divergence against the FP32 reference.

All five criteria remain open. Autocast's rounding is a useful oracle for checking the real kernels.

## Acceptance criteria
- [ ] Add BF16 tensor storage and kernel operands across Backend trait methods
- [ ] Implement BF16 GEMM with FP32 accumulation on CPU (Accelerate / NEON), Metal (Apple matrix units), and wgpu
- [ ] Implement BF16 attention forward and backward kernels with numerical stability guards
- [ ] Verify that batch size B >= 8 micro-batches fit in Apple Silicon unified memory without paging or thrashing
- [ ] Parity and loss divergence tests pass against FP32 reference, with a stated divergence bound (not only finite-and-different)

## Planned files
- ojas-core/src/backend.rs
- ojas-core/src/dtype.rs
- ojas-core/src/autocast.rs
- ojas-cpu/src/backend.rs
- ojas-metal/src/device.rs
- ojas-metal/kernels/ojas_backend.metal
- ojas-wgpu/src/backend.rs
