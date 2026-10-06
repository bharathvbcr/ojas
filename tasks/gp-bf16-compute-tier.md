---
id: "gp-bf16-compute-tier"
title: "Implement BF16 Operand / FP32 Accumulation Compute Tier and Muon NS5 Path"
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
  - "optimizer"
  - "muon"
  - "parity"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/backend.rs"
  - "ojas-core/src/dtype.rs"
  - "ojas-core/src/autocast.rs"
  - "ojas-cpu/src/backend.rs"
  - "ojas-cpu/src/optim.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-metal/kernels/ojas_backend.metal"
  - "ojas-wgpu/src/backend.rs"
acceptance_criteria:
  - "Add BF16 tensor storage and kernel operands across Backend trait methods"
  - "Implement BF16 GEMM with FP32 accumulation on CPU (Accelerate / NEON), Metal (Apple matrix units), and wgpu"
  - "Implement BF16 attention forward and backward kernels with numerical stability guards"
  - "Verify that batch size B >= 8 micro-batches fit in Apple Silicon unified memory without paging or thrashing"
  - "Parity and loss divergence tests pass against FP32 reference, with a stated divergence bound (not only finite-and-different)"
  - "Add optional bf16 compute path for Newton-Schulz 5 quintic iterations in Muon optimizer"
  - "Align update magnitudes exactly with PyTorch and nanolab stock optimizer steps"
  - "Measure throughput improvement from BF16 batched matrix multiplications during orthogonalization"
  - "Golden trace parity test verifying convergence within 1e-4 nats against stock nanolab trace"
---

# Task brief v1

## Title
Implement BF16 Operand / FP32 Accumulation Compute Tier and Muon NS5 Path

Task: gp-bf16-compute-tier
Type: feature
Status: ready
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: none
Labels: kernel, precision, performance, memory, optimizer, muon, parity

## Repositories
- ojas

## Description
Consolidated task combining the mixed-precision compute tier (`gp-bf16-compute-tier`) and the Muon BF16 Newton-Schulz optimizer path (`gp-muon-bf16-ns5`).

Ojas compute across backends is currently strictly f32. At realistic batch sizes (B >= 8), FP32 thrashes unified memory on Apple Silicon, and leaves Tensor Cores and Apple matrix units running at fractional FLOPS. Furthermore, upstream nanolab and PyTorch Muon execute the quintic Newton-Schulz iteration in bf16 (`X = G.bfloat16()`), causing Ojas f32 NS5 to drift up to 1.8e-4 nats over the first 5 steps compared to stock nanolab.

This task introduces a true BF16-operand / FP32-accumulation compute tier across the `Backend` trait (mixed-precision GEMM and attention), together with an optional BF16 Newton-Schulz 5 path in the Muon optimizer.

### Audit status and progress notes (2026-10-05)
Commit 1f26a4b landed **bf16 emulated in f32 storage**, not native BF16 compute:
- `Autocast<B>` (`ojas-core/src/autocast.rs`) rounds f32 values to bf16 precision, keeps them in f32 buffers, and tags them as rounded. The only new Backend op was `cast_bf16`, which takes f32 and returns f32.
- Linear and attention under autocast round inputs then invoke existing f32 kernels. Storage remains f32, saving no memory.
- Existing tests (`autocast_bf16_is_finite_and_moves_a_parameter`) only test finite variance; true convergence divergence bounds against FP32 reference remain open.
- The Muon NS5 iteration in `ojas-cpu/src/optim.rs` and `ojas-metal/src/device.rs` executes strictly in f32.

### Execution plan
1. Introduce BF16 tensor storage representation and operands to `Backend` trait methods.
2. Implement BF16 GEMM with FP32 accumulation on CPU (Accelerate / NEON), Metal (Apple matrix units), and wgpu.
3. Implement BF16 attention forward and backward kernels with numerical stability guards.
4. Implement optional BF16 Newton-Schulz 5 quintic iterations for Muon, measuring throughput speedup and golden trace parity within 1e-4 nats against nanolab.
5. Establish bounded loss divergence limits against FP32 baselines.

## Acceptance criteria
- [ ] Add BF16 tensor storage and kernel operands across Backend trait methods
- [ ] Implement BF16 GEMM with FP32 accumulation on CPU (Accelerate / NEON), Metal (Apple matrix units), and wgpu
- [ ] Implement BF16 attention forward and backward kernels with numerical stability guards
- [ ] Verify that batch size B >= 8 micro-batches fit in Apple Silicon unified memory without paging or thrashing
- [ ] Parity and loss divergence tests pass against FP32 reference, with a stated divergence bound (not only finite-and-different)
- [ ] Add optional bf16 compute path for Newton-Schulz 5 quintic iterations in Muon optimizer
- [ ] Align update magnitudes exactly with PyTorch and nanolab stock optimizer steps
- [ ] Measure throughput improvement from BF16 batched matrix multiplications during orthogonalization
- [ ] Golden trace parity test verifying convergence within 1e-4 nats against stock nanolab trace

## Planned files
- ojas-core/src/backend.rs
- ojas-core/src/dtype.rs
- ojas-core/src/autocast.rs
- ojas-cpu/src/backend.rs
- ojas-cpu/src/optim.rs
- ojas-metal/src/device.rs
- ojas-metal/kernels/ojas_backend.metal
- ojas-wgpu/src/backend.rs
