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
  - "ojas-cpu/src/backend.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-metal/kernels/ojas_backend.metal"
  - "ojas-wgpu/src/backend.rs"
acceptance_criteria:
  - "Add BF16 tensor storage and kernel operands across Backend trait methods"
  - "Implement BF16 GEMM with FP32 accumulation on CPU (Accelerate / NEON), Metal (Apple matrix units), and wgpu"
  - "Implement BF16 attention forward and backward kernels with numerical stability guards"
  - "Verify that batch size B >= 8 micro-batches fit in Apple Silicon unified memory without paging or thrashing"
  - "Parity and loss divergence tests pass against FP32 reference"
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

## Acceptance criteria
- [ ] Add BF16 tensor storage and kernel operands across Backend trait methods
- [ ] Implement BF16 GEMM with FP32 accumulation on CPU (Accelerate / NEON), Metal (Apple matrix units), and wgpu
- [ ] Implement BF16 attention forward and backward kernels with numerical stability guards
- [ ] Verify that batch size B >= 8 micro-batches fit in Apple Silicon unified memory without paging or thrashing
- [ ] Parity and loss divergence tests pass against FP32 reference

## Planned files
- ojas-core/src/backend.rs
- ojas-core/src/dtype.rs
- ojas-cpu/src/backend.rs
- ojas-metal/src/device.rs
- ojas-metal/kernels/ojas_backend.metal
- ojas-wgpu/src/backend.rs
