---
id: "gp-attention-lse-return"
title: "Return LogSumExp (LSE) from Forward Attention to Accelerate Backward Pass"
status: ready
priority: 1
severity: medium
type: refactor
owner: "unassigned"
due: "none"
labels:
  - "attention"
  - "performance"
  - "autograd"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/backend.rs"
  - "ojas-cpu/src/backend.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-metal/kernels/ojas_backend.metal"
  - "ojas-wgpu/src/backend.rs"
  - "ojas-autograd/src/tape.rs"
acceptance_criteria:
  - "Update Backend::causal_sdpa_forward trait signature to return (Tensor, Tensor) representing output activations and row-wise LSE statistics"
  - "Modify CPU, Metal, and wgpu attention forward kernels to write out LSE"
  - "Refactor causal_sdpa_backward to consume preserved LSE rather than executing an extra softmax recomputation pass"
  - "Benchmark attention backward latency reduction (targeting 10% to 25% speedup on Metal and wgpu)"
  - "Verify bit-level and numerical equivalence with existing backward test suites"
---

# Task brief v1

## Title
Return LogSumExp (LSE) from Forward Attention to Accelerate Backward Pass

Task: gp-attention-lse-return
Type: refactor
Status: ready
Priority: 1 (High)
Severity: medium
Owner: unassigned
Due: none
Labels: attention, performance, autograd

## Repositories
- ojas

## Description
Backend::causal_sdpa_forward currently returns only the activation output Tensor, dropping the row-wise softmax statistics (max and sum). As a consequence, causal_sdpa_backward is forced to re-execute a forward pass / softmax reduction pass to recompute statistics before calculating gradients.

This recomputation wastes an estimated 10%–25% of backward GPU execution time. This task updates the trait signature to return (Tensor, Tensor) (output and LSE) and consumes LSE directly in backward dispatches.

## Acceptance criteria
- [ ] Update Backend::causal_sdpa_forward trait signature to return (Tensor, Tensor) representing output activations and row-wise LSE statistics
- [ ] Modify CPU, Metal, and wgpu attention forward kernels to write out LSE
- [ ] Refactor causal_sdpa_backward to consume preserved LSE rather than executing an extra softmax recomputation pass
- [ ] Benchmark attention backward latency reduction (targeting 10% to 25% speedup on Metal and wgpu)
- [ ] Verify bit-level and numerical equivalence with existing backward test suites

## Planned files
- ojas-core/src/backend.rs
- ojas-cpu/src/backend.rs
- ojas-metal/src/device.rs
- ojas-metal/kernels/ojas_backend.metal
- ojas-wgpu/src/backend.rs
- ojas-autograd/src/tape.rs
