---
id: "gp-head-dim-256-gqa"
title: "Lift Head Dimension Cap to 256 and Support GQA in Training"
status: ready
priority: 0
severity: critical
type: feature
owner: "unassigned"
due: "none"
labels:
  - "attention"
  - "kernel"
  - "autograd"
  - "qwen35"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/backend.rs"
  - "ojas-kernels/src/geometry.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-metal/kernels/ojas_backend.metal"
  - "ojas-wgpu/src/backend.rs"
  - "ojas-autograd/src/tape.rs"
acceptance_criteria:
  - "Remove d <= 128 head dimension limit assertion across Metal and wgpu kernels"
  - "Implement tiled attention forward and backward kernels supporting head dimension 256"
  - "Generalize Backend::causal_sdpa_forward and backward to support Grouped-Query Attention (mismatched Q and KV head counts)"
  - "Autograd Tape records and backpropagates GQA projections correctly"
  - "Unit tests verifying head dimensions 64, 128, and 256 match analytical and CPU reference oracles"
---

# Task brief v1

## Title
Lift Head Dimension Cap to 256 and Support GQA in Training

Task: gp-head-dim-256-gqa
Type: feature
Status: ready
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: none
Labels: attention, kernel, autograd, qwen35

## Repositories
- ojas

## Description
Both Metal and wgpu attention kernels currently enforce a strict head dimension cap of 128 (METAL_MAX_HEAD_DIM and ATTENTION_MAX_HEAD_DIM). Modern architectures like Qwen3.5, Gemma, and Llama 3 require head dimension 256 with Grouped-Query Attention (GQA). In Ojas, any attempt to run head dimension 256 fails immediately with OjasError::UnsupportedHeadDim, and GQA currently only exists for inference decoding, not in the autograd training pass.

This task lifts the head dimension barrier to 256 and generalizes the causal SDPA training forward and backward passes to support GQA.

## Acceptance criteria
- [ ] Remove d <= 128 head dimension limit assertion across Metal and wgpu kernels
- [ ] Implement tiled attention forward and backward kernels supporting head dimension 256
- [ ] Generalize Backend::causal_sdpa_forward and backward to support Grouped-Query Attention (mismatched Q and KV head counts)
- [ ] Autograd Tape records and backpropagates GQA projections correctly
- [ ] Unit tests verifying head dimensions 64, 128, and 256 match analytical and CPU reference oracles

## Planned files
- ojas-core/src/backend.rs
- ojas-kernels/src/geometry.rs
- ojas-metal/src/device.rs
- ojas-metal/kernels/ojas_backend.metal
- ojas-wgpu/src/backend.rs
- ojas-autograd/src/tape.rs
