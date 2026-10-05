---
id: "gp-hybrid-layer-primitives"
title: "Standardize Hybrid Architecture Primitives (GDN, Conv1D) in Backend Trait"
status: backlog
priority: 2
severity: medium
type: feature
owner: "unassigned"
due: "none"
labels:
  - "qwen35"
  - "lappi"
  - "backend"
  - "kernel"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/backend.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-qwen35/src/model.rs"
acceptance_criteria:
  - "Add chunked_gdn_forward and chunked_gdn_backward methods to Backend trait"
  - "Add depthwise causal_conv1d_silu forward and backward methods to Backend trait"
  - "Add elementwise gated RMSNorm and partial RoPE with MRoPE collapse"
  - "Implement native Metal backends utilizing optimized kernels ported from tessl"
  - "Verify Qwen3.5 2B hybrid forward and backward execution through Tape"
---

# Task brief v1

## Title
Standardize Hybrid Architecture Primitives (GDN, Conv1D) in Backend Trait

Task: gp-hybrid-layer-primitives
Type: feature
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: qwen35, lappi, backend, kernel

## Repositories
- ojas

## Description
Hybrid architectures like Qwen3.5 mix linear attention (Gated Delta Networks / GDN) and causal 1D convolutions with standard attention layers. Currently, these primitives bypass the ojas_core::Backend trait entirely and exist only as isolated Metal kernels in tessl.

This task standardizes GDN chunked delta recurrence, causal depthwise conv1d, gated RMSNorm, and partial RoPE directly into the Backend trait and connects them to Tape autograd.

## Acceptance criteria
- [ ] Add chunked_gdn_forward and chunked_gdn_backward methods to Backend trait
- [ ] Add depthwise causal_conv1d_silu forward and backward methods to Backend trait
- [ ] Add elementwise gated RMSNorm and partial RoPE with MRoPE collapse
- [ ] Implement native Metal backends utilizing optimized kernels ported from tessl
- [ ] Verify Qwen3.5 2B hybrid forward and backward execution through Tape

## Planned files
- ojas-core/src/backend.rs
- ojas-metal/src/device.rs
- ojas-qwen35/src/model.rs
