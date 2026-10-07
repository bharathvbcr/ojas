---
id: "gp-wgpu-hybrid-ops"
title: "wgpu kernels for the Qwen3.5 hybrid ops: causal conv1d+SiLU, gated RMSNorm, partial RoPE"
status: backlog
priority: 2
severity: medium
type: feature
owner: "unassigned"
due: "none"
labels:
  - "wgpu"
  - "kernel"
  - "qwen35"
  - "parity"
repositories:
  - "ojas"
planned_files:
  - "ojas-wgpu/src/backend.rs"
  - "ojas-kernels/src/wgsl/"
  - "ojas-wgpu/tests/"
  - "docs/op-coverage.md"
acceptance_criteria:
  - "ojas-wgpu overrides the Backend methods for causal conv1d+SiLU, gated RMSNorm and partial RoPE (forward and backward) instead of inheriting the refusing trait defaults"
  - "Each kernel has a parity test against the CPU backend and the f64 reference, plus shape-first malformed-input and non-finite fault tests matching the existing wgpu suites"
  - "docs/op-coverage.md hybrid rows updated for wgpu"
  - "Starts after gp-autograd-and-model-primitives lands the Metal versions, so the trait signatures are settled"
---

# Task brief v1

## Title
wgpu kernels for the Qwen3.5 hybrid ops: causal conv1d+SiLU, gated RMSNorm, partial RoPE

Task: gp-wgpu-hybrid-ops
Type: feature
Status: backlog
Priority: 2 (Normal)
Severity: medium
Labels: wgpu, kernel, qwen35, parity

## Repositories
- ojas

## Description
Gap audit 2026-10-07. `gp-autograd-and-model-primitives` (in progress) Phase 3 names Metal only for these ops, and already notes wgpu GDN; the three non-GDN hybrid ops have no wgpu owner. ojas-wgpu inherits the trait defaults, which return `Unsupported` [V by the override list and `docs/op-coverage.md` hybrid rows]. Filed as its own card rather than rewriting the in-progress brief under a running agent.
