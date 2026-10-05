---
id: "gp-cuda-backend-provider"
title: "Implement Whole-Step Qwen3.5 CUDA Training Provider for GH200"
status: backlog
priority: 1
severity: high
type: feature
owner: "unassigned"
due: "none"
labels:
  - "cuda"
  - "gh200"
  - "backend"
  - "lappi"
repositories:
  - "ojas"
planned_files:
  - "ojas-cuda/Cargo.toml"
  - "ojas-cuda/src/lib.rs"
  - "ojas-qwen35/src/cuda.rs"
acceptance_criteria:
  - "Update cudarc pin from cuda-13040 to cuda-12080 to match GH200 driver environment"
  - "Implement Qwen35Cuda step provider mirroring the Metal ojas-qwen35 API"
  - "Provide NVRTC-compiled kernel pipelines for GEMM, attention, and in-place AdamW"
  - "Eliminate dead helper code (commit_resize) and replace mock tests with real GPU tests"
  - "Verify execution on GH200 with zero symbol panics"
---

# Task brief v1

## Title
Implement Whole-Step Qwen3.5 CUDA Training Provider for GH200

Task: gp-cuda-backend-provider
Type: feature
Status: backlog
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: cuda, gh200, backend, lappi

## Repositories
- ojas

## Description
ojas-cuda is currently a diagnostic probe that does not implement the ojas_core::Backend trait. It has never executed a kernel on a real GPU and is pinned to cuda-13040, which causes missing symbol panics when dynamically loading against GH200 CUDA 12.8 drivers. Consequently, Lappi GH200 training campaigns cannot run.

Following the architecture defined in docs/cuda-backend-scoping.md, this task implements the whole-step Qwen3.5 training provider on CUDA, updates the cudarc pin, and wires real NVRTC compute kernels.

## Acceptance criteria
- [ ] Update cudarc pin from cuda-13040 to cuda-12080 to match GH200 driver environment
- [ ] Implement Qwen35Cuda step provider mirroring the Metal ojas-qwen35 API
- [ ] Provide NVRTC-compiled kernel pipelines for GEMM, attention, and in-place AdamW
- [ ] Eliminate dead helper code (commit_resize) and replace mock tests with real GPU tests
- [ ] Verify execution on GH200 with zero symbol panics

## Planned files
- ojas-cuda/Cargo.toml
- ojas-cuda/src/lib.rs
- ojas-qwen35/src/cuda.rs
