---
id: "gp-backend-architecture-decisions"
title: "Resolve Backend Architecture Spikes: HIP Backend Scope and WGPU Cooperative-Matrix Safety"
status: backlog
priority: 2
severity: medium
type: spike
owner: "unassigned"
due: "none"
labels:
  - "backend"
  - "hip"
  - "wgpu"
  - "performance"
  - "architecture"
  - "needs-your-words"
repositories:
  - "ojas"
planned_files:
  - "ojas-hip/"
  - "ojas-device/src/lib.rs"
  - "docs/backends.md"
  - "docs/pytorch-parity-plan.md"
  - "ojas-wgpu/src/context.rs"
acceptance_criteria:
  - "A decision is recorded on ojas-hip: implement full HIP backend (file feature task), or document ojas-hip as a memory probe and make BackendId::Hip refuse clearly at selection time"
  - "The decision and rationale are recorded in docs/pytorch-parity-plan.md regarding allowing unsafe in ojas-wgpu for cooperative-matrix GEMM"
  - "If cooperative-matrix GEMM is approved, a follow-up feature task is filed with a specific GEMM benchmark throughput target"
---

# Task brief v1

## Title
Resolve Backend Architecture Spikes: HIP Backend Scope and WGPU Cooperative-Matrix Safety

Task: gp-backend-architecture-decisions
Type: spike
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: backend, hip, wgpu, performance, architecture, needs-your-words

## Repositories
- ojas

## Description
Consolidated architectural decision brief unifying pending backend policy choices that require project lead direction (`needs-your-words`):
- HIP backend implementation vs memory probe designation (`gp-hip-backend-decision`)
- Cooperative-matrix GEMM unsafe permission in `ojas-wgpu` (`gp-wgpu-coop-matrix-decision`)

### Details and Context
1. **HIP Backend Scope:** `ojas-hip` currently exists only as a minimal copy-roundtrip memory probe. It does not implement `ojas_core::Backend`. We need a definitive architectural decision: either commit to a full AMD HIP backend implementation on ROCm (defining roadmap and CI runners), or officially classify `ojas-hip` as a hardware probe in `docs/backends.md` and ensure `BackendId::Hip` refuses with a clear diagnostic at selection time.
2. **WGPU Cooperative-Matrix GEMM Unsafe:** wgpu GEMM currently runs at approximately 1.6–3.2 TFLOP/s against PyTorch's 5.2 TFLOP/s (`docs/pytorch-parity-plan.md:203,300`). Enabling GPU matrix units via the WebGPU cooperative-matrix extension would eliminate most of this gap. However, the extension requires allowing `unsafe` code inside `ojas-wgpu`, which currently maintains `#![forbid(unsafe_code)]`. We must decide whether to grant an unsafe exemption specifically for cooperative matrix acceleration.

## Acceptance criteria
- [ ] A decision is recorded on ojas-hip: implement full HIP backend (file feature task), or document ojas-hip as a memory probe and make BackendId::Hip refuse clearly at selection time
- [ ] The decision and rationale are recorded in docs/pytorch-parity-plan.md regarding allowing unsafe in ojas-wgpu for cooperative-matrix GEMM
- [ ] If cooperative-matrix GEMM is approved, a follow-up feature task is filed with a specific GEMM benchmark throughput target

## Planned files
- ojas-hip/
- ojas-device/src/lib.rs
- docs/backends.md
- docs/pytorch-parity-plan.md
- ojas-wgpu/src/context.rs
