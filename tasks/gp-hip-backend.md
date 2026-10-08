---
id: "gp-hip-backend"
title: "Implement a Full AMD HIP Backend on ROCm"
status: backlog
priority: 2
severity: medium
type: feature
owner: "unassigned"
due: "none"
labels:
  - "hip"
  - "rocm"
  - "backend"
repositories:
  - "ojas"
planned_files:
  - "ojas-hip/"
  - "ojas-capi/src/load.rs"
  - "go/api.go"
  - "docs/backends.md"
  - ".github/workflows/test.yml"
acceptance_criteria:
  - "A ROCm CI runner (self-hosted, AMD GPU, /opt/rocm present) runs `cargo test -p ojas-hip --features hip` on every change to ojas-hip; until it exists, no HIP execution result is reported as passed"
  - "ojas-hip implements ojas_core::Backend with the same op set and Fast-tier tolerances as WgpuBackend, checked by the ojas-oracle parity harness on the ROCm runner"
  - "Device memory is charged to the session Budget like Metal and wgpu, and HIP out-of-memory is Capacity, never NoDevice or Launch"
  - "A HIP device code is added to the C API and Go API; on a build without the hip feature it is refused at load with a message naming HIP as not compiled, never a CPU fallback"
  - "Faults follow the deferred-fault contract (docs/metal-deferred-faults.md) and surface at the next sync"
  - "Every unsafe site keeps a // SAFETY: comment; the crate stays forbid(unsafe_code) with the feature off"
---

# Task brief v1

## Title
Implement a Full AMD HIP Backend on ROCm

Task: gp-hip-backend
Type: feature
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: hip, rocm, backend

## Repositories
- ojas

## Description
Filed by `gp-backend-architecture-decisions`: the user chose (2026-10-08) to commit to a full HIP backend rather than classifying `ojas-hip` as a probe. Today `ojas-hip` is a copy-roundtrip probe on `hip-runtime-sys` 0.1.2 with no kernels. Its status mapping (NoDevice / Capacity / Launch), explicit device ordinal and SAFETY comments landed with the decision task and are the base this work extends.

### Roadmap
1. ROCm CI runner. Nothing can be claimed verified without one; this machine has no AMD GPU and no `/opt/rocm`.
2. Buffers and budget: device tensors, upload, download, charged allocations.
3. GEMM and elementwise ops, then attention, then the optimizer steps, each gated by the oracle parity harness.
4. C API and Go device code, wired last so no caller can select a half-built backend.

### Open question
Kernel source: HIP C++ compiled with hipcc at build time, or precompiled code objects. That choice decides the build dependency and needs asking before adding it.

## Acceptance criteria
- [ ] A ROCm CI runner (self-hosted, AMD GPU, /opt/rocm present) runs `cargo test -p ojas-hip --features hip` on every change to ojas-hip; until it exists, no HIP execution result is reported as passed
- [ ] ojas-hip implements ojas_core::Backend with the same op set and Fast-tier tolerances as WgpuBackend, checked by the ojas-oracle parity harness on the ROCm runner
- [ ] Device memory is charged to the session Budget like Metal and wgpu, and HIP out-of-memory is Capacity, never NoDevice or Launch
- [ ] A HIP device code is added to the C API and Go API; on a build without the hip feature it is refused at load with a message naming HIP as not compiled, never a CPU fallback
- [ ] Faults follow the deferred-fault contract (docs/metal-deferred-faults.md) and surface at the next sync
- [ ] Every unsafe site keeps a // SAFETY: comment; the crate stays forbid(unsafe_code) with the feature off

## Planned files
- ojas-hip/
- ojas-capi/src/load.rs
- go/api.go
- docs/backends.md
- .github/workflows/test.yml
