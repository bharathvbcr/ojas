---
id: "gp-backend-architecture-decisions"
title: "Resolve Backend Architecture Spikes: HIP Backend Scope, WGPU Cooperative-Matrix Safety and Prepaid Session Mode"
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
  - "A decision is recorded on the prepaid session mode (each session reserving its cap from the shared process ceiling at LOAD), deferred to the user in docs/adaptive-resources.md:224 and :227: today preflight checks room once and reserves nothing, so a second session can take the room between check_room and apply and a refusal inside apply can poison a trainer. If adopted, a feature task is filed; if not, the narrow race is documented as accepted"
  - "Whichever way the HIP decision goes, ojas-hip stops mislabelling failures and documents its unsafe: hip_status (ojas-hip/src/lib.rs:81-95) maps every non-OOM status, including memcpy, stream and event failures, to DeviceError::NoDevice; about 18 unsafe sites in the hip feature path (lib.rs:135-298) have no // SAFETY: comment; it always uses device 0. If the crate is reduced to a probe, the fix is scoped to what the probe keeps"
---

# Task brief v1

## Title
Resolve Backend Architecture Spikes: HIP Backend Scope, WGPU Cooperative-Matrix Safety and Prepaid Session Mode

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
3. **Prepaid session mode (added by the second gap audit 2026-10-07) [V doc]:** `docs/adaptive-resources.md:227` says "Preflight is a check at one moment, not a reservation... It is a decision for the user and is not made here", and :224 records "Prepaid session mode: deferred to the user". No card owned it.
4. **HIP hygiene either way (added 2026-10-07):** `hip_status` returns `NoDevice` for every status except 0 and 2 [V, lib.rs:81-95]; the unsafe-without-SAFETY count and the fixed device 0 are [A].

## Acceptance criteria
- [ ] A decision is recorded on ojas-hip: implement full HIP backend (file feature task), or document ojas-hip as a memory probe and make BackendId::Hip refuse clearly at selection time
- [ ] The decision and rationale are recorded in docs/pytorch-parity-plan.md regarding allowing unsafe in ojas-wgpu for cooperative-matrix GEMM
- [ ] If cooperative-matrix GEMM is approved, a follow-up feature task is filed with a specific GEMM benchmark throughput target
- [ ] A decision is recorded on the prepaid session mode (each session reserving its cap from the shared process ceiling at LOAD), deferred to the user in docs/adaptive-resources.md:224 and :227: today preflight checks room once and reserves nothing, so a second session can take the room between check_room and apply and a refusal inside apply can poison a trainer. If adopted, a feature task is filed; if not, the narrow race is documented as accepted
- [ ] Whichever way the HIP decision goes, ojas-hip stops mislabelling failures and documents its unsafe: hip_status (ojas-hip/src/lib.rs:81-95) maps every non-OOM status, including memcpy, stream and event failures, to DeviceError::NoDevice; about 18 unsafe sites in the hip feature path (lib.rs:135-298) have no // SAFETY: comment; it always uses device 0. If the crate is reduced to a probe, the fix is scoped to what the probe keeps

## Planned files
- ojas-hip/
- ojas-device/src/lib.rs
- docs/backends.md
- docs/pytorch-parity-plan.md
- ojas-wgpu/src/context.rs
