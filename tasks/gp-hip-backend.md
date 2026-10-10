---
id: "gp-hip-backend"
title: "Implement a Full AMD HIP Backend on ROCm"
status: backlog
priority: 3
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
  - "Device memory is charged to the session Budget like Metal and wgpu (today the probe copy is capped only by MAX_COPY_BYTES, ojas-hip/src/lib.rs:29); HIP out-of-memory is already Capacity (lib.rs:106) and stays so"
  - "A HIP device code is added to the C API and Go API; on a build without the hip feature it is refused at load with a message naming HIP as not compiled (device 6 hits the generic 'load: unknown device' today, ojas-capi/src/load.rs:306)"
  - "Faults follow the deferred-fault contract (docs/metal-deferred-faults.md) and surface at the next sync"
  - "Standing invariant: every unsafe site keeps a // SAFETY: comment and the crate stays forbid(unsafe_code) with the feature off (holds today, lib.rs:16)"
  - "HIP and CUDA share one host shape for error mapping, budget charging and upload (and the device helpers moved to ojas-device by gp-cuda-surface-and-device-helpers), so HIP does not become a third copy"
  - "Device loss surfaces as DeviceLost (HIP maps only to DeviceError today), and the probe's Drop cannot block without bound: the SAFETY claim that hipStreamDestroy waits for queued work (lib.rs:306-311) is checked against the ROCm docs, and hipFree/hipHostFree after a 30 s poll timeout (:397-406) are bounded"
---

# Task brief v1

## Title
Implement a Full AMD HIP Backend on ROCm

Task: gp-hip-backend
Type: feature
Status: backlog
Priority: 3 (Low)
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

### Re-audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- The HIP hygiene from the deleted spike gp-backend-architecture-decisions is done [A]: the status mapping (NoDevice for 3/4/35/100/101, Capacity for 2, Launch otherwise; lib.rs:103-119), SAFETY comments on every unsafe site (:212-411), open_ordinal and check_ordinal, and raw c_int FFI.
- Still open and needing the user: the kernel source question (hipcc at build time or precompiled code objects).

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- Sharpened [A, inferred from the documented HIP/CUDA semantics]: hipStreamDestroy returns at once, so after the 30 s poll timeout the Drop order reaches hipHostFree/hipFree with copies still queued. The probe either blocks without bound or DMA writes into freed pinned memory. Criterion 8 covers it.

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Lowered to P3: off the Mac (Metal) training path. Raise it again when CUDA, HIP or wgpu training becomes a goal.

## Acceptance criteria
- [ ] A ROCm CI runner (self-hosted, AMD GPU, /opt/rocm present) runs `cargo test -p ojas-hip --features hip` on every change to ojas-hip; until it exists, no HIP execution result is reported as passed
- [ ] ojas-hip implements ojas_core::Backend with the same op set and Fast-tier tolerances as WgpuBackend, checked by the ojas-oracle parity harness on the ROCm runner
- [ ] Device memory is charged to the session Budget like Metal and wgpu (today the probe copy is capped only by MAX_COPY_BYTES, ojas-hip/src/lib.rs:29); HIP out-of-memory is already Capacity (lib.rs:106) and stays so
- [ ] A HIP device code is added to the C API and Go API; on a build without the hip feature it is refused at load with a message naming HIP as not compiled (device 6 hits the generic 'load: unknown device' today, ojas-capi/src/load.rs:306)
- [ ] Faults follow the deferred-fault contract (docs/metal-deferred-faults.md) and surface at the next sync
- [ ] Standing invariant: every unsafe site keeps a // SAFETY: comment and the crate stays forbid(unsafe_code) with the feature off (holds today, lib.rs:16)
- [ ] HIP and CUDA share one host shape for error mapping, budget charging and upload (and the device helpers moved to ojas-device by gp-cuda-surface-and-device-helpers), so HIP does not become a third copy
- [ ] Device loss surfaces as DeviceLost (HIP maps only to DeviceError today), and the probe's Drop cannot block without bound: the SAFETY claim that hipStreamDestroy waits for queued work (lib.rs:306-311) is checked against the ROCm docs, and hipFree/hipHostFree after a 30 s poll timeout (:397-406) are bounded

## Planned files
- ojas-hip/
- ojas-capi/src/load.rs
- go/api.go
- docs/backends.md
- .github/workflows/test.yml
