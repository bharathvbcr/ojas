---
id: "gp-cuda-surface-and-device-helpers"
title: "Narrow ojas-cuda's public surface (57 pub modules, ~5.9k lines of harness shipped as API), retire or fold the second CUDA runtime (CudaDevice), move shared CUDA/HIP device helpers to ojas-device"
status: backlog
priority: 3
severity: medium
type: refactor
owner: "unassigned"
due: "none"
labels:
  - "cuda"
  - "hip"
  - "refactor"
  - "api"
  - "dedup"
repositories:
  - "ojas"
planned_files:
  - "ojas-cuda/src/lib.rs"
  - "ojas-cuda/src/wait.rs"
  - "ojas-cuda/src/inputs.rs"
  - "ojas-hip/src/lib.rs"
  - "ojas-device/src/lib.rs"
acceptance_criteria:
  - "ojas-cuda's harness modules move behind a non-default feature, a dev-only crate, or the bins' own module tree; the public API is the backend, the step provider and what the C ABI needs; the docs list what stays public"
  - "CudaDevice is either retired (callers moved to CudaRuntime after a two-signal caller check) or delegates to CudaRuntime for its context, stream, budget, wait bound and error mapping; docs/backends.md:74 is updated"
  - "poll_until and the free-memory fit check live once in ojas-device and CUDA and HIP call them; the HIP timeout message names the real deadline"
  - "One SplitMix implementation (ojas-kernels) serves ojas-cuda's inputs, ojas-autograd's tiny path and ojas-infer's sampler, with bit-identical outputs pinned by the existing tests"
  - "CUDA-C edits are compile-checked on the Mac; device behaviour is re-run on the GH200 before this is marked done"
  - "CudaDevice's affine slot cannot block forever after a timed-out stream wait: resizing replaces gpu.host (ojas-cuda/src/lib.rs:369-382, wait at :415) and cudarc's PinnedHostSlice::drop synchronizes its event (core.rs:1468-1473); the slot is poisoned or dropped only after the stream is known idle"
---

# Task brief v1

## Title
Narrow ojas-cuda's public surface (57 pub modules, ~5.9k lines of harness shipped as API), retire or fold the second CUDA runtime (CudaDevice), move shared CUDA/HIP device helpers to ojas-device

Task: gp-cuda-surface-and-device-helpers
Type: refactor
Status: backlog
Priority: 3 (Low)
Severity: medium
Owner: unassigned
Due: none
Labels: cuda, hip, refactor, api, dedup

## Repositories
- ojas

## Description
Filed by the 2026-10-09 task audit (at d431949).

- **Harness shipped as library API [A]:** ojas-cuda/src/lib.rs:43-117 exposes 57 public modules. About 5.9k of the crate's 23.4k source lines are smoke tests, goldens, CLI code, npy/json helpers and fixtures (report_cli, rung0_cli, *_smoke, k11_golden, tiny_fixture_published, npy, json, inputs, libprobe, host_ref). Their only consumers are src/bin/rung0.rs, src/bin/runga.rs and the tests.
- **A second CUDA runtime [A]:** CudaDevice (lib.rs:146-430) has its own context and default stream, allocations charged to no Budget, a hard-coded 30 s wait instead of sync_timeout, and its own driver-code mapping (cuda_status: everything except OOM is Launch) that contradicts the single mapping in error.rs. Its callers are its own tests and open(). Chesterton check: docs/backends.md:74 describes it as the device-probe API, so retire or fold it deliberately.
- **CUDA/HIP copies [A]:**
  - poll_until exists in ojas-cuda/src/wait.rs:15 and ojas-hip/src/lib.rs:149; the HIP copy always says 'within 30s' whatever deadline it was given.
  - device_bytes_fit (ojas-cuda/src/lib.rs:247) and hip_free_covers (ojas-hip/src/lib.rs:137) are the same check.
- **SplitMix copies [A]:**
  - ojas-cuda/src/inputs.rs:11-30 copies ojas_kernels::splitmix_f32 (harness.rs:32), although ojas-cuda already depends on ojas-kernels.
  - Other SplitMix64 copies in shipped code: ojas-autograd/src/tiny.rs:372 and ojas-infer/src/sample.rs:28.

Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- New: resizing the CudaDevice affine buffers after a timed-out wait can block forever. This is a bounded-wait issue, not a use-after-free [A].

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Lowered to P3: off the Mac (Metal) training path. Raise it again when CUDA, HIP or wgpu training becomes a goal.

## Acceptance criteria
- [ ] ojas-cuda's harness modules move behind a non-default feature, a dev-only crate, or the bins' own module tree; the public API is the backend, the step provider and what the C ABI needs; the docs list what stays public
- [ ] CudaDevice is either retired (callers moved to CudaRuntime after a two-signal caller check) or delegates to CudaRuntime for its context, stream, budget, wait bound and error mapping; docs/backends.md:74 is updated
- [ ] poll_until and the free-memory fit check live once in ojas-device and CUDA and HIP call them; the HIP timeout message names the real deadline
- [ ] One SplitMix implementation (ojas-kernels) serves ojas-cuda's inputs, ojas-autograd's tiny path and ojas-infer's sampler, with bit-identical outputs pinned by the existing tests
- [ ] CUDA-C edits are compile-checked on the Mac; device behaviour is re-run on the GH200 before this is marked done
- [ ] CudaDevice's affine slot cannot block forever after a timed-out stream wait: resizing replaces gpu.host (ojas-cuda/src/lib.rs:369-382, wait at :415) and cudarc's PinnedHostSlice::drop synchronizes its event (core.rs:1468-1473); the slot is poisoned or dropped only after the stream is known idle

## Planned files
- ojas-cuda/src/lib.rs
- ojas-cuda/src/wait.rs
- ojas-cuda/src/inputs.rs
- ojas-hip/src/lib.rs
- ojas-device/src/lib.rs
