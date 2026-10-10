---
id: "gp-cuda-host-followups"
title: "CUDA host side: bounded waits are not bounded on pageable copies, a timed-out copy can outlive its host buffer, an uncharged U32 shadow per upload, whole-stream drain per upload, device 0 only"
status: ready
priority: 3
severity: high
type: bug
owner: "unassigned"
due: "none"
labels:
  - "cuda"
  - "budget"
  - "errors"
  - "capi"
  - "performance"
repositories:
  - "ojas"
planned_files:
  - "ojas-cuda/src/backend.rs"
  - "ojas-cuda/src/runtime.rs"
  - "ojas-cuda/src/buffer.rs"
  - "ojas-cuda/src/kernels.rs"
  - "ojas-cuda/src/lib.rs"
  - "ojas-capi/src/session.rs"
  - "ojas-capi/src/owner.rs"
  - "go/ffi.go"
acceptance_criteria:
  - "One accounting owns every device byte (done in code by bcfb182: CudaRuntime::open_with shares the backend Budget, AllocBudget is a view of it, the cuBLAS workspace is charged); the #[ignore] device test backend_tensors_and_kernel_buffers_share_one_cap (tests/device_backend.rs:26) passes on the GH200"
  - "CudaBackend::open stops mapping every failure to DeviceError::NoDevice (backend.rs:45-56): wrong compute capability, budget/capacity and cuBLAS/NVRTC init failures get distinct kinds, and a capacity failure reaches C/Go as E_CAPACITY (ojas-capi/src/lib.rs:138-145 kinds only DeviceError::Capacity)"
  - "Uploads skip the memset and staging copy (done: clone_htod, runtime.rs:595-607; staging only for F16, backend.rs:144-153). Open: every U32 upload keeps an Arc<[u32]> host shadow (backend.rs:166-170) that nothing reads (buffer.rs:248-252 says so) and no budget charges; it is created only when an op reads it, or charged"
  - "REQUIRED_CC (runtime.rs:37) and the compute_90 / compute_90a literals (kernels.rs:61, :72) derive from one constant; the exact-match 90a gate stays exact"
  - "CudaDevice::launch_affine (ojas-cuda/src/lib.rs:~367-401) has no non-test unwrap and every unsafe launch carries a // SAFETY: comment; the other launch sites flagged by the audit (gemm.rs:168, gdn.rs:334/357, k0.rs) are read and given one where missing"
  - "CUDA is selectable from the C ABI and Go (DeviceKind today is Cpu/Metal/Wgpu, session.rs:167-175; go/ffi.go:66-70) through an owner thread like Metal's, since CudaBackend holds an Rc and is not Send (backend.rs:21-24); or the refusal is explicit and documented until gp-cuda-backend-provider lands"
  - "All CUDA-C/kernel edits are compile-checked only on the Mac (NVRTC never runs here); device tests run on the GH200 before any criterion is marked done"
  - "Every CUDA wait is actually bounded: cudarc 0.19.10's HostSlice for [T]/Vec is pageable (SyncOnDrop::Sync(None)) and its copies are raw cuMemcpy*Async_v2, so read_bytes (buffer.rs:285-308, no prior sync, unlike download at :230-238) and copy_in (runtime.rs:595-618) can block inside the driver behind a hung kernel before wait_stream's timeout applies. Copies go through pinned staging (then the timeout really bounds them, and the host buffer must outlive the copy, which is tested), or the real bound is documented; proven on the GH200 with a deliberately stalled kernel"
  - "An upload waits for its own copy, not the whole stream: htod_done drains all queued work on the single stream (runtime.rs:616-618, one stream at :425). An event per copy on the same in-order stream does not help; pinned async staging or a dedicated copy stream does, with the effect on a step measured"
  - "CudaBackend::open takes a device ordinal instead of always using device 0 (backend.rs:63, RuntimeConfig::default())"
  - "A Timeout poisons the backend (later ops refuse with DeviceLost or a typed timeout error) instead of leaving it usable with work possibly still in flight"
  - "Drift between each CUDA kernel and its hand-written host mirror (gdn_host, k11_host, ...) is caught somewhere other than the 48 #[ignore] sm_90 tests: the mirrors are generated from or checked against the kernel source, or the GH200 run of those tests is scheduled and recorded per change to a kernel"
---

# Task brief v1

## Title
CUDA host side: bounded waits are not bounded on pageable copies, a timed-out copy can outlive its host buffer, an uncharged U32 shadow per upload, whole-stream drain per upload, device 0 only

Task: gp-cuda-host-followups
Type: bug
Status: ready
Priority: 3 (Low)
Severity: high
Owner: unassigned
Due: none
Labels: cuda, budget, errors, capi, performance

## Repositories
- ojas

## Description
Second gap audit 2026-10-07. Labels: [V] re-read by the auditor; [A] read by an audit subagent, not re-read.

Filed separately because gp-cuda-backend-provider (ft-7162) is in review; its definition of done should not change under it. These are host-side `CudaBackend` (the `ojas_core::Backend` impl) issues, not the Qwen35Step provider.

- **Two accountings [V]:** upload reserves from `self.budget` then calls `self.rt.stream().alloc_zeros::<u8>(byte_len)` directly (backend.rs:136-141), so `CudaRuntime`'s own `AllocBudget` never sees backend tensors.
- **Open errors collapse [V]:** both `probe_libraries()` and `CudaRuntime::open(cfg)` errors become `DeviceError::NoDevice` (backend.rs:45-56).
- **Upload copies [V]:** `to_ne_bytes()` (backend.rs:128), `alloc_zeros` then `memcpy_htod` (:137-145), plus an `Arc<[u32]>` shadow for U32 (:146-148).
- **Arch stated twice [V]:** `REQUIRED_CC: (i32, i32) = (9, 0)` (runtime.rs:37) and `arch: "compute_90"` / `"compute_90a"` (kernels.rs:61, :72).
- **Unsafe/unwrap [V]:** `slot.as_mut().unwrap()` in `launch_affine` and an `unsafe {` launch block with no SAFETY line near lib.rs:401.
- **Not selectable [V]:** `pub enum DeviceKind { Cpu { threads }, Metal, Wgpu }` (ojas-capi/src/session.rs:167).

### Re-audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- bcfb182 ('one budget, typed open errors, bounded waits; refuse device 5', merged 38979b8) closed most of the original list in code [A]:
  - **Typed open errors:** From<CudaError> for DeviceError maps to NoDevice / Unsupported / Capacity / Compile / Init (error.rs:248-269). The capacity error reaches C as E_CAPACITY (test at ojas-capi/src/tests.rs:2214).
  - **One arch constant:** REQUIRED_CC is defined once at kernels.rs:37, and the NVRTC arch is derived from it at :84-104.
  - **launch_affine:** it has no unwrap, and every unsafe site in ojas-cuda/src has a SAFETY comment.
  - **C ABI selection:** device 5 is refused explicitly (load.rs:41-56, 294; Go DeviceCUDA).
- Not run: nothing here has run on a GPU. Criteria 1 and 7 stay open until the GH200 run [U].
- New high-severity items: pageable copies defeat the bounded-wait claim, and a timed-out copy can outlive its host buffer. Both are inferred from NVIDIA's documented cudaMemcpyAsync semantics for pageable memory and were not reproduced [A].
- CUDA OOM classification moved to gp-oom-error-class. The second CUDA runtime (CudaDevice in lib.rs) moved to gp-cuda-surface-and-device-helpers.

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- Corrected: the pageable-copy hang and the timed-out-copy use-after-free are alternatives, not two bugs. If the driver stages a pageable copy, the call returns only after the source is consumed. They are merged into criterion 8 [A, cudarc source read].
- The proposed 'event per copy' fix was WRONG: on a single in-order stream it waits for the same work. Criterion 10 is rewritten [A].
- New: nothing marks the backend unusable after a Timeout. CI can't see a CUDA kernel drifting from its host mirror [A].

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Lowered to P3: off the Mac (Metal) training path. Raise it again when CUDA, HIP or wgpu training becomes a goal.

## Acceptance criteria
- [ ] One accounting owns every device byte (done in code by bcfb182: CudaRuntime::open_with shares the backend Budget, AllocBudget is a view of it, the cuBLAS workspace is charged); the #[ignore] device test backend_tensors_and_kernel_buffers_share_one_cap (tests/device_backend.rs:26) passes on the GH200
- [x] CudaBackend::open stops mapping every failure to DeviceError::NoDevice (backend.rs:45-56): wrong compute capability, budget/capacity and cuBLAS/NVRTC init failures get distinct kinds, and a capacity failure reaches C/Go as E_CAPACITY (ojas-capi/src/lib.rs:138-145 kinds only DeviceError::Capacity)
- [ ] Uploads skip the memset and staging copy (done: clone_htod, runtime.rs:595-607; staging only for F16, backend.rs:144-153). Open: every U32 upload keeps an Arc<[u32]> host shadow (backend.rs:166-170) that nothing reads (buffer.rs:248-252 says so) and no budget charges; it is created only when an op reads it, or charged
- [x] REQUIRED_CC (runtime.rs:37) and the compute_90 / compute_90a literals (kernels.rs:61, :72) derive from one constant; the exact-match 90a gate stays exact
- [x] CudaDevice::launch_affine (ojas-cuda/src/lib.rs:~367-401) has no non-test unwrap and every unsafe launch carries a // SAFETY: comment; the other launch sites flagged by the audit (gemm.rs:168, gdn.rs:334/357, k0.rs) are read and given one where missing
- [x] CUDA is selectable from the C ABI and Go (DeviceKind today is Cpu/Metal/Wgpu, session.rs:167-175; go/ffi.go:66-70) through an owner thread like Metal's, since CudaBackend holds an Rc and is not Send (backend.rs:21-24); or the refusal is explicit and documented until gp-cuda-backend-provider lands
- [ ] All CUDA-C/kernel edits are compile-checked only on the Mac (NVRTC never runs here); device tests run on the GH200 before any criterion is marked done
- [ ] Every CUDA wait is actually bounded: cudarc 0.19.10's HostSlice for [T]/Vec is pageable (SyncOnDrop::Sync(None)) and its copies are raw cuMemcpy*Async_v2, so read_bytes (buffer.rs:285-308, no prior sync, unlike download at :230-238) and copy_in (runtime.rs:595-618) can block inside the driver behind a hung kernel before wait_stream's timeout applies. Copies go through pinned staging (then the timeout really bounds them, and the host buffer must outlive the copy, which is tested), or the real bound is documented; proven on the GH200 with a deliberately stalled kernel
- [ ] An upload waits for its own copy, not the whole stream: htod_done drains all queued work on the single stream (runtime.rs:616-618, one stream at :425). An event per copy on the same in-order stream does not help; pinned async staging or a dedicated copy stream does, with the effect on a step measured
- [ ] CudaBackend::open takes a device ordinal instead of always using device 0 (backend.rs:63, RuntimeConfig::default())
- [ ] A Timeout poisons the backend (later ops refuse with DeviceLost or a typed timeout error) instead of leaving it usable with work possibly still in flight
- [ ] Drift between each CUDA kernel and its hand-written host mirror (gdn_host, k11_host, ...) is caught somewhere other than the 48 #[ignore] sm_90 tests: the mirrors are generated from or checked against the kernel source, or the GH200 run of those tests is scheduled and recorded per change to a kernel

## Planned files
- ojas-cuda/src/backend.rs
- ojas-cuda/src/runtime.rs
- ojas-cuda/src/buffer.rs
- ojas-cuda/src/kernels.rs
- ojas-cuda/src/lib.rs
- ojas-capi/src/session.rs
- ojas-capi/src/owner.rs
- go/ffi.go
