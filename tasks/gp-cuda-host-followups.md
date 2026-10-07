---
id: "gp-cuda-host-followups"
title: "CUDA backend host side: one memory accounting (Budget vs AllocBudget), typed open errors, upload without memset and extra copies, one arch constant, C ABI selection"
status: backlog
priority: 2
severity: medium
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
  - "A failing test first: backend tensors plus kernel buffers can together exceed the cap today, because CudaBackend::upload charges only ojas_core::Budget (backend.rs:136) and allocates through cudarc directly, bypassing CudaRuntime::reserve and its AllocBudget/device-free check (runtime.rs:523-542, buffer.rs:179-185), while both budgets are sized budget.cap_bytes() (backend.rs:49-52). After the fix one accounting owns every device byte"
  - "CudaBackend::open stops mapping every failure to DeviceError::NoDevice (backend.rs:45-56): wrong compute capability, budget/capacity and cuBLAS/NVRTC init failures get distinct kinds, and a capacity failure reaches C/Go as E_CAPACITY (ojas-capi/src/lib.rs:138-145 kinds only DeviceError::Capacity)"
  - "Uploads skip the alloc_zeros memset before memcpy_htod (backend.rs:137-145; CudaRuntime::upload buffer.rs:197-199) and the to_ne_bytes staging copy where the source is already contiguous host f32/u32; the U32 shadow copy (backend.rs:146-148) is kept only where the token-range check needs it"
  - "REQUIRED_CC (runtime.rs:37) and the compute_90 / compute_90a literals (kernels.rs:61, :72) derive from one constant; the exact-match 90a gate stays exact"
  - "CudaDevice::launch_affine (ojas-cuda/src/lib.rs:~367-401) has no non-test unwrap and every unsafe launch carries a // SAFETY: comment; the other launch sites flagged by the audit (gemm.rs:168, gdn.rs:334/357, k0.rs) are read and given one where missing"
  - "CUDA is selectable from the C ABI and Go (DeviceKind today is Cpu/Metal/Wgpu, session.rs:167-175; go/ffi.go:66-70) through an owner thread like Metal's, since CudaBackend holds an Rc and is not Send (backend.rs:21-24); or the refusal is explicit and documented until gp-cuda-backend-provider lands"
  - "All CUDA-C/kernel edits are compile-checked only on the Mac (NVRTC never runs here); device tests run on the GH200 before any criterion is marked done"
---

# Task brief v1

## Title
CUDA backend host side: one memory accounting (Budget vs AllocBudget), typed open errors, upload without memset and extra copies, one arch constant, C ABI selection

Task: gp-cuda-host-followups
Type: bug
Status: backlog
Priority: 2 (Normal)
Severity: medium
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
