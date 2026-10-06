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
  - "ojas-cuda/src/backend.rs"
  - "ojas-cuda/src/step.rs"
  - "ojas-qwen35/src/cuda.rs"
acceptance_criteria:
  - "cudarc pinned to cuda-12080 to match the GH200 driver environment (already met before 1f26a4b)"
  - "Implement Qwen35Cuda step provider mirroring the Metal ojas-qwen35 API: forward, backward and adamw_step launch real kernels instead of the current stub"
  - "Provide NVRTC-compiled kernel pipelines for GEMM, attention (none exists yet), and in-place AdamW, wired into the step and into CudaBackend's Backend ops"
  - "Eliminate dead helper code (commit_resize) and replace mock tests with real GPU tests under ojas-cuda/tests"
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
Lappi GH200 training campaigns need a CUDA path for the whole Qwen3.5 training step. Following docs/cuda-backend-scoping.md, this task implements the whole-step Qwen3.5 training provider on CUDA and wires real NVRTC compute kernels into it.

### Progress (audit of 9668bfa, 2026-10-05, code inspection only; tests not executed)
1f26a4b's message says it added a "CUDA backend". What is actually there:
- **The pin premise is stale.** `ojas-cuda/Cargo.toml` was already `cuda-12080` before 1f26a4b. Criterion 1 is met.
- **`CudaBackend` is a shell.** `impl Backend for CudaBackend` (`ojas-cuda/src/backend.rs:73`) has working `upload`/`download`/`sync` only. All 28 compute ops return `Unsupported` "does not yet implement ..." (`backend.rs:179-486`). It also takes the refusing defaults for `linear_cross_entropy_mean`, `cached_attention_forward`, `kv_cache_write`, `optimizer_scratch_bytes` and `accumulate_grad`.
- **`Qwen35Step` is a stub.** `forward` returns a zero `hidden` tensor; `backward` only validates and bumps `BankState`; `adamw_step` only validates hyperparameters and increments counters. Nothing on the step path launches a kernel (`ojas-cuda/src/step.rs:366-469`).
- **Kernels exist, unwired:** GEMM (`gemm.rs`), AdamW (`k11_kernels.rs`), GDN, conv1d, CE rows, RMSNorm, embed. There is no attention/SDPA kernel; the only softmax is in CE rows.
- **Dead code remains.** `commit_resize` is still in `ojas-cuda/src/lib.rs:228` with `#[allow(dead_code)]` and is called only from tests.
- **Tests:** `ojas-cuda/tests/` holds only `fixtures/`. The real device tests are still in the leftover `ojas-qwen35-cuda/` crate, which is not a workspace member. See gp-cuda-crate-consolidation, which should land first.

## Acceptance criteria
- [x] cudarc pinned to cuda-12080 to match the GH200 driver environment (already met before 1f26a4b)
- [ ] Implement Qwen35Cuda step provider mirroring the Metal ojas-qwen35 API: forward, backward and adamw_step launch real kernels instead of the current stub
- [ ] Provide NVRTC-compiled kernel pipelines for GEMM, attention (none exists yet), and in-place AdamW, wired into the step and into CudaBackend's Backend ops
- [ ] Eliminate dead helper code (commit_resize) and replace mock tests with real GPU tests under ojas-cuda/tests
- [ ] Verify execution on GH200 with zero symbol panics

## Planned files
- ojas-cuda/Cargo.toml
- ojas-cuda/src/lib.rs
- ojas-cuda/src/backend.rs
- ojas-cuda/src/step.rs
- ojas-qwen35/src/cuda.rs
