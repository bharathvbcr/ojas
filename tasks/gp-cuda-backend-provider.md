---
id: "gp-cuda-backend-provider"
title: "Implement Whole-Step Qwen3.5 CUDA Training Provider and Consolidate Crates"
status: review
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
  - "cleanup"
  - "tests"
repositories:
  - "ojas"
planned_files:
  - "ojas-cuda/Cargo.toml"
  - "ojas-cuda/src/lib.rs"
  - "ojas-cuda/src/backend.rs"
  - "ojas-cuda/src/step.rs"
  - "ojas-cuda/src/buffer.rs"
  - "ojas-cuda/src/runtime.rs"
  - "ojas-cuda/tests/"
  - "ojas-qwen35-cuda/"
  - "ojas-qwen35/src/cuda.rs"
  - ".github/workflows/test.yml"
acceptance_criteria:
  - "cudarc pinned to cuda-12080 to match the GH200 driver environment (verified met)"
  - "Every test file under ojas-qwen35-cuda/tests is moved to ojas-cuda/tests (or deliberately dropped, with recorded reason)"
  - "The buffer.rs, lib.rs and runtime.rs differences between the two crates are reconciled into ojas-cuda"
  - "Host-side CUDA tests run in CI, and device tests build (cargo test -p ojas-cuda --features cuda --no-run)"
  - "ojas-qwen35-cuda/ is removed from the tree, and no doc or Cargo.toml still refers to it"
  - "Implement Qwen35Cuda step provider mirroring the Metal ojas-qwen35 API: forward, backward and adamw_step launch real kernels instead of the current stub"
  - "Provide NVRTC-compiled kernel pipelines for GEMM, attention, and in-place AdamW, wired into the step and CudaBackend ops"
  - "Eliminate dead helper code (commit_resize) and replace mock tests with real GPU tests under ojas-cuda/tests"
  - "Verify execution on GH200 with zero symbol panics"
---

# Task brief v1

## Title
Implement Whole-Step Qwen3.5 CUDA Training Provider and Consolidate Crates

Task: gp-cuda-backend-provider
Type: feature
Status: review
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: cuda, gh200, backend, lappi, cleanup, tests

## Repositories
- ojas

## Description
Consolidated task combining crate consolidation (`gp-cuda-crate-consolidation`) and whole-step training provider execution (`gp-cuda-backend-provider`).

Lappi GH200 training campaigns need an end-to-end CUDA path for Qwen3.5 training. Following `docs/cuda-backend-scoping.md`, this task reconciles the standalone prototype crate, integrates its test coverage into `ojas-cuda`, and implements the whole-step Qwen3.5 training provider with real NVRTC compute kernels.

### Audit status and progress notes (2026-10-05)
1. **The cudarc pin premise is satisfied:** `ojas-cuda/Cargo.toml` is already pinned to `cuda-12080`.
2. **Phase 1: Crate consolidation:** Commit 1f26a4b copied `ojas-qwen35-cuda` sources into `ojas-cuda/src`, but left the old crate in place: 443 tracked files with their own standalone workspace and lock file, unbuilt in CI. The shared `src` files are byte-identical except `buffer.rs`, `lib.rs`, and `runtime.rs`. Approximately 278 tests (43 `#[ignore]`d sm_90 device tests in `tests/device_*.rs`, plus host-side `reference_*.rs`, `fixture_pins.rs`, `runtime_refusal.rs`, `fmt_boundary.rs`) still reside only in the leftover crate.
3. **Phase 2: Whole-step provider and kernels:** `impl Backend for CudaBackend` (`ojas-cuda/src/backend.rs:73`) has working `upload`/`download`/`sync` only; all 28 compute ops return `Unsupported`. `Qwen35Step` (`ojas-cuda/src/step.rs:366-469`) is currently a stub returning a zero `hidden` tensor. Kernels exist unwired for GEMM (`gemm.rs`), AdamW (`k11_kernels.rs`), GDN, conv1d, CE rows, RMSNorm, and embed. No attention/SDPA kernel is wired. `commit_resize` in `ojas-cuda/src/lib.rs:228` remains dead code called only from tests.

### Progress notes (2026-10-06/07, session 0002b6; board task ft-7162975649786a98c1efc52eb7fb908b rev 4)
Phase 1 and the host-verifiable gap carry-overs landed in one commit. Verified on the Mac: `cargo test -p ojas-cuda` (debug; release tier at full density), clippy `-D warnings` with and without `--features cuda`, `cargo test -p ojas-cuda --features cuda --no-run`, `sitegen -check`. Nothing ran on a GPU.
- Tests: all 38 non-fixture files of `ojas-qwen35-cuda/tests` moved to `ojas-cuda/tests` (crate name rewritten); the fixtures tree was byte-identical in both crates (`diff -rq`), so the old copy was dropped. `ojas-qwen35-cuda/` removed.
- Reconciled: `buffer.rs` and `runtime.rs` were supersets in ojas-cuda already; `lib.rs` regained the crate doc (current) and `#![deny(unsafe_op_in_unsafe_fn)]`. `commit_resize` and its test deleted (the resize path allocates all three buffers before assigning any).
- CI: the gpu-compile job runs clippy and `cargo test -p ojas-cuda --features cuda --no-run`; host tests run in the linux job's workspace test.
- Left referring to the old crate, deliberately: `docs/cuda-backend-scoping.md:566` (a verbatim ruling), `docs/adaptive-resources.md` and `bench/results/2026-10-03-lappi-inference/fable-ruling.md` (dated records), and `ojas-cuda/tests/fixtures/gen_goldens.py` + `goldens/manifest.json` (sha-pinned provenance of the goldens).
- `ojas-qwen35/src/cuda.rs` (planned file) does not exist; nothing was created there.
- Still open: the step provider launching kernels, attention / AdamW in `CudaBackend`, the GH200 run (zero symbol panics), the GDN timing numbers, the sm_90 before/after K10 bench.

### Execution plan
- **Phase 1 (Consolidation):** Reconcile `buffer.rs`, `lib.rs`, and `runtime.rs`. Move test suites to `ojas-cuda/tests/`. Wire host-side CUDA tests into CI. Remove `ojas-qwen35-cuda/`.
- **Phase 2 (Kernels & Step Provider):** Wire NVRTC kernels for GEMM, attention, and in-place AdamW into `CudaBackend` and `Qwen35Step`. Wire the step provider to match `ojas-qwen35` Metal semantics. Verify execution on GH200 without symbol panics.

## Acceptance criteria
- [x] cudarc pinned to cuda-12080 to match the GH200 driver environment (verified met)
- [x] Every test file under ojas-qwen35-cuda/tests is moved to ojas-cuda/tests (or deliberately dropped, with recorded reason)
- [x] The buffer.rs, lib.rs and runtime.rs differences between the two crates are reconciled into ojas-cuda
- [x] Host-side CUDA tests run in CI, and device tests build (cargo test -p ojas-cuda --features cuda --no-run)
- [ ] ojas-qwen35-cuda/ is removed from the tree, and no doc or Cargo.toml still refers to it
- [ ] Implement Qwen35Cuda step provider mirroring the Metal ojas-qwen35 API: forward, backward and adamw_step launch real kernels instead of the current stub
- [ ] Provide NVRTC-compiled kernel pipelines for GEMM, attention, and in-place AdamW, wired into the step and CudaBackend ops
- [ ] Eliminate dead helper code (commit_resize) and replace mock tests with real GPU tests under ojas-cuda/tests
- [ ] Verify execution on GH200 with zero symbol panics

## Planned files
- ojas-cuda/Cargo.toml
- ojas-cuda/src/lib.rs
- ojas-cuda/src/backend.rs
- ojas-cuda/src/step.rs
- ojas-cuda/src/buffer.rs
- ojas-cuda/src/runtime.rs
- ojas-cuda/tests/
- ojas-qwen35-cuda/
- ojas-qwen35/src/cuda.rs
- .github/workflows/test.yml
