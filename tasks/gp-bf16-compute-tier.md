---
id: "gp-bf16-compute-tier"
title: "Implement BF16 Operand / FP32 Accumulation Compute Tier and Muon NS5 Path"
status: review
priority: 0
severity: critical
type: feature
owner: "unassigned"
due: "none"
labels:
  - "kernel"
  - "precision"
  - "performance"
  - "memory"
  - "optimizer"
  - "muon"
  - "parity"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/backend.rs"
  - "ojas-core/src/dtype.rs"
  - "ojas-core/src/autocast.rs"
  - "ojas-cpu/src/backend.rs"
  - "ojas-cpu/src/optim.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-metal/kernels/ojas_backend.metal"
  - "ojas-wgpu/src/backend.rs"
acceptance_criteria:
  - "Add BF16 tensor storage and kernel operands across Backend trait methods"
  - "Implement BF16 GEMM with FP32 accumulation on CPU (Accelerate / NEON), Metal (Apple matrix units), and wgpu"
  - "Implement BF16 attention forward and backward kernels with numerical stability guards"
  - "Verify that batch size B >= 8 micro-batches fit in Apple Silicon unified memory without paging or thrashing"
  - "Parity and loss divergence tests pass against FP32 reference, with a stated divergence bound (not only finite-and-different)"
  - "Add optional bf16 compute path for Newton-Schulz 5 quintic iterations in Muon optimizer"
  - "Align update magnitudes exactly with PyTorch and nanolab stock optimizer steps"
  - "Measure throughput improvement from BF16 batched matrix multiplications during orthogonalization"
  - "Golden trace parity test verifying convergence within 1e-4 nats against stock nanolab trace"
---

# Task brief v1

## Title
Implement BF16 Operand / FP32 Accumulation Compute Tier and Muon NS5 Path

Task: gp-bf16-compute-tier
Type: feature
Status: review
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: none
Labels: kernel, precision, performance, memory, optimizer, muon, parity

## Repositories
- ojas

## Description
Consolidated task combining the mixed-precision compute tier (`gp-bf16-compute-tier`) and the Muon BF16 Newton-Schulz optimizer path (`gp-muon-bf16-ns5`).

Ojas compute across backends is currently strictly f32. At realistic batch sizes (B >= 8), FP32 thrashes unified memory on Apple Silicon, and leaves Tensor Cores and Apple matrix units running at fractional FLOPS. Furthermore, upstream nanolab and PyTorch Muon execute the quintic Newton-Schulz iteration in bf16 (`X = G.bfloat16()`), causing Ojas f32 NS5 to drift up to 1.8e-4 nats over the first 5 steps compared to stock nanolab.

This task introduces a true BF16-operand / FP32-accumulation compute tier across the `Backend` trait (mixed-precision GEMM and attention), together with an optional BF16 Newton-Schulz 5 path in the Muon optimizer.

### Audit status and progress notes (2026-10-05)
Commit 1f26a4b landed **bf16 emulated in f32 storage**, not native BF16 compute:
- `Autocast<B>` (`ojas-core/src/autocast.rs`) rounds f32 values to bf16 precision, keeps them in f32 buffers, and tags them as rounded. The only new Backend op was `cast_bf16`, which takes f32 and returns f32.
- Linear and attention under autocast round inputs then invoke existing f32 kernels. Storage remains f32, saving no memory.
- Existing tests (`autocast_bf16_is_finite_and_moves_a_parameter`) only test finite variance; true convergence divergence bounds against FP32 reference remain open.
- The Muon NS5 iteration in `ojas-cpu/src/optim.rs` and `ojas-metal/src/device.rs` executes strictly in f32.

### Progress notes (2026-10-07): Muon bf16 NS5 slice done; storage tier not started
Done and verified (criteria 6-9; the session's commands and numbers are in its review summary):
- `ojas_core::Ns5Precision { F32 (default), Bf16 }` is a field of `MuonNs5Config` (`ns5`) and of `TrainConfig` (`muon_ns5`). F32 leaves `to_json` unchanged, so existing run ids hold.
- CPU (`ojas-cpu/src/optim.rs`) and Metal (`ojas-metal/src/device.rs` `fn muon`, kernels `ojas_axpby_bf16`, `ojas_ns_denom_bf16`, `ojas_axpby_fma`) run stock nanolab's bf16 iteration: every intermediate rounded to bf16, GEMMs on bf16 operands with f32 accumulation (Metal uses tessl's bf16 TensorOps lane when the device has it). wgpu refuses Bf16 with `Unsupported` before any charge or write.
- Update magnitudes: torch's `add(.., alpha=..)` is one fused multiply-add, so the Nesterov blend and the parameter update are `mul_add` on CPU (both precisions) and `fma` on Metal. The four Exact `muon_p` digests in `ojas-cpu/tests/exact_golden.rs` were re-recorded for that.
- New fixture `ojas-oracle/fixtures/tiny/muon_step_bf16.safetensors` (`python/muon_step.py`, regen deterministic): one stock nanolab `Muon.step` per case. CPU matches it bit for bit on 4 cases under Exact and Fast.
- Golden trace: `trace_parity_ns5(Ns5::Bf16)` against `trace_ns5_bf16` gives worst |dloss| 2.38e-5 over 5 steps (gate 1e-4; the f32 NS5 baseline was 1.81e-4 and failed). The 40-step bf16 curve passes (worst 1.33e-3, early steps within 2e-4). The bf16 trace's parameter gate is 5e-2, not 1e-4, because stock bf16 NS5 is discontinuous in its input: in torch alone a 1e-6 relative input change moves it by 1.9e-2 (`TRACE_BF16_PARAM_NORMWISE_REL_TOL`, pinned in `tests/parity_gates.rs`).
- Throughput (`docs/bench-gpu-vs-torch.md`, 2026-10-07): Metal bf16 NS5 is 1.35-1.47x faster than f32 NS5 and 0.94-1.07x torch MPS's own bf16 Muon. CPU bf16 is 1.14-1.40x slower than f32: Accelerate has no bf16 GEMM.

Not done (criteria 1-5): bf16 tensor storage and operands across `Backend`, bf16 GEMM on CPU/Metal/wgpu as a general op, bf16 attention forward and backward, the B >= 8 unified-memory study, and the bounded loss-divergence test against an FP32 baseline. Open design question for the user: does `DType::Bf16` become a kernel input on every op, or on matmul-class ops only (with norms, RoPE and the residual stream staying f32, as `Autocast` does today)? Also not done: `muon_ns5` in the C API / Go config tag table, a wgpu bf16 NS5, and a batched (stacked same-shape) Muon call.

### Execution plan
1. Introduce BF16 tensor storage representation and operands to `Backend` trait methods.
2. Implement BF16 GEMM with FP32 accumulation on CPU (Accelerate / NEON), Metal (Apple matrix units), and wgpu.
3. Implement BF16 attention forward and backward kernels with numerical stability guards.
4. Implement optional BF16 Newton-Schulz 5 quintic iterations for Muon, measuring throughput speedup and golden trace parity within 1e-4 nats against nanolab.
5. Establish bounded loss divergence limits against FP32 baselines.

## Acceptance criteria
- [ ] Add BF16 tensor storage and kernel operands across Backend trait methods
- [ ] Implement BF16 GEMM with FP32 accumulation on CPU (Accelerate / NEON), Metal (Apple matrix units), and wgpu
- [ ] Implement BF16 attention forward and backward kernels with numerical stability guards
- [ ] Verify that batch size B >= 8 micro-batches fit in Apple Silicon unified memory without paging or thrashing
- [ ] Parity and loss divergence tests pass against FP32 reference, with a stated divergence bound (not only finite-and-different)
- [ ] Add optional bf16 compute path for Newton-Schulz 5 quintic iterations in Muon optimizer
- [ ] Align update magnitudes exactly with PyTorch and nanolab stock optimizer steps
- [ ] Measure throughput improvement from BF16 batched matrix multiplications during orthogonalization
- [ ] Golden trace parity test verifying convergence within 1e-4 nats against stock nanolab trace

## Planned files
- ojas-core/src/backend.rs
- ojas-core/src/dtype.rs
- ojas-core/src/autocast.rs
- ojas-cpu/src/backend.rs
- ojas-cpu/src/optim.rs
- ojas-metal/src/device.rs
- ojas-metal/kernels/ojas_backend.metal
- ojas-wgpu/src/backend.rs
