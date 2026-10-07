---
id: "gp-gpu-step-throughput"
title: "GPU step throughput: wgpu AdamW check/apply, fused clip_grad_norm, residual-add backward aliasing, wgpu attention, wgpu per-dispatch cost, multi-tensor optimizers, open Metal rows"
status: ready
priority: 1
severity: medium
type: perf
owner: "unassigned"
due: "none"
labels:
  - "gpu"
  - "wgpu"
  - "metal"
  - "performance"
  - "optimizer"
  - "attention"
repositories:
  - "ojas"
planned_files:
  - "ojas-wgpu/src/backend.rs"
  - "ojas-wgpu/src/context.rs"
  - "ojas-kernels/src/wgsl/optim.wgsl"
  - "ojas-kernels/src/wgsl/attention.wgsl"
  - "ojas-kernels/src/geometry.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-metal/src/link.rs"
  - "ojas-core/src/backend.rs"
  - "bench/ojas_rows.rs"
  - "docs/bench-gpu-vs-torch.md"
acceptance_criteria:
  - "wgpu AdamW uses Metal's check-then-apply design (no per-step copies of p, m and v into scratch); first-fault rollback semantics unchanged and covered by the existing fault tests; adamw_full re-benched"
  - "clip_grad_norm on Metal and wgpu computes the global norm in one fused multi-tensor pass (max and sum of squares together) with the finite check folded into scale; dispatch count per call recorded before and after"
  - "residual_add_backward: a recorded decision on whether the GPU backends return shared clones of grad_output instead of two copies, with the accumulate_grad sole-owner consequence measured; implemented if it wins"
  - "wgpu attention re-benched on the current kernel (LSE return, native GQA) before any rewrite; if still under 0.5x torch-mps, a plan for subgroup/tiled WGSL attention is recorded with the measured target"
  - "The open Metal rows below torch (sdpa fwd d64, accumulate_grad, cross_entropy_bwd, block_fwd) and wgpu rows (block_fwd, gate_fwd/bwd, cross_entropy_bwd) each get a profile and either a fix or a written reason"
  - "wgpu FLUSH_AT (64) is A/B measured against at least two alternatives the way Metal's 256-dispatch overlap commit was; the chosen value cites the run"
  - "Metal Link::call per-op round-trip (~7 µs host) is measured on a decode-sized workload; fire-and-forget encoding with host-assigned ids is implemented or rejected with numbers"
  - "Every claimed speedup has a committed interleaved A/B benchmark under bench/results/ (min-of-N, spread per row)"
  - "wgpu's fixed cost per dispatch is cut and counted: each dispatch creates a fresh 64 B mapped uniform buffer (params, ojas-wgpu/src/context.rs:996-1002, via mapped() with an OOM error scope and a blocking pollster pop at :566-578) and a fresh bind group with a second blocking pop (:749-759); pool misses in alloc add a third (:498-510). Uniforms move to a ring or push constants, bind groups are cached by buffer identity, and error scopes are batched per flush; blocking pops per dispatch counted before and after"
  - "wgpu read() reuses MAP_READ staging buffers instead of creating one (plus an error scope) per read (context.rs:798-820); every sync, download and clip_grad_norm status read pays it today"
  - "wgpu binds offset views in place (binding offset or an offset parameter word) instead of copying any tensor with byte_offset != 0 into scratch on every op (ojas-wgpu/src/backend.rs:331-347, :377); the copy count on a fused-QKV slice is recorded before and after"
  - "Uploads on Metal and wgpu stop making a transient full host copy outside the host budget: to_ne_bytes() then a second copy into the mapped/device view (wgpu backend.rs:1443 then context.rs:585; Metal backend.rs:883 before the reservation, then device.rs:1117)"
  - "Optimizer steps can run multi-tensor: adamw_step and muon_ns5_step take one parameter or matrix per call (ojas-core/src/backend.rs ~976-993; trainer loop at ojas-model/src/trainer.rs:909-912); a batched AdamW over all parameters and batched Muon GEMMs (docs/pytorch-parity-plan.md:49, :128; bench-gpu-vs-torch.md:1246, :1631-1632) are implemented on Metal and wgpu or rejected with numbers"
  - "Metal fused linear cross-entropy computes each logits tile once on the gradient path: pass 2 remakes every tile unless there is a single vocabulary tile (ojas-metal/src/device.rs:2196-2203; default cols 4096 gives 13 tiles at V=50304), while torch does one forward GEMM (bench-gpu-vs-torch.md:1256); keep the tile or explain with a profile"
  - "Metal embedding backward stops ranking tokens in O(N^2): ojas_embed_place (ojas-metal/kernels/ojas_backend.metal:661-692) compares each token with every earlier one, while the host already holds the ids window (backend.rs:265-295) and wgpu sorts on the host (wgpu backend.rs:1511-1522); embed_bwd gets a bench row first"
  - "The wgpu Muon f32 row (0.65-0.71x torch fp32, bench/results/2026-10-07-muon-bf16/summary.md) is profiled and fixed or explained; wgpu's bf16 NS5 refusal (ojas-wgpu/src/backend.rs:2262) stays with gp-bf16-compute-tier"
  - "Metal per_head_sigmoid_gate_backward's ten standalone finite checks are folded into the gate kernels only if the metal_bench gate profile shows them at roughly 10-15% or more of the row (user decision 2026-10-02); residual_add_backward keeps its x/y checks on every backend (also a user decision)"
---

# Task brief v1

## Title
GPU step throughput: wgpu AdamW check/apply, fused clip_grad_norm, residual-add backward aliasing, wgpu attention, wgpu per-dispatch cost, multi-tensor optimizers, open Metal rows

Task: gp-gpu-step-throughput
Type: perf
Status: ready
Priority: 1 (High)
Severity: medium
Labels: gpu, wgpu, metal, performance, optimizer, attention

## Repositories
- ojas

## Description
Gap audit of 2026-10-07 (read-only, rg + Read; DevMap store unavailable that session). Follow-up to `gp-gpu-runtime-hardening`, which covers fault words, typed device-lost and trainer upload waits, not kernel throughput. Ratios below are from round 5 (`bench/results/2026-10-02-r5/summary.md`), which was flagged noisy, so every item starts with a quiet re-measure. V = verified in code or data, I = inferred.

1. **wgpu AdamW copies three tensors per step [V].** `ojas-wgpu/src/backend.rs:2216-2222` allocates `old_p/old_m/old_v` scratch and copies all of p, m, v before `ADAM_UPDATE`, then `ADAM_COMMIT` restores on fault. That is 3x parameter bytes of extra traffic and scratch. Metal uses a check kernel then an apply kernel with no copies (`ojas_adamw_check/apply`). r5 `adamw_full`: wgpu 0.71x torch-mps, Metal 1.53x (summary.md:61,114).
2. **clip_grad_norm is ~0.45x torch 2.15 nightly [V ratio, I cause]** (`docs/bench-gpu-vs-torch.md:80`). On Metal each gradient gets a finite check plus two reductions, each partial + final (`ojas-metal/src/device.rs:2121-2163`), so roughly 5 dispatches per tensor, ~850 for 170 tensors, plus a scale pass.
3. **residual_add_backward writes two full copies [V].** Metal `backend.rs:1396-1406` (`Cmd::AddBwd`), wgpu `backend.rs:1974-1987`. r5: 0.03x (Metal) / 0.02x (wgpu) of torch, which aliases. The trait doc does not require two owned outputs, but `accumulate_grad` adds in place only for a sole owner, so aliasing may move the copy; measure both.
4. **wgpu attention 0.09-0.22x torch-mps [V data]** (summary.md:86-91). Plain WGSL, no subgroups or matrix units, `ATTENTION_PARTS` = 4 threads per row (`attention.wgsl:1-2`). The co-op-matrix spike in `gp-backend-architecture-decisions` covers GEMM only.
5. **Open Metal/wgpu rows below torch [V]** listed as open in `docs/bench-gpu-vs-torch.md:131-142,286`.
6. **`FLUSH_AT = 64` never measured [V]** (`ojas-wgpu/src/context.rs:42`); Metal's 256 was A/B'd (`docs/metal-deferred-faults.md` §9.3).
7. **Metal per-op channel round-trip [I]** (`ojas-metal/src/link.rs:1-12`; ~7 µs host per call, `bench-gpu-vs-torch.md:109`). Matters most for decode and small ops.

Follow the mac resource limits: queue heavy runs, -j 2, no parallel GPU benches.

**Added by the second gap audit 2026-10-07** ([V] re-read by the auditor; [A] read by an audit subagent, not re-read):
8. **wgpu blocking error scopes per dispatch [V].** `push_error_scope` + `pollster::block_on(scope.pop())` at context.rs:501/508 (alloc), :569/576 (mapped, used by `params` at :996), :751/757 (bind group), :811/818 (read staging). Item 6's FLUSH_AT tuning cannot remove these.
9. **Offset views copied per op [V]** (backend.rs:333, :377 `if t.byte_offset() != 0`). How often it fires in a training step is [I].
10. **Metal linear CE remakes tiles [V]:** the doc at device.rs:2196-2203 says pass 2 "makes each tile again" unless there is one vocabulary tile.
11. **Metal embed rank is quadratic [V]:** the `ojas_embed_place` loop at ojas_backend.metal:677-688 walks every earlier tile. Its real cost is [I] until measured.
12. **Optimizers per tensor, upload copies, wgpu Muon row [A].**
13. Memory-lifecycle items (wgpu OOM retry, pool caps, Queue::drop) went to gp-device-memory-probes; the tape loss-seed scaling went to gp-cpu-autograd-hot-paths.
