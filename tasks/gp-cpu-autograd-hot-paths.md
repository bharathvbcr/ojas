---
id: "gp-cpu-autograd-hot-paths"
title: "CPU and tape hot paths: in-place gradient fan-in, parallel accumulate_grad, shape-general fast paths, loss-seed scaling, open CPU-vs-torch rows"
status: done
priority: 1
severity: medium
type: perf
owner: "unassigned"
due: "none"
labels:
  - "cpu"
  - "autograd"
  - "performance"
  - "simd"
  - "optimizer"
repositories:
  - "ojas"
planned_files:
  - "ojas-autograd/src/tape.rs"
  - "ojas-cpu/src/accum.rs"
  - "ojas-cpu/src/pointwise.rs"
  - "ojas-cpu/src/layout.rs"
  - "ojas-cpu/src/optim.rs"
  - "ojas-cpu/src/hybrid.rs"
  - "ojas-cpu/src/gdn.rs"
  - "ojas-cpu/src/pool/scoped.rs"
  - "ojas-cpu/tests/bench_ops.rs"
  - "docs/bench-cpu-vs-torch.md"
acceptance_criteria:
  - "Tape::acc accumulates fan-in gradients in place (accumulate_grad when the accumulator is solely owned) instead of residual_add_forward building a new tensor; Rec::Add backward shares the Arc-backed gradient with both parents; bit-identical results under Exact"
  - "accumulate_grad runs parallel with the refuse-before-write guarantee: a parallel fused finite check then a parallel add (two passes kept on purpose, documented at ojas-cpu/src/accum.rs:22-24)"
  - "Embedding NEON, permute mmov/pair-move and Muon no-transpose/single-Accelerate paths are generalised beyond the nanolab shapes (row==768, head dim 64, [B,1024,12,64], 2048x768) or each shape gate is justified in a comment; Qwen3.5 shapes (hidden >= 2048, D=256) benchmarked before and after"
  - "Hybrid-op backwards (conv1d, gated RMSNorm) parallelise their row passes with the deterministic split plain RMSNorm already uses; weight-gradient order stays bit-stable; CPU bench rows exist for GDN, conv1d, gated RMSNorm and partial RoPE"
  - "Muon Newton-Schulz reuses its five per-iteration buffers; X*X^T uses a symmetric product or the reason not to is recorded"
  - "A decision is recorded on the ~37 µs std::thread::scope spawn per scoped op (persistent pool writing borrowed outputs needs unsafe or a dependency, both need the user's approval)"
  - "Each open line in docs/bench-cpu-vs-torch.md:250-303 (mul/add fwd, add bwd, embedding fwd/bwd, SiLU fwd, AdamW vs fused torch, Muon 2048x768, permute) is fixed or marked won't-fix with numbers"
  - "Every speedup cites an interleaved min-of-N A/B run committed under bench/results/"
  - "Loss-gradient seed scaling stops materialising full-size tensors per micro-batch: with gradient accumulation K>1 the trainer seeds backward with 1/K (ojas-model/src/trainer.rs:739), so Tape::scale_loss_grad (ojas-autograd/src/tape.rs:1277-1296) never takes the unit shortcut; on CPU it copies the whole gradient out and back (to_f32_vec, scale, from_f32; for LinearCe that includes the vocab x d_model weight gradient), and on GPU broadcast_scalar (:1302-1315) builds a full-size tensor with two linear_forward calls before mul_forward allocates another. The seed is passed into the cross-entropy backward, or a scale-by-scalar op is added; results bit-identical under Exact"
  - "The conv1d backward weight-gradient pass (ojas-cpu/src/hybrid.rs:113-126) walks the activations once per (channel, tap) pair at channel stride; its loop order is made cache-friendly alongside the parallelisation item above, with the bit-stable reduction order kept"
---

# Task brief v1

## Title
CPU and tape hot paths: in-place gradient fan-in, parallel accumulate_grad, shape-general fast paths, loss-seed scaling, open CPU-vs-torch rows

Task: gp-cpu-autograd-hot-paths
Type: perf
Status: done
Priority: 1 (High)
Severity: medium
Owner: unassigned
Due: none
Labels: cpu, autograd, performance, simd, optimizer

## Repositories
- ojas

## Description
Gap audit 2026-10-07 (rg + Read, DevMap unavailable). V = verified, I = inferred.

1. **Tape fan-in allocates [V].** `ojas-autograd/src/tape.rs:1325-1332` sums with `backend.residual_add_forward(old, &grad)`, a new tensor every time. `ojas-cpu/src/accum.rs` already adds in place for a sole owner. `Rec::Add` backward (`tape.rs:1212-1218`) calls `residual_add_backward`, which writes two copies (`ojas-cpu/src/pointwise.rs:964-982`); `Tensor` is Arc-backed (`ojas-core/src/tensor.rs:105-107`). Add bwd is an open line in `docs/bench-cpu-vs-torch.md:295`.
2. **accumulate_grad is two serial passes [V].** `accum.rs:32-46`: a serial finiteness scan, then a serial add, despite receiving `exec`. The trainer calls it per parameter per micro-batch (`ojas-model/src/trainer.rs:874`).
3. **Fast paths only at nanolab shapes [V].** Embedding NEON only at `row == 768` (`pointwise.rs:87`); permute mmov only at head dim 64 (`layout.rs:257`), pair move only at `[B,1024,12,64]` (`layout.rs:267-272`); Muon only at 2048x768 / 768x2048 (`optim.rs:822,871`).
4. **Hybrid-op backwards are serial scalar loops [V].** conv1d backward `hybrid.rs:87-128` (recomputes `conv_pre` per element, channel-strided `gw` loop); gated RMSNorm backward `hybrid.rs:192-219`. The module doc only justifies serial weight-gradient order. No CPU bench rows for GDN/conv1d/gated RMSNorm/partial RoPE. (These files are the in-flight work of `gp-autograd-and-model-primitives`; coordinate with that lane before editing.)
5. **Muon NS5 allocates five Vecs per iteration [V]** (`optim.rs:874-929`) and runs X*X^T as a full GEMM.
6. **Scoped-op thread spawn ~37 µs [V doc]** (`ojas-cpu/src/pool/scoped.rs:7-11`), named as why mul/add trail torch (`bench-cpu-vs-torch.md:242`). Needs a policy ruling (unsafe or dependency).
7. **Fresh operands scanned for NaN one after another [V doc]** (`ojas-cpu/README.md:99`); fold into item 2's fused-pass work where it applies.

**Added by the second gap audit 2026-10-07** ([V] re-read by the auditor; [A] read by an audit subagent, not re-read):
8. **Loss seed scaled by materialising tensors [V].** `scale_loss_grad` (tape.rs:1277-1296): the CPU branch does `raw.to_f32_vec()`, multiplies each value by the seed, then `Tensor::from_f32`; other backends call `broadcast_scalar` (two `linear_forward` calls against ones vectors) then `mul_forward`. It runs for both `Rec::CrossEntropy` and both outputs of `Rec::LinearCe` (tape.rs:1238, :1254-1255). The 1/K seed at trainer.rs:739 is [A]. The GPU half lives in the same function, so it is owned here rather than on gp-gpu-step-throughput.
9. **conv1d gw loop order [A]:** item 4 names the channel-strided loop; this adds the access-order fix to the parallelisation work.

### Close-out audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

Closed by c85eecb (merge), a9af0a0, 1dea12d and d3fd0b4; the brief was never updated after them [A]:
- **In-place fan-in:** tape.rs:1406-1418, with Rec::Add sharing the gradient Arc at :1292-1305.
- **Shape gates:** generalised or justified (pointwise.rs:86; layout.rs:256, 266-282; optim.rs:828-842).
- **Hybrid backwards:** parallel (hybrid.rs:109-277), with bench rows in bench_ops.rs:659-741.
- **Muon NS5:** buffers reused and a symmetric product via cblas_ssyrk (optim.rs:740, 892, 908).
- **Spawn decision:** recorded (pool/scoped.rs:24-30).
- **Bench lines:** every line ruled (docs/bench-cpu-vs-torch.md:314-340).
- **A/B results:** committed (bench/results/2026-10-08-cpu-hot-paths/).
- **Seed scaling:** uses scale_grad (tape.rs:1369-1388).
- **conv1d weight gradient:** tap-major (hybrid.rs:138).

Leftovers moved to gp-cpu-gdn-and-pool-followups: the two std-only spawn mitigations (open and unmeasured), the cut tuned to a 6-thread machine (pointwise.rs:508), and the CPU GDN copies. GPU scale_grad is owned by gp-gpu-step-throughput.

## Acceptance criteria
- [x] Tape::acc accumulates fan-in gradients in place (accumulate_grad when the accumulator is solely owned) instead of residual_add_forward building a new tensor; Rec::Add backward shares the Arc-backed gradient with both parents; bit-identical results under Exact
- [x] accumulate_grad runs parallel with the refuse-before-write guarantee: a parallel fused finite check then a parallel add (two passes kept on purpose, documented at ojas-cpu/src/accum.rs:22-24)
- [x] Embedding NEON, permute mmov/pair-move and Muon no-transpose/single-Accelerate paths are generalised beyond the nanolab shapes (row==768, head dim 64, [B,1024,12,64], 2048x768) or each shape gate is justified in a comment; Qwen3.5 shapes (hidden >= 2048, D=256) benchmarked before and after
- [x] Hybrid-op backwards (conv1d, gated RMSNorm) parallelise their row passes with the deterministic split plain RMSNorm already uses; weight-gradient order stays bit-stable; CPU bench rows exist for GDN, conv1d, gated RMSNorm and partial RoPE
- [x] Muon Newton-Schulz reuses its five per-iteration buffers; X*X^T uses a symmetric product or the reason not to is recorded
- [x] A decision is recorded on the ~37 µs std::thread::scope spawn per scoped op (persistent pool writing borrowed outputs needs unsafe or a dependency, both need the user's approval)
- [x] Each open line in docs/bench-cpu-vs-torch.md:250-303 (mul/add fwd, add bwd, embedding fwd/bwd, SiLU fwd, AdamW vs fused torch, Muon 2048x768, permute) is fixed or marked won't-fix with numbers
- [x] Every speedup cites an interleaved min-of-N A/B run committed under bench/results/
- [x] Loss-gradient seed scaling stops materialising full-size tensors per micro-batch: with gradient accumulation K>1 the trainer seeds backward with 1/K (ojas-model/src/trainer.rs:739), so Tape::scale_loss_grad (ojas-autograd/src/tape.rs:1277-1296) never takes the unit shortcut; on CPU it copies the whole gradient out and back (to_f32_vec, scale, from_f32; for LinearCe that includes the vocab x d_model weight gradient), and on GPU broadcast_scalar (:1302-1315) builds a full-size tensor with two linear_forward calls before mul_forward allocates another. The seed is passed into the cross-entropy backward, or a scale-by-scalar op is added; results bit-identical under Exact
- [x] The conv1d backward weight-gradient pass (ojas-cpu/src/hybrid.rs:113-126) walks the activations once per (channel, tap) pair at channel stride; its loop order is made cache-friendly alongside the parallelisation item above, with the bit-stable reduction order kept

## Planned files
- ojas-autograd/src/tape.rs
- ojas-cpu/src/accum.rs
- ojas-cpu/src/pointwise.rs
- ojas-cpu/src/layout.rs
- ojas-cpu/src/optim.rs
- ojas-cpu/src/hybrid.rs
- ojas-cpu/src/gdn.rs
- ojas-cpu/src/pool/scoped.rs
- ojas-cpu/tests/bench_ops.rs
- docs/bench-cpu-vs-torch.md
