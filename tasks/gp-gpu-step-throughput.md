---
id: "gp-gpu-step-throughput"
title: "GPU step throughput: wgpu AdamW check/apply, fused clip_grad_norm, residual-add backward aliasing, wgpu attention, wgpu per-dispatch cost, multi-tensor optimizers, open Metal rows"
status: ready
priority: 1
severity: high
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
  - "wgpu AdamW uses Metal's check-then-apply design (ADAM_CHECK/ADAM_APPLY, ojas-wgpu/src/backend.rs:251-252, :2363-2409; code landed in 0e3254f): the adamw_full re-bench is committed under bench/results/ before this is ticked"
  - "clip_grad_norm on Metal and wgpu computes the global norm in one fused multi-tensor pass (code landed: wgpu clip.wgsl global_norm, backend.rs:973; Metal ojas_norm_multi, device.rs:2535-2593); the dispatch count before and after is recorded"
  - "residual_add_backward: a recorded decision on whether the GPU backends return shared clones of grad_output instead of two copies (Metal device.rs:2457, wgpu backend.rs:2138-2162), with the accumulate_grad sole-owner consequence measured; independent of that decision, wgpu fuses its three check_finite calls and two copies (five dispatches, backend.rs:2149-2160) into one kernel as Metal's ojas_add_bwd does"
  - "wgpu attention re-benched on the current kernel (LSE return, native GQA) before any rewrite; if still under 0.5x torch-mps, a plan for subgroup/tiled WGSL attention is recorded with the measured target"
  - "The open Metal rows below torch (sdpa fwd d64, accumulate_grad, cross_entropy_bwd, block_fwd) and wgpu rows (block_fwd, gate_fwd/bwd, cross_entropy_bwd) each get a profile and either a fix or a written reason"
  - "wgpu FLUSH_AT (64, ojas-wgpu/src/context.rs:56) is A/B measured against at least two alternatives with the step_probe sweep (examples/step_probe.rs:18-24, set_flush_at); the chosen value cites the run"
  - "Metal Link::call per-op round-trip (~7 µs host; ojas-metal/src/link.rs:661-662 allocates a sync_channel and blocks on every op, post is used only for Free) is measured on a decode-sized workload; fire-and-forget encoding with host-assigned ids is implemented or rejected with numbers"
  - "Every claimed speedup has a committed interleaved A/B benchmark under bench/results/ (min-of-N, spread per row): bench/results/2026-10-08-gpu-step-ab/ holds only scripts and its run_ab.sh:2 cites a task id that is not this brief; the 0e3254f changes are run and the out/ plus agg summary committed"
  - "wgpu's fixed cost per dispatch is cut and counted: each dispatch creates a fresh 64 B mapped uniform buffer (params, ojas-wgpu/src/context.rs:996-1002, via mapped() with an OOM error scope and a blocking pollster pop at :566-578) and a fresh bind group with a second blocking pop (:749-759); pool misses in alloc add a third (:498-510). Uniforms move to a ring or push constants, bind groups are cached by buffer identity, and error scopes are batched per flush; blocking pops per dispatch counted before and after"
  - "wgpu read() reuses MAP_READ staging buffers instead of creating one (plus an error scope) per read (context.rs:798-820); every sync, download and clip_grad_norm status read pays it today"
  - "wgpu binds offset views in place (binding offset or an offset parameter word) instead of copying any tensor with byte_offset != 0 into scratch on every op (ojas-wgpu/src/backend.rs:331-347, :377); the copy count on a fused-QKV slice is recorded before and after"
  - "Uploads on Metal and wgpu stop making a transient full host copy outside the host budget: to_ne_bytes() then a second copy into the mapped/device view (wgpu backend.rs:1443 then context.rs:585; Metal backend.rs:883 before the reservation, then device.rs:1117)"
  - "Optimizer steps can run multi-tensor: adamw_step and muon_ns5_step take one parameter or matrix per call (ojas-core/src/backend.rs:1081,1096; trainer loop at ojas-model/src/trainer.rs:909-912); a batched AdamW over all parameters and batched Muon GEMMs are implemented on Metal and wgpu or rejected with numbers"
  - "Metal fused linear cross-entropy computes each logits tile once on the gradient path: pass 2 remakes every tile unless there is a single vocabulary tile (ojas-metal/src/device.rs:2628-2636; default cols 4096 gives 13 tiles at V=50304); keep the tile or explain with a profile"
  - "Metal embedding backward stops ranking tokens in O(N^2): ojas_embed_place (ojas-metal/kernels/ojas_backend.metal:661-692) compares each token with every earlier one, while the host already holds the ids window (backend.rs:265-295) and wgpu sorts on the host (wgpu backend.rs:1511-1522); embed_bwd gets a bench row first"
  - "The wgpu Muon f32 row (0.65-0.71x torch fp32, bench/results/2026-10-07-muon-bf16/summary.md) is profiled and fixed or explained; wgpu's bf16 NS5 refusal (ojas-wgpu/src/backend.rs:2440) stays with gp-bf16-compute-tier"
  - "Metal per_head_sigmoid_gate_backward's standalone finite checks (seven self.check calls, device.rs:1778-1880) are folded into the gate kernels only if the metal_bench gate profile shows them at roughly 10-15% or more of the row (user decision 2026-10-02); residual_add_backward keeps its x/y checks on every backend (also a user decision)"
  - "Metal and wgpu (and CUDA/HIP when they gain ops) override scale_grad in place: today only the CPU does (ojas-cpu/src/backend.rs:1155) and the GPUs inherit the default (ojas-core/src/backend.rs:1165-1173: scalar upload, broadcast_scalar with two ones-vector uploads and two k=1 GEMMs, then mul_forward). It is hit by the trainer's 1/K seed when K > 1 (trainer.rs:739, :852) and by any non-root loss seeded None (tape.rs:1378-1381); about 2 x 154 MB transient at the 50304x768 CE weight gradient. A failing budget-peak test first"
  - "clip_grad_norm stops forcing a host sync mid-step: the trait returns f32, so Metal waits (Wait::ClipNorm, device.rs:2566) and wgpu reads back (backend.rs:2288) before the optimizer runs; Trainer::finish already syncs once per step (trainer.rs:926-929), so the gain is the mid-step stall only, measured before committing to the trait and StepReport.grad_norm API change (needs sign-off)"
  - "The wgpu per-dispatch cost test (tests/dispatch_cost.rs) also covers uploads: every upload today creates a fresh mapped buffer and pops a blocking error scope (context.rs:1045-1100, :899-920), so per-step token and target uploads each cost one blocking pop; uploads use the pool or the cost is measured and accepted"
  - "Unused multi-tensor slots stop binding group[0] read_write several times (wgpu backend.rs:2324 and global_norm; Metal device.rs:2551,2585); a dummy buffer removes the WebGPU writable-aliasing risk on stricter implementations"
---

# Task brief v1

## Title
GPU step throughput: wgpu AdamW check/apply, fused clip_grad_norm, residual-add backward aliasing, wgpu attention, wgpu per-dispatch cost, multi-tensor optimizers, open Metal rows

Task: gp-gpu-step-throughput
Type: perf
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
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

### Re-audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- **0e3254f landed most of this brief's code but none of its measurements.** Ticked on code read alone (the bench rows are still owed under criterion 8): per-dispatch cost (parameter ring with dynamic offset, bind-group cache, counted pops; tests/dispatch_cost.rs:16 asserts 0 pops / 0 new bind groups / 0 staging per steady dispatch), staging reuse in read() (STAGING_CACHE_BYTES), offset views bound in place (view_at, backend.rs:376-392, plus 8d92d73), uploads without a transient full host copy (for_each_ne_piece, context.rs:1045-1060; Metal charged before the call, backend.rs:1043-1064), Metal embedding backward no longer O(N^2) (ojas_embed_place gone; host-grouped gather at device.rs:1388) [A].
- Code done, measurement owed: wgpu AdamW check-then-apply, fused multi-tensor clip, FLUSH_AT sweep harness [A].
- Stale line references refreshed in every criterion [A].
- New: scale_grad default on GPUs, clip sync, wgpu residual_add_backward five dispatches, upload error-scope pops, aliased read_write slots [A].

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- scale_grad and the clip sync are CONFIRMED but narrower than first written. The scale_grad cost appears only with K > 1 or a non-root loss. The clip sync costs one mid-step stall, not a whole extra sync, because finish already syncs. Line references are corrected [A].

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Kept P1.
- Fine-tune briefs that depend on this one: gp-ft-step-memory-and-throughput.

## Acceptance criteria
- [ ] wgpu AdamW uses Metal's check-then-apply design (ADAM_CHECK/ADAM_APPLY, ojas-wgpu/src/backend.rs:251-252, :2363-2409; code landed in 0e3254f): the adamw_full re-bench is committed under bench/results/ before this is ticked
- [ ] clip_grad_norm on Metal and wgpu computes the global norm in one fused multi-tensor pass (code landed: wgpu clip.wgsl global_norm, backend.rs:973; Metal ojas_norm_multi, device.rs:2535-2593); the dispatch count before and after is recorded
- [ ] residual_add_backward: a recorded decision on whether the GPU backends return shared clones of grad_output instead of two copies (Metal device.rs:2457, wgpu backend.rs:2138-2162), with the accumulate_grad sole-owner consequence measured; independent of that decision, wgpu fuses its three check_finite calls and two copies (five dispatches, backend.rs:2149-2160) into one kernel as Metal's ojas_add_bwd does
- [ ] wgpu attention re-benched on the current kernel (LSE return, native GQA) before any rewrite; if still under 0.5x torch-mps, a plan for subgroup/tiled WGSL attention is recorded with the measured target
- [ ] The open Metal rows below torch (sdpa fwd d64, accumulate_grad, cross_entropy_bwd, block_fwd) and wgpu rows (block_fwd, gate_fwd/bwd, cross_entropy_bwd) each get a profile and either a fix or a written reason
- [ ] wgpu FLUSH_AT (64, ojas-wgpu/src/context.rs:56) is A/B measured against at least two alternatives with the step_probe sweep (examples/step_probe.rs:18-24, set_flush_at); the chosen value cites the run
- [ ] Metal Link::call per-op round-trip (~7 µs host; ojas-metal/src/link.rs:661-662 allocates a sync_channel and blocks on every op, post is used only for Free) is measured on a decode-sized workload; fire-and-forget encoding with host-assigned ids is implemented or rejected with numbers
- [ ] Every claimed speedup has a committed interleaved A/B benchmark under bench/results/ (min-of-N, spread per row): bench/results/2026-10-08-gpu-step-ab/ holds only scripts and its run_ab.sh:2 cites a task id that is not this brief; the 0e3254f changes are run and the out/ plus agg summary committed
- [x] wgpu's fixed cost per dispatch is cut and counted: each dispatch creates a fresh 64 B mapped uniform buffer (params, ojas-wgpu/src/context.rs:996-1002, via mapped() with an OOM error scope and a blocking pollster pop at :566-578) and a fresh bind group with a second blocking pop (:749-759); pool misses in alloc add a third (:498-510). Uniforms move to a ring or push constants, bind groups are cached by buffer identity, and error scopes are batched per flush; blocking pops per dispatch counted before and after
- [x] wgpu read() reuses MAP_READ staging buffers instead of creating one (plus an error scope) per read (context.rs:798-820); every sync, download and clip_grad_norm status read pays it today
- [x] wgpu binds offset views in place (binding offset or an offset parameter word) instead of copying any tensor with byte_offset != 0 into scratch on every op (ojas-wgpu/src/backend.rs:331-347, :377); the copy count on a fused-QKV slice is recorded before and after
- [x] Uploads on Metal and wgpu stop making a transient full host copy outside the host budget: to_ne_bytes() then a second copy into the mapped/device view (wgpu backend.rs:1443 then context.rs:585; Metal backend.rs:883 before the reservation, then device.rs:1117)
- [ ] Optimizer steps can run multi-tensor: adamw_step and muon_ns5_step take one parameter or matrix per call (ojas-core/src/backend.rs:1081,1096; trainer loop at ojas-model/src/trainer.rs:909-912); a batched AdamW over all parameters and batched Muon GEMMs are implemented on Metal and wgpu or rejected with numbers
- [ ] Metal fused linear cross-entropy computes each logits tile once on the gradient path: pass 2 remakes every tile unless there is a single vocabulary tile (ojas-metal/src/device.rs:2628-2636; default cols 4096 gives 13 tiles at V=50304); keep the tile or explain with a profile
- [x] Metal embedding backward stops ranking tokens in O(N^2): ojas_embed_place (ojas-metal/kernels/ojas_backend.metal:661-692) compares each token with every earlier one, while the host already holds the ids window (backend.rs:265-295) and wgpu sorts on the host (wgpu backend.rs:1511-1522); embed_bwd gets a bench row first
- [ ] The wgpu Muon f32 row (0.65-0.71x torch fp32, bench/results/2026-10-07-muon-bf16/summary.md) is profiled and fixed or explained; wgpu's bf16 NS5 refusal (ojas-wgpu/src/backend.rs:2440) stays with gp-bf16-compute-tier
- [ ] Metal per_head_sigmoid_gate_backward's standalone finite checks (seven self.check calls, device.rs:1778-1880) are folded into the gate kernels only if the metal_bench gate profile shows them at roughly 10-15% or more of the row (user decision 2026-10-02); residual_add_backward keeps its x/y checks on every backend (also a user decision)
- [ ] Metal and wgpu (and CUDA/HIP when they gain ops) override scale_grad in place: today only the CPU does (ojas-cpu/src/backend.rs:1155) and the GPUs inherit the default (ojas-core/src/backend.rs:1165-1173: scalar upload, broadcast_scalar with two ones-vector uploads and two k=1 GEMMs, then mul_forward). It is hit by the trainer's 1/K seed when K > 1 (trainer.rs:739, :852) and by any non-root loss seeded None (tape.rs:1378-1381); about 2 x 154 MB transient at the 50304x768 CE weight gradient. A failing budget-peak test first
- [ ] clip_grad_norm stops forcing a host sync mid-step: the trait returns f32, so Metal waits (Wait::ClipNorm, device.rs:2566) and wgpu reads back (backend.rs:2288) before the optimizer runs; Trainer::finish already syncs once per step (trainer.rs:926-929), so the gain is the mid-step stall only, measured before committing to the trait and StepReport.grad_norm API change (needs sign-off)
- [ ] The wgpu per-dispatch cost test (tests/dispatch_cost.rs) also covers uploads: every upload today creates a fresh mapped buffer and pops a blocking error scope (context.rs:1045-1100, :899-920), so per-step token and target uploads each cost one blocking pop; uploads use the pool or the cost is measured and accepted
- [ ] Unused multi-tensor slots stop binding group[0] read_write several times (wgpu backend.rs:2324 and global_norm; Metal device.rs:2551,2585); a dummy buffer removes the WebGPU writable-aliasing risk on stricter implementations

## Planned files
- ojas-wgpu/src/backend.rs
- ojas-wgpu/src/context.rs
- ojas-kernels/src/wgsl/optim.wgsl
- ojas-kernels/src/wgsl/attention.wgsl
- ojas-kernels/src/geometry.rs
- ojas-metal/src/backend.rs
- ojas-metal/src/device.rs
- ojas-metal/src/link.rs
- ojas-core/src/backend.rs
- bench/ojas_rows.rs
- docs/bench-gpu-vs-torch.md
