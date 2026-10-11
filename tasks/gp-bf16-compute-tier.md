---
id: "gp-bf16-compute-tier"
title: "BF16 storage and compute tier: bf16 operands across Backend, bf16 GEMM and attention on CPU/Metal/wgpu, B>=8 memory study, stated divergence bound (Muon bf16 NS5 slice done)"
status: ready
priority: 2
severity: high
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
  - "Decision first (needs the user): bf16 as an input on every Backend op, or on matmul-class ops only. Then bf16 tensor storage and kernel operands across the chosen methods; today only the CPU reports bf16_operands() = true (ojas-cpu/src/backend.rs:388), widening to f32 scratch for linear and SDPA only (a5ccdc5); Metal refuses (backend.rs:1257,1283,1427,1443) and wgpu inherits false"
  - "Implement BF16 GEMM with FP32 accumulation on CPU (Accelerate / NEON), Metal (Apple matrix units), and wgpu; Metal's bf16 GEMM exists only inside Muon (device.rs:2969)"
  - "Implement BF16 attention forward and backward kernels with numerical stability guards"
  - "Verify that batch size B >= 8 micro-batches fit in Apple Silicon unified memory without paging or thrashing"
  - "Parity and loss divergence tests pass against FP32 reference, with a stated divergence bound (not only finite-and-different); no such test exists outside the CUDA GDN tests"
  - "Add optional bf16 compute path for Newton-Schulz 5 quintic iterations in Muon optimizer"
  - "Align update magnitudes exactly with PyTorch and nanolab stock optimizer steps"
  - "Measure throughput improvement from BF16 batched matrix multiplications during orthogonalization"
  - "Golden trace parity test verifying convergence within 1e-4 nats against stock nanolab trace"
  - "wgpu bf16 NS5 (refused at ojas-wgpu/src/backend.rs:2440) is implemented or the refusal is documented as permanent"
  - "bf16/f16 NaN conversion is decided: autocast.rs:58-65 and half.rs:42-44 keep sign and payload where torch canonicalises; the chosen rule is pinned by a test"
---

# Task brief v1

## Title
BF16 storage and compute tier: bf16 operands across Backend, bf16 GEMM and attention on CPU/Metal/wgpu, B>=8 memory study, stated divergence bound (Muon bf16 NS5 slice done)

Task: gp-bf16-compute-tier
Type: feature
Status: ready
Priority: 2 (Normal)
Severity: high
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

### Re-audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- Status moved from review back to ready. Criteria 1-5, the storage and compute tier, have not started. The Muon bf16 NS5 slice (6-9) is done on CPU and Metal per the 2026-10-07 notes [A; exact_golden not re-run, U].
- Caveat on criterion 8: the 2026-10-07-muon-bf16 bench ran on c74f3ba with 172 dirty files, and the quoted 0.94x row is marked 'noisy - not quoted' in its own table [A].
- muon_ns5 in the C ABI and Go TrainConfig is owned by gp-capi-go-surface (TrainConfig parity), not here.

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- New: bf16 NaN canonicalisation differs from torch [A, torch behaviour from the auditor's memory, unverified].

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Raised P2 -> P1: the frozen-base fine-tune needs bf16 operands. Recommendation for criterion 1's decision, from the fine-tune track: bf16 on matmul-class and attention ops is enough for a frozen bf16 base, and bf16 on every op is not needed. Still the user's call.
- Fine-tune briefs that depend on this one: gp-ft-bf16-frozen-base.

### Fine-tune decisions (2026-10-09)
Fable ruled on the fine-tune track's decisions and the user approved them (2026-10-09), as relayed by sibling session ojas-7c. The record is in that session's scratchpad: fable-ft-decisions-2026-10-09.md, sections D1-D6.
- Tape path for LoRA.
- bf16 frozen base; no 4/8-bit.
- 2B-Base, with a 0.8B-Base smoke; 4B is out of the track.
- Lappi's data shape: rows up to 9,638 tokens, a 35,403-token budget, grad_accum 1.
- Lappi's own LRSchedule.
- The trainer moves into a new ojas-train crate.
Checked by this session [V]: the local Qwen3.5-2B snapshot exists (4.3 GB). ojas-qwen35/src/config.rs:503-508 still refuses unequal GDN value and key heads with 'tessl's gdn_train has no head grouping', while ../tessl/src/qwen35_train.rs now groups heads.
- Priority 1 -> 2: gp-ft-bf16-frozen-base now carries the four Metal bf16 lanes the fine-tune needs (linear forward, linear dX, SDPA forward/backward, bf16 load without widening) and lists this tier as related, not as a dependency. The general tier is off the critical path.

### Progress notes (2026-10-10/11): decisions made; wgpu tier, NaN rule, divergence bound and memory rows done; Metal lanes with gp-ft-bf16-frozen-base
Branch `gitpulse/bf16-storage-and-compute-tier-bf16-opera-c03cbd7e`, commits 1aa85b6..c59bf70 (not pushed or merged). Labels: [V] verified by a command this session ran, [I] inferred, [U] unverified.

Decisions. The user delegated them to Fable on 2026-10-10. Fable's rulings D1-D8 are summarised here.
- D1, criterion 1. Bf16 is a kernel input on the matmul-class ops only: linear_forward, linear_backward, causal_sdpa_forward, and causal_sdpa_backward except its lse. linear_backward_input joins the list when it lands. Every other op refuses Bf16 with Dtype. This ratifies the contract at ojas-core/src/backend.rs:1391-1409 and matches the fine-tune track's D3.
- D2, Metal (gp-ft-bf16-frozen-base owns ojas-metal). Under Exact, a Bf16 operand reaches the existing f32 arithmetic unchanged, by widening or a bf16-load variant. tessl's GemmOperands::Bf16 lane is never used for a mixed product: ensure_bf16 rounds the f32 operand (tessl gemm.rs:350), and the lane is chosen by runtime-global precision (gemm.rs:365). For all-Bf16 operands the lane may run under Exact only if the conformance runner shows bits identical to ExactF32. Until then it is allowed only under Fast. The device.rs:2977 comment is not evidence.
- D3, CPU. Widening plus Accelerate f32 sgemm is the CPU answer. That is bf16 storage over f32 compute: Accelerate has no bf16 GEMM. No BFMMLA kernel; it could not keep the Exact bits contract.
- D4, wgpu. Widen into charged scratch for the four ops; native bf16 GEMM stays with gp-wgpu-coop-matrix-gemm. D4b: implement wgpu bf16 NS5 with the CPU's scheme. Deviation: Fable asked for a NaN/Inf scan on the bf16 bits before any charge, as the CPU does. wgpu charges the widened scratch first. A bf16 NaN or infinity raises the op's deferred fault inside widen_bf16, reported at the next sync, which is wgpu's documented fault model for every op. A synchronous pre-scan would add a readback per op.
- D5, NaN. Keep sign, top payload and quiet bit 0x0040, and document the difference from torch.
- D6, divergence. ojas Autocast-bf16 against ojas f32 over the 40-step f32-NS5 trace. Bound: max(CURVE_LATE_ABS_TOL, 2 x torch's own worst autocast gap).
- D7, memory. Paging is judged on the process's own counters; vm_stat is context. Deviation: Fable's shape was Qwen3.5-2B frozen base plus LoRA at T=2048 with checkpointing on, f32 against a resident bf16 base. This branch has no LoRA or frozen-base path; those shapes are gp-ft-bf16-frozen-base's. The rows measured here are nanolab 124M at T=1024, checkpointing off, f32 against an autocast-bf16 region. At that scale the brief's premise that FP32 thrashes unified memory at B >= 8 was not reproduced: B=8 peaks at 10.4 GiB.

Done:
- Contract runner `ojas_oracle::bf16_operands::bf16_operand_contract` (1aa85b6), runnable on any backend [V]. It checks every F32/Bf16 mix of the four ops for bit equality with the all-F32 call: GQA and windowed attention, and Bf16 views at an odd element offset. It also checks 15 Dtype refusals, 26 NonFinite cases, and device to_bf16/to_f32 against the host bits. Results: CPU Exact, Fast and Autocast 192/15/26 [V]; wgpu 192/15/26 [V]; Metal not run here [U].
- wgpu bf16 operands, device to_bf16/to_f32, bf16_operands() = true (97874fb) [V]: Bf16 operands are widened exactly by widen_bf16 into charged job scratch, then the f32 kernels run.
- wgpu bf16 NS5 (97874fb) [V]:
  - against stock nanolab's four muon_step_bf16 cases, update rel is 0 [V]; the f32 contrast is 3.0e-2 to 3.6e-2;
  - zero-param wide steps give bf16 values;
  - against CPU bf16: 0 at 64x64, 64x192, 192x64 and 131x67, and 1.22e-2 at 768x768 (gate 5e-2);
  - throughput was not measured on wgpu [U].
- NaN rule (8c7a4cc) [V]. Measured with torch 2.13 on arm64 (CPU and MPS): f32->bf16 gives 0x7FC0 for every NaN, and f32->f16 keeps sign and payload exactly as ojas does. So the brief's premise held for bf16 only. The rule is pinned in ojas-io/tests/half_float.rs. ojas-cuda's host bf16 copy now delegates to ojas-core.
- Divergence bound (5324625) [V]. torch's own gap is 9.36e-4, so the bound is 2e-3. ojas's worst gap is 8.23e-4. ojas bf16 is 8.52e-4 from torch bf16; that is reported, not gated. parity_gates covers the failure modes.
- B >= 8 memory (c59bf70; bench/results/2026-10-10-bf16-memory) [V]:
  - nanolab 124M, T=1024, B=8: peak 10.4-10.6 GiB on CPU, wgpu and Metal, f32 and bf16 region; no process swaps; machine swapouts 0.
  - The bf16 rows are 0.00-0.24 GiB larger, as expected: the tape and masters are f32.
  - The paging rule was amended after the run (major faults include file-backed page-ins); the README says so.
- The whole ojas-wgpu suite passes; ojas-core, ojas-io, ojas-kernels, ojas-cuda (lib), ojas-oracle and ojas-model oracle_parity/trainer pass. clippy -D warnings passes on the 7 touched crates. sitegen -check is clean [V].

Open, and why (status review):
- Criterion 1, Metal. gp-ft-bf16-frozen-base (sibling session, worktree bf16-frozen-base-on-metal-four-bf16-oper-b35da9b5) wrote Metal widening lanes for all four ops and device casts. Unbuilt and untested when last reported [U]. It should run the conformance runner on Metal.
- Criterion 2, the GEMM half. No native bf16 GEMM runs as a general op on any backend: CPU and wgpu widen to f32 (D3, D4), and Metal's lanes widen. Metal's TensorOps bf16 lane (D2c) needs the bit-identity measurement first.
- Criterion 3. Attention takes Bf16 operands on CPU and wgpu by widening. The stability guards (online softmax, f32 lse, f32 accumulation) are the f32 kernels'. No bf16-load attention kernel was written; on Metal that is D2d, in the sibling's scope.
- Throughput of wgpu bf16 NS5 against f32 was not measured.
- Pre-fix evidence is inferred, not run [I]: the new wgpu tests were not run against the pre-change code. At HEAD, muon.rs asserted Unsupported for Ns5Precision::Bf16 and passed, and the four deleted refuse_bf16_operands calls returned Unsupported for any Bf16 operand.
- Collision check, 2026-10-11 [V]. gitpulse_collision_risk reports 0 overlaps, but it compares uncommitted changes only, so it cannot see this branch's commits. A manual comparison with the sibling worktree found two files touched by both branches, in different hunks: ojas-io/tests/half_float.rs and Cargo.lock. The sibling also cherry-picked 1aa85b6 as 0eb4c2d.
- 2026-10-11, after review [V]. The sibling session reported that its Metal lanes pass bf16_operand_contract on Metal (192/15/26), on its branch (ft-3cc9ed3f, in review). It also flagged that Autocast::prep_matmul made a 2-byte to_bf16 copy which every backend then widened into 4 more bytes. Fable ruled to fold prep_matmul into prep. That is 749f979: the region now rounds in F32 storage on every backend, and a caller-held Bf16 tensor still passes through.
  - New test: a_region_rounds_in_f32_storage_even_when_the_backend_takes_bf16 failed before the fix and passes after.
  - Every workspace test with "autocast" in its name passes, and the oracle, model and wgpu bf16 tests pass. clippy is clean.
  - B=8 re-run (bench/results/2026-10-10-bf16-memory/rows-2026-10-11.jsonl): the bf16 region's peak is 10.46 GiB on wgpu (was 10.64) and 10.50 GiB on CPU (was 10.53). Losses are unchanged.
- Merge check, reported by the sibling session [A, not re-run here]. It merged this branch at 749f979 into its own branch at d09031b with no conflicts; Cargo.lock keeps both ojas-oracle dev-dep entries. cargo test --release --no-fail-fast over ojas-core, ojas-metal, ojas-model, ojas-oracle and ojas-qwen35 gave 72 binaries ok and 0 failures. Since 749f979 this branch has changed only bench results and these notes.
- Follow-ups noted, not filed by this session: ft-69744e54, filed by the sibling, needs a contract change in this tier (a bf16 tied embedding table: embedding_forward and linear_cross_entropy_mean taking Bf16).
- Follow-ups not filed: a tape that keeps matmul-class operands as Bf16 (the activation-memory lever); tessl's Qwen35Grads::zeros_like gives a bf16 model a bf16 gradient bank (qwen35_train.rs:132-139, flagged by the sibling session).

### Execution plan
1. Introduce BF16 tensor storage representation and operands to `Backend` trait methods.
2. Implement BF16 GEMM with FP32 accumulation on CPU (Accelerate / NEON), Metal (Apple matrix units), and wgpu.
3. Implement BF16 attention forward and backward kernels with numerical stability guards.
4. Implement optional BF16 Newton-Schulz 5 quintic iterations for Muon, measuring throughput speedup and golden trace parity within 1e-4 nats against nanolab.
5. Establish bounded loss divergence limits against FP32 baselines.

## Acceptance criteria
- [ ] Decision first (needs the user): bf16 as an input on every Backend op, or on matmul-class ops only. Then bf16 tensor storage and kernel operands across the chosen methods; today only the CPU reports bf16_operands() = true (ojas-cpu/src/backend.rs:388), widening to f32 scratch for linear and SDPA only (a5ccdc5); Metal refuses (backend.rs:1257,1283,1427,1443) and wgpu inherits false
- [ ] Implement BF16 GEMM with FP32 accumulation on CPU (Accelerate / NEON), Metal (Apple matrix units), and wgpu; Metal's bf16 GEMM exists only inside Muon (device.rs:2969)
- [ ] Implement BF16 attention forward and backward kernels with numerical stability guards
- [x] Verify that batch size B >= 8 micro-batches fit in Apple Silicon unified memory without paging or thrashing
- [x] Parity and loss divergence tests pass against FP32 reference, with a stated divergence bound (not only finite-and-different); no such test exists outside the CUDA GDN tests
- [x] Add optional bf16 compute path for Newton-Schulz 5 quintic iterations in Muon optimizer
- [x] Align update magnitudes exactly with PyTorch and nanolab stock optimizer steps
- [x] Measure throughput improvement from BF16 batched matrix multiplications during orthogonalization
- [x] Golden trace parity test verifying convergence within 1e-4 nats against stock nanolab trace
- [x] wgpu bf16 NS5 (refused at ojas-wgpu/src/backend.rs:2440) is implemented or the refusal is documented as permanent
- [x] bf16/f16 NaN conversion is decided: autocast.rs:58-65 and half.rs:42-44 keep sign and payload where torch canonicalises; the chosen rule is pinned by a test

## Planned files
- ojas-core/src/backend.rs
- ojas-core/src/dtype.rs
- ojas-core/src/autocast.rs
- ojas-cpu/src/backend.rs
- ojas-cpu/src/optim.rs
- ojas-metal/src/device.rs
- ojas-metal/kernels/ojas_backend.metal
- ojas-wgpu/src/backend.rs
