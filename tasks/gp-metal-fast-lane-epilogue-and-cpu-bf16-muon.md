---
id: "gp-metal-fast-lane-epilogue-and-cpu-bf16-muon"
title: "Route ojas-metal Fast/Autocast to tessl cooperative TensorOps epilogues, fold clip scale into optimizer steps, and accelerate CPU/GPU bf16 Muon"
status: ready
priority: 1
severity: high
type: perf
owner: "unassigned"
due: "none"
labels:
  - "metal"
  - "cpu"
  - "simd"
  - "muon"
  - "bf16"
  - "performance"
  - "audit-2026-10-16"
repositories:
  - "ojas"
  - "tessl"
planned_files:
  - "ojas-metal/src/backend.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-core/src/backend.rs"
  - "ojas-model/src/trainer.rs"
  - "ojas-simd/src/arch.rs"
  - "ojas-cpu/src/optim.rs"
  - "docs/bench-gpu-vs-torch.md"
acceptance_criteria:
  - "Wire ojas-metal under Numerics::Fast and Autocast to tessl's 4-simdgroup 128x64 / 64x64 cooperative bf16 and tf32-relaxed TensorOps lanes and gemm_epilogue so Linear and conv2d are not bottlenecked on the 1-simdgroup 32x32 ExactF32 fallback (AUDIT_REPORT.md Risk #1, E1)"
  - "Update docs/bench-gpu-vs-torch.md:74-92 to retire the stale 18.96 ms (0.45x vs torch 2.15) clip_grad_norm_ claim: 7 interleaved rounds on M5 Pro (2026-10-16) proved main (after 0e3254f ojas_norm_multi) already runs clip_grad_norm_full in 6.60 ms min / 7.07 ms median (1.20x faster than torch 2.15 nightly's 7.90 ms min / 8.45 ms median), and wiring tessl's norm in (6.49 ms min / 7.45 ms median) is within noise because the in-place scaling pass moves 2x the bytes of the read-only norm"
  - "Decide (owner sign-off on Backend trait change) whether to fold the clip_grad_norm scale factor into the optimizer step (adamw_step / muon_ns5_step) to eliminate the 2x read+write gradient scaling pass and the mid-step Wait::ClipNorm host sync"
  - "Reject or explicitly handle grad_clip == 0 in ojas-model/src/trainer.rs:226 and ojas-core/src/backend.rs:172-188 where clip_scale(0, positive) == 0 currently zeroes all gradients, and fail closed when a trainable slot produces no gradient at trainer.rs:858-861 (ft-9c13f472)"
  - "Eliminate the 1.14x-1.40x CPU slowdown on Ns5Precision::Bf16 Muon (docs/bench-gpu-vs-torch.md:66-72) by adding ARM64 NEON/SME BFMMLA (vbfmmlaq_f32) intrinsics under Numerics::Fast in ojas-simd, and batch same-shape parameter matrices into 3D tensor banks for GPU Muon NS5"
---

# Task brief v1

## Title
Route ojas-metal Fast/Autocast to tessl cooperative TensorOps epilogues, fold clip scale into optimizer steps, and accelerate CPU/GPU bf16 Muon

Task: gp-metal-fast-lane-epilogue-and-cpu-bf16-muon
Type: perf
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: metal, cpu, simd, muon, bf16, performance, audit-2026-10-16

## Repositories
- ojas
- tessl

## Description
Addresses the concrete GPU and ARM64 CPU performance gaps identified in `ojas/docs/bench-gpu-vs-torch.md` and `ojas/AUDIT_REPORT.md` (Risk #1 `G-V2 / E1`, `ft-2c183b71`, `ft-9c13f472`):

1. **`ExactF32` Bottleneck in `ojas-metal` (`AUDIT_REPORT.md` Risk #1):** Because `ojas-metal` dispatches `GemmOperands::ExactF32`, it runs `tessl`'s 1-simdgroup 32x32 tile without epilogue fusion, leaving composed nanolab blocks at 0.92x-1.07x PyTorch MPS speed. Routing `Numerics::Fast` and `Autocast` to `tessl`'s 4-simdgroup 128x64 cooperative TensorOps kernels and `gemm_epilogue` unlocks M5 Pro's 26.6-29.0 TFLOP/s ceiling.
2. **Verified `clip_grad_norm_full` Reality & Optimizer-Fused Scaling (2026-10-16 GPU Run):** `docs/bench-gpu-vs-torch.md:74-92` compared PyTorch 2.15 nightly (`8.45 ms` median / `7.90 ms` min after `pytorch#198611`) against Round 5's pre-`0e3254f` `18.96 ms` measurement and concluded `ojas` was `0.45x` of PyTorch 2.15. On 2026-10-16, wiring `tessl`'s norm into `ojas-metal` and running 7 interleaved rounds on the M5 Pro showed:
   - `main` (with `0e3254f`'s `ojas_norm_multi`, never previously timed): **6.60 ms** (min of mins) / **7.07 ms** (median of medians) — already **~1.20x faster** than PyTorch 2.15 nightly (`8.45 ms` median).
   - `branch` (`tessl` norm wired in): **6.49 ms** (min of mins) / **7.45 ms** (median of medians) — indistinguishable within single-round spread (~5%).
   - **Root cause:** The L2 norm reduction (1x read) is no longer the bottleneck; scaling the gradients in place (1x read + 1x write = 2x data movement) dominates the time. Going faster requires folding the clip scale into the optimizer (`adamw_step` / `muon_ns5_step`), which changes the `Backend` trait and requires owner sign-off.
3. **CPU `bf16` Muon Slowdown & Single-Matrix GPU Muon (`bench-gpu-vs-torch.md:66-72`):** Because Apple Accelerate `sgemm` is `f32`-only, CPU `Ns5Precision::Bf16` executes `f32` GEMMs plus software `bf16` rounding passes and runs 1.14x-1.40x slower than `f32` (13.43 ms vs 9.62 ms on 768x768). Meanwhile, GPU Muon runs one matrix per call instead of batching across 3D tensor banks.

## Acceptance criteria
- [ ] Wire ojas-metal under Numerics::Fast and Autocast to tessl's 4-simdgroup 128x64 / 64x64 cooperative bf16 and tf32-relaxed TensorOps lanes and gemm_epilogue so Linear and conv2d are not bottlenecked on the 1-simdgroup 32x32 ExactF32 fallback (AUDIT_REPORT.md Risk #1, E1)
- [ ] Update docs/bench-gpu-vs-torch.md:74-92 to retire the stale 18.96 ms (0.45x vs torch 2.15) clip_grad_norm_ claim: 7 interleaved rounds on M5 Pro (2026-10-16) proved main (after 0e3254f ojas_norm_multi) already runs clip_grad_norm_full in 6.60 ms min / 7.07 ms median (1.20x faster than torch 2.15 nightly's 7.90 ms min / 8.45 ms median), and wiring tessl's norm in (6.49 ms min / 7.45 ms median) is within noise because the in-place scaling pass moves 2x the bytes of the read-only norm
- [ ] Decide (owner sign-off on Backend trait change) whether to fold the clip_grad_norm scale factor into the optimizer step (adamw_step / muon_ns5_step) to eliminate the 2x read+write gradient scaling pass and the mid-step Wait::ClipNorm host sync
- [ ] Reject or explicitly handle grad_clip == 0 in ojas-model/src/trainer.rs:226 and ojas-core/src/backend.rs:172-188 where clip_scale(0, positive) == 0 currently zeroes all gradients, and fail closed when a trainable slot produces no gradient at trainer.rs:858-861 (ft-9c13f472)
- [ ] Eliminate the 1.14x-1.40x CPU slowdown on Ns5Precision::Bf16 Muon (docs/bench-gpu-vs-torch.md:66-72) by adding ARM64 NEON/SME BFMMLA (vbfmmlaq_f32) intrinsics under Numerics::Fast in ojas-simd, and batch same-shape parameter matrices into 3D tensor banks for GPU Muon NS5

## Planned files
- ojas-metal/src/backend.rs
- ojas-metal/src/device.rs
- ojas-core/src/backend.rs
- ojas-model/src/trainer.rs
- ojas-simd/src/arch.rs
- ojas-cpu/src/optim.rs
- docs/bench-gpu-vs-torch.md
