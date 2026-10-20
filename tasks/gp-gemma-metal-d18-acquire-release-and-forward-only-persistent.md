---
id: "gp-gemma-metal-d18-acquire-release-and-forward-only-persistent"
title: "Megakernel probe: rerun D18 gate→down with MSL 4.1 acquire/release, then a forward-only-dependency persistent prototype"
status: ready
priority: 2
severity: low
type: research
owner: "unassigned"
due: "none"
labels:
  - "metal"
  - "megakernel"
  - "gemma-metal"
  - "research"
  - "research-2026-10-20"
repositories:
  - "ojas"
planned_files:
  - "gemma-metal/docs/layer_fusion_v1.md"
  - "gemma-metal/bench/fusion_ab.sh"
acceptance_criteria:
  - "D18's visibility test is rerun with device-scope acquire/release (atomics and fences per MSL 4.1) in place of relaxed; pass or fail is recorded with the macOS, Xcode and MSL versions"
  - "If it passes: a gate→down persistent prototype where a threadgroup only waits on producers launched before it (atomically assigned group ids, never a later group), with a spin-iteration bound that falls back to a two-dispatch path and records it"
  - "The prototype runs 1,000 decode steps with the display active (WindowServer sharing the GPU) and with it idle, and records any hang or fallback"
  - "fusion_ab exactness passes (fused ≡ unfused token stream) before any tok/s is quoted; dispatches/token and quiet tok/s are recorded"
  - "Stop rule: if the E4B tok/s gain is under 3%, the lane is parked with its numbers in layer_fusion_v1.md"
---

# Task brief v1

## Title
Megakernel probe: rerun D18 gate→down with MSL 4.1 acquire/release, then a forward-only-dependency persistent prototype

Task: gp-gemma-metal-d18-acquire-release-and-forward-only-persistent
Type: research
Status: ready
Priority: 2
Source: research session 2026-10-20 (LinkedIn article audit, Qwen3.8-27B inference and megakernel research)

## Description
D18 (gemma-metal/docs/layer_fusion_v1.md): a single-dispatch gate→down fusion with a **relaxed** grid barrier did not publish `mid` to the in-kernel DOWN at E4B dims (fused residual ~1e9). The test is locked as `persistent_interp_gate_down_q4_e4b_dims_visibility_blocker`.

MSL 4.1 (macOS 27) adds acquire/release memory ordering to atomics and barriers. That may fix the visibility half of the bug. It does not fix forward progress: Apple states Metal guarantees neither threadgroup launch order nor forward progress between threadgroups, so spin-waits across the grid are undefined behavior (developer forums thread 672532).

Prior art is CUDA/ROCm only (Hazy "No Bubbles", Mirage MPK, Cohere, Lucebox hybrid DeltaNet), and no Metal megakernel has been published. Your own data argues for a small payoff:
- tessl's batched dispatch costs 3–8 µs GPU time and under 1 µs host encode (bench/results/dispatch_host_overhead_after_m5pro.txt).
- E4B fusion that cut dispatches 600→510 was flat in tok/s.

Treat this as a bounded research bet with a stop rule.

## Acceptance criteria
- [ ] D18's visibility test is rerun with device-scope acquire/release (atomics and fences per MSL 4.1) in place of relaxed; pass or fail is recorded with the macOS, Xcode and MSL versions
- [ ] If it passes: a gate→down persistent prototype where a threadgroup only waits on producers launched before it (atomically assigned group ids, never a later group), with a spin-iteration bound that falls back to a two-dispatch path and records it
- [ ] The prototype runs 1,000 decode steps with the display active (WindowServer sharing the GPU) and with it idle, and records any hang or fallback
- [ ] fusion_ab exactness passes (fused ≡ unfused token stream) before any tok/s is quoted; dispatches/token and quiet tok/s are recorded
- [ ] Stop rule: if the E4B tok/s gain is under 3%, the lane is parked with its numbers in layer_fusion_v1.md
