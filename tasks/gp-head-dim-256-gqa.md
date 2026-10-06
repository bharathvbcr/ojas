---
id: "gp-head-dim-256-gqa"
title: "Lift Head Dimension Cap to 256 and Support GQA in Training"
status: review
priority: 0
severity: critical
type: feature
owner: "unassigned"
due: "none"
labels:
  - "attention"
  - "kernel"
  - "autograd"
  - "qwen35"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/backend.rs"
  - "ojas-kernels/src/geometry.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-metal/kernels/ojas_backend.metal"
  - "ojas-wgpu/src/backend.rs"
  - "ojas-autograd/src/tape.rs"
acceptance_criteria:
  - "Remove d <= 128 head dimension limit assertion across Metal and wgpu kernels"
  - "Implement tiled attention forward and backward kernels supporting head dimension 256"
  - "Generalize Backend::causal_sdpa_forward and backward to support Grouped-Query Attention (mismatched Q and KV head counts)"
  - "Autograd Tape records and backpropagates GQA projections correctly"
  - "Unit tests verifying head dimensions 64, 128, and 256 match analytical and CPU reference oracles"
---

# Task brief v1

## Title
Lift Head Dimension Cap to 256 and Support GQA in Training

Task: gp-head-dim-256-gqa
Type: feature
Status: review
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: none
Labels: attention, kernel, autograd, qwen35

## Repositories
- ojas

## Description
Both Metal and wgpu attention kernels currently enforce a strict head dimension cap of 128 (METAL_MAX_HEAD_DIM and ATTENTION_MAX_HEAD_DIM). Modern architectures like Qwen3.5, Gemma, and Llama 3 require head dimension 256 with Grouped-Query Attention (GQA). In Ojas, any attempt to run head dimension 256 fails immediately with OjasError::UnsupportedHeadDim, and GQA currently only exists for inference decoding, not in the autograd training pass.

This task lifts the head dimension barrier to 256 and generalizes the causal SDPA training forward and backward passes to support GQA.

### Progress (audit of 9668bfa, 2026-10-05, code inspection only; tests not executed)
Landed in 1f26a4b; moved to review. A reviewer should run the CPU, Metal and wgpu attention suites before marking it done.
- Cap raised to 256, not removed: `METAL_MAX_HEAD_DIM = 256` (`ojas-core/src/backend.rs:64`), `ATTN_MAX_HEAD_DIM = 256` (`ojas-metal/src/device.rs:67`), `ATTENTION_MAX_HEAD_DIM = 256` (`ojas-kernels/src/geometry.rs:166`). 257 is still refused on GPU.
- GQA in the trait: q is `[B,H,T,D]`, k/v are `[B,Hkv,T,D]` (`backend.rs:541-555`). CPU is native (`ojas-cpu/src/attn.rs:365-403`). Metal and wgpu expand K/V to the full head count and sum the gradients back (new `head_repeat.wgsl`), which costs extra scratch; native grouped kernels are follow-up gp-gqa-native-kernels.
- Tests: CPU `ojas-cpu/tests/ops.rs:757`; Metal `ojas-metal/tests/attention_forward.rs:95,137` (d = 96/128/192/256 fwd+bwd vs CPU); wgpu `ojas-wgpu/tests/parity.rs:223,273`.
- **Weak spot:** the only tape-level GQA test (`ojas-model/tests/forward.rs:817`) checks logits equal eval and that the k_proj gradient has the right shape and is nonzero. The numerical gradcheck (`ojas-autograd/tests/gradcheck.rs:352`) calls `causal_sdpa_backward` directly, not through the Tape. A tape-level GQA gradcheck is tracked in gp-gqa-native-kernels.

## Acceptance criteria
- [x] Remove d <= 128 head dimension limit assertion across Metal and wgpu kernels
- [x] Implement tiled attention forward and backward kernels supporting head dimension 256
- [x] Generalize Backend::causal_sdpa_forward and backward to support Grouped-Query Attention (mismatched Q and KV head counts)
- [x] Autograd Tape records and backpropagates GQA projections correctly
- [x] Unit tests verifying head dimensions 64, 128, and 256 match analytical and CPU reference oracles

## Planned files
- ojas-core/src/backend.rs
- ojas-kernels/src/geometry.rs
- ojas-metal/src/device.rs
- ojas-metal/kernels/ojas_backend.metal
- ojas-wgpu/src/backend.rs
- ojas-autograd/src/tape.rs
