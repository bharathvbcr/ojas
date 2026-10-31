---
id: "gp-bidirectional-attention"
title: "Bidirectional and cross attention with additive bias and key-length masks"
status: review
priority: 0
severity: critical
type: feature
owner: "ojas-78"
due: "2026-11-02"
labels:
  - "attention"
  - "vision"
  - "sam3"
  - "cross-attention"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/backend.rs"
  - "ojas-cpu/src/attn.rs"
  - "ojas-cpu/src/backend.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-autograd/src/tape.rs"
  - "ojas-oracle/src/parity.rs"
  - "ojas-metal/tests/attention_bidir.rs"
acceptance_criteria:
  - "Trait method with refusing default; wgpu/CUDA/HIP compile unchanged"
  - "CPU exact path matches torch `scaled_dot_product_attention` (f64 reference) for: Tq = Tk unmasked, Tq != Tk, additive bias, key_lens, head dims 64 and 256"
  - "Metal forward matches CPU within the existing attention tolerances on the same cases; uses tessl encoder_attn for the unmasked Tq = Tk case"
  - "Windowed ViT case (batch 9 x 576, D 64) and global case (1 x 5,184) benchmarked on M5 Pro, ms recorded"
  - "Backward (dq, dk, dv, dbias) passes gradcheck on CPU; Metal backward may follow after SpikeTrack inference works"
  - "causal_sdpa_* still pass all existing tests"
---
# Task brief v1

## Title
Bidirectional and cross attention with additive bias and key-length masks

Task: gp-bidirectional-attention
Type: feature
Status: review
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: 2026-11-02
Labels: attention, vision, sam3, cross-attention

## Repositories
- ojas

## Description
Context (2026-10-25): SpikeTrack (Meta hackathon, Oct 26 - Nov 4) runs SAM 3.1 on Apple Silicon. Audit [V]: the `Backend` trait (ojas-core/src/backend.rs) has only `causal_sdpa_*` (Tq = Tk), `linear_forward` without bias, `rms_norm_*`, 1-D `rope_half_split_*` / `rope_partial_*`, and no LayerNorm, GELU, conv2d, pooling or interpolation. SAM 3.1 needs all of these. tessl already has Metal kernels for several (named per item) [V]: wire Metal through them, as Ojas already does for GEMM and attention.

Order of work per op: trait method with a refusing default (the existing `default_framework_ops_refuse` pattern, so wgpu/CUDA/HIP keep compiling) -> CPU exact reference -> Metal -> ojas-oracle parity vs torch -> Tape forward/backward. Inference (forward) is what SpikeTrack needs first; backward is for later LoRA training in Ojas.

Hackathon note: if this is built before the hackathon starts, record it in SpikeTrack's CONTRIBUTIONS.md as pre-existing Ojas work with dates; if built during, list it as hackathon work. Read the Official Rules first.

SAM 3.1 uses non-causal attention everywhere [V paper/arch]:
- **Self-attention, Tq = Tk, no mask:** 32 ViT layers (head dim expected 64 [I]); 28 run over 3x3 windows of 24x24 tokens = a batch of 9 sequences of 576, 4 run globally over 5,184 tokens.
- **Cross-attention, Tq != Tk:** fusion encoder (image tokens -> prompt tokens), DETR decoder (queries -> 1,296 image tokens), memory attention (frame tokens -> memory of up to 7 frames + object pointers), SAM 2-style two-way mask decoder (tokens <-> image, both directions).
- **Additive bias on the logits:** the DETR decoder's box-aware relative position bias (boxRPB) adds a per-head `[Tq, Tk]` bias [V].
- **Key-length masks:** text prompts are up to 32 tokens, padded.

Proposed trait (sketch): `sdpa_forward(q [B,Hq,Tq,D], k,v [B,Hkv,Tk,D], opts { causal: bool, window: Option<usize>, bias: Option<&Tensor> /*[B|1,H|1,Tq,Tk]*/, key_lens: Option<&Tensor> }) -> (out, lse)` and matching `sdpa_backward`. Keep `causal_sdpa_*` as wrappers so existing callers do not change.

Metal: tessl `embedgemma2::encoder_attn(_bf16)` already does bidirectional attention over right-padded batches with per-sequence lengths, TensorOps tiled for head dim 64/256/512 [V, tessl/src/embedgemma2.rs:197-330]. It does not take Tq != Tk or a bias [I]; add those in an Ojas kernel (rows kernel first, tiled later) rather than editing tessl.

### Head dims SAM 3.1 actually uses (config.json) [V]
- 64: ViT (1024 / 16) and CLIP text encoder.
- 32: DETR encoder/decoder and geometry encoder (256 / 8).
- 16: SAM mask decoder cross-attention (256 / 2 downsample / 8 heads).
- 256: memory attention (256, 1 head).
tessl's tiled encoder kernels cover 64, 256, 512 [V], so 32 and 16 need a rows/simdgroup kernel in Ojas. They are small (200 queries, 256-wide), so a simple kernel is fine.
## Acceptance criteria
- [ ] Trait method with refusing default; wgpu/CUDA/HIP compile unchanged
- [ ] CPU exact path matches torch `scaled_dot_product_attention` (f64 reference) for: Tq = Tk unmasked, Tq != Tk, additive bias, key_lens, head dims 64 and 256
- [ ] Metal forward matches CPU within the existing attention tolerances on the same cases; uses tessl encoder_attn for the unmasked Tq = Tk case
- [ ] Windowed ViT case (batch 9 x 576, D 64) and global case (1 x 5,184) benchmarked on M5 Pro, ms recorded
- [ ] Backward (dq, dk, dv, dbias) passes gradcheck on CPU; Metal backward may follow after SpikeTrack inference works
- [ ] causal_sdpa_* still pass all existing tests

## Planned files
- ojas-core/src/backend.rs
- ojas-cpu/src/attn.rs
- ojas-cpu/src/backend.rs
- ojas-metal/src/backend.rs
- ojas-metal/src/device.rs
- ojas-autograd/src/tape.rs
- ojas-oracle/src/parity.rs
- ojas-metal/tests/attention_bidir.rs
