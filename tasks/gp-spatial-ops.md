---
id: "gp-spatial-ops"
title: "Conv2d, depthwise and transposed conv, pooling, interpolation and 2-D RoPE"
status: review
priority: 1
severity: high
type: feature
owner: "ojas-78"
due: "2026-11-02"
labels:
  - "conv2d"
  - "pooling"
  - "interpolation"
  - "rope2d"
  - "vision"
  - "sam3"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/backend.rs"
  - "ojas-cpu/src/conv2d.rs"
  - "ojas-cpu/src/backend.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-metal/kernels/ojas_backend.metal"
  - "ojas-autograd/src/tape.rs"
  - "ojas-oracle/src/parity.rs"
acceptance_criteria:
  - "Conv2d (stride, padding, groups = 1) forward/backward: CPU exact and Metal (im2col + tessl GEMM), torch parity incl. 14x14/14 patch embed"
  - "2-D axial RoPE op (no norm) forward/backward, parity vs a torch reference using SAM 3's PE frequencies once confirmed"
  - "Transposed conv2d stride 2 forward (backward optional), parity vs torch ConvTranspose2d"
  - "Max pool 2x2 and bilinear interpolate (both align_corners settings) forward, parity vs torch"
  - "Depthwise conv2d 7x7 forward, parity vs torch"
  - "Window partition/unpartition and pixel shuffle expressed with permute/views or a single copy op, with tests"
---
# Task brief v1

## Title
Conv2d, depthwise and transposed conv, pooling, interpolation and 2-D RoPE

Task: gp-spatial-ops
Type: feature
Status: review
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: 2026-11-02
Labels: conv2d, pooling, interpolation, rope2d, vision, sam3

## Repositories
- ojas

## Description
Context (2026-10-25): SpikeTrack (Meta hackathon, Oct 26 - Nov 4) runs SAM 3.1 on Apple Silicon. Audit [V]: the `Backend` trait (ojas-core/src/backend.rs) has only `causal_sdpa_*` (Tq = Tk), `linear_forward` without bias, `rms_norm_*`, 1-D `rope_half_split_*` / `rope_partial_*`, and no LayerNorm, GELU, conv2d, pooling or interpolation. SAM 3.1 needs all of these. tessl already has Metal kernels for several (named per item) [V]: wire Metal through them, as Ojas already does for GEMM and attention.

Order of work per op: trait method with a refusing default (the existing `default_framework_ops_refuse` pattern, so wgpu/CUDA/HIP keep compiling) -> CPU exact reference -> Metal -> ojas-oracle parity vs torch -> Tape forward/backward. Inference (forward) is what SpikeTrack needs first; backward is for later LoRA training in Ojas.

Hackathon note: if this is built before the hackathon starts, record it in SpikeTrack's CONTRIBUTIONS.md as pre-existing Ojas work with dates; if built during, list it as hackathon work. Read the Official Rules first.

Spatial ops SAM 3.1 needs [V where noted, else I]:
- **Conv2d via im2col + GEMM**, groups = 1, stride/padding/dilation: patch embed (14x14 stride 14 [V]), 1x1 projections, 3x3 stride-2 convs in the memory encoder's mask downsampler [I].
- **Depthwise conv2d 7x7**: memory-encoder fuser blocks (ConvNeXt-style in SAM 2) [I]. Direct kernel, not GEMM. Ojas has depthwise causal conv1d only [V].
- **Transposed conv2d (stride 2)**: multi-scale neck (x4, x2) and mask upsampling [I]. GEMM + col2im, or GEMM + pixel shuffle when kernel = stride.
- **Max pool 2x2** (neck, x0.5 level) [I].
- **Interpolation**: bilinear up/down (align_corners as the reference uses) for masks and features; input resize must match SAM 3's preprocessing exactly [U] (tessl has a torch-exact antialiased bicubic `resize_rgb_u8` [V] as a pattern).
- **2-D axial RoPE** with a positions buffer: rotate half the channels by x and half by y, every ViT layer [V], also memory attention [I]. tessl `axial_qkv_norm_rope` exists but is fused with RMS QK-norm [V]; Ojas needs a RoPE-only op (forward/backward).
- **Layout helpers**: window partition/unpartition (3x3 of 24x24), pixel shuffle, concat, slice, pad. Prefer composing `permute` + views; add an op only where a copy is unavoidable.

Order: im2col conv2d -> RoPE-2D -> transposed conv -> max pool -> bilinear -> depthwise conv.

### Config facts (facebook/sam3.1 config.json) [V]
- Neck upsamplers are transposed convs with kernel 2, stride 2 (non-overlapping): implement as GEMM + pixel shuffle, no col2im needed. Downsample level is a 2x2 pool.
- Memory encoder: 4 strided 3x3 convs (stride 2, padding 1) and 2 fuser blocks with depthwise 7x7 (padding 3) plus layer scale.
- 2-D RoPE theta 10000, over a 72x72 grid (ViT and memory attention).
## Acceptance criteria
- [ ] Conv2d (stride, padding, groups = 1) forward/backward: CPU exact and Metal (im2col + tessl GEMM), torch parity incl. 14x14/14 patch embed
- [ ] 2-D axial RoPE op (no norm) forward/backward, parity vs a torch reference using SAM 3's PE frequencies once confirmed
- [ ] Transposed conv2d stride 2 forward (backward optional), parity vs torch ConvTranspose2d
- [ ] Max pool 2x2 and bilinear interpolate (both align_corners settings) forward, parity vs torch
- [ ] Depthwise conv2d 7x7 forward, parity vs torch
- [ ] Window partition/unpartition and pixel shuffle expressed with permute/views or a single copy op, with tests

## Planned files
- ojas-core/src/backend.rs
- ojas-cpu/src/conv2d.rs
- ojas-cpu/src/backend.rs
- ojas-metal/src/backend.rs
- ojas-metal/kernels/ojas_backend.metal
- ojas-autograd/src/tape.rs
- ojas-oracle/src/parity.rs
