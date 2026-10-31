---
id: "gp-vision-norm-act-bias"
title: "LayerNorm, exact GELU, linear bias and broadcast elementwise ops"
status: review
priority: 0
severity: critical
type: feature
owner: "ojas-78"
due: "2026-11-02"
labels:
  - "layernorm"
  - "gelu"
  - "bias"
  - "vision"
  - "sam3"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/backend.rs"
  - "ojas-cpu/src/backend.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-metal/kernels/ojas_backend.metal"
  - "ojas-autograd/src/tape.rs"
  - "ojas-oracle/src/parity.rs"
acceptance_criteria:
  - "LayerNorm (weight, bias, eps) forward/backward: CPU exact, Metal via tessl bert kernels, torch parity"
  - "LayerNorm2d over channels of NCHW matches torch reference"
  - "GELU-erf forward/backward: CPU and Metal, max error vs torch recorded; tanh variant not used"
  - "linear_forward with optional bias (fused epilogue on Metal), backward returns dbias"
  - "Broadcast add/mul and scalar scale with backward; shapes refuse rather than silently broadcast wrong"
  - "Standalone last-axis softmax forward/backward"
---
# Task brief v1

## Title
LayerNorm, exact GELU, linear bias and broadcast elementwise ops

Task: gp-vision-norm-act-bias
Type: feature
Status: review
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: 2026-11-02
Labels: layernorm, gelu, bias, vision, sam3

## Repositories
- ojas

## Description
Context (2026-10-25): SpikeTrack (Meta hackathon, Oct 26 - Nov 4) runs SAM 3.1 on Apple Silicon. Audit [V]: the `Backend` trait (ojas-core/src/backend.rs) has only `causal_sdpa_*` (Tq = Tk), `linear_forward` without bias, `rms_norm_*`, 1-D `rope_half_split_*` / `rope_partial_*`, and no LayerNorm, GELU, conv2d, pooling or interpolation. SAM 3.1 needs all of these. tessl already has Metal kernels for several (named per item) [V]: wire Metal through them, as Ojas already does for GEMM and attention.

Order of work per op: trait method with a refusing default (the existing `default_framework_ops_refuse` pattern, so wgpu/CUDA/HIP keep compiling) -> CPU exact reference -> Metal -> ojas-oracle parity vs torch -> Tape forward/backward. Inference (forward) is what SpikeTrack needs first; backward is for later LoRA training in Ojas.

Hackathon note: if this is built before the hackathon starts, record it in SpikeTrack's CONTRIBUTIONS.md as pre-existing Ojas work with dates; if built during, list it as hackathon work. Read the Official Rules first.

Needed by every SAM block [I from ViT/DETR/SAM 2 designs; confirm from the checkpoint]:
- **LayerNorm with weight and bias** over the last axis, eps from config (ViT, DETR, fusion encoder). Also a channels-first `LayerNorm2d` (normalize over C of NCHW) used in necks and the memory encoder [I] - can be permute + LayerNorm.
- **Exact GELU (erf)**: ViT and decoder MLPs. Not the tanh form: it breaks parity.
- **Linear with bias**: `linear_forward` is bias-free today [V, op-coverage.md]; SAM's linears have biases. Either a fused bias epilogue (tessl GEMM supports fused epilogues [V]) or a separate broadcast add.
- **Broadcast elementwise**: add/mul with broadcasting (layer scale, positional embeddings), scalar scale, ReLU (some heads [I]).
- **Softmax over the last axis** as a standalone op (scoring heads, presence) if not covered by attention.

Metal sources [V]: tessl `bert::layer_norm`, `bert::bias_residual_layer_norm`, `bert::bias_gelu_erf` (tessl/src/bert.rs:120-322), tessl `softmax_rows_f32` (nn.rs:4785).

### Config facts [V]
- LayerNorm eps 1e-6 (vision, DETR) and 1e-5 (CLIP text). GELU in ViT, text and memory encoder; ReLU in DETR and memory-attention FFNs. QKV bias on.
## Acceptance criteria
- [ ] LayerNorm (weight, bias, eps) forward/backward: CPU exact, Metal via tessl bert kernels, torch parity
- [ ] LayerNorm2d over channels of NCHW matches torch reference
- [ ] GELU-erf forward/backward: CPU and Metal, max error vs torch recorded; tanh variant not used
- [ ] linear_forward with optional bias (fused epilogue on Metal), backward returns dbias
- [ ] Broadcast add/mul and scalar scale with backward; shapes refuse rather than silently broadcast wrong
- [ ] Standalone last-axis softmax forward/backward

## Planned files
- ojas-core/src/backend.rs
- ojas-cpu/src/backend.rs
- ojas-metal/src/backend.rs
- ojas-metal/kernels/ojas_backend.metal
- ojas-autograd/src/tape.rs
- ojas-oracle/src/parity.rs
