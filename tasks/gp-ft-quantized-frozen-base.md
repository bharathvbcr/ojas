---
id: "gp-ft-quantized-frozen-base"
title: "8/4-bit frozen base, gated and not built: build only if Lappi's mlx quantization measurement clears its bar and a shape does not fit in bf16; then block-quantized storage, dX-only dequant GEMM in tessl, one host quantizer"
status: backlog
priority: 2
severity: low
type: feature
owner: "unassigned"
due: "none"
labels:
  - "fine-tune"
  - "quantization"
  - "metal"
  - "gated"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/dtype.rs"
  - "ojas-core/src/quant.rs"
  - "ojas-metal/src/backend.rs"
  - "gemma-metal/src/quant.rs"
acceptance_criteria:
  - "Gate, a Lappi-side measurement written before it runs: quantize the bf16 Qwen3.5-2B-Base with mlx.core.quantize (mlx 0.31.2 installed) to affine q8 g64 and q4 g64, and score at least 500 Lappi validation prompts (never held-out; rule 3) with the 17-row letter log-softmax in bf16 vs q8 vs q4. A format is admitted only if argmax agreement >= 99.0% and mean |delta log-prob| on the gold letter <= 0.02 nats (Fable D3)"
  - "A format is built only if it is admitted and a demonstrated memory need exists: a pre-registered shape that does not fit in bf16 at micro-batch 1. 4-bit additionally needs the user to override Unsloth's advice in writing"
  - "If built: a block-quantized weight type in ojas-core (MLX affine q8/q4, group 64, bf16 scale and bias), dequantization bit-exact against mlx.core.dequantize, and a dX-only dequant GEMM on CPU (reference) and Metal (tiled for the run's row counts; kernels in canonical tessl per Lappi rule 6). Today tessl's quantized kernels are inference-only GEMV or M<=8 with no transposed variant [A]"
  - "If built: one owner for host quantization; gemma-metal's quantizer is moved behind ojas-core or replaced, fixing its bf16 rounding (ties up, NaN to Inf; gemma-metal/src/quant.rs:572-576), silent NaN/Inf groups (quant.rs:452-461) and panics on group size 0 (kernels.rs:377) [A]"
  - "8-bit AdamW is not needed (optimizer state covers LoRA, D17 and the span head only)"
---

# Task brief v1

## Title
8/4-bit frozen base, gated and not built: build only if Lappi's mlx quantization measurement clears its bar and a shape does not fit in bf16; then block-quantized storage, dX-only dequant GEMM in tessl, one host quantizer

Task: gp-ft-quantized-frozen-base
Type: feature
Status: backlog
Priority: 2 (Normal)
Severity: low
Owner: unassigned
Due: none
Labels: fine-tune, quantization, metal, gated

## Repositories
- ojas

## Description
Filed by the 2026-10-09 fine-tune audit (at d431949); gated by Fable's ruling D3 the same day. The user asked for a 16-, 8- or 4-bit base; 16-bit is the decided precision (gp-ft-bf16-frozen-base). Unsloth recommends against QLoRA on every Qwen3.5 model ('higher than normal quantization differences' [V fetched]) and memory does not force quantization on this Mac (2B and even 4B fit in bf16 [I]). The quantized path's real cost is weeks of tessl kernel work. The safetensors whole-file dtype refusal, which used to sit here, is now a P0 fix in gp-ft-training-path-correctness.

Depends on: the gate above; gp-imported-research-trees (gemma-metal's future). Umbrella: gp-ft-end-to-end-acceptance.

Labels: [V] re-read at d431949 by the writer of this brief (2026-10-09 fine-tune audit); [A] read by an audit subagent at d431949, not re-read; [U] unverified, needs a run or an upstream source read. No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] Gate, a Lappi-side measurement written before it runs: quantize the bf16 Qwen3.5-2B-Base with mlx.core.quantize (mlx 0.31.2 installed) to affine q8 g64 and q4 g64, and score at least 500 Lappi validation prompts (never held-out; rule 3) with the 17-row letter log-softmax in bf16 vs q8 vs q4. A format is admitted only if argmax agreement >= 99.0% and mean |delta log-prob| on the gold letter <= 0.02 nats (Fable D3)
- [ ] A format is built only if it is admitted and a demonstrated memory need exists: a pre-registered shape that does not fit in bf16 at micro-batch 1. 4-bit additionally needs the user to override Unsloth's advice in writing
- [ ] If built: a block-quantized weight type in ojas-core (MLX affine q8/q4, group 64, bf16 scale and bias), dequantization bit-exact against mlx.core.dequantize, and a dX-only dequant GEMM on CPU (reference) and Metal (tiled for the run's row counts; kernels in canonical tessl per Lappi rule 6). Today tessl's quantized kernels are inference-only GEMV or M<=8 with no transposed variant [A]
- [ ] If built: one owner for host quantization; gemma-metal's quantizer is moved behind ojas-core or replaced, fixing its bf16 rounding (ties up, NaN to Inf; gemma-metal/src/quant.rs:572-576), silent NaN/Inf groups (quant.rs:452-461) and panics on group size 0 (kernels.rs:377) [A]
- [ ] 8-bit AdamW is not needed (optimizer state covers LoRA, D17 and the span head only)

## Planned files
- ojas-core/src/dtype.rs
- ojas-core/src/quant.rs
- ojas-metal/src/backend.rs
- gemma-metal/src/quant.rs
