---
id: "gp-ft-quantized-stage2-resident-base"
title: "Stage 2, resident block-quantised frozen base: MLX-exact host quantiser and dequant oracle, U32+bf16 packed storage, dequant-to-bf16-scratch kernel in canonical tessl, loss bit-identical to the fake-quant run; fused dequant GEMM only if profiling says scratch traffic matters"
status: backlog
priority: 2
severity: low
type: feature
owner: "unassigned"
due: "none"
labels:
  - "quantization"
  - "gated"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/quant.rs"
  - "ojas-metal/src/backend.rs"
  - "gemma-metal/src/quant.rs"
acceptance_criteria:
  - "Starts only on the owner's yes after the fake-quant reduced 2B run (ft-da2c4ab1) is scored; the arithmetic it buys is ~1.29 GB at 2B (2.745 GB bf16 wrapped linears vs ~1.46 GB q8 g64), recorded with the ask."
  - "An MLX-exact affine quantiser and dequantiser in ojas-core (edge/side selection, at_zero snapping, eps 1e-7, w_max initialised to 0, round half away from zero, bf16 scale and bias via ojas_core::f32_to_bf16): quantised ints, scales, biases and dequantised values bit-exact against mx.quantize/mx.dequantize golden arrays from the Lappi gate tool for q8 and q4 on the six distinct 2B linear shapes (random, ties, all-negative group, constant group, zero group); NaN/Inf groups, group 0 and cols % group != 0 refused."
  - "Packed storage as a U32 tensor [out, in*bits/32] plus two Bf16 tensors [out, in/64] (no DType variant); load from an MLX-quantised safetensors; Budget-charged."
  - "A dequant_affine_{q8,q4}_bf16 elementwise kernel in canonical tessl (Lappi rule 6) and ojas-metal ops linear_forward_q / linear_backward_input_q (dequant to a reused bf16 scratch, then the existing GEMM); no weight gradient; CPU reference op for the oracle."
  - "Tape loss and adapter gradients over the resident-q8 base are bit-identical to the fake-quant bf16 base on the tiny fixture, CPU and Metal; the 2B rows (ft-2bbced18 criterion 8) report the saving and the dequant share of the step."
  - "A fused dequant GEMM (NT forward, NN dX, M up to 9,638) is built only if the stage-2 profile shows the scratch traffic above a stated share of the step; otherwise recorded as not needed."
  - "gemma-metal's quantiser delegates to the ojas-core owner or is left with its defects documented (bf16 ties-up rounding gemma-metal/src/quant.rs:572-576, silent NaN/Inf groups :452-461); one owner for host quantisation."
---

# Task brief v1

## Title
Stage 2, resident block-quantised frozen base: MLX-exact host quantiser and dequant oracle, U32+bf16 packed storage, dequant-to-bf16-scratch kernel in canonical tessl, loss bit-identical to the fake-quant run; fused dequant GEMM only if profiling says scratch traffic matters

Task: gp-ft-quantized-stage2-resident-base
Type: feature
Status: backlog
Priority: 2 (Normal)
Severity: low
Owner: unassigned
Due: none
Labels: quantization, gated

## Repositories
- ojas

## Description
Filed from Fable's final ruling of 2026-10-17 on the second fine-tune audit (section 4, new card). Split out of ft-a1a4ba4d: this track's first quantised runs train LoRA over a fake-quant bf16 base file, so resident block-quantised storage and its kernels are deferred here, gated on the owner's yes.

Umbrella: ft-da2c4ab1 (LoRA-2B end-to-end). Depends on: ft-a1a4ba4d, the fake-quant reduced 2B run.

Board card: ft-e42793759996e0a6508336b75c499879 (revision 1). The board is canonical; this brief was written from it by the 2026-10-17 brief sync.

## Acceptance criteria
- [ ] Starts only on the owner's yes after the fake-quant reduced 2B run (ft-da2c4ab1) is scored; the arithmetic it buys is ~1.29 GB at 2B (2.745 GB bf16 wrapped linears vs ~1.46 GB q8 g64), recorded with the ask.
- [ ] An MLX-exact affine quantiser and dequantiser in ojas-core (edge/side selection, at_zero snapping, eps 1e-7, w_max initialised to 0, round half away from zero, bf16 scale and bias via ojas_core::f32_to_bf16): quantised ints, scales, biases and dequantised values bit-exact against mx.quantize/mx.dequantize golden arrays from the Lappi gate tool for q8 and q4 on the six distinct 2B linear shapes (random, ties, all-negative group, constant group, zero group); NaN/Inf groups, group 0 and cols % group != 0 refused.
- [ ] Packed storage as a U32 tensor [out, in*bits/32] plus two Bf16 tensors [out, in/64] (no DType variant); load from an MLX-quantised safetensors; Budget-charged.
- [ ] A dequant_affine_{q8,q4}_bf16 elementwise kernel in canonical tessl (Lappi rule 6) and ojas-metal ops linear_forward_q / linear_backward_input_q (dequant to a reused bf16 scratch, then the existing GEMM); no weight gradient; CPU reference op for the oracle.
- [ ] Tape loss and adapter gradients over the resident-q8 base are bit-identical to the fake-quant bf16 base on the tiny fixture, CPU and Metal; the 2B rows (ft-2bbced18 criterion 8) report the saving and the dequant share of the step.
- [ ] A fused dequant GEMM (NT forward, NN dX, M up to 9,638) is built only if the stage-2 profile shows the scratch traffic above a stated share of the step; otherwise recorded as not needed.
- [ ] gemma-metal's quantiser delegates to the ojas-core owner or is left with its defects documented (bf16 ties-up rounding gemma-metal/src/quant.rs:572-576, silent NaN/Inf groups :452-461); one owner for host quantisation.

## Planned files
- ojas-core/src/quant.rs
- ojas-metal/src/backend.rs
- gemma-metal/src/quant.rs
