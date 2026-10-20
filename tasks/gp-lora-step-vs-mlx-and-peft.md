---
id: "gp-lora-step-vs-mlx-and-peft"
title: "Head-to-head LoRA step: ojas tape-lora vs mlx_lm.lora vs PEFT on PyTorch MPS (Qwen3.5-2B-Base)"
status: ready
priority: 2
severity: low
type: bench
owner: "unassigned"
due: "none"
labels:
  - "bench"
  - "lora"
  - "mlx"
  - "torch"
  - "research-2026-10-20"
repositories:
  - "ojas"
planned_files:
  - "bench/results/2026-10-17-lora-step/README.md"
  - "ojas-qwen35/tests/gpu_memory_rows.rs"
acceptance_criteria:
  - "The configuration is written before any run: Qwen3.5-2B-Base, rank 16, alpha 16, all 10 linear suffixes (150 sites), AdamW lr 1e-4 betas 0.9/0.999 wd 0, micro-batch 1, T in {2,048, 9,638}, checkpointing on, same token ids"
  - "mlx_lm.lora (version recorded) is run with the closest equivalent settings; differences that cannot be matched (bf16 vs f32 adapters, checkpointing granularity, fused head) are listed"
  - "PEFT on torch MPS (versions recorded) is run at both T; an out-of-memory or memguard stop is recorded as a result with the allocator message"
  - "Per arm: mean step wall over 3 steps after warm-up, tokens/s by the ojas convention (T−1 positions), device peak where the framework exposes it, and process lifetime-max footprint for all"
  - "Results under bench/results/<date>-lora-vs-mlx-peft/ with a README; every lane where ojas loses is reported as a red row"
---

# Task brief v1

## Title
Head-to-head LoRA step: ojas tape-lora vs mlx_lm.lora vs PEFT on PyTorch MPS (Qwen3.5-2B-Base)

Task: gp-lora-step-vs-mlx-and-peft
Type: bench
Status: ready
Priority: 2
Source: research session 2026-10-20 (LinkedIn article audit, Qwen3.8-27B inference and megakernel research)

## Description
ojas's committed LoRA rows (`bench/results/2026-10-17-lora-step`, merged 4da214b) have no external baseline:
- 2B-Base T=2,048: 6.65 GB, 335.9 tok/s
- 2B-Base T=9,638: 13.10 GB, 261.6 tok/s
- 0.8B-Base T=2,048: 3.94 GB, 658.2 tok/s

Readers of the article will ask "vs MLX?" first. Pre-register the configuration and publish whichever way it lands.

## Acceptance criteria
- [ ] The configuration is written before any run: Qwen3.5-2B-Base, rank 16, alpha 16, all 10 linear suffixes (150 sites), AdamW lr 1e-4 betas 0.9/0.999 wd 0, micro-batch 1, T in {2,048, 9,638}, checkpointing on, same token ids
- [ ] mlx_lm.lora (version recorded) is run with the closest equivalent settings; differences that cannot be matched (bf16 vs f32 adapters, checkpointing granularity, fused head) are listed
- [ ] PEFT on torch MPS (versions recorded) is run at both T; an out-of-memory or memguard stop is recorded as a result with the allocator message
- [ ] Per arm: mean step wall over 3 steps after warm-up, tokens/s by the ojas convention (T−1 positions), device peak where the framework exposes it, and process lifetime-max footprint for all
- [ ] Results under bench/results/<date>-lora-vs-mlx-peft/ with a README; every lane where ojas loses is reported as a red row
