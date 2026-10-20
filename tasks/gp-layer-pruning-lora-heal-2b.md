---
id: "gp-layer-pruning-lora-heal-2b"
title: "Layer pruning plus LoRA heal on Qwen3.5-2B: does quality survive, and what does decode gain?"
status: backlog
priority: 3
severity: none
type: research
owner: "unassigned"
due: "none"
labels:
  - "research"
  - "pruning"
  - "lora"
  - "qwen35"
  - "research-2026-10-20"
repositories:
  - "ojas"
acceptance_criteria:
  - "Candidate blocks chosen by angular distance between layer inputs and outputs on a fixed calibration set; the 3:1 GDN:attention pattern constraint is either respected or the deviation recorded"
  - "k in {2, 4, 6} layers removed, each healed with ojas tape-lora (rank and steps pre-registered)"
  - "Quality before and after heal on a fixed eval (perplexity plus the Lappi validation set) and decode tok/s are recorded per k"
  - "A recommendation on whether to try the same on Qwen3.8-27B in MLX is written down, with the k it would start from"
---

# Task brief v1

## Title
Layer pruning plus LoRA heal on Qwen3.5-2B: does quality survive, and what does decode gain?

Task: gp-layer-pruning-lora-heal-2b
Type: research
Status: backlog
Priority: 3
Source: research session 2026-10-20 (LinkedIn article audit, Qwen3.8-27B inference and megakernel research)

## Description
This is a cheap 2B-scale proxy before trying anything on Qwen3.8-27B.

Gromov et al. 2024 report removing up to about half the layers of some models with a QLoRA heal before quality collapses. There is no published Qwen3.x or hybrid DeltaNet result.

Decode bytes scale roughly with layers kept.

## Acceptance criteria
- [ ] Candidate blocks chosen by angular distance between layer inputs and outputs on a fixed calibration set; the 3:1 GDN:attention pattern constraint is either respected or the deviation recorded
- [ ] k in {2, 4, 6} layers removed, each healed with ojas tape-lora (rank and steps pre-registered)
- [ ] Quality before and after heal on a fixed eval (perplexity plus the Lappi validation set) and decode tok/s are recorded per k
- [ ] A recommendation on whether to try the same on Qwen3.8-27B in MLX is written down, with the k it would start from
