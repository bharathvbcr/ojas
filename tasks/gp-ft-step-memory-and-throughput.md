---
id: "gp-ft-step-memory-and-throughput"
title: "LoRA step cost, the items that decide D1: finite checks on frozen weights once per load, no dense embedding gradient, dW skipped for frozen linears, and the Tape-vs-provider tokens/s comparison at T=2048 and T=9,638"
status: backlog
priority: 1
severity: medium
type: performance
owner: "unassigned"
due: "none"
labels:
  - "fine-tune"
  - "performance"
  - "metal"
  - "memory"
  - "qwen35"
repositories:
  - "ojas"
planned_files:
  - "ojas-metal/src/device.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-autograd/src/tape.rs"
  - "ojas-model/src/qwen35/forward.rs"
  - "bench/"
acceptance_criteria:
  - "Every item lands with a committed interleaved A/B, min-of-N benchmark at the Qwen3.5-2B shape, T=2048 and T=9,638 micro-batch 1, under bench/results/, run under Lappi-decision/tools/mac_heavy.sh"
  - "Finite checks on frozen weights run once at load, not on every matmul (about 22 GB re-read per micro-step) [A]"
  - "With a frozen tied embedding, no embedding or tied-head weight gradient is computed or allocated (today the embedding backward writes a dense ~2 GB gradient and the fused CE re-adds the weight gradient per row tile) [A]"
  - "The dW skip from gp-ft-frozen-params-and-lora is measured end to end at the 2B step (its share of the backward is reported, not assumed)"
  - "Tokens/s and peak bytes of one bf16-base LoRA step on the Tape vs the tessl provider's full-FT step at T=2048 are recorded; this is Fable D1's reversal metric (Tape below half the provider's tokens/s means LoRA moves into tessl)"
  - "Tokens/s and peak bytes for ojas, torch MPS (the plain-torch LoRA arm) and mlx-lm LoRA at 0.8B and 2B, T=2048 and T=8192, are committed as comparison rows (not gates; Fable D7)"
---

# Task brief v1

## Title
LoRA step cost, the items that decide D1: finite checks on frozen weights once per load, no dense embedding gradient, dW skipped for frozen linears, and the Tape-vs-provider tokens/s comparison at T=2048 and T=9,638

Task: gp-ft-step-memory-and-throughput
Type: performance
Status: backlog
Priority: 1 (High)
Severity: medium
Owner: unassigned
Due: none
Labels: fine-tune, performance, metal, memory, qwen35

## Repositories
- ojas

## Description
Filed by the 2026-10-09 fine-tune audit (at d431949); split the same day per Fable's priorities into this P1 brief (the costs that decide whether the Tape path holds, D1) and gp-ft-step-throughput-followups (P2). Lappi's measurement on the M5 Pro: 452 tok/s at T=8192 on the provider's full-FT step [V by Fable, AUDIT/ojas-training-2026-10-01/mac-8k-train-step-bench.log:35]; a LoRA epoch per seed is about 175-200 h by arithmetic [I], so the Mac run is a 3-6 h truncated quick run by design.

Depends on: gp-ft-frozen-params-and-lora, gp-ft-bf16-frozen-base, gp-qwen35-2b-metal-tape-run. Related: gp-gpu-step-throughput, gp-device-memory-probes. Umbrella: gp-ft-end-to-end-acceptance.

Labels: [V] re-read at d431949 by the writer of this brief (2026-10-09 fine-tune audit); [A] read by an audit subagent at d431949, not re-read; [U] unverified, needs a run or an upstream source read. No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] Every item lands with a committed interleaved A/B, min-of-N benchmark at the Qwen3.5-2B shape, T=2048 and T=9,638 micro-batch 1, under bench/results/, run under Lappi-decision/tools/mac_heavy.sh
- [ ] Finite checks on frozen weights run once at load, not on every matmul (about 22 GB re-read per micro-step) [A]
- [ ] With a frozen tied embedding, no embedding or tied-head weight gradient is computed or allocated (today the embedding backward writes a dense ~2 GB gradient and the fused CE re-adds the weight gradient per row tile) [A]
- [ ] The dW skip from gp-ft-frozen-params-and-lora is measured end to end at the 2B step (its share of the backward is reported, not assumed)
- [ ] Tokens/s and peak bytes of one bf16-base LoRA step on the Tape vs the tessl provider's full-FT step at T=2048 are recorded; this is Fable D1's reversal metric (Tape below half the provider's tokens/s means LoRA moves into tessl)
- [ ] Tokens/s and peak bytes for ojas, torch MPS (the plain-torch LoRA arm) and mlx-lm LoRA at 0.8B and 2B, T=2048 and T=8192, are committed as comparison rows (not gates; Fable D7)

## Planned files
- ojas-metal/src/device.rs
- ojas-metal/src/backend.rs
- ojas-autograd/src/tape.rs
- ojas-model/src/qwen35/forward.rs
- bench/
