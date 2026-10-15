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

### Fine-tune track audit (2026-10-15, at 88be893)
Read-only re-audit of the fine-tune track at main 88be893. No cargo, GPU or device run was made in this audit.
Labels: [V] re-read by the writer of this note; [A] read by an audit subagent at 88be893, not re-read; [I] inferred.
DevMap (main checkout, generation 6968, fresh) and rg were used; Lappi was read at 4b0b3b8.
- Status: backlog, not started; one item half done [A].
- Criteria: the interleaved 2B A/B rows are open (only frozen_linear_bench.rs, T = 2,048, ignored, nothing committed) [A]; finite checks on frozen weights still run per call (device.rs:1410, 1505, 1534, 1568, 2850) [A]; no embedding gradient for a frozen table is done (tape.rs:1173-1175, frozen.rs:189) [A], the fused-CE half is open [V] (tape.rs:955-990: want_grad always true, both gradients required); the dW-skip share of a 2B step and the Tape-vs-provider tokens/s (the D1 metric) are unmeasured [A].
- Fix seam for the fused-CE half [A]: a per-gradient want flag on `Backend::linear_cross_entropy_mean`, honoured by CPU and Metal, so a frozen head allocates no [V, d] gradient (2.03 GB at 2B, held from forward to backward) and runs no gw GEMM (~9.8 TFLOP per step at T = 9,638 [I]).
- Stale: "embedding backward writes a dense ~2 GB gradient" is false for a frozen table; the "~22 GB re-read per micro-step" figure is consistent (~20.5 GB) but at ~300 GB/s is under 0.1 s of a ~30 s step [I], so its rank as a D1 decider is unsupported.
- Acceptance path: the fused-CE half should land before the long run (2 GB and a quarter of the head's GEMM work every step). The D1 tokens/s reading is owed: the provider's 452 tok/s at T = 8,192 (Lappi) against the Tape's full-gradient lower bound of ~279 tok/s at T = 9,638 (from wall_ms, which includes load) leaves the 0.5 reversal threshold close [I].
