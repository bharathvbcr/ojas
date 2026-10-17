---
id: "gp-ft-step-memory-and-throughput"
title: "LoRA step cost, the items that decide D1: finite checks on frozen weights once per load, no dense embedding gradient, dW skipped for frozen linears, and the Tape-vs-provider tokens/s comparison at T=2048 and T=9,638"
status: ready
priority: 1
severity: high
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
Status: ready
Priority: 1 (High)
Severity: high
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
- Status: backlog, not started; ~~one item half done [A]~~ (corrected 2026-10-17 at 5b67cdd: the fused-CE half is done, the rest is open; the board has the card at ready).
- Criteria: the interleaved 2B A/B rows are open (only frozen_linear_bench.rs, T = 2,048, ignored, nothing committed) [A]; finite checks on frozen weights still run per call (device.rs:1410, 1505, 1534, 1568, 2850) [A]; no embedding gradient for a frozen table is done (tape.rs:1173-1175, frozen.rs:189) [A], ~~the fused-CE half is open [V] (tape.rs:955-990: want_grad always true, both gradients required)~~ (corrected 2026-10-17 at 5b67cdd: the fused-CE half is done, Tape::fused_ce is tape.rs:968 with per-gradient flags through CeGrads, ojas-core/src/backend.rs:1748); the dW-skip share of a 2B step and the Tape-vs-provider tokens/s (the D1 metric) are unmeasured [A].
- Fix seam for the fused-CE half [A]: a per-gradient want flag on `Backend::linear_cross_entropy_mean`, honoured by CPU and Metal, so a frozen head allocates no [V, d] gradient (2.03 GB at 2B, held from forward to backward) and runs no gw GEMM (~9.8 TFLOP per step at T = 9,638 [I]).
- Stale: "embedding backward writes a dense ~2 GB gradient" is false for a frozen table; the "~22 GB re-read per micro-step" figure is consistent (~20.5 GB) but at ~300 GB/s is under 0.1 s of a ~30 s step [I], so its rank as a D1 decider is unsupported.
- Acceptance path: the fused-CE half should land before the long run (2 GB and a quarter of the head's GEMM work every step). The D1 tokens/s reading is owed: the provider's 452 tok/s at T = 8,192 (Lappi) against the Tape's full-gradient lower bound of ~279 tok/s at T = 9,638 (from wall_ms, which includes load) leaves the 0.5 reversal threshold close [I].

### Fable ruling (2026-10-17)

Synced from the GitPulse board by the 2026-10-17 brief sync; the board is canonical (Fable's final ruling of 2026-10-17, section 5), this section is the long-form record. Board card ft-2bbced18f11d70727f58e25696dcd5ed, revision 6: status ready, priority 1, severity high. Criterion numbers are the board's, counted from 0 in the order of the card's acceptance criteria.

Ruling set: status backlog -> ready, priority 1, severity medium -> high (criterion 7 is now the next GPU job on the track and gates amendment 1).

Marks (the text of the existing criterion is otherwise unchanged on the board; nothing deleted, no threshold moved):

- ft-2bbced18f11d70727f58e25696dcd5ed criterion 4: [NOTE 2026-10-17: decided by criterion 7's row against the provider's 452 tok/s at T=8192 (Lappi AUDIT/ojas-training-2026-10-01/mac-8k-train-step-bench.log:35); the Tape's full-gradient 233-281 tok/s at T=9,638 already exceeds half of 452, so the D1 reversal is not triggered by the measured upper bound; the provider bf16 T=2048 row was SIGTERMed by memguard and is not required]
- ft-2bbced18f11d70727f58e25696dcd5ed criterion 8: [SUPERSEDED 2026-10-17 for this track: the admitted quantised base is a fake-quant bf16 file for the first runs, so its rows equal the bf16 rows by construction; resident-q8 rows belong to the stage-2 card]

Criteria appended by the ruling, verbatim:

- [ ] ft-2bbced18f11d70727f58e25696dcd5ed criterion 9: (Fable 2026-10-17) Criterion 7 does not wait for ojas-train: a tape-lora arm in ojas-qwen35/tests/gpu_memory_rows.rs composed from merged primitives (bind_frozen, forward_loss_lora or forward_hidden_lora + gather_rows, r = 16 alpha = 16 on the ten LoraKind suffixes, bf16 linears and bf16 table, per-tensor Backend::adamw_step over the adapters), 2B-Base and 0.8B-Base at T = 2,048 and T = 9,638, each row with exact device peak, first-step wall, and tokens/s as the mean of 3 steps after one warmup step, under a 28 GiB MetalBackend budget and Lappi-decision/tools/mac_heavy.sh, rows under bench/results/. Its tokens/s is the input to amendment 1 (ft-da2c4ab1). Host footprint (memguard's metric) is recorded beside the device peak.

Board criteria and marks already on the card before the ruling (2026-10-17 owner, gap and fold entries, and earlier audit marks), not previously in this brief, verbatim:

- [ ] ft-2bbced18f11d70727f58e25696dcd5ed criterion 6: [w12 fold 2026-10-17, Fable ruling on bundle gp-uma-zero-copy-io-…] Measurement-gated, after G-IOREAD (ft-b301b6d6) lands: measure whether page-aligned mmap wrapped as a Metal shared buffer (newBufferWithBytesNoCopy -> tessl BufferKind::External, tessl/src/tensor.rs:90) removes the host copy of the frozen 2B base on load; build it only if the measured load peak or time justifies it, and record the measurement either way.
- [ ] ft-2bbced18f11d70727f58e25696dcd5ed criterion 7: (gap 2026-10-17) Closes Lappi GAP-OJAS-ADVICE-MAC-8K-FOOTPRINT-UNMEASURED-2026-10-01: the exact device peak of one full LoRA training step (forward, backward, AdamW over trainable entries with both moments, D17 and span head) at Qwen3.5-2B-Base, T=9,638 micro-batch 1, bf16 frozen base, activation checkpointing on and off, measured under a MetalBackend budget of 28 GiB (below the 32 GiB memguard) so a misfit is a typed refusal, not SIGTERM; row under bench/results/. This number decides whether the reduced 2B run in ft-da2c4ab1 can run on the Mac at the pre-registered width.

### LoRA step rows (2026-10-17, branch gp/lora-step-rows)

Criterion 9's `tape-lora` arm and criterion 7's exact peak are measured: `bench/results/2026-10-17-lora-step/` (README, rows.jsonl, env.txt), arm in `ojas-qwen35/tests/gpu_memory_rows.rs`. Rank 16 / alpha 16 on all ten LoraKind suffixes, bf16 base and table, AdamW over the adapters, 28 GiB device budget, checkpointing on: 0.8B-Base 658 tok/s (T = 2,048) and 499 (T = 9,638); 2B-Base 336 and 262 tok/s, device peak 6.65 and 13.10 GB, host footprint max 7.95 and 14.31 GB. 2B at T = 9,638 with checkpointing off is a typed `CapacityExceeded` refusal at the forward under the 28 GiB budget. Amendment 1's N at the 2B T = 9,638 rate is 127 (computed, not written into Lappi). Not measured: the D17 / span-head variant (`forward_hidden_lora` + `gather_rows`); the arm uses the fused all-position head.
