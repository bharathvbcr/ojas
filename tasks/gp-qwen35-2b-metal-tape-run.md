---
id: "gp-qwen35-2b-metal-tape-run"
title: "Run the real Qwen3.5-2B hybrid forward and backward through the Metal Tape, and measure 2B save/load and long-sequence headroom"
status: ready
priority: 0
severity: high
type: test
owner: "unassigned"
due: "none"
labels:
  - "qwen35"
  - "metal"
  - "autograd"
  - "measurement"
  - "fine-tune"
repositories:
  - "ojas"
planned_files:
  - "ojas-qwen35/tests/gpu_tape_2b.rs"
  - "ojas-qwen35/README.md"
  - "bench/results/"
acceptance_criteria:
  - "ojas-qwen35/tests/gpu_tape_2b.rs runs the real Qwen3.5-2B hybrid forward and backward through the Metal Tape (snapshot fetched and its revision recorded), and 2B save/load time and long-sequence peak memory headroom are measured and written into ojas-qwen35/README.md"
  - "The run uses a documented local snapshot instead of skipping when it is absent: today the local Qwen3.5-2B (instruct) snapshot at ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B serves (4.3 GB, single shard, same text config as 2B-Base: hidden 2048, 24 layers, GDN 16/16, attn 8/2, head_dim 256), with its revision and file shas recorded; no download needed"
  - "Gate bound vs the provider at T=128: loss <= 1e-4 rel, grads <= 1e-2 of each tensor's peak; then one forward+backward at T=2048 and at T=9,638, micro-batch 1, checkpointing on, with peak device bytes recorded under bench/results/"
  - "Reversal condition recorded: if the Tape fails the bound on real weights, or (after gp-ft-step-memory-and-throughput) its tokens/s at T=2048 is below half the provider's, LoRA moves into tessl's fused step (Fable D1)"
---

# Task brief v1

## Title
Run the real Qwen3.5-2B hybrid forward and backward through the Metal Tape, and measure 2B save/load and long-sequence headroom

Task: gp-qwen35-2b-metal-tape-run
Type: test
Status: ready
Priority: 0 (Urgent)
Severity: high
Owner: unassigned
Due: none
Labels: qwen35, metal, autograd, measurement, fine-tune

## Repositories
- ojas

## Description
Split out of gp-long-runs-and-quiet-benches (2026-10-09), which inherited it from gp-autograd-and-model-primitives. ojas-qwen35/tests/gpu_tape_2b.rs:96 exists but has never run, and ojas-qwen35/README.md:230-235 says 'Not measured'. Only the config and tokenizer are local, so the Qwen3.5-2B-Base weights must be downloaded first. Queue the run through Lappi-decision/tools/mac_heavy.sh.

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- P1 because this run is the first evidence that the fine-tune path holds at real scale on Metal.
- Fine-tune briefs that depend on this one: gp-ft-end-to-end-acceptance.

### Fine-tune decisions (2026-10-09)
Fable ruled on the fine-tune track's decisions and the user approved them (2026-10-09), as relayed by sibling session ojas-7c. The record is in that session's scratchpad: fable-ft-decisions-2026-10-09.md, sections D1-D6.
- Tape path for LoRA.
- bf16 frozen base; no 4/8-bit.
- 2B-Base, with a 0.8B-Base smoke; 4B is out of the track.
- Lappi's data shape: rows up to 9,638 tokens, a 35,403-token budget, grad_accum 1.
- Lappi's own LRSchedule.
- The trainer moves into a new ojas-train crate.
Checked by this session [V]: the local Qwen3.5-2B snapshot exists (4.3 GB). ojas-qwen35/src/config.rs:503-508 still refuses unequal GDN value and key heads with 'tessl's gdn_train has no head grouping', while ../tessl/src/qwen35_train.rs now groups heads.
- Priority 1 -> 0: this is D1's gate and the first GPU job of the fine-tune track.

## Acceptance criteria
- [ ] ojas-qwen35/tests/gpu_tape_2b.rs runs the real Qwen3.5-2B hybrid forward and backward through the Metal Tape (snapshot fetched and its revision recorded), and 2B save/load time and long-sequence peak memory headroom are measured and written into ojas-qwen35/README.md
- [ ] The run uses a documented local snapshot instead of skipping when it is absent: today the local Qwen3.5-2B (instruct) snapshot at ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B serves (4.3 GB, single shard, same text config as 2B-Base: hidden 2048, 24 layers, GDN 16/16, attn 8/2, head_dim 256), with its revision and file shas recorded; no download needed
- [ ] Gate bound vs the provider at T=128: loss <= 1e-4 rel, grads <= 1e-2 of each tensor's peak; then one forward+backward at T=2048 and at T=9,638, micro-batch 1, checkpointing on, with peak device bytes recorded under bench/results/
- [ ] Reversal condition recorded: if the Tape fails the bound on real weights, or (after gp-ft-step-memory-and-throughput) its tokens/s at T=2048 is below half the provider's, LoRA moves into tessl's fused step (Fable D1)

## Planned files
- ojas-qwen35/tests/gpu_tape_2b.rs
- ojas-qwen35/README.md
- bench/results/

### Fine-tune track audit (2026-10-15, at 88be893)
Read-only re-audit of the fine-tune track at main 88be893. No cargo, GPU or device run was made in this audit.
Labels: [V] re-read by the writer of this note; [A] read by an audit subagent at 88be893, not re-read; [I] inferred.
DevMap (main checkout, generation 6968, fresh) and rg were used; Lappi was read at 4b0b3b8.
- Status: ready; gpu_tape_2b.rs has never run (no result under bench/results/, README unchanged) [A]. Part of the memory half was measured by gpu_memory_rows (bench/results/2026-10-11-bf16-frozen-base, full-gradient Tape rows at T = 2,048 and 9,638) [V].
- Criteria: 1 open (ojas-qwen35/README.md:250-255 still "Not measured: save_state, load_state, Bf16Operands, longer sequences") [V]; 2 partial (snapshot recorded, file shas not) [A]; 3a open (the T = 128 Tape-vs-provider bounds are written, gpu_tape_2b.rs:38-39, never run) [A]; 3b met for the memory figures, full-gradient only [V]; 4 open (no LoRA step to time) [A].
- New, P0 [I]: the gate likely cannot run under the 32 GiB memguard as written. The provider holds 30.13 GB at open (measured) and the test stages a gradient subset through it (gpu_tape_2b.rs:16-18: "about 40 GB with a staged gradient table"); provider rows past T = 2,048 f32 were stopped on 2026-10-11. Options: save the provider reference once to disk from a run that fits, or read entries one at a time; or a user-approved memguard window. User decision.
- New, P1 [A]: a provider 2B save/load row serves no one under D1 (the provider is not the LoRA path); measure adapter save and resume on the Tape instead. User decision.
- New, P2 [A]: gpu_tape_2b samples memory after forward and backward only, where gpu_memory_rows reads the exact peak, and writes nothing to bench/results/.
- Stale: header P0 vs body "P1 because ... Priority 1 -> 0" (:52, :64); "only the config and tokenizer are local" (:47) is false since 2026-10-12 [V]; README:230-235 is now :250-255 [V].
