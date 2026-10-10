---
id: "gp-ft-trainer-loop"
title: "ojas-train: move Lappi qd-train's loop, LRSchedule, parameter groups, host AdamW and StepProvider into ojas now, and make the Qwen3.5 Tape tower a StepProvider with frozen-aware entries"
status: ready
priority: 1
severity: high
type: refactor
owner: "unassigned"
due: "none"
labels:
  - "fine-tune"
  - "trainer"
  - "optimizer"
  - "checkpoint"
  - "refactor"
repositories:
  - "ojas"
  - "Lappi-decision"
planned_files:
  - "ojas-train/"
  - "Cargo.toml"
  - "ojas-model/src/qwen35/forward.rs"
  - "ojas-metal/src/backend.rs"
acceptance_criteria:
  - "A new ojas-train crate holds Lappi crates/qd-train's training loop, LrSchedule, recipe parameter groups, host AdamW and the StepProvider trait, moved with their tests (git history kept; the schedule oracle test moves too). Lappi's qd-train depends on ojas-train and keeps only the rule-3 held-out door, shard reader, supervision, objective, span head, ledger and export; the moved copies are deleted from Lappi in the same change (the user approved the move on 2026-10-09; Fable D8)"
  - "Before the move starts, GitPulse shows no live lane holding Lappi crates/qd-train; if one does, the move waits for it to land (Fable D8 reversal condition)"
  - "StepProvider's ParamSpec gains `trainable` and a storage dtype, and accumulate may take several sequences; the tessl provider (ojas-qwen35) keeps implementing it unchanged in behaviour"
  - "The Qwen3.5 Tape tower implements StepProvider: forward_hidden rows at positions, an external gradient for those rows (the span head), backward into the trainable leaves only (LoRA + D17), grad_sq_norm and AdamW over trainable entries. Today only the provider has this seam (ojas-qwen35/src/step.rs:449-554) [A]"
  - "Through the Tape StepProvider with every entry trainable, a short CPU run on the tiny fixture matches the tessl provider's loss trajectory within a stated bound, and a Metal run matches the CPU run (folded from gp-qwen35-tape-training-loop)"
  - "Memory per step at the tiny and 0.8B shapes is reported with activation checkpointing on and off (folded from gp-qwen35-tape-training-loop)"
  - "AdamW with betas (0.9, 0.999), eps 1e-8, decoupled weight decay 0.01 on every trainable tensor matches torch.optim.AdamW over 100 steps within a stated relative bound on CPU and Metal"
  - "Metal AdamW over the trainable list is all-or-nothing: a non-finite update in tensor k leaves every tensor unstepped (check, then apply). Today each tensor is stepped by its own call (ojas-metal/src/backend.rs:2104, device.rs:2937) [A] (moved here from P0 by Fable)"
  - "A checkpoint holds the trainable entries, their moments, the schedule step, RNG state and the consumed-data digest, and names the frozen base by sha256; resume is bit-exact on CPU and a resumed Metal run equals an uninterrupted one"
  - "Before the first GPU step, a Metal-aware memory pre-flight plans the step at the run's widest bucket and refuses with the shape if it does not fit (the one-heavy-job lock's RSS cap does not see Metal memory; Lappi card ft-0eb828f178f)"
  - "The nanolab Trainer<B> in ojas-model is left alone in this track; a follow-up folds or retires it"
---

# Task brief v1

## Title
ojas-train: move Lappi qd-train's loop, LRSchedule, parameter groups, host AdamW and StepProvider into ojas now, and make the Qwen3.5 Tape tower a StepProvider with frozen-aware entries

Task: gp-ft-trainer-loop
Type: refactor
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: fine-tune, trainer, optimizer, checkpoint, refactor

## Repositories
- ojas
- Lappi-decision

## Description
Filed by the 2026-10-09 fine-tune audit (at d431949); rewritten the same day after the user approved Fable's D8: move the trainer into ojas now, not later. Neither existing loop runs the recipe and they are the same loop twice: ojas-model's Trainer is nanolab-only (AdamW hard-wired, trainer.rs:792 [A]); Lappi's qd_train::trainer has the semantics the recipe needs (per-entry lr_scale and weight decay vectors, bank accumulation, wall-clock cap, checkpoint/resume, consumed digest) but only over the full-FT provider. Writing a third loop for the Tape is the defect the audits name; moving the one that already encodes Lappi's parity is the reuse-first path, and it makes 'trained with ojas' a true sentence.

The schedule is Lappi's LRSchedule (warmup max(1, steps//20), cosine to min_lr 0; Fable D6, user-approved). HF's cosine-with-warmup is not needed for this track. With grad_accum 1 and a token-budget batch (D6), accumulation weighting is moot for the run.

Lappi-side bugs found on the way (--seed not seeding the data order, no --resume, hard-coded lr floor, stale per-layer LR refusal) are on Lappi's board (gp-ft-mac-trainer-blockers) and are fixed during the move, each with a test that fails first.

Depends on: gp-ft-frozen-params-and-lora, gp-ft-decision-head-and-data-path, gp-oom-error-class, gp-device-memory-probes. Umbrella: gp-ft-end-to-end-acceptance.

Folded in on 2026-10-09: gp-qwen35-tape-training-loop (Trainer support for the Qwen3.5 Tape tower, provider-trajectory match, checkpoint round trip, memory per step). Its Muon/AdamW split is dropped for this track: LoRA-2B uses AdamW only, and Lappi's training audit advises against Muon for fine-tuning an Adam-pretrained model (AUDIT/training-audit-2026-10-06.md:469-470).

Labels: [V] re-read at d431949 by the writer of this brief (2026-10-09 fine-tune audit); [A] read by an audit subagent at d431949, not re-read; [U] unverified, needs a run or an upstream source read. No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] A new ojas-train crate holds Lappi crates/qd-train's training loop, LrSchedule, recipe parameter groups, host AdamW and the StepProvider trait, moved with their tests (git history kept; the schedule oracle test moves too). Lappi's qd-train depends on ojas-train and keeps only the rule-3 held-out door, shard reader, supervision, objective, span head, ledger and export; the moved copies are deleted from Lappi in the same change (the user approved the move on 2026-10-09; Fable D8)
- [ ] Before the move starts, GitPulse shows no live lane holding Lappi crates/qd-train; if one does, the move waits for it to land (Fable D8 reversal condition)
- [ ] StepProvider's ParamSpec gains `trainable` and a storage dtype, and accumulate may take several sequences; the tessl provider (ojas-qwen35) keeps implementing it unchanged in behaviour
- [ ] The Qwen3.5 Tape tower implements StepProvider: forward_hidden rows at positions, an external gradient for those rows (the span head), backward into the trainable leaves only (LoRA + D17), grad_sq_norm and AdamW over trainable entries. Today only the provider has this seam (ojas-qwen35/src/step.rs:449-554) [A]
- [ ] Through the Tape StepProvider with every entry trainable, a short CPU run on the tiny fixture matches the tessl provider's loss trajectory within a stated bound, and a Metal run matches the CPU run (folded from gp-qwen35-tape-training-loop)
- [ ] Memory per step at the tiny and 0.8B shapes is reported with activation checkpointing on and off (folded from gp-qwen35-tape-training-loop)
- [ ] AdamW with betas (0.9, 0.999), eps 1e-8, decoupled weight decay 0.01 on every trainable tensor matches torch.optim.AdamW over 100 steps within a stated relative bound on CPU and Metal
- [ ] Metal AdamW over the trainable list is all-or-nothing: a non-finite update in tensor k leaves every tensor unstepped (check, then apply). Today each tensor is stepped by its own call (ojas-metal/src/backend.rs:2104, device.rs:2937) [A] (moved here from P0 by Fable)
- [ ] A checkpoint holds the trainable entries, their moments, the schedule step, RNG state and the consumed-data digest, and names the frozen base by sha256; resume is bit-exact on CPU and a resumed Metal run equals an uninterrupted one
- [ ] Before the first GPU step, a Metal-aware memory pre-flight plans the step at the run's widest bucket and refuses with the shape if it does not fit (the one-heavy-job lock's RSS cap does not see Metal memory; Lappi card ft-0eb828f178f)
- [ ] The nanolab Trainer<B> in ojas-model is left alone in this track; a follow-up folds or retires it

## Planned files
- ojas-train/
- Cargo.toml
- ojas-model/src/qwen35/forward.rs
- ojas-metal/src/backend.rs
