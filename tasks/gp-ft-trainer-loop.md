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
  - "Metal AdamW over the trainable list is all-or-nothing: a non-finite update in tensor k leaves every tensor unstepped (check, then apply). Today each tensor is stepped by its own call (~~ojas-metal/src/backend.rs:2104, device.rs:2937~~ ojas-metal/src/backend.rs:2270-2300 and device.rs:3277-3287, corrected 2026-10-17 at 5b67cdd; the old sites are linear_cross_entropy_mean) [A] (moved here from P0 by Fable)"
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
- [ ] Metal AdamW over the trainable list is all-or-nothing: a non-finite update in tensor k leaves every tensor unstepped (check, then apply). Today each tensor is stepped by its own call (~~ojas-metal/src/backend.rs:2104, device.rs:2937~~ ojas-metal/src/backend.rs:2270-2300 and device.rs:3277-3287, corrected 2026-10-17 at 5b67cdd; the old sites are linear_cross_entropy_mean) [A] (moved here from P0 by Fable)
- [ ] A checkpoint holds the trainable entries, their moments, the schedule step, RNG state and the consumed-data digest, and names the frozen base by sha256; resume is bit-exact on CPU and a resumed Metal run equals an uninterrupted one
- [ ] Before the first GPU step, a Metal-aware memory pre-flight plans the step at the run's widest bucket and refuses with the shape if it does not fit (the one-heavy-job lock's RSS cap does not see Metal memory; Lappi card ft-0eb828f178f)
- [ ] The nanolab Trainer<B> in ojas-model is left alone in this track; a follow-up folds or retires it

## Planned files
- ojas-train/
- Cargo.toml
- ojas-model/src/qwen35/forward.rs
- ojas-metal/src/backend.rs

### Fine-tune track audit (2026-10-15, at 88be893)
Read-only re-audit of the fine-tune track at main 88be893. No cargo, GPU or device run was made in this audit.
Labels: [V] re-read by the writer of this note; [A] read by an audit subagent at 88be893, not re-read; [I] inferred.
DevMap (main checkout, generation 6968, fresh) and rg were used; Lappi was read at 4b0b3b8.
- Status: ready; no ojas-train crate (not in Cargo.toml members) [A].
- Superseded in part [V]: Lappi's AUDIT/ojas-train-move-2026-10-14/ (fable-ruling.md, delta-report.md, host-adamw-contingency.md) re-scopes the move: 8 modules (step, trainer, schedule, recipe, adamw, mock, run_control, pyjson) plus ojas-qwen35's groups.rs with OptimizerPlan as the one per-entry table; sha256 hand-rolled unless the owner approves `sha2` (unanswered); pyjson onto ojas_io::JsonValue; edition 2024 [A]. Fold the ruling into this brief's criteria.
- Criteria: 1 open; 2 partial (the HANDOFF records one peer cleared; live lanes not re-checked); 3 open and mis-premised: ojas-qwen35 does not implement StepProvider, Lappi's qd-train-metal/src/provider.rs:297 does, so the adapter changes in the same step [A]; 4 open, seam cites moved (forward :535, backward :576, grad_sq_norm :668, adamw_step :693) and now feasible since LoRA merged at efa507f [A]; 5, 6, 9 open (no sha256 in ojas, ojas-data/src/rng.rs:5-6) [A]; 7 partial (one-step torch check only) [A]; 8 open: Metal AdamW still checks then writes per tensor (backend.rs:2266-2297); the brief's :2104 and device.rs:2937 now point at linear_cross_entropy_mean [A]; 10 partial (MemoryProbe, ResourcePlan and check_device_working_set exist; no planner for a Tape step) [A]; 11 holds [A].
- Stale [A]: the "Lappi-side bugs fixed during the move" are already fixed in Lappi (--seed seeds the data order, --resume exists, --min-lr wired: qd-train-metal/src/args.rs:102-121, run.rs:237, :261). ~~Lappi pins ojas at bc498c4, 35 commits behind main~~ (corrected 2026-10-17 at 5b67cdd: the Lappi-decision working tree's upstream.lock.json reads ojas 132b46ea63e87868f468c016e02fe99acf2bb164, read 2026-10-17; whether that is committed on Lappi's main is not verified here); the HANDOFF start sha 30a069d has drifted (6f88a00).
- New, P2 [A]: groups.rs imports ojas-qwen35's TensorSpec and Qwen35Error, and that crate is macOS-only (lib.rs:25); moving it needs a decoupled table type. P2 [A]: ojas-cuda/src/step.rs:285 already defines a different StepProvider and a second AdamWHyper. P3 [I]: no design for summing LoRA gradients across tapes for a 35,403-token batch of widths up to 9,638.

### Fable rulings (2026-10-15)
Fable ruled on the audit's open decisions on 2026-10-15; the owner decided two items directly (Lappi pin bump, hand-rolled sha256).
- **Criterion 4:** D17 is a host entry (see gp-ft-decision-head-and-data-path), not a Tape leaf.
- **sha256 is hand-rolled (owner, 2026-10-15); no sha2 dependency.** sha256.rs in ojas-train with FIPS 180-4 vectors and a parity pin against the pre-registration's recorded `safetensors_sha256` of the real 2B-Base file (Lappi campaign/lora-2b-preregistered.json).
- **Criterion 9 gains the adapter checkpoint save and resume on the Tape** that replaces the provider's retired 2B save/load row (gp-qwen35-2b-metal-tape-run).
- **Lappi pin bumped (owner, 2026-10-15):** bc498c4 -> 132b46e (ojas main, 38 commits), Lappi commit 21fb7ad on branch ojas-pin-88be893, not merged in Lappi. Verified: make pin PASS, qd-train-metal and qd-preflight clippy and tests, gpu_tiny --ignored 10/10. 132b46e is not on origin until ojas main is pushed. The later bump to the ojas-train sha is still owed.

### Fable ruling (2026-10-17)

Synced from the GitPulse board by the 2026-10-17 brief sync; the board is canonical (Fable's final ruling of 2026-10-17, section 5), this section is the long-form record. Board card ft-b5e64557f88bdb433700db571b7a78ec, revision 7: status ready, priority 1, severity high. Criterion numbers are the board's, counted from 0 in the order of the card's acceptance criteria.

Ruling set: status ready, priority 1, severity high (unchanged).

Marks (the text of the existing criterion is otherwise unchanged on the board; nothing deleted, no threshold moved):

- ft-b5e64557f88bdb433700db571b7a78ec criterion 7: [STALE CITE 2026-10-17: per-tensor check-then-write is now ojas-metal/src/backend.rs:2270-2300 (device.rs:3277-3287); the :2104/:2937 sites are linear_cross_entropy_mean]

Criteria appended by the ruling, verbatim:

- [ ] ft-b5e64557f88bdb433700db571b7a78ec criterion 15: (Fable 2026-10-17) The Tape StepProvider's accumulate takes one sequence per call, as Lappi's loop feeds it (crates/qd-train/src/trainer.rs:745-780: BankMode::Overwrite for the first sequence of a step, Add for the rest): the adapters' gradients and the host entries' (D17, span head) are summed in an f32 bank across the sequences of one token-budget batch (35,403 padded tokens: rows of one bucket width, widths up to 9,638), one tape per sequence, no activation kept across sequences, so the step's device peak is the widest sequence's, not the batch's. A test with a 3-row batch of unequal widths matches a torch run summing per-row backward() calls within the (a') bound.
- [ ] ft-b5e64557f88bdb433700db571b7a78ec criterion 16: (Fable 2026-10-17) RowTargets with scale 0 (the report-only full-vocab letter CE) returns the value, runs no backward and allocates no [V, d] gradient; a Budget-delta test pins it (the Tape fused_ce has no value-only call today: ojas-model/tests/qwen35_common/readout.rs records it through lin_ce).
- [ ] ft-b5e64557f88bdb433700db571b7a78ec criterion 17: (Fable 2026-10-17) Criterion 7's all-or-nothing is a two-pass update inside the Tape StepProvider's adamw_step over the trainable list (check every tensor's step finite, then apply every one), not a change to Backend::adamw_step's per-tensor contract; a test injects a non-finite gradient in tensor k > 0 and asserts every tensor and every moment unchanged.
- [ ] ft-b5e64557f88bdb433700db571b7a78ec criterion 18: (Fable 2026-10-17) Criterion 8's RNG state is the data/batch-order RNG and the consumed digest only: the recipe's LoRA dropout is 0 (Lappi campaign/lora-2b-preregistered.json recipe.lora.dropout), so no dropout key is checkpointed, and a non-zero dropout refuses to checkpoint until a key is stored. The checkpoint layout (tensor names, shapes, dtypes, base sha256, step, moments) is documented and versioned in ojas so Lappi's converter (ft-58e316a9) reads it by contract.
- [ ] ft-b5e64557f88bdb433700db571b7a78ec criterion 19: (Fable 2026-10-17) Criterion 14 is narrowed for this track: the quantised arm's first runs load a fake-quant bf16 base file (ft-a1a4ba4d), so ParamSpec's storage dtype names bf16 and the checkpoint names the base by that file's sha256 plus the quantizer record (format, group, mlx version) copied from the arm's pre-registration; the block-quantised storage dtype lands with the stage-2 card.

Board criteria and marks already on the card before the ruling (2026-10-17 owner, gap and fold entries, and earlier audit marks), not previously in this brief, verbatim:

- [ ] ft-b5e64557f88bdb433700db571b7a78ec criterion 11: (audit-2026-10-16) A toy image classifier (MLP today; CNN/ViT once the vision cards land) trains on Metal through ojas-train's loop and StepProvider; its params and AdamW moments round-trip through checkpoint and resume, and the next-step loss is bit-identical on CPU and on Metal.
- [ ] ft-b5e64557f88bdb433700db571b7a78ec criterion 12: (audit-2026-10-16) A `[Cout, Cin, k, k]` parameter can be assigned to AdamW or SGD without inferring Muon from rank (group-assignment test); constant, linear and step schedules exist with tests pinning the multiplier at step 0, the warmup end and the total.
- [ ] ft-b5e64557f88bdb433700db571b7a78ec criterion 13: (gap 2026-10-17) Adapter checkpoint on the Tape, replacing the retired provider 2B save/load row (Fable 2026-10-15): lora_A/lora_B for every adapted linear, D17 and the span head, with their AdamW moments, are written and read by name (today no adapter read or write exists in ojas src: rg -uu lora_A|lora_B finds only a doc comment, lora.rs:289); a kill -9 mid-run at the 2B-Base shape resumes to the same loss trajectory as an unkilled run. Blocks the reduced 2B run in ft-da2c4ab1.
- [ ] ft-b5e64557f88bdb433700db571b7a78ec criterion 14: (owner 2026-10-17, quantised arm) ParamSpec's storage dtype covers the block-quantised frozen type from ft-a1a4ba4d (format, group size, scale/bias dtype). A checkpoint of the quantised arm names its frozen base by the quantised tensors' sha256 plus the quantizer config, and resume refuses a base that differs.
