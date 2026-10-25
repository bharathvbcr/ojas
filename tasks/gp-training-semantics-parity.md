---
id: "gp-training-semantics-parity"
title: "Training semantics that diverge from PyTorch/nanolab: uneven micro-batches weighted per batch not per token, warmup 0 and infinite max_norm refused, all-ignored CE batches fail the step, AdamW beta1 < 0.5 differs on Metal"
status: ready
priority: 2
severity: medium
type: bug
owner: "unassigned"
due: "none"
labels:
  - "trainer"
  - "optimizer"
  - "parity"
  - "semantics"
repositories:
  - "ojas"
planned_files:
  - "ojas-model/src/trainer.rs"
  - "ojas-cpu/src/schedule.rs"
  - "ojas-core/src/backend.rs"
  - "ojas-cpu/src/fused_ce.rs"
  - "ojas-cpu/src/pointwise.rs"
  - "ojas-metal/kernels/ojas_backend.metal"
acceptance_criteria:
  - "A decision per item is recorded in docs/pytorch-parity-plan.md (match torch/nanolab, or keep with the reason), and every test pinning a changed behaviour is updated with that reason"
  - "Gradient accumulation over uneven micro-batches weights tokens equally (seed by valid-token share, as nanolab's equal batches imply), or uneven batches are refused up front; a test with rows 8 and 1 pins it"
  - "warmup_steps = 0 and an infinite max_norm behave as nanolab/torch do (multiplier 1.0; coefficient clamped to 1), or are refused with a message naming the torch behaviour"
  - "An all-ignored cross-entropy micro-batch inside a K > 1 step does not fail the step (contributes 0 with the valid-token weighting above), or the refusal is documented"
  - "AdamW's (1 - beta1) coefficient is computed the same way on CPU, Metal and wgpu; a test with beta1 < 0.5 compares the three bit-for-bit"
  - "grad_clip = 0 is either 'no clipping' (HF max_grad_norm=0 semantics) or refused at config validation; today it is accepted (ojas-model/src/trainer.rs:226) and clip_scale multiplies every gradient by 0 (ojas-core/src/backend.rs:170-174), so the run silently never learns; a test pins it (the Go-side default and docs follow this decision in gp-capi-go-surface)"
  - "A cosine schedule equal to HF get_cosine_schedule_with_warmup at every step (warmup multiplier step/warmup, so step 0 runs at lr 0; decay to 0, not the fixed 0.1 floor) is available, or the nanolab (step+1)/warmup form and 0.1 floor are documented as a deliberate difference; the formula is pinned against the installed transformers 5.19.0 source, not from memory; documentation of the nanolab/HF difference only, not fine-tune work: the fine-tune track uses Lappi's LRSchedule as its oracle (Fable D6)"
  - "A trainable parameter that receives no gradient in a step is skipped, as torch does, instead of failing the step (ojas-model/src/trainer.rs:858-861), so a head or adapter unused by a batch does not abort training"
---

# Task brief v1

## Title
Training semantics that diverge from PyTorch/nanolab: uneven micro-batches weighted per batch not per token, warmup 0 and infinite max_norm refused, all-ignored CE batches fail the step, AdamW beta1 < 0.5 differs on Metal

Task: gp-training-semantics-parity
Type: bug
Status: ready
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: trainer, optimizer, parity, semantics

## Repositories
- ojas

## Description
Filed by the second audit (2026-10-09). Each item is a decision first: match torch/nanolab or keep ojas's rule. Several current behaviours are pinned by tests, so a change must update those tests with the reason. Never weaken them silently.
- **Uneven micro-batches [A]:** every micro-batch is seeded 1/k (ojas-model/src/trainer.rs:739) and the reported loss is loss/k (:768). With rows of 8 and 1, a token in the 1-row batch counts 8x; ignore_index has the same bias. This is the known Hugging Face gradient-accumulation bug. nanolab never sends uneven batches, but ojas's API allows them.
- **warmup_steps = 0 is refused [A, R]:** schedule.rs:39-43, :130-135. nanolab treats warmup <= 0 as a multiplier of 1.0. The refusal is pinned by ojas-cpu/tests/framework_schedule.rs:98 and schedule_train.rs:66.
- **Infinite max_norm is refused [A, R]:** clip_scale(inf, n) returns NonFinite (ojas-core/src/backend.rs:2303-2316). torch clamps the coefficient to 1, the 'measure, don't clip' idiom. max_norm = f32::MAX overflows and fails the step.
- **All-ignored cross-entropy batches and -inf logits are errors [A]:** fused_ce.rs:118-120 and pointwise.rs:2013, 2121-2135. nanolab returns 0 via clamp(min=1); torch returns NaN for all-ignored and handles -inf. With K > 1, one all-padding micro-batch fails the step. Pinned by ojas-cpu/tests/ops.rs:762, ojas-model/tests/trainer.rs:202 and framework_ce.rs:255.
- **AdamW first moment [A, numerically checked by the auditor]:** Metal computes 1 - lerp_w on device (ojas_adamw_elem). CPU and WGSL pass f32(beta1) (ojas-wgpu/src/backend.rs:2393, optim.wgsl:41, ojas-cpu/src/optim.rs:196). For 333 of the 499 beta1 values i/1000 below 0.5 the f32 differs, e.g. beta1 = 0.001 gives 9.99987e-4 against 1.0e-3. Usual betas (>= 0.5) are identical.

Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

### Criteria from the fine-tune track (2026-10-09)
Sent by sibling session ojas-7c (QLoRA/LoRA fine-tune of Lappi on Mac). The code claims were checked by this session [V]: ojas-model/src/trainer.rs:858-861 refuses a trainable parameter with no gradient; ojas-qwen35/src/names.rs:286-296 refuses a tower split across files; clip_scale (ojas-core/src/backend.rs:170-174) returns max_norm/(norm+eps) clamped to 1, so max_norm 0 zeroes every gradient. The HF schedule formula is to be pinned against the installed transformers source, not memory.
Priority raised to 1: this brief blocks a LoRA run that should match HF Trainer.

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Kept P1: it blocks a LoRA run that should match HF Trainer.
- Fine-tune briefs that depend on this one: gp-ft-training-path-correctness, gp-ft-trainer-loop.

### Fine-tune decisions (2026-10-09)
Fable ruled on the fine-tune track's decisions and the user approved them (2026-10-09), as relayed by sibling session ojas-7c. The record is in that session's scratchpad: fable-ft-decisions-2026-10-09.md, sections D1-D6.
- Tape path for LoRA.
- bf16 frozen base; no 4/8-bit.
- 2B-Base, with a 0.8B-Base smoke; 4B is out of the track.
- Lappi's data shape: rows up to 9,638 tokens, a 35,403-token budget, grad_accum 1.
- Lappi's own LRSchedule.
- The trainer moves into a new ojas-train crate.
Checked by this session [V]: the local Qwen3.5-2B snapshot exists (4.3 GB). ojas-qwen35/src/config.rs:503-508 still refuses unequal GDN value and key heads with 'tessl's gdn_train has no head grouping', while ../tessl/src/qwen35_train.rs now groups heads.
- Priority 1 -> 2: the fine-tune track takes Lappi's LRSchedule as its oracle. Lappi's summed gradient accumulation is a documented choice (Lappi crates/qd-train/src/trainer.rs:15-17), not a bug. The remaining items here are parity with torch/nanolab.

## Acceptance criteria
- [ ] A decision per item is recorded in docs/pytorch-parity-plan.md (match torch/nanolab, or keep with the reason), and every test pinning a changed behaviour is updated with that reason
- [ ] Gradient accumulation over uneven micro-batches weights tokens equally (seed by valid-token share, as nanolab's equal batches imply), or uneven batches are refused up front; a test with rows 8 and 1 pins it
- [ ] warmup_steps = 0 and an infinite max_norm behave as nanolab/torch do (multiplier 1.0; coefficient clamped to 1), or are refused with a message naming the torch behaviour
- [ ] An all-ignored cross-entropy micro-batch inside a K > 1 step does not fail the step (contributes 0 with the valid-token weighting above), or the refusal is documented
- [ ] AdamW's (1 - beta1) coefficient is computed the same way on CPU, Metal and wgpu; a test with beta1 < 0.5 compares the three bit-for-bit
- [x] grad_clip = 0 is either 'no clipping' (HF max_grad_norm=0 semantics) or refused at config validation; today it is accepted (ojas-model/src/trainer.rs:226) and clip_scale multiplies every gradient by 0 (ojas-core/src/backend.rs:170-174), so the run silently never learns; a test pins it (the Go-side default and docs follow this decision in gp-capi-go-surface) — refused at validation (ft-e15ee76a 5aa04db, `a_zero_grad_clip_is_refused_at_validation`); `go/api.go` `GradClip` documents it, and the Go default stays 1.0
- [ ] A cosine schedule equal to HF get_cosine_schedule_with_warmup at every step (warmup multiplier step/warmup, so step 0 runs at lr 0; decay to 0, not the fixed 0.1 floor) is available, or the nanolab (step+1)/warmup form and 0.1 floor are documented as a deliberate difference; the formula is pinned against the installed transformers 5.19.0 source, not from memory; documentation of the nanolab/HF difference only, not fine-tune work: the fine-tune track uses Lappi's LRSchedule as its oracle (Fable D6)
- [ ] A trainable parameter that receives no gradient in a step is skipped, as torch does, instead of failing the step (ojas-model/src/trainer.rs:858-861), so a head or adapter unused by a batch does not abort training

## Planned files
- ojas-model/src/trainer.rs
- ojas-cpu/src/schedule.rs
- ojas-core/src/backend.rs
- ojas-cpu/src/fused_ce.rs
- ojas-cpu/src/pointwise.rs
- ojas-metal/kernels/ojas_backend.metal

### Fable ruling (2026-10-17)

Synced from the GitPulse board by the 2026-10-17 brief sync; the board is canonical (Fable's final ruling of 2026-10-17, section 5), this section is the long-form record. Board card ft-9c13f47224dbb204fe9534c264500a5b, revision 5: status ready, priority 2, severity medium. Criterion numbers are the board's, counted from 0 in the order of the card's acceptance criteria.

Ruling set: status ready, priority 2 (unchanged).

Criteria appended by the ruling, verbatim:

- [ ] ft-9c13f47224dbb204fe9534c264500a5b criterion 11: (Fable 2026-10-17) Not on the LoRA-2B path: the Tape StepProvider runs under Lappi's loop (its own clip max_norm/(norm+1e-6), non-finite refusal before AdamW, grad_accum 1), not ojas-model's Trainer; criteria 5, 7 and 10 apply to the nanolab Trainer only.

Board criteria and marks already on the card before the ruling (2026-10-17 owner, gap and fold entries, and earlier audit marks), not previously in this brief, verbatim:

- [ ] ft-9c13f47224dbb204fe9534c264500a5b criterion 8: (audit-2026-10-16) With SkipBatch, N consecutive NonFinite steps (N documented) are followed by a distinct error that does not advance the cursor; a test with `Fault::NanOutput` on every `linear_forward` pins N and the final cursor; either `resume_from` refuses a checkpoint whose `model.safetensors` holds a NaN (test edits one f32 in place, length unchanged) or the decision not to scan is recorded beside load.rs:16 with the reason; `skip_batch_advances_the_cursor_and_abort_keeps_it` is unchanged.
- [ ] ft-9c13f47224dbb204fe9534c264500a5b criterion 9: (audit-2026-10-16) `label_smoothing` and per-class `weight` on cross-entropy match torch `F.cross_entropy` in forward and gradient on CPU (f64 gradcheck, torch golden), with a Metal forward and backward kernel and a Metal-vs-CPU parity test at vocab 10 and 100; the fused linear-CE path either supports them or refuses with a typed error naming the option; Autocast and Gated forward the changed signature (ft-fe7dcee6's completeness tests).
- [ ] ft-9c13f47224dbb204fe9534c264500a5b criterion 10: [w12 fold 2026-10-17, Fable ruling] grad_clip == 0: trainer.rs:226 accepts 0 and clip_scale returns 0 (ojas-core/src/backend.rs:183-187), zeroing every gradient — torch clip_grad_norm_(max_norm=0) does the same; decide against nanolab's meaning of 0 (off vs zero) and document it at both sites with a test pinning the decision.
