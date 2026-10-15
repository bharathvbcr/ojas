---
id: "gp-ft-decision-head-and-data-path"
title: "Lappi's 17-row readout on the Tape: hidden-state forward split from the LM loss, gather at answer positions, 17-way CE with a trainable D17 delta, full-vocab letter CE report-only, right-padding proof, GPT-2 tokenizer refusal"
status: ready
priority: 1
severity: high
type: feature
owner: "unassigned"
due: "none"
labels:
  - "fine-tune"
  - "decision-head"
  - "qwen35"
  - "data"
repositories:
  - "ojas"
planned_files:
  - "ojas-model/src/qwen35/forward.rs"
  - "ojas-model/src/heads.rs"
  - "ojas-autograd/src/tape.rs"
  - "ojas-data/src/bpe.rs"
acceptance_criteria:
  - "forward_loss is split into forward_hidden (final-norm hidden states [B, T, D]) plus heads; the existing LM loss is rebuilt on top and its tests pass unchanged. Today forward_loss is the only entry point and exposes no hidden state (ojas-model/src/qwen35/forward.rs:109) [V]"
  - "Hidden rows can be gathered at per-example positions (the answer position) with backward; parity with torch index_select"
  - "The answer head is a 17-way CE over the 17 answer rows of the frozen tied embedding plus a trainable f32 delta D17 [17, H] initialised to 0: logits = h . (embed[answer_ids] + D17)^T. No gradient flows into the frozen embedding table; parity with torch on loss and dD17, dH (Fable D2, user-approved 2026-10-09)"
  - "The full-vocab letter CE at the same positions is computed without gradient and logged report-only, so every LoRA row carries the v0.1 'letter floor' number"
  - "A right-padded batch gives the same per-example loss and gradients as the same examples run alone, on CPU and Metal (exact because every Qwen3.5 op is causal [I]); left padding and packed multi-document rows are refused, since conv1d/GDN state and attention would leak across them"
  - "The GPT-2 pre-tokenizer refuses a tokenizer that is not GPT-2: load_hf_gpt2 / gpt2_split (ojas-data/src/bpe.rs:500, 582) [V] would turn Qwen's vocabulary into ids that differ from HF's. Refuse when tokenizer.json names another pre_tokenizer (moved here from P0 by Fable: a refusal, not a silent error, because Lappi tokenizes offline)"
  - "A Qwen tokenizer.json reader is out of this track: Lappi's shards are tokenized in Python and reach ojas as token ids (Fable priorities: 'last or never')"
---

# Task brief v1

## Title
Lappi's 17-row readout on the Tape: hidden-state forward split from the LM loss, gather at answer positions, 17-way CE with a trainable D17 delta, full-vocab letter CE report-only, right-padding proof, GPT-2 tokenizer refusal

Task: gp-ft-decision-head-and-data-path
Type: feature
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: fine-tune, decision-head, qwen35, data

## Repositories
- ojas

## Description
Filed by the 2026-10-09 fine-tune audit (at d431949); re-scoped by Fable's ruling the same day (D2 and correction 4). Decided: the new model keeps Lappi's served interface, a softmax over the 17 answer rows (up to 16 letters plus noul) of the tied embedding, trained as a 17-way CE with the rows carried as a trainable delta D17. Serving reads embed[answer ids] + D17 instead of embed[answer ids], an additive change on Lappi's side (qd-metal score_answer_rows, qd-export layout). No Clef hidden-state head for this model. v0.1 trained a 248,320-way CE at the answer position but serves a 17-way softmax; the 17-way objective removes that train/serve mismatch.

Reversal (Fable D2): the 0.8B smoke runs both objectives on the same seed and batch order; if full-vocab beats 17-way on val choice top-1 by more than 2 points and abstains less in-distribution, the pre-registration names full-vocab instead.

The span pointer head stays on the host as today (f32, pinned init) and reaches the Tower through the StepProvider seam in gp-ft-trainer-loop.

Depends on: gp-ft-training-path-correctness (label mask), gp-ft-frozen-params-and-lora. Umbrella: gp-ft-end-to-end-acceptance.

Labels: [V] re-read at d431949 by the writer of this brief (2026-10-09 fine-tune audit); [A] read by an audit subagent at d431949, not re-read; [U] unverified, needs a run or an upstream source read. No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] forward_loss is split into forward_hidden (final-norm hidden states [B, T, D]) plus heads; the existing LM loss is rebuilt on top and its tests pass unchanged. Today forward_loss is the only entry point and exposes no hidden state (ojas-model/src/qwen35/forward.rs:109) [V]
- [ ] Hidden rows can be gathered at per-example positions (the answer position) with backward; parity with torch index_select
- [ ] The answer head is a 17-way CE over the 17 answer rows of the frozen tied embedding plus a trainable f32 delta D17 [17, H] initialised to 0: logits = h . (embed[answer_ids] + D17)^T. No gradient flows into the frozen embedding table; parity with torch on loss and dD17, dH (Fable D2, user-approved 2026-10-09)
- [ ] The full-vocab letter CE at the same positions is computed without gradient and logged report-only, so every LoRA row carries the v0.1 'letter floor' number
- [ ] A right-padded batch gives the same per-example loss and gradients as the same examples run alone, on CPU and Metal (exact because every Qwen3.5 op is causal [I]); left padding and packed multi-document rows are refused, since conv1d/GDN state and attention would leak across them
- [ ] The GPT-2 pre-tokenizer refuses a tokenizer that is not GPT-2: load_hf_gpt2 / gpt2_split (ojas-data/src/bpe.rs:500, 582) [V] would turn Qwen's vocabulary into ids that differ from HF's. Refuse when tokenizer.json names another pre_tokenizer (moved here from P0 by Fable: a refusal, not a silent error, because Lappi tokenizes offline)
- [ ] A Qwen tokenizer.json reader is out of this track: Lappi's shards are tokenized in Python and reach ojas as token ids (Fable priorities: 'last or never')

## Planned files
- ojas-model/src/qwen35/forward.rs
- ojas-model/src/heads.rs
- ojas-autograd/src/tape.rs
- ojas-data/src/bpe.rs

### Fine-tune track audit (2026-10-15, at 88be893)
Read-only re-audit of the fine-tune track at main 88be893. No cargo, GPU or device run was made in this audit.
Labels: [V] re-read by the writer of this note; [A] read by an audit subagent at 88be893, not re-read; [I] inferred.
DevMap (main checkout, generation 6968, fresh) and rg were used; Lappi was read at 4b0b3b8.
- Status: ready; nothing implemented [A]. The label-mask dependency landed (forward_loss takes `ignore`, forward.rs:249).
- Criteria: 1 open: no forward_hidden for Qwen3.5 (h feeds lin_ce inline, forward.rs:~372); DevMap finds forward_hidden only in the GPT block (block.rs:339) [A]; 2 open: no gather op on Graph or Tape; inferred: `Tape::embedding` does not require a leaf table and its backward is a scatter-add (tape.rs:455, :1176), so a hidden-state gather may need no new op [I]; 3 open: no D17 in ojas; the parts exist (a 17-row lin_ce, frozen-gradient drop) and Lappi has AnswerRowDelta (python/qd_train/lora.py:245) [A]; 4 open; 5 partial: a right-pad -100 fixture exists (torch parity only), no alone-equals-batched test, no refusal API [A]; 6 open and mis-specified: load_hf_gpt2 (ojas-data/src/bpe.rs:582) never opens tokenizer.json, so the refusal needs a new probe on both entry points [A]; 7 n/a.
- New, P1 [A]: the StepProvider contract (Lappi crates/qd-train/src/step.rs) cannot carry the 17 answer ids or a report-only full-vocab CE; decide where D17 lives first (host-side like the span head is an option).
- New, P2 [V via gp-ft-frozen-params-and-lora]: the report-only full-vocab CE over a frozen tied table pays the [V, d] head gradient (tape.rs:955-990) until ft-2bbced18 lands.
- Stale: forward.rs:109 is now :241 [A].
- Acceptance path: criteria 1-5 and the contract decision are on it; 6 and 7 can wait (Lappi sends token ids).

### Fable rulings (2026-10-15)
Fable ruled on the audit's open decisions on 2026-10-15; the owner decided two items directly (Lappi pin bump, hand-rolled sha256).
- **D17 lives host-side, like the span head; no StepProvider contract change.** The contract already carries it: a loss outside the model reads the final hidden state at `hidden_at` and returns its gradient through the callback (Lappi crates/qd-train/src/step.rs:21-26), and host entries step on their own count via `adamw_entry_step` (trainer.rs:863-870). D17 is [17, 2048] f32.
  - The host reads `embed_tokens.weight[answer ids]` once from the checkpoint file by name (17 rows, bf16 to f32; never through `read_parameters`, which stages 7.5 GB), digests them, and holds D17 as a host entry. With R = E + D17 and z = h R^T: dz = (softmax(z) - onehot)/N, dh = dz R, dD17 = dz^T h. Where D17 and span-head positions coincide, one HiddenGrad callback pre-sums.
  - The report-only full-vocab letter CE goes through RowTargets with scale = 0 (validate allows >= 0, step.rs:125); a Tape provider may skip the backward of a zero-scale term, documented as an optimisation.
  - The host math is pinned against Lappi's AnswerRowDelta (python/qd_train/lora.py:245-300) with a committed fixture.
