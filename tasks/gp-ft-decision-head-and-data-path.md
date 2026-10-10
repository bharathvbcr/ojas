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
