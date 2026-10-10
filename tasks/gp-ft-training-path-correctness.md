---
id: "gp-ft-training-path-correctness"
title: "Fine-tune path bugs before any feature work: Qwen3.5 Tape loss cannot mask labels, save_state drops a half-filled gradient bank, a stale 4B refusal states a false reason, one unknown safetensors dtype rejects the whole file; widen the tiny fixture so torch sees batch, heads and CE tiling"
status: ready
priority: 0
severity: high
type: bug
owner: "unassigned"
due: "none"
labels:
  - "fine-tune"
  - "correctness"
  - "qwen35"
  - "autograd"
  - "safetensors"
repositories:
  - "ojas"
planned_files:
  - "ojas-model/src/qwen35/forward.rs"
  - "ojas-qwen35/src/state.rs"
  - "ojas-qwen35/src/config.rs"
  - "ojas-io/src/safetensors.rs"
  - "ojas-model/tests/"
  - "ojas-oracle/"
acceptance_criteria:
  - "The Qwen3.5 Tape forward_loss takes the same ignore argument as the GPT forward_loss (ojas-model/src/block.rs:430 `ignore: Option<u32>`) and passes it to lin_ce; today ojas-model/src/qwen35/forward.rs:195 passes None [V], so prompt and pad tokens are trained on with no error. A test with -100 labels on prompt and padding matches torch cross_entropy(ignore_index=-100) on CPU and Metal, and fails at d431949"
  - "Qwen35Step::save_state refuses with a typed error while the gradient bank holds a partial accumulation (or persists it and load_state restores it). Today save_state (ojas-qwen35/src/state.rs:613) never checks bank_state [V], so a checkpoint taken mid-accumulation silently loses those gradients on resume. A test covers the chosen outcome"
  - "The GDN head-grouping refusal states a true reason: ojas-qwen35/src/config.rs:503-508 says 'tessl's gdn_train has no head grouping' [V], but tessl now repeats key heads over value heads and sums the gradients back (tessl qwen35_train.rs:23-26) [V by Fable]. Either lift the refusal behind a grouped golden, or keep it with a message naming the real missing piece; a test pins the message. (Lifting it for 4B is gp-qwen35-above-2b's work, dormant in this track)"
  - "ojas-io parses every dtype in the safetensors spec and refuses a tensor only when it is read with an unsupported type; today one unknown dtype rejects the whole file (StDtype is F32/BF16/F16/I64/U16 only, ojas-io/src/safetensors.rs:26-34) [V]. A test reads the bf16 tensors of a file that also holds a U8 tensor"
  - "The tiny Qwen3.5 fixture is widened so the paths above are checked against torch: batch >= 2, >= 2 GDN heads, vocab > 8192 (more than one fused-CE column tile) and -100 labels; today it is batch 1, one GDN head, vocab 64 [A]. This is the fixture Fable's bound (a') is measured on"
  - "Each fix ships with a test that fails against d431949. Moved out by Fable's ruling (2026-10-09): the GPT-2 pre-tokenizer refusal to gp-ft-decision-head-and-data-path, Metal AdamW all-or-nothing to gp-ft-trainer-loop, provider lr_scale-0 freeze semantics to gp-ft-frozen-params-and-lora; trainer-level semantics stay in gp-training-semantics-parity"
---

# Task brief v1

## Title
Fine-tune path bugs before any feature work: Qwen3.5 Tape loss cannot mask labels, save_state drops a half-filled gradient bank, a stale 4B refusal states a false reason, one unknown safetensors dtype rejects the whole file; widen the tiny fixture so torch sees batch, heads and CE tiling

Task: gp-ft-training-path-correctness
Type: bug
Status: ready
Priority: 0 (Urgent)
Severity: high
Owner: unassigned
Due: none
Labels: fine-tune, correctness, qwen35, autograd, safetensors

## Repositories
- ojas

## Description
Filed by the 2026-10-09 fine-tune audit (at d431949); re-scoped the same day by Fable's ruling (Lappi AUDIT/lora-2b-2026-10-09/fable-ft-decisions.md, 'Task priorities'). Goal: fine-tune a new Lappi decision model (LoRA-2B) on the Mac with ojas, frozen bf16 Qwen3.5-2B-Base + LoRA, at parity with a plain-torch LoRA arm of the same recipe.

These are the P0 items: defects that would train or checkpoint the wrong thing, plus two small fixes whose current errors are false or too broad. The op math itself was checked against transformers and found correct: GDN (q/k L2-norm, beta, decay, softplus threshold 20), causal conv tap order, gated RMSNorm, (1+w) RMSNorm, partial RoPE, GQA mapping, fused CE, AdamW bias correction and eps [A].

The other P0 item of the track is the gating GPU run, gp-qwen35-2b-metal-tape-run (D1).

Depends on: gp-oracle-and-hardening-coverage (independent torch goldens). Umbrella: gp-ft-end-to-end-acceptance.

Labels: [V] re-read at d431949 by the writer of this brief (2026-10-09 fine-tune audit); [A] read by an audit subagent at d431949, not re-read; [U] unverified, needs a run or an upstream source read. No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] The Qwen3.5 Tape forward_loss takes the same ignore argument as the GPT forward_loss (ojas-model/src/block.rs:430 `ignore: Option<u32>`) and passes it to lin_ce; today ojas-model/src/qwen35/forward.rs:195 passes None [V], so prompt and pad tokens are trained on with no error. A test with -100 labels on prompt and padding matches torch cross_entropy(ignore_index=-100) on CPU and Metal, and fails at d431949
- [ ] Qwen35Step::save_state refuses with a typed error while the gradient bank holds a partial accumulation (or persists it and load_state restores it). Today save_state (ojas-qwen35/src/state.rs:613) never checks bank_state [V], so a checkpoint taken mid-accumulation silently loses those gradients on resume. A test covers the chosen outcome
- [ ] The GDN head-grouping refusal states a true reason: ojas-qwen35/src/config.rs:503-508 says 'tessl's gdn_train has no head grouping' [V], but tessl now repeats key heads over value heads and sums the gradients back (tessl qwen35_train.rs:23-26) [V by Fable]. Either lift the refusal behind a grouped golden, or keep it with a message naming the real missing piece; a test pins the message. (Lifting it for 4B is gp-qwen35-above-2b's work, dormant in this track)
- [ ] ojas-io parses every dtype in the safetensors spec and refuses a tensor only when it is read with an unsupported type; today one unknown dtype rejects the whole file (StDtype is F32/BF16/F16/I64/U16 only, ojas-io/src/safetensors.rs:26-34) [V]. A test reads the bf16 tensors of a file that also holds a U8 tensor
- [ ] The tiny Qwen3.5 fixture is widened so the paths above are checked against torch: batch >= 2, >= 2 GDN heads, vocab > 8192 (more than one fused-CE column tile) and -100 labels; today it is batch 1, one GDN head, vocab 64 [A]. This is the fixture Fable's bound (a') is measured on
- [ ] Each fix ships with a test that fails against d431949. Moved out by Fable's ruling (2026-10-09): the GPT-2 pre-tokenizer refusal to gp-ft-decision-head-and-data-path, Metal AdamW all-or-nothing to gp-ft-trainer-loop, provider lr_scale-0 freeze semantics to gp-ft-frozen-params-and-lora; trainer-level semantics stay in gp-training-semantics-parity

## Planned files
- ojas-model/src/qwen35/forward.rs
- ojas-qwen35/src/state.rs
- ojas-qwen35/src/config.rs
- ojas-io/src/safetensors.rs
- ojas-model/tests/
- ojas-oracle/
