---
id: "gp-ft-training-path-correctness"
title: "Fine-tune path bugs before any feature work: Qwen3.5 Tape loss cannot mask labels, save_state drops a half-filled gradient bank, a stale 4B refusal states a false reason, one unknown safetensors dtype rejects the whole file; widen the tiny fixture so torch sees batch, heads and CE tiling"
status: review
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

## Progress 2026-10-12 (branch gitpulse/ft-training-path-correctness)

Fixture regenerated with `ojas-oracle/python/qwen35_wide_fixture.py` (python3.14, torch 2.13, transformers 5.15; goldens are torch's, not hand-written).

- Tape loss ignore index: DONE. `ojas-model/src/qwen35/forward.rs` `forward_loss` takes `ignore: Option<u32>` (same slot as the GPT one) and passes it to `lin_ce`. Tests (`ojas-model/tests/qwen35_fixture.rs`): `the_cpu_tape_matches_transformers` (CPU) and `qwen35_metal_tape.rs::the_metal_tape_matches_transformers_and_the_cpu_tape` (Metal) run the -100-labelled fixture against torch `cross_entropy(ignore_index=-100)` within 1e-5 loss / 1e-4 gradient; plus `masked_labels_without_an_ignore_index_are_refused`, `only_the_ignore_index_masks_a_position`, `a_batch_with_every_target_ignored_is_refused`. At d431949 the argument does not exist (compile error), and `None` with -100 labels is an out-of-range target.
- save_state with a non-empty bank: DONE, chose refusal. `ojas-qwen35/src/state.rs` `save_state` returns `Qwen35Error::Invalid` for `Holds(_)`/`Invalid(_)`; nothing is written. Test `ojas-qwen35/tests/gpu_save_state_bank.rs` (ignored, Metal).
- Head-grouping refusal message: DONE, kept the refusal, message now names the real missing pieces (ojas `Qwen35Spec` one `gdn_heads`, Tape GDN path no key-head repeat, no grouped torch golden); tessl is not blamed. Test `cpu_config.rs::the_value_head_refusal_does_not_blame_tessl_for_a_missing_grouping`.
- safetensors dtypes: DONE. `StDtype` holds every spec dtype; an unfamiliar-but-spec dtype parses, typed reads of it are refused naming tensor and dtype, a name outside the spec still rejects the file (naming it). Tests `ojas-io/tests/unknown_dtype.rs`.
- Wide fixture: DONE in place (`ojas-model/tests/fixtures/qwen35_tiny`): batch 2, 2 GDN heads, vocab 8320 (> one 8192-column CE tile), -100 labels (prompt 20 in row 0, padding 15 in row 1).
- Moved-out items untouched, as ruled.

### Fine-tune track audit (2026-10-15, at 88be893)
Read-only re-audit of the fine-tune track at main 88be893. No cargo, GPU or device run was made in this audit.
Labels: [V] re-read by the writer of this note; [A] read by an audit subagent at 88be893, not re-read; [I] inferred.
DevMap (main checkout, generation 6968, fresh) and rg were used; Lappi was read at 4b0b3b8.
- Status: merged at b8a53ab [V]. The header says review, the body ready, the board in_progress: all stale. Propose: board done once the two GPU pins below have a recorded run, else review.
- Criteria: 1 met (forward_loss takes `ignore`, forward.rs:248 -> lin_ce :377; transformers parity tests use Some(IGNORE)) [A]; 2 met, but its only pin `gpu_save_state_refuses_a_half_filled_gradient_bank` (ojas-qwen35/tests/gpu_save_state_bank.rs:80) is #[ignore] and GPU-only, no recorded run [A]; 3 met (config.rs:505-512, pinned by cpu_config.rs:374) [A]; 4 met (safetensors.rs:37-57, unknown_dtype.rs:19) [A]; 5 met (2 GDN heads, vocab 8,320 = two 8,192-column CE tiles, batch 2 with -100 labels) [A]; 6 (fails at d431949) unverified.
- Stale: cited lines moved (forward.rs:195 -> :377, state.rs:613 -> :617, config.rs:503-508 -> :505-512, safetensors.rs:26-34 -> :24-57); "batch 1, one GDN head, vocab 64" is false; the Metal gradient bound is METAL_GRAD_BOUND = 2e-4 (qwen35_metal_tape.rs:25), not 1e-4 [A].
- New, P3: the head-grouping refusal says no grouped golden exists; tessl has `tiny_grouped_gdn_step_matches_transformers_autograd` (tessl tests/qwen35_train.rs:326) [A].
- Acceptance path: nothing left in code.

### Fable rulings (2026-10-15)
Fable ruled on the audit's open decisions on 2026-10-15; the owner decided two items directly (Lappi pin bump, hand-rolled sha256).
- **Board: review now, done after a recorded run** of its only criterion 2 pin, `gpu_save_state_refuses_a_half_filled_gradient_bank` (ojas-qwen35/tests/gpu_save_state_bank.rs, #[ignore], a tiny-fixture GPU test), and of gpu_parity.rs:654. A check that has not run is not reported as passed.
- **Recorded run (2026-10-15, main 132b46e, M5 Pro, under mac_heavy, `cargo test --release -p ojas-qwen35 --test gpu_save_state_bank --test gpu_parity -- --ignored --exact ... --test-threads=1`):** `gpu_save_state_refuses_a_half_filled_gradient_bank ... ok` (1 passed); `gpu_tiny_scale_zero_entry_still_feeds_the_norm ... ok` (1 passed, 9 filtered out).
- Board: done.
