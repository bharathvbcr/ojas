---
id: "gp-ft-export"
title: "LoRA-2B release export: merged dense bf16 tower plus D17 and span head through Lappi's ckpt_average and qd-export; seed averaging on merged weights only; PEFT-format adapters for the reference arm"
status: backlog
priority: 1
severity: medium
type: feature
owner: "unassigned"
due: "none"
labels:
  - "fine-tune"
  - "export"
  - "lora"
  - "safetensors"
repositories:
  - "ojas"
  - "Lappi-decision"
planned_files:
  - "ojas-model/src/lora.rs"
  - "ojas-model/src/export.rs"
  - "ojas-io/src/safetensors.rs"
acceptance_criteria:
  - "The merged tower is W' = W + (alpha/r) B A computed in f32 and rounded to bf16 once (RNE), into the same bf16 W the adapters trained over; the manifest records the base's sha256. transformers loads the merged tower and its logits match ojas within a stated bf16 bound"
  - "D17 and the span head export as documented safetensors tensors (D17 f32 or bf16, stated; span head f32) in the layout Lappi's qd-export and qd-metal read; qd-metal's change is additive (embed[answer ids] + D17)"
  - "Seed averaging averages merged weights (equivalently (alpha/r) B A, since the base is shared); averaging A and B separately is refused with a typed error (mean(B) mean(A) != mean(B A)). Lappi's ckpt_average learns the merged-weights mode (Lappi side)"
  - "Adapters (+ D17 + span head) are the checkpoint and resume unit and also export in PEFT's format: adapter_model.safetensors and adapter_config.json [V by Fable from peft/utils/constants.py]; the adapter key prefix and config fields are read from a pinned peft source before coding [U]"
  - "A PEFT round-trip test (export, load with peft, same logits) runs only if the user installs peft; until then it is reported as not run, never as passed"
---

# Task brief v1

## Title
LoRA-2B release export: merged dense bf16 tower plus D17 and span head through Lappi's ckpt_average and qd-export; seed averaging on merged weights only; PEFT-format adapters for the reference arm

Task: gp-ft-export
Type: feature
Status: backlog
Priority: 1 (High)
Severity: medium
Owner: unassigned
Due: none
Labels: fine-tune, export, lora, safetensors

## Repositories
- ojas
- Lappi-decision

## Description
Filed by the 2026-10-09 fine-tune audit (at d431949); updated by Fable's ruling D5 the same day. Decided: the release is a merged dense bf16 tower plus D17 and the span head, through Lappi's ckpt_average -> qd-export -> qd-metal, which read dense BF16/F16/F32 only. Adapters are the training artifact, not the served one. With a bf16 base (D3) the merge target is the very W the adapters trained over, so the quantized-base merge trap does not arise; the manifest still records the base hash.

Depends on: gp-ft-frozen-params-and-lora, gp-ft-decision-head-and-data-path. Umbrella: gp-ft-end-to-end-acceptance.

Labels: [V] re-read at d431949 by the writer of this brief (2026-10-09 fine-tune audit); [A] read by an audit subagent at d431949, not re-read; [U] unverified, needs a run or an upstream source read. No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] The merged tower is W' = W + (alpha/r) B A computed in f32 and rounded to bf16 once (RNE), into the same bf16 W the adapters trained over; the manifest records the base's sha256. transformers loads the merged tower and its logits match ojas within a stated bf16 bound
- [ ] D17 and the span head export as documented safetensors tensors (D17 f32 or bf16, stated; span head f32) in the layout Lappi's qd-export and qd-metal read; qd-metal's change is additive (embed[answer ids] + D17)
- [ ] Seed averaging averages merged weights (equivalently (alpha/r) B A, since the base is shared); averaging A and B separately is refused with a typed error (mean(B) mean(A) != mean(B A)). Lappi's ckpt_average learns the merged-weights mode (Lappi side)
- [ ] Adapters (+ D17 + span head) are the checkpoint and resume unit and also export in PEFT's format: adapter_model.safetensors and adapter_config.json [V by Fable from peft/utils/constants.py]; the adapter key prefix and config fields are read from a pinned peft source before coding [U]
- [ ] A PEFT round-trip test (export, load with peft, same logits) runs only if the user installs peft; until then it is reported as not run, never as passed

## Planned files
- ojas-model/src/lora.rs
- ojas-model/src/export.rs
- ojas-io/src/safetensors.rs

### Fine-tune track audit (2026-10-15, at 88be893)
Read-only re-audit of the fine-tune track at main 88be893. No cargo, GPU or device run was made in this audit.
Labels: [V] re-read by the writer of this note; [A] read by an audit subagent at 88be893, not re-read; [I] inferred.
DevMap (main checkout, generation 6968, fresh) and rg were used; Lappi was read at 4b0b3b8.
- Status: backlog, P2; nothing on the ojas side [A]. On the acceptance path earlier than its placement: the reduced 2B row exports through this card.
- Criteria: 1 partial: LoraLinear::merge folds in f32 and stores f32 (lora.rs:435-474), no bf16 round, no manifest, no ojas-model/src/export.rs, nothing maps split q/k/v back to the fused HF names [A]; 2 open on the ojas side (Lappi's qd-export/src/layout.rs:38-45 and qd-metal weights.rs:270 exist) [A]; 3 met on the Lappi side (merged-weight averaging, tools/ckpt_average.py) [A]; 4 open; 5 n/a.
- New, P1 [V]: the single-seed route. The Mac row is one seed, D17 is part of the served model, Lappi's Metal export contract carries only tower.* and span_head.* (crates/qd-train/src/export.rs:1-20), and ckpt_average needs at least 2 inputs (tools/ckpt_average.py:1585). Add a criterion: a one-seed LoRA + D17 export the scorer reads (or a merged-only fallback named in the pre-registration).
- Stale [A]: ojas-io already has a streaming writer (write_safetensors, safetensors.rs:437) and parses every dtype; the planned edit there is likely unnecessary.

### Fable rulings (2026-10-15)
Fable ruled on the audit's open decisions on 2026-10-15; the owner decided two items directly (Lappi pin bump, hand-rolled sha256).
- **New criterion (one-seed route):** the one-seed Metal export carries merged bf16 `tower.*`, f32 `span_head.*` and `d17.delta` [17, H] f32, with `n_tensors` counting it and a `d17` manifest block; Lappi's `_metal_export_weights` / `read_average` accept the optional `d17.` prefix and pass it to qd-metal's `set_answer_delta`. The scorer already scores one seed's Metal export without ckpt_average (Lappi tools/real_ft_run.py:7426-7434, :7741-7742) but refuses keys other than tower.* and span_head.* (:8006-8007); the release side already defines d17. (crates/qd-export/src/layout.rs:38-44, crates/qd-metal/src/weights.rs:334-360).
- **No merged-only fallback:** folding D17 into the 17 tied rows changes those tokens' input embeddings and full-vocab head rows, so it would score a model that was not trained. The Lappi half goes to Lappi's board (gp-ft-lora-2b-lane).
- Criterion 8 of gp-ft-frozen-params-and-lora (the model-level fold) is owned here.

### Fable ruling (2026-10-17)

Synced from the GitPulse board by the 2026-10-17 brief sync; the board is canonical (Fable's final ruling of 2026-10-17, section 5), this section is the long-form record. Board card ft-58e316a9bae421a3a2d71c26192aaea8, revision 5: status backlog, priority 1, severity medium. Criterion numbers are the board's, counted from 0 in the order of the card's acceptance criteria.

Ruling set: status backlog, priority 1 (unchanged), severity medium (unchanged).

Marks (the text of the existing criterion is otherwise unchanged on the board; nothing deleted, no threshold moved):

- ft-58e316a9bae421a3a2d71c26192aaea8 criterion 6: [DECIDED 2026-10-17: see the next criteria]

Criteria appended by the ruling, verbatim:

- [ ] ft-58e316a9bae421a3a2d71c26192aaea8 criterion 7: (Fable 2026-10-17) The quantised arm's merge target is dequant(Q(W)) + (alpha/r) B A, the base the adapters trained over (the fake-quant file), rounded once to bf16 RNE; merging into the original bf16 W is refused with a typed error naming the base-sha mismatch. The release is a dense bf16 tower scored by qd-export and qd-metal unchanged; serving a quantised base is not in this track (D5's reversal is not triggered). The manifest records base format, group size, mlx version and the base file's sha256.
- [ ] ft-58e316a9bae421a3a2d71c26192aaea8 criterion 8: (Fable 2026-10-17) The checkpoint-to-scorer bridge is Lappi's (ft-2ed2fd55): a converter reads ojas's adapter checkpoint (ft-b5e64557 criterion 13's documented layout) into the lora-merged layout tools/ckpt_average.py already accepts (frozen tower, lora_A/lora_B by module name, lora_r, lora_alpha, d17, span_head; ckpt_average.py:152-160) and closes GAP-LORA-CKPT-LAYOUT-NOT-OJAS-VERIFIED-2026-10-12 with a round-trip test; ojas's part here is a parity test that the merged tower from that route equals ojas's own per-site merge (lora.rs merge) within the stated bf16 bound.

Board criteria and marks already on the card before the ruling (2026-10-17 owner, gap and fold entries, and earlier audit marks), not previously in this brief, verbatim:

- [ ] ft-58e316a9bae421a3a2d71c26192aaea8 criterion 5: (audit-2026-10-16) `fuse_b`/`split_b` return `Result` and refuse wrong arity and every wrong length (short, long, empty) for each `LoraKind` with `OjasError::Shape`; `LoraSite::init` refuses an `out_parts` sum that overflows; `scale_constant` with a shape past the Budget returns `CapacityExceeded` without allocating.
