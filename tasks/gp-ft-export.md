---
id: "gp-ft-export"
title: "LoRA-2B release export: merged dense bf16 tower plus D17 and span head through Lappi's ckpt_average and qd-export; seed averaging on merged weights only; PEFT-format adapters for the reference arm"
status: backlog
priority: 2
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
Priority: 2 (Normal)
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
