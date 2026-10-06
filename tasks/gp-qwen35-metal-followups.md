---
id: "gp-qwen35-metal-followups"
title: "Qwen3.5 on Metal: per-parameter learning rates (blocked on tessl) and 2B headroom measurements"
status: backlog
priority: 3
severity: low
type: feature
owner: "unassigned"
due: "none"
labels:
  - "qwen35"
  - "metal"
  - "lappi"
  - "blocked"
repositories:
  - "ojas"
planned_files:
  - "ojas-qwen35/src/groups.rs"
  - "ojas-qwen35/src/step.rs"
  - "ojas-qwen35/README.md"
acceptance_criteria:
  - "Once tessl ships per-parameter LR, check_tessl_lr is removed and a test proves per-group LR scales the update"
  - "2B save/load and a longer sequence length are run on Metal, and the measured peak and headroom are recorded in the README"
---

# Task brief v1

## Title
Qwen3.5 on Metal: per-parameter learning rates (blocked on tessl) and 2B headroom measurements

Task: gp-qwen35-metal-followups
Type: feature
Status: backlog
Priority: 3 (Low)
Severity: low
Owner: unassigned
Due: none
Labels: qwen35, metal, lappi, blocked

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). 1. **Per-parameter learning rates are refused.** `check_tessl_lr` (`ojas-qwen35/src/groups.rs:278`, called at `step.rs:595`) refuses any `lr_scale != 1.0`, because tessl has no per-parameter LR yet. This waits on tessl branch `lappi-train-lrscale-mrope` (`ojas-qwen35/README.md:123-128`).
2. **Unmeasured at 2B.** `ojas-qwen35/README.md:222-232` lists 2B save/load, bf16 operands and longer sequences as never measured. Headroom is 1.34 GB of the Metal working set, so these may hit the memory refusal. Sharded snapshots must also keep every tower tensor in one file (`names.rs:210-252`). This needs a GPU window under the Mac resource lock.

## Acceptance criteria
- [ ] Once tessl ships per-parameter LR, check_tessl_lr is removed and a test proves per-group LR scales the update
- [ ] 2B save/load and a longer sequence length are run on Metal, and the measured peak and headroom are recorded in the README

## Planned files
- ojas-qwen35/src/groups.rs
- ojas-qwen35/src/step.rs
- ojas-qwen35/README.md
