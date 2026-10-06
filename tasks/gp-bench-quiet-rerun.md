---
id: "gp-bench-quiet-rerun"
title: "Re-run benchmarks on a quiet machine: torch parity spread rule and the typed-storage success check"
status: backlog
priority: 2
severity: medium
type: chore
owner: "unassigned"
due: "none"
labels:
  - "bench"
  - "evidence"
repositories:
  - "ojas"
planned_files:
  - "bench/"
  - "bench/results/"
  - "docs/bench-gpu-vs-torch.md"
  - "docs/pytorch-parity-plan.md"
  - "docs/typed-storage-plan.md"
acceptance_criteria:
  - "A full quiet GPU-vs-torch re-run is committed under bench/results/ with spread per row; rows still over 10% are marked as such in the docs"
  - "parity.sh, block_ab.sh and the sample profile are run for typed storage, and the result is recorded in typed-storage-plan.md"
  - "The ~9% embedding-forward regression is explained or fixed"
---

# Task brief v1

## Title
Re-run benchmarks on a quiet machine: torch parity spread rule and the typed-storage success check

Task: gp-bench-quiet-rerun
Type: chore
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: bench, evidence

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). Two benchmark claims are still pending runs:

1. **torch-parity ratios never passed the 10% spread rule.** `docs/pytorch-parity-plan.md:275-278` says completion criterion 4 is unmet, and `docs/bench-gpu-vs-torch.md:518` says all 94 rows fail it; only a few single-op runs were quiet (line 438). Until a quiet re-run exists in `bench/results/`, no ratio should be quoted as a number.
2. **The typed host storage plan was never checked for success.** `docs/typed-storage-plan.md:8,258` say the plan only counts as a success after `parity.sh`, `block_ab.sh` and a new `sample` profile show the storage cost shrinking. Line 210 records embedding forward ~9% slower after step 1, "not yet explained". No result exists in `bench/results/`.

These need a quiet machine and the Mac resource lock (ask first, -j 2).

## Acceptance criteria
- [ ] A full quiet GPU-vs-torch re-run is committed under bench/results/ with spread per row; rows still over 10% are marked as such in the docs
- [ ] parity.sh, block_ab.sh and the sample profile are run for typed storage, and the result is recorded in typed-storage-plan.md
- [ ] The ~9% embedding-forward regression is explained or fixed

## Planned files
- bench/
- bench/results/
- docs/bench-gpu-vs-torch.md
- docs/pytorch-parity-plan.md
- docs/typed-storage-plan.md
