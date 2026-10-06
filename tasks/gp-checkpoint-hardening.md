---
id: "gp-checkpoint-hardening"
title: "Harden checkpoint save/resume: RNG state layout, free-disk check, streaming device save"
status: backlog
priority: 2
severity: medium
type: feature
owner: "unassigned"
due: "none"
labels:
  - "checkpoint"
  - "robustness"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/checkpoint.rs"
  - "ojas-model/src/checkpoint.rs"
  - "ojas-io/src/"
  - "docs/checkpoint-v1.md"
acceptance_criteria:
  - "The rng_state layout is specified and versioned, and a test proves save, resume, and N more steps match an uninterrupted run bit for bit on CPU"
  - "Save checks free space against the expected size and refuses before writing"
  - "Device-tensor save streams in pieces, or the decision not to is recorded with the reason"
---

# Task brief v1

## Title
Harden checkpoint save/resume: RNG state layout, free-disk check, streaming device save

Task: gp-checkpoint-hardening
Type: feature
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: checkpoint, robustness

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). Three checkpoint gaps:

1. **No exact resume.** `CheckpointV1.rng_state` is an opaque byte string; the doc comment says "the counter-RNG field order is not part of this scaffold" (`ojas-core/src/checkpoint.rs:99-101`). A resumed run cannot be proven to reproduce the same data order and dropout.
2. **No free-space check.** No `statvfs` or free-space call exists in `ojas-io`, `ojas-model` or `ojas-capi` (`docs/adaptive-resources.md:250`). A full disk is found mid-write.
3. **Save headroom.** `docs/typed-storage-plan.md:189` proposes reading device bytes piece by piece during save to cut headroom to one piece. This moves where deferred faults are reported, so it is a separate, deliberate change. Reported in docs, not re-verified.

## Acceptance criteria
- [ ] The rng_state layout is specified and versioned, and a test proves save, resume, and N more steps match an uninterrupted run bit for bit on CPU
- [ ] Save checks free space against the expected size and refuses before writing
- [ ] Device-tensor save streams in pieces, or the decision not to is recorded with the reason

## Planned files
- ojas-core/src/checkpoint.rs
- ojas-model/src/checkpoint.rs
- ojas-io/src/
- docs/checkpoint-v1.md
