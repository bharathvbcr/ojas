---
id: "gp-hip-backend-decision"
title: "Decide whether ojas-hip becomes a backend or is documented as a probe"
status: backlog
priority: 3
severity: low
type: spike
owner: "unassigned"
due: "none"
labels:
  - "hip"
  - "backend"
  - "needs-your-words"
repositories:
  - "ojas"
planned_files:
  - "ojas-hip/"
  - "ojas-device/src/lib.rs"
  - "docs/backends.md"
acceptance_criteria:
  - "A decision is recorded: implement HIP (file the feature task), or document ojas-hip as a memory probe and make BackendId::Hip refuse clearly at selection time"
---

# Task brief v1

## Title
Decide whether ojas-hip becomes a backend or is documented as a probe

Task: gp-hip-backend-decision
Type: spike
Status: backlog
Priority: 3 (Low)
Severity: low
Owner: unassigned
Due: none
Labels: hip, backend, needs-your-words

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). `ojas-hip/src/lib.rs` (468 lines) has `open`, `device_count` and `copy_roundtrip` and no kernels. `BackendId::Hip` exists (`ojas-device/src/lib.rs:103`), but nothing implements `Backend` for it. CI never compiles the `hip` feature, because `hip-runtime-sys` 0.1.2 panics at build time without the ROCm headers (comment in the gpu-compile job). Selecting HIP in ojas-device therefore leads nowhere.

## Acceptance criteria
- [ ] A decision is recorded: implement HIP (file the feature task), or document ojas-hip as a memory probe and make BackendId::Hip refuse clearly at selection time

## Planned files
- ojas-hip/
- ojas-device/src/lib.rs
- docs/backends.md
