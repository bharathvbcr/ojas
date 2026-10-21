---
id: "gp-verify-unverified-commits"
title: "Compile and run the commits that landed unverified (AdamW 100-step oracle, LoRA memory preflight calibration)"
status: ready
priority: 1
severity: high
type: bug
owner: "unassigned"
due: "none"
labels:
  - "verification"
  - "oracle"
  - "lora"
  - "research-2026-10-20"
repositories:
  - "ojas"
planned_files: []
acceptance_criteria:
  - "525bab2 (AdamW vs torch.optim.AdamW 100-step oracle) compiles and runs; results recorded"
  - "9ffc289 (LoRA memory preflight calibration) compiles and runs; results recorded"
  - "Any failure is fixed or the commit reverted"
  - "main has a written rule that a commit marked UNVERIFIED must reference a tracking task"
---

# Task brief v1

## Title
Compile and run the commits that landed unverified (AdamW 100-step oracle, LoRA memory preflight calibration)

Task: gp-verify-unverified-commits
Type: bug
Status: ready
Priority: 1 (High)
Severity: high
Source: research session 2026-10-20 (article audit gaps)

## Description
These two commits on main say in their own messages that they were never compiled or run:
- `525bab2`: the AdamW vs `torch.optim.AdamW` 100-step oracle
- `9ffc289`: the LoRA memory preflight calibration

An oracle that has never run doesn't show that anything matches, and an uncalibrated preflight may refuse runs that would fit, or allow runs that won't.

## Acceptance criteria
- [ ] 525bab2 (AdamW vs torch.optim.AdamW 100-step oracle) compiles and runs; results recorded
- [ ] 9ffc289 (LoRA memory preflight calibration) compiles and runs; results recorded
- [ ] Any failure is fixed or the commit reverted
- [ ] main has a written rule that a commit marked UNVERIFIED must reference a tracking task
