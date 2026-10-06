---
id: "gp-pin-sibling-deps"
title: "Pin the tessl and gusset revisions ojas builds against"
status: backlog
priority: 2
severity: medium
type: chore
owner: "unassigned"
due: "none"
labels:
  - "build"
  - "reproducibility"
repositories:
  - "ojas"
planned_files:
  - "ojas-metal/Cargo.toml"
  - "ojas-qwen35/Cargo.toml"
  - "ojas-capi/Cargo.toml"
  - "ojas-gusset-engine/Cargo.toml"
  - ".github/workflows/test.yml"
acceptance_criteria:
  - "The tessl and gusset revisions ojas is known to work with are recorded in the repo (a pinned git dependency, or a checked-in ref file that CI reads)"
  - "CI defaults TESSL_REF and GUSSET_REF to the pinned revisions instead of main"
  - "A short doc note says how to bump the pins"
---

# Task brief v1

## Title
Pin the tessl and gusset revisions ojas builds against

Task: gp-pin-sibling-deps
Type: chore
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: build, reproducibility

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). ojas builds against whatever tessl and gusset checkouts happen to be beside it. `ojas-metal/Cargo.toml:22` and `ojas-qwen35/Cargo.toml:21` use `tessl = { path = "../../tessl" }`; `ojas-capi/Cargo.toml:11` and `ojas-gusset-engine/Cargo.toml:17` use `path = "../../../devtools/gusset/crates/gusset"`. CI checks both out at `main` by default (`.github/workflows/test.yml:39-40`, `TESSL_REF`/`GUSSET_REF`). Any ojas commit can therefore build differently tomorrow, and a breaking change on tessl main breaks ojas CI with no ojas change. (`docs/audit-resources.md:60`, S8, records the gusset half.)

At audit time, local tessl is 52c091c (clean) and gusset is 7ea6b35.

## Acceptance criteria
- [ ] The tessl and gusset revisions ojas is known to work with are recorded in the repo (a pinned git dependency, or a checked-in ref file that CI reads)
- [ ] CI defaults TESSL_REF and GUSSET_REF to the pinned revisions instead of main
- [ ] A short doc note says how to bump the pins

## Planned files
- ojas-metal/Cargo.toml
- ojas-qwen35/Cargo.toml
- ojas-capi/Cargo.toml
- ojas-gusset-engine/Cargo.toml
- .github/workflows/test.yml
