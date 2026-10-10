---
id: "gp-prepaid-session-mode"
title: "Opt-In Prepaid Session Mode: Reserve the Session Cap From the Process Ceiling at LOAD"
status: backlog
priority: 3
severity: medium
type: feature
owner: "unassigned"
due: "none"
labels:
  - "capi"
  - "budget"
  - "adaptive-resources"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/budget.rs"
  - "ojas-capi/src/session.rs"
  - "ojas-capi/src/load.rs"
  - "go/api.go"
  - "go/ffi.go"
  - "docs/adaptive-resources.md"
acceptance_criteria:
  - "A caller can request prepaid mode at LOAD (C API flag and Go option); the default is unchanged, and every existing capi and Go test passes without modification"
  - "In prepaid mode, LOAD charges the session's whole budget to the process ceiling before the device opens, and refuses with E_CAPACITY when the ceiling has less room; the reservation is released on FREE and on every load error path"
  - "A test with two sessions sharing the ceiling shows the race closed: a prepaid session's step cannot be refused inside apply because of the other session's allocation, while the same schedule without prepaid mode still reaches the documented race"
  - "Allocations inside a prepaid session draw from its reservation and never charge the shared root twice (suggested shape: the Lease holds one root Reservation of the full cap and the session gets a detached Budget::new(cap); Budget::child today charges the parent on every reservation, ojas-core/src/budget.rs:73-84); Budget::peak_bytes and check_room are stated to report session-local numbers"
  - "SetMemoryCeiling still refuses while any model, prepaid or not, is open (holds by construction today, ojas-capi/src/session.rs:136-142; a test pins it for a prepaid session)"
  - "docs/adaptive-resources.md records the mode, its trade (refused loads for an unpoisonable session) and the tests"
---

# Task brief v1

## Title
Opt-In Prepaid Session Mode: Reserve the Session Cap From the Process Ceiling at LOAD

Task: gp-prepaid-session-mode
Type: feature
Status: backlog
Priority: 3 (Low)
Severity: medium
Owner: unassigned
Due: none
Labels: capi, budget, adaptive-resources

## Repositories
- ojas

## Description
Filed by `gp-backend-architecture-decisions`: the user adopted prepaid session mode as an opt-in (2026-10-08). Today every session budget is a child of one root `Budget` (`ojas-capi/src/session.rs` `session_budget`), and trainer preflight is a point-in-time `check_room` that reserves nothing. Two sessions sharing the ceiling can therefore race: the other session takes room between this session's last `check_room` after clip and `apply`, and a refusal inside `apply` poisons that trainer. Prepaid mode closes the window by reserving the session's cap at LOAD, at the cost of refusing loads that would fit under the point-in-time check. It is opt-in so no load that succeeds today is refused by default.

### Re-audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- All criteria open; no prepaid code exists (rg -uu) [A]. The description is accurate. An implementation note has been added to criterion 4.

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Lowered to P3: off the Mac (Metal) training path. Raise it again when CUDA, HIP or wgpu training becomes a goal.

## Acceptance criteria
- [ ] A caller can request prepaid mode at LOAD (C API flag and Go option); the default is unchanged, and every existing capi and Go test passes without modification
- [ ] In prepaid mode, LOAD charges the session's whole budget to the process ceiling before the device opens, and refuses with E_CAPACITY when the ceiling has less room; the reservation is released on FREE and on every load error path
- [ ] A test with two sessions sharing the ceiling shows the race closed: a prepaid session's step cannot be refused inside apply because of the other session's allocation, while the same schedule without prepaid mode still reaches the documented race
- [ ] Allocations inside a prepaid session draw from its reservation and never charge the shared root twice (suggested shape: the Lease holds one root Reservation of the full cap and the session gets a detached Budget::new(cap); Budget::child today charges the parent on every reservation, ojas-core/src/budget.rs:73-84); Budget::peak_bytes and check_room are stated to report session-local numbers
- [ ] SetMemoryCeiling still refuses while any model, prepaid or not, is open (holds by construction today, ojas-capi/src/session.rs:136-142; a test pins it for a prepaid session)
- [ ] docs/adaptive-resources.md records the mode, its trade (refused loads for an unpoisonable session) and the tests

## Planned files
- ojas-core/src/budget.rs
- ojas-capi/src/session.rs
- ojas-capi/src/load.rs
- go/api.go
- go/ffi.go
- docs/adaptive-resources.md
