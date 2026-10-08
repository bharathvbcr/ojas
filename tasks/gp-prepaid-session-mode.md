---
id: "gp-prepaid-session-mode"
title: "Opt-In Prepaid Session Mode: Reserve the Session Cap From the Process Ceiling at LOAD"
status: backlog
priority: 2
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
  - "Allocations inside a prepaid session draw from its reservation and never charge the shared root twice; Budget::peak_bytes and check_room stay correct for both modes"
  - "SetMemoryCeiling still refuses while any model, prepaid or not, is open"
  - "docs/adaptive-resources.md records the mode, its trade (refused loads for an unpoisonable session) and the tests"
---

# Task brief v1

## Title
Opt-In Prepaid Session Mode: Reserve the Session Cap From the Process Ceiling at LOAD

Task: gp-prepaid-session-mode
Type: feature
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: capi, budget, adaptive-resources

## Repositories
- ojas

## Description
Filed by `gp-backend-architecture-decisions`: the user adopted prepaid session mode as an opt-in (2026-10-08). Today every session budget is a child of one root `Budget` (`ojas-capi/src/session.rs` `session_budget`), and trainer preflight is a point-in-time `check_room` that reserves nothing. Two sessions sharing the ceiling can therefore race: the other session takes room between this session's last `check_room` after clip and `apply`, and a refusal inside `apply` poisons that trainer. Prepaid mode closes the window by reserving the session's cap at LOAD, at the cost of refusing loads that would fit under the point-in-time check. It is opt-in so no load that succeeds today is refused by default.

## Acceptance criteria
- [ ] A caller can request prepaid mode at LOAD (C API flag and Go option); the default is unchanged, and every existing capi and Go test passes without modification
- [ ] In prepaid mode, LOAD charges the session's whole budget to the process ceiling before the device opens, and refuses with E_CAPACITY when the ceiling has less room; the reservation is released on FREE and on every load error path
- [ ] A test with two sessions sharing the ceiling shows the race closed: a prepaid session's step cannot be refused inside apply because of the other session's allocation, while the same schedule without prepaid mode still reaches the documented race
- [ ] Allocations inside a prepaid session draw from its reservation and never charge the shared root twice; Budget::peak_bytes and check_room stay correct for both modes
- [ ] SetMemoryCeiling still refuses while any model, prepaid or not, is open
- [ ] docs/adaptive-resources.md records the mode, its trade (refused loads for an unpoisonable session) and the tests

## Planned files
- ojas-core/src/budget.rs
- ojas-capi/src/session.rs
- ojas-capi/src/load.rs
- go/api.go
- go/ffi.go
- docs/adaptive-resources.md
