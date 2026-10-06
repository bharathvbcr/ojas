---
id: "gp-gusset-alloc-abort"
title: "Stop a failed Rust allocation from aborting the Go host process"
status: backlog
priority: 2
severity: medium
type: bug
owner: "unassigned"
due: "none"
labels:
  - "gusset"
  - "capi"
  - "robustness"
repositories:
  - "ojas"
planned_files:
  - "ojas-gusset-engine/src/lib.rs"
  - "ojas-capi/src/"
  - "go/"
acceptance_criteria:
  - "Large allocations on the engine path use fallible allocation (try_reserve) and surface OjasError::OutOfMemory-style errors to Go"
  - "Whatever abort risk remains is documented with its bound"
  - "A Go test drives an allocation failure under a low memory ceiling and gets an error, not a process abort"
---

# Task brief v1

## Title
Stop a failed Rust allocation from aborting the Go host process

Task: gp-gusset-alloc-abort
Type: bug
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: gusset, capi, robustness

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). The Go-hosted static library aborts the whole Go process on any failed infallible Rust allocation. The counting allocator returns null on failure, but `vec!`, `format!`, channels and the like still abort. No allocation-error hook is installed (`ojas-gusset-engine/src/lib.rs:6-10` says so). `docs/audit-resources.md:31` records it as C1. A host that runs ojas beside other work loses everything when one large tensor allocation fails, instead of getting an error.

## Acceptance criteria
- [ ] Large allocations on the engine path use fallible allocation (try_reserve) and surface OjasError::OutOfMemory-style errors to Go
- [ ] Whatever abort risk remains is documented with its bound
- [ ] A Go test drives an allocation failure under a low memory ceiling and gets an error, not a process abort

## Planned files
- ojas-gusset-engine/src/lib.rs
- ojas-capi/src/
- go/
