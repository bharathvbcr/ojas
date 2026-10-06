---
id: "gp-gpu-test-budget-readbacks"
title: "Eliminate Process-Wide Readback Race by Migrating GPU Tests to Budget::device_readbacks"
status: ready
priority: 2
severity: medium
type: bug
owner: "unassigned"
due: "none"
labels:
  - "testing"
  - "concurrency"
  - "gpu"
  - "budget"
repositories:
  - "ojas"
planned_files:
  - "ojas-wgpu/tests/muon.rs"
  - "ojas-wgpu/tests/permute.rs"
  - "ojas-wgpu/tests/residency.rs"
  - "ojas-wgpu/tests/parity.rs"
  - "ojas-autograd/tests/device_tape.rs"
  - "ojas-autograd/tests/wgpu_tape.rs"
  - "ojas-capi/src/tests.rs"
  - "ojas-metal/tests/backend_contract.rs"
  - "README.md"
acceptance_criteria:
  - "Migrate GPU test readback assertions from process-wide device_readbacks() to Budget::device_readbacks() scoped per budget tree"
  - "Verify all GPU tests (wgpu, autograd, capi, metal) pass cleanly under default multi-threaded test execution (cargo test without --test-threads=1)"
  - "Zero regressions in budget accounting and readback counting invariants"
---

# Task brief v1

## Title
Eliminate Process-Wide Readback Race by Migrating GPU Tests to Budget::device_readbacks

Task: gp-gpu-test-budget-readbacks
Type: bug
Status: ready
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: testing, concurrency, gpu, budget

## Repositories
- ojas

## Description
Currently, several GPU test suites (`ojas-wgpu/tests/muon.rs`, `permute.rs`, `residency.rs`, `ojas-autograd`, `ojas-capi`, and `ojas-metal`) assert zero unexpected device readbacks using the global `ojas_core::device_readbacks()` counter. Because this counter is process-wide (`AtomicU64`), concurrently running tests downloading tensors corrupt each other's readback counts, requiring tests to be run with `--test-threads=1`.

The class fix has already landed in `ojas-core`: `Budget::device_readbacks()` counts per budget tree from any thread (`budget_readbacks_count_only_their_own_tree_across_threads`).

This task migrates all GPU test assertions to the tree-scoped budget reader and verifies multi-threaded execution.

### Progress (audit of 9668bfa, 2026-10-05, code inspection only; tests not executed)
The migration is almost finished:
- wgpu tests (muon, permute, residency, parity) count through `readbacks(&g)`, which calls `g.budget().device_readbacks()` (`ojas-wgpu/tests/common/mod.rs:36`). Metal (`ojas-metal/tests/backend_contract.rs:569`) and autograd (`ojas-autograd/tests/device_tape.rs:469`) are per-budget too.
- **One site remains:** `readbacks_during` in `ojas-capi/src/tests.rs:1796-1800` still reads the process-wide `ojas_core::device_readbacks()`. An `rg` over the workspace finds no other caller.
- Criterion 2 is untested. `README.md:330,364,367` still tells people to run with `--test-threads=1`; drop that once a multi-threaded run passes.
- The planned files `ojas-capi/tests/readbacks.rs` and `ojas-metal/tests/trainer.rs` do not exist and have been replaced with the real paths.

## Acceptance criteria
- [ ] Migrate GPU test readback assertions from process-wide device_readbacks() to Budget::device_readbacks() scoped per budget tree
- [ ] Verify all GPU tests (wgpu, autograd, capi, metal) pass cleanly under default multi-threaded test execution (cargo test without --test-threads=1)
- [ ] Zero regressions in budget accounting and readback counting invariants

## Planned files
- ojas-wgpu/tests/muon.rs
- ojas-wgpu/tests/permute.rs
- ojas-wgpu/tests/residency.rs
- ojas-wgpu/tests/parity.rs
- ojas-autograd/tests/device_tape.rs
- ojas-autograd/tests/wgpu_tape.rs
- ojas-capi/src/tests.rs
- ojas-metal/tests/backend_contract.rs
- README.md
