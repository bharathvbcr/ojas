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
  - "ojas-capi/tests/readbacks.rs"
  - "ojas-metal/tests/trainer.rs"
acceptance_criteria:
  - "Migrate GPU test readback assertions from process-wide device_readbacks() to Budget::device_readbacks() scoped per budget tree"
  - "Verify all GPU tests (wgpu, autograd, capi, metal) pass cleanly under default multi-threaded test execution (cargo test without --test-threads=1)"
  - "Zero regressions in budget accounting and readback counting invariants"
---

# Task Brief: Eliminate Process-Wide Readback Race by Migrating GPU Tests to Budget::device_readbacks

## Context & Problem Statement
Currently, several GPU test suites (`ojas-wgpu/tests/muon.rs`, `permute.rs`, `residency.rs`, `ojas-autograd`, `ojas-capi`, and `ojas-metal`) assert zero unexpected device readbacks using the global `ojas_core::device_readbacks()` counter. Because this counter is process-wide (`AtomicU64`), concurrently running tests downloading tensors corrupt each other's readback counts, requiring tests to be run with `--test-threads=1`.

The class fix has already landed in `ojas-core`: `Budget::device_readbacks()` counts per budget tree from any thread (`budget_readbacks_count_only_their_own_tree_across_threads`).

## Scope & Implementation Details
1. **Migrate Test Assertions**:
   - Replace calls to `device_readbacks()` in `ojas-wgpu`, `ojas-autograd`, `ojas-capi`, and `ojas-metal` tests with `backend.budget().device_readbacks()` or the scoped test budget.
2. **Multi-Threaded Verification**:
   - Run `cargo test -p ojas-wgpu`, `cargo test -p ojas-autograd`, `cargo test -p ojas-capi`, and `cargo test -p ojas-metal` with default parallel thread execution to ensure complete isolation.
3. **Parity & Audit**:
   - Verify that test assertions still detect real unexpected device download traffic.
