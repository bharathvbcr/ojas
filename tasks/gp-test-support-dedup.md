---
id: "gp-test-support-dedup"
title: "Shared test support: one counting allocator, one set of backward references, one per-backend skip and device helper instead of copies in 30+ test files"
status: backlog
priority: 2
severity: low
type: refactor
owner: "unassigned"
due: "none"
labels:
  - "tests"
  - "refactor"
  - "dedup"
repositories:
  - "ojas"
planned_files:
  - "ojas-cpu/tests/"
  - "ojas-metal/tests/"
  - "ojas-wgpu/tests/"
  - "ojas-model/tests/common/mod.rs"
  - "ojas-cuda/tests/"
acceptance_criteria:
  - "One counting allocator in a dev-only test-support crate (or module per crate) replaces the 9 copies; each heap test still pins the same numbers"
  - "ojas-model tests use ojas-autograd's backward references instead of their own; ref_forward/ref_backward/assert_bits are defined once"
  - "The small helpers (vals, metal(), skip_or_fail, snapshot_dir, next_u64, pending, assert_all_pass, assert_not_wired) each have one definition; DevMap clone groups for them are empty"
  - "No test is deleted or weakened; the before and after test counts per crate are recorded with lines removed"
  - "One skip protocol: 4 skip_or_fail definitions (not 3), no_adapter and two metal() helpers implement two semantics (metal_hybrid_tape.rs:114 and qwen35_metal_tape.rs:30 skip on var_os().is_some(), so OJAS_ALLOW_NO_GPU=0 skips there but fails elsewhere); one helper in test support, with the narrowed skip rule from gp-test-suite-integrity"
---

# Task brief v1

## Title
Shared test support: one counting allocator, one set of backward references, one per-backend skip and device helper instead of copies in 30+ test files

Task: gp-test-support-dedup
Type: refactor
Status: backlog
Priority: 2 (Normal)
Severity: low
Owner: unassigned
Due: none
Labels: tests, refactor, dedup

## Repositories
- ojas

## Description
Filed by the 2026-10-09 task audit (at d431949), moved out of gp-structural-dedup (old criterion 13) and widened. Test helpers drift; every copy is a place a fix has to be made twice. Evidence is DevMap clone groups plus rg [A]:
- **Counting GlobalAlloc, 9 copies:** ojas-cpu/tests/{attention_fast,framework_ce_heap,redteam_linear_heap,redteam_ops_heap}.rs, ojas-wgpu/tests/upload_heap.rs, ojas-metal/tests/upload_heap.rs, ojas-io/tests/stream_heap.rs, ojas-model/tests/resume_memory.rs and ojas-device/src/bandwidth.rs (in its test module).
- **Backward references:** ojas-model/tests/common/mod.rs repeats ojas-autograd's linear/mul/residual/CE backward references. ref_forward, ref_backward and assert_bits are exact copies across kernel_harden.rs, linear_kernel.rs and ops.rs.
- **Small exact clones:**
  - `vals` x3 (checkpoint.rs, hybrid_tape.rs, metal_hybrid_tape.rs).
  - `metal()` x2 (ojas-model/tests/{metal_hybrid_tape,qwen35_metal_tape}.rs).
  - `skip_or_fail` x3.
  - `snapshot_dir` in gpu_parity.rs and gpu_tape_2b.rs.
  - `next_u64` x5.
  - ojas-metal/src/backend.rs::pending, copied into 3 Metal test files.
- **CUDA:** assert_all_pass is copied in four device tests, and assert_not_wired appears in src/step.rs and tests/step_refuses.rs.
- **Scope boundary:** ojas-autograd's own helpers stay with gp-ci-and-repo-hygiene.

Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- Skip-helper count corrected to 7 implementations with two semantics [A].

## Acceptance criteria
- [ ] One counting allocator in a dev-only test-support crate (or module per crate) replaces the 9 copies; each heap test still pins the same numbers
- [ ] ojas-model tests use ojas-autograd's backward references instead of their own; ref_forward/ref_backward/assert_bits are defined once
- [ ] The small helpers (vals, metal(), skip_or_fail, snapshot_dir, next_u64, pending, assert_all_pass, assert_not_wired) each have one definition; DevMap clone groups for them are empty
- [ ] No test is deleted or weakened; the before and after test counts per crate are recorded with lines removed
- [ ] One skip protocol: 4 skip_or_fail definitions (not 3), no_adapter and two metal() helpers implement two semantics (metal_hybrid_tape.rs:114 and qwen35_metal_tape.rs:30 skip on var_os().is_some(), so OJAS_ALLOW_NO_GPU=0 skips there but fails elsewhere); one helper in test support, with the narrowed skip rule from gp-test-suite-integrity

## Planned files
- ojas-cpu/tests/
- ojas-metal/tests/
- ojas-wgpu/tests/
- ojas-model/tests/common/mod.rs
- ojas-cuda/tests/
