---
id: "gp-wgpu-folded-grid-tails"
title: "wgpu norm_partial and sum_partial have no tail guard on a folded grid: spare workgroups read the chunk table out of range and race on the last partial, so the global grad norm drops a chunk at Qwen3.5-2B embedding size"
status: ready
priority: 1
severity: high
type: bug
owner: "unassigned"
due: "none"
labels:
  - "wgpu"
  - "kernel"
  - "bug"
  - "clip"
  - "muon"
  - "determinism"
repositories:
  - "ojas"
planned_files:
  - "ojas-kernels/src/wgsl/clip.wgsl"
  - "ojas-kernels/src/wgsl/reduce.wgsl"
  - "ojas-kernels/src/geometry.rs"
  - "ojas-wgpu/src/backend.rs"
  - "ojas-wgpu/tests/"
acceptance_criteria:
  - "A failing test first, on the capped-limit repro: with max_compute_workgroups_per_dimension capped at 2, clip_grad_norm over a 3-chunk tensor and the Muon global norm match the CPU exactly and repeat bit-for-bit across 20 runs; sum (reduce.wgsl) is covered the same way"
  - "norm_partial and sum_partial return before any table read or write when the flat id is past the real chunk count"
  - "Every WGSL kernel dispatched through fold_grid / groups() is listed with its tail guard (or why it needs none), and a test helper runs each folded-grid kernel under the capped limit, so a new kernel without a guard fails"
  - "The Metal and CUDA equivalents of folded or rounded-up grids are checked for the same class and the result recorded"
---

# Task brief v1

## Title
wgpu norm_partial and sum_partial have no tail guard on a folded grid: spare workgroups read the chunk table out of range and race on the last partial, so the global grad norm drops a chunk at Qwen3.5-2B embedding size

Task: gp-wgpu-folded-grid-tails
Type: bug
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: wgpu, kernel, bug, clip, muon, determinism

## Repositories
- ojas

## Description
Filed by the second audit (2026-10-09). The fix is for the bug class, not one kernel.

- **Contract [V]:** `fold_grid` (ojas-kernels/src/geometry.rs:103-124) rounds a count over max_per_dim up to max_per_dim x rows. Its doc says 'the kernel recovers the flat id ... and guards the tail'.
- **Two kernels break it [V]:**
  - `norm_partial` (clip.wgsl:52-89) computes `c = pw(2u) + flat_group(wg, nwg)`. It then reads `tbl[2u * slot_of(c)]` and writes `part[c]` with no `c < chunk_count` check.
  - `sum_partial` (reduce.wgsl:22-39) writes `y0[group]` with no `group < ceil(n / CHUNK)` check.
- **Host [A]:** clip dispatches `self.groups(end - first)` (ojas-wgpu/src/backend.rs:1003-1008), and sum dispatches `self.groups(groups)` (:837-845). Scratch is sized exactly (context.rs:1800-1803).
- **Effect [A + I]:**
  - wgpu 30.0.1 on Metal (and Vulkan without robustBufferAccess2) compiles with the Restrict bounds policy. Spare workgroups' writes are clamped onto the last real entry and race with its owner.
  - The last 4,096-value chunk can drop out of the norm at random: an under-reported norm, and results that stop repeating run to run.
- **Failing shape:** a single clip dispatch group binding more than 65,535 x 4,096 = 268,431,360 elements. The Qwen3.5-2B embedding or lm_head gradient (248,320 x 2,048 = 508,559,360) gives 124,160 chunks, a (65,535, 2) grid and 6,910 spare workgroups.
- **Cheap repro:** cap max_compute_workgroups_per_dimension at 2 through the context's limit cap (context.rs:693-695) and clip a 12,288-element tensor (3 chunks, grid (2, 2), one spare).

Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] A failing test first, on the capped-limit repro: with max_compute_workgroups_per_dimension capped at 2, clip_grad_norm over a 3-chunk tensor and the Muon global norm match the CPU exactly and repeat bit-for-bit across 20 runs; sum (reduce.wgsl) is covered the same way
- [ ] norm_partial and sum_partial return before any table read or write when the flat id is past the real chunk count
- [ ] Every WGSL kernel dispatched through fold_grid / groups() is listed with its tail guard (or why it needs none), and a test helper runs each folded-grid kernel under the capped limit, so a new kernel without a guard fails
- [ ] The Metal and CUDA equivalents of folded or rounded-up grids are checked for the same class and the result recorded

## Planned files
- ojas-kernels/src/wgsl/clip.wgsl
- ojas-kernels/src/wgsl/reduce.wgsl
- ojas-kernels/src/geometry.rs
- ojas-wgpu/src/backend.rs
- ojas-wgpu/tests/
