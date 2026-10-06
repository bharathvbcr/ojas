---
id: "gp-docs-sync-2026-10-05"
title: "Sync docs that contradict the code after 1f26a4b (CUDA status, test counts, stale residue)"
status: ready
priority: 1
severity: medium
type: chore
owner: "unassigned"
due: "none"
labels:
  - "docs"
repositories:
  - "ojas"
planned_files:
  - "ojas-cuda/README.md"
  - "ojas-hip/README.md"
  - "docs/backends.md"
  - "docs/status.md"
  - "docs/pytorch-parity-plan.md"
  - "docs/metal-deferred-faults.md"
  - "docs/typed-storage-plan.md"
  - "docs/adaptive-resources.md"
  - "docs/framework-design.md"
  - "ojas-simd/src/arch.rs"
  - "docs/reference/"
  - "site/"
acceptance_criteria:
  - "Every checklist item is fixed, or annotated with why it stays"
  - "docs/status.md test totals are recounted from an actual run (with the command and date), not copied"
  - "docs/reference and site are regenerated from the corrected markdown"
---

# Task brief v1

## Title
Sync docs that contradict the code after 1f26a4b (CUDA status, test counts, stale residue)

Task: gp-docs-sync-2026-10-05
Type: chore
Status: ready
Priority: 1 (High)
Severity: medium
Owner: unassigned
Due: none
Labels: docs

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). Several docs now say things the code contradicts. The most misleading is the CUDA status, but test counts and closed residue items are also stale. The audit checked each item below against the code, but only some of the line numbers were re-checked by hand. Re-verify each citation before editing. Where the doc and code disagree, the code wins.

Checklist:
- `ojas-cuda/README.md`, `docs/backends.md:32,74`, `docs/status.md:250,289-292` say ojas-cuda is a probe that does not implement `Backend` and has "8 passed". In fact `impl Backend for CudaBackend` exists (`ojas-cuda/src/backend.rs:73`), but its compute ops all refuse (see gp-cuda-backend-provider). Say exactly that.
- Commit 1f26a4b's wording ("BF16 autocast tier", "CUDA backend") overstates both; `docs/status.md` should say bf16 is emulated in f32 storage and the CUDA step is a stub.
- `ojas-hip/README.md` names `HipDevice::affine_f32`, which does not exist; the function is `copy_roundtrip`.
- `docs/status.md`: line 60 says 9 tasks (tasks/ now holds more); line 62 is a "Previous run 2026-10-04" heading with no body; line 310 says Metal refuses head dim above 64 (it is 256); line 121 says `gpu_real_2b` faults on its gradient read, but `ojas-qwen35/README.md:197-220` says a recycle step fixed it; line 284 labels ojas-infer CPU-only, but its README says the device decoder runs on wgpu and Metal.
- Test counts disagree: `docs/backends.md:71-72` says Metal 136 and wgpu 208/3 ignored; `docs/status.md:19-20` says 175 and 131. The status.md table's wgpu column drops from 208 to 131 with no explanation. Recount and explain.
- `docs/pytorch-parity-plan.md` lists residue that is already fixed: F10 Go error kind (line 169; `go/ffi.go:657-667` now matches by prefix), the CUDA pin (line 185; it is cuda-12080), "no binary has run on Linux" (lines 165, 184; the CI linux job runs tests), the capi demo double-charge (197-199), the ojas-io README names (228), and probably the wgpu cached-attention size limit (227; unconfirmed).
- `docs/metal-deferred-faults.md` §9.4 "stale text" items are all fixed.
- `docs/typed-storage-plan.md:6` says "Steps 2-4 are not started"; its own body (lines 211-223) says steps 2 and 3 are done, and step 4's cleanup is done too.
- `docs/adaptive-resources.md:256` says ojas depends on uncommitted tessl functions; `is_poisoned`, `current_allocated_bytes` and `has_unified_memory` are committed in tessl (`tessl/src/runtime.rs:87,813,822`). The remaining issue (no pinned revision) is gp-pin-sibling-deps.
- `ojas-simd/src/arch.rs:9` says MSRV 1.82; the workspace sets 1.97.
- `docs/framework-design.md:218` says Metal is "synchronous per op today"; it has deferred faults now.
- Re-render `docs/reference/` and `site/` after the markdown changes.

## Acceptance criteria
- [ ] Every checklist item is fixed, or annotated with why it stays
- [ ] docs/status.md test totals are recounted from an actual run (with the command and date), not copied
- [ ] docs/reference and site are regenerated from the corrected markdown

## Planned files
- ojas-cuda/README.md
- ojas-hip/README.md
- docs/backends.md
- docs/status.md
- docs/pytorch-parity-plan.md
- docs/metal-deferred-faults.md
- docs/typed-storage-plan.md
- docs/adaptive-resources.md
- docs/framework-design.md
- ojas-simd/src/arch.rs
- docs/reference/
- site/
