---
id: "gp-docs-and-bench-parity"
title: "Synchronize Reference Documentation and Complete Quiet Benchmark Parity Runs"
status: ready
priority: 1
severity: medium
type: chore
owner: "unassigned"
due: "none"
labels:
  - "docs"
  - "bench"
  - "evidence"
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
  - "bench/"
  - "bench/results/"
  - "docs/bench-gpu-vs-torch.md"
acceptance_criteria:
  - "Every checklist item identified in docs audit is fixed or annotated with why it stays"
  - "docs/status.md test totals are recounted from an actual run (with command and date), not copied"
  - "docs/reference and site are regenerated from corrected markdown"
  - "A full quiet GPU-vs-torch re-run is committed under bench/results/ with spread per row; rows still over 10% are marked as such in docs"
  - "parity.sh, block_ab.sh and the sample profile are run for typed storage, and result is recorded in docs/typed-storage-plan.md"
  - "The ~9% embedding-forward regression in typed storage step 1 is explained or fixed"
---

# Task brief v1

## Title
Synchronize Reference Documentation and Complete Quiet Benchmark Parity Runs

Task: gp-docs-and-bench-parity
Type: chore
Status: ready
Priority: 1 (High)
Severity: medium
Owner: unassigned
Due: none
Labels: docs, bench, evidence

## Repositories
- ojas

## Description
Consolidated task uniting reference documentation synchronization (`gp-docs-sync-2026-10-05`) and empirical quiet benchmark validation (`gp-bench-quiet-rerun`).

Multiple documents contain statements contradicted by codebase evolution following commit 1f26a4b (CUDA status, test counts, BF16 emulation semantics, residue items). Concurrently, benchmark claims in `docs/pytorch-parity-plan.md` and `docs/typed-storage-plan.md` require empirical execution on a quiet machine adhering to the 10% spread rule.

### Audit status and checklist (2026-10-05)
1. **Documentation Contradictions:**
   - `ojas-cuda/README.md`, `docs/backends.md:32,74`, `docs/status.md:250,289-292` state `ojas-cuda` is a probe that does not implement `Backend`. In fact, `impl Backend for CudaBackend` exists (`ojas-cuda/src/backend.rs:73`), but compute ops refuse pending provider completion.
   - Commit 1f26a4b notes ("BF16 autocast tier", "CUDA backend") overstate implementation; clarify that BF16 is emulated in f32 storage and CUDA step is currently a stub.
   - `ojas-hip/README.md` names `HipDevice::affine_f32`, which does not exist; the actual symbol is `copy_roundtrip`.
   - `docs/status.md`: line 60 notes tasks/ holds 9 tasks (update to reflect unified suite); line 62 has empty heading; line 310 states Metal head dim cap is 64 (it is 256); line 121 says `gpu_real_2b` faults on gradient read, whereas `ojas-qwen35/README.md` notes recycle step fixed it; line 284 labels `ojas-infer` CPU-only, but its README documents device decoder on wgpu and Metal.
   - Test counts disagree: `docs/backends.md:71-72` quotes Metal 136, wgpu 208/3 ignored; `docs/status.md:19-20` quotes 175 and 131. Recount from live run.
   - Clear closed residue in `docs/pytorch-parity-plan.md` (F10 Go error kind, CUDA pin, Linux CI tests, CAPI demo double-charge, ojas-io README names).
   - Re-render `docs/reference/` and `site/`.
2. **Benchmark Verification:**
   - Torch-parity ratios in `docs/bench-gpu-vs-torch.md:518` did not pass the 10% spread rule under noisy background loads. Quiet re-runs are required before quoting ratios.
   - Typed host storage plan (`docs/typed-storage-plan.md`) success criteria need execution of `parity.sh`, `block_ab.sh`, and `sample` profile, investigating the ~9% embedding-forward delta.

## Acceptance criteria
- [ ] Every checklist item identified in docs audit is fixed or annotated with why it stays
- [ ] docs/status.md test totals are recounted from an actual run (with command and date), not copied
- [ ] docs/reference and site are regenerated from corrected markdown
- [ ] A full quiet GPU-vs-torch re-run is committed under bench/results/ with spread per row; rows still over 10% are marked as such in docs
- [ ] parity.sh, block_ab.sh and the sample profile are run for typed storage, and result is recorded in docs/typed-storage-plan.md
- [ ] The ~9% embedding-forward regression in typed storage step 1 is explained or fixed

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
- bench/
- bench/results/
- docs/bench-gpu-vs-torch.md
