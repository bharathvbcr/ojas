---
id: "gp-docs-and-bench-parity"
title: "Docs match the code: head-dim/CUDA/GQA/activation claims, op-coverage one row per trait method, README and status test counts, bench rows for new ops"
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
  - "docs/op-coverage.md"
  - "docs/architecture.md"
  - "docs/audit.md"
  - "docs/shape-contract.md"
  - "docs/cuda-backend-scoping.md"
  - "README.md"
  - "site/index.html"
  - "ojas-metal/README.md"
  - "ojas-metal/src/lib.rs"
  - "ojas-infer/README.md"
  - "ojas-oracle/README.md"
  - "ojas-oracle/tests/parity_gates.rs"
  - "ojas-core/src/limits.rs"
  - "bench/README.md"
  - "bench/ojas_rows.rs"
  - "bench/torch_rows.py"
acceptance_criteria:
  - "Every checklist item identified in docs audit is fixed or annotated with why it stays"
  - "docs/status.md test totals are recounted from an actual run (with command and date), not copied (status.md:13 says 'suites not re-run'; README.md:330 still says '1510 passed (2026-10-05)'); generation of these numbers is gp-docs-pipeline-single-source's job, the content is this task's"
  - "docs/reference and site are regenerated from corrected markdown"
  - "2026-10-07 checklist (section 3) fixed: 'head dim 128' (site/index.html:995,1179,1240,1241,1333), 'at most 64' (ojas-metal/src/lib.rs:12-13), CUDA 'probe' (README.md:21,208,351; site/index.html:691,1346,1371) and 'not a Backend' (docs/architecture.md:59-60, while impl Backend for CudaBackend is at ojas-cuda/src/backend.rs:84) match the code"
  - "docs/op-coverage.md has one row per Backend trait method with its status per backend (adds chunked_gdn_forward/backward, cached_attention_forward, cast_bf16, sigmoid, gdn_log_decay and the *_saving variants; 0 rows today) and the file:// link at :87 is repo-relative"
  - "The activation claim at docs/pytorch-parity-plan.md:47 and op-coverage.md:102 is corrected: SiLU and Sigmoid exist (sigmoid_forward/backward, ojas-core/src/backend.rs:984,990); GELU, ReLU and Tanh do not and are either filed or stated as missing"
  - "bench rows exist for the new trait ops: GDN, causal conv1d, gated RMSNorm, partial RoPE, sliding-window SDPA, GQA SDPA (Hkv < H), SDPA at d=256 (ojas_rows.rs:481-483 has d64/d128 only); the bench/README.md:73 'recomputes' claim is corrected (the saved-LSE backward landed); the gate-saved citation is already done"
  - "Section 4 (second gap audit, 2026-10-07) fixed: present-but-called-missing claims (activation checkpointing, CPU conv1d/gated RMSNorm, typed DeviceLost), closed residue in pytorch-parity-plan.md (:180, :186, :198-199, :228), the Linux CI claims stated exactly (Go on Linux passes; Rust Linux tests blocked at fmt), the extra metal-deferred-faults.md section 9.4 lines, the wgpu QK-norm description, audit-resources.md S3/S6, dtype-policy.md:56, and audit-phase2/phase3/close marked historical"
  - "docs/checkpoint-v1.md matches the code: OjasError::InvalidCheckpoint and TruncatedCheckpoint do not exist (the code returns OutOfRange or IoError), step == u64::MAX is accepted by the reader and refused later at next_step; ojas-model/src/load.rs:124 'checked finite' contradicts its module doc at :16"
  - "The Metal simd_sum ordering claims agree: ojas_backend.metal (~:474) says fixed order, ojas-cuda/src/gdn_kernels.rs:31-34 says unspecified; the run-to-run repeatability claims for Metal RMSNorm, attention dr and cached attention state which is true, backed by a repeat-run test"
---

# Task brief v1

## Title
Docs match the code: head-dim/CUDA/GQA/activation claims, op-coverage one row per trait method, README and status test counts, bench rows for new ops

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
3. **Gap audit additions (2026-10-07, rg + Read; V = verified, I = inferred):**
   - Head dim (code: 256 at `ojas-core/src/backend.rs:64`, `ojas-kernels/src/geometry.rs:166`, `ojas-metal/src/device.rs:88`) [V]: "128" at `site/index.html:995,1179,1240,1241,1333` (same lines in the `docs/index.html` mirror), `docs/status.md:182`, `docs/pytorch-parity-plan.md:257`, `docs/cuda-backend-scoping.md:111,133` (or mark as a dated record); "64" at `ojas-metal/src/lib.rs:12-13` and `docs/audit.md` freeze F2 / defense R2.
   - CUDA called a probe / not a Backend [V]: `README.md:21,29,122,208,351,403`, `docs/architecture.md:59`, `site/index.html:691,972,1035,1242`.
   - GQA [V]: "repeats KV heads" at `ojas-metal/README.md:118` and `README.md:205` (native GQA now, `ojas-metal/README.md:85`); "Training v1 refuses GQA" at `docs/framework-design.md:49`; `ojas-infer/README.md:28` ("only the training tape refuses GQA").
   - "Every Backend op" on Metal [V]: `README.md:205,337`, `docs/backends.md` status table, `site/index.html:1240` — Metal inherits the refusing defaults for causal conv1d, gated RMSNorm and partial RoPE. `site/index.html:1023` still asks whether the Qwen3.5 ops move into the trait; they have.
   - Counts [V]: `README.md:33` says 1510 passed while its per-crate numbers sum to 1347; `README.md:360` "run 4: 1071"; `site/index.html:80,182` "1280+", `:1328,1352` "1286", `:1333` ojas-metal "136".
   - Other stale text [V]: `docs/architecture.md:47` calls ojas-infer CPU-only; `ojas-oracle/README.md:13-15` and `ojas-oracle/tests/parity_gates.rs:6` say parity runs "once ojas-model implements ParityModel" (it does); `docs/typed-storage-plan.md:6` "Steps 2-4 are not started" vs `:209-223`; `docs/shape-contract.md:10` says CPU has not adopted the validators (it has, e.g. `ojas-cpu/src/backend.rs:396,412,997,1035,1069`); `ojas-core/src/limits.rs:22` calls the `with_threads` refusal "a later change" (`ojas-cpu/src/pool.rs:37-38` does it); `docs/metal-deferred-faults.md` §9.4 says the capi early-return test is wgpu-only (`ojas-capi/src/tests.rs:2109` is the Metal twin).
   - op-coverage.md is missing trait methods (see criteria) [V]; `pytorch-parity-plan.md:47` claims GELU/ReLU/Sigmoid/Tanh, none of which exist in Rust (rg) [V].
   - bench [V]: `bench/README.md:34` cites `results/2026-10-06-gate-saved/` (absent); `:98-99` says backward recomputes, but SDPA backward takes saved `o`/`lse` (`bench/ojas_rows.rs:647-651`); SDPA rows only d64/d128 (`ojas_rows.rs:481-483`), decode only H = Hkv (`:886`).
   - The pipeline that lets this recur (site mirror, absolute links, ungated deploy, hand-copied counts) is `gp-docs-pipeline-single-source`.
4. **Second gap audit additions (2026-10-07; [V] re-read by the auditor, [A] read by an audit subagent, not re-read):**
   - Claimed missing, actually present: `docs/pytorch-parity-plan.md:53` "Activation checkpointing: Missing" and `docs/framework-design.md:160` "phase 2" [V: `ojas-model/src/trainer.rs:127` `pub activations: ActivationCheckpoint`; `block.rs:340` [A]]; `pytorch-parity-plan.md:59` calls causal conv1d and gated RMSNorm missing, but CPU implements them (`ojas-cpu/src/backend.rs:807,833,869`) [A]; `pytorch-parity-plan.md:187` "no typed device-lost variant" [V: `ojas-core/src/error.rs:48` `DeviceLost {`].
   - Residue now closed [A]: `pytorch-parity-plan.md:186` (`commit_resize` is gone from ojas-cuda/src), `:180` (empty-tensor permute now refused on every backend, `ojas-core/src/shapes.rs:272`, shape-contract D17), `:198-199` (capi `step.rs`/`run_part` gone), `:228` (`MAX_HEADER_SIZE`/`InvalidSafetensors` no longer exist).
   - Linux claims [V via CI]: `pytorch-parity-plan.md:184` "no binary has run on Linux" and `docs/adaptive-resources.md:250` "Go tests ran on macOS only" are out of date (`go (ubuntu-latest)` passes), but Rust Linux tests still never run because main CI stops at fmt (gp-ci-main-green). Say exactly that.
   - `docs/metal-deferred-faults.md` §9.4, two more stale lines [A]: the `bench/ojas_rows.rs:21-22` reference (lines 15-26 are now right) and the `Backend::sync` doc (`ojas-core/src/backend.rs:1021-1028` now names Metal).
   - `docs/bench-gpu-vs-torch.md` says wgpu QK-norm runs "two calls, 3 of 4 lanes idle" [A, line number unconfirmed]; `ojas-wgpu/src/backend.rs:1664-1679` is one job and `norm.wgsl:1-4` packs 4 rows per workgroup.
   - `docs/audit-resources.md:55` (S3, go/api.go "not emitting prefixes yet") and `:58` (S6, greedy is a two-token demo; go/api.go:523 now wraps GenerateIDs) are stale [A]; `docs/dtype-policy.md:56` "u32 vocab up to 50,304" understates u32 bins at V=248320 [I].
   - `docs/audit-phase2.md`, `audit-phase3.md` and `audit-close.md:65-68` describe `ojas-nn`, a tiny-step-only Metal step, CpuGpt without RoPE and "12 layers at 768 not implemented", all superseded by `ModelSpec::nanolab_124m`; mark them historical as `audit-cpu-hot.md:5` already is [A].
   - `pytorch-parity-plan.md:51` sends "the gates: missing" to gp-autograd-and-model-primitives, whose acceptance criteria do not list the GDN gates (only its progress notes do); keep the pointer accurate once that lane closes [V].

### Re-audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- Moved out: the quiet GPU-vs-torch rerun, the typed-storage parity.sh/block_ab.sh run and the ~9% embedding regression are measurement runs, not text fixes; they now live in gp-long-runs-and-quiet-benches. The typed-storage scripts are in the deleted target-matmul/, so that run depends on gp-docs-pipeline-single-source criteria 1-2.
- Brief WRONG and corrected: 'only SiLU exists' is false since 66c5e52 added Sigmoid [A].
- Overlap: the GQA decode bench row belongs to gp-inference-decode-path's decode benchmark.

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- New: checkpoint-v1.md names error variants that don't exist; Metal simd_sum determinism is claimed both ways [A].

## Acceptance criteria
- [ ] Every checklist item identified in docs audit is fixed or annotated with why it stays
- [ ] docs/status.md test totals are recounted from an actual run (with command and date), not copied (status.md:13 says 'suites not re-run'; README.md:330 still says '1510 passed (2026-10-05)'); generation of these numbers is gp-docs-pipeline-single-source's job, the content is this task's
- [ ] docs/reference and site are regenerated from corrected markdown
- [ ] 2026-10-07 checklist (section 3) fixed: 'head dim 128' (site/index.html:995,1179,1240,1241,1333), 'at most 64' (ojas-metal/src/lib.rs:12-13), CUDA 'probe' (README.md:21,208,351; site/index.html:691,1346,1371) and 'not a Backend' (docs/architecture.md:59-60, while impl Backend for CudaBackend is at ojas-cuda/src/backend.rs:84) match the code
- [ ] docs/op-coverage.md has one row per Backend trait method with its status per backend (adds chunked_gdn_forward/backward, cached_attention_forward, cast_bf16, sigmoid, gdn_log_decay and the *_saving variants; 0 rows today) and the file:// link at :87 is repo-relative
- [ ] The activation claim at docs/pytorch-parity-plan.md:47 and op-coverage.md:102 is corrected: SiLU and Sigmoid exist (sigmoid_forward/backward, ojas-core/src/backend.rs:984,990); GELU, ReLU and Tanh do not and are either filed or stated as missing
- [ ] bench rows exist for the new trait ops: GDN, causal conv1d, gated RMSNorm, partial RoPE, sliding-window SDPA, GQA SDPA (Hkv < H), SDPA at d=256 (ojas_rows.rs:481-483 has d64/d128 only); the bench/README.md:73 'recomputes' claim is corrected (the saved-LSE backward landed); the gate-saved citation is already done
- [ ] Section 4 (second gap audit, 2026-10-07) fixed: present-but-called-missing claims (activation checkpointing, CPU conv1d/gated RMSNorm, typed DeviceLost), closed residue in pytorch-parity-plan.md (:180, :186, :198-199, :228), the Linux CI claims stated exactly (Go on Linux passes; Rust Linux tests blocked at fmt), the extra metal-deferred-faults.md section 9.4 lines, the wgpu QK-norm description, audit-resources.md S3/S6, dtype-policy.md:56, and audit-phase2/phase3/close marked historical
- [ ] docs/checkpoint-v1.md matches the code: OjasError::InvalidCheckpoint and TruncatedCheckpoint do not exist (the code returns OutOfRange or IoError), step == u64::MAX is accepted by the reader and refused later at next_step; ojas-model/src/load.rs:124 'checked finite' contradicts its module doc at :16
- [ ] The Metal simd_sum ordering claims agree: ojas_backend.metal (~:474) says fixed order, ojas-cuda/src/gdn_kernels.rs:31-34 says unspecified; the run-to-run repeatability claims for Metal RMSNorm, attention dr and cached attention state which is true, backed by a repeat-run test

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
- docs/op-coverage.md
- docs/architecture.md
- docs/audit.md
- docs/shape-contract.md
- docs/cuda-backend-scoping.md
- README.md
- site/index.html
- ojas-metal/README.md
- ojas-metal/src/lib.rs
- ojas-infer/README.md
- ojas-oracle/README.md
- ojas-oracle/tests/parity_gates.rs
- ojas-core/src/limits.rs
- bench/README.md
- bench/ojas_rows.rs
- bench/torch_rows.py
