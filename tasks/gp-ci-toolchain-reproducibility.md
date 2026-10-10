---
id: "gp-ci-toolchain-reproducibility"
title: "Reproducible toolchain and missing CI jobs: silent-pass tiktoken test, pinned oracle env, MSRV and rustdoc jobs, workspace lints, local gate parity, oracle fixture provenance, scripts that exit 0 on failure"
status: backlog
priority: 2
severity: medium
type: chore
owner: "unassigned"
due: "none"
labels:
  - "ci"
  - "toolchain"
  - "reproducibility"
  - "oracle"
  - "lints"
  - "housekeeping"
repositories:
  - "ojas"
planned_files:
  - "ojas-data/src/bpe.rs"
  - ".github/workflows/test.yml"
  - "Cargo.toml"
  - "ojas-oracle/python/"
  - "ojas-oracle/README.md"
  - "bench/torch_rows.py"
  - "bench/run_paired.sh"
  - "bench/aggregate.py"
  - "ojas-cpu/benches/summarize_ops.py"
  - "scripts/ci_local.sh"
  - "scripts/simd_bench.sh"
  - ".gitignore"
acceptance_criteria:
  - "verified_20_strings_match_tiktoken_encode_ordinary (ojas-data/src/bpe.rs:1299-1316, hard-coded MLSystemsLab path at :1303) fails instead of returning when no GPT-2 rank table is found, and the second silent branch at :1334-1339 fails too; the table is vendored with its license checked (the untracked ojas-data/testdata/gpt2/ is decided here) or fetched by CI; TIKTOKEN_GPT2_BYTE_IDENTITY = \"verified-20-strings\" (bpe.rs:30), today asserted only against its own literal (:1039, :1313), is backed by that test"
  - "CI gains: a rustdoc build with -D warnings, an MSRV job on 1.97 (Cargo.toml rust-version) and a committed rust-toolchain.toml so a new stable clippy lint cannot break main (run 37855710888 failed on clippy 1.99 `manual_is_multiple_of`), gofmt -l and go vet, the determinism check, and a dependency audit"
  - "The Python oracle environment is pinned in a tracked uv lock/pyproject matching ojas-oracle/README.md:143 (torch 2.13, Python 3.14.7, numpy); regen_tiny.sh and bench/run_paired.sh stop hard-coding /opt/homebrew python; bench/torch_rows.py honours OJAS_NANOLAB_ROOT like ojas-oracle/python/common.py:28 instead of the hard-coded NANOLAB_ROOT (:50)"
  - "Python that is neither oracle nor binding is ported to Go/Rust or deleted: bench/aggregate.py, ojas-cpu/benches/summarize_ops.py, bench/results/2026-10-04-sweep/sweep_fit.py, and new since the brief bench/plot_all.py (706 lines), bench/plotlib.py (210) and the summarize/agg .py files in five bench/results/2026-10-08-* folders"
  - "Lint policy and shared deps move to [workspace.lints] / [workspace.dependencies]: one unsafe_code stance (forbid in 13 crates, deny in ojas-device/ojas-simd, none in ojas-cuda/ojas-hip/ojas-qwen35/ojas-gusset-engine), `ojas-core =` declared once instead of 15 times; libc is already pinned (0.2.189); [workspace.package] metadata the uncommitted Cargo.toml adds is either inherited by crates or dropped"
  - "scripts/ci_local.sh runs the same gate definition as test.yml under Lappi-decision/tools/mac_heavy.sh (today :19 runs `cargo check`, :2-3 claim 'no remote', and nothing takes the shared lock); scripts/simd_bench.sh stops printing 'skipped' and exiting 1 (:9-10)"
  - "Root and tree clutter that is not lane build output is removed or ignored: ojas_cpu-*.rcgu.o at the repo root, .site-patch/, bench/__pycache__, bench/out, empty docs/guide and site/guide, .DS_Store files"
  - "The nanolab revision the oracle fixtures come from is pinned and checked: today fixtures/tiny mixes two revisions (muon_step_bf16.safetensors records nanolab_head d3ceea15, the rest 07c32626), regen_tiny.sh uses whatever checkout is present, common.py:244-262 does not refuse a dirty nanolab tree, and golden.rs:244-246 checks only that the value looks like a commit id"
  - "ojas-oracle/python/check_determinism.sh compares regenerated fixtures against the committed files (git diff --exit-code or a byte compare into a temp dir) instead of overwriting the tracked fixtures/tiny and comparing runs only with each other (script lines 2-3, 19-25), so it can run in CI without dirtying the tree"
  - "A test compares the live BatchSampler with the committed batch_starts.json that forward.safetensors was built on (ojas-oracle/tests/golden.rs:329-354 checks only the starts against tokens.bin); ojas-oracle verifies the sha256 its gdn.rs:21-22 doc claims for the GDN goldens (today only ojas-cuda/tests/fixture_pins.rs does) and read_npy_f64 caps the file size"
  - "Runner scripts fail when a step fails or is skipped: bench/run_paired.sh:87-88 writes a crash row and carries on, aggregate.py never exits non-zero, gotip.sh:11 exits 0 so ci_local.sh:47 prints OK"
  - "The stale MSRV comment at ojas-simd/src/arch.rs:9 ('the workspace MSRV is 1.82'; Cargo.toml says 1.97) is corrected, and the value-only intrinsics it wraps in unsafe are unwrapped if 1.97 allows it"
  - "clippy::undocumented_unsafe_blocks is enabled where unsafe is allowed (ojas-simd, ojas-device, ojas-cuda, ojas-hip), and the sites it finds get true // SAFETY: comments: open gaps are ojas-simd/src/arch.rs (708, 900, 1355-1463), ojas-device/src/host.rs:443-471, bandwidth.rs, ojas-cuda runtime.rs; transpose_8x8_block's comment claims 16-byte alignment no caller establishes (arch.rs:743-756, wrapper lib.rs:844-866), and ojas-cuda lib.rs:311/:375 describe the flags argument rather than cudarc's real precondition"
  - "rustdoc builds with -D warnings: today 53 errors in 12 crates. Most are public docs linking private items (e.g. ojas-metal/src/backend.rs:1131,1170,1934,2141; ojas-core/src/backend.rs:1187,1222,1223,1247); about 20 in ojas-cuda link items that exist only with the cuda feature (backend.rs:19, budget.rs:3, wait.rs:6); plus ojas-wgpu/src/context.rs:96 -> linux_pressure and ojas-cuda/src/geometry.rs:91. Fix them before adding the rustdoc job"
  - "The MSRV job checks what ships: `cargo +1.97 check --workspace` passes, but only for default features without --all-targets, so test and example code and ojas-cuda's cuda feature are unchecked on 1.97"
---

# Task brief v1

## Title
Reproducible toolchain and missing CI jobs: silent-pass tiktoken test, pinned oracle env, MSRV and rustdoc jobs, workspace lints, local gate parity, oracle fixture provenance, scripts that exit 0 on failure

Task: gp-ci-toolchain-reproducibility
Type: chore
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: ci, toolchain, reproducibility, oracle, lints, housekeeping

## Repositories
- ojas

## Description
Gap audit 2026-10-07. Separate from `gp-ci-and-repo-hygiene`, which owns test parallelism, readback accounting, Metal CI protocol, sibling pins, bare ignores, scan.txt/analyze.py and the soak job. This card owns CI *job coverage* and environment reproducibility. V = verified, I = inferred.

- **A test that checks nothing off this machine [V]:** `ojas-data/src/bpe.rs:1071-1088`: `rank_table_dir()` looks in `ojas-data/testdata/gpt2` (absent, untracked) and a `/Users/bharath/.../MLSystemsLab/...` path, and the test `return`s when neither exists, so CI passes it vacuously. This violates "a check that could not run must never report the same result as one that passed".
- **Missing jobs [V]:** test.yml has fmt, clippy, test, Go test and `sitegen -check`; it has no rustdoc, MSRV, gofmt/vet, oracle determinism or dependency audit. `rust-version = "1.97"` (Cargo.toml:31) but every job installs stable and there is no rust-toolchain.toml.
- **Unpinned oracle env [V]:** no tracked requirements/pyproject/uv.lock; `bench/torch_rows.py:50` hard-codes `NANOLAB_ROOT = "/Users/bharath/Code/research/MLSystemsLab"`.
- **Policy Python [V]:** aggregate.py (241 lines), summarize_ops.py (176), sweep_fit.py (128) are reporting code, not oracles. torch_rows.py, torch_ops.py, torch_mps.py and ojas-oracle/python stay (legitimate oracles).
- **Local gate drift [V]:** `scripts/ci_local.sh:2-3`, `scripts/simd_bench.sh:9-11`.

Do not delete the target-* lane dirs here: they follow the lane convention, and `gp-docs-pipeline-single-source` must first rescue the evidence they hold.

**Added by the second gap audit 2026-10-07** ([V] re-read by the auditor; [A] read by an audit subagent, not re-read):
- **Two nanolab revisions in one fixture set [V]:** `rg -a 'nanolab_head'` over fixtures/tiny shows `d3ceea15...` in muon_step_bf16.safetensors and `07c32626...` in forward, init and trace_ns5_bf16. The pinned-environment criterion above covers Python and OJAS_NANOLAB_ROOT, not the nanolab revision.
- **Stale MSRV comment [V]:** ojas-simd/src/arch.rs:9 "the workspace MSRV is 1.82".
- **check_determinism.sh, batch_starts.json, gdn sha, run_paired/aggregate exit codes, ci_local after a skipped gotip, unsafe without SAFETY in ojas-cuda/ojas-hip [A]** (ojas-cuda/src/lib.rs launch block re-read [V]).
- Main CI is currently red; that is gp-ci-main-green, not this card.

### Re-audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- Done: root clutter (rcgu.o, .site-patch gone; bench/__pycache__, bench/out, .DS_Store ignored) [A, C].
- PARTIAL: libc pinned; no [workspace.lints] yet [A].
- Stale claims removed: the brief said ~18 HIP and one CUDA unsafe sites lacked SAFETY; both are now commented [A, C].
- Fixture provenance criteria 8-10 stay here; they overlap gp-oracle-and-hardening-coverage, which owns new goldens, not provenance.

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- A constant named 'verified' is asserted only against its own literal. Two SAFETY comments state the wrong precondition [A].

### Gate run (2026-10-09, clean mirror of d431949)
Run on a `git archive` mirror of HEAD under mac_heavy.sh [C]. Toolchain: rustc 1.99.0, clippy 0.1.99, rustfmt 1.10.0, go 1.27.2. Siblings built: tessl 56b62c05 (clean); gusset 159d352a (31 dirty entries, all agent/editor config, none under crates/ or internal/). Logs are in this session's scratchpad gates/logs/.

| Gate | Result |
|---|---|
| cargo fmt --all --check (and the 18-package -p list) | FAIL, 31 hunks, all in ojas files |
| clippy --workspace --all-targets -D warnings | PASS |
| cargo test --workspace --release, --test-threads=2, OJAS_ALLOW_NO_GPU unset | PASS: 1825 passed, 0 failed, 41 ignored, 211 suites; Metal and wgpu ran on the real GPU, no SKIP lines |
| rustdoc with -D warnings | FAIL, 53 errors in 12 crates |
| cargo +1.97 check --workspace | PASS (default features, no --all-targets) |
| CUDA clippy + --no-run with --features cuda; ojas-hip check | PASS |
| Go: engine build, go vet, go test -tags gusset_pkgconfig, gofmt -l | PASS |
| sitegen -check (clean clone) | FAIL: docs/bench-plots.md has no category |

## Acceptance criteria
- [ ] verified_20_strings_match_tiktoken_encode_ordinary (ojas-data/src/bpe.rs:1299-1316, hard-coded MLSystemsLab path at :1303) fails instead of returning when no GPT-2 rank table is found, and the second silent branch at :1334-1339 fails too; the table is vendored with its license checked (the untracked ojas-data/testdata/gpt2/ is decided here) or fetched by CI; TIKTOKEN_GPT2_BYTE_IDENTITY = "verified-20-strings" (bpe.rs:30), today asserted only against its own literal (:1039, :1313), is backed by that test
- [ ] CI gains: a rustdoc build with -D warnings, an MSRV job on 1.97 (Cargo.toml rust-version) and a committed rust-toolchain.toml so a new stable clippy lint cannot break main (run 37855710888 failed on clippy 1.99 `manual_is_multiple_of`), gofmt -l and go vet, the determinism check, and a dependency audit
- [ ] The Python oracle environment is pinned in a tracked uv lock/pyproject matching ojas-oracle/README.md:143 (torch 2.13, Python 3.14.7, numpy); regen_tiny.sh and bench/run_paired.sh stop hard-coding /opt/homebrew python; bench/torch_rows.py honours OJAS_NANOLAB_ROOT like ojas-oracle/python/common.py:28 instead of the hard-coded NANOLAB_ROOT (:50)
- [ ] Python that is neither oracle nor binding is ported to Go/Rust or deleted: bench/aggregate.py, ojas-cpu/benches/summarize_ops.py, bench/results/2026-10-04-sweep/sweep_fit.py, and new since the brief bench/plot_all.py (706 lines), bench/plotlib.py (210) and the summarize/agg .py files in five bench/results/2026-10-08-* folders
- [ ] Lint policy and shared deps move to [workspace.lints] / [workspace.dependencies]: one unsafe_code stance (forbid in 13 crates, deny in ojas-device/ojas-simd, none in ojas-cuda/ojas-hip/ojas-qwen35/ojas-gusset-engine), `ojas-core =` declared once instead of 15 times; libc is already pinned (0.2.189); [workspace.package] metadata the uncommitted Cargo.toml adds is either inherited by crates or dropped
- [ ] scripts/ci_local.sh runs the same gate definition as test.yml under Lappi-decision/tools/mac_heavy.sh (today :19 runs `cargo check`, :2-3 claim 'no remote', and nothing takes the shared lock); scripts/simd_bench.sh stops printing 'skipped' and exiting 1 (:9-10)
- [x] Root and tree clutter that is not lane build output is removed or ignored: ojas_cpu-*.rcgu.o at the repo root, .site-patch/, bench/__pycache__, bench/out, empty docs/guide and site/guide, .DS_Store files
- [ ] The nanolab revision the oracle fixtures come from is pinned and checked: today fixtures/tiny mixes two revisions (muon_step_bf16.safetensors records nanolab_head d3ceea15, the rest 07c32626), regen_tiny.sh uses whatever checkout is present, common.py:244-262 does not refuse a dirty nanolab tree, and golden.rs:244-246 checks only that the value looks like a commit id
- [ ] ojas-oracle/python/check_determinism.sh compares regenerated fixtures against the committed files (git diff --exit-code or a byte compare into a temp dir) instead of overwriting the tracked fixtures/tiny and comparing runs only with each other (script lines 2-3, 19-25), so it can run in CI without dirtying the tree
- [ ] A test compares the live BatchSampler with the committed batch_starts.json that forward.safetensors was built on (ojas-oracle/tests/golden.rs:329-354 checks only the starts against tokens.bin); ojas-oracle verifies the sha256 its gdn.rs:21-22 doc claims for the GDN goldens (today only ojas-cuda/tests/fixture_pins.rs does) and read_npy_f64 caps the file size
- [ ] Runner scripts fail when a step fails or is skipped: bench/run_paired.sh:87-88 writes a crash row and carries on, aggregate.py never exits non-zero, gotip.sh:11 exits 0 so ci_local.sh:47 prints OK
- [ ] The stale MSRV comment at ojas-simd/src/arch.rs:9 ('the workspace MSRV is 1.82'; Cargo.toml says 1.97) is corrected, and the value-only intrinsics it wraps in unsafe are unwrapped if 1.97 allows it
- [ ] clippy::undocumented_unsafe_blocks is enabled where unsafe is allowed (ojas-simd, ojas-device, ojas-cuda, ojas-hip), and the sites it finds get true // SAFETY: comments: open gaps are ojas-simd/src/arch.rs (708, 900, 1355-1463), ojas-device/src/host.rs:443-471, bandwidth.rs, ojas-cuda runtime.rs; transpose_8x8_block's comment claims 16-byte alignment no caller establishes (arch.rs:743-756, wrapper lib.rs:844-866), and ojas-cuda lib.rs:311/:375 describe the flags argument rather than cudarc's real precondition
- [ ] rustdoc builds with -D warnings: today 53 errors in 12 crates. Most are public docs linking private items (e.g. ojas-metal/src/backend.rs:1131,1170,1934,2141; ojas-core/src/backend.rs:1187,1222,1223,1247); about 20 in ojas-cuda link items that exist only with the cuda feature (backend.rs:19, budget.rs:3, wait.rs:6); plus ojas-wgpu/src/context.rs:96 -> linux_pressure and ojas-cuda/src/geometry.rs:91. Fix them before adding the rustdoc job
- [ ] The MSRV job checks what ships: `cargo +1.97 check --workspace` passes, but only for default features without --all-targets, so test and example code and ojas-cuda's cuda feature are unchecked on 1.97

## Planned files
- ojas-data/src/bpe.rs
- .github/workflows/test.yml
- Cargo.toml
- ojas-oracle/python/
- ojas-oracle/README.md
- bench/torch_rows.py
- bench/run_paired.sh
- bench/aggregate.py
- ojas-cpu/benches/summarize_ops.py
- scripts/ci_local.sh
- scripts/simd_bench.sh
- .gitignore
