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
  - "verified_20_strings_match_tiktoken_encode_ordinary (ojas-data/src/bpe.rs:1084) fails instead of returning when no GPT-2 rank table is found; the table is vendored under ojas-data/testdata/gpt2 (license checked) or fetched in CI; the /Users/bharath/.../MLSystemsLab fallback path is removed"
  - "CI gains: a rustdoc build with -D warnings, an MSRV job on 1.97 (Cargo.toml rust-version) or a committed rust-toolchain.toml, gofmt -l and go vet, the oracle check_determinism.sh, and a dependency audit (cargo-deny or cargo-audit; adding either tool needs the user's approval)"
  - "The Python oracle environment is pinned in a tracked uv lock/pyproject matching ojas-oracle/README.md:143 (torch 2.13, Python 3.14.7, numpy); regen_tiny.sh and bench/run_paired.sh stop hard-coding /opt/homebrew python; bench/torch_rows.py honours OJAS_NANOLAB_ROOT like ojas-oracle/python/common.py:28 instead of the hard-coded NANOLAB_ROOT (:50)"
  - "Python that is neither oracle nor binding (bench/aggregate.py, ojas-cpu/benches/summarize_ops.py, bench/results/2026-10-04-sweep/sweep_fit.py) is ported to Go or Rust with an output-equality check against the Python, then the Python is deleted in the same change"
  - "Lint policy and shared deps move to [workspace.lints] / [workspace.dependencies]: one unsafe_code stance (today forbid in most crates, deny in ojas-device/ojas-simd, none in ojas-qwen35/ojas-gusset-engine), ojas-core path declared once, libc pinned like every other dependency"
  - "scripts/ci_local.sh runs the same gate definition as test.yml (its 'no remote' comment is false: origin is github.com/bharathvbcr/ojas; it runs cargo check on cuda where CI runs clippy plus test --no-run); scripts/simd_bench.sh no longer exits 1 while printing 'skipped', and its std::simd gate is deleted or made meaningful (no workspace crate uses std::simd)"
  - "Root and tree clutter that is not lane build output is removed or ignored: ojas_cpu-*.rcgu.o at the repo root, .site-patch/, bench/__pycache__, bench/out, empty docs/guide and site/guide, .DS_Store files"
  - "The nanolab revision the oracle fixtures come from is pinned and checked: today fixtures/tiny mixes two revisions (muon_step_bf16.safetensors records nanolab_head d3ceea15, the rest 07c32626), regen_tiny.sh uses whatever checkout is present, common.py:244-262 does not refuse a dirty nanolab tree, and golden.rs:244-246 checks only that the value looks like a commit id"
  - "ojas-oracle/python/check_determinism.sh compares regenerated fixtures against the committed files (git diff --exit-code or a byte compare into a temp dir) instead of overwriting the tracked fixtures/tiny and comparing runs only with each other (script lines 2-3, 19-25), so it can run in CI without dirtying the tree"
  - "A test compares the live BatchSampler with the committed batch_starts.json that forward.safetensors was built on (ojas-oracle/tests/golden.rs:329-354 checks only the starts against tokens.bin); ojas-oracle verifies the sha256 its gdn.rs:21-22 doc claims for the GDN goldens (today only ojas-cuda/tests/fixture_pins.rs does) and read_npy_f64 caps the file size"
  - "Runner scripts fail when a step fails or is skipped: bench/run_paired.sh's run_lane writes a crash row and carries on and aggregate.py never exits non-zero, so a run where every lane crashed exits 0; scripts/ci_local.sh prints 'ci:local OK' after gotip.sh exits 0 on 'skipped (unverified)'"
  - "The stale MSRV comment at ojas-simd/src/arch.rs:9 ('the workspace MSRV is 1.82'; Cargo.toml says 1.97) is corrected, and the value-only intrinsics it wraps in unsafe are unwrapped if 1.97 allows it"
  - "clippy::undocumented_unsafe_blocks is enabled where unsafe is allowed (ojas-simd, ojas-device, ojas-cuda, ojas-hip), and the sites it finds get // SAFETY: comments (ojas-cuda/src/lib.rs:~401 has none; ojas-hip/src/lib.rs:135-298 has ~18 without)"
---

# Task brief v1

## Title
Reproducible toolchain and missing CI jobs: silent-pass tiktoken test, pinned oracle env, MSRV and rustdoc jobs, workspace lints, local gate parity, oracle fixture provenance, scripts that exit 0 on failure

Task: gp-ci-toolchain-reproducibility
Type: chore
Status: backlog
Priority: 2 (Normal)
Severity: medium
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
