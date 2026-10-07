---
id: "gp-ci-main-green"
title: "Main CI is red on every push: Linux job stops at fmt so Linux clippy and cargo test never run; CUDA compile-only job failed on missing bins"
status: ready
priority: 0
severity: high
type: bug
owner: "unassigned"
due: "none"
labels:
  - "ci"
  - "build"
  - "linux"
  - "fmt"
  - "cuda"
repositories:
  - "ojas"
planned_files:
  - ".github/workflows/test.yml"
  - "ojas-core/src/autocast.rs"
  - "rustfmt.toml"
acceptance_criteria:
  - "The test.yml run on main passes every job (linux fmt/clippy/test, CUDA / HIP compile-only, macos, go) on a pushed commit; the run id is recorded in docs/status.md. Local cargo fmt --all --check is not the gate: the shared checkout carries peer lanes' uncommitted diffs"
  - "Committed sources at HEAD are rustfmt-clean (today `git show HEAD:ojas-core/src/autocast.rs | rustfmt --check` reports 3 diffs at the lines CI flags: 1085, 1146, 1481)"
  - "The fmt step formats only ojas crates: `cargo fmt --all` follows path dependencies into the sibling tessl/gusset checkouts, so their formatting can fail ojas CI; the step lists ojas packages (-p ...) or the workflow documents why tessl formatting is ojas's gate"
  - "The CUDA / HIP compile-only job is confirmed green once c74f3ba (which adds ojas-cuda/src/bin/rung0.rs and runga.rs) is pushed; if not, the missing-target cause is fixed"
  - "A guard keeps manifests and tracked files in sync: a [[bin]]/[[test]]/[[example]] path that is not tracked fails a cheap pre-push or CI check before the expensive jobs run"
  - "Workflow actions move off the deprecated Node 20 runtime, and the ubuntu-latest move to Ubuntu 26 (2026-10-19) is either pinned or checked green"
---

# Task brief v1

## Title
Main CI is red on every push: Linux job stops at fmt so Linux clippy and cargo test never run; CUDA compile-only job failed on missing bins

Task: gp-ci-main-green
Type: bug
Status: ready
Priority: 0 (Urgent)
Severity: high
Labels: ci, build, linux, fmt, cuda

## Repositories
- ojas

## Description
Second gap audit 2026-10-07. Labels: [V] re-read by the auditor; [A] read by an audit subagent, not re-read.

**The last five `test.yml` runs on main all failed [V]** (`gh run list --workflow test.yml --branch main`): 37421451553 (268c640, 2026-10-06), 37365079870, 37360470939, 37354165992, 37353002002. In 37421451553:
- `linux (fmt, clippy, test)` fails at the `fmt` step (test.yml:81) with `Diff in .../ojas-core/src/autocast.rs:1085`, `:1146`, `:1481`. Linux clippy and `cargo test` therefore never run, so every "Linux CI" claim in the docs rests on a job that stops before testing (docs/pytorch-parity-plan.md:165, :184 [A]).
- `CUDA / HIP compile-only` fails at "ojas-cuda with the cuda feature" (test.yml:242) with `can't find bin rung0 at path ojas-cuda/src/bin/rung0.rs` (and `runga`), exit 101. Those files are tracked at HEAD c74f3ba, which is one of three commits not yet pushed (`main...origin/main [ahead 3]`), so this half is probably fixed on push [I].
- `go (ubuntu-latest)`, `go (macos-latest)` and `macos` pass.

**fmt at HEAD still fails [V]:** `git show HEAD:ojas-core/src/autocast.rs | rustfmt --edition 2021 --check` prints 3 `Diff in` hunks (rustfmt's stdin `--check` exits 0 on a diff, so count the hunks). The local working tree shows ~70 diffs [A], but most belong to peer lanes' uncommitted work and to the sibling tessl checkout that `cargo fmt --all` reaches through path dependencies.

Why separate from gp-ci-and-repo-hygiene: that card's "CI stays green" criterion assumes main is green today. It is not, and nothing else on the board says so.
