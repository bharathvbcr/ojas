---
id: "gp-ci-main-green"
title: "Main CI is red on every push: Linux fmt fails on ~15 files, sitegen -check fails on bench-plots.md, and Linux tests have not run since 85d1f5a (2026-10-05)"
status: ready
priority: 0
severity: critical
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
  - "scripts/sitegen/main.go"
acceptance_criteria:
  - "The test.yml run on main passes every job, including the Linux `cargo test` step (not only fmt and clippy), on a pushed commit; the run id is recorded in docs/status.md. Local cargo fmt --all --check is not the gate: the shared checkout carries peer lanes' uncommitted diffs"
  - "Committed sources at HEAD are rustfmt-clean under the CI toolchain (rustfmt 1.10.0): the 31 hunks at d431949 are ojas-core src/tensor.rs:2201; ojas-cpu tests/shape_first.rs:590; ojas-infer src/decode.rs:10, device.rs:306, gpt.rs:427; ojas-metal src/device.rs:37,3177, tests/clip_multi.rs:38, tests/shape_first.rs:798; ojas-wgpu src/backend.rs:240,3019, src/context.rs:1317,2105,2191,2212,2221, tests/dispatch_cost.rs:66,73,79,150, tests/shape_first.rs:757, examples/step_probe.rs:16; bench/ojas_rows.rs:860 and bench/step_probe.rs:404,521"
  - "The fmt step formats only ojas packages (-p list) so a future tessl/gusset formatting change cannot fail ojas CI; today every flagged diff is ojas's own code"
  - "The CUDA / HIP compile-only job is confirmed green once c74f3ba (which adds ojas-cuda/src/bin/rung0.rs and runga.rs) is pushed; if not, the missing-target cause is fixed"
  - "A guard keeps manifests and tracked files in sync: a [[bin]]/[[test]]/[[example]] path that is not tracked fails a cheap pre-push or CI check before the expensive jobs run"
  - "Workflow actions move off the deprecated Node 20 runtime, and the ubuntu-latest move to Ubuntu 26 (2026-10-19) is either pinned or checked green"
  - "sitegen -check passes at HEAD: scripts/sitegen/main.go:51-58 has no `bench-plots` doc category, so the go (ubuntu) job fails with 'docs/bench-plots.md has no category'; the category and the regenerated docs/ and site/ are committed (the fix exists only in a peer's uncommitted working tree today; coordinate rather than overwrite it)"
  - "Linux test failures that surface once fmt passes are triaged to green or to a filed brief: 52 non-merge commits since 85d1f5a (last passing Linux test step, run 37365079870) have never run their tests on Linux, and nine earlier runs failed in that step"
  - "Runs on main are never cancelled: concurrency cancel-in-progress (test.yml:33-35) is limited to pull requests, so every main commit gets a verdict (37315080004 and 37207155260 were cancelled outright)"
  - "ojas-metal and ojas-wgpu stop both defining an example named step_probe from the shared bench/step_probe.rs (output filename collision in target/release/examples, cargo#6313, may become a hard error); one owner or distinct names"
---

# Task brief v1

## Title
Main CI is red on every push: Linux fmt fails on ~15 files, sitegen -check fails on bench-plots.md, and Linux tests have not run since 85d1f5a (2026-10-05)

Task: gp-ci-main-green
Type: bug
Status: ready
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: none
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

### Re-audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- The `test` workflow has not passed in the last 24 runs [A, C: gh run list]. Latest two: 37855710888 (0e3254f) and 37859886140 (edeaf73) fail in linux/fmt, macos/clippy and go/sitegen -check [A, C: gh run view --log-failed].
- macos/clippy failures (`manual_is_multiple_of` at ojas-wgpu/tests/accumulate_grad.rs:260 under clippy 1.99; E0061 at bench/step_probe.rs:484,490 from the new `window` argument) are fixed at HEAD [A]; whether macOS clippy is fully clean is [U].
- Criterion 4 (CUDA / HIP compile-only) is met: gpu-compile succeeded on both runs [A, C].
- HEAD is 13 commits ahead of origin/main; pushing HEAD as is stays red on fmt and sitegen [A].
- Re-opened elsewhere: gp-cuda-backend-provider's 'host-side CUDA tests run in CI' was ticked, but the Linux test step was skipped on every run since c74f3ba [A].

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- The causes are CONFIRMED against runs 37859886140 and 37855710888 [C]. The fmt file list was incomplete: clip_multi.rs and ojas-wgpu's examples/step_probe.rs were missing. Criterion 3's tessl reach was speculative and is reworded [A].
- The sitegen category fix exists only in a peer's working tree [C].

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

- On macOS, every Rust and Go test passes at HEAD on real hardware, so main is red because of fmt and sitegen, not test failures. Linux results are still unknown until fmt passes (criterion 8).

## Acceptance criteria
- [ ] The test.yml run on main passes every job, including the Linux `cargo test` step (not only fmt and clippy), on a pushed commit; the run id is recorded in docs/status.md. Local cargo fmt --all --check is not the gate: the shared checkout carries peer lanes' uncommitted diffs
- [ ] Committed sources at HEAD are rustfmt-clean under the CI toolchain (rustfmt 1.10.0): the 31 hunks at d431949 are ojas-core src/tensor.rs:2201; ojas-cpu tests/shape_first.rs:590; ojas-infer src/decode.rs:10, device.rs:306, gpt.rs:427; ojas-metal src/device.rs:37,3177, tests/clip_multi.rs:38, tests/shape_first.rs:798; ojas-wgpu src/backend.rs:240,3019, src/context.rs:1317,2105,2191,2212,2221, tests/dispatch_cost.rs:66,73,79,150, tests/shape_first.rs:757, examples/step_probe.rs:16; bench/ojas_rows.rs:860 and bench/step_probe.rs:404,521
- [ ] The fmt step formats only ojas packages (-p list) so a future tessl/gusset formatting change cannot fail ojas CI; today every flagged diff is ojas's own code
- [x] The CUDA / HIP compile-only job is confirmed green once c74f3ba (which adds ojas-cuda/src/bin/rung0.rs and runga.rs) is pushed; if not, the missing-target cause is fixed
- [ ] A guard keeps manifests and tracked files in sync: a [[bin]]/[[test]]/[[example]] path that is not tracked fails a cheap pre-push or CI check before the expensive jobs run
- [ ] Workflow actions move off the deprecated Node 20 runtime, and the ubuntu-latest move to Ubuntu 26 (2026-10-19) is either pinned or checked green
- [ ] sitegen -check passes at HEAD: scripts/sitegen/main.go:51-58 has no `bench-plots` doc category, so the go (ubuntu) job fails with 'docs/bench-plots.md has no category'; the category and the regenerated docs/ and site/ are committed (the fix exists only in a peer's uncommitted working tree today; coordinate rather than overwrite it)
- [ ] Linux test failures that surface once fmt passes are triaged to green or to a filed brief: 52 non-merge commits since 85d1f5a (last passing Linux test step, run 37365079870) have never run their tests on Linux, and nine earlier runs failed in that step
- [ ] Runs on main are never cancelled: concurrency cancel-in-progress (test.yml:33-35) is limited to pull requests, so every main commit gets a verdict (37315080004 and 37207155260 were cancelled outright)
- [ ] ojas-metal and ojas-wgpu stop both defining an example named step_probe from the shared bench/step_probe.rs (output filename collision in target/release/examples, cargo#6313, may become a hard error); one owner or distinct names

## Planned files
- .github/workflows/test.yml
- ojas-core/src/autocast.rs
- rustfmt.toml
- scripts/sitegen/main.go
