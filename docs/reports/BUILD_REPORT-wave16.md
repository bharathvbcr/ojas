# BUILD_REPORT wave 16

Task: `ft-09edeb325bef2537a7d72e212befdd16` (revision 8). Criterion 8 only. The card is in review. It is not done. `verification` is null.

Claimed ready revision 6 to in_progress revision 7, sequence 27119, request_id `w16-claim-09edeb32`. Moved in_progress to review revision 8, sequence 27121, request_id `w16-review-09edeb32`. `previous_status` in_progress, `requested_status` review, `held_for_review` false, `summary_recorded` true, `verification` null. Done was not requested.

Worktree `/Users/bharath/Code/research/ojas-w16`, branch `gp/w16-oracle-nan`, HEAD `f327338`. Uncommitted. Primary `/Users/bharath/Code/research/ojas` was not edited. Nothing was committed, pushed, or merged.

This session is Grok. The implementation and the review are this session. The adversarial pass was a `code-reader` subagent, not a separate Opus or Sonnet reviewer.

## What landed

Criterion 8 of "Tests that cannot fail in CI". The oracle normwise gate accepted a non-finite reference: `normwise_rel` checked `got` only, a NaN `want` made the ratio NaN, and `NaN > tol` is false, so `check_tensors` returned `Ok`.

- `ojas-oracle/src/parity.rs`: `normwise_rel` returns `Err` on a non-finite reference. `check_tensors`, the forward-loss comparison, and the logits comparison use `!(rel <= tol)`. Both losses exactly 0 are a match (relative 0). A zero torch loss with a non-zero model loss still fails. A non-finite torch loss fails before the ratio.
- `ojas-oracle/tests/parity_gates.rs`: `normwise_and_tensor_checks_fail_closed` now requires `Err` for a NaN, `+inf`, and `-inf` reference, and `check_tensors` `Err` for a NaN want and an inf want. An equal finite pair still passes.
- `ojas-oracle/tests/golden.rs`: `embedded_f32_fixtures_are_finite` walks `tiny_init` params, `tiny_grads(Init|Step5)` grads plus the f32 `loss` and `loss_seed`, `tiny_trace(F32|Bf16)` params, and `tiny_forward` logits plus `loss`.

`git diff --stat` after reverting `Cargo.lock`: 3 files, 82 insertions, 9 deletions. `parity.rs` is the only file with deletions (9).

Chosen outcome for both losses 0: they match. `0/0` is NaN, and `!(NaN <= tol)` would reject a model that agrees with torch. The special case is only `loss == 0.0 && want_loss == 0.0` (`-0.0` compares equal). The unit test `both_zero_losses_match_and_a_non_finite_torch_loss_does_not` pins `loss_rel == 0` for that case and `Err` for want 0 with loss 1, NaN, infinity, a NaN model loss, and a finite miss of `2e-5` against tolerance `1e-5`.

`f32_set` still copies a non-finite value. The gate now rejects it. The criterion does not ask the loader to change. The adversarial pass did not call that unmet.

## Criteria left open

1. Skip helpers still skip on any open error. Not touched.
2. Go wgpu / `profile.rs` no-adapter PASS vs SKIP. Not touched.
3. wgpu parity tolerance floor of 1.0. Not touched. `ojas-wgpu/tests/parity.rs` is dirty on the softmax worktree, so a tolerance change that failed there could not be triaged in this wave.
4. Gradcheck tolerances. Not touched.
5. Independent tests for the weak done-claims. Not touched.
6. Metal test protocol and `docs/status.md`. Not touched.
7. Wall-clock bounds, bless switch, `stress.rs`. Not touched.
9. S7-03 CUDA comparators. Needs `ojas-cuda/src/small_common.rs`, dirty on `gp/w15-cuda-syntax`. `ojas-cuda/src/check.rs` was free and was left alone so the three comparators stay one change.
10. S3-08 `repeated_dispatches_are_deterministic`. Needs `ojas-wgpu/src/lib.rs`, dirty on `gp/w13-scratch`.
11. S5-13 Go cancel timing. Not touched. `go/model_test.go` was not dirty.
12. Five-repeat byte-identical reduction or matmul. Not touched.

The card leaving `ready` is the board move for this slice. Criteria 1–7 and 9–12 are not done. A person has to send the rest back before another wave can take them.

## Verification

Pre-fix, with the new `parity_gates` asserts and the old `normwise_rel`: `normwise_and_tensor_checks_fail_closed` panicked at `ojas-oracle/tests/parity_gates.rs:354`, `assertion failed: normwise_rel("t", &[1.0], &[f32::NAN]).is_err()`. EXIT 101. Log `target/w16-prefix.log`. Start `2026-10-26T22:18:38Z`, load `{ 133.74 102.27 83.40 }`, avail_gb 86. Compile of the test profile 5.86s. That is the failing test the fix is for.

Post-fix: `cargo test -p ojas-oracle -- --test-threads=1 --nocapture`. `CARGO_BUILD_JOBS=1`, `CARGO_TARGET_DIR` the worktree `target/`, `CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_INCREMENTAL=0`, `CARGO_NET_OFFLINE=true`. EXIT 0. 68 passed, 0 failed, 2 ignored. `ojas-oracle` recompiled in 3.45s. Log `target/w16-test.log`. Start `2026-10-26T22:19:18Z`, load `{ 129.18 104.81 85.18 }`, avail_gb 86.

The two ignored tests are pre-existing and were not changed: `a_long_sequence_stays_within_bound_of_the_f64_forward` (ignored, long, `--release`) and `generated_124m_init_follows_section_2` (needs a generated 124m safetensors file).

New tests in that log, all `ok`: `parity::tests::both_zero_losses_match_and_a_non_finite_torch_loss_does_not`, `embedded_f32_fixtures_are_finite`, `normwise_and_tensor_checks_fail_closed`.

`rustfmt --edition 2021 --check` on the three worktree paths printed nothing and exited 0.

`cargo` rewrote `Cargo.lock` (`gusset` 0.0.2 to 0.0.3, the path crate on disk). Reverted with `git checkout -- Cargo.lock` in the worktree. `git status` after that is the three source files only. Do not commit the lock.

Not run: the workspace suite, `dev verify`, benchmarks. `.devcouncil/config.yaml` does not exist, so `dev verify` cannot run. Adding it is still an owner decision. No hot path was changed, so there is no benchmark delta to report. Profile was unoptimized (`debuginfo` 0).

DevMap generation 7128, `is_fresh` true, store `/Users/bharath/Code/research/ojas/.devcouncil/codeintel/devmap.sqlite`, `repository.root` that path. `degraded_reason` still partial (22 `partial_parse` files, `pattern_recovered` truncated at 50/52). A file-level impact on `ojas-oracle/src/parity.rs` was truncated (`hidden` 180). Function-level `devmap_affected_tests` on `ojas-oracle/src/parity.rs::normwise_rel` at depth 2: `truncated` false, `walk_incomplete` (stopped at depth 2), 6 tests shown, `hidden` 0. Shown: `ojas-oracle/tests/parity_gates.rs`, `golden.rs`, `ojas_cpu_parity.rs`, `variants.rs`, `ojas-model/tests/oracle_parity.rs`, `ojas-wgpu/tests/muon.rs`. The ojas-oracle package run covers the first four. `oracle_parity` and `muon` were not run. For finite ratios, `rel > tol` and `!(rel <= tol)` agree, including equality. That equivalence is from reading the comparison, not from running those two suites.

Adversarial `code-reader` `01a17a4c-cd0b-7130-bf1d-f55f0d43862a`, 43 tool calls, exit 0, reply `no return`.

## After the wave

`gitpulse_insights` at `2026-10-26T22:25:15Z` completed: `deadline_expired` false, 10 worktrees scanned, 0 unscanned, `truncated` false, `sessions_ok` true. Claude pid 44032 is still busy on the primary checkout. Codex sessions 0.

Primary `main` moved during the wave from `f327338` to `4710bb2` (`Merge gp/78-metal: Metal conv2d and conv_transpose2d for the SAM backbone`, parents include `b3ce6f4` and `76a1f07`). `git diff --stat f327338..4710bb2` on `ojas-oracle/src/parity.rs`, `ojas-oracle/src/golden.rs`, `ojas-oracle/tests/parity_gates.rs`, and `ojas-oracle/tests/golden.rs` was empty. The wave 16 diff is against files that are the same on current main. The worktree was not rebased. It is 3 commits behind `4710bb2`.

Primary dirt is still `Cargo.lock` plus the three untracked `tasks/gp-*.md` files. This session did not edit them.

Overlaps now: `Cargo.lock` (primary, 78a, 78b, 78c, w13), `ojas-core/src/autocast.rs` and `ojas-core/src/lib.rs` (78b and the softmax worktree), `ojas-metal/kernels/ojas_backend.metal` (78a and the softmax worktree). None of those are the wave 16 files. `ojas-78a` is dirty in 9 files; at the preflight shell scan it was only `Cargo.lock`. A new clean detached worktree `/Users/bharath/Code/research/ojas-st` is at `4710bb2`. Ledger `error_code` is still `legacy_ledger_unconsolidated`.

DevMap generation is now 7129 (`is_fresh` true). The blast radius above was read at generation 7128, before this merge. It was not re-queried. The oracle files did not change in the merge, so the function-level test list is still the one this diff was checked against.

## Preflight

`gitpulse_insights` at the start of the wave deadline-expired (`scanned_worktrees` 0, `unscanned_worktrees` 8). That empty overlap was not treated as clean.

`git worktree list` showed 8 worktrees. `git status --porcelain` on each, before the edit:

- primary `main` `f327338`: `Cargo.lock`, untracked `tasks/gp-bidirectional-attention.md`, `tasks/gp-spatial-ops.md`, `tasks/gp-vision-norm-act-bias.md`. Claude pid 44032 was on this tree at the deadline-expired insights. Not edited.
- `ojas-78a` `gp/78-metal` `b3ce6f4`: `Cargo.lock` only.
- `ojas-78b` `gp/78-cpu-oracle` `afdaf98`: `Cargo.lock`, `docs/op-coverage.md`, capi/core/cpu/model reach files, `site/reference/op-coverage.html`, untracked `ojas-cpu/tests/relu.rs`.
- `ojas-78c` `gp/78-bench` `3eba4d2`: `Cargo.lock` only.
- `ojas-w14` `gp/w14-unverified` `d2922fb`: untracked `CONTRIBUTING.md`, `docs/reports/BUILD_REPORT-wave14.md`. Not reused.
- `ojas-w15` `gp/w15-cuda-syntax` `f327338`: `ojas-cuda/src/kernels.rs`, `ojas-cuda/src/small_common.rs`, untracked `docs/reports/BUILD_REPORT-wave15.md`. Not reused.
- `w13-scratch` `gp/w13-scratch` `85ffe0d`: `Cargo.lock`, four `Cargo.toml` files, `ojas-wgpu/src/context.rs`, `ojas-wgpu/src/lib.rs`, two wgpu tests, untracked wave 13 report.
- softmax worktree `42d5a9f`: core/cpu/cuda/metal/wgpu files listed in the wave 15 notes, plus `ojas-cpu/tests/softmax_collapse.rs` untracked. `ojas-wgpu/src/lib.rs` is w13, not this tree. `ojas-wgpu/tests/parity.rs` is this tree.

`gitpulse_collision_risk` with `path` set, after the shell scan, `ok` true, `truncated` false, `scanned_worktrees` 8, `unscanned_worktrees` 0, `items` empty, for `ojas-oracle/src/parity.rs`, `ojas-oracle/tests/parity_gates.rs`, and `ojas-oracle/tests/golden.rs`.

Load stayed above the bar (`load1` about 100 to 134). Compiles used `CARGO_BUILD_JOBS=1` and an 8 GB disk floor. The floor did not fire (avail_gb 86). `mac_heavy.sh` was not used.

## Fable

One card. Criterion 8 is a host-side bug that runs on this Mac. S7-03 and S3-08 are the audit-order neighbours and both sit on files another worktree holds, so they were not started. Both-zero losses match; the criterion asked for the outcome to be pinned, and `!(rel <= tol)` is the comparison on the ratio after that special case. The rest of the card stays open in review rather than in progress, so the next wave does not edit this card while it is claimed.

## Deferred

Still not started, for the same hold reasons as wave 15 unless a path above changed: `ft-2c183b71`, `ft-301ddd4c` (ready, not fetched), `ft-6ba746b4`, `ft-353fa922`, `ft-da2c4ab1`, `ft-9c13f472`, `ft-562b0add`, `ft-2dc2dc69`, `ft-29f7e4b6`, `ft-e15ee76a`, `ft-ee2a3753` (criterion 1 already on main; 3 and 6 blocked), `ft-a95d2145` (config is an owner decision), `ft-ebbed90b`, `ft-e6d8035e`, `ft-693bbfcb`, `ft-30dd709e`, `ft-f40eeed8`, `ft-6326debf`. In-progress cards were not touched: `ft-71629756`, `ft-a1a4ba4d`, `ft-2bbced18`, `ft-b5e64557`, `ft-53d639de`, `ft-96baeabc`, `ft-92c5e9c6`. Review cards not reopened: `ft-81e49280` (wave 13), `ft-68d0f581` (wave 14), `ft-9bed58f0` (wave 15).

## Owner decisions still pending

Not re-decided: E9 (`--locked` / SHA pins), E11b/OD10 push, E12 (`ft-69ef4ce1` back to ready), OD2 (C11 digest), OD3, OD6, OD7, OD12, and committing `.devcouncil/config.yaml` so `dev verify` can run. E6 stays decided: `check_room` read-only, charge child-first.

## New tasks

None.
