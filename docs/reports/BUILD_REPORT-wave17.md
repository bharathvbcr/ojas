# BUILD_REPORT wave 17

Task: `ft-29f7e4b675027cae97b0d647eb34031f` (revision 4). Criterion 2 only. The card is in review. It is not done. `verification` is null.

Claimed ready revision 2 to in_progress revision 3, sequence 27151, request_id `w17-claim-29f7e4b6`. Moved in_progress to review revision 4, sequence 27155, request_id `w17-review-29f7e4b6`. `previous_status` in_progress, `requested_status` review, `held_for_review` false, `summary_recorded` true, `verification` null. Done was not requested.

Worktree `/Users/bharath/Code/research/ojas-w17`, branch `gp/w17-tape-shape`, HEAD `105e20343c6c8a30dfd01356aa689944f76a8dc1`. Uncommitted. Primary `/Users/bharath/Code/research/ojas` was not edited. Nothing was committed, pushed, or merged.

This session is Grok. The implementation and the review are this session. The adversarial pass was a `code-reader` subagent.

## What landed

Criterion 2 (G-TAPE / CA-4): Tape backward keeps a backend gradient whose shape is not the forward variable's.

`Tape::acc` in `ojas-autograd/src/tape.rs` stored the first gradient with no comparison to `self.values[id].shape()`. `Backend::accumulate_grad` only requires the two gradients to share a shape, so one wrong shape was kept, and two identical wrong shapes were summed and kept. `fused_ce_asking` already rejects a fused cross-entropy gradient of the wrong shape. That check does not run for `silu`, `linear`, or the other `backward_one` arms. Those arms store through `acc`.

`acc` now returns `OjasError::Backend` when the gradient shape differs from the forward value, and it returns before either store. The detail is `Tape::backward: gradient {got:?} does not match forward var {id} {want:?}`. `backward_seeded` already clears every gradient slot when the walk returns an error, so a gradient stored earlier in that walk is wiped. The new test `a_wrong_weight_gradient_wipes_the_input_gradient_too` pins that.

A frozen node still returns before the shape check and stores nothing. A replay assignment copies a gradient that `acc` already checked against the checkpoint output, or the seed filled from that same shape, and `truncate` drops the replayed nodes before `backward` returns. The adversarial pass did not call either path a keep.

`git diff --stat` after reverting `Cargo.lock`: `ojas-autograd/src/tape.rs`, 16 insertions. New untracked test `ojas-autograd/tests/tape_grad_shape.rs`, 162 lines.

## Criteria left open

1. Exact-tier libm. Not touched. `ojas-cpu/src/fused_ce.rs`, `pointwise.rs`, `ojas-core/src/exp_exact.rs`, and `ojas-core/src/lib.rs` are dirty on the softmax worktree. The card's note that criteria 1–5 of `ft-562b0add` are already implemented was not re-verified by running tests. A primary read during preflight still showed `hybrid.rs` softplus on platform libm. That file was free and was left alone because the spec names the held files with it.
3. Generation counter on `Tape::clear`, panic-safe checkpoint replay, and Autocast from the recorded forward node. Not touched. `ojas-core/src/autocast.rs` is dirty on the softmax worktree. `Tape::clear` still has no generation counter.
4. Go C-ABI micro-batch ceiling. Not touched.
5. Miri and TSan CI, and the MSRV comment in `ojas-simd/src/arch.rs`. Not touched.

The card leaving `ready` is the board move for this slice. Criteria 1, 3, 4, and 5 are not done. A person has to send the rest back before another wave can take them.

`tape.rs` is the one hot file for this wave. `ojas-core/src/backend.rs`, `ojas-metal/src/device.rs`, and `ojas-metal/src/backend.rs` were not edited.

## Verification

Pre-fix, new tests, unmodified `acc`. `cargo test -p ojas-autograd --test tape_grad_shape -- --test-threads=1 --nocapture`. Env: `CARGO_BUILD_JOBS=1`, `CARGO_TARGET_DIR` the worktree `target/`, `CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_INCREMENTAL=0`, `CARGO_NET_OFFLINE=true`. Compile of the test profile 32.57s. Log `target/w17-prefix.log`. Result in that log: `FAILED. 1 passed; 3 failed`. The three `expect_err` tests panicked because `backward` returned `Ok(())`:

- `a_flat_silu_gradient_is_refused_and_the_leaf_keeps_nothing`
- `a_wrong_weight_gradient_wipes_the_input_gradient_too`
- `two_silu_paths_returning_the_same_wrong_shape_are_refused`

`a_real_silu_gradient_is_kept_with_the_forward_shape` passed. The wrapper's exit code was 1 because zsh rejected `status` as a read-only name after cargo had written the log. The log is the cargo result. There is no `EXIT:` line in that file.

Post-fix, same four tests, log `target/w17-shape.log`: `ok. 4 passed`, `EXIT:0`. That run was a 0.04s rebuild of an already-fixed library. The string `does not match forward var` is in `target/debug/deps/libojas_autograd-81c1b40a528483fe.rlib`.

Post-fix package: `cargo test -p ojas-autograd -- --test-threads=1`, same env. `ojas-autograd` recompiled in 6.75s. `EXIT:0`. Log `target/w17-test.log`. 125 passed, 0 failed, 1 ignored, plus doc-tests `0 passed`. The ignored test is the pre-existing `tape_bench` bench. `wgpu_tape` ran 4 tests and passed in 0.26s. `tape_grad_shape` passed again inside this run (4).

`rustfmt --edition 2021 --check` on the two worktree paths exited 0 after formatting the test file.

`cargo` rewrote `Cargo.lock` (`gusset` 0.0.2 to 0.0.3, the path crate on disk). Reverted with `git checkout -- Cargo.lock` in the worktree. After that, `git status` is `M ojas-autograd/src/tape.rs` and `?? ojas-autograd/tests/tape_grad_shape.rs`, plus this report. Do not commit the lock.

Not run: the workspace suite, `dev verify`, benchmarks. `.devcouncil/config.yaml` is absent in this worktree, so `dev verify` cannot run. Adding it is still an owner decision. The change is one shape compare before an existing store. No kernel or schedule changed, so there is no benchmark delta. Profile was unoptimized (`debuginfo` 0).

DevMap at claim time, generation 7130, `is_fresh` true, `source_freshness` true, store `/Users/bharath/Code/research/ojas/.devcouncil/codeintel/devmap.sqlite`. `devmap_impact` on `ojas-autograd/src/tape.rs::acc`, depth 2: `truncated` false, `walk_incomplete` (stopped at depth 2), 2 edges shown. The direct caller is `Tape::backward_one`. That walk is a lower bound. `devmap_affected_tests` on the same symbol, depth 2: `truncated` true, shown 9 of 33 test files, `walk_incomplete`, `fail_closed` true. The `ojas-autograd` package run covers the test binaries in that package. Tests outside the package that the truncated list withheld were not run. A correct backend gradient already has the forward shape, so those suites should be unchanged. That is inferred from the check, not from running them.

Adversarial `code-reader` `01a17a97-08ea-7652-a53f-c163a5c9dc9b`, 39 tool calls, exit 0, reply `no return`. It named one residual it did not treat as a miss: `unshare_leaf_grads` copies an already accepted leaf gradient through `residual_add_backward` and does not check the copy's shape. Stock CPU and Metal copies keep the source shape after a dims check. No new task was filed.

## Sequencing

This session's call, one card, one Mac-runnable slice:

- Take criterion 2 only. The walk still kept a mismatched gradient. The fused-CE contract check is a different site and is already tested.
- Leave criteria 1, 3, 4, and 5 on the card. Criterion 1 shares held softmax files. Criterion 3's autocast half is on a held file. Criteria 4 and 5 are separate concerns, and this wave already spent `tape.rs`.
- Do not take `ft-562b0add`. Its remaining criteria need `fused_ce.rs` and `pointwise.rs`, which the softmax worktree holds.
- Do not reopen waves 13–16.

## After the wave

`gitpulse_insights` completed: `deadline_expired` false, `duration_ms` 4506, 11 worktrees scanned, 0 unscanned, `truncated` false, `sessions_ok` true. Claude pid 44032 is still busy on the primary checkout. Codex sessions 0. This scan was before this report file existed, so `ojas-w17` showed 2 dirty files (`tape.rs` and the test). The report is a third untracked file.

Primary `main` is still `105e203`. Porcelain is still `M Cargo.lock` and the three untracked `tasks/gp-*.md` files. This session did not edit them.

`ojas-w17` is `gp/w17-tape-shape` at `105e203`, behind 0. Overlaps: `Cargo.lock` (primary, 78a, 78b, 78c, w13) and `ojas-metal/kernels/ojas_backend.metal` (78a and the softmax worktree). Neither is a wave 17 file. `ojas-78a` dirty count is 10. Ledger `error_code` is still `legacy_ledger_unconsolidated`.

DevMap generation is still 7130, `is_fresh` true. The index is the primary tree. It does not contain this worktree's edit. Callers of `acc` were not added.

## Preflight

`tape.rs` collision: `ok` true, `truncated` false, scanned 10, unscanned 0, items empty. DevMap generation moved from the previous wave's 7129 (`is_fresh` false) to 7130 (`is_fresh` true) before the claim. Load at the worktree add was `{ 58.78 54.68 39.90 }`. Disk had 72 GB free. The build used one cargo job. `mac_heavy.sh` was not used.

Ready at claim time included this card at revision 2. In-progress cards were left alone: `ft-71629756`, `ft-a1a4ba4d`, `ft-2bbced18`, `ft-b5e64557`, `ft-53d639de`, `ft-96baeabc`, `ft-92c5e9c6`.

## Owner decisions still pending

Not re-decided: E9 (`--locked` / SHA pins), E11b / OD10 push, E12 (`ft-69ef4ce1` back to ready), OD2 (C11 digest), OD3, OD6, OD7, OD12, and committing `.devcouncil/config.yaml` so `dev verify` can run. E6 stays as already decided: `check_room` read-only, charge child-first.

## New tasks

None.
