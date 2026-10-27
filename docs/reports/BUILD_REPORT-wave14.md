# Wave 14

Task: `ft-68d0f581ab8ca4788ee931dd276845e3`. Claimed at revision 2 (`in_progress`). Moved to review at revision 3. `verification` is null. Branch `gp/w14-unverified` at `d2922fb`, worktree `/Users/bharath/Code/research/ojas-w14`. Not merged. Not committed.

## Review

| Task | Result |
| --- | --- |
| `ft-68d0f581ab8ca4788ee931dd276845e3` | review. Criteria 1–3 met by the run below. Criterion 4 is written on this branch only (`CONTRIBUTING.md`), not on main. `verification` was null at claim. Not moved to done. |

No task moved to done.

## What ran

Worktree of `d2922fb`. `CARGO_BUILD_JOBS=1`, `CARGO_INCREMENTAL=0`, debuginfo 0, private `target/`, `CARGO_NET_OFFLINE=true`. Profile is `test` unoptimized. Log: `target/w14-test.log`.

`cargo test -p ojas-model --offline --test preflight -- --test-threads=1 --nocapture`

- Compiled tessl, ojas-metal, ojas-wgpu, ojas-model. `Finished test profile [unoptimized] in 1m 49s`.
- `25 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.11s`. Exit 0.
- `calibration_reproduces_each_row_within_one_percent` passed against `bench/results/2026-10-17-lora-step/rows.jsonl`. The test asserts a 1% fit. It does not print the measured error.
- The panic in `a_panicking_dry_run_releases_the_child_charge` is the panic that test catches (`preflight.rs:367`).

`cargo test -p ojas-model --offline --test adamw_torch_100 -- --test-threads=1 --nocapture`

- `Finished test profile [unoptimized] in 4.40s` (ojas-model rebuilt).
- `7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.65s`. Exit 0.
- Bound from the committed fixture: `2.3e-5`.
- CPU Exact worst logged peak-relative error: `p100.d17` `6.035e-6`. `p1.norm_nodecay` is `0` (the zero-gradient, no-decay identity).
- CPU Fast worst: `p100.lora_A` `1.971e-7`.
- Metal worst: `p100.lora_A` `1.971e-7`.
- `a_wrong_weight_decay_or_step_count_is_outside_the_bound` passed, so a wrong decay and a wrong step count each miss the bound.
- `a_missing_fixture_fails_loudly` is `should_panic` and passed.

No product code failed, so nothing was fixed and nothing was reverted. History was not rewritten.

Current primary `main` is `f327338` (three commits after `d2922fb`: `032e015`, `3eba4d2`, merge `f327338`). `git diff --stat d2922fb..f327338` on the AdamW test, the preflight sources, the fixture, and the rows file is empty. The run covers the copies of those files on current main.

## Criterion 4

`CONTRIBUTING.md` on this branch says a commit marked `UNVERIFIED`, or committed without compiling or running, must name the GitPulse task id (`ft-…` or `gp-…`) in the message body, and that history is not rewritten to remove it.

That file is untracked here. It is not on `main`. Primary is held (Claude was busy on it at the start of the wave) and this session does not commit. An adversarial pass returned on this criterion only. Landing the rule is a commit onto main, which this wave does not do.

## Benchmarks

No measured path changed. No benchmark binary was run. The 0.65s AdamW figure is the unoptimized test process, not a throughput delta.

## Deferred

Not started. Planned files were held, or the card is in progress, or it needs a machine this wave did not have.

- `ft-9c13f472` training semantics. Planned files overlap the 78b checkout and the softmax worktree. Criterion 10 waits on the vision chain.
- `ft-2c183b71` GPU step throughput. Needs `ojas-wgpu/src/context.rs` (dirty on `gp/w13-scratch`) and the Metal kernel (dirty in the softmax worktree).
- `ft-f40eeed8` softmax collapse. Its own worktree holds the files.
- `ft-562b0add`, `ft-6326debf`, `ft-e15ee76a`, `ft-29f7e4b6`. CPU exp / Metal kernel files are held.
- `ft-30dd709e`, `ft-da2c4ab1`, and the in-progress trainer and LoRA cards (`ft-b5e64557`, `ft-a1a4ba4d`, `ft-2bbced18`, `ft-71629756`, `ft-53d639de`, `ft-96baeabc`, `ft-92c5e9c6`). Not touched.
- `ft-ee2a3753` safetensors. The open slice touches `ojas-model/src/trainer.rs`, which the in-progress trainer card owns. The tessl page-in criterion waits on tessl `gp/wave10`.
- `ft-9bed58f0` CUDA host bugs. Several criteria need a GH200 run. The C ABI change is a public contract.

`ft-81e49280` (wave 13 scratch) stays in review. This wave did not edit `context.rs`.

## New tasks

None.

## Decisions

1. One card this wave: `ft-68d0f581`.
2. Worktree is a sibling of the repo (`/Users/bharath/Code/research/ojas-w14`) so `../../tessl` and `../../../devtools/gusset` resolve. Held `Cargo.toml` files were not edited.
3. No history rewrite. No merge. No rebase onto `f327338`.
4. The load gate (load1 < 12 and load5 < 18, plus 1 GiB free) expired after 25 minutes at load1 24.31, load5 48.51, 121281 free pages. The compile then ran at one job. A guard was set to stop only this cargo pid if free space fell under 8 GB. It finished with 13 GB free. The guard did not fire.
5. Cargo rewrote `Cargo.lock` gusset `0.0.2` to `0.0.3` from the live path crate. That line was reverted after the tests. It is not part of the result.
6. Criterion 4 stays off main until a commit is requested. The rule file is the branch artifact.

## Still pending from earlier waves

E9 (`--locked` / SHA pins), E11b/OD10 push, E12 (`ft-69ef4ce1` back to ready), OD2 (C11 digest), OD3, OD6, OD7, OD12, and committing `.devcouncil/config.yaml` so `dev verify` can run. E6 was already decided: `check_room` stays read-only.

## Not verified

- `dev verify`. This checkout has no `.devcouncil/config.yaml`.
- Full workspace suite. `ci_local.sh` is `cargo test --workspace --release`. This wave ran two test targets, unoptimized.
- Manvi `work.verify.task`. Do not treat a review move as done.
- Release-profile numbers. Debuginfo was off; opt-level was the test profile default (unoptimized).
- The wrong-recipe AdamW errors are not printed. The test only asserts that each wrong recipe has some checkpoint outside the bound, and that assertion passed.

## Roles

Implementation and the review note are this Grok session. The adversarial pass was code-reader `01a179fa-c111-75e2-914a-f2ee0999c29c`. It returned on criterion 4 and accepted criteria 1–3. This was not a Claude Opus or Sonnet review.
