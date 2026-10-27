# BUILD_REPORT wave 18

Task: `ft-30dd709ee6b761de3bb096e6f02fc561` (revision 3). All three criteria. The card is in review. It is not done. `verification` is null.

Claimed ready revision 1 to in_progress revision 2, sequence 27188, request_id `w18-claim-30dd709e`. Moved in_progress to review revision 3, sequence 27199, request_id `w18-review-30dd709e`. `previous_status` in_progress, `requested_status` review, `held_for_review` false, `summary_recorded` true, `verification` null. Done was not requested.

Worktree `/Users/bharath/Code/research/ojas-w18`, branch `gp/w18-lora-draw`, HEAD `105e20343c6c8a30dfd01356aa689944f76a8dc1`. Uncommitted. Primary `/Users/bharath/Code/research/ojas` was not edited. Lappi was not edited. Nothing was committed, pushed, or merged.

This session is Grok. The implementation and the review are this session. The adversarial pass was a `code-reader` subagent.

## What landed

`lora_adapter_layout` returned the provider's A-leaf order. Lappi, after amendment 4 (`200a2d2`), sorts that list with `canonical_draw_order`: layer index ascending, then `LORA_TARGET_SUFFIXES`. A module with no `layers.<digits>` place is refused. Two modules at one `(layer, suffix)` are refused.

`draw_position` and `canonical_draw_order` now live in `ojas-train/src/lappi/recipe.rs`, next to the suffix table. The layout calls `canonical_draw_order` on the A-module names and returns `(module, in_features)` in that order. `in_features` is still A's second axis. Errors are `StepError::Contract` with Lappi's `LoraInitError` Display text (`refused: ` is added once by `StepError`). `LoraHostNames` is unchanged. `lora_param_groups` still lists the `"lora"` group in provider entry order (`recipe.rs` builds that vec from `entries`, after `lora_adapter_layout` is used only as `?`). The RNG, digest, and `recompute_lora_a` were not copied.

The draw place is the `layers` component directly before the suffix. `layer_index` still answers the first `layers.<n>` in a name, for the learning-rate split, and is not this order. `model.layers.2.layers.0.self_attn.q_proj` is layer 0.

`git diff --stat` after reverting `Cargo.lock`: `ojas-train/src/lappi/recipe.rs`, 70 insertions, 6 deletions. New untracked test `ojas-train/tests/lora_adapter_layout.rs`, 268 lines.

## Criteria

1. Met. The layout is the canonical order, and the same malformed draw-order inputs Lappi refuses are refused: no layer index (bare suffix, empty index, non-digit, `xlayers`, `model.layer`, a `u64` overflow) and two modules at one position (leading-zero collision, two prefixes, the nested name against `layers.0`). A lone `layers.02` sorts as layer 2, after layer 1. `u64::MAX` is a place. `language_model.layers` is a place. A repeated A leaf is still `listed twice` before the sort. One bad name refuses the whole layout.

2. Met. `the_adapter_layout_is_canonical_whatever_order_the_provider_lists_entries_in` is the Lappi `lora_plan.rs` contract: reverse the provider list, the layout is identical; layer index is non-decreasing; a flat `self_attn.q_proj` leaf's error contains `layer index`. The same fixtures registered in alphabetical order (k before q) and in a different ojas order both equal one handwritten canonical list, so q is before k and `in_proj_qkv` is before `in_proj_z`. `the_lora_group_stays_in_provider_order` pins that the group list did not pick up the sort.

3. Met as far as this repo can take it. There is no ojas commit that contains the fix, because this session was not asked to commit. Do not pin `105e203`: that commit is the worktree base and does not contain the draw order. The commit to pin is the one that lands this diff, after a commit is requested. The Lappi pin bump was not made. Lappi is told that here and in the review summary on this card. `tools/a6/HANDOFF-for-lappi.md` was not edited. `/Users/bharath/Code/research/Lappi-decision` was not edited.

## Verification

Pre-fix, new tests, unmodified `lora_adapter_layout`. `cargo test -p ojas-train --test lora_adapter_layout -- --test-threads=1 --nocapture`. Env: `CARGO_BUILD_JOBS=1`, `CARGO_TARGET_DIR` the worktree `target/`, `CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_INCREMENTAL=0`. The first attempt with `CARGO_NET_OFFLINE=true` failed in 0.25s: `no matching package named wgpu` against the crates.io index. A `--locked` retry failed because cargo wanted to move path-dep `gusset` from 0.0.2 to 0.0.3. The run that produced the failure was online and unlocked. Log `target/w18-prefix.log`. `EXIT:101`. `FAILED. 2 passed; 2 failed`:

- `the_adapter_layout_is_canonical_whatever_order_the_provider_lists_entries_in` left the HF listing order, right the canonical list
- `a_shared_draw_position_is_refused_and_the_place_is_the_layers_before_the_suffix` got `Ok((8, [("model.layers.1.self_attn.q_proj", 4), ("model.layers.01.self_attn.q_proj", 4)]))`

`a_repeated_leaf_is_still_listed_twice_before_the_draw_order` and `the_lora_group_stays_in_provider_order` passed on the old code.

Post-fix, same four tests, log `target/w18-layout.log`: `ok. 4 passed`, `LAYOUT_EXIT:0`. Then the package, log `target/w18-test.log`, `PKG_EXIT:0`:

| binary | result |
|---|---|
| `src/lib.rs` | 26 passed |
| `adamw_decay_sensitive` | 2 passed |
| `adamw_oracle` | 2 passed |
| `frozen_entries` | 5 passed |
| `groups_by_name` | 3 passed |
| `lora_adapter_layout` | 4 passed (this binary, before two more tests) |
| `pyjson_oracle` | 5 passed |
| `schedule_adversarial` | 26 passed, 2 ignored |
| `schedule_oracle` | 4 passed |
| `schedules` | 15 passed |
| doc-tests | 0 |

92 passed, 0 failed, 2 ignored. The ignored tests are `desired_rate_is_finite_for_a_huge_finite_peak` and `desired_rate_never_exceeds_peak_for_decimal_peaks`, both already marked out of scope on `ft-b5e64557`.

Two tests were added after that package run, without another edit to `recipe.rs`: `a_leading_zero_sorts_as_its_integer_and_one_bad_name_refuses_the_rest` and `no_target_suffix_is_a_suffix_of_another`. Log `target/w18-layout2.log`: `ok. 6 passed`, `LAYOUT2_EXIT:0`. The other binaries were not re-run. They had just passed against the same `recipe.rs`.

`rustfmt --edition 2024 --check` on `ojas-train/src/lappi/recipe.rs` and `ojas-train/tests/lora_adapter_layout.rs` exited 0 (`FMT:0` before the package run and again before `w18-layout2`).

`cargo` rewrote `Cargo.lock` (`gusset` 0.0.2 to 0.0.3) on each unlocked run. Reverted with `git checkout -- Cargo.lock` in the worktree. After the last revert, `git diff --stat` is only `recipe.rs`. Do not commit the lock.

Not run: the workspace suite, `dev verify`, benchmarks, `ojas-model/tests/image_mlp_provider.rs`. `.devcouncil/config.yaml` is absent, so `dev verify` cannot run. Adding it is still an owner decision. This is a one-time sort of the adapter list, not a step hot path. No benchmark was added, and no speedup is claimed. Profile was unoptimized (`debuginfo` 0). Load at the worktree add was `{ 35.08 63.81 47.06 }`. The disk check stayed above the 8 GB floor. One cargo job. `mac_heavy.sh` was not used.

DevMap at claim, generation 7130, `is_fresh` true, `source_freshness` true, store `/Users/bharath/Code/research/ojas/.devcouncil/codeintel/devmap.sqlite`. `devmap_impact` on `ojas-train/src/lappi/recipe.rs::lora_adapter_layout`, depth 2: `truncated` false, 4 edges, all inside `recipe.rs`. The direct caller is `lora_param_groups`. The walk is still `walk_incomplete` because call extraction does not cover the whole corpus (html/css pattern recovery, two files with ERROR/MISSING regions). That is a lower bound, not a proof of no other caller. A repo grep of `*.rs` for `lora_adapter_layout` found only `recipe.rs` (the definition, the `?` in `lora_param_groups`, and this wave's test). `devmap_affected_tests` depth 2: `truncated` false, 7 files shown, `walk_incomplete` (stopped at depth 2), `fail_closed` true. The `ojas-train` package run covers the train files in that list. `ojas-model/tests/image_mlp_provider.rs` was listed and was not run. A grep of that file shows imports of `OptimizerRecipe`, `run_control`, `schedule`, `step`, and `trainer`, and no call to `lora_adapter_layout` or `lora_param_groups`. Unaffected is inferred from that grep, not from running the file.

Adversarial `code-reader` `01a17acc-0652-7743-8dce-3e01271b8ce7`, 20 tool calls, exit 0, about 569s, reply `no return`. It compared the worktree functions to Lappi `lora_init.rs` and `recipe.rs` on the cases in the prompt (boundary, leading zeros, overflow, nearest `layers` component, suffix-table order, ambiguous pairs, bare suffix, duplicate leaf, A/B pairing, gate suffixes, `in_features`, group order). No new task was filed.

## Sequencing

This session's call, one card, all three criteria. The card is the only ready P0 bug whose criteria can all be proved on this Mac, and `recipe.rs` was free.

- Not taken: `ft-9c13f472` (twelve criteria, held `tape.rs`, fused CE, Metal, and the in-progress trainer). Not taken: GPU step cards whose Metal or wgpu files are dirty on `ojas-78a`, the softmax worktree, or `w13`. No second card.
- No measured optimization. The request to optimize does not make a sort of the adapter list into a kernel change, and the perf cards' files are held.
- Do not reopen waves 13–17. Do not commit them.

## After the wave

`gitpulse_insights` `deadline_expired` true, `duration_ms` 11852. Collisions: `ok` false, `truncated` true, scanned 1, unscanned 8, `failed_worktrees` 3. That scan is not a clean repo. A follow-up `gitpulse_collision_risk` on the two edited paths, after the edits: `ok` true, `truncated` false, scanned 12, unscanned 0. Each path is dirty only in `ojas-w18`. `overlapping_files` 0.

12 worktrees. Primary `main` is still `105e203`, 4 dirty files (this session did not edit them). Claude pid 44032 is still busy on the primary checkout. Codex sessions 0. `ojas-w18` is on the list at `105e203`, behind 0. Its dirty count in that insights scan was unknown (`dirty_files` null). Ledger `error_code` is still `legacy_ledger_unconsolidated`. The ledger facet was `stale` true on this scan.

DevMap generation is still 7130, `is_fresh` true. The index is the primary tree. It does not contain this worktree's edit.

## Preflight

`recipe.rs` collision before the claim: `ok` true, `truncated` false, scanned 11, unscanned 0, items empty. The same insights call had `deadline_expired` true, so the repo-wide overlap list was not treated as clean. DevMap generation 7130, `is_fresh` true. In-progress cards were not touched. `ft-b5e64557` plans `ojas-train/` and was left on `trainer.rs`; this wave edited `recipe.rs` and a new test only.

## Still pending, not decided here

E9 (`--locked` / SHA pins), E11b / OD10 push, E12 (`ft-69ef4ce1` back to ready), OD2 (C11 digest), OD3, OD6, OD7, OD12, and committing `.devcouncil/config.yaml` so `dev verify` can run. The Lappi pin bump past `4123dfb` still needs the owner's yes in Lappi, and it needs an ojas commit of this diff first.
