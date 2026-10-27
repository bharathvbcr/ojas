# Wave 19

Task `ft-2dc2dc696fec8a4fbc0503e4057b9762` is in review at revision 3, sequence 27299, request id `w19-review-2dc2dc69`. Previous status was in_progress. `held_for_review` is false. `verification` is null. Done was not requested. Manvi has not verified it.

Worktree `/Users/bharath/Code/research/ojas-w19`, branch `gp/w19-audit-baseline`, HEAD `105e20343c6c8a30dfd01356aa689944f76a8dc1`. Nothing was committed. Primary was not edited.

## What this wave measured

This card is the audit-wave-1 baseline. It records the rows the later fix cards have to cite. It does not change the measured kernels, and it does not implement those fixes.

`git diff 105e203 d5d33cc -- ojas-cpu ojas-capi ojas-cuda ojas-core/src/tensor.rs` was empty. Main's 11 commits after `105e203` are the Metal rope / D=64 / conv / relu / bilinear / SAM merge (`d5d33cc`). The CPU, capi, and CUDA sources that were timed match that main. The worktree was not rebased onto it.

49 `OJAS_OP` rows are in `bench/results/2026-10-16-audit-baseline/rows.txt`. The citation table and the minima are in `README.md`. The pin, the dirty hash, and the per-group load are in `env.md`.

Every group started only after two samples with load1 under 12 and load5 under 18, and `yes=0`. `quiet=true` for every row. `mac_heavy.sh` was not used. The script is outside this repo.

Harness dirty hash, excluding the results files and excluding `Cargo.lock`: `b92b3fb70b786679598bb484eb3e2fa10c79223058a9086f5e2bbc09d9340373`.

Unlocked cargo had rewritten the worktree `Cargo.lock` from gusset 0.0.2 to 0.0.3. That was reverted. `git diff --stat -- Cargo.lock` was empty afterward. Primary's `Cargo.lock` was not touched.

## Criteria

1. CPU reduction, fused CE, and Exact gemv rows. Met. Threads 1 and 18. Fused CE is Exact and Fast at V=50304 (128×768) and V=248320 (1×2048). Gemv `1×2048×6144` is Exact only.
2. Muon, GDN T=2048, cold 16M NaN-scan ops, cached attention at 8k and 32k. Met. Same thread list.
3. capi `op_generate` argmax of 50304 logits with and without the pressure probe. Met. `probe_ok=true`, token 50303. No `Cargo.toml` edit. The new public function is `generate_argmax`. Session lookup and wire parse are in neither arm.
4. Host `CacheKey::new` of `GDN_PUBLISHED` plus `STRICT_SM90`. Met. `source_len=20879`. No device.
5. The directory, the env block, min / median / spread, and the citation table. The directory is written. The commit the criterion also names was not made. That is an owner hold.

No kernel speedup is claimed. At both fused-CE shapes Fast is faster than Exact on the whole op, so a reading of S4-03 that says the Fast flag does nothing is not what these rows show. The 65536×128 RMS backward and the V=248320 Exact backward are slower at 18 threads than at one. Those numbers are the baseline the fix cards cite. The fixes stay on their own cards.

## Checks

Verified:

- Release `--no-run` for the three binaries: `CPU_EXIT:0`, `CAPI_EXIT:0`, `CUDA_EXIT:0` in `target/w19-cpu-build.log`.
- Debug registration test `audit_baseline_cases_are_registered`: 1 passed, log `target/w19-register.log`.
- capi unit `generate_argmax`: 1 passed, 117 filtered, log `target/w19-capi-unit.log`.
- All eight timed groups: `GROUP_EXIT:0`. 49 `OJAS_OP` lines.
- `rustfmt --edition 2021 --check` on the five Rust files exited 0 before the timed runs. Those files were not edited after that.
- Collision on the five source paths, after the edits: each is dirty only in this worktree. `ok=true`, `truncated=false`, `scanned_worktrees=13`, `sessions_ok=true`. Repo-wide overlaps are `Cargo.lock`, `docs/op-coverage.md`, `ojas-core/src/autocast.rs`, and `ojas-core/src/lib.rs`, none of which this wave holds.
- Insights before the review move: `deadline_expired=false`, collisions `ok=true`, `truncated=false`, 13 worktrees scanned, 0 unscanned. One busy Claude session on primary (pid 44032). Ledger `legacy_ledger_unconsolidated`, `stale=true`. DevMap generation 7131, fresh. The affected-tests walk earlier in the wave was depth-capped; blast radius is a lower bound. The edits do not change production kernel symbols. `generate_argmax` is additive.

Not run, and not claimed:

- `dev verify`. `.devcouncil/config.yaml` does not exist. Adding it is an owner decision.
- The full workspace suite.
- A kernel A/B. There is no before/after on a changed hot path, so there is no 5% regression number.

Adversarial pass: code-reader `01a17b19-b4e8-7130-85a2-20d4c1c62c80` read the harness before the timed runs and reported no measurement-breaking bug. The two near-misses it rejected were the default thread list (6, 18), which the runs override with `OJAS_BENCH_THREADS=1,18`, and a capi refusal arm, which did not happen (`probe_ok=true`).

## Fable

GPU step-throughput, the torch 2.15 clip-grad scorecard, and the Metal Fast/Autocast cards were not taken. Their files are held, or they need a quiet paired torch run this machine did not have at selection time. This baseline is the audit-wave-1 card those later CPU fixes have to cite. One card this wave.

## Deferred

Still not touched: ft-2c183b71, ft-ebbed90b, ft-6326debf, ft-e15ee76a, ft-e6d8035e, ft-693bbfcb, ft-9c13f472, ft-562b0add, ft-845929b1, ft-9aaa6349, ft-bd6373d2, ft-ee2a3753, ft-a95d2145, ft-f40eeed8, and the in-progress cards ft-71629756, ft-a1a4ba4d, ft-2bbced18, ft-b5e64557, ft-53d639de, ft-96baeabc, ft-92c5e9c6. Waves 13–18 stay in review and were not reopened.

## New tasks

None. The downstream fixes already have cards. This wave does not file them again.

## Owner decisions still pending

- Committing `bench/results/2026-10-16-audit-baseline/` and the harness diff. Not done.
- E9 (`--locked` / SHA pins), E11b / OD10 push, E12 (ft-69ef4ce1 back to ready), OD2, OD3, OD6 (S4-09 design), OD7 (what to do about the generate probe after this row), OD8, OD12, and committing `.devcouncil/config.yaml`.
- A Lappi pin bump past `4123dfb` still needs the wave-18 diff committed first, then an explicit yes. This wave did not do that.
