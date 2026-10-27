# Wave 15

Task: ft-9bed58f086849d18953d833b3d4e1197 (CUDA host side). Claimed at revision 3, in progress at revision 4 (sequence 27108, request_id w15-claim-9bed58f0). Moved to review at revision 5 (sequence 27112, request_id w15-review-9bed58f0). `verification` is null. `held_for_review` is false. Branch `gp/w15-cuda-syntax` at f327338, worktree `/Users/bharath/Code/research/ojas-w15`. Not merged. Not committed.

Only criterion 13 (S7-01) was in scope. Criteria 1–12 and 14–17 stay open. Done was not requested.

## What landed (worktree only)

`kernels::all_modules()` returns the twelve production `KernelModule` consts: K0, GEMM_FFMA, K8, K11, GDN_PUBLISHED, COL_SUM, and the ce_rows, conv1d, embed, gates_published, qk_norm_rope, and rmsnorm modules.

`all_modules_is_every_production_kernel_module` fails if that list and the column-0 `pub const _: KernelModule` declarations under `ojas-cuda/src/**/*.rs` disagree. `module_names_in` skips a `#[cfg(test)] mod tests` body by brace depth. `a_const_after_the_test_module_is_still_required_and_one_inside_is_not` covers a const after that module and a column-0 const inside it.

The macOS `clang++ -fsyntax-only -std=c++17 -Werror` check walks `all_modules()` (twelve modules). The host shim gained `__fsqrt_rn`, plus `fmaf`, `rintf`, and `fabsf`, which K8, K11, GDN, and the GEMM source need to parse as host C++. Those declarations are in the test shim only. Kernel source is unchanged.

`a_planted_gdn_typo_fails_the_host_cxx_syntax_check` inserts `@@@` after the first `extern "C"` in `GDN_PUBLISHED.source` and requires clang to fail with a diagnostic. A missing `clang++` panics.

`ALL_MODULES` is still `[&K0, &GEMM_FFMA]`. `SMALL_MODULES` is still the seven-module device lane. Device NVRTC smoke paths were not retargeted.

Diff: `ojas-cuda/src/kernels.rs` +181, `ojas-cuda/src/small_common.rs` +107/−32, plus this report. Cargo.lock was rewritten by cargo (gusset 0.0.2 → 0.0.3 from the path crate) and restored with `git checkout -- Cargo.lock`. It is not part of the change.

## Verification

| Check | Result |
|---|---|
| `cargo test -p ojas-cuda --lib -- --test-threads=1` | 185 passed; 0 failed; 3 ignored; 1.42s; exit 0. Log: `target/w15-test2.log` in the worktree. |
| The four tests named above | Each line in that log is `ok`. |
| Ignored three | Pre-existing `k8_act` full-density ulp sweeps, marked ignored for `--release`. Not skipped to pass. |
| `rustfmt --edition 2021` | Clean on the two source files after format. |
| Adversarial code-reader `01a17a29-5fc5-7613-bb76-decc88f7349e` | `no return`. |
| `dev verify` | Not run. `.devcouncil/config.yaml` does not exist. Adding it is an owner decision. |
| Workspace `ci_local` / full suite | Not run. |
| GH200 device tests | Not run. This host has no CUDA device. Criterion 7 still requires them before any device criterion is marked done. |
| Benchmark | No timed path changed. No benchmark was run. No delta to report. |

Compile used `CARGO_BUILD_JOBS=1`, `CARGO_TARGET_DIR` in the worktree, debuginfo 0, incremental off, `--offline`. Load at start was about `{16.67 23.17 28.42}`, above the 12/18 bar. The lib test profile finished in 5.92s (cold) and 2.56s (the follow-up compile). Disk stayed near 105 GB. The 8 GB floor did not fire.

DevMap generation 7128, fresh. `devmap_impact` and `devmap_affected_tests` on `ojas-cuda/src/small_common.rs` and `kernels.rs` were truncated (`walk_incomplete`, depth 2; affected tests shown 9 of 31). That is unknown blast radius, not a clean bill. The edit does not change `SMALL_MODULES` or `ALL_MODULES` values. `repository.root` was `/Users/bharath/Code/research/ojas`.

Collision checks before the edit (`ojas-io/src/safetensors.rs`, `ojas-cuda/src/small_common.rs`, `ojas-cuda/src/kernels.rs`): `ok`, 7 worktrees scanned, 0 unscanned, 0 failed, `truncated` false, no holder. `sessions_ok` true.

## Criterion 13

Met on this worktree, uncommitted. Not on main.

## Still open on ft-9bed58f0

1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14 (S7-02), 15 (S7-04), 16 (S7-05), 17 (S7-06). Several need a GH200. Criterion 6 would touch the C ABI `DeviceKind`; it was not touched. S7-02 and S7-04 share `small_common.rs` with ft-09edeb32; this slice did not take those edits. The card is in review so a person can send the rest back to ready. It is not done.

## Why this card

ft-ee2a3753 criterion 1 (two whole-tensor host copies in typed safetensors reads) is already the streaming `read_typed` on main f327338 (`ojas-io/src/safetensors.rs`, one output plus a chunk of at most 128 KiB). That file was free. The heap test in `ojas-io/tests/stream_heap.rs` was not re-run this wave. Criteria 3 and 6 stay blocked (`ojas-metal/src/device.rs` is dirty on ojas-78a; `ojas-model/src/trainer.rs` is owned by in-progress ft-b5e64557). Criteria 2, 4, and 5 were previously marked MET and were not re-read. The card was left ready.

Audit wave 3's remaining Mac-runnable bug slice was S7-01. Wave order puts ft-9bed58f0 before ft-09edeb32.

## Deferred

Held by other worktrees or in-progress cards, from the preflight at 2026-10-26T21:23:08Z: GPU step throughput (context.rs, Metal kernel), docs op-coverage (ojas-78b), training semantics (ojas-78b), Exact-tier CE and platform libm (softmax worktree), Metal epilogues and red rows (kernel held), softmax collapse (its own dirty worktree), LoRA-2B and LoRA benches (trainer lane), ojas-train LoRA-A draw order (ft-b5e64557). ft-09edeb32 was not started. No new tasks were filed.

## Owner decisions still pending

E9 (`--locked` / SHA pins), E11b / OD10 push, E12 (ft-69ef4ce1 back to ready), OD2 (C11 digest), OD3, OD6, OD7, OD12, and committing `.devcouncil/config.yaml` so `dev verify` can run. Not asked. Not done.

## Roles

Implementation, the registry test, and this report are this Grok session. The adversarial pass was code-reader `01a17a29-5fc5-7613-bb76-decc88f7349e`. This session is not Opus or Sonnet.
