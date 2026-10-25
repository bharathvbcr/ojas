# Build report: ft-e15ee76a, Fast/Autocast Metal lanes, clip, grad_clip 0, CPU bf16 Muon

**Run:** 2026-10-24 to 2026-10-25, session route-ojas-metal-fast-autocast-to-tessl, in the GitPulse worktree
`.gitpulse/worktrees/route-ojas-metal-fast-autocast-to-tessl-6223aeac` on branch
`gitpulse/route-ojas-metal-fast-autocast-to-tessl-6223aeac`. A separate file so no peer report is overwritten.

**Base:** `main` at `42d5a9f`. **Result:** the commits below. ojas-7a merged `d351fc5` into local main as
`5f5666a` (`--no-ff`, 2026-10-25) after re-reading these logs, and merges the addendum commits after it.
**Not pushed.** This session's gate does not let it touch the main checkout.

**Card status:** review, not done. Two criteria are partial by Fable's ruling (2026-10-24), with the remainder
split into cards. Manvi `work.verify.task` is the owner's. The board refuses every call
(`unsupported DevCouncil schema version 10; this binary knows 9`), so the status and the cards are not on it yet.

**Builds:** the owner authorised a detached in-repo launcher (`backups/e15-tools/`, gitignored), with every cargo
step inside `Lappi-decision/tools/mac_heavy.sh` and this worktree's own target dir. The worktree reaches the
workspace through the alias `/Users/bharath/Code/research/ojas-e15`.

**Labels:** **[V]** verified this run by a command or a read, **[I]** inferred, **[U]** unverified.

## Criteria

| # | criterion | status |
| :-- | :-- | :-- |
| 1 | ojas-metal under Fast/Autocast on tessl's cooperative bf16 and tf32-relaxed lanes and gemm_epilogue, Linear and conv2d off the ExactF32 fallback | **Partial.** Autocast's bf16-exact Linear GEMMs (forward, both backward GEMMs, backward-input) run on tessl `GemmOperands::Bf16`, the 4-simdgroup cooperative kernels, on a TensorOps device [V]. tf32-relaxed, gemm_epilogue and untagged f32 are blocked in tessl [V, reasons below] → card `gp-metal-relaxed-f32-linear-lane`. conv2d does not exist in ojas-metal (`rg -uu`) [V] and is G-V2 (ft-96023b34). Measurement and the other GEMM sites → card `gp-metal-bf16-lane-measure-and-extend`. |
| 2 | clip_grad_norm_ gap against torch 2.15: multi-tensor norm in one command buffer | **Met in code before this branch** [V]: `Worker::clip_norm` is one tessl `SqNormPlan` binder scope for any count, then scale 24 gradients per dispatch; `ojas-metal/tests/clip_multi.rs` pins it. Norm and scale are two commands with a host readback between them [V]. The doc now quotes no ratio until the paired 2.15 run (`tasks/gp-clip-paired-torch215-norm-only-and-bw-ceiling.md`). |
| 3 | grad_clip 0 rejected or handled; fail closed on a trainable slot with no gradient | **Met** for ojas-model [V]. `TrainConfig::validate` refuses 0, -0 and non-finite (the C API and checkpoint resume both go through it). The no-gradient refusal predates this branch and is now pinned. The semantics brief asks for torch's skip instead → card `gp-no-gradient-slot-ruling`. arch02's separate trainer → card `gp-arch02-grad-clip-flag`. |
| 4 | CPU bf16 Muon slowdown removed with BFMMLA in ojas-simd; GPU Muon NS5 batched over 3D banks | **Partial.** Two of three bf16 rounding passes folded, bit identical: run on the oracle's 4 small cases [V]; on every route, including the 2048x768 view and one-`cblas_sgemm` routes the oracle does not reach, by reading (the change is inside the combine step all routes share, one rounding at each value's single read) [V by read, not run]; 768x768 bf16 about 6% faster, larger shapes within noise [V, A/B below]. Rust 1.99 has no bf16 NEON intrinsics [V] → card `gp-cpu-bf16-muon-bfmmla-kernel`. Bank NS5 changes the frozen Backend trait → card `gp-gpu-muon-bank-ns5`. |

## Commits

| sha | what |
| :-- | :-- |
| `5aa04db` | trainer refuses grad_clip 0; no-gradient refusal pinned; `clip_scale(0, n) == 0` documented and tested |
| `bc19d9a` | Metal Linear GEMMs on bf16-exact operands take tessl's bf16 cooperative lane; copies charged to the budget |
| `0a8f76a` | CPU bf16 Newton-Schulz: two rounding passes folded into the combination passes, bit identical |
| `95c4bb3` | clip doc section restructured |
| `5dde956` | clip doc: unpaired ratio out of the headline, measured build named (independent check) |
| `b352ca6` | Metal picks the bf16 lane by the device's TensorOps too; no copies charged without them (independent check) |
| `ad3272d` | Go `GradClip` documents the refusal; semantics brief criterion ticked |
| `ab2cdde` | the seven card drafts, `BUILD_REPORT-e15.cards.json` |
| `b42676d` | `DeviceInfo` derives `Debug`: `b352ca6` broke ojas-metal's lib-test build, caught by clippy |
| `f682199` | CPU Muon bf16 A/B, `bench/results/2026-10-25-cpu-muon-bf16-fold` |
| `1e39ad8` | bench-plots and site pages regenerated |
| `1e8fe6a`, `d351fc5` | this report |
| `4c1640a` | lane tests: an Apple M5 must report TensorOps; backward checks exact bits without them (Opus re-read) |

## Verification [V]

Logs are in `backups/e15-tools/logs/` in this worktree (gitignored; deleted with it).

**Branch, by step** (release, `--no-fail-fast`, Metal single-threaded):

| step | tree | result |
| :-- | :-- | :-- |
| ojas-core lib | `b42676d` | 154 passed, 3 ignored (verify1's first attempt timed out on the lock before running) |
| ojas-model lib | `95c4bb3` | 28 passed, incl. `a_trainable_slot_without_a_gradient_fails_the_step_and_changes_nothing` |
| ojas-model `tests/trainer.rs` | `95c4bb3` | 19 passed, incl. `a_zero_grad_clip_is_refused_at_validation` |
| ojas-cpu lib | `95c4bb3` | 56 passed, 5 ignored |
| ojas-oracle `ojas_cpu_parity` | `95c4bb3` | 4 passed, incl. the bf16 Muon step bit-identical to stock nanolab (Exact and Fast, 4 cases, 0 values differ) |
| ojas-metal, all targets | `b42676d` | 37 test binaries green; lib 81 passed, 1 ignored; `autocast_lane` 3/3; `clip_multi` all pass |
| ojas-model `qwen35_lora_metal` | working tree built 02:49:54Z, after the `b352ca6` edits were on disk and before `b42676d` [I from timestamps] | 4 passed (the end-to-end Metal Autocast case) |
| ojas-model `oracle_parity` | same working tree, built later | 16 passed |
| clippy `-p ojas-core -p ojas-cpu -p ojas-metal -p ojas-model --all-targets -D warnings` | `b42676d` | clean (verify1's run on `b352ca6` failed: `DeviceInfo` lacked `Debug`, fixed in `b42676d`) |
| `go run scripts/benchreport/main.go plots -check`, `go run scripts/sitegen/main.go -check` | `1e39ad8` | both up to date |

Only ojas-cpu's Muon, ojas-core's tag accessor, ojas-model's validation and ojas-metal changed in code. The
ojas-core, ojas-cpu, ojas-oracle and ojas-model lib/trainer steps ran before `b352ca6` and test crates it does
not touch; ojas-model's two Metal suites above built with it. ojas-metal's other dependants, ojas-qwen35 and
ojas-capi, are under "Addendum". Commits after `b42676d` change docs, bench data and generated site pages only.

**The bf16 lane ran (crit 1).** `autocast_lane.rs` at 256x192x320 (no split-K route) asserts +2 dispatches per
bf16 GEMM, which only tessl's `ensure_bf16` casts on the bf16 lane make. Before this run that figure was inferred
from tessl's source; it now passed on this M5 Pro (TensorOps). On a device without TensorOps the same tests
assert the exact lane's bits and no copy charge. That branch has not run here: this Mac has TensorOps, so
`b352ca6` changes nothing it can observe, and no test of it fails first on this device [I].

**Combined tree.** Both this branch and `gp/4b-gpu-step` change ojas-metal, so their merge was tested before
it is made. `c828c9e` is an unreferenced trial merge (`git merge-tree` + `git commit-tree`, no ref moved) of main
`e999e35`, this branch at `1e39ad8` and `gp/4b-gpu-step` at `7aba6b3`, both merges conflict-free. It was checked
out as a scratch worktree with its own target dir:

| step | result |
| :-- | :-- |
| `cargo test -p ojas-core -p ojas-cpu -p ojas-kernels` | 53 test binaries green |
| ojas-oracle `ojas_cpu_parity` | green |
| ojas-metal, all targets | 37 test binaries green; lib 81 passed, 1 ignored |
| ojas-wgpu, all targets | 28 test binaries green |
| ojas-model, all targets | 43 test binaries green |
| clippy over those six crates, `--all-targets -D warnings` | clean |

The first Metal and clippy attempt failed to compile ojas-metal's lib tests: `device.rs:3764`
`include_str!`s `../../../tessl/src/runtime.rs`, a path that holds only at the main checkout's depth. A link at
the scratch tree's level fixed it; the code was not changed. ojas-qwen35 and ojas-cuda tests rely on the same
sibling layout. That is the repo's design, not a defect of either branch.

**tessl:** main `8bc0468` when these built. Another session was building in tessl meanwhile; its working-tree
state was not checked (the gate refuses git against sibling repos) [U].

## Addendum (2026-10-25, after the merge)

**ojas-metal's other dependants** ran on `d351fc5`, the code ojas-7a merged as `5f5666a` [V]:

| step | result |
| :-- | :-- |
| ojas-qwen35, all targets | 18 test binaries green: 68 passed, 27 ignored (the `gpu_*` tests are `#[ignore]` by design; the 2B ones exceed the 32 GiB memguard) |
| ojas-capi, all targets | 116 passed, 1 ignored |
| clippy `-p ojas-qwen35 -p ojas-capi --all-targets -D warnings` | clean |

**Opus re-read of `b352ca6`, `b42676d` and `5dde956`.** It confirmed both fixes do what its findings 7 and 8
asked, with no correctness issue. One new weakness: every lane assertion branches on `has_tensorops()`, so a flag
that read false by mistake on a TensorOps Mac would pass all three tests on the exact lane. `4c1640a` adds
`an_apple_m5_reports_tensorops`: tessl sets the flag when its library holds `matmul2d_tensorops_f32`, and an
Apple M5 that reports none fails. Without TensorOps the backward test now checks gX and gW bits too. It ran in
verify5 on `4c1640a` [V]: the log shows `Apple M5 Pro: has_tensorops true`, `autocast_lane` 4/4, clippy on
ojas-metal `--all-targets -D warnings` clean. On `0a8f76a` it judged the fold bit identical on every route by
reading (the change is inside the combine step, shared by the view, `one_cblas` and band routes); it was run
only on the oracle's 4 small cases. The crit 4 row's labels say so.

**Not covered:** `4c1640a`'s two new checks have not run on main combined with ojas-4b's final Metal commit.
ojas-4b's verify9 runs every ojas-metal target on that combination, with the earlier `autocast_lane`.

## Independent check (Opus, 2026-10-25)

A read-only Opus verifier read the diffs, tessl at `8bc0468` and the logs. No correctness bug in the original
four commits. It found, and this branch then fixed: the clip doc's headline quoted an unpaired ratio its source
says not to quote and called the 2026-10-17 build "what main runs" (`5dde956`); the bf16 lane and its copy charge
ignored TensorOps (`b352ca6`). It confirmed the tessl reasons for crit 1 (relaxed is runtime-global and refuses
every ExactF32 call, `runtime.rs:1089`, `gemm.rs:1341-1347`; relaxed kernels are NN only; `gemm_epilogue`
accepts NN only, `gemm.rs:970`, and Linear forward is NT) and the bit-identity argument for `0a8f76a`
(each element rounded once at its single read). Its other findings are cards.

## CPU Muon bf16 A/B (crit 4)

`bench/results/2026-10-25-cpu-muon-bf16-fold`: 5 interleaved rounds, `42d5a9f` against `b42676d`, 6 threads.
The unchanged f32 rows are the control: 0.96–1.09 per round.

| shape | bf16 median new/base | new faster in | bf16 over f32, base → new |
| :-- | --: | :-- | :-- |
| 768x768 | 0.940 | 5/5 | 1.38x → 1.32x |
| 2048x768 | 0.987 (within noise) | 4/5 | 1.25x → 1.23x |
| 3072x768 | 0.967 (within noise) | 4/5 | 1.27x → 1.23x |

New round 1 ran at 1-minute load 38–95; the rest at 3.5–15.6. Without it 768x768 is still 0.94, 4 of 4.

## Cards

Drafts, as `gitpulse_add_task` arguments: [`BUILD_REPORT-e15.cards.json`](BUILD_REPORT-e15.cards.json).

| task_id | P | what |
| :-- | :-- | :-- |
| `gp-cpu-bf16-muon-bfmmla-kernel` | 2 | profile the remaining bf16 gap; BFMMLA kernel only if it beats Accelerate |
| `gp-gpu-muon-bank-ns5` | 2 | one Backend op for a bank of same-shape Muon slots (trait change, owner sign-off first) |
| `gp-metal-relaxed-f32-linear-lane` | 2 | untagged f32 Linear on a cooperative tile: a tessl per-call lane and a Numerics tier ruling |
| `gp-metal-bf16-lane-measure-and-extend` | 2 | A/B the bf16 lane; extend to linear CE, attention, gate; one cast per operand |
| `gp-metal-splitk-scratch-budget` | 2 | tessl's TN split-K partials are not charged to the Budget, on either lane [I] |
| `gp-arch02-grad-clip-flag` | 3 | arch02 `--grad-clip` does not change the device clip, and 0 is accepted |
| `gp-no-gradient-slot-ruling` | 2 | owner ruling: fail closed (this card) or skip as torch does (semantics brief) |

## Owner actions, in order

1. **Finish the GitPulse upgrade.** State at 2026-10-25 11:16Z [V]: `~/.cargo/bin/gitpulse-mcp` (1.5.0,
   installed 2026-10-24 22:49) now reads DevCouncil schema 10 but answers every call with
   `schema_upgrade_required: workbench schema 13 must be upgraded to 14 before this request, and this host does
   not upgrade a shared profile; open the desktop app once to upgrade it`. `GitPulse.app` is still the
   2026-10-17 build, and its `gitpulse-mcp` still says `unsupported DevCouncil schema version 10; this binary
   knows 9`. So: rebuild and reinstall the app from a session rooted in `/Users/bharath/Code/devtools/GitPulse`
   (commands in `BUILD_REPORT-7a.md`, "Owner commands"), open it once so it upgrades the workbench to 14, then
   confirm with `bun run check:schema-parity`.
2. **File the cards and set ft-e15ee76a to review**, from the ojas main checkout once the board answers:

   ```sh
   { printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"owner","version":"1"}}}' '{"jsonrpc":"2.0","method":"notifications/initialized"}'
     jq -c 'to_entries[] | {jsonrpc:"2.0",id:(10+.key),method:"tools/call",params:{name:"gitpulse_add_task",arguments:.value}}' docs/reports/BUILD_REPORT-e15.cards.json
     sleep 60; } | gitpulse-mcp 2>/dev/null | rg '"id":1[0-9]'
   ```

   A `related_tasks_exist` answer lists ids: read them, then resend that card with `reviewed_related`. Then
   `gitpulse_complete_task` with `task_id` `ft-e15ee76acc8f98cb3214afa3b9bcb2fa`, `status` `review`, and this
   report's path as the summary.
3. **Merge (ojas-7a is doing this).** ojas-7a merged `d351fc5` as `5f5666a` and merges this branch's final head
   after it. ojas-4b's `gp/4b-gpu-step` moved past `7aba6b3`; its verify9 tests its final head merged onto main
   `216f3ed`, which already contains `5f5666a`, and ojas-7a merges it after that run. If you merge by hand
   instead:

   ```sh
   git merge --no-ff gitpulse/route-ojas-metal-fast-autocast-to-tessl-6223aeac
   git merge --no-ff gp/4b-gpu-step   # only after ojas-4b reports verify9 green; see BUILD_REPORT-4b.md
   ```

4. **Clean up (ojas-7a is doing this, on this session's "clear").** It removes this worktree and deletes the
   branch:

   ```sh
   git worktree remove .gitpulse/worktrees/route-ojas-metal-fast-autocast-to-tessl-6223aeac
   git branch -d gitpulse/route-ojas-metal-fast-autocast-to-tessl-6223aeac
   ```

   That also deletes `backups/e15-tools/` and the logs cited here. ojas-4b's worktree `backups/x/wt-4b` holds a
   local `Cargo.toml`/`Cargo.lock` member drop that must not be committed; see `BUILD_REPORT-4b.md`.
5. **Manvi `work.verify.task`** for ft-e15ee76a, and the ruling on `gp-no-gradient-slot-ruling`.

## Left behind

- **Removed:** the scratch combined-tree worktree, every target dir of this session
  (`backups/e15-tools/target`, `target-base`, `target-combo`), the `42d5a9f` archive used as the A/B baseline,
  and the aliases `ojas-e15`, `ojas-e15-base` and `ojas-e15-combo`.
- **Kept until the worktree is removed** (gitignored, in this worktree): `backups/e15-tools/` with the
  launcher, the job scripts and every log cited above. Also the link `backups/e15-tools/tessl` →
  `/Users/bharath/Code/research/tessl`, which the gate refused to delete because its target lies outside the
  repo; `git worktree remove` deletes the link, not its target.
- **Not this session's:** `.gitpulse/worktrees/tessl` (a link from 2026-10-15) and the softmax-collapse worktree
  (see `BUILD_REPORT-7a.md`).
- No processes or jobs are running.
