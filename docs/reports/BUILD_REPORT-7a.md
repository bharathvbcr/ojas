# Build report: ojas-7a, training wave (in-progress and priority training cards)

**Run:** 2026-10-20 to 2026-10-22, orchestrated from the main checkout by session ojas-7a. This file is separate so
that no peer report (`BUILD_REPORT.md`, `-w11`, `-a6`) is overwritten.

**Base:** `main` at `40dd135`. **Result:** branch `gp/7a-training` at `6335e00`, five commits, **not merged to
main and not pushed** (merging to the default branch is the owner's call).

**Focus:** the owner asked to "finish in-progress and priority tasks for training".

**Roles:**
- Opus orchestrated, wrote the briefs and reviewed every diff. One lane was returned once.
- Sonnet ran 3 implementation lanes and 2 adversarial passes. L3 was itself the adversarial pass for ft-02256073.
- Fable ruled on scope and sequencing.

**Builds:** the owner authorised a detached in-repo launcher (`backups/7a-tools/`, gitignored), with every cargo
step inside `Lappi-decision/tools/mac_heavy.sh`. This session's shell gate refuses `mac_heavy.sh` directly.

**Labels:**
- **[V]** verified this run by a command or a read.
- **[I]** inferred.
- **[U]** unverified.

## Headline

| Card | Before | After | Evidence |
| :-- | :-- | :-- | :-- |
| ft-02256073 sharded Qwen3.5 load | in_progress | **review** (done requested; held for Manvi) | all 3 criteria [V]; adversarial 26/26; GPU sharded == single-file load, first logged runs |
| ft-b5e64557 ojas-train (P0) | in_progress | in_progress (rev 20) | crits 6, 11, 12, 13 met [V]; 1, 2, 5 wait on the owner's Lappi pin bump |
| ft-2bbced18 LoRA step cost (P1) | in_progress | in_progress (rev 14) | crits 1 and 4 met with the committed A/B [V]; 5, 6, 7 open |
| ft-71629756 CUDA provider | in_progress | in_progress, blocked marker logged | needs a CUDA host nobody has |
| ft-a1a4ba4d quantised arm, ft-da2c4ab1 LoRA-2B umbrella | in_progress / ready | unchanged | remaining work is Lappi-side or owner approvals |

## Commits on `gp/7a-training`

| sha | concern | card |
| :-- | :-- | :-- |
| `713fc2e` | `LrSchedule` Decay {Cosine, Constant, Linear, Step}; name-keyed group test; adversarial tests | ft-b5e64557 crit 13 |
| `d19928a` | `ImageMlpProvider` + `ClassificationObjective` through the unchanged `StepProvider`; `init_param` split; adversarial tests | ft-b5e64557 crit 12 |
| `4affd09` | sharded-checkpoint adversarial tests | ft-02256073 |
| `ca42bfd` | `tape-provider` memory rows; `gpu_memory_rows` `ids()` vocabulary bound | ft-b5e64557 crit 6 |
| `6335e00` | step-cost A/B rows | ft-2bbced18 crits 1, 4 |

## Verification [V]

**Baseline.** Unmodified `40dd135` ran in a clean detached worktree with its own target dir:
- ojas-train: all suites green (44 tests).
- ojas-qwen35: `cpu_names` 4/4, `sharded_checkpoint` 7/7, the GPU sharded load 1/1.
- ojas-model: `init` 3/3.

The worktree's workspace list dropped `ojas-capi` and `ojas-gusset-engine`, because gusset cannot resolve from
inside the repo. None of the tested crates depend on them.

**Final pass** (`backups/7a-tools/logs/verify3`, then `gpu-wave/check.log` on the formatted, committed tree):
- `cargo test --release` for ojas-train (10 suites), ojas-model (43) and ojas-qwen35 (18): all green.
- `cargo clippy -p ojas-train -p ojas-model -p ojas-qwen35 --all-targets -- -D warnings`: clean.
- `cargo check -p ojas-train --target aarch64-unknown-linux-gnu`: OK.
- Metal: the image-MLP test and the GPU sharded load both pass.
- After the fix rounds: ojas-train `schedule_adversarial` 26 pass, 2 ignored (F3/F4 below); ojas-model
  `image_mlp_adversarial` 34/34.

**Image MLP on Metal** (crit 12):
- The resumed run equals the uninterrupted one, bit for bit.
- CPU and Metal per-step losses agree within 6.6e-8 relative (bound 1e-4). Metal is `Numerics::Fast`, so equality
  across backends is not claimed.
- Loss falls from 2.25 to about 0.25 over 12 steps.

**Format.** The new files are rustfmt-clean. `scripts/fmt_check.sh` still fails on drift that predates this work, in
files no lane changed (`ojas-io/src/json.rs`, `ojas-model/src/qwen35/{bank,step_provider}.rs`, `trainer.rs`
:1384/:1579 and others). I confirmed it against the clean worktree, and it belongs to ft-b4f9ce4e.

**Site and plots.**
- `go run scripts/benchreport/main.go plots -check` is up to date.
- `go run scripts/sitegen/main.go -check` reports 10 stale site pages. This is pre-existing [I]: this branch touches
  no docs, site or README file, sitegen reads recorded test counts from `docs/data/test-runs.json`, and the stale
  pages include untouched crates. It also belongs to ft-b4f9ce4e.

Not run: `dev verify` (memory: it does not see board cards), Manvi `work.verify.task` (owner only), and the full
workspace suite (only the three touched crates and their dependants' targets).

## Benchmarks [V]

**Step-cost A/B** (`bench/results/2026-10-21-lora-stepcost-ab`):
- Setup: 2B-Base, interleaved, min-of-5, one arm per process under the lock, ojas `4affd09`, tessl `c2c1612`.

| | T = 2,048 | T = 9,638 |
| :-- | :-- | :-- |
| base scanned once (`freeze_checked`) | -0.7% (5/5 paired reps faster; below the 4.5-5.4% spread) | no measurable effect |
| full dW vs the skip | +9.4..+15.2% step; dW = 20% of backward; +5.18 GB peak | +10.5..+14.0%; 17%; +5.36 GB |
| LoRA tokens/s (best) | 372 | 322 |

- The T = 9,638 reps 4-5 ran under hours of contention and carry no weight in the reading.
- No regression: the only code on the timed path is test harness.

**Memory per step through `TapeStepProvider`** (`bench/results/2026-10-21-tape-provider-memory`):

| Shape | Checkpointing on | Checkpointing off |
| :-- | :-- | :-- |
| 0.8B-Base, T = 2,048 | 3.96 GB | 18.08 GB |
| 0.8B-Base, T = 9,638 | 7.87 GB | typed pre-flight refusal (needs 30.07 GB > 28 GiB) |
| tiny fixture | 0.33-1.19 GB | 0.34-1.40 GB |

## Adversarial findings

**Fixed:**
- Schedule:
  - F1: `Decay::from_json` returned unvalidated Step values.
  - F2: unknown decay members were ignored.
- Image MLP:
  - `batch 0` divided by zero.
  - An overflowing normalisation table was accepted, and a non-finite loss was returned as `Ok`.
  - `close()` accepted a plan of the wrong length.
  - `classes > 256` was accepted.
  - No Budget pre-check ran before the host init vectors were filled.
- Harness:
  - Two sharded tests shared a scratch label, and one assumed F32 for a BF16 tensor.
  - `gpu_memory_rows` drew real-vocabulary ids for the tiny fixture.

**Recorded, not fixed:**
- F3: `lr_at` can exceed `peak_lr` by one ulp. F4: a huge finite peak overflows to inf during warmup.
  - Both are inherited from Lappi's `LRSchedule` formula, and fixing them breaks bit parity. They are logged on
    ft-b5e64557.
- A refused AdamW consumes the bank (retry gets `Unfilled`). Characterised.
- Suspected `adamw_all_or_nothing` post-update overflow at lr 1e300: **not reproduced**. The test passes.

## Fable's decisions (2026-10-20)

- **Crit 12:**
  - Build the MLP now with no change to `StepProvider`. Its contract is literal: pixels are tokens, vocab 256, one
    row at position 0.
  - "Bit-identical" means resumed == uninterrupted on each backend. Across backends, 1e-4 relative.
  - Generalising the job type waits for the Lappi pin bump and the G-V1 vision ops.
- **Crit 13:**
  - `Decay` is a field on the moved `LrSchedule`. Cosine bits and its 4-key JSON are unchanged.
  - The rank-4 clause is met by name-keyed groups plus a test.
  - No SGD (no recipe names it). The groups.rs move (ft-b3201323) is not folded in.
- **Sequencing:** code lanes first, then the GPU queue (cheap first, the long A/B last).
  - ft-2bbced18 stays in_progress with criteria open.
  - ft-02256073 goes to review only if both the adversarial pass and the GPU row pass.
- **ft-71629756:** the blocked marker was logged; the owner moves it.

## New tasks filed

None. Each finding fits an existing card:
- F3/F4 and the Lappi converter note are on ft-b5e64557.
- The fmt and sitegen drift is ft-b4f9ce4e.

## Coordination

- tessl-ad held tessl main at `c2c1612` for the whole GPU wave, then fast-forwarded to `c0d8be5` (6aa333a). The
  re-run tiny rows carry `tessl: 6aa333a`.
- tessl-ad regenerated `tessl/target/qwen35_train_ref`. ojas's 2B `gpu_parity` case still cannot run here: it holds
  about 40 GB, over the 32 GiB memguard, as its own doc says.
- Next from tessl: wave 21-22 adds `Qwen35Model::train_step_fused_adamw`. ojas-qwen35 is to expose it as a new
  task.

## Owner decisions pending

1. **Merge `gp/7a-training` into main** (5 commits, +6.3k lines, mostly tests and rows), and push if wanted.
2. **The Lappi pin bump** for ft-b5e64557 crits 1, 2 and 5. The target becomes the merge sha, not `4123dfb`.
   Lappi's converter (ft-58e316a9) must refuse an unknown `decay` member.
3. **Manvi `work.verify.task`** for ft-02256073, now in review, and the older review cards listed on ft-da2c4ab1
   (owner item 8).
4. **ft-71629756:** move to blocked or archive.
5. **From the cards:**
   - ft-da2c4ab1 owner items 5 (q8 pre-registration), 6 (GPU time for the quantised arm) and 8.
   - The Lappi train shards for the 0.8B mechanics check.
   - Restoring the FineWeb shards for any quality claim.
6. **Disk:** 64 GiB free at the end (was 298 GiB at the start), mostly from other sessions. Below about 50 GiB a
   cold ojas build risks ENOSPC.

## Left behind

- `target-7a/` (1.9 GB) for follow-up runs on the branch.
- `backups/7a-tools/` and `backups/gp/` (gitignored): logs, jobs, the launcher.
- Symlinks `backups/tessl`, `backups/x/tessl` and `backups/devtools` (the gate refused removing links that point
  outside the repo).
- No processes, worktrees or jobs are running.
