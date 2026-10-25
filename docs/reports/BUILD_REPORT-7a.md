# Build report: ojas-7a, training wave (in-progress and priority training cards)

**Run:** 2026-10-20 to 2026-10-24, orchestrated from the main checkout by session ojas-7a. This file is separate so
that no peer report (`BUILD_REPORT.md`, `-w11`, `-a6`) is overwritten.

**Base:** `main` at `40dd135`. **Result:** branch `gp/7a-training`, merged into local main on the owner's
instruction as `6981f07` (2026-10-22, `--no-ff`; branch deleted). Then `42d5a9f`, the tessl commit-name note
for the a6 record. **Not pushed:** origin/main is still `4123dfb`.

**Independent check (2026-10-24):** an Opus verifier re-read every claim here against the code and the run logs.
No code or test claim was wrong. It corrected stale text in this report (merge status, card revision, the fmt
wording) and found the gaps listed under "Follow-up cards" below.

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
| ft-b5e64557 ojas-train (P0) | in_progress | in_progress (rev 21) | crits 6, 11, 12, 13 met [V]; 1, 2, 5 wait on the owner's Lappi pin bump |
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

**Format.** The new files are rustfmt-clean. `scripts/fmt_check.sh` still fails on drift that predates this work, on
lines no lane changed: `ojas-io/src/json.rs`, `ojas-model/src/qwen35/bank.rs`, and pre-existing spots in
`step_provider.rs`, `trainer.rs` (:1384/:1579 at base) and `gpu_memory_rows.rs`, files the lanes edited elsewhere.
I confirmed it against the clean worktree, and it belongs to ft-b4f9ce4e.

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

- **ft-091e7fc2** (inbox, P2): expose tessl's `Qwen35Model::train_step_fused_adamw` (tessl main `69f60b0`, the
  waves 21-22 merge) through ojas-qwen35, and map its torn-state latch to a typed error. The card carries
  tessl-ad's API notes, a parity criterion against the bank path, and a measured memory row (tessl reports the
  2B f32 AdamW arm drops from 34.9 to 25.3 GiB). Whether to add a C entry point for it is an owner decision.

Every adversarial finding fits an existing card:
- F3/F4 and the Lappi converter note are on ft-b5e64557.
- The fmt and sitegen drift is ft-b4f9ce4e.

## Coordination

- tessl-ad held tessl main at `c2c1612` for the whole GPU wave, then fast-forwarded to `c0d8be5` (6aa333a). The
  re-run tiny rows carry `tessl: 6aa333a`.
- tessl-ad regenerated `tessl/target/qwen35_train_ref`. ojas's 2B `gpu_parity` case still cannot run here: it holds
  about 40 GB, over the 32 GiB memguard, as its own doc says.
- tessl main then moved to `69f60b0` (a --no-ff merge of waves 21-22, ci_local 3472 passed / 0 failed per
  tessl-ad) after ojas-7a confirmed no ojas build was in flight.
- Post-merge: main `6981f07` passed on tessl `e636223` (all three crates' suites, clippy, Linux check, both
  Metal tests; logs `verify-merge`). Main `42d5a9f` then passed on tessl `fc778a2` (waves 23-24: `TensorSource`
  methods, the `Weight` enum, fail-closed loaders): `cargo check --workspace --all-targets`, ojas-qwen35,
  ojas-model and ojas-metal suites, the GPU sharded load and Tape provider tests (logs `check-tessl`). That
  second run did not include ojas-train or the image-MLP Metal test.
- tessl-ad pruned the 12 refs that kept the scrubbed LGPL blobs reachable (owner-approved in its session, Fable's
  amended criterion); the four old tessl names in `BUILD_REPORT-a6.md` are mapped by `42d5a9f`.

## Owner decisions pending

1. **Push main** to origin, if wanted (merged locally on 2026-10-22).
2. **The Lappi pin bump** for ft-b5e64557 crits 1, 2 and 5. The target becomes `6981f07` or later, not `4123dfb`.
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

## Follow-up cards (filed at wrap-up, 2026-10-24)

- **ft-307275ca** (P2): the gaps from the independent check. Crit 12's card wording, `Decay::from_json` error
  messages and the explicit cosine kind, the inherited F3/F4 rate edges, the image MLP's memory pre-check, the
  refused-AdamW bank behaviour, a re-run of ojas-train and the image-MLP Metal test on the current tessl, and the
  StepProvider job generalisation.
- **ft-6d69dd5b** (P1): the owner queue. Push, the Lappi pin bump, Manvi verify, ft-71629756, disk, and the
  unclaimed uncommitted doc edits in the main checkout.
- **ft-2bbced18** (logged, rev 15): rerun the A/B with the arm order rotated, and an A/B for crit 3 or a ruling
  that exempts it.
- **ft-091e7fc2** (P2, filed earlier): expose tessl's `train_step_fused_adamw`.

## Found at wrap-up, needing the owner (2026-10-24)

- **Orphaned uncommitted work:** worktree `.gitpulse/worktrees/repair-softmax-collapse-in-the-logit-los-4c0e0a2a`
  (branch `gitpulse/repair-softmax-collapse-…`, no commits beyond `42d5a9f`, no live session) holds an uncommitted
  softmax-collapse fix. It touches `ojas-core/src/{autocast,exp_exact,lib}.rs`,
  `ojas-cpu/src/{exp,fused_ce,pointwise}.rs`, `ojas-cpu/tests/heavy_ops.rs`, `ojas-cuda/src/ce_rows.rs`, and adds
  `ojas-cpu/tests/softmax_collapse.rs`. `git worktree remove` refused it as dirty, and it was not forced.
  - **Fable (2026-10-24): preserve, do not merge.** It is a complete-looking, uncommitted, **unverified** rewrite
    of cross-entropy to the target-excluded form (`loss = ln(1 + Σ_{j≠t} exp(z_j − z_t))`) on CPU, Metal, wgpu and
    CUDA, with tests on every backend. Past the f32 cliff (margin ≳ 16.6) the old target gradient rounds to 0 and
    the row stops summing to zero.
  - **Saved:** `backups/7a-tools/softmax-collapse-4c0e0a2a-vs-42d5a9f.patch` (gitignored; 16 files, 2,235 lines,
    sha256 `a09f082fb1e797287403d3e85fb10a575a4a77a73967631826df5805dc43e88c`). `git apply -p3 --check` against
    main passes. The worktree, its branch and its `.devcouncil/` are left in place.
  - **Before it can merge:**
    - The owner's yes on the contract change. It moves the pinned Exact `ce_loss`/`ce_grad` digests in
      `ojas-cpu/tests/exact_golden.rs:271-272`, which it does not re-record, so the CPU suite goes red until they
      are re-recorded. It also retires the "as tessl does" claim in `ojas-cuda/src/ce_rows.rs`.
    - A Metal test run.
    - A CPU Fast-path CE A/B (`bench_ops.rs::ce_case`): a per-lane skip compare in `exp_sub_sum`,
      `exp_sub_store` and `row_max` may defeat vectorisation.
    - The CUDA host-mirror tests.
  - **No overlap** with the live autocast branch. The main checkout's unclaimed README/status edits are not from
    this worktree.
  - **Why not committed as WIP:** the gate refuses git against the sibling worktree, so that needs a per-case yes.
- **Stale GitPulse reader:** at 2026-10-24 ~14:35 the installed `gitpulse-mcp` started refusing board reads with
  `unsupported DevCouncil schema version 10; this binary knows 9`. Card writes worked minutes earlier, and the
  wrap-up cards were filed before the change. Per the cross-app schema rule, GitPulse needs re-vendoring
  (`node scripts/vendor-crates.mjs`, `check:vendor-schema`), a rebuild and a reinstall, then
  `bun run check:schema-parity` from the GitPulse checkout. ojas-7a cannot run those from ojas.

## Cards pending filing (GitPulse blocked, 2026-10-24)

Fable ruled that the wrap-up is done with a named gap: these items are not on the board until GitPulse reads
schema 10. On 2026-10-25 Fable ruled to close out without waiting for the peers' verification. Both peer branches
stay unmerged, with their worktrees intact and their queued verify jobs left running.

**Owner actions, in order:**

1. **Free the cargo slots.** Both slots in `~/.cache/cargo-slots` have been held since at least 18:18Z on 2026-10-24
   (still held at 00:34Z on the 25th) by a process the gate would not let ojas-7a identify. Run
   `lsof ~/.cache/cargo-slots/slot1 ~/.cache/cargo-slots/slot2`. Card 3 below.
   - **Update, 2026-10-25:** the slots freed later on the 24th, and the peers' jobs then ran. The holder was never
     identified, so card 3 still stands for the bound on hold time.
2. **Rebuild GitPulse** for schema 10 with the commands below. Card 4 below.
   - **Update, 2026-10-25 ~11:20Z:** `~/.cargo/bin/gitpulse-mcp` was reinstalled at 22:49 on 2026-10-24 and now
     reads schema 10. Every call is still refused, now with `schema_upgrade_required: workbench schema 13 must be
     upgraded to 14 before this request, and this host does not upgrade a shared profile; open the desktop app
     once to upgrade it`. GitPulse.app is still the Oct 17 build, so what remains is to rebuild the app and open
     it once.
3. **File the cards:** `backups/gp/wrap5.json` and `backups/gp/wrap6.json` through `backups/gp/mkreq.sh`, plus the
   ft-2c183b71 note below and the seven ft-e15ee76a card drafts in `docs/reports/BUILD_REPORT-e15.cards.json`.
4. **Peer branches:**
   - The autocast branch (ft-e15ee76a) is **merged**: main `5f5666a`, a `--no-ff` merge of `d351fc5`. Its
     worktree and branch are removed once its session confirms it no longer needs them.
   - `gp/4b-gpu-step` (ft-2c183b71) merges only after ojas-4b reports green:

     ```sh
     git merge --no-ff gp/4b-gpu-step   # the Cargo.toml member drop is uncommitted and stays out
     ```

     Remove `backups/x/wt-4b` only after you discard or restore its local `Cargo.toml` edit.
5. **The five unclaimed edits in the main checkout:** `README.md`, `docs/status.md`, `site/index.html`,
   `docs/reports/BUILD_REPORT-a6.md` and `tools/a6/README.md`. They are untouched and not from ojas-7a; see
   ft-6d69dd5b.

**Owner commands** (from a session rooted in `/Users/bharath/Code/devtools/GitPulse`; script names from its
`package.json`):

```sh
bun run vendor                # node scripts/vendor-crates.mjs
bun run check:vendor-schema
bun run mcp:install           # cargo install gitpulse-mcp + gitpulse-hook, then record-install
bun run mcp:doctor            # OK on all three verdicts
bun run check:schema-parity   # also covers the gitpulse-mcp inside GitPulse.app, so rebuild the app too
```

Then file the cards below from ojas (`backups/gp/mkreq.sh backups/gp/wrap5.json <out.jsonl> <id>`, piped into
`gitpulse-mcp`).

1. **Orphaned target-excluded cross-entropy rewrite** (P2, medium, label `numerics`). This is
   `backups/gp/wrap5.json` verbatim:

   ```json
   [{"name":"gitpulse_add_task","arguments":{"repo_path":"/Users/bharath/Code/research/ojas","title":"Orphaned target-excluded cross-entropy rewrite (softmax collapse past the f32 cliff): owner decision, then verify and land","priority":2,"severity":"medium","labels":["numerics"],"description":"Found at ojas-7a wrap-up (2026-10-24). Worktree .gitpulse/worktrees/repair-softmax-collapse-in-the-logit-los-4c0e0a2a (branch gitpulse/repair-softmax-collapse-in-the-logit-los-4c0e0a2a at 42d5a9f, no commits, no live session) holds an uncommitted, unverified rewrite of cross-entropy to the target-excluded form loss = ln(1 + sum_{j!=t} exp(z_j - z_t)) on CPU, Metal, wgpu and CUDA, with tests on every backend (new ojas-cpu/tests/softmax_collapse.rs). Past the f32 cliff (margin >~ 16.6) the old p - 1 target gradient rounds to 0 and the row stops summing to zero. Saved as backups/7a-tools/softmax-collapse-4c0e0a2a-vs-42d5a9f.patch (16 files, sha256 a09f082fb1e797287403d3e85fb10a575a4a77a73967631826df5805dc43e88c; git apply -p3 --check passes on main). Fable: preserve, do not merge without the owner. Details: docs/reports/BUILD_REPORT-7a.md.","acceptance_criteria":["Owner decides the Exact CE contract change (it moves ojas-cpu/tests/exact_golden.rs:271-272 ce_loss/ce_grad digests and retires the 'as tessl does' claim in ojas-cuda/src/ce_rows.rs)","If yes: the patch committed on its own branch, the digests re-recorded with the decision cited, CPU/Metal/wgpu/CUDA-host tests green","CPU Fast-path CE A/B (bench_ops.rs::ce_case) shows no regression over 5% from the per-lane skip compare in exp_sub_sum/exp_sub_store/row_max, or the loops are restructured","The worktree and branch removed after the patch lands or is discarded"]}}]
   ```

2. **ft-2c183b71: `gp/4b-gpu-step` is at `def246a` and unverified.** That is one commit on `42d5a9f`: 8 files,
   +305/-57, across ojas-wgpu, ojas-kernels and ojas-core. The ojas-4b session's verify job is queued on the
   cargo slots, and its pre-fix red run timed out before it ran. Bench and profile criteria are still owed. Its
   worktree `backups/x/wt-4b` has a local `Cargo.toml` edit that drops the `ojas-capi` and `ojas-gusset-engine`
   members, which must never merge. Merge only after ojas-4b reports green. Log this on ft-2c183b71 rather than as
   a new card.
3. **Cargo-slot starvation** (new card, P1, `backups/gp/wrap6.json` entry 1). Both slots have been held for more
   than 6 hours by an unidentified holder, so every verify job in this repo starves. The card asks for the holder
   to be identified and released, and for mac_heavy to bound how long a slot can be held.
4. **GitPulse reader at schema 9** (new card, P1, `backups/gp/wrap6.json` entry 2). It needs the rebuild above,
   confirmed by `mcp:doctor` and `check:schema-parity`.
5. **ft-e15ee76a: the autocast branch is at `95c4bb3` and unverified.** That is 4 commits on `42d5a9f`: `5aa04db`,
   `bc19d9a`, `0a8f76a` and `95c4bb3`. Log this on ft-e15ee76a.
   - In its verify job, `STEPS.txt` shows `core-lib exit=142`, a slot timeout before any test ran. `model-lib` was
     still waiting at 00:34Z. Logs are in
     `.gitpulse/worktrees/route-…-6223aeac/backups/e15-tools/logs/verify1/`.
   - **Open:** `tests/autocast_lane.rs` asserts +2 dispatches per bf16 GEMM, a figure inferred from tessl source
     and not yet seen in a run. The CPU Muon bf16 A/B (`backups/e15-tools/jobs/muonab.sh`) has not run, so
     `0a8f76a`'s speed claim is unmeasured. Only its bit-identity is pinned, by `ojas_cpu_parity`.
   - The card goes to review, not done; the Manvi verify step is the owner's.
   - **Later (peer report):** once the slots freed, the peer reported these results: model-lib 28/28,
     model-trainer 19/19, cpu-lib 56 pass/5 ignored, oracle-cpu-parity 4/4, and all 36 metal binaries green,
     including autocast_lane 3/3, which confirms the dispatch arithmetic by a run. Still running then:
     lora-metal, oracle-parity and clippy; core-lib reruns in verify2, followed by the Muon A/B. ojas-7a did not
     check these logs. The peer's `docs/reports/BUILD_REPORT-e15.md` supersedes this item.
   - **Merged 2026-10-25 as `5f5666a`.** The branch finished at `d351fc5`, with `b352ca6` and `b42676d` coming from
     its independent Opus check. ojas-7a re-read the per-step logs and the trial-merge `c828c9e` logs. That trial
     combined main `e999e35` with `gp/4b-gpu-step` at `7aba6b3` and ran the core, cpu, kernels, oracle-parity,
     metal, wgpu and model suites. Every `test result` line is ok, there are no `FAILED` lines, and clippy exited 0.
     The commit after the trial adds only `BUILD_REPORT-e15.md`. The branch touches none of the five unclaimed
     dirty files.
   - **Not yet covered at merge:** the ojas-qwen35 and ojas-capi tests, which both depend on ojas-metal, and the
     verifier's reread of `b352ca6..b42676d`. The peer is running both on `d351fc5`, which has the same code as
     `5f5666a`, and will add an addendum to `BUILD_REPORT-e15.md`. ojas-7a merges that addendum, and any fix, when
     the peer sends "clear".

## Left behind

- `target-7a/` (1.9 GB) for follow-up runs on the branch.
- `backups/7a-tools/` and `backups/gp/` (gitignored): logs, jobs, the launcher.
- Symlinks `backups/tessl`, `backups/x/tessl` and `backups/devtools` (the gate refused removing links that point
  outside the repo).
- No processes, worktrees or jobs are running.
