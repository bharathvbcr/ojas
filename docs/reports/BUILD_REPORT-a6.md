# Build report: ojas-a6, wave Q1 (quantised Lappi LoRA training on the Mac)

**Run:** 2026-10-17, from about 20:00Z, orchestrated from the main checkout by session ojas-a6. The wave-12 report stays in `BUILD_REPORT.md` and the w1–w11 report in `BUILD_REPORT-w11.md`; this file is separate so that no peer report is overwritten.

**Base:** every lane branched from `main` at `04277f8`.

**Focus:** the owner redirected the run mid-way: "Focus the tasks on training the lappi decision model, quantised on mac; coordinate with agents in other repos; I will start training later." The audit wave order (AUDIT_REPORT.md §A5) was set aside for the cards on that path.

**Roles:**
- Opus orchestrated, wrote the briefs and reviewed the code.
- Sonnet ran 4 implementation lanes and 2 adversarial passes.
- Fable sequenced the wave.

**Labels:**
- **[V]** verified this run by a command or a read.
- **[I]** inferred.
- **[U]** unverified.

**tessl commit names (note added 2026-10-22):** tessl's history was rewritten on 2026-10-18 (see the run-9 bullet
below), so the four tessl commits this report cites have new names with identical trees. The report keeps the
names that were on disk when each run happened. The full map is `docs/history/2026-10-18-scrub-sha-map.tsv` in
tessl (102 rows). The old names stop resolving once tessl prunes the refs that still reach the pre-scrub history.

| cited here | current tessl name | what it was |
| :-- | :-- | :-- |
| `bf091a2` | `cef623ee701ee1f669fe421d5a88de45b6ee114a` | tessl main under re-run 8 |
| `a37a0c2` | `8c5131981d335b572d00247c79987385f661c0cf` | tessl main after the eg2 fast-forward |
| `41d8866` | `a083088bcf0f0d03c6d9d26b1c20bdb8421b1609` | tessl main under run 9 |
| `4e99ca6` | `a1fccca8bb10bea601a231a25c5b5ceb0d936f83` | the doc commit on 41d8866 |

`42f1d52` was never renamed.

## Headline

> **WRAP-UP (2026-10-19):** ojas main **`4123dfb` = origin/main**. It was pushed after the merge; ojas-a6 did not push. One branch (`main`), one worktree, no uncommitted tracked changes, no agents, scheduled jobs or verify runs left running.
>
> **Last verification (run 10) against tessl a8fc965:**
>
> | Check | Result |
> |---|---|
> | qwen35 CPU | 42/42 |
> | gpu_parity | 9/10 (the 2B fixture missing after the disk cleanup) |
> | Tape provider CPU | 14/14 |
> | Tape provider Metal | 5/5 |
> | clippy | clean |
>
> tessl since then (09efe84, 1c4d03e, and doc commits) changed nothing ojas builds (announced by tessl-a3 and tessl-d5).
>
> **Owner "choose the best" (2026-10-19), chosen by ojas-a6 following Fable's prior rulings:**
> - **Pin bump to `4123dfb`:** yes.
> - **q8 pre-registration:** approved by criteria.
> - **GPU time:** in Fable's order, with the 0.8B bf16 smoke first once Lappi's L1 and L2 exist.
> - **Stage 2:** not now.
> - **2B fixture:** regenerate under mac_heavy at the next quiet window (owner-run; cross-repo).
> - **ft-2bbced18 A/B bench:** defer until after the 0.8B smoke (decision-informing only; D1 was already settled by the measured rows).
> - **The 5 duplicate cards:** archive (owner's click; the board tool can't archive).
>
> Recorded in `LAPPI-WORK-ORDER.md` L0.
>
> **Open:**
> - Lappi `LAPPI-WORK-ORDER.md` L1–L6. The ojas pin target `4123dfb` is now on origin.
> - Owner: the pin-bump yes; a Lappi session; regenerating the tessl 2B fixture under mac_heavy (then re-run `gpu_parity`'s 2B test); the q8 pre-registration approval; GPU runs; archiving the 5 duplicate cards; `owner-bench-stepcost.sh` for ft-2bbced18 criteria 1 and 4.
> - ojas: criterion 5 (Tape vs Lappi's tessl provider, after the pin bump) and the tape-provider memory rows.

> **PHASE A + B MERGED AND VERIFIED (2026-10-18 ~14:45):** local main **`4123dfb`** (merge `fb057d4`), 216 commits ahead of origin, nothing pushed.
>
> | Check | Result |
> |---|---|
> | build | check clean |
> | ojas-io + ojas-train | 180/180 (moved Lappi oracles) |
> | ojas-model | 95/95 (TapeStepProvider, including the torch 3-row accumulate) |
> | Metal provider | 5/5 |
> | clippy | clean |
> | re-run 8 (main 54fa635 on tessl bf091a2) | green |
>
> **Branches:** only `main`; the phase-A and phase-B branches and worktrees are removed (merged).
>
> **Left for training:**
> - ojas: criterion 5 (vs Lappi's tessl provider, after the pin bump) and the GPU memory rows.
> - Lappi: `LAPPI-WORK-ORDER.md` L1–L6 (v6 shards, torch driver, quant gate tool, converter, the qd-train switch pinned to ojas `4123dfb`).
> - Owner: push ojas (so the pin resolves), the pin-bump yes, approvals and GPU runs.

> **VERIFIED (2026-10-18 09:15) [V logs `backups/a6-tools/logs/` and `rerun1..rerun7/`]:** local main **`94aa3af`** passes the following; nothing is pushed (172 commits ahead of origin).
>
> | Check | Result |
> |---|---|
> | build | OK |
> | ojas-io + ojas-train | 134/134 |
> | autograd | 33/33 |
> | full ojas-io/train/autograd/model | 614 pass |
> | model re-run | 137/137 |
> | Metal | 124/124, including `clip_multi` for the merged clip-sq-norm |
> | Metal lib | 81/81 |
> | qwen35 | 4/4 |
> | GPU LoRA step | passed |
> | Linux (aarch64) check of ojas-train | OK |
> | sha256 bench | 326–331 MiB/s |
> | `clippy -D warnings --keep-going --all-targets` over the six touched crates | clean |
>
> **Fixes the verification forced:**
>
> | Commit | Fix |
> |---|---|
> | `43d5e82` | preflight overflow mapped to a refusal |
> | `0a17930` | three stale tests |
> | `aaed8b3` | Metal `scale_grad` scales in place, so frozen weights refuse it |
> | `111097d` | unused import |
> | `d0fbe97`, `0e95fe7`, `81a81ca`, `94aa3af` | rust 1.99 clippy lints, including a boxed `AdapterError` quantizer record |
>
> **Still owed:**
> - the phase-A import (the owner runs it);
> - the ft-2bbced18 GPU A/B rows (`owner-bench-stepcost.sh`);
> - the `parts_words()` switch when tessl wave 15 lands;
> - deleting `target-a6-main` (2.2 GB; the blast-radius gate held my `rm -rf` for an approval that never came).
>
> The overnight watcher is stopped.

> **Merged, unverified (2026-10-17 17:30).** The owner chose "Commit to main unverified". Every a6 lane is now on **local** main as one commit per lane, `ffc6f2e`…`c695803`; nothing is pushed. Each commit was made by copying exactly that lane's files from its worktree, running rustfmt, and staging only those paths. The peer's dirty `.gitignore` and `bench/*` were never staged [V `git status`].
>
> | Commit | Lane |
> |---|---|
> | `ffc6f2e` | ojas-train and sha256 |
> | `3f87e76` | value-only CE |
> | `07f60c2` | adapters, bank, optim |
> | `4803dbd` | fq8 generator and tests |
> | `8e044d7` | merge export |
> | `525bab2` | AdamW-100 |
> | `9ffc289` | pre-flight |
> | `f0259b4` | freeze_checked and bench arms |
> | `b731d6d` | doc sync |
> | `f4bea36` | runbook |
> | `e85d7aa` | fq8 fixtures |
> | `03b450b` | AdamW fixture |
> | `c695803` | site |
>
> **Run in this session [V]:**
> - the fake-quant generator's 4 tests: OK;
> - `qwen35_fakequant_base.py`: 13 tensors, worst |dW| 3.906e-3;
> - the fq8 torch references: base loss 10.8579569, 118 tensors;
> - the default fixture regeneration: identical tensor bytes and header JSON, with only the metadata key order different;
> - `adamw_oracle.py`: gap 5.69e-6, bound 2.3e-5;
> - `sitegen` and `sitegen -check`: up to date (54 files), after registering the two new docs in `docCategory`.
>
> **Lane A, `e9547e1`:** sha256 moves to ojas-io with a streaming file digest; `load_hf_pinned` refuses a wrong base name, sha256 or sharded name before any allocation; ojas-train re-exports the hash. Opus-approved, and its tests are in the verify script. **Adversarial pass, `6f10af9`:** 2 defects fixed.
>   - A FIFO path hung forever, because it was opened before the regular-file check. The path is now stat'ed first.
>   - A file that grew or shrank while being hashed was accepted. `digest_exact` now takes the length once and refuses a change.
>
>   Also: the "cursor not moved" docs held only on Unix, so they are corrected and the cursor test is now `cfg(unix)`. The FIFO tests fail on their own 5 s deadline instead of hanging.
>
> **Branch cleanup (owner, 2026-10-17 ~19:40):** all twelve `gp/a6-*` branches and their worktrees under `../ojas-a6-*` are deleted.
> - Each branch pointed at an ancestor of main (04277f8, c695803 or e9547e1), so `git branch -d` was enough, never `-D`.
> - Before removal, every lane file in each worktree was checked as older than that lane's commit on main [V mtime scan, `backups/a6-tools/newest.py`], so the worktrees held nothing that was not on main.
> - **Kept:** `gp/clip-sq-norm` (one unmerged commit `2146553`; blocked, not stale: tessl main 02b1e93 still has no `fn sq_norm` [V rg]) and its staged merge worktree `../ojas-merge-clip`.
> - `owner-batch.sh`, `owner-merge.sh` and `owner-commit-q1.sh` are **obsolete** (they target the deleted worktrees). The one verification script is `owner-verify-main.sh`.
>
> **Not run:** cargo (compile, tests, clippy). That is `backups/a6-tools/owner-verify-main.sh`, owner-run, because the gate refuses mac_heavy. Until it is green, main may not build.

> **Code waves are paused (Fable, wave 4):** "Nine lane branches are written and reviewed but none has compiled. Further code waves stop here until you run `backups/a6-tools/owner-merge.sh` (when load is quiet or with `MAC_HEAVY_MAX_LOAD` raised for the CPU steps) and it comes back green. This wave produces only the runbook, the phase-A import script, and a static cross-lane check. Phase A itself still waits on Lappi's freeze sha."

- **Nothing in this wave has been compiled or run yet [V].** In this session the MANVI shell gate refuses three things:
  - `mac_heavy.sh`, because it lives outside the repo;
  - `git` and `rustfmt` in sibling worktrees;
  - Python run on worktree paths.

  The owner chose "hand me the commands": every build, test, fixture generation and commit goes to the owner as one script, `backups/a6-tools/owner-batch.sh`.
- **Board.** Both cards were moved to **in_progress**:
  - ft-b5e64557, ojas-train, rev 9;
  - ft-a1a4ba4d, the quantised arm, rev 5.

  Neither moves to review until the batch is green.
- **Critical path.** The quantised path to a run, per Fable's 2026-10-17 order, is:
  1. LoRA rows.
  2. 0.8B bf16 smoke.
  3. Amendment 1.
  4. Reduced 2B bf16 run.
  5. Quantised 0.8B smoke.
  6. Reduced quantised 2B run.

  The first quantised runs train the bf16 LoRA path over a **fake-quant** q8 g64 base file, so no quantised kernel is on the path. The smoke is blocked on ojas **ft-b5e64557**, which has no trainer for the Tape, and on Lappi **ft-3b8126cd**, which a live Lappi orchestrator is building now.

## 1. Lanes (all uncommitted in their worktrees, awaiting the batch)

| Lane | Worktree / branch | Card, criteria | What it built | Opus review | Adversarial |
|---|---|---|---|---|---|
| 1 | `../ojas-a6-train` `gp/a6-train` | ft-b5e64557, phase A-prep | New `ojas-train` crate: edition 2024 to match Lappi's moving code; deps ojas-core and ojas-io only. Hand-written FIPS 180-4 `sha256` (no sha2 in the workspace [V rg -uu]) with the NIST vectors, split and length sweeps, and a mod-2^64 length test. 64 MiB throughput bench (`bench/sha256_throughput.rs`). No file sits at a path phase A will import. | approved; reverted the attacker's edition change, see §4 | yes: 1 non-defect reversed, 1 test added |
| 2 | `../ojas-a6-adapters` `gp/a6-adapters` | ft-b5e64557 crit 14, 16, 18, 19 (seeds) | `qwen35::bank::AdapterBank`: per-sequence gradient sum, Overwrite then Add, shape-checked. `qwen35::optim::adamw_all_or_nothing`: pass 1 checks every entry, pass 2 runs `Backend::adamw_step` unchanged. `qwen35::adapters`: checkpoint by PEFT names with moments, step, RNG, digest and base sha256 plus the quantiser record; atomic via `replace_dir_with_if`; refuses non-zero dropout. Layout v1 is documented in `docs/adapter-checkpoint.md`. `check_save_target` was generalised to `check_target_dir`, with the message unchanged [V diff]. | returned once: the hash now covers metadata and the moments file is bound to its adapters file; approved | yes: 6 defects fixed, 11 tests |
| 3 | `../ojas-a6-valuece` `gp/a6-valuece` | ft-b5e64557 crit 17 | `Tape::linear_cross_entropy_value`: fused CE with `CeGrads::NONE`, no node, no `[N,d]`/`[V,d]` gradient. `fused_ce` now delegates to `fused_ce_asking`, keeping every contract check [V diff]. `RowTargets` and its scale are Lappi's type, so Lappi's scale-0 dispatch calls this method [V rg -uu]. | approved | yes: no defect, 2 tests |
| 4 | `../ojas-a6-fq8` `gp/a6-fq8` | ft-a1a4ba4d (Fable) crit 8 | `ojas-oracle/python/qwen35_fakequant_base.py`: mlx `dequantize(quantize(W, 64, 8))` on exactly the `model.layers.<n>.<suffix>.weight` wrapped linears; bf16 scale and bias; manifest with the mlx version, the call and sha256. `qwen35_lora_fixture.py --base`. `qwen35_common` gains a fixture parameter. `qwen35_lora_fq8{,_metal}.rs` run the (a')(b')(c') checks unchanged. Tiny-fixture in-features are 64, 128, 256 and 512, all multiples of 64 [V header]. | approved | yes: 3 defects fixed (silent NaN or zero-scale groups, suffix over-match, `--src == --out`) |

Phase A proper did **not** start, and should not yet. Lappi lanes wave-l1, which edits `qd-train/src/lib.rs`, `objective.rs` and a new `d17.rs`, and wave-l2, a new `lora_init.rs` that imports `run_control`, are live in Lappi worktrees [V by Fable]. ft-b5e64557 criterion 2 therefore binds.

## 1b. Wave 2 (owner: "merge them and next wave")

There was nothing to merge: every lane branch still sat at `04277f8`, uncommitted and uncompiled. The owner asked for commits before any branch switch, so:
- `backups/a6-tools/owner-commit-q1.sh` commits the four Q1 lanes on their own branches.
- `backups/a6-tools/owner-merge.sh` does the rest:
  1. commits all six lanes;
  2. runs the full batch;
  3. stops on any red step or any fixture byte drift;
  4. commits the rustfmt changes and the generated fixtures;
  5. merges each branch `--no-ff` into **local** main, refusing any lane whose files overlap main's dirty peer files;
  6. regenerates and checks the site;
  7. tests and clippies the merged main in its own `target-a6-main`.

  Nothing is pushed.

| Lane | Worktree / branch | Card | Built | Opus review | Adversarial |
|---|---|---|---|---|---|
| B | `../ojas-a6-docsync` `gp/a6-docsync` | ft-531216ed crit 4, ft-3cc9ed3f crit 8 | `ojas-qwen35/README.md` now matches the measured Tape rows. `docs/dtype-policy.md` states Metal's per-call bf16 widening (`backend.rs:364-369`). | approved; numbers spot-checked against the rows files [V] | n/a (docs only) |
| C | `../ojas-a6-stepcost` `gp/a6-stepcost` | ft-2bbced18 crit 1–4 | Audit: crit 3 already MET on main. `MetalBackend::freeze_checked`: scan once, skip the per-call scans, refuse writes. Bench arms `tape-lora-checked`/`tape-lora-fulldw` plus phase timing. Docs §3.1. | approved. Recorded: opt-in and Metal-only; the generic hook goes to phase B (ft-fe7dcee6 forwarding risk) | yes: 3 defects fixed (`to_f32` copies wrongly frozen, repeat freeze growing the mark list, fulldw silent fallback) |
| A | stacks on `gp/a6-train` | Fable item 9(i) | Waits on the owner's Q1 commit, per the owner. | — | — |

Lane 1's `bench/README.md` section was removed (the bench file documents itself), so the merge cannot collide with the peer's dirty `bench/README.md` [V: diff equals HEAD].

Criteria 1 and 4 need GPU rows: `backups/a6-tools/owner-bench-stepcost.sh` (interleaved, min-of-5, 2B, T = 2,048 and 9,638; about 3–5 h). It does not gate the merge, and ft-2bbced18 stays in_progress until the rows exist.

## 1c. Wave 3 (owner: "Next wave"; Fable picked the lanes)

Fable ruled to build narrowly: lanes that touch none of the six pending lanes' files, so merge order cannot conflict, and nothing stacked on uncommitted work.

| Lane | Worktree / branch | Card | Built | Opus review | Adversarial |
|---|---|---|---|---|---|
| L1 | `../ojas-a6-preflight` `gp/a6-preflight` | ft-b5e64557 crit 10 | `ojas_model::preflight`: `peak(T) = c0 + c1·T` per snapshot, fitted from `bench/results/2026-10-17-lora-step/rows.jsonl`. A typed `PreflightRefusal` names the shape. No rows means refuse. Extrapolation is flagged. `dry_run` at the widest bucket under a child Budget. Wiring before step 0 is phase A's. | approved | 1 defect fixed: with three or more points the fit could under-estimate a measured peak (a pass that then OOMs). `cover_measured` raises c0 until every measured row is covered. 7 tests. |
| L2 | `../ojas-a6-adamw100` `gp/a6-adamw100` | ft-b5e64557 crit 7 | `adamw_oracle.py`: torch fp32 and fp64 runs of the recipe, using closed-form gradients that Rust reproduces bit for bit. The bound is 4× the measured fp32/fp64 gap. CPU Exact, CPU Fast and Metal at steps 1/10/50/100. A sensitivity test shows a wrong decay or step count misses the bound. | approved | n/a (tests and oracle only) |
| L3 | `../ojas-a6-merge` `gp/a6-merge` | ft-58e316a9 (ojas side) | `merged_weight_bf16`: an f32 sum with one RNE round, no mutation. `Qwen35Lora::merge_bf16` streams (HF name, bf16) pairs through `fuse_b`. `merge`'s helpers are shared, with behaviour unchanged. Over a fake-quant base this exports dequant(Q(W)) automatically. | approved. Touches `params.rs`, so the parked lane A branches after the merge instead of stacking. | no defect; a `-0.0` doc claim corrected; 2 tests |

**Fixed in a pending lane (Fable caught it):** both local bases are named `model.safetensors-00001-of-00001.safetensors`, and layout v1 refused every `-N-of-M` name. One-of-one is now accepted (save and resume test, doc updated); every other `-N-of-M` is still refused.

**Fixed in the owner scripts:**
- torch lives in the user site, which `python -I` drops, so the torch steps use `-P`;
- the post-merge clippy step had `--manifest-path` after `--`;
- wave-3 commits now come before the batch, per the owner's commit-first rule.

All four scripts pass `bash -n`.

**Not picked (Fable):**
- an ojas reader for Lappi shards, which would bypass Lappi's rule-3 door; the mechanics check uses a Lappi dump;
- 0.8B-Base on the Tape, which already ran at both widths;
- the D17 head as library code, which waits for phase A, and Lappi's d17 lane is live.

**For the owner from Fable:** two Lappi agent worktrees both modify `upstream.lock.json`, so the pin bump may be in flight twice. Check before either commits.

## 1d. Wave 4 (no code; Fable)

- **Runbook:** `docs/lappi-lora-mac-runbook.md` on `gp/a6-runbook` (uncommitted).
  - **Runnable today:** `owner-merge.sh` (load-gated) and `owner-bench-stepcost.sh`.
  - **Done:** the LoRA rows; Lappi amendments 1 and 2 (N=127).
  - **Blocked:** phase A (Lappi freeze sha); steps 2, 4, 5 and 6 (no Tape trainer, torch driver or converter; step 5 also needs the Lappi gate tool and owner item 5).
- **New blocker [V]:** Lappi amendment 3 (`campaign/lora-2b-preregistered.json:260-261`) withdraws the v5r shards, because they contain MMLU test/dev and CLINC test items. LoRA-2B trains and is scored only on a shard set rebuilt under the v6 data rules (Lappi's work). N is re-derived if the width distribution changes. The ft-da2c4ab1 "v5r rows" mechanics check is superseded.
- **Phase-A import script:** `backups/a6-tools/owner-phase-a-import.sh <lappi-freeze-sha> [--execute]`. It runs git-filter-repo on a fresh Lappi clone and imports into `ojas-train/src/lappi/` and `tests/lappi/` on `gp/a6-phase-a-import`, never main. The gates are the sha, ancestry, `gp/a6-train` merged and a clean tree. It defaults to a dry run. Unsure items, flagged in its output: `pyjson_oracle.rs` and `tests/common/` import modules that stay in Lappi. Untested.
- **Cross-lane check [V by read]:**
  - no file is touched by two lanes;
  - no lane touches main's dirty files (each lane's copy equals HEAD);
  - only `gp/a6-train` edits the root `Cargo.toml` and `Cargo.lock`;
  - no API mismatch: freeze_checked never freezes adapter buffers, the preflight calibration skips the new arms' rows, and the merge refactor keeps the public API;
  - any merge order is clean.

  Notes: two fixtures exist only after the batch, and `QWEN35_MEM_PHASES` rows must not be mixed into timing comparisons. `gp/a6-runbook` was added to `owner-merge.sh`.
- **Environment passthrough:** verified. `mac_heavy.sh` runs `"$@" &` as a child (line 249), so `env CARGO_TARGET_DIR=…` reaches cargo.

## 2. Deferred or blocked

| Item | Why |
|---|---|
| ft-b5e64557 phase A (moving Lappi's qd-train with history) | Criterion 2: live Lappi lanes hold `qd-train`. It waits for wave-l1/l2 to land and for Lappi to freeze the moving set. When it runs, importing into an existing `ojas-train/` needs filter-repo plus `merge --allow-unrelated-histories`. |
| `groups.rs` move from ojas-qwen35 | `groups.rs` imports `Qwen35Error` and `names::TensorSpec` [V]. Orchestrator ruling: move it in phase A, where Lappi's recipe becomes a GroupSpec builder, instead of inventing a `ParamInfo` trait now. |
| The Tape `StepProvider`, the torch parity test for 3 unequal rows, and the entries helper from `Qwen35Lora` | Need the trait from phase A. |
| Pre-smoke mechanics check (ft-da2c4ab1, last criterion) | ojas has no reader for Lappi shards; it needs a Lappi row dump and a GPU window. |
| Duplicate bundle cards ft-29f7e4b6, ft-dc4bfe22, ft-e15ee76a, ft-ee2a3753, ft-a95d2145 | Fable ruled: log them as duplicates and move them to backlog. `gitpulse_complete_task` accepts only in_progress, review or done, and relabelling needs a whole-card overwrite, so they are **unchanged** on the board. Canonical cards per Fable: ft-29f7e4b6 → ft-562b0add, ft-15ebd6b1, ft-bd6373d2, ft-3902a4b3. ft-dc4bfe22 → the vision chain. ft-e15ee76a → ft-2c183b71, ft-69ef4ce1. ft-ee2a3753 → ft-b301b6d6, ft-54e8ce27, ft-2bbced18. ft-a95d2145 → ft-b90b30ae, ft-301ddd4c. |
| ft-2bbced18 | Fable reads criteria 7 and 9 as met on main (`0f13b37`, rows.jsonl; N=127 in Lappi amendment 2). The card is in_progress from an earlier lane and its other criteria were not re-checked, so this session did not move it. |

## 3. Benchmark deltas

None measured yet. The sha256 throughput row is new (there is no before), and it is the batch's last step. No existing hot path was changed, apart from one added delegation call in `Tape::fused_ce` [I: negligible; not benched].

## 4. Decisions

**Fable (2026-10-17):**
- The four lanes above.
- Phase A waits.
- ft-2bbced18 criteria 7 and 9 are met on main.
- The duplicates are logged and not deleted.
- Coordination: tessl-65 and the Lappi orchestrator, via the card log and the owner.

**Orchestrator rulings:**
- **Edition.** ojas-train is edition 2024 although the workspace is 2021. Lappi's qd-train is 2024 (`Lappi-decision/Cargo.toml:13` [V]), and that code moves in. The attacker's switch to 2021 was reverted.
- **groups.rs** is deferred to phase A.
- **Adapter layout v1** was changed in place before publication: the hash covers metadata, and the moments file is bound to its adapters file.

### Fable's ruling on the open owner decisions (2026-10-17; owner: "ask fable for decisions")

1. **sha2: no.** The hand-written SHA-256 stays. Phase A replaces Lappi's `sha2` call sites with it. The batch's bench decides: if min-of-7 is at least 150 MB/s it is closed; below that, the owner is asked for sha2, with the number attached.
2. **Lappi pin 5b67cdd → 04277f8: commit it.** Lappi's delta report already records the owner's "merge all". The range touches none of the five imported crates [V by Fable]. Later bumps happen on need: next to the a6 merge sha (for `linear_cross_entropy_value`), then to the phase-A sha. Each needs a clean ojas tree, a delta report and `make pin` PASS. **Executes:** a Lappi session.
3. **Lappi owner items:**
   - **5:** pre-approved **by criteria**. The q8 sibling pre-registration must contain exactly the final ruling's §6 item-5 list and be committed before any quantised result. Any deviation goes back to the owner.
   - **6:** the ~1 h quantised 0.8B smoke is approved. The ≤6 h reduced quantised 2B run is approved once the bf16 reduced row exists. The owner launches every heavy run, and the launch is the consent.
   - **8:** ojas-a6 first makes the text fixes on ft-531216ed (criterion 4 and the README) and ft-3cc9ed3f (criterion 8 docs) in the doc-sync commit, then logs "ready for verify". Manvi verify stays the owner's.
   - **9:** peft stays "not run".
   - **Recording:** "Owner, 2026-10-17: 'ask fable for decisions'. Fable ruled: …". **Executes:** a Lappi session.
4. **Merges and push.**
   - **Merges:** ojas-a6 merges the four lanes into local main once the batch is green and the owner has committed the worktrees.
   - **Push:** this **still needs a one-line owner yes**, because the original instruction reserved it. It publishes **146** commits, not 66 [V by Fable]. Fable recommends yes, because Lappi's lock url serves nothing until the push.
5. **Duplicates: archive them, do not merge.** Each one maps to 2–5 canonical cards, so `merge_tasks` would misfile most of its criteria. Archiving is the owner's one click (ids and the canonical map are in §2).
6. **ft-2bbced18:** log and move to ready. The board refuses →ready, so it stays in_progress with the log (done) [V ok:updated].
7. **Lock holder pid 30722: leave it.** It is a mac_heavy wrapper at 0 % CPU, waiting for load ≤ 12. Load is about 90 from other builds, and the wrapper will exit at MAX_WAIT. The batch will queue behind it. Run the batch when the machine is quiet, or raise `MAC_HEAVY_MAX_LOAD` for the CPU-only steps (the owner's choice).
8. **Adapter layout v1 binds, after two fixes:**
   - `base_sha256` names its file, and v1 refuses a sharded base;
   - fixed names for the D17 and span-head entries.

   FNV-1a is acceptable for v1. ojas-model → ojas-train is the dependency direction.

   **Done on `gp/a6-adapters` (uncommitted, unverified):**
   - `BaseIdentity.file_name` is stored as `ojas.adapter.base_file`.
   - Shard names and `.index.json` are refused with `ShardedBase`, and a different file name on resume gives `BaseFileMismatch`.
   - The host entries use Lappi's existing names (`qd-export/src/layout.rs`, `span_head.rs`): `d17.delta`, `span_head.{start_proj,end_proj}.weight`, `span_head.abstain_{start,end}`. Any other name is refused with `UnknownEntryName`.
   - New tests: `a_sharded_base_is_refused`, `an_unknown_host_entry_name_is_refused`, `resume_refuses_a_different_base_file_name`.
9. **Next wave after green:**
   - move `sha256` to ojas-io and pin base sha256 plus the quantiser record at `load_hf`;
   - the doc-only sync, including item 8's text fixes, and mark ft-da2c4ab1 criterion 20's "v5r rows" superseded (v5r refused; v6 rebuild);
   - ft-2bbced18 criteria 1–3 against the existing rows.

## 5. Coordination

- **Quiet slot for verification:** 2026-10-17, **21:15–23:15 local**, proposed to tessl-d5. At 20:16 the load was ~140 [V ps], from:
  - tessl-eg2 `cargo test --release --no-run` (pid 72050), about 15 rustc at once, outside mac_heavy;
  - a tessl agent worktree's `cargo test --lib`;
  - scholarlm `go test -race -count=2` (pid 38554);
  - GitPulse agent `cargo test` (pid 41252);
  - a Lappi agent's Python job (pid 50715).

  The heavy lock is held by the ojas job "gdn-mem" (pid 97046), which is only waiting on load (its child is `sleep 20`) and times out about 21:08. A one-shot check fires at 21:13 (session cron 6dfe6ff5) and notifies the owner whether the slot is open.
  - **Owner away overnight (2026-10-17):** the owner authorised ojas-a6 to run `owner-verify-main.sh` itself and fix failures. Jobs in this session:
    - session cron 85288622 at 21:13 tells tessl-d5 and launches the run via `backups/a6-tools/launch-verify.sh 80` (detached with nohup; pid in `logs/verify.pid`; `logs/VERIFY-DONE` sentinel at the end);
    - recurring cron 9e5e51bf at :07, :27 and :47 watches the run, and on completion releases tessl-d5, fixes failures (lane, adversarial pass and Opus review, one commit per concern, unverified label), re-runs the failing targets, and updates this report and the cards.
    - Never: push, kill processes this session didn't start, or touch other repos.
  - **21:13 slot start [V]:** load 79.8, and the verify runner is launched (pid **85224**, `MAC_HEAVY_MAX_LOAD=80`). Its first step is queued for the heavy lock behind the Lappi agent job "w4d-mutate-calib" (pid 98329, since 21:09); `gdn-mem` is gone. Still loading the machine: tessl-eg2's `cargo test --release --test gdn_state` (pid 68766, unreachable session), and tessl-d5's `msl_emu/check_qwen35.py` (pid 83921, past its 21:10 cutoff; tessl-d5 asked to stop it). Caveat: each step's alarm (1–1.5 h) also counts lock wait, so a long Lappi hold can time steps out unrun; the watcher re-launches if so.
  - **Verify run, in progress (22:13) [V logs]:**

    | Step | Result |
    |---|---|
    | build | exit 0; all touched crates and tests compile, one `unused import: Backend` warning that clippy will flag |
    | train | 134 passed, 0 failed |
    | autograd | 33 passed, 0 failed |
    | **model** | 201 passed, **4 failed** |
    | **metal** | 30 passed, **2 failed** |

    The four model failures:
    - `preflight::a_dry_run_refusal_with_a_huge_request_saturates`: `Budget::try_reserve` overflow is `OutOfRange`, not mapped to a refusal;
    - `qwen35_adapters_adversarial::awkward_entry_names_round_trip`: a stale test, from before the fixed host-name rule;
    - `qwen35_lora_fq8::the_fq8_base_changes…` and `qwen35_lora_merge_export::the_budget_peak…`: an F32-only read of a bf16 base.

    The two metal failures, both in `frozen_checked`: `scale_grad`, and a write to a deliberately frozen copy, are not refused.

    Also: `qwen35_optim`, `qwen35_readout` and `save_adversarial` never ran, because their test binaries vanished mid-run (purged, as in the memory note about target-dir housekeeping), so they are environmental. Fix lanes `gp/a6-fixmodel` and `gp/a6-fixmetal` (worktrees from main) started; nothing is committed to main while the run is live.
  - **23:33 update [V logs]:**

    | Step | Result |
    |---|---|
    | metallib | 81 passed, 0 failed |
    | qwen35 | 4 passed, 0 failed |
    | lorarow (GPU LoRA step) | 1 passed, 0 failed |
    | full | 614 passed, 4 failed (the same four model failures; this time `qwen35_optim`, `qwen35_readout` and `save_adversarial` ran and passed, so their earlier absence was environmental) |
    | clippy | waited more than 50 min for the heavy lock, behind back-to-back Lappi agent jobs (mac_heavy is not FIFO); expected to time out unrun |

    The slot was released to tessl-d5 at 23:33, because the GPU steps were done and the contention is Lappi's, not tessl's.

    Fix lanes, ready and approved:
    - `gp/a6-fixmodel`: preflight overflow mapped to a refusal; three stale tests updated to the intended behaviour.
    - `gp/a6-fixmetal`: Metal had no `scale_grad`, so the trait default replaced the gradient without passing the frozen check. The fix adds an in-place override and names the op in `Cmd::Scale`. Adversarial pass: no defect, 2 cosmetic fixes, 4 tests.
  - **23:40, fixes on main (unverified until the re-run):**

    | Commit | Fix |
    |---|---|
    | `43d5e82` | preflight overflow mapped to a refusal |
    | `0a17930` | three stale tests updated to the intended behaviour |
    | `aaed8b3` | Metal `scale_grad` scales in place |
    | `111097d` | unused `Backend` import removed |

    They were committed while the first run's clippy step was still waiting for the lock (nothing compiled), so its clippy, if it runs, lints the fixed tree. The re-run is `backups/a6-tools/owner-verify-rerun.sh`, launched by `launch-rerun.sh`. It covers the model tests including the three that vanished, the Metal fix targets plus `clip_multi`, clippy, the Linux check and the sha bench, with a keep-alive touch on `target-a6-main`. Watcher cron v2 (8c568914) launches the re-run when the first run ends, handles its result, and stops after 3 fix cycles.
  - **tessl:** the fast-forward target is now **527de20** (ca3efa8 plus a Cargo `exclude` line). Next, tessl-d5 will announce an eg2/integrate fast-forward (96 commits) with its file list first, and ojas will check the modules it builds by path. **Checked: no ojas effect [V read].** The only hunk in an ojas-used module is in `qwen35_train.rs`: its working-set bound now reads `dispatch_batching()` instead of `async_encode_enabled()`. ojas-qwen35 runs `qwen35_train` on its own `GpuRuntime::new()` (`step.rs:399`) with no async encode, no `async_scope` and no serialisation, so both evaluate to `false` there. ojas-metal enables async encode (`device.rs:523`, `gpu.rs:326`) but never calls `qwen35_train`. ojas told tessl-d5: no hold.
  - **00:14, first run finished [V logs]:** clippy exit 142 and linux exit 142, both alarms while waiting for the heavy lock behind Lappi agents, so neither ran. **sha-bench** exit 0: 64 MiB, 7 rounds on a loaded machine, one-shot min 195.98 ms (**326.6 MiB/s**, median 246.1), streamed 64 KiB min 193.20 ms (**331.3 MiB/s**, median 282.6). That clears Fable's 150 MB/s bar, so **no `sha2` dependency** (ruling item 1 closed). The re-run was launched at 00:14 (pid **86296**).
  - **Re-run, 01:15 [V logs]:** model step **137 passed, 0 failed** (10 binaries). That includes the four fixed tests and the three targets that vanished in the first run. It ran against tessl bf0b7b9. The Metal step was waiting for a free machine-wide cargo slot (`~/.cache/cargo-slots`, 2 slots, a throttle alongside mac_heavy).
  - **tessl main landed 527de20 during the re-run (~01:15).** All its gates are green, including device tests (`sq_norm` 4, `qwen35_adamw` 23, `qwen35_train` 28). Verified on disk: `sq_norm.rs` is present, and the pinned `runtime.rs` fragments are at lines 280, 284 and 294. **The remaining re-run steps (Metal, clippy, Linux) build against 527de20**, so a red Metal step must be checked against both sources. tessl-d5 was asked to hold the eg2 fast-forward until "rerun done".
  - **Re-run, ~03:30 [V logs]:**

    | Step | Result |
    |---|---|
    | model | 137 passed, 0 failed (on tessl bf0b7b9) |
    | **metal** | **106 passed, 0 failed** (`frozen_checked`, `frozen_adversarial`, `clip_multi`, lib; on tessl 527de20) |
    | clippy | still waiting: cargo slots held by GitPulse tests, then the heavy lock held by Lappi `w5d-ab`, load ~103; likely to time out unrun |

    **The eg2 fast-forward (527de20 → a37a0c2) was released early**, because ojas's tessl-sensitive steps were done, eg2 has no effect on modules ojas builds, and the remaining steps are clippy, a Linux check of ojas-train and the sha bench.
  - **Re-run 1 finished 05:04 [V `logs/rerun1/`]:**

    | Step | Result |
    |---|---|
    | model | 137 passed, 0 failed |
    | metal | 106 passed, 0 failed |
    | clippy, linux, sha-bench | exit 142: alarms while waiting, not code failures |

    The waits were on the heavy lock (Lappi `w5d-ab`, held since ~02:42), the cargo slots (GitPulse tests) and load ~97. **Re-run 2** was launched at ~05:25 (pid **58832**) with only clippy and the Linux check, each with a 4 h alarm; sha-bench is not repeated (it passed in the first run at 331 MiB/s). If these time out again, they are documented as not run.
  - **Morning, 2026-10-18 (owner back: "commit and merge all changes, next wave, delete old and stale branches"):**
    - **`gp/clip-sq-norm` merged into local main as `73cfe02`.** tessl main a37a0c2 now carries `sq_norm` with the branch's API. The merge is textually clean (`git merge-tree`). Both `Cmd::Scale` constructions in the merged tree carry main's `op` field, and nothing refers to the removed `ojas_norm_multi`/`ojas_norm_finish` kernels [V git grep]. **Not compiled or tested yet**: clippy re-run 2 lints it, and `clip_multi` plus the Metal suites go in the next re-run.
    - **`dfd2419`** carries the staging worktree's doc-only commit a797c1d: the README and bench docs now quote the shipped `sq_norm` clip time (7.45 ms median, about 1.13× torch 2.15 nightly), plus the bench-plots section. The site was regenerated on current main, and `sitegen -check` passes.
    - **Branches:** `gp/clip-sq-norm` deleted (merged); worktrees `ojas-clipnorm` and `ojas-merge-clip` removed without `--force`. a797c1d stays reachable through `refs/backup/2026-10-17-pass2/detached/ojas-merge-clip`. **ojas now has one branch, `main`**, at `dfd2419`, 169 commits ahead of origin; nothing pushed.
    - **tessl ft-c59ff950 done (in review):** `SqNormPlan::parts_words(&self) -> usize` (= 2 × pieces), now used by tessl's encode check and its allocation. It lands with tessl's next integration (wave 15). **ojas follow-up when it lands:** in ojas-metal's clip path (merged at 73cfe02), replace `self.fresh(2 * plan.pieces())` with `plan.parts_words()`, so the rule has one owner.
    - **Next wave trigger:** Lappi main fecfd5d now has `qd-train/src/{d17,lora_init}.rs`, so the lanes that blocked phase A have landed. Fable was asked whether the import can start without a Lappi freeze acknowledgement. **Fable: GO, freeze sha `fecfd5d22003a439c066df1289e1a0ef0a751df8`.**
      - wave-l1 and wave-l2 merged via Lappi's wave-integration branch (98c24d8, 3b5bc99, fixes through ac7adb0) into main at fecfd5d.
      - No Lappi worktree holds edits to the eight moving files. Six older worktrees carry a stale pre-L2 `recipe.rs`, inferred as not edited; the only live change is a non-moving `ledger.rs`. The freeze acknowledgement is waived, and Lappi is told afterwards.
      - Two script gates fail as written and were fixed without editing the script: branch `gp/a6-train` was recreated at ffc6f2e (an ancestor of main) for gate 3, and a clean detached worktree `../ojas-phase-a` was created for gate 4, because main's checkout has untracked files.
      - New seam for the rewire: L2 put `crate::d17::D17_ENTRY` and `span_head::PARAMETER_NAMES` into the moving `recipe.rs`.
      - The moving files use `sha2` and `serde_json`, so the rewire must replace them with `ojas_io::sha256` and `ojas_io::JsonValue`, not add the dependencies.
      - **The owner runs the import** (the gate blocks git and clone on Lappi from this session).
  - **Re-run 2 ended 07:34, nothing ran (my bug) [V `logs/rerun2/`]:** clippy and linux both failed with `failed to create directory target-a6-main/debug: Not a directory`.
    - Root cause: after the disk-full cleanup deleted `target-a6-main`, the re-run's keep-alive (`touch "$M/target-a6-main"` every 5 min) recreated that path as a 0-byte **file** at 07:32.
    - Fixed in `owner-verify-rerun.sh`: the keep-alive touches only an existing directory, and each step `mkdir -p`s the target dir and refuses if the path is a non-directory. The stray file was removed.
    - **Re-run 3 launched ~12:53 (pid 62874):** the Metal suites (`clip_multi`, `frozen_checked`, `frozen_adversarial`, `linear_ce`, `bf16_operands`, lib) for the clip-sq-norm merge, then clippy and the Linux check, each with a 4 h alarm.
  - **Re-run 3, done 08:02 [V `logs/rerun3/`]:**

    | Step | Result |
    |---|---|
    | **metal** | **124 passed, 0 failed** (`clip_multi`, `frozen_checked`, `frozen_adversarial`, `linear_ce`, `bf16_operands`, lib), so **the clip-sq-norm merge passes on tessl a37a0c2** |
    | **linux** | exit 0: ojas-train builds for aarch64-unknown-linux-gnu |
    | clippy | 3 errors, all the rust 1.99 lint `chunks_exact_to_as_chunks` in `ojas-io/src/sha256.rs` (:48, :111, :135) |

    **Fix cycle 1, `d0fbe97`:** `as_chunks` / `as_chunks_mut` in place of `chunks_exact`, with the same bytes and digest, pinned by the NIST vectors. It is a mechanical lint fix on already-reviewed code, so it was made directly by the orchestrator, with no Sonnet lane or adversarial pass (logged as a deliberate shortcut). **Re-run 4 launched (pid 46515):** the ojas-io and ojas-train tests, then clippy.
  - **Re-run 4, done 08:14 [V `logs/rerun4/`]:** io **134 passed, 0 failed** (the sha256 rewrite is behaviour-neutral). Clippy then got past ojas-io and stopped in ojas-model's library with **21 errors, all in a6 code**:
    - 18 × `result_large_err`, one root cause: `AdapterError::QuantizerMismatch` held two inline `Option<QuantizerRecord>`, making every `Result<_, AdapterError>` at least 144 bytes;
    - 2 × `manual_is_multiple_of`;
    - 1 × `chunks_exact_to_as_chunks`.

    **Fix cycle 2, `0e95fe7`:** the records are `Option<Box<QuantizerRecord>>`; the tests' `Some(_)`/`None` matches and the Display are unchanged; plus `is_multiple_of` and `as_chunks::<2>`. Made directly (mechanical lint fixes). **Re-run 5 launched (pid 34788):** the adapter and load-pin tests, then clippy, which may reach further crates and find more. **Fix-cycle budget: 2 of 3 used.**
  - **Re-run 5, done 08:35 [V `logs/rerun5/`]:** adapters, adversarial adapters and load-pinned **49 passed, 0 failed** (the boxed error is behaviour-neutral); the ojas-model library is now clippy-clean. One lint was left, `manual_is_multiple_of` at `ojas-model/tests/qwen35_optim.rs:32`. **Fix cycle 3 (the last), `81a81ca`.** **Re-run 6 launched (pid 42826):** clippy with `--keep-going`, so any remaining errors across all targets are reported in one pass and documented instead of fixed target by target.
  - **Re-run 6, done 08:54 [V `logs/rerun6/`]:** `clippy --keep-going` over all six crates and every target found **exactly 2 errors**: `single_element_loop` in `tests/qwen35_bank.rs:188` (left from the adversarial narrowing to `[Add]`) and `useless_vec` in `tests/qwen35_lora_merge_export.rs:461`. The 3-cycle cap was for unattended work; with the owner awake, both were fixed in **`94aa3af`** (no assertion changed). **Re-run 7 launched (pid 4082):** the two touched test files, then a final `clippy --keep-going`.
  - **Disk-full incident (~05:45–06:20, 2026-10-18):** ENOSPC on the data volume. Every Claude session's Bash failed (it could not create its output file in /private/tmp), ojas's included, so nothing could be freed from here. ojas sent the owner a desktop notification and asked tessl-d5, which was equally stuck, to stop writers; tessl-d5 stopped both its agents. By 06:27, 140 GiB was free (85% used) and `target-a6-main` had been emptied (0 B) by someone outside this session. Re-run 2 (pid 58832) survived; its clippy step waits for a cargo slot and will rebuild from scratch.
  - **Owner: "work on all not-ready gaps" (2026-10-18).**
    - **ojas gap 1 (phase A rewire):** compile and test running (relaunched with `MAC_HEAVY_MAX_LOAD=80`; the first launch inherited mac_heavy's default 12 by mistake and was stopped by its PIDs).
    - **ojas gap 2 (phase B, the Tape StepProvider):** branch `gp/a6-phase-b` in worktree `../ojas-phase-b`, from the phase-A tip, so it doesn't disturb the phase-A compile. A design is in progress (Plan agent); commits go through `phase-b-commit.sh`.
    - **Phase-B design (Plan agent):**
      - `TapeStepProvider<B>` in `ojas-model/src/qwen35/step_provider.rs`, with ojas-model depending on ojas-train (no cycle, no new crates);
      - one sequence per tape;
      - `ParamSpec.trainable` plus a storage dtype, defaulting so the tessl provider is unchanged;
      - `save_state_with(&CheckpointContext)`;
      - `DeviceHooks` for Metal's `freeze_checked` and `with_budget`;
      - 3 lanes: L1 ojas-train contract (after phase A compiles), L2 provider, L3 Metal tests.
    - **L2a committed `1160ada` on `gp/a6-phase-b`:** `AdapterBank::take_grads`/`sq_norm`, `DeviceHooks`, and the torch 3-row accumulate fixture generator (not yet generated; it runs inside phase B's verify script).
    - **Fable ruling (a), D17 and span-head ownership:** the loop's objective owns them (`trainer.json` plus `host.f32`, per-entry AdamW counts), and the provider checkpoint holds LoRA only.
      - Layout v1 needs no byte change; the doc is amended on phase B (`1132719`), and the Lappi work order L4 is updated (two-part `ckpt-<step>/`).
      - Follow-ups: the loop passes `rng_state` and digest via `save_state_with`, and the unused `Residence::Host` path in `bank.rs` is deleted in phase B.
      - Card criteria 4, 9, 14, 16 and 19 to reword (log only; the board can't edit criterion text).
    - **Phase B code-complete (unverified) on `gp/a6-phase-b`:**

      | Commit | Content |
      |---|---|
      | `1160ada` | L2a: bank groundwork, `DeviceHooks`, torch fixture generator |
      | `1132719` | doc: layout v1 holds LoRA only |
      | `72305ac` | L1: `ParamSpec.trainable`/storage, trainable-only views, `CheckpointContext`, five tests |
      | `edad1eb` | L2: `TapeStepProvider`, 10 CPU tests; unused `Residence::Host` deleted |
      | `c2047d7` | fixture generator aligned with the test's key layout |
      | `2406c28` | adversarial fixes: a refused Overwrite clears the bank; a failed preflight closes the gate (+4 tests) |
      | `c808046` | L3: Metal tests vs CPU, 3-row, all-or-nothing, kill-9 at 2B (bounded), `tape-provider` memory arm |

      Criterion 5 (vs the tessl provider) waits for Lappi's `qd-train-metal` after the pin bump.
    - **One combined verify run** for phase A plus phase B (`phase-b-verify.sh`: the accum fixture, check, train/model CPU tests, Metal provider tests, clippy; `MAC_HEAVY_MAX_LOAD=80`). The separate phase-A run was stopped by its PIDs to halve lock waits. The watcher cron repoints to it; up to 3 fix cycles.
    - **Lappi gaps 3–6:** `backups/a6-tools/LAPPI-WORK-ORDER.md` lists L1 v6 shards, L2 torch driver, L3 quant gate tool and fake-quant base files, L4 converter, L5 the qd-train move, and L6 decisions to record, each with the ojas contract it must meet. The owner hands it to a Lappi session; ojas cannot work in Lappi.
  - **Verify slot 13:15–16:15 (2026-10-18):** agreed with tessl-d5, whose 4 agents are stopped (one clippy overran by about 1 min). memguard is planned around, not bypassed: largest ojas process about 13–18 GB under a 28 GiB budget; pressure level 1. The owner was notified of the unreachable load sources (a GitPulse agent's rustc, a crates.io clang/rustc build, DevCouncil's devmap). **Phase A+B `cargo check`: exit 0 at 13:03**, so all the uncompiled code compiles.
  - **Re-run 8 GREEN (13:31) [V `logs/rerun8/`]:** main `54fa635` (`parts_words`) on tessl bf091a2. Metal **124/124**, ojas-qwen35 **14/14**, clippy clean. tessl-d5 was told "rerun done", but asked to hold wave 16 C/D until the slot is released (phase B's Metal step builds against tessl).
  - **Phase A+B run, partial (13:50):** check exit 0. **train: 176 passed, 4 failed**, all in `ojas-train/tests/frozen_entries.rs` (panics at :122/:123, one shared setup line). The moved Lappi oracles pass. **model: 94 passed, 1 failed**, `scale_zero_…allocates_no_vocab_gradient` (`qwen35_step_provider.rs:424`, the Budget-delta bound the lane flagged as tight); the **3-row accumulate test against the torch fixture passes**. Metal step running.
  - **Phase A+B run 1 complete [V `logs/phase-b-1/`]:**

    | Step | Result |
    |---|---|
    | check | 0 |
    | train | 176 passed, 4 failed |
    | model | 94 passed, 1 failed |
    | **Metal provider tests** | **5 passed, 0 failed** (Metal vs CPU, 3-row on Metal, all-or-nothing, …) |
    | clippy | 1 lint (`type_complexity`, `read_moments`) |

    **Fix cycle 1, `df4b547`:** the `frozen_entries` test config (cap within MAX_CAP_S; the 1-step test now 2 steps), the scale-0 test compares against scale 1, the `HostMoments` alias, plus the generated `qwen35_tiny_accum` fixture and `Cargo.lock`. The commit helpers no longer name the ignored `target` in `git add`, which had aborted the first commit attempt. **Run 2 launched ~14:20.**
  - **tessl 09efe84 → 1c4d03e (wave 19) announced; ojas said ok.** Bench binaries for the uncovered kernels, the `kernel_coverage` test and allowlist, and `hosted_tests.txt` only; no library, kernel, Cargo or numerics change, so no ojas check is needed. `kernel_coverage --check` now exits 0 (303 entry points, 301 dispatched, 2 allowlisted).
  - **tessl main landed 09efe84** (from a8fc965: tessl-a3's eg2 merge 5971091, whose ci:local passed 3425/0 with GPU Python and Go device tests under mac_heavy, plus a doc-only BRANCH_REPORT). Per tessl-a3, `runtime.rs`, `nn.rs`, `qwen35*`, `kernels/`, safetensors and Cargo are unchanged since a8fc965, so run 10 covers what ojas builds. The tip was verified on disk; the "unchanged" claim is tessl-a3's diff (the gate blocks git on the tessl repo from here).
  - **Run 10 done 2026-10-19 10:51: ojas 4123dfb VERIFIED on tessl a8fc965 (= wave 18's 49d905e + doc) [V `logs/rerun10/`]:** qwen35 CPU **42/42**; gpu_parity **9/10** (only the known missing 2B fixture, "No such file"); provider CPU **14/14**; **Metal 5/5**; **clippy clean**. Nothing moved from wave 18.
    - It was held about a day for disk space (free space swung between 1.5 and 33 GiB under other sessions' Go race tests, Lappi qd-prep scratch and tessl CI). The first launch was stopped by its own PIDs before building, and it was relaunched at 131 GiB free.
    - tessl-a3 (eg2 merge 5971091) and tessl-d5 were told; the gate cron is deleted.
  - **tessl a8fc965 → 9b95f5e (tessl-a3's eg2/integrate merge) announced by tessl-a3; ojas said ok.** It changes bench recordings, docs, bench/plot binaries and their tests, and doc comments only; the diff is empty for `runtime.rs`, `nn.rs`, `qwen35*`, `kernels/`, safetensors and Cargo. The run-10 gate builds the announced tip.
  - **tessl a1fccca → 49d905e (wave 18, on the scrubbed history) announced; ojas said ok (no run in flight).**
    - Contents: bench fixes; a new opt-in `nn::gemm_q4_mlx_tensorops` (g32/g64/g128; slots packed@0 sb@1 x@2 y@3 M@4 N@5 K@6), a candidate for the quantised arm's stage 2 (ft-e4279375, owner's yes pending); EmbedGemma2 config dedupe; the raised `train_step_bytes` bound.
    - No existing kernel's slots changed, and there are no numerics changes for existing paths.
    - The next ojas batch includes gpu_parity, the qwen35 CPU tests and the Tape provider tests on it.
  - **tessl heads-up (not landed): `train_step_bytes` / `step_forward` pre-flight raised** (it omitted GDN forward transients and CE staging; +25 to +405 MB at 2B, up to +3.8%). **No ojas impact [V rg]:** ojas never calls `train_step_bytes`, the Lappi path uses ojas's own Tape pre-flight (calibrated from measured rows), and tessl's 2B provider step already exceeds memguard's cap, so a refusal there just becomes typed. No hold.
  - **tessl history rewritten** by the eg2 session (filter-branch scrubbing an LGPL-derived `resample.rs`). Every tessl sha is renamed, with identical trees: **41d8866 (verified by run 9) = a083088**; the doc commit 4e99ca6 = **a1fccca** (current tip, verified on disk). ojas pins tessl only by path, and Lappi's lock does not pin tessl, so nothing breaks [V rg]. Provenance shas such as `42f1d52` in `bench/results/2026-10-17-lora-step/env.txt` stay historical and resolve through tessl's leftover refs; tessl-d5 was asked to keep them or publish a map. (2026-10-22: the map is
    published; see the note on tessl commit names at the top. `42f1d52` was never renamed.)
  - **Run 9 done 20:21: ojas 4123dfb VERIFIED on tessl 41d8866 [V `logs/rerun9/`]:** qwen35 CPU **42/42**; gpu_parity **9/10**; provider CPU **14/14**; **Metal provider 5/5**; **clippy clean**. Nothing moved numerically from tessl's `precise::rsqrt` change. tessl-d5 was told "verified", and the gate cron is deleted.
    - The one failure is **environmental**: `gpu_real_2b_single_step_matches_the_transformers_reference` cannot find `tessl/target/qwen35_train_ref/ids.npy`, which was wiped with tessl's `target/` in the disk cleanup.
    - Regenerating it is tessl's. tessl-d5's gate refuses `mac_heavy.sh`, and ojas declined to run it on tessl's behalf (that would be laundering a refused action, and it is cross-repo work).
    - **Owner action:** run from `/Users/bharath/Code/research/tessl`, under mac_heavy: `~/.venvs/ml-2026-10/bin/python tools/qwen35_ref/make_train_fixture.py 2b --tokens 128` (about 17 GB RAM, CPU). Then ojas re-runs that one test.
  - **tessl main landed 41d8866 (verified on disk).** The ojas check batch (gpu_parity, the qwen35 CPU tests and the Tape provider tests) is **held**: disk free is 27 GiB (182 at noon). A cold build (~10–15 GB, since the cache is 1.6 GB) risks a repeat of last night's ENOSPC. The owner was notified, and tessl-d5 was asked to hand over cleanup for its agents' targets. Runs at over ~50 GiB free.
  - **tessl 0819992 → 41d8866 announced; ojas said ok (no run in flight).**
    - Contents: wave 16 C (gemm bf16 casts bit-identical; `train_step_bytes` unchanged), D (one config reader; Qwen3.5 unchanged), wave 17 E (additive ABI 12→13; ojas pins no ABI) and F (`precise::rsqrt`).
    - Measured deltas: forward ≤4 ulp (qk-norm q ≤42 at eps 1e-12); qk-norm backward dproj ≤210 ulp (~2.5e-5 relative, inside gpu_parity's 1e-4); rms backward 0.
    - No ojas pub name changed.
    - ojas follow-up after it lands: a small batch of gpu_parity, the qwen35 CPU tests and the Tape provider tests against 41d8866.
  - **tessl heads-up (not landed):** RMSNorm and the qk-norm row kernel move to `precise::rsqrt` (forward ±2–4 ulp, backward 0 ulp). ojas-qwen35's exact-bit asserts are all self-consistency (same build), and its external checks are tolerances, so no re-pin is expected [V read]. gpu_parity is to be re-run after it lands.
  - **PHASE A IMPORTED (2026-10-18, owner: "You run it")** on `gp/a6-phase-a-import` @ **`62e0ec2`** (worktree `../ojas-phase-a`, from main 54fa635), at Lappi freeze `fecfd5d`.
    - Dry run: all gates passed. Then `--execute`.
    - `git log --follow ojas-train/src/lappi/trainer.rs` shows 13 Lappi commits [V].
    - The branch does not compile until the rewire; main is untouched.
    - `gp/a6-train` (the gate helper) is deleted. Branches are now `main` plus `gp/a6-phase-a-import`.
    - **Rewire lane started** (no new deps): `pub mod lappi`, path rewrite, `serde_json` → `ojas_io::JsonValue`, `sha2` → `ojas_io::sha256`, the recipe d17/span_head seam parameterised, the tests/common split, `isoformat_utc` inlined, test registration.
    - Commits on the branch go through `backups/a6-tools/phase-a-commit.sh`, which only commits in that worktree on that branch.
    - Lappi handoff updated with the freeze request.
  - **tessl main is 0819992**, a doc-only commit on bf091a2 (tessl-d5), so re-run 8 builds bf091a2's code. tessl-d5 now announces every main commit, and holds wave 16 C/D until ojas says "rerun done".
  - **tessl main landed bf091a2 (verified on disk). ojas `54fa635`:** the clip path's worker allocates the `sq_norm` parts buffer by `plan.parts_words()`. The caller-side budget reservation keeps its estimate (no plan exists yet), with its comment and `norm_chunks`'s doc now naming `parts_words`. **Re-run 8 launched (pid 44805)** against bf091a2: the Metal clip and frozen suites, ojas-qwen35 `cpu_groups`/`cpu_api` (tessl's `qwen35_train` changed), then clippy. Watched by cron (session job), which stops itself when the run ends.
  - **tessl fast-forward a37a0c2 → bf091a2 announced (2026-10-18 ~09:20); ojas said ok, since no re-run was in flight.**
    - Contents: wave 15 L1–L4, 5 newer eg2 commits, wave 16 A and B. All gates green.
    - Slot lists for the 33 Q4 and utils entries are IR-identical, and outputs are bit-identical.
    - **L4 is new:** ExactF32 fused-epilogue kernels (`matmul2d_tensorops_f32_exact_epi`, 128x64 and 64x64_sg4). `gemm_epilogue` with `GemmOperands::ExactF32` now fuses (bit-identical) instead of erroring; ojas has no caller yet (input for ft-e15ee76a).
    - Additive API: `SqNormPlan::parts_words()` and `Qwen35Model::train_step_ragged_into`.
    - The single-sequence `qwen35_train` step and `train_step_bytes` are pinned unchanged, and the three `runtime.rs` strings are byte-identical.
    - ojas follow-up after it lands: the `parts_words()` switch plus a Metal clip re-verify.
  - **tessl main is a37a0c2** (eg2 integration; verified on disk). Next from tessl: wave 15 L1–L3 (slots unchanged), then **L4, an ExactF32 cooperative GEMM epilogue**. L4 is the tessl-side piece the first audit's risk #1 and Fable's E1 ruling waited on (ojas Linear uses `GemmOperands::ExactF32`, which had no epilogue). When it lands it is input for ojas ft-e15ee76a / ft-2c183b71 and the vision chain's conv speed. Slot lists and shas will be announced before each fast-forward.
  - **tessl wave 15 L1 (heads-up, not merged):** the q4 MLX GEMV and GEMM kernels are templated into headers, with entry names and slots unchanged and outputs bit-identical, and `SIMD_*` moves to `kernels/q4_mlx_dot.h`. Its one ojas contact, `gemma-metal/bench/simd_occ_ab.sh`, is **already stale** [V]: it perl-edits `gemma-metal/kernels/gemv_q4_mlx.metal`, which no longer exists (only `.bak`, `.wip` and `.broken` copies remain), so it edits nothing today. Belongs to card ft-b90b30ae (the imported gemma-metal tree); that card is backlog, so the board refuses a log-only write and it is noted here. No hold.
  - **tessl main fast-forward bf0b7b9 → ca3efa8** (wave 14 plus `sq_norm` ea630de): ojas chose **(b)**, fast-forward only after "slot released by ojas", so the verify run attributes failures to ojas alone. Then ojas rebuilds against ca3efa8 as a separate check.
  - **tessl-d5 agreed.** Its subagents start no cargo or Metal job after 21:10, and wave 15 starts after 23:15. The tessl-eg2 build (pid 72050) and agent-ac621405 (pid 92521) belong to a separate tessl-eg2 Claude desktop session that neither ojas nor tessl-d5 can reach, so the owner must pause them.
  - **Coming in tessl wave 14:** `tessl::sq_norm` (ea630de). It unblocks `gp/clip-sq-norm`, and it changes the squared-norm numerics in `qwen35_adamw.metal`, which reach ojas-qwen35's provider. tessl-d5 was asked for the signature and the numerics delta before the fast-forward. **Answered (ea630de, from source):**
    - The API is `tessl::sq_norm::{sq_norm, SqWindow, SqNormPlan, decode_sq_norm}`, with `SQ_OUT_WORDS` = 6 and new shift buffers on the two squared-norm kernels.
    - Numerics: each piece is now power-of-two scaled. That is bit-identical in f32's normal range, and differs only where the old code overflowed or underflowed. AdamW step kernels are untouched.
    - **ojas main:** no direct use of the changed kernels or raw words [V rg]; it only calls `grad_sq_norm`. A moved pin on ordinary gradients is a regression.
    - **`gp/clip-sq-norm`:** matches the API (calls, 6-word out, compile-time asserts) [V git show]. The `parts` size of 2 × pieces is still exactly right: piece sums go in words [0, n) and shifts in [n, 2n), and buffer(3) binds at offset + 4n (tessl-d5, from ea630de's `encode`). tessl-d5 filed tessl **ft-c59ff950** (`gp-sq-norm-parts-words`, P3 backlog) for `SqNormPlan::parts_words()`. Its criteria: one owner for the rule, a test that encode refuses a buffer one word short, and ojas switching in the same change that adopts `sq_norm`. The ojas side lands with `gp/clip-sq-norm` and card ft-2c183b71. The board refuses a log-only write on a ready card, so this report is where it is tracked. Nothing in tessl wave 14 blocks ojas.

- **tessl wave 13 (2026-10-17, from tessl-d5):** fast-forwards tessl main from 4d3543f to 02b1e93. It changes 10 files and leaves kernels, the ABI and the ojas-facing API alone.
  - ojas replied **no hold**.
  - ojas does not use `IcbCopySmoke` [V rg]. Its only coupling is the `classify_alloc_matches_the_text_tessl_builds` test (`ojas-metal/src/device.rs:3728`), which pins three error strings in tessl's `src/runtime.rs`. All three are present on today's main (`runtime.rs:280, 284, 294`) [V rg].
  - tessl-d5 was asked to confirm they survive in 02b1e93. It did, and fast-forwarded. Verified on disk: tessl main is `02b1e93`, with the fragments at `runtime.rs:280/284/294` [V].
  - **Wave 14 is still in review:** an FA-2 attention template behind the entry names and slots gemma-metal binds. ojas's `gemma-metal/` depends on tessl (`Cargo.toml`, `build.rs`), so tessl-d5 was asked to announce the names, slots and any signature change ~30 min ahead. Its preview's tail slots match `gemma-metal/src/kernels.rs` exactly [V read]: h128 `kv_capacity`@13; h256 `out_bf16`@13 and `kv_capacity`@14; h512 `out_bf16`@12 and `kv_capacity`@13. No ojas change and no hold, if the final commit keeps them. tessl-d5 then sent the full `[[buffer(N)]]` lists, extracted from 4d3543f and from the template commit 059b2a0 and identical between them; they match ojas's bindings for slots 0–14 on all three entries [V read]. The final sha's extraction is to come with the announcement.
  - ojas's Metal suite was not run against it (load ~230; the gate blocks mac_heavy here).

- **tessl-65** (live) answered:
  - No tessl main fast-forward is queued, and it will announce shas first.
  - Its wave-13 lanes touch no kernel, ABI or ojas-facing API.
  - The heavy-lock holder `saved-mem-torch` (pid 30722) is **not** tessl-65's. The owner file says its cwd is ojas, started 19:36Z, so it belongs to another ojas session.
- **Lappi orchestrator** (not reachable by message). The asks are on ft-b5e64557's and ft-a1a4ba4d's logs:
  - post the landing sha of wave-l1/l2 and freeze `qd-train/src/{step,trainer,schedule,recipe,adamw,mock,run_control,pyjson,lib}.rs` for phase A;
  - name the adapter-checkpoint fields the converter needs (layout: ojas `docs/adapter-checkpoint.md` on `gp/a6-adapters`);
  - confirm the gate tool uses the same `mx.quantize(w, group_size=64, bits=8)` call;
  - call `Tape::linear_cross_entropy_value` for scale-0 rows.

## 6. Owner actions pending

1. **Run the batch:** `! bash backups/a6-tools/owner-batch.sh`.
   - It formats the code, generates and checks the fq8 fixture, and runs tests, clippy, the Linux check of ojas-train and the sha256 bench.
   - Every cargo step goes through `mac_heavy.sh` with its own target dir and a 40-minute bound.
   - Load was about 100 at 15:35 local time, and mac_heavy waits while load is above 12.
2. **Commits** in the four worktrees, once green. This session cannot run git there.
3. **Relay the Lappi asks** in §5.
4. **Push yes or no** for 146 commits; Fable recommends yes. The archive of the five duplicates is also yours. Everything else below was ruled by Fable on your delegation (§4), and the Lappi-side parts need a Lappi session. Items as originally listed:
   - `sha2` (unanswered means hand-rolled, as built);
   - the Lappi pin bump 5b67cdd → 04277f8, and later → the phase-A sha;
   - Lappi final-ruling items 5 (approve the q8 pre-registration), 6 (Mac GPU time for the quantised arm), 8 (Manvi verify of review cards) and 9 (peft);
   - merges and push;
   - archiving the five duplicates.
5. **The `saved-mem-torch` lock holder** (pid 30722): this session did not start it and will not kill it.

## 7. Gaps

- **GitPulse Insights:** every facet came back `ok: false` (git work shed under load) on both tries. Worktree and collision state were checked by hand: `git worktree list`, plus the clean siblings recorded in the branch report.
- **DevMap:** the index is stale (`source_freshness: false`, generation 7060). Lanes navigated by rg and reads; no claim here is confirmed by the graph.
- **Done gate:** `dev verify` and Manvi's done gate were not run. An agent's ceiling is review.
