> The w1–w11 GitPulse backlog report (session ojas-67, waves of 2026-10-13 to 10-17) is in `BUILD_REPORT-w11.md`. This file is the wave-12 report.

# Build report: GitPulse backlog, wave 12 (training on the Mac)

**Run:** 2026-10-16 18:00Z to 2026-10-17 02:10Z. Orchestrated from the main checkout.

**Base:** every lane branched from `main` at `040e1ed`.

**Focus:** training on the Mac, coordinated with the tessl repo.

**Roles:**
- Opus orchestrated, wrote the briefs and reviewed the code.
- Sonnet subagents implemented and attacked, at most 4 at a time.
- Fable made the judgment calls. The owner delegated owner decisions E6, OD1 and E17/OD8 to Fable on 2026-10-16.

The previous report (wave 1, 2026-10-13/14) is kept at `backups/2026-10-16/BUILD_REPORT-wave1-2026-10-14.md`.

**Evidence labels:**
- **[V]** verified this run by a command or log.
- **[I]** inferred.
- **[U]** unverified.

Every heavyq log cited below is under `/private/tmp/claude-501/-Users-bharath-Code-research-ojas/8c454257-e837-460c-9d17-abc2c8d1de13/scratchpad/heavyq/logs/`.

## Headline

- Four Mac-training cards were built, reviewed and moved to **review**. None was moved to done: Manvi gates done, `dev verify` was not run, and nothing is pushed.
- On the owner's instruction, relayed by session ojas-86, the four tips were handed to its consolidation pass, which merges them into local main. ojas-86 owns those merges; this session merged nothing.
- **Mid-run, tessl main fast-forwarded to `1b900ec`** with gp/wave10, including the chunked (WY) GDN merge `4b26182`. That turned every ojas Metal GDN backward red [V]: `chunked_gdn_backward: the backend charged 298264 workspace values; tessl needs 885760 bytes`.
  - Qwen3.5 training on Metal could not take a step.
  - This session filed **ft-a050deca** (p0 critical).
  - ojas-86 fixed it in `c4281d0`. ojas now asks tessl for the size, so no second copy of the formula exists, and the forward charges tessl's ChunkPrep transient. That fix is ojas-86's; it is unverified by this session.

## 1. Tasks moved to review

All five cards moved in_progress → review on 2026-10-17. Each card's summary carries the full evidence.

| Card | Branch @ tip | Opus review | Adversarial pass | Verify (heavyq) |
|---|---|---|---|---|
| **ft-29462f23** criterion 7 (safetensors 0.8.0 tags) | `gp/w12-ioread` @ `5f3de7f` | approved | yes (13 tests) | clippy 0 [V] |
| **ft-b301b6d6** G-IOREAD (streaming typed reads, header Budget check) | `gp/w12-ioread` @ `5f3de7f` | approved after fixes | yes (29 tests; 1 real bug, fixed) | clippy 0; 42 targets ok; only the 10 GDN Metal tests red [V] (`…014717-w12-ioread-final`) |
| **ft-fe6beb2e** C ABI SAVE data loss, cancel, RESUME preflight | `gp/w12-save` @ `dcfd76a` | approved after fixes | yes (27 tests; 3 findings, fixed) | clippy 0; `go test` ok 90 s; save paths green; only the GDN Metal tests red [V] (`…013109-w12-save-final`) |
| **ft-54e8ce27** G-BUDGET (check_room read-only, child-first) | `gp/w12-budget` @ `2509603` | approved | **no**, stopped by the wrap-up | core/capi/model green; only the GDN Metal tests red [V] (`…011533-w12-budget-final`) |
| **ft-f66c67c1** image data path + Metal MNIST step | `gp/w12-vision-data` @ `0c2441e` | approved with one fix (made by the orchestrator) | **no**, stopped by the wrap-up | parent `bb53eba` fully green: 21 image tests, the Metal step, clippy [V] (`…011539-w12-vision-final`). `0c2441e` is verified green: mnist_step loss is CPU 2.4691484 → 2.0016863 and Metal 2.4691486 → 2.0016863 (bit-equal after the step), and clippy exits 0 [V] (`…015713-w12-vision-step0`). |

### What each lane changed

**gp/w12-ioread**
- Commits: `5afa07c`, `23f748f`, `cacc1aa` (adversarial), `66997a9`, `5f3de7f`.
- Part 1: `StDtype` gained F8_E4M3FNUZ, F8_E5M2FNUZ and C64, parsed and carried.
- Part 2, `read_typed`: decodes through one `try_reserve_exact` output and a heap chunk. A 64 MiB BF16 read went from 67,108,872 B of extra heap to ≤ 1 MiB [V].
- Part 2, `load_hf`:
  - Sums header sizes per `HfKind` and calls `check_room` before any read.
  - Decodes runs straight into the charged outputs; the `Rows` staging is gone.
  - Peak on the tiny fixture fell from 4,263,173 B to 3,204,102 B (bound 3,392,272) [V].
- Part 2, `LoraLinear::merge`: no longer holds an uncharged weight-sized copy.
- Bench: tiny-fixture `load_hf`, min of 5, interleaved against main [V]. F32: 487 → 342 µs. Bf16: 378 → 278 µs.

**gp/w12-save**
- Commits: `616f46f`, `dfd6e8c`, `d9d2753`/`b13273a` (adversarial), plus the fixes up to `dcfd76a`.
- `ojas-io::replace_dir_with_if`: the predicate runs under `Site::lock` and again just before the first rename.
- Strict SAVE predicate (OD1). `.DS_Store` is accepted only as a regular file.
- New in-band kind `ojas:E_TARGET:` and Go `ErrTarget`.
- SAVE checks for cancel per tensor.
- RESUME reads the spec header and runs preflight before the device opens.

**gp/w12-budget** (`2509603`)
- `check_room` is a read-only walk up the chain. `charge` runs that check, then a child-first chain with rollback, and peaks move only after every level accepts.
- The `budget.rs:908-913` assertion went from 91 to 70, together with `adaptive-resources.md` C7b.
- A sibling-race test of 200k iterations fails on the parent-first code [I: by construction; the old-code job `…204841-w12-budget-old` ran, but its result lines were not retained in this report].

**gp/w12-vision-data** (`5ea6d00`, `bb53eba`, `0c2441e`)
- Std-only MNIST idx and CIFAR-10 readers with a typed `ImageError`.
- `ImageBatchSampler`: seeded, with a resumable cursor, normalisation, pad-4 crop and flip, and zero allocations per batch.
- `ojas-metal/tests/mnist_step.rs`: Metal and CPU agree, loss 2.4691486 vs 2.4691484 [V].
- `docs/data-formats.md`.
- Bench, a 128-image CIFAR batch [V]: 100 µs with augmentation off, 275 µs with crop+flip.
- The review fix in `0c2441e`: the lane passed `step = 1` to the first AdamW update. `Backend::adamw_step` documents `step` as "the count before this update" (`ojas-core/src/backend.rs:1139`), so the correct value is `0`.

## 2. Deferred or blocked (and why)

| Item | Why | Where it is tracked |
|---|---|---|
| ft-fe6beb2e criterion 4 (Qwen35Step::save_state under the replace lock) | `ojas-qwen35/src/state.rs` is rewritten on gp/w8-ckpt and gp/w10-readers (Fable: gate it, don't rebase). It is one hunk switching to `replace_dir_with_if`. | card summary |
| ft-fe6beb2e TOKENIZER mid-parse cancel | Needs a polled `load_hf_gpt2` in `ojas-data/src/bpe.rs`, held by gp/w11-readout (ojas-67). | card summary |
| ft-fe6beb2e cancelled-then-Closed `stuckCloseErr` | Root cause: gusset's 30 s join budget, plus `Handle.Close` caching its first error. The fix needs a Go in-flight tracker or a Rust job-count export. There is no deterministic failing test yet. | card summary |
| ft-b301b6d6 criterion 6 (LoRA merge peak = one weight + one chunk) | PARTIAL. The temporaries are charged but not streamed; row-chunking the matmul could change FP results on other backends. | card summary |
| ft-562b0add G-LIBM | Fable: deferred to after ojas-w11-oracle lands, because its planned test files collide (`ojas-cpu/tests/*`). E17 = make the claim true; OD8 = document per-target behaviour. | Fable ruling below |
| Every other Mac-training card | Blocked by live lanes' files (Fable's file matrix, §5). | §5 |
| Adversarial passes for budget and vision-data | Stopped by the owner's wrap-up request. | §1 |

## 3. Benchmark deltas

| Path | Before | After | Note |
|---|---|---|---|
| `load_hf` tiny fixture, F32 | 487 µs | 342 µs | min-of-5, interleaved [V] |
| `load_hf` tiny fixture, Bf16 | 378 µs | 278 µs | min-of-5, interleaved [V] |
| typed read, 64 MiB BF16, extra heap | 67,108,872 B | ≤ 1 MiB | counting allocator [V] |
| `load_hf` peak heap, tiny fixture | 4,263,173 B | 3,204,102 B | [V] |
| LoRA merge peak heap | 7,889,272 B | 3,170,728 B | [V] |
| CIFAR batch (128), aug off / crop+flip | — | 100 µs / 275 µs | new path [V] |
| `Budget::charge` ns/op | — | — | **not measured** [U]. It is on the per-step path; a follow-up should confirm it is within 5%. |

No regression over 5% was measured on any touched path. The Budget hot path was not measured.

## 4. New tasks filed and folds

**Filed:** **ft-a050decae5826321f0428469472298b6** (p0 critical): Metal GDN backward refuses every call since tessl's chunked GDN merge. ojas-86 took the fix (`c4281d0`). It was filed with `reviewed_related` naming all 24 listed cards.

**Folded** (one criterion appended per card; existing content resent unchanged):
- ft-3902a4b3: the MSRV comment in `ojas-simd/src/arch.rs:9` says 1.82 against 1.97.
- ft-2bbced18: zero-copy UMA mmap → `newBufferWithBytesNoCopy`, measurement-gated.
- ft-2c183b71: the `clip_grad_norm_` row against torch 2.15 nightly (18.96 vs 8.45 ms).
- ft-9c13f472: decide and document the `grad_clip == 0` semantics.
- ft-b90b30ae: land 0906ccc (`gemma-metal/build.rs` `gelu.h` → `act.h`).
- ft-b5fb19e6: one shuffle owner. Lift the sampler's Feistel permutation and retire `image.rs`'s Fisher-Yates.
- ft-30a7d34f: reword criterion 3 after E6, plus the residual 3-level transient race.

All seven were confirmed `ok: updated`. New revisions:

| Card | Revision |
|---|---|
| ft-3902a4b3 | 7 |
| ft-2bbced18 | 3 |
| ft-2c183b71 | 8 |
| ft-9c13f472 | 4 |
| ft-b90b30ae | 2 |
| ft-b5fb19e6 | 6 |
| ft-30a7d34f | 2 |

## 5. Fable's decisions (2026-10-16)

1. **E6 (G-BUDGET): option (a).** `check_room` is read-only and charging is child-first; the 91 assertion and C7b change together. ft-30a7d34f stays at backlog. A `check_room` call in `load_hf` is a point-in-time preflight.
2. **OD1 (SAVE): strict.** Criterion 4 is split out and gated on the gp/w8-ckpt merge, with no rebase.
3. **E17 + OD8 (G-LIBM): make the claim true.** Route through `log_sum_exp_exact` and add `ln1p` beside `ln_wide`. OD8: document per-target behaviour, with no runtime FMA. Defer until after ojas-w11-oracle.
4. **Waves:**
   - 12: ioread, vision-data, budget, save.
   - 13: G-LIBM, ft-2dc2dc69, SAVE criterion 4, ft-e55afcb5 criterion 9.
   - 14: ft-b5e64557 ojas-train, ft-2bbced18.
   - 15: G-GPUFAULT ojas half, ft-16173bf9.
   - 16: ft-9c13f472 alone, then ft-2c183b71.
   - 17: G-BENCH, then ft-da2c4ab1 LoRA-2B.
   - The vision chain interleaves from wave 15.
5. **tessl ownership:**
   - Wait-timeout policy and N=3 escalation: tessl. ojas already maps a poisoned runtime to `DeviceLost` (`ojas-metal/src/device.rs:556-566`), so AUDIT_REPORT.md:106 "no wait-timeout handling anywhere in ojas" is **stale**.
   - Exact-f32 epilogue: tessl. ojas stays on E1(a).
   - CheckpointView and per-entry mapping: tessl's API, ojas's consumer.
   - 2B memguard footprint: tessl provider. ojas owns the Tape-2B headroom. GPU runs are serialized.
   - Reader payload caps: each repo owns its own reader.
6. **Untracked `tasks/*.md` bundles: do not import.** Their novel items were folded (§4). The BFMMLA NEON Muon is deferred with no card. The DevCouncil config, the ojas-cw symlink and the fb907e3 hold are owner actions. Five-repo convergence is rejected as unbounded.
   - **Note:** board card ft-e15ee76a ("Route ojas-metal Fast/Autocast to tessl cooperative TensorOps epilogues…") matches the bundle `gp-metal-fast-lane-epilogue-and-cpu-bf16-muon.md`. Someone filed it after this ruling, and this session did not touch it. Its clip item duplicates the ft-2c183b71 fold.

**Orchestrator rulings (logged, not escalated):**
- `.DS_Store` allowed as a regular file in a SAVE target: a usability refinement of strict OD1 for Finder.
- The new `E_TARGET` / `ErrTarget` is an **additive** C-ABI/Go contract change. `ErrorKind` is `pub(crate)`, nothing is renumbered and no message changes, so it was not treated as breaking. Peer ojas-67 flagged it.
- The IO lane's ≤ 64 KiB read chunk stays uncharged on the heap: it is within the card's "+ one chunk" bound, and charging it would have meant weakening the exact-budget adversarial tests.
- ojas-metal gained a dev-dependency on the in-workspace `ojas-data` crate. This is not a new third-party dependency.

## 6. Owner decisions still pending

1. **Merge of the w12 tips.** ojas-86 is merging them into local main under your relayed instruction. Please confirm that instruction was yours. Budget and vision-data merge **without** an adversarial pass. Vision-data merges at `0c2441e`, whose own verify was pending when this was written.
2. **Push** (E11b / OD10). Local main is far ahead of origin, so no CI claim has run on GitHub.
3. **tessl gp/wave10 landed on tessl main** during this run. That was Fable's "biggest unblock", and it releases gp/w8-ckpt, gp/w10-readers, SAVE criterion 4 and ft-e55afcb5. It also broke ojas's Metal GDN (ft-a050deca, fixed by ojas-86 in `c4281d0`). From now on, consider gating tessl main fast-forwards on ojas's Metal suite.
4. **Still-open audit decisions:** E9 (`--locked`/SHA pins), E12 (ft-69ef4ce1 back to ready), OD2 (C11 digest), OD3, OD6, OD7, OD12 (orphaned pid 97075). E1-E5, E7, E8, E10, E11, E13, E14 and E16-E18 have recommendations and will go to Fable as each card starts, per your delegation.
5. **`dev verify`** was not run. `.devcouncil/state.sqlite` now exists (created by ojas-67), but no `.devcouncil/config.yaml` is committed.

## 7. Process notes

- **Build lock.** Every lane waited about 3 h on mac_heavy, because the w11 runner held the lock through idle periods. From about 21:00Z all w12 builds went through ojas-67's shared heavyq, by agreement.
- **Queue hygiene.** Two queue problems happened during the run:
  - A peer's FIFO test hung the queue for 1.5 h. The w12 lane rules now require self-bounded tests and a 40-minute per-step watchdog.
  - A housekeeping job may purge an idle `target/` directory over 10 GB. Every job now runs a keep-alive touch loop.
- **The tessl move mid-run** made "unchanged code fails" look like a branch bug. Every lane's Metal red was traced to `4b26182` before any blame was assigned.
- **Not done:**
  - `dev verify`;
  - `Budget::charge` bench;
  - adversarial passes on budget and vision-data;
  - CUDA paths (ojas-cuda `k11_golden.rs`, which cannot run on this Mac);
  - Linux (podman not used).

