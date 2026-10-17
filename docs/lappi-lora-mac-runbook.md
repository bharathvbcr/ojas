# Runbook: Lappi decision-model LoRA on the M5 Pro, bf16 then quantised (2026-10-17)

Audience: the owner, who starts the heavy runs. Base: ojas main `04277f8`, Lappi main as read 2026-10-17.
Labels: **[V]** read from the cited file today; **[I]** inferred. Nothing here was compiled or run.
A step whose driver does not exist is written `BLOCKED on <card/gap>` with no command. No command below is
invented: each names a script or test that exists today.

Sources (all read): ojas `BUILD_REPORT-a6.md`, `backups/a6-tools/card-ft-{da2c4ab1,a1a4ba4d,b5e64557,2bbced18}.json`,
`bench/results/2026-10-17-lora-step/README.md`, `ojas-qwen35/tests/gpu_memory_rows.rs`; Lappi
`HANDOFF/lora2b-audit2-2026-10-17.md`, `AUDIT/lora-2b-audit2-2026-10-17/fable-final-ruling.md`,
`AUDIT/lora-2b-quantised-2026-10-17/owner-decision.md`, `campaign/lora-2b-preregistered.json`,
`tools/mac_heavy.sh`, `CLAUDE.md`.

## 0. Today: three blockers, in order

| # | Blocker | State [V] | What the owner does |
|---|---|---|---|
| 1 | The nine a6 lanes are uncommitted and unverified | No lane has compiled (`BUILD_REPORT-a6.md` headline). Lanes: `gp/a6-train`, `-adapters`, `-valuece`, `-fq8`, `-docsync`, `-stepcost`, `-preflight`, `-adamw100`, `-merge` | Step 2 below runs the commits, the batch and the merges as one script |
| 2 | `owner-merge.sh` has not run | Commits all lanes, runs `owner-batch.sh`, stops on any red step or fixture drift, merges `--no-ff` into **local** main, regenerates the site, tests main in `target-a6-main`. Pushes nothing (script header) | `! bash backups/a6-tools/owner-merge.sh` from the ojas main checkout |
| 3 | Load | `mac_heavy.sh` waits while the 1-minute load average exceeds `MAC_HEAVY_MAX_LOAD` (default **12**) or the process count exceeds `MAC_HEAVY_MAX_PROCS` (default 2500); it exits 76 if that does not clear within `MAC_HEAVY_MAX_WAIT_S` (default 4 h) (`Lappi-decision/tools/mac_heavy.sh` header lines 15, 55-57). Load was about 72 at 16:38 [owner's note], far above 12 | Wait for a quiet machine, or raise `MAC_HEAVY_MAX_LOAD` on the launch line for **CPU-only** steps (the merge batch). That is the owner's choice. Never raise it for a GPU run: load 72 means other builds are running and the one-heavy-job rule exists because of the 2026-10-02 kernel panic at load ~350 |

Also queued at the lock: pid 30722, a `saved-mem-torch` mac_heavy wrapper that is not this session's and not tessl's
(`BUILD_REPORT-a6.md` section 4 item 7, section 5). Leave it. A job started behind it waits [V].

Run order for blocker 2, with the load override only for the CPU batch (the variable is read by `mac_heavy.sh`, which
the scripts call; whether the script forwards your environment is [I], check `owner-merge.sh` line 81 context):

```
! MAC_HEAVY_MAX_LOAD=<n> bash backups/a6-tools/owner-merge.sh
```

**Phase A (the ojas-train import) is BLOCKED on the Lappi freeze sha.** `backups/a6-tools/owner-phase-a-import.sh` is
**planned**: it does not exist yet [V `ls`, no such file]. It needs Lappi to land wave-l1/l2 and freeze
`qd-train/src/{step,trainer,schedule,recipe,adamw,mock,run_control,pyjson,lib}.rs` (ft-b5e64557 criterion 2; `BUILD_REPORT-a6.md`
section 5). Until it exists there is no ojas trainer for the Tape, and steps 2, 4, 5 and 6 below cannot start.

Also pending before any training step (Lappi `campaign/lora-2b-preregistered.json` amendment 3 [V]): the v5r shards are
**withdrawn**. LoRA-2B trains on a shard set rebuilt under the v6 data rules. The new shard paths, hashes and `max_width`
are `pending_before_step_0`, as are a passing `qd-train-data` containment check and `qd-prereg-bind`. If the width
distribution changes, N is re-derived by amendment 2's rule. Until that later amendment is committed, "no LoRA-2B run is a
registered row". The mechanics check on ft-da2c4ab1's last criterion names v5r train-shard rows: re-confirm which shard set
it may read before running it.

## 1. Fable's order

Order [V ft-da2c4ab1 criterion 12, ruling section 1]: LoRA rows, 0.8B bf16 smoke, amendment 1, reduced 2B bf16 run,
quantised 0.8B smoke, reduced quantised 2B run. **No quantised run starts before the bf16 reduced run's ledger row exists.**

Measured rates all come from `bench/results/2026-10-17-lora-step/README.md` (checkpointing on, 28 GiB budget):

| Row | T | Device peak | Host footprint (memguard's metric) | tokens/s |
|---|--:|--:|--:|--:|
| 0.8B-Base | 2,048 | 3.94 GB | 4.45 GB | 658.2 |
| 0.8B-Base | 9,638 | 7.83 GB | 8.41 GB | 499.5 |
| 2B-Base | 2,048 | 6.65 GB | 7.95 GB | 335.9 |
| 2B-Base | 9,638 | **13.10 GB** | 14.31 GB | **261.6** |
| 2B-Base | 9,638, checkpointing off | refused (`CapacityExceeded` at `forward`) | - | - |

Checkpointing must stay **on**: off does not fit 28 GiB at T=9,638 [V].

### 1.1 LoRA memory and tokens/s rows: DONE

| Item | Value |
|---|---|
| Card | ft-2bbced18 criteria 7 and 9 (status in_progress; the card is not moved by this runbook) |
| Rows | `bench/results/2026-10-17-lora-step/rows.jsonl`, committed at ojas `4da214b`; 2B T=9,638 is `rows.jsonl#L4`, 261.607027 tok/s [V Lappi amendment 2] |
| N | **127**: 17,280 x 261.607 / 35,403 = 127.69, floor 127; one step is 135.3 s [V README, amendment 2] |
| Owner | ojas (done) |
| Not covered | D17 and the span head (the arm scores every position through the fused tied head), real data, the provider and torch comparison rows [V README "What this does not show"] |

The same row can be re-taken. This is the command the handoff gives, and the test arm exists [V `gpu_memory_rows.rs:114`;
env vars `QWEN35_MEM_ARM`, `QWEN35_MEM_TOKENS`, `QWEN35_MEM_CKPT` (`on` default or `off`), `QWEN35_MEM_SNAPSHOT`]:

```
QWEN35_MEM_ARM=tape-lora QWEN35_MEM_TOKENS=9638 \
bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh lora-mem-row-2b-9638 \
  cargo test --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml -p ojas-qwen35 --release \
  --test gpu_memory_rows -- --ignored --test-threads=1 --nocapture gpu_memory_row
```

Do not repeat it unless a row is in doubt: it costs a GPU window and N is already fixed from the committed row.

### 1.2 0.8B bf16 smoke (bound d', 60 steps, both objectives): BLOCKED

| Item | Value |
|---|---|
| Prerequisites | ojas ft-b5e64557 (phase A import, then phase B: Tape `StepProvider`, adapter checkpoint) [status ready, none built]; Lappi ft-3b8126cd (D17 host objective, torch reference-arm driver, LoRA parameter plan with the `lora` ledger row writer, checkpoint-to-scorer converter) [live Lappi lane]; the v6 shard rebuild (amendment 3); section 0 blockers 1 to 3 |
| Owner of each piece | ojas: Tape `StepProvider`, adapter checkpoint (layout v1 in `docs/adapter-checkpoint.md` on `gp/a6-adapters`). Lappi: D17, torch driver, ledger row, converter, shards. Owner: launches, which is the consent |
| Command | **BLOCKED on ft-b5e64557 phase A/B and Lappi ft-3b8126cd.** No ojas or Lappi driver runs the Tape under Lappi's loop today (`GAP-LORA2B-TAPE-PROVIDER-MISSING-2026-10-17`, `GAP-LORA2B-TORCH-DRIVER-MISSING-2026-10-17`). The existing `qd-train-metal` (`--resume`, `--seed`, `--epochs`) drives the provider full fine-tune, not the Tape LoRA, so it is not this step [I, from the ruling's section 1 verdict] |
| Pass gate | ojas vs torch MPS bf16: \|delta loss\| <= 0.02 nats for steps 1-50, EMA(0.9) <= 0.05 after; fp32 torch CPU arm as the tight pin; both objectives (17-way D17 and full-vocab) [V ft-da2c4ab1 criterion 5] |
| Memory | Device peak 7.83 GB, host 8.41 GB at T=9,638 (above); 28 GiB MetalBackend budget under the 32 GiB memguard cap. The pre-flight (`gp/a6-preflight`, after merge) refuses a shape that does not fit instead of dying to SIGTERM |
| Wall time | 60 steps x 35,403 tokens / 499.5 tok/s = 70.9 s per step, about **71 min** [I; Fable's estimate is 35 min to 1 h, scaled by params] |
| Record | Per-step loss for ojas and both torch arms, peak bytes, tokens/s, a quick ledger row. Rows under ojas `bench/results/`. Not a promotion |
| Interim, not the smoke | ft-da2c4ab1's last criterion: a 60-step mechanics check composed from merged primitives (frozen base bit-unchanged, only adapters move). **No driver exists for it today** (the `gpu_lora_step_trains_only_the_adapters` test is tiny-fixture, 2 steps). BLOCKED, no command. It writes no ledger row and cannot be cited as the smoke |

### 1.3 Amendment 1 (Lappi; N fixed from the row): DONE in Lappi, with one reopening

| Item | Value |
|---|---|
| State [V] | `campaign/lora-2b-preregistered.json` `amendments[1]` (step definition, window rule) and `amendments[2]` (**n = 127**, row id `ojas@4da214b:bench/results/2026-10-17-lora-step/rows.jsonl#L4`, 261.607027 tok/s, warmup `max(1, N//20)` = 6, cosine over N) are committed on Lappi main; both recorded before any LoRA-2B result exists |
| Reopened by | `amendments[3]`: if the v6 rebuild changes the width distribution, N is **re-derived by amendment 2's rule** in the later data amendment. Otherwise N stays 127 [V] |
| Owner | Lappi writes the later amendment; the owner approved the amendment path ("merge all", "procced with fable's advice") [V] |
| Rule | An amendment made after any LoRA-2B result is read is void (`amendments_rule`). Write it **before step 0 of the smoke**, not after |
| Wall budget | 127 x 135.33 s = 17,186 s = 4.77 h, inside 17,280 s (80% of 6 h). A run that hits `WallClockCap` with fewer than 127 steps is no longer a registered row [V amendment 2] |

### 1.4 Reduced 2B bf16 run (kill -9 resume, export, Lappi scorer): BLOCKED

| Item | Value |
|---|---|
| Prerequisites | 1.2 passed; 1.3 data amendment committed; ft-b5e64557 adapter checkpoint (criterion 13: a kill -9 mid-run resumes to the same loss trajectory); ft-58e316a9 export (ojas `gp/a6-merge` provides `merge_bf16`, one RNE rounding; Lappi owns the converter, `GAP-LORA-CKPT-LAYOUT-NOT-OJAS-VERIFIED-2026-10-12`) |
| Owner of each piece | ojas: trainer, checkpoint, `merge_bf16`. Lappi: converter into the `lora-merged` layout, `ckpt_average --from lora-merged`, `qd-export`, `qd-metal` score, ledger row. Owner: launch |
| Command | **BLOCKED on ft-b5e64557 phase B, Lappi ft-3b8126cd (converter) and ft-58e316a9.** No command exists |
| Pass gate | N = 127 steps complete; a deliberate `kill -9` of **its own pid** mid-run resumes to the same loss trajectory as an unkilled run; export merges; Lappi's own scorer returns a quick row. The row cannot promote [V ft-da2c4ab1 criterion 6] |
| Memory | Device peak 13.10 GB, host 14.31 GB at the widest sequence; a step's peak is the widest sequence's plus a ~63 MB f32 adapter bank [V ruling section 2, I for the bank size]. 13.10 GB is the measured number; the 28 GiB budget is the guard |
| Wall time | **4.77 h** (127 x 135.3 s) plus load, a kill and a resume, and the export and score [I]. Approved window 3-6 h |
| Record | Ledger row with `n`, `row_id`, `tokens_per_s` copied from amendment 2; loss trajectory; peak bytes; tokens/s; the kill step and the resumed-vs-unkilled comparison; Lappi score. Name the shard hashes of the v6 rebuild |

### 1.5 Quantised 0.8B smoke (fake-quant q8 base from the Lappi gate tool): BLOCKED

| Item | Value |
|---|---|
| Authority | Owner item 6 (the ~1 h quantised 0.8B smoke) approved by Fable under the owner's "ask fable for decisions" delegation [V `BUILD_REPORT-a6.md` section 4 item 3]. The owner launches each run; the launch is the consent. The same delegation said the <=6 h reduced quantised 2B run is approved **once the bf16 reduced row exists** |
| Prerequisites | 1.4's ledger row exists; Lappi gate tool (`GAP-LORA2B-QUANT-GATE-TOOL-MISSING-2026-10-17`, ft-2ed2fd55 criterion 5) has written the 0.8B-Base fake-quant file and its sha256 (the mlx call `mx.quantize(w, group_size=64, bits=8)`; q8 g64, bf16 scale and bias) and the D3 gate has been **admitted** (argmax agreement >= 99.0%, mean \|delta log-prob\| <= 0.02 nats, >= 500 validation prompts, never held-out); **owner item 5** approved `campaign/lora-2b-q8-preregistered.json` and it is committed before any quantised result |
| Owner of each piece | Lappi: gate tool, base files, sibling pre-registration, family id `lora-2b-q8`. ojas: the trainer unchanged, the checkpoint naming the base by the file's sha256 plus the quantiser record. ojas lane `gp/a6-fq8` (after merge) adds the tiny-fixture (a')(b')(c') parity over fake-quant weights only; it is a test fixture, not the Lappi gate tool |
| Command | **BLOCKED on the Lappi gate tool (ft-2ed2fd55 criterion 5), owner item 5 (sibling pre-registration not drafted), and ft-b5e64557 phase A/B.** The ojas fixture generator `ojas-oracle/python/qwen35_fakequant_base.py` is for the tiny fixture and is not the 0.8B base writer. Do not substitute it |
| Pass gate | Same as 1.2 (bound d', 60 steps, both objectives), the torch arm run over the same fake-quant weights [V ft-da2c4ab1 criterion 8] |
| Memory | Rows equal the bf16 rows by construction, because the file is bf16 (the quantised-rows criterion is superseded) [V ft-2bbced18 criterion 8 note]. The saving is 1.29 GB of a 12-14 GB step, which nothing needs |
| Wall time | About 71 min, as 1.2 [I] |
| Record | Family `lora-2b-q8`, `base.format = mlx-affine-q8-g64-fakequant`, base-file sha256, mlx version in every row. `optimizer_recipe "lora"` or `"lora-q8"` as the Lappi lane decides before the first row [V ruling section 3 item 5] |

### 1.6 Reduced quantised 2B run (only after the bf16 row exists): BLOCKED

| Item | Value |
|---|---|
| Prerequisites | 1.4 row exists; 1.5 passed; the 2B-Base fake-quant file and sha256; the sibling pre-registration fixes its own N (**not** assumed to be 127; at the same rate N = 127 [I]) |
| Command | **BLOCKED on everything in 1.5 plus the Lappi converter.** Merge target is dequant(Q(W)) + (alpha/r) B A rounded once to bf16; merging into the original bf16 W is refused by base-sha mismatch [V ruling section 3 item 6] |
| Pass gate | Same shape as 1.4; scored by Lappi's scorer; reported beside the bf16 quick row (loss trajectory, score, peak bytes, tokens/s). Cannot promote |
| Memory, time | As 1.4: 13.10 GB device, about 4.8 h [I]. Approved up to 6 h |
| Record | As 1.4, with the `lora-2b-q8` family fields from 1.5 |

## 2. Safety

1. **One GPU job at a time.** Launch every heavy command through `mac_heavy.sh`; it holds one lock (`MAC_HEAVY_LOCK`), refuses
   below `MAC_HEAVY_MIN_FREE_GB` (default 15) free, and sets `CARGO_BUILD_JOBS=2` [V header]. Never start a GPU run beside
   `owner-bench-stepcost.sh` or the merge batch. Check ListAgents/GitPulse for a live GPU owner first.
2. **Kill only your own PIDs.** Record the PID at launch and `kill <pid>`. Never `pkill`, `killall` or `pkill -f`. pid 30722
   is not yours. For the kill -9 resume test, kill the training pid you started, not the `mac_heavy.sh` wrapper
   (the wrapper's SIGTERM path stops the whole tree [V header]).
3. **memguard SIGTERMs any single process above 32 GiB** (footprint, 2 s samples, SIGKILL 10 s later) and the largest process of
   2 GiB or more under critical pressure for 3 samples. It writes **no log** on this Mac: a kill shows as signal 15 with no
   `mac_heavy` footprint message [V header, `Lappi-decision/CLAUDE.md`]. Keep the 28 GiB MetalBackend budget. The measured
   2B step is 13.10 GB device and 14.31 GB host, so there is room, but a second process counts separately.
4. **Rule 3: never held-out or validation rows in training.** The smoke and runs read **train** shards only; the D3 gate may
   read validation prompts (never held-out). Amendment 3's containment check must exit 0 before step 0.
5. **The quantised rows never pool with bf16.** Family `lora-2b-q8`, `base.format` and the base-file sha256 in every row; a reader
   keyed on `optimizer_recipe` alone must also key on the family id. No quantised run before the bf16 reduced row exists.
6. **Bound d' is fixed** (0.02 nats steps 1-50, EMA(0.9) 0.05 after). Do not retune it after reading a result; an amendment
   made after a result is read is void.
7. The merge scripts never push and never touch main's dirty peer files (`.gitignore`, `bench/*`) [V `owner-merge.sh` header].

## 3. Open owner items

| # | Item | State | Notes |
|---|---|---|---|
| 1 | Run `! bash backups/a6-tools/owner-merge.sh` | open | Section 0. Then `owner-bench-stepcost.sh` (about 3-5 h GPU) only when the Mac is free; it does not gate the merge |
| 2 | Pin bumps | Lappi 5b67cdd to 04277f8 ruled **commit it** by Fable (a Lappi session executes); later bumps to the a6 merge sha (for `Tape::linear_cross_entropy_value`) and to the phase-A sha | Each bump needs a clean ojas tree, a delta report and `make pin` PASS; report deltas and ask before bumping. Two Lappi agent worktrees both edit `upstream.lock.json`: check before either commits |
| 3 | Push | open, needs a one-line yes | Publishes **146** commits; Fable recommends yes, since Lappi's lock url serves nothing until the push [V `BUILD_REPORT-a6.md` section 4 item 4] |
| 4 | Manvi verify (`work.verify.task`) | open, owner only; agents stop at review | ft-29462f23, ft-6fb212ce, ft-531216ed (text fix in the docsync lane), ft-3cc9ed3f. Not `dev verify` |
| 5 | Archive duplicate cards | open, one click each | ft-29f7e4b6, ft-dc4bfe22, ft-e15ee76a, ft-ee2a3753, ft-a95d2145. Archive, do not merge; canonical map in `BUILD_REPORT-a6.md` section 2 |
| 6 | sibling pre-registration (item 5) | open | Approve `lora-2b-q8-preregistered.json` once drafted; it must be committed before any quantised result |
| 7 | Quantised GPU time (item 6) | approved by Fable's delegation for the 0.8B smoke and for the reduced 2B run once the bf16 row exists | The launch is the consent |
| 8 | peft (item 9) | open; keep "not run" unless answered | Test is not run |
| 9 | 4-bit and stage 2 | **no** and **not now** [V owner-decision] | q4 is measured in the gate, not built |
| 10 | sha2 | hand-rolled `sha256` stays unless the batch's bench is below 150 MB/s (min-of-7) | Then the owner is asked, with the number |
| 11 | Window extension | optional | If the v6 rebuild shrinks N too far, the owner may extend the window instead of accepting a small N |
