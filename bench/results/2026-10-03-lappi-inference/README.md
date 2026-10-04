# Lappi on the Mac: base Qwen3.5-2B vs the latest Lappi release, per task, tessl vs PyTorch

2026-10-03 (UTC 02:07–02:31), Apple M5 Pro. **Speed only, report-only**: no accuracy was measured and
nothing here gates or promotes anything. Labels: **[V]** measured here, **[I]** inferred.

> **Which "latest Lappi" this is.** The release timed here is the 5-seed masters *average*. It was
> exported (00:32Z) before it was scored. J7′ has since scored it, on main at `312dced`, in
> `ledger/gh200-p6-f-j7prime-2026-10-01.jsonl`:
>
> | Artifact | Row | Choice | Span |
> |---|---|---|---|
> | Average | `c962cdd9` | 9016/10985 (82.1%) | 5600/7238 (77.4%), ~13 points below the single seeds' 90–91% |
> | Logit ensemble of the 5 seeds | `b45406b5` | 83.8% | 91.2% |
>
> Which artifact becomes F's candidate (the average, the ensemble, or neither) is the human's decision under G8; none is
> chosen yet. The speed numbers below hold for any single tower of this architecture. They say
> nothing about quality, and this release should not be called "the latest Lappi" in any quality
> comparison. The ensemble runs 5 towers per decision, so on the Mac it would cost roughly 5× the
> per-decision compute and ~19 GB of bf16 weights. That is inferred, not measured.

## What was compared

| | |
|---|---|
| Original model | Qwen/Qwen3.5-2B-Base, HF snapshot `b1485b2f`, tower weight hash `92f6bd1c…` [V] |
| Latest Lappi | `p4-v4-avg-masters-v1`: the qd-export release of the 5-seed masters average of run F (corpus v4; ft rows `973cd4e3`, `95fa4854`, `32990e1a`, `1845f471`, `62a11100`). Pulled read-only from the GH200's `/home/ubuntu/release/`; all 7 files match `release_manifest.json` sha256; tower weight hash `a68f19bc…` = the manifest's `expected_identity.weight_hash` [V] |
| Engines | **tessl** through qd-metal (Lappi's Mac decision backend); **PyTorch 2.12.1 MPS**, bf16, sdpa, the repo's own `qd_train.backbone.load_text_tower` |
| Not run: ojas | ojas has no Qwen3.5 inference path. `ojas-qwen35` is a training-step provider (forward/backward/AdamW over tessl kernels), not an inference engine |
| Tasks | Prompts from F's own val shard set (`ef06ab99…`, 18,223 sequences, the hash on F's eval rows). Per task family, the real prompt nearest the family's p50 and p90 prompt length, plus 2K/8K points at the needle-hunk suite's size (concatenated real val tokens) [V] |

## Results (min of N, ms; lower is better)

| Task | Point | Tokens | tessl prefill base / Lappi | PyTorch MPS base / Lappi | PyTorch ÷ tessl | tessl full decision base / Lappi |
|---|---|---|---|---|---|---|
| CLINC intent (16-way) | p50 | 200 | 42.5 / 42.4 | 92.5 / 93.0 | 2.19× | 217 / 216 |
| CLINC intent (16-way) | p90 | 207 (tessl at 200) | 42.5 / 42.4 | 88.4 / 87.1 | 2.07× | 217 / 216 |
| CLINC domain (10-way) | p50 | 167 | 39.6 / 39.3 | 74.8 / 84.1 | 2.01× | 210 / 209 |
| CLINC domain (10-way) | p90 | 171 (tessl at 167) | 39.6 / 39.3 | 82.5 / 80.4 | 2.06× | 210 / 209 |
| CLINC in-scope (2-way) | p50 | 131 | 38.1 / 38.2 | 121.2 / 118.5 | 3.14× | 190 / 189 |
| CLINC in-scope (2-way) | p90 | 135 (tessl at 131) | 38.1 / 38.2 | 119.5 / 119.7 | 3.14× | 190 / 189 |
| CLINC within-domain (15-way) | p50 | 196 (tessl at 200) | 42.5 / 42.4 | 90.6 / 92.2 | 2.15× | 217 / 216 |
| CLINC within-domain (15-way) | p90 | 203 (tessl at 200) | 42.5 / 42.4 | 88.2 / 87.5 | 2.07× | 217 / 216 |
| CSQA (5-way) | p50 | 151 | 40.9 / 38.7 | 124.6 / 114.5 | 3.00× | 207 / 207 |
| CSQA (5-way) | p90 | 161 (tessl at 167) | 39.6 / 39.3 | 85.3 / 82.5 | 2.13× | 210 / 209 |
| MMLU (4-way) | p50 | 184 | 41.3 / 40.7 | 80.8 / 79.9 | 1.96× | 211 / 210 |
| MMLU (4-way) | p90 | 331 | 61.4 / 61.8 | 119.9 / 124.3 | 1.98× | 249 / 250 |
| Code defect class (4-way) | p50 | 409 | 80.1 / 79.6 | 139.9 / 140.7 | 1.76× | 279 / 281 |
| Code defect class (4-way) | p90 | 770 | 138.9 / 138.7 | 249.9 / 249.5 | 1.80× | 377 / 377 |
| Code defect span | p50 | 400 (tessl at 409) | 80.1 / 79.6 | 136.7 / 137.4 | 1.72× | 279 / 281 |
| Code defect span | p90 | 763 (tessl at 770) | 138.9 / 138.7 | 235.5 / 234.6 | 1.69× | 377 / 377 |
| SQuAD answer span | p50 | 276 | 57.6 / 57.5 | 103.0 / 105.1 | 1.81× | 244 / 241 |
| SQuAD answer span | p90 | 376 (tessl at 409) | 80.1 / 79.6 | 128.1 / 124.8 | 1.58× | 279 / 281 |
| Long context (needle size) | T=2048 | 2048 | 343.6 / 360.2 | 640.1 / 637.3 | 1.82× | 763 / 773 |
| Long context (needle size) | T=8192 | 8192 | 1988.4 / 2133.3 | 3127.6 / 3123.9 | 1.52× | 3276 / 3293 |

Column definitions:

- **tessl prefill** and **PyTorch MPS** time the same work: one prefill of the prompt plus 17 answer-letter
  rows at the last position, with no full-vocabulary head. tessl is median-of-7 after 2 warm-up and reports
  min; PyTorch is 10 reps after 3 warm-up, base and Lappi interleaved, with the order flipped every rep.
- **tessl full decision** is qd-metal's product path: prefill, two read-only option passes (61 tokens each),
  three state digests and readback. It uses the best of two embed arms, min-of-7.
- The tessl T list was fixed; where a PyTorch prompt length is not in it, the nearest tessl T is shown.

## Readings

1. **Lappi costs nothing at inference over the base model [V].** The two share an architecture
   and differ only in weights. Lappi ÷ base:
   - tessl full decision: 0.985–1.013 at every length;
   - PyTorch, interleaved: 0.92–1.13 per point, median ≈ 1.00.

   The one bigger gap, tessl prefill-only at 2K/8K (1.05/1.07), did not reproduce:
   - in the decision run's prefill component at 8K, base reads 2006–2091 ms and Lappi 2057–2105 ms;
   - in PyTorch's interleaved run at 8K, the ratio is 0.999.

   The prefill-only runs were sequential, not interleaved, so I read that gap as run-to-run variation [I].
2. **tessl is the faster engine on every task [V]:**
   - **1.6–2.2×** PyTorch MPS on the 150–800-token prompts that make up these tasks;
   - **1.5×** at 8K.

   The two points under 1.7× are length-mismatched against tessl: SQuAD p90 is 376 tokens timed
   against tessl's 409, and defect span p90 is 763 against 770. tessl also loads in 4.2 s vs
   19–22 s for PyTorch.
3. **On qd-metal, the product decision path is ~190–380 ms per decision on these tasks [V].**
   - The host-side state digest is a large share of it. From the Lappi run's medians: 62% at T=131
     (121.7 of 195.1 ms), 59% at 200, 57% at 331, 54% at 409, 52% at 770, then 46% at 2K and 34% at 8K.
   - That digest cost is the open `GAP-QDM-STATE-DIGEST-HOST-SHA256-PER-DECODE-2026-10-02`.
   - Prefill alone is 27–130 ms at the task lengths.
4. **Two PyTorch points are unexplained [V, cause unknown].** The 131/135-token CLINC in-scope prompts
   and the 151-token CSQA p50 took ~115–125 ms. Same-model prompts of 161–207 tokens took 75–93 ms.
   - tessl shows no such step.
   - The effect held for both models and across all 10 reps, so it is shape-specific behaviour in
     PyTorch on MPS, not noise from one model.
   - Those points inflate the PyTorch ÷ tessl ratio to 3.0–3.1×.
5. **The argmax letters differ** between base and Lappi on 11 of 20 prompts. That is expected (they
   are different weights) and is **not** an accuracy measurement. The prompts were timed, not scored.

## What this is not

- No accuracy. Lappi's accuracy is in the Lappi ledger. The 5-seed average's GH200 gates row
  `c962cdd9` reads choice 9016/10985 (82.1%) and span 5600/7238 (77.4%). I read that row from the
  lead's read-only pull, `Lappi-decision/build/box-j7prime-pull.jsonl`; it is not yet in main's
  `ledger/` [V].
- No zero-shot accuracy for the untrained base on v4's val set was found. I searched Lappi's
  `ledger/`, `HANDOFF/` and `AUDIT/` for `zero.shot|frozen.base|rung.?1|untrained`. The only hits
  were on the pre-v4 commitpackft pipeline that `tools/mac_zero_shot.py` is wired to.
- One run per engine and model. The decision rows are `quick` by construction.
- Build state:
  - qd-metal was built from Lappi `997eb54` with another session's uncommitted `backend.rs` /
    `parity.rs` / `tokenizer.rs` edits in the tree;
  - it linked canonical tessl `cf65d9d5` with 29 dirty paths.

  The decision rows record both.

## Provenance

- **Slot.** Mac GPU time was taken under Lappi's heavy-job lock: hold `lappibench`, granted by the
  Lappi lead session. Every step passed the busy gate (load1 3–5, ~1,150 processes). Peak RSS was
  10.1 GB, with 14.4 GB allocated by the MPS driver. The build was cargo `-j 2`, 59 s.
- **No writes elsewhere.** Nothing was written to the Lappi tree, its `ledger/`, or the GH200.
  The build used `cargo --locked`. Afterwards, Lappi's `Cargo.lock` and tessl's `Cargo.lock` still
  carry their earlier mtimes: 2026-10-02 07:01 local and 2026-10-01 13:32 local. Their uncommitted
  diffs predate this run, and tessl's dirty-path count is still 29 [V].
- **Gaps.** The Lappi lead recorded two gaps from this run in Lappi's `gaps.jsonl` (main `5fe1c40`):
  - `GAP-QDM-BENCH-PREFILL-MODE-NO-SNAPSHOT-FLAG-2026-10-03`: the `HOME` workaround in step 3;
  - `GAP-TORCH-MPS-STEP-AT-131-151-TOKENS-2026-10-03`: reading 4.
- **qd-metal rows.** `fda0a554` (base) and `b8082ae4` (Lappi), in `raw/mac-qd-metal-bench-2026-10-03.jsonl`.
- **Raw data.** In `raw/`: the tessl logs, `torch-mps-results.json`, and `task_lengths.json`.
- **Scripts.** They live in `~/qd-campaign/lappi-bench-2026-10-03/`:
  - `task_lengths.py`;
  - `torch_task_speed.py`, where the timed forward is `tools/mac_torch_attrib.py`'s `last_only`, verbatim;
  - `run_tessl.sh` and `run_torch.sh`, each with its gate and watchdog;
  - `summarize.py`.

  The lane-local build is in `target/` there.
