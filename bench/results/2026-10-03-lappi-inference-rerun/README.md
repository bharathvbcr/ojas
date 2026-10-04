# Lappi vs base Qwen3.5-2B on the Mac, rerun on the faster decision path

2026-10-03, 18:31–18:42Z, Apple M5 Pro. **Speed only**: no accuracy was measured. Labels: **[V]** measured here, **[I]** inferred.

This reruns [this morning's benchmark](../2026-10-03-lappi-inference/README.md) with the same models, prompts and engines. Two things changed:

- **The tessl decision path.** It is now the `qdm-digest-parallel` build (Lappi `75c683b`), which hashes state with the CPU's SHA-256 instructions instead of the one-thread software hash.
- **Run order.** Base and Lappi ran in ABBA order (base, Lappi, Lappi, base). Each figure below is the min over a model's two runs.

## What was compared

| | |
|---|---|
| Original model | Qwen/Qwen3.5-2B-Base, snapshot `b1485b2f`, weight hash `92f6bd1c…` [V] |
| Latest Lappi | `p4-v4-avg-masters-v1`, the 5-seed masters average of run F; weight hash `a68f19bc…` [V]. No newer release exists: it is the only release manifest on this Mac or named in Lappi main's recent handoffs [V] |
| Engines | **tessl** through qd-metal (Lappi's Mac backend), and **PyTorch 2.12.1 MPS** in bf16 with sdpa |
| Not run: ojas | ojas still has no Qwen3.5 inference path. `ojas-qwen35` is a training-step provider, and `ojas-infer` serves only the nanolab GPT [V] |
| Tasks | The same prompts from run F's val shards (`ef06ab99…`), at each family's p50 and p90 length, plus 2K and 8K long context |

The bench binary is `767be22f…`. A cargo rebuild from the branch head inside the slot produced the identical binary, so the binary is the source as committed [V].

## Results (min ms; lower is better)

| Task | Point | Tokens | tessl prefill base / Lappi | PyTorch MPS base / Lappi | PyTorch ÷ tessl | tessl decision base / Lappi | This morning's decision |
|---|---|---|---|---|---|---|---|
| CLINC intent (16-way) | p50 | 200 | 44.0 / 43.9 | 100.2 / 98.4 | 2.26× | 99 / 100 | 217 / 216 |
| CLINC intent (16-way) | p90 | 207 | 44.0 / 43.9 | 98.1 / 100.1 | 2.26× | 99 / 100 | 217 / 216 |
| CLINC domain (10-way) | p50 | 167 | 41.3 / 41.6 | 94.5 / 87.4 | 2.19× | 96 / 98 | 210 / 209 |
| CLINC domain (10-way) | p90 | 171 | 41.3 / 41.6 | 90.6 / 91.4 | 2.20× | 96 / 98 | 210 / 209 |
| CLINC in-scope (2-way) | p50 | 131 | 39.6 / 39.9 | 133.1 / 125.7 | 3.25× | 83 / 83 | 190 / 189 |
| CLINC in-scope (2-way) | p90 | 135 | 39.6 / 39.9 | 148.5 / 151.5 | 3.77× | 83 / 83 | 190 / 189 |
| CLINC within-domain (15-way) | p50 | 196 | 44.0 / 43.9 | 98.3 / 98.5 | 2.24× | 99 / 100 | 217 / 216 |
| CLINC within-domain (15-way) | p90 | 203 | 44.0 / 43.9 | 97.2 / 97.9 | 2.22× | 99 / 100 | 217 / 216 |
| CSQA (5-way) | p50 | 151 | 40.1 / 40.0 | 137.3 / 134.6 | 3.39× | 97 / 94 | 207 / 207 |
| CSQA (5-way) | p90 | 161 | 41.3 / 41.6 | 95.2 / 98.5 | 2.34× | 96 / 98 | 210 / 209 |
| MMLU (4-way) | p50 | 184 | 42.7 / 42.0 | 112.4 / 111.9 | 2.65× | 97 / 97 | 211 / 210 |
| MMLU (4-way) | p90 | 331 | 64.4 / 64.1 | 139.2 / 140.0 | 2.17× | 122 / 123 | 249 / 250 |
| Code defect class (4-way) | p50 | 409 | 83.7 / 83.0 | 177.1 / 179.1 | 2.14× | 146 / 153 | 279 / 281 |
| Code defect class (4-way) | p90 | 770 | 150.9 / 145.7 | 302.3 / 306.8 | 2.05× | 217 / 218 | 377 / 377 |
| Code defect span | p50 | 400 | 83.7 / 83.0 | 151.2 / 150.9 | 1.81× | 146 / 153 | 279 / 281 |
| Code defect span | p90 | 763 | 150.9 / 145.7 | 262.8 / 263.1 | 1.77× | 217 / 218 | 377 / 377 |
| SQuAD answer span | p50 | 276 | 60.3 / 60.0 | 111.4 / 111.5 | 1.85× | 117 / 118 | 244 / 241 |
| SQuAD answer span | p90 | 376 | 83.7 / 83.0 | 140.5 / 140.9 | 1.69× | 146 / 153 | 279 / 281 |
| Long context | 2K | 2048 | 361.7 / 369.2 | 744.8 / 756.0 | 2.05× | 497 / 507 | 763 / 773 |
| Long context | 8K | 8192 | 2180.3 / 2235.0 | 3589.3 / 3605.7 | 1.63× | 2602 / 2583 | 3276 / 3293 |

How to read the columns:
- **tessl prefill** and **PyTorch MPS** time the same work: one prefill plus the 17 answer-letter rows at the last position.
  - tessl was timed at a fixed list of lengths. Where a task's length isn't on it, the nearest is used: 200 for 196–207, 167 for 161–171, 131 for 135, 409 for 376–400, and 770 for 763.
- **tessl decision** is the product path: prefill, two read-only option passes of 61 tokens each, three state digests and readback.
  - Its decision bench uses a frozen context, so the actual prefix runs a little shorter than the label:
    - 115 tokens at 131;
    - 165 at 167 and 184;
    - 193 at 200;
    - 310 at 331;
    - 399 at 409;
    - 762 at 770.
  - The per-T breakdown is in `raw/summary.txt`.

## Findings

1. **Lappi costs the same as the base model at inference [V].** Lappi ÷ base:

   | Path | Ratio range |
   |---|---|
   | tessl prefill | 0.97–1.03 |
   | tessl decision | 0.97–1.05 |
   | PyTorch | 0.92–1.04, median 1.00 |

   They share an architecture and differ only in weights. The 1.05 at T=409 is within the ABBA spread, where one model's two runs differ by up to 30%.
2. **A tessl decision is now about 2× faster than this morning [V].**

   | Prompt length | Then | Now |
   |---|---|---|
   | 131–200 tokens | 190–217 ms | 83–100 ms |
   | 409 tokens | 280 ms | 146–153 ms |
   | 770 tokens | 377 ms | 217 ms |
   | 2K | 1.5× faster | |
   | 8K | 1.27× faster | |

   - **What it costs now.** For example at T=200, using the median of one Lappi run: prefill 45 ms, the two option passes 50 ms, and the state digests 6 ms. This morning the digests cost 120+ ms at that length.
   - **What dominates.** At task lengths the time is now the model itself: prefill plus a ~50 ms floor for the option passes.
3. **tessl is still the faster engine everywhere [V].**
   - **1.7–2.7×** PyTorch MPS on the 160–800-token task prompts, **2.0×** at 2K and **1.6×** at 8K.
   - **3.3–3.8×** at 131–151 tokens: PyTorch's known slow step at those shapes (`GAP-TORCH-MPS-STEP-AT-131-151-TOKENS-2026-10-03`) is still there.
   - tessl loads in 2–3 s.
4. **The base and Lappi argmax letters agree on 9 of 20 prompts**, the same as this morning. That is expected: they are different weights, and it is not an accuracy measure.

## Caveats

- **The Mac was busier than this morning [V].** Load ran 4–9 (this morning 3–5), from other sessions' work.
  - Absolute times are 5–15% slower than this morning on paths the code change didn't touch: PyTorch's 2K went 640 → 745 ms, and tessl's prefill-only 8K went 1988 → 2180 ms.
  - So the decision-path speedup in finding 2 is, if anything, understated [I].
- **Drift inside the hold [V].** The fourth run, base decision, was 20–30% slower than the first at every T. Each figure takes the min of a model's two runs, which keeps that drift out of the comparison. Run-by-run numbers are in `raw/summary.txt`. Thermal throttling as the cause is [I].
- **No accuracy, and one hold per engine.** The decision ledger rows (4, in `raw/mac-qd-metal-bench-2026-10-03b.jsonl`) are `quick`.

## Provenance

- **Slot.** One `mac_heavy` hold, `bench-lappi-inference-rerun`, 18:31:56–18:41:47Z.
  - Each step passed the busy gate.
  - Peak 1-minute load was 8.62.
  - PyTorch peaked at 9.4 GB RSS and 14.4 GB of MPS memory.
- **Writes.** Nothing was written to the Lappi tree or its ledger. The decision rows went to this run's own ledger file.
- **Scripts.** `~/qd-campaign/lappi-bench-2026-10-03b/`: `run_all.sh` and `summarize.py`. They reuse this morning's `torch_task_speed.py` and `task_lengths.json`, unchanged.
