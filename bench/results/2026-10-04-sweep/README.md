# Fixed-cost and decode-batching sweep, 2026-10-04

**Question.** Does batching short on-device requests move the point where the
fixed per-op cost stops dominating?

**Run.**
- `BENCH_ROWS=sweep_ bash bench/run_paired.sh 5`, driven by
  `target-baseline/sweep-run.sh` under the Lappi lead's machine lock
  (`tools/mac_heavy.sh`), 15:54:10–15:55:28Z.
- 5 rounds, alternating lane order; 5 warm-ups and 20 timed iterations per row.
- Rows are `sweep_*` in `bench/ojas_rows.rs` and `bench/torch_rows.py`; their
  semantics are in `bench/README.md`.
- Apple M5 Pro, torch 2.13.0. ojas HEAD 1e108dc, with the benchmarked crates
  clean (the diff hash is that of an empty diff). tessl 0ef5f6b, clean.
- Every row passed the parity gate on Metal and wgpu in every round.

**Caveat: every row is "noisy - not quoted" under the 10% spread gate.**
- Another process kept the GPU 54–79% busy for the whole run
  (`load.jsonl`). Load was 5.5–6.6.
- Per-round medians spread 18–154%. Absolute times are direction only.
- The paired per-round ratios in `summary.md` are tighter; ranges that do not
  overlap are quoted below as robust.

**Files.**
- `summary.md`: the standard paired table (`bench/aggregate.py`).
- `fit.md`: floor, slope and crossover per runtime (`sweep_fit.py`, analysis only).
- `round*/`, `env*`, `load.jsonl`: the raw record. `ref/` is not kept.

## Reading

**1. Where the fixed cost stops dominating (`fit.md`).**
- An op's time is a fixed part plus a part that grows with its size.
- Metal's fixed part: about 0.13 ms (min) to 0.18 ms (median). torch's: about
  0.10 to 0.16 ms.
- The size where the two parts are equal (`t(N) = 2·t(1)`):

  | runtime | from the minimum | from the median |
  | :-- | --: | --: |
  | Metal | 4.1M values | 4.5M values |
  | wgpu | 5.8M values | 6.0M values |
  | torch | 3.3M values | 5.2M values |

- That is about 16–18 MB of f32 per op on Metal. The model `a/b` agrees
  within 10% for Metal.
- Below about 1M values, Metal's time barely moves with size
  (0.12–0.17 ms min), so the fixed cost is nearly all of it.
- This replaces the 2–5M estimate given before the run. Its lower end assumed
  standalone finite-check passes, which were folded into the kernels on
  2026-10-02 (`docs/bench-gpu-vs-torch.md`, "Fix 2").

**2. Batching decode requests (`fit.md`, `summary.md`).**
- One batched call: Metal's cost per request falls from 0.214 ms at B=1 to
  0.051 ms at B=16, about 4x. torch falls from 0.168 to 0.046.
- Against torch, batching mostly closes the gap on Metal. The median ratio is
  0.93x at B=16 (per-round range 0.83–1.09).
- Sending the same requests as separate calls before one sync costs more.
  Metal at B=16: 1.342 ms split vs 0.815 batched.
- Each extra separate call costs Metal about 0.035–0.045 ms. torch's figure is
  0.031, 0.000, 0.012 and 0.012 ms at B = 2, 4, 8, 16, which is noisy. At
  B ≥ 8, ojas's per-call cost is roughly 3–4x torch's.
- Robust: at B=16, Metal's ratio to torch is 0.83–1.09 batched but 0.63–0.74
  split, and the ranges do not overlap. wgpu: 0.63–0.76 vs 0.38–0.49.

**3. What the extra per-call cost is: profiled in
[`../2026-10-04-percall/`](../2026-10-04-percall/README.md).**
- About 78% is GPU time. A single-request dispatch (12 threadgroups) reaches
  about 140–180 GB/s. The batched dispatch reaches the ~256 GB/s bandwidth
  ceiling.
- About 18% is host time, about 7 µs per call. Output allocation and
  residency re-registration are about 0.6 µs of that.
- An earlier version of this section guessed that residency
  re-registration was the cause, from reading the source. The profile shows
  it is not.

**Not measured.**
- A run on an idle GPU.
