# torch `clip_grad_norm_` on MPS: 2.14.1 against the 2.15 nightly (2026-10-05)

## Why

The GPU rounds found ojas faster than torch on `clip_grad_norm_full`: 4.58× in round 2 and 8.13× in round 5. torch's side of that row goes through `linalg.vector_norm`. Up to torch 2.14, MPS ran each full norm reduction on one threadgroup. pytorch/pytorch#198611 ("[MPS] Port norm to the shared reduction kernels", merged 2026-09-25) fixes that for torch 2.15.

## What was run (verified)

- `BENCH_ROWS=clip_grad_norm_full bench/torch_rows.py time --iters 20`, warm-up 5. That is the torch side of `run_paired.sh`, for one row.
- Runs alternated: 2.14.1, nightly, 2.14.1, nightly, 2.14.1, nightly.
- torch 2.14.1 and 2.15.0.dev20261004 (git b82cf81), MPS, Apple M5 Pro, macOS 27.0.1.
- The ojas side was **not** re-run. The checkout did not build at the time (`ojas-cpu` calls `ojas_simd` functions that are not yet in the tree).
- The GPU read 54–70% busy before each run, so the medians are direction only.

## Results (`torch.jsonl`)

| torch | run 1 median | run 2 median | run 3 median | min of mins |
| :--- | ---: | ---: | ---: | ---: |
| 2.14.1 | 137.46 ms | 137.57 ms | 137.51 ms | 136.78 ms |
| 2.15.0.dev20261004 | 8.36 ms | 8.54 ms | 8.45 ms | 7.90 ms |

2.14.1 matches round 3's torch 2.13 median of 137.4 ms, so the slow path is unchanged through 2.14.1. The nightly is about 16× faster.

## Against ojas (unpaired, direction only)

Round 5 (`bench/results/2026-10-02-r5/summary.md`) measured ojas at 18.96 ms median on Metal and 19.12 ms on wgpu.

Against the nightly's 8.45 ms, ojas is about 0.45× torch on this row. The lead in the earlier rounds comes from the old torch kernel. A paired run with torch 2.15 is needed before the new ratio is quoted.
