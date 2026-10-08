# Paired GPU-vs-torch summary: /Users/bharath/Code/research/ojas/.gitpulse/worktrees/inference-decode-path-per-token-uploads-350e80eb/target/bench-decode-before

3 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.

```
date: 2026-10-08T03:21:54Z
git: 8967457e26f439e18fa17413250ce8dd508af54a (dirty files: 0)
uncommitted diff of the benchmarked crates (sha1): da39a3ee5e6b
tessl: 4e5faac188b2087cbec2b25920ca5988b53ed103 (dirty files: 20)
rustc: rustc 1.99.0 (b940084d7 2026-09-28)
cargo: cargo 1.99.0 (5f94df478 2026-08-27)
os: macOS 27.0.1 (26A434)
cpu: Apple M5 Pro
memory_bytes: 68719476736
metal_bin: Oct  7 22:21:50 2026 4117248
wgpu_bin: Oct  7 22:21:42 2026 6434432
decode_bin: Oct  7 22:21:54 2026 10099568
rounds: 3  iters: 20  warmup: 5  rows: gen_  lanes: ojas-metal ojas-wgpu ojas-cpu ojas-cpu-host torch-mps
```

### ojas-metal vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gen_prefill_p32 | 10.63 / 11.34 | 7.352 / 8.389 | 0.73x [0.68-0.77] | 14% / 1% | noisy - not quoted | 7.87e-05 (4.4e-06) |
| gen_greedy_p32_n32 | 357.3 / 397.3 | 175.1 / 190.7 | 0.48x [0.48-0.48] | 1% / 1% |  | 7.87e-05 (4.4e-06) |

### ojas-wgpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gen_prefill_p32 | 30.60 / 32.76 | 7.352 / 8.389 | 0.25x [0.25-0.26] | 1% / 1% |  | 8.63e-05 (4.9e-06) |
| gen_greedy_p32_n32 | 890.2 / 948.0 | 175.1 / 190.7 | 0.20x [0.20-0.20] | 1% / 1% |  | 8.63e-05 (4.9e-06) |

### ojas-cpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gen_prefill_p32 | 12.52 / 12.83 | 7.352 / 8.389 | 0.65x [0.65-0.66] | 2% / 1% |  | 8.82e-05 (5.0e-06) |
| gen_greedy_p32_n32 | 160.4 / 170.6 | 175.1 / 190.7 | 1.11x [1.11-1.14] | 3% / 1% |  | 8.82e-05 (5.0e-06) |

### ojas-cpu-host vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gen_prefill_p32 | 1613.0 / 1620.3 | 7.352 / 8.389 | 0.01x [0.01-0.01] | 0% / 1% |  | 8.77e-05 (4.9e-06) |
| gen_greedy_p32_n32 | 3176.3 / 3193.4 | 175.1 / 190.7 | 0.06x [0.06-0.06] | 0% / 1% |  | 8.77e-05 (4.9e-06) |

### Ranked: where ojas is slowest relative to torch (lowest ratio first)

| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |
| --: | :-- | :-- | :-- | --: | --: | :-- |
| 1 | ojas-cpu-host | gen_prefill_p32 | 0.01x [0.01-0.01] | 1620.3 | 8.389 |  |
| 2 | ojas-cpu-host | gen_greedy_p32_n32 | 0.06x [0.06-0.06] | 3193.4 | 190.7 |  |
| 3 | ojas-wgpu | gen_greedy_p32_n32 | 0.20x [0.20-0.20] | 948.0 | 190.7 |  |
| 4 | ojas-wgpu | gen_prefill_p32 | 0.25x [0.25-0.26] | 32.76 | 8.389 |  |
| 5 | ojas-metal | gen_greedy_p32_n32 | 0.48x [0.48-0.48] | 397.3 | 190.7 |  |
| 6 | ojas-cpu | gen_prefill_p32 | 0.65x [0.65-0.66] | 12.83 | 8.389 |  |
| 7 | ojas-metal | gen_prefill_p32 | 0.73x [0.68-0.77] | 11.34 | 8.389 | noisy |
| 8 | ojas-cpu | gen_greedy_p32_n32 | 1.11x [1.11-1.14] | 170.6 | 190.7 |  |

### Machine load around each runtime block

| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |
| --: | :-- | :-- | :-- | :-- | --: |
| 1 | ojas-metal | before | 22:22:06 | 5.53 5.16 6.55 | 77 |
| 1 | ojas-metal | after | 22:22:17 | 5.39 5.14 6.52 | 94 |
| 1 | ojas-wgpu | before | 22:22:17 | 5.39 5.14 6.52 | 0 |
| 1 | ojas-wgpu | after | 22:22:44 | 5.49 5.17 6.50 | 95 |
| 1 | ojas-cpu | before | 22:22:44 | 5.49 5.17 6.50 | 38 |
| 1 | ojas-cpu | after | 22:22:50 | 5.69 5.22 6.50 | 64 |
| 1 | ojas-cpu-host | before | 22:22:50 | 5.69 5.22 6.50 | 69 |
| 1 | ojas-cpu-host | after | 22:24:57 | 5.06 5.13 6.28 | 68 |
| 1 | torch-mps | before | 22:24:57 | 5.06 5.13 6.28 | 72 |
| 1 | torch-mps | after | 22:25:06 | 5.45 5.21 6.30 | 40 |
| 2 | torch-mps | before | 22:25:07 | 5.45 5.21 6.30 | 36 |
| 2 | torch-mps | after | 22:25:16 | 5.22 5.17 6.27 | 28 |
| 2 | ojas-cpu-host | before | 22:25:16 | 5.22 5.17 6.27 | 0 |
| 2 | ojas-cpu-host | after | 22:27:23 | 4.22 4.87 6.00 | 70 |
| 2 | ojas-cpu | before | 22:27:23 | 4.22 4.87 6.00 | 72 |
| 2 | ojas-cpu | after | 22:27:28 | 4.20 4.85 5.99 | 68 |
| 2 | ojas-wgpu | before | 22:27:28 | 4.20 4.85 5.99 | 60 |
| 2 | ojas-wgpu | after | 22:27:55 | 3.71 4.69 5.90 | 96 |
| 2 | ojas-metal | before | 22:27:55 | 3.71 4.69 5.90 | 30 |
| 2 | ojas-metal | after | 22:28:06 | 3.75 4.65 5.86 | 94 |
| 3 | ojas-metal | before | 22:28:06 | 3.75 4.65 5.86 | 38 |
| 3 | ojas-metal | after | 22:28:18 | 3.55 4.58 5.82 | 94 |
| 3 | ojas-wgpu | before | 22:28:18 | 3.55 4.58 5.82 | 38 |
| 3 | ojas-wgpu | after | 22:28:44 | 3.66 4.52 5.76 | 95 |
| 3 | ojas-cpu | before | 22:28:44 | 3.66 4.52 5.76 | 0 |
| 3 | ojas-cpu | after | 22:28:50 | 3.60 4.49 5.75 | 64 |
| 3 | ojas-cpu-host | before | 22:28:50 | 3.60 4.49 5.75 | 58 |
| 3 | ojas-cpu-host | after | 22:30:57 | 4.47 4.59 5.61 | 69 |
| 3 | torch-mps | before | 22:30:57 | 4.47 4.59 5.61 | 75 |
| 3 | torch-mps | after | 22:31:06 | 4.47 4.58 5.59 | 39 |

### Decode (gen_ rows)

Per-token ms = (greedy median - prefill median) / (N - 1), median over rounds. Traffic is per decode step, from DeviceDecoder's counters (uploads / bytes up / bytes read back).

| runtime | prefill median ms | decode ms / token | tokens / s | rounds | step traffic | greedy ids equal to torch |
| :-- | --: | --: | --: | --: | :-- | :-- |
| torch-mps | 8.389 | 5.880 | 170.1 | 3 | - | - |
| ojas-metal | 11.34 | 12.42 | 80.5 | 3 | 3 / 516 B / 201216 B | 32/32 (prefix 32) |
| ojas-wgpu | 32.76 | 29.52 | 33.9 | 3 | 3 / 516 B / 201216 B | 32/32 (prefix 32) |
| ojas-cpu | 12.83 | 5.090 | 196.5 | 3 | 3 / 516 B / 201216 B | 32/32 (prefix 32) |
| ojas-cpu-host | 1620.3 | 50.74 | 19.7 | 3 | - | 32/32 (prefix 32) |

### Load recorded before and after every row

| round | lane | samples | 1-min load min-max | GPU util % min / median / max |
| --: | :-- | --: | :-- | :-- |
| 1 | ojas-cpu-host | 4 | 5.2-5.7 | 69 / 70 / 70 |
| 1 | ojas-cpu | 4 | 5.5-5.7 | 38 / 43 / 61 |
| 1 | ojas-metal | 4 | 5.4-5.5 | 0 / 80 / 95 |
| 1 | ojas-wgpu | 4 | 5.4-5.5 | 39 / 72 / 96 |
| 1 | torch-mps | 4 | 5.1-5.5 | 40 / 71 / 87 |
| 2 | ojas-cpu-host | 4 | 4.2-5.2 | 42 / 68 / 69 |
| 2 | ojas-cpu | 4 | 4.2-4.2 | 62 / 65 / 71 |
| 2 | ojas-metal | 4 | 3.7-4.0 | 29 / 70 / 95 |
| 2 | ojas-wgpu | 4 | 3.7-4.2 | 47 / 80 / 97 |
| 2 | torch-mps | 4 | 5.2-5.5 | 42 / 66 / 87 |
| 3 | ojas-cpu-host | 4 | 3.6-4.5 | 64 / 69 / 74 |
| 3 | ojas-cpu | 4 | 3.6-3.7 | 40 / 60 / 73 |
| 3 | ojas-metal | 4 | 3.5-3.8 | 0 / 64 / 95 |
| 3 | ojas-wgpu | 4 | 3.5-3.7 | 40 / 70 / 96 |
| 3 | torch-mps | 4 | 4.5-4.5 | 39 / 72 / 86 |
