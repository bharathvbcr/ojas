# Paired GPU-vs-torch summary: out/decode-after

3 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.

```
date: 2026-10-10T17:11:44Z
git: 2b68d11fd7341c1443d8f7133901f559c356d8c8 (dirty files: 0)
uncommitted diff of the benchmarked crates (sha1): da39a3ee5e6b
tessl: de4747a25d7edf22d3c3f44b1991b7ef7f446651 (dirty files: 0)
rustc: rustc 1.99.0 (b940084d7 2026-09-28)
cargo: cargo 1.99.0 (5f94df478 2026-08-27)
os: macOS 27.0.1 (26A434)
cpu: Apple M5 Pro
memory_bytes: 68719476736
metal_bin: Oct 10 09:57:09 2026 4544720
wgpu_bin: Oct 10 09:56:59 2026 6539072
decode_bin: Oct 10 10:11:44 2026 10819616
rounds: 3  iters: 20  warmup: 5  rows: gen_  lanes: ojas-metal ojas-wgpu ojas-cpu ojas-cpu-host torch-mps
```

### ojas-metal vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gen_prefill_p32 | 8.761 / 8.869 | 5.915 / 7.351 | 0.84x [0.75-0.92] | 6% / 29% | noisy - not quoted | 7.87e-05 (4.4e-06) |
| gen_greedy_p32_n32 | 275.1 / 279.2 | 149.0 / 155.9 | 0.56x [0.48-0.57] | 13% / 6% | noisy - not quoted | 7.87e-05 (4.4e-06) |
| gen_topk50_p32_n32 | 302.3 / 305.1 | 168.5 / 206.1 | 0.67x [0.61-0.77] | 1% / 26% | noisy - not quoted | 7.87e-05 (4.4e-06) |
| gen_gqa4_prefill_p32 | 8.533 / 8.631 | 6.037 / 6.642 | 0.77x [0.72-1.06] | 0% / 47% | noisy - not quoted | 9.44e-05 (5.6e-06) |
| gen_gqa4_greedy_p32_n32 | 271.5 / 279.7 | 145.9 / 159.4 | 0.54x [0.49-0.71] | 18% / 29% | noisy - not quoted | 9.44e-05 (5.6e-06) |

### ojas-wgpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gen_prefill_p32 | 26.36 / 27.37 | 5.915 / 7.351 | 0.25x [0.11-0.27] | 184% / 29% | noisy - not quoted | 8.63e-05 (4.9e-06) |
| gen_greedy_p32_n32 | 739.3 / 744.7 | 149.0 / 155.9 | 0.21x [0.20-0.21] | 1% / 6% |  | 8.63e-05 (4.9e-06) |
| gen_topk50_p32_n32 | 780.2 / 811.9 | 168.5 / 206.1 | 0.24x [0.24-0.29] | 8% / 26% | noisy - not quoted | 8.63e-05 (4.9e-06) |
| gen_gqa4_prefill_p32 | 26.53 / 27.34 | 6.037 / 6.642 | 0.24x [0.22-0.34] | 7% / 47% | noisy - not quoted | 8.23e-05 (4.8e-06) |
| gen_gqa4_greedy_p32_n32 | 741.8 / 769.1 | 145.9 / 159.4 | 0.21x [0.19-0.25] | 5% / 29% | noisy - not quoted | 8.23e-05 (4.8e-06) |

### ojas-cpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gen_prefill_p32 | 11.31 / 22.14 | 5.915 / 7.351 | 0.36x [0.30-0.39] | 9% / 29% | noisy - not quoted | 8.82e-05 (5.0e-06) |
| gen_greedy_p32_n32 | 183.7 / 236.2 | 149.0 / 155.9 | 0.63x [0.63-0.71] | 15% / 6% | noisy - not quoted | 8.82e-05 (5.0e-06) |
| gen_topk50_p32_n32 | 129.6 / 216.1 | 168.5 / 206.1 | 0.95x [0.64-1.52] | 89% / 26% | noisy - not quoted | 8.82e-05 (5.0e-06) |
| gen_gqa4_prefill_p32 | 10.45 / 23.01 | 6.037 / 6.642 | 0.29x [0.27-0.81] | 104% / 47% | noisy - not quoted | 8.92e-05 (5.2e-06) |
| gen_gqa4_greedy_p32_n32 | 130.8 / 175.3 | 145.9 / 159.4 | 0.94x [0.66-1.10] | 34% / 29% | noisy - not quoted | 8.92e-05 (5.2e-06) |

### ojas-cpu-host vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gen_prefill_p32 | 17.18 / 22.99 | 5.915 / 7.351 | 0.32x [0.26-0.43] | 30% / 29% | noisy - not quoted | 8.82e-05 (5.0e-06) |
| gen_greedy_p32_n32 | 138.3 / 187.1 | 149.0 / 155.9 | 0.83x [0.70-0.97] | 32% / 6% | noisy - not quoted | 8.82e-05 (5.0e-06) |
| gen_topk50_p32_n32 | 141.2 / 198.1 | 168.5 / 206.1 | 1.14x [1.04-1.16] | 28% / 26% | noisy - not quoted | 8.82e-05 (5.0e-06) |
| gen_gqa4_prefill_p32 | 10.99 / 12.21 | 6.037 / 6.642 | 0.54x [0.39-0.55] | 108% / 47% | noisy - not quoted | 8.92e-05 (5.2e-06) |
| gen_gqa4_greedy_p32_n32 | 125.4 / 182.6 | 145.9 / 159.4 | 0.87x [0.87-1.16] | 72% / 29% | noisy - not quoted | 8.92e-05 (5.2e-06) |

### Ranked: where ojas is slowest relative to torch (lowest ratio first)

| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |
| --: | :-- | :-- | :-- | --: | --: | :-- |
| 1 | ojas-wgpu | gen_greedy_p32_n32 | 0.21x [0.20-0.21] | 744.7 | 155.9 |  |
| 2 | ojas-wgpu | gen_gqa4_greedy_p32_n32 | 0.21x [0.19-0.25] | 769.1 | 159.4 | noisy |
| 3 | ojas-wgpu | gen_topk50_p32_n32 | 0.24x [0.24-0.29] | 811.9 | 206.1 | noisy |
| 4 | ojas-wgpu | gen_gqa4_prefill_p32 | 0.24x [0.22-0.34] | 27.34 | 6.642 | noisy |
| 5 | ojas-wgpu | gen_prefill_p32 | 0.25x [0.11-0.27] | 27.37 | 7.351 | noisy |
| 6 | ojas-cpu | gen_gqa4_prefill_p32 | 0.29x [0.27-0.81] | 23.01 | 6.642 | noisy |
| 7 | ojas-cpu-host | gen_prefill_p32 | 0.32x [0.26-0.43] | 22.99 | 7.351 | noisy |
| 8 | ojas-cpu | gen_prefill_p32 | 0.36x [0.30-0.39] | 22.14 | 7.351 | noisy |
| 9 | ojas-metal | gen_gqa4_greedy_p32_n32 | 0.54x [0.49-0.71] | 279.7 | 159.4 | noisy |
| 10 | ojas-cpu-host | gen_gqa4_prefill_p32 | 0.54x [0.39-0.55] | 12.21 | 6.642 | noisy |
| 11 | ojas-metal | gen_greedy_p32_n32 | 0.56x [0.48-0.57] | 279.2 | 155.9 | noisy |
| 12 | ojas-cpu | gen_greedy_p32_n32 | 0.63x [0.63-0.71] | 236.2 | 155.9 | noisy |
| 13 | ojas-metal | gen_topk50_p32_n32 | 0.67x [0.61-0.77] | 305.1 | 206.1 | noisy |
| 14 | ojas-metal | gen_gqa4_prefill_p32 | 0.77x [0.72-1.06] | 8.631 | 6.642 | noisy |
| 15 | ojas-cpu-host | gen_greedy_p32_n32 | 0.83x [0.70-0.97] | 187.1 | 155.9 | noisy |
| 16 | ojas-metal | gen_prefill_p32 | 0.84x [0.75-0.92] | 8.869 | 7.351 | noisy |
| 17 | ojas-cpu-host | gen_gqa4_greedy_p32_n32 | 0.87x [0.87-1.16] | 182.6 | 159.4 | noisy |
| 18 | ojas-cpu | gen_gqa4_greedy_p32_n32 | 0.94x [0.66-1.10] | 175.3 | 159.4 | noisy |
| 19 | ojas-cpu | gen_topk50_p32_n32 | 0.95x [0.64-1.52] | 216.1 | 206.1 | noisy |
| 20 | ojas-cpu-host | gen_topk50_p32_n32 | 1.14x [1.04-1.16] | 198.1 | 206.1 | noisy |

### Machine load around each runtime block

| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |
| --: | :-- | :-- | :-- | :-- | --: |
| 1 | ojas-metal | before | 10:12:03 | 5.83 9.26 13.54 | 98 |
| 1 | ojas-metal | after | 10:12:29 | 8.50 9.55 13.52 | 88 |
| 1 | ojas-wgpu | before | 10:12:29 | 8.50 9.55 13.52 | 0 |
| 1 | ojas-wgpu | after | 10:13:58 | 17.60 11.34 13.75 | 94 |
| 1 | ojas-cpu | before | 10:13:58 | 17.60 11.34 13.75 | 0 |
| 1 | ojas-cpu | after | 10:14:16 | 24.52 13.34 14.41 | 0 |
| 1 | ojas-cpu-host | before | 10:14:16 | 24.52 13.34 14.41 | 0 |
| 1 | ojas-cpu-host | after | 10:14:32 | 26.98 14.45 14.79 | 0 |
| 1 | torch-mps | before | 10:14:32 | 26.98 14.45 14.79 | 0 |
| 1 | torch-mps | after | 10:14:57 | 25.71 15.23 15.06 | 65 |
| 2 | torch-mps | before | 10:14:57 | 25.71 15.23 15.06 | 0 |
| 2 | torch-mps | after | 10:15:23 | 22.54 15.29 15.09 | 75 |
| 2 | ojas-cpu-host | before | 10:15:23 | 22.54 15.29 15.09 | 82 |
| 2 | ojas-cpu-host | after | 10:15:42 | 22.85 15.84 15.29 | 83 |
| 2 | ojas-cpu | before | 10:15:42 | 22.85 15.84 15.29 | 85 |
| 2 | ojas-cpu | after | 10:16:01 | 22.87 16.29 15.46 | 95 |
| 2 | ojas-wgpu | before | 10:16:01 | 22.87 16.29 15.46 | 100 |
| 2 | ojas-wgpu | after | 10:17:12 | 12.98 14.62 14.89 | 94 |
| 2 | ojas-metal | before | 10:17:12 | 12.98 14.62 14.89 | 0 |
| 2 | ojas-metal | after | 10:17:37 | 10.42 13.91 14.63 | 92 |
| 3 | ojas-metal | before | 10:17:37 | 10.42 13.91 14.63 | 0 |
| 3 | ojas-metal | after | 10:18:01 | 7.87 13.02 14.28 | 91 |
| 3 | ojas-wgpu | before | 10:18:01 | 7.87 13.02 14.28 | 0 |
| 3 | ojas-wgpu | after | 10:19:08 | 14.87 13.60 14.38 | 93 |
| 3 | ojas-cpu | before | 10:19:08 | 14.87 13.60 14.38 | 0 |
| 3 | ojas-cpu | after | 10:19:31 | 24.41 15.92 15.19 | 0 |
| 3 | ojas-cpu-host | before | 10:19:31 | 24.41 15.92 15.19 | 0 |
| 3 | ojas-cpu-host | after | 10:19:45 | 21.65 15.61 15.09 | 0 |
| 3 | torch-mps | before | 10:19:45 | 21.65 15.61 15.09 | 0 |
| 3 | torch-mps | after | 10:20:09 | 16.35 14.84 14.82 | 0 |

### Decode (gen_ rows)

Per-token ms = (decode median - prefill median) / (N - 1), median over rounds, each decode row against its model's prefill row. Traffic is per decode step, from DeviceDecoder's counters (uploads / bytes up / bytes read back).

| runtime | row | prefill median ms | decode ms / token | tokens / s | rounds | step traffic | greedy ids equal to torch |
| :-- | :-- | --: | --: | --: | --: | :-- | :-- |
| torch-mps | gen_gqa4_greedy_p32_n32 | 6.642 | 4.927 | 203.0 | 3 | - | - |
| torch-mps | gen_greedy_p32_n32 | 7.351 | 4.791 | 208.7 | 3 | - | - |
| torch-mps | gen_topk50_p32_n32 | 7.351 | 6.411 | 156.0 | 3 | - | - |
| ojas-metal | gen_gqa4_greedy_p32_n32 | 8.631 | 8.744 | 114.4 | 3 | 0 / 0 B / 4 B | 32/32 (prefix 32) |
| ojas-metal | gen_greedy_p32_n32 | 8.869 | 8.722 | 114.7 | 3 | 0 / 0 B / 4 B | 32/32 (prefix 32) |
| ojas-metal | gen_topk50_p32_n32 | 8.869 | 9.556 | 104.6 | 3 | 1 / 4 B / 400 B | - |
| ojas-wgpu | gen_gqa4_greedy_p32_n32 | 27.34 | 23.93 | 41.8 | 3 | 0 / 0 B / 4 B | 32/32 (prefix 32) |
| ojas-wgpu | gen_greedy_p32_n32 | 27.37 | 23.11 | 43.3 | 3 | 0 / 0 B / 4 B | 32/32 (prefix 32) |
| ojas-wgpu | gen_topk50_p32_n32 | 27.37 | 24.49 | 40.8 | 3 | 1 / 4 B / 400 B | - |
| ojas-cpu | gen_gqa4_greedy_p32_n32 | 23.01 | 5.289 | 189.1 | 3 | 0 / 0 B / 0 B | 32/32 (prefix 32) |
| ojas-cpu | gen_greedy_p32_n32 | 22.14 | 6.906 | 144.8 | 3 | 0 / 0 B / 0 B | 32/32 (prefix 32) |
| ojas-cpu | gen_topk50_p32_n32 | 22.14 | 6.306 | 158.6 | 3 | 1 / 4 B / 0 B | - |
| ojas-cpu-host | gen_gqa4_greedy_p32_n32 | 12.21 | 5.497 | 181.9 | 3 | - | 32/32 (prefix 32) |
| ojas-cpu-host | gen_greedy_p32_n32 | 22.99 | 5.294 | 188.9 | 3 | - | 32/32 (prefix 32) |
| ojas-cpu-host | gen_topk50_p32_n32 | 22.99 | 5.650 | 177.0 | 3 | - | - |

Reading one `[1, vocab]` logit row back (last round, min of `iters`): `Tensor::to_host` against `DeviceBuffer::read_bytes` alone; the difference is the transient decode pass a read into a caller's buffer would save.

| runtime | bytes | to_host us | read_bytes us | decode pass us | iters |
| :-- | --: | --: | --: | --: | --: |
| ojas-metal | 201216 | 13.71 | 4.33 | 9.38 | 50 |
| ojas-wgpu | 201216 | 252.04 | 229.62 | 22.42 | 50 |

### Load recorded before and after every row

| round | lane | samples | 1-min load min-max | GPU util % min / median / max |
| --: | :-- | --: | :-- | :-- |
| 1 | ojas-cpu-host | 10 | 24.5-27.0 | 0 / 0 / 0 |
| 1 | ojas-cpu | 10 | 17.6-24.5 | 0 / 0 / 25 |
| 1 | ojas-metal | 10 | 5.5-8.5 | 0 / 90 / 94 |
| 1 | ojas-wgpu | 10 | 7.2-17.6 | 0 / 93 / 99 |
| 1 | torch-mps | 10 | 25.7-28.9 | 0 / 34 / 92 |
| 2 | ojas-cpu-host | 10 | 22.5-23.0 | 0 / 58 / 96 |
| 2 | ojas-cpu | 10 | 21.5-23.0 | 0 / 82 / 96 |
| 2 | ojas-metal | 10 | 10.4-13.0 | 0 / 86 / 93 |
| 2 | ojas-wgpu | 10 | 13.0-22.9 | 0 / 93 / 100 |
| 2 | torch-mps | 10 | 22.4-25.7 | 0 / 70 / 92 |
| 3 | ojas-cpu-host | 10 | 21.6-24.4 | 0 / 30 / 43 |
| 3 | ojas-cpu | 10 | 14.9-23.2 | 0 / 0 / 21 |
| 3 | ojas-metal | 10 | 7.9-10.4 | 0 / 90 / 93 |
| 3 | ojas-wgpu | 10 | 6.7-14.9 | 0 / 89 / 95 |
| 3 | torch-mps | 10 | 16.4-20.5 | 0 / 45 / 91 |
