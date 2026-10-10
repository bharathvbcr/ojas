# Paired GPU-vs-torch summary: out/decode-before

3 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.

```
date: 2026-10-10T17:03:58Z
git: de5efafc9895915a1f22acfea3d7910b6abb31af (dirty files: 0)
uncommitted diff of the benchmarked crates (sha1): da39a3ee5e6b
tessl: de4747a25d7edf22d3c3f44b1991b7ef7f446651 (dirty files: 0)
rustc: rustc 1.99.0 (b940084d7 2026-09-28)
cargo: cargo 1.99.0 (5f94df478 2026-08-27)
os: macOS 27.0.1 (26A434)
cpu: Apple M5 Pro
memory_bytes: 68719476736
metal_bin: Oct 10 09:57:09 2026 4544720
wgpu_bin: Oct 10 09:56:59 2026 6539072
decode_bin: Oct 10 10:03:58 2026 10761440
rounds: 3  iters: 20  warmup: 5  rows: gen_  lanes: ojas-metal ojas-wgpu ojas-cpu ojas-cpu-host torch-mps
```

### ojas-metal vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gen_prefill_p32 | 9.116 / 10.15 | 6.075 / 6.790 | 0.65x [0.47-0.69] | 71% / 19% | noisy - not quoted | 7.87e-05 (4.4e-06) |
| gen_greedy_p32_n32 | 299.2 / 315.6 | 151.7 / 156.0 | 0.51x [0.49-0.57] | 6% / 17% | noisy - not quoted | 7.87e-05 (4.4e-06) |
| gen_topk50_p32_n32 | 284.8 / 341.2 | 179.6 / 215.3 | 0.64x [0.58-0.75] | 28% / 38% | noisy - not quoted | 7.87e-05 (4.4e-06) |
| gen_gqa4_prefill_p32 | 9.173 / 11.88 | 6.072 / 6.730 | 0.56x [0.17-0.88] | 326% / 23% | noisy - not quoted | 9.44e-05 (5.6e-06) |
| gen_gqa4_greedy_p32_n32 | 272.3 / 287.6 | 149.5 / 174.5 | 0.61x [0.55-0.63] | 9% / 5% |  | 9.44e-05 (5.6e-06) |

### ojas-wgpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gen_prefill_p32 | 26.47 / 26.71 | 6.075 / 6.790 | 0.25x [0.25-0.30] | 4% / 19% | noisy - not quoted | 8.63e-05 (4.9e-06) |
| gen_greedy_p32_n32 | 743.6 / 768.6 | 151.7 / 156.0 | 0.20x [0.20-0.24] | 2% / 17% | noisy - not quoted | 8.63e-05 (4.9e-06) |
| gen_topk50_p32_n32 | 746.0 / 756.6 | 179.6 / 215.3 | 0.27x [0.24-0.34] | 4% / 38% | noisy - not quoted | 8.63e-05 (4.9e-06) |
| gen_gqa4_prefill_p32 | 26.57 / 29.09 | 6.072 / 6.730 | 0.23x [0.21-0.31] | 19% / 23% | noisy - not quoted | 8.23e-05 (4.8e-06) |
| gen_gqa4_greedy_p32_n32 | 741.2 / 766.8 | 149.5 / 174.5 | 0.23x [0.21-0.24] | 4% / 5% |  | 8.23e-05 (4.8e-06) |

### ojas-cpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gen_prefill_p32 | 11.27 / 11.69 | 6.075 / 6.790 | 0.58x [0.57-0.65] | 6% / 19% | noisy - not quoted | 8.82e-05 (5.0e-06) |
| gen_greedy_p32_n32 | 133.5 / 141.5 | 151.7 / 156.0 | 1.11x [1.04-1.28] | 6% / 17% | noisy - not quoted | 8.82e-05 (5.0e-06) |
| gen_topk50_p32_n32 | 130.5 / 142.4 | 179.6 / 215.3 | 1.51x [1.33-1.75] | 4% / 38% | noisy - not quoted | 8.82e-05 (5.0e-06) |
| gen_gqa4_prefill_p32 | 10.42 / 11.46 | 6.072 / 6.730 | 0.63x [0.57-0.72] | 12% / 23% | noisy - not quoted | 8.92e-05 (5.2e-06) |
| gen_gqa4_greedy_p32_n32 | 116.9 / 129.9 | 149.5 / 174.5 | 1.29x [1.28-1.43] | 12% / 5% | noisy - not quoted | 8.92e-05 (5.2e-06) |

### ojas-cpu-host vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gen_prefill_p32 | 11.92 / 12.61 | 6.075 / 6.790 | 0.54x [0.54-0.55] | 21% / 19% | noisy - not quoted | 8.82e-05 (5.0e-06) |
| gen_greedy_p32_n32 | 129.7 / 141.5 | 151.7 / 156.0 | 1.19x [1.10-1.25] | 11% / 17% | noisy - not quoted | 8.82e-05 (5.0e-06) |
| gen_topk50_p32_n32 | 129.9 / 143.9 | 179.6 / 215.3 | 1.47x [1.40-1.77] | 11% / 38% | noisy - not quoted | 8.82e-05 (5.0e-06) |
| gen_gqa4_prefill_p32 | 10.34 / 11.22 | 6.072 / 6.730 | 0.62x [0.59-0.73] | 4% / 23% | noisy - not quoted | 8.92e-05 (5.2e-06) |
| gen_gqa4_greedy_p32_n32 | 122.2 / 124.9 | 149.5 / 174.5 | 1.33x [1.30-1.41] | 8% / 5% |  | 8.92e-05 (5.2e-06) |

### Ranked: where ojas is slowest relative to torch (lowest ratio first)

| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |
| --: | :-- | :-- | :-- | --: | --: | :-- |
| 1 | ojas-wgpu | gen_greedy_p32_n32 | 0.20x [0.20-0.24] | 768.6 | 156.0 | noisy |
| 2 | ojas-wgpu | gen_gqa4_greedy_p32_n32 | 0.23x [0.21-0.24] | 766.8 | 174.5 |  |
| 3 | ojas-wgpu | gen_gqa4_prefill_p32 | 0.23x [0.21-0.31] | 29.09 | 6.730 | noisy |
| 4 | ojas-wgpu | gen_prefill_p32 | 0.25x [0.25-0.30] | 26.71 | 6.790 | noisy |
| 5 | ojas-wgpu | gen_topk50_p32_n32 | 0.27x [0.24-0.34] | 756.6 | 215.3 | noisy |
| 6 | ojas-metal | gen_greedy_p32_n32 | 0.51x [0.49-0.57] | 315.6 | 156.0 | noisy |
| 7 | ojas-cpu-host | gen_prefill_p32 | 0.54x [0.54-0.55] | 12.61 | 6.790 | noisy |
| 8 | ojas-metal | gen_gqa4_prefill_p32 | 0.56x [0.17-0.88] | 11.88 | 6.730 | noisy |
| 9 | ojas-cpu | gen_prefill_p32 | 0.58x [0.57-0.65] | 11.69 | 6.790 | noisy |
| 10 | ojas-metal | gen_gqa4_greedy_p32_n32 | 0.61x [0.55-0.63] | 287.6 | 174.5 |  |
| 11 | ojas-cpu-host | gen_gqa4_prefill_p32 | 0.62x [0.59-0.73] | 11.22 | 6.730 | noisy |
| 12 | ojas-cpu | gen_gqa4_prefill_p32 | 0.63x [0.57-0.72] | 11.46 | 6.730 | noisy |
| 13 | ojas-metal | gen_topk50_p32_n32 | 0.64x [0.58-0.75] | 341.2 | 215.3 | noisy |
| 14 | ojas-metal | gen_prefill_p32 | 0.65x [0.47-0.69] | 10.15 | 6.790 | noisy |
| 15 | ojas-cpu | gen_greedy_p32_n32 | 1.11x [1.04-1.28] | 141.5 | 156.0 | noisy |
| 16 | ojas-cpu-host | gen_greedy_p32_n32 | 1.19x [1.10-1.25] | 141.5 | 156.0 | noisy |
| 17 | ojas-cpu | gen_gqa4_greedy_p32_n32 | 1.29x [1.28-1.43] | 129.9 | 174.5 | noisy |
| 18 | ojas-cpu-host | gen_gqa4_greedy_p32_n32 | 1.33x [1.30-1.41] | 124.9 | 174.5 |  |
| 19 | ojas-cpu-host | gen_topk50_p32_n32 | 1.47x [1.40-1.77] | 143.9 | 215.3 | noisy |
| 20 | ojas-cpu | gen_topk50_p32_n32 | 1.51x [1.33-1.75] | 142.4 | 215.3 | noisy |

### Machine load around each runtime block

| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |
| --: | :-- | :-- | :-- | :-- | --: |
| 1 | ojas-metal | before | 10:04:16 | 10.10 15.60 17.58 | 90 |
| 1 | ojas-metal | after | 10:04:48 | 10.27 15.26 17.39 | 93 |
| 1 | ojas-wgpu | before | 10:04:48 | 10.27 15.26 17.39 | 0 |
| 1 | ojas-wgpu | after | 10:05:53 | 10.42 14.37 16.90 | 87 |
| 1 | ojas-cpu | before | 10:05:53 | 10.42 14.37 16.90 | 0 |
| 1 | ojas-cpu | after | 10:06:05 | 9.05 13.87 16.68 | 12 |
| 1 | ojas-cpu-host | before | 10:06:05 | 9.05 13.87 16.68 | 0 |
| 1 | ojas-cpu-host | after | 10:06:17 | 9.10 13.73 16.60 | 0 |
| 1 | torch-mps | before | 10:06:17 | 9.10 13.73 16.60 | 0 |
| 1 | torch-mps | after | 10:06:41 | 8.09 13.10 16.28 | 29 |
| 2 | torch-mps | before | 10:06:41 | 8.09 13.10 16.28 | 54 |
| 2 | torch-mps | after | 10:07:07 | 8.60 12.83 16.09 | 0 |
| 2 | ojas-cpu-host | before | 10:07:07 | 8.60 12.83 16.09 | 0 |
| 2 | ojas-cpu-host | after | 10:07:19 | 8.63 12.70 16.01 | 24 |
| 2 | ojas-cpu | before | 10:07:19 | 8.63 12.70 16.01 | 25 |
| 2 | ojas-cpu | after | 10:07:32 | 8.06 12.37 15.84 | 54 |
| 2 | ojas-wgpu | before | 10:07:32 | 8.06 12.37 15.84 | 0 |
| 2 | ojas-wgpu | after | 10:08:36 | 9.49 11.69 15.31 | 94 |
| 2 | ojas-metal | before | 10:08:36 | 9.49 11.69 15.31 | 0 |
| 2 | ojas-metal | after | 10:09:03 | 10.39 11.77 15.23 | 89 |
| 3 | ojas-metal | before | 10:09:03 | 10.39 11.77 15.23 | 0 |
| 3 | ojas-metal | after | 10:09:31 | 10.33 11.63 15.06 | 91 |
| 3 | ojas-wgpu | before | 10:09:31 | 10.33 11.63 15.06 | 0 |
| 3 | ojas-wgpu | after | 10:10:37 | 7.91 10.65 14.44 | 95 |
| 3 | ojas-cpu | before | 10:10:37 | 7.91 10.65 14.44 | 0 |
| 3 | ojas-cpu | after | 10:10:49 | 7.53 10.48 14.33 | 24 |
| 3 | ojas-cpu-host | before | 10:10:49 | 7.53 10.48 14.33 | 25 |
| 3 | ojas-cpu-host | after | 10:11:01 | 6.90 10.19 14.16 | 24 |
| 3 | torch-mps | before | 10:11:01 | 6.90 10.19 14.16 | 0 |
| 3 | torch-mps | after | 10:11:28 | 6.67 9.84 13.92 | 98 |

### Decode (gen_ rows)

Per-token ms = (decode median - prefill median) / (N - 1), median over rounds, each decode row against its model's prefill row. Traffic is per decode step, from DeviceDecoder's counters (uploads / bytes up / bytes read back).

| runtime | row | prefill median ms | decode ms / token | tokens / s | rounds | step traffic | greedy ids equal to torch |
| :-- | :-- | --: | --: | --: | --: | :-- | :-- |
| torch-mps | gen_gqa4_greedy_p32_n32 | 6.730 | 5.393 | 185.4 | 3 | - | - |
| torch-mps | gen_greedy_p32_n32 | 6.790 | 4.812 | 207.8 | 3 | - | - |
| torch-mps | gen_topk50_p32_n32 | 6.790 | 6.726 | 148.7 | 3 | - | - |
| ojas-metal | gen_gqa4_greedy_p32_n32 | 11.88 | 8.975 | 111.4 | 3 | 0 / 0 B / 4 B | 32/32 (prefix 32) |
| ojas-metal | gen_greedy_p32_n32 | 10.15 | 9.794 | 102.1 | 3 | 0 / 0 B / 4 B | 32/32 (prefix 32) |
| ojas-metal | gen_topk50_p32_n32 | 10.15 | 10.47 | 95.6 | 3 | 1 / 4 B / 201216 B | - |
| ojas-wgpu | gen_gqa4_greedy_p32_n32 | 29.09 | 23.80 | 42.0 | 3 | 0 / 0 B / 4 B | 32/32 (prefix 32) |
| ojas-wgpu | gen_greedy_p32_n32 | 26.71 | 23.93 | 41.8 | 3 | 0 / 0 B / 4 B | 32/32 (prefix 32) |
| ojas-wgpu | gen_topk50_p32_n32 | 26.71 | 23.55 | 42.5 | 3 | 1 / 4 B / 201216 B | - |
| ojas-cpu | gen_gqa4_greedy_p32_n32 | 11.46 | 3.849 | 259.8 | 3 | 0 / 0 B / 0 B | 32/32 (prefix 32) |
| ojas-cpu | gen_greedy_p32_n32 | 11.69 | 4.172 | 239.7 | 3 | 0 / 0 B / 0 B | 32/32 (prefix 32) |
| ojas-cpu | gen_topk50_p32_n32 | 11.69 | 4.216 | 237.2 | 3 | 1 / 4 B / 201216 B | - |
| ojas-cpu-host | gen_gqa4_greedy_p32_n32 | 11.22 | 3.664 | 272.9 | 3 | - | 32/32 (prefix 32) |
| ojas-cpu-host | gen_greedy_p32_n32 | 12.61 | 4.157 | 240.6 | 3 | - | 32/32 (prefix 32) |
| ojas-cpu-host | gen_topk50_p32_n32 | 12.61 | 4.173 | 239.6 | 3 | - | - |

Reading one `[1, vocab]` logit row back (last round, min of `iters`): `Tensor::to_host` against `DeviceBuffer::read_bytes` alone; the difference is the transient decode pass a read into a caller's buffer would save.

| runtime | bytes | to_host us | read_bytes us | decode pass us | iters |
| :-- | --: | --: | --: | --: | --: |
| ojas-metal | 201216 | 11.67 | 4.04 | 7.62 | 50 |
| ojas-wgpu | 201216 | 155.17 | 149.88 | 5.29 | 50 |

### Load recorded before and after every row

| round | lane | samples | 1-min load min-max | GPU util % min / median / max |
| --: | :-- | --: | :-- | :-- |
| 1 | ojas-cpu-host | 10 | 9.1-9.4 | 0 / 0 / 32 |
| 1 | ojas-cpu | 10 | 9.6-10.4 | 0 / 0 / 14 |
| 1 | ojas-metal | 10 | 10.1-13.4 | 0 / 92 / 100 |
| 1 | ojas-wgpu | 10 | 10.3-11.9 | 0 / 90 / 95 |
| 1 | torch-mps | 10 | 8.0-9.1 | 0 / 47 / 91 |
| 2 | ojas-cpu-host | 10 | 8.6-8.9 | 0 / 26 / 79 |
| 2 | ojas-cpu | 10 | 8.1-8.6 | 24 / 38 / 68 |
| 2 | ojas-metal | 10 | 9.5-11.1 | 0 / 70 / 91 |
| 2 | ojas-wgpu | 10 | 6.0-8.1 | 0 / 94 / 97 |
| 2 | torch-mps | 10 | 8.1-8.9 | 0 / 47 / 85 |
| 3 | ojas-cpu-host | 10 | 6.9-7.5 | 0 / 24 / 25 |
| 3 | ojas-cpu | 10 | 7.5-7.9 | 0 / 10 / 27 |
| 3 | ojas-metal | 10 | 9.9-11.0 | 0 / 76 / 92 |
| 3 | ojas-wgpu | 10 | 8.3-10.3 | 0 / 92 / 96 |
| 3 | torch-mps | 10 | 6.1-6.9 | 0 / 42 / 97 |
