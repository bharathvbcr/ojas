# Paired GPU-vs-torch summary: bench/out/2026-10-04-sweep

5 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.

```
date: 2026-10-04T15:54:45Z
git: 1e108dc3d04342c47a21dea87e91b05fe5e33f4d (dirty files: 5)
uncommitted diff of the benchmarked crates (sha1): da39a3ee5e6b
tessl: 0ef5f6b16d844112aa8414b9c9e6c08c5fbcf81a (dirty files: 0)
rustc: rustc 1.98.0 (88d9e12ae 2026-08-18)
cargo: cargo 1.98.0 (797e8a9bc 2026-08-05)
os: macOS 27.0.1 (26A434)
cpu: Apple M5 Pro
memory_bytes: 68719476736
metal_bin: Oct  4 10:54:45 2026 3957856
wgpu_bin: Oct  4 10:54:41 2026 6412704
rounds: 5  iters: 20  warmup: 5  rows: sweep_
```

### ojas-metal vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| sweep_silu_n1 | 0.109 / 0.183 | 0.076 / 0.164 | 0.84x [0.74-1.70] | 42% / 135% | noisy - not quoted | 1.49e-08 (7.1e-08) |
| sweep_silu_n4096 | 0.107 / 0.174 | 0.059 / 0.147 | 0.70x [0.50-1.06] | 56% / 37% | noisy - not quoted | 5.96e-08 (8.2e-08) |
| sweep_silu_n65536 | 0.121 / 0.198 | 0.067 / 0.130 | 0.66x [0.36-0.81] | 93% / 19% | noisy - not quoted | 5.96e-08 (8.2e-08) |
| sweep_silu_n262144 | 0.137 / 0.191 | 0.072 / 0.139 | 0.71x [0.55-0.91] | 33% / 44% | noisy - not quoted | 5.96e-08 (8.2e-08) |
| sweep_silu_n1048576 | 0.159 / 0.240 | 0.073 / 0.153 | 0.64x [0.58-0.78] | 34% / 44% | noisy - not quoted | 5.96e-08 (8.2e-08) |
| sweep_silu_n2097152 | 0.176 / 0.249 | 0.109 / 0.182 | 0.73x [0.67-1.06] | 20% / 32% | noisy - not quoted | 5.96e-08 (8.2e-08) |
| sweep_silu_n4194304 | 0.243 / 0.354 | 0.202 / 0.284 | 0.80x [0.77-0.93] | 20% / 3% | noisy - not quoted | 5.96e-08 (8.2e-08) |
| sweep_silu_n8388608 | 0.389 / 0.526 | 0.337 / 0.459 | 0.83x [0.79-1.06] | 18% / 18% | noisy - not quoted | 5.96e-08 (8.2e-08) |
| sweep_decode_b1 | 0.147 / 0.214 | 0.088 / 0.168 | 0.80x [0.69-0.85] | 50% / 46% | noisy - not quoted | 1.86e-08 (2.8e-07) |
| sweep_decode_b2 | 0.163 / 0.288 | 0.097 / 0.228 | 0.76x [0.67-1.15] | 42% / 77% | noisy - not quoted | 1.86e-08 (2.8e-07) |
| sweep_decode_x2 | 0.185 / 0.292 | 0.133 / 0.259 | 0.94x [0.57-0.99] | 19% / 51% | noisy - not quoted | 1.86e-08 (2.8e-07) |
| sweep_decode_b4 | 0.225 / 0.332 | 0.189 / 0.309 | 0.93x [0.69-0.98] | 35% / 13% | noisy - not quoted | 1.86e-08 (2.7e-07) |
| sweep_decode_x4 | 0.316 / 0.465 | 0.225 / 0.309 | 0.71x [0.55-0.80] | 52% / 23% | noisy - not quoted | 1.86e-08 (2.8e-07) |
| sweep_decode_b8 | 0.331 / 0.471 | 0.279 / 0.429 | 0.92x [0.74-1.07] | 28% / 20% | noisy - not quoted | 1.86e-08 (2.7e-07) |
| sweep_decode_x8 | 0.540 / 0.789 | 0.374 / 0.511 | 0.63x [0.59-0.78] | 38% / 11% | noisy - not quoted | 1.86e-08 (2.8e-07) |
| sweep_decode_b16 | 0.533 / 0.815 | 0.520 / 0.734 | 0.93x [0.83-1.09] | 21% / 16% | noisy - not quoted | 2.61e-08 (3.4e-07) |
| sweep_decode_x16 | 0.958 / 1.342 | 0.621 / 0.913 | 0.65x [0.63-0.74] | 19% / 23% | noisy - not quoted | 2.61e-08 (4.0e-07) |

### ojas-wgpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| sweep_silu_n1 | 0.158 / 0.241 | 0.076 / 0.164 | 0.68x [0.54-1.19] | 15% / 135% | noisy - not quoted | 1.49e-08 (7.1e-08) |
| sweep_silu_n4096 | 0.147 / 0.293 | 0.059 / 0.147 | 0.50x [0.31-0.86] | 105% / 37% | noisy - not quoted | 1.19e-07 (1.6e-07) |
| sweep_silu_n65536 | 0.152 / 0.284 | 0.067 / 0.130 | 0.46x [0.28-0.63] | 90% / 19% | noisy - not quoted | 1.19e-07 (1.6e-07) |
| sweep_silu_n262144 | 0.154 / 0.261 | 0.072 / 0.139 | 0.49x [0.33-0.66] | 154% / 44% | noisy - not quoted | 1.19e-07 (1.6e-07) |
| sweep_silu_n1048576 | 0.159 / 0.312 | 0.073 / 0.153 | 0.53x [0.44-0.83] | 76% / 44% | noisy - not quoted | 1.19e-07 (1.6e-07) |
| sweep_silu_n2097152 | 0.201 / 0.315 | 0.109 / 0.182 | 0.57x [0.37-0.82] | 114% / 32% | noisy - not quoted | 1.19e-07 (1.6e-07) |
| sweep_silu_n4194304 | 0.299 / 0.395 | 0.202 / 0.284 | 0.73x [0.58-0.80] | 34% / 3% | noisy - not quoted | 1.19e-07 (1.6e-07) |
| sweep_silu_n8388608 | 0.435 / 0.604 | 0.337 / 0.459 | 0.74x [0.71-0.80] | 20% / 18% | noisy - not quoted | 1.19e-07 (1.6e-07) |
| sweep_decode_b1 | 0.207 / 0.303 | 0.088 / 0.168 | 0.53x [0.50-0.79] | 17% / 46% | noisy - not quoted | 2.24e-08 (3.3e-07) |
| sweep_decode_b2 | 0.239 / 0.367 | 0.097 / 0.228 | 0.63x [0.51-1.05] | 19% / 77% | noisy - not quoted | 2.24e-08 (3.3e-07) |
| sweep_decode_x2 | 0.271 / 0.415 | 0.133 / 0.259 | 0.65x [0.41-0.79] | 33% / 51% | noisy - not quoted | 2.24e-08 (3.3e-07) |
| sweep_decode_b4 | 0.297 / 0.439 | 0.189 / 0.309 | 0.67x [0.64-0.78] | 19% / 13% | noisy - not quoted | 2.24e-08 (3.3e-07) |
| sweep_decode_x4 | 0.422 / 0.659 | 0.225 / 0.309 | 0.47x [0.43-0.52] | 22% / 23% | noisy - not quoted | 2.24e-08 (3.3e-07) |
| sweep_decode_b8 | 0.460 / 0.717 | 0.279 / 0.429 | 0.61x [0.47-0.74] | 45% / 20% | noisy - not quoted | 2.79e-08 (4.1e-07) |
| sweep_decode_x8 | 0.735 / 1.147 | 0.374 / 0.511 | 0.42x [0.41-0.51] | 26% / 11% | noisy - not quoted | 2.24e-08 (3.3e-07) |
| sweep_decode_b16 | 0.722 / 1.030 | 0.520 / 0.734 | 0.72x [0.63-0.76] | 40% / 16% | noisy - not quoted | 4.47e-08 (5.8e-07) |
| sweep_decode_x16 | 1.339 / 2.027 | 0.621 / 0.913 | 0.44x [0.38-0.49] | 26% / 23% | noisy - not quoted | 2.24e-08 (3.3e-07) |

### Ranked: where ojas is slowest relative to torch (lowest ratio first)

| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |
| --: | :-- | :-- | :-- | --: | --: | :-- |
| 1 | ojas-wgpu | sweep_decode_x8 | 0.42x [0.41-0.51] | 1.147 | 0.511 | noisy |
| 2 | ojas-wgpu | sweep_decode_x16 | 0.44x [0.38-0.49] | 2.027 | 0.913 | noisy |
| 3 | ojas-wgpu | sweep_silu_n65536 | 0.46x [0.28-0.63] | 0.284 | 0.130 | noisy |
| 4 | ojas-wgpu | sweep_decode_x4 | 0.47x [0.43-0.52] | 0.659 | 0.309 | noisy |
| 5 | ojas-wgpu | sweep_silu_n262144 | 0.49x [0.33-0.66] | 0.261 | 0.139 | noisy |
| 6 | ojas-wgpu | sweep_silu_n4096 | 0.50x [0.31-0.86] | 0.293 | 0.147 | noisy |
| 7 | ojas-wgpu | sweep_silu_n1048576 | 0.53x [0.44-0.83] | 0.312 | 0.153 | noisy |
| 8 | ojas-wgpu | sweep_decode_b1 | 0.53x [0.50-0.79] | 0.303 | 0.168 | noisy |
| 9 | ojas-wgpu | sweep_silu_n2097152 | 0.57x [0.37-0.82] | 0.315 | 0.182 | noisy |
| 10 | ojas-wgpu | sweep_decode_b8 | 0.61x [0.47-0.74] | 0.717 | 0.429 | noisy |
| 11 | ojas-metal | sweep_decode_x8 | 0.63x [0.59-0.78] | 0.789 | 0.511 | noisy |
| 12 | ojas-wgpu | sweep_decode_b2 | 0.63x [0.51-1.05] | 0.367 | 0.228 | noisy |
| 13 | ojas-metal | sweep_silu_n1048576 | 0.64x [0.58-0.78] | 0.240 | 0.153 | noisy |
| 14 | ojas-metal | sweep_decode_x16 | 0.65x [0.63-0.74] | 1.342 | 0.913 | noisy |
| 15 | ojas-wgpu | sweep_decode_x2 | 0.65x [0.41-0.79] | 0.415 | 0.259 | noisy |
| 16 | ojas-metal | sweep_silu_n65536 | 0.66x [0.36-0.81] | 0.198 | 0.130 | noisy |
| 17 | ojas-wgpu | sweep_decode_b4 | 0.67x [0.64-0.78] | 0.439 | 0.309 | noisy |
| 18 | ojas-wgpu | sweep_silu_n1 | 0.68x [0.54-1.19] | 0.241 | 0.164 | noisy |
| 19 | ojas-metal | sweep_silu_n4096 | 0.70x [0.50-1.06] | 0.174 | 0.147 | noisy |
| 20 | ojas-metal | sweep_decode_x4 | 0.71x [0.55-0.80] | 0.465 | 0.309 | noisy |
| 21 | ojas-metal | sweep_silu_n262144 | 0.71x [0.55-0.91] | 0.191 | 0.139 | noisy |
| 22 | ojas-wgpu | sweep_decode_b16 | 0.72x [0.63-0.76] | 1.030 | 0.734 | noisy |
| 23 | ojas-metal | sweep_silu_n2097152 | 0.73x [0.67-1.06] | 0.249 | 0.182 | noisy |
| 24 | ojas-wgpu | sweep_silu_n4194304 | 0.73x [0.58-0.80] | 0.395 | 0.284 | noisy |
| 25 | ojas-wgpu | sweep_silu_n8388608 | 0.74x [0.71-0.80] | 0.604 | 0.459 | noisy |
| 26 | ojas-metal | sweep_decode_b2 | 0.76x [0.67-1.15] | 0.288 | 0.228 | noisy |
| 27 | ojas-metal | sweep_decode_b1 | 0.80x [0.69-0.85] | 0.214 | 0.168 | noisy |
| 28 | ojas-metal | sweep_silu_n4194304 | 0.80x [0.77-0.93] | 0.354 | 0.284 | noisy |
| 29 | ojas-metal | sweep_silu_n8388608 | 0.83x [0.79-1.06] | 0.526 | 0.459 | noisy |
| 30 | ojas-metal | sweep_silu_n1 | 0.84x [0.74-1.70] | 0.183 | 0.164 | noisy |
| 31 | ojas-metal | sweep_decode_b8 | 0.92x [0.74-1.07] | 0.471 | 0.429 | noisy |
| 32 | ojas-metal | sweep_decode_b4 | 0.93x [0.69-0.98] | 0.332 | 0.309 | noisy |
| 33 | ojas-metal | sweep_decode_b16 | 0.93x [0.83-1.09] | 0.815 | 0.734 | noisy |
| 34 | ojas-metal | sweep_decode_x2 | 0.94x [0.57-0.99] | 0.292 | 0.259 | noisy |

### Machine load around each runtime block

| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |
| --: | :-- | :-- | :-- | :-- | --: |
| 1 | ojas-metal | before | 10:54:55 | 6.77 6.30 6.17 | 73 |
| 1 | ojas-metal | after | 10:54:57 | 6.77 6.30 6.17 | 76 |
| 1 | ojas-wgpu | before | 10:54:57 | 6.77 6.30 6.17 | 75 |
| 1 | ojas-wgpu | after | 10:54:59 | 6.63 6.28 6.16 | 68 |
| 1 | torch-mps | before | 10:54:59 | 6.63 6.28 6.16 | 75 |
| 1 | torch-mps | after | 10:55:02 | 6.63 6.28 6.16 | 71 |
| 2 | torch-mps | before | 10:55:02 | 6.63 6.28 6.16 | 72 |
| 2 | torch-mps | after | 10:55:05 | 6.50 6.26 6.15 | 71 |
| 2 | ojas-wgpu | before | 10:55:05 | 6.50 6.26 6.15 | 62 |
| 2 | ojas-wgpu | after | 10:55:06 | 6.50 6.26 6.15 | 71 |
| 2 | ojas-metal | before | 10:55:07 | 6.50 6.26 6.15 | 71 |
| 2 | ojas-metal | after | 10:55:08 | 6.38 6.23 6.15 | 72 |
| 3 | ojas-metal | before | 10:55:08 | 6.38 6.23 6.15 | 72 |
| 3 | ojas-metal | after | 10:55:09 | 6.38 6.23 6.15 | 69 |
| 3 | ojas-wgpu | before | 10:55:09 | 6.38 6.23 6.15 | 68 |
| 3 | ojas-wgpu | after | 10:55:11 | 6.38 6.23 6.15 | 76 |
| 3 | torch-mps | before | 10:55:11 | 6.38 6.23 6.15 | 79 |
| 3 | torch-mps | after | 10:55:15 | 6.19 6.20 6.13 | 74 |
| 4 | torch-mps | before | 10:55:15 | 6.19 6.20 6.13 | 77 |
| 4 | torch-mps | after | 10:55:18 | 6.09 6.18 6.13 | 71 |
| 4 | ojas-wgpu | before | 10:55:18 | 6.09 6.18 6.13 | 78 |
| 4 | ojas-wgpu | after | 10:55:20 | 6.09 6.18 6.13 | 54 |
| 4 | ojas-metal | before | 10:55:20 | 6.09 6.18 6.13 | 56 |
| 4 | ojas-metal | after | 10:55:21 | 6.09 6.18 6.13 | 56 |
| 5 | ojas-metal | before | 10:55:21 | 6.09 6.18 6.13 | 55 |
| 5 | ojas-metal | after | 10:55:23 | 6.16 6.19 6.13 | 58 |
| 5 | ojas-wgpu | before | 10:55:23 | 6.16 6.19 6.13 | 57 |
| 5 | ojas-wgpu | after | 10:55:24 | 6.16 6.19 6.13 | 57 |
| 5 | torch-mps | before | 10:55:24 | 6.16 6.19 6.13 | 57 |
| 5 | torch-mps | after | 10:55:27 | 6.16 6.19 6.13 | 56 |

### Load recorded before and after every row

| round | lane | samples | 1-min load min-max | GPU util % min / median / max |
| --: | :-- | --: | :-- | :-- |
| 1 | ojas-metal | 34 | 6.8-6.8 | 70 / 78 / 91 |
| 1 | ojas-wgpu | 34 | 6.6-6.8 | 62 / 78 / 87 |
| 1 | torch-mps | 34 | 6.6-6.6 | 67 / 72 / 91 |
| 2 | ojas-metal | 34 | 6.4-6.5 | 66 / 74 / 87 |
| 2 | ojas-wgpu | 34 | 6.5-6.5 | 60 / 74 / 87 |
| 2 | torch-mps | 34 | 6.5-6.5 | 68 / 72 / 92 |
| 3 | ojas-metal | 34 | 6.4-6.4 | 66 / 74 / 88 |
| 3 | ojas-wgpu | 34 | 6.4-6.4 | 66 / 74 / 92 |
| 3 | torch-mps | 34 | 6.2-6.4 | 72 / 78 / 93 |
| 4 | ojas-metal | 34 | 6.1-6.1 | 51 / 62 / 80 |
| 4 | ojas-wgpu | 34 | 6.1-6.1 | 71 / 77 / 86 |
| 4 | torch-mps | 34 | 6.1-6.2 | 68 / 76 / 92 |
| 5 | ojas-metal | 34 | 6.1-6.2 | 56 / 61 / 77 |
| 5 | ojas-wgpu | 34 | 6.2-6.2 | 54 / 62 / 78 |
| 5 | torch-mps | 34 | 6.2-6.2 | 54 / 58 / 87 |
