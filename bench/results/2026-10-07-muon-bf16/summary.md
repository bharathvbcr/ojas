# Paired GPU-vs-torch summary: /Users/bharath/Code/research/ojas/target-bf16-ns5/bench-paired2

5 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.

```
date: 2026-10-07T05:49:04Z
git: c74f3ba033256595eacece6fde5dcabe97fe6ee8 (dirty files: 172)
uncommitted diff of the benchmarked crates (sha1): 2652387b71d2
tessl: 045f58adecff0ad031770140ebc55563bc2394fa (dirty files: 33)
rustc: rustc 1.98.0 (88d9e12ae 2026-08-18)
cargo: cargo 1.98.0 (797e8a9bc 2026-08-05)
os: macOS 27.0.1 (26A434)
cpu: Apple M5 Pro
memory_bytes: 68719476736
metal_bin: Oct  7 00:49:04 2026 4117504
wgpu_bin: Oct  7 00:49:01 2026 6432064
rounds: 5  iters: 20  warmup: 5  rows: muon
```

### ojas-metal vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| muon_768x768 | 2.743 / 2.829 | 3.319 / 3.644 | 1.28x [1.23-1.31] | 5% / 6% |  | 1.49e-08 (4.5e-07) |
| muon_768x768 vs muon_768x768_bf16 | 2.743 / 2.829 | 1.732 / 1.965 | 0.67x [0.63-0.73] | 5% / 16% | noisy - not quoted | 1.49e-08 (4.5e-07) |
| muon_768x768_bf16 | 1.972 / 2.101 | 1.732 / 1.965 | 0.94x [0.81-0.99] | 10% / 16% | noisy - not quoted | 9.77e-05 (2.9e-03) |
| muon_2048x768 | 5.519 / 5.900 | 7.330 / 7.753 | 1.32x [1.30-1.33] | 1% / 2% |  | 1.63e-08 (4.9e-07) |
| muon_2048x768 vs muon_2048x768_bf16 | 5.519 / 5.900 | 4.159 / 4.404 | 0.75x [0.73-0.79] | 1% / 9% |  | 1.63e-08 (4.9e-07) |
| muon_2048x768_bf16 | 3.788 / 4.074 | 4.159 / 4.404 | 1.07x [1.04-1.16] | 4% / 9% |  | 7.97e-05 (2.4e-03) |
| muon_768x2048 | 5.383 / 5.770 | 7.167 / 7.619 | 1.32x [1.30-1.36] | 0% / 5% |  | 9.31e-09 (2.9e-07) |
| muon_768x2048 vs muon_768x2048_bf16 | 5.383 / 5.770 | 3.509 / 3.865 | 0.67x [0.66-0.69] | 0% / 3% |  | 9.31e-09 (2.9e-07) |
| muon_768x2048_bf16 | 3.707 / 3.933 | 3.509 / 3.865 | 0.98x [0.96-1.01] | 3% / 3% |  | 6.10e-05 (1.9e-03) |

### ojas-wgpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| muon_768x768 | 5.181 / 5.536 | 3.319 / 3.644 | 0.65x [0.64-0.68] | 6% / 6% |  | 1.49e-08 (4.5e-07) |
| muon_768x768 vs muon_768x768_bf16 | 5.181 / 5.536 | 1.732 / 1.965 | 0.35x [0.32-0.37] | 6% / 16% | noisy - not quoted | 1.49e-08 (4.5e-07) |
| muon_768x768_bf16 | - | - | - | - | **r1 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here; r2 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here; r3 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here; r4 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here; r5 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here** | - |
| muon_2048x768 | 10.66 / 10.99 | 7.330 / 7.753 | 0.71x [0.70-0.72] | 2% / 2% |  | 1.68e-08 (5.0e-07) |
| muon_2048x768 vs muon_2048x768_bf16 | 10.66 / 10.99 | 4.159 / 4.404 | 0.40x [0.39-0.42] | 2% / 9% |  | 1.68e-08 (5.0e-07) |
| muon_2048x768_bf16 | - | - | - | - | **r1 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here; r2 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here; r3 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here; r4 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here; r5 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here** | - |
| muon_768x2048 | 10.52 / 10.79 | 7.167 / 7.619 | 0.71x [0.69-0.73] | 2% / 5% |  | 9.31e-09 (2.9e-07) |
| muon_768x2048 vs muon_768x2048_bf16 | 10.52 / 10.79 | 3.509 / 3.865 | 0.36x [0.36-0.36] | 2% / 3% |  | 9.31e-09 (2.9e-07) |
| muon_768x2048_bf16 | - | - | - | - | **r1 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here; r2 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here; r3 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here; r4 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here; r5 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here** | - |

### Ranked: where ojas is slowest relative to torch (lowest ratio first)

| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |
| --: | :-- | :-- | :-- | --: | --: | :-- |
| 1 | ojas-wgpu | muon_768x768 vs muon_768x768_bf16 | 0.35x [0.32-0.37] | 5.536 | 1.965 | noisy |
| 2 | ojas-wgpu | muon_768x2048 vs muon_768x2048_bf16 | 0.36x [0.36-0.36] | 10.79 | 3.865 |  |
| 3 | ojas-wgpu | muon_2048x768 vs muon_2048x768_bf16 | 0.40x [0.39-0.42] | 10.99 | 4.404 |  |
| 4 | ojas-wgpu | muon_768x768 | 0.65x [0.64-0.68] | 5.536 | 3.644 |  |
| 5 | ojas-metal | muon_768x768 vs muon_768x768_bf16 | 0.67x [0.63-0.73] | 2.829 | 1.965 | noisy |
| 6 | ojas-metal | muon_768x2048 vs muon_768x2048_bf16 | 0.67x [0.66-0.69] | 5.770 | 3.865 |  |
| 7 | ojas-wgpu | muon_2048x768 | 0.71x [0.70-0.72] | 10.99 | 7.753 |  |
| 8 | ojas-wgpu | muon_768x2048 | 0.71x [0.69-0.73] | 10.79 | 7.619 |  |
| 9 | ojas-metal | muon_2048x768 vs muon_2048x768_bf16 | 0.75x [0.73-0.79] | 5.900 | 4.404 |  |
| 10 | ojas-metal | muon_768x768_bf16 | 0.94x [0.81-0.99] | 2.101 | 1.965 | noisy |
| 11 | ojas-metal | muon_768x2048_bf16 | 0.98x [0.96-1.01] | 3.933 | 3.865 |  |
| 12 | ojas-metal | muon_2048x768_bf16 | 1.07x [1.04-1.16] | 4.074 | 4.404 |  |
| 13 | ojas-metal | muon_768x768 | 1.28x [1.23-1.31] | 2.829 | 3.644 |  |
| 14 | ojas-metal | muon_2048x768 | 1.32x [1.30-1.33] | 5.900 | 7.753 |  |
| 15 | ojas-metal | muon_768x2048 | 1.32x [1.30-1.36] | 5.770 | 7.619 |  |

### Machine load around each runtime block

| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |
| --: | :-- | :-- | :-- | :-- | --: |
| 1 | ojas-metal | before | 00:49:12 | 5.17 3.55 3.16 | 42 |
| 1 | ojas-metal | after | 00:49:13 | 5.17 3.55 3.16 | 0 |
| 1 | ojas-wgpu | before | 00:49:13 | 5.17 3.55 3.16 | 0 |
| 1 | ojas-wgpu | after | 00:49:15 | 7.72 4.10 3.35 | 0 |
| 1 | torch-mps | before | 00:49:15 | 7.72 4.10 3.35 | 0 |
| 1 | torch-mps | after | 00:49:17 | 7.72 4.10 3.35 | 0 |
| 2 | torch-mps | before | 00:49:17 | 7.72 4.10 3.35 | 0 |
| 2 | torch-mps | after | 00:49:19 | 7.72 4.10 3.35 | 0 |
| 2 | ojas-wgpu | before | 00:49:19 | 7.72 4.10 3.35 | 0 |
| 2 | ojas-wgpu | after | 00:49:20 | 8.23 4.27 3.42 | 0 |
| 2 | ojas-metal | before | 00:49:20 | 8.23 4.27 3.42 | 0 |
| 2 | ojas-metal | after | 00:49:21 | 8.23 4.27 3.42 | 0 |
| 3 | ojas-metal | before | 00:49:21 | 8.23 4.27 3.42 | 0 |
| 3 | ojas-metal | after | 00:49:22 | 8.23 4.27 3.42 | 0 |
| 3 | ojas-wgpu | before | 00:49:22 | 8.23 4.27 3.42 | 0 |
| 3 | ojas-wgpu | after | 00:49:23 | 8.23 4.27 3.42 | 0 |
| 3 | torch-mps | before | 00:49:23 | 8.23 4.27 3.42 | 0 |
| 3 | torch-mps | after | 00:49:25 | 8.13 4.31 3.44 | 0 |
| 4 | torch-mps | before | 00:49:25 | 8.13 4.31 3.44 | 0 |
| 4 | torch-mps | after | 00:49:27 | 8.13 4.31 3.44 | 0 |
| 4 | ojas-wgpu | before | 00:49:28 | 8.13 4.31 3.44 | 0 |
| 4 | ojas-wgpu | after | 00:49:29 | 8.13 4.31 3.44 | 0 |
| 4 | ojas-metal | before | 00:49:29 | 8.13 4.31 3.44 | 0 |
| 4 | ojas-metal | after | 00:49:30 | 7.80 4.31 3.44 | 0 |
| 5 | ojas-metal | before | 00:49:30 | 7.80 4.31 3.44 | 0 |
| 5 | ojas-metal | after | 00:49:31 | 7.80 4.31 3.44 | 0 |
| 5 | ojas-wgpu | before | 00:49:31 | 7.80 4.31 3.44 | 0 |
| 5 | ojas-wgpu | after | 00:49:32 | 7.80 4.31 3.44 | 0 |
| 5 | torch-mps | before | 00:49:32 | 7.80 4.31 3.44 | 0 |
| 5 | torch-mps | after | 00:49:34 | 7.80 4.31 3.44 | 0 |

### Load recorded before and after every row

| round | lane | samples | 1-min load min-max | GPU util % min / median / max |
| --: | :-- | --: | :-- | :-- |
| 1 | ojas-metal | 12 | 5.2-5.2 | 0 / 48 / 92 |
| 1 | ojas-wgpu | 12 | 5.2-7.7 | 0 / 0 / 91 |
| 1 | torch-mps | 12 | 7.7-7.7 | 0 / 48 / 89 |
| 2 | ojas-metal | 12 | 8.2-8.2 | 0 / 35 / 92 |
| 2 | ojas-wgpu | 12 | 7.7-8.2 | 0 / 0 / 90 |
| 2 | torch-mps | 12 | 7.7-7.7 | 0 / 46 / 91 |
| 3 | ojas-metal | 12 | 8.2-8.2 | 0 / 35 / 94 |
| 3 | ojas-wgpu | 12 | 8.2-8.2 | 0 / 0 / 92 |
| 3 | torch-mps | 12 | 8.1-8.2 | 0 / 46 / 89 |
| 4 | ojas-metal | 12 | 7.8-8.1 | 0 / 35 / 92 |
| 4 | ojas-wgpu | 12 | 8.1-8.1 | 0 / 0 / 92 |
| 4 | torch-mps | 12 | 8.1-8.1 | 0 / 48 / 90 |
| 5 | ojas-metal | 12 | 7.8-7.8 | 0 / 34 / 92 |
| 5 | ojas-wgpu | 12 | 7.8-7.8 | 0 / 0 / 91 |
| 5 | torch-mps | 12 | 7.8-7.8 | 0 / 47 / 90 |
