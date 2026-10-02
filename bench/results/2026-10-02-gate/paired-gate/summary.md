# Paired GPU-vs-torch summary: /Users/bharath/Code/research/ojas/bench/results/2026-10-02-gate/paired-gate

5 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.

```
date: 2026-10-02T18:34:19Z
git: 408dfb9aebdfe6f6f5ff43a4fb4974e041685fdc (dirty files: 57)
uncommitted diff of the benchmarked crates (sha1): 5ec1cf798845
tessl: cf65d9d5d3deb97b0847e020a562ac8f15e6d5a9 (dirty files: 29)
rustc: rustc 1.98.0 (88d9e12ae 2026-08-18)
cargo: cargo 1.98.0 (797e8a9bc 2026-08-05)
os: macOS 27.0.1 (26A434)
cpu: Apple M5 Pro
memory_bytes: 68719476736
metal_bin: Oct  2 13:34:19 2026 3904800
wgpu_bin: Oct  2 13:34:12 2026 6377376
rounds: 5  iters: 20  warmup: 5  rows: gate
```

### ojas-metal vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gate_fwd | 0.437 / 0.656 | 0.405 / 0.518 | 0.77x [0.45-1.10] | 96% / 27% | noisy - not quoted | 8.94e-08 (1.0e-07) |
| gate_bwd | 0.795 / 0.862 | 0.757 / 0.834 | 0.96x [0.85-0.99] | 17% / 5% | noisy - not quoted | 9.16e-05 (8.9e-07) |

### ojas-wgpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gate_fwd | 0.507 / 0.656 | 0.405 / 0.518 | 0.80x [0.58-1.03] | 49% / 27% | noisy - not quoted | 1.19e-07 (1.4e-07) |
| gate_bwd | 1.519 / 1.596 | 0.757 / 0.834 | 0.52x [0.50-0.54] | 2% / 5% |  | 2.21e-04 (2.3e-06) |

### Ranked: where ojas is slowest relative to torch (lowest ratio first)

| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |
| --: | :-- | :-- | :-- | --: | --: | :-- |
| 1 | ojas-wgpu | gate_bwd | 0.52x [0.50-0.54] | 1.596 | 0.834 |  |
| 2 | ojas-metal | gate_fwd | 0.77x [0.45-1.10] | 0.656 | 0.518 | noisy |
| 3 | ojas-wgpu | gate_fwd | 0.80x [0.58-1.03] | 0.656 | 0.518 | noisy |
| 4 | ojas-metal | gate_bwd | 0.96x [0.85-0.99] | 0.862 | 0.834 | noisy |

### Machine load around each runtime block

| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |
| --: | :-- | :-- | :-- | :-- | --: |
| 1 | ojas-metal | before | 13:34:27 | 30.44 16.24 12.64 | 38 |
| 1 | ojas-metal | after | 13:34:27 | 30.44 16.24 12.64 | 0 |
| 1 | ojas-wgpu | before | 13:34:27 | 30.44 16.24 12.64 | 0 |
| 1 | ojas-wgpu | after | 13:34:28 | 30.48 16.49 12.74 | 0 |
| 1 | torch-mps | before | 13:34:28 | 30.48 16.49 12.74 | 0 |
| 1 | torch-mps | after | 13:34:29 | 30.48 16.49 12.74 | 36 |
| 2 | torch-mps | before | 13:34:29 | 30.48 16.49 12.74 | 37 |
| 2 | torch-mps | after | 13:34:30 | 30.48 16.49 12.74 | 34 |
| 2 | ojas-wgpu | before | 13:34:30 | 30.48 16.49 12.74 | 36 |
| 2 | ojas-wgpu | after | 13:34:30 | 30.48 16.49 12.74 | 0 |
| 2 | ojas-metal | before | 13:34:30 | 30.48 16.49 12.74 | 0 |
| 2 | ojas-metal | after | 13:34:31 | 30.48 16.49 12.74 | 0 |
| 3 | ojas-metal | before | 13:34:31 | 30.48 16.49 12.74 | 0 |
| 3 | ojas-metal | after | 13:34:31 | 30.48 16.49 12.74 | 0 |
| 3 | ojas-wgpu | before | 13:34:31 | 30.48 16.49 12.74 | 0 |
| 3 | ojas-wgpu | after | 13:34:31 | 30.48 16.49 12.74 | 0 |
| 3 | torch-mps | before | 13:34:31 | 30.48 16.49 12.74 | 0 |
| 3 | torch-mps | after | 13:34:32 | 30.48 16.49 12.74 | 31 |
| 4 | torch-mps | before | 13:34:32 | 30.48 16.49 12.74 | 0 |
| 4 | torch-mps | after | 13:34:33 | 29.32 16.48 12.76 | 19 |
| 4 | ojas-wgpu | before | 13:34:33 | 29.32 16.48 12.76 | 0 |
| 4 | ojas-wgpu | after | 13:34:34 | 29.32 16.48 12.76 | 0 |
| 4 | ojas-metal | before | 13:34:34 | 29.32 16.48 12.76 | 0 |
| 4 | ojas-metal | after | 13:34:34 | 29.32 16.48 12.76 | 0 |
| 5 | ojas-metal | before | 13:34:34 | 29.32 16.48 12.76 | 0 |
| 5 | ojas-metal | after | 13:34:34 | 29.32 16.48 12.76 | 0 |
| 5 | ojas-wgpu | before | 13:34:34 | 29.32 16.48 12.76 | 0 |
| 5 | ojas-wgpu | after | 13:34:34 | 29.32 16.48 12.76 | 0 |
| 5 | torch-mps | before | 13:34:35 | 29.32 16.48 12.76 | 0 |
| 5 | torch-mps | after | 13:34:36 | 29.32 16.48 12.76 | 41 |

### Load recorded before and after every row

| round | lane | samples | 1-min load min-max | GPU util % min / median / max |
| --: | :-- | --: | :-- | :-- |
| 1 | ojas-metal | 4 | 30.4-30.4 | 0 / 18 / 67 |
| 1 | ojas-wgpu | 4 | 30.4-30.5 | 0 / 42 / 62 |
| 1 | torch-mps | 4 | 30.5-30.5 | 0 / 48 / 54 |
| 2 | ojas-metal | 4 | 30.5-30.5 | 0 / 14 / 63 |
| 2 | ojas-wgpu | 4 | 30.5-30.5 | 0 / 42 / 72 |
| 2 | torch-mps | 4 | 30.5-30.5 | 0 / 36 / 49 |
| 3 | ojas-metal | 4 | 30.5-30.5 | 0 / 34 / 52 |
| 3 | ojas-wgpu | 4 | 30.5-30.5 | 0 / 36 / 76 |
| 3 | torch-mps | 4 | 30.5-30.5 | 0 / 38 / 49 |
| 4 | ojas-metal | 4 | 29.3-29.3 | 0 / 0 / 0 |
| 4 | ojas-wgpu | 4 | 29.3-29.3 | 0 / 38 / 74 |
| 4 | torch-mps | 4 | 29.3-29.3 | 0 / 33 / 48 |
| 5 | ojas-metal | 4 | 29.3-29.3 | 0 / 14 / 62 |
| 5 | ojas-wgpu | 4 | 29.3-29.3 | 0 / 0 / 71 |
| 5 | torch-mps | 4 | 29.3-29.3 | 0 / 37 / 41 |
