# Paired GPU-vs-torch summary: /Users/bharath/Code/research/ojas/bench/results/2026-10-02-gate/paired-gate-quiet

5 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.

```
date: 2026-10-02T18:45:14Z
git: 408dfb9aebdfe6f6f5ff43a4fb4974e041685fdc (dirty files: 57)
uncommitted diff of the benchmarked crates (sha1): 5ec1cf798845
tessl: cf65d9d5d3deb97b0847e020a562ac8f15e6d5a9 (dirty files: 29)
rustc: rustc 1.98.0 (88d9e12ae 2026-08-18)
cargo: cargo 1.98.0 (797e8a9bc 2026-08-05)
os: macOS 27.0.1 (26A434)
cpu: Apple M5 Pro
memory_bytes: 68719476736
metal_bin: Oct  2 13:45:14 2026 3904800
wgpu_bin: Oct  2 13:34:12 2026 6377376
rounds: 5  iters: 20  warmup: 5  rows: gate
```

### ojas-metal vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gate_fwd | 0.445 / 0.491 | 0.448 / 0.556 | 1.13x [1.01-1.37] | 14% / 48% | noisy - not quoted | 8.94e-08 (1.0e-07) |
| gate_bwd | 0.758 / 0.829 | 0.741 / 0.833 | 1.02x [0.96-1.11] | 8% / 15% | noisy - not quoted | 9.16e-05 (8.9e-07) |

### ojas-wgpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| gate_fwd | 0.533 / 0.758 | 0.448 / 0.556 | 0.90x [0.52-1.01] | 79% / 48% | noisy - not quoted | 1.19e-07 (1.4e-07) |
| gate_bwd | 1.540 / 1.740 | 0.741 / 0.833 | 0.50x [0.44-0.51] | 9% / 15% | noisy - not quoted | 2.21e-04 (2.3e-06) |

### Ranked: where ojas is slowest relative to torch (lowest ratio first)

| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |
| --: | :-- | :-- | :-- | --: | --: | :-- |
| 1 | ojas-wgpu | gate_bwd | 0.50x [0.44-0.51] | 1.740 | 0.833 | noisy |
| 2 | ojas-wgpu | gate_fwd | 0.90x [0.52-1.01] | 0.758 | 0.556 | noisy |
| 3 | ojas-metal | gate_bwd | 1.02x [0.96-1.11] | 0.829 | 0.833 | noisy |
| 4 | ojas-metal | gate_fwd | 1.13x [1.01-1.37] | 0.491 | 0.556 | noisy |

### Machine load around each runtime block

| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |
| --: | :-- | :-- | :-- | :-- | --: |
| 1 | ojas-metal | before | 13:45:22 | 9.05 16.05 15.16 | 66 |
| 1 | ojas-metal | after | 13:45:23 | 8.48 15.82 15.08 | 51 |
| 1 | ojas-wgpu | before | 13:45:23 | 8.48 15.82 15.08 | 45 |
| 1 | ojas-wgpu | after | 13:45:23 | 8.48 15.82 15.08 | 44 |
| 1 | torch-mps | before | 13:45:23 | 8.48 15.82 15.08 | 45 |
| 1 | torch-mps | after | 13:45:25 | 8.48 15.82 15.08 | 46 |
| 2 | torch-mps | before | 13:45:25 | 8.48 15.82 15.08 | 60 |
| 2 | torch-mps | after | 13:45:26 | 8.48 15.82 15.08 | 52 |
| 2 | ojas-wgpu | before | 13:45:26 | 8.48 15.82 15.08 | 80 |
| 2 | ojas-wgpu | after | 13:45:26 | 8.48 15.82 15.08 | 42 |
| 2 | ojas-metal | before | 13:45:26 | 8.48 15.82 15.08 | 45 |
| 2 | ojas-metal | after | 13:45:27 | 8.48 15.82 15.08 | 51 |
| 3 | ojas-metal | before | 13:45:27 | 8.48 15.82 15.08 | 46 |
| 3 | ojas-metal | after | 13:45:27 | 8.48 15.82 15.08 | 52 |
| 3 | ojas-wgpu | before | 13:45:27 | 8.48 15.82 15.08 | 49 |
| 3 | ojas-wgpu | after | 13:45:27 | 8.48 15.82 15.08 | 48 |
| 3 | torch-mps | before | 13:45:27 | 8.48 15.82 15.08 | 45 |
| 3 | torch-mps | after | 13:45:29 | 8.28 15.66 15.03 | 57 |
| 4 | torch-mps | before | 13:45:29 | 8.28 15.66 15.03 | 68 |
| 4 | torch-mps | after | 13:45:30 | 8.28 15.66 15.03 | 47 |
| 4 | ojas-wgpu | before | 13:45:30 | 8.28 15.66 15.03 | 51 |
| 4 | ojas-wgpu | after | 13:45:30 | 8.28 15.66 15.03 | 47 |
| 4 | ojas-metal | before | 13:45:30 | 8.28 15.66 15.03 | 49 |
| 4 | ojas-metal | after | 13:45:31 | 8.28 15.66 15.03 | 52 |
| 5 | ojas-metal | before | 13:45:31 | 8.28 15.66 15.03 | 46 |
| 5 | ojas-metal | after | 13:45:31 | 8.28 15.66 15.03 | 49 |
| 5 | ojas-wgpu | before | 13:45:31 | 8.28 15.66 15.03 | 46 |
| 5 | ojas-wgpu | after | 13:45:31 | 8.28 15.66 15.03 | 45 |
| 5 | torch-mps | before | 13:45:31 | 8.28 15.66 15.03 | 45 |
| 5 | torch-mps | after | 13:45:33 | 8.74 15.63 15.02 | 45 |

### Load recorded before and after every row

| round | lane | samples | 1-min load min-max | GPU util % min / median / max |
| --: | :-- | --: | :-- | :-- |
| 1 | ojas-metal | 4 | 8.5-9.1 | 53 / 70 / 82 |
| 1 | ojas-wgpu | 4 | 8.5-8.5 | 47 / 61 / 83 |
| 1 | torch-mps | 4 | 8.5-8.5 | 52 / 57 / 67 |
| 2 | ojas-metal | 4 | 8.5-8.5 | 44 / 52 / 71 |
| 2 | ojas-wgpu | 4 | 8.5-8.5 | 51 / 77 / 82 |
| 2 | torch-mps | 4 | 8.5-8.5 | 45 / 62 / 75 |
| 3 | ojas-metal | 4 | 8.5-8.5 | 44 / 53 / 72 |
| 3 | ojas-wgpu | 4 | 8.5-8.5 | 45 / 57 / 81 |
| 3 | torch-mps | 4 | 8.3-8.3 | 44 / 58 / 71 |
| 4 | ojas-metal | 4 | 8.3-8.3 | 43 / 56 / 72 |
| 4 | ojas-wgpu | 4 | 8.3-8.3 | 48 / 65 / 81 |
| 4 | torch-mps | 4 | 8.3-8.3 | 53 / 60 / 70 |
| 5 | ojas-metal | 4 | 8.3-8.3 | 46 / 52 / 75 |
| 5 | ojas-wgpu | 4 | 8.3-8.3 | 45 / 59 / 79 |
| 5 | torch-mps | 4 | 8.3-8.3 | 43 / 56 / 75 |
