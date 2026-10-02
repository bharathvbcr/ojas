# Paired GPU-vs-torch summary: /Users/bharath/Code/research/ojas/bench/results/2026-10-01-r4-smallops

5 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.

```
date: 2026-10-02T04:37:23Z
git: dab2a12fd2129565515954924e1a11737ca6ee47 (dirty files: 227)
uncommitted diff of the benchmarked crates (sha1): de67949a36ff
tessl: cf65d9d5d3deb97b0847e020a562ac8f15e6d5a9 (dirty files: 20)
rustc: rustc 1.98.0 (88d9e12ae 2026-08-18)
cargo: cargo 1.98.0 (797e8a9bc 2026-08-05)
os: macOS 27.0.1 (26A434)
cpu: Apple M5 Pro
memory_bytes: 68719476736
metal_bin: Oct  1 23:37:23 2026 3869936
wgpu_bin: Oct  1 23:37:11 2026 6376352
rounds: 5  iters: 20  warmup: 5  rows: floor_silu_1,rms_norm_fwd,rms_norm_bwd,rms_qk_norm_fwd,rope_,silu_,mul_,residual_add_,gate_,vres_,permute_bthd_bhtd,decode_attn_kv1024,accumulate_grad
```

### ojas-metal vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.095 / 0.189 | 0.094 / 0.136 | 0.76x [0.65-1.00] | 68% / 17% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| rms_norm_fwd | 0.334 / 0.436 | 0.152 / 0.206 | 0.49x [0.45-0.52] | 14% / 11% | noisy - not quoted | 3.58e-07 (2.0e-07) |
| rms_norm_bwd | 0.616 / 0.734 | 1.725 / 2.034 | 2.77x [2.65-2.82] | 4% / 8% |  | 3.81e-05 (2.8e-07) |
| rms_qk_norm_fwd | 0.528 / 0.645 | 0.273 / 0.354 | 0.54x [0.50-0.58] | 9% / 10% | noisy - not quoted | 3.58e-07 (1.7e-07) |
| rope_fwd | 0.316 / 0.384 | 1.167 / 1.328 | 3.45x [3.30-3.73] | 13% / 5% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| rope_bwd | 0.335 / 0.441 | 1.409 / 1.708 | 3.88x [3.60-4.62] | 28% / 10% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| silu_fwd | 0.790 / 0.873 | 0.354 / 0.432 | 0.49x [0.48-0.56] | 11% / 9% | noisy - not quoted | 2.38e-07 (6.1e-08) |
| silu_bwd | 1.115 / 1.246 | 0.497 / 0.597 | 0.50x [0.46-0.55] | 9% / 21% | noisy - not quoted | 1.19e-07 (1.1e-07) |
| mul_fwd | 1.108 / 1.246 | 0.481 / 0.581 | 0.47x [0.45-0.48] | 12% / 8% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| mul_bwd | 1.835 / 1.945 | 0.890 / 1.025 | 0.54x [0.49-0.55] | 6% / 15% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_fwd | 0.551 / 0.618 | 0.235 / 0.301 | 0.49x [0.40-0.55] | 9% / 31% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_bwd | 0.532 / 0.630 | 0.012 / 0.015 | 0.02x [0.02-0.07] | 12% / 210% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| gate_fwd | 0.512 / 0.597 | 0.391 / 0.523 | 0.84x [0.55-0.96] | 56% / 14% | noisy - not quoted | 8.94e-08 (1.0e-07) |
| gate_bwd | 2.015 / 2.120 | 0.751 / 1.024 | 0.47x [0.46-0.52] | 3% / 12% | noisy - not quoted | 2.25e-04 (2.3e-06) |
| vres_fwd | 0.463 / 0.550 | 0.443 / 0.522 | 0.94x [0.91-1.11] | 12% / 17% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| vres_bwd | 0.837 / 0.939 | 0.714 / 0.958 | 0.99x [0.89-1.04] | 10% / 13% | noisy - not quoted | 3.05e-05 (2.3e-07) |
| permute_bthd_bhtd | 0.318 / 0.360 | 0.359 / 0.415 | 1.13x [1.08-1.26] | 7% / 14% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| decode_attn_kv1024 | 0.138 / 0.169 | 0.109 / 0.159 | 0.96x [0.67-1.11] | 33% / 26% | noisy - not quoted | 1.86e-08 (3.2e-07) |
| accumulate_grad_50304x768 | 2.995 / 3.413 | 1.897 / 2.094 | 0.62x [0.58-0.64] | 5% / 7% |  | 0.00e+00 (0.0e+00) |

### ojas-wgpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.155 / 0.239 | 0.094 / 0.136 | 0.58x [0.49-0.65] | 30% / 17% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| rms_norm_fwd | 0.255 / 0.397 | 0.152 / 0.206 | 0.51x [0.49-0.61] | 27% / 11% | noisy - not quoted | 2.38e-07 (1.3e-07) |
| rms_norm_bwd | 0.470 / 0.615 | 1.725 / 2.034 | 3.16x [3.02-3.77] | 24% / 8% | noisy - not quoted | 3.34e-05 (2.6e-07) |
| rms_qk_norm_fwd | 0.604 / 0.780 | 0.273 / 0.354 | 0.44x [0.42-0.50] | 13% / 10% | noisy - not quoted | 2.38e-07 (1.2e-07) |
| rope_fwd | 0.234 / 0.312 | 1.167 / 1.328 | 4.25x [3.66-4.87] | 35% / 5% | noisy - not quoted | 1.19e-07 (6.2e-08) |
| rope_bwd | 0.208 / 0.297 | 1.409 / 1.708 | 5.76x [5.53-6.31] | 15% / 10% | noisy - not quoted | 1.19e-07 (6.1e-08) |
| silu_fwd | 0.408 / 0.491 | 0.354 / 0.432 | 0.94x [0.81-0.95] | 20% / 9% | noisy - not quoted | 7.15e-07 (1.8e-07) |
| silu_bwd | 0.552 / 0.663 | 0.497 / 0.597 | 0.89x [0.82-1.05] | 19% / 21% | noisy - not quoted | 4.77e-07 (4.3e-07) |
| mul_fwd | 0.558 / 0.694 | 0.481 / 0.581 | 0.82x [0.72-0.94] | 22% / 8% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| mul_bwd | 0.796 / 1.014 | 0.890 / 1.025 | 0.95x [0.92-1.28] | 38% / 15% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_fwd | 0.295 / 0.393 | 0.235 / 0.301 | 0.80x [0.54-0.88] | 33% / 31% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_bwd | 0.488 / 0.624 | 0.012 / 0.015 | 0.02x [0.02-0.06] | 20% / 210% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| gate_fwd | 0.526 / 0.673 | 0.391 / 0.523 | 0.76x [0.70-0.88] | 16% / 14% | noisy - not quoted | 1.19e-07 (1.4e-07) |
| gate_bwd | 1.558 / 1.828 | 0.751 / 1.024 | 0.57x [0.52-0.59] | 5% / 12% | noisy - not quoted | 2.21e-04 (2.3e-06) |
| vres_fwd | 0.311 / 0.394 | 0.443 / 0.522 | 1.33x [1.17-1.47] | 14% / 17% | noisy - not quoted | 5.96e-08 (6.0e-08) |
| vres_bwd | 0.499 / 0.644 | 0.714 / 0.958 | 1.44x [1.38-1.54] | 9% / 13% | noisy - not quoted | 3.05e-05 (2.3e-07) |
| permute_bthd_bhtd | 0.346 / 0.412 | 0.359 / 0.415 | 1.01x [0.83-1.09] | 18% / 14% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| decode_attn_kv1024 | 0.193 / 0.243 | 0.109 / 0.159 | 0.65x [0.52-0.77] | 44% / 26% | noisy - not quoted | 1.49e-08 (2.5e-07) |
| accumulate_grad_50304x768 | 1.970 / 2.212 | 1.897 / 2.094 | 0.92x [0.91-0.97] | 5% / 7% |  | 0.00e+00 (0.0e+00) |

### Ranked: where ojas is slowest relative to torch (lowest ratio first)

| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |
| --: | :-- | :-- | :-- | --: | --: | :-- |
| 1 | ojas-metal | residual_add_bwd | 0.02x [0.02-0.07] | 0.630 | 0.015 | noisy |
| 2 | ojas-wgpu | residual_add_bwd | 0.02x [0.02-0.06] | 0.624 | 0.015 | noisy |
| 3 | ojas-wgpu | rms_qk_norm_fwd | 0.44x [0.42-0.50] | 0.780 | 0.354 | noisy |
| 4 | ojas-metal | mul_fwd | 0.47x [0.45-0.48] | 1.246 | 0.581 | noisy |
| 5 | ojas-metal | gate_bwd | 0.47x [0.46-0.52] | 2.120 | 1.024 | noisy |
| 6 | ojas-metal | silu_fwd | 0.49x [0.48-0.56] | 0.873 | 0.432 | noisy |
| 7 | ojas-metal | residual_add_fwd | 0.49x [0.40-0.55] | 0.618 | 0.301 | noisy |
| 8 | ojas-metal | rms_norm_fwd | 0.49x [0.45-0.52] | 0.436 | 0.206 | noisy |
| 9 | ojas-metal | silu_bwd | 0.50x [0.46-0.55] | 1.246 | 0.597 | noisy |
| 10 | ojas-wgpu | rms_norm_fwd | 0.51x [0.49-0.61] | 0.397 | 0.206 | noisy |
| 11 | ojas-metal | mul_bwd | 0.54x [0.49-0.55] | 1.945 | 1.025 | noisy |
| 12 | ojas-metal | rms_qk_norm_fwd | 0.54x [0.50-0.58] | 0.645 | 0.354 | noisy |
| 13 | ojas-wgpu | gate_bwd | 0.57x [0.52-0.59] | 1.828 | 1.024 | noisy |
| 14 | ojas-metal | accumulate_grad_50304x768 | 0.62x [0.58-0.64] | 3.413 | 2.094 |  |
| 15 | ojas-wgpu | decode_attn_kv1024 | 0.65x [0.52-0.77] | 0.243 | 0.159 | noisy |
| 16 | ojas-wgpu | gate_fwd | 0.76x [0.70-0.88] | 0.673 | 0.523 | noisy |
| 17 | ojas-wgpu | residual_add_fwd | 0.80x [0.54-0.88] | 0.393 | 0.301 | noisy |
| 18 | ojas-wgpu | mul_fwd | 0.82x [0.72-0.94] | 0.694 | 0.581 | noisy |
| 19 | ojas-metal | gate_fwd | 0.84x [0.55-0.96] | 0.597 | 0.523 | noisy |
| 20 | ojas-wgpu | silu_bwd | 0.89x [0.82-1.05] | 0.663 | 0.597 | noisy |
| 21 | ojas-wgpu | accumulate_grad_50304x768 | 0.92x [0.91-0.97] | 2.212 | 2.094 |  |
| 22 | ojas-wgpu | silu_fwd | 0.94x [0.81-0.95] | 0.491 | 0.432 | noisy |
| 23 | ojas-metal | vres_fwd | 0.94x [0.91-1.11] | 0.550 | 0.522 | noisy |
| 24 | ojas-wgpu | mul_bwd | 0.95x [0.92-1.28] | 1.014 | 1.025 | noisy |
| 25 | ojas-metal | decode_attn_kv1024 | 0.96x [0.67-1.11] | 0.169 | 0.159 | noisy |
| 26 | ojas-metal | vres_bwd | 0.99x [0.89-1.04] | 0.939 | 0.958 | noisy |
| 27 | ojas-wgpu | permute_bthd_bhtd | 1.01x [0.83-1.09] | 0.412 | 0.415 | noisy |
| 28 | ojas-metal | permute_bthd_bhtd | 1.13x [1.08-1.26] | 0.360 | 0.415 | noisy |
| 29 | ojas-wgpu | vres_fwd | 1.33x [1.17-1.47] | 0.394 | 0.522 | noisy |
| 30 | ojas-wgpu | vres_bwd | 1.44x [1.38-1.54] | 0.644 | 0.958 | noisy |
| 31 | ojas-metal | rms_norm_bwd | 2.77x [2.65-2.82] | 0.734 | 2.034 |  |
| 32 | ojas-wgpu | rms_norm_bwd | 3.16x [3.02-3.77] | 0.615 | 2.034 | noisy |
| 33 | ojas-metal | rope_fwd | 3.45x [3.30-3.73] | 0.384 | 1.328 | noisy |
| 34 | ojas-metal | rope_bwd | 3.88x [3.60-4.62] | 0.441 | 1.708 | noisy |
| 35 | ojas-wgpu | rope_fwd | 4.25x [3.66-4.87] | 0.312 | 1.328 | noisy |
| 36 | ojas-wgpu | rope_bwd | 5.76x [5.53-6.31] | 0.297 | 1.708 | noisy |

### Machine load around each runtime block

| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |
| --: | :-- | :-- | :-- | :-- | --: |
| 1 | ojas-metal | before | 23:37:34 | 17.61 26.34 35.79 | 100 |
| 1 | ojas-metal | after | 23:37:37 | 17.96 26.27 35.71 | 100 |
| 1 | ojas-wgpu | before | 23:37:37 | 17.96 26.27 35.71 | 100 |
| 1 | ojas-wgpu | after | 23:37:39 | 17.96 26.27 35.71 | 100 |
| 1 | torch-mps | before | 23:37:39 | 17.96 26.27 35.71 | 100 |
| 1 | torch-mps | after | 23:37:44 | 17.96 26.13 35.61 | 100 |
| 2 | torch-mps | before | 23:37:44 | 17.96 26.13 35.61 | 100 |
| 2 | torch-mps | after | 23:37:49 | 19.73 26.36 35.63 | 100 |
| 2 | ojas-wgpu | before | 23:37:49 | 19.73 26.36 35.63 | 100 |
| 2 | ojas-wgpu | after | 23:37:51 | 19.59 26.22 35.53 | 100 |
| 2 | ojas-metal | before | 23:37:51 | 19.59 26.22 35.53 | 100 |
| 2 | ojas-metal | after | 23:37:53 | 19.59 26.22 35.53 | 100 |
| 3 | ojas-metal | before | 23:37:53 | 19.59 26.22 35.53 | 100 |
| 3 | ojas-metal | after | 23:37:56 | 19.30 26.05 35.42 | 100 |
| 3 | ojas-wgpu | before | 23:37:56 | 19.30 26.05 35.42 | 100 |
| 3 | ojas-wgpu | after | 23:37:58 | 19.30 26.05 35.42 | 100 |
| 3 | torch-mps | before | 23:37:58 | 19.30 26.05 35.42 | 100 |
| 3 | torch-mps | after | 23:38:03 | 19.04 25.89 35.30 | 100 |
| 4 | torch-mps | before | 23:38:03 | 19.04 25.89 35.30 | 100 |
| 4 | torch-mps | after | 23:38:08 | 19.43 25.85 35.24 | 100 |
| 4 | ojas-wgpu | before | 23:38:08 | 19.43 25.85 35.24 | 100 |
| 4 | ojas-wgpu | after | 23:38:10 | 19.43 25.85 35.24 | 100 |
| 4 | ojas-metal | before | 23:38:10 | 19.43 25.85 35.24 | 100 |
| 4 | ojas-metal | after | 23:38:12 | 19.08 25.67 35.12 | 100 |
| 5 | ojas-metal | before | 23:38:12 | 19.08 25.67 35.12 | 100 |
| 5 | ojas-metal | after | 23:38:14 | 19.08 25.67 35.12 | 100 |
| 5 | ojas-wgpu | before | 23:38:14 | 19.08 25.67 35.12 | 100 |
| 5 | ojas-wgpu | after | 23:38:17 | 18.83 25.51 35.00 | 100 |
| 5 | torch-mps | before | 23:38:17 | 18.83 25.51 35.00 | 100 |
| 5 | torch-mps | after | 23:38:21 | 18.76 25.39 34.90 | 100 |

### Load recorded before and after every row

| round | lane | samples | 1-min load min-max | GPU util % min / median / max |
| --: | :-- | --: | :-- | :-- |
| 1 | ojas-metal | 38 | 17.6-18.0 | 100 / 100 / 100 |
| 1 | ojas-wgpu | 38 | 18.0-18.0 | 100 / 100 / 100 |
| 1 | torch-mps | 38 | 18.0-18.0 | 100 / 100 / 100 |
| 2 | ojas-metal | 38 | 19.6-19.6 | 100 / 100 / 100 |
| 2 | ojas-wgpu | 38 | 19.6-19.7 | 100 / 100 / 100 |
| 2 | torch-mps | 38 | 18.0-19.7 | 100 / 100 / 100 |
| 3 | ojas-metal | 38 | 19.3-19.6 | 100 / 100 / 100 |
| 3 | ojas-wgpu | 38 | 19.3-19.3 | 100 / 100 / 100 |
| 3 | torch-mps | 38 | 19.0-19.3 | 100 / 100 / 100 |
| 4 | ojas-metal | 38 | 19.1-19.4 | 100 / 100 / 100 |
| 4 | ojas-wgpu | 38 | 19.4-19.4 | 100 / 100 / 100 |
| 4 | torch-mps | 38 | 19.0-19.4 | 100 / 100 / 100 |
| 5 | ojas-metal | 38 | 19.1-19.1 | 100 / 100 / 100 |
| 5 | ojas-wgpu | 38 | 18.8-19.1 | 100 / 100 / 100 |
| 5 | torch-mps | 38 | 18.8-18.8 | 100 / 100 / 100 |
