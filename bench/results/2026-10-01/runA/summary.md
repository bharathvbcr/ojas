# Paired GPU-vs-torch summary: /Users/bharath/Code/research/ojas/bench/out/run5

5 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.

```
date: 2026-10-01T21:34:10Z
git: dab2a12fd2129565515954924e1a11737ca6ee47 (dirty files: 165)
uncommitted diff of the benchmarked crates (sha1): c226ffba076d
tessl: cf65d9d5d3deb97b0847e020a562ac8f15e6d5a9 (dirty files: 20)
rustc: rustc 1.98.0 (88d9e12ae 2026-08-18)
cargo: cargo 1.98.0 (797e8a9bc 2026-08-05)
os: macOS 27.0.1 (26A434)
cpu: Apple M5 Pro
memory_bytes: 68719476736
metal_bin: Oct  1 16:34:10 2026 3689360
wgpu_bin: Oct  1 16:34:10 2026 6198064
rounds: 5  iters: 20  warmup: 5  rows: all
```

### ojas-metal vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.191 / 1.756 | 0.082 / 0.107 | 0.06x [0.05-0.07] | 4% / 26% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_fwd | 1.153 / 1.999 | 0.802 / 0.918 | 0.46x [0.28-0.48] | 88% / 17% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_bwd | 2.499 / 3.785 | 1.720 / 2.112 | 0.53x [0.47-0.73] | 54% / 12% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_fwd | 2.709 / 4.127 | 1.943 / 2.152 | 0.51x [0.42-0.59] | 42% / 5% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_bwd | 4.899 / 5.630 | 4.176 / 4.497 | 0.78x [0.55-0.83] | 59% / 7% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_fwd | 2.600 / 3.609 | 2.013 / 2.201 | 0.61x [0.48-0.65] | 44% / 6% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_bwd | 4.592 / 6.541 | 4.008 / 4.458 | 0.66x [0.42-0.73] | 76% / 8% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_fwd | 109.4 / 131.9 | 54.41 / 60.84 | 0.52x [0.42-0.54] | 20% / 26% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_bwd | 131.0 / 174.3 | 105.5 / 120.4 | 0.70x [0.57-0.82] | 31% / 39% | noisy - not quoted | 1.57e-04 (1.3e-05) |
| sdpa_b4h12t1024d64_fwd | 2.045 / 3.081 | 1.367 / 1.600 | 0.55x [0.39-0.61] | 68% / 19% | noisy - not quoted | 2.24e-07 (2.2e-07) |
| sdpa_b4h12t1024d64_bwd | 5.830 / 8.111 | 9.474 / 10.30 | 1.24x [1.14-1.37] | 29% / 13% | noisy - not quoted | 1.10e-06 (2.1e-06) |
| sdpa_b4h8t2048d64_fwd | 4.775 / 6.909 | 3.225 / 3.542 | 0.50x [0.47-0.65] | 23% / 16% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b4h8t2048d64_bwd | 17.30 / 19.41 | 24.21 / 25.39 | 1.38x [1.01-1.59] | 35% / 25% | noisy - not quoted | 1.31e-06 (2.5e-06) |
| sdpa_b2h8t1024d128_fwd | 1.333 / 2.029 | 1.074 / 1.202 | 0.61x [0.37-0.68] | 96% / 17% | noisy - not quoted | 2.09e-07 (2.1e-07) |
| sdpa_b2h8t1024d128_bwd | 5.210 / 6.417 | 4.395 / 5.318 | 0.81x [0.77-0.92] | 22% / 40% | noisy - not quoted | 1.10e-06 (2.6e-06) |
| rms_norm_fwd | 0.478 / 1.654 | 0.183 / 0.258 | 0.18x [0.09-0.70] | 142% / 226% | noisy - not quoted | 3.58e-07 (2.0e-07) |
| rms_norm_bwd | 1.939 / 3.389 | 2.037 / 2.515 | 0.74x [0.53-0.84] | 48% / 49% | noisy - not quoted | 2.14e-04 (1.5e-06) |
| rms_qk_norm_fwd | 2.001 / 4.736 | 0.304 / 0.534 | 0.11x [0.08-0.19] | 49% / 63% | noisy - not quoted | 3.58e-07 (1.7e-07) |
| rms_qk_norm_bwd | 50.48 / 53.15 | 4.369 / 5.276 | 0.10x [0.09-0.10] | 2% / 8% |  | 1.30e-03 (4.0e-06) |
| rope_fwd | 0.586 / 1.816 | 1.220 / 1.513 | 0.85x [0.68-1.51] | 124% / 19% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| rope_bwd | 0.440 / 1.182 | 1.722 / 2.081 | 1.81x [0.87-2.38] | 154% / 26% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| silu_fwd | 0.927 / 2.267 | 0.361 / 0.466 | 0.20x [0.13-0.28] | 96% / 51% | noisy - not quoted | 2.38e-07 (6.1e-08) |
| silu_bwd | 1.533 / 3.191 | 0.506 / 0.691 | 0.23x [0.18-0.30] | 38% / 62% | noisy - not quoted | 1.19e-07 (1.1e-07) |
| mul_fwd | 1.291 / 2.899 | 0.496 / 0.615 | 0.21x [0.21-0.24] | 26% / 31% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| mul_bwd | 2.328 / 4.131 | 0.866 / 1.387 | 0.35x [0.26-0.38] | 26% / 49% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_fwd | 0.641 / 1.559 | 0.223 / 0.345 | 0.18x [0.13-0.25] | 78% / 93% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_bwd | 0.593 / 1.878 | 0.012 / 0.016 | 0.01x [0.01-0.01] | 75% / 135% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| gate_fwd | 0.646 / 1.773 | 0.476 / 0.681 | 0.36x [0.26-0.98] | 75% / 124% | noisy - not quoted | 8.94e-08 (1.0e-07) |
| gate_bwd | 2.928 / 4.069 | 0.957 / 1.350 | 0.33x [0.26-0.64] | 32% / 104% | noisy - not quoted | 2.25e-04 (2.3e-06) |
| vres_fwd | 0.590 / 1.645 | 0.481 / 0.650 | 0.41x [0.23-0.57] | 121% / 23% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| vres_bwd | 0.946 / 1.943 | 0.840 / 1.192 | 0.64x [0.32-0.79] | 89% / 38% | noisy - not quoted | 3.05e-05 (2.3e-07) |
| permute_bthd_bhtd | 0.391 / 0.771 | 0.407 / 0.547 | 0.72x [0.23-1.27] | 214% / 79% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_fwd | 13.31 / 20.94 | 14.75 / 19.60 | 0.97x [0.76-1.06] | 13% / 35% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_bwd | 78.49 / 96.68 | 28.31 / 34.30 | 0.35x [0.33-0.38] | 7% / 14% | noisy - not quoted | 6.39e-14 (2.6e-10) |
| clip_grad_norm_full | 16.71 / 33.67 | 137.4 / 154.2 | 4.58x [4.46-4.75] | 7% / 1% |  | 3.05e-05 (3.0e-07) |
| adamw_full | 515.8 / 598.1 | 44.33 / 53.52 | 0.09x [0.08-0.10] | 7% / 11% | noisy - not quoted | 3.73e-09 (1.2e-07) |
| muon_768x768 | 4.565 / 6.507 | 4.005 / 4.759 | 0.71x [0.62-0.76] | 34% / 19% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| muon_768x768 vs muon_768x768_bf16 | 4.565 / 6.507 | 2.236 / 2.943 | 0.44x [0.41-0.47] | 34% / 27% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| muon_2048x768 | 9.782 / 12.01 | 8.145 / 9.158 | 0.75x [0.67-0.81] | 24% / 12% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_2048x768 vs muon_2048x768_bf16 | 9.782 / 12.01 | 4.831 / 5.721 | 0.45x [0.43-0.48] | 24% / 19% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_768x2048 | 9.908 / 11.73 | 8.615 / 9.080 | 0.77x [0.40-0.80] | 114% / 11% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| muon_768x2048 vs muon_768x2048_bf16 | 9.908 / 11.73 | 4.598 / 5.343 | 0.45x [0.23-0.49] | 114% / 10% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| block_fwd | 48.08 / 67.25 | 15.74 / 16.92 | 0.24x [0.24-0.27] | 24% / 9% | noisy - not quoted | 2.38e-07 (1.9e-07) |
| block_fwd_bwd | 179.1 / 230.5 | 77.08 / 83.64 | 0.37x [0.34-0.38] | 11% / 7% | noisy - not quoted | 9.06e-06 (2.5e-06) |

### ojas-wgpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.170 / 0.244 | 0.082 / 0.107 | 0.48x [0.21-0.50] | 147% / 26% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_fwd | 2.370 / 2.787 | 0.802 / 0.918 | 0.31x [0.30-0.33] | 14% / 17% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_bwd | 5.072 / 5.620 | 1.720 / 2.112 | 0.38x [0.32-0.40] | 14% / 12% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_fwd | 6.080 / 6.697 | 1.943 / 2.152 | 0.32x [0.27-0.35] | 21% / 5% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_bwd | 12.97 / 14.04 | 4.176 / 4.497 | 0.31x [0.29-0.34] | 15% / 7% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_fwd | 6.693 / 7.456 | 2.013 / 2.201 | 0.29x [0.24-0.31] | 31% / 6% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_bwd | 13.07 / 13.84 | 4.008 / 4.458 | 0.32x [0.28-0.34] | 13% / 8% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_fwd | 186.1 / 194.7 | 54.41 / 60.84 | 0.31x [0.28-0.36] | 5% / 26% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_bwd | 307.1 / 323.7 | 105.5 / 120.4 | 0.37x [0.34-0.43] | 10% / 39% | noisy - not quoted | 1.57e-04 (1.3e-05) |
| sdpa_b4h12t1024d64_fwd | 72.67 / 77.05 | 1.367 / 1.600 | 0.02x [0.02-0.02] | 12% / 19% | noisy - not quoted | 2.24e-07 (2.2e-07) |
| sdpa_b4h12t1024d64_bwd | 270.8 / 277.8 | 9.474 / 10.30 | 0.04x [0.04-0.04] | 3% / 13% | noisy - not quoted | 1.25e-06 (2.4e-06) |
| sdpa_b4h8t2048d64_fwd | 183.2 / 191.6 | 3.225 / 3.542 | 0.02x [0.02-0.02] | 6% / 16% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b4h8t2048d64_bwd | 694.0 / 705.7 | 24.21 / 25.39 | 0.04x [0.04-0.04] | 7% / 25% | noisy - not quoted | 1.37e-06 (2.6e-06) |
| sdpa_b2h8t1024d128_fwd | 68.78 / 91.63 | 1.074 / 1.202 | 0.01x [0.01-0.01] | 6% / 17% | noisy - not quoted | 2.09e-07 (2.1e-07) |
| sdpa_b2h8t1024d128_bwd | 267.7 / 280.7 | 4.395 / 5.318 | 0.02x [0.02-0.02] | 7% / 40% | noisy - not quoted | 1.07e-06 (2.3e-06) |
| rms_norm_fwd | 0.304 / 0.424 | 0.183 / 0.258 | 0.61x [0.46-1.07] | 321% / 226% | noisy - not quoted | 2.38e-07 (1.3e-07) |
| rms_norm_bwd | 0.584 / 0.896 | 2.037 / 2.515 | 3.42x [1.67-3.91] | 132% / 49% | noisy - not quoted | 3.34e-05 (2.6e-07) |
| rms_qk_norm_fwd | 1.948 / 2.624 | 0.304 / 0.534 | 0.17x [0.14-0.22] | 58% / 63% | noisy - not quoted | 2.38e-07 (1.2e-07) |
| rms_qk_norm_bwd | 4.744 / 5.421 | 4.369 / 5.276 | 0.95x [0.88-1.00] | 7% / 8% |  | 4.58e-04 (1.3e-06) |
| rope_fwd | 0.243 / 0.470 | 1.220 / 1.513 | 3.67x [3.15-4.45] | 47% / 19% | noisy - not quoted | 1.19e-07 (6.2e-08) |
| rope_bwd | 0.250 / 0.423 | 1.722 / 2.081 | 4.73x [3.88-6.78] | 66% / 26% | noisy - not quoted | 1.19e-07 (6.1e-08) |
| silu_fwd | 0.488 / 0.728 | 0.361 / 0.466 | 0.75x [0.55-0.79] | 45% / 51% | noisy - not quoted | 7.15e-07 (1.8e-07) |
| silu_bwd | 0.691 / 1.004 | 0.506 / 0.691 | 0.86x [0.47-1.00] | 60% / 62% | noisy - not quoted | 4.77e-07 (4.3e-07) |
| mul_fwd | 0.598 / 0.862 | 0.496 / 0.615 | 0.74x [0.56-0.78] | 43% / 31% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| mul_bwd | 0.906 / 1.263 | 0.866 / 1.387 | 1.06x [0.65-1.16] | 34% / 49% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_fwd | 0.328 / 0.513 | 0.223 / 0.345 | 0.72x [0.43-0.83] | 76% / 93% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_bwd | 0.628 / 0.898 | 0.012 / 0.016 | 0.02x [0.01-0.03] | 26% / 135% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| gate_fwd | 0.635 / 0.943 | 0.476 / 0.681 | 0.74x [0.67-1.62] | 35% / 124% | noisy - not quoted | 1.19e-07 (1.4e-07) |
| gate_bwd | 1.813 / 2.111 | 0.957 / 1.350 | 0.63x [0.61-1.21] | 6% / 104% | noisy - not quoted | 2.21e-04 (2.3e-06) |
| vres_fwd | 0.358 / 0.458 | 0.481 / 0.650 | 1.32x [1.14-1.48] | 17% / 23% | noisy - not quoted | 5.96e-08 (6.0e-08) |
| vres_bwd | 0.573 / 0.720 | 0.840 / 1.192 | 1.67x [1.56-2.17] | 20% / 38% | noisy - not quoted | 3.05e-05 (2.3e-07) |
| permute_bthd_bhtd | 0.411 / 0.628 | 0.407 / 0.547 | 0.89x [0.60-1.53] | 61% / 79% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_fwd | 6.317 / 7.139 | 14.75 / 19.60 | 2.79x [2.14-2.99] | 18% / 35% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_bwd | 57.75 / 63.71 | 28.31 / 34.30 | 0.53x [0.50-0.54] | 12% / 14% | noisy - not quoted | 6.39e-14 (2.6e-10) |
| clip_grad_norm_full | 21.34 / 24.18 | 137.4 / 154.2 | 6.36x [6.14-6.38] | 5% / 1% |  | 3.81e-05 (3.8e-07) |
| adamw_full | 66.35 / 75.15 | 44.33 / 53.52 | 0.70x [0.69-0.75] | 13% / 11% | noisy - not quoted | 3.73e-09 (1.2e-07) |
| muon_768x768 | 9.453 / 10.58 | 4.005 / 4.759 | 0.44x [0.42-0.47] | 22% / 19% | noisy - not quoted | 1.49e-08 (4.5e-07) |
| muon_768x768 vs muon_768x768_bf16 | 9.453 / 10.58 | 2.236 / 2.943 | 0.27x [0.27-0.28] | 22% / 27% | noisy - not quoted | 1.49e-08 (4.5e-07) |
| muon_2048x768 | 20.14 / 22.32 | 8.145 / 9.158 | 0.41x [0.41-0.44] | 8% / 12% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_2048x768 vs muon_2048x768_bf16 | 20.14 / 22.32 | 4.831 / 5.721 | 0.25x [0.24-0.28] | 8% / 19% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_768x2048 | 19.77 / 21.64 | 8.615 / 9.080 | 0.42x [0.41-0.46] | 14% / 11% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| muon_768x2048 vs muon_768x2048_bf16 | 19.77 / 21.64 | 4.598 / 5.343 | 0.25x [0.24-0.26] | 14% / 10% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| block_fwd | 115.3 / 120.0 | 15.74 / 16.92 | 0.14x [0.13-0.14] | 16% / 9% | noisy - not quoted | 2.38e-07 (1.9e-07) |
| block_fwd_bwd | 465.8 / 475.3 | 77.08 / 83.64 | 0.18x [0.17-0.18] | 3% / 7% |  | 8.58e-06 (2.3e-06) |

### Ranked: where ojas is slowest relative to torch (lowest ratio first)

| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |
| --: | :-- | :-- | :-- | --: | --: | :-- |
| 1 | ojas-metal | residual_add_bwd | 0.01x [0.01-0.01] | 1.878 | 0.016 | noisy |
| 2 | ojas-wgpu | sdpa_b2h8t1024d128_fwd | 0.01x [0.01-0.01] | 91.63 | 1.202 | noisy |
| 3 | ojas-wgpu | residual_add_bwd | 0.02x [0.01-0.03] | 0.898 | 0.016 | noisy |
| 4 | ojas-wgpu | sdpa_b4h8t2048d64_fwd | 0.02x [0.02-0.02] | 191.6 | 3.542 | noisy |
| 5 | ojas-wgpu | sdpa_b2h8t1024d128_bwd | 0.02x [0.02-0.02] | 280.7 | 5.318 | noisy |
| 6 | ojas-wgpu | sdpa_b4h12t1024d64_fwd | 0.02x [0.02-0.02] | 77.05 | 1.600 | noisy |
| 7 | ojas-wgpu | sdpa_b4h8t2048d64_bwd | 0.04x [0.04-0.04] | 705.7 | 25.39 | noisy |
| 8 | ojas-wgpu | sdpa_b4h12t1024d64_bwd | 0.04x [0.04-0.04] | 277.8 | 10.30 | noisy |
| 9 | ojas-metal | adamw_full | 0.09x [0.08-0.10] | 598.1 | 53.52 | noisy |
| 10 | ojas-metal | rms_qk_norm_bwd | 0.10x [0.09-0.10] | 53.15 | 5.276 |  |
| 11 | ojas-metal | rms_qk_norm_fwd | 0.11x [0.08-0.19] | 4.736 | 0.534 | noisy |
| 12 | ojas-wgpu | block_fwd | 0.14x [0.13-0.14] | 120.0 | 16.92 | noisy |
| 13 | ojas-wgpu | rms_qk_norm_fwd | 0.17x [0.14-0.22] | 2.624 | 0.534 | noisy |
| 14 | ojas-wgpu | block_fwd_bwd | 0.18x [0.17-0.18] | 475.3 | 83.64 |  |
| 15 | ojas-metal | rms_norm_fwd | 0.18x [0.09-0.70] | 1.654 | 0.258 | noisy |
| 16 | ojas-metal | residual_add_fwd | 0.18x [0.13-0.25] | 1.559 | 0.345 | noisy |
| 17 | ojas-metal | silu_fwd | 0.20x [0.13-0.28] | 2.267 | 0.466 | noisy |
| 18 | ojas-metal | mul_fwd | 0.21x [0.21-0.24] | 2.899 | 0.615 | noisy |
| 19 | ojas-metal | silu_bwd | 0.23x [0.18-0.30] | 3.191 | 0.691 | noisy |
| 20 | ojas-metal | block_fwd | 0.24x [0.24-0.27] | 67.25 | 16.92 | noisy |
| 21 | ojas-wgpu | muon_2048x768 vs muon_2048x768_bf16 | 0.25x [0.24-0.28] | 22.32 | 5.721 | noisy |
| 22 | ojas-wgpu | muon_768x2048 vs muon_768x2048_bf16 | 0.25x [0.24-0.26] | 21.64 | 5.343 | noisy |
| 23 | ojas-wgpu | muon_768x768 vs muon_768x768_bf16 | 0.27x [0.27-0.28] | 10.58 | 2.943 | noisy |
| 24 | ojas-wgpu | linear_down_fwd | 0.29x [0.24-0.31] | 7.456 | 2.201 | noisy |
| 25 | ojas-wgpu | linear_up_bwd | 0.31x [0.29-0.34] | 14.04 | 4.497 | noisy |
| 26 | ojas-wgpu | linear_qkv_fwd | 0.31x [0.30-0.33] | 2.787 | 0.918 | noisy |
| 27 | ojas-wgpu | linear_lmhead_fwd | 0.31x [0.28-0.36] | 194.7 | 60.84 | noisy |
| 28 | ojas-wgpu | linear_down_bwd | 0.32x [0.28-0.34] | 13.84 | 4.458 | noisy |
| 29 | ojas-wgpu | linear_up_fwd | 0.32x [0.27-0.35] | 6.697 | 2.152 | noisy |
| 30 | ojas-metal | gate_bwd | 0.33x [0.26-0.64] | 4.069 | 1.350 | noisy |
| 31 | ojas-metal | mul_bwd | 0.35x [0.26-0.38] | 4.131 | 1.387 | noisy |
| 32 | ojas-metal | cross_entropy_bwd | 0.35x [0.33-0.38] | 96.68 | 34.30 | noisy |
| 33 | ojas-metal | gate_fwd | 0.36x [0.26-0.98] | 1.773 | 0.681 | noisy |
| 34 | ojas-metal | block_fwd_bwd | 0.37x [0.34-0.38] | 230.5 | 83.64 | noisy |
| 35 | ojas-wgpu | linear_lmhead_bwd | 0.37x [0.34-0.43] | 323.7 | 120.4 | noisy |
| 36 | ojas-wgpu | linear_qkv_bwd | 0.38x [0.32-0.40] | 5.620 | 2.112 | noisy |
| 37 | ojas-metal | vres_fwd | 0.41x [0.23-0.57] | 1.645 | 0.650 | noisy |
| 38 | ojas-wgpu | muon_2048x768 | 0.41x [0.41-0.44] | 22.32 | 9.158 | noisy |
| 39 | ojas-wgpu | muon_768x2048 | 0.42x [0.41-0.46] | 21.64 | 9.080 | noisy |
| 40 | ojas-wgpu | muon_768x768 | 0.44x [0.42-0.47] | 10.58 | 4.759 | noisy |
| 41 | ojas-metal | muon_768x768 vs muon_768x768_bf16 | 0.44x [0.41-0.47] | 6.507 | 2.943 | noisy |
| 42 | ojas-metal | muon_2048x768 vs muon_2048x768_bf16 | 0.45x [0.43-0.48] | 12.01 | 5.721 | noisy |
| 43 | ojas-metal | muon_768x2048 vs muon_768x2048_bf16 | 0.45x [0.23-0.49] | 11.73 | 5.343 | noisy |
| 44 | ojas-metal | linear_qkv_fwd | 0.46x [0.28-0.48] | 1.999 | 0.918 | noisy |
| 45 | ojas-metal | sdpa_b4h8t2048d64_fwd | 0.50x [0.47-0.65] | 6.909 | 3.542 | noisy |
| 46 | ojas-metal | linear_up_fwd | 0.51x [0.42-0.59] | 4.127 | 2.152 | noisy |
| 47 | ojas-metal | linear_lmhead_fwd | 0.52x [0.42-0.54] | 131.9 | 60.84 | noisy |
| 48 | ojas-metal | linear_qkv_bwd | 0.53x [0.47-0.73] | 3.785 | 2.112 | noisy |
| 49 | ojas-wgpu | cross_entropy_bwd | 0.53x [0.50-0.54] | 63.71 | 34.30 | noisy |
| 50 | ojas-metal | sdpa_b4h12t1024d64_fwd | 0.55x [0.39-0.61] | 3.081 | 1.600 | noisy |
| 51 | ojas-wgpu | rms_norm_fwd | 0.61x [0.46-1.07] | 0.424 | 0.258 | noisy |
| 52 | ojas-metal | sdpa_b2h8t1024d128_fwd | 0.61x [0.37-0.68] | 2.029 | 1.202 | noisy |
| 53 | ojas-metal | linear_down_fwd | 0.61x [0.48-0.65] | 3.609 | 2.201 | noisy |
| 54 | ojas-wgpu | gate_bwd | 0.63x [0.61-1.21] | 2.111 | 1.350 | noisy |
| 55 | ojas-metal | vres_bwd | 0.64x [0.32-0.79] | 1.943 | 1.192 | noisy |
| 56 | ojas-metal | linear_down_bwd | 0.66x [0.42-0.73] | 6.541 | 4.458 | noisy |
| 57 | ojas-wgpu | adamw_full | 0.70x [0.69-0.75] | 75.15 | 53.52 | noisy |
| 58 | ojas-metal | linear_lmhead_bwd | 0.70x [0.57-0.82] | 174.3 | 120.4 | noisy |
| 59 | ojas-metal | muon_768x768 | 0.71x [0.62-0.76] | 6.507 | 4.759 | noisy |
| 60 | ojas-metal | permute_bthd_bhtd | 0.72x [0.23-1.27] | 0.771 | 0.547 | noisy |
| 61 | ojas-wgpu | residual_add_fwd | 0.72x [0.43-0.83] | 0.513 | 0.345 | noisy |
| 62 | ojas-wgpu | mul_fwd | 0.74x [0.56-0.78] | 0.862 | 0.615 | noisy |
| 63 | ojas-wgpu | gate_fwd | 0.74x [0.67-1.62] | 0.943 | 0.681 | noisy |
| 64 | ojas-metal | rms_norm_bwd | 0.74x [0.53-0.84] | 3.389 | 2.515 | noisy |
| 65 | ojas-wgpu | silu_fwd | 0.75x [0.55-0.79] | 0.728 | 0.466 | noisy |
| 66 | ojas-metal | muon_2048x768 | 0.75x [0.67-0.81] | 12.01 | 9.158 | noisy |
| 67 | ojas-metal | muon_768x2048 | 0.77x [0.40-0.80] | 11.73 | 9.080 | noisy |
| 68 | ojas-metal | linear_up_bwd | 0.78x [0.55-0.83] | 5.630 | 4.497 | noisy |
| 69 | ojas-metal | sdpa_b2h8t1024d128_bwd | 0.81x [0.77-0.92] | 6.417 | 5.318 | noisy |
| 70 | ojas-metal | rope_fwd | 0.85x [0.68-1.51] | 1.816 | 1.513 | noisy |
| 71 | ojas-wgpu | silu_bwd | 0.86x [0.47-1.00] | 1.004 | 0.691 | noisy |
| 72 | ojas-wgpu | permute_bthd_bhtd | 0.89x [0.60-1.53] | 0.628 | 0.547 | noisy |
| 73 | ojas-wgpu | rms_qk_norm_bwd | 0.95x [0.88-1.00] | 5.421 | 5.276 |  |
| 74 | ojas-metal | cross_entropy_fwd | 0.97x [0.76-1.06] | 20.94 | 19.60 | noisy |
| 75 | ojas-wgpu | mul_bwd | 1.06x [0.65-1.16] | 1.263 | 1.387 | noisy |
| 76 | ojas-metal | sdpa_b4h12t1024d64_bwd | 1.24x [1.14-1.37] | 8.111 | 10.30 | noisy |
| 77 | ojas-wgpu | vres_fwd | 1.32x [1.14-1.48] | 0.458 | 0.650 | noisy |
| 78 | ojas-metal | sdpa_b4h8t2048d64_bwd | 1.38x [1.01-1.59] | 19.41 | 25.39 | noisy |
| 79 | ojas-wgpu | vres_bwd | 1.67x [1.56-2.17] | 0.720 | 1.192 | noisy |
| 80 | ojas-metal | rope_bwd | 1.81x [0.87-2.38] | 1.182 | 2.081 | noisy |
| 81 | ojas-wgpu | cross_entropy_fwd | 2.79x [2.14-2.99] | 7.139 | 19.60 | noisy |
| 82 | ojas-wgpu | rms_norm_bwd | 3.42x [1.67-3.91] | 0.896 | 2.515 | noisy |
| 83 | ojas-wgpu | rope_fwd | 3.67x [3.15-4.45] | 0.470 | 1.513 | noisy |
| 84 | ojas-metal | clip_grad_norm_full | 4.58x [4.46-4.75] | 33.67 | 154.2 |  |
| 85 | ojas-wgpu | rope_bwd | 4.73x [3.88-6.78] | 0.423 | 2.081 | noisy |
| 86 | ojas-wgpu | clip_grad_norm_full | 6.36x [6.14-6.38] | 24.18 | 154.2 |  |

### Machine load around each runtime block

| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |
| --: | :-- | :-- | :-- | :-- | --: |
| 1 | ojas-metal | before | 16:34:31 | 18.26 32.78 37.41 | 93 |
| 1 | ojas-metal | after | 16:35:21 | 15.11 29.87 36.10 | 97 |
| 1 | ojas-wgpu | before | 16:35:21 | 15.11 29.87 36.10 | 70 |
| 1 | ojas-wgpu | after | 16:36:54 | 13.35 25.32 33.68 | 99 |
| 1 | torch-mps | before | 16:36:54 | 13.35 25.32 33.68 | 100 |
| 1 | torch-mps | after | 16:37:27 | 29.41 28.10 34.36 | 95 |
| 2 | torch-mps | before | 16:37:27 | 29.41 28.10 34.36 | 93 |
| 2 | torch-mps | after | 16:37:57 | 24.07 26.99 33.74 | 97 |
| 2 | ojas-wgpu | before | 16:37:57 | 24.07 26.99 33.74 | 95 |
| 2 | ojas-wgpu | after | 16:39:24 | 15.04 23.46 31.75 | 99 |
| 2 | ojas-metal | before | 16:39:24 | 15.04 23.46 31.75 | 97 |
| 2 | ojas-metal | after | 16:40:11 | 13.00 21.72 30.68 | 96 |
| 3 | ojas-metal | before | 16:40:11 | 13.00 21.72 30.68 | 94 |
| 3 | ojas-metal | after | 16:40:56 | 13.24 20.57 29.79 | 96 |
| 3 | ojas-wgpu | before | 16:40:56 | 13.24 20.57 29.79 | 94 |
| 3 | ojas-wgpu | after | 16:42:21 | 11.71 18.27 28.03 | 99 |
| 3 | torch-mps | before | 16:42:21 | 11.71 18.27 28.03 | 94 |
| 3 | torch-mps | after | 16:42:48 | 11.27 17.54 27.43 | 98 |
| 4 | torch-mps | before | 16:42:48 | 11.27 17.54 27.43 | 95 |
| 4 | torch-mps | after | 16:43:15 | 11.69 17.13 26.99 | 98 |
| 4 | ojas-wgpu | before | 16:43:15 | 11.69 17.13 26.99 | 93 |
| 4 | ojas-wgpu | after | 16:44:39 | 10.86 15.61 25.48 | 99 |
| 4 | ojas-metal | before | 16:44:39 | 10.86 15.61 25.48 | 95 |
| 4 | ojas-metal | after | 16:45:24 | 16.86 16.69 25.37 | 96 |
| 5 | ojas-metal | before | 16:45:24 | 16.86 16.69 25.37 | 93 |
| 5 | ojas-metal | after | 16:46:10 | 15.37 16.21 24.74 | 96 |
| 5 | ojas-wgpu | before | 16:46:10 | 15.37 16.21 24.74 | 94 |
| 5 | ojas-wgpu | after | 16:47:35 | 13.28 15.17 23.52 | 99 |
| 5 | torch-mps | before | 16:47:35 | 13.28 15.17 23.52 | 95 |
| 5 | torch-mps | after | 16:48:02 | 13.60 15.07 23.19 | 98 |
