# Paired GPU-vs-torch summary: /Users/bharath/Code/research/ojas/bench/out/run5c

5 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.

```
date: 2026-10-01T22:05:50Z
git: dab2a12fd2129565515954924e1a11737ca6ee47 (dirty files: 168)
uncommitted diff of the benchmarked crates (sha1): c226ffba076d
tessl: cf65d9d5d3deb97b0847e020a562ac8f15e6d5a9 (dirty files: 20)
rustc: rustc 1.98.0 (88d9e12ae 2026-08-18)
cargo: cargo 1.98.0 (797e8a9bc 2026-08-05)
os: macOS 27.0.1 (26A434)
cpu: Apple M5 Pro
memory_bytes: 68719476736
metal_bin: Oct  1 17:05:50 2026 3689360
wgpu_bin: Oct  1 17:05:50 2026 6198064
rounds: 5  iters: 30  warmup: 5  rows: all
```

### ojas-metal vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.130 / 1.699 | 0.083 / 0.146 | 0.08x [0.07-0.27] | 224% / 47% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_fwd | 1.085 / 1.926 | 0.799 / 0.926 | 0.49x [0.31-0.59] | 55% / 36% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_bwd | 1.898 / 2.924 | 1.598 / 2.050 | 0.73x [0.54-0.91] | 64% / 29% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_fwd | 2.452 / 3.846 | 1.916 / 2.213 | 0.58x [0.48-0.79] | 60% / 14% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_bwd | 4.178 / 5.781 | 3.859 / 4.511 | 0.77x [0.73-0.87] | 29% / 10% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_fwd | 2.291 / 3.652 | 1.965 / 2.245 | 0.62x [0.59-0.88] | 57% / 12% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_bwd | 4.570 / 6.691 | 3.742 / 4.447 | 0.65x [0.62-0.80] | 35% / 13% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_fwd | 108.8 / 133.7 | 50.34 / 57.44 | 0.43x [0.39-0.46] | 22% / 19% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_bwd | 134.4 / 176.0 | 100.5 / 110.0 | 0.62x [0.61-0.71] | 15% / 9% | noisy - not quoted | 1.57e-04 (1.3e-05) |
| sdpa_b4h12t1024d64_fwd | 1.953 / 2.976 | 1.244 / 1.430 | 0.54x [0.40-0.64] | 83% / 22% | noisy - not quoted | 2.24e-07 (2.2e-07) |
| sdpa_b4h12t1024d64_bwd | 5.722 / 7.410 | 9.135 / 10.40 | 1.40x [1.31-1.57] | 31% / 11% | noisy - not quoted | 1.10e-06 (2.1e-06) |
| sdpa_b4h8t2048d64_fwd | 4.340 / 6.039 | 3.023 / 3.441 | 0.59x [0.51-0.68] | 37% / 14% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b4h8t2048d64_bwd | 14.45 / 17.54 | 23.48 / 25.42 | 1.49x [1.45-1.57] | 20% / 12% | noisy - not quoted | 1.31e-06 (2.5e-06) |
| sdpa_b2h8t1024d128_fwd | 1.235 / 1.777 | 1.016 / 1.182 | 0.73x [0.38-0.86] | 128% / 22% | noisy - not quoted | 2.09e-07 (2.1e-07) |
| sdpa_b2h8t1024d128_bwd | 4.573 / 5.825 | 4.213 / 5.312 | 0.89x [0.81-0.99] | 25% / 17% | noisy - not quoted | 1.10e-06 (2.6e-06) |
| rms_norm_fwd | 0.478 / 1.460 | 0.181 / 0.234 | 0.17x [0.10-0.28] | 199% / 25% | noisy - not quoted | 3.58e-07 (2.0e-07) |
| rms_norm_bwd | 1.740 / 3.402 | 1.810 / 2.490 | 0.72x [0.67-1.04] | 99% / 30% | noisy - not quoted | 2.14e-04 (1.5e-06) |
| rms_qk_norm_fwd | 1.491 / 3.980 | 0.316 / 0.463 | 0.14x [0.12-0.18] | 90% / 130% | noisy - not quoted | 3.58e-07 (1.7e-07) |
| rms_qk_norm_bwd | 32.50 / 53.23 | 3.912 / 5.296 | 0.10x [0.10-0.15] | 63% / 37% | noisy - not quoted | 1.30e-03 (4.0e-06) |
| rope_fwd | 0.350 / 1.014 | 1.139 / 1.584 | 1.70x [0.72-2.64] | 369% / 43% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| rope_bwd | 0.378 / 0.928 | 1.391 / 2.126 | 2.20x [0.97-3.06] | 344% / 51% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| silu_fwd | 0.847 / 1.945 | 0.354 / 0.446 | 0.24x [0.15-0.50] | 268% / 24% | noisy - not quoted | 2.38e-07 (6.1e-08) |
| silu_bwd | 1.167 / 2.775 | 0.500 / 0.611 | 0.22x [0.21-0.49] | 180% / 29% | noisy - not quoted | 1.19e-07 (1.1e-07) |
| mul_fwd | 1.159 / 3.145 | 0.502 / 0.640 | 0.20x [0.17-0.58] | 205% / 32% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| mul_bwd | 1.813 / 4.606 | 0.910 / 1.320 | 0.30x [0.25-0.71] | 157% / 34% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_fwd | 0.610 / 1.498 | 0.233 / 0.307 | 0.22x [0.18-0.28] | 102% / 72% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_bwd | 0.572 / 1.504 | 0.011 / 0.013 | 0.01x [0.00-0.04] | 281% / 98% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| gate_fwd | 0.547 / 1.621 | 0.414 / 0.594 | 0.52x [0.23-0.91] | 279% / 93% | noisy - not quoted | 8.94e-08 (1.0e-07) |
| gate_bwd | 2.027 / 4.154 | 0.737 / 1.280 | 0.38x [0.26-0.64] | 117% / 100% | noisy - not quoted | 2.25e-04 (2.3e-06) |
| vres_fwd | 0.514 / 2.039 | 0.449 / 0.628 | 0.31x [0.23-1.06] | 139% / 128% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| vres_bwd | 0.890 / 2.267 | 0.704 / 1.216 | 0.71x [0.35-0.94] | 191% / 106% | noisy - not quoted | 3.05e-05 (2.3e-07) |
| permute_bthd_bhtd | 0.376 / 1.458 | 0.364 / 0.457 | 0.31x [0.20-0.68] | 224% / 68% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_fwd | 11.53 / 20.53 | 13.04 / 19.80 | 0.96x [0.88-1.47] | 84% / 44% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_bwd | 54.71 / 92.20 | 23.68 / 37.23 | 0.37x [0.37-0.53] | 72% / 54% | noisy - not quoted | 6.39e-14 (2.6e-10) |
| clip_grad_norm_full | 15.97 / 35.43 | 132.2 / 154.3 | 4.36x [4.29-9.25] | 119% / 18% | noisy - not quoted | 3.05e-05 (3.0e-07) |
| adamw_full | 386.6 / 550.6 | 33.20 / 54.41 | 0.10x [0.07-0.11] | 13% / 69% | noisy - not quoted | 3.73e-09 (1.2e-07) |
| muon_768x768 | 3.243 / 6.302 | 3.347 / 4.746 | 0.74x [0.63-0.97] | 39% / 34% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| muon_768x768 vs muon_768x768_bf16 | 3.243 / 6.302 | 1.728 / 2.858 | 0.46x [0.32-0.58] | 39% / 69% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| muon_2048x768 | 7.134 / 12.78 | 7.262 / 9.529 | 0.76x [0.68-1.08] | 51% / 26% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_2048x768 vs muon_2048x768_bf16 | 7.134 / 12.78 | 4.146 / 5.713 | 0.47x [0.43-0.64] | 51% / 36% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_768x2048 | 6.947 / 12.16 | 7.436 / 9.154 | 0.79x [0.75-0.91] | 37% / 25% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| muon_768x2048 vs muon_768x2048_bf16 | 6.947 / 12.16 | 3.622 / 5.471 | 0.45x [0.41-0.55] | 37% / 39% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| block_fwd | 40.90 / 63.20 | 14.38 / 16.47 | 0.26x [0.24-0.30] | 34% / 13% | noisy - not quoted | 2.38e-07 (1.9e-07) |
| block_fwd_bwd | 142.5 / 221.8 | 66.60 / 82.18 | 0.38x [0.34-0.50] | 51% / 24% | noisy - not quoted | 9.06e-06 (2.5e-06) |

### ojas-wgpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.172 / 0.283 | 0.083 / 0.146 | 0.43x [0.29-0.62] | 89% / 47% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_fwd | 2.349 / 2.741 | 0.799 / 0.926 | 0.36x [0.32-0.37] | 23% / 36% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_bwd | 4.844 / 5.509 | 1.598 / 2.050 | 0.35x [0.31-0.41] | 13% / 29% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_fwd | 5.628 / 6.465 | 1.916 / 2.213 | 0.33x [0.30-0.36] | 24% / 14% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_bwd | 12.10 / 13.02 | 3.859 / 4.511 | 0.32x [0.27-0.35] | 30% / 10% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_fwd | 5.974 / 6.797 | 1.965 / 2.245 | 0.31x [0.27-0.34] | 32% / 12% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_bwd | 11.70 / 12.38 | 3.742 / 4.447 | 0.34x [0.29-0.36] | 28% / 13% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_fwd | 165.9 / 181.2 | 50.34 / 57.44 | 0.30x [0.29-0.33] | 20% / 19% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_bwd | 292.8 / 322.2 | 100.5 / 110.0 | 0.35x [0.33-0.36] | 13% / 9% | noisy - not quoted | 1.57e-04 (1.3e-05) |
| sdpa_b4h12t1024d64_fwd | 72.92 / 77.40 | 1.244 / 1.430 | 0.02x [0.02-0.02] | 3% / 22% | noisy - not quoted | 2.24e-07 (2.2e-07) |
| sdpa_b4h12t1024d64_bwd | 264.0 / 281.4 | 9.135 / 10.40 | 0.04x [0.04-0.04] | 5% / 11% | noisy - not quoted | 1.25e-06 (2.4e-06) |
| sdpa_b4h8t2048d64_fwd | 182.1 / 192.2 | 3.023 / 3.441 | 0.02x [0.02-0.02] | 1% / 14% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b4h8t2048d64_bwd | 686.3 / 723.1 | 23.48 / 25.42 | 0.04x [0.03-0.04] | 4% / 12% | noisy - not quoted | 1.37e-06 (2.6e-06) |
| sdpa_b2h8t1024d128_fwd | 87.46 / 92.35 | 1.016 / 1.182 | 0.01x [0.01-0.01] | 2% / 22% | noisy - not quoted | 2.09e-07 (2.1e-07) |
| sdpa_b2h8t1024d128_bwd | 226.5 / 283.5 | 4.213 / 5.312 | 0.02x [0.02-0.02] | 25% / 17% | noisy - not quoted | 1.07e-06 (2.3e-06) |
| rms_norm_fwd | 0.292 / 0.378 | 0.181 / 0.234 | 0.59x [0.34-0.66] | 117% / 25% | noisy - not quoted | 2.38e-07 (1.3e-07) |
| rms_norm_bwd | 0.507 / 0.763 | 1.810 / 2.490 | 3.10x [2.40-4.10] | 75% / 30% | noisy - not quoted | 3.34e-05 (2.6e-07) |
| rms_qk_norm_fwd | 1.877 / 2.798 | 0.316 / 0.463 | 0.18x [0.16-0.33] | 70% / 130% | noisy - not quoted | 2.38e-07 (1.2e-07) |
| rms_qk_norm_bwd | 3.784 / 5.573 | 3.912 / 5.296 | 1.01x [0.86-1.02] | 48% / 37% | noisy - not quoted | 4.58e-04 (1.3e-06) |
| rope_fwd | 0.232 / 0.311 | 1.139 / 1.584 | 4.35x [2.78-5.32] | 103% / 43% | noisy - not quoted | 1.19e-07 (6.2e-08) |
| rope_bwd | 0.253 / 0.380 | 1.391 / 2.126 | 5.24x [3.57-6.46] | 109% / 51% | noisy - not quoted | 1.19e-07 (6.1e-08) |
| silu_fwd | 0.415 / 0.619 | 0.354 / 0.446 | 0.72x [0.32-0.84] | 225% / 24% | noisy - not quoted | 7.15e-07 (1.8e-07) |
| silu_bwd | 0.555 / 0.827 | 0.500 / 0.611 | 0.75x [0.54-0.98] | 112% / 29% | noisy - not quoted | 4.77e-07 (4.3e-07) |
| mul_fwd | 0.572 / 0.799 | 0.502 / 0.640 | 0.82x [0.56-0.88] | 88% / 32% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| mul_bwd | 0.845 / 1.326 | 0.910 / 1.320 | 1.00x [0.86-1.11] | 68% / 34% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_fwd | 0.307 / 0.444 | 0.233 / 0.307 | 0.81x [0.41-1.12] | 111% / 72% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_bwd | 0.515 / 0.902 | 0.011 / 0.013 | 0.01x [0.01-0.04] | 89% / 98% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| gate_fwd | 0.578 / 0.759 | 0.414 / 0.594 | 0.75x [0.50-0.89] | 84% / 93% | noisy - not quoted | 1.19e-07 (1.4e-07) |
| gate_bwd | 1.545 / 2.090 | 0.737 / 1.280 | 0.57x [0.49-0.77] | 29% / 100% | noisy - not quoted | 2.21e-04 (2.3e-06) |
| vres_fwd | 0.335 / 0.530 | 0.449 / 0.628 | 1.16x [1.01-2.07] | 63% / 128% | noisy - not quoted | 5.96e-08 (6.0e-08) |
| vres_bwd | 0.522 / 0.644 | 0.704 / 1.216 | 1.54x [1.25-1.98] | 78% / 106% | noisy - not quoted | 3.05e-05 (2.3e-07) |
| permute_bthd_bhtd | 0.383 / 0.648 | 0.364 / 0.457 | 0.77x [0.59-0.91] | 157% / 68% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_fwd | 6.373 / 7.008 | 13.04 / 19.80 | 2.54x [2.00-2.92] | 23% / 44% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_bwd | 47.50 / 62.41 | 23.68 / 37.23 | 0.55x [0.48-0.60] | 36% / 54% | noisy - not quoted | 6.39e-14 (2.6e-10) |
| clip_grad_norm_full | 15.88 / 23.27 | 132.2 / 154.3 | 6.61x [5.81-7.95] | 61% / 18% | noisy - not quoted | 3.81e-05 (3.8e-07) |
| adamw_full | 57.81 / 69.25 | 33.20 / 54.41 | 0.75x [0.55-0.80] | 25% / 69% | noisy - not quoted | 3.73e-09 (1.2e-07) |
| muon_768x768 | 8.457 / 10.20 | 3.347 / 4.746 | 0.46x [0.41-0.47] | 17% / 34% | noisy - not quoted | 1.49e-08 (4.5e-07) |
| muon_768x768 vs muon_768x768_bf16 | 8.457 / 10.20 | 1.728 / 2.858 | 0.28x [0.21-0.30] | 17% / 69% | noisy - not quoted | 1.49e-08 (4.5e-07) |
| muon_2048x768 | 18.08 / 21.70 | 7.262 / 9.529 | 0.44x [0.41-0.44] | 17% / 26% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_2048x768 vs muon_2048x768_bf16 | 18.08 / 21.70 | 4.146 / 5.713 | 0.27x [0.23-0.27] | 17% / 36% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_768x2048 | 17.53 / 20.80 | 7.436 / 9.154 | 0.45x [0.43-0.46] | 19% / 25% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| muon_768x2048 vs muon_768x2048_bf16 | 17.53 / 20.80 | 3.622 / 5.471 | 0.26x [0.23-0.27] | 19% / 39% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| block_fwd | 111.8 / 120.6 | 14.38 / 16.47 | 0.14x [0.13-0.14] | 6% / 13% | noisy - not quoted | 2.38e-07 (1.9e-07) |
| block_fwd_bwd | 450.4 / 486.2 | 66.60 / 82.18 | 0.17x [0.15-0.17] | 6% / 24% | noisy - not quoted | 8.58e-06 (2.3e-06) |

### Ranked: where ojas is slowest relative to torch (lowest ratio first)

| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |
| --: | :-- | :-- | :-- | --: | --: | :-- |
| 1 | ojas-metal | residual_add_bwd | 0.01x [0.00-0.04] | 1.504 | 0.013 | noisy |
| 2 | ojas-wgpu | sdpa_b2h8t1024d128_fwd | 0.01x [0.01-0.01] | 92.35 | 1.182 | noisy |
| 3 | ojas-wgpu | residual_add_bwd | 0.01x [0.01-0.04] | 0.902 | 0.013 | noisy |
| 4 | ojas-wgpu | sdpa_b4h8t2048d64_fwd | 0.02x [0.02-0.02] | 192.2 | 3.441 | noisy |
| 5 | ojas-wgpu | sdpa_b4h12t1024d64_fwd | 0.02x [0.02-0.02] | 77.40 | 1.430 | noisy |
| 6 | ojas-wgpu | sdpa_b2h8t1024d128_bwd | 0.02x [0.02-0.02] | 283.5 | 5.312 | noisy |
| 7 | ojas-wgpu | sdpa_b4h8t2048d64_bwd | 0.04x [0.03-0.04] | 723.1 | 25.42 | noisy |
| 8 | ojas-wgpu | sdpa_b4h12t1024d64_bwd | 0.04x [0.04-0.04] | 281.4 | 10.40 | noisy |
| 9 | ojas-metal | adamw_full | 0.10x [0.07-0.11] | 550.6 | 54.41 | noisy |
| 10 | ojas-metal | rms_qk_norm_bwd | 0.10x [0.10-0.15] | 53.23 | 5.296 | noisy |
| 11 | ojas-wgpu | block_fwd | 0.14x [0.13-0.14] | 120.6 | 16.47 | noisy |
| 12 | ojas-metal | rms_qk_norm_fwd | 0.14x [0.12-0.18] | 3.980 | 0.463 | noisy |
| 13 | ojas-wgpu | block_fwd_bwd | 0.17x [0.15-0.17] | 486.2 | 82.18 | noisy |
| 14 | ojas-metal | rms_norm_fwd | 0.17x [0.10-0.28] | 1.460 | 0.234 | noisy |
| 15 | ojas-wgpu | rms_qk_norm_fwd | 0.18x [0.16-0.33] | 2.798 | 0.463 | noisy |
| 16 | ojas-metal | mul_fwd | 0.20x [0.17-0.58] | 3.145 | 0.640 | noisy |
| 17 | ojas-metal | residual_add_fwd | 0.22x [0.18-0.28] | 1.498 | 0.307 | noisy |
| 18 | ojas-metal | silu_bwd | 0.22x [0.21-0.49] | 2.775 | 0.611 | noisy |
| 19 | ojas-metal | silu_fwd | 0.24x [0.15-0.50] | 1.945 | 0.446 | noisy |
| 20 | ojas-wgpu | muon_768x2048 vs muon_768x2048_bf16 | 0.26x [0.23-0.27] | 20.80 | 5.471 | noisy |
| 21 | ojas-metal | block_fwd | 0.26x [0.24-0.30] | 63.20 | 16.47 | noisy |
| 22 | ojas-wgpu | muon_2048x768 vs muon_2048x768_bf16 | 0.27x [0.23-0.27] | 21.70 | 5.713 | noisy |
| 23 | ojas-wgpu | muon_768x768 vs muon_768x768_bf16 | 0.28x [0.21-0.30] | 10.20 | 2.858 | noisy |
| 24 | ojas-wgpu | linear_lmhead_fwd | 0.30x [0.29-0.33] | 181.2 | 57.44 | noisy |
| 25 | ojas-metal | mul_bwd | 0.30x [0.25-0.71] | 4.606 | 1.320 | noisy |
| 26 | ojas-metal | vres_fwd | 0.31x [0.23-1.06] | 2.039 | 0.628 | noisy |
| 27 | ojas-wgpu | linear_down_fwd | 0.31x [0.27-0.34] | 6.797 | 2.245 | noisy |
| 28 | ojas-metal | permute_bthd_bhtd | 0.31x [0.20-0.68] | 1.458 | 0.457 | noisy |
| 29 | ojas-wgpu | linear_up_bwd | 0.32x [0.27-0.35] | 13.02 | 4.511 | noisy |
| 30 | ojas-wgpu | linear_up_fwd | 0.33x [0.30-0.36] | 6.465 | 2.213 | noisy |
| 31 | ojas-wgpu | linear_down_bwd | 0.34x [0.29-0.36] | 12.38 | 4.447 | noisy |
| 32 | ojas-wgpu | linear_lmhead_bwd | 0.35x [0.33-0.36] | 322.2 | 110.0 | noisy |
| 33 | ojas-wgpu | linear_qkv_bwd | 0.35x [0.31-0.41] | 5.509 | 2.050 | noisy |
| 34 | ojas-wgpu | linear_qkv_fwd | 0.36x [0.32-0.37] | 2.741 | 0.926 | noisy |
| 35 | ojas-metal | cross_entropy_bwd | 0.37x [0.37-0.53] | 92.20 | 37.23 | noisy |
| 36 | ojas-metal | gate_bwd | 0.38x [0.26-0.64] | 4.154 | 1.280 | noisy |
| 37 | ojas-metal | block_fwd_bwd | 0.38x [0.34-0.50] | 221.8 | 82.18 | noisy |
| 38 | ojas-metal | linear_lmhead_fwd | 0.43x [0.39-0.46] | 133.7 | 57.44 | noisy |
| 39 | ojas-wgpu | muon_2048x768 | 0.44x [0.41-0.44] | 21.70 | 9.529 | noisy |
| 40 | ojas-wgpu | muon_768x2048 | 0.45x [0.43-0.46] | 20.80 | 9.154 | noisy |
| 41 | ojas-metal | muon_768x2048 vs muon_768x2048_bf16 | 0.45x [0.41-0.55] | 12.16 | 5.471 | noisy |
| 42 | ojas-metal | muon_768x768 vs muon_768x768_bf16 | 0.46x [0.32-0.58] | 6.302 | 2.858 | noisy |
| 43 | ojas-wgpu | muon_768x768 | 0.46x [0.41-0.47] | 10.20 | 4.746 | noisy |
| 44 | ojas-metal | muon_2048x768 vs muon_2048x768_bf16 | 0.47x [0.43-0.64] | 12.78 | 5.713 | noisy |
| 45 | ojas-metal | linear_qkv_fwd | 0.49x [0.31-0.59] | 1.926 | 0.926 | noisy |
| 46 | ojas-metal | gate_fwd | 0.52x [0.23-0.91] | 1.621 | 0.594 | noisy |
| 47 | ojas-metal | sdpa_b4h12t1024d64_fwd | 0.54x [0.40-0.64] | 2.976 | 1.430 | noisy |
| 48 | ojas-wgpu | cross_entropy_bwd | 0.55x [0.48-0.60] | 62.41 | 37.23 | noisy |
| 49 | ojas-wgpu | gate_bwd | 0.57x [0.49-0.77] | 2.090 | 1.280 | noisy |
| 50 | ojas-metal | linear_up_fwd | 0.58x [0.48-0.79] | 3.846 | 2.213 | noisy |
| 51 | ojas-metal | sdpa_b4h8t2048d64_fwd | 0.59x [0.51-0.68] | 6.039 | 3.441 | noisy |
| 52 | ojas-wgpu | rms_norm_fwd | 0.59x [0.34-0.66] | 0.378 | 0.234 | noisy |
| 53 | ojas-metal | linear_down_fwd | 0.62x [0.59-0.88] | 3.652 | 2.245 | noisy |
| 54 | ojas-metal | linear_lmhead_bwd | 0.62x [0.61-0.71] | 176.0 | 110.0 | noisy |
| 55 | ojas-metal | linear_down_bwd | 0.65x [0.62-0.80] | 6.691 | 4.447 | noisy |
| 56 | ojas-metal | vres_bwd | 0.71x [0.35-0.94] | 2.267 | 1.216 | noisy |
| 57 | ojas-metal | rms_norm_bwd | 0.72x [0.67-1.04] | 3.402 | 2.490 | noisy |
| 58 | ojas-wgpu | silu_fwd | 0.72x [0.32-0.84] | 0.619 | 0.446 | noisy |
| 59 | ojas-metal | linear_qkv_bwd | 0.73x [0.54-0.91] | 2.924 | 2.050 | noisy |
| 60 | ojas-metal | sdpa_b2h8t1024d128_fwd | 0.73x [0.38-0.86] | 1.777 | 1.182 | noisy |
| 61 | ojas-metal | muon_768x768 | 0.74x [0.63-0.97] | 6.302 | 4.746 | noisy |
| 62 | ojas-wgpu | silu_bwd | 0.75x [0.54-0.98] | 0.827 | 0.611 | noisy |
| 63 | ojas-wgpu | gate_fwd | 0.75x [0.50-0.89] | 0.759 | 0.594 | noisy |
| 64 | ojas-wgpu | adamw_full | 0.75x [0.55-0.80] | 69.25 | 54.41 | noisy |
| 65 | ojas-metal | muon_2048x768 | 0.76x [0.68-1.08] | 12.78 | 9.529 | noisy |
| 66 | ojas-metal | linear_up_bwd | 0.77x [0.73-0.87] | 5.781 | 4.511 | noisy |
| 67 | ojas-wgpu | permute_bthd_bhtd | 0.77x [0.59-0.91] | 0.648 | 0.457 | noisy |
| 68 | ojas-metal | muon_768x2048 | 0.79x [0.75-0.91] | 12.16 | 9.154 | noisy |
| 69 | ojas-wgpu | residual_add_fwd | 0.81x [0.41-1.12] | 0.444 | 0.307 | noisy |
| 70 | ojas-wgpu | mul_fwd | 0.82x [0.56-0.88] | 0.799 | 0.640 | noisy |
| 71 | ojas-metal | sdpa_b2h8t1024d128_bwd | 0.89x [0.81-0.99] | 5.825 | 5.312 | noisy |
| 72 | ojas-metal | cross_entropy_fwd | 0.96x [0.88-1.47] | 20.53 | 19.80 | noisy |
| 73 | ojas-wgpu | mul_bwd | 1.00x [0.86-1.11] | 1.326 | 1.320 | noisy |
| 74 | ojas-wgpu | rms_qk_norm_bwd | 1.01x [0.86-1.02] | 5.573 | 5.296 | noisy |
| 75 | ojas-wgpu | vres_fwd | 1.16x [1.01-2.07] | 0.530 | 0.628 | noisy |
| 76 | ojas-metal | sdpa_b4h12t1024d64_bwd | 1.40x [1.31-1.57] | 7.410 | 10.40 | noisy |
| 77 | ojas-metal | sdpa_b4h8t2048d64_bwd | 1.49x [1.45-1.57] | 17.54 | 25.42 | noisy |
| 78 | ojas-wgpu | vres_bwd | 1.54x [1.25-1.98] | 0.644 | 1.216 | noisy |
| 79 | ojas-metal | rope_fwd | 1.70x [0.72-2.64] | 1.014 | 1.584 | noisy |
| 80 | ojas-metal | rope_bwd | 2.20x [0.97-3.06] | 0.928 | 2.126 | noisy |
| 81 | ojas-wgpu | cross_entropy_fwd | 2.54x [2.00-2.92] | 7.008 | 19.80 | noisy |
| 82 | ojas-wgpu | rms_norm_bwd | 3.10x [2.40-4.10] | 0.763 | 2.490 | noisy |
| 83 | ojas-wgpu | rope_fwd | 4.35x [2.78-5.32] | 0.311 | 1.584 | noisy |
| 84 | ojas-metal | clip_grad_norm_full | 4.36x [4.29-9.25] | 35.43 | 154.3 | noisy |
| 85 | ojas-wgpu | rope_bwd | 5.24x [3.57-6.46] | 0.380 | 2.126 | noisy |
| 86 | ojas-wgpu | clip_grad_norm_full | 6.61x [5.81-7.95] | 23.27 | 154.3 | noisy |

### Machine load around each runtime block

| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |
| --: | :-- | :-- | :-- | :-- | --: |
| 1 | ojas-metal | before | 17:06:07 | 12.33 13.64 16.67 | 57 |
| 1 | ojas-metal | after | 17:06:54 | 13.19 13.70 16.54 | 88 |
| 1 | ojas-wgpu | before | 17:06:54 | 13.19 13.70 16.54 | 0 |
| 1 | ojas-wgpu | after | 17:08:44 | 10.21 12.58 15.76 | 98 |
| 1 | torch-mps | before | 17:08:44 | 10.21 12.58 15.76 | 0 |
| 1 | torch-mps | after | 17:09:12 | 9.48 12.18 15.50 | 96 |
| 2 | torch-mps | before | 17:09:12 | 9.48 12.18 15.50 | 0 |
| 2 | torch-mps | after | 17:09:44 | 9.05 11.81 15.25 | 98 |
| 2 | ojas-wgpu | before | 17:09:44 | 9.05 11.81 15.25 | 93 |
| 2 | ojas-wgpu | after | 17:11:38 | 9.58 11.01 14.49 | 99 |
| 2 | ojas-metal | before | 17:11:38 | 9.58 11.01 14.49 | 94 |
| 2 | ojas-metal | after | 17:12:38 | 13.00 11.59 14.46 | 97 |
| 3 | ojas-metal | before | 17:12:38 | 13.00 11.59 14.46 | 94 |
| 3 | ojas-metal | after | 17:13:41 | 18.52 13.49 14.98 | 97 |
| 3 | ojas-wgpu | before | 17:13:41 | 18.52 13.49 14.98 | 94 |
| 3 | ojas-wgpu | after | 17:15:41 | 16.48 14.71 15.30 | 99 |
| 3 | torch-mps | before | 17:15:41 | 16.48 14.71 15.30 | 85 |
| 3 | torch-mps | after | 17:16:17 | 16.25 14.84 15.31 | 98 |
| 4 | torch-mps | before | 17:16:17 | 16.25 14.84 15.31 | 94 |
| 4 | torch-mps | after | 17:16:53 | 14.08 14.48 15.16 | 98 |
| 4 | ojas-wgpu | before | 17:16:54 | 14.08 14.48 15.16 | 93 |
| 4 | ojas-wgpu | after | 17:18:54 | 14.80 15.47 15.52 | 99 |
| 4 | ojas-metal | before | 17:18:54 | 14.80 15.47 15.52 | 92 |
| 4 | ojas-metal | after | 17:19:56 | 19.03 16.70 15.99 | 97 |
| 5 | ojas-metal | before | 17:19:56 | 19.03 16.70 15.99 | 93 |
| 5 | ojas-metal | after | 17:20:48 | 14.75 15.98 15.77 | 90 |
| 5 | ojas-wgpu | before | 17:20:48 | 14.75 15.98 15.77 | 0 |
| 5 | ojas-wgpu | after | 17:22:43 | 10.84 14.11 15.05 | 99 |
| 5 | torch-mps | before | 17:22:43 | 10.84 14.11 15.05 | 95 |
| 5 | torch-mps | after | 17:23:18 | 11.56 13.91 14.94 | 98 |
