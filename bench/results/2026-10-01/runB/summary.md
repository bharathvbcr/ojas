# Paired GPU-vs-torch summary: /Users/bharath/Code/research/ojas/bench/out/run5b

5 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.

```
date: 2026-10-01T21:48:36Z
git: dab2a12fd2129565515954924e1a11737ca6ee47 (dirty files: 166)
uncommitted diff of the benchmarked crates (sha1): c226ffba076d
tessl: cf65d9d5d3deb97b0847e020a562ac8f15e6d5a9 (dirty files: 20)
rustc: rustc 1.98.0 (88d9e12ae 2026-08-18)
cargo: cargo 1.98.0 (797e8a9bc 2026-08-05)
os: macOS 27.0.1 (26A434)
cpu: Apple M5 Pro
memory_bytes: 68719476736
metal_bin: Oct  1 16:48:35 2026 3689360
wgpu_bin: Oct  1 16:48:36 2026 6198064
rounds: 5  iters: 30  warmup: 5  rows: all
```

### ojas-metal vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.179 / 1.518 | 0.084 / 0.145 | 0.10x [0.06-0.24] | 179% / 37% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_fwd | 1.097 / 2.311 | 0.799 / 0.994 | 0.43x [0.37-0.65] | 58% / 20% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_bwd | 1.877 / 2.628 | 1.601 / 1.837 | 0.70x [0.59-0.98] | 36% / 29% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_fwd | 2.400 / 2.686 | 1.897 / 2.062 | 0.76x [0.56-0.89] | 53% / 12% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_bwd | 4.130 / 4.792 | 3.868 / 4.394 | 0.90x [0.85-1.08] | 23% / 16% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_fwd | 2.272 / 2.621 | 1.960 / 2.186 | 0.78x [0.70-0.92] | 38% / 14% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_bwd | 4.494 / 5.128 | 3.809 / 4.217 | 0.82x [0.68-0.96] | 39% / 13% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_fwd | 108.6 / 121.3 | 50.58 / 55.87 | 0.45x [0.42-0.55] | 21% / 21% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_bwd | 127.5 / 159.1 | 99.72 / 113.4 | 0.69x [0.66-0.83] | 19% / 18% | noisy - not quoted | 1.57e-04 (1.3e-05) |
| sdpa_b4h12t1024d64_fwd | 1.966 / 2.510 | 1.258 / 1.547 | 0.65x [0.42-1.98] | 48% / 259% | noisy - not quoted | 2.24e-07 (2.2e-07) |
| sdpa_b4h12t1024d64_bwd | 5.769 / 7.035 | 9.416 / 10.32 | 1.50x [1.34-1.64] | 26% / 11% | noisy - not quoted | 1.10e-06 (2.1e-06) |
| sdpa_b4h8t2048d64_fwd | 4.352 / 5.463 | 3.017 / 3.245 | 0.59x [0.50-0.76] | 50% / 12% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b4h8t2048d64_bwd | 14.61 / 16.51 | 22.92 / 25.95 | 1.61x [1.40-1.77] | 20% / 17% | noisy - not quoted | 1.31e-06 (2.5e-06) |
| sdpa_b2h8t1024d128_fwd | 1.228 / 1.774 | 1.005 / 1.248 | 0.70x [0.44-1.58] | 104% / 91% | noisy - not quoted | 2.09e-07 (2.1e-07) |
| sdpa_b2h8t1024d128_bwd | 4.429 / 5.598 | 4.202 / 5.220 | 0.92x [0.85-1.11] | 22% / 11% | noisy - not quoted | 1.10e-06 (2.6e-06) |
| rms_norm_fwd | 0.479 / 1.775 | 0.195 / 0.271 | 0.13x [0.12-0.47] | 224% / 44% | noisy - not quoted | 3.58e-07 (2.0e-07) |
| rms_norm_bwd | 1.759 / 2.502 | 1.718 / 2.445 | 0.97x [0.73-1.29] | 70% / 64% | noisy - not quoted | 2.14e-04 (1.5e-06) |
| rms_qk_norm_fwd | 1.519 / 2.866 | 0.307 / 0.437 | 0.18x [0.12-0.33] | 157% / 215% | noisy - not quoted | 3.58e-07 (1.7e-07) |
| rms_qk_norm_bwd | 32.53 / 34.16 | 3.934 / 5.085 | 0.13x [0.10-0.15] | 58% / 15% | noisy - not quoted | 1.30e-03 (4.0e-06) |
| rope_fwd | 0.361 / 1.677 | 1.161 / 1.317 | 0.81x [0.74-1.20] | 97% / 31% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| rope_bwd | 0.394 / 1.694 | 1.454 / 1.736 | 1.01x [0.99-1.79] | 146% / 37% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| silu_fwd | 0.835 / 2.088 | 0.357 / 0.411 | 0.21x [0.17-0.44] | 154% / 31% | noisy - not quoted | 2.38e-07 (6.1e-08) |
| silu_bwd | 1.181 / 2.453 | 0.491 / 0.615 | 0.24x [0.23-0.56] | 100% / 43% | noisy - not quoted | 1.19e-07 (1.1e-07) |
| mul_fwd | 1.168 / 2.457 | 0.460 / 0.536 | 0.24x [0.20-0.76] | 38% / 199% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| mul_bwd | 1.851 / 2.265 | 0.879 / 1.093 | 0.48x [0.22-1.67] | 119% / 285% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_fwd | 0.624 / 1.349 | 0.243 / 0.299 | 0.23x [0.20-1.10] | 122% / 777% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_bwd | 0.601 / 1.874 | 0.009 / 0.015 | 0.01x [0.01-0.02] | 197% / 301% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| gate_fwd | 0.571 / 2.079 | 0.401 / 0.580 | 0.29x [0.28-0.63] | 226% / 85% | noisy - not quoted | 8.94e-08 (1.0e-07) |
| gate_bwd | 1.997 / 2.737 | 0.754 / 1.279 | 0.43x [0.29-0.60] | 96% / 47% | noisy - not quoted | 2.25e-04 (2.3e-06) |
| vres_fwd | 0.524 / 1.811 | 0.462 / 0.571 | 0.32x [0.28-0.55] | 70% / 38% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| vres_bwd | 0.914 / 2.404 | 0.659 / 1.170 | 0.49x [0.37-0.66] | 90% / 51% | noisy - not quoted | 3.05e-05 (2.3e-07) |
| permute_bthd_bhtd | 0.368 / 1.686 | 0.382 / 0.480 | 0.28x [0.25-0.86] | 295% / 36% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_fwd | 11.20 / 12.98 | 12.84 / 19.30 | 1.03x [0.95-1.93] | 79% / 70% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_bwd | 44.91 / 68.31 | 24.49 / 27.19 | 0.38x [0.35-0.56] | 94% / 43% | noisy - not quoted | 6.39e-14 (2.6e-10) |
| clip_grad_norm_full | 16.10 / 18.72 | 132.1 / 135.5 | 7.16x [4.57-9.36] | 107% / 16% | noisy - not quoted | 3.05e-05 (3.0e-07) |
| adamw_full | 370.9 / 587.5 | 33.43 / 49.88 | 0.09x [0.07-0.11] | 40% / 54% | noisy - not quoted | 3.73e-09 (1.2e-07) |
| muon_768x768 | 3.315 / 5.233 | 3.417 / 4.567 | 0.78x [0.68-1.26] | 42% / 67% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| muon_768x768 vs muon_768x768_bf16 | 3.315 / 5.233 | 1.736 / 2.425 | 0.46x [0.43-0.51] | 42% / 44% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| muon_2048x768 | 7.345 / 9.490 | 7.445 / 9.216 | 0.88x [0.77-1.13] | 33% / 33% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_2048x768 vs muon_2048x768_bf16 | 7.345 / 9.490 | 4.352 / 5.822 | 0.50x [0.48-0.76] | 33% / 49% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_768x2048 | 7.014 / 8.673 | 7.350 / 9.100 | 0.94x [0.78-1.17] | 40% / 27% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| muon_768x2048 vs muon_768x2048_bf16 | 7.014 / 8.673 | 3.590 / 5.420 | 0.47x [0.45-0.65] | 40% / 40% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| block_fwd | 37.84 / 62.10 | 14.55 / 16.90 | 0.28x [0.23-0.37] | 25% / 34% | noisy - not quoted | 2.38e-07 (1.9e-07) |
| block_fwd_bwd | 142.4 / 168.9 | 66.75 / 81.77 | 0.43x [0.37-0.60] | 37% / 40% | noisy - not quoted | 9.06e-06 (2.5e-06) |

### ojas-wgpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.156 / 0.350 | 0.084 / 0.145 | 0.42x [0.35-0.59] | 131% / 37% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_fwd | 2.282 / 2.740 | 0.799 / 0.994 | 0.36x [0.30-0.37] | 35% / 20% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_bwd | 4.997 / 5.534 | 1.601 / 1.837 | 0.33x [0.29-0.41] | 21% / 29% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_fwd | 5.542 / 6.745 | 1.897 / 2.062 | 0.32x [0.28-0.35] | 27% / 12% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_bwd | 11.93 / 14.23 | 3.868 / 4.394 | 0.32x [0.30-0.33] | 29% / 16% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_fwd | 5.813 / 7.573 | 1.960 / 2.186 | 0.30x [0.28-0.33] | 35% / 14% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_bwd | 11.29 / 14.13 | 3.809 / 4.217 | 0.32x [0.28-0.34] | 34% / 13% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_fwd | 156.3 / 192.2 | 50.58 / 55.87 | 0.32x [0.27-0.32] | 24% / 21% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_bwd | 280.6 / 321.7 | 99.72 / 113.4 | 0.36x [0.32-0.38] | 23% / 18% | noisy - not quoted | 1.57e-04 (1.3e-05) |
| sdpa_b4h12t1024d64_fwd | 72.80 / 77.28 | 1.258 / 1.547 | 0.02x [0.02-0.06] | 5% / 259% | noisy - not quoted | 2.24e-07 (2.2e-07) |
| sdpa_b4h12t1024d64_bwd | 261.6 / 275.1 | 9.416 / 10.32 | 0.04x [0.04-0.04] | 4% / 11% | noisy - not quoted | 1.25e-06 (2.4e-06) |
| sdpa_b4h8t2048d64_fwd | 182.5 / 193.6 | 3.017 / 3.245 | 0.02x [0.02-0.02] | 3% / 12% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b4h8t2048d64_bwd | 685.2 / 706.2 | 22.92 / 25.95 | 0.04x [0.03-0.04] | 2% / 17% | noisy - not quoted | 1.37e-06 (2.6e-06) |
| sdpa_b2h8t1024d128_fwd | 68.35 / 70.83 | 1.005 / 1.248 | 0.02x [0.01-0.03] | 36% / 91% | noisy - not quoted | 2.09e-07 (2.1e-07) |
| sdpa_b2h8t1024d128_bwd | 223.1 / 234.9 | 4.202 / 5.220 | 0.02x [0.02-0.02] | 26% / 11% | noisy - not quoted | 1.07e-06 (2.3e-06) |
| rms_norm_fwd | 0.283 / 0.380 | 0.195 / 0.271 | 0.70x [0.47-1.08] | 70% / 44% | noisy - not quoted | 2.38e-07 (1.3e-07) |
| rms_norm_bwd | 0.502 / 0.684 | 1.718 / 2.445 | 3.53x [2.32-5.76] | 101% / 64% | noisy - not quoted | 3.34e-05 (2.6e-07) |
| rms_qk_norm_fwd | 1.834 / 2.158 | 0.307 / 0.437 | 0.19x [0.10-0.58] | 114% / 215% | noisy - not quoted | 2.38e-07 (1.2e-07) |
| rms_qk_norm_bwd | 3.771 / 4.152 | 3.934 / 5.085 | 1.08x [0.67-1.28] | 88% / 15% | noisy - not quoted | 4.58e-04 (1.3e-06) |
| rope_fwd | 0.255 / 0.348 | 1.161 / 1.317 | 3.80x [3.51-4.78] | 34% / 31% | noisy - not quoted | 1.19e-07 (6.2e-08) |
| rope_bwd | 0.247 / 0.332 | 1.454 / 1.736 | 5.23x [4.41-6.49] | 26% / 37% | noisy - not quoted | 1.19e-07 (6.1e-08) |
| silu_fwd | 0.437 / 0.580 | 0.357 / 0.411 | 0.70x [0.67-1.01] | 19% / 31% | noisy - not quoted | 7.15e-07 (1.8e-07) |
| silu_bwd | 0.563 / 0.790 | 0.491 / 0.615 | 0.78x [0.65-1.28] | 54% / 43% | noisy - not quoted | 4.77e-07 (4.3e-07) |
| mul_fwd | 0.559 / 0.749 | 0.460 / 0.536 | 0.77x [0.59-2.47] | 39% / 199% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| mul_bwd | 0.827 / 1.120 | 0.879 / 1.093 | 0.95x [0.77-4.36] | 51% / 285% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_fwd | 0.320 / 0.453 | 0.243 / 0.299 | 0.66x [0.58-7.17] | 40% / 777% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_bwd | 0.498 / 0.784 | 0.009 / 0.015 | 0.02x [0.01-0.09] | 61% / 301% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| gate_fwd | 0.526 / 0.811 | 0.401 / 0.580 | 0.72x [0.62-1.61] | 69% / 85% | noisy - not quoted | 1.19e-07 (1.4e-07) |
| gate_bwd | 1.508 / 1.872 | 0.754 / 1.279 | 0.59x [0.40-0.81] | 48% / 47% | noisy - not quoted | 2.21e-04 (2.3e-06) |
| vres_fwd | 0.309 / 0.459 | 0.462 / 0.571 | 1.20x [1.09-1.91] | 42% / 38% | noisy - not quoted | 5.96e-08 (6.0e-08) |
| vres_bwd | 0.472 / 0.605 | 0.659 / 1.170 | 1.46x [1.16-2.10] | 42% / 51% | noisy - not quoted | 3.05e-05 (2.3e-07) |
| permute_bthd_bhtd | 0.387 / 0.552 | 0.382 / 0.480 | 0.92x [0.77-1.19] | 36% / 36% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_fwd | 6.146 / 7.027 | 12.84 / 19.30 | 2.61x [1.93-3.21] | 8% / 70% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_bwd | 36.66 / 50.32 | 24.49 / 27.19 | 0.58x [0.54-0.87] | 56% / 43% | noisy - not quoted | 6.39e-14 (2.6e-10) |
| clip_grad_norm_full | 15.91 / 16.95 | 132.1 / 135.5 | 7.95x [6.86-9.10] | 34% / 16% | noisy - not quoted | 3.81e-05 (3.8e-07) |
| adamw_full | 47.98 / 63.29 | 33.43 / 49.88 | 0.75x [0.56-1.06] | 46% / 54% | noisy - not quoted | 3.73e-09 (1.2e-07) |
| muon_768x768 | 8.483 / 9.077 | 3.417 / 4.567 | 0.45x [0.40-0.68] | 14% / 67% | noisy - not quoted | 1.49e-08 (4.5e-07) |
| muon_768x768 vs muon_768x768_bf16 | 8.483 / 9.077 | 1.736 / 2.425 | 0.27x [0.22-0.32] | 14% / 44% | noisy - not quoted | 1.49e-08 (4.5e-07) |
| muon_2048x768 | 17.59 / 19.03 | 7.445 / 9.216 | 0.43x [0.41-0.54] | 22% / 33% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_2048x768 vs muon_2048x768_bf16 | 17.59 / 19.03 | 4.352 / 5.822 | 0.26x [0.25-0.36] | 22% / 49% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_768x2048 | 17.23 / 17.96 | 7.350 / 9.100 | 0.44x [0.43-0.57] | 21% / 27% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| muon_768x2048 vs muon_768x2048_bf16 | 17.23 / 17.96 | 3.590 / 5.420 | 0.25x [0.22-0.31] | 21% / 40% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| block_fwd | 109.8 / 116.2 | 14.55 / 16.90 | 0.14x [0.13-0.18] | 6% / 34% | noisy - not quoted | 2.38e-07 (1.9e-07) |
| block_fwd_bwd | 448.3 / 466.9 | 66.75 / 81.77 | 0.17x [0.15-0.21] | 3% / 40% | noisy - not quoted | 8.58e-06 (2.3e-06) |

### Ranked: where ojas is slowest relative to torch (lowest ratio first)

| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |
| --: | :-- | :-- | :-- | --: | --: | :-- |
| 1 | ojas-metal | residual_add_bwd | 0.01x [0.01-0.02] | 1.874 | 0.015 | noisy |
| 2 | ojas-wgpu | sdpa_b2h8t1024d128_fwd | 0.02x [0.01-0.03] | 70.83 | 1.248 | noisy |
| 3 | ojas-wgpu | sdpa_b4h8t2048d64_fwd | 0.02x [0.02-0.02] | 193.6 | 3.245 | noisy |
| 4 | ojas-wgpu | residual_add_bwd | 0.02x [0.01-0.09] | 0.784 | 0.015 | noisy |
| 5 | ojas-wgpu | sdpa_b4h12t1024d64_fwd | 0.02x [0.02-0.06] | 77.28 | 1.547 | noisy |
| 6 | ojas-wgpu | sdpa_b2h8t1024d128_bwd | 0.02x [0.02-0.02] | 234.9 | 5.220 | noisy |
| 7 | ojas-wgpu | sdpa_b4h8t2048d64_bwd | 0.04x [0.03-0.04] | 706.2 | 25.95 | noisy |
| 8 | ojas-wgpu | sdpa_b4h12t1024d64_bwd | 0.04x [0.04-0.04] | 275.1 | 10.32 | noisy |
| 9 | ojas-metal | adamw_full | 0.09x [0.07-0.11] | 587.5 | 49.88 | noisy |
| 10 | ojas-metal | rms_norm_fwd | 0.13x [0.12-0.47] | 1.775 | 0.271 | noisy |
| 11 | ojas-metal | rms_qk_norm_bwd | 0.13x [0.10-0.15] | 34.16 | 5.085 | noisy |
| 12 | ojas-wgpu | block_fwd | 0.14x [0.13-0.18] | 116.2 | 16.90 | noisy |
| 13 | ojas-wgpu | block_fwd_bwd | 0.17x [0.15-0.21] | 466.9 | 81.77 | noisy |
| 14 | ojas-metal | rms_qk_norm_fwd | 0.18x [0.12-0.33] | 2.866 | 0.437 | noisy |
| 15 | ojas-wgpu | rms_qk_norm_fwd | 0.19x [0.10-0.58] | 2.158 | 0.437 | noisy |
| 16 | ojas-metal | silu_fwd | 0.21x [0.17-0.44] | 2.088 | 0.411 | noisy |
| 17 | ojas-metal | residual_add_fwd | 0.23x [0.20-1.10] | 1.349 | 0.299 | noisy |
| 18 | ojas-metal | silu_bwd | 0.24x [0.23-0.56] | 2.453 | 0.615 | noisy |
| 19 | ojas-metal | mul_fwd | 0.24x [0.20-0.76] | 2.457 | 0.536 | noisy |
| 20 | ojas-wgpu | muon_768x2048 vs muon_768x2048_bf16 | 0.25x [0.22-0.31] | 17.96 | 5.420 | noisy |
| 21 | ojas-wgpu | muon_2048x768 vs muon_2048x768_bf16 | 0.26x [0.25-0.36] | 19.03 | 5.822 | noisy |
| 22 | ojas-wgpu | muon_768x768 vs muon_768x768_bf16 | 0.27x [0.22-0.32] | 9.077 | 2.425 | noisy |
| 23 | ojas-metal | block_fwd | 0.28x [0.23-0.37] | 62.10 | 16.90 | noisy |
| 24 | ojas-metal | permute_bthd_bhtd | 0.28x [0.25-0.86] | 1.686 | 0.480 | noisy |
| 25 | ojas-metal | gate_fwd | 0.29x [0.28-0.63] | 2.079 | 0.580 | noisy |
| 26 | ojas-wgpu | linear_down_fwd | 0.30x [0.28-0.33] | 7.573 | 2.186 | noisy |
| 27 | ojas-metal | vres_fwd | 0.32x [0.28-0.55] | 1.811 | 0.571 | noisy |
| 28 | ojas-wgpu | linear_up_bwd | 0.32x [0.30-0.33] | 14.23 | 4.394 | noisy |
| 29 | ojas-wgpu | linear_lmhead_fwd | 0.32x [0.27-0.32] | 192.2 | 55.87 | noisy |
| 30 | ojas-wgpu | linear_up_fwd | 0.32x [0.28-0.35] | 6.745 | 2.062 | noisy |
| 31 | ojas-wgpu | linear_down_bwd | 0.32x [0.28-0.34] | 14.13 | 4.217 | noisy |
| 32 | ojas-wgpu | linear_qkv_bwd | 0.33x [0.29-0.41] | 5.534 | 1.837 | noisy |
| 33 | ojas-wgpu | linear_qkv_fwd | 0.36x [0.30-0.37] | 2.740 | 0.994 | noisy |
| 34 | ojas-wgpu | linear_lmhead_bwd | 0.36x [0.32-0.38] | 321.7 | 113.4 | noisy |
| 35 | ojas-metal | cross_entropy_bwd | 0.38x [0.35-0.56] | 68.31 | 27.19 | noisy |
| 36 | ojas-metal | linear_qkv_fwd | 0.43x [0.37-0.65] | 2.311 | 0.994 | noisy |
| 37 | ojas-metal | gate_bwd | 0.43x [0.29-0.60] | 2.737 | 1.279 | noisy |
| 38 | ojas-wgpu | muon_2048x768 | 0.43x [0.41-0.54] | 19.03 | 9.216 | noisy |
| 39 | ojas-metal | block_fwd_bwd | 0.43x [0.37-0.60] | 168.9 | 81.77 | noisy |
| 40 | ojas-wgpu | muon_768x2048 | 0.44x [0.43-0.57] | 17.96 | 9.100 | noisy |
| 41 | ojas-metal | linear_lmhead_fwd | 0.45x [0.42-0.55] | 121.3 | 55.87 | noisy |
| 42 | ojas-wgpu | muon_768x768 | 0.45x [0.40-0.68] | 9.077 | 4.567 | noisy |
| 43 | ojas-metal | muon_768x768 vs muon_768x768_bf16 | 0.46x [0.43-0.51] | 5.233 | 2.425 | noisy |
| 44 | ojas-metal | muon_768x2048 vs muon_768x2048_bf16 | 0.47x [0.45-0.65] | 8.673 | 5.420 | noisy |
| 45 | ojas-metal | mul_bwd | 0.48x [0.22-1.67] | 2.265 | 1.093 | noisy |
| 46 | ojas-metal | vres_bwd | 0.49x [0.37-0.66] | 2.404 | 1.170 | noisy |
| 47 | ojas-metal | muon_2048x768 vs muon_2048x768_bf16 | 0.50x [0.48-0.76] | 9.490 | 5.822 | noisy |
| 48 | ojas-wgpu | cross_entropy_bwd | 0.58x [0.54-0.87] | 50.32 | 27.19 | noisy |
| 49 | ojas-metal | sdpa_b4h8t2048d64_fwd | 0.59x [0.50-0.76] | 5.463 | 3.245 | noisy |
| 50 | ojas-wgpu | gate_bwd | 0.59x [0.40-0.81] | 1.872 | 1.279 | noisy |
| 51 | ojas-metal | sdpa_b4h12t1024d64_fwd | 0.65x [0.42-1.98] | 2.510 | 1.547 | noisy |
| 52 | ojas-wgpu | residual_add_fwd | 0.66x [0.58-7.17] | 0.453 | 0.299 | noisy |
| 53 | ojas-metal | linear_lmhead_bwd | 0.69x [0.66-0.83] | 159.1 | 113.4 | noisy |
| 54 | ojas-wgpu | rms_norm_fwd | 0.70x [0.47-1.08] | 0.380 | 0.271 | noisy |
| 55 | ojas-metal | sdpa_b2h8t1024d128_fwd | 0.70x [0.44-1.58] | 1.774 | 1.248 | noisy |
| 56 | ojas-wgpu | silu_fwd | 0.70x [0.67-1.01] | 0.580 | 0.411 | noisy |
| 57 | ojas-metal | linear_qkv_bwd | 0.70x [0.59-0.98] | 2.628 | 1.837 | noisy |
| 58 | ojas-wgpu | gate_fwd | 0.72x [0.62-1.61] | 0.811 | 0.580 | noisy |
| 59 | ojas-wgpu | adamw_full | 0.75x [0.56-1.06] | 63.29 | 49.88 | noisy |
| 60 | ojas-metal | linear_up_fwd | 0.76x [0.56-0.89] | 2.686 | 2.062 | noisy |
| 61 | ojas-wgpu | mul_fwd | 0.77x [0.59-2.47] | 0.749 | 0.536 | noisy |
| 62 | ojas-metal | linear_down_fwd | 0.78x [0.70-0.92] | 2.621 | 2.186 | noisy |
| 63 | ojas-wgpu | silu_bwd | 0.78x [0.65-1.28] | 0.790 | 0.615 | noisy |
| 64 | ojas-metal | muon_768x768 | 0.78x [0.68-1.26] | 5.233 | 4.567 | noisy |
| 65 | ojas-metal | rope_fwd | 0.81x [0.74-1.20] | 1.677 | 1.317 | noisy |
| 66 | ojas-metal | linear_down_bwd | 0.82x [0.68-0.96] | 5.128 | 4.217 | noisy |
| 67 | ojas-metal | muon_2048x768 | 0.88x [0.77-1.13] | 9.490 | 9.216 | noisy |
| 68 | ojas-metal | linear_up_bwd | 0.90x [0.85-1.08] | 4.792 | 4.394 | noisy |
| 69 | ojas-wgpu | permute_bthd_bhtd | 0.92x [0.77-1.19] | 0.552 | 0.480 | noisy |
| 70 | ojas-metal | sdpa_b2h8t1024d128_bwd | 0.92x [0.85-1.11] | 5.598 | 5.220 | noisy |
| 71 | ojas-metal | muon_768x2048 | 0.94x [0.78-1.17] | 8.673 | 9.100 | noisy |
| 72 | ojas-wgpu | mul_bwd | 0.95x [0.77-4.36] | 1.120 | 1.093 | noisy |
| 73 | ojas-metal | rms_norm_bwd | 0.97x [0.73-1.29] | 2.502 | 2.445 | noisy |
| 74 | ojas-metal | rope_bwd | 1.01x [0.99-1.79] | 1.694 | 1.736 | noisy |
| 75 | ojas-metal | cross_entropy_fwd | 1.03x [0.95-1.93] | 12.98 | 19.30 | noisy |
| 76 | ojas-wgpu | rms_qk_norm_bwd | 1.08x [0.67-1.28] | 4.152 | 5.085 | noisy |
| 77 | ojas-wgpu | vres_fwd | 1.20x [1.09-1.91] | 0.459 | 0.571 | noisy |
| 78 | ojas-wgpu | vres_bwd | 1.46x [1.16-2.10] | 0.605 | 1.170 | noisy |
| 79 | ojas-metal | sdpa_b4h12t1024d64_bwd | 1.50x [1.34-1.64] | 7.035 | 10.32 | noisy |
| 80 | ojas-metal | sdpa_b4h8t2048d64_bwd | 1.61x [1.40-1.77] | 16.51 | 25.95 | noisy |
| 81 | ojas-wgpu | cross_entropy_fwd | 2.61x [1.93-3.21] | 7.027 | 19.30 | noisy |
| 82 | ojas-wgpu | rms_norm_bwd | 3.53x [2.32-5.76] | 0.684 | 2.445 | noisy |
| 83 | ojas-wgpu | rope_fwd | 3.80x [3.51-4.78] | 0.348 | 1.317 | noisy |
| 84 | ojas-wgpu | rope_bwd | 5.23x [4.41-6.49] | 0.332 | 1.736 | noisy |
| 85 | ojas-metal | clip_grad_norm_full | 7.16x [4.57-9.36] | 18.72 | 135.5 | noisy |
| 86 | ojas-wgpu | clip_grad_norm_full | 7.95x [6.86-9.10] | 16.95 | 135.5 | noisy |

### Machine load around each runtime block

| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |
| --: | :-- | :-- | :-- | :-- | --: |
| 1 | ojas-metal | before | 16:48:54 | 12.77 14.60 22.55 | 93 |
| 1 | ojas-metal | after | 16:49:54 | 12.66 14.26 21.88 | 96 |
| 1 | ojas-wgpu | before | 16:49:54 | 12.66 14.26 21.88 | 93 |
| 1 | ojas-wgpu | after | 16:51:49 | 28.47 17.17 21.92 | 99 |
| 1 | torch-mps | before | 16:51:49 | 28.47 17.17 21.92 | 0 |
| 1 | torch-mps | after | 16:52:25 | 21.93 16.80 21.60 | 97 |
| 2 | torch-mps | before | 16:52:25 | 21.93 16.80 21.60 | 94 |
| 2 | torch-mps | after | 16:52:58 | 16.89 16.11 21.15 | 98 |
| 2 | ojas-wgpu | before | 16:52:58 | 16.89 16.11 21.15 | 84 |
| 2 | ojas-wgpu | after | 16:54:50 | 14.70 15.74 20.40 | 98 |
| 2 | ojas-metal | before | 16:54:50 | 14.70 15.74 20.40 | 94 |
| 2 | ojas-metal | after | 16:55:44 | 12.53 14.92 19.80 | 88 |
| 3 | ojas-metal | before | 16:55:44 | 12.53 14.92 19.80 | 0 |
| 3 | ojas-metal | after | 16:56:38 | 15.06 15.57 19.76 | 87 |
| 3 | ojas-wgpu | before | 16:56:38 | 15.06 15.57 19.76 | 0 |
| 3 | ojas-wgpu | after | 16:58:25 | 14.10 15.04 19.06 | 98 |
| 3 | torch-mps | before | 16:58:25 | 14.10 15.04 19.06 | 0 |
| 3 | torch-mps | after | 16:58:58 | 16.69 15.61 19.10 | 93 |
| 4 | torch-mps | before | 16:58:58 | 16.69 15.61 19.10 | 41 |
| 4 | torch-mps | after | 16:59:30 | 14.99 15.35 18.89 | 94 |
| 4 | ojas-wgpu | before | 16:59:30 | 14.99 15.35 18.89 | 50 |
| 4 | ojas-wgpu | after | 17:01:20 | 11.80 13.91 17.88 | 98 |
| 4 | ojas-metal | before | 17:01:20 | 11.80 13.91 17.88 | 0 |
| 4 | ojas-metal | after | 17:02:09 | 11.27 13.47 17.49 | 85 |
| 5 | ojas-metal | before | 17:02:09 | 11.27 13.47 17.49 | 0 |
| 5 | ojas-metal | after | 17:02:56 | 15.33 14.30 17.59 | 88 |
| 5 | ojas-wgpu | before | 17:02:56 | 15.33 14.30 17.59 | 0 |
| 5 | ojas-wgpu | after | 17:04:44 | 13.32 14.21 17.18 | 98 |
| 5 | torch-mps | before | 17:04:44 | 13.32 14.21 17.18 | 0 |
| 5 | torch-mps | after | 17:05:12 | 12.27 13.87 16.95 | 96 |
