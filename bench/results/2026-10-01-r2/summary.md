# Paired GPU-vs-torch summary: /Users/bharath/Code/research/ojas/bench/out/r2

5 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.

```
date: 2026-10-02T00:01:02Z
git: dab2a12fd2129565515954924e1a11737ca6ee47 (dirty files: 195)
uncommitted diff of the benchmarked crates (sha1): a73d2f0ef611
tessl: cf65d9d5d3deb97b0847e020a562ac8f15e6d5a9 (dirty files: 20)
rustc: rustc 1.98.0 (88d9e12ae 2026-08-18)
cargo: cargo 1.98.0 (797e8a9bc 2026-08-05)
os: macOS 27.0.1 (26A434)
cpu: Apple M5 Pro
memory_bytes: 68719476736
metal_bin: Oct  1 19:01:02 2026 3849904
wgpu_bin: Oct  1 19:01:02 2026 6309952
rounds: 5  iters: 20  warmup: 5  rows: all
```

### ojas-metal vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.161 / 1.136 | 0.114 / 0.200 | 0.18x [0.11-0.23] | 139% / 51% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_fwd | 1.059 / 1.834 | 0.795 / 0.971 | 0.53x [0.35-0.80] | 157% / 19% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_bwd | 1.869 / 2.418 | 1.597 / 1.907 | 0.86x [0.75-0.94] | 36% / 32% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_fwd | 2.409 / 2.906 | 2.025 / 2.220 | 0.84x [0.58-0.85] | 56% / 17% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_bwd | 4.121 / 4.821 | 3.891 / 4.242 | 0.88x [0.87-1.00] | 6% / 9% |  | 0.00e+00 (0.0e+00) |
| linear_down_fwd | 2.266 / 2.734 | 2.027 / 2.229 | 0.84x [0.73-2.08] | 17% / 180% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_bwd | 4.552 / 5.229 | 3.839 / 4.328 | 0.85x [0.72-2.00] | 16% / 150% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_fwd | 111.4 / 118.3 | 53.29 / 54.78 | 0.46x [0.44-0.49] | 9% / 4% |  | 0.00e+00 (0.0e+00) |
| linear_lmhead_bwd | 136.1 / 156.2 | 103.3 / 108.7 | 0.70x [0.61-0.84] | 15% / 25% | noisy - not quoted | 1.57e-04 (1.3e-05) |
| sdpa_b4h12t1024d64_fwd | 1.872 / 2.418 | 1.281 / 1.474 | 0.57x [0.36-0.84] | 112% / 21% | noisy - not quoted | 2.24e-07 (2.2e-07) |
| sdpa_b4h12t1024d64_bwd | 5.652 / 6.473 | 9.425 / 9.976 | 1.53x [1.40-1.82] | 12% / 18% | noisy - not quoted | 1.10e-06 (2.1e-06) |
| sdpa_b4h8t2048d64_fwd | 4.161 / 5.146 | 3.070 / 3.269 | 0.65x [0.57-0.72] | 26% / 5% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b4h8t2048d64_bwd | 13.76 / 14.75 | 24.01 / 25.34 | 1.73x [1.44-2.06] | 24% / 21% | noisy - not quoted | 1.31e-06 (2.5e-06) |
| sdpa_b2h8t1024d128_fwd | 1.199 / 1.641 | 1.017 / 1.134 | 0.74x [0.45-1.23] | 66% / 81% | noisy - not quoted | 2.09e-07 (2.1e-07) |
| sdpa_b2h8t1024d128_bwd | 4.331 / 4.902 | 4.396 / 5.059 | 1.09x [0.93-1.41] | 14% / 38% | noisy - not quoted | 1.10e-06 (2.6e-06) |
| rms_norm_fwd | 0.411 / 1.135 | 0.207 / 0.291 | 0.22x [0.19-0.32] | 153% / 80% | noisy - not quoted | 3.58e-07 (2.0e-07) |
| rms_norm_bwd | 0.665 / 1.706 | 1.760 / 2.178 | 1.75x [1.18-2.37] | 147% / 98% | noisy - not quoted | 3.81e-05 (2.8e-07) |
| rms_qk_norm_fwd | 0.577 / 2.135 | 0.332 / 0.420 | 0.21x [0.17-0.37] | 174% / 25% | noisy - not quoted | 3.58e-07 (1.7e-07) |
| rms_qk_norm_bwd | 1.374 / 2.579 | 4.076 / 4.675 | 1.81x [1.48-3.50] | 100% / 18% | noisy - not quoted | 3.66e-04 (1.0e-06) |
| rope_fwd | 0.397 / 0.897 | 1.201 / 1.395 | 1.63x [0.90-2.89] | 246% / 15% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| rope_bwd | 0.679 / 2.093 | 1.496 / 1.815 | 0.88x [0.85-1.69] | 73% / 43% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| silu_fwd | 0.880 / 2.001 | 0.413 / 0.559 | 0.26x [0.19-0.35] | 53% / 30% | noisy - not quoted | 2.38e-07 (6.1e-08) |
| silu_bwd | 1.277 / 2.292 | 0.527 / 0.776 | 0.37x [0.24-0.48] | 61% / 45% | noisy - not quoted | 1.19e-07 (1.1e-07) |
| mul_fwd | 1.247 / 1.825 | 0.523 / 0.678 | 0.37x [0.20-0.50] | 135% / 42% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| mul_bwd | 1.916 / 2.552 | 0.929 / 1.220 | 0.44x [0.40-0.49] | 27% / 15% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_fwd | 0.604 / 2.180 | 0.272 / 0.364 | 0.20x [0.14-0.29] | 95% / 36% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_bwd | 0.631 / 2.286 | 0.009 / 0.016 | 0.01x [0.00-0.02] | 110% / 536% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| gate_fwd | 0.723 / 1.675 | 0.555 / 0.820 | 0.48x [0.20-0.86] | 253% / 43% | noisy - not quoted | 8.94e-08 (1.0e-07) |
| gate_bwd | 2.094 / 3.208 | 0.931 / 1.406 | 0.46x [0.36-0.50] | 55% / 27% | noisy - not quoted | 2.25e-04 (2.3e-06) |
| vres_fwd | 0.519 / 1.475 | 0.487 / 0.638 | 0.41x [0.26-0.90] | 211% / 21% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| vres_bwd | 0.914 / 1.739 | 0.707 / 1.092 | 0.55x [0.35-0.82] | 146% / 25% | noisy - not quoted | 3.05e-05 (2.3e-07) |
| permute_bthd_bhtd | 0.416 / 1.591 | 0.404 / 0.494 | 0.31x [0.24-0.78] | 89% / 126% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_fwd | 3.167 / 3.851 | 13.90 / 16.26 | 3.93x [3.57-4.63] | 29% / 13% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_bwd | 32.31 / 38.40 | 26.24 / 28.87 | 0.72x [0.61-0.77] | 16% / 9% | noisy - not quoted | 6.39e-14 (2.6e-10) |
| linear_ce_c1024x8192 | 246.1 / 266.3 | 208.8 / 221.0 | 0.88x [0.71-0.91] | 21% / 13% | noisy - not quoted | 9.54e-07 (6.7e-06) |
| linear_ce_c4096x50304 | 377.8 / 423.3 | 207.1 / 222.3 | 0.55x [0.50-0.57] | 11% / 10% | noisy - not quoted | 9.54e-07 (2.3e-05) |
| clip_grad_norm_full | 16.19 / 17.59 | 132.8 / 134.4 | 7.67x [7.02-7.96] | 14% / 1% | noisy - not quoted | 3.05e-05 (3.0e-07) |
| adamw_full | 200.8 / 244.5 | 35.04 / 39.61 | 0.16x [0.14-0.17] | 8% / 22% | noisy - not quoted | 3.73e-09 (1.2e-07) |
| muon_768x768 | 3.057 / 3.364 | 3.564 / 4.124 | 1.23x [1.15-1.30] | 13% / 5% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| muon_768x768 vs muon_768x768_bf16 | 3.057 / 3.364 | 1.733 / 2.176 | 0.64x [0.58-0.71] | 13% / 15% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| muon_2048x768 | 6.284 / 6.899 | 7.746 / 8.258 | 1.20x [1.18-1.22] | 4% / 3% |  | 1.68e-08 (5.0e-07) |
| muon_2048x768 vs muon_2048x768_bf16 | 6.284 / 6.899 | 4.190 / 4.727 | 0.68x [0.66-0.73] | 4% / 8% |  | 1.68e-08 (5.0e-07) |
| muon_768x2048 | 6.201 / 6.914 | 7.292 / 8.224 | 1.21x [1.16-1.54] | 6% / 32% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| muon_768x2048 vs muon_768x2048_bf16 | 6.201 / 6.914 | 3.638 / 4.178 | 0.60x [0.58-0.71] | 6% / 28% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| block_fwd | 35.17 / 49.22 | 14.49 / 15.31 | 0.32x [0.27-0.33] | 21% / 3% | noisy - not quoted | 2.38e-07 (1.9e-07) |
| block_fwd_bwd | 99.00 / 122.3 | 68.74 / 71.79 | 0.59x [0.56-0.64] | 10% / 5% |  | 9.30e-06 (2.5e-06) |
| decode_attn_kv1024 | 0.200 / 0.832 | 0.112 / 0.212 | 0.37x [0.22-1.16] | 232% / 1214% | noisy - not quoted | 1.86e-08 (3.2e-07) |
| accumulate_grad_50304x768 | 9.179 / 9.624 | 1.932 / 2.377 | 0.25x [0.21-0.53] | 4% / 144% | noisy - not quoted | 0.00e+00 (0.0e+00) |

### ojas-wgpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.207 / 0.359 | 0.114 / 0.200 | 0.52x [0.06-0.77] | 831% / 51% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_fwd | 1.725 / 2.167 | 0.795 / 0.971 | 0.47x [0.33-0.48] | 73% / 19% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_bwd | 3.188 / 3.453 | 1.597 / 1.907 | 0.55x [0.53-0.67] | 7% / 32% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_fwd | 4.151 / 4.652 | 2.025 / 2.220 | 0.47x [0.46-0.58] | 13% / 17% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_bwd | 8.094 / 8.729 | 3.891 / 4.242 | 0.49x [0.48-0.56] | 7% / 9% |  | 0.00e+00 (0.0e+00) |
| linear_down_fwd | 4.231 / 4.601 | 2.027 / 2.229 | 0.49x [0.46-1.30] | 6% / 180% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_bwd | 7.977 / 8.661 | 3.839 / 4.328 | 0.51x [0.48-1.21] | 5% / 150% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_fwd | 128.1 / 136.4 | 53.29 / 54.78 | 0.40x [0.39-0.41] | 3% / 4% |  | 0.00e+00 (0.0e+00) |
| linear_lmhead_bwd | 192.0 / 199.9 | 103.3 / 108.7 | 0.54x [0.50-0.63] | 5% / 25% | noisy - not quoted | 1.57e-04 (1.3e-05) |
| sdpa_b4h12t1024d64_fwd | 10.74 / 11.83 | 1.281 / 1.474 | 0.12x [0.12-0.14] | 4% / 21% | noisy - not quoted | 2.38e-07 (2.4e-07) |
| sdpa_b4h12t1024d64_bwd | 51.45 / 53.34 | 9.425 / 9.976 | 0.19x [0.16-0.22] | 16% / 18% | noisy - not quoted | 1.10e-06 (2.1e-06) |
| sdpa_b4h8t2048d64_fwd | 29.00 / 30.40 | 3.070 / 3.269 | 0.11x [0.09-0.11] | 15% / 5% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b4h8t2048d64_bwd | 137.5 / 144.3 | 24.01 / 25.34 | 0.18x [0.17-0.22] | 7% / 21% | noisy - not quoted | 1.37e-06 (2.6e-06) |
| sdpa_b2h8t1024d128_fwd | 10.98 / 11.85 | 1.017 / 1.134 | 0.10x [0.09-0.17] | 8% / 81% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b2h8t1024d128_bwd | 50.52 / 53.07 | 4.396 / 5.059 | 0.10x [0.09-0.13] | 3% / 38% | noisy - not quoted | 1.31e-06 (2.3e-06) |
| rms_norm_fwd | 0.296 / 0.430 | 0.207 / 0.291 | 0.67x [0.43-0.82] | 80% / 80% | noisy - not quoted | 2.38e-07 (1.3e-07) |
| rms_norm_bwd | 0.495 / 0.704 | 1.760 / 2.178 | 3.10x [2.19-6.04] | 40% / 98% | noisy - not quoted | 3.34e-05 (2.6e-07) |
| rms_qk_norm_fwd | 1.802 / 2.102 | 0.332 / 0.420 | 0.20x [0.19-0.24] | 13% / 25% | noisy - not quoted | 2.38e-07 (1.2e-07) |
| rms_qk_norm_bwd | 3.794 / 4.303 | 4.076 / 4.675 | 1.11x [1.06-1.27] | 4% / 18% | noisy - not quoted | 4.58e-04 (1.3e-06) |
| rope_fwd | 0.252 / 0.367 | 1.201 / 1.395 | 3.70x [2.22-5.14] | 106% / 15% | noisy - not quoted | 1.19e-07 (6.2e-08) |
| rope_bwd | 0.278 / 0.393 | 1.496 / 1.815 | 4.86x [2.66-6.15] | 86% / 43% | noisy - not quoted | 1.19e-07 (6.1e-08) |
| silu_fwd | 0.479 / 0.769 | 0.413 / 0.559 | 0.73x [0.50-1.18] | 83% / 30% | noisy - not quoted | 7.15e-07 (1.8e-07) |
| silu_bwd | 0.639 / 0.892 | 0.527 / 0.776 | 0.85x [0.55-1.12] | 46% / 45% | noisy - not quoted | 4.77e-07 (4.3e-07) |
| mul_fwd | 0.593 / 0.804 | 0.523 / 0.678 | 0.85x [0.77-1.01] | 40% / 42% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| mul_bwd | 0.876 / 1.162 | 0.929 / 1.220 | 1.01x [0.97-1.14] | 25% / 15% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_fwd | 0.365 / 0.552 | 0.272 / 0.364 | 0.62x [0.58-0.81] | 51% / 36% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_bwd | 0.681 / 1.062 | 0.009 / 0.016 | 0.01x [0.01-0.07] | 30% / 536% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| gate_fwd | 0.750 / 1.186 | 0.555 / 0.820 | 0.65x [0.58-0.98] | 30% / 43% | noisy - not quoted | 1.19e-07 (1.4e-07) |
| gate_bwd | 1.601 / 2.267 | 0.931 / 1.406 | 0.64x [0.52-0.69] | 52% / 27% | noisy - not quoted | 2.21e-04 (2.3e-06) |
| vres_fwd | 0.327 / 0.534 | 0.487 / 0.638 | 1.11x [0.93-1.82] | 88% / 21% | noisy - not quoted | 5.96e-08 (6.0e-08) |
| vres_bwd | 0.539 / 0.722 | 0.707 / 1.092 | 1.36x [1.12-1.69] | 59% / 25% | noisy - not quoted | 3.05e-05 (2.3e-07) |
| permute_bthd_bhtd | 0.581 / 0.772 | 0.404 / 0.494 | 0.65x [0.56-1.37] | 63% / 126% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_fwd | 6.336 / 7.374 | 13.90 / 16.26 | 2.21x [1.87-2.49] | 19% / 13% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_bwd | 38.55 / 47.74 | 26.24 / 28.87 | 0.60x [0.55-0.71] | 21% / 9% | noisy - not quoted | 6.39e-14 (2.6e-10) |
| linear_ce_c1024x8192 | 434.7 / 474.2 | 208.8 / 221.0 | 0.45x [0.42-0.52] | 12% / 13% | noisy - not quoted | 9.54e-07 (2.3e-05) |
| linear_ce_c4096x50304 | 359.4 / 385.1 | 207.1 / 222.3 | 0.58x [0.56-0.64] | 5% / 10% | noisy - not quoted | 1.82e-10 (2.3e-05) |
| clip_grad_norm_full | 16.12 / 17.18 | 132.8 / 134.4 | 7.81x [6.99-7.86] | 12% / 1% | noisy - not quoted | 3.81e-05 (3.8e-07) |
| adamw_full | 54.99 / 62.46 | 35.04 / 39.61 | 0.63x [0.61-0.64] | 20% / 22% | noisy - not quoted | 3.73e-09 (1.2e-07) |
| muon_768x768 | 5.448 / 6.207 | 3.564 / 4.124 | 0.65x [0.62-0.70] | 19% / 5% | noisy - not quoted | 1.49e-08 (4.5e-07) |
| muon_768x768 vs muon_768x768_bf16 | 5.448 / 6.207 | 1.733 / 2.176 | 0.35x [0.32-0.40] | 19% / 15% | noisy - not quoted | 1.49e-08 (4.5e-07) |
| muon_2048x768 | 12.57 / 13.83 | 7.746 / 8.258 | 0.60x [0.55-0.63] | 13% / 3% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_2048x768 vs muon_2048x768_bf16 | 12.57 / 13.83 | 4.190 / 4.727 | 0.33x [0.32-0.37] | 13% / 8% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_768x2048 | 11.97 / 12.77 | 7.292 / 8.224 | 0.64x [0.64-0.77] | 11% / 32% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| muon_768x2048 vs muon_768x2048_bf16 | 11.97 / 12.77 | 3.638 / 4.178 | 0.33x [0.30-0.40] | 11% / 28% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| block_fwd | 39.94 / 43.00 | 14.49 / 15.31 | 0.35x [0.32-0.36] | 12% / 3% | noisy - not quoted | 2.38e-07 (1.9e-07) |
| block_fwd_bwd | 148.7 / 158.1 | 68.74 / 71.79 | 0.47x [0.43-0.48] | 9% / 5% |  | 8.58e-06 (2.3e-06) |
| decode_attn_kv1024 | 0.245 / 0.414 | 0.112 / 0.212 | 0.68x [0.48-3.31] | 119% / 1214% | noisy - not quoted | 1.49e-08 (2.5e-07) |
| accumulate_grad_50304x768 | 2.043 / 2.412 | 1.932 / 2.377 | 0.91x [0.86-1.77] | 28% / 144% | noisy - not quoted | 0.00e+00 (0.0e+00) |

### Ranked: where ojas is slowest relative to torch (lowest ratio first)

| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |
| --: | :-- | :-- | :-- | --: | --: | :-- |
| 1 | ojas-metal | residual_add_bwd | 0.01x [0.00-0.02] | 2.286 | 0.016 | noisy |
| 2 | ojas-wgpu | residual_add_bwd | 0.01x [0.01-0.07] | 1.062 | 0.016 | noisy |
| 3 | ojas-wgpu | sdpa_b2h8t1024d128_bwd | 0.10x [0.09-0.13] | 53.07 | 5.059 | noisy |
| 4 | ojas-wgpu | sdpa_b2h8t1024d128_fwd | 0.10x [0.09-0.17] | 11.85 | 1.134 | noisy |
| 5 | ojas-wgpu | sdpa_b4h8t2048d64_fwd | 0.11x [0.09-0.11] | 30.40 | 3.269 | noisy |
| 6 | ojas-wgpu | sdpa_b4h12t1024d64_fwd | 0.12x [0.12-0.14] | 11.83 | 1.474 | noisy |
| 7 | ojas-metal | adamw_full | 0.16x [0.14-0.17] | 244.5 | 39.61 | noisy |
| 8 | ojas-wgpu | sdpa_b4h8t2048d64_bwd | 0.18x [0.17-0.22] | 144.3 | 25.34 | noisy |
| 9 | ojas-wgpu | sdpa_b4h12t1024d64_bwd | 0.19x [0.16-0.22] | 53.34 | 9.976 | noisy |
| 10 | ojas-wgpu | rms_qk_norm_fwd | 0.20x [0.19-0.24] | 2.102 | 0.420 | noisy |
| 11 | ojas-metal | residual_add_fwd | 0.20x [0.14-0.29] | 2.180 | 0.364 | noisy |
| 12 | ojas-metal | rms_qk_norm_fwd | 0.21x [0.17-0.37] | 2.135 | 0.420 | noisy |
| 13 | ojas-metal | rms_norm_fwd | 0.22x [0.19-0.32] | 1.135 | 0.291 | noisy |
| 14 | ojas-metal | accumulate_grad_50304x768 | 0.25x [0.21-0.53] | 9.624 | 2.377 | noisy |
| 15 | ojas-metal | silu_fwd | 0.26x [0.19-0.35] | 2.001 | 0.559 | noisy |
| 16 | ojas-metal | permute_bthd_bhtd | 0.31x [0.24-0.78] | 1.591 | 0.494 | noisy |
| 17 | ojas-metal | block_fwd | 0.32x [0.27-0.33] | 49.22 | 15.31 | noisy |
| 18 | ojas-wgpu | muon_768x2048 vs muon_768x2048_bf16 | 0.33x [0.30-0.40] | 12.77 | 4.178 | noisy |
| 19 | ojas-wgpu | muon_2048x768 vs muon_2048x768_bf16 | 0.33x [0.32-0.37] | 13.83 | 4.727 | noisy |
| 20 | ojas-wgpu | block_fwd | 0.35x [0.32-0.36] | 43.00 | 15.31 | noisy |
| 21 | ojas-wgpu | muon_768x768 vs muon_768x768_bf16 | 0.35x [0.32-0.40] | 6.207 | 2.176 | noisy |
| 22 | ojas-metal | silu_bwd | 0.37x [0.24-0.48] | 2.292 | 0.776 | noisy |
| 23 | ojas-metal | decode_attn_kv1024 | 0.37x [0.22-1.16] | 0.832 | 0.212 | noisy |
| 24 | ojas-metal | mul_fwd | 0.37x [0.20-0.50] | 1.825 | 0.678 | noisy |
| 25 | ojas-wgpu | linear_lmhead_fwd | 0.40x [0.39-0.41] | 136.4 | 54.78 |  |
| 26 | ojas-metal | vres_fwd | 0.41x [0.26-0.90] | 1.475 | 0.638 | noisy |
| 27 | ojas-metal | mul_bwd | 0.44x [0.40-0.49] | 2.552 | 1.220 | noisy |
| 28 | ojas-wgpu | linear_ce_c1024x8192 | 0.45x [0.42-0.52] | 474.2 | 221.0 | noisy |
| 29 | ojas-metal | gate_bwd | 0.46x [0.36-0.50] | 3.208 | 1.406 | noisy |
| 30 | ojas-metal | linear_lmhead_fwd | 0.46x [0.44-0.49] | 118.3 | 54.78 |  |
| 31 | ojas-wgpu | linear_qkv_fwd | 0.47x [0.33-0.48] | 2.167 | 0.971 | noisy |
| 32 | ojas-wgpu | block_fwd_bwd | 0.47x [0.43-0.48] | 158.1 | 71.79 |  |
| 33 | ojas-wgpu | linear_up_fwd | 0.47x [0.46-0.58] | 4.652 | 2.220 | noisy |
| 34 | ojas-metal | gate_fwd | 0.48x [0.20-0.86] | 1.675 | 0.820 | noisy |
| 35 | ojas-wgpu | linear_up_bwd | 0.49x [0.48-0.56] | 8.729 | 4.242 |  |
| 36 | ojas-wgpu | linear_down_fwd | 0.49x [0.46-1.30] | 4.601 | 2.229 | noisy |
| 37 | ojas-wgpu | linear_down_bwd | 0.51x [0.48-1.21] | 8.661 | 4.328 | noisy |
| 38 | ojas-metal | linear_qkv_fwd | 0.53x [0.35-0.80] | 1.834 | 0.971 | noisy |
| 39 | ojas-wgpu | linear_lmhead_bwd | 0.54x [0.50-0.63] | 199.9 | 108.7 | noisy |
| 40 | ojas-metal | vres_bwd | 0.55x [0.35-0.82] | 1.739 | 1.092 | noisy |
| 41 | ojas-metal | linear_ce_c4096x50304 | 0.55x [0.50-0.57] | 423.3 | 222.3 | noisy |
| 42 | ojas-wgpu | linear_qkv_bwd | 0.55x [0.53-0.67] | 3.453 | 1.907 | noisy |
| 43 | ojas-metal | sdpa_b4h12t1024d64_fwd | 0.57x [0.36-0.84] | 2.418 | 1.474 | noisy |
| 44 | ojas-wgpu | linear_ce_c4096x50304 | 0.58x [0.56-0.64] | 385.1 | 222.3 | noisy |
| 45 | ojas-metal | block_fwd_bwd | 0.59x [0.56-0.64] | 122.3 | 71.79 |  |
| 46 | ojas-wgpu | muon_2048x768 | 0.60x [0.55-0.63] | 13.83 | 8.258 | noisy |
| 47 | ojas-wgpu | cross_entropy_bwd | 0.60x [0.55-0.71] | 47.74 | 28.87 | noisy |
| 48 | ojas-metal | muon_768x2048 vs muon_768x2048_bf16 | 0.60x [0.58-0.71] | 6.914 | 4.178 | noisy |
| 49 | ojas-wgpu | residual_add_fwd | 0.62x [0.58-0.81] | 0.552 | 0.364 | noisy |
| 50 | ojas-wgpu | adamw_full | 0.63x [0.61-0.64] | 62.46 | 39.61 | noisy |
| 51 | ojas-metal | muon_768x768 vs muon_768x768_bf16 | 0.64x [0.58-0.71] | 3.364 | 2.176 | noisy |
| 52 | ojas-wgpu | muon_768x2048 | 0.64x [0.64-0.77] | 12.77 | 8.224 | noisy |
| 53 | ojas-wgpu | gate_bwd | 0.64x [0.52-0.69] | 2.267 | 1.406 | noisy |
| 54 | ojas-wgpu | permute_bthd_bhtd | 0.65x [0.56-1.37] | 0.772 | 0.494 | noisy |
| 55 | ojas-metal | sdpa_b4h8t2048d64_fwd | 0.65x [0.57-0.72] | 5.146 | 3.269 | noisy |
| 56 | ojas-wgpu | muon_768x768 | 0.65x [0.62-0.70] | 6.207 | 4.124 | noisy |
| 57 | ojas-wgpu | gate_fwd | 0.65x [0.58-0.98] | 1.186 | 0.820 | noisy |
| 58 | ojas-wgpu | rms_norm_fwd | 0.67x [0.43-0.82] | 0.430 | 0.291 | noisy |
| 59 | ojas-wgpu | decode_attn_kv1024 | 0.68x [0.48-3.31] | 0.414 | 0.212 | noisy |
| 60 | ojas-metal | muon_2048x768 vs muon_2048x768_bf16 | 0.68x [0.66-0.73] | 6.899 | 4.727 |  |
| 61 | ojas-metal | linear_lmhead_bwd | 0.70x [0.61-0.84] | 156.2 | 108.7 | noisy |
| 62 | ojas-metal | cross_entropy_bwd | 0.72x [0.61-0.77] | 38.40 | 28.87 | noisy |
| 63 | ojas-wgpu | silu_fwd | 0.73x [0.50-1.18] | 0.769 | 0.559 | noisy |
| 64 | ojas-metal | sdpa_b2h8t1024d128_fwd | 0.74x [0.45-1.23] | 1.641 | 1.134 | noisy |
| 65 | ojas-metal | linear_down_fwd | 0.84x [0.73-2.08] | 2.734 | 2.229 | noisy |
| 66 | ojas-metal | linear_up_fwd | 0.84x [0.58-0.85] | 2.906 | 2.220 | noisy |
| 67 | ojas-wgpu | mul_fwd | 0.85x [0.77-1.01] | 0.804 | 0.678 | noisy |
| 68 | ojas-wgpu | silu_bwd | 0.85x [0.55-1.12] | 0.892 | 0.776 | noisy |
| 69 | ojas-metal | linear_down_bwd | 0.85x [0.72-2.00] | 5.229 | 4.328 | noisy |
| 70 | ojas-metal | linear_qkv_bwd | 0.86x [0.75-0.94] | 2.418 | 1.907 | noisy |
| 71 | ojas-metal | rope_bwd | 0.88x [0.85-1.69] | 2.093 | 1.815 | noisy |
| 72 | ojas-metal | linear_ce_c1024x8192 | 0.88x [0.71-0.91] | 266.3 | 221.0 | noisy |
| 73 | ojas-metal | linear_up_bwd | 0.88x [0.87-1.00] | 4.821 | 4.242 |  |
| 74 | ojas-wgpu | accumulate_grad_50304x768 | 0.91x [0.86-1.77] | 2.412 | 2.377 | noisy |
| 75 | ojas-wgpu | mul_bwd | 1.01x [0.97-1.14] | 1.162 | 1.220 | noisy |
| 76 | ojas-metal | sdpa_b2h8t1024d128_bwd | 1.09x [0.93-1.41] | 4.902 | 5.059 | noisy |
| 77 | ojas-wgpu | vres_fwd | 1.11x [0.93-1.82] | 0.534 | 0.638 | noisy |
| 78 | ojas-wgpu | rms_qk_norm_bwd | 1.11x [1.06-1.27] | 4.303 | 4.675 | noisy |
| 79 | ojas-metal | muon_2048x768 | 1.20x [1.18-1.22] | 6.899 | 8.258 |  |
| 80 | ojas-metal | muon_768x2048 | 1.21x [1.16-1.54] | 6.914 | 8.224 | noisy |
| 81 | ojas-metal | muon_768x768 | 1.23x [1.15-1.30] | 3.364 | 4.124 | noisy |
| 82 | ojas-wgpu | vres_bwd | 1.36x [1.12-1.69] | 0.722 | 1.092 | noisy |
| 83 | ojas-metal | sdpa_b4h12t1024d64_bwd | 1.53x [1.40-1.82] | 6.473 | 9.976 | noisy |
| 84 | ojas-metal | rope_fwd | 1.63x [0.90-2.89] | 0.897 | 1.395 | noisy |
| 85 | ojas-metal | sdpa_b4h8t2048d64_bwd | 1.73x [1.44-2.06] | 14.75 | 25.34 | noisy |
| 86 | ojas-metal | rms_norm_bwd | 1.75x [1.18-2.37] | 1.706 | 2.178 | noisy |
| 87 | ojas-metal | rms_qk_norm_bwd | 1.81x [1.48-3.50] | 2.579 | 4.675 | noisy |
| 88 | ojas-wgpu | cross_entropy_fwd | 2.21x [1.87-2.49] | 7.374 | 16.26 | noisy |
| 89 | ojas-wgpu | rms_norm_bwd | 3.10x [2.19-6.04] | 0.704 | 2.178 | noisy |
| 90 | ojas-wgpu | rope_fwd | 3.70x [2.22-5.14] | 0.367 | 1.395 | noisy |
| 91 | ojas-metal | cross_entropy_fwd | 3.93x [3.57-4.63] | 3.851 | 16.26 | noisy |
| 92 | ojas-wgpu | rope_bwd | 4.86x [2.66-6.15] | 0.393 | 1.815 | noisy |
| 93 | ojas-metal | clip_grad_norm_full | 7.67x [7.02-7.96] | 17.59 | 134.4 | noisy |
| 94 | ojas-wgpu | clip_grad_norm_full | 7.81x [6.99-7.86] | 17.18 | 134.4 | noisy |

### Machine load around each runtime block

| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |
| --: | :-- | :-- | :-- | :-- | --: |
| 1 | ojas-metal | before | 19:01:20 | 13.31 14.37 15.88 | 64 |
| 1 | ojas-metal | after | 19:02:06 | 13.09 14.14 15.71 | 0 |
| 1 | ojas-wgpu | before | 19:02:06 | 13.09 14.14 15.71 | 0 |
| 1 | ojas-wgpu | after | 19:03:04 | 22.99 16.40 16.43 | 0 |
| 1 | torch-mps | before | 19:03:04 | 22.99 16.40 16.43 | 0 |
| 1 | torch-mps | after | 19:03:46 | 18.79 16.27 16.39 | 35 |
| 2 | torch-mps | before | 19:03:46 | 18.79 16.27 16.39 | 37 |
| 2 | torch-mps | after | 19:04:26 | 20.94 17.14 16.70 | 39 |
| 2 | ojas-wgpu | before | 19:04:26 | 20.94 17.14 16.70 | 43 |
| 2 | ojas-wgpu | after | 19:05:23 | 19.57 17.25 16.76 | 0 |
| 2 | ojas-metal | before | 19:05:23 | 19.57 17.25 16.76 | 0 |
| 2 | ojas-metal | after | 19:06:12 | 22.20 18.24 17.15 | 0 |
| 3 | ojas-metal | before | 19:06:12 | 22.20 18.24 17.15 | 0 |
| 3 | ojas-metal | after | 19:06:59 | 19.08 17.92 17.08 | 0 |
| 3 | ojas-wgpu | before | 19:06:59 | 19.08 17.92 17.08 | 0 |
| 3 | ojas-wgpu | after | 19:07:56 | 16.15 17.39 16.94 | 0 |
| 3 | torch-mps | before | 19:07:56 | 16.15 17.39 16.94 | 0 |
| 3 | torch-mps | after | 19:08:38 | 15.83 17.19 16.89 | 33 |
| 4 | torch-mps | before | 19:08:39 | 15.83 17.19 16.89 | 45 |
| 4 | torch-mps | after | 19:09:18 | 14.60 16.77 16.76 | 26 |
| 4 | ojas-wgpu | before | 19:09:18 | 14.60 16.77 16.76 | 53 |
| 4 | ojas-wgpu | after | 19:10:15 | 17.52 17.26 16.94 | 0 |
| 4 | ojas-metal | before | 19:10:15 | 17.52 17.26 16.94 | 0 |
| 4 | ojas-metal | after | 19:11:04 | 16.97 17.16 16.92 | 0 |
| 5 | ojas-metal | before | 19:11:04 | 16.97 17.16 16.92 | 0 |
| 5 | ojas-metal | after | 19:11:51 | 16.83 17.00 16.87 | 0 |
| 5 | ojas-wgpu | before | 19:11:51 | 16.83 17.00 16.87 | 0 |
| 5 | ojas-wgpu | after | 19:12:48 | 18.16 17.24 16.96 | 0 |
| 5 | torch-mps | before | 19:12:48 | 18.16 17.24 16.96 | 0 |
| 5 | torch-mps | after | 19:13:30 | 20.52 18.08 17.28 | 37 |

### Load recorded before and after every row

| round | lane | samples | 1-min load min-max | GPU util % min / median / max |
| --: | :-- | --: | :-- | :-- |
| 1 | ojas-metal | 90 | 12.4-13.3 | 0 / 36 / 99 |
| 1 | ojas-wgpu | 90 | 13.1-23.0 | 0 / 30 / 99 |
| 1 | torch-mps | 96 | 18.8-22.6 | 0 / 36 / 99 |
| 2 | ojas-metal | 90 | 19.4-22.6 | 0 / 36 / 99 |
| 2 | ojas-wgpu | 90 | 17.7-20.9 | 0 / 32 / 99 |
| 2 | torch-mps | 96 | 18.8-20.9 | 0 / 30 / 99 |
| 3 | ojas-metal | 90 | 18.3-22.2 | 0 / 0 / 99 |
| 3 | ojas-wgpu | 90 | 16.1-19.6 | 0 / 30 / 99 |
| 3 | torch-mps | 96 | 15.8-17.1 | 0 / 32 / 99 |
| 4 | ojas-metal | 90 | 16.9-17.6 | 0 / 32 / 99 |
| 4 | ojas-wgpu | 90 | 14.2-19.7 | 0 / 15 / 99 |
| 4 | torch-mps | 96 | 14.6-16.7 | 0 / 30 / 99 |
| 5 | ojas-metal | 90 | 15.5-17.2 | 0 / 34 / 99 |
| 5 | ojas-wgpu | 90 | 15.7-18.2 | 0 / 32 / 99 |
| 5 | torch-mps | 96 | 17.7-22.4 | 0 / 26 / 99 |
