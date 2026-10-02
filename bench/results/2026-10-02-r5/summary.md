# Paired GPU-vs-torch summary: /Users/bharath/Code/research/ojas/bench/results/2026-10-02-r5

5 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.

```
date: 2026-10-02T05:54:35Z
git: dab2a12fd2129565515954924e1a11737ca6ee47 (dirty files: 227)
uncommitted diff of the benchmarked crates (sha1): de67949a36ff
tessl: cf65d9d5d3deb97b0847e020a562ac8f15e6d5a9 (dirty files: 23)
rustc: rustc 1.98.0 (88d9e12ae 2026-08-18)
cargo: cargo 1.98.0 (797e8a9bc 2026-08-05)
os: macOS 27.0.1 (26A434)
cpu: Apple M5 Pro
memory_bytes: 68719476736
metal_bin: Oct  2 00:54:35 2026 3869328
wgpu_bin: Oct  2 00:54:22 2026 6376352
rounds: 5  iters: 20  warmup: 5  rows: all
```

### ojas-metal vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.097 / 0.196 | 0.068 / 0.125 | 0.49x [0.07-0.96] | 1367% / 111% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_fwd | 0.919 / 1.072 | 0.786 / 0.956 | 0.90x [0.79-1.07] | 27% / 23% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_bwd | 1.754 / 1.901 | 1.691 / 2.141 | 1.03x [0.69-1.63] | 75% / 54% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_fwd | 2.324 / 2.536 | 1.920 / 2.727 | 0.96x [0.55-1.13] | 115% / 37% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_bwd | 4.179 / 4.901 | 3.901 / 5.033 | 0.86x [0.75-1.06] | 47% / 23% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_fwd | 2.139 / 2.310 | 1.963 / 2.561 | 1.08x [0.94-1.28] | 22% / 32% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_bwd | 4.849 / 5.602 | 3.814 / 5.206 | 0.74x [0.68-1.03] | 42% / 35% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_fwd | 114.0 / 119.1 | 54.10 / 69.04 | 0.48x [0.46-0.59] | 25% / 29% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_bwd | 151.5 / 170.7 | 106.7 / 129.8 | 0.72x [0.66-0.77] | 7% / 19% | noisy - not quoted | 1.57e-04 (1.3e-05) |
| sdpa_b4h12t1024d64_fwd | 1.748 / 1.929 | 1.241 / 1.384 | 0.72x [0.47-0.74] | 54% / 10% | noisy - not quoted | 2.24e-07 (2.2e-07) |
| sdpa_b4h12t1024d64_bwd | 5.453 / 7.428 | 9.558 / 13.91 | 1.71x [1.39-2.25] | 35% / 43% | noisy - not quoted | 1.10e-06 (2.1e-06) |
| sdpa_b4h8t2048d64_fwd | 4.114 / 5.125 | 3.032 / 3.975 | 0.65x [0.61-0.84] | 52% / 27% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b4h8t2048d64_bwd | 13.96 / 17.38 | 25.00 / 32.46 | 1.79x [1.54-2.17] | 25% / 31% | noisy - not quoted | 1.31e-06 (2.5e-06) |
| sdpa_b2h8t1024d128_fwd | 0.985 / 1.135 | 1.016 / 1.161 | 1.01x [0.87-1.14] | 23% / 17% | noisy - not quoted | 2.09e-07 (2.1e-07) |
| sdpa_b2h8t1024d128_bwd | 4.233 / 5.198 | 4.356 / 6.752 | 1.03x [1.00-1.41] | 42% / 38% | noisy - not quoted | 1.10e-06 (2.6e-06) |
| rms_norm_fwd | 0.241 / 0.340 | 0.164 / 0.212 | 0.62x [0.02-0.66] | 3330% / 30% | noisy - not quoted | 3.58e-07 (2.0e-07) |
| rms_norm_bwd | 0.507 / 0.643 | 1.741 / 2.058 | 3.32x [2.96-3.91] | 18% / 17% | noisy - not quoted | 3.81e-05 (2.8e-07) |
| rms_qk_norm_fwd | 0.410 / 0.506 | 0.280 / 0.361 | 0.72x [0.61-0.79] | 13% / 15% | noisy - not quoted | 3.58e-07 (1.7e-07) |
| rms_qk_norm_bwd | 1.107 / 1.318 | 4.239 / 6.698 | 5.11x [3.47-5.64] | 15% / 41% | noisy - not quoted | 3.66e-04 (1.0e-06) |
| rope_fwd | 0.237 / 0.348 | 1.128 / 1.257 | 3.61x [3.44-4.12] | 17% / 12% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| rope_bwd | 0.248 / 0.341 | 1.422 / 1.658 | 4.90x [4.24-5.34] | 14% / 17% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| silu_fwd | 0.370 / 0.424 | 0.336 / 0.465 | 1.11x [1.01-1.21] | 7% / 26% | noisy - not quoted | 2.38e-07 (6.1e-08) |
| silu_bwd | 0.502 / 0.576 | 0.490 / 0.655 | 1.14x [1.00-1.16] | 11% / 29% | noisy - not quoted | 1.19e-07 (1.1e-07) |
| mul_fwd | 0.491 / 0.602 | 0.468 / 0.603 | 1.00x [0.88-9.03] | 10% / 999% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| mul_bwd | 0.756 / 0.864 | 0.865 / 1.007 | 1.14x [0.60-1.30] | 168% / 42% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_fwd | 0.253 / 0.331 | 0.231 / 0.341 | 1.03x [0.84-1.15] | 13% / 24% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_bwd | 0.364 / 0.457 | 0.010 / 0.016 | 0.03x [0.03-0.04] | 27% / 23% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| gate_fwd | 0.394 / 0.551 | 0.355 / 0.439 | 0.83x [0.71-0.89] | 24% / 16% | noisy - not quoted | 8.94e-08 (1.0e-07) |
| gate_bwd | 1.869 / 2.096 | 0.736 / 1.001 | 0.48x [0.46-0.55] | 38% / 59% | noisy - not quoted | 2.25e-04 (2.3e-06) |
| vres_fwd | 0.242 / 0.321 | 0.429 / 0.551 | 1.71x [1.44-1.93] | 20% / 16% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| vres_bwd | 0.451 / 0.604 | 0.667 / 0.904 | 1.42x [1.30-2.05] | 38% / 34% | noisy - not quoted | 3.05e-05 (2.3e-07) |
| permute_bthd_bhtd | 0.263 / 0.338 | 0.346 / 0.411 | 1.22x [1.04-25.51] | 17% / 2090% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_fwd | 3.277 / 4.639 | 13.49 / 17.93 | 3.93x [3.51-5.15] | 40% / 32% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_bwd | 36.02 / 50.70 | 24.97 / 33.16 | 0.61x [0.49-0.78] | 47% / 34% | noisy - not quoted | 6.39e-14 (2.6e-10) |
| linear_ce_c1024x8192 | 250.3 / 261.8 | 218.5 / 276.5 | 0.87x [0.79-1.11] | 38% / 28% | noisy - not quoted | 9.54e-07 (6.7e-06) |
| linear_ce_c4096x50304 | 278.1 / 307.2 | 206.2 / 258.0 | 0.78x [0.70-0.86] | 15% / 28% | noisy - not quoted | 9.54e-07 (2.3e-05) |
| clip_grad_norm_full | 14.31 / 18.96 | 135.7 / 150.7 | 8.13x [6.72-9.57] | 48% / 13% | noisy - not quoted | 3.05e-05 (3.0e-07) |
| adamw_full | 22.33 / 31.75 | 35.67 / 47.68 | 1.53x [1.23-2.18] | 63% / 50% | noisy - not quoted | 3.73e-09 (1.2e-07) |
| muon_768x768 | 2.801 / 4.118 | 3.556 / 5.249 | 1.31x [0.98-1.61] | 87% / 69% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| muon_768x768 vs muon_768x768_bf16 | 2.801 / 4.118 | 1.769 / 2.146 | 0.52x [0.40-0.72] | 87% / 6% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| muon_2048x768 | 5.583 / 8.178 | 7.974 / 10.20 | 1.32x [1.25-1.44] | 30% / 41% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_2048x768 vs muon_2048x768_bf16 | 5.583 / 8.178 | 4.515 / 6.706 | 0.83x [0.80-0.97] | 30% / 57% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_768x2048 | 5.452 / 8.044 | 7.487 / 9.845 | 1.32x [1.21-1.40] | 36% / 40% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| muon_768x2048 vs muon_768x2048_bf16 | 5.452 / 8.044 | 3.716 / 5.628 | 0.73x [0.67-0.81] | 36% / 56% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| block_fwd | 22.26 / 29.91 | 15.16 / 19.48 | 0.66x [0.49-0.75] | 39% / 32% | noisy - not quoted | 2.38e-07 (1.9e-07) |
| block_fwd_bwd | 65.83 / 87.36 | 74.18 / 93.53 | 1.07x [0.96-1.16] | 35% / 32% | noisy - not quoted | 9.30e-06 (2.5e-06) |
| decode_attn_kv1024 | 0.134 / 0.205 | 0.093 / 0.174 | 0.82x [0.72-5.55] | 35% / 834% | noisy - not quoted | 1.86e-08 (3.2e-07) |
| accumulate_grad_50304x768 | 3.019 / 5.819 | 1.947 / 2.794 | 0.57x [0.35-0.80] | 72% / 82% | noisy - not quoted | 0.00e+00 (0.0e+00) |

### ojas-wgpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.139 / 0.291 | 0.068 / 0.125 | 0.52x [0.38-0.74] | 105% / 111% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_fwd | 1.839 / 2.479 | 0.786 / 0.956 | 0.42x [0.34-0.51] | 34% / 23% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_bwd | 3.381 / 4.159 | 1.691 / 2.141 | 0.51x [0.49-0.73] | 9% / 54% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_fwd | 4.170 / 5.617 | 1.920 / 2.727 | 0.49x [0.27-0.49] | 119% / 37% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_bwd | 8.722 / 10.60 | 3.901 / 5.033 | 0.47x [0.45-0.51] | 16% / 23% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_fwd | 4.370 / 5.872 | 1.963 / 2.561 | 0.45x [0.40-0.50] | 36% / 32% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_bwd | 8.707 / 10.45 | 3.814 / 5.206 | 0.50x [0.42-0.59] | 27% / 35% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_fwd | 138.6 / 157.9 | 54.10 / 69.04 | 0.43x [0.38-0.45] | 10% / 29% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_bwd | 199.2 / 223.2 | 106.7 / 129.8 | 0.57x [0.52-0.59] | 7% / 19% | noisy - not quoted | 1.57e-04 (1.3e-05) |
| sdpa_b4h12t1024d64_fwd | 11.43 / 13.59 | 1.241 / 1.384 | 0.10x [0.09-0.11] | 30% / 10% | noisy - not quoted | 2.38e-07 (2.4e-07) |
| sdpa_b4h12t1024d64_bwd | 52.87 / 60.60 | 9.558 / 13.91 | 0.22x [0.18-0.23] | 15% / 43% | noisy - not quoted | 1.10e-06 (2.1e-06) |
| sdpa_b4h8t2048d64_fwd | 30.95 / 34.66 | 3.032 / 3.975 | 0.11x [0.10-0.11] | 16% / 27% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b4h8t2048d64_bwd | 144.9 / 157.2 | 25.00 / 32.46 | 0.20x [0.18-0.22] | 8% / 31% | noisy - not quoted | 1.37e-06 (2.6e-06) |
| sdpa_b2h8t1024d128_fwd | 11.25 / 13.64 | 1.016 / 1.161 | 0.09x [0.08-0.09] | 19% / 17% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b2h8t1024d128_bwd | 55.08 / 61.61 | 4.356 / 6.752 | 0.11x [0.08-0.11] | 7% / 38% | noisy - not quoted | 1.31e-06 (2.3e-06) |
| rms_norm_fwd | 0.267 / 0.361 | 0.164 / 0.212 | 0.51x [0.46-0.68] | 37% / 30% | noisy - not quoted | 2.38e-07 (1.3e-07) |
| rms_norm_bwd | 0.490 / 0.649 | 1.741 / 2.058 | 3.15x [2.96-3.89] | 16% / 17% | noisy - not quoted | 3.34e-05 (2.6e-07) |
| rms_qk_norm_fwd | 0.634 / 0.767 | 0.280 / 0.361 | 0.47x [0.41-0.49] | 3% / 15% | noisy - not quoted | 2.38e-07 (1.2e-07) |
| rms_qk_norm_bwd | 1.461 / 1.706 | 4.239 / 6.698 | 3.85x [2.86-4.07] | 9% / 41% | noisy - not quoted | 4.58e-04 (1.3e-06) |
| rope_fwd | 0.227 / 0.327 | 1.128 / 1.257 | 3.96x [3.16-4.89] | 48% / 12% | noisy - not quoted | 1.19e-07 (6.2e-08) |
| rope_bwd | 0.228 / 0.352 | 1.422 / 1.658 | 4.73x [3.99-5.53] | 26% / 17% | noisy - not quoted | 1.19e-07 (6.1e-08) |
| silu_fwd | 0.388 / 0.539 | 0.336 / 0.465 | 0.86x [0.76-1.05] | 27% / 26% | noisy - not quoted | 7.15e-07 (1.8e-07) |
| silu_bwd | 0.545 / 0.691 | 0.490 / 0.655 | 0.95x [0.88-1.00] | 20% / 29% | noisy - not quoted | 4.77e-07 (4.3e-07) |
| mul_fwd | 0.537 / 0.695 | 0.468 / 0.603 | 0.91x [0.76-7.52] | 34% / 999% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| mul_bwd | 0.789 / 1.004 | 0.865 / 1.007 | 1.03x [0.95-1.32] | 11% / 42% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_fwd | 0.291 / 0.355 | 0.231 / 0.341 | 0.93x [0.76-1.00] | 10% / 24% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_bwd | 0.479 / 0.631 | 0.010 / 0.016 | 0.02x [0.02-0.03] | 40% / 23% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| gate_fwd | 0.521 / 0.647 | 0.355 / 0.439 | 0.69x [0.54-0.78] | 39% / 16% | noisy - not quoted | 1.19e-07 (1.4e-07) |
| gate_bwd | 1.566 / 1.816 | 0.736 / 1.001 | 0.56x [0.52-0.82] | 15% / 59% | noisy - not quoted | 2.21e-04 (2.3e-06) |
| vres_fwd | 0.283 / 0.368 | 0.429 / 0.551 | 1.50x [1.15-1.63] | 43% / 16% | noisy - not quoted | 5.96e-08 (6.0e-08) |
| vres_bwd | 0.478 / 0.604 | 0.667 / 0.904 | 1.52x [1.37-1.74] | 18% / 34% | noisy - not quoted | 3.05e-05 (2.3e-07) |
| permute_bthd_bhtd | 0.364 / 0.454 | 0.346 / 0.411 | 0.90x [0.87-16.09] | 22% / 2090% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_fwd | 6.506 / 8.539 | 13.49 / 17.93 | 2.00x [1.75-2.24] | 48% / 32% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_bwd | 45.68 / 54.43 | 24.97 / 33.16 | 0.60x [0.50-0.62] | 13% / 34% | noisy - not quoted | 6.39e-14 (2.6e-10) |
| linear_ce_c1024x8192 | 457.1 / 568.4 | 218.5 / 276.5 | 0.50x [0.48-0.50] | 24% / 28% | noisy - not quoted | 9.54e-07 (2.3e-05) |
| linear_ce_c4096x50304 | 351.0 / 401.2 | 206.2 / 258.0 | 0.62x [0.55-0.67] | 15% / 28% | noisy - not quoted | 1.82e-10 (2.3e-05) |
| clip_grad_norm_full | 16.33 / 19.12 | 135.7 / 150.7 | 7.88x [6.36-8.58] | 40% / 13% | noisy - not quoted | 3.81e-05 (3.8e-07) |
| adamw_full | 54.74 / 66.82 | 35.67 / 47.68 | 0.71x [0.52-0.82] | 62% / 50% | noisy - not quoted | 3.73e-09 (1.2e-07) |
| muon_768x768 | 5.744 / 7.805 | 3.556 / 5.249 | 0.66x [0.62-0.82] | 30% / 69% | noisy - not quoted | 1.49e-08 (4.5e-07) |
| muon_768x768 vs muon_768x768_bf16 | 5.744 / 7.805 | 1.769 / 2.146 | 0.28x [0.25-0.35] | 30% / 6% | noisy - not quoted | 1.49e-08 (4.5e-07) |
| muon_2048x768 | 13.33 / 16.61 | 7.974 / 10.20 | 0.63x [0.53-0.70] | 38% / 41% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_2048x768 vs muon_2048x768_bf16 | 13.33 / 16.61 | 4.515 / 6.706 | 0.39x [0.35-0.47] | 38% / 57% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_768x2048 | 12.75 / 16.63 | 7.487 / 9.845 | 0.61x [0.57-0.68] | 30% / 40% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| muon_768x2048 vs muon_768x2048_bf16 | 12.75 / 16.63 | 3.716 / 5.628 | 0.34x [0.31-0.39] | 30% / 56% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| block_fwd | 41.46 / 52.24 | 15.16 / 19.48 | 0.37x [0.36-0.40] | 21% / 32% | noisy - not quoted | 2.38e-07 (1.9e-07) |
| block_fwd_bwd | 151.9 / 180.0 | 74.18 / 93.53 | 0.50x [0.49-0.56] | 22% / 32% | noisy - not quoted | 8.58e-06 (2.3e-06) |
| decode_attn_kv1024 | 0.185 / 0.227 | 0.093 / 0.174 | 0.68x [0.52-5.31] | 56% / 834% | noisy - not quoted | 1.49e-08 (2.5e-07) |
| accumulate_grad_50304x768 | 2.018 / 2.278 | 1.947 / 2.794 | 1.23x [0.83-1.62] | 17% / 82% | noisy - not quoted | 0.00e+00 (0.0e+00) |

### Ranked: where ojas is slowest relative to torch (lowest ratio first)

| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |
| --: | :-- | :-- | :-- | --: | --: | :-- |
| 1 | ojas-wgpu | residual_add_bwd | 0.02x [0.02-0.03] | 0.631 | 0.016 | noisy |
| 2 | ojas-metal | residual_add_bwd | 0.03x [0.03-0.04] | 0.457 | 0.016 | noisy |
| 3 | ojas-wgpu | sdpa_b2h8t1024d128_fwd | 0.09x [0.08-0.09] | 13.64 | 1.161 | noisy |
| 4 | ojas-wgpu | sdpa_b4h12t1024d64_fwd | 0.10x [0.09-0.11] | 13.59 | 1.384 | noisy |
| 5 | ojas-wgpu | sdpa_b4h8t2048d64_fwd | 0.11x [0.10-0.11] | 34.66 | 3.975 | noisy |
| 6 | ojas-wgpu | sdpa_b2h8t1024d128_bwd | 0.11x [0.08-0.11] | 61.61 | 6.752 | noisy |
| 7 | ojas-wgpu | sdpa_b4h8t2048d64_bwd | 0.20x [0.18-0.22] | 157.2 | 32.46 | noisy |
| 8 | ojas-wgpu | sdpa_b4h12t1024d64_bwd | 0.22x [0.18-0.23] | 60.60 | 13.91 | noisy |
| 9 | ojas-wgpu | muon_768x768 vs muon_768x768_bf16 | 0.28x [0.25-0.35] | 7.805 | 2.146 | noisy |
| 10 | ojas-wgpu | muon_768x2048 vs muon_768x2048_bf16 | 0.34x [0.31-0.39] | 16.63 | 5.628 | noisy |
| 11 | ojas-wgpu | block_fwd | 0.37x [0.36-0.40] | 52.24 | 19.48 | noisy |
| 12 | ojas-wgpu | muon_2048x768 vs muon_2048x768_bf16 | 0.39x [0.35-0.47] | 16.61 | 6.706 | noisy |
| 13 | ojas-wgpu | linear_qkv_fwd | 0.42x [0.34-0.51] | 2.479 | 0.956 | noisy |
| 14 | ojas-wgpu | linear_lmhead_fwd | 0.43x [0.38-0.45] | 157.9 | 69.04 | noisy |
| 15 | ojas-wgpu | linear_down_fwd | 0.45x [0.40-0.50] | 5.872 | 2.561 | noisy |
| 16 | ojas-wgpu | rms_qk_norm_fwd | 0.47x [0.41-0.49] | 0.767 | 0.361 | noisy |
| 17 | ojas-wgpu | linear_up_bwd | 0.47x [0.45-0.51] | 10.60 | 5.033 | noisy |
| 18 | ojas-metal | gate_bwd | 0.48x [0.46-0.55] | 2.096 | 1.001 | noisy |
| 19 | ojas-metal | linear_lmhead_fwd | 0.48x [0.46-0.59] | 119.1 | 69.04 | noisy |
| 20 | ojas-wgpu | linear_up_fwd | 0.49x [0.27-0.49] | 5.617 | 2.727 | noisy |
| 21 | ojas-wgpu | linear_ce_c1024x8192 | 0.50x [0.48-0.50] | 568.4 | 276.5 | noisy |
| 22 | ojas-wgpu | linear_down_bwd | 0.50x [0.42-0.59] | 10.45 | 5.206 | noisy |
| 23 | ojas-wgpu | block_fwd_bwd | 0.50x [0.49-0.56] | 180.0 | 93.53 | noisy |
| 24 | ojas-wgpu | rms_norm_fwd | 0.51x [0.46-0.68] | 0.361 | 0.212 | noisy |
| 25 | ojas-wgpu | linear_qkv_bwd | 0.51x [0.49-0.73] | 4.159 | 2.141 | noisy |
| 26 | ojas-metal | muon_768x768 vs muon_768x768_bf16 | 0.52x [0.40-0.72] | 4.118 | 2.146 | noisy |
| 27 | ojas-wgpu | gate_bwd | 0.56x [0.52-0.82] | 1.816 | 1.001 | noisy |
| 28 | ojas-metal | accumulate_grad_50304x768 | 0.57x [0.35-0.80] | 5.819 | 2.794 | noisy |
| 29 | ojas-wgpu | linear_lmhead_bwd | 0.57x [0.52-0.59] | 223.2 | 129.8 | noisy |
| 30 | ojas-wgpu | cross_entropy_bwd | 0.60x [0.50-0.62] | 54.43 | 33.16 | noisy |
| 31 | ojas-metal | cross_entropy_bwd | 0.61x [0.49-0.78] | 50.70 | 33.16 | noisy |
| 32 | ojas-wgpu | muon_768x2048 | 0.61x [0.57-0.68] | 16.63 | 9.845 | noisy |
| 33 | ojas-wgpu | linear_ce_c4096x50304 | 0.62x [0.55-0.67] | 401.2 | 258.0 | noisy |
| 34 | ojas-metal | rms_norm_fwd | 0.62x [0.02-0.66] | 0.340 | 0.212 | noisy |
| 35 | ojas-wgpu | muon_2048x768 | 0.63x [0.53-0.70] | 16.61 | 10.20 | noisy |
| 36 | ojas-metal | sdpa_b4h8t2048d64_fwd | 0.65x [0.61-0.84] | 5.125 | 3.975 | noisy |
| 37 | ojas-metal | block_fwd | 0.66x [0.49-0.75] | 29.91 | 19.48 | noisy |
| 38 | ojas-wgpu | muon_768x768 | 0.66x [0.62-0.82] | 7.805 | 5.249 | noisy |
| 39 | ojas-wgpu | decode_attn_kv1024 | 0.68x [0.52-5.31] | 0.227 | 0.174 | noisy |
| 40 | ojas-wgpu | gate_fwd | 0.69x [0.54-0.78] | 0.647 | 0.439 | noisy |
| 41 | ojas-wgpu | adamw_full | 0.71x [0.52-0.82] | 66.82 | 47.68 | noisy |
| 42 | ojas-metal | linear_lmhead_bwd | 0.72x [0.66-0.77] | 170.7 | 129.8 | noisy |
| 43 | ojas-metal | sdpa_b4h12t1024d64_fwd | 0.72x [0.47-0.74] | 1.929 | 1.384 | noisy |
| 44 | ojas-metal | rms_qk_norm_fwd | 0.72x [0.61-0.79] | 0.506 | 0.361 | noisy |
| 45 | ojas-metal | muon_768x2048 vs muon_768x2048_bf16 | 0.73x [0.67-0.81] | 8.044 | 5.628 | noisy |
| 46 | ojas-metal | linear_down_bwd | 0.74x [0.68-1.03] | 5.602 | 5.206 | noisy |
| 47 | ojas-metal | linear_ce_c4096x50304 | 0.78x [0.70-0.86] | 307.2 | 258.0 | noisy |
| 48 | ojas-metal | decode_attn_kv1024 | 0.82x [0.72-5.55] | 0.205 | 0.174 | noisy |
| 49 | ojas-metal | muon_2048x768 vs muon_2048x768_bf16 | 0.83x [0.80-0.97] | 8.178 | 6.706 | noisy |
| 50 | ojas-metal | gate_fwd | 0.83x [0.71-0.89] | 0.551 | 0.439 | noisy |
| 51 | ojas-metal | linear_up_bwd | 0.86x [0.75-1.06] | 4.901 | 5.033 | noisy |
| 52 | ojas-wgpu | silu_fwd | 0.86x [0.76-1.05] | 0.539 | 0.465 | noisy |
| 53 | ojas-metal | linear_ce_c1024x8192 | 0.87x [0.79-1.11] | 261.8 | 276.5 | noisy |
| 54 | ojas-metal | linear_qkv_fwd | 0.90x [0.79-1.07] | 1.072 | 0.956 | noisy |
| 55 | ojas-wgpu | permute_bthd_bhtd | 0.90x [0.87-16.09] | 0.454 | 0.411 | noisy |
| 56 | ojas-wgpu | mul_fwd | 0.91x [0.76-7.52] | 0.695 | 0.603 | noisy |
| 57 | ojas-wgpu | residual_add_fwd | 0.93x [0.76-1.00] | 0.355 | 0.341 | noisy |
| 58 | ojas-wgpu | silu_bwd | 0.95x [0.88-1.00] | 0.691 | 0.655 | noisy |
| 59 | ojas-metal | linear_up_fwd | 0.96x [0.55-1.13] | 2.536 | 2.727 | noisy |
| 60 | ojas-metal | mul_fwd | 1.00x [0.88-9.03] | 0.602 | 0.603 | noisy |
| 61 | ojas-metal | sdpa_b2h8t1024d128_fwd | 1.01x [0.87-1.14] | 1.135 | 1.161 | noisy |
| 62 | ojas-metal | linear_qkv_bwd | 1.03x [0.69-1.63] | 1.901 | 2.141 | noisy |
| 63 | ojas-metal | sdpa_b2h8t1024d128_bwd | 1.03x [1.00-1.41] | 5.198 | 6.752 | noisy |
| 64 | ojas-metal | residual_add_fwd | 1.03x [0.84-1.15] | 0.331 | 0.341 | noisy |
| 65 | ojas-wgpu | mul_bwd | 1.03x [0.95-1.32] | 1.004 | 1.007 | noisy |
| 66 | ojas-metal | block_fwd_bwd | 1.07x [0.96-1.16] | 87.36 | 93.53 | noisy |
| 67 | ojas-metal | linear_down_fwd | 1.08x [0.94-1.28] | 2.310 | 2.561 | noisy |
| 68 | ojas-metal | silu_fwd | 1.11x [1.01-1.21] | 0.424 | 0.465 | noisy |
| 69 | ojas-metal | mul_bwd | 1.14x [0.60-1.30] | 0.864 | 1.007 | noisy |
| 70 | ojas-metal | silu_bwd | 1.14x [1.00-1.16] | 0.576 | 0.655 | noisy |
| 71 | ojas-metal | permute_bthd_bhtd | 1.22x [1.04-25.51] | 0.338 | 0.411 | noisy |
| 72 | ojas-wgpu | accumulate_grad_50304x768 | 1.23x [0.83-1.62] | 2.278 | 2.794 | noisy |
| 73 | ojas-metal | muon_768x768 | 1.31x [0.98-1.61] | 4.118 | 5.249 | noisy |
| 74 | ojas-metal | muon_2048x768 | 1.32x [1.25-1.44] | 8.178 | 10.20 | noisy |
| 75 | ojas-metal | muon_768x2048 | 1.32x [1.21-1.40] | 8.044 | 9.845 | noisy |
| 76 | ojas-metal | vres_bwd | 1.42x [1.30-2.05] | 0.604 | 0.904 | noisy |
| 77 | ojas-wgpu | vres_fwd | 1.50x [1.15-1.63] | 0.368 | 0.551 | noisy |
| 78 | ojas-wgpu | vres_bwd | 1.52x [1.37-1.74] | 0.604 | 0.904 | noisy |
| 79 | ojas-metal | adamw_full | 1.53x [1.23-2.18] | 31.75 | 47.68 | noisy |
| 80 | ojas-metal | sdpa_b4h12t1024d64_bwd | 1.71x [1.39-2.25] | 7.428 | 13.91 | noisy |
| 81 | ojas-metal | vres_fwd | 1.71x [1.44-1.93] | 0.321 | 0.551 | noisy |
| 82 | ojas-metal | sdpa_b4h8t2048d64_bwd | 1.79x [1.54-2.17] | 17.38 | 32.46 | noisy |
| 83 | ojas-wgpu | cross_entropy_fwd | 2.00x [1.75-2.24] | 8.539 | 17.93 | noisy |
| 84 | ojas-wgpu | rms_norm_bwd | 3.15x [2.96-3.89] | 0.649 | 2.058 | noisy |
| 85 | ojas-metal | rms_norm_bwd | 3.32x [2.96-3.91] | 0.643 | 2.058 | noisy |
| 86 | ojas-metal | rope_fwd | 3.61x [3.44-4.12] | 0.348 | 1.257 | noisy |
| 87 | ojas-wgpu | rms_qk_norm_bwd | 3.85x [2.86-4.07] | 1.706 | 6.698 | noisy |
| 88 | ojas-metal | cross_entropy_fwd | 3.93x [3.51-5.15] | 4.639 | 17.93 | noisy |
| 89 | ojas-wgpu | rope_fwd | 3.96x [3.16-4.89] | 0.327 | 1.257 | noisy |
| 90 | ojas-wgpu | rope_bwd | 4.73x [3.99-5.53] | 0.352 | 1.658 | noisy |
| 91 | ojas-metal | rope_bwd | 4.90x [4.24-5.34] | 0.341 | 1.658 | noisy |
| 92 | ojas-metal | rms_qk_norm_bwd | 5.11x [3.47-5.64] | 1.318 | 6.698 | noisy |
| 93 | ojas-wgpu | clip_grad_norm_full | 7.88x [6.36-8.58] | 19.12 | 150.7 | noisy |
| 94 | ojas-metal | clip_grad_norm_full | 8.13x [6.72-9.57] | 18.96 | 150.7 | noisy |

### Machine load around each runtime block

| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |
| --: | :-- | :-- | :-- | :-- | --: |
| 1 | ojas-metal | before | 00:54:53 | 24.32 26.07 29.12 | 100 |
| 1 | ojas-metal | after | 00:55:34 | 25.19 26.04 28.96 | 100 |
| 1 | ojas-wgpu | before | 00:55:34 | 25.19 26.04 28.96 | 100 |
| 1 | ojas-wgpu | after | 00:56:35 | 24.95 25.80 28.67 | 100 |
| 1 | torch-mps | before | 00:56:35 | 24.95 25.80 28.67 | 100 |
| 1 | torch-mps | after | 00:57:20 | 28.05 26.41 28.74 | 100 |
| 2 | torch-mps | before | 00:57:20 | 28.05 26.41 28.74 | 100 |
| 2 | torch-mps | after | 00:58:04 | 24.44 25.67 28.35 | 100 |
| 2 | ojas-wgpu | before | 00:58:04 | 24.44 25.67 28.35 | 100 |
| 2 | ojas-wgpu | after | 00:59:03 | 20.90 24.58 27.77 | 100 |
| 2 | ojas-metal | before | 00:59:03 | 20.90 24.58 27.77 | 100 |
| 2 | ojas-metal | after | 00:59:38 | 20.26 24.00 27.42 | 100 |
| 3 | ojas-metal | before | 00:59:38 | 20.26 24.00 27.42 | 100 |
| 3 | ojas-metal | after | 01:00:16 | 19.86 23.43 27.07 | 100 |
| 3 | ojas-wgpu | before | 01:00:16 | 19.86 23.43 27.07 | 100 |
| 3 | ojas-wgpu | after | 01:01:21 | 18.10 22.33 26.40 | 100 |
| 3 | torch-mps | before | 01:01:21 | 18.10 22.33 26.40 | 100 |
| 3 | torch-mps | after | 01:02:08 | 21.43 22.34 26.15 | 100 |
| 4 | torch-mps | before | 01:02:08 | 21.43 22.34 26.15 | 100 |
| 4 | torch-mps | after | 01:02:54 | 17.97 21.30 25.57 | 100 |
| 4 | ojas-wgpu | before | 01:02:54 | 17.97 21.30 25.57 | 100 |
| 4 | ojas-wgpu | after | 01:04:03 | 25.10 22.51 25.67 | 100 |
| 4 | ojas-metal | before | 01:04:03 | 25.10 22.51 25.67 | 100 |
| 4 | ojas-metal | after | 01:04:45 | 22.68 22.22 25.42 | 100 |
| 5 | ojas-metal | before | 01:04:45 | 22.68 22.22 25.42 | 100 |
| 5 | ojas-metal | after | 01:05:27 | 21.92 21.99 25.16 | 100 |
| 5 | ojas-wgpu | before | 01:05:28 | 21.92 21.99 25.16 | 100 |
| 5 | ojas-wgpu | after | 01:06:33 | 18.43 21.01 24.56 | 100 |
| 5 | torch-mps | before | 01:06:33 | 18.43 21.01 24.56 | 100 |
| 5 | torch-mps | after | 01:07:24 | 22.07 21.36 24.47 | 100 |

### Load recorded before and after every row

| round | lane | samples | 1-min load min-max | GPU util % min / median / max |
| --: | :-- | --: | :-- | :-- |
| 1 | ojas-metal | 90 | 23.5-25.2 | 100 / 100 / 100 |
| 1 | ojas-wgpu | 90 | 24.2-26.4 | 100 / 100 / 100 |
| 1 | torch-mps | 96 | 23.9-28.1 | 100 / 100 / 100 |
| 2 | ojas-metal | 90 | 19.7-21.4 | 100 / 100 / 100 |
| 2 | ojas-wgpu | 90 | 20.9-25.9 | 100 / 100 / 100 |
| 2 | torch-mps | 96 | 23.2-28.6 | 100 / 100 / 100 |
| 3 | ojas-metal | 90 | 18.9-20.3 | 100 / 100 / 100 |
| 3 | ojas-wgpu | 90 | 18.1-21.0 | 100 / 100 / 100 |
| 3 | torch-mps | 96 | 16.9-21.4 | 100 / 100 / 100 |
| 4 | ojas-metal | 90 | 22.7-25.1 | 100 / 100 / 100 |
| 4 | ojas-wgpu | 90 | 17.7-25.9 | 100 / 100 / 100 |
| 4 | torch-mps | 96 | 18.0-21.4 | 100 / 100 / 100 |
| 5 | ojas-metal | 90 | 20.5-22.7 | 100 / 100 / 100 |
| 5 | ojas-wgpu | 90 | 18.4-21.9 | 100 / 100 / 100 |
| 5 | torch-mps | 96 | 17.5-22.1 | 100 / 100 / 100 |
