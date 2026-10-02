# Paired GPU-vs-torch summary: /Users/bharath/Code/research/ojas/bench/results/2026-10-01-r3b

5 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.

```
date: 2026-10-02T03:50:15Z
git: dab2a12fd2129565515954924e1a11737ca6ee47 (dirty files: 227)
uncommitted diff of the benchmarked crates (sha1): de67949a36ff
tessl: cf65d9d5d3deb97b0847e020a562ac8f15e6d5a9 (dirty files: 20)
rustc: rustc 1.98.0 (88d9e12ae 2026-08-18)
cargo: cargo 1.98.0 (797e8a9bc 2026-08-05)
os: macOS 27.0.1 (26A434)
cpu: Apple M5 Pro
memory_bytes: 68719476736
metal_bin: Oct  1 22:50:15 2026 3868432
wgpu_bin: Oct  1 22:50:11 2026 6376352
rounds: 5  iters: 20  warmup: 5  rows: all
```

### ojas-metal vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.114 / 0.384 | 0.073 / 0.118 | 0.28x [0.07-0.47] | 622% / 69% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_fwd | 1.054 / 1.884 | 0.785 / 1.093 | 0.65x [0.32-1.25] | 102% / 95% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_bwd | 1.911 / 2.162 | 1.601 / 2.415 | 1.12x [0.77-3.71] | 18% / 367% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_fwd | 2.413 / 3.594 | 1.896 / 2.799 | 0.76x [0.53-1.31] | 66% / 93% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_bwd | 4.449 / 4.861 | 3.917 / 5.070 | 0.85x [0.80-1.15] | 35% / 35% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_fwd | 2.271 / 3.142 | 1.973 / 2.504 | 0.78x [0.66-0.81] | 94% / 93% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_bwd | 5.163 / 5.741 | 3.875 / 5.066 | 0.73x [0.70-0.97] | 27% / 57% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_fwd | 109.1 / 117.0 | 53.69 / 68.18 | 0.55x [0.49-0.61] | 19% / 33% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_bwd | 166.4 / 192.4 | 131.2 / 148.8 | 0.76x [0.74-0.90] | 12% / 20% | noisy - not quoted | 1.57e-04 (1.3e-05) |
| sdpa_b4h12t1024d64_fwd | 2.152 / 3.507 | 1.336 / 1.959 | 0.62x [0.53-1.26] | 42% / 173% | noisy - not quoted | 2.24e-07 (2.2e-07) |
| sdpa_b4h12t1024d64_bwd | 6.093 / 7.490 | 9.235 / 11.87 | 1.50x [1.26-1.69] | 23% / 51% | noisy - not quoted | 1.10e-06 (2.1e-06) |
| sdpa_b4h8t2048d64_fwd | 4.554 / 5.568 | 3.108 / 4.123 | 0.63x [0.59-1.41] | 30% / 142% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b4h8t2048d64_bwd | 16.28 / 18.28 | 23.27 / 26.80 | 1.59x [1.21-1.75] | 16% / 40% | noisy - not quoted | 1.31e-06 (2.5e-06) |
| sdpa_b2h8t1024d128_fwd | 1.230 / 2.383 | 1.008 / 1.217 | 0.65x [0.39-2.97] | 105% / 279% | noisy - not quoted | 2.09e-07 (2.1e-07) |
| sdpa_b2h8t1024d128_bwd | 4.539 / 5.367 | 4.282 / 5.893 | 0.97x [0.83-1.44] | 40% / 68% | noisy - not quoted | 1.10e-06 (2.6e-06) |
| rms_norm_fwd | 0.361 / 1.289 | 0.159 / 0.226 | 0.29x [0.11-0.53] | 159% / 219% | noisy - not quoted | 3.58e-07 (2.0e-07) |
| rms_norm_bwd | 0.684 / 2.070 | 1.748 / 2.217 | 1.07x [0.59-5.56] | 273% / 222% | noisy - not quoted | 3.81e-05 (2.8e-07) |
| rms_qk_norm_fwd | 0.593 / 1.044 | 0.289 / 0.423 | 0.41x [0.17-1.13] | 156% / 162% | noisy - not quoted | 3.58e-07 (1.7e-07) |
| rms_qk_norm_bwd | 1.369 / 1.657 | 4.097 / 5.679 | 2.83x [2.15-5.54] | 101% / 104% | noisy - not quoted | 3.66e-04 (1.0e-06) |
| rope_fwd | 0.380 / 1.625 | 1.152 / 1.561 | 1.75x [0.63-3.37] | 245% / 363% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| rope_bwd | 0.373 / 1.637 | 1.421 / 1.831 | 3.12x [0.93-4.24] | 298% / 335% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| silu_fwd | 0.813 / 1.149 | 0.331 / 0.469 | 0.37x [0.15-0.76] | 148% / 108% | noisy - not quoted | 2.38e-07 (6.1e-08) |
| silu_bwd | 1.119 / 2.672 | 0.490 / 0.704 | 0.23x [0.20-1.19] | 24% / 455% | noisy - not quoted | 1.19e-07 (1.1e-07) |
| mul_fwd | 1.134 / 2.026 | 0.468 / 0.659 | 0.40x [0.20-0.77] | 83% / 213% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| mul_bwd | 1.796 / 3.036 | 0.859 / 1.133 | 0.40x [0.31-2.69] | 61% / 775% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_fwd | 0.557 / 1.858 | 0.221 / 0.333 | 0.28x [0.12-0.43] | 136% / 131% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_bwd | 0.567 / 1.748 | 0.012 / 0.017 | 0.01x [0.01-0.18] | 149% / 1122% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| gate_fwd | 0.554 / 2.085 | 0.383 / 0.511 | 0.22x [0.13-0.87] | 87% / 252% | noisy - not quoted | 8.94e-08 (1.0e-07) |
| gate_bwd | 1.951 / 2.697 | 0.712 / 1.194 | 0.38x [0.22-2.60] | 68% / 744% | noisy - not quoted | 2.25e-04 (2.3e-06) |
| vres_fwd | 0.515 / 1.805 | 0.439 / 0.614 | 0.33x [0.25-0.90] | 54% / 137% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| vres_bwd | 0.867 / 2.248 | 0.666 / 1.035 | 0.66x [0.32-1.01] | 121% / 215% | noisy - not quoted | 3.05e-05 (2.3e-07) |
| permute_bthd_bhtd | 0.326 / 1.602 | 0.355 / 0.498 | 0.29x [0.20-1.52] | 74% / 335% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_fwd | 3.138 / 4.358 | 13.10 / 15.73 | 3.64x [2.08-3.99] | 184% / 49% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_bwd | 31.55 / 51.99 | 23.52 / 26.53 | 0.60x [0.40-0.69] | 148% / 57% | noisy - not quoted | 6.39e-14 (2.6e-10) |
| linear_ce_c1024x8192 | 268.1 / 315.9 | 246.6 / 285.1 | 0.90x [0.80-0.93] | 12% / 21% | noisy - not quoted | 9.54e-07 (6.7e-06) |
| linear_ce_c4096x50304 | 298.6 / 318.0 | 206.8 / 251.8 | 0.81x [0.74-0.93] | 11% / 27% | noisy - not quoted | 9.54e-07 (2.3e-05) |
| clip_grad_norm_full | 16.78 / 18.60 | 132.2 / 138.7 | 7.19x [5.64-7.91] | 30% / 8% | noisy - not quoted | 3.05e-05 (3.0e-07) |
| adamw_full | 21.41 / 23.65 | 33.56 / 40.18 | 1.55x [1.18-5.35] | 102% / 597% | noisy - not quoted | 3.73e-09 (1.2e-07) |
| muon_768x768 | 2.866 / 3.827 | 3.388 / 4.667 | 1.22x [0.88-1.48] | 174% / 172% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| muon_768x768 vs muon_768x768_bf16 | 2.866 / 3.827 | 1.758 / 2.398 | 0.64x [0.48-0.93] | 174% / 160% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| muon_2048x768 | 5.747 / 6.411 | 7.529 / 8.082 | 1.26x [0.62-2.06] | 179% / 66% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_2048x768 vs muon_2048x768_bf16 | 5.747 / 6.411 | 4.228 / 4.784 | 0.73x [0.34-1.35] | 179% / 88% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_768x2048 | 5.667 / 6.316 | 7.248 / 7.944 | 1.23x [1.04-1.82] | 97% / 83% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| muon_768x2048 vs muon_768x2048_bf16 | 5.667 / 6.316 | 3.562 / 4.173 | 0.63x [0.45-1.48] | 97% / 129% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| block_fwd | 22.48 / 25.49 | 14.29 / 14.95 | 0.59x [0.23-0.72] | 272% / 34% | noisy - not quoted | 2.38e-07 (1.9e-07) |
| block_fwd_bwd | 71.10 / 76.68 | 67.62 / 70.67 | 0.92x [0.30-1.23] | 262% / 34% | noisy - not quoted | 9.30e-06 (2.5e-06) |
| decode_attn_kv1024 | 0.150 / 1.383 | 0.089 / 0.131 | 0.19x [0.07-1.99] | 270% / 2457% | noisy - not quoted | 1.86e-08 (3.2e-07) |
| accumulate_grad_50304x768 | 3.018 / 4.594 | 1.833 / 2.108 | 0.46x [0.29-0.76] | 83% / 50% | noisy - not quoted | 0.00e+00 (0.0e+00) |

### ojas-wgpu vs torch-mps

| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |
| :-- | --: | --: | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.149 / 0.244 | 0.073 / 0.118 | 0.52x [0.05-0.56] | 956% / 69% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_fwd | 1.745 / 2.055 | 0.785 / 1.093 | 0.42x [0.36-0.83] | 54% / 95% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_qkv_bwd | 3.366 / 3.809 | 1.601 / 2.415 | 0.48x [0.40-2.16] | 67% / 367% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_fwd | 4.108 / 4.430 | 1.896 / 2.799 | 0.51x [0.46-0.84] | 24% / 93% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_up_bwd | 8.602 / 9.715 | 3.917 / 5.070 | 0.48x [0.45-0.57] | 20% / 35% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_fwd | 4.316 / 5.190 | 1.973 / 2.504 | 0.43x [0.43-0.75] | 33% / 93% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_down_bwd | 8.684 / 9.455 | 3.875 / 5.066 | 0.45x [0.42-0.64] | 32% / 57% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_fwd | 131.4 / 152.9 | 53.69 / 68.18 | 0.45x [0.37-0.50] | 12% / 33% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| linear_lmhead_bwd | 211.4 / 259.6 | 131.2 / 148.8 | 0.58x [0.55-0.75] | 19% / 20% | noisy - not quoted | 1.57e-04 (1.3e-05) |
| sdpa_b4h12t1024d64_fwd | 12.06 / 13.92 | 1.336 / 1.959 | 0.14x [0.10-0.32] | 15% / 173% | noisy - not quoted | 2.38e-07 (2.4e-07) |
| sdpa_b4h12t1024d64_bwd | 55.04 / 60.63 | 9.235 / 11.87 | 0.20x [0.14-0.24] | 19% / 51% | noisy - not quoted | 1.10e-06 (2.1e-06) |
| sdpa_b4h8t2048d64_fwd | 31.87 / 35.98 | 3.108 / 4.123 | 0.10x [0.09-0.24] | 19% / 142% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b4h8t2048d64_bwd | 146.8 / 164.0 | 23.27 / 26.80 | 0.17x [0.14-0.20] | 11% / 40% | noisy - not quoted | 1.37e-06 (2.6e-06) |
| sdpa_b2h8t1024d128_fwd | 11.88 / 13.07 | 1.008 / 1.217 | 0.09x [0.08-0.33] | 8% / 279% | noisy - not quoted | 1.79e-07 (1.8e-07) |
| sdpa_b2h8t1024d128_bwd | 55.54 / 59.13 | 4.282 / 5.893 | 0.10x [0.08-0.13] | 6% / 68% | noisy - not quoted | 1.31e-06 (2.3e-06) |
| rms_norm_fwd | 0.262 / 0.368 | 0.159 / 0.226 | 0.70x [0.35-2.39] | 123% / 219% | noisy - not quoted | 2.38e-07 (1.3e-07) |
| rms_norm_bwd | 0.472 / 0.652 | 1.748 / 2.217 | 3.48x [2.43-8.88] | 208% / 222% | noisy - not quoted | 3.34e-05 (2.6e-07) |
| rms_qk_norm_fwd | 0.613 / 0.768 | 0.289 / 0.423 | 0.51x [0.44-0.62] | 199% / 162% | noisy - not quoted | 2.38e-07 (1.2e-07) |
| rms_qk_norm_bwd | 1.435 / 1.737 | 4.097 / 5.679 | 2.67x [1.77-5.47] | 109% / 104% | noisy - not quoted | 4.58e-04 (1.3e-06) |
| rope_fwd | 0.223 / 0.324 | 1.152 / 1.561 | 3.79x [1.83-22.94] | 444% / 363% | noisy - not quoted | 1.19e-07 (6.2e-08) |
| rope_bwd | 0.228 / 0.297 | 1.421 / 1.831 | 6.49x [2.81-13.10] | 176% / 335% | noisy - not quoted | 1.19e-07 (6.1e-08) |
| silu_fwd | 0.399 / 0.494 | 0.331 / 0.469 | 0.76x [0.29-1.48] | 230% / 108% | noisy - not quoted | 7.15e-07 (1.8e-07) |
| silu_bwd | 0.546 / 0.656 | 0.490 / 0.704 | 0.88x [0.73-2.28] | 125% / 455% | noisy - not quoted | 4.77e-07 (4.3e-07) |
| mul_fwd | 0.557 / 0.705 | 0.468 / 0.659 | 0.88x [0.41-1.55] | 164% / 213% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| mul_bwd | 0.824 / 1.011 | 0.859 / 1.133 | 1.21x [0.32-8.09] | 312% / 775% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_fwd | 0.284 / 0.444 | 0.221 / 0.333 | 0.63x [0.29-1.36] | 278% / 131% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| residual_add_bwd | 0.428 / 0.617 | 0.012 / 0.017 | 0.03x [0.02-0.11] | 172% / 1122% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| gate_fwd | 0.528 / 0.616 | 0.383 / 0.511 | 0.77x [0.34-1.48] | 159% / 252% | noisy - not quoted | 1.19e-07 (1.4e-07) |
| gate_bwd | 1.578 / 1.867 | 0.712 / 1.194 | 0.53x [0.47-1.95] | 106% / 744% | noisy - not quoted | 2.21e-04 (2.3e-06) |
| vres_fwd | 0.318 / 0.520 | 0.439 / 0.614 | 1.39x [0.93-2.51] | 112% / 137% | noisy - not quoted | 5.96e-08 (6.0e-08) |
| vres_bwd | 0.495 / 0.605 | 0.666 / 1.035 | 1.44x [1.26-1.93] | 196% / 215% | noisy - not quoted | 3.05e-05 (2.3e-07) |
| permute_bthd_bhtd | 0.348 / 0.545 | 0.355 / 0.498 | 1.02x [0.53-1.50] | 196% / 335% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_fwd | 6.108 / 6.731 | 13.10 / 15.73 | 2.04x [1.96-2.40] | 40% / 49% | noisy - not quoted | 0.00e+00 (0.0e+00) |
| cross_entropy_bwd | 34.31 / 41.03 | 23.52 / 26.53 | 0.63x [0.40-0.65] | 131% / 57% | noisy - not quoted | 6.39e-14 (2.6e-10) |
| linear_ce_c1024x8192 | 471.0 / 557.6 | 246.6 / 285.1 | 0.48x [0.46-0.58] | 20% / 21% | noisy - not quoted | 9.54e-07 (2.3e-05) |
| linear_ce_c4096x50304 | 373.1 / 423.8 | 206.8 / 251.8 | 0.59x [0.56-0.68] | 6% / 27% | noisy - not quoted | 1.82e-10 (2.3e-05) |
| clip_grad_norm_full | 14.99 / 20.47 | 132.2 / 138.7 | 6.47x [4.61-8.92] | 89% / 8% | noisy - not quoted | 3.81e-05 (3.8e-07) |
| adamw_full | 51.39 / 77.48 | 33.56 / 40.18 | 0.60x [0.51-1.87] | 150% / 597% | noisy - not quoted | 3.73e-09 (1.2e-07) |
| muon_768x768 | 5.536 / 7.884 | 3.388 / 4.667 | 0.62x [0.59-1.24] | 59% / 172% | noisy - not quoted | 1.49e-08 (4.5e-07) |
| muon_768x768 vs muon_768x768_bf16 | 5.536 / 7.884 | 1.758 / 2.398 | 0.34x [0.30-0.62] | 59% / 160% | noisy - not quoted | 1.49e-08 (4.5e-07) |
| muon_2048x768 | 12.84 / 15.68 | 7.529 / 8.082 | 0.60x [0.47-0.68] | 50% / 66% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_2048x768 vs muon_2048x768_bf16 | 12.84 / 15.68 | 4.228 / 4.784 | 0.35x [0.28-0.43] | 50% / 88% | noisy - not quoted | 1.68e-08 (5.0e-07) |
| muon_768x2048 | 12.47 / 13.74 | 7.248 / 7.944 | 0.60x [0.47-1.03] | 38% / 83% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| muon_768x2048 vs muon_768x2048_bf16 | 12.47 / 13.74 | 3.562 / 4.173 | 0.31x [0.25-0.51] | 38% / 129% | noisy - not quoted | 9.31e-09 (2.9e-07) |
| block_fwd | 41.69 / 50.48 | 14.29 / 14.95 | 0.30x [0.25-0.43] | 38% / 34% | noisy - not quoted | 2.38e-07 (1.9e-07) |
| block_fwd_bwd | 150.1 / 188.3 | 67.62 / 70.67 | 0.41x [0.34-0.48] | 21% / 34% | noisy - not quoted | 8.58e-06 (2.3e-06) |
| decode_attn_kv1024 | 0.212 / 0.340 | 0.089 / 0.131 | 0.47x [0.36-3.76] | 233% / 2457% | noisy - not quoted | 1.49e-08 (2.5e-07) |
| accumulate_grad_50304x768 | 1.997 / 2.806 | 1.833 / 2.108 | 0.92x [0.74-1.05] | 35% / 50% | noisy - not quoted | 0.00e+00 (0.0e+00) |

### Ranked: where ojas is slowest relative to torch (lowest ratio first)

| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |
| --: | :-- | :-- | :-- | --: | --: | :-- |
| 1 | ojas-metal | residual_add_bwd | 0.01x [0.01-0.18] | 1.748 | 0.017 | noisy |
| 2 | ojas-wgpu | residual_add_bwd | 0.03x [0.02-0.11] | 0.617 | 0.017 | noisy |
| 3 | ojas-wgpu | sdpa_b2h8t1024d128_fwd | 0.09x [0.08-0.33] | 13.07 | 1.217 | noisy |
| 4 | ojas-wgpu | sdpa_b2h8t1024d128_bwd | 0.10x [0.08-0.13] | 59.13 | 5.893 | noisy |
| 5 | ojas-wgpu | sdpa_b4h8t2048d64_fwd | 0.10x [0.09-0.24] | 35.98 | 4.123 | noisy |
| 6 | ojas-wgpu | sdpa_b4h12t1024d64_fwd | 0.14x [0.10-0.32] | 13.92 | 1.959 | noisy |
| 7 | ojas-wgpu | sdpa_b4h8t2048d64_bwd | 0.17x [0.14-0.20] | 164.0 | 26.80 | noisy |
| 8 | ojas-metal | decode_attn_kv1024 | 0.19x [0.07-1.99] | 1.383 | 0.131 | noisy |
| 9 | ojas-wgpu | sdpa_b4h12t1024d64_bwd | 0.20x [0.14-0.24] | 60.63 | 11.87 | noisy |
| 10 | ojas-metal | gate_fwd | 0.22x [0.13-0.87] | 2.085 | 0.511 | noisy |
| 11 | ojas-metal | silu_bwd | 0.23x [0.20-1.19] | 2.672 | 0.704 | noisy |
| 12 | ojas-metal | residual_add_fwd | 0.28x [0.12-0.43] | 1.858 | 0.333 | noisy |
| 13 | ojas-metal | rms_norm_fwd | 0.29x [0.11-0.53] | 1.289 | 0.226 | noisy |
| 14 | ojas-metal | permute_bthd_bhtd | 0.29x [0.20-1.52] | 1.602 | 0.498 | noisy |
| 15 | ojas-wgpu | block_fwd | 0.30x [0.25-0.43] | 50.48 | 14.95 | noisy |
| 16 | ojas-wgpu | muon_768x2048 vs muon_768x2048_bf16 | 0.31x [0.25-0.51] | 13.74 | 4.173 | noisy |
| 17 | ojas-metal | vres_fwd | 0.33x [0.25-0.90] | 1.805 | 0.614 | noisy |
| 18 | ojas-wgpu | muon_768x768 vs muon_768x768_bf16 | 0.34x [0.30-0.62] | 7.884 | 2.398 | noisy |
| 19 | ojas-wgpu | muon_2048x768 vs muon_2048x768_bf16 | 0.35x [0.28-0.43] | 15.68 | 4.784 | noisy |
| 20 | ojas-metal | silu_fwd | 0.37x [0.15-0.76] | 1.149 | 0.469 | noisy |
| 21 | ojas-metal | gate_bwd | 0.38x [0.22-2.60] | 2.697 | 1.194 | noisy |
| 22 | ojas-metal | mul_fwd | 0.40x [0.20-0.77] | 2.026 | 0.659 | noisy |
| 23 | ojas-metal | mul_bwd | 0.40x [0.31-2.69] | 3.036 | 1.133 | noisy |
| 24 | ojas-wgpu | block_fwd_bwd | 0.41x [0.34-0.48] | 188.3 | 70.67 | noisy |
| 25 | ojas-metal | rms_qk_norm_fwd | 0.41x [0.17-1.13] | 1.044 | 0.423 | noisy |
| 26 | ojas-wgpu | linear_qkv_fwd | 0.42x [0.36-0.83] | 2.055 | 1.093 | noisy |
| 27 | ojas-wgpu | linear_down_fwd | 0.43x [0.43-0.75] | 5.190 | 2.504 | noisy |
| 28 | ojas-wgpu | linear_lmhead_fwd | 0.45x [0.37-0.50] | 152.9 | 68.18 | noisy |
| 29 | ojas-wgpu | linear_down_bwd | 0.45x [0.42-0.64] | 9.455 | 5.066 | noisy |
| 30 | ojas-metal | accumulate_grad_50304x768 | 0.46x [0.29-0.76] | 4.594 | 2.108 | noisy |
| 31 | ojas-wgpu | decode_attn_kv1024 | 0.47x [0.36-3.76] | 0.340 | 0.131 | noisy |
| 32 | ojas-wgpu | linear_qkv_bwd | 0.48x [0.40-2.16] | 3.809 | 2.415 | noisy |
| 33 | ojas-wgpu | linear_up_bwd | 0.48x [0.45-0.57] | 9.715 | 5.070 | noisy |
| 34 | ojas-wgpu | linear_ce_c1024x8192 | 0.48x [0.46-0.58] | 557.6 | 285.1 | noisy |
| 35 | ojas-wgpu | rms_qk_norm_fwd | 0.51x [0.44-0.62] | 0.768 | 0.423 | noisy |
| 36 | ojas-wgpu | linear_up_fwd | 0.51x [0.46-0.84] | 4.430 | 2.799 | noisy |
| 37 | ojas-wgpu | gate_bwd | 0.53x [0.47-1.95] | 1.867 | 1.194 | noisy |
| 38 | ojas-metal | linear_lmhead_fwd | 0.55x [0.49-0.61] | 117.0 | 68.18 | noisy |
| 39 | ojas-wgpu | linear_lmhead_bwd | 0.58x [0.55-0.75] | 259.6 | 148.8 | noisy |
| 40 | ojas-metal | block_fwd | 0.59x [0.23-0.72] | 25.49 | 14.95 | noisy |
| 41 | ojas-wgpu | linear_ce_c4096x50304 | 0.59x [0.56-0.68] | 423.8 | 251.8 | noisy |
| 42 | ojas-metal | cross_entropy_bwd | 0.60x [0.40-0.69] | 51.99 | 26.53 | noisy |
| 43 | ojas-wgpu | muon_768x2048 | 0.60x [0.47-1.03] | 13.74 | 7.944 | noisy |
| 44 | ojas-wgpu | adamw_full | 0.60x [0.51-1.87] | 77.48 | 40.18 | noisy |
| 45 | ojas-wgpu | muon_2048x768 | 0.60x [0.47-0.68] | 15.68 | 8.082 | noisy |
| 46 | ojas-wgpu | muon_768x768 | 0.62x [0.59-1.24] | 7.884 | 4.667 | noisy |
| 47 | ojas-metal | sdpa_b4h12t1024d64_fwd | 0.62x [0.53-1.26] | 3.507 | 1.959 | noisy |
| 48 | ojas-metal | muon_768x2048 vs muon_768x2048_bf16 | 0.63x [0.45-1.48] | 6.316 | 4.173 | noisy |
| 49 | ojas-wgpu | cross_entropy_bwd | 0.63x [0.40-0.65] | 41.03 | 26.53 | noisy |
| 50 | ojas-metal | sdpa_b4h8t2048d64_fwd | 0.63x [0.59-1.41] | 5.568 | 4.123 | noisy |
| 51 | ojas-wgpu | residual_add_fwd | 0.63x [0.29-1.36] | 0.444 | 0.333 | noisy |
| 52 | ojas-metal | muon_768x768 vs muon_768x768_bf16 | 0.64x [0.48-0.93] | 3.827 | 2.398 | noisy |
| 53 | ojas-metal | linear_qkv_fwd | 0.65x [0.32-1.25] | 1.884 | 1.093 | noisy |
| 54 | ojas-metal | sdpa_b2h8t1024d128_fwd | 0.65x [0.39-2.97] | 2.383 | 1.217 | noisy |
| 55 | ojas-metal | vres_bwd | 0.66x [0.32-1.01] | 2.248 | 1.035 | noisy |
| 56 | ojas-wgpu | rms_norm_fwd | 0.70x [0.35-2.39] | 0.368 | 0.226 | noisy |
| 57 | ojas-metal | muon_2048x768 vs muon_2048x768_bf16 | 0.73x [0.34-1.35] | 6.411 | 4.784 | noisy |
| 58 | ojas-metal | linear_down_bwd | 0.73x [0.70-0.97] | 5.741 | 5.066 | noisy |
| 59 | ojas-metal | linear_up_fwd | 0.76x [0.53-1.31] | 3.594 | 2.799 | noisy |
| 60 | ojas-wgpu | silu_fwd | 0.76x [0.29-1.48] | 0.494 | 0.469 | noisy |
| 61 | ojas-metal | linear_lmhead_bwd | 0.76x [0.74-0.90] | 192.4 | 148.8 | noisy |
| 62 | ojas-wgpu | gate_fwd | 0.77x [0.34-1.48] | 0.616 | 0.511 | noisy |
| 63 | ojas-metal | linear_down_fwd | 0.78x [0.66-0.81] | 3.142 | 2.504 | noisy |
| 64 | ojas-metal | linear_ce_c4096x50304 | 0.81x [0.74-0.93] | 318.0 | 251.8 | noisy |
| 65 | ojas-metal | linear_up_bwd | 0.85x [0.80-1.15] | 4.861 | 5.070 | noisy |
| 66 | ojas-wgpu | mul_fwd | 0.88x [0.41-1.55] | 0.705 | 0.659 | noisy |
| 67 | ojas-wgpu | silu_bwd | 0.88x [0.73-2.28] | 0.656 | 0.704 | noisy |
| 68 | ojas-metal | linear_ce_c1024x8192 | 0.90x [0.80-0.93] | 315.9 | 285.1 | noisy |
| 69 | ojas-metal | block_fwd_bwd | 0.92x [0.30-1.23] | 76.68 | 70.67 | noisy |
| 70 | ojas-wgpu | accumulate_grad_50304x768 | 0.92x [0.74-1.05] | 2.806 | 2.108 | noisy |
| 71 | ojas-metal | sdpa_b2h8t1024d128_bwd | 0.97x [0.83-1.44] | 5.367 | 5.893 | noisy |
| 72 | ojas-wgpu | permute_bthd_bhtd | 1.02x [0.53-1.50] | 0.545 | 0.498 | noisy |
| 73 | ojas-metal | rms_norm_bwd | 1.07x [0.59-5.56] | 2.070 | 2.217 | noisy |
| 74 | ojas-metal | linear_qkv_bwd | 1.12x [0.77-3.71] | 2.162 | 2.415 | noisy |
| 75 | ojas-wgpu | mul_bwd | 1.21x [0.32-8.09] | 1.011 | 1.133 | noisy |
| 76 | ojas-metal | muon_768x768 | 1.22x [0.88-1.48] | 3.827 | 4.667 | noisy |
| 77 | ojas-metal | muon_768x2048 | 1.23x [1.04-1.82] | 6.316 | 7.944 | noisy |
| 78 | ojas-metal | muon_2048x768 | 1.26x [0.62-2.06] | 6.411 | 8.082 | noisy |
| 79 | ojas-wgpu | vres_fwd | 1.39x [0.93-2.51] | 0.520 | 0.614 | noisy |
| 80 | ojas-wgpu | vres_bwd | 1.44x [1.26-1.93] | 0.605 | 1.035 | noisy |
| 81 | ojas-metal | sdpa_b4h12t1024d64_bwd | 1.50x [1.26-1.69] | 7.490 | 11.87 | noisy |
| 82 | ojas-metal | adamw_full | 1.55x [1.18-5.35] | 23.65 | 40.18 | noisy |
| 83 | ojas-metal | sdpa_b4h8t2048d64_bwd | 1.59x [1.21-1.75] | 18.28 | 26.80 | noisy |
| 84 | ojas-metal | rope_fwd | 1.75x [0.63-3.37] | 1.625 | 1.561 | noisy |
| 85 | ojas-wgpu | cross_entropy_fwd | 2.04x [1.96-2.40] | 6.731 | 15.73 | noisy |
| 86 | ojas-wgpu | rms_qk_norm_bwd | 2.67x [1.77-5.47] | 1.737 | 5.679 | noisy |
| 87 | ojas-metal | rms_qk_norm_bwd | 2.83x [2.15-5.54] | 1.657 | 5.679 | noisy |
| 88 | ojas-metal | rope_bwd | 3.12x [0.93-4.24] | 1.637 | 1.831 | noisy |
| 89 | ojas-wgpu | rms_norm_bwd | 3.48x [2.43-8.88] | 0.652 | 2.217 | noisy |
| 90 | ojas-metal | cross_entropy_fwd | 3.64x [2.08-3.99] | 4.358 | 15.73 | noisy |
| 91 | ojas-wgpu | rope_fwd | 3.79x [1.83-22.94] | 0.324 | 1.561 | noisy |
| 92 | ojas-wgpu | clip_grad_norm_full | 6.47x [4.61-8.92] | 20.47 | 138.7 | noisy |
| 93 | ojas-wgpu | rope_bwd | 6.49x [2.81-13.10] | 0.297 | 1.831 | noisy |
| 94 | ojas-metal | clip_grad_norm_full | 7.19x [5.64-7.91] | 18.60 | 138.7 | noisy |

### Machine load around each runtime block

| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |
| --: | :-- | :-- | :-- | :-- | --: |
| 1 | ojas-metal | before | 22:50:32 | 18.05 17.93 19.83 | 100 |
| 1 | ojas-metal | after | 22:51:09 | 16.47 17.55 19.62 | 100 |
| 1 | ojas-wgpu | before | 22:51:09 | 16.47 17.55 19.62 | 100 |
| 1 | ojas-wgpu | after | 22:52:11 | 14.95 16.87 19.21 | 100 |
| 1 | torch-mps | before | 22:52:12 | 14.95 16.87 19.21 | 100 |
| 1 | torch-mps | after | 22:52:52 | 14.36 16.47 18.95 | 100 |
| 2 | torch-mps | before | 22:52:52 | 14.36 16.47 18.95 | 100 |
| 2 | torch-mps | after | 22:53:32 | 14.21 16.17 18.73 | 100 |
| 2 | ojas-wgpu | before | 22:53:32 | 14.21 16.17 18.73 | 100 |
| 2 | ojas-wgpu | after | 22:54:36 | 18.22 16.82 18.77 | 100 |
| 2 | ojas-metal | before | 22:54:36 | 18.22 16.82 18.77 | 100 |
| 2 | ojas-metal | after | 22:55:15 | 17.71 16.89 18.71 | 100 |
| 3 | ojas-metal | before | 22:55:15 | 17.71 16.89 18.71 | 100 |
| 3 | ojas-metal | after | 22:55:54 | 18.80 17.27 18.77 | 100 |
| 3 | ojas-wgpu | before | 22:55:54 | 18.80 17.27 18.77 | 100 |
| 3 | ojas-wgpu | after | 22:57:14 | 32.29 20.81 19.92 | 100 |
| 3 | torch-mps | before | 22:57:14 | 32.29 20.81 19.92 | 100 |
| 3 | torch-mps | after | 22:58:23 | 60.66 30.91 23.75 | 100 |
| 4 | torch-mps | before | 22:58:23 | 60.66 30.91 23.75 | 100 |
| 4 | torch-mps | after | 23:00:09 | 113.12 57.16 34.74 | 100 |
| 4 | ojas-wgpu | before | 23:00:09 | 113.12 57.16 34.74 | 100 |
| 4 | ojas-wgpu | after | 23:01:44 | 116.30 73.66 43.48 | 100 |
| 4 | ojas-metal | before | 23:01:44 | 116.30 73.66 43.48 | 100 |
| 4 | ojas-metal | after | 23:02:42 | 85.23 71.79 44.79 | 100 |
| 5 | ojas-metal | before | 23:02:42 | 85.23 71.79 44.79 | 100 |
| 5 | ojas-metal | after | 23:03:42 | 97.63 77.50 48.74 | 100 |
| 5 | ojas-wgpu | before | 23:03:42 | 97.63 77.50 48.74 | 100 |
| 5 | ojas-wgpu | after | 23:05:03 | 69.73 75.04 50.49 | 100 |
| 5 | torch-mps | before | 23:05:03 | 69.73 75.04 50.49 | 100 |
| 5 | torch-mps | after | 23:05:52 | 44.93 67.33 49.02 | 100 |

### Load recorded before and after every row

| round | lane | samples | 1-min load min-max | GPU util % min / median / max |
| --: | :-- | --: | :-- | :-- |
| 1 | ojas-metal | 90 | 16.5-18.1 | 100 / 100 / 100 |
| 1 | ojas-wgpu | 90 | 14.9-16.0 | 100 / 100 / 100 |
| 1 | torch-mps | 96 | 14.3-14.9 | 100 / 100 / 100 |
| 2 | ojas-metal | 90 | 17.8-18.8 | 100 / 100 / 100 |
| 2 | ojas-wgpu | 90 | 14.2-18.6 | 100 / 100 / 100 |
| 2 | torch-mps | 96 | 14.2-14.6 | 100 / 100 / 100 |
| 3 | ojas-metal | 90 | 17.4-19.6 | 100 / 100 / 100 |
| 3 | ojas-wgpu | 90 | 17.9-32.3 | 100 / 100 / 100 |
| 3 | torch-mps | 96 | 33.9-55.3 | 100 / 100 / 100 |
| 4 | ojas-metal | 90 | 84.1-116.3 | 100 / 100 / 100 |
| 4 | ojas-wgpu | 90 | 96.9-131.5 | 100 / 100 / 100 |
| 4 | torch-mps | 96 | 63.4-115.3 | 100 / 100 / 100 |
| 5 | ojas-metal | 90 | 85.0-101.5 | 100 / 100 / 100 |
| 5 | ojas-wgpu | 90 | 69.7-97.6 | 100 / 100 / 100 |
| 5 | torch-mps | 96 | 44.2-65.6 | 100 / 100 / 100 |
