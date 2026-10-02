# Cross-run direction check

- run A: /Users/bharath/Code/research/ojas/bench/results/2026-10-01/runA
- run B: /Users/bharath/Code/research/ojas/bench/results/2026-10-01/runB
- run C: /Users/bharath/Code/research/ojas/bench/results/2026-10-01/runC

### ojas-metal vs torch-mps

| row | run A median [min-max] | run B median [min-max] | run C median [min-max] | all rounds min-max | verdict |
| :-- | :-- | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.06x [0.05-0.07] | 0.10x [0.06-0.24] | 0.08x [0.07-0.27] | 0.05-0.27 | ojas slower in all 15 rounds |
| linear_qkv_fwd | 0.46x [0.28-0.48] | 0.43x [0.37-0.65] | 0.49x [0.31-0.59] | 0.28-0.65 | ojas slower in all 15 rounds |
| linear_qkv_bwd | 0.53x [0.47-0.73] | 0.70x [0.59-0.98] | 0.73x [0.54-0.91] | 0.47-0.98 | ojas slower in all 15 rounds |
| linear_up_fwd | 0.51x [0.42-0.59] | 0.76x [0.56-0.89] | 0.58x [0.48-0.79] | 0.42-0.89 | ojas slower in all 15 rounds |
| linear_up_bwd | 0.78x [0.55-0.83] | 0.90x [0.85-1.08] | 0.77x [0.73-0.87] | 0.55-1.08 | mixed |
| linear_down_fwd | 0.61x [0.48-0.65] | 0.78x [0.70-0.92] | 0.62x [0.59-0.88] | 0.48-0.92 | ojas slower in all 15 rounds |
| linear_down_bwd | 0.66x [0.42-0.73] | 0.82x [0.68-0.96] | 0.65x [0.62-0.80] | 0.42-0.96 | ojas slower in all 15 rounds |
| linear_lmhead_fwd | 0.52x [0.42-0.54] | 0.45x [0.42-0.55] | 0.43x [0.39-0.46] | 0.39-0.55 | ojas slower in all 15 rounds |
| linear_lmhead_bwd | 0.70x [0.57-0.82] | 0.69x [0.66-0.83] | 0.62x [0.61-0.71] | 0.57-0.83 | ojas slower in all 15 rounds |
| sdpa_b4h12t1024d64_fwd | 0.55x [0.39-0.61] | 0.65x [0.42-1.98] | 0.54x [0.40-0.64] | 0.39-1.98 | mixed |
| sdpa_b4h12t1024d64_bwd | 1.24x [1.14-1.37] | 1.50x [1.34-1.64] | 1.40x [1.31-1.57] | 1.14-1.64 | ojas faster in all 15 rounds |
| sdpa_b4h8t2048d64_fwd | 0.50x [0.47-0.65] | 0.59x [0.50-0.76] | 0.59x [0.51-0.68] | 0.47-0.76 | ojas slower in all 15 rounds |
| sdpa_b4h8t2048d64_bwd | 1.38x [1.01-1.59] | 1.61x [1.40-1.77] | 1.49x [1.45-1.57] | 1.01-1.77 | ojas faster in all 15 rounds |
| sdpa_b2h8t1024d128_fwd | 0.61x [0.37-0.68] | 0.70x [0.44-1.58] | 0.73x [0.38-0.86] | 0.37-1.58 | mixed |
| sdpa_b2h8t1024d128_bwd | 0.81x [0.77-0.92] | 0.92x [0.85-1.11] | 0.89x [0.81-0.99] | 0.77-1.11 | mixed |
| rms_norm_fwd | 0.18x [0.09-0.70] | 0.13x [0.12-0.47] | 0.17x [0.10-0.28] | 0.09-0.70 | ojas slower in all 15 rounds |
| rms_norm_bwd | 0.74x [0.53-0.84] | 0.97x [0.73-1.29] | 0.72x [0.67-1.04] | 0.53-1.29 | mixed |
| rms_qk_norm_fwd | 0.11x [0.08-0.19] | 0.18x [0.12-0.33] | 0.14x [0.12-0.18] | 0.08-0.33 | ojas slower in all 15 rounds |
| rms_qk_norm_bwd | 0.10x [0.09-0.10] | 0.13x [0.10-0.15] | 0.10x [0.10-0.15] | 0.09-0.15 | ojas slower in all 15 rounds |
| rope_fwd | 0.85x [0.68-1.51] | 0.81x [0.74-1.20] | 1.70x [0.72-2.64] | 0.68-2.64 | mixed |
| rope_bwd | 1.81x [0.87-2.38] | 1.01x [0.99-1.79] | 2.20x [0.97-3.06] | 0.87-3.06 | mixed |
| silu_fwd | 0.20x [0.13-0.28] | 0.21x [0.17-0.44] | 0.24x [0.15-0.50] | 0.13-0.50 | ojas slower in all 15 rounds |
| silu_bwd | 0.23x [0.18-0.30] | 0.24x [0.23-0.56] | 0.22x [0.21-0.49] | 0.18-0.56 | ojas slower in all 15 rounds |
| mul_fwd | 0.21x [0.21-0.24] | 0.24x [0.20-0.76] | 0.20x [0.17-0.58] | 0.17-0.76 | ojas slower in all 15 rounds |
| mul_bwd | 0.35x [0.26-0.38] | 0.48x [0.22-1.67] | 0.30x [0.25-0.71] | 0.22-1.67 | mixed |
| residual_add_fwd | 0.18x [0.13-0.25] | 0.23x [0.20-1.10] | 0.22x [0.18-0.28] | 0.13-1.10 | mixed |
| residual_add_bwd | 0.01x [0.01-0.01] | 0.01x [0.01-0.02] | 0.01x [0.00-0.04] | 0.00-0.04 | ojas slower in all 15 rounds |
| gate_fwd | 0.36x [0.26-0.98] | 0.29x [0.28-0.63] | 0.52x [0.23-0.91] | 0.23-0.98 | ojas slower in all 15 rounds |
| gate_bwd | 0.33x [0.26-0.64] | 0.43x [0.29-0.60] | 0.38x [0.26-0.64] | 0.26-0.64 | ojas slower in all 15 rounds |
| vres_fwd | 0.41x [0.23-0.57] | 0.32x [0.28-0.55] | 0.31x [0.23-1.06] | 0.23-1.06 | mixed |
| vres_bwd | 0.64x [0.32-0.79] | 0.49x [0.37-0.66] | 0.71x [0.35-0.94] | 0.32-0.94 | ojas slower in all 15 rounds |
| permute_bthd_bhtd | 0.72x [0.23-1.27] | 0.28x [0.25-0.86] | 0.31x [0.20-0.68] | 0.20-1.27 | mixed |
| cross_entropy_fwd | 0.97x [0.76-1.06] | 1.03x [0.95-1.93] | 0.96x [0.88-1.47] | 0.76-1.93 | mixed |
| cross_entropy_bwd | 0.35x [0.33-0.38] | 0.38x [0.35-0.56] | 0.37x [0.37-0.53] | 0.33-0.56 | ojas slower in all 15 rounds |
| clip_grad_norm_full | 4.58x [4.46-4.75] | 7.16x [4.57-9.36] | 4.36x [4.29-9.25] | 4.29-9.36 | ojas faster in all 15 rounds |
| adamw_full | 0.09x [0.08-0.10] | 0.09x [0.07-0.11] | 0.10x [0.07-0.11] | 0.07-0.11 | ojas slower in all 15 rounds |
| muon_768x768 | 0.71x [0.62-0.76] | 0.78x [0.68-1.26] | 0.74x [0.63-0.97] | 0.62-1.26 | mixed |
| muon_768x768 vs muon_768x768_bf16 | 0.44x [0.41-0.47] | 0.46x [0.43-0.51] | 0.46x [0.32-0.58] | 0.32-0.58 | ojas slower in all 15 rounds |
| muon_2048x768 | 0.75x [0.67-0.81] | 0.88x [0.77-1.13] | 0.76x [0.68-1.08] | 0.67-1.13 | mixed |
| muon_2048x768 vs muon_2048x768_bf16 | 0.45x [0.43-0.48] | 0.50x [0.48-0.76] | 0.47x [0.43-0.64] | 0.43-0.76 | ojas slower in all 15 rounds |
| muon_768x2048 | 0.77x [0.40-0.80] | 0.94x [0.78-1.17] | 0.79x [0.75-0.91] | 0.40-1.17 | mixed |
| muon_768x2048 vs muon_768x2048_bf16 | 0.45x [0.23-0.49] | 0.47x [0.45-0.65] | 0.45x [0.41-0.55] | 0.23-0.65 | ojas slower in all 15 rounds |
| block_fwd | 0.24x [0.24-0.27] | 0.28x [0.23-0.37] | 0.26x [0.24-0.30] | 0.23-0.37 | ojas slower in all 15 rounds |
| block_fwd_bwd | 0.37x [0.34-0.38] | 0.43x [0.37-0.60] | 0.38x [0.34-0.50] | 0.34-0.60 | ojas slower in all 15 rounds |

### ojas-wgpu vs torch-mps

| row | run A median [min-max] | run B median [min-max] | run C median [min-max] | all rounds min-max | verdict |
| :-- | :-- | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.48x [0.21-0.50] | 0.42x [0.35-0.59] | 0.43x [0.29-0.62] | 0.21-0.62 | ojas slower in all 15 rounds |
| linear_qkv_fwd | 0.31x [0.30-0.33] | 0.36x [0.30-0.37] | 0.36x [0.32-0.37] | 0.30-0.37 | ojas slower in all 15 rounds |
| linear_qkv_bwd | 0.38x [0.32-0.40] | 0.33x [0.29-0.41] | 0.35x [0.31-0.41] | 0.29-0.41 | ojas slower in all 15 rounds |
| linear_up_fwd | 0.32x [0.27-0.35] | 0.32x [0.28-0.35] | 0.33x [0.30-0.36] | 0.27-0.36 | ojas slower in all 15 rounds |
| linear_up_bwd | 0.31x [0.29-0.34] | 0.32x [0.30-0.33] | 0.32x [0.27-0.35] | 0.27-0.35 | ojas slower in all 15 rounds |
| linear_down_fwd | 0.29x [0.24-0.31] | 0.30x [0.28-0.33] | 0.31x [0.27-0.34] | 0.24-0.34 | ojas slower in all 15 rounds |
| linear_down_bwd | 0.32x [0.28-0.34] | 0.32x [0.28-0.34] | 0.34x [0.29-0.36] | 0.28-0.36 | ojas slower in all 15 rounds |
| linear_lmhead_fwd | 0.31x [0.28-0.36] | 0.32x [0.27-0.32] | 0.30x [0.29-0.33] | 0.27-0.36 | ojas slower in all 15 rounds |
| linear_lmhead_bwd | 0.37x [0.34-0.43] | 0.36x [0.32-0.38] | 0.35x [0.33-0.36] | 0.32-0.43 | ojas slower in all 15 rounds |
| sdpa_b4h12t1024d64_fwd | 0.02x [0.02-0.02] | 0.02x [0.02-0.06] | 0.02x [0.02-0.02] | 0.02-0.06 | ojas slower in all 15 rounds |
| sdpa_b4h12t1024d64_bwd | 0.04x [0.04-0.04] | 0.04x [0.04-0.04] | 0.04x [0.04-0.04] | 0.04-0.04 | ojas slower in all 15 rounds |
| sdpa_b4h8t2048d64_fwd | 0.02x [0.02-0.02] | 0.02x [0.02-0.02] | 0.02x [0.02-0.02] | 0.02-0.02 | ojas slower in all 15 rounds |
| sdpa_b4h8t2048d64_bwd | 0.04x [0.04-0.04] | 0.04x [0.03-0.04] | 0.04x [0.03-0.04] | 0.03-0.04 | ojas slower in all 15 rounds |
| sdpa_b2h8t1024d128_fwd | 0.01x [0.01-0.01] | 0.02x [0.01-0.03] | 0.01x [0.01-0.01] | 0.01-0.03 | ojas slower in all 15 rounds |
| sdpa_b2h8t1024d128_bwd | 0.02x [0.02-0.02] | 0.02x [0.02-0.02] | 0.02x [0.02-0.02] | 0.02-0.02 | ojas slower in all 15 rounds |
| rms_norm_fwd | 0.61x [0.46-1.07] | 0.70x [0.47-1.08] | 0.59x [0.34-0.66] | 0.34-1.08 | mixed |
| rms_norm_bwd | 3.42x [1.67-3.91] | 3.53x [2.32-5.76] | 3.10x [2.40-4.10] | 1.67-5.76 | ojas faster in all 15 rounds |
| rms_qk_norm_fwd | 0.17x [0.14-0.22] | 0.19x [0.10-0.58] | 0.18x [0.16-0.33] | 0.10-0.58 | ojas slower in all 15 rounds |
| rms_qk_norm_bwd | 0.95x [0.88-1.00] | 1.08x [0.67-1.28] | 1.01x [0.86-1.02] | 0.67-1.28 | mixed |
| rope_fwd | 3.67x [3.15-4.45] | 3.80x [3.51-4.78] | 4.35x [2.78-5.32] | 2.78-5.32 | ojas faster in all 15 rounds |
| rope_bwd | 4.73x [3.88-6.78] | 5.23x [4.41-6.49] | 5.24x [3.57-6.46] | 3.57-6.78 | ojas faster in all 15 rounds |
| silu_fwd | 0.75x [0.55-0.79] | 0.70x [0.67-1.01] | 0.72x [0.32-0.84] | 0.32-1.01 | mixed |
| silu_bwd | 0.86x [0.47-1.00] | 0.78x [0.65-1.28] | 0.75x [0.54-0.98] | 0.47-1.28 | mixed |
| mul_fwd | 0.74x [0.56-0.78] | 0.77x [0.59-2.47] | 0.82x [0.56-0.88] | 0.56-2.47 | mixed |
| mul_bwd | 1.06x [0.65-1.16] | 0.95x [0.77-4.36] | 1.00x [0.86-1.11] | 0.65-4.36 | mixed |
| residual_add_fwd | 0.72x [0.43-0.83] | 0.66x [0.58-7.17] | 0.81x [0.41-1.12] | 0.41-7.17 | mixed |
| residual_add_bwd | 0.02x [0.01-0.03] | 0.02x [0.01-0.09] | 0.01x [0.01-0.04] | 0.01-0.09 | ojas slower in all 15 rounds |
| gate_fwd | 0.74x [0.67-1.62] | 0.72x [0.62-1.61] | 0.75x [0.50-0.89] | 0.50-1.62 | mixed |
| gate_bwd | 0.63x [0.61-1.21] | 0.59x [0.40-0.81] | 0.57x [0.49-0.77] | 0.40-1.21 | mixed |
| vres_fwd | 1.32x [1.14-1.48] | 1.20x [1.09-1.91] | 1.16x [1.01-2.07] | 1.01-2.07 | ojas faster in all 15 rounds |
| vres_bwd | 1.67x [1.56-2.17] | 1.46x [1.16-2.10] | 1.54x [1.25-1.98] | 1.16-2.17 | ojas faster in all 15 rounds |
| permute_bthd_bhtd | 0.89x [0.60-1.53] | 0.92x [0.77-1.19] | 0.77x [0.59-0.91] | 0.59-1.53 | mixed |
| cross_entropy_fwd | 2.79x [2.14-2.99] | 2.61x [1.93-3.21] | 2.54x [2.00-2.92] | 1.93-3.21 | ojas faster in all 15 rounds |
| cross_entropy_bwd | 0.53x [0.50-0.54] | 0.58x [0.54-0.87] | 0.55x [0.48-0.60] | 0.48-0.87 | ojas slower in all 15 rounds |
| clip_grad_norm_full | 6.36x [6.14-6.38] | 7.95x [6.86-9.10] | 6.61x [5.81-7.95] | 5.81-9.10 | ojas faster in all 15 rounds |
| adamw_full | 0.70x [0.69-0.75] | 0.75x [0.56-1.06] | 0.75x [0.55-0.80] | 0.55-1.06 | mixed |
| muon_768x768 | 0.44x [0.42-0.47] | 0.45x [0.40-0.68] | 0.46x [0.41-0.47] | 0.40-0.68 | ojas slower in all 15 rounds |
| muon_768x768 vs muon_768x768_bf16 | 0.27x [0.27-0.28] | 0.27x [0.22-0.32] | 0.28x [0.21-0.30] | 0.21-0.32 | ojas slower in all 15 rounds |
| muon_2048x768 | 0.41x [0.41-0.44] | 0.43x [0.41-0.54] | 0.44x [0.41-0.44] | 0.41-0.54 | ojas slower in all 15 rounds |
| muon_2048x768 vs muon_2048x768_bf16 | 0.25x [0.24-0.28] | 0.26x [0.25-0.36] | 0.27x [0.23-0.27] | 0.23-0.36 | ojas slower in all 15 rounds |
| muon_768x2048 | 0.42x [0.41-0.46] | 0.44x [0.43-0.57] | 0.45x [0.43-0.46] | 0.41-0.57 | ojas slower in all 15 rounds |
| muon_768x2048 vs muon_768x2048_bf16 | 0.25x [0.24-0.26] | 0.25x [0.22-0.31] | 0.26x [0.23-0.27] | 0.22-0.31 | ojas slower in all 15 rounds |
| block_fwd | 0.14x [0.13-0.14] | 0.14x [0.13-0.18] | 0.14x [0.13-0.14] | 0.13-0.18 | ojas slower in all 15 rounds |
| block_fwd_bwd | 0.18x [0.17-0.18] | 0.17x [0.15-0.21] | 0.17x [0.15-0.17] | 0.15-0.21 | ojas slower in all 15 rounds |

