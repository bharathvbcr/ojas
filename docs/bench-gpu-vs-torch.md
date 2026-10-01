# GPU Backends versus PyTorch MPS

Paired A/B of ojas's two GPU backends, `MetalBackend` (`ojas-metal`) and
`WgpuBackend` (`ojas-wgpu`), against PyTorch 2.13 MPS, at nanolab's default
shapes: d 768, 12 heads of 64, SwiGLU hidden 2048, vocabulary 50304, B 4,
T 1024 (4096 rows). Each kernel is timed forward and backward where it has a
backward. One composed nanolab block is timed forward, and forward plus
backward. Everything is f32 on both sides.

The harness, the commands and the row definitions are in
[`bench/README.md`](../bench/README.md). The raw data of every run quoted here
is in `bench/results/2026-10-01/`.

> [!IMPORTANT]
> **The GPU was shared with other work for most of these runs.** `ioreg`
> "Device Utilization %" was 84–100% before nearly every runtime block of runs
> A and C. At 16:48 `ps` showed an ollama `llama-server` at 143% CPU and
> several Chrome renderer processes at about 100% CPU each. That observation
> is from one moment, not the whole run. The protocol's 10% spread rule
> therefore flags almost every row as "noisy - not quoted": 5 of 88 rows pass
> in run A, 0 in run C. This document does not quote a speedup for a
> flagged row as a precise number.
>
> Read the evidence in two tiers:
> - **Direction, verified.** The verdict holds where every per-round ratio
>   in all three runs (15 rounds, each pairing the two runtimes minutes
>   apart) lies on one side of 1. Pairing cancels contention that hits both
>   sides alike, so many flagged rows still have tight ratio ranges. One
>   example is wgpu linear at 0.27–0.37x across 15 rounds.
> - **Magnitude, unverified beyond the stated range.** The median ratio
>   quoted is run A's. The 15-round min-max beside it is the honest
>   uncertainty.
>
> Metal is likely hurt more by contention than torch or wgpu, because every
> Metal op waits on the device. The evidence:
> - Its per-op floor has a 0.13–0.19 ms minimum but a 1.5–1.8 ms median.
> - One unpaired smoke run after the three runs measured Metal
>   `block_fwd_bwd` at 131 ms median, against run A's 230 ms. That run is not
>   quoted, and the GPU load at the time was not sampled.
>
> Treat Metal's contended ratios as pessimistic (inferred).

## Toolchain and source state (verified, `bench/results/2026-10-01/run*/env.txt`)

| item | value |
| :-- | :-- |
| machine | Apple M5 Pro, 64 GiB, macOS 27.0.1 (26A434) |
| rustc / cargo | 1.98.0 (88d9e12ae 2026-08-18) / 1.98.0 (797e8a9bc 2026-08-05) |
| Python / torch | 3.14.7 (`/opt/homebrew/opt/python@3.14/bin/python3.14`) / 2.13.0, MPS |
| ojas | `dab2a12` **plus an uncommitted working tree**. The diff of ojas-core, ojas-metal, ojas-wgpu, ojas-kernels and ojas-device hashes to `c226ffba076d` in all three runs; other sessions were editing this checkout |
| tessl (ojas-metal's GEMMs) | `cf65d9d` plus 20 dirty files |
| wgpu | 30.0.1. The adapter is "Apple M5 Pro" with HAL **Metal**, recorded by the binary's `_device` line. So on this machine the portable backend runs through wgpu's Metal HAL. The binding cap is 4,294,967,292 bytes, so the 824 MB logits fit |
| build | release, `CARGO_TARGET_DIR=target-lane-bench`. The runner builds before every run; the binary mtimes are in env.txt |
| nanolab | `/Users/bharath/Code/research/MLSystemsLab/nanolab`, real `Block`, `GPT(Config())` 123,699,612 parameters in 170 tensors. torch_rows.py asserts the optimizer rows' shape list against it |

## Runs

| run | rounds x iters (warmup 5) | GPU before blocks | flagged rows (of 88) |
| :-- | :-- | :-- | --: |
| A, `run5` (primary: fewest flagged) | 5 x 20 | 93–100%, except 70% before round-1 wgpu | 83 |
| B, `run5b` | 5 x 30 | 84–95% in rounds 1–2, 0–50% in rounds 3–5 | 88 |
| C, `run5c` | 5 x 30 | 0–57% in round 1, 85–94% after | 88 |

The primary-run rule was fixed before run C finished: take the run with the
fewest flagged rows. The full before/after load tables are at the end of each
`summary.md`. `ioreg` is an instantaneous sample. In run B round 1 it went
from 0 to 94 within 36 s, so it bounds the contention but does not average it.

## Parity (verified)

Every row on both backends passed its gate, max |ojas - torch| / max |torch|
per output (1e-3; 1e-2 for Muon; 0 for permute), in every round of all three
runs. No row was refused. Both binaries' generator checks passed. The worst
normalized error was 1.3e-5, on `linear_lmhead_bwd`. The largest absolute
error was 1.3e-3, on Metal `rms_qk_norm_bwd`'s weight gradient (a sum over
49,152 rows; 4.0e-6 normalized).

Every linear **forward** matched torch MPS **bit for bit** (error 0), at all
four shapes, on both backends. The comparison is not vacuous. The same
harness reports 1.57e-4 on `linear_lmhead_bwd` (the TN `grad_w` product) and
nonzero errors on 30 other rows. That the GEMMs accumulate in the same
k-ascending FMA order is inferred, not checked.

The block rows compare the output and seven gradients (x, q_proj, ffn.down,
gate.weight, vr_lambda, norm1, v0) against nanolab's real `Block` plus torch
autograd. They pass at 2.5e-6 (Metal) and 2.3e-6 (wgpu) normalized. That is
the check on the hand-composed backward.

## Results: ojas MetalBackend vs torch MPS

ratio = torch median / ojas median, paired per round; > 1 means ojas is
faster. Times are run A, min over rounds / median of per-round medians, in ms.

| row | ojas min / median ms | torch min / median ms | ratio, run A median [min-max] | spread ojas / torch | run A flag | parity max abs (rel) | direction over 3 runs |
| :-- | --: | --: | :-- | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.191 / 1.756 | 0.082 / 0.107 | 0.06x [0.05-0.07] | 4% / 26% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_qkv_fwd | 1.153 / 1.999 | 0.802 / 0.918 | 0.46x [0.28-0.48] | 88% / 17% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_qkv_bwd | 2.499 / 3.785 | 1.720 / 2.112 | 0.53x [0.47-0.73] | 54% / 12% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_up_fwd | 2.709 / 4.127 | 1.943 / 2.152 | 0.51x [0.42-0.59] | 42% / 5% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_up_bwd | 4.899 / 5.630 | 4.176 / 4.497 | 0.78x [0.55-0.83] | 59% / 7% | noisy | 0.00e+00 (0.0e+00) | mixed (14/15 slower) |
| linear_down_fwd | 2.600 / 3.609 | 2.013 / 2.201 | 0.61x [0.48-0.65] | 44% / 6% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_down_bwd | 4.592 / 6.541 | 4.008 / 4.458 | 0.66x [0.42-0.73] | 76% / 8% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_lmhead_fwd | 109.4 / 131.9 | 54.41 / 60.84 | 0.52x [0.42-0.54] | 20% / 26% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_lmhead_bwd | 131.0 / 174.3 | 105.5 / 120.4 | 0.70x [0.57-0.82] | 31% / 39% | noisy | 1.57e-04 (1.3e-05) | slower, 15/15 |
| sdpa_b4h12t1024d64_fwd | 2.045 / 3.081 | 1.367 / 1.600 | 0.55x [0.39-0.61] | 68% / 19% | noisy | 2.24e-07 (2.2e-07) | mixed (14/15 slower) |
| sdpa_b4h12t1024d64_bwd | 5.830 / 8.111 | 9.474 / 10.30 | 1.24x [1.14-1.37] | 29% / 13% | noisy | 1.10e-06 (2.1e-06) | **faster, 15/15** |
| sdpa_b4h8t2048d64_fwd | 4.775 / 6.909 | 3.225 / 3.542 | 0.50x [0.47-0.65] | 23% / 16% | noisy | 1.79e-07 (1.8e-07) | slower, 15/15 |
| sdpa_b4h8t2048d64_bwd | 17.30 / 19.41 | 24.21 / 25.39 | 1.38x [1.01-1.59] | 35% / 25% | noisy | 1.31e-06 (2.5e-06) | **faster, 15/15** |
| sdpa_b2h8t1024d128_fwd | 1.333 / 2.029 | 1.074 / 1.202 | 0.61x [0.37-0.68] | 96% / 17% | noisy | 2.09e-07 (2.1e-07) | mixed (14/15 slower) |
| sdpa_b2h8t1024d128_bwd | 5.210 / 6.417 | 4.395 / 5.318 | 0.81x [0.77-0.92] | 22% / 40% | noisy | 1.10e-06 (2.6e-06) | mixed (14/15 slower) |
| rms_norm_fwd | 0.478 / 1.654 | 0.183 / 0.258 | 0.18x [0.09-0.70] | 142% / 226% | noisy | 3.58e-07 (2.0e-07) | slower, 15/15 |
| rms_norm_bwd | 1.939 / 3.389 | 2.037 / 2.515 | 0.74x [0.53-0.84] | 48% / 49% | noisy | 2.14e-04 (1.5e-06) | mixed (13/15 slower) |
| rms_qk_norm_fwd | 2.001 / 4.736 | 0.304 / 0.534 | 0.11x [0.08-0.19] | 49% / 63% | noisy | 3.58e-07 (1.7e-07) | slower, 15/15 |
| rms_qk_norm_bwd | 50.48 / 53.15 | 4.369 / 5.276 | 0.10x [0.09-0.10] | 2% / 8% | **quotable** | 1.30e-03 (4.0e-06) | slower, 15/15 |
| rope_fwd | 0.586 / 1.816 | 1.220 / 1.513 | 0.85x [0.68-1.51] | 124% / 19% | noisy | 0.00e+00 (0.0e+00) | mixed (9/15 slower) |
| rope_bwd | 0.440 / 1.182 | 1.722 / 2.081 | 1.81x [0.87-2.38] | 154% / 26% | noisy | 0.00e+00 (0.0e+00) | mixed (4/15 slower) |
| silu_fwd | 0.927 / 2.267 | 0.361 / 0.466 | 0.20x [0.13-0.28] | 96% / 51% | noisy | 2.38e-07 (6.1e-08) | slower, 15/15 |
| silu_bwd | 1.533 / 3.191 | 0.506 / 0.691 | 0.23x [0.18-0.30] | 38% / 62% | noisy | 1.19e-07 (1.1e-07) | slower, 15/15 |
| mul_fwd | 1.291 / 2.899 | 0.496 / 0.615 | 0.21x [0.21-0.24] | 26% / 31% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| mul_bwd | 2.328 / 4.131 | 0.866 / 1.387 | 0.35x [0.26-0.38] | 26% / 49% | noisy | 0.00e+00 (0.0e+00) | mixed (14/15 slower) |
| residual_add_fwd | 0.641 / 1.559 | 0.223 / 0.345 | 0.18x [0.13-0.25] | 78% / 93% | noisy | 0.00e+00 (0.0e+00) | mixed (14/15 slower) |
| residual_add_bwd | 0.593 / 1.878 | 0.012 / 0.016 | 0.01x [0.01-0.01] | 75% / 135% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| gate_fwd | 0.646 / 1.773 | 0.476 / 0.681 | 0.36x [0.26-0.98] | 75% / 124% | noisy | 8.94e-08 (1.0e-07) | slower, 15/15 |
| gate_bwd | 2.928 / 4.069 | 0.957 / 1.350 | 0.33x [0.26-0.64] | 32% / 104% | noisy | 2.25e-04 (2.3e-06) | slower, 15/15 |
| vres_fwd | 0.590 / 1.645 | 0.481 / 0.650 | 0.41x [0.23-0.57] | 121% / 23% | noisy | 0.00e+00 (0.0e+00) | mixed (14/15 slower) |
| vres_bwd | 0.946 / 1.943 | 0.840 / 1.192 | 0.64x [0.32-0.79] | 89% / 38% | noisy | 3.05e-05 (2.3e-07) | slower, 15/15 |
| permute_bthd_bhtd | 0.391 / 0.771 | 0.407 / 0.547 | 0.72x [0.23-1.27] | 214% / 79% | noisy | 0.00e+00 (0.0e+00) | mixed (14/15 slower) |
| cross_entropy_fwd | 13.31 / 20.94 | 14.75 / 19.60 | 0.97x [0.76-1.06] | 13% / 35% | noisy | 0.00e+00 (0.0e+00) | mixed (7/15 slower) |
| cross_entropy_bwd | 78.49 / 96.68 | 28.31 / 34.30 | 0.35x [0.33-0.38] | 7% / 14% | noisy | 6.39e-14 (2.6e-10) | slower, 15/15 |
| clip_grad_norm_full | 16.71 / 33.67 | 137.4 / 154.2 | 4.58x [4.46-4.75] | 7% / 1% | **quotable** | 3.05e-05 (3.0e-07) | **faster, 15/15** |
| adamw_full | 515.8 / 598.1 | 44.33 / 53.52 | 0.09x [0.08-0.10] | 7% / 11% | noisy | 3.73e-09 (1.2e-07) | slower, 15/15 |
| muon_768x768 (torch fp32 NS5) | 4.565 / 6.507 | 4.005 / 4.759 | 0.71x [0.62-0.76] | 34% / 19% | noisy | 0.00e+00 (0.0e+00) | mixed (14/15 slower) |
| muon_768x768 vs torch bf16 NS5 | 4.565 / 6.507 | 2.236 / 2.943 | 0.44x [0.41-0.47] | 34% / 27% | noisy | (timing only) | slower, 15/15 |
| muon_2048x768 (torch fp32 NS5) | 9.782 / 12.01 | 8.145 / 9.158 | 0.75x [0.67-0.81] | 24% / 12% | noisy | 1.68e-08 (5.0e-07) | mixed (13/15 slower) |
| muon_2048x768 vs torch bf16 NS5 | 9.782 / 12.01 | 4.831 / 5.721 | 0.45x [0.43-0.48] | 24% / 19% | noisy | (timing only) | slower, 15/15 |
| muon_768x2048 (torch fp32 NS5) | 9.908 / 11.73 | 8.615 / 9.080 | 0.77x [0.40-0.80] | 114% / 11% | noisy | 9.31e-09 (2.9e-07) | mixed (13/15 slower) |
| muon_768x2048 vs torch bf16 NS5 | 9.908 / 11.73 | 4.598 / 5.343 | 0.45x [0.23-0.49] | 114% / 10% | noisy | (timing only) | slower, 15/15 |
| block_fwd | 48.08 / 67.25 | 15.74 / 16.92 | 0.24x [0.24-0.27] | 24% / 9% | noisy | 2.38e-07 (1.9e-07) | slower, 15/15 |
| block_fwd_bwd | 179.1 / 230.5 | 77.08 / 83.64 | 0.37x [0.34-0.38] | 11% / 7% | noisy | 9.06e-06 (2.5e-06) | slower, 15/15 |

## Results: ojas WgpuBackend (Metal HAL) vs torch MPS

| row | ojas min / median ms | torch min / median ms | ratio, run A median [min-max] | spread ojas / torch | run A flag | parity max abs (rel) | direction over 3 runs |
| :-- | --: | --: | :-- | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.170 / 0.244 | 0.082 / 0.107 | 0.48x [0.21-0.50] | 147% / 26% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_qkv_fwd | 2.370 / 2.787 | 0.802 / 0.918 | 0.31x [0.30-0.33] | 14% / 17% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_qkv_bwd | 5.072 / 5.620 | 1.720 / 2.112 | 0.38x [0.32-0.40] | 14% / 12% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_up_fwd | 6.080 / 6.697 | 1.943 / 2.152 | 0.32x [0.27-0.35] | 21% / 5% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_up_bwd | 12.97 / 14.04 | 4.176 / 4.497 | 0.31x [0.29-0.34] | 15% / 7% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_down_fwd | 6.693 / 7.456 | 2.013 / 2.201 | 0.29x [0.24-0.31] | 31% / 6% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_down_bwd | 13.07 / 13.84 | 4.008 / 4.458 | 0.32x [0.28-0.34] | 13% / 8% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_lmhead_fwd | 186.1 / 194.7 | 54.41 / 60.84 | 0.31x [0.28-0.36] | 5% / 26% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_lmhead_bwd | 307.1 / 323.7 | 105.5 / 120.4 | 0.37x [0.34-0.43] | 10% / 39% | noisy | 1.57e-04 (1.3e-05) | slower, 15/15 |
| sdpa_b4h12t1024d64_fwd | 72.67 / 77.05 | 1.367 / 1.600 | 0.02x [0.02-0.02] | 12% / 19% | noisy | 2.24e-07 (2.2e-07) | slower, 15/15 |
| sdpa_b4h12t1024d64_bwd | 270.8 / 277.8 | 9.474 / 10.30 | 0.04x [0.04-0.04] | 3% / 13% | noisy | 1.25e-06 (2.4e-06) | slower, 15/15 |
| sdpa_b4h8t2048d64_fwd | 183.2 / 191.6 | 3.225 / 3.542 | 0.02x [0.02-0.02] | 6% / 16% | noisy | 1.79e-07 (1.8e-07) | slower, 15/15 |
| sdpa_b4h8t2048d64_bwd | 694.0 / 705.7 | 24.21 / 25.39 | 0.04x [0.04-0.04] | 7% / 25% | noisy | 1.37e-06 (2.6e-06) | slower, 15/15 |
| sdpa_b2h8t1024d128_fwd | 68.78 / 91.63 | 1.074 / 1.202 | 0.01x [0.01-0.01] | 6% / 17% | noisy | 2.09e-07 (2.1e-07) | slower, 15/15 |
| sdpa_b2h8t1024d128_bwd | 267.7 / 280.7 | 4.395 / 5.318 | 0.02x [0.02-0.02] | 7% / 40% | noisy | 1.07e-06 (2.3e-06) | slower, 15/15 |
| rms_norm_fwd | 0.304 / 0.424 | 0.183 / 0.258 | 0.61x [0.46-1.07] | 321% / 226% | noisy | 2.38e-07 (1.3e-07) | mixed (13/15 slower) |
| rms_norm_bwd | 0.584 / 0.896 | 2.037 / 2.515 | 3.42x [1.67-3.91] | 132% / 49% | noisy | 3.34e-05 (2.6e-07) | **faster, 15/15** |
| rms_qk_norm_fwd | 1.948 / 2.624 | 0.304 / 0.534 | 0.17x [0.14-0.22] | 58% / 63% | noisy | 2.38e-07 (1.2e-07) | slower, 15/15 |
| rms_qk_norm_bwd | 4.744 / 5.421 | 4.369 / 5.276 | 0.95x [0.88-1.00] | 7% / 8% | quotable | 4.58e-04 (1.3e-06) | mixed (8/15 slower) |
| rope_fwd | 0.243 / 0.470 | 1.220 / 1.513 | 3.67x [3.15-4.45] | 47% / 19% | noisy | 1.19e-07 (6.2e-08) | **faster, 15/15** |
| rope_bwd | 0.250 / 0.423 | 1.722 / 2.081 | 4.73x [3.88-6.78] | 66% / 26% | noisy | 1.19e-07 (6.1e-08) | **faster, 15/15** |
| silu_fwd | 0.488 / 0.728 | 0.361 / 0.466 | 0.75x [0.55-0.79] | 45% / 51% | noisy | 7.15e-07 (1.8e-07) | mixed (14/15 slower) |
| silu_bwd | 0.691 / 1.004 | 0.506 / 0.691 | 0.86x [0.47-1.00] | 60% / 62% | noisy | 4.77e-07 (4.3e-07) | mixed (13/15 slower) |
| mul_fwd | 0.598 / 0.862 | 0.496 / 0.615 | 0.74x [0.56-0.78] | 43% / 31% | noisy | 0.00e+00 (0.0e+00) | mixed (14/15 slower) |
| mul_bwd | 0.906 / 1.263 | 0.866 / 1.387 | 1.06x [0.65-1.16] | 34% / 49% | noisy | 0.00e+00 (0.0e+00) | mixed (8/15 slower) |
| residual_add_fwd | 0.328 / 0.513 | 0.223 / 0.345 | 0.72x [0.43-0.83] | 76% / 93% | noisy | 0.00e+00 (0.0e+00) | mixed (13/15 slower) |
| residual_add_bwd | 0.628 / 0.898 | 0.012 / 0.016 | 0.02x [0.01-0.03] | 26% / 135% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| gate_fwd | 0.635 / 0.943 | 0.476 / 0.681 | 0.74x [0.67-1.62] | 35% / 124% | noisy | 1.19e-07 (1.4e-07) | mixed (13/15 slower) |
| gate_bwd | 1.813 / 2.111 | 0.957 / 1.350 | 0.63x [0.61-1.21] | 6% / 104% | noisy | 2.21e-04 (2.3e-06) | mixed (14/15 slower) |
| vres_fwd | 0.358 / 0.458 | 0.481 / 0.650 | 1.32x [1.14-1.48] | 17% / 23% | noisy | 5.96e-08 (6.0e-08) | **faster, 15/15** |
| vres_bwd | 0.573 / 0.720 | 0.840 / 1.192 | 1.67x [1.56-2.17] | 20% / 38% | noisy | 3.05e-05 (2.3e-07) | **faster, 15/15** |
| permute_bthd_bhtd | 0.411 / 0.628 | 0.407 / 0.547 | 0.89x [0.60-1.53] | 61% / 79% | noisy | 0.00e+00 (0.0e+00) | mixed (11/15 slower) |
| cross_entropy_fwd | 6.317 / 7.139 | 14.75 / 19.60 | 2.79x [2.14-2.99] | 18% / 35% | noisy | 0.00e+00 (0.0e+00) | **faster, 15/15** |
| cross_entropy_bwd | 57.75 / 63.71 | 28.31 / 34.30 | 0.53x [0.50-0.54] | 12% / 14% | noisy | 6.39e-14 (2.6e-10) | slower, 15/15 |
| clip_grad_norm_full | 21.34 / 24.18 | 137.4 / 154.2 | 6.36x [6.14-6.38] | 5% / 1% | **quotable** | 3.81e-05 (3.8e-07) | **faster, 15/15** |
| adamw_full | 66.35 / 75.15 | 44.33 / 53.52 | 0.70x [0.69-0.75] | 13% / 11% | noisy | 3.73e-09 (1.2e-07) | mixed (14/15 slower) |
| muon_768x768 (torch fp32 NS5) | 9.453 / 10.58 | 4.005 / 4.759 | 0.44x [0.42-0.47] | 22% / 19% | noisy | 1.49e-08 (4.5e-07) | slower, 15/15 |
| muon_768x768 vs torch bf16 NS5 | 9.453 / 10.58 | 2.236 / 2.943 | 0.27x [0.27-0.28] | 22% / 27% | noisy | (timing only) | slower, 15/15 |
| muon_2048x768 (torch fp32 NS5) | 20.14 / 22.32 | 8.145 / 9.158 | 0.41x [0.41-0.44] | 8% / 12% | noisy | 1.68e-08 (5.0e-07) | slower, 15/15 |
| muon_2048x768 vs torch bf16 NS5 | 20.14 / 22.32 | 4.831 / 5.721 | 0.25x [0.24-0.28] | 8% / 19% | noisy | (timing only) | slower, 15/15 |
| muon_768x2048 (torch fp32 NS5) | 19.77 / 21.64 | 8.615 / 9.080 | 0.42x [0.41-0.46] | 14% / 11% | noisy | 9.31e-09 (2.9e-07) | slower, 15/15 |
| muon_768x2048 vs torch bf16 NS5 | 19.77 / 21.64 | 4.598 / 5.343 | 0.25x [0.24-0.26] | 14% / 10% | noisy | (timing only) | slower, 15/15 |
| block_fwd | 115.3 / 120.0 | 15.74 / 16.92 | 0.14x [0.13-0.14] | 16% / 9% | noisy | 2.38e-07 (1.9e-07) | slower, 15/15 |
| block_fwd_bwd | 465.8 / 475.3 | 77.08 / 83.64 | 0.18x [0.17-0.18] | 3% / 7% | **quotable** | 8.58e-06 (2.3e-06) | slower, 15/15 |

Quotable rows (both spreads at most 10% in run A), verified:
- Metal `clip_grad_norm_full`: 4.58x faster than torch.
- wgpu `clip_grad_norm_full`: 6.36x faster.
- Metal `rms_qk_norm_bwd`: 0.10x, about 10 times slower.
- wgpu `block_fwd_bwd`: 0.18x, about 5.6 times slower.
- wgpu `rms_qk_norm_bwd`: 0.95x. Within noise of parity; its direction is mixed across runs.

## Where ojas is slower than torch (plainly)

Rows where ojas was slower in **all 15 rounds** (verified direction), ranked
by run A's median ratio. The `floor` row is excluded, since it measures the
per-op fixed cost and is not work. Ranking by each row's best round out of
15 instead (the round most favorable to ojas) gives the same top 10, so the
order is not an artifact of contention.

| rank | backend | row | run A median ratio [min-max] | best of 15 rounds | ojas / torch median ms |
| --: | :-- | :-- | :-- | --: | :-- |
| 1 | Metal | residual_add_bwd | 0.01x [0.01-0.01] | 0.04x | 1.878 / 0.016 |
| 2 | wgpu | sdpa_b2h8t1024d128_fwd | 0.01x [0.01-0.01] | 0.03x | 91.63 / 1.202 |
| 3 | wgpu | residual_add_bwd | 0.02x [0.01-0.03] | 0.09x | 0.898 / 0.016 |
| 4 | wgpu | sdpa_b4h8t2048d64_fwd | 0.02x [0.02-0.02] | 0.02x | 191.6 / 3.542 |
| 5 | wgpu | sdpa_b2h8t1024d128_bwd | 0.02x [0.02-0.02] | 0.02x | 280.7 / 5.318 |
| 6 | wgpu | sdpa_b4h12t1024d64_fwd | 0.02x [0.02-0.02] | 0.06x | 77.05 / 1.600 |
| 7 | wgpu | sdpa_b4h8t2048d64_bwd | 0.04x [0.04-0.04] | 0.04x | 705.7 / 25.39 |
| 8 | wgpu | sdpa_b4h12t1024d64_bwd | 0.04x [0.04-0.04] | 0.04x | 277.8 / 10.30 |
| 9 | Metal | adamw_full | 0.09x [0.08-0.10] | 0.11x | 598.1 / 53.52 |
| 10 | Metal | rms_qk_norm_bwd | 0.10x [0.09-0.10] | 0.15x | 53.15 / 5.276 |
| 11 | Metal | rms_qk_norm_fwd | 0.11x [0.08-0.19] | 0.33x | 4.736 / 0.534 |
| 12 | wgpu | block_fwd | 0.14x [0.13-0.14] | 0.18x | 120.0 / 16.92 |
| 13 | wgpu | rms_qk_norm_fwd | 0.17x [0.14-0.22] | 0.58x | 2.624 / 0.534 |
| 14 | wgpu | block_fwd_bwd | 0.18x [0.17-0.18] | 0.21x | 475.3 / 83.64 |
| 15 | Metal | rms_norm_fwd | 0.18x [0.09-0.70] | 0.70x | 1.654 / 0.258 |
| 16 | Metal | silu_fwd | 0.20x [0.13-0.28] | 0.50x | 2.267 / 0.466 |
| 17 | Metal | mul_fwd | 0.21x [0.21-0.24] | 0.76x | 2.899 / 0.615 |
| 18 | Metal | silu_bwd | 0.23x [0.18-0.30] | 0.56x | 3.191 / 0.691 |
| 19 | Metal | block_fwd | 0.24x [0.24-0.27] | 0.37x | 67.25 / 16.92 |
| 20 | wgpu | muon vs torch bf16 NS5 (all three shapes) | 0.25x-0.27x | 0.31x-0.36x | see tables |
| 21 | wgpu | linear, all eight rows | 0.29x-0.38x | 0.34x-0.43x | see tables |
| 22 | Metal | gate_bwd / gate_fwd | 0.33x / 0.36x | 0.64x / 0.98x | 4.069 / 1.350, 1.773 / 0.681 |
| 23 | Metal | cross_entropy_bwd | 0.35x [0.33-0.38] | 0.56x | 96.68 / 34.30 |
| 24 | Metal | block_fwd_bwd | 0.37x [0.34-0.38] | 0.60x | 230.5 / 83.64 |
| 25 | wgpu | muon vs torch fp32 NS5 (all three shapes) | 0.41x-0.44x | 0.54x-0.68x | see tables |
| 26 | Metal | muon vs torch bf16 NS5 (all three shapes) | 0.44x-0.45x | 0.58x-0.76x | see tables |
| 27 | Metal | linear fwd (all four), qkv/down/lm_head bwd | 0.46x-0.70x | 0.55x-0.98x | see tables |
| 28 | Metal | sdpa_b4h8t2048d64_fwd | 0.50x [0.47-0.65] | 0.76x | 6.909 / 3.542 |
| 29 | wgpu | cross_entropy_bwd | 0.53x [0.50-0.54] | 0.87x | 63.71 / 34.30 |
| 30 | Metal | vres_bwd | 0.64x [0.32-0.79] | 0.94x | 1.943 / 1.192 |

**The composed block, the training-relevant number:**
- Metal: 0.37x forward + backward and 0.24x forward. That is about 2.7 and 4.2 times slower.
- wgpu: 0.18x forward + backward and 0.14x forward. That is about 5.6 and 7 times slower.
- Direction verified in 15/15 rounds for all four; magnitudes as ranged above.

### The worst five, with likely causes

`residual_add_bwd` (ranks 1 and 3) is an **API-shape artifact, not a kernel
problem**. ojas's `residual_add_backward` returns two fresh copies of the
incoming gradient. torch autograd returns the same tensor with no kernel;
its 0.016 ms is the timer floor. A composed backward passes the gradient
through: `bench/ojas_rows.rs::block_backward` never calls it. It costs
training nothing unless a caller (or `Tape`) calls it. This is verified from
the row semantics; whether `Tape` calls it was not checked.

1. **wgpu causal SDPA, 0.01–0.04x (verified direction; cause inferred from
   source).** `ojas-kernels/src/wgsl/attention.wgsl:50-100` gives each lane
   one query row of a 64-lane workgroup. Each lane holds private `q[D]` and
   `acc[D]` arrays (64 or 128 floats each, likely spilled), computes a scalar
   dot product per key, and does a branchy online-softmax rescale of `acc`
   per key. No matrix units are used. Metal's kernels for the same op are
   FlashAttention-2 on the TensorOps matrix units
   (`ojas-metal/kernels/ojas_backend.metal:701-719`). They run the same shapes
   25–45 times faster than wgpu (run A: 3.08 vs 77.05 ms forward, 8.11 vs
   277.8 ms backward at the nanolab shape). This one op is about 75% of wgpu's
   block forward + backward. Run A medians: SDPA forward + backward is
   355 ms of the block's 475 ms, inferred from row sums.

2. **Metal `adamw_full`, 0.09x (direction verified, cause verified from
   source).** `ojas-metal/src/device.rs:1404-1448`. Per tensor:
   - allocates three fresh buffers;
   - runs four `ojas_check_finite` passes over the inputs;
   - makes three copies out, runs tessl's AdamW step, and three more checks;
   - `finish` waits on the device and reads the status words back;
   - makes three copies back, then a second `synchronize`.

   That is two full device waits and about 13 full-tensor passes, times 170
   calls, about 3.5 ms per call. wgpu's AdamW (0.70x) decides
   finite-or-not on the device with no host wait per call (`ojas-wgpu/src/backend.rs` module docs).
   torch's is one `optimizer.step()`.

3. **Metal `rms_qk_norm` backward 0.10x and forward 0.11x (direction
   verified; cause verified from source).** `rms_qk_norm_backward` is two
   `rms_norm_backward` calls (`ojas-metal/src/backend.rs:716-729`). Each runs
   `ojas_rms_bwd_w` (`ojas_backend.metal:287-303`). That kernel uses one thread
   per column, so dim = **64 threads for the whole GPU**, and each loops over
   49,152 rows serially. The forward runs `ojas_rms_rstd` with one 256-lane
   threadgroup per 64-element row (`ojas_backend.metal:199-232`), so 3 of 4
   lanes are idle, plus a separate `ojas_rms_apply` pass. wgpu's QK-norm
   forward (0.17x) has the same one-256-lane-workgroup-per-row shape
   (`ojas-kernels/src/wgsl/norm.wgsl`). Its backward (0.95x) leaves the
   weight gradient out of the per-row kernel and reduces it with a two-stage
   column sum (`col_sum`, `ojas-wgpu/src/backend.rs:488`, called at 1269), not
   one thread per column. That is verified from source; that it explains
   wgpu's 9x lead over Metal on this row is inferred.

4. **Metal per-op fixed cost on small and elementwise ops, 0.18–0.36x
   (direction verified; cause verified from source, split inferred).** Rows
   affected: `rms_norm_fwd`, `silu`, `mul`, `residual_add_fwd`, `gate`,
   `vres_fwd`, `permute`. Every Metal op:
   - allocates and initialises a status buffer (`device.rs:328-334`);
   - runs a separate `ojas_check_finite` dispatch over **every input and
     every output** (`device.rs:337-345`, e.g. `linear` at 728-746);
   - ends with `finish`: a device wait plus a host read of the status words
     (`device.rs:370-393`);
   - crosses the channel to the device thread and back.

   The 1-element `floor_silu_1` row is 0.13–0.19 ms at best, but 1.5–1.8 ms
   median under this contention. wgpu's is 0.24–0.35 ms median, and torch's
   0.08–0.15. wgpu records into a shared encoder and checks finiteness inline
   (`report()` in the WGSL) at one sync per iteration. The Metal block runs 24
   ops forward and 53 forward + backward (counted from `bench/ojas_rows.rs`),
   each paying this. Run A's Metal block forward, 67 ms, is close to the
   49 ms sum of its kernel rows (inferred).

5. **wgpu GEMM, 0.29–0.38x on every linear row (direction verified; cause
   inferred).** `ojas-kernels/src/wgsl/gemm.wgsl:1-20` uses a 16x16 workgroup
   per 64x64 tile with scalar vec4 FMAs from workgroup memory and no
   simdgroup matrix. lm_head forward reaches 316 GFLOP / 194.7 ms = 1.6
   TFLOP/s, against torch's 5.2 TFLOP/s and Metal's 2.4 (tessl `ExactF32`).
   The same kernel family drives wgpu Muon (0.41–0.44x against torch's fp32
   NS5).

Next in line: Metal `cross_entropy_bwd` at 0.35x. `ojas_ce_rows`
(`ojas_backend.metal:605-659`) uses one 256-lane threadgroup per 50,304-wide
row, three passes over the row with `precise::exp` twice and
`precise::divide` twice per element. That comes on top of the finite checks
over the 824 MB input and the 824 MB output. Metal linear trails at
0.46–0.70x; part of that is the extra checks over the 824 MB lm_head output
and the inputs (inferred).

## Where ojas is faster (verified direction, 15/15 rounds)

| backend | row | run A median ratio [15-round range] | why (inferred) |
| :-- | :-- | :-- | :-- |
| Metal | clip_grad_norm_full | 4.58x [4.29-9.36] | one device reduction plus one scale. torch's `clip_grad_norm_` on MPS goes through `linalg.vector_norm` per tensor (pytorch-parity-plan.md section 3) |
| wgpu | clip_grad_norm_full | 6.36x [5.81-9.10] | same |
| Metal | sdpa backward, nanolab and T=2048 shapes | 1.24x and 1.38x [1.01-1.77] | FlashAttention-2 tiled backward; torch with grad runs `_scaled_dot_product_attention_math` |
| wgpu | rope fwd / bwd | 3.67x / 4.73x [2.78-6.78] | one kernel; torch's `apply_rope` is several kernels (negate, `cat`, two multiplies, an add) |
| wgpu | cross_entropy_fwd | 2.79x [1.93-3.21] | one fused pass; torch materializes `log_softmax` |
| wgpu | rms_norm_bwd | 3.42x [1.67-5.76] | one fused per-row kernel; torch's autograd backward of `F.rms_norm` is several kernels |
| wgpu | vres fwd / bwd | 1.32x / 1.67x [1.01-2.17] | fused; torch does sigmoid, two multiplies and an add |

Metal rope is mixed (its per-op fixed cost eats the fused-kernel advantage).
Metal SDPA backward at head dim 128 is mixed (14/15 rounds slower).

## Findings for the owners (no library code was changed here)

1. **`Tape` cannot drive the GPU block from these examples.** Neither
   `ojas-metal` nor `ojas-wgpu` depends on `ojas-autograd`, even as a
   dev-dependency, and this lane may not edit `Cargo.toml`. The block forward
   and backward are composed by hand from Backend ops. Parity against
   nanolab's real `Block` checks the composition, not `Tape`. A `Tape`-driven
   GPU row needs an `ojas-autograd` dev-dependency on one GPU crate, or an
   example in `ojas-autograd` with GPU dev-dependencies.
2. **`docs/bench-cpu-vs-torch.md:153` is stale.** It records Metal attention
   backward at T=2048 as 306 ms against torch's 60 ms. This harness measures
   `sdpa_b4h8t2048d64_bwd` at 16.5–19.4 ms (median across runs) against 25–26
   ms for torch, so ojas is faster in all 15 rounds. Its 3.86 ms vs 2.81 ms
   linear 2048³ line was not re-measured here. That doc is outside this
   lane's ownership.
3. **`docs/pytorch-parity-plan.md` section 3, target 2 ("a one-pass gradient
   norm, about 3 ms where torch takes 150") is partly met.** torch takes
   137–154 ms here. ojas takes 16–35 ms (Metal) and 16–25 ms (wgpu), which
   includes the scale pass, not 3 ms.
4. Fix candidates by measured cost:
   - wgpu attention on matrix units, or a tiled multi-row-per-lane design;
   - Metal AdamW without the copy-in/copy-out and double wait (or a batched
     multi-tensor step);
   - `ojas_rms_bwd_w` as a two-stage column reduction;
   - one finite check per op output instead of separate passes over every
     input and output on Metal;
   - a shared-encoder mode for Metal like wgpu's.

   None was attempted.

## Reproduce

```bash
bash /Users/bharath/Code/research/ojas/bench/run_paired.sh 5                    # run A protocol
BENCH_ITERS=30 bash /Users/bharath/Code/research/ojas/bench/run_paired.sh 5     # runs B and C
/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/bharath/Code/research/ojas/bench/aggregate.py \
    bench/results/2026-10-01/runA bench/results/2026-10-01/runB bench/results/2026-10-01/runC   # direction table
```

Each run takes about 15 minutes for 5 rounds at 20 iterations and 18 minutes
at 30. Run it on an idle GPU: check `ioreg -r -d 1 -w 0 -c IOAccelerator` for
"Device Utilization %" near 0 first. Under contention the spread flags will
say so, as they did here.
