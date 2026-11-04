# Q4 GEMV alone at E4B shapes: fraction of peak, against MLX (2026-11-04)

This answers ojas task ft-5b2a72ee, a precondition of the megakernel probe ft-e6d8035e. The question:
does tessl's Q4 GEMV, which gemma-metal's default decode runs, reach the memory-bandwidth ceiling and
match MLX when it runs alone?

## Setup

- **Kernel:** `gemv_q4_mlx_simd` from tessl 52bdd19 (`kernels/gemv_q4_mlx.metal`, compiled with
  `xcrun metal -std=metal4.0 -I tessl/kernels`), dispatched with tessl's geometry: 64 threads per
  threadgroup, 8 rows per threadgroup (`nn::simd_gemv_threadgroups`).
- **Weights:** synthetic MLX affine Q4, group 64, bf16 scale/bias pairs; bf16 x.
- **Cache:** each timed batch walks distinct weight copies totalling at least 256 MB, so the system
  level cache cannot hold them, as in decode.
- **Host:** `gemv.swift` (Metal API, GPU time of one command buffer per batch, plus wall time).
- **MLX:** `mlx_qmv.py`, MLX 0.32.4 `quantized_matmul` at the same shapes and group size. It is timed
  by wall clock, so it includes encode.
- **Ceiling:** a float4 read-reduce kernel over 512 MiB. A copy kernel is also reported.
- **Runs:** `run.sh`, 3 rounds interleaved (tessl, MLX, ...), 5 repeats each, inside one mac_heavy hold;
  load average 9-13. Raw output: `run.log`.
- **Parity:** an f64 CPU reference of the same packing on 256 rows per shape; max rel error 2.8e-4 in
  every round.

## Results [V, `run.log`]

The read ceiling is 275-281 GB/s (copy: 250-271 GB/s read+write).

| shape (rows x cols) | MB per call | tessl GPU time, min over rounds | tessl GB/s | of ceiling | MLX wall min | MLX GB/s |
|---|---|---|---|---|---|---|
| MLP gate or up (10240 x 2560) | 14.75 | 61.8-64.2 us | 230-239 | 83-85% | 62.9-75.2 us | 196-235 |
| MLP down (2560 x 10240) | 14.75 | 59.4-61.9 us | 238-248 | 87-89% | 66.0-74.6 us | 198-224 |
| q, sliding (2048 x 2560) | 2.95 | 13.4-14.0 us | 211-220 | 76-79% | 14.5-15.6 us | 189-204 |
| q, global (4096 x 2560) | 5.90 | 26.9-27.5 us | 215-220 | 78% | 27.2-30.8 us | 192-217 |
| k or v alone (512 x 2560) | 0.74 | 9.1-9.3 us | 80-81 | 29% | 4.0-4.3 us | 171-183 |
| o, sliding (2560 x 2048) | 2.95 | 13.0-13.4 us | 221-226 | 79-81% | 14.5-15.8 us | 187-204 |
| lm_head, tied (262144 x 2560) | 377.5 | 1.32-1.43 ms | 265-285 | 96-102% | 1.54-1.70 ms | 222-245 |

tessl's wall time per call, which includes encode as MLX's does, is 68-78 us for the MLP shapes.
The ranges are the per-round minimums over the 3 rounds.

## Reading

- **[V] The GEMV kernel is not the decode bottleneck.** Alone, it streams weights at 83-89% of the
  read ceiling at the MLP shapes, which are 75% of E4B's bytes. It matches or beats MLX at every shape
  but one. That one is a lone 512-row K or V projection: 64 threadgroups underfill the GPU, and MLX is
  2x faster there. gemma-metal fuses K and V into one 1,024-row dispatch, which this probe did not time.
- **[I] The 3x gap to mlx-lm sits between the kernels.** At ~240 GB/s, the ~2.86 GiB a token reads
  (DECISIONS D15) is ~13 ms of GEMV. A measured E4B token takes ~42 ms. The other ~29 ms is everything
  else: the non-GEMV kernels, barriers and waits across ~600 dispatches, and host work. mlx-lm's ~13 ms
  per token (D2) is about this GEMV floor.
- **Verdict on ft-e6d8035e:** by its own rule (the kernels match MLX alone), proceed with the persistent
  or megakernel prototype, or an equivalent that removes inter-kernel serialisation. First profile the
  ~29 ms with GPU timestamps per dispatch in the real decode graph, to split kernel time from gaps.
  This also corrects DECISIONS D20, which assumed GEMV bandwidth.

## Contention (DECISIONS D19)

This is a contended run: load average 9-13, not a quiet machine. The ceiling and both kernels were
timed under the same load in alternating rounds. Per-round minimums agree within 5% across the 3 rounds
on the MLP shapes; that spread is the noise floor of a repeated arm. Contention lowers measured
bandwidth, so the fractions of peak are lower bounds. The margin between tessl and MLX should be
re-read on a quiet machine before it is quoted.

## Not measured

- The fused kernels the default decode graph actually uses: gate/up + GELU, K∥V, bf16-x variants.
- Real E4B weights. The access pattern is the same; the values are synthetic.
- MLX's GPU-only time.
- The per-dispatch gaps inside a real token.
