# Column-panel tile walk for tessl's bf16 TN/NT and int8 GEMMs (2026-10-06/07)

tessl's exact-f32 GEMMs already walked a large B in column panels
(`../2026-10-02-gemm`). This extends that walk to the bf16 TN/NT coop kernels
(plain and accumulate) and the int8 dequant kernel, through one shared
`tile_walk<SM>` in tessl `kernels/matmul_tensorops.metal`:

- column panels of 512 rows of C (16 tile rows of 32, 4 of 128, 8 of 64);
- once B (the N×K operand) holds `N·K ≥ 2^23` elements: 32 MiB of f32, 16 MiB
  of bf16, 8 MiB of int8;
- except on a grid `tile_from_linear` walks in Morton order (square, with a
  power-of-two side), which keeps Morton. For exact f32 past the gate on such
  a grid, that is a change: it took panels before.

`tile-walk.diff` is the shader diff against tessl HEAD. Each tile's arithmetic
is unchanged; only the order threadgroups run in moves. Measured on M5 Pro,
with the GPU shared with other sessions' work.

| Path | What it holds |
|---|---|
| `sweep/tune{1,2}-r{1,2,3}.txt` | In-process sweeps, 3 rounds each (2026-10-06, 22:46–22:53 local). |
| `sweep/ratios.txt` | `sweep_ratio.sh` over all six: per case and variant, the ratio to the same round's production time, min / median / max. |
| `ab2/runs.txt` | Production A/B, 22 cases × 6 rounds × 2 sides (2026-10-07, ~17:01–17:04 UTC, load 8–11). Each run headed by `<case>-<old\|new>-r<round>`. |
| `ab2/summary.txt` | `ab_sum.sh` over those runs: per case, the new/old ratio of each back-to-back pair, min / median / max, and each side's best time. |
| `scripts/` | The scripts as run. Their paths are this machine's (`target-mathsrc/` is gitignored). |
| `tile-walk.diff` | The shader change. |

## How each was measured

**Sweep.** `tune_sweep.sh` builds `bench_gemm_tnnt_tune` with
`TESSL_GEMM_TUNE=1`, which adds the tune rig's copies of the coop kernels: one
on the production walk of the time, and panel variants with bands of 4, 8 and
16 tile rows (`_ph4`, `_ph8`, `_ph16`). Every variant of a case runs in the
same process, right after production, with 4 warm-up and 12 timed iterations.
Taking the ratio inside a round keeps the drift between rounds out of it. For
128-row tiles, `_ph4` is the 512-row band; for the 64-row accumulate tiles,
`_ph8` is.

**Production A/B.** `ab_tree.sh` copies tessl's working tree once and builds
the rig twice in one target dir. The old binary has tessl HEAD's shader
(`eb27c05f`) and the new one the working tree's (`3de61a9b`), so the two
differ in nothing else. The shipped shader, `538d18e8`, differs from
`3de61a9b` only in comments. `ab_prod.sh` then runs each case's public-API
production GEMM (`BENCH_PRODUCTION_ONLY=1`, 4 + 12 iterations) on the old
binary and on the new one back to back. It alternates which goes first by
case and by round, for 6 rounds.

An earlier production A/B (ab1, 2026-10-07 04:07 UTC) was discarded. Its old
and new binaries came from different trees, on either side of tessl `dade8e2`
(buffer page rounding), and it ran at load 10–33. Its controls read
0.41–1.59× per round.

## Sweep: the 512-row band against the walk it replaces

Medians of per-round ratios, 6 rounds for cases in both sweeps and 3 for the
bracket cases (`*_g*mib`). The full table, with the 1024- and 2048-row
bands, is `sweep/ratios.txt`.

| B (bf16) | Cases | 512-row band |
|---|---|---:|
| 24 MiB and up | NT and TN, plain and accumulate, 14 cases | 0.45–0.95× |
| 20 MiB and up, at K = 768 | NT, plain and accumulate, 5 cases | 0.45–0.73× |
| 20 MiB | `ntp_g20mib`, `ntp_g20mib_k768`, `tnp_g20mib`, `ntap_g20mib`, `tnap_g20mib` | 0.73–1.01× |
| 16 MiB (the gate) | `ntp_b16mib`, `tnp_b16mib`, `ntp_g16mib`, `ntap_g16mib`, `tnap_g16mib` | 0.87–1.02× |
| 12 MiB (under the gate) | `*_g12mib` | 0.91–1.00× |
| square power-of-two grids, against Morton | `*_morton_sq64`, `ntap_b16mib`, `tnap_b16mib` | 0.98–1.04× |

The 512-row band was best, or within noise of 1024 and 2048 rows, everywhere
but bf16 NT at K = 768 (`ntp_vocab_32768`: 0.64× against 0.61×). 1024 rows
lost up to 10% on bf16 TN accumulate (`tnap_g16mib`: 1.12× against 1.02×).

## Production A/B (ab2)

New/old time per back-to-back pair, over 6 rounds. Controls run the same walk
on both sides: B under the gate, a Morton grid, or exact f32 already in
panels.

| Case | GEMM | M×N×K | B | Role | Median | Min–max |
|---|---|---|---:|---|---:|---:|
| `i8nn_lmhead_50304` | int8 NN | 4096×50304×768 | 36.8 MiB | panels | 0.76 | 0.67–0.80 |
| `i8nn_vocab_32768` | int8 NN | 4096×32768×768 | 24 MiB | panels | 0.77 | 0.71–0.94 |
| `i8nn_longk_8192` | int8 NN | 4096×8192×8192 | 64 MiB | panels | 0.73 | 0.68–0.76 |
| `i8nn_b16m` | int8 NN | 4096×8192×2048 | 16 MiB | panels | 0.91 | 0.83–0.95 |
| `i8nn_b12m` | int8 NN | 3072×6144×2048 | 12 MiB | panels | 0.99 | 0.87–1.16 |
| `i8nn_b8m` | int8 NN | 3072×4096×2048 | 8 MiB | panels (at the gate) | 1.00 | 0.95–1.53 |
| `i8nn_b6m` | int8 NN | 3072×3072×2048 | 6 MiB | control | 1.06 | 0.92–1.42 |
| `i8nn_grid_small_b` | int8 NN | 8192×3072×768 | 2.3 MiB | control | 1.05 | 0.74–1.68 |
| `i8nn_morton_sq64` | int8 NN | 8192×4096×4096 | 16 MiB | control (Morton) | 1.00 | 0.98–1.04 |
| `ntp_lm_head_248320` | bf16 NT | 1024×248320×2048 | 970 MiB | panels | 0.72 | 0.65–0.75 |
| `ntap_vocab_32768` | bf16 NT accumulate | 4096×32768×768 | 48 MiB | panels | 0.50 | 0.42–0.55 |
| `tnap_lmhead_dw` | bf16 TN accumulate | 768×50304×4096 | 393 MiB | panels | 0.60 | 0.54–0.70 |
| `tnap_g20mib` | bf16 TN accumulate | 3072×2560×4096 | 20 MiB | panels | 1.03 | 0.96–1.13 |
| `ntp_g16mib` | bf16 NT | 3072×4096×2048 | 16 MiB | panels | 0.95 | 0.89–1.06 |
| `ntp_morton_sq64` | bf16 NT | 8192×4096×4096 | 32 MiB | control (Morton) | 0.99 | 0.95–1.02 |
| `tnp_morton_sq64` | bf16 TN | 8192×4096×4096 | 32 MiB | control (Morton) | 0.99 | 0.96–1.03 |
| `tnp_g12mib` | bf16 TN | 3072×1536×4096 | 12 MiB | control | 1.04 | 0.96–1.16 |
| `f32nt_morton_sq128` | f32 NT | 4096×4096×2048 | 32 MiB | panels → Morton | 1.02 | 0.94–1.10 |
| `f32tn_morton_sq128` | f32 TN | 4096×4096×2048 | 32 MiB | panels → Morton | 0.99 | 0.93–1.01 |
| `f32nt_morton_sq64` | f32 NT | 2048×2048×4096 | 32 MiB | panels → Morton | 1.00 | 0.98–1.09 |
| `f32tn_morton_sq64` | f32 TN | 2048×2048×4096 | 32 MiB | panels → Morton | 1.01 | 0.90–1.02 |
| `f32nt_vocab_16384` | f32 NT | 4096×16384×768 | 48 MiB | control (panels) | 0.99 | 0.95–1.06 |

## What it decided

- **int8 shares the gate.** Panels take 0.73–0.77× of the time with B of
  24 MiB or more, and are neutral at 8–12 MiB (medians 0.99–1.00×), so a
  separate int8 gate would buy nothing.
- **Square power-of-two grids keep Morton for every family.** bf16 panels
  measured 0.98–1.04× of Morton in the sweep, and exact-f32 Morton
  0.99–1.02× of panels here. The rule was settled beforehand: drop the Morton
  exception only if the f32 medians exceeded 1.05× with every round above
  1.0. They did not, and the code kept a single rule.
- **`tnap_g20mib` is not a regression.** The discarded ab1 put it at
  1.12–1.50×. The sweep (0.95×, in one process) and this A/B (1.03×,
  0.96–1.13) both put it within noise.

## Limits

- One machine (M5 Pro), with a GPU other sessions also used. Cases under
  ~2 ms swing widely per round: the int8 controls read 0.74–1.68×, and the
  three sub-2 ms controls had medians of 1.04–1.06×. Only the medians of
  the larger cases resolve a few percent.
- The gate was fitted on M5 Pro only.
- No test pins the speed. The tests guard the tile mapping: tessl's
  `panel_walk_matches_row_major_chunks_bit_for_bit` (`src/gemm.rs`, every
  bf16 TN/NT and exact-f32 lane, bit-exact against column chunks under the
  gate), `column_panels_cover_every_tile_exactly` (`tests/gemm_i8.rs`) and
  `exact_f32_column_panels_cover_every_tile`
  (`tests/gemm_ragged_shapes.rs`). All three failed with the partial-band
  clamp removed from `tile_from_linear_panel`. The int8 failure:
  `545x8200x1024: 200904 outputs wrong`, starting at row 512, the partial
  band.
