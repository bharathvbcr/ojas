# f32 linear_bias_forward adds its bias in place (F), 2026-11-03

Rules: `rules.md`, committed (`145c688`) before the job ran. Its mtime, 11:36:17 local, heads
`job4.log`. The job ran 18:27:47-18:40:17Z after 51 minutes waiting for the lock, at one-minute
load 4.5-16. Job: `job4.sh`. Op-level analysis: `summary-ops.txt`. End to end: `e2e/`,
`e2e/summary.txt`.

## Outcome in one line

**F speeds up the f32 `linear_bias_forward` by 11-13% at SpikeTrack's projection shapes without
moving a bit.** That holds at op level (hashes) and end to end: all seven SpikeTrack runs give
byte-identical predictions, base and F alike. The tracker step does not move measurably, as
predicted.

## Gates

| gate | result |
|---|---|
| 1. Bit identity, op level | **held**: `hash-P.txt` = `hash-F.txt` for `lin36k`, `linb36k`, `lin5k_up`, `linb5k_up`, `linb5k_down` |
| 2. Tests at F, then clippy | **held**: `linear_bias_fused` 3/3, `attention_flash_bits` 2/2, `encoder_ops` 11/11, `encoder_ops_edges` 7/7, `bf16_accumulation` 12/12, `autocast_precast_weight` 3/3, `attention_flash` 11/11, `attention_routing` 15/15 (3 ignored, as at base). `cargo clippy -p ojas-metal --release --all-targets --no-deps -- -D warnings` clean |
| 3. Speed, op level | see below: **faster** on both read arms; controls inside 1.00 ± 0.05 |
| 4. E1, end-to-end parity | **held**: `ctl-A`, `r1-A`, `r1-F`, `r2-F`, `r2-A`, `r3-A`, `r3-F` all byte-identical to `ctl-F` (`82.json`) |
| 4. E2, end-to-end speed | **not measured** (inside the noise), as predicted |

Op level: 8 rounds of P, F and P′ rotated, each process 3 warm-ups + 15 timed calls, synced on
both sides. Values are medians over rounds of process medians, in ms.

| arm | P | F | F/P per round | median (range) | N | verdict |
|---|---|---|---|---|---|---|
| `linb36k` `[36400,256] x [256,256]` + bias | 1.656 | 1.468 | .880 .916 .896 .899 .872 .869 .862 .897 | **0.888** (0.862-0.916) | 0.030 | **faster** |
| `linb5k_up` `[5184,256] x [2048,256]` + bias | 1.730 | 1.505 | .878 .890 .847 .862 .857 .866 .870 .891 | **0.868** (0.847-0.891) | 0.060 | **faster** |
| `lin36k` (control, untouched) | 1.381 | 1.403 | | 1.009 | 0.034 | in bar |
| `resadd36k` (control, untouched) | 0.584 | 0.577 | | 0.982 | 0.032 | in bar |
| `lin5k_up`, `badd36k`, `perm_head36k` (untouched) | | | | 1.002, 1.022, 1.013 | | not measured |

The prediction was 10-20% faster: met at 11% and 13%.

End to end (SpikeTrack `38e0a35` export built against P and against F; VISEM 82, 34 frames, 16
objects; steady frames ≥ 8, medians):

| stage | P (A) r1 / r2 / r3 | F r1 / r2 / r3 | F/P | noise | verdict |
|---|---|---|---|---|---|
| memory attention (device) | 283.8 / 302.4 / 308.1 | 277.6 / 292.0 / 287.3 | 0.978 / 0.965 / 0.932 | 0.086 | not measured |
| tracker step | 399.9 / 446.4 / 469.0 | 395.5 / 443.4 / 414.5 | 0.989 / 0.993 / 0.884 | 0.173 | not measured |

Every round's memory-attention ratio is below 1, but none clears the 8.6% noise. Load climbed from
6.6 to 16 during the end-to-end runs. Read from the op deltas, F saves about 0.19 ms on each of
the 12 projections over 36k memory tokens and 0.22 ms on each of the 4 FFN up projections. That
is ~3 ms per full-depth frame **[inferred]**, plus an unmeasured share on the 5,184-token
projections and on every other f32 linear with a bias in SpikeTrack (detector, necks).

## Whole-suite check (job 5, 18:51-18:55Z, `suite/`)

Run at tip `8b5dc42` (F plus docs and results; the library is F's):

| check | result |
|---|---|
| `cargo test --release -p ojas-metal -- --test-threads=1` | **passed**: 624 passed, 0 failed, 22 ignored |
| `cargo test --release -p ojas-model -- --test-threads=1` | **passed**: 432 passed, 0 failed, 8 ignored |
| `cargo clippy --workspace --all-targets -- -D warnings` | **failed**, on the pre-existing `ojas-cpu/src/quant_sparse.rs:468` `chunks_exact` lint (toolchain 1.99), the one `../2026-11-02-sam-xattn-q2/summary.md` recorded. That stops the run in `ojas-cpu`, so other crates were not linted by it. `ojas-metal` was linted clean with `--no-deps` in job 4. |

`./scripts/fmt_check.sh` and `./scripts/check_tracked_targets.sh` pass at the tip.

## What it is

- `ojas-metal/kernels/ojas_backend.metal` `ojas_bias_rows`: `y[row, col] += bias[col]` in place, a
  2-D grid, ST_OUT on a non-finite sum.
- `device.rs` `linear_bias`: its f32 branch is the exact-f32 GEMM, then that pass. It used to
  refuse; the backend never sent it the f32 lane.
- `backend.rs` `linear_bias_forward`: the f32 lane with a `[out]` bias now takes the fused
  command. The other bias shapes keep `linear_forward` + `broadcast_add_forward`.
- Fault reporting: a non-finite output is reported as `linear_bias_forward`, where the composed path
  named `linear_forward` (a non-finite GEMM output) or `broadcast_add_forward`.
- Budget: the fused command charges the same output plus at most a bias-sized copy, which the f32
  path does not make.
- Binaries: `bins.tsv`. The SpikeTrack builds: `e2e/build-st-*.log`.
