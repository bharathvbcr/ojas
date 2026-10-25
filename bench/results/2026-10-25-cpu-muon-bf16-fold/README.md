# CPU Muon, bf16 Newton-Schulz: folding the output rounding into the combination passes (2026-10-25)

## Why

ft-e15ee76a criterion 4 asks to remove the CPU bf16 Muon slowdown. Commit
`0a8f76a` (`ojas-cpu/src/optim.rs`, `newton_schulz`) drops two of the three
per-iteration `round_bf16_in_place` passes (over `A²` and `B·X`) and rounds
those values where the combination passes read them instead. Each element is
still rounded exactly once before its single use, so the result is bit
identical. ojas-oracle's `ojas_cpu_bf16_muon_step_matches_stock_nanolab`
(Exact and Fast, 4 cases, 0 differing values) passed on the branch. This run
times the change.

## What was run (verified)

- `muonab.sh` (the launcher copy kept here): the six Muon rows of
  `ojas-cpu/tests/bench_ops.rs` in `time` mode (`OJAS_BENCH_OPS`), 6 threads,
  `CpuBackend` Fast. Each row is min-of-n over n timed calls after two warm-ups
  (n = 10 at 768x768, 5 at the larger shapes), on a fresh input copy per call.
- 5 rounds, alternating which binary goes first (odd rounds base then new,
  even rounds new then base). Each side built and ran under
  `Lappi-decision/tools/mac_heavy.sh`, in its own target dir.
- base: ojas `42d5a9f` (a `git archive`, so no dirty files). new: ojas
  `b42676d`, the branch head. Of the branch's commits only `0a8f76a` touches
  the CPU Muon path. `bench_ops.rs` is byte-identical on both sides. ojas-cpu
  does not build tessl.
- Apple M5 Pro, macOS 27.0.1, rustc 1.99.0, release profile.
- 1-minute load after each run (`run.out`): new round 1 ran at 38–95, all
  others at 3.5–15.6.
- Raw rows: `ops.txt` (one `OJAS_OP` line per row, run and round, prefixed
  with the side and round). `summary.py ops.txt` prints the table below.

## Results

| op | base min-of-mins ms | new min-of-mins ms | base round spread | new round spread | per-round new/base (min_ms) | median ratio | new faster in |
| :-- | --: | --: | --: | --: | :-- | --: | :-- |
| muon_768x768 | 8.381 | 8.367 | +4.5% | +10.7% | 1.028 0.974 1.008 0.979 1.092 | 1.008 | 2/5 |
| muon_768x768_bf16 | 11.537 | 11.060 | +19.3% | +4.8% | 0.842 0.940 0.921 0.941 0.996 | 0.940 | 5/5 |
| muon_2048x768 | 18.166 | 18.347 | +4.5% | +5.1% | 1.007 0.983 1.057 1.008 1.027 | 1.008 | 1/5 |
| muon_2048x768_bf16 | 22.708 | 22.544 | +22.1% | +15.2% | 0.937 1.004 0.987 0.957 0.997 | 0.987 | 4/5 |
| muon_3072x768 | 22.779 | 22.901 | +10.8% | +5.7% | 0.960 0.963 0.967 1.000 1.050 | 0.967 | 3/5 |
| muon_3072x768_bf16 | 29.016 | 28.230 | +15.1% | +11.8% | 0.945 1.019 0.992 0.967 0.947 | 0.967 | 4/5 |

## Reading

- **The f32 rows are the control.** `0a8f76a` does not touch them, and
  their per-round ratios run 0.960–1.092 (medians 0.967–1.008). That is this
  run's noise floor.
- **768x768 bf16: about 6% faster** (median 0.940, faster in 5 of 5 rounds;
  without the loaded round 1, 0.94 in 4 of 4). It is the only row clearly
  outside the control's spread.
- **2048x768 and 3072x768 bf16: within noise** (medians 0.987 and 0.967,
  4 of 5 rounds faster). The 3072x768 f32 control itself sits at 0.967, so
  no gain is shown at the larger shapes.
- **The slowdown is reduced, not removed.** bf16 over f32 by min-of-mins:
  base 1.38x / 1.25x / 1.27x, new 1.32x / 1.23x / 1.23x (768x768,
  2048x768, 3072x768). The remaining gap is what card
  `gp-cpu-bf16-muon-bfmmla-kernel` addresses: first a profile attributing it
  between the GEMMs and the remaining rounding pass, then a BFMMLA kernel
  only if it beats Accelerate.
