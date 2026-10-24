# Memory per step through the fine-tune loop's provider, tiny and 0.8B-Base (2026-10-21)

Task `ft-b5e64557` criterion 6: memory per step through the Qwen3.5 Tape `StepProvider`, at the tiny and 0.8B
shapes, with activation checkpointing on and off. Arm `tape-provider` of
`ojas-qwen35/tests/gpu_memory_rows.rs`: `TapeStepProvider` (`ojas-model/src/qwen35/step_provider.rs`) on
`MetalBackend` under a 28 GiB budget (30,064,771,072 bytes), the base loaded pinned to its sha256 (bf16 linears and
table), rank 16 / alpha 16 adapters on the ten `LoraKind` linears, base frozen once at load (`freeze_checked`), a
passing `preflight`, then one warmup step and three timed steps of `accumulate` (one sequence at the given width,
every position a scored row) plus `adamw_step`. One process per row, under `Lappi-decision/tools/mac_heavy.sh`.

| model | T | checkpointing | device peak | host footprint (lifetime max) | first step | mean step (3 after warmup) | tokens/s | status |
| :-- | --: | :-- | --: | --: | --: | --: | --: | :-- |
| tiny fixture | 2,048 | on | 0.33 GB | 0.41 GB | 42 ms | 35.9 ms | 57,072 | ok |
| tiny fixture | 9,638 | on | 1.19 GB | 1.29 GB | 435 ms | 456.7 ms | 21,102 | ok |
| tiny fixture | 2,048 | off | 0.34 GB | 0.41 GB | 43 ms | 45.3 ms | 45,187 | ok |
| tiny fixture | 9,638 | off | 1.40 GB | 1.50 GB | 600 ms | 376.1 ms | 25,623 | ok |
| 0.8B-Base | 2,048 | on | 3.96 GB | 4.53 GB | 2.79 s | 2.79 s | 733.2 | ok |
| 0.8B-Base | 9,638 | on | 7.87 GB | 8.46 GB | 17.02 s | 17.22 s | 559.8 | ok |
| 0.8B-Base | 2,048 | off | 18.08 GB | 18.60 GB | 2.24 s | 2.22 s | 921.7 | ok |
| 0.8B-Base | 9,638 | off | refused at `preflight` | - | - | - | - | refused |

- **device peak**: `step_peak_bytes`, the exact device peak since the pre-flight, read after the timed steps.
- **host footprint**: `footprint_max_bytes`, the process's lifetime max `phys_footprint` (memguard's metric).
- **tokens/s**: positions (T - 1) / mean step wall, the arm's convention.
- **refused** is a typed row: at 0.8B-Base, T = 9,638, checkpointing off, the pre-flight's dry run needs
  30,070,117,160 bytes against the 30,064,771,072 cap (measured peak before the refusal 28,298,555,432 bytes).
  That shape does not fit 28 GiB without checkpointing; with checkpointing it peaks at 7.87 GB. Its process
  exited 101 with `test result: FAILED`: the harness prints the refusal row and then panics by design
  (`ojas-qwen35/tests/common/mod.rs`, `refused`), so a refusal row is always a failing test run.

## Provenance and caveats

- ojas `4affd09` (branch `gp/7a-training`) for the 0.8B rows, on tessl `c2c1612`. The tiny rows were rerun on the
  same ojas tree plus the `ids()` vocabulary bound in `gpu_memory_rows.rs` (the first run drew ids below 150,000,
  above the tiny fixture's 8,320-token vocabulary, and every tiny row was correctly refused by
  `SequenceJob::validate`); tessl had moved to `6aa333a` (c0d8be5 plus a report-only commit; the tessl session
  reports no kernel, numerics or API change). The ids for the 0.8B and 2B vocabularies are unchanged by the bound.
  Each row carries a `tessl` field.
- The tiny rows' timings are milliseconds on a 2-layer, 64-wide model: launch overhead, not compute, so the
  checkpointing on/off wall difference there is not a reading. Their memory column is the reading.
- One process per row, n = 1 per cell: these are memory rows, not an A/B; no timing claim is made from them.
