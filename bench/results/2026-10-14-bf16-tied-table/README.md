# bf16 tied table: 2B device memory, f32 vs bf16 table (2026-10-14)

Task `ft-69744e54`, the follow-up to `2026-10-11-bf16-frozen-base`. These are the peak device bytes of
one Qwen3.5-2B training step on Metal with every linear stored bf16, and the tied embedding table
(248,320 x 2,048) stored f32 against bf16. Each row is one process of
`ojas-qwen35/tests/gpu_memory_rows.rs` (`QWEN35_MEM_ARM`, `QWEN35_MEM_TOKENS`), run under
`mac_heavy.sh`, the two arms interleaved at each T. The raw rows are in `rows.jsonl` and the machine
is in `env.txt`. The measurement rules are those of `2026-10-11-bf16-frozen-base`: exact device
peaks (tessl's `peak_allocated_bytes`, reset once the weights are resident), Qwen/Qwen3.5-2B weights
(2B-Base's `text_config` is identical), deterministic ids below 150,000, micro-batch 1, every
position scored, every layer checkpointed.

**This is the same full-gradient step as 2026-10-11**, not the LoRA step: every parameter, the
frozen table included, takes an f32 gradient, so the table's 2.03 GB gradient is in every peak
below. No optimizer runs.

| T | table | table bytes | resident (device) | after forward | step peak | loss |
| --: | :-- | --: | --: | --: | --: | --: |
| 2,048 | f32 | 2.03 GB | 4.80 GB | 8.34 GB | **16.37 GB** | 13.909834 |
| 2,048 | bf16 | 1.02 GB | 3.78 GB | 7.32 GB | **15.35 GB** | 13.909834 |
| 9,638 | f32 | 2.03 GB | 4.80 GB | 9.95 GB | **18.97 GB** | 13.765928 |
| 9,638 | bf16 | 1.02 GB | 3.78 GB | 9.00 GB | **18.02 GB** | 13.765928 |

- **A bf16 table saves 1.02 GB resident**: exactly 1,017,118,720 bytes, the table's second two
  bytes (248,320 x 2,048 x 2), on the device and in the budget (4,781,866,240 to 3,764,747,520).
- **The step peak falls by the same 1,017,118,720 bytes at T = 2,048**, and by 950,009,856 at
  T = 9,638. The difference, 67,108,864 bytes, is one widened fused-head tile at the default chunk
  (8,192 columns x 2,048 x 4 bytes): the lane widens one vocabulary tile at a time into the staged
  tile `LceGeom` charges, never the whole table. That this tile is what sits at the 9,638 peak is
  inferred from the arithmetic, not traced.
- **No whole-table f32 transient exists.** A per-call widen would have added 2.03 GB at the LM
  head; the peaks above are lower than the f32-table arm's, not higher. The CPU and Metal unit
  tests pin the same thing by budget: the fused head's charge is the f32 call's plus exactly one
  `[chunk.cols, d]` block (`ojas-cpu/tests/bf16_table.rs`), and Metal runs the gather, backward
  and fused head under a budget that cannot hold a widened table (`ojas-metal/tests/bf16_operands.rs`).
- **The loss is the same in both arms** (to the six printed digits here; the fixture tests in
  `ojas-model` pin the loss and every gradient bit for bit on the CPU and Metal), as the operand
  contract requires: the checkpoint is bf16, so the f32 table holds the same values.
- **The f32-table rows reproduce 2026-10-11's `tape-bf16` rows byte for byte** (resident
  4,800,528,384, peaks 16,367,566,848 and 18,973,687,808), so the comparison is same-code.

## What this does not show

- **The LoRA step.** With a frozen table, the fused head still computes the table-sized weight
  gradient (tracked by ft-2bbced18, an ignored BREAK test in ojas-autograd), so a LoRA step's peak
  would include it too.
- **0.8B-Base rows**, and any backend but Metal. wgpu has no bf16 table lane and refuses a bf16
  table with `Dtype`.
