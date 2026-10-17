# LoRA step cost: device peak, host footprint and tokens/s, 0.8B-Base and 2B-Base (2026-10-17)

Task `ft-2bbced18` criterion 9 (Fable 2026-10-17), the `tape-lora` arm of
`ojas-qwen35/tests/gpu_memory_rows.rs`; it also carries criterion 7's exact peak. Each row is one process
(`QWEN35_MEM_ARM=tape-lora`, `QWEN35_MEM_TOKENS`, `QWEN35_MEM_CKPT`, `QWEN35_MEM_SNAPSHOT`), run alone under
`Lappi-decision/tools/mac_heavy.sh`, in the order of `rows.jsonl`. `env.txt` has the machine.

The step is composed from merged primitives: `bind_frozen` base (bf16 linears, bf16 tied table), rank 16 / alpha 16
adapters (f32; A random, B zero) on all ten `LoraKind` suffixes (150 sites at 0.8B and at 2B), `forward_loss_lora`
(the fused tied head over every position, micro-batch 1, no ignore index), `Tape::backward`, and one
`Backend::adamw_step` per adapter tensor with both f32 moments resident (lr 1e-4, betas 0.9/0.999, wd 0). Dropout
0. The device is a `MetalBackend` under a 28 GiB budget (30,064,771,072 bytes), below memguard's 32 GiB.

| model | T | checkpointing | device peak | host footprint (lifetime max) | first-step wall | mean step wall (3 after warmup) | tokens/s | status |
| :-- | --: | :-- | --: | --: | --: | --: | --: | :-- |
| 0.8B-Base | 2,048 | on | 3.94 GB | 4.45 GB | 2.96 s | 3.11 s | **658.2** | ok |
| 0.8B-Base | 9,638 | on | 7.83 GB | 8.41 GB | 20.17 s | 19.29 s | **499.5** | ok |
| 2B-Base | 2,048 | on | 6.65 GB | 7.95 GB | 6.04 s | 6.09 s | **335.9** | ok |
| 2B-Base | 9,638 | on | 13.10 GB | 14.31 GB | 37.30 s | 36.84 s | **261.6** | ok |
| 2B-Base | 9,638 | off | refused at `forward` | (resident 7.69 GB) | - | - | - | refused |

Reading the columns:
- **device peak**: `step_peak_bytes`, tessl's exact `peak_allocated`, reset once the weights and adapters are resident and
  read after the warmup step and three timed steps (first-step peak is in each row and is within 11 MB of it).
- **host footprint**: `footprint_max_bytes`, the process's `ri_lifetime_max_phys_footprint` (`proc_pid_rusage`, the
  figure memguard and `mac_heavy.sh` act on), which includes the weight-load transient, so it is the high-water
  mark of the whole process. `end_footprint_bytes` and the per-phase fields are in the rows.
- **tokens/s**: `positions / mean step wall`, positions = T - 1 (the arm's existing convention: T ids, T - 1 scored
  positions). Using T instead moves 9,638 by 0.01%. Step wall is tape, forward, backward, 150-site adapter
  gradient collection, AdamW over every adapter tensor and the final `sync`; it excludes data loading. The first step
  is the warmup and includes first-use shader compilation, so it is reported separately and not in the mean.
- **refused** is a typed row, not a SIGTERM: at T = 9,638 with checkpointing off the forward asked for 78,946,304
  bytes with 30,009,423,860 live of the 30,064,771,072 cap (`CapacityExceeded`). It is a result: that shape does not
  fit 28 GiB without checkpointing. Its device peak is "above the cap" and is not a number; the refusal row does not
  carry a footprint beyond the resident one.

## What this does not show

- **D17 and the span head** (criterion 7's wording): the arm scores every position through the fused tied head
  (`forward_loss_lora`), not a gathered-row head (`forward_hidden_lora` + `gather_rows`). A span-head step differs
  in the head's cost, not the tower's; it is not measured here.
- **Real data**: ids are deterministic noise below 150,000 and the same sequence every step; step walls do not
  depend on values, but there is no padding, dropout or per-example mask here.
- **The provider's LoRA-free full-FT step** (criterion 4) and the torch MPS / mlx-lm comparison rows (criterion 5):
  not in this task.
- Three timed steps per row: the step-to-step spread is in the rows (`step1_ms`..`step3_ms`); at 2B T = 9,638 the
  walls are 36.37, 36.69 and 37.46 s (3% spread), at 2B T = 2,048 the first timed step was 6.49 s against 5.89 s.
- The tiny-fixture guard (`gpu_lora_step_trains_only_the_adapters`) is ignored like the rows and runs on Metal, not
  the CPU: after two steps every adapter tensor moved, no base tensor changed a bit, and a gradient on a frozen
  base parameter is refused by the step. Removing the `adamw_step` call fails it (the loss does not change); binding
  the base with `bind` instead of `bind_frozen` fails it ("base parameter 0 holds a gradient").

## Amendment 1 (ft-da2c4ab1), computed here; not written into Lappi

`N = largest N <= 400 with N x 35,403 / tokens_per_s <= 17,280 s` (80% of 6 h = 17,280 s):
- 2B T = 9,638, checkpointing on, 261.607 tok/s: 17,280 x 261.607 / 35,403 = 127.69, so **N = 127**
  (one 35,403-token step is 135.3 s). Across the three timed steps' own rates the same rule gives 125 to 129.
- 2B T = 2,048, checkpointing on, 335.915 tok/s: 163.96, so N = 163 (105.4 s per step). Here the
  T = 2,048 rate is the higher one, so it is not the conservative figure the rule text expects.
