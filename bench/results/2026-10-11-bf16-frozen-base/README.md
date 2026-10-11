# bf16 frozen base: 2B device memory, f32 vs bf16 (2026-10-11)

Task `gp-ft-bf16-frozen-base`. These are the peak device bytes of one Qwen3.5-2B training step on
Metal, with the linear weights stored at f32 against bf16. Every arm runs micro-batch 1 and scores
every position. Each row is one process of
`ojas-qwen35/tests/gpu_memory_rows.rs` (`QWEN35_MEM_ARM`, `QWEN35_MEM_TOKENS`), run under
`mac_heavy.sh`. The raw rows are in `rows.jsonl` (a stopped process leaves its partial rows) and
the machine is in `env.txt`.

- **Device figures are exact peaks.** `step_peak_bytes` is tessl's `peak_allocated_bytes`, the
  device's `currentAllocatedSize` sampled at every buffer the pool creates. It is reset once the
  weights are resident (`MetalBackend::memory_and_reset_peak`,
  `Qwen35Step::reset_peak_device_bytes`). It is process-wide, so one arm runs per process.
- **The weights are Qwen/Qwen3.5-2B, not 2B-Base.** Only the 2B-Base config and tokenizer are on
  this machine. Its `text_config` is identical to 2B's key for key, so every shape and byte below
  is the same. The ids are deterministic and drawn below 150,000, so the loss is not a quality
  figure.
- **0.8B-Base is not measured.** Its weights are not on this machine, and Fable ruled the
  download out for this task (see "What this does not show").

## Tape rows: ojas `Tape` on `MetalBackend`, every layer checkpointed

`load_hf(…, linears)` stores every weight a linear reads at the arm's dtype. The tied embedding
(2.03 GB) and every norm, conv, `A_log` and `dt_bias` stay f32 in both arms. A bf16 weight is
widened on the device into a charged f32 copy for each op that reads it, and the copy is dropped
when the op returns.

**This is a full-gradient step, not the LoRA step.** The Tape has no frozen leaves yet
(`gp-ft-frozen-params-and-lora`). So every parameter, the frozen base included, takes an f32
gradient, and the peak over-counts a LoRA step's by those gradients (7.5 GB of weight gradients at
2B). Under Fable's ruling it can confirm a fit but cannot show a non-fit. No optimizer runs.

| T | linears | linear bytes | resident (device) | after forward | step peak | loss |
| --: | :-- | --: | --: | --: | --: | --: |
| 2,048 | f32 | 5.49 GB | 7.55 GB | 11.06 GB | **19.14 GB** | 13.909834 |
| 2,048 | bf16 | 2.75 GB | 4.80 GB | 8.34 GB | **16.37 GB** | 13.909834 |
| 9,638 | f32 | 5.49 GB | 7.55 GB | 12.74 GB | **22.73 GB** | 13.765928 |
| 9,638 | bf16 | 2.75 GB | 4.80 GB | 9.95 GB | **18.97 GB** | 13.765928 |

- **bf16 saves 2.75 GB resident**, exactly the linears' second two bytes (2,745,434,112 bytes).
  The budget's resident charge is 4,781,866,240 bytes, against 2 × 2,745,434,112 linear bytes
  plus 2,036,432,128 f32 bytes.
- **It saves 2.77 GB at the step peak at T = 2,048 and 3.75 GB at T = 9,638.** At 9,638 the peak
  falls by more than the resident saving. Not traced: the f32 arm's peak lands at a different
  point of the backward.
- **The loss is the same bits in both arms**, as the bf16 operand contract requires: the checkpoint
  is bf16, so the f32 load holds the same values.
- **T = 9,638 fits.** The bf16 peak is 18.97 GB of a 51.5 GB recommended working set, even with
  the full gradient this step over-counts. The pre-registered fallback (buckets capped at 8,192)
  is not triggered by this row. It could not be triggered by it in any case, since a full-gradient
  row only confirms a fit.

## Provider rows: `Qwen35Step` (tessl's whole step), full fine-tune, AdamW

`Qwen35Step::open(…, Precision)`. `F32` runs with `Numerics::ExactF32`. `Bf16` runs with tessl's
`Precision::Bf16` (bf16 weights, every GEMM input rounded to bf16), `Numerics::Bf16Operands`,
f32 masters (`UpdateRule::F32Master`) and f32 moments.

| T | precision | resident at open | after forward | step peak (with AdamW) |
| --: | :-- | --: | --: | --: |
| 2,048 | f32 | 30.13 GB | 33.08 GB | **33.89 GB** (loss 13.909834) |
| 2,048 | bf16 | 30.13 GB | 32.98 GB | stopped in the backward |
| 9,638 | f32 | 30.13 GB | stopped in the forward | — |
| 9,638 | bf16 | 30.13 GB | stopped in the forward | — |

- **Resident is the same 30,126,456,832 bytes at both precisions.** That is by design: 16 bytes
  per parameter either way (f32 4 + 4 + 8, bf16 2 + 2 weight and gradient bank + 4 f32 master
  + 8 moments). The bf16 provider saves memory only in activations: 0.10 GB after the forward at
  T = 2,048.
- **The bf16 bank is bf16.** tessl gives a bf16 model a bf16 gradient bank, so gradients summed
  across sequences are summed in bf16 (`tessl/src/qwen35_train.rs:132-139`). This is flagged and
  not changed.
- **"Stopped" means SIGTERM from the machine's `memguard`, inferred.** memguard (`-cap-gib 32`)
  terminates any process whose footprint passes 32 GiB, by SIGTERM. Its log file had been moved
  to the Trash and deleted while it ran, so no kill record was read. Each stopped process died by
  SIGTERM within seconds of passing about 33 GB of device allocation. The Tape arms, all under
  23 GB, were never stopped. These rows are not out-of-memory results: the 51.5 GB working set
  was never reached, and no Metal allocation failed.
- **The provider at 2B does not fit under this machine's 32 GiB guard past T = 2,048 f32.**
  Its 30.1 GB of state leaves under 4 GB for activations.

## What this does not show

- **The LoRA step itself** (acceptance criterion 7). It needs frozen leaves and
  `linear_backward_input` (`gp-ft-frozen-params-and-lora`). When those land, the same harness
  with a LoRA arm gives the row. Rows for 0.8B-Base need its weights downloaded.
- **A Metal out-of-memory at these shapes as `CapacityExceeded`** (criterion 8). Nothing here
  reached the device limit, and the typed class is `gp-oom-error-class`.
- **A bf16 embedding table.** It stays f32 here: the gather and the LM head are not in the four
  bf16 lanes (Fable). A bf16 table would save another 1.02 GB resident.
