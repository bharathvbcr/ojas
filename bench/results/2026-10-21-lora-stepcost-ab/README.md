# LoRA step cost A/B: base scanned once, and the frozen-linear dW skip, 2B-Base (2026-10-21)

Task `ft-2bbced18` criteria 1 and 4. Three arms of `ojas-qwen35/tests/gpu_memory_rows.rs`, each one process under
`Lappi-decision/tools/mac_heavy.sh`, interleaved arm by arm over 5 reps at T = 2,048 and 9,638 (micro-batch 1,
checkpointing on, 28 GiB `MetalBackend` budget). `env.txt` has the machine and the exact command.

- `tape-lora`: the LoRA step as it runs today: frozen bf16 base, rank 16 / alpha 16 adapters on the ten `LoraKind`
  linears (150 sites), dW skipped for frozen linears, every matmul's finite check scans its weight.
- `tape-lora-checked`: the same step with the base declared frozen once at load (`MetalBackend::freeze_checked`),
  so no matmul, gather or fused head rescans a weight (criterion 2's change); `freeze_ms` is that one scan.
- `tape-lora-fulldw`: every base linear weight a trainable leaf (table and norms frozen): the step the dW skip
  replaces (criterion 4's comparison).

`QWEN35_MEM_PHASES=on` waits for the forward before the backward starts, so `min_forward_ms` and
`min_backward_ms` split the step. Each process runs one warmup step and three timed steps; `min_step_ms` is the
fastest of the three. The reading is the minimum over the 5 reps.

## Min-of-5 (ms) and paired deltas

| T | arm | min step | min forward | min backward | best tokens/s | device peak |
| --: | :-- | --: | --: | --: | --: | --: |
| 2,048 | tape-lora | 5,480 | 2,295 | 3,178 | 372.0 | 6.63 GB |
| 2,048 | tape-lora-checked | 5,441 | 2,268 | 3,155 | 374.9 | 6.63 GB |
| 2,048 | tape-lora-fulldw | 6,314 | 2,312 | 3,970 | 323.9 | 11.81 GB |
| 9,638 | tape-lora | 29,478 | 11,906 | 17,536 | 322.0 | 13.09 GB |
| 9,638 | tape-lora-checked | 29,485 | 11,916 | 17,555 | 321.1 | 13.09 GB |
| 9,638 | tape-lora-fulldw | 33,203 | 12,028 | 21,149 | 281.8 | 18.45 GB |

Per-rep `min_step_ms` of each arm against `tape-lora` in the same rep (%):

| T | rep | checked | fulldw |
| --: | --: | --: | --: |
| 2,048 | 1 | -0.7 | +14.9 |
| 2,048 | 2 | -0.6 | +14.4 |
| 2,048 | 3 | -0.5 | +15.2 |
| 2,048 | 4 | -1.8 | +9.4 |
| 2,048 | 5 | -0.7 | +15.2 |
| 9,638 | 1 | +0.7 | +10.5 |
| 9,638 | 2 | 0.0 | +14.0 |
| 9,638 | 3 | +0.5 | +12.9 |
| 9,638 | 4 | +4.9 | +77.3 |
| 9,638 | 5 | +50.4 | +79.9 |

## Reading

- **Base scanned once at load (criterion 2):** at T = 2,048 it is faster in 5 of 5 paired reps, by 0.5-1.8%
  (min-of-5 -0.7%). At T = 9,638 the uncontended reps 1-3 give +0.7%, 0.0%, +0.5%: no effect distinguishable from
  the per-arm spread. The scan it removes is a small share of a step whose time is GEMM and attention; the
  one-time cost is 49 ms.
- **dW skip (criterion 4):** the skip saves 9.4-15.2% of the step at T = 2,048 (5 of 5 reps) and 10.5-14.0% at
  T = 9,638 (reps 1-3). The weight gradients are 792 of 3,970 ms (20%) of the full-dW backward at T = 2,048 and
  3,613 of 21,149 ms (17%) at T = 9,638; the forward is unchanged. The skip also removes 5.2-5.4 GB of device peak.
  The full-dW step fits the 28 GiB budget at both widths, so no refusal row.

## Noise and what this does not show

- **Contention:** the lock is per arm, so other sessions' jobs ran between arms. The T = 9,638 reps 4-5 ran during
  hours of machine contention (per-arm spread up to 3x, every arm slower) and are kept in `rows.jsonl`; the paired
  deltas above are read from reps 1-3. Min step and min backward come from reps 1-3; one minimum does not:
  `tape-lora`'s min forward at T = 9,638 (11,906 ms) is rep 4's (reps 1-3 give 11,926 ms).
- **Arm order was fixed, not rotated:** every rep ran `tape-lora`, then `-checked`, then `-fulldw`. The
  base-scanned-once reading (-0.7%, 5 of 5 reps) therefore cannot be told apart from a position effect; the dW-skip
  reading (+9 to +15%) is too large for position to explain. A rotated rerun is filed as a follow-up.
- **Rounding:** a few cells truncate rather than round (e.g. 5,441 is 5,441.999); no delta changes.
  Per-arm spread (max/min - 1) of `min_step_ms` over the uncontended reps is 4.5-5.4% at T = 2,048 (reps 1-5)
  and 1.2-2.7% at T = 9,638 (reps 1-3), so the criterion-2 effect at T = 2,048 (-0.7%) is below the spread and
  rests on its paired sign consistency (5 of 5), not on its size.
- **n:** 5 processes per arm and width, 3 timed steps each; no confidence interval is claimed beyond the sign
  consistency and spread above.
- **Scope:** every position is scored through the fused tied head (`forward_loss_lora`), not D17 and the span head;
  data loading is excluded.
