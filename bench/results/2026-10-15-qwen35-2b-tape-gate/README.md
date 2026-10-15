# The real 2B Tape step against transformers (2026-10-15)

Task `gp-qwen35-2b-metal-tape-run`, criterion 3a, as Fable redesigned it on 2026-10-15. ojas's
Tape runs one forward and backward of the real Qwen3.5-2B-Base on Metal, every layer a
checkpointed segment, and is held to transformers' own float32 step on the same 128 tokens.
The test is `ojas-qwen35/tests/gpu_tape_2b.rs`, run once under `mac_heavy.sh`. The raw rows are
in `rows.jsonl` and the machine is in `env.txt`.

**The reference** is tessl's `tools/qwen35_ref/make_train_fixture.py 2b --tokens 128`, a CPU
process. It holds the loss, the ids, every 1-D parameter's and conv weight's gradient, layers 0
and 3 whole, and the embedding gradient's rows for the prompt's ids and 64 others: 149
gradients. The test checks that the snapshot's weights file is the one the reference was made
from, and records its sha256, which is the Lappi pre-registration's `safetensors_sha256`.

**The bounds** are tessl's own for the real 2B against transformers
(`real_2b_step_matches_transformers`): loss within 1e-4 relative, each gradient within 1e-2 of
its own peak. They were written before this run.

## Result: within bounds

| | value |
| :-- | --: |
| loss, Tape | 3.81033921 |
| loss, transformers | 3.81016612 |
| loss, relative | 4.54e-5 (bound 1e-4) |
| gradients compared | 149 of 149 |
| worst gradient | 4.05e-3 of its peak, `layers.20.linear_attn.conv1d.weight` (bound 1e-2) |
| resident (device) | 7.55 GB |
| step peak (device, exact) | 17.60 GB |
| step wall time | 853 ms |

- **The loss gap is the forward's, not ojas's.** tessl's own step on the same reference
  gives 3.81033944 (4.55e-5), and tessl's inference forward 3.81033982. The Tape lands within
  2.3e-7 of the first. tessl's test explains the 4.6e-5: the two f32 forwards differ, and the
  gradients of two slightly different functions differ by up to about 4e-3 of a peak.
- **The device budget was 28 GiB**, below the 32 GiB memguard, so a step that did not fit
  would have been a typed refusal, not a SIGTERM. The step peak was 17.60 GB.
- Run twice, the second after a change to how the row records the sha256 (the row here is the
  second): 3.81033921 both times, the same worst gradient. Wall time was 1,714 ms in the first
  run and 853 ms in the second; it is one step and not a throughput figure.

## The provider's leg: stopped by the memguard

The provider was taken out of this process (Fable): it is 30.13 GB resident at open and stages
a whole f32 table to read back any gradient, over the guard. Its leg was to be tessl's
`real_2b_step_matches_transformers`, in its own process, on the same reference
(`QWEN35_2B_SAFETENSORS` at the file above).

That run reached the loss and was then stopped with SIGTERM and no `mac_heavy` RSS-cap line,
which is the memguard. The loss part passed: train 3.81033944, inference 3.81033982,
transformers 3.81016612 (4.55e-5). The gradient comparison did not run. The second row of
`rows.jsonl` is transcribed from that log. The guard is not raised or bypassed, so on this
machine the provider's real-2B gradients stay unchecked until tessl's test reads them back with
less held at once. That is a change to hand to tessl.

## What this does not show

- **Not the LoRA step.** Every parameter takes a gradient here, as in
  `2026-10-11-bf16-frozen-base`. The LoRA memory row is `gp-ft-bf16-frozen-base` criterion 7.
- **Not bf16.** The linears and the tied table load f32. The bf16 rows are
  `2026-10-11-bf16-frozen-base` and `2026-10-14-bf16-tied-table`.
- **One sequence of 128 tokens.** The long-sequence memory figures are the rows above.
