# Clip scale pass: share of a step (2026-11-09)

Task ft-a306b417 (spike) asks whether folding the gradient-clip coefficient into the optimizer step
is worth a `Backend` trait change. A fold would remove the clip's scale pass: one read and one write
of every gradient after the norm. This directory measures what that pass costs, as a share of the
step, on two paths:

- the nanolab-124M `Trainer::step` (123,699,612 gradients, AdamW plus Muon);
- the Qwen3.5-2B LoRA fine-tuning step (`TapeStepProvider::adamw_step`, 300 adapter tensors).

The card orders this after ft-ebbed90b (the paired clip run against torch 2.15). It ran first
because the owner asked for the measurement now. ft-ebbed90b's torch comparison is still open. The
`clip_norm_only` row it needs was run here for the Metal side only.

## Result

| path | scale-pass cost | step | share | how measured |
| --- | --- | --- | --- | --- |
| nanolab B=1, T=1024 | 4.14 ms (min), 5.23 ms (median) | 639 ms (min), 779 ms (median) | **0.65%** (min/min), 0.67% (median/median) | direct rows `clip_grad_norm_full` - `clip_norm_only` |
| nanolab B=8, T=1024 | same 4.14 ms | 2523 ms (min), 2741 ms (median) | **0.16%**, 0.19% | same |
| Qwen3.5-2B LoRA r16, T=2048 | +10.61 ms median paired (range -0.01 to +18.07, 9/10 slower) | 6549 ms median | **0.16%** | in-process A/B, `grad_scale` 0.5 vs 1.0 |

The bound for turning clipping off altogether is 6.63 ms, the largest `clip_grad_norm_full`
min-of-N of five processes. The trainer cannot run with clipping off, because `grad_clip` must be
finite and above 0 (`ojas-model/src/trainer.rs` config validation). So this bound comes from the
direct row. It is **1.04%** of the B=1 step and **0.26%** of the B=8 step.

**Verdict: no-go.** A fold would save at most about 0.65% of a pretraining step at B=1, 0.16% at
B=8, and 0.16% of a LoRA step. The whole clip costs at most about 1%. That does not pay for a
`Backend` trait change, a second AdamW contract (pre-clip buffer, opt-in mode), and a matching Muon
change. The decision is the owner's, and the card requires the owner's explicit sign-off for a
trait change either way.

**Rule disclosure:** I wrote the decision rule after reading the numbers. The card set no
threshold. Its own arithmetic predicted a "modest, bounded win", about 14% of optimizer-plus-clip
traffic. The measured shares are well under 1% of a step, so no threshold the card could have meant
would flip the verdict.

## Trainer A/B: the effect is below its noise floor

`nanolab_config_ab clip <B> <rounds> 1024` builds three nanolab-124M trainers in one process. They
use the same model init (seed 1337), the same token stream and `autocast` Off:

- `scale`: `grad_clip` = 1e-3. The first step's pre-clip norm is 23.1 at B=1 and 8.4 at B=8, so
  every step clips and runs the scale pass.
- `noscale`: `grad_clip` = 1e30. The norm runs, the coefficient is 1, and Metal skips the scale
  (`MetalBackend::clip_grad_norm`). This is the step a fold would leave.
- `noscale_aa`: identical to `noscale`. It is the A/A twin that gives the noise floor.

Each round runs one step of each arm in a rotated order, synced. The first 2 rounds are warm-up and
dropped. The driver ran 3 processes per batch size and alternated which batch size went first.

Checks that the arms ran different paths: in every process, every `scale` step's pre-clip norm was
above 1e-3. The two `noscale` arms' losses were bitwise equal. The `scale` arm's losses after
the first step differed from theirs. The data is in `trainer.jsonl` and was checked per process.

| batch | n paired steps | scale - noscale, median | A/A (twin - noscale), median | A/A \|delta\| p90 | scale slower | twin slower |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 84 (3 x 28) | -15.5 ms | -9.8 ms | 147.7 ms | 33/84 | 37/84 |
| 8 | 30 (3 x 10) | +7.6 ms | -24.0 ms | 221.4 ms | 17/30 | 13/30 |

Per-process medians of the paired delta:

- B=1: -35.8, -15.0, -11.7 ms (A/A -54.1, +0.1, +4.4 ms).
- B=8: -25.4, +46.6, -29.0 ms (A/A -22.3, -5.4, -69.7 ms).

Per-step jitter here is about ±150-220 ms, about 35 times the 4 ms effect. The trainer A/B cannot
resolve the scale pass. It does rule out a hidden in-trainer cost much larger than the direct rows
show: the scale arm sits inside the A/A distribution at both batch sizes. The share above comes from
the direct rows, which measure the same pass in isolation.

Min-of-N per arm (pooled):

- B=1: scale 643.8 ms, noscale 639.0 ms.
- B=8: scale 2554.3 ms, noscale 2523.3 ms.

The B=1 steps are slower than the 2026-11-02 config A/B (`../2026-11-02-nanolab-config-ab/`,
`b1-base` median 0.566 s), where this run's median is 0.779 s. The 1-minute load was 4.9 to 10.5
throughout (`runs.tsv`). The rows' `_load` probe read the GPU at 81-91% busy just after the trainer
part ended. Whether another GPU user or a code change since then accounts for the difference was
**not investigated**. Both arms of every pair saw the same conditions, so the paired deltas and the
A/A floor stand. The step denominators in the shares are this run's, and the 11-02 median would
raise the B=1 share to about 0.73%.

## Direct rows: the scale pass in isolation

`metal_vs_torch` with `BENCH_ROWS=clip_` runs these rows over the 170 nanolab gradient shapes:

- `clip_grad_norm_full`: norm plus scale.
- `clip_norm_only`: norm alone, `max_norm` 1e30.

Each row runs 5 warm-up calls and 20 timed calls, and checks parity against torch MPS's reference
outputs.

| process | full min / median ms | norm-only min / median ms | scale (full - norm), min / median |
| --- | --- | --- | --- |
| 1 | 6.63 / 7.91 | 2.39 / 2.83 | 4.24 / 5.08 |
| 2 | 6.38 / 7.91 | 2.10 / 2.53 | 4.28 / 5.39 |
| 3 | 6.07 / 6.43 | 2.31 / 2.81 | 3.76 / 3.62 |
| 4 | 6.04 / 8.11 | 2.16 / 2.88 | 3.88 / 5.23 |
| 5 | 6.19 / 8.36 | 2.05 / 2.13 | 4.14 / 6.24 |
| median | | | **4.14 / 5.23** |

Parity: max abs 2.3e-05, rel 2.8e-07 (full) and 2.3e-07 (norm), tol 1e-3, `ok` in all 10.

The scale pass moves about 0.99 GB: 123.7M f32 read and written. At 4.14 ms that is about 239
GB/s, in line with the doc's ~230 GB/s estimate. The difference also holds the host-side coefficient
arithmetic (`bench/README.md`).

Data: `clip_rows_normonly.jsonl` (the 5 processes above). `clip_rows.jsonl` is the driver's own
rows part. It ran 3 processes against `bench/out/run5c/ref`, which predates `clip_norm_only`, so its
norm-only rows are `no_ref` and only its `clip_grad_norm_full` rows (min 6.32, 5.96, 6.13 ms)
count. The reference was then written to `bench/out/2026-11-09-clipnorm/ref` (gitignored) by
`bench/torch_rows.py ref` under torch 2.12.1 MPS (`~/.venvs/ml`), and the 5 processes ran against
it under the lock (label `clipnorm-ref`, 13:21:55Z). `run.sh` now writes that reference itself when
it is missing.

## Fine-tuning path: `TapeStepProvider` on LoRA-2B

`gpu_memory_rows` arm `tape-provider-clip` (`QWEN35_MEM_TOKENS=2048`) runs Qwen3.5-2B-Base with
rank-16 LoRA on 150 sites, which gives 300 trainable tensors and 15,630,336 values. After one warm-up
step at `grad_scale` 1.0, it runs 10 rounds. Each round has one step at `grad_scale` 0.5, which runs
`scale_grad` on every gradient, and one at 1.0, which skips it. The order alternates per round. The
accumulate and AdamW phases are each synced before their clocks stop.

| arm | AdamW ms, median (min) | accumulate ms, median |
| --- | --- | --- |
| grad_scale 0.5 | 32.77 (21.39) | 6473.5 |
| grad_scale 1.0 | 21.31 (17.21) | 6529.0 |

The paired AdamW delta is +10.61 ms median (min -0.01, max +18.07), with 9/10 rounds slower. That is
0.162% of the 6548.6 ms median unscaled step. The losses differ between arms because the update
differs (`provider.jsonl`).

**Measured at `dd41efd`; `main` has moved since.** Merge `67c46bf` (ft-042351e4, in `da2e81e`)
reworked `TapeStepProvider::adamw_step`. It now scans the bank and runs `check_all` before the
scale. The scale loop itself is unchanged: one `scale_grad` per gradient, only when `grad_scale`
is not 1. So the paired delta above should still be the scale pass's cost, but the absolute AdamW
times are those of `dd41efd`, and this has **not been re-measured** at `da2e81e`. The trainer
and Metal clip path changed only in doc links between the two commits, so the nanolab rows are
unaffected.

**Inferred, not measured:** 15.6M values is about 125 MB of read plus write, about 0.5 ms at this
bandwidth. So the ~10 ms is mostly 300 per-tensor `Cmd::Scale` submissions, not bytes. A batched
multi-tensor scale on this path, like `ojas_scale_multi` on the clip path, would recover most of
it without any trait change. At 0.16% of a step it is not worth doing for speed alone either.

## Files

| file | holds |
| --- | --- |
| `run.sh` | the driver: trainer A/B, clip rows, provider A/B; run under `tools/mac_heavy.sh` |
| `env.txt` | git HEAD `dd41efd` (13 dirty files, the harness changes below among them), source sha1s, tessl HEAD, rustc 1.99.1, macOS 27.2, Apple M5 Pro, lock label |
| `runs.tsv` | one row per process: status, wall time, peak footprint, load before and after |
| `trainer.jsonl` | 6 trainer rows: per arm, the step seconds, losses and grad norms |
| `clip_rows.jsonl`, `clip_rows_normonly.jsonl` | the direct rows (see above for which count) |
| `provider.txt`, `provider.jsonl` | the provider test's output and its `ROW` lines (`preflight`, `first step`, `ok`) |
| `stderr/` | each process's stderr with `/usr/bin/time -l` |

Harness changes, uncommitted at the time of the run:

- `ojas-model/examples/nanolab_config_ab.rs`: new `clip` mode. After the run, one `spec.clone()`
  became `spec` (clippy `clone_on_copy`; `ModelSpec` is `Copy`) and the call was reformatted, so the file's sha1 no longer
  matches `env.txt`. The binary that ran is otherwise the same code.
- `ojas-qwen35/tests/gpu_memory_rows.rs`: new `tape-provider-clip` arm. Each provider step's phases
  are now synced separately.
- `ojas-qwen35/tests/common/mod.rs`: `reals`.

No production code changed.

Reproduce:

```bash
cargo build -j 2 --release -p ojas-model --example nanolab_config_ab
```

```bash
cargo build -j 2 --release -p ojas-metal --features metal --example metal_vs_torch
```

```bash
cargo test -j 2 -p ojas-qwen35 --release --test gpu_memory_rows --no-run
```

```bash
bash ../Lappi-decision/tools/mac_heavy.sh clipshare bash bench/results/2026-11-08-clip-scale-share/run.sh <out dir>
```
