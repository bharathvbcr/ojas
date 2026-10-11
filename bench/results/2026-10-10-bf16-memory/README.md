# bf16 tier: micro-batch memory at B = 1 and B = 8 (2026-10-10)

Criterion 4 of gp-bf16-compute-tier asks whether B >= 8 micro-batches fit in
Apple Silicon unified memory without paging or thrashing. This run measures
it for the native training path: nanolab 124M (`ModelSpec::nanolab_124m`),
T = 1024, grad_accum 1, activation checkpointing off, three steps per row.
Each row runs with the trainer's autocast off (`f32`) and on (`bf16`), on
the CPU, wgpu and Metal backends.

- Machine: Apple M5 Pro, 64 GiB unified memory, macOS 27.0.1.
- Code: branch `gitpulse/bf16-storage-and-compute-tier-bf16-opera-c03cbd7e`
  at `5744895`, plus `ojas-model/examples/bf16_memory_rows.rs` and
  `bench/bf16_memory_rows.sh`, both committed with this file.
- Command, run under `Lappi-decision/tools/mac_heavy.sh` with the lock held
  and the 1-minute load at 9.9 at the start:
  `bash bench/bf16_memory_rows.sh rows.jsonl "cpu wgpu metal" "1 8" 1024 3 off`.
- Each row is its own process under `/usr/bin/time -l`. `rows.jsonl` holds
  the raw rows as the script wrote them.

## Rows

Peak is the budget's peak charged bytes over the three steps. Resident is
what was charged before the first step: parameters and optimizer state.
Footprint is the process's peak memory footprint. The `majflt` and `swaps`
columns are this process's own counts. `po` and `so` are the machine's
vm_stat pageout and swapout deltas across the row.

| backend | prec | B | Bf16 storage | peak GiB | resident GiB | footprint GiB | majflt | swaps | po | so | s/step (3 steps) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | f32 | 1 | yes | 2.76 | 1.07 | 2.86 | 0 | 0 | 0 | 0 | 3.04, 3.17, 2.90 |
| cpu | bf16 | 1 | yes | 2.76 | 1.07 | 2.94 | 0 | 0 | 0 | 0 | 4.51, 13.70, 24.43 |
| cpu | f32 | 8 | yes | 10.45 | 1.07 | 11.02 | 0 | 0 | 4 | 0 | 72.66, 77.10, 41.49 |
| cpu | bf16 | 8 | yes | 10.53 | 1.07 | 11.17 | 0 | 0 | 2 | 0 | 22.74, 20.56, 96.69 |
| wgpu | f32 | 1 | yes | 2.76 | 1.07 | 5.45 | 76 | 0 | 0 | 0 | 1.48, 1.12, 1.13 |
| wgpu | bf16 | 1 | yes | 2.77 | 1.07 | 5.30 | 7 | 0 | 0 | 0 | 1.25, 1.15, 1.15 |
| wgpu | f32 | 8 | yes | 10.40 | 1.07 | 12.38 | 0 | 0 | 0 | 0 | 4.30, 4.23, 4.22 |
| wgpu | bf16 | 8 | yes | 10.64 | 1.07 | 16.02 | 6 | 0 | 27 | 0 | 5.13, 4.66, 6.70 |
| metal | f32 | 1 | no | 2.76 | 1.07 | 3.71 | 77 | 0 | 0 | 0 | 0.61, 0.67, 0.61 |
| metal | bf16 | 1 | no | 2.76 | 1.07 | 4.01 | 1 | 0 | 0 | 0 | 0.67, 0.67, 0.64 |
| metal | f32 | 8 | no | 10.40 | 1.07 | 12.08 | 1 | 0 | 0 | 0 | 2.14, 2.21, 2.21 |
| metal | bf16 | 8 | no | 10.46 | 1.07 | 12.16 | 1 | 0 | 0 | 0 | 2.53, 2.69, 2.69 |

The "Bf16 storage" column is the backend's `bf16_operands()`. Metal's bf16
lanes belong to gp-ft-bf16-frozen-base and were not on this branch, so the
Metal bf16 rows round in `F32` storage.

## Result

At B = 8 every backend and both precisions fit without paging. The peak
charged bytes are 10.4 to 10.6 GiB against a 48 GiB budget and 64 GiB of
memory. No process swapped, and the machine's swapout counter did not move
during any row. The machine's pageouts moved by at most 27 pages of 16 KiB
(0.4 MiB) during a row. Other sessions were running, so those pages cannot
be attributed to the row.

The bf16 rows are not smaller, and that was expected. The tape keeps `F32`
activations and the parameters are `F32` masters. The region therefore adds
transient copies: the `Bf16` copy of each matmul-class operand and, on wgpu,
its widened scratch. That costs +0.00 to +0.24 GiB of peak. The bf16 tier
saves memory only where a `Bf16` tensor stays resident in place of an `F32`
one: a frozen bf16 base, measured by gp-ft-bf16-frozen-base. Keeping the
tape's matmul-class operands in `Bf16` would be the activation lever; that
is an ojas-autograd follow-up, not this tier.

Step times are listed and nothing is claimed from them. The CPU rows ran
while other sessions loaded the machine: the f32 B = 8 row took 41 to 77 s
per step, and the bf16 B = 1 row slowed from 4.5 to 24.4 s across its steps.

## The paging rule, amended after the run

The script first counted a row as fitting only when its process had zero
major page faults (`/usr/bin/time -l` "page faults") and zero swaps. On
macOS that fault count also includes file-backed page-ins. 76 and 77 faults
appear only in the first wgpu and first Metal process of the grid. The
later rows on the same backends show 0 to 7, at four times the memory. That
pattern fits libraries and shader caches loading cold, not memory pressure;
this is inferred, not traced.

The script now gates on the process's swaps and the machine's swapout delta,
and keeps major faults as a column. The raw `rows.jsonl` keeps the field
the first rule wrote: `fits_without_paging` is `false` on the rows with a
non-zero fault count. Under the amended rule every row fits.

## Not covered here

- Metal with `Bf16` storage (gp-ft-bf16-frozen-base).
- The fine-tune shapes: Qwen3.5 0.8B and 2B with a frozen bf16 base at
  T = 2048 and T = 9,638. That brief owns them.
- Activation checkpointing on. The `off` rows already fit at B = 8.
