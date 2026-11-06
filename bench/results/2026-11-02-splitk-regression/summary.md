# Split-K cross-attention regression: bisected to d92a2e7

**Finding.** Split-K at q `[1,8,5184,32]` x k,v `[1,8,36320,32]`, f32, no bias, went
from about 97 ms to about 314 ms (3.2x) at **d92a2e7** ("feat(metal): opt-in bf16
accumulation, pool-ledger refusal, flash bias and cross-attention, in-kernel
finiteness", 2026-11-02 12:47). Every sampled point before it (i006, i091, i168,
i168x) reads the old level. From 8b88328 to fcad865 (d92a2e7's first parent), no diff
line in `ojas_backend.metal` touches the `xattn` kernels, and in `device.rs` only one
error-message string near the split planner changed. Every point after it through
HEAD 286a444 is the same binary. The output is unchanged: split-K's max abs difference from
the dispatcher is 1.25e-7 in every binary. It is the same computation, slower.

**Mechanism (supported by a diagnostic build).** d92a2e7 adds a runtime
`bf16_acc = (flags & 32u) != 0u` to `ojas_xattn_part_d##DD`. It branches on it inside
the per-key loop and inside the per-element `for c < DD` loops over `qv[DD]` and
`acc[DD]`. One build changes only that line to `const bool bf16_acc = false;`
(`diag-bf16-acc-false.patch`; diagnostic only, not a proposed fix). It brings split-K
back to 97.6 ms, the old level. The combine kernel's own new branch
(`splits_raw & 0x80000000u`) is not part of the cost: patching the part kernel alone
recovers everything. Why the uniform branch costs 3.2x is **inferred, not
verified**. The likely cause is that the compiler no longer unrolls the
`c`-loops, so `acc[]`/`qv[]` get dynamic indexing and spill out of registers. No
compiler statistics were read. The host side is correct: with bf16 off, `flags` bit 5
is 0 and `splits` carries no high bit.

## Bisect (all numbers measured, `splitk_probe` 3 warm-ups + 15 rounds per process)

Values are per-process split-K medians, then their median, over rounds that rotate
the order of the binaries back to back inside one lock window. Noise is the range of
per-process medians and the largest within-process p90 - p10.

| point | ojas | tessl | split-K median (range), ms | max p90-p10 | window |
|---|---|---|---|---|---|
| i006 | 8b88328 (10-28 merge of 2b14dca, the code a-at3g timed) | 789a132 | 97.6 (92.5–99.1) | 15.1 | 1 |
| i091 | 3ac4428 (before the 10-31 allocator / pool-ledger wiring) | 3c1f338 | 94.8 (90.8–100.5) | 23.8 | 1 |
| i168 | fcad865 (d92a2e7's first parent) | d2c3ea8 | 98.9 / 98.6 / 98.3 | 26.9 / 33.6 / 17.5 | 1 / 2 / 3 |
| i168x | fcad865 | **80caf39** | 97.2 (93.5–99.9) | 14.5 | 3 |
| **i169** | **d92a2e7** | 80caf39 | **314.6 (311.6–320.7)** | 47.4 | 3 |
| i169diag | d92a2e7, `bf16_acc` constant false in the part kernel | 80caf39 | 97.6 (93.7–98.5) | 16.0 | 3 |
| i175 | a1819b3 | 80caf39 | 317.1 (310.3–318.3) | 51.0 | 3 |
| i176 | 286a444 (HEAD) | e0ea9a0 | 309.9 / 316.1 / 314.3 | 54.8 / 69.0 / 69.1 | 1 / 2 / 3 |

- **The decisive step** is i168x → i169: same tessl, only ojas moves from fcad865 to
  d92a2e7. Gap 217 ms against a noise bar of 47.4 ms. All 6 rounds order them the
  same way. Beyond noise by `rules.md`.
- **tessl does not move it.** i168 → i168x changes only tessl (d2c3ea8 → 80caf39):
  98.3 → 97.2. i169, i175 and i176 (HEAD with tessl e0ea9a0) are byte-identical
  binaries (sha256 prefix `7a9d236bc0eae9f0`, `bins-manifest.tsv`).
- **Neither do the 10-31 merges or anything else between 10-27 and d92a2e7**: i006,
  i091 and i168 agree within noise.
- **Agreement with the record:** the old point reads 92.5–99.1 ms here, against
  79.6–93.6 ms on 10-27. The new point reads 307–322 ms, against 294.0 ms on 11-02.
  Both were measured under 1-minute load from 4.3 to 33 (windows 1–3), so the
  absolute values sit a few percent above the quiet records. The ratio is the stable
  quantity.

## Build caveat: co-dating

d92a2e7 does not compile against its co-dated tessl d2c3ea8. It calls
`GpuRuntime::alloc_internal_buffer` and `SharedBuf::to_gpu_buffer`, which tessl added
in 80caf39 two minutes later (window 2's three builds failed on this). The two
repositories must move together at this commit. The cross build i168x is what
separates them.

## Controls and deviations from rules.md

- **The fc1 GEMM control is not usable in this probe.** It read 13–33 ms on every
  binary, against about 8 ms when the flash bench runs it in its own row. Here each
  fc1 call sits between 50–320 ms attention calls, so it likely runs in another GPU
  clock state. By the rule as written ("fc1 within 10% across binaries") no window
  qualifies. The finding instead rests on:
  - the same-shape `dispatcher` arm, which agrees across binaries within its noise
    (window 3: 53.8 to 58.9 ms, ranges 5–7 ms, p90 - p10 up to 41 ms);
  - non-overlapping split-K ranges in every round of three windows.
- **The dispatcher (flash) arm moves about +5 ms at d92a2e7** (53.8 → 58.9 in
  window 3; i169diag, whose flash kernel is unpatched, also reads 58.2). That is
  within this probe's noise. It is consistent with the flash kernel taking similar
  runtime bf16 branches, but not established.
- Window 1's first attempt built only i006: Cargo rewrote the worktree's `Cargo.lock`
  gusset entry (0.0.2 → 0.0.3, a live path dependency ojas-metal does not use), and
  the next checkout refused. `build.sh` now restores tracked files before each
  checkout (`logs/job1-attempt1.log`).

## Who else is affected (from the handover, not re-measured)

Every split-K call runs the slowed part kernel whenever bf16 accumulation is off,
which is the default, so these callers likely pay a similar cost. Only the
36320-key, D32, no-bias shape was measured here, so the factor at other shapes is
unknown. The callers: biased calls under 56 threadgroups, calls with key lengths,
head dim 16, and short-Q calls under 384 groups. `ojas_cached_attn` gained the same
`splits_flag` packing in d92a2e7. Whether its kernel also slowed was not measured.

## Options for the owner (not implemented; kernels unchanged by this work)

1. Make `bf16_acc` a compile-time choice for the split-K part kernel: a Metal function
   constant, or two macro instantiations selected on the host. No kernel in
   `ojas-metal/kernels/` uses function constants today. Every bf16 switch is a
   runtime flag (38 `bf16_acc ?` sites in `ojas_backend.metal`), so other kernels may
   carry the same cost.
2. Unswitch by hand: one key loop for f32 and one for bf16, chosen once before the
   loop.

The diagnostic build shows that a compile-time-constant `bf16_acc` restores the old
time at this shape. It forced `false`; no function-constant pipeline was built. A parity and bf16-accumulation test run would be needed before either lands.

## Files

- `rules.md`: written before the first timed row.
- `scripts/`: the probe (`splitk_probe.rs`), build and pairing scripts, window jobs, `tabulate.sh`.
- `timeline.tsv`: ojas first-parent commits with co-dated tessl.
- `bins-manifest.tsv`: label, ojas, tessl, sha256 prefix.
- `diag-bf16-acc-false.patch`: the one-line diagnostic change.
- `logs/`: window logs (cargo progress lines stripped).
- `window{1,2,3}-table.tsv`: per-binary tables.
