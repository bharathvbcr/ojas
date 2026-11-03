# Rules: what bounds the flash forward, and the tracker's small-op baseline, 2026-11-03

**Written before job 3 ran.** `job3.log` prints this file's mtime first. Follows
`../2026-11-03-flash-fwd-specialize` (B and C: bit-identical, neither faster). This job changes no
kernel that is proposed for merge. It measures:

## Part 1: exp diagnostic (D1, never landed)

- **D1** = `3406d65` (A) plus `diag-fast-exp.patch`: the f32 flash body's two `precise::exp` become
  `fast::exp`. It is a diagnostic in the same sense as `../2026-11-02-splitk-regression`'s
  `diag-bf16-acc-false.patch`. It moves output bits by construction and is not a proposal.
- The binaries are `A` (job 1's, built in this worktree and target dir) and `D1` (built in this job
  from the same tree). `flash_fwd_probe time 3 15`, 6 rounds, order A D1 A2 and then reversed on
  odd rounds. `flash_fwd_probe hash` on D1 is recorded to show which cases moved.
- **Read:** on `x36400`, the per-round D1/A ratio against N = max |A2/A − 1|. `ctl` is the control,
  with the same 1.00 ± 0.05 bar as job 1.
  - every round < 1 − N: **the exp is a measurable share** of the kernel. The ratio bounds what any
    exp change could buy, and it costs bits.
  - otherwise: **exp is not the limit**.
- **Prediction:** D1 faster by 10-25%. Inferred from the per-score work at d = 32; not measured.

## Part 2: small-op baseline (no change, numbers for the report)

- `tracker_ops_probe 3 15`, 6 processes back to back, built at the branch tip. The library differs
  from `286a444` only in the bf16 flash entry, which no arm here runs. Arms and shapes are listed in
  the example's header. Per arm: the median over processes of each process's median (ms), the
  min-max of those medians, and the dispatches and pipelines of one call.
- **Q1 (permute):** the profile has the head swap `[1,36400,8,32]` at 1.4-1.6 ms. A residual add of
  `[1,36400,256]` (1.5x the bytes) takes 0.67 ms.
  - `perm_head36k` / `perm_ident36k` ≥ 1.5: the strided gather costs.
  - ≈ 1, and both ≥ 1.5x `resadd36k`: the row-copy kernel (`ojas_permute_runs`) itself is slow per
    byte, and a copy kernel with coalesced lanes is a bit-exact lever.
  - Otherwise: the profile's per-op sync and allocation explain the gap, and no kernel lever is
    shown.
- **Q2 (linears):** `linb36k − lin36k` against `badd36k`. This is the share of `img_k` / `ca_k` /
  `ca_v` that the separate bias pass costs. A fused epilogue would remove it; in tessl that is an
  exact-f32 NT epilogue kernel, whose bits are not established. The same for `lin5k_up` /
  `linb5k_up`.
- Nothing in part 2 is a speed claim.

## Part 3: test and lint

- `cargo test --release -p ojas-metal --test attention_flash_bits -- --test-threads=1` at the
  branch tip. It must pass: its golden hashes are A's, and B is bit-identical by job 1.
- `cargo clippy -p ojas-metal --release --all-targets --no-deps -- -D warnings` (the two new
  examples and the test are new `--all-targets` surfaces).
