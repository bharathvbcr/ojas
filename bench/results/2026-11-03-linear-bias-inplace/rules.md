# Rules: f32 linear_bias_forward adds its bias in place (F), 2026-11-03

**Written before job 4 ran.** `job4.log` prints this file's mtime first.

## The change

- **F (`aade8fa`):** under the f32 lane with a `[out]` bias, `linear_bias_forward` runs the
  exact-f32 GEMM and then `ojas_bias_rows`, which adds the bias in place and checks each sum.
- **Before:** `linear_forward` (scans of x and w, the GEMM, a scan of y), then
  `broadcast_add_forward` (a generic one-thread-per-element add into a second buffer).
- **Saved:** the y scan, the second 37 MB buffer, and the broadcast's per-element div/mod.
- **Bits:** the add is the broadcast kernel's f32 `a + b`, with `-ffp-contract=off`, so they
  should not move.
- **Who uses it:** SpikeTrack's projections (`img_k`, `ca_k`, `ca_v` over ~36,400 memory tokens;
  q/k/v/out, `img_q`, `ca_q`, `ca_out` and the FFN over 5,184 tokens), and every other f32 linear
  with a bias in SpikeTrack, the detector's included.
- **P (`c1676d6`):** the commit before F. Its library differs from `286a444` only by the bf16
  flash entry, which no f32 call runs (`../2026-11-03-flash-fwd-specialize`).

## Gates, fixed now

1. **Bit identity, op level (decides alone).** `tracker_ops_probe hash` on P and F prints the same
   hash for `lin36k`, `linb36k`, `lin5k_up`, `linb5k_up` and `linb5k_down`.
2. **Tests at F**, `--release --test-threads=1`, all must pass:
   - `linear_bias_fused`: bit-equal to the composed ops at SpikeTrack's shapes and ragged ones,
     dispatch identity, overflow faults;
   - `attention_flash_bits`;
   - `encoder_ops`, `encoder_ops_edges`;
   - `bf16_accumulation` (the bf16 lane of the same op);
   - `autocast_precast_weight`;
   - `attention_flash`, `attention_routing`.

   Then `cargo clippy -p ojas-metal --release --all-targets --no-deps -- -D warnings`.
3. **Speed, op level.** `tracker_ops_probe 3 15`, 8 rounds of P, F and P′ (P's binary again),
   rotated, odd rounds reversed. Per arm, the per-round F/P ratio of process medians.
   - N = max |P′/P − 1| on that arm.
   - **faster** if every round's ratio < 1 − N; **slower** if every round's ratio > 1 + N;
     otherwise **not measured**.
   - Read on `linb36k` and `linb5k_up`.
   - **Controls:** `lin36k` and `resadd36k` (code F does not touch). Their median F/P must lie in
     1.00 ± 0.05, else the window is `repeat` and the times are CONTENDED.
4. **End to end.** SpikeTrack `38e0a35` (export at `/Users/bharath/Code/kern1103/SpikeTrack`),
   built against P and against F in one tree and target dir. The runs follow the earlier rules'
   "End to end" section (`../2026-11-03-flash-fwd-specialize/rules.md`, `e2e_summary.py`, here with
   base P in the role of A):
   - an unprofiled F control;
   - P, F / F, P / P, F with `--profile stages`;
   - an unprofiled P control.
   - **E1 (decides alone):** every `82.json` byte-identical to the F control.
   - **E2:** memory-attention stage and tracker step, with the same faster / slower /
     not-measured rule.

## Predictions, fixed now

- Gate 1 holds; every test passes.
- `linb36k`: **faster**, to ~1.35-1.50 ms from ~1.68 (−10% to −20%). `linb5k_up`: **faster**, by a
  similar fraction. The controls do not move.
- E1 holds.
- E2: **not measured** on both rows. 12 projections over 36k tokens plus about 30 at 5,184 tokens
  per frame save ~4-6 ms of a ~285 ms stage, inside its noise. The op-level rows are where F is
  read.
