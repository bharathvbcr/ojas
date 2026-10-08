# Saved-sigmoid gate pair A/B (2026-10-07, M5 Pro)

`bench/gate_saved_ab.rs`, run through the ignored
`bench_gate_saved_against_recomputed` tests of `ojas-metal/tests/gate_saved.rs`
and `ojas-wgpu/tests/gate_saved.rs`. It compares the plain per-head gate pair
(`per_head_sigmoid_gate_forward`, then `per_head_sigmoid_gate_backward`, which
recomputes the gate logits and sigmoid) with the saved pair
(`per_head_sigmoid_gate_forward_saving`, then
`per_head_sigmoid_gate_backward_saved`, which reads the saved scale). The shape
is nanolab's: 4096 rows, d_model 768, 12 heads of 64. Each sample is one call
plus `Backend::sync`. Each round runs 6 × 20 iterations and alternates which
side of each pair goes first.

Environment: ojas 86a1096 (both benchmarked test files unmodified),
tessl 4e5faac (clean), rustc 1.99.0, macOS 27.0.1, Apple M5 Pro, release build.

| File | What it is |
|---|---|
| `metal-1.txt`, `metal-2.txt`, `wgpu-1.txt`, `wgpu-2.txt` | Rounds 1–2 (`run.sh`, log `run.log`). These started right after the release build, at 1-minute load 23.8, so they are noisy. |
| `metal-3.txt`, `metal-4.txt`, `wgpu-3.txt`, `wgpu-4.txt` | Rounds 3–4 (`run-quiet.sh`, log `run-quiet.log`). These ran from the prebuilt binaries, with no compile alongside, at 1-minute load 5.8–6.2. |

Ratio of saved to plain (below 1 means the saved side is faster), min / median
per round:

| Backend | Round | forward_saving / forward | backward_saved / backward |
|---|---|---|---|
| Metal | 1 (noisy) | 1.059 / 0.988 | 0.925 / 0.963 |
| Metal | 2 (noisy) | 1.077 / 0.994 | 0.911 / 0.913 |
| Metal | 3 | 1.010 / 1.017 | 0.928 / 0.938 |
| Metal | 4 | 0.998 / 1.007 | 0.926 / 0.939 |
| wgpu | 1 (noisy) | 1.007 / 0.995 | 0.867 / 0.885 |
| wgpu | 2 (noisy) | 0.974 / 1.007 | 0.868 / 0.858 |
| wgpu | 3 | 1.016 / 1.012 | 0.847 / 0.852 |
| wgpu | 4 | 1.003 / 1.011 | 0.843 / 0.857 |

Reading, from the quiet rounds:
- The saved backward is faster on both backends:
  - Metal: about 7% (min 696.5 → 646.2 µs, 703.5 → 651.4 µs).
  - wgpu: about 15% (min 1541.0 → 1305.8 µs, 1538.0 → 1296.7 µs).
- Saving the scale costs the forward at most about 2% at the median. The noisy rounds put Metal's min 6–8% higher, but the quiet rounds do not repeat that.

The pair's correctness is pinned by
`the_saved_pair_is_the_plain_pair_bit_for_bit_and_matches_cpu` in both test
files (the saved pair equals the plain pair bit for bit, and matches
`ojas-cpu`).

To repeat: `cargo test --release -p ojas-metal --test gate_saved -- --ignored
--nocapture bench_`, then the same with `-p ojas-wgpu`, one at a time and with
no build running.
