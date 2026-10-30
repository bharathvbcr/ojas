# Metal conv2d: bf16 lane and the 1x1 pointwise route (2026-10-30)

These are the results for ft-f3f95dbe, wave 8 Lane 1, merged as 86efb70. They come from `ojas-metal/examples/linear_gemm_bench.rs`, built with Metal Toolchain 27B47 (Xcode 27.2). The raw output is `bench-output.txt` in this directory.

The machine was moderately loaded throughout, with load averages of 7–10, so it was not quiet. Each process runs its arms interleaved. Times are min ms over n=3 processes unless stated.

## Verdict: the 1x1 route against the route it replaces

`[1,256,288,288]`, 256 to 256 with bias, 25 rounds a process:

| arm | dispatches | im2col | conv_nchw | min ms (over 3) | per-process medians ms |
|---|---|---|---|---|---|
| f32, new route | 6 | 0 | 0 | 3.739 | 4.64–4.77 |
| f32 via im2col (old) | 7 | 1 | 1 | 7.592 | 8.85–10.57 |
| bf16-exact, new route | 8 | 0 | 0 | 2.912 | 3.42–4.00 |
| bf16-exact via im2col (old) | 9 | 1 | 1 | 6.136 | 7.59–9.12 |

- The new route is 2.03x faster in f32 and 2.1x in bf16. The per-process min ranges do not overlap (3.74–4.12 against 7.59–8.23).
- The `COUNT` lines show the route actually ran: the new arm dispatches no im2col and no layout pass, plus one bias add (`bcast 1`).
- In this raw output the new-route rows carry the labels from before this commit, "conv2d f32 (im2col + GEMM + layout)" and "conv2d bf16-exact (im2col + bf16 GEMM + layout)". The example now labels them by what `conv2d_forward` does.
- Parity:
  - Output is bit-identical to the im2col route at 256x24², 256x72², c=1, o=37, o=1, one pixel and batch 3.
  - Where tessl splits K (c=1024 over 8192 pixels), the output is at most 9.5e-6 from the CPU, within the f32 sum bound.
  - Tests: `ojas-metal/tests/conv2d_pointwise.rs`.

## Criterion 1 rows

- **256-wide linear `[5184,256]x[256,256]`:**
  - f32 0.314; f32 + bias 0.328;
  - bias add alone 0.158;
  - bf16 0.234; bf16 + bias 0.261.
- **Command overhead:** sync only 0.002; one minimal command 0.120; sixteen commands then one sync 0.190. That is about 5 µs per extra queued command on about 0.12 ms of fixed commit-and-wait.
- **bf16-exact conv against f32 conv:** 144² −8%, 72² −13%, patch embed −40%, 1x1 −22%.
  - The 288² (−5%) and 1152² (+6%) rows of the first run are perturbed by the example's resident set (about 7 GB of bf16 im2col copies). Do not read them.
- **Memory encode `[1,32,1152,1152]`, 3x3 p1, 32 to 16:** f32 97.6 and bf16 103.7. The write-floor probe in the same process ran about 6x slower than expected, so this row is a memory-pressure artifact [inferred] and not a measurement of the kernels.

## No regression on the existing conv rows

The second job ran these rows from the old bench source, so the process holds the same tensors as base. They use n=5, interleaved base / new / control, where control is base run again.

| row | base min ms | new vs base | control vs base |
|---|---|---|---|
| neck 3x3 288² | 26.859 | +2.2% | +1.5% |
| neck 3x3 144² | 6.175 | −0.2% | +1.2% |
| neck 3x3 72² | 1.670 | −1.1% | −0.3% |
| patch embed | 1.640 | −0.9% | −0.2% |

- `attention_flash_bench` (flash arm, best-of-3 p10, the statistic that bench prints): base 13.72, new 13.76 (+0.3%), control 13.00.
- Round-to-round spread there reached 17% and the bench's own verdict was `repeat`, so it cannot resolve 5%. The attention path is untouched by this change.

## Not measured

- **The SpikeTrack in-frame re-profile (card criterion 4):** it needs an ojas pin bump, which is an owner decision. The peer's 16.2 ms in-frame figure for the old route does not reproduce in isolation (7.6–10.1 ms here). See ft-580e6c40 for the in-frame slowdown.
- **A torch comparison:** not run.
