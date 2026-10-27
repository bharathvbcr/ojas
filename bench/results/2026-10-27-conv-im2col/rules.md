Decision rules, fixed before the first measurement (committed ahead of the results). No timing of a new conv2d path had been seen when this file was written. The only timing seen on this branch is the old path's, from `bench/results/2026-10-27-linear-gemm` (neck 3x3 288^2: conv2d 53.7 ms, GEMM alone 17.0 ms).

Question. Can the im2col and layout passes of `MetalBackend::conv2d_forward` be cut from 36.7 ms to the gate below at the neck 3x3 p1 shapes, with the output unchanged bit for bit?

Pre-stated gate, verbatim from the coordinator's brief (Fable's ruling, scratchpad/fable-wave3.md):

- Correctness first. The new path's conv2d output is bit-identical to the current path's: same GEMM, same input bits. Check at the neck 3x3 p1 at 72^2, 144^2 and 288^2, the patch embed k14 s14, the 1x1 convs, the strided/padded memory-encoder convs, depthwise (if it shares the path), odd shapes, and groups > 1. Also check the dispatch count is as designed, faults are unchanged, and the budget/scratch charge is right and refuses cleanly.
- Speed. im2col + layout <= 12 ms at 288^2 and <= 4 ms at 144^2, measured as conv2d total minus GEMM alone, on the interleaved harness (5 warm-ups and 25 rounds, p10/p90), with the old path as an arm in the same run.
- If you miss the gate, report the numbers and where the time goes, and don't commit a slower or unproven path.

How the gate is measured (set here, before timing):

- Harness: `ojas-metal/examples/linear_gemm_bench.rs`, conv rows, `5 25`. Within a process every arm of a shape runs once per round, in a fixed order (interleaved), so conv2d and "GEMM alone f32" see the same machine state. The metric is median(conv2d) - median(GEMM alone), with p10 and p90 of each arm printed beside it. The gate is judged on the medians; the p10/p90 columns say whether the pass or fail is inside the noise.
- Old path as an arm: the old path's code is deleted in the same change, so it cannot be an arm of the new binary. The old arm is the binary built from this branch's base (928d443 plus the probe arms of this example, no routing change) and kept at `backups/78-tools/a-work/linear_gemm_bench-old`. Old and new processes are alternated, old new old new old new, each running the identical command; all six runs are reported. This is weaker than an in-process arm: the arms of the old and new binaries do not share a round. The summary says so beside every comparison.
- Probes (profiling only, not gate inputs): the same im2col matrix with 8 output channels, the thin GEMM of it, the write floor of the matrix and of the output (`fill`), and the product's transpose (`permute`). They say which pass dominates; they do not decide the gate.
- Bit-identity: the new conv2d output is compared with a reference formed from public ops on the same inputs (the CPU's im2col matrix, `linear_forward` on the exact-f32 lane, the layout and a separate bias add), by bits, at every shape listed in the gate. The old path's bits are taken from the old binary on the same inputs when a run needs them.

What would refute the result: a correctness case differing in any bit; the new path's conv2d minus GEMM-alone above 12 ms at 288^2 or above 4 ms at 144^2 on the median of the interleaved runs; the old arm's conv2d minus GEMM alone drifting from 36.7 ms by more than its own p10 to p90 spread between the old runs (then the machine, not the code, moved, and the run is repeated).
