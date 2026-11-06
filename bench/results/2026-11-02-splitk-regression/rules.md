# Split-K cross-attention regression: rules, fixed before the first timed row

Question. Which commit since 2026-10-27 slowed the split-K cross-attention
kernels (`ojas_xattn_part_d32` + `ojas_xattn_combine_d32`, reached through
`MetalBackend::attention_forward_saving_splitk`) at q `[1,8,5184,32]` x k,v
`[1,8,36320,32]`, f32, no bias? On record: 79.6–93.6 ms on 10-27
(`bench/results/2026-10-27-attention-stage2/bench-output.txt:33,75`,
`bench/results/2026-10-27-attention-mpp/a-at3g/attempt-{1,2,3,4,6}.txt`;
attempt 5 read 265.9 ms with its fc1 control at 3.91 TFLOP/s and pad-q at
3.78x its quiet anchor, so it is contention, not code), and 294.0 ms on 11-02
at HEAD 286a444 (`bench/results/2026-11-02-sam-xattn-q2/a-q2-m36320.txt:7`,
fc1 6.05 TFLOP/s, pad-q 1.05x its anchor).

Harness. `splitk_probe.rs` (copied here). One fixed source, dropped into every
build's `ojas-metal/examples/`, so only the library differs between binaries.
Arms: `fc1` (`linear_forward` f32 `[5184,1024] x [4736,1024]`, the GEMM
control), `split-K` (the arm under test) and `dispatcher`
(`attention_forward_saving` at the same shape, a same-shape control). A device
sync on both sides of every call; odd rounds walk the arms in reverse; each
process runs 3 warm-ups and 15 timed rounds. Before timing it prints split-K's
max abs difference from the dispatcher's output.

Builds. Each point is an ojas first-parent commit on main with the tessl
first-parent commit last committed at or before it (co-dated; tessl and ojas
path-depend on each other since 10-31). Both repositories are checked out as
detached worktrees under `/Users/bharath/Code/splitk-bisect/` (never in the
shared checkouts), built at `-j 2` into one target dir, and each binary is
copied out and hashed (`bins-manifest.tsv`). The timeline is
`timeline.tsv` (index, ojas, tessl, date, subject); index 6 (8b88328, the
merge that brought in 2b14dca, the code a-at3g timed) is the old point.

Pairing. The binaries of one comparison run back to back, one process each,
in an order that rotates every round (A B, B A, ...), several rounds, all
inside one `mac_heavy.sh` window. No cross-window comparisons.

Reading.
- Per binary, the split-K value is the median over rounds of the per-process
  split-K medians; its noise is the range (min to max) of those per-process
  medians and the within-process p90 - p10.
- A pair of points differs beyond noise when the gap between their split-K
  values exceeds the larger of the two binaries' per-process-median ranges
  and the larger within-process p90 - p10, and every round orders them the
  same way.
- A pair is read only if the fc1 medians of the two binaries agree within 10%
  in that window (else it is contended and rerun); the dispatcher arm is
  reported beside it.
- Reproduction gate: the first window must show the new point slow and the
  old point at its 10-27 level (roughly 80–95 ms). If the old point is also
  slow, the cause is machine state (OS, driver, GPU), not a commit; the
  bisect stops and that is the report.
- The first slow commit is the earliest timeline index whose split-K value
  is beyond noise above the old point's while its predecessor's is not. If
  ojas and tessl both move at that step, the cross builds (ojas_i x tessl_i+1,
  ojas_i+1 x tessl_i) separate them; a cross build that does not compile is
  reported as such.

What would refute a finding: the slow/fast split not repeating in a second
window; fc1 disagreeing between binaries; split-K output differing from the
dispatcher's beyond 1e-5 at the found commit (a different computation, not a
slower one).

No kernel or library change is made by this work.
