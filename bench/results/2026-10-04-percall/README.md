# Where the ~0.04 ms per extra decode call goes, 2026-10-04

**Question.** In the sweep (`../2026-10-04-sweep/`), 16 decode requests cost
Metal about 0.04 ms more per request as 16 calls than as one batched call.
What is that time?

**Run.**
- `metal_bench 20 percall` (`ojas-metal/examples/metal_bench.rs`, the
  `percall` group). Driven by `target-baseline/percall-run.sh` under the
  Lappi lead's lock, 16:02:29–16:03:04Z.
- Apple M5 Pro. 500 runs per scenario after 20 warm-ups.
- Before the run the GPU read 71% busy from other work. The 95% reading at the
  end was taken just after this run, so it cannot be attributed. Load was
  9.3–9.5.
- Raw output: `percall.md`.
- Shape: 16 requests, 12 heads of 64, 1024 cached positions each.

## Answer: mostly GPU time, not host bookkeeping

Medians, 16 calls against 1 batched call:

| part | 16 calls | 1 batched call | extra | share of the extra |
| :-- | --: | --: | --: | --: |
| whole run, call to sync return | 1.349 ms | 0.725 ms | 0.624 ms (0.042 per extra call) | 100% |
| host time inside the calls | 0.126 ms | 0.016 ms | 0.110 ms (7 µs per call) | about 18% |
| GPU time of the work (tessl directly, same kernel) | 1.050 ms | 0.562 ms | 0.487 ms (32 µs per call) | about 78% |

The GPU rows come from a separate tessl runtime running the same
`ojas_cached_attn` kernel. The shares are therefore approximate, and they
leave about 4% unassigned.

**1. The GPU part (verified).**
- One request on its own is a dispatch of 12 threadgroups (one per head). It
  reads 6.3 MB of cache in 34.5 µs (min), about 182 GB/s.
- 16 requests in one dispatch read 101 MB in 394 µs, about 256 GB/s. That is
  the same ceiling the elementwise kernels reach, so the batched call is
  limited by memory bandwidth.
- The same 16 requests as 16 dispatches take 718 µs (min), about 140 GB/s,
  against 394 µs batched. The reading (**inferred**): each small dispatch
  leaves most of the GPU idle, and the 16 do not run concurrently enough to
  fill it.
- Recording them with no barrier between them did not change this:
  718.7 µs min, 1037.9 µs median. Whether that scope really dropped the
  barriers was not counted, so "barriers are not the cause" is
  **unverified**. One alternative is that the GPU does not overlap them
  anyway.

**2. The host part (verified).**
- Each `MetalBackend` call costs about 7–8 µs of host time
  (0.126 ms over 16 calls).
- Recording the same 16 dispatches straight into tessl costs 1.2 µs each
  with a reused output buffer. With a fresh output per call, as
  `MetalBackend` does, it costs 1.8 µs. That includes 16 residency flushes
  and 16 pool allocations.
- So allocation and residency are about 0.6 µs per call. The rest of the
  host time, about 6 µs, sits between the caller and tessl: the channel to
  ojas's device thread, validation and bookkeeping. That last split is
  **inferred**; it was not sampled.

**This corrects an inference made earlier the same day.** The sweep README
and `docs/bench-gpu-vs-torch.md` had guessed that residency re-registration
was the likely cause. It is about 0.6 of the 42 µs.

## What would move it

- **Batch at the caller.** One call with B requests reaches the bandwidth
  ceiling. This is what the measurement says to do.
- **Fill the GPU with one request.** Today a decode is one threadgroup per
  head, so 12 threadgroups. Splitting the 1024 positions across several
  threadgroups per head (split-K over the cache, then a combine step) should
  raise single-request bandwidth toward the batched figure. **Not tried.**
- **Host path.** About 6 µs per call. It is worth shrinking only after the
  GPU part.

## Not measured

- The barrier counter for the no-barrier scope.
- Why torch's separate calls cost it only about 0.012 ms extra each
  (`../2026-10-04-sweep/fit.md`).
- A run on an idle GPU.
