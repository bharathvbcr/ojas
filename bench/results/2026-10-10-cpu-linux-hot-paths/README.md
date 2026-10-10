# CPU hot paths on Linux x86_64: interleaved A/B (2026-10-10)

**Base** is `edeaf73` with this tree's `ojas-cpu/tests/bench_ops.rs` copied
in, so both binaries run the same cases. **After** is this tree: the one-row
GEMV path, the fused cross-entropy tile reuse and row split, the pooled KV
prefix scan and decode heads, the reserved-chunk `add` forward, the parallel
embedding-gradient zero-fill and the GDN log-decay head split.

Linux x86_64, 4 cores, no `target-cpu` flags (so `gemm::fma` is a multiply
and an add), rustc 1.97.0, release, Fast numerics. 1-minute load was
1.4–1.6 throughout (`ab/load.txt`). Three rounds, odd rounds base then
after, even rounds after then base; each cell is a min-of-N inside one
process after two warm-up calls, and `summary.md` takes the minimum over
the rounds. Ratios between 0.9 and 1.1 are not changes: `mul`, `add`
backward and `embedding` forward are untouched controls.

## Reproduce

```bash
# in each tree (base: edeaf73 plus this tree's bench_ops.rs):
cargo test -p ojas-cpu --release --test bench_ops --no-run
bash run.sh BASE_BENCH_OPS AFTER_BENCH_OPS 3 OUT_DIR
python3 summarize.py OUT_DIR > summary.md
```

## Files

| file | what |
| :-- | :-- |
| `run.sh`, `summarize.py` | the runner and the table builder (analysis only) |
| `ab/` | 3 rounds of `bench_ops` per lane at 4 and 1 threads, and `load.txt` |
| `summary.md` | minimum over rounds, base against after |
| `bits.txt` | every case dumped by both binaries at 4 threads, Fast: 65 files, 0 differ |

## Not measured here

macOS (Accelerate and vDSP keep their own paths; the decode linears there
were already one Accelerate call under Fast, so only Exact takes the GEMV)
and the trainer's default cross-entropy chunk (1024 x 8192), which keeps
the three-tile passes and gains only the row split (`linear_ce`, 0.88).
