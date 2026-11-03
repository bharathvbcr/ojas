# What bounds the flash forward, and the tracker's small-op baseline, 2026-11-03

Rules: `rules.md`, written before the job (its mtime, 10:44:30 local, heads `job3.log`). The job
ran 17:29-17:30Z after 44 minutes waiting for the lock. One-minute load was 7-8. Files:
`time-d1.tsv`, `hash-D1.txt`, `hash-diff-D1.txt`, `ops.tsv` → `ops-summary.txt`, `bins.tsv`,
`build-*.log`, `diag-fast-exp.patch`.

## Part 1: the exp is not what bounds the d = 32 flash forward

D1 is A with `fast::exp` in place of the f32 body's two `precise::exp` calls
(`diag-fast-exp.patch`; never landed). Six rounds, A / D1 / A2 interleaved, `flash_fwd_probe time
3 15`:

| arm | A medians (ms) | D1/A per round | N (max \|A2/A − 1\|) | verdict |
|---|---|---|---|---|
| `x36400` | 47.4-48.5 | 0.961 0.987 1.009 1.004 0.970 0.984 | 0.097 | inside the noise |
| `x15600` | 19.2-19.6 | 0.965 0.978 0.990 0.988 0.975 0.997 | 0.094 | inside the noise |
| `ctl` | 5.53-5.57 | 0.996-1.050 | 0.078 | (control, in bar) |

- Kernel identity held: every flash call encoded `ojas_attn_fwd_d32`.
- D1 moves the output bits on all 9 f32 cases and none of the 3 bf16 cases (`hash-diff-D1.txt`),
  so the patched path did run.
- The prediction (10-25% faster) **failed**. D1/A sits around 0.98 and every round is inside N.
  So a cheaper exp would buy at most a few percent, at the cost of every output bit.
- With job 1 (`../2026-11-03-flash-fwd-specialize`) this rules out three suspects for the kernel's
  ~3.7 TFLOP/s at d = 32:
  - K/V re-reads (C: 64-row tile, no change);
  - the runtime bf16 branch (B: no change);
  - the exp (D1: inside the noise).
- What is left, **inferred and not measured**, is the structure:
  - two small `matmul2d` per 64-key block (32x64x32 and 32x32x64);
  - the S tile's store and load through threadgroup memory;
  - three barriers per block.

  Changing any of these changes the order of the row sums, and so the bits.

## Part 2: small-op baseline (6 processes, median of medians, range, min, ms)

| arm | median | range | min | dispatches (runs / generic / check) |
|---|---|---|---|---|
| `perm_head36k` (head swap, `[1,36400,8,32]`) | 0.462 | 0.437-0.481 | 0.418 | 1 (runs 1) |
| `perm_ident36k` (identity copy, same tensor) | 0.454 | 0.442-0.489 | 0.414 | 1 (runs 1) |
| `perm_generic36k` | 0.718 | 0.701-0.757 | 0.675 | 2 (generic 1, check 1) |
| `perm_head5k` | 0.220 | 0.214-0.242 | 0.172 | 1 (runs 1) |
| `resadd36k` | 0.607 | 0.593-0.642 | 0.538 | 1 |
| `lin36k` (`[36400,256] x [256,256]`) | 1.404 | 1.373-1.464 | 1.332 | 4 (check 3) |
| `linb36k` (+ bias) | 1.684 | 1.658-1.736 | 1.588 | 5 (check 3) |
| `badd36k` (bias add alone) | 0.476 | 0.467-0.510 | 0.423 | 1 |
| `lin5k_up` (`[5184,256] x [2048,256]`) | 1.425 | 1.418-1.465 | 1.363 | 4 (check 3) |
| `linb5k_up` (+ bias) | 1.749 | 1.728-1.823 | 1.669 | 5 (check 3) |

- **Q1 (permute): no kernel lever shown.**
  - The head swap costs the same as an identity copy through the same `ojas_permute_runs` kernel
    (0.462 vs 0.454 ms, 74.5 MB moved, ~160 GB/s).
  - The profile's 1.4-1.6 ms per call in place is about 3x what the kernel takes alone. The rest
    is the per-op sync and the state around the call in SpikeTrack, which a permute kernel cannot
    remove.
- **Q2 (linears): the bias pass is ~0.28 ms per 36k projection, ~0.32 ms at the FFN's up
  projection** (`linb − lin`).
  - A `linear_bias_forward` runs as five dispatches: three finite scans (x, w, y), the GEMM, and a
    generic broadcast add into a second 37 MB buffer.
  - That is the lever `aade8fa` takes (`../2026-11-03-linear-bias-inplace`).
- Every row above is one window at load 7-8; none is a speed claim.

## Part 3: test and lint (failed on this session's own code; rerun in the next job)

`attention_flash_bits` did not compile. Its golden loop zipped over the array by value
(`E0277`/`E0614`, `test-flash-bits.log`). The same error stopped clippy (`clippy.log`). The fix
is `c1676d6`. Neither result here says anything about the kernels.
