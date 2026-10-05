# Split cache walk for small decode calls, 2026-10-04

**Change.** `ojas_cached_attn` can walk the KV cache in `splits` parts, one
threadgroup each, and `ojas_cached_attn_merge` combines them in index order.
- `cached_attn_splits` (`ojas-metal/src/device.rs`) aims for 96 threadgroups,
  with each split at least 64 keys.
- One decode request (12 heads, 1024 keys) gets 8 splits. 2 requests get 4,
  4 requests get 2, and 8 or more requests get 1, which is the old single pass,
  bit for bit.

**Run.**
- `target-baseline/split-run.sh` under the Lappi lead's lock,
  16:14:47–16:51:38Z.
- Apple M5 Pro. Other work kept the GPU 69–94% busy.
- Raw output: `percall.md` and `ab/`.

## Correctness (verified)

- ojas-metal: all suites passed.
  - `kv_cache.rs` 10/10, including two new tests. One compares the split
    path with an f64 reference at uneven splits, empty splits, grouped-query
    heads, Tq = 70 and a batch. The other checks that NaN and infinity are
    reported in every split, at split boundaries, and for overflowing scores,
    and that positions past `kv_len` stay unread.
  - The device unit test `cached_attn_splits_only_where_the_gpu_would_sit_idle`
    passed.
  - `cached_attention_is_deterministic` passed. It runs one request at 1024
    keys, so it now takes the split path, and the results repeat bit for bit.
- Dependents `ojas-infer` and `ojas-model`: all passed.
- clippy on ojas-metal, all targets: no warnings in ojas-metal. The 7
  warnings it printed are in peer lanes' uncommitted ojas-simd and ojas-cpu
  code.
- Parity against torch in the A/B is unchanged: max |diff| 1.5e-8 to 2.6e-8
  on both sides.
- **Not done:** a mutation check that the new tests catch a broken merge.
  The new correctness tests would also pass on the old code, because the
  behaviour is meant to be the same. Only the policy unit test is specific to
  this change.

## GPU time (`percall.md`, tessl directly, 500 buffers)

**One request (12 heads, 1024 keys), median µs by split count:**

| splits | 1 | 2 | 4 | 8 | 16 | 32 |
| :-- | --: | --: | --: | --: | --: | --: |
| median µs | 48.2 | 39.2 | 38.0 | 37.7 | 44.8 | 59.4 |

- At 8 splits a single request's GPU time drops about 22%. Past 16 splits,
  the merge and the extra threadgroups cost more than they save.
- Ignore some minimums: two exceed any possible bandwidth (16 requests,
  1 split: 21.9 µs, "4594 GB/s"). On a GPU this busy, timestamps are
  unreliable at the low end. Medians are quoted.
- At 2, 4 and 16 requests, splitting changes the median by 0–10%. That is
  why the policy stops splitting once a call has about 96 threadgroups.

## Op time, interleaved A/B against the pre-change binary (`ab/summary.txt`)

8 rounds, alternating order, median of round minimums, ms:

| row | before | after | change |
| :-- | --: | --: | --: |
| 1 request, 1 call (`sweep_decode_b1`) | 0.150 | 0.151 | none |
| 2 requests, 1 call | 0.170 | 0.165 | −3% |
| 4 requests, 1 call | 0.236 | 0.226 | −4% |
| 16 requests, 1 call (no split; control) | 0.547 | 0.537 | −2% |
| 4 requests, 4 calls (`x4`) | 0.322 | 0.289 | −10% |
| 8 requests, 8 calls (`x8`) | 0.554 | 0.489 | −12% |
| 16 requests, 16 calls (`x16`) | 0.950 | 0.842 | −11% |

**Reading.**
- A lone request does not get faster end to end. Its ~10 µs GPU saving is
  lost in the ~0.13 ms fixed cost of sending work and waiting.
- Back-to-back separate calls get 10–12% faster. Each call saves its GPU
  time, and nothing hides the saving.
- Separate calls are still far slower than one batched call: 16 calls take
  0.842 ms against 0.537 ms. Batching at the caller remains the main fix.

## Side finding

- `ojas-metal/tests/linear_ce.rs` `matches_the_cpu_reference_at_the_bench_vocabulary`
  took 30 minutes in a debug build (1817 s for the binary). It is
  unrelated to this change, and it makes a full `cargo test -p ojas-metal`
  hard to run inside a short lock window.
