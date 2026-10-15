# CPU GDN follow-ups, 2026-10-14

Three rulings, from `baseline.txt` before the edit and from three process-alternated rounds after it (`interleave/`, `after.txt`). The alternation is two binaries, not one process. Load on the after rounds was about 22–34. No scorecard line is closed, and the mul-versus-torch gap is not closed.

## GDN forward

The forward still normalizes each row with the same ascending f64 sum of squares, the same epsilon, and the same cast to f32. It does not allocate the scale vector. When H is 1 the head's tokens are the token-major rows, so the kernel writes `out` directly and does not allocate the head-major buffer. When H is greater than 1 the head-major scratch and the copy stay: safe Rust cannot hand parallel tasks strided `&mut [f32]`, and the crate forbids `unsafe`. Exact-tier folds match the unmodified kernel.

Best-minimum deltas versus the unmodified binary re-run in the same alternation: H=1, 1 thread −2.7%; H=16, 6 threads +2.4%; B=4, H=1, 6 threads +3.1%. The H=1, 6-thread best minimum is +12.4% only because one unmodified sample hit 1850.0 µs against a 2480.0 µs median; the paired ratios are 0.90, 1.24 and 1.00. Backward code is unchanged (−9.4% on the best minimum, noise).

The budget test's cap is the new peak (H=1: 432 bytes, H=2: 832 bytes on the shapes in the test). On the unmodified kernel that H=1 cap failed with `CapacityExceeded { requested: 208, cap: 432, live: 272 }`.

## Pointwise

Adopted: 4 pieces when `threads >= 6` and n is in `[1024*1024, 1024*2048]`, otherwise the live cut. Four threads lose to the live cut inside the band (1.30× and 1.14×). Six and eight threads win inside it and lose at `1024*4096`. Threads 5 and 7 were not measured. Public mul at 8 threads: n=1M −41%, n=2M −21%. The 6-thread piece count did not change.

## Spawn

Mitigation A, adopted in `chunks_into_n`: same piece ranges, calling thread only, when output bytes are at most 2 MiB. The pre-change triad had 1 thread ahead through 1.5 MiB (30.8 µs vs 91.6 µs at 6 threads) and 2 threads ahead at 3 MiB, while 6 MiB serial lost to 6 threads (152.7 vs 133.2). Small RMSNorm at 6 threads −39% (89.4 → 54.8 µs). Large RMSNorm at 6 threads did not regress (paired ratios 0.89, 1.05, 0.75). Two threads on the small RMSNorm is +10.4% on the best minimum (paired 1.21, 1.07, 0.94). That is the serial cut replacing a spawn that sometimes tied; the triad at the same byte count had 1 thread ahead of 2 (17.0 vs 21.3 µs), so the cut stays serial.

Mitigation B, rejected. One extra thread versus five, trivial scope, 8 MiB stacks: 28.7 vs 55.5 µs minimum on the unmodified run (medians 39.2 and 75.9). A whole optimizer scope would edit `trainer.rs` / `optim.rs`.
