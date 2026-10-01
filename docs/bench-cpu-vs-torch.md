# CPU Reference versus PyTorch CPU Benchmarks

Wall time of one forward, backward, and AdamW step. The graph is the regression test in `ojas-cpu/tests/torch_ref.rs`: RMSNorm (eps `1e-6`), bias-free linear `y = x @ W.T`, half-split RoPE, causal SDPA with scale `1/sqrt(d)` and one head, mean cross-entropy, backward, then AdamW on `wq` (weight decay 0.1) and the norm weight (weight decay 0). The learning rate is `6e-4` times the cosine multiplier at step 8 (warmup 4, total 20). `wk`, `wv`, and `wo` receive weight gradients and are not written by AdamW. Moments start at zero.

```mermaid
flowchart TD
    subgraph BenchmarkGraph["Regression Benchmark Graph (torch_ref.rs)"]
        In["Input x [B, T, D]"] --> Norm["RMSNorm (eps = 1e-6)"]
        Norm --> Lin["Linear Projection (y = x @ W.T)"]
        Lin --> RoPE["Half-Split RoPE"]
        RoPE --> SDPA["Causal SDPA (scale = 1/sqrt(d))"]
        SDPA --> CE["Mean Cross-Entropy Loss"]
        CE --> Bwd["Backward Adjoint Pass"]
        Bwd --> Opt["AdamW Step (wq, norm_weight)"]
    end
```

Both timers use CPU. Torch uses `SDPBackend.MATH`. The tensors in the torch run were on `cpu`. This torch build also has MPS compiled in; the timed step did not move tensors there.

> [!IMPORTANT]
> Numbers below are **verified** from the commands in this file, on this machine, 2026-10-01. Two full runs were recorded. The within-run median is the median after dropping the warmup calls listed below. The second process did not repeat the first median closely on the tiny torch step, so both runs are listed and no single speedup is stated.

---

## Toolchain & System Profile

| Component | Verified Value |
| :--- | :--- |
| **rustc** | `rustc 1.98.0 (88d9e12ae 2026-08-18)` |
| **cargo** | `cargo 1.98.0 (797e8a9bc 2026-08-05)` |
| **Python** | 3.14.7, `/opt/homebrew/opt/python@3.14/bin/python3.14` |
| **torch** | 2.13.0, `/Users/bharath/Library/Python/3.14/lib/python/site-packages/torch/__init__.py` |
| **torch CPU threads** | `torch.get_num_threads() == 6`, interop threads 18 |
| **Rust threads** | Evaluated with 1 thread up to 6 threads via `CpuBackend::with_threads` |

Release profile from the workspace `Cargo.toml`: `panic = "unwind"`, `overflow-checks = true`.

---

## Commands

```bash
cd /Users/bharath/Code/research/ojas
cargo test -p ojas-cpu --release --test torch_ref -- --nocapture --test-threads=1
python3 /tmp/ojas_cpu_bench.py
```

The second invocation repeated `one_step_wall_time` only, then the same python script. `ojas-cpu` already built; criterion was not added.

The Rust clock wraps `tiny_step` / `regression_graph`, which builds every tensor from host slices, runs the step, and copies the loss and the updated `wq` and norm back to host vectors. `std::hint::black_box` is applied to that loss and those two vectors.

The torch clock wraps the forward, backward, and `AdamW.step`. Gradients are cleared inside that region. Weights are copied back and the AdamW state is cleared before the clock starts, so each call is one step from the initial parameters.

---

## Mathematical Correctness

The tiny Rust loss on the frozen `torch_ref` inputs stayed within `1e-4` of the frozen `LOSS`. The wall-time test printed `loss=3.4650407` and `loss_err=2.384e-7`. `one_step_matches_torch_2_13_float32` also passed, which checks that same loss and the other frozen tensors at `1e-4`.

The torch tiny loss, seed 0, matched the frozen bits before timing: `0x405dc33b`, absolute error `0`.

The larger shape has no frozen tensor file. The Rust run checked that its loss was finite (`4.852497` with the test's own generator). That value is not a torch match.

---

## Progression of Latency Optimizations

```mermaid
flowchart LR
    subgraph StepEvolution["Larger Step Latency Evolution (B=2, T=32, d=64)"]
        V0["Initial: 8.67 ms"] --> V1["Linear Kernel: 2.93 ms"]
        V1 --> V2["Attention Bwd: 1.70 ms"]
        V2 --> V3["Blocked Linear: 0.95 ms"]
        V3 --> V4["Attention Fwd: 0.58 ms"]
        V4 --> V5["RMSNorm/RoPE Tiling: 0.455 ms"]
    end
```

### Initial Baseline Medians

* Tiny: `B=1`, `T=4`, `d=16`, 1 head, `vocab=32`. 200 calls, drop the first 20, median of 180.
* Larger: `B=2`, `T=32`, `d=64`, 1 head, `vocab=128`. 20 calls, drop the first 4, median of 16.

| Run | Shape | Rust median | Torch CPU median |
| :--- | :--- | ---: | ---: |
| 1 | tiny | `4.572900000e-5` s | `1.755520498e-3` s |
| 1 | larger | `7.833208000e-3` s | `1.923291500e-3` s |
| 2 | tiny | `5.968750000e-5` s | `9.818130056e-4` s |
| 2 | larger | `8.672417000e-3` s | `1.707812495e-3` s |

*Readable scale:* Rust tiny ~46–60 µs vs Torch tiny ~0.98–1.76 ms. Rust larger ~7.83–8.67 ms vs Torch larger ~1.71–1.92 ms.

---

### Final Optimized Step Latency (Linear Tiles, RMSNorm, RoPE, View Reshape)

After applying contiguous register tiling, slice-based RMSNorm/RoPE, and zero-copy `Tensor::view` reshapes:

| Run | Shape | Rust median | Torch CPU median |
| :--- | :--- | ---: | ---: |
| 1 | tiny | `2.245800000e-5` s (~22.5 µs) | `3.540420003e-4` s (~354 µs) |
| 1 | larger | `4.620420000e-4` s (~462 µs) | `4.559369918e-4` s (~456 µs) |
| 2 | tiny | `2.245900000e-5` s (~22.5 µs) | `4.868955002e-4` s (~487 µs) |
| 2 | larger | `4.546040000e-4` s (~455 µs) | `4.629169998e-4` s (~463 µs) |

```mermaid
flowchart TD
    subgraph Breakdown["Larger Step Latency Breakdown (Final Optimizations)"]
        LinBwd["Linear Backward: 129.2 µs"]
        LinFwd["Linear Forward: 79.1 µs"]
        Rest["Rest (RMSNorm 15.3µs, RoPE 28.6µs, View 0.8µs): 55.6 µs"]
        CE["Cross-Entropy: 48.8 µs"]
        AttnBwd["Causal Attention Backward: 47.6 µs"]
        AdamW["AdamW Optimizer: 45.4 µs"]
        AttnFwd["Causal Attention Forward: 27.6 µs"]
    end
```

---

## Single Linear Projection Benchmark

Evaluating single `linear` operations at training dimensions. Both timers measure forward (`y = x @ W.T`) or backward (`grad_x` and `grad_w`). Rust uses `CpuBackend::with_threads(..., 6)`. Torch uses `torch.nn.functional.linear` with 6 threads.

### Earlier medians (Exact only, before the 2026-10-01 overhead work)

| Shape | Rust fwd | Torch fwd | Rust bwd | Torch bwd |
| :--- | ---: | ---: | ---: | ---: |
| **64×64×128** | 35.2 µs | 5.2 µs | 51.4 µs | 16.2 µs |
| **256×256×256** | 0.407 ms | 0.047 ms | 0.545 ms | 0.103 ms |
| **512×768×768** | 4.35 ms | 1.23 ms | 5.98 ms | 2.45 ms |

### 2026-10-01: both tiers, symmetric min-of-N

Apple M5 Pro (6 P-cores, 12 E-cores), load average 21–33 from other applications, so read the columns against each other, not as absolute speeds. Every cell is the minimum over 4 interleaved process pairs (baseline, current, baseline, ...), each itself a min of 30 forward and 15 backward calls. Torch is the same statistic (min of 30), run in the same window. "Before" is a frozen copy of the tree taken before this work; it is not a commit, so it cannot be rebuilt from git.

| 6 threads | Exact before | Exact now | Fast before | Fast now | Torch |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 64×64×128 fwd | 27 µs | 28 µs | 25 µs | 19 µs | 5 µs |
| 64×64×128 bwd | 41 µs | 33 µs | 32 µs | 23 µs | 18 µs |
| 256³ fwd | 0.245 ms | 0.190 ms | 0.141 ms | 0.092 ms | 0.062–0.072 ms |
| 256³ bwd | 0.454 ms | 0.368 ms | 0.244 ms | 0.171 ms | 0.144 ms |
| 512×768×768 fwd | 3.07 ms | 2.50 ms | 1.52 ms | 0.77 ms ¹ | 1.17–1.20 ms ¹ |
| 512×768×768 bwd | 5.82 ms | 5.07 ms | 2.47 ms | 2.55 ms | 2.25–2.29 ms |

¹ Three independent runs, each 4 interleaved pairs with torch run in the same window. Fast forward at 512×768×768: before 1.52 / 1.23 / 1.14 ms, now 0.77 / 0.62 / 0.66 ms, torch 1.17–1.20 / 1.26–1.31 / 0.81–0.87 ms. Fast forward was faster than torch in all three runs. Fast backward was 2.55 / 2.47 / 1.32 ms against torch 2.25–2.29 / 2.50–2.76 / 1.72–2.29 ms: parity within the noise, not a win. An earlier A/B by the CPU lane read "no change" at this shape (1.386 → 1.390 ms). That compared the minima of the two sides, and the baseline's minimum was one outlier pair. In 4 of its 5 six-thread pairs the live tree was faster (1.39–1.50 ms against 1.74–1.92 ms), and its harness was built at 16:32, before the lane's last allocation changes landed (`target-lane-cpu/prof-out/live1.txt`). Exact gains were 1.13–1.29× in every run.

At 18 threads, Exact 512×768×768 went from 1.85 to 1.44 ms forward and from 3.58 to 2.94 ms backward.

Committed reproduction of the "now" and "Torch" columns (medians rather than minimums):

```bash
OJAS_BENCH_THREADS=6 cargo test -p ojas-cpu --release --test bench_cpu -- --ignored --nocapture --test-threads=1 linear_wall_time_matrix
```

```bash
python3 ojas-cpu/benches/torch_linear.py --threads 6
```

What changed: `Tensor::to_f32_vec` and `to_u32_vec` decode with a vector copy, about 8× faster (`ojas-core/tests/bench_decode.rs`). Finite scans got cheaper. Transposed operands pack 4×4 at a time. The first reduction block starts from zero instead of loading C. `plan` makes up to 6 tiles per thread instead of 2. Linear outputs are charged to the budget for as long as they are held.

**Why Exact cannot reach torch here.** Torch's CPU `linear` on this build calls Accelerate (`BLAS_INFO=accelerate`), which runs on the matrix unit. Accelerate alone does 512×768×768 in 0.41 ms (`cargo run --release -p ojas-simd --example bench --features accelerate`, about 1.47 TFLOP/s). The Exact contract (separate multiply and add, ascending `k`) takes two vector instructions per multiply-add. Its micro-kernel measured 33.2 GMAC/s per P-core on L1-resident panels, against 63.5 for the FMA kernel; that is the FP-pipe bound for each. Six P-cores at 33 GMAC/s put a floor of about 1.5 ms on 512×768×768. Exact is the reproducible reference tier; Fast is the tier to compare with torch. Since 2026-10-01 Fast is the `CpuBackend` default and Exact is opt-in (`.with_numerics(Numerics::Exact)`).

**Where Fast still loses.** At 512×768×768, Fast backward is about 80% inside Accelerate (`sample` profile). The rest is the input finite scans, the output copy into a `Tensor`, and the input decode. All three come from the byte-backed `Tensor` storage, which needs a decoded copy per operand. Below 2²¹ multiply-adds, Fast deliberately stays on the packed FMA kernel, so its bits are the same on every thread count. Accelerate alone does 64×64×128 in about 2 µs, so lowering that cutoff would close the small-shape gap at the cost of that bit guarantee.

---

## Fast Numerics with Accelerate & GPU Backends

> [!WARNING]
> GPU benchmarks below were recorded under high system load average (load 30–48) on macOS 27. Read these timings as preliminary reference figures.

### 1. CPU with Apple Accelerate (`Numerics::Fast`)
On macOS, products $\ge 2^{21}$ multiply-adds route to Accelerate `cblas_sgemm` (18 threads):

| Shape | ojas Fast fwd | ojas Fast bwd | Accelerate alone | Torch CPU fwd | Torch CPU bwd |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **512×768×768** | ~0.89 ms | ~1.5 ms | 0.31 ms | 0.41 ms | 0.84 ms |
| **2048³** | ~15 ms | ~30 ms | 9.7 ms | 11.5 ms | 24.7 ms |

### 2. Metal (`MetalBackend`), Under Load
Compared against PyTorch MPS:

| Measurement | ojas Metal | Torch MPS |
| :--- | ---: | ---: |
| **Full step, d=768, B=4, T=128, V=50304, resident** | ~141 ms | 146 ms |
| **Linear forward, 2048³** | 3.86 ms | 2.81 ms |
| **Attention backward, T=2048** | 306 ms (old scalar kernel, superseded) | 60 ms |

The attention backward row is from the old scalar kernel. The tiled backward that replaced it is measured in [`bench-gpu-vs-torch.md`](bench-gpu-vs-torch.md).

### 3. wgpu (`WgpuBackend`)
* **GEMM 2048³ (Resident):** Operates at ~2.0 TFLOP/s (~7.5× faster than 6-thread CPU on the same shape).
* **Attention (T=2048):** Runs at approximately the same latency as CPU.
* **Tolerance:** `Fast` numerics match CPU within $10^{-4}$ relative tolerance.
