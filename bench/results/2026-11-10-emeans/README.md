# 2026-11-10 — `ojas-cluster` `e-means` vs 25-Iteration Lloyd on Apple M5 Pro

Interleaved A/B benchmark comparing 25-iteration full-pass Lloyd's $k$-means ([`ojas-cluster/src/lloyd.rs`](../../../ojas-cluster/src/lloyd.rs)) against scaling-law `e-means` (`canonical_v1`, [arXiv:2611.09501](https://arxiv.org/abs/2611.09501); [`ojas-cluster`](../../../ojas-cluster/README.md)) at `nepochs = 2.0` and `nepochs = 2.6` across multi-core CPU (`CpuMath::Fast`) and Apple Silicon Metal 4 (`F32`, `Bf16AssignF32Stats`, `Bf16` via [tessl](https://github.com/bharathvbcr/tessl)).

Raw machine-readable and human-readable log: [`emeans_m5pro.txt`](emeans_m5pro.txt).

## Reproduction Command

```sh
cargo run --release -p ojas-cluster --example emeans_bench | tee bench/results/2026-11-10-emeans/emeans_m5pro.txt
```

Benchmark driver: [`ojas-cluster/examples/emeans_bench.rs`](../../../ojas-cluster/examples/emeans_bench.rs).

## Machine, Toolchain & Load Conditions

| Field | Value |
| :--- | :--- |
| CPU / GPU | Apple M5 Pro ( Metal 4 `TensorOps` enabled ) |
| OS | macOS 27.2 (`aarch64`) |
| Rustc | `rustc 1.99.1 (58cf8f9c3 2026-10-13)` |
| Sibling `tessl` | `17471e7` (`fix(emeans): eliminate intra-threadgroup EMA device race for k=1024 bit-identity`) |
| CPU Pool | `pool = 18`, `host_available_parallelism = 18`, `simd_backend = neon` (`accelerate` enabled on macOS) |
| Memory Budget | `Budget::new(4 GiB)` (`peak_budget_bytes = 651,601,920`, `live_budget_bytes_before_drop = 158,319,360`, `live_budget_bytes_after_drop = 0` verified fail-closed after engine drop) |
| Host Load Average | `15.63 23.03 31.70` at start, `22.60 23.97 31.65` after Shape A, `30.47 25.67 31.83` after Shape B |

> [!NOTE]
> Because concurrent background processes held 1-minute machine load average at `15.6–30.5` during capture, absolute wall-clock latencies have wider `max_ms - min_ms` spread than an idle machine. The benchmark runs **cyclic-interleaved paired rounds** (`round = 1..=5`, cycling through all 9 contenders inside each round after 1 untimed warmup per contender) so every contender experiences the same contemporaneous thermal and scheduling conditions, and reports both global ratios (`speedup_min`, `speedup_med`) and within-round paired speedups (`paired_med [min-max]`).

## Methodology & Workloads

1. **Clustered Zipf-Weighted Gaussian Mixtures:** Each shape draws $k = 1024$ true component centers in $\mathbb{R}^d$ with $\sigma_{\text{center}} = 1 / \sqrt{d}$ and power-law component weights $w_j \propto (j + 16)^{-0.75}$, then samples $n$ points with isotropic perturbations $\sigma_{\text{noise}} = 0.25 / \sqrt{d}$. Uniform Forgy initialization (`init_centroids`, sampled once per shape without replacement) seeds multiple centroids inside dense head modes while leaving tail modes empty, exercising both MSE convergence and `e-means` Gumbel dead-cluster split revival (`Lloyd` leaves `42` empty clusters on Shape A and `71` on Shape B, whereas `e-means` revives `527–809` clusters across steps to populate `1022–1024 / 1024` clusters).
2. **Shared Initialization & Persistent Engines:** All 9 contenders on each shape start from the exact same `init_centroids` tensor. Single long-lived `EmeansEngine` instances (`Cpu`, `Metal::F32`, `Metal::Bf16AssignF32Stats`, `Metal::Bf16`) are reused across warmup and all 5 timed rounds so thread-pool and Metal slab-pool reuse match steady-state library usage. `lloyd_fit` runs unconditionally for all `25` passes (`tol = 0.0`, `CpuMath::Fast`, `threads = None`).
3. **Apples-to-Apples Objective:** Every contender's returned `f32` centroids are evaluated outside the timed window by the same cancellation-free `f64` oracle:

$$\text{MSE}_{f64}(X, C) = \frac{1}{n} \sum_{i=0}^{n-1} \min_{0 \le j < k} \sum_{m=0}^{d-1} \left(\text{f64}(X_{i,m}) - \text{f64}(C_{j,m})\right)^2$$

---

## Shape A Results (`n = 100_000, d = 128, k = 1024`)

- **Seed:** `0x2026111001281024`, Forgy initial `MSE (f64) = 0.530833`
- **Resolved `T = 2.0` recipe ([`ojas-cluster/src/recipe.rs`](../../../ojas-cluster/src/recipe.rs)):** `BatchLaw::Throughput`, `batch_size = 4096`, `alpha0 = 0.314537`, `epoch_rows = [100000, 100000]`, `total_steps = 50`
- **Resolved `T = 2.6` recipe:** `BatchLaw::Throughput`, `batch_size = 4096`, `alpha0 = 0.305934`, `epoch_rows = [100000, 60000, 100000]`, `total_steps = 65`

| Contender | `min_ms` | `median_ms` | `max_ms` | `spread_ms (%)` | `MSE (f64)` | `MSE vs Lloyd` | `speedup_min` | `speedup_med` | `paired_med [min-max]` | `non_empty` | `revived` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| `Lloyd 25-iter (CpuMath::Fast)` | 4005.60 | 4327.02 | 4976.89 | 971.29 (24.2%) | 0.228053 | +0.00% (1.0000x) | 1.00x | 1.00x | 1.00x [1.00-1.00x] | 982/1024 | 0 |
| `e-means T=2.0 (CpuMath::Fast)` | 376.94 | 387.35 | 488.56 | 111.62 (29.6%) | 0.170040 | -25.44% (0.7456x) | 10.63x | 11.17x | 11.06x [9.48-12.98x] | 1024/1024 | 532 |
| `e-means T=2.6 (CpuMath::Fast)` | 500.20 | 520.05 | 879.93 | 379.73 (75.9%) | 0.160376 | -29.68% (0.7032x) | 8.01x | 8.32x | 7.75x [5.66-8.33x] | 1024/1024 | 588 |
| `e-means T=2.0 (Metal::F32)` | 48.62 | 63.99 | 123.64 | 75.02 (154.3%) | 0.169943 | -25.48% (0.7452x) | 82.39x | 67.62x | 75.01x [37.46-82.39x] | 1024/1024 | 527 |
| `e-means T=2.6 (Metal::F32)` | 47.98 | 65.84 | 71.91 | 23.93 (49.9%) | 0.162137 | -28.90% (0.7110x) | 83.49x | 65.72x | 64.41x [63.30-83.90x] | 1024/1024 | 581 |
| `e-means T=2.0 (Metal::Bf16AssignF32Stats)` | 49.78 | 65.71 | 74.30 | 24.52 (49.3%) | 0.165726 | -27.33% (0.7267x) | 80.46x | 65.85x | 65.85x [62.33-80.46x] | 1024/1024 | 543 |
| `e-means T=2.6 (Metal::Bf16AssignF32Stats)` | 47.30 | 58.95 | 90.98 | 43.68 (92.4%) | 0.160264 | -29.73% (0.7027x) | 84.69x | 73.40x | 71.85x [54.70-84.69x] | 1024/1024 | 582 |
| `e-means T=2.0 (Metal::Bf16)` | 44.16 | 59.26 | 66.63 | 22.47 (50.9%) | 0.172521 | -24.35% (0.7565x) | 90.72x | 73.01x | 77.00x [73.01-90.72x] | 1024/1024 | 529 |
| `e-means T=2.6 (Metal::Bf16)` | 46.30 | 57.74 | 84.68 | 38.38 (82.9%) | 0.163665 | -28.23% (0.7177x) | 86.52x | 74.94x | 77.57x [58.77-86.52x] | 1024/1024 | 591 |

---

## Shape B Results (`n = 65_536, d = 512, k = 1024`)

- **Seed:** `0x2026111005121024`, Forgy initial `MSE (f64) = 0.610870`
- **Resolved `T = 2.0` recipe:** `BatchLaw::Throughput`, `batch_size = 2048`, `alpha0 = 0.203799`, `epoch_rows = [65536, 65536]`, `total_steps = 64`
- **Resolved `T = 2.6` recipe:** `BatchLaw::Throughput`, `batch_size = 4096`, `alpha0 = 0.369977`, `epoch_rows = [65536, 39322, 65536]`, `total_steps = 42`

| Contender | `min_ms` | `median_ms` | `max_ms` | `spread_ms (%)` | `MSE (f64)` | `MSE vs Lloyd` | `speedup_min` | `speedup_med` | `paired_med [min-max]` | `non_empty` | `revived` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| `Lloyd 25-iter (CpuMath::Fast)` | 4052.29 | 4766.71 | 5195.83 | 1143.53 (28.2%) | 0.248275 | +0.00% (1.0000x) | 1.00x | 1.00x | 1.00x [1.00-1.00x] | 953/1024 | 0 |
| `e-means T=2.0 (CpuMath::Fast)` | 428.62 | 572.35 | 750.19 | 321.57 (75.0%) | 0.197876 | -20.30% (0.7970x) | 9.45x | 8.33x | 8.33x [6.56-9.45x] | 1023/1024 | 801 |
| `e-means T=2.6 (CpuMath::Fast)` | 501.02 | 545.76 | 759.32 | 258.30 (51.6%) | 0.179865 | -27.55% (0.7245x) | 8.09x | 8.73x | 8.09x [6.48-8.73x] | 1023/1024 | 654 |
| `e-means T=2.0 (Metal::F32)` | 87.66 | 108.06 | 174.72 | 87.06 (99.3%) | 0.192224 | -22.58% (0.7742x) | 46.23x | 44.11x | 46.23x [28.18-47.27x] | 1022/1024 | 809 |
| `e-means T=2.6 (Metal::F32)` | 92.23 | 93.54 | 170.36 | 78.14 (84.7%) | 0.178966 | -27.92% (0.7208x) | 43.94x | 50.96x | 46.10x [28.90-50.96x] | 1023/1024 | 663 |
| `e-means T=2.0 (Metal::Bf16AssignF32Stats)` | 101.69 | 118.72 | 122.75 | 21.06 (20.7%) | 0.199768 | -19.54% (0.8046x) | 39.85x | 40.15x | 40.18x [33.96-46.88x] | 1023/1024 | 802 |
| `e-means T=2.6 (Metal::Bf16AssignF32Stats)` | 95.90 | 116.23 | 123.62 | 27.73 (28.9%) | 0.181201 | -27.02% (0.7298x) | 42.26x | 41.01x | 42.03x [36.51-49.71x] | 1022/1024 | 680 |
| `e-means T=2.0 (Metal::Bf16)` | 88.49 | 95.96 | 152.52 | 64.02 (72.3%) | 0.193460 | -22.08% (0.7792x) | 45.79x | 49.67x | 45.79x [34.07-52.87x] | 1024/1024 | 806 |
| `e-means T=2.6 (Metal::Bf16)` | 86.84 | 103.19 | 154.53 | 67.69 (78.0%) | 0.183158 | -26.23% (0.7377x) | 46.67x | 46.19x | 44.00x [33.62-56.69x] | 1024/1024 | 688 |

---

## Verified Observations

1. **Wall-Clock Speedup & Lower Distortion Simultaneously:** Every `e-means` contender (`T = 2.0` and `T = 2.6` across CPU and all three Metal 4 precision lanes) achieves **19.5%–29.7% lower exact `f64` MSE** than 25-iteration Lloyd (`0.1603–0.1725` vs `0.2281` on Shape A; `0.1790–0.1998` vs `0.2483` on Shape B) because Gumbel dead-cluster split revival reallocates centroids trapped in empty tail modes while mini-batch EMA + cosine decay escapes bad Forgy local minima in `2.0–2.6` passes instead of `25` full passes.
2. **CPU Speedup (`CpuMath::Fast`):** On the same 18-thread Apple M5 Pro CPU substrate, `e-means` runs **8.0x–10.6x faster** (`min_ms`) and **8.3x–11.2x faster** (`median_ms`) than 25-iteration `lloyd_fit`. Note on Shape B (`n = 65_536, d = 512`): `T = 2.6` resolves `batch_size = 4096` (`42` batch updates across `170,394` row visits) whereas `T = 2.0` resolves `batch_size = 2048` (`64` batch updates across `131,072` row visits), which explains why several contenders (`CpuMath::Fast` `median_ms` `545.76 ms` vs `572.35 ms`, `Metal::Bf16` `min_ms` `86.84 ms` vs `88.49 ms`, `Metal::Bf16AssignF32Stats` `min_ms` `95.90 ms` vs `101.69 ms`, and `Metal::F32` `median_ms` `93.54 ms` vs `108.06 ms`) are faster at `T = 2.6` due to fewer per-batch EMA/revival/centroid-prep barrier rounds.
3. **Metal 4 Speedup (`TensorOps`):** Against 25-iteration CPU Lloyd, `ClusterDevice::Metal` achieves **80.5x–90.7x `min_ms` speedup** (`65.7x–74.9x` `median_ms`) on Shape A (`d = 128`, `44.2–49.8 ms` `min_ms`) and **39.9x–46.7x `min_ms` speedup** (`40.2x–51.0x` `median_ms`) on compute-heavier Shape B (`d = 512`, `86.8–101.7 ms` `min_ms`, with `Metal::Bf16` `T = 2.6` fastest `min_ms` at `86.84 ms` and `Metal::F32` `T = 2.6` fastest `median_ms` at `93.54 ms`). Against same-epoch CPU `e-means`, Metal 4 is **7.6x–10.8x faster** on Shape A (`min_ms`) and **4.2x–5.8x faster** on Shape B (`min_ms`).
4. **0-ULP Repeat Bit-Identity Across CPU and Metal 4 (`unique_centroid_hashes = 1` in [`emeans_m5pro.txt`](emeans_m5pro.txt)):** Across 1 untimed warmup + 5 cyclic-interleaved rounds (`6` runs per contender, `108` runs total), **all 9 contenders**—`Lloyd 25-iter (CpuMath::Fast)`, `e-means` on `CpuMath::Fast`, and all three Metal 4 `TensorOps` precision lanes (`Metal::F32`, `Metal::Bf16AssignF32Stats`, `Metal::Bf16`)—recorded `unique_centroid_hashes = 1` with bit-for-bit identical `[k, d]` `f32` centroids, closing labels, cluster counts, and revived-cluster counts on both Shape A and Shape B (`k = 1024`). Repeat-run bit-identity is enforced fail-closed in [`ojas-cluster/examples/emeans_bench.rs`](../../../ojas-cluster/examples/emeans_bench.rs) and [`ojas-cluster/tests/metal_and_session.rs`](../../../ojas-cluster/tests/metal_and_session.rs).
