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
| Memory Budget | `Budget::new(4 GiB)` (`peak_budget_bytes = 651,601,920`, `live_bytes == 0` verified after engine drop) |
| Host Load Average | `77.54 57.68 45.36` at start, `75.58 60.26 47.02` after Shape A, `109.19 78.73 55.80` after Shape B |

> [!NOTE]
> Because concurrent background processes held machine load average at `58–109` during capture, absolute wall-clock latencies have wider `max_ms - min_ms` spread than an idle machine. The benchmark runs **cyclic-interleaved paired rounds** (`round = 1..=5`, cycling through all 9 contenders inside each round after 1 untimed warmup per contender) so every contender experiences the same contemporaneous thermal and scheduling conditions, and reports both global ratios (`speedup_min`, `speedup_med`) and within-round paired speedups (`paired_med [min-max]`).

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
| `Lloyd 25-iter (CpuMath::Fast)` | 5047.01 | 5285.93 | 6304.91 | 1257.90 (24.9%) | 0.228053 | +0.00% (1.0000x) | 1.00x | 1.00x | 1.00x [1.00-1.00x] | 982/1024 | 0 |
| `e-means T=2.0 (CpuMath::Fast)` | 494.40 | 591.82 | 656.89 | 162.49 (32.9%) | 0.170040 | -25.44% (0.7456x) | 10.21x | 8.93x | 10.21x [8.92-10.68x] | 1024/1024 | 532 |
| `e-means T=2.6 (CpuMath::Fast)` | 548.34 | 733.95 | 949.48 | 401.14 (73.2%) | 0.160376 | -29.68% (0.7032x) | 9.20x | 7.20x | 7.87x [6.41-9.64x] | 1024/1024 | 588 |
| `e-means T=2.0 (Metal::F32)` | 49.64 | 63.66 | 92.57 | 42.94 (86.5%) | 0.169943 | -25.48% (0.7452x) | 101.68x | 83.03x | 82.94x [65.76-101.68x] | 1024/1024 | 527 |
| `e-means T=2.6 (Metal::F32)` | 49.62 | 54.17 | 76.78 | 27.16 (54.7%) | 0.162137 | -28.90% (0.7110x) | 101.71x | 97.58x | 99.03x [79.29-114.80x] | 1024/1024 | 581 |
| `e-means T=2.0 (Metal::Bf16AssignF32Stats)` | 50.21 | 57.80 | 64.02 | 13.81 (27.5%) | 0.165726 | -27.33% (0.7267x) | 100.53x | 91.45x | 95.09x [82.46-109.08x] | 1024/1024 | 543 |
| `e-means T=2.6 (Metal::Bf16AssignF32Stats)` | 50.77 | 62.44 | 66.83 | 16.06 (31.6%) | 0.160264 | -29.73% (0.7027x) | 99.40x | 84.66x | 94.34x [80.67-104.11x] | 1024/1024 | 582 |
| `e-means T=2.0 (Metal::Bf16)` | 48.44 | 56.60 | 68.43 | 20.00 (41.3%) | 0.172521 | -24.35% (0.7565x) | 104.20x | 93.38x | 106.91x [88.95-109.01x] | 1024/1024 | 529 |
| `e-means T=2.6 (Metal::Bf16)` | 46.19 | 54.47 | 60.01 | 13.82 (29.9%) | 0.163665 | -28.23% (0.7177x) | 109.27x | 97.04x | 101.45x [90.61-115.74x] | 1024/1024 | 591 |

---

## Shape B Results (`n = 65_536, d = 512, k = 1024`)

- **Seed:** `0x2026111005121024`, Forgy initial `MSE (f64) = 0.610870`
- **Resolved `T = 2.0` recipe:** `BatchLaw::Throughput`, `batch_size = 2048`, `alpha0 = 0.203799`, `epoch_rows = [65536, 65536]`, `total_steps = 64`
- **Resolved `T = 2.6` recipe:** `BatchLaw::Throughput`, `batch_size = 4096`, `alpha0 = 0.369977`, `epoch_rows = [65536, 39322, 65536]`, `total_steps = 42`

| Contender | `min_ms` | `median_ms` | `max_ms` | `spread_ms (%)` | `MSE (f64)` | `MSE vs Lloyd` | `speedup_min` | `speedup_med` | `paired_med [min-max]` | `non_empty` | `revived` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| `Lloyd 25-iter (CpuMath::Fast)` | 5991.42 | 6935.69 | 8653.08 | 2661.65 (44.4%) | 0.248275 | +0.00% (1.0000x) | 1.00x | 1.00x | 1.00x [1.00-1.00x] | 953/1024 | 0 |
| `e-means T=2.0 (CpuMath::Fast)` | 654.95 | 866.32 | 1014.26 | 359.30 (54.9%) | 0.197876 | -20.30% (0.7970x) | 9.15x | 8.01x | 9.33x [6.84-10.28x] | 1023/1024 | 801 |
| `e-means T=2.6 (CpuMath::Fast)` | 725.66 | 1014.84 | 1566.73 | 841.06 (115.9%) | 0.179865 | -27.55% (0.7245x) | 8.26x | 6.83x | 7.12x [5.32-9.28x] | 1023/1024 | 654 |
| `e-means T=2.0 (Metal::F32)` | 148.07 | 211.90 | 286.01 | 137.95 (93.2%) | 0.192224 | -22.58% (0.7742x) | 40.46x | 32.73x | 31.77x [22.28-58.44x] | 1022/1024 | 809 |
| `e-means T=2.6 (Metal::F32)` | 126.32 | 131.04 | 132.69 | 6.36 (5.0%) | 0.178966 | -27.92% (0.7208x) | 47.43x | 52.93x | 53.66x [47.43-65.81x] | 1023/1024 | 663 |
| `e-means T=2.0 (Metal::Bf16AssignF32Stats)` | 108.61 | 138.45 | 146.31 | 37.70 (34.7%) | 0.199768 | -19.54% (0.8046x) | 55.17x | 50.10x | 55.17x [46.80-63.40x] | 1023/1024 | 802 |
| `e-means T=2.6 (Metal::Bf16AssignF32Stats)` | 112.95 | 144.79 | 180.84 | 67.88 (60.1%) | 0.181201 | -27.02% (0.7298x) | 53.04x | 47.90x | 47.90x [39.91-69.27x] | 1022/1024 | 680 |
| `e-means T=2.0 (Metal::Bf16)` | 108.06 | 136.56 | 224.17 | 116.10 (107.4%) | 0.193460 | -22.08% (0.7792x) | 55.44x | 50.79x | 50.79x [37.17-72.96x] | 1024/1024 | 806 |
| `e-means T=2.6 (Metal::Bf16)` | 91.72 | 128.13 | 210.75 | 119.03 (129.8%) | 0.183158 | -26.23% (0.7377x) | 65.33x | 54.13x | 52.94x [39.54-76.73x] | 1024/1024 | 688 |

---

## Verified Observations

1. **Wall-Clock Speedup & Lower Distortion Simultaneously:** Every `e-means` contender (`T = 2.0` and `T = 2.6` across CPU and all three Metal 4 precision lanes) achieves **19.5%–29.7% lower exact `f64` MSE** than 25-iteration Lloyd (`0.1603–0.1725` vs `0.2281` on Shape A; `0.1790–0.1998` vs `0.2483` on Shape B) because Gumbel dead-cluster split revival reallocates centroids trapped in empty tail modes while mini-batch EMA + cosine decay escapes bad Forgy local minima in `2.0–2.6` passes instead of `25` full passes.
2. **CPU Speedup (`CpuMath::Fast`):** On the same 18-thread Apple M5 Pro CPU substrate, `e-means` runs **8.3x–10.2x faster** (`min_ms`) and **6.8x–8.9x faster** (`median_ms`) than 25-iteration `lloyd_fit`. Note on Shape B (`n = 65_536, d = 512`): `T = 2.6` resolves `batch_size = 4096` (`42` batch updates across `170,394` row visits) whereas `T = 2.0` resolves `batch_size = 2048` (`64` batch updates across `131,072` row visits), which explains why `Metal::Bf16` and `Metal::F32` `min_ms` and `median_ms` are slightly faster at `T = 2.6` (`91.72 ms` vs `108.06 ms` `min_ms` and `128.13 ms` vs `136.56 ms` `median_ms` on `Bf16`) due to fewer per-batch EMA/revival/centroid-prep barrier rounds.
3. **Metal 4 Speedup (`TensorOps`):** Against 25-iteration CPU Lloyd, `ClusterDevice::Metal` achieves **99.4x–109.3x `min_ms` speedup** (`83.0x–97.6x` `median_ms`) on Shape A (`d = 128`, `46.2–50.8 ms` `min_ms`) and **40.5x–65.3x `min_ms` speedup** (`32.7x–54.1x` `median_ms`) on compute-heavier Shape B (`d = 512`, `91.7–148.1 ms` `min_ms`, with `Metal::Bf16` fastest at `91.72 ms` `min_ms` / `128.13 ms` `median_ms`). Against same-epoch CPU `e-means`, Metal 4 is **9.8x–11.9x faster** on Shape A (`min_ms`) and **4.4x–7.9x faster** on Shape B (`min_ms`).
4. **0-ULP Repeat Bit-Identity Across CPU and Metal 4 (`unique_centroid_hashes = 1` in [`emeans_m5pro.txt`](emeans_m5pro.txt)):** Across 1 untimed warmup + 5 cyclic-interleaved rounds (`6` runs per contender, `108` runs total), **all 9 contenders**—`Lloyd 25-iter (CpuMath::Fast)`, `e-means` on `CpuMath::Fast`, and all three Metal 4 `TensorOps` precision lanes (`Metal::F32`, `Metal::Bf16AssignF32Stats`, `Metal::Bf16`)—recorded `unique_centroid_hashes = 1` with bit-for-bit identical `[k, d]` `f32` centroids, closing labels, cluster counts, and revived-cluster counts on both Shape A and Shape B (`k = 1024`). Repeat-run bit-identity is enforced fail-closed in [`ojas-cluster/examples/emeans_bench.rs`](../../../ojas-cluster/examples/emeans_bench.rs) and [`ojas-cluster/tests/metal_and_session.rs`](../../../ojas-cluster/tests/metal_and_session.rs).
