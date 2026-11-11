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
| CPU Pool | `pool = 18`, `host_available_parallelism = 18`, `simd_backend = neon` (`accelerate` enabled on macOS) |
| Memory Budget | `Budget::new(4 GiB)` (`peak_budget_bytes = 651,601,920`, `live_bytes == 0` verified after engine drop) |
| Host Load Average | `67.87 52.73 57.02` at start, `55.99 52.15 56.54` after Shape A, `56.32 54.33 57.08` after Shape B |

> [!NOTE]
> Because concurrent background processes held machine load average at `52–68` during capture, absolute wall-clock latencies have wider `max_ms - min_ms` spread than an idle machine. The benchmark runs **cyclic-interleaved paired rounds** (`round = 1..=5`, cycling through all 9 contenders inside each round after 1 untimed warmup per contender) so every contender experiences the same contemporaneous thermal and scheduling conditions, and reports both global ratios (`speedup_min`, `speedup_med`) and within-round paired speedups (`paired_med [min-max]`).

## Methodology & Workloads

1. **Clustered Zipf-Weighted Gaussian Mixtures:** Each shape draws $k = 1024$ true component centers in $\mathbb{R}^d$ with $\sigma_{\text{center}} = 1 / \sqrt{d}$ and power-law component weights $w_j \propto (j + 16)^{-0.75}$, then samples $n$ points with isotropic perturbations $\sigma_{\text{noise}} = 0.25 / \sqrt{d}$. Uniform Forgy initialization (`init_centroids`, sampled once per shape without replacement) seeds multiple centroids inside dense head modes while leaving tail modes empty, exercising both MSE convergence and `e-means` Gumbel dead-cluster split revival (`Lloyd` leaves `42` empty clusters on Shape A and `71` on Shape B, whereas `e-means` revives `527–810` clusters across steps to populate `1022–1024 / 1024` clusters).
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
| `Lloyd 25-iter (CpuMath::Fast)` | 4793.41 | 4842.81 | 5160.61 | 367.20 (7.7%) | 0.228053 | +0.00% (1.0000x) | 1.00x | 1.00x | 1.00x [1.00-1.00x] | 982/1024 | 0 |
| `e-means T=2.0 (CpuMath::Fast)` | 471.12 | 493.62 | 557.21 | 86.08 (18.3%) | 0.170040 | -25.44% (0.7456x) | 10.17x | 9.81x | 9.93x [8.94-10.17x] | 1024/1024 | 532 |
| `e-means T=2.6 (CpuMath::Fast)` | 587.02 | 638.66 | 689.00 | 101.98 (17.4%) | 0.160376 | -29.68% (0.7032x) | 8.17x | 7.58x | 7.68x [7.49-8.17x] | 1024/1024 | 588 |
| `e-means T=2.0 (Metal::F32)` | 53.68 | 67.03 | 77.54 | 23.86 (44.4%) | 0.169836 | -25.53% (0.7447x) | 89.29x | 72.25x | 74.33x [66.56-89.29x] | 1024/1024 | 538 |
| `e-means T=2.6 (Metal::F32)` | 52.04 | 72.83 | 76.07 | 24.04 (46.2%) | 0.159899 | -29.89% (0.7011x) | 92.12x | 66.49x | 67.22x [63.66-95.75x] | 1024/1024 | 573 |
| `e-means T=2.0 (Metal::Bf16AssignF32Stats)` | 54.08 | 63.38 | 115.71 | 61.63 (114.0%) | 0.165726 | -27.33% (0.7267x) | 88.63x | 76.41x | 76.32x [44.60-88.63x] | 1024/1024 | 543 |
| `e-means T=2.6 (Metal::Bf16AssignF32Stats)` | 55.12 | 74.79 | 88.82 | 33.70 (61.1%) | 0.160264 | -29.73% (0.7027x) | 86.96x | 64.75x | 69.00x [53.96-87.85x] | 1024/1024 | 582 |
| `e-means T=2.0 (Metal::Bf16)` | 52.25 | 64.23 | 75.89 | 23.64 (45.2%) | 0.172511 | -24.35% (0.7565x) | 91.74x | 75.40x | 74.63x [67.04-92.58x] | 1024/1024 | 530 |
| `e-means T=2.6 (Metal::Bf16)` | 54.30 | 65.08 | 79.51 | 25.21 (46.4%) | 0.163411 | -28.34% (0.7166x) | 88.28x | 74.41x | 74.41x [64.90-89.09x] | 1024/1024 | 583 |

---

## Shape B Results (`n = 65_536, d = 512, k = 1024`)

- **Seed:** `0x2026111005121024`, Forgy initial `MSE (f64) = 0.610870`
- **Resolved `T = 2.0` recipe:** `BatchLaw::Throughput`, `batch_size = 2048`, `alpha0 = 0.203799`, `epoch_rows = [65536, 65536]`, `total_steps = 64`
- **Resolved `T = 2.6` recipe:** `BatchLaw::Throughput`, `batch_size = 4096`, `alpha0 = 0.369977`, `epoch_rows = [65536, 39322, 65536]`, `total_steps = 42`

| Contender | `min_ms` | `median_ms` | `max_ms` | `spread_ms (%)` | `MSE (f64)` | `MSE vs Lloyd` | `speedup_min` | `speedup_med` | `paired_med [min-max]` | `non_empty` | `revived` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| `Lloyd 25-iter (CpuMath::Fast)` | 5055.08 | 5480.39 | 5933.10 | 878.02 (17.4%) | 0.248275 | +0.00% (1.0000x) | 1.00x | 1.00x | 1.00x [1.00-1.00x] | 953/1024 | 0 |
| `e-means T=2.0 (CpuMath::Fast)` | 587.15 | 598.83 | 700.53 | 113.39 (19.3%) | 0.197876 | -20.30% (0.7970x) | 8.61x | 9.15x | 8.64x [8.47-9.63x] | 1023/1024 | 801 |
| `e-means T=2.6 (CpuMath::Fast)` | 656.68 | 707.02 | 791.79 | 135.10 (20.6%) | 0.179865 | -27.55% (0.7245x) | 7.70x | 7.75x | 7.53x [7.24-8.78x] | 1023/1024 | 654 |
| `e-means T=2.0 (Metal::F32)` | 104.98 | 145.71 | 176.62 | 71.64 (68.2%) | 0.192224 | -22.58% (0.7742x) | 48.15x | 37.61x | 35.12x [31.26-52.20x] | 1022/1024 | 809 |
| `e-means T=2.6 (Metal::F32)` | 114.78 | 116.15 | 194.92 | 80.15 (69.8%) | 0.178966 | -27.92% (0.7208x) | 44.04x | 47.18x | 44.58x [25.93-51.19x] | 1023/1024 | 663 |
| `e-means T=2.0 (Metal::Bf16AssignF32Stats)` | 134.29 | 136.24 | 240.96 | 106.67 (79.4%) | 0.199768 | -19.54% (0.8046x) | 37.64x | 40.23x | 37.10x [22.74-44.18x] | 1023/1024 | 802 |
| `e-means T=2.6 (Metal::Bf16AssignF32Stats)` | 123.09 | 139.86 | 197.91 | 74.82 (60.8%) | 0.181201 | -27.02% (0.7298x) | 41.07x | 39.18x | 36.58x [27.69-48.18x] | 1022/1024 | 680 |
| `e-means T=2.0 (Metal::Bf16)` | 98.28 | 109.55 | 157.13 | 58.85 (59.9%) | 0.193460 | -22.08% (0.7792x) | 51.44x | 50.03x | 46.14x [34.88-58.69x] | 1024/1024 | 806 |
| `e-means T=2.6 (Metal::Bf16)` | 92.99 | 96.95 | 116.64 | 23.65 (25.4%) | 0.183158 | -26.23% (0.7377x) | 54.36x | 56.53x | 56.53x [43.34-62.03x] | 1024/1024 | 688 |

---

## Verified Observations

1. **Wall-Clock Speedup & Lower Distortion Simultaneously:** Every `e-means` contender (`T = 2.0` and `T = 2.6` across CPU and all three Metal 4 precision lanes) achieves **19.5%–29.9% lower exact `f64` MSE** than 25-iteration Lloyd (`0.1599–0.1725` vs `0.2281` on Shape A; `0.1790–0.1998` vs `0.2483` on Shape B) because Gumbel dead-cluster split revival reallocates centroids trapped in empty tail modes while mini-batch EMA + cosine decay escapes bad Forgy local minima in `2.0–2.6` passes instead of `25` full passes.
2. **CPU Speedup (`CpuMath::Fast`):** On the same 18-thread Apple M5 Pro CPU substrate, `e-means` runs **7.7x–10.2x faster** (`min_ms`) and **7.6x–9.8x faster** (`median_ms`) than 25-iteration `lloyd_fit`. Note on Shape B (`n = 65_536, d = 512`): `T = 2.6` resolves `batch_size = 4096` (`42` batch updates across `170,394` row visits) whereas `T = 2.0` resolves `batch_size = 2048` (`64` batch updates across `131,072` row visits), which explains why `Metal::Bf16` and `Metal::F32` `median_ms` are slightly faster at `T = 2.6` (`96.95 ms` vs `109.55 ms` on `Bf16`) due to fewer per-batch EMA/revival/centroid-prep barrier rounds.
3. **Metal 4 Speedup (`TensorOps`):** Against 25-iteration CPU Lloyd, `ClusterDevice::Metal` achieves **87.0x–92.1x `min_ms` speedup** (`64.8x–76.4x` `median_ms`) on Shape A (`d = 128`, `52.0–55.1 ms` `min_ms`) and **37.6x–54.4x `min_ms` speedup** (`37.6x–56.5x` `median_ms`) on compute-heavier Shape B (`d = 512`, `93.0–134.3 ms` `min_ms`, with `Metal::Bf16` fastest at `92.99 ms` `min_ms` / `96.95 ms` `median_ms`). Against same-epoch CPU `e-means`, Metal 4 is **8.8x–11.3x faster** on Shape A (`min_ms`) and **4.4x–7.1x faster** on Shape B (`min_ms`).
4. **Repeat Bit-Identity (`unique_centroid_hashes` in [`emeans_m5pro.txt`](emeans_m5pro.txt)):** Across warmup + 5 rounds (`6` runs per contender), all CPU contenders (`Lloyd 25-iter`, `e-means T=2.0`, `e-means T=2.6`) recorded `unique_centroid_hashes = 1` (`0`-ULP identical centroid bits on every run). The three Metal 4 contenders recorded `unique_centroid_hashes = 3..6` across the 6 runs at `k = 1024` (for example, Shape B `Metal::Bf16AssignF32Stats` `T=2.0` produced bit-identical centroids on 5 of 6 runs with `MSE = 0.199768` and `0.197234` on round 3; Shape A `Metal::F32` `T=2.0` ranged across `0.165852..0.169943`), so median `MSE (f64)` across the 5 timed rounds is reported in the summary tables.
