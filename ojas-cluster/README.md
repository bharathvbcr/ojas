# ojas-cluster

`ojas-cluster` implements **scaling-law `e-means` clustering** (`canonical_v1`, [arXiv:2611.09501](https://arxiv.org/abs/2611.09501)) and a reference full-pass **Lloyd's $k$-means** baseline (`lloyd_fit`) for `ojas`. It executes over multi-core CPU SIMD/BLAS ([`ojas-simd`](../ojas-simd/README.md)) and Apple Silicon Metal 4 (`64x64` cooperative `TensorOps` and `32x32` `simdgroup_float8x8` kernels via [tessl](https://github.com/bharathvbcr/tessl)) behind a single safe-Rust engine and streaming session API (`#![forbid(unsafe_code)]`).

> [!NOTE]
> Following [`docs/modular-extensions.md`](../docs/modular-extensions.md), all generic clustering policy—`canonical_v1` scaling laws, cosine step decay, `SplitMix64` sampling and epoch permutations, `Budget` accounting, and streaming session state machines—lives once in `ojas-cluster`. When built on macOS with the default `metal` feature, `ClusterDevice::Metal` delegates GPU execution to a dedicated worker thread owning a private Metal 4 `GpuRuntime` and `EmeansGpuRunner` in `tessl`.

---

## Architecture & Dispatch

```mermaid
flowchart TD
    subgraph API["Public Surface (src/lib.rs, src/session.rs, src/lloyd.rs)"]
        Fit["emeans_fit / EmeansEngine::fit"]
        Assign["emeans_assign / EmeansEngine::assign"]
        Stream["EmeansSession (begin / accumulate / update / end_epoch / finalize)"]
        Lloyd["lloyd::lloyd_fit (full-pass baseline)"]
    end

    subgraph Policy["Host-Neutral Policy (src/recipe.rs, src/rng.rs)"]
        Recipe["resolve_recipe (BatchLaw, alpha0, fractional epoch_rows)"]
        RNG["SplitMix64 (Forgy init, epoch_permutation, Gumbel keys)"]
        BudgetGuard["ojas_core::Budget (host Scratch + PoolLedger charge)"]
    end

    subgraph CPU["ClusterDevice::Cpu (src/cpu.rs)"]
        CpuFast["CpuMath::Fast (Accelerate AMX / ojas-simd tiles)"]
        CpuDet["CpuMath::DeterministicFma (thread-independent FMA tiles)"]
        CpuExact["CpuMath::Exact (non-FMA ascending f32 dot products)"]
        CpuCSR["Ascending local-row CSR gather + f64 closing pass"]
    end

    subgraph Metal["ClusterDevice::Metal (macOS + metal feature)"]
        Worker["Dedicated Metal worker thread + PoolLedger"]
        TOps["64x64 cooperative TensorOps / 32x32 simdgroup argmin"]
        Radix["64-bit GPU radix sort + 256-row slice CSR gather"]
        Revive["GPU EMA update + SplitMix64 Gumbel split revival"]
    end

    Fit --> Recipe
    Stream --> Recipe
    Assign --> BudgetGuard
    Lloyd --> BudgetGuard
    Recipe --> RNG
    RNG --> BudgetGuard
    BudgetGuard --> CpuFast
    BudgetGuard --> CpuDet
    BudgetGuard --> CpuExact
    CpuFast --> CpuCSR
    CpuDet --> CpuCSR
    CpuExact --> CpuCSR
    BudgetGuard --> Worker
    Worker --> TOps
    TOps --> Radix
    Radix --> Revive
```

---

## `canonical_v1` Recipe & Algorithmic Mechanics

[`resolve_recipe`](src/recipe.rs) translates `(global_n, d, EmeansConfig)` into a [`ResolvedRecipe`](src/lib.rs) before any device memory is reserved:

1. **Subsampling (`max_points_per_centroid`):** When set to $m \ge 1$, [`sample_subset_sorted`](src/rng.rs) deterministically draws $n_{\text{eff}} = \min(n, m \cdot k)$ sorted row indices via partial Fisher-Yates over `SplitMix64`. Unsampled rows receive [`UNASSIGNED_LABEL`](src/lib.rs) (`u32::MAX`) when `capture_assignments` is enabled.
2. **Batch-Size Law (`BatchLaw`):** `BatchLaw::Auto` selects `Quality` ($C_B = 0.00439175$, capped at $\max(1, \lfloor n_{\text{eff}} / 2 \rfloor)$) when $d \le 16$, and `Throughput` ($C_B = 0.035134$, capped at $\text{round\_pow2}(n_{\text{eff}} / 2)$) when $d > 16$:

$$B_0 = \text{round\_pow2}\!\left(C_B \cdot k^{0.134749} \cdot n_{\text{eff}}^{0.876473} \cdot T^{0.813344}\right)$$

3. **Initial EMA Rate & Cosine Schedule:** Unless overridden via `EmeansConfig::alpha0`, the initial EMA rate $\alpha_0$ follows the empirical scaling law clamped to $[10^{-4}, 1.0]$ and decays over `total_steps` via [`scheduled_alpha`](src/recipe.rs):

$$\alpha_0 = \text{clamp}_{[10^{-4},\, 1.0]}\!\left(0.8244 \cdot k^{-0.4617} \cdot B^{0.9003} \cdot n_{\text{eff}}^{-0.4498} \cdot T^{-0.1057}\right)$$

$$\alpha(s) = \alpha_0 \cdot \left(0.1 + 0.9 \cdot \frac{1 + \cos(\pi \cdot \min(s, S) / \max(S, 1))}{2}\right)$$

4. **Fractional Epoch Schedule & Exact Closing Pass:** For $T = \lfloor T \rfloor + f$ with $\lfloor T \rfloor \ge 1$, `epoch_rows` places any fractional remainder pass $\text{round\_ties\_even}(f \cdot n_{\text{eff}})$ **before** a full $n_{\text{eff}}$-row final epoch (for example, $T = 2.6$ on $n = 100{,}000$ executes `[100_000, 60_000, 100_000]`). During the final epoch, every batch accumulates exact closing sums and counts alongside EMA updates so [`EmeansFit::centroids`](src/lib.rs) are finalized in `f64` from full-pass sufficient statistics (retaining the latest EMA centroid for any cluster with zero closing-pass assignments).
5. **EMA Update & Gumbel Dead-Cluster Split Revival:** Each mini-batch updates cluster usage $u_j \leftarrow (1 - \alpha) u_j + \alpha n_j$ (or $u_j \leftarrow \alpha n_j$ when $\alpha = 1$) and blends centroids with exact `weight == 1.0` replacement when $u_{\text{old}} = 0$. On steps $s \ge 1$ where $s \le 8$ or $s \equiv 0 \pmod 4$, clusters with $u_j < \tau \cdot \bar{u}$ ($\tau = 0.1$ for $d > 2$, $0.0$ for $d \le 2$) are revived by splitting live donor clusters ranked by stateless `SplitMix64` Gumbel keys $g_j = \ln(u_j) - \ln(-\ln U_j)$ with alternating $\pm\epsilon$ coordinate perturbations (`split_epsilon = 1e-4`), plus 1-ULP deterministic tie separation when `(1 +/- eps)` underflows representation precision.

---

## Execution Substrates & Precision Lanes

[`ClusterDevice`](src/lib.rs) never falls back silently across devices or features (`OjasError::Unsupported` when `ClusterDevice::Metal` is requested without macOS + `metal` or without a Metal 4 GPU):

| Substrate | Variant | Assignment Math | Accumulation & Closing Stats | Determinism Profile |
| :--- | :--- | :--- | :--- | :--- |
| CPU | `CpuMath::Fast` | Apple Accelerate AMX (macOS unpinned) or L2-tiled `ojas-simd` | Ascending row CSR gather (`f32` batch, `f64` closing divide) | Deterministic per host/OS (`unique_centroid_hashes = 1` across repeated M5 Pro runs); Accelerate AMX may vary across OS/thread heuristics |
| CPU | `CpuMath::DeterministicFma` | L2-tiled `ojas-simd` FMA micro-kernels (`sgemm_tile_with`) | Ascending row CSR gather (`256`-row slices, `f64` closing divide) | Bit-identical (`0` ULP) across arbitrary worker thread counts (`1..N`) |
| CPU | `CpuMath::Exact` | Non-FMA ascending-$d$ `f32` dot products | Ascending row CSR gather (`256`-row slices, `f64` closing divide) | Bit-identical (`0` ULP) across thread counts and CPU microarchitectures |
| Metal 4 | `MetalPrecision::Bf16` | `64x64` `bf16` cooperative `TensorOps` (`32x32` `simdgroup` fallback) | `bf16`-rounded rows gathered in `256`-row radix-CSR slices (`f32` acc) | Bit-identical (`0` ULP) across repeated fits and [`EmeansSession`](src/session.rs) streaming vs resident `fit_with_init` on the same GPU (verified through $k = 1024$) |
| Metal 4 | `MetalPrecision::Bf16AssignF32Stats` | `64x64` `bf16` cooperative `TensorOps` (`32x32` `simdgroup` fallback) | Unrounded `f32` rows gathered in `256`-row radix-CSR slices | Mixed-precision lane with `0`-ULP repeat-run and streaming-vs-resident bit-identity on the same GPU (verified through $k = 1024$) |
| Metal 4 | `MetalPrecision::F32` | `64x64` exact `f32` cooperative `TensorOps` (`RELAXED=false`) | Unrounded `f32` rows gathered in `256`-row radix-CSR slices | Full `f32` pipeline with `0`-ULP repeat-run and streaming-vs-resident bit-identity on the same GPU (verified through $k = 1024$) |

> [!IMPORTANT]
> Every host scratch buffer (`Scratch<T>`) and Metal pool slab (`OjasBudgetLedger` -> `tessl::PoolLedger`) charges [`ojas_core::Budget`](../ojas-core/README.md) **before** allocation and refuses over-cap requests with `OjasError::CapacityExceeded` without mutating engine state or poisoning the GPU runtime. Dropping `EmeansSession` and `EmeansEngine` (or calling `EmeansEngine::trim_pool`) returns `Budget::live_bytes()` to `0`.

---

## Usage

### One-Shot Fit & Assignment

```rust
use ojas_cluster::{emeans_assign, ClusterDevice, CpuMath, EmeansConfig, EmeansEngine};
use ojas_core::{Budget, Tensor};

let budget = Budget::new(256 * 1024 * 1024);
let engine = EmeansEngine::new(
    ClusterDevice::Cpu {
        threads: None,
        math: CpuMath::DeterministicFma,
    },
    &budget,
)?;

let x = Tensor::from_f32(&rows, &[n, d], &budget)?;
let mut cfg = EmeansConfig::new(k);
cfg.nepochs = 2.6;
cfg.seed = 42;

let fit = engine.fit(&x, &cfg)?;
let (labels, total_obj) = engine.assign(
    &x,
    &fit.centroids,
    cfg.spherical,
    cfg.assignment_chunk_size,
)?;
let (_one_shot_labels, _one_shot_obj) =
    emeans_assign(&x, &fit.centroids, cfg.spherical, engine.device(), &budget)?;
```

### Out-of-Core Streaming (`EmeansSession`)

[`EmeansSession`](src/session.rs) accepts arbitrary `[m, d]` `F32` or `Bf16` (`Host` or `Mapped`) row chunks bounded by `1 <= m <= session.remaining_in_batch()`, returning `(AccumulateStep, Option<Vec<u32>>)` and producing a `0`-ULP bit-identical [`EmeansFit`](src/lib.rs) relative to `EmeansEngine::fit_with_init`:

```rust
use ojas_cluster::EmeansSession;

let mut session = EmeansSession::begin(&engine, global_n, &init_centroids, &cfg)?;
let num_epochs = session.recipe().epoch_rows.len();
while session.epoch() < num_epochs {
    while session.remaining_in_epoch() > 0 {
        while session.remaining_in_batch() > 0 {
            let take = next_chunk_len.min(session.remaining_in_batch());
            let (step, _maybe_closing_labels) = session.accumulate(&next_chunk(take))?;
            if step.batch_ready {
                let _revived = session.update()?;
            }
        }
    }
    let _stat = session.end_epoch()?;
}
let fit = session.finalize()?;
```

---

## Apple M5 Pro Benchmark (`e-means` vs 25-Iter Lloyd)

Measured by [`examples/emeans_bench.rs`](examples/emeans_bench.rs) against `tessl @ 17471e7` (1 untimed warmup + 5 cyclic-interleaved paired rounds per shape on Apple M5 Pro, macOS 27.2, `rustc 1.99.1`, 18 CPU pool threads; full raw output and load disclosure in [`bench/results/2026-11-10-emeans`](../bench/results/2026-11-10-emeans/README.md)). All 9 contenders start from identical Forgy `init_centroids`, produce `0`-ULP identical centroid bits across all 6 runs (`unique_centroid_hashes = 1`), and are scored by exact `f64` Euclidean MSE $\frac{1}{n}\sum_i \min_j \|x_i - c_j\|_2^2$:

| Shape | Contender | `min_ms` | `median_ms` | `MSE (f64)` | `MSE vs Lloyd` | `speedup_min` | `speedup_med` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `100k x 128, k=1024` | `Lloyd 25-iter (CpuMath::Fast)` | 5047.01 | 5285.93 | 0.228053 | +0.00% | 1.00x | 1.00x |
| `100k x 128, k=1024` | `e-means T=2.0 (CpuMath::Fast)` | 494.40 | 591.82 | 0.170040 | -25.44% | 10.21x | 8.93x |
| `100k x 128, k=1024` | `e-means T=2.6 (CpuMath::Fast)` | 548.34 | 733.95 | 0.160376 | -29.68% | 9.20x | 7.20x |
| `100k x 128, k=1024` | `e-means T=2.0 (Metal::F32)` | 49.64 | 63.66 | 0.169943 | -25.48% | 101.68x | 83.03x |
| `100k x 128, k=1024` | `e-means T=2.6 (Metal::F32)` | 49.62 | 54.17 | 0.162137 | -28.90% | 101.71x | 97.58x |
| `100k x 128, k=1024` | `e-means T=2.0 (Metal::Bf16AssignF32Stats)` | 50.21 | 57.80 | 0.165726 | -27.33% | 100.53x | 91.45x |
| `100k x 128, k=1024` | `e-means T=2.6 (Metal::Bf16AssignF32Stats)` | 50.77 | 62.44 | 0.160264 | -29.73% | 99.40x | 84.66x |
| `100k x 128, k=1024` | `e-means T=2.0 (Metal::Bf16)` | 48.44 | 56.60 | 0.172521 | -24.35% | 104.20x | 93.38x |
| `100k x 128, k=1024` | `e-means T=2.6 (Metal::Bf16)` | 46.19 | 54.47 | 0.163665 | -28.23% | 109.27x | 97.04x |
| `65.5k x 512, k=1024` | `Lloyd 25-iter (CpuMath::Fast)` | 5991.42 | 6935.69 | 0.248275 | +0.00% | 1.00x | 1.00x |
| `65.5k x 512, k=1024` | `e-means T=2.0 (CpuMath::Fast)` | 654.95 | 866.32 | 0.197876 | -20.30% | 9.15x | 8.01x |
| `65.5k x 512, k=1024` | `e-means T=2.6 (CpuMath::Fast)` | 725.66 | 1014.84 | 0.179865 | -27.55% | 8.26x | 6.83x |
| `65.5k x 512, k=1024` | `e-means T=2.0 (Metal::F32)` | 148.07 | 211.90 | 0.192224 | -22.58% | 40.46x | 32.73x |
| `65.5k x 512, k=1024` | `e-means T=2.6 (Metal::F32)` | 126.32 | 131.04 | 0.178966 | -27.92% | 47.43x | 52.93x |
| `65.5k x 512, k=1024` | `e-means T=2.0 (Metal::Bf16AssignF32Stats)` | 108.61 | 138.45 | 0.199768 | -19.54% | 55.17x | 50.10x |
| `65.5k x 512, k=1024` | `e-means T=2.6 (Metal::Bf16AssignF32Stats)` | 112.95 | 144.79 | 0.181201 | -27.02% | 53.04x | 47.90x |
| `65.5k x 512, k=1024` | `e-means T=2.0 (Metal::Bf16)` | 108.06 | 136.56 | 0.193460 | -22.08% | 55.44x | 50.79x |
| `65.5k x 512, k=1024` | `e-means T=2.6 (Metal::Bf16)` | 91.72 | 128.13 | 0.183158 | -26.23% | 65.33x | 54.13x |

---

## Test Suites (18 integration tests)

- [`tests/recipe_and_rng.rs`](tests/recipe_and_rng.rs) (4 tests): `canonical_v1` batch-law resolution against upstream Python values, fractional epoch ordering and invalid-bound refusals, `SplitMix64` sampling/permutations and spherical prescaling, and collapsed-split 1-ULP separation across `-0.0`, subnormals, and finite extremes.
- [`tests/cpu_and_lloyd.rs`](tests/cpu_and_lloyd.rs) (8 tests): `CpuMath::DeterministicFma` and `CpuMath::Exact` multi-thread bit-identity and thread-ceiling refusal, pinned-thread vs Accelerate panel parity, `f64` closing-pass accumulation without `f32` stall, spherical zero-row/unassigned edge cases, `e-means` `T = 2.6` vs 25-iter `lloyd_fit` convergence, and fail-closed non-finite/overflow/`Budget` refusals.
- [`tests/metal_and_session.rs`](tests/metal_and_session.rs) (6 tests): `faiss/contrib/e_means` `canonical_v1` golden parity across CPU and all three Metal precision lanes, resident `fit` vs streaming `EmeansSession` bit-identity, `k = 1024` multi-epoch Gumbel revival repeat-run and streaming-vs-resident `0`-ULP bit-identity across all three Metal 4 precision lanes, cross-runtime `Mapped` (`SharedBuf`) zero-copy safety and `Budget` trim recovery without `GpuRuntime` poison, 200-iteration engine reuse and early-drop zero-leak checks, and dtype/state-machine refusals.
