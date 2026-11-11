//! Interleaved A/B benchmark comparing 25-iteration Lloyd's `k`-means (`CpuMath::Fast`)
//! against scaling-law `e-means` (`nepochs = 2.0` and `2.6`) across multi-core CPU SIMD/BLAS
//! and Apple Silicon Metal 4 (`F32`, `Bf16AssignF32Stats`, `Bf16`).
//!
//! ```text
//! cargo run --release -p ojas-cluster --example emeans_bench
//! ```

#![forbid(unsafe_code)]

use std::process::Command;
use std::time::Instant;

use ojas_cluster::{
    lloyd_fit, resolve_recipe, ClusterDevice, CpuMath, CpuPool, EmeansConfig, EmeansEngine,
    LloydConfig, MetalPrecision, SplitMix64,
};
use ojas_core::{Budget, OjasError, Tensor};

const ROUNDS: usize = 5;
const LLOYD_ITERS: usize = 25;
const BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const INIT_SEED_XOR: u64 = 0x494e_4954_4345_4e54;

struct ShapeSpec {
    id: &'static str,
    title: &'static str,
    n: usize,
    d: usize,
    k: usize,
    seed: u64,
}

const SHAPES: [ShapeSpec; 2] = [
    ShapeSpec {
        id: "shape_a",
        title: "Shape A (n=100_000, d=128, k=1024)",
        n: 100_000,
        d: 128,
        k: 1_024,
        seed: 0x2026_1110_0128_1024,
    },
    ShapeSpec {
        id: "shape_b",
        title: "Shape B (n=65_536, d=512, k=1024)",
        n: 65_536,
        d: 512,
        k: 1_024,
        seed: 0x2026_1110_0512_1024,
    },
];

struct Workload {
    spec: &'static ShapeSpec,
    x_rows: Vec<f32>,
    x: Tensor,
    init_centroids: Tensor,
    init_mse_f64: f64,
}

enum ContenderKind<'a> {
    Lloyd25 {
        cfg: LloydConfig,
    },
    Emeans {
        engine: &'a EmeansEngine,
        cfg: EmeansConfig,
    },
}

struct Contender<'a> {
    id: &'static str,
    label: &'static str,
    kind: ContenderKind<'a>,
}

#[derive(Clone)]
struct RoundSample {
    elapsed_ms: f64,
    mse_f64: f64,
    non_empty: usize,
    revived: usize,
}

struct CachedMse {
    bits: Vec<u32>,
    mse_f64: f64,
}

fn cmd_line(prog: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(prog).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn detect_cpu_brand() -> String {
    cmd_line("sysctl", &["-n", "machdep.cpu.brand_string"])
        .unwrap_or_else(|| std::env::consts::ARCH.to_string())
}

fn detect_os_version() -> String {
    if let Some(ver) = cmd_line("sw_vers", &["-productVersion"]) {
        format!("macOS {ver} ({})", std::env::consts::ARCH)
    } else {
        format!("{} ({})", std::env::consts::OS, std::env::consts::ARCH)
    }
}

fn detect_rustc_version() -> String {
    cmd_line("rustc", &["--version"]).unwrap_or_else(|| "rustc (unknown)".to_string())
}

fn detect_loadavg() -> String {
    cmd_line("sysctl", &["-n", "vm.loadavg"])
        .map(|s| {
            s.trim_matches(|c| c == '{' || c == '}' || c == ' ')
                .to_string()
        })
        .unwrap_or_else(|| "n/a".to_string())
}

fn try_open_metal(
    precision: MetalPrecision,
    budget: &Budget,
) -> Result<Option<EmeansEngine>, OjasError> {
    let compiled_with_metal = cfg!(all(target_os = "macos", feature = "metal"));
    let allow_no_gpu = std::env::var("OJAS_ALLOW_NO_GPU").ok().as_deref() == Some("1");
    match EmeansEngine::new(ClusterDevice::Metal { precision }, budget) {
        Ok(engine) => Ok(Some(engine)),
        Err(OjasError::Unsupported { .. }) if !compiled_with_metal => Ok(None),
        Err(err) if allow_no_gpu => {
            eprintln!("SKIP (OJAS_ALLOW_NO_GPU=1): Metal {precision:?} unavailable: {err}");
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

#[inline]
fn standard_normal_pair(rng: &mut SplitMix64) -> (f64, f64) {
    let u1 = ((rng.next_u64() >> 11) | 1) as f64 * (1.0 / ((1_u64 << 53) as f64));
    let u2 = (rng.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64));
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

fn fill_normal_f32(rng: &mut SplitMix64, scale: f64, dst: &mut [f32]) {
    let mut i = 0;
    while i + 1 < dst.len() {
        let (z0, z1) = standard_normal_pair(rng);
        dst[i] = (z0 * scale) as f32;
        dst[i + 1] = (z1 * scale) as f32;
        i += 2;
    }
    if i < dst.len() {
        let (z0, _) = standard_normal_pair(rng);
        dst[i] = (z0 * scale) as f32;
    }
}

/// Build a deterministic Zipf-weighted Gaussian mixture in `R^d` plus shared Forgy `init_centroids`.
///
/// Component centers have unit expected L2 norm (`sigma_center = 1 / sqrt(d)`) and within-cluster
/// perturbations have expected L2 norm `0.25` (`sigma_noise = 0.25 / sqrt(d)`). Power-law component
/// weights `w_j proportional to (j + 16)^(-0.75)` produce realistic cluster mass skew so uniform
/// Forgy initialization seeds multiple centroids inside dense modes while leaving tail clusters
/// unseeded, exercising both MSE convergence and `e-means` Gumbel split revival.
fn build_workload(
    spec: &'static ShapeSpec,
    eval_workers: usize,
    budget: &Budget,
) -> Result<Workload, OjasError> {
    let n = spec.n;
    let d = spec.d;
    let k = spec.k;
    let mut rng = SplitMix64::new(spec.seed);

    let inv_sqrt_d = 1.0 / (d as f64).sqrt();
    let center_scale = inv_sqrt_d;
    let noise_scale = 0.25 * inv_sqrt_d;

    let mut true_centers = vec![0.0_f32; k * d];
    fill_normal_f32(&mut rng, center_scale, &mut true_centers);

    let mut cdf = Vec::with_capacity(k);
    let mut weight_sum = 0.0_f64;
    for j in 0..k {
        weight_sum += ((j + 16) as f64).powf(-0.75);
        cdf.push(weight_sum);
    }
    for val in &mut cdf {
        *val /= weight_sum;
    }
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }

    let mut x_rows = vec![0.0_f32; n * d];
    let mut noise_buf = vec![0.0_f32; d];
    for r in 0..n {
        let u = (rng.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64));
        let comp = cdf.partition_point(|&p| p < u).min(k - 1);
        fill_normal_f32(&mut rng, noise_scale, &mut noise_buf);
        let c_row = &true_centers[comp * d..(comp + 1) * d];
        let x_row = &mut x_rows[r * d..(r + 1) * d];
        for col in 0..d {
            x_row[col] = c_row[col] + noise_buf[col];
        }
    }

    // Shared Forgy initialization: sample `k` distinct rows uniformly without replacement.
    let mut indices: Vec<u32> = (0..(n as u32)).collect();
    let mut init_rng = SplitMix64::new(SplitMix64::mix(spec.seed ^ INIT_SEED_XOR));
    for i in 0..k {
        let rem = (n - i) as u64;
        let pick = i + (init_rng.next_bounded_u64(rem) as usize);
        indices.swap(i, pick);
    }
    let mut init_c = vec![0.0_f32; k * d];
    for (j, &row_idx) in indices[..k].iter().enumerate() {
        let src_start = (row_idx as usize) * d;
        init_c[j * d..(j + 1) * d].copy_from_slice(&x_rows[src_start..src_start + d]);
    }

    let init_mse_f64 = exact_euclidean_mse_f64(&x_rows, &init_c, n, d, k, eval_workers);
    let x = Tensor::from_f32(&x_rows, &[n, d], budget)?;
    let init_centroids = Tensor::from_f32(&init_c, &[k, d], budget)?;

    Ok(Workload {
        spec,
        x_rows,
        x,
        init_centroids,
        init_mse_f64,
    })
}

/// Evaluate exact `f64` mean squared Euclidean distance `mean_i min_j sum_col (x[i,col] - c[j,col])^2`
/// with zero `||x||^2 + ||c||^2 - 2<x,c>` cancellation.
fn exact_euclidean_mse_f64(
    x: &[f32],
    centroids: &[f32],
    n: usize,
    d: usize,
    k: usize,
    workers: usize,
) -> f64 {
    assert_eq!(x.len(), n * d);
    assert_eq!(centroids.len(), k * d);

    let c_f64: Vec<f64> = centroids.iter().map(|&v| f64::from(v)).collect();
    let num_workers = workers.max(1).min(n);
    let rows_per_worker = n.div_ceil(num_workers);

    let partial_sums: Vec<f64> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(num_workers);
        let c_ref = &c_f64;
        for w in 0..num_workers {
            let r_start = w * rows_per_worker;
            if r_start >= n {
                break;
            }
            let r_end = (r_start + rows_per_worker).min(n);
            let x_slice = &x[r_start * d..r_end * d];
            handles.push(scope.spawn(move || {
                let mut x_row_f64 = vec![0.0_f64; d];
                let mut local_sum = 0.0_f64;
                let k4 = k / 4 * 4;
                for r in 0..(r_end - r_start) {
                    let xr = &x_slice[r * d..(r + 1) * d];
                    for (dst, &src) in x_row_f64.iter_mut().zip(xr.iter()) {
                        *dst = f64::from(src);
                    }
                    let mut best = f64::INFINITY;
                    let mut j = 0_usize;
                    while j < k4 {
                        let c0 = &c_ref[j * d..(j + 1) * d];
                        let c1 = &c_ref[(j + 1) * d..(j + 2) * d];
                        let c2 = &c_ref[(j + 2) * d..(j + 3) * d];
                        let c3 = &c_ref[(j + 3) * d..(j + 4) * d];
                        let mut s0 = 0.0_f64;
                        let mut s1 = 0.0_f64;
                        let mut s2 = 0.0_f64;
                        let mut s3 = 0.0_f64;
                        for col in 0..d {
                            let xv = x_row_f64[col];
                            let d0 = xv - c0[col];
                            let d1 = xv - c1[col];
                            let d2 = xv - c2[col];
                            let d3 = xv - c3[col];
                            s0 += d0 * d0;
                            s1 += d1 * d1;
                            s2 += d2 * d2;
                            s3 += d3 * d3;
                        }
                        if s0 < best {
                            best = s0;
                        }
                        if s1 < best {
                            best = s1;
                        }
                        if s2 < best {
                            best = s2;
                        }
                        if s3 < best {
                            best = s3;
                        }
                        j += 4;
                    }
                    while j < k {
                        let cj = &c_ref[j * d..(j + 1) * d];
                        let mut sj = 0.0_f64;
                        for col in 0..d {
                            let diff = x_row_f64[col] - cj[col];
                            sj += diff * diff;
                        }
                        if sj < best {
                            best = sj;
                        }
                        j += 1;
                    }
                    local_sum += best;
                }
                local_sum
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("MSE evaluation thread panicked"))
            .collect()
    });

    let mut total = 0.0_f64;
    for part in partial_sums {
        total += part;
    }
    total / (n as f64)
}

fn run_contender_once(
    contender: &Contender<'_>,
    workload: &Workload,
    budget: &Budget,
    eval_workers: usize,
    mse_cache: &mut Vec<CachedMse>,
) -> Result<RoundSample, OjasError> {
    match &contender.kind {
        ContenderKind::Lloyd25 { cfg } => {
            let t0 = Instant::now();
            let fit = lloyd_fit(&workload.x, Some(&workload.init_centroids), cfg, budget)?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;

            assert_eq!(
                fit.iter_objectives.len(),
                LLOYD_ITERS,
                "lloyd_fit must execute all {LLOYD_ITERS} passes unconditionally (tol = 0.0)"
            );
            assert_eq!(fit.labels.len(), workload.spec.n);
            let c_slice = fit.centroids.f32_slice()?;
            let mse_f64 = lookup_or_eval_mse(
                &workload.x_rows,
                c_slice,
                workload.spec.n,
                workload.spec.d,
                workload.spec.k,
                eval_workers,
                mse_cache,
            );
            let non_empty = fit.cluster_counts.iter().filter(|&&c| c > 0).count();
            Ok(RoundSample {
                elapsed_ms,
                mse_f64,
                non_empty,
                revived: 0,
            })
        }
        ContenderKind::Emeans { engine, cfg } => {
            let t0 = Instant::now();
            let fit = engine.fit_with_init(&workload.x, &workload.init_centroids, cfg)?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;

            assert_eq!(fit.labels.len(), workload.spec.n);
            let c_slice = fit.centroids.f32_slice()?;
            let mse_f64 = lookup_or_eval_mse(
                &workload.x_rows,
                c_slice,
                workload.spec.n,
                workload.spec.d,
                workload.spec.k,
                eval_workers,
                mse_cache,
            );
            let non_empty = fit.cluster_counts.iter().filter(|&&c| c > 0).count();
            let revived = fit.epoch_stats.iter().map(|s| s.revived_clusters).sum();
            Ok(RoundSample {
                elapsed_ms,
                mse_f64,
                non_empty,
                revived,
            })
        }
    }
}

fn lookup_or_eval_mse(
    x_rows: &[f32],
    centroids: &[f32],
    n: usize,
    d: usize,
    k: usize,
    eval_workers: usize,
    cache: &mut Vec<CachedMse>,
) -> f64 {
    let bits: Vec<u32> = centroids.iter().map(|v| v.to_bits()).collect();
    if let Some(hit) = cache.iter().find(|entry| entry.bits == bits) {
        return hit.mse_f64;
    }
    let mse_f64 = exact_euclidean_mse_f64(x_rows, centroids, n, d, k, eval_workers);
    cache.push(CachedMse { bits, mse_f64 });
    mse_f64
}

fn min_f64(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_f64(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn median_f64(xs: &[f64]) -> f64 {
    let mut sorted = xs.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        0.5 * (sorted[mid - 1] + sorted[mid])
    }
}

fn emeans_cfg(k: usize, nepochs: f64, seed: u64) -> EmeansConfig {
    let mut cfg = EmeansConfig::new(k);
    cfg.nepochs = nepochs;
    cfg.seed = seed;
    cfg.capture_assignments = true;
    cfg
}

fn run_bench() -> Result<(), OjasError> {
    let budget = Budget::new(BUDGET_BYTES);
    let pool_probe = CpuPool::new(None)?;
    let cpu_pool_threads = pool_probe.threads();
    drop(pool_probe);
    let host_parallelism = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let eval_workers = host_parallelism.clamp(1, 12);

    let cpu_engine = EmeansEngine::new(
        ClusterDevice::Cpu {
            threads: None,
            math: CpuMath::Fast,
        },
        &budget,
    )?;
    let metal_f32 = try_open_metal(MetalPrecision::F32, &budget)?;
    let metal_mixed = try_open_metal(MetalPrecision::Bf16AssignF32Stats, &budget)?;
    let metal_bf16 = try_open_metal(MetalPrecision::Bf16, &budget)?;
    let metal_active = metal_f32.is_some() && metal_mixed.is_some() && metal_bf16.is_some();

    println!("# ojas-cluster e-means vs 25-iter Lloyd A/B benchmark");
    println!("cpu_brand: {}", detect_cpu_brand());
    println!("os: {}", detect_os_version());
    println!("rustc: {}", detect_rustc_version());
    println!(
        "cpu_threads: pool={} host_available_parallelism={} simd_backend={}",
        cpu_pool_threads,
        host_parallelism,
        ojas_simd::backend_name()
    );
    println!(
        "metal_4_gpu: {}",
        if metal_active {
            "enabled (F32, Bf16AssignF32Stats, Bf16)"
        } else {
            "skipped"
        }
    );
    println!(
        "protocol: 1 untimed warmup per contender + {ROUNDS} cyclic-interleaved paired rounds per shape"
    );
    println!("objective: exact f64 Euclidean MSE = (1/n) * sum_i min_j ||x_i - c_j||_2^2");
    println!("loadavg_start: {}", detect_loadavg());
    println!();

    for spec in &SHAPES {
        let workload = build_workload(spec, eval_workers, &budget)?;
        let cfg_20 = emeans_cfg(spec.k, 2.0, spec.seed);
        let cfg_26 = emeans_cfg(spec.k, 2.6, spec.seed);
        let rec_20 = resolve_recipe(spec.n, spec.d, &cfg_20)?;
        let rec_26 = resolve_recipe(spec.n, spec.d, &cfg_26)?;

        let mut lloyd_cfg = LloydConfig::new(spec.k);
        lloyd_cfg.max_iters = LLOYD_ITERS;
        lloyd_cfg.tol = 0.0;
        lloyd_cfg.seed = spec.seed;
        lloyd_cfg.threads = None;
        lloyd_cfg.math = CpuMath::Fast;

        let mut contenders = vec![
            Contender {
                id: "lloyd_25_cpu_fast",
                label: "Lloyd 25-iter (CpuMath::Fast)",
                kind: ContenderKind::Lloyd25 { cfg: lloyd_cfg },
            },
            Contender {
                id: "emeans_t2.0_cpu_fast",
                label: "e-means T=2.0 (CpuMath::Fast)",
                kind: ContenderKind::Emeans {
                    engine: &cpu_engine,
                    cfg: cfg_20.clone(),
                },
            },
            Contender {
                id: "emeans_t2.6_cpu_fast",
                label: "e-means T=2.6 (CpuMath::Fast)",
                kind: ContenderKind::Emeans {
                    engine: &cpu_engine,
                    cfg: cfg_26.clone(),
                },
            },
        ];

        if let Some(ref eng) = metal_f32 {
            contenders.push(Contender {
                id: "emeans_t2.0_metal_f32",
                label: "e-means T=2.0 (Metal::F32)",
                kind: ContenderKind::Emeans {
                    engine: eng,
                    cfg: cfg_20.clone(),
                },
            });
            contenders.push(Contender {
                id: "emeans_t2.6_metal_f32",
                label: "e-means T=2.6 (Metal::F32)",
                kind: ContenderKind::Emeans {
                    engine: eng,
                    cfg: cfg_26.clone(),
                },
            });
        }
        if let Some(ref eng) = metal_mixed {
            contenders.push(Contender {
                id: "emeans_t2.0_metal_bf16_f32stats",
                label: "e-means T=2.0 (Metal::Bf16AssignF32Stats)",
                kind: ContenderKind::Emeans {
                    engine: eng,
                    cfg: cfg_20.clone(),
                },
            });
            contenders.push(Contender {
                id: "emeans_t2.6_metal_bf16_f32stats",
                label: "e-means T=2.6 (Metal::Bf16AssignF32Stats)",
                kind: ContenderKind::Emeans {
                    engine: eng,
                    cfg: cfg_26.clone(),
                },
            });
        }
        if let Some(ref eng) = metal_bf16 {
            contenders.push(Contender {
                id: "emeans_t2.0_metal_bf16",
                label: "e-means T=2.0 (Metal::Bf16)",
                kind: ContenderKind::Emeans {
                    engine: eng,
                    cfg: cfg_20,
                },
            });
            contenders.push(Contender {
                id: "emeans_t2.6_metal_bf16",
                label: "e-means T=2.6 (Metal::Bf16)",
                kind: ContenderKind::Emeans {
                    engine: eng,
                    cfg: cfg_26,
                },
            });
        }

        println!(
            "================================================================================"
        );
        println!("{}", spec.title);
        println!(
            "shape_meta: id={} n={} d={} k={} seed=0x{:016x} init_mse_f64={:.6}",
            spec.id, spec.n, spec.d, spec.k, spec.seed, workload.init_mse_f64
        );
        println!(
            "recipe_t2.0: law={:?} batch_size={} alpha0={:.6} epoch_rows={:?} total_steps={}",
            rec_20.resolved_batch_law,
            rec_20.batch_size,
            rec_20.alpha0,
            rec_20.epoch_rows,
            rec_20.total_steps
        );
        println!(
            "recipe_t2.6: law={:?} batch_size={} alpha0={:.6} epoch_rows={:?} total_steps={}",
            rec_26.resolved_batch_law,
            rec_26.batch_size,
            rec_26.alpha0,
            rec_26.epoch_rows,
            rec_26.total_steps
        );
        println!("loadavg_before_{}: {}", spec.id, detect_loadavg());

        let num_c = contenders.len();
        let mut mse_caches: Vec<Vec<CachedMse>> = (0..num_c).map(|_| Vec::new()).collect();
        let mut samples: Vec<Vec<RoundSample>> =
            (0..num_c).map(|_| Vec::with_capacity(ROUNDS)).collect();

        // Untimed steady-state warmup on every contender.
        for (c_idx, contender) in contenders.iter().enumerate() {
            let warm = run_contender_once(
                contender,
                &workload,
                &budget,
                eval_workers,
                &mut mse_caches[c_idx],
            )?;
            println!(
                "WARMUP shape={} contender={} elapsed_ms={:.3} mse_f64={:.6} non_empty={} revived={}",
                spec.id, contender.id, warm.elapsed_ms, warm.mse_f64, warm.non_empty, warm.revived
            );
        }

        // 5 cyclic-interleaved paired rounds across all contenders.
        for round in 0..ROUNDS {
            let mut round_slots: Vec<Option<RoundSample>> = (0..num_c).map(|_| None).collect();
            for offset in 0..num_c {
                let c_idx = (round + offset) % num_c;
                let sample = run_contender_once(
                    &contenders[c_idx],
                    &workload,
                    &budget,
                    eval_workers,
                    &mut mse_caches[c_idx],
                )?;
                round_slots[c_idx] = Some(sample);
            }
            for (c_idx, slot) in round_slots.into_iter().enumerate() {
                let s = slot.expect("every contender ran in round");
                println!(
                    "ROUND shape={} round={} contender={} elapsed_ms={:.3} mse_f64={:.6} non_empty={} revived={}",
                    spec.id,
                    round + 1,
                    contenders[c_idx].id,
                    s.elapsed_ms,
                    s.mse_f64,
                    s.non_empty,
                    s.revived
                );
                samples[c_idx].push(s);
            }
        }

        println!("loadavg_after_{}: {}", spec.id, detect_loadavg());
        println!();

        let lloyd_times: Vec<f64> = samples[0].iter().map(|s| s.elapsed_ms).collect();
        let lloyd_mses: Vec<f64> = samples[0].iter().map(|s| s.mse_f64).collect();
        let lloyd_min_ms = min_f64(&lloyd_times);
        let lloyd_med_ms = median_f64(&lloyd_times);
        let lloyd_med_mse = median_f64(&lloyd_mses);

        println!(
            "| {:<41} | {:>9} | {:>9} | {:>9} | {:>15} | {:>10} | {:>16} | {:>11} | {:>11} | {:>21} | {:>9} | {:>7} |",
            "Contender",
            "min_ms",
            "median_ms",
            "max_ms",
            "spread_ms (%)",
            "MSE (f64)",
            "MSE vs Lloyd",
            "speedup_min",
            "speedup_med",
            "paired_med [min-max]",
            "non_empty",
            "revived"
        );
        println!(
            "| :{:-<40} | {:-<8}: | {:-<8}: | {:-<8}: | {:-<14}: | {:-<9}: | {:-<15}: | {:-<10}: | {:-<10}: | :{:-<20} | {:-<8}: | {:-<6}: |",
            "", "", "", "", "", "", "", "", "", "", "", ""
        );

        struct RowSummary {
            id: &'static str,
            label: &'static str,
            c_min: f64,
            c_med: f64,
            c_max: f64,
            spread_ms: f64,
            spread_pct: f64,
            c_mse: f64,
            mse_delta_pct: f64,
            mse_ratio: f64,
            speedup_min: f64,
            speedup_med: f64,
            paired_med: f64,
            paired_min: f64,
            paired_max: f64,
            non_empty: usize,
            revived: usize,
            unique_hashes: usize,
        }

        let mut summaries = Vec::with_capacity(num_c);
        for (c_idx, contender) in contenders.iter().enumerate() {
            let times: Vec<f64> = samples[c_idx].iter().map(|s| s.elapsed_ms).collect();
            let mses: Vec<f64> = samples[c_idx].iter().map(|s| s.mse_f64).collect();
            let paired_ratios: Vec<f64> = lloyd_times
                .iter()
                .zip(times.iter())
                .map(|(&lt, &ct)| lt / ct)
                .collect();

            let c_min = min_f64(&times);
            let c_med = median_f64(&times);
            let c_max = max_f64(&times);
            let spread_ms = c_max - c_min;
            let spread_pct = (c_max / c_min - 1.0) * 100.0;

            let c_mse = median_f64(&mses);
            let mse_delta_pct = (c_mse - lloyd_med_mse) / lloyd_med_mse * 100.0;
            let mse_ratio = c_mse / lloyd_med_mse;

            let speedup_min = lloyd_min_ms / c_min;
            let speedup_med = lloyd_med_ms / c_med;
            let paired_med = median_f64(&paired_ratios);
            let paired_min = min_f64(&paired_ratios);
            let paired_max = max_f64(&paired_ratios);

            let last = &samples[c_idx][ROUNDS - 1];
            summaries.push(RowSummary {
                id: contender.id,
                label: contender.label,
                c_min,
                c_med,
                c_max,
                spread_ms,
                spread_pct,
                c_mse,
                mse_delta_pct,
                mse_ratio,
                speedup_min,
                speedup_med,
                paired_med,
                paired_min,
                paired_max,
                non_empty: last.non_empty,
                revived: last.revived,
                unique_hashes: mse_caches[c_idx].len(),
            });
        }

        for row in &summaries {
            println!(
                "| {:<41} | {:>9.2} | {:>9.2} | {:>9.2} | {:>7.2} ({:>4.1}%) | {:>10.6} | {:>+6.2}% ({:.4}x) | {:>10.2}x | {:>10.2}x | {:>5.2}x [{:>5.2}-{:>5.2}x] | {:>4}/{:<4} | {:>7} |",
                row.label,
                row.c_min,
                row.c_med,
                row.c_max,
                row.spread_ms,
                row.spread_pct,
                row.c_mse,
                row.mse_delta_pct,
                row.mse_ratio,
                row.speedup_min,
                row.speedup_med,
                row.paired_med,
                row.paired_min,
                row.paired_max,
                row.non_empty,
                spec.k,
                row.revived,
            );
        }
        println!();

        for row in &summaries {
            println!(
                "SUMMARY shape={} contender={} min_ms={:.3} median_ms={:.3} max_ms={:.3} spread_ms={:.3} spread_pct={:.2} mse_f64={:.8} mse_delta_pct={:+.4} mse_ratio={:.6} speedup_min={:.3} speedup_med={:.3} paired_speedup_med={:.3} non_empty={} revived={} unique_centroid_hashes={}",
                spec.id,
                row.id,
                row.c_min,
                row.c_med,
                row.c_max,
                row.spread_ms,
                row.spread_pct,
                row.c_mse,
                row.mse_delta_pct,
                row.mse_ratio,
                row.speedup_min,
                row.speedup_med,
                row.paired_med,
                row.non_empty,
                row.revived,
                row.unique_hashes,
            );
        }
        println!();
    }

    println!("peak_budget_bytes: {}", budget.peak_bytes());
    println!("live_budget_bytes_before_drop: {}", budget.live_bytes()?);
    Ok(())
}

fn main() {
    if let Err(err) = run_bench() {
        eprintln!("emeans_bench failed: {err}");
        std::process::exit(1);
    }
}
