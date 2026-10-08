//! Burn/Metal (CubeCL MSL) matmul lane for the cross-runtime GEMM comparison.
//!
//! Protocol pinned to `metal-native/src/bin/bench_gemm_sweep.rs` and
//! `bench/gemm_sweep_mlx.py`: same shapes, sync every iteration, median over
//! iters. Burn has `autotune` on, so warmup is raised — the first calls
//! calibrate kernel selection and are not representative.

use burn::prelude::*;
use burn::tensor::backend::Backend;
use std::time::Instant;

type B = burn::backend::Metal;
/// Burn's Metal backend is generic over the float type; bf16 is the lane that
/// matters for the comparison against metal-native's tuned bf16 GEMM.
type BBf = burn::backend::Metal<burn::tensor::bf16>;

const SHAPES: &[(usize, usize, usize, &str)] = &[
    (512, 512, 512, "square_512"),
    (1024, 1024, 1024, "square_1024"),
    (2048, 2048, 2048, "square_2048"),
    (4096, 4096, 4096, "square_4096"),
    (2048, 768, 768, "qkv_proj"),
    (8192, 3072, 768, "mlp_up"),
    (8192, 768, 3072, "mlp_down"),
    (4096, 4096, 1024, "tall_k1024"),
];

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn main() {
    // Name the concrete device type; Backend::Device resolves awkwardly here
    // (same workaround train.rs uses).
    let device: burn::backend::wgpu::WgpuDevice = Default::default();
    let warmup: usize = std::env::var("BENCH_WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(25);
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);

    println!("== f32 ==\n{:<12}{:>10}{:>12}", "shape", "ms", "GFLOP/s");
    for &(m, n, k, label) in SHAPES {
        let a: Tensor<B, 2> = Tensor::ones([m, k], &device) * 0.5;
        let b: Tensor<B, 2> = Tensor::ones([k, n], &device) * 0.5;
        let _ = B::sync(&device);

        for _ in 0..warmup {
            let c = a.clone().matmul(b.clone());
            let _ = c.into_data();
        }
        let _ = B::sync(&device);

        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let c = a.clone().matmul(b.clone());
            let _ = B::sync(&device);
            std::hint::black_box(&c);
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(samples);
        println!("{label:<12}{med:>10.3}{:>12.0}", (2.0 * m as f64 * n as f64 * k as f64) / (med * 1e6));
    }

    println!("\n== bf16 ==\n{:<12}{:>10}{:>12}", "shape", "ms", "GFLOP/s");
    for &(m, n, k, label) in SHAPES {
        let a: Tensor<BBf, 2> = Tensor::ones([m, k], &device) * 0.5;
        let b: Tensor<BBf, 2> = Tensor::ones([k, n], &device) * 0.5;
        let _ = BBf::sync(&device);
        for _ in 0..warmup {
            let c = a.clone().matmul(b.clone());
            let _ = BBf::sync(&device);
            std::hint::black_box(&c);
        }
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let c = a.clone().matmul(b.clone());
            let _ = BBf::sync(&device);
            std::hint::black_box(&c);
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median(samples);
        println!("{label:<12}{med:>10.3}{:>12.0}", (2.0 * m as f64 * n as f64 * k as f64) / (med * 1e6));
    }
}
