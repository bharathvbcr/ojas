//! candle/Metal matmul lane for the cross-runtime GEMM comparison.
//!
//! Protocol pinned to the metal-native, MLX/PyTorch and Burn lanes: same shapes,
//! synchronize every iteration, median over iters. Also measures candle's fixed
//! per-op cost (as was done for Burn) so kernel throughput can be separated from
//! framework dispatch overhead, and spot-checks numerics so a fast-but-wrong
//! result cannot be reported as a win.

use candle_core::{DType, Device, Tensor};
use std::time::Instant;

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

fn bench(dev: &Device, dt: DType, warmup: usize, iters: usize) -> candle_core::Result<()> {
    println!("\n== {dt:?} ==\n{:<12}{:>10}{:>12}", "shape", "ms", "GFLOP/s");
    for &(m, n, k, label) in SHAPES {
        let a = (Tensor::ones((m, k), dt, dev)? * 0.5)?;
        let b = (Tensor::ones((k, n), dt, dev)? * 0.5)?;
        dev.synchronize()?;
        for _ in 0..warmup {
            let c = a.matmul(&b)?;
            dev.synchronize()?;
            std::hint::black_box(&c);
        }
        let mut s = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let c = a.matmul(&b)?;
            dev.synchronize()?;
            std::hint::black_box(&c);
            s.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let med = median(s);
        println!("{label:<12}{med:>10.3}{:>12.0}",
                 (2.0 * m as f64 * n as f64 * k as f64) / (med * 1e6));
    }
    Ok(())
}

fn main() -> candle_core::Result<()> {
    let dev = Device::new_metal(0)?;
    let warmup: usize = std::env::var("BENCH_WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);

    // Numerics spot-check: a fast wrong kernel must not be reported as a win.
    let n = 256usize;
    let host: Vec<f32> = (0..n * n).map(|i| ((i % 17) as f32 - 8.0) / 32.0).collect();
    let a = Tensor::from_vec(host.clone(), (n, n), &dev)?;
    let b = Tensor::from_vec(host.clone(), (n, n), &dev)?;
    let got = a.matmul(&b)?.to_vec2::<f32>()?;
    let mut worst = 0f64;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0f64;
            for kk in 0..n { acc += host[i * n + kk] as f64 * host[kk * n + j] as f64; }
            worst = worst.max((got[i][j] as f64 - acc).abs());
        }
    }
    println!("numerics check (256^3 f32): max abs err {worst:.3e}");
    assert!(worst < 1e-2, "candle matmul is numerically wrong; timings would be meaningless");

    // Fixed per-op cost: a 16^3 matmul is ~0 FLOP, so whatever it costs is overhead.
    for &sz in &[16usize, 512] {
        let a = (Tensor::ones((sz, sz), DType::F32, &dev)? * 0.5)?;
        let b = (Tensor::ones((sz, sz), DType::F32, &dev)? * 0.5)?;
        for _ in 0..20 { let c = a.matmul(&b)?; dev.synchronize()?; std::hint::black_box(&c); }
        let mut s = Vec::new();
        for _ in 0..40 {
            let t0 = Instant::now();
            let c = a.matmul(&b)?;
            dev.synchronize()?;
            std::hint::black_box(&c);
            s.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        println!("per-op floor {sz:>4}^3: {:>8.3} ms", median(s));
    }

    bench(&dev, DType::F32, warmup, iters)?;
    bench(&dev, DType::BF16, warmup, iters)?;
    Ok(())
}
