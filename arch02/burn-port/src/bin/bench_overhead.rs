//! Isolate Burn's fixed per-op dispatch cost from GPU time, so the matmul
//! comparison is not just measuring framework overhead.
use burn::prelude::*;
use burn::tensor::backend::Backend;
use std::time::Instant;

type B = burn::backend::Metal;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let device: burn::backend::wgpu::WgpuDevice = Default::default();
    let iters = 40usize;

    // Bare sync with no queued work: the floor of the measurement loop itself.
    for _ in 0..25 { let _ = B::sync(&device); }
    let mut s = Vec::new();
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = B::sync(&device);
        s.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    println!("bare sync                 {:>8.3} ms", median(s));

    for &n in &[16usize, 64, 128, 256, 512] {
        let a: Tensor<B, 2> = Tensor::ones([n, n], &device) * 0.5;
        let b: Tensor<B, 2> = Tensor::ones([n, n], &device) * 0.5;
        let _ = B::sync(&device);
        for _ in 0..25 { let c = a.clone().matmul(b.clone()); let _ = B::sync(&device); std::hint::black_box(&c); }
        let mut s = Vec::new();
        for _ in 0..iters {
            let t0 = Instant::now();
            let c = a.clone().matmul(b.clone());
            let _ = B::sync(&device);
            std::hint::black_box(&c);
            s.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let med = median(s);
        println!("matmul {n:>4}^3            {med:>8.3} ms   {:>8.0} GFLOP/s",
                 (2.0 * (n as f64).powi(3)) / (med * 1e6));
    }

    // Rule out lazy-operand recomputation: build operands from concrete data so
    // nothing upstream of the matmul can be re-evaluated per iteration.
    for &n in &[16usize, 512] {
        let host = vec![0.5f32; n * n];
        let a: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(host.clone(), [n, n]), &device);
        let b: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(host, [n, n]), &device);
        let _ = a.clone().into_data();
        let _ = b.clone().into_data();
        for _ in 0..25 { let c = a.clone().matmul(b.clone()); let _ = B::sync(&device); std::hint::black_box(&c); }
        let mut s = Vec::new();
        for _ in 0..iters {
            let t0 = Instant::now();
            let c = a.clone().matmul(b.clone());
            let _ = B::sync(&device);
            std::hint::black_box(&c);
            s.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        println!("matmul {n:>4}^3 (materialized) {:>8.3} ms", median(s));
    }

    // Cross-check that sync really forces the work: compare against a forced
    // readback, which cannot be elided.
    let n = 1024usize;
    let a: Tensor<B, 2> = Tensor::ones([n, n], &device) * 0.5;
    let b: Tensor<B, 2> = Tensor::ones([n, n], &device) * 0.5;
    for _ in 0..10 { let _ = a.clone().matmul(b.clone()).into_data(); }
    let mut s = Vec::new();
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = a.clone().matmul(b.clone()).into_data();
        s.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    println!("matmul 1024^3 + readback  {:>8.3} ms", median(s));
}
