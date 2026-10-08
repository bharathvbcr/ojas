//! Isolated Mamba2 Conv1D forward timing (no full model overhead).

use std::time::Instant;
use tessl_arch02::mixers::mamba2_conv1d_fwd;

fn main() -> Result<(), String> {
    let rt = tessl_arch02::gpu_runtime()?;
    let b: usize = std::env::var("BENCH_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let t: usize = std::env::var("BENCH_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let c: usize = std::env::var("BENCH_C")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(768);
    let k: usize = 4;
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let warmup: usize = 10;

    let x = rt.alloc_tensor_f32(&[b, t, c])?;
    let w = rt.alloc_tensor_f32(&[c, k])?;
    let bias = rt.alloc_tensor_f32(&[c])?;
    x.buffer.write_f32(&vec![0.01f32; b * t * c]);
    w.buffer.write_f32(&vec![0.01f32; c * k]);
    bias.buffer.write_f32(&vec![0.0f32; c]);

    for _ in 0..warmup {
        let _ = mamba2_conv1d_fwd(&rt, &x, &w, &bias)?;
        rt.synchronize()?;
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = mamba2_conv1d_fwd(&rt, &x, &w, &bias)?;
    }
    rt.synchronize()?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let ms_per = elapsed_ms / iters as f64;
    let elems = (b * t * c) as f64;
    let gflops = (2.0 * elems * k as f64 * iters as f64) / (elapsed_ms * 1e6);

    println!("mamba2_conv1d_fwd | B={b} T={t} C={c} K={k} | {iters} iters after {warmup} warmup");
    println!("  {ms_per:.3} ms/iter | ~{gflops:.2} GFLOP/s (rough MAC count)");
    Ok(())
}
