//! A/B wall time of `Backend::linear_backward` against
//! `Backend::linear_backward_input` (the frozen-weight backward) at the
//! Qwen3.5-2B linear shapes (hidden 2048, intermediate 6144, 8 query heads
//! of 256 over 2 KV heads, 16 GDN heads of 128) with T = 2048 rows, on CPU
//! (Fast, 6 threads) and Metal. Ignored by default.
//!
//! Each row interleaves the two arms (A, B, A, B, ...) so drift hits both,
//! and reports the minimum of `N` runs per arm plus the peak budget charge
//! of one call. The dX of the two arms is asserted bit-equal on every row.
//!
//! ```text
//! cargo test -p ojas-model --release --test frozen_linear_bench -- --ignored --nocapture --test-threads=1
//! ```

use std::time::Instant;

use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;

const T: usize = 2048;
const N: usize = 9;

/// `(name, in, out)` of every distinct 2B linear.
const SHAPES: [(&str, usize, usize); 6] = [
    ("attn_q_proj", 2048, 4096),
    ("attn_kv_proj", 2048, 512),
    ("attn_o_gdn_z_gdn_out", 2048, 2048),
    ("gdn_in_proj_qkv", 2048, 6144),
    ("mlp_gate_up", 2048, 6144),
    ("mlp_down", 6144, 2048),
];

fn values(seed: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed * 97)
                % 1000;
            v as f32 / 500.0 - 1.0
        })
        .collect()
}

fn bits(b: &impl Backend, t: &Tensor) -> Vec<u32> {
    t.to_host(b.budget())
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn rows<B: Backend>(label: &str, b: &B, up: impl Fn(&Tensor) -> Tensor) {
    let host = Budget::new(8 << 30);
    for (name, kin, nout) in SHAPES {
        let x = up(&Tensor::from_f32(&values(1, T * kin), &[T, kin], &host).unwrap());
        let w = up(&Tensor::from_f32(&values(2, nout * kin), &[nout, kin], &host).unwrap());
        let g = up(&Tensor::from_f32(&values(3, T * nout), &[T, nout], &host).unwrap());
        let (mut full, mut only) = (Vec::new(), Vec::new());
        let (mut full_peak, mut only_peak) = (0, 0);
        for round in 0..N {
            let before = b.budget().live_bytes().unwrap();
            b.budget().reset_peak();
            let t0 = Instant::now();
            let (dx_full, dw) = b.linear_backward(&x, &w, &g).unwrap();
            b.sync().unwrap();
            full.push(t0.elapsed().as_secs_f64() * 1e3);
            full_peak = b.budget().peak_bytes() - before;
            drop(dw);

            let before = b.budget().live_bytes().unwrap();
            b.budget().reset_peak();
            let t0 = Instant::now();
            let dx_only = b.linear_backward_input(&x, &w, &g).unwrap();
            b.sync().unwrap();
            only.push(t0.elapsed().as_secs_f64() * 1e3);
            only_peak = b.budget().peak_bytes() - before;
            if round == 0 {
                assert_eq!(
                    bits(b, &dx_full),
                    bits(b, &dx_only),
                    "{label} {name}: dX differs"
                );
            }
        }
        let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
        let (f, o) = (min(&full), min(&only));
        println!(
            "OJAS_FROZEN_BENCH backend={label} shape={name} rows={T} in={kin} out={nout} n={N} \
             full_min_ms={f:.3} dx_only_min_ms={o:.3} saving={:.1}% full_peak_bytes={full_peak} dx_only_peak_bytes={only_peak}",
            100.0 * (1.0 - o / f)
        );
    }
}

#[test]
#[ignore = "benchmark; run with --release -- --ignored --nocapture --test-threads=1"]
fn frozen_linear_backward_cpu() {
    let cpu = CpuBackend::with_threads(Budget::new(24 << 30), 6).unwrap();
    cpu.start_workers().unwrap();
    rows("cpu_fast", &cpu, Tensor::clone);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "benchmark; run with --release -- --ignored --nocapture --test-threads=1"]
fn frozen_linear_backward_metal() {
    let m = ojas_metal::MetalBackend::new(Budget::new(24 << 30)).unwrap();
    rows("metal", &m, |t| m.upload(t).unwrap());
}
