//! A/B of the per-head gate pair with and without the saved sigmoid, generic
//! over `ojas_core::Backend`. Included by `#[path]` into
//! `ojas-metal/tests/gate_saved.rs` and `ojas-wgpu/tests/gate_saved.rs`
//! (their ignored `bench_*` tests).
//!
//! The shape is nanolab's: 4096 rows (batch 4 x 1024), d_model 768, 12
//! heads of 64. Each sample is one call plus `Backend::sync`, so it is the
//! device time of that call with the submit and wait. A round times every
//! variant once per iteration, alternating which goes first, so drift and
//! thermal state hit both sides alike. Reported: min and median over all
//! iterations, and the ratio of the minima.

use std::time::Instant;

use ojas_core::{Backend, Budget, OjasError, Tensor};

pub const ROWS: usize = 4096;
pub const D_MODEL: usize = 768;
pub const HEADS: usize = 12;
pub const HEAD_DIM: usize = 64;

/// Deterministic values in `[-scale, scale)`.
fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 40) as f32 / (1u64 << 24) as f32;
            (2.0 * u - 1.0) * scale
        })
        .collect()
}

pub struct Stat {
    pub name: &'static str,
    pub min_us: f64,
    pub median_us: f64,
}

fn stat(name: &'static str, mut us: Vec<f64>) -> Stat {
    us.sort_by(f64::total_cmp);
    Stat {
        name,
        min_us: us[0],
        median_us: us[us.len() / 2],
    }
}

fn timed<B: Backend>(b: &B, f: &mut dyn FnMut() -> Result<(), OjasError>) -> f64 {
    let start = Instant::now();
    f().expect("bench op");
    b.sync().expect("bench sync");
    start.elapsed().as_secs_f64() * 1e6
}

/// Time the four variants over `rounds` rounds of `iters` iterations each,
/// after `warmup` untimed calls of each. Returns forward, forward_saving,
/// backward, backward_saved.
pub fn run<B: Backend>(b: &B, warmup: usize, rounds: usize, iters: usize) -> [Stat; 4] {
    let host = Budget::new(1 << 30);
    let up = |shape: &[usize], seed: u64, scale: f32| -> Tensor {
        let n = shape.iter().product();
        b.upload(&Tensor::from_f32(&values(n, seed, scale), shape, &host).expect("host"))
            .expect("upload")
    };
    let x = up(&[ROWS, D_MODEL], 1, 1.0);
    let w = up(&[HEADS, D_MODEL], 2, 0.0625);
    let bias = up(&[HEADS], 3, 0.5);
    let attn = up(&[ROWS, HEADS, HEAD_DIM], 4, 1.0);
    let gy = up(&[ROWS, HEADS, HEAD_DIM], 5, 1.0);
    let (_, scales) = b
        .per_head_sigmoid_gate_forward_saving(&x, &w, &bias, &attn)
        .expect("forward_saving");
    let scales = scales.expect("this backend keeps the gate scale");
    b.sync().expect("sync");

    let mut fwd = || {
        b.per_head_sigmoid_gate_forward(&x, &w, &bias, &attn)
            .map(drop)
    };
    let mut fwd_save = || {
        b.per_head_sigmoid_gate_forward_saving(&x, &w, &bias, &attn)
            .map(drop)
    };
    let mut bwd = || {
        b.per_head_sigmoid_gate_backward(&x, &w, &bias, &attn, &gy)
            .map(drop)
    };
    let mut bwd_saved = || {
        b.per_head_sigmoid_gate_backward_saved(&x, &w, &bias, &attn, &gy, &scales)
            .map(drop)
    };
    let mut variants: [&mut dyn FnMut() -> Result<(), OjasError>; 4] =
        [&mut fwd, &mut fwd_save, &mut bwd, &mut bwd_saved];
    for f in variants.iter_mut() {
        for _ in 0..warmup {
            timed(b, &mut **f);
        }
    }
    let mut samples: [Vec<f64>; 4] = Default::default();
    for round in 0..rounds {
        for _ in 0..iters {
            // Each pair (plain, saved) alternates which runs first.
            for pair in [[0usize, 1], [2, 3]] {
                let order = if round % 2 == 0 {
                    pair
                } else {
                    [pair[1], pair[0]]
                };
                for i in order {
                    let t = timed(b, &mut *variants[i]);
                    samples[i].push(t);
                }
            }
        }
    }
    let [s0, s1, s2, s3] = samples;
    [
        stat("forward", s0),
        stat("forward_saving", s1),
        stat("backward (recomputes logits)", s2),
        stat("backward_saved", s3),
    ]
}

/// The table `run` measured, as markdown, with the saved/plain ratios.
pub fn report(device: &str, rounds: usize, iters: usize, s: &[Stat; 4]) -> String {
    let mut out = format!(
        "device: {device}; {ROWS} rows, d_model {D_MODEL}, {HEADS} heads of {HEAD_DIM}; \
         {rounds} rounds x {iters} iterations, alternating order; each sample is one call \
         plus Backend::sync\n\n| variant | min µs | median µs |\n|---|---:|---:|\n"
    );
    for v in s {
        out.push_str(&format!(
            "| {} | {:.1} | {:.1} |\n",
            v.name, v.min_us, v.median_us
        ));
    }
    out.push_str(&format!(
        "\nforward_saving / forward: min {:.3}, median {:.3}\n\
         backward_saved / backward: min {:.3}, median {:.3}\n",
        s[1].min_us / s[0].min_us,
        s[1].median_us / s[0].median_us,
        s[3].min_us / s[2].min_us,
        s[3].median_us / s[2].median_us,
    ));
    out
}
