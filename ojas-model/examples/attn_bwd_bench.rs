//! Causal attention backward latency and attention scratch, on Metal and
//! wgpu.
//!
//! `cargo run -p ojas-model --release --example attn_bwd_bench [iters] [metal|wgpu]`
//!
//! Per shape `[B, H, Hkv, T, D]` (multi-head at the bench shapes, and the
//! Qwen3.5-2B attention: 8 query heads over 2 KV heads of 256): the bytes a
//! forward and a backward charge beyond their outputs (the budget's peak
//! during the op minus what stays live after it), and the min and median of
//! `iters` timed backward runs (each the op and a `sync`) after two
//! warm-ups. The backward is timed from the forward's saved output and
//! row log-sum-exp, as a tape runs it.
//!
//! Every trait call sits in [`api`], so a build of an older trait runs the
//! same rows with only that module changed (the A/B in
//! `docs/pytorch-parity-plan.md`).

use std::time::Instant;

use ojas_core::{Backend, Budget, OjasError, Tensor};

type R<T> = Result<T, OjasError>;

mod api {
    use super::R;
    use ojas_core::{Backend, Tensor};

    /// What the forward keeps for the backward: the output and its row
    /// log-sum-exp.
    pub struct Saved {
        pub y: Tensor,
        pub lse: Tensor,
    }

    pub fn forward<B: Backend>(be: &B, q: &Tensor, k: &Tensor, v: &Tensor) -> R<Saved> {
        let (y, lse) = be.causal_sdpa_forward(q, k, v, None)?;
        Ok(Saved { y, lse })
    }

    pub fn backward<B: Backend>(
        be: &B,
        [q, k, v, gy]: [&Tensor; 4],
        s: &Saved,
    ) -> R<(Tensor, Tensor, Tensor)> {
        be.causal_sdpa_backward(q, k, v, &s.y, &s.lse, gy, None)
    }
}

/// `[B, H, Hkv, T, D]`.
const SHAPES: [[usize; 5]; 3] = [
    [4, 8, 8, 1024, 64],
    [2, 16, 16, 1024, 128],
    [2, 8, 2, 2048, 256],
];

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// Bytes `op` charged beyond what stays live after it.
fn scratch<B: Backend, T>(be: &B, op: impl FnOnce() -> R<T>) -> R<(T, u64)> {
    let budget = be.budget();
    budget.reset_peak();
    let out = op()?;
    be.sync()?;
    let peak = budget.peak_bytes();
    let live = budget.live_bytes()?;
    Ok((out, peak.saturating_sub(live)))
}

fn run<B: Backend>(name: &str, be: &B, iters: usize) -> R<()> {
    let host = Budget::new(u64::MAX);
    for [b, h, hkv, t, d] in SHAPES {
        let (qs, ks) = ([b, h, t, d], [b, hkv, t, d]);
        let n = |s: &[usize]| s.iter().product::<usize>();
        let up = |s: &[usize], seed| be.upload(&Tensor::from_f32(&values(n(s), seed), s, &host)?);
        let (q, k, v, gy) = (up(&qs, 1)?, up(&ks, 2)?, up(&ks, 3)?, up(&qs, 4)?);
        let (saved, fwd_scratch) = scratch(be, || api::forward(be, &q, &k, &v))?;
        let ops = [&q, &k, &v, &gy];
        let (grads, bwd_scratch) = scratch(be, || api::backward(be, ops, &saved))?;
        drop(grads);
        for _ in 0..2 {
            drop(api::backward(be, ops, &saved)?);
            be.sync()?;
        }
        let mut ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let g = api::backward(be, ops, &saved)?;
            be.sync()?;
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
            drop(g);
        }
        ms.sort_by(f64::total_cmp);
        println!(
            "{name} B{b} H{h} Hkv{hkv} T{t} D{d}: fwd scratch {fwd_scratch} B, bwd scratch {bwd_scratch} B, \
             bwd min {:.3} ms, median {:.3} ms",
            ms[0],
            ms[ms.len() / 2]
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iters = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(20usize)
        .max(1);
    let only = std::env::args().nth(2).unwrap_or_default();
    if only.is_empty() || only == "wgpu" {
        let gpu = ojas_wgpu::WgpuBackend::open(Budget::new(8 << 30))?;
        run("wgpu", &gpu, iters)?;
    }
    #[cfg(target_os = "macos")]
    if only.is_empty() || only == "metal" {
        let gpu = ojas_metal::MetalBackend::new(Budget::new(8 << 30))?;
        run("metal", &gpu, iters)?;
    }
    Ok(())
}
