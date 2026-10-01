//! Temporary A/B for this round, removed before hand-off: each workload at
//! the paired bench's shapes, with `OJAS_AB_OLD` toggling the old device
//! paths, interleaved in one process.
//!
//! `cargo run -p ojas-metal --release --example ab_round3 [iters] [rounds] [only]`

use std::time::Instant;

use ojas_core::{AdamWConfig, Backend, Budget, OjasError, Tensor};
use ojas_metal::MetalBackend;

type R<T> = Result<T, OjasError>;

const B: usize = 4;
const T: usize = 1024;
const H: usize = 12;
const D: usize = 64;
const DM: usize = 768;
const FF: usize = 2048;
const V: usize = 50304;
const N: usize = B * T;

fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn dev(be: &MetalBackend, shape: &[usize], seed: u64, scale: f32) -> R<Tensor> {
    let host = Budget::new(16 << 30);
    be.upload(&Tensor::from_f32(
        &values(shape.iter().product(), seed, scale),
        shape,
        &host,
    )?)
}

fn param_shapes() -> Vec<Vec<usize>> {
    let mut v = vec![vec![V, DM]];
    for _ in 0..12 {
        v.push(vec![DM]);
        v.push(vec![1]);
        for _ in 0..4 {
            v.push(vec![DM, DM]);
        }
        v.push(vec![D]);
        v.push(vec![D]);
        v.push(vec![H, DM]);
        v.push(vec![H]);
        v.push(vec![DM]);
        v.push(vec![FF, DM]);
        v.push(vec![FF, DM]);
        v.push(vec![DM, FF]);
    }
    v.push(vec![DM]);
    v
}

fn set_old(on: bool) {
    // SAFETY: no other thread reads the environment while no op is in
    // flight; every call returns before the next set.
    unsafe {
        if on {
            std::env::set_var("OJAS_AB_OLD", "1");
        } else {
            std::env::remove_var("OJAS_AB_OLD");
        }
    }
}

struct Work {
    name: &'static str,
    run: Box<dyn FnMut() -> R<()>>,
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().collect();
    let iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let only = args.get(3).cloned().unwrap_or_default();
    let be = MetalBackend::new(Budget::new(40 << 30))?;
    let mut works: Vec<Work> = Vec::new();
    if only.is_empty() || only == "adamw" {
        let shapes = param_shapes();
        let mut p = Vec::new();
        let mut g = Vec::new();
        let mut m = Vec::new();
        let mut v = Vec::new();
        for (i, s) in shapes.iter().enumerate() {
            p.push(dev(&be, s, 2000 + i as u64, 1.0 / 32.0)?);
            g.push(dev(&be, s, 3000 + i as u64, 1.0 / 64.0)?);
            m.push(dev(&be, s, 4000 + i as u64, 0.0)?);
            v.push(dev(&be, s, 5000 + i as u64, 0.0)?);
        }
        let b2 = be.clone();
        let mut step = 0u64;
        works.push(Work {
            name: "adamw_full (170 tensors)",
            run: Box::new(move || {
                let cfg = AdamWConfig::nanolab(1e-3, 0.1);
                for i in 0..p.len() {
                    b2.adamw_step(&mut p[i], &g[i], &mut m[i], &mut v[i], step, cfg)?;
                }
                step += 1;
                Ok(())
            }),
        });
    }
    if only == "adamw_sizes" {
        for n in [1usize, 768, 768 * 768, 2048 * 768, 50304 * 768] {
            let mut p = dev(&be, &[n], 1, 0.03)?;
            let g = dev(&be, &[n], 2, 0.01)?;
            let mut m = dev(&be, &[n], 3, 0.0)?;
            let mut v = dev(&be, &[n], 4, 0.0)?;
            let b2 = be.clone();
            let mut step = 0u64;
            let name: &'static str = Box::leak(format!("adamw one tensor n={n}").into_boxed_str());
            works.push(Work {
                name,
                run: Box::new(move || {
                    let cfg = AdamWConfig::nanolab(1e-3, 0.1);
                    b2.adamw_step(&mut p, &g, &mut m, &mut v, step, cfg)?;
                    step += 1;
                    Ok(())
                }),
            });
        }
    }
    if only.is_empty() || only == "rms" {
        let x = dev(&be, &[N, DM], 31, 1.0)?;
        let w = dev(&be, &[DM], 32, 1.0)?;
        let gy = dev(&be, &[N, DM], 33, 1.0)?;
        let (b2, x2, w2) = (be.clone(), x.clone(), w.clone());
        works.push(Work {
            name: "rms_norm_fwd [4096,768]",
            run: Box::new(move || b2.rms_norm_forward(&x2, &w2, 1e-6).map(drop)),
        });
        let b3 = be.clone();
        works.push(Work {
            name: "rms_norm_bwd [4096,768]",
            run: Box::new(move || b3.rms_norm_backward(&x, &w, &gy, 1e-6).map(drop)),
        });
    }
    if only.is_empty() || only == "qk" {
        let s = [B, T, H, D];
        let q = dev(&be, &s, 41, 1.0)?;
        let k = dev(&be, &s, 42, 1.0)?;
        let qw = dev(&be, &[D], 43, 1.0)?;
        let kw = dev(&be, &[D], 44, 1.0)?;
        let gq = dev(&be, &s, 45, 1.0)?;
        let gk = dev(&be, &s, 46, 1.0)?;
        let (b2, q2, k2, qw2, kw2) = (be.clone(), q.clone(), k.clone(), qw.clone(), kw.clone());
        works.push(Work {
            name: "rms_qk_norm_fwd [4,1024,12,64]",
            run: Box::new(move || b2.rms_qk_norm_forward(&q2, &k2, &qw2, &kw2, 1e-6).map(drop)),
        });
        let b3 = be.clone();
        works.push(Work {
            name: "rms_qk_norm_bwd [4,1024,12,64]",
            run: Box::new(move || {
                b3.rms_qk_norm_backward(&q, &k, &qw, &kw, &gq, &gk, 1e-6)
                    .map(drop)
            }),
        });
    }
    if only.is_empty() || only == "ce" {
        let l = dev(&be, &[N, V], 121, 4.0)?;
        let tv: Vec<u32> = (0..N).map(|i| ((i as u64 * 2654435761) % V as u64) as u32).collect();
        let t = be.upload(&Tensor::from_u32(&tv, &[N], &Budget::new(1 << 30))?)?;
        let (b2, l2, t2) = (be.clone(), l.clone(), t.clone());
        works.push(Work {
            name: "cross_entropy_fwd [4096,50304]",
            run: Box::new(move || b2.cross_entropy_mean_forward(&l2, &t2, None).map(drop)),
        });
        let b3 = be.clone();
        works.push(Work {
            name: "cross_entropy_bwd [4096,50304]",
            run: Box::new(move || b3.cross_entropy_mean_backward(&l, &t, None).map(drop)),
        });
    }
    if only.is_empty() || only == "floor" {
        let x = dev(&be, &[1], 1, 1.0)?;
        let b2 = be.clone();
        works.push(Work {
            name: "silu 1 element (floor)",
            run: Box::new(move || b2.silu_forward(&x).map(drop)),
        });
    }
    let mut samples: Vec<[Vec<f64>; 2]> = works.iter().map(|_| Default::default()).collect();
    for _ in 0..rounds {
        for (wi, w) in works.iter_mut().enumerate() {
            for variant in [0usize, 1] {
                set_old(variant == 1);
                for _ in 0..2 {
                    (w.run)()?;
                }
                for _ in 0..iters {
                    let t0 = Instant::now();
                    (w.run)()?;
                    samples[wi][variant].push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
    }
    set_old(false);
    println!("| op | new min | new median | old min | old median | n |");
    println!("|---|---:|---:|---:|---:|---:|");
    for (w, s) in works.iter().zip(samples.iter_mut()) {
        for v in s.iter_mut() {
            v.sort_by(f64::total_cmp);
        }
        println!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {} |",
            w.name,
            s[0][0],
            s[0][s[0].len() / 2],
            s[1][0],
            s[1][s[1].len() / 2],
            s[0].len()
        );
    }
    Ok(())
}
