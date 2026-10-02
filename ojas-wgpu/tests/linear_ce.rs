//! `Backend::linear_cross_entropy_mean` on wgpu (T3, gate G3).
//!
//! (a) equals the unfused `linear_forward -> cross_entropy_mean_forward /
//! backward -> linear_backward` composition on wgpu at 1e-6, measured as
//! max |fused - unfused| / max |unfused| per output; (c) runs under a budget
//! smaller than the `N x V` logits the unfused path needs, which that path
//! refuses; (d) matches the CPU reference (`Numerics::Exact`) at 1e-4. Chunks
//! that do not divide N or V, a 1 x 1 chunk, and a chunk larger than the
//! problem are covered, with and without ignored rows.

mod common;

use common::*;
use ojas_core::{Backend, Budget, CeChunk, OjasError, Tensor};
use ojas_wgpu::WgpuBackend;

struct Problem {
    x: Tensor,
    w: Tensor,
    t: Tensor,
    ignore: Option<u32>,
}

fn problem(n: usize, d: usize, v: usize, seed: u64, ignore_every: Option<usize>) -> Problem {
    let ignore = ignore_every.map(|_| v as u32 + 7);
    let ids: Vec<u32> = (0..n)
        .map(|i| match ignore_every {
            Some(k) if i % k == 1 => v as u32 + 7,
            _ => ((i * 2_654_435_761usize) % v) as u32,
        })
        .collect();
    // Scaled so logits spread over a few units, as a trained head's do.
    let scale = |seed, shape: &[usize], s: f32| {
        let k: usize = shape.iter().product();
        let vals: Vec<f32> = data(seed, k).iter().map(|x| x * s).collect();
        Tensor::from_f32(&vals, shape, host_budget()).unwrap()
    };
    Problem {
        x: scale(seed, &[n, d], 1.0),
        w: scale(seed + 1, &[v, d], 0.5),
        t: host_u32(&ids, &[n]),
        ignore,
    }
}

fn rel(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, v| m.max(f64::from(v.abs())));
    let worst = got.iter().zip(want).fold(0.0f64, |m, (a, b)| {
        assert!(a.is_finite(), "non-finite output {a}");
        m.max((f64::from(*a) - f64::from(*b)).abs())
    });
    if scale == 0.0 {
        worst
    } else {
        worst / scale
    }
}

fn vals(g: &WgpuBackend, t: &Tensor) -> Vec<f32> {
    g.download(t).unwrap().to_f32_vec().unwrap()
}

/// `[loss, grad_input, grad_weight]` from the unfused wgpu composition.
fn unfused(g: &WgpuBackend, p: &Problem) -> [Vec<f32>; 3] {
    let (x, w, t) = (
        g.upload(&p.x).unwrap(),
        g.upload(&p.w).unwrap(),
        g.upload(&p.t).unwrap(),
    );
    let y = g.linear_forward(&x, &w).unwrap();
    let loss = g.cross_entropy_mean_forward(&y, &t, p.ignore).unwrap();
    let gy = g.cross_entropy_mean_backward(&y, &t, p.ignore).unwrap();
    let (gx, gw) = g.linear_backward(&x, &w, &gy).unwrap();
    g.sync().unwrap();
    [vals(g, &loss), vals(g, &gx), vals(g, &gw)]
}

fn fused(g: &WgpuBackend, p: &Problem, chunk: CeChunk, want_grad: bool) -> Vec<Vec<f32>> {
    let (x, w, t) = (
        g.upload(&p.x).unwrap(),
        g.upload(&p.w).unwrap(),
        g.upload(&p.t).unwrap(),
    );
    let r = g
        .linear_cross_entropy_mean(&x, &w, &t, p.ignore, chunk, want_grad)
        .unwrap();
    assert_eq!(r.loss.shape(), &[] as &[usize]);
    assert_eq!(r.grad_input.is_some(), want_grad);
    assert_eq!(r.grad_weight.is_some(), want_grad);
    g.sync().unwrap();
    let mut out = vec![vals(g, &r.loss)];
    if let (Some(gx), Some(gw)) = (&r.grad_input, &r.grad_weight) {
        assert_eq!(gx.shape(), p.x.shape());
        assert_eq!(gw.shape(), p.w.shape());
        out.push(vals(g, gx));
        out.push(vals(g, gw));
    }
    out
}

fn equals_composition(tag: &str, p: &Problem, chunk: CeChunk) -> f64 {
    let g = fresh();
    let want = unfused(&g, p);
    let got = fused(&g, p, chunk, true);
    let mut worst = 0.0f64;
    for (name, (a, b)) in ["loss", "grad_input", "grad_weight"]
        .iter()
        .zip(got.iter().zip(&want))
    {
        let e = rel(a, b);
        assert!(e <= 1e-6, "{tag} {chunk:?} {name}: rel {e:.3e} > 1e-6");
        worst = worst.max(e);
    }
    let loss_only = fused(&g, p, chunk, false);
    assert_eq!(loss_only.len(), 1);
    assert_eq!(
        loss_only[0][0].to_bits(),
        got[0][0].to_bits(),
        "{tag}: loss differs without grads"
    );
    worst
}

#[test]
fn equals_the_unfused_composition_for_every_chunking() {
    let (n, d, v) = (300usize, 48usize, 1000usize);
    for (i, ignore_every) in [None, Some(3)].into_iter().enumerate() {
        let p = problem(n, d, v, 10 + i as u64, ignore_every);
        for chunk in [
            CeChunk {
                rows: 64,
                cols: 128,
            },
            CeChunk { rows: 7, cols: 333 },
            CeChunk {
                rows: 300,
                cols: 999,
            },
            CeChunk {
                rows: 299,
                cols: 1000,
            },
            CeChunk {
                rows: 128,
                cols: 1000,
            },
            CeChunk {
                rows: 10_000,
                cols: 1_000_000,
            },
        ] {
            let e = equals_composition(&format!("ignore {ignore_every:?}"), &p, chunk);
            println!("ignore {ignore_every:?} {chunk:?}: worst rel {e:.3e}");
        }
    }
}

#[test]
fn a_one_by_one_chunk_equals_the_composition() {
    let p = problem(5, 8, 11, 30, Some(2));
    let e = equals_composition("1x1", &p, CeChunk { rows: 1, cols: 1 });
    println!("1x1: worst rel {e:.3e}");
}

/// One tile covering the whole problem runs the same GEMM and the same row
/// max/sum reduction as the composition, so the loss is bit-identical. The
/// gradients use the same expression in a different kernel, and the Metal
/// compiler (fast math) may round its division differently there, so they
/// are held to the 1e-6 gate, not to bits.
#[test]
fn one_tile_covering_everything_gives_the_composition_loss_bit_for_bit() {
    let g = fresh();
    let p = problem(200, 64, 700, 40, Some(5));
    let want = unfused(&g, &p);
    let got = fused(
        &g,
        &p,
        CeChunk {
            rows: 200,
            cols: 700,
        },
        true,
    );
    assert_eq!(got[0][0].to_bits(), want[0][0].to_bits(), "loss");
    for (name, (a, b)) in ["grad_input", "grad_weight"]
        .iter()
        .zip(got[1..].iter().zip(&want[1..]))
    {
        let e = rel(a, b);
        assert!(e <= 1e-6, "{name}: rel {e:.3e}");
    }
}

#[test]
fn matches_the_cpu_reference() {
    let g = fresh();
    let c = cpu();
    let p = problem(257, 96, 3001, 50, Some(4));
    let y = c.linear_forward(&p.x, &p.w).unwrap();
    let loss = c.cross_entropy_mean_forward(&y, &p.t, p.ignore).unwrap();
    let gy = c.cross_entropy_mean_backward(&y, &p.t, p.ignore).unwrap();
    let (gx, gw) = c.linear_backward(&p.x, &p.w, &gy).unwrap();
    let got = fused(
        &g,
        &p,
        CeChunk {
            rows: 100,
            cols: 1024,
        },
        true,
    );
    for (name, (a, b)) in ["loss", "grad_input", "grad_weight"].iter().zip(
        got.iter()
            .zip([loss, gx, gw].iter().map(|t| t.to_f32_vec().unwrap())),
    ) {
        let e = rel(a, &b);
        assert!(e <= 1e-4, "{name}: rel {e:.3e}");
    }
}

/// The CPU's own fused op (`Numerics::Exact`), with a different chunking on
/// each side, at 1e-4.
#[test]
fn matches_the_cpu_fused_op() {
    let g = fresh();
    let c = cpu();
    for (i, ignore_every) in [None, Some(3)].into_iter().enumerate() {
        let p = problem(300, 64, 2001, 120 + i as u64, ignore_every);
        let want = c
            .linear_cross_entropy_mean(
                &p.x,
                &p.w,
                &p.t,
                p.ignore,
                CeChunk {
                    rows: 128,
                    cols: 512,
                },
                true,
            )
            .unwrap();
        let got = fused(
            &g,
            &p,
            CeChunk {
                rows: 77,
                cols: 300,
            },
            true,
        );
        let want = [
            want.loss.to_f32_vec().unwrap(),
            want.grad_input.unwrap().to_f32_vec().unwrap(),
            want.grad_weight.unwrap().to_f32_vec().unwrap(),
        ];
        for (name, (a, b)) in ["loss", "grad_input", "grad_weight"]
            .iter()
            .zip(got.iter().zip(&want))
        {
            let e = rel(a, b);
            assert!(e <= 1e-4, "{ignore_every:?} {name}: rel {e:.3e}");
        }
    }
}

#[test]
fn runs_under_a_budget_too_small_for_the_logits() {
    let (n, d, v) = (512usize, 64usize, 4096usize);
    let logits_bytes = (n * v * 4) as u64; // 8 MiB
    let budget = 4u64 << 20;
    assert!(budget < logits_bytes);
    let g = WgpuBackend::with_context(gpu().context().clone(), Budget::new(budget));
    let p = problem(n, d, v, 60, Some(9));
    let (x, w, t) = (
        g.upload(&p.x).unwrap(),
        g.upload(&p.w).unwrap(),
        g.upload(&p.t).unwrap(),
    );
    match g.linear_forward(&x, &w) {
        Err(OjasError::CapacityExceeded { .. }) => {}
        other => panic!("the unfused path should be refused: {other:?}"),
    }
    let r = g
        .linear_cross_entropy_mean(
            &x,
            &w,
            &t,
            p.ignore,
            CeChunk {
                rows: 128,
                cols: 512,
            },
            true,
        )
        .unwrap();
    g.sync().unwrap();
    let loss = vals(&g, &r.loss)[0];
    assert!(loss.is_finite() && loss > 0.0, "{loss}");
    // A tile the budget cannot hold is refused before anything runs.
    match g.linear_cross_entropy_mean(
        &x,
        &w,
        &t,
        p.ignore,
        CeChunk {
            rows: 512,
            cols: 4096,
        },
        true,
    ) {
        Err(OjasError::CapacityExceeded { .. }) => {}
        other => panic!("an N x V tile should not fit: {other:?}"),
    }
    g.sync().unwrap();
}

#[test]
fn results_repeat_bit_for_bit() {
    let g = fresh();
    let p = problem(130, 40, 777, 70, Some(6));
    let chunk = CeChunk {
        rows: 33,
        cols: 100,
    };
    let first = fused(&g, &p, chunk, true);
    for _ in 0..2 {
        let again = fused(&g, &p, chunk, true);
        for (a, b) in again.iter().zip(&first) {
            let a: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
            let b: Vec<u32> = b.iter().map(|x| x.to_bits()).collect();
            assert!(a == b);
        }
    }
}

#[test]
fn refusals_are_synchronous() {
    let g = fresh();
    let p = problem(6, 4, 9, 80, None);
    let (x, w, t) = (
        g.upload(&p.x).unwrap(),
        g.upload(&p.w).unwrap(),
        g.upload(&p.t).unwrap(),
    );
    let chunk = CeChunk { rows: 2, cols: 3 };
    let err = |r: Result<ojas_core::LinearCe, OjasError>| r.map(|_| ()).unwrap_err();
    // Every target ignored: the mean has no rows.
    let all = g.upload(&host_u32(&[3; 6], &[6])).unwrap();
    assert!(matches!(
        err(g.linear_cross_entropy_mean(&x, &w, &all, Some(3), chunk, true)),
        OjasError::NonFinite {
            op: "linear_cross_entropy_mean"
        }
    ));
    let far = g.upload(&host_u32(&[0, 1, 2, 9, 4, 5], &[6])).unwrap();
    assert!(matches!(
        err(g.linear_cross_entropy_mean(&x, &w, &far, None, chunk, true)),
        OjasError::OutOfRange { .. }
    ));
    let w_bad = g.upload(&host(1, &[9, 5])).unwrap();
    assert!(matches!(
        err(g.linear_cross_entropy_mean(&x, &w_bad, &t, None, chunk, true)),
        OjasError::Shape { .. }
    ));
    assert!(matches!(
        err(g.linear_cross_entropy_mean(&x, &w, &t, None, CeChunk { rows: 0, cols: 3 }, true)),
        OjasError::Shape { .. }
    ));
    assert!(matches!(
        err(g.linear_cross_entropy_mean(&p.x, &w, &t, None, chunk, true)),
        OjasError::Placement { .. }
    ));
    assert!(matches!(
        err(g.linear_cross_entropy_mean(&x, &w, &x, None, chunk, true)),
        OjasError::Dtype { .. }
    ));
    g.sync().unwrap();
}

#[test]
fn a_non_finite_input_surfaces_at_sync() {
    let g = fresh();
    let p = problem(20, 8, 50, 90, None);
    let mut xs = p.x.to_f32_vec().unwrap();
    for (i, poison) in [f32::NAN, f32::INFINITY].into_iter().enumerate() {
        xs[13 + i] = poison;
        let x = g
            .upload(&Tensor::from_f32(&xs, &[20, 8], host_budget()).unwrap())
            .unwrap();
        let (w, t) = (g.upload(&p.w).unwrap(), g.upload(&p.t).unwrap());
        let r = g.linear_cross_entropy_mean(&x, &w, &t, None, CeChunk { rows: 7, cols: 16 }, true);
        assert!(r.is_ok());
        match g.sync() {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, "linear_cross_entropy_mean"),
            other => panic!("{poison}: {other:?}"),
        }
    }
}

/// G3(d) at the bench shape: N = 4096, d = 768, V = 50304, against the CPU
/// unfused composition at `Numerics::Exact`. Heavy (the CPU side holds two
/// 824 MB logit tensors), so it runs on request.
#[test]
#[ignore = "heavy: G3(d) at N=4096, V=50304; run with --ignored"]
fn matches_the_cpu_at_the_bench_shape() {
    let g = WgpuBackend::open(Budget::new(8 << 30)).unwrap();
    let c = cpu();
    let p = problem(4096, 768, 50304, 100, Some(17));
    let t0 = std::time::Instant::now();
    let y = c.linear_forward(&p.x, &p.w).unwrap();
    let loss = c.cross_entropy_mean_forward(&y, &p.t, p.ignore).unwrap();
    let gy = c.cross_entropy_mean_backward(&y, &p.t, p.ignore).unwrap();
    drop(y);
    let (gx, gw) = c.linear_backward(&p.x, &p.w, &gy).unwrap();
    drop(gy);
    println!("cpu exact composition: {:.1} s", t0.elapsed().as_secs_f64());
    let got = fused(
        &g,
        &p,
        CeChunk {
            rows: 1024,
            cols: 8192,
        },
        true,
    );
    for (name, (a, b)) in ["loss", "grad_input", "grad_weight"].iter().zip(
        got.iter()
            .zip([loss, gx, gw].iter().map(|t| t.to_f32_vec().unwrap())),
    ) {
        let e = rel(a, &b);
        println!("{name}: rel {e:.3e}");
        assert!(e <= 1e-4, "{name}: rel {e:.3e}");
    }
}

/// Time at the bench shape (N = 4096, d = 768, V = 50304), interleaved, min
/// of N: the unfused composition (`linear_forward`, CE forward and
/// backward, `linear_backward`, with the full `N x V` logits and gradient)
/// against the fused op at several chunkings, with gradients.
#[test]
#[ignore = "benchmark; run with --ignored --nocapture in release"]
fn bench_fused_against_unfused() {
    let g = WgpuBackend::open(Budget::new(16 << 30)).unwrap();
    let p = problem(4096, 768, 50304, 110, Some(17));
    let (x, w, t) = (
        g.upload(&p.x).unwrap(),
        g.upload(&p.w).unwrap(),
        g.upload(&p.t).unwrap(),
    );
    let unfused_step = || {
        let s = std::time::Instant::now();
        let y = g.linear_forward(&x, &w).unwrap();
        let loss = g.cross_entropy_mean_forward(&y, &t, p.ignore).unwrap();
        let gy = g.cross_entropy_mean_backward(&y, &t, p.ignore).unwrap();
        let grads = g.linear_backward(&x, &w, &gy).unwrap();
        g.sync().unwrap();
        drop((y, loss, gy, grads));
        s.elapsed().as_secs_f64()
    };
    let fused_step = |chunk: CeChunk, want_grad: bool| {
        let s = std::time::Instant::now();
        let r = g
            .linear_cross_entropy_mean(&x, &w, &t, p.ignore, chunk, want_grad)
            .unwrap();
        g.sync().unwrap();
        drop(r);
        s.elapsed().as_secs_f64()
    };
    let chunks = [
        CeChunk {
            rows: 4096,
            cols: 50304,
        },
        CeChunk {
            rows: 1024,
            cols: 50304,
        },
        CeChunk {
            rows: 1024,
            cols: 8192,
        },
        CeChunk {
            rows: 4096,
            cols: 8192,
        },
        CeChunk {
            rows: 512,
            cols: 4096,
        },
    ];
    let arms = chunks.len() + 2;
    let run = |arm: usize| match arm {
        0 => unfused_step(),
        1 => fused_step(
            CeChunk {
                rows: 1024,
                cols: 8192,
            },
            false,
        ),
        a => fused_step(chunks[a - 2], true),
    };
    for a in 0..arms {
        run(a);
    }
    let mut best = vec![f64::INFINITY; arms];
    for round in 0..5 {
        for j in 0..arms {
            let a = if round % 2 == 0 { j } else { arms - 1 - j };
            best[a] = best[a].min(run(a));
        }
    }
    println!("| path | logit scratch | ms (min of 5) |");
    println!("| --- | ---: | ---: |");
    println!(
        "| unfused, with grads | {} MB x2 | {:.1} |",
        4096 * 50304 * 4 / 1_000_000,
        best[0] * 1e3
    );
    println!(
        "| fused 1024x8192, loss only | 33 MB | {:.1} |",
        best[1] * 1e3
    );
    for (i, c) in chunks.iter().enumerate() {
        println!(
            "| fused {}x{}, with grads | {} MB | {:.1} |",
            c.rows,
            c.cols,
            c.rows * c.cols * 4 / 1_000_000,
            best[i + 2] * 1e3
        );
    }
}
