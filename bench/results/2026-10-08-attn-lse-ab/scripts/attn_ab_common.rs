// Shared body of the attention backward A/B (gp-attention-kernels). Each
// tree's example supplies `Api`: the old tree (c74f3ba) has no saved
// statistics and no window; the new one passes the forward's output and lse.
// Every timed call is followed by `Backend::sync`. One process runs every
// shape; a control GEMM, identical in both trees, runs in the same loop so a
// device-wide drift between processes shows up in the control.

use ojas_core::{Backend, Budget, OjasError, Tensor};
use std::time::Instant;

pub trait Api<B: Backend> {
    const TREE: &'static str;
    const WINDOWS: &'static [Option<usize>];
    fn fwd(be: &B, q: &Tensor, k: &Tensor, v: &Tensor, w: Option<usize>)
        -> Result<Vec<Tensor>, OjasError>;
    fn bwd(
        be: &B,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        saved: &[Tensor],
        gy: &Tensor,
        w: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError>;
}

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn dev<B: Backend>(be: &B, host: &Budget, shape: &[usize], seed: u64) -> Tensor {
    let n = shape.iter().product();
    let t = Tensor::from_f32(&values(n, seed), shape, host).unwrap();
    be.upload(&t).unwrap()
}

fn stats(mut xs: Vec<f64>) -> (f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (xs[0], xs[xs.len() / 2])
}

fn timed<F: FnMut()>(be: &impl Backend, mut f: F) -> f64 {
    let t0 = Instant::now();
    f();
    be.sync().unwrap();
    t0.elapsed().as_secs_f64() * 1e3
}

/// `(tag, B, H, Hkv, T, D)`.
const SHAPES: [(&str, usize, usize, usize, usize, usize); 4] = [
    ("mha_b4h8t2048d64", 4, 8, 8, 2048, 64),
    ("mha_b2h8t1024d128", 2, 8, 8, 1024, 128),
    ("mha_b1h8t2048d256", 1, 8, 8, 2048, 256),
    ("qwen35_gqa_b1h8kv2t2048d256", 1, 8, 2, 2048, 256),
];

pub fn run<B: Backend, A: Api<B>>(be: &B, backend: &str) {
    let warm: usize = std::env::var("AB_WARM").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let iters: usize = std::env::var("AB_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let host = Budget::new(8 << 30);
    let x = dev(be, &host, &[2048, 1024], 11);
    let wt = dev(be, &host, &[1024, 1024], 12);
    for (tag, b, h, hkv, t, d) in SHAPES {
        let (qs, ks) = ([b, h, t, d], [b, hkv, t, d]);
        let q = dev(be, &host, &qs, 1);
        let k = dev(be, &host, &ks, 2);
        let v = dev(be, &host, &ks, 3);
        let gy = dev(be, &host, &qs, 4);
        for &w in A::WINDOWS {
            let saved = A::fwd(be, &q, &k, &v, w).unwrap();
            be.sync().unwrap();
            for _ in 0..warm {
                drop(A::fwd(be, &q, &k, &v, w).unwrap());
                drop(A::bwd(be, &q, &k, &v, &saved, &gy, w).unwrap());
                drop(be.linear_forward(&x, &wt).unwrap());
                be.sync().unwrap();
            }
            let (mut f, mut g, mut c) = (Vec::new(), Vec::new(), Vec::new());
            for i in 0..iters {
                // Alternate which op leads, so neither always follows the other.
                let mut one = |which: usize| match which {
                    0 => f.push(timed(be, || drop(A::fwd(be, &q, &k, &v, w).unwrap()))),
                    1 => g.push(timed(be, || {
                        drop(A::bwd(be, &q, &k, &v, &saved, &gy, w).unwrap())
                    })),
                    _ => c.push(timed(be, || drop(be.linear_forward(&x, &wt).unwrap()))),
                };
                let order = if i % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
                for which in order {
                    one(which);
                }
            }
            let ((fmin, fmed), (gmin, gmed), (cmin, cmed)) = (stats(f), stats(g), stats(c));
            let wtag = w.map_or("full".to_string(), |w| format!("w{w}"));
            println!(
                "AB tree={} backend={backend} shape={tag} window={wtag} fwd_min={fmin:.3} fwd_med={fmed:.3} bwd_min={gmin:.3} bwd_med={gmed:.3} ctl_min={cmin:.3} ctl_med={cmed:.3}",
                A::TREE
            );
        }
    }
}
