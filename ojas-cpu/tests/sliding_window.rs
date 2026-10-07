//! Causal attention with a sliding window (query `t` sees keys
//! `t - W < j <= t`), grouped-query heads, and the row log-sum-exp the
//! forward returns, against an f64 oracle: on the per-row kernel (`Exact`,
//! and `Fast` up to 256 positions) and the blocked one (`Fast` above).
//!
//! Also: a window of at least `T` is the whole prefix bit for bit, the
//! backward from the saved output and lse equals the recomputing backward
//! bit for bit, and a window of 0 is refused.

use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

fn backend(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 40), threads)
        .unwrap()
        .with_numerics(numerics)
}

fn tensor(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(values, shape, &Budget::new(u64::MAX)).unwrap()
}

/// Deterministic values in `[-amp, amp)`.
fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

/// `[B, H, Hkv, T, D]` and the window of one case.
#[derive(Clone, Copy, Debug)]
struct Case {
    b: usize,
    h: usize,
    hkv: usize,
    t: usize,
    d: usize,
    window: Option<usize>,
}

/// `[y, lse, gq, gk, gv]` in f64.
type Exact = [Vec<f64>; 5];

/// The f64 forward and backward of a case, with `gy` the output gradient.
fn oracle(c: Case, q: &[f32], k: &[f32], v: &[f32], gy: &[f32]) -> Exact {
    let Case {
        b,
        h,
        hkv,
        t,
        d,
        window,
    } = c;
    let rep = h / hkv;
    let scale = 1.0 / (d as f64).sqrt();
    let at =
        |x: &[f32], plane: usize, row: usize, col: usize| f64::from(x[(plane * t + row) * d + col]);
    let mut y = vec![0.0; b * h * t * d];
    let mut lse = vec![0.0; b * h * t];
    let mut gq = vec![0.0; b * h * t * d];
    let mut gk = vec![0.0; b * hkv * t * d];
    let mut gv = vec![0.0; b * hkv * t * d];
    for plane in 0..b * h {
        let kvp = (plane / h) * hkv + (plane % h) / rep;
        for i in 0..t {
            let first = window.map_or(0, |w| (i + 1).saturating_sub(w));
            let keys = first..i + 1;
            let s: Vec<f64> = keys
                .clone()
                .map(|j| {
                    (0..d)
                        .map(|c| at(q, plane, i, c) * at(k, kvp, j, c))
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s.iter().map(|x| (x - max).exp()).sum();
            lse[plane * t + i] = max + z.ln();
            let p: Vec<f64> = s.iter().map(|x| (x - max).exp() / z).collect();
            let dp: Vec<f64> = keys
                .clone()
                .map(|j| (0..d).map(|c| at(gy, plane, i, c) * at(v, kvp, j, c)).sum())
                .collect();
            let delta: f64 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
            for (n, j) in keys.enumerate() {
                let ds = p[n] * (dp[n] - delta) * scale;
                for c in 0..d {
                    y[(plane * t + i) * d + c] += p[n] * at(v, kvp, j, c);
                    gq[(plane * t + i) * d + c] += ds * at(k, kvp, j, c);
                    gk[(kvp * t + j) * d + c] += ds * at(q, plane, i, c);
                    gv[(kvp * t + j) * d + c] += p[n] * at(gy, plane, i, c);
                }
            }
        }
    }
    [y, lse, gq, gk, gv]
}

/// `max |got - want| / max(1, max |want|)`.
fn err(got: &Tensor, want: &[f64]) -> f64 {
    let got = got.to_f32_vec().unwrap();
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .fold(0.0f64, |m, (&a, &b)| m.max((f64::from(a) - b).abs()))
        / scale
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Every output of `be` on `c` within `tol` of the oracle; and the
/// saved-statistics backward equal to the recomputing one bit for bit.
fn check(be: &CpuBackend, c: Case, seed: u64, tol: f64) {
    let qs = [c.b, c.h, c.t, c.d];
    let ks = [c.b, c.hkv, c.t, c.d];
    let n = |s: &[usize]| s.iter().product::<usize>();
    let (q, k, v, gy) = (
        values(n(&qs), seed, 1.0),
        values(n(&ks), seed + 1, 1.0),
        values(n(&ks), seed + 2, 1.0),
        values(n(&qs), seed + 3, 1.0),
    );
    let want = oracle(c, &q, &k, &v, &gy);
    let (q, k, v, gy) = (
        tensor(&q, &qs),
        tensor(&k, &ks),
        tensor(&v, &ks),
        tensor(&gy, &qs),
    );
    let (y, lse) = be.causal_sdpa_forward(&q, &k, &v, c.window).unwrap();
    assert_eq!(lse.shape(), &[c.b, c.h, c.t], "{c:?}");
    let (gq, gk, gv) = be
        .causal_sdpa_backward(&q, &k, &v, &y, &lse, &gy, c.window)
        .unwrap();
    let got = [&y, &lse, &gq, &gk, &gv];
    for (name, (g, w)) in ["y", "lse", "gq", "gk", "gv"]
        .iter()
        .zip(got.iter().zip(&want))
    {
        let e = err(g, w);
        assert!(
            e <= tol,
            "{:?} {c:?} {name}: {e:e} > {tol:e}",
            be.numerics()
        );
    }
    let (rq, rk, rv) = be
        .causal_sdpa_backward_recompute(&q, &k, &v, &gy, c.window)
        .unwrap();
    for (name, (a, b)) in ["gq", "gk", "gv"]
        .iter()
        .zip([(&gq, &rq), (&gk, &rk), (&gv, &rv)])
    {
        assert_eq!(bits(a), bits(b), "{c:?} {name}: saved != recompute");
    }
}

/// Per-row kernel: windows inside, at and past `T`, grouped and not.
#[test]
fn windowed_attention_matches_the_f64_oracle_on_the_per_row_kernel() {
    for numerics in [Numerics::Exact, Numerics::Fast] {
        for threads in [1usize, 5] {
            let be = backend(threads, numerics);
            let mut seed = 100;
            for (b, h, hkv, t, d) in [(1usize, 2usize, 1usize, 13usize, 8usize), (2, 6, 3, 40, 16)]
            {
                for window in [
                    Some(1),
                    Some(2),
                    Some(5),
                    Some(t - 1),
                    Some(t),
                    Some(t + 7),
                    None,
                ] {
                    seed += 4;
                    check(
                        &be,
                        Case {
                            b,
                            h,
                            hkv,
                            t,
                            d,
                            window,
                        },
                        seed,
                        2e-6,
                    );
                }
            }
        }
    }
}

/// The blocked kernel (`Fast` above 256 positions): windows inside one
/// 256-row block, across blocks, and past `T`.
#[test]
fn windowed_attention_matches_the_f64_oracle_on_the_blocked_kernel() {
    for threads in [1usize, 7] {
        let be = backend(threads, Numerics::Fast);
        let mut seed = 900;
        for window in [
            Some(1),
            Some(17),
            Some(255),
            Some(256),
            Some(300),
            Some(599),
            None,
        ] {
            seed += 4;
            let c = Case {
                b: 1,
                h: 4,
                hkv: 2,
                t: 600,
                d: 32,
                window,
            };
            check(&be, c, seed, 4e-6);
        }
    }
}

/// A window of at least `T` is the whole prefix, bit for bit, in both
/// passes.
#[test]
fn a_window_of_at_least_t_is_the_whole_prefix_bit_for_bit() {
    for (numerics, t) in [(Numerics::Exact, 33usize), (Numerics::Fast, 300)] {
        let be = backend(3, numerics);
        let qs = [1, 4, t, 16];
        let ks = [1, 2, t, 16];
        let q = tensor(&values(4 * t * 16, 1, 1.0), &qs);
        let k = tensor(&values(2 * t * 16, 2, 1.0), &ks);
        let v = tensor(&values(2 * t * 16, 3, 1.0), &ks);
        let gy = tensor(&values(4 * t * 16, 4, 1.0), &qs);
        let (y0, l0) = be.causal_sdpa_forward(&q, &k, &v, None).unwrap();
        let g0 = be
            .causal_sdpa_backward(&q, &k, &v, &y0, &l0, &gy, None)
            .unwrap();
        for w in [t, t + 1, usize::MAX] {
            let (y, l) = be.causal_sdpa_forward(&q, &k, &v, Some(w)).unwrap();
            assert_eq!(bits(&y), bits(&y0), "{numerics:?} window {w} y");
            assert_eq!(bits(&l), bits(&l0), "{numerics:?} window {w} lse");
            let g = be
                .causal_sdpa_backward(&q, &k, &v, &y, &l, &gy, Some(w))
                .unwrap();
            for (a, b) in [(&g.0, &g0.0), (&g.1, &g0.1), (&g.2, &g0.2)] {
                assert_eq!(bits(a), bits(b), "{numerics:?} window {w} grads");
            }
        }
    }
}

#[test]
fn a_window_of_zero_is_refused_in_both_directions() {
    let be = backend(1, Numerics::Exact);
    let q = tensor(&values(2 * 4 * 8, 1, 1.0), &[1, 2, 4, 8]);
    let lse = tensor(&[0.0; 8], &[1, 2, 4]);
    let shape_err = |r: Result<_, OjasError>| matches!(r, Err(OjasError::Shape { .. }));
    assert!(shape_err(
        be.causal_sdpa_forward(&q, &q, &q, Some(0)).map(drop)
    ));
    assert!(shape_err(
        be.causal_sdpa_backward(&q, &q, &q, &q, &lse, &q, Some(0))
            .map(drop)
    ));
}
