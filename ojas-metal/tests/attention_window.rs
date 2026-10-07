//! Sliding-window and grouped-query causal attention, and the row
//! log-sum-exp the forward returns, against the CPU reference; plus the
//! bit-level contracts of the native grouped-query kernels and of the
//! saved-statistics backward.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Backend, Tensor};
use ojas_metal::MetalBackend;

fn tol(t: usize) -> f32 {
    2e-5 * (t as f32).sqrt().max(1.0)
}

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

/// `[q, k, v, gy]` host tensors of a `[B, H, T, D]` query over `Hkv` KV heads.
fn inputs(b: usize, h: usize, hkv: usize, t: usize, d: usize, seed: u64) -> [Tensor; 4] {
    let (qs, ks) = ([b, h, t, d], [b, hkv, t, d]);
    [
        rand(&qs, seed, 1.0),
        rand(&ks, seed + 1, 1.0),
        rand(&ks, seed + 2, 1.0),
        rand(&qs, seed + 3, 1.0),
    ]
}

/// Forward (output and lse) and the saved-statistics backward on Metal
/// against the CPU at one case.
fn parity(
    m: &MetalBackend,
    case: (usize, usize, usize, usize, usize),
    window: Option<usize>,
    seed: u64,
) {
    let (b, h, hkv, t, d) = case;
    let c = cpu();
    let [q, k, v, gy] = inputs(b, h, hkv, t, d, seed);
    let tag = format!("b{b} h{h}/{hkv} t{t} d{d} window {window:?}");
    let (wy, wl) = ok(&tag, c.causal_sdpa_forward(&q, &k, &v, window));
    let (wq, wk, wv) = ok(
        &tag,
        c.causal_sdpa_backward(&q, &k, &v, &wy, &wl, &gy, window),
    );
    let dd = [up(m, &q), up(m, &k), up(m, &v), up(m, &gy)];
    let (y, l) = ok(&tag, m.causal_sdpa_forward(&dd[0], &dd[1], &dd[2], window));
    same_tensor(&format!("{tag} y"), &y, &wy, tol(t), 1e-4);
    same_tensor(&format!("{tag} lse"), &l, &wl, tol(t), 1e-5);
    let (gq, gk, gv) = ok(
        &tag,
        m.causal_sdpa_backward(&dd[0], &dd[1], &dd[2], &y, &l, &dd[3], window),
    );
    same_tensor(&format!("{tag} dq"), &gq, &wq, tol(t), 1e-3);
    same_tensor(
        &format!("{tag} dk"),
        &gk,
        &wk,
        tol(t) * (h / hkv) as f32,
        1e-3,
    );
    same_tensor(
        &format!("{tag} dv"),
        &gv,
        &wv,
        tol(t) * (h / hkv) as f32,
        1e-3,
    );
}

/// Windows inside a 32-row tile, across tiles, one short of `T`, and past
/// it; multi-head and grouped; head dims at each compiled width.
#[test]
fn windowed_attention_matches_cpu() {
    let m = metal();
    let mut seed = 7000;
    for case in [
        (1usize, 2usize, 2usize, 67usize, 64usize),
        (2, 4, 2, 100, 128),
        (1, 8, 2, 129, 256),
        (1, 6, 3, 45, 13),
    ] {
        let t = case.3;
        for window in [
            Some(1),
            Some(5),
            Some(32),
            Some(33),
            Some(64),
            Some(t - 1),
            Some(t + 3),
            None,
        ] {
            seed += 4;
            parity(&m, case, window, seed);
        }
    }
}

/// Grouped-query heads read their KV head in place: the forward's output
/// and lse, and dQ, equal the multi-head kernel's on K and V repeated to
/// the query heads, bit for bit; dK and dV equal the repeated gradients
/// summed back within rounding.
#[test]
fn native_grouped_query_equals_repeated_heads() {
    let m = metal();
    for (b, h, hkv, t, d, window) in [
        (2usize, 8usize, 2usize, 70usize, 64usize, None),
        (1, 4, 1, 40, 256, Some(9)),
    ] {
        let rep = h / hkv;
        let [q, k, v, gy] = inputs(b, h, hkv, t, d, 7300);
        let repeat = |x: &Tensor| {
            let x = x.to_f32_vec().unwrap();
            let plane = t * d;
            let mut out = Vec::with_capacity(b * h * plane);
            for bh in 0..b * h {
                let src = (bh / h) * hkv + (bh % h) / rep;
                out.extend_from_slice(&x[src * plane..(src + 1) * plane]);
            }
            host(&out, &[b, h, t, d])
        };
        let (kr, vr) = (repeat(&k), repeat(&v));
        let dd = [up(&m, &q), up(&m, &k), up(&m, &v), up(&m, &gy)];
        let (dkr, dvr) = (up(&m, &kr), up(&m, &vr));
        let tag = format!("h{h}/{hkv} t{t} d{d} window {window:?}");
        let (y, l) = ok(&tag, m.causal_sdpa_forward(&dd[0], &dd[1], &dd[2], window));
        let (yr, lr) = ok(&tag, m.causal_sdpa_forward(&dd[0], &dkr, &dvr, window));
        assert_eq!(bits(&y), bits(&yr), "{tag}: output");
        assert_eq!(bits(&l), bits(&lr), "{tag}: lse");
        let (gq, gk, gv) = ok(
            &tag,
            m.causal_sdpa_backward(&dd[0], &dd[1], &dd[2], &y, &l, &dd[3], window),
        );
        let (gqr, gkr, gvr) = ok(
            &tag,
            m.causal_sdpa_backward(&dd[0], &dkr, &dvr, &yr, &lr, &dd[3], window),
        );
        assert_eq!(bits(&gq), bits(&gqr), "{tag}: dq");
        let sum_back = |g: &Tensor| {
            let g = down(g);
            let plane = t * d;
            let mut out = vec![0.0f32; b * hkv * plane];
            for bh in 0..b * h {
                let dst = (bh / h) * hkv + (bh % h) / rep;
                for (o, x) in out[dst * plane..(dst + 1) * plane]
                    .iter_mut()
                    .zip(&g[bh * plane..(bh + 1) * plane])
                {
                    *o += x;
                }
            }
            out
        };
        close(
            &format!("{tag} dk"),
            &down(&gk),
            &sum_back(&gkr),
            tol(t),
            1e-4,
        );
        close(
            &format!("{tag} dv"),
            &down(&gv),
            &sum_back(&gvr),
            tol(t),
            1e-4,
        );
    }
}

/// The backward from the saved output and lse equals the recomputing
/// backward bit for bit, and both repeat bit for bit.
#[test]
fn saved_backward_equals_the_recomputing_one_bit_for_bit() {
    let m = metal();
    let [q, k, v, gy] = inputs(1, 4, 2, 150, 128, 7400);
    let dd = [up(&m, &q), up(&m, &k), up(&m, &v), up(&m, &gy)];
    for window in [None, Some(40)] {
        let (y, l) = ok("fwd", m.causal_sdpa_forward(&dd[0], &dd[1], &dd[2], window));
        let saved = ok(
            "saved",
            m.causal_sdpa_backward(&dd[0], &dd[1], &dd[2], &y, &l, &dd[3], window),
        );
        for _ in 0..2 {
            let again = ok(
                "recompute",
                m.causal_sdpa_backward_recompute(&dd[0], &dd[1], &dd[2], &dd[3], window),
            );
            for (a, b) in [
                (&saved.0, &again.0),
                (&saved.1, &again.1),
                (&saved.2, &again.2),
            ] {
                assert_eq!(bits(a), bits(b), "window {window:?}");
            }
        }
    }
}

/// Keys outside a row's window never reach it: huge finite values in every
/// key and value row the window excludes leave those rows' output, lse and
/// dQ bit for bit unchanged. (Finite: the matrix units multiply whole
/// tiles, so an excluded entry contributes an exact `0 * x`, which a NaN
/// would not; a NaN operand is a refused input on Metal anyway.)
#[test]
fn keys_outside_the_window_never_reach_a_row() {
    let m = metal();
    let (h, t, d, w) = (2usize, 96usize, 64usize, 10usize);
    let [q, k, v, gy] = inputs(1, h, h, t, d, 7500);
    // Rows `cut..cut + w` see keys from `cut + 1 - w` on: poison every key
    // and value row before that.
    let cut = 60usize;
    let poison = |x: &Tensor| {
        let mut x = x.to_f32_vec().unwrap();
        for head in 0..h {
            for pos in 0..cut + 1 - w {
                for j in 0..d {
                    x[(head * t + pos) * d + j] = if j % 2 == 0 { 1.0e30 } else { -1.0e30 };
                }
            }
        }
        host(&x, &[1, h, t, d])
    };
    let dd = [up(&m, &q), up(&m, &k), up(&m, &v), up(&m, &gy)];
    let (kp, vp) = (up(&m, &poison(&k)), up(&m, &poison(&v)));
    let (y, l) = ok(
        "clean",
        m.causal_sdpa_forward(&dd[0], &dd[1], &dd[2], Some(w)),
    );
    let (yp, lp) = ok("poisoned", m.causal_sdpa_forward(&dd[0], &kp, &vp, Some(w)));
    let gq = ok(
        "clean bwd",
        m.causal_sdpa_backward(&dd[0], &dd[1], &dd[2], &y, &l, &dd[3], Some(w)),
    )
    .0;
    let gqp = ok(
        "poisoned bwd",
        m.causal_sdpa_backward(&dd[0], &kp, &vp, &yp, &lp, &dd[3], Some(w)),
    )
    .0;
    let (y, yp, gq, gqp) = (bits(&y), bits(&yp), bits(&gq), bits(&gqp));
    let (l, lp) = (bits(&l), bits(&lp));
    for head in 0..h {
        for row in cut..t {
            assert_eq!(
                l[head * t + row],
                lp[head * t + row],
                "lse head {head} row {row}"
            );
            for j in 0..d {
                let i = (head * t + row) * d + j;
                assert_eq!(y[i], yp[i], "y head {head} row {row}");
                assert_eq!(gq[i], gqp[i], "dq head {head} row {row}");
            }
        }
    }
}
