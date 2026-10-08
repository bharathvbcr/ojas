//! Sliding-window and grouped-query causal attention, and the row
//! log-sum-exp the forward returns, against the CPU reference; plus the
//! bit-level contracts of the native grouped-query kernels and of the
//! saved-statistics backward.

mod common;

use common::*;
use ojas_core::{Backend, Tensor};

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

/// `[q, k, v, gy]` host tensors of a `[B, H, T, D]` query over `Hkv` KV heads.
fn inputs(b: usize, h: usize, hkv: usize, t: usize, d: usize, seed: u64) -> [Tensor; 4] {
    let (qs, ks) = ([b, h, t, d], [b, hkv, t, d]);
    [
        host(seed, &qs),
        host(seed + 1, &ks),
        host(seed + 2, &ks),
        host(seed + 3, &qs),
    ]
}

/// Forward (output and lse) and the saved-statistics backward on wgpu
/// against the CPU at one case.
fn parity(case: (usize, usize, usize, usize, usize), window: Option<usize>, seed: u64) {
    let (b, h, hkv, t, d) = case;
    let c = cpu();
    let g = gpu();
    let [q, k, v, gy] = inputs(b, h, hkv, t, d, seed);
    let tag = format!("b{b} h{h}/{hkv} t{t} d{d} window {window:?}");
    let (wy, wl) = c.causal_sdpa_forward(&q, &k, &v, window).unwrap();
    let (wq, wk, wv) = c
        .causal_sdpa_backward(&q, &k, &v, &wy, &wl, &gy, window)
        .unwrap();
    let dd = [up(&q), up(&k), up(&v), up(&gy)];
    let (y, l) = g
        .causal_sdpa_forward(&dd[0], &dd[1], &dd[2], window)
        .unwrap();
    close(&format!("{tag} y"), &y, &wy);
    close(&format!("{tag} lse"), &l, &wl);
    let (gq, gk, gv) = g
        .causal_sdpa_backward(&dd[0], &dd[1], &dd[2], &y, &l, &dd[3], window)
        .unwrap();
    close(&format!("{tag} dq"), &gq, &wq);
    close(&format!("{tag} dk"), &gk, &wk);
    close(&format!("{tag} dv"), &gv, &wv);
    g.sync().unwrap();
}

/// Windows inside a block, across blocks, one short of `T`, and past it;
/// multi-head and grouped; padded and full head widths up to 256.
#[test]
fn windowed_attention_matches_cpu() {
    let mut seed = 8100;
    for case in [
        (1usize, 2usize, 2usize, 67usize, 64usize),
        (2, 4, 2, 100, 48),
        (1, 8, 2, 70, 256),
        (1, 6, 3, 45, 13),
    ] {
        let t = case.3;
        for window in [
            Some(1),
            Some(5),
            Some(16),
            Some(33),
            Some(t - 1),
            Some(t + 3),
            None,
        ] {
            seed += 4;
            parity(case, window, seed);
        }
    }
}

/// Grouped-query heads read their KV head in place: the output, lse and
/// dQ equal the multi-head kernels' on K and V repeated to the query heads,
/// bit for bit; dK and dV equal the repeated gradients summed back within
/// rounding.
#[test]
fn native_grouped_query_equals_repeated_heads() {
    let g = gpu();
    for (b, h, hkv, t, d, window) in [
        (2usize, 8usize, 2usize, 70usize, 64usize, None),
        (1, 4, 1, 40, 256, Some(9)),
    ] {
        let rep = h / hkv;
        let [q, k, v, gy] = inputs(b, h, hkv, t, d, 8300);
        let plane = t * d;
        let repeat = |x: &Tensor| {
            let x = x.to_f32_vec().unwrap();
            let mut out = Vec::with_capacity(b * h * plane);
            for bh in 0..b * h {
                let src = (bh / h) * hkv + (bh % h) / rep;
                out.extend_from_slice(&x[src * plane..(src + 1) * plane]);
            }
            Tensor::from_f32(&out, &[b, h, t, d], host_budget()).unwrap()
        };
        let dd = [up(&q), up(&k), up(&v), up(&gy)];
        let (dkr, dvr) = (up(&repeat(&k)), up(&repeat(&v)));
        let tag = format!("h{h}/{hkv} t{t} d{d} window {window:?}");
        let (y, l) = g
            .causal_sdpa_forward(&dd[0], &dd[1], &dd[2], window)
            .unwrap();
        let (yr, lr) = g.causal_sdpa_forward(&dd[0], &dkr, &dvr, window).unwrap();
        assert_eq!(bits(&y), bits(&yr), "{tag}: output");
        assert_eq!(bits(&l), bits(&lr), "{tag}: lse");
        let (gq, gk, gv) = g
            .causal_sdpa_backward(&dd[0], &dd[1], &dd[2], &y, &l, &dd[3], window)
            .unwrap();
        let (gqr, gkr, gvr) = g
            .causal_sdpa_backward(&dd[0], &dkr, &dvr, &yr, &lr, &dd[3], window)
            .unwrap();
        assert_eq!(bits(&gq), bits(&gqr), "{tag}: dq");
        let sum_back = |x: &Tensor| {
            let x = down(x);
            let mut out = vec![0.0f32; b * hkv * plane];
            for bh in 0..b * h {
                let dst = (bh / h) * hkv + (bh % h) / rep;
                for (o, v) in out[dst * plane..(dst + 1) * plane]
                    .iter_mut()
                    .zip(&x[bh * plane..(bh + 1) * plane])
                {
                    *o += v;
                }
            }
            out
        };
        close_vec(&format!("{tag} dk"), &down(&gk), &sum_back(&gkr));
        close_vec(&format!("{tag} dv"), &down(&gv), &sum_back(&gvr));
        g.sync().unwrap();
    }
}

/// The backward from the saved output and lse equals the recomputing
/// backward bit for bit, and repeats bit for bit.
#[test]
fn saved_backward_equals_the_recomputing_one_bit_for_bit() {
    let g = gpu();
    let [q, k, v, gy] = inputs(1, 4, 2, 150, 64, 8400);
    let dd = [up(&q), up(&k), up(&v), up(&gy)];
    for window in [None, Some(40)] {
        let (y, l) = g
            .causal_sdpa_forward(&dd[0], &dd[1], &dd[2], window)
            .unwrap();
        let saved = g
            .causal_sdpa_backward(&dd[0], &dd[1], &dd[2], &y, &l, &dd[3], window)
            .unwrap();
        for _ in 0..2 {
            let again = g
                .causal_sdpa_backward_recompute(&dd[0], &dd[1], &dd[2], &dd[3], window)
                .unwrap();
            for (a, b) in [
                (&saved.0, &again.0),
                (&saved.1, &again.1),
                (&saved.2, &again.2),
            ] {
                assert_eq!(bits(a), bits(b), "window {window:?}");
            }
        }
    }
    g.sync().unwrap();
}

/// Keys outside a row's window never reach it, not even as `0 * NaN`: NaN
/// in every key and value row that rows `cut..` exclude leaves those rows'
/// output, lse and dQ bit for bit unchanged. (Earlier rows do see the
/// poisoned keys, so that call faults.)
#[test]
fn keys_outside_the_window_never_reach_a_row() {
    let g = &fresh();
    g.sync().unwrap();
    let (h, t, d, w) = (2usize, 96usize, 48usize, 10usize);
    let [q, k, v, gy] = inputs(1, h, h, t, d, 8500);
    let cut = 60usize;
    let poison = |x: &Tensor| {
        let mut x = x.to_f32_vec().unwrap();
        for head in 0..h {
            for pos in 0..cut + 1 - w {
                for j in 0..d {
                    x[(head * t + pos) * d + j] = f32::NAN;
                }
            }
        }
        Tensor::from_f32(&x, &[1, h, t, d], host_budget()).unwrap()
    };
    let upl = |x: &Tensor| g.upload(x).unwrap();
    let dd = [upl(&q), upl(&k), upl(&v), upl(&gy)];
    let (kp, vp) = (upl(&poison(&k)), upl(&poison(&v)));
    let (y, l) = g
        .causal_sdpa_forward(&dd[0], &dd[1], &dd[2], Some(w))
        .unwrap();
    let gq = g
        .causal_sdpa_backward(&dd[0], &dd[1], &dd[2], &y, &l, &dd[3], Some(w))
        .unwrap()
        .0;
    g.sync().unwrap();
    let bitsg = |x: &Tensor| -> Vec<u32> {
        g.download(x)
            .unwrap()
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    let (yp, lp) = g.causal_sdpa_forward(&dd[0], &kp, &vp, Some(w)).unwrap();
    let gqp = g
        .causal_sdpa_backward(&dd[0], &kp, &vp, &yp, &lp, &dd[3], Some(w))
        .unwrap()
        .0;
    assert!(g.sync().is_err(), "rows before the cut see poisoned keys");
    let pick = |all: Vec<u32>, per: usize| {
        let mut out = Vec::new();
        for head in 0..h {
            out.extend_from_slice(&all[(head * t + cut) * per..(head * t + t) * per]);
        }
        out
    };
    assert_eq!(pick(bitsg(&y), d), pick(bitsg(&yp), d), "output");
    assert_eq!(pick(bitsg(&l), 1), pick(bitsg(&lp), 1), "lse");
    assert_eq!(pick(bitsg(&gq), d), pick(bitsg(&gqp), d), "dq");
}

/// Bytes a Qwen3.5-2B grouped-query attention call (8 query heads over 2
/// KV heads, head dim 256) charges beyond its operands, with the figure the
/// removed expand path charged (c74f3ba: K and V repeated to the query
/// heads in the forward, and K, V, dK and dV in the backward).
#[test]
fn grouped_query_charges_no_expanded_heads_at_the_qwen35_shape() {
    let g = own();
    let (b, h, hkv, t, d) = (1usize, 8usize, 2usize, 2048usize, 256usize);
    let (q_bytes, kv_bytes, rows_bytes) = (4 * b * h * t * d, 4 * b * hkv * t * d, 4 * b * h * t);
    // The removed path: the forward held K and V repeated to the query
    // heads; the backward also held dK and dV at query size, beside the
    // log-sum-exp and `Dr` rows it formed itself.
    let old_fwd = q_bytes + 2 * q_bytes;
    let old_bwd = q_bytes + 2 * kv_bytes + 2 * rows_bytes + 4 * q_bytes;
    let [q, k, v, gy] = inputs(b, h, hkv, t, d, 7700);
    let dd = [&q, &k, &v, &gy].map(|x| g.upload(x).unwrap());
    g.sync().unwrap();
    let live = g.budget().live_bytes().unwrap();
    g.budget().reset_peak();
    let (y, l) = g.causal_sdpa_forward(&dd[0], &dd[1], &dd[2], None).unwrap();
    g.sync().unwrap();
    let fwd_peak = (g.budget().peak_bytes() - live) as usize;
    let live = g.budget().live_bytes().unwrap();
    g.budget().reset_peak();
    let grads = g
        .causal_sdpa_backward(&dd[0], &dd[1], &dd[2], &y, &l, &dd[3], None)
        .unwrap();
    g.sync().unwrap();
    let bwd_peak = (g.budget().peak_bytes() - live) as usize;
    drop(grads);
    // Outputs only: y and the row lse; dQ, dK, dV and the two row
    // statistics.
    assert_eq!(fwd_peak, q_bytes + rows_bytes, "forward charge");
    assert_eq!(
        bwd_peak,
        q_bytes + 2 * kv_bytes + 2 * rows_bytes,
        "backward charge"
    );
    eprintln!(
        "wgpu qwen35 b{b} h{h}/{hkv} t{t} d{d}: forward {fwd_peak} B (expand path {old_fwd} B), backward {bwd_peak} B (expand path {old_bwd} B)"
    );
}
