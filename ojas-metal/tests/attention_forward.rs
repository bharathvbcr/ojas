//! The causal attention forward against the CPU reference at the tile
//! boundaries and at T = 2048, at every compiled head-dim width and at widths
//! it pads, plus its error contract.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Backend, OjasError, Tensor};
use ojas_metal::MetalBackend;

/// The forward tolerance of `backend_parity.rs`, widened with sqrt(T) as the
/// backward's is: a row sums up to T terms.
fn tol(t: usize) -> f32 {
    2e-5 * (t as f32).sqrt().max(1.0)
}

fn check_case(m: &MetalBackend, b: usize, h: usize, t: usize, d: usize, seed: u64) {
    let c = cpu();
    let shape = [b, h, t, d];
    let q = rand(&shape, seed, 1.0);
    let k = rand(&shape, seed + 1, 1.0);
    let v = rand(&shape, seed + 2, 1.0);
    let tag = format!("sdpa fwd b{b} h{h} t{t} d{d}");
    let want = ok(&tag, c.causal_sdpa_forward(&q, &k, &v));
    let got = ok(
        &tag,
        m.causal_sdpa_forward(&up(m, &q), &up(m, &k), &up(m, &v)),
    );
    same_tensor(&tag, &got, &want, tol(t), 1e-4);
}

#[test]
fn forward_matches_cpu_at_tile_boundaries() {
    let m = metal();
    let mut cases = 0;
    for &t in &[1usize, 2, 17, 31, 32, 33, 64, 255] {
        for &d in &[16usize, 32, 64] {
            check_case(&m, 2, 3, t, d, (t * 1000 + d) as u64 + 5);
            cases += 1;
        }
    }
    assert_eq!(cases, 24);
}

/// Head dims that are not a compiled width run zero-padded to the next one.
#[test]
fn forward_matches_cpu_at_padded_head_dims() {
    let m = metal();
    for &d in &[1usize, 8, 20, 33, 63] {
        for &t in &[5usize, 40] {
            check_case(&m, 1, 2, t, d, (d * 77 + t) as u64 + 9);
        }
    }
}

#[test]
fn forward_matches_cpu_at_t2048() {
    let m = metal();
    check_case(&m, 1, 2, 2048, 64, 2048);
    check_case(&m, 2, 1, 2048, 16, 4096);
}

/// The paired-bench shape `sdpa_b4h12t1024d64`, including lengths that are
/// not a multiple of the 64-wide key tile (the 1024 row count is).
#[test]
fn forward_matches_cpu_at_the_bench_shape() {
    let m = metal();
    check_case(&m, 4, 12, 1024, 64, 7100);
    check_case(&m, 1, 1, 65, 64, 7101);
    check_case(&m, 1, 1, 63, 64, 7102);
}

/// Large-magnitude scores: the online softmax must rescale, not overflow.
#[test]
fn forward_matches_cpu_with_sharp_softmax() {
    let (m, c) = (metal(), cpu());
    let shape = [1usize, 2, 100, 32];
    let q = rand(&shape, 1, 8.0);
    let k = rand(&shape, 2, 8.0);
    let v = rand(&shape, 3, 1.0);
    let want = ok("cpu", c.causal_sdpa_forward(&q, &k, &v));
    let got = ok(
        "metal",
        m.causal_sdpa_forward(&up(&m, &q), &up(&m, &k), &up(&m, &v)),
    );
    same_tensor("sharp", &got, &want, tol(100), 1e-4);
}

/// Head dims above 64 run through `MetalBackend` up to
/// `METAL_MAX_HEAD_DIM` (256). The kernels are compiled to that width.
/// 257 is refused. Device tests cover the same ladder.
#[test]
fn head_dims_above_64_follow_the_core_limit() {
    let (m, c) = (metal(), cpu());
    let limit = ojas_core::METAL_MAX_HEAD_DIM as usize;
    for &d in &[96usize, 128, 192, 256, 257] {
        let shape = [2usize, 2, 45, d];
        let q = rand(&shape, d as u64, 1.0);
        let k = rand(&shape, d as u64 + 1, 1.0);
        let v = rand(&shape, d as u64 + 2, 1.0);
        let g = rand(&shape, d as u64 + 3, 1.0);
        let dd = [up(&m, &q), up(&m, &k), up(&m, &v), up(&m, &g)];
        let fwd = m.causal_sdpa_forward(&dd[0], &dd[1], &dd[2]);
        let bwd = m.causal_sdpa_backward(&dd[0], &dd[1], &dd[2], &dd[3]);
        if d > limit {
            assert!(
                matches!(fwd, Err(OjasError::UnsupportedHeadDim { .. })),
                "d{d}: {fwd:?}"
            );
            assert!(
                matches!(bwd, Err(OjasError::UnsupportedHeadDim { .. })),
                "d{d}: {bwd:?}"
            );
            continue;
        }
        let tag = format!("d{d}");
        same_tensor(
            &tag,
            &ok(&tag, fwd),
            &ok(&tag, c.causal_sdpa_forward(&q, &k, &v)),
            tol(45),
            1e-4,
        );
        let (gq, gk, gv) = ok(&tag, bwd);
        let (wq, wk, wv) = ok(&tag, c.causal_sdpa_backward(&q, &k, &v, &g));
        same_tensor(&format!("{tag} dq"), &gq, &wq, tol(45), 1e-3);
        same_tensor(&format!("{tag} dk"), &gk, &wk, tol(45), 1e-3);
        same_tensor(&format!("{tag} dv"), &gv, &wv, tol(45), 1e-3);
    }
}

/// Grouped-query causal SDPA against the CPU reference, including a head
/// dim the 256-wide kernel pads and one it runs natively.
#[test]
fn grouped_query_matches_cpu() {
    let (m, c) = (metal(), cpu());
    let cases = [
        (1usize, 4, 2, 8, 32),
        (1, 2, 1, 17, 192),
        (2, 6, 3, 5, 13),
        (1, 4, 1, 9, 256),
        (1, 6, 2, 7, 129),
    ];
    for (i, (b, h, hkv, t, d)) in cases.into_iter().enumerate() {
        let qs = [b, h, t, d];
        let ks = [b, hkv, t, d];
        let seed = 8000 + i as u64;
        let q = rand(&qs, seed, 1.0);
        let k = rand(&ks, seed + 1, 1.0);
        let v = rand(&ks, seed + 2, 1.0);
        let g = rand(&qs, seed + 3, 1.0);
        let tag = format!("gqa h{h}/{hkv} t{t} d{d}");
        let dd = [up(&m, &q), up(&m, &k), up(&m, &v), up(&m, &g)];
        same_tensor(
            &tag,
            &ok(&tag, m.causal_sdpa_forward(&dd[0], &dd[1], &dd[2])),
            &ok(&tag, c.causal_sdpa_forward(&q, &k, &v)),
            tol(t),
            1e-4,
        );
        let (gq, gk, gv) = ok(&tag, m.causal_sdpa_backward(&dd[0], &dd[1], &dd[2], &dd[3]));
        let (wq, wk, wv) = ok(&tag, c.causal_sdpa_backward(&q, &k, &v, &g));
        same_tensor(&format!("{tag} dq"), &gq, &wq, tol(t), 1e-3);
        same_tensor(&format!("{tag} dk"), &gk, &wk, tol(t), 1e-3);
        same_tensor(&format!("{tag} dv"), &gv, &wv, tol(t), 1e-3);
    }
}

#[test]
fn forward_is_deterministic() {
    let m = metal();
    let shape = [2usize, 2, 100, 64];
    let d: Vec<Tensor> = (0..3).map(|i| up(&m, &rand(&shape, 60 + i, 1.0))).collect();
    let first = down(&ok("first", m.causal_sdpa_forward(&d[0], &d[1], &d[2])));
    for _ in 0..3 {
        let again = down(&ok("again", m.causal_sdpa_forward(&d[0], &d[1], &d[2])));
        assert!(
            first
                .iter()
                .zip(&again)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "two runs differ"
        );
    }
}

/// Inputs at a byte offset that is a multiple of 4 but not of 16 read their
/// own window.
#[test]
fn forward_reads_views_at_unaligned_offsets() {
    let (m, c) = (metal(), cpu());
    let shape = [1usize, 2, 37, 32];
    let n: usize = shape.iter().product();
    let strides = [2 * 37 * 32, 37 * 32, 32, 1];
    let mut device = Vec::new();
    let mut host_inputs = Vec::new();
    for i in 0..3u64 {
        let pad = 1 + i as usize;
        let all = values(n + pad, 80 + i, 1.0);
        let big = up(&m, &host(&all, &[n + pad]));
        device.push(ok("view", big.view(&shape, &strides, pad * 4)));
        host_inputs.push(host(&all[pad..], &shape));
    }
    let want = ok(
        "cpu",
        c.causal_sdpa_forward(&host_inputs[0], &host_inputs[1], &host_inputs[2]),
    );
    let got = ok(
        "metal",
        m.causal_sdpa_forward(&device[0], &device[1], &device[2]),
    );
    same_tensor("offset fwd", &got, &want, tol(37), 1e-4);
}

#[test]
fn non_finite_inputs_and_overflowing_scores_are_refused() {
    let m = metal();
    let shape = [1usize, 2, 40, 64];
    let n: usize = shape.iter().product();
    for which in 0..3 {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut ins: Vec<Tensor> = (0..3).map(|i| rand(&shape, 90 + i, 1.0)).collect();
            let mut vals = values(n, 98, 1.0);
            vals[n / 2 + which] = bad;
            ins[which] = host(&vals, &shape);
            let d: Vec<Tensor> = ins.iter().map(|t| up(&m, t)).collect();
            let r = m.causal_sdpa_forward(&d[0], &d[1], &d[2]);
            deferred(
                &m,
                &format!("input {which} = {bad}"),
                r,
                "causal_sdpa_forward",
            );
        }
    }
    // Finite inputs whose scores overflow f32.
    let huge = up(&m, &host(&vec![3.0e19f32; n], &shape));
    let small = up(&m, &rand(&shape, 7, 1.0));
    let r = m.causal_sdpa_forward(&huge, &huge, &small);
    deferred(&m, "overflowing scores", r, "causal_sdpa_forward");
}
