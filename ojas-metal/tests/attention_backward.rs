//! The tiled causal attention backward against the CPU reference at the
//! tile boundaries (T = 1, 2, 17, 64, 255) and at T = 2048, at every compiled
//! head-dim width and at widths it pads, plus its error contract.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Backend, Tensor};
use ojas_metal::MetalBackend;

/// The forward/backward attention tolerance of `backend_parity.rs`.
fn tol(t: usize) -> f32 {
    2e-5 * (t as f32).sqrt().max(1.0)
}

fn check_case(m: &MetalBackend, b: usize, h: usize, t: usize, d: usize, seed: u64) {
    let c = cpu();
    let shape = [b, h, t, d];
    let q = rand(&shape, seed, 1.0);
    let k = rand(&shape, seed + 1, 1.0);
    let v = rand(&shape, seed + 2, 1.0);
    let g = rand(&shape, seed + 3, 1.0);
    let tag = format!("sdpa bwd b{b} h{h} t{t} d{d}");
    let (wq, wk, wv) = ok(&tag, c.causal_sdpa_backward_recompute(&q, &k, &v, &g, None));
    let dd = [up(m, &q), up(m, &k), up(m, &v), up(m, &g)];
    let (gq, gk, gv) = ok(
        &tag,
        m.causal_sdpa_backward_recompute(&dd[0], &dd[1], &dd[2], &dd[3], None),
    );
    same_tensor(&format!("{tag} dq"), &gq, &wq, tol(t), 1e-3);
    same_tensor(&format!("{tag} dk"), &gk, &wk, tol(t), 1e-3);
    same_tensor(&format!("{tag} dv"), &gv, &wv, tol(t), 1e-3);
}

#[test]
fn backward_matches_cpu_at_tile_boundaries() {
    let m = metal();
    let mut cases = 0;
    for &t in &[1usize, 2, 17, 31, 32, 33, 64, 255] {
        for &d in &[16usize, 32, 64] {
            check_case(&m, 2, 3, t, d, (t * 1000 + d) as u64);
            cases += 1;
        }
    }
    assert_eq!(cases, 24);
}

/// Head dims that are not a compiled width run zero-padded to the next one.
#[test]
fn backward_matches_cpu_at_padded_head_dims() {
    let m = metal();
    for &d in &[1usize, 8, 20, 33, 63] {
        for &t in &[5usize, 40] {
            check_case(&m, 1, 2, t, d, (d * 77 + t) as u64);
        }
    }
}

#[test]
fn backward_matches_cpu_at_t2048() {
    let m = metal();
    check_case(&m, 1, 2, 2048, 64, 2048);
    check_case(&m, 2, 1, 2048, 16, 4096);
}

/// The paired-bench shape `sdpa_b4h12t1024d64`, plus lengths that straddle
/// the 64-wide key tile.
#[test]
fn backward_matches_cpu_at_the_bench_shape() {
    let m = metal();
    check_case(&m, 4, 12, 1024, 64, 7200);
    check_case(&m, 1, 1, 65, 64, 7201);
    check_case(&m, 1, 1, 63, 64, 7202);
}

#[test]
fn backward_is_deterministic() {
    let m = metal();
    let shape = [2usize, 2, 100, 64];
    let d: Vec<Tensor> = (0..4).map(|i| up(&m, &rand(&shape, 50 + i, 1.0))).collect();
    let first = ok(
        "first",
        m.causal_sdpa_backward_recompute(&d[0], &d[1], &d[2], &d[3], None),
    );
    for _ in 0..3 {
        let again = ok(
            "again",
            m.causal_sdpa_backward_recompute(&d[0], &d[1], &d[2], &d[3], None),
        );
        for (a, b) in [
            (&first.0, &again.0),
            (&first.1, &again.1),
            (&first.2, &again.2),
        ] {
            let (a, b) = (down(a), down(b));
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "two runs differ"
            );
        }
    }
}

/// Inputs at a byte offset that is a multiple of 4 but not of 16 read their
/// own window.
#[test]
fn backward_reads_views_at_unaligned_offsets() {
    let (m, c) = (metal(), cpu());
    let shape = [1usize, 2, 37, 32];
    let n: usize = shape.iter().product();
    let strides = [2 * 37 * 32, 37 * 32, 32, 1];
    let mut device = Vec::new();
    let mut host_inputs = Vec::new();
    for i in 0..4u64 {
        let pad = 1 + i as usize; // 4..16 bytes in
        let all = values(n + pad, 70 + i, 1.0);
        let big = up(&m, &host(&all, &[n + pad]));
        device.push(ok("view", big.view(&shape, &strides, pad * 4)));
        host_inputs.push(host(&all[pad..], &shape));
    }
    let (wq, wk, wv) = ok(
        "cpu",
        c.causal_sdpa_backward_recompute(
            &host_inputs[0],
            &host_inputs[1],
            &host_inputs[2],
            &host_inputs[3],
            None,
        ),
    );
    let (gq, gk, gv) = ok(
        "metal",
        m.causal_sdpa_backward_recompute(&device[0], &device[1], &device[2], &device[3], None),
    );
    same_tensor("offset dq", &gq, &wq, tol(37), 1e-3);
    same_tensor("offset dk", &gk, &wk, tol(37), 1e-3);
    same_tensor("offset dv", &gv, &wv, tol(37), 1e-3);
}

#[test]
fn non_finite_inputs_and_overflowing_scores_are_refused() {
    let m = metal();
    let shape = [1usize, 2, 40, 64];
    let n: usize = shape.iter().product();
    let rows = [1usize, 2, 40];
    let c = cpu();
    // q, k, v, output, lse, grad_output: a clean forward's output and lse,
    // then one operand of six poisoned.
    let qkv: Vec<Tensor> = (0..3).map(|i| rand(&shape, 90 + i, 1.0)).collect();
    let (out, lse) = ok(
        "cpu forward",
        c.causal_sdpa_forward(&qkv[0], &qkv[1], &qkv[2], None),
    );
    let clean = [
        qkv[0].clone(),
        qkv[1].clone(),
        qkv[2].clone(),
        out,
        lse,
        rand(&shape, 93, 1.0),
    ];
    for which in 0..6 {
        for bad in [f32::NAN, f32::INFINITY] {
            let mut ins = clean.clone();
            let mut vals = ins[which].to_f32_vec().expect("host values");
            let at = vals.len() / 2 + which;
            vals[at] = bad;
            ins[which] = host(&vals, ins[which].shape());
            let d: Vec<Tensor> = ins.iter().map(|t| up(&m, t)).collect();
            let r = m.causal_sdpa_backward(&d[0], &d[1], &d[2], &d[3], &d[4], &d[5], None);
            deferred(
                &m,
                &format!("operand {which} = {bad}"),
                r,
                "causal_sdpa_backward",
            );
        }
    }
    // Finite operands whose scores overflow f32.
    let huge = host(&vec![3.0e19f32; n], &shape);
    let small = rand(&shape, 7, 1.0);
    let zero_rows = host(&vec![0.0f32; rows.iter().product()], &rows);
    let (hq, s, z) = (up(&m, &huge), up(&m, &small), up(&m, &zero_rows));
    let r = m.causal_sdpa_backward(&hq, &hq, &s, &s, &z, &s, None);
    deferred(&m, "overflowing scores", r, "causal_sdpa_backward");
}
