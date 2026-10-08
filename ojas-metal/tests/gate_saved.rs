//! The saved-sigmoid gate pair on Metal: `per_head_sigmoid_gate_forward_saving`
//! keeps the per-head sigmoid and `per_head_sigmoid_gate_backward_saved`
//! reads it instead of recomputing the logits. The saving forward's output
//! and the saved backward's gradients are bit for bit the plain pair's, and
//! all of them match the CPU reference.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::{Backend, OjasError, Tensor};

const GEMM_ATOL: f32 = 2e-5;
const GEMM_RTOL: f32 = 2e-4;
const PW_ATOL: f32 = 1e-6;
const PW_RTOL: f32 = 1e-5;

/// `sigmoid(x @ W^T + b)` per (row, head), in f64.
fn host_scales(x: &[f32], w: &[f32], b: &[f32], rows: usize, din: usize) -> Vec<f32> {
    let heads = b.len();
    let mut out = Vec::with_capacity(rows * heads);
    for r in 0..rows {
        for h in 0..heads {
            let mut z = f64::from(b[h]);
            for k in 0..din {
                z += f64::from(x[r * din + k]) * f64::from(w[h * din + k]);
            }
            out.push((1.0 / (1.0 + (-z).exp())) as f32);
        }
    }
    out
}

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

#[test]
fn the_saved_pair_is_the_plain_pair_bit_for_bit_and_matches_cpu() {
    let (m, c) = (metal(), cpu());
    for &(b, t, din, h, dh) in &[
        (1usize, 1usize, 1usize, 1usize, 1usize),
        (2, 17, 11, 3, 16),
        (1, 127, 67, 8, 64),
    ] {
        let tag = format!("gate {b}x{t}x{din} h{h} d{dh}");
        let x = rand(&[b, t, din], 21, 1.0);
        let w = rand(&[h, din], 22, 1.0);
        let bias = rand(&[h], 23, 1.0);
        let a = rand(&[b, t, h, dh], 24, 1.0);
        let gy = rand(&[b, t, h, dh], 25, 1.0);
        let d = [
            up(&m, &x),
            up(&m, &w),
            up(&m, &bias),
            up(&m, &a),
            up(&m, &gy),
        ];
        let tol = GEMM_ATOL * din.max(dh * b * t) as f32;

        let plain = ok(
            "forward",
            m.per_head_sigmoid_gate_forward(&d[0], &d[1], &d[2], &d[3]),
        );
        let (y, scales) = ok(
            "forward_saving",
            m.per_head_sigmoid_gate_forward_saving(&d[0], &d[1], &d[2], &d[3]),
        );
        let scales = scales.expect("Metal keeps the scale");
        assert_eq!(scales.shape(), [b * t, h], "{tag}");
        assert_eq!(bits(&y), bits(&plain), "{tag}: saving forward output");
        let want_y = ok("cpu", c.per_head_sigmoid_gate_forward(&x, &w, &bias, &a));
        same_tensor(&format!("{tag} y"), &y, &want_y, tol, GEMM_RTOL);
        let want_scales = host_scales(
            &ok("x", x.to_f32_vec()),
            &ok("w", w.to_f32_vec()),
            &ok("b", bias.to_f32_vec()),
            b * t,
            din,
        );
        close(
            &format!("{tag} scales"),
            &down(&scales),
            &want_scales,
            tol,
            GEMM_RTOL,
        );

        let recomputed = ok(
            "backward",
            m.per_head_sigmoid_gate_backward(&d[0], &d[1], &d[2], &d[3], &d[4]),
        );
        let saved = ok(
            "backward_saved",
            m.per_head_sigmoid_gate_backward_saved(&d[0], &d[1], &d[2], &d[3], &d[4], &scales),
        );
        for (name, s, r) in [
            ("dx", &saved.input, &recomputed.input),
            ("dw", &saved.weight, &recomputed.weight),
            ("db", &saved.bias, &recomputed.bias),
            ("dattn", &saved.attn_out, &recomputed.attn_out),
        ] {
            assert_eq!(bits(s), bits(r), "{tag} {name}: saved vs recomputed");
        }
        let want = ok(
            "cpu bwd",
            c.per_head_sigmoid_gate_backward(&x, &w, &bias, &a, &gy),
        );
        same_tensor("gate dx", &saved.input, &want.input, tol, GEMM_RTOL);
        same_tensor("gate dw", &saved.weight, &want.weight, tol, GEMM_RTOL);
        same_tensor("gate db", &saved.bias, &want.bias, tol, GEMM_RTOL);
        same_tensor(
            "gate dattn",
            &saved.attn_out,
            &want.attn_out,
            PW_ATOL,
            PW_RTOL,
        );
        ok("sync", m.sync());
    }
}

#[test]
fn the_saved_backward_refuses_a_wrong_scale_and_still_checks_the_bias() {
    let m = metal();
    let (rows, din, heads, hd) = (4, 3, 2, 5);
    let x = up(&m, &rand(&[rows, din], 31, 1.0));
    let w = up(&m, &rand(&[heads, din], 32, 1.0));
    let bias = up(&m, &rand(&[heads], 33, 1.0));
    let attn = up(&m, &rand(&[rows, heads, hd], 34, 1.0));
    let gy = up(&m, &rand(&[rows, heads, hd], 35, 1.0));
    let (_y, scales) = ok(
        "forward_saving",
        m.per_head_sigmoid_gate_forward_saving(&x, &w, &bias, &attn),
    );
    let scales = scales.expect("Metal keeps the scale");

    let wrong = up(&m, &rand(&[heads, rows], 36, 1.0));
    let refused = m.per_head_sigmoid_gate_backward_saved(&x, &w, &bias, &attn, &gy, &wrong);
    assert!(
        matches!(refused, Err(OjasError::Shape { .. })),
        "{refused:?}"
    );
    let on_host = rand(&[rows, heads], 37, 1.0);
    let refused = m.per_head_sigmoid_gate_backward_saved(&x, &w, &bias, &attn, &gy, &on_host);
    assert!(
        matches!(refused, Err(OjasError::Placement { .. })),
        "{refused:?}"
    );
    ok("sync", m.sync());

    // The saved kernel never reads the bias, but a NaN bias is still the
    // op's deferred fault, as in the recomputing backward.
    let bad_bias = up(&m, &host(&[f32::NAN, 0.5], &[heads]));
    ok(
        "backward_saved records",
        m.per_head_sigmoid_gate_backward_saved(&x, &w, &bad_bias, &attn, &gy, &scales),
    );
    match m.sync() {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "per_head_sigmoid_gate_backward"),
        other => panic!("a NaN bias passed the saved backward: {other:?}"),
    }
    ok("sync", m.sync());
}

#[path = "../../bench/gate_saved_ab.rs"]
mod gate_ab;

/// `cargo test --release -p ojas-metal --test gate_saved -- --ignored
/// --nocapture bench_`; results in `bench/results/2026-10-07-gate-saved/`.
#[test]
#[ignore = "benchmark; run with --ignored --nocapture in release"]
fn bench_gate_saved_against_recomputed() {
    let m = metal();
    let (rounds, iters) = (6, 20);
    let stats = gate_ab::run(&m, 5, rounds, iters);
    let device = format!("Metal {}", m.device_name());
    println!("{}", gate_ab::report(&device, rounds, iters, &stats));
}
