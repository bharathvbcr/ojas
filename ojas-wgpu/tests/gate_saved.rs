//! The saved-sigmoid gate pair: `per_head_sigmoid_gate_forward_saving`
//! keeps the per-head sigmoid and `per_head_sigmoid_gate_backward_saved`
//! reads it instead of recomputing the logits. The saving forward's output
//! and the saved backward's gradients are bit for bit the plain pair's, and
//! all of them match the CPU reference.

mod common;

use common::*;
use ojas_core::{Backend, OjasError, Tensor};

const SHAPES: [(usize, usize, usize, usize); 4] = [
    (1, 1, 1, 1),
    (17, 13, 3, 5),
    (257, 64, 8, 64),
    (65, 129, 2, 7),
];

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
    let g = gpu();
    let c = cpu();
    for &(rows, din, heads, hd) in &SHAPES {
        let tag = format!("gate {rows}x{din} h{heads} d{hd}");
        let x = host(700, &[rows, din]);
        let w = host(701, &[heads, din]);
        let bias = host(702, &[heads]);
        let attn = host(703, &[rows, heads, hd]);
        let gy = host(704, &[rows, heads, hd]);
        let d = [up(&x), up(&w), up(&bias), up(&attn), up(&gy)];

        let plain = g
            .per_head_sigmoid_gate_forward(&d[0], &d[1], &d[2], &d[3])
            .unwrap();
        let (y, scales) = g
            .per_head_sigmoid_gate_forward_saving(&d[0], &d[1], &d[2], &d[3])
            .unwrap();
        let scales = scales.expect("wgpu keeps the scale");
        assert_eq!(scales.shape(), [rows, heads], "{tag}");
        assert_eq!(bits(&y), bits(&plain), "{tag}: saving forward output");
        close(
            &format!("{tag} y"),
            &y,
            &c.per_head_sigmoid_gate_forward(&x, &w, &bias, &attn)
                .unwrap(),
        );
        let want_scales = host_scales(
            &x.to_f32_vec().unwrap(),
            &w.to_f32_vec().unwrap(),
            &bias.to_f32_vec().unwrap(),
            rows,
            din,
        );
        close_vec(&format!("{tag} scales"), &down(&scales), &want_scales);

        let recomputed = g
            .per_head_sigmoid_gate_backward(&d[0], &d[1], &d[2], &d[3], &d[4])
            .unwrap();
        let saved = g
            .per_head_sigmoid_gate_backward_saved(&d[0], &d[1], &d[2], &d[3], &d[4], &scales)
            .unwrap();
        for (name, a, b) in [
            ("dx", &saved.input, &recomputed.input),
            ("dw", &saved.weight, &recomputed.weight),
            ("db", &saved.bias, &recomputed.bias),
            ("dattn", &saved.attn_out, &recomputed.attn_out),
        ] {
            assert_eq!(bits(a), bits(b), "{tag} {name}: saved vs recomputed");
        }
        let want = c
            .per_head_sigmoid_gate_backward(&x, &w, &bias, &attn, &gy)
            .unwrap();
        close(&format!("{tag} dx"), &saved.input, &want.input);
        close(&format!("{tag} dw"), &saved.weight, &want.weight);
        close(&format!("{tag} db"), &saved.bias, &want.bias);
        close(&format!("{tag} dattn"), &saved.attn_out, &want.attn_out);
        g.sync().unwrap();
    }
}

#[test]
fn the_saved_backward_refuses_a_wrong_scale_and_still_checks_the_bias() {
    let g = fresh();
    let (rows, din, heads, hd) = (4, 3, 2, 5);
    // Uploads go to this backend: a tensor from the shared `gpu()` is
    // another device's, which every op refuses as a placement error.
    let up_to = |t: &Tensor| g.upload(t).unwrap();
    let x = up_to(&host(710, &[rows, din]));
    let w = up_to(&host(711, &[heads, din]));
    let bias = up_to(&host(712, &[heads]));
    let attn = up_to(&host(713, &[rows, heads, hd]));
    let gy = up_to(&host(714, &[rows, heads, hd]));
    let (_y, scales) = g
        .per_head_sigmoid_gate_forward_saving(&x, &w, &bias, &attn)
        .unwrap();
    let scales = scales.unwrap();

    let wrong = up_to(&host(715, &[heads, rows]));
    let refused = g.per_head_sigmoid_gate_backward_saved(&x, &w, &bias, &attn, &gy, &wrong);
    assert!(
        matches!(refused, Err(OjasError::Shape { .. })),
        "{refused:?}"
    );
    let on_host = host(716, &[rows, heads]);
    let refused = g.per_head_sigmoid_gate_backward_saved(&x, &w, &bias, &attn, &gy, &on_host);
    assert!(
        matches!(refused, Err(OjasError::Placement { .. })),
        "{refused:?}"
    );
    g.sync().unwrap();

    // The saved kernel never reads the bias, but a NaN bias is still the
    // op's deferred fault, as in the recomputing backward.
    let bad_bias = up_to(&Tensor::from_f32(&[f32::NAN, 0.5], &[heads], host_budget()).unwrap());
    g.per_head_sigmoid_gate_backward_saved(&x, &w, &bad_bias, &attn, &gy, &scales)
        .unwrap();
    match g.sync() {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "per_head_sigmoid_gate_backward"),
        other => panic!("a NaN bias passed the saved backward: {other:?}"),
    }
    g.sync().unwrap();
}

#[path = "../../bench/gate_saved_ab.rs"]
mod gate_ab;

/// `cargo test --release -p ojas-wgpu --test gate_saved -- --ignored
/// --nocapture bench_`; results in `bench/results/2026-10-07-gate-saved/`.
#[test]
#[ignore = "benchmark; run with --ignored --nocapture in release"]
fn bench_gate_saved_against_recomputed() {
    let g = fresh();
    let (rounds, iters) = (6, 20);
    let stats = gate_ab::run(&g, 5, rounds, iters);
    let device = format!("wgpu {}", g.context().adapter_name());
    println!("{}", gate_ab::report(&device, rounds, iters, &stats));
}
