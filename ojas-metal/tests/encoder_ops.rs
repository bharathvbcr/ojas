//! The encoder-family forward ops on Metal against the CPU backend:
//! `layer_norm_forward`, `channel_layer_norm_forward` and `gelu_erf_forward`
//! on tessl's BERT kernels (`bert::layer_norm`, `bert::bias_gelu_erf`),
//! `linear_bias_forward`, `broadcast_add_forward` / `broadcast_mul_forward`,
//! `scale_forward` and `softmax_forward` on `ojas_backend.metal`, plus the
//! refusals and the deferred non-finite fault. The CPU itself is held to
//! `f64` references in `ojas-cpu/tests/encoder_ops.rs`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Backend, OjasError};

const EPS: f32 = 1e-5;

#[test]
fn layer_norm_matches_cpu() {
    let (m, c) = (metal(), cpu());
    for (i, shape) in [
        vec![5, 16],
        vec![2, 3, 24],
        vec![1, 768],
        vec![4, 1, 33],
        vec![9],
    ]
    .iter()
    .enumerate()
    {
        let dim = *shape.last().unwrap();
        let seed = 10 * i as u64;
        let x = rand(shape, seed, 3.0);
        let w = rand(&[dim], seed + 1, 2.0);
        let b = rand(&[dim], seed + 2, 1.0);
        let want = ok("cpu", c.layer_norm_forward(&x, &w, &b, EPS));
        let got = ok(
            "metal",
            m.layer_norm_forward(&up(&m, &x), &up(&m, &w), &up(&m, &b), EPS),
        );
        ok("sync", m.sync());
        same_tensor(&format!("{shape:?}"), &got, &want, 1e-4, 1e-4);
    }
}

#[test]
fn channel_layer_norm_matches_cpu() {
    let (m, c) = (metal(), cpu());
    // [N, C], [N, C, 1, 1] (spatial 1), [N, C, L] and [N, C, H, W].
    for (i, shape) in [
        vec![3, 8],
        vec![2, 8, 1, 1],
        vec![2, 6, 5],
        vec![1, 16, 7, 9],
        vec![2, 256, 4, 4],
    ]
    .iter()
    .enumerate()
    {
        let ch = shape[1];
        let seed = 100 + 10 * i as u64;
        let x = rand(shape, seed, 2.0);
        let w = rand(&[ch], seed + 1, 2.0);
        let b = rand(&[ch], seed + 2, 1.0);
        let want = ok("cpu", c.channel_layer_norm_forward(&x, &w, &b, 1e-6));
        let got = ok(
            "metal",
            m.channel_layer_norm_forward(&up(&m, &x), &up(&m, &w), &up(&m, &b), 1e-6),
        );
        ok("sync", m.sync());
        same_tensor(&format!("{shape:?}"), &got, &want, 1e-4, 1e-4);
    }
}

#[test]
fn layer_norm_refuses_a_zero_eps_it_cannot_run() {
    let m = metal();
    let x = up(&m, &rand(&[2, 4], 1, 1.0));
    let w = up(&m, &rand(&[4], 2, 1.0));
    let r = m.layer_norm_forward(&x, &w, &w, 0.0);
    assert!(
        matches!(
            r,
            Err(OjasError::Unsupported {
                op: "layer_norm_forward",
                ..
            })
        ),
        "{r:?}"
    );
}

#[test]
fn gelu_erf_matches_cpu_across_the_tails() {
    let (m, c) = (metal(), cpu());
    let xs: Vec<f32> = (-300..=300).map(|i| i as f32 * 0.03).collect();
    let x = host(&xs, &[xs.len()]);
    let want = ok("cpu", c.gelu_erf_forward(&x));
    let got = ok("metal", m.gelu_erf_forward(&up(&m, &x)));
    ok("sync", m.sync());
    same_tensor("gelu", &got, &want, 1e-6, 1e-5);
    // A 2-D shape keeps its layout.
    let y = host(&values(6 * 41, 3, 4.0), &[6, 41]);
    let want = ok("cpu", c.gelu_erf_forward(&y));
    let got = ok("metal", m.gelu_erf_forward(&up(&m, &y)));
    same_tensor("gelu 2d", &got, &want, 1e-6, 1e-5);
}

#[test]
fn linear_bias_matches_cpu() {
    let (m, c) = (metal(), cpu());
    for (i, (shape, out)) in [(vec![4, 9], 6), (vec![2, 3, 17], 5), (vec![64, 128], 96)]
        .into_iter()
        .enumerate()
    {
        let input = *shape.last().unwrap();
        let seed = 200 + 10 * i as u64;
        let x = rand(&shape, seed, 1.0);
        let w = rand(&[out, input], seed + 1, 1.0);
        let b = rand(&[out], seed + 2, 1.0);
        let want = ok("cpu", c.linear_bias_forward(&x, &w, &b));
        let got = ok(
            "metal",
            m.linear_bias_forward(&up(&m, &x), &up(&m, &w), &up(&m, &b)),
        );
        ok("sync", m.sync());
        same_tensor(&format!("{shape:?}"), &got, &want, 1e-4, 1e-4);
    }
}

#[test]
fn broadcast_add_and_mul_match_cpu_bit_for_bit() {
    let (m, c) = (metal(), cpu());
    let cases: [(&[usize], &[usize]); 11] = [
        (&[4, 5], &[4, 5]),
        (&[4, 5], &[5]),
        (&[5], &[4, 5]),
        (&[4, 5], &[4, 1]),
        (&[4, 1], &[1, 5]),
        (&[2, 3, 4, 5], &[3, 1, 1]),
        (&[2, 3, 4, 5], &[1, 4, 5]),
        (&[1, 3, 1, 5], &[2, 1, 4, 1]),
        (&[2, 3, 4], &[1]),
        (&[2, 3], &[]),
        (&[1, 1, 1, 1, 1, 1, 1, 3], &[2, 2, 2, 2, 2, 2, 2, 1]),
    ];
    for (i, (ashape, bshape)) in cases.into_iter().enumerate() {
        let seed = 300 + 10 * i as u64;
        let (a, b) = (rand(ashape, seed, 2.0), rand(bshape, seed + 1, 2.0));
        let (am, bm) = (up(&m, &a), up(&m, &b));
        let what = format!("{ashape:?} {bshape:?}");
        let add = ok("add", m.broadcast_add_forward(&am, &bm));
        let mul = ok("mul", m.broadcast_mul_forward(&am, &bm));
        ok("sync", m.sync());
        // One IEEE operation per element on both sides: the bits agree.
        for (name, got, want) in [
            ("add", &add, ok("cpu add", c.broadcast_add_forward(&a, &b))),
            ("mul", &mul, ok("cpu mul", c.broadcast_mul_forward(&a, &b))),
        ] {
            assert_eq!(got.shape(), want.shape(), "{name} {what}");
            let (g, w) = (down(got), ok("want", want.to_f32_vec()));
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&g), bits(&w), "{name} {what}");
        }
    }
}

#[test]
fn scale_matches_cpu_bit_for_bit() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[3, 5, 7], 400, 3.0);
    let scale = 0.176_776_7f32;
    let want = ok("cpu", c.scale_forward(&x, scale));
    let got = ok("metal", m.scale_forward(&up(&m, &x), scale));
    ok("sync", m.sync());
    same_tensor("scale", &got, &want, 0.0, 0.0);
}

#[test]
fn softmax_matches_cpu() {
    let (m, c) = (metal(), cpu());
    for (i, shape) in [
        vec![5, 16],
        vec![2, 3, 49],
        vec![9],
        vec![4, 1],
        vec![3, 1000],
    ]
    .iter()
    .enumerate()
    {
        let x = rand(shape, 500 + i as u64, 8.0);
        let want = ok("cpu", c.softmax_forward(&x));
        let got = ok("metal", m.softmax_forward(&up(&m, &x)));
        ok("sync", m.sync());
        same_tensor(&format!("{shape:?}"), &got, &want, 1e-6, 1e-5);
    }
}

#[test]
fn shape_errors_are_the_cpus_and_immediate() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[2, 3, 4], 1, 1.0);
    let w3 = rand(&[3], 2, 1.0);
    let w4 = rand(&[4], 3, 1.0);
    let bad_b = rand(&[4], 4, 1.0);
    let (xm, w3m, w4m, bm) = (up(&m, &x), up(&m, &w3), up(&m, &w4), up(&m, &bad_b));
    let shape = |r: Result<_, OjasError>| matches!(r, Err(OjasError::Shape { .. }));
    assert!(shape(m.layer_norm_forward(&xm, &w3m, &w4m, EPS)));
    assert!(shape(c.layer_norm_forward(&x, &w3, &w4, EPS)));
    assert!(shape(m.channel_layer_norm_forward(&xm, &w4m, &w4m, EPS)));
    assert!(shape(
        m.broadcast_add_forward(&xm, &up(&m, &rand(&[2, 2], 5, 1.0)))
    ));
    assert!(shape(m.linear_bias_forward(
        &xm,
        &up(&m, &rand(&[2, 4], 6, 1.0)),
        &bm
    )));
}

#[test]
fn a_non_finite_operand_is_a_deferred_fault_naming_the_op() {
    let m = metal();
    let mut bad = values(2 * 8, 7, 1.0);
    bad[5] = f32::NAN;
    let x = up(&m, &host(&bad, &[2, 8]));
    let w = up(&m, &rand(&[8], 8, 1.0));
    deferred(
        &m,
        "layer_norm",
        m.layer_norm_forward(&x, &w, &w, EPS),
        "layer_norm_forward",
    );
    deferred(&m, "gelu", m.gelu_erf_forward(&x), "gelu_erf_forward");
    deferred(&m, "softmax", m.softmax_forward(&x), "softmax_forward");
    deferred(&m, "scale", m.scale_forward(&x, 2.0), "scale_forward");
    let ok_x = up(&m, &rand(&[2, 8], 9, 1.0));
    deferred(
        &m,
        "broadcast add",
        m.broadcast_add_forward(&x, &ok_x),
        "broadcast_add_forward",
    );
    deferred(
        &m,
        "broadcast mul",
        m.broadcast_mul_forward(&ok_x, &x),
        "broadcast_mul_forward",
    );
}

#[test]
fn an_overflowing_output_is_a_deferred_fault() {
    let m = metal();
    let x = up(&m, &host(&[3.0e38, 1.0], &[2]));
    deferred(&m, "scale", m.scale_forward(&x, 4.0), "scale_forward");
    deferred(
        &m,
        "broadcast mul",
        m.broadcast_mul_forward(&x, &up(&m, &host(&[4.0], &[1]))),
        "broadcast_mul_forward",
    );
}
