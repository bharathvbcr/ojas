//! Edge cases for the Metal encoder ops that `encoder_ops.rs` does not reach:
//! widths 1 and 4736, widths that are not a multiple of the 32-wide
//! simdgroup, offset (non-zero, non-16-aligned) device views, a large-mean
//! row, a masked softmax row, and `Autocast<&MetalBackend>` reaching the
//! Metal methods.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Autocast, Backend, OjasError, MAX_BROADCAST_RANK, MAX_PERMUTE_RANK};

/// `Cmd::Bcast` sizes its shape and stride arrays by MAX_PERMUTE_RANK and
/// `MetalBackend::broadcast` indexes them up to the broadcast rank, so the
/// broadcast cap may never exceed the permute cap.
const _: () = assert!(MAX_BROADCAST_RANK <= MAX_PERMUTE_RANK);

#[test]
fn layer_norm_matches_cpu_at_width_one_and_at_awkward_widths() {
    let (m, c) = (metal(), cpu());
    // 1: one lane; 31/33/63/65: either side of a simdgroup and of two;
    // 1000: not a power of two; 1024 * 4 and 4736: SAM's widths, past the
    // 1024-lane threadgroup cap so every lane strides.
    for (i, dim) in [1usize, 2, 31, 33, 63, 65, 1000, 4096, 4736]
        .into_iter()
        .enumerate()
    {
        let seed = 700 + 10 * i as u64;
        let x = rand(&[3, dim], seed, 3.0);
        let w = rand(&[dim], seed + 1, 2.0);
        let b = rand(&[dim], seed + 2, 1.0);
        let want = ok("cpu", c.layer_norm_forward(&x, &w, &b, 1e-6));
        let got = ok(
            "metal",
            m.layer_norm_forward(&up(&m, &x), &up(&m, &w), &up(&m, &b), 1e-6),
        );
        ok("sync", m.sync());
        same_tensor(&format!("dim {dim}"), &got, &want, 1e-4, 1e-4);
    }
}

#[test]
fn layer_norm_of_a_large_mean_row_matches_an_f64_reference() {
    let m = metal();
    let (dim, offset, eps) = (4736usize, 1000.0f32, 1e-5f32);
    let data: Vec<f32> = values(dim, 9, 1.7)
        .into_iter()
        .map(|x| x + offset)
        .collect();
    let (w, b) = (host(&vec![1.0; dim], &[dim]), host(&vec![0.0; dim], &[dim]));
    let got = down(&ok(
        "metal",
        m.layer_norm_forward(
            &up(&m, &host(&data, &[1, dim])),
            &up(&m, &w),
            &up(&m, &b),
            eps,
        ),
    ));
    ok("sync", m.sync());
    let x64: Vec<f64> = data.iter().map(|&x| f64::from(x)).collect();
    let mean = x64.iter().sum::<f64>() / dim as f64;
    let var = x64.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / dim as f64;
    let rstd = 1.0 / (var + f64::from(eps)).sqrt();
    let want: Vec<f32> = x64.iter().map(|x| ((x - mean) * rstd) as f32).collect();
    close("large mean", &got, &want, 1e-4, 1e-4);
}

#[test]
fn layer_norm_refuses_a_negative_eps_it_cannot_run() {
    let m = metal();
    let x = up(&m, &rand(&[2, 4], 1, 1.0));
    let w = up(&m, &rand(&[4], 2, 1.0));
    for eps in [-1e-3f32, -0.0, 0.0] {
        let r = m.layer_norm_forward(&x, &w, &w, eps);
        assert!(
            matches!(
                r,
                Err(OjasError::Unsupported {
                    op: "layer_norm_forward",
                    ..
                })
            ),
            "eps {eps}: {r:?}"
        );
    }
    let r = m.channel_layer_norm_forward(&up(&m, &rand(&[2, 4, 3], 3, 1.0)), &w, &w, 0.0);
    assert!(matches!(r, Err(OjasError::Unsupported { .. })), "{r:?}");
}

#[test]
fn offset_device_views_give_the_cpu_answer() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[4, 33], 1, 3.0);
    let w = rand(&[33], 2, 2.0);
    let b = rand(&[33], 3, 1.0);
    let xv = offset_view(&m, &x, 3, 10);
    let wv = offset_view(&m, &w, 5, 11);
    let bv = offset_view(&m, &b, 7, 12);
    let want = ok("cpu ln", c.layer_norm_forward(&x, &w, &b, 1e-6));
    let got = ok("metal ln", m.layer_norm_forward(&xv, &wv, &bv, 1e-6));
    same_tensor("layer_norm view", &got, &want, 1e-4, 1e-4);

    let want = ok("cpu gelu", c.gelu_erf_forward(&x));
    let got = ok("metal gelu", m.gelu_erf_forward(&xv));
    same_tensor("gelu view", &got, &want, 1e-6, 1e-5);

    let want = ok("cpu softmax", c.softmax_forward(&x));
    let got = ok("metal softmax", m.softmax_forward(&xv));
    same_tensor("softmax view", &got, &want, 1e-6, 1e-5);

    let want = ok("cpu scale", c.scale_forward(&x, 0.125));
    let got = ok("metal scale", m.scale_forward(&xv, 0.125));
    same_tensor("scale view", &got, &want, 0.0, 0.0);

    let want = ok("cpu add", c.broadcast_add_forward(&x, &w));
    let got = ok("metal add", m.broadcast_add_forward(&xv, &wv));
    same_tensor("broadcast view", &got, &want, 0.0, 0.0);

    let img = rand(&[2, 6, 5], 4, 2.0);
    let (cw, cb) = (rand(&[6], 5, 1.0), rand(&[6], 6, 1.0));
    let want = ok(
        "cpu chan",
        c.channel_layer_norm_forward(&img, &cw, &cb, 1e-6),
    );
    let got = ok(
        "metal chan",
        m.channel_layer_norm_forward(
            &offset_view(&m, &img, 3, 13),
            &offset_view(&m, &cw, 1, 14),
            &offset_view(&m, &cb, 5, 15),
            1e-6,
        ),
    );
    ok("sync", m.sync());
    same_tensor("channel view", &got, &want, 1e-4, 1e-4);
}

#[test]
fn softmax_matches_cpu_on_large_logits_and_wide_rows() {
    let (m, c) = (metal(), cpu());
    // Scores of 80..90 overflow a naive exp; the row maximum is subtracted.
    let mut big = values(6 * 64, 21, 5.0);
    for (i, x) in big.iter_mut().enumerate() {
        *x += 85.0 + (i % 7) as f32;
    }
    for (i, (shape, data)) in [
        (vec![6, 64], big),
        (vec![2, 4736], values(2 * 4736, 22, 9.0)),
        (vec![3, 1], values(3, 23, 1.0)),
    ]
    .into_iter()
    .enumerate()
    {
        let x = host(&data, &shape);
        let want = ok("cpu", c.softmax_forward(&x));
        let got = ok("metal", m.softmax_forward(&up(&m, &x)));
        ok("sync", m.sync());
        same_tensor(&format!("case {i}"), &got, &want, 1e-6, 1e-5);
    }
}

#[test]
fn softmax_of_a_masked_row_is_a_deferred_fault_like_the_cpus_refusal() {
    let m = metal();
    let mut data = values(2 * 8, 31, 1.0);
    data[3] = f32::NEG_INFINITY;
    let x = up(&m, &host(&data, &[2, 8]));
    deferred(
        &m,
        "masked softmax",
        m.softmax_forward(&x),
        "softmax_forward",
    );
    let all_masked = up(&m, &host(&[f32::NEG_INFINITY; 4], &[1, 4]));
    deferred(
        &m,
        "all -inf",
        m.softmax_forward(&all_masked),
        "softmax_forward",
    );
}

#[test]
fn autocast_over_metal_reaches_the_metal_methods_not_the_refusing_default() {
    let m = metal();
    let a = Autocast::new(&m);
    let x = up(&m, &rand(&[2, 8], 41, 1.0));
    let w = up(&m, &rand(&[8], 42, 1.0));
    let wt = up(&m, &rand(&[3, 8], 43, 1.0));
    let bias3 = up(&m, &rand(&[3], 44, 1.0));
    let chan = up(&m, &rand(&[2, 8, 2], 45, 1.0));
    let unsupported = |op: &str, r: Result<_, OjasError>| {
        let r: Result<ojas_core::Tensor, OjasError> = r;
        assert!(
            !matches!(r, Err(OjasError::Unsupported { .. })),
            "{op} fell to a refusing default: {r:?}"
        );
        ok(op, r);
    };
    unsupported("layer_norm", a.layer_norm_forward(&x, &w, &w, 1e-6));
    unsupported(
        "channel_layer_norm",
        a.channel_layer_norm_forward(&chan, &w, &w, 1e-6),
    );
    unsupported("gelu", a.gelu_erf_forward(&x));
    unsupported("linear_bias", a.linear_bias_forward(&x, &wt, &bias3));
    unsupported("bcast add", a.broadcast_add_forward(&x, &w));
    unsupported("bcast mul", a.broadcast_mul_forward(&x, &w));
    unsupported("scale", a.scale_forward(&x, 0.5));
    unsupported("softmax", a.softmax_forward(&x));
    ok("sync", m.sync());
}
