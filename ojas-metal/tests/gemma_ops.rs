//! Integration tests for Gemma Embedding kernels on Metal:
//! `gelu_tanh_forward`, `gelu_tanh_backward`, `seq_mean_forward`, and
//! `l2_normalize_forward`, comparing against the CPU reference and verifying
//! error contracts.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Backend, OjasError};

#[test]
fn gelu_tanh_forward_and_backward_match_cpu() {
    let m = metal();
    let c = cpu();

    let mut xs = Vec::new();
    let mut x = -6.0f32;
    while x <= 6.0 {
        xs.push(x);
        x += 0.125;
    }
    let n = xs.len();

    let host_x = host(&xs, &[n]);
    let dev_x = up(&m, &host_x);

    let want_fwd = ok("cpu fwd", c.gelu_tanh_forward(&host_x));
    let got_fwd = ok("metal fwd", m.gelu_tanh_forward(&dev_x));
    same_tensor("gelu_tanh_forward", &got_fwd, &want_fwd, 1e-4, 1e-4);

    let gy = rand(&[n], 42, 1.0);
    let dev_gy = up(&m, &gy);

    let want_bwd = ok("cpu bwd", c.gelu_tanh_backward(&host_x, &gy));
    let got_bwd = ok("metal bwd", m.gelu_tanh_backward(&dev_x, &dev_gy));
    same_tensor("gelu_tanh_backward", &got_bwd, &want_bwd, 1e-4, 1e-4);
}

#[test]
fn seq_mean_forward_matches_cpu() {
    let m = metal();
    let c = cpu();

    // B = 3, T = 8, D = 16
    let b = 3;
    let t = 8;
    let d = 16;
    let x = rand(&[b, t, d], 1234, 1.0);
    let dev_x = up(&m, &x);

    let lens_data = vec![5u32, 8u32, 2u32];
    let lens = host_u32(&lens_data, &[b]);
    let dev_lens = up(&m, &lens);

    let want = ok("cpu seq_mean", c.seq_mean_forward(&x, &lens));
    let got = ok("metal seq_mean", m.seq_mean_forward(&dev_x, &dev_lens));
    same_tensor("seq_mean_forward", &got, &want, 1e-4, 1e-4);

    // Edge case: zero length sentence
    let zero_lens_data = vec![0u32, 4u32, 0u32];
    let zero_lens = host_u32(&zero_lens_data, &[b]);
    let dev_zero_lens = up(&m, &zero_lens);

    let want_zero = ok("cpu seq_mean zero", c.seq_mean_forward(&x, &zero_lens));
    let got_zero = ok(
        "metal seq_mean zero",
        m.seq_mean_forward(&dev_x, &dev_zero_lens),
    );
    same_tensor(
        "seq_mean_forward zero lens",
        &got_zero,
        &want_zero,
        1e-4,
        1e-4,
    );
}

#[test]
fn l2_normalize_forward_matches_cpu() {
    let m = metal();
    let c = cpu();

    let rows = 4;
    let dim = 32;
    let x = rand(&[rows, dim], 999, 2.0);
    let dev_x = up(&m, &x);

    let eps = 1e-12;
    let want = ok("cpu l2_norm", c.l2_normalize_forward(&x, eps));
    let got = ok("metal l2_norm", m.l2_normalize_forward(&dev_x, eps));
    same_tensor("l2_normalize_forward", &got, &want, 1e-4, 1e-4);

    // Zero row normalization produces zero row without NaN
    let zero_data = vec![0.0f32; dim];
    let zero_x = host(&zero_data, &[1, dim]);
    let dev_zero_x = up(&m, &zero_x);
    let got_zero = ok(
        "metal l2_norm zero",
        m.l2_normalize_forward(&dev_zero_x, eps),
    );
    let want_zero = ok("cpu l2_norm zero", c.l2_normalize_forward(&zero_x, eps));
    same_tensor(
        "l2_normalize_forward zero row",
        &got_zero,
        &want_zero,
        1e-6,
        1e-6,
    );

    // Unsupported epsilon on metal
    let bad_eps_result = m.l2_normalize_forward(&dev_x, 1e-5);
    assert!(
        matches!(bad_eps_result, Err(OjasError::Unsupported { .. })),
        "expected Unsupported for non-1e-12 eps on Metal, got {bad_eps_result:?}"
    );
}

#[test]
fn rms_qkv_norm_forward_matches_cpu() {
    let (m, c) = (metal(), cpu());
    let q = rand(&[2, 4, 16, 64], 42, 2.0);
    let k = rand(&[2, 4, 16, 64], 43, 2.0);
    let v = rand(&[2, 4, 16, 64], 44, 2.0);
    let qw = rand(&[64], 45, 1.0);
    let kw = rand(&[64], 46, 1.0);
    let eps = 1e-6f32;

    let dev_q = up(&m, &q);
    let dev_k = up(&m, &k);
    let dev_v = up(&m, &v);
    let dev_qw = up(&m, &qw);
    let dev_kw = up(&m, &kw);

    let (got_q, got_k, got_v) = ok(
        "metal rms_qkv",
        m.rms_qkv_norm_forward(&dev_q, &dev_k, &dev_v, Some(&dev_qw), Some(&dev_kw), eps),
    );
    let (want_q, want_k, want_v) = ok(
        "cpu rms_qkv",
        c.rms_qkv_norm_forward(&q, &k, &v, Some(&qw), Some(&kw), eps),
    );

    same_tensor("rms_qkv_norm_forward q", &got_q, &want_q, 1e-5, 1e-5);
    same_tensor("rms_qkv_norm_forward k", &got_k, &want_k, 1e-5, 1e-5);
    same_tensor("rms_qkv_norm_forward v", &got_v, &want_v, 1e-5, 1e-5);

    // Also test weightless Q and K
    let (got_qw, got_kw, got_vw) = ok(
        "metal rms_qkv weightless",
        m.rms_qkv_norm_forward(&dev_q, &dev_k, &dev_v, None, None, eps),
    );
    let (want_qw, want_kw, want_vw) = ok(
        "cpu rms_qkv weightless",
        c.rms_qkv_norm_forward(&q, &k, &v, None, None, eps),
    );

    same_tensor(
        "rms_qkv_norm_forward weightless q",
        &got_qw,
        &want_qw,
        1e-5,
        1e-5,
    );
    same_tensor(
        "rms_qkv_norm_forward weightless k",
        &got_kw,
        &want_kw,
        1e-5,
        1e-5,
    );
    same_tensor(
        "rms_qkv_norm_forward weightless v",
        &got_vw,
        &want_vw,
        1e-5,
        1e-5,
    );
}
