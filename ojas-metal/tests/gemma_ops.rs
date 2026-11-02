//! Integration tests for Gemma Embedding kernels on Metal:
//! `gelu_tanh_forward`, `gelu_tanh_backward`, `seq_mean_forward`,
//! `seq_mean_backward` and `l2_normalize_forward`, comparing against the CPU
//! reference and verifying error contracts. Where a new kernel answers an op
//! the dispatch count is held too, so a parity pass cannot come from a path
//! that never ran.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Backend, OjasError};
use ojas_metal::MetalBackend;

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

/// Dispatches one call adds, so a parity pass cannot come from a path that
/// never ran: `ojas_seq_mean_bwd` plus the gradient's finite scan and the
/// output's.
fn dispatches_of(m: &MetalBackend, f: &dyn Fn()) -> u64 {
    ok("sync", m.sync());
    let before = m.dispatches();
    f();
    ok("sync", m.sync());
    m.dispatches() - before
}

#[test]
fn seq_mean_backward_matches_cpu_and_runs_its_own_kernel() {
    let m = metal();
    let c = cpu();

    // B = 3, T = 8, D = 16, with a length over T, a padded one and a zero.
    let (b, t, d) = (3usize, 8usize, 16usize);
    let x = rand(&[b, t, d], 1235, 1.0);
    let dev_x = up(&m, &x);
    let grad = rand(&[b, d], 1236, 1.0);
    let dev_grad = up(&m, &grad);

    for lens_data in [vec![5u32, 8, 2], vec![99u32, 0, 1]] {
        let lens = host_u32(&lens_data, &[b]);
        let dev_lens = up(&m, &lens);
        let want = ok("cpu seq_mean_bwd", c.seq_mean_backward(&x, &lens, &grad));
        let got = ok(
            "metal seq_mean_bwd",
            m.seq_mean_backward(&dev_x, &dev_lens, &dev_grad),
        );
        same_tensor(
            &format!("seq_mean_backward lens {lens_data:?}"),
            &got,
            &want,
            1e-6,
            1e-6,
        );
        // The padding rows are exactly zero on the device too, not merely
        // within tolerance.
        let got = down(&got);
        for (bi, &len) in lens_data.iter().enumerate() {
            for row in (len as usize).min(t)..t {
                for col in 0..d {
                    let v = got[(bi * t + row) * d + col];
                    assert_eq!(v.to_bits(), 0.0f32.to_bits(), "b {bi} row {row}: {v}");
                }
            }
        }
    }

    // The op ran on the device: the gradient's finite scan and its own
    // kernel, which flags its own output.
    let lens = up(&m, &host_u32(&[5u32, 8, 2], &[b]));
    let n = dispatches_of(&m, &|| {
        drop(ok(
            "metal seq_mean_bwd",
            m.seq_mean_backward(&dev_x, &lens, &dev_grad),
        ))
    });
    assert_eq!(n, 2, "seq_mean_backward dispatches");

    // A non-finite gradient is a fault, as it is on the CPU; a non-finite
    // value in x is not read by either backend.
    let mut bad = grad.to_f32_vec().expect("grad values");
    bad[d] = f32::NAN;
    let dev_bad = up(&m, &host(&bad, &[b, d]));
    let r = m.seq_mean_backward(&dev_x, &lens, &dev_bad);
    let r = r.and_then(|out| m.sync().map(|()| out));
    assert!(
        matches!(r, Err(OjasError::NonFinite { .. })),
        "expected NonFinite for a NaN gradient, got {r:?}"
    );
}

/// A NaN past a sequence's length is refused by both backends: the mean never
/// reads those rows, but neither accepts an operand that holds one. (The host
/// module's note that the CPU passes such an operand was wrong; the CPU's
/// `f32_operands` scans the whole tensor.)
#[test]
fn a_non_finite_value_in_padding_is_refused_by_both_backends() {
    let m = metal();
    let c = cpu();
    let (b, t, d) = (2usize, 3usize, 4usize);
    let lens_data = [3u32, 1];
    let lens = host_u32(&lens_data, &[b]);
    let dev_lens = up(&m, &lens);
    for bad in [f32::NAN, f32::INFINITY] {
        // Row 2 of sequence 1, which has length 1: padding.
        let mut data = vec![0.5f32; b * t * d];
        data[(t + 2) * d] = bad;
        let host_x = host(&data, &[b, t, d]);
        let dev_x = up(&m, &host_x);
        let on_cpu = c.seq_mean_forward(&host_x, &lens);
        assert!(
            matches!(on_cpu, Err(OjasError::NonFinite { .. })),
            "cpu accepted {bad} in padding: {on_cpu:?}"
        );
        let on_metal = m
            .seq_mean_forward(&dev_x, &dev_lens)
            .and_then(|out| m.sync().map(|()| out));
        assert!(
            matches!(on_metal, Err(OjasError::NonFinite { .. })),
            "metal accepted {bad} in padding: {on_metal:?}"
        );
    }
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
fn l2_normalize_backward_matches_cpu_and_runs_its_own_kernel() {
    let m = metal();
    let c = cpu();

    let (rows, dim) = (6usize, 32usize);
    let mut data = rand(&[rows, dim], 998, 2.0).to_f32_vec().expect("values");
    // A zero row and a row far below eps = 1e-12 take the clamp branch; the
    // rest are ordinary rows well above it.
    data[4 * dim..5 * dim].fill(0.0);
    data[5 * dim..6 * dim].fill(1e-20);
    let x = host(&data, &[rows, dim]);
    let dev_x = up(&m, &x);
    let grad = rand(&[rows, dim], 997, 1.0);
    let dev_grad = up(&m, &grad);

    let eps = 1e-12f32;
    let want = ok("cpu l2_norm_bwd", c.l2_normalize_backward(&x, &grad, eps));
    let got = ok(
        "metal l2_norm_bwd",
        m.l2_normalize_backward(&dev_x, &dev_grad, eps),
    );
    // The clamped rows scale by 1e12, so the tolerance is relative.
    same_tensor("l2_normalize_backward", &got, &want, 1e-4, 1e-5);

    // The clamp branch on the device: a zero row takes grad / eps, not zero
    // and not a NaN from 0 / 0.
    let got_v = down(&got);
    let g = grad.to_f32_vec().expect("grad values");
    for col in 0..dim {
        let i = 4 * dim + col;
        let want_c = g[i] / eps;
        assert!(
            (got_v[i] - want_c).abs() <= 1e-5 * want_c.abs(),
            "zero row col {col}: {} != {want_c}",
            got_v[i]
        );
    }

    // The op ran on the device: two finite scans and its own kernel.
    let n = dispatches_of(&m, &|| {
        drop(ok(
            "metal l2_norm_bwd",
            m.l2_normalize_backward(&dev_x, &dev_grad, eps),
        ))
    });
    assert_eq!(n, 3, "l2_normalize_backward dispatches");

    // The backward is routed for the forward's epsilon only, so the pair
    // agrees on what it normalizes.
    let bad = m.l2_normalize_backward(&dev_x, &dev_grad, 1e-5);
    assert!(
        matches!(bad, Err(OjasError::Unsupported { .. })),
        "expected Unsupported for non-1e-12 eps on Metal, got {bad:?}"
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
