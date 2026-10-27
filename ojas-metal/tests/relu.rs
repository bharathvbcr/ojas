//! `relu_forward` on Metal (`ojas_relu_fwd`) against the CPU backend: bit
//! for bit (every output is an input value or zero, so nothing rounds), at
//! the detector's FFN shapes, with signed zeros, subnormals and the extremes,
//! on offset device views; refusals equal the validator's; a non-finite input
//! is a deferred fault under `relu_forward`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{relu_forward_dims, Autocast, Backend, Tensor};

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn same_bits(what: &str, got: &Tensor, want: &Tensor) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let (got, want) = (down(got), ok(what, want.to_f32_vec()));
    assert_eq!(bits(&got), bits(&want), "{what}");
    eprintln!("MAXERR {what}: 0e0 (bitwise)");
}

#[test]
fn detector_ffn_shapes_match_cpu_bitwise() {
    let (m, c) = (metal(), cpu());
    for (i, shape) in [vec![5184, 2048], vec![201, 2048], vec![1], vec![3, 5, 7]]
        .into_iter()
        .enumerate()
    {
        let x = rand(&shape, 10 + i as u64, 3.0);
        let want = ok("cpu", c.relu_forward(&x));
        let got = ok("metal", m.relu_forward(&up(&m, &x)));
        ok("sync", m.sync());
        same_bits(&format!("{shape:?}"), &got, &want);
    }
}

/// What the CPU returns, bit for bit (`x < 0 ? 0 : x`, ojas-cpu
/// `relu_forward`): `-0.0` stays `-0.0` (it is not below zero), a negative
/// subnormal becomes `+0`, and every other negative becomes `+0`. A NaN or
/// an infinity never reaches that code: the CPU refuses it with `NonFinite`,
/// and Metal records the same refusal as a deferred fault (the test below).
///
/// The GPU's float compare treats a negative subnormal as zero, so a kernel
/// written `v < 0.0f ? 0.0f : v` passes `-1e-45` through: this test fails on
/// that kernel (it did, at index 262 of the list below, `0x80000001` for the
/// CPU's `0`).
#[test]
fn signed_zeros_subnormals_and_extremes_match_cpu_bitwise() {
    let (m, c) = (metal(), cpu());
    let specials = [
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::from_bits(0x007f_ffff),
        -f32::from_bits(0x007f_ffff),
        f32::MAX,
        f32::MIN,
        1e-30,
        -1e-30,
    ];
    let mut data = values(257, 3, 4.0);
    data.extend(specials);
    let x = host(&data, &[data.len()]);
    let want = ok("cpu", c.relu_forward(&x));
    let got = ok("metal", m.relu_forward(&up(&m, &x)));
    ok("sync", m.sync());
    same_bits("special values", &got, &want);
    // Spelled out, so a CPU change shows here too.
    let got = down(&got);
    let at = |i: usize| got[257 + i].to_bits();
    assert_eq!(at(1), (-0.0f32).to_bits(), "-0.0 stays -0.0");
    assert_eq!(at(3), 0, "a negative normal minimum becomes +0");
    assert_eq!(at(5), 0, "a negative subnormal becomes +0");
    assert_eq!(at(7), 0, "the largest negative subnormal becomes +0");
    assert_eq!(at(4), 1, "a positive subnormal is kept");
    assert_eq!(at(9), 0, "f32::MIN becomes +0");
    assert_eq!(at(10), 1e-30f32.to_bits(), "a small positive is kept");
    assert_eq!(at(11), 0, "a small negative becomes +0");
}

#[test]
fn offset_device_views_give_the_cpu_answer() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[4, 33], 20, 2.0);
    let want = ok("cpu", c.relu_forward(&x));
    let got = ok("metal", m.relu_forward(&offset_view(&m, &x, 3, 21)));
    ok("sync", m.sync());
    same_bits("offset view", &got, &want);
}

#[test]
fn a_malformed_call_is_refused_as_the_validator_refuses_it() {
    let m = metal();
    let cases: Vec<(&str, Tensor)> = vec![
        ("empty", host(&[], &[0, 3])),
        ("u32", host_u32(&[1, 2], &[2])),
    ];
    for (what, x) in cases {
        let want = relu_forward_dims(&x).expect_err(what);
        let err = m.relu_forward(&x).expect_err(what);
        assert_eq!(format!("{err:?}"), format!("{want:?}"), "{what} (host)");
        ok("sync", m.sync());
    }
}

#[test]
fn a_non_finite_input_is_a_deferred_fault_under_the_op_name() {
    let m = metal();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut data = values(64, 5, 1.0);
        data[9] = bad;
        // The CPU refuses it at the call; Metal records the same refusal.
        let r = cpu().relu_forward(&host(&data, &[8, 8]));
        assert!(
            matches!(
                r,
                Err(ojas_core::OjasError::NonFinite { op: "relu_forward" })
            ),
            "{bad}: {r:?}"
        );
        let x = up(&m, &host(&data, &[8, 8]));
        deferred(&m, &format!("{bad}"), m.relu_forward(&x), "relu_forward");
    }
}

#[test]
fn autocast_over_metal_reaches_the_metal_op() {
    let m = metal();
    let a = Autocast::new(&m);
    let x = up(&m, &rand(&[2, 8], 30, 1.0));
    let r = a.relu_forward(&x);
    assert!(
        !matches!(r, Err(ojas_core::OjasError::Unsupported { .. })),
        "relu fell to a refusing default: {r:?}"
    );
    ok("relu", r);
    ok("sync", m.sync());
}
