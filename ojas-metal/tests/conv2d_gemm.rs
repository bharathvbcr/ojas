//! `conv2d_forward` on Metal against the CPU backend: im2col, tessl's exact
//! f32 GEMM and a layout pass, for the SAM cases (the 14 x 14 stride-14
//! patch embedding without bias, 1 x 1 and 3 x 3 padded convolutions with
//! bias at the neck's 256 channels). Every other configuration is
//! `Unsupported` and names the op and the configuration; a malformed call is
//! refused as the validator refuses it; a non-finite operand is a deferred
//! fault under `conv2d_forward`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{conv2d_forward_dims, Backend, Conv2dParams, OjasError, Tensor};
use ojas_metal::MetalBackend;

const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-4;

fn params(stride: [usize; 2], padding: [usize; 2]) -> Conv2dParams {
    Conv2dParams {
        stride,
        padding,
        ..Conv2dParams::new()
    }
}

/// CPU and Metal agree on `conv2d_forward` of `x` `[n, c, h, w]` by a
/// `[o, c, kh, kw]` weight, with or without a bias.
fn check(
    m: &MetalBackend,
    what: &str,
    x: &[usize],
    w: &[usize],
    with_bias: bool,
    p: Conv2dParams,
    seed: u64,
) {
    let c = cpu();
    let x = rand(x, seed, 1.0);
    let w = rand(w, seed + 1, 0.1);
    let bias = with_bias.then(|| rand(&[w.shape()[0]], seed + 2, 1.0));
    let want = ok("cpu", c.conv2d_forward(&x, &w, bias.as_ref(), p));
    let dev_bias = bias.as_ref().map(|b| up(m, b));
    let got = ok(
        "metal",
        m.conv2d_forward(&up(m, &x), &up(m, &w), dev_bias.as_ref(), p),
    );
    ok("sync", m.sync());
    same_tensor(what, &got, &want, ATOL, RTOL);
}

#[test]
fn patch_embedding_at_the_full_sam_size_matches_cpu() {
    let m = metal();
    // [1, 3, 1008, 1008] -> [1, 1024, 72, 72], no bias.
    check(
        &m,
        "patch embed 1008",
        &[1, 3, 1008, 1008],
        &[1024, 3, 14, 14],
        false,
        params([14, 14], [0, 0]),
        1,
    );
}

#[test]
fn patch_embedding_drops_the_trailing_pixels_and_takes_any_kernel_shape() {
    let m = metal();
    // 30 x 44 under k14 s14 gives 2 x 3 windows; the last 2 and 2 columns
    // and rows are dropped, as torch drops them.
    check(
        &m,
        "30x44 k14",
        &[2, 3, 30, 44],
        &[16, 3, 14, 14],
        true,
        params([14, 14], [0, 0]),
        2,
    );
    check(
        &m,
        "k4x2 s4x2",
        &[1, 3, 9, 7],
        &[5, 3, 4, 2],
        false,
        params([4, 2], [0, 0]),
        3,
    );
    check(
        &m,
        "k14 one window",
        &[1, 1, 14, 14],
        &[1, 1, 14, 14],
        true,
        params([14, 14], [0, 0]),
        4,
    );
}

#[test]
fn one_by_one_with_bias_at_the_neck_channel_counts_matches_cpu() {
    let m = metal();
    let s1 = params([1, 1], [0, 0]);
    check(
        &m,
        "1x1 256->256 72",
        &[1, 256, 72, 72],
        &[256, 256, 1, 1],
        true,
        s1,
        10,
    );
    check(
        &m,
        "1x1 1024->256",
        &[2, 1024, 12, 12],
        &[256, 1024, 1, 1],
        true,
        s1,
        11,
    );
    check(
        &m,
        "1x1 no bias",
        &[1, 5, 4, 6],
        &[3, 5, 1, 1],
        false,
        s1,
        12,
    );
}

#[test]
fn three_by_three_padded_with_bias_at_the_neck_channel_counts_matches_cpu() {
    let m = metal();
    let p1 = params([1, 1], [1, 1]);
    check(
        &m,
        "3x3 p1 256 72",
        &[1, 256, 72, 72],
        &[256, 256, 3, 3],
        true,
        p1,
        20,
    );
    check(
        &m,
        "3x3 p1 256 odd",
        &[2, 256, 17, 23],
        &[256, 256, 3, 3],
        true,
        p1,
        21,
    );
}

#[test]
fn stride_one_at_other_kernels_and_paddings_matches_cpu() {
    let m = metal();
    let s1 = |pad: [usize; 2]| params([1, 1], pad);
    check(
        &m,
        "3x3 p0",
        &[2, 4, 9, 11],
        &[6, 4, 3, 3],
        true,
        s1([0, 0]),
        30,
    );
    check(
        &m,
        "5x5 p2",
        &[1, 3, 8, 8],
        &[4, 3, 5, 5],
        true,
        s1([2, 2]),
        31,
    );
    check(
        &m,
        "3x3 p(0,1)",
        &[1, 3, 6, 9],
        &[2, 3, 3, 3],
        false,
        s1([0, 1]),
        32,
    );
    check(
        &m,
        "3x1 p(2,0)",
        &[1, 2, 7, 5],
        &[3, 2, 3, 1],
        true,
        s1([2, 0]),
        33,
    );
    check(
        &m,
        "kernel as large as the padded image",
        &[1, 2, 3, 3],
        &[2, 2, 5, 5],
        true,
        s1([1, 1]),
        34,
    );
    check(
        &m,
        "one pixel",
        &[1, 1, 1, 1],
        &[1, 1, 1, 1],
        true,
        s1([0, 0]),
        35,
    );
}

#[test]
fn offset_device_views_give_the_cpu_answer() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[1, 3, 6, 6], 40, 1.0);
    let w = rand(&[4, 3, 3, 3], 41, 0.1);
    let b = rand(&[4], 42, 1.0);
    let p = params([1, 1], [1, 1]);
    let want = ok("cpu", c.conv2d_forward(&x, &w, Some(&b), p));
    let got = ok(
        "metal",
        m.conv2d_forward(
            &offset_view(&m, &x, 3, 50),
            &offset_view(&m, &w, 5, 51),
            Some(&offset_view(&m, &b, 7, 52)),
            p,
        ),
    );
    ok("sync", m.sync());
    same_tensor("views", &got, &want, ATOL, RTOL);
}

#[test]
fn host_operands_are_refused_for_placement_after_the_shape_check() {
    let m = metal();
    let x = rand(&[1, 2, 5, 5], 60, 1.0);
    let w = rand(&[3, 2, 3, 3], 61, 0.1);
    let r = m.conv2d_forward(&x, &w, None, params([1, 1], [1, 1]));
    assert!(
        matches!(
            r,
            Err(OjasError::Placement {
                op: "conv2d_forward",
                ..
            })
        ),
        "{r:?}"
    );
}

#[test]
fn a_configuration_the_gemm_does_not_cover_is_unsupported_and_named() {
    let m = metal();
    let x = up(&m, &rand(&[1, 4, 12, 12], 70, 1.0));
    let w3 = up(&m, &rand(&[4, 4, 3, 3], 71, 1.0));
    let w2 = up(&m, &rand(&[4, 2, 3, 3], 72, 1.0));
    let w1 = up(&m, &rand(&[4, 1, 3, 3], 73, 1.0));
    let cases: Vec<(&str, &Tensor, Conv2dParams)> = vec![
        ("stride 2 with a 3x3 kernel", &w3, params([2, 2], [1, 1])),
        ("stride 2 unpadded 3x3", &w3, params([2, 2], [0, 0])),
        ("kernel == stride with padding", &w3, params([3, 3], [1, 1])),
        ("anisotropic stride", &w3, params([1, 2], [0, 0])),
        (
            "dilation 2",
            &w3,
            Conv2dParams {
                dilation: [2, 2],
                ..Conv2dParams::new()
            },
        ),
        (
            "groups 2",
            &w2,
            Conv2dParams {
                groups: 2,
                ..Conv2dParams::new()
            },
        ),
        (
            "depthwise",
            &w1,
            Conv2dParams {
                groups: 4,
                padding: [1, 1],
                ..Conv2dParams::new()
            },
        ),
    ];
    for (what, w, p) in cases {
        // The validator accepts it: only Metal declines.
        ok(what, conv2d_forward_dims(&x, w, None, p));
        let r = m.conv2d_forward(&x, w, None, p);
        match r {
            Err(OjasError::Unsupported { op, ref detail }) => {
                assert_eq!(op, "conv2d_forward", "{what}");
                assert!(
                    detail.contains("stride") && detail.contains("groups"),
                    "{what}: the detail names the configuration: {detail}"
                );
            }
            other => panic!("{what}: expected Unsupported, got {other:?}"),
        }
        ok("sync", m.sync());
    }
}

#[test]
fn a_malformed_call_is_refused_as_the_validator_refuses_it() {
    let m = metal();
    let x = rand(&[1, 4, 8, 8], 80, 1.0);
    let w = rand(&[4, 4, 3, 3], 81, 1.0);
    let p = params([1, 1], [1, 1]);
    let cases: Vec<(&str, Tensor, Tensor, Option<Tensor>, Conv2dParams)> = vec![
        ("rank 3 input", rand(&[4, 8, 8], 1, 1.0), w.clone(), None, p),
        (
            "weight channels",
            x.clone(),
            rand(&[4, 3, 3, 3], 2, 1.0),
            None,
            p,
        ),
        (
            "bias length",
            x.clone(),
            w.clone(),
            Some(rand(&[5], 3, 1.0)),
            p,
        ),
        (
            "zero stride",
            x.clone(),
            w.clone(),
            None,
            params([0, 1], [1, 1]),
        ),
        (
            "kernel past the padded image",
            rand(&[1, 4, 2, 2], 4, 1.0),
            rand(&[4, 4, 5, 5], 5, 1.0),
            None,
            params([1, 1], [0, 0]),
        ),
        (
            "u32 input",
            host_u32(&[0; 4 * 8 * 8], &[1, 4, 8, 8]),
            w.clone(),
            None,
            p,
        ),
    ];
    for (what, x, w, bias, p) in cases {
        let want = conv2d_forward_dims(&x, &w, bias.as_ref(), p).expect_err(what);
        let host_err = m.conv2d_forward(&x, &w, bias.as_ref(), p).expect_err(what);
        assert_eq!(
            format!("{host_err:?}"),
            format!("{want:?}"),
            "{what} (host)"
        );
        if x.dtype() == ojas_core::DType::F32 {
            let dev_bias = bias.as_ref().map(|b| up(&m, b));
            let r = m.conv2d_forward(&up(&m, &x), &up(&m, &w), dev_bias.as_ref(), p);
            let err = r.expect_err(what);
            assert_eq!(format!("{err:?}"), format!("{want:?}"), "{what} (device)");
        }
        ok("sync", m.sync());
    }
}

#[test]
fn a_non_finite_operand_is_a_deferred_fault_under_the_op_name() {
    let m = metal();
    let p = params([1, 1], [1, 1]);
    let shapes: [&[usize]; 3] = [&[1, 2, 5, 5], &[3, 2, 3, 3], &[3]];
    for (which, name) in ["input", "weight", "bias"].into_iter().enumerate() {
        for bad in [f32::NAN, f32::INFINITY] {
            let mut ops: Vec<Tensor> = shapes
                .iter()
                .enumerate()
                .map(|(i, s)| rand(s, 90 + i as u64, 1.0))
                .collect();
            let mut data = ok("host", ops[which].to_f32_vec());
            data[1] = bad;
            ops[which] = host(&data, shapes[which]);
            let dev: Vec<Tensor> = ops.iter().map(|t| up(&m, t)).collect();
            let r = m.conv2d_forward(&dev[0], &dev[1], Some(&dev[2]), p);
            deferred(&m, &format!("{bad} in {name}"), r, "conv2d_forward");
        }
    }
}

#[test]
fn the_output_of_a_finite_overflow_is_a_deferred_fault_too() {
    let m = metal();
    // Finite operands whose products overflow f32: the output scan catches it.
    let x = up(&m, &host(&[3.0e38; 4], &[1, 1, 2, 2]));
    let w = up(&m, &host(&[3.0e38; 4], &[1, 1, 2, 2]));
    let r = m.conv2d_forward(&x, &w, None, Conv2dParams::new());
    deferred(&m, "overflow", r, "conv2d_forward");
}

/// The neck's larger scales (144 x 144 and 288 x 288 at 256 channels); the
/// CPU reference takes a while, so they run on request.
#[test]
#[ignore = "slow CPU reference; run with --ignored"]
fn the_neck_scales_144_and_288_match_cpu() {
    let m = metal();
    let p1 = params([1, 1], [1, 1]);
    check(
        &m,
        "3x3 p1 256 144",
        &[1, 256, 144, 144],
        &[256, 256, 3, 3],
        true,
        p1,
        100,
    );
    check(
        &m,
        "3x3 p1 256 288",
        &[1, 256, 288, 288],
        &[256, 256, 3, 3],
        true,
        p1,
        101,
    );
    check(
        &m,
        "1x1 256 288",
        &[1, 256, 288, 288],
        &[256, 256, 1, 1],
        true,
        params([1, 1], [0, 0]),
        102,
    );
}
