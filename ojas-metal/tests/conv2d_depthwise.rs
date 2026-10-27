//! Depthwise `conv2d_forward` (`groups == in == out`, one filter per
//! channel) on Metal's direct kernel `ojas_dwconv` against the CPU backend:
//! the SAM memory encoder's 7 x 7 padded depthwise convolution at 256
//! channels over 72 x 72, strides, odd sizes and a padding at and past the
//! kernel edge, with and without bias. The kernel takes the taps in the CPU's
//! order and adds the bias last, so parity is bitwise. Other grouped
//! convolutions and dilation stay `Unsupported`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{conv2d_forward_dims, Backend, Conv2dParams, OjasError, Tensor};
use ojas_metal::MetalBackend;

/// Bound for the one case that is not on the direct kernel (see `check`).
const GEMM_ATOL: f32 = 1e-6;
const GEMM_RTOL: f32 = 1e-6;

fn depthwise(c: usize, stride: [usize; 2], padding: [usize; 2]) -> Conv2dParams {
    Conv2dParams {
        stride,
        padding,
        groups: c,
        ..Conv2dParams::new()
    }
}

/// CPU and Metal agree on a depthwise convolution of `x` `[n, c, h, w]` by a
/// `[c, 1, kh, kw]` weight.
fn check(
    m: &MetalBackend,
    what: &str,
    x: &[usize],
    k: [usize; 2],
    with_bias: bool,
    (stride, padding): ([usize; 2], [usize; 2]),
    seed: u64,
) {
    let c = x[1];
    let (cpu, p) = (cpu(), depthwise(c, stride, padding));
    let x = rand(x, seed, 1.0);
    let w = rand(&[c, 1, k[0], k[1]], seed + 1, 0.3);
    let bias = with_bias.then(|| rand(&[c], seed + 2, 1.0));
    let want = ok("cpu", cpu.conv2d_forward(&x, &w, bias.as_ref(), p));
    let dev_bias = bias.as_ref().map(|b| up(m, b));
    let got = ok(
        "metal",
        m.conv2d_forward(&up(m, &x), &up(m, &w), dev_bias.as_ref(), p),
    );
    ok("sync", m.sync());
    if c > 1 {
        same_bits(what, &got, &want);
    } else {
        // One channel is `groups == 1`: the im2col GEMM path, which sums in
        // another order than the CPU's direct loop, so not bitwise.
        same_tensor(what, &got, &want, GEMM_ATOL, GEMM_RTOL);
    }
}

#[test]
fn the_memory_encoder_depthwise_7x7_at_256_channels_matches_cpu() {
    let m = metal();
    check(
        &m,
        "dw 7x7 p3 256 72",
        &[1, 256, 72, 72],
        [7, 7],
        true,
        ([1, 1], [3, 3]),
        1,
    );
    check(
        &m,
        "dw 7x7 p3 256 no bias",
        &[1, 256, 72, 72],
        [7, 7],
        false,
        ([1, 1], [3, 3]),
        2,
    );
}

#[test]
fn odd_sizes_strides_and_edge_paddings_match_cpu() {
    let m = metal();
    check(
        &m,
        "dw 3x3 p1",
        &[2, 5, 9, 11],
        [3, 3],
        true,
        ([1, 1], [1, 1]),
        10,
    );
    check(
        &m,
        "dw 3x3 s2 p1 odd",
        &[1, 4, 13, 8],
        [3, 3],
        true,
        ([2, 2], [1, 1]),
        11,
    );
    check(
        &m,
        "dw 7x7 p0",
        &[1, 3, 10, 10],
        [7, 7],
        false,
        ([1, 1], [0, 0]),
        12,
    );
    check(
        &m,
        "dw 3x5 s(1,2) p(1,2)",
        &[1, 3, 7, 12],
        [3, 5],
        true,
        ([1, 2], [1, 2]),
        13,
    );
    check(
        &m,
        "dw 1x1",
        &[1, 6, 5, 5],
        [1, 1],
        true,
        ([1, 1], [0, 0]),
        14,
    );
    // Padding at the kernel edge and past it: some windows see only zeros.
    check(
        &m,
        "dw 3x3 p3",
        &[1, 2, 4, 4],
        [3, 3],
        true,
        ([1, 1], [3, 3]),
        15,
    );
    check(
        &m,
        "dw 3x3 p5 s2",
        &[1, 2, 4, 5],
        [3, 3],
        true,
        ([2, 2], [5, 5]),
        16,
    );
    check(
        &m,
        "dw 7x7 p7",
        &[1, 2, 3, 3],
        [7, 7],
        false,
        ([1, 1], [7, 7]),
        17,
    );
    check(
        &m,
        "dw one pixel",
        &[1, 1, 1, 1],
        [3, 3],
        true,
        ([1, 1], [1, 1]),
        18,
    );
    check(
        &m,
        "dw a single channel, groups 1",
        &[1, 1, 6, 6],
        [3, 3],
        true,
        ([1, 1], [1, 1]),
        19,
    );
}

#[test]
fn offset_device_views_give_the_cpu_answer() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[1, 4, 6, 6], 40, 1.0);
    let w = rand(&[4, 1, 3, 3], 41, 0.3);
    let b = rand(&[4], 42, 1.0);
    let p = depthwise(4, [1, 1], [1, 1]);
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
    same_bits("views", &got, &want);
}

#[test]
fn other_grouped_convolutions_and_dilation_stay_unsupported_and_named() {
    let m = metal();
    let x = up(&m, &rand(&[1, 4, 8, 8], 60, 1.0));
    let w_g2 = up(&m, &rand(&[4, 2, 3, 3], 61, 1.0));
    let w_mult = up(&m, &rand(&[8, 1, 3, 3], 62, 1.0));
    let w_dw = up(&m, &rand(&[4, 1, 3, 3], 63, 1.0));
    let cases: Vec<(&str, &Tensor, Conv2dParams)> = vec![
        (
            "groups 2 of 4",
            &w_g2,
            Conv2dParams {
                groups: 2,
                ..Conv2dParams::new()
            },
        ),
        // Channel multiplier 2: 4 groups, 8 output channels.
        (
            "depthwise multiplier 2",
            &w_mult,
            depthwise(4, [1, 1], [1, 1]),
        ),
        (
            "dilated depthwise",
            &w_dw,
            Conv2dParams {
                dilation: [2, 2],
                ..depthwise(4, [1, 1], [2, 2])
            },
        ),
    ];
    for (what, w, p) in cases {
        ok(what, conv2d_forward_dims(&x, w, None, p));
        match m.conv2d_forward(&x, w, None, p) {
            Err(OjasError::Unsupported { op, ref detail }) => {
                assert_eq!(op, "conv2d_forward", "{what}");
                assert!(
                    detail.contains("groups") && detail.contains("dilation"),
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
    let x = rand(&[1, 4, 8, 8], 70, 1.0);
    let w = rand(&[4, 1, 3, 3], 71, 1.0);
    let p = depthwise(4, [1, 1], [1, 1]);
    let cases: Vec<(&str, Tensor, Tensor, Option<Tensor>, Conv2dParams)> = vec![
        (
            "weight channels",
            x.clone(),
            rand(&[4, 2, 3, 3], 1, 1.0),
            None,
            p,
        ),
        (
            "bias length",
            x.clone(),
            w.clone(),
            Some(rand(&[3], 2, 1.0)),
            p,
        ),
        (
            "groups not dividing",
            x.clone(),
            w.clone(),
            None,
            depthwise(3, [1, 1], [1, 1]),
        ),
        (
            "zero stride",
            x.clone(),
            w.clone(),
            None,
            depthwise(4, [0, 1], [1, 1]),
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
        let dev_bias = bias.as_ref().map(|b| up(&m, b));
        let r = m.conv2d_forward(&up(&m, &x), &up(&m, &w), dev_bias.as_ref(), p);
        let err = r.expect_err(what);
        assert_eq!(format!("{err:?}"), format!("{want:?}"), "{what} (device)");
        ok("sync", m.sync());
    }
}

#[test]
fn a_non_finite_operand_is_a_deferred_fault_under_the_op_name() {
    let m = metal();
    let p = depthwise(2, [1, 1], [1, 1]);
    let shapes: [&[usize]; 3] = [&[1, 2, 5, 5], &[2, 1, 3, 3], &[2]];
    for (which, name) in ["input", "weight", "bias"].into_iter().enumerate() {
        for bad in [f32::NAN, f32::INFINITY] {
            let mut ops: Vec<Tensor> = shapes
                .iter()
                .enumerate()
                .map(|(i, s)| rand(s, 80 + i as u64, 1.0))
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
