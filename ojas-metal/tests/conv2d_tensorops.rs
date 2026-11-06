//! `conv2d_forward` on tessl's matrix-unit convolution (`Conv2dTensorOps`): a
//! covered convolution (3x3, stride 1, equal padding, output channels a
//! multiple of 16) runs no `ojas_im2col` and no `ojas_conv_nchw` and matches the
//! CPU backend; anything else keeps the im2col route, bit for bit what `Off`
//! computes. What ran is read from `MetalBackend::kernel_dispatches` (Ojas's
//! own kernels; tessl's are not counted).

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{BackendCore, BackendModule, Budget, Conv2dParams, OjasError, Tensor};
use ojas_metal::{Conv2dTensorOps, MetalBackend};

const IM2COL: &str = "ojas_im2col";
const LAYOUT: &str = "ojas_conv_nchw";

/// `op`'s result and how many `ojas_im2col` and `ojas_conv_nchw` it ran.
fn counted(m: &MetalBackend, op: impl FnOnce() -> Result<Tensor, OjasError>) -> (Tensor, u64, u64) {
    ok("sync", m.sync());
    let (i0, l0) = (m.kernel_dispatches(IM2COL), m.kernel_dispatches(LAYOUT));
    let out = ok("counted op", op());
    ok("sync", m.sync());
    (
        out,
        m.kernel_dispatches(IM2COL) - i0,
        m.kernel_dispatches(LAYOUT) - l0,
    )
}

fn params(pad: usize, stride: usize) -> Conv2dParams {
    Conv2dParams {
        stride: [stride, stride],
        padding: [pad, pad],
        ..Conv2dParams::new()
    }
}

/// Relative L2 distance of `got` from `want`.
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut e, mut n) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        assert!(g.is_finite(), "non-finite output {g}");
        e += (*g as f64 - *w as f64).powi(2);
        n += (*w as f64).powi(2);
    }
    (e / n).sqrt()
}

#[test]
fn covered_convolutions_take_the_op_and_match_the_cpu() {
    let m = metal();
    let c = cpu();
    // [n, c_in, h, w], c_out, padding, with bias: SAM's 256-channel tail conv
    // at a width that is not a whole number of the op's 32-wide tiles, a
    // narrow output, and padding 0.
    for (i, (shape, o, pad, with_bias)) in [
        ([1usize, 256, 40, 40], 64usize, 1usize, true),
        ([2, 32, 19, 45], 16, 1, false),
        ([1, 64, 24, 30], 48, 0, true),
    ]
    .into_iter()
    .enumerate()
    {
        let seed = 40 + i as u64 * 3;
        let x = rand(&shape, seed, 1.0);
        let w = rand(&[o, shape[1], 3, 3], seed + 1, 0.05);
        let b = rand(&[o], seed + 2, 1.0);
        let bias = with_bias.then_some(&b);
        let want = down(&ok(
            "cpu conv",
            c.conv2d_forward(&x, &w, bias, params(pad, 1)),
        ));
        let (xm, wm, bm) = (up(&m, &x), up(&m, &w), up(&m, &b));
        let bias_m = with_bias.then_some(&bm);
        for (mode, tol) in [(Conv2dTensorOps::F32, 1e-5), (Conv2dTensorOps::F16, 2e-3)] {
            m.set_conv2d_tensorops(mode);
            let (y, im2col, layout) =
                counted(&m, || m.conv2d_forward(&xm, &wm, bias_m, params(pad, 1)));
            assert_eq!(
                (im2col, layout),
                (0, 0),
                "{mode:?} {shape:?}: the im2col route ran"
            );
            let err = rel_l2(&down(&y), &want);
            assert!(
                err < tol,
                "{mode:?} {shape:?} -> {o}: relative L2 {err:.3e} over {tol:e}"
            );
        }
        m.set_conv2d_tensorops(Conv2dTensorOps::Off);
        let (_, im2col, _) = counted(&m, || m.conv2d_forward(&xm, &wm, bias_m, params(pad, 1)));
        assert_eq!(im2col, 1, "Off still takes im2col");
    }
}

#[test]
fn uncovered_convolutions_keep_the_im2col_bits() {
    let m = metal();
    // Stride 2, a 5x5 kernel, and 24 output channels: none is covered.
    for (i, (shape, o, k, stride)) in [
        ([1usize, 16, 20, 20], 32usize, 3usize, 2usize),
        ([1, 16, 20, 20], 32, 5, 1),
        ([1, 16, 20, 20], 24, 3, 1),
    ]
    .into_iter()
    .enumerate()
    {
        let x = up(&m, &rand(&shape, 70 + i as u64, 1.0));
        let w = up(&m, &rand(&[o, shape[1], k, k], 71 + i as u64, 0.05));
        m.set_conv2d_tensorops(Conv2dTensorOps::Off);
        let (want, _, _) = counted(&m, || m.conv2d_forward(&x, &w, None, params(1, stride)));
        m.set_conv2d_tensorops(Conv2dTensorOps::F16);
        let (got, im2col, _) = counted(&m, || m.conv2d_forward(&x, &w, None, params(1, stride)));
        assert_eq!(
            im2col, 1,
            "{shape:?} k{k} s{stride} -> {o}: should keep im2col"
        );
        assert!(
            bits(&down(&got)) == bits(&down(&want)),
            "{shape:?} k{k} s{stride} -> {o}: bits differ from Off"
        );
    }
    m.set_conv2d_tensorops(Conv2dTensorOps::Off);
}

/// The op's scratch is charged, not the im2col route's: the op pads the
/// output width to whole 32-wide tiles, so for a one-column output its padded
/// input copy is several times the im2col matrix. A budget that holds the
/// im2col route's whole charge (output, matrix, product and a weight copy)
/// refuses the op; one that holds the op's output, padded input and weights
/// runs it.
#[test]
fn the_op_charges_its_own_scratch() {
    let m = metal();
    let (c, o, h) = (64usize, 16usize, 8usize);
    let x = up(&m, &rand(&[1, c, h, 1], 80, 1.0));
    let w = up(&m, &rand(&[o, c, 3, 3], 81, 0.05));
    let f32s = |n: usize| (n * 4) as u64;
    let out = h * o;
    let im2col = h * c * 9 + out + o * c * 9;
    let op = (h + 2) * (32 + 2) * c + 9 * c * o;
    assert!(
        op > im2col,
        "the shape must make the op's scratch the larger"
    );
    let im2col_room = m.with_budget(Budget::new(f32s(out + im2col)));
    let op_room = m.with_budget(Budget::new(f32s(out + op)));

    m.set_conv2d_tensorops(Conv2dTensorOps::Off);
    let (_, ran, _) = counted(&im2col_room, || {
        im2col_room.conv2d_forward(&x, &w, None, params(1, 1))
    });
    assert_eq!(ran, 1, "Off fits its own charge on the im2col route");
    for mode in [Conv2dTensorOps::F32, Conv2dTensorOps::F16] {
        m.set_conv2d_tensorops(mode);
        assert!(
            matches!(
                im2col_room.conv2d_forward(&x, &w, None, params(1, 1)),
                Err(OjasError::CapacityExceeded { .. })
            ),
            "{mode:?}: charged as im2col"
        );
        let (_, ran, _) = counted(&op_room, || {
            op_room.conv2d_forward(&x, &w, None, params(1, 1))
        });
        assert_eq!(ran, 0, "{mode:?}: the op should run within its own charge");
    }
    m.set_conv2d_tensorops(Conv2dTensorOps::Off);
}
