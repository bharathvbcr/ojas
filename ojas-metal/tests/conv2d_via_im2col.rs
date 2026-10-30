//! `MetalBackend::conv2d_via_im2col`, the reference arm of the pointwise
//! route's tests and bench: a 1 x 1 stride 1 unpadded convolution through
//! `ojas_im2col`, the NT GEMM and `ojas_conv_nchw` whatever the shape. It runs
//! those kernels, gives the CPU's answer, and any other shape takes the same
//! route as `conv2d_forward`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Backend, Conv2dParams};

const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-4;

#[test]
fn a_pointwise_conv_through_im2col_runs_im2col_and_the_layout_pass() {
    let m = metal();
    let p = Conv2dParams::new();
    let x = rand(&[2, 16, 6, 8], 1, 1.0);
    let w = rand(&[24, 16, 1, 1], 2, 0.1);
    let b = rand(&[24], 3, 1.0);
    let want = ok("cpu", cpu().conv2d_forward(&x, &w, Some(&b), p));
    let (xd, wd, bd) = (up(&m, &x), up(&m, &w), up(&m, &b));
    ok("sync", m.sync());
    let (i0, l0) = (
        m.kernel_dispatches("ojas_im2col"),
        m.kernel_dispatches("ojas_conv_nchw"),
    );
    let via = ok("via", m.conv2d_via_im2col(&xd, &wd, Some(&bd), p));
    ok("sync", m.sync());
    assert_eq!(m.kernel_dispatches("ojas_im2col") - i0, 1);
    assert_eq!(m.kernel_dispatches("ojas_conv_nchw") - l0, 1);
    same_tensor("via im2col", &via, &want, ATOL, RTOL);

    let direct = ok("direct", m.conv2d_forward(&xd, &wd, Some(&bd), p));
    ok("sync", m.sync());
    close("direct vs via", &down(&direct), &down(&via), ATOL, RTOL);
    eprintln!(
        "BITS direct vs via im2col: equal {}",
        bits(&down(&direct)) == bits(&down(&via))
    );
}

#[test]
fn any_other_shape_through_im2col_is_the_forward_route() {
    let m = metal();
    let p = Conv2dParams {
        padding: [1, 1],
        ..Conv2dParams::new()
    };
    let x = up(&m, &rand(&[1, 6, 9, 9], 4, 1.0));
    let w = up(&m, &rand(&[10, 6, 3, 3], 5, 0.1));
    let a = ok("forward", m.conv2d_forward(&x, &w, None, p));
    let b = ok("via", m.conv2d_via_im2col(&x, &w, None, p));
    ok("sync", m.sync());
    assert_eq!(bits(&down(&a)), bits(&down(&b)));
}
