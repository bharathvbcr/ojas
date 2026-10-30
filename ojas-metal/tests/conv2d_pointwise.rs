//! A 1 x 1, stride 1, unpadded `conv2d_forward` on Metal skips im2col and the
//! layout pass: per image one NN GEMM `W [o, c] @ X_n [c, pixels]` writes the
//! NCHW output, and the bias is added in place on `ojas_bcast_binary`. What
//! ran is read from `MetalBackend::kernel_dispatches` (Ojas's own kernels;
//! tessl's GEMMs are not counted) and the dispatch total. The result is
//! checked against the CPU backend and against the route the convolution took
//! before (host im2col, `linear_forward`'s NT GEMM, host layout and bias). A
//! view the GEMM cannot read in place, or images that do not start on 16
//! bytes, take im2col as before.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Autocast, Backend, Budget, Conv2dParams, OjasError, Tensor};
use ojas_metal::MetalBackend;

const IM2COL: &str = "ojas_im2col";
const LAYOUT: &str = "ojas_conv_nchw";
const BCAST: &str = "ojas_bcast_binary";

const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-4;

/// Ojas kernels one call ran.
#[derive(Debug, PartialEq, Eq)]
struct Ran {
    im2col: u64,
    layout: u64,
    bcast: u64,
    all: u64,
}

/// `op`'s result and what it dispatched.
fn counted<T>(m: &MetalBackend, op: impl FnOnce() -> Result<T, OjasError>) -> (T, Ran) {
    ok("sync", m.sync());
    let read = || (m.kernel_dispatches(IM2COL), m.kernel_dispatches(LAYOUT));
    let (i0, l0) = read();
    let (b0, a0) = (m.kernel_dispatches(BCAST), m.dispatches());
    let out = ok("counted op", op());
    ok("sync", m.sync());
    let (i1, l1) = read();
    let ran = Ran {
        im2col: i1 - i0,
        layout: l1 - l0,
        bcast: m.kernel_dispatches(BCAST) - b0,
        all: m.dispatches() - a0,
    };
    (out, ran)
}

/// `x [n, c, h, w]` as the old route formed it: the im2col matrix of a 1 x 1
/// kernel (`[n * h * w, c]`), `linear_forward`'s NT GEMM against the `[o, c]`
/// weight, then the layout pass and the bias add on the host.
fn via_im2col_and_linear(
    m: &MetalBackend,
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
) -> Vec<f32> {
    let [n, c, h, wd] = [x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]];
    let o = w.shape()[0];
    let px = h * wd;
    let xs = ok("x", x.to_f32_vec());
    let mut cols = vec![0.0f32; n * px * c];
    for img in 0..n {
        for ch in 0..c {
            for p in 0..px {
                cols[(img * px + p) * c + ch] = xs[(img * c + ch) * px + p];
            }
        }
    }
    let prod = down(&ok(
        "linear",
        m.linear_forward(
            &up(m, &host(&cols, &[n * px, c])),
            &up(m, &host(&ok("w", w.to_f32_vec()), &[o, c])),
        ),
    ));
    let bs = bias.map(|b| ok("bias", b.to_f32_vec()));
    let mut y = vec![0.0f32; n * o * px];
    for img in 0..n {
        for ch in 0..o {
            for p in 0..px {
                let v = prod[(img * px + p) * o + ch];
                y[(img * o + ch) * px + p] = bs.as_ref().map_or(v, |b| v + b[ch]);
            }
        }
    }
    y
}

/// One pointwise case against the CPU and the old route; the dispatch
/// record and the result's gap to the old route's bits.
fn case(m: &MetalBackend, what: &str, x: [usize; 4], o: usize, with_bias: bool, seed: u64) -> Ran {
    let p = Conv2dParams::new();
    let c = cpu();
    let (xh, wh) = (rand(&x, seed, 1.0), rand(&[o, x[1], 1, 1], seed + 1, 0.1));
    let bh = with_bias.then(|| rand(&[o], seed + 2, 1.0));
    let want = ok("cpu", c.conv2d_forward(&xh, &wh, bh.as_ref(), p));
    let (xd, wd) = (up(m, &xh), up(m, &wh));
    let bd = bh.as_ref().map(|b| up(m, b));
    let (got, ran) = counted(m, || m.conv2d_forward(&xd, &wd, bd.as_ref(), p));
    same_tensor(what, &got, &want, ATOL, RTOL);
    let old = via_im2col_and_linear(m, &xh, &wh, bh.as_ref());
    let new = down(&got);
    close(&format!("{what} vs old route"), &new, &old, ATOL, RTOL);
    eprintln!(
        "BITS {what}: equal to the old route {}",
        bits(&new) == bits(&old)
    );
    ran
}

fn skipped(ran: &Ran, with_bias: bool) {
    assert_eq!(ran.im2col, 0, "im2col ran: {ran:?}");
    assert_eq!(ran.layout, 0, "the layout pass ran: {ran:?}");
    assert_eq!(ran.bcast, u64::from(with_bias), "bias adds: {ran:?}");
}

#[test]
fn a_pointwise_conv_runs_no_im2col_and_no_layout_pass() {
    let m = metal();
    // The neck's 256-channel 1 x 1, at a size the CPU reference can follow.
    for with_bias in [false, true] {
        let ran = case(&m, "256 at 24x24", [1, 256, 24, 24], 256, with_bias, 1);
        skipped(&ran, with_bias);
    }
}

#[test]
fn one_input_channel_matches_cpu() {
    let m = metal();
    for with_bias in [false, true] {
        let ran = case(&m, "c=1", [1, 1, 17, 13], 8, with_bias, 10);
        skipped(&ran, with_bias);
    }
}

#[test]
fn an_output_count_off_the_tile_matches_cpu() {
    let m = metal();
    // o = 37 and 33 pixels (a ragged 32-wide tile on both axes), then o = 1
    // and a single pixel.
    for (what, x, o) in [
        ("o=37", [1, 20, 3, 11], 37),
        ("o=1", [1, 5, 4, 4], 1),
        ("one pixel", [1, 64, 1, 1], 33),
    ] {
        let ran = case(&m, what, x, o, true, 20);
        skipped(&ran, true);
    }
}

#[test]
fn a_batch_runs_one_gemm_per_image_in_the_output_layout() {
    let m = metal();
    // 3 images; c * pixels and o * pixels are multiples of 4, so every image
    // starts on 16 bytes.
    let ran = case(&m, "batch 3", [3, 6, 4, 8], 10, true, 30);
    skipped(&ran, true);
    let ran = case(&m, "batch 3 no bias", [3, 6, 4, 8], 10, false, 31);
    skipped(&ran, false);
}

#[test]
fn a_batch_whose_images_do_not_start_on_16_bytes_takes_im2col() {
    let m = metal();
    // c * pixels = 3 * 5 = 15: image 1 would start 60 bytes in.
    let ran = case(&m, "batch 2 odd", [2, 3, 1, 5], 4, true, 40);
    assert_eq!(ran.im2col, 1, "{ran:?}");
    assert_eq!(ran.layout, 1, "{ran:?}");
    assert_eq!(ran.bcast, 0, "{ran:?}");
    // One image of the same odd size is read in place.
    let ran = case(&m, "batch 1 odd", [1, 3, 1, 5], 4, true, 41);
    skipped(&ran, true);
}

#[test]
fn a_view_the_gemm_cannot_read_in_place_takes_im2col() {
    let m = metal();
    let p = Conv2dParams::new();
    let x = rand(&[1, 8, 6, 6], 50, 1.0);
    let w = rand(&[12, 8, 1, 1], 51, 0.1);
    let b = rand(&[12], 52, 1.0);
    let want = ok("cpu", cpu().conv2d_forward(&x, &w, Some(&b), p));

    // A misaligned input window: im2col reads any 4-byte offset.
    let xd = offset_view(&m, &x, 3, 53);
    let (got, ran) = counted(&m, || {
        m.conv2d_forward(&xd, &up(&m, &w), Some(&up(&m, &b)), p)
    });
    assert_eq!((ran.im2col, ran.layout), (1, 1), "{ran:?}");
    same_tensor("misaligned x", &got, &want, ATOL, RTOL);

    // An aligned window not at the buffer start, a misaligned weight (copied
    // for the GEMM, as a linear's is) and a misaligned bias: pointwise.
    let mut padded = values(4, 54, 1.0);
    padded.extend(ok("x", x.to_f32_vec()));
    let big = up(&m, &host(&padded, &[padded.len()]));
    let xa = ok("narrow", big.narrow(16, x.shape(), x.strides()));
    let (got, ran) = counted(&m, || {
        m.conv2d_forward(
            &xa,
            &offset_view(&m, &w, 3, 55),
            Some(&offset_view(&m, &b, 7, 56)),
            p,
        )
    });
    skipped(&ran, true);
    same_tensor("aligned window, odd weight", &got, &want, ATOL, RTOL);
}

#[test]
fn a_long_k_over_a_wide_c_takes_tessls_split_k_and_stays_within_the_sum_bound() {
    let m = metal();
    // c = 1024 over 8192 pixels and o = 64 rows: tessl's NN splits K in
    // partitions of 256 here, summing in another order than the old route's
    // NT GEMM, so the gap is the f32 dot-product bound, not zero.
    let ran = case(&m, "k=1024 split", [1, 1024, 64, 128], 64, true, 60);
    skipped(&ran, true);
}

#[test]
fn a_pointwise_conv_of_a_neck_scale_matches_the_old_route() {
    let m = metal();
    let p = Conv2dParams::new();
    let xh = rand(&[1, 256, 72, 72], 70, 1.0);
    let wh = rand(&[256, 256, 1, 1], 71, 0.1);
    let bh = rand(&[256], 72, 1.0);
    let (x, w, b) = (up(&m, &xh), up(&m, &wh), up(&m, &bh));
    let (got, ran) = counted(&m, || m.conv2d_forward(&x, &w, Some(&b), p));
    skipped(&ran, true);
    let old = via_im2col_and_linear(&m, &xh, &wh, Some(&bh));
    let new = down(&got);
    close("72x72", &new, &old, ATOL, RTOL);
    eprintln!(
        "BITS 72x72: equal to the old route {}",
        bits(&new) == bits(&old)
    );
}

#[test]
fn a_non_finite_operand_is_a_deferred_fault_under_the_op_name() {
    let m = metal();
    let p = Conv2dParams::new();
    let shapes: [&[usize]; 3] = [&[2, 4, 3, 4], &[5, 4, 1, 1], &[5]];
    for (which, name) in ["input", "weight", "bias"].into_iter().enumerate() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut ops: Vec<Tensor> = shapes
                .iter()
                .enumerate()
                .map(|(i, s)| rand(s, 80 + i as u64, 1.0))
                .collect();
            let mut data = ok("host", ops[which].to_f32_vec());
            let last = data.len() - 1;
            data[last] = bad;
            ops[which] = host(&data, shapes[which]);
            let dev: Vec<Tensor> = ops.iter().map(|t| up(&m, t)).collect();
            ok("sync", m.sync());
            let before = m.kernel_dispatches(IM2COL);
            let r = m.conv2d_forward(&dev[0], &dev[1], Some(&dev[2]), p);
            deferred(&m, &format!("{bad} in {name}"), r, "conv2d_forward");
            assert_eq!(
                m.kernel_dispatches(IM2COL),
                before,
                "{bad} in {name}: the fault came from a route that ran im2col"
            );
        }
    }
}

#[test]
fn a_finite_overflow_is_a_deferred_fault_with_and_without_bias() {
    let m = metal();
    let x = up(&m, &host(&[3.0e38; 8], &[2, 1, 2, 2]));
    let w = up(&m, &host(&[3.0e38], &[1, 1, 1, 1]));
    let r = m.conv2d_forward(&x, &w, None, Conv2dParams::new());
    deferred(&m, "overflow", r, "conv2d_forward");
    let bias = up(&m, &host(&[1.0], &[1]));
    let r = m.conv2d_forward(&x, &w, Some(&bias), Conv2dParams::new());
    deferred(&m, "overflow with bias", r, "conv2d_forward");
}

#[test]
fn a_bias_that_overflows_the_sum_is_a_deferred_fault() {
    let m = metal();
    // A finite product plus a finite bias that overflows f32 when added.
    let x = up(&m, &host(&[1.0; 4], &[1, 1, 2, 2]));
    let w = up(&m, &host(&[3.0e38], &[1, 1, 1, 1]));
    let bias = up(&m, &host(&[3.0e38], &[1]));
    let r = m.conv2d_forward(&x, &w, Some(&bias), Conv2dParams::new());
    deferred(&m, "bias sum overflow", r, "conv2d_forward");
}

/// The 1 x 1 convolution on bf16-exact operands: tessl's bf16 NN lane, which
/// casts the image and the weight to bf16 inside the call (two dispatches),
/// for products that are exact in f32, so the result differs from the exact
/// lane's only by the order of the f32 sums.
#[test]
fn a_bf16_exact_pointwise_conv_runs_the_bf16_lane() {
    let m = metal();
    let ac = Autocast::new(m.clone());
    let p = Conv2dParams::new();
    let tag = |t: &Tensor| {
        let tagged = ok("cast_bf16", ac.cast_bf16(&up(&m, t)));
        assert!(tagged.bf16_exact());
        let plain = up(&m, &host(&down(&tagged), t.shape()));
        (tagged, plain)
    };
    let casts = if m.has_tensorops() { 2 } else { 0 };
    // The neck's shape in small, one input channel short of a tile edge on
    // every axis (c = 5, 63 pixels, o = 3), and a batch.
    for (i, (xs, o)) in [([1, 64, 12, 12], 48), ([1, 5, 7, 9], 3), ([2, 6, 4, 8], 10)]
        .into_iter()
        .enumerate()
    {
        let seed = 90 + 10 * i as u64;
        let x = rand(&xs, seed, 1.0);
        let w = rand(&[o, xs[1], 1, 1], seed + 1, 1.0);
        let bh = rand(&[o], seed + 2, 1.0);
        let b = up(&m, &bh);
        let ((xt, xp), (wt, wp)) = (tag(&x), tag(&w));

        let (exact, exact_ran) = counted(&m, || m.conv2d_forward(&xp, &wp, Some(&b), p));
        let (lane, lane_ran) = counted(&m, || m.conv2d_forward(&xt, &wt, Some(&b), p));
        skipped(&lane_ran, true);
        // The bf16 lane narrows the image and the weight on each image's GEMM.
        assert_eq!(
            lane_ran.all,
            exact_ran.all + casts * xs[0] as u64,
            "case {i}: bf16 lane casts"
        );
        // The CPU reference on the same bf16-exact values.
        let want = ok(
            "cpu",
            cpu().conv2d_forward(
                &host(&down(&xp), x.shape()),
                &host(&down(&wp), w.shape()),
                Some(&bh),
                p,
            ),
        );
        let want = ok("want", want.to_f32_vec());
        close("bf16 lane vs cpu", &down(&lane), &want, 1e-4, 1e-4);
        close(
            "bf16 lane vs exact",
            &down(&lane),
            &down(&exact),
            1e-4,
            1e-4,
        );
        if !m.has_tensorops() {
            assert_eq!(bits(&down(&lane)), bits(&down(&exact)), "no TensorOps");
        }
    }
}

/// The bf16 copies of a pointwise conv (the input image, the weight once per
/// image) are charged with the op's scratch: a budget that holds the exact
/// lane's call (only its output: no im2col matrix, no product) but not the
/// copies refuses the bf16 lane.
#[test]
fn the_pointwise_bf16_lane_charges_its_operand_copies() {
    let m = metal();
    let ac = Autocast::new(m.clone());
    let p = Conv2dParams::new();
    let (nb, c, o, px) = (2usize, 64usize, 48usize, 16 * 16);
    let tag = |t: &Tensor| ok("cast_bf16", ac.cast_bf16(&up(&m, t)));
    let x = tag(&rand(&[nb, c, 16, 16], 95, 1.0));
    let w = tag(&rand(&[o, c, 1, 1], 96, 1.0));
    let plain = |t: &Tensor| up(&m, &host(&down(t), t.shape()));
    let (xp, wp) = (plain(&x), plain(&w));
    let out = (nb * o * px * 4) as u64;
    let copies = ((nb * c * px + o * c * nb) * 2) as u64;
    let tight = m.with_budget(Budget::new(out + copies / 2));
    if m.has_tensorops() {
        assert!(matches!(
            tight.conv2d_forward(&x, &w, None, p),
            Err(OjasError::CapacityExceeded { .. })
        ));
    } else {
        ok(
            "no TensorOps: no copies",
            tight.conv2d_forward(&x, &w, None, p),
        );
    }
    ok("exact lane", tight.conv2d_forward(&xp, &wp, None, p));
}
