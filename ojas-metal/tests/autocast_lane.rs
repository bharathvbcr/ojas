//! Linear on bf16-exact operands (`Tensor::bf16_exact`, what `Autocast`
//! hands Metal in a bf16 region) runs on tessl's bf16 lane, the
//! 4-simdgroup cooperative TensorOps kernels, instead of the exact-f32
//! 1-simdgroup tile. The products are the exact lane's, so the outputs
//! differ only by the order of the f32 sums. A pair with one untagged
//! operand, and a `Bf16` operand, stay on the exact-f32 lane. A device
//! without TensorOps runs every pair on the exact-f32 lane, which these
//! tests assert there instead.
#![cfg(target_os = "macos")]

mod common;

use common::{close, cpu, down, host, metal, ok, rand, up};
use ojas_core::{Autocast, Backend, Budget, Conv2dParams, OjasError, Tensor};
use ojas_metal::MetalBackend;

const ROWS: usize = 256;
const KIN: usize = 192;
const NOUT: usize = 320;
/// Order-only: K = 192 products of values in [-1, 1) summed in another
/// order. One bf16 rounding of an output (2^-9 relative) is far outside it.
const ATOL: f32 = 1e-4;
const RTOL: f32 = 1e-5;

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

/// Dispatches (tessl binder scopes) `op` encodes, and its result.
fn counted<T>(m: &MetalBackend, op: impl FnOnce() -> Result<T, OjasError>) -> (T, u64) {
    ok("sync", m.sync());
    let before = m.dispatches();
    let out = ok("counted op", op());
    ok("sync", m.sync());
    (out, m.dispatches() - before)
}

/// The lane tests below branch on `has_tensorops`, so a flag that read false
/// by mistake would pass them on the exact lane alone. tessl sets it when its
/// library holds `matmul2d_tensorops_f32`; the M5 Pro these tests were
/// written on has it, so an Apple M5 that reports none is a broken build.
#[test]
fn an_apple_m5_reports_tensorops() {
    let m = metal();
    eprintln!("{}: has_tensorops {}", m.device_name(), m.has_tensorops());
    if m.device_name().starts_with("Apple M5") {
        assert!(
            m.has_tensorops(),
            "{} reports no TensorOps: tessl's library lacks matmul2d_tensorops_f32",
            m.device_name()
        );
    }
}

/// Cast dispatches a GEMM on two bf16-exact operands adds: tessl casts each
/// f32 operand to bf16 on the bf16 lane, which only a TensorOps device runs.
fn casts(m: &MetalBackend) -> u64 {
    if m.has_tensorops() {
        2
    } else {
        0
    }
}

/// `t` rounded by `Autocast` (tagged bf16-exact), and an untagged upload of
/// the same bits.
fn rounded(m: &MetalBackend, ac: &Autocast<MetalBackend>, t: &Tensor) -> (Tensor, Tensor) {
    let tagged = ok("cast_bf16", ac.cast_bf16(&up(m, t)));
    assert!(tagged.bf16_exact());
    let plain = up(m, &host(&down(&tagged), t.shape()));
    assert!(!plain.bf16_exact());
    (tagged, plain)
}

#[test]
fn bf16_exact_linear_forward_runs_the_bf16_lane() {
    let m = metal();
    let ac = Autocast::new(m.clone());
    let (x, xp) = rounded(&m, &ac, &rand(&[ROWS, KIN], 1, 1.0));
    let (w, wp) = rounded(&m, &ac, &rand(&[NOUT, KIN], 2, 1.0));

    let (exact, exact_n) = counted(&m, || m.linear_forward(&xp, &wp));
    let (lane, lane_n) = counted(&m, || m.linear_forward(&x, &w));
    // The bf16 lane casts its two f32 operands to bf16 before the GEMM.
    assert_eq!(lane_n, exact_n + casts(&m), "bf16 lane dispatches");
    close("forward", &down(&lane), &down(&exact), ATOL, RTOL);
    if !m.has_tensorops() {
        assert_eq!(bits(&lane), bits(&exact), "no TensorOps: the exact lane");
    }

    // One untagged operand: the exact lane, bit for bit.
    let (mixed, mixed_n) = counted(&m, || m.linear_forward(&x, &wp));
    assert_eq!(mixed_n, exact_n, "mixed pair dispatches");
    assert_eq!(bits(&mixed), bits(&exact));

    // A Bf16 operand keeps its widened copy's bits (the bf16-operand
    // contract), so it is not routed by its dtype.
    let wh = ok("to_bf16", m.to_bf16(&wp));
    let half = ok("half", m.linear_forward(&x, &wh));
    assert_eq!(bits(&half), bits(&exact));
}

#[test]
fn bf16_exact_linear_backward_runs_the_bf16_lane_per_gemm() {
    let m = metal();
    let ac = Autocast::new(m.clone());
    let (x, xp) = rounded(&m, &ac, &rand(&[ROWS, KIN], 3, 1.0));
    let (w, wp) = rounded(&m, &ac, &rand(&[NOUT, KIN], 4, 1.0));
    let (gy, gyp) = rounded(&m, &ac, &rand(&[ROWS, NOUT], 5, 1.0));

    let ((gx_e, gw_e), exact_n) = counted(&m, || m.linear_backward(&xp, &wp, &gyp));
    let ((gx, gw), lane_n) = counted(&m, || m.linear_backward(&x, &w, &gy));
    // Two GEMMs on the bf16 lane, each casting its two operands.
    assert_eq!(lane_n, exact_n + 2 * casts(&m), "bf16 lane dispatches");
    close("grad_input", &down(&gx), &down(&gx_e), ATOL, RTOL);
    // gW sums over the 256 rows.
    close("grad_weight", &down(&gw), &down(&gw_e), 2.0 * ATOL, RTOL);
    if !m.has_tensorops() {
        assert_eq!(bits(&gx), bits(&gx_e), "no TensorOps: the exact lane");
        assert_eq!(bits(&gw), bits(&gw_e), "no TensorOps: the exact lane");
    }

    // Only gW's pair (gy, x) is bf16-exact: gX keeps the exact lane's bits.
    let ((gx_m, gw_m), mixed_n) = counted(&m, || m.linear_backward(&x, &wp, &gy));
    assert_eq!(mixed_n, exact_n + casts(&m), "one GEMM on the bf16 lane");
    assert_eq!(bits(&gx_m), bits(&gx_e));
    assert_eq!(bits(&gw_m), bits(&gw));

    let (gx_i, input_n) = counted(&m, || m.linear_backward_input(&x, &w, &gy));
    let (gx_ie, input_exact_n) = counted(&m, || m.linear_backward_input(&xp, &wp, &gyp));
    assert_eq!(input_n, input_exact_n + casts(&m), "bf16 lane dispatches");
    assert_eq!(bits(&gx_i), bits(&gx));
    assert_eq!(bits(&gx_ie), bits(&gx_e));
}

/// The bf16 copies tessl makes are charged with the op's scratch: a budget
/// that holds the exact lane's call but not the copies refuses the bf16
/// lane before anything runs, and the exact lane still runs under it.
/// Without TensorOps no copies are made, so none are charged.
#[test]
fn the_bf16_lane_charges_its_operand_copies() {
    let m = metal();
    let ac = Autocast::new(m.clone());
    let (x, xp) = rounded(&m, &ac, &rand(&[ROWS, KIN], 6, 1.0));
    let (w, wp) = rounded(&m, &ac, &rand(&[NOUT, KIN], 7, 1.0));
    // The operands sit on `m`'s budget; `tight` charges only the call.
    let out = (ROWS * NOUT * 4) as u64;
    let copies = ((ROWS * KIN + NOUT * KIN) * 2) as u64;
    let tight = m.with_budget(Budget::new(out + copies / 2));
    if m.has_tensorops() {
        assert!(matches!(
            tight.linear_forward(&x, &w),
            Err(OjasError::CapacityExceeded { .. })
        ));
    } else {
        ok("no TensorOps: no copies", tight.linear_forward(&x, &w));
    }
    ok("exact lane", tight.linear_forward(&xp, &wp));
}

/// `conv2d_forward` as the exact-f32 route forms it, from public ops: host
/// im2col (`[n * oh * ow, c * kh * kw]`), `linear_forward` on that matrix
/// and the `[o, c * kh * kw]` weight, then the layout pass and the bias add on
/// the host. The convolution's exact lane runs that same NT GEMM, so its
/// bits must equal this.
fn conv_by_linear(
    m: &MetalBackend,
    x: &Tensor,
    w: &Tensor,
    bias: &Tensor,
    p: Conv2dParams,
) -> Vec<f32> {
    let [n, c, h, wd] = [x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]];
    let [o, kh, kw] = [w.shape()[0], w.shape()[2], w.shape()[3]];
    let (oh, ow) = (
        (h + 2 * p.padding[0] - kh) / p.stride[0] + 1,
        (wd + 2 * p.padding[1] - kw) / p.stride[1] + 1,
    );
    let (xs, width) = (ok("x", x.to_f32_vec()), c * kh * kw);
    let mut cols = vec![0.0f32; n * oh * ow * width];
    for img in 0..n {
        for pixel in 0..oh * ow {
            let (oy, ox) = (pixel / ow, pixel % ow);
            for ch in 0..c {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let iy = (oy * p.stride[0] + ky) as isize - p.padding[0] as isize;
                        let ix = (ox * p.stride[1] + kx) as isize - p.padding[1] as isize;
                        if iy >= 0 && iy < h as isize && ix >= 0 && ix < wd as isize {
                            cols[(img * oh * ow + pixel) * width + (ch * kh + ky) * kw + kx] =
                                xs[((img * c + ch) * h + iy as usize) * wd + ix as usize];
                        }
                    }
                }
            }
        }
    }
    let prod = down(&ok(
        "linear",
        m.linear_forward(
            &up(m, &host(&cols, &[n * oh * ow, width])),
            &up(m, &host(&ok("w", w.to_f32_vec()), &[o, width])),
        ),
    ));
    let bs = ok("bias", bias.to_f32_vec());
    let mut y = vec![0.0f32; n * o * oh * ow];
    for img in 0..n {
        for ch in 0..o {
            for pixel in 0..oh * ow {
                y[(img * o + ch) * oh * ow + pixel] =
                    prod[(img * oh * ow + pixel) * o + ch] + bs[ch];
            }
        }
    }
    y
}

/// The convolution cases of the lane tests: a padded 3 x 3 (as the neck's),
/// a strided, unpadded 4 x 4 (as the patch embedding), K = 54 and 48, and a
/// ragged one (K = 27, 63 pixels, 5 output channels: off every tile edge).
fn conv_cases() -> [([usize; 4], [usize; 4], Conv2dParams); 3] {
    [
        (
            [1, 3, 7, 9],
            [5, 3, 3, 3],
            Conv2dParams {
                padding: [1, 1],
                ..Conv2dParams::new()
            },
        ),
        (
            [2, 6, 9, 9],
            [10, 6, 3, 3],
            Conv2dParams {
                padding: [1, 1],
                ..Conv2dParams::new()
            },
        ),
        (
            [1, 3, 16, 16],
            [12, 3, 4, 4],
            Conv2dParams {
                stride: [4, 4],
                ..Conv2dParams::new()
            },
        ),
    ]
}

#[test]
fn bf16_exact_conv2d_forward_runs_the_bf16_lane() {
    let m = metal();
    let ac = Autocast::new(m.clone());
    for (i, (xs, ws, p)) in conv_cases().into_iter().enumerate() {
        let seed = 10 * (i as u64 + 1);
        let (x, xp) = rounded(&m, &ac, &rand(&xs, seed, 1.0));
        let (w, wp) = rounded(&m, &ac, &rand(&ws, seed + 1, 1.0));
        let bias = up(&m, &rand(&[ws[0]], seed + 2, 1.0));

        let (exact, exact_n) = counted(&m, || m.conv2d_forward(&xp, &wp, Some(&bias), p));
        let (lane, lane_n) = counted(&m, || m.conv2d_forward(&x, &w, Some(&bias), p));
        // The bf16 lane casts the im2col matrix and the weight to bf16
        // before the GEMM.
        assert_eq!(
            lane_n,
            exact_n + casts(&m),
            "case {i}: bf16 lane dispatches"
        );
        close(
            "lane vs exact lane",
            &down(&lane),
            &down(&exact),
            ATOL,
            RTOL,
        );
        // The CPU reference on the same bf16-exact values.
        let want = ok(
            "cpu",
            cpu().conv2d_forward(
                &host(&down(&xp), &xs),
                &host(&down(&wp), &ws),
                Some(&ok("bias", bias.to_host(&Budget::new(1 << 30)))),
                p,
            ),
        );
        close(
            "lane vs cpu",
            &down(&lane),
            &ok("want", want.to_f32_vec()),
            ATOL,
            RTOL,
        );
        if !m.has_tensorops() {
            assert_eq!(bits(&lane), bits(&exact), "no TensorOps: the exact lane");
        }

        // One untagged operand: the exact lane, bit for bit.
        let (mixed, mixed_n) = counted(&m, || m.conv2d_forward(&x, &wp, Some(&bias), p));
        assert_eq!(mixed_n, exact_n, "case {i}: mixed pair dispatches");
        assert_eq!(bits(&mixed), bits(&exact), "case {i}: mixed pair bits");

        // The exact lane is the NT GEMM a linear runs, unchanged: its bits
        // are those of im2col on the host, `linear_forward` and the layout.
        let by_linear = conv_by_linear(
            &m,
            &host(&down(&xp), &xs),
            &host(&down(&wp), &ws),
            &ok("bias", bias.to_host(&Budget::new(1 << 30))),
            p,
        );
        let got: Vec<u32> = down(&exact).iter().map(|v| v.to_bits()).collect();
        let want: Vec<u32> = by_linear.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            got, want,
            "case {i}: f32 conv is not the exact NT GEMM's bits"
        );
    }
}

/// An f32 convolution (neither operand tagged) stays on the exact-f32 NT
/// GEMM, bit for bit what it was before the lane was threaded through.
#[test]
fn an_f32_conv_stays_on_the_exact_lane() {
    let m = metal();
    for (i, (xs, ws, p)) in conv_cases().into_iter().enumerate() {
        let (x, w) = (rand(&xs, 70 + i as u64, 1.0), rand(&ws, 80 + i as u64, 1.0));
        let bias = rand(&[ws[0]], 90 + i as u64, 1.0);
        let got = ok(
            "conv",
            m.conv2d_forward(&up(&m, &x), &up(&m, &w), Some(&up(&m, &bias)), p),
        );
        let by_linear = conv_by_linear(&m, &x, &w, &bias, p);
        assert_eq!(
            bits(&got),
            by_linear.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "case {i}"
        );
    }
}

/// The bf16 copies tessl makes of a convolution's im2col matrix and weight
/// are charged with the op's scratch, as a linear's are.
#[test]
fn the_conv_bf16_lane_charges_its_operand_copies() {
    let m = metal();
    let ac = Autocast::new(m.clone());
    let p = Conv2dParams {
        padding: [1, 1],
        ..Conv2dParams::new()
    };
    let (x, xp) = rounded(&m, &ac, &rand(&[1, 8, 10, 10], 100, 1.0));
    let (w, wp) = rounded(&m, &ac, &rand(&[12, 8, 3, 3], 101, 1.0));
    let (cols, prod, out) = (100 * 72, 100 * 12, 12 * 100);
    let exact_call = ((cols + prod + out) * 4) as u64;
    let copies = ((cols + 12 * 72) * 2) as u64;
    let tight = m.with_budget(Budget::new(exact_call + copies / 2));
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
