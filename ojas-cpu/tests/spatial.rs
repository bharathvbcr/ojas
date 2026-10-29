//! The spatial ops of SAM-style vision models on the CPU: `conv2d_forward`
//! (patch embedding, 1x1, strided 3x3, grouped, depthwise 7x7),
//! `conv_transpose2d_forward`, `max_pool2d_forward`,
//! `interpolate_bilinear_forward`, `rope_interleaved_*` with the axial 2D
//! tables, and the view-plus-permute layout ops (`window_partition`,
//! `window_merge`, `pixel_shuffle`, `pixel_unshuffle`). Forward values are
//! checked against f64 references written the direct way (a seven-loop
//! convolution, a scatter-add transposed convolution, the complex product of
//! SAM2's `apply_rotary_enc`), and the bit-level contract is that the thread
//! count does not change a bit.

mod common;

use common::{assert_nonfinite, assert_range, assert_shape, bits};
use ojas_core::{
    axial_rope_2d_tables, pixel_shuffle, pixel_unshuffle, window_merge, window_partition, Backend,
    Budget, Conv2dParams, ConvTranspose2dParams, Numerics, Tensor,
};
use ojas_cpu::CpuBackend;

fn cpu(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 28), threads)
        .unwrap()
        .with_numerics(numerics)
}

/// Deterministic values in `[lo, hi)`.
fn vals(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (hi - lo) * ((s >> 11) as f64 / (1u64 << 53) as f64)
        })
        .collect()
}

/// Random values already rounded to f32, so the reference sees exactly what
/// the backend does.
fn rounded(seed: u64, n: usize) -> Vec<f64> {
    vals(seed, n, -1.0, 1.0)
        .into_iter()
        .map(|x| f64::from(x as f32))
        .collect()
}

fn t(b: &CpuBackend, v: &[f64], shape: &[usize]) -> Tensor {
    let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    Tensor::from_f32(&f, shape, b.budget()).unwrap()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

fn close(name: &str, got: &[f32], want: &[f64], atol: f64, rtol: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = atol + rtol * w.abs();
        assert!(
            (f64::from(g) - w).abs() <= tol,
            "{name}[{i}]: got {g}, want {w}, tol {tol}"
        );
    }
}

// ---- references ----------------------------------------------------------

struct Conv {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    o: usize,
    k: [usize; 2],
    p: Conv2dParams,
}

fn out_len(size: usize, pad: usize, k: usize, dil: usize, stride: usize) -> usize {
    (size + 2 * pad - dil * (k - 1) - 1) / stride + 1
}

/// The textbook seven-loop convolution.
fn conv_ref(cv: &Conv, x: &[f64], w: &[f64], bias: Option<&[f64]>) -> Vec<f64> {
    let (ho, wo) = (
        out_len(
            cv.h,
            cv.p.padding[0],
            cv.k[0],
            cv.p.dilation[0],
            cv.p.stride[0],
        ),
        out_len(
            cv.w,
            cv.p.padding[1],
            cv.k[1],
            cv.p.dilation[1],
            cv.p.stride[1],
        ),
    );
    let (cg, og) = (cv.c / cv.p.groups, cv.o / cv.p.groups);
    let mut y = vec![0.0; cv.n * cv.o * ho * wo];
    for b in 0..cv.n {
        for oc in 0..cv.o {
            let g = oc / og;
            for oh in 0..ho {
                for ow in 0..wo {
                    let mut acc = bias.map_or(0.0, |b| b[oc]);
                    for ic in 0..cg {
                        for i in 0..cv.k[0] {
                            for j in 0..cv.k[1] {
                                let ih = (oh * cv.p.stride[0] + i * cv.p.dilation[0]) as isize
                                    - cv.p.padding[0] as isize;
                                let iw = (ow * cv.p.stride[1] + j * cv.p.dilation[1]) as isize
                                    - cv.p.padding[1] as isize;
                                if ih < 0 || iw < 0 || ih >= cv.h as isize || iw >= cv.w as isize {
                                    continue;
                                }
                                let xi = ((b * cv.c + g * cg + ic) * cv.h + ih as usize) * cv.w
                                    + iw as usize;
                                let wi = ((oc * cg + ic) * cv.k[0] + i) * cv.k[1] + j;
                                acc += x[xi] * w[wi];
                            }
                        }
                    }
                    y[((b * cv.o + oc) * ho + oh) * wo + ow] = acc;
                }
            }
        }
    }
    y
}

fn params(stride: usize, pad: usize, dil: usize, groups: usize) -> Conv2dParams {
    Conv2dParams {
        stride: [stride, stride],
        padding: [pad, pad],
        dilation: [dil, dil],
        groups,
    }
}

#[test]
fn conv2d_matches_the_reference() {
    // (name, N, C, H, W, O, kernel, params, bias)
    let cases = [
        (
            "patch embed 16x16 s16",
            2,
            3,
            32,
            48,
            8,
            [16, 16],
            params(16, 0, 1, 1),
            true,
        ),
        ("1x1", 2, 6, 5, 7, 4, [1, 1], params(1, 0, 1, 1), true),
        (
            "1x1 no bias",
            1,
            5,
            4,
            4,
            3,
            [1, 1],
            params(1, 0, 1, 1),
            false,
        ),
        (
            "3x3 s2 p1",
            2,
            4,
            9,
            11,
            5,
            [3, 3],
            params(2, 1, 1, 1),
            true,
        ),
        (
            "3x3 s1 p1",
            1,
            3,
            6,
            6,
            4,
            [3, 3],
            params(1, 1, 1, 1),
            false,
        ),
        (
            "3x3 dilated 2",
            1,
            2,
            9,
            9,
            3,
            [3, 3],
            params(1, 2, 2, 1),
            true,
        ),
        (
            "grouped 3x3",
            1,
            6,
            7,
            7,
            4,
            [3, 3],
            params(1, 1, 1, 2),
            true,
        ),
        (
            "grouped 1x1",
            2,
            8,
            4,
            5,
            6,
            [1, 1],
            params(1, 0, 1, 2),
            false,
        ),
        (
            "depthwise 7x7 p3",
            2,
            5,
            12,
            10,
            5,
            [7, 7],
            params(1, 3, 1, 5),
            true,
        ),
        (
            "depthwise 3x3 s2",
            1,
            4,
            9,
            9,
            4,
            [3, 3],
            params(2, 1, 1, 4),
            false,
        ),
        (
            "depthwise multiplier 2",
            1,
            3,
            6,
            6,
            6,
            [3, 3],
            params(1, 1, 1, 3),
            true,
        ),
        (
            "1x1 strided",
            1,
            4,
            8,
            8,
            3,
            [1, 1],
            params(2, 0, 1, 1),
            true,
        ),
        (
            "non-square kernel",
            1,
            2,
            8,
            7,
            3,
            [2, 3],
            Conv2dParams {
                stride: [1, 2],
                padding: [1, 0],
                dilation: [1, 1],
                groups: 1,
            },
            true,
        ),
    ];
    for (name, n, c, h, w, o, k, p, with_bias) in cases {
        let cv = Conv {
            n,
            c,
            h,
            w,
            o,
            k,
            p,
        };
        let x = rounded(1, n * c * h * w);
        let wt = rounded(2, o * (c / p.groups) * k[0] * k[1]);
        let bias = rounded(3, o);
        let bias_ref = with_bias.then_some(bias.as_slice());
        let want = conv_ref(&cv, &x, &wt, bias_ref);
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let be = cpu(2, numerics);
            let xt = t(&be, &x, &[n, c, h, w]);
            let wtt = t(&be, &wt, &[o, c / p.groups, k[0], k[1]]);
            let bt = t(&be, &bias, &[o]);
            let y = be
                .conv2d_forward(&xt, &wtt, with_bias.then_some(&bt), p)
                .unwrap();
            let (ho, wo) = (
                out_len(h, p.padding[0], k[0], p.dilation[0], p.stride[0]),
                out_len(w, p.padding[1], k[1], p.dilation[1], p.stride[1]),
            );
            assert_eq!(y.shape(), &[n, o, ho, wo], "{name} shape");
            close(
                &format!("{name} {numerics:?}"),
                &host(&y),
                &want,
                1e-5,
                1e-5,
            );
        }
    }
}

/// Scatter-add transposed convolution, the definition.
#[allow(clippy::too_many_arguments)]
fn conv_transpose_ref(
    (n, c, h, w): (usize, usize, usize, usize),
    o: usize,
    k: [usize; 2],
    p: ConvTranspose2dParams,
    x: &[f64],
    wt: &[f64],
    bias: Option<&[f64]>,
) -> (Vec<f64>, [usize; 2]) {
    let ho = (h - 1) * p.stride[0] + k[0] + p.output_padding[0] - 2 * p.padding[0];
    let wo = (w - 1) * p.stride[1] + k[1] + p.output_padding[1] - 2 * p.padding[1];
    let mut y = vec![0.0; n * o * ho * wo];
    for b in 0..n {
        for oc in 0..o {
            for v in &mut y[(b * o + oc) * ho * wo..(b * o + oc + 1) * ho * wo] {
                *v = bias.map_or(0.0, |bi| bi[oc]);
            }
        }
        for ic in 0..c {
            for ih in 0..h {
                for iw in 0..w {
                    let xv = x[((b * c + ic) * h + ih) * w + iw];
                    for oc in 0..o {
                        for i in 0..k[0] {
                            for j in 0..k[1] {
                                let oh = (ih * p.stride[0] + i) as isize - p.padding[0] as isize;
                                let ow = (iw * p.stride[1] + j) as isize - p.padding[1] as isize;
                                if oh < 0 || ow < 0 || oh >= ho as isize || ow >= wo as isize {
                                    continue;
                                }
                                y[((b * o + oc) * ho + oh as usize) * wo + ow as usize] +=
                                    xv * wt[((ic * o + oc) * k[0] + i) * k[1] + j];
                            }
                        }
                    }
                }
            }
        }
    }
    (y, [ho, wo])
}

#[test]
fn conv_transpose2d_matches_the_scatter_reference() {
    let cases = [
        (
            "2x2 s2 (mask decoder upscale)",
            (2, 6, 5, 4),
            3,
            [2, 2],
            ConvTranspose2dParams {
                stride: [2, 2],
                ..Default::default()
            },
            true,
        ),
        (
            "3x3 s2 p1 op1",
            (1, 3, 4, 5),
            4,
            [3, 3],
            ConvTranspose2dParams {
                stride: [2, 2],
                padding: [1, 1],
                output_padding: [1, 1],
            },
            true,
        ),
        (
            "3x3 s1",
            (1, 2, 4, 4),
            2,
            [3, 3],
            ConvTranspose2dParams::default(),
            false,
        ),
        (
            "4x4 s2 p1",
            (1, 4, 6, 6),
            3,
            [4, 4],
            ConvTranspose2dParams {
                stride: [2, 2],
                padding: [1, 1],
                output_padding: [0, 0],
            },
            true,
        ),
        (
            "asymmetric",
            (1, 2, 3, 5),
            2,
            [2, 3],
            ConvTranspose2dParams {
                stride: [2, 3],
                padding: [0, 1],
                output_padding: [1, 2],
            },
            true,
        ),
    ];
    for (name, (n, c, h, w), o, k, p, with_bias) in cases {
        let x = rounded(4, n * c * h * w);
        let wt = rounded(5, c * o * k[0] * k[1]);
        let bias = rounded(6, o);
        let (want, [ho, wo]) = conv_transpose_ref(
            (n, c, h, w),
            o,
            k,
            p,
            &x,
            &wt,
            with_bias.then_some(bias.as_slice()),
        );
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let be = cpu(2, numerics);
            let bt = t(&be, &bias, &[o]);
            let y = be
                .conv_transpose2d_forward(
                    &t(&be, &x, &[n, c, h, w]),
                    &t(&be, &wt, &[c, o, k[0], k[1]]),
                    with_bias.then_some(&bt),
                    p,
                )
                .unwrap();
            assert_eq!(y.shape(), &[n, o, ho, wo], "{name} shape");
            close(
                &format!("{name} {numerics:?}"),
                &host(&y),
                &want,
                1e-5,
                1e-5,
            );
        }
    }
}

#[test]
fn max_pool2d_matches_the_reference() {
    for (n, c, h, w, k, s) in [
        (2, 3, 8, 6, [2, 2], [2, 2]),
        (1, 2, 9, 7, [2, 2], [2, 2]),
        (1, 2, 9, 9, [3, 3], [2, 2]),
        (1, 1, 5, 6, [2, 3], [1, 2]),
    ] {
        let x = rounded(7, n * c * h * w);
        let (ho, wo) = ((h - k[0]) / s[0] + 1, (w - k[1]) / s[1] + 1);
        let mut want = vec![f64::NEG_INFINITY; n * c * ho * wo];
        for plane in 0..n * c {
            for oh in 0..ho {
                for ow in 0..wo {
                    for i in 0..k[0] {
                        for j in 0..k[1] {
                            let v = x[(plane * h + oh * s[0] + i) * w + ow * s[1] + j];
                            let slot = &mut want[(plane * ho + oh) * wo + ow];
                            *slot = slot.max(v);
                        }
                    }
                }
            }
        }
        let be = cpu(2, Numerics::Exact);
        let y = be
            .max_pool2d_forward(&t(&be, &x, &[n, c, h, w]), k, s)
            .unwrap();
        assert_eq!(y.shape(), &[n, c, ho, wo]);
        close("max pool", &host(&y), &want, 0.0, 0.0);
    }
}

#[test]
fn bilinear_matches_torch_conventions() {
    let be = cpu(1, Numerics::Exact);
    let x = t(&be, &[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    // torch.nn.functional.interpolate(x, size=(4, 4), mode="bilinear",
    // align_corners=False)
    let y = be.interpolate_bilinear_forward(&x, [4, 4], false).unwrap();
    close(
        "2x2 to 4x4",
        &host(&y),
        &[
            1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0,
        ],
        1e-7,
        0.0,
    );
    // align_corners=True hits the corners exactly.
    let y = be.interpolate_bilinear_forward(&x, [3, 3], true).unwrap();
    close(
        "2x2 to 3x3 align",
        &host(&y),
        &[1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0],
        1e-7,
        0.0,
    );
    // Same size is the identity either way.
    let data = rounded(8, 2 * 3 * 5 * 7);
    let img = t(&be, &data, &[2, 3, 5, 7]);
    for align in [false, true] {
        let y = be
            .interpolate_bilinear_forward(&img, [5, 7], align)
            .unwrap();
        close("identity", &host(&y), &data, 1e-7, 0.0);
    }
}

#[test]
fn bilinear_matches_an_f64_reference_when_resizing() {
    for (h, w, oh, ow, align) in [
        (4, 5, 9, 11, false),
        (4, 5, 9, 11, true),
        (9, 11, 4, 5, false),
        (9, 11, 4, 5, true),
        (6, 6, 1, 1, true),
        (6, 6, 1, 1, false),
        (16, 16, 64, 64, false),
    ] {
        let x = rounded(9, 2 * h * w);
        let axis = |size: usize, out: usize, o: usize| -> (usize, usize, f64) {
            let src = if align {
                if out > 1 {
                    o as f64 * (size - 1) as f64 / (out - 1) as f64
                } else {
                    0.0
                }
            } else {
                ((o as f64 + 0.5) * size as f64 / out as f64 - 0.5).max(0.0)
            };
            let i0 = (src.floor() as usize).min(size - 1);
            (
                i0,
                (i0 + 1).min(size - 1),
                (src - i0 as f64).clamp(0.0, 1.0),
            )
        };
        let mut want = vec![0.0; 2 * oh * ow];
        for plane in 0..2 {
            for yy in 0..oh {
                for xx in 0..ow {
                    let (y0, y1, ly) = axis(h, oh, yy);
                    let (x0, x1, lx) = axis(w, ow, xx);
                    let at = |r: usize, c: usize| x[plane * h * w + r * w + c];
                    want[(plane * oh + yy) * ow + xx] = (1.0 - ly)
                        * ((1.0 - lx) * at(y0, x0) + lx * at(y0, x1))
                        + ly * ((1.0 - lx) * at(y1, x0) + lx * at(y1, x1));
                }
            }
        }
        let be = cpu(2, Numerics::Exact);
        let y = be
            .interpolate_bilinear_forward(&t(&be, &x, &[1, 2, h, w]), [oh, ow], align)
            .unwrap();
        close(
            &format!("{h}x{w} to {oh}x{ow} align {align}"),
            &host(&y),
            &want,
            1e-5,
            1e-5,
        );
    }
}

// ---- RoPE ----------------------------------------------------------------

#[test]
fn rope_interleaved_is_the_complex_product_and_its_backward_inverts_it() {
    let (b, tt, h, d) = (2, 5, 3, 8);
    let x = rounded(10, b * tt * h * d);
    let angles = vals(11, tt * d / 2, -3.0, 3.0);
    let (cos, sin): (Vec<f64>, Vec<f64>) = angles.iter().map(|a| (a.cos(), a.sin())).unzip();
    let be = cpu(2, Numerics::Exact);
    let (ct, st) = (t(&be, &cos, &[tt, d / 2]), t(&be, &sin, &[tt, d / 2]));
    let (cos, sin) = (rounded_back(&ct), rounded_back(&st));
    let xt = t(&be, &x, &[b, tt, h, d]);
    let y = be.rope_interleaved_forward(&xt, &ct, &st).unwrap();
    let mut want = vec![0.0; x.len()];
    for row in 0..b * tt * h {
        let ti = (row / h) % tt;
        for i in 0..d / 2 {
            // (x0 + i x1) * (cos + i sin)
            let (x0, x1) = (x[row * d + 2 * i], x[row * d + 2 * i + 1]);
            let (c, s) = (cos[ti * d / 2 + i], sin[ti * d / 2 + i]);
            want[row * d + 2 * i] = x0 * c - x1 * s;
            want[row * d + 2 * i + 1] = x0 * s + x1 * c;
        }
    }
    close("rope forward", &host(&y), &want, 1e-6, 1e-6);
    // The backward is the rotation by -angle, so it undoes the forward.
    let back = be.rope_interleaved_backward(&y, &ct, &st).unwrap();
    close("rope round trip", &host(&back), &x, 1e-6, 1e-6);
}

fn rounded_back(t: &Tensor) -> Vec<f64> {
    host(t).into_iter().map(f64::from).collect()
}

#[test]
fn axial_tables_rotate_columns_then_rows() {
    let be = cpu(1, Numerics::Exact);
    let (d, gh, gw, theta) = (16usize, 3usize, 4usize, 10000.0f64);
    let (cos, sin) = axial_rope_2d_tables(d, gh, gw, theta, 1.0, be.budget()).unwrap();
    assert_eq!(cos.shape(), &[gh * gw, d / 2]);
    assert_eq!(sin.shape(), &[gh * gw, d / 2]);
    let (c, s) = (host(&cos), host(&sin));
    let quarter = d / 4;
    for tok in 0..gh * gw {
        let (col, row) = ((tok % gw) as f64, (tok / gw) as f64);
        for j in 0..quarter {
            let f = 1.0 / theta.powf((4 * j) as f64 / d as f64);
            for (pair, pos) in [(j, col), (quarter + j, row)] {
                let a = pos * f;
                assert!(
                    (f64::from(c[tok * d / 2 + pair]) - a.cos()).abs() < 1e-6,
                    "cos tok {tok} pair {pair}"
                );
                assert!(
                    (f64::from(s[tok * d / 2 + pair]) - a.sin()).abs() < 1e-6,
                    "sin tok {tok} pair {pair}"
                );
            }
        }
    }
    // Position scaling stretches the angles.
    let (c2, _) = axial_rope_2d_tables(d, gh, gw, theta, 0.5, be.budget()).unwrap();
    let a = (3.0f64 * 0.5).cos();
    assert!((f64::from(host(&c2)[3 * d / 2]) - a).abs() < 1e-6);
    // Refusals.
    assert_range(axial_rope_2d_tables(6, 2, 2, theta, 1.0, be.budget()));
    assert_range(axial_rope_2d_tables(8, 2, 2, 1.0, 1.0, be.budget()));
    assert_range(axial_rope_2d_tables(8, 2, 2, theta, 0.0, be.budget()));
    assert_shape(axial_rope_2d_tables(8, 0, 2, theta, 1.0, be.budget()));
    // An angle that overflows f32 is refused, not stored as NaN.
    assert_range(axial_rope_2d_tables(8, 1, 4, theta, f32::MAX, be.budget()));
}

// ---- layout ops ------------------------------------------------------------

#[test]
fn window_split_and_merge_follow_the_grid_and_invert_each_other() {
    let be = cpu(1, Numerics::Exact);
    let (b, h, w, c, ws) = (2usize, 4usize, 6usize, 3usize, 2usize);
    let data: Vec<f64> = (0..b * h * w * c).map(|i| i as f64).collect();
    let x = t(&be, &data, &[b, h, w, c]);
    let win = window_partition(&be, &x, ws).unwrap();
    assert_eq!(win.shape(), &[b * 2 * 3, ws, ws, c]);
    let got = host(&win);
    let (nh, nw) = (h / ws, w / ws);
    for bi in 0..b {
        for gy in 0..nh {
            for gx in 0..nw {
                for i in 0..ws {
                    for j in 0..ws {
                        for ch in 0..c {
                            let from = ((bi * h + gy * ws + i) * w + gx * ws + j) * c + ch;
                            let to = ((((bi * nh + gy) * nw + gx) * ws + i) * ws + j) * c + ch;
                            assert_eq!(f64::from(got[to]), data[from]);
                        }
                    }
                }
            }
        }
    }
    let merged = window_merge(&be, &win, h, w).unwrap();
    assert_eq!(merged.shape(), &[b, h, w, c]);
    assert_eq!(host(&merged), host(&x));
    // 72 tokens in windows of 24, as the 1008-pixel ViT does.
    let big = t(&be, &vec![0.0; 72 * 72 * 2], &[1, 72, 72, 2]);
    assert_eq!(
        window_partition(&be, &big, 24).unwrap().shape(),
        &[9, 24, 24, 2]
    );
    // No padding is done; the caller pads.
    assert_shape(window_partition(&be, &x, 4));
    assert_shape(window_merge(&be, &win, 4, 8));
}

#[test]
fn pixel_shuffle_matches_torch_and_unshuffle_inverts_it() {
    let be = cpu(1, Numerics::Exact);
    let (n, c, r, h, w) = (2usize, 3usize, 2usize, 3usize, 4usize);
    let data: Vec<f64> = (0..n * c * r * r * h * w).map(|i| i as f64).collect();
    let x = t(&be, &data, &[n, c * r * r, h, w]);
    let y = pixel_shuffle(&be, &x, r).unwrap();
    assert_eq!(y.shape(), &[n, c, h * r, w * r]);
    let got = host(&y);
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h * r {
                for ow in 0..w * r {
                    // torch: out[n, c, h*r + i, w*r + j] = in[n, c*r*r + i*r + j, h, w]
                    let (hh, i, ww, j) = (oh / r, oh % r, ow / r, ow % r);
                    let src = ((ni * c * r * r + ci * r * r + i * r + j) * h + hh) * w + ww;
                    let dst = ((ni * c + ci) * h * r + oh) * w * r + ow;
                    assert_eq!(f64::from(got[dst]), data[src]);
                }
            }
        }
    }
    let back = pixel_unshuffle(&be, &y, r).unwrap();
    assert_eq!(back.shape(), x.shape());
    assert_eq!(host(&back), host(&x));
    assert_shape(pixel_shuffle(&be, &x, 3));
    assert_shape(pixel_unshuffle(&be, &y, 5));
}

// ---- contracts -------------------------------------------------------------

#[test]
fn the_thread_count_does_not_change_a_bit() {
    let x = rounded(12, 2 * 8 * 20 * 20);
    let dense = rounded(13, 12 * 8 * 3 * 3);
    let depthwise = rounded(14, 8 * 7 * 7);
    let up = rounded(15, 8 * 4 * 2 * 2);
    let bias12 = rounded(16, 12);
    let bias8 = rounded(17, 8);
    let bias4 = rounded(18, 4);
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let run = |threads: usize| {
            let be = cpu(threads, numerics);
            let xt = t(&be, &x, &[2, 8, 20, 20]);
            let conv = be
                .conv2d_forward(
                    &xt,
                    &t(&be, &dense, &[12, 8, 3, 3]),
                    Some(&t(&be, &bias12, &[12])),
                    params(2, 1, 1, 1),
                )
                .unwrap();
            let dw = be
                .conv2d_forward(
                    &xt,
                    &t(&be, &depthwise, &[8, 1, 7, 7]),
                    Some(&t(&be, &bias8, &[8])),
                    params(1, 3, 1, 8),
                )
                .unwrap();
            let tr = be
                .conv_transpose2d_forward(
                    &xt,
                    &t(&be, &up, &[8, 4, 2, 2]),
                    Some(&t(&be, &bias4, &[4])),
                    ConvTranspose2dParams {
                        stride: [2, 2],
                        ..Default::default()
                    },
                )
                .unwrap();
            let pool = be.max_pool2d_forward(&xt, [2, 2], [2, 2]).unwrap();
            let rs = be
                .interpolate_bilinear_forward(&xt, [33, 47], false)
                .unwrap();
            [conv, dw, tr, pool, rs].map(|y| bits(&host(&y)))
        };
        assert_eq!(run(1), run(4), "{numerics:?}");
    }
}

#[test]
fn refusals_name_the_shape_and_non_finite_values() {
    let be = cpu(1, Numerics::Exact);
    let x = t(&be, &rounded(20, 2 * 4 * 5 * 5), &[2, 4, 5, 5]);
    let w = t(&be, &rounded(21, 6 * 4 * 3 * 3), &[6, 4, 3, 3]);
    let p = Conv2dParams::new();
    // Rank, groups, weight channels, bias length, window larger than input.
    assert_shape(be.conv2d_forward(&t(&be, &[1.0; 25], &[5, 5]), &w, None, p));
    assert_shape(be.conv2d_forward(&x, &w, None, Conv2dParams { groups: 3, ..p }));
    assert_shape(be.conv2d_forward(
        &x,
        &t(&be, &rounded(22, 6 * 3 * 3 * 3), &[6, 3, 3, 3]),
        None,
        p,
    ));
    assert_shape(be.conv2d_forward(&x, &w, Some(&t(&be, &[0.0; 5], &[5])), p));
    assert_shape(be.conv2d_forward(
        &x,
        &w,
        None,
        Conv2dParams {
            stride: [0, 1],
            ..p
        },
    ));
    let big = t(&be, &rounded(23, 6 * 4 * 7 * 7), &[6, 4, 7, 7]);
    assert_shape(be.conv2d_forward(&x, &big, None, p));
    assert_shape(be.max_pool2d_forward(&x, [6, 6], [1, 1]));
    assert_shape(be.max_pool2d_forward(&x, [2, 2], [0, 2]));
    assert_shape(be.interpolate_bilinear_forward(&x, [0, 4], false));
    let tw = t(&be, &rounded(24, 4 * 2 * 2 * 2), &[4, 2, 2, 2]);
    assert_shape(be.conv_transpose2d_forward(
        &x,
        &tw,
        None,
        ConvTranspose2dParams {
            stride: [2, 2],
            padding: [0, 0],
            output_padding: [2, 0],
        },
    ));
    assert_shape(be.rope_interleaved_forward(
        &t(&be, &[0.0; 12], &[1, 2, 1, 6]),
        &t(&be, &[0.0; 4], &[2, 2]),
        &t(&be, &[0.0; 4], &[2, 2]),
    ));
    // A NaN in any operand, the bias included.
    let mut bad = rounded(25, 2 * 4 * 5 * 5);
    bad[7] = f64::NAN;
    assert_nonfinite(be.conv2d_forward(&t(&be, &bad, &[2, 4, 5, 5]), &w, None, p));
    let nan_bias = t(&be, &[0.0, 0.0, f64::NAN, 0.0, 0.0, 0.0], &[6]);
    assert_nonfinite(be.conv2d_forward(&x, &w, Some(&nan_bias), p));
}

#[test]
fn a_budget_too_small_for_the_output_or_the_im2col_matrix_is_refused() {
    let big = cpu(1, Numerics::Exact);
    let x = t(&big, &rounded(30, 3 * 16 * 16), &[1, 3, 16, 16]);
    let w = t(&big, &rounded(31, 4 * 3 * 4 * 4), &[4, 3, 4, 4]);
    let p = Conv2dParams {
        stride: [4, 4],
        ..Conv2dParams::new()
    };
    // The operands fit; the 4x4x4 output and the 48 x 16 im2col matrix do not.
    let operands = (3 * 16 * 16 + 4 * 3 * 4 * 4) * 4;
    let tight = CpuBackend::with_threads(Budget::new(operands as u64 + 64), 1).unwrap();
    let (x, w) = (
        x.to_host(tight.budget()).unwrap(),
        w.to_host(tight.budget()).unwrap(),
    );
    common::assert_capacity(tight.conv2d_forward(&x, &w, None, p));
}

#[test]
fn a_budget_too_small_for_the_bilinear_axis_tables_is_refused() {
    // A 1 x 4096 output is 16 KiB, which the cap holds, but its axis tables
    // are about 96 KiB of heap that the cap must refuse before they exist.
    let big = cpu(1, Numerics::Exact);
    let x = t(&big, &[1.0, 2.0], &[1, 1, 1, 2]);
    let tight = CpuBackend::with_threads(Budget::new(32 * 1024), 1).unwrap();
    let x = x.to_host(tight.budget()).unwrap();
    common::assert_capacity(tight.interpolate_bilinear_forward(&x, [1, 4096], false));
}

// ---- backward --------------------------------------------------------------
//
// A convolution is linear in its input and, separately, in its weight, so
// the gradient of `L = <f(x, w), gy>` with respect to `x[i]` is exactly
// `<f(e_i, w), gy>` with `e_i` the i-th unit tensor (and likewise for `w`).
// The references below take those adjoints from the f64 forward
// references above, one unit tensor at a time: they share nothing with the
// gathers and GEMMs under test.

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn unit(n: usize, i: usize) -> Vec<f64> {
    let mut e = vec![0.0; n];
    e[i] = 1.0;
    e
}

/// `gy` summed over images and plane, per channel.
fn bias_ref(gy: &[f64], n: usize, o: usize, plane: usize) -> Vec<f64> {
    (0..o)
        .map(|oc| {
            (0..n)
                .map(|b| {
                    gy[(b * o + oc) * plane..(b * o + oc + 1) * plane]
                        .iter()
                        .sum::<f64>()
                })
                .sum()
        })
        .collect()
}

#[test]
fn conv2d_backward_is_the_adjoint_of_the_reference() {
    // (name, N, C, H, W, O, kernel, params, bias)
    let cases = [
        (
            "patch embed 4x4 s4",
            2,
            3,
            8,
            12,
            4,
            [4, 4],
            params(4, 0, 1, 1),
            true,
        ),
        (
            "1x1 in place",
            2,
            5,
            4,
            3,
            3,
            [1, 1],
            params(1, 0, 1, 1),
            true,
        ),
        ("1x1 s2", 1, 3, 5, 5, 2, [1, 1], params(2, 0, 1, 1), false),
        ("3x3 s2 p1", 2, 4, 7, 6, 5, [3, 3], params(2, 1, 1, 1), true),
        (
            "3x3 s1 p1",
            1,
            3,
            5,
            5,
            4,
            [3, 3],
            params(1, 1, 1, 1),
            false,
        ),
        (
            "3x3 dilated 2 p2",
            1,
            2,
            7,
            7,
            3,
            [3, 3],
            params(1, 2, 2, 1),
            true,
        ),
        ("grouped 2", 2, 4, 5, 6, 6, [3, 3], params(1, 1, 1, 2), true),
        (
            "depthwise 7x7 p3",
            1,
            3,
            8,
            8,
            3,
            [7, 7],
            params(1, 3, 1, 3),
            true,
        ),
        (
            "depthwise x2 multiplier",
            1,
            2,
            5,
            5,
            4,
            [3, 3],
            params(2, 1, 1, 2),
            false,
        ),
        (
            "2x3 stride 2x1",
            1,
            2,
            6,
            5,
            3,
            [2, 3],
            Conv2dParams {
                stride: [2, 1],
                padding: [0, 1],
                dilation: [1, 1],
                groups: 1,
            },
            true,
        ),
    ];
    for (name, n, c, h, w, o, k, p, with_bias) in cases {
        let cv = Conv {
            n,
            c,
            h,
            w,
            o,
            k,
            p,
        };
        let (nx, nw) = (n * c * h * w, o * (c / p.groups) * k[0] * k[1]);
        let x = rounded(40, nx);
        let wt = rounded(41, nw);
        let bias = with_bias.then(|| rounded(42, o));
        let y = conv_ref(&cv, &x, &wt, bias.as_deref());
        let gy = rounded(43, y.len());
        let plane = y.len() / (n * o);
        let want_x: Vec<f64> = (0..nx)
            .map(|i| dot(&conv_ref(&cv, &unit(nx, i), &wt, None), &gy))
            .collect();
        let want_w: Vec<f64> = (0..nw)
            .map(|i| dot(&conv_ref(&cv, &x, &unit(nw, i), None), &gy))
            .collect();
        let mut first: Option<Vec<Vec<u32>>> = None;
        for numerics in [Numerics::Exact, Numerics::Fast] {
            for threads in [1, 3] {
                let be = cpu(threads, numerics);
                let wshape = [o, c / p.groups, k[0], k[1]];
                let bias_t = bias.as_ref().map(|b| t(&be, b, &[o]));
                let g = be
                    .conv2d_backward(
                        &t(&be, &x, &[n, c, h, w]),
                        &t(&be, &wt, &wshape),
                        bias_t.as_ref(),
                        &t(&be, &gy, &conv_out_shape(&cv)),
                        p,
                    )
                    .unwrap();
                assert_eq!(g.input.shape(), &[n, c, h, w], "{name}");
                assert_eq!(g.weight.shape(), &wshape, "{name}");
                close(
                    &format!("{name} grad x"),
                    &host(&g.input),
                    &want_x,
                    2e-5,
                    1e-5,
                );
                close(
                    &format!("{name} grad w"),
                    &host(&g.weight),
                    &want_w,
                    2e-5,
                    1e-5,
                );
                assert_eq!(
                    g.bias.is_some(),
                    with_bias,
                    "{name}: bias gradient presence"
                );
                if let Some(gb) = &g.bias {
                    assert_eq!(gb.shape(), &[o]);
                    close(
                        &format!("{name} grad b"),
                        &host(gb),
                        &bias_ref(&gy, n, o, plane),
                        2e-5,
                        1e-5,
                    );
                }
                let mut all = vec![bits(&host(&g.input)), bits(&host(&g.weight))];
                all.extend(g.bias.as_ref().map(|b| bits(&host(b))));
                // The thread count never changes a bit, under either numerics.
                if threads == 1 {
                    first = Some(all);
                } else {
                    assert_eq!(
                        Some(all),
                        first,
                        "{name} {numerics:?}: threads changed the bits"
                    );
                }
            }
        }
    }
}

/// `[N, O, Ho, Wo]` of [`conv_ref`].
fn conv_out_shape(cv: &Conv) -> [usize; 4] {
    [
        cv.n,
        cv.o,
        out_len(
            cv.h,
            cv.p.padding[0],
            cv.k[0],
            cv.p.dilation[0],
            cv.p.stride[0],
        ),
        out_len(
            cv.w,
            cv.p.padding[1],
            cv.k[1],
            cv.p.dilation[1],
            cv.p.stride[1],
        ),
    ]
}

#[test]
fn conv_transpose2d_backward_is_the_adjoint_of_the_scatter_reference() {
    // (name, N, C, H, W, O, kernel, params, bias)
    let tp = |stride: usize, pad: usize, out_pad: usize| ConvTranspose2dParams {
        stride: [stride, stride],
        padding: [pad, pad],
        output_padding: [out_pad, out_pad],
    };
    let cases = [
        (
            "kernel == stride 2",
            2,
            3,
            4,
            5,
            4,
            [2, 2],
            tp(2, 0, 0),
            true,
        ),
        ("3x3 s2 p1 op1", 1, 2, 4, 3, 3, [3, 3], tp(2, 1, 1), true),
        ("3x3 s1 p1", 1, 3, 4, 4, 2, [3, 3], tp(1, 1, 0), false),
        ("4x4 s2 p1", 2, 2, 3, 3, 2, [4, 4], tp(2, 1, 0), true),
        (
            "2x3 stride 3x2",
            1,
            2,
            3,
            4,
            2,
            [2, 3],
            ConvTranspose2dParams {
                stride: [3, 2],
                padding: [0, 1],
                output_padding: [2, 1],
            },
            false,
        ),
    ];
    for (name, n, c, h, w, o, k, p, with_bias) in cases {
        let (nx, nw) = (n * c * h * w, c * o * k[0] * k[1]);
        let x = rounded(50, nx);
        let wt = rounded(51, nw);
        let bias = with_bias.then(|| rounded(52, o));
        let (y, [ho, wo]) = conv_transpose_ref((n, c, h, w), o, k, p, &x, &wt, bias.as_deref());
        let gy = rounded(53, y.len());
        let fwd = |x: &[f64], wt: &[f64]| conv_transpose_ref((n, c, h, w), o, k, p, x, wt, None).0;
        let want_x: Vec<f64> = (0..nx).map(|i| dot(&fwd(&unit(nx, i), &wt), &gy)).collect();
        let want_w: Vec<f64> = (0..nw).map(|i| dot(&fwd(&x, &unit(nw, i)), &gy)).collect();
        let mut first: Option<Vec<Vec<u32>>> = None;
        for numerics in [Numerics::Exact, Numerics::Fast] {
            for threads in [1, 3] {
                let be = cpu(threads, numerics);
                let bias_t = bias.as_ref().map(|b| t(&be, b, &[o]));
                let g = be
                    .conv_transpose2d_backward(
                        &t(&be, &x, &[n, c, h, w]),
                        &t(&be, &wt, &[c, o, k[0], k[1]]),
                        bias_t.as_ref(),
                        &t(&be, &gy, &[n, o, ho, wo]),
                        p,
                    )
                    .unwrap();
                assert_eq!(g.input.shape(), &[n, c, h, w], "{name}");
                assert_eq!(g.weight.shape(), &[c, o, k[0], k[1]], "{name}");
                close(
                    &format!("{name} grad x"),
                    &host(&g.input),
                    &want_x,
                    2e-5,
                    1e-5,
                );
                close(
                    &format!("{name} grad w"),
                    &host(&g.weight),
                    &want_w,
                    2e-5,
                    1e-5,
                );
                assert_eq!(
                    g.bias.is_some(),
                    with_bias,
                    "{name}: bias gradient presence"
                );
                if let Some(gb) = &g.bias {
                    close(
                        &format!("{name} grad b"),
                        &host(gb),
                        &bias_ref(&gy, n, o, ho * wo),
                        2e-5,
                        1e-5,
                    );
                }
                let mut all = vec![bits(&host(&g.input)), bits(&host(&g.weight))];
                all.extend(g.bias.as_ref().map(|b| bits(&host(b))));
                if threads == 1 {
                    first = Some(all);
                } else {
                    assert_eq!(
                        Some(all),
                        first,
                        "{name} {numerics:?}: threads changed the bits"
                    );
                }
            }
        }
    }
}

/// Max pooling's gradient the scatter way: each window, in ascending
/// order, adds its `gy` to the first input (in `(kh, kw)` order) holding
/// its maximum.
fn max_pool_backward_ref(
    (n, c, h, w): (usize, usize, usize, usize),
    k: [usize; 2],
    s: [usize; 2],
    x: &[f64],
    gy: &[f64],
) -> Vec<f64> {
    let (ho, wo) = ((h - k[0]) / s[0] + 1, (w - k[1]) / s[1] + 1);
    let mut gx = vec![0.0; n * c * h * w];
    for plane in 0..n * c {
        for oh in 0..ho {
            for ow in 0..wo {
                let mut best = (plane * h + oh * s[0]) * w + ow * s[1];
                for i in 0..k[0] {
                    for j in 0..k[1] {
                        let at = (plane * h + oh * s[0] + i) * w + ow * s[1] + j;
                        if x[at] > x[best] {
                            best = at;
                        }
                    }
                }
                gx[best] += gy[(plane * ho + oh) * wo + ow];
            }
        }
    }
    gx
}

#[test]
fn max_pool2d_backward_matches_the_scatter_reference() {
    for (n, c, h, w, k, s) in [
        (2, 3, 8, 6, [2, 2], [2, 2]),
        (1, 2, 9, 7, [2, 2], [2, 2]),
        (1, 2, 9, 9, [3, 3], [2, 2]),
        (1, 2, 7, 7, [3, 3], [1, 1]),
        (1, 1, 5, 6, [2, 3], [1, 2]),
    ] {
        let x = rounded(60, n * c * h * w);
        let (ho, wo) = ((h - k[0]) / s[0] + 1, (w - k[1]) / s[1] + 1);
        let gy = rounded(61, n * c * ho * wo);
        let want = max_pool_backward_ref((n, c, h, w), k, s, &x, &gy);
        let mut first = None;
        for threads in [1, 3] {
            let be = cpu(threads, Numerics::Exact);
            let gx = be
                .max_pool2d_backward(
                    &t(&be, &x, &[n, c, h, w]),
                    &t(&be, &gy, &[n, c, ho, wo]),
                    k,
                    s,
                )
                .unwrap();
            assert_eq!(gx.shape(), &[n, c, h, w]);
            close("max pool backward", &host(&gx), &want, 1e-6, 0.0);
            let got = bits(&host(&gx));
            if let Some(first) = &first {
                assert_eq!(&got, first, "threads changed the bits");
            } else {
                first = Some(got);
            }
        }
    }
}

#[test]
fn max_pool2d_backward_credits_the_first_maximum_of_a_tie() {
    // Every value ties: each 2x2 window's gradient goes to its top-left
    // input, as torch's `max_pool2d` picks the first maximum; with stride 1
    // the overlapping windows of a tied row add up on their shared inputs.
    let be = cpu(1, Numerics::Exact);
    let x = t(&be, &[0.5; 9], &[1, 1, 3, 3]);
    let gy = t(&be, &[1.0, 2.0, 4.0, 8.0], &[1, 1, 2, 2]);
    let gx = be.max_pool2d_backward(&x, &gy, [2, 2], [1, 1]).unwrap();
    assert_eq!(host(&gx), [1.0, 2.0, 0.0, 4.0, 8.0, 0.0, 0.0, 0.0, 0.0]);
    // The forward reads the same element.
    let x = t(&be, &[3.0, 3.0, 1.0, 3.0], &[1, 1, 2, 2]);
    let gy = t(&be, &[1.0], &[1, 1, 1, 1]);
    assert_eq!(
        host(&be.max_pool2d_forward(&x, [2, 2], [2, 2]).unwrap()),
        [3.0]
    );
    let gx = be.max_pool2d_backward(&x, &gy, [2, 2], [2, 2]).unwrap();
    assert_eq!(host(&gx), [1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn spatial_backwards_refuse_a_malformed_or_nonfinite_gradient() {
    let be = cpu(1, Numerics::Exact);
    let x = t(&be, &rounded(70, 2 * 4 * 4), &[1, 2, 4, 4]);
    let w = t(&be, &rounded(71, 3 * 2 * 3 * 3), &[3, 2, 3, 3]);
    let p = params(1, 1, 1, 1);
    let good = t(&be, &rounded(72, 3 * 4 * 4), &[1, 3, 4, 4]);
    let short = t(&be, &rounded(72, 3 * 4 * 3), &[1, 3, 4, 3]);
    let mut nan = rounded(72, 3 * 4 * 4);
    nan[5] = f64::NAN;
    let nan = t(&be, &nan, &[1, 3, 4, 4]);
    be.conv2d_backward(&x, &w, None, &good, p).unwrap();
    assert_shape(be.conv2d_backward(&x, &w, None, &short, p));
    assert_nonfinite(be.conv2d_backward(&x, &w, None, &nan, p));
    let wrong_bias = t(&be, &[0.0; 2], &[2]);
    assert_shape(be.conv2d_backward(&x, &w, Some(&wrong_bias), &good, p));

    let wt = t(&be, &rounded(73, 2 * 3 * 2 * 2), &[2, 3, 2, 2]);
    let tp = ConvTranspose2dParams {
        stride: [2, 2],
        ..ConvTranspose2dParams::new()
    };
    let up = t(&be, &rounded(74, 3 * 8 * 8), &[1, 3, 8, 8]);
    be.conv_transpose2d_backward(&x, &wt, None, &up, tp)
        .unwrap();
    assert_shape(be.conv_transpose2d_backward(&x, &wt, None, &good, tp));

    let pooled = t(&be, &rounded(75, 2 * 2 * 2), &[1, 2, 2, 2]);
    be.max_pool2d_backward(&x, &pooled, [2, 2], [2, 2]).unwrap();
    assert_shape(be.max_pool2d_backward(&x, &good, [2, 2], [2, 2]));
    let mut bad = rounded(75, 2 * 2 * 2);
    bad[0] = f64::INFINITY;
    assert_nonfinite(be.max_pool2d_backward(&x, &t(&be, &bad, &[1, 2, 2, 2]), [2, 2], [2, 2]));
}

#[test]
fn a_budget_too_small_for_the_conv2d_backward_columns_is_refused() {
    // The operands and both gradients fit; the two 18 x 16 column matrices
    // of the 3x3 padded convolution do not.
    let big = cpu(1, Numerics::Exact);
    let x = t(&big, &rounded(80, 2 * 4 * 4), &[1, 2, 4, 4]);
    let w = t(&big, &rounded(81, 2 * 2 * 3 * 3), &[2, 2, 3, 3]);
    let gy = t(&big, &rounded(82, 2 * 4 * 4), &[1, 2, 4, 4]);
    let held = (2 * 4 * 4 + 2 * 2 * 3 * 3 + 2 * 4 * 4) * 4;
    let grads = (2 * 4 * 4 + 2 * 2 * 3 * 3) * 4;
    let tight = CpuBackend::with_threads(Budget::new((held + grads + 64) as u64), 1).unwrap();
    let (x, w, gy) = (
        x.to_host(tight.budget()).unwrap(),
        w.to_host(tight.budget()).unwrap(),
        gy.to_host(tight.budget()).unwrap(),
    );
    common::assert_capacity(tight.conv2d_backward(&x, &w, None, &gy, params(1, 1, 1, 1)));
}
