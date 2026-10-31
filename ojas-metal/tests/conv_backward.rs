//! `conv2d_backward` and `conv_transpose2d_backward` on Metal, for SAM's
//! configurations: the 14 x 14 stride-14 patch embedding, 1 x 1, 3 x 3 with
//! padding 1, the strided 3 x 3 (stride 2, padding 1) and the depthwise 7 x 7
//! (padding 3), each with and without a bias, and the transposed convolution
//! whose kernel equals its stride.
//!
//! Tolerances, fixed before the first run:
//!
//! - `dx` and `dw` of a dense convolution go through tessl's exact-f32 GEMM,
//!   whose summation order is not the CPU's: held to the CPU at
//!   `ATOL` / `RTOL`.
//! - The bias gradient, and both gradients of a depthwise convolution, are
//!   gathers in the CPU's order: held to the CPU's bits.
//! - Against torch (`ojas-oracle/fixtures/conv_bwd/golden.txt`, made by
//!   `ojas-oracle/python/conv_backward_fixture.py`): each gradient's max error
//!   over the float64 pair, normalized by the pair's largest magnitude, at
//!   most `TORCH_TOL`, for Metal and for the CPU reference.
//!
//! What ran is read from `MetalBackend::kernel_dispatches`. A result repeats
//! bit for bit. Dilation, a grouped convolution that is not depthwise, and a
//! transposed convolution with kernel != stride (or padding) are
//! `Unsupported` and name the backward and the configuration.

#![cfg(target_os = "macos")]

mod common;

use std::collections::HashMap;

use common::*;
use ojas_core::{
    Backend, Budget, Conv2dParams, ConvGrad, ConvTranspose2dParams, OjasError, Tensor,
};
use ojas_metal::MetalBackend;

const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-4;
const TORCH_TOL: f64 = 1e-5;

const IM2COL: &str = "ojas_im2col";
const COL2IM: &str = "ojas_col2im";
const DW_X: &str = "ojas_dwconv_bwd_x";
const DW_W: &str = "ojas_dwconv_bwd_w";
const CHAN_SUM: &str = "ojas_chan_sum";
const KERNELS: [&str; 5] = [IM2COL, COL2IM, DW_X, DW_W, CHAN_SUM];

#[derive(Clone, Copy, Debug)]
enum Kind {
    Conv(Conv2dParams),
    Transpose(ConvTranspose2dParams),
}

/// One backward problem: the forward's operands and the output gradient.
struct Case {
    what: String,
    kind: Kind,
    x: Tensor,
    w: Tensor,
    bias: Option<Tensor>,
    gy: Tensor,
}

impl Case {
    fn depthwise(&self) -> bool {
        matches!(self.kind, Kind::Conv(p) if p.groups > 1)
    }
}

fn conv(stride: usize, padding: usize, groups: usize) -> Kind {
    Kind::Conv(Conv2dParams {
        stride: [stride, stride],
        padding: [padding, padding],
        dilation: [1, 1],
        groups,
    })
}

/// A case with random operands; `gy` has the forward's output shape.
fn make(what: &str, kind: Kind, x: &[usize], w: &[usize], with_bias: bool, seed: u64) -> Case {
    let c = cpu();
    let (xt, wt) = (rand(x, seed, 1.0), rand(w, seed + 1, 0.2));
    let out_ch = match kind {
        Kind::Conv(_) => w[0],
        Kind::Transpose(_) => w[1],
    };
    let bias = with_bias.then(|| rand(&[out_ch], seed + 2, 1.0));
    let y = match kind {
        Kind::Conv(p) => ok("forward", c.conv2d_forward(&xt, &wt, bias.as_ref(), p)),
        Kind::Transpose(p) => ok(
            "forward",
            c.conv_transpose2d_forward(&xt, &wt, bias.as_ref(), p),
        ),
    };
    let gy = rand(y.shape(), seed + 3, 1.0);
    Case {
        what: format!("{what}{}", if with_bias { " +bias" } else { "" }),
        kind,
        x: xt,
        w: wt,
        bias,
        gy,
    }
}

/// SAM's five convolutions and the transposed one, at shapes the CPU
/// reference follows quickly, each with and without a bias.
fn sam_cases() -> Vec<Case> {
    let t = Kind::Transpose(ConvTranspose2dParams {
        stride: [2, 2],
        padding: [0, 0],
        output_padding: [0, 0],
    });
    let mut cases = Vec::new();
    for (i, with_bias) in [false, true].into_iter().enumerate() {
        let s = 100 * (i as u64 + 1);
        cases.push(make(
            "patch embed k14 s14",
            conv(14, 0, 1),
            &[2, 3, 30, 44],
            &[16, 3, 14, 14],
            with_bias,
            s,
        ));
        cases.push(make(
            "pointwise 1x1",
            conv(1, 0, 1),
            &[2, 24, 9, 11],
            &[12, 24, 1, 1],
            with_bias,
            s + 10,
        ));
        cases.push(make(
            "3x3 p1",
            conv(1, 1, 1),
            &[2, 12, 13, 10],
            &[16, 12, 3, 3],
            with_bias,
            s + 20,
        ));
        cases.push(make(
            "3x3 s2 p1",
            conv(2, 1, 1),
            &[2, 8, 15, 12],
            &[10, 8, 3, 3],
            with_bias,
            s + 30,
        ));
        cases.push(make(
            "depthwise 7x7 p3",
            conv(1, 3, 6),
            &[2, 6, 11, 13],
            &[6, 1, 7, 7],
            with_bias,
            s + 40,
        ));
        cases.push(make(
            "transpose 2x2 s2",
            t,
            &[2, 7, 5, 6],
            &[7, 4, 2, 2],
            with_bias,
            s + 50,
        ));
    }
    cases
}

/// The gradients of `case` on `be`, with the operands as given.
fn grads(be: &impl Backend, case: &Case, x: &Tensor, w: &Tensor, gy: &Tensor) -> ConvGrad {
    match case.kind {
        Kind::Conv(p) => ok(
            &case.what,
            be.conv2d_backward(x, w, case.bias.as_ref(), gy, p),
        ),
        Kind::Transpose(p) => ok(
            &case.what,
            be.conv_transpose2d_backward(x, w, case.bias.as_ref(), gy, p),
        ),
    }
}

/// The gradients on Metal, from device copies of the operands.
fn on_metal(m: &MetalBackend, case: &Case) -> ConvGrad {
    let dev = Case {
        what: case.what.clone(),
        kind: case.kind,
        x: up(m, &case.x),
        w: up(m, &case.w),
        bias: case.bias.as_ref().map(|b| up(m, b)),
        gy: up(m, &case.gy),
    };
    let out = grads(m, &dev, &dev.x, &dev.w, &dev.gy);
    ok("sync", m.sync());
    out
}

fn counts(m: &MetalBackend) -> [u64; 5] {
    KERNELS.map(|k| m.kernel_dispatches(k))
}

#[test]
fn every_sam_configuration_matches_cpu() {
    let (m, c) = (metal(), cpu());
    for case in sam_cases() {
        let want = grads(&c, &case, &case.x, &case.w, &case.gy);
        let got = on_metal(&m, &case);
        let what = &case.what;
        if case.depthwise() {
            same_bits(&format!("{what} dx"), &got.input, &want.input);
            same_bits(&format!("{what} dw"), &got.weight, &want.weight);
        } else {
            same_tensor(&format!("{what} dx"), &got.input, &want.input, ATOL, RTOL);
            same_tensor(&format!("{what} dw"), &got.weight, &want.weight, ATOL, RTOL);
        }
        match (&got.bias, &want.bias) {
            (Some(g), Some(w)) => same_bits(&format!("{what} db"), g, w),
            (None, None) => assert!(case.bias.is_none(), "{what}: db missing"),
            _ => panic!("{what}: db is Some on one backend only"),
        }
        assert_eq!(got.bias.is_some(), case.bias.is_some(), "{what}");
    }
}

#[test]
fn each_route_runs_its_own_kernels() {
    let m = metal();
    for case in sam_cases() {
        ok("sync", m.sync());
        let before = counts(&m);
        let _ = on_metal(&m, &case);
        let after = counts(&m);
        let ran: Vec<u64> = (0..5).map(|i| after[i] - before[i]).collect();
        let bias = u64::from(case.bias.is_some());
        let want: [u64; 5] = match case.kind {
            // im2col, col2im
            Kind::Conv(p) if p.groups == 1 => [1, 1, 0, 0, bias],
            // dx and dw direct kernels, no im2col
            Kind::Conv(_) => [0, 0, 1, 1, bias],
            // gy as im2col columns, no col2im
            Kind::Transpose(_) => [1, 0, 0, 0, bias],
        };
        assert_eq!(
            ran, want,
            "{}: [im2col, col2im, dw_x, dw_w, chan_sum]",
            case.what
        );
    }
}

#[test]
fn a_result_repeats_bit_for_bit() {
    let m = metal();
    for case in sam_cases() {
        let a = on_metal(&m, &case);
        let b = on_metal(&m, &case);
        assert_eq!(
            bits(&down(&a.input)),
            bits(&down(&b.input)),
            "{} dx",
            case.what
        );
        assert_eq!(
            bits(&down(&a.weight)),
            bits(&down(&b.weight)),
            "{} dw",
            case.what
        );
        if let (Some(x), Some(y)) = (&a.bias, &b.bias) {
            assert_eq!(bits(&down(x)), bits(&down(y)), "{} db", case.what);
        }
    }
}

#[test]
fn other_strides_paddings_and_odd_extents_match_cpu() {
    let (m, c) = (metal(), cpu());
    let cases = [
        make(
            "3x3 s2 p0",
            conv(2, 0, 1),
            &[1, 3, 12, 9],
            &[5, 3, 3, 3],
            true,
            300,
        ),
        make(
            "5x5 s3 p2",
            conv(3, 2, 1),
            &[2, 2, 14, 11],
            &[3, 2, 5, 5],
            false,
            310,
        ),
        make(
            "kernel as large as the padded image",
            conv(1, 1, 1),
            &[1, 2, 3, 3],
            &[2, 2, 5, 5],
            true,
            320,
        ),
        make(
            "one pixel",
            conv(1, 0, 1),
            &[1, 1, 1, 1],
            &[1, 1, 1, 1],
            true,
            330,
        ),
        make(
            "depthwise 3x3 s2 p1",
            conv(2, 1, 4),
            &[1, 4, 9, 8],
            &[4, 1, 3, 3],
            true,
            340,
        ),
        make(
            "transpose 3x3 s3",
            Kind::Transpose(ConvTranspose2dParams {
                stride: [3, 3],
                padding: [0, 0],
                output_padding: [0, 0],
            }),
            &[1, 3, 4, 2],
            &[3, 2, 3, 3],
            true,
            350,
        ),
        make(
            "transpose 2x3 s2x3",
            Kind::Transpose(ConvTranspose2dParams {
                stride: [2, 3],
                padding: [0, 0],
                output_padding: [0, 0],
            }),
            &[2, 2, 3, 3],
            &[2, 3, 2, 3],
            false,
            360,
        ),
    ];
    for case in cases {
        let want = grads(&c, &case, &case.x, &case.w, &case.gy);
        let got = on_metal(&m, &case);
        let what = &case.what;
        same_tensor(&format!("{what} dx"), &got.input, &want.input, ATOL, RTOL);
        same_tensor(&format!("{what} dw"), &got.weight, &want.weight, ATOL, RTOL);
        if let (Some(g), Some(w)) = (&got.bias, &want.bias) {
            same_bits(&format!("{what} db"), g, w);
        }
    }
}

#[test]
fn offset_device_views_give_the_cpu_answer() {
    let (m, c) = (metal(), cpu());
    let case = make(
        "3x3 p1 views",
        conv(1, 1, 1),
        &[1, 3, 6, 6],
        &[4, 3, 3, 3],
        true,
        400,
    );
    let want = grads(&c, &case, &case.x, &case.w, &case.gy);
    let bias = case.bias.as_ref().map(|b| offset_view(&m, b, 7, 404));
    let dev = Case {
        what: case.what.clone(),
        kind: case.kind,
        x: offset_view(&m, &case.x, 3, 401),
        w: offset_view(&m, &case.w, 5, 402),
        bias,
        gy: offset_view(&m, &case.gy, 9, 403),
    };
    let got = grads(&m, &dev, &dev.x, &dev.w, &dev.gy);
    ok("sync", m.sync());
    same_tensor("views dx", &got.input, &want.input, ATOL, RTOL);
    same_tensor("views dw", &got.weight, &want.weight, ATOL, RTOL);
    if let (Some(g), Some(w)) = (&got.bias, &want.bias) {
        same_bits("views db", g, w);
    }
    let dw_case = make(
        "depthwise views",
        conv(1, 3, 3),
        &[1, 3, 8, 7],
        &[3, 1, 7, 7],
        true,
        410,
    );
    let want = grads(&c, &dw_case, &dw_case.x, &dw_case.w, &dw_case.gy);
    let dev = Case {
        what: dw_case.what.clone(),
        kind: dw_case.kind,
        x: offset_view(&m, &dw_case.x, 3, 411),
        w: offset_view(&m, &dw_case.w, 5, 412),
        bias: dw_case.bias.as_ref().map(|b| offset_view(&m, b, 7, 413)),
        gy: offset_view(&m, &dw_case.gy, 9, 414),
    };
    let got = grads(&m, &dev, &dev.x, &dev.w, &dev.gy);
    ok("sync", m.sync());
    same_bits("depthwise views dx", &got.input, &want.input);
    same_bits("depthwise views dw", &got.weight, &want.weight);
}

fn assert_unsupported(what: &str, op: &str, r: Result<ConvGrad, OjasError>, needles: &[&str]) {
    match r {
        Err(OjasError::Unsupported {
            op: got,
            ref detail,
        }) => {
            assert_eq!(got, op, "{what}");
            for n in needles {
                assert!(detail.contains(n), "{what}: the detail names {n}: {detail}");
            }
        }
        other => panic!("{what}: expected Unsupported, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn a_configuration_the_kernels_do_not_cover_is_unsupported_and_named() {
    let m = metal();
    let x = up(&m, &rand(&[1, 4, 12, 12], 500, 1.0));
    let w3 = up(&m, &rand(&[4, 4, 3, 3], 501, 1.0));
    let w2 = up(&m, &rand(&[4, 2, 3, 3], 502, 1.0));
    let conv_cases = [
        (
            "dilation 2",
            &w3,
            Conv2dParams {
                dilation: [2, 2],
                padding: [2, 2],
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
    ];
    for (what, w, p) in conv_cases {
        let out = ojas_core::conv2d_forward_dims(&x, w, None, p).expect("the validator accepts it");
        let gy = up(&m, &rand(&out.out_shape(), 503, 1.0));
        assert_unsupported(
            what,
            "conv2d_backward",
            m.conv2d_backward(&x, w, None, &gy, p),
            &["dilation", "groups"],
        );
        ok("sync", m.sync());
    }
    let wt = up(&m, &rand(&[4, 3, 3, 3], 504, 1.0));
    let t_cases = [
        (
            "kernel 3 != stride 2",
            ConvTranspose2dParams {
                stride: [2, 2],
                ..ConvTranspose2dParams::new()
            },
        ),
        (
            "kernel 3 stride 3 padding 1",
            ConvTranspose2dParams {
                stride: [3, 3],
                padding: [1, 1],
                ..ConvTranspose2dParams::new()
            },
        ),
        (
            "kernel 3 stride 3 output padding 1",
            ConvTranspose2dParams {
                stride: [3, 3],
                output_padding: [1, 1],
                ..ConvTranspose2dParams::new()
            },
        ),
    ];
    for (what, p) in t_cases {
        let out = ojas_core::conv_transpose2d_forward_dims(&x, &wt, None, p)
            .expect("the validator accepts it");
        let gy = up(&m, &rand(&out.out_shape(), 505, 1.0));
        assert_unsupported(
            what,
            "conv_transpose2d_backward",
            m.conv_transpose2d_backward(&x, &wt, None, &gy, p),
            &["kernel", "stride", "padding"],
        );
        ok("sync", m.sync());
    }
}

#[test]
fn host_operands_and_malformed_calls_are_refused_as_the_validator_refuses() {
    let m = metal();
    let c = cpu();
    let x = rand(&[1, 2, 5, 5], 600, 1.0);
    let w = rand(&[3, 2, 3, 3], 601, 0.1);
    let gy = rand(&[1, 3, 5, 5], 602, 1.0);
    let p = Conv2dParams {
        padding: [1, 1],
        ..Conv2dParams::new()
    };
    let r = m.conv2d_backward(&x, &w, None, &gy, p);
    assert!(
        matches!(
            r,
            Err(OjasError::Placement {
                op: "conv2d_backward",
                ..
            })
        ),
        "{:?}",
        r.map(|_| ())
    );
    // A gradient of the wrong shape is the validator's refusal, the CPU's too.
    let bad = rand(&[1, 3, 4, 5], 603, 1.0);
    let want = c.conv2d_backward(&x, &w, None, &bad, p).map(|_| ());
    let got = m
        .conv2d_backward(&up(&m, &x), &up(&m, &w), None, &up(&m, &bad), p)
        .map(|_| ());
    assert!(want.is_err(), "the CPU accepted a mis-shaped gradient");
    assert_eq!(format!("{got:?}"), format!("{want:?}"));
}

#[test]
fn a_non_finite_operand_is_a_deferred_fault_under_the_backward() {
    let m = metal();
    let good = make(
        "nan",
        conv(1, 1, 1),
        &[1, 2, 5, 5],
        &[3, 2, 3, 3],
        true,
        700,
    );
    let poison = |t: &Tensor| {
        let mut v = down(&up(&m, t));
        v[1] = f32::NAN;
        up(&m, &host(&v, t.shape()))
    };
    let (x, w, gy) = (up(&m, &good.x), up(&m, &good.w), up(&m, &good.gy));
    let bias = good.bias.as_ref().map(|b| up(&m, b));
    let Kind::Conv(p) = good.kind else {
        unreachable!()
    };
    deferred(
        &m,
        "nan x",
        m.conv2d_backward(&poison(&good.x), &w, bias.as_ref(), &gy, p),
        "conv2d_backward",
    );
    deferred(
        &m,
        "nan w",
        m.conv2d_backward(&x, &poison(&good.w), bias.as_ref(), &gy, p),
        "conv2d_backward",
    );
    deferred(
        &m,
        "nan gy",
        m.conv2d_backward(&x, &w, bias.as_ref(), &poison(&good.gy), p),
        "conv2d_backward",
    );
    let nan_bias = poison(good.bias.as_ref().unwrap());
    deferred(
        &m,
        "nan bias",
        m.conv2d_backward(&x, &w, Some(&nan_bias), &gy, p),
        "conv2d_backward",
    );
    // The depthwise and transposed routes scan as well.
    let dw = make(
        "nan depthwise",
        conv(1, 3, 3),
        &[1, 3, 8, 8],
        &[3, 1, 7, 7],
        false,
        710,
    );
    let Kind::Conv(dp) = dw.kind else {
        unreachable!()
    };
    deferred(
        &m,
        "nan depthwise gy",
        m.conv2d_backward(&up(&m, &dw.x), &up(&m, &dw.w), None, &poison(&dw.gy), dp),
        "conv2d_backward",
    );
    let tp = make(
        "nan transpose",
        Kind::Transpose(ConvTranspose2dParams {
            stride: [2, 2],
            ..ConvTranspose2dParams::new()
        }),
        &[1, 3, 4, 4],
        &[3, 2, 2, 2],
        false,
        720,
    );
    let Kind::Transpose(tparams) = tp.kind else {
        unreachable!()
    };
    deferred(
        &m,
        "nan transpose gy",
        m.conv_transpose2d_backward(
            &up(&m, &tp.x),
            &up(&m, &tp.w),
            None,
            &poison(&tp.gy),
            tparams,
        ),
        "conv_transpose2d_backward",
    );
    // A clean call afterwards still works.
    let _ = on_metal(&m, &good);
}

#[test]
fn scratch_that_does_not_fit_the_budget_is_refused() {
    // The im2col matrix and the column gradient (18432 values each) and the
    // permuted gradient (2048) do not fit 128 KiB beside the operands and
    // outputs.
    let m = MetalBackend::new(Budget::new(128 << 10)).expect("device");
    let case = make(
        "3x3 p1 big scratch",
        conv(1, 1, 1),
        &[1, 8, 16, 16],
        &[8, 8, 3, 3],
        true,
        800,
    );
    let (x, w, gy) = (up(&m, &case.x), up(&m, &case.w), up(&m, &case.gy));
    let bias = case.bias.as_ref().map(|b| up(&m, b));
    let Kind::Conv(p) = case.kind else {
        unreachable!()
    };
    let r = m.conv2d_backward(&x, &w, bias.as_ref(), &gy, p);
    assert!(
        matches!(r, Err(OjasError::CapacityExceeded { .. })),
        "{:?}",
        r.map(|_| ())
    );
    ok("sync", m.sync());
}

// ---- torch ----

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../ojas-oracle/fixtures/conv_bwd/golden.txt"
);

struct Golden {
    op: String,
    name: String,
    stride: [usize; 2],
    padding: [usize; 2],
    groups: usize,
    f32s: HashMap<String, (Vec<usize>, Vec<f32>)>,
    f64s: HashMap<String, Vec<f64>>,
}

fn pair(v: &str) -> [usize; 2] {
    let mut it = v.split(',').map(|s| s.parse::<usize>().expect("pair"));
    [it.next().expect("first"), it.next().expect("second")]
}

fn parse_golden() -> Vec<Golden> {
    let text = std::fs::read_to_string(GOLDEN).unwrap_or_else(|e| {
        panic!("{GOLDEN}: {e} (run ojas-oracle/python/conv_backward_fixture.py)")
    });
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("ojas-conv-bwd-golden-v1"));
    assert!(lines.next().is_some_and(|l| l.starts_with("torch ")));
    let mut out = Vec::new();
    let mut cur: Option<Golden> = None;
    for line in lines {
        let parts: Vec<&str> = line.split(' ').collect();
        match parts[0] {
            "case" => {
                let kv = |key: &str| {
                    parts[3..]
                        .iter()
                        .find_map(|p| p.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
                        .unwrap_or_else(|| panic!("case line without {key}: {line}"))
                };
                cur = Some(Golden {
                    op: parts[1].to_string(),
                    name: parts[2].to_string(),
                    stride: pair(kv("stride")),
                    padding: pair(kv("padding")),
                    groups: kv("groups").parse().expect("groups"),
                    f32s: HashMap::new(),
                    f64s: HashMap::new(),
                });
            }
            "t" => {
                let g = cur.as_mut().expect("t outside a case");
                let dims: Vec<usize> = parts[3]
                    .split(',')
                    .map(|d| d.parse().expect("dim"))
                    .collect();
                match parts[2] {
                    "f32" => {
                        let v = parts[4..]
                            .iter()
                            .map(|h| f32::from_bits(u32::from_str_radix(h, 16).expect("hex")))
                            .collect();
                        g.f32s.insert(parts[1].to_string(), (dims, v));
                    }
                    "f64" => {
                        let v = parts[4..]
                            .iter()
                            .map(|h| f64::from_bits(u64::from_str_radix(h, 16).expect("hex")))
                            .collect();
                        g.f64s.insert(parts[1].to_string(), v);
                    }
                    other => panic!("dtype {other}"),
                }
            }
            "end" => out.push(cur.take().expect("end outside a case")),
            other => panic!("unknown line {other}"),
        }
    }
    out
}

/// `max |got - want| / max |want|`, or the absolute error when `want` is 0.
fn normalized(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((f64::from(*g) - w).abs()));
    if scale == 0.0 {
        err
    } else {
        err / scale
    }
}

#[test]
fn gradients_match_torch_autograd_per_configuration() {
    let (m, c) = (metal(), cpu());
    let mut worst = 0.0f64;
    for g in parse_golden() {
        let tensor = |k: &str| {
            let (dims, v) = &g.f32s[k];
            host(v, dims)
        };
        let (x, w, gy) = (tensor("x"), tensor("w"), tensor("gy"));
        let bias = g.f32s.contains_key("b").then(|| tensor("b"));
        let kind = if g.op == "conv2d" {
            Kind::Conv(Conv2dParams {
                stride: g.stride,
                padding: g.padding,
                dilation: [1, 1],
                groups: g.groups,
            })
        } else {
            Kind::Transpose(ConvTranspose2dParams {
                stride: g.stride,
                padding: g.padding,
                output_padding: [0, 0],
            })
        };
        let case = Case {
            what: g.name.clone(),
            kind,
            x,
            w,
            bias,
            gy,
        };
        let cpu_grads = grads(&c, &case, &case.x, &case.w, &case.gy);
        let metal_grads = on_metal(&m, &case);
        for (backend, got) in [("cpu", &cpu_grads), ("metal", &metal_grads)] {
            let rows: [(&str, Option<&Tensor>); 3] = [
                ("dx", Some(&got.input)),
                ("dw", Some(&got.weight)),
                ("db", got.bias.as_ref()),
            ];
            for (name, tensor) in rows {
                let Some(tensor) = tensor else {
                    assert!(
                        !g.f64s.contains_key(&format!("{name}_64")),
                        "{} {backend}: {name} missing",
                        g.name
                    );
                    continue;
                };
                let values = if tensor.device().is_some() {
                    down(tensor)
                } else {
                    ok(name, tensor.to_f32_vec())
                };
                let err64 = normalized(&values, &g.f64s[&format!("{name}_64")]);
                let want32: Vec<f64> = g.f32s[name].1.iter().copied().map(f64::from).collect();
                let err32 = normalized(&values, &want32);
                eprintln!("TORCH {:<24} {backend:<5} {name}: vs f64 {err64:.2e}, vs torch f32 {err32:.2e}", g.name);
                assert!(
                    err64 <= TORCH_TOL,
                    "{} {backend} {name}: normalized error {err64:e} over the float64 gradient",
                    g.name
                );
                worst = worst.max(err64);
            }
        }
    }
    eprintln!("TORCH worst normalized error over the float64 pair: {worst:.2e}");
}
