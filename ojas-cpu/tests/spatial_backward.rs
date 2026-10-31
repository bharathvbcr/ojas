//! `Backend::group_norm_backward` and `Backend::interpolate_bilinear_backward`
//! on the CPU.
//!
//! - Gradcheck: GroupNorm's three gradients are held to central differences
//!   of an `f64` forward written here from the definition, on the loss
//!   `sum(gy * y)`.
//! - Adjoint: the bilinear resample is linear, so the gradient at input pixel
//!   `i` is exactly `<f(e_i), gy>` for the unit image `e_i`. The references
//!   take every gradient from an `f64` forward, one unit image at a time, and
//!   share nothing with the gather under test. Two relations are exact by
//!   hand: an integer halving credits each input a quarter of its output's
//!   gradient, and the same size is the identity.
//! - Relations are bit-exact: GroupNorm with one group over `[N, C, 1, 1]` is
//!   LayerNorm over `[N, C]`.
//! - The bits do not depend on the thread count, and `Fast` and `Exact` agree
//!   (no backward here uses the exponential).
//! - Refusals use the variants and op names the neighbouring ops use: a
//!   non-finite operand, a shape that does not fit, a group count that does
//!   not divide the channels, an overflowing gradient and an over-budget
//!   output or axis table.
//!
//! The values against torch are in `sam_backward_golden.rs`.

use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_capacity, assert_nonfinite, assert_range, assert_shape, bits, SplitMix64};

const THREADS: [usize; 4] = [1, 2, 7, 18];

fn backend(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 33), threads)
        .unwrap()
        .with_numerics(numerics)
}

fn exact() -> CpuBackend {
    backend(1, Numerics::Exact)
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

fn wide(values: &[f32]) -> Vec<f64> {
    values.iter().copied().map(f64::from).collect()
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
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

fn assert_close(what: &str, got: &[f32], want: &[f64], tol: f64) {
    let err = normalized(got, want);
    assert!(err <= tol, "{what}: normalized error {err:e} > {tol:e}");
}

/// Central differences of `loss` at `point`, step `1e-6 * (1 + |x|)`.
fn central(point: &[f64], loss: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    let mut p = point.to_vec();
    (0..point.len())
        .map(|i| {
            let h = 1e-6 * (1.0 + point[i].abs());
            p[i] = point[i] + h;
            let up = loss(&p);
            p[i] = point[i] - h;
            let down = loss(&p);
            p[i] = point[i];
            (up - down) / (2.0 * h)
        })
        .collect()
}

fn dot(gy: &[f64], y: &[f64]) -> f64 {
    gy.iter().zip(y).map(|(g, y)| g * y).sum()
}

// ---- GroupNorm ----

/// torch's `F.group_norm` in `f64` throughout, `x` shaped `[n, c, spatial]`.
fn group_norm_f64(
    x: &[f64],
    w: &[f64],
    b: &[f64],
    [n, c, spatial]: [usize; 3],
    groups: usize,
    eps: f64,
) -> Vec<f64> {
    let per = c / groups;
    let len = per * spatial;
    let mut out = vec![0.0; x.len()];
    for img in 0..n {
        for g in 0..groups {
            let start = (img * c + g * per) * spatial;
            let vals = &x[start..start + len];
            let mean = vals.iter().sum::<f64>() / len as f64;
            let var = vals.iter().map(|q| (q - mean).powi(2)).sum::<f64>() / len as f64;
            let rstd = 1.0 / (var + eps).sqrt();
            for (k, q) in vals.iter().enumerate() {
                let ch = g * per + k / spatial;
                out[start + k] = (q - mean) * rstd * w[ch] + b[ch];
            }
        }
    }
    out
}

const FD_TOL: f64 = 1e-5;

#[test]
fn group_norm_backward_matches_central_differences() {
    let mut rng = SplitMix64(21);
    let be = exact();
    // Groups of one channel, several, and the whole image; one position, a
    // column, a plane; a batch of one; a large eps.
    for (shape, groups, eps) in [
        ([2usize, 8, 3, 5], 1usize, 1e-5f32),
        ([2, 8, 3, 5], 2, 1e-5),
        ([2, 8, 3, 5], 8, 1e-6),
        ([2, 16, 3, 5], 8, 1e-5),
        ([3, 6, 1, 1], 3, 1e-5),
        ([1, 4, 2, 2], 4, 1e-5),
        ([1, 2, 1, 3], 1, 1e-5),
        ([2, 4, 7, 1], 2, 1e-3),
    ] {
        let [n, c, h, w] = shape;
        let spatial = h * w;
        let x = rng.vec(numel(&shape), 3.0);
        let weight = rng.vec(c, 2.0);
        let bias = rng.vec(c, 1.0);
        let gy = rng.vec(numel(&shape), 1.0);
        let got = be
            .group_norm_backward(
                &t(&x, &shape),
                &t(&weight, &[c]),
                &t(&bias, &[c]),
                &t(&gy, &shape),
                groups,
                eps,
            )
            .unwrap();
        let f = |x: &[f64], w: &[f64], b: &[f64]| {
            group_norm_f64(x, w, b, [n, c, spatial], groups, f64::from(eps))
        };
        let (x64, w64, b64, g64) = (wide(&x), wide(&weight), wide(&bias), wide(&gy));
        let fx = central(&x64, |p| dot(&g64, &f(p, &w64, &b64)));
        let fw = central(&w64, |p| dot(&g64, &f(&x64, p, &b64)));
        let fb = central(&b64, |p| dot(&g64, &f(&x64, &w64, p)));
        let name = format!("group_norm {shape:?} groups {groups} eps {eps:e}");
        for (what, grad, want, want_shape) in [
            ("dx", &got.input, fx, &shape[..]),
            ("dw", &got.weight, fw, &[c][..]),
            ("db", &got.bias, fb, &[c][..]),
        ] {
            assert_eq!(grad.shape(), want_shape, "{name} {what} shape");
            assert_close(&format!("{name} {what}"), &v(grad), &want, FD_TOL);
        }
    }
}

#[test]
fn group_norm_with_one_group_over_a_vector_is_layer_norm_bit_for_bit() {
    // `[N, C, 1, 1]` with one group is `N` rows of `C`: the same statistics,
    // the same ascending sums and the same rounding as LayerNorm's backward.
    let mut rng = SplitMix64(22);
    let be = exact();
    for (n, c) in [(4usize, 6usize), (1, 9), (7, 2), (3, 33)] {
        let x = rng.vec(n * c, 3.0);
        let w = rng.vec(c, 2.0);
        let b = rng.vec(c, 1.0);
        let gy = rng.vec(n * c, 1.0);
        let eps = 1e-5;
        let gn = be
            .group_norm_backward(
                &t(&x, &[n, c, 1, 1]),
                &t(&w, &[c]),
                &t(&b, &[c]),
                &t(&gy, &[n, c, 1, 1]),
                1,
                eps,
            )
            .unwrap();
        let ln = be
            .layer_norm_backward(
                &t(&x, &[n, c]),
                &t(&w, &[c]),
                &t(&b, &[c]),
                &t(&gy, &[n, c]),
                eps,
            )
            .unwrap();
        assert_eq!(bits(&v(&gn.input)), bits(&v(&ln.input)), "dx {n}x{c}");
        assert_eq!(bits(&v(&gn.weight)), bits(&v(&ln.weight)), "dw {n}x{c}");
        assert_eq!(bits(&v(&gn.bias)), bits(&v(&ln.bias)), "db {n}x{c}");
    }
}

#[test]
fn group_norm_bias_gradient_is_the_channel_sum_of_the_incoming_gradient() {
    // `db[c] = sum over the batch and positions of gy`, one f64 sum rounded
    // once, whatever the groups and the input.
    let mut rng = SplitMix64(23);
    let be = exact();
    let shape = [3usize, 8, 4, 5];
    let x = rng.vec(numel(&shape), 3.0);
    let gy = rng.vec(numel(&shape), 1.0);
    for groups in [1usize, 2, 4, 8] {
        let got = be
            .group_norm_backward(
                &t(&x, &shape),
                &t(&rng.vec(8, 1.0), &[8]),
                &t(&rng.vec(8, 1.0), &[8]),
                &t(&gy, &shape),
                groups,
                1e-5,
            )
            .unwrap();
        let spatial = shape[2] * shape[3];
        let want: Vec<u32> = (0..8)
            .map(|c| {
                let mut sum = 0.0f64;
                for b in 0..shape[0] {
                    let at = (b * 8 + c) * spatial;
                    for &g in &gy[at..at + spatial] {
                        sum += f64::from(g);
                    }
                }
                (sum as f32).to_bits()
            })
            .collect();
        assert_eq!(bits(&v(&got.bias)), want, "groups {groups}");
    }
}

#[test]
fn group_norm_backward_does_not_depend_on_the_thread_count_or_the_numerics() {
    // 262144 values in 64 rows of 4096: several chunks at every thread count.
    let mut rng = SplitMix64(24);
    let shape = [8usize, 32, 32, 32];
    let x = t(&rng.vec(numel(&shape), 2.0), &shape);
    let gy = t(&rng.vec(numel(&shape), 1.0), &shape);
    let w = t(&rng.vec(32, 1.0), &[32]);
    let b = t(&rng.vec(32, 1.0), &[32]);
    let want = exact()
        .group_norm_backward(&x, &w, &b, &gy, 8, 1e-5)
        .unwrap();
    for numerics in [Numerics::Exact, Numerics::Fast] {
        for threads in THREADS {
            let got = backend(threads, numerics)
                .group_norm_backward(&x, &w, &b, &gy, 8, 1e-5)
                .unwrap();
            for (what, g, w) in [
                ("dx", &got.input, &want.input),
                ("dw", &got.weight, &want.weight),
                ("db", &got.bias, &want.bias),
            ] {
                assert_eq!(
                    bits(&v(g)),
                    bits(&v(w)),
                    "{what} at {threads} threads, {numerics:?}"
                );
            }
        }
    }
}

#[test]
fn group_norm_backward_refuses_what_it_cannot_mean() {
    const OP: &str = "group_norm_backward";
    let be = exact();
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 4, 1, 2]);
    let (w, b) = (t(&[1.0; 4], &[4]), t(&[0.0; 4], &[4]));
    let gy = t(&[0.5; 8], &[1, 4, 1, 2]);
    assert_eq!(
        be.group_norm_backward(&x, &w, &b, &gy, 2, 1e-5)
            .unwrap()
            .input
            .shape(),
        &[1, 4, 1, 2]
    );
    // 3 groups do not divide 4 channels; 0 groups is out of range.
    assert_shape(be.group_norm_backward(&x, &w, &b, &gy, 3, 1e-5));
    assert_range(be.group_norm_backward(&x, &w, &b, &gy, 0, 1e-5));
    // The gradient must be shaped like the input.
    assert_shape(be.group_norm_backward(&x, &w, &b, &t(&[0.5; 4], &[1, 4, 1, 1]), 2, 1e-5));
    // eps must be finite and above 0.
    assert_nonfinite(be.group_norm_backward(&x, &w, &b, &gy, 2, f32::NAN));
    assert_range(be.group_norm_backward(&x, &w, &b, &gy, 2, 0.0));
    // A NaN or infinity in any operand is refused under the backward's name.
    for bad in [f32::NAN, f32::INFINITY] {
        let mut poisoned = [1.0f32; 8];
        poisoned[5] = bad;
        let p = t(&poisoned, &[1, 4, 1, 2]);
        for result in [
            be.group_norm_backward(&p, &w, &b, &gy, 2, 1e-5),
            be.group_norm_backward(&x, &w, &b, &p, 2, 1e-5),
        ] {
            match result {
                Err(OjasError::NonFinite { op }) => assert_eq!(op, OP),
                other => panic!("expected NonFinite, got {other:?}"),
            }
        }
    }
    // An empty extent is refused, not divided by.
    assert_shape(be.group_norm_backward(
        &t(&[], &[1, 4, 0, 2]),
        &w,
        &b,
        &t(&[], &[1, 4, 0, 2]),
        2,
        1e-5,
    ));
}

#[test]
fn group_norm_backward_refuses_an_overflowing_gradient_and_a_small_budget() {
    // Four incoming gradients of 3e38 sum to 1.2e39 in the bias gradient,
    // which is not an `f32`: refused, not returned as infinity.
    let be = exact();
    let x = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let (w, b) = (t(&[1.0], &[1]), t(&[0.0], &[1]));
    match be.group_norm_backward(&x, &w, &b, &t(&[3e38; 4], &[1, 1, 2, 2]), 1, 1e-5) {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "group_norm_backward"),
        other => panic!("expected NonFinite, got {other:?}"),
    }
    // The three gradients are charged before they exist.
    let tight = CpuBackend::new(Budget::new(64));
    let (x, gy) = (t(&[0.5; 64], &[1, 4, 4, 4]), t(&[1.0; 64], &[1, 4, 4, 4]));
    assert_capacity(tight.group_norm_backward(
        &x,
        &t(&[1.0; 4], &[4]),
        &t(&[0.0; 4], &[4]),
        &gy,
        2,
        1e-5,
    ));
}

// ---- bilinear ----

/// torch's bilinear resample of `planes` images `[h, w]` to `[oh, ow]`, in
/// `f64` throughout, with the source index of `area_pixel_compute_source_index`.
fn bilinear_f64(
    x: &[f64],
    planes: usize,
    [h, w]: [usize; 2],
    [oh, ow]: [usize; 2],
    align: bool,
) -> Vec<f64> {
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
    let mut out = vec![0.0; planes * oh * ow];
    for plane in 0..planes {
        for yy in 0..oh {
            for xx in 0..ow {
                let (y0, y1, ly) = axis(h, oh, yy);
                let (x0, x1, lx) = axis(w, ow, xx);
                let at = |r: usize, c: usize| x[plane * h * w + r * w + c];
                out[(plane * oh + yy) * ow + xx] = (1.0 - ly)
                    * ((1.0 - lx) * at(y0, x0) + lx * at(y0, x1))
                    + ly * ((1.0 - lx) * at(y1, x0) + lx * at(y1, x1));
            }
        }
    }
    out
}

/// The gradient of `<resample(x), gy>` at every input pixel: the resample of
/// the unit image of that pixel, dotted with `gy`.
fn bilinear_adjoint_f64(
    gy: &[f64],
    planes: usize,
    size: [usize; 2],
    out_size: [usize; 2],
    align: bool,
) -> Vec<f64> {
    let [h, w] = size;
    let p = out_size[0] * out_size[1];
    let mut dx = vec![0.0; planes * h * w];
    for i in 0..h * w {
        // One image of the unit pixel: the same resample for every plane.
        let mut unit = vec![0.0; h * w];
        unit[i] = 1.0;
        let y = bilinear_f64(&unit, 1, size, out_size, align);
        for plane in 0..planes {
            dx[plane * h * w + i] = dot(&gy[plane * p..(plane + 1) * p], &y);
        }
    }
    dx
}

const BILINEAR_CASES: [([usize; 2], [usize; 2]); 12] = [
    ([4, 5], [9, 11]),
    ([9, 11], [4, 5]),
    ([5, 7], [9, 13]),
    ([5, 7], [3, 4]),
    ([1, 6], [1, 3]),
    ([1, 4], [1, 9]),
    ([6, 1], [3, 1]),
    ([3, 1], [7, 1]),
    ([6, 6], [1, 1]),
    ([1, 1], [4, 5]),
    ([8, 8], [8, 8]),
    ([7, 3], [14, 6]),
];

/// The backend builds its source indices in `f32` as torch does and the
/// reference in `f64`: for sizes up to 14 the two differ in a weight by at
/// most about 1e-6, so a gradient is held to 5e-6 of the largest one.
const BILINEAR_F64_TOL: f64 = 5e-6;

#[test]
fn bilinear_backward_is_the_adjoint_of_the_f64_resample() {
    let mut rng = SplitMix64(31);
    let be = exact();
    for ([h, w], [oh, ow]) in BILINEAR_CASES {
        for align in [false, true] {
            let planes = 3;
            let x = t(&rng.vec(planes * h * w, 1.0), &[1, planes, h, w]);
            let gy = rng.vec(planes * oh * ow, 1.0);
            let got = be
                .interpolate_bilinear_backward(&x, &t(&gy, &[1, planes, oh, ow]), align)
                .unwrap();
            let want = bilinear_adjoint_f64(&wide(&gy), planes, [h, w], [oh, ow], align);
            assert_eq!(got.shape(), &[1, planes, h, w]);
            assert_close(
                &format!("{h}x{w} to {oh}x{ow} align {align}"),
                &v(&got),
                &want,
                BILINEAR_F64_TOL,
            );
        }
    }
}

#[test]
fn bilinear_backward_batches_and_channels_are_independent_planes() {
    let mut rng = SplitMix64(32);
    let be = exact();
    let (n, c, h, w, oh, ow) = (2usize, 3usize, 5usize, 6usize, 8usize, 4usize);
    let gy = rng.vec(n * c * oh * ow, 1.0);
    let x = t(&rng.vec(n * c * h * w, 1.0), &[n, c, h, w]);
    let whole = be
        .interpolate_bilinear_backward(&x, &t(&gy, &[n, c, oh, ow]), true)
        .unwrap();
    let want = bilinear_adjoint_f64(&wide(&gy), n * c, [h, w], [oh, ow], true);
    assert_close("n x c planes", &v(&whole), &want, BILINEAR_F64_TOL);
    // Each plane alone gives the plane's bits.
    let plane = h * w;
    for p in 0..n * c {
        let one = be
            .interpolate_bilinear_backward(
                &t(&vec![0.0; plane], &[1, 1, h, w]),
                &t(&gy[p * oh * ow..(p + 1) * oh * ow], &[1, 1, oh, ow]),
                true,
            )
            .unwrap();
        assert_eq!(
            bits(&v(&one)),
            bits(&v(&whole)[p * plane..(p + 1) * plane]),
            "plane {p}"
        );
    }
}

#[test]
fn bilinear_backward_of_an_integer_halving_credits_each_input_a_quarter() {
    // Half-pixel centers: output (i, j) of a 4x4 -> 2x2 reads the 2x2 block
    // of inputs at rows 2i..2i+2, columns 2j..2j+2 with weight 1/4 each, so
    // every input gets a quarter of its output's gradient, exactly.
    let be = exact();
    let x = t(&[0.0; 16], &[1, 1, 4, 4]);
    let gy = [1.0f32, -2.0, 3.0, 8.0];
    let got = v(&be
        .interpolate_bilinear_backward(&x, &t(&gy, &[1, 1, 2, 2]), false)
        .unwrap());
    let want: Vec<f32> = (0..16)
        .map(|k| gy[(k / 8) * 2 + (k % 4) / 2] / 4.0)
        .collect();
    assert_eq!(bits(&got), bits(&want));
}

#[test]
fn bilinear_backward_at_the_same_size_is_the_identity() {
    let mut rng = SplitMix64(33);
    let be = exact();
    for align in [false, true] {
        let (h, w) = (5, 7);
        let gy = rng.vec(2 * h * w, 1.0);
        let got = be
            .interpolate_bilinear_backward(
                &t(&vec![0.0; 2 * h * w], &[1, 2, h, w]),
                &t(&gy, &[1, 2, h, w]),
                align,
            )
            .unwrap();
        assert_eq!(bits(&v(&got)), bits(&gy), "align {align}");
    }
}

#[test]
fn bilinear_backward_conserves_the_total_gradient() {
    // Each output is a convex combination of inputs, so the weights it gives
    // its inputs sum to 1 and `sum(dx) == sum(gy)`.
    let mut rng = SplitMix64(34);
    let be = exact();
    for ([h, w], [oh, ow]) in BILINEAR_CASES {
        for align in [false, true] {
            let gy = rng.vec(oh * ow, 1.0);
            let got = be
                .interpolate_bilinear_backward(
                    &t(&vec![0.0; h * w], &[1, 1, h, w]),
                    &t(&gy, &[1, 1, oh, ow]),
                    align,
                )
                .unwrap();
            let (have, want) = (
                v(&got).iter().map(|&q| f64::from(q)).sum::<f64>(),
                wide(&gy).iter().sum::<f64>(),
            );
            assert!(
                (have - want).abs() <= 1e-5 * (1.0 + want.abs()),
                "{h}x{w} to {oh}x{ow} align {align}: {have} != {want}"
            );
        }
    }
}

#[test]
fn bilinear_backward_does_not_depend_on_the_thread_count_or_the_numerics() {
    // 64 planes of 1600 input pixels: several chunks at every thread count.
    let mut rng = SplitMix64(35);
    let x_shape = [4usize, 16, 40, 40];
    let x = t(&vec![0.0; numel(&x_shape)], &x_shape);
    for (out, align) in [
        ([57usize, 63usize], false),
        ([57, 63], true),
        ([13, 17], true),
    ] {
        let gy = t(
            &rng.vec(4 * 16 * out[0] * out[1], 1.0),
            &[4, 16, out[0], out[1]],
        );
        let want = exact()
            .interpolate_bilinear_backward(&x, &gy, align)
            .unwrap();
        for numerics in [Numerics::Exact, Numerics::Fast] {
            for threads in THREADS {
                let got = backend(threads, numerics)
                    .interpolate_bilinear_backward(&x, &gy, align)
                    .unwrap();
                assert_eq!(
                    bits(&v(&got)),
                    bits(&v(&want)),
                    "{out:?} align {align} at {threads} threads, {numerics:?}"
                );
            }
        }
    }
}

#[test]
fn bilinear_backward_refuses_what_it_cannot_mean() {
    const OP: &str = "interpolate_bilinear_backward";
    let be = exact();
    let x = t(&[0.0; 12], &[1, 2, 2, 3]);
    let gy = t(&[1.0; 16], &[1, 2, 4, 2]);
    assert_eq!(
        be.interpolate_bilinear_backward(&x, &gy, false)
            .unwrap()
            .shape(),
        &[1, 2, 2, 3]
    );
    // The batch and channels of the gradient must be the input's.
    assert_shape(be.interpolate_bilinear_backward(&x, &t(&[1.0; 8], &[1, 1, 4, 2]), false));
    assert_shape(be.interpolate_bilinear_backward(&x, &t(&[1.0; 16], &[2, 1, 4, 2]), false));
    // Both must be images.
    assert_shape(be.interpolate_bilinear_backward(&x, &t(&[1.0; 16], &[2, 4, 2]), false));
    // A NaN or infinity in the gradient (or the input) is refused under the
    // backward's name.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut poisoned = [1.0f32; 16];
        poisoned[9] = bad;
        match be.interpolate_bilinear_backward(&x, &t(&poisoned, &[1, 2, 4, 2]), true) {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, OP),
            other => panic!("expected NonFinite, got {other:?}"),
        }
        let mut bad_x = [0.0f32; 12];
        bad_x[3] = bad;
        match be.interpolate_bilinear_backward(&t(&bad_x, &[1, 2, 2, 3]), &gy, true) {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, OP),
            other => panic!("expected NonFinite, got {other:?}"),
        }
    }
    // An empty extent is refused.
    assert_shape(be.interpolate_bilinear_backward(&x, &t(&[], &[1, 2, 0, 2]), false));
}

#[test]
fn a_budget_too_small_for_the_gradient_or_the_axis_tables_is_refused() {
    // A 1 x 4096 gradient (16 KiB) fits a 32 KiB cap, but the axis tables
    // built from its width are about 224 KiB of heap, which the cap must
    // refuse before they exist.
    let big = exact();
    let tight = CpuBackend::with_threads(Budget::new(32 * 1024), 1).unwrap();
    let x = t(&[1.0, 2.0], &[1, 1, 1, 2]);
    let gy = big
        .interpolate_bilinear_forward(&x, [1, 4096], false)
        .unwrap();
    let (x, gy) = (
        x.to_host(tight.budget()).unwrap(),
        gy.to_host(tight.budget()).unwrap(),
    );
    assert_capacity(tight.interpolate_bilinear_backward(&x, &gy, false));
    // A cap that holds the axis tables (about 736 bytes here) but not the
    // 256-byte gradient as well refuses the output before it is allocated.
    let tiny = CpuBackend::new(Budget::new(800));
    let x = t(&[0.0; 64], &[1, 1, 8, 8]);
    let gy = t(&[1.0; 16], &[1, 1, 4, 4]);
    assert_capacity(tiny.interpolate_bilinear_backward(&x, &gy, false));
}
