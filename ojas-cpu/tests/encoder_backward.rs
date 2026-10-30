//! The backward of LayerNorm (plain and across the channels of an image),
//! the exact GELU, a linear layer's bias, broadcast add and multiply, and
//! softmax on the CPU (ft-96baeabc, criteria 1, 3, 4 (the bias gradient), 5
//! and 6).
//!
//! - Gradcheck: each gradient is held to a central difference of an `f64`
//!   forward of the same op, on the loss `sum(gy * y)`.
//! - Torch: each gradient is compared with torch autograd's, in float32 and
//!   in float64, on the cases of `ojas-oracle/fixtures/encoder_bwd/golden.txt`
//!   (made by `ojas-oracle/python/encoder_backward_fixture.py`). The
//!   measured errors are printed (`cargo test -- --nocapture`).
//! - Relations are bit-exact: the bias gradient is the row sums of the
//!   incoming gradient, the linear's other gradients are `linear_backward`'s,
//!   a channel LayerNorm backward is a permute around the plain one.
//! - The bits do not depend on the thread count, and `Fast` and `Exact`
//!   agree (no backward here uses the exponential).
//! - A NaN or infinity in any operand, an empty extent, a shape that does not
//!   fit, an overflowing gradient and an over-budget output are refused with
//!   the variant every op uses, naming the backward.

use std::collections::HashMap;

use ojas_core::{erf_f64, AffineGrad, Backend, Budget, DType, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

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

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().copied().map(f32::to_bits).collect()
}

fn wide(values: &[f32]) -> Vec<f64> {
    values.iter().copied().map(f64::from).collect()
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Deterministic splitmix64, so no test needs a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        2.0 * ((self.next() >> 40) as f32 / (1u64 << 24) as f32) - 1.0
    }

    fn vec(&mut self, n: usize, scale: f32, shift: f32) -> Vec<f32> {
        (0..n).map(|_| scale * self.unit() + shift).collect()
    }

    fn tensor(&mut self, shape: &[usize], scale: f32, shift: f32) -> Tensor {
        t(&self.vec(numel(shape), scale, shift), shape)
    }
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

// ---- f64 forwards, the references the gradients are differenced from ----

fn layer_norm_f64(x: &[f64], w: &[f64], b: &[f64], dim: usize, eps: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(dim) {
        let mean = row.iter().sum::<f64>() / dim as f64;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / dim as f64;
        let rstd = 1.0 / (var + eps).sqrt();
        out.extend(
            row.iter()
                .zip(w.iter().zip(b))
                .map(|(v, (w, b))| (v - mean) * rstd * w + b),
        );
    }
    out
}

/// Channel LayerNorm of `[batch, channels, spatial]`.
fn channel_norm_f64(
    x: &[f64],
    w: &[f64],
    b: &[f64],
    [batch, channels, spatial]: [usize; 3],
    eps: f64,
) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for n in 0..batch {
        for s in 0..spatial {
            let at = |c: usize| (n * channels + c) * spatial + s;
            let mean = (0..channels).map(|c| x[at(c)]).sum::<f64>() / channels as f64;
            let var = (0..channels)
                .map(|c| (x[at(c)] - mean).powi(2))
                .sum::<f64>()
                / channels as f64;
            let rstd = 1.0 / (var + eps).sqrt();
            for c in 0..channels {
                out[at(c)] = (x[at(c)] - mean) * rstd * w[c] + b[c];
            }
        }
    }
    out
}

fn gelu_f64(x: f64) -> f64 {
    0.5 * x * (1.0 + erf_f64(x * std::f64::consts::FRAC_1_SQRT_2))
}

/// `d gelu / dx` by the closed form, with libm's exponential (a second path
/// from the one the backend uses).
fn gelu_slope_f64(x: f64) -> f64 {
    let cdf = 0.5 * (1.0 + erf_f64(x * std::f64::consts::FRAC_1_SQRT_2));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    cdf + x * pdf
}

fn linear_bias_f64(x: &[f64], w: &[f64], b: &[f64], inn: usize, out: usize) -> Vec<f64> {
    let mut y = Vec::new();
    for row in x.chunks_exact(inn) {
        for o in 0..out {
            let dot: f64 = row
                .iter()
                .zip(&w[o * inn..(o + 1) * inn])
                .map(|(a, b)| a * b)
                .sum();
            y.push(dot + b[o]);
        }
    }
    y
}

/// Row-major strides of `shape`, zero where the extent is 1, aligned to the
/// last axis of an output of rank `rank`.
fn stretched(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut strides = vec![0; rank];
    let mut run = 1;
    for k in 0..shape.len() {
        let axis = rank - 1 - k;
        let ext = shape[shape.len() - 1 - k];
        strides[axis] = if ext == 1 { 0 } else { run };
        run *= ext;
    }
    strides
}

/// `f(a, b)` over the broadcast of two shapes, in `f64`, with the output shape.
fn broadcast_f64(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
    f: impl Fn(f64, f64) -> f64,
) -> (Vec<f64>, Vec<usize>) {
    let rank = a_shape.len().max(b_shape.len());
    let pick = |k: usize| {
        let ad = a_shape
            .len()
            .checked_sub(rank - k)
            .map_or(1, |i| a_shape[i]);
        let bd = b_shape
            .len()
            .checked_sub(rank - k)
            .map_or(1, |i| b_shape[i]);
        ad.max(bd)
    };
    let out_shape: Vec<usize> = (0..rank).map(pick).collect();
    let (sa, sb) = (stretched(a_shape, rank), stretched(b_shape, rank));
    let mut out = Vec::new();
    for flat in 0..numel(&out_shape) {
        let (mut rest, mut ia, mut ib) = (flat, 0, 0);
        for axis in (0..rank).rev() {
            let c = rest % out_shape[axis];
            rest /= out_shape[axis];
            ia += c * sa[axis];
            ib += c * sb[axis];
        }
        out.push(f(a[ia], b[ib]));
    }
    (out, out_shape)
}

fn softmax_f64(x: &[f64], dim: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for row in x.chunks_exact(dim) {
        let max = row.iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v));
        let e: Vec<f64> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f64 = e.iter().sum();
        out.extend(e.iter().map(|v| v / sum));
    }
    out
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

// ---- gradcheck ----

const FD_TOL: f64 = 1e-5;

/// Checks one op's three gradients (`x`, `w`, `b`) against central differences
/// of its `f64` forward `f(x, w, b)`.
fn check_affine(
    what: &str,
    shapes: [&[usize]; 3],
    data: [&[f32]; 3],
    gy: &[f32],
    f: impl Fn(&[f64], &[f64], &[f64]) -> Vec<f64>,
    got: &AffineGrad,
) {
    let [x, w, b] = data.map(wide);
    let gy = wide(gy);
    let fx = central(&x, |p| dot(&gy, &f(p, &w, &b)));
    let fw = central(&w, |p| dot(&gy, &f(&x, p, &b)));
    let fb = central(&b, |p| dot(&gy, &f(&x, &w, p)));
    for (name, grad, want, shape) in [
        ("dx", &got.input, fx, shapes[0]),
        ("dw", &got.weight, fw, shapes[1]),
        ("db", &got.bias, fb, shapes[2]),
    ] {
        assert_eq!(grad.shape(), shape, "{what} {name} shape");
        assert_close(&format!("{what} {name}"), &v(grad), &want, FD_TOL);
    }
}

#[test]
fn layer_norm_backward_matches_central_differences() {
    let mut rng = Rng(11);
    let be = exact();
    for shape in [
        vec![5, 16],
        vec![2, 3, 24],
        vec![7],
        vec![1, 1],
        vec![4, 1, 33],
        vec![3, 3],
    ] {
        let dim = *shape.last().unwrap();
        // A negative eps is accepted while `var + eps` stays positive, as in
        // the forward; zero and 1e-12 are the small-eps edge on rows with
        // variance.
        for eps in [1e-5f32, 1e-6, 0.0, 1e-12, -1e-3] {
            if dim == 1 && eps <= 0.0 {
                continue;
            }
            let x = rng.vec(numel(&shape), 3.0, 0.0);
            let w = rng.vec(dim, 2.0, 0.0);
            let b = rng.vec(dim, 1.0, 0.0);
            let gy = rng.vec(numel(&shape), 1.0, 0.0);
            let got = be
                .layer_norm_backward(
                    &t(&x, &shape),
                    &t(&w, &[dim]),
                    &t(&b, &[dim]),
                    &t(&gy, &shape),
                    eps,
                )
                .unwrap();
            let f = |x: &[f64], w: &[f64], b: &[f64]| layer_norm_f64(x, w, b, dim, f64::from(eps));
            check_affine(
                &format!("layer_norm {shape:?} eps {eps:e}"),
                [&shape, &[dim], &[dim]],
                [&x, &w, &b],
                &gy,
                f,
                &got,
            );
        }
    }
}

#[test]
fn channel_layer_norm_backward_matches_central_differences() {
    let mut rng = Rng(12);
    let be = exact();
    for shape in [
        vec![2, 6, 3, 4],
        vec![3, 6],
        vec![2, 5, 7],
        vec![1, 8, 1, 1],
        vec![2, 1, 3],
    ] {
        let (batch, channels) = (shape[0], shape[1]);
        let spatial = numel(&shape[2..]);
        for eps in [1e-6f32, 1e-5, 0.0] {
            if channels == 1 && eps <= 0.0 {
                continue;
            }
            let x = rng.vec(numel(&shape), 3.0, 0.0);
            let w = rng.vec(channels, 2.0, 0.0);
            let b = rng.vec(channels, 1.0, 0.0);
            let gy = rng.vec(numel(&shape), 1.0, 0.0);
            let got = be
                .channel_layer_norm_backward(
                    &t(&x, &shape),
                    &t(&w, &[channels]),
                    &t(&b, &[channels]),
                    &t(&gy, &shape),
                    eps,
                )
                .unwrap();
            let f = |x: &[f64], w: &[f64], b: &[f64]| {
                channel_norm_f64(x, w, b, [batch, channels, spatial], f64::from(eps))
            };
            check_affine(
                &format!("channel_layer_norm {shape:?} eps {eps:e}"),
                [&shape, &[channels], &[channels]],
                [&x, &w, &b],
                &gy,
                f,
                &got,
            );
        }
    }
}

#[test]
fn gelu_erf_backward_matches_central_differences() {
    let mut rng = Rng(13);
    let mut x = rng.vec(48, 4.0, 0.0);
    x.extend([
        0.0, 1e-3, -1e-3, 0.5, -0.5, 1.0, -1.0, 2.5, -2.5, 5.9, -5.9, 7.5, -7.5,
    ]);
    let gy = rng.vec(x.len(), 1.0, 0.0);
    let got = exact()
        .gelu_erf_backward(&t(&x, &[x.len()]), &t(&gy, &[x.len()]))
        .unwrap();
    let want = central(&wide(&x), |p| {
        dot(
            &wide(&gy),
            &p.iter().map(|&x| gelu_f64(x)).collect::<Vec<_>>(),
        )
    });
    assert_eq!(got.shape(), &[x.len()]);
    assert_close("gelu_erf_backward", &v(&got), &want, FD_TOL);
}

#[test]
fn linear_bias_backward_matches_central_differences() {
    let mut rng = Rng(14);
    let be = exact();
    for (lead, inn, out) in [
        (vec![4], 7, 5),
        (vec![2, 3], 7, 4),
        (vec![1], 9, 6),
        (vec![5], 1, 1),
    ] {
        let x_shape: Vec<usize> = lead.iter().copied().chain([inn]).collect();
        let y_shape: Vec<usize> = lead.iter().copied().chain([out]).collect();
        let x = rng.vec(numel(&x_shape), 1.0, 0.0);
        let w = rng.vec(out * inn, 0.5, 0.0);
        let b = rng.vec(out, 1.0, 0.0);
        let gy = rng.vec(numel(&y_shape), 1.0, 0.0);
        let got = be
            .linear_bias_backward(
                &t(&x, &x_shape),
                &t(&w, &[out, inn]),
                &t(&b, &[out]),
                &t(&gy, &y_shape),
            )
            .unwrap();
        let f = |x: &[f64], w: &[f64], b: &[f64]| linear_bias_f64(x, w, b, inn, out);
        check_affine(
            &format!("linear_bias {x_shape:?} -> {out}"),
            [&x_shape, &[out, inn], &[out]],
            [&x, &w, &b],
            &gy,
            f,
            &got,
        );
    }
}

/// Shape pairs: a bias row, size-1 axes on both operands (stretched along
/// non-adjacent axes), scalars, a column, equal shapes, a leading axis one
/// operand lacks, and the largest rank.
fn broadcast_shapes_under_test() -> Vec<(Vec<usize>, Vec<usize>)> {
    let pairs: [(&[usize], &[usize]); 11] = [
        (&[3, 8], &[8]),
        (&[2, 1, 4], &[1, 3, 1]),
        (&[], &[5]),
        (&[1], &[4, 1]),
        (&[3, 4], &[3, 4]),
        (&[6, 2, 3], &[2, 1]),
        (&[2, 1, 3, 1], &[1, 4, 1, 5]),
        (&[5, 1], &[1, 5]),
        (&[], &[]),
        (&[1, 1, 1, 1, 1, 1, 1, 3], &[2, 1, 1, 1, 1, 1, 1, 1]),
        (&[4], &[1]),
    ];
    pairs
        .iter()
        .map(|(a, b)| (a.to_vec(), b.to_vec()))
        .collect()
}

#[test]
fn broadcast_backward_matches_central_differences() {
    let mut rng = Rng(15);
    let be = exact();
    for (a_shape, b_shape) in broadcast_shapes_under_test() {
        let a = rng.vec(numel(&a_shape), 2.0, 0.0);
        let b = rng.vec(numel(&b_shape), 2.0, 0.0);
        let probe = broadcast_f64(&wide(&a), &a_shape, &wide(&b), &b_shape, |x, y| x + y);
        let out_shape = probe.1;
        let gy = rng.vec(numel(&out_shape), 1.0, 0.0);
        let (ta, tb, tg) = (t(&a, &a_shape), t(&b, &b_shape), t(&gy, &out_shape));
        let name = format!("{a_shape:?} with {b_shape:?}");
        for (op, mul) in [("add", false), ("mul", true)] {
            let (ga, gb) = if mul {
                be.broadcast_mul_backward(&ta, &tb, &tg).unwrap()
            } else {
                be.broadcast_add_backward(&ta, &tb, &tg).unwrap()
            };
            let f = |a: &[f64], b: &[f64]| {
                broadcast_f64(
                    a,
                    &a_shape,
                    b,
                    &b_shape,
                    |x, y| if mul { x * y } else { x + y },
                )
                .0
            };
            let want_a = central(&wide(&a), |p| dot(&wide(&gy), &f(p, &wide(&b))));
            let want_b = central(&wide(&b), |p| dot(&wide(&gy), &f(&wide(&a), p)));
            assert_eq!(ga.shape(), a_shape.as_slice(), "{op} {name} da shape");
            assert_eq!(gb.shape(), b_shape.as_slice(), "{op} {name} db shape");
            assert_close(&format!("{op} {name} da"), &v(&ga), &want_a, FD_TOL);
            assert_close(&format!("{op} {name} db"), &v(&gb), &want_b, FD_TOL);
        }
    }
}

#[test]
fn softmax_backward_matches_central_differences() {
    let mut rng = Rng(16);
    let be = exact();
    for shape in [
        vec![4, 10],
        vec![2, 3, 7],
        vec![5, 1],
        vec![3, 513],
        vec![9],
    ] {
        let dim = *shape.last().unwrap();
        let x = rng.vec(numel(&shape), 3.0, 0.0);
        let gy = rng.vec(numel(&shape), 1.0, 0.0);
        let y = be.softmax_forward(&t(&x, &shape)).unwrap();
        let got = be.softmax_backward(&y, &t(&gy, &shape)).unwrap();
        let want = central(&wide(&x), |p| dot(&wide(&gy), &softmax_f64(p, dim)));
        assert_eq!(got.shape(), shape.as_slice());
        assert_close(&format!("softmax {shape:?}"), &v(&got), &want, FD_TOL);
    }
}

// ---- adversarial values ----

/// The LayerNorm input gradient in closed form, in `f64`, for a row set: a
/// second path from the kernel's (statistics by naive sums).
fn layer_norm_grad_f64(x: &[f64], w: &[f64], gy: &[f64], dim: usize, eps: f64) -> Vec<f64> {
    let mut out = Vec::new();
    for (row, g) in x.chunks_exact(dim).zip(gy.chunks_exact(dim)) {
        let mean = row.iter().sum::<f64>() / dim as f64;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / dim as f64;
        let rstd = 1.0 / (var + eps).sqrt();
        let xhat: Vec<f64> = row.iter().map(|v| (v - mean) * rstd).collect();
        let dxhat: Vec<f64> = g.iter().zip(w).map(|(g, w)| g * w).collect();
        let m1 = dxhat.iter().sum::<f64>() / dim as f64;
        let m2 = dxhat.iter().zip(&xhat).map(|(a, b)| a * b).sum::<f64>() / dim as f64;
        out.extend(
            xhat.iter()
                .zip(&dxhat)
                .map(|(xh, dh)| rstd * (dh - m1 - xh * m2)),
        );
    }
    out
}

#[test]
fn layer_norm_backward_keeps_its_accuracy_on_rows_with_a_large_mean() {
    // SAM's MLP width and a mean 1000x the spread; and a mean of 1e5.
    let be = exact();
    for (dim, offset) in [(4736usize, 1000.0f32), (257, 1e5)] {
        let rows = 3;
        let mut rng = Rng(21);
        let x = rng.vec(rows * dim, 1.7, offset);
        let w = rng.vec(dim, 1.0, 1.0);
        let b = vec![0.0f32; dim];
        let gy = rng.vec(rows * dim, 1.0, 0.0);
        let got = be
            .layer_norm_backward(
                &t(&x, &[rows, dim]),
                &t(&w, &[dim]),
                &t(&b, &[dim]),
                &t(&gy, &[rows, dim]),
                1e-5,
            )
            .unwrap();
        let want = layer_norm_grad_f64(&wide(&x), &wide(&w), &wide(&gy), dim, 1e-5);
        assert_close(
            &format!("dx at offset {offset} width {dim}"),
            &v(&got.input),
            &want,
            1e-5,
        );
    }
}

#[test]
fn layer_norm_backward_of_constant_rows_follows_the_eps_edge() {
    let be = exact();
    let (x, w, b) = (
        t(&[3.0; 8], &[2, 4]),
        t(&[2.0, 3.0, 4.0, 5.0], &[4]),
        t(&[0.0; 4], &[4]),
    );
    let gy = t(&[1.0, -2.0, 0.5, 4.0, 0.25, 1.5, -1.0, 2.0], &[2, 4]);
    // A zero deviation and eps 0 (or less): `var + eps` is not positive.
    for eps in [0.0f32, -1e-6] {
        match be.layer_norm_backward(&x, &w, &b, &gy, eps) {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, "layer_norm_backward"),
            other => panic!("eps {eps}: expected NonFinite, got {other:?}"),
        }
    }
    // The smallest normal eps: rstd is 8.6e18 and `xhat` is 0, so the gradient
    // is `rstd * (g * w - mean(g * w))`.
    let eps = f32::MIN_POSITIVE;
    let got = be.layer_norm_backward(&x, &w, &b, &gy, eps).unwrap();
    let rstd = 1.0 / f64::from(eps).sqrt();
    let want: Vec<f64> = [[1.0, -2.0, 0.5, 4.0], [0.25, 1.5, -1.0, 2.0]]
        .iter()
        .flat_map(|g: &[f64; 4]| {
            let dxhat: Vec<f64> = g
                .iter()
                .zip([2.0, 3.0, 4.0, 5.0])
                .map(|(g, w)| g * w)
                .collect();
            let mean = dxhat.iter().sum::<f64>() / 4.0;
            dxhat.into_iter().map(move |d| rstd * (d - mean))
        })
        .collect();
    assert_close(
        "dx of constant rows at the smallest eps",
        &v(&got.input),
        &want,
        1e-6,
    );
    // `xhat` is 0, so the weight gradient is 0 and the bias gradient is the column sums.
    assert_eq!(v(&got.weight), [0.0; 4]);
    assert_eq!(v(&got.bias), [1.25, -0.5, -0.5, 6.0]);
}

#[test]
fn gelu_erf_backward_covers_the_saturation_edge_and_the_far_tails() {
    let be = exact();
    let xs = [
        0.0f32,
        -0.0,
        1e-45,
        -1e-45,
        f32::MIN_POSITIVE,
        0.5,
        -0.5,
        4.9,
        -4.9,
        8.4,
        -8.4,
        8.5,
        -8.5,
        8.6,
        -8.6,
        14.4,
        -14.4,
        14.5,
        -14.5,
        15.0,
        -15.0,
        20.0,
        -20.0,
        30.0,
        -30.0,
        1e30,
        -1e30,
        f32::MAX,
        f32::MIN,
    ];
    let n = xs.len();
    let got = v(&be
        .gelu_erf_backward(&t(&xs, &[n]), &t(&vec![1.0; n], &[n]))
        .unwrap());
    for (&x, &g) in xs.iter().zip(&got) {
        let want = gelu_slope_f64(f64::from(x));
        assert!(
            (f64::from(g) - want).abs() <= 1e-6 * want.abs() + 1e-13,
            "x = {x:e}: {g:e} vs {want:e}"
        );
    }
    // Far positive is the gradient itself; from -15 on `-x * phi(x)` is below
    // the smallest f32 and the slope is zero.
    for (&x, &g) in xs.iter().zip(&got) {
        if x >= 9.0 {
            assert_eq!(g, 1.0, "x = {x:e}");
        }
        if x <= -15.0 {
            assert_eq!(g, 0.0, "x = {x:e}");
        }
    }
    // Where torch's f32 exponential has underflowed (|x| over 14.4) the term
    // still scales a gradient near the f32 maximum.
    for (x, dy) in [(-15.0f32, 1e38f32), (-18.0, 3e38), (-18.5, f32::MAX)] {
        let got = v(&be
            .gelu_erf_backward(&t(&[x], &[1]), &t(&[dy], &[1]))
            .unwrap())[0];
        let want = f64::from(dy) * gelu_slope_f64(f64::from(x));
        assert!(
            (f64::from(got) - want).abs() <= 1e-6 * want.abs(),
            "x = {x}: {got:e} vs {want:e}"
        );
    }
}

// ---- relations ----

#[test]
fn linear_bias_backward_is_linear_backward_and_the_row_sums() {
    let mut rng = Rng(31);
    let be = exact();
    for (lead, inn, out) in [(vec![6], 5, 3), (vec![2, 4], 9, 7), (vec![1], 4, 2)] {
        let x_shape: Vec<usize> = lead.iter().copied().chain([inn]).collect();
        let y_shape: Vec<usize> = lead.iter().copied().chain([out]).collect();
        let x = rng.tensor(&x_shape, 1.0, 0.0);
        let w = rng.tensor(&[out, inn], 0.5, 0.0);
        let b = rng.tensor(&[out], 1.0, 0.0);
        let gy = rng.tensor(&y_shape, 1.0, 0.0);
        let got = be.linear_bias_backward(&x, &w, &b, &gy).unwrap();
        let (gx, gw) = be.linear_backward(&x, &w, &gy).unwrap();
        assert_eq!(bits(&v(&got.input)), bits(&v(&gx)));
        assert_eq!(bits(&v(&got.weight)), bits(&v(&gw)));
        // One f64 sum a column, ascending over the rows, rounded once.
        let g = v(&gy);
        let want: Vec<f32> = (0..out)
            .map(|o| {
                g.chunks_exact(out)
                    .map(|row| f64::from(row[o]))
                    .sum::<f64>() as f32
            })
            .collect();
        assert_eq!(bits(&v(&got.bias)), bits(&want));
        assert_eq!(got.bias.shape(), &[out]);
    }
}

#[test]
fn linear_bias_backward_takes_bf16_operands_like_linear_backward() {
    let mut rng = Rng(32);
    let be = exact();
    let (x, w) = (rng.tensor(&[4, 6], 1.0, 0.0), rng.tensor(&[5, 6], 0.5, 0.0));
    let (b, gy) = (rng.tensor(&[5], 1.0, 0.0), rng.tensor(&[4, 5], 1.0, 0.0));
    let (xh, wh, gh) = (
        be.to_bf16(&x).unwrap(),
        be.to_bf16(&w).unwrap(),
        be.to_bf16(&gy).unwrap(),
    );
    assert_eq!(gh.dtype(), DType::Bf16);
    let got = be.linear_bias_backward(&xh, &wh, &b, &gh).unwrap();
    let widen = |h: &Tensor| be.to_f32(h).unwrap();
    let want = be
        .linear_bias_backward(&widen(&xh), &widen(&wh), &b, &widen(&gh))
        .unwrap();
    for (g, w) in [
        (&got.input, &want.input),
        (&got.weight, &want.weight),
        (&got.bias, &want.bias),
    ] {
        assert_eq!(g.dtype(), DType::F32);
        assert_eq!(bits(&v(g)), bits(&v(w)));
    }
    // The bias is F32 only, as for the forward.
    let bh = be.to_bf16(&b).unwrap();
    match be.linear_bias_backward(&xh, &wh, &bh, &gh) {
        Err(OjasError::Dtype { op, .. }) => assert_eq!(op, "linear_bias_backward"),
        other => panic!("expected Dtype, got {other:?}"),
    }
}

#[test]
fn channel_layer_norm_backward_is_a_permute_around_layer_norm_backward() {
    let mut rng = Rng(33);
    let be = exact();
    for shape in [
        vec![2, 6, 3, 4],
        vec![3, 5],
        vec![2, 4, 9],
        vec![1, 7, 1, 1],
    ] {
        let (batch, channels) = (shape[0], shape[1]);
        let spatial = numel(&shape[2..]);
        let x = rng.vec(numel(&shape), 3.0, 0.5);
        let gy = rng.vec(numel(&shape), 1.0, 0.0);
        let (w, b) = (
            rng.tensor(&[channels], 1.0, 1.0),
            rng.tensor(&[channels], 1.0, 0.0),
        );
        let direct = be
            .channel_layer_norm_backward(&t(&x, &shape), &w, &b, &t(&gy, &shape), 1e-6)
            .unwrap();
        // [batch, channels, spatial] -> [batch, spatial, channels] and back.
        let to_last = |data: &[f32]| {
            let mut out = vec![0.0f32; data.len()];
            for n in 0..batch {
                for c in 0..channels {
                    for s in 0..spatial {
                        out[(n * spatial + s) * channels + c] =
                            data[(n * channels + c) * spatial + s];
                    }
                }
            }
            out
        };
        let to_first = |data: &[f32]| {
            let mut out = vec![0.0f32; data.len()];
            for n in 0..batch {
                for c in 0..channels {
                    for s in 0..spatial {
                        out[(n * channels + c) * spatial + s] =
                            data[(n * spatial + s) * channels + c];
                    }
                }
            }
            out
        };
        let last = [batch * spatial, channels];
        let via = be
            .layer_norm_backward(
                &t(&to_last(&x), &last),
                &w,
                &b,
                &t(&to_last(&gy), &last),
                1e-6,
            )
            .unwrap();
        assert_eq!(
            bits(&v(&direct.input)),
            bits(&to_first(&v(&via.input))),
            "dx {shape:?}"
        );
        assert_eq!(
            bits(&v(&direct.weight)),
            bits(&v(&via.weight)),
            "dw {shape:?}"
        );
        assert_eq!(bits(&v(&direct.bias)), bits(&v(&via.bias)), "db {shape:?}");
    }
}

#[test]
fn broadcast_backward_of_equal_shapes_is_a_copy_and_a_product() {
    let mut rng = Rng(34);
    let be = exact();
    let shape = [3, 5];
    let (a, b, gy) = (
        rng.tensor(&shape, 2.0, 0.0),
        rng.tensor(&shape, 2.0, 0.0),
        rng.tensor(&shape, 1.0, 0.0),
    );
    let (ga, gb) = be.broadcast_add_backward(&a, &b, &gy).unwrap();
    assert_eq!(bits(&v(&ga)), bits(&v(&gy)));
    assert_eq!(bits(&v(&gb)), bits(&v(&gy)));
    let (ga, gb) = be.broadcast_mul_backward(&a, &b, &gy).unwrap();
    let (want_a, want_b) = be.mul_backward(&a, &b, &gy).unwrap();
    assert_eq!(bits(&v(&ga)), bits(&v(&want_a)));
    assert_eq!(bits(&v(&gb)), bits(&v(&want_b)));
}

#[test]
fn broadcast_backward_sums_each_stretched_axis_once_in_output_order() {
    let be = exact();
    // a [2, 1, 3] against b [4, 1]: the output is [2, 4, 3]; da sums the 4
    // axis, db sums the 2 and 3 axes. The values are exact in f32, so any
    // order gives the same sum and this pins the index arithmetic.
    let gy: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]);
    let b = t(&[10.0, 20.0, 30.0, 40.0], &[4, 1]);
    let (ga, gb) = be
        .broadcast_add_backward(&a, &b, &t(&gy, &[2, 4, 3]))
        .unwrap();
    let mut want_a = [0.0f32; 6];
    let mut want_b = [0.0f32; 4];
    for n in 0..2 {
        for j in 0..4 {
            for s in 0..3 {
                let g = gy[(n * 4 + j) * 3 + s];
                want_a[n * 3 + s] += g;
                want_b[j] += g;
            }
        }
    }
    assert_eq!(v(&ga), want_a);
    assert_eq!(v(&gb), want_b);
    let (ga, gb) = be
        .broadcast_mul_backward(&a, &b, &t(&gy, &[2, 4, 3]))
        .unwrap();
    let av = v(&a);
    let bv = v(&b);
    let mut want_a = [0.0f32; 6];
    let mut want_b = [0.0f32; 4];
    for n in 0..2 {
        for j in 0..4 {
            for s in 0..3 {
                let g = gy[(n * 4 + j) * 3 + s];
                want_a[n * 3 + s] += g * bv[j];
                want_b[j] += g * av[n * 3 + s];
            }
        }
    }
    assert_eq!(v(&ga), want_a);
    assert_eq!(v(&gb), want_b);
}

// ---- thread count and numerics ----

/// Runs `op` on every thread count under both contracts and demands the bits
/// of each output equal those of one thread under `Exact`.
fn assert_same_bits_everywhere(name: &str, op: impl Fn(&CpuBackend) -> Vec<Tensor>) {
    let want: Vec<Vec<u32>> = op(&exact()).iter().map(|t| bits(&v(t))).collect();
    for threads in THREADS {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let got: Vec<Vec<u32>> = op(&backend(threads, numerics))
                .iter()
                .map(|t| bits(&v(t)))
                .collect();
            assert_eq!(got, want, "{name}: {threads} threads, {numerics:?}");
        }
    }
}

#[test]
fn every_backward_is_bit_identical_across_threads_and_numerics() {
    // Outputs over 2 MiB, so the row passes really split across threads.
    let mut rng = Rng(41);
    let (rows, dim) = (1100usize, 512usize);
    let (x, gy) = (
        rng.tensor(&[rows, dim], 3.0, 0.3),
        rng.tensor(&[rows, dim], 1.0, 0.0),
    );
    let (w, b) = (rng.tensor(&[dim], 1.0, 1.0), rng.tensor(&[dim], 1.0, 0.0));
    assert_same_bits_everywhere("layer_norm_backward", |be| {
        let g = be.layer_norm_backward(&x, &w, &b, &gy, 1e-5).unwrap();
        vec![g.input, g.weight, g.bias]
    });
    assert_same_bits_everywhere("gelu_erf_backward", |be| {
        vec![be.gelu_erf_backward(&x, &gy).unwrap()]
    });
    // The forward's exponential differs between the contracts, so the saved
    // output is made once.
    let y = exact().softmax_forward(&x).unwrap();
    assert_same_bits_everywhere("softmax_backward", |be| {
        vec![be.softmax_backward(&y, &gy).unwrap()]
    });
    let col = rng.tensor(&[rows, 1], 2.0, 0.0);
    assert_same_bits_everywhere("broadcast_add_backward", |be| {
        let (ga, gb) = be.broadcast_add_backward(&x, &w, &gy).unwrap();
        let (gc, gd) = be.broadcast_add_backward(&x, &col, &gy).unwrap();
        vec![ga, gb, gc, gd]
    });
    assert_same_bits_everywhere("broadcast_mul_backward", |be| {
        let (ga, gb) = be.broadcast_mul_backward(&x, &w, &gy).unwrap();
        let (gc, gd) = be.broadcast_mul_backward(&x, &col, &gy).unwrap();
        vec![ga, gb, gc, gd]
    });
    let (img_shape, channels) = ([4usize, 64, 48, 48], 64usize);
    let img = rng.tensor(&img_shape, 3.0, 0.3);
    let img_gy = rng.tensor(&img_shape, 1.0, 0.0);
    let (cw, cb) = (
        rng.tensor(&[channels], 1.0, 1.0),
        rng.tensor(&[channels], 1.0, 0.0),
    );
    assert_same_bits_everywhere("channel_layer_norm_backward", |be| {
        let g = be
            .channel_layer_norm_backward(&img, &cw, &cb, &img_gy, 1e-6)
            .unwrap();
        vec![g.input, g.weight, g.bias]
    });
}

#[test]
fn linear_bias_backward_is_bit_identical_across_threads() {
    // The bias gradient does not depend on the contract; the product's
    // gradients do under Fast on macOS (one Accelerate call), so those are
    // held across threads under Exact.
    let mut rng = Rng(42);
    let (rows, inn, out) = (700usize, 96usize, 80usize);
    let (x, w) = (
        rng.tensor(&[rows, inn], 1.0, 0.0),
        rng.tensor(&[out, inn], 0.5, 0.0),
    );
    let (b, gy) = (
        rng.tensor(&[out], 1.0, 0.0),
        rng.tensor(&[rows, out], 1.0, 0.0),
    );
    let want = exact().linear_bias_backward(&x, &w, &b, &gy).unwrap();
    for threads in THREADS {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let got = backend(threads, numerics)
                .linear_bias_backward(&x, &w, &b, &gy)
                .unwrap();
            assert_eq!(
                bits(&v(&got.bias)),
                bits(&v(&want.bias)),
                "{threads} threads {numerics:?}"
            );
            if numerics == Numerics::Exact {
                assert_eq!(
                    bits(&v(&got.input)),
                    bits(&v(&want.input)),
                    "dx, {threads} threads"
                );
                assert_eq!(
                    bits(&v(&got.weight)),
                    bits(&v(&want.weight)),
                    "dw, {threads} threads"
                );
            }
        }
    }
}

// ---- refusals ----

/// One backward under test: how to build its operands in a budget, and how to
/// call it.
struct Op {
    name: &'static str,
    args: fn(&Budget) -> Vec<Tensor>,
    call: fn(&CpuBackend, &[Tensor]) -> Result<(), OjasError>,
}

fn tb(budget: &Budget, seed: u64, shape: &[usize]) -> Tensor {
    let data = Rng(seed).vec(numel(shape), 1.0, 0.0);
    Tensor::from_f32(&data, shape, budget).unwrap()
}

fn ops() -> Vec<Op> {
    vec![
        Op {
            name: "layer_norm_backward",
            args: |b| {
                vec![
                    tb(b, 1, &[3, 8]),
                    tb(b, 2, &[8]),
                    tb(b, 3, &[8]),
                    tb(b, 4, &[3, 8]),
                ]
            },
            call: |be, a| {
                be.layer_norm_backward(&a[0], &a[1], &a[2], &a[3], 1e-5)
                    .map(drop)
            },
        },
        Op {
            name: "channel_layer_norm_backward",
            args: |b| {
                vec![
                    tb(b, 5, &[2, 4, 3, 3]),
                    tb(b, 6, &[4]),
                    tb(b, 7, &[4]),
                    tb(b, 8, &[2, 4, 3, 3]),
                ]
            },
            call: |be, a| {
                be.channel_layer_norm_backward(&a[0], &a[1], &a[2], &a[3], 1e-5)
                    .map(drop)
            },
        },
        Op {
            name: "gelu_erf_backward",
            args: |b| vec![tb(b, 9, &[3, 8]), tb(b, 10, &[3, 8])],
            call: |be, a| be.gelu_erf_backward(&a[0], &a[1]).map(drop),
        },
        Op {
            name: "softmax_backward",
            args: |b| vec![tb(b, 11, &[3, 8]), tb(b, 12, &[3, 8])],
            call: |be, a| be.softmax_backward(&a[0], &a[1]).map(drop),
        },
        Op {
            name: "broadcast_add_backward",
            args: |b| vec![tb(b, 13, &[3, 8]), tb(b, 14, &[8]), tb(b, 15, &[3, 8])],
            call: |be, a| be.broadcast_add_backward(&a[0], &a[1], &a[2]).map(drop),
        },
        Op {
            name: "broadcast_mul_backward",
            args: |b| vec![tb(b, 16, &[3, 8]), tb(b, 17, &[8]), tb(b, 18, &[3, 8])],
            call: |be, a| be.broadcast_mul_backward(&a[0], &a[1], &a[2]).map(drop),
        },
        Op {
            name: "linear_bias_backward",
            args: |b| {
                vec![
                    tb(b, 19, &[3, 8]),
                    tb(b, 20, &[5, 8]),
                    tb(b, 21, &[5]),
                    tb(b, 22, &[3, 5]),
                ]
            },
            call: |be, a| {
                be.linear_bias_backward(&a[0], &a[1], &a[2], &a[3])
                    .map(drop)
            },
        },
    ]
}

fn big() -> Budget {
    Budget::new(1 << 30)
}

fn replaced(args: &[Tensor], at: usize, data: &[f32], shape: &[usize]) -> Vec<Tensor> {
    let mut out = args.to_vec();
    out[at] = Tensor::from_f32(data, shape, &big()).unwrap();
    out
}

#[test]
fn a_nan_or_infinity_in_any_operand_is_refused_naming_the_backward() {
    let be = exact();
    for op in ops() {
        let args = (op.args)(&big());
        for at in 0..args.len() {
            for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut data = v(&args[at]);
                let last = data.len() - 1;
                data[last] = poison;
                let bad = replaced(&args, at, &data, args[at].shape());
                match (op.call)(&be, &bad) {
                    Err(OjasError::NonFinite { op: named }) => assert_eq!(named, op.name),
                    other => panic!(
                        "{} operand {at} = {poison}: expected NonFinite, got {other:?}",
                        op.name
                    ),
                }
            }
        }
        (op.call)(&be, &args).unwrap();
    }
}

#[test]
fn an_empty_extent_is_refused_as_an_empty_tensor() {
    let be = exact();
    // Each op's first operand with a zero leading extent, and the others
    // reshaped to match where the op needs it.
    let empty: Vec<(&str, Vec<Vec<usize>>)> = vec![
        (
            "layer_norm_backward",
            vec![vec![0, 8], vec![8], vec![8], vec![0, 8]],
        ),
        (
            "channel_layer_norm_backward",
            vec![vec![0, 4, 3, 3], vec![4], vec![4], vec![0, 4, 3, 3]],
        ),
        ("gelu_erf_backward", vec![vec![0, 8], vec![0, 8]]),
        ("softmax_backward", vec![vec![0, 8], vec![0, 8]]),
        (
            "broadcast_add_backward",
            vec![vec![0, 8], vec![8], vec![0, 8]],
        ),
        (
            "broadcast_mul_backward",
            vec![vec![3, 8], vec![0, 1], vec![0, 8]],
        ),
        (
            "linear_bias_backward",
            vec![vec![0, 8], vec![5, 8], vec![5], vec![0, 5]],
        ),
    ];
    let budget = big();
    for op in ops() {
        let (_, shapes) = empty.iter().find(|(name, _)| *name == op.name).unwrap();
        let args: Vec<Tensor> = shapes
            .iter()
            .map(|s| Tensor::zeros(s, DType::F32, &budget).unwrap())
            .collect();
        match (op.call)(&be, &args) {
            Err(OjasError::Shape { op: named, detail }) => {
                assert_eq!(named, op.name);
                assert!(detail.contains("empty tensor"), "{}: {detail}", op.name);
            }
            other => panic!("{}: expected Shape, got {other:?}", op.name),
        }
    }
}

fn shape_refused(r: Result<(), OjasError>, name: &str, what: &str) {
    match r {
        Err(OjasError::Shape { op, detail }) => {
            assert_eq!(op, name, "{what}");
            assert!(!detail.is_empty());
        }
        other => panic!("{name}: {what}: expected Shape, got {other:?}"),
    }
}

#[test]
fn shapes_that_do_not_fit_are_refused() {
    let be = exact();
    let b = big();
    let z = |shape: &[usize]| Tensor::zeros(shape, DType::F32, &b).unwrap();
    // The gradient is shaped like the output, always.
    let g_bad = z(&[8, 3]);
    let ln = (z(&[3, 8]), z(&[8]), z(&[8]));
    shape_refused(
        be.layer_norm_backward(&ln.0, &ln.1, &ln.2, &g_bad, 1e-5)
            .map(drop),
        "layer_norm_backward",
        "grad shape",
    );
    shape_refused(
        be.layer_norm_backward(&ln.0, &z(&[7]), &ln.2, &z(&[3, 8]), 1e-5)
            .map(drop),
        "layer_norm_backward",
        "weight",
    );
    shape_refused(
        be.layer_norm_backward(&ln.0, &ln.1, &z(&[2, 8]), &z(&[3, 8]), 1e-5)
            .map(drop),
        "layer_norm_backward",
        "bias",
    );
    shape_refused(
        be.layer_norm_backward(&z(&[]), &ln.1, &ln.2, &z(&[]), 1e-5)
            .map(drop),
        "layer_norm_backward",
        "rank 0",
    );
    let img = z(&[2, 4, 3, 3]);
    shape_refused(
        be.channel_layer_norm_backward(&img, &z(&[4]), &z(&[4]), &z(&[2, 4, 9]), 1e-5)
            .map(drop),
        "channel_layer_norm_backward",
        "grad shape",
    );
    shape_refused(
        be.channel_layer_norm_backward(&img, &z(&[3]), &z(&[4]), &img, 1e-5)
            .map(drop),
        "channel_layer_norm_backward",
        "weight",
    );
    shape_refused(
        be.channel_layer_norm_backward(&z(&[4]), &z(&[4]), &z(&[4]), &z(&[4]), 1e-5)
            .map(drop),
        "channel_layer_norm_backward",
        "rank 1",
    );
    shape_refused(
        be.gelu_erf_backward(&z(&[3, 8]), &z(&[24])).map(drop),
        "gelu_erf_backward",
        "grad shape",
    );
    shape_refused(
        be.softmax_backward(&z(&[3, 8]), &z(&[8, 3])).map(drop),
        "softmax_backward",
        "grad shape",
    );
    shape_refused(
        be.softmax_backward(&z(&[]), &z(&[])).map(drop),
        "softmax_backward",
        "rank 0",
    );
    // Broadcast: extents that differ with neither 1, a gradient that is not
    // the broadcast's shape (the unbroadcast shape included), a rank over 8.
    for (name, mul) in [
        ("broadcast_add_backward", false),
        ("broadcast_mul_backward", true),
    ] {
        let call = |a: &Tensor, b: &Tensor, g: &Tensor| {
            if mul {
                be.broadcast_mul_backward(a, b, g).map(drop)
            } else {
                be.broadcast_add_backward(a, b, g).map(drop)
            }
        };
        shape_refused(
            call(&z(&[2, 3]), &z(&[4]), &z(&[2, 3])),
            name,
            "incompatible extents",
        );
        shape_refused(
            call(&z(&[2, 3]), &z(&[3]), &z(&[3])),
            name,
            "grad shaped like b",
        );
        shape_refused(
            call(&z(&[2, 1]), &z(&[1, 3]), &z(&[2, 1])),
            name,
            "grad shaped like a",
        );
        shape_refused(call(&z(&[1; 9]), &z(&[1]), &z(&[1; 9])), name, "rank 9");
        shape_refused(
            call(&z(&[3, 8]), &z(&[8]), &z(&[3, 8, 1])),
            name,
            "grad rank",
        );
    }
    shape_refused(
        be.linear_bias_backward(&z(&[3, 8]), &z(&[5, 8]), &z(&[5]), &z(&[5, 3]))
            .map(drop),
        "linear_bias_backward",
        "grad shape",
    );
    shape_refused(
        be.linear_bias_backward(&z(&[3, 8]), &z(&[5, 8]), &z(&[4]), &z(&[3, 5]))
            .map(drop),
        "linear_bias_backward",
        "bias",
    );
    shape_refused(
        be.linear_bias_backward(&z(&[3, 7]), &z(&[5, 8]), &z(&[5]), &z(&[3, 5]))
            .map(drop),
        "linear_bias_backward",
        "in-features",
    );
    // A wrong dtype is refused naming the backward.
    let ids = Tensor::from_u32(&[0; 24], &[3, 8], &b).unwrap();
    match be.gelu_erf_backward(&ids, &z(&[3, 8])) {
        Err(OjasError::Dtype { op, .. }) => assert_eq!(op, "gelu_erf_backward"),
        other => panic!("expected Dtype, got {other:?}"),
    }
    // A non-finite eps is refused before any work.
    for eps in [f32::NAN, f32::INFINITY] {
        match be.layer_norm_backward(&ln.0, &ln.1, &ln.2, &z(&[3, 8]), eps) {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, "layer_norm_backward"),
            other => panic!("eps {eps}: expected NonFinite, got {other:?}"),
        }
        match be.channel_layer_norm_backward(&img, &z(&[4]), &z(&[4]), &img, eps) {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, "channel_layer_norm_backward"),
            other => panic!("eps {eps}: expected NonFinite, got {other:?}"),
        }
    }
}

#[test]
fn a_rank_eight_broadcast_is_accepted() {
    let be = exact();
    let (a_shape, b_shape) = ([1usize, 1, 1, 1, 1, 1, 1, 3], [2usize, 1, 1, 1, 1, 1, 1, 1]);
    let (a, b) = (t(&[1.0, 2.0, 3.0], &a_shape), t(&[10.0, 20.0], &b_shape));
    let gy = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 1, 1, 1, 1, 1, 3]);
    let (ga, gb) = be.broadcast_mul_backward(&a, &b, &gy).unwrap();
    assert_eq!(
        v(&ga),
        [
            10.0 * 1.0 + 20.0 * 4.0,
            10.0 * 2.0 + 20.0 * 5.0,
            10.0 * 3.0 + 20.0 * 6.0
        ]
    );
    assert_eq!(
        v(&gb),
        [
            1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0,
            1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0
        ]
    );
}

#[test]
fn a_gradient_that_overflows_f32_is_refused_not_returned_as_infinity() {
    let be = exact();
    let m = f32::MAX;
    let ones = |n: usize| vec![1.0f32; n];
    let nonfinite = |r: Result<(), OjasError>, name: &str| match r {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, name),
        other => panic!("{name}: expected NonFinite, got {other:?}"),
    };
    // GELU: slope 1.13 at 1.5 times f32::MAX.
    nonfinite(
        be.gelu_erf_backward(&t(&[1.5], &[1]), &t(&[m], &[1]))
            .map(drop),
        "gelu_erf_backward",
    );
    // LayerNorm: two rows of [1, -1] with every gradient f32::MAX; the weight
    // and bias gradients sum two of them.
    let x = t(&[1.0, -1.0, 1.0, -1.0], &[2, 2]);
    let (w, bias, gy) = (t(&ones(2), &[2]), t(&[0.0; 2], &[2]), t(&[m; 4], &[2, 2]));
    nonfinite(
        be.layer_norm_backward(&x, &w, &bias, &gy, 1e-5).map(drop),
        "layer_norm_backward",
    );
    let img = t(&[1.0, -1.0, 1.0, -1.0], &[2, 2, 1]);
    nonfinite(
        be.channel_layer_norm_backward(&img, &w, &bias, &gy.reshape(&[2, 2, 1]).unwrap(), 1e-5)
            .map(drop),
        "channel_layer_norm_backward",
    );
    // Broadcast: the stretched operand's gradient sums two of f32::MAX.
    let (a, one, g2) = (t(&ones(2), &[2]), t(&[1.0], &[1]), t(&[m, m], &[2]));
    nonfinite(
        be.broadcast_add_backward(&a, &one, &g2).map(drop),
        "broadcast_add_backward",
    );
    nonfinite(
        be.broadcast_mul_backward(&a, &one, &g2).map(drop),
        "broadcast_mul_backward",
    );
    // Linear: the bias gradient sums two rows of f32::MAX.
    let (lx, lw) = (t(&[0.0; 2], &[2, 1]), t(&[0.0], &[1, 1]));
    nonfinite(
        be.linear_bias_backward(&lx, &lw, &t(&[0.0], &[1]), &t(&[m, m], &[2, 1]))
            .map(drop),
        "linear_bias_backward",
    );
    // Softmax: an output of 2 (not a probability, but not refused) doubles it.
    nonfinite(
        be.softmax_backward(&t(&[2.0, 2.0], &[2]), &t(&[m, m], &[2]))
            .map(drop),
        "softmax_backward",
    );
}

#[test]
fn an_over_budget_gradient_is_refused_and_leaves_no_charge() {
    let be_budget = |bytes: u64| {
        let budget = Budget::new(bytes);
        let be = CpuBackend::with_threads(budget.clone(), 1)
            .unwrap()
            .with_numerics(Numerics::Exact);
        (budget, be)
    };
    for op in ops() {
        // Operands alone: no room for any gradient.
        let probe = Budget::new(1 << 30);
        let inputs: u64 = (op.args)(&probe)
            .iter()
            .map(|t| t.num_elements().unwrap() as u64 * 4)
            .sum();
        let (budget, be) = be_budget(inputs);
        let args = (op.args)(&budget);
        assert_eq!(budget.live_bytes().unwrap(), inputs, "{}", op.name);
        match (op.call)(&be, &args) {
            Err(OjasError::CapacityExceeded { .. }) => {}
            other => panic!("{}: expected CapacityExceeded, got {other:?}", op.name),
        }
        assert_eq!(
            budget.live_bytes().unwrap(),
            inputs,
            "{}: a refusal left a charge",
            op.name
        );
        // The same call with room succeeds and releases its scratch.
        let (budget, be) = be_budget(inputs + (1 << 20));
        let args = (op.args)(&budget);
        (op.call)(&be, &args).unwrap();
        assert_eq!(
            budget.live_bytes().unwrap(),
            inputs,
            "{}: scratch outlived the call",
            op.name
        );
    }
}

// ---- torch ----

const GOLDEN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../ojas-oracle/fixtures/encoder_bwd/golden.txt"
));

struct Golden {
    op: String,
    name: String,
    eps: f32,
    tensors: HashMap<String, (Vec<usize>, Vec<f64>)>,
}

impl Golden {
    fn shape(&self, key: &str) -> &[usize] {
        &self.tensors[key].0
    }

    /// The float32 tensor `key` as a tensor.
    fn tensor(&self, key: &str) -> Tensor {
        let (shape, data) = &self.tensors[key];
        let f: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        t(&f, shape)
    }

    fn values(&self, key: &str) -> &[f64] {
        &self.tensors[key].1
    }
}

fn parse_golden() -> Vec<Golden> {
    let mut lines = GOLDEN.lines();
    assert_eq!(lines.next(), Some("ojas-encoder-bwd-golden-v1"));
    let torch = lines.next().expect("torch version line");
    assert!(torch.starts_with("torch "), "{torch}");
    let mut cases = Vec::new();
    let mut open: Option<Golden> = None;
    for line in lines {
        let mut f = line.split_whitespace();
        match f.next() {
            Some("case") => {
                let (op, name, eps) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
                let eps = if eps == "-" {
                    0.0
                } else {
                    f32::from_bits(u32::from_str_radix(eps, 16).unwrap())
                };
                open = Some(Golden {
                    op: op.into(),
                    name: name.into(),
                    eps,
                    tensors: HashMap::new(),
                });
            }
            Some("t") => {
                let (key, dtype, dims) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
                let shape: Vec<usize> = if dims == "-" {
                    vec![]
                } else {
                    dims.split(',').map(|d| d.parse().unwrap()).collect()
                };
                let data: Vec<f64> = f
                    .map(|h| match dtype {
                        "f32" => f64::from(f32::from_bits(u32::from_str_radix(h, 16).unwrap())),
                        "f64" => f64::from_bits(u64::from_str_radix(h, 16).unwrap()),
                        other => panic!("dtype {other}"),
                    })
                    .collect();
                assert_eq!(
                    data.len(),
                    numel(&shape),
                    "{key} of {:?}",
                    open.as_ref().map(|c| &c.name)
                );
                open.as_mut()
                    .expect("tensor outside a case")
                    .tensors
                    .insert(key.into(), (shape, data));
            }
            Some("end") => cases.push(open.take().expect("end outside a case")),
            None => {}
            Some(other) => panic!("unknown line {other}"),
        }
    }
    assert!(open.is_none(), "unterminated case");
    cases
}

/// What one gradient measured against torch: the normalized error against
/// float32 torch and against float64 torch.
struct Measured {
    case: String,
    tensor: &'static str,
    vs_f32: f64,
    vs_f64: f64,
}

/// Held to float64 torch at what an `f32` result of an exact value can be:
/// the backward rounds an `f64` value once (measured at most 2.6e-7, the
/// softmax cases, whose saved output is torch's own `f32` one).
const TORCH_F64_TOL: f64 = 1e-6;

/// Against float32 torch the bound is torch's own rounding: a few `f32` ulp
/// (measured at most 1.6e-7), except where torch's `f32` statistics lose
/// digits to a row mean far from zero (`ln_mean_offset`, mean 1000, 8.3e-5;
/// `cln_nlc`, mean -50, 2.3e-5), which float64 torch and this backend agree
/// on to 5e-8.
fn torch_f32_tol(case: &str) -> f64 {
    match case {
        "ln_mean_offset" => 2e-4,
        "cln_nlc" => 1e-4,
        _ => 5e-7,
    }
}

fn compare(m: &mut Vec<Measured>, case: &Golden, tensor: &'static str, got: &Tensor, key: &str) {
    let g = v(got);
    assert_eq!(got.shape(), case.shape(key), "{} {tensor} shape", case.name);
    let vs_f32 = normalized(&g, case.values(key));
    let vs_f64 = normalized(&g, case.values(&format!("{key}_64")));
    m.push(Measured {
        case: case.name.clone(),
        tensor,
        vs_f32,
        vs_f64,
    });
}

#[test]
fn the_backwards_match_torch_autograd_on_the_golden_cases() {
    let be = exact();
    let cases = parse_golden();
    assert!(cases.len() >= 31, "the fixture lost cases: {}", cases.len());
    let mut m: Vec<Measured> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for c in &cases {
        *seen.entry(c.op.clone()).or_default() += 1;
        match c.op.as_str() {
            "ln" | "cln" => {
                let (x, w, b, gy) = (c.tensor("x"), c.tensor("w"), c.tensor("b"), c.tensor("gy"));
                let (fwd, g) = if c.op == "ln" {
                    (
                        be.layer_norm_forward(&x, &w, &b, c.eps).unwrap(),
                        be.layer_norm_backward(&x, &w, &b, &gy, c.eps).unwrap(),
                    )
                } else {
                    (
                        be.channel_layer_norm_forward(&x, &w, &b, c.eps).unwrap(),
                        be.channel_layer_norm_backward(&x, &w, &b, &gy, c.eps)
                            .unwrap(),
                    )
                };
                // The parse is right if the forward reproduces torch's output.
                assert!(
                    normalized(&v(&fwd), c.values("y_64")) < 1e-4,
                    "{} forward",
                    c.name
                );
                compare(&mut m, c, "dx", &g.input, "dx");
                compare(&mut m, c, "dw", &g.weight, "dw");
                compare(&mut m, c, "db", &g.bias, "db");
            }
            "gelu" => {
                let (x, gy) = (c.tensor("x"), c.tensor("gy"));
                let fwd = be.gelu_erf_forward(&x).unwrap();
                assert!(
                    normalized(&v(&fwd), c.values("y_64")) < 1e-6,
                    "{} forward",
                    c.name
                );
                compare(
                    &mut m,
                    c,
                    "dx",
                    &be.gelu_erf_backward(&x, &gy).unwrap(),
                    "dx",
                );
            }
            "lin" => {
                let (x, w, b, gy) = (c.tensor("x"), c.tensor("w"), c.tensor("b"), c.tensor("gy"));
                let fwd = be.linear_bias_forward(&x, &w, &b).unwrap();
                assert!(
                    normalized(&v(&fwd), c.values("y_64")) < 1e-5,
                    "{} forward",
                    c.name
                );
                let g = be.linear_bias_backward(&x, &w, &b, &gy).unwrap();
                compare(&mut m, c, "dx", &g.input, "dx");
                compare(&mut m, c, "dw", &g.weight, "dw");
                compare(&mut m, c, "db", &g.bias, "db");
            }
            "badd" | "bmul" => {
                let (a, b, gy) = (c.tensor("a"), c.tensor("b"), c.tensor("gy"));
                let (ga, gb) = if c.op == "badd" {
                    be.broadcast_add_backward(&a, &b, &gy).unwrap()
                } else {
                    be.broadcast_mul_backward(&a, &b, &gy).unwrap()
                };
                compare(&mut m, c, "da", &ga, "da");
                compare(&mut m, c, "db", &gb, "db");
            }
            "softmax" => {
                // Torch's own forward output is the backward's operand.
                let (y, gy) = (c.tensor("y"), c.tensor("gy"));
                compare(
                    &mut m,
                    c,
                    "dx",
                    &be.softmax_backward(&y, &gy).unwrap(),
                    "dx",
                );
            }
            other => panic!("unknown op {other} in the fixture"),
        }
    }
    for op in ["ln", "cln", "gelu", "lin", "badd", "bmul", "softmax"] {
        assert!(seen.contains_key(op), "no golden case for {op}");
    }
    let worst = |f: fn(&Measured) -> f64| m.iter().map(f).fold(0.0f64, f64::max);
    eprintln!(
        "torch goldens: {} cases, {} gradients; max normalized error {:.2e} vs torch f32, {:.2e} vs torch f64",
        cases.len(),
        m.len(),
        worst(|x| x.vs_f32),
        worst(|x| x.vs_f64)
    );
    for x in &m {
        eprintln!(
            "  {:<22} {:<3} vs f32 {:.2e}  vs f64 {:.2e}",
            x.case, x.tensor, x.vs_f32, x.vs_f64
        );
    }
    for x in &m {
        assert!(
            x.vs_f64 <= TORCH_F64_TOL,
            "{} {}: {:e} against torch float64 (> {TORCH_F64_TOL:e})",
            x.case,
            x.tensor,
            x.vs_f64
        );
        // A dimension of 1 has an input gradient of exactly 0 (the row is its
        // own mean); float64 torch gets 0 too (checked above), float32 torch
        // leaves rounding noise, which normalized by itself is not a bound.
        if x.case == "ln_dim_one" && x.tensor == "dx" {
            assert_eq!(x.vs_f64, 0.0, "ln_dim_one dx is exactly 0");
            continue;
        }
        assert!(
            x.vs_f32 <= torch_f32_tol(&x.case),
            "{} {}: {:e} against torch float32 (> {:e})",
            x.case,
            x.tensor,
            x.vs_f32,
            torch_f32_tol(&x.case)
        );
    }
}
