//! Qwen3.5's hybrid-layer ops on the CPU: `causal_conv1d_silu`,
//! `gated_rms_norm` and `rope_partial`, forward against f64 references of
//! the published formulas (tessl's `tests/common/qwen35.rs` oracles,
//! transformers' `Qwen3_5RMSNormGated` and causal conv), backward against
//! central differences of those references, and the bit-level contracts:
//! partial RoPE over the whole head is `rope_half_split`, the tail passes
//! through, and the thread count does not change a bit.

mod common;

use common::{assert_capacity, assert_nonfinite, assert_shape, bits};
use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

fn cpu(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 26), threads)
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

fn f32s(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

/// The f64 values of the f32 tensor the test hands the backend.
fn rounded(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x as f32)).collect()
}

fn t(b: &CpuBackend, v: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_f32(&f32s(v), shape, b.budget()).unwrap()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// `got` against `want`: `|got - want| <= atol + rtol * |want|`.
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

/// Central difference of the scalar `f` at `x`, coordinate by coordinate.
fn numeric_grad(x: &[f64], f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    const H: f64 = 1e-5;
    let mut p = x.to_vec();
    (0..x.len())
        .map(|i| {
            let o = p[i];
            p[i] = o + H;
            let up = f(&p);
            p[i] = o - H;
            let down = f(&p);
            p[i] = o;
            (up - down) / (2.0 * H)
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ---- causal conv1d + SiLU ------------------------------------------------

/// tessl's `conv1d_silu_f64` from a zero state: `y[b, t, c] =
/// silu(sum_j w[c, j] * x_ext[b, t + j, c])`, `x_ext` = `K - 1` zeros then
/// `x`.
fn conv_ref(x: &[f64], w: &[f64], b: usize, t: usize, c: usize, k: usize) -> Vec<f64> {
    let mut y = vec![0.0; b * t * c];
    for bi in 0..b {
        for ci in 0..c {
            for ti in 0..t {
                let mut acc = 0.0;
                for j in 0..k {
                    let pos = ti + j;
                    if pos >= k - 1 {
                        acc += w[ci * k + j] * x[(bi * t + pos - (k - 1)) * c + ci];
                    }
                }
                y[(bi * t + ti) * c + ci] = silu(acc);
            }
        }
    }
    y
}

#[test]
fn conv1d_forward_and_backward_match_the_reference() {
    for (b, tt, c, k) in [
        (1, 1, 1, 4),
        (2, 9, 5, 4),
        (1, 3, 2, 6),
        (3, 7, 4, 2),
        (1, 5, 3, 1),
    ] {
        let x = rounded(&vals(1, b * tt * c, -2.0, 2.0));
        let w = rounded(&vals(2, c * k, -1.0, 1.0));
        let gy = rounded(&vals(3, b * tt * c, -1.0, 1.0));
        let want = conv_ref(&x, &w, b, tt, c, k);
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let be = cpu(1, numerics);
            let what = format!("[{b}, {tt}, {c}] k {k} {numerics:?}");
            let (xt, wt) = (t(&be, &x, &[b, tt, c]), t(&be, &w, &[c, k]));
            let y = be.causal_conv1d_silu_forward(&xt, &wt).unwrap();
            assert_eq!(y.shape(), &[b, tt, c]);
            close(&format!("{what} y"), &host(&y), &want, 1e-6, 1e-5);
            let (gx, gw) = be
                .causal_conv1d_silu_backward(&xt, &wt, &t(&be, &gy, &[b, tt, c]))
                .unwrap();
            let nx = numeric_grad(&x, |p| dot(&conv_ref(p, &w, b, tt, c, k), &gy));
            let nw = numeric_grad(&w, |p| dot(&conv_ref(&x, p, b, tt, c, k), &gy));
            close(&format!("{what} gx"), &host(&gx), &nx, 1e-5, 1e-4);
            close(&format!("{what} gw"), &host(&gw), &nw, 1e-5, 1e-4);
        }
    }
}

/// Output `t` reads only inputs `t - K + 1 ..= t`: a change at time `s`
/// moves no output before `s`.
#[test]
fn conv1d_is_causal() {
    let be = cpu(1, Numerics::Exact);
    let (b, tt, c, k) = (1, 8, 3, 4);
    let x = vals(4, b * tt * c, -1.0, 1.0);
    let w = vals(5, c * k, -1.0, 1.0);
    let base = host(
        &be.causal_conv1d_silu_forward(&t(&be, &x, &[b, tt, c]), &t(&be, &w, &[c, k]))
            .unwrap(),
    );
    for s in 0..tt {
        let mut moved = x.clone();
        for ci in 0..c {
            moved[s * c + ci] += 0.5;
        }
        let y = host(
            &be.causal_conv1d_silu_forward(&t(&be, &moved, &[b, tt, c]), &t(&be, &w, &[c, k]))
                .unwrap(),
        );
        assert_eq!(bits(&y[..s * c]), bits(&base[..s * c]), "change at {s}");
        assert_ne!(
            bits(&y[s * c..(s + 1) * c]),
            bits(&base[s * c..(s + 1) * c])
        );
    }
}

// ---- gated RMSNorm --------------------------------------------------------

/// tessl's `gated_rms_norm_f64`: `w * x / sqrt(mean(x^2) + eps) * silu(z)`
/// per `d`-wide row.
fn gated_ref(x: &[f64], z: &[f64], w: &[f64], d: usize, eps: f64) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for (r, (xr, zr)) in x.chunks(d).zip(z.chunks(d)).enumerate() {
        let ss: f64 = xr.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / d as f64 + eps).sqrt();
        for i in 0..d {
            out[r * d + i] = w[i] * xr[i] * inv * silu(zr[i]);
        }
    }
    out
}

#[test]
fn gated_rms_norm_forward_and_backward_match_the_reference() {
    const EPS: f32 = 1e-6;
    for shape in [vec![1usize], vec![3, 7], vec![2, 3, 4, 16]] {
        let d = *shape.last().unwrap();
        let n: usize = shape.iter().product();
        let x = rounded(&vals(6, n, -2.0, 2.0));
        let z = rounded(&vals(7, n, -3.0, 3.0));
        let w = rounded(&vals(8, d, 0.5, 1.5));
        let gy = rounded(&vals(9, n, -1.0, 1.0));
        let eps = f64::from(EPS);
        let want = gated_ref(&x, &z, &w, d, eps);
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let be = cpu(1, numerics);
            let what = format!("{shape:?} {numerics:?}");
            let (xt, zt, wt) = (t(&be, &x, &shape), t(&be, &z, &shape), t(&be, &w, &[d]));
            let y = be.gated_rms_norm_forward(&xt, &zt, &wt, EPS).unwrap();
            close(&format!("{what} y"), &host(&y), &want, 1e-6, 1e-5);
            let g = be
                .gated_rms_norm_backward(&xt, &zt, &wt, &t(&be, &gy, &shape), EPS)
                .unwrap();
            let f = |x: &[f64], z: &[f64], w: &[f64]| dot(&gated_ref(x, z, w, d, eps), &gy);
            close(
                &format!("{what} gx"),
                &host(&g.input),
                &numeric_grad(&x, |p| f(p, &z, &w)),
                1e-4,
                1e-3,
            );
            close(
                &format!("{what} gz"),
                &host(&g.gate),
                &numeric_grad(&z, |p| f(&x, p, &w)),
                1e-5,
                1e-4,
            );
            close(
                &format!("{what} gw"),
                &host(&g.weight),
                &numeric_grad(&w, |p| f(&x, &z, p)),
                1e-5,
                1e-4,
            );
        }
    }
}

// ---- partial RoPE ---------------------------------------------------------

fn tables(be: &CpuBackend, time: usize, rotary: usize) -> (Tensor, Tensor) {
    let budget = be.budget();
    let mrope = ojas_core::MropeSection {
        section: [rotary / 2, 0, 0],
        interleaved: false,
    };
    ojas_core::mrope_text_tables(3, time, mrope, rotary, 1e4, budget).unwrap()
}

/// With `R = D` the op is `rope_half_split` bit for bit, both ways.
#[test]
fn rope_partial_over_the_whole_head_is_rope_half_split() {
    let be = cpu(1, Numerics::Exact);
    let shape = [2, 5, 3, 8];
    let x = t(&be, &vals(10, 240, -2.0, 2.0), &shape);
    let (cos, sin) = tables(&be, 5, 8);
    let a = be.rope_partial_forward(&x, &cos, &sin).unwrap();
    let b = be.rope_half_split_forward(&x, &cos, &sin).unwrap();
    assert_eq!(bits(&host(&a)), bits(&host(&b)));
    let a = be.rope_partial_backward(&x, &cos, &sin).unwrap();
    let b = be.rope_half_split_backward(&x, &cos, &sin).unwrap();
    assert_eq!(bits(&host(&a)), bits(&host(&b)));
}

/// `R < D`: the leading `R` of each head are `rope_half_split` of those `R`
/// values alone, and the tail is the input bit for bit; backward is the
/// adjoint (`<rope(x), g> = <x, rope_backward(g)>`).
#[test]
fn rope_partial_turns_the_leading_values_and_passes_the_tail() {
    let be = cpu(1, Numerics::Exact);
    let (b, tt, h, d, r) = (2, 6, 2, 16, 4);
    let xv = vals(11, b * tt * h * d, -2.0, 2.0);
    let x = t(&be, &xv, &[b, tt, h, d]);
    let (cos, sin) = tables(&be, tt, r);
    let y = host(&be.rope_partial_forward(&x, &cos, &sin).unwrap());
    let head: Vec<f64> = xv.chunks(d).flat_map(|row| row[..r].to_vec()).collect();
    let lead = be
        .rope_half_split_forward(&t(&be, &head, &[b, tt, h, r]), &cos, &sin)
        .unwrap();
    let lead = host(&lead);
    let xs = f32s(&xv);
    for (row, (yr, xr)) in y.chunks(d).zip(xs.chunks(d)).enumerate() {
        assert_eq!(
            bits(&yr[..r]),
            bits(&lead[row * r..(row + 1) * r]),
            "row {row} lead"
        );
        assert_eq!(bits(&yr[r..]), bits(&xr[r..]), "row {row} tail");
    }
    let gv = vals(12, b * tt * h * d, -1.0, 1.0);
    let g = host(
        &be.rope_partial_backward(&t(&be, &gv, &[b, tt, h, d]), &cos, &sin)
            .unwrap(),
    );
    let lhs: f64 = y
        .iter()
        .zip(f32s(&gv))
        .map(|(&a, b)| f64::from(a) * f64::from(b))
        .sum();
    let rhs: f64 = g
        .iter()
        .zip(&xs)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    assert!(
        (lhs - rhs).abs() < 1e-4 * lhs.abs().max(1.0),
        "{lhs} vs {rhs}"
    );
}

#[test]
fn bits_do_not_depend_on_the_thread_count() {
    let run = |threads: usize| {
        let be = cpu(threads, Numerics::Fast);
        let x = t(&be, &vals(13, 2 * 64 * 24, -2.0, 2.0), &[2, 64, 24]);
        let w = t(&be, &vals(14, 24 * 4, -1.0, 1.0), &[24, 4]);
        let gy = t(&be, &vals(15, 2 * 64 * 24, -1.0, 1.0), &[2, 64, 24]);
        let y = be.causal_conv1d_silu_forward(&x, &w).unwrap();
        let (gx, gw) = be.causal_conv1d_silu_backward(&x, &w, &gy).unwrap();
        let z = t(&be, &vals(16, 2 * 64 * 24, -2.0, 2.0), &[2, 64, 24]);
        let nw = t(&be, &vals(17, 24, 0.5, 1.5), &[24]);
        let n = be.gated_rms_norm_forward(&x, &z, &nw, 1e-6).unwrap();
        let ng = be.gated_rms_norm_backward(&x, &z, &nw, &gy, 1e-6).unwrap();
        let x4 = x.reshape(&[2, 64, 2, 12]).unwrap();
        let (cos, sin) = tables(&be, 64, 8);
        let r = be.rope_partial_forward(&x4, &cos, &sin).unwrap();
        [y, gx, gw, n, ng.input, ng.gate, ng.weight, r].map(|t| bits(&host(&t)))
    };
    assert_eq!(run(1), run(6));
}

/// The CPU's f32 sigmoid, as `pointwise::sigmoid` writes it.
fn sigmoid32(x: f32) -> f32 {
    if x >= 0.0 {
        let z = ojas_core::exp_exact(-x);
        1.0 / (1.0 + z)
    } else {
        let z = ojas_core::exp_exact(x);
        z / (1.0 + z)
    }
}

fn silu_slope32(a: f32) -> (f32, f32) {
    let s = sigmoid32(a);
    (a * s, s * (1.0 + a * (1.0 - s)))
}

/// The conv1d backward as one serial loop nest, the order every sum took
/// before the passes were split across threads (2026-10-07).
fn conv_bwd_serial(x: &[f32], w: &[f32], gy: &[f32], dims: [usize; 4]) -> (Vec<f32>, Vec<f32>) {
    let [batch, time, ch, k] = dims;
    let pre = |b: usize, t: usize, c: usize| {
        let mut acc = 0.0f32;
        for j in 0..k {
            if let Some(src) = (t + j).checked_sub(k - 1) {
                acc += w[c * k + j] * x[(b * time + src) * ch + c];
            }
        }
        acc
    };
    let mut da = vec![0.0f32; x.len()];
    for b in 0..batch {
        for t in 0..time {
            for c in 0..ch {
                let at = (b * time + t) * ch + c;
                da[at] = gy[at] * silu_slope32(pre(b, t, c)).1;
            }
        }
    }
    let mut gx = vec![0.0f32; x.len()];
    for b in 0..batch {
        for s in 0..time {
            for c in 0..ch {
                let mut acc = 0.0f32;
                for j in 0..k {
                    let t = s + (k - 1) - j;
                    if t < time {
                        acc += w[c * k + j] * da[(b * time + t) * ch + c];
                    }
                }
                gx[(b * time + s) * ch + c] = acc;
            }
        }
    }
    let mut gw = vec![0.0f32; ch * k];
    for c in 0..ch {
        for j in 0..k {
            let mut acc = 0.0f32;
            for b in 0..batch {
                for t in 0..time {
                    if let Some(src) = (t + j).checked_sub(k - 1) {
                        acc += da[(b * time + t) * ch + c] * x[(b * time + src) * ch + c];
                    }
                }
            }
            gw[c * k + j] = acc;
        }
    }
    (gx, gw)
}

/// The gated RMSNorm backward as one serial row loop, the order every sum
/// took before the passes were split across threads (2026-10-07).
fn gated_bwd_serial(
    x: &[f32],
    z: &[f32],
    w: &[f32],
    gy: &[f32],
    dim: usize,
    eps: f32,
) -> [Vec<f32>; 3] {
    let (mut gx, mut gz, mut gw) = (vec![0.0; x.len()], vec![0.0; x.len()], vec![0.0; dim]);
    for row in 0..x.len() / dim {
        let span = row * dim..(row + 1) * dim;
        let (xs, zs, gs) = (&x[span.clone()], &z[span.clone()], &gy[span.clone()]);
        let mut sum_sq = 0.0f32;
        for &v in xs {
            sum_sq += v * v;
        }
        let r = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        let each = || xs.iter().zip(zs).zip(gs).zip(w);
        let mut dot = 0.0f32;
        for (((&xv, &zv), &gv), &wv) in each() {
            let dn = (gv * wv) * silu_slope32(zv).0;
            dot += dn * (xv * r);
        }
        let mean = dot / dim as f32;
        let outs = gx[span.clone()].iter_mut().zip(&mut gz[span]).zip(&mut gw);
        for ((((&xv, &zv), &gv), &wv), ((dx, dz), dw)) in each().zip(outs) {
            let n = xv * r;
            let (g, slope) = silu_slope32(zv);
            let dn = (gv * wv) * g;
            *dx = r * (dn - n * mean);
            *dz = ((gv * wv) * n) * slope;
            *dw += (gv * n) * g;
        }
    }
    [gx, gz, gw]
}

/// Shapes large enough that every pass is cut into several pieces (rows,
/// conv channels, norm columns), at 1, 3 and 6 threads: each output equals
/// the serial loop nest bit for bit.
#[test]
fn split_backward_passes_match_the_serial_loops_bit_for_bit() {
    let (batch, time, ch, k) = (2usize, 512usize, 96usize, 4usize);
    let n = batch * time * ch;
    let x = f32s(&vals(31, n, -2.0, 2.0));
    let w = f32s(&vals(32, ch * k, -1.0, 1.0));
    let gy = f32s(&vals(33, n, -1.0, 1.0));
    let (want_gx, want_gw) = conv_bwd_serial(&x, &w, &gy, [batch, time, ch, k]);
    let (rows, dim) = (4096usize, 32usize);
    let nx = f32s(&vals(34, rows * dim, -2.0, 2.0));
    let nz = f32s(&vals(35, rows * dim, -2.0, 2.0));
    let nw = f32s(&vals(36, dim, 0.5, 1.5));
    let ng = f32s(&vals(37, rows * dim, -1.0, 1.0));
    let want_norm = gated_bwd_serial(&nx, &nz, &nw, &ng, dim, 1e-6);
    for threads in [1, 3, 6] {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let be = cpu(threads, numerics);
            let f = |v: &[f32], shape: &[usize]| {
                Tensor::from_f32(v, shape, be.budget()).unwrap()
            };
            let (gx, gw) = be
                .causal_conv1d_silu_backward(
                    &f(&x, &[batch, time, ch]),
                    &f(&w, &[ch, k]),
                    &f(&gy, &[batch, time, ch]),
                )
                .unwrap();
            assert_eq!(bits(&host(&gx)), bits(&want_gx), "conv gx, {threads} threads");
            assert_eq!(bits(&host(&gw)), bits(&want_gw), "conv gw, {threads} threads");
            let g = be
                .gated_rms_norm_backward(
                    &f(&nx, &[rows, dim]),
                    &f(&nz, &[rows, dim]),
                    &f(&nw, &[dim]),
                    &f(&ng, &[rows, dim]),
                    1e-6,
                )
                .unwrap();
            for (name, got, want) in [
                ("dx", &g.input, &want_norm[0]),
                ("dz", &g.gate, &want_norm[1]),
                ("dw", &g.weight, &want_norm[2]),
            ] {
                assert_eq!(bits(&host(got)), bits(want), "{name}, {threads} threads");
            }
        }
    }
}

/// Shapes are refused by the `ojas_core::shapes` validators with nothing
/// charged; a NaN is refused; an output with no room is CapacityExceeded.
#[test]
fn refusals() {
    let host_budget = Budget::new(1 << 20);
    let f = |v: f32, shape: &[usize]| {
        let n = shape.iter().product();
        Tensor::from_f32(&vec![v; n], shape, &host_budget).unwrap()
    };
    let none = CpuBackend::new(Budget::new(0));
    assert_shape(none.causal_conv1d_silu_forward(&f(1.0, &[2, 3]), &f(1.0, &[3, 4])));
    assert_shape(none.causal_conv1d_silu_forward(&f(1.0, &[1, 2, 3]), &f(1.0, &[2, 4])));
    assert_shape(none.causal_conv1d_silu_backward(
        &f(1.0, &[1, 2, 3]),
        &f(1.0, &[3, 4]),
        &f(1.0, &[1, 3, 3]),
    ));
    assert_shape(none.gated_rms_norm_forward(
        &f(1.0, &[2, 4]),
        &f(1.0, &[2, 3]),
        &f(1.0, &[4]),
        1e-6,
    ));
    assert_shape(none.gated_rms_norm_forward(
        &f(1.0, &[2, 4]),
        &f(1.0, &[2, 4]),
        &f(1.0, &[3]),
        1e-6,
    ));
    assert!(matches!(
        none.gated_rms_norm_forward(&f(1.0, &[2, 4]), &f(1.0, &[2, 4]), &f(1.0, &[4]), f32::NAN),
        Err(OjasError::NonFinite { .. })
    ));
    let (x4, c) = (f(1.0, &[1, 2, 1, 8]), f(1.0, &[2, 4]));
    assert_shape(none.rope_partial_forward(&x4, &f(1.0, &[2, 3]), &f(1.0, &[2, 3])));
    assert_shape(none.rope_partial_forward(&x4, &f(1.0, &[2, 10]), &f(1.0, &[2, 10])));
    assert_shape(none.rope_partial_forward(&x4, &f(1.0, &[3, 4]), &f(1.0, &[3, 4])));
    assert_shape(none.rope_partial_forward(&f(1.0, &[2, 8]), &c, &c));
    assert_shape(none.rope_partial_backward(&x4, &c, &f(1.0, &[2, 2])));
    assert_eq!(none.budget().live_bytes().unwrap(), 0);
    // Well-formed calls under budget 0 need room for their output.
    assert_capacity(none.causal_conv1d_silu_forward(&f(1.0, &[1, 2, 3]), &f(1.0, &[3, 4])));
    assert_capacity(none.rope_partial_forward(&x4, &c, &c));
    let ample = CpuBackend::new(Budget::new(1 << 20));
    assert_nonfinite(ample.causal_conv1d_silu_forward(&f(f32::NAN, &[1, 2, 3]), &f(1.0, &[3, 4])));
    assert_nonfinite(ample.gated_rms_norm_forward(
        &f(1.0, &[2, 4]),
        &f(f32::INFINITY, &[2, 4]),
        &f(1.0, &[4]),
        1e-6,
    ));
    assert_nonfinite(ample.rope_partial_forward(&f(f32::NAN, &[1, 2, 1, 8]), &c, &c));
    assert_eq!(ample.budget().live_bytes().unwrap(), 0);
}
