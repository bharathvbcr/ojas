//! LayerNorm (plain and channel-wise), exact GELU, linear with bias,
//! broadcast add and multiply, scalar scale and softmax on the CPU.
//!
//! - Each op is held to an `f64` reference (tolerances are per-tensor
//!   `max |got - ref| / max |ref|`).
//! - Relations between ops are bit-exact: a channel LayerNorm is a permute,
//!   a LayerNorm and a permute back; a linear with bias is the linear and
//!   then a broadcast add.
//! - The bits do not depend on the thread count, and `Fast` and `Exact`
//!   agree except for the softmax exponential.
//! - A NaN or infinity in any operand, a shape that does not fit and an
//!   over-budget output are refused with the variant every op uses.

use ojas_core::{erf_f64, Backend, Budget, DType, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_capacity, assert_nonfinite, assert_shape, bits, SplitMix64};

const THREADS: [usize; 4] = [1, 2, 7, 18];

fn backend(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 32), threads)
        .unwrap()
        .with_numerics(numerics)
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

fn rand_t(rng: &mut SplitMix64, shape: &[usize], scale: f32) -> Tensor {
    t(&rng.vec(shape.iter().product(), scale), shape)
}

/// `max |got - want| / max |want|`.
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

fn f64s(x: &Tensor) -> Vec<f64> {
    v(x).into_iter().map(f64::from).collect()
}

/// LayerNorm of every `dim`-wide row of `x`, in `f64`.
fn layer_norm_ref(x: &[f64], w: &[f64], b: &[f64], dim: usize, eps: f64) -> Vec<f64> {
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

fn gelu_ref(x: f64) -> f64 {
    0.5 * x * (1.0 + erf_f64(x * std::f64::consts::FRAC_1_SQRT_2))
}

// ---- LayerNorm ----

#[test]
fn layer_norm_matches_an_f64_reference() {
    let mut rng = SplitMix64(1);
    for shape in [
        vec![5, 16],
        vec![2, 3, 24],
        vec![7],
        vec![1, 1],
        vec![4, 1, 33],
    ] {
        let dim = *shape.last().unwrap();
        let x = rand_t(&mut rng, &shape, 3.0);
        let w = rand_t(&mut rng, &[dim], 2.0);
        let b = rand_t(&mut rng, &[dim], 1.0);
        let want = layer_norm_ref(&f64s(&x), &f64s(&w), &f64s(&b), dim, 1e-5);
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let got = backend(1, numerics)
                .layer_norm_forward(&x, &w, &b, 1e-5)
                .unwrap();
            assert_eq!(got.shape(), shape.as_slice());
            assert_close(&format!("{shape:?}"), &v(&got), &want, 2e-6);
        }
    }
}

#[test]
fn layer_norm_of_a_constant_row_is_the_bias() {
    let x = t(&[3.0; 8], &[2, 4]);
    let w = t(&[2.0, 3.0, 4.0, 5.0], &[4]);
    let b = t(&[0.5, -1.0, 0.25, 7.0], &[4]);
    let y = backend(1, Numerics::Exact)
        .layer_norm_forward(&x, &w, &b, 1e-5)
        .unwrap();
    assert_eq!(v(&y), [0.5, -1.0, 0.25, 7.0, 0.5, -1.0, 0.25, 7.0]);
}

#[test]
fn layer_norm_with_no_variance_and_no_eps_is_nonfinite() {
    let x = t(&[3.0; 4], &[1, 4]);
    let w = t(&[1.0; 4], &[4]);
    let b = t(&[0.0; 4], &[4]);
    assert_nonfinite(backend(1, Numerics::Exact).layer_norm_forward(&x, &w, &b, 0.0));
}

#[test]
fn channel_layer_norm_is_permute_layer_norm_permute_bit_for_bit() {
    let mut rng = SplitMix64(2);
    for shape in [
        vec![2, 6, 5, 4],
        vec![1, 8, 3],
        vec![3, 5],
        vec![1, 4, 1, 1],
    ] {
        let c = shape[1];
        let x = rand_t(&mut rng, &shape, 2.0);
        let w = rand_t(&mut rng, &[c], 2.0);
        let b = rand_t(&mut rng, &[c], 1.0);
        let rank = shape.len();
        // [N, C, *S] -> [N, *S, C] and back.
        let mut to_last: Vec<usize> = vec![0];
        to_last.extend(2..rank);
        to_last.push(1);
        let mut back = vec![0; rank];
        for (out_axis, &in_axis) in to_last.iter().enumerate() {
            back[in_axis] = out_axis;
        }
        for threads in THREADS {
            for numerics in [Numerics::Exact, Numerics::Fast] {
                let cpu = backend(threads, numerics);
                let got = cpu.channel_layer_norm_forward(&x, &w, &b, 1e-6).unwrap();
                let moved = cpu.permute(&x, &to_last).unwrap();
                let normed = cpu.layer_norm_forward(&moved, &w, &b, 1e-6).unwrap();
                let want = cpu.permute(&normed, &back).unwrap();
                assert_eq!(got.shape(), shape.as_slice());
                assert_eq!(bits(&v(&got)), bits(&v(&want)), "{shape:?} x{threads}");
            }
        }
    }
}

#[test]
fn layer_norms_do_not_depend_on_the_thread_count() {
    let mut rng = SplitMix64(3);
    // Large enough to be cut into several chunks (ROW_MIN_ELEMS = 2^15).
    let x = rand_t(&mut rng, &[6, 64, 40, 40], 2.0);
    let w = rand_t(&mut rng, &[64], 1.5);
    let b = rand_t(&mut rng, &[64], 1.0);
    let flat = x.reshape(&[6 * 64 * 40, 40]).unwrap();
    let w40 = rand_t(&mut rng, &[40], 1.5);
    let b40 = rand_t(&mut rng, &[40], 1.0);
    let base = backend(1, Numerics::Fast);
    let chan = v(&base.channel_layer_norm_forward(&x, &w, &b, 1e-6).unwrap());
    let row = v(&base.layer_norm_forward(&flat, &w40, &b40, 1e-6).unwrap());
    for threads in THREADS {
        let cpu = backend(threads, Numerics::Fast);
        let c = v(&cpu.channel_layer_norm_forward(&x, &w, &b, 1e-6).unwrap());
        let r = v(&cpu.layer_norm_forward(&flat, &w40, &b40, 1e-6).unwrap());
        assert_eq!(bits(&c), bits(&chan), "channel x{threads}");
        assert_eq!(bits(&r), bits(&row), "row x{threads}");
    }
}

// ---- GELU ----

#[test]
fn gelu_erf_matches_known_values() {
    let x = t(&[0.5, 1.0, -1.0, 2.0, -2.0, 0.1, 0.0], &[7]);
    let y = backend(1, Numerics::Exact).gelu_erf_forward(&x).unwrap();
    let want = [
        0.345_731_230_637_006_56,
        0.841_344_746_068_542_9,
        -0.158_655_253_931_457_07,
        1.954_499_736_103_641_6,
        -0.045_500_263_896_358_42,
        0.053_982_783_727_702_904,
        0.0,
    ];
    assert_close("gelu", &v(&y), &want, 2e-7);
}

#[test]
fn gelu_erf_matches_its_reference_over_a_sweep_and_is_not_the_tanh_form() {
    let xs: Vec<f32> = (-400..=400).map(|i| i as f32 * 0.02).collect();
    let x = t(&xs, &[xs.len()]);
    let want: Vec<f64> = xs.iter().map(|&v| gelu_ref(f64::from(v))).collect();
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let got = v(&backend(1, numerics).gelu_erf_forward(&x).unwrap());
        for ((&g, &w), &xv) in got.iter().zip(&want).zip(&xs) {
            assert!(
                (f64::from(g) - w).abs() <= 2e-7 * w.abs() + 1e-15,
                "gelu({xv}) = {g}, want {w}"
            );
        }
    }
    // tanh approximation of gelu(1.0) differs at 1.5e-4.
    let tanh_1 = 0.5 * (1.0 + (0.797_884_560_802_865_4f64 * (1.0 + 0.044_715)).tanh());
    let g1 = v(&backend(1, Numerics::Exact)
        .gelu_erf_forward(&t(&[1.0], &[1]))
        .unwrap())[0];
    assert!((f64::from(g1) - tanh_1).abs() > 1e-4);
}

#[test]
fn gelu_erf_bits_are_the_same_at_every_thread_count_and_numerics() {
    let mut rng = SplitMix64(4);
    let x = rand_t(&mut rng, &[3, 70_001], 4.0);
    let base = bits(&v(&backend(1, Numerics::Exact)
        .gelu_erf_forward(&x)
        .unwrap()));
    for threads in THREADS {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let got = backend(threads, numerics).gelu_erf_forward(&x).unwrap();
            assert_eq!(bits(&v(&got)), base, "x{threads} {numerics:?}");
        }
    }
}

// ---- Linear with bias ----

#[test]
fn linear_bias_is_the_linear_then_a_broadcast_add_bit_for_bit() {
    let mut rng = SplitMix64(5);
    for (shape, out) in [(vec![4, 9], 6), (vec![2, 3, 17], 5), (vec![11], 3)] {
        let input = *shape.last().unwrap();
        let x = rand_t(&mut rng, &shape, 1.0);
        let w = rand_t(&mut rng, &[out, input], 1.0);
        let b = rand_t(&mut rng, &[out], 1.0);
        for threads in THREADS {
            for numerics in [Numerics::Exact, Numerics::Fast] {
                let cpu = backend(threads, numerics);
                let got = cpu.linear_bias_forward(&x, &w, &b).unwrap();
                let lin = cpu.linear_forward(&x, &w).unwrap();
                let want = cpu.broadcast_add_forward(&lin, &b).unwrap();
                assert_eq!(got.shape(), want.shape());
                assert_eq!(bits(&v(&got)), bits(&v(&want)), "{shape:?} x{threads}");
            }
        }
    }
}

#[test]
fn linear_bias_matches_an_f64_reference() {
    let x = t(&[1.0, 2.0, 3.0, -1.0, 0.5, 4.0], &[2, 3]);
    let w = t(&[0.5, -1.0, 2.0, 1.0, 1.0, 1.0], &[2, 3]);
    let b = t(&[10.0, -20.0], &[2]);
    let y = backend(1, Numerics::Exact)
        .linear_bias_forward(&x, &w, &b)
        .unwrap();
    // [1,2,3].[.5,-1,2] = 4.5; [1,2,3].[1,1,1] = 6; [-1,.5,4].[.5,-1,2] = 7; sum = 3.5.
    assert_eq!(v(&y), [14.5, -14.0, 17.0, -16.5]);
}

#[test]
fn linear_bias_takes_a_bf16_weight_like_linear_forward() {
    let mut rng = SplitMix64(6);
    let cpu = backend(1, Numerics::Exact);
    let x = rand_t(&mut rng, &[3, 8], 1.0);
    let w = rand_t(&mut rng, &[4, 8], 1.0);
    let b = rand_t(&mut rng, &[4], 1.0);
    let wh = cpu.to_bf16(&w).unwrap();
    assert_eq!(wh.dtype(), DType::Bf16);
    let got = cpu.linear_bias_forward(&x, &wh, &b).unwrap();
    let want = cpu
        .broadcast_add_forward(&cpu.linear_forward(&x, &wh).unwrap(), &b)
        .unwrap();
    assert_eq!(bits(&v(&got)), bits(&v(&want)));
    // The bias is f32 only.
    let bh = cpu.to_bf16(&b).unwrap();
    assert!(matches!(
        cpu.linear_bias_forward(&x, &w, &bh),
        Err(OjasError::Dtype { .. })
    ));
}

// ---- Broadcast add and multiply ----

/// numpy broadcast of `a` and `b` by index arithmetic, in `f64`.
fn broadcast_ref(
    a: &[f32],
    ashape: &[usize],
    b: &[f32],
    bshape: &[usize],
    f: impl Fn(f64, f64) -> f64,
) -> (Vec<usize>, Vec<f64>) {
    let rank = ashape.len().max(bshape.len());
    let pad = |s: &[usize]| {
        let mut p = vec![1; rank - s.len()];
        p.extend_from_slice(s);
        p
    };
    let (pa, pb) = (pad(ashape), pad(bshape));
    let out: Vec<usize> = pa.iter().zip(&pb).map(|(&x, &y)| x.max(y)).collect();
    let n: usize = out.iter().product();
    let mut vals = Vec::with_capacity(n);
    for flat in 0..n {
        let (mut rest, mut ai, mut bi, mut astride, mut bstride) = (flat, 0, 0, 1, 1);
        for axis in (0..rank).rev() {
            let i = rest % out[axis];
            rest /= out[axis];
            ai += if pa[axis] == 1 { 0 } else { i } * astride;
            bi += if pb[axis] == 1 { 0 } else { i } * bstride;
            astride *= pa[axis];
            bstride *= pb[axis];
        }
        vals.push(f(f64::from(a[ai]), f64::from(b[bi])));
    }
    (out, vals)
}

#[test]
fn broadcast_add_and_mul_match_numpy_over_many_shape_pairs() {
    let cases: [(&[usize], &[usize]); 14] = [
        (&[4, 5], &[4, 5]),
        (&[4, 5], &[5]),
        (&[5], &[4, 5]),
        (&[4, 5], &[4, 1]),
        (&[4, 1], &[1, 5]),
        (&[2, 3, 4, 5], &[3, 1, 1]),
        (&[2, 3, 4, 5], &[1, 4, 5]),
        (&[1, 3, 1, 5], &[2, 1, 4, 1]),
        (&[2, 3, 4], &[1]),
        (&[1], &[2, 3, 4]),
        (&[2, 3], &[]),
        (&[], &[]),
        (&[7], &[7]),
        (&[1, 1, 1, 1, 1, 1, 1, 3], &[2, 2, 2, 2, 2, 2, 2, 1]),
    ];
    let mut rng = SplitMix64(7);
    for (ashape, bshape) in cases {
        let a = rand_t(&mut rng, ashape, 2.0);
        let b = rand_t(&mut rng, bshape, 2.0);
        let (av, bv) = (v(&a), v(&b));
        for threads in [1, 7] {
            let cpu = backend(threads, Numerics::Exact);
            for (name, op) in [("add", 0), ("mul", 1)] {
                let (shape, want) = broadcast_ref(&av, ashape, &bv, bshape, |x, y| {
                    if op == 0 {
                        x + y
                    } else {
                        x * y
                    }
                });
                let got = if op == 0 {
                    cpu.broadcast_add_forward(&a, &b)
                } else {
                    cpu.broadcast_mul_forward(&a, &b)
                }
                .unwrap();
                assert_eq!(
                    got.shape(),
                    shape.as_slice(),
                    "{name} {ashape:?} {bshape:?}"
                );
                // One f32 operation per element: the f32 rounding of the f64 result.
                let rounded: Vec<u32> = want.iter().map(|&w| (w as f32).to_bits()).collect();
                assert_eq!(bits(&v(&got)), rounded, "{name} {ashape:?} {bshape:?}");
            }
        }
    }
}

#[test]
fn broadcast_ops_do_not_depend_on_the_thread_count() {
    let mut rng = SplitMix64(8);
    let a = rand_t(&mut rng, &[4, 64, 33, 33], 2.0);
    let b = rand_t(&mut rng, &[64, 1, 1], 2.0);
    let base = backend(1, Numerics::Fast);
    let add = bits(&v(&base.broadcast_add_forward(&a, &b).unwrap()));
    let mul = bits(&v(&base.broadcast_mul_forward(&b, &a).unwrap()));
    for threads in THREADS {
        let cpu = backend(threads, Numerics::Fast);
        assert_eq!(bits(&v(&cpu.broadcast_add_forward(&a, &b).unwrap())), add);
        assert_eq!(bits(&v(&cpu.broadcast_mul_forward(&b, &a).unwrap())), mul);
    }
}

#[test]
fn broadcast_refuses_shapes_that_do_not_align() {
    let cpu = backend(1, Numerics::Exact);
    let a = t(&[0.0; 6], &[2, 3]);
    assert_shape(cpu.broadcast_add_forward(&a, &t(&[0.0; 4], &[4])));
    assert_shape(cpu.broadcast_mul_forward(&a, &t(&[0.0; 4], &[2, 2])));
    assert_shape(cpu.broadcast_add_forward(&a, &t(&[0.0; 6], &[3, 2])));
    // Rank above the cap.
    let deep = t(&[0.0; 2], &[1, 1, 1, 1, 1, 1, 1, 1, 2]);
    assert_shape(cpu.broadcast_add_forward(&deep, &t(&[0.0], &[1])));
}

// ---- Scale ----

#[test]
fn scale_multiplies_every_element_once() {
    let mut rng = SplitMix64(9);
    let x = rand_t(&mut rng, &[3, 5, 7], 3.0);
    let scale = 0.176_776_7f32;
    for threads in THREADS {
        let got = backend(threads, Numerics::Fast)
            .scale_forward(&x, scale)
            .unwrap();
        let want: Vec<f32> = v(&x).iter().map(|&e| e * scale).collect();
        assert_eq!(got.shape(), x.shape());
        assert_eq!(bits(&v(&got)), bits(&want));
    }
    // The input is untouched.
    assert_eq!(v(&x).len(), 105);
}

// ---- Softmax ----

fn softmax_ref(x: &[f64], dim: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(dim) {
        let max = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        let e: Vec<f64> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f64 = e.iter().sum();
        out.extend(e.iter().map(|v| v / sum));
    }
    out
}

#[test]
fn softmax_matches_an_f64_reference_and_rows_sum_to_one() {
    let mut rng = SplitMix64(10);
    for shape in [vec![5, 16], vec![2, 3, 49], vec![9], vec![4, 1]] {
        let dim = *shape.last().unwrap();
        let x = rand_t(&mut rng, &shape, 8.0);
        let want = softmax_ref(&f64s(&x), dim);
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let got = backend(1, numerics).softmax_forward(&x).unwrap();
            assert_eq!(got.shape(), shape.as_slice());
            let got = v(&got);
            assert_close(&format!("{shape:?} {numerics:?}"), &got, &want, 2e-6);
            for row in got.chunks_exact(dim) {
                let sum: f32 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "row sums to {sum}");
            }
        }
    }
}

#[test]
fn softmax_is_stable_for_large_scores_and_uniform_for_equal_ones() {
    let big = t(&[1000.0, 1001.0, 999.0, -3.0e38, 3.0e38, 0.0], &[2, 3]);
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let cpu = backend(1, numerics);
        let got = v(&cpu.softmax_forward(&big).unwrap());
        assert!(got.iter().all(|p| p.is_finite() && *p >= 0.0));
        let first = softmax_ref(&[1000.0, 1001.0, 999.0], 3);
        assert_close("large", &got[..3], &first, 2e-6);
        // The spread between -3e38 and 3e38 is -inf after the shift: weight 0.
        assert_eq!(got[3], 0.0, "{numerics:?}");
        assert_eq!(got[4], 1.0, "{numerics:?}");
        assert_eq!(got[5], 0.0, "{numerics:?}");
        let flat = v(&cpu.softmax_forward(&t(&[2.5; 8], &[8])).unwrap());
        assert!(flat.iter().all(|&p| p == 0.125), "{numerics:?}");
    }
}

#[test]
fn softmax_bits_do_not_depend_on_the_thread_count() {
    let mut rng = SplitMix64(11);
    let x = rand_t(&mut rng, &[4, 3000, 37], 6.0);
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let base = bits(&v(&backend(1, numerics).softmax_forward(&x).unwrap()));
        for threads in THREADS {
            let got = backend(threads, numerics).softmax_forward(&x).unwrap();
            assert_eq!(bits(&v(&got)), base, "{numerics:?} x{threads}");
        }
    }
}

// ---- Refusals ----

fn nan_in(shape: &[usize]) -> Tensor {
    let mut data = vec![0.5f32; shape.iter().product()];
    data[0] = f32::NAN;
    t(&data, shape)
}

fn inf_in(shape: &[usize]) -> Tensor {
    let mut data = vec![0.5f32; shape.iter().product()];
    *data.last_mut().unwrap() = f32::INFINITY;
    t(&data, shape)
}

#[test]
fn every_op_refuses_a_non_finite_operand() {
    let cpu = backend(2, Numerics::Exact);
    let ok2 = t(&[0.5; 6], &[2, 3]);
    let w3 = t(&[1.0; 3], &[3]);
    let ok4 = t(&[0.5; 24], &[2, 3, 2, 2]);
    for bad in [nan_in(&[2, 3]), inf_in(&[2, 3])] {
        assert_nonfinite(cpu.layer_norm_forward(&bad, &w3, &w3, 1e-5));
        assert_nonfinite(cpu.gelu_erf_forward(&bad));
        assert_nonfinite(cpu.softmax_forward(&bad));
        assert_nonfinite(cpu.scale_forward(&bad, 2.0));
        assert_nonfinite(cpu.broadcast_add_forward(&bad, &ok2));
        assert_nonfinite(cpu.broadcast_mul_forward(&ok2, &bad));
        assert_nonfinite(cpu.linear_bias_forward(
            &bad,
            &t(&[1.0; 6], &[2, 3]),
            &t(&[0.0; 2], &[2]),
        ));
    }
    for bad in [nan_in(&[3]), inf_in(&[3])] {
        assert_nonfinite(cpu.layer_norm_forward(&ok2, &bad, &w3, 1e-5));
        assert_nonfinite(cpu.layer_norm_forward(&ok2, &w3, &bad, 1e-5));
        assert_nonfinite(cpu.channel_layer_norm_forward(&ok4, &bad, &w3, 1e-5));
        assert_nonfinite(cpu.channel_layer_norm_forward(&ok4, &w3, &bad, 1e-5));
        assert_nonfinite(cpu.linear_bias_forward(&ok2, &t(&[1.0; 9], &[3, 3]), &bad));
    }
    for bad in [nan_in(&[2, 3, 2, 2]), inf_in(&[2, 3, 2, 2])] {
        assert_nonfinite(cpu.channel_layer_norm_forward(&bad, &w3, &w3, 1e-5));
    }
    for eps in [f32::NAN, f32::INFINITY] {
        assert_nonfinite(cpu.layer_norm_forward(&ok2, &w3, &w3, eps));
        assert_nonfinite(cpu.channel_layer_norm_forward(&ok4, &w3, &w3, eps));
    }
    for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_nonfinite(cpu.scale_forward(&ok2, scale));
    }
}

#[test]
fn an_output_that_overflows_is_refused() {
    let cpu = backend(1, Numerics::Exact);
    let big = t(&[3.0e38, 1.0], &[2]);
    assert_nonfinite(cpu.scale_forward(&big, 4.0));
    assert_nonfinite(cpu.broadcast_add_forward(&big, &t(&[3.0e38], &[1])));
    assert_nonfinite(cpu.broadcast_mul_forward(&big, &t(&[4.0], &[1])));
}

#[test]
fn shapes_that_do_not_fit_are_shape_errors() {
    let cpu = backend(1, Numerics::Exact);
    let x = t(&[0.0; 24], &[2, 3, 4]);
    let w4 = t(&[1.0; 4], &[4]);
    let w3 = t(&[1.0; 3], &[3]);
    assert_shape(cpu.layer_norm_forward(&x, &w3, &w4, 1e-5));
    assert_shape(cpu.layer_norm_forward(&x, &w4, &w3, 1e-5));
    assert_shape(cpu.layer_norm_forward(&t(&[1.0], &[]), &w4, &w4, 1e-5));
    assert_shape(cpu.channel_layer_norm_forward(&x, &w4, &w4, 1e-5));
    assert_shape(cpu.channel_layer_norm_forward(&t(&[1.0; 3], &[3]), &w3, &w3, 1e-5));
    assert_shape(cpu.softmax_forward(&t(&[1.0], &[])));
    let lw = t(&[0.0; 8], &[2, 4]);
    assert_shape(cpu.linear_bias_forward(&x, &lw, &t(&[0.0; 3], &[3])));
    assert_shape(cpu.linear_bias_forward(&x, &t(&[0.0; 6], &[2, 3]), &t(&[0.0; 2], &[2])));
    // Empty tensors are refused like everywhere else.
    assert_shape(cpu.gelu_erf_forward(&t(&[], &[0, 3])));
}

#[test]
fn an_over_budget_output_is_capacity_exceeded_and_releases_its_charge() {
    let cpu = CpuBackend::new(Budget::new(4096));
    let x = t(&[0.5; 4096], &[64, 64]);
    let w = t(&[1.0; 64], &[64]);
    let before = cpu.budget().live_bytes().unwrap();
    assert_capacity(cpu.gelu_erf_forward(&x));
    assert_capacity(cpu.softmax_forward(&x));
    assert_capacity(cpu.scale_forward(&x, 2.0));
    assert_capacity(cpu.layer_norm_forward(&x, &w, &w, 1e-5));
    assert_capacity(cpu.broadcast_add_forward(&x, &w));
    assert_eq!(cpu.budget().live_bytes().unwrap(), before);
}
