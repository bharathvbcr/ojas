//! The EmbeddingGemma 2 encoder ops on the CPU: `gelu_tanh_forward` and
//! `gelu_tanh_backward`, `seq_mean_forward` (the masked mean pool of a
//! sentence embedding model) with `seq_mean_backward`, and
//! `l2_normalize_forward`.
//!
//! Each op is held to an independent `f64` reference written here (libm
//! `tanh` and `exp` for the GELU, plain loops for the two reductions), at the
//! edge rows the contracts name (zero vectors, length 0, length over `T`,
//! large and subnormal-range values), to the same bits at every thread count
//! and under both `Numerics`, and to the refusals under each op's own name.
//! The torch goldens are in `gemma_ops_golden.rs`; nothing here needs torch.

use ojas_core::{Backend, Budget, DType, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_nonfinite, assert_range, assert_shape, bits, f32t, u32t, SplitMix64};

fn backend(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 33), threads)
        .unwrap()
        .with_numerics(numerics)
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

/// Every thread count and numerics tier this file compares.
fn tiers() -> Vec<CpuBackend> {
    vec![
        backend(1, Numerics::Exact),
        backend(2, Numerics::Exact),
        backend(5, Numerics::Exact),
        backend(3, Numerics::Fast),
    ]
}

// ---- GELU (tanh) ----

const KAPPA: f64 = 0.797_884_560_802_865_4;
const CUBIC: f64 = 0.044_715;

/// `x * sigmoid(u)` with libm `exp`: the identity form, accurate in both
/// tails (a different `exp` from the backend's).
fn gelu_ref(x: f64) -> f64 {
    let u = 2.0 * KAPPA * (x + CUBIC * x * x * x);
    let s = if u >= 0.0 {
        1.0 / (1.0 + (-u).exp())
    } else {
        u.exp() / (1.0 + u.exp())
    };
    x * s
}

/// torch's own form, `0.5 x (1 + tanh(..))`, accurate away from the tails.
fn gelu_torch_form(x: f64) -> f64 {
    0.5 * x * (1.0 + (KAPPA * (x + CUBIC * x * x * x)).tanh())
}

/// torch's backward form, `0.5 (1 + t) + 0.5 x (1 - t^2) kappa (1 + 3 c x^2)`.
fn slope_torch_form(x: f64) -> f64 {
    let t = (KAPPA * (x + CUBIC * x * x * x)).tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * KAPPA * (1.0 + 3.0 * CUBIC * x * x)
}

/// The derivative of [`gelu_ref`] by the quotient rule on `x * sigmoid(u)`.
fn slope_ref(x: f64) -> f64 {
    let u = 2.0 * KAPPA * (x + CUBIC * x * x * x);
    let e = (-u.abs()).exp();
    let s = if u >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    };
    let du = 2.0 * KAPPA * (1.0 + 3.0 * CUBIC * x * x);
    s + x * (e / ((1.0 + e) * (1.0 + e))) * du
}

fn gelu_grid() -> Vec<f32> {
    let mut xs: Vec<f32> = Vec::new();
    let mut x = -10.0f32;
    while x <= 10.0 {
        xs.push(x);
        x += 0.0613;
    }
    // Near zero, a power of two apart, both signs.
    for k in 1..24 {
        let e = 2.0f32.powi(-k);
        xs.push(e);
        xs.push(-e);
    }
    xs
}

#[test]
fn gelu_tanh_forward_matches_f64_references_up_to_ten() {
    let xs = gelu_grid();
    let n = xs.len();
    let cpu = backend(2, Numerics::Exact);
    let y = v(&cpu.gelu_tanh_forward(&f32t(&cpu, &xs, &[n])).unwrap());
    for (&x, &got) in xs.iter().zip(&y) {
        let want = gelu_ref(f64::from(x));
        // One f32 rounding, plus the reference's libm exp.
        assert!(
            (f64::from(got) - want).abs() <= 1.3e-7 * want.abs() + 1e-45,
            "x = {x}: {got:e} vs {want:e}"
        );
        // torch's form agrees wherever 1 + tanh does not cancel.
        if x > -3.0 {
            let torch = gelu_torch_form(f64::from(x));
            assert!(
                (f64::from(got) - torch).abs() <= 1.3e-7 * torch.abs() + 1e-14,
                "x = {x}: {got:e} vs torch form {torch:e}"
            );
        }
    }
}

#[test]
fn gelu_tanh_backward_matches_f64_references_up_to_ten() {
    let xs = gelu_grid();
    let n = xs.len();
    let mut rng = SplitMix64(5);
    let gs = rng.vec(n, 3.0);
    let cpu = backend(2, Numerics::Exact);
    let dx = v(&cpu
        .gelu_tanh_backward(&f32t(&cpu, &xs, &[n]), &f32t(&cpu, &gs, &[n]))
        .unwrap());
    for ((&x, &g), &got) in xs.iter().zip(&gs).zip(&dx) {
        let want = f64::from(g) * slope_ref(f64::from(x));
        assert!(
            (f64::from(got) - want).abs() <= 1.3e-7 * want.abs() + 1e-37,
            "x = {x}, g = {g}: {got:e} vs {want:e}"
        );
        if x.abs() <= 6.0 {
            let torch = f64::from(g) * slope_torch_form(f64::from(x));
            assert!(
                (f64::from(got) - torch).abs() <= 1.3e-7 * torch.abs() + 1e-13,
                "x = {x}, g = {g}: {got:e} vs torch form {torch:e}"
            );
        }
    }
}

#[test]
fn gelu_tanh_edge_values() {
    let cpu = backend(1, Numerics::Exact);
    let xs = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        9.0,
        50.0,
        1e30,
        f32::MAX,
        -10.5,
        -11.5,
        -14.0,
        -1e30,
        f32::MIN,
    ];
    let y = v(&cpu
        .gelu_tanh_forward(&f32t(&cpu, &xs, &[xs.len()]))
        .unwrap());
    assert_eq!(y[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(y[1].to_bits(), (-0.0f32).to_bits());
    // gelu_tanh(1) = 0.8411919906082768; gelu_tanh(-1) = -0.15880800939172324.
    assert!((f64::from(y[2]) - 0.841_191_990_608_276_8).abs() < 1e-7);
    assert!((f64::from(y[3]) + 0.158_808_009_391_723_24).abs() < 1e-7);
    // Far positive is x itself.
    for i in [4, 5, 6, 7] {
        assert_eq!(y[i], xs[i], "x = {}", xs[i]);
    }
    // The f32 subnormal tail is a rounded nonzero, then a signed zero.
    assert!(y[8] < 0.0 && y[8] > -1e-41, "x = -10.5: {:e}", y[8]);
    for i in [9, 10, 11, 12] {
        assert_eq!(y[i].to_bits(), (-0.0f32).to_bits(), "x = {}", xs[i]);
    }
    // The derivative at 0 is 1/2, and far positive / negative are 1 and 0.
    let g = [1.0f32; 13];
    let dx = v(&cpu
        .gelu_tanh_backward(&f32t(&cpu, &xs, &[xs.len()]), &f32t(&cpu, &g, &[g.len()]))
        .unwrap());
    assert_eq!(dx[0], 0.5);
    assert_eq!(dx[1], 0.5);
    for i in [4, 5, 6, 7] {
        assert_eq!(dx[i], 1.0, "x = {}", xs[i]);
    }
    for i in [10, 11, 12] {
        assert_eq!(dx[i], 0.0, "x = {}", xs[i]);
    }
}

#[test]
fn gelu_tanh_is_bit_identical_across_threads_and_numerics() {
    let n = 200_003;
    let mut rng = SplitMix64(7);
    let xs = rng.vec(n, 12.0);
    let gs = rng.vec(n, 2.0);
    let mut first: Option<(Vec<u32>, Vec<u32>)> = None;
    for cpu in tiers() {
        let x = f32t(&cpu, &xs, &[n]);
        let y = bits(&v(&cpu.gelu_tanh_forward(&x).unwrap()));
        let dx = bits(&v(&cpu
            .gelu_tanh_backward(&x, &f32t(&cpu, &gs, &[n]))
            .unwrap()));
        match &first {
            None => first = Some((y, dx)),
            Some((y0, dx0)) => {
                assert_eq!(&y, y0, "forward bits changed");
                assert_eq!(&dx, dx0, "backward bits changed");
            }
        }
    }
}

#[test]
fn gelu_tanh_refuses_what_it_cannot_compute_under_its_own_name() {
    let cpu = backend(1, Numerics::Exact);
    let named = |r: Result<Tensor, OjasError>, want: &str| match r {
        Err(OjasError::NonFinite { op })
        | Err(OjasError::Shape { op, .. })
        | Err(OjasError::Dtype { op, .. }) => assert_eq!(op, want),
        other => panic!("{want}: expected a refusal, got {other:?}"),
    };
    let x = f32t(&cpu, &[0.5, 1.5], &[2]);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let b = f32t(&cpu, &[0.5, bad], &[2]);
        assert_nonfinite(cpu.gelu_tanh_forward(&b));
        named(cpu.gelu_tanh_forward(&b), "gelu_tanh_forward");
        assert_nonfinite(cpu.gelu_tanh_backward(&b, &x));
        assert_nonfinite(cpu.gelu_tanh_backward(&x, &b));
        named(cpu.gelu_tanh_backward(&b, &x), "gelu_tanh_backward");
    }
    // A gradient that does not match the input.
    let short = f32t(&cpu, &[1.0], &[1]);
    assert_shape(cpu.gelu_tanh_backward(&x, &short));
    named(cpu.gelu_tanh_backward(&x, &short), "gelu_tanh_backward");
    // An integer tensor.
    let ids = u32t(&cpu, &[1, 2], &[2]);
    named(cpu.gelu_tanh_forward(&ids), "gelu_tanh_forward");
    // gelu_tanh'(1.5) is above 1: a gradient at the f32 maximum overflows.
    let big = f32t(&cpu, &[f32::MAX, f32::MAX], &[2]);
    named(cpu.gelu_tanh_backward(&x, &big), "gelu_tanh_backward");
    assert_nonfinite(cpu.gelu_tanh_backward(&x, &big));
}

// ---- masked mean (seq_mean) ----

/// The mean of the first `min(len, t)` rows of each sequence, as
/// sentence-transformers' `Pooling(mean)` computes it: the masked sum over
/// `clamp(count, 1e-9)`, in `f64` with an ascending sum, rounded once.
fn mean_ref(x: &[f32], lens: &[u32], t: usize, d: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for (b, &len) in lens.iter().enumerate() {
        let rows = (len as usize).min(t);
        for col in 0..d {
            let mut sum = 0.0f64;
            for row in 0..rows {
                sum += f64::from(x[(b * t + row) * d + col]);
            }
            out.push((sum / (rows as f64).max(1e-9)) as f32);
        }
    }
    out
}

#[test]
fn seq_mean_is_the_masked_mean_of_the_leading_rows() {
    // 37 columns cross the kernel's 16-column blocks; lengths cover a full
    // sequence, a short one, zero, one row, and one over T.
    let (b, t, d) = (5usize, 9usize, 37usize);
    let cpu = backend(2, Numerics::Exact);
    let mut rng = SplitMix64(11);
    let x = rng.vec(b * t * d, 4.0);
    let lens = [9u32, 4, 0, 1, 99];
    let got = cpu
        .seq_mean_forward(&f32t(&cpu, &x, &[b, t, d]), &u32t(&cpu, &lens, &[b]))
        .unwrap();
    assert_eq!(got.shape(), [b, d]);
    assert_eq!(got.dtype(), DType::F32);
    let got = v(&got);
    assert_eq!(bits(&got), bits(&mean_ref(&x, &lens, t, d)));
    // Length 0 is +0.0 everywhere (0 / 1e-9 in sentence-transformers).
    assert!(got[2 * d..3 * d].iter().all(|e| e.to_bits() == 0));
    // Length 1 is the row itself.
    assert_eq!(bits(&got[3 * d..4 * d]), bits(&x[3 * t * d..3 * t * d + d]));
    // A closed form: the mean of rows 0..2 of [1,2,3 / 5,6,7] is [3, 4, 5].
    let small = f32t(
        &cpu,
        &[1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 9.0, 9.0],
        &[1, 3, 3],
    );
    let mean = cpu
        .seq_mean_forward(&small, &u32t(&cpu, &[2], &[1]))
        .unwrap();
    assert_eq!(v(&mean), [3.0, 4.0, 5.0]);
}

#[test]
fn seq_mean_never_reads_rows_at_or_past_the_length() {
    let (b, t, d) = (3usize, 7usize, 20usize);
    let cpu = backend(1, Numerics::Exact);
    let mut rng = SplitMix64(12);
    let clean = rng.vec(b * t * d, 1.0);
    let lens = [3u32, 7, 0];
    let mut poisoned = clean.clone();
    for (bi, &len) in lens.iter().enumerate() {
        for row in len as usize..t {
            poisoned[(bi * t + row) * d..(bi * t + row + 1) * d].fill(3.0e38);
        }
    }
    let run = |x: &[f32]| {
        v(&cpu
            .seq_mean_forward(&f32t(&cpu, x, &[b, t, d]), &u32t(&cpu, &lens, &[b]))
            .unwrap())
    };
    assert_eq!(bits(&run(&clean)), bits(&run(&poisoned)));
}

#[test]
fn seq_mean_is_bit_identical_across_threads_and_numerics() {
    let (b, t, d) = (67usize, 33usize, 49usize);
    let mut rng = SplitMix64(13);
    let x = rng.vec(b * t * d, 6.0);
    let lens: Vec<u32> = (0..b).map(|_| rng.below(t + 3) as u32).collect();
    let mut first: Option<Vec<u32>> = None;
    for cpu in tiers() {
        let got = bits(&v(&cpu
            .seq_mean_forward(&f32t(&cpu, &x, &[b, t, d]), &u32t(&cpu, &lens, &[b]))
            .unwrap()));
        assert_eq!(got, bits(&mean_ref(&x, &lens, t, d)));
        match &first {
            None => first = Some(got),
            Some(want) => assert_eq!(&got, want),
        }
    }
}

#[test]
fn seq_mean_refuses_what_it_cannot_pool_under_its_own_name() {
    let cpu = backend(1, Numerics::Exact);
    let (b, t, d) = (2usize, 3usize, 4usize);
    let x = f32t(&cpu, &vec![0.5; b * t * d], &[b, t, d]);
    let lens = u32t(&cpu, &[3, 1], &[b]);
    cpu.seq_mean_forward(&x, &lens).unwrap();
    let named = |r: Result<Tensor, OjasError>| match r {
        Err(OjasError::NonFinite { op })
        | Err(OjasError::Shape { op, .. })
        | Err(OjasError::Dtype { op, .. }) => assert_eq!(op, "seq_mean_forward"),
        other => panic!("expected a refusal, got {other:?}"),
    };
    // A float tensor for the lengths.
    let float_lens = f32t(&cpu, &[3.0, 1.0], &[b]);
    named(cpu.seq_mean_forward(&x, &float_lens));
    // One length per sequence.
    let three = u32t(&cpu, &[3, 1, 2], &[3]);
    assert_shape(cpu.seq_mean_forward(&x, &three));
    named(cpu.seq_mean_forward(&x, &three));
    // x is rank 3.
    let flat = f32t(&cpu, &vec![0.5; t * d], &[t, d]);
    assert_shape(cpu.seq_mean_forward(&flat, &lens));
    // A NaN or infinity anywhere in x, even past the length, is refused.
    for bad in [f32::NAN, f32::INFINITY] {
        let mut data = vec![0.5; b * t * d];
        data[(t + 2) * d] = bad;
        let poisoned = f32t(&cpu, &data, &[b, t, d]);
        assert_nonfinite(cpu.seq_mean_forward(&poisoned, &lens));
    }
    // A budget with 16 bytes left for the 32-byte [B, D] output refuses, it
    // does not clamp.
    let tight = CpuBackend::new(Budget::new((b * t * d * 4 + b * 4 + 16) as u64));
    let tx = f32t(&tight, &vec![0.5; b * t * d], &[b, t, d]);
    let tl = u32t(&tight, &[3, 1], &[b]);
    let r = tight.seq_mean_forward(&tx, &tl);
    assert!(
        matches!(r, Err(OjasError::CapacityExceeded { .. })),
        "{r:?}"
    );
}

/// `grad[b, d] / min(len[b], T)` on the rows the mean read, `+0.0` past
/// them: the gradient of [`mean_ref`], written independently of it.
fn mean_grad_ref(grad: &[f32], lens: &[u32], t: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; lens.len() * t * d];
    for (b, &len) in lens.iter().enumerate() {
        let rows = (len as usize).min(t);
        if rows == 0 {
            continue;
        }
        for row in 0..rows {
            for col in 0..d {
                out[(b * t + row) * d + col] = (f64::from(grad[b * d + col]) / rows as f64) as f32;
            }
        }
    }
    out
}

#[test]
fn seq_mean_backward_divides_by_the_rows_the_mean_read() {
    // The same lengths the forward test uses: full, short, zero, one row,
    // and one over T.
    let (b, t, d) = (5usize, 9usize, 37usize);
    let cpu = backend(2, Numerics::Exact);
    let mut rng = SplitMix64(31);
    let x = rng.vec(b * t * d, 4.0);
    let grad = rng.vec(b * d, 2.0);
    let lens = [9u32, 4, 0, 1, 99];
    let got = cpu
        .seq_mean_backward(
            &f32t(&cpu, &x, &[b, t, d]),
            &u32t(&cpu, &lens, &[b]),
            &f32t(&cpu, &grad, &[b, d]),
        )
        .unwrap();
    assert_eq!(got.shape(), [b, t, d]);
    assert_eq!(got.dtype(), DType::F32);
    let got = v(&got);
    assert_eq!(bits(&got), bits(&mean_grad_ref(&grad, &lens, t, d)));
    // Sequence 4 asks for 99 of 9 rows: every row takes grad / 9, not
    // grad / 99, and none is left out.
    for row in 0..t {
        for col in 0..d {
            let want = (f64::from(grad[4 * d + col]) / 9.0) as f32;
            assert_eq!(got[(4 * t + row) * d + col].to_bits(), want.to_bits());
        }
    }
    // Sequence 2 has length 0: the whole block is +0.0.
    assert!(got[(2 * t) * d..(3 * t) * d]
        .iter()
        .all(|e| e.to_bits() == 0));
    // Sequence 1 has length 4 of 9: rows 4.. are +0.0.
    assert!(got[(t + 4) * d..(t + 9) * d]
        .iter()
        .all(|e| e.to_bits() == 0));
}

#[test]
fn seq_mean_backward_is_bit_identical_across_threads_and_numerics() {
    let (b, t, d) = (67usize, 33usize, 49usize);
    let mut rng = SplitMix64(32);
    let x = rng.vec(b * t * d, 6.0);
    let grad = rng.vec(b * d, 3.0);
    let lens: Vec<u32> = (0..b).map(|_| rng.below(t + 3) as u32).collect();
    let mut first: Option<Vec<u32>> = None;
    for cpu in tiers() {
        let got = bits(&v(&cpu
            .seq_mean_backward(
                &f32t(&cpu, &x, &[b, t, d]),
                &u32t(&cpu, &lens, &[b]),
                &f32t(&cpu, &grad, &[b, d]),
            )
            .unwrap()));
        assert_eq!(got, bits(&mean_grad_ref(&grad, &lens, t, d)));
        match &first {
            None => first = Some(got),
            Some(want) => assert_eq!(&got, want),
        }
    }
}

#[test]
fn seq_mean_backward_refuses_what_it_cannot_spread_under_its_own_name() {
    let cpu = backend(1, Numerics::Exact);
    let (b, t, d) = (2usize, 3usize, 4usize);
    let x = f32t(&cpu, &vec![0.5; b * t * d], &[b, t, d]);
    let lens = u32t(&cpu, &[3, 1], &[b]);
    let grad = f32t(&cpu, &vec![0.25; b * d], &[b, d]);
    cpu.seq_mean_backward(&x, &lens, &grad).unwrap();
    let named = |r: Result<Tensor, OjasError>| match r {
        Err(OjasError::NonFinite { op })
        | Err(OjasError::Shape { op, .. })
        | Err(OjasError::Dtype { op, .. }) => assert_eq!(op, "seq_mean_backward"),
        other => panic!("expected a refusal, got {other:?}"),
    };
    // A float tensor for the lengths.
    named(cpu.seq_mean_backward(&x, &f32t(&cpu, &[3.0, 1.0], &[b]), &grad));
    // One length per sequence.
    let three = u32t(&cpu, &[3, 1, 2], &[3]);
    assert_shape(cpu.seq_mean_backward(&x, &three, &grad));
    named(cpu.seq_mean_backward(&x, &three, &grad));
    // x is rank 3.
    let flat = f32t(&cpu, &vec![0.5; t * d], &[t, d]);
    assert_shape(cpu.seq_mean_backward(&flat, &lens, &grad));
    // The gradient is the forward's [B, D] output.
    let wide = f32t(&cpu, &vec![0.25; b * t * d], &[b, t, d]);
    assert_shape(cpu.seq_mean_backward(&x, &lens, &wide));
    named(cpu.seq_mean_backward(&x, &lens, &wide));
    // A NaN or infinity in the gradient is refused.
    for bad in [f32::NAN, f32::INFINITY] {
        let mut data = vec![0.25; b * d];
        data[d] = bad;
        assert_nonfinite(cpu.seq_mean_backward(&x, &lens, &f32t(&cpu, &data, &[b, d])));
    }
    // The backward reads none of x's values, so a non-finite one in x --
    // which the forward refuses -- does not refuse here.
    let mut poisoned = vec![0.5; b * t * d];
    poisoned[(t + 2) * d] = f32::NAN;
    cpu.seq_mean_backward(&f32t(&cpu, &poisoned, &[b, t, d]), &lens, &grad)
        .expect("x is read for its shape only");
    // A budget with 16 bytes left for the 96-byte [B, T, D] gradient
    // refuses, it does not clamp.
    let tight = CpuBackend::new(Budget::new((b * t * d * 4 + b * 4 + b * d * 4 + 16) as u64));
    let r = tight.seq_mean_backward(
        &f32t(&tight, &vec![0.5; b * t * d], &[b, t, d]),
        &u32t(&tight, &[3, 1], &[b]),
        &f32t(&tight, &vec![0.25; b * d], &[b, d]),
    );
    assert!(
        matches!(r, Err(OjasError::CapacityExceeded { .. })),
        "{r:?}"
    );
}

// ---- L2 normalization ----

/// `x / max(||x||, eps)` row by row in `f64`, rounded once.
fn normalize_ref(x: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = Vec::new();
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let norm = row
            .iter()
            .map(|&e| f64::from(e) * f64::from(e))
            .sum::<f64>()
            .sqrt();
        let denom = norm.max(f64::from(eps));
        out.extend(row.iter().map(|&e| (f64::from(e) / denom) as f32));
    }
    out
}

#[test]
fn l2_normalize_divides_each_row_by_its_norm() {
    let dim = 31usize;
    let mut rng = SplitMix64(21);
    let mut data = rng.vec(8 * dim, 3.0);
    // Rows: random x4, zero, negative zeros, tiny, large.
    data[4 * dim..5 * dim].fill(0.0);
    data[5 * dim..6 * dim].fill(-0.0);
    data[6 * dim..7 * dim].fill(1e-20);
    data[7 * dim..8 * dim].iter_mut().for_each(|e| *e *= 1e30);
    let cpu = backend(2, Numerics::Exact);
    let x = f32t(&cpu, &data, &[8, dim]);
    let eps = 1e-12;
    let y = cpu.l2_normalize_forward(&x, eps).unwrap();
    assert_eq!(y.shape(), x.shape());
    let y = v(&y);
    assert_eq!(bits(&y), bits(&normalize_ref(&data, 8, dim, eps)));
    // Unit rows where the norm is above eps.
    for r in [0usize, 1, 2, 3, 7] {
        let sq: f64 = y[r * dim..(r + 1) * dim]
            .iter()
            .map(|&e| f64::from(e) * f64::from(e))
            .sum();
        assert!((sq - 1.0).abs() < 1e-6, "row {r}: |y|^2 = {sq}");
    }
    // A zero row stays zero (+0.0), negative zeros stay negative zeros.
    assert!(y[4 * dim..5 * dim].iter().all(|e| e.to_bits() == 0));
    assert!(y[5 * dim..6 * dim]
        .iter()
        .all(|e| e.to_bits() == (-0.0f32).to_bits()));
    // A row whose norm (5.6e-20) is below eps divides by eps.
    let tiny = 1e-20f64 / f64::from(eps);
    assert!(y[6 * dim..7 * dim]
        .iter()
        .all(|&e| (f64::from(e) - tiny).abs() <= 3e-7 * tiny));
}

#[test]
fn l2_normalize_clamps_the_norm_at_eps() {
    let cpu = backend(1, Numerics::Exact);
    let x = f32t(&cpu, &[3e-7, 4e-7, 3e-7, 4e-7], &[2, 2]);
    // The norm is 5e-7: below eps = 1e-6 the divisor is eps, so the row
    // shrinks; at eps = 1e-12 it is a unit row.
    let big = v(&cpu.l2_normalize_forward(&x, 1e-6).unwrap());
    assert!(
        (big[0] - 0.3).abs() < 1e-6 && (big[1] - 0.4).abs() < 1e-6,
        "{big:?}"
    );
    let unit = v(&cpu.l2_normalize_forward(&x, 1e-12).unwrap());
    assert!(
        (unit[0] - 0.6).abs() < 1e-6 && (unit[1] - 0.8).abs() < 1e-6,
        "{unit:?}"
    );
}

#[test]
fn l2_normalize_of_a_matryoshka_prefix_is_the_slice_then_normalize() {
    let (rows, dim) = (6usize, 256usize);
    let cpu = backend(2, Numerics::Exact);
    let mut rng = SplitMix64(22);
    let data = rng.vec(rows * dim, 2.0);
    let x = f32t(&cpu, &data, &[rows, dim]);
    for prefix in [128usize, 64] {
        let head = cpu.slice(&x, 1, 0, prefix).unwrap();
        let got = v(&cpu.l2_normalize_forward(&head, 1e-12).unwrap());
        let mut flat = Vec::new();
        for r in 0..rows {
            flat.extend_from_slice(&data[r * dim..r * dim + prefix]);
        }
        assert_eq!(bits(&got), bits(&normalize_ref(&flat, rows, prefix, 1e-12)));
    }
}

#[test]
fn l2_normalize_is_bit_identical_across_threads_and_numerics() {
    let (rows, dim) = (1013usize, 77usize);
    let mut rng = SplitMix64(23);
    let data = rng.vec(rows * dim, 5.0);
    let mut first: Option<Vec<u32>> = None;
    for cpu in tiers() {
        let got = bits(&v(&cpu
            .l2_normalize_forward(&f32t(&cpu, &data, &[rows, dim]), 1e-12)
            .unwrap()));
        match &first {
            None => first = Some(got),
            Some(want) => assert_eq!(&got, want),
        }
    }
}

#[test]
fn l2_normalize_refuses_what_it_cannot_normalize_under_its_own_name() {
    let cpu = backend(1, Numerics::Exact);
    let x = f32t(&cpu, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    cpu.l2_normalize_forward(&x, 1e-12).unwrap();
    // eps must be finite and above 0.
    assert_nonfinite(cpu.l2_normalize_forward(&x, f32::NAN));
    assert_nonfinite(cpu.l2_normalize_forward(&x, f32::INFINITY));
    assert_range(cpu.l2_normalize_forward(&x, 0.0));
    assert_range(cpu.l2_normalize_forward(&x, -1e-12));
    // The operand is [rows, dim].
    assert_shape(cpu.l2_normalize_forward(&f32t(&cpu, &[1.0, 2.0], &[2]), 1e-12));
    assert_shape(cpu.l2_normalize_forward(&f32t(&cpu, &[1.0; 8], &[2, 2, 2]), 1e-12));
    let ids = u32t(&cpu, &[1, 2], &[1, 2]);
    assert!(matches!(
        cpu.l2_normalize_forward(&ids, 1e-12),
        Err(OjasError::Dtype {
            op: "l2_normalize_forward",
            ..
        })
    ));
    for bad in [f32::NAN, f32::INFINITY] {
        let b = f32t(&cpu, &[1.0, bad], &[1, 2]);
        match cpu.l2_normalize_forward(&b, 1e-12) {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, "l2_normalize_forward"),
            other => panic!("expected NonFinite, got {other:?}"),
        }
    }
}

/// The gradient of [`normalize_ref`], written independently of it: above the
/// clamp `(g - y (y . g)) / n` with `y = x / n`, at or below it `g / eps`.
fn normalize_grad_ref(x: &[f32], g: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let eps = f64::from(eps);
    let mut out = Vec::new();
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let gr = &g[r * dim..(r + 1) * dim];
        let norm = row
            .iter()
            .map(|&e| f64::from(e) * f64::from(e))
            .sum::<f64>()
            .sqrt();
        if norm <= eps {
            out.extend(gr.iter().map(|&e| (f64::from(e) / eps) as f32));
            continue;
        }
        let dot: f64 = row
            .iter()
            .zip(gr)
            .map(|(&v, &gv)| (f64::from(v) / norm) * f64::from(gv))
            .sum();
        out.extend(
            row.iter()
                .zip(gr)
                .map(|(&v, &gv)| ((f64::from(gv) - (f64::from(v) / norm) * dot) / norm) as f32),
        );
    }
    out
}

#[test]
fn l2_normalize_backward_projects_above_eps_and_divides_by_eps_below() {
    let dim = 31usize;
    let mut rng = SplitMix64(41);
    let mut data = rng.vec(8 * dim, 3.0);
    // Rows: random x4, zero, negative zeros, tiny, large -- the forward
    // test's rows, so both branches and both zero signs are covered.
    data[4 * dim..5 * dim].fill(0.0);
    data[5 * dim..6 * dim].fill(-0.0);
    data[6 * dim..7 * dim].fill(1e-20);
    data[7 * dim..8 * dim].iter_mut().for_each(|e| *e *= 1e30);
    let grad = rng.vec(8 * dim, 2.0);
    let cpu = backend(2, Numerics::Exact);
    let eps = 1e-12f32;
    let got = cpu
        .l2_normalize_backward(
            &f32t(&cpu, &data, &[8, dim]),
            &f32t(&cpu, &grad, &[8, dim]),
            eps,
        )
        .unwrap();
    assert_eq!(got.shape(), [8, dim]);
    assert_eq!(got.dtype(), DType::F32);
    let got = v(&got);
    assert_eq!(
        bits(&got),
        bits(&normalize_grad_ref(&data, &grad, 8, dim, eps))
    );
    // The zero rows (norm 0 <= eps) take grad / eps, not zero: the forward
    // divided them by the constant eps, so the gradient flows.
    for r in [4usize, 5] {
        for col in 0..dim {
            let want = (f64::from(grad[r * dim + col]) / f64::from(eps)) as f32;
            assert_eq!(got[r * dim + col].to_bits(), want.to_bits(), "row {r}");
        }
    }
    // A row above the clamp is orthogonal to itself after the projection:
    // <x, dx> is 0 to rounding, which the clamp branch would not give.
    for r in [0usize, 1, 2, 3] {
        let dot: f64 = (0..dim)
            .map(|c| f64::from(data[r * dim + c]) * f64::from(got[r * dim + c]))
            .sum();
        let scale: f64 = (0..dim)
            .map(|c| f64::from(data[r * dim + c]).abs() * f64::from(got[r * dim + c]).abs())
            .sum();
        assert!(
            dot.abs() <= 1e-6 * scale.max(1.0),
            "row {r}: <x, dx> = {dot}"
        );
    }
}

#[test]
fn l2_normalize_backward_is_bit_identical_across_threads_and_numerics() {
    let (rows, dim) = (1013usize, 77usize);
    let mut rng = SplitMix64(43);
    let data = rng.vec(rows * dim, 5.0);
    let grad = rng.vec(rows * dim, 2.0);
    let eps = 1e-12f32;
    let want = normalize_grad_ref(&data, &grad, rows, dim, eps);
    let mut first: Option<Vec<u32>> = None;
    for cpu in tiers() {
        let got = bits(&v(&cpu
            .l2_normalize_backward(
                &f32t(&cpu, &data, &[rows, dim]),
                &f32t(&cpu, &grad, &[rows, dim]),
                eps,
            )
            .unwrap()));
        assert_eq!(got, bits(&want));
        match &first {
            None => first = Some(got),
            Some(want) => assert_eq!(&got, want),
        }
    }
}

#[test]
fn l2_normalize_backward_refuses_what_it_cannot_differentiate_under_its_own_name() {
    let cpu = backend(1, Numerics::Exact);
    let x = f32t(&cpu, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let g = f32t(&cpu, &[0.5, 0.25, -0.5, 1.0], &[2, 2]);
    cpu.l2_normalize_backward(&x, &g, 1e-12).unwrap();
    // eps must be finite and above 0, as the forward requires.
    assert_nonfinite(cpu.l2_normalize_backward(&x, &g, f32::NAN));
    assert_range(cpu.l2_normalize_backward(&x, &g, 0.0));
    assert_range(cpu.l2_normalize_backward(&x, &g, -1e-12));
    // The operand is [rows, dim] and the gradient is its shape.
    assert_shape(cpu.l2_normalize_backward(
        &f32t(&cpu, &[1.0, 2.0], &[2]),
        &f32t(&cpu, &[1.0, 2.0], &[2]),
        1e-12,
    ));
    assert_shape(cpu.l2_normalize_backward(&x, &f32t(&cpu, &[1.0; 6], &[3, 2]), 1e-12));
    assert_shape(cpu.l2_normalize_backward(&x, &f32t(&cpu, &[1.0; 4], &[4, 1]), 1e-12));
    let ids = u32t(&cpu, &[1, 2, 3, 4], &[2, 2]);
    for (bad_x, bad_g) in [(&ids, &g), (&x, &ids)] {
        assert!(matches!(
            cpu.l2_normalize_backward(bad_x, bad_g, 1e-12),
            Err(OjasError::Dtype {
                op: "l2_normalize_backward",
                ..
            })
        ));
    }
    // A non-finite value in either operand is refused: both are read.
    for bad in [f32::NAN, f32::INFINITY] {
        let poisoned = f32t(&cpu, &[1.0, bad, 1.0, 1.0], &[2, 2]);
        match cpu.l2_normalize_backward(&poisoned, &g, 1e-12) {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, "l2_normalize_backward"),
            other => panic!("expected NonFinite, got {other:?}"),
        }
        assert_nonfinite(cpu.l2_normalize_backward(&x, &poisoned, 1e-12));
    }
}

#[test]
fn rms_qkv_norm_forward_matches_rms_norm_and_weightless_formula() {
    let cpu = backend(1, Numerics::Exact);
    let q = f32t(&cpu, &[2.0; 64], &[1, 1, 1, 64]);
    let k = f32t(&cpu, &[3.0; 64], &[1, 1, 1, 64]);
    let v = f32t(&cpu, &[4.0; 64], &[1, 1, 1, 64]);
    let qw = f32t(&cpu, &[1.5; 64], &[64]);
    let kw = f32t(&cpu, &[0.5; 64], &[64]);
    let eps = 1e-6f32;

    let (qn, kn, vn) = cpu
        .rms_qkv_norm_forward(&q, &k, &v, Some(&qw), Some(&kw), eps)
        .expect("rms_qkv_norm_forward");

    let want_q = cpu.rms_norm_forward(&q, &qw, eps).unwrap();
    let want_k = cpu.rms_norm_forward(&k, &kw, eps).unwrap();

    assert_eq!(qn.to_f32_vec().unwrap(), want_q.to_f32_vec().unwrap());
    assert_eq!(kn.to_f32_vec().unwrap(), want_k.to_f32_vec().unwrap());

    // V is normalized weightless: rms of [4.0; 64] is sqrt(16.0 + 1e-6) ≈ 4.0, out ≈ 1.0.
    let v_data = vn.to_f32_vec().unwrap();
    for &x in &v_data {
        assert!((x - 1.0).abs() < 1e-4);
    }

    // Weightless Q and K when weights are None
    let (qn_none, kn_none, _) = cpu
        .rms_qkv_norm_forward(&q, &k, &v, None, None, eps)
        .expect("weightless q and k");
    for &x in &qn_none.to_f32_vec().unwrap() {
        assert!((x - 1.0).abs() < 1e-4);
    }
    for &x in &kn_none.to_f32_vec().unwrap() {
        assert!((x - 1.0).abs() < 1e-4);
    }
}
