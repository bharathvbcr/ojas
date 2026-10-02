//! The heavy ops read their inputs in place and charge only what they
//! allocate: embedding forward and backward, cross-entropy forward and
//! backward, `clip_grad_norm` and `adamw_step` (plus Muon's config order).
//!
//! Each test pins one property of that rewrite against a scalar reference
//! written here: Exact bits (and, for the element-wise ops, Fast bits) equal
//! to the reference; Fast within a tolerance measured against `f64`; bits
//! that do not depend on the thread count (1, 2, 7, 18); the charge; and the
//! order in which a NaN, a malformed operand, a refused budget and a shared
//! target are reported now that the NaN scan is folded into the compute
//! passes. Shapes are large enough to cut into several parallel blocks.

mod common;

use common::{bits, SplitMix64};
use ojas_core::{AdamWConfig, Backend, Budget, MuonNs5Config, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const THREADS: [usize; 4] = [1, 2, 7, 18];

fn backend(threads: usize, numerics: Numerics, cap: u64) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(cap), threads)
        .unwrap()
        .with_numerics(numerics)
}

/// Inputs live on their own budget, so the backend's sees only the op.
fn f(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn u(data: &[u32], shape: &[usize]) -> Tensor {
    Tensor::from_u32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

fn is_nonfinite<T: std::fmt::Debug>(r: Result<T, OjasError>) -> bool {
    matches!(r, Err(OjasError::NonFinite { .. }))
}

// ---------------------------------------------------------------- embedding

/// Rows are copied bit for bit (negative zero and subnormals included), the
/// output is the only charge, and the result does not depend on threads.
#[test]
fn embedding_forward_copies_rows_bit_for_bit_and_charges_only_the_output() {
    let (vocab, dim, tokens) = (3000usize, 96usize, 777usize);
    let mut rng = SplitMix64(0xe1);
    let mut table = rng.vec(vocab * dim, 1.0);
    table[5] = -0.0;
    table[6] = f32::from_bits(1);
    table[7] = -f32::MAX;
    let ids: Vec<u32> = (0..tokens)
        .map(|i| {
            if i % 50 == 0 {
                0
            } else {
                rng.below(vocab) as u32
            }
        })
        .collect();
    let (tt, it) = (f(&table, &[vocab, dim]), u(&ids, &[7, 111]));
    let want: Vec<u32> = ids
        .iter()
        .flat_map(|&id| bits(&table[id as usize * dim..(id as usize + 1) * dim]))
        .collect();
    let out_bytes = (tokens * dim * 4) as u64;
    for threads in THREADS {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let be = backend(threads, numerics, out_bytes);
            let y = be.embedding_forward(&tt, &it).unwrap();
            assert_eq!(y.shape(), &[7, 111, dim]);
            assert!(bits(&v(&y)) == want, "threads {threads} {numerics:?}");
            assert_eq!(be.budget().live_bytes().unwrap(), out_bytes);
            drop(y);
            let tight = backend(threads, numerics, out_bytes - 1);
            assert!(matches!(
                tight.embedding_forward(&tt, &it),
                Err(OjasError::CapacityExceeded { .. })
            ));
            assert_eq!(tight.budget().live_bytes().unwrap(), 0);
        }
    }
}

/// A NaN in a row no id reads is still refused: Metal and wgpu scan the
/// whole table, so the CPU does too. It outranks an out-of-range id and a
/// budget with no room.
#[test]
fn embedding_refuses_a_nan_in_an_unread_row_whatever_the_budget() {
    let (vocab, dim) = (4000usize, 64usize);
    let mut table = vec![0.25f32; vocab * dim];
    table[(vocab - 1) * dim + 3] = f32::NAN;
    let tt = f(&table, &[vocab, dim]);
    let ids = u(&[0, 1, 2], &[3]);
    let grad = f(&[1.0; 3 * 64], &[3, dim]);
    for threads in THREADS {
        let be = backend(threads, Numerics::Fast, 0);
        assert!(
            is_nonfinite(be.embedding_forward(&tt, &ids)),
            "fwd {threads}"
        );
        assert!(
            is_nonfinite(be.embedding_backward(&tt, &ids, &grad)),
            "bwd {threads}"
        );
        let bad_ids = u(&[0, vocab as u32, 2], &[3]);
        assert!(is_nonfinite(be.embedding_forward(&tt, &bad_ids)));
    }
}

/// The table gradient is the f32 sum from 0 of each id's rows in token
/// order, the output is its only charge, and an overflowing sum is refused.
#[test]
fn embedding_backward_sums_in_token_order_and_charges_only_the_table() {
    let (vocab, dim, tokens) = (500usize, 48usize, 3000usize);
    let mut rng = SplitMix64(0xe2);
    let table = f(&rng.vec(vocab * dim, 1.0), &[vocab, dim]);
    let ids: Vec<u32> = (0..tokens).map(|_| rng.below(vocab / 10) as u32).collect();
    let grad = rng.vec(tokens * dim, 1.0);
    let mut want = vec![0.0f32; vocab * dim];
    for (n, &id) in ids.iter().enumerate() {
        for c in 0..dim {
            want[id as usize * dim + c] += grad[n * dim + c];
        }
    }
    let (it, gt) = (u(&ids, &[tokens]), f(&grad, &[tokens, dim]));
    let out_bytes = (vocab * dim * 4) as u64;
    for threads in THREADS {
        let be = backend(threads, Numerics::Fast, out_bytes);
        let g = be.embedding_backward(&table, &it, &gt).unwrap();
        assert!(bits(&v(&g)) == bits(&want), "threads {threads}");
        assert_eq!(be.budget().live_bytes().unwrap(), out_bytes);
    }
    let big = f(&[f32::MAX; 2 * 48], &[2, dim]);
    let same = u(&[1, 1], &[2]);
    let be = backend(7, Numerics::Fast, u64::MAX);
    assert!(is_nonfinite(be.embedding_backward(&table, &same, &big)));
    assert_eq!(be.budget().live_bytes().unwrap(), 0);
}

// ------------------------------------------------------------ cross-entropy

struct Ce {
    logits: Vec<f32>,
    targets: Vec<u32>,
    rows: usize,
    vocab: usize,
    ignore: Option<u32>,
}

fn ce_case(rows: usize, vocab: usize, scale: f32, seed: u64) -> Ce {
    let mut rng = SplitMix64(seed);
    let logits = rng.vec(rows * vocab, scale);
    let ignore = Some(vocab as u32 + 3);
    let targets = (0..rows)
        .map(|i| {
            if i % 9 == 4 {
                vocab as u32 + 3
            } else {
                rng.below(vocab) as u32
            }
        })
        .collect();
    Ce {
        logits,
        targets,
        rows,
        vocab,
        ignore,
    }
}

/// The reference contract, as `ojas-cpu` computed it before this lane: per
/// valid row, the ascending maximum, libm `exp` of `x - max` summed in
/// ascending order, `term = max + ln(sum) - x[target]`, the gradient
/// `(e / sum) / valid` with `1 / valid` taken off at the target; the loss is
/// the f32 sum of the terms in row order divided by the valid count.
fn ce_exact_reference(c: &Ce) -> (f32, Vec<f32>) {
    let valid = c.targets.iter().filter(|&&t| Some(t) != c.ignore).count() as f32;
    let mut grad = vec![0.0f32; c.rows * c.vocab];
    let mut total = 0.0f32;
    for n in 0..c.rows {
        if Some(c.targets[n]) == c.ignore {
            continue;
        }
        let row = &c.logits[n * c.vocab..(n + 1) * c.vocab];
        let mut max = f32::NEG_INFINITY;
        for &x in row {
            if x > max {
                max = x;
            }
        }
        let mut sum = 0.0f32;
        let mut e = vec![0.0f32; c.vocab];
        for (col, &x) in row.iter().enumerate() {
            e[col] = (x - max).exp();
            sum += e[col];
        }
        let class = c.targets[n] as usize;
        total += max + sum.ln() - row[class];
        for col in 0..c.vocab {
            grad[n * c.vocab + col] = (e[col] / sum) / valid;
        }
        grad[n * c.vocab + class] -= 1.0 / valid;
    }
    (total / valid, grad)
}

/// The same quantities in `f64`.
fn ce_f64(c: &Ce) -> (f64, Vec<f64>) {
    let valid = c.targets.iter().filter(|&&t| Some(t) != c.ignore).count() as f64;
    let mut grad = vec![0.0f64; c.rows * c.vocab];
    let mut total = 0.0f64;
    for n in 0..c.rows {
        if Some(c.targets[n]) == c.ignore {
            continue;
        }
        let row: Vec<f64> = c.logits[n * c.vocab..(n + 1) * c.vocab]
            .iter()
            .map(|&x| f64::from(x))
            .collect();
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row.iter().map(|x| (x - max).exp()).sum();
        let class = c.targets[n] as usize;
        total += max + sum.ln() - row[class];
        for col in 0..c.vocab {
            grad[n * c.vocab + col] = (row[col] - max).exp() / sum / valid;
        }
        grad[n * c.vocab + class] -= 1.0 / valid;
    }
    (total / valid, grad)
}

fn ce_run(be: &CpuBackend, c: &Ce) -> (f32, Vec<f32>) {
    let (lt, tt) = (f(&c.logits, &[c.rows, c.vocab]), u(&c.targets, &[c.rows]));
    let loss = v(&be.cross_entropy_mean_forward(&lt, &tt, c.ignore).unwrap())[0];
    let grad = v(&be.cross_entropy_mean_backward(&lt, &tt, c.ignore).unwrap());
    (loss, grad)
}

/// Exact keeps the reference bits; both contracts' bits do not depend on the
/// thread count. 600 rows of 1000 cut into three 262-row blocks.
#[test]
fn cross_entropy_exact_keeps_the_reference_bits_at_every_thread_count() {
    let c = ce_case(600, 1000, 4.0, 0xce);
    let (want_loss, want_grad) = ce_exact_reference(&c);
    let fast_one = ce_run(&backend(1, Numerics::Fast, u64::MAX), &c);
    for threads in THREADS {
        let (loss, grad) = ce_run(&backend(threads, Numerics::Exact, u64::MAX), &c);
        assert_eq!(
            loss.to_bits(),
            want_loss.to_bits(),
            "Exact loss, threads {threads}"
        );
        assert!(
            bits(&grad) == bits(&want_grad),
            "Exact grad, threads {threads}"
        );
        let (loss, grad) = ce_run(&backend(threads, Numerics::Fast, u64::MAX), &c);
        assert_eq!(
            loss.to_bits(),
            fast_one.0.to_bits(),
            "Fast loss, threads {threads}"
        );
        assert!(
            bits(&grad) == bits(&fast_one.1),
            "Fast grad, threads {threads}"
        );
    }
}

/// Fast's vector `exp` (1 ulp measured) and lane sums move the loss and
/// gradient by rounding only: against `f64`, Fast and the Exact reference
/// are both within 1e-6 relative (the loss) and 1e-6 of the largest gradient
/// magnitude, at the nanolab vocabulary and with large logits. Measured on
/// aarch64: loss 1.3e-7 (Fast) and 4.2e-8 (Exact) at 48 x 50304, 2.1e-7 for
/// both at 300 x 1000 with logits to 30; gradient at most 5.2e-8 for both.
#[test]
fn cross_entropy_fast_and_exact_are_within_1e_6_of_f64() {
    for (rows, vocab, scale) in [(48usize, 50304usize, 2.0f32), (300, 1000, 30.0)] {
        let c = ce_case(rows, vocab, scale, 0xcf + rows as u64);
        let (loss64, grad64) = ce_f64(&c);
        let gmax = grad64.iter().fold(0.0f64, |m, g| m.max(g.abs()));
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let (loss, grad) = ce_run(&backend(7, numerics, u64::MAX), &c);
            let loss_err = (f64::from(loss) - loss64).abs() / loss64.abs();
            let grad_err = grad
                .iter()
                .zip(&grad64)
                .fold(0.0f64, |m, (&a, &b)| m.max((f64::from(a) - b).abs()))
                / gmax;
            println!("ce {rows}x{vocab} scale {scale} {numerics:?}: loss rel {loss_err:e}, grad rel {grad_err:e}");
            assert!(loss_err <= 1e-6, "{numerics:?} loss rel error {loss_err:e}");
            assert!(grad_err <= 1e-6, "{numerics:?} grad rel error {grad_err:e}");
        }
    }
}

/// The NaN scan is the loss pass. A NaN in an ignored row is still refused,
/// and a NaN outranks an out-of-range target, an all-ignored batch, a
/// mis-laid-out target tensor and a budget too small for the gradient, as
/// scanning the logits first would order them.
#[test]
fn cross_entropy_nan_outranks_every_later_refusal() {
    let (rows, vocab) = (300usize, 1000usize);
    let c = ce_case(rows, vocab, 1.0, 0xd0);
    let mut logits = c.logits.clone();
    let ignored_row = (0..rows).find(|&n| Some(c.targets[n]) == c.ignore).unwrap();
    logits[ignored_row * vocab + 17] = f32::NAN;
    let lt = f(&logits, &[rows, vocab]);
    let good = f(&c.logits, &[rows, vocab]);
    let tt = u(&c.targets, &[rows]);
    let mut bad = c.targets.clone();
    bad[0] = vocab as u32;
    let bad = u(&bad, &[rows]);
    let all_ignored = u(&vec![vocab as u32 + 3; rows], &[rows]);
    let strided = u(&vec![0u32; 2 * rows], &[2 * rows])
        .view(&[rows], &[2], 0)
        .unwrap();
    for threads in THREADS {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let be = backend(threads, numerics, 0);
            for t in [&tt, &bad, &all_ignored, &strided] {
                assert!(is_nonfinite(
                    be.cross_entropy_mean_forward(&lt, t, c.ignore)
                ));
                assert!(is_nonfinite(
                    be.cross_entropy_mean_backward(&lt, t, c.ignore)
                ));
            }
            assert_eq!(be.budget().live_bytes().unwrap(), 0);
            // Without the NaN each refusal is its own.
            assert!(matches!(
                be.cross_entropy_mean_forward(&good, &bad, c.ignore),
                Err(OjasError::OutOfRange { .. })
            ));
            assert!(matches!(
                be.cross_entropy_mean_backward(&good, &tt, c.ignore),
                Err(OjasError::CapacityExceeded { .. })
            ));
            assert!(matches!(
                be.cross_entropy_mean_forward(&good, &strided, c.ignore),
                Err(OjasError::Shape { .. })
            ));
        }
    }
}

/// The forward forms no gradient: its only charge is the 4-byte loss. The
/// backward's only charge is the gradient it returns.
#[test]
fn cross_entropy_charges_only_what_it_returns() {
    let c = ce_case(300, 1000, 1.0, 0xd1);
    let (lt, tt) = (f(&c.logits, &[300, 1000]), u(&c.targets, &[300]));
    let grad_bytes = 300 * 1000 * 4;
    for threads in [1usize, 7] {
        let be = backend(threads, Numerics::Fast, 4);
        let loss = be.cross_entropy_mean_forward(&lt, &tt, c.ignore).unwrap();
        assert_eq!(be.budget().live_bytes().unwrap(), 4);
        drop(loss);
        let be = backend(threads, Numerics::Fast, grad_bytes);
        let g = be.cross_entropy_mean_backward(&lt, &tt, c.ignore).unwrap();
        assert_eq!(be.budget().live_bytes().unwrap(), grad_bytes);
        drop(g);
        let tight = backend(threads, Numerics::Fast, grad_bytes - 1);
        assert!(matches!(
            tight.cross_entropy_mean_backward(&lt, &tt, c.ignore),
            Err(OjasError::CapacityExceeded { .. })
        ));
    }
}

// --------------------------------------------------------------------- clip

/// 1.2M values in three gradients: two blocks of the Fast norm's fixed
/// partition plus a partial one, and a tiny gradient.
fn clip_grads(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SplitMix64(seed);
    vec![
        rng.vec(1_100_000, 0.01),
        rng.vec(7, 0.5),
        rng.vec(150_001, 0.02),
    ]
}

fn tensors(grads: &[Vec<f32>]) -> Vec<Tensor> {
    grads.iter().map(|g| f(g, &[g.len()])).collect()
}

/// Exact is the ascending `f64` sum of squares, bit for bit. Fast sums fixed
/// blocks, so its bits do not depend on the thread count, and it is within
/// 1e-7 relative of `f64` (torch's own f32 norm is further off at nanolab
/// size; neither contract matches it). The scale is applied as `g * scale`.
#[test]
fn clip_norm_is_the_f64_sum_and_the_scale_is_one_multiply() {
    let grads = clip_grads(0xc1);
    let mut sum = 0.0f64;
    for g in &grads {
        for &x in g {
            sum += f64::from(x) * f64::from(x);
        }
    }
    let exact_norm = sum.sqrt() as f32;
    let max_norm = 1.0f32;
    let mut fast_one = None;
    for threads in THREADS {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let be = backend(threads, numerics, u64::MAX);
            let mut ts = tensors(&grads);
            let norm = be.clip_grad_norm(&mut ts, max_norm).unwrap();
            match numerics {
                Numerics::Exact => assert_eq!(norm.to_bits(), exact_norm.to_bits()),
                Numerics::Fast => {
                    let rel = (f64::from(norm) - sum.sqrt()).abs() / sum.sqrt();
                    assert!(rel <= 1e-7, "Fast norm rel error {rel:e}");
                    let first = *fast_one.get_or_insert(norm.to_bits());
                    assert_eq!(norm.to_bits(), first, "Fast norm at {threads} threads");
                }
            }
            let scale = (max_norm / (norm + 1e-6)).min(1.0);
            assert!(scale < 1.0);
            for (t, g) in ts.iter().zip(&grads) {
                let want: Vec<f32> = g.iter().map(|&x| x * scale).collect();
                assert!(bits(&v(t)) == bits(&want), "{numerics:?} threads {threads}");
            }
        }
    }
}

/// Scaling charges one buffer as long as the largest gradient; a norm under
/// `max_norm` charges nothing and leaves every bit.
#[test]
fn clip_charges_one_largest_gradient_and_nothing_when_it_does_not_scale() {
    let grads = clip_grads(0xc2);
    let largest = 1_100_000u64 * 4;
    let be = backend(7, Numerics::Fast, 0);
    let mut ts = tensors(&grads);
    be.clip_grad_norm(&mut ts, 1e6).unwrap();
    for (t, g) in ts.iter().zip(&grads) {
        assert!(bits(&v(t)) == bits(g));
    }
    let tight = backend(7, Numerics::Fast, largest - 1);
    let mut ts = tensors(&grads);
    assert!(matches!(
        tight.clip_grad_norm(&mut ts, 1.0),
        Err(OjasError::CapacityExceeded { .. })
    ));
    for (t, g) in ts.iter().zip(&grads) {
        assert!(bits(&v(t)) == bits(g), "a refusal wrote a gradient");
    }
    let fits = backend(7, Numerics::Fast, largest);
    let mut ts = tensors(&grads);
    fits.clip_grad_norm(&mut ts, 1.0).unwrap();
    assert_eq!(fits.budget().live_bytes().unwrap(), 0);
}

/// The norm pass is the NaN scan: a NaN anywhere is refused before any
/// write or charge, and outranks a later gradient's layout refusal.
#[test]
fn clip_nan_is_refused_before_a_later_layout_error_and_any_charge() {
    let mut grads = clip_grads(0xc3);
    grads[2][150_000] = f32::INFINITY;
    let be = backend(7, Numerics::Fast, 0);
    let mut ts = tensors(&grads);
    assert!(is_nonfinite(be.clip_grad_norm(&mut ts, 1.0)));
    let base = f(&[1.0; 8], &[8]);
    let strided = base.view(&[4], &[2], 0).unwrap();
    let mut with_layout_error = vec![ts[2].clone(), strided];
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let be = backend(7, numerics, 0);
        assert!(is_nonfinite(be.clip_grad_norm(&mut with_layout_error, 1.0)));
    }
}

// -------------------------------------------------------------------- AdamW

/// The reference element arithmetic, as `ojas-cpu` computed it before this
/// lane (f64 moments, torch `lerp`, `eps` outside the square root).
fn adam_reference(
    p: &[f32],
    g: &[f32],
    m: &[f32],
    v2: &[f32],
    step_before: u64,
    c: AdamWConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let step = step_before + 1;
    let bc1 = 1.0 - ojas_core::pow_u64(c.beta1, step);
    let bc2 = 1.0 - ojas_core::pow_u64(c.beta2, step);
    let (mut ps, mut ms, mut vs) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..p.len() {
        let gi = f64::from(g[i]);
        let mut pi = f64::from(p[i]);
        let mut mi = f64::from(m[i]);
        let mut vi = f64::from(v2[i]);
        if c.weight_decay != 0.0 {
            pi *= 1.0 - c.lr * c.weight_decay;
        }
        if 1.0 - c.beta1 < 0.5 {
            mi += (1.0 - c.beta1) * (gi - mi);
        } else {
            mi = gi - (gi - mi) * c.beta1;
        }
        vi = c.beta2 * vi + (1.0 - c.beta2) * gi * gi;
        let denom = vi.sqrt() / bc2.sqrt() + c.eps;
        let delta = (-(c.lr / bc1)) * mi / denom;
        if c.weight_decay == 0.0 && delta == 0.0 {
            ps.push(p[i]);
        } else {
            ps.push((pi + delta) as f32);
        }
        ms.push(mi as f32);
        vs.push(vi as f32);
    }
    (ps, ms, vs)
}

/// The Fast element arithmetic: the scalars formed in f64 as
/// `adam_reference` forms them, rounded to f32 once, then every element
/// operation in f32 (torch single-tensor AdamW on f32 tensors).
fn adam_reference_fast(
    p: &[f32],
    g: &[f32],
    m: &[f32],
    v2: &[f32],
    step_before: u64,
    c: AdamWConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let step = step_before + 1;
    let bc1 = 1.0 - ojas_core::pow_u64(c.beta1, step);
    let bc2 = 1.0 - ojas_core::pow_u64(c.beta2, step);
    let beta1 = c.beta1 as f32;
    let beta2 = c.beta2 as f32;
    let one_minus_b1 = (1.0 - c.beta1) as f32;
    let one_minus_b2 = (1.0 - c.beta2) as f32;
    let step_size = (c.lr / bc1) as f32;
    let bc2_sqrt = bc2.sqrt() as f32;
    let eps = c.eps as f32;
    let decay = (1.0 - c.lr * c.weight_decay) as f32;
    let (mut ps, mut ms, mut vs) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..p.len() {
        let gi = g[i];
        let mut pi = p[i];
        let mi = if 1.0 - c.beta1 < 0.5 {
            m[i] + one_minus_b1 * (gi - m[i])
        } else {
            gi - (gi - m[i]) * beta1
        };
        let vi = beta2 * v2[i] + one_minus_b2 * gi * gi;
        let denom = vi.sqrt() / bc2_sqrt + eps;
        let delta = (-step_size) * mi / denom;
        if c.weight_decay != 0.0 {
            pi *= decay;
        }
        if c.weight_decay == 0.0 && delta == 0.0 {
            ps.push(p[i]);
        } else {
            ps.push(pi + delta);
        }
        ms.push(mi);
        vs.push(vi);
    }
    (ps, ms, vs)
}

fn adam_inputs(n: usize, seed: u64) -> [Vec<f32>; 4] {
    let mut rng = SplitMix64(seed);
    let mut p = rng.vec(n, 0.05);
    let mut g = rng.vec(n, 0.01);
    let m = rng.vec(n, 0.001);
    let v2: Vec<f32> = rng.vec(n, 1e-5).iter().map(|x| x.abs()).collect();
    // A zero step keeps a negative zero; a zero gradient with zero moments
    // gives one.
    p[3] = -0.0;
    g[3] = 0.0;
    [p, g, m, v2]
}

/// Both contracts give their reference bits at every thread count (Exact the
/// f64 element arithmetic, Fast the f32 one), with and without weight decay
/// and on both sides of torch's `lerp` switch. 300k values cut into three
/// 131072-value blocks. Fast is also within a few f32 ulps of Exact.
#[test]
fn adamw_keeps_the_reference_bits_in_place_at_every_thread_count() {
    let n = 300_001usize;
    let [p, g, m, v2] = adam_inputs(n, 0xa1);
    let mut m_zero = m.clone();
    m_zero[3] = 0.0;
    let mut v_zero = v2.clone();
    v_zero[3] = 0.0;
    for config in [
        AdamWConfig::nanolab(6e-4, 0.0),
        AdamWConfig::nanolab(1e-3, 0.1),
        AdamWConfig {
            beta1: 0.3,
            ..AdamWConfig::nanolab(1e-3, 0.0)
        },
    ] {
        let exact = adam_reference(&p, &g, &m_zero, &v_zero, 4, config);
        let fast = adam_reference_fast(&p, &g, &m_zero, &v_zero, 4, config);
        for (name, e, f) in [
            ("param", &exact.0, &fast.0),
            ("moment1", &exact.1, &fast.1),
            ("moment2", &exact.2, &fast.2),
        ] {
            let scale = e.iter().fold(0.0f32, |s, x| s.max(x.abs()));
            let err = e
                .iter()
                .zip(f)
                .fold(0.0f32, |s, (a, b)| s.max((a - b).abs()));
            assert!(
                err <= 1e-6 * scale,
                "{name} {config:?}: fast {err:e} > 1e-6 * {scale:e}"
            );
        }
        for threads in THREADS {
            for numerics in [Numerics::Exact, Numerics::Fast] {
                let want = if numerics == Numerics::Exact {
                    &exact
                } else {
                    &fast
                };
                let be = backend(threads, numerics, (n * 4) as u64);
                let (mut pt, gt, mut mt, mut vt) =
                    (f(&p, &[n]), f(&g, &[n]), f(&m_zero, &[n]), f(&v_zero, &[n]));
                be.adamw_step(&mut pt, &gt, &mut mt, &mut vt, 4, config)
                    .unwrap();
                let what = format!("{config:?} threads {threads} {numerics:?}");
                assert!(bits(&v(&pt)) == bits(&want.0), "param {what}");
                assert!(bits(&v(&mt)) == bits(&want.1), "moment1 {what}");
                assert!(bits(&v(&vt)) == bits(&want.2), "moment2 {what}");
                assert_eq!(be.budget().live_bytes().unwrap(), 0);
            }
        }
    }
}

/// One `len`-value buffer is the whole charge; one f32 less refuses and
/// writes nothing.
#[test]
fn adamw_charges_one_buffer() {
    let n = 200_000usize;
    let [p, g, m, v2] = adam_inputs(n, 0xa2);
    let config = AdamWConfig::nanolab(1e-3, 0.1);
    let be = backend(7, Numerics::Fast, (n * 4 - 4) as u64);
    let (mut pt, gt, mut mt, mut vt) = (f(&p, &[n]), f(&g, &[n]), f(&m, &[n]), f(&v2, &[n]));
    assert!(matches!(
        be.adamw_step(&mut pt, &gt, &mut mt, &mut vt, 0, config),
        Err(OjasError::CapacityExceeded { .. })
    ));
    assert!(bits(&v(&pt)) == bits(&p) && bits(&v(&mt)) == bits(&m) && bits(&v(&vt)) == bits(&v2));
}

/// The update pass is the NaN scan. A NaN in any input outranks a bad
/// config, a budget with no room and a shared target, and a step whose
/// result overflows writes nothing.
#[test]
fn adamw_nan_outranks_config_budget_and_sharing_and_a_refusal_writes_nothing() {
    let n = 300_001usize;
    let [p, g, m, v2] = adam_inputs(n, 0xa3);
    let mut v_nan = v2.clone();
    v_nan[n - 1] = f32::NAN;
    let good = AdamWConfig::nanolab(1e-3, 0.1);
    let bad = AdamWConfig { lr: -1.0, ..good };
    for threads in [1usize, 7] {
        let (mut pt, gt, mut mt, mut vt) = (f(&p, &[n]), f(&g, &[n]), f(&m, &[n]), f(&v_nan, &[n]));
        let roomy = backend(threads, Numerics::Fast, u64::MAX);
        assert!(is_nonfinite(
            roomy.adamw_step(&mut pt, &gt, &mut mt, &mut vt, 0, bad)
        ));
        let empty = backend(threads, Numerics::Fast, 0);
        assert!(is_nonfinite(
            empty.adamw_step(&mut pt, &gt, &mut mt, &mut vt, 0, good)
        ));
        let shared = mt.clone();
        assert!(is_nonfinite(
            roomy.adamw_step(&mut pt, &gt, &mut mt, &mut vt, 0, good)
        ));
        drop(shared);
        assert!(is_nonfinite(
            roomy.adamw_step(&mut pt, &gt, &mut mt, &mut vt, 0, good)
        ));
        assert!(
            bits(&v(&pt)) == bits(&p) && bits(&v(&mt)) == bits(&m),
            "a refusal wrote"
        );
        // Without the NaN the config is its own refusal.
        let mut vt = f(&v2, &[n]);
        assert!(matches!(
            empty.adamw_step(&mut pt, &gt, &mut mt, &mut vt, 0, bad),
            Err(OjasError::OutOfRange { .. })
        ));
    }
    // A finite f64 step whose parameter overflows f32 writes nothing.
    let huge = p.clone();
    let (mut pt, gt, mut mt, mut vt) = (f(&huge, &[n]), f(&g, &[n]), f(&m, &[n]), f(&v2, &[n]));
    let big_lr = AdamWConfig::nanolab(1e300, 0.0);
    let be = backend(7, Numerics::Fast, u64::MAX);
    assert!(is_nonfinite(
        be.adamw_step(&mut pt, &gt, &mut mt, &mut vt, 0, big_lr)
    ));
    assert!(
        bits(&v(&pt)) == bits(&huge) && bits(&v(&mt)) == bits(&m) && bits(&v(&vt)) == bits(&v2)
    );
}

// --------------------------------------------------------------------- Muon

/// The Muon config is checked before any charge (shape contract D14).
#[test]
fn muon_config_is_refused_before_the_budget() {
    let mut rng = SplitMix64(0x30);
    let (rows, cols) = (64usize, 48usize);
    let (mut p, g, mut m) = (
        f(&rng.vec(rows * cols, 0.1), &[rows, cols]),
        f(&rng.vec(rows * cols, 0.1), &[rows, cols]),
        f(&rng.vec(rows * cols, 0.1), &[rows, cols]),
    );
    let bad = MuonNs5Config {
        lr: -0.1,
        ..MuonNs5Config::nanolab_default()
    };
    let be = backend(7, Numerics::Fast, 0);
    assert!(matches!(
        be.muon_ns5_step(&mut p, &g, &mut m, bad),
        Err(OjasError::OutOfRange { .. })
    ));
    let nan = MuonNs5Config {
        momentum: f64::NAN,
        ..MuonNs5Config::nanolab_default()
    };
    assert!(is_nonfinite(be.muon_ns5_step(&mut p, &g, &mut m, nan)));
    assert!(matches!(
        be.muon_ns5_step(&mut p, &g, &mut m, MuonNs5Config::nanolab_default()),
        Err(OjasError::CapacityExceeded { .. })
    ));
}
