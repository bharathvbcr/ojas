//! `linear_cross_entropy_mean` (T3) against the composed path
//! `linear_forward → cross_entropy_mean_forward/backward → linear_backward`.
//!
//! Under `Numerics::Exact` the loss and both gradients are bitwise equal to
//! the composed path for every chunk, including chunks that divide neither N
//! nor V: the gradient products accumulate into one buffer through
//! `gemm_acc`, continuing the composed GEMM's ascending sums. Under Fast the
//! two paths agree to `max|a - b| <= FAST_REL * max|b|` over each tensor (an
//! elementwise relative bound would fail spuriously where softmax - onehot
//! cancels to near zero).

mod common;

use common::{assert_capacity, assert_nonfinite, assert_range, assert_shape, bits, SplitMix64};
use ojas_core::{Backend, Budget, CeChunk, DType, LinearCe, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

/// Fast mode: the composed GEMM over all of V can be one Accelerate call
/// with its own order, so neither side is a fixed sequence.
const FAST_REL: f32 = 1e-5;

struct Case {
    x: Tensor,
    w: Tensor,
    t: Tensor,
    ignore: Option<u32>,
}

fn case(rng: &mut SplitMix64, n: usize, d: usize, v: usize, ignore: Option<u32>) -> Case {
    let ib = Budget::new(u64::MAX);
    let x = Tensor::from_f32(&rng.vec(n * d, 1.0), &[n, d], &ib).unwrap();
    let w = Tensor::from_f32(&rng.vec(v * d, 0.5), &[v, d], &ib).unwrap();
    let mut targets: Vec<u32> = (0..n).map(|_| rng.below(v) as u32).collect();
    if let Some(ig) = ignore {
        for (i, t) in targets.iter_mut().enumerate() {
            if i % 4 == 1 {
                *t = ig;
            }
        }
    }
    let t = Tensor::from_u32(&targets, &[n], &ib).unwrap();
    Case { x, w, t, ignore }
}

type Grads = (Tensor, Tensor);

fn composed(be: &CpuBackend, c: &Case) -> (Tensor, Grads) {
    let logits = be.linear_forward(&c.x, &c.w).unwrap();
    let loss = be
        .cross_entropy_mean_forward(&logits, &c.t, c.ignore)
        .unwrap();
    let g = be
        .cross_entropy_mean_backward(&logits, &c.t, c.ignore)
        .unwrap();
    let grads = be.linear_backward(&c.x, &c.w, &g).unwrap();
    (loss, grads)
}

fn f(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

/// `max|got - want| / max|want|`, asserted `<= rel`.
fn rel_err(what: &str, got: &[f32], want: &[f32], rel: f32) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}");
    let scale = want
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()))
        .max(f32::MIN_POSITIVE);
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        err <= rel * scale,
        "{what}: max abs error {err} > {rel} * {scale}"
    );
    err / scale
}

fn shapes() -> Vec<(usize, usize, usize)> {
    vec![
        (1, 1, 1),
        (1, 3, 7),
        (7, 5, 13),
        (33, 16, 50),
        (64, 24, 97),
        (130, 32, 300),
    ]
}

fn chunks(n: usize, v: usize) -> Vec<CeChunk> {
    let mut out = vec![
        CeChunk { rows: 1, cols: 1 },
        CeChunk { rows: 2, cols: 3 },
        CeChunk { rows: 5, cols: 7 },
        CeChunk { rows: n, cols: v },
        CeChunk {
            rows: n + 3,
            cols: v + 5,
        },
        CeChunk { rows: n, cols: 7 },
        CeChunk { rows: 3, cols: v },
        CeChunk {
            rows: 64,
            cols: 128,
        },
    ];
    out.dedup();
    out
}

fn ignores(v: usize) -> [Option<u32>; 3] {
    [None, Some(0), Some(v as u32 + 7)]
}

#[test]
fn exact_loss_and_grads_are_bitwise_for_every_chunk() {
    let mut rng = SplitMix64(0xce01);
    for threads in [1usize, 4] {
        let be = CpuBackend::with_threads(Budget::new(u64::MAX), threads)
            .unwrap()
            .with_numerics(Numerics::Exact);
        for (n, d, v) in shapes() {
            for ignore in ignores(v) {
                let c = case(&mut rng, n, d, v, ignore);
                // An all-ignored batch is covered by its own test.
                let ids = c.t.to_u32_vec().unwrap();
                if ignore.is_some_and(|ig| ids.iter().all(|&t| t == ig)) {
                    continue;
                }
                let (want_loss, (want_gx, want_gw)) = composed(&be, &c);
                for chunk in chunks(n, v) {
                    let what = format!("t{threads} n{n} d{d} v{v} {ignore:?} {chunk:?}");
                    let got = be
                        .linear_cross_entropy_mean(&c.x, &c.w, &c.t, ignore, chunk, true)
                        .unwrap();
                    assert_exact(&what, &got, &want_loss, &want_gx, &want_gw);
                }
            }
        }
    }
}

fn assert_exact(what: &str, got: &LinearCe, loss: &Tensor, gx: &Tensor, gw: &Tensor) {
    assert_eq!(got.loss.shape(), loss.shape(), "{what}");
    assert_eq!(bits(&f(&got.loss)), bits(&f(loss)), "{what} loss");
    assert_eq!(
        bits(&f(got.grad_input.as_ref().unwrap())),
        bits(&f(gx)),
        "{what} grad_input"
    );
    assert_eq!(
        bits(&f(got.grad_weight.as_ref().unwrap())),
        bits(&f(gw)),
        "{what} grad_weight"
    );
}

/// Products large enough that the GEMM splits each chunk product across the
/// pool, so the accumulating path's per-tile snapshot is exercised.
#[test]
fn exact_split_products_stay_bitwise() {
    let mut rng = SplitMix64(0xce07);
    let (n, d, v) = (300usize, 256usize, 2000usize);
    let c = case(&mut rng, n, d, v, Some(v as u32 + 1));
    for threads in [1usize, 6] {
        let be = CpuBackend::with_threads(Budget::new(u64::MAX), threads)
            .unwrap()
            .with_numerics(Numerics::Exact);
        let (want_loss, (want_gx, want_gw)) = composed(&be, &c);
        for chunk in [
            CeChunk {
                rows: 256,
                cols: 1024,
            },
            CeChunk {
                rows: 97,
                cols: 701,
            },
            CeChunk { rows: n, cols: v },
        ] {
            let got = be
                .linear_cross_entropy_mean(&c.x, &c.w, &c.t, c.ignore, chunk, true)
                .unwrap();
            let what = format!("split t{threads} {chunk:?}");
            assert_exact(&what, &got, &want_loss, &want_gx, &want_gw);
        }
    }
}

#[test]
fn fast_matches_the_composed_path_within_tolerance() {
    let mut rng = SplitMix64(0xce02);
    let mut worst = 0.0f32;
    for threads in [1usize, 4] {
        let be = CpuBackend::with_threads(Budget::new(u64::MAX), threads).unwrap();
        assert_eq!(be.numerics(), Numerics::Fast);
        for (n, d, v) in [(33usize, 16usize, 50usize), (130, 64, 300), (257, 96, 1031)] {
            for ignore in ignores(v) {
                let c = case(&mut rng, n, d, v, ignore);
                let (want_loss, (want_gx, want_gw)) = composed(&be, &c);
                for chunk in chunks(n, v) {
                    let what = format!("fast t{threads} n{n} d{d} v{v} {ignore:?} {chunk:?}");
                    let got = be
                        .linear_cross_entropy_mean(&c.x, &c.w, &c.t, ignore, chunk, true)
                        .unwrap();
                    worst = worst
                        .max(rel_err(
                            &format!("{what} loss"),
                            &f(&got.loss),
                            &f(&want_loss),
                            FAST_REL,
                        ))
                        .max(rel_err(
                            &format!("{what} grad_input"),
                            &f(got.grad_input.as_ref().unwrap()),
                            &f(&want_gx),
                            FAST_REL,
                        ))
                        .max(rel_err(
                            &format!("{what} grad_weight"),
                            &f(got.grad_weight.as_ref().unwrap()),
                            &f(&want_gw),
                            FAST_REL,
                        ));
                }
            }
        }
    }
    println!("Fast worst relative error vs composed: {worst:e}");
}

#[test]
fn loss_only_call_returns_no_gradients_and_the_same_loss() {
    let mut rng = SplitMix64(0xce03);
    let be = CpuBackend::new(Budget::new(u64::MAX)).with_numerics(Numerics::Exact);
    let c = case(&mut rng, 21, 8, 40, Some(40));
    let (want, _) = composed(&be, &c);
    let got = be
        .linear_cross_entropy_mean(
            &c.x,
            &c.w,
            &c.t,
            c.ignore,
            CeChunk { rows: 4, cols: 9 },
            false,
        )
        .unwrap();
    assert!(got.grad_input.is_none() && got.grad_weight.is_none());
    assert_eq!(bits(&f(&got.loss)), bits(&f(&want)));
}

#[test]
fn one_valid_row_uses_the_global_count_and_all_ignored_is_nonfinite() {
    let be = CpuBackend::new(Budget::new(u64::MAX)).with_numerics(Numerics::Exact);
    let mut rng = SplitMix64(0xce04);
    let (n, d, v) = (9usize, 6usize, 11usize);
    let ib = Budget::new(u64::MAX);
    let x = Tensor::from_f32(&rng.vec(n * d, 1.0), &[n, d], &ib).unwrap();
    let w = Tensor::from_f32(&rng.vec(v * d, 1.0), &[v, d], &ib).unwrap();
    let ignore = Some(99u32);
    let mut ids = vec![99u32; n];
    ids[6] = 4;
    let t = Tensor::from_u32(&ids, &[n], &ib).unwrap();
    let c = Case { x, w, t, ignore };
    let (want_loss, (want_gx, want_gw)) = composed(&be, &c);
    for chunk in [CeChunk { rows: 2, cols: 3 }, CeChunk { rows: n, cols: v }] {
        let got = be
            .linear_cross_entropy_mean(&c.x, &c.w, &c.t, ignore, chunk, true)
            .unwrap();
        assert_exact(
            &format!("one valid row {chunk:?}"),
            &got,
            &want_loss,
            &want_gx,
            &want_gw,
        );
        let gx = f(got.grad_input.as_ref().unwrap());
        // Every ignored row's grad_input is exactly zero.
        for row in (0..n).filter(|&r| r != 6) {
            assert!(gx[row * d..(row + 1) * d].iter().all(|&g| g == 0.0));
        }
    }
    let all_ignored = Tensor::from_u32(&[99u32; 9], &[n], &ib).unwrap();
    let live = be.budget().live_bytes().unwrap();
    assert_nonfinite(be.linear_cross_entropy_mean(
        &c.x,
        &c.w,
        &all_ignored,
        ignore,
        CeChunk { rows: 2, cols: 3 },
        true,
    ));
    assert_eq!(be.budget().live_bytes().unwrap(), live);
}

#[test]
fn refusals_follow_linear_ce_dims_and_target_range() {
    let be = CpuBackend::new(Budget::new(u64::MAX));
    let ib = Budget::new(u64::MAX);
    let x = Tensor::from_f32(&[0.5; 12], &[3, 4], &ib).unwrap();
    let w = Tensor::from_f32(&[0.25; 20], &[5, 4], &ib).unwrap();
    let t = Tensor::from_u32(&[0, 1, 4], &[3], &ib).unwrap();
    let ok = CeChunk { rows: 2, cols: 2 };
    assert!(be
        .linear_cross_entropy_mean(&x, &w, &t, None, ok, true)
        .is_ok());
    assert_shape(be.linear_cross_entropy_mean(
        &x,
        &w,
        &t,
        None,
        CeChunk { rows: 0, cols: 2 },
        true,
    ));
    assert_shape(be.linear_cross_entropy_mean(
        &x,
        &w,
        &t,
        None,
        CeChunk { rows: 2, cols: 0 },
        true,
    ));
    let w3 = Tensor::from_f32(&[0.25; 15], &[5, 3], &ib).unwrap();
    assert_shape(be.linear_cross_entropy_mean(&x, &w3, &t, None, ok, true));
    let t2 = Tensor::from_u32(&[0, 1], &[2], &ib).unwrap();
    assert_shape(be.linear_cross_entropy_mean(&x, &w, &t2, None, ok, true));
    let x3 = Tensor::from_f32(&[0.5; 12], &[1, 3, 4], &ib).unwrap();
    assert_shape(be.linear_cross_entropy_mean(&x3, &w, &t, None, ok, true));
    let tf = Tensor::from_f32(&[0.0; 3], &[3], &ib).unwrap();
    assert!(matches!(
        be.linear_cross_entropy_mean(&x, &w, &tf, None, ok, true),
        Err(OjasError::Dtype {
            expected: DType::U32,
            ..
        })
    ));
    // A target outside the vocabulary that is not the ignore index.
    let bad = Tensor::from_u32(&[0, 5, 1], &[3], &ib).unwrap();
    assert_range(be.linear_cross_entropy_mean(&x, &w, &bad, None, ok, true));
    assert_range(be.linear_cross_entropy_mean(&x, &w, &bad, Some(4), ok, true));
    assert!(be
        .linear_cross_entropy_mean(&x, &w, &bad, Some(5), ok, true)
        .is_ok());
    assert_eq!(be.budget().live_bytes().unwrap(), 0);
}

/// NaN and infinity in either operand are refused before any charge: a
/// backend with no room at all reports `NonFinite`, not `CapacityExceeded`.
#[test]
fn nonfinite_inputs_are_refused_before_any_charge() {
    let be = CpuBackend::new(Budget::new(0));
    let ib = Budget::new(u64::MAX);
    let t = Tensor::from_u32(&[0, 1, 2], &[3], &ib).unwrap();
    let chunk = CeChunk { rows: 2, cols: 2 };
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut xs = vec![0.5f32; 12];
        xs[7] = bad;
        let x_bad = Tensor::from_f32(&xs, &[3, 4], &ib).unwrap();
        let x = Tensor::from_f32(&[0.5; 12], &[3, 4], &ib).unwrap();
        let mut ws = vec![0.25f32; 20];
        ws[19] = bad;
        let w_bad = Tensor::from_f32(&ws, &[5, 4], &ib).unwrap();
        let w = Tensor::from_f32(&[0.25; 20], &[5, 4], &ib).unwrap();
        assert_nonfinite(be.linear_cross_entropy_mean(&x_bad, &w, &t, None, chunk, true));
        assert_nonfinite(be.linear_cross_entropy_mean(&x, &w_bad, &t, None, chunk, false));
        assert_eq!(be.budget().live_bytes().unwrap(), 0);
    }
    // Finite inputs whose logits overflow are NonFinite too, as in
    // `linear_forward`.
    let big = CpuBackend::new(Budget::new(u64::MAX));
    let x = Tensor::from_f32(&[3.0e38; 4], &[1, 4], &ib).unwrap();
    let w = Tensor::from_f32(&[3.0e38; 8], &[2, 4], &ib).unwrap();
    let t1 = Tensor::from_u32(&[1], &[1], &ib).unwrap();
    assert_nonfinite(big.linear_cross_entropy_mean(&x, &w, &t1, None, chunk, true));
    assert_eq!(big.budget().live_bytes().unwrap(), 0);
}

/// Every cap from nothing up to the op's need is either accepted or refused
/// with `CapacityExceeded`, and a refusal leaves nothing charged.
#[test]
fn budget_refusal_leaves_the_budget_balanced() {
    let mut rng = SplitMix64(0xce05);
    let c = case(&mut rng, 37, 12, 61, Some(3));
    let chunk = CeChunk { rows: 8, cols: 16 };
    for want_grad in [false, true] {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let mut accepted = None;
            for cap in (0..=40_000u64).step_by(256) {
                let be = CpuBackend::new(Budget::new(cap)).with_numerics(numerics);
                match be.linear_cross_entropy_mean(&c.x, &c.w, &c.t, c.ignore, chunk, want_grad) {
                    Ok(out) => {
                        accepted.get_or_insert(cap);
                        drop(out);
                    }
                    Err(OjasError::CapacityExceeded { .. }) => {
                        assert!(
                            accepted.is_none(),
                            "cap {cap} refused after a smaller cap ran"
                        );
                    }
                    Err(err) => panic!("cap {cap}: unexpected {err:?}"),
                }
                assert_eq!(be.budget().live_bytes().unwrap(), 0, "cap {cap}");
            }
            let cap = accepted.expect("no cap up to 40000 bytes admitted the op");
            println!("want_grad {want_grad} {numerics:?}: smallest cap admitted {cap} bytes");
        }
    }
}

/// G3(c): a budget well below the `N * V * 4` bytes of the logits runs the
/// fused op, while the composed `linear_forward` is refused for the logits.
#[test]
fn a_budget_below_the_logits_runs_the_fused_op_and_refuses_the_composed_one() {
    let (n, d, v) = (4096usize, 64usize, 50304usize);
    let logits_bytes = (n * v * 4) as u64;
    let cap = logits_bytes / 8;
    let mut rng = SplitMix64(0xce06);
    let c = case(&mut rng, n, d, v, Some(u32::MAX));
    let be = CpuBackend::with_threads(Budget::new(cap), 4).unwrap();
    let chunk = CeChunk {
        rows: 512,
        cols: 8192,
    };
    let out = be
        .linear_cross_entropy_mean(&c.x, &c.w, &c.t, c.ignore, chunk, true)
        .unwrap();
    let loss = f(&out.loss)[0];
    assert!(loss.is_finite() && loss > 0.0, "{loss}");
    assert_eq!(out.grad_input.as_ref().unwrap().shape(), &[n, d]);
    assert_eq!(out.grad_weight.as_ref().unwrap().shape(), &[v, d]);
    drop(out);
    assert_capacity(be.linear_forward(&c.x, &c.w));
    assert_eq!(be.budget().live_bytes().unwrap(), 0);
    println!("cap {cap} bytes = logits {logits_bytes} / 8: fused ran, composed refused");
}
