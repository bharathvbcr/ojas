//! Softmax Collapse: past the f32 cliff (`1 + exp(-margin)` rounding to 1,
//! margin above ~16.6) a kernel that forms `p - 1` writes a zero target
//! gradient while the other classes stay positive, so the row no longer sums
//! to 0. The loss here is softplus of the negated margin, and the target
//! gradient is the negation of the other classes.
//!
//! The oracle is f64 and does not call the f32 helper.

use ojas_core::{Backend, Budget, CeChunk, CeGrads, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

fn backend(numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(u64::MAX), 2)
        .unwrap()
        .with_numerics(numerics)
}

fn f32_tensor(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn u32_tensor(data: &[u32], shape: &[usize]) -> Tensor {
    Tensor::from_u32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

/// `|got - want|` in units of the f32 spacing at `want`.
fn ulps(got: f32, want: f64) -> f64 {
    let w = want as f32;
    let spacing = if w == 0.0 || !w.is_normal() {
        f64::from(f32::from_bits(1))
    } else {
        let next = f32::from_bits(w.abs().to_bits() + 1);
        f64::from(next) - f64::from(w.abs())
    };
    (f64::from(got) - want).abs() / spacing
}

/// Target-excluded cross-entropy of one row, in f64. An empty other-set is
/// loss 0 and a zero gradient.
fn oracle(row: &[f32], target: usize) -> (f64, Vec<f64>) {
    let z: Vec<f64> = row.iter().copied().map(f64::from).collect();
    let mut max_o = f64::NEG_INFINITY;
    for (j, &v) in z.iter().enumerate() {
        if j != target && v > max_o {
            max_o = v;
        }
    }
    let mut sum_o = 0.0;
    if max_o.is_finite() {
        for (j, &v) in z.iter().enumerate() {
            if j != target {
                sum_o += (v - max_o).exp();
            }
        }
    }
    if sum_o == 0.0 {
        return (0.0, vec![0.0; z.len()]);
    }
    let delta = max_o - z[target];
    let (loss, scale) = if delta <= 0.0 {
        let e = delta.exp();
        let q = sum_o * e;
        (q.ln_1p(), e / (1.0 + q))
    } else {
        let e = (-delta).exp();
        (delta + (sum_o + e).ln(), 1.0 / (e + sum_o))
    };
    let grad = z
        .iter()
        .enumerate()
        .map(|(j, &v)| {
            if j == target {
                -sum_o * scale
            } else {
                (v - max_o).exp() * scale
            }
        })
        .collect();
    (loss, grad)
}

fn assert_matches(what: &str, got_loss: f32, got_grad: &[f32], row: &[f32], target: usize) {
    let (want_loss, want_grad) = oracle(row, target);
    let loss_ulps = ulps(got_loss, want_loss);
    assert!(
        loss_ulps <= 16.0,
        "{what}: loss {got_loss} vs {want_loss} ({loss_ulps} ulps)"
    );
    assert_eq!(got_grad.len(), want_grad.len(), "{what}: grad length");
    for (j, (&got, &want)) in got_grad.iter().zip(&want_grad).enumerate() {
        let u = ulps(got, want);
        assert!(u <= 16.0, "{what}: grad[{j}] {got} vs {want} ({u} ulps)");
    }
    let sum: f64 = got_grad.iter().map(|g| f64::from(*g)).sum();
    let peak = got_grad.iter().fold(0.0f32, |m, g| m.max(g.abs()));
    if peak == 0.0 {
        assert_eq!(sum, 0.0, "{what}: zero row summed to {sum}");
        return;
    }
    let next = f32::from_bits(peak.to_bits() + 1);
    let spacing = f64::from(next) - f64::from(peak);
    assert!(
        sum.abs() <= 8.0 * spacing,
        "{what}: row sum {sum} of {got_grad:?} is more than 8 ulps of the peak {peak}"
    );
}

fn unfused(
    numerics: Numerics,
    logits: &[f32],
    vocab: usize,
    targets: &[u32],
    ignore: Option<u32>,
) -> (f32, Vec<f32>) {
    let be = backend(numerics);
    let rows = targets.len();
    let lt = f32_tensor(logits, &[rows, vocab]);
    let tt = u32_tensor(targets, &[rows]);
    let loss = be
        .cross_entropy_mean_forward(&lt, &tt, ignore)
        .unwrap()
        .to_f32_vec()
        .unwrap()[0];
    let grad = be
        .cross_entropy_mean_backward(&lt, &tt, ignore)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    (loss, grad)
}

/// `[9, -9, -9]` at target 0, margin 18. A second ignored row (ignore id
/// outside the vocabulary) keeps the mean divisor at 1. Fused and unfused,
/// both numerics, and both vocabulary tilings.
#[test]
fn a_row_past_the_fp32_cliff_keeps_a_nonzero_target_gradient() {
    let row = [9.0f32, -9.0, -9.0];
    let logits = [9.0f32, -9.0, -9.0, 1.0, 2.0, 3.0];
    let targets = [0u32, 99];
    let ignore = Some(99u32);
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let (loss, grad) = unfused(numerics, &logits, 3, &targets, ignore);
        assert!(
            grad[0] != 0.0,
            "{numerics:?}: target gradient collapsed ({grad:?})"
        );
        assert_matches(&format!("{numerics:?} unfused"), loss, &grad[..3], &row, 0);
        assert!(
            grad[3..].iter().all(|g| *g == 0.0),
            "{numerics:?}: ignored row {grad:?}"
        );
    }

    let x = f32_tensor(&[1.0, 1.0], &[2, 1]);
    let w = f32_tensor(&row, &[3, 1]);
    let tt = u32_tensor(&targets, &[2]);
    let (want_loss, want_grad) = unfused(Numerics::Exact, &logits, 3, &targets, ignore);
    for cols in [1usize, 3] {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let be = backend(numerics);
            let out = be
                .linear_cross_entropy_mean(
                    &x,
                    &w,
                    &tt,
                    ignore,
                    CeChunk { rows: 2, cols },
                    CeGrads {
                        input: false,
                        weight: true,
                    },
                )
                .unwrap();
            let loss = out.loss.to_f32_vec().unwrap()[0];
            let dw = out.grad_weight.unwrap().to_f32_vec().unwrap();
            let tag = format!("{numerics:?} fused cols {cols}");
            if numerics == Numerics::Exact {
                assert_eq!(loss.to_bits(), want_loss.to_bits(), "{tag} loss bits");
                assert_eq!(dw, want_grad[..3], "{tag} grad bits");
            }
            assert!(dw[0] != 0.0, "{tag}: target gradient collapsed ({dw:?})");
            assert_matches(&tag, loss, &dw, &row, 0);
        }
    }
}

/// Below the cliff, and on the positive-margin branch, the same oracle.
#[test]
fn rows_off_the_cliff_match_the_f64_oracle() {
    let cases: &[(&[f32], usize)] = &[
        (&[1.0, 0.0, -0.5], 0),
        (&[0.0, 3.0, 2.5], 0),
        (&[0.25, 0.25, 0.25], 1),
    ];
    for numerics in [Numerics::Exact, Numerics::Fast] {
        for (row, target) in cases {
            let (loss, grad) = unfused(numerics, row, row.len(), &[*target as u32], None);
            assert_matches(&format!("{numerics:?} {row:?}"), loss, &grad, row, *target);
        }
    }
}

/// One class has nothing to compete with: loss 0 and a zero gradient, not
/// `ln(0)`.
#[test]
fn a_one_class_row_is_loss_zero_and_a_zero_gradient() {
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let (loss, grad) = unfused(numerics, &[4.0], 1, &[0], None);
        assert_eq!(loss, 0.0, "{numerics:?}");
        assert_eq!(grad, [0.0], "{numerics:?}");
    }
}

/// Skipping the target does not skip the finiteness scan. A NaN in an
/// ignored row is still refused.
#[test]
fn a_nan_in_an_ignored_row_is_refused() {
    let logits = f32_tensor(&[9.0, -9.0, -9.0, f32::NAN, 0.0, 0.0], &[2, 3]);
    let targets = u32_tensor(&[0, 99], &[2]);
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let be = backend(numerics);
        let fwd = be.cross_entropy_mean_forward(&logits, &targets, Some(99));
        assert!(
            matches!(fwd, Err(OjasError::NonFinite { .. })),
            "{numerics:?} forward: {fwd:?}"
        );
        let bwd = be.cross_entropy_mean_backward(&logits, &targets, Some(99));
        assert!(
            matches!(bwd, Err(OjasError::NonFinite { .. })),
            "{numerics:?} backward: {bwd:?}"
        );
    }
}

/// Textbook cross-entropy of one row in f64: `max + ln(sum exp(z - max)) - z_t`
/// and `softmax - onehot`. The target is in the sum, and nothing here shares a
/// formula with the target-excluded oracle above, so a mistake in that
/// formulation shows as a disagreement between the two. Accurate in f64 up to
/// a margin of about 36 (`1 + exp(-margin)` rounds to 1 in f64 past that), far
/// beyond the f32 cliff.
fn standard_f64(row: &[f32], target: usize) -> (f64, Vec<f64>) {
    let z: Vec<f64> = row.iter().copied().map(f64::from).collect();
    let max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = z.iter().map(|v| (v - max).exp()).sum();
    let loss = max + sum.ln() - z[target];
    let grad = z
        .iter()
        .enumerate()
        .map(|(j, v)| (v - max).exp() / sum - if j == target { 1.0 } else { 0.0 })
        .collect();
    (loss, grad)
}

/// A row of `vocab` classes in `[-3, 3)` from a fixed integer sequence, with
/// the target's logit set `margin` above the largest other logit.
fn margin_row(vocab: usize, target: usize, margin: f32) -> Vec<f32> {
    let mut state = 0x2545_f491u32 ^ (vocab as u32) << 8 ^ target as u32;
    let mut row: Vec<f32> = (0..vocab)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 6.0 - 3.0
        })
        .collect();
    let max_other = row
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != target)
        .fold(f32::NEG_INFINITY, |m, (_, &v)| m.max(v));
    row[target] = max_other + margin;
    row
}

fn assert_matches_standard(what: &str, loss: f32, grad: &[f32], row: &[f32], target: usize) {
    let (want_loss, want_grad) = standard_f64(row, target);
    let loss_ulps = ulps(loss, want_loss);
    assert!(
        loss_ulps <= 16.0,
        "{what}: loss {loss} vs standard f64 {want_loss} ({loss_ulps} ulps)"
    );
    for (j, (&got, &want)) in grad.iter().zip(&want_grad).enumerate() {
        let u = ulps(got, want);
        assert!(
            u <= 16.0,
            "{what}: grad[{j}] {got} vs standard f64 {want} ({u} ulps)"
        );
    }
}

/// At margins up to 18, where f64 standard cross-entropy is still accurate,
/// the target-excluded kernels agree with it: a wrong excluded formulation
/// could not pass both this and the oracle above. Vocabulary 40 with targets
/// in lane 0 of the first chunk (0), in a later chunk's lane (21: chunk 1,
/// lane 5), at a chunk edge (16, 31), in the tail past the last whole chunk
/// (37, 39), reaching the lane-chunk skip in `exp_sub_sum` and `exp_sub_store`
/// (Fast), and the fused path with a tile narrower than the vocabulary so the
/// target sits in a later column tile.
#[test]
fn excluded_cross_entropy_matches_standard_f64_up_to_margin_18_at_any_target_lane() {
    let vocab = 40usize;
    for target in [0usize, 5, 16, 21, 31, 37, 39] {
        for margin in [-4.0f32, 0.0, 1.0, 8.0, 16.0, 18.0] {
            let row = margin_row(vocab, target, margin);
            for numerics in [Numerics::Exact, Numerics::Fast] {
                let tag = format!("{numerics:?} target {target} margin {margin}");
                let (loss, grad) = unfused(numerics, &row, vocab, &[target as u32], None);
                assert_matches_standard(&format!("{tag} unfused"), loss, &grad, &row, target);
                assert_matches(&format!("{tag} unfused vs excluded"), loss, &grad, &row, target);

                let x = f32_tensor(&[1.0], &[1, 1]);
                let w = f32_tensor(&row, &[vocab, 1]);
                let tt = u32_tensor(&[target as u32], &[1]);
                for cols in [7usize, vocab] {
                    let be = backend(numerics);
                    let out = be
                        .linear_cross_entropy_mean(
                            &x,
                            &w,
                            &tt,
                            None,
                            CeChunk { rows: 1, cols },
                            CeGrads {
                                input: false,
                                weight: true,
                            },
                        )
                        .unwrap();
                    let loss = out.loss.to_f32_vec().unwrap()[0];
                    let dw = out.grad_weight.unwrap().to_f32_vec().unwrap();
                    let tag = format!("{tag} fused cols {cols}");
                    assert_matches_standard(&tag, loss, &dw, &row, target);
                }
            }
        }
    }
}

/// One class: loss 0 and a gradient of positive zero, bit for bit. The CPU
/// used to write `-sum * scale` for the target, `-0.0` when both are zero,
/// where the GPU kernels write `+0.0`. `-0.0 == 0.0`, so only the bits tell.
#[test]
fn a_one_class_row_writes_positive_zero_not_negative_zero() {
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let (loss, grad) = unfused(numerics, &[4.0], 1, &[0], None);
        assert_eq!(loss.to_bits(), 0, "{numerics:?} loss bits");
        assert_eq!(grad[0].to_bits(), 0, "{numerics:?} unfused grad bits");

        let x = f32_tensor(&[1.0], &[1, 1]);
        let w = f32_tensor(&[4.0], &[1, 1]);
        let tt = u32_tensor(&[0], &[1]);
        let out = backend(numerics)
            .linear_cross_entropy_mean(
                &x,
                &w,
                &tt,
                None,
                CeChunk { rows: 1, cols: 1 },
                CeGrads {
                    input: true,
                    weight: true,
                },
            )
            .unwrap();
        assert_eq!(
            out.loss.to_f32_vec().unwrap()[0].to_bits(),
            0,
            "{numerics:?} fused loss bits"
        );
        for (name, g) in [("dx", out.grad_input), ("dw", out.grad_weight)] {
            let g = g.unwrap().to_f32_vec().unwrap();
            assert_eq!(g[0].to_bits(), 0, "{numerics:?} fused {name} bits");
        }
    }
}
