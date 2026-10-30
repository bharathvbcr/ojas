//! Softmax Collapse on the CPU (ft-f40eeed8).
//!
//! A max-shifted softmax forms `p_t - 1` at the target, and past the f32
//! cliff (`1 + sum_{j != t} e^(z_j - z_t)` rounding to 1, a margin above
//! about 16.6) that is exactly 0 while every other class stays positive, so
//! the logit gradient no longer sums to 0 and the loss reads 0.
//!
//! The rule under test: a row keeps the standard form bit for bit unless its
//! target logit is the row maximum and the working-precision sum of
//! `e^(z - max)` is exactly 1.0. Those rows alone take the target-excluded
//! form with `S = sum_{j != t} e^(z_j - z_t)`: loss `log1p(S)`, target
//! gradient `-S / (1 + S) / valid`, others `e^(z_j - z_t) / (1 + S) / valid`.
//!
//! Two kinds of check:
//!
//! * the switched rows against an f64 oracle that never forms `p - 1` (it
//!   is the target-excluded expression in f64, cross-checked against the
//!   textbook form where f64 still resolves it);
//! * the rows that do not switch against digests recorded from af8bd9c, the
//!   commit before this change, by running this file there. They are bit
//!   comparisons, not a recompute.

use ojas_core::{exp_exact, Backend, Budget, CeChunk, CeGrads, Numerics, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_nonfinite, bits};

fn backend(numerics: Numerics, threads: usize) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 30), threads)
        .unwrap()
        .with_numerics(numerics)
}

fn f32t(be: &CpuBackend, data: &[f32], shape: &[usize]) -> Tensor {
    common::f32t(be, data, shape)
}

fn fnv(values: impl IntoIterator<Item = u32>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        for byte in v.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
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

/// One row of the cross-entropy, in f64, with no `p - 1`: `S` is the sum of
/// `e^(z_j - z_t)` over the other classes, the loss is `ln(1 + S)`, the
/// target gradient `-S / (1 + S)` and each other class `e^(z_j - z_t) / (1 +
/// S)`, all over `denom`. A one-class row is loss 0 and gradient 0.
fn oracle(row: &[f32], target: usize, denom: f64) -> (f64, Vec<f64>, f64) {
    let zt = f64::from(row[target]);
    let e: Vec<f64> = row
        .iter()
        .enumerate()
        .map(|(j, &v)| {
            if j == target {
                0.0
            } else {
                (f64::from(v) - zt).exp()
            }
        })
        .collect();
    let s: f64 = e.iter().sum();
    let grad = e
        .iter()
        .enumerate()
        .map(|(j, &ej)| {
            if j == target {
                -s / (1.0 + s) / denom
            } else {
                ej / (1.0 + s) / denom
            }
        })
        .collect();
    (s.ln_1p(), grad, s)
}

/// The textbook `softmax - onehot` and `lse - z_t` in f64, for the margins
/// where `1 + e^-margin` is still resolved there (up to about 36).
fn textbook(row: &[f32], target: usize) -> (f64, Vec<f64>) {
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

/// A row of `vocab` classes in `[-3, 3)` on a 2^-12 grid (so every f32
/// difference of two logits is exact), with the target `margin` above the
/// largest other logit.
fn margin_row(vocab: usize, target: usize, margin: f32) -> Vec<f32> {
    let mut state = 0x2545_f491u32 ^ (vocab as u32) << 8 ^ target as u32;
    let mut row: Vec<f32> = (0..vocab)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let x = (state >> 8) as f32 / (1u32 << 24) as f32 * 6.0 - 3.0;
            (x * 4096.0).round() / 4096.0
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

/// The standard form's f32 sum in the Exact reference order: `e^(z - max)`
/// by `exp_exact`, added in ascending index order.
fn exact_sum(row: &[f32]) -> f32 {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &v in row {
        sum += exp_exact(v - max);
    }
    sum
}

/// Whether the Exact tier switches this row: the target is the row maximum
/// and the sum is exactly 1.0.
fn exact_switches(row: &[f32], target: usize) -> bool {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    row[target] == max && exact_sum(row) == 1.0
}

/// Loss and the `[rows, vocab]` gradient of the unfused pair.
fn unfused(
    be: &CpuBackend,
    logits: &[f32],
    vocab: usize,
    targets: &[u32],
    ignore: Option<u32>,
) -> (f32, Vec<f32>) {
    let rows = targets.len();
    let lt = f32t(be, logits, &[rows, vocab]);
    let tt = common::u32t(be, targets, &[rows]);
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

/// The same loss and gradient through `linear_cross_entropy_mean`, whose
/// logits tile is `x · Wᵀ` with `x` the identity and `W` the transposed
/// logits, so every logit is exact. `cols` is the vocabulary tile.
fn fused(
    be: &CpuBackend,
    logits: &[f32],
    vocab: usize,
    targets: &[u32],
    ignore: Option<u32>,
    cols: usize,
) -> (f32, Vec<f32>) {
    let rows = targets.len();
    let mut x = vec![0.0f32; rows * rows];
    for r in 0..rows {
        x[r * rows + r] = 1.0;
    }
    let mut w = vec![0.0f32; vocab * rows];
    for r in 0..rows {
        for v in 0..vocab {
            w[v * rows + r] = logits[r * vocab + v];
        }
    }
    let out = be
        .linear_cross_entropy_mean(
            &f32t(be, &x, &[rows, rows]),
            &f32t(be, &w, &[vocab, rows]),
            &common::u32t(be, targets, &[rows]),
            ignore,
            CeChunk { rows, cols },
            CeGrads {
                input: false,
                weight: true,
            },
        )
        .unwrap();
    let gw = out.grad_weight.unwrap().to_f32_vec().unwrap();
    let mut grad = vec![0.0f32; rows * vocab];
    for r in 0..rows {
        for v in 0..vocab {
            grad[r * vocab + v] = gw[v * rows + r];
        }
    }
    (out.loss.to_f32_vec().unwrap()[0], grad)
}

/// A switched row: nonzero target gradient, the row summing to 0 within a
/// few ulps of its peak, and loss and gradients within `gate` ulps of the
/// f64 oracle.
fn assert_switched_row(what: &str, loss: f32, grad: &[f32], row: &[f32], target: usize) {
    let (want_loss, want_grad, s) = oracle(row, target, 1.0);
    assert!(s > 0.0, "{what}: a row with no other class cannot switch");
    assert!(
        grad[target] < 0.0,
        "{what}: target gradient collapsed to {} (S {s:e})",
        grad[target]
    );
    let gate = 16.0;
    let l = ulps(loss, want_loss);
    assert!(
        l <= gate,
        "{what}: loss {loss:e} vs log1p(S) {want_loss:e} ({l} ulps)"
    );
    for (j, (&got, &want)) in grad.iter().zip(&want_grad).enumerate() {
        let u = ulps(got, want);
        assert!(
            u <= gate,
            "{what}: grad[{j}] {got:e} vs {want:e} ({u} ulps)"
        );
    }
    let sum: f64 = grad.iter().map(|g| f64::from(*g)).sum();
    let peak = grad.iter().fold(0.0f32, |m, g| m.max(g.abs()));
    let spacing = f64::from(f32::from_bits(peak.to_bits() + 1)) - f64::from(peak);
    // The target gradient is `S / (1 + S)` with S an f32 sum of `vocab - 1`
    // terms, while the others are single roundings, so the row sum carries
    // the sum's accumulated error: a few half-ulp steps in a random walk,
    // `sqrt(vocab)` of them, on top of the final roundings.
    let gate = 4.0 + (grad.len() as f64).sqrt();
    assert!(
        sum.abs() <= gate * spacing,
        "{what}: row sums to {sum:e}, more than {gate:.1} ulps of the peak {peak:e}"
    );
}

/// `[9, -9, -9]` at target 0, margin 18: the paper's row. A second ignored
/// row (an id outside the vocabulary) keeps the mean divisor at 1.
#[test]
fn a_row_past_the_fp32_cliff_keeps_a_nonzero_target_gradient() {
    let row = [9.0f32, -9.0, -9.0];
    let logits = [9.0f32, -9.0, -9.0, 1.0, 2.0, 3.0];
    let targets = [0u32, 99];
    let ignore = Some(99u32);
    let (_, _, s) = oracle(&row, 0, 1.0);
    assert!((s - 2.0 * (-18.0f64).exp()).abs() < 1e-15);
    // The oracle agrees with the textbook form where f64 still resolves it.
    let (tl, tg) = textbook(&row, 0);
    let (ol, og, _) = oracle(&row, 0, 1.0);
    assert!((tl - ol).abs() <= 1e-6 * ol, "textbook {tl:e} vs {ol:e}");
    for (t, o) in tg.iter().zip(&og) {
        assert!((t - o).abs() <= 1e-6 * o.abs(), "textbook {t:e} vs {o:e}");
    }
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let be = backend(numerics, 2);
        let (loss, grad) = unfused(&be, &logits, 3, &targets, ignore);
        assert_switched_row(&format!("{numerics:?} unfused"), loss, &grad[..3], &row, 0);
        assert!(
            grad[3..].iter().all(|g| *g == 0.0),
            "{numerics:?}: ignored row"
        );
        for cols in [2usize, 3] {
            let (loss, grad) = fused(&be, &logits, 3, &targets, ignore, cols);
            assert_switched_row(
                &format!("{numerics:?} fused cols {cols}"),
                loss,
                &grad[..3],
                &row,
                0,
            );
            assert!(grad[3..].iter().all(|g| *g == 0.0));
        }
    }
}

const VOCAB: usize = 301;
/// Targets in lane 0 and lane 7 of the first 8-wide chunk, at a chunk edge,
/// mid-row, in lane 0 of a late chunk and in the 5-wide tail.
const TARGETS: [usize; 8] = [0, 7, 8, 63, 150, 296, 299, 300];
const MARGINS: [f32; 14] = [
    14.0, 15.0, 16.0, 16.6, 17.0, 17.5, 18.0, 19.0, 20.0, 22.0, 25.0, 30.0, 36.0, 40.0,
];

/// Margins 14 to 40 at a 301-class vocabulary, every target position, both
/// tiers, the unfused path and the fused path at tiles of 64 and the whole
/// vocabulary. The Exact tier switches exactly where [`exact_switches`] says
/// and those rows meet the oracle; Fast is held to the oracle where the sum
/// is certain to be 1.0 (S below 2^-25). Every row, switched or not, keeps a
/// nonzero target gradient.
#[test]
fn the_margin_sweep_across_the_cliff_keeps_the_target_gradient() {
    let mut switched_exact = 0;
    for &target in &TARGETS {
        for &margin in &MARGINS {
            let row = margin_row(VOCAB, target, margin);
            let t = [target as u32];
            let (_, _, s) = oracle(&row, target, 1.0);
            let tag = format!("target {target} margin {margin} S {s:e}");
            let exact = backend(Numerics::Exact, 2);
            let (loss, grad) = unfused(&exact, &row, VOCAB, &t, None);
            assert!(
                grad[target] < 0.0,
                "Exact {tag}: target gradient {}",
                grad[target]
            );
            if exact_switches(&row, target) {
                switched_exact += 1;
                assert_switched_row(&format!("Exact {tag}"), loss, &grad, &row, target);
            }
            for cols in [64usize, VOCAB] {
                let (fl, fg) = fused(&exact, &row, VOCAB, &t, None, cols);
                assert_eq!(
                    bits(&fg),
                    bits(&grad),
                    "fused cols {cols} vs unfused: {tag}"
                );
                assert_eq!(
                    fl.to_bits(),
                    loss.to_bits(),
                    "fused cols {cols} loss: {tag}"
                );
            }
            let fast = backend(Numerics::Fast, 2);
            let (loss, grad) = unfused(&fast, &row, VOCAB, &t, None);
            assert!(
                grad[target] < 0.0,
                "Fast {tag}: target gradient {}",
                grad[target]
            );
            if s < 2.0f64.powi(-25) {
                assert_switched_row(&format!("Fast {tag}"), loss, &grad, &row, target);
            }
        }
    }
    assert!(
        switched_exact >= 8,
        "the sweep reached {switched_exact} switched rows"
    );
}

/// A switched row's bits do not depend on the thread count or on where the
/// pool splits the rows.
#[test]
fn switched_rows_are_bit_identical_across_thread_counts() {
    let rows: Vec<Vec<f32>> = TARGETS
        .iter()
        .map(|&t| margin_row(VOCAB, t, 30.0))
        .collect();
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let targets: Vec<u32> = TARGETS.iter().map(|&t| t as u32).collect();
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let (l1, g1) = unfused(&backend(numerics, 1), &flat, VOCAB, &targets, None);
        for threads in [2usize, 7, 18] {
            let (l, g) = unfused(&backend(numerics, threads), &flat, VOCAB, &targets, None);
            assert_eq!(
                l.to_bits(),
                l1.to_bits(),
                "{numerics:?} {threads} threads: loss"
            );
            assert_eq!(bits(&g), bits(&g1), "{numerics:?} {threads} threads: grad");
        }
        for (r, &t) in TARGETS.iter().enumerate() {
            assert!(g1[r * VOCAB + t] < 0.0, "{numerics:?} row {r}");
        }
    }
}

/// Switched and ordinary rows in one batch: each switched row meets the
/// oracle at the batch's divisor, and the ordinary rows are untouched.
#[test]
fn a_batch_mixes_switched_and_ordinary_rows() {
    let vocab = 40;
    let specs = [
        (5usize, 30.0f32),
        (11, 0.5),
        (39, 25.0),
        (0, -2.0),
        (20, 40.0),
    ];
    let rows: Vec<Vec<f32>> = specs
        .iter()
        .map(|&(t, m)| margin_row(vocab, t, m))
        .collect();
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let targets: Vec<u32> = specs.iter().map(|&(t, _)| t as u32).collect();
    let denom = specs.len() as f64;
    let be = backend(Numerics::Exact, 3);
    let (loss, grad) = unfused(&be, &flat, vocab, &targets, None);
    let mut total = 0.0f64;
    for (r, &(t, _)) in specs.iter().enumerate() {
        let row = &rows[r];
        let g = &grad[r * vocab..(r + 1) * vocab];
        let (l, want, _) = oracle(row, t, denom);
        total += l;
        if exact_switches(row, t) {
            assert!(g[t] < 0.0, "row {r}");
        }
        for (j, (&got, &w)) in g.iter().zip(&want).enumerate() {
            // The ordinary rows at margin 0.5 and -2 are not near the cliff,
            // so the standard form meets the oracle too.
            assert!(ulps(got, w) <= 64.0, "row {r} grad[{j}] {got:e} vs {w:e}");
        }
    }
    let l = ulps(loss, total / denom);
    assert!(l <= 64.0, "loss {loss:e} vs {:e} ({l} ulps)", total / denom);
}

/// Two-class rows past the cliff (`1 + S` is 1.0): the target gradient is the
/// exact negation of the other class's in both tiers and through the fused
/// path, whichever lane the target is in, so the row sums to exactly 0. This
/// is the relation the other backends are held to as well.
#[test]
fn two_class_rows_past_the_cliff_sum_to_exactly_zero() {
    for target in [0usize, 1] {
        for margin in 18..=40 {
            let mut row = [0.0f32; 2];
            row[target] = margin as f32;
            for numerics in [Numerics::Exact, Numerics::Fast] {
                let be = backend(numerics, 2);
                let t = [target as u32];
                let (_, grad) = unfused(&be, &row, 2, &t, None);
                let (_, fused_grad) = fused(&be, &row, 2, &t, None, 1);
                for (name, g) in [("unfused", grad), ("fused", fused_grad)] {
                    assert_eq!(
                        g[target].to_bits(),
                        (-g[1 - target]).to_bits(),
                        "{numerics:?} {name} target {target} margin {margin}: {g:?}"
                    );
                    assert!(g[target] < 0.0);
                }
            }
        }
    }
}

/// One-class rows and an all-ignored batch behave as before: loss and
/// gradient are +0.0 (the standard form's `x - x`), and no valid row is
/// refused.
#[test]
fn one_class_rows_and_an_all_ignored_batch_keep_todays_behaviour() {
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let be = backend(numerics, 2);
        let (loss, grad) = unfused(&be, &[5.0, -3.0, 0.0], 1, &[0, 0, 0], None);
        assert_eq!(loss.to_bits(), 0, "{numerics:?} loss");
        assert!(
            grad.iter().all(|g| g.to_bits() == 0),
            "{numerics:?} grad {grad:?}"
        );
        let (loss, grad) = fused(&be, &[5.0, -3.0, 0.0], 1, &[0, 0, 0], None, 1);
        assert_eq!(loss.to_bits(), 0, "{numerics:?} fused loss");
        assert!(
            grad.iter().all(|g| g.to_bits() == 0),
            "{numerics:?} fused grad {grad:?}"
        );

        let lt = f32t(&be, &[9.0, -9.0, -9.0, 1.0, 2.0, 3.0], &[2, 3]);
        let tt = common::u32t(&be, &[1, 1], &[2]);
        assert_nonfinite(be.cross_entropy_mean_forward(&lt, &tt, Some(1)));
        assert_nonfinite(be.cross_entropy_mean_backward(&lt, &tt, Some(1)));
    }
}

/// Rows past the margin where even the excluded sum underflows: S is 0, so
/// the standard form's zeros stand and nothing turns into a NaN.
#[test]
fn a_row_whose_other_classes_underflow_keeps_zero_gradients() {
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let be = backend(numerics, 2);
        let row = [200.0f32, 0.0, -1.0, 3.0];
        let (loss, grad) = unfused(&be, &row, 4, &[0], None);
        assert_eq!(loss.to_bits(), 0, "{numerics:?}");
        assert!(
            grad.iter().all(|g| g.to_bits() == 0),
            "{numerics:?} {grad:?}"
        );
    }
}

/// A batch the digests cover: vocabulary, flat logits, targets, the ignored
/// id, and whether the Fast tier is digested too (it is not for the cliff
/// batch: Fast adds in lane order, so a row the Exact order leaves above 1.0
/// can sum to exactly 1.0 there and switch).
type Batch = (usize, Vec<f32>, Vec<u32>, Option<u32>, bool);

/// Rows whose standard form is not 0 at the target, so none switches: the
/// sweep rows the Exact reference sum leaves above 1.0 (every target
/// position, margins 14 to 40), margins from -6 to 13 at vocabularies 3 and
/// 40, ties at the maximum (sum at least 2), targets far below the maximum,
/// and an ignored row with a huge margin between valid ones.
fn ordinary_batches() -> Vec<Batch> {
    let mut batches = Vec::new();
    let mut cliff = (Vec::new(), Vec::new());
    for &target in &TARGETS {
        for &margin in &MARGINS {
            let row = margin_row(VOCAB, target, margin);
            if !exact_switches(&row, target) {
                cliff.0.extend(row);
                cliff.1.push(target as u32);
            }
        }
    }
    batches.push((VOCAB, cliff.0, cliff.1, None, false));
    for vocab in [3usize, 40] {
        let mut flat = Vec::new();
        let mut targets = Vec::new();
        for &target in &[0usize, 1, vocab - 1] {
            for &margin in &[-6.0f32, -1.0, 0.0, 3.0, 9.0, 13.0] {
                flat.extend(margin_row(vocab, target, margin));
                targets.push(target as u32);
            }
        }
        batches.push((vocab, flat, targets, None, true));
    }
    let mut flat = vec![9.0f32, 9.0, -30.0, 9.0, -30.0, 9.0, 9.0, 9.0, 9.0];
    let mut targets = vec![0u32, 0, 2];
    flat.extend([30.0f32, -30.0, 5.0, -2.0, -40.0, 40.0]);
    targets.extend([1u32, 0]);
    flat.extend([1.0f32, 2.0, 3.0, 50.0, -50.0, 0.0]);
    targets.extend([2u32, 7]);
    batches.push((3, flat, targets, Some(7), true));
    batches
}

/// Digests of the ordinary batches, recorded by running this test at
/// af8bd9c: `[exact grad, exact loss, fast grad, fast loss, fused grad,
/// fused loss]` per batch. The loss digests hold on Apple silicon only (the
/// platform `ln`, as `exact_golden`'s `ce_loss`); the cliff batch has no
/// Fast digests.
const ORDINARY: [[u64; 6]; 4] = [
    [
        0xc194f19f8aae5197,
        0x181bb62a01db3171,
        0,
        0,
        0xc194f19f8aae5197,
        0x181bb62a01db3171,
    ],
    [
        0xaf5fb08e0e8b0696,
        0x3f1c61d5677034fe,
        0xaf5fb08e0e8b0696,
        0x3f1c61d5677034fe,
        0xaf5fb08e0e8b0696,
        0x3f1c61d5677034fe,
    ],
    [
        0xdc2b3bf073033f7c,
        0x81615a40c37d9cef,
        0x1b22d66874aa32cf,
        0x81615a40c37d9cef,
        0xdc2b3bf073033f7c,
        0x81615a40c37d9cef,
    ],
    [
        0xc51b2b04a6f04754,
        0xf5eaa116f7a2611c,
        0x6c9d98b2f39291ee,
        0xf5eaa116f7a2611c,
        0xc51b2b04a6f04754,
        0xf5eaa116f7a2611c,
    ],
];

#[test]
fn ordinary_rows_are_bit_identical_to_the_commit_before() {
    let recorded_here = cfg!(all(target_os = "macos", target_arch = "aarch64"));
    let names = [
        "exact grad",
        "exact loss",
        "fast grad",
        "fast loss",
        "fused grad",
        "fused loss",
    ];
    let mut seen = Vec::new();
    for (b, (vocab, flat, targets, ignore, fast)) in ordinary_batches().into_iter().enumerate() {
        for (r, &t) in targets.iter().enumerate() {
            if ignore == Some(t) {
                continue;
            }
            let row = &flat[r * vocab..(r + 1) * vocab];
            assert!(
                !exact_switches(row, t as usize),
                "batch {b} row {r} switches"
            );
        }
        let mut got = [0u64; 6];
        for (k, numerics) in [Numerics::Exact, Numerics::Fast].into_iter().enumerate() {
            if numerics == Numerics::Fast && !fast {
                continue;
            }
            let (loss, grad) = unfused(&backend(numerics, 3), &flat, vocab, &targets, ignore);
            got[2 * k] = fnv(grad.iter().map(|g| g.to_bits()));
            got[2 * k + 1] = fnv([loss.to_bits()]);
        }
        let cols = vocab.clamp(1, 7);
        let (loss, grad) = fused(
            &backend(Numerics::Exact, 3),
            &flat,
            vocab,
            &targets,
            ignore,
            cols,
        );
        got[4] = fnv(grad.iter().map(|g| g.to_bits()));
        got[5] = fnv([loss.to_bits()]);
        eprintln!("ORDINARY batch {b} {got:#018x?}");
        seen.push((got, fast));
    }
    for (b, (got, fast)) in seen.iter().enumerate() {
        for (k, name) in names.iter().enumerate() {
            let skipped = (k % 2 == 1 && !recorded_here) || ((k == 2 || k == 3) && !fast);
            if !skipped {
                assert_eq!(
                    got[k], ORDINARY[b][k],
                    "batch {b} {name}: {:#018x} recorded {:#018x}",
                    got[k], ORDINARY[b][k]
                );
            }
        }
    }
}
