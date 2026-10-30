//! K10 Softmax Collapse on the host mirror (ft-f40eeed8).
//!
//! `ce_rows_mirror` is the operation-for-operation host twin of the device
//! walk (`qd_ce_lse_update_f32`, `qd_ce_softmax_grad_f32`), so it is where the
//! kernel arithmetic is checked on a machine with no GPU. The device kernels
//! themselves are compiled by `--features cuda` and are NOT RUN here.
//!
//! A row keeps the standard form bit for bit unless its target logit is the
//! row maximum and the f32 sum of `e^(z - max)` is exactly 1.0; only those
//! rows take the target-excluded form (loss `ln(1 + S)`, target gradient
//! `-S / (1 + S)`, others `e^(z_j - z_t) / (1 + S)`, `S` the sum over the
//! other classes). The oracle is [`reference::ce`] (f64, no `p - 1`) and a
//! second formula written here; the rows that do not switch are held to
//! digests recorded from af8bd9c, the commit before this change.
//!
//! A row is built as `hidden = 1`'s one-row problem: `h = [1]` and `W` the
//! column of logits, so the logits are exact and `dW` is the logit gradient.
//! The batches use `h` the identity and `W = logitsᵀ`, which is exact too.

mod reference;

use ojas_cuda::ce_rows::{ce_rows_mirror, CeMirror, CeRowsPlan, Reduction};
use ojas_cuda::host_ref::Operands;
use ojas_cuda::small_common::Window;
use reference::ce::{cross_entropy_rows_f64, Reduction as RefReduction};

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

/// `(loss, gradient, S)` of one row from the target-excluded expression, in
/// f64 (written here, apart from `reference::ce`).
fn oracle(row: &[f32], target: usize) -> (f64, Vec<f64>, f64) {
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
                -s / (1.0 + s)
            } else {
                ej / (1.0 + s)
            }
        })
        .collect();
    (s.ln_1p(), grad, s)
}

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

/// The mirror over `rows` logit rows (`logits` is `[rows, vocab]`) in chunks
/// of `chunk` classes: `(per-row losses, [rows, vocab] gradient)`. `h` is the
/// identity and `W` the transposed logits, so `dW[v][r] = G[r][v]`.
fn mirror(
    logits: &[f32],
    vocab: usize,
    targets: &[u32],
    chunk: u64,
    reduction: Reduction,
) -> (Vec<f64>, f64, Vec<f32>) {
    let rows = targets.len();
    let mut h = vec![0.0f32; rows * rows];
    for r in 0..rows {
        h[r * rows + r] = 1.0;
    }
    let mut w = vec![0.0f32; vocab * rows];
    for r in 0..rows {
        for v in 0..vocab {
            w[v * rows + r] = logits[r * vocab + v];
        }
    }
    let plan = CeRowsPlan::new(
        (rows as u64, rows as u64, vocab as u64),
        chunk,
        (rows as u64, Window::dense(rows as u64)),
        reduction,
    )
    .unwrap();
    let picked: Vec<u32> = (0..rows as u32).collect();
    let CeMirror { out, dw, .. } = ce_rows_mirror(
        &plan,
        &h,
        &w,
        (&picked, targets),
        Operands::ExactF32,
        Some(1.0),
    )
    .unwrap();
    let dw = dw.unwrap();
    let mut grad = vec![0.0f32; rows * vocab];
    for r in 0..rows {
        for v in 0..vocab {
            grad[r * vocab + v] = dw[v * rows + r];
        }
    }
    (out.per_row, out.loss, grad)
}

fn assert_switched(what: &str, loss: f64, grad: &[f32], row: &[f32], target: usize) {
    let (want_loss, want_grad, s) = oracle(row, target);
    assert!(
        grad[target] < 0.0,
        "{what}: target gradient collapsed to {} (S {s:e})",
        grad[target]
    );
    let l = ulps(loss as f32, want_loss);
    assert!(
        l <= 16.0,
        "{what}: loss {loss:e} vs log1p(S) {want_loss:e} ({l} ulps)"
    );
    for (j, (&got, &want)) in grad.iter().zip(&want_grad).enumerate() {
        let u = ulps(got, want);
        assert!(
            u <= 16.0,
            "{what}: grad[{j}] {got:e} vs {want:e} ({u} ulps)"
        );
    }
    let sum: f64 = grad.iter().map(|g| f64::from(*g)).sum();
    let peak = grad.iter().fold(0.0f32, |m, g| m.max(g.abs()));
    let spacing = f64::from(f32::from_bits(peak.to_bits() + 1)) - f64::from(peak);
    let gate = 4.0 + (grad.len() as f64).sqrt();
    assert!(
        sum.abs() <= gate * spacing,
        "{what}: row sums to {sum:e}, more than {gate:.1} ulps of the peak {peak:e}"
    );
}

/// `[9, -9, -9]` at target 0, margin 18, in chunks of 2 (so the target's chunk
/// is the first and the tail is one wide) and of 3 (one chunk).
#[test]
fn a_row_past_the_fp32_cliff_keeps_a_nonzero_target_gradient() {
    let row = [9.0f32, -9.0, -9.0];
    for chunk in [2u64, 3] {
        let (per_row, loss, grad) = mirror(&row, 3, &[0], chunk, Reduction::Mean);
        assert_eq!(per_row[0], loss);
        assert_switched(&format!("chunk {chunk}"), loss, &grad, &row, 0);
    }
    // The independent f64 reference agrees with the oracle written here.
    let h = [1.0f64];
    let w: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
    let want = cross_entropy_rows_f64(&h, &w, &[0], &[0], 1, 3, RefReduction::Mean, 1.0);
    let (loss, grad, _) = oracle(&row, 0);
    assert!(
        (want.loss - loss).abs() <= 1e-15 * loss,
        "{} vs {loss}",
        want.loss
    );
    for (a, b) in want.dw.iter().zip(&grad) {
        assert!((a - b).abs() <= 1e-15 * b.abs(), "{a} vs {b}");
    }
}

const VOCAB: usize = 301;
const TARGETS: [usize; 8] = [0, 7, 8, 63, 150, 296, 299, 300];
const MARGINS: [f32; 14] = [
    14.0, 15.0, 16.0, 16.6, 17.0, 17.5, 18.0, 19.0, 20.0, 22.0, 25.0, 30.0, 36.0, 40.0,
];

/// Margins 14 to 40 at 301 classes, every target position, in chunks of 64
/// (the carried sum is rescaled across five chunk boundaries): a row whose sum
/// is certain to be 1.0 whatever the order (S below 2^-25) switches, keeps a
/// nonzero target gradient and meets the oracle. Rows above that bound stay on
/// the standard form, and on this kernel that form's target gradient is
/// already 0 for some of them well before the sum reaches 1.0: `lse = m +
/// ln(s)` is rounded at the spacing of `m` (about 2e-6 at a logit of 20), so
/// `ln(s)` below half of that is lost and `p_t` is exactly 1. The rule
/// (target at the maximum, `s == 1.0`) leaves those rows alone, so the test
/// asserts no gradient for them.
#[test]
fn the_margin_sweep_across_the_cliff_keeps_the_target_gradient() {
    let mut certain = 0;
    for &target in &TARGETS {
        for &margin in &MARGINS {
            let row = margin_row(VOCAB, target, margin);
            let t = [target as u32];
            let (_, _, s) = oracle(&row, target);
            let tag = format!("target {target} margin {margin} S {s:e}");
            let (per_row, _, grad) = mirror(&row, VOCAB, &t, 64, Reduction::Mean);
            if s < 2.0f64.powi(-25) {
                certain += 1;
                assert_switched(&tag, per_row[0], &grad, &row, target);
            }
        }
    }
    assert!(certain >= 40, "only {certain} rows were certain to switch");
}

/// Two-class rows straddling the cliff: past it (1 + S is 1.0) the target
/// gradient is the exact negation of the other class's, so the row sums to 0.
#[test]
fn two_class_rows_past_the_cliff_sum_to_exactly_zero() {
    for target in [0usize, 1] {
        for margin in 18..=40 {
            let mut row = [0.0f32; 2];
            row[target] = margin as f32;
            let (_, _, grad) = mirror(&row, 2, &[target as u32], 1, Reduction::Mean);
            assert_eq!(
                grad[target].to_bits(),
                (-grad[1 - target]).to_bits(),
                "target {target} margin {margin}: {grad:?}"
            );
            assert!(grad[target] < 0.0);
        }
    }
}

/// A one-class row and a row whose other classes underflow keep the standard
/// form's +0.0.
#[test]
fn one_class_and_underflowed_rows_keep_positive_zero() {
    let (per_row, loss, grad) = mirror(&[5.0], 1, &[0], 1, Reduction::Mean);
    assert_eq!((per_row[0].to_bits(), loss.to_bits()), (0, 0));
    assert!(grad.iter().all(|g| g.to_bits() == 0), "{grad:?}");
    let (per_row, _, grad) = mirror(&[200.0, 0.0, -1.0, 3.0], 4, &[0], 3, Reduction::Mean);
    assert_eq!(per_row[0].to_bits(), 0);
    assert!(grad.iter().all(|g| g.to_bits() == 0), "{grad:?}");
}

/// Rows the standard form leaves alone: vocabularies 3, 40 and 301 at margins
/// to 13 (301 only where the sum is above 1.0 in any order), ties at the
/// maximum, and targets far below it. The digests are of the per-row losses,
/// the reduced loss and the logit gradient, recorded by running this test at
/// af8bd9c. The gradient comes from the host twins of `exp` and `log`, which
/// are bit-reproducible, so it holds on every platform; the losses are formed
/// in f64 with the platform's `ln`, so they are asserted on Apple silicon only,
/// where they were recorded (as `exact_golden`'s `ce_loss`).
fn ordinary_batches() -> Vec<(usize, Vec<f32>, Vec<u32>)> {
    let mut batches = Vec::new();
    for vocab in [3usize, 40] {
        let mut flat = Vec::new();
        let mut targets = Vec::new();
        for &target in &[0usize, 1, vocab - 1] {
            for &margin in &[-6.0f32, -1.0, 0.0, 3.0, 9.0, 13.0] {
                flat.extend(margin_row(vocab, target, margin));
                targets.push(target as u32);
            }
        }
        batches.push((vocab, flat, targets));
    }
    let mut flat = Vec::new();
    let mut targets = Vec::new();
    for &target in &TARGETS {
        for &margin in &[-3.0f32, 0.0, 6.0, 10.0, 12.0, 13.0] {
            let row = margin_row(VOCAB, target, margin);
            let (_, _, s) = oracle(&row, target);
            assert!(
                s > VOCAB as f64 * 2.0f64.powi(-23),
                "margin {margin} is near the cliff"
            );
            flat.extend(row);
            targets.push(target as u32);
        }
    }
    batches.push((VOCAB, flat, targets));
    let mut flat = vec![9.0f32, 9.0, -30.0, 9.0, -30.0, 9.0, 9.0, 9.0, 9.0];
    let mut targets = vec![0u32, 0, 2];
    flat.extend([30.0f32, -30.0, 5.0, -2.0, -40.0, 40.0]);
    targets.extend([2u32, 0]);
    batches.push((3, flat, targets));
    batches
}

/// `[per-row loss, reduced loss, gradient]` per batch, chunked at 7.
const ORDINARY: [[u64; 3]; 4] = [
    [0x7e8625e0aec5a0d4, 0xf75d01a8b0a20ab6, 0x9e54dc87a6e825c1],
    [0x7e7f57d0e046218a, 0x5832bfb0c5a66934, 0x05a2fa6cc80a801e],
    [0x060e85aff49d23a0, 0x919df8cb664f657c, 0xd5fd1f010d5d52c0],
    [0x3ecc0fbc39a8eea0, 0x45f8c073fe71053b, 0x4dd13c3cccc36cba],
];

#[test]
fn ordinary_rows_are_bit_identical_to_the_commit_before() {
    let recorded_here = cfg!(all(target_os = "macos", target_arch = "aarch64"));
    let f64s = |v: &[f64]| {
        fnv(v
            .iter()
            .flat_map(|x| [x.to_bits() as u32, (x.to_bits() >> 32) as u32]))
    };
    let seen: Vec<[u64; 3]> = ordinary_batches()
        .into_iter()
        .map(|(vocab, flat, targets)| {
            let (per_row, loss, grad) = mirror(&flat, vocab, &targets, 7, Reduction::Mean);
            [
                f64s(&per_row),
                f64s(&[loss]),
                fnv(grad.iter().map(|g| g.to_bits())),
            ]
        })
        .collect();
    for (b, got) in seen.iter().enumerate() {
        eprintln!("ORDINARY batch {b} {got:#018x?}");
    }
    for (b, got) in seen.iter().enumerate() {
        assert_eq!(got[2], ORDINARY[b][2], "batch {b}: gradient digest");
        if recorded_here {
            assert_eq!(
                got[..2],
                ORDINARY[b][..2],
                "batch {b}: per-row loss and loss digests"
            );
        }
    }
}
