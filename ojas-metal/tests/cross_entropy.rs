//! Cross-entropy forward and backward against the CPU reference at the
//! paired bench's vocabulary (50,304), with ignored and sharp rows, and its
//! error contract: a NaN or infinity anywhere in the logits, including a row
//! whose target is ignored, is refused.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Backend, OjasError, Tensor};

const V: usize = 50304;

fn targets(rows: usize, seed: u64) -> Vec<u32> {
    ids(rows, seed, V as u32)
}

#[test]
fn matches_cpu_at_the_bench_vocabulary() {
    let (m, c) = (metal(), cpu());
    let rows = 384;
    for (scale, ignore) in [(4.0f32, None), (4.0, Some(7u32)), (80.0, None)] {
        let logits = rand(&[rows, V], 121 + scale as u64, scale);
        let mut t = targets(rows, 122);
        if let Some(ig) = ignore {
            for i in (0..rows).step_by(5) {
                t[i] = ig;
            }
        }
        let ht = host_u32(&t, &[rows]);
        let (dl, dt) = (up(&m, &logits), up(&m, &ht));
        let tag = format!("ce scale {scale} ignore {ignore:?}");
        same_tensor(
            &format!("{tag} loss"),
            &ok(&tag, m.cross_entropy_mean_forward(&dl, &dt, ignore)),
            &ok(&tag, c.cross_entropy_mean_forward(&logits, &ht, ignore)),
            1e-5,
            1e-5,
        );
        same_tensor(
            &format!("{tag} grad"),
            &ok(&tag, m.cross_entropy_mean_backward(&dl, &dt, ignore)),
            &ok(&tag, c.cross_entropy_mean_backward(&logits, &ht, ignore)),
            1e-9,
            1e-4,
        );
    }
}

#[test]
fn is_deterministic() {
    let m = metal();
    let rows = 64;
    let dl = up(&m, &rand(&[rows, V], 5, 4.0));
    let dt = up(&m, &host_u32(&targets(rows, 6), &[rows]));
    let bits = |t: &Tensor| -> Vec<u32> { down(t).iter().map(|v| v.to_bits()).collect() };
    let l0 = bits(&ok("fwd", m.cross_entropy_mean_forward(&dl, &dt, None)));
    let g0 = bits(&ok("bwd", m.cross_entropy_mean_backward(&dl, &dt, None)));
    for _ in 0..3 {
        assert_eq!(
            bits(&ok("fwd", m.cross_entropy_mean_forward(&dl, &dt, None))),
            l0
        );
        assert_eq!(
            bits(&ok("bwd", m.cross_entropy_mean_backward(&dl, &dt, None))),
            g0
        );
    }
}

#[test]
fn a_non_finite_logit_anywhere_is_refused_even_in_an_ignored_row() {
    let m = metal();
    let rows = 6;
    let ignore = 3u32;
    let mut t = targets(rows, 9);
    t[2] = ignore;
    let dt = up(&m, &host_u32(&t, &[rows]));
    // (row, column): inside a counted row, the last element of the last row,
    // and inside the ignored row 2.
    for (row, col) in [(0usize, 17usize), (rows - 1, V - 1), (2, 40000)] {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut l = values(rows * V, 10, 4.0);
            l[row * V + col] = bad;
            let dl = up(&m, &host(&l, &[rows, V]));
            let f = m.cross_entropy_mean_forward(&dl, &dt, Some(ignore));
            deferred(
                &m,
                &format!("fwd ({row}, {col}) = {bad}"),
                f,
                "cross_entropy_mean_forward",
            );
            let g = m.cross_entropy_mean_backward(&dl, &dt, Some(ignore));
            deferred(
                &m,
                &format!("bwd ({row}, {col}) = {bad}"),
                g,
                "cross_entropy_mean_backward",
            );
        }
    }
}

/// The targets are checked on the host, so an out-of-range target is
/// refused at the call even when a logit is non-finite (the precedence the
/// deferred-fault contract flips); the call records nothing, so nothing is
/// pending afterwards.
#[test]
fn an_out_of_range_target_is_refused_at_once_even_beside_a_non_finite_logit() {
    let m = metal();
    let mut l = values(2 * V, 11, 1.0);
    l[V + 3] = f32::NAN;
    let dl = up(&m, &host(&l, &[2, V]));
    let dt = up(&m, &host_u32(&[1, V as u32 + 5], &[2]));
    let g = m.cross_entropy_mean_backward(&dl, &dt, None);
    assert!(matches!(g, Err(OjasError::OutOfRange { .. })), "{g:?}");
    assert!(m.sync().is_ok(), "the refused call must record nothing");
    let finite = up(&m, &rand(&[2, V], 12, 1.0));
    let g = m.cross_entropy_mean_backward(&finite, &dt, None);
    assert!(matches!(g, Err(OjasError::OutOfRange { .. })), "{g:?}");
}

// ---- Softmax Collapse (ft-f40eeed8) -------------------------------------
//
// A row keeps the standard form bit for bit unless its target logit is the
// row maximum and the f32 sum of `e^(z - max)` is exactly 1.0; only those
// rows take the target-excluded form (S = sum over j != t of e^(z_j - z_t),
// loss log1p(S), target gradient -S / (1 + S) / valid, others e^(z_j - z_t) /
// (1 + S) / valid). The oracle below is f64, forms no `p - 1`, and its
// summation order is not the kernel's, so rows are held to it only where the
// f32 sum is certain to be 1.0 whatever the order (S below 2^-25).

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

/// `(loss, gradient, S)` of one row at divisor `denom`, in f64, from the
/// target-excluded expression.
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

/// A row of `vocab` classes in `[-3, 3)` on a 2^-12 grid, with the target
/// `margin` above the largest other logit.
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

/// Loss and `[rows, vocab]` gradient on the device.
fn device_ce(
    m: &ojas_metal::MetalBackend,
    logits: &[f32],
    vocab: usize,
    targets: &[u32],
    ignore: Option<u32>,
) -> (f32, Vec<f32>) {
    let rows = targets.len();
    let dl = up(m, &host(logits, &[rows, vocab]));
    let dt = up(m, &host_u32(targets, &[rows]));
    let loss = down(&ok(
        "ce forward",
        m.cross_entropy_mean_forward(&dl, &dt, ignore),
    ))[0];
    let grad = down(&ok(
        "ce backward",
        m.cross_entropy_mean_backward(&dl, &dt, ignore),
    ));
    (loss, grad)
}

/// Loss and gradient on the CPU Exact reference.
fn cpu_ce(logits: &[f32], vocab: usize, targets: &[u32], ignore: Option<u32>) -> (f32, Vec<f32>) {
    let c = cpu();
    let rows = targets.len();
    let lt = host(logits, &[rows, vocab]);
    let tt = host_u32(targets, &[rows]);
    let loss = ok(
        "cpu forward",
        c.cross_entropy_mean_forward(&lt, &tt, ignore),
    )
    .to_f32_vec()
    .unwrap()[0];
    let grad = ok(
        "cpu backward",
        c.cross_entropy_mean_backward(&lt, &tt, ignore),
    )
    .to_f32_vec()
    .unwrap();
    (loss, grad)
}

/// One row meets the oracle: nonzero target gradient, the row summing to 0
/// within a few ulps of its peak, loss and gradients within `gate` ulps.
fn assert_switched(
    what: &str,
    loss: f32,
    grad: &[f32],
    row: &[f32],
    target: usize,
    denom: f64,
    gate: f64,
) {
    let (want_loss, want_grad, s) = oracle(row, target, denom);
    assert!(
        grad[target] < 0.0,
        "{what}: target gradient collapsed to {} (S {s:e})",
        grad[target]
    );
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
    let sum_gate = 4.0 + (grad.len() as f64).sqrt();
    assert!(
        sum.abs() <= sum_gate * spacing,
        "{what}: row sums to {sum:e}, more than {sum_gate:.1} ulps of the peak {peak:e}"
    );
}

/// `[9, -9, -9]` at target 0, margin 18, with an ignored row after it so the
/// mean divisor is 1.
#[test]
fn a_row_past_the_fp32_cliff_keeps_a_nonzero_target_gradient() {
    let m = metal();
    let row = [9.0f32, -9.0, -9.0];
    let (loss, grad) = device_ce(&m, &[9.0, -9.0, -9.0, 1.0, 2.0, 3.0], 3, &[0, 99], Some(99));
    assert_switched("cliff", loss, &grad[..3], &row, 0, 1.0, 16.0);
    assert!(grad[3..].iter().all(|g| *g == 0.0), "ignored row {grad:?}");
}

const VOCAB_SWEEP: usize = 301;
const SWEEP_TARGETS: [usize; 8] = [0, 7, 8, 63, 150, 296, 299, 300];
const SWEEP_MARGINS: [f32; 7] = [16.0, 17.0, 18.0, 20.0, 25.0, 30.0, 40.0];

/// Margins 16 to 40 at a 301-class vocabulary with the target at lane 0,
/// lane 7, a chunk edge, mid-row, and in the tail: every row keeps a nonzero
/// target gradient, and each row whose sum is certain to be 1.0 meets the
/// f64 oracle.
#[test]
fn the_margin_sweep_across_the_cliff_keeps_the_target_gradient() {
    let m = metal();
    let mut flat = Vec::new();
    let mut targets = Vec::new();
    let mut rows = Vec::new();
    for &t in &SWEEP_TARGETS {
        for &margin in &SWEEP_MARGINS {
            let row = margin_row(VOCAB_SWEEP, t, margin);
            flat.extend_from_slice(&row);
            targets.push(t as u32);
            rows.push((t, row, margin));
        }
    }
    let denom = rows.len() as f64;
    let (_, grad) = device_ce(&m, &flat, VOCAB_SWEEP, &targets, None);
    let mut certain = Vec::new();
    for (r, (t, row, margin)) in rows.iter().enumerate() {
        let g = &grad[r * VOCAB_SWEEP..(r + 1) * VOCAB_SWEEP];
        let (_, want, s) = oracle(row, *t, denom);
        assert!(
            g[*t] < 0.0,
            "target {t} margin {margin}: target gradient {}",
            g[*t]
        );
        if s < 2.0f64.powi(-25) {
            certain.push(r);
            for (j, (&got, &w)) in g.iter().zip(&want).enumerate() {
                let u = ulps(got, w);
                assert!(
                    u <= 16.0,
                    "target {t} margin {margin}: grad[{j}] {got:e} vs {w:e} ({u} ulps)"
                );
            }
            let sum: f64 = g.iter().map(|x| f64::from(*x)).sum();
            let peak = g.iter().fold(0.0f32, |a, x| a.max(x.abs()));
            let spacing = f64::from(f32::from_bits(peak.to_bits() + 1)) - f64::from(peak);
            let gate = 4.0 + (g.len() as f64).sqrt();
            assert!(
                sum.abs() <= gate * spacing,
                "target {t} margin {margin}: row sums to {sum:e}, peak {peak:e}"
            );
        }
    }
    assert!(
        certain.len() >= 24,
        "only {} rows were certain to switch",
        certain.len()
    );
    // The mean loss of a batch of switched rows alone is the mean of their
    // log1p(S): a mixed batch would carry the standard form's rounding of the
    // rows near the cliff.
    let flat: Vec<f32> = certain.iter().flat_map(|&r| rows[r].1.clone()).collect();
    let targets: Vec<u32> = certain.iter().map(|&r| rows[r].0 as u32).collect();
    let (loss, _) = device_ce(&m, &flat, VOCAB_SWEEP, &targets, None);
    let want: f64 = certain
        .iter()
        .map(|&r| oracle(&rows[r].1, rows[r].0, 1.0).0)
        .sum::<f64>()
        / certain.len() as f64;
    let u = ulps(loss, want);
    assert!(u <= 16.0, "mean loss {loss:e} vs {want:e} ({u} ulps)");
}

/// Two-class rows straddling the cliff, target at either lane, margins 14 to
/// 40, against the CPU Exact tier. The one other class makes `S = e^(z_o -
/// z_t)` a single `exp`, so wherever the device's `exp` and the CPU's agree
/// the sums agree, and then every gradient bit must agree, the target's
/// included. Past the cliff the target gradient is the exact negation of the
/// other class's on any backend (`1 + S` is 1.0), so the row sums to 0.
#[test]
fn two_class_rows_straddling_the_cliff_agree_with_the_cpu_where_the_sums_agree() {
    let m = metal();
    let mut flat = Vec::new();
    let mut targets = Vec::new();
    let mut keys = Vec::new();
    for target in [0usize, 1] {
        for margin in 14..=40 {
            let mut row = [0.0f32; 2];
            row[target] = margin as f32;
            flat.extend_from_slice(&row);
            targets.push(target as u32);
            keys.push((target, margin));
        }
    }
    let (_, gpu_grad) = device_ce(&m, &flat, 2, &targets, None);
    let (_, cpu_grad) = cpu_ce(&flat, 2, &targets, None);
    let mut agreed = 0;
    for (r, &(target, margin)) in keys.iter().enumerate() {
        let other = 1 - target;
        let (g, c) = (&gpu_grad[2 * r..2 * r + 2], &cpu_grad[2 * r..2 * r + 2]);
        assert!(g[target] < 0.0, "target {target} margin {margin}: {g:?}");
        if g[other].to_bits() == c[other].to_bits() {
            agreed += 1;
            assert_eq!(
                g[target].to_bits(),
                c[target].to_bits(),
                "target {target} margin {margin}: the sums agree, the target gradient does not ({g:?} vs {c:?})"
            );
        }
        if margin >= 18 {
            assert_eq!(
                g[target].to_bits(),
                (-g[other]).to_bits(),
                "target {target} margin {margin}: the row does not sum to 0 ({g:?})"
            );
        }
    }
    assert!(
        agreed >= keys.len() / 2,
        "the exps agreed on only {agreed} of {} rows",
        keys.len()
    );
}

/// A one-class row has no other class: loss and gradient are +0.0, as the
/// standard form's `x - x` gives.
#[test]
fn a_one_class_row_keeps_positive_zero() {
    let m = metal();
    let (loss, grad) = device_ce(&m, &[5.0, -3.0, 0.0], 1, &[0, 0, 0], None);
    assert_eq!(loss.to_bits(), 0);
    assert!(grad.iter().all(|g| g.to_bits() == 0), "{grad:?}");
}

/// A batch the digests cover: vocabulary, flat logits, targets and the ignored id.
type Batch = (usize, Vec<f32>, Vec<u32>, Option<u32>);

/// Rows the standard form leaves alone (none has its target at the maximum
/// with a sum of 1.0, in any summation order): ties at the maximum, a target
/// far below it, and moderate margins at vocabularies 3 and 40, and margins
/// to 13 at 301 classes. Their loss and gradient bits equal the digests
/// recorded from af8bd9c on `RECORDED_DEVICE`; on another device the bits
/// are not asserted and the test says so.
fn ordinary_batches() -> Vec<Batch> {
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
        batches.push((vocab, flat, targets, None));
    }
    let mut flat = Vec::new();
    let mut targets = Vec::new();
    for &target in &SWEEP_TARGETS {
        for &margin in &[-3.0f32, 0.0, 6.0, 10.0, 12.0, 13.0] {
            let row = margin_row(VOCAB_SWEEP, target, margin);
            let (_, _, s) = oracle(&row, target, 1.0);
            assert!(
                s > VOCAB_SWEEP as f64 * 2.0f64.powi(-23),
                "margin {margin} is near the cliff"
            );
            flat.extend(row);
            targets.push(target as u32);
        }
    }
    batches.push((VOCAB_SWEEP, flat, targets, None));
    let mut flat = vec![9.0f32, 9.0, -30.0, 9.0, -30.0, 9.0, 9.0, 9.0, 9.0];
    let mut targets = vec![0u32, 0, 2];
    flat.extend([30.0f32, -30.0, 5.0, -2.0, -40.0, 40.0]);
    targets.extend([1u32, 0]);
    flat.extend([1.0f32, 2.0, 3.0, 50.0, -50.0, 0.0]);
    targets.extend([2u32, 7]);
    batches.push((3, flat, targets, Some(7)));
    batches
}

/// The device the digests below were recorded on, and the digests: `[loss,
/// gradient]` per batch.
const RECORDED_DEVICE: &str = "Apple M5 Pro";
const ORDINARY: [[u64; 2]; 4] = [
    [0x81bc81cabd88d95f, 0xf714c455f8f566b9],
    [0xb2e5c827cac86aac, 0xe5fd5ad160807db3],
    [0x60fb318f4316eb19, 0xb305cb252551bfcd],
    [0x106c70c20c0f4647, 0xe92efbfc204465c9],
];

#[test]
fn ordinary_rows_are_bit_identical_to_the_commit_before() {
    let m = metal();
    let name = m.device_name().to_string();
    for (b, (vocab, flat, targets, ignore)) in ordinary_batches().into_iter().enumerate() {
        let (loss, grad) = device_ce(&m, &flat, vocab, &targets, ignore);
        let got = [fnv([loss.to_bits()]), fnv(grad.iter().map(|g| g.to_bits()))];
        eprintln!("ORDINARY device {name:?} batch {b} {got:#018x?}");
        if name == RECORDED_DEVICE {
            assert_eq!(got, ORDINARY[b], "batch {b}: loss and gradient digests");
        } else {
            eprintln!("ordinary-rows digests not asserted: recorded on {RECORDED_DEVICE:?}, running on {name:?}");
        }
    }
}
