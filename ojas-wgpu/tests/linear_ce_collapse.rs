//! Softmax Collapse through `Backend::linear_cross_entropy_mean` on wgpu
//! (`lce_stats`, `lce_finish`, `lce_grad`; ft-f40eeed8), on a real adapter.
//!
//! A row keeps the standard form bit for bit unless its target logit is the
//! row maximum and the f32 sum of `e^(z - max)` is exactly 1.0; only those rows
//! take the target-excluded form (S = sum over j != t of e^(z_j - z_t), loss
//! log1p(S), target gradient -S / (1 + S) / valid, others e^(z_j - z_t) / (1 +
//! S) / valid). The logits tile is `x · Wᵀ` with `x` the identity and `W` the
//! transposed logits, so every logit is exact and `grad_weight[v][r]` is the
//! logit gradient `G[r][v]`. The oracle is f64 and forms no `p - 1`; rows are
//! held to it only where the f32 sum is certain to be 1.0 whatever the order
//! (S below 2^-25).
//!
//! WGSL's `exp` is only specified to 3 + 2|x| ulps, so a row's gate is that
//! bound at the largest exponent argument it uses, as in `parity.rs`.

mod common;

use common::*;
use ojas_core::{Backend, CeChunk, CeGrads, Tensor};

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

/// The ulp gate of a row: WGSL's `exp` bound at the largest argument, plus the
/// division and rounding.
fn gate_of(row: &[f32], target: usize) -> f64 {
    let zt = row[target];
    let worst = row
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != target)
        .fold(0.0f32, |m, (_, &v)| m.max((v - zt).abs()));
    4.0 + 2.0 * f64::from(worst)
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

/// Loss and `[rows, vocab]` logit gradient of the fused op, the vocabulary cut
/// into tiles of `cols` and the rows into tiles of `row_tile`.
fn lce(
    logits: &[f32],
    vocab: usize,
    targets: &[u32],
    ignore: Option<u32>,
    (row_tile, cols): (usize, usize),
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
    let t =
        |data: &[f32], shape: &[usize]| up(&Tensor::from_f32(data, shape, host_budget()).unwrap());
    let out = gpu()
        .linear_cross_entropy_mean(
            &t(&x, &[rows, rows]),
            &t(&w, &[vocab, rows]),
            &up(&host_u32(targets, &[rows])),
            ignore,
            CeChunk {
                rows: row_tile,
                cols,
            },
            CeGrads {
                input: false,
                weight: true,
            },
        )
        .unwrap();
    let loss = down(&out.loss)[0];
    let gw = down(out.grad_weight.as_ref().expect("grad_weight"));
    let mut grad = vec![0.0f32; rows * vocab];
    for r in 0..rows {
        for v in 0..vocab {
            grad[r * vocab + v] = gw[v * rows + r];
        }
    }
    (loss, grad)
}

fn assert_row(what: &str, g: &[f32], want: &[f64], gate: f64) {
    for (j, (&got, &w)) in g.iter().zip(want).enumerate() {
        let u = ulps(got, w);
        assert!(
            u <= gate,
            "{what}: grad[{j}] {got:e} vs {w:e} ({u} ulps, gate {gate})"
        );
    }
    let sum: f64 = g.iter().map(|x| f64::from(*x)).sum();
    let peak = g.iter().fold(0.0f32, |a, x| a.max(x.abs()));
    let spacing = f64::from(f32::from_bits(peak.to_bits() + 1)) - f64::from(peak);
    let sum_gate = gate + (g.len() as f64).sqrt();
    assert!(
        sum.abs() <= sum_gate * spacing,
        "{what}: row sums to {sum:e}, more than {sum_gate:.1} ulps of the peak {peak:e}"
    );
}

/// `[9, -9, -9]` at target 0, margin 18, with an ignored row after it so the
/// mean divisor is 1; the vocabulary in one tile and in tiles of 2 (the target
/// in the first tile, the other classes in later ones).
#[test]
fn a_row_past_the_fp32_cliff_keeps_a_nonzero_target_gradient() {
    let row = [9.0f32, -9.0, -9.0];
    let (_, want, s) = oracle(&row, 0, 1.0);
    let gate = gate_of(&row, 0);
    for cols in [2usize, 3] {
        let (loss, grad) = lce(
            &[9.0, -9.0, -9.0, 1.0, 2.0, 3.0],
            3,
            &[0, 99],
            Some(99),
            (2, cols),
        );
        assert!(
            grad[0] < 0.0,
            "cols {cols}: target gradient collapsed to {} (S {s:e})",
            grad[0]
        );
        let l = ulps(loss, s.ln_1p());
        assert!(
            l <= gate,
            "cols {cols}: loss {loss:e} ({l} ulps, gate {gate})"
        );
        assert_row(&format!("cols {cols}"), &grad[..3], &want, gate);
        assert!(grad[3..].iter().all(|g| *g == 0.0), "ignored row {grad:?}");
    }
}

const VOCAB: usize = 301;
const TARGETS: [usize; 8] = [0, 7, 8, 63, 150, 296, 299, 300];
const MARGINS: [f32; 7] = [16.0, 17.0, 18.0, 20.0, 25.0, 30.0, 40.0];

/// Margins 16 to 40 at 301 classes, the target in lane 0, lane 7, at a chunk
/// edge, mid-row and in the tail, the vocabulary in tiles of 64 and of 301 and
/// the rows in tiles of 8: each row whose sum is certain to be 1.0 meets the
/// f64 oracle at its gate, and a batch of those rows alone has the mean of
/// their log1p(S) as its loss.
#[test]
fn the_margin_sweep_across_the_cliff_keeps_the_target_gradient() {
    let mut flat = Vec::new();
    let mut targets = Vec::new();
    let mut rows = Vec::new();
    for &t in &TARGETS {
        for &margin in &MARGINS {
            let row = margin_row(VOCAB, t, margin);
            flat.extend_from_slice(&row);
            targets.push(t as u32);
            rows.push((t, row, margin));
        }
    }
    let denom = rows.len() as f64;
    for cols in [64usize, VOCAB] {
        let (_, grad) = lce(&flat, VOCAB, &targets, None, (8, cols));
        let mut certain = 0;
        for (r, (t, row, margin)) in rows.iter().enumerate() {
            let (_, want, s) = oracle(row, *t, denom);
            if s < 2.0f64.powi(-25) {
                certain += 1;
                let g = &grad[r * VOCAB..(r + 1) * VOCAB];
                assert!(
                    g[*t] < 0.0,
                    "cols {cols} target {t} margin {margin}: {}",
                    g[*t]
                );
                assert_row(
                    &format!("cols {cols} target {t} margin {margin}"),
                    g,
                    &want,
                    gate_of(row, *t),
                );
            }
        }
        assert!(certain >= 24, "only {certain} rows were certain to switch");
    }
    let certain: Vec<usize> = (0..rows.len())
        .filter(|&r| oracle(&rows[r].1, rows[r].0, 1.0).2 < 2.0f64.powi(-25))
        .collect();
    let flat: Vec<f32> = certain.iter().flat_map(|&r| rows[r].1.clone()).collect();
    let targets: Vec<u32> = certain.iter().map(|&r| rows[r].0 as u32).collect();
    let (loss, _) = lce(&flat, VOCAB, &targets, None, (8, 64));
    let want = certain
        .iter()
        .map(|&r| oracle(&rows[r].1, rows[r].0, 1.0).0)
        .sum::<f64>()
        / certain.len() as f64;
    let gate = certain
        .iter()
        .map(|&r| gate_of(&rows[r].1, rows[r].0))
        .fold(0.0f64, f64::max);
    let u = ulps(loss, want);
    assert!(
        u <= gate,
        "mean loss {loss:e} vs {want:e} ({u} ulps, gate {gate})"
    );
}

/// Two-class rows straddling the cliff against the CPU Exact tier. Past the
/// cliff the target gradient is the exact negation of the other class's and
/// within WGSL's exp bound of the CPU's; where the device's `exp` happens to
/// match the CPU's bit for bit the sums agree and every gradient bit must.
#[test]
fn two_class_rows_straddling_the_cliff_agree_with_the_cpu_where_the_sums_agree() {
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
    let rows = keys.len();
    let (_, gpu_grad) = lce(&flat, 2, &targets, None, (rows, 1));
    let cpu_grad = cpu()
        .cross_entropy_mean_backward(
            &Tensor::from_f32(&flat, &[rows, 2], host_budget()).unwrap(),
            &host_u32(&targets, &[rows]),
            None,
        )
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let mut agreed = 0;
    for (r, &(target, margin)) in keys.iter().enumerate() {
        let other = 1 - target;
        let (g, c) = (&gpu_grad[2 * r..2 * r + 2], &cpu_grad[2 * r..2 * r + 2]);
        if g[other].to_bits() == c[other].to_bits() {
            agreed += 1;
            assert_eq!(
                g[target].to_bits(),
                c[target].to_bits(),
                "target {target} margin {margin}: the sums agree, the target gradient does not ({g:?} vs {c:?})"
            );
        }
        if margin >= 18 {
            assert!(g[target] < 0.0, "target {target} margin {margin}: {g:?}");
            assert_eq!(
                g[target].to_bits(),
                (-g[other]).to_bits(),
                "target {target} margin {margin}: the row does not sum to 0 ({g:?})"
            );
            let u = ulps(g[target], f64::from(c[target]));
            let gate = 4.0 + 2.0 * f64::from(margin);
            assert!(
                u <= gate,
                "target {target} margin {margin}: {} vs cpu {} ({u} ulps, gate {gate})",
                g[target],
                c[target]
            );
        }
    }
    eprintln!(
        "exps agreed with the CPU on {agreed} of {rows} rows on {}",
        gpu().context().adapter_name()
    );
}

/// A one-class row has no other class: loss and gradient are +0.0.
#[test]
fn a_one_class_row_keeps_positive_zero() {
    let (loss, grad) = lce(&[5.0, -3.0, 0.0], 1, &[0, 0, 0], None, (3, 1));
    assert_eq!(loss.to_bits(), 0);
    assert!(grad.iter().all(|g| g.to_bits() == 0), "{grad:?}");
}

/// A batch the digests cover: vocabulary, flat logits, targets, ignored id.
type Batch = (usize, Vec<f32>, Vec<u32>, Option<u32>);

/// Rows the standard form leaves alone (none has its target at the maximum
/// with a sum of 1.0, in any summation order): ties at the maximum, a target
/// far below it, and moderate margins at vocabularies 3 and 40, and margins to
/// 13 at 301 classes. Their loss and gradient bits equal the digests recorded
/// from af8bd9c on `RECORDED_ADAPTER`; on another adapter the bits are not
/// asserted and the test says so.
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
    for &target in &TARGETS {
        for &margin in &[-3.0f32, 0.0, 6.0, 10.0, 12.0, 13.0] {
            let row = margin_row(VOCAB, target, margin);
            let (_, _, s) = oracle(&row, target, 1.0);
            assert!(
                s > VOCAB as f64 * 2.0f64.powi(-23),
                "margin {margin} is near the cliff"
            );
            flat.extend(row);
            targets.push(target as u32);
        }
    }
    batches.push((VOCAB, flat, targets, None));
    let mut flat = vec![9.0f32, 9.0, -30.0, 9.0, -30.0, 9.0, 9.0, 9.0, 9.0];
    let mut targets = vec![0u32, 0, 2];
    flat.extend([30.0f32, -30.0, 5.0, -2.0, -40.0, 40.0]);
    targets.extend([2u32, 0]);
    flat.extend([1.0f32, 2.0, 3.0, 50.0, -50.0, 0.0]);
    targets.extend([2u32, 7]);
    batches.push((3, flat, targets, Some(7)));
    batches
}

/// The adapter the digests were recorded on, and the digests: `[loss, gradient]`
/// per batch.
const RECORDED_ADAPTER: &str = "Apple M5 Pro";
const ORDINARY: [[u64; 2]; 4] = [
    [0x81bc81cabd88d95f, 0x2515eec5c2e0dbd6],
    [0xb2e5c827cac86aac, 0xe597039afd2e392d],
    [0x60fb318f4316eb19, 0x398f62b537beca7a],
    [0xc1fe72ea5afff431, 0x6971a83da55a9b81],
];

#[test]
fn ordinary_rows_are_bit_identical_to_the_commit_before() {
    let name = gpu().context().adapter_name().to_string();
    let mut seen = Vec::new();
    for (vocab, flat, targets, ignore) in ordinary_batches() {
        let rows = targets.len();
        let (loss, grad) = lce(
            &flat,
            vocab,
            &targets,
            ignore,
            (rows.min(8), vocab.clamp(1, 7)),
        );
        seen.push([fnv([loss.to_bits()]), fnv(grad.iter().map(|g| g.to_bits()))]);
    }
    for (b, got) in seen.iter().enumerate() {
        eprintln!("ORDINARY adapter {name:?} batch {b} {got:#018x?}");
    }
    if name == RECORDED_ADAPTER {
        for (b, got) in seen.iter().enumerate() {
            assert_eq!(*got, ORDINARY[b], "batch {b}: loss and gradient digests");
        }
    } else {
        eprintln!("ordinary-rows digests not asserted: recorded on {RECORDED_ADAPTER:?}, running on {name:?}");
    }
}
