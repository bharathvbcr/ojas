//! Shape first for the heavy ops: embedding forward and backward,
//! cross-entropy forward and backward, `clip_grad_norm`, `adamw_step` and
//! `muon_ns5_step` (`docs/shape-contract.md`).
//!
//! A malformed call is refused by its `ojas_core` shape validator before the
//! backend charges anything, scans for NaN or reads an id, so it returns the
//! validator's variant, op and detail whatever the budget holds:
//! - under `Budget::new(0)`;
//! - under a cap that holds every input's bytes but not the output;
//! - with a NaN present in an operand (D3);
//! - for the embedding backward, with an out-of-range id as well (D11);
//! - with two mistyped operands, the first in argument order is the one
//!   named (D5, D6).
//!
//! Inputs live on their own unbounded budget, so the backend's budget sees
//! only what the op charges. Every cell runs and is printed before the test
//! asserts, so a failure lists each cell that broke.

use ojas_core::{
    adamw_step_dims, clip_grad_norm_dims, cross_entropy_mean_backward_dims,
    cross_entropy_mean_forward_dims, embedding_backward_dims, embedding_forward_dims,
    muon_ns5_step_dims, AdamWConfig, Backend, Budget, MuonNs5Config, OjasError, Tensor,
};
use ojas_cpu::CpuBackend;

fn f(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn u(data: &[u32], shape: &[usize]) -> Tensor {
    Tensor::from_u32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn fill(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 7) as f32 * 0.25 - 0.5).collect()
}

fn ft(shape: &[usize]) -> Tensor {
    f(&fill(shape.iter().product()), shape)
}

/// `shape` filled, with a NaN at element 0.
fn nan(shape: &[usize]) -> Tensor {
    let mut v = fill(shape.iter().product());
    v[0] = f32::NAN;
    f(&v, shape)
}

fn bytes(t: &[&Tensor]) -> u64 {
    t.iter()
        .map(|t| (t.num_elements().unwrap() * t.dtype().size()) as u64)
        .sum()
}

/// One op call on a backend.
type Call = Box<dyn Fn(&CpuBackend) -> Result<(), OjasError>>;

struct Cell {
    name: String,
    cap: u64,
    call: Call,
    /// `Debug` of the validator's refusal.
    want: String,
}

fn cell(
    name: impl Into<String>,
    cap: u64,
    want: &str,
    call: impl Fn(&CpuBackend) -> Result<(), OjasError> + 'static,
) -> Cell {
    Cell {
        name: name.into(),
        cap,
        call: Box::new(call),
        want: want.to_string(),
    }
}

/// The validator's refusal, the oracle each cell must return.
fn want<T: std::fmt::Debug>(r: Result<T, OjasError>) -> String {
    match r {
        Err(e) => format!("{e:?}"),
        Ok(v) => panic!("the validator accepted a malformed call: {v:?}"),
    }
}

/// The three budget cells (zero, input bytes only, unbounded) for one
/// malformed call.
fn budgets(
    out: &mut Vec<Cell>,
    name: &str,
    inputs: u64,
    want: &str,
    call: impl Fn(&CpuBackend) -> Result<(), OjasError> + Clone + 'static,
) {
    for (label, cap) in [
        ("budget 0", 0),
        ("cap = input bytes", inputs),
        ("unbounded", u64::MAX),
    ] {
        out.push(cell(format!("{name}: {label}"), cap, want, call.clone()));
    }
}

fn cells() -> Vec<Cell> {
    let mut out = Vec::new();

    // Embedding forward: a rank-3 table.
    {
        let (table, ids) = (ft(&[4, 3, 2]), u(&[0, 1, 2], &[3]));
        let w = want(embedding_forward_dims(&table, &ids));
        let inputs = bytes(&[&table, &ids]);
        let (t2, i2) = (table.clone(), ids.clone());
        budgets(
            &mut out,
            "embedding_forward rank-3 table",
            inputs,
            &w,
            move |b| b.embedding_forward(&t2, &i2).map(drop),
        );
        // D3: a NaN in the table does not outrank the shape.
        let table_nan = nan(&[4, 3, 2]);
        out.push(cell(
            "embedding_forward rank-3 table with a NaN",
            u64::MAX,
            &w,
            move |b| b.embedding_forward(&table_nan, &ids).map(drop),
        ));
        // D5: both dtypes wrong; the table is named first.
        let (bad_table, bad_ids) = (u(&[0; 8], &[4, 2]), ft(&[3]));
        let w = want(embedding_forward_dims(&bad_table, &bad_ids));
        assert!(w.contains("expected: F32"), "{w}");
        out.push(cell(
            "embedding_forward table and ids both mistyped",
            u64::MAX,
            &w,
            move |b| b.embedding_forward(&bad_table, &bad_ids).map(drop),
        ));
    }

    // Embedding backward: grad shaped [3, 5] for dim 4.
    {
        let (table, ids, grad) = (ft(&[8, 4]), u(&[0, 1, 2], &[3]), ft(&[3, 5]));
        let w = want(embedding_backward_dims(&table, &ids, &grad));
        let inputs = bytes(&[&table, &ids, &grad]);
        let (t2, i2, g2) = (table.clone(), ids.clone(), grad.clone());
        budgets(
            &mut out,
            "embedding_backward grad shape",
            inputs,
            &w,
            move |b| b.embedding_backward(&t2, &i2, &g2).map(drop),
        );
        let (t3, i3) = (table.clone(), ids.clone());
        let grad_nan = nan(&[3, 5]);
        out.push(cell(
            "embedding_backward grad shape with a NaN grad",
            u64::MAX,
            &w,
            move |b| b.embedding_backward(&t3, &i3, &grad_nan).map(drop),
        ));
        // D11: an id outside the vocabulary does not outrank the grad shape.
        let bad_ids = u(&[0, 99, 2], &[3]);
        out.push(cell(
            "embedding_backward grad shape with an out-of-range id",
            u64::MAX,
            &w,
            move |b| b.embedding_backward(&table, &bad_ids, &grad).map(drop),
        ));
    }

    // Cross-entropy: targets [3] for logits [4, 5].
    for backward in [false, true] {
        let op = if backward {
            "cross_entropy_mean_backward"
        } else {
            "cross_entropy_mean_forward"
        };
        let dims = move |l: &Tensor, t: &Tensor| {
            if backward {
                want(cross_entropy_mean_backward_dims(l, t))
            } else {
                want(cross_entropy_mean_forward_dims(l, t))
            }
        };
        let run = move |b: &CpuBackend, l: &Tensor, t: &Tensor| {
            if backward {
                b.cross_entropy_mean_backward(l, t, None).map(drop)
            } else {
                b.cross_entropy_mean_forward(l, t, None).map(drop)
            }
        };
        let (logits, targets) = (ft(&[4, 5]), u(&[0, 1, 2], &[3]));
        let w = dims(&logits, &targets);
        let inputs = bytes(&[&logits, &targets]);
        let (l2, t2) = (logits.clone(), targets.clone());
        budgets(
            &mut out,
            &format!("{op} target shape"),
            inputs,
            &w,
            move |b| run(b, &l2, &t2),
        );
        let logits_nan = nan(&[4, 5]);
        out.push(cell(
            format!("{op} target shape with a NaN logit"),
            u64::MAX,
            &w,
            move |b| run(b, &logits_nan, &targets),
        ));
        // D6: both dtypes wrong; the logits are named first.
        let (bad_logits, bad_targets) = (u(&[0; 20], &[4, 5]), ft(&[4]));
        let w = dims(&bad_logits, &bad_targets);
        assert!(w.contains("expected: F32"), "{w}");
        out.push(cell(
            format!("{op} logits and targets both mistyped"),
            u64::MAX,
            &w,
            move |b| run(b, &bad_logits, &bad_targets),
        ));
    }

    // Clip: a zero-extent gradient.
    {
        let grads = vec![ft(&[16]), ft(&[2, 0])];
        let w = want(clip_grad_norm_dims(&grads));
        let inputs = bytes(&[&grads[0]]);
        budgets(
            &mut out,
            "clip_grad_norm zero extent",
            inputs,
            &w,
            move |b| b.clip_grad_norm(&mut grads.clone(), 1.0).map(drop),
        );
        // D3: after a gradient holding a NaN.
        let grads = vec![nan(&[16]), ft(&[2, 0])];
        out.push(cell(
            "clip_grad_norm zero extent after a NaN gradient",
            u64::MAX,
            &w,
            move |b| b.clip_grad_norm(&mut grads.clone(), 1.0).map(drop),
        ));
    }

    // AdamW: grad [4, 5] for param [4, 4].
    {
        let (p, g, m1, m2) = (ft(&[4, 4]), ft(&[4, 5]), ft(&[4, 4]), ft(&[4, 4]));
        let w = want(adamw_step_dims(&p, &g, &m1, &m2));
        let inputs = bytes(&[&p, &g, &m1, &m2]);
        let cfg = AdamWConfig::nanolab(1e-3, 0.1);
        let run = move |b: &CpuBackend, p: &Tensor, g: &Tensor, m1: &Tensor, m2: &Tensor| {
            let (mut p, mut m1, mut m2) = (p.clone(), m1.clone(), m2.clone());
            b.adamw_step(&mut p, g, &mut m1, &mut m2, 0, cfg)
        };
        let (p2, g2, a2, b2) = (p.clone(), g.clone(), m1.clone(), m2.clone());
        budgets(&mut out, "adamw_step grad shape", inputs, &w, move |b| {
            run(b, &p2, &g2, &a2, &b2)
        });
        let p_nan = nan(&[4, 4]);
        out.push(cell(
            "adamw_step grad shape with a NaN param",
            u64::MAX,
            &w,
            move |b| run(b, &p_nan, &g, &m1, &m2),
        ));
    }

    // Muon: a rank-1 parameter.
    {
        let (p, g, m) = (ft(&[16]), ft(&[16]), ft(&[16]));
        let w = want(muon_ns5_step_dims(&p, &g, &m));
        let inputs = bytes(&[&p, &g, &m]);
        let cfg = MuonNs5Config::nanolab_default();
        let run = move |b: &CpuBackend, p: &Tensor, g: &Tensor, m: &Tensor| {
            let (mut p, mut m) = (p.clone(), m.clone());
            b.muon_ns5_step(&mut p, g, &mut m, cfg)
        };
        let (p2, g2, m2) = (p.clone(), g.clone(), m.clone());
        budgets(
            &mut out,
            "muon_ns5_step rank-1 param",
            inputs,
            &w,
            move |b| run(b, &p2, &g2, &m2),
        );
        let g_nan = nan(&[16]);
        out.push(cell(
            "muon_ns5_step rank-1 param with a NaN grad",
            u64::MAX,
            &w,
            move |b| run(b, &p, &g_nan, &m),
        ));
    }
    out
}

/// Every cell returns its validator's error and leaves the budget idle, at
/// one thread and at seven.
#[test]
fn malformed_heavy_calls_return_the_validator_error_whatever_the_budget() {
    let mut failures = Vec::new();
    for c in cells() {
        for threads in [1usize, 7] {
            let be = CpuBackend::with_threads(Budget::new(c.cap), threads).unwrap();
            let got = (c.call)(&be).map_err(|e| format!("{e:?}"));
            let live = be.budget().live_bytes().unwrap();
            let ok = got.as_ref().err() == Some(&c.want) && live == 0;
            let verdict = if ok {
                "ok".to_string()
            } else {
                format!("FAIL: got {got:?}, live {live}, want {}", c.want)
            };
            println!("{} ({threads} threads): {verdict}", c.name);
            if !ok {
                failures.push(format!("{} ({threads} threads): {verdict}", c.name));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} cells failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
