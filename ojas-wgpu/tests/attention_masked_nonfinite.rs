//! A non-finite value-row outside a query's causal and sliding window, on
//! wgpu: the contract, pinned for the forward and the backward.
//!
//! Key `j` is read by queries `j <= t < j + W` and by no other, and query `j`
//! always reads its own key, so no call has a non-finite `V` row that no query
//! sees. What this backend guarantees (`attention.wgsl` skips keys outside a
//! row's window, so they are never multiplied, not even as `0 * NaN`):
//!
//! - the call faults: it returns, and the next `sync` reports `NonFinite`
//!   naming the op, because the rows that do see the row are non-finite;
//! - the rows that cannot see the row hold exactly what the same call with
//!   that `V` row zeroed holds, bit for bit: output rows and `dQ` rows
//!   outside `j <= t < j + W`; and the whole log-sum-exp, which is formed
//!   from the scores, and the whole `dV`, which never reads `V`.
//!
//! The CPU refuses the call before it runs and Metal faults through its
//! output check; see `ojas-cpu/tests/attention_masked_nonfinite.rs` and
//! `ojas-metal/tests/attention_masked_nonfinite.rs`.

mod common;

use common::*;
use ojas_core::{Backend, OjasError, Tensor};
use ojas_wgpu::WgpuBackend;

/// `B = 1`, 4 query heads over 2 KV heads, `T = 96`, `D = 48`, window 10.
const H: usize = 4;
const HKV: usize = 2;
const T: usize = 96;
const D: usize = 48;
const W: usize = 10;
/// The poisoned key: KV head 1, position 20. Queries 20..30 see it.
const KV_HEAD: usize = 1;
const POS: usize = 20;

const BAD: [(&str, f32); 3] = [
    ("NaN", f32::NAN),
    ("+inf", f32::INFINITY),
    ("-inf", f32::NEG_INFINITY),
];

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn sees(t: usize) -> bool {
    (POS..POS + W).contains(&t)
}

/// `[q, k, v, gy]` as host tensors, with `V`'s row `POS` of KV head
/// `KV_HEAD` set by `fill`.
fn host_inputs(fill: impl Fn(usize) -> Option<f32>) -> [Tensor; 4] {
    let mut v = data(9300 + 2, HKV * T * D);
    for j in 0..D {
        if let Some(x) = fill(j) {
            v[(KV_HEAD * T + POS) * D + j] = x;
        }
    }
    let f = |seed, n: usize, shape: &[usize]| {
        Tensor::from_f32(&data(seed, n), shape, host_budget()).unwrap()
    };
    [
        f(9300, H * T * D, &[1, H, T, D]),
        f(9300 + 1, HKV * T * D, &[1, HKV, T, D]),
        Tensor::from_f32(&v, &[1, HKV, T, D], host_budget()).unwrap(),
        f(9300 + 3, H * T * D, &[1, H, T, D]),
    ]
}

fn up_on(g: &WgpuBackend, t: &Tensor) -> Tensor {
    g.upload(t).unwrap()
}

fn host_of(g: &WgpuBackend, t: &Tensor) -> Vec<f32> {
    g.download(t).unwrap().to_f32_vec().unwrap()
}

fn nonfinite_op(r: Result<(), OjasError>) -> &'static str {
    match r {
        Err(OjasError::NonFinite { op }) => op,
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

/// Forward outputs, lse and the saved-statistics backward of one `V`.
struct Run {
    y: Vec<f32>,
    lse: Vec<f32>,
    gq: Vec<f32>,
    gv: Vec<f32>,
}

/// The forward and the backward (from `saved`'s host output and lse, or this
/// forward's own) on `g` with value tensor `v`. Returns what each call
/// produced and the faults `sync` reported after the forward and after the
/// backward.
fn run(
    g: &WgpuBackend,
    [q, k, v, gy]: &[Tensor; 4],
    saved: Option<(&Tensor, &Tensor)>,
) -> (Run, Option<&'static str>, Option<&'static str>) {
    let (q, k, v, gy) = (up_on(g, q), up_on(g, k), up_on(g, v), up_on(g, gy));
    let (y, lse) = g.causal_sdpa_forward(&q, &k, &v, Some(W)).unwrap();
    let after_forward = match g.sync() {
        Ok(()) => None,
        Err(OjasError::NonFinite { op }) => Some(op),
        Err(e) => panic!("sync after forward: {e:?}"),
    };
    // Host statistics are uploaded; this forward's own stay where they are.
    let uploaded = saved.map(|(sy, sl)| (up_on(g, sy), up_on(g, sl)));
    let (sy, sl) = uploaded.as_ref().map_or((&y, &lse), |(a, b)| (a, b));
    let (gq, _, gv) = g
        .causal_sdpa_backward(&q, &k, &v, sy, sl, &gy, Some(W))
        .unwrap();
    let after_backward = match g.sync() {
        Ok(()) => None,
        Err(OjasError::NonFinite { op }) => Some(op),
        Err(e) => panic!("sync after backward: {e:?}"),
    };
    let out = Run {
        y: host_of(g, &y),
        lse: host_of(g, &lse),
        gq: host_of(g, &gq),
        gv: host_of(g, &gv),
    };
    (out, after_forward, after_backward)
}

#[test]
fn a_non_finite_value_row_faults_and_the_rows_that_cannot_see_it_are_those_of_a_zeroed_row() {
    let g = &fresh();
    g.sync().unwrap();
    // The zeroed row: no fault, finite, and the reference for the rest.
    let zeroed = host_inputs(|_| Some(0.0));
    let (want, f, b) = run(g, &zeroed, None);
    assert_eq!((f, b), (None, None), "a zeroed row faulted");
    assert!(want.y.iter().chain(&want.gq).all(|v| v.is_finite()));
    // The saved output and lse of the zeroed call: a poisoned backward is
    // given finite saved statistics, so its fault is its own `V` row's.
    let up_zero =
        |t: &Vec<f32>, shape: &[usize]| Tensor::from_f32(t, shape, host_budget()).unwrap();
    let (sy, sl) = (
        up_zero(&want.y, &[1, H, T, D]),
        up_zero(&want.lse, &[1, H, T]),
    );
    for (name, bad) in BAD {
        for (shape, poisoned) in [
            ("row", host_inputs(|_| Some(bad))),
            (
                "first element",
                host_inputs(|j| Some(if j == 0 { bad } else { 0.5 })),
            ),
        ] {
            let tag = format!("{name} {shape}");
            let (got, after_forward, after_backward) = run(g, &poisoned, Some((&sy, &sl)));
            assert_eq!(after_forward, Some("causal_sdpa_forward"), "{tag}");
            assert_eq!(after_backward, Some("causal_sdpa_backward"), "{tag}");
            // Rows that cannot see key POS in the poisoned plane's heads, and
            // every row of the heads on the other KV plane.
            for head in 0..H {
                let plane = head / (H / HKV);
                for t in 0..T {
                    let at = (head * T + t) * D;
                    if plane == KV_HEAD && sees(t) {
                        continue;
                    }
                    assert_eq!(
                        bits(&got.y[at..at + D]),
                        bits(&want.y[at..at + D]),
                        "{tag}: output head {head} row {t}"
                    );
                    assert_eq!(
                        bits(&got.gq[at..at + D]),
                        bits(&want.gq[at..at + D]),
                        "{tag}: dQ head {head} row {t}"
                    );
                }
            }
            assert_eq!(bits(&got.lse), bits(&want.lse), "{tag}: lse reads no V");
            assert_eq!(bits(&got.gv), bits(&want.gv), "{tag}: dV reads no V");
            // The rows that do see it are not finite: that is the fault.
            let seen = (KV_HEAD * (H / HKV)) * T + POS;
            assert!(
                got.y[seen * D..(seen + 1) * D]
                    .iter()
                    .any(|v| !v.is_finite()),
                "{tag}: the row that sees the poisoned key is finite"
            );
        }
    }
    g.sync().unwrap();
}

/// A poisoned `V` whose rows are all outside every row's window but its own
/// still faults: there is no window setting that hides it from query `j`.
#[test]
fn a_window_of_one_still_reads_the_poisoned_row_through_its_own_query() {
    let g = &fresh();
    g.sync().unwrap();
    let [q, k, v, _] = host_inputs(|_| Some(f32::NAN));
    let (q, k, v) = (up_on(g, &q), up_on(g, &k), up_on(g, &v));
    let _ = g.causal_sdpa_forward(&q, &k, &v, Some(1)).unwrap();
    assert_eq!(nonfinite_op(g.sync()), "causal_sdpa_forward");
}
