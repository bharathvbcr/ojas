//! A non-finite value-row outside a query's causal and sliding window, on
//! the CPU: the contract, pinned for the forward and the backward.
//!
//! Key `j` is read by queries `j <= t < j + W` and by no other. A NaN, `+inf`
//! or `-inf` in `V[j]` is therefore invisible to every other query, but not to
//! query `j` itself, which always attends to its own key. So no call has a
//! non-finite `V` row that no query sees, and what the backends differ in is
//! how the call fails and what the rows that cannot see `j` hold afterwards.
//!
//! On the CPU the call fails before it runs: every operand is scanned and a
//! non-finite value anywhere in `q`, `k` or `v` is `NonFinite` naming the op,
//! nothing charged. (The wgpu and Metal counterparts are
//! `ojas-wgpu/tests/attention_masked_nonfinite.rs` and
//! `ojas-metal/tests/attention_masked_nonfinite.rs`.)
//!
//! With the row zeroed instead, the call succeeds and the `V` row influences
//! exactly the rows that see it: every output, log-sum-exp and `dQ` row that
//! cannot see key `j` equals the clean call's bit for bit, the whole
//! log-sum-exp equals it (it is formed from the scores), and `dV` equals it
//! (it never reads `V`).

use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

/// `B = 1`, 4 query heads over 2 KV heads, `T = 96`, `D = 16`, window 10.
const H: usize = 4;
const HKV: usize = 2;
const T: usize = 96;
const D: usize = 16;
const W: usize = 10;
/// The poisoned key: KV head 1, position 20. Queries 20..30 see it.
const KV_HEAD: usize = 1;
const POS: usize = 20;

fn backend(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 30), threads)
        .unwrap()
        .with_numerics(numerics)
}

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn tensor(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

struct Inputs {
    q: Tensor,
    k: Tensor,
    v: Vec<f32>,
    gy: Tensor,
}

impl Inputs {
    fn new() -> Self {
        Self {
            q: tensor(&values(H * T * D, 1), &[1, H, T, D]),
            k: tensor(&values(HKV * T * D, 2), &[1, HKV, T, D]),
            v: values(HKV * T * D, 3),
            gy: tensor(&values(H * T * D, 4), &[1, H, T, D]),
        }
    }

    /// `v` with row `POS` of KV head `KV_HEAD` set by `fill`.
    fn v_with(&self, fill: impl Fn(usize) -> f32) -> Tensor {
        let mut v = self.v.clone();
        for j in 0..D {
            v[(KV_HEAD * T + POS) * D + j] = fill(j);
        }
        tensor(&v, &[1, HKV, T, D])
    }

    fn clean_v(&self) -> Tensor {
        tensor(&self.v, &[1, HKV, T, D])
    }
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Whether query row `t` sees key `POS`.
fn sees(t: usize) -> bool {
    (POS..POS + W).contains(&t)
}

const BAD: [(&str, f32); 3] = [
    ("NaN", f32::NAN),
    ("+inf", f32::INFINITY),
    ("-inf", f32::NEG_INFINITY),
];

fn configs() -> Vec<(usize, Numerics)> {
    let mut out = Vec::new();
    for threads in [1usize, 3] {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            out.push((threads, numerics));
        }
    }
    out
}

fn non_finite(r: Result<impl std::fmt::Debug, OjasError>) -> &'static str {
    match r {
        Err(OjasError::NonFinite { op }) => op,
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

/// The forward and every backward entry refuse a non-finite `V` row that
/// the queries outside its window cannot see, for each bad value, a full
/// row or one element, both thread counts and both numerics, and charge
/// nothing for the refusal.
#[test]
fn a_non_finite_value_row_refuses_the_whole_call() {
    let i = Inputs::new();
    for (threads, numerics) in configs() {
        let be = backend(threads, numerics);
        let (y, lse) = be
            .causal_sdpa_forward(&i.q, &i.k, &i.clean_v(), Some(W))
            .unwrap();
        let before = be.budget().live_bytes().unwrap();
        for (name, bad) in BAD {
            for (shape, poisoned) in [
                ("row", i.v_with(|_| bad)),
                (
                    "first element",
                    i.v_with(|j| if j == 0 { bad } else { 0.5 }),
                ),
            ] {
                let tag = format!("{name} {shape} threads {threads} {numerics:?}");
                let forward = be.causal_sdpa_forward(&i.q, &i.k, &poisoned, Some(W));
                assert_eq!(non_finite(forward), "causal_sdpa_forward", "{tag}");
                let backward =
                    be.causal_sdpa_backward(&i.q, &i.k, &poisoned, &y, &lse, &i.gy, Some(W));
                assert_eq!(non_finite(backward), "causal_sdpa_backward", "{tag}");
                let recompute =
                    be.causal_sdpa_backward_recompute(&i.q, &i.k, &poisoned, &i.gy, Some(W));
                assert!(
                    matches!(
                        non_finite(recompute),
                        "causal_sdpa_forward" | "causal_sdpa_backward"
                    ),
                    "{tag}"
                );
                assert_eq!(
                    be.budget().live_bytes().unwrap(),
                    before,
                    "{tag}: a refusal charged"
                );
            }
        }
    }
}

/// With the row zeroed the call is accepted, and the `V` row reaches only the
/// rows that see it: the other rows' outputs and `dQ` rows, the whole
/// log-sum-exp and the whole `dV` equal the clean call's bit for bit.
#[test]
fn a_zeroed_value_row_reaches_only_the_rows_that_see_it() {
    let i = Inputs::new();
    for (threads, numerics) in configs() {
        let be = backend(threads, numerics);
        let tag = format!("threads {threads} {numerics:?}");
        let clean_v = i.clean_v();
        let zeroed_v = i.v_with(|_| 0.0);
        let (y, lse) = be
            .causal_sdpa_forward(&i.q, &i.k, &clean_v, Some(W))
            .unwrap();
        let (yz, lsez) = be
            .causal_sdpa_forward(&i.q, &i.k, &zeroed_v, Some(W))
            .unwrap();
        assert!(host(&yz).iter().all(|v| v.is_finite()), "{tag}");
        assert_eq!(
            bits(&host(&lse)),
            bits(&host(&lsez)),
            "{tag}: lse reads no V"
        );
        let (gq, _, gv) = be
            .causal_sdpa_backward(&i.q, &i.k, &clean_v, &y, &lse, &i.gy, Some(W))
            .unwrap();
        let (gqz, _, gvz) = be
            .causal_sdpa_backward(&i.q, &i.k, &zeroed_v, &yz, &lsez, &i.gy, Some(W))
            .unwrap();
        assert_eq!(bits(&host(&gv)), bits(&host(&gvz)), "{tag}: dV reads no V");
        let (y, yz) = (host(&y), host(&yz));
        let (gq, gqz) = (host(&gq), host(&gqz));
        let mut seen_differs = false;
        for head in 0..H {
            // Query head `head` reads KV head `head / (H / HKV)`.
            let reads_poisoned_plane = head / (H / HKV) == KV_HEAD;
            for t in 0..T {
                let at = (head * T + t) * D;
                let same_y = bits(&y[at..at + D]) == bits(&yz[at..at + D]);
                let same_gq = bits(&gq[at..at + D]) == bits(&gqz[at..at + D]);
                if reads_poisoned_plane && sees(t) {
                    seen_differs |= !same_y;
                } else {
                    assert!(
                        same_y,
                        "{tag}: output head {head} row {t} reads the zeroed row"
                    );
                    assert!(
                        same_gq,
                        "{tag}: dQ head {head} row {t} reads the zeroed row"
                    );
                }
            }
        }
        assert!(
            seen_differs,
            "{tag}: zeroing the row changed no row that sees it"
        );
    }
}
