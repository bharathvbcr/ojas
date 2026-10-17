//! A non-finite value-row outside a query's causal and sliding window, on
//! Metal: the contract, pinned for the forward and the backward.
//!
//! Key `j` is read by queries `j <= t < j + W` and by no other, and query `j`
//! always reads its own key, so no call has a non-finite `V` row that no query
//! sees. What this backend does:
//!
//! - **Forward.** The tiled kernel zeroes `P` (not `V`) for keys outside a
//!   row's window and then multiplies the whole 64-key `V` tile, so a
//!   non-finite `V` row in a visited tile reaches the `O` of every row that
//!   visits the tile as `0 * NaN`, whether or not that row's window holds the
//!   key. Nothing reads `O` after that: the host's output check raises the
//!   `NonFinite` fault of the op (`device.rs`, `sdpa`), reported by the next
//!   `sync`. The log-sum-exp is formed from the scores and is not touched.
//! - **Backward.** Every operand is checked up front (`ST_IN`), so a
//!   non-finite `V` row faults with the backward's own name.
//!
//! The contract is the fault, which is what the tests assert. They do not
//! assert what `O` holds after it, and the kernel is not changed (its
//! comment states what is and is not zeroed; `the_kernel_comment...` below
//! pins that text). The CPU refuses the call before it runs and wgpu skips
//! out-of-window keys, so only rows that see the key are non-finite there;
//! see `ojas-cpu/tests/attention_masked_nonfinite.rs` and
//! `ojas-wgpu/tests/attention_masked_nonfinite.rs`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Backend, Tensor};
use ojas_metal::MetalBackend;

/// `B = 1`, 4 query heads over 2 KV heads, `T = 96`, `D = 64`, window 10.
const H: usize = 4;
const HKV: usize = 2;
const T: usize = 96;
const D: usize = 64;
const W: usize = 10;
/// The poisoned key: KV head 1, position 20. Queries 20..30 see it; the
/// 64-key tile holding it is visited by query blocks 0..=2.
const KV_HEAD: usize = 1;
const POS: usize = 20;

const BAD: [(&str, f32); 3] = [
    ("NaN", f32::NAN),
    ("+inf", f32::INFINITY),
    ("-inf", f32::NEG_INFINITY),
];

type Fill = Box<dyn Fn(usize) -> Option<f32>>;

/// The poisoned row of `bad` throughout, and with `bad` in element 0 only.
fn fills(bad: f32) -> [(&'static str, Fill); 2] {
    [
        ("row", Box::new(move |_| Some(bad))),
        (
            "first element",
            Box::new(move |j| Some(if j == 0 { bad } else { 0.5 })),
        ),
    ]
}

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

/// `[q, k, v, gy]` on the device, with `V`'s row `POS` of KV head `KV_HEAD`
/// set by `fill` (`None` keeps the value).
fn inputs(m: &MetalBackend, fill: impl Fn(usize) -> Option<f32>) -> [Tensor; 4] {
    let mut v = values(HKV * T * D, 9302, 1.0);
    for j in 0..D {
        if let Some(x) = fill(j) {
            v[(KV_HEAD * T + POS) * D + j] = x;
        }
    }
    [
        up(m, &rand(&[1, H, T, D], 9300, 1.0)),
        up(m, &rand(&[1, HKV, T, D], 9301, 1.0)),
        up(m, &host(&v, &[1, HKV, T, D])),
        up(m, &rand(&[1, H, T, D], 9303, 1.0)),
    ]
}

#[test]
fn a_zeroed_value_row_is_accepted_and_the_lse_reads_no_value() {
    let m = metal();
    let [q, k, v, gy] = inputs(&m, |_| Some(0.0));
    let (y, lse) = ok("zeroed forward", m.causal_sdpa_forward(&q, &k, &v, Some(W)));
    let grads = ok(
        "zeroed backward",
        m.causal_sdpa_backward(&q, &k, &v, &y, &lse, &gy, Some(W)),
    );
    ok("sync", m.sync());
    assert!(down(&y).iter().all(|x| x.is_finite()));
    assert!(down(&grads.0).iter().all(|x| x.is_finite()));
}

#[test]
fn a_non_finite_value_row_is_a_deferred_fault_in_the_forward_and_leaves_the_lse_alone() {
    let m = metal();
    let [q, k, v, _] = inputs(&m, |_| Some(0.0));
    let (_, want_lse) = ok("zeroed forward", m.causal_sdpa_forward(&q, &k, &v, Some(W)));
    ok("sync", m.sync());
    let want = bits(&want_lse);
    for (name, bad) in BAD {
        for (shape, fill) in fills(bad) {
            let tag = format!("{name} {shape}");
            let [q, k, v, _] = inputs(&m, fill);
            let r = m.causal_sdpa_forward(&q, &k, &v, Some(W));
            // `deferred`: the op records, the next sync names it once.
            let (_, lse) = deferred(&m, &tag, r, "causal_sdpa_forward");
            assert_eq!(bits(&lse), want, "{tag}: the lse is formed from the scores");
        }
    }
}

#[test]
fn a_non_finite_value_row_is_a_deferred_fault_in_the_backward() {
    let m = metal();
    // Finite saved statistics from the zeroed forward: the fault below is
    // the poisoned `V` row's own.
    let [q, k, v0, gy] = inputs(&m, |_| Some(0.0));
    let (y, lse) = ok(
        "zeroed forward",
        m.causal_sdpa_forward(&q, &k, &v0, Some(W)),
    );
    ok("sync", m.sync());
    for (name, bad) in BAD {
        for (shape, fill) in fills(bad) {
            let tag = format!("{name} {shape}");
            let [_, _, v, _] = inputs(&m, fill);
            let r = m.causal_sdpa_backward(&q, &k, &v, &y, &lse, &gy, Some(W));
            deferred(&m, &tag, r, "causal_sdpa_backward");
        }
    }
}

/// No window hides the row from its own query: with the narrowest window
/// the call still faults.
#[test]
fn a_window_of_one_still_reads_the_poisoned_row_through_its_own_query() {
    let m = metal();
    let [q, k, v, _] = inputs(&m, |_| Some(f32::NAN));
    let r = m.causal_sdpa_forward(&q, &k, &v, Some(1));
    deferred(&m, "window 1", r, "causal_sdpa_forward");
}

/// The kernel header says what is zeroed: the probability and the score
/// gradient, not the `V` (or `K`, `dO`) operand, whose out-of-window rows in
/// a visited tile still multiply as `0 * x`. It used to say that every block
/// entry outside the window is set to 0 before it is multiplied, which holds
/// for `P` and `dS` and is false for `V`.
#[test]
fn the_kernel_comment_says_p_is_zeroed_and_v_is_not() {
    let source = include_str!("../kernels/ojas_backend.metal");
    assert!(
        !source.contains("every block entry\n// outside the window or past T is set to exactly 0"),
        "the false claim is back"
    );
    for needle in [
        "what is zeroed, not those operands",
        "0 * NaN",
        "NonFinite fault",
    ] {
        assert!(source.contains(needle), "kernel comment lost {needle:?}");
    }
}

/// The padding-row lead from the readout adversary: a NaN, `+inf` or `-inf`
/// embedding row at a padding id (a key after the real positions), gathered by
/// `embedding_forward` and run through causal attention as q, k and v.
///
/// What is true on Metal, established by running:
/// - `embedding_forward` does check its output: it is the first faulting op
///   in recording order, and a `sync` or `Backend::download` reports
///   `NonFinite { op: "embedding_forward" }`;
/// - the real positions' attention output is non-finite as well (the `0 * NaN`
///   of a visited tile, above), and a raw read (`Tensor::to_host`) returns
///   those values with no error, because a raw read is not a fault point
///   (`docs/metal-deferred-faults.md`, "Tensor::to_host ... the fault stays
///   pending"). A caller that reads only through `to_host` and never calls
///   `sync` or `Backend::download` is handed NaN without an error. That is the
///   documented contract, and it is what the readout adversary's helpers do
///   (`qwen35_common`'s `to_host`);
/// - the fault is not lost: the next `Backend::download` reports it, once.
#[test]
fn a_non_finite_padding_embedding_row_reaches_the_real_positions_and_faults_at_the_next_sync_point()
{
    const REAL: usize = 40;
    for (name, bad) in BAD {
        let m = metal();
        let mut table = values(64 * D, 9400, 1.0);
        table[63 * D..].fill(bad);
        let mut ids = vec![1u32; T];
        for (i, id) in ids.iter_mut().enumerate().take(REAL) {
            *id = (i % 60) as u32;
        }
        ids[REAL..].fill(63);
        let table = up(&m, &host(&table, &[64, D]));
        let ids = up(&m, &host_u32(&ids, &[1, T]));
        let x = ok("embedding", m.embedding_forward(&table, &ids));
        let heads = ok("reshape", x.reshape(&[1, T, 1, D]));
        let qkv = ok("permute", m.permute(&heads, &[0, 2, 1, 3]));
        let (y, _) = ok("attention", m.causal_sdpa_forward(&qkv, &qkv, &qkv, None));
        // Raw read, no sync: the real positions are non-finite and no error
        // is raised.
        let raw = down(&y);
        let real = &raw[..REAL * D];
        assert!(
            real.iter().all(|v| !v.is_finite()),
            "{name}: real positions"
        );
        // The backend's own read reports the first fault in recording order,
        // and only once.
        match m.download(&y) {
            Err(ojas_core::OjasError::NonFinite { op }) => {
                assert_eq!(op, "embedding_forward", "{name}")
            }
            other => panic!("{name}: download must report the fault: {other:?}"),
        }
        ok("a fault is reported once", m.sync());
    }
}
