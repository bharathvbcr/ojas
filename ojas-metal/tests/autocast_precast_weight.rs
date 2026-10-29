//! A weight rounded once at load (`Autocast::cast_bf16`, an `F32` tensor
//! tagged bf16-exact) and reused as the weight of every call under
//! `Autocast<MetalBackend>`: `Autocast::prep` passes it through, so the
//! per-call weight rounding is gone and the GEMM runs on tessl's bf16 lane.
//! SpikeTrack's trunk weights take this route
//! (`Backbone::with_trunk_linears_bf16`), so it is pinned here as supported
//! usage.
//!
//! These are characterization tests: they describe what the code does at
//! 7fd9fc3 and pass there, and nothing in ojas changed with them. They pin
//! four things for `linear_forward` and `linear_bias_forward` in a bf16
//! region: (a) the pre-cast weight's result has the bits of the same call on
//! the untagged `F32` weight; (b) one `ojas_round_bf16` dispatch fewer per
//! call; (c) the GEMM is the bf16 lane's (two tessl operand casts more than
//! the exact lane's, on a device with TensorOps); (d) an in-place write to
//! the pre-cast weight clears its tag, and the call then matches the
//! untagged path on the written values.
#![cfg(target_os = "macos")]

mod common;

use common::{bits, down, host, metal, ok, rand, up};
use ojas_core::{bf16_to_f32, f32_to_bf16, Autocast, AutocastMode, Backend, OjasError, Tensor};
use ojas_metal::MetalBackend;

/// The fc1 shape's ragged neighbour: rows and `n` off every tile edge, `k`
/// not a multiple of a tile, so the lane's edge handling is in the compare.
const ROWS: usize = 197;
const KIN: usize = 200;
const NOUT: usize = 321;
const ROUND: &str = "ojas_round_bf16";

/// Dispatches (tessl binder scopes), and `ojas_round_bf16` dispatches, `op`
/// encodes, and its result.
fn counted<T>(m: &MetalBackend, op: impl FnOnce() -> Result<T, OjasError>) -> (T, u64, u64) {
    ok("sync", m.sync());
    let (all, rounds) = (m.dispatches(), m.kernel_dispatches(ROUND));
    let out = ok("counted op", op());
    ok("sync", m.sync());
    (
        out,
        m.dispatches() - all,
        m.kernel_dispatches(ROUND) - rounds,
    )
}

/// Operand casts tessl adds on the bf16 lane: one per f32 operand, on a
/// device with TensorOps only.
fn casts(m: &MetalBackend) -> u64 {
    if m.has_tensorops() {
        2
    } else {
        0
    }
}

/// An untagged upload of `t`'s values.
fn plain(m: &MetalBackend, t: &Tensor) -> Tensor {
    let p = up(m, &host(&down(t), t.shape()));
    assert!(!p.bf16_exact());
    p
}

/// `x` and an untagged `w` whose values are not bf16 values (a bf16 round
/// changes them), the bias, and `w` rounded once.
fn operands(m: &MetalBackend, ac: &Autocast<MetalBackend>) -> [Tensor; 4] {
    let x = up(m, &rand(&[ROWS, KIN], 11, 1.0));
    let w = up(m, &rand(&[NOUT, KIN], 12, 1.0));
    let b = up(m, &rand(&[NOUT], 13, 1.0));
    let w_pre = ok("cast_bf16", ac.cast_bf16(&w));
    assert!(w_pre.bf16_exact(), "cast_bf16 tags its result");
    assert!(!w.bf16_exact() && !x.bf16_exact());
    let changed = down(&w)
        .iter()
        .zip(down(&w_pre))
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(changed > 0, "w must not already be bf16 values");
    [x, w, b, w_pre]
}

#[test]
fn a_precast_weight_gives_the_untagged_weights_bits_with_one_round_fewer() {
    let m = metal();
    let ac = Autocast::new(m.clone());
    let [x, w, b, w_pre] = operands(&m, &ac);
    let _region = ok("region", ac.autocast_region(AutocastMode::Bf16));

    let (y_plain, n_plain, r_plain) = counted(&m, || ac.linear_forward(&x, &w));
    let (y_pre, n_pre, r_pre) = counted(&m, || ac.linear_forward(&x, &w_pre));
    // (a) the same bits.
    assert_eq!(bits(&down(&y_pre)), bits(&down(&y_plain)), "linear_forward");
    // (b) the input round, the output round and, for the untagged weight
    // only, the weight round.
    assert_eq!(r_plain, 3, "untagged weight: x, w and output rounds");
    assert_eq!(r_pre, 2, "pre-cast weight: x and output rounds");
    assert_eq!(n_plain, n_pre + 1, "one dispatch fewer per call");
    // A second call reads the same tagged weight: still no weight round.
    let (y_again, n_again, r_again) = counted(&m, || ac.linear_forward(&x, &w_pre));
    assert_eq!((n_again, r_again), (n_pre, r_pre), "repeat call");
    assert_eq!(bits(&down(&y_again)), bits(&down(&y_pre)));

    // The bias op is the same route.
    let (yb_plain, nb_plain, rb_plain) = counted(&m, || ac.linear_bias_forward(&x, &w, &b));
    let (yb_pre, nb_pre, rb_pre) = counted(&m, || ac.linear_bias_forward(&x, &w_pre, &b));
    assert_eq!(
        bits(&down(&yb_pre)),
        bits(&down(&yb_plain)),
        "linear_bias_forward"
    );
    assert_eq!((rb_plain, rb_pre), (3, 2), "bias op rounds");
    assert_eq!(nb_plain, nb_pre + 1, "bias op: one dispatch fewer");
}

#[test]
fn a_precast_weight_runs_the_bf16_lane() {
    let m = metal();
    let ac = Autocast::new(m.clone());
    let [x, _w, _b, w_pre] = operands(&m, &ac);
    // The exact lane's call on the same rounded operands, outside a region.
    let xp = plain(&m, &ok("round x", ac.cast_bf16(&x)));
    let wp = plain(&m, &w_pre);
    let (_, exact_n, exact_r) = counted(&m, || m.linear_forward(&xp, &wp));
    assert_eq!(exact_r, 0, "the backend itself rounds nothing");

    let _region = ok("region", ac.autocast_region(AutocastMode::Bf16));
    let (_, n_pre, r_pre) = counted(&m, || ac.linear_forward(&x, &w_pre));
    // The exact lane's GEMM, the two rounds the wrapper makes, and tessl's
    // cast of each of the GEMM's two f32 operands to bf16.
    assert_eq!(
        n_pre,
        exact_n + r_pre + casts(&m),
        "bf16 lane dispatches (has_tensorops {})",
        m.has_tensorops()
    );
}

#[test]
fn an_in_place_write_clears_the_tag_and_the_call_matches_the_untagged_path() {
    let m = metal();
    let ac = Autocast::new(m.clone());
    let [x, _w, _b, w_pre] = operands(&m, &ac);
    let mut w_hot = ok("cast_bf16", ac.cast_bf16(&plain(&m, &w_pre)));
    assert!(w_hot.bf16_exact());
    let _region = ok("region", ac.autocast_region(AutocastMode::Bf16));
    let before = ok("before", ac.linear_forward(&x, &w_hot));

    ok("scale_grad", ac.scale_grad(&mut w_hot, 0.3));
    assert!(!w_hot.bf16_exact(), "an in-place write clears the tag");
    let written = down(&w_hot);
    assert!(
        written
            .iter()
            .any(|&v| bf16_to_f32(f32_to_bf16(v)).to_bits() != v.to_bits()),
        "the write must leave values that are not bf16 values"
    );

    // The written weight is an ordinary F32 one: the wrapper rounds it per
    // call, as it does an untagged upload of the same values.
    let (after, _, r_after) = counted(&m, || ac.linear_forward(&x, &w_hot));
    assert_eq!(r_after, 3, "the written weight is rounded again");
    let (want, _, _) = counted(&m, || ac.linear_forward(&x, &plain(&m, &w_hot)));
    assert_eq!(bits(&down(&after)), bits(&down(&want)), "written weight");
    assert_ne!(
        bits(&down(&after)),
        bits(&down(&before)),
        "the write must change the product"
    );
}
