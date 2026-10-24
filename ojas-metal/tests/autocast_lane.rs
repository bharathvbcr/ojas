//! Linear on bf16-exact operands (`Tensor::bf16_exact`, what `Autocast`
//! hands Metal in a bf16 region) runs on tessl's bf16 lane, the
//! 4-simdgroup cooperative TensorOps kernels, instead of the exact-f32
//! 1-simdgroup tile. The products are the exact lane's, so the outputs
//! differ only by the order of the f32 sums. A pair with one untagged
//! operand, and a `Bf16` operand, stay on the exact-f32 lane.

mod common;

use common::{close, down, host, metal, ok, rand, up};
use ojas_core::{Autocast, Backend, Budget, OjasError, Tensor};
use ojas_metal::MetalBackend;

const ROWS: usize = 256;
const KIN: usize = 192;
const NOUT: usize = 320;
/// Order-only: K = 192 products of values in [-1, 1) summed in another
/// order. One bf16 rounding of an output (2^-9 relative) is far outside it.
const ATOL: f32 = 1e-4;
const RTOL: f32 = 1e-5;

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

/// Dispatches (tessl binder scopes) `op` encodes, and its result.
fn counted<T>(m: &MetalBackend, op: impl FnOnce() -> Result<T, OjasError>) -> (T, u64) {
    ok("sync", m.sync());
    let before = m.dispatches();
    let out = ok("counted op", op());
    ok("sync", m.sync());
    (out, m.dispatches() - before)
}

/// `t` rounded by `Autocast` (tagged bf16-exact), and an untagged upload of
/// the same bits.
fn rounded(m: &MetalBackend, ac: &Autocast<MetalBackend>, t: &Tensor) -> (Tensor, Tensor) {
    let tagged = ok("cast_bf16", ac.cast_bf16(&up(m, t)));
    assert!(tagged.bf16_exact());
    let plain = up(m, &host(&down(&tagged), t.shape()));
    assert!(!plain.bf16_exact());
    (tagged, plain)
}

#[test]
fn bf16_exact_linear_forward_runs_the_bf16_lane() {
    let m = metal();
    let ac = Autocast::new(m.clone());
    let (x, xp) = rounded(&m, &ac, &rand(&[ROWS, KIN], 1, 1.0));
    let (w, wp) = rounded(&m, &ac, &rand(&[NOUT, KIN], 2, 1.0));

    let (exact, exact_n) = counted(&m, || m.linear_forward(&xp, &wp));
    let (lane, lane_n) = counted(&m, || m.linear_forward(&x, &w));
    // The bf16 lane casts its two f32 operands to bf16 before the GEMM.
    assert_eq!(lane_n, exact_n + 2, "bf16 lane dispatches");
    close("forward", &down(&lane), &down(&exact), ATOL, RTOL);

    // One untagged operand: the exact lane, bit for bit.
    let (mixed, mixed_n) = counted(&m, || m.linear_forward(&x, &wp));
    assert_eq!(mixed_n, exact_n, "mixed pair dispatches");
    assert_eq!(bits(&mixed), bits(&exact));

    // A Bf16 operand keeps its widened copy's bits (the bf16-operand
    // contract), so it is not routed by its dtype.
    let wh = ok("to_bf16", m.to_bf16(&wp));
    let half = ok("half", m.linear_forward(&x, &wh));
    assert_eq!(bits(&half), bits(&exact));
}

#[test]
fn bf16_exact_linear_backward_runs_the_bf16_lane_per_gemm() {
    let m = metal();
    let ac = Autocast::new(m.clone());
    let (x, xp) = rounded(&m, &ac, &rand(&[ROWS, KIN], 3, 1.0));
    let (w, wp) = rounded(&m, &ac, &rand(&[NOUT, KIN], 4, 1.0));
    let (gy, gyp) = rounded(&m, &ac, &rand(&[ROWS, NOUT], 5, 1.0));

    let ((gx_e, gw_e), exact_n) = counted(&m, || m.linear_backward(&xp, &wp, &gyp));
    let ((gx, gw), lane_n) = counted(&m, || m.linear_backward(&x, &w, &gy));
    // Two GEMMs on the bf16 lane, each casting its two operands.
    assert_eq!(lane_n, exact_n + 4, "bf16 lane dispatches");
    close("grad_input", &down(&gx), &down(&gx_e), ATOL, RTOL);
    // gW sums over the 256 rows.
    close("grad_weight", &down(&gw), &down(&gw_e), 2.0 * ATOL, RTOL);

    // Only gW's pair (gy, x) is bf16-exact: gX keeps the exact lane's bits.
    let ((gx_m, gw_m), mixed_n) = counted(&m, || m.linear_backward(&x, &wp, &gy));
    assert_eq!(mixed_n, exact_n + 2, "one GEMM on the bf16 lane");
    assert_eq!(bits(&gx_m), bits(&gx_e));
    assert_eq!(bits(&gw_m), bits(&gw));

    let (gx_i, input_n) = counted(&m, || m.linear_backward_input(&x, &w, &gy));
    let (gx_ie, input_exact_n) = counted(&m, || m.linear_backward_input(&xp, &wp, &gyp));
    assert_eq!(input_n, input_exact_n + 2, "bf16 lane dispatches");
    assert_eq!(bits(&gx_i), bits(&gx));
    assert_eq!(bits(&gx_ie), bits(&gx_e));
}

/// The bf16 copies tessl makes are charged with the op's scratch: a budget
/// that holds the exact lane's call but not the copies refuses the bf16
/// lane before anything runs, and the exact lane still runs under it.
#[test]
fn the_bf16_lane_charges_its_operand_copies() {
    let m = metal();
    let ac = Autocast::new(m.clone());
    let (x, xp) = rounded(&m, &ac, &rand(&[ROWS, KIN], 6, 1.0));
    let (w, wp) = rounded(&m, &ac, &rand(&[NOUT, KIN], 7, 1.0));
    // The operands sit on `m`'s budget; `tight` charges only the call.
    let out = (ROWS * NOUT * 4) as u64;
    let copies = ((ROWS * KIN + NOUT * KIN) * 2) as u64;
    let tight = m.with_budget(Budget::new(out + copies / 2));
    assert!(matches!(
        tight.linear_forward(&x, &w),
        Err(OjasError::CapacityExceeded { .. })
    ));
    ok("exact lane", tight.linear_forward(&xp, &wp));
}
