//! ft-757cb6b2 criterion 8: a GEMM-backed op whose operand or output holds
//! more than `i32::MAX` elements is refused on the host as `Unsupported`,
//! before any reservation, allocation or dispatch. tessl's own
//! `require_i32_extent` (../tessl/src/gemm.rs) refuses the same quantity
//! later, on the device thread, after work was charged; the host bound makes
//! that refusal early and typed.
//!
//! Every case here uses a budget far below the oversize output, so nothing
//! of that size can be allocated: the only way to return `Unsupported` is
//! to refuse before the reservation, and the only way to return
//! `CapacityExceeded` is to have reached it. (Backward and gate operands
//! that are themselves oversize cannot be built here without an 8 GiB
//! tensor; the in-crate unit tests in `backend.rs` pin those bounds on the
//! dims directly.)

#![cfg(target_os = "macos")]

use ojas_core::{BackendCore, BackendFloat, BackendModule, Budget, OjasError, Tensor};
use ojas_metal::MetalBackend;

fn up(m: &MetalBackend, shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let host = Tensor::from_f32(&vec![0.5; n], shape, &Budget::new(1 << 26)).expect("host");
    m.upload(&host).expect("upload")
}

/// `[65537, 1] x [32768, 1]^T` outputs 65537 * 32768 = 2^31 + 2^15 elements.
#[test]
fn linear_forward_output_past_i32_is_unsupported_before_any_work() {
    let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
    let x = up(&m, &[65537, 1]);
    let w = up(&m, &[32768, 1]);
    m.sync().expect("settle");
    let dispatches = m.dispatches();
    let live = m.budget().live_bytes().expect("live");
    let allocated = m.memory().expect("memory").allocated;
    match m.linear_forward(&x, &w) {
        Err(OjasError::Unsupported { op, detail }) => {
            assert_eq!(op, "linear_forward");
            assert!(detail.contains("i32"), "{detail}");
        }
        other => panic!("expected Unsupported naming the i32 limit, got {other:?}"),
    }
    assert_eq!(m.dispatches(), dispatches, "nothing dispatched");
    assert_eq!(m.budget().live_bytes().expect("live"), live, "nothing charged");
    assert_eq!(
        m.memory().expect("memory").allocated,
        allocated,
        "nothing allocated"
    );
    m.sync().expect("clean");
}

/// Control: an output under the bound (2^30 + 2^14 elements, 4 GiB) is not
/// refused as `Unsupported`; with this 1 GiB budget it is a capacity
/// refusal. The host bound is tessl's, not tighter.
#[test]
fn linear_forward_output_under_i32_is_not_refused_as_unsupported() {
    let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
    let x = up(&m, &[65537, 1]);
    let w = up(&m, &[16384, 1]);
    match m.linear_forward(&x, &w) {
        Err(OjasError::CapacityExceeded { .. }) => {}
        other => panic!("expected CapacityExceeded, got {other:?}"),
    }
}
