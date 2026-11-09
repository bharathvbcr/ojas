//! The raw-read contract (`DeviceBuffer::read_bytes`, `Tensor::to_host`): on
//! a Metal tensor both are a byte read that checks no fault. An op that
//! detects a non-finite value on the device returns `Ok`, `to_host` of its
//! output returns `Ok` with the bytes as the kernel wrote them (the NaN
//! included), and the fault stays pending until the next `sync` (or
//! `download`), which names the op and reports it once. A caller that needs
//! checked bytes syncs or downloads first.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::{BackendCore, BackendFloat, Budget, OjasError};

fn nonfinite_op(r: Result<impl std::fmt::Debug, OjasError>) -> &'static str {
    match r {
        Err(OjasError::NonFinite { op }) => op,
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

/// `to_host` of the faulted output returns the bytes unchecked and leaves the
/// fault pending for `sync`.
#[test]
fn to_host_of_a_faulted_output_is_a_raw_read_and_sync_reports_the_op() {
    let m = metal();
    let x = up(&m, &host(&[1.0, f32::NAN, 2.0], &[3]));
    let y = ok("silu records, it does not refuse", m.silu_forward(&x));

    // The raw read: Ok, with the bytes the kernel wrote.
    let raw = match y.to_host(&Budget::new(GIB)) {
        Ok(h) => ok("decode", h.to_f32_vec()),
        Err(e) => panic!("to_host must be a raw read, got {e:?}"),
    };
    assert_eq!(raw.len(), 3);
    assert!(raw[0].is_finite() && raw[2].is_finite(), "{raw:?}");
    assert!(raw[1].is_nan(), "the NaN is returned as written: {raw:?}");

    // Reading again does not consume or check the fault either.
    assert!(y.to_host(&Budget::new(GIB)).is_ok(), "a second raw read");

    // The fault was pending the whole time: the next sync names the op, once.
    assert_eq!(nonfinite_op(m.sync()), "silu_forward");
    m.sync().expect("reported once");
}

/// `download` is the checked read: it reports the fault `to_host` left pending.
#[test]
fn download_reports_the_fault_a_raw_read_left_pending() {
    let m = metal();
    let x = up(&m, &host(&[f32::INFINITY, 1.0], &[2]));
    let y = ok("silu records", m.silu_forward(&x));
    let raw = ok("raw read", y.to_host(&Budget::new(GIB)));
    assert!(ok("decode", raw.to_f32_vec())
        .iter()
        .any(|v| !v.is_finite()));
    assert_eq!(nonfinite_op(m.download(&y)), "silu_forward");
    m.sync().expect("download reported it once");
}
