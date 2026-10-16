//! Adversarial, host-only tests for card ft-cab0fdf1 on CUDA: the conversion
//! `CudaError -> OjasError` and the budget view the runtime reserves
//! through. No device is needed; nothing here calls the driver.
//!
//! Tests marked `#[ignore = "BREAK: ..."]` assert the property the card
//! promises and currently fail.

use ojas_core::OjasError;
use ojas_cuda::budget::AllocBudget;
use ojas_cuda::error::{CUDA_ERROR_OUT_OF_MEMORY, STICKY_DRIVER_CODES};
use ojas_cuda::CudaError;

/// Every driver status maps to exactly one outcome: only status 2 is
/// capacity, the sticky set is device-lost, the rest are backend errors; a
/// status's text never changes the class.
#[test]
fn every_driver_code_has_one_class_whatever_its_text() {
    let texts = [
        "out of memory",
        "CUDA_ERROR_OUT_OF_MEMORY",
        "",
        "device lost",
        "capacity",
    ];
    for code in 0..3000u32 {
        for text in texts {
            let e = CudaError::from_driver_code("op", code, text);
            let mapped = OjasError::from(e);
            let class = match mapped {
                OjasError::CapacityExceeded { .. } => "capacity",
                OjasError::DeviceLost { .. } => "lost",
                OjasError::Backend { .. } => "backend",
                ref other => panic!("{code} {text:?}: {other:?}"),
            };
            let want = if code == CUDA_ERROR_OUT_OF_MEMORY {
                "capacity"
            } else if STICKY_DRIVER_CODES.contains(&code) {
                "lost"
            } else {
                "backend"
            };
            assert_eq!(class, want, "code {code} text {text:?}");
        }
    }
}

/// `with_live` touches only `Capacity`, keeps the last value, and keeps
/// u64 edges.
#[test]
fn with_live_is_total_and_idempotent() {
    for live in [0, 1, u64::MAX] {
        let e = CudaError::capacity_of("op", 10, 5)
            .with_live(1)
            .with_live(live);
        assert!(
            matches!(
                OjasError::from(e),
                OjasError::CapacityExceeded { requested: 10, cap: 5, live: l } if l == live
            ),
            "{live}"
        );
    }
    for e in [
        CudaError::invalid("op", "x"),
        CudaError::from_driver_code("op", 1, "x"),
        CudaError::NotCompiled,
    ] {
        let kind = e.kind();
        assert_eq!(e.with_live(9).kind(), kind);
    }
}

/// The runtime's real device OOM (status 2 from the driver) reaches the
/// caller as `CapacityExceeded` carrying no figure at all: requested 0,
/// cap 0, live 0. `live: 0` here is indistinguishable from an empty budget.
#[test]
#[ignore = "BREAK: a driver out-of-memory (status 2) converts to CapacityExceeded {requested: 0, cap: 0, live: 0}; nothing attaches the figures (error.rs from_driver_code + From)"]
fn a_driver_oom_carries_figures() {
    let e = CudaError::from_driver_code("cuMemAlloc", CUDA_ERROR_OUT_OF_MEMORY, "oom");
    match OjasError::from(e) {
        OjasError::CapacityExceeded {
            requested,
            cap,
            live,
        } => assert!(
            requested != 0 || cap != 0 || live != 0,
            "all three figures are 0"
        ),
        other => panic!("{other:?}"),
    }
}

/// The budget refusal on the runtime's own path
/// (`CudaRuntime::reserve` -> `AllocBudget::reserve`) is `Capacity` with
/// the figures only in its text: converted, `requested`, `cap` and `live`
/// are all 0 although the budget knew them.
#[test]
#[ignore = "BREAK: AllocBudget::reserve (budget.rs:71-86) flattens the budget's CapacityExceeded{requested,cap,live} into CudaError::capacity(text); From<CudaError> then reports 0/0/0. The lane's with_live covers check_free only"]
fn a_budget_refusal_through_the_runtime_view_keeps_its_figures() {
    let budget = AllocBudget::new(100).unwrap();
    let _held = budget.reserve(60, "held").unwrap();
    let err = budget.reserve(50, "w").map(|_| ()).unwrap_err();
    match OjasError::from(err) {
        OjasError::CapacityExceeded {
            requested,
            cap,
            live,
        } => {
            assert_eq!((requested, cap, live), (50, 100, 60));
        }
        other => panic!("{other:?}"),
    }
}

/// `CudaRuntime::reserve` charges the budget and then calls `check_free`
/// while still holding the reservation, so `check_free`'s `budget.used()`
/// includes the request it is refusing. This models that sequence on the
/// public budget view: the figure a refusal would carry is `held + request`,
/// not the `held` every other backend reports.
#[test]
#[ignore = "BREAK (by reading runtime.rs:573-574 and 580-592; modeled here, not run on a device): check_free's live includes the refused request's own reservation"]
fn the_live_figure_excludes_the_refused_request() {
    let budget = AllocBudget::new(1000).unwrap();
    let _held = budget.reserve(100, "held").unwrap();
    let request = 300;
    let reservation = budget.reserve(request, "w").unwrap();
    let seen_by_check_free = budget.used().unwrap();
    drop(reservation);
    assert_eq!(seen_by_check_free, 100, "live includes the refused request");
}

/// A refusal through the budget view releases nothing it did not take and
/// leaves the count where it was; zero and u64-edge requests do not wrap.
#[test]
fn budget_refusals_do_not_change_the_count() {
    let budget = AllocBudget::new(100).unwrap();
    let held = budget.reserve(60, "held").unwrap();
    for req in [41, 100, u64::MAX, u64::MAX - 59] {
        let r = budget.reserve(req, "w");
        assert!(matches!(r, Err(CudaError::Capacity { .. })), "{req}");
        assert_eq!(budget.used().unwrap(), 60, "{req}");
    }
    drop(held);
    assert_eq!(budget.used().unwrap(), 0);
}
