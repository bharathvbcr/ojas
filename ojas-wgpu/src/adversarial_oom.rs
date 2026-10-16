//! Adversarial tests for card ft-cab0fdf1 on wgpu: the state a refused
//! allocation leaves behind (device, budget, pool, error scopes), and what
//! `live` reports.
//!
//! Tests marked `#[ignore = "BREAK: ..."]` assert the property the card
//! promises and currently fail.

use super::*;
use crate::WgpuBackend;
use ojas_core::{Backend, Budget, OjasError, Tensor};

fn host(n: usize, budget: &Budget) -> Tensor {
    Tensor::from_f32(&vec![1.0; n], &[n], budget).expect("host tensor")
}

/// After the retried allocation is refused twice, the context is still a
/// working device: no wgpu error is pending, the pool is empty and takes
/// buffers again, and a submit/read round trip succeeds.
#[test]
fn a_refused_allocation_leaves_the_device_usable() {
    let ctx = WgpuContext::open().expect("wgpu adapter");
    let keep = ctx.tensor_buffer(1 << 20).expect("alloc");
    drop(ctx.tensor_buffer(1 << 20).expect("alloc"));
    ctx.fail_next_allocs(2);
    let err = ctx.tensor_buffer(2 << 20).expect_err("two refusals");
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");
    ctx.check_errors()
        .expect("no stray wgpu error after a refusal");
    assert_eq!(ctx.pooled_bytes(), 0, "the retry released the pool");
    drop(
        ctx.tensor_buffer(2 << 20)
            .expect("the device allocates again"),
    );
    ctx.sync().expect("and still submits and reads");
    ctx.check_errors().expect("clean after the round trip");
    drop(keep);
}

/// A refusal through the backend charges nothing: the budget's live bytes
/// are what they were, a later upload fits the room the refused one asked
/// for, and releasing it returns the budget to where it started.
#[test]
fn a_refused_upload_restores_the_budget() {
    let budget = Budget::new(1 << 22);
    let be = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let resident = be.upload(&host(1 << 16, &budget)).expect("resident upload");
    let live = budget.live_bytes().unwrap();
    for failures in [1, 2] {
        let src = host(1 << 18, &budget);
        be.context().fail_next_allocs(failures);
        let r = be.upload(&src);
        match (failures, &r) {
            (1, Ok(_)) => {}
            (_, Err(OjasError::CapacityExceeded { .. })) if failures >= 2 => {}
            _ => panic!("{failures} injected failures: {:?}", r.map(|_| ())),
        }
        drop(r);
        drop(src);
        assert_eq!(
            budget.live_bytes().unwrap(),
            live,
            "a refusal left a charge"
        );
        be.context().check_errors().expect("no stray error");
    }
    assert_eq!(budget.live_bytes().unwrap(), live);
    let again = be.upload(&host(1 << 18, &budget)).expect("room is back");
    drop(again);
    drop(resident);
    assert_eq!(budget.live_bytes().unwrap(), 0);
}

/// Criterion 3 asks for `CapacityExceeded` "with live bytes from the
/// budget". `out_of_memory` reports the allocator's total instead (0 when
/// it cannot say), so a refusal under a budget holding bytes shows a figure
/// that is not the budget's.
#[test]
#[ignore = "BREAK: out_of_memory reports generate_allocator_report().total_reserved_bytes (0 when unavailable), not the budget's live bytes the card asks for (context.rs:928-939)"]
fn the_live_figure_is_the_budgets() {
    let budget = Budget::new(1 << 22);
    let be = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let _resident = be.upload(&host(1 << 16, &budget)).expect("resident");
    let live = budget.live_bytes().unwrap();
    assert!(live > 0);
    be.context().fail_next_allocs(2);
    let err = be.upload(&host(1 << 18, &budget)).map(|_| ()).unwrap_err();
    match err {
        OjasError::CapacityExceeded { live: got, .. } => assert_eq!(got, live),
        other => panic!("{other:?}"),
    }
}

/// `live` of 0 must mean an empty device. When the allocator report is
/// unavailable the figure is 0 anyway, so a caller cannot tell "empty" from
/// "unknown"; with a resident 4 MiB buffer the figure must not be 0.
#[test]
#[ignore = "BREAK: on this host (wgpu over Metal) generate_allocator_report() is None, so every CapacityExceeded.live is 0 whatever is resident (context.rs:933-937)"]
fn live_is_not_zero_beside_a_resident_buffer() {
    let ctx = WgpuContext::open().expect("wgpu adapter");
    let _resident = ctx.tensor_buffer(4 << 20).expect("alloc");
    ctx.fail_next_allocs(2);
    let err = ctx.tensor_buffer(1 << 20).expect_err("refused");
    match err {
        OjasError::CapacityExceeded { live, .. } => assert!(
            live >= 4 << 20,
            "live {live} with 4 MiB resident (0 means the report was unavailable)"
        ),
        other => panic!("{other:?}"),
    }
}

/// A request past the device's buffer limit is a validation error, which an
/// out-of-memory scope does not catch: `alloc` hands back a buffer wgpu has
/// already marked invalid, and the error surfaces at a later, unrelated
/// call. Every backend caller checks `check_bytes` first, so this is a gap
/// in the layer below them, not a reachable fault today.
#[test]
#[ignore = "BREAK (latent, callers guard it): alloc() of a size past max_buffer_size returns Ok(invalid buffer); the error is reported by the next unrelated check_errors as Backend"]
fn a_request_past_the_buffer_limit_is_refused_at_the_alloc() {
    let cap = wgpu::Limits {
        max_buffer_size: 1 << 14,
        ..wgpu::Limits::default()
    };
    let ctx = WgpuContext::open_capped(&cap).expect("wgpu adapter");
    let r = ctx.tensor_buffer(1 << 16);
    assert!(
        matches!(r, Err(OjasError::CapacityExceeded { .. })),
        "alloc returned {:?}",
        r.as_ref().map(|_| "an Ok buffer")
    );
}

/// Every refusal path pops the scope it pushed: counted pops advance by one
/// per real (non-injected) creation, and a later creation sees no stale
/// scope capturing its error.
#[test]
fn real_creations_pop_exactly_one_scope() {
    let ctx = WgpuContext::open().expect("wgpu adapter");
    let before = ctx.stats().scope_pops;
    for i in 1..=8u64 {
        drop(ctx.tensor_buffer(4096 * i).expect("alloc"));
    }
    ctx.trim_pool();
    let after = ctx.stats().scope_pops;
    assert_eq!(after - before, 8, "one pop per creation");
    ctx.check_errors().expect("clean");
}
