//! tessl's buffer pool charging an `ojas_core::Budget` through
//! `BudgetLedger`: what is charged is the bucket tessl allocates, and a
//! charge is refused exactly when the buckets would pass the cap.
//!
//! Needs a Metal device. Nothing here goes through `MetalBackend`: the ledger
//! is installed on a bare tessl runtime, because `device.rs` does not use it
//! yet (plan phase P3a).

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::sync::Arc;

use ojas_core::{Budget, OjasError};
use ojas_metal::BudgetLedger;
use tessl::{BufferKind, GpuRuntime, PoolResident};

const MIB: usize = 1 << 20;

fn bucket(n: usize, kind: BufferKind) -> u64 {
    GpuRuntime::allocated_bytes_for(n, kind)
}

/// `(requested, cap, live)` of a pool error that reads as a refused charge.
/// `OjasError` has no `PartialEq`, so the numbers are compared, not the error.
fn refusal(error: &str) -> Option<(u64, u64, u64)> {
    match BudgetLedger::capacity_error(error) {
        Some(OjasError::CapacityExceeded {
            requested,
            cap,
            live,
        }) => Some((requested, cap, live)),
        _ => None,
    }
}

/// A tessl runtime whose pool charges a child pair of `root`.
fn runtime_on(root: &Budget) -> (Arc<GpuRuntime>, Arc<BudgetLedger>) {
    let rt = GpuRuntime::new().expect("Metal device");
    let ledger = BudgetLedger::new(root, u64::MAX);
    rt.set_pool_ledger(Some(ledger.as_pool_ledger()))
        .expect("a fresh runtime takes a ledger");
    (rt, ledger)
}

#[test]
fn what_is_charged_is_the_allocated_bucket() {
    let root = Budget::new(u64::MAX / 2);
    let (rt, ledger) = runtime_on(&root);
    let sizes = [
        (5_000, BufferKind::Cold),
        (300_000, BufferKind::Cold),
        (MIB, BufferKind::Cold),
        (MIB + 1, BufferKind::Cold),
        (100_000, BufferKind::Hot),
        (3 * MIB, BufferKind::Hot),
    ];
    let mut held = Vec::new();
    let mut total = 0;
    for (n, kind) in sizes {
        held.push(rt.alloc_buffer_kind(n, kind).unwrap());
        total += bucket(n, kind);
        assert_eq!(ledger.live_bytes(), total, "{n} {kind:?}");
        assert_eq!(root.live_bytes().unwrap(), total, "{n} {kind:?}");
        assert_eq!(rt.pool_resident().live_bytes, total, "{n} {kind:?}");
    }
    assert_eq!(ledger.cached_bytes(), 0);

    // Dropped, then one allocation of a bucket none of them has: the pool
    // finds the GPU idle and releases them. Cold ones park on the free list
    // and stay charged to the cache child; Hot ones are destroyed.
    drop(held);
    let probe = rt.alloc_buffer_kind(1, BufferKind::Cold).unwrap();
    let cold: u64 = sizes
        .iter()
        .filter(|(_, k)| *k == BufferKind::Cold)
        .map(|&(n, k)| bucket(n, k))
        .sum();
    assert_eq!(ledger.live_bytes(), bucket(1, BufferKind::Cold));
    assert_eq!(ledger.cached_bytes(), cold);
    assert_eq!(
        root.live_bytes().unwrap(),
        cold + bucket(1, BufferKind::Cold)
    );
    assert_eq!(
        rt.pool_resident(),
        PoolResident {
            live_bytes: bucket(1, BufferKind::Cold),
            cached_bytes: cold
        }
    );
    assert_eq!(ledger.unmatched_releases(), 0);
    drop(probe);
}

#[test]
fn refusal_is_exactly_at_the_bucket_total() {
    let (a, b) = (MIB, 100_000);
    let (bucket_a, bucket_b) = (bucket(a, BufferKind::Cold), bucket(b, BufferKind::Hot));
    let total = bucket_a + bucket_b;

    // One byte short: the second buffer is refused with the cap's numbers.
    let root = Budget::new(total - 1);
    let (rt, ledger) = runtime_on(&root);
    let _a = rt.alloc_buffer_kind(a, BufferKind::Cold).unwrap();
    let err = rt
        .alloc_buffer_kind(b, BufferKind::Hot)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        refusal(&err),
        Some((bucket_b, total - 1, bucket_a)),
        "{err}"
    );
    assert_eq!(
        root.live_bytes().unwrap(),
        bucket_a,
        "a refusal charges nothing"
    );
    assert_eq!(ledger.live_bytes(), bucket_a);

    // Exactly the total: both fit, and the smallest bucket does not.
    let root = Budget::new(total);
    let (rt, ledger) = runtime_on(&root);
    let first = rt.alloc_buffer_kind(a, BufferKind::Cold).unwrap();
    let _second = rt.alloc_buffer_kind(b, BufferKind::Hot).unwrap();
    assert_eq!(root.live_bytes().unwrap(), total);
    let err = rt
        .alloc_buffer_kind(1, BufferKind::Cold)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        refusal(&err),
        Some((bucket(1, BufferKind::Cold), total, total)),
        "{err}"
    );
    assert_eq!(root.live_bytes().unwrap(), total);
    assert_eq!(root.peak_bytes(), total, "the refusal raised no peak");

    // The first buffer is dropped and parks on the free list: still
    // resident, still charged. The next allocation of another bucket is
    // refused once, the pool trims the cache and retries, and it fits.
    drop(first);
    let small = rt.alloc_buffer_kind(1, BufferKind::Cold).unwrap();
    assert_eq!(ledger.cached_bytes(), 0, "the ladder trimmed the free list");
    assert_eq!(
        root.live_bytes().unwrap(),
        bucket_b + bucket(1, BufferKind::Cold)
    );
    assert_eq!(ledger.unmatched_releases(), 0);
    drop(small);
}

#[test]
fn dropping_the_runtime_returns_every_charge_to_the_budget() {
    let root = Budget::new(u64::MAX / 2);
    let (rt, ledger) = runtime_on(&root);
    drop(rt.alloc_buffer_kind(MIB, BufferKind::Cold).unwrap());
    let outlives = rt.alloc_buffer_kind(2 * MIB, BufferKind::Cold).unwrap();
    drop(rt.alloc_buffer_kind(3 * MIB, BufferKind::Hot).unwrap());
    assert!(root.live_bytes().unwrap() > 0);

    drop(rt);
    // Only the buffer that is still held outlives its runtime.
    assert_eq!(
        root.live_bytes().unwrap(),
        bucket(2 * MIB, BufferKind::Cold)
    );
    drop(outlives);
    assert_eq!(root.live_bytes().unwrap(), 0);
    assert_eq!(ledger.live_bytes() + ledger.cached_bytes(), 0);
    assert_eq!(ledger.unmatched_releases(), 0);
}
