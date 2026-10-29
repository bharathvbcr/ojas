//! `MetalBackend::set_mem_cap` moves the byte cap the device thread waits at,
//! so the `mem_cap` waits of a chain of unsynced calls can be counted at two
//! caps for the same chain (ft-580e6c40). Before, the cap was a function of the
//! budget alone and nothing outside the crate's own tests could change it.
//!
//! The wait counts are per backend (its own device thread) and deterministic in
//! its pool state; run this binary with `--test-threads=1` anyway, so no other
//! test's allocations sit beside the one being counted.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::{Backend, Budget, OjasError, Tensor};
use ojas_metal::{MetalBackend, WaitCounts};

const MIB: u64 = 1 << 20;
/// Calls per chain: 40 MiB of outputs, past a 16 MiB cap, under a 64 MiB budget.
const CHAIN: usize = 40;

/// A backend whose 64 MiB budget opens with a memory cap of a quarter of it,
/// 16 MiB, and a 1 MiB tensor to run `silu` on.
fn open() -> (MetalBackend, Tensor) {
    let m = ok("Metal", MetalBackend::new(Budget::new(64 * MIB)));
    let x = up(&m, &rand(&[(MIB / 4) as usize], 1, 1.0));
    ok("sync", m.sync());
    (m, x)
}

/// A synced start, `CHAIN` unsynced `silu` calls on a 1 MiB tensor (outputs
/// dropped at once), and the wait counts of the chain alone.
fn chain(m: &MetalBackend, x: &Tensor) -> WaitCounts {
    ok("sync", m.sync());
    let before = m.wait_counts();
    for _ in 0..CHAIN {
        drop(ok("silu", m.silu_forward(x)));
    }
    let counts = m.wait_counts().since(&before);
    ok("sync", m.sync());
    counts
}

fn refused_out_of_range<T: std::fmt::Debug>(r: Result<T, OjasError>) -> bool {
    matches!(
        r,
        Err(OjasError::OutOfRange {
            op: "set_mem_cap",
            ..
        })
    )
}

#[test]
fn set_mem_cap_moves_the_waits_of_an_unsynced_chain() {
    let (m, x) = open();
    let default = ok("mem_cap", m.mem_cap());
    assert_eq!(default, 16 * MIB, "a quarter of the 64 MiB budget");

    // One chain warms the pool, so the counts below are in its steady state.
    let _warm = chain(&m, &x);
    let default_a = chain(&m, &x);
    let default_b = chain(&m, &x);
    assert!(
        default_a.mem_cap >= 1,
        "40 MiB of outputs past the 16 MiB default cap: {default_a:?}"
    );
    assert_eq!(
        default_a, default_b,
        "the counts of the same chain repeat from a synced start"
    );

    ok("set_mem_cap", m.set_mem_cap(MIB));
    assert_eq!(ok("mem_cap", m.mem_cap()), MIB);
    let small = chain(&m, &x);
    assert!(
        small.mem_cap > default_a.mem_cap,
        "a 1 MiB cap waits more often than the 16 MiB default: {small:?} against {default_a:?}"
    );

    // The budget itself, 64 MiB, is the largest cap accepted and is above the chain.
    ok("set_mem_cap", m.set_mem_cap(64 * MIB));
    let raised = chain(&m, &x);
    assert_eq!(
        (raised.mem_cap, raised.freed),
        (0, 0),
        "a 64 MiB cap is above the chain's 40 MiB: {raised:?}"
    );

    ok("set_mem_cap", m.set_mem_cap(default));
    assert_eq!(
        chain(&m, &x),
        default_a,
        "restoring the default cap restores the default counts"
    );
}

#[test]
fn a_zero_or_over_budget_cap_is_refused_and_leaves_the_cap_unchanged() {
    let (m, x) = open();
    let default = ok("mem_cap", m.mem_cap());
    let _warm = chain(&m, &x);
    let counts = chain(&m, &x);

    assert!(refused_out_of_range(m.set_mem_cap(0)));
    assert!(refused_out_of_range(m.set_mem_cap(64 * MIB + 1)));
    assert!(refused_out_of_range(m.set_mem_cap(u64::MAX)));
    assert_eq!(
        ok("mem_cap", m.mem_cap()),
        default,
        "refusals change nothing"
    );
    assert_eq!(
        chain(&m, &x),
        counts,
        "a refused cap leaves the waits as they were"
    );

    // The budget is accepted; one byte past it was not.
    ok("set_mem_cap", m.set_mem_cap(64 * MIB));
    assert_eq!(ok("mem_cap", m.mem_cap()), 64 * MIB);
}

/// The setter is a device command: lowering the cap below the bytes allocated
/// since the last waited commit, with work recorded, waits once, and returns.
#[test]
fn lowering_the_cap_with_work_pending_waits_once_and_does_not_hang() {
    let (m, x) = open();
    let before = m.wait_counts();
    // Five unsynced calls: 5 MiB allocated since the last wait, under the
    // 16 MiB cap, so nothing has waited yet.
    for _ in 0..5 {
        drop(ok("silu", m.silu_forward(&x)));
    }
    assert_eq!(
        m.wait_counts().since(&before),
        WaitCounts::default(),
        "five 1 MiB outputs wait for nothing under a 16 MiB cap"
    );
    ok("set_mem_cap", m.set_mem_cap(MIB));
    assert_eq!(
        m.wait_counts().since(&before),
        WaitCounts {
            mem_cap: 1,
            ..WaitCounts::default()
        },
        "5 MiB allocated is past the new 1 MiB cap: the setter's own commit waits once"
    );
    ok("sync", m.sync());
}

/// A 12 MiB budget (memory cap 3 MiB) with eight 1 MiB tensors dropped since
/// the last waited commit: 8 MiB of frees past the cap, so the next device
/// command waits first (`WaitCounts::freed`). Returns the backend, a small
/// tensor to run a command on, and the counts at that point.
fn with_frees_past_the_cap() -> (MetalBackend, Tensor, WaitCounts) {
    let m = ok("Metal", MetalBackend::new(Budget::new(12 * MIB)));
    let x = up(&m, &rand(&[64], 1, 1.0));
    let big: Vec<Tensor> = (0..8)
        .map(|j| up(&m, &rand(&[(MIB / 4) as usize], 2 + j, 1.0)))
        .collect();
    drop(ok("silu", m.silu_forward(&x)));
    // A waited commit zeroes the bytes freed since one: the drop below is the
    // only reason the next command would wait.
    ok("sync", m.sync());
    drop(big);
    let counts = m.wait_counts();
    (m, x, counts)
}

/// What the next device command does with those frees: one `freed` wait.
fn the_frees_are_still_pending(m: &MetalBackend, x: &Tensor, since: &WaitCounts) {
    drop(ok("silu", m.silu_forward(x)));
    assert_eq!(
        m.wait_counts().since(since),
        WaitCounts {
            freed: 1,
            ..WaitCounts::default()
        },
        "the 8 MiB of frees were still waiting for the next command"
    );
}

/// Reading the cap is not a device command (ft-580e6c40): it does not run the
/// freed-bytes check or the commit triggers, so it neither waits nor
/// dispatches, and the frees it left alone are still pending afterwards.
#[test]
fn reading_the_cap_issues_no_device_command() {
    let (m, x, counts) = with_frees_past_the_cap();
    let dispatches = m.dispatches();
    for _ in 0..3 {
        assert_eq!(ok("mem_cap", m.mem_cap()), 3 * MIB);
    }
    assert_eq!(
        m.wait_counts(),
        counts,
        "a read of the cap waited for a commit"
    );
    assert_eq!(m.dispatches(), dispatches);
    the_frees_are_still_pending(&m, &x, &counts);
}

/// A refused cap is refused at the boundary (ft-580e6c40): before the device
/// runs its freed-bytes check, so it recycles nothing and waits for nothing,
/// and the cap stays what it was.
#[test]
fn a_refused_cap_runs_no_recycle() {
    let (m, x, counts) = with_frees_past_the_cap();
    assert!(refused_out_of_range(m.set_mem_cap(0)));
    assert!(refused_out_of_range(m.set_mem_cap(12 * MIB + 1)));
    assert!(refused_out_of_range(m.set_mem_cap(u64::MAX)));
    assert_eq!(
        m.wait_counts(),
        counts,
        "a refused set waited for a commit before it refused"
    );
    assert_eq!(ok("mem_cap", m.mem_cap()), 3 * MIB);
    the_frees_are_still_pending(&m, &x, &counts);
}

/// The cap belongs to the device thread, so every `with_budget` handle reads
/// and moves the same one, and the bound is the opening backend's budget, not
/// the handle's.
#[test]
fn the_cap_is_shared_across_with_budget_handles() {
    let (m, _x) = open();
    let small = m.with_budget(Budget::new(8 * MIB));
    ok("set_mem_cap", small.set_mem_cap(2 * MIB));
    assert_eq!(ok("mem_cap", m.mem_cap()), 2 * MIB);
    assert_eq!(ok("mem_cap", small.mem_cap()), 2 * MIB);
    // 32 MiB is above the handle's 8 MiB budget and inside the opening 64 MiB one.
    ok("set_mem_cap", small.set_mem_cap(32 * MIB));
    assert_eq!(ok("mem_cap", m.mem_cap()), 32 * MIB);
    assert!(refused_out_of_range(small.set_mem_cap(64 * MIB + 1)));
    assert_eq!(ok("mem_cap", m.mem_cap()), 32 * MIB);
}
