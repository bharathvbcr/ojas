//! `MetalBackend::set_mem_cap` moves the byte cap the device thread waits at,
//! so the `mem_cap` waits of a chain of unsynced calls can be counted at two
//! caps for the same chain (ft-580e6c40). Before, the cap was a function of the
//! budget alone and nothing outside the crate's own tests could change it.
//!
//! This binary holds this one test: the counts are deterministic in the pool
//! state only while nothing else allocates beside it.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::{Backend, Budget, OjasError, Tensor};
use ojas_metal::{MetalBackend, WaitCounts};

const MIB: u64 = 1 << 20;
/// Calls per chain: 40 MiB of outputs, past a 16 MiB cap, under 1 GiB.
const CHAIN: usize = 40;

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

#[test]
fn set_mem_cap_moves_the_waits_of_an_unsynced_chain() {
    // A 64 MiB budget opens with a memory cap of a quarter of it, 16 MiB.
    let m = ok("Metal", MetalBackend::new(Budget::new(64 * MIB)));
    let x = up(&m, &rand(&[(MIB / 4) as usize], 1, 1.0));

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
    let small = chain(&m, &x);
    assert!(
        small.mem_cap > default_a.mem_cap,
        "a 1 MiB cap waits more often than the 16 MiB default: {small:?} against {default_a:?}"
    );

    ok("set_mem_cap", m.set_mem_cap(1 << 30));
    let raised = chain(&m, &x);
    assert_eq!(
        (raised.mem_cap, raised.freed),
        (0, 0),
        "a 1 GiB cap is above the chain's 40 MiB: {raised:?}"
    );

    ok("set_mem_cap", m.set_mem_cap(16 * MIB));
    assert_eq!(
        chain(&m, &x),
        default_a,
        "restoring the default cap restores the default counts"
    );

    let refused = m.set_mem_cap(0);
    assert!(
        matches!(
            &refused,
            Err(OjasError::OutOfRange {
                op: "set_mem_cap",
                ..
            })
        ),
        "{refused:?}"
    );
    assert_eq!(
        chain(&m, &x),
        default_a,
        "a refused zero cap leaves the cap as it was"
    );
}
