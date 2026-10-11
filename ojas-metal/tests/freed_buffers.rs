//! A dropped tensor's charge leaves the budget at once, but its buffer stays
//! on the device until a commit after the drop completes. Frees past the
//! memory cap since the last waited commit make the next command wait first
//! (`WaitCounts::freed`), so what the device holds past the budget's live
//! bytes stays under the memory cap of frees plus the pool cap.
//!
//! The device's allocated size is the whole process's, so this binary holds
//! this one test and nothing else allocates beside it.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::{Backend, Budget, Tensor};
use ojas_metal::{MetalBackend, WaitCounts};

const MIB: u64 = 1 << 20;

/// A 12 MiB budget opens with a memory cap and a pool cap of a quarter of
/// it, 3 MiB each. Eight dropped 1 MiB tensors are past the memory cap, so
/// the next command waits once and the recycle leaves at most the pool cap
/// of them cached. Before, nothing counted frees: all eight stayed on the
/// device across the next command.
#[test]
fn frees_past_the_memory_cap_are_recycled_before_the_next_command() {
    let m = ok("Metal", MetalBackend::new(Budget::new(12 * MIB)));
    let pool_cap = ok("memory", m.memory()).pool_cache_cap;
    assert_eq!(pool_cap, 3 * MIB);
    let device = || ok("memory", m.memory()).allocated;
    let x = up(&m, &rand(&[64], 1, 1.0));
    ok("sync", m.sync());
    let held = device();
    let big: Vec<Tensor> = (0..8)
        .map(|j| up(&m, &rand(&[(MIB / 4) as usize], 2 + j, 1.0)))
        .collect();
    // A sync with work recorded is a waited commit, which zeroes the bytes
    // allocated since one: the drops below are the only reason to wait.
    drop(ok("silu", m.silu_forward(&x)));
    ok("sync", m.sync());
    assert!(
        device() >= held + 8 * MIB,
        "the eight tensors are on the device"
    );

    let before = m.wait_counts();
    drop(big);
    let y = ok("silu", m.silu_forward(&x));
    let after = device();
    // The pool keeps at most its cap; y and its rounding take under 1 MiB.
    assert!(
        after < held + pool_cap + MIB,
        "8 MiB freed past a 3 MiB cap: the device holds {} bytes past the {held} before them",
        after.saturating_sub(held)
    );
    assert_eq!(
        m.wait_counts().since(&before),
        WaitCounts {
            freed: 1,
            ..WaitCounts::default()
        }
    );

    // Frees under the cap wait for nothing.
    let small: Vec<Tensor> = (0..2).map(|j| up(&m, &rand(&[64], 20 + j, 1.0))).collect();
    ok("sync", m.sync());
    let before = m.wait_counts();
    drop(small);
    drop(ok("silu", m.silu_forward(&y)));
    assert_eq!(m.wait_counts().since(&before), WaitCounts::default());
    ok("sync", m.sync());
}
