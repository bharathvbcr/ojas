//! `MetalBackend::waits` against tessl's own commit counter, so the wait
//! count cannot hide a commit tessl made on its own. tessl's counters are
//! process-wide, so this binary holds one test.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::Backend;
use tessl::infer_trace;

#[test]
fn recording_commits_nothing_until_sync_and_overlap_commits_do_not_wait() {
    let m = metal();
    let x = up(&m, &rand(&[64], 1, 1.0));
    ok("settle", m.sync());
    infer_trace::set_enabled(true);
    infer_trace::reset_token_counters();
    let w0 = m.waits();
    // A few ops, fewer dispatches than one overlap commit takes.
    let mut y = x.clone();
    for _ in 0..20 {
        y = ok("silu", m.silu_forward(&y));
    }
    assert_eq!(infer_trace::snapshot().commits, 0, "recording committed");
    assert_eq!(m.waits(), w0);
    ok("sync", m.sync());
    assert_eq!(infer_trace::snapshot().commits, 1, "sync commits once");
    assert_eq!(m.waits() - w0, 1);
    // Enough ops for several overlap commits: tessl commits, nobody waits.
    infer_trace::reset_token_counters();
    let w1 = m.waits();
    for _ in 0..600 {
        y = ok("silu", m.silu_forward(&y));
    }
    let overlapped = infer_trace::snapshot().commits;
    assert!(overlapped >= 2, "expected overlap commits, saw {overlapped}");
    assert_eq!(m.waits(), w1, "overlap commits must not wait");
    ok("sync", m.sync());
    assert_eq!(m.waits() - w1, 1);
    assert_eq!(infer_trace::snapshot().commits, overlapped + 1);
    infer_trace::set_enabled(false);
    assert!(down(&y).iter().all(|v| v.is_finite()));
}
