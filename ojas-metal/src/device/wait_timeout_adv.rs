//! e7adv attack 5: how the worker types an error that reaches its serve seam.
//!
//! `Worker::lost_if_poisoned` retypes a leftover tessl timeout string as the
//! retryable [`OjasError::WaitTimeout`], and keeps every other error's kind.
//! These tests drive it with the real tessl text, a text an ojas path wrapped
//! with a prefix, and errors that are not timeouts, on a healthy and on a
//! poisoned runtime.

use super::*;
use ojas_core::BackendId;

/// The shape of tessl's own budget-exceeded error (`event_wait_timeout_error`
/// in `tessl/src/runtime.rs`); `a_real_timeout_text_is_wait_timeout...` asserts
/// tessl's predicate accepts it, so a drift in the prefix fails loudly there.
const REAL: &str = "Metal 4 SharedEvent wait budget exceeded (timed out): waited 1 ms for lifetime event value 7 (reached 3, submitted 7); the GPU work may only be slow, so the runtime is not poisoned: retry";

fn worker() -> Worker {
    Worker::open(Arc::new(Waits::default()), Budget::new(1 << 40)).expect("Metal device")
}

fn backend(detail: &str) -> OjasError {
    metal_err(detail.to_string())
}

fn is_backend(e: &OjasError) -> bool {
    matches!(
        e,
        OjasError::Backend {
            id: BackendId::Metal,
            ..
        }
    )
}

/// A real timeout text is the typed timeout carrying the runtime's budget; a
/// text an ojas path wrapped with a prefix is not recognised (tessl's predicate
/// is `starts_with`), so it stays a plain `Backend` error, never `Ok` and never
/// `DeviceLost`; a non-timeout `Backend` error stays `Backend`; typed errors
/// pass through untouched.
#[test]
fn a_real_timeout_text_is_wait_timeout_and_everything_else_keeps_its_kind() {
    assert!(
        GpuRuntime::is_event_wait_timeout(REAL),
        "the literal no longer matches tessl's predicate"
    );
    let w = worker();
    w.rt.set_event_wait_ms(1).expect("budget");

    let real = w.lost_if_poisoned(backend(REAL));
    assert!(
        matches!(
            &real,
            OjasError::WaitTimeout {
                backend: BackendId::Metal,
                waited_ms: 1,
                ..
            }
        ),
        "a real timeout text: {real:?}"
    );
    assert_eq!(w.waits.timeouts(), 1);

    let wrapped = w.lost_if_poisoned(backend(&format!("adamw_step: {REAL}")));
    assert!(
        is_backend(&wrapped),
        "a prefixed timeout text must stay Backend (not retryable, not lost, not Ok): {wrapped:?}"
    );
    assert_eq!(
        w.waits.timeouts(),
        1,
        "a wrapped text was counted as a timeout"
    );

    let mid = w.lost_if_poisoned(backend(&format!("{REAL} ... wait budget exceeded")));
    assert!(
        matches!(mid, OjasError::WaitTimeout { .. }),
        "the prefix decides, not the rest: {mid:?}"
    );

    let plain = w.lost_if_poisoned(backend("newBuffer(64) failed"));
    assert!(is_backend(&plain), "a plain Backend error: {plain:?}");

    let near = w.lost_if_poisoned(backend("Metal 4 SharedEvent wait timed out (poison)"));
    assert!(is_backend(&near), "a near-miss text: {near:?}");

    for typed in [
        OjasError::Poisoned,
        OjasError::CapacityExceeded {
            requested: 1,
            cap: 1,
            live: 1,
        },
        OjasError::wait_timeout(BackendId::Metal, 9, "already typed"),
    ] {
        let kept = format!("{typed:?}");
        let out = w.lost_if_poisoned(typed);
        assert_eq!(format!("{out:?}"), kept, "a typed error changed kind");
    }
    w.rt.set_event_wait_ms(120_000).expect("budget");
}

/// On a poisoned runtime the poison wins: even a genuine timeout text is the
/// loss, because the runtime will not run it again.
#[test]
fn on_a_poisoned_runtime_even_a_real_timeout_text_is_device_lost() {
    let w = worker();
    w.rt.poison_as_shared_event_timeout_for_test();
    assert!(w.rt.is_poisoned());
    for text in [REAL, "newBuffer(64) failed", "anything"] {
        let out = w.lost_if_poisoned(backend(text));
        assert!(
            matches!(
                &out,
                OjasError::DeviceLost {
                    backend: BackendId::Metal,
                    ..
                }
            ),
            "{text:?} on a poisoned runtime: {out:?}"
        );
    }
    assert_eq!(w.waits.timeouts(), 0, "a loss was counted as a timeout");
}
