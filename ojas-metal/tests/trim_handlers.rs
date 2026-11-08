//! Trim handlers of a budget shared by several Metal backends, and the
//! refusals that keep a handler from deadlocking (ft-44871303).
#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use common::*;
use ojas_core::{BackendCore, Budget, BudgetTrimHandler};
use ojas_metal::MetalBackend;

#[test]
fn with_budget_installs_this_pool_once() {
    let m = metal();
    let other = Budget::new(CAP);
    assert!(other.trim_handlers().is_empty());
    let _shared = m.with_budget(other.clone());
    assert_eq!(other.trim_handlers().len(), 1, "no handler installed");
    // The same link on the same budget again is still one pool.
    let _again = m.with_budget(other.clone());
    assert_eq!(other.trim_handlers().len(), 1, "one pool registered twice");
    // A child already sees it through its parent.
    let child = other.child(CAP);
    let _on_child = m.with_budget(child.clone());
    assert_eq!(child.trim_handlers().len(), 1);
    assert!(child.trim_handler().is_some());
}

/// A second backend on a budget that already has a handler for another pool
/// leaves both registered, so pressure trims both.
#[test]
fn two_backends_on_one_budget_register_both_pools() {
    let budget = Budget::new(CAP);
    let a = ok("a", MetalBackend::new(budget.clone()));
    assert_eq!(budget.trim_handlers().len(), 1);
    let _b = ok("b", MetalBackend::new(budget.clone()));
    assert_eq!(
        budget.trim_handlers().len(),
        2,
        "the second replaced the first"
    );
    drop(a);
    let _c = ok("c", MetalBackend::new(budget.clone()));
    assert_eq!(
        budget.trim_handlers().len(),
        2,
        "the dropped backend's handler stayed"
    );
}

/// Panics when asked whether its pool is alive, which `add_trim_handler`
/// does while holding the handler lock: that panic poisons the lock. It
/// panics once; afterwards it is an ordinary live handler.
#[derive(Debug)]
struct PanicsUnderLock(AtomicBool);

impl PanicsUnderLock {
    fn armed() -> Arc<Self> {
        Arc::new(Self(AtomicBool::new(true)))
    }
}

impl BudgetTrimHandler for PanicsUnderLock {
    fn trim_memory(&self, _needed_bytes: u64) -> u64 {
        0
    }
    fn is_live(&self) -> bool {
        if self.0.swap(false, Ordering::SeqCst) {
            panic!("poison the handler lock, injected by the test");
        }
        true
    }
}

/// A poisoned handler lock is recovered, not a permanent refusal: a later
/// `add_trim_handler` installs its handler and `new` on that budget succeeds
/// with its handler registered (ft-44871303).
#[test]
fn a_poisoned_handler_lock_does_not_refuse_new_backends() {
    let budget = Budget::new(CAP);
    budget.add_trim_handler(PanicsUnderLock::armed());
    let b = budget.clone();
    let panicked = thread::spawn(move || {
        b.add_trim_handler(PanicsUnderLock::armed());
    })
    .join();
    assert!(panicked.is_err(), "the lock was not poisoned");

    let m = ok("new", MetalBackend::new(budget.clone()));
    assert!(
        budget
            .trim_handlers()
            .iter()
            .any(|h| h.pool_key().is_some()),
        "new returned Ok with no trim handler installed"
    );
    let shared = m.with_budget(budget.clone());
    ok("sync", shared.sync());
}

/// A dropped backend's handler is not listed, without waiting for the next
/// registration to prune it.
#[test]
fn a_dropped_backend_is_not_listed_as_a_trim_handler() {
    let budget = Budget::new(CAP);
    let a = ok("a", MetalBackend::new(budget.clone()));
    let _b = ok("b", MetalBackend::new(budget.clone()));
    assert_eq!(budget.trim_handlers().len(), 2);
    drop(a);
    assert_eq!(
        budget.trim_handlers().len(),
        1,
        "the dropped backend's handler was still listed"
    );
}
