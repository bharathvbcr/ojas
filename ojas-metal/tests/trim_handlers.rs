//! Trim handlers of a budget shared by several Metal backends, and the
//! refusals that keep a handler from deadlocking (ft-44871303).
#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use std::sync::Arc;
use std::thread;

use common::*;
use ojas_core::{Budget, BudgetTrimHandler, OjasError};
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
/// does while holding the handler lock: that panic poisons the lock.
#[derive(Debug)]
struct PanicsUnderLock;

impl BudgetTrimHandler for PanicsUnderLock {
    fn trim_memory(&self, _needed_bytes: u64) -> u64 {
        0
    }
    fn is_live(&self) -> bool {
        panic!("poison the handler lock, injected by the test")
    }
}

/// A poisoned handler lock is an error from `new`, not an `Ok` backend with
/// no handler installed.
#[test]
fn new_on_a_budget_with_a_poisoned_handler_lock_is_an_error() {
    let budget = Budget::new(CAP);
    ok("add", budget.add_trim_handler(Arc::new(PanicsUnderLock)));
    let b = budget.clone();
    let panicked = thread::spawn(move || {
        let _ = b.add_trim_handler(Arc::new(PanicsUnderLock));
    })
    .join();
    assert!(panicked.is_err(), "the lock was not poisoned");

    match MetalBackend::new(budget) {
        Err(OjasError::Poisoned) => {}
        Ok(_) => panic!("new returned Ok with no trim handler installed"),
        Err(other) => panic!("{other:?}"),
    }
}
