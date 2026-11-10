//! Test seams for the wgfadv adversarial tests (the hold slots of ft-84e2b74e).
//!
//! A per-thread hook runs at fixed points of [`WgpuContext::read_once`] and
//! may pass, fail the read with an error, or panic inside it. It is
//! thread-local, so a test arms exactly the thread whose read it means to
//! break, and a hook may block, to hold a lease while other reads proceed.
use super::*;
use std::cell::RefCell;

/// Where in a read the hook runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    /// The slot is leased, nothing recorded.
    AfterLease,
    /// `HoldLease::record` ran (the lease owns the folded slots), the hold
    /// itself is not yet recorded into the encoder.
    AfterRecord,
    /// Submitted, before the wait for the submission.
    BeforeWait,
    /// The submission completed, before the map is polled.
    AfterWait,
    /// The staging buffer is mapped, the host has not copied it yet.
    AfterMap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Act {
    Pass,
    Fail,
    Panic,
}

type Hook = Box<dyn FnMut(Stage) -> Act>;

thread_local! {
    static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
}

/// Arm this thread's reads with `hook`.
pub(crate) fn set_hook(hook: impl FnMut(Stage) -> Act + 'static) {
    HOOK.with(|c| *c.borrow_mut() = Some(Box::new(hook)));
}

pub(crate) fn clear_hook() {
    HOOK.with(|c| *c.borrow_mut() = None);
}

impl WgpuContext {
    /// Run this thread's hook at `stage`: an error for `Fail`, an unwind for
    /// `Panic`.
    pub(super) fn fire(&self, stage: Stage) -> Result<(), OjasError> {
        let act = HOOK.with(|c| {
            // Taken out while it runs, so a hook that blocks holds no borrow.
            let taken = c.borrow_mut().take();
            let mut hook = taken;
            let act = hook.as_mut().map_or(Act::Pass, |h| h(stage));
            let mut slot = c.borrow_mut();
            if slot.is_none() {
                *slot = hook;
            }
            act
        });
        match act {
            Act::Pass => Ok(()),
            Act::Fail => Err(self.failure(format!("injected {stage:?} failure"))),
            Act::Panic => panic!("injected {stage:?} panic"),
        }
    }

    /// `(free, orphans)` slot lists right now; a slot in neither is leased.
    pub(crate) fn slot_census(&self) -> (Vec<u32>, Vec<u32>) {
        let state = lock(&self.inner.slots.state);
        (
            state.free.clone(),
            state.orphans.values().copied().collect(),
        )
    }
}
