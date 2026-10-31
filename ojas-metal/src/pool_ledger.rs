//! tessl's pool ledger over `ojas_core::Budget` (decision D3: the budget
//! counts what is resident).
//!
//! tessl's `BufferPool` charges the bucket length of every pool buffer to a
//! [`tessl::PoolLedger`] before it asks Metal for the memory. [`BudgetLedger`]
//! is that ledger: two children of one parent [`Budget`],
//!
//! - `live`, the buffers handed out or dropped and waiting for the GPU, and
//! - `cache`, the buffers parked on tessl's free list,
//!
//! so the free-list cache is visible beside the live bytes and both count
//! against the parent's cap. A refused charge is
//! [`OjasError::CapacityExceeded`] with the refusing level's numbers; the
//! pool has already trimmed its free list and retried once by then, and it
//! never waits on the GPU. [`BudgetLedger::capacity_error`] reads tessl's
//! refusal text into that error through tessl's own reader.
//!
//! Nothing in this crate installs it yet: `device.rs` still runs its own
//! `Book` / `mem_cap` / `classify_alloc` layer. The cut that replaces them is
//! P3a of the unification plan, after the Metal kernel lanes land.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use ojas_core::{Budget, BudgetTrimHandler, OjasError, Reservation};
use tessl::{LedgerClass, LedgerRefused, PoolLedger};

/// The reservations held for each balance, by the size they were taken at.
///
/// A `Budget` is charged through a [`Reservation`] that releases when
/// dropped, and a reservation cannot be split. tessl releases exactly the
/// bucket lengths it charged, so a release finds a reservation of its own
/// size.
#[derive(Default)]
struct Held {
    live: HashMap<u64, Vec<Reservation>>,
    cached: HashMap<u64, Vec<Reservation>>,
    /// Releases for a size with no reservation held: a charge the pool never
    /// made. A correct pool leaves this at zero.
    unmatched: u64,
}

impl Held {
    fn of(&mut self, class: LedgerClass) -> &mut HashMap<u64, Vec<Reservation>> {
        match class {
            LedgerClass::Live => &mut self.live,
            LedgerClass::Cached => &mut self.cached,
        }
    }
}

#[derive(Debug)]
struct NoOpTrim;

impl BudgetTrimHandler for NoOpTrim {
    fn trim_memory(&self, _needed_bytes: u64) -> u64 {
        0
    }
}

/// A [`tessl::PoolLedger`] over two children of one [`Budget`].
pub struct BudgetLedger {
    live: Budget,
    cache: Budget,
    held: Mutex<Held>,
}

impl BudgetLedger {
    /// A ledger charging `root`. The live child may take all of `root`'s cap;
    /// the cache child is capped at `cache_cap_bytes` (tessl bounds the free
    /// list by its own cache cap as well).
    pub fn new(root: &Budget, cache_cap_bytes: u64) -> Arc<Self> {
        let live = root.child(root.cap_bytes());
        let cache = root.child(cache_cap_bytes);

        // Child budgets must not delegate to root's trim handler.
        // BufferPool::charge_fresh already drives its own ladder,
        // so try_charge must fail fast without recursing back to the device thread.
        let noop: Arc<dyn BudgetTrimHandler> = Arc::new(NoOpTrim);
        live.set_trim_handler(Arc::clone(&noop));
        cache.set_trim_handler(noop);

        Arc::new(Self {
            live,
            cache,
            held: Mutex::new(Held::default()),
        })
    }

    /// This ledger as the trait object tessl installs.
    pub fn as_pool_ledger(self: &Arc<Self>) -> Arc<dyn PoolLedger> {
        self.clone()
    }

    /// Bytes charged for buffers handed out or waiting for the GPU.
    pub fn live_bytes(&self) -> u64 {
        self.live.live_bytes().unwrap_or(0)
    }

    /// Bytes charged for buffers parked on tessl's free list.
    pub fn cached_bytes(&self) -> u64 {
        self.cache.live_bytes().unwrap_or(0)
    }

    /// Releases that matched no charge. Zero unless the pool is wrong.
    pub fn unmatched_releases(&self) -> u64 {
        self.lock().unmatched
    }

    /// The error for a pool allocation failure that `error` (a tessl pool
    /// message) reports as a refused charge; `None` for any other message.
    pub fn capacity_error(error: &str) -> Option<OjasError> {
        LedgerRefused::parse(error).map(Self::refusal_error)
    }

    /// A refused charge as the error a caller sees.
    pub fn refusal_error(refused: LedgerRefused) -> OjasError {
        OjasError::CapacityExceeded {
            requested: refused.requested,
            cap: refused.cap,
            live: refused.live,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Held> {
        self.held.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn budget(&self, class: LedgerClass) -> &Budget {
        match class {
            LedgerClass::Live => &self.live,
            LedgerClass::Cached => &self.cache,
        }
    }

    /// The refusal for a reservation `OjasError` that is not a plain cap
    /// hit (an overflowing counter): numbers that cannot be exceeded.
    fn refused(&self, class: LedgerClass, bytes: u64, error: OjasError) -> LedgerRefused {
        match error {
            OjasError::CapacityExceeded {
                requested,
                cap,
                live,
            } => LedgerRefused {
                requested,
                cap,
                live,
            },
            _ => LedgerRefused {
                requested: bytes,
                cap: self.budget(class).cap_bytes(),
                live: u64::MAX,
            },
        }
    }
}

impl PoolLedger for BudgetLedger {
    fn try_charge(&self, class: LedgerClass, bytes: u64) -> Result<(), LedgerRefused> {
        let reservation = self
            .budget(class)
            .try_reserve(bytes)
            .map_err(|e| self.refused(class, bytes, e))?;
        self.lock()
            .of(class)
            .entry(bytes)
            .or_default()
            .push(reservation);
        Ok(())
    }

    fn release(&self, class: LedgerClass, bytes: u64) {
        // Dropped outside the lock: dropping releases the charge up the chain.
        let reservation = {
            let mut held = self.lock();
            let popped = held.of(class).get_mut(&bytes).and_then(Vec::pop);
            if popped.is_none() {
                held.unmatched += 1;
            }
            popped
        };
        drop(reservation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KIB: u64 = 1024;

    #[test]
    fn charges_and_releases_balance_per_class() {
        let root = Budget::new(64 * KIB);
        let ledger = BudgetLedger::new(&root, 64 * KIB);
        ledger.try_charge(LedgerClass::Live, 16 * KIB).unwrap();
        ledger.try_charge(LedgerClass::Live, 16 * KIB).unwrap();
        ledger.try_charge(LedgerClass::Cached, 8 * KIB).unwrap();
        assert_eq!(ledger.live_bytes(), 32 * KIB);
        assert_eq!(ledger.cached_bytes(), 8 * KIB);
        assert_eq!(
            root.live_bytes().unwrap(),
            40 * KIB,
            "the parent sees both balances"
        );

        ledger.release(LedgerClass::Live, 16 * KIB);
        ledger.release(LedgerClass::Cached, 8 * KIB);
        assert_eq!(ledger.live_bytes(), 16 * KIB);
        assert_eq!(ledger.cached_bytes(), 0);
        assert_eq!(root.live_bytes().unwrap(), 16 * KIB);
        assert_eq!(ledger.unmatched_releases(), 0);
    }

    #[test]
    fn refusal_is_exactly_at_the_parent_cap_and_charges_nothing() {
        let root = Budget::new(48 * KIB);
        let ledger = BudgetLedger::new(&root, 48 * KIB);
        ledger.try_charge(LedgerClass::Live, 32 * KIB).unwrap();
        ledger.try_charge(LedgerClass::Cached, 16 * KIB).unwrap();
        // The cap is full across both children.
        let refused = ledger.try_charge(LedgerClass::Live, 1).unwrap_err();
        assert_eq!(
            refused,
            LedgerRefused {
                requested: 1,
                cap: 48 * KIB,
                live: 48 * KIB
            }
        );
        assert_eq!(root.live_bytes().unwrap(), 48 * KIB);
        assert_eq!(root.peak_bytes(), 48 * KIB, "a refusal raises no peak");

        // Room one byte short refuses; room for exactly the request accepts.
        ledger.release(LedgerClass::Cached, 16 * KIB);
        assert!(ledger.try_charge(LedgerClass::Live, 16 * KIB + 1).is_err());
        ledger.try_charge(LedgerClass::Live, 16 * KIB).unwrap();
        assert_eq!(root.live_bytes().unwrap(), 48 * KIB);
    }

    #[test]
    fn the_cache_child_has_its_own_cap() {
        let root = Budget::new(64 * KIB);
        let ledger = BudgetLedger::new(&root, 16 * KIB);
        ledger.try_charge(LedgerClass::Cached, 16 * KIB).unwrap();
        let refused = ledger.try_charge(LedgerClass::Cached, 1).unwrap_err();
        assert_eq!(refused.cap, 16 * KIB);
        assert_eq!(refused.live, 16 * KIB);
        ledger.try_charge(LedgerClass::Live, 48 * KIB).unwrap();
    }

    #[test]
    fn transfer_moves_a_charge_without_needing_room() {
        let root = Budget::new(16 * KIB);
        let ledger = BudgetLedger::new(&root, 16 * KIB);
        ledger.try_charge(LedgerClass::Cached, 16 * KIB).unwrap();
        // The cap is full, and a free-list hit still moves to live.
        ledger
            .transfer(16 * KIB, LedgerClass::Cached, LedgerClass::Live)
            .unwrap();
        assert_eq!((ledger.live_bytes(), ledger.cached_bytes()), (16 * KIB, 0));
        ledger
            .transfer(16 * KIB, LedgerClass::Live, LedgerClass::Cached)
            .unwrap();
        assert_eq!((ledger.live_bytes(), ledger.cached_bytes()), (0, 16 * KIB));
        assert_eq!(ledger.unmatched_releases(), 0);
    }

    #[test]
    fn a_failed_transfer_has_released_its_source() {
        let root = Budget::new(16 * KIB);
        let ledger = BudgetLedger::new(&root, 8 * KIB);
        ledger.try_charge(LedgerClass::Live, 16 * KIB).unwrap();
        // The cache child caps at 8 KiB, so the move is refused; the tessl
        // pool then destroys the buffer, so no charge may remain.
        assert!(ledger
            .transfer(16 * KIB, LedgerClass::Live, LedgerClass::Cached)
            .is_err());
        assert_eq!((ledger.live_bytes(), ledger.cached_bytes()), (0, 0));
        assert_eq!(root.live_bytes().unwrap(), 0);
    }

    #[test]
    fn a_release_without_a_charge_is_counted_not_applied() {
        let root = Budget::new(16 * KIB);
        let ledger = BudgetLedger::new(&root, 16 * KIB);
        ledger.try_charge(LedgerClass::Live, 4 * KIB).unwrap();
        ledger.release(LedgerClass::Live, 8 * KIB);
        ledger.release(LedgerClass::Cached, 4 * KIB);
        assert_eq!(ledger.unmatched_releases(), 2);
        assert_eq!(
            ledger.live_bytes(),
            4 * KIB,
            "an unmatched release frees nothing"
        );
    }

    /// `OjasError` has no `PartialEq`: read the numbers of a capacity error.
    fn numbers(error: Option<OjasError>) -> Option<(u64, u64, u64)> {
        match error {
            Some(OjasError::CapacityExceeded {
                requested,
                cap,
                live,
            }) => Some((requested, cap, live)),
            _ => None,
        }
    }

    #[test]
    fn a_refusal_is_a_capacity_error_with_its_numbers() {
        let refused = LedgerRefused {
            requested: 256,
            cap: 1 << 20,
            live: 1 << 20,
        };
        let want = Some((256, 1 << 20, 1 << 20));
        assert_eq!(numbers(Some(BudgetLedger::refusal_error(refused))), want);
        assert_eq!(
            numbers(BudgetLedger::capacity_error(&refused.to_string())),
            want
        );
        assert!(BudgetLedger::capacity_error("newBuffer(4096) failed").is_none());
    }

    #[test]
    fn an_overflowing_counter_is_refused_with_numbers_that_cannot_be_exceeded() {
        // A charge on top of u64::MAX live bytes is `OutOfRange` from the
        // budget, not a cap hit; the ledger still refuses it as a capacity
        // error rather than letting it through.
        let root = Budget::new(u64::MAX);
        let ledger = BudgetLedger::new(&root, u64::MAX);
        ledger.try_charge(LedgerClass::Live, u64::MAX).unwrap();
        let refused = ledger.try_charge(LedgerClass::Live, 1).unwrap_err();
        assert_eq!(refused.requested, 1);
        assert_eq!(refused.live, u64::MAX);
    }
}
