use crate::OjasError;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Byte cap for allocations this process is willing to make.
///
/// [`Budget::try_reserve`] returns [`OjasError::CapacityExceeded`] when
/// `live + request` would pass the cap. It does not shrink the request.
/// Adding past `u64::MAX` is [`OjasError::OutOfRange`] and does not wrap
/// the counter.
///
/// The live counter is an [`AtomicU64`] updated with `fetch_update`. A
/// [`Budget::child`] charges its parent to completion, then itself. The two
/// counters are never locked together. If the child refuses, the parent
/// charge is released.
#[derive(Clone, Debug)]
pub struct Budget {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    cap_bytes: u64,
    live_bytes: AtomicU64,
    parent: Option<Budget>,
    /// Device-to-host copies charged to this budget or a descendant.
    readbacks: AtomicU64,
    readback_bytes: AtomicU64,
}

/// Charges `bytes` against a [`Budget`] until dropped.
///
/// Not cloneable: one reservation releases once. A child reservation
/// releases the child and then each ancestor.
#[derive(Debug)]
pub struct Reservation {
    bytes: u64,
    budget: Budget,
}

/// A vector whose byte length stays charged for the allocation's lifetime.
///
/// `data` is declared before `reservation`. Fields drop in reverse order, so
/// the `Vec` is freed and then the reservation releases the charge.
#[derive(Debug)]
pub struct Scratch<T> {
    data: Vec<T>,
    reservation: Reservation,
}

impl Budget {
    pub fn new(cap_bytes: u64) -> Self {
        Self {
            inner: Arc::new(Inner {
                cap_bytes,
                live_bytes: AtomicU64::new(0),
                parent: None,
                readbacks: AtomicU64::new(0),
                readback_bytes: AtomicU64::new(0),
            }),
        }
    }

    /// A budget with its own cap that also charges this one.
    ///
    /// Creating the child does not charge either counter. [`Budget::try_reserve`]
    /// on the child charges this budget first, then the child, and rolls the
    /// parent charge back if the child refuses.
    pub fn child(&self, cap_bytes: u64) -> Self {
        Self {
            inner: Arc::new(Inner {
                cap_bytes,
                live_bytes: AtomicU64::new(0),
                parent: Some(self.clone()),
                readbacks: AtomicU64::new(0),
                readback_bytes: AtomicU64::new(0),
            }),
        }
    }

    /// `(calls, bytes)` of device-to-host copies [`crate::Tensor::to_host`]
    /// charged to this budget or to any of its descendants. Monotonic.
    ///
    /// Unlike the process-wide [`crate::device_readbacks`], this counts only
    /// the budget tree a caller owns. Two tests that each build a backend on
    /// their own `Budget` cannot see each other's readbacks, whatever thread
    /// made them. A readback made on another thread into this tree is still
    /// counted, so "zero readbacks" cannot pass because the copy happened
    /// elsewhere.
    pub fn device_readbacks(&self) -> (u64, u64) {
        (
            self.inner.readbacks.load(Ordering::Acquire),
            self.inner.readback_bytes.load(Ordering::Acquire),
        )
    }

    /// Count one readback of `bytes` here and in every ancestor.
    pub(crate) fn record_readback(&self, bytes: u64) {
        let mut node = Some(self);
        while let Some(budget) = node {
            budget.inner.readbacks.fetch_add(1, Ordering::AcqRel);
            budget
                .inner
                .readback_bytes
                .fetch_add(bytes, Ordering::AcqRel);
            node = budget.inner.parent.as_ref();
        }
    }

    pub fn cap_bytes(&self) -> u64 {
        self.inner.cap_bytes
    }

    pub fn live_bytes(&self) -> Result<u64, OjasError> {
        Ok(self.inner.live_bytes.load(Ordering::Acquire))
    }

    /// Reserve `bytes`, or refuse.
    ///
    /// A request of `0` returns a reservation and does not change the counter.
    /// The call does not touch the allocator.
    pub fn try_reserve(&self, bytes: u64) -> Result<Reservation, OjasError> {
        if bytes == 0 {
            return Ok(Reservation {
                bytes: 0,
                budget: self.clone(),
            });
        }
        self.charge(bytes)?;
        Ok(Reservation {
            bytes,
            budget: self.clone(),
        })
    }

    fn charge(&self, bytes: u64) -> Result<(), OjasError> {
        if let Some(parent) = &self.inner.parent {
            parent.charge(bytes)?;
            if let Err(err) = self.charge_self(bytes) {
                parent.release(bytes);
                return Err(err);
            }
            Ok(())
        } else {
            self.charge_self(bytes)
        }
    }

    fn charge_self(&self, bytes: u64) -> Result<(), OjasError> {
        let cap = self.inner.cap_bytes;
        match self
            .inner
            .live_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                let next = live.checked_add(bytes)?;
                if next > cap {
                    None
                } else {
                    Some(next)
                }
            }) {
            Ok(_) => Ok(()),
            Err(live) => {
                if live.checked_add(bytes).is_none() {
                    Err(OjasError::OutOfRange {
                        op: "Budget::try_reserve",
                        detail: format!("live {live} + {bytes} overflows u64"),
                    })
                } else {
                    Err(OjasError::CapacityExceeded {
                        requested: bytes,
                        cap,
                        live,
                    })
                }
            }
        }
    }

    fn release(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        self.release_self(bytes);
        if let Some(parent) = &self.inner.parent {
            parent.release(bytes);
        }
    }

    fn release_self(&self, bytes: u64) {
        let _ = self
            .inner
            .live_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                Some(live.saturating_sub(bytes))
            });
    }
}

impl Reservation {
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.release(self.bytes);
        self.bytes = 0;
    }
}

impl<T: Clone + Default> Scratch<T> {
    /// Allocate `len` default elements and charge `len * size_of::<T>()` bytes.
    ///
    /// The vector is reserved with `try_reserve_exact`. If that fails, the
    /// charge is released and nothing is returned. The reservation is the
    /// last field, so dropping a scratch frees the vector before the charge
    /// comes off the budget.
    pub fn try_alloc(len: usize, budget: &Budget) -> Result<Self, OjasError> {
        let bytes = (len as u64)
            .checked_mul(std::mem::size_of::<T>() as u64)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Scratch::try_alloc",
                detail: format!("{len} * {} overflows u64", std::mem::size_of::<T>()),
            })?;
        let reservation = budget.try_reserve(bytes)?;
        let mut data = Vec::new();
        if data.try_reserve_exact(len).is_err() {
            drop(reservation);
            return Err(OjasError::CapacityExceeded {
                requested: bytes,
                cap: budget.cap_bytes(),
                live: budget.live_bytes()?,
            });
        }
        data.resize(len, T::default());
        Ok(Self { data, reservation })
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Move the vector and its reservation out. `Drop` does not run.
    /// Callers free the vector before dropping the reservation.
    pub(crate) fn into_raw(self) -> (Vec<T>, Reservation) {
        let Self { data, reservation } = self;
        (data, reservation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Barrier;

    #[test]
    fn cap_is_exact_and_refusal_does_not_charge() {
        let budget = Budget::new(100);
        let a = budget.try_reserve(60).unwrap();
        let err = budget.try_reserve(41).unwrap_err();
        assert!(matches!(
            err,
            OjasError::CapacityExceeded {
                requested: 41,
                cap: 100,
                live: 60
            }
        ));
        assert_eq!(budget.live_bytes().unwrap(), 60);
        let b = budget.try_reserve(40).unwrap();
        assert_eq!(budget.live_bytes().unwrap(), 100);
        let zero = budget.try_reserve(0).unwrap();
        drop(a);
        drop(zero);
        assert_eq!(budget.live_bytes().unwrap(), 40);
        drop(b);
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    #[test]
    fn zero_cap_and_u64_edges() {
        let zero = Budget::new(0);
        assert!(zero.try_reserve(0).is_ok());
        assert!(matches!(
            zero.try_reserve(1),
            Err(OjasError::CapacityExceeded { .. })
        ));
        let max = Budget::new(u64::MAX);
        let all = max.try_reserve(u64::MAX).unwrap();
        assert!(matches!(
            max.try_reserve(1),
            Err(OjasError::OutOfRange { .. })
        ));
        assert_eq!(max.live_bytes().unwrap(), u64::MAX);
        drop(all);
        assert_eq!(max.live_bytes().unwrap(), 0);
        let one = max.try_reserve(1).unwrap();
        assert!(matches!(
            max.try_reserve(u64::MAX),
            Err(OjasError::OutOfRange { .. })
        ));
        drop(one);
        assert_eq!(max.live_bytes().unwrap(), 0);
        assert!(matches!(
            Budget::new(5).try_reserve(u64::MAX),
            Err(OjasError::CapacityExceeded { .. })
        ));
    }

    #[test]
    fn clones_share_one_counter() {
        let budget = Budget::new(10);
        let other = budget.clone();
        let r = other.try_reserve(7).unwrap();
        assert_eq!(budget.live_bytes().unwrap(), 7);
        assert!(budget.try_reserve(4).is_err());
        drop(other);
        assert_eq!(budget.live_bytes().unwrap(), 7);
        drop(r);
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    #[test]
    fn concurrent_reserve_release_never_exceeds_cap_or_leaks() {
        const THREADS: u64 = 16;
        const OPS: u64 = 10_000;
        const CAP: u64 = 1_000;
        let budget = Budget::new(CAP);
        let refused = Arc::new(AtomicU64::new(0));
        let barrier = Arc::new(Barrier::new(THREADS as usize));
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let budget = budget.clone();
                let refused = Arc::clone(&refused);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ t;
                    let mut held: Vec<Reservation> = Vec::new();
                    barrier.wait();
                    for _ in 0..OPS {
                        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                        let mut z = state;
                        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                        z ^= z >> 31;
                        if z.is_multiple_of(3) && !held.is_empty() {
                            held.swap_remove((z as usize >> 8) % held.len());
                        } else {
                            match budget.try_reserve((z >> 16) % 130) {
                                Ok(r) => held.push(r),
                                Err(OjasError::CapacityExceeded {
                                    live,
                                    cap,
                                    requested,
                                }) => {
                                    assert_eq!(cap, CAP);
                                    assert!(live + requested > CAP);
                                    refused.fetch_add(1, Ordering::Relaxed);
                                }
                                Err(other) => panic!("unexpected {other}"),
                            }
                        }
                        let live = budget.live_bytes().unwrap();
                        assert!(live <= CAP, "live {live} > cap");
                        let mine: u64 = held.iter().map(|r| r.bytes).sum();
                        assert!(mine <= live);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(refused.load(Ordering::Relaxed) > 0);
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    #[test]
    fn child_charges_parent_first_and_rolls_back() {
        let parent = Budget::new(100);
        let child = parent.child(40);
        let held = child.try_reserve(30).unwrap();
        assert_eq!(held.bytes(), 30);
        assert_eq!(child.live_bytes().unwrap(), 30);
        assert_eq!(parent.live_bytes().unwrap(), 30);

        let err = child.try_reserve(20).unwrap_err();
        assert!(matches!(
            err,
            OjasError::CapacityExceeded {
                requested: 20,
                cap: 40,
                live: 30
            }
        ));
        assert_eq!(parent.live_bytes().unwrap(), 30);
        assert_eq!(child.live_bytes().unwrap(), 30);

        let err = child.try_reserve(80).unwrap_err();
        assert!(matches!(
            err,
            OjasError::CapacityExceeded {
                cap: 100,
                live: 30,
                ..
            }
        ));
        assert_eq!(parent.live_bytes().unwrap(), 30);
        assert_eq!(child.live_bytes().unwrap(), 30);

        drop(held);
        assert_eq!(child.live_bytes().unwrap(), 0);
        assert_eq!(parent.live_bytes().unwrap(), 0);
    }

    #[test]
    fn nested_child_release_reaches_the_root() {
        let root = Budget::new(50);
        let mid = root.child(50);
        let leaf = mid.child(50);
        let held = leaf.try_reserve(20).unwrap();
        assert_eq!(root.live_bytes().unwrap(), 20);
        assert_eq!(mid.live_bytes().unwrap(), 20);
        assert_eq!(leaf.live_bytes().unwrap(), 20);
        drop(held);
        assert_eq!(root.live_bytes().unwrap(), 0);
        assert_eq!(mid.live_bytes().unwrap(), 0);
        assert_eq!(leaf.live_bytes().unwrap(), 0);
    }

    #[test]
    fn sibling_children_share_the_parent_cap() {
        let parent = Budget::new(100);
        let a = parent.child(80);
        let b = parent.child(80);
        let held_a = a.try_reserve(60).unwrap();
        let err = b.try_reserve(50).unwrap_err();
        assert!(matches!(
            err,
            OjasError::CapacityExceeded {
                cap: 100,
                live: 60,
                ..
            }
        ));
        assert_eq!(a.live_bytes().unwrap(), 60);
        assert_eq!(b.live_bytes().unwrap(), 0);
        let held_b = b.try_reserve(40).unwrap();
        assert_eq!(parent.live_bytes().unwrap(), 100);
        drop(held_a);
        drop(held_b);
        assert_eq!(parent.live_bytes().unwrap(), 0);
    }

    #[test]
    fn scratch_holds_one_charge_until_drop() {
        let budget = Budget::new(100);
        let scratch = Scratch::<u8>::try_alloc(16, &budget).unwrap();
        assert_eq!(scratch.len(), 16);
        assert_eq!(budget.live_bytes().unwrap(), 16);
        assert!(scratch.as_slice().iter().all(|b| *b == 0));
        drop(scratch);
        assert_eq!(budget.live_bytes().unwrap(), 0);
        assert!(matches!(
            Scratch::<u8>::try_alloc(101, &budget),
            Err(OjasError::CapacityExceeded { live: 0, .. })
        ));
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    #[test]
    fn scratch_reservation_failure_does_not_leave_a_charge() {
        let budget = Budget::new(u64::MAX);
        let err = Scratch::<u8>::try_alloc(isize::MAX as usize, &budget).unwrap_err();
        assert!(matches!(err, OjasError::CapacityExceeded { live: 0, .. }));
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }
}
