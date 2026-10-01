use crate::OjasError;
use std::sync::{Arc, Mutex};

/// Byte cap for allocations this process is willing to make.
///
/// [`Budget::try_reserve`] returns [`OjasError::CapacityExceeded`] when
/// `live + request` would pass the cap. It does not shrink the request.
/// Adding past `u64::MAX` is [`OjasError::OutOfRange`] and does not wrap
/// the counter.
#[derive(Clone, Debug)]
pub struct Budget {
    cap_bytes: u64,
    live_bytes: Arc<Mutex<u64>>,
}

/// Charges `bytes` against a [`Budget`] until dropped.
///
/// Not cloneable: one reservation releases once.
#[derive(Debug)]
pub struct Reservation {
    bytes: u64,
    budget: Budget,
}

impl Budget {
    pub fn new(cap_bytes: u64) -> Self {
        Self {
            cap_bytes,
            live_bytes: Arc::new(Mutex::new(0)),
        }
    }

    pub fn cap_bytes(&self) -> u64 {
        self.cap_bytes
    }

    pub fn live_bytes(&self) -> Result<u64, OjasError> {
        self.lock().map(|guard| *guard)
    }

    /// Reserve `bytes`, or refuse.
    ///
    /// A request of `0` returns a reservation and does not change the counter.
    /// The call does not touch the allocator.
    pub fn try_reserve(&self, bytes: u64) -> Result<Reservation, OjasError> {
        let mut live = self.lock()?;
        if bytes == 0 {
            return Ok(Reservation {
                bytes: 0,
                budget: self.clone(),
            });
        }
        let next = live.checked_add(bytes).ok_or_else(|| OjasError::OutOfRange {
            op: "Budget::try_reserve",
            detail: format!("live {} + {bytes} overflows u64", *live),
        })?;
        if next > self.cap_bytes {
            return Err(OjasError::CapacityExceeded {
                requested: bytes,
                cap: self.cap_bytes,
                live: *live,
            });
        }
        *live = next;
        drop(live);
        Ok(Reservation {
            bytes,
            budget: self.clone(),
        })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, u64>, OjasError> {
        self.live_bytes.lock().map_err(|_| OjasError::Poisoned)
    }

    fn release(&self, bytes: u64) -> Result<(), OjasError> {
        if bytes == 0 {
            return Ok(());
        }
        let mut live = self.lock()?;
        *live = live.saturating_sub(bytes);
        Ok(())
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // A poisoned lock is already an error. Drop must not panic.
        let _ = self.budget.release(self.bytes);
        self.bytes = 0;
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
        assert!(matches!(zero.try_reserve(1), Err(OjasError::CapacityExceeded { .. })));
        let max = Budget::new(u64::MAX);
        let all = max.try_reserve(u64::MAX).unwrap();
        assert!(matches!(max.try_reserve(1), Err(OjasError::OutOfRange { .. })));
        assert_eq!(max.live_bytes().unwrap(), u64::MAX);
        drop(all);
        assert_eq!(max.live_bytes().unwrap(), 0);
        let one = max.try_reserve(1).unwrap();
        assert!(matches!(max.try_reserve(u64::MAX), Err(OjasError::OutOfRange { .. })));
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
                        if z % 3 == 0 && !held.is_empty() {
                            held.swap_remove((z as usize >> 8) % held.len());
                        } else {
                            match budget.try_reserve((z >> 16) % 130) {
                                Ok(r) => held.push(r),
                                Err(OjasError::CapacityExceeded { live, cap, requested }) => {
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
}
