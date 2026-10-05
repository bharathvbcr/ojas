//! A bounded device-allocation budget.
//!
//! Every [`crate::runtime::CudaRuntime`] allocation first takes a
//! [`Reservation`] from its budget; the reservation is released when the
//! buffer drops. A request that would take the total past the cap is refused
//! with [`CudaError::Capacity`] before the driver is asked, so a sizing bug
//! fails loud at the cap instead of exhausting the device the campaign
//! shares. The count is lock-free and never exceeds the cap, including under
//! concurrent reservations.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::error::CudaError;

/// A byte cap and the bytes currently reserved against it.
#[derive(Debug)]
pub struct AllocBudget {
    cap: u64,
    used: AtomicU64,
}

/// Bytes held against an [`AllocBudget`] until dropped.
#[derive(Debug)]
pub struct Reservation {
    budget: Arc<AllocBudget>,
    bytes: u64,
}

impl AllocBudget {
    /// A budget of `cap` bytes. A cap of 0 is refused: it could reserve nothing.
    pub fn new(cap: u64) -> Result<Arc<Self>, CudaError> {
        if cap == 0 {
            return Err(CudaError::invalid(
                "AllocBudget::new",
                "a budget of 0 bytes cannot hold any buffer",
            ));
        }
        Ok(Arc::new(AllocBudget {
            cap,
            used: AtomicU64::new(0),
        }))
    }

    /// The cap in bytes.
    pub fn cap(&self) -> u64 {
        self.cap
    }

    /// Bytes reserved now.
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Acquire)
    }

    /// Reserve `bytes` for `what`, or refuse without changing the count.
    pub fn reserve(self: &Arc<Self>, bytes: u64, what: &str) -> Result<Reservation, CudaError> {
        let mut current = self.used.load(Ordering::Acquire);
        loop {
            let next = current.checked_add(bytes).ok_or_else(|| {
                CudaError::capacity(what, format!("{current} + {bytes} bytes overflows u64"))
            })?;
            if next > self.cap {
                return Err(CudaError::capacity(
                    what,
                    format!(
                        "{bytes} bytes requested, {current} of the {} byte budget already reserved",
                        self.cap
                    ),
                ));
            }
            match self.used.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(Reservation {
                        budget: Arc::clone(self),
                        bytes,
                    })
                }
                Err(actual) => current = actual,
            }
        }
    }
}

impl Reservation {
    /// The reserved byte count.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // Every reservation added exactly `bytes`, so this cannot underflow.
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// `len` elements of `elem_bytes` each, in bytes, overflow-checked.
pub fn bytes_for(len: usize, elem_bytes: usize, what: &str) -> Result<u64, CudaError> {
    let bytes = len.checked_mul(elem_bytes).ok_or_else(|| {
        CudaError::capacity(
            what,
            format!("{len} elements of {elem_bytes} bytes overflow usize"),
        )
    })?;
    u64::try_from(bytes)
        .map_err(|_| CudaError::capacity(what, format!("{bytes} bytes do not fit in u64")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_count_and_release_on_drop() {
        let budget = AllocBudget::new(100).unwrap();
        let a = budget.reserve(60, "a").unwrap();
        assert_eq!(budget.used(), 60);
        let b = budget.reserve(40, "b").unwrap();
        assert_eq!(budget.used(), 100);
        drop(a);
        assert_eq!(budget.used(), 40);
        drop(b);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn a_request_past_the_cap_is_refused_and_changes_nothing() {
        let budget = AllocBudget::new(100).unwrap();
        let _held = budget.reserve(90, "held").unwrap();
        let err = budget.reserve(11, "too big").unwrap_err();
        assert_eq!(err.kind(), "capacity", "{err}");
        assert_eq!(budget.used(), 90);
        assert!(budget.reserve(10, "fits").is_ok());
    }

    #[test]
    fn overflowing_requests_are_refused() {
        let budget = AllocBudget::new(u64::MAX).unwrap();
        let _held = budget.reserve(u64::MAX - 1, "held").unwrap();
        assert!(budget.reserve(2, "wraps").is_err());
        assert!(bytes_for(usize::MAX, 4, "huge").is_err());
        assert_eq!(bytes_for(3, 4, "small").unwrap(), 12);
    }

    #[test]
    fn a_zero_cap_is_refused() {
        assert!(AllocBudget::new(0).is_err());
    }

    #[test]
    fn concurrent_reservations_never_exceed_the_cap() {
        let budget = AllocBudget::new(1000).unwrap();
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let budget = Arc::clone(&budget);
                std::thread::spawn(move || {
                    let mut held = Vec::new();
                    for _ in 0..500 {
                        match budget.reserve(7, "t") {
                            Ok(r) => held.push(r),
                            Err(_) => {
                                held.pop();
                            }
                        }
                        assert!(budget.used() <= 1000);
                    }
                    held.len()
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(budget.used(), 0);
    }
}
