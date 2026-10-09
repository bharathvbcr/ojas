//! A bounded device-allocation budget.
//!
//! Every [`crate::runtime::CudaRuntime`] allocation first takes a
//! [`Reservation`] from its budget; the reservation is released when the
//! buffer drops. A request that would take the total past the cap is refused
//! with [`CudaError::Capacity`] before the driver is asked, so a sizing bug
//! fails loud at the cap instead of exhausting the device the campaign
//! shares.
//!
//! [`AllocBudget`] is a view of an [`ojas_core::Budget`], not a second
//! counter: the runtime's kernel buffers and cuBLAS workspace and
//! [`crate::CudaBackend`]'s tensors all charge the one `Budget` the runtime
//! was opened with ([`crate::runtime::CudaRuntime::open_with`]), so together
//! they never pass its cap, and a session budget that is a
//! [`Budget::child`] also charges its parent. The count is lock-free and
//! never exceeds the cap, including under concurrent reservations
//! (`Budget::try_reserve`).

use ojas_core::{Budget, OjasError};

use crate::error::CudaError;

/// Bytes held against an [`AllocBudget`] until dropped: the shared
/// [`Budget`]'s own reservation.
pub use ojas_core::Reservation;

/// A byte cap and the bytes currently reserved against it, in this crate's
/// error type.
#[derive(Clone, Debug)]
pub struct AllocBudget {
    budget: Budget,
}

impl AllocBudget {
    /// A budget of `cap` bytes that nothing else charges. A cap of 0 is
    /// refused: it could reserve nothing.
    pub fn new(cap: u64) -> Result<Self, CudaError> {
        Self::over(Budget::new(cap))
    }

    /// The view of `budget`: every reservation here charges it (and its
    /// ancestors). A cap of 0 is refused as [`CudaError::Capacity`].
    pub fn over(budget: Budget) -> Result<Self, CudaError> {
        if budget.cap_bytes() == 0 {
            return Err(CudaError::capacity(
                "AllocBudget::over",
                "a budget of 0 bytes cannot hold any buffer",
            ));
        }
        Ok(AllocBudget { budget })
    }

    /// The shared budget, for a caller that charges it in `ojas_core` terms.
    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    /// The cap in bytes.
    pub fn cap(&self) -> u64 {
        self.budget.cap_bytes()
    }

    /// Bytes reserved now, by every holder of the shared budget.
    pub fn used(&self) -> Result<u64, CudaError> {
        self.budget
            .live_bytes()
            .map_err(|e| CudaError::invalid("AllocBudget::used", e.to_string()))
    }

    /// Reserve `bytes` for `what`, or refuse without changing the count.
    pub fn reserve(&self, bytes: u64, what: &str) -> Result<Reservation, CudaError> {
        self.budget.try_reserve(bytes).map_err(|e| match e {
            OjasError::CapacityExceeded {
                requested,
                cap,
                live,
            } => CudaError::capacity(
                what,
                format!(
                    "{requested} bytes requested, {live} of the {cap} byte budget already reserved"
                ),
            ),
            // `live + bytes` overflowed u64.
            other => CudaError::capacity(what, other.to_string()),
        })
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
        assert_eq!(budget.used().unwrap(), 60);
        let b = budget.reserve(40, "b").unwrap();
        assert_eq!(budget.used().unwrap(), 100);
        drop(a);
        assert_eq!(budget.used().unwrap(), 40);
        drop(b);
        assert_eq!(budget.used().unwrap(), 0);
    }

    #[test]
    fn a_request_past_the_cap_is_refused_and_changes_nothing() {
        let budget = AllocBudget::new(100).unwrap();
        let _held = budget.reserve(90, "held").unwrap();
        let err = budget.reserve(11, "too big").unwrap_err();
        assert_eq!(err.kind(), "capacity", "{err}");
        assert!(err.to_string().starts_with("too big: "), "{err}");
        assert_eq!(budget.used().unwrap(), 90);
        assert!(budget.reserve(10, "fits").is_ok());
    }

    #[test]
    fn overflowing_requests_are_refused() {
        let budget = AllocBudget::new(u64::MAX).unwrap();
        let _held = budget.reserve(u64::MAX - 1, "held").unwrap();
        let err = budget.reserve(2, "wraps").unwrap_err();
        assert_eq!(err.kind(), "capacity", "{err}");
        assert!(bytes_for(usize::MAX, 4, "huge").is_err());
        assert_eq!(bytes_for(3, 4, "small").unwrap(), 12);
    }

    #[test]
    fn a_zero_cap_is_refused() {
        let err = AllocBudget::new(0).unwrap_err();
        assert_eq!(err.kind(), "capacity", "{err}");
        assert!(AllocBudget::over(Budget::new(0)).is_err());
    }

    /// The runtime's view and the backend's `Budget` are one counter: bytes
    /// reserved through either are refused by the other at the shared cap.
    /// Before, `AllocBudget` kept its own count, so each side could fill its
    /// own cap and together hold twice it.
    #[test]
    fn the_view_and_the_backend_budget_share_one_count() {
        let shared = Budget::new(100);
        let view = AllocBudget::over(shared.clone()).unwrap();
        let kernel = view.reserve(60, "kernel buffer").unwrap();
        assert_eq!(shared.live_bytes().unwrap(), 60);
        assert!(matches!(
            shared.try_reserve(41),
            Err(OjasError::CapacityExceeded {
                requested: 41,
                cap: 100,
                live: 60
            })
        ));
        let tensor = shared.try_reserve(40).unwrap();
        assert_eq!(view.used().unwrap(), 100);
        assert_eq!(view.reserve(1, "one more").unwrap_err().kind(), "capacity");
        drop(kernel);
        drop(tensor);
        assert_eq!(view.used().unwrap(), 0);
    }

    /// A view of a child budget charges the parent too, so a session's
    /// device bytes count against the process ceiling.
    #[test]
    fn a_view_of_a_child_budget_charges_the_parent() {
        let root = Budget::new(100);
        let view = AllocBudget::over(root.child(80)).unwrap();
        let _root_held = root.try_reserve(30).unwrap();
        let err = view.reserve(71, "past the root").unwrap_err();
        assert_eq!(err.kind(), "capacity", "{err}");
        let _held = view.reserve(70, "fits both").unwrap();
        assert_eq!(root.live_bytes().unwrap(), 100);
    }

    /// 16 threads reserve and release random sizes, half through the
    /// runtime's view of a child budget and half straight on the child or on
    /// its parent, as a backend, a kernel buffer and another session would.
    /// Neither cap is ever passed, a refusal changes nothing, and every byte
    /// comes back.
    #[test]
    fn stress_mixed_holders_never_pass_either_cap_and_release_everything() {
        const ROOT_CAP: u64 = 10_000;
        const CHILD_CAP: u64 = 6_000;
        let root = Budget::new(ROOT_CAP);
        let child = root.child(CHILD_CAP);
        let view = AllocBudget::over(child.clone()).unwrap();
        let handles: Vec<_> = (0..16u64)
            .map(|t| {
                let (root, child, view) = (root.clone(), child.clone(), view.clone());
                std::thread::spawn(move || {
                    let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ t;
                    let mut next = move || {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        seed
                    };
                    let mut held: Vec<Reservation> = Vec::new();
                    for _ in 0..4_000 {
                        let bytes = next() % 700;
                        let got = match next() % 3 {
                            0 => view.reserve(bytes, "view").map_err(|e| {
                                assert_eq!(e.kind(), "capacity", "{e}");
                            }),
                            1 => child.try_reserve(bytes).map_err(|e| {
                                assert!(matches!(e, OjasError::CapacityExceeded { .. }), "{e}");
                            }),
                            _ => root.try_reserve(bytes).map_err(|e| {
                                assert!(matches!(e, OjasError::CapacityExceeded { .. }), "{e}");
                            }),
                        };
                        match got {
                            Ok(r) => held.push(r),
                            Err(()) => {
                                if !held.is_empty() {
                                    let at = (next() as usize) % held.len();
                                    held.swap_remove(at);
                                }
                            }
                        }
                        assert!(child.live_bytes().unwrap() <= CHILD_CAP);
                        assert!(root.live_bytes().unwrap() <= ROOT_CAP);
                        assert!(view.used().unwrap() <= view.cap());
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(view.used().unwrap(), 0);
        assert_eq!(root.live_bytes().unwrap(), 0);
        assert!(root.peak_bytes() <= ROOT_CAP && child.peak_bytes() <= CHILD_CAP);
    }

    /// Sizes at the edges: 0, 1, the exact remaining room, one past it,
    /// and values near `u64::MAX` that would overflow the count.
    #[test]
    fn edge_sizes_are_exact_at_the_cap_and_never_wrap() {
        let view = AllocBudget::new(1_000).unwrap();
        let zero = view.reserve(0, "zero").unwrap();
        assert_eq!(view.used().unwrap(), 0);
        let one = view.reserve(1, "one").unwrap();
        let rest = view.reserve(999, "exactly the rest").unwrap();
        assert_eq!(view.used().unwrap(), 1_000);
        for bytes in [1, 2, 1_000, u64::MAX / 2, u64::MAX - 1, u64::MAX] {
            let err = view.reserve(bytes, "past").unwrap_err();
            assert_eq!(err.kind(), "capacity", "{bytes}: {err}");
            assert_eq!(view.used().unwrap(), 1_000, "{bytes}");
        }
        drop((zero, one, rest));
        assert_eq!(view.used().unwrap(), 0);
        let huge = AllocBudget::new(u64::MAX).unwrap();
        let _all = huge.reserve(u64::MAX, "all").unwrap();
        assert_eq!(huge.reserve(1, "wraps").unwrap_err().kind(), "capacity");
        assert_eq!(huge.used().unwrap(), u64::MAX);
    }

    #[test]
    fn concurrent_reservations_never_exceed_the_cap() {
        let budget = AllocBudget::new(1000).unwrap();
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let budget = budget.clone();
                std::thread::spawn(move || {
                    let mut held = Vec::new();
                    for _ in 0..500 {
                        match budget.reserve(7, "t") {
                            Ok(r) => held.push(r),
                            Err(_) => {
                                held.pop();
                            }
                        }
                        assert!(budget.used().unwrap() <= 1000);
                    }
                    held.len()
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(budget.used().unwrap(), 0);
    }
}
