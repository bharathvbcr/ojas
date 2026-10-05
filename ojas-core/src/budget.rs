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
/// The live counter is an [`AtomicU64`] updated with `try_update`. A
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
    /// Highest `live_bytes` since creation or the last
    /// [`Budget::reset_peak`].
    peak_bytes: AtomicU64,
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
                peak_bytes: AtomicU64::new(0),
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
                peak_bytes: AtomicU64::new(0),
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

    /// The highest live charge this budget has held since it was created or
    /// [`Budget::reset_peak`] last ran. Never below the current live bytes.
    ///
    /// Each accepted charge records the exact value it reached, so
    /// concurrent charges and releases cannot hide a peak. The figure can
    /// only over-report: when a [`Budget::child`] refuses, the charge it
    /// rolls back on this budget was briefly live and stays counted.
    pub fn peak_bytes(&self) -> u64 {
        self.inner.peak_bytes.load(Ordering::Acquire)
    }

    /// Restart the peak at the current live bytes, to measure one phase.
    pub fn reset_peak(&self) {
        let live = self.inner.live_bytes.load(Ordering::Acquire);
        self.inner.peak_bytes.store(live, Ordering::Release);
        // A charge that landed between the two operations raised live; keep
        // the invariant `peak >= live`.
        let now = self.inner.live_bytes.load(Ordering::Acquire);
        self.inner.peak_bytes.fetch_max(now, Ordering::AcqRel);
    }

    /// Refuse now if `bytes` more would not fit this budget and every
    /// ancestor, exactly as [`Budget::try_reserve`] would; charge nothing.
    ///
    /// For preflight checks. It reads one moment: another holder of an
    /// ancestor can take the room afterwards, so a later charge can still be
    /// refused. A caller that needs the room kept holds a [`Reservation`].
    pub fn check_room(&self, bytes: u64) -> Result<(), OjasError> {
        self.try_reserve(bytes).map(drop)
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
            .try_update(Ordering::AcqRel, Ordering::Acquire, |live| {
                let next = live.checked_add(bytes)?;
                if next > cap {
                    None
                } else {
                    Some(next)
                }
            }) {
            Ok(prev) => {
                // The exact value this charge reached (`prev + bytes` was
                // checked inside the update), so concurrent releases cannot
                // hide it from the peak.
                self.inner
                    .peak_bytes
                    .fetch_max(prev + bytes, Ordering::AcqRel);
                Ok(())
            }
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
            .try_update(Ordering::AcqRel, Ordering::Acquire, |live| {
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

    /// `src` copied into a charged vector, with no zero-fill first.
    ///
    /// [`Self::try_alloc`] writes a default into every element and callers
    /// that then `copy_from_slice` write each one again. This reserves and
    /// `extend_from_slice`s, so each element is written once. The charge is
    /// `src.len() * size_of::<T>()`, released if the reserve fails and held
    /// by the scratch otherwise. An empty `src` charges nothing.
    pub fn try_from_slice(src: &[T], budget: &Budget) -> Result<Self, OjasError> {
        let len = src.len();
        let bytes = (len as u64)
            .checked_mul(std::mem::size_of::<T>() as u64)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Scratch::try_from_slice",
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
        data.extend_from_slice(src);
        Ok(Self { data, reservation })
    }

    /// Charge `len` elements and reserve that capacity, then let `write` append them.
    ///
    /// Nothing is zero-filled and spare capacity is not handed out as
    /// `&mut [T]`. `write` appends (typically [`Vec::extend_from_slice`])
    /// until the length is `len`; each appended element is written once.
    /// The charge is `len * size_of::<T>()`. It is released when the reserve
    /// fails, and when `write` leaves any other length: the vector is
    /// dropped before the reservation. An empty `len` charges nothing.
    /// `write` is not called if the budget or the reserve refuses.
    pub fn try_extend(
        len: usize,
        budget: &Budget,
        write: impl FnOnce(&mut Vec<T>),
    ) -> Result<Self, OjasError> {
        let bytes = (len as u64)
            .checked_mul(std::mem::size_of::<T>() as u64)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Scratch::try_extend",
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
        write(&mut data);
        if data.len() != len {
            let got = data.len();
            drop(data);
            drop(reservation);
            return Err(OjasError::Shape {
                op: "Scratch::try_extend",
                detail: format!("appended {got} elements, charged {len}"),
            });
        }
        Ok(Self { data, reservation })
    }

    /// Move `data` into a scratch that `reservation` already charged.
    ///
    /// `reservation.bytes()` must equal `data.len() * size_of::<T>()`.
    /// Otherwise `data` is dropped, the reservation is released, and this
    /// refuses: a buffer is not adopted unless that charge is exactly the
    /// element bytes. This does not allocate and does not copy elements.
    /// An empty vector with a zero-byte reservation charges nothing.
    pub fn try_adopt(data: Vec<T>, reservation: Reservation) -> Result<Self, OjasError> {
        let bytes = (data.len() as u64).checked_mul(std::mem::size_of::<T>() as u64);
        if bytes != Some(reservation.bytes()) {
            let got = data.len();
            let held = reservation.bytes();
            drop(data);
            drop(reservation);
            return Err(OjasError::OutOfRange {
                op: "Scratch::try_adopt",
                detail: format!("reservation of {held} bytes != {got} elements"),
            });
        }
        Ok(Self { data, reservation })
    }

    /// Charge two buffers of `len` and append into both from one `write`.
    ///
    /// Each buffer is charged `len * size_of::<T>()` and reserved with
    /// [`Vec::try_reserve_exact`] before `write` runs. `write` appends until
    /// both lengths are `len`; nothing is zero-filled. The charges are
    /// released when either reserve fails, and when `write` leaves any other
    /// length: both vectors are dropped before either reservation. A failed
    /// second reserve releases the first charge. An empty `len` charges
    /// nothing. `write` is not called if the budget or a reserve refuses.
    pub fn try_extend_pair(
        len: usize,
        budget: &Budget,
        write: impl FnOnce(&mut Vec<T>, &mut Vec<T>),
    ) -> Result<(Self, Self), OjasError> {
        let bytes = (len as u64)
            .checked_mul(std::mem::size_of::<T>() as u64)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Scratch::try_extend_pair",
                detail: format!("{len} * {} overflows u64", std::mem::size_of::<T>()),
            })?;
        let first = budget.try_reserve(bytes)?;
        let second = match budget.try_reserve(bytes) {
            Ok(reservation) => reservation,
            Err(err) => {
                drop(first);
                return Err(err);
            }
        };
        let mut left = Vec::new();
        let mut right = Vec::new();
        if left.try_reserve_exact(len).is_err() {
            return Err(pair_reserve_refused(
                budget, bytes, left, right, first, second,
            )?);
        }
        if right.try_reserve_exact(len).is_err() {
            return Err(pair_reserve_refused(
                budget, bytes, left, right, first, second,
            )?);
        }
        write(&mut left, &mut right);
        if left.len() != len || right.len() != len {
            let (got_left, got_right) = (left.len(), right.len());
            drop(left);
            drop(right);
            drop(first);
            drop(second);
            return Err(OjasError::Shape {
                op: "Scratch::try_extend_pair",
                detail: format!("appended {got_left} and {got_right} elements, charged {len} each"),
            });
        }
        Ok((
            Self {
                data: left,
                reservation: first,
            },
            Self {
                data: right,
                reservation: second,
            },
        ))
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

/// Both allocations are freed and both charges released, then the refusal
/// reports the live counter after that release.
fn pair_reserve_refused<T>(
    budget: &Budget,
    bytes: u64,
    left: Vec<T>,
    right: Vec<T>,
    first: Reservation,
    second: Reservation,
) -> Result<OjasError, OjasError> {
    drop(left);
    drop(right);
    drop(first);
    drop(second);
    Ok(OjasError::CapacityExceeded {
        requested: bytes,
        cap: budget.cap_bytes(),
        live: budget.live_bytes()?,
    })
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

    /// A copied scratch charges once, keeps every bit (including -0), and
    /// a refused copy leaves the budget where it was.
    #[test]
    fn try_from_slice_charges_once_and_a_refusal_releases() {
        let budget = Budget::new(16);
        let src = [-0.0f32, 1.0, f32::from_bits(0x0000_0001), 4.0];
        let scratch = Scratch::<f32>::try_from_slice(&src, &budget).unwrap();
        assert_eq!(scratch.len(), 4);
        assert_eq!(
            scratch
                .as_slice()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            src.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(budget.live_bytes().unwrap(), 16);
        let err = Scratch::<f32>::try_from_slice(&[1.0], &budget).unwrap_err();
        assert!(matches!(
            err,
            OjasError::CapacityExceeded { requested: 4, .. }
        ));
        assert_eq!(budget.live_bytes().unwrap(), 16);
        let empty = Scratch::<f32>::try_from_slice(&[], &budget).unwrap();
        assert!(empty.is_empty());
        assert_eq!(budget.live_bytes().unwrap(), 16);
        drop(scratch);
        drop(empty);
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    /// Appends charge once, keep `-0` bits, and a refusal or a short append
    /// leaves the budget where it started.
    #[test]
    fn try_extend_charges_once_and_a_short_append_releases() {
        let budget = Budget::new(16);
        let src = [-0.0f32, 1.0, f32::from_bits(0x0000_0001), 4.0];
        let scratch = Scratch::<f32>::try_extend(src.len(), &budget, |dst| {
            dst.extend_from_slice(&src[..2]);
            dst.extend_from_slice(&src[2..]);
        })
        .unwrap();
        assert_eq!(
            scratch
                .as_slice()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            src.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(budget.live_bytes().unwrap(), 16);

        let mut called = false;
        let err = Scratch::<f32>::try_extend(1, &budget, |_| {
            called = true;
        })
        .unwrap_err();
        assert!(!called);
        assert!(matches!(
            err,
            OjasError::CapacityExceeded { requested: 4, .. }
        ));
        assert_eq!(budget.live_bytes().unwrap(), 16);

        let fresh = Budget::new(16);
        let err = Scratch::<f32>::try_extend(src.len(), &fresh, |dst| {
            dst.extend_from_slice(&src[..2]);
        })
        .unwrap_err();
        assert!(matches!(err, OjasError::Shape { .. }));
        assert_eq!(fresh.live_bytes().unwrap(), 0);

        let empty = Scratch::<f32>::try_extend(0, &fresh, |_| {}).unwrap();
        assert!(empty.is_empty());
        assert_eq!(fresh.live_bytes().unwrap(), 0);
        drop(scratch);
        drop(empty);
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    /// Two buffers are charged separately, keep `-0` bits, and a second
    /// reserve that does not fit releases the first without calling `write`.
    #[test]
    fn try_extend_pair_charges_each_and_a_second_refusal_releases_the_first() {
        let budget = Budget::new(32);
        let src = [-0.0f32, 1.0, f32::from_bits(0x0000_0001), 4.0];
        let (left, right) = Scratch::<f32>::try_extend_pair(src.len(), &budget, |a, b| {
            for chunk in src.chunks(2) {
                a.extend_from_slice(chunk);
                b.extend_from_slice(chunk);
            }
        })
        .unwrap();
        let want: Vec<u32> = src.iter().map(|v| v.to_bits()).collect();
        let bits_of =
            |s: &Scratch<f32>| s.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits_of(&left), want);
        assert_eq!(bits_of(&right), want);
        assert_eq!(budget.live_bytes().unwrap(), 32);
        drop(right);
        assert_eq!(
            budget.live_bytes().unwrap(),
            16,
            "each buffer is its own charge"
        );
        drop(left);
        assert_eq!(budget.live_bytes().unwrap(), 0);

        let tight = Budget::new(4);
        let mut called = false;
        let err = Scratch::<f32>::try_extend_pair(1, &tight, |_, _| {
            called = true;
        })
        .unwrap_err();
        assert!(!called);
        assert!(matches!(
            err,
            OjasError::CapacityExceeded { requested: 4, .. }
        ));
        assert_eq!(
            tight.live_bytes().unwrap(),
            0,
            "the first charge was released"
        );

        let fresh = Budget::new(32);
        let err = Scratch::<f32>::try_extend_pair(src.len(), &fresh, |a, b| {
            a.extend_from_slice(&src);
            b.extend_from_slice(&src[..2]);
        })
        .unwrap_err();
        assert!(matches!(err, OjasError::Shape { .. }));
        assert_eq!(fresh.live_bytes().unwrap(), 0);

        let (empty_a, empty_b) = Scratch::<f32>::try_extend_pair(0, &fresh, |_, _| {}).unwrap();
        assert!(empty_a.is_empty() && empty_b.is_empty());
        assert_eq!(fresh.live_bytes().unwrap(), 0);

        let overflow = (u64::MAX / 4) + 1;
        let err =
            Scratch::<f32>::try_extend_pair(overflow as usize, &Budget::new(u64::MAX), |_, _| {
                panic!("an overflowing length must not append");
            });
        assert!(matches!(err, Err(OjasError::OutOfRange { .. })));
        drop(empty_a);
        drop(empty_b);
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
    fn try_adopt_keeps_the_charge_and_a_mismatch_releases_it() {
        let budget = Budget::new(64);
        let reservation = budget.try_reserve(16).unwrap();
        let data = vec![-0.0f32, 1.0, f32::from_bits(0x7fc0_0001), 4.0];
        let scratch = Scratch::try_adopt(data, reservation).unwrap();
        assert_eq!(scratch.len(), 4);
        assert_eq!(scratch.as_slice()[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(scratch.as_slice()[2].to_bits(), 0x7fc0_0001);
        assert_eq!(budget.live_bytes().unwrap(), 16);
        drop(scratch);
        assert_eq!(budget.live_bytes().unwrap(), 0);

        let reservation = budget.try_reserve(8).unwrap();
        let err = Scratch::<f32>::try_adopt(vec![1.0], reservation).unwrap_err();
        assert!(matches!(
            err,
            OjasError::OutOfRange {
                op: "Scratch::try_adopt",
                ..
            }
        ));
        assert_eq!(
            budget.live_bytes().unwrap(),
            0,
            "a mismatch releases the charge"
        );

        let reservation = budget.try_reserve(0).unwrap();
        let empty = Scratch::<f32>::try_adopt(Vec::<f32>::new(), reservation).unwrap();
        assert!(empty.is_empty());
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    #[test]
    fn scratch_reservation_failure_does_not_leave_a_charge() {
        let budget = Budget::new(u64::MAX);
        let err = Scratch::<u8>::try_alloc(isize::MAX as usize, &budget).unwrap_err();
        assert!(matches!(err, OjasError::CapacityExceeded { live: 0, .. }));
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    #[test]
    fn peak_tracks_the_high_water_mark_and_resets_to_live() {
        let parent = Budget::new(100);
        let child = parent.child(50);
        assert_eq!((parent.peak_bytes(), child.peak_bytes()), (0, 0));
        let a = child.try_reserve(30).unwrap();
        let b = parent.try_reserve(40).unwrap();
        assert_eq!((parent.peak_bytes(), child.peak_bytes()), (70, 30));
        // Refused by the parent: no level moves.
        assert!(child.try_reserve(31).is_err());
        assert_eq!((parent.peak_bytes(), child.peak_bytes()), (70, 30));
        // Refused by the child after the parent accepted: the rolled-back
        // parent charge was live for a moment and over-reports (never under).
        assert!(child.try_reserve(21).is_err());
        assert_eq!((parent.peak_bytes(), child.peak_bytes()), (91, 30));
        assert_eq!(parent.live_bytes().unwrap(), 70);
        drop(a);
        assert_eq!(parent.peak_bytes(), 91, "a release keeps the peak");
        parent.reset_peak();
        assert_eq!(parent.peak_bytes(), 40, "reset restarts at live");
        drop(b);
        assert_eq!(parent.peak_bytes(), 40);
        let max = Budget::new(u64::MAX);
        let all = max.try_reserve(u64::MAX).unwrap();
        assert_eq!(max.peak_bytes(), u64::MAX);
        drop(all);
    }

    #[test]
    fn check_room_answers_like_try_reserve_and_charges_nothing() {
        let parent = Budget::new(100);
        let child = parent.child(80);
        let held = parent.try_reserve(30).unwrap();
        child.check_room(70).unwrap();
        assert!(matches!(
            child.check_room(71),
            Err(OjasError::CapacityExceeded { cap: 100, .. })
        ));
        assert!(matches!(
            child.check_room(81),
            Err(OjasError::CapacityExceeded { .. })
        ));
        assert!(matches!(Budget::new(u64::MAX).check_room(u64::MAX), Ok(())));
        assert_eq!(parent.live_bytes().unwrap(), 30);
        assert_eq!(child.live_bytes().unwrap(), 0);
        drop(held);
    }

    #[test]
    fn concurrent_peak_is_never_below_any_observed_live() {
        let budget = Budget::new(1 << 20);
        let seen = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for i in 1..=2_000u64 {
                        let r = budget.try_reserve(i % 97 + 1).unwrap();
                        let live = budget.live_bytes().unwrap();
                        seen.fetch_max(live, Ordering::AcqRel);
                        drop(r);
                    }
                });
            }
        });
        assert!(budget.peak_bytes() >= seen.load(Ordering::Acquire));
        assert!(budget.peak_bytes() <= 4 * 97);
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }
}
