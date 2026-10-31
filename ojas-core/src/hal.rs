//! The portable device-extension contract.
//!
//! A device extension (Metal over tessl today) owns one allocator and one
//! command timeline. When its buffers are host-visible (unified memory,
//! `StorageModeShared`) the CPU and the GPU already see the same bytes; this
//! module names the pieces that let ojas use that without copying and
//! without a second ledger:
//!
//! - [`AllocError`]: a typed allocation failure, so a caller never parses
//!   the allocator's message text. It maps to [`OjasError::CapacityExceeded`].
//! - [`MappedBuffer`]: a [`DeviceBuffer`] whose bytes the host may borrow in
//!   place through a [`HostLease`] (shared) or a [`HostLeaseMut`]
//!   (exclusive).
//! - [`Timeline`]: the device's monotonic completion counter, the thing a
//!   lease waits on before it hands out bytes.
//! - [`Provenance`]: which arithmetic tier wrote a buffer's bytes.
//!
//! Every method defaults to refusing with [`OjasError::Unsupported`], wrapped
//! in [`FenceError::Refused`] where the method reports fence outcomes. A
//! backend that cannot lend its memory (wgpu, CUDA, HIP keep counted copies)
//! implements none of this, and a caller that needs a lease is told so; the
//! bytes are never copied behind its back. Nothing in the crate calls these
//! items yet: tensors do not carry a mapped payload, so this module changes
//! no behaviour.
//!
//! # Fence rules
//!
//! The rules a [`MappedBuffer`] implementation follows, and a caller may
//! rely on:
//!
//! - **Whole-buffer fences.** Each buffer carries the timeline values of its
//!   last GPU write and last GPU use. The fence is the whole buffer, never a
//!   byte range, because Metal makes even non-overlapping unsynchronized
//!   writes to one buffer undefined.
//! - **Read leases** wait until the timeline has completed the buffer's last
//!   write. Several read leases may be live at once.
//! - **Write leases** wait until the timeline has completed the buffer's last
//!   use (read or write) and are exclusive: a second lease of either kind is
//!   refused with [`FenceError::Busy`] while one is live.
//! - **A GPU bind against a live lease is refused** with an error. It never
//!   blocks the device thread.
//! - **Never wait on an uncommitted value.** A stamp above
//!   [`Timeline::submitted`] answers [`FenceError::NotSubmitted`]; the
//!   caller flushes the pending work through the device thread and retries.
//! - **Poison is checked on every fence result.** After a failed or
//!   timed-out command buffer the timeline answers [`FenceError::Poisoned`]
//!   from every later call; no lease is handed out on a poisoned device.

use crate::backend::{BackendId, Numerics};
use crate::tensor::DeviceBuffer;
use crate::OjasError;
use std::fmt;

/// A typed allocation failure from a device extension's pool.
///
/// The allocator reports a kind, never a message to parse: the byte counts
/// are the allocator's own, so a caller can retry, trim a cache or refuse
/// without reading text. Convert with `OjasError::from`; the three capacity
/// kinds become [`OjasError::CapacityExceeded`] and the cap is never lowered
/// to fit the request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocError {
    /// The shared [`crate::Budget`] refused the charge: `requested` bytes on
    /// top of `live` would pass `cap`. A cache trim may cure it.
    OverBudget { requested: u64, cap: u64, live: u64 },
    /// The device refused a buffer that was inside the budget (the driver
    /// returned no buffer). `cap` is the device's recommended working set.
    /// Deferred releases that have not recycled yet may cure it, so a waited
    /// commit and one retry is the caller's choice.
    Exhausted { requested: u64, cap: u64, live: u64 },
    /// The request is past what the device can ever hold. `cap` is that
    /// limit. Permanent: no wait or trim cures it.
    TooLarge { requested: u64, cap: u64, live: u64 },
    /// The device can no longer allocate or run work.
    DeviceLost { backend: BackendId, detail: String },
}

impl AllocError {
    /// Bytes the failed request asked for, when the kind records them.
    pub fn requested(&self) -> Option<u64> {
        match self {
            AllocError::OverBudget { requested, .. }
            | AllocError::Exhausted { requested, .. }
            | AllocError::TooLarge { requested, .. } => Some(*requested),
            AllocError::DeviceLost { .. } => None,
        }
    }

    /// Whether freeing memory (trimming a cache, draining deferred releases)
    /// and trying once more can succeed. Only the two kinds that depend on
    /// what is resident now can; [`AllocError::TooLarge`] and
    /// [`AllocError::DeviceLost`] are permanent.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            AllocError::OverBudget { .. } | AllocError::Exhausted { .. }
        )
    }
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::OverBudget {
                requested,
                cap,
                live,
            } => write!(
                f,
                "allocation over budget: requested {requested} bytes, cap {cap}, live {live}"
            ),
            AllocError::Exhausted {
                requested,
                cap,
                live,
            } => write!(
                f,
                "device refused allocation: requested {requested} bytes, \
                 working set {cap}, live {live}"
            ),
            AllocError::TooLarge {
                requested,
                cap,
                live,
            } => write!(
                f,
                "allocation past the device limit: requested {requested} bytes, \
                 limit {cap}, live {live}"
            ),
            AllocError::DeviceLost { backend, detail } => {
                write!(f, "backend {backend:?}: device lost: {detail}")
            }
        }
    }
}

impl std::error::Error for AllocError {}

impl From<AllocError> for OjasError {
    fn from(e: AllocError) -> OjasError {
        match e {
            AllocError::OverBudget {
                requested,
                cap,
                live,
            }
            | AllocError::Exhausted {
                requested,
                cap,
                live,
            }
            | AllocError::TooLarge {
                requested,
                cap,
                live,
            } => OjasError::CapacityExceeded {
                requested,
                cap,
                live,
            },
            AllocError::DeviceLost { backend, detail } => OjasError::DeviceLost { backend, detail },
        }
    }
}

/// What a fenced host access or a [`Timeline`] call can answer other than
/// success.
#[derive(Debug)]
pub enum FenceError {
    /// `value` is above [`Timeline::submitted`]: the work that would reach it
    /// has not been committed, so waiting would never return. The caller
    /// flushes pending work through the device thread and retries.
    NotSubmitted { value: u64, submitted: u64 },
    /// A conflicting lease is live (a write lease needs exclusivity), or a
    /// GPU bind was refused against a live host view. Refused, never
    /// blocked; the caller drops its lease or retries later.
    Busy { detail: String },
    /// The device was poisoned by a failed or timed-out command buffer.
    /// Permanent: every later fence call answers the same.
    Poisoned,
    /// A bounded wait ended before the timeline reached `value`.
    TimedOut { value: u64, completed: u64 },
    /// The implementation does not provide this call (the default), or
    /// refused it with a reason of its own. There is no host fallback.
    Refused(OjasError),
}

impl FenceError {
    /// The error for a method a backend does not implement.
    pub fn unsupported(op: &'static str, what: &str) -> FenceError {
        FenceError::Refused(OjasError::Unsupported {
            op,
            detail: format!(
                "{what} is not implemented by this backend; \
                 there is no host fallback"
            ),
        })
    }

    /// Whether the caller can retry after flushing, waiting or dropping a
    /// lease. [`FenceError::Poisoned`] and [`FenceError::Refused`] are
    /// permanent.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            FenceError::NotSubmitted { .. } | FenceError::Busy { .. } | FenceError::TimedOut { .. }
        )
    }

    /// Fold into [`OjasError`] for a caller with nothing better to do than
    /// fail. `backend` names the device that answered. Poison and a timeout
    /// are device loss, as a failed or timed-out command buffer is on
    /// Metal; [`FenceError::Refused`] passes through unchanged.
    pub fn into_ojas(self, backend: BackendId) -> OjasError {
        match self {
            FenceError::NotSubmitted { value, submitted } => OjasError::Backend {
                id: backend,
                detail: format!(
                    "timeline value {value} is above the submitted value {submitted}; \
                     flush pending work first"
                ),
            },
            FenceError::Busy { detail } => OjasError::Backend {
                id: backend,
                detail: format!("host lease conflict: {detail}"),
            },
            FenceError::Poisoned => OjasError::DeviceLost {
                backend,
                detail: "the device is poisoned".to_string(),
            },
            FenceError::TimedOut { value, completed } => OjasError::DeviceLost {
                backend,
                detail: format!("timed out waiting for timeline value {value} (at {completed})"),
            },
            FenceError::Refused(e) => e,
        }
    }
}

impl fmt::Display for FenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FenceError::NotSubmitted { value, submitted } => write!(
                f,
                "timeline value {value} not submitted (submitted {submitted})"
            ),
            FenceError::Busy { detail } => write!(f, "busy: {detail}"),
            FenceError::Poisoned => write!(f, "poisoned"),
            FenceError::TimedOut { value, completed } => {
                write!(f, "timed out at {completed} waiting for {value}")
            }
            FenceError::Refused(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for FenceError {}

/// A device's monotonic completion counter.
///
/// Values count committed command buffers. `completed <= submitted` always,
/// and both only grow. A buffer's last-write and last-use stamps are values
/// of this timeline; a host lease waits for the right one.
///
/// Every method checks poison first: on a poisoned device each returns
/// [`FenceError::Poisoned`], including a `wait` whose value was already
/// reached. The defaults refuse, so a backend with no timeline implements
/// nothing.
pub trait Timeline: Send + Sync {
    /// The highest value the device has been asked to run (committed).
    fn submitted(&self) -> Result<u64, FenceError> {
        Err(FenceError::unsupported(
            "Timeline::submitted",
            "a device timeline",
        ))
    }

    /// The highest value the device has finished. Never above
    /// [`Timeline::submitted`].
    fn completed(&self) -> Result<u64, FenceError> {
        Err(FenceError::unsupported(
            "Timeline::completed",
            "a device timeline",
        ))
    }

    /// Block until [`Timeline::completed`] reaches `value`. A `value` above
    /// [`Timeline::submitted`] answers [`FenceError::NotSubmitted`] at once
    /// instead of waiting: nothing waits on an uncommitted value. A `value`
    /// already completed returns without blocking.
    fn wait(&self, _value: u64) -> Result<(), FenceError> {
        Err(FenceError::unsupported(
            "Timeline::wait",
            "a device timeline",
        ))
    }
}

/// A shared, read-only borrow of a [`MappedBuffer`]'s bytes.
///
/// Holding it keeps the buffer from being written by the device (a GPU bind
/// that would write it is refused) and from being freed. Dropping it
/// releases the borrow. `Send + Sync` so a lease can move to a worker
/// thread; the implementation that makes that sound is the device
/// extension's, not this crate's.
pub trait HostLease: Send + Sync {
    /// The buffer's bytes, the whole allocation.
    fn as_bytes(&self) -> &[u8];

    /// The bytes as `f32`. [`OjasError::Shape`] if the length is not a
    /// multiple of 4 or the address is not 4-aligned. The default refuses: a
    /// lease built without a cast cannot view the bytes as another type.
    fn as_f32(&self) -> Result<&[f32], OjasError> {
        Err(OjasError::Unsupported {
            op: "HostLease::as_f32",
            detail: "this lease does not view its bytes as f32".to_string(),
        })
    }

    /// The bytes as `u32`; the same rules as [`HostLease::as_f32`].
    fn as_u32(&self) -> Result<&[u32], OjasError> {
        Err(OjasError::Unsupported {
            op: "HostLease::as_u32",
            detail: "this lease does not view its bytes as u32".to_string(),
        })
    }
}

/// An exclusive, writable borrow of a [`MappedBuffer`]'s bytes.
///
/// It reads too ([`HostLease`] is a supertrait). A tensor built over a
/// buffer a write lease touched must forget what it cached about the bytes:
/// the all-finite stamp resets to unknown and the compute tag to f32, as
/// `host_data_mut` does for a host tensor. That reset is the tensor layer's
/// job and is not part of this contract.
pub trait HostLeaseMut: HostLease {
    /// The buffer's bytes, writable.
    fn as_bytes_mut(&mut self) -> &mut [u8];

    /// The bytes as writable `f32`; the rules of [`HostLease::as_f32`].
    fn as_f32_mut(&mut self) -> Result<&mut [f32], OjasError> {
        Err(OjasError::Unsupported {
            op: "HostLeaseMut::as_f32_mut",
            detail: "this lease does not view its bytes as f32".to_string(),
        })
    }

    /// The bytes as writable `u32`; the rules of [`HostLease::as_f32`].
    fn as_u32_mut(&mut self) -> Result<&mut [u32], OjasError> {
        Err(OjasError::Unsupported {
            op: "HostLeaseMut::as_u32_mut",
            detail: "this lease does not view its bytes as u32".to_string(),
        })
    }
}

/// A [`DeviceBuffer`] whose bytes the host can borrow in place.
///
/// Implemented by a backend whose memory is host-visible. The default
/// methods refuse, so an extension that cannot lend its memory implements
/// nothing and keeps [`DeviceBuffer::read_bytes`] as its only way back to
/// the host, a counted copy.
///
/// The fence rules in the module docs bind implementations: leases fence the
/// whole buffer, a read waits for the last write, a write waits for the last
/// use and is exclusive, and a stamp the device has not committed answers
/// [`FenceError::NotSubmitted`] instead of waiting.
pub trait MappedBuffer: DeviceBuffer {
    /// Borrow the whole buffer for reading, after the device finished its
    /// last write.
    fn host_read(&self) -> Result<Box<dyn HostLease>, FenceError> {
        Err(FenceError::unsupported(
            "MappedBuffer::host_read",
            "a host read lease",
        ))
    }

    /// Borrow the whole buffer for writing, after the device finished its
    /// last use. Exclusive: refused with [`FenceError::Busy`] while any
    /// lease is live.
    fn host_write(&self) -> Result<Box<dyn HostLeaseMut>, FenceError> {
        Err(FenceError::unsupported(
            "MappedBuffer::host_write",
            "a host write lease",
        ))
    }
}

/// Which arithmetic tier produced a buffer's bytes.
///
/// It sits beside the compute tag a tensor already carries. The CPU's
/// [`Numerics::Exact`] contract is a property of its fixed reduction order,
/// so a mapped operand changes neither order nor alignment assumptions; but
/// bytes a GPU wrote come from a [`Numerics::Fast`] backend and are not
/// bit-reproducible against the CPU reference. The tag records that, so an
/// Exact path can allow, record or refuse such an operand by policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Provenance {
    /// Written by the host or by a backend under [`Numerics::Exact`].
    #[default]
    Exact,
    /// Written by a backend under [`Numerics::Fast`].
    Fast,
}

impl Provenance {
    /// The tier of a result computed only from operands of tiers `self` and
    /// `other`: Exact only if both are.
    pub fn join(self, other: Provenance) -> Provenance {
        match (self, other) {
            (Provenance::Exact, Provenance::Exact) => Provenance::Exact,
            _ => Provenance::Fast,
        }
    }

    /// Whether the bytes carry the Exact bit-for-bit guarantee.
    pub fn is_exact(self) -> bool {
        self == Provenance::Exact
    }
}

impl From<Numerics> for Provenance {
    fn from(n: Numerics) -> Provenance {
        match n {
            Numerics::Exact => Provenance::Exact,
            Numerics::Fast => Provenance::Fast,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;

    #[derive(Debug)]
    struct Plain;

    impl DeviceBuffer for Plain {
        fn backend(&self) -> BackendId {
            BackendId::Metal
        }
        fn byte_len(&self) -> usize {
            16
        }
        fn read_bytes(&self, _offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
            Ok(vec![0; len])
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl MappedBuffer for Plain {}

    struct NoTimeline;
    impl Timeline for NoTimeline {}

    struct Bytes(Vec<u8>);
    impl HostLease for Bytes {
        fn as_bytes(&self) -> &[u8] {
            &self.0
        }
    }
    impl HostLeaseMut for Bytes {
        fn as_bytes_mut(&mut self) -> &mut [u8] {
            &mut self.0
        }
    }

    fn is_unsupported(e: &OjasError, op: &str) -> bool {
        matches!(e, OjasError::Unsupported { op: got, .. } if *got == op)
    }

    fn refused_op(e: FenceError) -> Option<&'static str> {
        match e {
            FenceError::Refused(OjasError::Unsupported { op, .. }) => Some(op),
            _ => None,
        }
    }

    /// Calls every method of the three defaulted traits on a minimal
    /// implementor and reports `(trait, method, refused-with-its-own-name)`.
    fn default_calls() -> Vec<(&'static str, &'static str, bool)> {
        let plain = Plain;
        let timeline = NoTimeline;
        let mut lease = Bytes(vec![0; 8]);
        vec![
            (
                "Timeline",
                "submitted",
                refused_op(timeline.submitted().unwrap_err()) == Some("Timeline::submitted"),
            ),
            (
                "Timeline",
                "completed",
                refused_op(timeline.completed().unwrap_err()) == Some("Timeline::completed"),
            ),
            (
                "Timeline",
                "wait",
                refused_op(timeline.wait(1).unwrap_err()) == Some("Timeline::wait"),
            ),
            ("HostLease", "as_bytes", lease.as_bytes().len() == 8),
            (
                "HostLease",
                "as_f32",
                is_unsupported(&lease.as_f32().unwrap_err(), "HostLease::as_f32"),
            ),
            (
                "HostLease",
                "as_u32",
                is_unsupported(&lease.as_u32().unwrap_err(), "HostLease::as_u32"),
            ),
            (
                "HostLeaseMut",
                "as_bytes_mut",
                lease.as_bytes_mut().len() == 8,
            ),
            (
                "HostLeaseMut",
                "as_f32_mut",
                is_unsupported(&lease.as_f32_mut().unwrap_err(), "HostLeaseMut::as_f32_mut"),
            ),
            (
                "HostLeaseMut",
                "as_u32_mut",
                is_unsupported(&lease.as_u32_mut().unwrap_err(), "HostLeaseMut::as_u32_mut"),
            ),
            (
                "MappedBuffer",
                "host_read",
                refused_op(plain.host_read().err().unwrap()) == Some("MappedBuffer::host_read"),
            ),
            (
                "MappedBuffer",
                "host_write",
                refused_op(plain.host_write().err().unwrap()) == Some("MappedBuffer::host_write"),
            ),
        ]
    }

    /// The `fn` names declared in `pub trait <name>` in this file's source.
    fn trait_methods(name: &str) -> Vec<&'static str> {
        let source = include_str!("hal.rs");
        let needle = format!("pub trait {name}");
        let start = source
            .match_indices(&needle)
            .map(|(i, _)| i)
            .find(|&i| {
                !source[i + needle.len()..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_')
            })
            .unwrap_or_else(|| panic!("no trait {name} in hal.rs"));
        let end = start + source[start..].find("\n}\n").unwrap();
        source[start..end]
            .lines()
            .filter_map(|line| {
                let rest = line.strip_prefix("    fn ")?;
                Some(&rest[..rest.find(['(', '<'])?])
            })
            .collect()
    }

    /// The default-refusal harness above must call exactly the methods the
    /// traits declare: a method added to a trait without a default-behaviour
    /// check (or a stale check) fails here, as
    /// `the_method_list_names_every_trait_method` does for `Backend`.
    #[test]
    fn the_hal_list_names_every_trait_method() {
        let calls = default_calls();
        for name in ["Timeline", "HostLease", "HostLeaseMut", "MappedBuffer"] {
            let mut declared = trait_methods(name);
            let mut checked: Vec<&str> = calls
                .iter()
                .filter(|(t, _, _)| *t == name)
                .map(|(_, m, _)| *m)
                .collect();
            declared.sort_unstable();
            checked.sort_unstable();
            assert!(!declared.is_empty(), "the scan found no methods of {name}");
            assert_eq!(
                declared, checked,
                "{name}: trait methods and default checks disagree"
            );
        }
    }

    #[test]
    fn every_default_refuses_or_is_the_required_accessor() {
        for (t, m, ok) in default_calls() {
            assert!(ok, "{t}::{m} did not behave as its default should");
        }
    }

    #[test]
    fn a_refusing_default_names_no_host_fallback() {
        let e = Plain.host_read().err().unwrap();
        assert!(!e.is_retryable(), "a refusal is permanent");
        assert!(e.to_string().contains("no host fallback"), "{e}");
        let ojas = e.into_ojas(BackendId::Metal);
        assert!(is_unsupported(&ojas, "MappedBuffer::host_read"), "{ojas}");
    }

    #[test]
    fn a_mapped_buffer_is_still_a_device_buffer() {
        let b: &dyn MappedBuffer = &Plain;
        assert_eq!(b.byte_len(), 16);
        assert_eq!(b.backend(), BackendId::Metal);
        assert_eq!(b.read_bytes(0, 4).unwrap(), vec![0; 4]);
    }

    #[test]
    fn the_lease_traits_are_send_and_sync() {
        fn needs_send_sync<T: ?Sized + Send + Sync>() {}
        needs_send_sync::<dyn HostLease>();
        needs_send_sync::<dyn HostLeaseMut>();
        needs_send_sync::<dyn Timeline>();
        needs_send_sync::<dyn MappedBuffer>();
    }

    #[test]
    fn a_write_lease_reads_through_its_supertrait() {
        let mut lease = Bytes(vec![1, 2, 3, 4]);
        lease.as_bytes_mut()[0] = 9;
        let read: &dyn HostLease = &lease;
        assert_eq!(read.as_bytes(), &[9, 2, 3, 4]);
    }

    #[test]
    fn capacity_kinds_map_to_capacity_exceeded_with_the_same_numbers() {
        for e in [
            AllocError::OverBudget {
                requested: 64,
                cap: 100,
                live: 90,
            },
            AllocError::Exhausted {
                requested: 64,
                cap: 100,
                live: 90,
            },
            AllocError::TooLarge {
                requested: 64,
                cap: 100,
                live: 90,
            },
        ] {
            assert_eq!(e.requested(), Some(64));
            match OjasError::from(e.clone()) {
                OjasError::CapacityExceeded {
                    requested,
                    cap,
                    live,
                } => assert_eq!((requested, cap, live), (64, 100, 90), "{e}"),
                other => panic!("{e} became {other}"),
            }
        }
    }

    #[test]
    fn device_loss_maps_to_device_lost_and_has_no_size() {
        let e = AllocError::DeviceLost {
            backend: BackendId::Metal,
            detail: "poisoned".to_string(),
        };
        assert_eq!(e.requested(), None);
        assert!(!e.is_retryable());
        match OjasError::from(e) {
            OjasError::DeviceLost { backend, detail } => {
                assert_eq!(backend, BackendId::Metal);
                assert_eq!(detail, "poisoned");
            }
            other => panic!("became {other}"),
        }
    }

    #[test]
    fn only_state_dependent_allocation_failures_are_retryable() {
        let (requested, cap, live) = (1, 2, 3);
        assert!(AllocError::OverBudget {
            requested,
            cap,
            live
        }
        .is_retryable());
        assert!(AllocError::Exhausted {
            requested,
            cap,
            live
        }
        .is_retryable());
        assert!(!AllocError::TooLarge {
            requested,
            cap,
            live
        }
        .is_retryable());
    }

    #[test]
    fn fence_errors_fold_into_ojas_errors() {
        let id = BackendId::Metal;
        assert!(matches!(
            FenceError::Poisoned.into_ojas(id),
            OjasError::DeviceLost {
                backend: BackendId::Metal,
                ..
            }
        ));
        assert!(matches!(
            FenceError::TimedOut {
                value: 5,
                completed: 3
            }
            .into_ojas(id),
            OjasError::DeviceLost { .. }
        ));
        assert!(matches!(
            FenceError::NotSubmitted {
                value: 5,
                submitted: 4
            }
            .into_ojas(id),
            OjasError::Backend {
                id: BackendId::Metal,
                ..
            }
        ));
        assert!(matches!(
            FenceError::Busy {
                detail: "x".to_string()
            }
            .into_ojas(id),
            OjasError::Backend { .. }
        ));
        let inner = OjasError::Poisoned;
        assert!(matches!(
            FenceError::Refused(inner).into_ojas(id),
            OjasError::Poisoned
        ));
    }

    #[test]
    fn retryability_matches_the_documented_cases() {
        assert!(FenceError::NotSubmitted {
            value: 2,
            submitted: 1
        }
        .is_retryable());
        assert!(FenceError::Busy {
            detail: String::new()
        }
        .is_retryable());
        assert!(FenceError::TimedOut {
            value: 2,
            completed: 1
        }
        .is_retryable());
        assert!(!FenceError::Poisoned.is_retryable());
        assert!(!FenceError::unsupported("op", "x").is_retryable());
    }

    #[test]
    fn provenance_is_exact_only_when_every_operand_is() {
        use Provenance::{Exact, Fast};
        assert_eq!(Provenance::default(), Exact);
        assert_eq!(Exact.join(Exact), Exact);
        assert_eq!(Exact.join(Fast), Fast);
        assert_eq!(Fast.join(Exact), Fast);
        assert_eq!(Fast.join(Fast), Fast);
        assert!(Exact.is_exact());
        assert!(!Fast.is_exact());
    }

    #[test]
    fn provenance_follows_the_backend_numerics() {
        assert_eq!(Provenance::from(Numerics::Exact), Provenance::Exact);
        assert_eq!(Provenance::from(Numerics::Fast), Provenance::Fast);
    }

    /// A model implementation of the documented rules: it exercises the
    /// trait shapes (a `Box<dyn HostLease>` from `&self`, a timeline shared
    /// by reference) the way a device extension will, so a signature that
    /// cannot express the fence protocol fails to compile here.
    #[test]
    fn the_fence_protocol_is_expressible_against_the_traits() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::{Arc, Mutex};

        struct Clock {
            submitted: AtomicU64,
            completed: AtomicU64,
            poisoned: std::sync::atomic::AtomicBool,
        }
        impl Clock {
            fn check(&self) -> Result<(), FenceError> {
                if self.poisoned.load(Ordering::SeqCst) {
                    Err(FenceError::Poisoned)
                } else {
                    Ok(())
                }
            }
        }
        impl Timeline for Clock {
            fn submitted(&self) -> Result<u64, FenceError> {
                self.check()?;
                Ok(self.submitted.load(Ordering::SeqCst))
            }
            fn completed(&self) -> Result<u64, FenceError> {
                self.check()?;
                Ok(self.completed.load(Ordering::SeqCst))
            }
            fn wait(&self, value: u64) -> Result<(), FenceError> {
                self.check()?;
                let submitted = self.submitted.load(Ordering::SeqCst);
                if value > submitted {
                    return Err(FenceError::NotSubmitted { value, submitted });
                }
                // The model device finishes everything it was given.
                self.completed.fetch_max(value, Ordering::SeqCst);
                Ok(())
            }
        }

        #[derive(Debug)]
        struct Buf {
            clock: Arc<Clock>,
            last_write: u64,
            leased: Mutex<bool>,
        }
        impl DeviceBuffer for Buf {
            fn backend(&self) -> BackendId {
                BackendId::Metal
            }
            fn byte_len(&self) -> usize {
                4
            }
            fn read_bytes(&self, _o: usize, len: usize) -> Result<Vec<u8>, OjasError> {
                Ok(vec![0; len])
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
        impl fmt::Debug for Clock {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("Clock")
            }
        }
        impl MappedBuffer for Buf {
            fn host_read(&self) -> Result<Box<dyn HostLease>, FenceError> {
                self.clock.wait(self.last_write)?;
                if *self.leased.lock().unwrap() {
                    return Err(FenceError::Busy {
                        detail: "a write lease is live".to_string(),
                    });
                }
                Ok(Box::new(Bytes(vec![7; 4])))
            }
        }

        let clock = Arc::new(Clock {
            submitted: AtomicU64::new(3),
            completed: AtomicU64::new(1),
            poisoned: std::sync::atomic::AtomicBool::new(false),
        });
        let committed = Buf {
            clock: clock.clone(),
            last_write: 3,
            leased: Mutex::new(false),
        };
        let lease = committed.host_read().unwrap();
        assert_eq!(lease.as_bytes(), &[7; 4]);
        assert_eq!(clock.completed().unwrap(), 3);

        let uncommitted = Buf {
            clock: clock.clone(),
            last_write: 4,
            leased: Mutex::new(false),
        };
        match uncommitted.host_read().err().unwrap() {
            FenceError::NotSubmitted { value, submitted } => assert_eq!((value, submitted), (4, 3)),
            other => panic!("{other}"),
        }
        *committed.leased.lock().unwrap() = true;
        assert!(matches!(
            committed.host_read().err().unwrap(),
            FenceError::Busy { .. }
        ));

        clock.poisoned.store(true, Ordering::SeqCst);
        for r in [
            clock.wait(0).err(),
            clock.completed().err(),
            clock.submitted().err(),
            committed.host_read().err(),
        ] {
            assert!(matches!(r, Some(FenceError::Poisoned)));
        }
    }
}
