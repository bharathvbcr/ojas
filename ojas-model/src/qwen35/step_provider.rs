//! The device seam of the Qwen3.5 step provider.
//!
//! The tower is generic over [`ojas_core::Backend`], and the trait has no
//! method for the two things a device-resident training step needs beyond
//! it: marking a frozen weight so the device refuses any write to it, and
//! rebinding the backend to a smaller [`Budget`] once a plan is known. Both
//! are inherent methods of the concrete backend (`MetalBackend::freeze_checked`
//! and `MetalBackend::with_budget`). A trait method would have to be
//! forwarded by every wrapper backend (autocast, tracing, test doubles), and
//! a wrapper that forgets one silently drops the guarantee (bug class
//! ft-fe7dcee6); a pair of closures the caller owns cannot be forgotten by a
//! wrapper, because no wrapper sits between them and the concrete backend.
//!
//! A Metal caller passes
//!
//! ```ignore
//! DeviceHooks {
//!     freeze: Some(Box::new(|m, t| m.freeze_checked(t))),
//!     rebind: Some(Box::new(|m, b| m.with_budget(b))),
//! }
//! ```
//!
//! and a CPU caller uses [`DeviceHooks::default`], which has neither: a
//! missing hook means the backend has no such notion, not that the step
//! skipped it.

use ojas_core::{Budget, OjasError, Tensor};

/// Freeze a weight on the device.
pub type FreezeHook<B> = Box<dyn Fn(&B, &Tensor) -> Result<(), OjasError>>;

/// Another handle on the backend that charges the given budget.
pub type RebindHook<B> = Box<dyn Fn(&B, Budget) -> B>;

/// The optional device operations a step provider needs from a concrete
/// backend `B`; see the module docs.
pub struct DeviceHooks<B> {
    /// Mark `tensor` frozen on `backend` so a write to it is refused.
    pub freeze: Option<FreezeHook<B>>,
    /// A handle on `backend`'s device charging a different budget.
    pub rebind: Option<RebindHook<B>>,
}

impl<B> Default for DeviceHooks<B> {
    fn default() -> Self {
        Self {
            freeze: None,
            rebind: None,
        }
    }
}
