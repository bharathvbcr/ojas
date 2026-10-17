//! AdamW over a list of trainable tensors, all or nothing.
//!
//! [`Backend::adamw_step`] is one parameter: it checks its own step and
//! writes only if that step is finite. Called once per tensor, a
//! non-finite step in tensor `k` leaves tensors `0..k` stepped and the rest
//! not, and the moments of those first `k` have moved with them. That
//! contract is unchanged; [`adamw_all_or_nothing`] wraps it in two passes.
//!
//! **Pass 1** touches nothing. For every entry it checks the four operands'
//! shapes and `F32` dtype ([`adamw_step_dims`]), the config and step
//! ([`check_adamw`], so a bad config is refused before tensor 0 moves, not
//! at tensor 0), that the parameter and both moments can be updated in
//! place (uniquely owned, not shared with a clone the caller still holds;
//! a host tensor also contiguous and finite), and that every gradient value
//! is safe.
//!
//! **Pass 2** calls [`Backend::adamw_step`] on each entry in order, then
//! [`Backend::sync`] so a deferring backend's fault surfaces here.
//!
//! # What "safe" means, and what it costs
//!
//! `Backend` has no finite-check op, so pass 1 reads every gradient to the
//! host ([`Tensor::to_host`], after one [`Backend::sync`]) and scans it: a
//! value is safe when `g * g` is a finite f32, i.e. `g` is finite and
//! `|g| <= `[`GRAD_ABS_LIMIT`] (`2^64 - 2^40`). That is exactly the
//! condition for the second moment `v' = b2 v + (1 - b2) g^2` and the first
//! `m' = b1 m + (1 - b1) g` to stay finite: both are convex combinations of
//! a finite prior moment and a finite term, so neither can overflow, and
//! the update `lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)` is then finite
//! for finite `p`, `lr` and `eps > 0`. It assumes finite parameters and
//! moments on entry for device tensors (a host tensor is scanned), which
//! holds for state this module and
//! [`super::adapters`] produced (the checkpoint reader refuses a non-finite
//! one). It does not download the parameters or moments; that would be
//! four times the traffic.
//!
//! Cost: one readback of each gradient (`4 * n` bytes per tensor, charged to
//! the `budget` passed in, one tensor at a time) and one scan of it. On a
//! host-resident tensor `to_host` is a handle copy and the cost is the
//! scan alone. The backend's own per-tensor device check still runs inside
//! pass 2; if it nevertheless refuses entry `k` (a state the scan cannot
//! see), the error is [`OptimError::Applied`] and says how many entries
//! had already been stepped. Nothing is retried.

use std::fmt;

use ojas_core::{
    adamw_step_dims, check_adamw, AdamWConfig, Backend, Budget, DType, OjasError, Tensor,
};

/// The largest f32 whose square is a finite f32 (`2^64 - 2^40`, just under
/// `sqrt(f32::MAX)`): the largest gradient magnitude pass 1 accepts.
pub const GRAD_ABS_LIMIT: f32 = 1.844_674_3e19;

/// One tensor's step: its parameter, gradient, both moments and the config
/// (already carrying this entry's learning rate and weight decay).
pub struct AdamWEntry<'a> {
    pub name: &'a str,
    pub param: &'a mut Tensor,
    pub grad: &'a Tensor,
    pub moment1: &'a mut Tensor,
    pub moment2: &'a mut Tensor,
    pub config: AdamWConfig,
}

/// Why [`adamw_all_or_nothing`] stopped.
#[derive(Debug)]
pub enum OptimError {
    /// Pass 1 found a gradient value that is NaN, infinite or too large to
    /// square. Nothing was written.
    NonFiniteGrad {
        entry: usize,
        name: String,
        /// Index of the first such value in row-major order.
        position: usize,
        value: f32,
    },
    /// Pass 1 refused entry `entry`'s operands, config or step, or could not
    /// read its gradient. Nothing was written.
    Refused {
        entry: usize,
        name: String,
        source: OjasError,
    },
    /// Pass 1 could not synchronise the backend. Nothing was written.
    Sync(OjasError),
    /// Pass 2: the backend refused entry `entry` after pass 1 passed. The
    /// `applied` entries before it were stepped.
    Applied {
        entry: usize,
        name: String,
        applied: usize,
        source: OjasError,
    },
    /// Pass 2 stepped every entry, then the backend reported a fault on
    /// the closing sync: every entry must be treated as invalid.
    SyncAfterApply(OjasError),
}

impl fmt::Display for OptimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteGrad {
                entry,
                name,
                position,
                value,
            } => write!(
                f,
                "adamw: entry {entry} ({name}): gradient value {value} at element {position} \
                 is not finite or overflows when squared; no tensor was stepped"
            ),
            Self::Refused {
                entry,
                name,
                source,
            } => write!(
                f,
                "adamw: entry {entry} ({name}): {source}; no tensor was stepped"
            ),
            Self::Sync(e) => write!(
                f,
                "adamw: backend sync before the update: {e}; no tensor was stepped"
            ),
            Self::Applied {
                entry,
                name,
                applied,
                source,
            } => write!(
                f,
                "adamw: entry {entry} ({name}): {source}; {applied} earlier entries were stepped"
            ),
            Self::SyncAfterApply(e) => {
                write!(f, "adamw: backend fault after every entry was stepped: {e}")
            }
        }
    }
}

impl std::error::Error for OptimError {}

/// Step every entry once at `step` (the count before this update, as
/// [`Backend::adamw_step`] takes it), or none. See the module docs for the
/// two passes, the finiteness condition and its cost. `budget` is charged
/// for one downloaded gradient at a time.
pub fn adamw_all_or_nothing<B: Backend + ?Sized>(
    backend: &B,
    entries: &mut [AdamWEntry<'_>],
    step: u64,
    budget: &Budget,
) -> Result<(), OptimError> {
    backend.sync().map_err(OptimError::Sync)?;
    for (i, e) in entries.iter_mut().enumerate() {
        check_entry(i, e, step, budget)?;
    }
    for (i, e) in entries.iter_mut().enumerate() {
        backend
            .adamw_step(e.param, e.grad, e.moment1, e.moment2, step, e.config)
            .map_err(|source| OptimError::Applied {
                entry: i,
                name: e.name.to_string(),
                applied: i,
                source,
            })?;
    }
    backend.sync().map_err(OptimError::SyncAfterApply)
}

/// Whether a backend's in-place update of `t` would be accepted, without
/// writing: a uniquely owned allocation (a host tensor also contiguous), and
/// for a host tensor finite values. A clone of a parameter or moment kept
/// alive by the caller, or a non-finite host moment, makes the backend refuse
/// at that entry in pass 2, after the entries before it have moved.
fn updatable(t: &mut Tensor) -> Result<(), OjasError> {
    if t.device().is_some() {
        return t.device_buffer_mut().map(|_| ());
    }
    let n = t.num_elements()?;
    t.ensure_writable_f32(n)?;
    if t.all_finite_cached(|s| Ok(s.iter().all(|x| x.is_finite())))? {
        Ok(())
    } else {
        Err(OjasError::NonFinite {
            op: "adamw_all_or_nothing",
        })
    }
}

fn check_entry(
    i: usize,
    e: &mut AdamWEntry<'_>,
    step: u64,
    budget: &Budget,
) -> Result<(), OptimError> {
    let name: &str = e.name;
    let refused = |source: OjasError| OptimError::Refused {
        entry: i,
        name: name.to_string(),
        source,
    };
    adamw_step_dims(&*e.param, e.grad, &*e.moment1, &*e.moment2).map_err(refused)?;
    for t in [&*e.param, e.grad, &*e.moment1, &*e.moment2] {
        if t.dtype() != DType::F32 {
            return Err(refused(OjasError::Dtype {
                op: "adamw_all_or_nothing",
                expected: DType::F32,
                got: t.dtype(),
            }));
        }
    }
    check_adamw(e.config, step).map_err(refused)?;
    for t in [&mut *e.param, &mut *e.moment1, &mut *e.moment2] {
        updatable(t).map_err(refused)?;
    }
    let host = e.grad.to_host(budget).map_err(refused)?;
    let values = host.f32_slice().map_err(refused)?;
    match values.iter().position(|g| !(g * g).is_finite()) {
        None => Ok(()),
        Some(position) => Err(OptimError::NonFiniteGrad {
            entry: i,
            name: e.name.to_string(),
            position,
            value: values[position],
        }),
    }
}
