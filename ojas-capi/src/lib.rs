//! In-process load, train, save, resume, tokenize, sample and free for the
//! Go API (`docs/framework-design.md` §7; the opcode table is in
//! [`engine`]).
//!
//! Each session holds a nanolab model on its device: a CPU session on
//! [`ojas_cpu::CpuBackend`], a Metal session on its own
//! `ojas_metal::MetalBackend`, a wgpu session on its own
//! [`ojas_wgpu::WgpuBackend`] (with no adapter the load fails and no session
//! is created). The parameters stay on that device; a step reads back its
//! loss, a sample one logit row per forward. GENERATE with caller logits
//! calls [`ojas_infer::argmax_token`] on the host for every device.

#![forbid(unsafe_code)]

mod engine;
mod gate;
mod generate;
mod load;
mod model;
mod owner;
mod session;
mod tokenize;
mod train;
mod wire;

pub use engine::{
    dispatch, install_engine, OP_FREE, OP_GENERATE, OP_INSPECT, OP_LOAD, OP_NEW, OP_PANIC,
    OP_RESUME, OP_SAMPLE, OP_SAVE, OP_SET_MEMORY_CEILING, OP_STEP_RETIRED, OP_TOKENIZE,
    OP_TOKENIZER, OP_TRAIN_OPEN, OP_TRAIN_STEP,
};
pub use generate::{argmax, GEN_LOGITS};
pub use load::{inspect, resolve_dir_under_root, resolve_under_root, INLINE_PATH_MAX};
pub use session::{
    clear_last_error, clear_sessions, last_error, memory_ceiling, reset_sessions, session_count,
    set_last_error, set_memory_ceiling, set_model_root, take_last_error, try_free, DeviceKind,
    Session, DEFAULT_MEMORY_CEILING_BYTES,
};

/// The in-band kinds the Go side maps to sentinel errors (`go/ffi.go`).
///
/// A kind is chosen where a typed [`ojas_core::OjasError`] (or a check with
/// a known meaning) produces the error, never by reading message text: the
/// text carries user paths and tensor names.
///
/// On Metal and wgpu a non-finite or device-lost fault is reported at the
/// device's next sync point. Every call ends with [`settled`], so the kind
/// may come from an earlier op of the same call, never from an earlier call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ErrorKind {
    Capacity,
    DeviceLost,
    NonFinite,
    /// Another call holds the session's model (its `try_lock` failed).
    Busy,
    /// The model's trainer was left partly updated by a failed optimizer
    /// step ([`ojas_core::OjasError::Poisoned`]), or a call panicked while
    /// holding the model. Resume from a checkpoint.
    Poisoned,
}

impl ErrorKind {
    pub(crate) fn prefix(self) -> &'static str {
        match self {
            ErrorKind::Capacity => "ojas:E_CAPACITY: ",
            ErrorKind::DeviceLost => "ojas:E_DEVICE_LOST: ",
            ErrorKind::NonFinite => "ojas:E_NONFINITE: ",
            ErrorKind::Busy => "ojas:E_BUSY: ",
            ErrorKind::Poisoned => "ojas:E_POISONED: ",
        }
    }
}

/// `msg` with the kind's prefix.
pub(crate) fn kinded(kind: ErrorKind, msg: impl std::fmt::Display) -> String {
    format!("{}{msg}", kind.prefix())
}

/// The kind an error carries, by variant: [`OjasError::CapacityExceeded`],
/// [`OjasError::NonFinite`] and [`OjasError::Poisoned`] map directly.
/// `OjasError` has no device-lost variant, so an [`OjasError::Backend`] is
/// device-lost when its `detail`, which only a backend writes, contains the
/// markers `ojas-wgpu` (`context.rs`, "device lost") and `ojas-metal`
/// (`gpu.rs`, "runtime poisoned") emit. Every other variant has no kind,
/// whatever its text says.
///
/// [`OjasError::CapacityExceeded`]: ojas_core::OjasError::CapacityExceeded
/// [`OjasError::NonFinite`]: ojas_core::OjasError::NonFinite
/// [`OjasError::Poisoned`]: ojas_core::OjasError::Poisoned
/// [`OjasError::Backend`]: ojas_core::OjasError::Backend
pub(crate) fn kind_of(err: &ojas_core::OjasError) -> Option<ErrorKind> {
    use ojas_core::OjasError;
    match err {
        OjasError::CapacityExceeded { .. } => Some(ErrorKind::Capacity),
        OjasError::NonFinite { .. } => Some(ErrorKind::NonFinite),
        OjasError::Poisoned => Some(ErrorKind::Poisoned),
        OjasError::Backend { detail, .. }
            if detail.contains("device lost") || detail.contains("runtime poisoned") =>
        {
            Some(ErrorKind::DeviceLost)
        }
        _ => None,
    }
}

/// Finish one C-ABI call on a session's device `backend`: drain the faults it
/// deferred before the call returns, so a fault recorded by this call is never
/// reported by the session's next one.
///
/// A deferring backend (Metal and wgpu; see their contracts) records a
/// non-finite fault and reports it at its next sync point. A call that
/// returns early, on a cancel or a later error, would otherwise leave it
/// pending. The deferred fault was recorded before whatever ended the call,
/// so it is the first error in recording order and it wins over `result`'s
/// error.
pub(crate) fn settled<B: ojas_core::Backend + ?Sized, T>(
    context: &str,
    backend: &B,
    result: Result<T, String>,
) -> Result<T, String> {
    match backend.sync() {
        Ok(()) => result,
        Err(fault) => Err(ojas_error(context, &fault)),
    }
}

/// `context: err`, prefixed with the kind of `err` when it has one.
pub(crate) fn ojas_error(context: &str, err: &ojas_core::OjasError) -> String {
    match kind_of(err) {
        Some(kind) => kinded(kind, format!("{context}: {err}")),
        None => format!("{context}: {err}"),
    }
}

/// As [`ojas_error`] for a device open: only
/// [`ojas_device::DeviceError::Capacity`] has a kind.
pub(crate) fn device_error(context: &str, err: &ojas_device::DeviceError) -> String {
    match err {
        ojas_device::DeviceError::Capacity { .. } => {
            kinded(ErrorKind::Capacity, format!("{context}: {err}"))
        }
        _ => format!("{context}: {err}"),
    }
}

pub const SESSION_CAP: usize = session::SESSION_CAP;

#[cfg(test)]
mod ops_tests;
#[cfg(test)]
mod tests;
