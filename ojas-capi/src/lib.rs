//! In-process load, step, generate, and free for the Go API.
//!
//! Each session records where it computes. A CPU or CPU-parallel load runs
//! step and the greedy generate on [`ojas_cpu::CpuBackend`]. A Metal load
//! opens an `ojas_metal::MetalBackend` that the session keeps; its step and
//! greedy generate run on that device, and only the returned values (the
//! loss, the last row's two logits) are read back. A wgpu load does the same
//! with an [`ojas_wgpu::WgpuBackend`]; with no adapter the load fails and no
//! session is created. Generate with caller logits calls
//! [`ojas_infer::argmax_token`] on the host for every device.

#![forbid(unsafe_code)]

mod engine;
mod generate;
mod load;
mod owner;
mod session;
mod step;

pub use engine::{dispatch, install_engine, OP_FREE, OP_GENERATE, OP_LOAD, OP_PANIC, OP_STEP};
pub use generate::{generate, GEN_GREEDY, GEN_LOGITS};
pub use load::{resolve_under_root, INLINE_PATH_MAX};
pub use session::{
    clear_last_error, clear_sessions, last_error, load_model, reset_sessions, session_count,
    set_last_error, set_model_root, take_last_error, try_free, Session,
};

/// The in-band kinds the Go side maps to sentinel errors (`go/ffi.go`).
///
/// A kind is chosen where a typed [`ojas_core::OjasError`] (or a check with
/// a known meaning) produces the error, never by reading message text: the
/// text carries user paths and tensor names. Go also defines
/// `ojas:E_BUSY:`; nothing in Rust produces it, so it has no variant here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ErrorKind {
    Capacity,
    DeviceLost,
    NonFinite,
}

impl ErrorKind {
    pub(crate) fn prefix(self) -> &'static str {
        match self {
            ErrorKind::Capacity => "ojas:E_CAPACITY: ",
            ErrorKind::DeviceLost => "ojas:E_DEVICE_LOST: ",
            ErrorKind::NonFinite => "ojas:E_NONFINITE: ",
        }
    }
}

/// `msg` with the kind's prefix.
pub(crate) fn kinded(kind: ErrorKind, msg: impl std::fmt::Display) -> String {
    format!("{}{msg}", kind.prefix())
}

/// The kind an error carries, by variant: [`OjasError::CapacityExceeded`]
/// and [`OjasError::NonFinite`] map directly. `OjasError` has no device-lost
/// variant, so an [`OjasError::Backend`] is device-lost when its `detail`,
/// which only a backend writes, starts with or contains the markers
/// `ojas-wgpu` (`context.rs`, "device lost") and `ojas-metal` (`gpu.rs`,
/// "runtime poisoned") emit. Every other variant has no kind, whatever its
/// text says.
///
/// [`OjasError::CapacityExceeded`]: ojas_core::OjasError::CapacityExceeded
/// [`OjasError::NonFinite`]: ojas_core::OjasError::NonFinite
/// [`OjasError::Backend`]: ojas_core::OjasError::Backend
pub(crate) fn kind_of(err: &ojas_core::OjasError) -> Option<ErrorKind> {
    use ojas_core::OjasError;
    match err {
        OjasError::CapacityExceeded { .. } => Some(ErrorKind::Capacity),
        OjasError::NonFinite { .. } => Some(ErrorKind::NonFinite),
        OjasError::Backend { detail, .. }
            if detail.contains("device lost") || detail.contains("runtime poisoned") =>
        {
            Some(ErrorKind::DeviceLost)
        }
        _ => None,
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
pub use step::{
    step, step_with_policy, StepInput, StepStats, MODE_HEADER, MODE_LOGITS, MODE_TOKENS,
};

pub const SESSION_CAP: usize = session::SESSION_CAP;

#[cfg(test)]
mod tests;
