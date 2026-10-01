//! The one error type of this crate.

use std::fmt;

use ojas_core::OjasError;
use ojas_io::IoError;

/// Every refusal and failure, named. Nothing in this crate falls back to a
/// default when one of these is the answer.
#[derive(Debug)]
pub enum Qwen35Error {
    /// `config.json` (or a snapshot index) describes something this provider
    /// does not implement, or cannot be read.
    Config(String),
    /// A capability this provider does not have today. `needs` names what
    /// would add it.
    Unsupported { what: String, needs: String },
    /// A caller argument, refused before anything runs.
    Invalid { op: &'static str, detail: String },
    /// A file could not be read, written or validated.
    Io(String),
    /// ojas-core refused (AdamW scalars, clip coefficient, tensor shape).
    Ojas(OjasError),
    /// tessl refused or failed.
    Tessl { op: &'static str, detail: String },
    /// An earlier failure left device state partly written. Nothing runs on
    /// this provider again; open a new one.
    Poisoned(String),
}

impl fmt::Display for Qwen35Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Qwen35Error::Config(d) => write!(f, "config: {d}"),
            Qwen35Error::Unsupported { what, needs } => write!(f, "unsupported: {what} (needs {needs})"),
            Qwen35Error::Invalid { op, detail } => write!(f, "{op}: {detail}"),
            Qwen35Error::Io(d) => write!(f, "io: {d}"),
            Qwen35Error::Ojas(e) => write!(f, "ojas-core: {e}"),
            Qwen35Error::Tessl { op, detail } => write!(f, "tessl: {op}: {detail}"),
            Qwen35Error::Poisoned(d) => write!(f, "poisoned: {d}"),
        }
    }
}

impl std::error::Error for Qwen35Error {}

impl From<OjasError> for Qwen35Error {
    fn from(e: OjasError) -> Self {
        Qwen35Error::Ojas(e)
    }
}

impl From<IoError> for Qwen35Error {
    fn from(e: IoError) -> Self {
        Qwen35Error::Io(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Qwen35Error>;

/// `Invalid` with an owned detail.
pub(crate) fn invalid(op: &'static str, detail: impl Into<String>) -> Qwen35Error {
    Qwen35Error::Invalid {
        op,
        detail: detail.into(),
    }
}

/// tessl's `String` errors, tagged with the call that produced them.
pub(crate) fn tessl(op: &'static str) -> impl Fn(String) -> Qwen35Error {
    move |detail| Qwen35Error::Tessl { op, detail }
}
