use std::fmt;

/// Recoverable IO or format failure. Library paths return this instead of panicking.
///
/// A failure that came from the operating system keeps the original
/// [`std::io::Error`]: [`Self::io_kind`] names its kind and
/// [`std::error::Error::source`] returns it, so a caller can branch on
/// `NotFound` or `PermissionDenied` without parsing [`fmt::Display`] text or
/// re-probing the path. A failure the crate itself raises (a refusal, a
/// format error, a cap) has no source and `io_kind()` is `None`. Not `Clone`,
/// because `std::io::Error` is not.
#[derive(Debug)]
pub struct IoError {
    detail: String,
    refusal: bool,
    source: Option<std::io::Error>,
}

impl IoError {
    /// Public so a [`crate::replace_dir_with`] fill closure can report its own
    /// failures.
    pub fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
            refusal: false,
            source: None,
        }
    }

    /// A file-system failure: `detail` is the full message (the caller formats
    /// it, usually `"{what}: {err}"`) and `err` is kept as the
    /// [`std::error::Error::source`]. Not a refusal.
    pub fn io(detail: impl Into<String>, err: std::io::Error) -> Self {
        Self {
            detail: detail.into(),
            refusal: false,
            source: Some(err),
        }
    }

    /// Re-describes `inner` under a new `detail` (a path or tensor prefix)
    /// and keeps the `std::io::Error` it carried, if any. Not a refusal,
    /// whatever `inner` was, which is what the format-only wrapping did.
    pub fn wrap(detail: impl Into<String>, inner: IoError) -> Self {
        Self {
            detail: detail.into(),
            refusal: false,
            source: inner.source,
        }
    }

    /// The kind of the `std::io::Error` behind this failure, or `None` when
    /// the crate raised it itself (a refusal, a format error, a cap).
    pub fn io_kind(&self) -> Option<std::io::ErrorKind> {
        self.source.as_ref().map(std::io::Error::kind)
    }

    /// A refusal of the content or its size (a byte cap, invalid UTF-8), as
    /// opposed to a failure of the file system. See [`Self::is_refusal`].
    pub(crate) fn refusal(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
            refusal: true,
            source: None,
        }
    }

    /// Whether a bounded reader ([`crate::read_regular_text`]) refused what it
    /// found (over the byte cap, or not UTF-8), rather than failing to open or
    /// read it. A caller maps the two to different error variants.
    pub fn is_refusal(&self) -> bool {
        self.refusal
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for IoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e as &(dyn std::error::Error + 'static))
    }
}

impl From<IoError> for String {
    fn from(err: IoError) -> Self {
        err.detail
    }
}
