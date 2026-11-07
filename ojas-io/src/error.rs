use std::fmt;

/// Recoverable IO or format failure. Library paths return this instead of panicking.
#[derive(Debug)]
pub struct IoError {
    detail: String,
    refusal: bool,
}

impl IoError {
    /// Public so a [`crate::replace_dir_with`] fill closure can report its own
    /// failures.
    pub fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
            refusal: false,
        }
    }

    /// A refusal of the content or its size (a byte cap, invalid UTF-8), as
    /// opposed to a failure of the file system. See [`Self::is_refusal`].
    pub(crate) fn refusal(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
            refusal: true,
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

impl std::error::Error for IoError {}

impl From<IoError> for String {
    fn from(err: IoError) -> Self {
        err.detail
    }
}
