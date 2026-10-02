use std::fmt;

/// Recoverable IO or format failure. Library paths return this instead of panicking.
#[derive(Debug)]
pub struct IoError {
    detail: String,
}

impl IoError {
    /// Public so a [`crate::replace_dir_with`] fill closure can report its own
    /// failures.
    pub fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
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
