//! Free space on the filesystem that holds a path.
//!
//! macOS reads `statfs`, whose block counts are 64-bit; its `statvfs` has a
//! 32-bit `fsblkcnt_t`. Linux reads `statvfs`. Either way the answer is the
//! space an unprivileged writer can still allocate (`f_bavail`), not the
//! root-reserved total (`f_bfree`). A failed call is an error carrying the
//! OS error; no size is guessed.

#![allow(unsafe_code)]

use std::io;
use std::path::Path;

/// Bytes an unprivileged writer can still allocate on the filesystem that
/// holds `path`, which must exist. Other platforms are
/// [`io::ErrorKind::Unsupported`].
pub fn available_disk_bytes(path: &Path) -> io::Result<u64> {
    imp::available(path)
}

/// `blocks * block_size`, saturating: a product past `u64` is still "more
/// than any write asks for".
fn bytes_of(blocks: u64, block_size: u64) -> u64 {
    blocks.saturating_mul(block_size)
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
mod imp {
    use std::ffi::CString;
    use std::io;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    fn c_path(path: &Path) -> io::Result<CString> {
        CString::new(path.as_os_str().as_bytes()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{}: path contains a NUL byte", path.display()),
            )
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn available(path: &Path) -> io::Result<u64> {
        let c = c_path(path)?;
        // SAFETY: an all-zero `statfs` is a valid value of this plain C struct.
        let mut st = unsafe { std::mem::zeroed::<libc::statfs>() };
        // SAFETY: `c` is NUL-terminated and outlives the call; `st` is
        // writable and is the struct `statfs` fills.
        let rc = unsafe { libc::statfs(c.as_ptr(), &mut st) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(super::bytes_of(st.f_bavail, u64::from(st.f_bsize)))
    }

    #[cfg(target_os = "linux")]
    pub(super) fn available(path: &Path) -> io::Result<u64> {
        let c = c_path(path)?;
        // SAFETY: an all-zero `statvfs` is a valid value of this plain C struct.
        let mut st = unsafe { std::mem::zeroed::<libc::statvfs>() };
        // SAFETY: `c` is NUL-terminated and outlives the call; `st` is
        // writable and is the struct `statvfs` fills.
        let rc = unsafe { libc::statvfs(c.as_ptr(), &mut st) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        // `fsblkcnt_t` and `c_ulong` are 32-bit on some Linux targets and
        // 64-bit on others; the conversion is the identity on the latter.
        #[allow(clippy::useless_conversion)]
        let (blocks, size) = (u64::from(st.f_bavail), u64::from(st.f_frsize));
        Ok(super::bytes_of(blocks, size))
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod imp {
    use std::io;
    use std::path::Path;

    pub(super) fn available(path: &Path) -> io::Result<u64> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "{}: free disk space is not probed on this platform",
                path.display()
            ),
        ))
    }
}

#[cfg(all(test, any(target_os = "macos", target_os = "linux")))]
mod tests {
    use super::*;

    #[test]
    fn a_real_directory_reports_some_space_and_a_missing_one_errors() {
        let dir = std::env::temp_dir();
        let free = available_disk_bytes(&dir).unwrap();
        assert!(free > 0, "{}: 0 bytes free", dir.display());

        let missing = dir.join(format!("ojas-no-such-dir-{}", std::process::id()));
        let err = available_disk_bytes(&missing).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound, "{err}");

        let nul = Path::new("a\0b");
        let err = available_disk_bytes(nul).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "{err}");
    }

    #[test]
    fn the_product_saturates() {
        assert_eq!(bytes_of(3, 4096), 12_288);
        assert_eq!(bytes_of(u64::MAX, 2), u64::MAX);
        assert_eq!(bytes_of(0, u64::MAX), 0);
    }
}
