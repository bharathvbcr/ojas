//! Opening a file without following a final-component symlink.
//!
//! `O_NOFOLLOW` has no std constant and its value differs by target, so the
//! table below is the workspace's one copy of it. Values are from the libc
//! crate's target modules (0.2.189: `unix/bsd/mod.rs`, and under
//! `unix/linux_like/` the `linux/gnu` and `android` arch modules); a target
//! not listed is refused rather than opened without the flag.

use crate::error::IoError;
use std::fs::File;
use std::path::Path;

/// `O_NOFOLLOW` for this target, or `None` where the table has no entry.
#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly"
))]
const O_NOFOLLOW: Option<i32> = Some(0x0100);
#[cfg(all(
    any(target_os = "linux", target_os = "android"),
    any(target_arch = "aarch64", target_arch = "arm")
))]
const O_NOFOLLOW: Option<i32> = Some(0x8000);
#[cfg(all(
    target_os = "linux",
    any(target_arch = "powerpc", target_arch = "powerpc64")
))]
const O_NOFOLLOW: Option<i32> = Some(0x8000);
#[cfg(all(
    any(target_os = "linux", target_os = "android"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
const O_NOFOLLOW: Option<i32> = Some(0x2_0000);
#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "riscv64",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "s390x",
        target_arch = "loongarch64",
        target_arch = "sparc64"
    )
))]
const O_NOFOLLOW: Option<i32> = Some(0x2_0000);
#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly",
    all(
        any(target_os = "linux", target_os = "android"),
        any(
            target_arch = "aarch64",
            target_arch = "arm",
            target_arch = "x86",
            target_arch = "x86_64"
        )
    ),
    all(
        target_os = "linux",
        any(
            target_arch = "powerpc",
            target_arch = "powerpc64",
            target_arch = "riscv64",
            target_arch = "mips",
            target_arch = "mips64",
            target_arch = "s390x",
            target_arch = "loongarch64",
            target_arch = "sparc64"
        )
    )
)))]
const O_NOFOLLOW: Option<i32> = None;

/// Open `path` read-only, refusing a final-component symlink.
///
/// Three checks: `lstat` refuses a symlink (dangling or not) and anything but
/// a regular file before opening, so a FIFO cannot block the open; the open
/// itself carries `O_NOFOLLOW`, so a symlink swapped in after the `lstat`
/// fails with `ELOOP` instead of being followed; and the opened file must be
/// the same regular file (device and inode) the `lstat` saw. Intermediate
/// directories may still be symlinks. On a target without a known
/// `O_NOFOLLOW` this refuses instead of opening.
pub fn open_nofollow(path: &Path) -> Result<File, IoError> {
    let what = path.display();
    let Some(flag) = O_NOFOLLOW else {
        return Err(IoError::new(format!(
            "{what}: open_nofollow has no O_NOFOLLOW value for this target"
        )));
    };
    let before =
        std::fs::symlink_metadata(path).map_err(|e| IoError::new(format!("{what}: {e}")))?;
    if before.file_type().is_symlink() {
        return Err(IoError::new(format!("{what}: is a symbolic link")));
    }
    if !before.is_file() {
        return Err(IoError::new(format!("{what}: not a regular file")));
    }
    let file = open_flagged(path, flag).map_err(|e| IoError::new(format!("{what}: {e}")))?;
    let after = file
        .metadata()
        .map_err(|e| IoError::new(format!("{what}: {e}")))?;
    if !after.is_file() || identity(&before) != identity(&after) {
        return Err(IoError::new(format!("{what}: changed while it was opened")));
    }
    Ok(file)
}

#[cfg(unix)]
fn open_flagged(path: &Path, flag: i32) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(flag)
        .open(path)
}

#[cfg(not(unix))]
fn open_flagged(path: &Path, flag: i32) -> std::io::Result<File> {
    let _ = (path, flag);
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "no O_NOFOLLOW off unix",
    ))
}

#[cfg(unix)]
fn identity(m: &std::fs::Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    (m.dev(), m.ino())
}

#[cfg(not(unix))]
fn identity(m: &std::fs::Metadata) -> (u64, u64) {
    (m.len(), 0)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::test_util::tmp_dir;

    #[test]
    fn plain_file_opens_and_symlinks_are_refused() {
        let dir = tmp_dir("nofollow");
        let real = dir.0.join("real");
        std::fs::write(&real, b"x").unwrap();
        let mut got = String::new();
        std::io::Read::read_to_string(&mut open_nofollow(&real).unwrap(), &mut got).unwrap();
        assert_eq!(got, "x");

        let link = dir.0.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let err = open_nofollow(&link).unwrap_err();
        assert!(err.detail().contains("symbolic link"), "{err}");

        let dangling = dir.0.join("dangling");
        std::os::unix::fs::symlink(dir.0.join("nowhere"), &dangling).unwrap();
        let err = open_nofollow(&dangling).unwrap_err();
        assert!(err.detail().contains("symbolic link"), "{err}");

        let err = open_nofollow(&dir.0).unwrap_err();
        assert!(err.detail().contains("not a regular file"), "{err}");
        assert!(open_nofollow(&dir.0.join("missing")).is_err());
    }

    /// The table's value really is `O_NOFOLLOW` here: an open carrying it
    /// fails on a symlink with `ELOOP`, which a wrong bit would not do. This
    /// is the path a symlink swapped in after the `lstat` takes.
    #[cfg(any(
        target_os = "macos",
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    #[test]
    fn the_flag_makes_the_kernel_refuse_a_symlink() {
        // sys/errno.h on macOS; asm-generic/errno.h on these Linux arches.
        #[cfg(target_os = "macos")]
        const ELOOP: i32 = 62;
        #[cfg(target_os = "linux")]
        const ELOOP: i32 = 40;
        let dir = tmp_dir("nofollow-flag");
        let real = dir.0.join("real");
        std::fs::write(&real, b"x").unwrap();
        let link = dir.0.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let flag = O_NOFOLLOW.expect("this target has an entry");
        let err = open_flagged(&link, flag).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(ELOOP), "{err}");
        assert!(open_flagged(&real, flag).is_ok());
    }
}
