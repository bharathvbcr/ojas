//! Opening a file without following a final-component symlink, and opening a
//! regular file without blocking on one that is not.
//!
//! `O_NOFOLLOW` and `O_NONBLOCK` have no std constants and their values differ
//! by target, so the tables below are the workspace's one copy of them. Values
//! are from the libc crate's target modules (0.2.189: `unix/bsd/mod.rs`, and
//! under `unix/linux_like/` the `linux/gnu` and `android` arch modules).
//! For `O_NOFOLLOW` a target not listed is refused rather than opened without
//! the flag; for `O_NONBLOCK` see [`open_regular_file`], which keeps its other
//! two checks either way.

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

/// `O_NONBLOCK` for this target, or `None` where the table has no entry.
/// 0x0004 on the BSDs and macOS, 0o4000 on the Linux arches below; MIPS uses
/// another value and is deliberately absent rather than guessed.
#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly"
))]
const O_NONBLOCK: Option<i32> = Some(0x0004);
#[cfg(all(
    any(target_os = "linux", target_os = "android"),
    any(
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "powerpc",
        target_arch = "powerpc64",
        target_arch = "riscv64",
        target_arch = "s390x",
        target_arch = "loongarch64",
        target_arch = "sparc64"
    )
))]
const O_NONBLOCK: Option<i32> = Some(0o4000);
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
            target_arch = "x86_64",
            target_arch = "powerpc",
            target_arch = "powerpc64",
            target_arch = "riscv64",
            target_arch = "s390x",
            target_arch = "loongarch64",
            target_arch = "sparc64"
        )
    )
)))]
const O_NONBLOCK: Option<i32> = None;

/// Open `path` read-only and refuse anything that is not a regular file,
/// without blocking on one that is not. Symlinks are followed.
///
/// Returns the handle and its length from the one `fstat` that judged it, so a
/// caller validating a header against the length and re-checking the handle's
/// identity later cannot have the two disagree.
///
/// Opening a FIFO for reading blocks until a writer appears, so a checkpoint
/// naming a FIFO as a shard would hang the loader rather than be refused.
/// Three things prevent that, and the first two hold on every target:
///
/// * `stat` (which follows symlinks, because a Hugging Face cache snapshot
///   links its shards into `../../blobs`) refuses anything but a regular file
///   before the open;
/// * the opened handle is judged again by `fstat`, so the refusal does not
///   rest on the path alone;
/// * the open carries `O_NONBLOCK` where this target's value is known, which
///   also closes the window between the `stat` and the open. On a target
///   without an entry the flag is omitted and that window stays open; the
///   other two checks are unaffected.
///
/// `O_NONBLOCK` is ignored for regular files, so it changes nothing for the
/// files this is meant to open.
///
/// **Error contract.** Every failure carries the OS error as its source
/// ([`IoError::io_kind`] is `Some`), except the not-a-regular-file refusal
/// (a directory, FIFO, socket or device), which is raised by this crate and
/// has none (`io_kind()` is `None`). Callers classify on that difference
/// (tessl's `bounded_file::open_regular` does), so a new sourceless refusal of
/// a regular file, or a sourced error for a non-regular one, is a contract
/// change.
pub fn open_regular_file(path: &Path) -> Result<(File, u64), IoError> {
    let what = path.display();
    let before = std::fs::metadata(path).map_err(|e| IoError::io(format!("{what}: {e}"), e))?;
    if !before.is_file() {
        return Err(IoError::new(format!("{what}: not a regular file")));
    }
    let file = match O_NONBLOCK {
        Some(flag) => open_flagged(path, flag),
        None => File::open(path),
    }
    .map_err(|e| IoError::io(format!("{what}: {e}"), e))?;
    let after = file
        .metadata()
        .map_err(|e| IoError::io(format!("{what}: {e}"), e))?;
    if !after.is_file() {
        return Err(IoError::new(format!("{what}: not a regular file")));
    }
    Ok((file, after.len()))
}

/// Read the regular file at `path` as UTF-8 text, refusing one longer than
/// `max_bytes` before it is allocated. `cap_name` names the cap in the refusal
/// (`"index cap"`, `"JSON input limit"`).
///
/// The open is [`open_regular_file`]'s, so a directory or FIFO is refused
/// without blocking, and the length it returns is judged against `max_bytes`
/// before any buffer exists. The read is then bounded to `max_bytes + 1`: a
/// file that grew after the `fstat` is refused with the same error rather than
/// truncated, because a truncated JSON document that happens to parse is worse
/// than a refusal. Invalid UTF-8 is refused. Every refusal names `path`; the
/// cap refusal also names the observed length (`"at least"` when it came from
/// the read, not the `fstat`) and `max_bytes`.
///
/// Symlinks are followed on purpose: a Hugging Face cache snapshot links its
/// config, index and shards into `../../blobs`, and refusing that would refuse
/// every real snapshot. This replaces `std::fs::read_to_string`, which sizes
/// its allocation from whatever the file turns out to be and blocks on a FIFO.
pub fn read_regular_text(path: &Path, max_bytes: u64, cap_name: &str) -> Result<String, IoError> {
    use std::io::Read;
    let what = path.display();
    let (file, len) = open_regular_file(path)?;
    if len > max_bytes {
        return Err(IoError::refusal(format!(
            "{what}: {len} bytes, above the {max_bytes}-byte {cap_name}"
        )));
    }
    // `len` is at most `max_bytes`, so this is a bounded allocation.
    let mut text = String::with_capacity(usize::try_from(len).unwrap_or(0));
    file.take(max_bytes.saturating_add(1))
        .read_to_string(&mut text)
        .map_err(|e| {
            let detail = format!("{what}: {e}");
            if e.kind() == std::io::ErrorKind::InvalidData {
                IoError::refusal(detail)
            } else {
                IoError::io(detail, e)
            }
        })?;
    if text.len() as u64 > max_bytes {
        return Err(IoError::refusal(format!(
            "{what}: at least {} bytes, above the {max_bytes}-byte {cap_name}",
            text.len()
        )));
    }
    Ok(text)
}

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
        std::fs::symlink_metadata(path).map_err(|e| IoError::io(format!("{what}: {e}"), e))?;
    if before.file_type().is_symlink() {
        return Err(IoError::new(format!("{what}: is a symbolic link")));
    }
    if !before.is_file() {
        return Err(IoError::new(format!("{what}: not a regular file")));
    }
    let file = open_flagged(path, flag).map_err(|e| IoError::io(format!("{what}: {e}"), e))?;
    let after = file
        .metadata()
        .map_err(|e| IoError::io(format!("{what}: {e}"), e))?;
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

    fn assert_sourceless_refusal(p: &Path) {
        let err = open_regular_file(p).unwrap_err();
        let want = format!("{}: not a regular file", p.display());
        assert!(std::error::Error::source(&err).is_none(), "{p:?}: {err:?}");
        assert_eq!(err.io_kind(), None, "{p:?}: {err:?}");
        assert_eq!(err.detail(), want, "{p:?}");
    }

    fn assert_sourced(p: &Path, kind: std::io::ErrorKind) {
        let err = open_regular_file(p).unwrap_err();
        assert!(std::error::Error::source(&err).is_some(), "{p:?}: {err:?}");
        assert_eq!(err.io_kind(), Some(kind), "{p:?}: {err:?}");
    }

    /// `open_regular_file`'s sourceless-error contract (ft-629e52ef): a
    /// directory, FIFO, socket, device and a symlink to a FIFO are refused with
    /// no source and no `io_kind`; a missing path and a dangling symlink fail
    /// with `NotFound` and a source; regular files (empty, one byte, behind a
    /// symlink) open with the right length. Every refused input here is caught
    /// by the `stat` refusal; the `fstat` refusal after the open is reachable
    /// only by swapping the path between the two calls and is not pinned here.
    #[test]
    fn open_regular_file_error_contract() {
        use std::io::ErrorKind;
        let dir = tmp_dir("orf");

        assert_sourceless_refusal(&dir.0);

        let fifo = dir.0.join("f");
        let made = std::process::Command::new("/usr/bin/mkfifo")
            .arg(&fifo)
            .status()
            .unwrap();
        assert!(made.success());
        let (tx, rx) = std::sync::mpsc::channel();
        let f = fifo.clone();
        std::thread::spawn(move || {
            let _ = tx.send(open_regular_file(&f));
        });
        let err = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("a FIFO blocked open_regular_file")
            .unwrap_err();
        assert!(std::error::Error::source(&err).is_none(), "{err:?}");
        assert_eq!(err.io_kind(), None, "{err:?}");
        assert_eq!(
            err.detail(),
            format!("{}: not a regular file", fifo.display())
        );

        let sock = dir.0.join("s");
        assert!(
            sock.as_os_str().len() < 100,
            "socket path too long for sun_path: {sock:?}"
        );
        let _listener = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        assert_sourceless_refusal(&sock);

        assert_sourceless_refusal(Path::new("/dev/null"));

        let to_fifo = dir.0.join("lf");
        std::os::unix::fs::symlink(&fifo, &to_fifo).unwrap();
        assert_sourceless_refusal(&to_fifo);

        assert_sourced(&dir.0.join("missing"), ErrorKind::NotFound);
        let dangling = dir.0.join("d");
        std::os::unix::fs::symlink(dir.0.join("nowhere"), &dangling).unwrap();
        assert_sourced(&dangling, ErrorKind::NotFound);

        let empty = dir.0.join("e");
        std::fs::write(&empty, b"").unwrap();
        assert_eq!(open_regular_file(&empty).unwrap().1, 0);
        let one = dir.0.join("o");
        std::fs::write(&one, b"x").unwrap();
        assert_eq!(open_regular_file(&one).unwrap().1, 1);
        let to_one = dir.0.join("lo");
        std::os::unix::fs::symlink(&one, &to_one).unwrap();
        assert_eq!(open_regular_file(&to_one).unwrap().1, 1);
    }

    fn err_is_refusal(p: &Path) -> bool {
        read_regular_text(p, 64, "cap").unwrap_err().is_refusal()
    }

    /// `read_regular_text`: the cap is on bytes, at the cap passes and one past
    /// it is refused naming the path and both numbers, a directory and a FIFO
    /// are refused without blocking, and invalid UTF-8 is refused.
    #[test]
    fn read_regular_text_is_bounded_and_refuses_non_files() {
        let dir = tmp_dir("read-text");
        let at = dir.0.join("at");
        std::fs::write(&at, "a".repeat(64)).unwrap();
        assert_eq!(read_regular_text(&at, 64, "cap").unwrap().len(), 64);

        let over = dir.0.join("over");
        std::fs::write(&over, "a".repeat(65)).unwrap();
        let err = read_regular_text(&over, 64, "test cap").unwrap_err();
        let d = err.detail();
        assert!(
            d.contains("over") && d.contains("65 bytes") && d.contains("64-byte test cap"),
            "{d}"
        );

        let err = read_regular_text(&dir.0, 64, "cap").unwrap_err();
        assert!(err.detail().contains("not a regular file"), "{err}");

        let bad = dir.0.join("bad");
        std::fs::write(&bad, [0xff, 0xfe]).unwrap();
        assert!(read_regular_text(&bad, 64, "cap").unwrap_err().is_refusal());
        assert!(err_is_refusal(&over));
        let missing = read_regular_text(&dir.0.join("missing"), 64, "cap").unwrap_err();
        assert!(!missing.is_refusal(), "{missing}");
        assert!(!read_regular_text(&dir.0, 64, "cap")
            .unwrap_err()
            .is_refusal());

        let fifo = dir.0.join("fifo");
        let made = std::process::Command::new("/usr/bin/mkfifo")
            .arg(&fifo)
            .status()
            .unwrap();
        assert!(made.success());
        let (tx, rx) = std::sync::mpsc::channel();
        let f = fifo.clone();
        std::thread::spawn(move || {
            let _ = tx.send(read_regular_text(&f, 64, "cap"));
        });
        let got = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("a FIFO blocked the reader");
        assert!(got.unwrap_err().detail().contains("not a regular file"));
    }
}
