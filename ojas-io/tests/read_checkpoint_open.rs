//! ft-2ac45473 c3: `read_checkpoint` opens through `open_regular_file`, so a FIFO is refused
//! instead of blocking the open until a writer appears.

#![cfg(unix)]

use ojas_io::read_checkpoint;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ojas-io-readckpt-{tag}-{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A FIFO named as the checkpoint is refused as "not a regular file", with no OS source, and
/// within the bound. Pre-fix (b7ecc97) `File::open` of a FIFO blocks for a writer that never
/// comes, so the 10 s bound fails the test.
#[test]
fn a_fifo_is_refused_not_opened() {
    let dir = scratch("fifo");
    let fifo = dir.join("state.ojck");
    let made = std::process::Command::new("/usr/bin/mkfifo")
        .arg(&fifo)
        .status()
        .expect("mkfifo could not run");
    assert!(made.success(), "mkfifo failed");

    let (tx, rx) = mpsc::channel();
    let f = fifo.clone();
    std::thread::spawn(move || {
        let _ = tx.send(read_checkpoint(&f));
    });
    let err = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("read_checkpoint blocked on a FIFO")
        .unwrap_err();
    assert_eq!(err.io_kind(), None, "{err:?}");
    assert_eq!(
        err.detail(),
        format!("{}: not a regular file", fifo.display())
    );
    std::fs::remove_dir_all(dir).ok();
}

/// A missing path keeps its typed kind (the OS error stays the source), as before.
#[test]
fn a_missing_path_still_reports_not_found() {
    let dir = scratch("missing");
    let err = read_checkpoint(&dir.join("nope.ojck")).unwrap_err();
    assert_eq!(err.io_kind(), Some(std::io::ErrorKind::NotFound), "{err:?}");
    std::fs::remove_dir_all(dir).ok();
}
