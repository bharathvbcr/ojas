//! `IoError` keeps the `std::io::Error` a file-system failure came from:
//! `io_kind()` names its kind and `source()` returns it. A failure the crate
//! raises itself has neither. The Display text is unchanged either way.

use std::error::Error;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use ojas_io::{open_nofollow, open_regular_file, read_regular_text, IoError};

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("ojas-io-kind-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        // Restore access so the removal below can reach the mode-000 file.
        if let Ok(rd) = std::fs::read_dir(&self.0) {
            for e in rd.flatten() {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ = std::fs::set_permissions(e.path(), std::fs::Permissions::from_mode(0o600));
                }
            }
        }
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn assert_kind(err: &IoError, kind: ErrorKind) {
    assert_eq!(err.io_kind(), Some(kind), "io_kind of {err}");
    let src = err.source().expect("source() is the io::Error");
    let io = src.downcast_ref::<std::io::Error>().expect("source downcasts to io::Error");
    assert_eq!(io.kind(), kind);
}

fn both_openers(path: &Path) -> Vec<(&'static str, IoError)> {
    vec![
        ("open_regular_file", open_regular_file(path).unwrap_err()),
        ("read_regular_text", read_regular_text(path, 1024, "test cap").unwrap_err()),
    ]
}

#[test]
fn not_found_keeps_its_kind() {
    let d = Dir::new("nf");
    let missing = d.0.join("missing");
    for (who, e) in both_openers(&missing) {
        eprintln!("{who}: {e}");
        assert_kind(&e, ErrorKind::NotFound);
        assert!(!e.is_refusal());
    }
    assert_kind(&open_nofollow(&missing).unwrap_err(), ErrorKind::NotFound);
}

#[cfg(unix)]
#[test]
fn permission_denied_keeps_its_kind() {
    use std::os::unix::fs::PermissionsExt;
    let d = Dir::new("pd");
    let f = d.0.join("locked");
    std::fs::write(&f, b"x").unwrap();
    std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o000)).unwrap();
    if std::fs::File::open(&f).is_ok() {
        eprintln!("skipping: a mode-000 file is readable here (running as root?)");
        return;
    }
    for (who, e) in both_openers(&f) {
        eprintln!("{who}: {e}");
        assert_kind(&e, ErrorKind::PermissionDenied);
    }
}

#[cfg(unix)]
#[test]
fn not_a_directory_keeps_its_kind() {
    let d = Dir::new("nd");
    let f = d.0.join("file");
    std::fs::write(&f, b"x").unwrap();
    let through = f.join("child");
    // Ask std which kind it reports, so the test does not assume one.
    let want = std::fs::metadata(&through).unwrap_err().kind();
    assert_eq!(want, ErrorKind::NotADirectory, "std's kind for file/child on this host");
    for (who, e) in both_openers(&through) {
        eprintln!("{who}: {e}");
        assert_kind(&e, want);
    }
}

#[test]
fn synthetic_refusals_have_no_kind_or_source() {
    let d = Dir::new("syn");
    let f = d.0.join("big");
    std::fs::write(&f, vec![b'a'; 64]).unwrap();
    let over = read_regular_text(&f, 8, "test cap").unwrap_err();
    assert!(over.is_refusal());
    assert_eq!(over.io_kind(), None);
    assert!(over.source().is_none());

    let dir = open_regular_file(&d.0).unwrap_err();
    assert!(dir.to_string().ends_with("not a regular file"), "{dir}");
    assert_eq!(dir.io_kind(), None);
    assert!(dir.source().is_none());

    assert_eq!(IoError::new("plain").io_kind(), None);
}

#[test]
fn display_text_is_unchanged() {
    let d = Dir::new("disp");
    let missing = d.0.join("missing");
    let raw = std::fs::metadata(&missing).unwrap_err();
    let want = format!("{}: {raw}", missing.display());
    assert_eq!(open_regular_file(&missing).unwrap_err().to_string(), want);
    assert_eq!(read_regular_text(&missing, 8, "cap").unwrap_err().to_string(), want);
    assert_eq!(open_regular_file(&missing).unwrap_err().detail(), want);
}

#[test]
fn constructors_keep_source_and_wrap_keeps_it_through_a_prefix() {
    let raw = std::io::Error::new(ErrorKind::TimedOut, "slow");
    let e = IoError::io(format!("ctx: {raw}"), raw);
    assert_kind(&e, ErrorKind::TimedOut);
    assert_eq!(e.to_string(), "ctx: slow");
    let w = IoError::wrap(format!("outer: {}", e.detail()), e);
    assert_kind(&w, ErrorKind::TimedOut);
    assert_eq!(w.to_string(), "outer: ctx: slow");
    assert!(!w.is_refusal());
}
