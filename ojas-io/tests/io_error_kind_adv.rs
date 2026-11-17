//! Adversarial coverage for `IoError`'s kept `std::io::Error` (card ft-aa3f1f18), lane `ioadv`.
//!
//! The lane's own tests cover `open_regular_file`, `read_regular_text` and `open_nofollow`.
//! These cover every other public entry point that maps an io failure, including through
//! one or two `IoError::wrap` layers, and pin the Display text each one produced before the
//! change (derived from the format strings at da2e81e, not from the code under test).

use std::error::Error;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use ojas_io::sha256::open_sha256_hex;
use ojas_io::{
    read_checkpoint, read_regular_text, replace_dir_with, replace_dir_with_if, write_safetensors,
    CheckpointView, IoError, SafeTensors,
};

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("ojas-io-adv-{}-{tag}", std::process::id()));
        restore_and_remove(&p);
        std::fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        restore_and_remove(&self.0);
    }
}

#[cfg(unix)]
fn chmod(p: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode));
}

#[cfg(unix)]
fn restore_and_remove(p: &Path) {
    chmod(p, 0o700);
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            chmod(&e.path(), 0o700);
        }
    }
    let _ = std::fs::remove_dir_all(p);
}

#[cfg(not(unix))]
fn restore_and_remove(p: &Path) {
    let _ = std::fs::remove_dir_all(p);
}

fn kind_of_source(e: &IoError) -> Option<ErrorKind> {
    e.source()
        .and_then(|s| s.downcast_ref::<std::io::Error>())
        .map(std::io::Error::kind)
}

#[track_caller]
fn assert_kind(who: &str, e: &IoError, kind: ErrorKind) {
    assert_eq!(e.io_kind(), Some(kind), "{who}: io_kind of `{e}`");
    assert_eq!(
        kind_of_source(e),
        Some(kind),
        "{who}: source() kind of `{e}`"
    );
    assert!(!e.is_refusal(), "{who}: `{e}` must not be a refusal");
}

fn raw_kind(p: &Path) -> ErrorKind {
    std::fs::metadata(p).unwrap_err().kind()
}

/// A tiny valid safetensors file, so the success path of a reader is not what fails.
fn empty_safetensors(path: &Path) {
    write_safetensors(path, &[], &[]).unwrap();
}

fn root() -> bool {
    // A mode-000 file is readable by root; detect by trying, once, so concurrent tests
    // never share the probe directory.
    static ROOT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ROOT.get_or_init(|| {
        let d = Dir::new("rootprobe");
        let f = d.0.join("f");
        std::fs::write(&f, b"x").unwrap();
        #[cfg(unix)]
        chmod(&f, 0o000);
        std::fs::File::open(&f).is_ok()
    })
}

// ---------------------------------------------------------------------------------------
// attack 4: kind at the boundary, through wrap layers
// ---------------------------------------------------------------------------------------

#[test]
fn missing_path_reports_not_found_through_every_reader() {
    let d = Dir::new("nf");
    let m = d.0.join("missing");
    let cases: Vec<(&str, IoError)> = vec![
        ("read_checkpoint", read_checkpoint(&m).unwrap_err()),
        (
            "SafeTensors::open",
            SafeTensors::open(&m).map(drop).unwrap_err(),
        ),
        (
            "CheckpointView::open",
            CheckpointView::open(&m).map(drop).unwrap_err(),
        ),
        (
            "open_sha256_hex",
            open_sha256_hex(&m).map(drop).unwrap_err(),
        ),
    ];
    for (who, e) in &cases {
        assert_kind(who, e, ErrorKind::NotFound);
    }
}

#[cfg(unix)]
#[test]
fn path_through_a_file_reports_not_a_directory_through_every_reader() {
    let d = Dir::new("nd");
    let f = d.0.join("file");
    std::fs::write(&f, b"x").unwrap();
    let through = f.join("child");
    let want = raw_kind(&through);
    assert_eq!(want, ErrorKind::NotADirectory);
    let cases: Vec<(&str, IoError)> = vec![
        ("read_checkpoint", read_checkpoint(&through).unwrap_err()),
        (
            "SafeTensors::open",
            SafeTensors::open(&through).map(drop).unwrap_err(),
        ),
        (
            "CheckpointView::open",
            CheckpointView::open(&through).map(drop).unwrap_err(),
        ),
        (
            "open_sha256_hex",
            open_sha256_hex(&through).map(drop).unwrap_err(),
        ),
    ];
    for (who, e) in &cases {
        assert_kind(who, e, want);
    }
}

#[cfg(unix)]
#[test]
fn unreadable_file_reports_permission_denied_through_every_reader() {
    if root() {
        eprintln!("skipping: mode 000 is readable here (root?)");
        return;
    }
    let d = Dir::new("pd");
    let f = d.0.join("locked");
    std::fs::write(&f, b"x").unwrap();
    chmod(&f, 0o000);
    let cases: Vec<(&str, IoError)> = vec![
        ("read_checkpoint", read_checkpoint(&f).unwrap_err()),
        (
            "SafeTensors::open",
            SafeTensors::open(&f).map(drop).unwrap_err(),
        ),
        (
            "CheckpointView::open",
            CheckpointView::open(&f).map(drop).unwrap_err(),
        ),
        (
            "open_sha256_hex",
            open_sha256_hex(&f).map(drop).unwrap_err(),
        ),
    ];
    for (who, e) in &cases {
        assert_kind(who, e, ErrorKind::PermissionDenied);
    }
}

/// A directory with no model file: the `NotFound` comes from the second of two nested
/// openers (`open_single` -> `SafeTensors::open_with_limit` -> `open_regular_file`).
#[test]
fn checkpoint_dir_without_model_reports_not_found() {
    let d = Dir::new("nomodel");
    let e = CheckpointView::open(&d.0).map(drop).unwrap_err();
    assert_kind("CheckpointView::open(dir)", &e, ErrorKind::NotFound);
}

/// An index naming a shard that is not there.
#[test]
fn index_naming_a_missing_shard_reports_not_found() {
    let d = Dir::new("shard");
    std::fs::write(
        d.0.join("model.safetensors.index.json"),
        br#"{"weight_map":{"t":"gone.safetensors"}}"#,
    )
    .unwrap();
    let e = CheckpointView::open(&d.0).map(drop).unwrap_err();
    assert_kind(
        "CheckpointView::open(index, missing shard)",
        &e,
        ErrorKind::NotFound,
    );
}

/// `symlink_metadata(dir/index)` fails with PermissionDenied when the directory cannot be
/// searched (open.rs / safetensors.rs:966 branch).
#[cfg(unix)]
#[test]
fn unsearchable_checkpoint_dir_reports_permission_denied() {
    if root() {
        eprintln!("skipping: root");
        return;
    }
    let d = Dir::new("nosearch");
    let sub = d.0.join("snap");
    std::fs::create_dir(&sub).unwrap();
    chmod(&sub, 0o000);
    let r = CheckpointView::open(&sub).map(drop);
    chmod(&sub, 0o700);
    assert_kind(
        "CheckpointView::open(0o000 dir)",
        &r.unwrap_err(),
        ErrorKind::PermissionDenied,
    );
}

#[test]
fn writers_report_the_kind_of_the_parent_failure() {
    let d = Dir::new("w");
    let missing_parent = d.0.join("nope").join("out.safetensors");
    let e = write_safetensors(&missing_parent, &[], &[]).unwrap_err();
    assert_kind("write_safetensors(missing parent)", &e, ErrorKind::NotFound);

    let f = d.0.join("file");
    std::fs::write(&f, b"x").unwrap();
    let under_file = f.join("out.safetensors");
    let e = write_safetensors(&under_file, &[], &[]).unwrap_err();
    #[cfg(unix)]
    assert_kind(
        "write_safetensors(parent is a file)",
        &e,
        ErrorKind::NotADirectory,
    );
    #[cfg(not(unix))]
    let _ = e;
}

/// The rename onto an existing non-empty directory fails inside `replace_with`'s staged
/// closure; that error is re-wrapped in the cleanup arm (replace.rs:47-54).
#[cfg(unix)]
#[test]
fn rename_failure_in_replace_keeps_the_rename_error() {
    let d = Dir::new("renamefail");
    let target = d.0.join("out.safetensors");
    std::fs::create_dir(&target).unwrap();
    std::fs::write(target.join("keep"), b"x").unwrap();
    let e = write_safetensors(&target, &[], &[]).unwrap_err();
    let want = std::fs::rename(d.0.join("nonexistent-src"), &target).is_err();
    assert!(want);
    assert!(
        e.io_kind().is_some(),
        "rename failure lost its io::Error: {e}"
    );
    let src = e.source().unwrap().to_string();
    assert!(
        e.detail().contains(&src),
        "message `{e}` should contain its source `{src}`"
    );
    assert!(
        e.detail().starts_with(&format!("{}: ", target.display())),
        "{e}"
    );
}

/// `fill` returns an io-backed IoError; `replace_dir_with` re-wraps it (replace.rs:306-316).
#[test]
fn fill_error_kind_survives_replace_dir_with() {
    let d = Dir::new("fill");
    let target = d.0.join("out");
    let e = replace_dir_with(&target, |_| {
        let raw = std::io::Error::new(ErrorKind::TimedOut, "slow disk");
        Err(IoError::io(format!("fill: {raw}"), raw))
    })
    .unwrap_err();
    assert_kind("replace_dir_with(fill)", &e, ErrorKind::TimedOut);
    assert_eq!(
        e.to_string(),
        format!("{}: fill: slow disk", target.display())
    );
}

/// Same through `allow` (replace.rs:287), with an existing target.
#[test]
fn allow_error_kind_survives_replace_dir_with_if() {
    let d = Dir::new("allow");
    let target = d.0.join("out");
    std::fs::create_dir(&target).unwrap();
    let e = replace_dir_with_if(
        &target,
        |_| {
            let raw = std::io::Error::new(ErrorKind::TimedOut, "slow allow");
            Err(IoError::io(format!("allow: {raw}"), raw))
        },
        |_| Ok(()),
    )
    .unwrap_err();
    assert_kind("replace_dir_with_if(allow)", &e, ErrorKind::TimedOut);
}

/// The two-error cleanup arm of `replace_dir_with_if` (replace.rs:308-316): `fill` fails with
/// a TimedOut error, then removing the stage also fails (the parent was made read-only).
/// The message leads with `fill`'s error, so that must be the source, not the cleanup error.
#[cfg(unix)]
#[test]
fn double_failure_keeps_the_error_the_message_leads_with() {
    if root() {
        eprintln!("skipping: root");
        return;
    }
    let d = Dir::new("double");
    let parent = d.0.join("p");
    std::fs::create_dir(&parent).unwrap();
    let target = parent.join("out");
    let p2 = parent.clone();
    let r = replace_dir_with(&target, |_| {
        chmod(&p2, 0o500);
        let raw = std::io::Error::new(ErrorKind::TimedOut, "fill failed first");
        Err(IoError::io(format!("fill: {raw}"), raw))
    });
    chmod(&parent, 0o700);
    let e = r.unwrap_err();
    assert!(
        e.detail().contains("also failed"),
        "cleanup did not fail, the arm was not exercised: {e}"
    );
    assert!(e.detail().contains("fill failed first"), "{e}");
    assert_kind("double failure", &e, ErrorKind::TimedOut);
}

/// A directory is refused by the crate itself ("not a regular file"), so it has no kind.
#[cfg(unix)]
#[test]
fn directory_is_a_synthetic_refusal_with_no_kind() {
    let d = Dir::new("dir");
    let cases: Vec<(&str, IoError)> = vec![
        ("read_checkpoint", read_checkpoint(&d.0).unwrap_err()),
        (
            "SafeTensors::open",
            SafeTensors::open(&d.0).map(drop).unwrap_err(),
        ),
        (
            "open_sha256_hex",
            open_sha256_hex(&d.0).map(drop).unwrap_err(),
        ),
    ];
    for (who, e) in &cases {
        assert_eq!(e.io_kind(), None, "{who}: {e}");
        assert!(e.source().is_none(), "{who}: {e}");
        assert!(e.detail().ends_with("not a regular file"), "{who}: {e}");
    }
}

// ---------------------------------------------------------------------------------------
// attack 1: Display text, expected strings derived from the pre-change format strings
// ---------------------------------------------------------------------------------------

#[test]
fn display_text_matches_the_pre_change_format_strings() {
    let d = Dir::new("disp");
    let m = d.0.join("missing");
    let raw = std::fs::metadata(&m).unwrap_err();
    let path = m.display();
    // checkpoint.rs: `{what}: {e}`
    assert_eq!(
        read_checkpoint(&m).unwrap_err().to_string(),
        format!("{path}: {raw}")
    );
    // safetensors.rs open_with_limit: the open_regular_file text, unchanged by both wraps
    assert_eq!(
        SafeTensors::open(&m).map(drop).unwrap_err().to_string(),
        format!("{path}: {raw}")
    );
    // safetensors.rs CheckpointView::open_with_limit: `{what}: {e}`
    assert_eq!(
        CheckpointView::open(&m).map(drop).unwrap_err().to_string(),
        format!("{path}: {raw}")
    );
    // sha256.rs open_sha256_hex: `{describe}: {e}` (describe is the plain path here)
    assert_eq!(
        open_sha256_hex(&m).map(drop).unwrap_err().to_string(),
        format!("{path}: {raw}")
    );
    // read_regular_text goes through open_regular_file
    assert_eq!(
        read_regular_text(&m, 8, "cap").unwrap_err().to_string(),
        format!("{path}: {raw}")
    );
    // replace.rs ParentLock::acquire (replace_with takes the parent lock before its sweep, so a
    // missing parent is reported there, ft-2ac45473 c5): `{what}: parent {parent}: {e}`
    let out = d.0.join("nope").join("o.safetensors");
    let dir = d.0.join("nope");
    let raw_dir = std::fs::canonicalize(&dir).map(drop).unwrap_err();
    assert_eq!(
        write_safetensors(&out, &[], &[]).unwrap_err().to_string(),
        format!("{}: parent {}: {raw_dir}", out.display(), dir.display())
    );
    // replace.rs Site::lock: `{what}: parent {parent}: {e}`
    let tgt = d.0.join("nope").join("t");
    let raw_parent = std::fs::symlink_metadata(&dir).unwrap_err();
    assert_eq!(
        replace_dir_with(&tgt, |_| Ok(())).unwrap_err().to_string(),
        format!("{}: parent {}: {raw_parent}", tgt.display(), dir.display())
    );
    // wrap sites that prefix: index text names the index first, then the inner detail
    std::fs::write(d.0.join("model.safetensors.index.json"), b"{}").unwrap();
    let e = CheckpointView::open(&d.0).map(drop).unwrap_err();
    let idx = d.0.join("model.safetensors.index.json");
    assert!(
        e.to_string().starts_with(&format!("{}: ", idx.display())),
        "{e}"
    );
    assert_eq!(
        e.io_kind(),
        None,
        "a parse refusal of the index has no io source: {e}"
    );
}

// ---------------------------------------------------------------------------------------
// attack 3: the refusal flag across public paths
// ---------------------------------------------------------------------------------------

/// `CheckpointView::open_index` returns `read_regular_text`'s error with `?`, so a UTF-8
/// refusal keeps its flag through the public API.
#[test]
fn index_utf8_refusal_keeps_its_flag_and_has_no_source() {
    let d = Dir::new("utf8");
    std::fs::write(d.0.join("model.safetensors.index.json"), [0xff, 0xfe, 0x00]).unwrap();
    let e = CheckpointView::open(&d.0).map(drop).unwrap_err();
    assert!(
        e.is_refusal(),
        "refusal flag lost through CheckpointView::open: {e}"
    );
    assert_eq!(e.io_kind(), None);
    assert!(e.source().is_none());
}

/// `IoError::wrap` documents "not a refusal, whatever `inner` was". A public caller can
/// build that situation from `read_regular_text`. This pins the documented behaviour; it is
/// the pre-existing format-only wrap's behaviour too (see the lane report).
#[test]
fn wrap_drops_the_refusal_flag_as_documented() {
    let d = Dir::new("wraprefusal");
    let f = d.0.join("big");
    std::fs::write(&f, vec![b'a'; 64]).unwrap();
    let inner = read_regular_text(&f, 8, "cap").unwrap_err();
    assert!(inner.is_refusal());
    let w = IoError::wrap(format!("outer: {}", inner.detail()), inner);
    assert!(!w.is_refusal());
    assert_eq!(w.io_kind(), None);
}

/// Empty `Dir` helper sanity so an unused-function lint cannot hide a skipped file.
#[test]
fn fixtures_work() {
    let d = Dir::new("fixture");
    empty_safetensors(&d.0.join("a.safetensors"));
    assert!(SafeTensors::open(&d.0.join("a.safetensors")).is_ok());
}
