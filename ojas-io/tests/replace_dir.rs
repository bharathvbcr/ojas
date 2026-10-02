//! `replace_dir_with` and `recover_replaced_dir`, unix only.
#![cfg(unix)]

use ojas_io::{recover_replaced_dir, replace_dir_with, IoError};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static SEQ: AtomicU64 = AtomicU64::new(0);

/// A fresh empty directory under the system temp dir, removed on drop.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = make_writable(&self.0);
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn make_writable(p: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(p, fs::Permissions::from_mode(0o755))
}

fn scratch(tag: &str) -> Scratch {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("ojas-io-dir-{}-{tag}-{n}", std::process::id()));
    fs::create_dir(&p).unwrap();
    Scratch(p)
}

/// Every file under `dir`, by path relative to it.
fn snapshot(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for e in fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(root, &p, out);
            } else {
                let rel = p.strip_prefix(root).unwrap().to_string_lossy().into_owned();
                out.insert(rel, fs::read(&p).unwrap());
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(dir, dir, &mut out);
    out
}

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    v.sort();
    v
}

/// Fill a stage with `model`, `state` and `sub/inner`, all tagged with `tag`.
fn fill_with(tag: &str) -> impl FnOnce(&Path) -> Result<(), IoError> + '_ {
    move |stage| {
        assert!(names(stage).is_empty(), "stage was not empty");
        fs::write(stage.join("model"), format!("model {tag}")).unwrap();
        fs::write(stage.join("state"), format!("state {tag}")).unwrap();
        fs::create_dir(stage.join("sub")).unwrap();
        fs::write(stage.join("sub/inner"), format!("inner {tag}")).unwrap();
        Ok(())
    }
}

fn expected(tag: &str) -> BTreeMap<String, Vec<u8>> {
    [
        ("model", format!("model {tag}")),
        ("state", format!("state {tag}")),
        ("sub/inner", format!("inner {tag}")),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.into_bytes()))
    .collect()
}

fn expect_err(r: Result<impl std::fmt::Debug, IoError>, needle: &str) {
    match r {
        Ok(v) => panic!("expected an error containing {needle:?}, got Ok({v:?})"),
        Err(e) => assert!(e.detail().contains(needle), "expected {needle:?} in {e}"),
    }
}

#[test]
fn creates_then_replaces_and_leaves_nothing_beside_the_target() {
    let s = scratch("basic");
    let target = s.0.join("ckpt");
    replace_dir_with(&target, fill_with("one")).unwrap();
    assert_eq!(snapshot(&target), expected("one"));
    // A reader holding a file of the old directory keeps its bytes.
    let held = fs::File::open(target.join("model")).unwrap();
    replace_dir_with(&target, fill_with("two")).unwrap();
    assert_eq!(snapshot(&target), expected("two"));
    assert_eq!(names(&s.0), ["ckpt"]);
    let mut old = String::new();
    std::io::Read::read_to_string(&mut &held, &mut old).unwrap();
    assert_eq!(old, "model one");
}

#[test]
fn a_failing_or_panicking_fill_leaves_the_old_directory_and_no_stage() {
    let s = scratch("fail");
    let target = s.0.join("ckpt");
    replace_dir_with(&target, fill_with("old")).unwrap();

    let err = replace_dir_with(&target, |stage| {
        fs::write(stage.join("model"), "partial").unwrap();
        Err(IoError::new("disk full"))
    });
    expect_err(err, "disk full");
    assert_eq!(snapshot(&target), expected("old"));
    assert_eq!(names(&s.0), ["ckpt"]);

    let caught = std::panic::catch_unwind(|| {
        replace_dir_with(&target, |stage| {
            fs::write(stage.join("model"), "partial").unwrap();
            panic!("fill panicked");
        })
    });
    assert!(caught.is_err());
    assert_eq!(snapshot(&target), expected("old"));
    assert_eq!(names(&s.0), ["ckpt"]);
    // The lock and the same-thread guard were released by the unwind.
    replace_dir_with(&target, fill_with("after")).unwrap();
    assert_eq!(snapshot(&target), expected("after"));

    // A fill that fails on a fresh target leaves no target at all.
    let fresh = s.0.join("fresh");
    expect_err(
        replace_dir_with(&fresh, |_| Err(IoError::new("nope"))),
        "nope",
    );
    assert!(!fresh.exists());
    assert_eq!(names(&s.0), ["ckpt"]);
}

#[test]
fn symlinks_and_non_directories_are_refused_untouched() {
    let s = scratch("links");
    let real = s.0.join("real");
    replace_dir_with(&real, fill_with("real")).unwrap();

    let link = s.0.join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    expect_err(
        replace_dir_with(&link, fill_with("x")),
        "is a symbolic link",
    );
    expect_err(recover_replaced_dir(&link), "is a symbolic link");
    assert!(fs::symlink_metadata(&link)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(snapshot(&real), expected("real"));

    let parent_link = s.0.join("plink");
    std::os::unix::fs::symlink(&s.0, &parent_link).unwrap();
    expect_err(
        replace_dir_with(&parent_link.join("ckpt"), fill_with("x")),
        "is a symbolic link",
    );
    assert!(!s.0.join("ckpt").exists());

    let file = s.0.join("file");
    fs::write(&file, b"keep").unwrap();
    expect_err(replace_dir_with(&file, fill_with("x")), "not a directory");
    assert_eq!(fs::read(&file).unwrap(), b"keep");

    // A symlink created inside the stage is refused at sync time.
    expect_err(
        replace_dir_with(&real, |stage| {
            std::os::unix::fs::symlink("/etc/hosts", stage.join("model")).unwrap();
            Ok(())
        }),
        "is a symbolic link",
    );
    assert_eq!(snapshot(&real), expected("real"));
    assert_eq!(names(&s.0), ["file", "link", "plink", "real"]);
}

#[test]
fn a_nested_call_for_the_same_parent_is_refused_instead_of_deadlocking() {
    let s = scratch("nested");
    let a = s.0.join("a");
    let b = s.0.join("b");
    replace_dir_with(&a, fill_with("a0")).unwrap();
    let err = replace_dir_with(&a, |stage| {
        fill_with("a1")(stage)?;
        replace_dir_with(&b, fill_with("b"))
    });
    expect_err(err, "would deadlock");
    assert_eq!(snapshot(&a), expected("a0"));
    assert!(!b.exists());
    // A nested call for a different parent is fine.
    let other = scratch("nested-other");
    replace_dir_with(&a, |stage| {
        fill_with("a2")(stage)?;
        replace_dir_with(&other.0.join("b"), fill_with("b"))
    })
    .unwrap();
    assert_eq!(snapshot(&a), expected("a2"));
    assert_eq!(snapshot(&other.0.join("b")), expected("b"));
}

#[test]
fn two_threads_replacing_one_target_serialize() {
    let s = scratch("race");
    let target = s.0.join("ckpt");
    replace_dir_with(&target, fill_with("start")).unwrap();
    let rounds = 25;
    std::thread::scope(|scope| {
        for who in ["x", "y"] {
            let target = &target;
            scope.spawn(move || {
                for i in 0..rounds {
                    let tag = format!("{who}{i}");
                    replace_dir_with(target, |stage| {
                        fill_with(&tag)(stage)?;
                        // Hold the stage open long enough for the other
                        // thread to try its sweep.
                        std::thread::yield_now();
                        Ok(())
                    })
                    .unwrap();
                }
            });
        }
    });
    let got = snapshot(&target);
    let tag = String::from_utf8(got["model"].clone()).unwrap();
    let tag = tag.strip_prefix("model ").unwrap().to_string();
    assert!(
        tag == format!("x{}", rounds - 1) || tag == format!("y{}", rounds - 1),
        "{tag}"
    );
    assert_eq!(
        got,
        expected(&tag),
        "files from different writers were mixed"
    );
    assert_eq!(names(&s.0), ["ckpt"]);
}

#[test]
fn a_read_only_parent_fails_cleanly() {
    use std::os::unix::fs::PermissionsExt;
    let s = scratch("ro");
    let target = s.0.join("ckpt");
    replace_dir_with(&target, fill_with("old")).unwrap();
    fs::set_permissions(&s.0, fs::Permissions::from_mode(0o555)).unwrap();
    if fs::create_dir(s.0.join("probe")).is_ok() {
        // Running as root: permissions are not enforced, nothing to test.
        eprintln!("skipped: the read-only bit is not enforced for this user");
        return;
    }
    expect_err(
        replace_dir_with(&target, fill_with("new")),
        "cannot create stage",
    );
    expect_err(
        replace_dir_with(&s.0.join("fresh"), fill_with("new")),
        "cannot create stage",
    );
    make_writable(&s.0).unwrap();
    assert_eq!(snapshot(&target), expected("old"));
    assert_eq!(names(&s.0), ["ckpt"]);
}

/// The backup and stage names a writer with this pid would use.
fn leftover(dir: &Path, name: &str, seq: u64, suffix: &str) -> PathBuf {
    dir.join(format!(".{name}.{}.{seq}.{suffix}", std::process::id()))
}

#[test]
fn a_crash_between_the_renames_is_recovered_to_the_old_directory() {
    let s = scratch("crash");
    let target = s.0.join("ckpt");
    replace_dir_with(&target, fill_with("old")).unwrap();
    // State after dying between the renames: the target moved to a backup,
    // and a finished stage that never got renamed.
    let backup = leftover(&s.0, "ckpt", 900_001, "old");
    fs::rename(&target, &backup).unwrap();
    let stage = leftover(&s.0, "ckpt", 900_002, "stage");
    fs::create_dir(&stage).unwrap();
    fill_with("lost")(&stage).unwrap();
    // Lookalikes that are not this target's leftovers stay.
    let keep = [
        s.0.join(".ckpt.old"),
        s.0.join(".ckpt.1.2.3.old"),
        s.0.join(".other.1.2.old"),
        s.0.join(".ckpt.1.x.stage"),
    ];
    for k in &keep {
        fs::create_dir(k).unwrap();
    }

    assert!(recover_replaced_dir(&target).unwrap());
    assert_eq!(snapshot(&target), expected("old"));
    assert!(!backup.exists() && !stage.exists());
    assert!(keep.iter().all(|k| k.is_dir()));
    assert!(!recover_replaced_dir(&target).unwrap());

    // The same state is also recovered by the next replace, which then
    // replaces the restored directory.
    fs::rename(&target, &backup).unwrap();
    replace_dir_with(&target, fill_with("new")).unwrap();
    assert_eq!(snapshot(&target), expected("new"));
    assert!(!backup.exists());
}

#[test]
fn a_crash_after_the_second_rename_leaves_litter_the_next_call_removes() {
    let s = scratch("litter");
    let target = s.0.join("ckpt");
    replace_dir_with(&target, fill_with("new")).unwrap();
    let backup = leftover(&s.0, "ckpt", 900_003, "old");
    fs::create_dir(&backup).unwrap();
    fill_with("old")(&backup).unwrap();
    assert!(!recover_replaced_dir(&target).unwrap());
    assert_eq!(snapshot(&target), expected("new"));
    assert_eq!(names(&s.0), ["ckpt"]);
}

#[test]
fn two_backups_and_no_target_are_refused_untouched() {
    let s = scratch("ambiguous");
    let target = s.0.join("ckpt");
    for (seq, tag) in [(900_004, "a"), (900_005, "b")] {
        let b = leftover(&s.0, "ckpt", seq, "old");
        fs::create_dir(&b).unwrap();
        fill_with(tag)(&b).unwrap();
    }
    expect_err(recover_replaced_dir(&target), "2 backups");
    expect_err(replace_dir_with(&target, fill_with("x")), "2 backups");
    assert!(!target.exists());
    assert_eq!(names(&s.0).len(), 2);
}

#[test]
fn a_stage_deeper_than_the_cap_or_holding_a_fifo_is_refused() {
    let s = scratch("deep");
    let target = s.0.join("ckpt");
    replace_dir_with(&target, fill_with("old")).unwrap();
    let err = replace_dir_with(&target, |stage| {
        let mut p = stage.to_path_buf();
        for i in 0..18 {
            p = p.join(format!("d{i}"));
        }
        fs::create_dir_all(&p).unwrap();
        Ok(())
    });
    expect_err(err, "staged tree deeper than 16");
    assert_eq!(snapshot(&target), expected("old"));
    assert_eq!(names(&s.0), ["ckpt"]);

    let err = replace_dir_with(&target, |stage| {
        let made = std::process::Command::new("mkfifo")
            .arg(stage.join("pipe"))
            .status()
            .map_err(|e| IoError::new(format!("mkfifo: {e}")))?;
        assert!(made.success(), "mkfifo failed");
        Ok(())
    });
    expect_err(err, "is not a regular file or directory");
    assert_eq!(snapshot(&target), expected("old"));
    assert_eq!(names(&s.0), ["ckpt"]);
}

#[test]
fn a_stage_that_cannot_be_removed_is_named_in_the_error() {
    use std::os::unix::fs::PermissionsExt;
    let s = scratch("stuck");
    let target = s.0.join("ckpt");
    replace_dir_with(&target, fill_with("old")).unwrap();
    let locked = std::cell::RefCell::new(PathBuf::new());
    let err = replace_dir_with(&target, |stage| {
        let sub = stage.join("sub");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("f"), b"x").unwrap();
        fs::set_permissions(&sub, fs::Permissions::from_mode(0o555)).unwrap();
        *locked.borrow_mut() = sub;
        Err(IoError::new("fill failed"))
    })
    .expect_err("fill error was not returned");
    let sub = locked.into_inner();
    let stage = sub.parent().unwrap().to_path_buf();
    let removable = !stage.exists();
    make_writable(&sub).unwrap_or(());
    if removable {
        // Running as root: the read-only bit did not stop the removal.
        eprintln!("skipped: the read-only bit is not enforced for this user");
        return;
    }
    assert!(err.detail().contains("fill failed"), "{err}");
    assert!(
        err.detail()
            .contains(&format!("removing stage {} also failed", stage.display())),
        "{err}"
    );
    assert_eq!(snapshot(&target), expected("old"));
    // With the permission restored, the next call sweeps the stale stage.
    replace_dir_with(&target, fill_with("new")).unwrap();
    assert_eq!(snapshot(&target), expected("new"));
    assert_eq!(names(&s.0), ["ckpt"]);
}
