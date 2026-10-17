//! Adversarial cases for `replace_dir_with_if`: what the lock does and does
//! not exclude, a target swapped for a symlink mid-replace, odd target
//! paths, and a stale backup meeting a refusal. Unix only.
#![cfg(unix)]

use ojas_io::{replace_dir_with_if, IoError};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static SEQ: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn scratch(tag: &str) -> Scratch {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("ojas-io-adv-{}-{tag}-{n}", std::process::id()));
    fs::create_dir(&p).unwrap();
    // The system temp dir may sit under a symlink; the lock only checks the
    // immediate parent, so canonicalize once here.
    Scratch(p.canonicalize().unwrap())
}

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    v.sort();
    v
}

fn tree(dir: &Path) -> BTreeMap<String, String> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) {
        for e in fs::read_dir(dir).unwrap() {
            let e = e.unwrap();
            let p = e.path();
            let rel = p.strip_prefix(root).unwrap().to_string_lossy().into_owned();
            let kind = e.file_type().unwrap();
            if kind.is_symlink() {
                out.insert(
                    rel,
                    format!("link {}", fs::read_link(&p).unwrap().display()),
                );
            } else if kind.is_dir() {
                out.insert(rel.clone(), "dir".into());
                walk(root, &p, out);
            } else {
                out.insert(rel, format!("file {:?}", fs::read(&p).unwrap()));
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(dir, dir, &mut out);
    out
}

fn marker(stage: &Path) -> Result<(), IoError> {
    fs::write(stage.join("marker"), "new").unwrap();
    Ok(())
}

fn needs_marker(dir: &Path) -> Result<(), IoError> {
    if fs::symlink_metadata(dir.join("marker")).is_ok() {
        Ok(())
    } else {
        Err(IoError::new("no marker"))
    }
}

fn old_target(s: &Scratch) -> PathBuf {
    let t = s.0.join("ckpt");
    fs::create_dir(&t).unwrap();
    fs::write(t.join("marker"), "old").unwrap();
    t
}

/// The parent lock is an flock, so it excludes any other open file
/// description on the parent, in this process or another one: a second
/// descriptor cannot take it while the predicate runs.
#[test]
fn the_lock_excludes_other_descriptors_not_just_threads() {
    let s = scratch("flock");
    let t = old_target(&s);
    let blocked = std::cell::Cell::new(false);
    replace_dir_with_if(
        &t,
        |_| {
            let other = fs::File::open(&s.0).unwrap();
            blocked.set(other.try_lock().is_err());
            Ok(())
        },
        marker,
    )
    .unwrap();
    assert!(blocked.get(), "a second descriptor took the parent lock");
}

/// Allows an existing target only when it holds exactly `marker`.
fn only_marker(dir: &Path) -> Result<(), IoError> {
    if names(dir) == ["marker"] {
        Ok(())
    } else {
        Err(IoError::new("not only a marker"))
    }
}

/// A writer that does not take the parent flock (a plain `open`/`write` into
/// the target, another tool, Finder, a shell) adds a file after the first
/// predicate call passed. The predicate runs again just before the rename, so
/// the swap is refused, the stray file survives and no stage is left.
#[test]
fn an_unlocked_writer_after_the_predicate_is_caught_by_the_recheck() {
    let s = scratch("stray");
    let t = old_target(&s);
    let err = replace_dir_with_if(&t, only_marker, |stage| {
        fs::write(t.join("late-user-file.txt"), "precious").unwrap();
        marker(stage)
    })
    .unwrap_err();
    assert!(err.detail().contains("not only a marker"), "{err}");
    assert_eq!(names(&t), ["late-user-file.txt", "marker"]);
    assert_eq!(fs::read(t.join("late-user-file.txt")).unwrap(), b"precious");
    assert_eq!(fs::read(t.join("marker")).unwrap(), b"old");
    assert_eq!(names(&s.0), ["ckpt"], "a stage or backup was left");
}

/// The target swapped for a symlink to another directory after the first
/// check: the re-check sees it is no longer a plain directory and refuses.
/// Nothing is deleted through the link, the other directory survives, and the
/// stage is removed.
#[test]
fn a_target_swapped_for_a_symlink_after_the_check_is_refused_untouched() {
    let s = scratch("swap");
    let t = old_target(&s);
    let outside = s.0.join("outside");
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join("sentinel"), "mine").unwrap();
    let moved = s.0.join("moved");
    let err = replace_dir_with_if(&t, needs_marker, |stage| {
        fs::rename(&t, &moved).unwrap();
        std::os::unix::fs::symlink(&outside, &t).unwrap();
        marker(stage)
    })
    .unwrap_err();
    assert!(
        err.detail().contains("no longer a plain directory"),
        "{err}"
    );
    assert!(fs::symlink_metadata(&t).unwrap().file_type().is_symlink());
    assert_eq!(fs::read(outside.join("sentinel")).unwrap(), b"mine");
    assert_eq!(fs::read(moved.join("marker")).unwrap(), b"old");
    assert_eq!(names(&s.0), ["ckpt", "moved", "outside"]);
}

/// Same swap but the symlink is placed where the stage rename lands: the
/// swap must not write the stage into the symlink's destination.
#[test]
fn a_symlink_planted_as_the_target_during_fill_is_replaced_not_entered() {
    let s = scratch("plant");
    let t = s.0.join("fresh"); // absent target
    let outside = s.0.join("outside");
    fs::create_dir(&outside).unwrap();
    let r = replace_dir_with_if(
        &t,
        |_| Ok(()),
        |stage| {
            std::os::unix::fs::symlink(&outside, &t).unwrap();
            marker(stage)
        },
    );
    // rename(stage, symlink-to-dir) is ENOTDIR/EEXIST on a symlink: it must
    // fail and leave the outside directory empty.
    assert!(r.is_err(), "{r:?}");
    assert!(names(&outside).is_empty(), "wrote through the symlink");
    assert!(fs::symlink_metadata(&t).unwrap().file_type().is_symlink());
    assert_eq!(names(&s.0), ["fresh", "outside"], "stage was left behind");
}

#[test]
fn odd_target_paths_are_refused_or_normalized_without_touching_anything() {
    let s = scratch("paths");
    let t = old_target(&s);
    let before = tree(&s.0);
    for odd in [
        PathBuf::from("."),
        PathBuf::from(".."),
        s.0.join("ckpt/.."),
        s.0.join("."),
        PathBuf::from("/"),
        PathBuf::new(),
    ] {
        let r = replace_dir_with_if(&odd, |_| Ok(()), marker);
        assert!(r.is_err(), "{odd:?} was accepted");
    }
    assert_eq!(tree(&s.0), before);
    // A trailing slash and a `./` component name the same directory.
    let mut slash = t.clone().into_os_string();
    slash.push("/");
    let called = std::cell::Cell::new(0);
    replace_dir_with_if(
        Path::new(&slash),
        |p| {
            called.set(called.get() + 1);
            needs_marker(p)
        },
        marker,
    )
    .unwrap();
    // Once before the stage is made, once just before the rename.
    assert_eq!(called.get(), 2);
    assert_eq!(fs::read(t.join("marker")).unwrap(), b"new");
    assert_eq!(names(&s.0), ["ckpt"]);
}

/// A target that is a symlink to an allowed checkpoint directory is
/// refused before the predicate runs.
#[test]
fn a_target_that_is_a_symlink_to_an_allowed_dir_is_refused_untouched() {
    let s = scratch("linkdir");
    let real = old_target(&s);
    let link = s.0.join("alias");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let before = tree(&s.0);
    let asked = std::cell::Cell::new(false);
    let err = replace_dir_with_if(
        &link,
        |_| {
            asked.set(true);
            Ok(())
        },
        marker,
    )
    .unwrap_err();
    assert!(err.detail().contains("symbolic link"), "{err}");
    assert!(!asked.get());
    assert_eq!(tree(&s.0), before);
}

#[test]
fn a_parent_that_is_a_symlink_is_refused() {
    let s = scratch("linkparent");
    let real = s.0.join("real");
    fs::create_dir(&real).unwrap();
    let link = s.0.join("alias");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let r = replace_dir_with_if(&link.join("ckpt"), |_| Ok(()), marker);
    assert!(r.unwrap_err().detail().contains("symbolic link"));
    assert!(names(&real).is_empty());
}

/// Recovery runs before the predicate: a lone backup with no target is
/// restored, and then the predicate judges the restored directory. A
/// refusal leaves the restored directory as the target, whole, with no
/// backup and no stage.
#[test]
fn a_lone_backup_is_restored_before_the_predicate_and_a_refusal_keeps_it() {
    let s = scratch("recover-refuse");
    let backup = s.0.join(format!(".ckpt.{}.7.old", std::process::id()));
    fs::create_dir(&backup).unwrap();
    fs::write(backup.join("junk"), "not a checkpoint").unwrap();
    let seen = std::cell::RefCell::new(Vec::new());
    let err = replace_dir_with_if(
        &s.0.join("ckpt"),
        |p| {
            seen.borrow_mut().push(names(p));
            needs_marker(p)
        },
        marker,
    )
    .unwrap_err();
    assert!(err.detail().contains("no marker"), "{err}");
    assert_eq!(*seen.borrow(), vec![vec!["junk".to_string()]]);
    assert_eq!(names(&s.0), ["ckpt"]);
    assert_eq!(
        fs::read(s.0.join("ckpt/junk")).unwrap(),
        b"not a checkpoint"
    );
}

/// The same with a valid old checkpoint as the backup: restored, allowed,
/// replaced, nothing left.
#[test]
fn a_lone_valid_backup_is_restored_then_replaced() {
    let s = scratch("recover-allow");
    let backup = s.0.join(format!(".ckpt.{}.7.old", std::process::id()));
    fs::create_dir(&backup).unwrap();
    fs::write(backup.join("marker"), "old").unwrap();
    replace_dir_with_if(&s.0.join("ckpt"), needs_marker, marker).unwrap();
    assert_eq!(names(&s.0), ["ckpt"]);
    assert_eq!(fs::read(s.0.join("ckpt/marker")).unwrap(), b"new");
}

/// CHARACTERIZATION. With the target present, recovery deletes leftover
/// backups and runs before the predicate, so a refusal does not leave the
/// parent exactly as found: stale backups are gone. The doc comment says a
/// refusal "touches nothing: the target, its parent and any stage are as
/// they were". A stale stage survives (the sweep runs after the predicate);
/// a stale backup does not.
#[test]
fn a_refusal_still_removes_stale_backups_but_keeps_stale_stages() {
    let s = scratch("stale-refuse");
    let t = old_target(&s);
    fs::remove_file(t.join("marker")).unwrap();
    let pid = std::process::id();
    let backup = s.0.join(format!(".ckpt.{pid}.3.old"));
    fs::create_dir(&backup).unwrap();
    fs::write(backup.join("old-data"), "stale").unwrap();
    let stage = s.0.join(format!(".ckpt.{pid}.4.stage"));
    fs::create_dir(&stage).unwrap();
    replace_dir_with_if(&t, needs_marker, marker).unwrap_err();
    assert!(!backup.exists(), "stale backup survived a refusal");
    assert!(stage.exists(), "stale stage was swept before the refusal");
}

/// Two backups and no target is refused before the predicate runs.
#[test]
fn two_backups_and_no_target_refuse_before_the_predicate() {
    let s = scratch("two-backups");
    let pid = std::process::id();
    for n in [1, 2] {
        fs::create_dir(s.0.join(format!(".ckpt.{pid}.{n}.old"))).unwrap();
    }
    let asked = std::cell::Cell::new(false);
    let err = replace_dir_with_if(
        &s.0.join("ckpt"),
        |_| {
            asked.set(true);
            Ok(())
        },
        marker,
    )
    .unwrap_err();
    assert!(err.detail().contains("2 backups"), "{err}");
    assert!(!asked.get());
    assert_eq!(names(&s.0).len(), 2);
}

/// A panic in the predicate releases the lock and leaves the target alone.
#[test]
fn a_panicking_predicate_releases_the_lock_and_touches_nothing() {
    let s = scratch("panic");
    let t = old_target(&s);
    let before = tree(&s.0);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = replace_dir_with_if(&t, |_| panic!("boom"), marker);
    }));
    assert!(r.is_err());
    assert_eq!(tree(&s.0), before);
    replace_dir_with_if(&t, needs_marker, marker).unwrap();
}
