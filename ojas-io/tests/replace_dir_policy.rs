//! `replace_dir_with_if`: the caller's policy for an existing target runs
//! under the parent lock, before anything is touched. Unix only.
#![cfg(unix)]

use ojas_io::{replace_dir_with, replace_dir_with_if, IoError};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

static SEQ: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn scratch(tag: &str) -> Scratch {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("ojas-io-policy-{}-{tag}-{n}", std::process::id()));
    fs::create_dir(&p).unwrap();
    Scratch(p)
}

/// Every entry under `dir` by relative path: its bytes for a file, `None`
/// for a directory, the link text for a symlink.
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
                out.insert(rel.clone(), "dir".to_string());
                walk(root, &p, out);
            } else if kind.is_file() {
                out.insert(rel, format!("file {:?}", fs::read(&p).unwrap()));
            } else {
                out.insert(rel, "special".to_string());
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

fn write_marker(stage: &Path) -> Result<(), IoError> {
    fs::write(stage.join("marker"), "new").unwrap();
    Ok(())
}

/// Allows an existing target only when it holds a file named `marker`.
fn needs_marker(dir: &Path) -> Result<(), IoError> {
    if fs::symlink_metadata(dir.join("marker")).is_ok() {
        Ok(())
    } else {
        Err(IoError::new("no marker"))
    }
}

#[test]
fn an_absent_target_is_not_offered_to_the_predicate() {
    let s = scratch("absent");
    let target = s.0.join("ckpt");
    let asked = AtomicBool::new(false);
    replace_dir_with_if(
        &target,
        |_| {
            asked.store(true, Ordering::SeqCst);
            Err(IoError::new("never allowed"))
        },
        write_marker,
    )
    .unwrap();
    assert!(!asked.load(Ordering::SeqCst));
    assert_eq!(names(&target), ["marker"]);
}

#[test]
fn an_allowed_target_is_replaced_whole_and_sees_its_own_path() {
    let s = scratch("allow");
    let target = s.0.join("ckpt");
    fs::create_dir(&target).unwrap();
    fs::write(target.join("marker"), "old").unwrap();
    fs::write(target.join("extra"), "old").unwrap();
    let seen = std::cell::RefCell::new(None);
    replace_dir_with_if(
        &target,
        |p| {
            *seen.borrow_mut() = Some(p.to_path_buf());
            needs_marker(p)
        },
        write_marker,
    )
    .unwrap();
    assert_eq!(seen.into_inner().as_deref(), Some(target.as_path()));
    assert_eq!(
        tree(&target),
        BTreeMap::from([("marker".to_string(), format!("file {:?}", b"new"))])
    );
    assert_eq!(
        names(&s.0),
        ["ckpt"],
        "a stage or backup was left beside it"
    );
}

/// A refusal returns the predicate's detail, never calls `fill`, and leaves
/// every entry of the target (files, a nested directory, a symlink, a FIFO)
/// and its parent exactly as they were.
#[test]
fn a_refusal_touches_nothing() {
    let s = scratch("refuse");
    let target = s.0.join("runs");
    fs::create_dir_all(target.join("a/deep")).unwrap();
    fs::write(target.join("a/deep/w"), [1u8, 2, 3]).unwrap();
    fs::write(target.join("STATE.OJCK"), "case variant").unwrap();
    std::os::unix::fs::symlink("elsewhere", target.join("link")).unwrap();
    let fifo = std::process::Command::new("mkfifo")
        .arg(target.join("pipe"))
        .status()
        .unwrap();
    assert!(fifo.success());
    let before = tree(&target);
    let parent_before = names(&s.0);
    let filled = AtomicBool::new(false);
    let err = replace_dir_with_if(&target, needs_marker, |_| {
        filled.store(true, Ordering::SeqCst);
        Ok(())
    })
    .unwrap_err();
    assert!(err.detail().contains("no marker"), "{err}");
    assert!(
        err.detail().contains(&target.display().to_string()),
        "the error does not name the target: {err}"
    );
    assert!(!filled.load(Ordering::SeqCst), "fill ran after a refusal");
    assert_eq!(tree(&target), before);
    assert_eq!(names(&s.0), parent_before);
}

/// The predicate runs inside the lock: a nested replace for the same parent
/// from it is the lock's own deadlock refusal, and another thread's replace
/// of the same target does not reach its own predicate until this one
/// returns.
#[test]
fn the_predicate_runs_under_the_parent_lock() {
    let s = scratch("lock");
    let target = s.0.join("ckpt");
    replace_dir_with(&target, write_marker).unwrap();
    let sibling = s.0.join("other");
    let nested = std::cell::RefCell::new(None);
    replace_dir_with_if(
        &target,
        |_| {
            *nested.borrow_mut() = Some(replace_dir_with(&sibling, write_marker));
            Ok(())
        },
        write_marker,
    )
    .unwrap();
    let nested = nested.into_inner().unwrap().unwrap_err();
    assert!(nested.detail().contains("already running"), "{nested}");

    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let second_entered = Arc::new(AtomicBool::new(false));
    let first = {
        let target = target.clone();
        std::thread::spawn(move || {
            let mut first = true;
            replace_dir_with_if(
                &target,
                |_| {
                    // The predicate runs twice; only the first call parks.
                    if std::mem::take(&mut first) {
                        entered_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                    }
                    Ok(())
                },
                write_marker,
            )
        })
    };
    entered_rx.recv().unwrap();
    let second = {
        let target = target.clone();
        let flag = Arc::clone(&second_entered);
        std::thread::spawn(move || {
            replace_dir_with_if(
                &target,
                |_| {
                    flag.store(true, Ordering::SeqCst);
                    Ok(())
                },
                write_marker,
            )
        })
    };
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        !second_entered.load(Ordering::SeqCst),
        "a second predicate ran while the first held the lock"
    );
    release_tx.send(()).unwrap();
    first.join().unwrap().unwrap();
    second.join().unwrap().unwrap();
    assert!(second_entered.load(Ordering::SeqCst));
}

/// Writers to one target serialize, and a refused writer deletes nothing: the
/// directory always holds exactly one writer's whole content.
#[test]
fn concurrent_writers_serialize_and_a_refusal_deletes_nothing() {
    let s = scratch("race");
    let target = s.0.join("ckpt");
    replace_dir_with(&target, write_marker).unwrap();
    let allowed = Arc::new(AtomicUsize::new(0));
    let refused = Arc::new(AtomicUsize::new(0));
    let mut threads = Vec::new();
    for i in 0..8 {
        let target = target.clone();
        let (allowed, refused) = (Arc::clone(&allowed), Arc::clone(&refused));
        threads.push(std::thread::spawn(move || {
            for round in 0..10 {
                // Odd threads refuse every other round.
                let refuse = i % 2 == 1 && round % 2 == 0;
                let r = replace_dir_with_if(
                    &target,
                    |p| {
                        if refuse {
                            Err(IoError::new("refused"))
                        } else {
                            needs_marker(p)
                        }
                    },
                    |stage| {
                        fs::write(stage.join("marker"), format!("{i}-{round}")).unwrap();
                        fs::write(stage.join("payload"), format!("{i}-{round}")).unwrap();
                        Ok(())
                    },
                );
                match r {
                    Ok(()) => allowed.fetch_add(1, Ordering::SeqCst),
                    Err(e) => {
                        assert!(e.detail().contains("refused"), "{e}");
                        refused.fetch_add(1, Ordering::SeqCst)
                    }
                };
                let marker = fs::read_to_string(target.join("marker")).unwrap();
                let payload = fs::read_to_string(target.join("payload")).ok();
                assert!(
                    payload.is_none() || payload.as_deref() == Some(marker.as_str()),
                    "a torn directory: marker {marker:?}, payload {payload:?}"
                );
            }
        }));
    }
    for t in threads {
        t.join().unwrap();
    }
    assert_eq!(allowed.load(Ordering::SeqCst), 60);
    assert_eq!(refused.load(Ordering::SeqCst), 20);
    assert_eq!(names(&s.0), ["ckpt"]);
}
