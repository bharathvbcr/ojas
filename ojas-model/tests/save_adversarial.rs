//! Adversarial cases for SAVE's target policy and cancellation:
//! `check_save_target` against odd entries and names, `save_polled` cut at
//! every poll, and recovery meeting a refusal.
#![cfg(unix)]

mod common;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use common::{token_bin, TempBin, TempDir};
use ojas_core::{Budget, Numerics, Tensor};
use ojas_cpu::{CosineSchedule, CpuBackend, LrSchedule};
use ojas_model::{
    check_save_target, init_params, ModelSpec, SaveError, TrainConfig, Trainer, MODEL_FILE,
    OPTIM_FILE, STATE_FILE,
};

fn config() -> TrainConfig {
    let schedule = LrSchedule::Cosine(CosineSchedule::new(2, 40).unwrap());
    TrainConfig::nanolab(2, 32, 2, 1337, schedule)
}

fn trainer(budget: &Budget) -> (TempBin, Trainer<CpuBackend>) {
    let params: Vec<Tensor> = init_params(&ModelSpec::tiny(), 5, &Budget::new(1 << 30)).unwrap();
    let (tmp, bin) = token_bin(20_000, 256);
    let backend = CpuBackend::new(budget.clone()).with_numerics(Numerics::Exact);
    let mut t = Trainer::new(backend, ModelSpec::tiny(), &params, bin, config()).unwrap();
    t.step().unwrap();
    (tmp, t)
}

/// Every entry (hidden ones too) under `root`, by relative path.
fn tree(root: &Path) -> BTreeMap<String, String> {
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
            } else if kind.is_file() {
                out.insert(rel, format!("file {:x?}", fs::read(&p).unwrap()));
            } else {
                out.insert(rel, "special".into());
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

fn dir_with(root: &Path, name: &str, build: impl FnOnce(&Path)) -> PathBuf {
    let d = root.join(name);
    fs::create_dir(&d).unwrap();
    build(&d);
    d
}

fn state(d: &Path) {
    fs::write(d.join(STATE_FILE), b"s").unwrap();
}

fn mkfifo(p: &Path) {
    assert!(std::process::Command::new("mkfifo")
        .arg(p)
        .status()
        .unwrap()
        .success());
}

#[test]
fn every_odd_entry_kind_refuses_a_checkpoint_shaped_directory() {
    let td = TempDir::new();
    let outside = td.path.join("outside-file");
    fs::write(&outside, b"x").unwrap();
    let cases: Vec<(&str, PathBuf)> = vec![
        (
            "fifo-optim",
            dir_with(&td.path, "fifo", |d| {
                state(d);
                mkfifo(&d.join(OPTIM_FILE));
            }),
        ),
        (
            "socket-model",
            dir_with(&td.path, "sock", |d| {
                state(d);
                // Keep the listener alive until the end of the loop below.
                std::mem::forget(
                    std::os::unix::net::UnixListener::bind(d.join(MODEL_FILE)).unwrap(),
                );
            }),
        ),
        (
            "dangling-symlink",
            dir_with(&td.path, "dangling", |d| {
                state(d);
                std::os::unix::fs::symlink("/nonexistent/x", d.join(MODEL_FILE)).unwrap();
            }),
        ),
        (
            "dir-named-model",
            dir_with(&td.path, "subdir", |d| {
                state(d);
                fs::create_dir(d.join(MODEL_FILE)).unwrap();
            }),
        ),
        (
            "symlink-to-outside-file",
            dir_with(&td.path, "outlink", |d| {
                state(d);
                std::os::unix::fs::symlink(&outside, d.join(MODEL_FILE)).unwrap();
            }),
        ),
        (
            "state-is-a-symlink",
            dir_with(&td.path, "statelink", |d| {
                std::os::unix::fs::symlink(&outside, d.join(STATE_FILE)).unwrap();
            }),
        ),
        (
            "state-is-a-dir",
            dir_with(&td.path, "statedir", |d| {
                fs::create_dir(d.join(STATE_FILE)).unwrap();
            }),
        ),
    ];
    for (what, dir) in &cases {
        assert!(check_save_target(dir).is_err(), "{what} was accepted");
    }
}

/// Names that look like `state.ojck` to a human or to a normalizing
/// filesystem are compared byte for byte, as the entry is listed on disk.
#[test]
fn lookalike_names_are_refused_by_their_on_disk_bytes() {
    let td = TempDir::new();
    for (i, name) in [
        "state.ojck\u{200b}",
        "state.ojck\u{0301}",
        "state\u{200b}.ojck",
        "State.ojck",
        "STATE.OJCK",
        "state.ojck ",
        "state.ojck.",
        " state.ojck",
        "state.ojck\u{feff}",
    ]
    .iter()
    .enumerate()
    {
        let d = dir_with(&td.path, &format!("d{i}"), |_| {});
        fs::write(d.join(name), b"x").unwrap();
        assert!(check_save_target(&d).is_err(), "{name:?} was accepted");
        // And next to a genuine state file it is a stray.
        state(&d);
        assert!(
            check_save_target(&d).is_err(),
            "{name:?} beside state.ojck was accepted"
        );
    }
    // The on-disk name decides, not the name used to create it: on a
    // case-insensitive volume a later `STATE.OJCK` open reaches the same
    // file, and the directory still lists `state.ojck`.
    let d = dir_with(&td.path, "real", state);
    let _ = fs::OpenOptions::new()
        .write(true)
        .open(d.join("STATE.OJCK"));
    assert!(check_save_target(&d).is_ok());
}

/// JUDGMENT CALL. Finder drops `.DS_Store` into any directory it opens. The
/// strict policy refuses such a checkpoint: a re-save to a directory the
/// user browsed in Finder fails until the file is removed by hand.
#[test]
fn a_finder_ds_store_blocks_re_saving_a_checkpoint() {
    let td = TempDir::new();
    let d = dir_with(&td.path, "ckpt", |d| {
        state(d);
        fs::write(d.join(MODEL_FILE), b"m").unwrap();
        fs::write(d.join(OPTIM_FILE), b"o").unwrap();
        fs::write(d.join(".DS_Store"), b"finder").unwrap();
    });
    let err = check_save_target(&d).unwrap_err();
    assert!(err.contains(".DS_Store"), "{err}");
    let empty_but_ds = dir_with(&td.path, "ds-only", |d| {
        fs::write(d.join(".DS_Store"), b"finder").unwrap()
    });
    assert!(check_save_target(&empty_but_ds).is_err());
}

/// A refused real save over each odd directory leaves every byte alone.
#[test]
fn a_refused_save_over_odd_directories_changes_nothing() {
    let td = TempDir::new();
    let budget = Budget::new(1 << 30);
    let (_bin, t) = trainer(&budget);
    let target = dir_with(&td.path, "ckpt", |d| {
        state(d);
        mkfifo(&d.join(OPTIM_FILE));
        fs::write(d.join(MODEL_FILE), b"m").unwrap();
    });
    let before = tree(&td.path);
    let live = budget.live_bytes().unwrap();
    assert!(matches!(
        t.save_polled(&target, &mut || Ok(())),
        Err(SaveError::TargetRefused(_))
    ));
    assert_eq!(tree(&td.path), before);
    assert_eq!(budget.live_bytes().unwrap(), live);
}

/// Cut SAVE at every poll it makes over an existing checkpoint, and over an
/// absent path: each cut is `Cancelled`, the old tree is byte-identical (no
/// stage, no backup), and nothing stays charged to the budget. A cancel that
/// only starts after the last poll cannot be seen: the save succeeds.
#[test]
fn a_cut_at_every_poll_leaves_the_old_checkpoint_and_no_stage() {
    let td = TempDir::new();
    let budget = Budget::new(1 << 30);
    let (_bin, t) = trainer(&budget);
    t.save(&td.ckpt()).unwrap();
    let polls = std::cell::Cell::new(0usize);
    t.save_polled(&td.ckpt(), &mut || {
        polls.set(polls.get() + 1);
        Ok(())
    })
    .unwrap();
    let n = polls.get();
    assert!(n > 2, "{n} polls");
    let before = tree(&td.path);
    let live = budget.live_bytes().unwrap();
    for (label, target) in [("existing", td.ckpt()), ("absent", td.path.join("fresh"))] {
        for cut in 1..=n {
            let seen = std::cell::Cell::new(0usize);
            let r = t.save_polled(&target, &mut || {
                seen.set(seen.get() + 1);
                if seen.get() >= cut {
                    Err("cancelled".to_string())
                } else {
                    Ok(())
                }
            });
            match r {
                Err(SaveError::Cancelled(m)) => assert_eq!(m, "cancelled"),
                other => panic!("{label} cut {cut}/{n}: {other:?}"),
            }
            assert_eq!(seen.get(), cut, "{label} cut {cut}: polled past the cancel");
            assert_eq!(tree(&td.path), before, "{label} cut {cut}: tree changed");
            assert_eq!(budget.live_bytes().unwrap(), live, "{label} cut {cut}");
        }
    }
    // A cancel that only lands after the final poll is not seen: success.
    let seen = std::cell::Cell::new(0usize);
    t.save_polled(&td.ckpt(), &mut || {
        seen.set(seen.get() + 1);
        if seen.get() > n {
            Err("cancelled".to_string())
        } else {
            Ok(())
        }
    })
    .unwrap();
    assert_eq!(seen.get(), n);
}

/// 100 cancelled saves at a rotating cut: no stage dirs pile up, no budget
/// bytes leak, and the lock is free each time.
#[test]
fn a_hundred_cancelled_saves_leak_nothing() {
    let td = TempDir::new();
    let budget = Budget::new(1 << 30);
    let (_bin, t) = trainer(&budget);
    t.save(&td.ckpt()).unwrap();
    let polls = std::cell::Cell::new(0usize);
    t.save_polled(&td.ckpt(), &mut || {
        polls.set(polls.get() + 1);
        Ok(())
    })
    .unwrap();
    let n = polls.get();
    let before = tree(&td.path);
    let live = budget.live_bytes().unwrap();
    let peak_marker = budget.live_bytes().unwrap();
    for i in 0..100 {
        let cut = 1 + i % n;
        let seen = std::cell::Cell::new(0usize);
        let r = t.save_polled(&td.ckpt(), &mut || {
            seen.set(seen.get() + 1);
            if seen.get() >= cut {
                Err("cancelled".to_string())
            } else {
                Ok(())
            }
        });
        assert!(matches!(r, Err(SaveError::Cancelled(_))), "round {i}");
    }
    assert_eq!(tree(&td.path), before);
    assert_eq!(budget.live_bytes().unwrap(), live);
    assert_eq!(live, peak_marker);
    assert_eq!(fs::read_dir(&td.path).unwrap().count(), 1);
}

/// A poll that panics mid-save unwinds through the stage guard and the
/// lock: the old tree is intact, no stage remains, and the next save works.
#[test]
fn a_panicking_poll_leaves_the_old_checkpoint_and_frees_the_lock() {
    let td = TempDir::new();
    let budget = Budget::new(1 << 30);
    let (_bin, t) = trainer(&budget);
    t.save(&td.ckpt()).unwrap();
    let before = tree(&td.path);
    let seen = std::cell::Cell::new(0usize);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = t.save_polled(&td.ckpt(), &mut || {
            seen.set(seen.get() + 1);
            if seen.get() == 3 {
                panic!("poll panicked");
            }
            Ok(())
        });
    }));
    assert!(r.is_err());
    assert_eq!(tree(&td.path), before);
    t.save(&td.ckpt()).unwrap();
}

/// A crash between the two renames leaves only a backup of a valid
/// checkpoint: SAVE restores it, the policy then allows it, and the new
/// checkpoint replaces it with nothing left beside.
#[test]
fn save_after_a_crash_between_the_renames_recovers_then_replaces() {
    let td = TempDir::new();
    let budget = Budget::new(1 << 30);
    let (_bin, t) = trainer(&budget);
    t.save(&td.ckpt()).unwrap();
    let backup = td.path.join(format!(".ckpt.{}.9.old", std::process::id()));
    fs::rename(td.ckpt(), &backup).unwrap();
    t.save(&td.ckpt()).unwrap();
    let names: Vec<_> = fs::read_dir(&td.path)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(names, ["ckpt"]);
}

/// The same crash state but the backup is not a checkpoint (the user's own
/// directory, moved aside by an earlier crash): recovery restores it, the
/// policy refuses, and the restored directory is whole.
#[test]
fn save_after_a_crash_with_a_non_checkpoint_backup_restores_and_refuses() {
    let td = TempDir::new();
    let budget = Budget::new(1 << 30);
    let (_bin, t) = trainer(&budget);
    let backup = td.path.join(format!(".ckpt.{}.9.old", std::process::id()));
    fs::create_dir(&backup).unwrap();
    fs::write(backup.join("thesis.tex"), b"irreplaceable").unwrap();
    let err = t.save_polled(&td.ckpt(), &mut || Ok(())).unwrap_err();
    assert!(matches!(err, SaveError::TargetRefused(_)), "{err}");
    assert_eq!(
        fs::read(td.ckpt().join("thesis.tex")).unwrap(),
        b"irreplaceable"
    );
    assert!(!backup.exists());
}
