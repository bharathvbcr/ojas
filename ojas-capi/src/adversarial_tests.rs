//! Adversarial cases for SAVE and RESUME through the C-ABI request layer:
//! path tricks, cancellation cut at every poll, and the RESUME header
//! preflight against lying, missing, huge and non-regular weights files.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use crate::session;
use crate::tests::{
    fresh, load_exact, nano_spec, never, step, train_fields, train_open, write_bin,
};
use crate::train;
use crate::wire::{tag, Writer};

fn tree(root: &Path) -> BTreeMap<String, String> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let e = e.unwrap();
            let p = e.path();
            let rel = p.strip_prefix(root).unwrap().to_string_lossy().into_owned();
            let kind = e.file_type().unwrap();
            if kind.is_symlink() {
                out.insert(rel, "link".into());
            } else if kind.is_dir() {
                out.insert(rel.clone(), "dir".into());
                walk(root, &p, out);
            } else if kind.is_file() {
                out.insert(rel, format!("{:x?}", std::fs::read(&p).unwrap()));
            } else {
                out.insert(rel, "special".into());
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

fn trained(dir: &Path) -> session::Session {
    write_bin(dir, "tokens.bin");
    let s = load_exact("model.safetensors").unwrap();
    train_open(s.id, "tokens.bin").unwrap();
    step(s.id).unwrap();
    s
}

fn save_with(id: u64, dir: &str, check: crate::gate::Check) -> Result<(), String> {
    let mut p = id.to_le_bytes().to_vec();
    p.extend_from_slice(dir.as_bytes());
    train::save_request(&p, check)
}

fn save(id: u64, dir: &str) -> Result<(), String> {
    save_with(id, dir, never())
}

fn resume_payload(dir: &str, budget: Option<u64>) -> Vec<u8> {
    let w = train_fields(Writer::default().str(tag::PATH, dir), "tokens.bin").u32(tag::NUMERICS, 1);
    match budget {
        Some(b) => w.u64(tag::BUDGET, b).finish(),
        None => w.finish(),
    }
}

/// `..`, a trailing slash, a `.`-only path, the root itself through an
/// in-root symlink, and an ancestor of the root are all refused and change
/// nothing under the root.
#[test]
fn save_path_tricks_are_refused_and_touch_nothing() {
    let (_g, dir) = fresh();
    let s = trained(&dir);
    save(s.id, "ckpt").unwrap();
    std::os::unix::fs::symlink(".", dir.join("self")).unwrap();
    std::os::unix::fs::symlink("..", dir.join("up")).unwrap();
    let before = tree(&dir);
    for raw in [
        ".",
        "./",
        "ckpt/",
        "ckpt/.",
        "ckpt/..",
        "../",
        "..",
        "a/../ckpt",
        "self",
        "up",
        "self/ckpt/..",
        "up/x",
        "ckpt//",
        "//ckpt",
    ] {
        let r = save(s.id, raw);
        assert!(r.is_err(), "{raw:?} was accepted: {r:?}");
    }
    assert_eq!(tree(&dir), before);
    // `ckpt/.` is the same directory as `ckpt` once components are
    // normalized: a legitimate re-save, not a trick.
    save(s.id, "ckpt/.").unwrap();
}

/// A symlink inside the root to an earlier checkpoint resolves to the real
/// directory and saves into it (CHARACTERIZATION: the alias itself is not
/// replaced by a directory).
#[test]
fn a_symlink_to_a_checkpoint_saves_into_the_real_directory() {
    let (_g, dir) = fresh();
    let s = trained(&dir);
    save(s.id, "ckpt").unwrap();
    std::os::unix::fs::symlink("ckpt", dir.join("alias")).unwrap();
    save(s.id, "alias").unwrap();
    assert!(std::fs::symlink_metadata(dir.join("alias"))
        .unwrap()
        .file_type()
        .is_symlink());
    assert!(dir.join("ckpt").join(ojas_model::STATE_FILE).is_file());
}

/// Cut SAVE at every check the request makes. Each cut is the cancel
/// message with the old tree byte-identical; the one past the last is a
/// success: a cancel after the swap is never reported as cancelled.
#[test]
fn save_cut_at_every_check_keeps_the_tree_and_a_late_cancel_is_success() {
    let (_g, dir) = fresh();
    let s = trained(&dir);
    save(s.id, "ckpt").unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&count);
    save_with(
        s.id,
        "ckpt",
        Box::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }),
    )
    .unwrap();
    let n = count.load(Ordering::SeqCst);
    let before = tree(&dir);
    for cut in 1..=n {
        let seen = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&seen);
        let err = save_with(
            s.id,
            "ckpt",
            Box::new(move || {
                if c.fetch_add(1, Ordering::SeqCst) + 1 >= cut {
                    Err("cancelled: Explicit".into())
                } else {
                    Ok(())
                }
            }),
        )
        .unwrap_err();
        assert_eq!(err, "cancelled: Explicit", "cut {cut}/{n}");
        assert_eq!(tree(&dir), before, "cut {cut}/{n}");
    }
    let seen = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&seen);
    save_with(
        s.id,
        "ckpt",
        Box::new(move || {
            if c.fetch_add(1, Ordering::SeqCst) + 1 > n {
                Err("cancelled: Explicit".into())
            } else {
                Ok(())
            }
        }),
    )
    .unwrap();
    // The session still trains after all that.
    step(s.id).unwrap();
}

/// 100 cancelled SAVEs: no stage or backup entries, and the session is
/// still usable.
#[test]
fn a_hundred_cancelled_saves_leave_the_root_unchanged() {
    let (_g, dir) = fresh();
    let s = trained(&dir);
    save(s.id, "ckpt").unwrap();
    let before = tree(&dir);
    for i in 0..100usize {
        let cut = 1 + i % 12;
        let seen = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&seen);
        let r = save_with(
            s.id,
            if i % 2 == 0 { "ckpt" } else { "fresh" },
            Box::new(move || {
                if c.fetch_add(1, Ordering::SeqCst) + 1 >= cut {
                    Err("cancelled: Explicit".into())
                } else {
                    Ok(())
                }
            }),
        );
        // A cut past the final check succeeds and writes `fresh`; remove it
        // so the comparison below stays about cancelled saves.
        if r.is_ok() {
            let _ = std::fs::remove_dir_all(dir.join("fresh"));
        }
    }
    assert_eq!(tree(&dir), before);
    save(s.id, "ckpt").unwrap();
}

fn patch_header(path: &Path, from: &[u8], to: &[u8]) {
    assert_eq!(from.len(), to.len());
    let mut bytes = std::fs::read(path).unwrap();
    let at = bytes
        .windows(from.len())
        .position(|w| w == from)
        .expect("pattern in header");
    bytes[at..at + from.len()].copy_from_slice(to);
    std::fs::write(path, bytes).unwrap();
}

/// A header whose `ojas.spec` is smaller than the tensors that follow: the
/// preflight accepts the small budget the lying spec fits in, and the
/// directory check that follows must refuse. No session, no panic.
#[test]
fn resume_with_a_lying_spec_header_is_refused_and_leaks_no_session() {
    let (_g, dir) = fresh();
    let s = trained(&dir);
    save(s.id, "ckpt").unwrap();
    let real = ojas_model::param_bytes(&nano_spec()).unwrap();
    let mut small = nano_spec();
    small.n_layer = 1;
    let lying = ojas_model::param_bytes(&small).unwrap();
    assert!(lying < real);
    patch_header(
        &dir.join("ckpt").join(ojas_model::MODEL_FILE),
        br#"\"n_layer\":2"#,
        br#"\"n_layer\":1"#,
    );
    let before = session::session_count();
    let tree_before = tree(&dir);
    let err = train::resume_request(&resume_payload("ckpt", Some(lying * 4)), never());
    assert!(err.is_err(), "a lying header was accepted");
    assert_eq!(session::session_count(), before);
    assert_eq!(tree(&dir), tree_before);
}

#[test]
fn resume_with_a_missing_or_garbled_spec_is_refused() {
    let (_g, dir) = fresh();
    let s = trained(&dir);
    save(s.id, "ckpt").unwrap();
    let model = dir.join("ckpt").join(ojas_model::MODEL_FILE);
    let good = std::fs::read(&model).unwrap();
    let before = session::session_count();
    // The key renamed: same length, so the header still parses.
    patch_header(&model, b"ojas.spec", b"ojas.spoc");
    let err = train::resume_request(&resume_payload("ckpt", None), never()).unwrap_err();
    assert!(err.contains("spec"), "{err}");
    std::fs::write(&model, &good).unwrap();
    // Empty file, 8 bytes, and a header length past every bound.
    for bytes in [
        Vec::new(),
        vec![0u8; 8],
        u64::MAX.to_le_bytes().to_vec(),
        100_000_001u64.to_le_bytes().to_vec(),
        {
            let mut v = 99_999_999u64.to_le_bytes().to_vec();
            v.extend_from_slice(b"{}");
            v
        },
    ] {
        std::fs::write(&model, &bytes).unwrap();
        let t = std::time::Instant::now();
        let r = train::resume_request(&resume_payload("ckpt", None), never());
        assert!(r.is_err(), "{} bytes accepted", bytes.len());
        assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
    }
    assert_eq!(session::session_count(), before);
}

#[test]
fn resume_refuses_a_symlink_a_directory_and_a_fifo_for_the_weights_file() {
    let (_g, dir) = fresh();
    let s = trained(&dir);
    save(s.id, "ckpt").unwrap();
    let model = dir.join("ckpt").join(ojas_model::MODEL_FILE);
    let real = dir.join("real.safetensors");
    std::fs::rename(&model, &real).unwrap();
    let before = session::session_count();

    std::os::unix::fs::symlink(&real, &model).unwrap();
    let err = train::resume_request(&resume_payload("ckpt", None), never()).unwrap_err();
    assert!(err.contains("symbolic link"), "{err}");
    std::fs::remove_file(&model).unwrap();

    std::fs::create_dir(&model).unwrap();
    assert!(train::resume_request(&resume_payload("ckpt", None), never()).is_err());
    std::fs::remove_dir(&model).unwrap();

    assert!(std::process::Command::new("mkfifo")
        .arg(&model)
        .status()
        .unwrap()
        .success());
    let (tx, rx) = mpsc::channel();
    let payload = resume_payload("ckpt", None);
    std::thread::spawn(move || {
        let r = train::resume_request(&payload, never());
        let _ = tx.send(r.map(|s| s.id));
    });
    let r = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("RESUME blocked on a FIFO in place of the weights file");
    assert!(r.is_err());
    assert_eq!(session::session_count(), before);
}
