//! ft-2ac45473 c5: a file replace takes the parent directory's exclusive `flock`, the same
//! lock the directory replace takes, so its sweep of leftover `.{name}.{pid}.{seq}.tmp` files
//! cannot remove another writer's live temp file.

#![cfg(unix)]

use ojas_io::{write_safetensors, SafeTensors};
use std::fs::File;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ojas-io-filelock-{tag}-{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write(path: &std::path::Path) -> Result<(), ojas_io::IoError> {
    write_safetensors(path, &[], &[])
}

fn assert_reads(path: &std::path::Path) {
    let st = SafeTensors::open(path).unwrap();
    assert_eq!(st.names().count(), 0);
}

/// While another descriptor holds the parent's lock, `write_safetensors` (a `replace_with`)
/// blocks; once the lock is released it completes and the file reads back.
///
/// Pre-fix (b7ecc97) `replace_with` never touched the parent lock: the write finished at once,
/// so the "still blocked" assertion failed.
#[test]
fn a_file_replace_waits_for_the_parent_lock() {
    let dir = scratch("wait");
    let target = dir.join("state.safetensors");
    let holder = File::open(&dir).unwrap();
    holder.lock().unwrap();

    let (tx, rx) = mpsc::channel();
    let t = target.clone();
    let writer = std::thread::spawn(move || {
        let _ = tx.send(write(&t));
    });
    assert!(
        rx.recv_timeout(Duration::from_millis(500)).is_err(),
        "the replace finished while another descriptor held the parent lock"
    );
    assert!(!target.exists(), "the target appeared under the held lock");

    drop(holder);
    rx.recv_timeout(Duration::from_secs(10))
        .expect("the replace did not resume after the lock was released")
        .unwrap();
    writer.join().unwrap();
    assert_reads(&target);
}

/// Two replaces of different names in one directory, from two threads, both finish: the lock
/// serializes them, it does not deadlock them.
#[test]
fn concurrent_file_replaces_in_one_directory_all_finish() {
    let dir = scratch("many");
    let (tx, rx) = mpsc::channel();
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let tx = tx.clone();
            let target = dir.join(format!("state-{i}.safetensors"));
            std::thread::spawn(move || {
                for _ in 0..5 {
                    if let Err(e) = write(&target) {
                        let _ = tx.send(Err(e));
                        return;
                    }
                }
                let _ = tx.send(Ok(i));
            })
        })
        .collect();
    drop(tx);
    for _ in 0..4 {
        rx.recv_timeout(Duration::from_secs(30))
            .expect("a writer hung")
            .unwrap();
    }
    for h in handles {
        h.join().unwrap();
    }
    for i in 0..4 {
        assert_reads(&dir.join(format!("state-{i}.safetensors")));
    }
}
