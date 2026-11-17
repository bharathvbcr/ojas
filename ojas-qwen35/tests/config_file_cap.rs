//! `Qwen35TextConfig::from_file` bounds the read before the file is a String.
//!
//! `from_file` was `std::fs::read_to_string` followed by `from_json`, so the
//! 16 MiB JSON cap ran on text already resident: a multi-gigabyte
//! `config.json` was allocated whole before anything could refuse it. Every
//! test here uses a temp file the test creates and removes; the over-cap case
//! is one byte over.
//!
//! The crate root is `cfg(target_os = "macos")`, so this file follows it.

#![cfg(target_os = "macos")]

use ojas_qwen35::{Qwen35Error, Qwen35TextConfig};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::{Path, PathBuf};

const REAL: &str = include_str!("fixtures/qwen35_2b_base_config.json");
const CAP: usize = 16 << 20;

thread_local! {
    /// Largest single allocation made on this thread while armed.
    static LARGEST: Cell<usize> = const { Cell::new(0) };
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        note(l.size());
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        note(new);
        System.realloc(p, l, new)
    }
}

fn note(n: usize) {
    // `try_with`: the slots may be gone while a thread is being torn down.
    let _ = ARMED.try_with(|a| {
        if a.get() {
            let _ = LARGEST.try_with(|m| m.set(m.get().max(n)));
        }
    });
}

#[global_allocator]
static A: Counting = Counting;

struct Tmp(PathBuf);

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tmp(tag: &str) -> Tmp {
    Tmp(std::env::temp_dir().join(format!("ojas-qwen35-cfgcap-{}-{tag}", std::process::id())))
}

/// The real config padded with trailing whitespace to `len` bytes.
fn padded(len: usize) -> String {
    assert!(REAL.len() <= len);
    format!("{REAL}{}", " ".repeat(len - REAL.len()))
}

fn config_error(path: &Path) -> String {
    match Qwen35TextConfig::from_file(path) {
        Err(Qwen35Error::Config(d)) => d,
        Err(other) => panic!("expected Qwen35Error::Config, got {other:?}"),
        Ok(_) => panic!("accepted"),
    }
}

fn io_error(path: &Path) -> String {
    match Qwen35TextConfig::from_file(path) {
        Err(Qwen35Error::Io(d)) => d,
        Err(other) => panic!("expected Qwen35Error::Io, got {other:?}"),
        Ok(_) => panic!("accepted"),
    }
}

#[test]
fn a_file_one_byte_over_the_cap_is_refused_without_allocating_it() {
    let t = tmp("over");
    std::fs::write(&t.0, padded(CAP + 1)).unwrap();
    LARGEST.with(|m| m.set(0));
    ARMED.with(|a| a.set(true));
    let result = Qwen35TextConfig::from_file(&t.0);
    ARMED.with(|a| a.set(false));
    let largest = LARGEST.with(|m| m.get());
    let Err(Qwen35Error::Config(d)) = result else {
        panic!("expected a Config refusal, got {result:?}");
    };
    assert!(
        d.contains(&t.0.display().to_string())
            && d.contains(&format!("{} bytes", CAP + 1))
            && d.contains(&format!("{CAP}-byte")),
        "{d}"
    );
    assert!(
        largest < CAP / 2,
        "allocated {largest} bytes before refusing"
    );
}

#[test]
fn a_file_of_exactly_the_cap_reaches_the_parser() {
    let t = tmp("at");
    std::fs::write(&t.0, padded(CAP)).unwrap();
    let from_file = Qwen35TextConfig::from_file(&t.0).unwrap();
    let from_text = Qwen35TextConfig::from_json(REAL).unwrap();
    assert_eq!(from_file.vocab, from_text.vocab);
}

#[test]
fn from_json_over_the_cap_is_unchanged() {
    let err = Qwen35TextConfig::from_json(&padded(CAP + 1)).unwrap_err();
    let Qwen35Error::Config(d) = err else {
        panic!("expected Config");
    };
    assert!(d.contains("exceeds the 16777216-byte limit"), "{d}");
}

#[test]
fn a_directory_is_refused_as_io() {
    let t = tmp("dir");
    std::fs::create_dir(&t.0).unwrap();
    assert!(io_error(&t.0).contains("not a regular file"));
}

#[test]
fn a_missing_file_stays_io_and_invalid_utf8_is_config() {
    let t = tmp("missing");
    assert!(io_error(&t.0).contains(&t.0.display().to_string()));
    std::fs::write(&t.0, [0xff, 0xfe]).unwrap();
    assert!(config_error(&t.0).contains(&t.0.display().to_string()));
}

#[cfg(unix)]
#[test]
fn a_fifo_is_refused_without_blocking() {
    let t = tmp("fifo");
    let made = std::process::Command::new("/usr/bin/mkfifo")
        .arg(&t.0)
        .status()
        .unwrap();
    assert!(made.success());
    let (tx, rx) = std::sync::mpsc::channel();
    let p = t.0.clone();
    std::thread::spawn(move || {
        let _ = tx.send(Qwen35TextConfig::from_file(&p).map(|_| ()));
    });
    let got = rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("a FIFO blocked from_file");
    match got {
        Err(Qwen35Error::Io(d)) => assert!(d.contains("not a regular file"), "{d}"),
        other => panic!("expected an Io refusal, got {other:?}"),
    }
}
