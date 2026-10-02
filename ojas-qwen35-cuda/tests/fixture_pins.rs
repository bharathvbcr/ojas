//! The two fixtures L-cuda-M1 copied in are byte-identical to their sources
//! and to their pins, and nothing unpinned rides along.
//!
//! - `tests/fixtures/adamw_decay_sensitive/`: L-oracle's decay-sensitive
//!   AdamW golden (Lappi `crates/qd-train/tests/fixtures/adamw-decay-sensitive/`,
//!   merged at Lappi 14601bc). Pinned by its own `manifest.json`, whose sha256
//!   is [`DECAY_MANIFEST_SHA256`]. K11's device golden; `runga` embeds it.
//! - `tests/fixtures/qwen35_train_published/`: tessl's tiny Qwen3.5 training
//!   fixture (`tessl/tests/fixtures/qwen35_train/`), pinned by
//!   `qwen35_train_published_SHA256SUMS`. The tiny-fixture loader reads it.
//!
//! The comparison with the live source fails loudly when the source checkout
//! is absent: a pin test that silently skips is a pin test that passed
//! unexamined.

mod reference;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use ojas_qwen35_cuda::k11_golden::DECAY_MANIFEST_SHA256;
use reference::sha256;

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn files_in(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| {
            let e = e.unwrap_or_else(|err| panic!("{}: {err}", dir.display()));
            assert!(
                e.file_type().map(|t| t.is_file()).unwrap_or(false),
                "{}: not a regular file",
                e.path().display()
            );
            e.file_name().into_string().expect("UTF-8 file name")
        })
        .collect()
}

fn hex_of(path: &Path) -> String {
    sha256::file_hex(path).unwrap_or_else(|e| panic!("{e}"))
}

/// `(file, sha256)` pairs from the manifest's `"files"` object.
fn manifest_pins(manifest: &Path) -> Vec<(String, String)> {
    let text = std::fs::read_to_string(manifest).expect("manifest.json");
    let v = ojas_io::parse_json(&text).unwrap_or_else(|e| panic!("manifest.json: {e}"));
    v.get_object("files")
        .and_then(|f| f.as_object())
        .unwrap_or_else(|e| panic!("manifest.json files: {e}"))
        .iter()
        .map(|(k, x)| {
            let sha = x
                .get_str("sha256")
                .unwrap_or_else(|e| panic!("manifest.json {k}: {e}"));
            (k.clone(), sha.to_string())
        })
        .collect()
}

#[test]
fn the_decay_sensitive_copy_equals_its_pins() {
    let dir = crate_dir().join("tests/fixtures/adamw_decay_sensitive");
    assert_eq!(
        hex_of(&dir.join("manifest.json")),
        DECAY_MANIFEST_SHA256,
        "manifest.json is not L-oracle's"
    );
    let pins = manifest_pins(&dir.join("manifest.json"));
    let mut want: BTreeSet<String> = pins.iter().map(|(f, _)| f.clone()).collect();
    want.insert("manifest.json".to_string());
    assert_eq!(
        files_in(&dir),
        want,
        "the copy holds files the manifest does not pin, or lacks one"
    );
    assert_eq!(pins.len(), 4, "golden, inputs, preregistration, table");
    for (file, sha) in &pins {
        assert_eq!(
            &hex_of(&dir.join(file)),
            sha,
            "{file}: differs from its pin"
        );
    }
}

#[test]
fn the_decay_sensitive_copy_equals_lappis_live_files() {
    let ours = crate_dir().join("tests/fixtures/adamw_decay_sensitive");
    let live = crate_dir()
        .join("../../Lappi-decision/crates/qd-train/tests/fixtures/adamw-decay-sensitive");
    assert!(
        live.is_dir(),
        "{}: Lappi's checkout is absent, so the copy cannot be compared with its source. \
         Run this test where Lappi-decision sits beside ojas.",
        live.display()
    );
    assert_eq!(files_in(&ours), files_in(&live), "file sets differ");
    for f in files_in(&ours) {
        let (a, b) = (
            std::fs::read(ours.join(&f)).expect("ours"),
            std::fs::read(live.join(&f)).expect("live"),
        );
        assert!(a == b, "{f}: the copy differs from Lappi's live file");
    }
}

fn train_sums() -> Vec<(String, String)> {
    let sums = crate_dir().join("tests/fixtures/qwen35_train_published_SHA256SUMS");
    std::fs::read_to_string(&sums)
        .unwrap_or_else(|e| panic!("{}: {e}", sums.display()))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (sha, file) = l
                .split_once("  ")
                .unwrap_or_else(|| panic!("malformed SHA256SUMS line {l:?}"));
            (file.to_string(), sha.to_string())
        })
        .collect()
}

#[test]
fn the_tiny_fixture_copy_equals_its_sums() {
    let dir = crate_dir().join("tests/fixtures/qwen35_train_published");
    let sums = train_sums();
    assert_eq!(sums.len(), 31, "config, 27 gradients, ids, loss, model");
    let listed: BTreeSet<String> = sums.iter().map(|(f, _)| f.clone()).collect();
    assert_eq!(listed.len(), sums.len(), "a file is listed twice");
    assert_eq!(files_in(&dir), listed, "unpinned or missing files");
    for (file, sha) in &sums {
        assert_eq!(
            &hex_of(&dir.join(file)),
            sha,
            "{file}: differs from its sum"
        );
    }
}

#[test]
fn the_tiny_fixture_copy_equals_tessls_live_fixture() {
    let ours = crate_dir().join("tests/fixtures/qwen35_train_published");
    let live = crate_dir().join("../../tessl/tests/fixtures/qwen35_train");
    assert!(
        live.is_dir(),
        "{}: tessl's checkout is absent, so the copy cannot be compared with its source",
        live.display()
    );
    assert_eq!(files_in(&ours), files_in(&live), "file sets differ");
    for f in files_in(&ours) {
        let (a, b) = (
            std::fs::read(ours.join(&f)).expect("ours"),
            std::fs::read(live.join(&f)).expect("live"),
        );
        assert!(a == b, "{f}: the copy differs from tessl's live file");
    }
}
