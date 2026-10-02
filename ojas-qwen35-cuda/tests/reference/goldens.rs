//! The float64 torch goldens in `tests/fixtures/goldens/`, written by
//! `tests/fixtures/gen_goldens.py`, and the checks that pin them: every file's
//! sha256 against `manifest.json`, nothing on disk that the manifest does not
//! list, and the manifest's record of the generator's own sha256 against the
//! generator on disk (so editing the generator without regenerating fails).

use std::path::{Path, PathBuf};

use super::{npy, sha256};

pub fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/goldens")
}

pub fn generator() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen_goldens.py")
}

pub fn manifest_text() -> String {
    std::fs::read_to_string(dir().join("manifest.json"))
        .expect("tests/fixtures/goldens/manifest.json")
}

/// The manifest's `"files"` object as `(file name, sha256 hex)` pairs. The
/// generator writes it with `json.dumps(indent=1, sort_keys=True)`; a manifest
/// in any other shape is refused rather than half-read.
pub fn manifest_files(text: &str) -> Vec<(String, String)> {
    let start = text
        .find("\"files\": {")
        .expect("manifest has no \"files\" object")
        + "\"files\": {".len();
    let body = &text[start
        ..start
            + text[start..]
                .find('}')
                .expect("unterminated \"files\" object")];
    let mut out = Vec::new();
    for entry in body.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        let (k, v) = entry
            .split_once(':')
            .unwrap_or_else(|| panic!("malformed manifest entry {entry:?}"));
        let unq = |s: &str| {
            s.trim()
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .unwrap_or_else(|| panic!("unquoted manifest token {s:?}"))
                .to_string()
        };
        let (name, hex) = (unq(k), unq(v));
        assert!(
            hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()),
            "manifest sha256 for {name} is not 64 hex digits: {hex:?}"
        );
        out.push((name, hex));
    }
    assert!(
        !out.is_empty(),
        "the manifest lists no files; a golden suite must not pass vacuously"
    );
    out
}

/// The manifest's string value for a top-level `key`, e.g. `generator_sha256`.
pub fn manifest_string(text: &str, key: &str) -> String {
    let needle = format!("\"{key}\": \"");
    let at = text
        .find(&needle)
        .unwrap_or_else(|| panic!("manifest has no string {key:?}"))
        + needle.len();
    text[at..at + text[at..].find('"').expect("unterminated string")].to_string()
}

/// Verifies every pin. Returns how many files were checked.
pub fn verify() -> usize {
    let text = manifest_text();
    let files = manifest_files(&text);
    for (name, want) in &files {
        let got = sha256::file_hex(&dir().join(name)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            &got, want,
            "{name}: sha256 differs from manifest.json (regenerated without updating it?)"
        );
    }
    let mut on_disk: Vec<String> = std::fs::read_dir(dir())
        .expect("goldens dir")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .filter(|n| n != "manifest.json")
        .collect();
    on_disk.sort();
    let mut listed: Vec<String> = files.iter().map(|(n, _)| n.clone()).collect();
    listed.sort();
    assert_eq!(
        on_disk, listed,
        "the goldens directory holds files the manifest does not pin, or lacks pinned ones"
    );
    let gen_sha = sha256::file_hex(&generator()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        gen_sha,
        manifest_string(&text, "generator_sha256"),
        "gen_goldens.py changed since the goldens were generated; regenerate them"
    );
    files.len()
}

pub fn load(name: &str) -> npy::Npy {
    npy::read(&dir().join(format!("{name}.npy"))).unwrap_or_else(|e| panic!("golden {name}: {e}"))
}

/// A `<f8` golden. An `<f4` file is refused: these goldens are float64 by
/// construction, and a narrowed one would be a silently weaker golden.
pub fn f64s(name: &str) -> Vec<f64> {
    load(name)
        .f64s()
        .unwrap_or_else(|e| panic!("golden {name}: {e}"))
        .to_vec()
}

/// `(shape, data)` of a `<f8` golden.
pub fn f64s_shaped(name: &str) -> (Vec<usize>, Vec<f64>) {
    let a = load(name);
    let d = a
        .f64s()
        .unwrap_or_else(|e| panic!("golden {name}: {e}"))
        .to_vec();
    (a.shape, d)
}

/// An `<i8` golden of non-negative indices.
pub fn indices(name: &str) -> Vec<usize> {
    load(name)
        .i64s()
        .unwrap_or_else(|e| panic!("golden {name}: {e}"))
        .iter()
        .map(|&x| {
            usize::try_from(x).unwrap_or_else(|_| panic!("golden {name}: negative index {x}"))
        })
        .collect()
}

/// `max |got - want| / max |want|`, asserted `<= bound`; an all-zero golden
/// demands exact zeros. Prints the measurement either way.
pub fn assert_rel(label: &str, got: &[f64], want: &[f64], bound: f64) -> f64 {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length {} vs golden {}",
        got.len(),
        want.len()
    );
    let mag = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let mut worst = (0.0f64, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{label}[{i}]: non-finite {g}");
        let e = (g - w).abs();
        if e > worst.0 {
            worst = (e, i);
        }
    }
    if mag == 0.0 {
        assert!(
            worst.0 == 0.0,
            "{label}: golden is all zeros, reference is not (max {:.3e})",
            worst.0
        );
        eprintln!("{label}: golden all zeros, reference exact zeros");
        return 0.0;
    }
    let rel = worst.0 / mag;
    eprintln!("{label}: {rel:.3e} of max|golden| {mag:.4e} (bound {bound:.0e})");
    assert!(
        rel <= bound,
        "{label}: max err {:.3e} at {} (got {} want {}), {rel:.3e} of max|golden|, over {bound:.0e}",
        worst.0,
        worst.1,
        got[worst.1],
        want[worst.1]
    );
    rel
}
