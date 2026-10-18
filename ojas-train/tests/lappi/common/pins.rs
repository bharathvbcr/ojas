//! Fixture pins, checked by the fixtures' own consumers before the bytes are used.
//!
//! The human's decision of 2026-10-02 ("allow pinned fixtures only") lets a tracked binary
//! through `crates/qd-runtime/tests/tracked_source_is_text.rs` only when its sha256 is recorded
//! by a manifest under its fixture root. That gate asks: *is this digest on record?* This module
//! asks the question a consumer needs answered before it computes anything from a fixture: *is the
//! file at this path the bytes on record?* A fixture changed in place therefore fails here too, in
//! qd-train, and the failure names the file and both digests before any parity number is built on
//! it.
//!
//! Two manifest forms, each a fixture set's whole directory:
//!
//! * [`Form::Sha256Sums`]: `<dir>/SHA256SUMS`, coreutils `shasum -a 256` output. The grammar is
//!   the one defined in `tracked_source_is_text.rs`'s module docs (64 lower-case hex digits, two
//!   spaces or ` *`, a relative path with no `.` or `..` component; `\n` lines, no `\r`, no blank
//!   line, at least one line, no escaped names). It is parsed here as strictly, and additionally a
//!   path may be listed only once. The two parsers are kept separate because sharing one would
//!   need a cross-crate `#[path]` include. A disagreement between them fails one side's test; it
//!   cannot pass quietly.
//! * [`Form::JsonFiles`]: `<dir>/manifest.json`'s `files` map, `{"<relative path>": {"sha256":
//!   "<64 lower-case hex>", "bytes": <n>}}`: L-oracle's form (`tiny-published`,
//!   `adamw-decay-sensitive`). Both fields are required and both are checked.
//!
//! Both forms are checked for **completeness** as well as for each pin. Every regular file under
//! the directory, except the manifest itself, must be listed. A file the manifest does not list
//! could be read without a check, so it fails here and is named: pin it, or delete it if it is not
//! a fixture (a Finder `.DS_Store` would be one). A symlink or any other non-regular entry fails
//! too, because what it points at is not what was pinned.
//!
//! The pins are written by `shasum`, never by this module. The command is in
//! `tracked_source_is_text.rs`'s module docs. Nothing here can bless the bytes on disk.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use sha2::{Digest, Sha256};

/// The coreutils manifest's file name.
pub const SHA256SUMS: &str = "SHA256SUMS";

/// The JSON manifest's file name.
pub const MANIFEST_JSON: &str = "manifest.json";

/// More entries than this under one fixture set is refused, not walked: the largest set today
/// (`span-head`) holds 271 files with its `SHA256SUMS`.
const MAX_SET_ENTRIES: usize = 4096;

/// Which manifest a fixture set carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Form {
    /// `<dir>/SHA256SUMS`.
    Sha256Sums,
    /// `<dir>/manifest.json`, its `files` map.
    JsonFiles,
}

impl Form {
    pub fn manifest_name(self) -> &'static str {
        match self {
            Form::Sha256Sums => SHA256SUMS,
            Form::JsonFiles => MANIFEST_JSON,
        }
    }
}

/// Lower-case hex sha256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Lower-case hex sha256 of the file at `path`; panics naming the path if it cannot be read.
pub fn sha256_file_hex(path: &Path) -> String {
    sha256_hex(&std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// A relative path made only of plain components.
fn plain_relative(path: &str) -> Option<PathBuf> {
    let p = Path::new(path);
    (!path.is_empty()
        && !p.is_absolute()
        && p.components().all(|c| matches!(c, Component::Normal(_))))
    .then(|| p.to_path_buf())
}

/// One file's pin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pin {
    pub sha256: String,
    /// The JSON form records the length; `SHA256SUMS` does not.
    pub bytes: Option<u64>,
}

/// A `SHA256SUMS` body as `path -> pin`, or the first reason it is outside the grammar.
pub fn parse_sha256sums(bytes: &[u8]) -> Result<BTreeMap<PathBuf, Pin>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("not UTF-8: {e}"))?;
    if text.contains('\r') {
        return Err("contains `\\r`; lines are separated by `\\n` alone".into());
    }
    let body = text.strip_suffix('\n').unwrap_or(text);
    if body.is_empty() {
        return Err("has no lines, so it pins nothing".into());
    }
    let mut pins = BTreeMap::new();
    for (i, line) in body.split('\n').enumerate() {
        let n = i + 1;
        if line.is_empty() {
            return Err(format!("line {n} is blank"));
        }
        if line.starts_with('\\') {
            return Err(format!(
                "line {n} is coreutils' escaped-name form, which is not accepted"
            ));
        }
        let (digest, rest) = line
            .split_at_checked(64)
            .ok_or_else(|| format!("line {n} is shorter than a 64-digit digest: {line:?}"))?;
        if !is_sha256_hex(digest) {
            return Err(format!(
                "line {n}: {digest:?} is not 64 lower-case hex digits"
            ));
        }
        let path = rest
            .strip_prefix("  ")
            .or_else(|| rest.strip_prefix(" *"))
            .ok_or_else(|| {
                format!("line {n}: the digest is not followed by two spaces or ` *`: {line:?}")
            })?;
        let rel = plain_relative(path).ok_or_else(|| {
            format!("line {n}: path {path:?} must be non-empty, relative, with no `.` or `..`")
        })?;
        let pin = Pin {
            sha256: digest.to_owned(),
            bytes: None,
        };
        if pins.insert(rel, pin).is_some() {
            return Err(format!("line {n}: {path:?} is listed twice"));
        }
    }
    Ok(pins)
}

/// A JSON manifest's `files` map as `path -> pin`, or every reason it is not one.
pub fn parse_files_map(
    manifest: &serde_json::Value,
) -> Result<BTreeMap<PathBuf, Pin>, Vec<String>> {
    let Some(files) = manifest.get("files").and_then(serde_json::Value::as_object) else {
        return Err(vec!["has no `files` object".into()]);
    };
    if files.is_empty() {
        return Err(vec!["`files` is empty, so it pins nothing".into()]);
    }
    let mut pins = BTreeMap::new();
    let mut errors = Vec::new();
    for (path, entry) in files {
        let Some(rel) = plain_relative(path) else {
            errors.push(format!("files key {path:?} is not a plain relative path"));
            continue;
        };
        let sha256 = entry.get("sha256").and_then(serde_json::Value::as_str);
        let bytes = entry.get("bytes").and_then(serde_json::Value::as_u64);
        match (sha256, bytes) {
            (Some(s), Some(b)) if is_sha256_hex(s) => {
                pins.insert(rel, Pin { sha256: s.to_owned(), bytes: Some(b) });
            }
            _ => errors.push(format!(
                "files.{path:?} needs `sha256` (64 lower-case hex digits) and `bytes` (an integer), got {entry}"
            )),
        }
    }
    if errors.is_empty() {
        Ok(pins)
    } else {
        Err(errors)
    }
}

/// Every regular file under `dir`, relative to it. A symlink or other non-regular entry, an
/// unreadable directory, or more than [`MAX_SET_ENTRIES`] entries is an error.
pub fn files_under(dir: &Path) -> Result<BTreeSet<PathBuf>, Vec<String>> {
    let mut out = BTreeSet::new();
    let mut errors = Vec::new();
    let mut seen = 0usize;
    let mut stack = vec![PathBuf::new()];
    while let Some(rel) = stack.pop() {
        let entries = match std::fs::read_dir(dir.join(&rel)) {
            Ok(e) => e,
            Err(e) => {
                errors.push(format!("{}: cannot list: {e}", dir.join(&rel).display()));
                continue;
            }
        };
        for entry in entries {
            seen += 1;
            if seen > MAX_SET_ENTRIES {
                return Err(vec![format!(
                    "{} holds more than {MAX_SET_ENTRIES} entries; refused rather than walked",
                    dir.display()
                )]);
            }
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    errors.push(format!(
                        "{}: cannot read an entry: {e}",
                        dir.join(&rel).display()
                    ));
                    continue;
                }
            };
            let child = rel.join(entry.file_name());
            match entry.file_type() {
                Ok(t) if t.is_dir() => stack.push(child),
                Ok(t) if t.is_file() => {
                    out.insert(child);
                }
                Ok(_) => errors.push(format!(
                    "{}: not a regular file or directory (a symlink?); a pin covers bytes, not a link",
                    child.display()
                )),
                Err(e) => errors.push(format!("{}: {e}", child.display())),
            }
        }
    }
    if errors.is_empty() {
        Ok(out)
    } else {
        Err(errors)
    }
}

/// Every pin against the file it names, and every file under `dir` but `manifest` against the
/// pins. Returns how many files were verified, or every failure, sorted.
fn check_pins(
    dir: &Path,
    manifest: &str,
    pins: &BTreeMap<PathBuf, Pin>,
) -> Result<usize, Vec<String>> {
    let mut errors = Vec::new();
    let on_disk = files_under(dir)?;
    for (rel, pin) in pins {
        let path = dir.join(rel);
        if !on_disk.contains(rel) {
            errors.push(format!(
                "{}: listed in {manifest} but not a file under the set",
                rel.display()
            ));
            continue;
        }
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                errors.push(format!("{}: cannot read: {e}", rel.display()));
                continue;
            }
        };
        if let Some(want) = pin.bytes
            && bytes.len() as u64 != want
        {
            errors.push(format!(
                "{}: {} bytes, {manifest} pins {want}",
                rel.display(),
                bytes.len()
            ));
        }
        let got = sha256_hex(&bytes);
        if got != pin.sha256 {
            errors.push(format!(
                "{}: sha256 is {got}, {manifest} pins {}",
                rel.display(),
                pin.sha256
            ));
        }
    }
    for rel in &on_disk {
        if rel.as_os_str() != manifest && !pins.contains_key(rel) {
            errors.push(format!(
                "{}: under the set but not in {manifest}. Pin it (the `shasum` command is in \
                 crates/qd-runtime/tests/tracked_source_is_text.rs), or remove it if it is not a fixture",
                rel.display()
            ));
        }
    }
    if errors.is_empty() {
        Ok(pins.len())
    } else {
        errors.sort();
        Err(errors)
    }
}

/// Verify the fixture set at `dir` against its manifest. Returns how many files it pins and
/// verified, or every failure.
pub fn check(dir: &Path, form: Form) -> Result<usize, Vec<String>> {
    let manifest = form.manifest_name();
    let raw = std::fs::read(dir.join(manifest))
        .map_err(|e| vec![format!("{}: cannot read {manifest}: {e}", dir.display())])?;
    let pins = match form {
        Form::Sha256Sums => parse_sha256sums(&raw).map_err(|e| vec![format!("{manifest} {e}")])?,
        Form::JsonFiles => {
            let value: serde_json::Value = serde_json::from_slice(&raw)
                .map_err(|e| vec![format!("{manifest} does not parse: {e}")])?;
            parse_files_map(&value).map_err(|es| {
                es.into_iter()
                    .map(|e| format!("{manifest} {e}"))
                    .collect::<Vec<_>>()
            })?
        }
    };
    check_pins(dir, manifest, &pins)
}

/// `dir`, after its fixture set has verified against its manifest in this process; panics naming
/// every failure otherwise. Each set is checked once per test binary: only a success is cached, so
/// every test that reaches a failing set fails with the reason rather than with a poisoned lock.
pub fn verified(dir: &Path, form: Form) -> PathBuf {
    static DONE: OnceLock<Mutex<BTreeSet<(PathBuf, Form)>>> = OnceLock::new();
    let done = DONE.get_or_init(|| Mutex::new(BTreeSet::new()));
    let key = (dir.to_path_buf(), form);
    if done.lock().expect("pin cache lock").contains(&key) {
        return key.0;
    }
    match check(dir, form) {
        Ok(_) => {
            done.lock().expect("pin cache lock").insert(key.clone());
            key.0
        }
        Err(errors) => panic!(
            "fixture set {} does not match its {}; its bytes are not the ones on record, so nothing \
             is computed from them:\n  {}",
            dir.display(),
            form.manifest_name(),
            errors.join("\n  ")
        ),
    }
}
