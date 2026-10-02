//! `cargo fmt` must not be able to reach a file outside this crate.
//!
//! rustfmt formats every out-of-line module it can resolve, and a `#[path]`
//! attribute can point a module anywhere on disk. This crate used to compile
//! tessl's `tests/common/gdn_train.rs` through `#[path = "../../../tessl/…"]`,
//! so `cargo fmt` here would have rewritten a tessl file (rule 6;
//! `GAP-L-CUDA-M0-CARGO-FMT-WOULD-EDIT-TESSL-2026-10-01`). The include is now an
//! in-crate copy (`tests/reference_gdn_published_vs_tessl.rs`).
//!
//! This test keeps the whole class closed rather than that one case: it walks
//! every `.rs` file under `src/` and `tests/`, resolves every `#[path = "…"]`
//! against the directory of the file that declares it (the Rust reference's
//! rule for modules outside inline blocks), follows symlinks, and fails if
//! any target lies outside this crate's directory. `include!`,
//! `include_str!` and `include_bytes!` are not checked: rustfmt does not
//! follow them.

use std::path::{Path, PathBuf};

/// The walk refuses a tree larger than this, rather than running unbounded.
const MAX_FILES: usize = 5_000;

/// Every `#[path = "…"]` value in `text`, skipping `//` comment lines.
fn path_attributes(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        let mut rest = trimmed;
        while let Some(at) = rest.find("#[path") {
            let after = rest[at + "#[path".len()..].trim_start();
            rest = &rest[at + "#[path".len()..];
            let Some(after_eq) = after.strip_prefix('=') else {
                continue;
            };
            let Some(quoted) = after_eq.trim_start().strip_prefix('"') else {
                continue;
            };
            if let Some(end) = quoted.find('"') {
                out.push(quoted[..end].to_string());
            }
        }
    }
    out
}

/// Every `.rs` file under `root`, bounded by [`MAX_FILES`].
fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("cannot list {}: {e}", dir.display()));
        for entry in entries {
            let entry =
                entry.unwrap_or_else(|e| panic!("cannot read an entry of {}: {e}", dir.display()));
            let path = entry.path();
            let kind = entry
                .file_type()
                .unwrap_or_else(|e| panic!("cannot stat {}: {e}", path.display()));
            if kind.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|x| x == "rs") {
                out.push(path);
            }
            assert!(
                out.len() <= MAX_FILES,
                "more than {MAX_FILES} .rs files under {}: refusing an unbounded walk",
                root.display()
            );
        }
    }
    out.sort();
    out
}

/// `path` with `.` and `..` applied lexically (no symlinks followed).
fn lexical(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// `(declaring file, attribute value, resolved target)` for every `#[path]`
/// under `src/` and `tests/` whose target is not inside `crate_dir`.
fn escaping_paths(crate_dir: &Path) -> (usize, Vec<(PathBuf, String, PathBuf)>) {
    let crate_dir = crate_dir
        .canonicalize()
        .unwrap_or_else(|e| panic!("cannot canonicalize {}: {e}", crate_dir.display()));
    let mut seen = 0usize;
    let mut bad = Vec::new();
    for top in ["src", "tests"] {
        for file in rust_files(&crate_dir.join(top)) {
            let text = std::fs::read_to_string(&file)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", file.display()));
            let dir = file.parent().expect("a file has a parent directory");
            for value in path_attributes(&text) {
                seen += 1;
                let joined = dir.join(&value);
                match joined.canonicalize() {
                    Ok(resolved) if resolved.starts_with(&crate_dir) => {}
                    Ok(resolved) => bad.push((file.clone(), value, resolved)),
                    // A target that does not resolve counts as outside: nothing
                    // proves where it points. (`Path::starts_with` on the
                    // unresolved join would compare `..` components literally
                    // and call `crate/tests/../../../x` inside.)
                    Err(_) => bad.push((file.clone(), value, lexical(&joined))),
                }
            }
        }
    }
    (seen, bad)
}

#[test]
fn no_path_attribute_in_this_crate_leaves_the_crate() {
    let (seen, bad) = escaping_paths(Path::new(env!("CARGO_MANIFEST_DIR")));
    for (file, value, resolved) in &bad {
        eprintln!(
            "{}: #[path = {value:?}] resolves to {}, outside the crate",
            file.display(),
            resolved.display()
        );
    }
    assert!(
        bad.is_empty(),
        "{} of {seen} #[path] attributes leave the crate: cargo fmt would format files outside it",
        bad.len()
    );
    // The crate does use #[path] (the tessl copy); finding none means the scan
    // looked at nothing, which must not pass as clean.
    assert!(
        seen > 0,
        "no #[path] attribute found: the scan did not run over the crate"
    );
    eprintln!("{seen} #[path] attributes, all inside the crate");
}

#[test]
fn the_scanner_reads_path_attributes_the_way_rustc_writes_them() {
    // Written as `PATH` and lowered at run time, so this file's own scan does
    // not read these sample attributes as real ones.
    let text = r#"
#[PATH = "a.rs"]
mod a;
#[PATH="b/c.rs"] mod c;
    #[allow(dead_code)] #[PATH = "../../../tessl/tests/common/gdn_train.rs"] mod d;
// #[PATH = "commented.rs"]
//! doc text naming `#[PATH]` without a value
#[PATHological = "x"]
"#
    .replace("PATH", "path");
    assert_eq!(
        path_attributes(&text),
        vec![
            "a.rs".to_string(),
            "b/c.rs".to_string(),
            "../../../tessl/tests/common/gdn_train.rs".to_string()
        ]
    );
}

#[test]
fn the_pre_fix_tessl_include_is_flagged() {
    // The exact attribute this crate carried before the fix, declared from
    // tests/: it must resolve outside the crate.
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .canonicalize()
        .expect("crate dir");
    let declared_in = crate_dir.join("tests");
    let target = declared_in.join("../../../tessl/tests/common/gdn_train.rs");
    // Lexically, so the answer does not depend on tessl being checked out.
    let resolved = lexical(&target);
    assert!(!resolved.starts_with(&crate_dir), "{}", resolved.display());
    // The trap the scan avoids: the unresolved join "starts with" the crate.
    assert!(target.starts_with(&crate_dir));
}

#[test]
fn an_unresolvable_or_escaping_path_is_reported() {
    // <scratch>/crate is the crate; <scratch>/outside.rs exists beside it.
    let scratch = std::env::temp_dir().join(format!("qd-fmt-boundary-{}", std::process::id()));
    let dir = scratch.join("crate");
    let tests = dir.join("tests");
    std::fs::create_dir_all(dir.join("src")).expect("src");
    std::fs::create_dir_all(&tests).expect("tests");
    std::fs::write(
        dir.join("src/lib.rs"),
        "#[path = \"inner.rs\"]\nmod inner;\n",
    )
    .expect("lib");
    std::fs::write(dir.join("src/inner.rs"), "").expect("inner");
    std::fs::write(scratch.join("outside.rs"), "").expect("outside");
    std::fs::write(
        tests.join("t.rs"),
        "#[path = \"../../outside.rs\"]\nmod a;\n#[path = \"../../missing/x.rs\"]\nmod b;\n",
    )
    .expect("t");
    let (seen, bad) = escaping_paths(&dir);
    std::fs::remove_dir_all(&scratch).expect("cleanup");
    assert_eq!(seen, 3);
    let flagged: Vec<&str> = bad.iter().map(|(_, v, _)| v.as_str()).collect();
    assert_eq!(flagged, vec!["../../outside.rs", "../../missing/x.rs"]);
}
