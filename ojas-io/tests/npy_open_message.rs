//! `read_npy` names the path once when the open fails: `IoError`'s Display is already
//! `"<path>: <detail>"`, so the reader must not prefix the path again.

use std::path::PathBuf;

use ojas_io::npy::{read_npy, read_npy_with_limit};

fn scratch_dir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("ojas-io-npy-open-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn missing_npy_names_the_path_once() {
    let dir = scratch_dir("missing");
    let path = dir.join("absent.npy");
    let os_text = std::fs::metadata(&path).unwrap_err().to_string();
    let want = format!("{}: {os_text}", path.display());

    for (name, got) in [
        ("read_npy", read_npy(&path).unwrap_err()),
        (
            "read_npy_with_limit",
            read_npy_with_limit(&path, Some(1 << 20)).unwrap_err(),
        ),
    ] {
        assert_eq!(got, want, "{name}");
        assert_eq!(got.matches("absent.npy").count(), 1, "{name}: {got}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn directory_npy_names_the_path_once() {
    let dir = scratch_dir("dir");
    let path = dir.join("is_a_dir.npy");
    std::fs::create_dir_all(&path).unwrap();
    let got = read_npy(&path).unwrap_err();
    assert_eq!(got, format!("{}: not a regular file", path.display()));
    assert_eq!(got.matches("is_a_dir.npy").count(), 1, "{got}");
    let _ = std::fs::remove_dir_all(&dir);
}
