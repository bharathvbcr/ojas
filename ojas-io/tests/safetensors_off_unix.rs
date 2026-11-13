//! Off unix a safetensors file cannot be stamped with an inode and device, so
//! opening one is refused with `Unsupported` instead of reading without the
//! identity re-check. Compiled only off unix: compile-checked with
//! `--target x86_64-pc-windows-msvc --tests`, runtime unverified.
#![cfg(not(unix))]

use ojas_io::safetensors::SafeTensors;
use std::io::ErrorKind;

#[test]
fn open_with_limit_refuses_with_unsupported() {
    let path = std::env::temp_dir().join(format!("ojas-io-st-off-unix-{}", std::process::id()));
    // A valid empty safetensors file: header length 2, header "{}".
    let mut bytes = 2u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    std::fs::write(&path, bytes).unwrap();
    let got = SafeTensors::open_with_limit(&path, None);
    let _ = std::fs::remove_file(&path);
    let err = got.unwrap_err();
    assert_eq!(err.io_kind(), Some(ErrorKind::Unsupported), "{err}");
}
