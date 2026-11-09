//! Chain printing (card ft-aa3f1f18, lane `ioadv`, attack 5).
//!
//! `IoError`'s Display already contains the `std::io::Error` text, and `source()` returns
//! that same error. A generic chain walker (`anyhow`-style `{:#}`, `eyre`, a loop over
//! `source()`) therefore prints the io text twice. This test states the property a walker
//! would want, and pins the CURRENT behaviour (the text appears twice), which confirms the
//! duplication. It is the specified consequence of "Display must stay byte-identical" plus
//! `source()`, not a defect of one site. No in-repo or tessl code walks an IoError chain
//! (the only `source()` walker, ojas-train `TrainError::source`, never holds an IoError).
//! If a later change makes the walk print the text once, update this test.

use std::error::Error;
use std::fmt::Write as _;

use ojas_io::open_regular_file;

fn walk(e: &(dyn Error + 'static)) -> String {
    let mut out = e.to_string();
    let mut cur = e.source();
    while let Some(s) = cur {
        let _ = write!(out, ": {s}");
        cur = s.source();
    }
    out
}

#[test]
fn chain_walk_repeats_the_io_text_once_per_layer() {
    let p = std::env::temp_dir().join(format!("ojas-io-chain-{}-missing", std::process::id()));
    let e = open_regular_file(&p).unwrap_err();
    let raw = std::fs::metadata(&p).unwrap_err().to_string();
    let printed = walk(&e);
    assert_eq!(
        printed.matches(&raw).count(),
        2,
        "io text `{raw}` should appear in Display and again from source(): {printed}"
    );
}
