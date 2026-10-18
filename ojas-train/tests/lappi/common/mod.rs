//! Shared support for the moved `qd-train` oracle tests: the crate directory, the trainer
//! oracle's loader and the `float.hex` readers. [`pins`] checks a fixture set against its
//! sha256 manifest.

#![allow(dead_code)]

pub mod pins;

use std::path::PathBuf;

use ojas_io::json::{JsonValue, parse_json};
use ojas_train::lappi::pyjson::float_fromhex;

/// `ojas-train`.
pub fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `tools/qd_train_oracle_trainer.py --out` this file.
pub fn trainer_oracle() -> JsonValue {
    let path = crate_dir()
        .join("tests")
        .join("lappi")
        .join("fixtures")
        .join("trainer-oracle.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    parse_json(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub fn fhex(v: &JsonValue) -> f64 {
    float_fromhex(v.as_str().expect("a float.hex string")).expect("parse float.hex")
}

pub fn fhex_list(v: &JsonValue) -> Vec<f64> {
    v.as_array().expect("a list").iter().map(fhex).collect()
}
