//! D5 pin: ojas-qwen35 is the host-neutral half of the Qwen3.5 route and holds
//! no tessl whole-step shell. The Tape tower is fed by the config, so that
//! path must keep working without the shell.
//!
//! Fails on the tree before the retirement: `src/step.rs` imported
//! `tessl::qwen35_train` and `tessl::qwen35_adamw`, and `Qwen35Step` was
//! exported.
#![cfg(target_os = "macos")]

use std::fs;
use std::path::Path;

use ojas_qwen35::Qwen35TextConfig;

/// Every symbol of tessl's retired Qwen3.5 training layer. A source file of
/// this crate naming one means the shell is back (or half-removed).
const RETIRED_TESSL_PATHS: [&str; 6] = [
    "qwen35_train::",
    "qwen35_adamw::",
    "Qwen35Grads",
    "PendingStep",
    "tessl_qwen35_train",
    "tessl_qwen35_adamw",
];

#[test]
fn no_source_file_reaches_tessls_retired_training_layer() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut seen = 0usize;
    for entry in fs::read_dir(&src).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let text = fs::read_to_string(&path).unwrap();
        // Doc comments may name the retired layer to say it is gone.
        for (n, line) in text.lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            for needle in RETIRED_TESSL_PATHS {
                assert!(
                    !code.contains(needle),
                    "{}:{} names {needle}: the tessl step shell was retired (D5)",
                    path.display(),
                    n + 1
                );
            }
        }
        seen += 1;
    }
    assert!(seen >= 5, "expected the crate's modules, found {seen}");
}

#[test]
fn the_step_shell_files_are_gone() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    assert!(!src.join("step.rs").exists(), "src/step.rs is retired");
}

#[test]
fn the_tape_spec_still_builds_from_the_tiny_fixture_config() {
    let cfg = Qwen35TextConfig::from_file(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tessl/tests/fixtures/qwen35_train/config.json"),
    )
    .unwrap();
    cfg.tape_spec().validate().unwrap();
}
