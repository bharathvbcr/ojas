//! `Qwen35TextConfig::tape_spec` against this crate's own name map: the
//! hybrid graph on ojas's Tape (`ojas_model::qwen35`) reads exactly the
//! tensors, under exactly the names and shapes, that the provider's
//! `tower_tensors` lists, for the real 2B's config and for tessl's tiny
//! fixture. CPU only.

#![cfg(target_os = "macos")]

use ojas_model::qwen35::{hf_tensors, Qwen35Mixer};
use ojas_qwen35::{tower_tensors, Qwen35TextConfig};

const REAL: &str = include_str!("fixtures/qwen35_2b_base_config.json");

fn tiny() -> Qwen35TextConfig {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tessl/tests/fixtures/qwen35_train/config.json");
    Qwen35TextConfig::from_file(&path).unwrap()
}

fn same_tensors(cfg: &Qwen35TextConfig) {
    let spec = cfg.tape_spec();
    spec.validate().unwrap();
    // `tower_tensors` names are relative to `tower_prefix`.
    let mut tape: Vec<(String, Vec<usize>)> = hf_tensors(&spec, "")
        .into_iter()
        .map(|t| (t.name, t.shape))
        .collect();
    let mut provider: Vec<(String, Vec<usize>)> = tower_tensors(cfg)
        .into_iter()
        .map(|t| (t.name, t.shape))
        .collect();
    tape.sort();
    provider.sort();
    assert_eq!(tape, provider);
}

#[test]
fn the_tape_reads_the_providers_tensors_for_the_real_2b() {
    let cfg = Qwen35TextConfig::from_json(REAL).unwrap();
    same_tensors(&cfg);
    let spec = cfg.tape_spec();
    assert_eq!(spec.layers.len(), 24);
    assert_eq!(
        spec.layers
            .iter()
            .filter(|k| **k == Qwen35Mixer::Attention)
            .count(),
        6
    );
    assert_eq!(
        (spec.hidden, spec.vocab, spec.head_dim),
        (2048, 248320, 256)
    );
    assert_eq!((spec.q_heads, spec.kv_heads, spec.rotary_dim), (8, 2, 64));
    assert_eq!(
        (spec.gdn_heads, spec.gdn_key_dim, spec.gdn_value_dim),
        (16, 128, 128)
    );
    assert_eq!(spec.rope_theta, 1e7);
}

#[test]
fn the_tape_reads_the_providers_tensors_for_the_tiny_fixture() {
    same_tensors(&tiny());
}
