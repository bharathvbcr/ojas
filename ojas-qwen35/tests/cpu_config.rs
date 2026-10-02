//! `config.json` parsing and refusal, on the CPU (no GPU runtime is opened).
//!
//! `fixtures/qwen35_2b_base_config.json` is Qwen3.5-2B-Base's `config.json`
//! byte for byte (sha256 ed1c1723...b4, the HF snapshot blob d30a15be; also
//! Lappi's `crates/qd-export/tests/fixtures/qwen35_2b_base_config.json`).
//! The ignored `snapshot_header_matches_the_name_map` re-checks the bytes
//! against the snapshot. Every refusal below mutates that text and first
//! asserts the mutation happened, so no case passes vacuously.

#![cfg(target_os = "macos")]

use ojas_qwen35::{LayerKind, Qwen35Error, Qwen35TextConfig};

const REAL: &str = include_str!("fixtures/qwen35_2b_base_config.json");

fn tiny_config_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tessl/tests/fixtures/qwen35_train/config.json")
}

fn mutate(from: &str, to: &str) -> String {
    assert!(REAL.contains(from), "the fixture lacks {from:?}");
    let out = REAL.replacen(from, to, 1);
    assert_ne!(out, REAL);
    out
}

fn refused(text: &str, needle: &str) {
    match Qwen35TextConfig::from_json(text) {
        Ok(_) => panic!("accepted; expected a refusal containing {needle:?}"),
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains(needle), "{msg:?} lacks {needle:?}");
        }
    }
}

#[test]
fn the_real_2b_config_parses_to_the_2b() {
    let c = Qwen35TextConfig::from_json(REAL).unwrap();
    assert_eq!((c.hidden, c.intermediate, c.vocab), (2048, 6144, 248_320));
    assert_eq!(c.layers.len(), 24);
    for (l, k) in c.layers.iter().enumerate() {
        let want = if l % 4 == 3 {
            LayerKind::FullAttention
        } else {
            LayerKind::LinearAttention
        };
        assert_eq!(*k, want, "layer {l}");
    }
    assert_eq!(
        (
            c.gdn_key_heads,
            c.gdn_value_heads,
            c.gdn_key_dim,
            c.gdn_value_dim,
            c.conv_kernel
        ),
        (16, 16, 128, 128, 4)
    );
    assert_eq!(
        (c.q_heads, c.kv_heads, c.head_dim, c.rotary_dim),
        (8, 2, 256, 64)
    );
    assert_eq!((c.rope_theta, c.rms_norm_eps), (1e7, 1e-6));
    let m = c.mrope.as_ref().expect("the 2B declares MRoPE");
    assert_eq!(m.section, [11, 11, 10]);
    assert_eq!(m.interleaved, Some(true));
    assert_eq!(c.tower_prefix, "model.language_model.");
    assert_eq!(c.max_position_embeddings, Some(262_144));
    let mut reserved = c.reserved_token_ids.clone();
    reserved.sort();
    assert_eq!(
        reserved,
        [
            ("image_token_id".to_string(), 248_056),
            ("video_token_id".to_string(), 248_057),
            ("vision_end_token_id".to_string(), 248_054),
            ("vision_start_token_id".to_string(), 248_053),
        ]
    );
    // Hand count: embedding 248320 x 2048; 18 GDN layers of 58,814,624; 6
    // attention layers of 52,433,408; the final norm's 2048.
    assert_eq!(c.parameter_count(), 1_881_825_088);
    assert!(c
        .canonical_summary()
        .contains("layers=LLLFLLLFLLLFLLLFLLLFLLLF"));
}

#[test]
fn tessls_tiny_fixture_parses_as_a_text_only_config() {
    let c = Qwen35TextConfig::from_file(&tiny_config_path()).unwrap();
    assert_eq!(c.tower_prefix, "model.");
    assert_eq!(
        c.layers,
        [LayerKind::LinearAttention, LayerKind::FullAttention]
    );
    assert_eq!((c.hidden, c.intermediate, c.vocab), (64, 128, 64));
    assert_eq!(
        (c.q_heads, c.kv_heads, c.head_dim, c.rotary_dim),
        (2, 1, 256, 64)
    );
    assert_eq!((c.gdn_key_heads, c.gdn_value_heads), (1, 1));
    assert!(c.mrope.is_none());
    assert!(c.reserved_token_ids.is_empty());
}

#[test]
fn unknown_keys_are_refused_by_name() {
    refused(
        &mutate(
            "\"head_dim\": 256,",
            "\"head_dim\": 256, \"sliding_window\": 4096,",
        ),
        "text_config.sliding_window",
    );
    refused(
        &mutate(
            "\"image_token_id\": 248056,",
            "\"image_token_id\": 248056, \"audio_token_id\": 7,",
        ),
        "config.audio_token_id",
    );
    refused(
        &mutate(
            "\"rope_type\": \"default\",",
            "\"rope_type\": \"default\", \"factor\": 2.0,",
        ),
        "rope_parameters.factor",
    );
    refused(
        &mutate(
            "\"head_dim\": 256,",
            "\"head_dim\": 256, \"num_experts\": 8,",
        ),
        "text_config.num_experts",
    );
}

#[test]
fn shapes_the_training_kernels_do_not_run_are_refused() {
    refused(
        &mutate("\"head_dim\": 256", "\"head_dim\": 128"),
        "head_dim 128 is not supported",
    );
    refused(
        &mutate("\"num_attention_heads\": 8", "\"num_attention_heads\": 7"),
        "num_attention_heads 7 is not a multiple of num_key_value_heads 2",
    );
    refused(
        &mutate(
            "\"linear_num_value_heads\": 16",
            "\"linear_num_value_heads\": 32",
        ),
        "value heads must equal key heads",
    );
    refused(
        &mutate(
            "\"linear_key_head_dim\": 128",
            "\"linear_key_head_dim\": 64",
        ),
        "linear_key_head_dim 64",
    );
    refused(
        &mutate(
            "\"linear_value_head_dim\": 128",
            "\"linear_value_head_dim\": 120",
        ),
        "not a multiple of 16",
    );
    refused(
        &mutate(
            "\"linear_conv_kernel_dim\": 4",
            "\"linear_conv_kernel_dim\": 9",
        ),
        "linear_conv_kernel_dim 9",
    );
    refused(
        &mutate("\"num_hidden_layers\": 24", "\"num_hidden_layers\": 23"),
        "24 entries for num_hidden_layers 23",
    );
    let swapped = mutate(
        "\"linear_attention\",\n            \"full_attention\",",
        "\"full_attention\",\n            \"linear_attention\",",
    );
    refused(&swapped, "disagrees with full_attention_interval 4");
}

/// The JSON edges `config.json` reading relies on, the same under the old
/// crate-private reader and ojas-io's:
/// - 32 nested containers parse, and are then refused as not an object;
///   33 are refused by the reader;
/// - more than 16 MiB is refused unread;
/// - a duplicate key is refused;
/// - a dimension written as a float is not an integer;
/// - `attention_dropout` written `0` is the implemented `0.0`.
#[test]
fn the_json_edges_config_reading_relies_on() {
    let nested = |n: usize| "[".repeat(n) + &"]".repeat(n);
    refused(&nested(32), "the root is not an object");
    refused(&nested(33), "nesting deeper than 32");
    refused(&(" ".repeat(16 << 20) + "{}"), "16777216");
    refused(
        &mutate(
            "\"hidden_size\": 2048,",
            "\"hidden_size\": 2048, \"hidden_size\": 2048,",
        ),
        "duplicate key",
    );
    refused(
        &mutate("\"hidden_size\": 2048", "\"hidden_size\": 2048.0"),
        "text_config.hidden_size must be a non-negative integer, got the number 2048",
    );
    Qwen35TextConfig::from_json(&mutate(
        "\"attention_dropout\": 0.0",
        "\"attention_dropout\": 0",
    ))
    .unwrap();
    refused(
        &mutate("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\""),
        "text_config.hidden_act is the string \"gelu\"; this provider implements only the string \"silu\"",
    );
}

/// A float field written as an integer that `f64` cannot hold exactly is
/// refused, not rounded (ojas-io's `as_f64`). The old reader rounded
/// 9007199254740993 to 9007199254740992.0 and went on.
#[test]
fn a_float_field_holding_an_inexact_integer_is_refused() {
    refused(
        &mutate(
            "\"rope_theta\": 10000000",
            "\"rope_theta\": 9007199254740993",
        ),
        "text_config.rope_parameters.rope_theta is the number 9007199254740993: JSON: integer is not \
         exactly representable as f64",
    );
}

#[test]
fn semantics_the_step_does_not_implement_are_refused() {
    refused(
        &mutate("\"attention_dropout\": 0.0", "\"attention_dropout\": 0.1"),
        "attention_dropout",
    );
    refused(
        &mutate(
            "\"mamba_ssm_dtype\": \"float32\"",
            "\"mamba_ssm_dtype\": \"bfloat16\"",
        ),
        "mamba_ssm_dtype",
    );
    refused(
        &mutate("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\""),
        "hidden_act",
    );
    refused(
        &mutate("\"attention_bias\": false", "\"attention_bias\": true"),
        "attention_bias",
    );
    refused(
        &mutate("\"attn_output_gate\": true", "\"attn_output_gate\": false"),
        "attn_output_gate",
    );
    refused(
        &mutate("\"mlp_only_layers\": []", "\"mlp_only_layers\": [3]"),
        "mlp_only_layers",
    );
    refused(
        &mutate(
            "\"tie_word_embeddings\": true,\n        \"use_cache\"",
            "\"tie_word_embeddings\": false,\n        \"use_cache\"",
        ),
        "untied LM head",
    );
    refused(
        &mutate(
            "\"model_type\": \"qwen3_5_text\"",
            "\"model_type\": \"qwen3_text\"",
        ),
        "qwen3_5_text",
    );
    refused(
        &mutate(
            "\"head_dim\": 256,",
            "\"head_dim\": 256, \"rope_scaling\": {\"type\": \"yarn\"},",
        ),
        "rope_scaling",
    );
}

#[test]
fn rope_and_mrope_are_read_and_refused_where_they_do_not_collapse() {
    refused(
        &mutate("\"rope_type\": \"default\"", "\"rope_type\": \"yarn\""),
        "only \"default\"",
    );
    // Sections that do not cover rotary_dim / 2 = 32 frequencies exactly.
    for bad in ["[11, 11, 11]", "[11, 11]", "[16, 16, 16]"] {
        match Qwen35TextConfig::from_json(&mutate(
            "[\n                11,\n                11,\n                10\n            ]",
            bad,
        )) {
            Err(Qwen35Error::Unsupported { what, .. }) => {
                assert!(what.contains("rotary_dim / 2 = 32"), "{what}")
            }
            other => panic!("{bad}: {:?}", other.map(|c| c.mrope)),
        }
    }
    refused(
        &mutate(
            "[\n                11,\n                11,\n                10\n            ]",
            "[11, 0, 21]",
        ),
        "must be in 1..=u32::MAX",
    );
    refused(
        &mutate(
            "\"mrope_interleaved\": true",
            "\"mrope_interleaved\": \"yes\"",
        ),
        "not a bool",
    );
    let no_section = mutate(
        "\"mrope_section\": [\n                11,\n                11,\n                10\n            ],",
        "",
    );
    refused(
        &no_section,
        "mrope_interleaved is set without mrope_section",
    );
    // A flat copy beside rope_parameters must agree with it.
    refused(
        &mutate(
            "\"head_dim\": 256,",
            "\"head_dim\": 256, \"partial_rotary_factor\": 0.5,",
        ),
        "disagrees with rope_parameters",
    );
    // Interleaving does not matter for text-only input: false parses too.
    let c = Qwen35TextConfig::from_json(&mutate(
        "\"mrope_interleaved\": true",
        "\"mrope_interleaved\": false",
    ))
    .unwrap();
    assert_eq!(c.mrope.unwrap().interleaved, Some(false));
    // A partial rotary factor that leaves an odd rotated width.
    refused(
        &mutate(
            "\"partial_rotary_factor\": 0.25",
            "\"partial_rotary_factor\": 0.2",
        ),
        "not an even whole number",
    );
}

#[test]
fn malformed_json_is_a_config_error() {
    for bad in ["", "[]", "{\"text_config\": 1}", "{\"a\": 1, \"a\": 2}"] {
        assert!(
            matches!(
                Qwen35TextConfig::from_json(bad),
                Err(Qwen35Error::Config(_))
            ),
            "{bad:?}"
        );
    }
}
