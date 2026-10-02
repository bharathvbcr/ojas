//! The tower's tensor map and the header pre-flight, on the CPU.
//!
//! Three tiers check the map: the real 2B config here (shapes by hand), tessl's
//! tiny fixture header here (exact set equality, read with ojas-io), and the
//! real snapshot header in the ignored `snapshot_header_matches_the_name_map`
//! (CPU only, reads the HF cache). The order against tessl's live
//! `parameter_table` is `gpu_name_map_is_tessls_parameter_table`.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use ojas_io::StDtype;
use ojas_qwen35::{check_header, tower_tensors, Qwen35TextConfig, Snapshot};

const REAL: &str = include_str!("fixtures/qwen35_2b_base_config.json");

fn tiny_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tessl/tests/fixtures/qwen35_train")
}

fn shape_of(cfg: &Qwen35TextConfig, name: &str) -> Vec<usize> {
    tower_tensors(cfg)
        .into_iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("{name} is not in the map"))
        .shape
}

#[test]
fn the_2b_map_names_and_shapes_every_tower_tensor() {
    let c = Qwen35TextConfig::from_json(REAL).unwrap();
    let t = tower_tensors(&c);
    // 1 embedding + 18 GDN layers x 14 + 6 attention layers x 11 + 1 final norm.
    assert_eq!(t.len(), 1 + 18 * 14 + 6 * 11 + 1);
    assert_eq!(
        t.len(),
        320,
        "the snapshot holds 320 model.language_model.* tensors"
    );
    assert_eq!(t[0].name, "embed_tokens.weight");
    assert_eq!(t.last().unwrap().name, "norm.weight");
    for (name, shape) in [
        ("embed_tokens.weight", vec![248_320, 2048]),
        ("layers.0.linear_attn.in_proj_qkv.weight", vec![6144, 2048]),
        ("layers.0.linear_attn.in_proj_z.weight", vec![2048, 2048]),
        ("layers.0.linear_attn.in_proj_b.weight", vec![16, 2048]),
        ("layers.0.linear_attn.in_proj_a.weight", vec![16, 2048]),
        ("layers.0.linear_attn.out_proj.weight", vec![2048, 2048]),
        ("layers.0.linear_attn.conv1d.weight", vec![6144, 1, 4]),
        ("layers.0.linear_attn.A_log", vec![16]),
        ("layers.0.linear_attn.dt_bias", vec![16]),
        ("layers.0.linear_attn.norm.weight", vec![128]),
        ("layers.3.self_attn.q_proj.weight", vec![4096, 2048]),
        ("layers.3.self_attn.k_proj.weight", vec![512, 2048]),
        ("layers.3.self_attn.v_proj.weight", vec![512, 2048]),
        ("layers.3.self_attn.o_proj.weight", vec![2048, 2048]),
        ("layers.3.self_attn.q_norm.weight", vec![256]),
        ("layers.3.self_attn.k_norm.weight", vec![256]),
        ("layers.23.mlp.gate_proj.weight", vec![6144, 2048]),
        ("layers.23.mlp.up_proj.weight", vec![6144, 2048]),
        ("layers.23.mlp.down_proj.weight", vec![2048, 6144]),
        ("layers.23.input_layernorm.weight", vec![2048]),
        ("layers.23.post_attention_layernorm.weight", vec![2048]),
        ("norm.weight", vec![2048]),
    ] {
        assert_eq!(shape_of(&c, name), shape, "{name}");
    }
    // tessl's parameter-table order inside a layer.
    let l3: Vec<&str> = t
        .iter()
        .filter(|x| x.name.starts_with("layers.3."))
        .map(|x| x.name.as_str())
        .collect();
    assert_eq!(
        l3,
        [
            "layers.3.self_attn.q_proj.weight",
            "layers.3.self_attn.k_proj.weight",
            "layers.3.self_attn.v_proj.weight",
            "layers.3.self_attn.o_proj.weight",
            "layers.3.self_attn.q_norm.weight",
            "layers.3.self_attn.k_norm.weight",
            "layers.3.mlp.gate_proj.weight",
            "layers.3.mlp.up_proj.weight",
            "layers.3.mlp.down_proj.weight",
            "layers.3.input_layernorm.weight",
            "layers.3.post_attention_layernorm.weight",
        ]
    );
    let names: std::collections::BTreeSet<&str> = t.iter().map(|x| x.name.as_str()).collect();
    assert_eq!(names.len(), t.len(), "every name once");
}

#[test]
fn tessls_tiny_fixture_header_is_exactly_the_map() {
    let s = Snapshot::from_dir(&tiny_dir()).unwrap();
    assert_eq!(s.header.tower_tensors, tower_tensors(&s.config).len());
    assert!(s.header.not_loaded.is_empty(), "{:?}", s.header.not_loaded);
    assert_eq!(s.weights_path.file_name().unwrap(), "model.safetensors");
}

/// Synthetic headers built from the map, then broken one way each.
#[test]
fn a_header_that_is_not_the_tower_is_refused() {
    let c = Qwen35TextConfig::from_json(REAL).unwrap();
    let map = tower_tensors(&c);
    let full: Vec<(String, StDtype, Vec<u64>)> = map
        .iter()
        .map(|t| {
            (
                format!("{}{}", c.tower_prefix, t.name),
                StDtype::BF16,
                t.shape.iter().map(|&d| d as u64).collect(),
            )
        })
        .chain([(
            "model.visual.blocks.0.attn.qkv.weight".to_string(),
            StDtype::BF16,
            vec![3072, 1024],
        )])
        .chain([("mtp.fc.weight".to_string(), StDtype::BF16, vec![2048, 4096])])
        .collect();
    let run = |entries: &[(String, StDtype, Vec<u64>)]| {
        check_header(
            &c,
            entries
                .iter()
                .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
            "synthetic",
        )
    };
    let ok = run(&full).unwrap();
    assert_eq!(ok.tower_tensors, 320);
    assert_eq!(ok.not_loaded.get("model.visual"), Some(&1));
    assert_eq!(ok.not_loaded.get("mtp.fc"), Some(&1));

    let expect_err = |entries: Vec<(String, StDtype, Vec<u64>)>, needle: &str| {
        let e = run(&entries)
            .err()
            .unwrap_or_else(|| panic!("{needle}: accepted"))
            .to_string();
        assert!(e.contains(needle), "{e:?} lacks {needle:?}");
    };
    let mut missing = full.clone();
    missing.remove(5);
    expect_err(missing, "1 tower tensors are missing");
    let mut wrong_shape = full.clone();
    wrong_shape[1].2 = vec![6144, 2047];
    expect_err(wrong_shape, "has shape [6144, 2047]");
    let mut extra = full.clone();
    extra.push((
        "model.language_model.layers.0.mlp.gate_proj.bias".into(),
        StDtype::BF16,
        vec![6144],
    ));
    expect_err(extra, "not a parameter of the tower");
    let mut head = full.clone();
    head.push(("lm_head.weight".into(), StDtype::BF16, vec![248_320, 2048]));
    expect_err(head, "ties the LM head");
    let mut ints = full.clone();
    ints[0].1 = StDtype::I64;
    expect_err(ints, "not a float");
}

/// The real Qwen3.5-2B-Base snapshot in the Hugging Face cache: its
/// `config.json` is the fixture byte for byte, and its single weights file's
/// header (read with ojas-io, header only) is exactly the map plus the vision
/// tower and the MTP block, which are not loaded. CPU only; ignored because it
/// needs the snapshot. `QWEN35_2B_SNAPSHOT` overrides the directory.
#[test]
#[ignore]
fn snapshot_header_matches_the_name_map() {
    let dir = match std::env::var_os("QWEN35_2B_SNAPSHOT") {
        Some(d) => PathBuf::from(d),
        None => {
            let root = PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots");
            let dirs: Vec<PathBuf> = std::fs::read_dir(&root)
                .unwrap_or_else(|e| panic!("{}: {e}", root.display()))
                .map(|e| e.unwrap().path())
                .filter(|p| p.join("config.json").is_file())
                .collect();
            assert_eq!(
                dirs.len(),
                1,
                "expected one snapshot under {}",
                root.display()
            );
            dirs[0].clone()
        }
    };
    assert_eq!(
        std::fs::read_to_string(dir.join("config.json")).unwrap(),
        REAL
    );
    let s = Snapshot::from_dir(&dir).unwrap();
    assert_eq!(s.header.tower_tensors, 320);
    let want: std::collections::BTreeMap<String, usize> = [
        ("model.visual", 297),
        ("mtp.layers", 11),
        ("mtp.fc", 1),
        ("mtp.norm", 1),
        ("mtp.pre_fc_norm_embedding", 1),
        ("mtp.pre_fc_norm_hidden", 1),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    assert_eq!(s.header.not_loaded, want);
    eprintln!("weights: {}", s.weights_path.display());
}

/// A sharded snapshot's `model.safetensors.index.json`: the tower must sit
/// in one plain file name, and every other shape of index is refused by
/// what it is. Built on the tiny fixture, with its weights copied under a
/// shard name.
#[test]
fn a_snapshot_index_resolves_to_one_tower_file_or_is_refused() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("ojas-qwen35-index-{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(tiny_dir().join("config.json"), dir.join("config.json")).unwrap();
    let shard = "model-00001-of-00001.safetensors";
    std::fs::copy(tiny_dir().join("model.safetensors"), dir.join(shard)).unwrap();
    let names: Vec<String> = {
        let c = Qwen35TextConfig::from_file(&dir.join("config.json")).unwrap();
        tower_tensors(&c)
            .into_iter()
            .map(|t| format!("{}{}", c.tower_prefix, t.name))
            .collect()
    };
    let index = |entries: &[(String, String)]| {
        let body: Vec<String> = entries.iter().map(|(k, v)| format!("{k:?}: {v}")).collect();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            format!(
                "{{\"metadata\": {{\"total_size\": 1}}, \"weight_map\": {{{}}}}}",
                body.join(", ")
            ),
        )
        .unwrap();
    };
    let all_in = |file: &str| -> Vec<(String, String)> {
        names
            .iter()
            .map(|n| (n.clone(), format!("{file:?}")))
            .collect()
    };
    let refused = |needle: &str| {
        let e = Snapshot::from_dir(&dir).expect_err(needle).to_string();
        assert!(e.contains(needle), "{e:?} lacks {needle:?}");
    };

    index(&all_in(shard));
    let s = Snapshot::from_dir(&dir).unwrap();
    assert_eq!(s.weights_path, dir.join(shard));
    assert_eq!(s.header.tower_tensors, names.len());

    let mut two = all_in(shard);
    two[1].1 = "\"model-00002-of-00002.safetensors\"".into();
    index(&two);
    refused("span 2 files");
    let mut up = all_in(shard);
    up[0].1 = "\"../model.safetensors\"".into();
    index(&up);
    refused("not a file name");
    let mut num = all_in(shard);
    num[0].1 = "3".into();
    index(&num);
    refused("maps to the number 3");
    index(&[("vision.x".into(), format!("{shard:?}"))]);
    refused("no tensor under \"model.\"");
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        "{\"metadata\": {}}",
    )
    .unwrap();
    refused("no weight_map object");
    std::fs::write(dir.join("model.safetensors.index.json"), "[]").unwrap();
    refused("no weight_map object");
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        "{\"weight_map\": {\"a\": \"x\", \"a\": \"y\"}}",
    )
    .unwrap();
    refused("duplicate key");
}
