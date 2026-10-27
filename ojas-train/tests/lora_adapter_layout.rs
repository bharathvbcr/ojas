//! Canonical LoRA-A draw order for [`lora_adapter_layout`].
//!
//! Equal in contract to Lappi `crates/qd-train/tests/lora_plan.rs`
//! `the_adapter_layout_is_canonical_whatever_order_the_provider_lists_entries_in`:
//! reversing the provider list does not change the layout, the layer index is
//! non-decreasing, and a leaf with no layer index is refused. A provider listed
//! in alphabetical HF key order and one listed in a different ojas order yield
//! that same layout. Two modules at one draw position are refused.

use ojas_train::lappi::recipe::{self, LoraHostNames};
use ojas_train::lappi::step::ParamSpec;

fn pair(module: &str, rank: usize, in_features: usize, out: usize) -> [ParamSpec; 2] {
    [
        ParamSpec::new(format!("{module}.lora_A.weight"), &[rank, in_features]),
        ParamSpec::new(format!("{module}.lora_B.weight"), &[out, rank]),
    ]
}

fn push_pairs(order: &[(&str, usize)]) -> Vec<ParamSpec> {
    let mut out = Vec::with_capacity(order.len() * 2);
    for (module, inn) in order {
        out.extend(pair(module, 8, *inn, 16));
    }
    out
}

/// Not canonical. Alphabetical state-dict order puts `k_proj` before `q_proj`.
fn hf_order() -> Vec<(&'static str, usize)> {
    vec![
        ("model.layers.0.linear_attn.in_proj_qkv", 8),
        ("model.layers.0.mlp.gate_proj", 9),
        ("model.layers.0.self_attn.k_proj", 7),
        ("model.layers.0.self_attn.q_proj", 4),
        ("model.layers.1.mlp.down_proj", 11),
        ("model.layers.1.self_attn.q_proj", 5),
    ]
}

/// Not canonical, and not the HF list: layer 1 first, and suffixes reversed
/// against D6 inside a layer.
fn ojas_order() -> Vec<(&'static str, usize)> {
    vec![
        ("model.layers.1.mlp.down_proj", 11),
        ("model.layers.0.self_attn.k_proj", 7),
        ("model.layers.0.mlp.gate_proj", 9),
        ("model.layers.1.self_attn.q_proj", 5),
        ("model.layers.0.self_attn.q_proj", 4),
        ("model.layers.0.linear_attn.in_proj_qkv", 8),
    ]
}

fn canonical() -> Vec<(String, usize)> {
    [
        ("model.layers.0.self_attn.q_proj", 4),
        ("model.layers.0.self_attn.k_proj", 7),
        ("model.layers.0.linear_attn.in_proj_qkv", 8),
        ("model.layers.0.mlp.gate_proj", 9),
        ("model.layers.1.self_attn.q_proj", 5),
        ("model.layers.1.mlp.down_proj", 11),
    ]
    .into_iter()
    .map(|(m, inn)| (m.to_owned(), inn))
    .collect()
}

#[test]
fn the_adapter_layout_is_canonical_whatever_order_the_provider_lists_entries_in() {
    let hf = push_pairs(&hf_order());
    let ojas = push_pairs(&ojas_order());
    let want = canonical();

    let (rank, from_hf) = recipe::lora_adapter_layout(&hf).unwrap();
    let (rank_ojas, from_ojas) = recipe::lora_adapter_layout(&ojas).unwrap();
    assert_eq!(rank, 8);
    assert_eq!(rank_ojas, rank);
    assert_eq!(from_hf, want, "HF registration order");
    assert_eq!(from_ojas, want, "ojas registration order");
    assert_eq!(from_hf, from_ojas);

    let mut reversed = hf.clone();
    reversed.reverse();
    assert_eq!(
        recipe::lora_adapter_layout(&reversed).unwrap(),
        (rank, from_hf.clone())
    );

    let layer = |m: &str| -> u64 {
        m.split('.')
            .skip_while(|c| *c != "layers")
            .nth(1)
            .unwrap()
            .parse()
            .unwrap()
    };
    assert!(
        from_hf.windows(2).all(|w| layer(&w[0].0) <= layer(&w[1].0)),
        "{from_hf:?}"
    );
    let q = from_hf
        .iter()
        .position(|m| m.0.ends_with("self_attn.q_proj") && layer(&m.0) == 0)
        .unwrap();
    let k = from_hf
        .iter()
        .position(|m| m.0.ends_with("self_attn.k_proj"))
        .unwrap();
    assert!(q < k, "D6 puts q_proj before k_proj, got {from_hf:?}");

    let flat = [
        ParamSpec::new("self_attn.q_proj.lora_A.weight", &[16, 4]),
        ParamSpec::new("self_attn.q_proj.lora_B.weight", &[4, 16]),
    ];
    let err = recipe::lora_adapter_layout(&flat).unwrap_err().to_string();
    assert!(err.contains("layer index"), "{err}");
}

#[test]
fn a_shared_draw_position_is_refused_and_the_place_is_the_layers_before_the_suffix() {
    let ambiguous_at = |modules: [&str; 2]| {
        let mut entries = Vec::new();
        for module in modules {
            entries.extend(pair(module, 8, 4, 4));
        }
        let err = recipe::lora_adapter_layout(&entries)
            .unwrap_err()
            .to_string();
        assert!(err.contains("ambiguous"), "{err}");
        for module in modules {
            assert!(err.contains(module), "{err}");
        }
        err
    };
    // Leading zeros parse as the same layer. Lappi refuses that; it does not keep both.
    ambiguous_at([
        "model.layers.1.self_attn.q_proj",
        "model.layers.01.self_attn.q_proj",
    ]);
    ambiguous_at([
        "foo.layers.0.self_attn.q_proj",
        "bar.layers.0.self_attn.q_proj",
    ]);

    let refuse = |module: &str| {
        let entries = pair(module, 8, 4, 4);
        let err = recipe::lora_adapter_layout(&entries)
            .unwrap_err()
            .to_string();
        assert!(err.contains("layer index"), "{module}: {err}");
    };
    refuse("self_attn.q_proj");
    refuse("layers..self_attn.q_proj");
    refuse("layers.x.self_attn.q_proj");
    refuse("xlayers.0.self_attn.q_proj");
    refuse("model.layer.0.self_attn.q_proj");
    refuse("model.layers.18446744073709551616.self_attn.q_proj");

    // `layer_index` returns the first `layers.<n>`. The draw place is the
    // component directly before the suffix, so this is layer 0, not layer 2.
    let nested = "model.layers.2.layers.0.self_attn.q_proj";
    let (_, layout) = recipe::lora_adapter_layout(&pair(nested, 8, 4, 4)).unwrap();
    assert_eq!(layout, vec![(nested.to_owned(), 4)]);
    ambiguous_at([nested, "model.layers.0.self_attn.q_proj"]);

    let mut backwards = Vec::new();
    for module in [
        "model.layers.3.mlp.down_proj",
        "model.layers.3.linear_attn.in_proj_z",
        "model.layers.3.linear_attn.in_proj_qkv",
        "model.layers.3.self_attn.q_proj",
    ] {
        backwards.extend(pair(module, 2, 6, 6));
    }
    let (_, layout) = recipe::lora_adapter_layout(&backwards).unwrap();
    assert_eq!(
        layout.iter().map(|(m, _)| m.as_str()).collect::<Vec<_>>(),
        vec![
            "model.layers.3.self_attn.q_proj",
            "model.layers.3.linear_attn.in_proj_qkv",
            "model.layers.3.linear_attn.in_proj_z",
            "model.layers.3.mlp.down_proj",
        ]
    );

    let max_layer = "model.layers.18446744073709551615.self_attn.v_proj";
    let (_, layout) = recipe::lora_adapter_layout(&pair(max_layer, 8, 3, 3)).unwrap();
    assert_eq!(layout, vec![(max_layer.to_owned(), 3)]);

    let prefixed = "model.language_model.layers.2.self_attn.v_proj";
    let (_, layout) = recipe::lora_adapter_layout(&pair(prefixed, 8, 5, 5)).unwrap();
    assert_eq!(layout, vec![(prefixed.to_owned(), 5)]);

    // B before A is still one adapter, in the canonical place.
    let module = "model.layers.4.mlp.up_proj";
    let [a, b] = pair(module, 8, 9, 9);
    let (_, layout) = recipe::lora_adapter_layout(&[b, a]).unwrap();
    assert_eq!(layout, vec![(module.to_owned(), 9)]);
}

#[test]
fn a_repeated_leaf_is_still_listed_twice_before_the_draw_order() {
    let mut twice = pair("model.layers.0.self_attn.q_proj", 8, 4, 4).to_vec();
    twice.push(ParamSpec::new(
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        &[8, 4],
    ));
    let err = recipe::lora_adapter_layout(&twice).unwrap_err().to_string();
    assert!(err.contains("listed twice"), "{err}");
}

#[test]
fn the_lora_group_stays_in_provider_order() {
    let hf = push_pairs(&hf_order());
    let names = LoraHostNames {
        d17_entry: "host.d17",
        span_head: &[],
    };
    let host = vec![ParamSpec::new("host.d17", &[1])];
    let groups = recipe::lora_param_groups(&hf, &host, &names).unwrap();
    assert_eq!(groups[0].name, "lora");
    assert_eq!(
        groups[0].params,
        hf.iter().map(|e| e.name.clone()).collect::<Vec<_>>()
    );
    assert_eq!(groups[1].params, vec!["host.d17".to_owned()]);
}

#[test]
fn a_leading_zero_sorts_as_its_integer_and_one_bad_name_refuses_the_rest() {
    // "02" sorts before "1" as text. The draw order is the integer.
    let mut entries = Vec::new();
    for module in [
        "model.layers.02.self_attn.q_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
    ] {
        entries.extend(pair(module, 8, 4, 4));
    }
    let (_, layout) = recipe::lora_adapter_layout(&entries).unwrap();
    assert_eq!(
        layout.iter().map(|(m, _)| m.as_str()).collect::<Vec<_>>(),
        vec![
            "model.layers.0.self_attn.k_proj",
            "model.layers.1.self_attn.q_proj",
            "model.layers.02.self_attn.q_proj",
        ]
    );

    let mut mixed = pair("model.layers.0.self_attn.q_proj", 8, 4, 4).to_vec();
    mixed.extend(pair("self_attn.k_proj", 8, 4, 4));
    let err = recipe::lora_adapter_layout(&mixed).unwrap_err().to_string();
    assert!(err.contains("self_attn.k_proj"), "{err}");
    assert!(err.contains("layer index"), "{err}");
}

#[test]
fn no_target_suffix_is_a_suffix_of_another() {
    for (i, a) in recipe::LORA_TARGET_SUFFIXES.iter().enumerate() {
        for (j, b) in recipe::LORA_TARGET_SUFFIXES.iter().enumerate() {
            if i != j {
                assert!(
                    !a.ends_with(*b),
                    "{a} ends with {b}; draw_position keeps the earlier table row"
                );
            }
        }
    }
}
