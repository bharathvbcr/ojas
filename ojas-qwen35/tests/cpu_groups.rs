//! Per-entry learning-rate scales and weight decays from a caller's group
//! spec, built against the real 2B's map on the CPU.

#![cfg(target_os = "macos")]

use ojas_qwen35::{
    tower_tensors, GroupSpec, LrRule, OptimizerPlan, Qwen35Error, Qwen35TextConfig, Select, WdRule,
};

const REAL: &str = include_str!("fixtures/qwen35_2b_base_config.json");

fn table() -> Vec<ojas_qwen35::TensorSpec> {
    tower_tensors(&Qwen35TextConfig::from_json(REAL).unwrap())
}

/// transformers' `Trainer` decay exclusions written as data: `bias`,
/// `layernorm`, `rmsnorm` anywhere, a `norm` segment, a `_norm` suffix.
fn no_decay() -> Select {
    Select::AnyOf(vec![
        Select::Contains("bias".into()),
        Select::Contains("layernorm".into()),
        Select::Contains("rmsnorm".into()),
        Select::Segment("norm".into()),
        Select::SegmentSuffix("_norm".into()),
    ])
}

fn lower(n: u32) -> Select {
    Select::AnyOf(vec![Select::Embedding, Select::LayersBelow(n)])
}

/// The Lappi-shaped spec: embeddings and the lowest 8 layers at 0.1x, decay
/// 0.01 except the exclusions.
fn lappi_like() -> GroupSpec {
    GroupSpec {
        lr: vec![
            LrRule {
                label: "lower".into(),
                select: lower(8),
                lr_scale: 0.1,
            },
            LrRule {
                label: "upper".into(),
                select: Select::Not(Box::new(lower(8))),
                lr_scale: 1.0,
            },
        ],
        weight_decay: vec![
            WdRule {
                label: "no_decay".into(),
                select: no_decay(),
                weight_decay: 0.0,
            },
            WdRule {
                label: "decay".into(),
                select: Select::Not(Box::new(no_decay())),
                weight_decay: 0.01,
            },
        ],
    }
}

#[test]
fn embeddings_and_the_lowest_layers_take_the_scale() {
    let t = table();
    let plan = OptimizerPlan::build(&t, &lappi_like()).unwrap();
    assert_eq!(plan.lr_scale().len(), t.len());
    let mut lowered = 0;
    for (spec, &s) in t.iter().zip(plan.lr_scale()) {
        let want_low = spec.name == "embed_tokens.weight" || spec.layer().is_some_and(|l| l < 8);
        assert_eq!(s, if want_low { 0.1 } else { 1.0 }, "{}", spec.name);
        lowered += usize::from(want_low);
    }
    // Layers 0..8: six GDN layers (0,1,2,4,5,6) of 14 and two attention
    // layers (3,7) of 11, plus the embedding.
    assert_eq!(lowered, 1 + 6 * 14 + 2 * 11);
    // The final norm is not a layer and is not lowered.
    assert_eq!(plan.lr_scale()[t.len() - 1], 1.0);
    let total: usize = plan.groups().iter().map(|g| g.tensors).sum();
    assert_eq!(total, t.len());
    let elements: u64 = plan.groups().iter().map(|g| g.elements).sum();
    assert_eq!(elements, 1_881_825_088);
    assert_eq!(plan.groups().len(), 4, "{:?}", plan.groups());
}

/// The exclusions are data: written as above they reproduce tessl's own
/// `excluded_from_weight_decay` on every 2B parameter (norms and `dt_bias`
/// at 0, `A_log` decayed), and a spec that decays everything decays norms.
#[test]
fn weight_decay_exclusions_are_the_callers_data() {
    let t = table();
    let plan = OptimizerPlan::build(&t, &lappi_like()).unwrap();
    for (spec, &wd) in t.iter().zip(plan.weight_decay()) {
        let tessl_says = tessl::qwen35_adamw::excluded_from_weight_decay(&spec.name);
        assert_eq!(wd, if tessl_says { 0.0 } else { 0.01 }, "{}", spec.name);
    }
    let get =
        |p: &OptimizerPlan, n: &str| p.weight_decay()[t.iter().position(|x| x.name == n).unwrap()];
    assert_eq!(get(&plan, "layers.0.linear_attn.dt_bias"), 0.0);
    assert_eq!(get(&plan, "layers.0.linear_attn.A_log"), 0.01);
    assert_eq!(get(&plan, "layers.0.linear_attn.norm.weight"), 0.0);
    assert_eq!(get(&plan, "layers.3.self_attn.q_norm.weight"), 0.0);
    assert_eq!(get(&plan, "norm.weight"), 0.0);

    let everything = GroupSpec {
        lr: vec![LrRule {
            label: "all".into(),
            select: Select::All,
            lr_scale: 1.0,
        }],
        weight_decay: vec![WdRule {
            label: "all".into(),
            select: Select::All,
            weight_decay: 0.01,
        }],
    };
    let plan = OptimizerPlan::build(&t, &everything).unwrap();
    assert!(plan.weight_decay().iter().all(|&w| w == 0.01));
    assert_eq!(get(&plan, "norm.weight"), 0.01);
    assert_eq!(get(&plan, "layers.0.linear_attn.dt_bias"), 0.01);
    plan.check_tessl_lr().unwrap();
}

#[test]
fn a_spec_that_is_not_a_partition_is_refused() {
    let t = table();
    let refused = |spec: GroupSpec, needle: &str| {
        let e = OptimizerPlan::build(&t, &spec)
            .err()
            .unwrap_or_else(|| panic!("{needle}: accepted"))
            .to_string();
        assert!(e.contains(needle), "{e:?} lacks {needle:?}");
    };
    let wd_all = || {
        vec![WdRule {
            label: "all".into(),
            select: Select::All,
            weight_decay: 0.0,
        }]
    };
    // Overlap: the embedding matches both rules.
    refused(
        GroupSpec {
            lr: vec![
                LrRule {
                    label: "lower".into(),
                    select: lower(8),
                    lr_scale: 0.1,
                },
                LrRule {
                    label: "all".into(),
                    select: Select::All,
                    lr_scale: 1.0,
                },
            ],
            weight_decay: wd_all(),
        },
        "embed_tokens.weight matches learning-rate rules [\"lower\", \"all\"]",
    );
    // Gap: the final norm matches no rule.
    refused(
        GroupSpec {
            lr: vec![
                LrRule {
                    label: "lower".into(),
                    select: lower(8),
                    lr_scale: 0.1,
                },
                LrRule {
                    label: "upper".into(),
                    select: Select::LayersFrom(8),
                    lr_scale: 1.0,
                },
            ],
            weight_decay: wd_all(),
        },
        "norm.weight matches no learning-rate rule",
    );
    // A dead rule (a typo'd segment) is refused, not ignored.
    refused(
        GroupSpec {
            lr: vec![LrRule {
                label: "all".into(),
                select: Select::All,
                lr_scale: 1.0,
            }],
            weight_decay: vec![
                WdRule {
                    label: "typo".into(),
                    select: Select::Segment("nrom".into()),
                    weight_decay: 0.0,
                },
                WdRule {
                    label: "rest".into(),
                    select: Select::Not(Box::new(Select::Segment("nrom".into()))),
                    weight_decay: 0.01,
                },
            ],
        },
        "weight-decay rule \"typo\" matches no parameter",
    );
    let one_lr = |s: f64| {
        vec![LrRule {
            label: "all".into(),
            select: Select::All,
            lr_scale: s,
        }]
    };
    refused(
        GroupSpec {
            lr: one_lr(f64::NAN),
            weight_decay: wd_all(),
        },
        "must be finite",
    );
    refused(
        GroupSpec {
            lr: one_lr(-1.0),
            weight_decay: wd_all(),
        },
        "must be finite and >= 0",
    );
    refused(
        GroupSpec {
            lr: one_lr(1.0),
            weight_decay: vec![WdRule {
                label: "all".into(),
                select: Select::All,
                weight_decay: f32::INFINITY,
            }],
        },
        "must be finite",
    );
    refused(
        GroupSpec {
            lr: vec![],
            weight_decay: wd_all(),
        },
        "no learning-rate rules",
    );
    refused(
        GroupSpec {
            lr: vec![
                LrRule {
                    label: "x".into(),
                    select: lower(8),
                    lr_scale: 0.1,
                },
                LrRule {
                    label: "x".into(),
                    select: Select::Not(Box::new(lower(8))),
                    lr_scale: 1.0,
                },
            ],
            weight_decay: wd_all(),
        },
        "empty or repeated",
    );
}

/// tessl's AdamW has one learning rate: a plan whose scales are not all 1.0
/// is refused before any device work, naming what would lift it.
#[test]
fn a_per_entry_learning_rate_is_refused_until_tessl_has_lr_scale() {
    let plan = OptimizerPlan::build(&table(), &lappi_like()).unwrap();
    match plan.check_tessl_lr() {
        Err(Qwen35Error::Unsupported { what, needs }) => {
            assert!(what.contains("0.1 on embed_tokens.weight"), "{what}");
            assert!(needs.contains("lappi-train-lrscale-mrope"), "{needs}");
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
    // A uniform scale other than 1.0 is refused too: folding it into the
    // learning rate is the caller's decision.
    let uniform = GroupSpec {
        lr: vec![LrRule {
            label: "all".into(),
            select: Select::All,
            lr_scale: 0.5,
        }],
        weight_decay: lappi_like().weight_decay,
    };
    assert!(matches!(
        OptimizerPlan::build(&table(), &uniform)
            .unwrap()
            .check_tessl_lr(),
        Err(Qwen35Error::Unsupported { .. })
    ));
}
