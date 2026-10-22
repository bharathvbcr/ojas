//! Optimizer groups are chosen by parameter NAME, never by tensor rank.
//!
//! `ojas-train` depends only on `ojas-core` and `ojas-io`; its optimizer table
//! (`recipe::optimizer_table`) reads `ParamSpec::name` and nothing of `ParamSpec::shape` but the
//! element count. So a `[Cout, Cin, k, k]` convolution weight is an AdamW entry like any other,
//! and no rule here infers Muon (or anything else) from `ndim`, which is what `ojas-cpu`'s
//! `optim_group` does and what this layer must not copy.
//!
//! Scope: AdamW only. No recipe in this crate names SGD, so there is nothing to assign a
//! parameter to; a future SGD group would be one more name-keyed row in the same table.

use ojas_train::lappi::mock::ToyProvider;
use ojas_train::lappi::recipe::{
    self, BETA1, EPS, EntryHyper, LowerLayers, OptimizerRecipe, WEIGHT_DECAY,
};
use ojas_train::lappi::step::{AdamWHyper, ParamSpec, StepProvider, check_per_entry};

const LR: f64 = 1e-2;
const BETA2: f64 = 0.999;

fn recipe_with(lower: Option<LowerLayers>) -> OptimizerRecipe {
    OptimizerRecipe {
        lr: LR,
        beta2: BETA2,
        lower_layers: lower,
    }
}

fn table(entries: &[ParamSpec], lower: Option<LowerLayers>) -> Vec<EntryHyper> {
    recipe::optimizer_table(entries, &[], &recipe_with(lower)).expect("optimizer table")
}

fn row<'a>(rows: &'a [EntryHyper], name: &str) -> &'a EntryHyper {
    rows.iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("no row named {name}"))
}

const SPLIT: Option<LowerLayers> = Some(LowerLayers {
    n: 1,
    lr_scale: 0.1,
});

#[test]
fn a_four_dimensional_weight_runs_through_adamw_with_the_plain_row() {
    let shape = [8usize, 3, 3, 3];
    let mut provider = ToyProvider::new(8, 4, 2, 42)
        .unwrap()
        .with_trainable_entry("conv1.weight", &shape)
        .unwrap();
    let entries = provider.parameters().to_vec();
    let conv = entries
        .iter()
        .position(|e| e.name == "conv1.weight")
        .unwrap();
    assert_eq!(entries[conv].shape, shape);

    let rows = table(&entries, None);
    let r = row(&rows, "conv1.weight");
    assert_eq!(r.lr_scale, 1.0);
    assert_eq!(r.weight_decay, WEIGHT_DECAY);
    assert_eq!(r.weight_decay, 0.01);
    assert_eq!((r.beta1, r.eps, r.beta2), (BETA1, EPS, BETA2));

    let scales: Vec<f64> = rows.iter().map(|r| r.lr_scale).collect();
    let decays: Vec<f64> = rows.iter().map(|r| r.weight_decay).collect();
    check_per_entry(entries.len(), &scales, &decays).expect("the row is a valid per-entry row");

    let hyper = AdamWHyper {
        lr: LR,
        beta1: BETA1,
        beta2: BETA2,
        eps: EPS,
        grad_scale: 1.0,
    };
    provider
        .adamw_step(&hyper, 1, &scales, &decays)
        .expect("AdamW steps the 4-D entry");
    assert_eq!(provider.step_count(), 1);

    // The gradient is zero, so AdamW's only effect on the entry is the decoupled decay
    // `p * (1 - lr * weight_decay)`, at the plain learning rate (scale 1.0).
    let values = provider.read_parameters().unwrap();
    assert_eq!(values[conv].len(), 8 * 3 * 3 * 3);
    let want = 0.5 * (1.0 - LR * 0.01);
    for v in &values[conv] {
        assert!((f64::from(*v) - want).abs() < 1e-6, "{v} vs {want}");
    }
}

#[test]
fn a_name_based_rule_moves_the_row_whatever_the_rank() {
    let provider = ToyProvider::new(8, 4, 2, 42)
        .unwrap()
        .with_trainable_entry("layers.0.conv.weight", &[8, 3, 3, 3])
        .unwrap()
        .with_trainable_entry("conv1.weight", &[8, 3, 3, 3])
        .unwrap()
        .with_trainable_entry("layers.1.conv.weight", &[8, 3, 3, 3])
        .unwrap();
    let entries = provider.parameters().to_vec();
    let rows = table(&entries, SPLIT);

    assert_eq!(
        row(&rows, "layers.0.conv.weight").lr_scale,
        0.1,
        "a lower-layer NAME is scaled although the tensor is 4-D"
    );
    assert_eq!(
        row(&rows, "layers.0.mlp.weight").lr_scale,
        0.1,
        "the 2-D weight of the same layer gets the same scale: rank is not consulted"
    );
    assert_eq!(
        row(&rows, "conv1.weight").lr_scale,
        1.0,
        "the same 4-D shape under a name the rule does not match is untouched"
    );
    assert_eq!(
        row(&rows, "layers.1.conv.weight").lr_scale,
        1.0,
        "a 4-D entry in an upper layer is untouched"
    );
    assert_eq!(
        row(&rows, "norm.weight").lr_scale,
        1.0,
        "a 1-D entry is untouched"
    );
    assert!(rows.iter().all(|r| r.weight_decay == WEIGHT_DECAY));
}

#[test]
fn the_table_ignores_the_shape_of_every_entry() {
    let a = [
        ParamSpec::new("layers.0.w", &[8, 3, 3, 3]),
        ParamSpec::new("tail.w", &[8, 3, 3, 3]),
    ];
    let b = [
        ParamSpec::new("layers.0.w", &[216]),
        ParamSpec::new("tail.w", &[24, 9]),
    ];
    for lower in [None, SPLIT] {
        assert_eq!(
            table(&a, lower),
            table(&b, lower),
            "same names, same element counts, different ranks: the same table"
        );
    }
}
