//! Self-validation of the K11 host reference (AdamW, torch single-tensor
//! order, per-parameter weight decay and learning-rate scale, and the squared
//! gradient norm) against the float64 `torch.optim.AdamW` golden over F's
//! parameter groups as Lappi's own `layerwise_param_groups` and `apply_lr` build
//! them (`tests/fixtures/goldens/adamw_f_*`).
//!
//! Bounds, written before the first run:
//!
//! - parameters: `1e-13` of the largest `|p|`. Each step is ~15 correctly
//!   rounded operations per element (decay, lerp, two moment ops, sqrt, divide,
//!   addcdiv) with the parameter itself carrying one rounding of `|p| u` per
//!   write: about `15 * 5 * 1.1e-16 = 8e-15` relative over five steps; torch's
//!   vectorized `lerp`/`addcmul` may fuse a multiply-add the scalar order does
//!   not, an ulp per op. 1e-13 is ~12x that, and still 30x below the smallest
//!   semantic slip it must catch here: eps inside the bias correction moves
//!   `p` by ~3e-12 at F's lr (`eps (1/sqrt(bc2) - 1) lr / |g|` at step 1), and
//!   an `lr_scale` dropped from the decay by `0.9 lr wd |p| = 9e-8`.
//! - squared gradient norm: `1e-13` relative (~100 squares summed:
//!   `100 u = 1.1e-14`).

mod reference;

use reference::adamw::{
    adamw_step_f64, f_hyper, grad_sq_norm_f64, layer_index, lower_layer_lr_scale, torch_lerp,
    State, F_LOWER_LAYERS, F_LOWER_LR_SCALE, F_WEIGHT_DECAY,
};
use reference::goldens::{self, assert_rel};

const PARAM_BOUND: f64 = 1e-13;
const NORM_BOUND: f64 = 1e-13;

/// Splits a flat `[N]` golden into the parameter tensors' sizes.
fn split(flat: &[f64], sizes: &[usize]) -> Vec<Vec<f64>> {
    assert_eq!(
        flat.len(),
        sizes.iter().sum::<usize>(),
        "flat golden length vs sizes"
    );
    let mut at = 0;
    sizes
        .iter()
        .map(|&n| {
            let v = flat[at..at + n].to_vec();
            at += n;
            v
        })
        .collect()
}

fn names() -> Vec<String> {
    std::fs::read_to_string(goldens::dir().join("adamw_f_names.txt"))
        .expect("adamw_f_names.txt")
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn k11_lower_layer_split_is_lappis_layerwise_param_groups() {
    let names = names();
    let golden = goldens::f64s("adamw_f_lr_scale");
    assert_eq!(names.len(), golden.len(), "one lr_scale per parameter");
    for (n, &want) in names.iter().zip(&golden) {
        let got = lower_layer_lr_scale(n, F_LOWER_LAYERS, F_LOWER_LR_SCALE);
        assert!(
            got.to_bits() == want.to_bits(),
            "{n}: lr_scale {got}, Lappi's layerwise_param_groups gave {want}"
        );
    }
    // F's split covers the decoder layers 0-7 only; the embedding and final norm are 1.0.
    assert!(
        golden.contains(&0.1) && golden.contains(&1.0),
        "the golden must hold both groups"
    );
    // The regex's edges: `layers.10.` is layer 10, not 1; an index needs a following `.`;
    // the name must start or follow a `.`; `visual` is excluded.
    assert_eq!(layer_index("model.layers.10.mlp.w"), Some(10));
    assert_eq!(layer_index("layers.3.x"), Some(3));
    assert_eq!(layer_index("model.sublayers.3.x"), None);
    assert_eq!(layer_index("model.layers.3"), None);
    assert_eq!(layer_index("model.layers.x.layers.5.w"), Some(5));
    assert_eq!(lower_layer_lr_scale("model.visual.layers.2.w", 8, 0.1), 1.0);
    assert_eq!(
        lower_layer_lr_scale("model.embed_tokens.weight", 8, 0.1),
        1.0
    );
}

#[test]
fn k11_adamw_matches_torch_over_fs_groups_for_five_steps() {
    let names = names();
    let sizes = goldens::indices("adamw_f_sizes");
    let hyper_g = goldens::f64s("adamw_f_hyper"); // [base_lr, beta1, beta2, eps]
    let h = f_hyper(hyper_g[0]);
    for (what, ours, theirs) in [
        ("beta1", h.beta1, hyper_g[1]),
        ("beta2", h.beta2, hyper_g[2]),
        ("eps", h.eps, hyper_g[3]),
    ] {
        assert!(
            ours.to_bits() == theirs.to_bits(),
            "F's {what}: reference {ours}, golden {theirs}"
        );
    }
    let wd_golden = goldens::f64s("adamw_f_weight_decay");
    assert!(
        wd_golden
            .iter()
            .all(|&w| w.to_bits() == F_WEIGHT_DECAY.to_bits()),
        "F decays every parameter at 0.01"
    );
    let wd = vec![F_WEIGHT_DECAY; names.len()];
    let lr_scale: Vec<f64> = names
        .iter()
        .map(|n| lower_layer_lr_scale(n, F_LOWER_LAYERS, F_LOWER_LR_SCALE))
        .collect();

    let mut params = split(&goldens::f64s("adamw_f_p0"), &sizes);
    let (gs, grads) = goldens::f64s_shaped("adamw_f_grads");
    let after = goldens::f64s("adamw_f_params");
    let sq = goldens::f64s("adamw_f_grad_sq_norm");
    let (steps, n) = (gs[0], gs[1]);
    assert!(steps >= 3, "the golden must cover at least 3 steps");
    let mut state = State::new(&sizes);
    for s in 0..steps {
        let g = split(&grads[s * n..(s + 1) * n], &sizes);
        assert_rel(
            &format!("K11 step {} grad_sq_norm", s + 1),
            &[grad_sq_norm_f64(&g)],
            &[sq[s]],
            NORM_BOUND,
        );
        adamw_step_f64(&mut params, &g, &mut state, h, &wd, &lr_scale);
        let flat: Vec<f64> = params.iter().flatten().copied().collect();
        assert_rel(
            &format!("K11 step {} params", s + 1),
            &flat,
            &after[s * n..(s + 1) * n],
            PARAM_BOUND,
        );
    }
    assert_eq!(state.step, u64::try_from(steps).expect("steps fit u64"));
}

#[test]
fn k11_torch_lerp_switches_form_at_one_half() {
    // Below 0.5: start + w (end - start); at or above: end - (end - start)(1 - w).
    let (a, b) = (0.3f64, 1.7f64);
    assert_eq!(
        torch_lerp(a, b, 0.1).to_bits(),
        (a + 0.1 * (b - a)).to_bits()
    );
    assert_eq!(
        torch_lerp(a, b, 0.9).to_bits(),
        (b - (b - a) * (1.0 - 0.9)).to_bits()
    );
    assert_eq!(torch_lerp(a, b, 0.0), a);
    assert_eq!(torch_lerp(a, b, 1.0), b);
}

#[test]
#[should_panic(expected = "lr_scale[1]")]
fn k11_adamw_refuses_a_zero_lr_scale() {
    let mut p = vec![vec![1.0], vec![1.0]];
    let g = vec![vec![0.5], vec![0.5]];
    let mut st = State::new(&[1, 1]);
    adamw_step_f64(
        &mut p,
        &g,
        &mut st,
        f_hyper(1e-3),
        &[0.01, 0.01],
        &[1.0, 0.0],
    );
}
