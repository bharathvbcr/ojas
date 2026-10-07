//! **K11: AdamW in torch's single-tensor order, with a per-parameter weight
//! decay and learning-rate scale, plus the gradient's squared norm**, f64.
//!
//! One step on parameter tensor `i` (torch `optim/adam.py:416-545` with
//! `decoupled_weight_decay`, `foreach=False`, `fused=False`, amsgrad and
//! maximize off), with `lr_i = lr * lr_scale[i]`, which is how Lappi's
//! `apply_lr` sets each group's rate (`python/qd_train/optim.py:224-241`):
//!
//! ```text
//! step += 1
//! p  = p * (1 - lr_i * wd[i])                         decoupled decay, first
//! m  = lerp(m, g, 1 - beta1)                          torch's lerp: m + w (g - m) for |w| < 0.5,
//!                                                     else g - (g - m)(1 - w)
//! v  = v * beta2 + ((1 - beta2) g) g                  addcmul
//! bc1 = 1 - beta1^step,  bc2 = 1 - beta2^step         (f64 pow, as Python's float **)
//! denom = sqrt(v) / bc2^0.5 + eps                     bias correction outside the sqrt; eps after it
//! p  = p + (-(lr_i / bc1)) * (m / denom)              addcdiv
//! ```
//!
//! `grad_sq_norm` is `sum_i sum_k g_ik^2`, the square of the norm
//! `clip_grad_norm_` scales by.
//!
//! [`lower_layer_lr_scale`] is F's split, ported from Lappi's
//! `layerwise_param_groups` (`optim.py:247,250-316`): a parameter whose name
//! carries a decoder-layer index `layers.<i>.` (at the start or after a `.`)
//! with `i < lower_layers_n`, and no `visual` in its name, trains at
//! `lower_lr_scale`; everything else, embeddings and the final norm included,
//! at 1.0. F: `lower_layers_n = 8`, `lower_lr_scale = 0.1`, `eps = 1e-8`,
//! `weight_decay = 0.01` on every parameter, betas `(0.9, 0.999)`
//! (`optim.py:63,336-338`).
//!
//! # Validation
//!
//! - **golden**: `tests/fixtures/goldens/adamw_f_*`, five steps of
//!   `torch.optim.AdamW` float64 over F's groups as Lappi's own
//!   `layerwise_param_groups` + `apply_lr` build them, and the per-step squared
//!   gradient norm. tessl's `tests/qwen35_adamw.rs` holds no committed golden
//!   (its reference is an in-test formula, with no `lr_scale`).

/// `beta1`, `beta2`, `eps` and the base learning rate.
#[derive(Clone, Copy, Debug)]
pub struct Hyper {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

/// F's optimizer constants (`optim.py:63,336-338`); `lr` is the schedule's.
pub fn f_hyper(lr: f64) -> Hyper {
    Hyper {
        lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
    }
}

/// F's weight decay, on every parameter.
pub const F_WEIGHT_DECAY: f64 = 0.01;
/// F's lower group: decoder layers `0..8` at 0.1x.
pub const F_LOWER_LAYERS: usize = 8;
pub const F_LOWER_LR_SCALE: f64 = 0.1;

/// The first decoder-layer index in `name`, as Python's
/// `re.search(r"(?:^|\.)layers\.(\d+)\.", name)` finds it.
pub fn layer_index(name: &str) -> Option<usize> {
    const KEY: &str = "layers.";
    let mut from = 0;
    while let Some(off) = name[from..].find(KEY) {
        let at = from + off;
        let starts = at == 0 || name.as_bytes()[at - 1] == b'.';
        let rest = &name[at + KEY.len()..];
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        if starts && digits > 0 && rest.as_bytes().get(digits) == Some(&b'.') {
            return Some(rest[..digits].parse().expect("ascii digits parse"));
        }
        from = at + 1;
    }
    None
}

/// The learning-rate scale `layerwise_param_groups` gives `name`.
pub fn lower_layer_lr_scale(name: &str, lower_layers_n: usize, lower_lr_scale: f64) -> f64 {
    match layer_index(name) {
        Some(i) if i < lower_layers_n && !name.contains("visual") => lower_lr_scale,
        _ => 1.0,
    }
}

/// torch's scalar `lerp(self, end, weight)` (ATen `Lerp.h`).
pub fn torch_lerp(start: f64, end: f64, weight: f64) -> f64 {
    if weight.abs() < 0.5 {
        start + weight * (end - start)
    } else {
        end - (end - start) * (1.0 - weight)
    }
}

/// Moments and the step count, one `m` and `v` per parameter tensor.
#[derive(Clone, Debug)]
pub struct State {
    pub m: Vec<Vec<f64>>,
    pub v: Vec<Vec<f64>>,
    pub step: u64,
}

impl State {
    pub fn new(sizes: &[usize]) -> Self {
        Self {
            m: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            v: sizes.iter().map(|&n| vec![0.0; n]).collect(),
            step: 0,
        }
    }
}

/// One AdamW step over every parameter tensor, in place.
pub fn adamw_step_f64(
    params: &mut [Vec<f64>],
    grads: &[Vec<f64>],
    state: &mut State,
    h: Hyper,
    wd: &[f64],
    lr_scale: &[f64],
) {
    let n = params.len();
    for (name, got) in [
        ("grads", grads.len()),
        ("m", state.m.len()),
        ("v", state.v.len()),
        ("wd", wd.len()),
        ("lr_scale", lr_scale.len()),
    ] {
        assert_eq!(got, n, "adamw: {name} has {got} entries for {n} parameters");
    }
    for (what, x) in [
        ("lr", h.lr),
        ("beta1", h.beta1),
        ("beta2", h.beta2),
        ("eps", h.eps),
    ] {
        assert!(
            x.is_finite() && x >= 0.0,
            "adamw: {what} = {x} must be finite and non-negative"
        );
    }
    assert!(
        h.beta1 < 1.0 && h.beta2 < 1.0,
        "adamw: betas must be below 1"
    );
    state.step = state
        .step
        .checked_add(1)
        .expect("adamw: step count overflow");
    let step = f64::from(u32::try_from(state.step).expect("adamw: step fits u32"));
    let bc1 = 1.0 - h.beta1.powf(step);
    let bc2 = 1.0 - h.beta2.powf(step);
    let bc2_sqrt = bc2.powf(0.5);
    for i in 0..n {
        let (p, g, m, v) = (&mut params[i], &grads[i], &mut state.m[i], &mut state.v[i]);
        assert!(
            g.len() == p.len() && m.len() == p.len() && v.len() == p.len(),
            "adamw: parameter {i} lengths differ"
        );
        assert!(
            wd[i].is_finite() && wd[i] >= 0.0,
            "adamw: wd[{i}] = {}",
            wd[i]
        );
        assert!(
            lr_scale[i].is_finite() && lr_scale[i] > 0.0,
            "adamw: lr_scale[{i}] = {}",
            lr_scale[i]
        );
        if let Some(k) = g.iter().position(|x| !x.is_finite()) {
            panic!("adamw: grad {i}[{k}] is not finite");
        }
        let lr_i = h.lr * lr_scale[i];
        let decay = 1.0 - lr_i * wd[i];
        let step_size = lr_i / bc1;
        for k in 0..p.len() {
            p[k] *= decay;
            m[k] = torch_lerp(m[k], g[k], 1.0 - h.beta1);
            v[k] = v[k] * h.beta2 + ((1.0 - h.beta2) * g[k]) * g[k];
            let denom = v[k].sqrt() / bc2_sqrt + h.eps;
            p[k] += -step_size * (m[k] / denom);
        }
    }
}

/// `sum_i sum_k g_ik^2`, tensors in order, elements in order.
pub fn grad_sq_norm_f64(grads: &[Vec<f64>]) -> f64 {
    grads
        .iter()
        .map(|g| g.iter().map(|x| x * x).sum::<f64>())
        .sum()
}
