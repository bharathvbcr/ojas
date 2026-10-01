//! Lappi's fine-tuning optimizer recipe, as the PyTorch trainer runs it, and the per-entry
//! vectors a [`crate::step::StepProvider`] is driven with.
//!
//! **Where each number comes from (read 2026-10-01, `GAP-OJAS-ADVICE-F-ADAMW-EPS-WD-UNREAD-2026-10-01`):**
//!
//! * betas `(0.9, beta2)`, `beta2` default 0.999 (`python/qd_train/optim.py:63,545`);
//! * eps 1e-8 and weight decay 0.01: `MasterWeightAdamW`'s defaults (`optim.py:336-339`), which
//!   `build_optimizer` never overrides (`optim.py:547`), handed to `torch.optim.AdamW` over every
//!   group (`optim.py:402-405`). torch applies a group's `weight_decay` to every tensor in it, and
//!   no group sets its own, so **every** parameter decays at 0.01: the norms, `dt_bias`, the tied
//!   embedding and the span head included. tessl's `default_weight_decay` (transformers' Trainer
//!   exclusions, norms and `dt_bias` at 0) is a different optimizer and is never used here;
//! * clip 1.0 over the tower and the head together (`python/qd_train/backbone.py:927,1193`);
//! * the layer-wise split (`optim.py:250-316`, ported from RSI-Jev): decoder layers
//!   `0 .. lower_layers_n - 1` at `lower_lr_scale` times the schedule, everything else (the span
//!   head included) at 1.0; refused when it would scale nothing, or everything.

use crate::step::{ParamSpec, StepError};

/// AdamW's first-moment decay, fixed in `build_optimizer` (`betas = (0.9, beta2)`).
pub const BETA1: f64 = 0.9;
/// torch's default and every Lappi row's unless `--beta2` was given.
pub const DEFAULT_BETA2: f64 = 0.999;
/// `MasterWeightAdamW(eps=1e-8)`.
pub const EPS: f64 = 1e-8;
/// `MasterWeightAdamW(weight_decay=0.01)`, on every parameter.
pub const WEIGHT_DECAY: f64 = 0.01;
/// `QwenDecisionStep(max_grad_norm=1.0)`.
pub const MAX_GRAD_NORM: f64 = 1.0;
/// `torch.nn.utils.clip_grad_norm_`'s guard in `max_norm / (total_norm + 1e-6)`.
pub const CLIP_EPS: f64 = 1e-6;

/// The layer-wise learning-rate split (`--lower-layers-n`, `--lower-layers-lr-scale`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowerLayers {
    pub n: usize,
    pub lr_scale: f64,
}

/// The optimizer half of an FT recipe.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerRecipe {
    pub lr: f64,
    pub beta2: f64,
    pub lower_layers: Option<LowerLayers>,
}

impl OptimizerRecipe {
    pub fn validate(&self) -> Result<(), StepError> {
        if !(self.lr.is_finite() && self.lr > 0.0) {
            return Err(StepError::Contract(format!("lr must be positive and finite, got {}", self.lr)));
        }
        if !(self.beta2 > 0.0 && self.beta2 < 1.0) {
            return Err(StepError::Contract(format!("beta2 must be in (0, 1), got {}", self.beta2)));
        }
        if let Some(l) = self.lower_layers {
            if l.n == 0 {
                return Err(StepError::Contract(
                    "lower_layers_n must be at least 1; zero is the plain optimizer, so build no split".into(),
                ));
            }
            if !(l.lr_scale.is_finite() && l.lr_scale > 0.0) {
                return Err(StepError::Contract(format!(
                    "lower_lr_scale must be finite and positive, got {}. Zero would freeze the group \
                     while it still paid for moments",
                    l.lr_scale
                )));
            }
        }
        Ok(())
    }
}

/// The decoder-layer index in a parameter name: `(?:^|\.)layers\.(\d+)\.` (`optim.py:247`).
pub fn layer_index(name: &str) -> Option<usize> {
    let bytes = name.as_bytes();
    let needle = b"layers.";
    let mut from = 0;
    while let Some(off) = name[from..].find("layers.") {
        let at = from + off;
        let boundary = at == 0 || bytes[at - 1] == b'.';
        let digits_start = at + needle.len();
        let digits_len = bytes[digits_start..].iter().take_while(|b| b.is_ascii_digit()).count();
        if boundary && digits_len > 0 && bytes.get(digits_start + digits_len) == Some(&b'.') {
            return name[digits_start..digits_start + digits_len].parse().ok();
        }
        from = at + 1;
    }
    None
}

/// Per-entry learning-rate scales for the provider's entries, `layerwise_param_groups`'s rule:
/// `1.0` everywhere without a split; with one, `lr_scale` for every entry whose layer index is
/// below `n` (a name containing `visual` is never split), `1.0` for the rest. Refused when the
/// split matches nothing, covers every layer, or leaves the base group empty -- the host
/// parameters (the span head) join the base group, so `host_entries > 0` keeps it non-empty.
pub fn lr_scales(entries: &[ParamSpec], lower: Option<LowerLayers>, host_entries: usize) -> Result<Vec<f64>, StepError> {
    let Some(l) = lower else {
        return Ok(vec![1.0; entries.len()]);
    };
    let mut out = Vec::with_capacity(entries.len());
    let mut deepest: Option<usize> = None;
    let (mut lower_count, mut base_count) = (0usize, host_entries);
    for e in entries {
        match layer_index(&e.name).filter(|_| !e.name.contains("visual")) {
            Some(i) => {
                deepest = Some(deepest.map_or(i, |d| d.max(i)));
                if i < l.n {
                    out.push(l.lr_scale);
                    lower_count += 1;
                } else {
                    out.push(1.0);
                    base_count += 1;
                }
            }
            None => {
                out.push(1.0);
                base_count += 1;
            }
        }
    }
    if lower_count == 0 {
        return Err(StepError::Contract(format!(
            "lower_layers_n={} matched no trainable parameter: no name carries a 'layers.<i>.' \
             index. The recipe would record a split that scaled nothing.",
            l.n
        )));
    }
    if let Some(d) = deepest
        && l.n > d + 1
    {
        return Err(StepError::Contract(format!(
            "lower_layers_n={} but the deepest layer is {d}: the lower group would be the whole \
             tower, which is a global learning rate recorded as a layer-wise one",
            l.n
        )));
    }
    if base_count == 0 {
        return Err(StepError::Contract("every trainable parameter fell in the lower group".into()));
    }
    Ok(out)
}

/// Lappi's per-entry weight decay: [`WEIGHT_DECAY`] on every entry, whatever its name.
pub fn weight_decays(entries: &[ParamSpec]) -> Vec<f64> {
    vec![WEIGHT_DECAY; entries.len()]
}

/// `clip_grad_norm_`'s coefficient from the squared norm of every clipped gradient (the
/// provider's bank and the host parameters together): `max_norm / (norm + 1e-6)`, at most 1.
/// Refused when the norm is not finite: torch would go on and write NaN parameters; this loop
/// stops before any moment changes.
pub fn clip_coefficient(sq_norm: f64, max_norm: f64) -> Result<f64, StepError> {
    if !(sq_norm.is_finite() && sq_norm >= 0.0) {
        return Err(StepError::Contract(format!(
            "the gradient's squared norm is {sq_norm}: a non-finite gradient would write NaN into \
             every parameter and both moments"
        )));
    }
    let coef = max_norm / (sq_norm.sqrt() + CLIP_EPS);
    Ok(coef.min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specs(names: &[&str]) -> Vec<ParamSpec> {
        names.iter().map(|n| ParamSpec::new(*n, &[1])).collect()
    }

    #[test]
    fn the_layer_index_is_read_as_the_python_regex_reads_it() {
        assert_eq!(layer_index("layers.3.mlp.gate_proj.weight"), Some(3));
        assert_eq!(layer_index("model.layers.12.input_layernorm.weight"), Some(12));
        assert_eq!(layer_index("embed_tokens.weight"), None);
        assert_eq!(layer_index("norm.weight"), None);
        assert_eq!(layer_index("xlayers.3.w"), None, "a boundary is required before 'layers.'");
        assert_eq!(layer_index("layers.x.w"), None);
        assert_eq!(layer_index("layers.3"), None, "the index must be followed by a dot");
        assert_eq!(layer_index("sublayers.1.layers.4.w"), Some(4));
    }

    #[test]
    fn without_a_split_every_entry_trains_at_the_schedules_rate() {
        let e = specs(&["embed_tokens.weight", "layers.0.w"]);
        assert_eq!(lr_scales(&e, None, 0).unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn the_split_scales_the_lower_layers_and_refuses_a_split_that_means_nothing() {
        let e = specs(&["embed_tokens.weight", "layers.0.a", "layers.1.a", "layers.2.a", "norm.weight"]);
        let two = Some(LowerLayers { n: 2, lr_scale: 0.1 });
        assert_eq!(lr_scales(&e, two, 0).unwrap(), vec![1.0, 0.1, 0.1, 1.0, 1.0]);
        // Past the deepest layer: the "lower" group would be the whole tower.
        assert!(lr_scales(&e, Some(LowerLayers { n: 4, lr_scale: 0.1 }), 0).is_err());
        // n == deepest + 1 is admitted: the base keeps the embedding and the norm.
        assert!(lr_scales(&e, Some(LowerLayers { n: 3, lr_scale: 0.1 }), 0).is_ok());
        // No layer names at all.
        assert!(lr_scales(&specs(&["embed_tokens.weight"]), two, 1).is_err());
        // Every entry in the lower group and no host parameter to keep the base non-empty.
        let only = specs(&["layers.0.a", "layers.1.a"]);
        assert!(lr_scales(&only, two, 0).is_err());
        assert!(lr_scales(&only, two, 1).is_ok(), "the span head joins the base group");
        // A vision tower's layers are never split.
        let vis = specs(&["visual.layers.0.a", "layers.0.a", "norm.weight"]);
        assert_eq!(lr_scales(&vis, Some(LowerLayers { n: 1, lr_scale: 0.5 }), 0).unwrap(), vec![1.0, 0.5, 1.0]);
    }

    #[test]
    fn weight_decay_is_lappis_on_every_entry_not_tesslls_exclusions() {
        let e = specs(&["layers.0.input_layernorm.weight", "layers.0.linear_attn.dt_bias", "norm.weight"]);
        assert_eq!(weight_decays(&e), vec![0.01; 3]);
    }

    #[test]
    fn the_clip_coefficient_is_torchs_and_refuses_a_non_finite_norm() {
        assert_eq!(clip_coefficient(0.25, 1.0).unwrap(), 1.0, "a norm under the bound is not scaled");
        let c = clip_coefficient(16.0, 1.0).unwrap();
        assert_eq!(c, 1.0 / (4.0 + 1e-6));
        assert!(clip_coefficient(f64::NAN, 1.0).is_err());
        assert!(clip_coefficient(f64::INFINITY, 1.0).is_err());
    }
}
