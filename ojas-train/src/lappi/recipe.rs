//! Lappi's fine-tuning optimizer recipe, as the PyTorch trainer runs it, and the per-entry
//! vectors a [`super::step::StepProvider`] is driven with.
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

use super::step::{ParamSpec, StepError};

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
            return Err(StepError::Contract(format!(
                "lr must be positive and finite, got {}",
                self.lr
            )));
        }
        if !(self.beta2 > 0.0 && self.beta2 < 1.0) {
            return Err(StepError::Contract(format!(
                "beta2 must be in (0, 1), got {}",
                self.beta2
            )));
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
        let digits_len = bytes[digits_start..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count();
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
pub fn lr_scales(
    entries: &[ParamSpec],
    lower: Option<LowerLayers>,
    host_entries: usize,
) -> Result<Vec<f64>, StepError> {
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
        return Err(StepError::Contract(
            "every trainable parameter fell in the lower group".into(),
        ));
    }
    Ok(out)
}

/// Lappi's per-entry weight decay: [`WEIGHT_DECAY`] on every entry, whatever its name.
pub fn weight_decays(entries: &[ParamSpec]) -> Vec<f64> {
    vec![WEIGHT_DECAY; entries.len()]
}

/// One parameter's AdamW settings, as torch's param group holds them for it.
#[derive(Debug, Clone, PartialEq)]
pub struct EntryHyper {
    pub name: String,
    pub lr_scale: f64,
    pub weight_decay: f64,
    pub eps: f64,
    pub beta1: f64,
    pub beta2: f64,
}

/// The whole per-parameter optimizer table the loop runs: the provider's entries in their
/// order, then the host entries (the span head, which `layerwise_param_groups` puts in the
/// base group). Amendment 2 (ii) holds it equal, name by name and with no tolerance, to what
/// `layerwise_param_groups` + `build_optimizer` produce on the 2B
/// (`tests/optimizer_table.rs`), and the loop derives every vector it hands
/// [`super::step::StepProvider::adamw_step`] from it, so the table tested is the table run.
pub fn optimizer_table(
    entries: &[ParamSpec],
    host: &[ParamSpec],
    recipe: &OptimizerRecipe,
) -> Result<Vec<EntryHyper>, StepError> {
    recipe.validate()?;
    let scales = lr_scales(entries, recipe.lower_layers, host.len())?;
    let row = |name: &str, lr_scale: f64| EntryHyper {
        name: name.to_owned(),
        lr_scale,
        weight_decay: WEIGHT_DECAY,
        eps: EPS,
        beta1: BETA1,
        beta2: recipe.beta2,
    };
    Ok(entries
        .iter()
        .zip(&scales)
        .map(|(e, &s)| row(&e.name, s))
        .chain(host.iter().map(|e| row(&e.name, 1.0)))
        .collect())
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

// ---------------------------------------------------------------------------------------------
// The LoRA-2B plan (`python/qd_train/lora.py::lora_param_groups`, D6 in
// `campaign/lora-2b-preregistered.json`): adapters at the schedule's rate, D17 and the span head
// at `HEAD_LR_SCALE` of it, weight decay 0.01 on every trainable tensor, and no lower-layer
// split. The provider's entries are the *trainable* parameters, so every one of them must be an
// adapter leaf; the host entries must be D17 or a span-head tensor.
// ---------------------------------------------------------------------------------------------

/// `lora.HEAD_LR_SCALE = D17_LR / LORA_LR` = 1e-4 / 2e-4: D17 and the span head share the one
/// schedule at half the adapters' rate.
pub const HEAD_LR_SCALE: f64 = 0.5;
/// `lora.LORA_WEIGHT_DECAY`: on every trainable tensor, equal to [`WEIGHT_DECAY`] and stated
/// again because the pre-registration states it for this recipe.
pub const LORA_WEIGHT_DECAY: f64 = 0.01;
/// `lora.LORA_TARGET_SUFFIXES` (D6's table), matched on the dotted tail of a module name.
pub const LORA_TARGET_SUFFIXES: [&str; 10] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];
/// `lora.LORA_UNADAPTED_LINEAR_SUFFIXES`: GDN linears the table leaves out on purpose.
pub const LORA_UNADAPTED_LINEAR_SUFFIXES: [&str; 2] =
    ["linear_attn.in_proj_a", "linear_attn.in_proj_b"];

const LORA_A_LEAF: &str = ".lora_A.weight";
const LORA_B_LEAF: &str = ".lora_B.weight";

/// One of `lora_param_groups`' optimizer groups: `"lora"` (every adapter leaf), `"d17"` and
/// `"span_head"`, with the names of the parameters in it in the order they were given.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraGroup {
    pub name: &'static str,
    pub lr_scale: f64,
    pub weight_decay: f64,
    pub params: Vec<String>,
}

fn is_target(module: &str) -> bool {
    LORA_TARGET_SUFFIXES
        .iter()
        .any(|s| module == *s || module.ends_with(&format!(".{s}")))
}

/// A module's place in the canonical LoRA-A draw order: `(layer index, position in
/// [`LORA_TARGET_SUFFIXES`])`, read from `<prefix>layers.<n>.<suffix>`. Any other name is refused;
/// there is no fallback order. The place is the `layers` component directly before the suffix.
/// [`layer_index`] answers a different question (the first `layers.<n>` in a name, for the
/// learning-rate split) and is not this order.
pub fn draw_position(module: &str) -> Result<(u64, usize), StepError> {
    for (position, suffix) in LORA_TARGET_SUFFIXES.iter().enumerate() {
        let Some(head) = module
            .strip_suffix(suffix)
            .and_then(|h| h.strip_suffix('.'))
        else {
            continue;
        };
        let Some((before, index)) = head.rsplit_once('.') else {
            continue;
        };
        let named_layers = before == "layers" || before.ends_with(".layers");
        let all_digits =
            named_layers && !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit());
        // `parse` alone accepts a leading `+`, so the digit scan gates it.
        let parsed = if all_digits {
            index.parse::<u64>().ok()
        } else {
            None
        };
        if let Some(n) = parsed {
            return Ok((n, position));
        }
    }
    Err(StepError::Contract(format!(
        "{module:?} is not '<prefix>layers.<layer index>.<suffix>' for a suffix in D6's target \
         table: the LoRA-A draw order is (layer index, suffix-table position) and a module with \
         neither has no place in it"
    )))
}

/// `names` sorted into the canonical draw order (layer index ascending, then the suffix table),
/// as indices into `names`. Refuses a name with no draw position and two names that share one.
pub fn canonical_draw_order(names: &[&str]) -> Result<Vec<usize>, StepError> {
    let mut keyed = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        keyed.push((draw_position(name)?, i));
    }
    keyed.sort_unstable();
    for pair in keyed.windows(2) {
        if pair[0].0 == pair[1].0 {
            let first = names[pair[0].1];
            let second = names[pair[1].1];
            let layer = pair[0].0.0;
            let suffix = LORA_TARGET_SUFFIXES[pair[0].0.1];
            return Err(StepError::Contract(format!(
                "{first:?} and {second:?} are both layer {layer} {suffix:?}: the draw order is \
                 ambiguous"
            )));
        }
    }
    Ok(keyed.into_iter().map(|(_, i)| i).collect())
}

/// The adapted modules of a LoRA plan in the canonical draw order (layer index ascending, then
/// D6's suffix table), whatever order the provider listed the `A` leaves in: `(module,
/// in_features)` and the one rank every adapter shares. Refuses what [`lora_param_groups`]
/// refuses, and a module that has no layer index or shares a draw position with another.
pub fn lora_adapter_layout(
    entries: &[ParamSpec],
) -> Result<(usize, Vec<(String, usize)>), StepError> {
    let refuse = |m: String| Err(StepError::Contract(m));
    let mut a_of: Vec<(String, usize, usize)> = Vec::new();
    let mut b_of: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    for e in entries {
        let (module, is_a) = if let Some(m) = e.name.strip_suffix(LORA_A_LEAF) {
            (m, true)
        } else if let Some(m) = e.name.strip_suffix(LORA_B_LEAF) {
            (m, false)
        } else {
            return refuse(format!(
                "{} is trainable but is not an adapter leaf (<module>{LORA_A_LEAF} or \
                 <module>{LORA_B_LEAF}): a LoRA run trains the adapters, D17 and the span head \
                 over a frozen base, and a base parameter that trains is a different model",
                e.name
            ));
        };
        if LORA_UNADAPTED_LINEAR_SUFFIXES
            .iter()
            .any(|s| module == *s || module.ends_with(&format!(".{s}")))
        {
            return refuse(format!(
                "{}: {module} is a GDN gate that D6's table leaves unadapted on purpose \
                 (in_proj_a, in_proj_b)",
                e.name
            ));
        }
        if !is_target(module) {
            return refuse(format!(
                "{}: module {module} is not in D6's target table {LORA_TARGET_SUFFIXES:?}",
                e.name
            ));
        }
        let &[d0, d1] = e.shape.as_slice() else {
            return refuse(format!(
                "{}: an adapter leaf is a matrix, got shape {:?}",
                e.name, e.shape
            ));
        };
        if d0 == 0 || d1 == 0 {
            return refuse(format!("{}: shape {:?} has a zero axis", e.name, e.shape));
        }
        if is_a {
            if a_of.iter().any(|(m, _, _)| m == module) {
                return refuse(format!("{} is listed twice", e.name));
            }
            a_of.push((module.to_owned(), d0, d1));
        } else if b_of.insert(module.to_owned(), (d0, d1)).is_some() {
            return refuse(format!("{} is listed twice", e.name));
        }
    }
    if a_of.is_empty() {
        return refuse(
            "no adapter parameters: a LoRA run over a model that carries no lora_A/lora_B would \
             train nothing and report a run"
                .into(),
        );
    }
    let rank = a_of[0].1;
    for (module, r, _) in &a_of {
        let Some(&(out, br)) = b_of.get(module) else {
            return refuse(format!("{module} has a lora_A and no lora_B"));
        };
        if *r != rank || br != rank || out == 0 {
            return refuse(format!(
                "{module}: A has rank {r}, B has rank {br}; every adapter shares rank {rank}"
            ));
        }
    }
    if let Some(m) = b_of.keys().find(|m| !a_of.iter().any(|(a, _, _)| a == *m)) {
        return refuse(format!("{m} has a lora_B and no lora_A"));
    }
    // The draw order is canonical (layer index, then D6's suffix table), not the provider's.
    let names: Vec<&str> = a_of.iter().map(|(m, _, _)| m.as_str()).collect();
    let order = canonical_draw_order(&names)?;
    Ok((
        rank,
        order
            .into_iter()
            .map(|i| (a_of[i].0.clone(), a_of[i].2))
            .collect(),
    ))
}

/// The host-entry names a LoRA plan recognises, passed by the caller so this crate holds no
/// name of the caller's own modules: the one D17 entry and the span head's tensors.
#[derive(Debug, Clone, Copy)]
pub struct LoraHostNames<'a> {
    /// The single D17 host entry's name.
    pub d17_entry: &'a str,
    /// Every span-head tensor name.
    pub span_head: &'a [&'a str],
}

/// What `lora_param_groups` returns for these parameters, by name: `"lora"` (adapters, 1.0),
/// then `"d17"` and `"span_head"` (each [`HEAD_LR_SCALE`]) when the host holds them, all at
/// [`LORA_WEIGHT_DECAY`]. `names` says which host entries are D17 and which are span-head
/// tensors. Refused: no adapter, an adapter outside D6's table (the GDN gates included), half
/// an adapter, a module with no draw position, two modules at one draw position, a trainable
/// base parameter, a host entry that is neither D17 nor a span-head tensor, and any name given
/// twice.
pub fn lora_param_groups(
    entries: &[ParamSpec],
    host: &[ParamSpec],
    names: &LoraHostNames<'_>,
) -> Result<Vec<LoraGroup>, StepError> {
    lora_adapter_layout(entries)?;
    let mut seen = std::collections::BTreeSet::new();
    let (mut d17, mut head): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
    for h in host {
        if !seen.insert(h.name.as_str()) {
            return Err(StepError::Contract(format!(
                "host entry {} is listed twice",
                h.name
            )));
        }
        if h.name == names.d17_entry {
            d17.push(h.name.clone());
        } else if names.span_head.contains(&h.name.as_str()) {
            head.push(h.name.clone());
        } else {
            return Err(StepError::Contract(format!(
                "host entry {} is neither D17 ({}) nor a span-head tensor: a base \
                 parameter marked trainable under the lora recipe",
                h.name, names.d17_entry
            )));
        }
    }
    let group = |name, lr_scale, params| LoraGroup {
        name,
        lr_scale,
        weight_decay: LORA_WEIGHT_DECAY,
        params,
    };
    let mut out = vec![group(
        "lora",
        1.0,
        entries.iter().map(|e| e.name.clone()).collect(),
    )];
    if !d17.is_empty() {
        out.push(group("d17", HEAD_LR_SCALE, d17));
    }
    if !head.is_empty() {
        out.push(group("span_head", HEAD_LR_SCALE, head));
    }
    Ok(out)
}

/// The LoRA analogue of [`optimizer_table`]: the same per-entry rows (provider entries, then host
/// entries) with the scales of [`lora_param_groups`]. The 0.1x lower-layer split is refused here,
/// because D6 dropped it for LoRA: a layer index in an adapter's name scales nothing.
pub fn lora_optimizer_table(
    entries: &[ParamSpec],
    host: &[ParamSpec],
    recipe: &OptimizerRecipe,
    names: &LoraHostNames<'_>,
) -> Result<Vec<EntryHyper>, StepError> {
    recipe.validate()?;
    if recipe.lower_layers.is_some() {
        return Err(StepError::Contract(
            "the lora recipe has no lower-layer split (D6 dropped the 0.1x split for LoRA); \
             a split request under it would record a scale the adapters never train at"
                .into(),
        ));
    }
    let groups = lora_param_groups(entries, host, names)?;
    let scale_of = |name: &str| {
        groups
            .iter()
            .find(|g| g.params.iter().any(|p| p == name))
            .map_or(1.0, |g| g.lr_scale)
    };
    Ok(entries
        .iter()
        .chain(host)
        .map(|e| EntryHyper {
            name: e.name.clone(),
            lr_scale: scale_of(&e.name),
            weight_decay: LORA_WEIGHT_DECAY,
            eps: EPS,
            beta1: BETA1,
            beta2: recipe.beta2,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specs(names: &[&str]) -> Vec<ParamSpec> {
        names.iter().map(|n| ParamSpec::new(*n, &[1])).collect()
    }

    #[test]
    fn lora_groups_classify_host_entries_by_the_names_the_caller_passes() {
        let names = LoraHostNames {
            d17_entry: "host.d17",
            span_head: &["host.head.a", "host.head.b"],
        };
        let adapters = vec![
            ParamSpec::new("layers.0.self_attn.q_proj.lora_A.weight", &[2, 4]),
            ParamSpec::new("layers.0.self_attn.q_proj.lora_B.weight", &[3, 2]),
        ];
        let host = specs(&["host.d17", "host.head.b", "host.head.a"]);
        let groups = lora_param_groups(&adapters, &host, &names).unwrap();
        let by_name = |n: &str| groups.iter().find(|g| g.name == n).unwrap().params.clone();
        assert_eq!(by_name("d17"), vec!["host.d17"]);
        assert_eq!(by_name("span_head"), vec!["host.head.b", "host.head.a"]);
        // A name the caller did not pass is a base parameter marked trainable.
        let stray = specs(&["host.other"]);
        assert!(lora_param_groups(&adapters, &stray, &names).is_err());
        // The same entry set under different names is classified by those names.
        let renamed = LoraHostNames {
            d17_entry: "host.other",
            span_head: &[],
        };
        assert!(lora_param_groups(&adapters, &stray, &renamed).is_ok());
    }

    #[test]
    fn the_layer_index_is_read_as_the_python_regex_reads_it() {
        assert_eq!(layer_index("layers.3.mlp.gate_proj.weight"), Some(3));
        assert_eq!(
            layer_index("model.layers.12.input_layernorm.weight"),
            Some(12)
        );
        assert_eq!(layer_index("embed_tokens.weight"), None);
        assert_eq!(layer_index("norm.weight"), None);
        assert_eq!(
            layer_index("xlayers.3.w"),
            None,
            "a boundary is required before 'layers.'"
        );
        assert_eq!(layer_index("layers.x.w"), None);
        assert_eq!(
            layer_index("layers.3"),
            None,
            "the index must be followed by a dot"
        );
        assert_eq!(layer_index("sublayers.1.layers.4.w"), Some(4));
    }

    #[test]
    fn without_a_split_every_entry_trains_at_the_schedules_rate() {
        let e = specs(&["embed_tokens.weight", "layers.0.w"]);
        assert_eq!(lr_scales(&e, None, 0).unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn the_split_scales_the_lower_layers_and_refuses_a_split_that_means_nothing() {
        let e = specs(&[
            "embed_tokens.weight",
            "layers.0.a",
            "layers.1.a",
            "layers.2.a",
            "norm.weight",
        ]);
        let two = Some(LowerLayers {
            n: 2,
            lr_scale: 0.1,
        });
        assert_eq!(
            lr_scales(&e, two, 0).unwrap(),
            vec![1.0, 0.1, 0.1, 1.0, 1.0]
        );
        // Past the deepest layer: the "lower" group would be the whole tower.
        assert!(
            lr_scales(
                &e,
                Some(LowerLayers {
                    n: 4,
                    lr_scale: 0.1
                }),
                0
            )
            .is_err()
        );
        // n == deepest + 1 is admitted: the base keeps the embedding and the norm.
        assert!(
            lr_scales(
                &e,
                Some(LowerLayers {
                    n: 3,
                    lr_scale: 0.1
                }),
                0
            )
            .is_ok()
        );
        // No layer names at all.
        assert!(lr_scales(&specs(&["embed_tokens.weight"]), two, 1).is_err());
        // Every entry in the lower group and no host parameter to keep the base non-empty.
        let only = specs(&["layers.0.a", "layers.1.a"]);
        assert!(lr_scales(&only, two, 0).is_err());
        assert!(
            lr_scales(&only, two, 1).is_ok(),
            "the span head joins the base group"
        );
        // A vision tower's layers are never split.
        let vis = specs(&["visual.layers.0.a", "layers.0.a", "norm.weight"]);
        assert_eq!(
            lr_scales(
                &vis,
                Some(LowerLayers {
                    n: 1,
                    lr_scale: 0.5
                }),
                0
            )
            .unwrap(),
            vec![1.0, 0.5, 1.0]
        );
    }

    #[test]
    fn weight_decay_is_lappis_on_every_entry_not_tesslls_exclusions() {
        let e = specs(&[
            "layers.0.input_layernorm.weight",
            "layers.0.linear_attn.dt_bias",
            "norm.weight",
        ]);
        assert_eq!(weight_decays(&e), vec![0.01; 3]);
    }

    #[test]
    fn the_clip_coefficient_is_torchs_and_refuses_a_non_finite_norm() {
        assert_eq!(
            clip_coefficient(0.25, 1.0).unwrap(),
            1.0,
            "a norm under the bound is not scaled"
        );
        let c = clip_coefficient(16.0, 1.0).unwrap();
        assert_eq!(c, 1.0 / (4.0 + 1e-6));
        assert!(clip_coefficient(f64::NAN, 1.0).is_err());
        assert!(clip_coefficient(f64::INFINITY, 1.0).is_err());
    }
}
