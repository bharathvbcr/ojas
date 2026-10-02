//! Per-parameter learning-rate scales and weight decays, from the caller's
//! group spec. Nothing here has a default: which parameters decay, and which
//! learn at a reduced rate, is the recipe's data, never this crate's (tessl's
//! `default_weight_decay` and `excluded_from_weight_decay` are deliberately
//! not called).
//!
//! A spec is two rule lists. Like torch's `param_groups`, each list is a
//! partition: every parameter matches exactly one learning-rate rule and
//! exactly one weight-decay rule, and every rule matches at least one
//! parameter. Anything else is refused by name, so an overlap, a gap or a
//! rule that a typo made dead cannot change the optimizer silently.

use crate::error::{Qwen35Error, Result};
use crate::names::TensorSpec;

/// Which parameters a rule takes, by transformers' name below the tower
/// prefix (`layers.3.linear_attn.dt_bias`).
#[derive(Clone, Debug, PartialEq)]
pub enum Select {
    All,
    /// `embed_tokens.weight` (the tied embedding and LM head).
    Embedding,
    /// `norm.weight`, the final norm.
    FinalNorm,
    /// Every `layers.{l}.*` with `l < n`.
    LayersBelow(u32),
    /// Every `layers.{l}.*` with `l >= n`.
    LayersFrom(u32),
    /// Some `.`-separated segment equals this (`"norm"` takes
    /// `linear_attn.norm.weight` and `norm.weight`).
    Segment(String),
    /// Some segment ends with this (`"_norm"` takes `q_norm`, `k_norm`).
    SegmentSuffix(String),
    /// The name contains this anywhere (`"layernorm"`, `"bias"`).
    Contains(String),
    /// Exactly this name.
    Name(String),
    AnyOf(Vec<Select>),
    AllOf(Vec<Select>),
    Not(Box<Select>),
}

impl Select {
    pub fn matches(&self, t: &TensorSpec) -> bool {
        let name = t.name.as_str();
        match self {
            Select::All => true,
            Select::Embedding => name == "embed_tokens.weight",
            Select::FinalNorm => name == "norm.weight",
            Select::LayersBelow(n) => t.layer().is_some_and(|l| l < *n),
            Select::LayersFrom(n) => t.layer().is_some_and(|l| l >= *n),
            Select::Segment(s) => name.split('.').any(|seg| seg == s),
            Select::SegmentSuffix(s) => name.split('.').any(|seg| seg.ends_with(s.as_str())),
            Select::Contains(s) => name.contains(s.as_str()),
            Select::Name(s) => name == s,
            Select::AnyOf(v) => v.iter().any(|s| s.matches(t)),
            Select::AllOf(v) => v.iter().all(|s| s.matches(t)),
            Select::Not(s) => !s.matches(t),
        }
    }
}

/// Parameters `select` takes learn at `lr * lr_scale`.
#[derive(Clone, Debug, PartialEq)]
pub struct LrRule {
    pub label: String,
    pub select: Select,
    pub lr_scale: f64,
}

/// Parameters `select` takes decay at `weight_decay` (decoupled, AdamW's).
/// f32 because tessl's AdamW takes its per-entry decays as f32.
#[derive(Clone, Debug, PartialEq)]
pub struct WdRule {
    pub label: String,
    pub select: Select,
    pub weight_decay: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroupSpec {
    pub lr: Vec<LrRule>,
    pub weight_decay: Vec<WdRule>,
}

/// One cell of the partition the spec induces: the parameters that share a
/// learning-rate rule and a weight-decay rule. For a ledger row.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupSummary {
    pub lr_label: String,
    pub wd_label: String,
    pub lr_scale: f64,
    pub weight_decay: f32,
    pub tensors: usize,
    pub elements: u64,
}

/// Per-entry vectors, aligned with the table they were built against.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizerPlan {
    names: Vec<String>,
    lr_scale: Vec<f64>,
    weight_decay: Vec<f32>,
    groups: Vec<GroupSummary>,
}

fn which<R>(
    rules: &[R],
    label: impl Fn(&R) -> &str,
    select: impl Fn(&R) -> &Select,
    t: &TensorSpec,
    kind: &str,
) -> Result<usize> {
    let hits: Vec<usize> = (0..rules.len())
        .filter(|&i| select(&rules[i]).matches(t))
        .collect();
    match hits.as_slice() {
        [one] => Ok(*one),
        [] => Err(Qwen35Error::Invalid {
            op: "OptimizerPlan::build",
            detail: format!(
                "{} matches no {kind} rule; every parameter needs exactly one",
                t.name
            ),
        }),
        many => Err(Qwen35Error::Invalid {
            op: "OptimizerPlan::build",
            detail: format!(
                "{} matches {kind} rules {:?}; every parameter needs exactly one",
                t.name,
                many.iter().map(|&i| label(&rules[i])).collect::<Vec<_>>()
            ),
        }),
    }
}

fn check_labels<'a>(labels: impl Iterator<Item = &'a str>, kind: &str) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for l in labels {
        if l.is_empty() || !seen.insert(l) {
            return Err(Qwen35Error::Invalid {
                op: "OptimizerPlan::build",
                detail: format!("{kind} rule label {l:?} is empty or repeated"),
            });
        }
    }
    if seen.is_empty() {
        return Err(Qwen35Error::Invalid {
            op: "OptimizerPlan::build",
            detail: format!("no {kind} rules; the spec must say what every parameter takes"),
        });
    }
    Ok(())
}

impl OptimizerPlan {
    /// The per-entry vectors of `spec` over `table` (a provider's live
    /// parameter table, or [`crate::names::tower_tensors`] on the CPU).
    pub fn build(table: &[TensorSpec], spec: &GroupSpec) -> Result<Self> {
        check_labels(spec.lr.iter().map(|r| r.label.as_str()), "learning-rate")?;
        check_labels(
            spec.weight_decay.iter().map(|r| r.label.as_str()),
            "weight-decay",
        )?;
        for r in &spec.lr {
            if !(r.lr_scale.is_finite() && r.lr_scale >= 0.0) {
                return Err(Qwen35Error::Invalid {
                    op: "OptimizerPlan::build",
                    detail: format!(
                        "lr rule {:?}: lr_scale {} must be finite and >= 0",
                        r.label, r.lr_scale
                    ),
                });
            }
        }
        for r in &spec.weight_decay {
            if !(r.weight_decay.is_finite() && r.weight_decay >= 0.0) {
                return Err(Qwen35Error::Invalid {
                    op: "OptimizerPlan::build",
                    detail: format!(
                        "wd rule {:?}: weight_decay {} must be finite and >= 0",
                        r.label, r.weight_decay
                    ),
                });
            }
        }
        if table.is_empty() {
            return Err(Qwen35Error::Invalid {
                op: "OptimizerPlan::build",
                detail: "the parameter table is empty".into(),
            });
        }
        let mut lr_hits = vec![0usize; spec.lr.len()];
        let mut wd_hits = vec![0usize; spec.weight_decay.len()];
        let mut lr_scale = Vec::with_capacity(table.len());
        let mut weight_decay = Vec::with_capacity(table.len());
        let mut groups: Vec<GroupSummary> = Vec::new();
        for t in table {
            let li = which(&spec.lr, |r| &r.label, |r| &r.select, t, "learning-rate")?;
            let wi = which(
                &spec.weight_decay,
                |r| &r.label,
                |r| &r.select,
                t,
                "weight-decay",
            )?;
            lr_hits[li] += 1;
            wd_hits[wi] += 1;
            let (lr, wd) = (&spec.lr[li], &spec.weight_decay[wi]);
            lr_scale.push(lr.lr_scale);
            weight_decay.push(wd.weight_decay);
            match groups
                .iter_mut()
                .find(|g| g.lr_label == lr.label && g.wd_label == wd.label)
            {
                Some(g) => {
                    g.tensors += 1;
                    g.elements += t.numel() as u64;
                }
                None => groups.push(GroupSummary {
                    lr_label: lr.label.clone(),
                    wd_label: wd.label.clone(),
                    lr_scale: lr.lr_scale,
                    weight_decay: wd.weight_decay,
                    tensors: 1,
                    elements: t.numel() as u64,
                }),
            }
        }
        for (hits, label, kind) in spec
            .lr
            .iter()
            .zip(&lr_hits)
            .map(|(r, &h)| (h, &r.label, "learning-rate"))
            .chain(
                spec.weight_decay
                    .iter()
                    .zip(&wd_hits)
                    .map(|(r, &h)| (h, &r.label, "weight-decay")),
            )
        {
            if hits == 0 {
                return Err(Qwen35Error::Invalid {
                    op: "OptimizerPlan::build",
                    detail: format!("{kind} rule {label:?} matches no parameter"),
                });
            }
        }
        Ok(Self {
            names: table.iter().map(|t| t.name.clone()).collect(),
            lr_scale,
            weight_decay,
            groups,
        })
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn lr_scale(&self) -> &[f64] {
        &self.lr_scale
    }

    pub fn weight_decay(&self) -> &[f32] {
        &self.weight_decay
    }

    pub fn groups(&self) -> &[GroupSummary] {
        &self.groups
    }

    /// Whether tessl's AdamW, as it stands, can run this plan: it takes one
    /// learning rate for every entry and has no per-entry scale, so any
    /// `lr_scale` other than exactly 1.0 is refused here, before any device
    /// work. Folding a uniform scale into the learning rate is left to the
    /// caller, who owns the learning rate.
    pub fn check_tessl_lr(&self) -> Result<()> {
        if let Some((i, s)) = self.lr_scale.iter().enumerate().find(|(_, &s)| s != 1.0) {
            return Err(Qwen35Error::Unsupported {
                what: format!(
                    "per-parameter learning-rate scale {s} on {} (and every other entry not at 1.0)",
                    self.names[i]
                ),
                needs: "a per-entry lr_scale in tessl's Qwen35Model::adamw_step (in progress on tessl branch \
                        lappi-train-lrscale-mrope; not on the tessl this crate builds against)"
                    .into(),
            });
        }
        Ok(())
    }

    /// The plan was built against exactly `table` (names, in order).
    pub(crate) fn check_table(&self, table: &[TensorSpec]) -> Result<()> {
        if self.names.len() != table.len()
            || self.names.iter().zip(table).any(|(n, t)| *n != t.name)
        {
            return Err(Qwen35Error::Invalid {
                op: "adamw_step",
                detail:
                    "the optimizer plan was built against another parameter table; rebuild it from \
                         Qwen35Step::parameter_table"
                        .into(),
            });
        }
        Ok(())
    }
}
