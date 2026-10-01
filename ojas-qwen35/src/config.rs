//! `config.json` → the Qwen3.5 text tower this provider trains, or a refusal
//! that names the field.
//!
//! Two parsers read the same text, on purpose. tessl's
//! `Qwen35Config::from_config_json` stays the canonical constructor of the
//! model's shape (and keeps its own refusals); this module adds what a
//! training provider must refuse that tessl does not, or refuses only later:
//!
//! - **every key is known.** A key outside the tables below is refused by
//!   name. The tables say which keys are read, which must hold one value, and
//!   which do not touch the text tower's training step (and why);
//! - GDN value heads equal to key heads (tessl refuses only inside a step:
//!   `gdn_train` has no head grouping);
//! - query heads a non-zero multiple of KV heads (tessl checks only inside
//!   `AttnTrainDims::validate`);
//! - `head_dim` and the GDN key dim the training kernels are compiled for,
//!   and a GDN value dim their tile divides;
//! - `attention_dropout` 0 (a training step with dropout is another step);
//! - `mamba_ssm_dtype` `float32` (tessl's GDN state is f32);
//! - the conv width tessl's `conv1d_silu` accepts (2..=8, `qwen35.rs`);
//! - `rope_parameters`: `rope_type` `default`, and the MRoPE fields tessl
//!   neither reads nor refuses. `mrope_section` is accepted only when its
//!   entries are positive and sum to `rotary_dim / 2`: then, on text-only
//!   input (one position stream copied to t, h and w), every rotary
//!   frequency's angle is `position * inv_freq` whichever stream the section
//!   assigns it to, interleaved or not, which is the plain partial RoPE tessl
//!   computes. Text-only is enforced per sequence: the vision special-token ids
//!   this config declares are refused in input ([`Qwen35TextConfig::reserved_token_ids`]).
//!
//! Then both parsers' shapes are compared field by field; any disagreement is
//! refused.

use crate::error::{Qwen35Error, Result};
use crate::json::{self, Json};

/// Which mixer a layer runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// The gated delta net (`"linear_attention"`).
    LinearAttention,
    /// Gated softmax attention (`"full_attention"`).
    FullAttention,
}

/// The MRoPE fields, as read. See the module docs for when they are accepted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mrope {
    pub section: Vec<u32>,
    pub interleaved: Option<bool>,
}

/// The text tower's shape, validated against what tessl's training kernels
/// implement. Built only by [`Qwen35TextConfig::from_json`].
#[derive(Clone, Debug)]
pub struct Qwen35TextConfig {
    pub hidden: u32,
    pub intermediate: u32,
    pub vocab: u32,
    pub layers: Vec<LayerKind>,
    pub gdn_key_heads: u32,
    pub gdn_value_heads: u32,
    pub gdn_key_dim: u32,
    pub gdn_value_dim: u32,
    pub conv_kernel: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// `head_dim * partial_rotary_factor`.
    pub rotary_dim: u32,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub mrope: Option<Mrope>,
    /// Longest sequence the config declares; longer ones are refused.
    pub max_position_embeddings: Option<u64>,
    /// Where the tower's tensors live in the checkpoint:
    /// `"model.language_model."` under a `text_config`, `"model."` for a
    /// text-only config.
    pub tower_prefix: String,
    /// Vision special tokens the root config declares (`image_token_id`, ...).
    /// They switch transformers to multimodal positions, so a sequence
    /// holding one is refused.
    pub reserved_token_ids: Vec<(String, u32)>,
    pub(crate) tessl: tessl::qwen35_model::Qwen35Config,
}

/// `text_config` keys this module reads (each is checked below).
const TEXT_READ: &[&str] = &[
    "attention_bias",
    "attention_dropout",
    "attn_output_gate",
    "full_attention_interval",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "intermediate_size",
    "layer_types",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "mamba_ssm_dtype",
    "max_position_embeddings",
    "mlp_only_layers",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "partial_rotary_factor",
    "rms_norm_eps",
    "rope_parameters",
    "rope_scaling",
    "rope_theta",
    "rope_type",
    "tie_word_embeddings",
    "vocab_size",
];

/// `text_config` keys that do not change the text tower's training step.
const TEXT_INERT: &[(&str, &str)] = &[
    ("architectures", "class names; the tower is the same"),
    ("bos_token_id", "tokenizer metadata"),
    ("eos_token_id", "tokenizer metadata"),
    ("pad_token_id", "tokenizer metadata"),
    ("dtype", "checkpoint storage dtype; every weight is widened to f32"),
    ("torch_dtype", "checkpoint storage dtype; every weight is widened to f32"),
    ("initializer_range", "initialisation only; every weight comes from the checkpoint"),
    ("use_cache", "the inference KV cache"),
    ("transformers_version", "provenance"),
    (
        "mtp_num_hidden_layers",
        "the MTP block is outside the text tower's forward and loss; its mtp.* tensors are not loaded",
    ),
    (
        "mtp_use_dedicated_embeddings",
        "the MTP block is outside the text tower's forward and loss; its mtp.* tensors are not loaded",
    ),
];

/// Root keys of a `Qwen3_5ForConditionalGeneration` config this module reads.
const ROOT_READ: &[&str] = &[
    "image_token_id",
    "model_type",
    "text_config",
    "tie_word_embeddings",
    "video_token_id",
    "vision_end_token_id",
    "vision_start_token_id",
];

/// Root keys of that config that do not change the text tower's step.
const ROOT_INERT: &[(&str, &str)] = &[
    ("architectures", "class names; the tower is the same"),
    ("transformers_version", "provenance"),
    ("dtype", "checkpoint storage dtype; every weight is widened to f32"),
    ("torch_dtype", "checkpoint storage dtype; every weight is widened to f32"),
    (
        "vision_config",
        "the vision tower: text-only input never runs it, and its model.visual.* tensors are not loaded",
    ),
];

/// The vision special tokens a root config may declare.
const RESERVED_TOKEN_KEYS: &[&str] = &[
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
];

/// `rope_parameters` keys this module reads; any other is refused.
const ROPE_KEYS: &[&str] = &[
    "mrope_interleaved",
    "mrope_section",
    "partial_rotary_factor",
    "rope_theta",
    "rope_type",
];

/// tessl's `conv1d_silu` refuses widths outside this range
/// (`tessl/src/qwen35.rs`, the `kernel_width must be 2..=8` check).
const CONV_KERNEL_RANGE: std::ops::RangeInclusive<u32> = 2..=8;

fn cfg_err(detail: impl Into<String>) -> Qwen35Error {
    Qwen35Error::Config(detail.into())
}

fn refuse_unknown_keys(obj: &Json, read: &[&str], inert: &[(&str, &str)], where_: &str) -> Result<()> {
    let fields = obj
        .fields()
        .ok_or_else(|| cfg_err(format!("{where_} is not an object")))?;
    for (k, _) in fields {
        if !read.contains(&k.as_str()) && !inert.iter().any(|(i, _)| i == k) {
            return Err(cfg_err(format!(
                "{where_}.{k} is not a key this provider knows; refusing rather than ignoring it"
            )));
        }
    }
    Ok(())
}

fn uint_of(v: &Json, path: &str) -> Result<u64> {
    match v {
        Json::Num { uint: Some(n), .. } => Ok(*n),
        other => Err(cfg_err(format!("{path} must be a non-negative integer, got {}", other.kind()))),
    }
}

/// A required positive integer that fits `u32`.
fn dim(obj: &Json, key: &str) -> Result<u32> {
    let v = obj
        .get(key)
        .ok_or_else(|| cfg_err(format!("text_config.{key} is missing")))?;
    let n = uint_of(v, &format!("text_config.{key}"))?;
    if n == 0 {
        return Err(cfg_err(format!("text_config.{key} is 0")));
    }
    u32::try_from(n).map_err(|_| cfg_err(format!("text_config.{key} = {n} exceeds u32")))
}

fn float_of(v: &Json, path: &str) -> Result<f64> {
    match v {
        Json::Num { value, .. } => Ok(*value),
        other => Err(cfg_err(format!("{path} must be a number, got {}", other.kind()))),
    }
}

fn require(obj: &Json, key: &str, want: &Json, why: &str) -> Result<()> {
    let got = obj
        .get(key)
        .ok_or_else(|| cfg_err(format!("text_config.{key} is missing ({why})")))?;
    let same = match (got, want) {
        (Json::Num { value: a, .. }, Json::Num { value: b, .. }) => a == b,
        (a, b) => a == b,
    };
    if !same {
        return Err(cfg_err(format!(
            "text_config.{key} is {}; this provider implements only {} ({why})",
            got.kind(),
            want.kind()
        )));
    }
    Ok(())
}

impl Qwen35TextConfig {
    /// Read `path` (a snapshot's `config.json`).
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).map_err(|e| Qwen35Error::Io(format!("{}: {e}", path.display())))?;
        Self::from_json(&text).map_err(|e| match e {
            Qwen35Error::Config(d) => Qwen35Error::Config(format!("{}: {d}", path.display())),
            other => other,
        })
    }

    /// Parse and validate. The root may be a `Qwen3_5ForConditionalGeneration`
    /// config (with `text_config`) or a text-only `qwen3_5_text` config.
    pub fn from_json(text: &str) -> Result<Self> {
        let root = json::parse(text, "config.json").map_err(cfg_err)?;
        if root.fields().is_none() {
            return Err(cfg_err("the root is not an object"));
        }
        let (tc, tower_prefix, reserved_token_ids) = match root.get("text_config") {
            Some(tc) => {
                refuse_unknown_keys(&root, ROOT_READ, ROOT_INERT, "config")?;
                match root.get("model_type") {
                    Some(Json::Str(s)) if s == "qwen3_5" => {}
                    other => {
                        return Err(cfg_err(format!(
                            "config.model_type is {}; this provider reads \"qwen3_5\" with a text_config",
                            other.map_or("missing".into(), Json::kind)
                        )))
                    }
                }
                let mut reserved = Vec::new();
                for &k in RESERVED_TOKEN_KEYS {
                    if let Some(v) = root.get(k) {
                        let id = uint_of(v, &format!("config.{k}"))?;
                        let id = u32::try_from(id).map_err(|_| cfg_err(format!("config.{k} = {id} exceeds u32")))?;
                        reserved.push((k.to_string(), id));
                    }
                }
                (tc, "model.language_model.".to_string(), reserved)
            }
            None => (&root, "model.".to_string(), Vec::new()),
        };
        refuse_unknown_keys(tc, TEXT_READ, TEXT_INERT, "text_config")?;
        match tc.get("model_type") {
            Some(Json::Str(s)) if s == "qwen3_5_text" => {}
            other => {
                return Err(cfg_err(format!(
                    "text_config.model_type is {}; this provider reads \"qwen3_5_text\"",
                    other.map_or("missing".into(), Json::kind)
                )))
            }
        }

        // Semantics the step hard-codes.
        let s = |v: &str| Json::Str(v.into());
        require(tc, "hidden_act", &s("silu"), "the MLP is SwiGLU")?;
        require(tc, "attention_bias", &Json::Bool(false), "the fused projections have no bias")?;
        require(tc, "attn_output_gate", &Json::Bool(true), "the attention output is gated")?;
        if tc.get("attention_dropout").is_some() {
            require(
                tc,
                "attention_dropout",
                &Json::Num { value: 0.0, uint: Some(0) },
                "a training step with dropout is another step",
            )?;
        }
        if tc.get("mamba_ssm_dtype").is_some() {
            require(tc, "mamba_ssm_dtype", &s("float32"), "tessl's GDN state is f32")?;
        }
        match tc.get("mlp_only_layers") {
            None => {}
            Some(Json::Array(a)) if a.is_empty() => {}
            Some(v) => return Err(cfg_err(format!("text_config.mlp_only_layers is {}; every layer has a mixer here", v.kind()))),
        }
        match tc.get("rope_scaling") {
            None | Some(Json::Null) => {}
            Some(v) => return Err(cfg_err(format!("text_config.rope_scaling is {}; only plain RoPE is implemented", v.kind()))),
        }
        let ties: Vec<&Json> = [root.get("tie_word_embeddings"), tc.get("tie_word_embeddings")]
            .into_iter()
            .flatten()
            .collect();
        if ties.is_empty() {
            return Err(cfg_err("tie_word_embeddings is not set; the step needs the tied LM head"));
        }
        if let Some(t) = ties.iter().find(|t| **t != &Json::Bool(true)) {
            return Err(cfg_err(format!(
                "tie_word_embeddings is {}; an untied LM head is not implemented",
                t.kind()
            )));
        }

        let hidden = dim(tc, "hidden_size")?;
        let intermediate = dim(tc, "intermediate_size")?;
        let vocab = dim(tc, "vocab_size")?;
        let n_layers = dim(tc, "num_hidden_layers")?;
        let interval = match tc.get("full_attention_interval") {
            Some(_) => Some(dim(tc, "full_attention_interval")?),
            None => None,
        };
        let layers: Vec<LayerKind> = match tc.get("layer_types") {
            Some(Json::Array(a)) => a
                .iter()
                .enumerate()
                .map(|(i, t)| match t {
                    Json::Str(s) if s == "linear_attention" => Ok(LayerKind::LinearAttention),
                    Json::Str(s) if s == "full_attention" => Ok(LayerKind::FullAttention),
                    t => Err(cfg_err(format!("text_config.layer_types[{i}] is {}, not a known layer type", t.kind()))),
                })
                .collect::<Result<_>>()?,
            Some(v) => return Err(cfg_err(format!("text_config.layer_types is {}, not an array", v.kind()))),
            None => {
                let every = interval.ok_or_else(|| cfg_err("neither layer_types nor full_attention_interval is set"))?;
                (0..n_layers)
                    .map(|l| {
                        if (l + 1) % every == 0 {
                            LayerKind::FullAttention
                        } else {
                            LayerKind::LinearAttention
                        }
                    })
                    .collect()
            }
        };
        if layers.len() != n_layers as usize {
            return Err(cfg_err(format!(
                "text_config.layer_types has {} entries for num_hidden_layers {n_layers}",
                layers.len()
            )));
        }
        if let Some(every) = interval {
            for (l, k) in layers.iter().enumerate() {
                let full = (l as u32 + 1) % every == 0;
                if full != (*k == LayerKind::FullAttention) {
                    return Err(cfg_err(format!(
                        "text_config.layer_types[{l}] disagrees with full_attention_interval {every}"
                    )));
                }
            }
        }

        // GDN.
        let gdn_key_dim = dim(tc, "linear_key_head_dim")?;
        if gdn_key_dim != tessl::gdn_train::GDN_TRAIN_DK || gdn_key_dim != tessl::qwen35::GDN_KEY_DIM {
            return Err(cfg_err(format!(
                "text_config.linear_key_head_dim {gdn_key_dim} is not supported: the GDN kernels are compiled for {}",
                tessl::gdn_train::GDN_TRAIN_DK
            )));
        }
        let gdn_key_heads = dim(tc, "linear_num_key_heads")?;
        let gdn_value_heads = dim(tc, "linear_num_value_heads")?;
        if gdn_value_heads != gdn_key_heads {
            return Err(cfg_err(format!(
                "text_config.linear_num_value_heads {gdn_value_heads} != linear_num_key_heads {gdn_key_heads}: \
                 tessl's gdn_train has no head grouping, so value heads must equal key heads"
            )));
        }
        let gdn_value_dim = dim(tc, "linear_value_head_dim")?;
        if gdn_value_dim % tessl::gdn_train::GDN_TRAIN_BV != 0 {
            return Err(cfg_err(format!(
                "text_config.linear_value_head_dim {gdn_value_dim} is not a multiple of {}, the GDN training tile",
                tessl::gdn_train::GDN_TRAIN_BV
            )));
        }
        let conv_kernel = dim(tc, "linear_conv_kernel_dim")?;
        if !CONV_KERNEL_RANGE.contains(&conv_kernel) {
            return Err(cfg_err(format!(
                "text_config.linear_conv_kernel_dim {conv_kernel} is outside {CONV_KERNEL_RANGE:?}, the widths conv1d_silu runs"
            )));
        }

        // Attention.
        let q_heads = dim(tc, "num_attention_heads")?;
        let kv_heads = dim(tc, "num_key_value_heads")?;
        if q_heads % kv_heads != 0 {
            return Err(cfg_err(format!(
                "text_config.num_attention_heads {q_heads} is not a multiple of num_key_value_heads {kv_heads}: \
                 grouped-query attention needs a whole number of query heads per KV head"
            )));
        }
        let head_dim = dim(tc, "head_dim")?;
        if head_dim != tessl::attn_train::ATTN_TRAIN_HEAD_DIM {
            return Err(cfg_err(format!(
                "text_config.head_dim {head_dim} is not supported: the attention training kernels are compiled for {}",
                tessl::attn_train::ATTN_TRAIN_HEAD_DIM
            )));
        }

        // RoPE: rope_parameters (transformers 5), else the flat fields.
        let (theta, factor, mrope) = match tc.get("rope_parameters") {
            Some(rope) => {
                refuse_unknown_keys(rope, ROPE_KEYS, &[], "text_config.rope_parameters")?;
                let rt = rope.get("rope_type");
                let theta = float_of(
                    rope.get("rope_theta").ok_or_else(|| cfg_err("text_config.rope_parameters.rope_theta is missing"))?,
                    "text_config.rope_parameters.rope_theta",
                )?;
                let factor = match rope.get("partial_rotary_factor") {
                    Some(v) => float_of(v, "text_config.rope_parameters.partial_rotary_factor")?,
                    None => 1.0,
                };
                check_rope_type(rt, "text_config.rope_parameters.rope_type")?;
                // Flat copies beside rope_parameters must agree with it.
                for (k, want) in [("rope_theta", theta), ("partial_rotary_factor", factor)] {
                    if let Some(v) = tc.get(k) {
                        let got = float_of(v, &format!("text_config.{k}"))?;
                        if got != want {
                            return Err(cfg_err(format!(
                                "text_config.{k} = {got} disagrees with rope_parameters.{k} = {want}"
                            )));
                        }
                    }
                }
                if let Some(t) = tc.get("rope_type") {
                    check_rope_type(Some(t), "text_config.rope_type")?;
                }
                let mrope = match (rope.get("mrope_section"), rope.get("mrope_interleaved")) {
                    (None, None) => None,
                    (None, Some(_)) => {
                        return Err(cfg_err("rope_parameters.mrope_interleaved is set without mrope_section"))
                    }
                    (Some(sec), inter) => {
                        let section = match sec {
                            Json::Array(a) => a
                                .iter()
                                .map(|v| {
                                    let n = uint_of(v, "rope_parameters.mrope_section[]")?;
                                    u32::try_from(n)
                                        .ok()
                                        .filter(|&n| n > 0)
                                        .ok_or_else(|| cfg_err(format!("rope_parameters.mrope_section entry {n} must be in 1..=u32::MAX")))
                                })
                                .collect::<Result<Vec<u32>>>()?,
                            v => return Err(cfg_err(format!("rope_parameters.mrope_section is {}, not an array", v.kind()))),
                        };
                        let interleaved = match inter {
                            None => None,
                            Some(Json::Bool(b)) => Some(*b),
                            Some(v) => {
                                return Err(cfg_err(format!("rope_parameters.mrope_interleaved is {}, not a bool", v.kind())))
                            }
                        };
                        Some(Mrope { section, interleaved })
                    }
                };
                (theta, factor, mrope)
            }
            None => {
                let theta = float_of(
                    tc.get("rope_theta").ok_or_else(|| cfg_err("text_config.rope_theta is missing"))?,
                    "text_config.rope_theta",
                )?;
                let factor = match tc.get("partial_rotary_factor") {
                    Some(v) => float_of(v, "text_config.partial_rotary_factor")?,
                    None => 1.0,
                };
                check_rope_type(tc.get("rope_type"), "text_config.rope_type")?;
                (theta, factor, None)
            }
        };
        if !(theta.is_finite() && theta > 0.0) {
            return Err(cfg_err(format!("rope_theta {theta} must be finite and positive")));
        }
        let rotary = f64::from(head_dim) * factor;
        if !(factor > 0.0 && factor <= 1.0) || rotary.fract() != 0.0 || (rotary as u32) % 2 != 0 {
            return Err(cfg_err(format!(
                "head_dim {head_dim} x partial_rotary_factor {factor} = {rotary} is not an even whole number of rotated dims"
            )));
        }
        let rotary_dim = rotary as u32;
        if let Some(m) = &mrope {
            let sum: u64 = m.section.iter().map(|&n| u64::from(n)).sum();
            if sum != u64::from(rotary_dim / 2) {
                return Err(Qwen35Error::Unsupported {
                    what: format!(
                        "rope_parameters.mrope_section {:?} sums to {sum}, not rotary_dim / 2 = {}",
                        m.section,
                        rotary_dim / 2
                    ),
                    needs: "an MRoPE whose sections cover the rotary frequencies exactly; only then does \
                            text-only MRoPE equal the plain partial RoPE tessl computes"
                        .into(),
                });
            }
        }

        let eps = float_of(
            tc.get("rms_norm_eps").ok_or_else(|| cfg_err("text_config.rms_norm_eps is missing"))?,
            "text_config.rms_norm_eps",
        )?;
        if !(eps.is_finite() && eps > 0.0) {
            return Err(cfg_err(format!("text_config.rms_norm_eps {eps} must be finite and positive")));
        }
        let max_position_embeddings = match tc.get("max_position_embeddings") {
            None => None,
            Some(v) => Some(uint_of(v, "text_config.max_position_embeddings")?),
        };

        // tessl's own reading of the same text: its refusals stay canonical,
        // and its shape is the one the model is built from.
        let tessl_cfg =
            tessl::qwen35_model::Qwen35Config::from_config_json(text).map_err(|e| cfg_err(format!("tessl: {e}")))?;

        let cfg = Self {
            hidden,
            intermediate,
            vocab,
            layers,
            gdn_key_heads,
            gdn_value_heads,
            gdn_key_dim,
            gdn_value_dim,
            conv_kernel,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            rope_theta: theta,
            rms_norm_eps: eps,
            mrope,
            max_position_embeddings,
            tower_prefix,
            reserved_token_ids,
            tessl: tessl_cfg,
        };
        cfg.cross_check()?;
        Ok(cfg)
    }

    /// Every field tessl's parser also reads must be the same value.
    fn cross_check(&self) -> Result<()> {
        let t = &self.tessl;
        let kinds: Vec<LayerKind> = t
            .layers
            .iter()
            .map(|k| match k {
                tessl::qwen35_model::LayerKind::LinearAttention => LayerKind::LinearAttention,
                tessl::qwen35_model::LayerKind::FullAttention => LayerKind::FullAttention,
            })
            .collect();
        let pairs: [(&str, String, String); 14] = [
            ("hidden_size", t.hidden.to_string(), self.hidden.to_string()),
            ("intermediate_size", t.intermediate.to_string(), self.intermediate.to_string()),
            ("vocab_size", t.vocab.to_string(), self.vocab.to_string()),
            ("layer_types", format!("{kinds:?}"), format!("{:?}", self.layers)),
            ("linear_num_key_heads", t.gdn.k_heads().to_string(), self.gdn_key_heads.to_string()),
            ("linear_num_value_heads", t.gdn.v_heads().to_string(), self.gdn_value_heads.to_string()),
            ("linear_value_head_dim", t.gdn.v_dim().to_string(), self.gdn_value_dim.to_string()),
            ("linear_conv_kernel_dim", t.conv_kernel.to_string(), self.conv_kernel.to_string()),
            ("num_attention_heads", t.attn.q_heads().to_string(), self.q_heads.to_string()),
            ("num_key_value_heads", t.attn.kv_heads().to_string(), self.kv_heads.to_string()),
            ("head_dim", t.attn.head_dim().to_string(), self.head_dim.to_string()),
            ("rotary_dim", t.rotary_dim.to_string(), self.rotary_dim.to_string()),
            ("rope_theta (f32)", t.rope_theta.to_string(), (self.rope_theta as f32).to_string()),
            ("rms_norm_eps (f32)", t.rms_norm_eps.to_string(), (self.rms_norm_eps as f32).to_string()),
        ];
        for (name, theirs, ours) in pairs {
            if theirs != ours {
                return Err(cfg_err(format!(
                    "{name}: tessl reads {theirs}, ojas-qwen35 reads {ours}; refusing a config the two parsers disagree on"
                )));
            }
        }
        Ok(())
    }

    /// Parameters (elements) of the text tower: what f32 masters, a gradient
    /// bank and each AdamW moment hold.
    pub fn parameter_count(&self) -> u64 {
        crate::names::tower_tensors(self).iter().map(|t| t.numel() as u64).sum()
    }

    /// An exact, readable one-line statement of the shape, stored in every
    /// state file so a checkpoint is never loaded into another shape.
    pub fn canonical_summary(&self) -> String {
        let layers: String = self
            .layers
            .iter()
            .map(|k| match k {
                LayerKind::LinearAttention => 'L',
                LayerKind::FullAttention => 'F',
            })
            .collect();
        format!(
            "qwen3_5_text;hidden={};intermediate={};vocab={};layers={layers};gdn={}k/{}v/{}kd/{}vd/conv{};\
             attn={}q/{}kv/{}d;rotary={};theta={};eps={};mrope={:?};prefix={}",
            self.hidden,
            self.intermediate,
            self.vocab,
            self.gdn_key_heads,
            self.gdn_value_heads,
            self.gdn_key_dim,
            self.gdn_value_dim,
            self.conv_kernel,
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.rope_theta,
            self.rms_norm_eps,
            self.mrope,
            self.tower_prefix
        )
    }
}

fn check_rope_type(v: Option<&Json>, path: &str) -> Result<()> {
    match v {
        None => Ok(()),
        Some(Json::Str(s)) if s == "default" => Ok(()),
        Some(v) => Err(cfg_err(format!("{path} is {}; only \"default\" is implemented", v.kind()))),
    }
}
