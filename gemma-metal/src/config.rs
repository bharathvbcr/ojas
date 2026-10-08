//! Gemma 4 HF config deserializer (text + MTP assistant).
//!
//! Accepts:
//! - `model_type: "gemma4"` wrapper with nested `text_config`
//! - bare `model_type: "gemma4_text"`
//! - `model_type: "gemma4_assistant"` (MTP drafter; parsed, not executed in Phase 1)

use serde::Deserialize;
use std::fs;
use std::path::Path;

use crate::error::{Error, Result};

/// Sliding vs global attention layer kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

impl LayerType {
    pub fn is_sliding(self) -> bool {
        matches!(self, LayerType::SlidingAttention)
    }

    pub fn is_global(self) -> bool {
        matches!(self, LayerType::FullAttention)
    }
}

/// RoPE parameters for one attention family.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct RopeFamily {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_rope_type() -> String {
    "default".into()
}

/// Dual RoPE: local (sliding) vs global (full / p-RoPE).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct RopeParameters {
    pub sliding_attention: RopeFamily,
    pub full_attention: RopeFamily,
}

/// Text backbone (`Gemma4TextConfig`).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Gemma4TextConfig {
    #[serde(default = "default_vocab")]
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_global_head_dim")]
    pub global_head_dim: usize,
    /// KV heads for global layers; `None` → use `num_key_value_heads`.
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_activation: String,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub layer_types: Vec<LayerType>,
    #[serde(default = "default_softcap")]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// PLE dim; `0` means no PLE (31B).
    #[serde(default)]
    pub hidden_size_per_layer_input: usize,
    #[serde(default = "default_vocab")]
    pub vocab_size_per_layer_input: usize,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<EosIds>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub attention_bias: bool,
}

fn default_vocab() -> usize {
    262_144
}
fn default_head_dim() -> usize {
    256
}
fn default_global_head_dim() -> usize {
    512
}
fn default_hidden_act() -> String {
    "gelu_pytorch_tanh".into()
}
fn default_rms_eps() -> f64 {
    1e-6
}
fn default_max_pos() -> usize {
    131_072
}
fn default_softcap() -> Option<f32> {
    Some(30.0)
}

/// EOS may be a single id or a list in HF JSON.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum EosIds {
    One(u32),
    Many(Vec<u32>),
}

impl EosIds {
    pub fn as_slice(&self) -> &[u32] {
        match self {
            EosIds::One(id) => std::slice::from_ref(id),
            EosIds::Many(v) => v,
        }
    }
}

impl Gemma4TextConfig {
    /// First layer index that is a KV consumer (no K/V weights).
    /// E4B: `42 - 18 = 24`.
    pub fn first_kv_shared(&self) -> usize {
        self.num_hidden_layers.saturating_sub(self.num_kv_shared_layers)
    }

    pub fn has_ple(&self) -> bool {
        self.hidden_size_per_layer_input > 0
    }

    pub fn sliding_window_or(&self, fallback: usize) -> usize {
        self.sliding_window.unwrap_or(fallback)
    }

    pub fn global_kv_heads(&self) -> usize {
        self.num_global_key_value_heads
            .unwrap_or(self.num_key_value_heads)
    }

    pub fn layer_type(&self, layer: usize) -> Result<LayerType> {
        self.layer_types
            .get(layer)
            .copied()
            .ok_or_else(|| Error::Config(format!("layer_types missing index {layer}")))
    }

    /// Validate shapes known for E4B / 31B and fill defaults.
    pub fn validate(&mut self) -> Result<()> {
        if self.layer_types.is_empty() {
            return Err(Error::Config(
                "layer_types empty — required for dual FA / KV layout".into(),
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(Error::Config(format!(
                "layer_types len {} != num_hidden_layers {}",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        if self.num_kv_shared_layers > self.num_hidden_layers {
            return Err(Error::Config(format!(
                "num_kv_shared_layers {} > num_hidden_layers {}",
                self.num_kv_shared_layers, self.num_hidden_layers
            )));
        }
        if self.head_dim == 0 || self.global_head_dim == 0 {
            return Err(Error::Config("head_dim / global_head_dim must be > 0".into()));
        }
        Ok(())
    }

    /// Canonical E4B text shapes (plan / HF `google/gemma-4-E4B-it`).
    pub fn e4b_preset() -> Self {
        let mut layer_types = Vec::with_capacity(42);
        for i in 0..42 {
            // Pattern: 5 sliding + 1 full, ending on full.
            if (i + 1) % 6 == 0 {
                layer_types.push(LayerType::FullAttention);
            } else {
                layer_types.push(LayerType::SlidingAttention);
            }
        }
        Self {
            vocab_size: 262_144,
            hidden_size: 2560,
            intermediate_size: 10_240,
            num_hidden_layers: 42,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 256,
            global_head_dim: 512,
            num_global_key_value_heads: None,
            attention_k_eq_v: false,
            num_kv_shared_layers: 18,
            hidden_activation: "gelu_pytorch_tanh".into(),
            rms_norm_eps: 1e-6,
            max_position_embeddings: 131_072,
            sliding_window: Some(512),
            layer_types,
            final_logit_softcapping: Some(30.0),
            tie_word_embeddings: true,
            hidden_size_per_layer_input: 256,
            vocab_size_per_layer_input: 262_144,
            rope_parameters: Some(RopeParameters {
                sliding_attention: RopeFamily {
                    rope_theta: 10_000.0,
                    rope_type: "default".into(),
                    partial_rotary_factor: None,
                },
                full_attention: RopeFamily {
                    rope_theta: 1_000_000.0,
                    rope_type: "proportional".into(),
                    partial_rotary_factor: Some(0.25),
                },
            }),
            bos_token_id: Some(2),
            eos_token_id: Some(EosIds::One(1)),
            pad_token_id: Some(0),
            attention_bias: false,
        }
    }

    /// Canonical 31B text shapes (plan / HF `google/gemma-4-31B-it`).
    pub fn b31_preset() -> Self {
        let mut layer_types = Vec::with_capacity(60);
        for i in 0..60 {
            if (i + 1) % 6 == 0 {
                layer_types.push(LayerType::FullAttention);
            } else {
                layer_types.push(LayerType::SlidingAttention);
            }
        }
        Self {
            vocab_size: 262_144,
            hidden_size: 5376,
            intermediate_size: 21_504,
            num_hidden_layers: 60,
            num_attention_heads: 32,
            num_key_value_heads: 16,
            head_dim: 256,
            global_head_dim: 512,
            num_global_key_value_heads: Some(4),
            attention_k_eq_v: true,
            num_kv_shared_layers: 0,
            hidden_activation: "gelu_pytorch_tanh".into(),
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262_144,
            sliding_window: Some(1024),
            layer_types,
            final_logit_softcapping: Some(30.0),
            tie_word_embeddings: true,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: 262_144,
            rope_parameters: Some(RopeParameters {
                sliding_attention: RopeFamily {
                    rope_theta: 10_000.0,
                    rope_type: "default".into(),
                    partial_rotary_factor: None,
                },
                full_attention: RopeFamily {
                    rope_theta: 1_000_000.0,
                    rope_type: "proportional".into(),
                    partial_rotary_factor: Some(0.25),
                },
            }),
            bos_token_id: Some(2),
            eos_token_id: Some(EosIds::One(1)),
            pad_token_id: Some(0),
            attention_bias: false,
        }
    }
}

/// MTP assistant / drafter config (`gemma4_assistant`). Parsed for Phase 5; unused in decode loop yet.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Gemma4AssistantConfig {
    pub backbone_hidden_size: usize,
    #[serde(default = "default_centroids")]
    pub num_centroids: usize,
    #[serde(default = "default_centroid_topk")]
    pub centroid_intermediate_top_k: usize,
    #[serde(default)]
    pub use_ordered_embeddings: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub text_config: Gemma4TextConfig,
}

fn default_centroids() -> usize {
    2048
}
fn default_centroid_topk() -> usize {
    32
}

/// Top-level HF config: multimodal wrapper, bare text, or assistant.
#[derive(Clone, Debug, PartialEq)]
pub enum Gemma4Config {
    /// `model_type: gemma4` with nested text backbone (vision/audio ignored for v1).
    Multimodal { text: Gemma4TextConfig },
    /// Bare `gemma4_text`.
    Text(Gemma4TextConfig),
    /// MTP drafter.
    Assistant(Gemma4AssistantConfig),
}

impl Gemma4Config {
    pub fn text(&self) -> &Gemma4TextConfig {
        match self {
            Gemma4Config::Multimodal { text } => text,
            Gemma4Config::Text(t) => t,
            Gemma4Config::Assistant(a) => &a.text_config,
        }
    }

    pub fn text_mut(&mut self) -> &mut Gemma4TextConfig {
        match self {
            Gemma4Config::Multimodal { text } => text,
            Gemma4Config::Text(t) => t,
            Gemma4Config::Assistant(a) => &mut a.text_config,
        }
    }

    pub fn assistant(&self) -> Option<&Gemma4AssistantConfig> {
        match self {
            Gemma4Config::Assistant(a) => Some(a),
            _ => None,
        }
    }

    pub fn from_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)?;
        Self::from_value(v)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let s = fs::read_to_string(path.as_ref())
            .map_err(|e| Error::Io(format!("read config {}: {e}", path.as_ref().display())))?;
        Self::from_json_str(&s)
    }

    pub fn from_value(v: serde_json::Value) -> Result<Self> {
        let model_type = v
            .get("model_type")
            .and_then(|x| x.as_str())
            .unwrap_or("");

        let mut cfg = match model_type {
            "gemma4" => {
                let text: Gemma4TextConfig = serde_json::from_value(
                    v.get("text_config")
                        .cloned()
                        .ok_or_else(|| Error::Config("gemma4 missing text_config".into()))?,
                )?;
                Gemma4Config::Multimodal { text }
            }
            "gemma4_text" => Gemma4Config::Text(serde_json::from_value(v)?),
            "gemma4_assistant" => {
                #[derive(Deserialize)]
                struct RawAssistant {
                    backbone_hidden_size: usize,
                    #[serde(default = "default_centroids")]
                    num_centroids: usize,
                    #[serde(default = "default_centroid_topk")]
                    centroid_intermediate_top_k: usize,
                    #[serde(default)]
                    use_ordered_embeddings: bool,
                    #[serde(default)]
                    tie_word_embeddings: bool,
                    text_config: Gemma4TextConfig,
                }
                let raw: RawAssistant = serde_json::from_value(v)?;
                Gemma4Config::Assistant(Gemma4AssistantConfig {
                    backbone_hidden_size: raw.backbone_hidden_size,
                    num_centroids: raw.num_centroids,
                    centroid_intermediate_top_k: raw.centroid_intermediate_top_k,
                    use_ordered_embeddings: raw.use_ordered_embeddings,
                    tie_word_embeddings: raw.tie_word_embeddings,
                    text_config: raw.text_config,
                })
            }
            // Some exports omit model_type but nest text_config.
            _ if v.get("text_config").is_some() => {
                let text: Gemma4TextConfig =
                    serde_json::from_value(v.get("text_config").cloned().unwrap())?;
                Gemma4Config::Multimodal { text }
            }
            other => {
                return Err(Error::Config(format!(
                    "unsupported model_type '{other}' (expected gemma4 / gemma4_text / gemma4_assistant)"
                )));
            }
        };
        cfg.text_mut().validate()?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const E4B_JSON: &str = r#"{
      "model_type": "gemma4",
      "text_config": {
        "attention_k_eq_v": false,
        "final_logit_softcapping": 30.0,
        "global_head_dim": 512,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 2560,
        "hidden_size_per_layer_input": 256,
        "intermediate_size": 10240,
        "layer_types": [
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention"
        ],
        "max_position_embeddings": 131072,
        "model_type": "gemma4_text",
        "num_attention_heads": 8,
        "num_hidden_layers": 42,
        "num_key_value_heads": 2,
        "num_kv_shared_layers": 18,
        "sliding_window": 512,
        "vocab_size": 262144,
        "vocab_size_per_layer_input": 262144,
        "rope_parameters": {
          "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
            "rope_type": "proportional"
          },
          "sliding_attention": {
            "rope_theta": 10000.0,
            "rope_type": "default"
          }
        }
      }
    }"#;

    const B31_JSON: &str = r#"{
      "model_type": "gemma4",
      "text_config": {
        "attention_k_eq_v": true,
        "final_logit_softcapping": 30.0,
        "global_head_dim": 512,
        "head_dim": 256,
        "hidden_size": 5376,
        "hidden_size_per_layer_input": 0,
        "intermediate_size": 21504,
        "layer_types": [
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention"
        ],
        "max_position_embeddings": 262144,
        "num_attention_heads": 32,
        "num_global_key_value_heads": 4,
        "num_hidden_layers": 60,
        "num_key_value_heads": 16,
        "num_kv_shared_layers": 0,
        "sliding_window": 1024,
        "vocab_size": 262144
      }
    }"#;

    #[test]
    fn parse_e4b_shapes() {
        let cfg = Gemma4Config::from_json_str(E4B_JSON).unwrap();
        let t = cfg.text();
        assert_eq!(t.hidden_size, 2560);
        assert_eq!(t.intermediate_size, 10_240);
        assert_eq!(t.num_hidden_layers, 42);
        assert_eq!(t.num_kv_shared_layers, 18);
        assert_eq!(t.first_kv_shared(), 24);
        assert!(t.has_ple());
        assert_eq!(t.hidden_size_per_layer_input, 256);
        assert!(!t.attention_k_eq_v);
        assert_eq!(t.sliding_window, Some(512));
        assert_eq!(t.layer_types.len(), 42);
        assert_eq!(t.layer_types[5], LayerType::FullAttention);
        assert_eq!(t.layer_types[0], LayerType::SlidingAttention);
        let rope = t.rope_parameters.as_ref().unwrap();
        assert_eq!(rope.full_attention.partial_rotary_factor, Some(0.25));
    }

    #[test]
    fn parse_31b_shapes() {
        let cfg = Gemma4Config::from_json_str(B31_JSON).unwrap();
        let t = cfg.text();
        assert_eq!(t.hidden_size, 5376);
        assert_eq!(t.num_hidden_layers, 60);
        assert_eq!(t.first_kv_shared(), 60);
        assert!(!t.has_ple());
        assert!(t.attention_k_eq_v);
        assert_eq!(t.global_kv_heads(), 4);
        assert_eq!(t.sliding_window, Some(1024));
    }

    #[test]
    fn presets_match_plan() {
        let e4b = Gemma4TextConfig::e4b_preset();
        assert_eq!(e4b.first_kv_shared(), 24);
        assert_eq!(e4b.num_hidden_layers - e4b.num_kv_shared_layers, 24);
        let b31 = Gemma4TextConfig::b31_preset();
        assert_eq!(b31.num_kv_shared_layers, 0);
        assert!(b31.attention_k_eq_v);
    }

    #[test]
    fn parse_assistant_e4b() {
        let json = r#"{
          "model_type": "gemma4_assistant",
          "backbone_hidden_size": 2560,
          "num_centroids": 2048,
          "centroid_intermediate_top_k": 32,
          "use_ordered_embeddings": true,
          "text_config": {
            "hidden_size": 256,
            "intermediate_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_kv_shared_layers": 4,
            "hidden_size_per_layer_input": 0,
            "layer_types": [
              "sliding_attention","sliding_attention","sliding_attention","full_attention"
            ],
            "sliding_window": 512,
            "vocab_size": 262144
          }
        }"#;
        let cfg = Gemma4Config::from_json_str(json).unwrap();
        let a = cfg.assistant().unwrap();
        assert_eq!(a.backbone_hidden_size, 2560);
        assert_eq!(a.num_centroids, 2048);
        assert_eq!(a.text_config.hidden_size, 256);
        assert_eq!(a.text_config.num_hidden_layers, 4);
    }
}
