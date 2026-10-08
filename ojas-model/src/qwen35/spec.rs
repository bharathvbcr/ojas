//! The Qwen3.5 text tower's shape, as the hybrid graph needs it.
//!
//! This is a plain validated struct, not a `config.json` reader: ojas-qwen35's
//! `Qwen35TextConfig` stays the one parser of the Hugging Face config and
//! converts into this with `Qwen35TextConfig::tape_spec`.

use ojas_core::OjasError;

/// Which mixer a layer runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35Mixer {
    /// The gated delta net (`"linear_attention"`).
    GatedDeltaNet,
    /// Gated softmax attention (`"full_attention"`).
    Attention,
}

/// The text tower's dims. Every count is positive; see [`Qwen35Spec::validate`].
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35Spec {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: Vec<Qwen35Mixer>,
    /// Attention query heads, a multiple of `kv_heads`.
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// The leading `rotary_dim` of each attention head turn: `head_dim *
    /// partial_rotary_factor`, even.
    pub rotary_dim: usize,
    pub rope_theta: f64,
    /// Gated delta net heads. The delta rule has no head grouping, so key
    /// and value heads are one count.
    pub gdn_heads: usize,
    pub gdn_key_dim: usize,
    pub gdn_value_dim: usize,
    /// Taps of the depthwise causal conv in front of the delta rule.
    pub conv_width: usize,
    pub eps: f32,
}

fn refuse(detail: String) -> OjasError {
    OjasError::Shape {
        op: "Qwen35Spec::validate",
        detail,
    }
}

impl Qwen35Spec {
    /// Refuse a zero count, query heads that are not a multiple of the KV
    /// heads, a rotary width that is odd or wider than a head, a theta that
    /// is not finite and above 1, and an eps that is not finite and positive.
    pub fn validate(&self) -> Result<(), OjasError> {
        let counts = [
            ("vocab", self.vocab),
            ("hidden", self.hidden),
            ("intermediate", self.intermediate),
            ("layers", self.layers.len()),
            ("q_heads", self.q_heads),
            ("kv_heads", self.kv_heads),
            ("head_dim", self.head_dim),
            ("rotary_dim", self.rotary_dim),
            ("gdn_heads", self.gdn_heads),
            ("gdn_key_dim", self.gdn_key_dim),
            ("gdn_value_dim", self.gdn_value_dim),
            ("conv_width", self.conv_width),
        ];
        if let Some((name, _)) = counts.iter().find(|(_, n)| *n == 0) {
            return Err(refuse(format!("{name} is 0")));
        }
        if !self.q_heads.is_multiple_of(self.kv_heads) {
            return Err(refuse(format!(
                "q_heads {} is not a multiple of kv_heads {}",
                self.q_heads, self.kv_heads
            )));
        }
        if !self.rotary_dim.is_multiple_of(2) || self.rotary_dim > self.head_dim {
            return Err(refuse(format!(
                "rotary_dim {} must be even and at most head_dim {}",
                self.rotary_dim, self.head_dim
            )));
        }
        if !(self.rope_theta.is_finite() && self.rope_theta > 1.0) {
            return Err(refuse(format!(
                "rope_theta {} must be finite and above 1",
                self.rope_theta
            )));
        }
        if !(self.eps.is_finite() && self.eps > 0.0) {
            return Err(refuse(format!(
                "eps {} must be finite and positive",
                self.eps
            )));
        }
        Ok(())
    }

    /// Channels of `in_proj_qkv` and its conv: `[q | k | v]`.
    pub fn gdn_qkv_channels(&self) -> [usize; 3] {
        let k = self.gdn_heads * self.gdn_key_dim;
        [k, k, self.gdn_heads * self.gdn_value_dim]
    }
}
