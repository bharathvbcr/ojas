//! Model weights on GPU.
//!
//! # Weight layout (metal-native)
//!
//! | Kind | Layout | Notes |
//! |------|--------|-------|
//! | Embedding tables (`tok_emb`, `bigram.embed`, `ve.embed`) | `[vocab, dim]` | row-gather |
//! | Linear / bank matrices | `[in, out]` | `x @ W` (Burn convention) |
//! | Python / golden `.npy` banks | `[out, in]` | transpose last-2 on load |
//!
//! Tied logits use a transposed view `tok_emb_t: [C, V]` for GEMM.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

use crate::npy::{read_npy, transpose_last2};
use crate::runtime::GpuRuntime;
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MixerKind {
    Attention,
    MinGRU,
    Mamba2,
}

impl Default for MixerKind {
    fn default() -> Self {
        MixerKind::Attention
    }
}

impl MixerKind {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "attention" | "attn" => Ok(MixerKind::Attention),
            "mingru" => Ok(MixerKind::MinGRU),
            "mamba2" | "mamba" => Ok(MixerKind::Mamba2),
            other => Err(format!("unknown mixer kind '{other}'")),
        }
    }
}

/// RADA-style pad/repeat of a short mixer list to `num_layers`.
pub fn expand_layer_mixers(pattern: &[MixerKind], num_layers: usize) -> Vec<MixerKind> {
    if pattern.is_empty() {
        return Vec::new();
    }
    (0..num_layers)
        .map(|i| pattern[i % pattern.len()])
        .collect()
}

/// Default hybrid pattern for even L: three Mamba-2 + one Attention per period.
pub fn default_hybrid_pattern(num_layers: usize) -> Vec<MixerKind> {
    let unit = [
        MixerKind::Mamba2,
        MixerKind::Mamba2,
        MixerKind::Mamba2,
        MixerKind::Attention,
    ];
    (0..num_layers).map(|i| unit[i % unit.len()]).collect()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    #[serde(default)]
    pub mixer: MixerKind,

    /// Per-layer mixer kinds (hybrid). Empty → homogeneous `mixer`.
    #[serde(default)]
    pub layer_mixers: Vec<MixerKind>,

    /// MinGRU value-residual probe (W_v / W_v0_up blend into h_pre).
    #[serde(default)]
    pub value_residual: bool,

    pub batch: usize,
    pub seq_len: usize,
    pub num_layers: usize,
    pub model_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub mlp_dim: usize,
    pub vocab_size: usize,
    pub bigram_vocab: usize,
    pub bigram_dim: usize,
    pub ve_dim: usize,
    pub rope_dims: usize,
    pub rope_base: f32,
    pub logit_softcap: f32,
    pub ve_layers: Vec<usize>,
    pub xsa_last_n: usize,
    pub ln_scale: bool,

    /// Mamba-2 state dimension (nanolab default 64).
    #[serde(default = "default_d_state")]
    pub d_state: usize,
    /// Mamba-2 depthwise conv kernel width.
    #[serde(default = "default_d_conv")]
    pub d_conv: usize,
    /// SSD chunk size cap (nanolab: min(mixer_chunk, block_size)).
    #[serde(default = "default_mixer_chunk")]
    pub mixer_chunk: usize,
}

fn default_d_state() -> usize {
    64
}

fn default_d_conv() -> usize {
    4
}

fn default_mixer_chunk() -> usize {
    32
}

impl ModelConfig {
    pub fn sota_toy() -> Self {
        Self {
            mixer: MixerKind::Attention,
            layer_mixers: vec![],
            value_residual: false,
            batch: 16,
            seq_len: 256,
            num_layers: 4,
            model_dim: 128,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            mlp_dim: 384,
            vocab_size: 1024,
            bigram_vocab: 512,
            bigram_dim: 48,
            ve_dim: 24,
            rope_dims: 8,
            rope_base: 10000.0,
            logit_softcap: 30.0,
            ve_layers: vec![2, 3],
            xsa_last_n: 2,
            ln_scale: true,
            d_state: 64,
            d_conv: 4,
            mixer_chunk: 32,
        }
    }

    /// ~16M Soft scale-up from `sota_toy`, kept Metal-FA friendly (`head_dim=32`,
    /// GQA group=2, mlp=3×). Lands at ~16.41M trainable params (within ±20% of 16M).
    ///
    /// Default step shape: **B=16 × T=256 = 4096 tok/step** (same tokens/step as
    /// sota toy). M5 Pro Soft A/B (2026-07-13): B16/T256 + FA_TILED ≈ **954 ms**
    /// vs B8/T512 ≈ **1259 ms** (~24% faster, same RSS). Override via train
    /// `--batch` / `--seq-len`.
    pub fn medium_16m() -> Self {
        Self {
            mixer: MixerKind::Attention,
            layer_mixers: vec![],
            value_residual: false,
            batch: 16,
            seq_len: 256,
            num_layers: 12,
            model_dim: 384,
            num_heads: 12,
            num_kv_heads: 6,
            head_dim: 32,
            mlp_dim: 1152, // 3.0 * 384
            vocab_size: 1024,
            bigram_vocab: 512,
            bigram_dim: 48,
            ve_dim: 24,
            rope_dims: 8,
            rope_base: 10000.0,
            logit_softcap: 30.0,
            ve_layers: vec![10, 11],
            xsa_last_n: 4,
            ln_scale: true,
            d_state: 64,
            d_conv: 4,
            mixer_chunk: 32,
        }
    }

    /// Exact 128,367,988-parameter Arch02 target for the M5 Pro study.
    ///
    /// This preserves the 16M model's head dimension, GQA ratio, 3x MLP,
    /// lean auxiliary widths, and B16/T256 step shape while scaling depth and
    /// width.  Keeping D=32 is important for the validated Metal FA kernels.
    pub fn arch02_128m() -> Self {
        Self {
            mixer: MixerKind::Attention,
            layer_mixers: vec![],
            value_residual: false,
            batch: 16,
            seq_len: 256,
            num_layers: 24,
            model_dim: 768,
            num_heads: 24,
            num_kv_heads: 12,
            head_dim: 32,
            mlp_dim: 2304,
            vocab_size: 1024,
            bigram_vocab: 512,
            bigram_dim: 48,
            ve_dim: 24,
            rope_dims: 8,
            rope_base: 10000.0,
            logit_softcap: 30.0,
            ve_layers: vec![22, 23],
            xsa_last_n: 8,
            ln_scale: true,
            d_state: 64,
            d_conv: 4,
            mixer_chunk: 32,
        }
    }

    /// Resolve `--preset` names used by `bin/train`.
    pub fn from_preset(name: &str) -> Result<Self, String> {
        match name {
            "sota" | "sota_toy" | "toy" => Ok(Self::sota_toy()),
            "16m" | "medium_16m" | "medium" => Ok(Self::medium_16m()),
            "128m" | "arch02-128m" | "arch02_128m" => Ok(Self::arch02_128m()),
            other => Err(format!(
                "unknown --preset '{other}' (expected sota|16m|arch02-128m)"
            )),
        }
    }

    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    pub fn mamba_d_inner(&self) -> usize {
        2 * self.model_dim
    }

    pub fn mamba_n_head(&self) -> usize {
        (self.mamba_d_inner() / 64).max(1)
    }

    pub fn mamba_head_dim(&self) -> usize {
        self.mamba_d_inner() / self.mamba_n_head()
    }

    pub fn mamba_conv_dim(&self) -> usize {
        self.mamba_d_inner() + 2 * self.d_state
    }

    pub fn mamba_in_proj_out(&self) -> usize {
        2 * self.mamba_d_inner() + 2 * self.d_state + self.mamba_n_head()
    }

    pub fn mingru_hidden(&self) -> usize {
        2 * self.model_dim
    }

    /// Exact trainable parameter count for the metal-native weight layout
    /// (tied logits; no separate LM head).
    pub fn count_params(&self) -> usize {
        let n = self.num_layers;
        let c = self.model_dim;
        let kv = self.kv_dim();
        let hid = self.mingru_hidden();
        let mlp = self.mlp_dim;
        let n_skip = n / 2;
        let n_attn = self.mixer_count(MixerKind::Attention);
        let n_mingru = self.mixer_count(MixerKind::MinGRU);
        let n_mamba = self.mixer_count(MixerKind::Mamba2);
        let mut total = 0usize;
        total += self.vocab_size * c; // tok_emb (tied)
        total += self.bigram_vocab * self.bigram_dim + self.bigram_dim * c + 1; // bigram
        total += c; // smear_gate
        total += self.vocab_size * self.ve_dim + self.ve_dim * kv + 1; // ve shared
        total += self.ve_layers.len(); // ve_layer_scales
        total += n_skip * c; // skip_weights

        if n_attn > 0 {
            total += 2 * n_attn * c * c; // qo_bank
            total += 2 * n_attn * c * kv; // kv_bank
        }
        if n_mingru > 0 {
            total += n_mingru * (c * hid * 2 + hid * c);
            if self.value_residual {
                total += n_mingru * (c * kv + kv * hid);
            }
        }
        if n_mamba > 0 {
            let d_inner = self.mamba_d_inner();
            let n_head = self.mamba_n_head();
            let conv_dim = self.mamba_conv_dim();
            let in_out = self.mamba_in_proj_out();
            total += n_mamba
                * (c * in_out
                    + conv_dim * self.d_conv
                    + conv_dim
                    + d_inner * c
                    + n_head * 3
                    + d_inner);
        }

        total += 2 * n * c * mlp; // mlp_up + mlp_down
        total += n * (self.num_heads + 2 + c + c + 2 * c); // blocks
        total
    }

    /// Resolved per-layer mixer list (hybrid or homogeneous).
    pub fn resolved_layer_mixers(&self) -> Vec<MixerKind> {
        if self.layer_mixers.is_empty() {
            vec![self.mixer; self.num_layers]
        } else {
            self.layer_mixers.clone()
        }
    }

    pub fn layer_mixer(&self, layer: usize) -> MixerKind {
        self.resolved_layer_mixers()[layer]
    }

    pub fn is_hybrid(&self) -> bool {
        !self.layer_mixers.is_empty()
    }

    pub fn mixer_count(&self, kind: MixerKind) -> usize {
        self.resolved_layer_mixers()
            .iter()
            .filter(|&&m| m == kind)
            .count()
    }

    pub fn mixer_local_idx(&self, layer: usize) -> usize {
        let kind = self.layer_mixer(layer);
        self.resolved_layer_mixers()[..layer]
            .iter()
            .filter(|&&m| m == kind)
            .count()
    }

    pub fn is_attention_layer(&self, layer: usize) -> bool {
        self.layer_mixer(layer) == MixerKind::Attention
    }

    pub fn is_mingru_layer(&self, layer: usize) -> bool {
        self.layer_mixer(layer) == MixerKind::MinGRU
    }

    pub fn attn_local_idx(&self, layer: usize) -> Option<usize> {
        if !self.is_attention_layer(layer) {
            return None;
        }
        Some(self.mixer_local_idx(layer))
    }

    /// First layer that captures v0 (first Attention, or first MinGRU when VR).
    pub fn first_v0_layer(&self) -> Option<usize> {
        for layer in 0..self.num_layers {
            match self.layer_mixer(layer) {
                MixerKind::Attention => return Some(layer),
                MixerKind::MinGRU if self.value_residual => return Some(layer),
                _ => {}
            }
        }
        None
    }

    pub fn captures_v0(&self, layer: usize) -> bool {
        self.first_v0_layer() == Some(layer)
    }

    /// VE layer scales keyed on attention-local indices.
    pub fn ve_attn_index(&self, layer: usize) -> Option<usize> {
        let ai = self.attn_local_idx(layer)?;
        self.ve_layers.iter().position(|&l| l == ai)
    }

    /// XSA on the last `xsa_last_n` attention layers (not global tail).
    pub fn use_xsa_attn(&self, layer: usize) -> bool {
        let Some(ai) = self.attn_local_idx(layer) else {
            return false;
        };
        let n_attn = self.mixer_count(MixerKind::Attention);
        ai >= n_attn.saturating_sub(self.xsa_last_n)
    }

    pub fn optimizes_vr_lambda(&self, layer: usize) -> bool {
        if self.value_residual && self.is_mingru_layer(layer) {
            return self.mixer_local_idx(layer) > 0;
        }
        if self.is_attention_layer(layer) {
            return self.attn_local_idx(layer).unwrap_or(0) > 0;
        }
        false
    }

    pub fn optimizes_q_gain(&self, layer: usize) -> bool {
        self.is_attention_layer(layer)
    }

    /// Soft FA / kernel shape gate: simdgroup FA supports D≤64; TensorOps FA
    /// probe requires D==32. Prefer head_dim=32 for the 16M Soft path.
    pub fn validate_metal_shape(&self) -> Result<(), String> {
        if self.model_dim != self.num_heads * self.head_dim {
            return Err(format!(
                "model_dim {} != num_heads {} * head_dim {}",
                self.model_dim, self.num_heads, self.head_dim
            ));
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err(format!(
                "GQA: num_heads {} not divisible by num_kv_heads {}",
                self.num_heads, self.num_kv_heads
            ));
        }
        if self.head_dim == 0 || self.head_dim > 64 {
            return Err(format!(
                "head_dim {} outside Metal FA support (1..=64)",
                self.head_dim
            ));
        }
        if self.num_layers < 2 || self.num_layers % 2 != 0 {
            return Err(format!(
                "num_layers {} must be even (≥2) for U-net skip pairing",
                self.num_layers
            ));
        }
        Ok(())
    }

    pub fn f32_eps(&self) -> f32 {
        f32::EPSILON // matches torch.finfo(float32).eps
    }

    pub fn xsa_eps(&self) -> f32 {
        1e-12
    }

    pub fn ln_scale_factor(&self, layer: usize) -> f32 {
        if self.ln_scale {
            1.0 / ((layer + 1) as f32).sqrt()
        } else {
            1.0
        }
    }

    pub fn use_xsa(&self, layer: usize) -> bool {
        if self.is_hybrid() || self.mixer_count(MixerKind::Attention) < self.num_layers {
            return self.use_xsa_attn(layer);
        }
        layer >= self.num_layers.saturating_sub(self.xsa_last_n)
    }

    pub fn ve_scale_index(&self, layer: usize) -> Option<usize> {
        if self.is_hybrid() || self.mixer_count(MixerKind::Attention) < self.num_layers {
            return self.ve_attn_index(layer);
        }
        self.ve_layers.iter().position(|&l| l == layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sota_toy_param_count() {
        let n = ModelConfig::sota_toy().count_params();
        assert_eq!(n, 780_188, "sota toy drifted: {n}");
    }

    #[test]
    fn medium_16m_param_count_near_16m() {
        let cfg = ModelConfig::medium_16m();
        cfg.validate_metal_shape().unwrap();
        let n = cfg.count_params();
        assert!(
            (12_800_000..=19_200_000).contains(&n),
            "16M preset outside ±20%: {n}"
        );
        assert_eq!(cfg.head_dim, 32);
        assert_eq!(cfg.num_heads % cfg.num_kv_heads, 0);
        assert_eq!(n, 16_411_948);
        // Soft step-time winner (DECISIONS M12): same 4096 tok/step as sota toy.
        assert_eq!(cfg.batch, 16);
        assert_eq!(cfg.seq_len, 256);
    }

    #[test]
    fn preset_aliases() {
        assert_eq!(
            ModelConfig::from_preset("16m").unwrap().count_params(),
            ModelConfig::medium_16m().count_params()
        );
        assert_eq!(
            ModelConfig::from_preset("sota").unwrap().count_params(),
            ModelConfig::sota_toy().count_params()
        );
        assert_eq!(
            ModelConfig::from_preset("arch02-128m")
                .unwrap()
                .count_params(),
            128_367_988
        );
    }

    #[test]
    fn arch02_128m_exact_shape_and_count() {
        let cfg = ModelConfig::arch02_128m();
        cfg.validate_metal_shape().unwrap();
        assert_eq!(cfg.count_params(), 128_367_988);
        assert_eq!(cfg.kv_dim(), 384);
        assert_eq!(cfg.model_dim, 768);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.batch * cfg.seq_len, 4096);
    }
}

pub struct BlockWeights {
    pub q_gain: Tensor,    // [H]
    pub vr_lambda: Tensor, // [2]
    pub attn_scale: Tensor,
    pub mlp_scale: Tensor,
    pub resid_mix: Tensor, // [2, C]
}

pub struct Weights {
    pub cfg: ModelConfig,
    pub tok_emb: Tensor,   // [V, C]
    pub tok_emb_t: Tensor, // [C, V] for logits GEMM
    pub bigram_emb: Tensor,
    pub bigram_proj: Tensor, // [Db, C]
    pub bigram_scale: Tensor,
    pub smear_gate: Tensor,
    pub ve_emb: Tensor,
    pub ve_proj: Tensor, // [De, kv]
    pub ve_scale: Tensor,
    pub ve_layer_scales: Vec<Tensor>,
    pub skip_weights: Tensor, // [2, C]
    pub mingru_to_z: Option<Tensor>,
    pub mingru_to_h: Option<Tensor>,
    pub mingru_out: Option<Tensor>,
    pub mingru_v_proj: Option<Tensor>,
    pub mingru_v0_up: Option<Tensor>,
    pub mamba_in_proj: Option<Tensor>,
    /// Mamba depthwise Conv1d bank: `[n_mamba, conv_dim, d_conv]` row-major.
    /// Per-layer slice is `[conv_dim, d_conv]` = kernel `w[c, k]` (PyTorch/nanolab `(C, K)`).
    pub mamba_conv1d_weight: Option<Tensor>,
    pub mamba_conv1d_bias: Option<Tensor>,
    pub mamba_out_proj: Option<Tensor>,
    pub mamba_a_log: Option<Tensor>,
    pub mamba_d: Option<Tensor>,
    pub mamba_dt_bias: Option<Tensor>,
    pub mamba_norm: Option<Tensor>,
    pub qo_bank: Tensor, // [8, C, C]  [in,out] after load
    pub kv_bank: Tensor, // [8, C, kv]

    pub mlp_up: Tensor,   // [4, C, mlp]
    pub mlp_down: Tensor, // [4, mlp, C]
    pub blocks: Vec<BlockWeights>,
    pub rope_cos: Tensor, // [T, half]
    pub rope_sin: Tensor,
    /// Audit 4 P1b: persistent bf16 copies of GEMM banks (masters stay f32).
    pub bf16_banks: Option<Bf16WeightBanks>,
}

/// Persistent bf16 weight banks (f32 masters remain source of truth for optim).
pub struct Bf16WeightBanks {
    pub qo_bank: Tensor,
    pub kv_bank: Tensor,
    pub mlp_up: Tensor,
    pub mlp_down: Tensor,
    pub ve_proj: Tensor,
    pub bigram_proj: Tensor,
}

impl Weights {
    pub fn load_from_golden(
        rt: &Arc<GpuRuntime>,
        golden: &Path,
        cfg: ModelConfig,
    ) -> Result<Self, String> {
        Self::load_from_python_npy(rt, &golden.join("weights_init"), cfg)
    }

    /// Load a Python-layout weight tree (`tok_emb/weight.npy`, banks as `[out,in]`, …).
    /// Same layout as `golden/weights_init` and `dump_stepN/weights`.
    pub fn load_from_python_npy(
        rt: &Arc<GpuRuntime>,
        wdir: &Path,
        cfg: ModelConfig,
    ) -> Result<Self, String> {
        let tok = load_f32(rt, &wdir.join("tok_emb/weight.npy"))?;
        let mut tok_t_data = tok.buffer.read_f32();
        let mut tok_t_shape = tok.shape().to_vec();
        transpose_last2(&mut tok_t_data, &mut tok_t_shape)?;
        let tok_emb_t = upload(rt, &tok_t_shape, &tok_t_data)?;

        let bigram_emb = load_f32(rt, &wdir.join("bigram/embed/weight.npy"))?;
        let bigram_proj = load_linear_transpose(rt, &wdir.join("bigram/proj/weight.npy"))?;
        let bigram_scale = load_f32(rt, &wdir.join("bigram/scale.npy"))?;
        let smear_gate = load_f32(rt, &wdir.join("smear/gate.npy"))?;

        let ve_emb = load_f32(rt, &wdir.join("ve_shared/embed/weight.npy"))?;
        let ve_proj = load_linear_transpose(rt, &wdir.join("ve_shared/proj/weight.npy"))?;
        let ve_scale = load_f32(rt, &wdir.join("ve_shared/scale.npy"))?;
        let mut ve_layer_scales = Vec::new();
        for i in 0..cfg.ve_layers.len() {
            ve_layer_scales.push(load_f32(
                rt,
                &wdir.join(format!("ve_layer_scales/{i}.npy")),
            )?);
        }

        let skip_weights = load_f32(rt, &wdir.join("skip_weights.npy"))?;
        let c = cfg.model_dim;
        let kv = cfg.kv_dim();

        let n_attn = cfg.mixer_count(MixerKind::Attention);
        let n_mingru = cfg.mixer_count(MixerKind::MinGRU);
        let n_mamba = cfg.mixer_count(MixerKind::Mamba2);

        let qo_bank = if wdir.join("qo_bank.npy").exists() && n_attn > 0 {
            load_bank_transpose(rt, &wdir.join("qo_bank.npy"))?
        } else {
            upload(rt, &[n_attn.max(0), c, c], &[])?
        };
        let kv_bank = if wdir.join("kv_bank.npy").exists() && n_attn > 0 {
            load_bank_transpose(rt, &wdir.join("kv_bank.npy"))?
        } else {
            upload(rt, &[n_attn.max(0), c, kv], &[])?
        };

        let mingru_to_z = if wdir.join("mingru_to_z.npy").exists() && n_mingru > 0 {
            Some(load_bank_transpose(rt, &wdir.join("mingru_to_z.npy"))?)
        } else {
            None
        };
        let mingru_to_h = if wdir.join("mingru_to_h.npy").exists() && n_mingru > 0 {
            Some(load_bank_transpose(rt, &wdir.join("mingru_to_h.npy"))?)
        } else {
            None
        };
        let mingru_out = if wdir.join("mingru_out.npy").exists() && n_mingru > 0 {
            Some(load_bank_transpose(rt, &wdir.join("mingru_out.npy"))?)
        } else {
            None
        };
        let mingru_v_proj = if wdir.join("mingru_v_proj.npy").exists() && cfg.value_residual {
            Some(load_bank_transpose(rt, &wdir.join("mingru_v_proj.npy"))?)
        } else {
            None
        };
        let mingru_v0_up = if wdir.join("mingru_v0_up.npy").exists() && cfg.value_residual {
            Some(load_bank_transpose(rt, &wdir.join("mingru_v0_up.npy"))?)
        } else {
            None
        };

        let mamba_in_proj = if wdir.join("mamba_in_proj.npy").exists() && n_mamba > 0 {
            Some(load_bank_transpose(rt, &wdir.join("mamba_in_proj.npy"))?)
        } else {
            None
        };
        let mamba_conv1d_weight = if wdir.join("mamba_conv1d_weight.npy").exists() && n_mamba > 0 {
            Some(load_f32(rt, &wdir.join("mamba_conv1d_weight.npy"))?)
        } else {
            None
        };
        let mamba_conv1d_bias = if wdir.join("mamba_conv1d_bias.npy").exists() && n_mamba > 0 {
            Some(load_f32(rt, &wdir.join("mamba_conv1d_bias.npy"))?)
        } else {
            None
        };
        let mamba_out_proj = if wdir.join("mamba_out_proj.npy").exists() && n_mamba > 0 {
            Some(load_bank_transpose(rt, &wdir.join("mamba_out_proj.npy"))?)
        } else {
            None
        };
        let mamba_a_log = if wdir.join("mamba_a_log.npy").exists() && n_mamba > 0 {
            Some(load_f32(rt, &wdir.join("mamba_a_log.npy"))?)
        } else {
            None
        };
        let mamba_d = if wdir.join("mamba_d.npy").exists() && n_mamba > 0 {
            Some(load_f32(rt, &wdir.join("mamba_d.npy"))?)
        } else {
            None
        };
        let mamba_dt_bias = if wdir.join("mamba_dt_bias.npy").exists() && n_mamba > 0 {
            Some(load_f32(rt, &wdir.join("mamba_dt_bias.npy"))?)
        } else {
            None
        };
        let mamba_norm = if wdir.join("mamba_norm.npy").exists() && n_mamba > 0 {
            Some(load_f32(rt, &wdir.join("mamba_norm.npy"))?)
        } else {
            None
        };

        let mlp_up = load_bank_transpose(rt, &wdir.join("mlp_up_bank.npy"))?;
        let mlp_down = load_bank_transpose(rt, &wdir.join("mlp_down_bank.npy"))?;

        let mut blocks = Vec::new();
        for i in 0..cfg.num_layers {
            let base = wdir.join(format!("blocks/{i}"));
            blocks.push(BlockWeights {
                q_gain: load_f32(rt, &base.join("attn/q_gain.npy"))?,
                vr_lambda: load_f32(rt, &base.join("attn/vr_lambda.npy"))?,
                attn_scale: load_f32(rt, &base.join("attn_scale.npy"))?,
                mlp_scale: load_f32(rt, &base.join("mlp_scale.npy"))?,
                resid_mix: load_f32(rt, &base.join("resid_mix.npy"))?,
            });
        }

        let (rope_cos, rope_sin) = make_rope(rt, &cfg)?;

        Ok(Self {
            cfg,
            tok_emb: tok,
            tok_emb_t,
            bigram_emb,
            bigram_proj,
            bigram_scale,
            smear_gate,
            ve_emb,
            ve_proj,
            ve_scale,
            ve_layer_scales,
            skip_weights,
            qo_bank,
            kv_bank,
            mingru_to_z,
            mingru_to_h,
            mingru_out,
            mingru_v_proj,
            mingru_v0_up,
            mamba_in_proj,
            mamba_conv1d_weight,
            mamba_conv1d_bias,
            mamba_out_proj,
            mamba_a_log,
            mamba_d,
            mamba_dt_bias,
            mamba_norm,
            mlp_up,
            mlp_down,
            blocks,
            rope_cos,
            rope_sin,
            bf16_banks: None,
        })
    }

    /// Allocate (once) and refresh persistent bf16 GEMM banks from f32 masters.
    pub fn ensure_bf16_banks(&mut self, rt: &Arc<GpuRuntime>) -> Result<(), String> {
        use crate::gemm::cast_f32_to_bf16_hot;
        let banks = Bf16WeightBanks {
            qo_bank: cast_f32_to_bf16_hot(&self.qo_bank)?,
            kv_bank: cast_f32_to_bf16_hot(&self.kv_bank)?,
            mlp_up: cast_f32_to_bf16_hot(&self.mlp_up)?,
            mlp_down: cast_f32_to_bf16_hot(&self.mlp_down)?,
            ve_proj: cast_f32_to_bf16_hot(&self.ve_proj)?,
            bigram_proj: cast_f32_to_bf16_hot(&self.bigram_proj)?,
        };
        let _ = rt;
        self.bf16_banks = Some(banks);
        Ok(())
    }

    /// Re-cast f32 masters → bf16 banks after an optim step (Bf16 training path).
    pub fn refresh_bf16_banks(&mut self, rt: &Arc<GpuRuntime>) -> Result<(), String> {
        if self.bf16_banks.is_none() {
            return self.ensure_bf16_banks(rt);
        }
        use crate::gemm::cast_f32_to_bf16_into;
        let Some(ref banks) = self.bf16_banks else {
            return Ok(());
        };
        // In-place refresh into hot-resident banks (no realloc / residency churn).
        cast_f32_to_bf16_into(&self.qo_bank, &banks.qo_bank)?;
        cast_f32_to_bf16_into(&self.kv_bank, &banks.kv_bank)?;
        cast_f32_to_bf16_into(&self.mlp_up, &banks.mlp_up)?;
        cast_f32_to_bf16_into(&self.mlp_down, &banks.mlp_down)?;
        cast_f32_to_bf16_into(&self.ve_proj, &banks.ve_proj)?;
        cast_f32_to_bf16_into(&self.bigram_proj, &banks.bigram_proj)?;
        Ok(())
    }

    /// Slice bank matrix `bank[i]` as a zero-copy view (byte offset into bank).
    pub fn bank_matrix(
        &self,
        _rt: &Arc<GpuRuntime>,
        bank: &Tensor,
        index: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor, String> {
        let elems = rows * cols;
        Ok(bank.view(&[rows, cols], index * elems))
    }
}

fn upload(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Result<Tensor, String> {
    let t = rt.alloc_tensor_f32_hot(shape)?;
    t.buffer.write_f32(data);
    Ok(t)
}

fn load_f32(rt: &Arc<GpuRuntime>, path: &Path) -> Result<Tensor, String> {
    let arr = read_npy(path)?;
    let data = arr.f32_slice()?;
    // Scalar → shape [1]
    let shape = if arr.shape.is_empty() {
        vec![1]
    } else {
        arr.shape.clone()
    };
    upload(rt, &shape, data)
}

fn load_linear_transpose(rt: &Arc<GpuRuntime>, path: &Path) -> Result<Tensor, String> {
    let arr = read_npy(path)?;
    let mut data = arr.f32_slice()?.to_vec();
    let mut shape = arr.shape.clone();
    transpose_last2(&mut data, &mut shape)?;
    upload(rt, &shape, &data)
}

fn load_bank_transpose(rt: &Arc<GpuRuntime>, path: &Path) -> Result<Tensor, String> {
    load_linear_transpose(rt, path)
}

pub(crate) fn make_rope(
    rt: &Arc<GpuRuntime>,
    cfg: &ModelConfig,
) -> Result<(Tensor, Tensor), String> {
    let rd = cfg.rope_dims;
    let half = rd / 2;
    let mut cos = vec![0.0f32; cfg.seq_len * half];
    let mut sin = vec![0.0f32; cfg.seq_len * half];
    for i in 0..half {
        let inv_freq = 1.0 / cfg.rope_base.powf((2 * i) as f32 / rd as f32);
        for t in 0..cfg.seq_len {
            let ang = (t as f32) * inv_freq;
            cos[t * half + i] = ang.cos();
            sin[t * half + i] = ang.sin();
        }
    }
    Ok((
        upload(rt, &[cfg.seq_len, half], &cos)?,
        upload(rt, &[cfg.seq_len, half], &sin)?,
    ))
}
