//! On-device clip + banked Muon NS5 + fused AdamW+EMA (Phase 3).
//!
//! Steady-state clip keeps the coefficient in a 4-byte device buffer consumed by
//! scale kernels — no host `into_scalar` / readback of the norm except when
//! logging (`read_norm = true`).
//!
//! **Default (`ClipMode::Soft`):** after grad scale, **Muon** step+WD *=
//! `sqrt(clip_coef)`; **AdamW** (embed + scalars) *= `clip_coef` (Match). Soft √c
//! on Muon alone preserves bank learning under chronic clip; Soft-on-AdamW was
//! rejected after 20k FA_TILED Soft exploded ~3.5–4.5k (vr_λ / attn_scale climb,
//! Soft clip→0, EMA rebound 2.06→2.39). Soft-Muon + Match-AdamW continues stable
//! through ≥8k from a Soft 3k dump.
//!
//! **`ClipMode::Match` (`--clip-match`):** AdamW+Muon *= `clip_coef` (most conservative).
//! **`ClipMode::Python` (`--clip-python`):** unity coef — CUDA formula, diverges.

use std::sync::Arc;

use objc2_metal::MTLComputePipelineState;
use serde::{Deserialize, Serialize};

use crate::dispatch::{dispatch_1d, set_f32, set_gpu_buf, set_tensor, set_u32, Binder};
use crate::model_bwd::Grads;
use crate::optimizer_registry::OptimizerKind;
use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::{DType, GpuBuffer, Tensor};
use crate::weights::Weights;

pub const NS_A: f32 = 3.4445;
pub const NS_B: f32 = -4.7750;
pub const NS_C: f32 = 2.0315;
pub const NS_EPS: f32 = 1e-7;
pub const NS_STEPS: u32 = 5;
pub const CLIP_EPS: f32 = 1e-6;
pub const OPTIM_ATOL: f32 = 1e-4;
pub const POLAR_COEFFS: [(f32, f32, f32); 5] = [
    (8.156_555, -22.483_294, 15.878_77),
    (4.042_93, -2.808_917_5, 0.500_017_8),
    (3.891_667_8, -2.772_484_3, 0.506_064_83),
    (3.285_753_7, -2.368_129_5, 0.464_490_23),
    (2.346_541_4, -1.709_782_8, 0.423_235_5),
];

#[derive(Clone, Copy)]
enum MuonOrthogonalizer {
    NewtonSchulz(u32),
    PolarExpress,
}

impl MuonOrthogonalizer {
    fn kernel_kind(self) -> u32 {
        u32::from(matches!(self, Self::PolarExpress))
    }

    fn norm_scale(self) -> f32 {
        if matches!(self, Self::PolarExpress) {
            1.02
        } else {
            1.0
        }
    }

    fn coefficients(self) -> Vec<(f32, f32, f32)> {
        match self {
            Self::NewtonSchulz(steps) => vec![(NS_A, NS_B, NS_C); steps as usize],
            Self::PolarExpress => POLAR_COEFFS.to_vec(),
        }
    }
}

/// Sota training hyperparams matching golden exporter / burn-port `sota_toy`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OptimHyperparams {
    pub matrix_lr: f32,
    pub tied_embed_lr: f32,
    pub scalar_lr: f32,
    pub weight_decay: f32,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_eps: f32,
    pub muon_momentum_start: f32,
    pub muon_momentum_end: f32,
    pub muon_momentum_warmup: usize,
    pub sophia_rho: f32,
    pub sophia_hessian_interval: usize,
    pub schedule_free_warmup: usize,
    pub mona_beta_a: f32,
    pub mona_alpha: f32,
    pub muown_direction_scale: f32,
    pub grad_clip: f32,
    pub ema_decay: f32,
}

impl Default for OptimHyperparams {
    fn default() -> Self {
        Self {
            matrix_lr: 0.025,
            tied_embed_lr: 0.035,
            scalar_lr: 0.025,
            weight_decay: 0.04,
            adam_beta1: 0.9,
            adam_beta2: 0.95,
            adam_eps: 1e-8,
            muon_momentum_start: 0.92,
            muon_momentum_end: 0.95,
            muon_momentum_warmup: 1500,
            sophia_rho: 0.04,
            sophia_hessian_interval: 10,
            schedule_free_warmup: 0,
            mona_beta_a: 0.99,
            mona_alpha: 0.0,
            muown_direction_scale: 0.2,
            grad_clip: 0.3,
            ema_decay: 0.997,
        }
    }
}

impl OptimHyperparams {
    pub fn muon_momentum(&self, step: usize) -> f32 {
        let frac = (step as f32 / self.muon_momentum_warmup as f32).min(1.0);
        (1.0 - frac) * self.muon_momentum_start + frac * self.muon_momentum_end
    }
}

/// LR multiplier schedule for Soft long-horizon runs.
///
/// **Classic (burn-port / Python 20k):** `--warmdown N` only → linear 1→0 over the
/// final N steps (`WARMDOWN_ITERS=3500` on 20k). No warmup.
///
/// **Long Soft (100k):** constant LR then an earlier main warmdown to a floor,
/// hold, optional final decay — see [`LrSchedule`]. Soft-split under constant LR
/// rebounds ~21k (gnorm→thousands); linear 1→0 only in the last 10% never fires.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct LrSchedule {
    pub total_iters: usize,
    /// Absolute step where main warmdown begins. `None` → `total_iters - warmdown_iters`.
    pub warmdown_start: Option<usize>,
    /// Length of main linear decay (1 → [`Self::lr_floor`]). `0` with no start → off.
    pub warmdown_iters: usize,
    /// Multiplier after the main warmdown window (held until final warmdown).
    pub lr_floor: f32,
    /// Last N steps: linear from the held level → 0. `0` = no final decay.
    pub final_warmdown: usize,
}

impl LrSchedule {
    /// Classic final-N linear warmdown (floor 0, no final phase).
    pub fn from_warmdown(total_iters: usize, warmdown_iters: usize) -> Self {
        Self {
            total_iters,
            warmdown_start: None,
            warmdown_iters,
            lr_floor: 0.0,
            final_warmdown: 0,
        }
    }

    /// Resolved main-warmdown `[start, start+len)`.
    pub fn main_window(&self) -> (usize, usize) {
        let start = self.warmdown_start.unwrap_or_else(|| {
            if self.warmdown_iters == 0 {
                self.total_iters
            } else {
                self.total_iters.saturating_sub(self.warmdown_iters)
            }
        });
        let len = if self.warmdown_iters > 0 {
            self.warmdown_iters
        } else if self.warmdown_start.is_some() {
            self.total_iters.saturating_sub(start)
        } else {
            0
        };
        (start.min(self.total_iters), len)
    }

    /// LR mul ignoring `--final-warmdown` (constant / main decay / floor hold).
    pub fn base_mul_at(&self, step: usize) -> f32 {
        let (start, len) = self.main_window();
        if len == 0 || step < start {
            1.0
        } else if step >= start.saturating_add(len) {
            self.lr_floor.max(0.0)
        } else {
            let t = (step - start) as f32 / len as f32;
            (1.0 + t * (self.lr_floor - 1.0)).max(0.0)
        }
    }

    pub fn mul_at(&self, step: usize) -> f32 {
        let base = self.base_mul_at(step);
        if self.final_warmdown == 0 {
            return base;
        }
        let fs = self.total_iters.saturating_sub(self.final_warmdown);
        if step < fs {
            return base;
        }
        if self.total_iters == 0 || step >= self.total_iters {
            return 0.0;
        }
        let base_fs = self.base_mul_at(fs);
        (base_fs * ((self.total_iters - step) as f32 / self.final_warmdown as f32)).max(0.0)
    }
}

/// Linear LR warmdown over the final `warmdown_iters` of a `total_iters` run.
/// Matches burn-port / Python sprint (`WARMDOWN_ITERS=3500` on 20k). No warmup.
pub fn lr_mul_at(step: usize, total_iters: usize, warmdown_iters: usize) -> f32 {
    LrSchedule::from_warmdown(total_iters, warmdown_iters).mul_at(step)
}

/// How AdamW/Muon consume the on-device clip coefficient after grads are scaled.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ClipMode {
    /// AdamW+Muon step+WD *= clip_coef (most conservative).
    Match,
    /// Optim uses c=1 (CUDA formula; diverges ~2500 on metal).
    Python,
    /// Muon *= sqrt(clip_coef); AdamW *= clip_coef — default (long-horizon Soft).
    #[default]
    Soft,
}

/// Device-resident clip buffers (4-byte coef + optional norm for log steps).
pub struct ClipState {
    pub total_sq: GpuBuffer,
    pub clip_coef: GpuBuffer,
    pub clip_soft: GpuBuffer,
    pub norm: GpuBuffer,
    /// Always 1.0 — passed to AdamW/Muon in Python mode.
    pub clip_unity: GpuBuffer,
}

impl ClipState {
    pub fn new(rt: &Arc<GpuRuntime>) -> Result<Self, String> {
        let total_sq = rt.alloc_buffer_hot(4)?;
        let clip_coef = rt.alloc_buffer_hot(4)?;
        let clip_soft = rt.alloc_buffer_hot(4)?;
        let norm = rt.alloc_buffer_hot(4)?;
        let clip_unity = rt.alloc_buffer_hot(4)?;
        total_sq.zero();
        clip_coef.write_f32(&[1.0]);
        clip_soft.write_f32(&[1.0]);
        norm.zero();
        clip_unity.write_f32(&[1.0]);
        Ok(Self {
            total_sq,
            clip_coef,
            clip_soft,
            norm,
            clip_unity,
        })
    }

    /// Coef for AdamW step+WD (embed + scalars). Soft uses Match `clip_coef`.
    pub fn adamw_coef(&self, mode: ClipMode) -> &GpuBuffer {
        match mode {
            ClipMode::Match | ClipMode::Soft => &self.clip_coef,
            ClipMode::Python => &self.clip_unity,
        }
    }

    /// Coef for Muon step+WD. Soft uses `sqrt(clip_coef)`.
    pub fn muon_coef(&self, mode: ClipMode) -> &GpuBuffer {
        match mode {
            ClipMode::Match => &self.clip_coef,
            ClipMode::Python => &self.clip_unity,
            ClipMode::Soft => &self.clip_soft,
        }
    }

    /// Back-compat alias → [`Self::muon_coef`].
    pub fn optim_coef(&self, mode: ClipMode) -> &GpuBuffer {
        self.muon_coef(mode)
    }
}

fn zeros_like(rt: &Arc<GpuRuntime>, t: &Tensor) -> Result<Tensor, String> {
    let z = rt.alloc_tensor_f32_hot(&t.shape())?;
    z.buffer.zero();
    Ok(z)
}

fn clone_like(rt: &Arc<GpuRuntime>, t: &Tensor) -> Result<Tensor, String> {
    let z = rt.alloc_tensor_f32_hot(&t.shape())?;
    z.buffer.write_f32(&t.buffer.read_f32());
    Ok(z)
}

/// AdamW moments for one parameter tensor.
pub struct AdamSlot {
    pub exp_avg: Tensor,
    pub exp_avg_sq: Tensor,
    pub aux: Tensor,
    pub origin: Tensor,
}

impl AdamSlot {
    fn zeros(rt: &Arc<GpuRuntime>, t: &Tensor) -> Result<Self, String> {
        Ok(Self {
            exp_avg: zeros_like(rt, t)?,
            exp_avg_sq: zeros_like(rt, t)?,
            aux: zeros_like(rt, t)?,
            origin: zeros_like(rt, t)?,
        })
    }

    fn for_kind(rt: &Arc<GpuRuntime>, t: &Tensor, kind: OptimizerKind) -> Result<Self, String> {
        Ok(Self {
            exp_avg: if kind == OptimizerKind::ScheduleFreeAdamw {
                clone_like(rt, t)?
            } else {
                zeros_like(rt, t)?
            },
            exp_avg_sq: zeros_like(rt, t)?,
            aux: zeros_like(rt, t)?,
            origin: if kind == OptimizerKind::Prodigy {
                clone_like(rt, t)?
            } else {
                zeros_like(rt, t)?
            },
        })
    }
}

pub struct BlockAdam {
    pub q_gain: AdamSlot,
    pub vr_lambda: AdamSlot,
    pub attn_scale: AdamSlot,
    pub mlp_scale: AdamSlot,
    pub resid_mix: AdamSlot,
}

/// Full optimizer + EMA state.
pub struct OptimState {
    pub kind: OptimizerKind,
    pub hp: OptimHyperparams,
    pub clip: ClipState,
    /// Default [`ClipMode::Match`] for metal stability.
    pub clip_mode: ClipMode,
    pub step: usize, // completed Adam/Muon steps (0 before first)
    // Embed AdamW
    pub tok_emb: AdamSlot,
    pub bigram_emb: AdamSlot,
    pub ve_emb: AdamSlot,
    // Scalar AdamW
    pub bigram_proj: AdamSlot,
    pub bigram_scale: AdamSlot,
    pub smear_gate: AdamSlot,
    pub ve_proj: AdamSlot,
    pub ve_scale: AdamSlot,
    pub ve_layer_scales: Vec<AdamSlot>,
    pub skip_weights: AdamSlot,
    pub blocks: Vec<BlockAdam>,
    // Muon momentum (Burn layout)
    pub mom_qo: Tensor,
    pub mom_kv: Tensor,
    pub mom_up: Tensor,
    pub mom_dn: Tensor,
    pub var_qo: Tensor,
    pub var_kv: Tensor,
    pub var_up: Tensor,
    pub var_dn: Tensor,
    pub prev_qo: Tensor,
    pub prev_kv: Tensor,
    pub prev_up: Tensor,
    pub prev_dn: Tensor,
    pub mag_v_qo: Tensor,
    pub mag_v_kv: Tensor,
    pub mag_v_up: Tensor,
    pub mag_v_dn: Tensor,
    pub prodigy_d: f32,
    pub prodigy_d_max: f32,
    pub prodigy_d_numerator: f32,
    pub muon_scratch: GpuBuffer,
    // EMA copies (same layout as Weights)
    pub ema_tok_emb: Tensor,
    pub ema_bigram_emb: Tensor,
    pub ema_bigram_proj: Tensor,
    pub ema_bigram_scale: Tensor,
    pub ema_smear_gate: Tensor,
    pub ema_ve_emb: Tensor,
    pub ema_ve_proj: Tensor,
    pub ema_ve_scale: Tensor,
    pub ema_ve_layer_scales: Vec<Tensor>,
    pub ema_skip_weights: Tensor,
    pub ema_qo: Tensor,
    pub ema_kv: Tensor,
    pub ema_up: Tensor,
    pub ema_dn: Tensor,
    pub ema_blocks: Vec<BlockEma>,
    pub mom_mingru_to_z: Option<Tensor>,
    pub var_mingru_to_z: Option<Tensor>,
    pub prev_mingru_to_z: Option<Tensor>,
    pub mag_v_mingru_to_z: Option<Tensor>,
    pub ema_mingru_to_z: Option<Tensor>,
    pub mom_mingru_to_h: Option<Tensor>,
    pub var_mingru_to_h: Option<Tensor>,
    pub prev_mingru_to_h: Option<Tensor>,
    pub mag_v_mingru_to_h: Option<Tensor>,
    pub ema_mingru_to_h: Option<Tensor>,
    pub mom_mingru_out: Option<Tensor>,
    pub var_mingru_out: Option<Tensor>,
    pub prev_mingru_out: Option<Tensor>,
    pub mag_v_mingru_out: Option<Tensor>,
    pub ema_mingru_out: Option<Tensor>,
    pub mom_mingru_v_proj: Option<Tensor>,
    pub var_mingru_v_proj: Option<Tensor>,
    pub prev_mingru_v_proj: Option<Tensor>,
    pub mag_v_mingru_v_proj: Option<Tensor>,
    pub ema_mingru_v_proj: Option<Tensor>,
    pub mom_mingru_v0_up: Option<Tensor>,
    pub var_mingru_v0_up: Option<Tensor>,
    pub prev_mingru_v0_up: Option<Tensor>,
    pub mag_v_mingru_v0_up: Option<Tensor>,
    pub ema_mingru_v0_up: Option<Tensor>,
    pub mom_mamba_in_proj: Option<Tensor>,
    pub var_mamba_in_proj: Option<Tensor>,
    pub prev_mamba_in_proj: Option<Tensor>,
    pub mag_v_mamba_in_proj: Option<Tensor>,
    pub ema_mamba_in_proj: Option<Tensor>,
    pub mamba_conv1d_weight: Option<AdamSlot>,
    pub ema_mamba_conv1d_weight: Option<Tensor>,
    pub mamba_conv1d_bias: Option<AdamSlot>,
    pub ema_mamba_conv1d_bias: Option<Tensor>,
    pub mom_mamba_out_proj: Option<Tensor>,
    pub var_mamba_out_proj: Option<Tensor>,
    pub prev_mamba_out_proj: Option<Tensor>,
    pub mag_v_mamba_out_proj: Option<Tensor>,
    pub ema_mamba_out_proj: Option<Tensor>,
    pub mamba_a_log: Option<AdamSlot>,
    pub ema_mamba_a_log: Option<Tensor>,
    pub mamba_d: Option<AdamSlot>,
    pub ema_mamba_d: Option<Tensor>,
    pub mamba_dt_bias: Option<AdamSlot>,
    pub ema_mamba_dt_bias: Option<Tensor>,
    pub mamba_norm: Option<AdamSlot>,
    pub ema_mamba_norm: Option<Tensor>,
}

pub struct BlockEma {
    pub q_gain: Tensor,
    pub vr_lambda: Tensor,
    pub attn_scale: Tensor,
    pub mlp_scale: Tensor,
    pub resid_mix: Tensor,
}

impl OptimState {
    pub fn new(rt: &Arc<GpuRuntime>, w: &Weights, hp: OptimHyperparams) -> Result<Self, String> {
        Self::new_for_kind(rt, w, hp, OptimizerKind::default())
    }

    pub fn new_for_kind(
        rt: &Arc<GpuRuntime>,
        w: &Weights,
        hp: OptimHyperparams,
        kind: OptimizerKind,
    ) -> Result<Self, String> {
        let mut ve_layer_scales = Vec::new();
        let mut ema_ve_layer_scales = Vec::new();
        for s in &w.ve_layer_scales {
            ve_layer_scales.push(AdamSlot::for_kind(rt, s, kind)?);
            ema_ve_layer_scales.push(clone_like(rt, s)?);
        }
        let mut blocks = Vec::new();
        let mut ema_blocks = Vec::new();
        for b in &w.blocks {
            blocks.push(BlockAdam {
                q_gain: AdamSlot::for_kind(rt, &b.q_gain, kind)?,
                vr_lambda: AdamSlot::for_kind(rt, &b.vr_lambda, kind)?,
                attn_scale: AdamSlot::for_kind(rt, &b.attn_scale, kind)?,
                mlp_scale: AdamSlot::for_kind(rt, &b.mlp_scale, kind)?,
                resid_mix: AdamSlot::for_kind(rt, &b.resid_mix, kind)?,
            });
            ema_blocks.push(BlockEma {
                q_gain: clone_like(rt, &b.q_gain)?,
                vr_lambda: clone_like(rt, &b.vr_lambda)?,
                attn_scale: clone_like(rt, &b.attn_scale)?,
                mlp_scale: clone_like(rt, &b.mlp_scale)?,
                resid_mix: clone_like(rt, &b.resid_mix)?,
            });
        }

        let scratch_bytes = muon_scratch_bytes(w) * 4;
        let muon_scratch = rt.alloc_buffer_hot(scratch_bytes)?;
        muon_scratch.zero();

        Ok(Self {
            kind,
            hp,
            clip: ClipState::new(rt)?,
            clip_mode: ClipMode::Soft,
            step: 0,
            tok_emb: AdamSlot::for_kind(rt, &w.tok_emb, kind)?,
            mom_mingru_to_z: w.mingru_to_z.as_ref().map(|t| clone_like(rt, t).unwrap()),
            var_mingru_to_z: w.mingru_to_z.as_ref().map(|t| clone_like(rt, t).unwrap()),
            prev_mingru_to_z: w.mingru_to_z.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mag_v_mingru_to_z: w.mingru_to_z.as_ref().map(|t| clone_like(rt, t).unwrap()),
            ema_mingru_to_z: w.mingru_to_z.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mom_mingru_to_h: w.mingru_to_h.as_ref().map(|t| clone_like(rt, t).unwrap()),
            var_mingru_to_h: w.mingru_to_h.as_ref().map(|t| clone_like(rt, t).unwrap()),
            prev_mingru_to_h: w.mingru_to_h.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mag_v_mingru_to_h: w.mingru_to_h.as_ref().map(|t| clone_like(rt, t).unwrap()),
            ema_mingru_to_h: w.mingru_to_h.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mom_mingru_out: w.mingru_out.as_ref().map(|t| clone_like(rt, t).unwrap()),
            var_mingru_out: w.mingru_out.as_ref().map(|t| clone_like(rt, t).unwrap()),
            prev_mingru_out: w.mingru_out.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mag_v_mingru_out: w.mingru_out.as_ref().map(|t| clone_like(rt, t).unwrap()),
            ema_mingru_out: w.mingru_out.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mom_mingru_v_proj: w.mingru_v_proj.as_ref().map(|t| clone_like(rt, t).unwrap()),
            var_mingru_v_proj: w.mingru_v_proj.as_ref().map(|t| clone_like(rt, t).unwrap()),
            prev_mingru_v_proj: w.mingru_v_proj.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mag_v_mingru_v_proj: w.mingru_v_proj.as_ref().map(|t| clone_like(rt, t).unwrap()),
            ema_mingru_v_proj: w.mingru_v_proj.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mom_mingru_v0_up: w.mingru_v0_up.as_ref().map(|t| clone_like(rt, t).unwrap()),
            var_mingru_v0_up: w.mingru_v0_up.as_ref().map(|t| clone_like(rt, t).unwrap()),
            prev_mingru_v0_up: w.mingru_v0_up.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mag_v_mingru_v0_up: w.mingru_v0_up.as_ref().map(|t| clone_like(rt, t).unwrap()),
            ema_mingru_v0_up: w.mingru_v0_up.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mom_mamba_in_proj: w.mamba_in_proj.as_ref().map(|t| clone_like(rt, t).unwrap()),
            var_mamba_in_proj: w.mamba_in_proj.as_ref().map(|t| clone_like(rt, t).unwrap()),
            prev_mamba_in_proj: w.mamba_in_proj.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mag_v_mamba_in_proj: w.mamba_in_proj.as_ref().map(|t| clone_like(rt, t).unwrap()),
            ema_mamba_in_proj: w.mamba_in_proj.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mamba_conv1d_weight: w
                .mamba_conv1d_weight
                .as_ref()
                .map(|t| AdamSlot::for_kind(rt, t, kind).unwrap()),
            ema_mamba_conv1d_weight: w
                .mamba_conv1d_weight
                .as_ref()
                .map(|t| clone_like(rt, t).unwrap()),
            mamba_conv1d_bias: w
                .mamba_conv1d_bias
                .as_ref()
                .map(|t| AdamSlot::for_kind(rt, t, kind).unwrap()),
            ema_mamba_conv1d_bias: w
                .mamba_conv1d_bias
                .as_ref()
                .map(|t| clone_like(rt, t).unwrap()),
            mom_mamba_out_proj: w
                .mamba_out_proj
                .as_ref()
                .map(|t| clone_like(rt, t).unwrap()),
            var_mamba_out_proj: w
                .mamba_out_proj
                .as_ref()
                .map(|t| clone_like(rt, t).unwrap()),
            prev_mamba_out_proj: w
                .mamba_out_proj
                .as_ref()
                .map(|t| clone_like(rt, t).unwrap()),
            mag_v_mamba_out_proj: w
                .mamba_out_proj
                .as_ref()
                .map(|t| clone_like(rt, t).unwrap()),
            ema_mamba_out_proj: w
                .mamba_out_proj
                .as_ref()
                .map(|t| clone_like(rt, t).unwrap()),
            mamba_a_log: w
                .mamba_a_log
                .as_ref()
                .map(|t| AdamSlot::for_kind(rt, t, kind).unwrap()),
            ema_mamba_a_log: w.mamba_a_log.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mamba_d: w
                .mamba_d
                .as_ref()
                .map(|t| AdamSlot::for_kind(rt, t, kind).unwrap()),
            ema_mamba_d: w.mamba_d.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mamba_dt_bias: w
                .mamba_dt_bias
                .as_ref()
                .map(|t| AdamSlot::for_kind(rt, t, kind).unwrap()),
            ema_mamba_dt_bias: w.mamba_dt_bias.as_ref().map(|t| clone_like(rt, t).unwrap()),
            mamba_norm: w
                .mamba_norm
                .as_ref()
                .map(|t| AdamSlot::for_kind(rt, t, kind).unwrap()),
            ema_mamba_norm: w.mamba_norm.as_ref().map(|t| clone_like(rt, t).unwrap()),

            bigram_emb: AdamSlot::for_kind(rt, &w.bigram_emb, kind)?,
            ve_emb: AdamSlot::for_kind(rt, &w.ve_emb, kind)?,
            bigram_proj: AdamSlot::for_kind(rt, &w.bigram_proj, kind)?,
            bigram_scale: AdamSlot::for_kind(rt, &w.bigram_scale, kind)?,
            smear_gate: AdamSlot::for_kind(rt, &w.smear_gate, kind)?,
            ve_proj: AdamSlot::for_kind(rt, &w.ve_proj, kind)?,
            ve_scale: AdamSlot::for_kind(rt, &w.ve_scale, kind)?,
            ve_layer_scales,
            skip_weights: AdamSlot::for_kind(rt, &w.skip_weights, kind)?,
            blocks,
            mom_qo: if kind == OptimizerKind::ScheduleFreeAdamw {
                clone_like(rt, &w.qo_bank)?
            } else {
                zeros_like(rt, &w.qo_bank)?
            },
            mom_kv: if kind == OptimizerKind::ScheduleFreeAdamw {
                clone_like(rt, &w.kv_bank)?
            } else {
                zeros_like(rt, &w.kv_bank)?
            },
            mom_up: if kind == OptimizerKind::ScheduleFreeAdamw {
                clone_like(rt, &w.mlp_up)?
            } else {
                zeros_like(rt, &w.mlp_up)?
            },
            mom_dn: if kind == OptimizerKind::ScheduleFreeAdamw {
                clone_like(rt, &w.mlp_down)?
            } else {
                zeros_like(rt, &w.mlp_down)?
            },
            var_qo: zeros_like(rt, &w.qo_bank)?,
            var_kv: zeros_like(rt, &w.kv_bank)?,
            var_up: zeros_like(rt, &w.mlp_up)?,
            var_dn: zeros_like(rt, &w.mlp_down)?,
            prev_qo: if matches!(kind, OptimizerKind::MuownAdamw | OptimizerKind::Prodigy) {
                clone_like(rt, &w.qo_bank)?
            } else {
                zeros_like(rt, &w.qo_bank)?
            },
            prev_kv: if matches!(kind, OptimizerKind::MuownAdamw | OptimizerKind::Prodigy) {
                clone_like(rt, &w.kv_bank)?
            } else {
                zeros_like(rt, &w.kv_bank)?
            },
            prev_up: if matches!(kind, OptimizerKind::MuownAdamw | OptimizerKind::Prodigy) {
                clone_like(rt, &w.mlp_up)?
            } else {
                zeros_like(rt, &w.mlp_up)?
            },
            prev_dn: if matches!(kind, OptimizerKind::MuownAdamw | OptimizerKind::Prodigy) {
                clone_like(rt, &w.mlp_down)?
            } else {
                zeros_like(rt, &w.mlp_down)?
            },
            mag_v_qo: zeros_like(rt, &w.qo_bank)?,
            mag_v_kv: zeros_like(rt, &w.kv_bank)?,
            mag_v_up: zeros_like(rt, &w.mlp_up)?,
            mag_v_dn: zeros_like(rt, &w.mlp_down)?,
            prodigy_d: 1e-6,
            prodigy_d_max: 1e-6,
            prodigy_d_numerator: 0.0,
            muon_scratch,
            ema_tok_emb: clone_like(rt, &w.tok_emb)?,
            ema_bigram_emb: clone_like(rt, &w.bigram_emb)?,
            ema_bigram_proj: clone_like(rt, &w.bigram_proj)?,
            ema_bigram_scale: clone_like(rt, &w.bigram_scale)?,
            ema_smear_gate: clone_like(rt, &w.smear_gate)?,
            ema_ve_emb: clone_like(rt, &w.ve_emb)?,
            ema_ve_proj: clone_like(rt, &w.ve_proj)?,
            ema_ve_scale: clone_like(rt, &w.ve_scale)?,
            ema_ve_layer_scales,
            ema_skip_weights: clone_like(rt, &w.skip_weights)?,
            ema_qo: clone_like(rt, &w.qo_bank)?,
            ema_kv: clone_like(rt, &w.kv_bank)?,
            ema_up: clone_like(rt, &w.mlp_up)?,
            ema_dn: clone_like(rt, &w.mlp_down)?,
            ema_blocks,
        })
    }
}

fn muon_scratch_stride(rows: usize, cols: usize) -> usize {
    let mat = rows * cols;
    let p = rows.min(cols);
    let q = rows.max(cols);
    mat + p * q + 2 * p * p
}

fn muon_scratch_bytes(w: &Weights) -> usize {
    use crate::weights::MixerKind;
    let c = w.cfg.model_dim;
    let kv = w.cfg.kv_dim();
    let mlp = w.cfg.mlp_dim;
    let n_layers = w.cfg.num_layers;
    let n_attn = w.cfg.mixer_count(MixerKind::Attention);
    let n_mingru = w.cfg.mixer_count(MixerKind::MinGRU);
    let n_mamba = w.cfg.mixer_count(MixerKind::Mamba2);
    let hid = w.cfg.mingru_hidden();
    let d_inner = w.cfg.mamba_d_inner();
    let in_out = w.cfg.mamba_in_proj_out();

    let qo = 2 * n_attn * muon_scratch_stride(c, c);
    let kvb = 2 * n_attn * muon_scratch_stride(c, kv);
    let up = n_layers * muon_scratch_stride(c, mlp);
    let dn = n_layers * muon_scratch_stride(mlp, c);
    let mut max = qo.max(kvb).max(up).max(dn);
    if n_mingru > 0 {
        max = max
            .max(n_mingru * muon_scratch_stride(c, hid))
            .max(n_mingru * muon_scratch_stride(hid, c));
        if w.cfg.value_residual {
            max = max
                .max(n_mingru * muon_scratch_stride(c, kv))
                .max(n_mingru * muon_scratch_stride(kv, hid));
        }
    }
    if n_mamba > 0 {
        max = max
            .max(n_mamba * muon_scratch_stride(c, in_out))
            .max(n_mamba * muon_scratch_stride(d_inner, c));
    }
    // One scratch buffer sized for the largest bank (reused across dispatches).
    // +256 f32 slack avoids MiB rounding edge failures on the largest SSM banks.
    max + 256
}

fn bank_scale(rows: usize, cols: usize) -> f32 {
    ((cols as f32) / (rows as f32)).max(1.0).sqrt()
}

fn collect_grad_refs(grads: &Grads) -> Vec<&Tensor> {
    let mut v: Vec<&Tensor> = vec![
        &grads.tok_emb,
        &grads.bigram_emb,
        &grads.bigram_proj,
        &grads.bigram_scale,
        &grads.smear_gate,
        &grads.ve_emb,
        &grads.ve_proj,
        &grads.ve_scale,
        &grads.skip_weights,
        &grads.qo_bank,
        &grads.kv_bank,
        &grads.mlp_up,
        &grads.mlp_down,
    ];
    for s in &grads.ve_layer_scales {
        v.push(s);
    }
    for t in [
        &grads.mingru_to_z,
        &grads.mingru_to_h,
        &grads.mingru_out,
        &grads.mingru_v_proj,
        &grads.mingru_v0_up,
        &grads.mamba_in_proj,
        &grads.mamba_conv1d_weight,
        &grads.mamba_conv1d_bias,
        &grads.mamba_out_proj,
        &grads.mamba_a_log,
        &grads.mamba_d,
        &grads.mamba_dt_bias,
        &grads.mamba_norm,
    ] {
        if let Some(g) = t {
            v.push(g);
        }
    }
    for b in &grads.blocks {
        v.push(&b.q_gain);
        v.push(&b.vr_lambda);
        v.push(&b.attn_scale);
        v.push(&b.mlp_scale);
        v.push(&b.resid_mix);
    }
    v
}

/// On-device global L2 clip. Clip coefficient stays on device.
/// When `read_norm` is true, returns the host-visible norm (log steps only).
///
/// Encode packing: all sq-reduces share one compute encoder, then coef, then
/// all scales — few encoder/CB boundaries instead of 2N+2 sync dispatches.
pub fn clip_grad_norm_device(
    rt: &Arc<GpuRuntime>,
    grads: &Grads,
    clip: &ClipState,
    max_norm: f32,
    read_norm: bool,
) -> Result<Option<f32>, String> {
    let pipes_zero = rt.pipeline("zero_scalar_f32")?;
    let pipes_sq = rt.pipeline("grad_sq_reduce_f32")?;
    let pipes_coef = rt.pipeline("clip_coef_f32")?;
    let pipes_scale = rt.pipeline("scale_by_clip_coef_f32")?;

    let refs = collect_grad_refs(grads);
    let width_sq = pipes_sq.threadExecutionWidth() as usize;
    let width_sc = pipes_scale.threadExecutionWidth() as usize;

    // total_sq = 0
    dispatch_1d(rt, &pipes_zero, 1, |bnd| {
        set_gpu_buf(bnd, &clip.total_sq, 0);
        set_u32(bnd, 1, 1);
    })?;

    // Packed sq-reduce over all grad tensors.
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&pipes_sq);
        for t in &refs {
            let n = t.numel();
            if n == 0 {
                continue;
            }
            let tpt = width_sq.min(n).max(1);
            let groups = (n + tpt - 1) / tpt;
            set_tensor(bnd, t, 0);
            set_gpu_buf(bnd, &clip.total_sq, 1);
            set_u32(bnd, n as u32, 2);
            bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
        }
        Ok(())
    })?;

    // Barrier so coef sees the atomic total_sq writes (async encoder path).
    rt.with_binder(|bnd| {
        bnd.barrier();
        bnd.set_pipeline(&pipes_coef);
        set_gpu_buf(bnd, &clip.total_sq, 0);
        set_gpu_buf(bnd, &clip.clip_coef, 1);
        set_gpu_buf(bnd, &clip.norm, 2);
        set_f32(bnd, max_norm, 3);
        set_f32(bnd, CLIP_EPS, 4);
        bnd.dispatch(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
        Ok(())
    })?;

    // Soft coef = sqrt(clip_coef) for ClipMode::Soft (cheap; always kept fresh).
    let pipes_soft = rt.pipeline("clip_soft_coef_f32")?;
    rt.with_binder(|bnd| {
        bnd.barrier();
        bnd.set_pipeline(&pipes_soft);
        set_gpu_buf(bnd, &clip.clip_coef, 0);
        set_gpu_buf(bnd, &clip.clip_soft, 1);
        bnd.dispatch(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
        Ok(())
    })?;

    // Packed scale-by-coef.
    rt.with_binder(|bnd| {
        bnd.barrier();
        bnd.set_pipeline(&pipes_scale);
        for t in &refs {
            let n = t.numel();
            if n == 0 {
                continue;
            }
            let tpt = width_sc.min(n).max(1);
            let groups = (n + tpt - 1) / tpt;
            set_tensor(bnd, t, 0);
            set_gpu_buf(bnd, &clip.clip_coef, 1);
            set_u32(bnd, n as u32, 2);
            bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
        }
        Ok(())
    })?;

    if read_norm {
        rt.synchronize()?;
        Ok(Some(clip.norm.contents_f32()[0]))
    } else {
        Ok(None)
    }
}

fn adamw_ema_one(
    rt: &Arc<GpuRuntime>,
    param: &Tensor,
    grad: &Tensor,
    slot: &AdamSlot,
    ema: Option<&Tensor>,
    clip_coef: &GpuBuffer,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    step_size: f32,
    bias2_sqrt_inv: f32,
    ema_decay: f32,
) -> Result<(), String> {
    let pipe = rt.pipeline("adamw_ema_f32")?;
    let n = param.numel();
    if n == 0 {
        return Ok(());
    }
    let do_ema = if ema.is_some() { 1u32 } else { 0u32 };
    let ema_t = ema.unwrap_or(param);
    dispatch_1d(rt, &pipe, n, |bnd| {
        encode_adamw_ema(
            bnd,
            param,
            grad,
            slot,
            ema_t,
            clip_coef,
            lr,
            beta1,
            beta2,
            eps,
            wd,
            step_size,
            bias2_sqrt_inv,
            ema_decay,
            n,
            do_ema,
        );
    })
}

fn encode_adamw_ema(
    bnd: &mut Binder<'_>,
    param: &Tensor,
    grad: &Tensor,
    slot: &AdamSlot,
    ema_t: &Tensor,
    clip_coef: &GpuBuffer,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    step_size: f32,
    bias2_sqrt_inv: f32,
    ema_decay: f32,
    n: usize,
    do_ema: u32,
) {
    set_tensor(bnd, param, 0);
    set_tensor(bnd, grad, 1);
    set_tensor(bnd, &slot.exp_avg, 2);
    set_tensor(bnd, &slot.exp_avg_sq, 3);
    set_tensor(bnd, ema_t, 4);
    set_f32(bnd, lr, 5);
    set_f32(bnd, beta1, 6);
    set_f32(bnd, beta2, 7);
    set_f32(bnd, eps, 8);
    set_f32(bnd, wd, 9);
    set_f32(bnd, step_size, 10);
    set_f32(bnd, bias2_sqrt_inv, 11);
    set_f32(bnd, ema_decay, 12);
    set_u32(bnd, n as u32, 13);
    set_u32(bnd, do_ema, 14);
    set_gpu_buf(bnd, clip_coef, 15);
}

/// Pack many tiny AdamW tensors into one compute encoder (segment pack).
fn adamw_ema_segment_pack(
    rt: &Arc<GpuRuntime>,
    items: &[(&Tensor, &Tensor, &AdamSlot, Option<&Tensor>)],
    clip_coef: &GpuBuffer,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    step_size: f32,
    bias2_sqrt_inv: f32,
    ema_decay: f32,
) -> Result<(), String> {
    let pipe = rt.pipeline("adamw_ema_f32")?;
    let width = pipe.threadExecutionWidth() as usize;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&pipe);
        for &(param, grad, slot, ema) in items {
            let n = param.numel();
            if n == 0 {
                continue;
            }
            let do_ema = if ema.is_some() { 1u32 } else { 0u32 };
            let ema_t = ema.unwrap_or(param);
            let tpt = width.min(n).max(1);
            let groups = (n + tpt - 1) / tpt;
            encode_adamw_ema(
                bnd,
                param,
                grad,
                slot,
                ema_t,
                clip_coef,
                lr,
                beta1,
                beta2,
                eps,
                wd,
                step_size,
                bias2_sqrt_inv,
                ema_decay,
                n,
                do_ema,
            );
            bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
        }
        Ok(())
    })
}

fn ema_one(rt: &Arc<GpuRuntime>, ema: &Tensor, live: &Tensor, decay: f32) -> Result<(), String> {
    let pipe = rt.pipeline("ema_update_f32")?;
    let n = ema.numel();
    dispatch_1d(rt, &pipe, n, |bnd| {
        set_tensor(bnd, ema, 0);
        set_tensor(bnd, live, 1);
        set_f32(bnd, decay, 2);
        set_u32(bnd, n as u32, 3);
    })
}

#[derive(Clone, Copy)]
enum ResearchRole {
    Matrix,
    Embed,
    Auxiliary,
}

struct ResearchItem<'a> {
    param: &'a Tensor,
    grad: &'a Tensor,
    state1: &'a Tensor,
    state2: &'a Tensor,
    ema: &'a Tensor,
    role: ResearchRole,
    decay: bool,
}

struct ProdigyItem<'a> {
    base: ResearchItem<'a>,
    s: &'a Tensor,
    p0: &'a Tensor,
}

fn research_items<'a>(w: &'a Weights, g: &'a Grads, s: &'a OptimState) -> Vec<ResearchItem<'a>> {
    let mut items = vec![
        ResearchItem {
            param: &w.tok_emb,
            grad: &g.tok_emb,
            state1: &s.tok_emb.exp_avg,
            state2: &s.tok_emb.exp_avg_sq,
            ema: &s.ema_tok_emb,
            role: ResearchRole::Embed,
            decay: true,
        },
        ResearchItem {
            param: &w.bigram_emb,
            grad: &g.bigram_emb,
            state1: &s.bigram_emb.exp_avg,
            state2: &s.bigram_emb.exp_avg_sq,
            ema: &s.ema_bigram_emb,
            role: ResearchRole::Embed,
            decay: true,
        },
        ResearchItem {
            param: &w.ve_emb,
            grad: &g.ve_emb,
            state1: &s.ve_emb.exp_avg,
            state2: &s.ve_emb.exp_avg_sq,
            ema: &s.ema_ve_emb,
            role: ResearchRole::Embed,
            decay: true,
        },
        ResearchItem {
            param: &w.bigram_proj,
            grad: &g.bigram_proj,
            state1: &s.bigram_proj.exp_avg,
            state2: &s.bigram_proj.exp_avg_sq,
            ema: &s.ema_bigram_proj,
            role: ResearchRole::Auxiliary,
            decay: true,
        },
        ResearchItem {
            param: &w.bigram_scale,
            grad: &g.bigram_scale,
            state1: &s.bigram_scale.exp_avg,
            state2: &s.bigram_scale.exp_avg_sq,
            ema: &s.ema_bigram_scale,
            role: ResearchRole::Auxiliary,
            decay: false,
        },
        ResearchItem {
            param: &w.smear_gate,
            grad: &g.smear_gate,
            state1: &s.smear_gate.exp_avg,
            state2: &s.smear_gate.exp_avg_sq,
            ema: &s.ema_smear_gate,
            role: ResearchRole::Auxiliary,
            decay: false,
        },
        ResearchItem {
            param: &w.ve_proj,
            grad: &g.ve_proj,
            state1: &s.ve_proj.exp_avg,
            state2: &s.ve_proj.exp_avg_sq,
            ema: &s.ema_ve_proj,
            role: ResearchRole::Auxiliary,
            decay: true,
        },
        ResearchItem {
            param: &w.ve_scale,
            grad: &g.ve_scale,
            state1: &s.ve_scale.exp_avg,
            state2: &s.ve_scale.exp_avg_sq,
            ema: &s.ema_ve_scale,
            role: ResearchRole::Auxiliary,
            decay: false,
        },
        ResearchItem {
            param: &w.skip_weights,
            grad: &g.skip_weights,
            state1: &s.skip_weights.exp_avg,
            state2: &s.skip_weights.exp_avg_sq,
            ema: &s.ema_skip_weights,
            role: ResearchRole::Auxiliary,
            decay: false,
        },
        ResearchItem {
            param: &w.qo_bank,
            grad: &g.qo_bank,
            state1: &s.mom_qo,
            state2: &s.var_qo,
            ema: &s.ema_qo,
            role: ResearchRole::Matrix,
            decay: true,
        },
        ResearchItem {
            param: &w.kv_bank,
            grad: &g.kv_bank,
            state1: &s.mom_kv,
            state2: &s.var_kv,
            ema: &s.ema_kv,
            role: ResearchRole::Matrix,
            decay: true,
        },
        ResearchItem {
            param: &w.mlp_up,
            grad: &g.mlp_up,
            state1: &s.mom_up,
            state2: &s.var_up,
            ema: &s.ema_up,
            role: ResearchRole::Matrix,
            decay: true,
        },
        ResearchItem {
            param: &w.mlp_down,
            grad: &g.mlp_down,
            state1: &s.mom_dn,
            state2: &s.var_dn,
            ema: &s.ema_dn,
            role: ResearchRole::Matrix,
            decay: true,
        },
    ];
    for i in 0..w.ve_layer_scales.len() {
        items.push(ResearchItem {
            param: &w.ve_layer_scales[i],
            grad: &g.ve_layer_scales[i],
            state1: &s.ve_layer_scales[i].exp_avg,
            state2: &s.ve_layer_scales[i].exp_avg_sq,
            ema: &s.ema_ve_layer_scales[i],
            role: ResearchRole::Auxiliary,
            decay: false,
        });
    }
    for i in 0..w.blocks.len() {
        let wb = &w.blocks[i];
        let gb = &g.blocks[i];
        let sb = &s.blocks[i];
        let eb = &s.ema_blocks[i];
        items.extend([
            ResearchItem {
                param: &wb.q_gain,
                grad: &gb.q_gain,
                state1: &sb.q_gain.exp_avg,
                state2: &sb.q_gain.exp_avg_sq,
                ema: &eb.q_gain,
                role: ResearchRole::Auxiliary,
                decay: false,
            },
            ResearchItem {
                param: &wb.vr_lambda,
                grad: &gb.vr_lambda,
                state1: &sb.vr_lambda.exp_avg,
                state2: &sb.vr_lambda.exp_avg_sq,
                ema: &eb.vr_lambda,
                role: ResearchRole::Auxiliary,
                decay: false,
            },
            ResearchItem {
                param: &wb.attn_scale,
                grad: &gb.attn_scale,
                state1: &sb.attn_scale.exp_avg,
                state2: &sb.attn_scale.exp_avg_sq,
                ema: &eb.attn_scale,
                role: ResearchRole::Auxiliary,
                decay: false,
            },
            ResearchItem {
                param: &wb.mlp_scale,
                grad: &gb.mlp_scale,
                state1: &sb.mlp_scale.exp_avg,
                state2: &sb.mlp_scale.exp_avg_sq,
                ema: &eb.mlp_scale,
                role: ResearchRole::Auxiliary,
                decay: false,
            },
            ResearchItem {
                param: &wb.resid_mix,
                grad: &gb.resid_mix,
                state1: &sb.resid_mix.exp_avg,
                state2: &sb.resid_mix.exp_avg_sq,
                ema: &eb.resid_mix,
                role: ResearchRole::Auxiliary,
                decay: false,
            },
        ]);
    }
    items
}

fn prodigy_items<'a>(w: &'a Weights, g: &'a Grads, s: &'a OptimState) -> Vec<ProdigyItem<'a>> {
    let base = research_items(w, g, s);
    let mut extras: Vec<(&Tensor, &Tensor)> = vec![
        (&s.tok_emb.aux, &s.tok_emb.origin),
        (&s.bigram_emb.aux, &s.bigram_emb.origin),
        (&s.ve_emb.aux, &s.ve_emb.origin),
        (&s.bigram_proj.aux, &s.bigram_proj.origin),
        (&s.bigram_scale.aux, &s.bigram_scale.origin),
        (&s.smear_gate.aux, &s.smear_gate.origin),
        (&s.ve_proj.aux, &s.ve_proj.origin),
        (&s.ve_scale.aux, &s.ve_scale.origin),
        (&s.skip_weights.aux, &s.skip_weights.origin),
        (&s.mag_v_qo, &s.prev_qo),
        (&s.mag_v_kv, &s.prev_kv),
        (&s.mag_v_up, &s.prev_up),
        (&s.mag_v_dn, &s.prev_dn),
    ];
    for slot in &s.ve_layer_scales {
        extras.push((&slot.aux, &slot.origin));
    }
    for block in &s.blocks {
        extras.extend([
            (&block.q_gain.aux, &block.q_gain.origin),
            (&block.vr_lambda.aux, &block.vr_lambda.origin),
            (&block.attn_scale.aux, &block.attn_scale.origin),
            (&block.mlp_scale.aux, &block.mlp_scale.origin),
            (&block.resid_mix.aux, &block.resid_mix.origin),
        ]);
    }
    assert_eq!(base.len(), extras.len());
    base.into_iter()
        .zip(extras)
        .map(|(base, (state_s, p0))| ProdigyItem {
            base,
            s: state_s,
            p0,
        })
        .collect()
}

fn research_algorithm(kind: OptimizerKind) -> Option<u32> {
    match kind {
        OptimizerKind::Adamw => Some(0),
        OptimizerKind::Lion => Some(1),
        OptimizerKind::CautiousLion => Some(2),
        OptimizerKind::CautiousAdamw => Some(3),
        OptimizerKind::SgdMomentum => Some(4),
        OptimizerKind::Sophia => Some(5),
        OptimizerKind::ScheduleFreeAdamw => Some(6),
        _ => None,
    }
}

fn research_optimizer_step(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    grads: &Grads,
    state: &OptimState,
    apply_ema: bool,
    lr_mul: f32,
) -> Result<(), String> {
    let algorithm = research_algorithm(state.kind)
        .ok_or_else(|| format!("{} has no generic native algorithm", state.kind))?;
    let hp = &state.hp;
    let step = state.step + 1;
    let beta1 = if algorithm == 4 {
        0.9
    } else if algorithm == 5 {
        0.965
    } else {
        hp.adam_beta1
    };
    let beta2 = if algorithm == 1 || algorithm == 2 {
        0.99
    } else {
        hp.adam_beta2
    };
    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);
    let hessian_update = u32::from(algorithm == 5 && step % hp.sophia_hessian_interval.max(1) == 0);
    let n_sf = step as f32;
    let weight_sum = n_sf * (n_sf + 1.0) * (2.0 * n_sf + 1.0) / 6.0;
    let ckp1 = n_sf * n_sf / weight_sum;
    let sf_warm = hp.schedule_free_warmup;
    let sf_mul = if sf_warm == 0 {
        1.0
    } else {
        (step as f32 / sf_warm as f32).min(1.0)
    };
    let effective_lr_mul = if algorithm == 6 { sf_mul } else { lr_mul };
    let update = rt.pipeline("research_optimizer_ema_f32")?;
    let count = rt.pipeline("cautious_mask_count_f32")?;
    let zero = rt.pipeline("zero_scalar_f32")?;
    let clip = state.clip.adamw_coef(state.clip_mode);
    for item in research_items(w, grads, state) {
        let base_lr = match item.role {
            ResearchRole::Matrix => hp.matrix_lr,
            ResearchRole::Embed => hp.tied_embed_lr,
            ResearchRole::Auxiliary => hp.scalar_lr,
        };
        let lr = base_lr * effective_lr_mul;
        let mut wd = if item.decay { hp.weight_decay } else { 0.0 };
        if algorithm == 1 || algorithm == 2 {
            wd *= 3.0;
        }
        let n = item.param.numel();
        let width = update.threadExecutionWidth() as usize;
        let tpt = width.min(n).max(1);
        let groups = n.div_ceil(tpt);
        rt.with_binder(|bnd| {
            if algorithm == 2 || algorithm == 3 {
                bnd.set_pipeline(&zero);
                set_gpu_buf(bnd, &state.clip.total_sq, 0);
                set_u32(bnd, 1, 1);
                bnd.dispatch(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
                bnd.barrier();
                bnd.set_pipeline(&count);
                set_tensor(bnd, item.grad, 0);
                set_tensor(bnd, item.state1, 1);
                set_gpu_buf(bnd, &state.clip.total_sq, 2);
                set_f32(bnd, beta1, 3);
                set_u32(bnd, algorithm, 4);
                set_u32(bnd, n as u32, 5);
                bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
                bnd.barrier();
            }
            bnd.set_pipeline(&update);
            set_tensor(bnd, item.param, 0);
            set_tensor(bnd, item.grad, 1);
            set_tensor(bnd, item.state1, 2);
            set_tensor(bnd, item.state2, 3);
            set_tensor(bnd, item.ema, 4);
            set_gpu_buf(bnd, clip, 5);
            set_u32(bnd, algorithm, 6);
            set_f32(bnd, lr, 7);
            set_f32(bnd, beta1, 8);
            set_f32(bnd, beta2, 9);
            set_f32(bnd, hp.adam_eps, 10);
            set_f32(bnd, wd, 11);
            set_f32(bnd, bc1, 12);
            set_f32(bnd, bc2, 13);
            set_f32(bnd, hp.ema_decay, 14);
            set_u32(bnd, n as u32, 15);
            set_u32(bnd, u32::from(apply_ema), 16);
            set_gpu_buf(bnd, &state.clip.total_sq, 17);
            set_f32(bnd, hp.sophia_rho, 18);
            set_u32(bnd, hessian_update, 19);
            set_f32(bnd, ckp1, 20);
            bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
            Ok(())
        })?;
    }
    Ok(())
}

fn prodigy_optimizer_step(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    grads: &Grads,
    state: &mut OptimState,
    apply_ema: bool,
) -> Result<(), String> {
    let hp = &state.hp;
    let beta1 = hp.adam_beta1;
    let beta2 = hp.adam_beta2;
    let beta3 = beta2.sqrt();
    let d0 = 1e-6f32;
    let d = state.prodigy_d;
    let k = state.step + 1;
    let bc = (1.0 - beta2.powi(k as i32)).sqrt() / (1.0 - beta1.powi(k as i32));
    let dlr = d * bc;
    let zero = rt.pipeline("zero_scalar_f32")?;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&zero);
        set_gpu_buf(bnd, &state.clip.total_sq, 0);
        set_u32(bnd, 1, 1);
        bnd.dispatch(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
        bnd.barrier();
        set_gpu_buf(bnd, &state.clip.norm, 0);
        set_u32(bnd, 1, 1);
        bnd.dispatch(mtl_size(1, 1, 1), mtl_size(1, 1, 1));
        Ok(())
    })?;
    let accumulate = rt.pipeline("prodigy_accumulate_f32")?;
    {
        let items = prodigy_items(w, grads, state);
        for item in items {
            let n = item.base.param.numel();
            let width = accumulate.threadExecutionWidth() as usize;
            let tpt = width.min(n).max(1);
            let groups = n.div_ceil(tpt);
            let wd = if item.base.decay {
                hp.weight_decay
            } else {
                0.0
            };
            rt.with_binder(|bnd| {
                bnd.set_pipeline(&accumulate);
                set_tensor(bnd, item.base.param, 0);
                set_tensor(bnd, item.base.grad, 1);
                set_tensor(bnd, item.base.state1, 2);
                set_tensor(bnd, item.base.state2, 3);
                set_tensor(bnd, item.s, 4);
                set_tensor(bnd, item.p0, 5);
                set_gpu_buf(bnd, &state.clip.total_sq, 6);
                set_gpu_buf(bnd, &state.clip.norm, 7);
                set_gpu_buf(bnd, state.clip.adamw_coef(state.clip_mode), 8);
                set_f32(bnd, d, 9);
                set_f32(bnd, d0, 10);
                set_f32(bnd, dlr, 11);
                set_f32(bnd, beta1, 12);
                set_f32(bnd, beta2, 13);
                set_f32(bnd, beta3, 14);
                set_f32(bnd, wd, 15);
                set_u32(bnd, n as u32, 16);
                bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
                Ok(())
            })?;
        }
    }
    // Prodigy requires one global scalar decision per step; all per-parameter
    // contributions stay device-resident until this single synchronization.
    rt.synchronize()?;
    let numer_step = state.clip.total_sq.contents_f32()[0];
    let denom = state.clip.norm.contents_f32()[0];
    let numer = state.prodigy_d_numerator * beta3 + numer_step;
    let mut new_d = d;
    if denom > 0.0 {
        new_d = new_d.max(numer / ((1.0 - beta3) * denom));
    }
    state.prodigy_d = new_d;
    state.prodigy_d_max = state.prodigy_d_max.max(new_d);
    state.prodigy_d_numerator = numer;

    let finalize = rt.pipeline("prodigy_finalize_ema_f32")?;
    for item in prodigy_items(w, grads, state) {
        let n = item.base.param.numel();
        let width = finalize.threadExecutionWidth() as usize;
        let tpt = width.min(n).max(1);
        let groups = n.div_ceil(tpt);
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&finalize);
            set_tensor(bnd, item.base.param, 0);
            set_tensor(bnd, item.base.state1, 1);
            set_tensor(bnd, item.base.state2, 2);
            set_tensor(bnd, item.base.ema, 3);
            set_gpu_buf(bnd, state.clip.adamw_coef(state.clip_mode), 4);
            set_f32(bnd, new_d, 5);
            set_f32(bnd, dlr, 6);
            set_f32(bnd, hp.adam_eps, 7);
            set_f32(bnd, hp.ema_decay, 8);
            set_u32(bnd, u32::from(apply_ema), 9);
            set_u32(bnd, n as u32, 10);
            bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
            Ok(())
        })?;
    }
    Ok(())
}

fn muon_bank(
    rt: &Arc<GpuRuntime>,
    param: &Tensor,
    grad: &Tensor,
    momentum: &Tensor,
    aux_state: &Tensor,
    prev_state: &Tensor,
    extra_state: &Tensor,
    scratch: &GpuBuffer,
    clip_coef: &GpuBuffer,
    n: u32,
    rows: u32,
    cols: u32,
    lr: f32,
    mom: f32,
    wd: f32,
    scale: f32,
    ema: Option<&Tensor>,
    ema_decay: f32,
    orthogonalizer: MuonOrthogonalizer,
    post_kind: u32,
    post_beta2: f32,
    pre_kind: u32,
    pre_beta: f32,
    pre_alpha: f32,
    first_step: bool,
    optim_step: usize,
    direction_scale: f32,
) -> Result<(), String> {
    if rt.has_tensorops() && (rows.min(cols) >= 256 || post_kind != 0 || pre_kind != 0) {
        return muon_bank_tensorops(
            rt,
            param,
            grad,
            momentum,
            aux_state,
            prev_state,
            extra_state,
            scratch,
            clip_coef,
            n,
            rows,
            cols,
            lr,
            mom,
            wd,
            scale,
            ema,
            ema_decay,
            orthogonalizer,
            post_kind,
            post_beta2,
            pre_kind,
            pre_beta,
            pre_alpha,
            first_step,
            optim_step,
            direction_scale,
        );
    }
    let tpg = 256usize;
    if let Some(ema_t) = ema {
        let pipe = rt.pipeline("muon_bank_ns5_ema_f32")?;
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            set_tensor(bnd, param, 0);
            set_tensor(bnd, grad, 1);
            set_tensor(bnd, momentum, 2);
            set_gpu_buf(bnd, scratch, 3);
            set_tensor(bnd, ema_t, 4);
            set_u32(bnd, n, 5);
            set_u32(bnd, rows, 6);
            set_u32(bnd, cols, 7);
            set_f32(bnd, lr, 8);
            set_f32(bnd, mom, 9);
            set_f32(bnd, wd, 10);
            set_f32(bnd, scale, 11);
            set_f32(bnd, NS_EPS, 12);
            set_u32(bnd, orthogonalizer.coefficients().len() as u32, 13);
            set_f32(bnd, NS_A, 14);
            set_f32(bnd, NS_B, 15);
            set_f32(bnd, NS_C, 16);
            set_f32(bnd, ema_decay, 17);
            set_gpu_buf(bnd, clip_coef, 18);
            set_u32(
                bnd,
                if crate::ab_flags::muon_simdgroup() {
                    1
                } else {
                    0
                },
                19,
            );
            set_u32(bnd, orthogonalizer.kernel_kind(), 20);
            bnd.dispatch(mtl_size(n as usize, 1, 1), mtl_size(tpg, 1, 1));
            Ok(())
        })
    } else {
        let pipe = rt.pipeline("muon_bank_ns5_f32")?;
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&pipe);
            set_tensor(bnd, param, 0);
            set_tensor(bnd, grad, 1);
            set_tensor(bnd, momentum, 2);
            set_gpu_buf(bnd, scratch, 3);
            set_u32(bnd, n, 4);
            set_u32(bnd, rows, 5);
            set_u32(bnd, cols, 6);
            set_f32(bnd, lr, 7);
            set_f32(bnd, mom, 8);
            set_f32(bnd, wd, 9);
            set_f32(bnd, scale, 10);
            set_f32(bnd, NS_EPS, 11);
            set_u32(bnd, orthogonalizer.coefficients().len() as u32, 12);
            set_f32(bnd, NS_A, 13);
            set_f32(bnd, NS_B, 14);
            set_f32(bnd, NS_C, 15);
            set_gpu_buf(bnd, clip_coef, 16);
            set_u32(
                bnd,
                if crate::ab_flags::muon_simdgroup() {
                    1
                } else {
                    0
                },
                17,
            );
            set_u32(bnd, orthogonalizer.kernel_kind(), 18);
            bnd.dispatch(mtl_size(n as usize, 1, 1), mtl_size(tpg, 1, 1));
            Ok(())
        })
    }
}

fn zero_tensor_f32(rt: &Arc<GpuRuntime>, tensor: &Tensor) -> Result<(), String> {
    let pipeline = rt.pipeline("zero_f32")?;
    let n = tensor.numel();
    dispatch_1d(rt, &pipeline, n, |bnd| {
        set_tensor(bnd, tensor, 0);
        set_u32(bnd, n as u32, 1);
    })
}

fn batched_tensorops_gemm(
    rt: &Arc<GpuRuntime>,
    kernel: &str,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    batch: usize,
    m: usize,
    out_n: usize,
    k: usize,
) -> Result<(), String> {
    // mode::multiply requires a cleared C (MPP). Separate zero+gemm binders;
    // Audit 9B fused them in one with_binder — no step win (REJECT).
    zero_tensor_f32(rt, c)?;
    let pipeline = rt.pipeline(kernel)?;
    let tiles_n = out_n.div_ceil(32);
    let tiles_m = m.div_ceil(32);
    let groups = batch * tiles_n * tiles_m;
    let tpt = pipeline.threadExecutionWidth() as usize;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&pipeline);
        set_tensor(bnd, a, 0);
        set_tensor(bnd, b, 1);
        set_tensor(bnd, c, 2);
        set_u32(bnd, m as u32, 3);
        set_u32(bnd, out_n as u32, 4);
        set_u32(bnd, k as u32, 5);
        set_u32(bnd, tiles_n as u32, 6);
        set_u32(bnd, tiles_m as u32, 7);
        set_u32(bnd, batch as u32, 8);
        bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn muon_bank_tensorops(
    rt: &Arc<GpuRuntime>,
    param: &Tensor,
    grad: &Tensor,
    momentum: &Tensor,
    aux_state: &Tensor,
    prev_state: &Tensor,
    extra_state: &Tensor,
    scratch: &GpuBuffer,
    clip_coef: &GpuBuffer,
    n: u32,
    rows: u32,
    cols: u32,
    lr: f32,
    mom: f32,
    wd: f32,
    scale: f32,
    ema: Option<&Tensor>,
    ema_decay: f32,
    orthogonalizer: MuonOrthogonalizer,
    post_kind: u32,
    post_beta2: f32,
    pre_kind: u32,
    pre_beta: f32,
    pre_alpha: f32,
    first_step: bool,
    optim_step: usize,
    direction_scale: f32,
) -> Result<(), String> {
    let (batch, rows, cols) = (n as usize, rows as usize, cols as usize);
    let mat = rows * cols;
    let p = rows.min(cols);
    let mat_bank = batch * mat;
    let gram_bank = batch * p * p;
    let needed = 2 * mat_bank + 2 * gram_bank;
    if needed * 4 > scratch.nbytes() {
        return Err(format!(
            "Muon TensorOps scratch needs {} MiB but has {} MiB",
            needed * 4 / (1024 * 1024),
            scratch.nbytes() / (1024 * 1024)
        ));
    }
    // Nested detail under METAL_NATIVE_OPTIM_PROFILE — one report per bank call.
    let mut mprof = crate::model_bwd::BwdProf::new_labeled(
        rt,
        crate::ab_flags::optim_profile(),
        "muon_bank_detail",
    )?;
    let flat = Tensor::from_buffer(rt, scratch.clone(), &[scratch.nbytes() / 4], DType::F32, 0)?;
    let x = flat.view(&[batch, rows, cols], 0);
    let y = flat.view(&[batch, rows, cols], mat_bank);
    let gram = flat.view(&[batch, p, p], 2 * mat_bank);
    let gram2 = flat.view(&[batch, p, p], 2 * mat_bank + gram_bank);

    if pre_kind == 2 {
        let prepare = rt.pipeline("muown_prepare_f32")?;
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&prepare);
            set_tensor(bnd, param, 0);
            set_tensor(bnd, grad, 1);
            set_tensor(bnd, momentum, 2);
            set_tensor(bnd, prev_state, 3);
            set_tensor(bnd, &x, 4);
            set_u32(bnd, batch as u32, 5);
            set_u32(bnd, rows as u32, 6);
            set_u32(bnd, cols as u32, 7);
            set_f32(bnd, mom, 8);
            set_f32(bnd, NS_EPS, 9);
            bnd.dispatch(mtl_size(batch, 1, 1), mtl_size(256, 1, 1));
            Ok(())
        })?;
    } else {
        let prepare = rt.pipeline("muon_tensorops_prepare_f32")?;
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&prepare);
            set_tensor(bnd, grad, 0);
            set_tensor(bnd, momentum, 1);
            set_tensor(bnd, &x, 2);
            set_u32(bnd, batch as u32, 3);
            set_u32(bnd, mat as u32, 4);
            set_f32(bnd, mom, 5);
            set_f32(bnd, NS_EPS, 6);
            set_f32(bnd, orthogonalizer.norm_scale(), 7);
            set_tensor(bnd, aux_state, 8);
            set_tensor(bnd, prev_state, 9);
            set_u32(bnd, pre_kind, 10);
            set_f32(bnd, pre_beta, 11);
            set_f32(bnd, pre_alpha, 12);
            set_u32(bnd, u32::from(first_step), 13);
            bnd.dispatch(mtl_size(batch, 1, 1), mtl_size(256, 1, 1));
            Ok(())
        })?;
    }
    mprof.lap(rt, "muon_prepare")?;

    let poly = rt.pipeline("muon_tensorops_poly_combine_f32")?;
    let x_combine = rt.pipeline("muon_tensorops_x_combine_f32")?;
    for (a, b, c) in orthogonalizer.coefficients() {
        if rows <= cols {
            batched_tensorops_gemm(
                rt,
                "matmul2d_tensorops_batched_nt_f32",
                &x,
                &x,
                &gram,
                batch,
                rows,
                rows,
                cols,
            )?;
        } else {
            batched_tensorops_gemm(
                rt,
                "matmul2d_tensorops_batched_tn_f32",
                &x,
                &x,
                &gram,
                batch,
                cols,
                cols,
                rows,
            )?;
        }
        mprof.lap(rt, "muon_xxt")?;
        batched_tensorops_gemm(
            rt,
            "matmul2d_tensorops_batched_f32",
            &gram,
            &gram,
            &gram2,
            batch,
            p,
            p,
            p,
        )?;
        mprof.lap(rt, "muon_a2")?;
        dispatch_1d(rt, &poly, gram_bank, |bnd| {
            set_tensor(bnd, &gram, 0);
            set_tensor(bnd, &gram2, 1);
            set_f32(bnd, b, 2);
            set_f32(bnd, c, 3);
            set_u32(bnd, gram_bank as u32, 4);
        })?;
        if rows <= cols {
            batched_tensorops_gemm(
                rt,
                "matmul2d_tensorops_batched_f32",
                &gram2,
                &x,
                &y,
                batch,
                rows,
                cols,
                rows,
            )?;
        } else {
            batched_tensorops_gemm(
                rt,
                "matmul2d_tensorops_batched_f32",
                &x,
                &gram2,
                &y,
                batch,
                rows,
                cols,
                cols,
            )?;
        }
        dispatch_1d(rt, &x_combine, mat_bank, |bnd| {
            set_tensor(bnd, &x, 0);
            set_tensor(bnd, &y, 1);
            set_f32(bnd, a, 2);
            set_u32(bnd, mat_bank as u32, 3);
        })?;
        mprof.lap(rt, "muon_bx_poly")?;
    }

    if post_kind == 1 {
        let post = rt.pipeline("normuon_row_post_f32")?;
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&post);
            set_tensor(bnd, &x, 0);
            set_tensor(bnd, aux_state, 1);
            set_u32(bnd, batch as u32, 2);
            set_u32(bnd, rows as u32, 3);
            set_u32(bnd, cols as u32, 4);
            set_f32(bnd, post_beta2, 5);
            set_f32(bnd, 1e-10, 6);
            bnd.dispatch(mtl_size(batch, 1, 1), mtl_size(256, 1, 1));
            Ok(())
        })?;
    }

    if post_kind == 2 {
        let finalize = rt.pipeline("muown_finalize_f32")?;
        let ema_tensor = ema.unwrap_or(param);
        let step = (optim_step + 1) as i32;
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&finalize);
            set_tensor(bnd, param, 0);
            set_tensor(bnd, grad, 1);
            set_tensor(bnd, &x, 2);
            set_tensor(bnd, prev_state, 3);
            set_tensor(bnd, aux_state, 4);
            set_tensor(bnd, extra_state, 5);
            set_tensor(bnd, ema_tensor, 6);
            set_gpu_buf(bnd, clip_coef, 7);
            set_u32(bnd, batch as u32, 8);
            set_u32(bnd, rows as u32, 9);
            set_u32(bnd, cols as u32, 10);
            set_f32(bnd, lr, 11);
            set_f32(bnd, 0.9, 12);
            set_f32(bnd, 0.95, 13);
            set_f32(bnd, NS_EPS, 14);
            set_f32(bnd, direction_scale, 15);
            set_f32(bnd, wd, 16);
            set_f32(bnd, 1.0 - 0.9f32.powi(step), 17);
            set_f32(bnd, 1.0 - 0.95f32.powi(step), 18);
            set_f32(bnd, ema_decay, 19);
            set_u32(bnd, u32::from(ema.is_some()), 20);
            bnd.dispatch(mtl_size(batch, 1, 1), mtl_size(256, 1, 1));
            Ok(())
        })?;
        mprof.lap(rt, "muon_finalize")?;
        mprof.report();
        return Ok(());
    }

    let finalize = rt.pipeline("muon_tensorops_finalize_f32")?;
    let ema_tensor = ema.unwrap_or(param);
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&finalize);
        set_tensor(bnd, param, 0);
        set_tensor(bnd, &x, 1);
        set_tensor(bnd, ema_tensor, 2);
        set_gpu_buf(bnd, clip_coef, 3);
        set_f32(bnd, lr, 4);
        set_f32(bnd, wd, 5);
        set_f32(bnd, scale, 6);
        set_f32(bnd, ema_decay, 7);
        set_u32(bnd, mat_bank as u32, 8);
        set_u32(bnd, u32::from(ema.is_some()), 9);
        let width = finalize.threadExecutionWidth() as usize;
        let groups = mat_bank.div_ceil(width);
        bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(width, 1, 1));
        Ok(())
    })?;
    mprof.lap(rt, "muon_finalize")?;
    mprof.report();
    Ok(())
}

/// Refresh tied `tok_emb_t` after AdamW updates `tok_emb` (GPU transpose).
pub fn refresh_tok_emb_t(rt: &Arc<GpuRuntime>, w: &mut Weights) -> Result<(), String> {
    let rows = w.tok_emb.shape()[0];
    let cols = w.tok_emb.shape()[1];
    assert_eq!(w.tok_emb_t.shape(), &[cols, rows]);
    let p = rt.pipeline("transpose2d_f32")?;
    dispatch_1d(rt, &p, rows * cols, |bnd| {
        set_tensor(bnd, &w.tok_emb, 0);
        set_tensor(bnd, &w.tok_emb_t, 1);
        set_u32(bnd, rows as u32, 2);
        set_u32(bnd, cols as u32, 3);
    })?;
    Ok(())
}

/// One optimizer step after grads are filled (and clipped): embed AdamW → scalar
/// AdamW → Muon banks → EMA on Muon params (embed/scalar EMA fused into AdamW).
pub fn optim_step(
    rt: &Arc<GpuRuntime>,
    w: &mut Weights,
    grads: &Grads,
    state: &mut OptimState,
    apply_ema: bool,
    lr_mul: f32,
) -> Result<(), String> {
    if state.kind == OptimizerKind::Prodigy {
        prodigy_optimizer_step(rt, w, grads, state, apply_ema)?;
        state.step += 1;
        return Ok(());
    }
    if research_algorithm(state.kind).is_some() {
        research_optimizer_step(rt, w, grads, state, apply_ema, lr_mul)?;
        state.step += 1;
        return Ok(());
    }
    let (orthogonalizer, post_kind, pre_kind) = match state.kind {
        OptimizerKind::MuonNs5Adamw => (MuonOrthogonalizer::NewtonSchulz(5), 0, 0),
        OptimizerKind::MuonNs3Adamw => (MuonOrthogonalizer::NewtonSchulz(3), 0, 0),
        OptimizerKind::MuonPolarAdamw => (MuonOrthogonalizer::PolarExpress, 0, 0),
        OptimizerKind::NorMuonAdamw => (MuonOrthogonalizer::NewtonSchulz(5), 1, 0),
        OptimizerKind::MonaAdamw => (MuonOrthogonalizer::NewtonSchulz(5), 0, 1),
        OptimizerKind::MuownAdamw => (MuonOrthogonalizer::NewtonSchulz(5), 2, 2),
        other => {
            return Err(format!(
                "optimizer {other} is registered for the research oracle but its native Metal step is not parity-qualified"
            ))
        }
    };
    let hp = &state.hp;
    let pre_alpha = if hp.mona_alpha == 0.0 {
        -1.0 / (2.0 * (1.0 - hp.mona_beta_a))
    } else {
        hp.mona_alpha
    };
    let t = (state.step + 1) as i32;
    let beta1 = hp.adam_beta1;
    let beta2 = hp.adam_beta2;
    let bc1 = 1.0 - beta1.powi(t);
    let bc2 = 1.0 - beta2.powi(t);
    let bias2_sqrt_inv = 1.0 / bc2.sqrt();
    let decay = if apply_ema { hp.ema_decay } else { 0.0 };
    // Soft: AdamW uses Match coef; Muon uses √c (see ClipMode::Soft).
    let adamw_coef = state.clip.adamw_coef(state.clip_mode);
    let muon_coef = state.clip.muon_coef(state.clip_mode);

    // Audit 8: METAL_NATIVE_OPTIM_PROFILE=1 — same synced sectioning that
    // located the FA bwd bottleneck. Optimizer is ~23% of step post-Audit-7 and
    // has never been broken down (M13 measured it only in aggregate).
    let mut oprof = crate::model_bwd::BwdProf::new_labeled(
        rt,
        crate::ab_flags::optim_profile(),
        "optim_profile",
    )?;

    // --- embed AdamW (+ EMA), packed ---
    let lr_e = hp.tied_embed_lr * lr_mul;
    let step_e = lr_e / bc1;
    adamw_ema_segment_pack(
        rt,
        &[
            (
                &w.tok_emb,
                &grads.tok_emb,
                &state.tok_emb,
                if apply_ema {
                    Some(&state.ema_tok_emb)
                } else {
                    None
                },
            ),
            (
                &w.bigram_emb,
                &grads.bigram_emb,
                &state.bigram_emb,
                if apply_ema {
                    Some(&state.ema_bigram_emb)
                } else {
                    None
                },
            ),
            (
                &w.ve_emb,
                &grads.ve_emb,
                &state.ve_emb,
                if apply_ema {
                    Some(&state.ema_ve_emb)
                } else {
                    None
                },
            ),
        ],
        adamw_coef,
        lr_e,
        beta1,
        beta2,
        hp.adam_eps,
        hp.weight_decay,
        step_e,
        bias2_sqrt_inv,
        decay,
    )?;
    refresh_tok_emb_t(rt, w)?;

    oprof.lap(rt, "embed_adamw")?;

    // --- scalar AdamW (+ EMA), packed into one encoder ---
    let lr_s = hp.scalar_lr * lr_mul;
    let step_s = lr_s / bc1;
    let mut scalar_items: Vec<(&Tensor, &Tensor, &AdamSlot, Option<&Tensor>)> = Vec::new();
    scalar_items.push((
        &w.bigram_proj,
        &grads.bigram_proj,
        &state.bigram_proj,
        if apply_ema {
            Some(&state.ema_bigram_proj)
        } else {
            None
        },
    ));
    scalar_items.push((
        &w.bigram_scale,
        &grads.bigram_scale,
        &state.bigram_scale,
        if apply_ema {
            Some(&state.ema_bigram_scale)
        } else {
            None
        },
    ));
    scalar_items.push((
        &w.smear_gate,
        &grads.smear_gate,
        &state.smear_gate,
        if apply_ema {
            Some(&state.ema_smear_gate)
        } else {
            None
        },
    ));
    scalar_items.push((
        &w.ve_proj,
        &grads.ve_proj,
        &state.ve_proj,
        if apply_ema {
            Some(&state.ema_ve_proj)
        } else {
            None
        },
    ));
    scalar_items.push((
        &w.ve_scale,
        &grads.ve_scale,
        &state.ve_scale,
        if apply_ema {
            Some(&state.ema_ve_scale)
        } else {
            None
        },
    ));
    scalar_items.push((
        &w.skip_weights,
        &grads.skip_weights,
        &state.skip_weights,
        if apply_ema {
            Some(&state.ema_skip_weights)
        } else {
            None
        },
    ));
    for i in 0..w.ve_layer_scales.len() {
        scalar_items.push((
            &w.ve_layer_scales[i],
            &grads.ve_layer_scales[i],
            &state.ve_layer_scales[i],
            if apply_ema {
                Some(&state.ema_ve_layer_scales[i])
            } else {
                None
            },
        ));
    }
    for i in 0..w.blocks.len() {
        if w.cfg.optimizes_q_gain(i) {
            scalar_items.push((
                &w.blocks[i].q_gain,
                &grads.blocks[i].q_gain,
                &state.blocks[i].q_gain,
                if apply_ema {
                    Some(&state.ema_blocks[i].q_gain)
                } else {
                    None
                },
            ));
        }
        if w.cfg.optimizes_vr_lambda(i) {
            scalar_items.push((
                &w.blocks[i].vr_lambda,
                &grads.blocks[i].vr_lambda,
                &state.blocks[i].vr_lambda,
                if apply_ema {
                    Some(&state.ema_blocks[i].vr_lambda)
                } else {
                    None
                },
            ));
        }
        scalar_items.push((
            &w.blocks[i].attn_scale,
            &grads.blocks[i].attn_scale,
            &state.blocks[i].attn_scale,
            if apply_ema {
                Some(&state.ema_blocks[i].attn_scale)
            } else {
                None
            },
        ));
        scalar_items.push((
            &w.blocks[i].mlp_scale,
            &grads.blocks[i].mlp_scale,
            &state.blocks[i].mlp_scale,
            if apply_ema {
                Some(&state.ema_blocks[i].mlp_scale)
            } else {
                None
            },
        ));
        scalar_items.push((
            &w.blocks[i].resid_mix,
            &grads.blocks[i].resid_mix,
            &state.blocks[i].resid_mix,
            if apply_ema {
                Some(&state.ema_blocks[i].resid_mix)
            } else {
                None
            },
        ));
    }
    adamw_ema_segment_pack(
        rt,
        &scalar_items,
        adamw_coef,
        lr_s,
        beta1,
        beta2,
        hp.adam_eps,
        hp.weight_decay,
        step_s,
        bias2_sqrt_inv,
        decay,
    )?;

    oprof.lap(rt, "scalar_adamw")?;

    // --- Muon (4 bank dispatches; EMA fused when enabled) ---
    let mom = hp.muon_momentum(state.step);
    let lr_m = hp.matrix_lr * lr_mul;
    let c = w.cfg.model_dim as u32;
    let kv = w.cfg.kv_dim() as u32;
    let mlp = w.cfg.mlp_dim as u32;
    let n_l = w.cfg.num_layers as u32;
    let ema_d = if apply_ema { hp.ema_decay } else { 0.0 };

    use crate::weights::MixerKind;

    let n_attn = w.cfg.mixer_count(MixerKind::Attention) as u32;

    if n_attn > 0 {
        muon_bank(
            rt,
            &w.qo_bank,
            &grads.qo_bank,
            &state.mom_qo,
            &state.var_qo,
            &state.prev_qo,
            &state.mag_v_qo,
            &state.muon_scratch,
            muon_coef,
            2 * n_attn,
            c,
            c,
            lr_m,
            mom,
            hp.weight_decay,
            bank_scale(c as usize, c as usize),
            if apply_ema { Some(&state.ema_qo) } else { None },
            ema_d,
            orthogonalizer,
            post_kind,
            hp.adam_beta2,
            pre_kind,
            hp.mona_beta_a,
            pre_alpha,
            state.step == 0,
            state.step,
            hp.muown_direction_scale,
        )?;
        oprof.lap(rt, "muon_qo")?;

        muon_bank(
            rt,
            &w.kv_bank,
            &grads.kv_bank,
            &state.mom_kv,
            &state.var_kv,
            &state.prev_kv,
            &state.mag_v_kv,
            &state.muon_scratch,
            muon_coef,
            2 * n_attn,
            c,
            kv,
            lr_m,
            mom,
            hp.weight_decay,
            bank_scale(c as usize, kv as usize),
            if apply_ema { Some(&state.ema_kv) } else { None },
            ema_d,
            orthogonalizer,
            post_kind,
            hp.adam_beta2,
            pre_kind,
            hp.mona_beta_a,
            pre_alpha,
            state.step == 0,
            state.step,
            hp.muown_direction_scale,
        )?;
        oprof.lap(rt, "muon_kv")?;
    }

    let n_mamba = w.cfg.mixer_count(MixerKind::Mamba2) as u32;
    let n_mingru = w.cfg.mixer_count(MixerKind::MinGRU) as u32;

    if n_mamba > 0 {
        let d_inner = w.cfg.mamba_d_inner() as u32;
        let in_out = w.cfg.mamba_in_proj_out() as u32;
        if let (
            Some(ref pw),
            Some(ref pg),
            Some(ref pm),
            Some(ref pv),
            Some(ref pp),
            Some(ref ema),
        ) = (
            w.mamba_in_proj.as_ref(),
            grads.mamba_in_proj.as_ref(),
            state.mom_mamba_in_proj.as_ref(),
            state.var_mamba_in_proj.as_ref(),
            state.prev_mamba_in_proj.as_ref(),
            state.ema_mamba_in_proj.as_ref(),
        ) {
            muon_bank(
                rt,
                pw,
                pg,
                pm,
                pv,
                pp,
                &state.mag_v_mamba_in_proj.as_ref().unwrap(),
                &state.muon_scratch,
                muon_coef,
                n_mamba,
                c,
                in_out,
                lr_m,
                mom,
                hp.weight_decay,
                bank_scale(c as usize, in_out as usize),
                if apply_ema { Some(ema) } else { None },
                ema_d,
                orthogonalizer,
                post_kind,
                hp.adam_beta2,
                pre_kind,
                hp.mona_beta_a,
                pre_alpha,
                state.step == 0,
                state.step,
                hp.muown_direction_scale,
            )?;
        }
        if let (
            Some(ref pw),
            Some(ref pg),
            Some(ref pm),
            Some(ref pv),
            Some(ref pp),
            Some(ref ema),
        ) = (
            w.mamba_out_proj.as_ref(),
            grads.mamba_out_proj.as_ref(),
            state.mom_mamba_out_proj.as_ref(),
            state.var_mamba_out_proj.as_ref(),
            state.prev_mamba_out_proj.as_ref(),
            state.ema_mamba_out_proj.as_ref(),
        ) {
            muon_bank(
                rt,
                pw,
                pg,
                pm,
                pv,
                pp,
                &state.mag_v_mamba_out_proj.as_ref().unwrap(),
                &state.muon_scratch,
                muon_coef,
                n_mamba,
                d_inner,
                c,
                lr_m,
                mom,
                hp.weight_decay,
                bank_scale(d_inner as usize, c as usize),
                if apply_ema { Some(ema) } else { None },
                ema_d,
                orthogonalizer,
                post_kind,
                hp.adam_beta2,
                pre_kind,
                hp.mona_beta_a,
                pre_alpha,
                state.step == 0,
                state.step,
                hp.muown_direction_scale,
            )?;
        }
        let mut mamba_scalar: Vec<(&Tensor, &Tensor, &AdamSlot, Option<&Tensor>)> = Vec::new();
        if let (Some(t), Some(g), Some(s), Some(e)) = (
            w.mamba_conv1d_weight.as_ref(),
            grads.mamba_conv1d_weight.as_ref(),
            state.mamba_conv1d_weight.as_ref(),
            state.ema_mamba_conv1d_weight.as_ref(),
        ) {
            mamba_scalar.push((t, g, s, if apply_ema { Some(e) } else { None }));
        }
        if let (Some(t), Some(g), Some(s), Some(e)) = (
            w.mamba_conv1d_bias.as_ref(),
            grads.mamba_conv1d_bias.as_ref(),
            state.mamba_conv1d_bias.as_ref(),
            state.ema_mamba_conv1d_bias.as_ref(),
        ) {
            mamba_scalar.push((t, g, s, if apply_ema { Some(e) } else { None }));
        }
        if let (Some(t), Some(g), Some(s), Some(e)) = (
            w.mamba_a_log.as_ref(),
            grads.mamba_a_log.as_ref(),
            state.mamba_a_log.as_ref(),
            state.ema_mamba_a_log.as_ref(),
        ) {
            mamba_scalar.push((t, g, s, if apply_ema { Some(e) } else { None }));
        }
        if let (Some(t), Some(g), Some(s), Some(e)) = (
            w.mamba_d.as_ref(),
            grads.mamba_d.as_ref(),
            state.mamba_d.as_ref(),
            state.ema_mamba_d.as_ref(),
        ) {
            mamba_scalar.push((t, g, s, if apply_ema { Some(e) } else { None }));
        }
        if let (Some(t), Some(g), Some(s), Some(e)) = (
            w.mamba_dt_bias.as_ref(),
            grads.mamba_dt_bias.as_ref(),
            state.mamba_dt_bias.as_ref(),
            state.ema_mamba_dt_bias.as_ref(),
        ) {
            mamba_scalar.push((t, g, s, if apply_ema { Some(e) } else { None }));
        }
        if let (Some(t), Some(g), Some(s), Some(e)) = (
            w.mamba_norm.as_ref(),
            grads.mamba_norm.as_ref(),
            state.mamba_norm.as_ref(),
            state.ema_mamba_norm.as_ref(),
        ) {
            mamba_scalar.push((t, g, s, if apply_ema { Some(e) } else { None }));
        }
        if !mamba_scalar.is_empty() {
            adamw_ema_segment_pack(
                rt,
                &mamba_scalar,
                adamw_coef,
                lr_s,
                beta1,
                beta2,
                hp.adam_eps,
                hp.weight_decay,
                step_s,
                bias2_sqrt_inv,
                decay,
            )?;
        }
    }

    if n_mingru > 0 {
        let hid = w.cfg.mingru_hidden() as u32;
        let kv = w.cfg.kv_dim() as u32;
        let mut mingru_banks: Vec<(
            &Option<Tensor>,
            &Option<Tensor>,
            &Option<Tensor>,
            &Option<Tensor>,
            &Option<Tensor>,
            &Option<Tensor>,
            &Option<Tensor>,
            u32,
            u32,
        )> = vec![
            (
                &w.mingru_to_z,
                &grads.mingru_to_z,
                &state.mom_mingru_to_z,
                &state.var_mingru_to_z,
                &state.prev_mingru_to_z,
                &state.ema_mingru_to_z,
                &state.mag_v_mingru_to_z,
                c,
                hid,
            ),
            (
                &w.mingru_to_h,
                &grads.mingru_to_h,
                &state.mom_mingru_to_h,
                &state.var_mingru_to_h,
                &state.prev_mingru_to_h,
                &state.ema_mingru_to_h,
                &state.mag_v_mingru_to_h,
                c,
                hid,
            ),
            (
                &w.mingru_out,
                &grads.mingru_out,
                &state.mom_mingru_out,
                &state.var_mingru_out,
                &state.prev_mingru_out,
                &state.ema_mingru_out,
                &state.mag_v_mingru_out,
                hid,
                c,
            ),
        ];
        if w.cfg.value_residual {
            mingru_banks.push((
                &w.mingru_v_proj,
                &grads.mingru_v_proj,
                &state.mom_mingru_v_proj,
                &state.var_mingru_v_proj,
                &state.prev_mingru_v_proj,
                &state.ema_mingru_v_proj,
                &state.mag_v_mingru_v_proj,
                c,
                kv,
            ));
            mingru_banks.push((
                &w.mingru_v0_up,
                &grads.mingru_v0_up,
                &state.mom_mingru_v0_up,
                &state.var_mingru_v0_up,
                &state.prev_mingru_v0_up,
                &state.ema_mingru_v0_up,
                &state.mag_v_mingru_v0_up,
                kv,
                hid,
            ));
        }
        for (pw, pg, pm, pv, pp, ema, mag, rows, cols) in mingru_banks {
            if let (Some(pw), Some(pg), Some(pm), Some(pv), Some(pp), Some(ema), Some(mag)) = (
                pw.as_ref(),
                pg.as_ref(),
                pm.as_ref(),
                pv.as_ref(),
                pp.as_ref(),
                ema.as_ref(),
                mag.as_ref(),
            ) {
                muon_bank(
                    rt,
                    pw,
                    pg,
                    pm,
                    pv,
                    pp,
                    mag,
                    &state.muon_scratch,
                    muon_coef,
                    n_mingru,
                    rows,
                    cols,
                    lr_m,
                    mom,
                    hp.weight_decay,
                    bank_scale(rows as usize, cols as usize),
                    if apply_ema { Some(ema) } else { None },
                    ema_d,
                    orthogonalizer,
                    post_kind,
                    hp.adam_beta2,
                    pre_kind,
                    hp.mona_beta_a,
                    pre_alpha,
                    state.step == 0,
                    state.step,
                    hp.muown_direction_scale,
                )?;
            }
        }
    }

    muon_bank(
        rt,
        &w.mlp_up,
        &grads.mlp_up,
        &state.mom_up,
        &state.var_up,
        &state.prev_up,
        &state.mag_v_up,
        &state.muon_scratch,
        muon_coef,
        n_l,
        c,
        mlp,
        lr_m,
        mom,
        hp.weight_decay,
        bank_scale(c as usize, mlp as usize),
        if apply_ema { Some(&state.ema_up) } else { None },
        ema_d,
        orthogonalizer,
        post_kind,
        hp.adam_beta2,
        pre_kind,
        hp.mona_beta_a,
        pre_alpha,
        state.step == 0,
        state.step,
        hp.muown_direction_scale,
    )?;
    oprof.lap(rt, "muon_mlp_up")?;
    muon_bank(
        rt,
        &w.mlp_down,
        &grads.mlp_down,
        &state.mom_dn,
        &state.var_dn,
        &state.prev_dn,
        &state.mag_v_dn,
        &state.muon_scratch,
        muon_coef,
        n_l,
        mlp,
        c,
        lr_m,
        mom,
        hp.weight_decay,
        bank_scale(mlp as usize, c as usize),
        if apply_ema { Some(&state.ema_dn) } else { None },
        ema_d,
        orthogonalizer,
        post_kind,
        hp.adam_beta2,
        pre_kind,
        hp.mona_beta_a,
        pre_alpha,
        state.step == 0,
        state.step,
        hp.muown_direction_scale,
    )?;
    oprof.lap(rt, "muon_mlp_down")?;

    oprof.report();

    state.step += 1;
    Ok(())
}

/// Copy EMA shadow weights into live `Weights` (for final sliding BPB).
/// Uses on-device `gpu_copy` blits (no host round-trip).
pub fn copy_ema_into_weights(
    rt: &Arc<GpuRuntime>,
    state: &OptimState,
    w: &mut Weights,
) -> Result<(), String> {
    use crate::tensor::gpu_copy;
    let blit = |dst: &Tensor, src: &Tensor| -> Result<(), String> {
        assert_eq!(dst.numel(), src.numel());
        gpu_copy(src, dst)
    };
    blit(&w.tok_emb, &state.ema_tok_emb)?;
    blit(&w.bigram_emb, &state.ema_bigram_emb)?;
    blit(&w.bigram_proj, &state.ema_bigram_proj)?;
    blit(&w.bigram_scale, &state.ema_bigram_scale)?;
    blit(&w.smear_gate, &state.ema_smear_gate)?;
    blit(&w.ve_emb, &state.ema_ve_emb)?;
    blit(&w.ve_proj, &state.ema_ve_proj)?;
    blit(&w.ve_scale, &state.ema_ve_scale)?;
    for (d, s) in w
        .ve_layer_scales
        .iter()
        .zip(state.ema_ve_layer_scales.iter())
    {
        blit(d, s)?;
    }
    blit(&w.skip_weights, &state.ema_skip_weights)?;
    blit(&w.qo_bank, &state.ema_qo)?;
    blit(&w.kv_bank, &state.ema_kv)?;
    blit(&w.mlp_up, &state.ema_up)?;
    blit(&w.mlp_down, &state.ema_dn)?;
    let blit_opt = |dst: &Option<Tensor>, src: &Option<Tensor>| -> Result<(), String> {
        if let (Some(d), Some(s)) = (dst, src) {
            blit(d, s)?;
        }
        Ok(())
    };
    blit_opt(&w.mingru_to_z, &state.ema_mingru_to_z)?;
    blit_opt(&w.mingru_to_h, &state.ema_mingru_to_h)?;
    blit_opt(&w.mingru_out, &state.ema_mingru_out)?;
    blit_opt(&w.mingru_v_proj, &state.ema_mingru_v_proj)?;
    blit_opt(&w.mingru_v0_up, &state.ema_mingru_v0_up)?;
    blit_opt(&w.mamba_in_proj, &state.ema_mamba_in_proj)?;
    blit_opt(&w.mamba_conv1d_weight, &state.ema_mamba_conv1d_weight)?;
    blit_opt(&w.mamba_conv1d_bias, &state.ema_mamba_conv1d_bias)?;
    blit_opt(&w.mamba_out_proj, &state.ema_mamba_out_proj)?;
    blit_opt(&w.mamba_a_log, &state.ema_mamba_a_log)?;
    blit_opt(&w.mamba_d, &state.ema_mamba_d)?;
    blit_opt(&w.mamba_dt_bias, &state.ema_mamba_dt_bias)?;
    blit_opt(&w.mamba_norm, &state.ema_mamba_norm)?;
    for (i, b) in w.blocks.iter().enumerate() {
        blit(&b.q_gain, &state.ema_blocks[i].q_gain)?;
        blit(&b.vr_lambda, &state.ema_blocks[i].vr_lambda)?;
        blit(&b.attn_scale, &state.ema_blocks[i].attn_scale)?;
        blit(&b.mlp_scale, &state.ema_blocks[i].mlp_scale)?;
        blit(&b.resid_mix, &state.ema_blocks[i].resid_mix)?;
    }

    rt.synchronize()?;
    refresh_tok_emb_t(rt, w)?;
    Ok(())
}

/// Zero all gradient buffers on GPU (Phase C). Packed into one binder (P2).
pub fn zero_grads(rt: &Arc<GpuRuntime>, grads: &Grads) -> Result<(), String> {
    let p = rt.pipeline("zero_f32")?;
    let width = p.threadExecutionWidth() as usize;
    let mut refs: Vec<&Tensor> = vec![
        &grads.tok_emb,
        &grads.bigram_emb,
        &grads.bigram_proj,
        &grads.bigram_scale,
        &grads.smear_gate,
        &grads.ve_emb,
        &grads.ve_proj,
        &grads.ve_scale,
        &grads.skip_weights,
        &grads.qo_bank,
        &grads.kv_bank,
        &grads.mlp_up,
        &grads.mlp_down,
    ];
    for s in &grads.ve_layer_scales {
        refs.push(s);
    }
    for t in [
        &grads.mingru_to_z,
        &grads.mingru_to_h,
        &grads.mingru_out,
        &grads.mingru_v_proj,
        &grads.mingru_v0_up,
        &grads.mamba_in_proj,
        &grads.mamba_conv1d_weight,
        &grads.mamba_conv1d_bias,
        &grads.mamba_out_proj,
        &grads.mamba_a_log,
        &grads.mamba_d,
        &grads.mamba_dt_bias,
        &grads.mamba_norm,
    ] {
        if let Some(g) = t {
            refs.push(g);
        }
    }
    for b in &grads.blocks {
        refs.push(&b.q_gain);
        refs.push(&b.vr_lambda);
        refs.push(&b.attn_scale);
        refs.push(&b.mlp_scale);
        refs.push(&b.resid_mix);
    }
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&p);
        for (i, t) in refs.iter().enumerate() {
            if i > 0 {
                bnd.barrier();
            }
            let n = t.numel();
            if n == 0 {
                continue;
            }
            let tpt = width.min(n).max(1);
            let groups = (n + tpt - 1) / tpt;
            set_tensor(bnd, t, 0);
            set_u32(bnd, n as u32, 1);
            bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
        }
        Ok(())
    })
}

#[cfg(test)]
mod ns5_tests {
    use super::*;
    use crate::init::init_weights_seeded;
    use crate::model_bwd::Grads;
    use crate::runtime::GpuRuntime;
    use crate::weights::ModelConfig;

    #[test]
    fn native_research_controls_match_first_step_formulas() {
        for kind in [
            OptimizerKind::Adamw,
            OptimizerKind::Lion,
            OptimizerKind::CautiousAdamw,
            OptimizerKind::CautiousLion,
            OptimizerKind::SgdMomentum,
            OptimizerKind::Sophia,
            OptimizerKind::ScheduleFreeAdamw,
            OptimizerKind::Prodigy,
        ] {
            let rt = crate::gpu_runtime().expect("gpu");
            let cfg = ModelConfig::sota_toy();
            let mut w = init_weights_seeded(&rt, cfg, 77).expect("weights");
            let grads = Grads::zeros_like(&rt, &w).expect("grads");
            for value in grads.qo_bank.buffer.contents_f32().iter_mut() {
                *value = 0.2;
            }
            let before = w.qo_bank.buffer.contents_f32()[0];
            let hp = OptimHyperparams {
                matrix_lr: 0.01,
                weight_decay: 0.0,
                ..OptimHyperparams::default()
            };
            let mut state = OptimState::new_for_kind(&rt, &w, hp.clone(), kind).expect("state");
            optim_step(&rt, &mut w, &grads, &mut state, true, 1.0).expect("step");
            rt.synchronize().expect("sync");
            let actual = w.qo_bank.buffer.contents_f32()[0];
            let expected = match kind {
                OptimizerKind::Adamw | OptimizerKind::CautiousAdamw => {
                    before - hp.matrix_lr * 0.2 / (0.2 + hp.adam_eps)
                }
                OptimizerKind::Lion | OptimizerKind::CautiousLion | OptimizerKind::Sophia => {
                    before - hp.matrix_lr
                }
                OptimizerKind::SgdMomentum => before - hp.matrix_lr * 0.2,
                OptimizerKind::ScheduleFreeAdamw => {
                    let denom = ((1.0 - hp.adam_beta2) * 0.2f32.powi(2)).sqrt() + hp.adam_eps;
                    before - hp.matrix_lr * 0.2 / denom
                }
                OptimizerKind::Prodigy => {
                    let d = 1e-6f32;
                    let bc = (1.0 - hp.adam_beta2).sqrt() / (1.0 - hp.adam_beta1);
                    let dlr = d * bc;
                    let m = d * (1.0 - hp.adam_beta1) * 0.2;
                    let v = d * d * (1.0 - hp.adam_beta2) * 0.2f32.powi(2);
                    before - dlr * m / (v.sqrt() + d * hp.adam_eps)
                }
                _ => unreachable!(),
            };
            assert!(
                (actual - expected).abs() < 2e-5,
                "{kind}: actual={actual} expected={expected}"
            );
            assert_eq!(state.step, 1);
        }
    }

    fn run_ns_only(
        rt: &std::sync::Arc<GpuRuntime>,
        g: &Tensor,
        n: u32,
        rows: u32,
        cols: u32,
        ns_steps: u32,
    ) -> Result<Vec<f32>, String> {
        run_orth_only(
            rt,
            g,
            n,
            rows,
            cols,
            MuonOrthogonalizer::NewtonSchulz(ns_steps),
        )
    }

    fn run_orth_only(
        rt: &std::sync::Arc<GpuRuntime>,
        g: &Tensor,
        n: u32,
        rows: u32,
        cols: u32,
        orthogonalizer: MuonOrthogonalizer,
    ) -> Result<Vec<f32>, String> {
        // param starts as zeros; momentum zeros; after muon with lr=1, wd=0, scale=1, mom=0:
        // buf = g, go = g, u = ns5(g), param = 0 - 1*1*u = -u
        let param = zeros_like(rt, g)?;
        let mom = zeros_like(rt, g)?;
        let aux = zeros_like(rt, g)?;
        let prev = zeros_like(rt, g)?;
        let extra = zeros_like(rt, g)?;
        let nbytes = muon_scratch_bytes_for(n as usize, rows as usize, cols as usize) * 4;
        let scratch = rt.alloc_buffer(nbytes)?;
        scratch.zero();
        let clip_coef = rt.alloc_buffer(4)?;
        clip_coef.write_f32(&[1.0]);
        muon_bank(
            rt,
            &param,
            g,
            &mom,
            &aux,
            &prev,
            &extra,
            &scratch,
            &clip_coef,
            n,
            rows,
            cols,
            1.0,
            0.0,
            0.0,
            1.0,
            None,
            0.0,
            orthogonalizer,
            0,
            0.95,
            0,
            0.99,
            0.0,
            true,
            0,
            0.2,
        )?;
        let p = param.buffer.read_f32();
        Ok(p.iter().map(|x| -x).collect())
    }

    fn run_ns_tensorops_only(
        rt: &std::sync::Arc<GpuRuntime>,
        g: &Tensor,
        n: u32,
        rows: u32,
        cols: u32,
        ns_steps: u32,
    ) -> Result<Vec<f32>, String> {
        let param = zeros_like(rt, g)?;
        let mom = zeros_like(rt, g)?;
        let aux = zeros_like(rt, g)?;
        let prev = zeros_like(rt, g)?;
        let extra = zeros_like(rt, g)?;
        let p = rows.min(cols) as usize;
        let mat = rows as usize * cols as usize;
        let scratch = rt.alloc_buffer(n as usize * (2 * mat + 2 * p * p) * 4)?;
        let clip = rt.alloc_buffer(4)?;
        clip.write_f32(&[1.0]);
        muon_bank_tensorops(
            rt,
            &param,
            g,
            &mom,
            &aux,
            &prev,
            &extra,
            &scratch,
            &clip,
            n,
            rows,
            cols,
            1.0,
            0.0,
            0.0,
            1.0,
            None,
            0.0,
            MuonOrthogonalizer::NewtonSchulz(ns_steps),
            0,
            0.95,
            0,
            0.99,
            0.0,
            true,
            0,
            0.2,
        )?;
        rt.synchronize()?;
        Ok(param.buffer.read_f32().iter().map(|x| -x).collect())
    }

    fn muon_scratch_bytes_for(n: usize, rows: usize, cols: usize) -> usize {
        n * muon_scratch_stride(rows, cols)
    }

    #[test]
    fn metal_ns5_matches_numpy_fixture() {
        let rt = crate::gpu_runtime().expect("gpu");
        // Generate same fixture as python seed 42
        // We embed a tiny deterministic check: square 4x4
        let n = 1u32;
        let r = 4u32;
        let c = 4u32;
        let data: Vec<f32> = (0..16).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
        let g = rt.alloc_tensor_f32(&[1, 4, 4]).unwrap();
        g.buffer.write_f32(&data);
        let got = run_ns_only(&rt, &g, n, r, c, NS_STEPS).unwrap();
        // numpy reference inline
        // compute host ns5
        let exp = host_ns(&data, 4, 4, NS_STEPS as usize);
        let max_abs = got
            .iter()
            .zip(exp.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, crate::parity::max_finite_error);
        eprintln!("4x4 ns5 max_abs={max_abs}");
        assert!(max_abs < 1e-4, "ns5 mismatch {max_abs}");
    }

    fn host_ns(g: &[f32], rows: usize, cols: usize, steps: usize) -> Vec<f32> {
        host_orth(g, rows, cols, &vec![(NS_A, NS_B, NS_C); steps], 1.0)
    }

    fn host_orth(
        g: &[f32],
        rows: usize,
        cols: usize,
        coefficients: &[(f32, f32, f32)],
        norm_scale: f32,
    ) -> Vec<f32> {
        let needs_t = rows > cols;
        let p = rows.min(cols);
        let q = rows.max(cols);
        let mut x = vec![0.0f32; p * q];
        for i in 0..p {
            for j in 0..q {
                x[i * q + j] = if needs_t {
                    g[j * cols + i]
                } else {
                    g[i * cols + j]
                };
            }
        }
        let mut nrm = 0.0f32;
        for v in &x {
            nrm += v * v;
        }
        nrm = nrm.sqrt() * norm_scale + 1e-7;
        for v in &mut x {
            *v /= nrm;
        }
        let mut aa = vec![0.0f32; p * p];
        let mut bb = vec![0.0f32; p * p];
        let mut tmp = vec![0.0f32; p * q];
        for &(a, b, c) in coefficients {
            for i in 0..p {
                for j in 0..p {
                    let mut s = 0.0f32;
                    for k in 0..q {
                        s += x[i * q + k] * x[j * q + k];
                    }
                    aa[i * p + j] = s;
                }
            }
            for i in 0..p {
                for j in 0..p {
                    let mut a2 = 0.0f32;
                    for k in 0..p {
                        a2 += aa[i * p + k] * aa[k * p + j];
                    }
                    bb[i * p + j] = b * aa[i * p + j] + c * a2;
                }
            }
            for i in 0..p {
                for j in 0..q {
                    let mut bx = 0.0f32;
                    for k in 0..p {
                        bx += bb[i * p + k] * x[k * q + j];
                    }
                    tmp[i * q + j] = a * x[i * q + j] + bx;
                }
            }
            x.copy_from_slice(&tmp);
        }
        let mut out = vec![0.0f32; rows * cols];
        for rr in 0..rows {
            for cc in 0..cols {
                out[rr * cols + cc] = if needs_t {
                    x[cc * q + rr]
                } else {
                    x[rr * q + cc]
                };
            }
        }
        out
    }

    #[test]
    fn metal_polar_express_matches_host_fixture() {
        let rt = crate::gpu_runtime().expect("gpu");
        let data: Vec<f32> = (0..2 * 16 * 8)
            .map(|i| ((i % 23) as f32) * 0.011 - 0.10)
            .collect();
        let g = rt.alloc_tensor_f32(&[2, 16, 8]).unwrap();
        g.buffer.write_f32(&data);
        let got = run_orth_only(&rt, &g, 2, 16, 8, MuonOrthogonalizer::PolarExpress).unwrap();
        let mut expected = Vec::with_capacity(data.len());
        for matrix in data.chunks_exact(16 * 8) {
            expected.extend(host_orth(matrix, 16, 8, &POLAR_COEFFS, 1.02));
        }
        let max_abs = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, crate::parity::max_finite_error);
        assert!(max_abs < 2e-3, "Polar Express mismatch {max_abs}");
    }

    #[test]
    fn metal_normuon_row_state_matches_host_fixture() {
        let rt = crate::gpu_runtime().expect("gpu");
        let (batch, rows, cols) = (2u32, 8u32, 16u32);
        let mat = rows as usize * cols as usize;
        let data: Vec<f32> = (0..batch as usize * mat)
            .map(|i| ((i % 29) as f32) * 0.009 - 0.12)
            .collect();
        let g = rt
            .alloc_tensor_f32(&[batch as usize, rows as usize, cols as usize])
            .unwrap();
        g.buffer.write_f32(&data);
        let param = zeros_like(&rt, &g).unwrap();
        let mom = zeros_like(&rt, &g).unwrap();
        let row_state = zeros_like(&rt, &g).unwrap();
        let prev = zeros_like(&rt, &g).unwrap();
        let extra = zeros_like(&rt, &g).unwrap();
        let p = rows.min(cols) as usize;
        let scratch = rt
            .alloc_buffer(batch as usize * (2 * mat + 2 * p * p) * 4)
            .unwrap();
        let clip = rt.alloc_buffer(4).unwrap();
        clip.write_f32(&[1.0]);
        muon_bank_tensorops(
            &rt,
            &param,
            &g,
            &mom,
            &row_state,
            &prev,
            &extra,
            &scratch,
            &clip,
            batch,
            rows,
            cols,
            1.0,
            0.0,
            0.0,
            1.0,
            None,
            0.0,
            MuonOrthogonalizer::NewtonSchulz(5),
            1,
            0.95,
            0,
            0.99,
            0.0,
            true,
            0,
            0.2,
        )
        .unwrap();
        rt.synchronize().unwrap();
        let got: Vec<f32> = param.buffer.read_f32().into_iter().map(|v| -v).collect();
        let mut expected = Vec::with_capacity(data.len());
        for matrix in data.chunks_exact(mat) {
            let mut x = host_ns(matrix, rows as usize, cols as usize, 5);
            let before = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            for col in 0..cols as usize {
                let row_ms = (0..rows as usize)
                    .map(|row| x[row * cols as usize + col].powi(2))
                    .sum::<f32>()
                    / rows as f32;
                let second = 0.05 * row_ms;
                let inv = 1.0 / (second.sqrt() + 1e-10);
                for row in 0..rows as usize {
                    x[row * cols as usize + col] *= inv;
                }
            }
            let after = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            for value in &mut x {
                *value *= before / (after + 1e-10);
            }
            expected.extend(x);
        }
        let max_abs = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, crate::parity::max_finite_error);
        assert!(max_abs < 2e-3, "NorMuon mismatch {max_abs}");
        assert!(row_state.buffer.contents_f32()[0] > 0.0);
    }

    #[test]
    fn metal_mona_two_steps_match_host_fixture() {
        let rt = crate::gpu_runtime().expect("gpu");
        let (rows, cols) = (8u32, 16u32);
        let mat = rows as usize * cols as usize;
        let g1_data: Vec<f32> = (0..mat).map(|i| (i % 17) as f32 * 0.01 - 0.08).collect();
        let g2_data: Vec<f32> = (0..mat).map(|i| (i % 19) as f32 * 0.008 - 0.06).collect();
        let g = rt
            .alloc_tensor_f32(&[1, rows as usize, cols as usize])
            .unwrap();
        let param = zeros_like(&rt, &g).unwrap();
        let mom = zeros_like(&rt, &g).unwrap();
        let acc = zeros_like(&rt, &g).unwrap();
        let prev = zeros_like(&rt, &g).unwrap();
        let extra = zeros_like(&rt, &g).unwrap();
        let p = rows.min(cols) as usize;
        let scratch = rt.alloc_buffer((2 * mat + 2 * p * p) * 4).unwrap();
        let clip = rt.alloc_buffer(4).unwrap();
        clip.write_f32(&[1.0]);
        let beta = 0.95;
        let beta_a = 0.99;
        let alpha = -1.0 / (2.0 * (1.0 - beta_a));
        for (step, values) in [g1_data.as_slice(), g2_data.as_slice()]
            .into_iter()
            .enumerate()
        {
            g.buffer.write_f32(values);
            muon_bank_tensorops(
                &rt,
                &param,
                &g,
                &mom,
                &acc,
                &prev,
                &extra,
                &scratch,
                &clip,
                1,
                rows,
                cols,
                1.0,
                beta,
                0.0,
                1.0,
                None,
                0.0,
                MuonOrthogonalizer::NewtonSchulz(5),
                0,
                0.95,
                1,
                beta_a,
                alpha,
                step == 0,
                step,
                0.2,
            )
            .unwrap();
        }
        rt.synchronize().unwrap();
        let mut host_mom = vec![0.0f32; mat];
        let mut host_acc = vec![0.0f32; mat];
        let mut host_prev = vec![0.0f32; mat];
        let mut expected_param = vec![0.0f32; mat];
        for (step, values) in [g1_data.as_slice(), g2_data.as_slice()]
            .into_iter()
            .enumerate()
        {
            let mut transformed = vec![0.0f32; mat];
            for i in 0..mat {
                let diff = if step == 0 {
                    0.0
                } else {
                    values[i] - host_prev[i]
                };
                host_acc[i] = beta_a * host_acc[i] + (1.0 - beta_a) * diff;
                transformed[i] = values[i] + alpha * host_acc[i];
                host_prev[i] = values[i];
                host_mom[i] = beta * host_mom[i] + transformed[i];
                transformed[i] += beta * host_mom[i];
            }
            let update = host_ns(&transformed, rows as usize, cols as usize, 5);
            for i in 0..mat {
                expected_param[i] -= update[i];
            }
        }
        let got = param.buffer.read_f32();
        let max_abs = got
            .iter()
            .zip(expected_param.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, crate::parity::max_finite_error);
        assert!(max_abs < 3e-3, "MONA mismatch {max_abs}");
        assert!((acc.buffer.contents_f32()[0] - host_acc[0]).abs() < 1e-6);
    }

    #[test]
    fn metal_muown_first_step_matches_host_fixture() {
        let rt = crate::gpu_runtime().expect("gpu");
        let (rows, cols) = (8u32, 16u32);
        let mat = rows as usize * cols as usize;
        let initial: Vec<f32> = (0..mat).map(|i| (i % 31) as f32 * 0.004 - 0.06).collect();
        let grad_data: Vec<f32> = (0..mat).map(|i| (i % 23) as f32 * 0.006 - 0.05).collect();
        let param = rt
            .alloc_tensor_f32(&[1, rows as usize, cols as usize])
            .unwrap();
        param.buffer.write_f32(&initial);
        let grad = zeros_like(&rt, &param).unwrap();
        grad.buffer.write_f32(&grad_data);
        let direction = clone_like(&rt, &param).unwrap();
        let direction_mom = zeros_like(&rt, &param).unwrap();
        let mag_m = zeros_like(&rt, &param).unwrap();
        let mag_v = zeros_like(&rt, &param).unwrap();
        let p = rows.min(cols) as usize;
        let scratch = rt.alloc_buffer((2 * mat + 2 * p * p) * 4).unwrap();
        let clip = rt.alloc_buffer(4).unwrap();
        clip.write_f32(&[1.0]);
        muon_bank_tensorops(
            &rt,
            &param,
            &grad,
            &direction_mom,
            &mag_m,
            &direction,
            &mag_v,
            &scratch,
            &clip,
            1,
            rows,
            cols,
            0.01,
            0.95,
            0.0,
            1.0,
            None,
            0.0,
            MuonOrthogonalizer::NewtonSchulz(5),
            2,
            0.95,
            2,
            0.99,
            0.0,
            true,
            0,
            0.2,
        )
        .unwrap();
        rt.synchronize().unwrap();

        let mut nesterov = vec![0.0f32; mat];
        let mut radial = vec![0.0f32; cols as usize];
        for col in 0..cols as usize {
            let r = (0..rows as usize)
                .map(|row| initial[row * cols as usize + col].powi(2))
                .sum::<f32>()
                .sqrt()
                .max(NS_EPS);
            for row in 0..rows as usize {
                let i = row * cols as usize + col;
                radial[col] += grad_data[i] * initial[i] / r;
            }
            for row in 0..rows as usize {
                let i = row * cols as usize + col;
                let tangent = grad_data[i] - initial[i] / r * radial[col];
                nesterov[i] = (1.0 + 0.95) * tangent;
            }
        }
        let ortho = host_ns(&nesterov, rows as usize, cols as usize, 5);
        let mut expected = initial.clone();
        let dir_alpha = 0.01 * 0.2 * (cols.max(rows) as f32).sqrt();
        for col in 0..cols as usize {
            let old_mag = (0..rows as usize)
                .map(|row| initial[row * cols as usize + col].powi(2))
                .sum::<f32>()
                .sqrt();
            let new_mag = old_mag - 0.01 * radial[col] / (radial[col].abs() + NS_EPS);
            let new_r = (0..rows as usize)
                .map(|row| {
                    let i = row * cols as usize + col;
                    (initial[i] - dir_alpha * ortho[i]).powi(2)
                })
                .sum::<f32>()
                .sqrt()
                .max(NS_EPS);
            for row in 0..rows as usize {
                let i = row * cols as usize + col;
                expected[i] = new_mag / new_r * (initial[i] - dir_alpha * ortho[i]);
            }
        }
        let got = param.buffer.read_f32();
        let max_abs = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, crate::parity::max_finite_error);
        assert!(max_abs < 3e-3, "Muown mismatch {max_abs}");
        assert!(mag_v.buffer.contents_f32()[0] > 0.0);
    }

    #[test]
    fn metal_ns5_tall_matrix() {
        let rt = crate::gpu_runtime().expect("gpu");
        let data: Vec<f32> = (0..128 * 64)
            .map(|i| ((i % 11) as f32) * 0.01 - 0.05)
            .collect();
        let g = rt.alloc_tensor_f32(&[1, 128, 64]).unwrap();
        g.buffer.write_f32(&data);
        let got = run_ns_only(&rt, &g, 1, 128, 64, NS_STEPS).unwrap();
        let exp = host_ns(&data, 128, 64, NS_STEPS as usize);
        let max_abs = got
            .iter()
            .zip(exp.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, crate::parity::max_finite_error);
        eprintln!("128x64 ns5 max_abs={max_abs}");
        assert!(max_abs < 1e-3, "tall ns5 mismatch {max_abs}");
    }

    #[test]
    fn metal_ns3_matches_host_fixture() {
        let rt = crate::gpu_runtime().expect("gpu");
        let data: Vec<f32> = (0..3 * 16 * 8)
            .map(|i| ((i % 17) as f32) * 0.015 - 0.11)
            .collect();
        let g = rt.alloc_tensor_f32(&[3, 16, 8]).unwrap();
        g.buffer.write_f32(&data);
        let got = run_ns_only(&rt, &g, 3, 16, 8, 3).unwrap();
        let mut exp = Vec::with_capacity(data.len());
        for matrix in data.chunks_exact(16 * 8) {
            exp.extend(host_ns(matrix, 16, 8, 3));
        }
        let max_abs = got
            .iter()
            .zip(exp.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, crate::parity::max_finite_error);
        assert!(max_abs < 1e-3, "ns3 mismatch {max_abs}");
    }

    #[test]
    fn metal_tensorops_batched_ns3_tall_and_wide_match_host() {
        let rt = crate::gpu_runtime().expect("gpu");
        for &(n, rows, cols) in &[(3u32, 16u32, 8u32), (2, 8, 16)] {
            let matrix = rows as usize * cols as usize;
            let data: Vec<f32> = (0..n as usize * matrix)
                .map(|i| ((i % 19) as f32) * 0.013 - 0.09)
                .collect();
            let g = rt
                .alloc_tensor_f32(&[n as usize, rows as usize, cols as usize])
                .unwrap();
            g.buffer.write_f32(&data);
            let got = run_ns_tensorops_only(&rt, &g, n, rows, cols, 3).unwrap();
            let mut expected = Vec::with_capacity(data.len());
            for m in data.chunks_exact(matrix) {
                expected.extend(host_ns(m, rows as usize, cols as usize, 3));
            }
            let max_abs = got
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, crate::parity::max_finite_error);
            assert!(max_abs < 1e-3, "TensorOps {rows}x{cols} mismatch {max_abs}");
        }
    }

    #[test]
    fn adamw_one_step_matches_host() {
        let rt = crate::gpu_runtime().expect("gpu");
        let n = 128usize;
        let data: Vec<f32> = (0..n).map(|i| 0.01 * (i as f32)).collect();
        let grad: Vec<f32> = (0..n).map(|i| 0.001 * ((i % 5) as f32 - 2.0)).collect();
        let param = rt.alloc_tensor_f32(&[n]).unwrap();
        param.buffer.write_f32(&data);
        let g = rt.alloc_tensor_f32(&[n]).unwrap();
        g.buffer.write_f32(&grad);
        let slot = AdamSlot::zeros(&rt, &param).unwrap();
        let lr = 0.025f32;
        let beta1 = 0.9f32;
        let beta2 = 0.95f32;
        let eps = 1e-8f32;
        let wd = 0.04f32;
        let t = 1i32;
        let bc1 = 1.0 - beta1.powi(t);
        let bc2 = 1.0 - beta2.powi(t);
        let step_size = lr / bc1;
        let bias2 = 1.0 / bc2.sqrt();
        let clip_coef = rt.alloc_buffer(4).unwrap();
        clip_coef.write_f32(&[1.0]);
        adamw_ema_one(
            &rt, &param, &g, &slot, None, &clip_coef, lr, beta1, beta2, eps, wd, step_size, bias2,
            0.0,
        )
        .unwrap();
        // host
        let mut m = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        let mut ph = data.clone();
        for i in 0..n {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
            let denom = v[i].sqrt() * bias2 + eps;
            ph[i] = ph[i] * (1.0 - lr * wd) - step_size * (m[i] / denom);
        }
        let got = param.buffer.read_f32();
        let max_abs = got
            .iter()
            .zip(ph.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, crate::parity::max_finite_error);
        let mom_err = slot
            .exp_avg
            .buffer
            .read_f32()
            .iter()
            .zip(m.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, crate::parity::max_finite_error);
        eprintln!("adamw param max_abs={max_abs} mom max_abs={mom_err}");
        assert!(max_abs < 1e-6 && mom_err < 1e-6);
    }

    #[test]
    fn lr_warmdown_schedule() {
        let total = 20_000usize;
        let wd = 3_500usize;
        assert_eq!(lr_mul_at(0, total, wd), 1.0);
        assert_eq!(lr_mul_at(16_499, total, wd), 1.0);
        assert!((lr_mul_at(16_500, total, wd) - 1.0).abs() < 1e-6);
        assert!((lr_mul_at(18_250, total, wd) - 0.5).abs() < 1e-6);
        assert!(lr_mul_at(19_999, total, wd) > 0.0);
        assert_eq!(lr_mul_at(100, total, 0), 1.0);
    }

    #[test]
    fn lr_wsd_long_horizon_schedule() {
        // 100k Soft recipe: constant → main warmdown to floor → hold → final→0.
        let sched = LrSchedule {
            total_iters: 100_000,
            warmdown_start: Some(16_000),
            warmdown_iters: 24_000,
            lr_floor: 0.1,
            final_warmdown: 10_000,
        };
        assert!((sched.mul_at(0) - 1.0).abs() < 1e-6);
        assert!((sched.mul_at(15_999) - 1.0).abs() < 1e-6);
        assert!((sched.mul_at(16_000) - 1.0).abs() < 1e-6);
        // Mid main warmdown: t=0.5 → 0.55.
        assert!((sched.mul_at(28_000) - 0.55).abs() < 1e-5);
        assert!((sched.mul_at(40_000) - 0.1).abs() < 1e-6);
        assert!((sched.mul_at(89_999) - 0.1).abs() < 1e-6);
        assert!((sched.mul_at(90_000) - 0.1).abs() < 1e-6);
        assert!((sched.mul_at(95_000) - 0.05).abs() < 1e-5);
        assert!(sched.mul_at(99_999) > 0.0);
        assert!(sched.mul_at(99_999) < 0.01);
    }

    #[test]
    fn lr_warmdown_start_only_decays_to_end() {
        let sched = LrSchedule {
            total_iters: 100_000,
            warmdown_start: Some(20_000),
            warmdown_iters: 0,
            lr_floor: 0.0,
            final_warmdown: 0,
        };
        assert_eq!(sched.main_window(), (20_000, 80_000));
        assert!((sched.mul_at(20_000) - 1.0).abs() < 1e-6);
        assert!((sched.mul_at(60_000) - 0.5).abs() < 1e-5);
    }
}
