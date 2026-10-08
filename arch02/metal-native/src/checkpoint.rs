//! Dump metal-native GPU weights / optimizer state to a Python-layout `.npy` tree.
//!
//! Layout matches `golden/weights_init` / `golden/optim_step3` / the Python exporter:
//! - embeddings stay `[vocab, dim]`
//! - linear / bank matrices are written as `[out, in]` (transpose last-2 from
//!   metal-native `[in, out]`)

use std::path::Path;
use std::sync::Arc;

use objc2_metal::MTLComputePipelineState;
use serde::{Deserialize, Serialize};

use crate::dispatch::{set_gpu_buf, set_tensor, set_u32};
use crate::npy::{transpose_last2, write_npy_f32};
use crate::optim::{AdamSlot, ClipMode, LrSchedule, OptimHyperparams, OptimState};
use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::{DType, GpuBuffer, Tensor};
use crate::weights::ModelConfig;
use crate::weights::Weights;

pub const CHECKPOINT_VERSION: u32 = 7;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingCheckpointMeta {
    pub version: u32,
    pub step: usize,
    pub data_cursor_tokens: usize,
    pub seed: u64,
    pub preset: String,
    pub config: ModelConfig,
    pub parameter_count: usize,
    pub optimizer: String,
    pub hyperparams: OptimHyperparams,
    pub clip_mode: ClipMode,
    pub schedule: LrSchedule,
    pub bf16_precision: bool,
    pub bf16_shadows_saved: bool,
}

fn save_f32(path: &Path, t: &Tensor) -> Result<(), String> {
    let data = t.buffer.read_f32();
    write_npy_f32(path, &t.shape(), &data)
}

fn save_bf16_bits(path: &Path, t: &Tensor) -> Result<(), String> {
    if t.dtype != DType::BF16 || t.byte_offset() != 0 {
        return Err(format!(
            "bf16 checkpoint tensor {} must be an owning, zero-offset BF16 tensor",
            path.display()
        ));
    }
    let backing = t.buffer.contents_u16();
    if backing.len() < t.numel() {
        return Err(format!(
            "bf16 checkpoint tensor {} has {} backing elements, needs {}",
            path.display(),
            backing.len(),
            t.numel()
        ));
    }
    let bits = &backing[..t.numel()];
    let bytes = unsafe {
        std::slice::from_raw_parts(bits.as_ptr().cast::<u8>(), bits.len() * size_of::<u16>())
    };
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn load_bf16_bits(path: &Path, t: &Tensor) -> Result<(), String> {
    if t.dtype != DType::BF16 || t.byte_offset() != 0 {
        return Err(format!(
            "bf16 checkpoint tensor {} must be an owning, zero-offset BF16 tensor",
            path.display()
        ));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let expected = t.numel() * size_of::<u16>();
    if bytes.len() != expected {
        return Err(format!(
            "bf16 checkpoint tensor {} is {} bytes, expected {expected}",
            path.display(),
            bytes.len()
        ));
    }
    let bits: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    let mut backing = t.buffer.contents_u16();
    if backing.len() < bits.len() {
        return Err(format!(
            "bf16 destination {} has {} backing elements, needs {}",
            path.display(),
            backing.len(),
            bits.len()
        ));
    }
    backing[..bits.len()].copy_from_slice(&bits);
    Ok(())
}

/// Persist the actual bf16 shadow bits instead of relying on recasting masters.
pub fn save_bf16_shadows(w: &Weights, dir: &Path) -> Result<bool, String> {
    let Some(banks) = &w.bf16_banks else {
        return Ok(false);
    };
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    for (name, tensor) in [
        ("qo_bank.bf16le", &banks.qo_bank),
        ("kv_bank.bf16le", &banks.kv_bank),
        ("mlp_up.bf16le", &banks.mlp_up),
        ("mlp_down.bf16le", &banks.mlp_down),
        ("ve_proj.bf16le", &banks.ve_proj),
        ("bigram_proj.bf16le", &banks.bigram_proj),
    ] {
        save_bf16_bits(&dir.join(name), tensor)?;
    }
    Ok(true)
}

/// Restore every persisted bf16 shadow after the runtime allocates its hot banks.
pub fn load_bf16_shadows(w: &mut Weights, dir: &Path) -> Result<(), String> {
    let banks = w.bf16_banks.as_ref().ok_or_else(|| {
        "checkpoint contains bf16 shadows but engine is not in bf16 mode".to_string()
    })?;
    for (name, tensor) in [
        ("qo_bank.bf16le", &banks.qo_bank),
        ("kv_bank.bf16le", &banks.kv_bank),
        ("mlp_up.bf16le", &banks.mlp_up),
        ("mlp_down.bf16le", &banks.mlp_down),
        ("ve_proj.bf16le", &banks.ve_proj),
        ("bigram_proj.bf16le", &banks.bigram_proj),
    ] {
        load_bf16_bits(&dir.join(name), tensor)?;
    }
    Ok(())
}

/// Save a metal-native `[in, out]` (or bank `[N, in, out]`) tensor as Python
/// `[out, in]` / `[N, out, in]`.
fn save_linear_python(path: &Path, t: &Tensor) -> Result<(), String> {
    let mut data = t.buffer.read_f32();
    let mut shape = t.shape().to_vec();
    transpose_last2(&mut data, &mut shape)?;
    write_npy_f32(path, &shape, &data)
}

fn save_scalar_f32(path: &Path, v: f32) -> Result<(), String> {
    write_npy_f32(path, &[], &[v])
}

/// Save optional Muon bank momentum/aux/prev/mag_v (mirrors `load_muon_momentum_python_npy`).
fn save_muon_bank_opt(
    muon: &Path,
    stem: &str,
    mom: Option<&Tensor>,
    var: Option<&Tensor>,
    prev: Option<&Tensor>,
    mag: Option<&Tensor>,
) -> Result<(), String> {
    let Some(mom) = mom else {
        return Ok(());
    };
    save_linear_python(&muon.join(format!("{stem}_momentum_buffer.npy")), mom)?;
    if let Some(t) = var {
        save_linear_python(&muon.join(format!("{stem}_aux_state.npy")), t)?;
    }
    if let Some(t) = prev {
        save_linear_python(&muon.join(format!("{stem}_prev_state.npy")), t)?;
    }
    if let Some(t) = mag {
        save_linear_python(&muon.join(format!("{stem}_mag_v.npy")), t)?;
    }
    Ok(())
}

fn save_ssm_ema_banks(state: &OptimState, dir: &Path) -> Result<(), String> {
    if let Some(t) = &state.ema_mingru_to_z {
        save_linear_python(&dir.join("mingru_to_z.npy"), t)?;
    }
    if let Some(t) = &state.ema_mingru_to_h {
        save_linear_python(&dir.join("mingru_to_h.npy"), t)?;
    }
    if let Some(t) = &state.ema_mingru_out {
        save_linear_python(&dir.join("mingru_out.npy"), t)?;
    }
    if let Some(t) = &state.ema_mingru_v_proj {
        save_linear_python(&dir.join("mingru_v_proj.npy"), t)?;
    }
    if let Some(t) = &state.ema_mingru_v0_up {
        save_linear_python(&dir.join("mingru_v0_up.npy"), t)?;
    }
    if let Some(t) = &state.ema_mamba_in_proj {
        save_linear_python(&dir.join("mamba_in_proj.npy"), t)?;
    }
    if let Some(t) = &state.ema_mamba_conv1d_weight {
        save_f32(&dir.join("mamba_conv1d_weight.npy"), t)?;
    }
    if let Some(t) = &state.ema_mamba_conv1d_bias {
        save_f32(&dir.join("mamba_conv1d_bias.npy"), t)?;
    }
    if let Some(t) = &state.ema_mamba_out_proj {
        save_linear_python(&dir.join("mamba_out_proj.npy"), t)?;
    }
    if let Some(t) = &state.ema_mamba_a_log {
        save_f32(&dir.join("mamba_a_log.npy"), t)?;
    }
    if let Some(t) = &state.ema_mamba_d {
        save_f32(&dir.join("mamba_d.npy"), t)?;
    }
    if let Some(t) = &state.ema_mamba_dt_bias {
        save_f32(&dir.join("mamba_dt_bias.npy"), t)?;
    }
    if let Some(t) = &state.ema_mamba_norm {
        save_f32(&dir.join("mamba_norm.npy"), t)?;
    }
    Ok(())
}

fn load_ssm_ema_banks(state: &mut OptimState, dir: &Path) -> Result<(), String> {
    let load_opt = |dst: &mut Option<Tensor>, name: &str, transpose: bool| -> Result<(), String> {
        let Some(dst) = dst.as_mut() else {
            return Ok(());
        };
        let path = dir.join(name);
        if path.exists() {
            load_tensor_f32(&path, dst, transpose)?;
        }
        Ok(())
    };
    load_opt(&mut state.ema_mingru_to_z, "mingru_to_z.npy", true)?;
    load_opt(&mut state.ema_mingru_to_h, "mingru_to_h.npy", true)?;
    load_opt(&mut state.ema_mingru_out, "mingru_out.npy", true)?;
    load_opt(&mut state.ema_mingru_v_proj, "mingru_v_proj.npy", true)?;
    load_opt(&mut state.ema_mingru_v0_up, "mingru_v0_up.npy", true)?;
    load_opt(&mut state.ema_mamba_in_proj, "mamba_in_proj.npy", true)?;
    load_opt(
        &mut state.ema_mamba_conv1d_weight,
        "mamba_conv1d_weight.npy",
        false,
    )?;
    load_opt(
        &mut state.ema_mamba_conv1d_bias,
        "mamba_conv1d_bias.npy",
        false,
    )?;
    load_opt(&mut state.ema_mamba_out_proj, "mamba_out_proj.npy", true)?;
    load_opt(&mut state.ema_mamba_a_log, "mamba_a_log.npy", false)?;
    load_opt(&mut state.ema_mamba_d, "mamba_d.npy", false)?;
    load_opt(&mut state.ema_mamba_dt_bias, "mamba_dt_bias.npy", false)?;
    load_opt(&mut state.ema_mamba_norm, "mamba_norm.npy", false)?;
    Ok(())
}

fn save_adam_slot(
    dir: &Path,
    stem: &str,
    slot: &AdamSlot,
    transpose: bool,
    step: f32,
) -> Result<(), String> {
    let avg = dir.join(format!("{stem}_exp_avg.npy"));
    let sq = dir.join(format!("{stem}_exp_avg_sq.npy"));
    let step_p = dir.join(format!("{stem}_step.npy"));
    let aux = dir.join(format!("{stem}_aux.npy"));
    let origin = dir.join(format!("{stem}_origin.npy"));
    if transpose {
        save_linear_python(&avg, &slot.exp_avg)?;
        save_linear_python(&sq, &slot.exp_avg_sq)?;
        save_linear_python(&aux, &slot.aux)?;
        save_linear_python(&origin, &slot.origin)?;
    } else {
        save_f32(&avg, &slot.exp_avg)?;
        save_f32(&sq, &slot.exp_avg_sq)?;
        save_f32(&aux, &slot.aux)?;
        save_f32(&origin, &slot.origin)?;
    }
    save_scalar_f32(&step_p, step)?;
    Ok(())
}

fn load_tensor_f32(path: &Path, dst: &Tensor, transpose: bool) -> Result<(), String> {
    let arr = crate::npy::read_npy(path)?;
    let mut data = arr.f32_slice()?.to_vec();
    let mut shape = arr.shape.clone();
    if transpose {
        transpose_last2(&mut data, &mut shape)?;
    }
    if shape != dst.shape() {
        return Err(format!(
            "checkpoint tensor {} shape {:?}, expected {:?}",
            path.display(),
            shape,
            dst.shape()
        ));
    }
    dst.buffer.write_f32(&data);
    Ok(())
}

fn load_adam_slot(dir: &Path, stem: &str, slot: &AdamSlot, transpose: bool) -> Result<(), String> {
    load_tensor_f32(
        &dir.join(format!("{stem}_exp_avg.npy")),
        &slot.exp_avg,
        transpose,
    )?;
    load_tensor_f32(
        &dir.join(format!("{stem}_exp_avg_sq.npy")),
        &slot.exp_avg_sq,
        transpose,
    )?;
    load_tensor_f32(&dir.join(format!("{stem}_aux.npy")), &slot.aux, transpose)?;
    load_tensor_f32(
        &dir.join(format!("{stem}_origin.npy")),
        &slot.origin,
        transpose,
    )
}

fn load_adam_slot_opt(
    dir: &Path,
    stem: &str,
    slot: &Option<AdamSlot>,
    transpose: bool,
) -> Result<(), String> {
    let Some(slot) = slot else {
        return Ok(());
    };
    let path = dir.join(format!("{stem}_exp_avg.npy"));
    if path.exists() {
        load_adam_slot(dir, stem, slot, transpose)?;
    }
    Ok(())
}

fn muon_bank_scale(rows: usize, cols: usize) -> f32 {
    ((cols as f32) / (rows as f32)).max(1.0).sqrt()
}

/// Write `dir/` in the same tree shape as `golden/weights_init`.
pub fn save_weights_python_npy(
    _rt: &Arc<GpuRuntime>,
    w: &Weights,
    dir: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;

    save_f32(&dir.join("tok_emb/weight.npy"), &w.tok_emb)?;
    save_f32(&dir.join("bigram/embed/weight.npy"), &w.bigram_emb)?;
    save_linear_python(&dir.join("bigram/proj/weight.npy"), &w.bigram_proj)?;
    save_f32(&dir.join("bigram/scale.npy"), &w.bigram_scale)?;
    save_f32(&dir.join("smear/gate.npy"), &w.smear_gate)?;

    save_f32(&dir.join("ve_shared/embed/weight.npy"), &w.ve_emb)?;
    save_linear_python(&dir.join("ve_shared/proj/weight.npy"), &w.ve_proj)?;
    save_f32(&dir.join("ve_shared/scale.npy"), &w.ve_scale)?;
    for (i, s) in w.ve_layer_scales.iter().enumerate() {
        save_f32(&dir.join(format!("ve_layer_scales/{i}.npy")), s)?;
    }

    save_f32(&dir.join("skip_weights.npy"), &w.skip_weights)?;
    save_linear_python(&dir.join("qo_bank.npy"), &w.qo_bank)?;
    save_linear_python(&dir.join("kv_bank.npy"), &w.kv_bank)?;
    if let Some(t) = &w.mingru_to_z {
        if "Matrix" == "Matrix" {
            save_linear_python(&dir.join("mingru_to_z.npy"), t)?;
        } else {
            save_f32(&dir.join("mingru_to_z.npy"), t)?;
        }
    }
    if let Some(t) = &w.mingru_to_h {
        if "Matrix" == "Matrix" {
            save_linear_python(&dir.join("mingru_to_h.npy"), t)?;
        } else {
            save_f32(&dir.join("mingru_to_h.npy"), t)?;
        }
    }
    if let Some(t) = &w.mingru_out {
        if "Matrix" == "Matrix" {
            save_linear_python(&dir.join("mingru_out.npy"), t)?;
        } else {
            save_f32(&dir.join("mingru_out.npy"), t)?;
        }
    }
    if let Some(t) = &w.mingru_v_proj {
        save_linear_python(&dir.join("mingru_v_proj.npy"), t)?;
    }
    if let Some(t) = &w.mingru_v0_up {
        save_linear_python(&dir.join("mingru_v0_up.npy"), t)?;
    }
    if let Some(t) = &w.mamba_in_proj {
        if "Matrix" == "Matrix" {
            save_linear_python(&dir.join("mamba_in_proj.npy"), t)?;
        } else {
            save_f32(&dir.join("mamba_in_proj.npy"), t)?;
        }
    }
    if let Some(t) = &w.mamba_conv1d_weight {
        if "Scalar" == "Matrix" {
            save_linear_python(&dir.join("mamba_conv1d_weight.npy"), t)?;
        } else {
            save_f32(&dir.join("mamba_conv1d_weight.npy"), t)?;
        }
    }
    if let Some(t) = &w.mamba_conv1d_bias {
        if "Scalar" == "Matrix" {
            save_linear_python(&dir.join("mamba_conv1d_bias.npy"), t)?;
        } else {
            save_f32(&dir.join("mamba_conv1d_bias.npy"), t)?;
        }
    }
    if let Some(t) = &w.mamba_out_proj {
        if "Matrix" == "Matrix" {
            save_linear_python(&dir.join("mamba_out_proj.npy"), t)?;
        } else {
            save_f32(&dir.join("mamba_out_proj.npy"), t)?;
        }
    }
    if let Some(t) = &w.mamba_a_log {
        if "Scalar" == "Matrix" {
            save_linear_python(&dir.join("mamba_a_log.npy"), t)?;
        } else {
            save_f32(&dir.join("mamba_a_log.npy"), t)?;
        }
    }
    if let Some(t) = &w.mamba_d {
        if "Scalar" == "Matrix" {
            save_linear_python(&dir.join("mamba_d.npy"), t)?;
        } else {
            save_f32(&dir.join("mamba_d.npy"), t)?;
        }
    }
    if let Some(t) = &w.mamba_dt_bias {
        if "Scalar" == "Matrix" {
            save_linear_python(&dir.join("mamba_dt_bias.npy"), t)?;
        } else {
            save_f32(&dir.join("mamba_dt_bias.npy"), t)?;
        }
    }
    if let Some(t) = &w.mamba_norm {
        save_f32(&dir.join("mamba_norm.npy"), t)?;
    }

    save_linear_python(&dir.join("mlp_up_bank.npy"), &w.mlp_up)?;
    save_linear_python(&dir.join("mlp_down_bank.npy"), &w.mlp_down)?;

    for (i, b) in w.blocks.iter().enumerate() {
        let base = dir.join(format!("blocks/{i}"));
        save_f32(&base.join("attn/q_gain.npy"), &b.q_gain)?;
        save_f32(&base.join("attn/vr_lambda.npy"), &b.vr_lambda)?;
        save_f32(&base.join("attn_scale.npy"), &b.attn_scale)?;
        save_f32(&base.join("mlp_scale.npy"), &b.mlp_scale)?;
        save_f32(&base.join("resid_mix.npy"), &b.resid_mix)?;
    }

    let meta = format!(
        "{{\n  \"source\": \"metal-native\",\n  \"layout\": \"python\",\n  \"num_layers\": {},\n  \"model_dim\": {},\n  \"vocab_size\": {}\n}}\n",
        w.cfg.num_layers, w.cfg.model_dim, w.cfg.vocab_size
    );
    std::fs::write(dir.join("manifest.json"), meta).map_err(|e| format!("manifest: {e}"))?;
    Ok(())
}

/// Dump Adam moments + Muon momentum (+ params) in the `optim_step3/` golden layout.
///
/// Tree mirrors `golden/optim_step3/`:
/// - `params/` — live weights (Python layout)
/// - `adamw_embed/` / `adamw_scalar/` — `*_exp_avg.npy`, `*_exp_avg_sq.npy`, `*_step.npy`
/// - `muon/` — `*_momentum_buffer.npy` (+ `*_scale.npy` meta)
pub fn save_optim_state_python_npy(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    state: &OptimState,
    dir: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;

    // Params subtree (same layout as weights_init).
    save_weights_python_npy(rt, w, &dir.join("params"))?;

    let step = state.step as f32;
    let embed = dir.join("adamw_embed");
    save_adam_slot(
        &embed.join("tok_emb"),
        "weight",
        &state.tok_emb,
        false,
        step,
    )?;
    save_adam_slot(
        &embed.join("bigram/embed"),
        "weight",
        &state.bigram_emb,
        false,
        step,
    )?;
    save_adam_slot(
        &embed.join("ve_shared/embed"),
        "weight",
        &state.ve_emb,
        false,
        step,
    )?;

    let scalar = dir.join("adamw_scalar");
    save_adam_slot(
        &scalar.join("bigram/proj"),
        "weight",
        &state.bigram_proj,
        true,
        step,
    )?;
    save_adam_slot(
        &scalar.join("bigram"),
        "scale",
        &state.bigram_scale,
        false,
        step,
    )?;
    save_adam_slot(
        &scalar.join("smear"),
        "gate",
        &state.smear_gate,
        false,
        step,
    )?;
    save_adam_slot(
        &scalar.join("ve_shared/proj"),
        "weight",
        &state.ve_proj,
        true,
        step,
    )?;
    save_adam_slot(
        &scalar.join("ve_shared"),
        "scale",
        &state.ve_scale,
        false,
        step,
    )?;
    for (i, slot) in state.ve_layer_scales.iter().enumerate() {
        // Golden naming: ve_layer_scales/{i}_exp_avg.npy (stem includes index).
        save_adam_slot(
            &scalar.join("ve_layer_scales"),
            &format!("{i}"),
            slot,
            false,
            step,
        )?;
    }
    save_adam_slot(&scalar, "skip_weights", &state.skip_weights, false, step)?;

    for (i, b) in state.blocks.iter().enumerate() {
        let base = scalar.join(format!("blocks/{i}"));
        save_adam_slot(&base.join("attn"), "q_gain", &b.q_gain, false, step)?;
        save_adam_slot(&base.join("attn"), "vr_lambda", &b.vr_lambda, false, step)?;
        save_adam_slot(&base, "attn_scale", &b.attn_scale, false, step)?;
        save_adam_slot(&base, "mlp_scale", &b.mlp_scale, false, step)?;
        save_adam_slot(&base, "resid_mix", &b.resid_mix, false, step)?;
    }

    let save_mamba_adam =
        |stem: &str, slot: &Option<AdamSlot>, transpose: bool| -> Result<(), String> {
            if let Some(s) = slot {
                save_adam_slot(&scalar, stem, s, transpose, step)?;
            }
            Ok(())
        };
    save_mamba_adam("mamba_conv1d_weight", &state.mamba_conv1d_weight, false)?;
    save_mamba_adam("mamba_conv1d_bias", &state.mamba_conv1d_bias, false)?;
    save_mamba_adam("mamba_a_log", &state.mamba_a_log, false)?;
    save_mamba_adam("mamba_d", &state.mamba_d, false)?;
    save_mamba_adam("mamba_dt_bias", &state.mamba_dt_bias, false)?;
    save_mamba_adam("mamba_norm", &state.mamba_norm, false)?;

    let muon = dir.join("muon");
    save_linear_python(&muon.join("qo_bank_momentum_buffer.npy"), &state.mom_qo)?;
    save_linear_python(&muon.join("kv_bank_momentum_buffer.npy"), &state.mom_kv)?;
    save_linear_python(&muon.join("mlp_up_bank_momentum_buffer.npy"), &state.mom_up)?;
    save_linear_python(
        &muon.join("mlp_down_bank_momentum_buffer.npy"),
        &state.mom_dn,
    )?;
    save_linear_python(&muon.join("qo_bank_aux_state.npy"), &state.var_qo)?;
    save_linear_python(&muon.join("kv_bank_aux_state.npy"), &state.var_kv)?;
    save_linear_python(&muon.join("mlp_up_bank_aux_state.npy"), &state.var_up)?;
    save_linear_python(&muon.join("mlp_down_bank_aux_state.npy"), &state.var_dn)?;
    save_linear_python(&muon.join("qo_bank_prev_state.npy"), &state.prev_qo)?;
    save_linear_python(&muon.join("kv_bank_prev_state.npy"), &state.prev_kv)?;
    save_linear_python(&muon.join("mlp_up_bank_prev_state.npy"), &state.prev_up)?;
    save_linear_python(&muon.join("mlp_down_bank_prev_state.npy"), &state.prev_dn)?;
    save_linear_python(&muon.join("qo_bank_mag_v.npy"), &state.mag_v_qo)?;
    save_linear_python(&muon.join("kv_bank_mag_v.npy"), &state.mag_v_kv)?;
    save_linear_python(&muon.join("mlp_up_bank_mag_v.npy"), &state.mag_v_up)?;
    save_linear_python(&muon.join("mlp_down_bank_mag_v.npy"), &state.mag_v_dn)?;
    save_scalar_f32(&muon.join("prodigy_d.npy"), state.prodigy_d)?;
    save_scalar_f32(&muon.join("prodigy_d_max.npy"), state.prodigy_d_max)?;
    save_scalar_f32(
        &muon.join("prodigy_d_numerator.npy"),
        state.prodigy_d_numerator,
    )?;

    let c = w.cfg.model_dim;
    let kv = w.cfg.kv_dim();
    let mlp = w.cfg.mlp_dim;
    // scale = sqrt(max(1, out/in)) with metal-native [in, out] → out=last dim.
    save_scalar_f32(&muon.join("qo_bank_scale.npy"), muon_bank_scale(c, c))?;
    save_scalar_f32(&muon.join("kv_bank_scale.npy"), muon_bank_scale(c, kv))?;
    save_scalar_f32(&muon.join("mlp_up_bank_scale.npy"), muon_bank_scale(c, mlp))?;
    save_scalar_f32(
        &muon.join("mlp_down_bank_scale.npy"),
        muon_bank_scale(mlp, c),
    )?;

    save_muon_bank_opt(
        &muon,
        "mingru_to_z",
        state.mom_mingru_to_z.as_ref(),
        state.var_mingru_to_z.as_ref(),
        state.prev_mingru_to_z.as_ref(),
        state.mag_v_mingru_to_z.as_ref(),
    )?;
    save_muon_bank_opt(
        &muon,
        "mingru_to_h",
        state.mom_mingru_to_h.as_ref(),
        state.var_mingru_to_h.as_ref(),
        state.prev_mingru_to_h.as_ref(),
        state.mag_v_mingru_to_h.as_ref(),
    )?;
    save_muon_bank_opt(
        &muon,
        "mingru_out",
        state.mom_mingru_out.as_ref(),
        state.var_mingru_out.as_ref(),
        state.prev_mingru_out.as_ref(),
        state.mag_v_mingru_out.as_ref(),
    )?;
    save_muon_bank_opt(
        &muon,
        "mingru_v_proj",
        state.mom_mingru_v_proj.as_ref(),
        state.var_mingru_v_proj.as_ref(),
        state.prev_mingru_v_proj.as_ref(),
        state.mag_v_mingru_v_proj.as_ref(),
    )?;
    save_muon_bank_opt(
        &muon,
        "mingru_v0_up",
        state.mom_mingru_v0_up.as_ref(),
        state.var_mingru_v0_up.as_ref(),
        state.prev_mingru_v0_up.as_ref(),
        state.mag_v_mingru_v0_up.as_ref(),
    )?;
    save_muon_bank_opt(
        &muon,
        "mamba_in_proj",
        state.mom_mamba_in_proj.as_ref(),
        state.var_mamba_in_proj.as_ref(),
        state.prev_mamba_in_proj.as_ref(),
        state.mag_v_mamba_in_proj.as_ref(),
    )?;
    save_muon_bank_opt(
        &muon,
        "mamba_out_proj",
        state.mom_mamba_out_proj.as_ref(),
        state.var_mamba_out_proj.as_ref(),
        state.prev_mamba_out_proj.as_ref(),
        state.mag_v_mamba_out_proj.as_ref(),
    )?;

    let meta = format!(
        "{{\n  \"source\": \"metal-native\",\n  \"layout\": \"optim_step3\",\n  \"optim_step\": {},\n  \"num_layers\": {},\n  \"model_dim\": {},\n  \"vocab_size\": {}\n}}\n",
        state.step, w.cfg.num_layers, w.cfg.model_dim, w.cfg.vocab_size
    );
    std::fs::write(dir.join("manifest.json"), meta).map_err(|e| format!("manifest: {e}"))?;
    Ok(())
}

/// Save all EMA tensors in the same logical tree as model weights.
pub fn save_ema_state_python_npy(state: &OptimState, dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    save_f32(&dir.join("tok_emb/weight.npy"), &state.ema_tok_emb)?;
    save_f32(&dir.join("bigram/embed/weight.npy"), &state.ema_bigram_emb)?;
    save_linear_python(&dir.join("bigram/proj/weight.npy"), &state.ema_bigram_proj)?;
    save_f32(&dir.join("bigram/scale.npy"), &state.ema_bigram_scale)?;
    save_f32(&dir.join("smear/gate.npy"), &state.ema_smear_gate)?;
    save_f32(&dir.join("ve_shared/embed/weight.npy"), &state.ema_ve_emb)?;
    save_linear_python(&dir.join("ve_shared/proj/weight.npy"), &state.ema_ve_proj)?;
    save_f32(&dir.join("ve_shared/scale.npy"), &state.ema_ve_scale)?;
    for (i, t) in state.ema_ve_layer_scales.iter().enumerate() {
        save_f32(&dir.join(format!("ve_layer_scales/{i}.npy")), t)?;
    }
    save_f32(&dir.join("skip_weights.npy"), &state.ema_skip_weights)?;
    save_linear_python(&dir.join("qo_bank.npy"), &state.ema_qo)?;
    save_linear_python(&dir.join("kv_bank.npy"), &state.ema_kv)?;

    save_linear_python(&dir.join("mlp_up_bank.npy"), &state.ema_up)?;
    save_linear_python(&dir.join("mlp_down_bank.npy"), &state.ema_dn)?;
    save_ssm_ema_banks(state, dir)?;
    for (i, b) in state.ema_blocks.iter().enumerate() {
        let base = dir.join(format!("blocks/{i}"));
        save_f32(&base.join("attn/q_gain.npy"), &b.q_gain)?;
        save_f32(&base.join("attn/vr_lambda.npy"), &b.vr_lambda)?;
        save_f32(&base.join("attn_scale.npy"), &b.attn_scale)?;
        save_f32(&base.join("mlp_scale.npy"), &b.mlp_scale)?;
        save_f32(&base.join("resid_mix.npy"), &b.resid_mix)?;
    }
    Ok(())
}

pub fn load_ema_state_python_npy(state: &mut OptimState, dir: &Path) -> Result<(), String> {
    load_tensor_f32(&dir.join("tok_emb/weight.npy"), &state.ema_tok_emb, false)?;
    load_tensor_f32(
        &dir.join("bigram/embed/weight.npy"),
        &state.ema_bigram_emb,
        false,
    )?;
    load_tensor_f32(
        &dir.join("bigram/proj/weight.npy"),
        &state.ema_bigram_proj,
        true,
    )?;
    load_tensor_f32(
        &dir.join("bigram/scale.npy"),
        &state.ema_bigram_scale,
        false,
    )?;
    load_tensor_f32(&dir.join("smear/gate.npy"), &state.ema_smear_gate, false)?;
    load_tensor_f32(
        &dir.join("ve_shared/embed/weight.npy"),
        &state.ema_ve_emb,
        false,
    )?;
    load_tensor_f32(
        &dir.join("ve_shared/proj/weight.npy"),
        &state.ema_ve_proj,
        true,
    )?;
    load_tensor_f32(&dir.join("ve_shared/scale.npy"), &state.ema_ve_scale, false)?;
    for (i, t) in state.ema_ve_layer_scales.iter().enumerate() {
        load_tensor_f32(&dir.join(format!("ve_layer_scales/{i}.npy")), t, false)?;
    }
    load_tensor_f32(
        &dir.join("skip_weights.npy"),
        &state.ema_skip_weights,
        false,
    )?;
    load_tensor_f32(&dir.join("qo_bank.npy"), &state.ema_qo, true)?;
    load_tensor_f32(&dir.join("kv_bank.npy"), &state.ema_kv, true)?;

    load_tensor_f32(&dir.join("mlp_up_bank.npy"), &state.ema_up, true)?;
    load_tensor_f32(&dir.join("mlp_down_bank.npy"), &state.ema_dn, true)?;
    load_ssm_ema_banks(state, dir)?;
    for (i, b) in state.ema_blocks.iter().enumerate() {
        let base = dir.join(format!("blocks/{i}"));
        load_tensor_f32(&base.join("attn/q_gain.npy"), &b.q_gain, false)?;
        load_tensor_f32(&base.join("attn/vr_lambda.npy"), &b.vr_lambda, false)?;
        load_tensor_f32(&base.join("attn_scale.npy"), &b.attn_scale, false)?;
        load_tensor_f32(&base.join("mlp_scale.npy"), &b.mlp_scale, false)?;
        load_tensor_f32(&base.join("resid_mix.npy"), &b.resid_mix, false)?;
    }
    Ok(())
}

/// Restore every Adam moment, Muon momentum bank, EMA tensor, and optimizer step.
pub fn load_optim_state_python_npy(
    state: &mut OptimState,
    optim_dir: &Path,
    ema_dir: &Path,
) -> Result<(), String> {
    let embed = optim_dir.join("adamw_embed");
    load_adam_slot(&embed.join("tok_emb"), "weight", &state.tok_emb, false)?;
    load_adam_slot(
        &embed.join("bigram/embed"),
        "weight",
        &state.bigram_emb,
        false,
    )?;
    load_adam_slot(
        &embed.join("ve_shared/embed"),
        "weight",
        &state.ve_emb,
        false,
    )?;

    let scalar = optim_dir.join("adamw_scalar");
    load_adam_slot(
        &scalar.join("bigram/proj"),
        "weight",
        &state.bigram_proj,
        true,
    )?;
    load_adam_slot(&scalar.join("bigram"), "scale", &state.bigram_scale, false)?;
    load_adam_slot(&scalar.join("smear"), "gate", &state.smear_gate, false)?;
    load_adam_slot(
        &scalar.join("ve_shared/proj"),
        "weight",
        &state.ve_proj,
        true,
    )?;
    load_adam_slot(&scalar.join("ve_shared"), "scale", &state.ve_scale, false)?;
    for (i, slot) in state.ve_layer_scales.iter().enumerate() {
        load_adam_slot(
            &scalar.join("ve_layer_scales"),
            &format!("{i}"),
            slot,
            false,
        )?;
    }
    load_adam_slot(&scalar, "skip_weights", &state.skip_weights, false)?;
    for (i, b) in state.blocks.iter().enumerate() {
        let base = scalar.join(format!("blocks/{i}"));
        load_adam_slot(&base.join("attn"), "q_gain", &b.q_gain, false)?;
        load_adam_slot(&base.join("attn"), "vr_lambda", &b.vr_lambda, false)?;
        load_adam_slot(&base, "attn_scale", &b.attn_scale, false)?;
        load_adam_slot(&base, "mlp_scale", &b.mlp_scale, false)?;
        load_adam_slot(&base, "resid_mix", &b.resid_mix, false)?;
    }
    load_adam_slot_opt(
        &scalar,
        "mamba_conv1d_weight",
        &state.mamba_conv1d_weight,
        false,
    )?;
    load_adam_slot_opt(
        &scalar,
        "mamba_conv1d_bias",
        &state.mamba_conv1d_bias,
        false,
    )?;
    load_adam_slot_opt(&scalar, "mamba_a_log", &state.mamba_a_log, false)?;
    load_adam_slot_opt(&scalar, "mamba_d", &state.mamba_d, false)?;
    load_adam_slot_opt(&scalar, "mamba_dt_bias", &state.mamba_dt_bias, false)?;
    load_adam_slot_opt(&scalar, "mamba_norm", &state.mamba_norm, false)?;
    load_muon_momentum_python_npy(&Arc::clone(state.mom_qo.runtime()), state, optim_dir)?;
    load_ema_state_python_npy(state, ema_dir)?;

    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(optim_dir.join("manifest.json"))
            .map_err(|e| format!("read optim manifest: {e}"))?,
    )
    .map_err(|e| format!("parse optim manifest: {e}"))?;
    state.step = manifest
        .get("optim_step")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "optim manifest missing optim_step".to_string())? as usize;
    Ok(())
}

pub fn save_training_checkpoint(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    state: &OptimState,
    root: &Path,
    meta: &TrainingCheckpointMeta,
) -> Result<(), String> {
    rt.synchronize()?;
    std::fs::create_dir_all(root).map_err(|e| format!("mkdir {}: {e}", root.display()))?;
    save_weights_python_npy(rt, w, &root.join("weights"))?;
    let shadows_saved = save_bf16_shadows(w, &root.join("bf16_shadows"))?;
    if shadows_saved != meta.bf16_shadows_saved {
        return Err(format!(
            "checkpoint metadata says bf16_shadows_saved={}, actual={shadows_saved}",
            meta.bf16_shadows_saved
        ));
    }
    save_optim_state_python_npy(rt, w, state, &root.join("optim"))?;
    save_ema_state_python_npy(state, &root.join("ema"))?;
    let json = serde_json::to_vec_pretty(meta).map_err(|e| format!("checkpoint meta: {e}"))?;
    std::fs::write(root.join("checkpoint.json"), json)
        .map_err(|e| format!("write checkpoint meta: {e}"))?;
    Ok(())
}

pub fn read_training_checkpoint_meta(root: &Path) -> Result<TrainingCheckpointMeta, String> {
    let bytes = std::fs::read(root.join("checkpoint.json"))
        .map_err(|e| format!("read checkpoint meta: {e}"))?;
    let meta: TrainingCheckpointMeta =
        serde_json::from_slice(&bytes).map_err(|e| format!("parse checkpoint meta: {e}"))?;
    if meta.version != CHECKPOINT_VERSION {
        return Err(format!(
            "checkpoint version {} unsupported (expected {})",
            meta.version, CHECKPOINT_VERSION
        ));
    }
    Ok(meta)
}

/// Load Muon momentum banks from a `dump_step*/optim/muon/` tree (Python `[N,out,in]`).
/// Adam moments stay fresh — enough to remove the zero-momentum continue confound.
pub fn load_muon_momentum_python_npy(
    _rt: &Arc<GpuRuntime>,
    state: &mut OptimState,
    optim_dir: &Path,
) -> Result<(), String> {
    let muon = optim_dir.join("muon");
    let load = |dst: &Tensor, name: &str| -> Result<(), String> {
        let path = muon.join(name);
        let arr = crate::npy::read_npy(&path)?;
        let mut data = arr.f32_slice()?.to_vec();
        let mut sh = arr.shape.clone();
        crate::npy::transpose_last2(&mut data, &mut sh)?;
        if sh != dst.shape() {
            return Err(format!(
                "muon {name}: shape {:?} vs dst {:?}",
                sh, dst.shape()
            ));
        }
        dst.buffer.write_f32(&data);
        Ok(())
    };
    load(&state.mom_qo, "qo_bank_momentum_buffer.npy")?;
    load(&state.mom_kv, "kv_bank_momentum_buffer.npy")?;
    load(&state.mom_up, "mlp_up_bank_momentum_buffer.npy")?;
    load(&state.mom_dn, "mlp_down_bank_momentum_buffer.npy")?;
    load(&state.var_qo, "qo_bank_aux_state.npy")?;
    load(&state.var_kv, "kv_bank_aux_state.npy")?;
    load(&state.var_up, "mlp_up_bank_aux_state.npy")?;
    load(&state.var_dn, "mlp_down_bank_aux_state.npy")?;
    load(&state.prev_qo, "qo_bank_prev_state.npy")?;
    load(&state.prev_kv, "kv_bank_prev_state.npy")?;
    load(&state.prev_up, "mlp_up_bank_prev_state.npy")?;
    load(&state.prev_dn, "mlp_down_bank_prev_state.npy")?;
    load(&state.mag_v_qo, "qo_bank_mag_v.npy")?;
    load(&state.mag_v_kv, "kv_bank_mag_v.npy")?;
    load(&state.mag_v_up, "mlp_up_bank_mag_v.npy")?;
    load(&state.mag_v_dn, "mlp_down_bank_mag_v.npy")?;
    let load_opt = |dst: Option<&Tensor>, name: &str| -> Result<(), String> {
        let Some(dst) = dst else {
            return Ok(());
        };
        let path = muon.join(name);
        if !path.exists() {
            return Ok(());
        }
        load(dst, name)
    };
    load_opt(
        state.mom_mingru_to_z.as_ref(),
        "mingru_to_z_momentum_buffer.npy",
    )?;
    load_opt(state.var_mingru_to_z.as_ref(), "mingru_to_z_aux_state.npy")?;
    load_opt(
        state.prev_mingru_to_z.as_ref(),
        "mingru_to_z_prev_state.npy",
    )?;
    load_opt(state.mag_v_mingru_to_z.as_ref(), "mingru_to_z_mag_v.npy")?;
    load_opt(
        state.mom_mingru_to_h.as_ref(),
        "mingru_to_h_momentum_buffer.npy",
    )?;
    load_opt(state.var_mingru_to_h.as_ref(), "mingru_to_h_aux_state.npy")?;
    load_opt(
        state.prev_mingru_to_h.as_ref(),
        "mingru_to_h_prev_state.npy",
    )?;
    load_opt(state.mag_v_mingru_to_h.as_ref(), "mingru_to_h_mag_v.npy")?;
    load_opt(
        state.mom_mingru_out.as_ref(),
        "mingru_out_momentum_buffer.npy",
    )?;
    load_opt(state.var_mingru_out.as_ref(), "mingru_out_aux_state.npy")?;
    load_opt(state.prev_mingru_out.as_ref(), "mingru_out_prev_state.npy")?;
    load_opt(state.mag_v_mingru_out.as_ref(), "mingru_out_mag_v.npy")?;
    load_opt(
        state.mom_mingru_v_proj.as_ref(),
        "mingru_v_proj_momentum_buffer.npy",
    )?;
    load_opt(
        state.var_mingru_v_proj.as_ref(),
        "mingru_v_proj_aux_state.npy",
    )?;
    load_opt(
        state.prev_mingru_v_proj.as_ref(),
        "mingru_v_proj_prev_state.npy",
    )?;
    load_opt(
        state.mag_v_mingru_v_proj.as_ref(),
        "mingru_v_proj_mag_v.npy",
    )?;
    load_opt(
        state.mom_mingru_v0_up.as_ref(),
        "mingru_v0_up_momentum_buffer.npy",
    )?;
    load_opt(
        state.var_mingru_v0_up.as_ref(),
        "mingru_v0_up_aux_state.npy",
    )?;
    load_opt(
        state.prev_mingru_v0_up.as_ref(),
        "mingru_v0_up_prev_state.npy",
    )?;
    load_opt(state.mag_v_mingru_v0_up.as_ref(), "mingru_v0_up_mag_v.npy")?;
    load_opt(
        state.mom_mamba_in_proj.as_ref(),
        "mamba_in_proj_momentum_buffer.npy",
    )?;
    load_opt(
        state.var_mamba_in_proj.as_ref(),
        "mamba_in_proj_aux_state.npy",
    )?;
    load_opt(
        state.prev_mamba_in_proj.as_ref(),
        "mamba_in_proj_prev_state.npy",
    )?;
    load_opt(
        state.mag_v_mamba_in_proj.as_ref(),
        "mamba_in_proj_mag_v.npy",
    )?;
    load_opt(
        state.mom_mamba_out_proj.as_ref(),
        "mamba_out_proj_momentum_buffer.npy",
    )?;
    load_opt(
        state.var_mamba_out_proj.as_ref(),
        "mamba_out_proj_aux_state.npy",
    )?;
    load_opt(
        state.prev_mamba_out_proj.as_ref(),
        "mamba_out_proj_prev_state.npy",
    )?;
    load_opt(
        state.mag_v_mamba_out_proj.as_ref(),
        "mamba_out_proj_mag_v.npy",
    )?;
    let scalar = |name: &str| -> Result<f32, String> {
        let arr = crate::npy::read_npy(&muon.join(name))?;
        arr.f32_slice()?
            .first()
            .copied()
            .ok_or_else(|| format!("empty optimizer scalar {name}"))
    };
    state.prodigy_d = scalar("prodigy_d.npy")?;
    state.prodigy_d_max = scalar("prodigy_d_max.npy")?;
    state.prodigy_d_numerator = scalar("prodigy_d_numerator.npy")?;
    Ok(())
}

/// L2 / Frobenius norm of a device tensor (host readback).
pub fn tensor_l2_norm(t: &Tensor) -> f64 {
    let data = t.buffer.read_f32();
    let mut s = 0.0f64;
    for &x in &data {
        let v = x as f64;
        s += v * v;
    }
    s.sqrt()
}

/// Scalar / bank norms for late-run divergence bisect (Phase 0).
#[derive(Debug, Clone, serde::Serialize)]
pub struct DivergenceNorms {
    pub resid_mix: Vec<f64>,
    pub vr_lambda: Vec<f64>,
    pub attn_scale: Vec<f64>,
    pub smear_gate: f64,
    pub bank_qo: f64,
    pub bank_kv: f64,
    pub bank_mlp_up: f64,
    pub bank_mlp_down: f64,
    pub mom_qo: f64,
    pub mom_kv: f64,
    pub mom_mlp_up: f64,
    pub mom_mlp_down: f64,
}

/// Collect weight + Muon-momentum norms after a synchronized optim step.
pub fn collect_divergence_norms(w: &Weights, state: &OptimState) -> DivergenceNorms {
    let mut resid_mix = Vec::with_capacity(w.blocks.len());
    let mut vr_lambda = Vec::with_capacity(w.blocks.len());
    let mut attn_scale = Vec::with_capacity(w.blocks.len());
    for b in &w.blocks {
        resid_mix.push(tensor_l2_norm(&b.resid_mix));
        vr_lambda.push(tensor_l2_norm(&b.vr_lambda));
        attn_scale.push(tensor_l2_norm(&b.attn_scale));
    }
    DivergenceNorms {
        resid_mix,
        vr_lambda,
        attn_scale,
        smear_gate: tensor_l2_norm(&w.smear_gate),
        bank_qo: tensor_l2_norm(&w.qo_bank),
        bank_kv: tensor_l2_norm(&w.kv_bank),
        bank_mlp_up: tensor_l2_norm(&w.mlp_up),
        bank_mlp_down: tensor_l2_norm(&w.mlp_down),
        mom_qo: tensor_l2_norm(&state.mom_qo),
        mom_kv: tensor_l2_norm(&state.mom_kv),
        mom_mlp_up: tensor_l2_norm(&state.mom_up),
        mom_mlp_down: tensor_l2_norm(&state.mom_dn),
    }
}

/// GPU-reduced divergence telemetry.  Only one f32 scalar per tensor crosses
/// to the host, replacing the former multi-GiB bank readback at 128M scale.
pub fn collect_divergence_norms_device(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    state: &OptimState,
) -> Result<DivergenceNorms, String> {
    let mut refs: Vec<&Tensor> = Vec::new();
    for b in &w.blocks {
        refs.push(&b.resid_mix);
    }
    for b in &w.blocks {
        refs.push(&b.vr_lambda);
    }
    for b in &w.blocks {
        refs.push(&b.attn_scale);
    }
    refs.extend([
        &w.smear_gate,
        &w.qo_bank,
        &w.kv_bank,
        &w.mlp_up,
        &w.mlp_down,
        &state.mom_qo,
        &state.mom_kv,
        &state.mom_up,
        &state.mom_dn,
    ]);

    let mut sums: Vec<GpuBuffer> = Vec::with_capacity(refs.len());
    for _ in &refs {
        let b = rt.alloc_buffer(4)?;
        b.zero();
        sums.push(b);
    }
    let pipe = rt.pipeline("grad_sq_reduce_f32")?;
    let width = pipe.threadExecutionWidth() as usize;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&pipe);
        for (t, sum) in refs.iter().zip(sums.iter()) {
            let n = t.numel();
            let tpt = width.min(n).max(1);
            let groups = (n + tpt - 1) / tpt;
            set_tensor(bnd, t, 0);
            set_gpu_buf(bnd, sum, 1);
            set_u32(bnd, n as u32, 2);
            bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
        }
        Ok(())
    })?;
    rt.synchronize()?;
    let values: Vec<f64> = sums
        .iter()
        .map(|b| (b.contents_f32()[0] as f64).sqrt())
        .collect();
    let n = w.blocks.len();
    let resid_mix = values[0..n].to_vec();
    let vr_lambda = values[n..2 * n].to_vec();
    let attn_scale = values[2 * n..3 * n].to_vec();
    let tail = &values[3 * n..];
    Ok(DivergenceNorms {
        resid_mix,
        vr_lambda,
        attn_scale,
        smear_gate: tail[0],
        bank_qo: tail[1],
        bank_kv: tail[2],
        bank_mlp_up: tail[3],
        bank_mlp_down: tail[4],
        mom_qo: tail[5],
        mom_kv: tail[6],
        mom_mlp_up: tail[7],
        mom_mlp_down: tail[8],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init::init_weights_seeded;
    use crate::optim::OptimState;
    use crate::weights::{expand_layer_mixers, MixerKind, ModelConfig};

    #[test]
    fn bf16_shadow_io_preserves_logical_extent_bits() {
        let rt = crate::gpu_runtime().expect("gpu");
        let tensor = rt.alloc_tensor_bf16(&[600_001]).expect("bf16 tensor");
        tensor.buffer.contents_u16()[0] = 0x3f80;
        tensor.buffer.contents_u16()[600_000] = 0xbf80;
        let path = std::env::temp_dir().join(format!(
            "arch02_bf16_shadow_{}_{}.bf16le",
            std::process::id(),
            std::thread::current().name().unwrap_or("worker")
        ));
        let _ = std::fs::remove_file(&path);
        save_bf16_bits(&path, &tensor).expect("save logical bf16");
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 600_001 * 2);
        tensor.buffer.contents_u16()[0] = 0;
        tensor.buffer.contents_u16()[600_000] = 0;
        load_bf16_bits(&path, &tensor).expect("load logical bf16");
        assert_eq!(tensor.buffer.contents_u16()[0], 0x3f80);
        assert_eq!(tensor.buffer.contents_u16()[600_000], 0xbf80);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn full_checkpoint_round_trip_restores_adam_muon_ema_and_cursor() {
        let rt = crate::gpu_runtime().expect("gpu");
        let cfg = ModelConfig::sota_toy();
        let mut w = init_weights_seeded(&rt, cfg.clone(), 42).expect("init");
        w.ensure_bf16_banks(&rt).expect("bf16 shadows");
        rt.synchronize().expect("bf16 cast");
        let hp = OptimHyperparams::default();
        let mut state = OptimState::new(&rt, &w, hp.clone()).expect("state");
        state.step = 7;
        state.tok_emb.exp_avg.buffer.contents_f32()[3] = 1.25;
        state.bigram_proj.exp_avg_sq.buffer.contents_f32()[5] = 2.5;
        state.mom_qo.buffer.contents_f32()[11] = -3.0;
        state.ema_up.buffer.contents_f32()[17] = 4.0;
        w.tok_emb.buffer.contents_f32()[0] = 0.75;
        let shadow_bit = 0x3f81;
        w.bf16_banks.as_ref().unwrap().qo_bank.buffer.contents_u16()[7] = shadow_bit;

        let root = std::env::temp_dir().join(format!(
            "arch02_checkpoint_test_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("worker")
        ));
        let _ = std::fs::remove_dir_all(&root);
        let meta = TrainingCheckpointMeta {
            version: CHECKPOINT_VERSION,
            step: 7,
            data_cursor_tokens: 123_456,
            seed: 42,
            preset: "sota".into(),
            config: cfg.clone(),
            parameter_count: cfg.count_params(),
            optimizer: "muon_ns5_adamw".into(),
            hyperparams: hp.clone(),
            clip_mode: ClipMode::Soft,
            schedule: LrSchedule::from_warmdown(2_000, 350),
            bf16_precision: true,
            bf16_shadows_saved: true,
        };
        save_training_checkpoint(&rt, &w, &state, &root, &meta).expect("save");

        let loaded_meta = read_training_checkpoint_meta(&root).expect("meta");
        assert_eq!(loaded_meta, meta);
        let mut w2 =
            Weights::load_from_python_npy(&rt, &root.join("weights"), cfg).expect("weights");
        w2.ensure_bf16_banks(&rt).expect("loaded bf16 shadows");
        rt.synchronize().expect("loaded bf16 cast");
        load_bf16_shadows(&mut w2, &root.join("bf16_shadows")).expect("load bf16 bits");
        let mut state2 = OptimState::new(&rt, &w2, hp).expect("state2");
        load_optim_state_python_npy(&mut state2, &root.join("optim"), &root.join("ema"))
            .expect("load full state");
        assert_eq!(state2.step, 7);
        assert_eq!(w2.tok_emb.buffer.contents_f32()[0], 0.75);
        assert_eq!(state2.tok_emb.exp_avg.buffer.contents_f32()[3], 1.25);
        assert_eq!(state2.bigram_proj.exp_avg_sq.buffer.contents_f32()[5], 2.5);
        assert_eq!(state2.mom_qo.buffer.contents_f32()[11], -3.0);
        assert_eq!(state2.ema_up.buffer.contents_f32()[17], 4.0);
        assert_eq!(
            w2.bf16_banks
                .as_ref()
                .unwrap()
                .qo_bank
                .buffer
                .contents_u16()[7],
            shadow_bit
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn mamba_adam_slots_round_trip_through_checkpoint() {
        let rt = crate::gpu_runtime().expect("gpu");
        let mut cfg = ModelConfig::sota_toy();
        cfg.layer_mixers = expand_layer_mixers(&[MixerKind::Mamba2], cfg.num_layers);
        let w = init_weights_seeded(&rt, cfg.clone(), 42).expect("init");
        let hp = OptimHyperparams::default();
        let mut state = OptimState::new(&rt, &w, hp.clone()).expect("state");
        state.step = 3;
        state
            .mamba_conv1d_weight
            .as_ref()
            .unwrap()
            .exp_avg
            .buffer
            .contents_f32()[0] = 0.42;
        state
            .mamba_conv1d_bias
            .as_ref()
            .unwrap()
            .exp_avg_sq
            .buffer
            .contents_f32()[2] = 1.75;
        state
            .mamba_a_log
            .as_ref()
            .unwrap()
            .exp_avg
            .buffer
            .contents_f32()[0] = -0.5;
        state.mamba_d.as_ref().unwrap().aux.buffer.contents_f32()[3] = 2.25;
        state
            .mamba_dt_bias
            .as_ref()
            .unwrap()
            .origin
            .buffer
            .contents_f32()[1] = -1.5;
        state
            .mamba_norm
            .as_ref()
            .unwrap()
            .exp_avg_sq
            .buffer
            .contents_f32()[7] = 3.5;

        let root = std::env::temp_dir().join(format!(
            "arch02_mamba_adam_checkpoint_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("worker")
        ));
        let _ = std::fs::remove_dir_all(&root);
        let meta = TrainingCheckpointMeta {
            version: CHECKPOINT_VERSION,
            step: 3,
            data_cursor_tokens: 0,
            seed: 42,
            preset: "sota".into(),
            config: cfg.clone(),
            parameter_count: cfg.count_params(),
            optimizer: "muon_ns5_adamw".into(),
            hyperparams: hp.clone(),
            clip_mode: ClipMode::Soft,
            schedule: LrSchedule::from_warmdown(2_000, 350),
            bf16_precision: false,
            bf16_shadows_saved: false,
        };
        save_training_checkpoint(&rt, &w, &state, &root, &meta).expect("save");

        let w2 = Weights::load_from_python_npy(&rt, &root.join("weights"), cfg).expect("weights");
        let mut state2 = OptimState::new(&rt, &w2, hp).expect("state2");
        load_optim_state_python_npy(&mut state2, &root.join("optim"), &root.join("ema"))
            .expect("load full state");
        assert_eq!(state2.step, 3);
        assert_eq!(
            state2
                .mamba_conv1d_weight
                .as_ref()
                .unwrap()
                .exp_avg
                .buffer
                .contents_f32()[0],
            0.42
        );
        assert_eq!(
            state2
                .mamba_conv1d_bias
                .as_ref()
                .unwrap()
                .exp_avg_sq
                .buffer
                .contents_f32()[2],
            1.75
        );
        assert_eq!(
            state2
                .mamba_a_log
                .as_ref()
                .unwrap()
                .exp_avg
                .buffer
                .contents_f32()[0],
            -0.5
        );
        assert_eq!(
            state2.mamba_d.as_ref().unwrap().aux.buffer.contents_f32()[3],
            2.25
        );
        assert_eq!(
            state2
                .mamba_dt_bias
                .as_ref()
                .unwrap()
                .origin
                .buffer
                .contents_f32()[1],
            -1.5
        );
        assert_eq!(
            state2
                .mamba_norm
                .as_ref()
                .unwrap()
                .exp_avg_sq
                .buffer
                .contents_f32()[7],
            3.5
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn mamba_conv1d_weight_init_save_load_layout_matches() {
        let rt = crate::gpu_runtime().expect("gpu");
        let mut cfg = ModelConfig::sota_toy();
        cfg.layer_mixers = expand_layer_mixers(&[MixerKind::Mamba2], cfg.num_layers);
        let w = init_weights_seeded(&rt, cfg.clone(), 42).expect("init");
        let orig = w.mamba_conv1d_weight.as_ref().expect("conv weight");
        let orig_data = orig.buffer.read_f32();
        let expected_shape = orig.shape().to_vec();

        let dir = std::env::temp_dir().join(format!(
            "arch02_mamba_conv_layout_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("worker")
        ));
        let _ = std::fs::remove_dir_all(&dir);
        save_weights_python_npy(&rt, &w, &dir).expect("save weights");

        let w2 = Weights::load_from_python_npy(&rt, &dir, cfg).expect("load weights");
        let loaded = w2.mamba_conv1d_weight.as_ref().expect("loaded conv weight");
        assert_eq!(loaded.shape(), expected_shape);
        assert_eq!(loaded.buffer.read_f32(), orig_data);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn mamba_conv1d_weight_adam_slots_preserve_layout() {
        let rt = crate::gpu_runtime().expect("gpu");
        let mut cfg = ModelConfig::sota_toy();
        cfg.layer_mixers = expand_layer_mixers(&[MixerKind::Mamba2], cfg.num_layers);
        let w = init_weights_seeded(&rt, cfg.clone(), 42).expect("init");
        let hp = OptimHyperparams::default();
        let mut state = OptimState::new(&rt, &w, hp.clone()).expect("state");
        state.step = 2;
        let slot = state.mamba_conv1d_weight.as_mut().unwrap();
        for (i, v) in slot.exp_avg.buffer.contents_f32().iter_mut().enumerate() {
            *v = (i as f32) * 0.01 - 0.5;
        }

        let root = std::env::temp_dir().join(format!(
            "arch02_mamba_conv_adam_layout_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("worker")
        ));
        let _ = std::fs::remove_dir_all(&root);
        let meta = TrainingCheckpointMeta {
            version: CHECKPOINT_VERSION,
            step: 2,
            data_cursor_tokens: 0,
            seed: 42,
            preset: "sota".into(),
            config: cfg.clone(),
            parameter_count: cfg.count_params(),
            optimizer: "muon_ns5_adamw".into(),
            hyperparams: hp.clone(),
            clip_mode: ClipMode::Soft,
            schedule: LrSchedule::from_warmdown(2_000, 350),
            bf16_precision: false,
            bf16_shadows_saved: false,
        };
        save_training_checkpoint(&rt, &w, &state, &root, &meta).expect("save");

        let w2 = Weights::load_from_python_npy(&rt, &root.join("weights"), cfg).expect("weights");
        let mut state2 = OptimState::new(&rt, &w2, hp).expect("state2");
        load_optim_state_python_npy(&mut state2, &root.join("optim"), &root.join("ema"))
            .expect("load full state");
        assert_eq!(
            state2
                .mamba_conv1d_weight
                .as_ref()
                .unwrap()
                .exp_avg
                .buffer
                .read_f32(),
            state
                .mamba_conv1d_weight
                .as_ref()
                .unwrap()
                .exp_avg
                .buffer
                .read_f32()
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
