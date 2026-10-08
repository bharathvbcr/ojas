//! Hand-written backward: mirrors Phase 1 fwd using tape stash.
//! Grad layout matches Weights ([in,out] for linears). Compare vs golden after
//! transpose-to-Python + clip_grad_norm_(0.3).

use std::sync::Arc;

use crate::dispatch::{dispatch_1d, dispatch_2d_tg, set_f32, set_tensor, set_u32};
use crate::gemm::{
    gemm_nt_accum_train, gemm_nt_train, gemm_tn_accum_train, gemm_tn_train, gemm_train,
    select_backend, GemmBackend,
};
use crate::mixers::{mamba2_bwd, mamba2_conv1d_bwd, mingru_bwd, mingru_vr_blend_bwd};
use crate::optim::{clip_grad_norm_device, ClipState};
use crate::runtime::{GpuRuntime, PrecisionMode};
use crate::ssm_glue::{
    accum_slice_grad, flatten_heads, mamba2_d_skip_bwd, mamba2_log_da_bwd, mamba2_x_scaled_bwd,
    mul_bwd, rms_norm_weight_bwd, silu_bwd, silu_bwd_store, softplus_bias_bwd, unflatten_heads,
};
use crate::tape::{LayerTape, Tape};
use crate::tensor::Tensor;
use crate::weights::Weights;

pub const GRAD_CLIP: f32 = 0.3;
pub const BWD_ATOL: f32 = 1e-4;

fn use_persistent_bf16(rt: &GpuRuntime, backend: GemmBackend) -> bool {
    rt.precision() == PrecisionMode::Bf16 && backend == GemmBackend::TensorOps && rt.has_tensorops()
}

/// Audit 7: pre-cast a shared f32 grad operand to bf16 **once** when each
/// consuming GEMM would otherwise `ensure_bf16` it separately
/// (`METAL_NATIVE_BWD_CAST_ONCE=1`). Returns a plain clone when the flag is
/// off, the tensor is already bf16, or the GEMM path would not cast (f32 mode
/// / no TensorOps). The consuming GEMMs see bit-identical inputs — same cast
/// kernel, same data — so this is numerics-neutral by construction.
fn bf16_once(rt: &GpuRuntime, backend: GemmBackend, t: &Tensor) -> Result<Tensor, String> {
    if crate::ab_flags::bwd_cast_once()
        && use_persistent_bf16(rt, backend)
        && t.dtype == crate::tensor::DType::F32
    {
        crate::gemm::cast_f32_to_bf16(t)
    } else {
        Ok(t.clone())
    }
}

/// Audit 7: per-section backward wall clock (`METAL_NATIVE_BWD_PROFILE=1`).
/// Synchronizes between sections so each lap ≈ GPU time of the section.
/// Diagnostic shares only — never read step_ms as a gate under this flag.
pub(crate) struct BwdProf {
    enabled: bool,
    label: &'static str,
    last: std::time::Instant,
    acc: Vec<(&'static str, f64)>,
}

impl BwdProf {
    fn new(rt: &Arc<GpuRuntime>) -> Result<Self, String> {
        Self::new_labeled(rt, crate::ab_flags::bwd_profile(), "bwd_profile")
    }

    /// Shared by the forward profiler (`METAL_NATIVE_FWD_PROFILE`).
    pub(crate) fn new_labeled(
        rt: &Arc<GpuRuntime>,
        enabled: bool,
        label: &'static str,
    ) -> Result<Self, String> {
        if enabled {
            rt.synchronize()?;
        }
        Ok(Self {
            enabled,
            label,
            last: std::time::Instant::now(),
            acc: Vec::new(),
        })
    }

    pub(crate) fn lap(&mut self, rt: &Arc<GpuRuntime>, name: &'static str) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        rt.synchronize()?;
        let now = std::time::Instant::now();
        let ms = now.duration_since(self.last).as_secs_f64() * 1e3;
        self.last = now;
        if let Some(e) = self.acc.iter_mut().find(|(n, _)| *n == name) {
            e.1 += ms;
        } else {
            self.acc.push((name, ms));
        }
        Ok(())
    }

    pub(crate) fn report(&self) {
        if !self.enabled || self.acc.is_empty() {
            return;
        }
        let total: f64 = self.acc.iter().map(|(_, v)| v).sum();
        let mut rows = self.acc.clone();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!(
            "{}: total {total:.1} ms (synced sections; shares, not a gate)",
            self.label
        );
        for (name, ms) in rows {
            eprintln!(
                "  {name:<16} {ms:>9.2} ms  {:>5.1}%",
                ms / total.max(1e-9) * 100.0
            );
        }
    }
}

/// Zero a tensor's logical window via GPU `zero_f32`.
///
/// Host `GpuBuffer::zero` races an in-flight Metal 4 CB. Note: `GpuBuffer::zero`
/// on a bump **view** would memset the entire slab (`nbytes` = capacity) — never
/// use it for bump windows.
pub(crate) fn zero_tensor_device(t: &Tensor) -> Result<(), String> {
    let rt = t.runtime();
    let n = t.numel();
    if n == 0 {
        return Ok(());
    }
    let p = rt.pipeline("zero_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, t, 0);
        set_u32(bnd, n as u32, 1);
    })
}

pub struct BlockGrads {
    pub q_gain: Tensor,
    pub vr_lambda: Tensor,
    pub attn_scale: Tensor,
    pub mlp_scale: Tensor,
    pub resid_mix: Tensor,
}

/// Parameter gradients (same shapes/layout as [`Weights`]).
pub struct Grads {
    pub tok_emb: Tensor,
    pub bigram_emb: Tensor,
    pub bigram_proj: Tensor,
    pub bigram_scale: Tensor,
    pub smear_gate: Tensor,
    pub ve_emb: Tensor,
    pub ve_proj: Tensor,
    pub ve_scale: Tensor,
    pub ve_layer_scales: Vec<Tensor>,
    pub skip_weights: Tensor,
    pub qo_bank: Tensor,
    pub kv_bank: Tensor,
    pub mingru_to_z: Option<Tensor>,
    pub mingru_to_h: Option<Tensor>,
    pub mingru_out: Option<Tensor>,
    pub mingru_v_proj: Option<Tensor>,
    pub mingru_v0_up: Option<Tensor>,
    pub mamba_in_proj: Option<Tensor>,
    pub mamba_conv1d_weight: Option<Tensor>,
    pub mamba_conv1d_bias: Option<Tensor>,
    pub mamba_out_proj: Option<Tensor>,
    pub mamba_a_log: Option<Tensor>,
    pub mamba_d: Option<Tensor>,
    pub mamba_dt_bias: Option<Tensor>,
    pub mamba_norm: Option<Tensor>,

    pub mlp_up: Tensor,
    pub mlp_down: Tensor,
    pub blocks: Vec<BlockGrads>,
}

impl Grads {
    pub fn zeros_like(rt: &Arc<GpuRuntime>, w: &Weights) -> Result<Self, String> {
        let z = |t: &Tensor| -> Result<Tensor, String> {
            // Hot: grads persist across steps (not cold-recycled temps).
            rt.alloc_tensor_f32_hot(&t.shape())
        };
        let mut blocks = Vec::new();
        for b in &w.blocks {
            blocks.push(BlockGrads {
                q_gain: z(&b.q_gain)?,
                vr_lambda: z(&b.vr_lambda)?,
                attn_scale: z(&b.attn_scale)?,
                mlp_scale: z(&b.mlp_scale)?,
                resid_mix: z(&b.resid_mix)?,
            });
        }
        let mut ve_layer_scales = Vec::new();
        for s in &w.ve_layer_scales {
            ve_layer_scales.push(z(s)?);
        }
        Ok(Self {
            tok_emb: z(&w.tok_emb)?,
            bigram_emb: z(&w.bigram_emb)?,
            bigram_proj: z(&w.bigram_proj)?,
            bigram_scale: z(&w.bigram_scale)?,
            smear_gate: z(&w.smear_gate)?,
            ve_emb: z(&w.ve_emb)?,
            ve_proj: z(&w.ve_proj)?,
            ve_scale: z(&w.ve_scale)?,
            ve_layer_scales,
            skip_weights: z(&w.skip_weights)?,
            qo_bank: z(&w.qo_bank)?,
            kv_bank: z(&w.kv_bank)?,
            mingru_to_z: w.mingru_to_z.as_ref().map(|t| z(t).unwrap()),
            mingru_to_h: w.mingru_to_h.as_ref().map(|t| z(t).unwrap()),
            mingru_out: w.mingru_out.as_ref().map(|t| z(t).unwrap()),
            mingru_v_proj: w.mingru_v_proj.as_ref().map(|t| z(t).unwrap()),
            mingru_v0_up: w.mingru_v0_up.as_ref().map(|t| z(t).unwrap()),
            mamba_in_proj: w.mamba_in_proj.as_ref().map(|t| z(t).unwrap()),
            mamba_conv1d_weight: w.mamba_conv1d_weight.as_ref().map(|t| z(t).unwrap()),
            mamba_conv1d_bias: w.mamba_conv1d_bias.as_ref().map(|t| z(t).unwrap()),
            mamba_out_proj: w.mamba_out_proj.as_ref().map(|t| z(t).unwrap()),
            mamba_a_log: w.mamba_a_log.as_ref().map(|t| z(t).unwrap()),
            mamba_d: w.mamba_d.as_ref().map(|t| z(t).unwrap()),
            mamba_dt_bias: w.mamba_dt_bias.as_ref().map(|t| z(t).unwrap()),
            mamba_norm: w.mamba_norm.as_ref().map(|t| z(t).unwrap()),
            mlp_up: z(&w.mlp_up)?,
            mlp_down: z(&w.mlp_down)?,
            blocks,
        })
    }
}

/// Full backward from taped forward. Applies on-device `clip_grad_norm_(GRAD_CLIP)`
/// unless `clip` is false (for finite-difference checks).
pub fn backward_f32(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    tape: &Tape,
    grads: &mut Grads,
) -> Result<(), String> {
    backward_f32_opts(rt, w, tape, grads, true)
}

pub fn backward_f32_opts(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    tape: &Tape,
    grads: &mut Grads,
    clip: bool,
) -> Result<(), String> {
    backward_f32_opts_clip(rt, w, tape, grads, clip, None, false).map(|_| ())
}

/// Like [`backward_f32_opts`] but reuses a persistent [`ClipState`] (training loop).
/// When `read_norm` is true, returns the clipped grad L2 (log steps only).
pub fn backward_f32_opts_clip(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    tape: &Tape,
    grads: &mut Grads,
    clip: bool,
    clip_state: Option<&ClipState>,
    read_norm: bool,
) -> Result<Option<f32>, String> {
    let cfg = &w.cfg;
    let backend = select_backend(rt);
    let b = cfg.batch;
    let tlen = cfg.seq_len;
    let c = cfg.model_dim;
    let bt = b * tlen;
    let n = cfg.num_layers;
    let n_enc = n / 2;
    let eps = cfg.f32_eps();
    let h = cfg.num_heads;
    let hkv = cfg.num_kv_heads;
    let d = cfg.head_dim;
    let kv = cfg.kv_dim();
    let mlp = cfg.mlp_dim;
    let vsz = cfg.vocab_size;

    let ids = tape.input_ids.as_ref().ok_or("tape.input_ids")?;
    let tgts = tape.target_ids.as_ref().ok_or("tape.target_ids")?;
    let logits_post = tape.logits_post.as_ref().ok_or("tape.logits_post")?;
    let final_norm = tape.final_norm.as_ref().ok_or("tape.final_norm")?;
    let pre_final = tape.pre_final_norm.as_ref().ok_or("tape.pre_final_norm")?;
    let x0 = tape.x0.as_ref().ok_or("tape.x0")?;
    let v0 = tape.v0.as_ref().ok_or("tape.v0")?;

    let mut prof = BwdProf::new(rt)?;

    // ---- head: CE + softcap → d_logits_pre (post only; Phase F dropped pre stash) ----
    let d_logits = rt.alloc_tensor_f32(&[bt, vsz])?;
    {
        let p = rt.pipeline("ce_softcap_bwd_f32")?;
        let post_flat = reshape_view(logits_post, &[bt, vsz]);
        dispatch_1d(rt, &p, bt, |bnd| {
            set_tensor(bnd, &post_flat, 0);
            set_tensor(bnd, tgts, 1);
            set_tensor(bnd, &d_logits, 2);
            set_u32(bnd, bt as u32, 3);
            set_u32(bnd, vsz as u32, 4);
            set_f32(bnd, cfg.logit_softcap, 5);
        })?;
    }
    prof.lap(rt, "head_ce")?;

    // logits = final_norm @ tok_emb_t; d(tok_emb_t) = FN^T @ dL; dFN = dL @ tok_emb_t^T
    let fn_flat = reshape_view(final_norm, &[bt, c]);
    let d_fn = rt.alloc_tensor_f32(&[bt, c])?;
    {
        // dX = dY @ W^T with W=tok_emb_t [C,V] → use tok_emb [V,C] as NT right operand
        // Actually: W_t = tok_emb_t [C,V]; dY @ W_t^T = dY @ tok_emb. Direct NN with tok_emb.
        // Audit 7: d_logits feeds two GEMMs — cast once under BWD_CAST_ONCE.
        let d_logits_op = bf16_once(rt, backend, &d_logits)?;
        gemm_train(&d_logits_op, &w.tok_emb, &d_fn, backend)?;
        // dW_t = X^T @ dY → [C,V]; then transpose-add into tok_emb [V,C]
        let d_tok_t = rt.alloc_tensor_f32(&[c, vsz])?;
        gemm_tn_train(&fn_flat, &d_logits_op, &d_tok_t, backend)?;
        let d_tok = transpose2d(rt, &d_tok_t)?;
        add_inplace(rt, &grads.tok_emb, &d_tok)?;
    }
    prof.lap(rt, "head_gemms")?;

    // final RMSNorm bwd
    let mut dx = rt.alloc_tensor_f32(&[b, tlen, c])?;
    {
        let p = rt.pipeline("rms_norm_bwd_f32")?;
        let d_fn3 = reshape_view(&d_fn, &[b, tlen, c]);
        dispatch_1d(rt, &p, bt, |bnd| {
            set_tensor(bnd, pre_final, 0);
            set_tensor(bnd, &d_fn3, 1);
            set_tensor(bnd, &dx, 2);
            set_u32(bnd, bt as u32, 3);
            set_u32(bnd, c as u32, 4);
            set_f32(bnd, eps, 5);
        })?;
    }
    prof.lap(rt, "final_rms")?;

    // Cross-layer accumulators
    let dx0 = rt.alloc_tensor_f32(&[b, tlen, c])?;
    zero_tensor_device(&dx0)?;
    let dv0 = rt.alloc_tensor_f32(&[b, tlen, hkv, d])?;
    zero_tensor_device(&dv0)?;
    // d(skip_i) for encoder outputs — index by encoder layer
    let d_skips: Vec<Tensor> = (0..n_enc)
        .map(|_| {
            let t = rt.alloc_tensor_f32(&[b, tlen, c]).unwrap();
            zero_tensor_device(&t).unwrap();
            t
        })
        .collect();

    // ---- layers reverse ----
    for layer in (0..n).rev() {
        let is_decoder = layer >= n_enc;
        let lt = tape.layer.get(layer).ok_or("layer tape")?;
        let bw = &w.blocks[layer];

        // If decoder: undo skip-add on dx first (dx arrives as d(after_skip))
        if is_decoder {
            let skip_i = layer - n_enc;
            let enc_layer = n_enc - 1 - skip_i; // LIFO: layer2→enc1, layer3→enc0
            let skip = tape.skips.get(enc_layer).ok_or("skip tape")?;
            let sw_t = w.skip_weights.view(&[c], skip_i * c);
            let dsw = rt.alloc_tensor_f32(&[c])?;
            zero_tensor_device(&dsw)?;
            let dskip = rt.alloc_tensor_f32(&[b, tlen, c])?;
            let dx_stream = rt.alloc_tensor_f32(&[b, tlen, c])?;
            {
                let p = rt.pipeline("skip_add_bwd_f32")?;
                dispatch_1d(rt, &p, bt, |bnd| {
                    set_tensor(bnd, skip, 0);
                    set_tensor(bnd, &sw_t, 1);
                    set_tensor(bnd, &dx, 2);
                    set_tensor(bnd, &dx_stream, 3);
                    set_tensor(bnd, &dskip, 4);
                    set_tensor(bnd, &dsw, 5);
                    set_u32(bnd, bt as u32, 6);
                    set_u32(bnd, c as u32, 7);
                })?;
            }
            add_inplace(rt, &d_skips[enc_layer], &dskip)?;
            let dest = grads.skip_weights.view(&[c], skip_i * c);
            add_inplace(rt, &dest, &dsw)?;
            dx = dx_stream;
        }

        // Add skip grad into layer output for encoder layers
        let x_mid = lt.x_mid.as_ref().ok_or("x_mid")?;
        let x_in = lt.x_in.as_ref().ok_or("x_in")?;
        let x_stream = lt.x_stream.as_ref().ok_or("x_stream")?;
        let attn_out = lt.attn_out.as_ref().ok_or("attn_out")?;
        let mlp_out = lt.mlp_out.as_ref().ok_or("mlp_out")?;
        let mlp_in = lt.mlp_in.as_ref().ok_or("mlp_in")?;
        let mlp_pre = lt.mlp_pre_act.as_ref().ok_or("mlp_pre")?;
        let mlp_hid = lt.mlp_hidden.as_ref().ok_or("mlp_hid")?;
        let attn_in = lt.attn_in.as_ref().ok_or("attn_in")?;

        if !is_decoder {
            add_inplace(rt, &dx, &d_skips[layer])?;
        }
        prof.lap(rt, "skip_glue")?;

        // --- MLP residual ---
        let d_mlp_out = rt.alloc_tensor_f32(&[b, tlen, c])?;
        let dx_mid = rt.alloc_tensor_f32(&[b, tlen, c])?;
        {
            let p = rt.pipeline("residual_scale_add_bwd_f32")?;
            // qkv_post_bwd owns one (token, query-head) per thread.  This is
            // dimension-safe for C=768/KV=384 and avoids fixed token-local
            // scratch arrays that previously truncated gradients.
            dispatch_1d(rt, &p, bt * h, |bnd| {
                set_tensor(bnd, mlp_out, 0);
                set_tensor(bnd, &bw.mlp_scale, 1);
                set_tensor(bnd, &dx, 2);
                set_tensor(bnd, &dx_mid, 3);
                set_tensor(bnd, &d_mlp_out, 4);
                set_tensor(bnd, &grads.blocks[layer].mlp_scale, 5);
                set_u32(bnd, bt as u32, 6);
                set_u32(bnd, c as u32, 7);
            })?;
        }
        prof.lap(rt, "resid_glue")?;

        // MLP down: mlp_out = hidden @ down_w
        let down_w = if use_persistent_bf16(rt, backend) {
            if let Some(ref bf) = w.bf16_banks {
                w.bank_matrix(rt, &bf.mlp_down, layer, mlp, c)?
            } else {
                w.bank_matrix(rt, &w.mlp_down, layer, mlp, c)?
            }
        } else {
            w.bank_matrix(rt, &w.mlp_down, layer, mlp, c)?
        };
        let d_hid = rt.alloc_tensor_f32(&[bt, mlp])?;
        let d_mlp_flat = reshape_view(&d_mlp_out, &[bt, c]);
        let hid_flat = reshape_view(mlp_hid, &[bt, mlp]);
        {
            // Audit 7: d_mlp_flat feeds dH + dW — cast once under BWD_CAST_ONCE.
            let d_mlp_op = bf16_once(rt, backend, &d_mlp_flat)?;
            // dH = dY @ W^T
            gemm_nt_train(&d_mlp_op, &down_w, &d_hid, backend)?;
            // dW += H^T @ dY directly into bank view
            let dw = grads.mlp_down.view(&[mlp, c], layer * mlp * c);
            gemm_tn_accum_train(&hid_flat, &d_mlp_op, &dw, backend)?;
        }
        prof.lap(rt, "mlp_gemms")?;

        // act bwd
        let d_pre_act = rt.alloc_tensor_f32(&[bt, mlp])?;
        {
            let p = rt.pipeline("mlp_act_bwd_f32")?;
            dispatch_1d(rt, &p, bt * mlp, |bnd| {
                set_tensor(bnd, mlp_pre, 0);
                set_tensor(bnd, &d_hid, 1);
                set_tensor(bnd, &d_pre_act, 2);
                set_u32(bnd, (bt * mlp) as u32, 3);
            })?;
        }
        prof.lap(rt, "mlp_act")?;

        // up: hidden_pre = mlp_in @ up_w
        let up_w = if use_persistent_bf16(rt, backend) {
            if let Some(ref bf) = w.bf16_banks {
                w.bank_matrix(rt, &bf.mlp_up, layer, c, mlp)?
            } else {
                w.bank_matrix(rt, &w.mlp_up, layer, c, mlp)?
            }
        } else {
            w.bank_matrix(rt, &w.mlp_up, layer, c, mlp)?
        };
        let d_mlp_in = rt.alloc_tensor_f32(&[bt, c])?;
        let mi_flat = reshape_view(mlp_in, &[bt, c]);
        {
            // Audit 7: d_pre_act feeds dX + dW — cast once under BWD_CAST_ONCE.
            let d_pre_op = bf16_once(rt, backend, &d_pre_act)?;
            gemm_nt_train(&d_pre_op, &up_w, &d_mlp_in, backend)?;
            let dw = grads.mlp_up.view(&[c, mlp], layer * c * mlp);
            gemm_tn_accum_train(&mi_flat, &d_pre_op, &dw, backend)?;
        }
        prof.lap(rt, "mlp_gemms")?;

        // ln_scale + rms on x_mid + attn residual_scale_add (fused megakernel;
        // METAL_NATIVE_RESID_BWD_FUSE=0 → unfused Soft-bisect path).
        let scale = cfg.ln_scale_factor(layer);
        let d_attn_out = rt.alloc_tensor_f32(&[b, tlen, c])?;
        let dx_in = rt.alloc_tensor_f32(&[b, tlen, c])?;
        {
            let d_mi = reshape_view(&d_mlp_in, &[b, tlen, c]);
            if crate::ab_flags::resid_bwd_fuse() {
                // Audit 8: row-block variant removes 3.1M per-element device
                // atomics per call and reduces `d` (already materialized in
                // dx_in) in a second coalesced pass. Not bit-identical.
                let rowblock = crate::ab_flags::glue_rowblock();
                let name = if rowblock {
                    "residual_scale_add_rms_norm_scale_bwd_noatom_f32"
                } else {
                    "residual_scale_add_rms_norm_scale_bwd_f32"
                };
                let p = rt.pipeline(name)?;
                dispatch_1d(rt, &p, bt, |bnd| {
                    set_tensor(bnd, x_mid, 0);
                    set_tensor(bnd, &d_mi, 1);
                    set_tensor(bnd, attn_out, 2);
                    set_tensor(bnd, &bw.attn_scale, 3);
                    set_tensor(bnd, &dx_mid, 4);
                    set_tensor(bnd, &dx_in, 5);
                    set_tensor(bnd, &d_attn_out, 6);
                    set_tensor(bnd, &grads.blocks[layer].attn_scale, 7);
                    set_u32(bnd, bt as u32, 8);
                    set_u32(bnd, c as u32, 9);
                    set_f32(bnd, eps, 10);
                    set_f32(bnd, scale, 11);
                })?;
                if rowblock {
                    let rb = crate::ab_flags::glue_row_blocks().min(bt).max(1);
                    let pr = rt.pipeline("reduce_dscale_rowblock_f32")?;
                    dispatch_1d(rt, &pr, c * rb, |bnd| {
                        set_tensor(bnd, &dx_in, 0);
                        set_tensor(bnd, attn_out, 1);
                        set_tensor(bnd, &grads.blocks[layer].attn_scale, 2);
                        set_u32(bnd, bt as u32, 3);
                        set_u32(bnd, c as u32, 4);
                        set_u32(bnd, rb as u32, 5);
                    })?;
                }
            } else {
                // Unfused: rms_norm_scale_bwd accum into dx_mid, then residual_scale_add_bwd.
                let p_rms = rt.pipeline("rms_norm_scale_bwd_f32")?;
                dispatch_1d(rt, &p_rms, bt, |bnd| {
                    set_tensor(bnd, x_mid, 0);
                    set_tensor(bnd, &d_mi, 1);
                    set_tensor(bnd, &dx_mid, 2); // unused when accum=1
                    set_tensor(bnd, &dx_mid, 3);
                    set_u32(bnd, bt as u32, 4);
                    set_u32(bnd, c as u32, 5);
                    set_f32(bnd, eps, 6);
                    set_f32(bnd, scale, 7);
                    set_u32(bnd, 1, 8); // accum
                })?;
                let p_res = rt.pipeline("residual_scale_add_bwd_f32")?;
                dispatch_1d(rt, &p_res, bt, |bnd| {
                    set_tensor(bnd, attn_out, 0);
                    set_tensor(bnd, &bw.attn_scale, 1);
                    set_tensor(bnd, &dx_mid, 2);
                    set_tensor(bnd, &dx_in, 3);
                    set_tensor(bnd, &d_attn_out, 4);
                    set_tensor(bnd, &grads.blocks[layer].attn_scale, 5);
                    set_u32(bnd, bt as u32, 6);
                    set_u32(bnd, c as u32, 7);
                })?;
            }
        }
        prof.lap(rt, "resid_glue")?;

        let d_attn_in = match cfg.layer_mixer(layer) {
            crate::weights::MixerKind::Attention => attention_bwd(
                rt,
                w,
                grads,
                layer,
                &attn_in,
                &d_attn_out,
                ids,
                backend,
                Some(v0),
                b,
                tlen,
                c,
                kv,
                h,
                hkv,
                d,
                bt,
                dv0.clone(),
                lt,
                &mut prof,
            )?,
            crate::weights::MixerKind::Mamba2 => {
                let out = mamba2_bwd_rust(
                    rt,
                    w,
                    grads,
                    layer,
                    &attn_in,
                    &d_attn_out,
                    backend,
                    b,
                    tlen,
                    c,
                    bt,
                    lt,
                )?;
                prof.lap(rt, "mixer_ssm")?;
                out
            }
            crate::weights::MixerKind::MinGRU => {
                let out = mingru_bwd_rust(
                    rt,
                    w,
                    grads,
                    layer,
                    &attn_in,
                    &d_attn_out,
                    backend,
                    Some(v0),
                    b,
                    tlen,
                    c,
                    kv,
                    hkv,
                    d,
                    bt,
                    dv0.clone(),
                    lt,
                )?;
                prof.lap(rt, "mixer_ssm")?;
                out
            }
        };

        // ln_scale + rms for attn + resid_mix (fused megakernel; fuse flag for Soft bisect)
        let dx_stream = rt.alloc_tensor_f32(&[b, tlen, c])?;
        {
            let din = reshape_view(&d_attn_in, &[b, tlen, c]);
            if crate::ab_flags::resid_bwd_fuse() {
                let rowblock = crate::ab_flags::glue_rowblock();
                let name = if rowblock {
                    "resid_mix_rms_norm_scale_bwd_noatom_f32"
                } else {
                    "resid_mix_rms_norm_scale_bwd_f32"
                };
                let p = rt.pipeline(name)?;
                dispatch_1d(rt, &p, bt, |bnd| {
                    set_tensor(bnd, x_in, 0);
                    set_tensor(bnd, &din, 1);
                    set_tensor(bnd, x_stream, 2);
                    set_tensor(bnd, x0, 3);
                    set_tensor(bnd, &bw.resid_mix, 4);
                    set_tensor(bnd, &dx_in, 5);
                    set_tensor(bnd, &dx_stream, 6);
                    set_tensor(bnd, &dx0, 7);
                    set_tensor(bnd, &grads.blocks[layer].resid_mix, 8);
                    set_u32(bnd, bt as u32, 9);
                    set_u32(bnd, c as u32, 10);
                    set_f32(bnd, eps, 11);
                    set_f32(bnd, scale, 12);
                })?;
                if rowblock {
                    // dmix twin does *two* atomics per element inline (6.2M);
                    // this reduces both accumulators in one pass.
                    let rb = crate::ab_flags::glue_row_blocks().min(bt).max(1);
                    let pr = rt.pipeline("reduce_dmix_rowblock_f32")?;
                    dispatch_1d(rt, &pr, c * rb, |bnd| {
                        set_tensor(bnd, &dx_in, 0);
                        set_tensor(bnd, x_stream, 1);
                        set_tensor(bnd, x0, 2);
                        set_tensor(bnd, &grads.blocks[layer].resid_mix, 3);
                        set_u32(bnd, bt as u32, 4);
                        set_u32(bnd, c as u32, 5);
                        set_u32(bnd, rb as u32, 6);
                    })?;
                }
            } else {
                let p_rms = rt.pipeline("rms_norm_scale_bwd_f32")?;
                dispatch_1d(rt, &p_rms, bt, |bnd| {
                    set_tensor(bnd, x_in, 0);
                    set_tensor(bnd, &din, 1);
                    set_tensor(bnd, &dx_in, 2);
                    set_tensor(bnd, &dx_in, 3);
                    set_u32(bnd, bt as u32, 4);
                    set_u32(bnd, c as u32, 5);
                    set_f32(bnd, eps, 6);
                    set_f32(bnd, scale, 7);
                    set_u32(bnd, 1, 8); // accum into dx_in
                })?;
                let p_mix = rt.pipeline("resid_mix_bwd_simple_f32")?;
                dispatch_1d(rt, &p_mix, bt, |bnd| {
                    set_tensor(bnd, x_stream, 0);
                    set_tensor(bnd, x0, 1);
                    set_tensor(bnd, &bw.resid_mix, 2);
                    set_tensor(bnd, &dx_in, 3);
                    set_tensor(bnd, &dx_stream, 4);
                    set_tensor(bnd, &dx0, 5);
                    set_tensor(bnd, &grads.blocks[layer].resid_mix, 6);
                    set_u32(bnd, bt as u32, 7);
                    set_u32(bnd, c as u32, 8);
                })?;
            }
        }
        prof.lap(rt, "resid_glue")?;

        dx = dx_stream;
    }

    // dx is now d(stem); add dx0 (resid_mix path) into stem grad
    add_inplace(rt, &dx, &dx0)?;
    prof.lap(rt, "stem_glue")?;

    // smear bwd
    let stem_post = tape.stem_post_norm.as_ref().ok_or("stem_post")?;
    let stem_pre = tape.stem_pre_norm.as_ref().ok_or("stem_pre")?;
    let d_post = rt.alloc_tensor_f32(&[b, tlen, c])?;
    zero_tensor_device(&d_post)?;
    {
        let p = rt.pipeline("stem_smear_bwd_f32")?;
        dispatch_1d(rt, &p, bt, |bnd| {
            set_tensor(bnd, stem_post, 0);
            set_tensor(bnd, &w.smear_gate, 1);
            set_tensor(bnd, &dx, 2);
            set_tensor(bnd, &d_post, 3);
            set_tensor(bnd, &grads.smear_gate, 4);
            set_u32(bnd, b as u32, 5);
            set_u32(bnd, tlen as u32, 6);
            set_u32(bnd, c as u32, 7);
        })?;
    }

    // rms stem bwd
    let d_pre = rt.alloc_tensor_f32(&[b, tlen, c])?;
    {
        let p = rt.pipeline("rms_norm_bwd_f32")?;
        dispatch_1d(rt, &p, bt, |bnd| {
            set_tensor(bnd, stem_pre, 0);
            set_tensor(bnd, &d_post, 1);
            set_tensor(bnd, &d_pre, 2);
            set_u32(bnd, bt as u32, 3);
            set_u32(bnd, c as u32, 4);
            set_f32(bnd, eps, 5);
        })?;
    }

    // stem embed bwd via gather + GEMM (Phase D)
    {
        let db = cfg.bigram_dim;
        let d_pre_flat = reshape_view(&d_pre, &[bt, c]);
        // tok scatter
        {
            let p = rt.pipeline("stem_scatter_tok_f32")?;
            dispatch_1d(rt, &p, bt, |bnd| {
                set_tensor(bnd, ids, 0);
                set_tensor(bnd, &d_pre_flat, 1);
                set_tensor(bnd, &grads.tok_emb, 2);
                set_u32(bnd, bt as u32, 3);
                set_u32(bnd, c as u32, 4);
            })?;
        }
        // re-gather bigram rows + hash
        let bg_rows = rt.alloc_tensor_f32(&[bt, db])?;
        let hash_scratch = rt.alloc_tensor_f32(&[b, tlen, c])?; // unused sink for tok re-gather
        let hash_idx = {
            let nbytes = bt * 4;
            let buf = rt.alloc_buffer(nbytes)?;
            Tensor::from_buffer(rt, buf, &[bt], crate::tensor::DType::F32, 0)?
        };
        {
            let p = rt.pipeline("stem_gather_f32")?;
            dispatch_1d(rt, &p, bt, |bnd| {
                set_tensor(bnd, ids, 0);
                set_tensor(bnd, &w.tok_emb, 1);
                set_tensor(bnd, &w.bigram_emb, 2);
                set_tensor(bnd, &hash_scratch, 3);
                set_tensor(bnd, &bg_rows, 4);
                set_tensor(bnd, &hash_idx, 5);
                set_u32(bnd, b as u32, 6);
                set_u32(bnd, tlen as u32, 7);
                set_u32(bnd, c as u32, 8);
                set_u32(bnd, cfg.bigram_vocab as u32, 9);
                set_u32(bnd, db as u32, 10);
            })?;
        }
        let bg_proj_out = rt.alloc_tensor_f32(&[bt, c])?;
        let bg_w = if use_persistent_bf16(rt, backend) {
            if let Some(ref bf) = w.bf16_banks {
                bf.bigram_proj.clone()
            } else {
                w.bigram_proj.clone()
            }
        } else {
            w.bigram_proj.clone()
        };
        gemm_train(&bg_rows, &bg_w, &bg_proj_out, backend)?;
        let d_bg_proj = rt.alloc_tensor_f32(&[bt, c])?;
        {
            let p = rt.pipeline("stem_scale_bwd_f32")?;
            dispatch_1d(rt, &p, bt * c, |bnd| {
                set_tensor(bnd, &d_pre_flat, 0);
                set_tensor(bnd, &bg_proj_out, 1);
                set_tensor(bnd, &w.bigram_scale, 2);
                set_tensor(bnd, &d_bg_proj, 3);
                set_tensor(bnd, &grads.bigram_scale, 4);
                set_u32(bnd, (bt * c) as u32, 5);
            })?;
        }
        // Audit 7: d_bg_proj feeds dW + d_emb — cast once under BWD_CAST_ONCE.
        let d_bg_op = bf16_once(rt, backend, &d_bg_proj)?;
        // d_proj = bg_rows^T @ d_bg_proj
        {
            let dw = grads.bigram_proj.view(&[db, c], 0);
            gemm_tn_accum_train(&bg_rows, &d_bg_op, &dw, backend)?;
        }
        // d_emb_dense = d_bg_proj @ proj^T
        {
            let d_emb = rt.alloc_tensor_f32(&[bt, db])?;
            gemm_nt_train(&d_bg_op, &bg_w, &d_emb, backend)?;
            let p = rt.pipeline("stem_scatter_bigram_f32")?;
            dispatch_1d(rt, &p, bt, |bnd| {
                set_tensor(bnd, &hash_idx, 0);
                set_tensor(bnd, &d_emb, 1);
                set_tensor(bnd, &grads.bigram_emb, 2);
                set_u32(bnd, bt as u32, 3);
                set_u32(bnd, db as u32, 4);
            })?;
        }
    }

    prof.lap(rt, "stem_gemms")?;

    let res = if clip {
        match clip_state {
            Some(cs) => clip_grad_norm_device(rt, grads, cs, GRAD_CLIP, read_norm),
            None => {
                let owned = ClipState::new(rt)?;
                clip_grad_norm_device(rt, grads, &owned, GRAD_CLIP, read_norm)
            }
        }
    } else {
        Ok(None)
    };
    prof.lap(rt, "clip")?;
    prof.report();
    res
}

fn reshape_view(t: &Tensor, shape: &[usize]) -> Tensor {
    let numel: usize = shape.iter().product();
    assert_eq!(numel, t.numel());
    Tensor::from_buffer(t.runtime(), t.buffer.clone(), shape, t.dtype, t.byte_offset())
        .expect("these views are built over a buffer this crate just allocated")
}

fn upload(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Result<Tensor, String> {
    let t = rt.alloc_tensor_f32(shape)?;
    t.buffer.write_f32(data);
    Ok(t)
}

fn slice_host(t: &Tensor, off: usize, n: usize) -> Vec<f32> {
    // Views may have a byte_offset; read only the logical window.
    let base = t.byte_offset() / 4;
    let all = t.buffer.read_f32();
    all[base + off..base + off + n].to_vec()
}

fn transpose2d(rt: &Arc<GpuRuntime>, t: &Tensor) -> Result<Tensor, String> {
    assert_eq!(t.shape().len(), 2);
    let rows = t.shape()[0];
    let cols = t.shape()[1];
    let out = rt.alloc_tensor_f32(&[cols, rows])?;
    let p = rt.pipeline("transpose2d_f32")?;
    dispatch_1d(rt, &p, rows * cols, |bnd| {
        set_tensor(bnd, t, 0);
        set_tensor(bnd, &out, 1);
        set_u32(bnd, rows as u32, 2);
        set_u32(bnd, cols as u32, 3);
    })?;
    Ok(out)
}

fn add_inplace(rt: &Arc<GpuRuntime>, dst: &Tensor, src: &Tensor) -> Result<(), String> {
    assert_eq!(dst.numel(), src.numel());
    let p = rt.pipeline("add_inplace_f32")?;
    dispatch_1d(rt, &p, dst.numel(), |bnd| {
        set_tensor(bnd, dst, 0);
        set_tensor(bnd, src, 1);
        set_u32(bnd, dst.numel() as u32, 2);
    })?;
    Ok(())
}

fn attention_bwd(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    grads: &Grads,
    layer: usize,
    attn_in: &Tensor,
    d_attn_out: &Tensor,
    ids: &Tensor,
    backend: GemmBackend,
    v0: Option<&Tensor>,
    b: usize,
    tlen: usize,
    c: usize,
    kv: usize,
    h: usize,
    hkv: usize,
    d: usize,
    bt: usize,
    dv0: Tensor,
    lt: &crate::tape::LayerTape,
    prof: &mut BwdProf,
) -> Result<Tensor, String> {
    let cfg = &w.cfg;
    let ai = cfg
        .attn_local_idx(layer)
        .ok_or("attention_bwd on non-attn layer")?;
    let n_attn = cfg.mixer_count(crate::weights::MixerKind::Attention);

    let attn_y = lt.attn_y.as_ref().unwrap();
    let attn_y_flash = lt.attn_y_flash.as_ref().unwrap();
    let v_mixed = lt.v_mixed.as_ref().unwrap();
    let q = lt.q.as_ref().unwrap();
    let k = lt.k.as_ref().unwrap();
    let _dq_pre = rt.alloc_tensor_f32(&[bt, h * d])?;
    let _dk_pre = rt.alloc_tensor_f32(&[bt, kv])?;
    let _dv_pre = rt.alloc_tensor_f32(&[bt, kv])?;
    let _dve = rt.alloc_tensor_f32(&[bt, kv])?;
    let q_pre = lt.q_pre.as_ref().unwrap();
    let k_pre = lt.k_pre.as_ref().unwrap();
    let v_pre = lt.v_pre.as_ref().unwrap();

    let raw_v = lt.raw_v.as_ref().unwrap();
    let bw = &w.blocks[layer];
    let eps = cfg.f32_eps();
    let _use_ve = cfg.ve_scale_index(layer).is_some();
    let _use_v0 = v0.is_some();
    let v0_buf = v0.unwrap_or(raw_v); // Dummy fallback if none, just like fwd

    // Original extracted code (some vars might need re-binding, we'll fix compiler errors next if any)
    // out proj: attn_out = y @ out_w — d_y is [b,t,h,d] ≡ [bt,c] layout.
    let out_w = if use_persistent_bf16(rt, backend) {
        if let Some(ref bf) = w.bf16_banks {
            w.bank_matrix(rt, &bf.qo_bank, n_attn + ai, c, c)?
        } else {
            w.bank_matrix(rt, &w.qo_bank, n_attn + ai, c, c)?
        }
    } else {
        w.bank_matrix(rt, &w.qo_bank, n_attn + ai, c, c)?
    };
    let d_y = rt.alloc_tensor_f32(&[b, tlen, h, d])?;
    {
        let d_ao = reshape_view(&d_attn_out, &[bt, c]);
        let y_flat = reshape_view(attn_y, &[bt, c]);
        let dy_flat = reshape_view(&d_y, &[bt, c]);
        // Audit 7: d_ao feeds dY + dW — cast once under BWD_CAST_ONCE.
        let d_ao_op = bf16_once(rt, backend, &d_ao)?;
        gemm_nt_train(&d_ao_op, &out_w, &dy_flat, backend)?;
        let dw = grads.qo_bank.view(&[c, c], (n_attn + ai) * c * c);
        gemm_tn_accum_train(&y_flat, &d_ao_op, &dw, backend)?;
    }
    prof.lap(rt, "attn_out_gemms")?;

    // XSA bwd
    let (d_y_flash, dv_flash) = if cfg.use_xsa(layer) {
        let d_y_flash = rt.alloc_tensor_f32(&[b, tlen, h, d])?;
        let dv_flash = rt.alloc_tensor_f32(&[b, tlen, hkv, d])?;
        zero_tensor_device(&dv_flash)?;
        let p = rt.pipeline("xsa_bwd_f32")?;
        let n_xsa = b * tlen * hkv;
        dispatch_1d(rt, &p, n_xsa, |bnd| {
            set_tensor(bnd, attn_y_flash, 0);
            set_tensor(bnd, v_mixed, 1);
            set_tensor(bnd, &d_y, 2);
            set_tensor(bnd, &d_y_flash, 3);
            set_tensor(bnd, &dv_flash, 4);
            set_u32(bnd, b as u32, 5);
            set_u32(bnd, tlen as u32, 6);
            set_u32(bnd, h as u32, 7);
            set_u32(bnd, hkv as u32, 8);
            set_u32(bnd, d as u32, 9);
            set_f32(bnd, cfg.xsa_eps(), 10);
        })?;
        (d_y_flash, Some(dv_flash))
    } else {
        // No XSA: flash output == post-XSA; reuse d_y buffer.
        (d_y.clone(), None)
    };
    prof.lap(rt, "xsa")?;

    // Flash attn bwd: default row-wise O(T²)+LSE (pre-Phase2); tiled BR=BC=32
    // via METAL_NATIVE_FA_TILED=1. Under Bf16: bf16 Q/K/V, f32 O/dO/L/Delta/grads.
    const BR: usize = 32;
    const BC: usize = 32;
    let q_blocks = (tlen + BR - 1) / BR;
    let k_blocks = (tlen + BC - 1) / BC;
    let attn_lse = lt.attn_lse.as_ref().ok_or("attn_lse")?;
    let delta = rt.alloc_tensor_f32(&[b, h, tlen])?;
    {
        let p = rt.pipeline("flash_attn_bwd_delta_f32")?;
        dispatch_1d(rt, &p, b * h * tlen, |bnd| {
            set_tensor(bnd, attn_y_flash, 0);
            set_tensor(bnd, &d_y_flash, 1);
            set_tensor(bnd, &delta, 2);
            set_u32(bnd, b as u32, 3);
            set_u32(bnd, tlen as u32, 4);
            set_u32(bnd, h as u32, 5);
            set_u32(bnd, d as u32, 6);
        })?;
    }
    prof.lap(rt, "fa_delta")?;
    let dq = rt.alloc_tensor_f32(&[b, tlen, h, d])?;
    let dk = rt.alloc_tensor_f32(&[b, tlen, hkv, d])?;
    let dv = rt.alloc_tensor_f32(&[b, tlen, hkv, d])?;
    {
        let scale = 1.0 / (d as f32).sqrt();
        crate::model_fwd::require_flash_head_dim(d)?;
        // Skip bf16 QKV cast→flash: GEMM accum is f32, so f32 FA bwd is the
        // direct path (same as fwd). Tiled bf16 twins remain available if
        // persistent bf16 QKV lands later.
        let use_tiled = crate::ab_flags::fa_tiled_bwd();
        if use_tiled {
            // Audit 7: the Phase 4 bf16 tiled twins are drop-in
            // signature-compatible (only Q/K/V become bfloat) and were
            // previously unreferenced. `METAL_NATIVE_FA_BF16=1` now
            // reaches them, so bf16 FA bwd is available on the tiled
            // path too — not only the specialized row path.
            // Require *both* twins: falling back on only one would bind
            // bf16 buffers to an f32 kernel.
            let tiled_bf16 = crate::ab_flags::fa_bf16_row()
                && rt.precision() == PrecisionMode::Bf16
                && rt.pipeline("flash_attn_bwd_dq_bf16").is_ok()
                && rt.pipeline("flash_attn_bwd_dkv_bf16").is_ok();
            let (q_t, k_t, v_t) = if tiled_bf16 {
                (
                    crate::gemm::cast_f32_to_bf16(q)?,
                    crate::gemm::cast_f32_to_bf16(k)?,
                    crate::gemm::cast_f32_to_bf16(v_mixed)?,
                )
            } else {
                (q.clone(), k.clone(), v_mixed.clone())
            };
            let (t1, t2) = if tiled_bf16 {
                ("flash_attn_bwd_dq_bf16", "flash_attn_bwd_dkv_bf16")
            } else {
                ("flash_attn_bwd_dq_f32", "flash_attn_bwd_dkv_f32")
            };
            let p1 = rt.pipeline(t1)?;
            dispatch_2d_tg(rt, &p1, q_blocks, b * h, BR, |bnd| {
                set_tensor(bnd, &q_t, 0);
                set_tensor(bnd, &k_t, 1);
                set_tensor(bnd, &v_t, 2);
                set_tensor(bnd, &d_y_flash, 3);
                set_tensor(bnd, attn_lse, 4);
                set_tensor(bnd, &delta, 5);
                set_tensor(bnd, &dq, 6);
                set_u32(bnd, b as u32, 7);
                set_u32(bnd, tlen as u32, 8);
                set_u32(bnd, h as u32, 9);
                set_u32(bnd, hkv as u32, 10);
                set_u32(bnd, d as u32, 11);
                set_f32(bnd, scale, 12);
            })?;
            let p2 = rt.pipeline(t2)?;
            dispatch_2d_tg(rt, &p2, k_blocks, b * hkv, BC, |bnd| {
                set_tensor(bnd, &q_t, 0);
                set_tensor(bnd, &k_t, 1);
                set_tensor(bnd, &v_t, 2);
                set_tensor(bnd, &d_y_flash, 3);
                set_tensor(bnd, attn_lse, 4);
                set_tensor(bnd, &delta, 5);
                set_tensor(bnd, &dk, 6);
                set_tensor(bnd, &dv, 7);
                set_u32(bnd, b as u32, 8);
                set_u32(bnd, tlen as u32, 9);
                set_u32(bnd, h as u32, 10);
                set_u32(bnd, hkv as u32, 11);
                set_u32(bnd, d as u32, 12);
                set_f32(bnd, scale, 13);
            })?;
        } else {
            // Audit 7 row-path variants. `fa_fast_row` is the
            // head-dim-specialized kernel (identical numerics; hoists
            // loop-invariant Q/dO + K/V and keeps accumulators in
            // registers). `fa_bf16_row` additionally reads bf16 Q/K/V,
            // which also matches the bf16 forward's taped LSE.
            // Both require head_dim == 32; otherwise fall back.
            // bf16 FA bwd is an approximation, not a consistency fix:
            // `model_fwd::use_bf16_flash` is hard-coded false, so the
            // forward always runs f32 flash. Gate on Bf16 precision so
            // an `--f32` run cannot silently break the 1e-4 bwd goldens.
            let bf16_ok = rt.precision() == PrecisionMode::Bf16;
            let specialized = crate::ab_flags::fa_fast_row() && d == 32;
            if crate::ab_flags::fa_fast_row() && d != 32 {
                // Correct but silent otherwise: the flag would look
                // enabled while the generic kernels ran.
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    eprintln!(
                        "warn: METAL_NATIVE_FA_FAST/FA_BF16 require head_dim == 32 \
                                 (this model has {d}); using generic row FA bwd"
                    );
                });
            }
            let use_bf16_fa = specialized && crate::ab_flags::fa_bf16_row() && bf16_ok;
            let (q_op, k_op, v_op) = if use_bf16_fa {
                (
                    crate::gemm::cast_f32_to_bf16(q)?,
                    crate::gemm::cast_f32_to_bf16(k)?,
                    crate::gemm::cast_f32_to_bf16(v_mixed)?,
                )
            } else {
                (q.clone(), k.clone(), v_mixed.clone())
            };
            let (n1, n2) = match (specialized, use_bf16_fa) {
                (true, true) => (
                    "flash_attn_bwd_dq_row_d32_bf16",
                    "flash_attn_bwd_dkv_row_d32_bf16",
                ),
                (true, false) => (
                    "flash_attn_bwd_dq_row_d32_f32",
                    "flash_attn_bwd_dkv_row_d32_f32",
                ),
                _ => ("flash_attn_bwd_dq_row_f32", "flash_attn_bwd_dkv_row_f32"),
            };
            let p1 = rt.pipeline(n1)?;
            dispatch_1d(rt, &p1, b * h * tlen, |bnd| {
                set_tensor(bnd, &q_op, 0);
                set_tensor(bnd, &k_op, 1);
                set_tensor(bnd, &v_op, 2);
                set_tensor(bnd, &d_y_flash, 3);
                set_tensor(bnd, attn_lse, 4);
                set_tensor(bnd, &delta, 5);
                set_tensor(bnd, &dq, 6);
                set_u32(bnd, b as u32, 7);
                set_u32(bnd, tlen as u32, 8);
                set_u32(bnd, h as u32, 9);
                set_u32(bnd, hkv as u32, 10);
                set_u32(bnd, d as u32, 11);
                set_f32(bnd, scale, 12);
            })?;
            let p2 = rt.pipeline(n2)?;
            dispatch_1d(rt, &p2, b * hkv * tlen, |bnd| {
                set_tensor(bnd, &q_op, 0);
                set_tensor(bnd, &k_op, 1);
                set_tensor(bnd, &v_op, 2);
                set_tensor(bnd, &d_y_flash, 3);
                set_tensor(bnd, attn_lse, 4);
                set_tensor(bnd, &delta, 5);
                set_tensor(bnd, &dk, 6);
                set_tensor(bnd, &dv, 7);
                set_u32(bnd, b as u32, 8);
                set_u32(bnd, tlen as u32, 9);
                set_u32(bnd, h as u32, 10);
                set_u32(bnd, hkv as u32, 11);
                set_u32(bnd, d as u32, 12);
                set_f32(bnd, scale, 13);
            })?;
        }
    }
    // Non-XSA: no dv_flash contribution.
    if let Some(ref dv_flash) = dv_flash {
        add_inplace(rt, &dv, dv_flash)?;
    }
    prof.lap(rt, "fa_dqdkv")?;

    // qkv_post bwd
    let use_ve = cfg.ve_scale_index(layer).is_some();
    let use_v0 = cfg.attn_local_idx(layer).unwrap_or(0) > 0;
    let ve = if let Some(ref ve) = lt.ve {
        ve.clone()
    } else {
        rt.alloc_tensor_f32(&[bt, kv])?
    };
    let dq_pre = rt.alloc_tensor_f32(&[bt, h * d])?;
    let dk_pre = rt.alloc_tensor_f32(&[bt, kv])?;
    let dv_pre = rt.alloc_tensor_f32(&[bt, kv])?;
    let dve = rt.alloc_tensor_f32(&[bt, kv])?;
    {
        let p = rt.pipeline("qkv_post_bwd_f32")?;
        let dq_f = reshape_view(&dq, &[bt, h * d]);
        let dk_f = reshape_view(&dk, &[bt, kv]);
        let dv_f = reshape_view(&dv, &[bt, kv]);
        dispatch_1d(rt, &p, bt, |bnd| {
            set_tensor(bnd, q_pre, 0);
            set_tensor(bnd, k_pre, 1);
            set_tensor(bnd, v_pre, 2);
            set_tensor(bnd, &ve, 3);
            set_tensor(bnd, v0_buf, 4);
            set_tensor(bnd, raw_v, 5);
            set_tensor(bnd, &bw.vr_lambda, 6);
            set_tensor(bnd, &bw.q_gain, 7);
            set_tensor(bnd, &w.rope_cos, 8);
            set_tensor(bnd, &w.rope_sin, 9);
            set_tensor(bnd, &dq_f, 10);
            set_tensor(bnd, &dk_f, 11);
            set_tensor(bnd, &dv_f, 12);
            set_tensor(bnd, &dq_pre, 13);
            set_tensor(bnd, &dk_pre, 14);
            set_tensor(bnd, &dv_pre, 15);
            set_tensor(bnd, &dve, 16);
            set_tensor(bnd, &dv0, 17);
            set_tensor(bnd, &grads.blocks[layer].vr_lambda, 18);
            set_tensor(bnd, &grads.blocks[layer].q_gain, 19);
            set_u32(bnd, b as u32, 20);
            set_u32(bnd, tlen as u32, 21);
            set_u32(bnd, h as u32, 22);
            set_u32(bnd, hkv as u32, 23);
            set_u32(bnd, d as u32, 24);
            set_u32(bnd, cfg.rope_dims as u32, 25);
            set_u32(bnd, use_ve as u32, 26);
            set_u32(bnd, use_v0 as u32, 27);
            set_f32(bnd, eps, 28);
        })?;
    }
    prof.lap(rt, "qkv_post")?;

    // Layer 0: dv0 from later layers lands on raw v (= v_pre path)
    if cfg.captures_v0(layer) {
        let dv0_flat = reshape_view(&dv0, &[bt, kv]);
        add_inplace(rt, &dv_pre, &dv0_flat)?;
    }

    // VE bwd via gather + GEMM (Phase D)
    if let Some(vi) = cfg.ve_scale_index(layer) {
        let de = cfg.ve_dim;
        let rows = rt.alloc_tensor_f32(&[bt, de])?;
        {
            let p = rt.pipeline("ve_gather_f32")?;
            dispatch_1d(rt, &p, bt, |bnd| {
                set_tensor(bnd, ids, 0);
                set_tensor(bnd, &w.ve_emb, 1);
                set_tensor(bnd, &rows, 2);
                set_u32(bnd, bt as u32, 3);
                set_u32(bnd, de as u32, 4);
            })?;
        }
        let h_pre = rt.alloc_tensor_f32(&[bt, kv])?;
        let ve_w = if use_persistent_bf16(rt, backend) {
            if let Some(ref bf) = w.bf16_banks {
                bf.ve_proj.clone()
            } else {
                w.ve_proj.clone()
            }
        } else {
            w.ve_proj.clone()
        };
        gemm_train(&rows, &ve_w, &h_pre, backend)?;
        let d_h = rt.alloc_tensor_f32(&[bt, kv])?;
        {
            let p = rt.pipeline("ve_scale_bwd_f32")?;
            dispatch_1d(rt, &p, bt * kv, |bnd| {
                set_tensor(bnd, &h_pre, 0);
                set_tensor(bnd, &dve, 1);
                set_tensor(bnd, &w.ve_scale, 2);
                set_tensor(bnd, &w.ve_layer_scales[vi], 3);
                set_tensor(bnd, &d_h, 4);
                set_tensor(bnd, &grads.ve_scale, 5);
                set_tensor(bnd, &grads.ve_layer_scales[vi], 6);
                set_u32(bnd, (bt * kv) as u32, 7);
            })?;
        }
        // Audit 7: d_h feeds dW + d_emb — cast once under BWD_CAST_ONCE.
        let d_h_op = bf16_once(rt, backend, &d_h)?;
        {
            let dw = grads.ve_proj.view(&[de, kv], 0);
            gemm_tn_accum_train(&rows, &d_h_op, &dw, backend)?;
        }
        {
            let d_emb = rt.alloc_tensor_f32(&[bt, de])?;
            gemm_nt_train(&d_h_op, &ve_w, &d_emb, backend)?;
            let p = rt.pipeline("ve_scatter_emb_f32")?;
            dispatch_1d(rt, &p, bt, |bnd| {
                set_tensor(bnd, ids, 0);
                set_tensor(bnd, &d_emb, 1);
                set_tensor(bnd, &grads.ve_emb, 2);
                set_u32(bnd, bt as u32, 3);
                set_u32(bnd, de as u32, 4);
            })?;
        }
    }
    prof.lap(rt, "ve")?;

    // Q/K/V GEMM bwd → d(attn_in): NT-accum into d_attn_in + TN-accum into banks.
    let (q_w, k_w, v_w) = if use_persistent_bf16(rt, backend) {
        if let Some(ref bf) = w.bf16_banks {
            (
                w.bank_matrix(rt, &bf.qo_bank, ai, c, c)?,
                w.bank_matrix(rt, &bf.kv_bank, ai, c, kv)?,
                w.bank_matrix(rt, &bf.kv_bank, n_attn + ai, c, kv)?,
            )
        } else {
            (
                w.bank_matrix(rt, &w.qo_bank, ai, c, c)?,
                w.bank_matrix(rt, &w.kv_bank, ai, c, kv)?,
                w.bank_matrix(rt, &w.kv_bank, n_attn + ai, c, kv)?,
            )
        }
    } else {
        (
            w.bank_matrix(rt, &w.qo_bank, ai, c, c)?,
            w.bank_matrix(rt, &w.kv_bank, ai, c, kv)?,
            w.bank_matrix(rt, &w.kv_bank, n_attn + ai, c, kv)?,
        )
    };
    let ai_flat = reshape_view(attn_in, &[bt, c]);
    let d_attn_in = rt.alloc_tensor_f32(&[bt, c])?;
    zero_tensor_device(&d_attn_in)?;
    {
        // Audit 7: ai_flat feeds three dW GEMMs and dq/dk/dv_pre feed
        // dX + dW each — cast once under BWD_CAST_ONCE (biggest
        // duplicate-cast site: 5 casts/attn layer saved).
        let ai_op = bf16_once(rt, backend, &ai_flat)?;
        let dq_op = bf16_once(rt, backend, &dq_pre)?;
        let dk_op = bf16_once(rt, backend, &dk_pre)?;
        let dv_op = bf16_once(rt, backend, &dv_pre)?;
        gemm_nt_accum_train(&dq_op, &q_w, &d_attn_in, backend)?;
        let dw_q = grads.qo_bank.view(&[c, c], ai * c * c);
        gemm_tn_accum_train(&ai_op, &dq_op, &dw_q, backend)?;
        gemm_nt_accum_train(&dk_op, &k_w, &d_attn_in, backend)?;
        let dw_k = grads.kv_bank.view(&[c, kv], ai * c * kv);
        gemm_tn_accum_train(&ai_op, &dk_op, &dw_k, backend)?;
        gemm_nt_accum_train(&dv_op, &v_w, &d_attn_in, backend)?;
        let dw_v = grads.kv_bank.view(&[c, kv], (n_attn + ai) * c * kv);
        gemm_tn_accum_train(&ai_op, &dv_op, &dw_v, backend)?;
    }
    prof.lap(rt, "qkv_gemms")?;
    Ok(d_attn_in)
}

fn mamba2_bwd_rust(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    grads: &Grads,
    layer: usize,
    attn_in: &Tensor,
    d_attn_out: &Tensor,
    backend: GemmBackend,
    b: usize,
    tlen: usize,
    c: usize,
    bt: usize,
    lt: &LayerTape,
) -> Result<Tensor, String> {
    let cfg = &w.cfg;
    let d_inner = cfg.mamba_d_inner();
    let d_state = cfg.d_state;
    let n_head = cfg.mamba_n_head();
    let head_dim = cfg.mamba_head_dim();
    let conv_dim = cfg.mamba_conv_dim();
    let in_out = cfg.mamba_in_proj_out();
    let eps = 1e-6f32;

    let out_proj = w.mamba_out_proj.as_ref().ok_or("mamba_out_proj")?;
    let in_proj = w.mamba_in_proj.as_ref().ok_or("mamba_in_proj")?;
    let conv_w = w
        .mamba_conv1d_weight
        .as_ref()
        .ok_or("mamba_conv1d_weight")?;
    let a_log = w.mamba_a_log.as_ref().ok_or("mamba_a_log")?;
    let d_param = w.mamba_d.as_ref().ok_or("mamba_d")?;
    let dt_bias = w.mamba_dt_bias.as_ref().ok_or("mamba_dt_bias")?;
    let norm_w = w.mamba_norm.as_ref().ok_or("mamba_norm")?;

    let mi = cfg.mixer_local_idx(layer);

    let out_w = w.bank_matrix(rt, out_proj, mi, d_inner, c)?;
    let in_w = w.bank_matrix(rt, in_proj, mi, c, in_out)?;
    let layer_conv_w = w.bank_matrix(rt, conv_w, mi, conv_dim, cfg.d_conv)?;
    let layer_a_log = a_log.view(&[n_head], mi * n_head);
    let layer_d = d_param.view(&[n_head], mi * n_head);
    let layer_dt_bias = dt_bias.view(&[n_head], mi * n_head);
    let layer_norm = norm_w.view(&[d_inner], mi * d_inner);

    let y_gated = lt.mamba_y_pre_out.as_ref().ok_or("mamba_y_pre_out")?;
    let y_norm = lt.mamba_y_norm.as_ref().ok_or("mamba_y_norm")?;
    let y_flat = lt.mamba_y_flat.as_ref().ok_or("mamba_y_flat")?;
    let z_pre = lt.mamba_z_pre_silu.as_ref().ok_or("mamba_z_pre_silu")?;
    let z_gate = rt.alloc_tensor_f32(&[bt, d_inner])?;
    crate::ssm_glue::silu_fwd(rt, z_pre, &z_gate)?;

    let d_out_flat = reshape_view(d_attn_out, &[bt, c]);
    let d_y_gated = rt.alloc_tensor_f32(&[bt, d_inner])?;
    gemm_nt_train(&d_out_flat, &out_w, &d_y_gated, backend)?;
    let dw_out = grads
        .mamba_out_proj
        .as_ref()
        .unwrap()
        .view(&[d_inner, c], mi * d_inner * c);
    gemm_tn_accum_train(y_gated, &d_out_flat, &dw_out, backend)?;

    let d_y_norm = rt.alloc_tensor_f32(&[bt, d_inner])?;
    let d_z_silu = rt.alloc_tensor_f32(&[bt, d_inner])?;
    mul_bwd(rt, y_norm, &z_gate, &d_y_gated, &d_y_norm, &d_z_silu)?;

    let d_z = rt.alloc_tensor_f32(&[bt, d_inner])?;
    silu_bwd(rt, z_pre, &d_z_silu, &d_z)?;

    let d_y_flat = rt.alloc_tensor_f32(&[bt, d_inner])?;
    let d_norm_w = grads
        .mamba_norm
        .as_ref()
        .unwrap()
        .view(&[d_inner], mi * d_inner);
    rms_norm_weight_bwd(
        rt,
        y_flat,
        &layer_norm,
        &d_y_norm,
        &d_y_flat,
        &d_norm_w,
        bt,
        d_inner,
        eps,
    )?;

    let xs_heads = lt.mamba_xs.as_ref().ok_or("mamba_xs")?;
    let h_states = lt.mamba_h_states.as_ref().ok_or("mamba_h_states")?;
    let x_scaled = lt.mamba_x_scaled.as_ref().ok_or("mamba_x_scaled")?;
    let bm = lt.mamba_bm.as_ref().ok_or("mamba_bm")?;
    let cm = lt.mamba_cm.as_ref().ok_or("mamba_cm")?;
    let log_da = lt.mamba_log_da.as_ref().ok_or("mamba_log_da")?;
    let dt = lt.mamba_dt.as_ref().ok_or("mamba_dt")?;

    let d_ssd_y = unflatten_heads(
        &reshape_view(&d_y_flat, &[b, tlen, d_inner]),
        n_head,
        head_dim,
    );
    let dxs_heads = rt.alloc_tensor_f32(&[b, tlen, n_head, head_dim])?;
    zero_tensor_device(&dxs_heads)?;
    let dd = rt.alloc_tensor_f32(&[n_head])?;
    mamba2_d_skip_bwd(
        rt, &d_ssd_y, xs_heads, &layer_d, &dxs_heads, &dd, b, tlen, n_head, head_dim,
    )?;
    {
        let dd_full = grads.mamba_d.as_ref().unwrap().view(&[n_head], mi * n_head);
        add_inplace(rt, &dd_full, &dd)?;
    }

    let bwd = mamba2_bwd(rt, x_scaled, bm, cm, log_da, h_states, &d_ssd_y)?;

    let ddt = rt.alloc_tensor_f32(&[b, tlen, n_head])?;
    zero_tensor_device(&ddt)?;
    mamba2_x_scaled_bwd(
        rt,
        &bwd.grad_x_scaled,
        xs_heads,
        dt,
        &dxs_heads,
        &ddt,
        b,
        tlen,
        n_head,
        head_dim,
    )?;

    let da_log = rt.alloc_tensor_f32(&[n_head])?;
    mamba2_log_da_bwd(
        rt,
        &bwd.grad_log_da,
        dt,
        &layer_a_log,
        &ddt,
        &da_log,
        b,
        tlen,
        n_head,
    )?;
    {
        let da_full = grads
            .mamba_a_log
            .as_ref()
            .unwrap()
            .view(&[n_head], mi * n_head);
        add_inplace(rt, &da_full, &da_log)?;
    }

    let dt_raw = lt.mamba_dt_raw.as_ref().ok_or("mamba_dt_raw")?;
    let ddt_raw = rt.alloc_tensor_f32(&[b, tlen, n_head])?;
    let ddt_bias = rt.alloc_tensor_f32(&[n_head])?;
    softplus_bias_bwd(
        rt,
        dt_raw,
        &layer_dt_bias,
        &ddt,
        &ddt_raw,
        &ddt_bias,
        b,
        tlen,
        n_head,
    )?;
    {
        let db = grads
            .mamba_dt_bias
            .as_ref()
            .unwrap()
            .view(&[n_head], mi * n_head);
        add_inplace(rt, &db, &ddt_bias)?;
    }

    let dxs_flat = flatten_heads(&dxs_heads, d_inner);
    let dbm_flat = reshape_view(&bwd.grad_b_h, &[bt, d_state]);
    let dcm_flat = reshape_view(&bwd.grad_c_h, &[bt, d_state]);

    let xbc_post = lt.mamba_xbc_post.as_ref().ok_or("mamba_xbc_post")?;
    let dxbc_silu = rt.alloc_tensor_f32(&[bt, conv_dim])?;
    zero_tensor_device(&dxbc_silu)?;
    accum_slice_grad(rt, &dxbc_silu, &dxs_flat, bt, conv_dim, d_inner, 0)?;
    accum_slice_grad(rt, &dxbc_silu, &dbm_flat, bt, conv_dim, d_state, d_inner)?;
    accum_slice_grad(
        rt,
        &dxbc_silu,
        &dcm_flat,
        bt,
        conv_dim,
        d_state,
        d_inner + d_state,
    )?;

    let dxbc_conv = rt.alloc_tensor_f32(&[bt, conv_dim])?;
    silu_bwd_store(rt, xbc_post, &dxbc_silu, &dxbc_conv)?;

    let xbc_pre = lt.mamba_xbc_pre.as_ref().ok_or("mamba_xbc_pre")?;
    let conv_bwd = mamba2_conv1d_bwd(
        rt,
        xbc_pre,
        &layer_conv_w,
        &reshape_view(&dxbc_conv, &[b, tlen, conv_dim]),
    )?;
    {
        let dw = grads
            .mamba_conv1d_weight
            .as_ref()
            .unwrap()
            .view(&[conv_dim, cfg.d_conv], mi * conv_dim * cfg.d_conv);
        add_inplace(rt, &dw, &conv_bwd.grad_w)?;
        let db = grads
            .mamba_conv1d_bias
            .as_ref()
            .unwrap()
            .view(&[conv_dim], mi * conv_dim);
        add_inplace(rt, &db, &conv_bwd.grad_bias)?;
    }

    let dzxbcdt = rt.alloc_tensor_f32(&[bt, in_out])?;
    zero_tensor_device(&dzxbcdt)?;
    accum_slice_grad(rt, &dzxbcdt, &d_z, bt, in_out, d_inner, 0)?;
    accum_slice_grad(
        rt,
        &dzxbcdt,
        &reshape_view(&conv_bwd.grad_x, &[bt, conv_dim]),
        bt,
        in_out,
        conv_dim,
        d_inner,
    )?;
    accum_slice_grad(
        rt,
        &dzxbcdt,
        &reshape_view(&ddt_raw, &[bt, n_head]),
        bt,
        in_out,
        n_head,
        d_inner + conv_dim,
    )?;

    let d_attn_in = rt.alloc_tensor_f32(&[bt, c])?;
    zero_tensor_device(&d_attn_in)?;
    let ai_flat = reshape_view(attn_in, &[bt, c]);
    gemm_nt_accum_train(&dzxbcdt, &in_w, &d_attn_in, backend)?;
    let dw_in = grads
        .mamba_in_proj
        .as_ref()
        .unwrap()
        .view(&[c, in_out], mi * c * in_out);
    gemm_tn_accum_train(&ai_flat, &dzxbcdt, &dw_in, backend)?;

    Ok(reshape_view(&d_attn_in, &[b, tlen, c]))
}

fn mingru_bwd_rust(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    grads: &Grads,
    layer: usize,
    attn_in: &Tensor,
    d_attn_out: &Tensor,
    backend: GemmBackend,
    v0: Option<&Tensor>,
    b: usize,
    tlen: usize,
    c: usize,
    kv: usize,
    _hkv: usize,
    _d: usize,
    bt: usize,
    dv0: Tensor,
    lt: &LayerTape,
) -> Result<Tensor, String> {
    let cfg = &w.cfg;
    let hid = cfg.mingru_hidden();
    let mi = cfg.mixer_local_idx(layer);
    let use_vr = cfg.value_residual;
    let bw = &w.blocks[layer];
    let to_z = w.mingru_to_z.as_ref().ok_or("mingru_to_z")?;
    let to_h = w.mingru_to_h.as_ref().ok_or("mingru_to_h")?;
    let out_bank = w.mingru_out.as_ref().ok_or("mingru_out")?;

    let z_w = w.bank_matrix(rt, to_z, mi, c, hid)?;
    let h_w = w.bank_matrix(rt, to_h, mi, c, hid)?;
    let out_w = w.bank_matrix(rt, out_bank, mi, hid, c)?;

    let h_out = lt.mingru_h_out.as_ref().ok_or("mingru_h_out")?;
    let z_raw = lt.mingru_z_raw.as_ref().ok_or("mingru_z_raw")?;
    let h_pre = lt.mingru_h_pre.as_ref().ok_or("mingru_h_pre")?;
    let h_raw = lt.mingru_h_raw.as_ref().ok_or("mingru_h_raw")?;

    let d_out_flat = reshape_view(d_attn_out, &[bt, c]);
    let d_h_flat = rt.alloc_tensor_f32(&[bt, hid])?;
    gemm_nt_train(&d_out_flat, &out_w, &d_h_flat, backend)?;
    let dw_out = grads
        .mingru_out
        .as_ref()
        .unwrap()
        .view(&[hid, c], mi * hid * c);
    gemm_tn_accum_train(
        &reshape_view(h_out, &[bt, hid]),
        &d_out_flat,
        &dw_out,
        backend,
    )?;

    let d_h_3d = reshape_view(&d_h_flat, &[b, tlen, hid]);
    let (d_z, d_h_pre) = mingru_bwd(rt, z_raw, h_pre, h_out, &d_h_3d)?;

    let d_h_pre_flat = reshape_view(&d_h_pre, &[bt, hid]);
    let h_raw_flat = reshape_view(h_raw, &[bt, hid]);
    let d_h_raw_flat = rt.alloc_tensor_f32(&[bt, hid])?;
    zero_tensor_device(&d_h_raw_flat)?;

    let use_v0 = use_vr && v0.is_some() && cfg.mixer_local_idx(layer) > 0;
    if use_vr {
        let v0_up_flat = if use_v0 {
            reshape_view(
                lt.mingru_v0_up.as_ref().ok_or("mingru_v0_up tape")?,
                &[bt, hid],
            )
        } else {
            rt.alloc_tensor_f32(&[bt, hid])?
        };
        let d_v0_up = rt.alloc_tensor_f32(&[bt, hid])?;
        zero_tensor_device(&d_v0_up)?;
        mingru_vr_blend_bwd(
            rt,
            &d_h_pre_flat,
            &h_raw_flat,
            &v0_up_flat,
            &bw.vr_lambda,
            &d_h_raw_flat,
            &d_v0_up,
            &grads.blocks[layer].vr_lambda,
            use_v0,
        )?;
        if use_v0 {
            let v0_up_bank = w.mingru_v0_up.as_ref().ok_or("mingru_v0_up")?;
            let v0_up_w = w.bank_matrix(rt, v0_up_bank, mi, kv, hid)?;
            let dv0_flat = reshape_view(&dv0, &[bt, kv]);
            gemm_nt_accum_train(&d_v0_up, &v0_up_w, &dv0_flat, backend)?;
            let v0_flat = reshape_view(v0.unwrap(), &[bt, kv]);
            let dw_v0 = grads
                .mingru_v0_up
                .as_ref()
                .unwrap()
                .view(&[kv, hid], mi * kv * hid);
            gemm_tn_accum_train(&v0_flat, &d_v0_up, &dw_v0, backend)?;
        }
    } else {
        // No VR: h_pre == h_raw path.
        let p = rt.pipeline("copy_f32")?;
        dispatch_1d(rt, &p, bt * hid, |bnd| {
            set_tensor(bnd, &d_h_pre_flat, 0);
            set_tensor(bnd, &d_h_raw_flat, 1);
            set_u32(bnd, (bt * hid) as u32, 2);
        })?;
    }

    let d_z_flat = reshape_view(&d_z, &[bt, hid]);
    let d_attn_in = rt.alloc_tensor_f32(&[bt, c])?;
    zero_tensor_device(&d_attn_in)?;
    let ai_flat = reshape_view(attn_in, &[bt, c]);
    gemm_nt_accum_train(&d_z_flat, &z_w, &d_attn_in, backend)?;
    gemm_nt_accum_train(&d_h_raw_flat, &h_w, &d_attn_in, backend)?;
    let dw_z = grads
        .mingru_to_z
        .as_ref()
        .unwrap()
        .view(&[c, hid], mi * c * hid);
    let dw_h = grads
        .mingru_to_h
        .as_ref()
        .unwrap()
        .view(&[c, hid], mi * c * hid);
    gemm_tn_accum_train(&ai_flat, &d_z_flat, &dw_z, backend)?;
    gemm_tn_accum_train(&ai_flat, &d_h_raw_flat, &dw_h, backend)?;

    if cfg.captures_v0(layer) && use_vr {
        let dv0_flat = reshape_view(&dv0, &[bt, kv]);
        let v_proj = w.mingru_v_proj.as_ref().ok_or("mingru_v_proj")?;
        let v_w = w.bank_matrix(rt, v_proj, mi, c, kv)?;
        gemm_nt_accum_train(&dv0_flat, &v_w, &d_attn_in, backend)?;
        let dw_v = grads
            .mingru_v_proj
            .as_ref()
            .unwrap()
            .view(&[c, kv], mi * c * kv);
        gemm_tn_accum_train(&ai_flat, &dv0_flat, &dw_v, backend)?;
    }

    Ok(reshape_view(&d_attn_in, &[b, tlen, c]))
}
