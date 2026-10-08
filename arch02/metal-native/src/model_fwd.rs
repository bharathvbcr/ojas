//! Phase 1 forward path: per-op Metal kernels + GEMM.
//!
//! Phase H: under `PrecisionMode::Bf16`, linear/GEMM ops use bf16 TensorOps
//! (f32 accumulate) via `gemm_train`; softcap+CE stay f32. Residual-stream
//! megakernels write bf16 norm outs (`resid_mix_rms_norm_scale_bf16` /
//! `residual_scale_add_rms_norm_scale_bf16`) when available; tape intermediates
//! stay f32. Flash uses bf16-input / f32-accum FA-2 with LSE when TensorOps is
//! present (optional `--flash-tensorops` probe). Activation GEMM weight casts
//! are reused (persistent bf16 operands).
//!
//! Phase B: tape stashes are buffer views/moves. Audit 6 made qkv_post / xsa /
//! mlp_act out-of-place so pre-values no longer need deep_copy.

use std::sync::Arc;

use crate::dispatch::{dispatch_1d, dispatch_2d_tg, set_f32, set_tensor, set_u32};
use crate::gemm::{cast_f32_to_bf16, gemm_train, select_backend, GemmBackend};
use crate::mixers::{mamba2_conv1d_fwd, mamba2_fwd, mingru_fwd, mingru_vr_blend_fwd};
use crate::runtime::{GpuRuntime, PrecisionMode};
use crate::ssm_glue::{
    mamba2_d_skip_fwd, mamba2_log_da, mamba2_x_scaled, mul_fwd, reshape_heads, rms_norm_weight_fwd,
    silu_fwd, silu_fwd_store, slice_last_dim, softplus_bias_fwd,
};
use crate::tape::Tape;
use crate::tensor::Tensor;
use crate::weights::{ModelConfig, Weights};
use objc2_metal::MTLBuffer;

pub struct ForwardOutputs {
    pub stem: Tensor,
    pub v0: Tensor,
    pub layer_attn_out: Vec<Tensor>,
    pub layer_mlp_out: Vec<Tensor>,
    pub layer_x: Vec<Tensor>,
    pub layer_after_skip: Vec<Option<Tensor>>,
    pub final_norm: Tensor,
    pub logits_pre: Tensor,
    pub logits_post: Tensor,
    /// Host-visible loss (sync only when read; train defers via `loss_device`).
    pub loss: f32,
    /// On-device mean CE (read on log steps / parity).
    pub loss_device: Tensor,
}

/// Run full sota-toy forward; stash activations on `tape` for Phase 2 bwd.
pub fn forward_f32(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    input_ids_i32: &[i32],
    target_ids_i32: &[i32],
    tape: &mut Tape,
) -> Result<ForwardOutputs, String> {
    let cfg = &w.cfg;
    let b = cfg.batch;
    let t = cfg.seq_len;
    let bt = b * t;
    let ids = upload_i32(rt, &[b, t], input_ids_i32)?;
    let tgts = upload_i32(rt, &[bt], target_ids_i32)?;
    forward_f32_uploaded(rt, w, ids, tgts, tape)
}

/// Forward with pre-uploaded (e.g. ping-pong) id/target buffers.
pub fn forward_f32_uploaded(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    ids: Tensor,
    tgts: Tensor,
    tape: &mut Tape,
) -> Result<ForwardOutputs, String> {
    let cfg = &w.cfg;
    let b = cfg.batch;
    let t = cfg.seq_len;
    let c = cfg.model_dim;
    let bt = b * t;
    let backend = select_backend(rt);
    let eps = cfg.f32_eps();

    tape.clear_activations();
    tape.step += 1;

    tape.input_ids = Some(ids.clone());
    tape.target_ids = Some(tgts.clone());

    // ---- stem ----
    let (stem, stem_pre, stem_post) = stem_fwd(rt, w, &ids)?;
    tape.stem_pre_norm = Some(stem_pre);
    tape.stem_post_norm = Some(stem_post);
    // One buffer, four views (stem is never overwritten in place).
    tape.stem = Some(stem.clone());
    tape.x0 = Some(stem.clone());
    let x0 = stem.clone();
    let mut x = stem.clone();

    let n_enc = cfg.num_layers / 2;
    let mut skips: Vec<Tensor> = Vec::new();
    let mut layer_attn_out = Vec::new();
    let mut layer_mlp_out = Vec::new();
    let mut layer_x = Vec::new();
    let mut layer_after_skip: Vec<Option<Tensor>> = vec![None; cfg.num_layers];
    let mut v0: Option<Tensor> = None;

    // Audit 8: METAL_NATIVE_FWD_PROFILE=1 — same synced section timing as the
    // backward profiler that located the FA bwd bottleneck.
    let mut fprof =
        crate::model_bwd::BwdProf::new_labeled(rt, crate::ab_flags::fwd_profile(), "fwd_profile")?;
    fprof.lap(rt, "stem")?;

    for layer in 0..cfg.num_layers {
        let is_decoder = layer >= n_enc;
        let mut skip_fused: Option<(Tensor, Tensor, Tensor)> = None; // after_skip, x_in, attn_in
        if is_decoder {
            let skip_i = layer - n_enc;
            let skip = skips.pop().ok_or("skip stack empty")?;
            let sw_t = w.skip_weights.view(&[c], skip_i * c);
            // P2: fuse skip_add + resid_mix + RMSNorm*scale (f32 or bf16 stream twin).
            let use_bf16_stream = rt.precision() == PrecisionMode::Bf16
                && rt.pipeline("skip_resid_mix_rms_norm_scale_bf16").is_ok();
            let can_fuse = (rt.precision() == PrecisionMode::F32
                && rt.pipeline("skip_resid_mix_rms_norm_scale_f32").is_ok())
                || use_bf16_stream;
            if can_fuse {
                let bw = &w.blocks[layer];
                let ln = cfg.ln_scale_factor(layer);
                let after_skip = rt.alloc_tensor_f32(&x.shape())?;
                let x_in = rt.alloc_tensor_f32(&x.shape())?;
                let attn_in = if use_bf16_stream {
                    rt.alloc_tensor_bf16(&x.shape())?
                } else {
                    rt.alloc_tensor_f32(&x.shape())?
                };
                let pname = if use_bf16_stream {
                    "skip_resid_mix_rms_norm_scale_bf16"
                } else {
                    "skip_resid_mix_rms_norm_scale_f32"
                };
                let p = rt.pipeline(pname)?;
                dispatch_1d(rt, &p, bt, |bnd| {
                    set_tensor(bnd, &x, 0);
                    set_tensor(bnd, &skip, 1);
                    set_tensor(bnd, &sw_t, 2);
                    set_tensor(bnd, &x0, 3);
                    set_tensor(bnd, &bw.resid_mix, 4);
                    set_tensor(bnd, &after_skip, 5);
                    set_tensor(bnd, &x_in, 6);
                    set_tensor(bnd, &attn_in, 7);
                    set_u32(bnd, bt as u32, 8);
                    set_u32(bnd, c as u32, 9);
                    set_f32(bnd, eps, 10);
                    set_f32(bnd, ln, 11);
                })?;
                layer_after_skip[layer] = Some(after_skip.clone());
                if let Some(lt) = tape.layer.get_mut(layer) {
                    lt.x_in = Some(x_in.clone());
                    // P1e: keep bf16 attn_in on tape for bwd dW GEMMs (no cast tax).
                    lt.attn_in = Some(attn_in.clone());
                    lt.after_skip = Some(after_skip.clone());
                }
                x = after_skip;
                skip_fused = Some((x.clone(), x_in, attn_in));
            } else {
                x = skip_add(rt, &x, &skip, &sw_t)?;
                layer_after_skip[layer] = Some(x.clone());
                if let Some(lt) = tape.layer.get_mut(layer) {
                    lt.after_skip = Some(x.clone());
                }
            }
        }

        if let Some(lt) = tape.layer.get_mut(layer) {
            lt.x_stream = Some(x.clone());
        }
        fprof.lap(rt, "skip_resid_glue")?;

        let (attn_out, mlp_out, x_new, raw_v) = block_fwd(
            rt,
            w,
            layer,
            &x,
            &x0,
            &ids,
            v0.as_ref(),
            backend,
            Some(tape),
            skip_fused.map(|(_, xin, attn)| (xin, attn)),
            Some(&mut fprof),
        )?;
        if cfg.captures_v0(layer) {
            v0 = raw_v;
            if let Some(ref v) = v0 {
                tape.v0 = Some(v.clone());
            }
        }
        // ForwardOutputs for parity: Arc clones, not deep_copies.
        layer_attn_out.push(attn_out.clone());
        layer_mlp_out.push(mlp_out.clone());
        x = x_new;
        layer_x.push(x.clone());
        if !is_decoder {
            // Keep a live stack for decoder skip-add *and* a tape copy for bwd
            // (decoder pops the live stack, so we must stash before that).
            tape.skips.push(x.clone());
            skips.push(x.clone());
        }
    }

    // ---- head ----
    fprof.lap(rt, "layers_tail")?;
    tape.pre_final_norm = Some(x.clone());
    let final_norm = rms_norm(rt, &x, bt, c, eps)?;
    tape.final_norm = Some(final_norm.clone());

    // logits GEMM then fused softcap+CE (Phase F: drop logits_pre tape stash).
    let x_flat = reshape_view(&final_norm, &[bt, c]);
    let logits_pre = rt.alloc_tensor_f32(&[bt, cfg.vocab_size])?;
    gemm_train(&x_flat, &w.tok_emb_t, &logits_pre, backend)?;
    let logits_post = rt.alloc_tensor_f32(&[bt, cfg.vocab_size])?;
    let row_loss = rt.alloc_tensor_f32(&[bt])?;
    {
        let p = rt.pipeline("softcap_ce_row_f32")?;
        dispatch_1d(rt, &p, bt, |bnd| {
            set_tensor(bnd, &logits_pre, 0);
            set_tensor(bnd, &tgts, 1);
            set_tensor(bnd, &logits_post, 2);
            set_tensor(bnd, &row_loss, 3);
            set_u32(bnd, bt as u32, 4);
            set_u32(bnd, cfg.vocab_size as u32, 5);
            set_f32(bnd, cfg.logit_softcap, 6);
        })?;
    }
    let loss_device = rt.alloc_tensor_f32(&[1])?;
    {
        let p2 = rt.pipeline("mean_reduce_f32")?;
        let tpg = 256usize;
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&p2);
            set_tensor(bnd, &row_loss, 0);
            set_tensor(bnd, &loss_device, 1);
            set_u32(bnd, bt as u32, 2);
            bnd.dispatch(
                crate::runtime::mtl_size(1, 1, 1),
                crate::runtime::mtl_size(tpg, 1, 1),
            );
            Ok(())
        })?;
    }
    tape.logits_pre = None;
    fprof.lap(rt, "head")?;
    fprof.report();
    tape.logits_post = Some(reshape_view(&logits_post, &[b, t, cfg.vocab_size]));
    tape.loss = Some(loss_device.clone());
    let loss = f32::NAN;

    let v0_out = if let Some(v) = v0 {
        v
    } else {
        let dummy = rt.alloc_tensor_f32(&[b, t, cfg.num_kv_heads, cfg.head_dim])?;
        crate::model_bwd::zero_tensor_device(&dummy)?;
        tape.v0 = Some(dummy.clone());
        dummy
    };

    Ok(ForwardOutputs {
        stem,
        v0: v0_out,
        layer_attn_out,
        layer_mlp_out,
        layer_x,
        layer_after_skip,
        final_norm,
        logits_pre: reshape_view(&logits_pre, &[b, t, cfg.vocab_size]),
        logits_post: reshape_view(&logits_post, &[b, t, cfg.vocab_size]),
        loss,
        loss_device,
    })
}

impl ForwardOutputs {
    /// Synchronize and read mean CE (parity / log steps).
    pub fn read_loss(&self, rt: &GpuRuntime) -> Result<f32, String> {
        rt.synchronize()?;
        Ok(self.loss_device.buffer.read_f32()[0])
    }
}

/// Eval / inference forward: no tape stashes (skips deep_copies for in-place
/// ops), no CE mean sync. Returns softcapped logits `[B, T, V]`.
pub fn forward_infer_f32(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    input_ids_i32: &[i32],
) -> Result<Tensor, String> {
    let cfg = &w.cfg;
    let b = cfg.batch;
    let t = cfg.seq_len;
    let c = cfg.model_dim;
    let bt = b * t;
    let backend = select_backend(rt);
    let eps = cfg.f32_eps();

    let ids = upload_i32(rt, &[b, t], input_ids_i32)?;
    let (stem, _pre, _post) = stem_fwd(rt, w, &ids)?;
    let x0 = stem.clone();
    let mut x = stem;

    let n_enc = cfg.num_layers / 2;
    let mut skips: Vec<Tensor> = Vec::new();
    let mut v0: Option<Tensor> = None;

    for layer in 0..cfg.num_layers {
        let is_decoder = layer >= n_enc;
        if is_decoder {
            let skip_i = layer - n_enc;
            let skip = skips.pop().ok_or("skip stack empty")?;
            let sw_t = w.skip_weights.view(&[c], skip_i * c);
            x = skip_add(rt, &x, &skip, &sw_t)?;
        }

        let (_attn_out, _mlp_out, x_new, raw_v) = block_fwd(
            rt,
            w,
            layer,
            &x,
            &x0,
            &ids,
            v0.as_ref(),
            backend,
            None,
            None,
            None,
        )?;
        if layer == 0 {
            v0 = raw_v;
        }
        x = x_new;
        if !is_decoder {
            skips.push(x.clone());
        }
    }

    let final_norm = rms_norm(rt, &x, bt, c, eps)?;
    let x_flat = reshape_view(&final_norm, &[bt, c]);
    let logits_pre = rt.alloc_tensor_f32(&[bt, cfg.vocab_size])?;
    gemm_train(&x_flat, &w.tok_emb_t, &logits_pre, backend)?;
    let logits_post = softcap(rt, &logits_pre, cfg.logit_softcap)?;
    Ok(reshape_view(&logits_post, &[b, t, cfg.vocab_size]))
}

fn block_fwd(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    layer: usize,
    x: &Tensor,
    x0: &Tensor,
    ids: &Tensor,
    v0: Option<&Tensor>,
    backend: GemmBackend,
    mut tape: Option<&mut Tape>,
    // When set (decoder skip_resid fuse), skip resid_mix megakernel.
    pre_attn: Option<(Tensor, Tensor)>,
    // Audit 8: optional section profiler (METAL_NATIVE_FWD_PROFILE).
    mut fprof: Option<&mut crate::model_bwd::BwdProf>,
) -> Result<(Tensor, Tensor, Tensor, Option<Tensor>), String> {
    let cfg = &w.cfg;

    let b = cfg.batch;
    let tlen = cfg.seq_len;
    let c = cfg.model_dim;
    let bt = b * tlen;
    let h = cfg.num_heads;
    let hkv = cfg.num_kv_heads;
    let d = cfg.head_dim;
    let kv = cfg.kv_dim();
    let mlp = cfg.mlp_dim;
    let eps = cfg.f32_eps();
    let bw = &w.blocks[layer];

    // resid_mix + RMSNorm*ln_scale (fused megakernel; still writes x_in for tape/bwd)
    let ln = cfg.ln_scale_factor(layer);
    let use_bf16_stream = rt.precision() == PrecisionMode::Bf16
        && rt.pipeline("resid_mix_rms_norm_scale_bf16").is_ok();
    let (x_in, attn_in) = if let Some((xin, attn)) = pre_attn {
        (xin, attn)
    } else {
        let x_in = rt.alloc_tensor_f32(&x.shape())?;
        let attn_in = if use_bf16_stream {
            rt.alloc_tensor_bf16(&x.shape())?
        } else {
            rt.alloc_tensor_f32(&x.shape())?
        };
        {
            let pname = if use_bf16_stream {
                "resid_mix_rms_norm_scale_bf16"
            } else {
                "resid_mix_rms_norm_scale_f32"
            };
            let p = rt.pipeline(pname)?;
            dispatch_1d(rt, &p, bt, |bnd| {
                set_tensor(bnd, x, 0);
                set_tensor(bnd, x0, 1);
                set_tensor(bnd, &bw.resid_mix, 2);
                set_tensor(bnd, &x_in, 3);
                set_tensor(bnd, &attn_in, 4);
                set_u32(bnd, bt as u32, 5);
                set_u32(bnd, c as u32, 6);
                set_f32(bnd, eps, 7);
                set_f32(bnd, ln, 8);
            })?;
        }
        (x_in, attn_in)
    };
    if let Some(lt) = tape.as_mut().and_then(|t| t.layer.get_mut(layer)) {
        lt.x_in = Some(x_in.clone());
        // P1e: keep stream bf16 attn_in for bwd dW (resid bwd uses x_in, not attn_in).
        lt.attn_in = Some(attn_in.clone());
    }

    let (attn_out, raw_v) = match cfg.layer_mixer(layer) {
        crate::weights::MixerKind::Attention => attention_fwd(
            rt,
            w,
            layer,
            &attn_in,
            ids,
            v0,
            backend,
            tape.as_deref_mut(),
            b,
            tlen,
            c,
            kv,
            h,
            hkv,
            d,
            bt,
            bw,
            use_bf16_stream,
            fprof.as_deref_mut(),
        )?,
        crate::weights::MixerKind::Mamba2 => mamba2_fwd_rust(
            rt,
            w,
            layer,
            &attn_in,
            ids,
            backend,
            tape.as_deref_mut(),
            b,
            tlen,
            c,
        )?,
        crate::weights::MixerKind::MinGRU => mingru_fwd_rust(
            rt,
            w,
            layer,
            &attn_in,
            ids,
            v0,
            backend,
            tape.as_deref_mut(),
            b,
            tlen,
            c,
            kv,
            hkv,
            d,
            bw,
        )?,
    };
    if let Some(lt) = tape.as_mut().and_then(|t| t.layer.get_mut(layer)) {
        lt.attn_out = Some(attn_out.clone());
    }
    if let Some(p) = fprof.as_deref_mut() {
        p.lap(rt, "mixer_other")?;
    }

    // attn residual + MLP RMSNorm*ln_scale (fused; writes x_mid for tape)
    let mut x_out = rt.alloc_tensor_f32(&x_in.shape())?;
    let mlp_in = if use_bf16_stream {
        rt.alloc_tensor_bf16(&x_in.shape())?
    } else {
        rt.alloc_tensor_f32(&x_in.shape())?
    };
    {
        let pname = if use_bf16_stream {
            "residual_scale_add_rms_norm_scale_bf16"
        } else {
            "residual_scale_add_rms_norm_scale_f32"
        };
        let p = rt.pipeline(pname)?;
        dispatch_1d(rt, &p, bt, |bnd| {
            set_tensor(bnd, &x_in, 0);
            set_tensor(bnd, &attn_out, 1);
            set_tensor(bnd, &bw.attn_scale, 2);
            set_tensor(bnd, &x_out, 3);
            set_tensor(bnd, &mlp_in, 4);
            set_u32(bnd, bt as u32, 5);
            set_u32(bnd, c as u32, 6);
            set_f32(bnd, eps, 7);
            set_f32(bnd, ln, 8);
        })?;
    }
    if let Some(lt) = tape.as_mut().and_then(|t| t.layer.get_mut(layer)) {
        lt.x_mid = Some(x_out.clone());
        // P1e: keep stream bf16 mlp_in for bwd up-proj dW.
        lt.mlp_in = Some(mlp_in.clone());
    }
    // Audit 8b: split the old single "mlp" lap — it spanned the resid+RMSNorm
    // megakernel, both GEMMs, the activation and the residual add, which made
    // its 4.1 TFLOP/s figure uninterpretable vs the backward's 10.7.
    if let Some(p) = fprof.as_deref_mut() {
        p.lap(rt, "mlp_norm")?;
    }
    let up_w = w.bank_matrix(rt, &w.mlp_up, layer, c, mlp)?;
    let down_w = w.bank_matrix(rt, &w.mlp_down, layer, mlp, c)?;
    let m2 = reshape_view(&mlp_in, &[bt, c]);
    let hidden_pre = rt.alloc_tensor_f32(&[bt, mlp])?;
    if use_persistent_bf16(rt, backend) {
        let m2_bf = if use_bf16_stream {
            m2.clone()
        } else {
            cast_f32_to_bf16(&m2)?
        };
        let up_w_bf = if let Some(ref bf) = w.bf16_banks {
            w.bank_matrix(rt, &bf.mlp_up, layer, c, mlp)?
        } else {
            cast_f32_to_bf16(&up_w)?
        };
        gemm_train(&m2_bf, &up_w_bf, &hidden_pre, backend)?;
    } else {
        gemm_train(&m2, &up_w, &hidden_pre, backend)?;
    }
    if let Some(p) = fprof.as_deref_mut() {
        p.lap(rt, "mlp_up_gemm")?;
    }
    // mlp_act is out-of-place — tape pre-act without deep_copy.
    if let Some(lt) = tape.as_mut().and_then(|t| t.layer.get_mut(layer)) {
        lt.mlp_pre_act = Some(hidden_pre.clone());
    }
    // Audit 9C: under Bf16, fuse act → bf16 and skip the post-act cast into
    // the down-proj GEMM. Pre-act stays f32 for bwd (`mlp_act_bwd_f32`).
    let (hidden_for_down, taped_hidden) = if use_persistent_bf16(rt, backend) {
        let hidden_bf = rt.alloc_tensor_bf16(&[bt, mlp])?;
        mlp_act_to_bf16(rt, &hidden_pre, &hidden_bf)?;
        if let Some(p) = fprof.as_deref_mut() {
            p.lap(rt, "mlp_act")?;
        }
        (hidden_bf.clone(), hidden_bf)
    } else {
        let hidden = rt.alloc_tensor_f32(&[bt, mlp])?;
        mlp_act(rt, &hidden_pre, &hidden)?;
        if let Some(p) = fprof.as_deref_mut() {
            p.lap(rt, "mlp_act")?;
        }
        (hidden.clone(), hidden)
    };
    if let Some(lt) = tape.as_mut().and_then(|t| t.layer.get_mut(layer)) {
        lt.mlp_hidden = Some(taped_hidden);
    }
    let mlp_out_flat = rt.alloc_tensor_f32(&[bt, c])?;
    if use_persistent_bf16(rt, backend) {
        let down_w_bf = if let Some(ref bf) = w.bf16_banks {
            w.bank_matrix(rt, &bf.mlp_down, layer, mlp, c)?
        } else {
            cast_f32_to_bf16(&down_w)?
        };
        gemm_train(&hidden_for_down, &down_w_bf, &mlp_out_flat, backend)?;
    } else {
        gemm_train(&hidden_for_down, &down_w, &mlp_out_flat, backend)?;
    }
    let mlp_out = reshape_view(&mlp_out_flat, &[b, tlen, c]);
    if let Some(lt) = tape.as_mut().and_then(|t| t.layer.get_mut(layer)) {
        lt.mlp_out = Some(mlp_out.clone());
    }
    if let Some(p) = fprof.as_deref_mut() {
        p.lap(rt, "mlp_down_gemm")?;
    }
    x_out = residual_scale_add(rt, &x_out, &mlp_out, &bw.mlp_scale)?;
    if let Some(lt) = tape.as_mut().and_then(|t| t.layer.get_mut(layer)) {
        lt.x_out = Some(x_out.clone());
    }

    if let Some(p) = fprof.as_deref_mut() {
        p.lap(rt, "mlp_resid")?;
    }

    let raw_ret = if cfg.captures_v0(layer) {
        Some(raw_v)
    } else {
        None
    };
    Ok((attn_out, mlp_out, x_out, raw_ret))
}

// ---- kernel wrappers ----

fn stem_fwd(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    ids: &Tensor,
) -> Result<(Tensor, Tensor, Tensor), String> {
    let cfg = &w.cfg;
    let b = cfg.batch;
    let t = cfg.seq_len;
    let c = cfg.model_dim;
    let bt = b * t;
    let db = cfg.bigram_dim;
    let backend = select_backend(rt);

    let pre = rt.alloc_tensor_f32(&[b, t, c])?;
    let bg_rows = rt.alloc_tensor_f32(&[bt, db])?;
    let hash_idx = {
        let buf = rt.alloc_buffer(bt * 4)?;
        Tensor::from_buffer(rt, buf, &[bt], crate::tensor::DType::F32, 0)?
    };
    let p = rt.pipeline("stem_gather_f32")?;
    dispatch_1d(rt, &p, bt, |bnd| {
        set_tensor(bnd, ids, 0);
        set_tensor(bnd, &w.tok_emb, 1);
        set_tensor(bnd, &w.bigram_emb, 2);
        set_tensor(bnd, &pre, 3);
        set_tensor(bnd, &bg_rows, 4);
        set_tensor(bnd, &hash_idx, 5);
        set_u32(bnd, b as u32, 6);
        set_u32(bnd, t as u32, 7);
        set_u32(bnd, c as u32, 8);
        set_u32(bnd, cfg.bigram_vocab as u32, 9);
        set_u32(bnd, db as u32, 10);
    })?;
    let bg_proj = rt.alloc_tensor_f32(&[bt, c])?;
    let pre_flat = reshape_view(&pre, &[bt, c]);
    let bg_w = if use_persistent_bf16(rt, backend) {
        if let Some(ref bf) = w.bf16_banks {
            bf.bigram_proj.clone()
        } else {
            w.bigram_proj.clone()
        }
    } else {
        w.bigram_proj.clone()
    };
    gemm_train(&bg_rows, &bg_w, &bg_proj, backend)?;
    let p2 = rt.pipeline("stem_axpy_scale_f32")?;
    dispatch_1d(rt, &p2, bt * c, |bnd| {
        set_tensor(bnd, &pre_flat, 0);
        set_tensor(bnd, &bg_proj, 1);
        set_tensor(bnd, &w.bigram_scale, 2);
        set_u32(bnd, (bt * c) as u32, 3);
    })?;
    let _ = hash_idx; // hash recomputed in bwd

    // Fused RMSNorm+smear when available (else two kernels).
    if rt.pipeline("rms_norm_smear_f32").is_ok() {
        let post = rt.alloc_tensor_f32(&[b, t, c])?;
        let out = rt.alloc_tensor_f32(&[b, t, c])?;
        let p3 = rt.pipeline("rms_norm_smear_f32")?;
        dispatch_1d(rt, &p3, bt, |bnd| {
            set_tensor(bnd, &pre, 0);
            set_tensor(bnd, &w.smear_gate, 1);
            set_tensor(bnd, &post, 2);
            set_tensor(bnd, &out, 3);
            set_u32(bnd, b as u32, 4);
            set_u32(bnd, t as u32, 5);
            set_u32(bnd, c as u32, 6);
            set_f32(bnd, cfg.f32_eps(), 7);
        })?;
        return Ok((out, pre, post));
    }
    let normed = rms_norm(rt, &pre, bt, c, cfg.f32_eps())?;
    let out = rt.alloc_tensor_f32(&[b, t, c])?;
    let p3 = rt.pipeline("stem_smear_f32")?;
    dispatch_1d(rt, &p3, bt, |bnd| {
        set_tensor(bnd, &normed, 0);
        set_tensor(bnd, &w.smear_gate, 1);
        set_tensor(bnd, &out, 2);
        set_u32(bnd, b as u32, 3);
        set_u32(bnd, t as u32, 4);
        set_u32(bnd, c as u32, 5);
    })?;
    Ok((out, pre, normed))
}

fn rms_norm(
    rt: &Arc<GpuRuntime>,
    x: &Tensor,
    rows: usize,
    c: usize,
    eps: f32,
) -> Result<Tensor, String> {
    rms_norm_scale(rt, x, rows, c, eps, 1.0)
}

/// Fused RMSNorm + scalar scale (Phase 4 megakernel glue).
fn rms_norm_scale(
    rt: &Arc<GpuRuntime>,
    x: &Tensor,
    rows: usize,
    c: usize,
    eps: f32,
    scale: f32,
) -> Result<Tensor, String> {
    let out = rt.alloc_tensor_f32(&x.shape())?;
    let p = rt.pipeline("rms_norm_scale_f32")?;
    dispatch_1d(rt, &p, rows, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, &out, 1);
        set_u32(bnd, rows as u32, 2);
        set_u32(bnd, c as u32, 3);
        set_f32(bnd, eps, 4);
        set_f32(bnd, scale, 5);
    })?;
    Ok(out)
}

fn ve_fwd(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    ids: &Tensor,
    ve_idx: usize,
) -> Result<Tensor, String> {
    let cfg = &w.cfg;
    let bt = cfg.batch * cfg.seq_len;
    let de = cfg.ve_dim;
    let kv = cfg.kv_dim();
    let backend = select_backend(rt);
    let rows = rt.alloc_tensor_f32(&[bt, de])?;
    let p = rt.pipeline("ve_gather_f32")?;
    dispatch_1d(rt, &p, bt, |bnd| {
        set_tensor(bnd, ids, 0);
        set_tensor(bnd, &w.ve_emb, 1);
        set_tensor(bnd, &rows, 2);
        set_u32(bnd, bt as u32, 3);
        set_u32(bnd, de as u32, 4);
    })?;
    let out = rt.alloc_tensor_f32(&[bt, kv])?;
    let ve_w = if use_persistent_bf16(rt, backend) {
        if let Some(ref bf) = w.bf16_banks {
            bf.ve_proj.clone()
        } else {
            w.ve_proj.clone()
        }
    } else {
        w.ve_proj.clone()
    };
    gemm_train(&rows, &ve_w, &out, backend)?;
    let p2 = rt.pipeline("ve_scale_out_f32")?;
    dispatch_1d(rt, &p2, bt * kv, |bnd| {
        set_tensor(bnd, &out, 0);
        set_tensor(bnd, &w.ve_scale, 1);
        set_tensor(bnd, &w.ve_layer_scales[ve_idx], 2);
        set_u32(bnd, (bt * kv) as u32, 3);
    })?;
    Ok(out)
}

fn qkv_post(
    rt: &Arc<GpuRuntime>,
    q_in: &Tensor,
    k_in: &Tensor,
    v_in: &Tensor,
    q_out: &Tensor,
    k_out: &Tensor,
    v_out: &Tensor,
    ve: &Tensor,
    v0: &Tensor,
    raw_v: &Tensor,
    vr_lambda: &Tensor,
    q_gain: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    cfg: &ModelConfig,
    use_ve: bool,
    use_v0: bool,
) -> Result<(), String> {
    let bt = cfg.batch * cfg.seq_len;
    let p = rt.pipeline("qkv_post_f32")?;
    dispatch_1d(rt, &p, bt, |bnd| {
        set_tensor(bnd, q_in, 0);
        set_tensor(bnd, k_in, 1);
        set_tensor(bnd, v_in, 2);
        set_tensor(bnd, q_out, 3);
        set_tensor(bnd, k_out, 4);
        set_tensor(bnd, v_out, 5);
        set_tensor(bnd, ve, 6);
        set_tensor(bnd, v0, 7);
        set_tensor(bnd, raw_v, 8);
        set_tensor(bnd, vr_lambda, 9);
        set_tensor(bnd, q_gain, 10);
        set_tensor(bnd, cos, 11);
        set_tensor(bnd, sin, 12);
        set_u32(bnd, cfg.batch as u32, 13);
        set_u32(bnd, cfg.seq_len as u32, 14);
        set_u32(bnd, cfg.num_heads as u32, 15);
        set_u32(bnd, cfg.num_kv_heads as u32, 16);
        set_u32(bnd, cfg.head_dim as u32, 17);
        set_u32(bnd, cfg.rope_dims as u32, 18);
        set_u32(bnd, use_ve as u32, 19);
        set_u32(bnd, use_v0 as u32, 20);
        set_f32(bnd, cfg.f32_eps(), 21);
    })?;
    Ok(())
}

fn use_persistent_bf16(rt: &GpuRuntime, backend: GemmBackend) -> bool {
    rt.precision() == PrecisionMode::Bf16 && backend == GemmBackend::TensorOps && rt.has_tensorops()
}

fn use_bf16_flash(rt: &GpuRuntime) -> bool {
    // GEMM under Bf16 still accumulates f32 Q/K/V; casting to bf16 just to
    // re-enter flash is a pure round-trip tax. Prefer f32 FA on f32 QKV.
    // (Persistent bf16 QKV would re-enable this; not the current path.)
    let _ = rt;
    false
}

/// Metal flash attention kernels stage at most 64 head elements and otherwise
/// clamp with `min(D, 64u)`, which drops the tail without an error. The host
/// refuses that shape. The kernel clamp stays as a backstop.
pub(crate) fn require_flash_head_dim(head_dim: usize) -> Result<(), String> {
    if head_dim == 0 || head_dim > 64 {
        return Err(format!(
            "head_dim {head_dim} is outside Metal flash-attention support (1..=64)"
        ));
    }
    Ok(())
}

fn flash_attn(
    rt: &Arc<GpuRuntime>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    o: &Tensor,
    lse: &Tensor,
    cfg: &ModelConfig,
) -> Result<(), String> {
    require_flash_head_dim(cfg.head_dim)?;
    // FA-2 tiles: BR=BC=32 (matches kernels/flash_attn_fwd.metal / bf16 twin).
    const BR: usize = 32;
    let scale = 1.0 / (cfg.head_dim as f32).sqrt();
    let q_blocks = (cfg.seq_len + BR - 1) / BR;
    let groups_y = cfg.batch * cfg.num_heads;

    // Optional TensorOps multi-block probe (fwd only; bwd stays simdgroup).
    if rt.flash_tensorops() && cfg.head_dim == 32 {
        if let Ok(p) = rt.pipeline("flash_attn_tensorops_online_f32") {
            dispatch_2d_tg(rt, &p, q_blocks, groups_y, BR, |bnd| {
                set_tensor(bnd, q, 0);
                set_tensor(bnd, k, 1);
                set_tensor(bnd, v, 2);
                set_tensor(bnd, o, 3);
                set_tensor(bnd, lse, 4);
                set_u32(bnd, cfg.batch as u32, 5);
                set_u32(bnd, cfg.seq_len as u32, 6);
                set_u32(bnd, cfg.num_heads as u32, 7);
                set_u32(bnd, cfg.num_kv_heads as u32, 8);
                set_u32(bnd, cfg.head_dim as u32, 9);
                set_f32(bnd, scale, 10);
            })?;
            return Ok(());
        }
    }

    // Phase G Soft quality: FA-2 blockwise online softmax (rowmax over BC +
    // one rescale). Prefer DH=32 specialized twin on the sota Soft shape.
    if crate::ab_flags::fa_blocksoft() {
        let name = if cfg.head_dim == 32 && rt.pipeline("flash_attn_fwd_blocksoft_d32_f32").is_ok()
        {
            "flash_attn_fwd_blocksoft_d32_f32"
        } else {
            "flash_attn_fwd_blocksoft_f32"
        };
        if let Ok(p) = rt.pipeline(name) {
            dispatch_2d_tg(rt, &p, q_blocks, groups_y, BR, |bnd| {
                set_tensor(bnd, q, 0);
                set_tensor(bnd, k, 1);
                set_tensor(bnd, v, 2);
                set_tensor(bnd, o, 3);
                set_tensor(bnd, lse, 4);
                set_u32(bnd, cfg.batch as u32, 5);
                set_u32(bnd, cfg.seq_len as u32, 6);
                set_u32(bnd, cfg.num_heads as u32, 7);
                set_u32(bnd, cfg.num_kv_heads as u32, 8);
                set_u32(bnd, cfg.head_dim as u32, 9);
                set_f32(bnd, scale, 10);
            })?;
            return Ok(());
        }
    }

    // Audit 8: head-dim-specialized forward flash. Same FA-2 math; removes the
    // runtime-`d_lim` register spills and halves threadgroup staging. Under
    // Bf16 this also selects the bf16 twin — the first path that actually
    // reaches bf16 forward flash, since `use_bf16_flash` is hard-coded false.
    if crate::ab_flags::fa_fwd_fast() && cfg.head_dim == 32 {
        let bf16 =
            rt.precision() == PrecisionMode::Bf16 && rt.pipeline("flash_attn_fwd_d32_bf16").is_ok();
        let (q_op, k_op, v_op) = if bf16 {
            (
                cast_f32_to_bf16(q)?,
                cast_f32_to_bf16(k)?,
                cast_f32_to_bf16(v)?,
            )
        } else {
            (q.clone(), k.clone(), v.clone())
        };
        let name = if bf16 {
            "flash_attn_fwd_d32_bf16"
        } else {
            "flash_attn_fwd_d32_f32"
        };
        if let Ok(p) = rt.pipeline(name) {
            dispatch_2d_tg(rt, &p, q_blocks, groups_y, BR, |bnd| {
                set_tensor(bnd, &q_op, 0);
                set_tensor(bnd, &k_op, 1);
                set_tensor(bnd, &v_op, 2);
                set_tensor(bnd, o, 3);
                set_tensor(bnd, lse, 4);
                set_u32(bnd, cfg.batch as u32, 5);
                set_u32(bnd, cfg.seq_len as u32, 6);
                set_u32(bnd, cfg.num_heads as u32, 7);
                set_u32(bnd, cfg.num_kv_heads as u32, 8);
                set_u32(bnd, cfg.head_dim as u32, 9);
                set_f32(bnd, scale, 10);
            })?;
            return Ok(());
        }
    }

    if use_bf16_flash(rt) {
        let q_bf = cast_f32_to_bf16(q)?;
        let k_bf = cast_f32_to_bf16(k)?;
        let v_bf = cast_f32_to_bf16(v)?;
        let p = rt.pipeline("flash_attn_fwd_bf16")?;
        dispatch_2d_tg(rt, &p, q_blocks, groups_y, BR, |bnd| {
            set_tensor(bnd, &q_bf, 0);
            set_tensor(bnd, &k_bf, 1);
            set_tensor(bnd, &v_bf, 2);
            set_tensor(bnd, o, 3);
            set_tensor(bnd, lse, 4);
            set_u32(bnd, cfg.batch as u32, 5);
            set_u32(bnd, cfg.seq_len as u32, 6);
            set_u32(bnd, cfg.num_heads as u32, 7);
            set_u32(bnd, cfg.num_kv_heads as u32, 8);
            set_u32(bnd, cfg.head_dim as u32, 9);
            set_f32(bnd, scale, 10);
        })?;
        return Ok(());
    }

    let p = rt.pipeline("flash_attn_fwd_f32")?;
    dispatch_2d_tg(rt, &p, q_blocks, groups_y, BR, |bnd| {
        set_tensor(bnd, q, 0);
        set_tensor(bnd, k, 1);
        set_tensor(bnd, v, 2);
        set_tensor(bnd, o, 3);
        set_tensor(bnd, lse, 4);
        set_u32(bnd, cfg.batch as u32, 5);
        set_u32(bnd, cfg.seq_len as u32, 6);
        set_u32(bnd, cfg.num_heads as u32, 7);
        set_u32(bnd, cfg.num_kv_heads as u32, 8);
        set_u32(bnd, cfg.head_dim as u32, 9);
        set_f32(bnd, scale, 10);
    })?;
    Ok(())
}

fn xsa_fwd(
    rt: &Arc<GpuRuntime>,
    y_in: &Tensor,
    y_out: &Tensor,
    v: &Tensor,
    cfg: &ModelConfig,
) -> Result<(), String> {
    let n = cfg.batch * cfg.seq_len * cfg.num_kv_heads;
    let p = rt.pipeline("xsa_fwd_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, y_in, 0);
        set_tensor(bnd, y_out, 1);
        set_tensor(bnd, v, 2);
        set_u32(bnd, cfg.batch as u32, 3);
        set_u32(bnd, cfg.seq_len as u32, 4);
        set_u32(bnd, cfg.num_heads as u32, 5);
        set_u32(bnd, cfg.num_kv_heads as u32, 6);
        set_u32(bnd, cfg.head_dim as u32, 7);
        set_f32(bnd, cfg.xsa_eps(), 8);
    })?;
    Ok(())
}

fn resid_mix(
    rt: &Arc<GpuRuntime>,
    x: &Tensor,
    x0: &Tensor,
    mix: &Tensor,
    rows: usize,
    c: usize,
) -> Result<Tensor, String> {
    let out = rt.alloc_tensor_f32(&x.shape())?;
    let p = rt.pipeline("resid_mix_f32")?;
    dispatch_1d(rt, &p, rows, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, x0, 1);
        set_tensor(bnd, mix, 2);
        set_tensor(bnd, &out, 3);
        set_u32(bnd, rows as u32, 4);
        set_u32(bnd, c as u32, 5);
    })?;
    Ok(out)
}

fn residual_scale_add(
    rt: &Arc<GpuRuntime>,
    x: &Tensor,
    branch: &Tensor,
    scale: &Tensor,
) -> Result<Tensor, String> {
    let out = rt.alloc_tensor_f32(&x.shape())?;
    let rows: usize = out.shape()[..out.shape().len() - 1].iter().product();
    let c = *out.shape().last().unwrap();
    let p = rt.pipeline("residual_scale_add_f32")?;
    dispatch_1d(rt, &p, rows, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, branch, 1);
        set_tensor(bnd, scale, 2);
        set_tensor(bnd, &out, 3);
        set_u32(bnd, rows as u32, 4);
        set_u32(bnd, c as u32, 5);
    })?;
    Ok(out)
}

fn skip_add(
    rt: &Arc<GpuRuntime>,
    x: &Tensor,
    skip: &Tensor,
    skip_w: &Tensor,
) -> Result<Tensor, String> {
    let out = rt.alloc_tensor_f32(&x.shape())?;
    let rows: usize = out.shape()[..out.shape().len() - 1].iter().product();
    let c = *out.shape().last().unwrap();
    let p = rt.pipeline("skip_add_f32")?;
    dispatch_1d(rt, &p, rows, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, skip, 1);
        set_tensor(bnd, skip_w, 2);
        set_tensor(bnd, &out, 3);
        set_u32(bnd, rows as u32, 4);
        set_u32(bnd, c as u32, 5);
    })?;
    Ok(out)
}

fn mlp_act(rt: &Arc<GpuRuntime>, x: &Tensor, y: &Tensor) -> Result<(), String> {
    let n = x.numel();
    assert_eq!(n, y.numel());
    let p = rt.pipeline("mlp_act_sq_leaky_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, y, 1);
        set_u32(bnd, n as u32, 2);
    })?;
    Ok(())
}

fn mlp_act_to_bf16(rt: &Arc<GpuRuntime>, x: &Tensor, y: &Tensor) -> Result<(), String> {
    let n = x.numel();
    assert_eq!(n, y.numel());
    assert_eq!(x.dtype, crate::tensor::DType::F32);
    assert_eq!(y.dtype, crate::tensor::DType::BF16);
    let p = rt.pipeline("mlp_act_sq_leaky_f32_to_bf16")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, y, 1);
        set_u32(bnd, n as u32, 2);
    })?;
    Ok(())
}

fn softcap(rt: &Arc<GpuRuntime>, pre: &Tensor, softcap: f32) -> Result<Tensor, String> {
    let out = rt.alloc_tensor_f32(&pre.shape())?;
    let n = pre.numel();
    let p = rt.pipeline("softcap_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, pre, 0);
        set_tensor(bnd, &out, 1);
        set_f32(bnd, softcap, 2);
        set_u32(bnd, n as u32, 3);
    })?;
    Ok(out)
}

fn ce_mean(
    rt: &Arc<GpuRuntime>,
    logits: &Tensor,
    targets: &Tensor,
    rows: usize,
    v: usize,
) -> Result<f32, String> {
    let out = ce_mean_device(rt, logits, targets, rows, v)?;
    rt.synchronize()?;
    Ok(out.buffer.read_f32()[0])
}

// ---- utils ----

fn reshape_view(t: &Tensor, shape: &[usize]) -> Tensor {
    let numel: usize = shape.iter().product();
    assert_eq!(numel, t.numel());
    Tensor::from_buffer(t.runtime(), t.buffer.clone(), shape, t.dtype, t.byte_offset())
        .expect("these views are built over a buffer this crate just allocated")
}

fn upload_i32(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[i32]) -> Result<Tensor, String> {
    let numel: usize = shape.iter().product();
    assert_eq!(data.len(), numel);
    let nbytes = numel * 4;
    let buf = rt.alloc_buffer(nbytes)?;
    let ptr = buf.metal().contents().as_ptr() as *mut i32;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, numel);
    }
    // dtype is logical; contents are i32.
    Tensor::from_buffer(rt, buf, shape, crate::tensor::DType::F32, 0)
}

fn alloc_i32_empty(rt: &Arc<GpuRuntime>, shape: &[usize]) -> Result<Tensor, String> {
    let numel: usize = shape.iter().product();
    let buf = rt.alloc_buffer(numel * 4)?;
    Tensor::from_buffer(rt, buf, shape, crate::tensor::DType::F32, 0)
}

fn write_i32_tensor(t: &Tensor, data: &[i32]) {
    let numel = t.numel();
    assert_eq!(data.len(), numel);
    let ptr =
        (t.buffer.metal().contents().as_ptr() as *mut u8).wrapping_add(t.byte_offset()) as *mut i32;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, numel);
    }
}

/// Ping-pong host→GPU input uploads so step N+1 prep does not race step N.
pub struct DualInputBuffers {
    ids: [Tensor; 2],
    tgts: [Tensor; 2],
    slot: usize,
    in_flight: [bool; 2],
}

impl DualInputBuffers {
    pub fn new(rt: &Arc<GpuRuntime>, batch: usize, seq_len: usize) -> Result<Self, String> {
        let bt = batch * seq_len;
        Ok(Self {
            ids: [
                alloc_i32_empty(rt, &[batch, seq_len])?,
                alloc_i32_empty(rt, &[batch, seq_len])?,
            ],
            tgts: [alloc_i32_empty(rt, &[bt])?, alloc_i32_empty(rt, &[bt])?],
            slot: 0,
            in_flight: [false, false],
        })
    }

    /// Upload into the next free slot. Synchronizes if that slot is still in flight.
    pub fn upload(
        &mut self,
        rt: &Arc<GpuRuntime>,
        input_ids: &[i32],
        target_ids: &[i32],
    ) -> Result<(Tensor, Tensor), String> {
        let s = self.slot;
        if self.in_flight[s] {
            rt.synchronize()?;
            self.in_flight = [false, false];
        }
        write_i32_tensor(&self.ids[s], input_ids);
        write_i32_tensor(&self.tgts[s], target_ids);
        self.in_flight[s] = true;
        self.slot = 1 - s;
        Ok((self.ids[s].clone(), self.tgts[s].clone()))
    }

    /// Clear in-flight flags after a host-visible sync.
    pub fn mark_synced(&mut self) {
        self.in_flight = [false, false];
    }
}

/// Device CE mean without host readback (train path reads after step sync).
pub fn ce_mean_device(
    rt: &Arc<GpuRuntime>,
    logits: &Tensor,
    targets: &Tensor,
    rows: usize,
    v: usize,
) -> Result<Tensor, String> {
    let row_loss = rt.alloc_tensor_f32(&[rows])?;
    let p = rt.pipeline("ce_row_f32")?;
    dispatch_1d(rt, &p, rows, |bnd| {
        set_tensor(bnd, logits, 0);
        set_tensor(bnd, targets, 1);
        set_tensor(bnd, &row_loss, 2);
        set_u32(bnd, rows as u32, 3);
        set_u32(bnd, v as u32, 4);
    })?;
    let out = rt.alloc_tensor_f32(&[1])?;
    let p2 = rt.pipeline("mean_reduce_f32")?;
    let tpg = 256usize;
    rt.with_binder(|bnd| {
        bnd.set_pipeline(&p2);
        set_tensor(bnd, &row_loss, 0);
        set_tensor(bnd, &out, 1);
        set_u32(bnd, rows as u32, 2);
        bnd.dispatch(
            crate::runtime::mtl_size(1, 1, 1),
            crate::runtime::mtl_size(tpg, 1, 1),
        );
        Ok(())
    })?;
    Ok(out)
}

pub fn forward_stub() {}

fn attention_fwd(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    layer: usize,
    attn_in: &Tensor,
    ids: &Tensor,
    v0: Option<&Tensor>,
    backend: GemmBackend,
    mut tape: Option<&mut Tape>,
    b: usize,
    tlen: usize,
    c: usize,
    kv: usize,
    h: usize,
    hkv: usize,
    d: usize,
    bt: usize,
    bw: &crate::weights::BlockWeights,
    use_bf16_stream: bool,
    mut fprof: Option<&mut crate::model_bwd::BwdProf>,
) -> Result<(Tensor, Tensor), String> {
    let cfg = &w.cfg;
    let _eps = cfg.f32_eps();
    let ai = cfg
        .attn_local_idx(layer)
        .ok_or("attention_fwd on non-attn layer")?;
    let n_attn = cfg.mixer_count(crate::weights::MixerKind::Attention);
    let q_w = w.bank_matrix(rt, &w.qo_bank, ai, c, c)?;
    let out_w = w.bank_matrix(rt, &w.qo_bank, n_attn + ai, c, c)?;
    let k_w = w.bank_matrix(rt, &w.kv_bank, ai, c, kv)?;
    let v_w = w.bank_matrix(rt, &w.kv_bank, n_attn + ai, c, kv)?;

    let x2 = reshape_view(&attn_in, &[bt, c]);
    // Persistent bf16 banks (P1b): use pre-cast bank views; else cast-per-GEMM.
    let (x2_gemm, q_w_g, k_w_g, v_w_g) = if use_persistent_bf16(rt, backend) {
        if let Some(ref bf) = w.bf16_banks {
            let q_bf = w.bank_matrix(rt, &bf.qo_bank, ai, c, c)?;
            let k_bf = w.bank_matrix(rt, &bf.kv_bank, ai, c, kv)?;
            let v_bf = w.bank_matrix(rt, &bf.kv_bank, n_attn + ai, c, kv)?;
            let x_bf = if use_bf16_stream {
                x2.clone()
            } else {
                cast_f32_to_bf16(&x2)?
            };
            (x_bf, q_bf, k_bf, v_bf)
        } else if use_bf16_stream {
            (
                x2.clone(),
                cast_f32_to_bf16(&q_w)?,
                cast_f32_to_bf16(&k_w)?,
                cast_f32_to_bf16(&v_w)?,
            )
        } else {
            (
                cast_f32_to_bf16(&x2)?,
                cast_f32_to_bf16(&q_w)?,
                cast_f32_to_bf16(&k_w)?,
                cast_f32_to_bf16(&v_w)?,
            )
        }
    } else {
        (x2.clone(), q_w.clone(), k_w.clone(), v_w.clone())
    };
    let q_pre = rt.alloc_tensor_f32(&[bt, h * d])?;
    let k_pre = rt.alloc_tensor_f32(&[bt, kv])?;
    let v_pre = rt.alloc_tensor_f32(&[bt, kv])?;
    gemm_train(&x2_gemm, &q_w_g, &q_pre, backend)?;
    gemm_train(&x2_gemm, &k_w_g, &k_pre, backend)?;
    gemm_train(&x2_gemm, &v_w_g, &v_pre, backend)?;
    if let Some(p) = fprof.as_deref_mut() {
        p.lap(rt, "qkv_gemms")?;
    }

    // Tape GEMM pre values (qkv_post is out-of-place — no deep_copy).
    if let Some(lt) = tape.as_mut().and_then(|t| t.layer.get_mut(layer)) {
        lt.q_pre = Some(q_pre.clone());
        lt.k_pre = Some(k_pre.clone());
        lt.v_pre = Some(v_pre.clone());
    }

    // VE
    let use_ve = cfg.ve_scale_index(layer).is_some();
    let ve_buf = if let Some(vi) = cfg.ve_scale_index(layer) {
        let ve = ve_fwd(rt, w, ids, vi)?;
        if let Some(lt) = tape.as_mut().and_then(|t| t.layer.get_mut(layer)) {
            lt.ve = Some(ve.clone());
        }
        Some(ve)
    } else {
        Some(rt.alloc_tensor_f32(&[bt, kv])?)
    };
    let ve = ve_buf.as_ref().unwrap();

    let raw_v = rt.alloc_tensor_f32(&[b, tlen, hkv, d])?;
    let v0_buf = match v0 {
        Some(v) => v.clone(),
        None => rt.alloc_tensor_f32(&[b, tlen, hkv, d])?,
    };
    let use_v0 = v0.is_some();

    let q = rt.alloc_tensor_f32(&[bt, h * d])?;
    let k = rt.alloc_tensor_f32(&[bt, kv])?;
    let v = rt.alloc_tensor_f32(&[bt, kv])?;
    qkv_post(
        rt,
        &q_pre,
        &k_pre,
        &v_pre,
        &q,
        &k,
        &v,
        ve,
        &v0_buf,
        &raw_v,
        &bw.vr_lambda,
        &bw.q_gain,
        &w.rope_cos,
        &w.rope_sin,
        cfg,
        use_ve,
        use_v0,
    )?;
    if let Some(p) = fprof.as_deref_mut() {
        p.lap(rt, "qkv_post_ve")?;
    }

    let q_h = reshape_view(&q, &[b, tlen, h, d]);
    let k_h = reshape_view(&k, &[b, tlen, hkv, d]);
    let v_h = reshape_view(&v, &[b, tlen, hkv, d]);
    let y_flash = rt.alloc_tensor_f32(&[b, tlen, h, d])?;
    let lse = rt.alloc_tensor_f32(&[b, h, tlen])?;
    flash_attn(rt, &q_h, &k_h, &v_h, &y_flash, &lse, cfg)?;
    if let Some(p) = fprof.as_deref_mut() {
        p.lap(rt, "flash")?;
    }

    // XSA is out-of-place — tape flash output without deep_copy.
    let use_xsa = cfg.use_xsa(layer);
    let y = if use_xsa {
        let y_out = rt.alloc_tensor_f32(&[b, tlen, h, d])?;
        xsa_fwd(rt, &y_flash, &y_out, &v_h, cfg)?;
        if let Some(lt) = tape.as_mut().and_then(|t| t.layer.get_mut(layer)) {
            lt.attn_y_flash = Some(y_flash);
            lt.attn_lse = Some(lse);
            lt.attn_y = Some(y_out.clone());
        }
        y_out
    } else {
        if let Some(lt) = tape.as_mut().and_then(|t| t.layer.get_mut(layer)) {
            lt.attn_y_flash = Some(y_flash.clone());
            lt.attn_y = Some(y_flash.clone());
            lt.attn_lse = Some(lse);
        }
        y_flash
    };

    if let Some(lt) = tape.as_mut().and_then(|t| t.layer.get_mut(layer)) {
        lt.q = Some(q_h.clone());
        lt.k = Some(k_h.clone());
        lt.v_mixed = Some(v_h.clone());
        lt.raw_v = Some(raw_v.clone());
    }
    if let Some(p) = fprof.as_deref_mut() {
        p.lap(rt, "xsa")?;
    }

    let y_flat = reshape_view(&y, &[bt, c]);
    let attn_out = rt.alloc_tensor_f32(&[bt, c])?;
    if use_persistent_bf16(rt, backend) {
        let y_bf = cast_f32_to_bf16(&y_flat)?;
        let out_w_bf = if let Some(ref bf) = w.bf16_banks {
            w.bank_matrix(rt, &bf.qo_bank, n_attn + ai, c, c)?
        } else {
            cast_f32_to_bf16(&out_w)?
        };
        gemm_train(&y_bf, &out_w_bf, &attn_out, backend)?;
        // P1e: reuse bf16 Y for bwd out-proj dW (FA O stays f32 on attn_y_flash).
        if let Some(lt) = tape.as_mut().and_then(|t| t.layer.get_mut(layer)) {
            lt.attn_y = Some(reshape_view(&y_bf, &[b, tlen, h, d]));
        }
    } else {
        gemm_train(&y_flat, &out_w, &attn_out, backend)?;
    }
    let attn_out = reshape_view(&attn_out, &[b, tlen, c]);
    if let Some(lt) = tape.as_mut().and_then(|t| t.layer.get_mut(layer)) {
        lt.attn_out = Some(attn_out.clone());
    }
    if let Some(p) = fprof.as_deref_mut() {
        p.lap(rt, "attn_out_gemm")?;
    }
    Ok((attn_out, raw_v))
}

fn mamba2_fwd_rust(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    layer: usize,
    attn_in: &Tensor,
    _ids: &Tensor,
    backend: GemmBackend,
    tape: Option<&mut Tape>,
    b: usize,
    tlen: usize,
    c: usize,
) -> Result<(Tensor, Tensor), String> {
    let cfg = &w.cfg;
    let bt = b * tlen;
    let d_inner = cfg.mamba_d_inner();
    let d_state = cfg.d_state;
    let n_head = cfg.mamba_n_head();
    let head_dim = cfg.mamba_head_dim();
    let conv_dim = cfg.mamba_conv_dim();
    let in_out = cfg.mamba_in_proj_out();
    let eps = 1e-6f32;

    let in_proj = w.mamba_in_proj.as_ref().ok_or("mamba_in_proj missing")?;
    let out_proj = w.mamba_out_proj.as_ref().ok_or("mamba_out_proj missing")?;
    let conv_w = w
        .mamba_conv1d_weight
        .as_ref()
        .ok_or("mamba_conv1d_weight missing")?;
    let conv_b = w
        .mamba_conv1d_bias
        .as_ref()
        .ok_or("mamba_conv1d_bias missing")?;
    let a_log = w.mamba_a_log.as_ref().ok_or("mamba_a_log missing")?;
    let d_param = w.mamba_d.as_ref().ok_or("mamba_d missing")?;
    let dt_bias = w.mamba_dt_bias.as_ref().ok_or("mamba_dt_bias missing")?;
    let norm_w = w.mamba_norm.as_ref().ok_or("mamba_norm missing")?;

    let mi = cfg.mixer_local_idx(layer);

    let in_w = w.bank_matrix(rt, in_proj, mi, c, in_out)?;
    let out_w = w.bank_matrix(rt, out_proj, mi, d_inner, c)?;
    let layer_conv_w = w.bank_matrix(rt, conv_w, mi, conv_dim, cfg.d_conv)?;
    let layer_conv_b = conv_b.view(&[conv_dim], mi * conv_dim);
    let layer_a_log = a_log.view(&[n_head], mi * n_head);
    let layer_d = d_param.view(&[n_head], mi * n_head);
    let layer_dt_bias = dt_bias.view(&[n_head], mi * n_head);
    let layer_norm = norm_w.view(&[d_inner], mi * d_inner);

    let ai_flat = reshape_view(attn_in, &[bt, c]);
    let zxbcdt = rt.alloc_tensor_f32(&[bt, in_out])?;
    gemm_train(&ai_flat, &in_w, &zxbcdt, backend)?;

    let zxbcdt_3d = reshape_view(&zxbcdt, &[b, tlen, in_out]);
    let z = slice_last_dim(&zxbcdt_3d, 0, d_inner);
    let xbc_pre = slice_last_dim(&zxbcdt_3d, d_inner, conv_dim);
    let dt_raw = slice_last_dim(&zxbcdt_3d, d_inner + conv_dim, n_head);

    let xbc_conv = mamba2_conv1d_fwd(rt, &xbc_pre, &layer_conv_w, &layer_conv_b)?;
    let xbc_post = rt.alloc_tensor_f32(&[b, tlen, conv_dim])?;
    let xbc = rt.alloc_tensor_f32(&[b, tlen, conv_dim])?;
    silu_fwd_store(rt, &xbc_conv, &xbc, &xbc_post)?;

    let xs = slice_last_dim(&xbc, 0, d_inner);
    let bm = slice_last_dim(&xbc, d_inner, d_state);
    let cm = slice_last_dim(&xbc, d_inner + d_state, d_state);

    let dt = rt.alloc_tensor_f32(&[b, tlen, n_head])?;
    softplus_bias_fwd(rt, &dt_raw, &layer_dt_bias, &dt, b, tlen, n_head)?;

    let xs_heads = reshape_heads(&xs, n_head, head_dim);
    let x_scaled = rt.alloc_tensor_f32(&[b, tlen, n_head, head_dim])?;
    mamba2_x_scaled(rt, &xs_heads, &dt, &x_scaled, b, tlen, n_head, head_dim)?;

    let log_da = rt.alloc_tensor_f32(&[b, tlen, n_head])?;
    mamba2_log_da(rt, &dt, &layer_a_log, &log_da, b, tlen, n_head)?;

    let ssd = mamba2_fwd(rt, &x_scaled, &bm, &cm, &log_da)?;
    mamba2_d_skip_fwd(rt, &ssd.y, &xs_heads, &layer_d, b, tlen, n_head, head_dim)?;

    let y_flat = rt.alloc_tensor_f32(&[bt, d_inner])?;
    {
        let y_3d = reshape_view(&ssd.y, &[b, tlen, d_inner]);
        let p = rt.pipeline("copy_f32")?;
        dispatch_1d(rt, &p, bt * d_inner, |bnd| {
            set_tensor(bnd, &y_3d, 0);
            set_tensor(bnd, &y_flat, 1);
            set_u32(bnd, (bt * d_inner) as u32, 2);
        })?;
    }

    let y_norm = rt.alloc_tensor_f32(&[bt, d_inner])?;
    rms_norm_weight_fwd(rt, &y_flat, &layer_norm, &y_norm, bt, d_inner, eps)?;

    let z_silu = rt.alloc_tensor_f32(&[bt, d_inner])?;
    let z_flat = reshape_view(&z, &[bt, d_inner]);
    silu_fwd(rt, &z_flat, &z_silu)?;

    let y_gated = rt.alloc_tensor_f32(&[bt, d_inner])?;
    mul_fwd(rt, &y_norm, &z_silu, &y_gated)?;

    let attn_out_flat = rt.alloc_tensor_f32(&[bt, c])?;
    gemm_train(&y_gated, &out_w, &attn_out_flat, backend)?;
    let attn_out = reshape_view(&attn_out_flat, &[b, tlen, c]);

    if let Some(lt) = tape.and_then(|t| t.layer.get_mut(layer)) {
        lt.mamba_z = Some(z.clone());
        lt.mamba_z_pre_silu = Some(z_flat.clone());
        lt.mamba_xbc_pre = Some(xbc_pre);
        lt.mamba_xbc_post = Some(xbc_post);
        lt.mamba_xbc = Some(xbc);
        lt.mamba_xs = Some(xs_heads);
        lt.mamba_bm = Some(bm);
        lt.mamba_cm = Some(cm);
        lt.mamba_dt_raw = Some(dt_raw);
        lt.mamba_dt = Some(dt);
        lt.mamba_x_scaled = Some(x_scaled);
        lt.mamba_log_da = Some(log_da);
        lt.mamba_ssd_y = Some(ssd.y.clone());
        lt.mamba_h_states = Some(ssd.h_states);
        lt.mamba_y_flat = Some(y_flat);
        lt.mamba_y_norm = Some(y_norm);
        lt.mamba_y_pre_out = Some(y_gated.clone());
    }

    let raw_v = rt.alloc_tensor_f32(&[b, tlen, 1, 1])?;
    Ok((attn_out, raw_v))
}

fn mingru_fwd_rust(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    layer: usize,
    attn_in: &Tensor,
    _ids: &Tensor,
    v0: Option<&Tensor>,
    backend: GemmBackend,
    tape: Option<&mut Tape>,
    b: usize,
    tlen: usize,
    c: usize,
    kv: usize,
    hkv: usize,
    d: usize,
    bw: &crate::weights::BlockWeights,
) -> Result<(Tensor, Tensor), String> {
    let cfg = &w.cfg;
    let bt = b * tlen;
    let hid = cfg.mingru_hidden();
    let mi = cfg.mixer_local_idx(layer);
    let use_vr = cfg.value_residual;

    let to_z = w.mingru_to_z.as_ref().ok_or("mingru_to_z missing")?;
    let to_h = w.mingru_to_h.as_ref().ok_or("mingru_to_h missing")?;
    let out_w_bank = w.mingru_out.as_ref().ok_or("mingru_out missing")?;

    let z_w = w.bank_matrix(rt, to_z, mi, c, hid)?;
    let h_w = w.bank_matrix(rt, to_h, mi, c, hid)?;
    let out_w = w.bank_matrix(rt, out_w_bank, mi, hid, c)?;

    let ai_flat = reshape_view(attn_in, &[bt, c]);
    let z_raw = rt.alloc_tensor_f32(&[bt, hid])?;
    let h_raw = rt.alloc_tensor_f32(&[bt, hid])?;
    gemm_train(&ai_flat, &z_w, &z_raw, backend)?;
    gemm_train(&ai_flat, &h_w, &h_raw, backend)?;

    let raw_v = if use_vr {
        let v_proj = w.mingru_v_proj.as_ref().ok_or("mingru_v_proj missing")?;
        let v_w = w.bank_matrix(rt, v_proj, mi, c, kv)?;
        let v_flat = rt.alloc_tensor_f32(&[bt, kv])?;
        gemm_train(&ai_flat, &v_w, &v_flat, backend)?;
        reshape_view(&v_flat, &[b, tlen, hkv, d])
    } else {
        rt.alloc_tensor_f32(&[b, tlen, 1, 1])?
    };

    let (h_pre, v0_up_buf) = if use_vr && v0.is_some() {
        let v0_up_bank = w.mingru_v0_up.as_ref().ok_or("mingru_v0_up missing")?;
        let v0_up_w = w.bank_matrix(rt, v0_up_bank, mi, kv, hid)?;
        let v0_flat = reshape_view(v0.unwrap(), &[bt, kv]);
        let v0_up = rt.alloc_tensor_f32(&[bt, hid])?;
        gemm_train(&v0_flat, &v0_up_w, &v0_up, backend)?;
        let h_pre = rt.alloc_tensor_f32(&[bt, hid])?;
        mingru_vr_blend_fwd(rt, &h_raw, &v0_up, &bw.vr_lambda, &h_pre, true)?;
        (h_pre, Some(v0_up))
    } else {
        (h_raw.clone(), None)
    };

    let z_3d = reshape_view(&z_raw, &[b, tlen, hid]);
    let h_3d = reshape_view(&h_pre, &[b, tlen, hid]);
    let h_out = mingru_fwd(rt, &z_3d, &h_3d)?;

    let h_flat = reshape_view(&h_out, &[bt, hid]);
    let attn_out_flat = rt.alloc_tensor_f32(&[bt, c])?;
    gemm_train(&h_flat, &out_w, &attn_out_flat, backend)?;
    let attn_out = reshape_view(&attn_out_flat, &[b, tlen, c]);

    if let Some(lt) = tape.and_then(|t| t.layer.get_mut(layer)) {
        lt.mingru_z_raw = Some(z_3d);
        lt.mingru_h_raw = Some(reshape_view(&h_raw, &[b, tlen, hid]));
        lt.mingru_h_pre = Some(h_3d);
        lt.mingru_v0_up = v0_up_buf.map(|v| reshape_view(&v, &[b, tlen, hid]));
        lt.mingru_h_out = Some(h_out);
        if use_vr {
            lt.raw_v = Some(raw_v.clone());
        }
    }

    Ok((attn_out, raw_v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_head_dim_above_64_is_refused_on_the_host() {
        let err = require_flash_head_dim(65).unwrap_err();
        assert!(err.contains("65"), "{err}");
        assert!(err.contains("1..=64"), "{err}");
        require_flash_head_dim(64).unwrap();
        assert!(require_flash_head_dim(0).is_err());
    }

    /// `flash_attn_tensorops_online_f32` returns without writing O/L when
    /// `D != 32`. Enabling the optional probe must therefore select the normal
    /// flash implementation for every other supported head dimension.
    #[test]
    fn flash_tensorops_opt_in_d64_falls_back_before_shader_early_return() {
        let rt = crate::gpu_runtime().expect("GpuRuntime");
        let cfg = ModelConfig {
            batch: 1,
            seq_len: 2,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            model_dim: 128,
            ..ModelConfig::sota_toy()
        };
        cfg.validate_metal_shape().expect("D=64 is supported");

        let make_input = |shape: &[usize], len: usize, modulus: usize, scale: f32| {
            let tensor = rt.alloc_tensor_f32(shape).expect("input allocation");
            let values: Vec<f32> = (0..len)
                .map(|i| ((i % modulus) as f32 - modulus as f32 * 0.5) * scale)
                .collect();
            tensor.buffer.write_f32(&values);
            tensor
        };
        let q_len = cfg.batch * cfg.seq_len * cfg.num_heads * cfg.head_dim;
        let kv_len = cfg.batch * cfg.seq_len * cfg.num_kv_heads * cfg.head_dim;
        let q = make_input(
            &[cfg.batch, cfg.seq_len, cfg.num_heads, cfg.head_dim],
            q_len,
            17,
            0.02,
        );
        let k = make_input(
            &[cfg.batch, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim],
            kv_len,
            13,
            0.025,
        );
        let v = make_input(
            &[cfg.batch, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim],
            kv_len,
            11,
            0.03,
        );

        let run = |tensorops: bool| {
            rt.set_flash_tensorops(tensorops);
            let o = rt
                .alloc_tensor_f32(&[cfg.batch, cfg.seq_len, cfg.num_heads, cfg.head_dim])
                .expect("O allocation");
            let lse = rt
                .alloc_tensor_f32(&[cfg.batch, cfg.num_heads, cfg.seq_len])
                .expect("LSE allocation");
            o.buffer.write_f32(&vec![f32::NAN; q_len]);
            lse.buffer
                .write_f32(&vec![f32::NAN; cfg.batch * cfg.num_heads * cfg.seq_len]);
            flash_attn(&rt, &q, &k, &v, &o, &lse, &cfg).expect("flash dispatch");
            rt.synchronize().expect("flash completion");
            (o.buffer.read_f32(), lse.buffer.read_f32())
        };

        let (reference_o, reference_lse) = run(false);
        let (fallback_o, fallback_lse) = run(true);
        assert!(
            fallback_o
                .iter()
                .chain(&fallback_lse)
                .all(|x| x.is_finite()),
            "D=64 TensorOps opt-in reached the D=32-only shader and left output unwritten"
        );
        let max_abs = reference_o
            .iter()
            .chain(&reference_lse)
            .zip(fallback_o.iter().chain(&fallback_lse))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(max_abs, 0.0, "D=64 fallback differs from normal flash");
    }
}
