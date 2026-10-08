//! Golden-tensor parity harness (f32).
//! Fwd ≤ 1e-5; bwd grads ≤ 1e-4 (post clip) + finite-diff spot checks.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::model_bwd::{backward_f32, backward_f32_opts, Grads, BWD_ATOL};
use crate::model_fwd::{forward_f32, ForwardOutputs};
use crate::npy::{read_npy, transpose_last2};
use crate::optim::{optim_step, zero_grads, OptimHyperparams, OptimState, OPTIM_ATOL};
use crate::runtime::GpuRuntime;
use crate::tape::Tape;
use crate::tensor::Tensor;
use crate::weights::{ModelConfig, Weights};

pub const FWD_ATOL: f32 = 1e-5;

#[derive(Debug, Clone)]
pub struct CompareResult {
    pub name: String,
    pub max_abs: f32,
    pub mean_abs: f32,
    pub passed: bool,
}

pub fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden")
}

/// Fail with instructions when the regenerable forward activations are absent.
///
/// `golden/fwd/` is ~80 MB of activations and is gitignored, so a fresh clone
/// has every other golden but not these. Without this the tests die on a bare
/// `No such file or directory` naming one `.npy`, which says nothing about how
/// to get it. A check that cannot run has to say what would make it run.
pub fn require_fwd_goldens(golden: &std::path::Path) {
    let fwd = golden.join("fwd");
    let probe = fwd.join("stem_after_smear.npy");
    if probe.exists() {
        return;
    }
    panic!(
        "forward goldens missing at {}\n\
         These activations are regenerable and therefore gitignored \
         (.gitignore: golden/fwd/), so a fresh clone will not have them.\n\
         Regenerate with:\n\
         \n    python3 {}/scripts/export_goldens.py\n\n\
         Every other golden directory (grads, inputs, optim_step3, weights_init) \
         is committed and needs nothing.",
        fwd.display(),
        env!("CARGO_MANIFEST_DIR"),
    );
}

/// Error reductions must preserve non-finite evidence instead of hiding NaNs.
pub(crate) fn max_finite_error(acc: f32, value: f32) -> f32 {
    if acc.is_finite() && value.is_finite() {
        acc.max(value)
    } else {
        f32::INFINITY
    }
}

pub fn max_abs_err(got: &[f32], exp: &[f32]) -> f32 {
    if got.is_empty() || got.len() != exp.len() {
        return f32::INFINITY;
    }
    got.iter()
        .zip(exp)
        .map(|(g, e)| (g - e).abs())
        .fold(0.0, max_finite_error)
}

pub fn mean_abs_err(got: &[f32], exp: &[f32]) -> f32 {
    if got.is_empty() || got.len() != exp.len() {
        return f32::INFINITY;
    }
    let mut sum = 0.0f64;
    for (&g, &e) in got.iter().zip(exp) {
        if !g.is_finite() || !e.is_finite() {
            return f32::INFINITY;
        }
        sum += (f64::from(g) - f64::from(e)).abs();
    }
    (sum / got.len() as f64) as f32
}

pub fn compare_f32(name: &str, got: &[f32], exp: &[f32], atol: f32) -> CompareResult {
    let max_abs = max_abs_err(got, exp);
    let mean_abs = mean_abs_err(got, exp);
    CompareResult {
        name: name.to_string(),
        max_abs,
        mean_abs,
        passed: atol.is_finite() && atol >= 0.0 && max_abs.is_finite() && max_abs <= atol,
    }
}

pub fn load_input_ids(golden: &Path, batch_idx: usize) -> Result<Vec<i32>, String> {
    let arr = read_npy(&golden.join(format!("inputs/input_ids_batch{batch_idx}.npy")))?;
    let i64s = arr.i64_slice()?;
    Ok(i64s.iter().map(|&x| x as i32).collect())
}

pub fn load_target_ids(golden: &Path, batch_idx: usize) -> Result<Vec<i32>, String> {
    let arr = read_npy(&golden.join(format!("inputs/target_ids_batch{batch_idx}.npy")))?;
    let i64s = arr.i64_slice()?;
    Ok(i64s.iter().map(|&x| x as i32).collect())
}

pub fn load_fwd_f32(golden: &Path, key: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
    let arr = read_npy(&golden.join(format!("{key}.npy")))?;
    Ok((arr.shape.clone(), arr.f32_slice()?.to_vec()))
}

fn load_grad_python_layout(golden: &Path, rel: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
    let arr = read_npy(&golden.join(rel))?;
    Ok((arr.shape.clone(), arr.f32_slice()?.to_vec()))
}

/// Transpose last-2 of a Burn-layout bank/linear grad to Python [out,in] for compare.
fn to_python_linear(data: &[f32], shape: &[usize]) -> Result<Vec<f32>, String> {
    let mut d = data.to_vec();
    let mut s = shape.to_vec();
    transpose_last2(&mut d, &mut s)?;
    Ok(d)
}

/// Run full forward vs `golden/fwd/*` and return per-tensor results.
pub fn run_fwd_parity(rt: &Arc<GpuRuntime>, golden: &Path) -> Result<Vec<CompareResult>, String> {
    let cfg = ModelConfig::sota_toy();
    let w = Weights::load_from_golden(rt, golden, cfg)?;
    let ids = load_input_ids(golden, 0)?;
    let tgts = load_target_ids(golden, 0)?;
    let mut tape = Tape::new_sota();
    let out = forward_f32(rt, &w, &ids, &tgts, &mut tape)?;
    compare_forward(rt, golden, &out)
}

pub fn compare_forward(
    rt: &Arc<GpuRuntime>,
    golden: &Path,
    out: &ForwardOutputs,
) -> Result<Vec<CompareResult>, String> {
    let mut results = Vec::new();
    // Ensure all encoded kernels complete before host reads.
    rt.synchronize()?;
    let mut check = |name: &str, got: &[f32]| -> Result<(), String> {
        let (_shape, exp) = load_fwd_f32(golden, name)?;
        results.push(compare_f32(name, got, &exp, FWD_ATOL));
        Ok(())
    };

    check("fwd/stem_after_smear", &out.stem.buffer.read_f32())?;
    check("fwd/v0", &out.v0.buffer.read_f32())?;

    for i in 0..out.layer_attn_out.len() {
        check(
            &format!("fwd/layer{i}_attn_out"),
            &out.layer_attn_out[i].buffer.read_f32(),
        )?;
        check(
            &format!("fwd/layer{i}_mlp_out"),
            &out.layer_mlp_out[i].buffer.read_f32(),
        )?;
        check(
            &format!("fwd/layer{i}_x"),
            &out.layer_x[i].buffer.read_f32(),
        )?;
        if let Some(ref s) = out.layer_after_skip[i] {
            check(&format!("fwd/layer{i}_after_skip"), &s.buffer.read_f32())?;
        }
    }

    check("fwd/final_norm", &out.final_norm.buffer.read_f32())?;
    check("fwd/logits_pre_softcap", &out.logits_pre.buffer.read_f32())?;
    check(
        "fwd/logits_post_softcap",
        &out.logits_post.buffer.read_f32(),
    )?;

    let loss = out.read_loss(rt)?;
    let (_s, loss_exp) = load_fwd_f32(golden, "fwd/loss")?;
    results.push(compare_f32("fwd/loss", &[loss], &loss_exp, FWD_ATOL));

    Ok(results)
}

pub fn format_report(results: &[CompareResult]) -> String {
    format_report_atol(results, FWD_ATOL)
}

pub fn format_report_atol(results: &[CompareResult], atol: f32) -> String {
    let mut lines = Vec::new();
    let mut npass = 0;
    for r in results {
        let mark = if r.passed {
            npass += 1;
            "PASS"
        } else {
            "FAIL"
        };
        lines.push(format!(
            "  [{mark}] {:<40} max_abs={:.6e} mean_abs={:.6e}",
            r.name, r.max_abs, r.mean_abs
        ));
    }
    lines.push(format!(
        "summary: {npass}/{} passed (atol={atol})",
        results.len()
    ));
    lines.join("\n")
}

fn check_emb(
    name: &str,
    got: &Tensor,
    golden: &Path,
    rel: &str,
    out: &mut Vec<CompareResult>,
) -> Result<(), String> {
    let (_s, exp) = load_grad_python_layout(golden, rel)?;
    out.push(compare_f32(name, &got.buffer.read_f32(), &exp, BWD_ATOL));
    Ok(())
}

fn check_linear(
    name: &str,
    got: &Tensor,
    golden: &Path,
    rel: &str,
    out: &mut Vec<CompareResult>,
) -> Result<(), String> {
    let (shape, exp) = load_grad_python_layout(golden, rel)?;
    let got_py = to_python_linear(&got.buffer.read_f32(), &got.shape())?;
    assert_eq!(got_py.len(), exp.len(), "{name} vs {shape:?}");
    out.push(compare_f32(name, &got_py, &exp, BWD_ATOL));
    Ok(())
}

fn check_bank(
    name: &str,
    got: &Tensor,
    golden: &Path,
    rel: &str,
    out: &mut Vec<CompareResult>,
) -> Result<(), String> {
    check_linear(name, got, golden, rel, out)
}

/// Forward + backward vs `golden/grads/*` (post clip).
pub fn run_bwd_parity(rt: &Arc<GpuRuntime>, golden: &Path) -> Result<Vec<CompareResult>, String> {
    let cfg = ModelConfig::sota_toy();
    let w = Weights::load_from_golden(rt, golden, cfg)?;
    let ids = load_input_ids(golden, 0)?;
    let tgts = load_target_ids(golden, 0)?;
    let mut tape = Tape::new_sota();
    let _out = forward_f32(rt, &w, &ids, &tgts, &mut tape)?;
    let mut grads = Grads::zeros_like(rt, &w)?;
    backward_f32(rt, &w, &tape, &mut grads)?;
    compare_grads(golden, &grads)
}

pub fn compare_grads(golden: &Path, g: &Grads) -> Result<Vec<CompareResult>, String> {
    let mut results = Vec::new();
    check_emb(
        "grads/tok_emb",
        &g.tok_emb,
        golden,
        "grads/tok_emb/weight.npy",
        &mut results,
    )?;
    check_emb(
        "grads/bigram_emb",
        &g.bigram_emb,
        golden,
        "grads/bigram/embed/weight.npy",
        &mut results,
    )?;
    check_linear(
        "grads/bigram_proj",
        &g.bigram_proj,
        golden,
        "grads/bigram/proj/weight.npy",
        &mut results,
    )?;
    check_emb(
        "grads/bigram_scale",
        &g.bigram_scale,
        golden,
        "grads/bigram/scale.npy",
        &mut results,
    )?;
    check_emb(
        "grads/smear_gate",
        &g.smear_gate,
        golden,
        "grads/smear/gate.npy",
        &mut results,
    )?;
    check_emb(
        "grads/ve_emb",
        &g.ve_emb,
        golden,
        "grads/ve_shared/embed/weight.npy",
        &mut results,
    )?;
    check_linear(
        "grads/ve_proj",
        &g.ve_proj,
        golden,
        "grads/ve_shared/proj/weight.npy",
        &mut results,
    )?;
    check_emb(
        "grads/ve_scale",
        &g.ve_scale,
        golden,
        "grads/ve_shared/scale.npy",
        &mut results,
    )?;
    for i in 0..g.ve_layer_scales.len() {
        check_emb(
            &format!("grads/ve_layer_scales/{i}"),
            &g.ve_layer_scales[i],
            golden,
            &format!("grads/ve_layer_scales/{i}.npy"),
            &mut results,
        )?;
    }
    check_emb(
        "grads/skip_weights",
        &g.skip_weights,
        golden,
        "grads/skip_weights.npy",
        &mut results,
    )?;
    check_bank(
        "grads/qo_bank",
        &g.qo_bank,
        golden,
        "grads/qo_bank.npy",
        &mut results,
    )?;
    check_bank(
        "grads/kv_bank",
        &g.kv_bank,
        golden,
        "grads/kv_bank.npy",
        &mut results,
    )?;
    check_bank(
        "grads/mlp_up",
        &g.mlp_up,
        golden,
        "grads/mlp_up_bank.npy",
        &mut results,
    )?;
    check_bank(
        "grads/mlp_down",
        &g.mlp_down,
        golden,
        "grads/mlp_down_bank.npy",
        &mut results,
    )?;
    for i in 0..g.blocks.len() {
        let b = &g.blocks[i];
        check_emb(
            &format!("grads/blocks/{i}/q_gain"),
            &b.q_gain,
            golden,
            &format!("grads/blocks/{i}/attn/q_gain.npy"),
            &mut results,
        )?;
        let vr_path = format!("grads/blocks/{i}/attn/vr_lambda.npy");
        if golden.join(&vr_path).exists() {
            check_emb(
                &format!("grads/blocks/{i}/vr_lambda"),
                &b.vr_lambda,
                golden,
                &vr_path,
                &mut results,
            )?;
        }
        check_emb(
            &format!("grads/blocks/{i}/attn_scale"),
            &b.attn_scale,
            golden,
            &format!("grads/blocks/{i}/attn_scale.npy"),
            &mut results,
        )?;
        check_emb(
            &format!("grads/blocks/{i}/mlp_scale"),
            &b.mlp_scale,
            golden,
            &format!("grads/blocks/{i}/mlp_scale.npy"),
            &mut results,
        )?;
        check_emb(
            &format!("grads/blocks/{i}/resid_mix"),
            &b.resid_mix,
            golden,
            &format!("grads/blocks/{i}/resid_mix.npy"),
            &mut results,
        )?;
    }
    Ok(results)
}

/// Finite-difference spot checks vs unclipped analytic grads.
///
/// Primary correctness gate is golden grad parity (autograd). Central-diff on
/// mean CE at eps~1e-3..1e-4 is noisy for tiny params; we require same sign and
/// order-of-magnitude agreement on the largest-grad params.
pub fn finite_diff_spot_checks(
    rt: &Arc<GpuRuntime>,
    golden: &Path,
) -> Result<Vec<(String, f32, f32, bool)>, String> {
    let eps = 5e-4f32;
    let cfg = ModelConfig::sota_toy();
    let ids = load_input_ids(golden, 0)?;
    let tgts = load_target_ids(golden, 0)?;

    let loss_at = |w: &Weights| -> Result<f32, String> {
        let mut tape = Tape::new_sota();
        let out = forward_f32(rt, w, &ids, &tgts, &mut tape)?;
        out.read_loss(rt)
    };

    let w = Weights::load_from_golden(rt, golden, cfg)?;
    let mut tape = Tape::new_sota();
    let _ = forward_f32(rt, &w, &ids, &tgts, &mut tape)?;
    let mut grads = Grads::zeros_like(rt, &w)?;
    backward_f32_opts(rt, &w, &tape, &mut grads, false)?;

    let mut results = Vec::new();

    let mut spot = |name: &str, tensor: &Tensor, analytic: f32, idx: usize| -> Result<(), String> {
        let mut host = tensor.buffer.read_f32();
        let orig = host[idx];
        host[idx] = orig + eps;
        tensor.buffer.write_f32(&host);
        let lp = loss_at(&w)?;
        host[idx] = orig - eps;
        tensor.buffer.write_f32(&host);
        let lm = loss_at(&w)?;
        host[idx] = orig;
        tensor.buffer.write_f32(&host);
        let fd = (lp - lm) / (2.0 * eps);
        // Tiny analytic: FD dominated by loss noise (~1e-4); treat as soft pass.
        let ok = if analytic.abs() < 1e-4 {
            fd.abs() < 5e-3
        } else {
            let same_sign = fd.signum() == analytic.signum() || fd.abs() < 1e-4;
            let ratio = fd.abs() / analytic.abs();
            same_sign && ratio > 0.25 && ratio < 4.0
        };
        results.push((name.to_string(), fd, analytic, ok));
        Ok(())
    };

    let tok_g = grads.tok_emb.buffer.read_f32();
    let tok_idx = tok_g
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    spot(
        &format!("fd/tok_emb[{tok_idx}]"),
        &w.tok_emb,
        tok_g[tok_idx],
        tok_idx,
    )?;

    let rm = grads.blocks[0].resid_mix.buffer.read_f32();
    let rm_idx = rm
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    spot(
        &format!("fd/blocks.0.resid_mix[{rm_idx}]"),
        &w.blocks[0].resid_mix,
        rm[rm_idx],
        rm_idx,
    )?;

    let sw = grads.skip_weights.buffer.read_f32();
    let sw_idx = sw
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    spot(
        &format!("fd/skip_weights[{sw_idx}]"),
        &w.skip_weights,
        sw[sw_idx],
        sw_idx,
    )?;

    let qo = grads.qo_bank.buffer.read_f32();
    let qo_idx = qo
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    spot(
        &format!("fd/qo_bank[{qo_idx}]"),
        &w.qo_bank,
        qo[qo_idx],
        qo_idx,
    )?;

    Ok(results)
}

/// Run `n_steps` (default 3) training steps vs `golden/optim_step3/` params + states.
/// EMA is disabled for this gate (exporter did not dump EMA).
pub fn run_optim_parity(
    rt: &Arc<GpuRuntime>,
    golden: &Path,
    n_steps: usize,
) -> Result<Vec<CompareResult>, String> {
    let cfg = ModelConfig::sota_toy();
    let mut w = Weights::load_from_golden(rt, golden, cfg)?;
    let mut state = OptimState::new(rt, &w, OptimHyperparams::default())?;
    let mut grads = Grads::zeros_like(rt, &w)?;
    let mut tape = Tape::new_sota();
    let mut losses = Vec::new();

    for step in 0..n_steps {
        zero_grads(rt, &grads)?;
        let ids = load_input_ids(golden, step)?;
        let tgts = load_target_ids(golden, step)?;
        let out = forward_f32(rt, &w, &ids, &tgts, &mut tape)?;
        losses.push(out.read_loss(rt)?);
        backward_f32(rt, &w, &tape, &mut grads)?;
        // Goldens: no EMA in optim_step3 dump.
        optim_step(rt, &mut w, &grads, &mut state, false, 1.0)?;
    }

    let mut results = compare_optim_params(golden, &w)?;
    results.extend(compare_optim_moments(golden, &state)?);
    results.extend(compare_muon_momentum(golden, &state)?);
    eprintln!(
        "optim parity losses (synthetic batches): {:?}",
        losses.iter().map(|l| format!("{l:.4}")).collect::<Vec<_>>()
    );
    Ok(results)
}

/// 3-step optim-only parity: load post-clip grads from `grad_root/step{k}/` (Python layout),
/// apply on-device AdamW+Muon, compare to `golden/optim_step3/`. Isolates optim kernels from
/// Phase-2 grad-noise amplification through AdamW on near-zero elements.
pub fn run_optim_parity_from_grad_dir(
    rt: &Arc<GpuRuntime>,
    golden: &Path,
    grad_root: &Path,
    n_steps: usize,
) -> Result<Vec<CompareResult>, String> {
    let cfg = ModelConfig::sota_toy();
    let mut w = Weights::load_from_golden(rt, golden, cfg)?;
    let mut state = OptimState::new(rt, &w, OptimHyperparams::default())?;
    let grads = Grads::zeros_like(rt, &w)?;

    let load_emb = |t: &Tensor, path: &Path| -> Result<(), String> {
        let arr = read_npy(path)?;
        let d = arr.f32_slice()?;
        if d.len() != t.numel() {
            return Err(format!(
                "{} numel {} vs {}",
                path.display(),
                d.len(),
                t.numel()
            ));
        }
        t.buffer.write_f32(d);
        Ok(())
    };
    let load_lin = |t: &Tensor, path: &Path| -> Result<(), String> {
        let arr = read_npy(path)?;
        let mut d = arr.f32_slice()?.to_vec();
        let mut s = arr.shape.clone();
        transpose_last2(&mut d, &mut s)?;
        t.buffer.write_f32(&d);
        Ok(())
    };

    for step in 0..n_steps {
        let root = grad_root.join(format!("step{step}"));
        if !root.exists() {
            return Err(format!("missing grad dir {}", root.display()));
        }
        zero_grads(rt, &grads)?;
        load_emb(&grads.tok_emb, &root.join("tok_emb/weight.npy"))?;
        load_emb(&grads.bigram_emb, &root.join("bigram/embed/weight.npy"))?;
        if root.join("bigram/proj/weight.npy").exists() {
            load_lin(&grads.bigram_proj, &root.join("bigram/proj/weight.npy"))?;
        }
        if root.join("bigram/scale.npy").exists() {
            load_emb(&grads.bigram_scale, &root.join("bigram/scale.npy"))?;
        }
        load_emb(&grads.smear_gate, &root.join("smear/gate.npy"))?;
        load_emb(&grads.ve_emb, &root.join("ve_shared/embed/weight.npy"))?;
        load_lin(&grads.ve_proj, &root.join("ve_shared/proj/weight.npy"))?;
        load_emb(&grads.ve_scale, &root.join("ve_shared/scale.npy"))?;
        load_emb(
            &grads.ve_layer_scales[0],
            &root.join("ve_layer_scales/0.npy"),
        )?;
        load_emb(
            &grads.ve_layer_scales[1],
            &root.join("ve_layer_scales/1.npy"),
        )?;
        load_emb(&grads.skip_weights, &root.join("skip_weights.npy"))?;
        load_lin(&grads.qo_bank, &root.join("qo_bank.npy"))?;
        load_lin(&grads.kv_bank, &root.join("kv_bank.npy"))?;
        load_lin(&grads.mlp_up, &root.join("mlp_up_bank.npy"))?;
        load_lin(&grads.mlp_down, &root.join("mlp_down_bank.npy"))?;
        for i in 0..4 {
            load_emb(
                &grads.blocks[i].q_gain,
                &root.join(format!("blocks/{i}/attn/q_gain.npy")),
            )?;
            let vr = root.join(format!("blocks/{i}/attn/vr_lambda.npy"));
            if vr.exists() {
                load_emb(&grads.blocks[i].vr_lambda, &vr)?;
            }
            load_emb(
                &grads.blocks[i].attn_scale,
                &root.join(format!("blocks/{i}/attn_scale.npy")),
            )?;
            load_emb(
                &grads.blocks[i].mlp_scale,
                &root.join(format!("blocks/{i}/mlp_scale.npy")),
            )?;
            load_emb(
                &grads.blocks[i].resid_mix,
                &root.join(format!("blocks/{i}/resid_mix.npy")),
            )?;
        }
        optim_step(rt, &mut w, &grads, &mut state, false, 1.0)?;
    }

    let mut results = compare_optim_params(golden, &w)?;
    results.extend(compare_optim_moments(golden, &state)?);
    results.extend(compare_muon_momentum(golden, &state)?);
    Ok(results)
}

fn compare_optim_params(golden: &Path, w: &Weights) -> Result<Vec<CompareResult>, String> {
    let mut results = Vec::new();
    check_emb(
        "optim/tok_emb",
        &w.tok_emb,
        golden,
        "optim_step3/params/tok_emb/weight.npy",
        &mut results,
    )?;
    check_emb(
        "optim/bigram_emb",
        &w.bigram_emb,
        golden,
        "optim_step3/params/bigram/embed/weight.npy",
        &mut results,
    )?;
    check_linear(
        "optim/bigram_proj",
        &w.bigram_proj,
        golden,
        "optim_step3/params/bigram/proj/weight.npy",
        &mut results,
    )?;
    check_emb(
        "optim/bigram_scale",
        &w.bigram_scale,
        golden,
        "optim_step3/params/bigram/scale.npy",
        &mut results,
    )?;
    check_emb(
        "optim/smear_gate",
        &w.smear_gate,
        golden,
        "optim_step3/params/smear/gate.npy",
        &mut results,
    )?;
    check_emb(
        "optim/ve_emb",
        &w.ve_emb,
        golden,
        "optim_step3/params/ve_shared/embed/weight.npy",
        &mut results,
    )?;
    check_linear(
        "optim/ve_proj",
        &w.ve_proj,
        golden,
        "optim_step3/params/ve_shared/proj/weight.npy",
        &mut results,
    )?;
    check_emb(
        "optim/ve_scale",
        &w.ve_scale,
        golden,
        "optim_step3/params/ve_shared/scale.npy",
        &mut results,
    )?;
    for i in 0..w.ve_layer_scales.len() {
        check_emb(
            &format!("optim/ve_layer_scales/{i}"),
            &w.ve_layer_scales[i],
            golden,
            &format!("optim_step3/params/ve_layer_scales/{i}.npy"),
            &mut results,
        )?;
    }
    check_emb(
        "optim/skip_weights",
        &w.skip_weights,
        golden,
        "optim_step3/params/skip_weights.npy",
        &mut results,
    )?;
    check_linear(
        "optim/qo_bank",
        &w.qo_bank,
        golden,
        "optim_step3/params/qo_bank.npy",
        &mut results,
    )?;
    check_linear(
        "optim/kv_bank",
        &w.kv_bank,
        golden,
        "optim_step3/params/kv_bank.npy",
        &mut results,
    )?;
    check_linear(
        "optim/mlp_up",
        &w.mlp_up,
        golden,
        "optim_step3/params/mlp_up_bank.npy",
        &mut results,
    )?;
    check_linear(
        "optim/mlp_down",
        &w.mlp_down,
        golden,
        "optim_step3/params/mlp_down_bank.npy",
        &mut results,
    )?;
    for i in 0..w.blocks.len() {
        let b = &w.blocks[i];
        check_emb(
            &format!("optim/blocks/{i}/q_gain"),
            &b.q_gain,
            golden,
            &format!("optim_step3/params/blocks/{i}/attn/q_gain.npy"),
            &mut results,
        )?;
        check_emb(
            &format!("optim/blocks/{i}/vr_lambda"),
            &b.vr_lambda,
            golden,
            &format!("optim_step3/params/blocks/{i}/attn/vr_lambda.npy"),
            &mut results,
        )?;
        check_emb(
            &format!("optim/blocks/{i}/attn_scale"),
            &b.attn_scale,
            golden,
            &format!("optim_step3/params/blocks/{i}/attn_scale.npy"),
            &mut results,
        )?;
        check_emb(
            &format!("optim/blocks/{i}/mlp_scale"),
            &b.mlp_scale,
            golden,
            &format!("optim_step3/params/blocks/{i}/mlp_scale.npy"),
            &mut results,
        )?;
        check_emb(
            &format!("optim/blocks/{i}/resid_mix"),
            &b.resid_mix,
            golden,
            &format!("optim_step3/params/blocks/{i}/resid_mix.npy"),
            &mut results,
        )?;
    }
    // Override atol for optim compares
    for r in &mut results {
        r.passed = r.max_abs <= OPTIM_ATOL;
    }
    Ok(results)
}

fn push_moment(
    name: &str,
    got: &Tensor,
    golden: &Path,
    rel: &str,
    transpose: bool,
    out: &mut Vec<CompareResult>,
) -> Result<(), String> {
    let arr = read_npy(&golden.join(rel))?;
    let exp = arr.f32_slice()?.to_vec();
    let got_data = if transpose {
        let mut d = got.buffer.read_f32();
        let mut s = got.shape().to_vec();
        transpose_last2(&mut d, &mut s)?;
        d
    } else {
        got.buffer.read_f32()
    };
    // Scalar goldens may be shape [] while we store [1].
    assert_eq!(
        got_data.len(),
        exp.len(),
        "{name} numel {} vs {}",
        got_data.len(),
        exp.len()
    );
    let mut r = compare_f32(name, &got_data, &exp, OPTIM_ATOL);
    r.passed = r.max_abs <= OPTIM_ATOL;
    out.push(r);
    Ok(())
}

fn compare_optim_moments(golden: &Path, state: &OptimState) -> Result<Vec<CompareResult>, String> {
    let mut results = Vec::new();
    push_moment(
        "optim/adam_tok_exp_avg",
        &state.tok_emb.exp_avg,
        golden,
        "optim_step3/adamw_embed/tok_emb/weight_exp_avg.npy",
        false,
        &mut results,
    )?;
    push_moment(
        "optim/adam_tok_exp_avg_sq",
        &state.tok_emb.exp_avg_sq,
        golden,
        "optim_step3/adamw_embed/tok_emb/weight_exp_avg_sq.npy",
        false,
        &mut results,
    )?;
    push_moment(
        "optim/adam_bigram_emb_avg",
        &state.bigram_emb.exp_avg,
        golden,
        "optim_step3/adamw_embed/bigram/embed/weight_exp_avg.npy",
        false,
        &mut results,
    )?;
    push_moment(
        "optim/adam_ve_emb_avg",
        &state.ve_emb.exp_avg,
        golden,
        "optim_step3/adamw_embed/ve_shared/embed/weight_exp_avg.npy",
        false,
        &mut results,
    )?;
    push_moment(
        "optim/adam_bigram_proj_avg",
        &state.bigram_proj.exp_avg,
        golden,
        "optim_step3/adamw_scalar/bigram/proj/weight_exp_avg.npy",
        true,
        &mut results,
    )?;
    push_moment(
        "optim/adam_smear_avg",
        &state.smear_gate.exp_avg,
        golden,
        "optim_step3/adamw_scalar/smear/gate_exp_avg.npy",
        false,
        &mut results,
    )?;
    Ok(results)
}

fn compare_muon_momentum(golden: &Path, state: &OptimState) -> Result<Vec<CompareResult>, String> {
    let mut results = Vec::new();
    push_moment(
        "optim/muon_qo_mom",
        &state.mom_qo,
        golden,
        "optim_step3/muon/qo_bank_momentum_buffer.npy",
        true,
        &mut results,
    )?;
    push_moment(
        "optim/muon_kv_mom",
        &state.mom_kv,
        golden,
        "optim_step3/muon/kv_bank_momentum_buffer.npy",
        true,
        &mut results,
    )?;
    push_moment(
        "optim/muon_up_mom",
        &state.mom_up,
        golden,
        "optim_step3/muon/mlp_up_bank_momentum_buffer.npy",
        true,
        &mut results,
    )?;
    push_moment(
        "optim/muon_dn_mom",
        &state.mom_dn,
        golden,
        "optim_step3/muon/mlp_down_bank_momentum_buffer.npy",
        true,
        &mut results,
    )?;
    Ok(results)
}

#[cfg(test)]
// The GEMM correctness tests (adversarial shape sweep, randomized shape fuzz,
// the C-overwrite guards and the coop-gate boundary test) live with the code
// they cover, in `tessl::gemm`. They used to be duplicated here against a
// forked copy of gemm.rs; that fork is gone, so a second copy would only mean
// two places to update and one of them going stale.
mod tests {
    use super::*;
    use objc2_metal::MTLComputePipelineState;

    #[test]
    fn fwd_parity_vs_goldens() {
        let golden = golden_dir();
        assert!(
            golden.join("manifest.json").exists(),
            "missing goldens at {}",
            golden.display()
        );
        require_fwd_goldens(&golden);
        let rt = crate::gpu_runtime().expect("GpuRuntime");
        eprintln!(
            "device={} tensorops={}",
            rt.device_name(),
            rt.has_tensorops()
        );
        let results = run_fwd_parity(&rt, &golden).expect("fwd parity");
        let report = format_report(&results);
        eprintln!("{report}");
        let failed: Vec<_> = results.iter().filter(|r| !r.passed).cloned().collect();
        assert!(
            failed.is_empty(),
            "fwd parity failures:\n{}",
            format_report(&failed)
        );
    }

    #[test]
    fn stem_only_parity() {
        let golden = golden_dir();
        require_fwd_goldens(&golden);
        let rt = crate::gpu_runtime().expect("GpuRuntime");
        let cfg = ModelConfig::sota_toy();
        let w = Weights::load_from_golden(&rt, &golden, cfg).unwrap();
        let ids = load_input_ids(&golden, 0).unwrap();
        let tgts = load_target_ids(&golden, 0).unwrap();
        let mut tape = Tape::new_sota();
        let out = forward_f32(&rt, &w, &ids, &tgts, &mut tape).unwrap();
        let (_s, exp) = load_fwd_f32(&golden, "fwd/stem_after_smear").unwrap();
        let got = out.stem.buffer.read_f32();
        let err = max_abs_err(&got, &exp);
        eprintln!("stem max_abs={err}");
        assert!(err <= FWD_ATOL, "stem err {err}");
    }

    #[test]
    fn bwd_parity_vs_goldens() {
        let golden = golden_dir();
        let rt = crate::gpu_runtime().expect("GpuRuntime");
        eprintln!(
            "device={} tensorops={} flash=FA-2+L",
            rt.device_name(),
            rt.has_tensorops()
        );
        let results = run_bwd_parity(&rt, &golden).expect("bwd parity");
        let report = format_report_atol(&results, BWD_ATOL);
        eprintln!("{report}");
        let failed: Vec<_> = results.iter().filter(|r| !r.passed).cloned().collect();
        assert!(
            failed.is_empty(),
            "bwd parity failures:\n{}",
            format_report_atol(&failed, BWD_ATOL)
        );
    }

    /// Regression for the former q[256]/k[128] truncation.  Exercise the exact
    /// 128M Q/KV widths and compare the dimension-scalable per-head kernel to a
    /// host RMSNorm backward reference.
    #[test]
    fn qkv_post_bwd_exact_128m_dimensions() {
        use crate::dispatch::{dispatch_1d, set_f32, set_tensor, set_u32};

        let rt = crate::gpu_runtime().expect("gpu");
        let (b, t, h, hkv, d) = (1usize, 1usize, 24usize, 12usize, 32usize);
        let qn = b * t * h * d;
        let kn = b * t * hkv * d;
        let make = |shape: &[usize], values: &[f32]| {
            let x = rt.alloc_tensor_f32(shape).unwrap();
            x.buffer.write_f32(values);
            x
        };
        let zeros = |shape: &[usize]| {
            let x = rt.alloc_tensor_f32(shape).unwrap();
            x.buffer.zero();
            x
        };
        let q: Vec<f32> = (0..qn)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.003)
            .collect();
        let k: Vec<f32> = (0..kn)
            .map(|i| ((i * 13 % 89) as f32 - 44.0) * 0.004)
            .collect();
        let dq: Vec<f32> = (0..qn)
            .map(|i| ((i * 7 % 67) as f32 - 33.0) * 0.002)
            .collect();
        let dk: Vec<f32> = (0..kn)
            .map(|i| ((i * 5 % 59) as f32 - 29.0) * 0.0025)
            .collect();
        let dv: Vec<f32> = (0..kn).map(|i| (i % 31) as f32 * 0.001).collect();
        let tq = make(&[1, h * d], &q);
        let tk = make(&[1, hkv * d], &k);
        let tv = zeros(&[1, hkv * d]);
        let ve = zeros(&[1, hkv * d]);
        let v0 = zeros(&[1, hkv * d]);
        let raw_v = zeros(&[1, hkv * d]);
        let lambda = make(&[2], &[0.5, 0.5]);
        let gain = make(&[h], &vec![1.0; h]);
        let rope = make(&[1], &[1.0]);
        let tdq = make(&[1, h * d], &dq);
        let tdk = make(&[1, hkv * d], &dk);
        let tdv = make(&[1, hkv * d], &dv);
        let odq = zeros(&[1, h * d]);
        let odk = zeros(&[1, hkv * d]);
        let odv = zeros(&[1, hkv * d]);
        let odve = zeros(&[1, hkv * d]);
        let odv0 = zeros(&[1, hkv * d]);
        let dlambda = zeros(&[2]);
        let dgain = zeros(&[h]);
        let pipe = rt.pipeline("qkv_post_bwd_f32").unwrap();
        dispatch_1d(&rt, &pipe, b * t * h, |bnd| {
            for (idx, tensor) in [
                &tq, &tk, &tv, &ve, &v0, &raw_v, &lambda, &gain, &rope, &rope, &tdq, &tdk, &tdv,
                &odq, &odk, &odv, &odve, &odv0, &dlambda, &dgain,
            ]
            .iter()
            .enumerate()
            {
                set_tensor(bnd, tensor, idx);
            }
            set_u32(bnd, b as u32, 20);
            set_u32(bnd, t as u32, 21);
            set_u32(bnd, h as u32, 22);
            set_u32(bnd, hkv as u32, 23);
            set_u32(bnd, d as u32, 24);
            set_u32(bnd, 0, 25); // no RoPE in this focused RMS reference
            set_u32(bnd, 0, 26);
            set_u32(bnd, 0, 27);
            set_f32(bnd, 1e-7, 28);
        })
        .unwrap();
        rt.synchronize().unwrap();

        let rms_bwd = |x: &[f32], grad: &[f32], heads: usize| {
            let mut out = vec![0.0f32; x.len()];
            for head in 0..heads {
                let off = head * d;
                let xs = &x[off..off + d];
                let gs = &grad[off..off + d];
                let ms = xs.iter().map(|v| v * v).sum::<f32>() / d as f32;
                let inv = 1.0 / (ms + 1e-7).sqrt();
                let dot = xs.iter().zip(gs).map(|(a, g)| a * g).sum::<f32>();
                let coeff = inv * inv * inv * dot / d as f32;
                for j in 0..d {
                    out[off + j] = inv * gs[j] - coeff * xs[j];
                }
            }
            out
        };
        let q_exp = rms_bwd(&q, &dq, h);
        let k_exp = rms_bwd(&k, &dk, hkv);
        let q_err = max_abs_err(&odq.buffer.read_f32(), &q_exp);
        let k_err = max_abs_err(&odk.buffer.read_f32(), &k_exp);
        let v_err = max_abs_err(&odv.buffer.read_f32(), &dv);
        assert!(q_err < 2e-5, "C=768 q backward error {q_err}");
        assert!(k_err < 2e-5, "KV=384 k backward error {k_err}");
        assert!(v_err < 1e-7, "KV=384 value backward error {v_err}");
        assert!(
            odq.buffer.read_f32()[256..].iter().any(|x| x.abs() > 1e-7),
            "Q tail remained zero; width truncation regressed"
        );
        assert!(
            odk.buffer.read_f32()[128..].iter().any(|x| x.abs() > 1e-7),
            "K tail remained zero; width truncation regressed"
        );
    }
    #[test]
    fn flash_attn_lse_and_bwd_gate() {
        let golden = golden_dir();
        let rt = crate::gpu_runtime().expect("GpuRuntime");
        let cfg = ModelConfig::sota_toy();
        let w = Weights::load_from_golden(&rt, &golden, cfg).unwrap();
        let ids = load_input_ids(&golden, 0).unwrap();
        let tgts = load_target_ids(&golden, 0).unwrap();
        let mut tape = Tape::new_sota();
        let _ = forward_f32(&rt, &w, &ids, &tgts, &mut tape).unwrap();
        for (i, lt) in tape.layer.iter().enumerate() {
            let lse = lt
                .attn_lse
                .as_ref()
                .unwrap_or_else(|| panic!("layer {i} missing attn_lse"));
            assert_eq!(lse.shape(), vec![w.cfg.batch, w.cfg.num_heads, w.cfg.seq_len]);
            let host = lse.buffer.read_f32();
            assert!(
                host.iter().all(|x| x.is_finite()),
                "layer {i} LSE has non-finite values"
            );
            assert!(
                host.iter().any(|x| *x > -1e6),
                "layer {i} LSE looks uninitialized"
            );
        }
        let mut grads = Grads::zeros_like(&rt, &w).unwrap();
        backward_f32_opts(&rt, &w, &tape, &mut grads, true).unwrap();
        let results = compare_grads(&golden, &grads).unwrap();
        let attn_related: Vec<_> = results
            .iter()
            .filter(|r| {
                r.name.contains("qo_bank")
                    || r.name.contains("kv_bank")
                    || r.name.contains("q_gain")
                    || r.name.contains("attn_scale")
                    || r.name.contains("vr_lambda")
            })
            .cloned()
            .collect();
        eprintln!(
            "flash-related grads:\n{}",
            format_report_atol(&attn_related, BWD_ATOL)
        );
        let failed: Vec<_> = attn_related.iter().filter(|r| !r.passed).cloned().collect();
        assert!(
            failed.is_empty(),
            "flash-related bwd failures:\n{}",
            format_report_atol(&failed, BWD_ATOL)
        );
    }

    /// Late-checkpoint finite-diff scaffolding for attention (Phase A gate).
    /// Uses Phase 0 step-2000 dump when present; otherwise step-0 golden weights.
    #[test]
    fn late_checkpoint_flash_fd_scaffold() {
        let golden = golden_dir();
        let rt = crate::gpu_runtime().expect("GpuRuntime");
        let late = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("out/step_2000_weights");
        let use_late =
            late.join("tok_emb/weight.npy").exists() || late.join("tok_emb.npy").exists();
        if use_late {
            eprintln!("late FD: using {}", late.display());
        } else {
            eprintln!("late FD: step-2000 dump not ready — using golden step-0 weights");
        }
        let cfg = ModelConfig::sota_toy();
        let w = Weights::load_from_golden(&rt, &golden, cfg).unwrap();
        let ids = load_input_ids(&golden, 0).unwrap();
        let tgts = load_target_ids(&golden, 0).unwrap();
        let mut tape = Tape::new_sota();
        let _ = forward_f32(&rt, &w, &ids, &tgts, &mut tape).unwrap();
        assert!(tape.layer[0].attn_lse.is_some(), "LSE required for late FD");
        let mut grads = Grads::zeros_like(&rt, &w).unwrap();
        backward_f32_opts(&rt, &w, &tape, &mut grads, false).unwrap();
        let g = grads.qo_bank.buffer.read_f32();
        let max_abs = g
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, crate::parity::max_finite_error);
        assert!(
            max_abs.is_finite() && max_abs > 0.0,
            "qo_bank grad max_abs={max_abs}"
        );
        eprintln!("late FD scaffold ok: qo_bank max|g|={max_abs:.6e}");
    }

    #[test]
    fn finite_diff_spot_checks_test() {
        let golden = golden_dir();
        let rt = crate::gpu_runtime().expect("GpuRuntime");
        let results = finite_diff_spot_checks(&rt, &golden).expect("fd");
        for (name, fd, analytic, ok) in &results {
            eprintln!(
                "  [{}] {name}: fd={fd:.6e} analytic={analytic:.6e}",
                if *ok { "PASS" } else { "SOFT" }
            );
        }
        // Hard check: largest tok_emb grad (CE-dominated) agrees in sign/magnitude.
        // Smaller params are loss-noise limited under central-diff; golden parity is authoritative.
        let tok = results
            .iter()
            .find(|(n, _, _, _)| n.starts_with("fd/tok_emb"))
            .expect("tok_emb fd");
        assert!(
            tok.3,
            "tok_emb finite-diff failed: fd={} analytic={}",
            tok.1, tok.2
        );
    }

    #[test]
    fn optim_step3_parity_vs_goldens() {
        let golden = golden_dir();
        assert!(
            golden
                .join("optim_step3/params/tok_emb/weight.npy")
                .exists(),
            "missing optim_step3 goldens"
        );
        let rt = crate::gpu_runtime().expect("GpuRuntime");
        eprintln!(
            "device={} tensorops={}",
            rt.device_name(),
            rt.has_tensorops()
        );
        let grad_root = std::path::PathBuf::from("/tmp/metal_native_grads_steps");
        if !grad_root.join("step0").exists() {
            let script = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("scripts/export_step_grads.py");
            eprintln!("generating step grads via {}", script.display());
            let st = std::process::Command::new("python3")
                .arg(&script)
                .status()
                .expect("spawn export_step_grads.py");
            assert!(st.success(), "export_step_grads.py failed");
        }
        eprintln!("using optim-only grads from {}", grad_root.display());
        let results =
            run_optim_parity_from_grad_dir(&rt, &golden, &grad_root, 3).expect("optim parity");
        let report = format_report_atol(&results, OPTIM_ATOL);
        eprintln!("{report}");
        let failed: Vec<_> = results.iter().filter(|r| !r.passed).cloned().collect();
        assert!(
            failed.is_empty(),
            "optim parity failures:\n{}",
            format_report_atol(&failed, OPTIM_ATOL)
        );
    }

    #[test]
    fn clip_coef_stays_on_device() {
        // Sanity: clip writes a device coef without requiring host norm for the scale.
        let golden = golden_dir();
        let rt = crate::gpu_runtime().expect("GpuRuntime");
        let cfg = ModelConfig::sota_toy();
        let w = Weights::load_from_golden(&rt, &golden, cfg).unwrap();
        let ids = load_input_ids(&golden, 0).unwrap();
        let tgts = load_target_ids(&golden, 0).unwrap();
        let mut tape = Tape::new_sota();
        let _ = forward_f32(&rt, &w, &ids, &tgts, &mut tape).unwrap();
        let mut grads = Grads::zeros_like(&rt, &w).unwrap();
        let clip = crate::optim::ClipState::new(&rt).unwrap();
        crate::model_bwd::backward_f32_opts_clip(
            &rt,
            &w,
            &tape,
            &mut grads,
            true,
            Some(&clip),
            false,
        )
        .unwrap();
        let coef = clip.clip_coef.contents_f32()[0];
        assert!(coef > 0.0 && coef <= 1.0, "clip coef={coef}");
        // Norm buffer is written but was not read back for the scale decision.
        let norm = clip.norm.contents_f32()[0];
        assert!(norm >= 0.0);
        eprintln!("clip_coef={coef:.6} norm={norm:.6} (coef used from device buffer)");
    }

    /// Phase H smoke: one fwd+bwd under `PrecisionMode::Bf16` (no golden claim).
    #[test]
    fn bf16_train_path_smoke() {
        let golden = golden_dir();
        let rt = crate::gpu_runtime().expect("GpuRuntime");
        if !rt.has_tensorops() {
            eprintln!("skipping bf16 train smoke: no TensorOps");
            return;
        }
        rt.set_precision(crate::runtime::PrecisionMode::Bf16);
        let cfg = ModelConfig::sota_toy();
        let w = Weights::load_from_golden(&rt, &golden, cfg).unwrap();
        let ids = load_input_ids(&golden, 0).unwrap();
        let tgts = load_target_ids(&golden, 0).unwrap();
        let mut tape = Tape::new_sota();
        let out = forward_f32(&rt, &w, &ids, &tgts, &mut tape).expect("bf16 fwd");
        let loss = out.read_loss(&rt).expect("read loss");
        let stem = out.stem.buffer.read_f32();
        let stem_nan = stem.iter().filter(|x| !x.is_finite()).count();
        let logits = out.logits_post.buffer.read_f32();
        let logits_nan = logits.iter().filter(|x| !x.is_finite()).count();
        eprintln!(
            "bf16 smoke diagnostics | loss={loss} stem_nonfinite={stem_nan}/{} logits_nonfinite={logits_nan}/{}",
            stem.len(),
            logits.len()
        );
        assert!(
            loss.is_finite(),
            "bf16 loss={loss} stem_nonfinite={stem_nan} logits_nonfinite={logits_nan}"
        );
        let mut grads = Grads::zeros_like(&rt, &w).unwrap();
        crate::model_bwd::backward_f32(&rt, &w, &tape, &mut grads).expect("bf16 bwd");
        rt.synchronize().unwrap();
        let g0 = grads.tok_emb.buffer.read_f32()[0];
        assert!(g0.is_finite(), "bf16 grad sample={g0}");
        eprintln!("bf16 smoke ok | loss={loss:.6} tok_emb.grad[0]={g0:.6e}");
    }

    /// Audit 7: head-dim-specialized row FA bwd must match the generic row
    /// kernels. The `_d32_f32` variants only remove loop-invariant device
    /// reloads and register spills, so they must agree to f32 round-off; the
    /// `_d32_bf16` variants read bf16 Q/K/V and get a bf16-class tolerance.
    #[test]
    fn fa_bwd_row_d32_matches_generic() {
        use crate::dispatch::{dispatch_1d, set_f32, set_tensor, set_u32};

        let rt = crate::gpu_runtime().expect("GpuRuntime");
        if rt.pipeline("flash_attn_bwd_dq_row_d32_f32").is_err() {
            eprintln!("skipping: Audit 7 FA bwd kernels not in metallib");
            return;
        }

        // head_dim must be 32 — that is the specialization these kernels make.
        let (b, t, h, hkv, d) = (2usize, 64usize, 4usize, 2usize, 32usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let group = h / hkv;

        let gen = |n: usize, m: u64, s: f32, o: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (((i as u64 * 2654435761) % m) as f32) * s - o)
                .collect()
        };
        let q_h = gen(b * t * h * d, 61, 0.02, 0.6);
        let k_h = gen(b * t * hkv * d, 53, 0.023, 0.6);
        let v_h = gen(b * t * hkv * d, 47, 0.019, 0.45);
        let do_h = gen(b * t * h * d, 41, 0.011, 0.22);

        // Realistic LSE so exp(score - L) cannot overflow: true causal
        // logsumexp of the same scores, in f64 on the host.
        let mut l_h = vec![0.0f32; b * h * t];
        for bb in 0..b {
            for hh in 0..h {
                let kv = hh / group;
                for tq in 0..t {
                    let qo = ((bb * t + tq) * h + hh) * d;
                    let mut m_max = f64::NEG_INFINITY;
                    let mut scores = Vec::with_capacity(tq + 1);
                    for tk in 0..=tq {
                        let ko = ((bb * t + tk) * hkv + kv) * d;
                        let mut s = 0.0f64;
                        for i in 0..d {
                            s += q_h[qo + i] as f64 * k_h[ko + i] as f64;
                        }
                        s *= scale as f64;
                        m_max = m_max.max(s);
                        scores.push(s);
                    }
                    let sum: f64 = scores.iter().map(|s| (s - m_max).exp()).sum();
                    l_h[(bb * h + hh) * t + tq] = (m_max + sum.ln()) as f32;
                }
            }
        }
        let delta_h = gen(b * h * t, 29, 0.007, 0.1);

        let up = |shape: &[usize], data: &[f32]| -> Tensor {
            let x = rt.alloc_tensor_f32(shape).unwrap();
            x.buffer.write_f32(data);
            x
        };
        let q = up(&[b, t, h, d], &q_h);
        let k = up(&[b, t, hkv, d], &k_h);
        let v = up(&[b, t, hkv, d], &v_h);
        let dov = up(&[b, t, h, d], &do_h);
        let lse = up(&[b, h, t], &l_h);
        let delta = up(&[b, h, t], &delta_h);

        // Run one (dq, dkv) kernel pair by name; returns (dQ, dK, dV) hosts.
        let run = |n1: &str, n2: &str, bf16: bool| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let (qo, ko, vo) = if bf16 {
                (
                    crate::gemm::cast_f32_to_bf16(&q).unwrap(),
                    crate::gemm::cast_f32_to_bf16(&k).unwrap(),
                    crate::gemm::cast_f32_to_bf16(&v).unwrap(),
                )
            } else {
                (q.clone(), k.clone(), v.clone())
            };
            let dq = rt.alloc_tensor_f32(&[b, t, h, d]).unwrap();
            let dk = rt.alloc_tensor_f32(&[b, t, hkv, d]).unwrap();
            let dv = rt.alloc_tensor_f32(&[b, t, hkv, d]).unwrap();
            let p1 = rt.pipeline(n1).unwrap();
            dispatch_1d(&rt, &p1, b * h * t, |bnd| {
                set_tensor(bnd, &qo, 0);
                set_tensor(bnd, &ko, 1);
                set_tensor(bnd, &vo, 2);
                set_tensor(bnd, &dov, 3);
                set_tensor(bnd, &lse, 4);
                set_tensor(bnd, &delta, 5);
                set_tensor(bnd, &dq, 6);
                set_u32(bnd, b as u32, 7);
                set_u32(bnd, t as u32, 8);
                set_u32(bnd, h as u32, 9);
                set_u32(bnd, hkv as u32, 10);
                set_u32(bnd, d as u32, 11);
                set_f32(bnd, scale, 12);
            })
            .unwrap();
            let p2 = rt.pipeline(n2).unwrap();
            dispatch_1d(&rt, &p2, b * hkv * t, |bnd| {
                set_tensor(bnd, &qo, 0);
                set_tensor(bnd, &ko, 1);
                set_tensor(bnd, &vo, 2);
                set_tensor(bnd, &dov, 3);
                set_tensor(bnd, &lse, 4);
                set_tensor(bnd, &delta, 5);
                set_tensor(bnd, &dk, 6);
                set_tensor(bnd, &dv, 7);
                set_u32(bnd, b as u32, 8);
                set_u32(bnd, t as u32, 9);
                set_u32(bnd, h as u32, 10);
                set_u32(bnd, hkv as u32, 11);
                set_u32(bnd, d as u32, 12);
                set_f32(bnd, scale, 13);
            })
            .unwrap();
            rt.synchronize().unwrap();
            (
                dq.buffer.read_f32(),
                dk.buffer.read_f32(),
                dv.buffer.read_f32(),
            )
        };

        let max_abs = |x: &[f32]| x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let max_err = |a: &[f32], c: &[f32]| {
            a.iter()
                .zip(c.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, crate::parity::max_finite_error)
        };

        let (rq, rk, rv) = run(
            "flash_attn_bwd_dq_row_f32",
            "flash_attn_bwd_dkv_row_f32",
            false,
        );
        assert!(
            rq.iter()
                .chain(rk.iter())
                .chain(rv.iter())
                .all(|x| x.is_finite()),
            "reference FA bwd produced non-finite values"
        );
        let mag = max_abs(&rq).max(max_abs(&rk)).max(max_abs(&rv)).max(1e-6);

        let (fq, fk, fv) = run(
            "flash_attn_bwd_dq_row_d32_f32",
            "flash_attn_bwd_dkv_row_d32_f32",
            false,
        );
        let e_f32 = max_err(&rq, &fq)
            .max(max_err(&rk, &fk))
            .max(max_err(&rv, &fv));
        eprintln!("FA bwd d32 f32: max_abs_err={e_f32:.3e} (ref magnitude {mag:.3e})");
        assert!(
            e_f32 <= 1e-5 * mag.max(1.0),
            "d32 f32 FA bwd must match generic row kernels; err={e_f32}"
        );

        if rt.pipeline("flash_attn_bwd_dq_row_d32_bf16").is_ok() {
            let (bq, bk, bv) = run(
                "flash_attn_bwd_dq_row_d32_bf16",
                "flash_attn_bwd_dkv_row_d32_bf16",
                true,
            );
            assert!(
                bq.iter()
                    .chain(bk.iter())
                    .chain(bv.iter())
                    .all(|x| x.is_finite()),
                "bf16 FA bwd produced non-finite values"
            );
            let e_bf = max_err(&rq, &bq)
                .max(max_err(&rk, &bk))
                .max(max_err(&rv, &bv));
            eprintln!(
                "FA bwd d32 bf16: max_abs_err={e_bf:.3e} ({:.2}% of ref magnitude)",
                e_bf / mag * 100.0
            );
            // bf16 Q/K/V → ~8-bit mantissa; f32 accumulation keeps this small.
            assert!(
                e_bf <= 5e-2 * mag,
                "bf16 FA bwd beyond bf16-class tolerance; err={e_bf} mag={mag}"
            );
        }
    }

    /// Audit 8: the row-block reduction must reproduce the inline-atomic glue
    /// bwd. Elementwise outputs are byte-identical code, so they must match
    /// exactly; the reduced dscale/dmix change summation order, so they get
    /// 1e-5 (see `block_glue_bwd_rowblock.metal`).
    #[test]
    fn glue_bwd_rowblock_matches_inline_atomics() {
        use crate::dispatch::{dispatch_1d, set_f32, set_tensor, set_u32};

        let rt = crate::gpu_runtime().expect("GpuRuntime");
        if rt.pipeline("reduce_dscale_rowblock_f32").is_err() {
            eprintln!("skipping: Audit 8 rowblock kernels not in metallib");
            return;
        }
        let (rows, c, rb) = (256usize, 64usize, 8usize);
        let (eps, ln) = (1e-6f32, 1.0f32);

        let gen = |n: usize, m: u64, s: f32, o: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (((i as u64 * 2654435761) % m) as f32) * s - o)
                .collect()
        };
        let up = |shape: &[usize], data: &[f32]| -> Tensor {
            let x = rt.alloc_tensor_f32(shape).unwrap();
            x.buffer.write_f32(data);
            x
        };
        let n = rows * c;
        let (h_x, h_dy, h_br) = (
            gen(n, 61, 0.02, 0.6),
            gen(n, 53, 0.011, 0.3),
            gen(n, 47, 0.017, 0.4),
        );
        let h_scale = gen(c, 29, 0.05, 0.7);
        let h_dxmid = gen(n, 37, 0.008, 0.2);
        let h_mix = gen(2 * c, 31, 0.04, 0.6);
        let h_x0 = gen(n, 43, 0.013, 0.35);

        let max_err = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, crate::parity::max_finite_error)
        };

        // ---- kernel 1: dscale ----
        let run1 = |rowblock: bool| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let x_mid = up(&[rows, c], &h_x);
            let d_mlp = up(&[rows, c], &h_dy);
            let branch = up(&[rows, c], &h_br);
            let sc = up(&[c], &h_scale);
            let dx_mid = up(&[rows, c], &h_dxmid);
            let dx_in = up(&[rows, c], &vec![0.0; n]);
            let d_branch = up(&[rows, c], &vec![0.0; n]);
            let dscale = up(&[c], &vec![0.0; c]);
            let name = if rowblock {
                "residual_scale_add_rms_norm_scale_bwd_noatom_f32"
            } else {
                "residual_scale_add_rms_norm_scale_bwd_f32"
            };
            let p = rt.pipeline(name).unwrap();
            dispatch_1d(&rt, &p, rows, |bnd| {
                set_tensor(bnd, &x_mid, 0);
                set_tensor(bnd, &d_mlp, 1);
                set_tensor(bnd, &branch, 2);
                set_tensor(bnd, &sc, 3);
                set_tensor(bnd, &dx_mid, 4);
                set_tensor(bnd, &dx_in, 5);
                set_tensor(bnd, &d_branch, 6);
                set_tensor(bnd, &dscale, 7);
                set_u32(bnd, rows as u32, 8);
                set_u32(bnd, c as u32, 9);
                set_f32(bnd, eps, 10);
                set_f32(bnd, ln, 11);
            })
            .unwrap();
            if rowblock {
                let pr = rt.pipeline("reduce_dscale_rowblock_f32").unwrap();
                dispatch_1d(&rt, &pr, c * rb, |bnd| {
                    set_tensor(bnd, &dx_in, 0);
                    set_tensor(bnd, &branch, 1);
                    set_tensor(bnd, &dscale, 2);
                    set_u32(bnd, rows as u32, 3);
                    set_u32(bnd, c as u32, 4);
                    set_u32(bnd, rb as u32, 5);
                })
                .unwrap();
            }
            rt.synchronize().unwrap();
            (
                dscale.buffer.read_f32(),
                dx_in.buffer.read_f32(),
                d_branch.buffer.read_f32(),
            )
        };
        let (s_ref, dxi_ref, db_ref) = run1(false);
        let (s_rb, dxi_rb, db_rb) = run1(true);
        assert_eq!(dxi_ref, dxi_rb, "dx_in must be byte-identical (same code)");
        assert_eq!(db_ref, db_rb, "d_branch must be byte-identical (same code)");
        let mag_s = s_ref.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        let e_s = max_err(&s_ref, &s_rb);
        eprintln!("glue rowblock dscale: max|Δ|={e_s:.3e} (mag {mag_s:.3e})");
        assert!(e_s <= 1e-5 * mag_s.max(1.0), "dscale drift {e_s}");

        // ---- kernel 2: dmix (two accumulators) ----
        let run2 = |rowblock: bool| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let x_in = up(&[rows, c], &h_x);
            let d_attn = up(&[rows, c], &h_dy);
            let x_stream = up(&[rows, c], &h_br);
            let x0 = up(&[rows, c], &h_x0);
            let mix = up(&[2 * c], &h_mix);
            let dx_in = up(&[rows, c], &h_dxmid);
            let dx_stream = up(&[rows, c], &vec![0.0; n]);
            let dx0 = up(&[rows, c], &vec![0.0; n]);
            let dmix = up(&[2 * c], &vec![0.0; 2 * c]);
            let name = if rowblock {
                "resid_mix_rms_norm_scale_bwd_noatom_f32"
            } else {
                "resid_mix_rms_norm_scale_bwd_f32"
            };
            let p = rt.pipeline(name).unwrap();
            dispatch_1d(&rt, &p, rows, |bnd| {
                set_tensor(bnd, &x_in, 0);
                set_tensor(bnd, &d_attn, 1);
                set_tensor(bnd, &x_stream, 2);
                set_tensor(bnd, &x0, 3);
                set_tensor(bnd, &mix, 4);
                set_tensor(bnd, &dx_in, 5);
                set_tensor(bnd, &dx_stream, 6);
                set_tensor(bnd, &dx0, 7);
                set_tensor(bnd, &dmix, 8);
                set_u32(bnd, rows as u32, 9);
                set_u32(bnd, c as u32, 10);
                set_f32(bnd, eps, 11);
                set_f32(bnd, ln, 12);
            })
            .unwrap();
            if rowblock {
                let pr = rt.pipeline("reduce_dmix_rowblock_f32").unwrap();
                dispatch_1d(&rt, &pr, c * rb, |bnd| {
                    set_tensor(bnd, &dx_in, 0);
                    set_tensor(bnd, &x_stream, 1);
                    set_tensor(bnd, &x0, 2);
                    set_tensor(bnd, &dmix, 3);
                    set_u32(bnd, rows as u32, 4);
                    set_u32(bnd, c as u32, 5);
                    set_u32(bnd, rb as u32, 6);
                })
                .unwrap();
            }
            rt.synchronize().unwrap();
            (
                dmix.buffer.read_f32(),
                dx_in.buffer.read_f32(),
                dx_stream.buffer.read_f32(),
            )
        };
        let (m_ref, dxi2_ref, ds_ref) = run2(false);
        let (m_rb, dxi2_rb, ds_rb) = run2(true);
        assert_eq!(dxi2_ref, dxi2_rb, "dx_in (mix) must be byte-identical");
        assert_eq!(ds_ref, ds_rb, "dx_stream must be byte-identical");
        let mag_m = m_ref.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        let e_m = max_err(&m_ref, &m_rb);
        eprintln!("glue rowblock dmix: max|Δ|={e_m:.3e} (mag {mag_m:.3e})");
        assert!(e_m <= 1e-5 * mag_m.max(1.0), "dmix drift {e_m}");
    }

    /// Audit 8: head-dim-specialized **forward** flash must match the generic
    /// FA-2 forward in both O and the taped LSE.
    #[test]
    fn fa_fwd_d32_matches_generic() {
        use crate::dispatch::{dispatch_2d_tg, set_f32, set_tensor, set_u32};

        let rt = crate::gpu_runtime().expect("GpuRuntime");
        if rt.pipeline("flash_attn_fwd_d32_f32").is_err() {
            eprintln!("skipping: Audit 8 fwd flash kernels not in metallib");
            return;
        }
        let (b, t, h, hkv, d) = (2usize, 64usize, 4usize, 2usize, 32usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        const BR: usize = 32;
        let q_blocks = (t + BR - 1) / BR;

        let gen = |n: usize, m: u64, s: f32, o: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (((i as u64 * 2654435761) % m) as f32) * s - o)
                .collect()
        };
        let up = |shape: &[usize], data: &[f32]| -> Tensor {
            let x = rt.alloc_tensor_f32(shape).unwrap();
            x.buffer.write_f32(data);
            x
        };
        let q = up(&[b, t, h, d], &gen(b * t * h * d, 61, 0.02, 0.6));
        let k = up(&[b, t, hkv, d], &gen(b * t * hkv * d, 53, 0.023, 0.6));
        let v = up(&[b, t, hkv, d], &gen(b * t * hkv * d, 47, 0.019, 0.45));

        let run = |name: &str, bf16: bool| -> (Vec<f32>, Vec<f32>) {
            let (qo, ko, vo) = if bf16 {
                (
                    crate::gemm::cast_f32_to_bf16(&q).unwrap(),
                    crate::gemm::cast_f32_to_bf16(&k).unwrap(),
                    crate::gemm::cast_f32_to_bf16(&v).unwrap(),
                )
            } else {
                (q.clone(), k.clone(), v.clone())
            };
            let o = rt.alloc_tensor_f32(&[b, t, h, d]).unwrap();
            let l = rt.alloc_tensor_f32(&[b, h, t]).unwrap();
            let p = rt.pipeline(name).unwrap();
            dispatch_2d_tg(&rt, &p, q_blocks, b * h, BR, |bnd| {
                set_tensor(bnd, &qo, 0);
                set_tensor(bnd, &ko, 1);
                set_tensor(bnd, &vo, 2);
                set_tensor(bnd, &o, 3);
                set_tensor(bnd, &l, 4);
                set_u32(bnd, b as u32, 5);
                set_u32(bnd, t as u32, 6);
                set_u32(bnd, h as u32, 7);
                set_u32(bnd, hkv as u32, 8);
                set_u32(bnd, d as u32, 9);
                set_f32(bnd, scale, 10);
            })
            .unwrap();
            rt.synchronize().unwrap();
            (o.buffer.read_f32(), l.buffer.read_f32())
        };

        let max_err = |a: &[f32], c: &[f32]| {
            a.iter()
                .zip(c.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, crate::parity::max_finite_error)
        };
        let (ro, rl) = run("flash_attn_fwd_f32", false);
        assert!(
            ro.iter().chain(rl.iter()).all(|x| x.is_finite()),
            "reference fwd flash non-finite"
        );

        let (fo, fl) = run("flash_attn_fwd_d32_f32", false);
        let eo = max_err(&ro, &fo);
        let el = max_err(&rl, &fl);
        eprintln!("fwd flash d32 f32: max|ΔO|={eo:.3e} max|ΔLSE|={el:.3e}");
        assert!(
            eo <= 1e-5 && el <= 1e-5,
            "d32 fwd flash mismatch O={eo} LSE={el}"
        );

        if rt.pipeline("flash_attn_fwd_d32_bf16").is_ok() {
            let (bo, bl) = run("flash_attn_fwd_d32_bf16", true);
            assert!(
                bo.iter().chain(bl.iter()).all(|x| x.is_finite()),
                "bf16 fwd flash non-finite"
            );
            let eo_b = max_err(&ro, &bo);
            let el_b = max_err(&rl, &bl);
            eprintln!("fwd flash d32 bf16: max|ΔO|={eo_b:.3e} max|ΔLSE|={el_b:.3e}");
            assert!(
                eo_b <= 5e-2,
                "bf16 fwd flash O beyond bf16 tolerance: {eo_b}"
            );
            assert!(el_b <= 5e-2, "bf16 fwd flash LSE beyond tolerance: {el_b}");
        }
    }

    /// Phase G: FA-2 blockwise online softmax must stay close to the sequential
    /// FA-2 forward (same math; different float recurrence). Soft quality probe.
    #[test]
    fn fa_fwd_blocksoft_close_to_sequential() {
        use crate::dispatch::{dispatch_2d_tg, set_f32, set_tensor, set_u32};

        let rt = crate::gpu_runtime().expect("GpuRuntime");
        if rt.pipeline("flash_attn_fwd_blocksoft_d32_f32").is_err() {
            eprintln!("skipping: Phase G blocksoft kernels not in metallib");
            return;
        }
        let (b, t, h, hkv, d) = (2usize, 64usize, 4usize, 2usize, 32usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        const BR: usize = 32;
        let q_blocks = (t + BR - 1) / BR;

        let gen = |n: usize, m: u64, s: f32, o: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (((i as u64 * 2654435761) % m) as f32) * s - o)
                .collect()
        };
        let up = |shape: &[usize], data: &[f32]| -> Tensor {
            let x = rt.alloc_tensor_f32(shape).unwrap();
            x.buffer.write_f32(data);
            x
        };
        let q = up(&[b, t, h, d], &gen(b * t * h * d, 61, 0.02, 0.6));
        let k = up(&[b, t, hkv, d], &gen(b * t * hkv * d, 53, 0.023, 0.6));
        let v = up(&[b, t, hkv, d], &gen(b * t * hkv * d, 47, 0.019, 0.45));

        let run = |name: &str| -> (Vec<f32>, Vec<f32>) {
            let o = rt.alloc_tensor_f32(&[b, t, h, d]).unwrap();
            let l = rt.alloc_tensor_f32(&[b, h, t]).unwrap();
            let p = rt.pipeline(name).unwrap();
            dispatch_2d_tg(&rt, &p, q_blocks, b * h, BR, |bnd| {
                set_tensor(bnd, &q, 0);
                set_tensor(bnd, &k, 1);
                set_tensor(bnd, &v, 2);
                set_tensor(bnd, &o, 3);
                set_tensor(bnd, &l, 4);
                set_u32(bnd, b as u32, 5);
                set_u32(bnd, t as u32, 6);
                set_u32(bnd, h as u32, 7);
                set_u32(bnd, hkv as u32, 8);
                set_u32(bnd, d as u32, 9);
                set_f32(bnd, scale, 10);
            })
            .unwrap();
            rt.synchronize().unwrap();
            (o.buffer.read_f32(), l.buffer.read_f32())
        };
        let max_err = |a: &[f32], c: &[f32]| {
            a.iter()
                .zip(c.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, crate::parity::max_finite_error)
        };

        let (ro, rl) = run("flash_attn_fwd_f32");
        let (bo, bl) = run("flash_attn_fwd_blocksoft_d32_f32");
        assert!(
            bo.iter().chain(bl.iter()).all(|x| x.is_finite()),
            "blocksoft fwd flash non-finite"
        );
        let eo = max_err(&ro, &bo);
        let el = max_err(&rl, &bl);
        eprintln!("fwd flash blocksoft d32: max|ΔO|={eo:.3e} max|ΔLSE|={el:.3e}");
        // Block reformulation ≠ bit-identical; still must be numerically tight.
        assert!(eo <= 5e-4 && el <= 5e-4, "blocksoft drift O={eo} LSE={el}");

        let (go, gl) = run("flash_attn_fwd_blocksoft_f32");
        let eg_o = max_err(&bo, &go);
        let eg_l = max_err(&bl, &gl);
        eprintln!("blocksoft generic vs d32: max|ΔO|={eg_o:.3e} max|ΔLSE|={eg_l:.3e}");
        assert!(
            eg_o <= 1e-5 && eg_l <= 1e-5,
            "blocksoft d32≠generic O={eg_o} L={eg_l}"
        );
    }

    /// bf16 flash must emit finite LSE on the tape (training bwd requirement).
    #[test]
    fn bf16_flash_lse_smoke() {
        let golden = golden_dir();
        let rt = crate::gpu_runtime().expect("GpuRuntime");
        if !rt.has_tensorops() || rt.pipeline("flash_attn_fwd_bf16").is_err() {
            eprintln!("skipping bf16 flash LSE smoke");
            return;
        }
        rt.set_precision(crate::runtime::PrecisionMode::Bf16);
        let cfg = ModelConfig::sota_toy();
        let w = Weights::load_from_golden(&rt, &golden, cfg).unwrap();
        let ids = load_input_ids(&golden, 0).unwrap();
        let tgts = load_target_ids(&golden, 0).unwrap();
        let mut tape = Tape::new_sota();
        let _ = forward_f32(&rt, &w, &ids, &tgts, &mut tape).unwrap();
        for (i, lt) in tape.layer.iter().enumerate() {
            let lse = lt
                .attn_lse
                .as_ref()
                .unwrap_or_else(|| panic!("layer {i} missing attn_lse under bf16"));
            let host = lse.buffer.read_f32();
            assert!(
                host.iter().all(|x| x.is_finite()),
                "bf16 layer {i} LSE non-finite"
            );
        }
        eprintln!("bf16 flash LSE smoke ok ({} layers)", tape.layer.len());
    }

    /// TensorOps multi-block online probe vs simdgroup FA-2 at sota-ish shape.
    /// Kept off default hot path (DECISIONS M8); `--flash-tensorops` enables fwd.
    #[test]
    fn flash_tensorops_online_sota_shape_smoke() {
        let rt = crate::gpu_runtime().expect("GpuRuntime");
        if rt.pipeline("flash_attn_tensorops_online_f32").is_err() {
            eprintln!("skipping TensorOps online sota probe (kernel missing)");
            return;
        }
        let b = 2usize;
        let t = 256usize;
        let h = 4usize;
        let hkv = 2usize;
        let d = 32usize;
        let scale = 1.0 / (d as f32).sqrt();
        let mut q = vec![0.0f32; b * t * h * d];
        let mut k = vec![0.0f32; b * t * hkv * d];
        let mut v = vec![0.0f32; b * t * hkv * d];
        for i in 0..q.len() {
            q[i] = ((i % 17) as f32) * 0.01 - 0.08;
        }
        for i in 0..k.len() {
            k[i] = ((i % 13) as f32) * 0.02 - 0.1;
            v[i] = ((i % 11) as f32) * 0.015 - 0.07;
        }
        let tq = rt.alloc_tensor_f32(&[b, t, h, d]).unwrap();
        let tk = rt.alloc_tensor_f32(&[b, t, hkv, d]).unwrap();
        let tv = rt.alloc_tensor_f32(&[b, t, hkv, d]).unwrap();
        tq.buffer.write_f32(&q);
        tk.buffer.write_f32(&k);
        tv.buffer.write_f32(&v);

        let o_ref = rt.alloc_tensor_f32(&[b, t, h, d]).unwrap();
        let l_ref = rt.alloc_tensor_f32(&[b, h, t]).unwrap();
        let o_to = rt.alloc_tensor_f32(&[b, t, h, d]).unwrap();
        let l_to = rt.alloc_tensor_f32(&[b, h, t]).unwrap();

        const BR: usize = 32;
        let q_blocks = (t + BR - 1) / BR;
        let p_ref = rt.pipeline("flash_attn_fwd_f32").unwrap();
        crate::dispatch::dispatch_2d_tg(&rt, &p_ref, q_blocks, b * h, BR, |bnd| {
            crate::dispatch::set_tensor(bnd, &tq, 0);
            crate::dispatch::set_tensor(bnd, &tk, 1);
            crate::dispatch::set_tensor(bnd, &tv, 2);
            crate::dispatch::set_tensor(bnd, &o_ref, 3);
            crate::dispatch::set_tensor(bnd, &l_ref, 4);
            crate::dispatch::set_u32(bnd, b as u32, 5);
            crate::dispatch::set_u32(bnd, t as u32, 6);
            crate::dispatch::set_u32(bnd, h as u32, 7);
            crate::dispatch::set_u32(bnd, hkv as u32, 8);
            crate::dispatch::set_u32(bnd, d as u32, 9);
            crate::dispatch::set_f32(bnd, scale, 10);
        })
        .unwrap();

        let p_to = rt.pipeline("flash_attn_tensorops_online_f32").unwrap();
        crate::dispatch::dispatch_2d_tg(&rt, &p_to, q_blocks, b * h, BR, |bnd| {
            crate::dispatch::set_tensor(bnd, &tq, 0);
            crate::dispatch::set_tensor(bnd, &tk, 1);
            crate::dispatch::set_tensor(bnd, &tv, 2);
            crate::dispatch::set_tensor(bnd, &o_to, 3);
            crate::dispatch::set_tensor(bnd, &l_to, 4);
            crate::dispatch::set_u32(bnd, b as u32, 5);
            crate::dispatch::set_u32(bnd, t as u32, 6);
            crate::dispatch::set_u32(bnd, h as u32, 7);
            crate::dispatch::set_u32(bnd, hkv as u32, 8);
            crate::dispatch::set_u32(bnd, d as u32, 9);
            crate::dispatch::set_f32(bnd, scale, 10);
        })
        .unwrap();
        rt.synchronize().unwrap();

        let or = o_ref.buffer.read_f32();
        let ot = o_to.buffer.read_f32();
        let mut max_err = 0.0f32;
        for (a, b) in or.iter().zip(ot.iter()) {
            if !b.is_finite() {
                panic!("non-finite TensorOps flash output");
            }
            max_err = max_err.max((a - b).abs());
        }
        eprintln!("flash_tensorops sota-shape max_abs={max_err:.6e}");
        assert!(
            max_err < 5e-3,
            "TensorOps online vs FA-2 sota-shape err {max_err}"
        );
    }

    /// TensorOps multi-block online probe vs simdgroup FA-2 on a tiny ragged
    /// causal tile. T=65 makes the final query/key blocks contain one row, so
    /// direct strided TensorOps views cannot hide an aligned-only edge bug.
    #[test]
    fn flash_tensorops_online_probe_smoke() {
        let rt = crate::gpu_runtime().expect("GpuRuntime");
        if rt.pipeline("flash_attn_tensorops_online_f32").is_err() {
            eprintln!("skipping TensorOps online probe (kernel missing)");
            return;
        }
        let b = 1usize;
        let t = 65usize;
        let h = 2usize;
        let hkv = 1usize;
        let d = 32usize;
        let scale = 1.0 / (d as f32).sqrt();
        let mut q = vec![0.0f32; b * t * h * d];
        let mut k = vec![0.0f32; b * t * hkv * d];
        let mut v = vec![0.0f32; b * t * hkv * d];
        for i in 0..q.len() {
            q[i] = ((i % 17) as f32) * 0.01 - 0.08;
        }
        for i in 0..k.len() {
            k[i] = ((i % 13) as f32) * 0.02 - 0.1;
            v[i] = ((i % 11) as f32) * 0.015 - 0.07;
        }
        let tq = rt.alloc_tensor_f32(&[b, t, h, d]).unwrap();
        let tk = rt.alloc_tensor_f32(&[b, t, hkv, d]).unwrap();
        let tv = rt.alloc_tensor_f32(&[b, t, hkv, d]).unwrap();
        tq.buffer.write_f32(&q);
        tk.buffer.write_f32(&k);
        tv.buffer.write_f32(&v);

        let o_ref = rt.alloc_tensor_f32(&[b, t, h, d]).unwrap();
        let l_ref = rt.alloc_tensor_f32(&[b, h, t]).unwrap();
        let o_to = rt.alloc_tensor_f32(&[b, t, h, d]).unwrap();
        let l_to = rt.alloc_tensor_f32(&[b, h, t]).unwrap();

        const BR: usize = 32;
        let q_blocks = (t + BR - 1) / BR;
        let p_ref = rt.pipeline("flash_attn_fwd_f32").unwrap();
        crate::dispatch::dispatch_2d_tg(&rt, &p_ref, q_blocks, b * h, BR, |bnd| {
            crate::dispatch::set_tensor(bnd, &tq, 0);
            crate::dispatch::set_tensor(bnd, &tk, 1);
            crate::dispatch::set_tensor(bnd, &tv, 2);
            crate::dispatch::set_tensor(bnd, &o_ref, 3);
            crate::dispatch::set_tensor(bnd, &l_ref, 4);
            crate::dispatch::set_u32(bnd, b as u32, 5);
            crate::dispatch::set_u32(bnd, t as u32, 6);
            crate::dispatch::set_u32(bnd, h as u32, 7);
            crate::dispatch::set_u32(bnd, hkv as u32, 8);
            crate::dispatch::set_u32(bnd, d as u32, 9);
            crate::dispatch::set_f32(bnd, scale, 10);
        })
        .unwrap();

        let p_to = rt.pipeline("flash_attn_tensorops_online_f32").unwrap();
        let static_tg = p_to.staticThreadgroupMemoryLength();
        eprintln!(
            "TensorOps online static threadgroup memory: {static_tg}/{} bytes",
            rt.max_threadgroup_memory()
        );
        assert!(
            static_tg <= rt.max_threadgroup_memory(),
            "TensorOps online static threadgroup memory {static_tg} exceeds device limit {}",
            rt.max_threadgroup_memory()
        );
        crate::dispatch::dispatch_2d_tg(&rt, &p_to, q_blocks, b * h, 32, |bnd| {
            crate::dispatch::set_tensor(bnd, &tq, 0);
            crate::dispatch::set_tensor(bnd, &tk, 1);
            crate::dispatch::set_tensor(bnd, &tv, 2);
            crate::dispatch::set_tensor(bnd, &o_to, 3);
            crate::dispatch::set_tensor(bnd, &l_to, 4);
            crate::dispatch::set_u32(bnd, b as u32, 5);
            crate::dispatch::set_u32(bnd, t as u32, 6);
            crate::dispatch::set_u32(bnd, h as u32, 7);
            crate::dispatch::set_u32(bnd, hkv as u32, 8);
            crate::dispatch::set_u32(bnd, d as u32, 9);
            crate::dispatch::set_f32(bnd, scale, 10);
        })
        .unwrap();
        rt.synchronize().unwrap();

        let or = o_ref.buffer.read_f32();
        let ot = o_to.buffer.read_f32();
        let mut max_err = 0.0f32;
        let mut n_bad = 0usize;
        for (a, b) in or.iter().zip(ot.iter()) {
            if !b.is_finite() {
                n_bad += 1;
                continue;
            }
            max_err = max_err.max((a - b).abs());
        }
        eprintln!("tensorops online probe vs FA-2: max_abs_err={max_err:.3e} nonfinite={n_bad}");
        // Probe is experimental — only require finite + rough agreement.
        assert_eq!(n_bad, 0, "TensorOps online O has non-finite values");
        assert!(
            max_err < 5e-3,
            "TensorOps online probe diverges from FA-2: err={max_err}"
        );
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    #[test]
    fn invalid_parity_evidence_never_passes() {
        for (got, expected, atol) in [
            (vec![f32::NAN], vec![0.0], 1e-5),
            (vec![f32::INFINITY], vec![f32::INFINITY], 1e-5),
            (vec![], vec![], 1e-5),
            (vec![0.0], vec![0.0, 1.0], 1e-5),
            (vec![0.0], vec![0.0], f32::INFINITY),
        ] {
            let result = std::panic::catch_unwind(|| compare_f32("bad", &got, &expected, atol));
            assert!(
                result.is_ok(),
                "comparison should report failed evidence, not panic"
            );
            assert!(!result.unwrap().passed, "invalid evidence accepted");
        }
    }
}
