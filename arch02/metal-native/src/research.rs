//! Device-reduced optimizer-study telemetry. Weight snapshots remain on the GPU;
//! only fixed-size scalar summaries cross to the host on requested log steps.

use std::sync::Arc;

use objc2_metal::MTLComputePipelineState;
use serde::Serialize;

use crate::dispatch::{set_f32, set_gpu_buf, set_tensor, set_u32};
use crate::model_bwd::Grads;
use crate::runtime::{mtl_size, GpuRuntime};
use crate::tensor::{GpuBuffer, Tensor};
use crate::weights::Weights;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Matrix = 0,
    Embed = 1,
    Auxiliary = 2,
}

#[derive(Debug, Clone, Serialize)]
pub struct RoleNorms {
    pub matrix: f64,
    pub embedding: f64,
    pub auxiliary: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResearchTelemetry {
    pub gradient_norm_by_role: RoleNorms,
    pub update_norm_by_role: RoleNorms,
    pub orthogonality_error_sampled: f64,
    pub row_log_drift: f64,
    pub spectral_proxy_log_drift: f64,
    pub nonfinite_values: u64,
}

pub struct WeightSnapshot {
    tensors: Vec<Tensor>,
    roles: Vec<Role>,
}

fn weight_refs(w: &Weights) -> (Vec<&Tensor>, Vec<Role>) {
    let mut tensors = vec![
        &w.tok_emb,
        &w.bigram_emb,
        &w.ve_emb,
        &w.bigram_proj,
        &w.bigram_scale,
        &w.smear_gate,
        &w.ve_proj,
        &w.ve_scale,
        &w.skip_weights,
        &w.qo_bank,
        &w.kv_bank,
        &w.mlp_up,
        &w.mlp_down,
    ];
    let mut roles = vec![
        Role::Embed,
        Role::Embed,
        Role::Embed,
        Role::Auxiliary,
        Role::Auxiliary,
        Role::Auxiliary,
        Role::Auxiliary,
        Role::Auxiliary,
        Role::Auxiliary,
        Role::Matrix,
        Role::Matrix,
        Role::Matrix,
        Role::Matrix,
    ];
    for t in &w.ve_layer_scales {
        tensors.push(t);
        roles.push(Role::Auxiliary);
    }
    for b in &w.blocks {
        tensors.extend([
            &b.q_gain,
            &b.vr_lambda,
            &b.attn_scale,
            &b.mlp_scale,
            &b.resid_mix,
        ]);
        roles.extend([Role::Auxiliary; 5]);
    }
    (tensors, roles)
}

fn grad_refs(g: &Grads) -> Vec<&Tensor> {
    let mut tensors = vec![
        &g.tok_emb,
        &g.bigram_emb,
        &g.ve_emb,
        &g.bigram_proj,
        &g.bigram_scale,
        &g.smear_gate,
        &g.ve_proj,
        &g.ve_scale,
        &g.skip_weights,
        &g.qo_bank,
        &g.kv_bank,
        &g.mlp_up,
        &g.mlp_down,
    ];
    for t in &g.ve_layer_scales {
        tensors.push(t);
    }
    for b in &g.blocks {
        tensors.extend([
            &b.q_gain,
            &b.vr_lambda,
            &b.attn_scale,
            &b.mlp_scale,
            &b.resid_mix,
        ]);
    }
    tensors
}

pub fn capture_weight_snapshot(w: &Weights) -> Result<WeightSnapshot, String> {
    let (refs, roles) = weight_refs(w);
    let tensors = refs
        .into_iter()
        .map(Tensor::deep_copy)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(WeightSnapshot { tensors, roles })
}

fn scalar(rt: &Arc<GpuRuntime>) -> Result<GpuBuffer, String> {
    let value = rt.alloc_buffer(4)?;
    value.zero();
    Ok(value)
}

pub fn collect_research_telemetry(
    rt: &Arc<GpuRuntime>,
    snapshot: &WeightSnapshot,
    w: &Weights,
    grads: &Grads,
) -> Result<ResearchTelemetry, String> {
    let (after, roles) = weight_refs(w);
    let grad = grad_refs(grads);
    if after.len() != snapshot.tensors.len() || after.len() != grad.len() || roles != snapshot.roles
    {
        return Err("research telemetry tensor ordering mismatch".into());
    }
    let update_sq = [scalar(rt)?, scalar(rt)?, scalar(rt)?];
    let grad_sq = [scalar(rt)?, scalar(rt)?, scalar(rt)?];
    let nonfinite = scalar(rt)?;
    let row_drift = scalar(rt)?;
    let spectral = scalar(rt)?;
    let orth = scalar(rt)?;
    let diff = rt.pipeline("research_diff_sq_reduce_f32")?;
    for i in 0..after.len() {
        let n = after[i].numel();
        let width = diff.threadExecutionWidth() as usize;
        let tpt = width.min(n).max(1);
        let groups = n.div_ceil(tpt);
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&diff);
            set_tensor(bnd, &snapshot.tensors[i], 0);
            set_tensor(bnd, after[i], 1);
            set_tensor(bnd, grad[i], 2);
            set_gpu_buf(bnd, &update_sq[roles[i] as usize], 3);
            set_gpu_buf(bnd, &grad_sq[roles[i] as usize], 4);
            set_gpu_buf(bnd, &nonfinite, 5);
            set_u32(bnd, n as u32, 6);
            bnd.dispatch(mtl_size(groups, 1, 1), mtl_size(tpt, 1, 1));
            Ok(())
        })?;
    }
    let drift = rt.pipeline("research_matrix_drift_f32")?;
    let c = w.cfg.model_dim;
    let banks = [
        (9usize, 2 * w.cfg.num_layers, c, c),
        (10, 2 * w.cfg.num_layers, c, w.cfg.kv_dim()),
        (11, w.cfg.num_layers, c, w.cfg.mlp_dim),
        (12, w.cfg.num_layers, w.cfg.mlp_dim, c),
    ];
    let mut row_count = 0usize;
    let mut matrix_count = 0usize;
    let mut pair_count = 0usize;
    for (index, batch, rows, cols) in banks {
        row_count += batch * cols;
        matrix_count += batch;
        pair_count += batch * cols.saturating_sub(1);
        rt.with_binder(|bnd| {
            bnd.set_pipeline(&drift);
            set_tensor(bnd, &snapshot.tensors[index], 0);
            set_tensor(bnd, after[index], 1);
            set_gpu_buf(bnd, &row_drift, 2);
            set_gpu_buf(bnd, &spectral, 3);
            set_gpu_buf(bnd, &orth, 4);
            set_u32(bnd, batch as u32, 5);
            set_u32(bnd, rows as u32, 6);
            set_u32(bnd, cols as u32, 7);
            set_f32(bnd, 1e-12, 8);
            bnd.dispatch(mtl_size(batch, 1, 1), mtl_size(256, 1, 1));
            Ok(())
        })?;
    }
    rt.synchronize()?;
    // Copy each scalar before opening the next exclusive mapping. Temporary
    // guards in a struct literal otherwise live until the whole statement ends.
    let scalar = |buffer: &GpuBuffer| buffer.contents_f32()[0];
    let role_norms = |values: &[GpuBuffer; 3]| RoleNorms {
        matrix: (scalar(&values[0]) as f64).sqrt(),
        embedding: (scalar(&values[1]) as f64).sqrt(),
        auxiliary: (scalar(&values[2]) as f64).sqrt(),
    };
    Ok(ResearchTelemetry {
        gradient_norm_by_role: role_norms(&grad_sq),
        update_norm_by_role: role_norms(&update_sq),
        orthogonality_error_sampled: scalar(&orth) as f64 / pair_count.max(1) as f64,
        row_log_drift: scalar(&row_drift) as f64 / row_count.max(1) as f64,
        spectral_proxy_log_drift: scalar(&spectral) as f64 / matrix_count.max(1) as f64,
        nonfinite_values: scalar(&nonfinite).max(0.0) as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init::init_weights_seeded;
    use crate::model_bwd::Grads;
    use crate::weights::ModelConfig;

    #[test]
    fn device_research_telemetry_reports_role_norms_and_drift() {
        let rt = crate::gpu_runtime().expect("gpu");
        let w = init_weights_seeded(&rt, ModelConfig::sota_toy(), 19).expect("weights");
        let grads = Grads::zeros_like(&rt, &w).expect("grads");
        for value in grads.qo_bank.buffer.contents_f32().iter_mut() {
            *value = 1.0;
        }
        let snapshot = capture_weight_snapshot(&w).expect("snapshot");
        rt.synchronize().expect("snapshot sync");
        w.qo_bank.buffer.contents_f32()[0] += 3.0;
        let telemetry = collect_research_telemetry(&rt, &snapshot, &w, &grads).expect("telemetry");
        assert!((telemetry.update_norm_by_role.matrix - 3.0).abs() < 1e-5);
        assert!(telemetry.gradient_norm_by_role.matrix > 1.0);
        assert_eq!(telemetry.nonfinite_values, 0);
        assert!(telemetry.row_log_drift > 0.0);
    }
}
