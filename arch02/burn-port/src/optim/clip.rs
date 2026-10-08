//! Global gradient-norm clipping (python: clip_grad_norm_(params, 0.3)).
//!
//! Burn's built-in clipping is per-tensor; the reference clips the GLOBAL L2
//! norm across all parameters before any optimizer step.
//!
//! Mac-optimization: the first draft read every parameter's squared-sum back to
//! the host (`.into_scalar()` ~200×/step) — each readback forces a full GPU
//! pipeline flush, serializing the queue. This version keeps every squared-sum
//! as a device tensor, reduces them on-device, and reads back exactly ONE
//! scalar (the global norm) per step. Per-group norms (for logging) are read
//! back only when explicitly requested (log cadence).

use burn::module::ParamId;
use burn::optim::GradientsParams;
use burn::prelude::*;

/// A gradient group: ids partitioned by tensor rank.
pub struct GradGroup {
    pub ids_1d: Vec<ParamId>,
    pub ids_2d: Vec<ParamId>,
}

/// On-device sum of squared gradient elements in one group → scalar tensor [1].
///
/// The per-tensor squared-sums are concatenated and reduced with ONE final sum
/// instead of a chain of dependent scalar adds — a ~200-deep dependency chain
/// serializes the GPU queue, which showed up as ~1.5 s/step in phase profiles.
pub fn grad_sq_norm<B: Backend>(grads: &GradientsParams, group: &GradGroup) -> Tensor<B, 1> {
    let mut parts: Vec<Tensor<B, 1>> = Vec::with_capacity(group.ids_1d.len() + group.ids_2d.len());
    for id in &group.ids_1d {
        if let Some(g) = grads.get::<B, 1>(*id) {
            parts.push((g.clone() * g).sum());
        }
    }
    for id in &group.ids_2d {
        if let Some(g) = grads.get::<B, 2>(*id) {
            parts.push((g.clone() * g).sum());
        }
    }
    match parts.len() {
        0 => Tensor::<B, 1>::zeros([1], &grads_device::<B>(grads, group)),
        1 => parts.pop().unwrap(),
        _ => Tensor::cat(parts, 0).sum(),
    }
}

/// Device of the first available grad in the group (for the empty-accumulator
/// fallback).
fn grads_device<B: Backend>(grads: &GradientsParams, group: &GradGroup) -> B::Device {
    for id in &group.ids_2d {
        if let Some(g) = grads.get::<B, 2>(*id) {
            return g.device();
        }
    }
    for id in &group.ids_1d {
        if let Some(g) = grads.get::<B, 1>(*id) {
            return g.device();
        }
    }
    Default::default()
}

/// Scale every gradient in a group by a device scalar factor (async, no readback).
pub fn scale_grads<B: Backend>(grads: &mut GradientsParams, group: &GradGroup, factor: Tensor<B, 1>) {
    for id in &group.ids_1d {
        if let Some(g) = grads.remove::<B, 1>(*id) {
            let f = factor.clone().reshape([1]);
            grads.register::<B, 1>(*id, g * f);
        }
    }
    for id in &group.ids_2d {
        if let Some(g) = grads.remove::<B, 2>(*id) {
            let f = factor.clone().reshape([1, 1]);
            grads.register::<B, 2>(*id, g * f);
        }
    }
}

/// Result of a clip step.
pub struct ClipReport {
    /// Pre-clip global L2 norm (single readback).
    pub global_norm: f64,
    /// Applied factor = min(1, max_norm / global_norm).
    pub factor: f64,
    /// Per-group L2 norms, aligned to the input `groups` order — only present
    /// when `want_group_norms` was set (extra readbacks).
    pub group_norms: Option<Vec<f64>>,
}

/// Clip the global L2 norm across all groups to `max_norm`.
///
/// One scalar readback (the global norm) per step. When over the threshold,
/// grads are scaled by a device-resident factor; when under, nothing is
/// touched. Per-group norms are read back only if `want_group_norms`.
pub fn clip_global_norm<B: Backend>(
    groups: &mut [(&mut GradientsParams, &GradGroup)],
    max_norm: f64,
    want_group_norms: bool,
) -> ClipReport {
    // Per-group squared norms as device scalars.
    let group_sq: Vec<Tensor<B, 1>> = groups
        .iter()
        .map(|(g, grp)| grad_sq_norm::<B>(g, grp))
        .collect();

    // Optional per-group norms for logging (extra readbacks).
    let group_norms = if want_group_norms {
        Some(
            group_sq
                .iter()
                .map(|s| s.clone().sqrt().into_scalar().elem::<f64>())
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    // Global norm: sum group squared norms on-device, one readback.
    let mut total = group_sq[0].clone();
    for s in &group_sq[1..] {
        total = total + s.clone();
    }
    let device = total.device();
    let global_norm = total.sqrt().into_scalar().elem::<f64>();

    let mut factor = 1.0;
    if global_norm > max_norm && global_norm > 0.0 {
        factor = max_norm / global_norm;
        let factor_t = Tensor::<B, 1>::from_floats([factor as f32], &device);
        for (grads, group) in groups.iter_mut() {
            scale_grads::<B>(grads, group, factor_t.clone());
        }
    }

    ClipReport {
        global_norm,
        factor,
        group_norms,
    }
}
