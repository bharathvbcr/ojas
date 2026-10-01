//! The host AdamW against `torch.optim.AdamW(foreach=False)` on CPU f32 tensors, six steps
//! with a schedule, weight decay 0.01 and `clip_grad_norm_` in front of it, as the PyTorch
//! trainer drives it (values dumped by `tools/qd_train_oracle_trainer.py`).
//!
//! The bound is tessl's AdamW bound (`docs/qwen35.md`: within 1e-6 over 3 steps), applied per
//! tensor as `|ours - torch| <= 1e-6 * max(|torch|)`; bit equality is not claimed (ATen's
//! vector and scalar-tail kernels may fuse differently) and the count of bit-identical
//! elements is printed rather than asserted.

mod common;

use qd_train::adamw::{adamw_update, Moments};
use qd_train::recipe;
use qd_train::step::AdamWHyper;

fn as_f32(v: &serde_json::Value) -> Vec<f32> {
    common::fhex_list(v).into_iter().map(|x| x as f32).collect()
}

fn close(name: &str, got: &[f32], want: &[f32]) -> usize {
    let scale = want.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(f32::MIN_POSITIVE);
    let mut same = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-6 * scale, "{name}[{i}]: {g:e} vs torch {w:e} (max |torch| {scale:e})");
        same += usize::from(g.to_bits() == w.to_bits());
    }
    same
}

#[test]
fn host_adamw_tracks_torch_adamw_step_by_step() {
    let o = common::oracle();
    let a = &o["adamw"];
    assert_eq!(common::fhex(&a["weight_decay"]), recipe::WEIGHT_DECAY);
    assert_eq!(common::fhex(&a["eps"]), recipe::EPS);
    let beta2 = common::fhex(&a["beta2"]);
    let mut params: Vec<Vec<f32>> = a["init"].as_array().unwrap().iter().map(as_f32).collect();
    let mut moments: Vec<Moments> = params.iter().map(|p| Moments::zeros(p.len())).collect();
    let (mut same, mut total) = (0, 0);
    for (k, step) in a["steps"].as_array().unwrap().iter().enumerate() {
        let lr = common::fhex(&step["lr"]);
        let hyper = AdamWHyper {
            lr,
            beta1: recipe::BETA1,
            beta2,
            eps: recipe::EPS,
            grad_scale: 1.0,
        };
        let grads: Vec<Vec<f32>> = step["clipped_grads"].as_array().unwrap().iter().map(as_f32).collect();
        for i in 0..params.len() {
            adamw_update(&mut params[i], &grads[i], &mut moments[i], &hyper, lr, recipe::WEIGHT_DECAY, k as u64 + 1)
                .unwrap();
            let want_p = as_f32(&step["params"][i]);
            let want_m = as_f32(&step["m"][i]);
            let want_v = as_f32(&step["v"][i]);
            same += close(&format!("step {k} param {i}"), &params[i], &want_p);
            same += close(&format!("step {k} m {i}"), &moments[i].m, &want_m);
            same += close(&format!("step {k} v {i}"), &moments[i].v, &want_v);
            total += 3 * want_p.len();
        }
    }
    println!("host AdamW vs torch: {same} of {total} values bit-identical");
}

#[test]
fn the_clip_coefficient_matches_torchs_norm() {
    let o = common::oracle();
    for (k, step) in o["adamw"]["steps"].as_array().unwrap().iter().enumerate() {
        let grads: Vec<Vec<f32>> = step["grads"].as_array().unwrap().iter().map(as_f32).collect();
        let sq: f64 = grads.iter().flatten().map(|&x| f64::from(x) * f64::from(x)).sum();
        let torch_norm = common::fhex(&step["torch_total_norm"]);
        assert!((sq.sqrt() - torch_norm).abs() <= 1e-6 * torch_norm, "step {k}: {} vs {torch_norm}", sq.sqrt());
        let ours = recipe::clip_coefficient(sq, 1.0).unwrap();
        let theirs = common::fhex(&step["coef"]);
        assert!((ours - theirs).abs() <= 1e-6 * theirs, "step {k}: coefficient {ours} vs {theirs}");
        // Applying it as f32, the way the update reads it, reproduces torch's clipped gradient.
        for (i, g) in grads.iter().enumerate() {
            let want = as_f32(&step["clipped_grads"][i]);
            let got: Vec<f32> = g.iter().map(|x| x * ours as f32).collect();
            close(&format!("step {k} clipped grad {i}"), &got, &want);
        }
    }
    let c = &o["clip"];
    let grads: Vec<f32> = c["grads"].as_array().unwrap().iter().flat_map(as_f32).collect();
    let sq: f64 = grads.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let torch_norm = common::fhex(&c["torch_total_norm"]);
    assert!((sq.sqrt() - torch_norm).abs() <= 1e-6 * torch_norm);
}
