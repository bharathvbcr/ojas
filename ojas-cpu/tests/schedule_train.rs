//! Cosine multiplier, accumulation, and the Muon/AdamW split.
//! These assertions do not compile against a crate that only has bare AdamW and Muon steps.

use ojas_core::{AdamWConfig, Backend, Budget, MuonNs5Config, OjasError, Tensor};
use ojas_cpu::{
    clip_grads, mean_micrograds, optim_group, scaled_lr, CosineSchedule, CpuBackend,
    GradAccumulator, HybridOptimizer, HybridParam, OptimGroup, ADAM_HYBRID_WEIGHT_DECAY,
    MUON_WEIGHT_DECAY,
};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().copied().map(f32::to_bits).collect()
}

fn assert_range(result: Result<(), OjasError>) {
    match result {
        Err(OjasError::OutOfRange { .. }) => {}
        other => panic!("expected OutOfRange, got {other:?}"),
    }
}

fn assert_nonfinite(result: Result<(), OjasError>) {
    match result {
        Err(OjasError::NonFinite { .. }) => {}
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

#[test]
fn cosine_multiplier_matches_the_hand_derived_shape() {
    // warmup 4, total 10, floor 0.1. Warmup values are exact dyadic ratios.
    // step 7: t = 3/6 = 1/2, cos(pi/2) = 0, so
    // 0.1 + 0.9 * 0.5 * (1 + 0) = 0.55. Hand-derived, not a second copy of the
    // function under test. f64 cos(pi/2) is a rounding error around 0.
    // step 10: t = 1, cos(pi) = -1, so the multiplier is the floor 0.1.
    let sched = CosineSchedule::new(4, 10).unwrap();
    assert_eq!(sched.multiplier(0).unwrap(), 0.25);
    assert_eq!(sched.multiplier(1).unwrap(), 0.5);
    assert_eq!(sched.multiplier(2).unwrap(), 0.75);
    assert_eq!(sched.multiplier(3).unwrap(), 1.0);
    assert_eq!(sched.multiplier(4).unwrap(), 1.0);
    let mid = sched.multiplier(7).unwrap();
    assert!(
        (mid - 0.55).abs() < 1e-12,
        "hand-derived step 7 is 0.55, got {mid}"
    );
    let floor = sched.multiplier(10).unwrap();
    assert!(
        (floor - 0.1).abs() < 1e-12,
        "hand-derived step 10 is the floor 0.1, got {floor}"
    );
    let past = sched.multiplier(100).unwrap();
    assert!((past - 0.1).abs() < 1e-12);

    let peak_ratio = 0.025 / 6e-4;
    for step in 0..12 {
        let mult = sched.multiplier(step).unwrap();
        let matrix = scaled_lr(0.025, mult).unwrap();
        let adam = scaled_lr(6e-4, mult).unwrap();
        assert!((matrix / adam - peak_ratio).abs() < 1e-9);
    }
}

#[test]
fn warmup_zero_and_step_overflow_are_errors() {
    assert_range(CosineSchedule::new(0, 20).map(|_| ()));
    assert_range(CosineSchedule::new(4, 0).map(|_| ()));
    let sched = CosineSchedule::new(4, 20).unwrap();
    assert_range(sched.multiplier(u64::MAX).map(|_| ()));
    assert_range(sched.multiplier(1 << 53).map(|_| ()));
    assert_nonfinite(scaled_lr(f64::NAN, 1.0).map(|_| ()));
}

#[test]
fn accumulation_sums_then_divides_once_and_refuses_zero() {
    assert_range(mean_micrograds(&[]).map(|_| ()));
    assert_range(GradAccumulator::new(4).unwrap().mean().map(|_| ()));

    // Exact dyadics: (1+5)/2 = 3, (3+7)/2 = 5.
    let mean = mean_micrograds(&[&[1.0, 3.0], &[5.0, 7.0]]).unwrap();
    assert_eq!(mean, vec![3.0, 5.0]);

    // Division by 2 is exact for many normal f32 values, so the two orders
    // often match. This pair does not: sum-then-divide is bit 8317574 and
    // dividing each micro-batch first is bit 8317573. Those bits were read
    // off an f32 execution, not derived by hand.
    let a = 1.5384231e-38_f32;
    let b = 7.926575e-39_f32;
    let per_micro = a / 2.0 + b / 2.0;
    assert_eq!(per_micro.to_bits(), 8_317_573);
    let got = mean_micrograds(&[&[a], &[b]]).unwrap()[0];
    assert_eq!(got.to_bits(), 8_317_574);

    let mut acc = GradAccumulator::new(1).unwrap();
    acc.add(&[1.0]).unwrap();
    acc.add(&[3.0]).unwrap();
    assert_eq!(acc.count(), 2);
    assert_eq!(acc.mean().unwrap(), vec![2.0]);
    assert_nonfinite(acc.add(&[f32::NAN]).map(|_| ()));
    assert_eq!(acc.count(), 2);
    assert_eq!(acc.mean().unwrap(), vec![2.0]);
}

#[test]
fn groups_send_matrices_to_muon_and_vectors_to_adam() {
    assert_eq!(optim_group(2, true), OptimGroup::AdamEmbedding);
    assert_eq!(optim_group(2, false), OptimGroup::MuonMatrix);
    assert_eq!(optim_group(1, false), OptimGroup::AdamVector);
    assert_eq!(optim_group(1, true), OptimGroup::AdamEmbedding);
    assert_eq!(MUON_WEIGHT_DECAY, 0.1);
    assert_eq!(ADAM_HYBRID_WEIGHT_DECAY, 0.0);
}

#[test]
fn one_hybrid_step_matches_adamw_and_muon_and_nan_updates_neither() {
    let cpu = CpuBackend::new(Budget::new(1 << 20)).with_numerics(ojas_core::Numerics::Exact);
    let mut opt = HybridOptimizer::new(vec![
        HybridParam::new(
            OptimGroup::AdamEmbedding,
            6e-4,
            &[2, 2],
            vec![0.2, -0.1, 0.0, 0.4],
        )
        .unwrap(),
        HybridParam::new(
            OptimGroup::MuonMatrix,
            0.025,
            &[2, 2],
            vec![0.2, -0.1, 0.0, 0.4],
        )
        .unwrap(),
        HybridParam::new(OptimGroup::AdamVector, 6e-4, &[2], vec![1.0, -0.5]).unwrap(),
    ])
    .unwrap();
    opt.params[0].grad = vec![0.3, -0.2, 0.1, 0.05];
    opt.params[1].grad = vec![0.3, -0.2, 0.1, 0.05];
    opt.params[2].grad = vec![0.4, -0.25];
    let mult = CosineSchedule::new(4, 10).unwrap().multiplier(0).unwrap();
    assert_eq!(mult, 0.25);
    opt.step(mult).unwrap();
    assert_eq!(opt.step_index, 1);

    let mut emb = Tensor::from_f32(&[0.2, -0.1, 0.0, 0.4], &[2, 2], cpu.budget()).unwrap();
    let emb_g = Tensor::from_f32(&[0.3, -0.2, 0.1, 0.05], &[2, 2], cpu.budget()).unwrap();
    let mut m1 = Tensor::from_f32(&[0.0; 4], &[2, 2], cpu.budget()).unwrap();
    let mut m2 = Tensor::from_f32(&[0.0; 4], &[2, 2], cpu.budget()).unwrap();
    cpu.adamw_step(
        &mut emb,
        &emb_g,
        &mut m1,
        &mut m2,
        0,
        AdamWConfig::nanolab(scaled_lr(6e-4, mult).unwrap(), 0.0),
    )
    .unwrap();
    assert_eq!(bits(&opt.params[0].param), bits(&emb.to_f32_vec().unwrap()));

    let mut mat = Tensor::from_f32(&[0.2, -0.1, 0.0, 0.4], &[2, 2], cpu.budget()).unwrap();
    let mat_g = Tensor::from_f32(&[0.3, -0.2, 0.1, 0.05], &[2, 2], cpu.budget()).unwrap();
    let mut mom = Tensor::from_f32(&[0.0; 4], &[2, 2], cpu.budget()).unwrap();
    cpu.muon_ns5_step(
        &mut mat,
        &mat_g,
        &mut mom,
        MuonNs5Config {
            lr: scaled_lr(0.025, mult).unwrap(),
            momentum: 0.99,
            weight_decay: 0.1,
            nesterov: true,
            ns5: ojas_core::Ns5Precision::F32,
        },
    )
    .unwrap();
    assert_eq!(bits(&opt.params[1].param), bits(&mat.to_f32_vec().unwrap()));
    assert_ne!(
        bits(&opt.params[0].param),
        bits(&opt.params[1].param),
        "embedding must not take the Muon update"
    );

    let mut bad = HybridOptimizer::new(vec![
        HybridParam::new(OptimGroup::AdamVector, 6e-4, &[1], vec![1.25]).unwrap(),
        HybridParam::new(OptimGroup::MuonMatrix, 0.025, &[1, 1], vec![0.5]).unwrap(),
    ])
    .unwrap();
    bad.params[0].grad = vec![f32::NAN];
    bad.params[1].grad = vec![0.3];
    let adam_bits = bits(&bad.params[0].param);
    let muon_bits = bits(&bad.params[1].param);
    let m_bits = bits(&bad.params[1].moment1);
    assert_nonfinite(bad.step(1.0));
    assert_eq!(bad.step_index, 0);
    assert_eq!(bits(&bad.params[0].param), adam_bits);
    assert_eq!(bits(&bad.params[1].param), muon_bits);
    assert_eq!(bits(&bad.params[1].moment1), m_bits);

    bad.step_index = u64::MAX;
    bad.params[0].grad = vec![0.1];
    bad.params[1].grad = vec![0.1];
    assert_range(bad.step(1.0));
    assert_eq!(bad.step_index, u64::MAX);
    assert_eq!(bits(&bad.params[0].param), adam_bits);
    assert_eq!(bits(&bad.params[1].param), muon_bits);
}

#[test]
fn two_adam_steps_are_not_the_accumulated_step() {
    let mut once = HybridOptimizer::new(vec![HybridParam::new(
        OptimGroup::AdamVector,
        0.1,
        &[1],
        vec![1.0],
    )
    .unwrap()])
    .unwrap();
    once.params[0].grad = vec![1.0];
    once.step(1.0).unwrap();

    let mut twice = HybridOptimizer::new(vec![HybridParam::new(
        OptimGroup::AdamVector,
        0.1,
        &[1],
        vec![1.0],
    )
    .unwrap()])
    .unwrap();
    twice.params[0].grad = vec![1.0];
    twice.step(1.0).unwrap();
    twice.params[0].grad = vec![1.0];
    twice.step(1.0).unwrap();
    assert_eq!(once.step_index, 1);
    assert_eq!(twice.step_index, 2);
    assert_ne!(bits(&once.params[0].param), bits(&twice.params[0].param));

    let mut parts = vec![vec![3.0, 4.0]];
    let before = bits(&parts[0]);
    let norm = clip_grads(&mut parts, 1.0).unwrap();
    assert!((norm - 5.0).abs() < 1e-6);
    assert_ne!(bits(&parts[0]), before);
}
