//! One-layer train loop: bit-identical reruns, and micro-batch reduction.

use ojas_autograd::{TinyTrain, TokenBatch};
use ojas_cpu::OptimGroup;

fn batch(ids: &[u32], targets: &[u32], rows: usize) -> TokenBatch {
    TokenBatch::try_new(ids.to_vec(), targets.to_vec(), rows, 4).unwrap()
}

fn full() -> TokenBatch {
    batch(&[1, 5, 9, 12, 3, 7, 2, 8], &[5, 9, 12, 4, 7, 2, 8, 1], 2)
}

fn halves() -> (TokenBatch, TokenBatch) {
    (
        batch(&[1, 5, 9, 12], &[5, 9, 12, 4], 1),
        batch(&[3, 7, 2, 8], &[7, 2, 8, 1], 1),
    )
}

fn run20() -> (Vec<u32>, Vec<(u32, Vec<u32>)>) {
    let mut train = TinyTrain::new(1).unwrap();
    assert!(
        train.param_count() < 20_000,
        "param count {}",
        train.param_count()
    );
    assert_eq!(train.group("tok_emb").unwrap(), OptimGroup::AdamEmbedding);
    assert_eq!(train.group("wq").unwrap(), OptimGroup::MuonMatrix);
    assert_eq!(train.group("gate_w").unwrap(), OptimGroup::MuonMatrix);
    assert_eq!(train.group("gate_b").unwrap(), OptimGroup::AdamVector);
    assert_eq!(train.group("norm1").unwrap(), OptimGroup::AdamVector);
    assert_eq!(train.group("vr_lambda").unwrap(), OptimGroup::AdamVector);
    let data = full();
    let mut losses = Vec::with_capacity(20);
    for _ in 0..20 {
        losses.push(train.step(std::slice::from_ref(&data)).unwrap());
    }
    assert_eq!(train.step_index(), 20);
    let params: Vec<(u32, Vec<u32>)> = train
        .names()
        .iter()
        .map(|name| {
            let values = train.param(name).unwrap();
            (
                values[0].to_bits(),
                values.iter().copied().map(f32::to_bits).collect(),
            )
        })
        .collect();
    let loss_bits = losses.iter().copied().map(f32::to_bits).collect();
    (loss_bits, params)
}

#[test]
fn twenty_steps_are_bit_identical_across_two_runs() {
    let (loss_a, params_a) = run20();
    let (loss_b, params_b) = run20();
    assert_eq!(loss_a, loss_b);
    assert_eq!(params_a, params_b);
    assert!(loss_a.iter().all(|bits| f32::from_bits(*bits).is_finite()));
    assert_ne!(loss_a[0], loss_a[19], "loss did not move across 20 steps");
    let lambda = params_a.iter().map(|(_, values)| values).nth(9).unwrap();
    assert_eq!(lambda, &vec![1.0f32.to_bits()]);
}

#[test]
fn two_halves_reduce_to_the_full_batch_gradient() {
    let full_model = TinyTrain::new(7).unwrap();
    let (loss_full, grad_full) = full_model.backward(&full()).unwrap();
    let (left, right) = halves();
    let (loss_l, grad_l) = full_model.backward(&left).unwrap();
    let (loss_r, grad_r) = full_model.backward(&right).unwrap();
    let mut loss_sum = 0.0f32;
    loss_sum += loss_l;
    loss_sum += loss_r;
    let loss_mean = loss_sum / 2.0;
    assert!(
        (loss_mean - loss_full).abs() < 1e-5,
        "{loss_mean} vs {loss_full}"
    );

    let mut worst = 0.0f32;
    for (name, ((full_g, left_g), right_g)) in full_model
        .names()
        .iter()
        .zip(grad_full.iter().zip(grad_l.iter()).zip(grad_r.iter()))
    {
        let reduced =
            ojas_autograd::reduce_micrograds(&[left_g.as_slice(), right_g.as_slice()]).unwrap();
        assert_eq!(reduced.len(), full_g.len(), "{name}");
        for (got, expect) in reduced.iter().zip(full_g.iter()) {
            worst = worst.max((got - expect).abs());
        }
    }
    assert!(
        worst < 1e-5,
        "half-batch mean differs from the full-batch grad by {worst}; reduction is sum then /K"
    );

    let mut pooled = TinyTrain::new(7).unwrap();
    let mut split = TinyTrain::new(7).unwrap();
    let loss_p = pooled.step(std::slice::from_ref(&full())).unwrap();
    let (left, right) = halves();
    let loss_s = split.step(&[left, right]).unwrap();
    assert_eq!(pooled.step_index(), 1);
    assert_eq!(split.step_index(), 1);
    assert!((loss_p - loss_s).abs() < 1e-5);
    let mut param_worst = 0.0f32;
    for name in pooled.names() {
        let a = pooled.param(name).unwrap();
        let b = split.param(name).unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            param_worst = param_worst.max((x - y).abs());
        }
    }
    assert!(
        param_worst < 1e-5,
        "one step on the full batch differs from one step on two halves by {param_worst}"
    );
}

#[test]
fn empty_accumulation_does_not_step() {
    let mut train = TinyTrain::new(1).unwrap();
    let err = train.step(&[]).unwrap_err();
    assert!(matches!(err, ojas_core::OjasError::OutOfRange { .. }));
    assert_eq!(train.step_index(), 0);
    assert_eq!(train.param("vr_lambda").unwrap(), &[1.0]);
}
