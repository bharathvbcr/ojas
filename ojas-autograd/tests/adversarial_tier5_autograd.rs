//! Tier 5: Adversarial Coverage Hardening (Autograd Tape Chaining & Checkpointing)
//!
//! White-box coverage audit suite exercising:
//! 1. Multi-op vision & Gemma autograd chaining (conv2d -> max_pool2d -> relu -> gelu_tanh -> linear_bias)
//! 2. DAG fan-out and fan-in gradient accumulation & leaf unsharing
//! 3. Frozen leaf gradient isolation
//! 4. Activation checkpointing equivalence on vision pipelines
//! 5. Error boundaries and non-finite backward seed rejection

#![allow(clippy::identity_op, clippy::useless_vec, clippy::needless_range_loop)]

use ojas_autograd::{Tape, Var};
use ojas_core::{Budget, Conv2dParams, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

fn cpu_exact() -> CpuBackend {
    CpuBackend::with_threads(Budget::new(2 << 30), 2)
        .expect("CpuBackend::with_threads")
        .with_numerics(Numerics::Exact)
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().expect("to_f32_vec")
}

fn host_bits(t: &Tensor) -> Vec<u32> {
    host(t).into_iter().map(f32::to_bits).collect()
}

fn upload(be: &CpuBackend, data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, be.budget()).expect("from_f32")
}

#[test]
fn tier5_autograd_01_vision_and_gemma_pipeline_chaining() {
    let be = cpu_exact();
    let mut tape = Tape::new(be.clone());

    // 1. Leaf input x: [1, 2, 8, 8]
    let x_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.05 - 3.2).collect();
    let x_t = upload(&be, &x_data, &[1, 2, 8, 8]);
    let x_var = tape.leaf(x_t).expect("x leaf");

    // 2. Conv2d: 2 in-channels, 4 out-channels, 3x3 kernel, pad 1 -> out [1, 4, 8, 8]
    let w_conv_t = upload(&be, &vec![0.2f32; 4 * 2 * 3 * 3], &[4, 2, 3, 3]);
    let b_conv_t = upload(&be, &[0.1f32; 4], &[4]);
    let w_conv_var = tape.leaf(w_conv_t).expect("w_conv leaf");
    let b_conv_var = tape.leaf(b_conv_t).expect("b_conv leaf");

    let params = Conv2dParams {
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
        groups: 1,
    };
    let conv_var = tape
        .conv2d(x_var, w_conv_var, Some(b_conv_var), params)
        .expect("conv2d");

    // 3. MaxPool2d: 2x2 kernel, stride 2 -> out [1, 4, 4, 4]
    let pool_var = tape
        .max_pool2d(conv_var, [2, 2], [2, 2])
        .expect("max_pool2d");

    // 4. Relu
    let relu_var = tape.relu(pool_var).expect("relu");

    // 5. GeluTanh
    let gelu_var = tape.gelu_tanh(relu_var).expect("gelu_tanh");

    // 6. Reshape to flat [1, 64]
    let flat_var = tape.reshape(gelu_var, &[1, 64]).expect("reshape");

    // 7. LinearBias to scalar [1, 1]
    let w_lin_t = upload(&be, &vec![0.1f32; 1 * 64], &[1, 64]);
    let b_lin_t = upload(&be, &[0.0f32], &[1]);
    let w_lin_var = tape.leaf(w_lin_t).expect("w_lin leaf");
    let b_lin_var = tape.leaf(b_lin_t).expect("b_lin leaf");
    let out_var = tape
        .linear_bias(flat_var, w_lin_var, b_lin_var)
        .expect("linear_bias");

    // Backward pass
    tape.backward(out_var).expect("backward");

    // Verify all leaf gradients exist, are finite, and have non-zero elements
    let gx = tape.grad(x_var).expect("gx exists");
    assert_eq!(gx.shape(), &[1, 2, 8, 8]);
    assert!(host(gx).iter().any(|&v| v != 0.0));
    assert!(host(gx).iter().all(|v| v.is_finite()));

    let gw_conv = tape.grad(w_conv_var).expect("gw_conv exists");
    assert_eq!(gw_conv.shape(), &[4, 2, 3, 3]);
    assert!(host(gw_conv).iter().all(|v| v.is_finite()));

    let gb_conv = tape.grad(b_conv_var).expect("gb_conv exists");
    assert_eq!(gb_conv.shape(), &[4]);
    assert!(host(gb_conv).iter().all(|v| v.is_finite()));

    let gw_lin = tape.grad(w_lin_var).expect("gw_lin exists");
    assert_eq!(gw_lin.shape(), &[1, 64]);
    assert!(host(gw_lin).iter().all(|v| v.is_finite()));
}

#[test]
fn tier5_autograd_02_fan_out_fan_in_residual_gradient_unsharing() {
    let be = cpu_exact();
    let mut tape = Tape::new(be.clone());

    // Single leaf x: [1, 2, 4, 4]
    let x_t = upload(&be, &vec![1.5f32; 32], &[1, 2, 4, 4]);
    let x_var = tape.leaf(x_t).expect("x leaf");

    // Branch 1: Bilinear upsample to [1, 2, 8, 8] then downsample back to [1, 2, 4, 4]
    let up_var = tape.interpolate_bilinear(x_var, [8, 8], false).expect("up");
    let down_var = tape
        .interpolate_bilinear(up_var, [4, 4], false)
        .expect("down");

    // Branch 2: Relu
    let relu_var = tape.relu(x_var).expect("relu");

    // Recombine via Add: both branches flow into add
    let add_var = tape.add(down_var, relu_var).expect("add");

    tape.backward_seeded(add_var, 1.0).expect("backward");

    // Leaf x must have accumulated gradients from BOTH branches without allocation corruption
    let gx = tape.grad(x_var).expect("gx exists");
    assert_eq!(gx.shape(), &[1, 2, 4, 4]);
    let vals = host(gx);
    assert!(vals.iter().all(|v| v.is_finite()));
    // Branch 1 is identity-like (up + down) with scale ~1.0, Branch 2 on positive x=1.5 has relu' = 1.0.
    // Total gradient for each pixel should be approximately 2.0!
    for &v in &vals {
        assert!(
            (v - 2.0).abs() < 0.1,
            "accumulated gradient should be ~2.0, got {v}"
        );
    }
}

#[test]
fn tier5_autograd_03_frozen_leaf_gradient_isolation() {
    let be = cpu_exact();
    let mut tape = Tape::new(be.clone());

    let x_t = upload(&be, &[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let w_t = upload(&be, &[0.5; 8], &[2, 4]);
    let b_t = upload(&be, &[0.1, 0.2], &[2]);

    let x_var = tape.leaf(x_t).expect("x");
    let w_var = tape.frozen_leaf(w_t).expect("w frozen");
    let b_var = tape.leaf(b_t).expect("b");

    let lin_var = tape.linear_bias(x_var, w_var, b_var).expect("linear_bias");
    let gelu_var = tape.gelu_tanh(lin_var).expect("gelu_tanh");

    tape.backward_seeded(gelu_var, 1.0).expect("backward");

    // Trainable leaves x and b must have gradients
    assert!(tape.grad(x_var).is_some());
    assert!(tape.grad(b_var).is_some());
    // Frozen leaf w must NOT have gradient
    assert!(
        tape.grad(w_var).is_none(),
        "frozen leaf must have no gradient"
    );
}

#[test]
fn tier5_autograd_04_checkpointed_vision_segment_equivalence() {
    let be = cpu_exact();

    // Run 1: Checkpointed tape
    let mut tape1 = Tape::new(be.clone());
    let x_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.0).collect();
    let x_t1 = upload(&be, &x_data, &[1, 1, 8, 8]);
    let x1 = tape1.leaf(x_t1).expect("x1");

    let outs1 = tape1
        .checkpoint(|t| {
            let pool = t.max_pool2d(x1, [2, 2], [2, 2])?;
            let gelu = t.gelu_tanh(pool)?;
            let relu = t.relu(gelu)?;
            Ok(vec![relu])
        })
        .expect("checkpoint");
    tape1.backward_seeded(outs1[0], 1.0).expect("backward1");
    let gx1 = host_bits(tape1.grad(x1).expect("gx1"));

    // Run 2: Uncheckpointed tape
    let mut tape2 = Tape::new(be.clone());
    let x_t2 = upload(&be, &x_data, &[1, 1, 8, 8]);
    let x2 = tape2.leaf(x_t2).expect("x2");
    let pool2 = tape2.max_pool2d(x2, [2, 2], [2, 2]).expect("pool2");
    let gelu2 = tape2.gelu_tanh(pool2).expect("gelu2");
    let relu2 = tape2.relu(gelu2).expect("relu2");
    tape2.backward_seeded(relu2, 1.0).expect("backward2");
    let gx2 = host_bits(tape2.grad(x2).expect("gx2"));

    // Both must match bit for bit!
    assert_eq!(
        gx1, gx2,
        "activation checkpointing on vision ops must produce bit-identical gradients"
    );
}

#[test]
fn tier5_autograd_05_tape_error_and_nonfinite_seed_boundaries() {
    let be = cpu_exact();
    let mut tape = Tape::new(be.clone());
    let x_t = upload(&be, &[1.0; 4], &[2, 2]);
    let x_var = tape.leaf(x_t).expect("x");

    // 1. Non-finite backward seeds
    assert!(matches!(
        tape.backward_seeded(x_var, f32::NAN),
        Err(OjasError::NonFinite { .. })
    ));
    assert!(matches!(
        tape.backward_seeded(x_var, f32::INFINITY),
        Err(OjasError::NonFinite { .. })
    ));
    assert!(matches!(
        tape.backward_seeded(x_var, f32::NEG_INFINITY),
        Err(OjasError::NonFinite { .. })
    ));

    // 2. Variable not on tape
    assert!(matches!(
        tape.backward(Var(9999)),
        Err(OjasError::OutOfRange { .. })
    ));

    // 3. Empty checkpoint body
    assert!(matches!(
        tape.checkpoint(|_| Ok(vec![])),
        Err(OjasError::Shape { .. })
    ));

    // 4. Nested checkpoint rejected
    let res = tape.checkpoint(|t| t.checkpoint(|_| Ok(vec![x_var])));
    assert!(matches!(res, Err(OjasError::Unsupported { .. })));
}
