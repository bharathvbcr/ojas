//! Peak scratch must be charged before an op allocates it.
//!
//! A budget that can hold the output tensor and cannot hold the packed weight
//! used by the linear inner loop has to refuse.

use ojas_core::{Backend, Budget, OjasError, Tensor};
use ojas_cpu::CpuBackend;

#[test]
fn linear_forward_refuses_budget_that_fits_output_but_not_packed_weight() {
    let budget = Budget::new(1_300);
    let cpu = CpuBackend::new(budget.clone());
    let x = Tensor::from_f32(&[1.0; 64], &[1, 64], &budget).expect("input fits");
    let weight = Tensor::from_f32(&[0.5; 256], &[4, 64], &budget).expect("weight fits");
    let err = cpu
        .linear_forward(&x, &weight)
        .expect_err("packed weight exceeds the remaining budget");
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err}");
}
