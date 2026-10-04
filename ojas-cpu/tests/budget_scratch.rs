//! Peak scratch must be charged before an op allocates it.
//!
//! A budget that can hold the output tensor and cannot hold the packed weight
//! used by the linear inner loop has to refuse.

use ojas_core::{
    AdamWConfig, Backend, Budget, MuonNs5Config, Numerics, OjasError, OptimizerKind, Tensor,
};
use ojas_cpu::CpuBackend;

fn values(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) ^ seed) as f32 / u32::MAX as f32 - 0.5)
        .collect()
}

/// The figure `optimizer_scratch_bytes` reports is exactly what each step
/// charges above its operands (the budget's peak), for every thread count,
/// numerics contract and orientation; and a budget one byte short of it is
/// refused before either target changes, while one with exactly that room
/// runs.
#[test]
fn reported_optimizer_scratch_is_the_measured_peak_and_the_exact_room() {
    let shapes = [(1, 1), (1, 7), (7, 1), (3, 5), (64, 17), (17, 64), (96, 96)];
    for threads in [1, 4] {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            for &(rows, cols) in &shapes {
                let n = rows * cols;
                let what = format!("[{rows}, {cols}] threads {threads} {numerics:?}");
                let probe = CpuBackend::with_threads(Budget::new(1 << 30), threads)
                    .unwrap()
                    .with_numerics(numerics);
                let muon = probe
                    .optimizer_scratch_bytes(OptimizerKind::MuonNs5, rows, cols)
                    .unwrap()
                    .expect("the CPU reports Muon scratch");
                let adam = probe
                    .optimizer_scratch_bytes(OptimizerKind::AdamW, rows, cols)
                    .unwrap();
                assert_eq!(adam, Some(0), "{what}: AdamW is in place");
                assert!(muon > 0, "{what}");
                // The parameter, its gradient and the momentum: 3n f32.
                let operands = (3 * n * 4) as u64;
                for (extra, ok) in [(muon - 1, false), (muon, true)] {
                    let budget = Budget::new(operands + extra);
                    let cpu = CpuBackend::with_threads(budget.clone(), threads)
                        .unwrap()
                        .with_numerics(numerics);
                    let mut p = Tensor::from_f32(&values(n, 1), &[rows, cols], &budget).unwrap();
                    let g = Tensor::from_f32(&values(n, 2), &[rows, cols], &budget).unwrap();
                    let mut m = Tensor::from_f32(&values(n, 3), &[rows, cols], &budget).unwrap();
                    let before = (p.to_f32_vec().unwrap(), m.to_f32_vec().unwrap());
                    let live = budget.live_bytes().unwrap();
                    budget.reset_peak();
                    let result =
                        cpu.muon_ns5_step(&mut p, &g, &mut m, MuonNs5Config::nanolab_default());
                    if ok {
                        result.unwrap_or_else(|e| panic!("{what}: exact room refused: {e}"));
                        assert_eq!(budget.peak_bytes() - live, muon, "{what}: peak");
                    } else {
                        let err = result.expect_err("one byte short");
                        assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{what}");
                        let after = (p.to_f32_vec().unwrap(), m.to_f32_vec().unwrap());
                        assert_eq!(before, after, "{what}: a refusal wrote");
                    }
                    assert_eq!(budget.live_bytes().unwrap(), live, "{what}: leaked");
                }
                let budget = Budget::new(1 << 30);
                let cpu = CpuBackend::with_threads(budget.clone(), threads)
                    .unwrap()
                    .with_numerics(numerics);
                let mut p = Tensor::from_f32(&values(n, 1), &[rows, cols], &budget).unwrap();
                let g = Tensor::from_f32(&values(n, 2), &[rows, cols], &budget).unwrap();
                let mut m1 = Tensor::zeros(&[rows, cols], ojas_core::DType::F32, &budget).unwrap();
                let mut m2 = Tensor::zeros(&[rows, cols], ojas_core::DType::F32, &budget).unwrap();
                let live = budget.live_bytes().unwrap();
                budget.reset_peak();
                cpu.adamw_step(
                    &mut p,
                    &g,
                    &mut m1,
                    &mut m2,
                    0,
                    AdamWConfig::nanolab(1e-3, 0.0),
                )
                .unwrap();
                assert_eq!(budget.peak_bytes(), live, "{what}: AdamW charged scratch");
            }
        }
    }
}

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
