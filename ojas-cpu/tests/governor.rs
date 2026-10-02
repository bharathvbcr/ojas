//! Resource governor: thread ceiling, held scratch, cancel, and real parallelism.
//!
//! Shapes above `TASK_MACS * 2` multiply-adds leave the calling thread. Smaller
//! linear tests stay serial even when several threads are configured.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ojas_core::{Backend, Budget, OjasError, CPU_THREAD_CEILING, RMS_NORM_EPS};
use ojas_cpu::{CpuBackend, GradAccumulator};

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn f32_fill(budget: &Budget, n: usize, shape: &[usize], value: f32) -> ojas_core::Tensor {
    ojas_core::Tensor::from_f32(&vec![value; n], shape, budget).unwrap()
}

#[test]
fn with_threads_refuses_zero_and_above_the_ceiling_without_clamping() {
    let budget = Budget::new(64);
    assert!(matches!(
        CpuBackend::with_threads(budget.clone(), 0),
        Err(OjasError::OutOfRange { .. })
    ));
    let ceiling = usize::try_from(CPU_THREAD_CEILING).unwrap();
    assert!(matches!(
        CpuBackend::with_threads(budget.clone(), ceiling + 1),
        Err(OjasError::OutOfRange {
            op: "CpuBackend::with_threads",
            ..
        })
    ));
    assert_eq!(
        CpuBackend::with_threads(budget, ceiling).unwrap().threads(),
        ceiling
    );
}

/// `u32::MAX` is far above the fixed ceiling. The call returns before any
/// worker is spawned; the pool is created only after the count is accepted.
#[test]
fn with_threads_refuses_u32_max_without_spawning() {
    let err = CpuBackend::with_threads(Budget::new(64), u32::MAX as usize).unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::OutOfRange {
                op: "CpuBackend::with_threads",
                ..
            }
        ),
        "{err}"
    );
    let text = err.to_string();
    assert!(
        text.contains("exceeds") && text.contains(&format!("{}", u32::MAX)),
        "{text}"
    );
}

#[test]
fn causal_sdpa_refuses_scratch_when_the_output_tensor_would_fit() {
    // One head, T=8, D=4. Query, key and value are 384 bytes. The output is
    // 128 bytes. Scores, probabilities and the packed keys are 192 bytes and
    // are live with that output.
    let elems = 8 * 4;
    let qkv = 3 * elems * 4;
    let output = elems * 4;
    let budget = Budget::new((qkv + output) as u64);
    let cpu = CpuBackend::new(budget.clone());
    let q = f32_fill(&budget, elems, &[1, 1, 8, 4], 0.1);
    let k = f32_fill(&budget, elems, &[1, 1, 8, 4], 0.2);
    let v = f32_fill(&budget, elems, &[1, 1, 8, 4], 0.3);
    let err = cpu.causal_sdpa_forward(&q, &k, &v).unwrap_err();
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err}");
    assert_eq!(budget.live_bytes().unwrap(), qkv as u64);
}

#[test]
fn cross_entropy_forward_holds_no_gradient_and_charges_only_the_loss() {
    // Logits 128 bytes, targets 16, scalar loss 4. The forward reads both in
    // place and forms no gradient, so the loss is its only charge: room for
    // it succeeds and one byte less refuses.
    for (room, fits) in [(4u64, true), (3, false)] {
        let budget = Budget::new(128 + 16 + room);
        let cpu = CpuBackend::new(budget.clone());
        let logits = ojas_core::Tensor::from_f32(&[0.0; 32], &[4, 8], &budget).unwrap();
        let targets = ojas_core::Tensor::from_u32(&[0, 1, 2, 3], &[4], &budget).unwrap();
        let got = cpu.cross_entropy_mean_forward(&logits, &targets, None);
        if fits {
            let loss = got.unwrap().to_f32_vec().unwrap()[0];
            assert!((loss - 8f32.ln()).abs() < 1e-6, "{loss}");
        } else {
            let err = got.unwrap_err();
            assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err}");
            assert_eq!(budget.live_bytes().unwrap(), 128 + 16);
        }
    }
}

#[test]
fn embedding_backward_refuses_the_table_before_it_fits_only_as_an_output() {
    // Table 24, ids 4, grad 8. The scatter writes another vocab*dim table
    // (24 bytes). Eight bytes of headroom cannot hold it.
    let budget = Budget::new(24 + 4 + 8 + 8);
    let cpu = CpuBackend::new(budget.clone());
    let table =
        ojas_core::Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &budget).unwrap();
    let ids = ojas_core::Tensor::from_u32(&[0], &[1], &budget).unwrap();
    let gy = ojas_core::Tensor::from_f32(&[1.0, 1.0], &[1, 2], &budget).unwrap();
    let err = cpu.embedding_backward(&table, &ids, &gy).unwrap_err();
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err}");
}

#[test]
fn grad_accumulator_refuses_a_width_the_allocator_will_not_reserve() {
    // One past the largest `f32` buffer `Layout` can describe. `try_reserve`
    // refuses it; the old `vec![0.0; width]` panicked with capacity overflow.
    let width = (isize::MAX as usize) / std::mem::size_of::<f32>() + 1;
    let err = GradAccumulator::new(width).unwrap_err();
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");
}

fn linear_bits(threads: usize, rows: usize, kin: usize, nout: usize) -> Vec<u32> {
    let cpu = CpuBackend::with_threads(Budget::new(64 << 20), threads).unwrap();
    let x = f32_fill(cpu.budget(), rows * kin, &[rows, kin], 0.25);
    let w = f32_fill(cpu.budget(), nout * kin, &[nout, kin], 0.5);
    let y = cpu.linear_forward(&x, &w).unwrap();
    bits(&y.to_f32_vec().unwrap())
}

#[test]
fn linear_above_two_task_macs_matches_one_thread_and_several() {
    // 32 * 256 * 256 = 2_097_152 = 2 * TASK_MACS. Below that the pool stays idle.
    let serial = linear_bits(1, 32, 256, 256);
    let parallel = linear_bits(4, 32, 256, 256);
    assert_eq!(serial, parallel);
    assert_eq!(serial.len(), 32 * 256);
}

#[test]
fn linear_one_row_above_two_task_macs_matches_one_thread_and_several() {
    // 1 * 2048 * 1024 = 2_097_152. The row axis is 1; column tiles still split.
    let serial = linear_bits(1, 1, 2048, 1024);
    let parallel = linear_bits(4, 1, 2048, 1024);
    assert_eq!(serial, parallel);
    assert_eq!(serial.len(), 1024);
}

#[test]
fn rms_rope_and_gate_refuse_before_their_working_buffers() {
    let budget = Budget::new(16 + 16);
    let cpu = CpuBackend::new(budget.clone());
    let x = f32_fill(&budget, 4, &[4], 0.5);
    let w = f32_fill(&budget, 4, &[4], 1.0);
    let err = cpu.rms_norm_forward(&x, &w, RMS_NORM_EPS).unwrap_err();
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err}");

    let budget = Budget::new(16);
    let cpu = CpuBackend::new(budget.clone());
    let wide = CpuBackend::new(Budget::new(1 << 20));
    let x = f32_fill(&budget, 4, &[4], 1.0);
    let cos = f32_fill(wide.budget(), 4, &[4], 1.0);
    let sin = f32_fill(wide.budget(), 4, &[4], 0.0);
    let err = cpu.rope_half_split_forward(&x, &cos, &sin).unwrap_err();
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err}");

    let budget = Budget::new(8 + 16 + 8 + 8);
    let cpu = CpuBackend::new(budget.clone());
    let x = f32_fill(&budget, 2, &[1, 2], 1.0);
    let w = f32_fill(&budget, 4, &[2, 2], 0.0);
    let b = f32_fill(&budget, 2, &[2], 0.0);
    let attn = f32_fill(&budget, 2, &[1, 2, 1], 1.0);
    let err = cpu
        .per_head_sigmoid_gate_forward(&x, &w, &b, &attn)
        .unwrap_err();
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err}");
}

#[test]
fn cancel_between_rows_matches_the_noop_and_can_stop_the_op() {
    let serial = linear_bits(1, 8, 32, 16);
    let cpu = CpuBackend::with_threads(Budget::new(8 << 20), 2).unwrap();
    let hits = Arc::new(AtomicUsize::new(0));
    cpu.set_cancel({
        let hits = Arc::clone(&hits);
        move || {
            hits.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    });
    let rows = 8usize;
    let kin = 32usize;
    let nout = 16usize;
    let x = f32_fill(cpu.budget(), rows * kin, &[rows, kin], 0.25);
    let w = f32_fill(cpu.budget(), nout * kin, &[nout, kin], 0.5);
    let y = cpu.linear_forward(&x, &w).unwrap();
    assert_eq!(bits(&y.to_f32_vec().unwrap()), serial);
    assert!(hits.load(Ordering::Relaxed) >= 1);

    cpu.set_cancel(|| {
        Err(OjasError::Unsupported {
            op: "cancel",
            detail: "stop".to_string(),
        })
    });
    let err = cpu.linear_forward(&x, &w).unwrap_err();
    assert!(
        matches!(err, OjasError::Unsupported { op: "cancel", .. }),
        "{err}"
    );
}

#[test]
fn row_split_above_the_row_grain_matches_one_thread_and_several() {
    // `Exec::rows` splits once `rows / (ROW_MIN_ELEMS / dim) >= 2`.
    // dim 64 => min chunk 512 rows; 1024 rows is two chunks. `map_rows` is gone.
    let rows = 1024usize;
    let dim = 64usize;
    let run = |threads: usize| {
        let cpu = CpuBackend::with_threads(Budget::new(8 << 20), threads).unwrap();
        let x = f32_fill(cpu.budget(), rows * dim, &[rows, dim], 0.5);
        let w = f32_fill(cpu.budget(), dim, &[dim], 1.25);
        let y = cpu.rms_norm_forward(&x, &w, 1e-6).unwrap();
        bits(&y.to_f32_vec().unwrap())
    };
    assert_eq!(run(1), run(4));
}
