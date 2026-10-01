//! Tape over a parallel CPU backend matches the one-thread backend bit for bit.

use ojas_autograd::Tape;
use ojas_core::{Budget, Tensor};
use ojas_cpu::CpuBackend;

#[test]
fn parallel_cpu_tape_matches_serial_linear_gradient() {
    let serial = CpuBackend::new(Budget::new(8 << 20));
    let parallel = CpuBackend::with_threads(Budget::new(8 << 20), 4).unwrap();
    let rows = 64usize;
    let kin = 8usize;
    let nout = 8usize;
    let x: Vec<f32> = (0..rows * kin)
        .map(|i| ((i % 17) as f32) * 0.05 - 0.4)
        .collect();
    let w: Vec<f32> = (0..nout * kin)
        .map(|i| ((i % 13) as f32) * 0.07 - 0.3)
        .collect();
    let gs = grad(&serial, &x, &w, rows, kin, nout);
    let gp = grad(&parallel, &x, &w, rows, kin, nout);
    assert_eq!(bits(&gs.0), bits(&gp.0));
    assert_eq!(bits(&gs.1), bits(&gp.1));
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().copied().map(f32::to_bits).collect()
}

fn grad(
    cpu: &CpuBackend,
    x: &[f32],
    w: &[f32],
    rows: usize,
    kin: usize,
    nout: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut tape = Tape::new(cpu.clone());
    let xv = tape
        .leaf(Tensor::from_f32(x, &[rows, kin], cpu.budget()).unwrap())
        .unwrap();
    let wv = tape
        .leaf(Tensor::from_f32(w, &[nout, kin], cpu.budget()).unwrap())
        .unwrap();
    let y = tape.linear(xv, wv).unwrap();
    tape.backward(y).unwrap();
    (
        tape.grad(xv).unwrap().to_f32_vec().unwrap(),
        tape.grad(wv).unwrap().to_f32_vec().unwrap(),
    )
}
