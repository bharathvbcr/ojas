//! `Tape` on a device backend. `Resident` reports `BackendId::Wgpu`, keeps
//! every tensor in its own `DeviceBuffer`, and refuses any input that is not
//! one of those with `Placement`. Its kernels are the CPU ones, so its
//! gradients must equal a CPU tape's bit for bit.
//!
//! Readbacks are counted on the tape backend's own budget
//! (`Budget::device_readbacks`), so tests running in parallel cannot disturb
//! them. Both the double and the reference run `Numerics::Exact` (the CPU
//! default is `Fast`); the comparison is bit for bit.

mod common;

use common::{bits, on, record, serial, Resident};
use ojas_autograd::Tape;
use ojas_core::{Backend, BackendId, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

fn reference(doubled: bool) -> (Vec<u32>, [Vec<u32>; 3]) {
    let mut tape = Tape::new(CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact));
    let g = record(&mut tape, doubled).unwrap();
    tape.backward(g.root).unwrap();
    let grads = g.params.map(|v| bits(tape.grad(v).unwrap()));
    (bits(tape.value(g.loss).unwrap()), grads)
}

fn run_on_device(doubled: bool) {
    let _serial = serial();
    let (want_loss, want_grads) = reference(doubled);
    let mut tape = Tape::new(Resident::new());
    let before = tape.backend().budget().device_readbacks();
    let g = record(&mut tape, doubled).expect("forward on a device backend");
    tape.backward(g.root).expect("backward on a device backend");
    assert_eq!(
        tape.backend().budget().device_readbacks(),
        before,
        "forward or backward read a tensor back to the host"
    );
    for (name, var, want) in [
        ("table", g.params[0], &want_grads[0]),
        ("w1", g.params[1], &want_grads[1]),
        ("w2", g.params[2], &want_grads[2]),
    ] {
        let grad = tape.grad(var).expect("leaf gradient");
        assert_eq!(
            grad.device(),
            Some(BackendId::Wgpu),
            "{name} grad left the device"
        );
        assert_eq!(
            &bits(&tape.backend().download(grad).unwrap()),
            want,
            "{name} grad"
        );
    }
    let mid = tape.backend().budget().device_readbacks();
    let loss = tape
        .backend()
        .download(tape.value(g.loss).unwrap())
        .unwrap();
    assert_eq!(
        tape.backend().budget().device_readbacks(),
        (mid.0 + 1, mid.1 + 4),
        "the loss is one 4-byte read"
    );
    assert_eq!(bits(&loss), want_loss);
}

#[test]
fn device_tape_backward_reads_nothing_back() {
    run_on_device(false);
}

#[test]
fn device_tape_scales_a_non_root_cross_entropy_on_the_device() {
    run_on_device(true);
}

#[test]
fn device_tape_refuses_another_devices_tensor() {
    let _serial = serial();
    let mut tape = Tape::new(Resident::new());
    let budget = Budget::new(1 << 10);
    let host = Tensor::from_f32(&[1.0, 2.0], &[2], &budget).unwrap();
    let metal = on(BackendId::Metal, &host, &budget).unwrap();
    assert!(matches!(
        tape.leaf(metal),
        Err(OjasError::Placement {
            expected: Some(BackendId::Wgpu),
            found: Some(BackendId::Metal),
            ..
        })
    ));
}
