//! G6 (`Tape::backward_seeded`) on a real Metal device. A seed of 0.5 gives
//! exactly half the seed-1 gradients, bit for bit: scaling by a power of two
//! commutes with every rounding in the backward chain. `seeded.rs` runs the
//! same gate (`common::gate_g6`) on the CPU and the `Resident` doubles,
//! `wgpu_tape.rs` on a wgpu adapter. Readbacks are counted on the Metal
//! backend's own budget (`Budget::device_readbacks`), so parallel tests
//! cannot disturb them. A missing device fails unless `OJAS_ALLOW_NO_GPU=1`.

#![cfg(target_os = "macos")]

mod common;

use common::{bits, gate_g6, grads, record, serial};
use ojas_autograd::Tape;
use ojas_core::{Backend, BackendId, Budget, Numerics};
use ojas_cpu::CpuBackend;
use ojas_metal::MetalBackend;

const ALLOW_NO_GPU: &str = "OJAS_ALLOW_NO_GPU";
/// Metal against the CPU `Exact` tape: relative to the gradient's own peak.
const DEVICE_TOL: f32 = 2e-4;

/// The Metal backend, or `None` when `OJAS_ALLOW_NO_GPU=1` and none opens.
fn open() -> Option<MetalBackend> {
    match MetalBackend::new(Budget::new(1 << 28)) {
        Ok(gpu) => Some(gpu),
        Err(err) if std::env::var(ALLOW_NO_GPU).as_deref() == Ok("1") => {
            eprintln!("SKIP ({ALLOW_NO_GPU}=1): MetalBackend::new: {err}");
            None
        }
        Err(err) => panic!(
            "MetalBackend::new failed: {err}. This test needs a Metal 4 device; set \
             {ALLOW_NO_GPU}=1 to skip it explicitly on a machine without one"
        ),
    }
}

/// G6 on a real device, after checking that Metal repeats its own bits.
#[test]
fn metal_power_of_two_seeds_scale_gradients_bitwise() {
    let _serial = serial();
    let Some(gpu) = open() else {
        return;
    };
    for doubled in [false, true] {
        assert_eq!(
            grads(&gpu, doubled, 1.0),
            grads(&gpu, doubled, 1.0),
            "doubled {doubled}: Metal does not repeat its own bits"
        );
    }
    gate_g6(|| &gpu, "metal");
}

/// The seeded backward on Metal reads nothing back and leaves the leaf
/// gradients on the device, uniquely owned, and the seed means the same
/// thing as on the CPU `Exact` tape: Metal at seed 0.5 is within the device
/// tolerance of each gradient's own peak.
#[test]
fn metal_seeded_backward_stays_on_the_device_and_agrees_with_the_cpu() {
    let _serial = serial();
    let Some(gpu) = open() else {
        return;
    };
    for doubled in [false, true] {
        let mut cpu_tape =
            Tape::new(CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact));
        let cg = record(&mut cpu_tape, doubled).unwrap();
        cpu_tape.backward_seeded(cg.root, 0.5).unwrap();

        let mut tape = Tape::new(&gpu);
        let g = record(&mut tape, doubled).expect("forward on Metal");
        let before = tape.backend().budget().device_readbacks();
        tape.backward_seeded(g.root, 0.5)
            .expect("backward on Metal");
        assert_eq!(
            tape.backend().budget().device_readbacks(),
            before,
            "seeded backward read a tensor back"
        );
        for (i, (var, cvar)) in g.params.iter().zip(cg.params).enumerate() {
            let mut grad = tape.take_grad(*var).expect("leaf gradient");
            assert_eq!(grad.device(), Some(BackendId::Metal));
            if let Err(e) = grad.device_buffer_mut() {
                panic!("the taken Metal gradient is not uniquely owned: {e}");
            }
            let got = bits(&tape.backend().download(&grad).unwrap());
            let want = bits(cpu_tape.grad(cvar).unwrap());
            let peak = want
                .iter()
                .map(|b| f32::from_bits(*b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                peak > 0.0,
                "doubled {doubled} param {i}: the CPU gradient is all zero"
            );
            for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                let (g, w) = (f32::from_bits(*g), f32::from_bits(*w));
                assert!(
                    (g - w).abs() <= DEVICE_TOL * peak,
                    "doubled {doubled} param {i}[{j}]: metal {g} cpu {w} peak {peak}"
                );
            }
        }
    }
}
