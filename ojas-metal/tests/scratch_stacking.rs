//! `Backend::optimizer_scratch_bytes` is an upper bound on what one step
//! reserves, so a budget with room for the operands and one step's figure
//! carries any number of back-to-back `muon_ns5_step` calls: none is refused
//! after another has updated its parameter in place. Metal reserves scratch
//! per call and, on exhaustion, makes a waited commit and retries
//! (`Worker::retrying`, `ojas-metal/src/device.rs`); this is the same
//! three-step shape as wgpu's
//! `consecutive_muon_steps_stay_inside_one_steps_scratch`
//! (`ojas-wgpu/tests/accumulate_grad.rs`), where the recorder holds scratch
//! until the next submit.
#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::{Backend, Budget, DType, MuonNs5Config, OptimizerKind, Tensor};
use ojas_metal::MetalBackend;

#[test]
fn consecutive_muon_steps_stay_inside_one_steps_scratch() {
    let (rows, cols) = (2048usize, 6144usize);
    let shape = [rows, cols];
    let bytes = (rows * cols * 4) as u64;
    let probe = metal();
    let scratch = ok(
        "scratch bytes",
        probe.optimizer_scratch_bytes(OptimizerKind::MuonNs5, rows, cols),
    )
    .expect("Metal reports its optimizer scratch");
    drop(probe);
    // Three parameters, each with a gradient and a momentum.
    let cap = (9 * bytes + scratch) * 11 / 10;
    let m = ok("new", MetalBackend::new(Budget::new(cap)));
    let mut params = Vec::new();
    let mut moms = Vec::new();
    let mut grads = Vec::new();
    for i in 0..3u64 {
        params.push(up(&m, &rand(&shape, 10 + i, 1.0)));
        grads.push(up(&m, &rand(&shape, 20 + i, 1.0)));
        let zeros = ok(
            "zeros",
            Tensor::zeros(&shape, DType::F32, &Budget::new(2 * bytes)),
        );
        moms.push(up(&m, &zeros));
    }
    let cfg = MuonNs5Config::nanolab_default();
    for i in 0..3 {
        ok(
            &format!("step {i} of 3, no sync between"),
            m.muon_ns5_step(&mut params[i], &grads[i], &mut moms[i], cfg),
        );
    }
    ok("sync", m.sync());
}
