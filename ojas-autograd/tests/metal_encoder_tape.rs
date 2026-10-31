//! The encoder ops on a tape over the Metal backend: `linear_bias`,
//! `gelu_erf`, `layer_norm`, `channel_layer_norm`, `broadcast_mul`,
//! `broadcast_add` and `softmax` record on the device and their backwards run
//! as Metal kernels (`kernels/encoder_bwd.metal`), with no change to the tape:
//! `Tape` is generic over `Backend` and calls each backward by name
//! (`ojas-autograd/src/tape.rs`, `Rec::LayerNorm` ... `Rec::Softmax`).
//!
//! - The gradients of every leaf agree with the CPU `Exact` tape's within
//!   `DEVICE_TOL` of each gradient's own peak (stated before measuring: the
//!   forward chain differs between the backends by a few `f32` ulp, and the
//!   backward amplifies it through a softmax and two norms).
//! - They stay on the device, and nothing is read back.
//! - The backward kernels were dispatched: it did not fall back to the host.
//!
//! A missing device fails unless `OJAS_ALLOW_NO_GPU=1`.

#![cfg(target_os = "macos")]

use ojas_autograd::{Tape, Var};
use ojas_core::{Backend, BackendId, Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;
use ojas_metal::MetalBackend;

const ALLOW_NO_GPU: &str = "OJAS_ALLOW_NO_GPU";
/// Metal against the CPU `Exact` tape, relative to each gradient's own peak.
const DEVICE_TOL: f32 = 1e-4;
const EPS: f32 = 1e-5;
const SCALE: f32 = 0.7;

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

fn vals(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            -1.0 + 2.0 * ((s >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

fn host(seed: u64, shape: &[usize], near_one: bool) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = vals(seed, n)
        .iter()
        .map(|v| if near_one { 1.0 + 0.4 * v } else { *v })
        .collect();
    Tensor::from_f32(&data, shape, &Budget::new(u64::MAX)).unwrap()
}

/// The encoder block's leaves: x, w, b, g, beta, s, p, r.
fn block() -> [Tensor; 8] {
    [
        host(1, &[2, 3, 6], false),
        host(2, &[5, 6], false),
        host(3, &[5], false),
        host(4, &[5], true),
        host(5, &[5], false),
        host(6, &[3, 1], true),
        host(7, &[5], false),
        host(8, &[2, 3, 5], false),
    ]
}

/// linear with bias -> exact GELU -> LayerNorm -> times a per-token scale ->
/// plus a bias row -> scalar scale -> softmax -> times a fixed weighting.
fn net<B: Backend>(tape: &mut Tape<B>, v: &[Var; 8]) -> Var {
    let [x, w, b, g, beta, s, p, r] = *v;
    let h = tape.linear_bias(x, w, b).unwrap();
    let h = tape.gelu_erf(h).unwrap();
    let h = tape.layer_norm(h, g, beta, EPS).unwrap();
    let h = tape.broadcast_mul(h, s).unwrap();
    let h = tape.broadcast_add(h, p).unwrap();
    let h = tape.scale(h, SCALE).unwrap();
    let y = tape.softmax(h).unwrap();
    tape.broadcast_mul(y, r).unwrap()
}

/// The channel branch of a neck: LayerNorm2d -> GELU -> plus a `[C, 1, 1]` bias.
fn neck<B: Backend>(tape: &mut Tape<B>, v: &[Var; 4]) -> Var {
    let [x, g, b, q] = *v;
    let h = tape.channel_layer_norm(x, g, b, 1e-6).unwrap();
    let h = tape.gelu_erf(h).unwrap();
    tape.broadcast_add(h, q).unwrap()
}

fn cpu_tape() -> Tape<CpuBackend> {
    Tape::new(CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact))
}

/// Metal's gradient of each leaf against the CPU tape's, on the device.
fn agree(
    gpu: &MetalBackend,
    tape: &Tape<&MetalBackend>,
    vars: &[Var],
    cpu: &Tape<CpuBackend>,
    cvars: &[Var],
    names: &[&str],
) {
    for ((var, cvar), name) in vars.iter().zip(cvars).zip(names) {
        let grad = tape
            .grad(*var)
            .unwrap_or_else(|| panic!("{name}: no gradient"));
        assert_eq!(
            grad.device(),
            Some(BackendId::Metal),
            "{name}: left the device"
        );
        let got = gpu.download(grad).unwrap().to_f32_vec().unwrap();
        let want = cpu.grad(*cvar).unwrap().to_f32_vec().unwrap();
        let peak = want.iter().fold(0.0f32, |m, w| m.max(w.abs()));
        assert!(peak > 0.0, "{name}: the CPU gradient is all zero");
        let worst = got
            .iter()
            .zip(&want)
            .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
        eprintln!("{name}: worst {:.2e} of peak {peak:.2e}", worst / peak);
        assert!(
            worst <= DEVICE_TOL * peak,
            "{name}: {worst:e} of peak {peak:e}"
        );
    }
}

const BACKWARD_KERNELS: [&str; 5] = [
    "ojas_ln_bwd_rows",
    "ojas_gelu_erf_bwd",
    "ojas_softmax_bwd_rows",
    "ojas_bcast_grad_part",
    "ojas_cols_fin4",
];

#[test]
fn the_encoder_block_backward_runs_on_metal_and_agrees_with_the_cpu() {
    let Some(gpu) = open() else {
        return;
    };
    const NAMES: [&str; 8] = ["x", "w", "b", "g", "beta", "s", "p", "r"];
    let k = block();
    let mut cpu = cpu_tape();
    let cv = k.each_ref().map(|t| cpu.leaf(t.clone()).unwrap());
    let z = net(&mut cpu, &cv);
    cpu.backward(z).unwrap();

    let mut tape = Tape::new(&gpu);
    let v = k
        .each_ref()
        .map(|t| tape.leaf(gpu.upload(t).unwrap()).unwrap());
    let z = net(&mut tape, &v);
    let before = BACKWARD_KERNELS.map(|n| gpu.kernel_dispatches(n));
    let reads = gpu.budget().device_readbacks();
    tape.backward(z).unwrap();
    gpu.sync().unwrap();
    for (n, b) in BACKWARD_KERNELS.iter().zip(before) {
        assert!(
            gpu.kernel_dispatches(n) > b,
            "the backward never dispatched {n}"
        );
    }
    assert_eq!(
        gpu.budget().device_readbacks(),
        reads,
        "a tensor was read back"
    );
    agree(&gpu, &tape, &v, &cpu, &cv, &NAMES);
}

#[test]
fn a_checkpointed_encoder_segment_runs_its_backward_on_metal() {
    let Some(gpu) = open() else {
        return;
    };
    const NAMES: [&str; 8] = ["x", "w", "b", "g", "beta", "s", "p", "r"];
    let k = block();
    let mut cpu = cpu_tape();
    let cv = k.each_ref().map(|t| cpu.leaf(t.clone()).unwrap());
    let z = net(&mut cpu, &cv);
    cpu.backward(z).unwrap();

    let mut tape = Tape::new(&gpu);
    let v = k
        .each_ref()
        .map(|t| tape.leaf(gpu.upload(t).unwrap()).unwrap());
    let out = tape.checkpoint(|t| Ok(vec![net(t, &v)])).unwrap();
    tape.backward(out[0]).unwrap();
    agree(&gpu, &tape, &v, &cpu, &cv, &NAMES);
}

#[test]
fn the_channel_layer_norm_neck_runs_its_backward_on_metal() {
    let Some(gpu) = open() else {
        return;
    };
    const NAMES: [&str; 4] = ["x", "g", "b", "q"];
    let k = [
        host(11, &[2, 4, 3, 3], false),
        host(12, &[4], true),
        host(13, &[4], false),
        host(14, &[4, 1, 1], false),
    ];
    let mut cpu = cpu_tape();
    let cv = k.each_ref().map(|t| cpu.leaf(t.clone()).unwrap());
    let z = neck(&mut cpu, &cv);
    cpu.backward(z).unwrap();

    let mut tape = Tape::new(&gpu);
    let v = k
        .each_ref()
        .map(|t| tape.leaf(gpu.upload(t).unwrap()).unwrap());
    let z = neck(&mut tape, &v);
    let before = gpu.kernel_dispatches("ojas_cln_bwd_pos");
    tape.backward(z).unwrap();
    gpu.sync().unwrap();
    assert!(
        gpu.kernel_dispatches("ojas_cln_bwd_pos") > before,
        "the neck's backward never dispatched ojas_cln_bwd_pos"
    );
    agree(&gpu, &tape, &v, &cpu, &cv, &NAMES);
}
