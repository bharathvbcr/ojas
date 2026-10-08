//! The Qwen3.5 hybrid tower on the Metal tape: tessl's tiny fixture (one
//! gated delta net layer, one gated attention layer, the 2B's head dims)
//! forward and backward against transformers' autograd and against the CPU
//! tape, and per-layer activation checkpointing against the direct
//! recording, bit for bit. A missing device fails unless
//! `OJAS_ALLOW_NO_GPU=1`.
//!
//! Every op runs natively on Metal: the gated delta rule on tessl's
//! `gdn_train` kernels, conv1d + SiLU and the gated norm on tessl's `qwen35`
//! kernels, and the rest on ojas's own.

#![cfg(target_os = "macos")]

mod qwen35_common;

use ojas_core::{Backend, Budget, Tensor};
use ojas_metal::MetalBackend;
use ojas_model::qwen35::Qwen35Params;
use ojas_model::ActivationCheckpoint;
use qwen35_common::*;

/// Metal's gated delta rule runs within 2e-4 of the CPU's
/// (`ojas-metal/tests/gdn.rs`), so the gradients behind it are held to that
/// against both the CPU tape and transformers.
const METAL_GRAD_BOUND: f32 = 2e-4;

fn metal() -> Option<MetalBackend> {
    match MetalBackend::new(Budget::new(8 << 30)) {
        Ok(m) => Some(m),
        Err(e) if std::env::var_os("OJAS_ALLOW_NO_GPU").is_some() => {
            eprintln!("no Metal device: {e:?}");
            None
        }
        Err(e) => panic!("no Metal device (set OJAS_ALLOW_NO_GPU=1 to skip): {e:?}"),
    }
}

fn on_metal(m: &MetalBackend, activations: ActivationCheckpoint) -> Run {
    run(m.clone(), |t| m.upload(t).unwrap(), activations)
}

fn host(p: &Qwen35Params<Tensor>) -> Vec<Vec<f32>> {
    p.values()
        .into_iter()
        .map(|t| t.to_host(&budget()).unwrap().to_f32_vec().unwrap())
        .collect()
}

#[test]
fn the_metal_tape_matches_transformers_and_the_cpu_tape() {
    let Some(m) = metal() else { return };
    let got = on_metal(&m, ActivationCheckpoint::Off);
    let want = reference_loss();
    let rel = (f64::from(got.loss) - want).abs() / want.abs();
    eprintln!(
        "metal loss {} vs transformers {want} (rel {rel:.2e})",
        got.loss
    );
    assert!(rel <= LOSS_BOUND, "loss {} vs {want}", got.loss);
    let worst = compare_to_reference(&got.grads, METAL_GRAD_BOUND);
    eprintln!("worst gradient against transformers: {worst:.3e} of its peak");

    let cpu_run = run(cpu(), Tensor::clone, ActivationCheckpoint::Off);
    let (g, c) = (host(&got.grads), host(&cpu_run.grads));
    for (i, (g, c)) in g.iter().zip(&c).enumerate() {
        let peak = c.iter().fold(0.0f32, |p, x| p.max(x.abs()));
        let err = g
            .iter()
            .zip(c)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            err / peak <= METAL_GRAD_BOUND,
            "parameter {i}: {:.3e} of its peak against the CPU tape",
            err / peak
        );
    }
}

#[test]
fn checkpointed_layers_give_the_same_bits_on_metal() {
    let Some(m) = metal() else { return };
    let bits = |p: &Qwen35Params<Tensor>| -> Vec<Vec<u32>> {
        host(p)
            .into_iter()
            .map(|v| v.iter().map(|x| x.to_bits()).collect())
            .collect()
    };
    let direct = on_metal(&m, ActivationCheckpoint::Off);
    let ckpt = on_metal(&m, ActivationCheckpoint::Blocks);
    assert_eq!(direct.loss.to_bits(), ckpt.loss.to_bits());
    assert_eq!(bits(&direct.grads), bits(&ckpt.grads));
}
