//! LoRA on the Qwen3.5 tape on Metal: the adapter gradients against the CPU
//! tape, a bf16 frozen base through the dX-only backward, and Metal against
//! the CPU with bf16 operands (bound (b'): loss within 2^-8). A missing
//! device fails unless `OJAS_ALLOW_NO_GPU=1`.

#![cfg(target_os = "macos")]

mod qwen35_common;

use ojas_core::{Autocast, Budget, DType, Tensor};
use ojas_metal::MetalBackend;
use ojas_model::qwen35::{load_hf, Qwen35Params};
use qwen35_common::lora::*;
use qwen35_common::*;

/// Metal's gated delta rule runs within 2e-4 of the CPU's, so the adapter
/// gradients behind it are held to that against the CPU tape.
const METAL_GRAD_BOUND: f32 = 2e-4;
/// Metal against the CPU with bf16 operands, on the loss.
const BF16_LOSS_BOUND: f64 = 1.0 / 256.0;

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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn fixed(run: &LoraRun) -> Vec<Vec<u32>> {
    run.grads
        .iter()
        .flat_map(|(_, a, b)| [bits(a), bits(b)])
        .collect()
}

/// Metal's adapter gradients against the CPU tape's, at `fx`'s scaling.
fn adapter_gradients_match_the_cpu_tape(fx: Fixture) {
    let Some(m) = metal() else { return };
    let (d, host, sites) = (data(), host_params(), fixture_sites(&fx.file(), true));
    let on_cpu = lora_step(cpu(), &host, &sites, &d, &fx.opts());
    let on_gpu = lora_step(m, &host, &sites, &d, &fx.opts());
    let r = (f64::from(on_gpu.loss) - f64::from(on_cpu.loss)).abs() / f64::from(on_cpu.loss);
    eprintln!(
        "{}: metal loss {} vs cpu {} (rel {r:.2e})",
        fx.dir, on_gpu.loss, on_cpu.loss
    );
    assert!(r <= LORA_LOSS_BOUND.max(2e-5), "loss rel {r:.3e}");
    for ((module, ga, gb), (_, ca, cb)) in on_gpu.grads.iter().zip(&on_cpu.grads) {
        let ra = peak_ratio(ga, ca, module);
        let rb = peak_ratio(gb, cb, module);
        eprintln!("{module}: dA {ra:.2e} dB {rb:.2e} of peak");
        assert!(ra <= METAL_GRAD_BOUND && rb <= METAL_GRAD_BOUND, "{module}");
    }
}

#[test]
fn metal_adapter_gradients_match_the_cpu_tape() {
    adapter_gradients_match_the_cpu_tape(SCALING_2);
}

/// The pre-registered run's alpha == r: no scale node on either device.
#[test]
fn metal_adapter_gradients_match_the_cpu_tape_at_scaling_one() {
    adapter_gradients_match_the_cpu_tape(SCALING_1);
}

/// A bf16 frozen base: every base linear is `Bf16`, so the dX-only backward
/// takes a bf16 weight. The fixture's checkpoint is bf16, so the f32 load
/// holds the same values and the run is the f32 run bit for bit.
#[test]
fn a_bf16_frozen_base_gives_the_f32_bits_through_dx_only_on_metal() {
    let Some(m) = metal() else { return };
    let (d, sites) = (data(), fixture_sites(&lora_file(), true));
    let f32_base = host_params();
    let bf16_base: Qwen35Params<Tensor> = load_hf(
        &spec(),
        &base_file(),
        PREFIX,
        DType::Bf16,
        DType::F32,
        &budget(),
    )
    .unwrap();
    let full = lora_step(m.clone(), &f32_base, &sites, &d, &Opts::new());
    let half = lora_step(m, &bf16_base, &sites, &d, &Opts::new());
    assert_eq!(half.loss.to_bits(), full.loss.to_bits());
    assert_eq!(fixed(&half), fixed(&full));
}

/// Bound (b'), Metal against CPU with bf16 operands, on the loss: both run
/// the same step inside an `Autocast` bf16 region.
#[test]
fn metal_matches_cpu_within_two_to_the_minus_eight_with_bf16_operands() {
    let Some(m) = metal() else { return };
    let (d, host, sites) = (data(), host_params(), fixture_sites(&lora_file(), true));
    let o = Opts {
        bf16: true,
        ..Opts::new()
    };
    let cpu_run = lora_step(Autocast::new(cpu()), &host, &sites, &d, &o);
    let gpu_run = lora_step(Autocast::new(m), &host, &sites, &d, &o);
    let r = (f64::from(gpu_run.loss) - f64::from(cpu_run.loss)).abs() / f64::from(cpu_run.loss);
    eprintln!(
        "bf16 operands: metal {} vs cpu {} (rel {r:.2e}, bound {BF16_LOSS_BOUND:.2e})",
        gpu_run.loss, cpu_run.loss
    );
    assert!(r <= BF16_LOSS_BOUND, "loss rel {r:.3e}");
    // bf16 operands are not f32: the region took effect.
    let f32_run = lora_step(cpu(), &host, &sites, &d, &Opts::new());
    assert_ne!(cpu_run.loss.to_bits(), f32_run.loss.to_bits());
}
