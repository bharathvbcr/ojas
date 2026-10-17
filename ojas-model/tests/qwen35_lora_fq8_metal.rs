//! LoRA on the Qwen3.5 tape on Metal over the fake-quant q8 base
//! (`qwen35_tiny_fq8`, see `qwen35_lora_fq8.rs`): the adapter gradients
//! against the CPU tape at the bounds of `qwen35_lora_metal.rs`, and a bf16
//! frozen base through the dX-only backward (the fake-quant values are bf16,
//! so the bf16 load is exact). A missing device fails unless
//! `OJAS_ALLOW_NO_GPU=1`.

#![cfg(target_os = "macos")]

mod qwen35_common;

use ojas_core::{Budget, DType, Tensor};
use ojas_metal::MetalBackend;
use ojas_model::qwen35::{load_hf, Qwen35Params};
use qwen35_common::lora::*;
use qwen35_common::*;

/// Metal's gated delta rule runs within 2e-4 of the CPU's, so the adapter
/// gradients behind it are held to that against the CPU tape.
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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn fixed(run: &LoraRun) -> Vec<Vec<u32>> {
    run.grads
        .iter()
        .flat_map(|(_, a, b)| [bits(a), bits(b)])
        .collect()
}

/// Metal's adapter gradients against the CPU tape's, over the fq8 base.
#[test]
fn metal_adapter_gradients_match_the_cpu_tape_over_the_fq8_base() {
    let Some(m) = metal() else { return };
    let fx = FQ8;
    let (d, host, sites) = (fx.data(), fx.host_params(), fixture_sites(&fx.file(), true));
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

/// The Metal loss against torch's a.loss over the fq8 base, at the loss bound
/// the CPU tape is held to plus Metal's own 2e-5.
#[test]
fn metal_loss_matches_torch_over_the_fq8_base() {
    let Some(m) = metal() else { return };
    let fx = FQ8;
    let file = fx.file();
    let run = lora_step(
        m,
        &fx.host_params(),
        &fixture_sites(&file, true),
        &fx.data(),
        &fx.opts(),
    );
    let want = read(&file, "a.loss")[0];
    let r = rel(run.loss, want);
    eprintln!(
        "{}: metal a' loss {} vs torch {want} (rel {r:.2e})",
        fx.dir, run.loss
    );
    assert!(r <= LORA_LOSS_BOUND.max(2e-5), "loss rel {r:.3e}");
}

/// A bf16 frozen base: the fq8 checkpoint is bf16, so the f32 load holds the
/// same values and the run is the f32 run bit for bit.
#[test]
fn a_bf16_fq8_base_gives_the_f32_bits_through_dx_only_on_metal() {
    let Some(m) = metal() else { return };
    let fx = FQ8;
    let (d, sites) = (fx.data(), fixture_sites(&fx.file(), true));
    let f32_base = fx.host_params();
    let bf16_base: Qwen35Params<Tensor> = load_hf(
        &spec(),
        &fx.base_file(),
        PREFIX,
        DType::Bf16,
        DType::F32,
        &budget(),
    )
    .unwrap();
    let full = lora_step(m.clone(), &f32_base, &sites, &d, &fx.opts());
    let half = lora_step(m, &bf16_base, &sites, &d, &fx.opts());
    assert_eq!(half.loss.to_bits(), full.loss.to_bits());
    assert_eq!(fixed(&half), fixed(&full));
}
