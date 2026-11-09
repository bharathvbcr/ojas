//! Adversarial lane bench (ft-042351e4): one optimizer step of the small CPU
//! LoRA provider, timed through the public `StepProvider` API only so the same
//! file builds at dd41efd and at c6aa5f6. Prints `BENCH step_ns=<min> norm_step_ns=<min>`
//! (min of REPS inner repetitions); the driver script interleaves the two
//! builds and takes the min of the runs. Never fails.

mod qwen35_common;

use std::time::Instant;

use ojas_core::DType;
use ojas_cpu::CpuBackend;
use ojas_io::sha256::open_sha256_hex;
use ojas_model::qwen35::adapters::BaseIdentity;
use ojas_model::qwen35::{DeviceHooks, LoraInit, OpenConfig, TapeStepProvider};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};
use ojas_train::lappi::step::{AdamWHyper, BankMode, RowTargets, SequenceJob, StepProvider};
use qwen35_common::lora::*;
use qwen35_common::*;

const REPS: usize = 40;

fn identity(base: &std::path::Path) -> BaseIdentity {
    BaseIdentity {
        file_name: "model.safetensors".to_string(),
        sha256: open_sha256_hex(base).unwrap().1,
        quantizer: None,
    }
}

fn feed(p: &mut TapeStepProvider<CpuBackend>, salt: usize) {
    for (k, mode) in [BankMode::Overwrite, BankMode::Add].into_iter().enumerate() {
        let t = 10 - k;
        let tokens: Vec<u32> = (0..t)
            .map(|i| ((i * 37 + (salt + k) * 101 + 11) % 8320) as u32)
            .collect();
        let positions: Vec<u32> = (t / 2..t).map(|q| q as u32).collect();
        let targets: Vec<u32> = positions
            .iter()
            .map(|&q| tokens[(q as usize + 1) % t])
            .collect();
        let job = SequenceJob {
            tokens: &tokens,
            rows: RowTargets {
                positions: &positions,
                targets: &targets,
                scale: 1.0,
            },
            hidden_at: &[],
        };
        p.accumulate(&job, mode, None).unwrap();
    }
}

#[test]
fn bench_one_step() {
    let base = dir().join("model.safetensors");
    let cfg = OpenConfig {
        spec: spec(),
        prefix: PREFIX.to_string(),
        base_path: base.clone(),
        base_id: identity(&base),
        linears: DType::F32,
        table: DType::F32,
        widest: 16,
        chunk: DEFAULT_CE_CHUNK,
        activations: ActivationCheckpoint::Off,
    };
    let init = LoraInit {
        config: config(),
        targets: targets(),
        adapters: fixture_sites(&lora_file(), true),
    };
    let mut p = TapeStepProvider::open_lora(cpu(), DeviceHooks::default(), cfg, init).unwrap();
    let n = p.parameters().iter().filter(|s| s.trainable).count();
    let lr_scale = vec![1.0; n];
    let wd = vec![0.01; n];
    let hyper = AdamWHyper {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: 0.5,
    };
    let (mut step_min, mut norm_min) = (u128::MAX, u128::MAX);
    for rep in 0..REPS {
        feed(&mut p, rep);
        let next = p.step_count() + 1;
        let t = Instant::now();
        p.adamw_step(&hyper, next, &lr_scale, &wd).unwrap();
        step_min = step_min.min(t.elapsed().as_nanos());

        feed(&mut p, rep + 1000);
        let next = p.step_count() + 1;
        let t = Instant::now();
        p.grad_sq_norm().unwrap();
        p.adamw_step(&hyper, next, &lr_scale, &wd).unwrap();
        norm_min = norm_min.min(t.elapsed().as_nanos());
    }
    println!("BENCH step_ns={step_min} norm_step_ns={norm_min}");
}
