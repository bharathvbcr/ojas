//! Wall time of one `forward_loss` forward and backward on the tape, on the
//! largest Qwen3.5 test fixture (`tests/fixtures/qwen35_tiny`, batch 2, 69
//! positions, vocab 8320), CPU and Metal. Ignored by default.
//!
//! It also prints `PIN <label> loss=<bits> grads=<hash>`: the loss's f32 bits
//! and an FNV-1a hash of every gradient's bits, which two builds must agree
//! on exactly.
//!
//! One invocation times `OJAS_BENCH_N` runs (default 1) and prints the
//! minimum as `BENCH <label> <seconds>`. The refactor of `forward_loss` onto
//! `forward_hidden` is judged by interleaving a binary built before it with
//! one built after, one invocation each, and taking the min of each arm.
//!
//! ```text
//! cargo test -p ojas-model --release --test qwen35_loss_bench -- --ignored --nocapture
//! ```

mod qwen35_common;

use std::time::Instant;

use ojas_autograd::Tape;
use ojas_core::{Backend, Tensor};
use ojas_io::SafeTensors;
use ojas_model::qwen35::{bind, forward_loss, load_hf, Qwen35Tables};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};
use qwen35_common::*;

/// FNV-1a over the bits of every parameter gradient, in `values()` order.
fn grad_hash<B: Backend>(
    vars: &ojas_model::qwen35::Qwen35Params<ojas_autograd::Var>,
    tape: &Tape<B>,
) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in vars.values() {
        let g = tape.grad(*v).expect("a gradient");
        let host = g.to_host(&budget()).unwrap().to_f32_vec().unwrap();
        for x in host {
            for byte in x.to_bits().to_le_bytes() {
                h = (h ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    h
}

fn time<B: Backend + Clone>(label: &str, backend: B, up: impl Fn(&Tensor) -> Tensor) {
    let n: usize = std::env::var("OJAS_BENCH_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let reference = SafeTensors::open(&dir().join("reference.safetensors")).unwrap();
    let host = load_hf(
        &s,
        &model,
        PREFIX,
        ojas_core::DType::F32,
        ojas_core::DType::F32,
        &budget(),
    )
    .unwrap();
    let (ids, targets) = ids_and_targets(&reference);
    let tables = Qwen35Tables::new(&s, ids.shape()[1], &budget()).unwrap();
    let tables = Qwen35Tables {
        cos: up(&tables.cos),
        sin: up(&tables.sin),
        ones_hidden: up(&tables.ones_hidden),
        ones_head: up(&tables.ones_head),
    };
    let device = host.try_map(&mut |t| Ok::<_, ()>(up(t))).unwrap();
    let mut best = f64::MAX;
    let mut pin = (0u32, 0u64);
    for _ in 0..n + 1 {
        let t0 = Instant::now();
        let mut tape = Tape::new(backend.clone());
        let vars = bind(&mut tape, &device).unwrap();
        let loss = forward_loss(
            &mut tape,
            &s,
            &vars,
            &ids,
            &targets,
            &tables,
            Some(IGNORE),
            DEFAULT_CE_CHUNK,
            ActivationCheckpoint::Off,
        )
        .unwrap();
        tape.backward(loss).unwrap();
        let value = tape.value(loss).unwrap().to_host(&budget()).unwrap();
        best = best.min(t0.elapsed().as_secs_f64());
        pin = (
            value.to_f32_vec().unwrap()[0].to_bits(),
            grad_hash(&vars, &tape),
        );
    }
    println!("BENCH {label} {best:.6}");
    println!("PIN {label} loss={:#010x} grads={:#018x}", pin.0, pin.1);
}

#[test]
#[ignore]
fn forward_loss_cpu() {
    time("cpu", cpu(), Tensor::clone);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn forward_loss_metal() {
    let Ok(m) = ojas_metal::MetalBackend::new(ojas_core::Budget::new(8 << 30)) else {
        println!("BENCH metal unavailable");
        return;
    };
    time("metal", m.clone(), |t| m.upload(t).unwrap());
}
