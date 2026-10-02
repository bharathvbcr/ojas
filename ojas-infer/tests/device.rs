//! G7 on the GPU backends: `DeviceDecoder` on wgpu and on Metal against
//! `CpuGpt::forward_token` (`Numerics::Exact`) at [`TOL`], for MHA and GQA:
//! an 8-token prefill, 36 single-token decode steps, then a 3-token prefill
//! onto the warm cache. Every call reads back exactly one tensor, the
//! `[1, vocab]` logit row, counted on the backend's own budget.
//!
//! A missing device fails unless `OJAS_ALLOW_NO_GPU=1`, so it is never
//! counted as a pass. The tests take one lock, so they run one at a time.

use std::sync::{Mutex, MutexGuard};

use ojas_core::{Backend, Budget, Numerics, Tensor};
use ojas_infer::{CpuGpt, DeviceDecoder, GptConfig, GptWeights, KvCache, SplitMix64};
use ojas_model::{param_table, Init, ModelParams};

const TOL: f64 = 1e-4;
const PROMPT: usize = 8;
const DECODE_STEPS: usize = 36;
const TAIL: usize = 3;
const ALLOW_NO_GPU: &str = "OJAS_ALLOW_NO_GPU";

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|p| p.into_inner())
}

fn spec(n_kv_head: usize) -> GptConfig {
    GptConfig {
        vocab: 96,
        n_embd: 48,
        n_layer: 2,
        n_head: 4,
        n_kv_head,
        head_dim: 16,
        hidden: 64,
        max_seq: PROMPT + DECODE_STEPS + TAIL,
        rope_base: 10000.0,
        rms_eps: 1e-6,
        tie_embeddings: true,
    }
}

/// Every parameter non-degenerate: matrices uniform with unit-variance
/// outputs for unit-variance inputs (halved for `o_proj` and `ffn.down`),
/// norm weights in `[0.75, 1.25)`, biases and lambdas in `[-0.5, 0.5)`.
fn weights(spec: &GptConfig, seed: u64, budget: &Budget) -> GptWeights {
    let mut r = SplitMix64::new(seed);
    let flat = param_table(spec)
        .unwrap()
        .iter()
        .map(|info| {
            let n = info.numel();
            let v: Vec<f32> = match (info.init, info.shape.as_slice()) {
                (Init::Ones, _) => (0..n).map(|_| (0.75 + 0.5 * r.next_f64()) as f32).collect(),
                (_, &[_, fan_in]) => {
                    let half =
                        info.name.ends_with("o_proj.weight") || info.name.ends_with("down.weight");
                    let scale = if half { 0.5 } else { 1.0 };
                    let a = scale * (3.0 / fan_in as f64).sqrt();
                    (0..n)
                        .map(|_| ((r.next_f64() * 2.0 - 1.0) * a) as f32)
                        .collect()
                }
                _ => (0..n).map(|_| (r.next_f64() - 0.5) as f32).collect(),
            };
            Tensor::from_f32(&v, &info.shape, budget).unwrap()
        })
        .collect();
    ModelParams::from_flat(spec, flat).unwrap()
}

fn tokens(spec: &GptConfig) -> Vec<u32> {
    let n = PROMPT + DECODE_STEPS + TAIL;
    (0..n as u32)
        .map(|i| (i * 37 + 11) % spec.vocab as u32)
        .collect()
}

fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    if got.iter().chain(want).any(|v| !v.is_finite()) {
        return f64::INFINITY;
    }
    let scale = want.iter().fold(1.0f64, |m, v| m.max(f64::from(v.abs())));
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs())
        .fold(0.0, f64::max)
        / scale
}

/// The CPU reference: the last position's logits after each call's tokens.
fn reference(spec: &GptConfig, w: &GptWeights, calls: &[&[u32]]) -> Vec<Vec<f32>> {
    let budget = Budget::new(1 << 26);
    let model = CpuGpt::new(spec, w).unwrap().with_numerics(Numerics::Exact);
    let mut cache = KvCache::for_model(&model, spec.max_seq, &budget).unwrap();
    calls
        .iter()
        .map(|call| {
            let mut last = Vec::new();
            for &id in *call {
                last = model.forward_token(id, &mut cache).unwrap();
            }
            last
        })
        .collect()
}

fn check<B: Backend>(name: &str, open: impl Fn() -> B) {
    for n_kv_head in [4, 2] {
        let spec = spec(n_kv_head);
        let host = Budget::new(1 << 26);
        let w = weights(&spec, 0x6007 + n_kv_head as u64, &host);
        let ids = tokens(&spec);
        let (prompt, rest) = ids.split_at(PROMPT);
        let (steps, tail) = rest.split_at(DECODE_STEPS);
        let mut calls: Vec<&[u32]> = vec![prompt];
        calls.extend(steps.chunks(1));
        calls.push(tail);
        let want = reference(&spec, &w, &calls);

        let mut dec = DeviceDecoder::new(open(), &spec, &w, spec.max_seq).unwrap();
        let mut worst = 0.0f64;
        for (i, (call, want)) in calls.iter().zip(&want).enumerate() {
            let before = dec.backend().budget().device_readbacks();
            let got = dec.forward(call).unwrap();
            let after = dec.backend().budget().device_readbacks();
            assert_eq!(
                (after.0 - before.0, after.1 - before.1),
                (1, 4 * spec.vocab as u64),
                "{name} call {i}: readbacks (calls, bytes)"
            );
            let err = rel_err(&got, want);
            worst = worst.max(err);
            assert!(
                err <= TOL,
                "{name} n_kv_head {n_kv_head} call {i} ({} tokens): {err:e}",
                call.len()
            );
        }
        assert_eq!(dec.len(), ids.len());
        eprintln!(
            "{name} G7 n_head {} n_kv_head {n_kv_head}: prefill {PROMPT} + {DECODE_STEPS} decode \
             steps + prefill {TAIL} onto the warm cache, worst rel err vs CPU Exact {worst:e}, \
             1 readback of {} bytes per call",
            spec.n_head,
            4 * spec.vocab
        );
    }
}

fn skip_or_fail(what: &str, err: &dyn std::fmt::Display) {
    if std::env::var(ALLOW_NO_GPU).as_deref() == Ok("1") {
        eprintln!("SKIP ({ALLOW_NO_GPU}=1): {what}: {err}");
    } else {
        panic!("{what} failed: {err}. Set {ALLOW_NO_GPU}=1 to skip explicitly");
    }
}

#[test]
fn wgpu_device_decoder_matches_cpu() {
    let _guard = serial();
    if let Err(err) = ojas_wgpu::WgpuBackend::open(Budget::new(1 << 30)) {
        return skip_or_fail("WgpuBackend::open", &err);
    }
    check("wgpu", || {
        ojas_wgpu::WgpuBackend::open(Budget::new(1 << 30)).unwrap()
    });
}

#[cfg(target_os = "macos")]
#[test]
fn metal_device_decoder_matches_cpu() {
    let _guard = serial();
    if let Err(err) = ojas_metal::MetalBackend::new(Budget::new(1 << 30)) {
        return skip_or_fail("MetalBackend::new", &err);
    }
    check("metal", || {
        ojas_metal::MetalBackend::new(Budget::new(1 << 30)).unwrap()
    });
}
