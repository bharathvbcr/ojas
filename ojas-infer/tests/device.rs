//! G7 on the GPU backends: `DeviceDecoder` on wgpu and on Metal against
//! `CpuGpt::forward_token` (`Numerics::Exact`) at [`TOL`], for MHA and GQA,
//! with and without a sliding window:
//! an 8-token prefill, 36 single-token decode steps, then a 3-token prefill
//! onto the warm cache. Every call reads back exactly one tensor, the
//! `[1, vocab]` logit row, counted on the backend's own budget.
//!
//! A missing device fails unless `OJAS_ALLOW_NO_GPU=1`, so it is never
//! counted as a pass. The tests take one lock, so they run one at a time.

use std::sync::{Mutex, MutexGuard};

use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_infer::{argmax_token, CpuGpt, DeviceDecoder, GptConfig, GptWeights, KvCache, SplitMix64};
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
        window: None,
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

/// With a sliding window of 6 the caches are 11-slot rings that the 47
/// positions wrap four times, and the 8-token prompt runs as two pieces.
fn check<B: Backend>(name: &str, open: impl Fn() -> B) {
    for (n_kv_head, window) in [(4, None), (2, None), (4, Some(6)), (2, Some(6))] {
        let spec = GptConfig {
            window,
            ..spec(n_kv_head)
        };
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
        assert_eq!(dec.slots(), window.map_or(spec.max_seq, |w| 2 * w - 1));
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
            "{name} G7 n_head {} n_kv_head {n_kv_head} window {window:?}: prefill {PROMPT} + {DECODE_STEPS} decode \
             steps + prefill {TAIL} onto the warm cache, worst rel err vs CPU Exact {worst:e}, \
             1 readback of {} bytes per call",
            spec.n_head,
            4 * spec.vocab
        );
    }
}

/// Greedy decode on the device: the ids equal CPU `Exact`'s greedy ids;
/// each forward reads back one 4-byte id, and the only upload is the
/// prompt's ids, since every emitted id is fed back from the device.
fn check_greedy<B: Backend>(name: &str, open: impl Fn() -> B) {
    for n_kv_head in [4, 2] {
        let spec = spec(n_kv_head);
        let host = Budget::new(1 << 26);
        let w = weights(&spec, 0x6007 + n_kv_head as u64, &host);
        let ids = tokens(&spec);
        let prompt = &ids[..PROMPT];
        let n = DECODE_STEPS;
        let model = CpuGpt::new(&spec, &w)
            .unwrap()
            .with_numerics(Numerics::Exact);
        let mut cache = KvCache::for_model(&model, spec.max_seq, &host).unwrap();
        let want = model.greedy_decode(prompt, &mut cache, n).unwrap();

        let mut dec = DeviceDecoder::new(open(), &spec, &w, spec.max_seq).unwrap();
        let (r0, t0) = (dec.backend().budget().device_readbacks(), dec.traffic());
        let got = dec.greedy_decode(prompt, n).unwrap();
        let (r1, t1) = (dec.backend().budget().device_readbacks(), dec.traffic());
        assert_eq!(got, want, "{name} n_kv_head {n_kv_head}");
        assert_eq!(dec.len(), cache.len());
        let calls = n as u64;
        assert_eq!(
            (r1.0 - r0.0, r1.1 - r0.1),
            (calls, 4 * calls),
            "{name}: readbacks (calls, bytes)"
        );
        assert_eq!(
            (t1.uploads - t0.uploads, t1.upload_bytes - t0.upload_bytes),
            (1, 4 * PROMPT as u64),
            "{name}: uploads"
        );
        eprintln!(
            "{name} greedy n_kv_head {n_kv_head}: {n} ids equal CPU Exact, {calls} readbacks of 4 \
             bytes, 1 upload"
        );
    }
}

/// `argmax_rows` on the device at Qwen's vocabulary, with hundreds of ties
/// per row: the ids equal `argmax_token`'s; they index an embedding table
/// straight from the device (no host copy exists); a table smaller than
/// their bound is refused; a NaN is `NonFinite` by the next sync.
fn check_argmax<B: Backend>(name: &str, open: impl Fn() -> B) {
    const COLS: usize = 248_320;
    const ROWS: usize = 3;
    const DIM: usize = 4;
    let be = open();
    let mut r = SplitMix64::new(9);
    let mut host: Vec<f32> = (0..ROWS * COLS)
        .map(|_| ((r.next_u64() >> 40) % 512) as f32 * 0.5 - 100.0)
        .collect();
    // Row 2's maximum is held by -0.0 then +0.0: `>` keeps the first.
    for v in &mut host[2 * COLS..] {
        *v = -v.abs() - 1.0;
    }
    host[2 * COLS + 7] = -0.0;
    host[2 * COLS + 9] = 0.0;
    let want: Vec<u32> = host
        .chunks(COLS)
        .map(|row| argmax_token(row).unwrap())
        .collect();
    assert_eq!(want[2], 7);
    let up = |v: &[f32], shape: &[usize]| {
        be.upload(&Tensor::from_f32(v, shape, be.budget()).unwrap())
            .unwrap()
    };
    let ids = be.argmax_rows(&up(&host, &[ROWS, COLS])).unwrap();
    assert!(
        ids.device().is_some(),
        "{name}: argmax_rows left the device"
    );
    let got = be.download(&ids).unwrap();
    be.sync().unwrap();
    assert_eq!(got.u32_slice().unwrap(), want.as_slice(), "{name}");

    let table: Vec<f32> = (0..COLS * DIM).map(|i| (i / DIM) as f32).collect();
    let rows = be
        .embedding_forward(&up(&table, &[COLS, DIM]), &ids)
        .unwrap();
    let rows = be.download(&rows).unwrap();
    be.sync().unwrap();
    let rows = rows.to_f32_vec().unwrap();
    for (i, &id) in want.iter().enumerate() {
        assert_eq!(
            rows[i * DIM..(i + 1) * DIM],
            [id as f32; DIM],
            "{name} row {i}"
        );
    }
    let small = up(&table[..(COLS - 1) * DIM], &[COLS - 1, DIM]);
    let err = be.embedding_forward(&small, &ids).unwrap_err();
    assert!(matches!(err, OjasError::OutOfRange { .. }), "{name}: {err}");

    // A fresh backend: a deferred fault may leave the first one poisoned.
    let be = open();
    host[COLS + 5] = f32::NAN;
    let x = be
        .upload(&Tensor::from_f32(&host, &[ROWS, COLS], be.budget()).unwrap())
        .unwrap();
    let err = be
        .argmax_rows(&x)
        .and_then(|ids| be.download(&ids))
        .and_then(|_| be.sync())
        .unwrap_err();
    assert!(matches!(err, OjasError::NonFinite { .. }), "{name}: {err}");
    eprintln!("{name} argmax_rows: [{ROWS}, {COLS}] equals argmax_token; ids feed embedding");
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
    let open = || ojas_wgpu::WgpuBackend::open(Budget::new(1 << 30)).unwrap();
    check("wgpu", open);
    check_greedy("wgpu", open);
    check_argmax("wgpu", open);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_device_decoder_matches_cpu() {
    let _guard = serial();
    if let Err(err) = ojas_metal::MetalBackend::new(Budget::new(1 << 30)) {
        return skip_or_fail("MetalBackend::new", &err);
    }
    let open = || ojas_metal::MetalBackend::new(Budget::new(1 << 30)).unwrap();
    check("metal", open);
    check_greedy("metal", open);
    check_argmax("metal", open);
}
