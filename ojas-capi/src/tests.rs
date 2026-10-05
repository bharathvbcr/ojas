//! Tests carried over from the header-only Load, the payload-only Step and
//! the two-token greedy demo, re-pointed at the real ops on a checked-in
//! nanolab model (`tests/fixtures/nano.safetensors`), plus the shared
//! helpers the new-op tests in `ops_tests.rs` use.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ojas_core::{Backend, Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::{encode_safetensors, StDtype, TensorOut};
use ojas_model::{param_table, ModelSpec, SPEC_METADATA_KEY};

use crate::engine::{self, OP_LOAD};
use crate::gate::Check;
use crate::load::{self, INLINE_PATH_MAX};
use crate::session::{self, DeviceKind, SESSION_CAP};
use crate::train::{self, StepOut};
use crate::wire::{tag, Writer};

pub(crate) const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/nano.safetensors"
);
/// The seed `FIXTURE` was initialised with.
pub(crate) const NANO_SEED: u64 = 1337;
/// Tokens in the synthetic training bin.
pub(crate) const BIN_TOKENS: usize = 4096;
pub(crate) const SEQ: u32 = 8;

/// The fixture's spec: 2 layers, d 16, 2 x 8 heads, SwiGLU 64, vocab 64,
/// block 32. About 38 KB of f32 weights.
pub(crate) fn nano_spec() -> ModelSpec {
    ModelSpec {
        vocab: 64,
        n_embd: 16,
        n_layer: 2,
        n_head: 2,
        n_kv_head: 2,
        head_dim: 8,
        hidden: ojas_model::swiglu_hidden(16),
        max_seq: 32,
        rope_base: 10000.0,
        rms_eps: 1e-6,
        tie_embeddings: true,
    }
}

/// `init_params(nano_spec(), NANO_SEED)` as a safetensors file with the
/// spec in its metadata, optionally with `poison` set to NaN.
pub(crate) fn model_bytes(poison: Option<&str>) -> Vec<u8> {
    let spec = nano_spec();
    let budget = Budget::new(1 << 24);
    let params = ojas_model::init_params(&spec, NANO_SEED, &budget).unwrap();
    let table = param_table(&spec).unwrap();
    let data: Vec<Vec<u8>> = table
        .iter()
        .zip(&params)
        .map(|(info, t)| {
            let mut v = t.to_f32_vec().unwrap();
            if poison == Some(info.name.as_str()) {
                v[0] = f32::NAN;
            }
            v.iter().flat_map(|x| x.to_le_bytes()).collect()
        })
        .collect();
    let shapes: Vec<Vec<u64>> = table
        .iter()
        .map(|i| i.shape.iter().map(|&d| d as u64).collect())
        .collect();
    let items: Vec<TensorOut<'_>> = table
        .iter()
        .zip(&data)
        .zip(&shapes)
        .map(|((info, data), shape)| TensorOut {
            name: &info.name,
            dtype: StDtype::F32,
            shape,
            data,
        })
        .collect();
    let spec_json = spec.to_json().unwrap();
    encode_safetensors(&items, &[(SPEC_METADATA_KEY, &spec_json)]).unwrap()
}

/// The checked-in fixture is exactly `init_params` of its spec and seed,
/// written by `ojas_io::encode_safetensors`. `OJAS_BLESS_FIXTURE=1`
/// rewrites it.
#[test]
fn the_fixture_is_init_params_of_its_spec() {
    let want = model_bytes(None);
    if std::env::var_os("OJAS_BLESS_FIXTURE").is_some() {
        std::fs::write(FIXTURE, &want).unwrap();
    }
    let got = std::fs::read(FIXTURE).expect("fixture missing; run with OJAS_BLESS_FIXTURE=1");
    assert_eq!(got.len(), want.len());
    assert!(
        got == want,
        "the fixture differs from init_params({NANO_SEED})"
    );
    assert!(got.len() < 64 * 1024, "{} bytes", got.len());
    nano_spec().validate_for_training().unwrap();
}

pub(crate) fn guard() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|err| err.into_inner())
}

fn unique() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// A new model root holding the fixture as `model.safetensors`.
fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ojas-capi-{}-{}", std::process::id(), unique()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(FIXTURE, dir.join("model.safetensors")).unwrap();
    session::set_model_root(dir.to_str().unwrap()).unwrap();
    dir
}

pub(crate) fn fresh() -> (MutexGuard<'static, ()>, PathBuf) {
    let guard = guard();
    session::clear_sessions();
    let dir = scratch();
    (guard, dir)
}

/// A one-tensor file with no spec: a valid safetensors header, not a model.
pub(crate) fn write_tensor(path: &Path) {
    let header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut file = File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header).unwrap();
    file.write_all(&[0, 0, 0, 0]).unwrap();
}

/// A headerless u16 token bin of `BIN_TOKENS` ids below the fixture vocab.
pub(crate) fn write_bin(dir: &Path, name: &str) {
    let bytes: Vec<u8> = (0..BIN_TOKENS)
        .flat_map(|i| (((i * 7 + i / 64) % 64) as u16).to_le_bytes())
        .collect();
    std::fs::write(dir.join(name), bytes).unwrap();
}

pub(crate) fn write_bin_u32(dir: &Path, name: &str) {
    let bytes: Vec<u8> = (0..BIN_TOKENS)
        .flat_map(|i| (((i * 7 + i / 64) % 64) as u32).to_le_bytes())
        .collect();
    std::fs::write(dir.join(name), bytes).unwrap();
}

pub(crate) fn never() -> Check {
    Box::new(|| Ok(()))
}

pub(crate) fn call(opcode: u32, payload: &[u8]) -> Result<Vec<u8>, String> {
    engine::dispatch(&engine::context_for(opcode, false), payload).map(engine::bytes_of)
}

pub(crate) fn placement(w: Writer, device: u32, threads: u32) -> Writer {
    w.u32(tag::DEVICE, device).u32(tag::THREADS, threads)
}

pub(crate) fn load_payload(device: u32, threads: u32, path: &str) -> Vec<u8> {
    placement(Writer::default(), device, threads)
        .str(tag::PATH, path)
        .finish()
}

pub(crate) fn load_device(kind: u32, threads: u32, path: &str) -> Result<session::Session, String> {
    load::load_request(&load_payload(kind, threads, path), never())
}

/// A CPU session of `path` with exact numerics, so its results are bitwise.
pub(crate) fn load_exact(path: &str) -> Result<session::Session, String> {
    let payload = Writer::default()
        .str(tag::PATH, path)
        .u32(tag::NUMERICS, 1)
        .finish();
    load::load_request(&payload, never())
}

pub(crate) fn load_path(path: &str) -> Result<session::Session, String> {
    load_device(load::DEVICE_CPU, 1, path)
}

/// The train fields of the CPU tests: B 2, T 8, K 2, cosine 2/40, nanolab
/// learning rates and clip, abort on a non-finite value.
pub(crate) fn train_fields(w: Writer, bin: &str) -> Writer {
    w.str(tag::TOKEN_BIN, bin)
        .u32(tag::BIN_FORMAT, train::BIN_HEADERLESS)
        .u32(tag::BATCH, 2)
        .u32(tag::SEQ, SEQ)
        .u32(tag::ACCUM, 2)
        .u64(tag::DATA_SEED, 7)
        .u32(tag::SCHEDULE, train::SCHEDULE_COSINE)
        .u64(tag::WARMUP, 2)
        .u64(tag::TOTAL, 40)
        .f64(tag::MATRIX_LR, ojas_model::NANOLAB_MATRIX_LR)
        .f64(tag::ADAM_LR, ojas_model::NANOLAB_ADAM_LR)
        .f32(tag::GRAD_CLIP, ojas_model::NANOLAB_GRAD_CLIP)
        .u32(tag::ON_NONFINITE, 0)
}

pub(crate) fn open_payload(id: u64, fields: Writer) -> Vec<u8> {
    let mut p = id.to_le_bytes().to_vec();
    p.extend_from_slice(&fields.finish());
    p
}

pub(crate) fn train_open(id: u64, bin: &str) -> Result<(), String> {
    train::open_request(
        &open_payload(id, train_fields(Writer::default(), bin)),
        never(),
    )
}

pub(crate) fn sampled_payload(id: u64) -> Vec<u8> {
    let mut p = id.to_le_bytes().to_vec();
    p.extend_from_slice(&train::STEP_SAMPLED.to_le_bytes());
    p
}

/// TRAIN_STEP mode 1 over `batches` of `(rows, x, y)`.
pub(crate) fn tokens_payload(id: u64, seq: u32, batches: &[(u32, &[u32], &[u32])]) -> Vec<u8> {
    let mut p = id.to_le_bytes().to_vec();
    p.extend_from_slice(&train::STEP_TOKENS.to_le_bytes());
    p.extend_from_slice(&(batches.len() as u32).to_le_bytes());
    p.extend_from_slice(&seq.to_le_bytes());
    for (rows, x, y) in batches {
        p.extend_from_slice(&rows.to_le_bytes());
        for v in x.iter().chain(y.iter()) {
            p.extend_from_slice(&v.to_le_bytes());
        }
    }
    p
}

pub(crate) fn step(id: u64) -> Result<StepOut, String> {
    train::step_request(&sampled_payload(id), never())
}

pub(crate) fn step_out(bytes: &[u8]) -> StepOut {
    assert_eq!(bytes.len(), train::STEP_RESULT_BYTES, "{bytes:?}");
    let f32_at = |i: usize| f32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    let f64_at = |i: usize| f64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
    let u64_at = |i: usize| u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
    StepOut {
        loss: f32_at(0),
        grad_norm: f32_at(4),
        matrix_lr: f64_at(8),
        adam_lr: f64_at(16),
        step: u64_at(24),
        tokens: u64_at(32),
    }
}

pub(crate) fn sample_payload(
    id: u64,
    prompt: &[u32],
    temperature: f32,
    max_new: u32,
    seed: u64,
) -> Vec<u8> {
    open_payload(
        id,
        Writer::default()
            .f32(tag::TEMPERATURE, temperature)
            .u64(tag::SEED, seed)
            .u32(tag::MAX_NEW, max_new)
            .u32s(tag::PROMPT, prompt),
    )
}

pub(crate) fn sample(
    id: u64,
    prompt: &[u32],
    temperature: f32,
    max_new: u32,
    seed: u64,
) -> Result<Vec<u32>, String> {
    crate::generate::sample_request(
        &sample_payload(id, prompt, temperature, max_new, seed),
        never(),
    )
}

pub(crate) fn greedy_payload(id: u64, prompt: &[u32]) -> Vec<u8> {
    sample_payload(id, prompt, 0.0, 1, 0)
}

pub(crate) fn ids_of(out: &[u8]) -> Vec<u32> {
    out.as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(u32::from_le_bytes)
        .collect()
}

fn argmax_payload(id: u64, logits: &[f32]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&id.to_le_bytes());
    payload.extend_from_slice(&crate::GEN_LOGITS.to_le_bytes());
    payload.extend_from_slice(&(logits.len() as u32).to_le_bytes());
    for v in logits {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    payload
}

fn free_payload(id: u64) -> Vec<u8> {
    id.to_le_bytes().to_vec()
}

/// A model and token batch for a reference forward on a plain CPU backend.
fn eval_loss(x: &[u32], y: &[u32], rows: usize) -> f32 {
    use ojas_model::{bind, forward_loss, Eval, Rope, DEFAULT_CE_CHUNK};
    let spec = nano_spec();
    let budget = Budget::new(1 << 26);
    let params = ojas_model::init_params(&spec, NANO_SEED, &budget).unwrap();
    let mut eval = Eval::new(CpuBackend::new(budget.clone()).with_numerics(Numerics::Exact));
    let bound = bind(&mut eval, &spec, &params).unwrap();
    let seq = x.len() / rows;
    let rope = Rope::new(&spec, seq, &budget).unwrap();
    let ids = Tensor::from_u32(x, &[rows, seq], &budget).unwrap();
    let targets = Tensor::from_u32(y, &[x.len()], &budget).unwrap();
    let loss = forward_loss(
        &mut eval,
        &spec,
        &bound,
        &ids,
        &targets,
        &rope,
        None,
        DEFAULT_CE_CHUNK,
    )
    .unwrap();
    eval.backend()
        .download(&loss)
        .unwrap()
        .to_f32_vec()
        .unwrap()[0]
}

// ---- carried over from the header-only Load ----

#[test]
fn missing_file_and_path_escapes_are_rejected() {
    let (_g, dir) = fresh();
    let err = load_path("no-such.safetensors").unwrap_err();
    assert!(err.contains("missing file"), "{err}");
    for raw in [
        "../outside.safetensors",
        "sub/../../etc/passwd",
        "/etc/passwd",
    ] {
        let err = load_path(raw).unwrap_err();
        assert!(
            err.contains("..") || err.contains("relative") || err.contains("escapes"),
            "{raw}: {err}"
        );
        let err = load::inspect(raw).unwrap_err();
        assert!(
            err.contains("..") || err.contains("relative") || err.contains("escapes"),
            "{raw}: {err}"
        );
    }
    let long = "a".repeat(INLINE_PATH_MAX + 1);
    let err = load::inspect(&long).unwrap_err();
    assert!(err.contains("4096"), "{err}");
    let err = load::load_request(
        &Writer::default().raw(tag::PATH, long.as_bytes()).finish(),
        never(),
    )
    .unwrap_err();
    assert!(err.contains("4096"), "{err}");

    let outside = dir.join("outside-target.safetensors");
    std::fs::copy(FIXTURE, &outside).unwrap();
    // Move the file out of the root, then point a name inside the root at it.
    let escaped = std::env::temp_dir().join(format!("ojas-capi-out-{}", unique()));
    std::fs::rename(&outside, &escaped).unwrap();
    std::os::unix::fs::symlink(&escaped, dir.join("alias")).unwrap();
    let err = load_path("alias").unwrap_err();
    assert!(err.contains("escapes"), "{err}");
    let err = load::inspect("alias").unwrap_err();
    assert!(err.contains("escapes"), "{err}");
    let _ = std::fs::remove_file(escaped);
}

/// Inspect is the header-only Load that was: it counts tensors and refuses
/// a bad header. Load now reads the model: a file without a spec is refused.
#[test]
fn inspect_counts_tensors_and_rejects_a_bad_header() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("one.safetensors"));
    assert_eq!(load::inspect("one.safetensors").unwrap(), 1);
    let out = call(crate::OP_INSPECT, b"one.safetensors").unwrap();
    assert_eq!(out, 1u32.to_le_bytes());
    let mut bad = File::create(dir.join("bad.safetensors")).unwrap();
    bad.write_all(&100u64.to_le_bytes()).unwrap();
    let err = load::inspect("bad.safetensors").unwrap_err();
    assert!(err.contains("safetensors"), "{err}");

    let tensors = ojas_model::param_count(&nano_spec()).unwrap() as u32;
    assert_eq!(load::inspect("model.safetensors").unwrap(), tensors);
    let session = load_path("model.safetensors").unwrap();
    assert_eq!(session.tensors, tensors);
    assert_ne!(session.id, 0);
    let err = load_path("one.safetensors").unwrap_err();
    assert!(err.contains("ojas.spec"), "{err}");
    let err = load_path("bad.safetensors").unwrap_err();
    assert!(err.contains("safetensors"), "{err}");
    assert_eq!(session::session_count(), 1);
}

#[test]
fn free_unknown_and_double_free_fail() {
    let (_g, _dir) = fresh();
    let id = load_path("model.safetensors").unwrap().id;
    session::try_free(id).unwrap();
    let err = session::try_free(id).unwrap_err();
    assert!(err.contains("unknown model"), "{err}");
    let err = session::try_free(99).unwrap_err();
    assert!(err.contains("unknown model"), "{err}");
}

#[test]
fn session_table_is_capped_at_64() {
    let (_g, _dir) = fresh();
    let mut ids = Vec::new();
    for _ in 0..SESSION_CAP {
        ids.push(load_path("model.safetensors").unwrap().id);
    }
    let err = load_path("model.safetensors").unwrap_err();
    assert!(err.contains("capacity exceeded"), "{err}");
    session::try_free(ids[0]).unwrap();
    let again = load_path("model.safetensors").unwrap().id;
    assert_ne!(again, ids[0]);
    session::try_free(again).unwrap();
    for id in ids.into_iter().skip(1) {
        session::try_free(id).unwrap();
    }
    assert_eq!(session::session_count(), 0);
}

// ---- carried over from the payload-only Step ----

/// The first step's loss is the model's: bit-equal to a CPU Exact `Eval`
/// forward of the same batch. A different batch gives a different loss, a
/// bad shape leaves the session, and a non-finite value is `E_NONFINITE`.
#[test]
fn step_tokens_match_an_eval_forward_and_a_bad_shape_does_not_free() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let s = load_exact("model.safetensors").unwrap();
    train_open(s.id, "tokens.bin").unwrap();
    let x: Vec<u32> = (0..2 * SEQ).map(|i| (i * 5 + 1) % 64).collect();
    let y: Vec<u32> = (0..2 * SEQ).map(|i| (i * 5 + 6) % 64).collect();
    let got = step_out(
        &call(
            crate::OP_TRAIN_STEP,
            &tokens_payload(s.id, SEQ, &[(2, &x, &y)]),
        )
        .unwrap(),
    );
    assert_eq!(got.loss.to_bits(), eval_loss(&x, &y, 2).to_bits());
    assert!(got.grad_norm.is_finite() && got.grad_norm > 0.0, "{got:?}");
    assert_eq!(got.step, 1);
    assert_eq!(got.tokens, 2 * u64::from(SEQ));
    let mult = ojas_cpu::CosineSchedule::new(2, 40)
        .unwrap()
        .multiplier(0)
        .unwrap();
    assert_eq!(
        got.matrix_lr,
        ojas_cpu::scaled_lr(ojas_model::NANOLAB_MATRIX_LR, mult).unwrap()
    );
    assert_eq!(
        got.adam_lr,
        ojas_cpu::scaled_lr(ojas_model::NANOLAB_ADAM_LR, mult).unwrap()
    );

    let y2: Vec<u32> = y.iter().map(|t| (t + 3) % 64).collect();
    let other = step_out(
        &call(
            crate::OP_TRAIN_STEP,
            &tokens_payload(s.id, SEQ, &[(2, &x, &y2)]),
        )
        .unwrap(),
    );
    assert_ne!(got.loss.to_bits(), other.loss.to_bits());

    let err = call(
        crate::OP_TRAIN_STEP,
        &tokens_payload(s.id, SEQ, &[(2, &x[..15], &y[..15])]),
    )
    .unwrap_err();
    assert!(err.contains("truncated") || err.contains("shape"), "{err}");
    let err = call(
        crate::OP_TRAIN_STEP,
        &tokens_payload(s.id, SEQ - 1, &[(2, &x[..14], &y[..14])]),
    )
    .unwrap_err();
    assert!(err.contains("shape"), "{err}");
    session::require(s.id).unwrap();

    s.inject("linear_cross_entropy_mean", || {
        ojas_core::OjasError::NonFinite { op: "test" }
    });
    let nan = call(
        crate::OP_TRAIN_STEP,
        &tokens_payload(s.id, SEQ, &[(2, &x, &y)]),
    )
    .unwrap_err();
    assert!(nan.starts_with("ojas:E_NONFINITE:"), "{nan}");
    assert!(nan.contains("non-finite"), "{nan}");
    // Nothing was committed: the next step is step 3.
    let next = step_out(
        &call(
            crate::OP_TRAIN_STEP,
            &tokens_payload(s.id, SEQ, &[(2, &x, &y)]),
        )
        .unwrap(),
    );
    assert_eq!(next.step, 3);
    session::try_free(s.id).unwrap();
}

/// A sampled step reads its batches from the bin: the first one is the
/// sampler's first batch pair, so its loss is the mean of their forwards.
#[test]
fn sampled_steps_run_the_model_on_the_bins_batches() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let s = load_exact("model.safetensors").unwrap();
    train_open(s.id, "tokens.bin").unwrap();
    let bin = ojas_data::TokenBin::open_headerless(&dir.join("tokens.bin")).unwrap();
    let mut sampler = ojas_data::BatchSampler::new(
        &bin,
        ojas_data::SamplerConfig {
            seq_len: SEQ as usize,
            batch: 2,
            seed: 7,
        },
    )
    .unwrap();
    let a = sampler.next_batch().unwrap();
    let b = sampler.next_batch().unwrap();
    let got = step(s.id).unwrap();
    let want = (eval_loss(&a.x, &a.y, 2) * 0.5) + (eval_loss(&b.x, &b.y, 2) * 0.5);
    assert!(
        (got.loss - want).abs() <= 1e-6 * want.abs(),
        "{} vs {want}",
        got.loss
    );
    assert_eq!(got.tokens, 2 * 2 * u64::from(SEQ));
    let moved = step(s.id).unwrap();
    assert_eq!(moved.step, 2);
    assert_ne!(got.loss.to_bits(), moved.loss.to_bits());
}

#[test]
fn sampled_steps_run_the_model_on_u32_token_bins() {
    let (_g, dir) = fresh();
    write_bin_u32(&dir, "tokens_u32.bin");
    let s = load_exact("model.safetensors").unwrap();
    let mut w = Writer::default();
    w = w
        .str(tag::TOKEN_BIN, "tokens_u32.bin")
        .u32(tag::BIN_FORMAT, train::BIN_HEADERLESS_U32)
        .u32(tag::BATCH, 2)
        .u32(tag::SEQ, SEQ)
        .u32(tag::ACCUM, 2)
        .u64(tag::DATA_SEED, 7)
        .u32(tag::SCHEDULE, train::SCHEDULE_COSINE)
        .u64(tag::WARMUP, 2)
        .u64(tag::TOTAL, 40)
        .f64(tag::MATRIX_LR, ojas_model::NANOLAB_MATRIX_LR)
        .f64(tag::ADAM_LR, ojas_model::NANOLAB_ADAM_LR)
        .f32(tag::GRAD_CLIP, ojas_model::NANOLAB_GRAD_CLIP)
        .u32(tag::ON_NONFINITE, 0);
    train::open_request(&open_payload(s.id, w), never()).unwrap();

    let bin = ojas_data::TokenBin::open_headerless_u32(&dir.join("tokens_u32.bin")).unwrap();
    let mut sampler = ojas_data::BatchSampler::new(
        &bin,
        ojas_data::SamplerConfig {
            seq_len: SEQ as usize,
            batch: 2,
            seed: 7,
        },
    )
    .unwrap();
    let a = sampler.next_batch().unwrap();
    let b = sampler.next_batch().unwrap();
    let got = step(s.id).unwrap();
    let want = (eval_loss(&a.x, &a.y, 2) * 0.5) + (eval_loss(&b.x, &b.y, 2) * 0.5);
    assert!(
        (got.loss - want).abs() <= 1e-6 * want.abs(),
        "{} vs {want}",
        got.loss
    );
    assert_eq!(got.tokens, 2 * 2 * u64::from(SEQ));
    let moved = step(s.id).unwrap();
    assert_eq!(moved.step, 2);
}

/// A step payload with no mode, an unknown mode, or a session with no
/// trainer is refused, and the session stays.
#[test]
fn a_step_without_a_mode_or_a_trainer_is_refused() {
    let (_g, _dir) = fresh();
    let id = load_path("model.safetensors").unwrap().id;
    let err = call(crate::OP_TRAIN_STEP, &id.to_le_bytes()).unwrap_err();
    assert!(err.contains("truncated"), "{err}");
    let mut payload = id.to_le_bytes().to_vec();
    payload.extend_from_slice(&7u32.to_le_bytes());
    let err = call(crate::OP_TRAIN_STEP, &payload).unwrap_err();
    assert!(err.contains("unknown mode"), "{err}");
    let err = call(crate::OP_TRAIN_STEP, &sampled_payload(id)).unwrap_err();
    assert!(err.contains("no trainer"), "{err}");
    session::require(id).unwrap();
    session::try_free(id).unwrap();
}

/// A token id outside the vocabulary is refused; the last id in it is not.
#[test]
fn a_step_refuses_ids_outside_the_vocabulary() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let id = load_path("model.safetensors").unwrap().id;
    train_open(id, "tokens.bin").unwrap();
    let ok: Vec<u32> = vec![63; SEQ as usize];
    call(
        crate::OP_TRAIN_STEP,
        &tokens_payload(id, SEQ, &[(1, &ok, &ok)]),
    )
    .unwrap();
    for bad in [64u32, 1 << 16, u32::MAX] {
        let mut x = ok.clone();
        x[3] = bad;
        let err = call(
            crate::OP_TRAIN_STEP,
            &tokens_payload(id, SEQ, &[(1, &x, &ok)]),
        )
        .unwrap_err();
        assert!(
            err.contains("range") || err.contains("vocab"),
            "{bad}: {err}"
        );
        let err = call(
            crate::OP_TRAIN_STEP,
            &tokens_payload(id, SEQ, &[(1, &ok, &x)]),
        )
        .unwrap_err();
        assert!(
            err.contains("range") || err.contains("vocab") || err.contains("target"),
            "{bad}: {err}"
        );
    }
    session::try_free(id).unwrap();
}

#[test]
fn rows_times_seq_does_not_wrap() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let id = load_path("model.safetensors").unwrap().id;
    train_open(id, "tokens.bin").unwrap();
    let mut p = id.to_le_bytes().to_vec();
    for word in [train::STEP_TOKENS, 1, u32::MAX, u32::MAX] {
        p.extend_from_slice(&word.to_le_bytes());
    }
    let err = call(crate::OP_TRAIN_STEP, &p).unwrap_err();
    assert!(
        err.contains("overflows") || err.contains("truncated"),
        "{err}"
    );
    let mut p = id.to_le_bytes().to_vec();
    for word in [train::STEP_TOKENS, u32::MAX, SEQ] {
        p.extend_from_slice(&word.to_le_bytes());
    }
    let err = call(crate::OP_TRAIN_STEP, &p).unwrap_err();
    assert!(err.contains("truncated"), "{err}");
}

/// The cancel check runs between the ops of a step, and a step cancelled
/// at any of them commits nothing: the next step is the same step with the
/// same loss bits.
#[test]
fn a_step_cancelled_between_any_two_ops_commits_nothing() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let reference = {
        let s = load_exact("model.safetensors").unwrap();
        train_open(s.id, "tokens.bin").unwrap();
        let polls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&polls);
        let out = train::step_request(
            &sampled_payload(s.id),
            Box::new(move || {
                counter.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }),
        )
        .unwrap();
        session::try_free(s.id).unwrap();
        (out, polls.load(Ordering::Relaxed))
    };
    let (want, polls) = reference;
    assert!(
        polls > 100,
        "a check was dropped from the step path: {polls}"
    );
    for cut in [1, 2, polls / 3, polls / 2, polls - 1, polls] {
        let s = load_exact("model.safetensors").unwrap();
        train_open(s.id, "tokens.bin").unwrap();
        let seen = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&seen);
        let err = train::step_request(
            &sampled_payload(s.id),
            Box::new(move || {
                if counter.fetch_add(1, Ordering::Relaxed) + 1 == cut {
                    Err("cancelled: Explicit".to_string())
                } else {
                    Ok(())
                }
            }),
        )
        .unwrap_err();
        assert_eq!(err, "cancelled: Explicit", "cut {cut}");
        let next = step(s.id).unwrap();
        assert_eq!(next.step, 1, "cut {cut}: the cancelled step committed");
        assert_eq!(next.loss.to_bits(), want.loss.to_bits(), "cut {cut}");
        assert_eq!(
            next.grad_norm.to_bits(),
            want.grad_norm.to_bits(),
            "cut {cut}"
        );
        session::try_free(s.id).unwrap();
    }
}

/// A budget that holds the model and its trainer but not a step's
/// activations refuses the step with `E_CAPACITY`, and leaves the trainer
/// usable (Save refuses a poisoned one).
#[test]
fn a_step_returns_capacity_when_its_activations_do_not_fit() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let open_at = |budget: u64| -> Result<u64, String> {
        let payload = Writer::default()
            .str(tag::PATH, "model.safetensors")
            .u64(tag::BUDGET, budget)
            .finish();
        let s = load::load_request(&payload, never())?;
        match train_open(s.id, "tokens.bin") {
            Ok(()) => Ok(s.id),
            Err(e) => {
                session::try_free(s.id).unwrap();
                Err(e)
            }
        }
    };
    // Smallest budget whose load and trainer fit.
    let (mut lo, mut hi) = (1u64, 1u64 << 24);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match open_at(mid) {
            Ok(id) => {
                session::try_free(id).unwrap();
                hi = mid;
            }
            Err(e) => {
                assert!(e.starts_with("ojas:E_CAPACITY:"), "{mid}: {e}");
                lo = mid + 1;
            }
        }
    }
    let id = open_at(lo).unwrap();
    let err = step(id).unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    assert!(err.contains("capacity exceeded"), "{err}");
    train::save_request(
        &{
            let mut p = id.to_le_bytes().to_vec();
            p.extend_from_slice(b"ckpt");
            p
        },
        never(),
    )
    .unwrap();
    assert!(dir.join("ckpt/state.ojck").is_file());
}

// ---- carried over from the greedy demo ----

/// Caller-logits argmax is unchanged. Greedy sampling reads the model: it
/// equals `CpuGpt`'s greedy decode of the same weights, and an id outside
/// the vocabulary is out of range.
#[test]
fn argmax_refuses_nan_and_greedy_sampling_reads_the_model() {
    let err = crate::argmax(&[f32::NAN, f32::NAN]).unwrap_err();
    assert!(err.contains("non-finite"), "{err}");
    assert!(err.contains("argmax_token"), "{err}");
    assert_eq!(crate::argmax(&[0.1, 2.5, 0.2]).unwrap(), 1);

    let (_g, _dir) = fresh();
    let id = load_path("model.safetensors").unwrap().id;
    let spec = nano_spec();
    let budget = Budget::new(1 << 26);
    let params = ojas_model::ModelParams::from_flat(
        &spec,
        ojas_model::init_params(&spec, NANO_SEED, &budget).unwrap(),
    )
    .unwrap();
    let gpt = ojas_infer::CpuGpt::new(&spec, &params).unwrap();
    for prompt in [&[0u32][..], &[5], &[1, 2, 3], &[63, 0, 17, 9]] {
        let mut cache = ojas_infer::KvCache::for_model(&gpt, 32, &budget).unwrap();
        let want = gpt.greedy_decode(prompt, &mut cache, 4).unwrap();
        let got = ids_of(&call(crate::OP_SAMPLE, &sample_payload(id, prompt, 0.0, 4, 0)).unwrap());
        assert_eq!(got, want, "{prompt:?}");
        let one = ids_of(&call(crate::OP_SAMPLE, &greedy_payload(id, prompt)).unwrap());
        assert_eq!(one, want[..1], "{prompt:?}");
    }
    let err = call(crate::OP_SAMPLE, &greedy_payload(id, &[64])).unwrap_err();
    assert!(err.contains("range") || err.contains("vocab"), "{err}");
    session::try_free(id).unwrap();
}

/// Sampling a prompt that would pass the model's context is `E_CAPACITY`
/// before any forward; so is a session budget too small for the KV cache.
#[test]
fn sample_returns_capacity_past_the_context_or_the_budget() {
    let (_g, _dir) = fresh();
    let id = load_path("model.safetensors").unwrap().id;
    let prompt: Vec<u32> = (0..30).collect();
    let err = sample(id, &prompt, 0.0, 4, 0).unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    sample(id, &prompt, 0.0, 3, 0).unwrap();

    // Weights fit; the cache and activations of a full-context sample do not.
    // The weights, plus the one bounded chunk the loader decodes each tensor
    // through (charged beside it): the largest tensor, at most
    // `LE_READ_CHUNK_BYTES`.
    let spec = nano_spec();
    let largest = param_table(&spec)
        .unwrap()
        .iter()
        .map(|i| i.shape.iter().product::<usize>() as u64 * 4)
        .max()
        .unwrap()
        .min(ojas_core::LE_READ_CHUNK_BYTES as u64);
    let weights = ojas_model::param_bytes(&spec).unwrap();
    let payload = Writer::default()
        .str(tag::PATH, "model.safetensors")
        .u64(tag::BUDGET, weights + largest)
        .finish();
    let tight = load::load_request(&payload, never()).unwrap();
    let err = sample(tight.id, &prompt, 0.0, 3, 0).unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    assert!(err.contains("capacity exceeded"), "{err}");
}

/// A cancel at any poll of a sample returns the cancel, and the next sample
/// is the reference.
#[test]
fn sample_honors_a_cancel_mid_decode() {
    let (_g, _dir) = fresh();
    let id = load_path("model.safetensors").unwrap().id;
    let polls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&polls);
    let payload = sample_payload(id, &[3, 4, 5], 0.8, 6, 11);
    let want = crate::generate::sample_request(
        &payload,
        Box::new(move || {
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }),
    )
    .unwrap();
    let polls = polls.load(Ordering::Relaxed);
    assert!(polls > 20, "{polls}");
    for cut in [1, polls / 2, polls] {
        let seen = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&seen);
        let err = crate::generate::sample_request(
            &payload,
            Box::new(move || {
                if counter.fetch_add(1, Ordering::Relaxed) + 1 == cut {
                    Err("cancelled: Explicit".to_string())
                } else {
                    Ok(())
                }
            }),
        )
        .unwrap_err();
        assert_eq!(err, "cancelled: Explicit", "cut {cut}");
        assert_eq!(sample(id, &[3, 4, 5], 0.8, 6, 11).unwrap(), want);
    }
}

// ---- table, payload and boundary tests ----

#[test]
fn a_poisoned_session_table_recovers_once_and_then_keeps_sessions() {
    let (_g, _dir) = fresh();
    let before = load_path("model.safetensors").unwrap().id;
    session::poison_table();
    // The first lock after the panic drops whatever the panic left behind.
    assert!(session::require(before).is_err());
    let after = load_path("model.safetensors").unwrap().id;
    assert!(after > before, "{after} <= {before}");
    session::require(after).unwrap();
    assert_eq!(session::session_count(), 1);
    session::try_free(after).unwrap();
}

#[test]
fn concurrent_open_step_sample_close_keeps_the_cap_and_never_reuses_an_id() {
    const THREADS: usize = 16;
    const ROUNDS: usize = 150;
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let issued = Arc::new(Mutex::new(std::collections::HashSet::new()));
    let peak = Arc::new(AtomicUsize::new(0));
    let load = load_payload(load::DEVICE_CPU, 1, "model.safetensors");
    let x: Vec<u32> = (0..SEQ).collect();
    let workers: Vec<_> = (0..THREADS)
        .map(|t| {
            let issued = Arc::clone(&issued);
            let peak = Arc::clone(&peak);
            let load = load.clone();
            let x = x.clone();
            std::thread::spawn(move || {
                let mut held = Vec::new();
                for round in 0..ROUNDS {
                    match call(OP_LOAD, &load) {
                        Ok(out) => {
                            let id = u64::from_le_bytes(out[..8].try_into().unwrap());
                            assert!(issued.lock().unwrap().insert(id), "id {id} reused");
                            held.push(id);
                        }
                        Err(err) => assert!(err.contains("capacity exceeded"), "{err}"),
                    }
                    peak.fetch_max(session::session_count(), Ordering::Relaxed);
                    let Some(&id) = held.last() else { continue };
                    // A session kept from an earlier round already trains.
                    if let Err(err) = train_open(id, "tokens.bin") {
                        assert!(err.contains("already has a trainer"), "{err}");
                    }
                    call(
                        crate::OP_TRAIN_STEP,
                        &tokens_payload(id, SEQ, &[(1, &x, &x)]),
                    )
                    .unwrap();
                    assert_eq!(
                        call(crate::OP_GENERATE, &argmax_payload(id, &[0.0, 2.0])).unwrap(),
                        1u32.to_le_bytes()
                    );
                    call(crate::OP_SAMPLE, &greedy_payload(id, &[0, 1])).unwrap();
                    if (round + t) % 3 != 0 {
                        let id = held.pop().unwrap();
                        call(crate::OP_FREE, &free_payload(id)).unwrap();
                        let err = call(crate::OP_FREE, &free_payload(id)).unwrap_err();
                        assert!(err.contains("unknown model"), "{err}");
                        let err = call(crate::OP_TRAIN_STEP, &sampled_payload(id)).unwrap_err();
                        assert!(err.contains("unknown model"), "{err}");
                    }
                }
                for id in held {
                    call(crate::OP_FREE, &free_payload(id)).unwrap();
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    assert!(peak.load(Ordering::Relaxed) <= SESSION_CAP);
    assert_eq!(session::session_count(), 0);
    assert!(issued.lock().unwrap().len() >= THREADS);
}

#[test]
fn racing_loads_fill_exactly_the_cap() {
    const THREADS: usize = SESSION_CAP * 2;
    let (_g, _dir) = fresh();
    let barrier = Arc::new(std::sync::Barrier::new(THREADS));
    let results: Vec<_> = (0..THREADS)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                load_path("model.safetensors")
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    let ids: std::collections::HashSet<u64> = results
        .iter()
        .filter_map(|r| r.as_ref().ok().map(|s| s.id))
        .collect();
    assert_eq!(ids.len(), SESSION_CAP);
    for err in results.iter().filter_map(|r| r.as_ref().err()) {
        assert!(err.contains("capacity exceeded"), "{err}");
    }
    assert_eq!(session::session_count(), SESSION_CAP);
    for id in ids {
        session::try_free(id).unwrap();
    }
    assert_eq!(session::session_count(), 0);
}

#[test]
fn malformed_payloads_are_errors_and_never_panic() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let id = load_path("model.safetensors").unwrap().id;
    let trained = load_path("model.safetensors").unwrap().id;
    train_open(trained, "tokens.bin").unwrap();
    let no_panic = |opcode: u32, payload: &[u8]| -> Result<Vec<u8>, String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| call(opcode, payload)))
            .unwrap_or_else(|_| panic!("opcode {opcode} panicked on {payload:?}"))
    };

    let x: Vec<u32> = (0..SEQ).collect();
    // Fixed-shape payloads: every strict prefix and one extra byte must fail.
    let valid = [
        (
            crate::OP_TRAIN_STEP,
            tokens_payload(trained, SEQ, &[(1, &x, &x)]),
        ),
        (crate::OP_TRAIN_STEP, sampled_payload(trained)),
        (crate::OP_GENERATE, argmax_payload(id, &[0.0, 2.0, 1.0])),
        (crate::OP_SAMPLE, sample_payload(id, &[0, 1, 1], 0.5, 2, 3)),
        (
            crate::OP_LOAD,
            load_payload(load::DEVICE_CPU, 1, "model.safetensors"),
        ),
        (crate::OP_FREE, free_payload(u64::MAX)),
    ];
    let mut prefixes = 0usize;
    for (opcode, payload) in &valid {
        if *opcode != crate::OP_FREE {
            let out = no_panic(*opcode, payload).unwrap();
            if *opcode == crate::OP_LOAD {
                session::try_free(u64::from_le_bytes(out[..8].try_into().unwrap())).unwrap();
            }
        }
        for cut in 0..payload.len() {
            prefixes += 1;
            no_panic(*opcode, &payload[..cut]).unwrap_err();
        }
        let mut long = payload.clone();
        long.push(0);
        no_panic(*opcode, &long).unwrap_err();
    }
    assert!(prefixes > 100, "{prefixes}");

    // Count fields at their limits must fail on length, not allocate or wrap.
    for (k, seq, rows) in [
        (u32::MAX, u32::MAX, u32::MAX),
        (u32::MAX, 1, 1),
        (1, u32::MAX, 2),
        (1, 1, u32::MAX),
        (0, 1, 1),
        (65_536, 65_536, 65_536),
    ] {
        let mut p = trained.to_le_bytes().to_vec();
        for word in [train::STEP_TOKENS, k, seq, rows] {
            p.extend_from_slice(&word.to_le_bytes());
        }
        p.extend_from_slice(&[0u8; 16]);
        no_panic(crate::OP_TRAIN_STEP, &p).unwrap_err();
    }
    for mode in [crate::GEN_LOGITS, 2, 0, u32::MAX] {
        let mut p = id.to_le_bytes().to_vec();
        p.extend_from_slice(&mode.to_le_bytes());
        p.extend_from_slice(&u32::MAX.to_le_bytes());
        p.extend_from_slice(&[0u8; 8]);
        no_panic(crate::OP_GENERATE, &p).unwrap_err();
    }
    no_panic(crate::OP_GENERATE, &argmax_payload(id, &[])).unwrap_err();
    no_panic(crate::OP_SAMPLE, &sample_payload(id, &[], 0.0, 1, 0)).unwrap_err();
    no_panic(
        crate::OP_GENERATE,
        &argmax_payload(id, &[f32::INFINITY, 0.0]),
    )
    .unwrap_err();
    for path in [
        &b"\xff\xfe"[..],
        b"a\0b",
        b"",
        b".",
        b"./",
        b"//",
        b"model.safetensors/",
    ] {
        let got = no_panic(crate::OP_INSPECT, path);
        assert!(
            got.is_err(),
            "{:?} inspected: {got:?}",
            String::from_utf8_lossy(path)
        );
        let got = no_panic(OP_LOAD, &Writer::default().raw(tag::PATH, path).finish());
        assert!(
            got.is_err(),
            "{:?} loaded: {got:?}",
            String::from_utf8_lossy(path)
        );
    }

    // Deterministic xorshift so a failure reproduces.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let ops = [
        crate::OP_TRAIN_STEP,
        crate::OP_GENERATE,
        crate::OP_LOAD,
        crate::OP_NEW,
        crate::OP_TRAIN_OPEN,
        crate::OP_SAMPLE,
        crate::OP_TOKENIZE,
        crate::OP_TOKENIZER,
        crate::OP_RESUME,
    ];
    const RANDOM: usize = 20_000;
    for _ in 0..RANDOM {
        let opcode = ops[(next() % ops.len() as u64) as usize];
        let len = (next() % 96) as usize;
        let mut payload: Vec<u8> = (0..len).map(|_| next() as u8).collect();
        if len >= 8 && next() % 2 == 0 {
            payload[..8].copy_from_slice(&[id, trained][(next() % 2) as usize].to_le_bytes());
        }
        if len >= 12 {
            let mode = (next() % 3) as u32;
            payload[8..12].copy_from_slice(&mode.to_le_bytes());
        }
        if let Ok(out) = no_panic(opcode, &payload) {
            if matches!(opcode, crate::OP_LOAD | crate::OP_NEW | crate::OP_RESUME)
                && out.len() == 12
            {
                let _ = session::try_free(u64::from_le_bytes(out[..8].try_into().unwrap()));
            }
        }
    }
    session::reset_sessions();
}

#[test]
fn cancel_is_reported_before_the_handler_runs() {
    let _g = guard();
    for op in [
        OP_LOAD,
        crate::OP_NEW,
        crate::OP_TRAIN_STEP,
        crate::OP_SAMPLE,
        crate::OP_RESUME,
    ] {
        let ctx = engine::context_for(op, true);
        let err = engine::dispatch(&ctx, b"model.safetensors").unwrap_err();
        assert_eq!(err, "cancelled: Explicit");
    }
}

#[test]
fn opcode_zero_a_duplicate_opcode_and_the_retired_step_are_refused() {
    let _g = guard();
    gusset::clear_engine_handlers();
    let err = gusset::register_engine(0, |_ctx, _input| Ok(Vec::<u8>::new())).unwrap_err();
    assert!(err.contains("opcode 0"), "{err}");
    gusset::register_engine(crate::OP_LOAD, |_ctx, _input| Ok(Vec::<u8>::new())).unwrap();
    let err =
        gusset::register_engine(crate::OP_LOAD, |_ctx, _input| Ok(Vec::<u8>::new())).unwrap_err();
    assert!(err.contains("already registered"), "{err}");
    engine::install_engine().unwrap();
    let err = call(crate::OP_STEP_RETIRED, &[0u8; 32]).unwrap_err();
    assert!(err.contains("unknown opcode 2"), "{err}");
}

#[test]
fn small_results_stay_byte_vectors_before_the_allocator_api() {
    let out = engine::stage(vec![9, 8, 7]);
    match out {
        gusset::JobOutput::Bytes(bytes) => assert_eq!(bytes, vec![9, 8, 7]),
        other => panic!("small result left the pre-1.100 path: {other:?}"),
    }
}

#[test]
fn device_fields_select_cpu_parallel_and_bound_its_thread_count() {
    let (_g, _dir) = fresh();
    let session = load_device(load::DEVICE_CPU_PARALLEL, 4, "model.safetensors").unwrap();
    assert_eq!(
        session.device,
        DeviceKind::Cpu { threads: 4 },
        "{session:?}"
    );
    session::try_free(session.id).unwrap();
    let session = load_device(
        load::DEVICE_CPU_PARALLEL,
        load::MAX_CPU_THREADS,
        "model.safetensors",
    )
    .unwrap();
    assert_eq!(
        session.device,
        DeviceKind::Cpu {
            threads: load::MAX_CPU_THREADS as usize
        }
    );
    session::try_free(session.id).unwrap();

    // The auto device sizes its pool from this machine; `threads` is not
    // read, so a 0 there is not refused.
    let auto = load_device(load::DEVICE_CPU_AUTO, 0, "model.safetensors").unwrap();
    let ceiling = ojas_device::ResourcePlan::derive(
        &ojas_device::ResourcePolicy::new(u64::MAX),
        &ojas_device::probe_system(),
        &[] as &[crate::profile::NoProbe],
    )
    .thread_ceiling;
    assert_eq!(
        auto.device,
        DeviceKind::Cpu {
            threads: load::auto_threads(ceiling).unwrap()
        }
    );
    session::try_free(auto.id).unwrap();

    let err = load_device(9, 1, "model.safetensors").unwrap_err();
    assert!(err.contains("unknown device"), "{err}");
    let err = load_device(load::DEVICE_CPU_PARALLEL, 0, "model.safetensors").unwrap_err();
    assert!(err.contains("thread count is 0"), "{err}");
    for threads in [load::MAX_CPU_THREADS + 1, 1 << 20, u32::MAX] {
        let err = load_device(load::DEVICE_CPU_PARALLEL, threads, "model.safetensors")
            .expect_err("an unbounded thread count was accepted");
        assert!(err.contains("exceeds"), "{threads}: {err}");
    }
    let payload = load_payload(load::DEVICE_CPU, 1, "");
    for cut in 1..20 {
        let err = load::load_request(&payload[..cut], never()).unwrap_err();
        assert!(err.contains("truncated"), "{cut}: {err}");
    }
    for (numerics, device, want) in [
        (3u32, load::DEVICE_CPU, "unknown numerics"),
        (1, load::DEVICE_METAL, "fixed by"),
        (2, load::DEVICE_WGPU, "fixed by"),
    ] {
        let p = placement(Writer::default(), device, 1)
            .u32(tag::NUMERICS, numerics)
            .str(tag::PATH, "model.safetensors")
            .finish();
        let err = load::load_request(&p, never()).unwrap_err();
        assert!(err.contains(want), "{numerics} {device}: {err}");
    }
    let err = load::load_request(
        &Writer::default()
            .str(tag::PATH, "model.safetensors")
            .u64(tag::BUDGET, 0)
            .finish(),
        never(),
    )
    .unwrap_err();
    assert!(err.contains("budget is 0"), "{err}");
    assert_eq!(session::session_count(), 0);
}

/// Far above the load ceiling. The error is returned before a session, and
/// therefore before `CpuBackend::with_threads`, exists.
#[test]
fn cpu_parallel_load_refuses_a_thread_bomb() {
    let (_g, _dir) = fresh();
    let before = session::session_count();
    let err = load_device(load::DEVICE_CPU_PARALLEL, u32::MAX, "model.safetensors")
        .expect_err("a thread count above 256 was accepted");
    assert!(err.contains("exceeds"), "{err}");
    assert!(err.contains("256"), "{err}");
    assert_eq!(session::session_count(), before);
}

/// A session budget above the process ceiling (1 GiB by default) can never
/// be met, so the load refuses it with `E_CAPACITY` before any device opens
/// and adds no session.
#[test]
fn load_refuses_a_session_budget_above_the_process_ceiling() {
    let (_g, _dir) = fresh();
    let payload = Writer::default()
        .str(tag::PATH, "model.safetensors")
        .u64(tag::BUDGET, 2 << 30)
        .finish();
    let err = load::load_request(&payload, never()).unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    assert!(err.contains("process ceiling"), "{err}");
    assert_eq!(session::session_count(), 0);
}

/// Restores the default process ceiling when a ceiling test ends, pass or
/// fail, so a small test ceiling never leaks into the next test.
struct DefaultCeiling;

impl Drop for DefaultCeiling {
    fn drop(&mut self) {
        session::clear_sessions();
        if let Err(err) = session::set_memory_ceiling(session::DEFAULT_MEMORY_CEILING_BYTES) {
            eprintln!("could not restore the default memory ceiling: {err}");
        }
    }
}

fn set_ceiling(bytes: u64) -> Result<Vec<u8>, String> {
    call(crate::OP_SET_MEMORY_CEILING, &bytes.to_le_bytes())
}

fn load_with_budget(budget: u64) -> Result<session::Session, String> {
    load::load_request(
        &Writer::default()
            .str(tag::PATH, "model.safetensors")
            .u64(tag::BUDGET, budget)
            .finish(),
        never(),
    )
}

/// Sessions draw from one process ceiling. With the ceiling set to the
/// smallest that holds one load, two sessions whose own budgets each equal
/// it cannot both load: the second is `E_CAPACITY`. Freeing the first
/// releases its charge, and the second then loads.
#[test]
fn sessions_together_cannot_pass_the_process_ceiling() {
    let (_g, _dir) = fresh();
    let _restore = DefaultCeiling;
    assert_eq!(
        session::memory_ceiling().unwrap(),
        (session::DEFAULT_MEMORY_CEILING_BYTES, 0)
    );
    // Smallest ceiling (and equal session budget) one load fits under.
    let (mut lo, mut hi) = (1u64, 1u64 << 24);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        set_ceiling(mid).unwrap();
        match load_with_budget(mid) {
            Ok(s) => {
                session::try_free(s.id).unwrap();
                hi = mid;
            }
            Err(e) => {
                assert!(e.starts_with("ojas:E_CAPACITY:"), "{mid}: {e}");
                lo = mid + 1;
            }
        }
    }
    set_ceiling(lo).unwrap();
    let first = load_with_budget(lo).unwrap();
    let (cap, held) = session::memory_ceiling().unwrap();
    assert_eq!(cap, lo);
    assert!(held > 0 && held <= lo, "{held} of {lo}");

    let err = load_with_budget(lo).unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    assert!(err.contains("capacity exceeded"), "{err}");
    assert_eq!(session::session_count(), 1);
    // The refused load released whatever it had charged.
    assert_eq!(session::memory_ceiling().unwrap(), (lo, held));

    session::try_free(first.id).unwrap();
    drop(first);
    assert_eq!(session::memory_ceiling().unwrap(), (lo, 0));
    let second = load_with_budget(lo).unwrap();
    session::try_free(second.id).unwrap();
}

/// The setter refuses 0, a malformed payload, and any change while a model
/// holds the ceiling (a `Budget`'s cap is fixed, so a new root under live
/// sessions would split the accounting). Once every model is freed it
/// takes effect.
#[test]
fn the_ceiling_changes_only_with_no_model_open() {
    let (_g, _dir) = fresh();
    let _restore = DefaultCeiling;
    let err = set_ceiling(0).unwrap_err();
    assert!(err.contains("0 bytes"), "{err}");
    for bad in [&[][..], &[0u8; 7][..], &[0u8; 9][..]] {
        let err = call(crate::OP_SET_MEMORY_CEILING, bad).unwrap_err();
        assert!(err.contains("shape"), "{err}");
    }
    assert_eq!(
        session::memory_ceiling().unwrap().0,
        session::DEFAULT_MEMORY_CEILING_BYTES
    );

    let s = load_path("model.safetensors").unwrap();
    let err = set_ceiling(2 << 30).unwrap_err();
    assert!(err.contains("free them first"), "{err}");
    // Freed from the table while a call still holds the model: the lease
    // lives until that call returns, so the ceiling is still pinned.
    let (release, held) = s.hold();
    session::try_free(s.id).unwrap();
    drop(s);
    let err = set_ceiling(2 << 30).unwrap_err();
    assert!(err.contains("free them first"), "{err}");
    release.send(()).unwrap();
    held.join().unwrap();
    assert_eq!(
        session::memory_ceiling().unwrap().0,
        session::DEFAULT_MEMORY_CEILING_BYTES
    );

    set_ceiling(2 << 30).unwrap();
    assert_eq!(session::memory_ceiling().unwrap(), (2 << 30, 0));
    // A session budget the raised ceiling covers now loads.
    let big = load_with_budget(2 << 30).unwrap();
    session::try_free(big.id).unwrap();
}

/// A ceiling above the machine's memory is `E_CAPACITY` and changes
/// nothing; one at the limit is accepted. On this host the limit is read.
#[test]
fn a_ceiling_above_the_machine_is_refused() {
    let (_g, _dir) = fresh();
    let _restore = DefaultCeiling;
    let limit = 8u64 << 30;
    let err = session::set_memory_ceiling_within(limit + 1, Some(limit)).unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    assert!(err.contains("exceeds this machine"), "{err}");
    assert_eq!(
        session::memory_ceiling().unwrap(),
        (session::DEFAULT_MEMORY_CEILING_BYTES, 0)
    );
    session::set_memory_ceiling_within(limit, Some(limit)).unwrap();
    assert_eq!(session::memory_ceiling().unwrap(), (limit, 0));
    // An unreadable machine is not checked.
    session::set_memory_ceiling_within(u64::MAX, None).unwrap();
    let real = session::hard_memory_limit().expect("this host reports its memory");
    let err = set_ceiling(real + 1).unwrap_err();
    assert!(err.contains("exceeds this machine"), "{err}");
}

/// The auto device's thread count: the plan's ceiling, clamped to
/// `MAX_CPU_THREADS`, and a refusal (not a guess) when it is unknown.
#[test]
fn auto_threads_clamp_the_ceiling_and_refuse_an_unknown_one() {
    use ojas_device::MemoryReport;
    assert_eq!(load::auto_threads(MemoryReport::Known(1)), Ok(1));
    assert_eq!(load::auto_threads(MemoryReport::Known(18)), Ok(18));
    let max = load::MAX_CPU_THREADS as usize;
    assert_eq!(load::auto_threads(MemoryReport::Known(256)), Ok(max));
    assert_eq!(load::auto_threads(MemoryReport::Known(1024)), Ok(max));
    assert_eq!(load::auto_threads(MemoryReport::Known(u64::MAX)), Ok(max));
    for bad in [MemoryReport::Unknown, MemoryReport::Known(0)] {
        let err = load::auto_threads(bad).unwrap_err();
        assert!(
            err.contains("cannot read this machine's CPU count"),
            "{err}"
        );
    }
}

/// Under critical memory pressure every call that allocates model-sized
/// memory is refused with `E_PRESSURE` (never `E_CAPACITY`, which means the
/// work will not fit) before it starts, and FREE, SAVE and the queries still
/// run; milder or unknown pressure refuses nothing.
#[test]
fn critical_pressure_refuses_allocating_calls_and_keeps_save_and_free() {
    use crate::engine::admit;
    use ojas_device::MemoryPressure;
    let allocating = [
        crate::OP_LOAD,
        crate::OP_NEW,
        crate::OP_TRAIN_OPEN,
        crate::OP_TRAIN_STEP,
        crate::OP_RESUME,
        crate::OP_SAMPLE,
        crate::OP_GENERATE,
    ];
    let other = [
        crate::OP_FREE,
        crate::OP_SAVE,
        crate::OP_INSPECT,
        crate::OP_TOKENIZE,
        crate::OP_SET_MEMORY_CEILING,
        crate::OP_SYSTEM_PROFILE,
    ];
    for op in allocating {
        let err = admit(op, MemoryPressure::Critical).unwrap_err();
        assert!(err.starts_with("ojas:E_PRESSURE: "), "{op}: {err}");
        assert!(!err.contains("E_CAPACITY"), "{op}: {err}");
        assert!(err.contains("critical memory pressure"), "{err}");
    }
    for op in other {
        admit(op, MemoryPressure::Critical).unwrap();
    }
    for pressure in [
        MemoryPressure::Normal,
        MemoryPressure::Warning,
        MemoryPressure::Unknown,
    ] {
        for op in allocating.iter().chain(&other) {
            admit(*op, pressure).unwrap();
        }
    }
}

/// A Metal head dimension past the kernel limit, and a model larger than
/// its session budget, are refused before the device opens. Head dim 256
/// is inside that limit.
#[test]
fn preflight_refuses_before_the_device_opens() {
    use crate::model::Placement;
    let spec = ojas_model::ModelSpec::tiny();
    let metal = |budget_bytes| Placement {
        device: session::DeviceKind::Metal,
        budget_bytes,
        numerics: None,
    };
    load::preflight("load", &metal(1 << 30), &spec).unwrap();
    let limit = ojas_core::METAL_MAX_HEAD_DIM as usize;
    let wide = ojas_model::ModelSpec {
        head_dim: limit,
        ..spec
    };
    load::preflight("load", &metal(1 << 30), &wide).unwrap();
    let over = ojas_model::ModelSpec {
        head_dim: limit + 1,
        ..spec
    };
    let err = load::preflight("load", &metal(1 << 30), &over).unwrap_err();
    assert!(
        err.contains(&format!("unsupported head dim {}", limit + 1)),
        "{err}"
    );
    assert!(err.contains(&format!("limit {limit}")), "{err}");
    // The CPU runs any head_dim the spec itself accepts.
    let cpu = Placement {
        device: session::DeviceKind::Cpu { threads: 1 },
        budget_bytes: 1 << 30,
        numerics: None,
    };
    load::preflight("load", &cpu, &wide).unwrap();
    let bytes = ojas_model::param_bytes(&spec).unwrap();
    load::preflight("new", &metal(bytes), &spec).unwrap();
    let err = load::preflight("new", &metal(bytes - 1), &spec).unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
}

/// A model the session's own budget cannot hold is `E_CAPACITY` at load,
/// and no session is added.
#[test]
fn load_refuses_a_model_its_budget_cannot_hold() {
    let (_g, _dir) = fresh();
    let payload = Writer::default()
        .str(tag::PATH, "model.safetensors")
        .u64(tag::BUDGET, 4096)
        .finish();
    let err = load::load_request(&payload, never()).unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    assert!(err.contains("capacity exceeded"), "{err}");
    assert_eq!(session::session_count(), 0);
}

#[test]
fn nonfinite_and_capacity_prefixes_survive_the_gusset_boundary() {
    let err = crate::argmax(&[f32::NAN]).unwrap_err();
    assert!(err.starts_with("ojas:E_NONFINITE:"), "{err}");
    assert!(err.contains("non-finite"), "{err}");
    assert_eq!(err.matches("ojas:E_").count(), 1, "{err}");

    let (_g, _dir) = fresh();
    for _ in 0..SESSION_CAP {
        load_path("model.safetensors").unwrap();
    }
    let err = call(OP_LOAD, &load_payload(0, 1, "model.safetensors")).unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    assert!(err.contains("capacity exceeded"), "{err}");
    // The boundary adds nothing: one prefix, set where the error was made.
    assert_eq!(err.matches("ojas:E_").count(), 1, "{err}");
    session::reset_sessions();
}

/// F10: the kind comes from the typed error, never from text. A missing
/// file whose name spells a kind, or its in-band prefix, is still an
/// unkinded load error.
#[test]
fn a_user_path_never_selects_an_error_kind() {
    let (_g, _dir) = fresh();
    for name in [
        "busy_model.safetensors",
        "capacity exceeded.safetensors",
        "x capacity: y.safetensors",
        "non-finite.safetensors",
        "device lost.safetensors",
        "dead device.safetensors",
        "runtime poisoned.safetensors",
        "poisoned.safetensors",
        "ojas:E_BUSY: x.safetensors",
        "ojas:E_CAPACITY: x.safetensors",
        "ojas:E_POISONED: x.safetensors",
        "ojas:E_PRESSURE: x.safetensors",
        "memory pressure.safetensors",
    ] {
        let err = call(OP_LOAD, &load_payload(0, 1, name)).unwrap_err();
        assert!(!err.starts_with("ojas:E_"), "{name}: {err}");
        assert!(err.contains("missing file"), "{name}: {err}");
        assert!(err.contains(name), "{name}: {err}");
    }
}

/// Every kind has its own prefix, and `go/ffi.go`'s `inBandKinds` lists
/// each one, so a kind added here cannot reach Go as a plain error. The
/// match is exhaustive: a new variant does not compile until it is listed.
#[test]
fn every_error_kind_has_a_distinct_prefix_that_go_decodes() {
    use crate::ErrorKind;
    const ALL: [ErrorKind; 6] = [
        ErrorKind::Capacity,
        ErrorKind::DeviceLost,
        ErrorKind::NonFinite,
        ErrorKind::Busy,
        ErrorKind::Poisoned,
        ErrorKind::Pressure,
    ];
    for (i, kind) in ALL.into_iter().enumerate() {
        let at = match kind {
            ErrorKind::Capacity => 0,
            ErrorKind::DeviceLost => 1,
            ErrorKind::NonFinite => 2,
            ErrorKind::Busy => 3,
            ErrorKind::Poisoned => 4,
            ErrorKind::Pressure => 5,
        };
        assert_eq!(at, i, "{kind:?} is out of place");
    }
    let go = include_str!("../../go/ffi.go");
    for kind in ALL {
        let prefix = kind.prefix();
        assert!(
            prefix.starts_with("ojas:E_") && prefix.ends_with(": "),
            "{prefix:?}"
        );
        let quoted = format!("{{\"{}\",", prefix.trim_end());
        assert!(go.contains(&quoted), "go/ffi.go does not decode {prefix:?}");
        for other in ALL {
            if other != kind {
                assert!(
                    !prefix.starts_with(other.prefix().trim_end()),
                    "{kind:?} {other:?}"
                );
            }
        }
    }
}

#[test]
fn error_kinds_follow_the_ojas_error_variant() {
    use crate::{kind_of, ojas_error, ErrorKind};
    use ojas_core::{BackendId, OjasError};
    let cases = [
        (
            OjasError::CapacityExceeded {
                requested: 2,
                cap: 1,
                live: 0,
            },
            Some(ErrorKind::Capacity),
        ),
        (OjasError::NonFinite { op: "x" }, Some(ErrorKind::NonFinite)),
        (OjasError::Poisoned, Some(ErrorKind::Poisoned)),
        (
            OjasError::Backend {
                id: BackendId::Wgpu,
                detail: "device lost (Destroyed): gone".into(),
            },
            Some(ErrorKind::DeviceLost),
        ),
        (
            OjasError::Backend {
                id: BackendId::Metal,
                detail: "runtime poisoned after adamw".into(),
            },
            Some(ErrorKind::DeviceLost),
        ),
        (
            OjasError::Backend {
                id: BackendId::Metal,
                detail: "kernel compile failed".into(),
            },
            None,
        ),
        // Text that names a kind inside another variant selects nothing.
        (
            OjasError::Shape {
                op: "load",
                detail: "busy_model.safetensors: capacity exceeded, non-finite, poisoned".into(),
            },
            None,
        ),
        (
            OjasError::OutOfRange {
                op: "load",
                detail: "device lost".into(),
            },
            None,
        ),
    ];
    for (err, want) in cases {
        assert_eq!(kind_of(&err), want, "{err}");
        let text = ojas_error("ctx", &err);
        match want {
            Some(kind) => assert_eq!(text, format!("{}ctx: {err}", kind.prefix())),
            None => assert_eq!(text, format!("ctx: {err}")),
        }
    }
}

#[test]
fn a_successful_root_clears_last_error_and_a_full_take_does_not_leave_it() {
    let _g = guard();
    session::set_last_error("stale message");
    let dir = scratch();
    assert!(
        session::last_error().is_empty(),
        "{}",
        session::last_error()
    );
    let _ = dir;

    session::set_last_error("hello");
    let mut short = [0u8; 2];
    let n = session::take_last_error(&mut short);
    assert_eq!(n, 5);
    assert_eq!(&short, b"he");
    assert_eq!(session::last_error(), "hello");

    let mut full = [0u8; 8];
    let n = session::take_last_error(&mut full);
    assert_eq!(n, 5);
    assert_eq!(&full[..5], b"hello");
    assert!(session::last_error().is_empty());

    session::set_last_error("kept");
    assert_eq!(session::take_last_error(&mut []), 4);
    assert_eq!(session::last_error(), "kept");
    session::clear_last_error();
}

// ---- device sessions ----

/// Load a wgpu session, or `None` when this machine has no wgpu adapter. In
/// that case the load must be the open's error and must leave no session, so
/// no CPU session stands in for the device.
pub(crate) fn load_wgpu_or_skip(path: &str) -> Option<session::Session> {
    match load_device(load::DEVICE_WGPU, 1, path) {
        Ok(session) => {
            assert_eq!(session.device, DeviceKind::Wgpu, "{session:?}");
            Some(session)
        }
        Err(err) => {
            assert!(
                ojas_wgpu::WgpuContext::open().is_err(),
                "a wgpu adapter opens but the load failed: {err}"
            );
            assert!(err.starts_with("wgpu:"), "{err}");
            let err = call(OP_LOAD, &load_payload(load::DEVICE_WGPU, 1, path)).unwrap_err();
            assert!(err.starts_with("wgpu:"), "{err}");
            assert_eq!(session::session_count(), 0);
            eprintln!("no wgpu adapter; checked that the load fails closed: {err}");
            None
        }
    }
}

fn relative_gap(a: f32, b: f32) -> f32 {
    (a - b).abs() / b.abs().max(f32::MIN_POSITIVE)
}

/// `(calls, bytes)` of device readbacks made by `f`. A step on `CpuBackend`
/// reads nothing back from a device, so a device session that fell back to
/// the CPU reports `(0, 0)` here.
fn readbacks_during<T>(f: impl FnOnce() -> T) -> (T, (u64, u64)) {
    let (calls, bytes) = ojas_core::device_readbacks();
    let out = f();
    let (calls_after, bytes_after) = ojas_core::device_readbacks();
    (out, (calls_after - calls, bytes_after - bytes))
}

/// A device session trains on its device: each step matches the same step
/// of a CPU session within 1e-4 relative while reading back only the 4-byte
/// loss (G10). A NaN weight is `E_NONFINITE`, reported by the step that hit
/// it; the refused step leaves no deferred fault behind, so Save on the same
/// session, which syncs, succeeds.
fn assert_device_steps_match_cpu(device: session::Session, dir: &Path) {
    write_bin(dir, "tokens.bin");
    let cpu = load_path("model.safetensors").unwrap();
    train_open(cpu.id, "tokens.bin").unwrap();
    let (opened, reads) = readbacks_during(|| train_open(device.id, "tokens.bin"));
    opened.unwrap();
    // Trainer::new on a device session downloads each resident weight once.
    assert_eq!(reads.0, u64::from(device.tensors), "{reads:?}");
    for n in 0..3 {
        let (on_cpu, cpu_reads) =
            readbacks_during(|| call(crate::OP_TRAIN_STEP, &sampled_payload(cpu.id)));
        let (on_dev, dev_reads) =
            readbacks_during(|| call(crate::OP_TRAIN_STEP, &sampled_payload(device.id)));
        let on_cpu = step_out(&on_cpu.unwrap());
        let on_dev = step_out(&on_dev.unwrap());
        assert_eq!(cpu_reads, (0, 0), "step {n}: the CPU step read a device");
        assert_eq!(
            dev_reads,
            (1, 4),
            "step {n}: the device step must read back the loss and nothing else"
        );
        assert!(
            relative_gap(on_dev.loss, on_cpu.loss) <= 1e-4,
            "step {n}: loss {} vs cpu {}",
            on_dev.loss,
            on_cpu.loss
        );
        assert!(
            relative_gap(on_dev.grad_norm, on_cpu.grad_norm) <= 1e-4,
            "step {n}: grad norm {} vs cpu {}",
            on_dev.grad_norm,
            on_cpu.grad_norm
        );
        assert_eq!(on_dev.matrix_lr, on_cpu.matrix_lr);
        assert_eq!(on_dev.step, on_cpu.step);
    }

    std::fs::write(
        dir.join("nan.safetensors"),
        model_bytes(Some("blocks.0.norm1.weight")),
    )
    .unwrap();
    let nan = load_device(
        match device.device {
            DeviceKind::Metal => load::DEVICE_METAL,
            _ => load::DEVICE_WGPU,
        },
        1,
        "nan.safetensors",
    )
    .unwrap();
    train_open(nan.id, "tokens.bin").unwrap();
    let err = call(crate::OP_TRAIN_STEP, &sampled_payload(nan.id)).unwrap_err();
    assert!(err.starts_with("ojas:E_NONFINITE:"), "{err}");
    let mut save = nan.id.to_le_bytes().to_vec();
    save.extend_from_slice(b"nan-ckpt");
    call(crate::OP_SAVE, &save).unwrap();
    for id in [device.id, cpu.id, nan.id] {
        session::try_free(id).unwrap();
    }
    assert_eq!(session::session_count(), 0);
}

/// Sampling on a device session matches the CPU session token for token and
/// reads back one `[vocab]` logit row per forward. Caller-logits argmax
/// reads nothing from the device.
fn assert_device_samples_match_cpu(device: session::Session) {
    let cpu = load_path("model.safetensors").unwrap();
    let vocab = nano_spec().vocab as u64;
    for prompt in [&[0u32][..], &[1], &[0, 1, 1], &[1, 0, 0, 1, 0]] {
        let want = call(crate::OP_SAMPLE, &sample_payload(cpu.id, prompt, 0.0, 3, 0)).unwrap();
        let (got, reads) = readbacks_during(|| {
            call(
                crate::OP_SAMPLE,
                &sample_payload(device.id, prompt, 0.0, 3, 0),
            )
        });
        assert_eq!(got.unwrap(), want, "{prompt:?}");
        assert_eq!(reads, (3, 3 * vocab * 4), "{prompt:?}");
    }
    let (token, reads) =
        readbacks_during(|| call(crate::OP_GENERATE, &argmax_payload(device.id, &[0.0, 2.0])));
    assert_eq!(token.unwrap(), 1u32.to_le_bytes());
    assert_eq!(reads, (0, 0));
    let err = call(crate::OP_SAMPLE, &greedy_payload(device.id, &[64])).unwrap_err();
    assert!(err.contains("range") || err.contains("vocab"), "{err}");
    let err = call(crate::OP_SAMPLE, &greedy_payload(device.id, &[])).unwrap_err();
    assert!(err.contains("prompt is empty"), "{err}");
    session::try_free(device.id).unwrap();
    session::try_free(cpu.id).unwrap();
}

/// A device load checks the file before opening the device, and a cancel
/// while the device opens returns the cancel and leaves no session.
fn assert_device_load_refuses_a_bad_file_and_honors_cancel(device: u32) {
    let err = load_device(device, 1, "missing.safetensors").unwrap_err();
    assert!(err.contains("missing file"), "{err}");
    let polls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&polls);
    // The first poll is the load's own entry check; cancel at the second,
    // which is the wait for the device open.
    let err = load::load_request(
        &load_payload(device, 1, "model.safetensors"),
        Box::new(move || {
            if counter.fetch_add(1, Ordering::Relaxed) == 0 {
                Ok(())
            } else {
                Err("cancelled: Explicit".to_string())
            }
        }),
    )
    .unwrap_err();
    assert_eq!(err, "cancelled: Explicit");
    assert_eq!(session::session_count(), 0);
}

/// A call on a deferring backend that returns early (here a cancel after
/// some forward ops ran on a NaN weight) must not leave that fault for the
/// session's next call. The cancelled call reports the fault, which is the
/// earliest error in recording order, and the next call, which syncs, is
/// clean.
fn assert_an_early_return_does_not_leak(device: u32, dir: &Path) {
    std::fs::write(
        dir.join("nan.safetensors"),
        model_bytes(Some("blocks.0.norm1.weight")),
    )
    .unwrap();
    write_bin(dir, "tokens.bin");
    let s = load_device(device, 1, "nan.safetensors").unwrap();
    let payload = sample_payload(s.id, &[1, 2, 3], 0.0, 1, 0);
    let polls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&polls);
    // Count the polls of an uncancelled call; it fails at its sync.
    let whole = crate::generate::sample_request(
        &payload,
        Box::new(move || {
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }),
    )
    .unwrap_err();
    assert!(whole.starts_with("ojas:E_NONFINITE:"), "{whole}");
    let cut = polls.load(Ordering::Relaxed) / 2;
    let seen = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&seen);
    let cancelled = crate::generate::sample_request(
        &payload,
        Box::new(move || {
            if counter.fetch_add(1, Ordering::Relaxed) + 1 == cut {
                Err("cancelled: Explicit".to_string())
            } else {
                Ok(())
            }
        }),
    )
    .unwrap_err();
    assert!(
        cancelled.starts_with("ojas:E_NONFINITE:"),
        "the cancelled call must report its deferred fault: {cancelled}"
    );
    // TrainOpen runs no op on the NaN weight and ends with a sync.
    let next = train_open(s.id, "tokens.bin");
    assert!(
        next.is_ok(),
        "the next call inherited the previous call's fault: {next:?}"
    );
    session::try_free(s.id).unwrap();
}

#[test]
fn wgpu_session_trains_on_the_device_and_matches_a_cpu_session() {
    let (_g, dir) = fresh();
    if let Some(wgpu) = load_wgpu_or_skip("model.safetensors") {
        assert_device_steps_match_cpu(wgpu, &dir);
    }
}

#[test]
fn wgpu_session_samples_on_the_device_and_matches_cpu() {
    let (_g, _dir) = fresh();
    if let Some(wgpu) = load_wgpu_or_skip("model.safetensors") {
        assert_device_samples_match_cpu(wgpu);
    }
}

#[test]
fn wgpu_load_refuses_a_bad_file_and_honors_cancel() {
    let (_g, _dir) = fresh();
    if load_wgpu_or_skip("model.safetensors").is_some() {
        session::clear_sessions();
        assert_device_load_refuses_a_bad_file_and_honors_cancel(load::DEVICE_WGPU);
    }
}

#[test]
fn an_early_return_never_leaks_a_deferred_fault_into_the_next_call() {
    let (_g, dir) = fresh();
    if load_wgpu_or_skip("model.safetensors").is_some() {
        session::clear_sessions();
        assert_an_early_return_does_not_leak(load::DEVICE_WGPU, &dir);
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn metal_without_macos_is_an_error_not_a_cpu_session() {
    let (_g, _dir) = fresh();
    let err = load_device(load::DEVICE_METAL, 1, "model.safetensors").unwrap_err();
    assert!(err.starts_with("metal:"), "{err}");
    assert_eq!(session::session_count(), 0);
}

/// Set to `1` to skip a test whose GPU does not open instead of failing it,
/// as in `ojas-model` and `ojas-infer`. CI sets it on the hosted macOS
/// runner, whose GPU has no Metal 4 command queue.
#[cfg(target_os = "macos")]
const ALLOW_NO_GPU: &str = "OJAS_ALLOW_NO_GPU";

/// A device that did not open: a `SKIP` line under `OJAS_ALLOW_NO_GPU=1`,
/// a failure otherwise.
#[cfg(target_os = "macos")]
pub(crate) fn skip_or_fail(what: &str, err: &str) {
    if std::env::var(ALLOW_NO_GPU).as_deref() == Ok("1") {
        eprintln!("SKIP ({ALLOW_NO_GPU}=1): {what}: {err}");
    } else {
        panic!("{what} failed: {err}. Set {ALLOW_NO_GPU}=1 to skip explicitly");
    }
}

/// Load a Metal session. When the device does not open, the load must be
/// the open's "metal:" error and leave no session, so no CPU session stands
/// in for Metal; then [`skip_or_fail`] decides.
#[cfg(target_os = "macos")]
fn load_metal_or_skip(path: &str) -> Option<session::Session> {
    match load_device(load::DEVICE_METAL, 1, path) {
        Ok(metal) => {
            assert_eq!(metal.device, DeviceKind::Metal, "{metal:?}");
            Some(metal)
        }
        Err(err) => {
            assert!(err.starts_with("metal:"), "{err}");
            assert_eq!(session::session_count(), 0);
            skip_or_fail("Metal load", &err);
            None
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_session_trains_on_the_device_and_matches_a_cpu_session() {
    let (_g, dir) = fresh();
    if let Some(metal) = load_metal_or_skip("model.safetensors") {
        assert_device_steps_match_cpu(metal, &dir);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_session_samples_on_the_device_and_matches_cpu() {
    let (_g, _dir) = fresh();
    if let Some(metal) = load_metal_or_skip("model.safetensors") {
        assert_device_samples_match_cpu(metal);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_load_refuses_a_bad_file_and_honors_cancel() {
    let (_g, _dir) = fresh();
    assert_device_load_refuses_a_bad_file_and_honors_cancel(load::DEVICE_METAL);
}

/// The same on Metal, which defers non-finite faults as wgpu does
/// (`docs/metal-deferred-faults.md`).
#[cfg(target_os = "macos")]
#[test]
fn an_early_return_never_leaks_a_deferred_metal_fault_into_the_next_call() {
    let (_g, dir) = fresh();
    if load_metal_or_skip("model.safetensors").is_some() {
        session::clear_sessions();
        assert_an_early_return_does_not_leak(load::DEVICE_METAL, &dir);
    }
}

/// A flag the test keeps: cancelling through the job's own context reaches
/// the gate mid-step, and the step commits nothing.
#[test]
fn a_job_context_cancel_reaches_the_model_ops() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let s = load_exact("model.safetensors").unwrap();
    train_open(s.id, "tokens.bin").unwrap();
    let flag = Arc::new(AtomicBool::new(false));
    let ctx = engine::context_with(crate::OP_TRAIN_STEP, Arc::clone(&flag));
    let check = engine::cancel_check(&ctx);
    let polls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&polls);
    let mut inner = check;
    let err = train::step_request(
        &sampled_payload(s.id),
        Box::new(move || {
            if counter.fetch_add(1, Ordering::Relaxed) == 40 {
                flag.store(true, Ordering::SeqCst);
            }
            inner()
        }),
    )
    .unwrap_err();
    assert_eq!(err, "cancelled: Explicit");
    assert!(polls.load(Ordering::Relaxed) > 40);
    assert_eq!(step(s.id).unwrap().step, 1);
}

#[test]
fn system_profile_routes_through_dispatch_and_reads_only() {
    let _g = guard();
    let before = crate::memory_ceiling().unwrap();
    let out = call(crate::OP_SYSTEM_PROFILE, &[]).unwrap();
    assert_eq!(u32::from_le_bytes(out[0..4].try_into().unwrap()), 1);
    let count = u32::from_le_bytes(out[4..8].try_into().unwrap()) as usize;
    assert_eq!(out.len(), 8 + 9 * count);
    let budgeted = call(crate::OP_SYSTEM_PROFILE, &(1u64 << 20).to_le_bytes()).unwrap();
    assert_eq!(budgeted[8], 1, "the budget entry is always known");
    let budget = u64::from_le_bytes(budgeted[9..17].try_into().unwrap());
    assert!(budget <= 1 << 20);
    assert!(call(crate::OP_SYSTEM_PROFILE, &[0u8; 3]).is_err());
    assert_eq!(
        crate::memory_ceiling().unwrap(),
        before,
        "a profile changes nothing"
    );
}
