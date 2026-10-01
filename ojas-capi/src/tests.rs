use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;

use crate::engine::{self, OP_LOAD};
use crate::generate::{self, GenerateBody, GEN_GREEDY, GEN_LOGITS};
use crate::load::{self, INLINE_PATH_MAX};
use crate::session::{self, SESSION_CAP};
use crate::step::{self, StepInput, MODE_LOGITS, MODE_TOKENS};

fn guard() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|err| err.into_inner())
}

fn scratch() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ojas-capi-{}-{}",
        std::process::id(),
        unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    session::set_model_root(dir.to_str().unwrap()).unwrap();
    dir
}

fn unique() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn write_tensor(path: &Path) {
    let header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut file = File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    file.write_all(header).unwrap();
    file.write_all(&[0, 0, 0, 0]).unwrap();
}

fn fresh() -> (MutexGuard<'static, ()>, std::path::PathBuf) {
    let guard = guard();
    session::clear_sessions();
    let dir = scratch();
    (guard, dir)
}

#[test]
fn missing_file_and_path_escapes_are_rejected() {
    let (_g, dir) = fresh();
    let err = load::load_path("no-such.safetensors").unwrap_err();
    assert!(err.contains("missing file"), "{err}");
    for raw in ["../outside.safetensors", "sub/../../etc/passwd", "/etc/passwd"] {
        let err = load::load_path(raw).unwrap_err();
        assert!(
            err.contains("..") || err.contains("relative") || err.contains("escapes"),
            "{raw}: {err}"
        );
    }
    let long = "a".repeat(INLINE_PATH_MAX + 1);
    let err = load::load_path(&long).unwrap_err();
    assert!(err.contains("4096"), "{err}");

    let outside = dir.join("outside-target.safetensors");
    write_tensor(&outside);
    // Move the file out of the root, then point a name inside the root at it.
    let escaped = std::env::temp_dir().join(format!("ojas-capi-out-{}", unique()));
    std::fs::rename(&outside, &escaped).unwrap();
    std::os::unix::fs::symlink(&escaped, dir.join("alias")).unwrap();
    let err = load::load_path("alias").unwrap_err();
    assert!(err.contains("escapes"), "{err}");
    let _ = std::fs::remove_file(escaped);
}

#[test]
fn load_counts_tensors_and_rejects_a_bad_header() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let session = load::load_path("model.safetensors").unwrap();
    assert_eq!(session.tensors, 1);
    assert_ne!(session.id, 0);
    let mut bad = File::create(dir.join("bad.safetensors")).unwrap();
    bad.write_all(&100u64.to_le_bytes()).unwrap();
    let err = load::load_path("bad.safetensors").unwrap_err();
    assert!(err.contains("safetensors"), "{err}");
}

#[test]
fn free_unknown_and_double_free_fail() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let id = load::load_path("model.safetensors").unwrap().id;
    session::try_free(id).unwrap();
    let err = session::try_free(id).unwrap_err();
    assert!(err.contains("unknown model"), "{err}");
    let err = session::try_free(99).unwrap_err();
    assert!(err.contains("unknown model"), "{err}");
}

#[test]
fn session_table_is_capped_at_64() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let mut ids = Vec::new();
    for _ in 0..SESSION_CAP {
        ids.push(load::load_path("model.safetensors").unwrap().id);
    }
    let err = load::load_path("model.safetensors").unwrap_err();
    assert!(err.contains("capacity exceeded"), "{err}");
    session::try_free(ids[0]).unwrap();
    let again = load::load_path("model.safetensors").unwrap().id;
    assert_ne!(again, ids[0]);
    session::try_free(again).unwrap();
    for id in ids.into_iter().skip(1) {
        session::try_free(id).unwrap();
    }
    assert_eq!(session::session_count(), 0);
}

#[test]
fn step_logits_match_cpu_cross_entropy_and_a_bad_shape_does_not_free() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let id = load::load_path("model.safetensors").unwrap().id;
    let logits = [0.0f32, 0.0];
    let targets = [0u32];
    let got = step::step(StepInput {
        mode: MODE_LOGITS,
        batch: 1,
        seq: 1,
        n_classes: 2,
        step: 0,
        lr: 1.0e-3,
        logits: &logits,
        tokens: &[],
        targets_u32: &targets,
        targets_u16: &[],
    })
    .unwrap();
    let budget = Budget::new(1 << 20);
    let cpu = CpuBackend::new(budget.clone());
    let logit_t = Tensor::from_f32(&logits, &[1, 2], &budget).unwrap();
    let target_t = Tensor::from_u32(&targets, &[1], &budget).unwrap();
    let expected = cpu
        .cross_entropy_mean_forward(&logit_t, &target_t, None)
        .unwrap()
        .to_f32_vec()
        .unwrap()[0];
    assert_eq!(got.loss.to_bits(), expected.to_bits());
    assert!(got.grad_norm.is_finite() && got.grad_norm > 0.0, "{got:?}");
    assert_eq!(got.lr, 1.0e-3);

    let other = step::step(StepInput {
        mode: MODE_LOGITS,
        batch: 1,
        seq: 1,
        n_classes: 2,
        step: 0,
        lr: 1.0e-3,
        logits: &[5.0, -1.0],
        tokens: &[],
        targets_u32: &[1],
        targets_u16: &[],
    })
    .unwrap();
    assert_ne!(got.loss.to_bits(), other.loss.to_bits());

    let err = step::step(StepInput {
        mode: MODE_LOGITS,
        batch: 1,
        seq: 1,
        n_classes: 2,
        step: 0,
        lr: 1.0e-3,
        logits: &[1.0],
        tokens: &[],
        targets_u32: &[0],
        targets_u16: &[],
    })
    .unwrap_err();
    assert!(err.contains("shape"), "{err}");
    session::require(id).unwrap();

    let nan = step::step(StepInput {
        mode: MODE_LOGITS,
        batch: 1,
        seq: 1,
        n_classes: 2,
        step: 0,
        lr: 1.0e-3,
        logits: &[f32::NAN, 0.0],
        tokens: &[],
        targets_u32: &[0],
        targets_u16: &[],
    })
    .unwrap_err();
    assert!(nan.contains("non-finite"), "{nan}");
    session::try_free(id).unwrap();
}

#[test]
fn step_tokens_run_linear_then_cross_entropy() {
    let _g = guard();
    let tokens = [3u16];
    let targets = [0u16];
    let got = step::step(StepInput {
        mode: MODE_TOKENS,
        batch: 1,
        seq: 1,
        n_classes: 2,
        step: 3,
        lr: 2.0e-3,
        logits: &[],
        tokens: &tokens,
        targets_u32: &[],
        targets_u16: &targets,
    })
    .unwrap();
    let budget = Budget::new(1 << 20);
    let cpu = CpuBackend::new(budget.clone());
    let x = Tensor::from_f32(&[3.0], &[1, 1], &budget).unwrap();
    let weight = Tensor::from_f32(&[1.0, 2.0], &[2, 1], &budget).unwrap();
    let logits = cpu.linear_forward(&x, &weight).unwrap();
    let target = Tensor::from_u32(&[0], &[1], &budget).unwrap();
    let expected = cpu
        .cross_entropy_mean_forward(&logits, &target, None)
        .unwrap()
        .to_f32_vec()
        .unwrap()[0];
    assert_eq!(got.loss.to_bits(), expected.to_bits());
    assert_eq!(got.lr, 2.0e-3);

    let moved = step::step(StepInput {
        mode: MODE_TOKENS,
        batch: 1,
        seq: 1,
        n_classes: 2,
        step: 3,
        lr: 2.0e-3,
        logits: &[],
        tokens: &[1],
        targets_u32: &[],
        targets_u16: &[0],
    })
    .unwrap();
    assert_ne!(got.loss.to_bits(), moved.loss.to_bits());
}

#[test]
fn header_only_step_is_a_shape_error() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let id = load::load_path("model.safetensors").unwrap().id;
    let mut payload = Vec::new();
    payload.extend_from_slice(&id.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&1u32.to_le_bytes());
    payload.extend_from_slice(&1u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    let ctx = engine::context_for(crate::OP_STEP, false);
    let err = engine::dispatch(&ctx, &payload).unwrap_err();
    assert!(err.contains("missing logits"), "{err}");
    session::require(id).unwrap();
    session::try_free(id).unwrap();
}

#[test]
fn generate_argmax_refuses_nan_and_greedy_reads_the_prompt() {
    let err = generate::generate(
        GEN_LOGITS,
        GenerateBody {
            logits: &[f32::NAN, f32::NAN],
            prompt: &[],
        },
    )
    .unwrap_err();
    assert!(err.contains("non-finite"), "{err}");
    assert!(err.contains("argmax_token"), "{err}");

    let token = generate::generate(
        GEN_LOGITS,
        GenerateBody {
            logits: &[0.1, 2.5, 0.2],
            prompt: &[],
        },
    )
    .unwrap();
    assert_eq!(token, 1);

    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let id = load::load_path("model.safetensors").unwrap().id;
    let mut payload = Vec::new();
    payload.extend_from_slice(&id.to_le_bytes());
    payload.extend_from_slice(&GEN_GREEDY.to_le_bytes());
    payload.extend_from_slice(&1u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    let ctx = engine::context_for(crate::OP_GENERATE, false);
    let out = engine::dispatch(&ctx, &payload).unwrap();
    assert_eq!(out.len(), 4);
    let token = u32::from_le_bytes(out.try_into().unwrap());
    assert!(token < 2, "{token}");

    payload.clear();
    payload.extend_from_slice(&id.to_le_bytes());
    payload.extend_from_slice(&GEN_GREEDY.to_le_bytes());
    payload.extend_from_slice(&1u32.to_le_bytes());
    payload.extend_from_slice(&7u32.to_le_bytes());
    let err = engine::dispatch(&ctx, &payload).unwrap_err();
    assert!(err.contains("out of range") || err.contains("OutOfRange") || err.contains("range"), "{err}");
    session::try_free(id).unwrap();
}

fn token_step(id: u64, n_classes: u32, tokens: &[u16], targets: &[u16]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&id.to_le_bytes());
    payload.extend_from_slice(&MODE_TOKENS.to_le_bytes());
    payload.extend_from_slice(&1u32.to_le_bytes());
    payload.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    payload.extend_from_slice(&n_classes.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&1.0e-3f32.to_le_bytes());
    for t in tokens.iter().chain(targets) {
        payload.extend_from_slice(&t.to_le_bytes());
    }
    payload
}

fn logits_step(id: u64, n_classes: u32, logits: &[f32], targets: &[u32]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&id.to_le_bytes());
    payload.extend_from_slice(&MODE_LOGITS.to_le_bytes());
    payload.extend_from_slice(&1u32.to_le_bytes());
    payload.extend_from_slice(&(targets.len() as u32).to_le_bytes());
    payload.extend_from_slice(&n_classes.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&1.0e-3f32.to_le_bytes());
    for v in logits {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    for t in targets {
        payload.extend_from_slice(&t.to_le_bytes());
    }
    payload
}

fn greedy_payload(id: u64, prompt: &[u32]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&id.to_le_bytes());
    payload.extend_from_slice(&GEN_GREEDY.to_le_bytes());
    payload.extend_from_slice(&(prompt.len() as u32).to_le_bytes());
    for t in prompt {
        payload.extend_from_slice(&t.to_le_bytes());
    }
    payload
}

fn argmax_payload(id: u64, logits: &[f32]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&id.to_le_bytes());
    payload.extend_from_slice(&GEN_LOGITS.to_le_bytes());
    payload.extend_from_slice(&(logits.len() as u32).to_le_bytes());
    for v in logits {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    payload
}

fn free_payload(id: u64) -> Vec<u8> {
    id.to_le_bytes().to_vec()
}

fn call(opcode: u32, payload: &[u8]) -> Result<Vec<u8>, String> {
    engine::dispatch(&engine::context_for(opcode, false), payload)
}

#[test]
fn a_poisoned_session_table_recovers_once_and_then_keeps_sessions() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let before = load::load_path("model.safetensors").unwrap().id;
    session::poison_table();
    // The first lock after the panic drops whatever the panic left behind.
    assert!(session::require(before).is_err());
    let after = load::load_path("model.safetensors").unwrap().id;
    assert!(after > before, "{after} <= {before}");
    session::require(after).unwrap();
    assert_eq!(session::session_count(), 1);
    session::try_free(after).unwrap();
}

#[test]
fn token_step_refuses_classes_no_u16_target_can_reach() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let id = load::load_path("model.safetensors").unwrap().id;
    let ok = call(crate::OP_STEP, &token_step(id, 1 << 16, &[3], &[0]));
    assert!(ok.is_ok(), "{ok:?}");
    for n_classes in [(1 << 16) + 1, 1 << 28, u32::MAX] {
        let err = call(crate::OP_STEP, &token_step(id, n_classes, &[3], &[0])).unwrap_err();
        assert!(err.contains("n_classes"), "{n_classes}: {err}");
    }
    session::try_free(id).unwrap();
}

#[test]
fn concurrent_open_step_generate_close_keeps_the_cap_and_never_reuses_an_id() {
    const THREADS: usize = 16;
    const ROUNDS: usize = 150;
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let issued = std::sync::Arc::new(Mutex::new(std::collections::HashSet::new()));
    let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let workers: Vec<_> = (0..THREADS)
        .map(|t| {
            let issued = std::sync::Arc::clone(&issued);
            let peak = std::sync::Arc::clone(&peak);
            std::thread::spawn(move || {
                let mut held = Vec::new();
                for round in 0..ROUNDS {
                    match call(OP_LOAD, b"model.safetensors") {
                        Ok(out) => {
                            let id = u64::from_le_bytes(out[..8].try_into().unwrap());
                            assert!(issued.lock().unwrap().insert(id), "id {id} reused");
                            held.push(id);
                        }
                        Err(err) => assert!(err.contains("capacity exceeded"), "{err}"),
                    }
                    peak.fetch_max(session::session_count(), std::sync::atomic::Ordering::Relaxed);
                    let Some(&id) = held.last() else { continue };
                    call(crate::OP_STEP, &logits_step(id, 2, &[0.0, 1.0], &[1])).unwrap();
                    call(crate::OP_STEP, &token_step(id, 2, &[3], &[0])).unwrap();
                    assert_eq!(call(crate::OP_GENERATE, &argmax_payload(id, &[0.0, 2.0])).unwrap(), 1u32.to_le_bytes());
                    call(crate::OP_GENERATE, &greedy_payload(id, &[0, 1])).unwrap();
                    if (round + t) % 3 != 0 {
                        let id = held.pop().unwrap();
                        call(crate::OP_FREE, &free_payload(id)).unwrap();
                        let err = call(crate::OP_FREE, &free_payload(id)).unwrap_err();
                        assert!(err.contains("unknown model"), "{err}");
                        let err = call(crate::OP_STEP, &logits_step(id, 2, &[0.0, 1.0], &[1])).unwrap_err();
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
    assert!(peak.load(std::sync::atomic::Ordering::Relaxed) <= SESSION_CAP);
    assert_eq!(session::session_count(), 0);
    assert!(issued.lock().unwrap().len() >= THREADS);
}

#[test]
fn racing_loads_fill_exactly_the_cap() {
    const THREADS: usize = SESSION_CAP * 2;
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
    let results: Vec<_> = (0..THREADS)
        .map(|_| {
            let barrier = std::sync::Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                load::load_path("model.safetensors")
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    let ids: std::collections::HashSet<u64> =
        results.iter().filter_map(|r| r.as_ref().ok().map(|s| s.id)).collect();
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
    write_tensor(&dir.join("model.safetensors"));
    let id = load::load_path("model.safetensors").unwrap().id;
    let no_panic = |opcode: u32, payload: &[u8]| -> Result<Vec<u8>, String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| call(opcode, payload)))
            .unwrap_or_else(|_| panic!("opcode {opcode} panicked on {payload:?}"))
    };

    let valid = [
        (crate::OP_STEP, logits_step(id, 2, &[0.0, 1.0, 2.0, 0.5], &[1, 0])),
        (crate::OP_STEP, token_step(id, 3, &[3, 1], &[0, 2])),
        (crate::OP_GENERATE, argmax_payload(id, &[0.0, 2.0, 1.0])),
        (crate::OP_GENERATE, greedy_payload(id, &[0, 1, 1])),
        (crate::OP_FREE, free_payload(u64::MAX)),
    ];
    let mut prefixes = 0usize;
    for (opcode, payload) in &valid {
        if *opcode != crate::OP_FREE {
            no_panic(*opcode, payload).unwrap();
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
    let header = |mode: u32, batch: u32, seq: u32, classes: u32| {
        let mut p = id.to_le_bytes().to_vec();
        for word in [mode, batch, seq, classes, 0] {
            p.extend_from_slice(&word.to_le_bytes());
        }
        p.extend_from_slice(&1.0e-3f32.to_le_bytes());
        p.extend_from_slice(&[0u8; 16]);
        p
    };
    for mode in [MODE_LOGITS, MODE_TOKENS, 0, 3, u32::MAX] {
        for (batch, seq, classes) in [
            (u32::MAX, u32::MAX, u32::MAX),
            (u32::MAX, 1, 2),
            (1, u32::MAX, 2),
            (1, 1, u32::MAX),
            (0, 1, 2),
            (1, 0, 2),
            (65_536, 65_536, 65_536),
        ] {
            no_panic(crate::OP_STEP, &header(mode, batch, seq, classes)).unwrap_err();
        }
    }
    for mode in [GEN_LOGITS, GEN_GREEDY, 0, u32::MAX] {
        let mut p = id.to_le_bytes().to_vec();
        p.extend_from_slice(&mode.to_le_bytes());
        p.extend_from_slice(&u32::MAX.to_le_bytes());
        p.extend_from_slice(&[0u8; 8]);
        no_panic(crate::OP_GENERATE, &p).unwrap_err();
    }
    let err = no_panic(crate::OP_STEP, &logits_step(id, 2, &[0.0, 0.0], &[5])).unwrap_err();
    assert!(!err.is_empty());
    no_panic(crate::OP_STEP, &token_step(id, 2, &[3], &[9])).unwrap_err();
    no_panic(crate::OP_GENERATE, &argmax_payload(id, &[])).unwrap_err();
    no_panic(crate::OP_GENERATE, &greedy_payload(id, &[])).unwrap_err();
    no_panic(crate::OP_GENERATE, &argmax_payload(id, &[f32::INFINITY, 0.0])).unwrap_err();
    no_panic(crate::OP_STEP, &logits_step(id, 2, &[f32::NAN, 0.0], &[0])).unwrap_err();
    for path in [&b"\xff\xfe"[..], b"a\0b", b"", b".", b"./", b"//", b"model.safetensors/"] {
        let got = no_panic(OP_LOAD, path);
        assert!(got.is_err(), "{:?} loaded: {got:?}", String::from_utf8_lossy(path));
    }

    // Deterministic xorshift so a failure reproduces.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    const RANDOM: usize = 20_000;
    for _ in 0..RANDOM {
        let opcode = [crate::OP_STEP, crate::OP_GENERATE, crate::OP_LOAD][(next() % 3) as usize];
        let len = (next() % 96) as usize;
        let mut payload: Vec<u8> = (0..len).map(|_| next() as u8).collect();
        if len >= 8 && next() % 2 == 0 {
            payload[..8].copy_from_slice(&id.to_le_bytes());
        }
        if len >= 12 {
            let mode = (next() % 4) as u32;
            payload[8..12].copy_from_slice(&mode.to_le_bytes());
        }
        if opcode == crate::OP_STEP && len >= 24 && next() % 2 == 0 {
            let classes = [0u32, 1, 2, 3, 65_536, u32::MAX][(next() % 6) as usize];
            payload[20..24].copy_from_slice(&classes.to_le_bytes());
        }
        let _ = no_panic(opcode, &payload);
    }
    session::reset_sessions();
}

#[test]
fn cancel_is_reported_before_the_handler_runs() {
    let _g = guard();
    let ctx = engine::context_for(OP_LOAD, true);
    let err = engine::dispatch(&ctx, b"model.safetensors").unwrap_err();
    assert_eq!(err, "cancelled: Explicit");
}

#[test]
fn opcode_zero_and_a_duplicate_opcode_are_refused() {
    let _g = guard();
    gusset::clear_engine_handlers();
    let err = gusset::register_engine(0, |_ctx, _input| Ok(Vec::<u8>::new())).unwrap_err();
    assert!(err.contains("opcode 0"), "{err}");
    gusset::register_engine(crate::OP_LOAD, |_ctx, _input| Ok(Vec::<u8>::new())).unwrap();
    let err = gusset::register_engine(crate::OP_LOAD, |_ctx, _input| Ok(Vec::<u8>::new()))
        .unwrap_err();
    assert!(err.contains("already registered"), "{err}");
    engine::install_engine().unwrap();
}
