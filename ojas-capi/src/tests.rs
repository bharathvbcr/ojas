use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;

use crate::engine::{self, OP_LOAD};
use crate::generate::{self, GenerateBody, GEN_GREEDY, GEN_LOGITS};
use crate::load::{self, INLINE_PATH_MAX};
use crate::session::{self, Compute, SESSION_CAP};

const CPU: Compute = Compute::Cpu { threads: 1 };
use crate::step::{self, StepInput, MODE_LOGITS, MODE_TOKENS};

fn guard() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|err| err.into_inner())
}

fn scratch() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ojas-capi-{}-{}", std::process::id(), unique()));
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
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
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
    for raw in [
        "../outside.safetensors",
        "sub/../../etc/passwd",
        "/etc/passwd",
    ] {
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
    let out = engine::bytes_of(engine::dispatch(&ctx, &payload).unwrap());
    assert_eq!(out.len(), 4);
    let token = u32::from_le_bytes(out.try_into().unwrap());
    assert!(token < 2, "{token}");

    payload.clear();
    payload.extend_from_slice(&id.to_le_bytes());
    payload.extend_from_slice(&GEN_GREEDY.to_le_bytes());
    payload.extend_from_slice(&1u32.to_le_bytes());
    payload.extend_from_slice(&7u32.to_le_bytes());
    let err = engine::dispatch(&ctx, &payload).unwrap_err();
    assert!(
        err.contains("out of range") || err.contains("OutOfRange") || err.contains("range"),
        "{err}"
    );
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
    engine::dispatch(&engine::context_for(opcode, false), payload).map(engine::bytes_of)
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
                    peak.fetch_max(
                        session::session_count(),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    let Some(&id) = held.last() else { continue };
                    call(crate::OP_STEP, &logits_step(id, 2, &[0.0, 1.0], &[1])).unwrap();
                    call(crate::OP_STEP, &token_step(id, 2, &[3], &[0])).unwrap();
                    assert_eq!(
                        call(crate::OP_GENERATE, &argmax_payload(id, &[0.0, 2.0])).unwrap(),
                        1u32.to_le_bytes()
                    );
                    call(crate::OP_GENERATE, &greedy_payload(id, &[0, 1])).unwrap();
                    if (round + t) % 3 != 0 {
                        let id = held.pop().unwrap();
                        call(crate::OP_FREE, &free_payload(id)).unwrap();
                        let err = call(crate::OP_FREE, &free_payload(id)).unwrap_err();
                        assert!(err.contains("unknown model"), "{err}");
                        let err = call(crate::OP_STEP, &logits_step(id, 2, &[0.0, 1.0], &[1]))
                            .unwrap_err();
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
    write_tensor(&dir.join("model.safetensors"));
    let id = load::load_path("model.safetensors").unwrap().id;
    let no_panic = |opcode: u32, payload: &[u8]| -> Result<Vec<u8>, String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| call(opcode, payload)))
            .unwrap_or_else(|_| panic!("opcode {opcode} panicked on {payload:?}"))
    };

    let valid = [
        (
            crate::OP_STEP,
            logits_step(id, 2, &[0.0, 1.0, 2.0, 0.5], &[1, 0]),
        ),
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
    no_panic(
        crate::OP_GENERATE,
        &argmax_payload(id, &[f32::INFINITY, 0.0]),
    )
    .unwrap_err();
    no_panic(crate::OP_STEP, &logits_step(id, 2, &[f32::NAN, 0.0], &[0])).unwrap_err();
    for path in [
        &b"\xff\xfe"[..],
        b"a\0b",
        b"",
        b".",
        b"./",
        b"//",
        b"model.safetensors/",
    ] {
        let got = no_panic(OP_LOAD, path);
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
    let err =
        gusset::register_engine(crate::OP_LOAD, |_ctx, _input| Ok(Vec::<u8>::new())).unwrap_err();
    assert!(err.contains("already registered"), "{err}");
    engine::install_engine().unwrap();
}

#[test]
fn small_results_stay_byte_vectors_before_the_allocator_api() {
    let out = engine::stage(vec![9, 8, 7]);
    match out {
        gusset::JobOutput::Bytes(bytes) => assert_eq!(bytes, vec![9, 8, 7]),
        other => panic!("small result left the pre-1.100 path: {other:?}"),
    }
}

fn device_payload(kind: u32, threads: u32, path: &str) -> Vec<u8> {
    let mut payload = b"OJDV".to_vec();
    payload.extend_from_slice(&kind.to_le_bytes());
    payload.extend_from_slice(&threads.to_le_bytes());
    payload.extend_from_slice(path.as_bytes());
    payload
}

fn load_device(kind: u32, threads: u32, path: &str) -> Result<session::Session, String> {
    load::load_request(&device_payload(kind, threads, path), || Ok(()))
}

#[test]
fn device_header_selects_cpu_parallel_and_bounds_its_thread_count() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let session = load_device(load::DEVICE_CPU_PARALLEL, 4, "model.safetensors").unwrap();
    assert!(
        matches!(session.compute, Compute::Cpu { threads: 4 }),
        "{session:?}"
    );
    session::try_free(session.id).unwrap();
    let session = load_device(
        load::DEVICE_CPU_PARALLEL,
        load::MAX_CPU_THREADS,
        "model.safetensors",
    )
    .unwrap();
    assert!(
        matches!(session.compute, Compute::Cpu { threads } if threads == load::MAX_CPU_THREADS as usize),
        "{session:?}"
    );
    session::try_free(session.id).unwrap();

    let err = load_device(9, 1, "model.safetensors").unwrap_err();
    assert!(err.contains("unknown device"), "{err}");
    let err = load_device(load::DEVICE_CPU_PARALLEL, 0, "model.safetensors").unwrap_err();
    assert!(err.contains("thread count is 0"), "{err}");
    for threads in [load::MAX_CPU_THREADS + 1, 1 << 20, u32::MAX] {
        let err = load_device(load::DEVICE_CPU_PARALLEL, threads, "model.safetensors")
            .expect_err("an unbounded thread count was accepted");
        assert!(err.contains("exceeds"), "{threads}: {err}");
    }
    for cut in 4..12 {
        let payload = device_payload(load::DEVICE_CPU, 1, "");
        let err = load::load_request(&payload[..cut], || Ok(())).unwrap_err();
        assert!(err.contains("short"), "{cut}: {err}");
    }
    assert_eq!(session::session_count(), 0);
}

/// Far above the load ceiling. The error is returned before a session, and
/// therefore before `CpuBackend::with_threads`, exists.
#[test]
fn cpu_parallel_load_refuses_a_thread_bomb() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let before = session::session_count();
    let err = load_device(load::DEVICE_CPU_PARALLEL, u32::MAX, "model.safetensors")
        .expect_err("a thread count above 256 was accepted");
    assert!(err.contains("exceeds"), "{err}");
    assert!(err.contains("256"), "{err}");
    assert_eq!(session::session_count(), before);
}

/// Load a wgpu session, or `None` when this machine has no wgpu adapter. In
/// that case the load must be the open's error and must leave no session, so
/// no CPU session stands in for the device.
fn load_wgpu_or_skip(path: &str) -> Option<session::Session> {
    match load_device(load::DEVICE_WGPU, 1, path) {
        Ok(session) => {
            assert!(matches!(session.compute, Compute::Wgpu(_)), "{session:?}");
            Some(session)
        }
        Err(err) => {
            assert!(
                ojas_wgpu::WgpuContext::open().is_err(),
                "a wgpu adapter opens but the load failed: {err}"
            );
            assert!(err.starts_with("wgpu:"), "{err}");
            let err = call(OP_LOAD, &device_payload(load::DEVICE_WGPU, 1, path)).unwrap_err();
            assert!(err.starts_with("wgpu:"), "{err}");
            assert_eq!(session::session_count(), 0);
            eprintln!("no wgpu adapter; checked that the load fails closed: {err}");
            None
        }
    }
}

/// A wgpu session holds a `WgpuBackend`, and a step on it matches the same
/// step on a `CpuBackend` session within 1e-4 relative while reading back
/// only the 4-byte loss. A step that ran on the CPU reads back nothing.
#[test]
fn wgpu_session_steps_on_the_device_and_matches_a_cpu_step() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let Some(wgpu) = load_wgpu_or_skip("model.safetensors") else {
        return;
    };
    assert_eq!(wgpu.tensors, 1);
    let cpu = load_device(load::DEVICE_CPU, 1, "model.safetensors").unwrap();

    let logits: Vec<f32> = (0..30)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.37)
        .collect();
    let targets = [0u32, 4, 2, 1, 3, 4];
    let cases = [
        (logits_step(0, 5, &logits, &targets), "logits"),
        (
            token_step(0, 6, &[3, 1, 4, 1, 5], &[5, 0, 2, 4, 1]),
            "tokens",
        ),
    ];
    for (payload, name) in cases {
        let with_id = |id: u64| {
            let mut p = payload.clone();
            p[..8].copy_from_slice(&id.to_le_bytes());
            p
        };
        let (on_cpu, cpu_reads) = readbacks_during(|| call(crate::OP_STEP, &with_id(cpu.id)));
        let (on_wgpu, wgpu_reads) = readbacks_during(|| call(crate::OP_STEP, &with_id(wgpu.id)));
        let on_cpu = stats_of(&on_cpu.unwrap());
        let on_wgpu = stats_of(&on_wgpu.unwrap());
        assert_eq!(cpu_reads, (0, 0), "{name}: the CPU step read a device");
        assert_eq!(
            wgpu_reads,
            (1, 4),
            "{name}: the wgpu step must read back the loss and nothing else"
        );
        assert!(
            relative_gap(on_wgpu[0], on_cpu[0]) <= 1e-4,
            "{name}: loss {} vs cpu {}",
            on_wgpu[0],
            on_cpu[0]
        );
        assert!(
            relative_gap(on_wgpu[1], on_cpu[1]) <= 1e-4,
            "{name}: grad norm {} vs cpu {}",
            on_wgpu[1],
            on_cpu[1]
        );
        assert_eq!(on_wgpu[2], on_cpu[2], "{name}: lr");
    }

    let nan = call(
        crate::OP_STEP,
        &logits_step(wgpu.id, 2, &[f32::NAN, 0.0], &[0]),
    )
    .unwrap_err();
    assert!(nan.contains("ojas:E_NONFINITE:"), "{nan}");
    let bad = call(crate::OP_STEP, &logits_step(wgpu.id, 2, &[0.0, 0.0], &[5])).unwrap_err();
    assert!(!bad.is_empty());
    // A refused step leaves no deferred fault behind for the next one.
    let after = call(crate::OP_STEP, &logits_step(wgpu.id, 2, &[0.0, 1.0], &[1])).unwrap();
    assert!(stats_of(&after)[0].is_finite());

    session::try_free(wgpu.id).unwrap();
    session::try_free(cpu.id).unwrap();
    assert_eq!(session::session_count(), 0);
}

/// Greedy generate on a wgpu session runs the embedding and linear on the
/// device and reads back the last row's two logits (8 bytes), matching the
/// CPU token. Caller-logits argmax reads nothing from the device.
#[test]
fn wgpu_session_generates_on_the_device_and_matches_cpu() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let Some(wgpu) = load_wgpu_or_skip("model.safetensors") else {
        return;
    };
    let cpu = load_device(load::DEVICE_CPU, 1, "model.safetensors").unwrap();
    for prompt in [&[0u32][..], &[1], &[0, 1, 1], &[1, 0, 0, 1, 0]] {
        let want = call(crate::OP_GENERATE, &greedy_payload(cpu.id, prompt)).unwrap();
        let (got, reads) =
            readbacks_during(|| call(crate::OP_GENERATE, &greedy_payload(wgpu.id, prompt)));
        assert_eq!(got.unwrap(), want, "{prompt:?}");
        assert_eq!(reads, (1, 8), "{prompt:?}");
    }
    let (token, reads) =
        readbacks_during(|| call(crate::OP_GENERATE, &argmax_payload(wgpu.id, &[0.0, 2.0])));
    assert_eq!(token.unwrap(), 1u32.to_le_bytes());
    assert_eq!(reads, (0, 0));
    let err = call(crate::OP_GENERATE, &greedy_payload(wgpu.id, &[7])).unwrap_err();
    assert!(err.contains("range") || err.contains("Range"), "{err}");
    let err = call(crate::OP_GENERATE, &greedy_payload(wgpu.id, &[])).unwrap_err();
    assert!(err.contains("prompt is empty"), "{err}");
    session::try_free(wgpu.id).unwrap();
    session::try_free(cpu.id).unwrap();
}

/// A wgpu load checks the file before opening the device, and a cancel
/// while the device opens returns the cancel and leaves no session.
#[test]
fn wgpu_load_refuses_a_bad_file_and_honors_cancel() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let err = load_device(load::DEVICE_WGPU, 1, "missing.safetensors").unwrap_err();
    assert!(err.contains("missing file"), "{err}");
    let err = load::load_request(
        &device_payload(load::DEVICE_WGPU, 1, "model.safetensors"),
        || Err("cancelled: Explicit".to_string()),
    )
    .unwrap_err();
    assert_eq!(err, "cancelled: Explicit");
    assert_eq!(session::session_count(), 0);
}

#[cfg(not(target_os = "macos"))]
#[test]
fn metal_without_macos_is_an_error_not_a_cpu_session() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let err = load_device(load::DEVICE_METAL, 1, "model.safetensors").unwrap_err();
    assert!(err.starts_with("metal:"), "{err}");
    assert_eq!(session::session_count(), 0);
}

fn relative_gap(a: f32, b: f32) -> f32 {
    (a - b).abs() / b.abs().max(f32::MIN_POSITIVE)
}

fn stats_of(out: &[u8]) -> [f32; 3] {
    assert_eq!(out.len(), 12, "{out:?}");
    [0, 4, 8].map(|i| f32::from_le_bytes(out[i..i + 4].try_into().unwrap()))
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

/// A Metal session holds a `MetalBackend`, and a step on it matches the same
/// step on a `CpuBackend` session within 1e-4 relative while reading back
/// only the 4-byte loss: the logits, targets, gradients and AdamW state stay
/// on the device.
#[cfg(target_os = "macos")]
#[test]
fn metal_session_steps_on_the_device_and_matches_a_cpu_step() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let metal = load_device(load::DEVICE_METAL, 1, "model.safetensors").expect("Metal device");
    assert!(matches!(metal.compute, Compute::Metal(_)), "{metal:?}");
    assert_eq!(metal.tensors, 1);
    let cpu = load_device(load::DEVICE_CPU, 1, "model.safetensors").unwrap();

    let logits: Vec<f32> = (0..30)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.37)
        .collect();
    let targets = [0u32, 4, 2, 1, 3, 4];
    let cases = [
        (logits_step(0, 5, &logits, &targets), "logits"),
        (
            token_step(0, 6, &[3, 1, 4, 1, 5], &[5, 0, 2, 4, 1]),
            "tokens",
        ),
    ];
    for (payload, name) in cases {
        let with_id = |id: u64| {
            let mut p = payload.clone();
            p[..8].copy_from_slice(&id.to_le_bytes());
            p
        };
        let (on_cpu, cpu_reads) = readbacks_during(|| call(crate::OP_STEP, &with_id(cpu.id)));
        let (on_metal, metal_reads) = readbacks_during(|| call(crate::OP_STEP, &with_id(metal.id)));
        let on_cpu = stats_of(&on_cpu.unwrap());
        let on_metal = stats_of(&on_metal.unwrap());
        assert_eq!(cpu_reads, (0, 0), "{name}: the CPU step read a device");
        assert_eq!(
            metal_reads,
            (1, 4),
            "{name}: the Metal step must read back the loss and nothing else"
        );
        assert!(
            relative_gap(on_metal[0], on_cpu[0]) <= 1e-4,
            "{name}: loss {} vs cpu {}",
            on_metal[0],
            on_cpu[0]
        );
        assert!(
            relative_gap(on_metal[1], on_cpu[1]) <= 1e-4,
            "{name}: grad norm {} vs cpu {}",
            on_metal[1],
            on_cpu[1]
        );
        assert_eq!(on_metal[2], on_cpu[2], "{name}: lr");
    }

    let nan = call(
        crate::OP_STEP,
        &logits_step(metal.id, 2, &[f32::NAN, 0.0], &[0]),
    )
    .unwrap_err();
    assert!(
        nan.contains("non-finite") || nan.contains("NonFinite"),
        "{nan}"
    );
    let bad = call(crate::OP_STEP, &logits_step(metal.id, 2, &[0.0, 0.0], &[5])).unwrap_err();
    assert!(!bad.is_empty());

    session::try_free(metal.id).unwrap();
    session::try_free(cpu.id).unwrap();
    assert_eq!(session::session_count(), 0);
}

/// Greedy generate on a Metal session runs the embedding and linear on the
/// device and reads back the last row's two logits (8 bytes), matching the
/// CPU token. Caller-logits argmax reads nothing from the device.
#[cfg(target_os = "macos")]
#[test]
fn metal_session_generates_on_the_device_and_matches_cpu() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let metal = load_device(load::DEVICE_METAL, 1, "model.safetensors").expect("Metal device");
    let cpu = load_device(load::DEVICE_CPU, 1, "model.safetensors").unwrap();
    for prompt in [&[0u32][..], &[1], &[0, 1, 1], &[1, 0, 0, 1, 0]] {
        let want = call(crate::OP_GENERATE, &greedy_payload(cpu.id, prompt)).unwrap();
        let (got, reads) =
            readbacks_during(|| call(crate::OP_GENERATE, &greedy_payload(metal.id, prompt)));
        assert_eq!(got.unwrap(), want, "{prompt:?}");
        assert_eq!(reads, (1, 8), "{prompt:?}");
    }
    let (token, reads) =
        readbacks_during(|| call(crate::OP_GENERATE, &argmax_payload(metal.id, &[0.0, 2.0])));
    assert_eq!(token.unwrap(), 1u32.to_le_bytes());
    assert_eq!(reads, (0, 0));
    let err = call(crate::OP_GENERATE, &greedy_payload(metal.id, &[7])).unwrap_err();
    assert!(err.contains("range") || err.contains("Range"), "{err}");
    let err = call(crate::OP_GENERATE, &greedy_payload(metal.id, &[])).unwrap_err();
    assert!(err.contains("prompt is empty"), "{err}");
    session::try_free(metal.id).unwrap();
    session::try_free(cpu.id).unwrap();
}

/// A Metal load checks the file before opening the device, and a cancel
/// while the device opens returns the cancel and leaves no session.
#[cfg(target_os = "macos")]
#[test]
fn metal_load_refuses_a_bad_file_and_honors_cancel() {
    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    let err = load_device(load::DEVICE_METAL, 1, "missing.safetensors").unwrap_err();
    assert!(err.contains("missing file"), "{err}");
    let err = load::load_request(
        &device_payload(load::DEVICE_METAL, 1, "model.safetensors"),
        || Err("cancelled: Explicit".to_string()),
    )
    .unwrap_err();
    assert_eq!(err, "cancelled: Explicit");
    assert_eq!(session::session_count(), 0);
}

#[test]
fn rows_times_classes_does_not_wrap() {
    let err = step::step(StepInput {
        mode: MODE_LOGITS,
        batch: u32::MAX,
        seq: u32::MAX,
        n_classes: 2,
        step: 0,
        lr: 1.0e-3,
        logits: &[],
        tokens: &[],
        targets_u32: &[],
        targets_u16: &[],
    })
    .unwrap_err();
    assert!(err.contains("overflows"), "{err}");
}

#[test]
fn token_step_refuses_a_linear_output_the_process_budget_cannot_hold() {
    let rows = 5_000usize;
    let tokens = vec![0u16; rows];
    let targets = vec![0u16; rows];
    let err = step::step(StepInput {
        mode: MODE_TOKENS,
        batch: rows as u32,
        seq: 1,
        n_classes: 1 << 16,
        step: 0,
        lr: 1.0e-3,
        logits: &[],
        tokens: &tokens,
        targets_u32: &[],
        targets_u16: &targets,
    })
    .unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    assert!(err.contains("capacity exceeded"), "{err}");
}

/// Under a caller budget too small for the whole request, `allow_split`
/// false is `ojas:E_CAPACITY:` and `allow_split` true runs row-parts whose
/// combined loss and gradient norm match the unsplit step. A policy that
/// names a GPU is refused rather than run on the CPU.
#[test]
fn step_with_policy_splits_rows_or_refuses_capacity_and_gpu_devices() {
    let rows = 64usize;
    let n_classes = 32u32;
    let logits: Vec<f32> = (0..rows * n_classes as usize)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.21)
        .collect();
    let targets: Vec<u32> = (0..rows as u32).map(|r| (r * 5) % n_classes).collect();
    let input = || StepInput {
        mode: MODE_LOGITS,
        batch: rows as u32,
        seq: 1,
        n_classes,
        step: 0,
        lr: 1.0e-3,
        logits: &logits,
        tokens: &[],
        targets_u32: &targets,
        targets_u16: &[],
    };
    let whole = step::step(input()).unwrap();
    assert_eq!(whole.split, None);

    let mut policy = ojas_device::ResourcePolicy::new(10_000);
    let err = step::step_with_policy(input(), &policy, || Ok(())).unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");

    policy.allow_split = true;
    let parts = step::step_with_policy(input(), &policy, || Ok(())).unwrap();
    assert_eq!(parts.split, Some(3), "{parts:?}");
    assert!(
        relative_gap(parts.loss, whole.loss) <= 1e-5,
        "{parts:?} vs {whole:?}"
    );
    assert!(
        relative_gap(parts.grad_norm, whole.grad_norm) <= 1e-5,
        "{parts:?} vs {whole:?}"
    );

    for device in [
        ojas_device::Device::Metal,
        ojas_device::Device::Vulkan,
        ojas_device::Device::Cuda,
    ] {
        policy.devices = vec![device, ojas_device::Device::Cpu];
        let err = step::step_with_policy(input(), &policy, || Ok(())).unwrap_err();
        assert!(err.contains("refusing CPU fallback"), "{device:?}: {err}");
        assert!(err.contains(&format!("{device:?}")), "{err}");
    }
}

/// 1000 rows do not fit the caller budget as one part or as two equal parts.
/// `allow_split` false is capacity. `allow_split` true uses three unequal
/// parts, and a second run keeps the same `k` and the same bits.
#[test]
fn split_1000_rows_into_three_unequal_parts() {
    let n_classes = 4usize;
    let parts = step::row_parts(1000, 3).unwrap();
    assert_eq!(parts, vec![334, 333, 333]);
    let budget = step::part_limit_bytes(parts[0], n_classes).unwrap();
    assert!(step::part_limit_bytes(500, n_classes).unwrap() > budget);
    assert!(step::part_limit_bytes(1000, n_classes).unwrap() > budget);
    let logits: Vec<f32> = (0..1000 * n_classes)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.15)
        .collect();
    let targets: Vec<u32> = (0..1000u32).map(|r| r % n_classes as u32).collect();
    let input = || StepInput {
        mode: MODE_LOGITS,
        batch: 1000,
        seq: 1,
        n_classes: n_classes as u32,
        step: 2,
        lr: 1.0e-3,
        logits: &logits,
        tokens: &[],
        targets_u32: &targets,
        targets_u16: &[],
    };
    let mut policy = ojas_device::ResourcePolicy::new(budget);
    let err = step::step_with_policy(input(), &policy, || Ok(())).unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    assert!(err.contains("capacity exceeded"), "{err}");
    policy.allow_split = true;
    let first = step::step_with_policy(input(), &policy, || Ok(())).unwrap();
    let second = step::step_with_policy(input(), &policy, || Ok(())).unwrap();
    assert_eq!(first.split, Some(3));
    assert_eq!(second.split, Some(3));
    assert_eq!(first.loss.to_bits(), second.loss.to_bits());
    assert_eq!(first.grad_norm.to_bits(), second.grad_norm.to_bits());
    assert_eq!(first.lr.to_bits(), second.lr.to_bits());
}

/// Logits and targets fit, and so do the host copies `ojas-cpu` charges
/// for them. The cross-entropy scratch, which is at least 64 KiB, is the
/// reservation that misses. An allocator abort on an uncharged `std`
/// allocation (`format!` / channel / spawn) is accepted. This test does not
/// catch `abort`.
#[test]
fn step_returns_capacity_when_the_cross_entropy_scratch_does_not_fit() {
    let rows = 4096usize;
    let n_classes = 4usize;
    let logits_bytes = (rows * n_classes * 4) as u64;
    let target_bytes = (rows * 4) as u64;
    let scratch_elems = rows * n_classes + n_classes;
    let scratch_bytes = (scratch_elems * 4) as u64;
    assert!(scratch_bytes >= 64 * 1024, "{scratch_bytes}");
    // The step's logits and targets tensors, then the charged host copy of
    // each that cross_entropy_mean_forward reads (ojas-cpu f32_in, u32_in).
    let live_before_scratch = 2 * (logits_bytes + target_bytes);
    let budget = live_before_scratch + 32 * 1024;
    assert!(budget < live_before_scratch + scratch_bytes);
    let logits = vec![0.1f32; rows * n_classes];
    let targets = vec![0u32; rows];
    let err = step::step_with_policy(
        StepInput {
            mode: MODE_LOGITS,
            batch: rows as u32,
            seq: 1,
            n_classes: n_classes as u32,
            step: 0,
            lr: 1.0e-3,
            logits: &logits,
            tokens: &[],
            targets_u32: &targets,
            targets_u16: &[],
        },
        &ojas_device::ResourcePolicy::new(budget),
        || Ok(()),
    )
    .expect_err("the step allocated past a budget that cannot hold the scratch");
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    assert!(err.contains("capacity exceeded"), "{err}");
    assert!(
        err.contains(&format!("requested {scratch_bytes}")),
        "the failing reserve was not the {scratch_bytes}-byte scratch: {err}"
    );
    assert!(
        err.contains(&format!("live {live_before_scratch}")),
        "an earlier reserve failed, so this is not the later scratch miss: {err}"
    );
}

/// The embedding scratch is the reservation that misses a 1 MiB generate
/// budget. Before it, in order: the prompt-sized output charge `P`, the
/// 8-byte table, the id tensor `P`, then the host copies `embedding_forward`
/// charges, the table's 8 bytes and the ids' `P`. So the miss is the scratch
/// exactly when `3P + 16 <= cap < 4P + 16`; the prompt length is chosen for
/// that. Tiny `std` allocations may still abort; this test does not catch
/// `abort`.
#[test]
fn generate_returns_capacity_when_the_embedding_scratch_does_not_fit() {
    let cap = 1u64 << 20;
    let tokens = 80_000usize;
    let prompt = vec![0u32; tokens];
    let scratch_bytes = (tokens * 4) as u64;
    assert!(scratch_bytes >= 64 * 1024);
    assert!(3 * scratch_bytes + 16 <= cap, "an earlier charge would miss");
    assert!(cap < 4 * scratch_bytes + 16, "the scratch would fit");
    let err = generate::generate_checked(
        &CPU,
        GEN_GREEDY,
        GenerateBody {
            logits: &[],
            prompt: &prompt,
        },
        || Ok(()),
    )
    .expect_err("greedy generate allocated past its 1 MiB budget");
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    assert!(err.contains("capacity exceeded"), "{err}");
    assert!(
        err.contains(&format!("requested {scratch_bytes}")),
        "the failing reserve was not the embedding scratch: {err}"
    );
    let live_before = scratch_bytes + 8 + scratch_bytes + 8 + scratch_bytes;
    assert!(
        err.contains(&format!("live {live_before}")),
        "an earlier reserve failed, so this is not the later embedding miss: {err}"
    );
}

/// Smallest caller budget under which an unsplit token step succeeds.
fn min_token_step_budget(rows: u32, classes: u32, upper: u64) -> u64 {
    let tokens = vec![1u16; rows as usize];
    let targets = vec![0u16; rows as usize];
    let attempt = |budget: u64| {
        let policy = ojas_device::ResourcePolicy::new(budget);
        step::step_with_policy(
            StepInput {
                mode: MODE_TOKENS,
                batch: rows,
                seq: 1,
                n_classes: classes,
                step: 0,
                lr: 1.0e-3,
                logits: &[],
                tokens: &tokens,
                targets_u32: &[],
                targets_u16: &targets,
            },
            &policy,
            || Ok(()),
        )
    };
    let mut lo = 0u64;
    let mut hi = upper;
    let mut need = None;
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        match attempt(mid) {
            Ok(_) => {
                need = Some(mid);
                if mid == 0 {
                    break;
                }
                hi = mid - 1;
            }
            Err(err) if err.starts_with("ojas:E_CAPACITY:") => {
                lo = mid.saturating_add(1);
            }
            Err(err) => panic!("token step failed for another reason at {mid}: {err}"),
        }
    }
    need.expect("token step never fit")
}

/// Token mode used to reserve `rows * n_classes` f32s and then let the
/// linear charge that output again. The minimum budget is derived from the
/// charges `ojas-cpu` makes, input host copies included (its
/// `validate.rs`), and must be met exactly.
///
/// With `Y = rows * classes * 4`, `T = rows * 4`, `C = classes * 4`:
/// - cross-entropy forward/backward peak: logits `Y` and their copy `Y`,
///   targets `T` and their copy `T`, scratch `Y + C`, the loss 4:
///   `3Y + 2T + C + 4`;
/// - linear peak: `x` (`T`) and the class weight (`C`) and their copies,
///   the output `Y`, and the GEMM packing scratch of
///   `ojas-cpu/src/gemm.rs` `scratch(rows, 1, classes)` on one thread:
///   `(6 * ceil(rows / 6) + 16 * ceil(classes / 16)) * 4` bytes (MR 6,
///   NR 16, no tile buffer): `2T + 2C + Y + G`.
///
/// At 64 x 64 the cross-entropy peak binds. At 1 x 64 the linear peak binds
/// by 20 bytes, so a second charge of the output during the linear would
/// raise the minimum by `Y`.
#[test]
fn token_output_is_charged_once() {
    let peaks = |rows: u64, classes: u64| {
        let (y, t, c) = (rows * classes * 4, rows * 4, classes * 4);
        let gemm = (6 * rows.div_ceil(6) + 16 * classes.div_ceil(16)) * 4;
        (3 * y + 2 * t + c + 4, 2 * t + 2 * c + y + gemm, y)
    };

    let (ce, linear, _) = peaks(64, 64);
    assert!(ce > linear, "{ce} {linear}");
    assert_eq!(ce, 49_924);
    let need = min_token_step_budget(64, 64, ce + 4_096);
    assert_eq!(need, ce, "64x64: minimum budget is the cross-entropy peak");

    let (ce, linear, output) = peaks(1, 64);
    assert!(linear > ce, "{ce} {linear}");
    assert_eq!(linear, 1_056);
    let need = min_token_step_budget(1, 64, linear + 4_096);
    assert_eq!(
        need, linear,
        "1x64: minimum budget is the linear peak; {} would mean the output \
         ({output} bytes) is charged twice",
        linear + output
    );
}

/// `ojas-cpu` charges the linear output from before the GEMM until its copy
/// into the returned tensor exists, so while that copy runs both are charged
/// (`2Y`). The split planner's per-part bound `3Y + 2S + 256` must still admit
/// every step it plans. These shapes are output-dominated: `Y` is far larger
/// than the GEMM packing scratch, which is where the second charge binds.
#[test]
fn split_planner_bound_admits_output_dominated_token_steps() {
    for (rows, classes) in [(1u32, 50_304u32), (8, 4_096), (3, 65_536)] {
        let limit = step::part_limit_bytes(rows as usize, classes as usize).unwrap();
        let need = min_token_step_budget(rows, classes, limit + 4_096);
        assert!(
            need <= limit,
            "{rows}x{classes}: the step needs {need} bytes but the planner admits it at {limit}"
        );
    }
}

#[test]
fn step_cancel_runs_between_cross_entropy_forward_backward_and_clip() {
    let mut calls = 0u32;
    let err = step::step_checked(
        &CPU,
        StepInput {
            mode: MODE_LOGITS,
            batch: 1,
            seq: 1,
            n_classes: 2,
            step: 0,
            lr: 1.0e-3,
            logits: &[0.0, 1.0],
            tokens: &[],
            targets_u32: &[0],
            targets_u16: &[],
        },
        || {
            calls += 1;
            if calls == 4 {
                Err("cancelled: Explicit".to_string())
            } else {
                Ok(())
            }
        },
    )
    .unwrap_err();
    assert_eq!(err, "cancelled: Explicit");
    assert_eq!(calls, 4);

    calls = 0;
    step::step_checked(
        &CPU,
        StepInput {
            mode: MODE_LOGITS,
            batch: 1,
            seq: 1,
            n_classes: 2,
            step: 0,
            lr: 1.0e-3,
            logits: &[0.0, 1.0],
            tokens: &[],
            targets_u32: &[0],
            targets_u16: &[],
        },
        || {
            calls += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(calls, 6, "a check was dropped from the step path");
}

#[test]
fn greedy_honors_cancel_after_the_budget_charge() {
    let mut calls = 0u32;
    let err = generate::generate_checked(
        &CPU,
        GEN_GREEDY,
        GenerateBody {
            logits: &[],
            prompt: &[0],
        },
        || {
            calls += 1;
            if calls == 4 {
                Err("cancelled: Explicit".to_string())
            } else {
                Ok(())
            }
        },
    )
    .unwrap_err();
    assert_eq!(err, "cancelled: Explicit");
    assert_eq!(calls, 4);
}

#[test]
fn nonfinite_and_capacity_prefixes_survive_the_gusset_boundary() {
    let err = generate::generate(
        GEN_LOGITS,
        GenerateBody {
            logits: &[f32::NAN],
            prompt: &[],
        },
    )
    .unwrap_err();
    assert!(err.starts_with("ojas:E_NONFINITE:"), "{err}");
    assert!(err.contains("non-finite"), "{err}");

    assert_eq!(err.matches("ojas:E_").count(), 1, "{err}");

    let (_g, dir) = fresh();
    write_tensor(&dir.join("model.safetensors"));
    for _ in 0..SESSION_CAP {
        load::load_path("model.safetensors").unwrap();
    }
    let err = call(OP_LOAD, b"model.safetensors").unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    assert!(err.contains("capacity exceeded"), "{err}");
    // The boundary adds nothing: one prefix, set where the error was made.
    assert_eq!(err.matches("ojas:E_").count(), 1, "{err}");
    session::reset_sessions();
}

/// F10: the kind comes from the typed error, never from text. A missing
/// file whose name spells a kind is still an unkinded load error.
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
    ] {
        let err = call(OP_LOAD, name.as_bytes()).unwrap_err();
        assert!(!err.starts_with("ojas:E_"), "{name}: {err}");
        assert!(err.contains("missing file"), "{name}: {err}");
        assert!(err.contains(name), "{name}: {err}");
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
                detail: "busy_model.safetensors: capacity exceeded, non-finite".into(),
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
