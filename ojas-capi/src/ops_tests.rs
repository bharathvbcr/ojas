//! Success and every refusal kind of the ops added with the real model:
//! NEW, TRAIN_OPEN, TRAIN_STEP, SAVE, RESUME, TOKENIZER, TOKENIZE, SAMPLE.

use std::path::Path;

use ojas_core::{BackendId, OjasError};

use crate::load;
use crate::session;
use crate::tests::{
    call, fresh, load_exact, load_path, model_bytes, nano_spec, never, open_payload, sample,
    sample_payload, sampled_payload, step, step_out, tokens_payload, train_fields, train_open,
    write_bin, write_tensor, NANO_SEED, SEQ,
};
use crate::train;
use crate::wire::{tag, Writer};

fn spec_fields(w: Writer) -> Writer {
    let s = nano_spec();
    w.u32(tag::VOCAB, s.vocab as u32)
        .u32(tag::N_EMBD, s.n_embd as u32)
        .u32(tag::N_LAYER, s.n_layer as u32)
        .u32(tag::N_HEAD, s.n_head as u32)
        .u32(tag::N_KV_HEAD, s.n_kv_head as u32)
        .u32(tag::HEAD_DIM, s.head_dim as u32)
        .u32(tag::HIDDEN, s.hidden as u32)
        .u32(tag::MAX_SEQ, s.max_seq as u32)
        .f64(tag::ROPE_BASE, s.rope_base)
        .f64(tag::RMS_EPS, s.rms_eps)
}

fn new_model(seed: u64) -> Result<session::Session, String> {
    load::new_request(
        &spec_fields(Writer::default()).u64(tag::SEED, seed).finish(),
        never(),
    )
}

fn save(id: u64, dir: &str) -> Result<(), String> {
    let mut p = id.to_le_bytes().to_vec();
    p.extend_from_slice(dir.as_bytes());
    train::save_request(&p, never())
}

fn resume_fields(dir: &str, bin: &str) -> Writer {
    train_fields(Writer::default().str(tag::PATH, dir), bin).u32(tag::NUMERICS, 1)
}

fn resume(dir: &str, bin: &str) -> Result<session::Session, String> {
    train::resume_request(&resume_fields(dir, bin).finish(), never())
}

/// NEW is `init_params` of the spec and seed: it samples exactly as LOAD of
/// the fixture made from the same seed, and a different seed differs.
#[test]
fn new_is_a_fresh_init_of_the_spec_and_seed() {
    let (_g, _dir) = fresh();
    let fresh_model = new_model(NANO_SEED).unwrap();
    assert_eq!(
        fresh_model.tensors,
        ojas_model::param_count(&nano_spec()).unwrap() as u32
    );
    assert_eq!(fresh_model.path, None);
    let loaded = load_path("model.safetensors").unwrap();
    let other = new_model(NANO_SEED + 1).unwrap();
    let prompt = [3u32, 1, 4, 1, 5];
    let a = sample(fresh_model.id, &prompt, 0.9, 8, 42).unwrap();
    assert_eq!(a, sample(loaded.id, &prompt, 0.9, 8, 42).unwrap());
    assert_eq!(a.len(), 8);
    let drawn = |id| sample(id, &prompt, 1.0, 16, 42).unwrap();
    assert_eq!(drawn(fresh_model.id), drawn(loaded.id));
    assert_ne!(
        drawn(fresh_model.id),
        drawn(other.id),
        "a different init seed drew the same tokens"
    );
}

#[test]
fn new_refuses_a_bad_spec_and_an_unknown_or_missing_field() {
    let (_g, _dir) = fresh();
    let cases: [(Vec<u8>, &str); 5] = [
        (spec_fields(Writer::default()).finish(), "seed is required"),
        (
            Writer::default().u64(tag::SEED, 1).finish(),
            "vocab is required",
        ),
        (
            spec_fields(Writer::default())
                .u64(tag::SEED, 1)
                .u32(tag::TOP_K, 1)
                .finish(),
            "unknown option field 51",
        ),
        (
            spec_fields(Writer::default())
                .u64(tag::SEED, 1)
                .u32(tag::HEAD_DIM, 7)
                .finish(),
            "repeats",
        ),
        (
            spec_fields(Writer::default().u32(tag::MAX_SEQ, 0))
                .u64(tag::SEED, 1)
                .finish(),
            "repeats",
        ),
    ];
    for (payload, want) in cases {
        let err = load::new_request(&payload, never()).unwrap_err();
        assert!(err.contains(want), "{want}: {err}");
    }
    // An odd head_dim cannot be half-split for RoPE.
    let mut w = Writer::default();
    let s = nano_spec();
    for (t, v) in [
        (tag::VOCAB, s.vocab as u32),
        (tag::N_EMBD, 14),
        (tag::N_LAYER, 2),
        (tag::N_HEAD, 2),
        (tag::N_KV_HEAD, 2),
        (tag::HEAD_DIM, 7),
        (tag::HIDDEN, 64),
        (tag::MAX_SEQ, 32),
    ] {
        w = w.u32(t, v);
    }
    let bad = w
        .f64(tag::ROPE_BASE, 1e4)
        .f64(tag::RMS_EPS, 1e-6)
        .u64(tag::SEED, 1)
        .finish();
    let err = load::new_request(&bad, never()).unwrap_err();
    assert!(err.contains("head_dim"), "{err}");
    let tiny = spec_fields(Writer::default())
        .u64(tag::SEED, 1)
        .u64(tag::BUDGET, 1024)
        .finish();
    let err = load::new_request(&tiny, never()).unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    assert_eq!(session::session_count(), 0);
}

#[test]
fn train_open_refuses_each_bad_setup_and_commits_nothing() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    std::fs::write(dir.join("odd.bin"), [1u8, 2, 3]).unwrap();
    // A name inside the root that points outside it.
    let outside = std::env::temp_dir().join(format!("ojas-capi-bin-out-{}", std::process::id()));
    write_bin(
        &std::env::temp_dir(),
        outside.file_name().unwrap().to_str().unwrap(),
    );
    std::os::unix::fs::symlink(&outside, dir.join("link.bin")).unwrap();
    let id = load_path("model.safetensors").unwrap().id;
    let with = |w: Writer| open_payload(id, w);
    let base = || train_fields(Writer::default(), "tokens.bin");
    let cases: Vec<(Vec<u8>, &str)> = vec![
        (
            with(train_fields(Writer::default(), "missing.bin")),
            "missing file",
        ),
        (with(train_fields(Writer::default(), "../tokens.bin")), ".."),
        (with(train_fields(Writer::default(), "link.bin")), "escapes"),
        (
            with(train_fields(Writer::default(), "odd.bin")),
            "truncated token bin",
        ),
        (with(base().u32(tag::BATCH, 0)), "repeats"),
        (
            with(
                Writer::default()
                    .str(tag::TOKEN_BIN, "tokens.bin")
                    .u32(tag::BIN_FORMAT, 0),
            ),
            "is required",
        ),
        (
            with(base().f64(tag::DECAY_FRAC, 0.2)),
            "cosine schedule refuses",
        ),
        (with(base().u32(tag::NUMERICS, 1)), "unknown option field 5"),
        (open_payload(id + 1000, base()), "unknown model"),
    ];
    for (payload, want) in cases {
        let err = train::open_request(&payload, never()).unwrap_err();
        assert!(err.contains(want), "{want}: {err}");
    }
    // Field values the trainer refuses.
    for (t, v, want) in [
        (tag::SEQ, 64u32, "max_seq"),
        (tag::ACCUM, 0, "non-zero"),
        (tag::SCHEDULE, 9, "unknown schedule"),
        (tag::ON_NONFINITE, 4, "unknown on_nonfinite"),
        (tag::BIN_FORMAT, 7, "unknown bin_format"),
        (tag::AUTOCAST, 2, "unknown autocast"),
    ] {
        let mut w = Writer::default();
        let fields = [
            (tag::BATCH, 2u32),
            (tag::SEQ, SEQ),
            (tag::ACCUM, 2),
            (tag::SCHEDULE, train::SCHEDULE_COSINE),
            (tag::ON_NONFINITE, 0),
            (tag::BIN_FORMAT, 0),
            (tag::AUTOCAST, 0),
        ];
        for (ft, fv) in fields {
            w = w.u32(ft, if ft == t { v } else { fv });
        }
        let w = w
            .str(tag::TOKEN_BIN, "tokens.bin")
            .u64(tag::DATA_SEED, 1)
            .u64(tag::WARMUP, 2)
            .u64(tag::TOTAL, 40)
            .f64(tag::MATRIX_LR, 0.025)
            .f64(tag::ADAM_LR, 6e-4)
            .f32(tag::GRAD_CLIP, 1.0);
        let err = train::open_request(&open_payload(id, w), never()).unwrap_err();
        assert!(err.contains(want), "{want}: {err}");
    }
    // Nothing was committed: the model still samples and has no trainer.
    let err = step(id).unwrap_err();
    assert!(err.contains("no trainer"), "{err}");
    train_open(id, "tokens.bin").unwrap();
    let err = train_open(id, "tokens.bin").unwrap_err();
    assert!(err.contains("already has a trainer"), "{err}");
    // Grouped-query attention trains through the same causal SDPA.
    let s = nano_spec();
    let mut w = Writer::default();
    for (t, v) in [
        (tag::VOCAB, s.vocab as u32),
        (tag::N_EMBD, s.n_embd as u32),
        (tag::N_LAYER, s.n_layer as u32),
        (tag::N_HEAD, 2),
        (tag::N_KV_HEAD, 1),
        (tag::HEAD_DIM, s.head_dim as u32),
        (tag::HIDDEN, s.hidden as u32),
        (tag::MAX_SEQ, s.max_seq as u32),
    ] {
        w = w.u32(t, v);
    }
    let gqa = load::new_request(
        &w.f64(tag::ROPE_BASE, s.rope_base)
            .f64(tag::RMS_EPS, s.rms_eps)
            .u64(tag::SEED, 1)
            .finish(),
        never(),
    )
    .unwrap();
    sample(gqa.id, &[1, 2], 0.0, 2, 0).unwrap();
    train_open(gqa.id, "tokens.bin").unwrap();
}

#[test]
fn train_open_refuses_a_trainer_the_budget_cannot_hold() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let payload = Writer::default()
        .str(tag::PATH, "model.safetensors")
        // Twice the weights: room to sample, not for a second copy plus moments.
        .u64(tag::BUDGET, 2 * model_bytes(None).len() as u64)
        .finish();
    let s = load::load_request(&payload, never()).unwrap();
    sample(s.id, &[1, 2], 0.0, 1, 0).unwrap();
    let err = train_open(s.id, "tokens.bin").unwrap_err();
    assert!(err.starts_with("ojas:E_CAPACITY:"), "{err}");
    // The resident weights were kept.
    sample(s.id, &[1, 2], 0.0, 1, 0).unwrap();
    let err = step(s.id).unwrap_err();
    assert!(err.contains("no trainer"), "{err}");
}

/// A fault in the optimizer leaves the trainer partly updated: that step and
/// every later Step, Save and Sample is `E_POISONED` until a Resume.
#[test]
fn an_optimizer_fault_poisons_the_trainer() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let s = load_path("model.safetensors").unwrap();
    train_open(s.id, "tokens.bin").unwrap();
    step(s.id).unwrap();
    save(s.id, "good").unwrap();
    s.inject("adamw_step", || OjasError::Backend {
        id: BackendId::Cpu,
        detail: "injected optimizer fault".into(),
    });
    let err = step(s.id).unwrap_err();
    assert!(err.contains("injected optimizer fault"), "{err}");
    for err in [
        step(s.id).unwrap_err(),
        save(s.id, "bad").unwrap_err(),
        sample(s.id, &[1], 0.0, 1, 0).unwrap_err(),
    ] {
        assert!(err.starts_with("ojas:E_POISONED:"), "{err}");
    }
    assert!(!dir.join("bad").exists());
    let resumed = resume("good", "tokens.bin").unwrap();
    assert_eq!(step(resumed.id).unwrap().step, 2);
}

/// A device lost in the forward phase is `E_DEVICE_LOST` and commits
/// nothing; one in the optimizer poisons the trainer.
#[test]
fn a_lost_device_is_its_own_kind() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let s = load_path("model.safetensors").unwrap();
    train_open(s.id, "tokens.bin").unwrap();
    s.inject("linear_forward", || OjasError::Backend {
        id: BackendId::Wgpu,
        detail: "device lost (Destroyed): injected".into(),
    });
    let err = step(s.id).unwrap_err();
    assert!(err.starts_with("ojas:E_DEVICE_LOST:"), "{err}");
    assert_eq!(step(s.id).unwrap().step, 1);
    s.inject("muon_ns5_step", || OjasError::Backend {
        id: BackendId::Metal,
        detail: "runtime poisoned: injected".into(),
    });
    let err = step(s.id).unwrap_err();
    assert!(err.starts_with("ojas:E_DEVICE_LOST:"), "{err}");
    let err = step(s.id).unwrap_err();
    assert!(err.starts_with("ojas:E_POISONED:"), "{err}");
    let fresh_session = load_path("model.safetensors").unwrap();
    fresh_session.inject("embedding_forward", || OjasError::Backend {
        id: BackendId::Wgpu,
        detail: "device lost: injected".into(),
    });
    let err = sample(fresh_session.id, &[1], 0.0, 1, 0).unwrap_err();
    assert!(err.starts_with("ojas:E_DEVICE_LOST:"), "{err}");
    sample(fresh_session.id, &[1], 0.0, 1, 0).unwrap();
}

/// While another call holds a model, every op on it is `E_BUSY`; a model a
/// panicking call left behind is `E_POISONED`. Other models are untouched.
#[test]
fn a_held_model_is_busy_and_a_panicked_one_is_poisoned() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let s = load_path("model.safetensors").unwrap();
    let other = load_path("model.safetensors").unwrap();
    let (release, holder) = s.hold();
    let id = s.id;
    let mut tok = id.to_le_bytes().to_vec();
    tok.extend_from_slice(&crate::tokenize::TOKENIZE_ENCODE.to_le_bytes());
    let attempts: Vec<(&str, Result<Vec<u8>, String>)> = vec![
        (
            "train_open",
            train_open(id, "tokens.bin").map(|_| Vec::new()),
        ),
        ("step", call(crate::OP_TRAIN_STEP, &sampled_payload(id))),
        (
            "sample",
            call(crate::OP_SAMPLE, &sample_payload(id, &[1], 0.0, 1, 0)),
        ),
        ("save", save(id, "ckpt").map(|_| Vec::new())),
        ("tokenize", call(crate::OP_TOKENIZE, &tok)),
    ];
    for (name, got) in attempts {
        let err = got.unwrap_err();
        assert!(err.starts_with("ojas:E_BUSY:"), "{name}: {err}");
    }
    sample(other.id, &[1], 0.0, 1, 0).unwrap();
    release.send(()).unwrap();
    holder.join().unwrap();
    sample(id, &[1], 0.0, 1, 0).unwrap();
    s.poison_state();
    let err = sample(id, &[1], 0.0, 1, 0).unwrap_err();
    assert!(err.starts_with("ojas:E_POISONED:"), "{err}");
    session::try_free(id).unwrap();
}

/// G9 through the C ABI: 4 steps straight equal 2 steps, Save, a new
/// session from Resume, then 2 more, bit for bit, and both models then
/// sample the same tokens.
#[test]
fn save_and_resume_continue_the_run_bit_for_bit() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let straight = load_exact("model.safetensors").unwrap();
    train_open(straight.id, "tokens.bin").unwrap();
    let want: Vec<_> = (0..4).map(|_| step(straight.id).unwrap()).collect();

    let first = load_exact("model.safetensors").unwrap();
    train_open(first.id, "tokens.bin").unwrap();
    let mut got: Vec<_> = (0..2).map(|_| step(first.id).unwrap()).collect();
    std::fs::create_dir_all(dir.join("runs")).unwrap();
    save(first.id, "runs/ckpt").unwrap();
    session::try_free(first.id).unwrap();
    let resumed = resume("runs/ckpt", "tokens.bin").unwrap();
    assert_eq!(resumed.tensors, straight.tensors);
    assert_eq!(
        resumed.path.as_deref(),
        Some(dir.canonicalize().unwrap().join("runs/ckpt").as_path())
    );
    got.extend((0..2).map(|_| step(resumed.id).unwrap()));
    for (a, b) in want.iter().zip(&got) {
        assert_eq!(a.loss.to_bits(), b.loss.to_bits(), "{a:?} {b:?}");
        assert_eq!(a.grad_norm.to_bits(), b.grad_norm.to_bits(), "{a:?} {b:?}");
        assert_eq!(a.step, b.step);
        assert_eq!(a.matrix_lr.to_bits(), b.matrix_lr.to_bits());
    }
    let prompt = [9u32, 8, 7];
    assert_eq!(
        sample(straight.id, &prompt, 0.7, 6, 5).unwrap(),
        sample(resumed.id, &prompt, 0.7, 6, 5).unwrap()
    );
    // Saving again replaces the directory whole.
    save(resumed.id, "runs/ckpt").unwrap();
    let again = resume("runs/ckpt", "tokens.bin").unwrap();
    assert_eq!(step(again.id).unwrap().step, 5);
}

#[test]
fn save_and_resume_refuse_and_add_nothing() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let untrained = load_path("model.safetensors").unwrap();
    let err = save(untrained.id, "ckpt").unwrap_err();
    assert!(err.contains("no trainer"), "{err}");
    let s = load_exact("model.safetensors").unwrap();
    train_open(s.id, "tokens.bin").unwrap();
    step(s.id).unwrap();
    for (raw, want) in [
        ("../ckpt", ".."),
        ("/tmp/ckpt", "relative"),
        ("model.safetensors", "not a directory"),
        ("missing/ckpt", "missing parent"),
        ("", "empty"),
    ] {
        let err = save(s.id, raw).unwrap_err();
        assert!(err.contains(want), "{raw}: {err}");
    }
    let outside = std::env::temp_dir().join(format!("ojas-capi-ckpt-out-{}", std::process::id()));
    std::fs::create_dir_all(&outside).unwrap();
    std::os::unix::fs::symlink(&outside, dir.join("escape")).unwrap();
    let err = save(s.id, "escape/ckpt").unwrap_err();
    assert!(err.contains("escapes"), "{err}");
    save(s.id, "ckpt").unwrap();
    let before = session::session_count();

    let refusals: Vec<(Vec<u8>, &str)> = vec![
        (
            resume_fields("nowhere", "tokens.bin").finish(),
            "missing directory",
        ),
        (resume_fields("escape", "tokens.bin").finish(), "escapes"),
        (
            resume_fields("ckpt", "tokens.bin")
                .u64(tag::BUDGET, 2048)
                .finish(),
            "ojas:E_CAPACITY:",
        ),
        (
            train_fields(Writer::default().str(tag::PATH, "ckpt"), "tokens.bin")
                .raw(tag::TOKENIZER_HASH, &[7u8; 32])
                .finish(),
            "tokenizer hash",
        ),
        (
            {
                // The same fields with another data seed.
                let mut w = Writer::default().str(tag::PATH, "ckpt");
                for (t, v) in [
                    (tag::BATCH, 2u32),
                    (tag::SEQ, SEQ),
                    (tag::ACCUM, 2),
                    (tag::SCHEDULE, train::SCHEDULE_COSINE),
                    (tag::ON_NONFINITE, 0),
                    (tag::BIN_FORMAT, 0),
                ] {
                    w = w.u32(t, v);
                }
                w.str(tag::TOKEN_BIN, "tokens.bin")
                    .u64(tag::DATA_SEED, 8)
                    .u64(tag::WARMUP, 2)
                    .u64(tag::TOTAL, 40)
                    .f64(tag::MATRIX_LR, ojas_model::NANOLAB_MATRIX_LR)
                    .f64(tag::ADAM_LR, ojas_model::NANOLAB_ADAM_LR)
                    .f32(tag::GRAD_CLIP, ojas_model::NANOLAB_GRAD_CLIP)
                    .finish()
            },
            "differs",
        ),
    ];
    for (payload, want) in refusals {
        let err = train::resume_request(&payload, never()).unwrap_err();
        assert!(err.contains(want), "{want}: {err}");
    }
    // A truncated optimizer file is refused with nothing applied.
    let optim = dir.join("ckpt").join(ojas_model::OPTIM_FILE);
    let bytes = std::fs::read(&optim).unwrap();
    std::fs::write(&optim, &bytes[..bytes.len() - 4]).unwrap();
    let err = resume("ckpt", "tokens.bin").unwrap_err();
    assert!(err.contains("optim"), "{err}");
    assert_eq!(session::session_count(), before);
    let _ = std::fs::remove_dir_all(&outside);
}

/// A vocabulary of the 256 byte pieces plus merges `h e` and `l l`.
fn write_tokenizer(dir: &Path) {
    let map = ojas_data::bytes_to_unicode();
    let mut vocab = String::from("{");
    for (i, ch) in map.iter().enumerate() {
        let piece = match ch {
            '"' => "\\\"".to_string(),
            '\\' => "\\\\".to_string(),
            c => c.to_string(),
        };
        vocab.push_str(&format!("\"{piece}\":{i},"));
    }
    vocab.push_str("\"he\":256,\"ll\":257}");
    std::fs::write(dir.join("vocab.json"), vocab).unwrap();
    std::fs::write(dir.join("merges.txt"), "#version: 0.2\nh e\nl l\n").unwrap();
}

fn tokenizer_payload(id: u64, vocab: &str, merges: &str) -> Vec<u8> {
    open_payload(
        id,
        Writer::default()
            .str(tag::VOCAB_JSON, vocab)
            .str(tag::MERGES_TXT, merges),
    )
}

fn tokenize(id: u64, mode: u32, body: &[u8]) -> Result<Vec<u8>, String> {
    let mut p = id.to_le_bytes().to_vec();
    p.extend_from_slice(&mode.to_le_bytes());
    p.extend_from_slice(body);
    call(crate::OP_TOKENIZE, &p)
}

#[test]
fn a_tokenizer_round_trips_text_and_refuses_what_it_cannot_read() {
    let (_g, dir) = fresh();
    write_tokenizer(&dir);
    let id = load_path("model.safetensors").unwrap().id;
    let err = tokenize(id, crate::tokenize::TOKENIZE_ENCODE, b"hello").unwrap_err();
    assert!(err.contains("no tokenizer"), "{err}");
    let outside = std::env::temp_dir().join(format!("ojas-capi-vocab-out-{}", std::process::id()));
    std::fs::copy(dir.join("vocab.json"), &outside).unwrap();
    std::os::unix::fs::symlink(&outside, dir.join("link.json")).unwrap();
    for (vocab, merges, want) in [
        ("missing.json", "merges.txt", "missing file"),
        ("link.json", "merges.txt", "escapes"),
        ("../vocab.json", "merges.txt", ".."),
        ("model.safetensors", "merges.txt", "tokenizer:"),
    ] {
        let err = call(crate::OP_TOKENIZER, &tokenizer_payload(id, vocab, merges)).unwrap_err();
        assert!(err.contains(want), "{vocab}: {err}");
    }
    call(
        crate::OP_TOKENIZER,
        &tokenizer_payload(id, "vocab.json", "merges.txt"),
    )
    .unwrap();
    let ids = tokenize(
        id,
        crate::tokenize::TOKENIZE_ENCODE,
        "hello héllo".as_bytes(),
    )
    .unwrap();
    let ids: Vec<u32> = crate::tests::ids_of(&ids);
    assert_eq!(&ids[..3], &[256, 257, u32::from(b'o')], "{ids:?}");
    let body: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
    let text = tokenize(id, crate::tokenize::TOKENIZE_DECODE, &body).unwrap();
    assert_eq!(text, "hello héllo".as_bytes());
    for (mode, body, want) in [
        (
            crate::tokenize::TOKENIZE_DECODE,
            &[1u8, 2, 3][..],
            "whole number",
        ),
        (
            crate::tokenize::TOKENIZE_DECODE,
            &999u32.to_le_bytes()[..],
            "outside the vocabulary",
        ),
        (
            crate::tokenize::TOKENIZE_ENCODE,
            &[0xffu8, 0xfe][..],
            "utf-8",
        ),
        (9, &[][..], "unknown mode"),
    ] {
        let err = tokenize(id, mode, body).unwrap_err();
        assert!(err.contains(want), "{want}: {err}");
    }
}

/// Seeded sampling repeats; a stop token ends the output after it; every
/// sampling field is validated.
#[test]
fn sample_is_seeded_stops_and_refuses_bad_options() {
    let (_g, _dir) = fresh();
    let id = load_path("model.safetensors").unwrap().id;
    let prompt = [2u32, 7, 1];
    let a = sample(id, &prompt, 1.0, 12, 99).unwrap();
    assert_eq!(a, sample(id, &prompt, 1.0, 12, 99).unwrap());
    assert_eq!(a.len(), 12);
    let stop = a[3];
    let stopped = crate::tests::ids_of(
        &call(
            crate::OP_SAMPLE,
            &open_payload(
                id,
                Writer::default()
                    .f32(tag::TEMPERATURE, 1.0)
                    .u64(tag::SEED, 99)
                    .u32(tag::MAX_NEW, 12)
                    .u32s(tag::STOP, &[stop])
                    .u32s(tag::PROMPT, &prompt),
            ),
        )
        .unwrap(),
    );
    let first = a.iter().position(|&t| t == stop).unwrap();
    assert_eq!(stopped, a[..=first]);
    let topk1 = |seed| {
        crate::tests::ids_of(
            &call(
                crate::OP_SAMPLE,
                &open_payload(
                    id,
                    Writer::default()
                        .f32(tag::TEMPERATURE, 1.0)
                        .u32(tag::TOP_K, 1)
                        .u64(tag::SEED, seed)
                        .u32(tag::MAX_NEW, 5)
                        .u32s(tag::PROMPT, &prompt),
                ),
            )
            .unwrap(),
        )
    };
    assert_eq!(
        topk1(1),
        sample(id, &prompt, 0.0, 5, 0).unwrap(),
        "top_k 1 is greedy"
    );
    assert_eq!(topk1(1), topk1(2));
    let bad = |w: Writer| call(crate::OP_SAMPLE, &open_payload(id, w)).unwrap_err();
    let base = || {
        Writer::default()
            .u64(tag::SEED, 1)
            .u32(tag::MAX_NEW, 2)
            .u32s(tag::PROMPT, &prompt)
    };
    for (err, want) in [
        (bad(base().f32(tag::TEMPERATURE, -1.0)), "temperature"),
        (bad(base().f32(tag::TEMPERATURE, f32::NAN)), "temperature"),
        (
            bad(base().f32(tag::TEMPERATURE, 1.0).u32(tag::TOP_K, 0)),
            "top_k",
        ),
        (
            bad(base().f32(tag::TEMPERATURE, 1.0).f32(tag::TOP_P, 1.5)),
            "top_p",
        ),
        (bad(base()), "temperature is required"),
        (
            bad(base().f32(tag::TEMPERATURE, 1.0).u32s(tag::STOP, &[64])),
            "vocab",
        ),
        (
            bad(base().f32(tag::TEMPERATURE, 1.0).str(tag::PATH, "x")),
            "unknown option field 1",
        ),
    ] {
        assert!(err.contains(want), "{want}: {err}");
    }
    let err = sample(id + 1000, &prompt, 0.0, 1, 0).unwrap_err();
    assert!(err.contains("unknown model"), "{err}");
    let zero = sample(id, &prompt, 0.0, 0, 0).unwrap();
    assert!(zero.is_empty());
}

/// TRAIN_STEP refusals not covered by the carried-over step tests.
#[test]
fn train_step_refuses_a_free_trainer_session_and_reports_the_policy() {
    let (_g, dir) = fresh();
    write_bin(&dir, "tokens.bin");
    let s = load_path("model.safetensors").unwrap();
    // SkipBatch moves the cursor past a non-finite batch; the error stays.
    let payload = open_payload(
        s.id,
        Writer::default()
            .str(tag::TOKEN_BIN, "tokens.bin")
            .u32(tag::BIN_FORMAT, 0)
            .u32(tag::BATCH, 2)
            .u32(tag::SEQ, SEQ)
            .u32(tag::ACCUM, 2)
            .u64(tag::DATA_SEED, 7)
            .u32(tag::SCHEDULE, train::SCHEDULE_WSD)
            .u64(tag::WARMUP, 2)
            .u64(tag::TOTAL, 40)
            .f64(tag::DECAY_FRAC, 0.2)
            .f64(tag::MATRIX_LR, 0.025)
            .f64(tag::ADAM_LR, 6e-4)
            .f32(tag::GRAD_CLIP, 1.0)
            .u32(tag::ON_NONFINITE, 1),
    );
    train::open_request(&payload, never()).unwrap();
    s.inject("linear_cross_entropy_mean", || OjasError::NonFinite {
        op: "test",
    });
    let err = step(s.id).unwrap_err();
    assert!(err.starts_with("ojas:E_NONFINITE:"), "{err}");
    let after = step(s.id).unwrap();
    assert_eq!(after.step, 1);
    assert_eq!(after.tokens, 2 * 2 * u64::from(SEQ));
    let x: Vec<u32> = (0..SEQ).collect();
    let err = call(
        crate::OP_TRAIN_STEP,
        &tokens_payload(s.id + 1000, SEQ, &[(1, &x, &x)]),
    )
    .unwrap_err();
    assert!(err.contains("unknown model"), "{err}");
    let out = step_out(&call(crate::OP_TRAIN_STEP, &sampled_payload(s.id)).unwrap());
    assert_eq!(out.step, 2);
    // Inspect is file-only and leaves sessions alone.
    write_tensor(&dir.join("t.safetensors"));
    assert_eq!(load::inspect("t.safetensors").unwrap(), 1);
}
