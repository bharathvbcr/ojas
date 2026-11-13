//! Adversarial lane for ft-042351e4 (`b3/tsp-adv`): attacks on the
//! `TapeStepProvider` validate-before-take, zero-marker and one-scan changes
//! of `b3/tsp` (c6aa5f6). Every test states the attack in its name; a failure
//! is a BREAK of the claim.
//!
//! 1. check/apply divergence (seeded differential against an independent oracle).
//! 2. a stale scan summary (random op interleavings against a fresh replay).
//! 3. zero-marker semantics (against explicit zero tensors, a refusal in
//!    between, a checkpoint round trip).
//! 4. a refusal at the first, middle and last leaf with `grad_scale != 1`.
//! 5. the Metal readback counts (macOS only).

mod qwen35_common;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use ojas_core::{DType, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::sha256::open_sha256_hex;
use ojas_model::qwen35::adapters::BaseIdentity;
use ojas_model::qwen35::{DeviceHooks, LoraInit, OpenConfig, TapeStepProvider};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};
use ojas_train::lappi::step::{
    AdamWHyper, BankMode, CheckpointContext, RowTargets, SequenceJob, StepError, StepProvider,
};
use qwen35_common::lora::*;
use qwen35_common::*;

type P = TapeStepProvider<CpuBackend>;

const BASE_FILE: &str = "model.safetensors";

/// `2^64 - 2^40`, the largest f32 whose square is finite (optim.rs).
const LIMIT: f32 = 1.844_674_3e19;

static DIRS: AtomicU64 = AtomicU64::new(0);

struct Tmp(PathBuf);

impl Tmp {
    fn new() -> Self {
        let n = DIRS.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("ojas tsp adv {}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Rng(u64);

impl Rng {
    fn bump(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.bump() % n as u64) as usize
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn bits_all(v: &[Vec<f32>]) -> Vec<Vec<u32>> {
    v.iter().map(|x| bits(x)).collect()
}

fn base_identity() -> BaseIdentity {
    static SHA: OnceLock<String> = OnceLock::new();
    let sha = SHA
        .get_or_init(|| open_sha256_hex(&dir().join(BASE_FILE)).unwrap().1)
        .clone();
    BaseIdentity {
        file_name: BASE_FILE.to_string(),
        sha256: sha,
        quantizer: None,
    }
}

fn open_config(widest: usize) -> OpenConfig {
    OpenConfig {
        spec: spec(),
        prefix: PREFIX.to_string(),
        base_path: dir().join(BASE_FILE),
        base_id: base_identity(),
        linears: DType::F32,
        table: DType::F32,
        widest,
        chunk: DEFAULT_CE_CHUNK,
        activations: ActivationCheckpoint::Off,
    }
}

fn lora_init(nonzero_b: bool) -> LoraInit {
    LoraInit {
        config: config(),
        targets: targets(),
        adapters: fixture_sites(&lora_file(), nonzero_b),
    }
}

fn open(nonzero_b: bool) -> P {
    TapeStepProvider::open_lora(
        cpu(),
        DeviceHooks::default(),
        open_config(16),
        lora_init(nonzero_b),
    )
    .unwrap()
}

fn full_provider() -> P {
    TapeStepProvider::open_full(cpu(), DeviceHooks::default(), open_config(16)).unwrap()
}

fn n_trainable<B: ojas_core::Backend>(p: &TapeStepProvider<B>) -> usize {
    p.parameters().iter().filter(|s| s.trainable).count()
}

struct Job {
    tokens: Vec<u32>,
    positions: Vec<u32>,
    targets: Vec<u32>,
    scale: f32,
}

impl Job {
    fn view(&self) -> SequenceJob<'_> {
        SequenceJob {
            tokens: &self.tokens,
            rows: RowTargets {
                positions: &self.positions,
                targets: &self.targets,
                scale: self.scale,
            },
            hidden_at: &[],
        }
    }
}

fn job(t: usize, salt: usize, scale: f32) -> Job {
    let tokens: Vec<u32> = (0..t)
        .map(|i| ((i * 37 + salt * 101 + 11) % 8320) as u32)
        .collect();
    let positions: Vec<u32> = (t / 2..t).map(|p| p as u32).collect();
    let targets = positions
        .iter()
        .map(|&p| tokens[(p as usize + 1) % t])
        .collect();
    Job {
        tokens,
        positions,
        targets,
        scale,
    }
}

fn hyper(grad_scale: f64) -> AdamWHyper {
    AdamWHyper {
        lr: 5e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale,
    }
}

fn per_entry(n: usize) -> (Vec<f64>, Vec<f64>) {
    (
        (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.5 }).collect(),
        (0..n)
            .map(|i| if i % 3 == 0 { 0.01 } else { 0.0 })
            .collect(),
    )
}

fn acc(p: &mut P, j: &Job, mode: BankMode) -> Result<f64, StepError> {
    p.accumulate(&j.view(), mode, None)
}

fn step(p: &mut P, grad_scale: f64) -> Result<(), StepError> {
    let (lr_scale, wd) = per_entry(n_trainable(p));
    let next = p.step_count() + 1;
    p.adamw_step(&hyper(grad_scale), next, &lr_scale, &wd)
}

fn leaf_len(p: &P, i: usize) -> usize {
    p.bank().entries()[i].shape.iter().product()
}

fn host_leaf(p: &P, i: usize) -> Vec<f32> {
    p.bank()
        .grad_on(p.backend(), i)
        .unwrap()
        .to_f32_vec()
        .unwrap()
}

fn put_leaf(p: &mut P, i: usize, v: &[f32]) {
    let shape = p.bank().entries()[i].shape.clone();
    let t = Tensor::from_f32(v, &shape, &budget()).unwrap();
    p.bank_mut()
        .accumulate(&cpu(), i, BankMode::Overwrite, t)
        .unwrap();
}

/// What a bank slot holds, for a bit-exact comparison.
#[derive(PartialEq, Debug)]
enum View {
    Empty,
    Marker,
    Dense(Vec<u32>),
}

struct Snap {
    bank: Vec<View>,
    params: Vec<Vec<u32>>,
    moments: Vec<(Vec<u32>, Vec<u32>)>,
    step: u64,
}

fn views(p: &P) -> Vec<View> {
    (0..p.leaf_count())
        .map(|i| {
            if p.bank().is_zero(i).unwrap() {
                View::Marker
            } else {
                match p.bank().grad(i) {
                    Ok(t) => View::Dense(bits(&t.to_f32_vec().unwrap())),
                    Err(_) => View::Empty,
                }
            }
        })
        .collect()
}

fn snap(p: &mut P) -> Snap {
    // Moments are readable on the LoRA provider only; the full provider
    // compares parameters, bank and step.
    let moments = p
        .read_moments()
        .map(|m| m.iter().map(|(m, v)| (bits(m), bits(v))).collect())
        .unwrap_or_default();
    Snap {
        bank: views(p),
        params: bits_all(&p.read_parameters().unwrap()),
        moments,
        step: p.step_count(),
    }
}

/// The first field that differs, if any.
fn diff(a: &Snap, b: &Snap) -> Option<String> {
    if let Some(i) = (0..a.bank.len()).find(|&i| a.bank[i] != b.bank[i]) {
        return Some(format!("bank leaf {i} differs"));
    }
    if a.params != b.params {
        return Some("parameters differ".into());
    }
    if a.moments != b.moments {
        return Some("moments differ".into());
    }
    (a.step != b.step).then(|| format!("step {} vs {}", a.step, b.step))
}

fn diff_state(a: &Snap, b: &Snap) -> Option<String> {
    if a.params != b.params {
        return Some("parameters differ".into());
    }
    if a.moments != b.moments {
        return Some("moments differ".into());
    }
    (a.step != b.step).then(|| format!("step {} vs {}", a.step, b.step))
}

// ---------------------------------------------------------------------------
// Attack 1: check/apply divergence
// ---------------------------------------------------------------------------

fn specials() -> Vec<f32> {
    vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        LIMIT,
        f32::from_bits(LIMIT.to_bits() + 1),
        f32::from_bits(LIMIT.to_bits() - 1),
        1e-45,
        -0.0,
        1e19,
        -1e19,
        3e19,
        f32::MAX,
        0.0,
        1e-38,
    ]
}

fn scales() -> Vec<f64> {
    vec![
        1.0,
        0.5,
        2.0,
        1.0 + f64::from(f32::EPSILON),
        1.0 - f64::from(f32::EPSILON) / 2.0,
        1.000_000_000_1,
        1e-40,
        1e-50,
        0.0,
        -1.0,
        -0.0,
        1e19,
        1e30,
        3e38,
        1e300,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        5.4e-20,
        2.0f64.powi(-64),
    ]
}

#[test]
fn check_all_and_apply_all_agree_over_seeded_random_batches() {
    let specials = specials();
    let scales = scales();
    let mut failures = Vec::new();
    let mut tally = [0usize; 2];
    for seed in 1..=90u64 {
        let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut p = open(true);
        let n = p.leaf_count();
        acc(
            &mut p,
            &job(8 + r.below(5), seed as usize, 1.0),
            BankMode::Overwrite,
        )
        .unwrap();
        if r.below(3) == 0 {
            acc(&mut p, &job(9, seed as usize + 7, 0.5), BankMode::Add).unwrap();
        }
        for i in 0..n {
            if r.below(6) == 0 {
                p.bank_mut()
                    .accumulate_zero(i, BankMode::Overwrite)
                    .unwrap();
            }
        }
        let mut poisoned = None::<f32>;
        if r.below(4) != 0 {
            for _ in 0..1 + r.below(3) {
                let leaf = r.below(n);
                let mut v = host_leaf(&p, leaf);
                let at = r.below(v.len());
                let val = specials[r.below(specials.len())];
                v[at] = val;
                put_leaf(&mut p, leaf, &v);
                if val.is_finite() && val != 0.0 {
                    poisoned = Some(val);
                }
            }
        }
        let scale: f64 = match (poisoned, r.below(3)) {
            (Some(v), 0 | 1) => {
                let f = [1.0 - 1e-6, 1.0, 1.0 + 1e-7, 1.0 + 1e-6, 1.0 - 1e-7][r.below(5)];
                (f64::from(LIMIT) / f64::from(v.abs())) * f
            }
            _ => scales[r.below(scales.len())],
        };
        let what = format!("seed {seed} scale {scale:e} poisoned {poisoned:?}");

        // The oracle, independent of optim.rs: the scale is a finite f32 and
        // every scaled gradient squares to a finite f32.
        let s32 = scale as f32;
        // `AdamWHyper::validate` refuses a non-positive or non-finite scale up front.
        let mut want_ok = scale.is_finite() && scale > 0.0 && s32.is_finite();
        if want_ok {
            'leaves: for i in 0..n {
                for g in host_leaf(&p, i) {
                    let x = g * s32;
                    if !(x * x).is_finite() {
                        want_ok = false;
                        break 'leaves;
                    }
                }
            }
        }

        let before = snap(&mut p);
        match step(&mut p, scale) {
            Ok(()) => {
                tally[0] += 1;
                if !want_ok {
                    failures.push(format!("{what}: accepted a batch the oracle refuses"));
                }
                if p.bank().is_complete() || p.step_count() != 1 {
                    failures.push(format!("{what}: an accepted step left the bank/step wrong"));
                }
            }
            Err(e) => {
                tally[1] += 1;
                let msg = e.to_string();
                if want_ok {
                    failures.push(format!("{what}: refused a batch the oracle accepts: {msg}"));
                }
                if !msg.contains("no tensor was stepped")
                    && !msg.contains("hyperparameters out of range")
                {
                    failures.push(format!("{what}: not a clean pre-apply refusal: {msg}"));
                }
                if let Some(d) = diff(&before, &snap(&mut p)) {
                    failures.push(format!("{what}: refusal changed state: {d} ({msg})"));
                }
            }
        }
    }
    eprintln!("differential: accepted {} refused {}", tally[0], tally[1]);
    assert!(
        tally[0] > 5 && tally[1] > 5,
        "the sweep must hit both arms: {tally:?}"
    );
    assert!(
        failures.is_empty(),
        "{} BREAKs:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Attack 2: stale scan summary
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Op {
    AccO(usize, usize, f32),
    AccA(usize, usize, f32),
    Norm,
    Put(usize, f32),
    Marker(usize),
    MarkerAdd(usize),
    Clear,
    Step(f64),
}

fn gen_ops(r: &mut Rng, n: usize, len: usize) -> Vec<Op> {
    let mut ops = Vec::new();
    for _ in 0..len {
        ops.push(match r.below(10) {
            0 => Op::AccO(r.below(40), 8 + r.below(4), [1.0, 0.5, 0.0][r.below(3)]),
            1 | 2 => Op::AccA(r.below(40), 8 + r.below(4), [1.0, 0.5, 0.0][r.below(3)]),
            3 | 4 => Op::Norm,
            5 => Op::Put(
                r.below(n),
                [0.25, -0.5, 0.75, 1.5, 0.125, f32::NAN][r.below(6)],
            ),
            6 => Op::Marker(r.below(n)),
            7 => Op::MarkerAdd(r.below(n)),
            8 if r.below(3) == 0 => Op::Clear,
            8 => Op::AccO(r.below(40), 8 + r.below(4), 1.0),
            _ => Op::Step([1.0, 0.5][r.below(2)]),
        });
    }
    ops
}

fn run_op(p: &mut P, op: Op) -> String {
    match op {
        Op::AccO(s, t, sc) => format!(
            "{:?}",
            acc(p, &job(t, s, sc), BankMode::Overwrite)
                .map(f64::to_bits)
                .map_err(|e| e.to_string())
        ),
        Op::AccA(s, t, sc) => format!(
            "{:?}",
            acc(p, &job(t, s, sc), BankMode::Add)
                .map(f64::to_bits)
                .map_err(|e| e.to_string())
        ),
        Op::Norm => format!(
            "{:?}",
            p.grad_sq_norm()
                .map(f64::to_bits)
                .map_err(|e| e.to_string())
        ),
        Op::Put(i, v) => {
            let len = leaf_len(p, i);
            put_leaf(p, i, &vec![v; len]);
            "put".into()
        }
        Op::Marker(i) => format!(
            "{:?}",
            p.bank_mut().accumulate_zero(i, BankMode::Overwrite).is_ok()
        ),
        Op::MarkerAdd(i) => format!(
            "{:?}",
            p.bank_mut().accumulate_zero(i, BankMode::Add).is_ok()
        ),
        Op::Clear => {
            p.bank_mut().clear();
            "clear".into()
        }
        Op::Step(s) => format!("{:?}", step(p, s).map_err(|e| e.to_string())),
    }
}

#[test]
fn a_cached_scan_never_survives_any_bank_change_interleaving() {
    let mut failures = Vec::new();
    let mut norms_ok = 0usize;
    let mut norms_err = 0usize;
    for seed in 1..=30u64 {
        let mut r = Rng(seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1);
        let probe = open(true);
        let ops = gen_ops(&mut r, probe.leaf_count(), 18);
        drop(probe);

        // Subject: every op, norms included.
        let mut subject = open(true);
        let mut logs = Vec::new();
        for (k, op) in ops.iter().enumerate() {
            let log = run_op(&mut subject, *op);
            if matches!(op, Op::Norm) {
                // A fresh provider that replays the prefix without any norm.
                let mut fresh = open(true);
                for o in &ops[..k] {
                    if !matches!(o, Op::Norm) {
                        run_op(&mut fresh, *o);
                    }
                }
                let want = run_op(&mut fresh, Op::Norm);
                if want.starts_with("Ok") {
                    norms_ok += 1;
                } else {
                    norms_err += 1;
                }
                if log != want {
                    failures.push(format!(
                        "seed {seed} op {k} {op:?}: norm {log} but a fresh bank gives {want}\nops: {:?}",
                        &ops[..=k]
                    ));
                }
            }
            logs.push(log);
        }
        // Reference: the same ops without a single norm.
        let mut reference = open(true);
        for (k, op) in ops.iter().enumerate() {
            if matches!(op, Op::Norm) {
                continue;
            }
            let log = run_op(&mut reference, *op);
            if log != logs[k] {
                failures.push(format!(
                    "seed {seed} op {k} {op:?}: with norms {} without {log}",
                    logs[k]
                ));
            }
        }
        let (a, b) = (snap(&mut subject), snap(&mut reference));
        if let Some(d) = diff(&a, &b) {
            failures.push(format!(
                "seed {seed}: final state differs: {d}\nops: {ops:?}"
            ));
        }
        // And one more update on the survivors, after a norm on one side only.
        let _ = subject.grad_sq_norm();
        let (ra, rb) = (step(&mut subject, 0.5), step(&mut reference, 0.5));
        if ra.is_ok() != rb.is_ok() {
            failures.push(format!("seed {seed}: closing step {ra:?} vs {rb:?}"));
        } else if let Some(d) = diff(&snap(&mut subject), &snap(&mut reference)) {
            failures.push(format!("seed {seed}: closing step state differs: {d}"));
        }
    }
    eprintln!("norm checks: ok {norms_ok} err {norms_err}");
    assert!(norms_ok > 3, "the sweep barely asked for a successful norm");
    assert!(
        failures.is_empty(),
        "{} BREAKs:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn the_scanned_norm_equals_the_unscanned_norm_bit_for_bit_with_markers() {
    for seed in 1..=10u64 {
        let mut r = Rng(seed.wrapping_mul(0xA24B_AED4_963E_E407) | 1);
        let mut p = open(true);
        acc(&mut p, &job(10, seed as usize, 1.0), BankMode::Overwrite).unwrap();
        for i in 0..p.leaf_count() {
            if r.below(3) == 0 {
                p.bank_mut()
                    .accumulate_zero(i, BankMode::Overwrite)
                    .unwrap();
            }
        }
        let plain = p.bank().sq_norm(p.backend()).unwrap();
        let scanned = p.grad_sq_norm().unwrap();
        assert_eq!(plain.to_bits(), scanned.to_bits(), "seed {seed}");
    }
}

// ---------------------------------------------------------------------------
// Attack 3: zero marker semantics
// ---------------------------------------------------------------------------

fn explicit_zero_bank(p: &mut P) {
    for i in 0..p.leaf_count() {
        let len = leaf_len(p, i);
        put_leaf(p, i, &vec![0.0; len]);
    }
}

#[test]
fn a_zero_marker_step_equals_a_step_over_explicit_zero_tensors_in_both_modes() {
    for (what, mk) in [
        ("lora", open as fn(bool) -> P),
        ("full", |_| full_provider()),
    ] {
        for grad_scale in [1.0, 0.5] {
            let zero = job(8, 3, 0.0);
            let mut marker = mk(true);
            let initial = snap(&mut marker).params;
            acc(&mut marker, &zero, BankMode::Overwrite).unwrap();
            acc(&mut marker, &zero, BankMode::Add).unwrap();
            assert!(
                (0..marker.leaf_count()).all(|i| marker.bank().is_zero(i).unwrap()),
                "{what}: scale 0 then scale 0 left a non-marker"
            );
            step(&mut marker, grad_scale).unwrap();

            let mut explicit = mk(true);
            explicit_zero_bank(&mut explicit);
            step(&mut explicit, grad_scale).unwrap();

            let (a, b) = (snap(&mut marker), snap(&mut explicit));
            assert_eq!(
                diff_state(&a, &b),
                None,
                "{what} grad_scale {grad_scale}: marker step differs from explicit zeros"
            );
            eprintln!(
                "{what} grad_scale {grad_scale}: zero-gradient step moved the parameters: {}",
                a.params != initial
            );
        }
    }
}

#[test]
fn a_zero_marker_survives_a_refusal_and_the_retry_equals_a_run_that_never_refused() {
    let make = || {
        let mut p = open(true);
        acc(&mut p, &job(8, 3, 0.0), BankMode::Overwrite).unwrap();
        p
    };
    let n = make().leaf_count();
    for leaf in [0, n / 2, n - 1] {
        for grad_scale in [1.0, 0.5] {
            let what = format!("leaf {leaf} grad_scale {grad_scale}");
            let mut p = make();
            let len = leaf_len(&p, leaf);
            put_leaf(&mut p, leaf, &vec![f32::NAN; len]);
            let before = snap(&mut p);
            let err = step(&mut p, grad_scale).unwrap_err();
            assert!(
                err.to_string().contains("no tensor was stepped"),
                "{what}: {err}"
            );
            assert_eq!(diff(&before, &snap(&mut p)), None, "{what}");
            for j in (0..n).filter(|&j| j != leaf) {
                assert!(
                    p.bank().is_zero(j).unwrap(),
                    "{what}: leaf {j} lost its marker"
                );
            }
            p.bank_mut()
                .accumulate_zero(leaf, BankMode::Overwrite)
                .unwrap();
            step(&mut p, grad_scale).unwrap();
            let mut control = make();
            step(&mut control, grad_scale).unwrap();
            assert_eq!(
                diff_state(&snap(&mut p), &snap(&mut control)),
                None,
                "{what}"
            );
        }
    }
}

fn saved_bytes(p: &mut P, ctx: &CheckpointContext) -> Vec<(String, Vec<u8>)> {
    let tmp = Tmp::new();
    let mut files: Vec<(String, Vec<u8>)> = p
        .save_state_with(&tmp.0, ctx)
        .unwrap()
        .iter()
        .map(|f| {
            (
                f.trim_prefix(&tmp.0).to_string_lossy().into_owned(),
                std::fs::read(f).unwrap(),
            )
        })
        .collect();
    files.sort();
    files
}

#[test]
fn a_checkpoint_taken_with_a_zero_marker_in_the_bank_equals_one_taken_with_an_empty_bank() {
    let ctx = CheckpointContext {
        optimizer_step: 0,
        seed: 7,
        epoch: 0,
        consumed_digest: "ab".repeat(32),
    };
    let mut with_marker = open(true);
    acc(&mut with_marker, &job(8, 3, 0.0), BankMode::Overwrite).unwrap();
    let mut empty = open(true);
    let a = saved_bytes(&mut with_marker, &ctx);
    let b = saved_bytes(&mut empty, &ctx);
    assert!(!a.is_empty());
    assert_eq!(a, b, "the checkpoint depends on the bank");
    // The marker is still pending after the save.
    assert!(with_marker.bank().is_zero(0).unwrap());
}

#[test]
fn a_resumed_run_equals_an_unresumed_one_when_a_zero_marker_step_came_first() {
    let ctx = CheckpointContext {
        optimizer_step: 1,
        seed: 7,
        epoch: 0,
        consumed_digest: "ab".repeat(32),
    };
    let run = |p: &mut P, from: u64, to: u64| {
        for s in from..to {
            acc(
                p,
                &job(9 + (s as usize % 3), s as usize, 1.0),
                BankMode::Overwrite,
            )
            .unwrap();
            acc(p, &job(10, s as usize + 50, 0.0), BankMode::Add).unwrap();
            step(p, 0.5).unwrap();
        }
    };
    let mut straight = open(true);
    // Step 1: scale 0 only (markers into adamw).
    acc(&mut straight, &job(8, 3, 0.0), BankMode::Overwrite).unwrap();
    step(&mut straight, 0.5).unwrap();
    let tmp = Tmp::new();
    straight.save_state_with(&tmp.0, &ctx).unwrap();
    let mut resumed = open(false);
    resumed.load_state_with(&tmp.0, &ctx).unwrap();
    assert_eq!(diff_state(&snap(&mut straight), &snap(&mut resumed)), None);
    run(&mut straight, 1, 4);
    run(&mut resumed, 1, 4);
    assert_eq!(diff_state(&snap(&mut straight), &snap(&mut resumed)), None);
}

#[test]
fn a_marker_then_a_real_gradient_equals_explicit_zeros_then_the_real_one_in_value_and_step() {
    for (what, mk) in [
        ("lora", open as fn(bool) -> P),
        ("full", |_| full_provider()),
    ] {
        let real = job(10, 1, 1.0);
        let mut marker = mk(true);
        acc(&mut marker, &job(8, 3, 0.0), BankMode::Overwrite).unwrap();
        acc(&mut marker, &real, BankMode::Add).unwrap();
        let mut explicit = mk(true);
        explicit_zero_bank(&mut explicit);
        acc(&mut explicit, &real, BankMode::Add).unwrap();

        let (va, vb) = (views(&marker), views(&explicit));
        let mut bit_diffs = 0usize;
        let mut value_diffs = 0usize;
        for (a, b) in va.iter().zip(&vb) {
            match (a, b) {
                (View::Dense(a), View::Dense(b)) => {
                    for (x, y) in a.iter().zip(b) {
                        if x != y {
                            bit_diffs += 1;
                            if f32::from_bits(*x) != f32::from_bits(*y) {
                                value_diffs += 1;
                            }
                        }
                    }
                }
                _ => panic!("{what}: a slot is not dense after the real gradient"),
            }
        }
        eprintln!("{what}: bank bit differences {bit_diffs}, value differences {value_diffs}");
        assert_eq!(value_diffs, 0, "{what}: marker + real differs in value");
        step(&mut marker, 0.5).unwrap();
        step(&mut explicit, 0.5).unwrap();
        assert_eq!(
            diff_state(&snap(&mut marker), &snap(&mut explicit)),
            None,
            "{what}: the step differs ({bit_diffs} bank bit differences, sign of zero?)"
        );
    }
}

// ---------------------------------------------------------------------------
// Attack 4: refusal at first/middle/last leaf with grad_scale != 1
// ---------------------------------------------------------------------------

#[test]
fn a_refusal_at_any_leaf_with_a_clip_scale_leaves_the_bank_bit_identical_and_the_retry_clean() {
    let clean_job = job(10, 1, 1.0);
    let add_job = job(9, 2, 0.5);
    let n = open(true).leaf_count();
    // (value, scale, refused?)
    let cases = [
        (f32::NAN, 0.5, true),
        (1e19, 2.0, true),
        (1e19, 1.5, false),
        (-1e19, 2.0, true),
        (5e18, 4.0, true),
        (1e19, 1e-3, false),
        (1e19, 1e30, true),
    ];
    let mut failures = Vec::new();
    for leaf in [0, n / 2, n - 1] {
        for (val, scale, refused) in cases {
            let what = format!("leaf {leaf} value {val:e} scale {scale:e}");
            let build = || {
                let mut p = open(true);
                acc(&mut p, &clean_job, BankMode::Overwrite).unwrap();
                acc(&mut p, &add_job, BankMode::Add).unwrap();
                p
            };
            let mut p = build();
            let clean = host_leaf(&p, leaf);
            let mut v = clean.clone();
            let at = v.len() / 2;
            v[at] = val;
            put_leaf(&mut p, leaf, &v);
            let before = snap(&mut p);
            match (step(&mut p, scale), refused) {
                (Err(e), true) => {
                    let msg = e.to_string();
                    if !msg.contains("no tensor was stepped") {
                        failures.push(format!("{what}: {msg}"));
                    }
                    if let Some(d) = diff(&before, &snap(&mut p)) {
                        failures.push(format!("{what}: refusal changed state: {d}"));
                    }
                    put_leaf(&mut p, leaf, &clean);
                    let mut control = build();
                    if step(&mut p, 1.0).is_err() || step(&mut control, 1.0).is_err() {
                        failures.push(format!("{what}: clean retry refused"));
                    } else if let Some(d) = diff_state(&snap(&mut p), &snap(&mut control)) {
                        failures.push(format!("{what}: retry differs from never-refused: {d}"));
                    }
                }
                (Ok(()), false) => {
                    if p.step_count() != 1 {
                        failures.push(format!("{what}: step count {}", p.step_count()));
                    }
                }
                (Ok(()), true) => failures.push(format!("{what}: accepted, should refuse")),
                (Err(e), false) => failures.push(format!("{what}: refused, should accept: {e}")),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} BREAKs:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Attack 5: Metal readback counts
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod on_metal {
    use super::*;
    use ojas_core::BackendCore;
    use ojas_metal::MetalBackend;
    use ojas_model::preflight::{Calibration, Point};

    type M = TapeStepProvider<MetalBackend>;

    fn provider() -> M {
        let hooks = DeviceHooks {
            freeze: Some(Box::new(|m: &MetalBackend, t: &Tensor| m.freeze_checked(t))),
            rebind: Some(Box::new(|m: &MetalBackend, b: ojas_core::Budget| {
                m.with_budget(b)
            })),
        };
        let backend = MetalBackend::new(ojas_core::Budget::new(2 << 30))
            .unwrap_or_else(|e| panic!("no Metal device: {e:?}"));
        let mut p =
            TapeStepProvider::open_lora(backend, hooks, open_config(16), lora_init(true)).unwrap();
        let calibration = Calibration::from_points(
            "tiny",
            1,
            &[
                Point {
                    tokens: 8,
                    peak_bytes: 1 << 20,
                },
                Point {
                    tokens: 16,
                    peak_bytes: 2 << 20,
                },
            ],
        )
        .unwrap();
        p.preflight(&calibration, 16, &[16], 1 << 30, 0).unwrap();
        p
    }

    fn rb(p: &M) -> u64 {
        p.backend().budget().device_readbacks().0
    }

    fn mstep(p: &mut M, grad_scale: f64) -> Result<(), StepError> {
        let (lr_scale, wd) = per_entry(n_trainable(p));
        let next = p.step_count() + 1;
        p.adamw_step(&hyper(grad_scale), next, &lr_scale, &wd)
    }

    fn macc(p: &mut M, j: &Job, mode: BankMode) {
        p.accumulate(&j.view(), mode, None).unwrap();
    }

    #[test]
    fn the_readback_count_follows_the_data_not_the_calls() {
        let mut p = provider();
        let n = p.leaf_count() as u64;

        // No norm: the update reads each gradient once.
        macc(&mut p, &job(10, 1, 1.0), BankMode::Overwrite);
        let s = rb(&p);
        mstep(&mut p, 0.5).unwrap();
        assert_eq!(rb(&p) - s, n, "step without a norm");

        // A norm twice reads once; the update after them reads nothing.
        macc(&mut p, &job(10, 2, 1.0), BankMode::Overwrite);
        let s = rb(&p);
        p.grad_sq_norm().unwrap();
        let a = rb(&p);
        p.grad_sq_norm().unwrap();
        let b = rb(&p);
        mstep(&mut p, 0.5).unwrap();
        let c = rb(&p);
        assert_eq!((a - s, b - a, c - b), (n, 0, 0), "norm, norm, step");

        // norm, then data changes, then the update: the change forces a re-read.
        macc(&mut p, &job(10, 3, 1.0), BankMode::Overwrite);
        let s = rb(&p);
        p.grad_sq_norm().unwrap();
        let a = rb(&p);
        macc(&mut p, &job(9, 4, 1.0), BankMode::Add);
        let b = rb(&p);
        mstep(&mut p, 0.5).unwrap();
        let c = rb(&p);
        assert_eq!((a - s, c - b), (n, n), "norm, add, step must re-read once");

        // norm, add, norm, step: n, n, 0.
        macc(&mut p, &job(10, 5, 1.0), BankMode::Overwrite);
        let s = rb(&p);
        p.grad_sq_norm().unwrap();
        let a = rb(&p);
        macc(&mut p, &job(9, 6, 1.0), BankMode::Add);
        let a2 = rb(&p);
        p.grad_sq_norm().unwrap();
        let b = rb(&p);
        mstep(&mut p, 1.0).unwrap();
        let c = rb(&p);
        assert_eq!((a - s, b - a2, c - b), (n, n, 0), "norm, add, norm, step");

        // A refusal keeps the summary (the slots did not change): the retry
        // reads nothing for the scan, and the refusal itself reads a bounded
        // handful (the unsafe gradient, to locate the value).
        macc(&mut p, &job(10, 7, 1.0), BankMode::Overwrite);
        let s = rb(&p);
        assert!(mstep(&mut p, 1e30).is_err());
        let a = rb(&p);
        assert!(mstep(&mut p, 1e30).is_err());
        let b = rb(&p);
        mstep(&mut p, 1.0).unwrap();
        let c = rb(&p);
        eprintln!(
            "refusal readbacks: first {} second {} retry {}",
            a - s,
            b - a,
            c - b
        );
        assert!(a - s <= n + n, "first refusal read {}", a - s);
        assert!(b - a <= n, "second refusal read {}", b - a);
        assert_eq!(
            c - b,
            0,
            "the clean retry re-read after a refusal that changed nothing"
        );

        // Zero markers: no readback at all, for the norm or the update.
        macc(&mut p, &job(8, 3, 0.0), BankMode::Overwrite);
        let s = rb(&p);
        assert_eq!(p.grad_sq_norm().unwrap(), 0.0);
        mstep(&mut p, 0.5).unwrap();
        assert_eq!(rb(&p) - s, 0, "markers were read back");
    }
}
