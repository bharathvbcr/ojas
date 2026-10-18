//! `TapeStepProvider` (`ojas-model/src/qwen35/step_provider.rs`) on Metal.
//! `#[ignore]` and named `gpu_*`, as in `gpu_memory_rows.rs`: run it alone,
//! under the machine's heavy-job lock.
//!
//! ```text
//! cargo test -p ojas-qwen35 --release --test gpu_tape_provider \
//!     -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! # What is here
//!
//! - (a) `gpu_tape_provider_matches_the_cpu_over_accumulate_and_adamw_steps`:
//!   the tiny two-layer fixture (`ojas-model/tests/fixtures/qwen35_tiny`, rank
//!   4 / alpha 8 adapters from `qwen35_tiny_lora`), three optimizer steps of
//!   two sequences each (an outside loss on the first, a clip scale and
//!   per-entry learning-rate and weight-decay vectors), once through
//!   `TapeStepProvider<MetalBackend>` (freeze hook `freeze_checked`, rebind
//!   hook `with_budget`, a passing pre-flight) and once on `CpuBackend`. The
//!   bounds are `qwen35_lora_metal.rs`'s Metal-against-CPU ones: each step's
//!   loss within `2e-5` relative, each adapter entry within `2e-4` of its own
//!   peak (`METAL_GRAD_BOUND`, Metal's gated delta rule runs within `2e-4` of
//!   the CPU's).
//! - (b) `gpu_three_row_accumulate_matches_the_torch_fixture`: the CPU test's
//!   three sequences of unequal width over
//!   `ojas-model/tests/fixtures/qwen35_tiny_accum/accum.safetensors` (its
//!   header, `qwen35_step_provider.rs`, says what the file holds; the torch
//!   generator is run in the owner's batch), on Metal, at the Metal bounds
//!   above. A missing file fails loudly.
//! - (c) `gpu_a_non_finite_gradient_leaves_every_tensor_and_moment_unchanged`:
//!   the all-or-nothing AdamW on Metal. A NaN gradient poisoned into the second
//!   leaf (the first entry's gradient is finite, so a per-tensor loop would
//!   have moved it) leaves every parameter and moment bit for bit and the step
//!   count at 0, and a clean control provider does step.
//! - (d) `gpu_tape_provider_survives_kill_9_at_the_2b_shape`: kill and resume
//!   at the real Qwen3.5-2B-Base shape. Heavy; the snapshot is
//!   `QWEN35_2B_SNAPSHOT`, or the one snapshot of `Qwen/Qwen3.5-2B-Base` in the
//!   Hugging Face cache (as `gpu_tape_2b.rs`). See below.
//!
//! A missing Metal device fails (a, b, c) unless `OJAS_ALLOW_NO_GPU=1`, as in
//! `qwen35_lora_metal.rs`.
//!
//! # (d): how the kill is bounded
//!
//! The parent (the `--ignored` test) runs `QWEN35_KR_STEPS` (default 5)
//! optimizer steps uninterrupted in its own process, drops that provider, then
//! re-executes this test binary (`std::env::current_exe`) as a child that runs
//! the one test `gpu_tape_provider_kill_resume_child` under
//! `OJAS_KR_ROLE=child`. The child opens the provider, trains, and after every
//! step saves `<dir>/step-<s>/provider/...` and then publishes
//! `<dir>/step-<s>.done` (written under a temporary name and renamed, so a
//! marker names a checkpoint that is complete; it holds the child's losses
//! for steps `1..=s`, as f64 bit patterns). The parent polls for
//! `step-<QWEN35_KR_KILL_AFTER>.done` (default 2) every 25 ms and then
//! `Child::kill`s the child (SIGKILL), so the child is inside its next step.
//! Every wait has a deadline and none blocks forever:
//!
//! - the poll for the marker gives up after `CHILD_DEADLINE` (30 minutes: the
//!   base's sha256, the open, the pre-flight's dry run and the steps), killing
//!   the child, and returns early with the child's exit status if it exits
//!   (crashes) first (`try_wait` each tick);
//! - after the kill the parent reaps with `try_wait` for at most `REAP_DEADLINE`
//!   (60 seconds) and asserts the child died of signal 9, not of an exit;
//! - a `ChildGuard` kills and reaps the child if the parent panics anywhere
//!   after the spawn.
//!
//! The parent then resumes in a fresh provider from the highest marker present
//! (the child may have finished a step between the marker it waited for and
//! the kill; a resume point at the last step means the kill landed too late
//! and fails the test, so raise `QWEN35_KR_STEPS`), through
//! `load_state_with` under the same seed, epoch and consumed digest, and
//! asserts, bit for bit on the same backend: the child's losses up to the
//! resume point equal the uninterrupted run's, the resumed losses equal its
//! remaining losses, and the final adapters equal.
//!
//! # Not covered here
//!
//! Card criterion 5 (ft-b5e64557), the loss and gradient comparison against
//! the tessl provider, needs Lappi's `qd-train-metal` provider: it implements
//! `StepProvider` in Lappi, not in ojas, so it is deferred until after the pin
//! bump. (a) here plus the CPU fixture tests in
//! `ojas-model/tests/qwen35_step_provider.rs` cover the Tape side.

#![cfg(target_os = "macos")]

mod common;

use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ojas_core::{Backend, Budget, DType, Numerics, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::sha256::open_sha256_hex;
use ojas_io::SafeTensors;
use ojas_metal::MetalBackend;
use ojas_model::lora::{LoraConfig, LoraSite};
use ojas_model::preflight::{Calibration, Point};
use ojas_model::qwen35::adapters::BaseIdentity;
use ojas_model::qwen35::{
    lora_targets, DeviceHooks, LoraInit, LoraTarget, OpenConfig, Qwen35Lora, Qwen35Spec,
    TapeStepProvider, LORA_KINDS,
};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};
use ojas_qwen35::Snapshot;
use ojas_train::lappi::step::{
    AdamWHyper, BankMode, CheckpointContext, RowTargets, SequenceJob, StepError, StepProvider,
};

use common::{single_weights_file, snapshot_dir};

type P<B> = TapeStepProvider<B>;

const GIB: u64 = 1 << 30;
const RANK: usize = 4;
const ALPHA: f32 = 8.0;
/// `qwen35_lora_metal.rs`: Metal against the CPU, on the loss (`LORA_LOSS_BOUND`
/// is 1e-5, raised to 2e-5 there) and on a gradient or adapter, of its peak.
const METAL_LOSS_BOUND: f64 = 2e-5;
const METAL_GRAD_BOUND: f32 = 2e-4;
const TINY_WIDEST: usize = 16;

static DIRS: AtomicU64 = AtomicU64::new(0);

/// A scratch directory removed on drop.
struct Tmp(PathBuf);

impl Tmp {
    fn new() -> Self {
        let n = DIRS.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("ojas tape provider {}-{n}", std::process::id()));
        std::fs::remove_dir_all(&p).ok();
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.0) {
            eprintln!("could not remove {}: {e}", self.0.display());
        }
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn bits_all(v: &[Vec<f32>]) -> Vec<Vec<u32>> {
    v.iter().map(|x| bits(x)).collect()
}

fn peak_ratio(got: &[f32], want: &[f32], what: &str) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}");
    let peak = want.iter().fold(0.0f32, |p, x| p.max(x.abs()));
    assert!(peak > 0.0, "{what}: an all-zero reference checks nothing");
    got.iter()
        .zip(want)
        .map(|(a, b)| {
            assert!(a.is_finite(), "{what}: non-finite");
            (a - b).abs()
        })
        .fold(0.0f32, f32::max)
        / peak
}

// ---- the tiny fixture -------------------------------------------------------

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../ojas-model/tests/fixtures")
}

/// The tiny base, its spec and its LoRA targets.
struct Tiny {
    spec: Qwen35Spec,
    prefix: String,
    base_path: PathBuf,
    base_id: BaseIdentity,
    targets: Vec<LoraTarget>,
}

fn tiny() -> Tiny {
    let dir = fixtures().join("qwen35_tiny");
    let snap = Snapshot::from_dir(&dir).unwrap();
    let spec = snap.config.tape_spec();
    let prefix = snap.config.tower_prefix.clone();
    let base_path = single_weights_file(&snap.weights_path);
    let targets = {
        let file = SafeTensors::open(&base_path).unwrap();
        lora_targets(&spec, &file, &prefix, &LORA_KINDS).unwrap()
    };
    let base_id = base_identity(&base_path);
    Tiny {
        spec,
        prefix,
        base_path,
        base_id,
        targets,
    }
}

/// The base's file name and sha256: a Hugging Face cache blob is named by its
/// sha256, so a link to one is read (not hashed); any other file is hashed.
/// `load_hf_pinned` checks the claim against the file's bytes either way.
fn base_identity(path: &Path) -> BaseIdentity {
    let blob = std::fs::read_link(path).ok().and_then(|l| {
        l.file_name()
            .and_then(|n| n.to_str())
            .filter(|n| n.len() == 64 && n.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
            .map(str::to_string)
    });
    let sha256 = blob.unwrap_or_else(|| open_sha256_hex(path).unwrap().1);
    BaseIdentity {
        file_name: path.file_name().unwrap().to_string_lossy().into_owned(),
        sha256,
        quantizer: None,
    }
}

/// Rank 4 / alpha 8 adapters from `qwen35_tiny_lora` (`A` random, `B` non-zero),
/// host tensors.
fn tiny_adapters(t: &Tiny) -> Qwen35Lora<LoraSite<Tensor>> {
    let path = fixtures().join("qwen35_tiny_lora").join("lora.safetensors");
    let file = SafeTensors::open(&path).unwrap();
    let budget = Budget::new(GIB);
    let mut lora = Qwen35Lora::empty(&t.spec);
    for target in &t.targets {
        let m = target.module();
        let a = file
            .read_f32_widened(&format!("init.{m}.lora_A.weight"))
            .unwrap()
            .1;
        let fused_b = file
            .read_f32_widened(&format!("bnz.{m}.lora_B.weight"))
            .unwrap()
            .1;
        let a = Tensor::from_f32(&a, &[RANK, target.in_features], &budget).unwrap();
        let b = target
            .split_b(&t.spec, &fused_b, RANK)
            .into_iter()
            .zip(&target.out_parts)
            .map(|(v, &rows)| Tensor::from_f32(&v, &[rows, RANK], &budget).unwrap())
            .collect();
        *lora.slot_mut(target.layer, target.kind).unwrap() = Some(LoraSite { a, b });
    }
    lora
}

fn open_tiny<B: Backend>(backend: B, hooks: DeviceHooks<B>) -> P<B> {
    let t = tiny();
    let cfg = OpenConfig {
        spec: t.spec.clone(),
        prefix: t.prefix.clone(),
        base_path: t.base_path.clone(),
        base_id: t.base_id.clone(),
        linears: DType::F32,
        table: DType::F32,
        widest: TINY_WIDEST,
        chunk: DEFAULT_CE_CHUNK,
        activations: ActivationCheckpoint::Off,
    };
    let lora = LoraInit {
        config: LoraConfig::new(RANK, ALPHA, 0.0).unwrap(),
        adapters: tiny_adapters(&t),
        targets: t.targets,
    };
    TapeStepProvider::open_lora(backend, hooks, cfg, lora).unwrap()
}

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(4 << 30)).with_numerics(Numerics::Exact)
}

fn metal() -> Option<MetalBackend> {
    match MetalBackend::new(Budget::new(2 * GIB)) {
        Ok(m) => Some(m),
        Err(e) if std::env::var_os("OJAS_ALLOW_NO_GPU").is_some() => {
            eprintln!("no Metal device: {e:?}");
            None
        }
        Err(e) => panic!("no Metal device (set OJAS_ALLOW_NO_GPU=1 to skip): {e:?}"),
    }
}

/// The two hooks a Metal caller passes (`step_provider.rs`' module docs).
fn metal_hooks() -> DeviceHooks<MetalBackend> {
    DeviceHooks {
        freeze: Some(Box::new(|m: &MetalBackend, t: &Tensor| m.freeze_checked(t))),
        rebind: Some(Box::new(|m: &MetalBackend, b: Budget| m.with_budget(b))),
    }
}

/// A tiny provider on Metal that has passed its pre-flight.
fn metal_provider() -> Option<P<MetalBackend>> {
    let m = metal()?;
    let mut p = open_tiny(m, metal_hooks());
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
    let tokens = TINY_WIDEST as u64;
    p.preflight(&calibration, TINY_WIDEST, &[tokens], GIB, 0)
        .unwrap();
    Some(p)
}

/// An owned sequence job.
struct Job {
    tokens: Vec<u32>,
    positions: Vec<u32>,
    targets: Vec<u32>,
    hidden: Vec<u32>,
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
            hidden_at: &self.hidden,
        }
    }
}

/// `t` tokens, rows at the second half, next-token targets, and (optionally)
/// the outside loss reading the first and last position.
fn job(t: usize, salt: usize, scale: f32, with_hidden: bool) -> Job {
    let tokens: Vec<u32> = (0..t)
        .map(|i| ((i * 37 + salt * 101 + 11) % 8000) as u32)
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
        hidden: if with_hidden {
            vec![0, (t - 1) as u32]
        } else {
            Vec::new()
        },
        scale,
    }
}

fn outside(_: &[u32], hidden: &[f32]) -> Result<Vec<f32>, StepError> {
    Ok(hidden.iter().map(|x| 0.01 * x).collect())
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
        (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.5 })
            .collect(),
        (0..n)
            .map(|i| if i % 3 == 0 { 0.01 } else { 0.0 })
            .collect(),
    )
}

fn n_trainable<B: Backend>(p: &P<B>) -> usize {
    p.parameters().iter().filter(|s| s.trainable).count()
}

/// Optimizer steps `from..to`, two sequences each (the second with a smaller
/// scale and no outside loss), a clip scale of 0.5 and per-entry vectors.
/// Returns each step's loss sum.
fn run_steps<B: Backend>(p: &mut P<B>, from: u64, to: u64) -> Vec<f64> {
    let (lr_scale, wd) = per_entry(n_trainable(p));
    let mut losses = Vec::new();
    for s in from..to {
        let a = job(8 + (s as usize % 3), s as usize, 1.0, true);
        let b = job(12, s as usize + 50, 0.5, false);
        let mut ext = outside;
        let la = p
            .accumulate(&a.view(), BankMode::Overwrite, Some(&mut ext))
            .unwrap();
        let lb = p.accumulate(&b.view(), BankMode::Add, None).unwrap();
        losses.push(la + lb);
        let next = p.step_count() + 1;
        p.adamw_step(&hyper(0.5), next, &lr_scale, &wd).unwrap();
    }
    losses
}

// ---- (a) ---------------------------------------------------------------------

#[test]
#[ignore]
fn gpu_tape_provider_matches_the_cpu_over_accumulate_and_adamw_steps() {
    let Some(mut gpu) = metal_provider() else {
        return;
    };
    let mut host = open_tiny(cpu(), DeviceHooks::default());
    let on_gpu = run_steps(&mut gpu, 0, 3);
    let on_cpu = run_steps(&mut host, 0, 3);
    assert_eq!(gpu.step_count(), 3);
    assert_eq!(host.step_count(), 3);
    for (i, (g, c)) in on_gpu.iter().zip(&on_cpu).enumerate() {
        let r = (g - c).abs() / c.abs();
        eprintln!("step {i}: metal loss {g} vs cpu {c} (rel {r:.2e})");
        assert!(g.is_finite() && *c > 0.0, "step {i}: {g} vs {c}");
        assert!(r <= METAL_LOSS_BOUND, "step {i}: loss rel {r:.3e}");
    }
    let got = gpu.read_parameters().unwrap();
    let want = host.read_parameters().unwrap();
    let specs: Vec<_> = gpu.parameters().iter().filter(|s| s.trainable).collect();
    assert_eq!(got.len(), specs.len());
    assert_eq!(want.len(), specs.len());
    let mut worst = 0.0f32;
    for ((s, g), c) in specs.iter().zip(&got).zip(&want) {
        let r = peak_ratio(g, c, &s.name);
        eprintln!("{}: {r:.3e} of peak", s.name);
        assert!(r <= METAL_GRAD_BOUND, "{}: {r:.3e} of its peak", s.name);
        worst = worst.max(r);
    }
    eprintln!("worst adapter ratio {worst:.3e}");
}

// ---- (b) ---------------------------------------------------------------------

#[test]
#[ignore]
fn gpu_three_row_accumulate_matches_the_torch_fixture() {
    let Some(mut p) = metal_provider() else {
        return;
    };
    let path = fixtures()
        .join("qwen35_tiny_accum")
        .join("accum.safetensors");
    let file = SafeTensors::open(&path).unwrap_or_else(|e| {
        panic!(
            "{} is missing or unreadable ({}): generate the torch accumulation fixture described \
             in ojas-model/tests/qwen35_step_provider.rs",
            path.display(),
            e.detail()
        )
    });
    let ints = |name: &str| -> Vec<u32> {
        file.read_i64(name)
            .unwrap()
            .1
            .iter()
            .map(|&v| u32::try_from(v).unwrap())
            .collect()
    };
    let scale = file.read_f32("scale").unwrap().1[0];
    let hidden_width = tiny().spec.hidden;
    let (mut total, mut want_total) = (0.0f64, 0.0f64);
    let mut widths = Vec::new();
    for i in 0..3 {
        let tokens = ints(&format!("ids.{i}"));
        let positions = ints(&format!("positions.{i}"));
        let targets = ints(&format!("targets.{i}"));
        let hidden = ints(&format!("hidden_at.{i}"));
        let dh = file.read_f32(&format!("dh.{i}")).unwrap().1;
        assert_eq!(dh.len(), hidden.len() * hidden_width, "dh.{i}");
        widths.push(tokens.len());
        let job = SequenceJob {
            tokens: &tokens,
            rows: RowTargets {
                positions: &positions,
                targets: &targets,
                scale,
            },
            hidden_at: &hidden,
        };
        let mut calls = 0;
        let mut ext = |pos: &[u32], h: &[f32]| -> Result<Vec<f32>, StepError> {
            assert_eq!(pos, hidden.as_slice());
            assert_eq!(h.len(), dh.len());
            calls += 1;
            Ok(dh.clone())
        };
        let mode = if i == 0 {
            BankMode::Overwrite
        } else {
            BankMode::Add
        };
        let got = p.accumulate(&job, mode, Some(&mut ext)).unwrap();
        assert_eq!(calls, 1, "the outside loss is asked once per sequence");
        let want = f64::from(file.read_f32(&format!("loss.{i}")).unwrap().1[0]);
        let rel = (got - want).abs() / want.abs();
        assert!(
            rel <= METAL_LOSS_BOUND,
            "loss.{i}: {got} vs {want}, {rel:.3e}"
        );
        total += got;
        want_total += want;
    }
    assert!(
        widths[0] != widths[1] && widths[1] != widths[2] && widths[0] != widths[2],
        "the fixture's rows must have unequal widths: {widths:?}"
    );
    let rel = (total - want_total).abs() / want_total.abs();
    assert!(
        rel <= METAL_LOSS_BOUND,
        "summed loss {total} vs {want_total}"
    );

    let grads = p.read_gradients().unwrap();
    let specs: Vec<_> = p.parameters().iter().filter(|s| s.trainable).collect();
    assert_eq!(grads.len(), specs.len());
    let mut worst = 0.0f32;
    for (s, g) in specs.iter().zip(&grads) {
        let want = file.read_f32(&format!("grad.{}", s.name)).unwrap().1;
        let ratio = peak_ratio(g, &want, &s.name);
        eprintln!("{}: {ratio:.3e} of peak", s.name);
        assert!(
            ratio <= METAL_GRAD_BOUND,
            "{}: {ratio:.3e} of its peak",
            s.name
        );
        worst = worst.max(ratio);
    }
    eprintln!("worst gradient ratio {worst:.3e}");
}

// ---- (c) ---------------------------------------------------------------------

#[test]
#[ignore]
fn gpu_a_non_finite_gradient_leaves_every_tensor_and_moment_unchanged() {
    let Some(mut p) = metal_provider() else {
        return;
    };
    p.accumulate(&job(10, 1, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    let params = bits_all(&p.read_parameters().unwrap());
    let moments = |p: &mut P<MetalBackend>| -> Vec<(Vec<u32>, Vec<u32>)> {
        p.read_moments()
            .unwrap()
            .iter()
            .map(|(m, v)| (bits(m), bits(v)))
            .collect()
    };
    let moments_before = moments(&mut p);

    // Leaf 1 is the first B part: entry 0's gradient is finite, so a loop of
    // per-tensor steps would have moved entry 0 before reaching it.
    let shape = p.bank().entries()[1].shape.clone();
    let n: usize = shape.iter().product();
    let device = p.backend().clone();
    let host = Tensor::from_f32(&vec![f32::NAN; n], &shape, &Budget::new(GIB)).unwrap();
    let nan = device.upload(&host).unwrap();
    p.bank_mut()
        .accumulate(&device, 1, BankMode::Overwrite, nan)
        .unwrap();
    let (lr_scale, wd) = per_entry(n_trainable(&p));
    let err = p.adamw_step(&hyper(1.0), 1, &lr_scale, &wd).unwrap_err();
    assert!(
        err.to_string().contains("no tensor was stepped"),
        "{err}"
    );
    assert_eq!(bits_all(&p.read_parameters().unwrap()), params);
    assert_eq!(moments(&mut p), moments_before);
    assert_eq!(p.step_count(), 0);

    // The control: the same provider with a clean gradient does step.
    let mut q = metal_provider().unwrap();
    q.accumulate(&job(10, 1, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    q.adamw_step(&hyper(1.0), 1, &lr_scale, &wd).unwrap();
    assert_eq!(q.step_count(), 1);
    assert_ne!(bits_all(&q.read_parameters().unwrap()), params);
}

// ---- (d) ---------------------------------------------------------------------

const ROLE_ENV: &str = "OJAS_KR_ROLE";
const DIR_ENV: &str = "OJAS_KR_DIR";
const STEPS_ENV: &str = "QWEN35_KR_STEPS";
const KILL_AFTER_ENV: &str = "QWEN35_KR_KILL_AFTER";
const TOKENS_ENV: &str = "QWEN35_KR_TOKENS";
const CHILD_TEST: &str = "gpu_tape_provider_kill_resume_child";
/// The child's open, pre-flight and first steps at the 2B shape, all in.
const CHILD_DEADLINE: Duration = Duration::from_secs(30 * 60);
/// A SIGKILLed process is reaped at once; this is the bound on "at once".
const REAP_DEADLINE: Duration = Duration::from_secs(60);
const POLL: Duration = Duration::from_millis(25);
const KR_SEED: u64 = 7;
const KR_LORA_SEED: u64 = 0x10_4a;
const KR_DEVICE_BUDGET: u64 = 28 * GIB;

fn env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|e| panic!("{name} must be an integer, got {v:?}: {e}")),
        Err(std::env::VarError::NotPresent) => default,
        Err(e) => panic!("{name}: {e}"),
    }
}

/// The checkpoint context after `done` optimizer steps: the digest changes
/// with the step, so a resume from the wrong directory is refused.
fn kr_context(done: u64) -> CheckpointContext {
    CheckpointContext {
        optimizer_step: done,
        seed: KR_SEED,
        epoch: 0,
        consumed_digest: format!("{done:064x}"),
    }
}

/// The 2B provider on Metal under the 28 GiB budget, pre-flighted at `tokens`
/// (the calibration is nominal, so the gate is the dry run under the child
/// budget, which charges against the 28 GiB cap).
fn open_2b(tokens: usize) -> P<MetalBackend> {
    let dir = snapshot_dir("QWEN35_2B_SNAPSHOT");
    let snap = Snapshot::from_dir(&dir).unwrap();
    let spec = snap.config.tape_spec();
    let prefix = snap.config.tower_prefix.clone();
    let base_path = single_weights_file(&snap.weights_path);
    let host_budget = Budget::new(24 * GIB);
    let config = LoraConfig::new(16, 16.0, 0.0).unwrap();
    let (targets, adapters) = {
        let file = SafeTensors::open(&base_path).unwrap();
        let targets = lora_targets(&spec, &file, &prefix, &LORA_KINDS).unwrap();
        let adapters = Qwen35Lora::init(&spec, &targets, &config, KR_LORA_SEED, &host_budget)
            .unwrap();
        (targets, adapters)
    };
    let backend = MetalBackend::new(Budget::new(KR_DEVICE_BUDGET)).unwrap();
    let cfg = OpenConfig {
        spec,
        prefix,
        base_id: base_identity(&base_path),
        base_path,
        linears: DType::Bf16,
        table: DType::Bf16,
        widest: tokens,
        chunk: DEFAULT_CE_CHUNK,
        activations: ActivationCheckpoint::Blocks,
    };
    let mut p = TapeStepProvider::open_lora(
        backend,
        metal_hooks(),
        cfg,
        LoraInit {
            config,
            targets,
            adapters,
        },
    )
    .unwrap();
    let calibration = Calibration::from_points(
        "qwen3.5-2b-base nominal",
        8 * GIB,
        &[
            Point {
                tokens: (tokens / 2).max(1) as u64,
                peak_bytes: 10 * GIB,
            },
            Point {
                tokens: tokens as u64,
                peak_bytes: 12 * GIB,
            },
        ],
    )
    .unwrap();
    p.preflight(&calibration, tokens, &[tokens as u64], KR_DEVICE_BUDGET, 0)
        .unwrap();
    p
}

/// `n + 1` deterministic ids below 150000 (ordinary text tokens) for `step`.
fn kr_ids(step: u64, n: usize) -> Vec<u32> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64 ^ step.wrapping_mul(0xd1b5_4a32_d192_ed03);
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n + 1 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.push((s % 150_000) as u32);
    }
    out
}

/// Optimizer step `step` (0-based): one sequence of `tokens` tokens, every
/// position scored, then AdamW. Returns the loss sum.
fn kr_step(p: &mut P<MetalBackend>, step: u64, tokens: usize) -> f64 {
    let ids = kr_ids(step, tokens);
    let positions: Vec<u32> = (0..tokens as u32).collect();
    let job = SequenceJob {
        tokens: &ids[..tokens],
        rows: RowTargets {
            positions: &positions,
            targets: &ids[1..],
            scale: 1.0 / tokens as f32,
        },
        hidden_at: &[],
    };
    let loss = p.accumulate(&job, BankMode::Overwrite, None).unwrap();
    let entries = n_trainable(p);
    let hyper = AdamWHyper {
        lr: 1e-4,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: 1.0,
    };
    let next = p.step_count() + 1;
    p.adamw_step(&hyper, next, &vec![1.0; entries], &vec![0.01; entries])
        .unwrap();
    loss
}

fn marker(dir: &Path, step: u64) -> PathBuf {
    dir.join(format!("step-{step}.done"))
}

fn checkpoint_dir(dir: &Path, step: u64) -> PathBuf {
    dir.join(format!("step-{step}"))
}

/// The child: train, and after every step publish a complete checkpoint, then
/// a marker holding the losses so far. Run by the parent under
/// `OJAS_KR_ROLE=child`; with no role it does nothing, so the plain
/// `--ignored` run passes over it.
#[test]
#[ignore]
fn gpu_tape_provider_kill_resume_child() {
    match std::env::var(ROLE_ENV).as_deref() {
        Ok("child") => {}
        Err(std::env::VarError::NotPresent) => {
            eprintln!("{CHILD_TEST}: not the child ({ROLE_ENV} unset), nothing to do");
            return;
        }
        other => panic!("{ROLE_ENV} is unset or child, got {other:?}"),
    }
    let dir = PathBuf::from(std::env::var_os(DIR_ENV).unwrap_or_else(|| panic!("{DIR_ENV}")));
    let steps = env_u64(STEPS_ENV, 5);
    let tokens = env_u64(TOKENS_ENV, 128) as usize;
    let mut p = open_2b(tokens);
    let mut losses: Vec<f64> = Vec::new();
    for s in 0..steps {
        losses.push(kr_step(&mut p, s, tokens));
        let done = s + 1;
        let ckpt = checkpoint_dir(&dir, done);
        std::fs::create_dir_all(&ckpt).unwrap();
        p.save_state_with(&ckpt, &kr_context(done)).unwrap();
        let body: String = losses.iter().map(|l| format!("{:016x}\n", l.to_bits())).collect();
        let tmp = dir.join(format!("step-{done}.done.tmp"));
        std::fs::write(&tmp, body).unwrap();
        std::fs::rename(&tmp, marker(&dir, done)).unwrap();
        eprintln!("child: step {done} checkpointed");
    }
}

/// Kills and reaps its child if the parent unwinds before it did.
struct ChildGuard(Option<Child>);

impl ChildGuard {
    fn child(&mut self) -> &mut Child {
        self.0.as_mut().expect("the child was already reaped")
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            if let Err(e) = c.kill() {
                eprintln!("child {}: kill on cleanup: {e}", c.id());
            }
            if let Err(e) = reap(&mut c, REAP_DEADLINE) {
                eprintln!("child {}: {e}", c.id());
            }
        }
    }
}

/// `child`'s exit status, polling `try_wait` for at most `deadline`.
fn reap(child: &mut Child, deadline: Duration) -> Result<ExitStatus, String> {
    let until = Instant::now() + deadline;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) if Instant::now() >= until => {
                return Err(format!("not reaped within {deadline:?}"))
            }
            Ok(None) => std::thread::sleep(POLL),
            Err(e) => return Err(format!("try_wait: {e}")),
        }
    }
}

fn read_losses(path: &Path) -> Vec<u64> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .map(|l| u64::from_str_radix(l, 16).unwrap())
        .collect()
}

#[test]
#[ignore]
fn gpu_tape_provider_survives_kill_9_at_the_2b_shape() {
    use std::os::unix::process::ExitStatusExt;

    let steps = env_u64(STEPS_ENV, 5);
    let kill_after = env_u64(KILL_AFTER_ENV, 2);
    let tokens = env_u64(TOKENS_ENV, 128) as usize;
    assert!(
        kill_after >= 1 && kill_after + 1 < steps,
        "{KILL_AFTER_ENV} ({kill_after}) must leave at least two steps after it of \
         {STEPS_ENV} ({steps})"
    );

    // The uninterrupted run, in this process; its provider is gone before the
    // child opens its own.
    let (reference, ref_params) = {
        let mut p = open_2b(tokens);
        let losses: Vec<u64> = (0..steps)
            .map(|s| kr_step(&mut p, s, tokens).to_bits())
            .collect();
        (losses, bits_all(&p.read_parameters().unwrap()))
    };
    assert!(
        reference.iter().all(|b| f64::from_bits(*b).is_finite()),
        "the uninterrupted run has a non-finite loss"
    );

    let tmp = Tmp::new();
    let exe = std::env::current_exe().unwrap();
    let mut guard = ChildGuard(Some(
        Command::new(exe)
            .args([
                "--ignored",
                "--exact",
                CHILD_TEST,
                "--test-threads=1",
                "--nocapture",
            ])
            .env(ROLE_ENV, "child")
            .env(DIR_ENV, &tmp.0)
            .env(STEPS_ENV, steps.to_string())
            .env(TOKENS_ENV, tokens.to_string())
            .spawn()
            .unwrap(),
    ));

    // Wait for the checkpoint of step `kill_after`, bounded, and stop early if
    // the child exits first.
    let until = Instant::now() + CHILD_DEADLINE;
    while !marker(&tmp.0, kill_after).is_file() {
        if let Some(status) = guard.child().try_wait().unwrap() {
            panic!("the child exited ({status}) before checkpointing step {kill_after}");
        }
        assert!(
            Instant::now() < until,
            "the child did not checkpoint step {kill_after} within {CHILD_DEADLINE:?}"
        );
        std::thread::sleep(POLL);
    }
    guard.child().kill().unwrap();
    let status = reap(guard.child(), REAP_DEADLINE).unwrap();
    guard.0 = None;
    assert_eq!(
        status.signal(),
        Some(9),
        "the child was not killed by SIGKILL: {status}"
    );

    // The last complete checkpoint: the child may have finished more steps
    // between the marker waited for and the kill.
    let resume = (kill_after..=steps)
        .rev()
        .find(|s| marker(&tmp.0, *s).is_file())
        .unwrap();
    assert!(
        resume < steps,
        "the child finished all {steps} steps before the kill: raise {STEPS_ENV}"
    );
    eprintln!("killed after step {kill_after}; resuming from step {resume} of {steps}");

    let child_losses = read_losses(&marker(&tmp.0, resume));
    assert_eq!(child_losses.len() as u64, resume);
    assert_eq!(
        child_losses,
        reference[..resume as usize],
        "the child's losses before the kill differ from the uninterrupted run's"
    );

    let mut resumed = open_2b(tokens);
    let fresh = bits_all(&resumed.read_parameters().unwrap());
    resumed
        .load_state_with(&checkpoint_dir(&tmp.0, resume), &kr_context(resume))
        .unwrap();
    assert_eq!(resumed.step_count(), resume);
    assert_ne!(
        bits_all(&resumed.read_parameters().unwrap()),
        fresh,
        "the load left the freshly initialised adapters"
    );
    // A checkpoint is the step's, not another's: the digest names the step.
    assert!(resumed
        .load_state_with(&checkpoint_dir(&tmp.0, resume), &kr_context(resume - 1))
        .is_err());
    assert_eq!(resumed.step_count(), resume, "a refused load moved state");

    let rest: Vec<u64> = (resume..steps)
        .map(|s| kr_step(&mut resumed, s, tokens).to_bits())
        .collect();
    assert_eq!(
        rest,
        reference[resume as usize..],
        "the resumed loss trajectory differs from the uninterrupted run's"
    );
    assert_eq!(resumed.step_count(), steps);
    assert_eq!(
        bits_all(&resumed.read_parameters().unwrap()),
        ref_params,
        "the final adapters differ from the uninterrupted run's"
    );
}
