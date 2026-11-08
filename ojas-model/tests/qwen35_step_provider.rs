//! `TapeStepProvider` (`ojas-model/src/qwen35/step_provider.rs`): the Qwen3.5
//! Tape tower as a training loop's `StepProvider`, CPU, on the widened tiny
//! fixture (`qwen35_tiny`, rank 4 / alpha 8 adapters from `qwen35_tiny_lora`).
//!
//! # The torch accumulation fixture
//!
//! `three_row_accumulate_matches_the_torch_fixture` reads
//! `tests/fixtures/qwen35_tiny_accum/accum.safetensors`, which a torch
//! generator (run in the owner's batch, not here) writes over the same base
//! and the same adapters (`qwen35_tiny_lora`'s `init.*.lora_A.weight` and
//! `bnz.*.lora_B.weight`, rank 4, alpha 8). Three sequences of UNEQUAL width,
//! `i = 0, 1, 2`, each its own forward and `backward()`, summed:
//!
//! - `ids.{i}` I64 `[T_i]`: the tokens, positions from 0.
//! - `positions.{i}`, `targets.{i}` I64 `[n_i]`: the scored rows; row `j`
//!   predicts `targets[j]` from the hidden state at `positions[j]` through the
//!   tied embedding (full vocabulary, no ignore index).
//! - `hidden_at.{i}` I64 `[m_i]` (distinct) and `dh.{i}` F32 `[m_i, hidden]`:
//!   the outside loss `sum <dh, H[hidden_at]>` over the final-norm output.
//! - `scale` F32 `[1]`: the row scale, the same for every sequence.
//! - `loss.{i}` F32 `[1]`: sequence `i`'s SUM of its `n_i` row cross-entropies.
//! - `grad.{name}` F32: for every trainable entry, under its PEFT name
//!   (`<module>.lora_A.weight` `[r, in]`, `<module>.lora_B.weight` the FUSED
//!   `[out, r]`), the gradient of `sum_i (scale * loss_i + <dh_i, H_i[..]>)`.
//!
//! A missing file fails the test loudly; it is never skipped.

mod qwen35_common;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use ojas_core::{
    forward_backend_methods, BackendCore, BackendFloat, BackendId, BackendInt, BackendModule,
    DType, Numerics, Tensor,
};
use ojas_cpu::CpuBackend;
use ojas_io::sha256::open_sha256_hex;
use ojas_io::SafeTensors;
use ojas_model::lora::LoraConfig;
use ojas_model::preflight::{Calibration, Point};
use ojas_model::qwen35::adapters::{lora_a_name, lora_b_name, BaseIdentity};
use ojas_model::qwen35::{hf_tensors, DeviceHooks, LoraInit, OpenConfig, TapeStepProvider};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};
use ojas_train::lappi::step::{
    AdamWHyper, BankMode, CheckpointContext, RowTargets, SequenceJob, StepError, StepProvider,
};
use qwen35_common::lora::*;
use qwen35_common::*;

type P = TapeStepProvider<CpuBackend>;

const BASE_FILE: &str = "model.safetensors";

static DIRS: AtomicU64 = AtomicU64::new(0);

struct Tmp(PathBuf);

impl Tmp {
    fn new() -> Self {
        let n = DIRS.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("ojas step provider {}-{n}", std::process::id()));
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

fn open(widest: usize, nonzero_b: bool) -> P {
    TapeStepProvider::open_lora(
        cpu(),
        DeviceHooks::default(),
        open_config(widest),
        lora_init(nonzero_b),
    )
    .unwrap()
}

fn expect_err(r: Result<P, StepError>) -> StepError {
    match r {
        Err(e) => e,
        Ok(_) => panic!("the provider opened"),
    }
}

fn n_trainable(p: &P) -> usize {
    p.parameters().iter().filter(|s| s.trainable).count()
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
        (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.5 }).collect(),
        (0..n)
            .map(|i| if i % 3 == 0 { 0.01 } else { 0.0 })
            .collect(),
    )
}

/// Optimizer steps `from..to`, two sequences each (the second with a smaller
/// scale and no outside loss), a clip scale of 0.5 and per-entry vectors.
/// Returns each step's loss sum.
fn run_steps(p: &mut P, from: u64, to: u64) -> Vec<f64> {
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

#[test]
fn three_row_accumulate_matches_the_torch_fixture() {
    let path = dir()
        .parent()
        .unwrap()
        .join("qwen35_tiny_accum")
        .join("accum.safetensors");
    let file = SafeTensors::open(&path).unwrap_or_else(|e| {
        panic!(
            "{} is missing or unreadable ({}): generate the torch accumulation fixture described \
             in this file's header",
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
    let hidden_width = spec().hidden;
    let mut p = open(16, true);
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
            rel <= LORA_LOSS_BOUND,
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
        rel <= LORA_LOSS_BOUND,
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
            ratio <= LORA_GRAD_BOUND,
            "{}: {ratio:.3e} of its peak",
            s.name
        );
        worst = worst.max(ratio);
    }
    eprintln!("worst gradient ratio {worst:.3e}");
}

#[test]
fn the_table_lists_the_frozen_base_then_the_fused_adapters_and_read_parameters_fuses_b() {
    let mut p = open(16, true);
    let t = targets();
    let hf = hf_tensors(&spec(), PREFIX);
    assert_eq!(p.parameters().len(), hf.len() + 2 * t.len());
    for (entry, h) in p.parameters().iter().zip(&hf) {
        assert!(!entry.trainable, "{} is frozen", entry.name);
        assert_eq!(entry.name, h.name);
        assert_eq!(entry.shape, h.shape);
    }
    assert_eq!(n_trainable(&p), 2 * t.len());

    let file = lora_file();
    let params = p.read_parameters().unwrap();
    assert_eq!(params.len(), 2 * t.len());
    let trainable: Vec<_> = p.parameters().iter().filter(|s| s.trainable).collect();
    for (i, target) in t.iter().enumerate() {
        let m = target.module();
        assert_eq!(trainable[2 * i].name, lora_a_name(m));
        assert_eq!(trainable[2 * i + 1].name, lora_b_name(m));
        let out: usize = target.out_parts.iter().sum();
        assert_eq!(trainable[2 * i + 1].shape, vec![out, RANK]);
        assert_eq!(
            bits(&params[2 * i]),
            bits(&read(&file, &format!("init.{m}.lora_A.weight"))),
            "{m}: A"
        );
        // A fused B read back is the fused B the adapters were built from,
        // however many parts the model holds it as.
        assert_eq!(
            bits(&params[2 * i + 1]),
            bits(&read(&file, &format!("bnz.{m}.lora_B.weight"))),
            "{m}: fused B"
        );
    }
    assert!(
        t.iter().any(|x| x.out_parts.len() > 1),
        "the fixture has a fused B"
    );
}

#[test]
fn a_non_finite_gradient_in_a_later_entry_leaves_every_tensor_and_moment_unchanged() {
    let mut p = open(16, true);
    p.accumulate(&job(10, 1, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    let params = bits_all(&p.read_parameters().unwrap());
    let moments: Vec<(Vec<u32>, Vec<u32>)> = p
        .read_moments()
        .unwrap()
        .iter()
        .map(|(m, v)| (bits(m), bits(v)))
        .collect();

    // Leaf 1 is the first B part: entry 0's gradient is finite, so a loop of
    // per-tensor steps would have moved entry 0 before reaching it.
    let shape = p.bank().entries()[1].shape.clone();
    let n: usize = shape.iter().product();
    let nan = Tensor::from_f32(&vec![f32::NAN; n], &shape, &budget()).unwrap();
    p.bank_mut()
        .accumulate(&cpu(), 1, BankMode::Overwrite, nan)
        .unwrap();
    let (lr_scale, wd) = per_entry(n_trainable(&p));
    let err = p.adamw_step(&hyper(1.0), 1, &lr_scale, &wd).unwrap_err();
    assert!(err.to_string().contains("no tensor was stepped"), "{err}");
    assert_eq!(bits_all(&p.read_parameters().unwrap()), params);
    let after: Vec<(Vec<u32>, Vec<u32>)> = p
        .read_moments()
        .unwrap()
        .iter()
        .map(|(m, v)| (bits(m), bits(v)))
        .collect();
    assert_eq!(after, moments);
    assert_eq!(p.step_count(), 0);

    // The control: the same provider with a clean gradient does step.
    let mut q = open(16, true);
    q.accumulate(&job(10, 1, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    q.adamw_step(&hyper(1.0), 1, &lr_scale, &wd).unwrap();
    assert_eq!(q.step_count(), 1);
    assert_ne!(bits_all(&q.read_parameters().unwrap()), params);
}

#[test]
fn scale_zero_returns_the_value_runs_no_backward_and_allocates_no_vocab_gradient() {
    let mut p = open(16, true);
    let budget = p.backend().budget().clone();
    let zero = job(8, 3, 0.0, false);
    let one = Job {
        scale: 1.0,
        ..job(8, 3, 0.0, false)
    };

    let live = budget.live_bytes().unwrap();
    budget.reset_peak();
    let v0 = p
        .accumulate(&zero.view(), BankMode::Overwrite, None)
        .unwrap();
    let delta_zero = budget.peak_bytes().saturating_sub(live);
    // No backward ran: the bank holds zeros, and it is complete, so an
    // Overwrite replaced whatever the last step left.
    assert!(p.bank().is_complete());
    for g in p.read_gradients().unwrap() {
        assert!(g.iter().all(|x| *x == 0.0));
    }
    assert_eq!(p.grad_sq_norm().unwrap(), 0.0);

    // The same value as the node-recording path, bit for bit.
    let live = budget.live_bytes().unwrap();
    budget.reset_peak();
    let v1 = p
        .accumulate(&one.view(), BankMode::Overwrite, None)
        .unwrap();
    let delta_one = budget.peak_bytes().saturating_sub(live);
    assert_eq!(v0.to_bits(), v1.to_bits());
    assert!(v1 > 0.0);
    // The forward activations and the fused-CE chunk scratch alone exceed one
    // [V, d], so the bound is relative: on the same input and provider the
    // scale-0 step must peak strictly below the scale-1 step, which runs the
    // backward. (In LoRA mode the tied head is frozen, so neither step
    // allocates a [V, d] gradient; the zero-gradient bank above is the
    // evidence for the scale-0 step.)
    assert!(
        delta_zero < delta_one,
        "a scale-0 step charged {delta_zero} bytes, not below the scale-1 step's {delta_one}"
    );

    // A scale-0 sequence adds nothing to a sum already in the bank.
    let before = bits_all(&p.read_gradients().unwrap());
    p.accumulate(&zero.view(), BankMode::Add, None).unwrap();
    assert_eq!(bits_all(&p.read_gradients().unwrap()), before);
    assert!(p.grad_sq_norm().unwrap() > 0.0);
}

#[test]
fn six_straight_steps_equal_three_then_a_checkpoint_a_fresh_provider_and_three_more() {
    let mut straight = open(16, true);
    let straight_losses = run_steps(&mut straight, 0, 6);

    let tmp = Tmp::new();
    let mut first = open(16, true);
    let first_losses = run_steps(&mut first, 0, 3);
    let ctx = CheckpointContext {
        optimizer_step: 3,
        seed: 7,
        epoch: 0,
        consumed_digest: "ab".repeat(32),
    };
    let files = first.save_state_with(&tmp.0, &ctx).unwrap();
    assert!(
        !files.is_empty() && files.iter().all(|f| f.is_file()),
        "{files:?}"
    );

    // A fresh provider over different adapters (B zero): the load is what
    // makes it the checkpointed one.
    let mut resumed = open(16, false);
    assert_ne!(
        bits_all(&resumed.read_parameters().unwrap()),
        bits_all(&first.read_parameters().unwrap())
    );
    for (what, wrong) in [
        (
            "step",
            CheckpointContext {
                optimizer_step: 2,
                ..ctx.clone()
            },
        ),
        (
            "seed",
            CheckpointContext {
                seed: 8,
                ..ctx.clone()
            },
        ),
        (
            "epoch",
            CheckpointContext {
                epoch: 1,
                ..ctx.clone()
            },
        ),
        (
            "digest",
            CheckpointContext {
                consumed_digest: "cd".repeat(32),
                ..ctx.clone()
            },
        ),
    ] {
        assert!(
            resumed.load_state_with(&tmp.0, &wrong).is_err(),
            "a checkpoint resumed under another {what}"
        );
        assert_eq!(
            resumed.step_count(),
            0,
            "{what}: a refused load moved state"
        );
    }
    resumed.load_state_with(&tmp.0, &ctx).unwrap();
    assert_eq!(resumed.step_count(), 3);
    assert_eq!(
        bits_all(&resumed.read_parameters().unwrap()),
        bits_all(&first.read_parameters().unwrap())
    );
    let rest = run_steps(&mut resumed, 3, 6);

    let mut stitched = first_losses;
    stitched.extend(rest);
    let a: Vec<u64> = straight_losses.iter().map(|x| x.to_bits()).collect();
    let b: Vec<u64> = stitched.iter().map(|x| x.to_bits()).collect();
    assert_eq!(a, b, "the loss trajectories differ");
    assert_eq!(straight.step_count(), 6);
    assert_eq!(resumed.step_count(), 6);
    assert_eq!(
        bits_all(&straight.read_parameters().unwrap()),
        bits_all(&resumed.read_parameters().unwrap())
    );
    let moments = |p: &mut P| -> Vec<(Vec<u32>, Vec<u32>)> {
        p.read_moments()
            .unwrap()
            .iter()
            .map(|(m, v)| (bits(m), bits(v)))
            .collect()
    };
    assert_eq!(moments(&mut straight), moments(&mut resumed));
}

#[test]
fn a_checkpoint_of_another_base_or_rank_is_refused() {
    let tmp = Tmp::new();
    let mut p = open(16, true);
    run_steps(&mut p, 0, 1);
    let ctx = CheckpointContext {
        optimizer_step: 1,
        seed: 1,
        epoch: 0,
        consumed_digest: "00".repeat(32),
    };
    p.save_state_with(&tmp.0, &ctx).unwrap();

    let mut other_base = open_config(16);
    other_base.base_id.sha256 = "0".repeat(64);
    let refused = expect_err(TapeStepProvider::open_lora(
        cpu(),
        DeviceHooks::default(),
        other_base,
        lora_init(true),
    ));
    assert!(refused.to_string().contains("sha256"), "{refused}");

    // The saved rank is 4, alpha 8: another alpha is another run.
    let mut same = open(16, true);
    let mut other = lora_init(true);
    other.config = LoraConfig::new(RANK, 4.0, 0.0).unwrap();
    let mut other_alpha =
        TapeStepProvider::open_lora(cpu(), DeviceHooks::default(), open_config(16), other).unwrap();
    let err = other_alpha.load_state_with(&tmp.0, &ctx).unwrap_err();
    assert!(err.to_string().contains("LoRA"), "{err}");
    same.load_state_with(&tmp.0, &ctx).unwrap();
}

#[test]
fn refusals_are_loud_and_leave_the_step_intact() {
    let mut p = open(16, true);
    let tmp = Tmp::new();
    assert!(matches!(
        p.save_state(&tmp.0),
        Err(StepError::Unsupported(_))
    ));
    assert!(matches!(
        p.load_state(&tmp.0),
        Err(StepError::Unsupported(_))
    ));

    let mut dropout = lora_init(true);
    dropout.config = LoraConfig::new(RANK, 8.0, 0.1).unwrap();
    let e = expect_err(TapeStepProvider::open_lora(
        cpu(),
        DeviceHooks::default(),
        open_config(16),
        dropout,
    ));
    assert!(e.to_string().contains("dropout"), "{e}");

    // Hidden positions with no callback, and a callback of the wrong size.
    let with_hidden = job(8, 2, 1.0, true);
    assert!(matches!(
        p.accumulate(&with_hidden.view(), BankMode::Overwrite, None),
        Err(StepError::Contract(_))
    ));
    let mut short =
        |_: &[u32], h: &[f32]| -> Result<Vec<f32>, StepError> { Ok(vec![0.0; h.len() - 1]) };
    assert!(matches!(
        p.accumulate(&with_hidden.view(), BankMode::Overwrite, Some(&mut short)),
        Err(StepError::Contract(_))
    ));
    let mut nan =
        |_: &[u32], h: &[f32]| -> Result<Vec<f32>, StepError> { Ok(vec![f32::NAN; h.len()]) };
    assert!(matches!(
        p.accumulate(&with_hidden.view(), BankMode::Overwrite, Some(&mut nan)),
        Err(StepError::Contract(_))
    ));
    // A sequence wider than the tables.
    let wide = job(17, 2, 1.0, false);
    assert!(matches!(
        p.accumulate(&wide.view(), BankMode::Overwrite, None),
        Err(StepError::Contract(_))
    ));
    // An Add with no Overwrite this step.
    assert!(p
        .accumulate(&job(8, 2, 1.0, false).view(), BankMode::Add, None)
        .is_err());

    // A wrong step number is refused before the bank is touched.
    p.accumulate(&job(8, 2, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    let (lr_scale, wd) = per_entry(n_trainable(&p));
    assert!(matches!(
        p.adamw_step(&hyper(1.0), 2, &lr_scale, &wd),
        Err(StepError::Contract(_))
    ));
    assert!(matches!(
        p.adamw_step(&hyper(1.0), 1, &lr_scale[1..], &wd),
        Err(StepError::Contract(_))
    ));
    assert!(p.bank().is_complete());
    p.adamw_step(&hyper(1.0), 1, &lr_scale, &wd).unwrap();
    assert_eq!(p.step_count(), 1);
}

#[test]
fn the_full_provider_trains_every_hugging_face_tensor() {
    let mut p =
        TapeStepProvider::open_full(cpu(), DeviceHooks::default(), open_config(16)).unwrap();
    let hf = hf_tensors(&spec(), PREFIX);
    assert_eq!(p.parameters().len(), hf.len());
    assert_eq!(n_trainable(&p), hf.len());
    let before = p.read_parameters().unwrap();
    for (values, h) in before.iter().zip(&hf) {
        assert_eq!(
            values.len(),
            h.shape.iter().product::<usize>(),
            "{}",
            h.name
        );
    }
    let a = job(10, 4, 1.0, true);
    let mut ext = outside;
    let loss = p
        .accumulate(&a.view(), BankMode::Overwrite, Some(&mut ext))
        .unwrap();
    assert!(loss.is_finite() && loss > 0.0);
    assert!(p.grad_sq_norm().unwrap() > 0.0);
    let (lr_scale, wd) = per_entry(n_trainable(&p));
    p.adamw_step(&hyper(1.0), 1, &lr_scale, &wd).unwrap();
    let after = p.read_parameters().unwrap();
    assert_ne!(
        bits(&after[0]),
        bits(&before[0]),
        "the embedding did not move"
    );
    // A full fine-tune has no adapter checkpoint.
    assert!(matches!(
        p.save_state_with(
            &Tmp::new().0,
            &CheckpointContext {
                optimizer_step: 1,
                seed: 0,
                epoch: 0,
                consumed_digest: "00".to_string(),
            }
        ),
        Err(StepError::Unsupported(_))
    ));
}

/// The CPU backend behind a device's id: every op is the CPU's, the id is
/// not, so the provider sees a backend that is not the CPU.
struct DeviceLike(CpuBackend);

macro_rules! all_but_id {
    (id, { $($item:tt)* }) => {};
    ($name:ident, { $($item:tt)* }) => {
        $($item)*
    };
}

impl BackendCore for DeviceLike {
    ojas_core::with_backend_methods!(forward_backend_methods, { self, self.0, none, all_but_id, core});

    fn id(&self) -> BackendId {
        BackendId::Metal
    }
}

impl BackendFloat for DeviceLike {
    ojas_core::with_backend_methods!(forward_backend_methods, { self, self.0, none, all_but_id, float});
}

impl BackendInt for DeviceLike {
    ojas_core::with_backend_methods!(forward_backend_methods, { self, self.0, none, all_but_id, int});
}

impl BackendModule for DeviceLike {
    ojas_core::with_backend_methods!(forward_backend_methods, { self, self.0, none, all_but_id, module});
}

fn device_like() -> DeviceLike {
    DeviceLike(CpuBackend::new(ojas_core::Budget::new(4 << 30)).with_numerics(Numerics::Exact))
}

fn device_provider(hooks: DeviceHooks<DeviceLike>) -> TapeStepProvider<DeviceLike> {
    TapeStepProvider::open_lora(device_like(), hooks, open_config(16), lora_init(true)).unwrap()
}

fn with_rebind() -> DeviceHooks<DeviceLike> {
    DeviceHooks {
        freeze: Some(Box::new(
            |_: &DeviceLike, _: &Tensor| -> Result<(), ojas_core::OjasError> { Ok(()) },
        )),
        rebind: Some(Box::new(|_: &DeviceLike, budget: ojas_core::Budget| {
            DeviceLike(CpuBackend::new(budget).with_numerics(Numerics::Exact))
        })),
    }
}

#[test]
fn a_device_backend_refuses_to_accumulate_until_the_preflight_has_passed() {
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
    let j = job(8, 5, 1.0, false);

    let mut p = device_provider(with_rebind());
    let err = p
        .accumulate(&j.view(), BankMode::Overwrite, None)
        .unwrap_err();
    assert!(err.to_string().contains("pre-flight"), "{err}");

    // A plan that does not fit refuses with the shape, and the gate stays shut.
    let tight = p
        .preflight(&calibration, 8, &[8, 16], 1 << 19, 0)
        .unwrap_err();
    assert!(tight.to_string().contains("preflight refused"), "{tight}");
    assert!(p.accumulate(&j.view(), BankMode::Overwrite, None).is_err());

    // A bucket wider than the tables is refused too: 16 tokens round up to
    // the 32-token bucket, and the tables hold 16 positions.
    let wide = p
        .preflight(&calibration, 16, &[8, 32], 1 << 30, 0)
        .unwrap_err();
    assert!(wide.to_string().contains("wider"), "{wide}");
    assert!(p.accumulate(&j.view(), BankMode::Overwrite, None).is_err());
}

#[test]
fn a_passing_preflight_opens_the_gate_and_a_missing_rebind_never_passes() {
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
    let j = job(8, 5, 1.0, false);

    let mut blind = device_provider(DeviceHooks::default());
    let err = blind
        .preflight(&calibration, 8, &[8, 16], 1 << 30, 0)
        .unwrap_err();
    assert!(matches!(err, StepError::Unsupported(_)), "{err}");
    assert!(blind
        .accumulate(&j.view(), BankMode::Overwrite, None)
        .is_err());

    let mut p = device_provider(with_rebind());
    let plan = p.preflight(&calibration, 8, &[8, 16], 1 << 30, 0).unwrap();
    assert_eq!(plan.bucket, 8);
    let v = p.accumulate(&j.view(), BankMode::Overwrite, None).unwrap();
    assert!(v.is_finite() && v > 0.0);
    // The dry run's scratch bank is not the provider's.
    assert!(p.bank().is_complete());
}

fn grads_of(p: &mut P) -> Vec<Vec<f32>> {
    p.read_gradients().unwrap()
}

/// Largest absolute difference between two gradient sets, over the largest
/// absolute value of `want`.
fn max_rel(got: &[Vec<f32>], want: &[Vec<f32>]) -> f32 {
    assert_eq!(got.len(), want.len());
    let peak = want.iter().flatten().fold(0.0f32, |p, x| p.max(x.abs()));
    assert!(peak > 0.0, "an all-zero reference checks nothing");
    let diff = got
        .iter()
        .flatten()
        .zip(want.iter().flatten())
        .fold(0.0f32, |d, (a, b)| d.max((a - b).abs()));
    diff / peak
}

/// A fresh provider's bank after one `Overwrite` of `j`.
fn grads_after(j: &Job, with_outside: bool) -> Vec<Vec<f32>> {
    let mut p = open(16, true);
    if with_outside {
        let mut ext = outside;
        p.accumulate(&j.view(), BankMode::Overwrite, Some(&mut ext))
            .unwrap();
    } else {
        p.accumulate(&j.view(), BankMode::Overwrite, None).unwrap();
    }
    grads_of(&mut p)
}

#[test]
fn a_refused_overwrite_empties_the_bank_so_later_adds_cannot_extend_the_previous_sum() {
    let mut p = open(16, true);
    p.accumulate(&job(8, 2, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    assert!(p.bank().is_complete());

    // The next step's first sequence fails in its callback, after the tower ran.
    let with_hidden = job(8, 3, 1.0, true);
    let mut failing = |_: &[u32], _: &[f32]| -> Result<Vec<f32>, StepError> {
        Err(StepError::Contract("the outside loss refused".into()))
    };
    assert!(p
        .accumulate(&with_hidden.view(), BankMode::Overwrite, Some(&mut failing))
        .is_err());
    assert!(
        !p.bank().is_complete(),
        "a failed Overwrite left the previous step's sum in the bank"
    );
    // A caller that ignored the error and went on with Add is refused.
    assert!(p
        .accumulate(&job(8, 4, 1.0, false).view(), BankMode::Add, None)
        .is_err());
    // Validation refusals are the same: nothing of the old sum survives.
    p.accumulate(&job(8, 2, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    assert!(p
        .accumulate(&job(17, 2, 1.0, false).view(), BankMode::Overwrite, None)
        .is_err());
    assert!(!p.bank().is_complete());
    let (lr_scale, wd) = per_entry(n_trainable(&p));
    assert!(p.adamw_step(&hyper(1.0), 1, &lr_scale, &wd).is_err());
    assert_eq!(p.step_count(), 0);
}

#[test]
fn a_failed_add_keeps_the_sum_of_the_sequences_before_it_and_leaks_no_device_memory() {
    let mut p = open(16, true);
    let budget = p.backend().budget().clone();
    p.accumulate(&job(8, 2, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    p.accumulate(&job(9, 3, 1.0, false).view(), BankMode::Add, None)
        .unwrap();
    let live = budget.live_bytes().unwrap();
    let before = bits_all(&grads_of(&mut p));

    let with_hidden = job(8, 4, 1.0, true);
    let mut failing = |_: &[u32], _: &[f32]| -> Result<Vec<f32>, StepError> {
        Err(StepError::Contract("the outside loss refused".into()))
    };
    assert!(p
        .accumulate(&with_hidden.view(), BankMode::Add, Some(&mut failing))
        .is_err());
    assert_eq!(
        budget.live_bytes().unwrap(),
        live,
        "a failed sequence leaked"
    );
    assert_eq!(bits_all(&grads_of(&mut p)), before);

    let mut ext = outside;
    p.accumulate(&job(12, 5, 1.0, true).view(), BankMode::Add, Some(&mut ext))
        .unwrap();
    assert_eq!(
        budget.live_bytes().unwrap(),
        live,
        "a sequence's tape outlived it"
    );
}

#[test]
fn the_row_scale_is_linear_and_the_two_roots_add() {
    // scale * N on the mean is scale times the gradient of the sum.
    let one = grads_after(&job(10, 1, 1.0, false), false);
    let half = grads_after(&job(10, 1, 0.5, false), false);
    let half_of_one: Vec<Vec<f32>> = one
        .iter()
        .map(|g| g.iter().map(|x| 0.5 * x).collect())
        .collect();
    assert!(max_rel(&half, &half_of_one) < 1e-5);

    // Rows and outside loss in one backward are the sum of each alone, and
    // the provider does not scale the outside gradient.
    let both = grads_after(&job(10, 1, 0.7, true), true);
    let rows_only = grads_after(
        &Job {
            hidden: Vec::new(),
            ..job(10, 1, 0.7, true)
        },
        false,
    );
    let dh_only = grads_after(
        &Job {
            positions: Vec::new(),
            targets: Vec::new(),
            ..job(10, 1, 0.7, true)
        },
        true,
    );
    let sum: Vec<Vec<f32>> = rows_only
        .iter()
        .zip(&dh_only)
        .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
        .collect();
    assert!(max_rel(&both, &sum) < 1e-4);

    // Scale 0 with an outside loss still back-propagates the outside
    // gradient: it equals the outside loss alone.
    let zero_rows = grads_after(&job(10, 1, 0.0, true), true);
    assert!(max_rel(&zero_rows, &dh_only) < 1e-5);
}

#[test]
fn a_failed_preflight_closes_a_gate_an_earlier_pass_opened() {
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
    let j = job(8, 5, 1.0, false);
    let mut p = device_provider(with_rebind());
    p.preflight(&calibration, 8, &[8, 16], 1 << 30, 0).unwrap();
    // The pre-flight leaves no state behind: the bank is empty, no step taken.
    assert!(!p.bank().is_complete());
    assert_eq!(p.step_count(), 0);
    p.accumulate(&j.view(), BankMode::Overwrite, None).unwrap();

    assert!(p.preflight(&calibration, 8, &[8, 16], 1 << 19, 0).is_err());
    let err = p
        .accumulate(&j.view(), BankMode::Overwrite, None)
        .unwrap_err();
    assert!(err.to_string().contains("pre-flight"), "{err}");

    // A second passing pre-flight reopens it.
    p.preflight(&calibration, 8, &[8, 16], 1 << 30, 0).unwrap();
    p.accumulate(&j.view(), BankMode::Overwrite, None).unwrap();
}

// ---------------------------------------------- sequence refusals (ft-3c6ed4be)

/// A LoRA provider whose spec carries the config-declared sequence limits.
fn open_limited(max_positions: Option<usize>, reserved: Vec<u32>) -> P {
    let mut cfg = open_config(16);
    cfg.spec.max_positions = max_positions;
    cfg.spec.reserved_token_ids = reserved;
    TapeStepProvider::open_lora(cpu(), DeviceHooks::default(), cfg, lora_init(true)).unwrap()
}

/// Run `bad` as an `Add` after a good `Overwrite`: it is refused with a message
/// holding `needle`, the bank's sum is bit-identical afterwards, and the
/// backend's budget never rose, so nothing was gathered, scored or put on a tape.
fn refused_before_any_work(p: &mut P, bad: &Job, needle: &str) {
    // Short enough for every limit these tests set.
    p.accumulate(&job(5, 2, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    let before = bits_all(&p.read_gradients().unwrap());
    let budget = p.backend().budget().clone();
    let live = budget.live_bytes().unwrap();
    budget.reset_peak();
    let err = p
        .accumulate(&bad.view(), BankMode::Add, None)
        .expect_err("the sequence was accepted");
    assert!(
        matches!(err, StepError::Contract(_)) && err.to_string().contains(needle),
        "{err}"
    );
    // The bank first, on its own: a refusal that wrote the bank (cleared it, or changed a
    // gradient) without allocating anything must still fail here, before the budget check
    // below could mask or replace the verdict.
    assert!(
        p.bank().is_complete(),
        "the refused sequence left the bank incomplete"
    );
    assert_eq!(
        bits_all(&p.read_gradients().unwrap()),
        before,
        "the refused sequence changed the bank's sum"
    );
    assert_eq!(
        budget.peak_bytes().saturating_sub(live),
        0,
        "the refused sequence allocated before it was refused"
    );
}

#[test]
fn a_reserved_token_id_is_refused_before_any_work_and_names_the_id() {
    let mut p = open_limited(None, vec![4242, 4243]);
    let mut bad = job(8, 2, 1.0, false);
    bad.tokens[3] = 4243;
    refused_before_any_work(&mut p, &bad, "4243");
    // The same job without the reserved id runs.
    p.accumulate(&job(8, 2, 1.0, false).view(), BankMode::Add, None)
        .unwrap();
}

#[test]
fn a_sequence_past_max_position_embeddings_is_refused_before_any_work() {
    let mut p = open_limited(Some(6), Vec::new());
    refused_before_any_work(&mut p, &job(8, 2, 1.0, false), "max_position_embeddings 6");
    // At the limit it runs; and with the field absent only the table width bounds it.
    p.accumulate(&job(6, 2, 1.0, false).view(), BankMode::Add, None)
        .unwrap();
    let mut open_ended = open_limited(None, Vec::new());
    open_ended
        .accumulate(&job(16, 2, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    assert!(open_ended
        .accumulate(&job(17, 2, 1.0, false).view(), BankMode::Add, None)
        .is_err());
}

#[test]
fn a_repeated_row_position_is_refused_before_any_work_and_distinct_ones_run() {
    let mut p = open(16, true);
    let mut dup = job(8, 2, 1.0, false);
    dup.positions = vec![4, 5, 4];
    dup.targets = vec![1, 2, 3];
    refused_before_any_work(&mut p, &dup, "row position 4");
    let mut distinct = job(8, 2, 1.0, false);
    distinct.positions = vec![4, 5, 6];
    distinct.targets = vec![1, 2, 3];
    p.accumulate(&distinct.view(), BankMode::Add, None).unwrap();
}

#[test]
fn scale_zero_with_rows_is_not_refused_and_a_bad_scale_still_is() {
    let mut p = open(16, true);
    let zero = job(8, 2, 0.0, false);
    assert!(p
        .accumulate(&zero.view(), BankMode::Overwrite, None)
        .is_ok());
    for bad in [-1.0f32, f32::NAN, f32::INFINITY] {
        let j = job(8, 2, bad, false);
        assert!(p.accumulate(&j.view(), BankMode::Overwrite, None).is_err());
    }
}

/// A scale the tape's seed check refuses must be refused on every arm that
/// back-propagates the rows, the combined one included, and a refused Add
/// leaves the bank's sum as it was.
#[test]
fn a_subnormal_row_scale_is_refused_on_the_combined_arm_as_on_the_rows_only_arm() {
    let mut p = open(16, true);
    p.accumulate(&job(8, 2, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    let before = bits_all(&p.read_gradients().unwrap());
    let ran = std::cell::Cell::new(0usize);
    let mut ext = |a: &[u32], h: &[f32]| -> Result<Vec<f32>, StepError> {
        ran.set(ran.get() + 1);
        outside(a, h)
    };
    // The seed is `scale * 4` for these four rows, so both stay subnormal.
    for tiny in [1e-40f32, f32::MIN_POSITIVE / 8.0] {
        let rows_only = p
            .accumulate(&job(8, 3, tiny, false).view(), BankMode::Add, None)
            .expect_err("rows only accepted a subnormal scale");
        let combined = p
            .accumulate(&job(8, 3, tiny, true).view(), BankMode::Add, Some(&mut ext))
            .expect_err("rows plus outside loss accepted a subnormal scale");
        assert_eq!(
            rows_only.to_string(),
            combined.to_string(),
            "the two arms refuse with different errors"
        );
        assert!(p.bank().is_complete());
        assert_eq!(
            bits_all(&p.read_gradients().unwrap()),
            before,
            "the refused combined sequence wrote a gradient"
        );
    }
    assert!(ran.get() > 0, "the combined arm never reached its callback");
}

/// Scale 0 with rows stays legal whatever sits beside it, and the arms that
/// do not multiply a scale in are unchanged.
#[test]
fn zero_and_bad_scales_on_the_combined_arm_and_the_other_arms_are_unchanged() {
    let mut p = open(16, true);
    let mut ext = outside;
    // Scale 0 with rows and an outside loss: the outside loss alone.
    p.accumulate(
        &job(8, 2, 0.0, true).view(),
        BankMode::Overwrite,
        Some(&mut ext),
    )
    .unwrap();
    // Scale 0 with rows and nothing else: value only, a zero gradient.
    p.accumulate(&job(8, 2, 0.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    assert!(p
        .read_gradients()
        .unwrap()
        .iter()
        .flatten()
        .all(|x| *x == 0.0));
    // Negative, NaN and infinite scales are refused on the combined arm too.
    for bad in [-1.0f32, -1e-40, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut ext = outside;
        assert!(
            p.accumulate(
                &job(8, 2, bad, true).view(),
                BankMode::Overwrite,
                Some(&mut ext)
            )
            .is_err(),
            "scale {bad} was accepted"
        );
    }
    // An outside loss alone takes no scale.
    let dh_only = Job {
        positions: Vec::new(),
        targets: Vec::new(),
        ..job(8, 2, 1.0, true)
    };
    let mut ext = outside;
    p.accumulate(&dh_only.view(), BankMode::Overwrite, Some(&mut ext))
        .unwrap();
}

/// A normal scale on the combined arm is the rows-only seeded gradient plus
/// the outside gradient, and repeating it is bit-identical.
#[test]
fn a_normal_scale_on_the_combined_arm_is_the_seeded_rows_gradient_plus_the_outside_one() {
    for scale in [1.0f32, 0.25, 1e-3] {
        let both = grads_after(&job(10, 4, scale, true), true);
        assert_eq!(
            bits_all(&both),
            bits_all(&grads_after(&job(10, 4, scale, true), true)),
            "the combined arm is not deterministic at scale {scale}"
        );
        let rows = grads_after(
            &Job {
                hidden: Vec::new(),
                ..job(10, 4, scale, true)
            },
            false,
        );
        let dh = grads_after(
            &Job {
                positions: Vec::new(),
                targets: Vec::new(),
                ..job(10, 4, scale, true)
            },
            true,
        );
        let sum: Vec<Vec<f32>> = rows
            .iter()
            .zip(&dh)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
            .collect();
        assert!(max_rel(&both, &sum) < 1e-4, "scale {scale}");
    }
}

fn tiny_calibration() -> Calibration {
    Calibration::from_points(
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
    .unwrap()
}

fn device_provider_limited(
    max_positions: Option<usize>,
    reserved: Vec<u32>,
) -> TapeStepProvider<DeviceLike> {
    let mut cfg = open_config(16);
    cfg.spec.max_positions = max_positions;
    cfg.spec.reserved_token_ids = reserved;
    TapeStepProvider::open_lora(device_like(), with_rebind(), cfg, lora_init(true)).unwrap()
}

#[test]
fn the_preflight_refuses_a_bucket_longer_than_max_position_embeddings() {
    let calibration = tiny_calibration();
    let mut p = device_provider_limited(Some(4), Vec::new());
    // 8 tokens round up to the 8-token bucket, which the config's limit of 4 forbids.
    let err = p
        .preflight(&calibration, 8, &[8, 16], 1 << 30, 0)
        .unwrap_err();
    let text = err.to_string();
    assert!(
        matches!(err, StepError::Contract(_))
            && text.contains("bucket of 8 tokens")
            && text.contains("max_position_embeddings 4"),
        "{text}"
    );
    // The gate stays shut.
    let j = job(4, 5, 1.0, false);
    assert!(p
        .accumulate(&j.view(), BankMode::Overwrite, None)
        .unwrap_err()
        .to_string()
        .contains("pre-flight"));
    // A limit that holds the bucket plans as before.
    let mut ok = device_provider_limited(Some(8), Vec::new());
    assert_eq!(
        ok.preflight(&calibration, 8, &[8, 16], 1 << 30, 0)
            .unwrap()
            .bucket,
        8
    );
}

#[test]
fn the_preflight_dry_run_never_uses_a_reserved_token_id() {
    let calibration = tiny_calibration();
    let vocab = spec().vocab as u32;
    // Every id reserved: there is no token the dry run may use, so no plan is approved.
    let mut none_left = device_provider_limited(None, (0..vocab).collect());
    let err = none_left
        .preflight(&calibration, 8, &[8, 16], 1 << 30, 0)
        .unwrap_err();
    assert!(err.to_string().contains("reserved"), "{err}");
    // The old dry run's tokens (`i % vocab`) were ids 0..8; with those reserved the plan still
    // succeeds, on the next ids.
    let mut some_left = device_provider_limited(None, (0..8).collect());
    assert_eq!(
        some_left
            .preflight(&calibration, 8, &[8, 16], 1 << 30, 0)
            .unwrap()
            .bucket,
        8
    );
}

#[test]
fn a_reserved_token_id_as_a_row_target_is_refused_before_any_work() {
    let mut p = open_limited(None, vec![4243]);
    let mut bad = job(8, 2, 1.0, false);
    bad.targets[0] = 4243;
    // Otherwise valid: the id is in the vocabulary and no token of the sequence is reserved.
    assert!(!bad.tokens.contains(&4243));
    refused_before_any_work(&mut p, &bad, "row target 4243");
    let mut as_target_msg = open_limited(None, vec![4243]);
    as_target_msg
        .accumulate(&job(5, 2, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    let err = as_target_msg
        .accumulate(&bad.view(), BankMode::Add, None)
        .unwrap_err();
    assert!(err.to_string().contains("target"), "{err}");
}

// ---- ft-042351e4: validate before take, zero markers, one readback, dry-run budget, rng encoding

fn bank_host(p: &P) -> Vec<Vec<f32>> {
    (0..p.leaf_count())
        .map(|i| p.bank().grad(i).unwrap().to_f32_vec().unwrap())
        .collect()
}

fn poison(p: &mut P, leaf: usize, value: f32) {
    let shape = p.bank().entries()[leaf].shape.clone();
    let n: usize = shape.iter().product();
    let t = Tensor::from_f32(&vec![value; n], &shape, &budget()).unwrap();
    p.bank_mut()
        .accumulate(&cpu(), leaf, BankMode::Overwrite, t)
        .unwrap();
}

fn moments_bits(p: &mut P) -> Vec<(Vec<u32>, Vec<u32>)> {
    p.read_moments()
        .unwrap()
        .iter()
        .map(|(m, v)| (bits(m), bits(v)))
        .collect()
}

#[test]
fn a_refused_adamw_step_leaves_the_bank_unscaled_and_complete_at_any_entry() {
    for grad_scale in [1.0, 0.5] {
        let n = open(16, true).leaf_count();
        for leaf in [0, n / 2, n - 1] {
            for bad in [f32::NAN, f32::INFINITY] {
                let what = format!("scale {grad_scale}, leaf {leaf} of {n}, value {bad}");
                let mut p = open(16, true);
                p.accumulate(&job(10, 1, 1.0, false).view(), BankMode::Overwrite, None)
                    .unwrap();
                let clean = bank_host(&p)[leaf].clone();
                poison(&mut p, leaf, bad);
                let bank_before = bits_all(&bank_host(&p));
                let params = bits_all(&p.read_parameters().unwrap());
                let moments = moments_bits(&mut p);

                let (lr_scale, wd) = per_entry(n_trainable(&p));
                let err = p
                    .adamw_step(&hyper(grad_scale), 1, &lr_scale, &wd)
                    .unwrap_err();
                assert!(err.to_string().contains("no tensor was stepped"), "{what}: {err}");
                assert!(
                    p.bank().is_complete(),
                    "{what}: the refusal emptied the bank"
                );
                assert_eq!(
                    bits_all(&bank_host(&p)),
                    bank_before,
                    "{what}: the bank's sums moved (scaled or lost)"
                );
                assert_eq!(bits_all(&p.read_parameters().unwrap()), params, "{what}");
                assert_eq!(moments_bits(&mut p), moments, "{what}");
                assert_eq!(p.step_count(), 0, "{what}");

                // Put the clean gradient back: the same bank now steps, and lands
                // where a provider that never saw the poison lands.
                let shape = p.bank().entries()[leaf].shape.clone();
                let t = Tensor::from_f32(&clean, &shape, &budget()).unwrap();
                p.bank_mut()
                    .accumulate(&cpu(), leaf, BankMode::Overwrite, t)
                    .unwrap();
                p.adamw_step(&hyper(grad_scale), 1, &lr_scale, &wd).unwrap();
                let mut control = open(16, true);
                control
                    .accumulate(&job(10, 1, 1.0, false).view(), BankMode::Overwrite, None)
                    .unwrap();
                control
                    .adamw_step(&hyper(grad_scale), 1, &lr_scale, &wd)
                    .unwrap();
                assert_eq!(
                    bits_all(&p.read_parameters().unwrap()),
                    bits_all(&control.read_parameters().unwrap()),
                    "{what}: the retried step differs from a clean one"
                );
            }
        }
    }
}

#[test]
fn a_scale_that_overflows_a_finite_gradient_is_refused_with_the_bank_intact() {
    let mut p = open(16, true);
    p.accumulate(&job(10, 1, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    let bank_before = bits_all(&bank_host(&p));
    let params = bits_all(&p.read_parameters().unwrap());
    let (lr_scale, wd) = per_entry(n_trainable(&p));
    // Every gradient is finite and small; times 1e30 its square is not a finite f32.
    let err = p.adamw_step(&hyper(1e30), 1, &lr_scale, &wd).unwrap_err();
    assert!(err.to_string().contains("no tensor was stepped"), "{err}");
    assert!(p.bank().is_complete());
    assert_eq!(bits_all(&bank_host(&p)), bank_before);
    assert_eq!(bits_all(&p.read_parameters().unwrap()), params);
    assert_eq!(p.step_count(), 0);
    p.adamw_step(&hyper(1.0), 1, &lr_scale, &wd).unwrap();
    assert_eq!(p.step_count(), 1);
}

#[test]
fn adamw_on_an_empty_bank_is_refused_and_takes_no_step() {
    let mut p = open(16, true);
    let params = bits_all(&p.read_parameters().unwrap());
    let (lr_scale, wd) = per_entry(n_trainable(&p));
    assert!(p.adamw_step(&hyper(1.0), 1, &lr_scale, &wd).is_err());
    assert_eq!(p.step_count(), 0);
    assert_eq!(bits_all(&p.read_parameters().unwrap()), params);
    // Half-filled is empty too, and nothing was taken from it.
    p.accumulate(&job(8, 2, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    p.bank_mut().clear();
    assert!(p.adamw_step(&hyper(1.0), 1, &lr_scale, &wd).is_err());
}

#[test]
fn a_clean_step_after_a_norm_is_identical_to_one_without_it() {
    let run = |ask_norm: bool| {
        let mut p = open(16, true);
        let mut ext = outside;
        p.accumulate(&job(10, 1, 1.0, true).view(), BankMode::Overwrite, Some(&mut ext))
            .unwrap();
        p.accumulate(&job(9, 2, 0.5, false).view(), BankMode::Add, None)
            .unwrap();
        let norm = if ask_norm {
            Some(p.grad_sq_norm().unwrap().to_bits())
        } else {
            None
        };
        let (lr_scale, wd) = per_entry(n_trainable(&p));
        p.adamw_step(&hyper(0.5), 1, &lr_scale, &wd).unwrap();
        (norm, bits_all(&p.read_parameters().unwrap()), moments_bits(&mut p))
    };
    let (norm, params_a, moments_a) = run(true);
    let (_, params_b, moments_b) = run(false);
    assert!(norm.is_some());
    assert_eq!(params_a, params_b);
    assert_eq!(moments_a, moments_b);
}

#[test]
fn the_norm_after_a_changed_bank_is_not_the_cached_one() {
    let mut p = open(16, true);
    p.accumulate(&job(10, 1, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    let a = p.grad_sq_norm().unwrap();
    p.accumulate(&job(9, 2, 1.0, false).view(), BankMode::Add, None)
        .unwrap();
    let b = p.grad_sq_norm().unwrap();
    assert_ne!(a.to_bits(), b.to_bits(), "a stale norm survived an Add");
    p.accumulate(&job(10, 1, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    assert_eq!(p.grad_sq_norm().unwrap().to_bits(), a.to_bits());
    poison(&mut p, 0, f32::NAN);
    assert!(p.grad_sq_norm().is_err(), "a stale norm hid a NaN");
}

fn full_provider() -> P {
    TapeStepProvider::open_full(cpu(), DeviceHooks::default(), open_config(16)).unwrap()
}

#[test]
fn a_scale_zero_sequence_allocates_no_gradient_in_either_mode() {
    for (what, mut p) in [("lora", open(16, true)), ("full", full_provider())] {
        let budget = p.backend().budget().clone();
        let zero = job(8, 3, 0.0, false);
        let live = budget.live_bytes().unwrap();
        p.accumulate(&zero.view(), BankMode::Overwrite, None)
            .unwrap();
        // The bank holds no tensor for it: no [V, d] (full) and no adapter-sized
        // zeros (LoRA) outlive the sequence.
        assert_eq!(
            budget.live_bytes().unwrap(),
            live,
            "{what}: a scale-0 gradient was materialised in the bank"
        );
        assert!(p.bank().is_complete(), "{what}");
        assert_eq!(p.grad_sq_norm().unwrap(), 0.0, "{what}");
        p.accumulate(&zero.view(), BankMode::Add, None).unwrap();
        assert_eq!(budget.live_bytes().unwrap(), live, "{what}: Add");
        // The step still runs on it (zero gradients, weight decay and moments).
        let (lr_scale, wd) = per_entry(n_trainable(&p));
        p.adamw_step(&hyper(1.0), 1, &lr_scale, &wd).unwrap();
        assert_eq!(p.step_count(), 1, "{what}");
    }
}

#[test]
fn a_zero_sequence_then_a_real_one_sums_to_the_real_one() {
    let real = job(10, 1, 1.0, false);
    let want = grads_after(&real, false);
    let mut p = open(16, true);
    p.accumulate(&job(8, 3, 0.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    p.accumulate(&real.view(), BankMode::Add, None).unwrap();
    assert!(max_rel(&grads_of(&mut p), &want) < 1e-6);
    // And a zero marker reads as zeros.
    let mut q = open(16, true);
    q.accumulate(&job(8, 3, 0.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    assert!(grads_of(&mut q).iter().flatten().all(|x| *x == 0.0));
}

#[cfg(target_os = "macos")]
fn metal() -> ojas_metal::MetalBackend {
    ojas_metal::MetalBackend::new(ojas_core::Budget::new(2 << 30))
        .unwrap_or_else(|e| panic!("no Metal device: {e:?}"))
}

#[cfg(target_os = "macos")]
fn metal_provider() -> TapeStepProvider<ojas_metal::MetalBackend> {
    use ojas_metal::MetalBackend;
    let hooks = DeviceHooks {
        freeze: Some(Box::new(|m: &MetalBackend, t: &Tensor| m.freeze_checked(t))),
        rebind: Some(Box::new(|m: &MetalBackend, b: ojas_core::Budget| {
            m.with_budget(b)
        })),
    };
    let mut p = TapeStepProvider::open_lora(metal(), hooks, open_config(16), lora_init(true))
        .unwrap();
    p.preflight(&tiny_calibration(), 16, &[16], 1 << 30, 0)
        .unwrap();
    p
}

/// Device readbacks the budget has counted so far.
#[cfg(target_os = "macos")]
fn readbacks(p: &TapeStepProvider<ojas_metal::MetalBackend>) -> u64 {
    p.backend().budget().device_readbacks().0
}

#[cfg(target_os = "macos")]
#[test]
fn on_metal_the_norm_and_the_update_read_each_gradient_once() {
    let mut p = metal_provider();
    let n = p.leaf_count() as u64;
    let (lr_scale, wd) = per_entry(n_trainable_any(&p));
    p.accumulate(&job(10, 1, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    p.accumulate(&job(9, 2, 1.0, false).view(), BankMode::Add, None)
        .unwrap();

    let start = readbacks(&p);
    p.grad_sq_norm().unwrap();
    let after_norm = readbacks(&p);
    p.adamw_step(&hyper(0.5), 1, &lr_scale, &wd).unwrap();
    let after_step = readbacks(&p);
    assert_eq!(after_norm - start, n, "the norm reads each gradient once");
    assert_eq!(
        after_step - after_norm,
        0,
        "the update re-read gradients the norm had already read"
    );

    // Without a norm the update reads each gradient once, not zero times.
    let mut q = metal_provider();
    q.accumulate(&job(10, 1, 1.0, false).view(), BankMode::Overwrite, None)
        .unwrap();
    let start = readbacks(&q);
    q.adamw_step(&hyper(0.5), 1, &lr_scale, &wd).unwrap();
    assert_eq!(readbacks(&q) - start, n);
}

#[cfg(target_os = "macos")]
fn n_trainable_any<B: ojas_core::Backend>(p: &TapeStepProvider<B>) -> usize {
    p.parameters().iter().filter(|s| s.trainable).count()
}

type Seen = std::rc::Rc<std::cell::RefCell<Vec<ojas_core::Budget>>>;

fn recording_hooks(seen: &Seen, honour: bool) -> DeviceHooks<DeviceLike> {
    let seen = seen.clone();
    DeviceHooks {
        freeze: Some(Box::new(
            |_: &DeviceLike, _: &Tensor| -> Result<(), ojas_core::OjasError> { Ok(()) },
        )),
        rebind: Some(Box::new(move |_: &DeviceLike, budget: ojas_core::Budget| {
            seen.borrow_mut().push(budget.clone());
            let charged = if honour {
                budget
            } else {
                ojas_core::Budget::new(4 << 30)
            };
            DeviceLike(CpuBackend::new(charged).with_numerics(Numerics::Exact))
        })),
    }
}

#[test]
fn the_dry_run_measures_only_what_the_rebound_backend_charges_to_its_budget() {
    let calibration = tiny_calibration();

    let seen: Seen = Default::default();
    let mut p = device_provider(recording_hooks(&seen, true));
    p.preflight(&calibration, 8, &[8, 16], 1 << 30, 0).unwrap();
    let budgets = seen.borrow();
    assert_eq!(budgets.len(), 1, "one dry run, one rebind");
    assert!(
        budgets[0].peak_bytes() > 0,
        "a hook that rebinds to its budget must measure a peak"
    );

    // A hook that ignores its budget charges elsewhere: the measured peak is 0
    // and the pre-flight still passes (the documented blind spot).
    let blind: Seen = Default::default();
    let mut q = device_provider(recording_hooks(&blind, false));
    q.preflight(&calibration, 8, &[8, 16], 1 << 30, 0).unwrap();
    assert_eq!(blind.borrow()[0].peak_bytes(), 0);
}

#[test]
fn the_rng_state_is_the_seed_then_the_epoch_little_endian_and_round_trips() {
    let tmp = Tmp::new();
    let mut p = open(16, true);
    let ctx = CheckpointContext {
        optimizer_step: 0,
        seed: 0x0102_0304_0506_0708,
        epoch: 0x1112_1314_1516_1718,
        consumed_digest: "ab".repeat(32),
    };
    p.save_state_with(&tmp.0, &ctx).unwrap();
    let file = SafeTensors::open(&tmp.0.join("provider").join("adapters.safetensors")).unwrap();
    assert_eq!(
        file.metadata()["ojas.adapter.rng_state"],
        "08070605040302011817161514131211"
    );

    let mut resumed = open(16, false);
    resumed.load_state_with(&tmp.0, &ctx).unwrap();
    // Swapped, or either half changed by one bit, is another data order.
    for (what, wrong) in [
        (
            "swapped",
            CheckpointContext {
                seed: ctx.epoch,
                epoch: ctx.seed,
                ..ctx.clone()
            },
        ),
        (
            "seed",
            CheckpointContext {
                seed: ctx.seed ^ 1,
                ..ctx.clone()
            },
        ),
        (
            "epoch",
            CheckpointContext {
                epoch: ctx.epoch ^ (1 << 63),
                ..ctx.clone()
            },
        ),
    ] {
        assert!(resumed.load_state_with(&tmp.0, &wrong).is_err(), "{what}");
    }
}
