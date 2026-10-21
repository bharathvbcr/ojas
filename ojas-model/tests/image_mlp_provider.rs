//! `ImageMlpProvider` (`ojas-model/src/image_mlp.rs`): a toy image classifier trained through
//! `ojas-train`'s loop and `StepProvider`, on the CPU and on Metal. The idx fixture is generated
//! here (each class a stripe of bright pixels over noise, so the classes are learnable across
//! batches); nothing is downloaded.
//!
//! # What "bit-identical" means here
//!
//! The CPU runs under `Numerics::Exact` and Metal under its default `Numerics::Fast`, so the two
//! backends' bits are not guaranteed to agree: the Metal test holds their per-step losses to the
//! 1e-4 relative bound `ojas-metal`'s `mnist_step` and `training_step` use. The checkpoint claim
//! is read as: on each backend, a run resumed from a checkpoint (a fresh provider, the loop's
//! `ResumeState::read`) equals the uninterrupted run bit for bit in every loss, every parameter
//! and both AdamW moments.

use std::io::Cursor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use ojas_core::{Backend, Budget, Numerics};
use ojas_cpu::CpuBackend;
use ojas_data::{Augment, ImageBatchSampler, ImageSamplerConfig, ImageSet, Normalization};
use ojas_io::{encode_f32_as, write_safetensors, SafeTensors, StDtype, TensorOut};
use ojas_model::image_mlp::{
    ClassificationObjective, EpochBatches, ImageBatch, ImageMlpProvider, ImageMlpSpec, STATE_FILE,
};
use ojas_train::lappi::recipe::OptimizerRecipe;
use ojas_train::lappi::run_control::{ConsumedPrefix, MonotonicClock, WallClockCap};
use ojas_train::lappi::schedule::LrSchedule;
use ojas_train::lappi::step::{
    AdamWHyper, BankMode, CheckpointContext, RowTargets, SequenceJob, StepError, StepProvider,
};
use ojas_train::lappi::trainer::{
    train, CheckpointPolicy, ConsumedBatch, Hooks, Objective, ResumeState, TrainConfig, TrainError,
    TrainResult,
};

const N: usize = 96;
const BATCH: usize = 8;
const PIXELS: usize = 28 * 28;
const HIDDEN: usize = 32;
const CLASSES: usize = 10;
const SEED: u64 = 7;
const LR: f64 = 2e-3;
/// Metal's `Numerics::Fast` against the CPU reference on a loss that has passed through
/// backward and an optimizer update: the figure `mnist_step.rs` and `training_step.rs` use.
#[cfg(target_os = "macos")]
const LOSS_RTOL: f64 = 1e-4;

static DIRS: AtomicU64 = AtomicU64::new(0);

struct Tmp(PathBuf);

impl Tmp {
    fn new() -> Self {
        let n = DIRS.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("ojas-image-mlp-{}-{n}", std::process::id()));
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

/// An idx image/label pair of `N` 28 x 28 images: class `c` lights every pixel whose index is
/// `c` modulo 10, over noise.
fn idx_fixture() -> (Vec<u8>, Vec<u8>) {
    let mut img = vec![0, 0, 8, 3];
    for d in [N as u32, 28, 28] {
        img.extend_from_slice(&d.to_be_bytes());
    }
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    for i in 0..N {
        for p in 0..PIXELS {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let noise = ((s >> 33) % 60) as u8;
            img.push(if p % CLASSES == i % CLASSES {
                200 + noise / 2
            } else {
                noise
            });
        }
    }
    let mut lab = vec![0, 0, 8, 1];
    lab.extend_from_slice(&(N as u32).to_be_bytes());
    lab.extend((0..N).map(|i| (i % CLASSES) as u8));
    (img, lab)
}

fn image_set() -> ImageSet {
    let (img, lab) = idx_fixture();
    ImageSet::from_mnist(Cursor::new(img), Cursor::new(lab), &Budget::new(1 << 24)).unwrap()
}

fn sampler_config() -> ImageSamplerConfig {
    ImageSamplerConfig {
        batch: BATCH,
        seed: SEED,
        drop_last: true,
        normalization: Normalization::MNIST,
        augment: Augment::NONE,
    }
}

fn spec() -> ImageMlpSpec {
    ImageMlpSpec {
        pixels: PIXELS,
        hidden: HIDDEN,
        classes: CLASSES,
        mean: vec![0.1307],
        std: vec![0.3081],
        seed: 11,
    }
}

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(8 << 30)).with_numerics(Numerics::Exact)
}

fn config(total_steps: u64, checkpoint: Option<CheckpointPolicy>) -> TrainConfig {
    TrainConfig {
        schedule: LrSchedule::new(LR, total_steps, 1, LR / 10.0).unwrap(),
        cap: WallClockCap::new(3600.0).unwrap(),
        grad_accum: 1,
        optimizer: OptimizerRecipe {
            lr: LR,
            beta2: 0.999,
            lower_layers: None,
        },
        max_grad_norm: 1.0,
        epoch: 0,
        epochs: 1,
        seed: SEED,
        checkpoint,
        eta: None,
    }
}

/// The loop over the first `batches` batches of epoch 0 of a `total_steps` schedule.
fn run<B: Backend>(
    provider: &mut ImageMlpProvider<B>,
    set: &ImageSet,
    total_steps: u64,
    batches: usize,
    checkpoint: Option<CheckpointPolicy>,
    resume: Option<ResumeState>,
) -> TrainResult {
    train(
        provider,
        &mut ClassificationObjective::new(),
        EpochBatches::new(set, sampler_config(), 0)
            .unwrap()
            .take(batches),
        &config(total_steps, checkpoint),
        &MonotonicClock::new(),
        resume,
        &mut Hooks::default(),
    )
    .unwrap()
}

fn losses(r: &TrainResult) -> Vec<f64> {
    r.steps.iter().map(|s| s.loss).collect()
}

fn loss_bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

type State = (Vec<Vec<u32>>, Vec<(Vec<u32>, Vec<u32>)>);

fn state<B: Backend>(p: &mut ImageMlpProvider<B>) -> State {
    (
        bits_all(&p.read_parameters().unwrap()),
        p.read_moments()
            .unwrap()
            .iter()
            .map(|(m, v)| (bits(m), bits(v)))
            .collect(),
    )
}

/// Six straight steps, and three steps, a loop checkpoint, a fresh provider, `ResumeState` and
/// three more. Asserts the two agree in every loss, parameter and moment and returns the
/// per-step losses.
fn resume_is_bit_identical<B: Backend>(make: impl Fn() -> B) -> Vec<f64> {
    let set = image_set();
    let mut straight = ImageMlpProvider::open(make(), spec()).unwrap();
    let whole = run(&mut straight, &set, 6, 6, None, None);
    assert_eq!(whole.optimizer_steps, 6);

    let tmp = Tmp::new();
    let policy = CheckpointPolicy {
        dir: tmp.0.join("checkpoints"),
        every: 0,
    };
    let mut first = ImageMlpProvider::open(make(), spec()).unwrap();
    let head = run(&mut first, &set, 6, 3, Some(policy), None);
    assert_eq!(head.optimizer_steps, 3);
    let ckpt = head
        .final_checkpoint
        .clone()
        .expect("a checkpoint at step 3");
    assert!(ckpt.join("provider").join(STATE_FILE).is_file());

    // A different init seed: the load is what makes this the checkpointed provider.
    let mut resumed = ImageMlpProvider::open(make(), ImageMlpSpec { seed: 99, ..spec() }).unwrap();
    assert_ne!(state(&mut resumed).0, state(&mut first).0);
    let tail = run(
        &mut resumed,
        &set,
        6,
        6,
        None,
        Some(ResumeState::read(&ckpt).unwrap()),
    );
    assert_eq!(tail.optimizer_steps, 6);
    assert_eq!(resumed.step_count(), 6);

    assert_eq!(
        loss_bits(&losses(&whole)),
        loss_bits(&losses(&tail)),
        "the resumed run's losses differ from the uninterrupted run's"
    );
    assert_eq!(whole.consumed_digest, tail.consumed_digest);
    assert_eq!(state(&mut straight), state(&mut resumed));
    // The moments are not all zero: the comparison above is not of two untouched optimizers.
    assert!(state(&mut resumed)
        .1
        .iter()
        .any(|(m, _)| m.iter().any(|&x| f32::from_bits(x) != 0.0)));
    losses(&whole)
}

#[test]
fn six_straight_steps_equal_three_a_checkpoint_a_fresh_provider_and_three_more() {
    let l = resume_is_bit_identical(cpu);
    eprintln!("cpu per-step losses: {l:?}");
}

#[test]
fn the_loss_falls_over_the_run() {
    let set = image_set();
    let mut p = ImageMlpProvider::open(cpu(), spec()).unwrap();
    let r = run(&mut p, &set, 12, 12, None, None);
    let l = losses(&r);
    eprintln!("cpu losses over 12 steps: {l:?}");
    assert_eq!(l.len(), 12);
    assert!(l.iter().all(|x| x.is_finite()));
    let head: f64 = l[..3].iter().sum::<f64>() / 3.0;
    let tail: f64 = l[9..].iter().sum::<f64>() / 3.0;
    assert!(tail < head * 0.9, "loss did not fall: {head} -> {tail}");
}

/// An owned single-image job.
struct Job {
    tokens: Vec<u32>,
    positions: Vec<u32>,
    targets: Vec<u32>,
    hidden_at: Vec<u32>,
    scale: f32,
}

impl Job {
    fn ok() -> Self {
        Self {
            tokens: (0..PIXELS).map(|i| (i * 7 % 256) as u32).collect(),
            positions: vec![0],
            targets: vec![3],
            hidden_at: Vec::new(),
            scale: 0.25,
        }
    }

    fn view(&self) -> SequenceJob<'_> {
        SequenceJob {
            tokens: &self.tokens,
            rows: RowTargets {
                positions: &self.positions,
                targets: &self.targets,
                scale: self.scale,
            },
            hidden_at: &self.hidden_at,
        }
    }
}

fn fresh() -> ImageMlpProvider<CpuBackend> {
    ImageMlpProvider::open(cpu(), spec()).unwrap()
}

#[test]
fn the_job_contract_is_enforced_literally() {
    let p = &mut fresh();
    assert_eq!(p.vocab_size(), 256);
    assert_eq!(p.hidden_size(), HIDDEN);
    let good = p
        .accumulate(&Job::ok().view(), BankMode::Overwrite, None)
        .unwrap();
    assert!(good.is_finite() && good > 0.0);
    let kept = p.grad_sq_norm().unwrap();
    assert!(kept > 0.0);

    let cases: Vec<(&str, Job)> = vec![
        (
            "position 1",
            Job {
                positions: vec![1],
                ..Job::ok()
            },
        ),
        (
            "two rows",
            Job {
                positions: vec![0, 0],
                targets: vec![1, 2],
                ..Job::ok()
            },
        ),
        (
            "no rows",
            Job {
                positions: vec![],
                targets: vec![],
                ..Job::ok()
            },
        ),
        (
            "label == classes",
            Job {
                targets: vec![CLASSES as u32],
                ..Job::ok()
            },
        ),
        (
            "label far out of range",
            Job {
                targets: vec![u32::MAX],
                ..Job::ok()
            },
        ),
        (
            "empty image",
            Job {
                tokens: vec![],
                ..Job::ok()
            },
        ),
        (
            "one pixel short",
            Job {
                tokens: Job::ok().tokens[1..].to_vec(),
                ..Job::ok()
            },
        ),
        (
            "one pixel long",
            Job {
                tokens: [Job::ok().tokens, vec![0]].concat(),
                ..Job::ok()
            },
        ),
        (
            "pixel above 255",
            Job {
                tokens: vec![256; PIXELS],
                ..Job::ok()
            },
        ),
        (
            "NaN scale",
            Job {
                scale: f32::NAN,
                ..Job::ok()
            },
        ),
        (
            "negative scale",
            Job {
                scale: -1.0,
                ..Job::ok()
            },
        ),
    ];
    for (what, job) in &cases {
        let err = p
            .accumulate(&job.view(), BankMode::Add, None)
            .expect_err(what);
        assert!(matches!(err, StepError::Contract(_)), "{what}: {err}");
        // A refused Add keeps the sum before it.
        assert_eq!(p.grad_sq_norm().unwrap(), kept, "{what}: the bank moved");
    }

    let hidden = Job {
        hidden_at: vec![0],
        ..Job::ok()
    };
    let mut callback =
        |_: &[u32], h: &[f32]| -> Result<Vec<f32>, StepError> { Ok(vec![0.0; h.len()]) };
    for (what, external) in [
        ("no callback", None),
        ("a callback", Some(&mut callback as _)),
    ] {
        let err = p
            .accumulate(&hidden.view(), BankMode::Add, external)
            .expect_err(what);
        assert!(matches!(err, StepError::Unsupported(_)), "{what}: {err}");
    }

    // A refused first sequence of a step leaves no sum for the next Add to extend.
    let bad = Job {
        positions: vec![1],
        ..Job::ok()
    };
    assert!(p
        .accumulate(&bad.view(), BankMode::Overwrite, None)
        .is_err());
    assert!(p.grad_sq_norm().is_err());
    assert!(p
        .accumulate(&Job::ok().view(), BankMode::Add, None)
        .is_err());
    assert_eq!(p.step_count(), 0);
}

#[test]
fn a_scale_of_zero_returns_the_value_and_writes_zero_gradients() {
    let p = &mut fresh();
    let one = p
        .accumulate(&Job::ok().view(), BankMode::Overwrite, None)
        .unwrap();
    let zero = Job {
        scale: 0.0,
        ..Job::ok()
    };
    let value = p
        .accumulate(&zero.view(), BankMode::Overwrite, None)
        .unwrap();
    assert_eq!(value.to_bits(), one.to_bits());
    assert_eq!(p.grad_sq_norm().unwrap(), 0.0);
}

#[test]
fn the_gradient_scale_is_linear_in_the_row_scale() {
    let p = &mut fresh();
    let half = Job {
        scale: 0.5,
        ..Job::ok()
    };
    let full = Job {
        scale: 1.0,
        ..Job::ok()
    };
    p.accumulate(&half.view(), BankMode::Overwrite, None)
        .unwrap();
    let a = p.grad_sq_norm().unwrap();
    p.accumulate(&full.view(), BankMode::Overwrite, None)
        .unwrap();
    let b = p.grad_sq_norm().unwrap();
    assert!((b / a - 4.0).abs() < 1e-5, "{a} {b}");
}

#[test]
fn adamw_refuses_a_wrong_step_and_moves_nothing() {
    let p = &mut fresh();
    p.accumulate(&Job::ok().view(), BankMode::Overwrite, None)
        .unwrap();
    let before = state(p);
    let hyper = AdamWHyper {
        lr: LR,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: 1.0,
    };
    let (scale, decay) = (vec![1.0; 2], vec![0.01; 2]);
    assert!(matches!(
        p.adamw_step(&hyper, 2, &scale, &decay),
        Err(StepError::Contract(_))
    ));
    assert!(matches!(
        p.adamw_step(&hyper, 1, &scale[..1], &decay),
        Err(StepError::Contract(_))
    ));
    assert_eq!(state(p), before);
    p.adamw_step(&hyper, 1, &scale, &decay).unwrap();
    assert_eq!(p.step_count(), 1);
    assert_ne!(state(p).0, before.0);
}

#[derive(Debug)]
enum Damage {
    None,
    Metadata(&'static str, &'static str),
    ExtraTensor,
    MissingTensor,
    NanIn(&'static str),
    InfIn(&'static str),
    WrongShape,
}

/// Re-write the checkpoint `src` to `dst` through `write_safetensors` with `damage` applied.
fn rewrite(src: &std::path::Path, dst: &std::path::Path, damage: &Damage) {
    let file = SafeTensors::open(src).unwrap();
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = file
        .names()
        .map(|n| {
            let (shape, values) = file.read_f32(n).unwrap();
            (n.to_owned(), shape, values)
        })
        .collect();
    let mut meta: Vec<(String, String)> = file
        .metadata()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    match damage {
        Damage::None => {}
        Damage::Metadata(key, value) => {
            meta.iter_mut().find(|(k, _)| k == key).unwrap().1 = (*value).to_owned();
        }
        Damage::ExtraTensor => tensors.push(("stray".into(), vec![1], vec![0.0])),
        Damage::MissingTensor => {
            tensors.pop();
        }
        Damage::NanIn(name) | Damage::InfIn(name) => {
            let t = tensors.iter_mut().find(|(n, _, _)| n == name).unwrap();
            t.2[0] = if matches!(damage, Damage::NanIn(_)) {
                f32::NAN
            } else {
                f32::INFINITY
            };
        }
        Damage::WrongShape => {
            let t = tensors
                .iter_mut()
                .find(|(n, _, _)| n == "classifier.w2.weight")
                .unwrap();
            let n = t.2.len() as u64;
            t.1 = vec![n, 1];
        }
    }
    let data: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, v)| encode_f32_as(StDtype::F32, v).unwrap())
        .collect();
    let items: Vec<TensorOut<'_>> = tensors
        .iter()
        .zip(&data)
        .map(|((name, shape, _), bytes)| TensorOut {
            name,
            dtype: StDtype::F32,
            shape,
            data: bytes,
        })
        .collect();
    let meta: Vec<(&str, &str)> = meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    write_safetensors(dst, &items, &meta).unwrap();
}

#[test]
fn a_checkpoint_that_is_not_this_providers_or_this_loops_is_refused_and_moves_nothing() {
    let set = image_set();
    let tmp = Tmp::new();
    let mut p = fresh();
    run(&mut p, &set, 6, 2, None, None);
    let ctx = CheckpointContext {
        optimizer_step: 2,
        seed: SEED,
        epoch: 0,
        consumed_digest: "ab".repeat(32),
    };
    assert!(matches!(
        p.save_state(&tmp.0),
        Err(StepError::Unsupported(_))
    ));
    assert!(matches!(
        p.load_state(&tmp.0),
        Err(StepError::Unsupported(_))
    ));
    let wrong_step = CheckpointContext {
        optimizer_step: 1,
        ..ctx.clone()
    };
    assert!(p.save_state_with(&tmp.0, &wrong_step).is_err());
    let files = p.save_state_with(&tmp.0, &ctx).unwrap();
    assert_eq!(files, vec![tmp.0.join(STATE_FILE)]);

    let mut q = ImageMlpProvider::open(cpu(), ImageMlpSpec { seed: 5, ..spec() }).unwrap();
    let untouched = state(&mut q);
    for (what, wrong) in [
        (
            "step",
            CheckpointContext {
                optimizer_step: 3,
                ..ctx.clone()
            },
        ),
        (
            "seed",
            CheckpointContext {
                seed: SEED + 1,
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
        assert!(q.load_state_with(&tmp.0, &wrong).is_err(), "{what}");
        assert_eq!(
            state(&mut q),
            untouched,
            "{what}: a refused load moved state"
        );
        assert_eq!(q.step_count(), 0, "{what}");
    }

    let other = ImageMlpProvider::open(
        cpu(),
        ImageMlpSpec {
            hidden: HIDDEN + 1,
            ..spec()
        },
    );
    let mut other = other.unwrap();
    assert!(
        other.load_state_with(&tmp.0, &ctx).is_err(),
        "another hidden width"
    );

    let good = std::fs::read(tmp.0.join(STATE_FILE)).unwrap();
    let dir = Tmp::new();
    std::fs::write(dir.0.join(STATE_FILE), &good[..good.len() - 4]).unwrap();
    assert!(q.load_state_with(&dir.0, &ctx).is_err(), "truncated");
    assert_eq!(
        state(&mut q),
        untouched,
        "truncated: a refused load moved state"
    );

    for damage in [
        Damage::Metadata("format", "other"),
        Damage::Metadata("version", "2"),
        Damage::Metadata("hidden", "33"),
        Damage::Metadata("step", "3"),
        Damage::ExtraTensor,
        Damage::MissingTensor,
        Damage::NanIn("classifier.w2.weight.adam_v"),
        Damage::InfIn("classifier.w1.weight"),
        Damage::WrongShape,
    ] {
        let dir = Tmp::new();
        rewrite(&tmp.0.join(STATE_FILE), &dir.0.join(STATE_FILE), &damage);
        assert!(q.load_state_with(&dir.0, &ctx).is_err(), "{damage:?}");
        assert_eq!(
            state(&mut q),
            untouched,
            "{damage:?}: a refused load moved state"
        );
    }
    // The control: the same rewrite with nothing damaged loads.
    let dir = Tmp::new();
    rewrite(
        &tmp.0.join(STATE_FILE),
        &dir.0.join(STATE_FILE),
        &Damage::None,
    );
    q.load_state_with(&dir.0, &ctx).unwrap();
    assert_eq!(state(&mut q), state(&mut p));
    let mut q = ImageMlpProvider::open(cpu(), ImageMlpSpec { seed: 5, ..spec() }).unwrap();

    q.load_state_with(&tmp.0, &ctx).unwrap();
    assert_eq!(q.step_count(), 2);
    assert_eq!(state(&mut q), state(&mut p));
}

#[test]
fn the_objective_plans_one_sequence_per_image_at_the_batch_mean_scale() {
    let set = image_set();
    let batch: ImageBatch = EpochBatches::new(&set, sampler_config(), 0)
        .unwrap()
        .nth(2)
        .unwrap()
        .unwrap();
    assert_eq!(batch.index, 2);
    assert_eq!(batch.items.len(), BATCH);

    let mut sampler = ImageBatchSampler::new(&set, sampler_config()).unwrap();
    for (k, item) in batch.items.iter().enumerate() {
        let expected = sampler.index_at(0, 2 * BATCH + k).unwrap();
        assert_eq!(item.index as usize, expected);
        assert_eq!(item.pixels, set.image(expected));
        assert_eq!(item.label, set.label(expected));
    }

    let mut objective = ClassificationObjective::new();
    let plan = objective.plan(&batch).unwrap();
    assert_eq!(plan.sequences.len(), BATCH);
    assert_eq!(plan.row_scale, 1.0 / BATCH as f32);
    for (seq, item) in plan.sequences.iter().zip(&batch.items) {
        assert_eq!(seq.positions, vec![0]);
        assert_eq!(seq.targets, vec![u32::from(item.label)]);
        assert!(seq.hidden_at.is_empty());
        assert_eq!(seq.tokens.len(), PIXELS);
        assert!(seq
            .tokens
            .iter()
            .zip(&item.pixels)
            .all(|(t, p)| *t == u32::from(*p)));
    }
    let micro = objective.close(&[2.0; BATCH]).unwrap();
    assert!((micro.total - 2.0).abs() < 1e-6, "{}", micro.total);
    assert!(objective.host().is_none() && objective.host_ref().is_none());
    assert!(objective.hidden_grad(0, &[0], &[0.0], 1).is_err());

    let empty = ImageBatch {
        index: 0,
        items: Vec::new(),
    };
    assert!(objective.plan(&empty).is_err());

    // The consumed digest depends on the batch index and on which images it holds.
    let (mut a, mut b, mut c) = (
        ConsumedPrefix::new(),
        ConsumedPrefix::new(),
        ConsumedPrefix::new(),
    );
    batch.fold_into(&mut a).unwrap();
    batch.fold_into(&mut b).unwrap();
    ImageBatch {
        index: 3,
        ..batch.clone()
    }
    .fold_into(&mut c)
    .unwrap();
    assert_eq!(a.hexdigest(), b.hexdigest());
    assert_ne!(a.hexdigest(), c.hexdigest());
    let mut d = ConsumedPrefix::new();
    let mut swapped = batch.clone();
    swapped.items.swap(0, 1);
    swapped.fold_into(&mut d).unwrap();
    assert_ne!(a.hexdigest(), d.hexdigest());
}

#[test]
fn epoch_batches_refuses_augmentation_and_covers_the_epoch_once() {
    let set = image_set();
    let augmented = ImageSamplerConfig {
        augment: Augment::CIFAR10,
        ..sampler_config()
    };
    assert!(matches!(
        EpochBatches::new(&set, augmented, 0),
        Err(TrainError::Refused(_))
    ));
    let mut seen = vec![0u32; N];
    let all: Vec<ImageBatch> = EpochBatches::new(&set, sampler_config(), 0)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(all.len(), N / BATCH);
    for (i, b) in all.iter().enumerate() {
        assert_eq!(b.index, i as u64);
        for item in &b.items {
            seen[item.index as usize] += 1;
        }
    }
    assert!(seen.iter().all(|&c| c == 1));
    // A short last batch is kept when drop_last is off.
    let keep = ImageSamplerConfig {
        batch: 10,
        drop_last: false,
        ..sampler_config()
    };
    let sizes: Vec<usize> = EpochBatches::new(&set, keep, 0)
        .unwrap()
        .map(|b| b.unwrap().items.len())
        .collect();
    assert_eq!(sizes, [10, 10, 10, 10, 10, 10, 10, 10, 10, 6]);
}

/// The same run on Metal: resume bit-identity on the device, the per-step loss against the CPU
/// within the Fast tolerance, and a falling loss.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs a Metal device; run alone with --ignored --test-threads=1"]
fn gpu_resume_is_bit_identical_and_matches_the_cpu_within_the_fast_tolerance() {
    use ojas_metal::MetalBackend;

    let metal = || match MetalBackend::new(Budget::new(8 << 30)) {
        Ok(m) => m,
        Err(e) => panic!("Metal device unavailable: {e:?}"),
    };
    let on_cpu = resume_is_bit_identical(cpu);
    let on_gpu = resume_is_bit_identical(metal);
    eprintln!("cpu   per-step losses: {on_cpu:?}");
    eprintln!("metal per-step losses: {on_gpu:?}");
    for (step, (g, c)) in on_gpu.iter().zip(&on_cpu).enumerate() {
        let rel = (g - c).abs() / c.abs();
        eprintln!("step {}: metal {g} cpu {c} rel {rel:.3e}", step + 1);
        assert!(
            rel <= LOSS_RTOL,
            "step {}: rel {rel:.3e} > {LOSS_RTOL}",
            step + 1
        );
    }

    let set = image_set();
    let mut p = ImageMlpProvider::open(metal(), spec()).unwrap();
    let l = losses(&run(&mut p, &set, 12, 12, None, None));
    eprintln!("metal losses over 12 steps: {l:?}");
    let head: f64 = l[..3].iter().sum::<f64>() / 3.0;
    let tail: f64 = l[9..].iter().sum::<f64>() / 3.0;
    assert!(
        tail < head * 0.9,
        "loss did not fall on Metal: {head} -> {tail}"
    );
}
