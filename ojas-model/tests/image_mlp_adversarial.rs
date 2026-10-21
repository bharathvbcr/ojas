//! Adversarial CPU tests for `ImageMlpProvider` (`ojas-model/src/image_mlp.rs`), public API only.
//!
//! Tests named `finding_*` assert what the contract SHOULD do and are expected to FAIL against the
//! current implementation: each one is a reported defect (see the A2 report). Every other test is
//! expected to pass. Nothing here was compiled by its author (no cargo while the verification job
//! ran): fix a typo before reading a compile error as a verdict.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ojas_core::{Budget, Numerics};
use ojas_cpu::CpuBackend;
use ojas_data::{Augment, ImageSamplerConfig, ImageSet, Normalization};
use ojas_io::{encode_f32_as, write_safetensors, SafeTensors, StDtype, TensorOut};
use ojas_model::image_mlp::{
    ClassificationObjective, EpochBatches, ImageBatch, ImageItem, ImageMlpProvider, ImageMlpSpec,
    STATE_FILE,
};
use ojas_train::lappi::run_control::ConsumedPrefix;
use ojas_train::lappi::step::{
    AdamWHyper, BankMode, CheckpointContext, RowTargets, SequenceJob, StepError, StepProvider,
};
use ojas_train::lappi::trainer::{ConsumedBatch, Objective, TrainError};

const N: usize = 96;
const PIXELS: usize = 28 * 28;
const HIDDEN: usize = 16;
const CLASSES: usize = 10;
const SEED: u64 = 7;
const LR: f64 = 2e-3;

static DIRS: AtomicU64 = AtomicU64::new(0);

struct Tmp(PathBuf);

impl Tmp {
    fn new() -> Self {
        let n = DIRS.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("ojas-image-mlp-adv-{}-{n}", std::process::id()));
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

fn sampler_config(batch: usize, drop_last: bool) -> ImageSamplerConfig {
    ImageSamplerConfig {
        batch,
        seed: SEED,
        drop_last,
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

type P = ImageMlpProvider<CpuBackend>;

fn fresh() -> P {
    ImageMlpProvider::open(cpu(), spec()).unwrap()
}

fn fresh_seeded(seed: u64) -> P {
    ImageMlpProvider::open(cpu(), ImageMlpSpec { seed, ..spec() }).unwrap()
}

type State = (Vec<Vec<u32>>, Vec<(Vec<u32>, Vec<u32>)>);

fn state(p: &mut P) -> State {
    (
        p.read_parameters()
            .unwrap()
            .iter()
            .map(|x| bits(x))
            .collect(),
        p.read_moments()
            .unwrap()
            .iter()
            .map(|(m, v)| (bits(m), bits(v)))
            .collect(),
    )
}

fn hyper() -> AdamWHyper {
    AdamWHyper {
        lr: LR,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: 1.0,
    }
}

fn per_entry() -> (Vec<f64>, Vec<f64>) {
    (vec![1.0; 2], vec![0.01; 2])
}

struct Job {
    tokens: Vec<u32>,
    positions: Vec<u32>,
    targets: Vec<u32>,
    hidden_at: Vec<u32>,
    scale: f32,
}

impl Job {
    fn ok() -> Self {
        Self::nth(0)
    }

    /// A different valid image and label for each `k`.
    fn nth(k: usize) -> Self {
        Self {
            tokens: (0..PIXELS)
                .map(|i| ((i * 7 + k * 13) % 256) as u32)
                .collect(),
            positions: vec![0],
            targets: vec![(k % CLASSES) as u32],
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

fn acc(p: &mut P, job: &Job, mode: BankMode) -> Result<f64, StepError> {
    p.accumulate(&job.view(), mode, None)
}

/// `n` full steps: Overwrite one image, then AdamW at the next step number.
fn advance(p: &mut P, n: usize) {
    let (s, d) = per_entry();
    for _ in 0..n {
        let k = p.step_count() as usize;
        let next = p.step_count() + 1;
        acc(p, &Job::nth(k), BankMode::Overwrite).unwrap();
        p.adamw_step(&hyper(), next, &s, &d).unwrap();
    }
}

type AdamCase = (&'static str, AdamWHyper, u64, Vec<f64>, Vec<f64>);

fn rel_close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs())
}

fn is_contract(e: &StepError) -> bool {
    matches!(e, StepError::Contract(_))
}

// ---------------------------------------------------------------------------------------------
// SequenceJob edge cases
// ---------------------------------------------------------------------------------------------

/// Every malformed job: refused with the expected variant (`SequenceJob::validate`,
/// `image_mlp.rs:236-271`), a refused `Add` keeps the sum (`accumulate` only clears on
/// `Overwrite`, `image_mlp.rs:394-398`), a refused `Overwrite` empties the bank so neither
/// `grad_sq_norm` nor `adamw_step` can run on the previous step's sum, and no parameter or moment
/// moves.
#[test]
fn malformed_jobs_are_refused_keep_an_add_sum_and_clear_on_overwrite() {
    let toks = Job::ok().tokens;
    let mut high_first = toks.clone();
    high_first[0] = 256;
    let mut high_last = toks.clone();
    *high_last.last_mut().unwrap() = 256;
    let mut max_tok = toks.clone();
    max_tok[5] = u32::MAX;
    let mut two_images = toks.clone();
    two_images.extend_from_slice(&toks);

    // (what, job, expected variant is Unsupported)
    let cases: Vec<(&str, Job, bool)> = vec![
        (
            "empty tokens",
            Job {
                tokens: vec![],
                ..Job::ok()
            },
            false,
        ),
        (
            "token 256 first",
            Job {
                tokens: high_first,
                ..Job::ok()
            },
            false,
        ),
        (
            "token 256 last",
            Job {
                tokens: high_last,
                ..Job::ok()
            },
            false,
        ),
        (
            "token u32::MAX",
            Job {
                tokens: max_tok,
                ..Job::ok()
            },
            false,
        ),
        (
            "one pixel short",
            Job {
                tokens: toks[1..].to_vec(),
                ..Job::ok()
            },
            false,
        ),
        (
            "one pixel long",
            Job {
                tokens: [toks.clone(), vec![0]].concat(),
                ..Job::ok()
            },
            false,
        ),
        (
            "two images concatenated",
            Job {
                tokens: two_images,
                ..Job::ok()
            },
            false,
        ),
        (
            "positions [0,0] targets [1,1]",
            Job {
                positions: vec![0, 0],
                targets: vec![1, 1],
                ..Job::ok()
            },
            false,
        ),
        (
            "positions [0,0] targets [1]",
            Job {
                positions: vec![0, 0],
                targets: vec![1],
                ..Job::ok()
            },
            false,
        ),
        (
            "positions [0] targets []",
            Job {
                targets: vec![],
                ..Job::ok()
            },
            false,
        ),
        (
            "positions [] targets [3]",
            Job {
                positions: vec![],
                ..Job::ok()
            },
            false,
        ),
        (
            "position u32::MAX",
            Job {
                positions: vec![u32::MAX],
                ..Job::ok()
            },
            false,
        ),
        (
            "position == pixels",
            Job {
                positions: vec![PIXELS as u32],
                ..Job::ok()
            },
            false,
        ),
        (
            "label == classes",
            Job {
                targets: vec![CLASSES as u32],
                ..Job::ok()
            },
            false,
        ),
        (
            "label u32::MAX",
            Job {
                targets: vec![u32::MAX],
                ..Job::ok()
            },
            false,
        ),
        (
            "scale NaN",
            Job {
                scale: f32::NAN,
                ..Job::ok()
            },
            false,
        ),
        (
            "scale +inf",
            Job {
                scale: f32::INFINITY,
                ..Job::ok()
            },
            false,
        ),
        (
            "scale -inf",
            Job {
                scale: f32::NEG_INFINITY,
                ..Job::ok()
            },
            false,
        ),
        (
            "scale -1",
            Job {
                scale: -1.0,
                ..Job::ok()
            },
            false,
        ),
        (
            "scale -min_positive",
            Job {
                scale: -f32::MIN_POSITIVE,
                ..Job::ok()
            },
            false,
        ),
        (
            "hidden_at duplicate",
            Job {
                hidden_at: vec![0, 0],
                ..Job::ok()
            },
            false,
        ),
        (
            "hidden_at out of range",
            Job {
                hidden_at: vec![PIXELS as u32],
                ..Job::ok()
            },
            false,
        ),
        (
            "hidden_at [0]",
            Job {
                hidden_at: vec![0],
                ..Job::ok()
            },
            true,
        ),
        (
            "hidden_at [0, 5]",
            Job {
                hidden_at: vec![0, 5],
                ..Job::ok()
            },
            true,
        ),
    ];
    let (s, d) = per_entry();
    for (what, job, unsupported) in &cases {
        let p = &mut fresh();
        acc(p, &Job::ok(), BankMode::Overwrite).unwrap();
        let kept = p.grad_sq_norm().unwrap();
        let before = state(p);

        let err = acc(p, job, BankMode::Add).expect_err(what);
        assert_eq!(
            matches!(err, StepError::Unsupported(_)),
            *unsupported,
            "{what}: {err}"
        );
        assert_eq!(is_contract(&err), !*unsupported, "{what}: {err}");
        assert_eq!(
            p.grad_sq_norm().unwrap().to_bits(),
            kept.to_bits(),
            "{what}: Add moved the bank"
        );

        let err = acc(p, job, BankMode::Overwrite).expect_err(what);
        assert_eq!(
            matches!(err, StepError::Unsupported(_)),
            *unsupported,
            "{what}: {err}"
        );
        assert!(
            p.grad_sq_norm().is_err(),
            "{what}: a refused Overwrite left the old sum"
        );
        assert!(
            p.adamw_step(&hyper(), 1, &s, &d).is_err(),
            "{what}: AdamW ran on a cleared bank"
        );
        assert!(
            acc(p, &Job::ok(), BankMode::Add).is_err(),
            "{what}: Add extended a cleared bank"
        );
        assert_eq!(state(p), before, "{what}: parameters or moments moved");
        assert_eq!(p.step_count(), 0, "{what}");
    }
}

/// `scale == -0.0` passes `validate` (`-0.0 >= 0.0`, `SequenceJob::validate`) and takes the
/// value-only branch (`-0.0 > 0.0` is false, `image_mlp.rs:300,312`): the same value as any scale,
/// zero gradients. A subnormal positive scale is a real backward whose gradient may flush to zero
/// but never to a non-finite value.
#[test]
fn zero_negative_zero_and_subnormal_scales() {
    let p = &mut fresh();
    let reference = acc(
        p,
        &Job {
            scale: 1.0,
            ..Job::ok()
        },
        BankMode::Overwrite,
    )
    .unwrap();
    for scale in [0.0f32, -0.0f32] {
        let v = acc(p, &Job { scale, ..Job::ok() }, BankMode::Overwrite).unwrap();
        assert_eq!(v.to_bits(), reference.to_bits(), "scale {scale:?}");
        assert_eq!(p.grad_sq_norm().unwrap(), 0.0, "scale {scale:?}");
    }
    for scale in [1e-45f32, 1e-40, f32::MIN_POSITIVE / 2.0] {
        let err = acc(p, &Job { scale, ..Job::ok() }, BankMode::Overwrite).expect_err("subnormal");
        assert!(is_contract(&err), "scale {scale:?}: {err}");
        assert!(
            p.grad_sq_norm().is_err(),
            "scale {scale:?}: a refused Overwrite left a sum"
        );
    }
    for scale in [f32::MIN_POSITIVE, 1e-30] {
        let v = acc(p, &Job { scale, ..Job::ok() }, BankMode::Overwrite).unwrap();
        assert_eq!(
            v.to_bits(),
            reference.to_bits(),
            "scale {scale:?}: the value is the unscaled loss"
        );
        let n = p.grad_sq_norm().unwrap();
        assert!(n.is_finite() && n >= 0.0, "scale {scale:?}: {n}");
    }
    // A zero-scale Add leaves a real sum untouched, bit for bit.
    acc(p, &Job::ok(), BankMode::Overwrite).unwrap();
    let kept = p.grad_sq_norm().unwrap();
    acc(
        p,
        &Job {
            scale: 0.0,
            ..Job::ok()
        },
        BankMode::Add,
    )
    .unwrap();
    assert_eq!(p.grad_sq_norm().unwrap().to_bits(), kept.to_bits());
}

/// A finite but enormous scale (`validate` accepts anything finite) puts huge or infinite
/// gradients in the bank. Whatever `accumulate` and `grad_sq_norm` say, AdamW must refuse
/// (`check_entry` flags any `g * g` overflow, `optim.rs:218`), move nothing, and a following
/// sane Overwrite+step must still work. With `grad_scale` set to the clip coefficient the loop
/// would pass, a 1e30-scaled bank is steppable.
#[test]
fn enormous_scales_never_reach_the_parameters() {
    let (s, d) = per_entry();
    let sane = acc(&mut fresh(), &Job::ok(), BankMode::Overwrite).unwrap();
    for scale in [1e30f32, 1e38, f32::MAX] {
        let p = &mut fresh();
        let before = state(p);
        let r = acc(p, &Job { scale, ..Job::ok() }, BankMode::Overwrite);
        let norm = p.grad_sq_norm();
        eprintln!("scale {scale:e}: accumulate {r:?}, grad_sq_norm {norm:?}");
        match &r {
            Ok(v) => assert_eq!(
                v.to_bits(),
                sane.to_bits(),
                "the loss does not depend on the scale"
            ),
            Err(_) => assert!(norm.is_err(), "a failed Overwrite must leave an empty bank"),
        }
        if r.is_ok() {
            assert!(
                p.adamw_step(&hyper(), 1, &s, &d).is_err(),
                "scale {scale:e}: AdamW took an overflowing gradient"
            );
            assert_eq!(
                state(p),
                before,
                "scale {scale:e}: refused AdamW moved state"
            );
            assert_eq!(p.step_count(), 0);
            // The refused step consumed the bank (take_grads precedes the finiteness check,
            // image_mlp.rs:426): a retry of the same step is Unfilled, an Add has no base.
            assert!(
                p.grad_sq_norm().is_err(),
                "characterization: the bank is consumed"
            );
            assert!(acc(p, &Job::ok(), BankMode::Add).is_err());
        }
        advance(p, 1);
        assert_eq!(
            p.step_count(),
            1,
            "scale {scale:e}: no recovery after a refusal"
        );
    }

    let p = &mut fresh();
    let before = state(p);
    acc(
        p,
        &Job {
            scale: 1e30,
            ..Job::ok()
        },
        BankMode::Overwrite,
    )
    .unwrap();
    match p.grad_sq_norm() {
        Ok(n) => {
            assert!(n.is_finite() && n > 1e50, "{n}");
            let clipped = AdamWHyper {
                grad_scale: 1.0 / n.sqrt(),
                ..hyper()
            };
            p.adamw_step(&clipped, 1, &s, &d).unwrap();
            let after = state(p);
            assert_ne!(after.0, before.0);
            assert!(after
                .0
                .iter()
                .flatten()
                .all(|b| f32::from_bits(*b).is_finite()));
        }
        Err(e) => eprintln!("1e30 scale overflowed the f32 gradient: {e}"),
    }
}

// ---------------------------------------------------------------------------------------------
// BankMode sequences
// ---------------------------------------------------------------------------------------------

#[test]
fn bank_mode_sequences() {
    let (s, d) = per_entry();
    let single = |job: &Job| {
        let p = &mut fresh();
        acc(p, job, BankMode::Overwrite).unwrap();
        p.grad_sq_norm().unwrap()
    };
    let (j0, j1) = (Job::nth(0), Job::nth(1));

    // Add before any Overwrite: no base, refused, bank still empty.
    let p = &mut fresh();
    assert!(acc(p, &j0, BankMode::Add).is_err());
    assert!(p.grad_sq_norm().is_err());
    assert!(p.adamw_step(&hyper(), 1, &s, &d).is_err());
    assert_eq!(p.step_count(), 0);

    // Two Overwrites: the second replaces, it does not accumulate.
    acc(p, &j0, BankMode::Overwrite).unwrap();
    acc(p, &j1, BankMode::Overwrite).unwrap();
    assert_eq!(p.grad_sq_norm().unwrap().to_bits(), single(&j1).to_bits());

    // Overwrite + Add of the same job doubles every gradient: the norm is exactly 4x.
    acc(p, &j0, BankMode::Overwrite).unwrap();
    acc(p, &j0, BankMode::Add).unwrap();
    assert!(rel_close(
        p.grad_sq_norm().unwrap(),
        4.0 * single(&j0),
        1e-9
    ));

    // Overwrite ok, Overwrite refused, Add: the refusal emptied the bank, the Add has no base.
    acc(p, &j0, BankMode::Overwrite).unwrap();
    let bad = Job {
        hidden_at: vec![0],
        ..Job::nth(0)
    };
    assert!(acc(p, &bad, BankMode::Overwrite).is_err());
    assert!(acc(p, &j0, BankMode::Add).is_err());
    assert!(p.grad_sq_norm().is_err());
    // ...and a fresh Overwrite opens the next step normally.
    acc(p, &j0, BankMode::Overwrite).unwrap();
    assert_eq!(p.grad_sq_norm().unwrap().to_bits(), single(&j0).to_bits());

    // Accumulate after a refused adamw_step: a step-number refusal precedes take_grads
    // (image_mlp.rs:419-426) so the bank survives and an Add extends it.
    let kept = p.grad_sq_norm().unwrap();
    assert!(p.adamw_step(&hyper(), 5, &s, &d).is_err());
    assert_eq!(
        p.grad_sq_norm().unwrap().to_bits(),
        kept.to_bits(),
        "a refused step consumed the bank"
    );
    acc(p, &j0, BankMode::Add).unwrap();
    assert!(rel_close(p.grad_sq_norm().unwrap(), 4.0 * kept, 1e-9));
    p.adamw_step(&hyper(), 1, &s, &d).unwrap();
    assert_eq!(p.step_count(), 1);
    // A successful step consumes the bank: no second step on the same sum, no Add onto nothing.
    assert!(p.grad_sq_norm().is_err());
    assert!(acc(p, &j0, BankMode::Add).is_err());
    let after = state(p);
    assert!(p.adamw_step(&hyper(), 2, &s, &d).is_err());
    assert_eq!(state(p), after);
    assert_eq!(p.step_count(), 1);
}

// ---------------------------------------------------------------------------------------------
// adamw_step
// ---------------------------------------------------------------------------------------------

/// Every refusal that precedes `take_grads` (`hyper.validate`, `check_per_entry`, the step-number
/// check, `image_mlp.rs:417-425`) leaves parameters, both moments, the step count AND the bank
/// untouched. The provider is at step 1, so the only acceptable next step is 2.
#[test]
fn adamw_refusals_move_nothing_and_keep_the_bank() {
    let p = &mut fresh();
    advance(p, 1);
    acc(p, &Job::nth(1), BankMode::Overwrite).unwrap();
    let before = state(p);
    let kept = p.grad_sq_norm().unwrap();
    let (s, d) = per_entry();
    let h = hyper();

    let nan = f64::NAN;
    let inf = f64::INFINITY;
    let cases: Vec<AdamCase> = vec![
        ("step 0", h, 0, s.clone(), d.clone()),
        ("step 1 (replay)", h, 1, s.clone(), d.clone()),
        ("step current+2", h, 3, s.clone(), d.clone()),
        ("step u64::MAX", h, u64::MAX, s.clone(), d.clone()),
        ("lr_scale empty", h, 2, vec![], d.clone()),
        ("lr_scale 1 long", h, 2, vec![1.0], d.clone()),
        ("lr_scale 3 long", h, 2, vec![1.0; 3], d.clone()),
        ("weight_decay empty", h, 2, s.clone(), vec![]),
        ("weight_decay 1 long", h, 2, s.clone(), vec![0.0]),
        ("weight_decay 3 long", h, 2, s.clone(), vec![0.0; 3]),
        ("lr_scale NaN", h, 2, vec![nan, 1.0], d.clone()),
        ("lr_scale inf", h, 2, vec![1.0, inf], d.clone()),
        ("lr_scale 0", h, 2, vec![0.0, 1.0], d.clone()),
        ("lr_scale negative", h, 2, vec![1.0, -1.0], d.clone()),
        ("weight_decay NaN", h, 2, s.clone(), vec![nan, 0.0]),
        ("weight_decay negative", h, 2, s.clone(), vec![0.0, -0.1]),
        ("weight_decay inf", h, 2, s.clone(), vec![inf, 0.0]),
        (
            "lr NaN",
            AdamWHyper { lr: nan, ..h },
            2,
            s.clone(),
            d.clone(),
        ),
        (
            "lr inf",
            AdamWHyper { lr: inf, ..h },
            2,
            s.clone(),
            d.clone(),
        ),
        ("lr 0", AdamWHyper { lr: 0.0, ..h }, 2, s.clone(), d.clone()),
        (
            "lr negative",
            AdamWHyper { lr: -1e-3, ..h },
            2,
            s.clone(),
            d.clone(),
        ),
        (
            "beta1 1",
            AdamWHyper { beta1: 1.0, ..h },
            2,
            s.clone(),
            d.clone(),
        ),
        (
            "beta1 NaN",
            AdamWHyper { beta1: nan, ..h },
            2,
            s.clone(),
            d.clone(),
        ),
        (
            "beta2 negative",
            AdamWHyper { beta2: -0.1, ..h },
            2,
            s.clone(),
            d.clone(),
        ),
        (
            "eps 0",
            AdamWHyper { eps: 0.0, ..h },
            2,
            s.clone(),
            d.clone(),
        ),
        (
            "eps NaN",
            AdamWHyper { eps: nan, ..h },
            2,
            s.clone(),
            d.clone(),
        ),
        (
            "grad_scale 0",
            AdamWHyper {
                grad_scale: 0.0,
                ..h
            },
            2,
            s.clone(),
            d.clone(),
        ),
        (
            "grad_scale NaN",
            AdamWHyper {
                grad_scale: nan,
                ..h
            },
            2,
            s.clone(),
            d.clone(),
        ),
        (
            "grad_scale inf",
            AdamWHyper {
                grad_scale: inf,
                ..h
            },
            2,
            s.clone(),
            d.clone(),
        ),
        (
            "grad_scale negative",
            AdamWHyper {
                grad_scale: -1.0,
                ..h
            },
            2,
            s.clone(),
            d.clone(),
        ),
    ];
    for (what, hy, step, scale, decay) in &cases {
        let err = p.adamw_step(hy, *step, scale, decay).expect_err(what);
        assert!(is_contract(&err), "{what}: {err}");
        assert_eq!(state(p), before, "{what}: state moved");
        assert_eq!(p.step_count(), 1, "{what}");
        assert_eq!(
            p.grad_sq_norm().unwrap().to_bits(),
            kept.to_bits(),
            "{what}: the bank was consumed"
        );
    }

    // Control: the valid call goes through and advances exactly one step.
    p.adamw_step(&h, 2, &s, &d).unwrap();
    assert_eq!(p.step_count(), 2);
    assert_ne!(state(p).0, before.0);
}

/// A finite lr times a finite lr_scale can overflow to infinity (`hyper.lr * lr_scale[i]`,
/// `image_mlp.rs:448`). `check_adamw` refuses a non-finite lr before any update, so nothing moves
/// (the bank is consumed by then, which is fine).
#[test]
fn an_overflowing_effective_lr_is_refused_before_any_update() {
    let p = &mut fresh();
    acc(p, &Job::ok(), BankMode::Overwrite).unwrap();
    let before = state(p);
    let big = AdamWHyper {
        lr: 1e308,
        ..hyper()
    };
    let (_, d) = per_entry();
    assert!(p.adamw_step(&big, 1, &[1e10, 1.0], &d).is_err());
    assert_eq!(state(p), before);
    assert_eq!(p.step_count(), 0);
}

/// A finite lr whose update overflows f32 must not write infinities into the parameters
/// ("fail closed"): the validity checks look at the PRE-update state only
/// (`check_entry`, `optim.rs:190-227`). Expected to fail: lr 1e300 is accepted by
/// `AdamWHyper::validate` (finite, > 0) and `check_adamw`.
#[test]
fn finding_a_huge_finite_lr_must_not_write_non_finite_parameters() {
    let p = &mut fresh();
    acc(p, &Job::ok(), BankMode::Overwrite).unwrap();
    let before = state(p);
    let (s, d) = per_entry();
    let r = p.adamw_step(
        &AdamWHyper {
            lr: 1e300,
            ..hyper()
        },
        1,
        &s,
        &d,
    );
    let after = state(p);
    let finite = after
        .0
        .iter()
        .flatten()
        .all(|b| f32::from_bits(*b).is_finite());
    assert!(
        r.is_err() || finite,
        "adamw_step returned Ok and left non-finite parameters (step_count {})",
        p.step_count()
    );
    if r.is_err() {
        assert_eq!(after, before);
    }
}

/// A bank that really holds a NaN or infinity must never reach AdamW. The only way to put a
/// non-finite value in without a poisoned file is the spec: `std = 1e-40` is finite and positive
/// (`Normalization::new` accepts it) but `1 / 1e-40` overflows f32, so the normalised input holds
/// infinities. Whether `open` accepts it is the next test's question; if it does, the invariants
/// below must hold.
#[test]
fn a_non_finite_bank_never_reaches_adamw() {
    let bad = ImageMlpSpec {
        mean: vec![0.0],
        std: vec![1e-40],
        ..spec()
    };
    let Ok(mut p) = ImageMlpProvider::open(cpu(), bad) else {
        return;
    };
    let before = state(&mut p);
    let r = acc(
        &mut p,
        &Job {
            scale: 1.0,
            ..Job::ok()
        },
        BankMode::Overwrite,
    );
    let norm = p.grad_sq_norm();
    let (s, d) = per_entry();
    let step = p.adamw_step(&hyper(), 1, &s, &d);
    eprintln!("accumulate {r:?}, norm {norm:?}, adamw {step:?}");
    if r.is_ok() && norm.is_err() {
        assert!(step.is_err(), "a non-finite gradient reached AdamW");
    }
    if step.is_err() {
        assert_eq!(state(&mut p), before);
        assert_eq!(p.step_count(), 0);
    } else {
        assert!(state(&mut p)
            .0
            .iter()
            .flatten()
            .all(|b| f32::from_bits(*b).is_finite()));
    }
}

// ---------------------------------------------------------------------------------------------
// ImageMlpSpec validation
// ---------------------------------------------------------------------------------------------

#[test]
fn spec_validation_is_a_typed_error() {
    let nan = f32::NAN;
    let cases: Vec<(&str, ImageMlpSpec)> = vec![
        (
            "mean 1 std 2",
            ImageMlpSpec {
                mean: vec![0.1],
                std: vec![1.0, 1.0],
                ..spec()
            },
        ),
        (
            "mean 2 std 1",
            ImageMlpSpec {
                mean: vec![0.1, 0.2],
                std: vec![1.0],
                ..spec()
            },
        ),
        (
            "no channels",
            ImageMlpSpec {
                mean: vec![],
                std: vec![],
                ..spec()
            },
        ),
        (
            "4 channels (> MAX_CHANNELS, divides 784)",
            ImageMlpSpec {
                mean: vec![0.0; 4],
                std: vec![1.0; 4],
                ..spec()
            },
        ),
        (
            "std 0",
            ImageMlpSpec {
                std: vec![0.0],
                ..spec()
            },
        ),
        (
            "std negative",
            ImageMlpSpec {
                std: vec![-1.0],
                ..spec()
            },
        ),
        (
            "std NaN",
            ImageMlpSpec {
                std: vec![nan],
                ..spec()
            },
        ),
        (
            "std inf",
            ImageMlpSpec {
                std: vec![f32::INFINITY],
                ..spec()
            },
        ),
        (
            "mean NaN",
            ImageMlpSpec {
                mean: vec![nan],
                ..spec()
            },
        ),
        (
            "mean inf",
            ImageMlpSpec {
                mean: vec![f32::NEG_INFINITY],
                ..spec()
            },
        ),
        (
            "3 channels do not divide 784",
            ImageMlpSpec {
                mean: vec![0.0; 3],
                std: vec![1.0; 3],
                ..spec()
            },
        ),
        (
            "pixels 0",
            ImageMlpSpec {
                pixels: 0,
                ..spec()
            },
        ),
        (
            "hidden 0",
            ImageMlpSpec {
                hidden: 0,
                ..spec()
            },
        ),
        (
            "classes 0",
            ImageMlpSpec {
                classes: 0,
                ..spec()
            },
        ),
    ];
    for (what, s) in cases {
        let err = ImageMlpProvider::open(cpu(), s).err().expect(what);
        assert!(is_contract(&err), "{what}: {err}");
    }

    // Control: a 3-channel spec whose pixels divide opens and trains one finite image.
    let rgb = ImageMlpSpec {
        pixels: 786,
        mean: vec![0.4, 0.5, 0.6],
        std: vec![0.2, 0.3, 0.4],
        ..spec()
    };
    let mut p = ImageMlpProvider::open(cpu(), rgb).unwrap();
    let job = Job {
        tokens: (0..786).map(|i| (i % 256) as u32).collect(),
        ..Job::ok()
    };
    let v = acc(&mut p, &job, BankMode::Overwrite).unwrap();
    assert!(v.is_finite() && v > 0.0);
}

/// `pixels * hidden` overflowing usize, or fitting but asking for exabytes, is a typed error from
/// `init_param` (`checked_mul`, `try_reserve_exact`), not a panic and not an attempt to allocate:
/// each is refused before a single weight is drawn. (Sizes that fit in virtual memory but exceed
/// the backend budget are deliberately not tried: `open` has no budget check before the host Vec
/// is filled, see the report.)
#[test]
fn absurd_sizes_are_refused_not_allocated() {
    let cases = [
        (
            "hidden usize::MAX x pixels 784",
            ImageMlpSpec {
                hidden: usize::MAX,
                ..spec()
            },
        ),
        (
            "pixels usize::MAX x hidden 16",
            ImageMlpSpec {
                pixels: usize::MAX,
                ..spec()
            },
        ),
        (
            "2^31 x 2^31 elements (bytes overflow u64)",
            ImageMlpSpec {
                hidden: 1 << 31,
                pixels: 1 << 31,
                ..spec()
            },
        ),
        (
            "2^40 x 2^20 elements (4 EiB)",
            ImageMlpSpec {
                hidden: 1 << 40,
                pixels: 1 << 20,
                ..spec()
            },
        ),
        (
            "pixels 2^60 hidden 1",
            ImageMlpSpec {
                hidden: 1,
                pixels: 1 << 60,
                ..spec()
            },
        ),
    ];
    for (what, s) in cases {
        assert!(ImageMlpProvider::open(cpu(), s).is_err(), "{what}");
    }
}

/// A normalisation that is individually valid but whose table is not finite (`1 / 1e-40`
/// overflows f32 in `ImageMlpSpec::lut`, `image_mlp.rs:118-130`) makes every image an infinity
/// vector and every loss NaN. Expected to fail: `validate` (`image_mlp.rs:98-115`) only checks
/// `Normalization::new` and the channel split.
#[test]
fn finding_a_normalization_whose_table_overflows_must_be_refused() {
    for (mean, std) in [(0.0f32, 1e-40f32), (3e38, 1e-30), (-3e38, 0.5e-30)] {
        let s = ImageMlpSpec {
            mean: vec![mean],
            std: vec![std],
            ..spec()
        };
        assert!(
            ImageMlpProvider::open(cpu(), s).is_err(),
            "mean {mean:e} std {std:e}: open accepted a spec whose LUT holds non-finite values"
        );
    }
}

/// `vocab_size` is the pixel alphabet (256) and `SequenceJob::validate(256)` also bounds the
/// row TARGET by it (`step.rs:183`), so a classifier with more than 256 classes cannot be
/// trained on a label >= 256 although `label < classes` is the stated contract
/// (`image_mlp.rs:263-269`). Expected to fail. (`ImageItem.label` is a u8, so only the provider
/// is affected, not the loop.)
#[test]
fn finding_a_classifier_wider_than_the_label_alphabet_is_refused_at_open() {
    let wide = ImageMlpProvider::open(
        cpu(),
        ImageMlpSpec {
            classes: 257,
            ..spec()
        },
    );
    match wide {
        Err(StepError::Contract(m)) => assert!(m.contains("vocabulary"), "{m}"),
        Err(e) => panic!("expected a contract refusal, got {e}"),
        Ok(_) => panic!("257 classes opened, but a label 256 can never be sent"),
    }
    let mut p = ImageMlpProvider::open(
        cpu(),
        ImageMlpSpec {
            classes: 256,
            ..spec()
        },
    )
    .unwrap();
    let job = Job {
        targets: vec![255],
        ..Job::ok()
    };
    assert!(acc(&mut p, &job, BankMode::Overwrite).is_ok());
}

/// One class: the cross-entropy is exactly 0 and its gradient is 0; AdamW on a zero gradient is
/// a legal step.
#[test]
fn a_single_class_has_zero_loss_and_a_legal_step() {
    let mut p = ImageMlpProvider::open(
        cpu(),
        ImageMlpSpec {
            classes: 1,
            ..spec()
        },
    )
    .unwrap();
    let job = Job {
        targets: vec![0],
        ..Job::ok()
    };
    let v = acc(&mut p, &job, BankMode::Overwrite).unwrap();
    assert!(v.abs() < 1e-6, "{v}");
    let (s, d) = per_entry();
    p.adamw_step(&hyper(), 1, &s, &d).unwrap();
    assert!(p
        .read_parameters()
        .unwrap()
        .iter()
        .flatten()
        .all(|x| x.is_finite()));
}

// ---------------------------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------------------------

fn ctx(step: u64) -> CheckpointContext {
    CheckpointContext {
        optimizer_step: step,
        seed: SEED,
        epoch: 0,
        consumed_digest: "ab".repeat(32),
    }
}

/// A provider that took `n` steps and its checkpoint under a fresh directory.
fn saved(n: usize) -> (Tmp, P, CheckpointContext) {
    let tmp = Tmp::new();
    let mut p = fresh();
    advance(&mut p, n);
    let c = ctx(n as u64);
    let files = p.save_state_with(&tmp.0, &c).unwrap();
    assert_eq!(files, vec![tmp.0.join(STATE_FILE)]);
    (tmp, p, c)
}

struct StFile {
    tensors: Vec<(String, StDtype, Vec<u64>, Vec<f32>)>,
    meta: Vec<(String, String)>,
}

fn read_file(path: &Path) -> StFile {
    let f = SafeTensors::open(path).unwrap();
    StFile {
        tensors: f
            .names()
            .map(|n| {
                let dtype = f.info(n).unwrap().dtype;
                let (shape, values) = f.read_f32_widened(n).unwrap();
                (n.to_owned(), dtype, shape, values)
            })
            .collect(),
        meta: f
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    }
}

fn write_file(path: &Path, f: &StFile) {
    let data: Vec<Vec<u8>> = f
        .tensors
        .iter()
        .map(|(_, dt, _, v)| encode_f32_as(*dt, v).unwrap())
        .collect();
    let items: Vec<TensorOut<'_>> = f
        .tensors
        .iter()
        .zip(&data)
        .map(|((name, dtype, shape, _), bytes)| TensorOut {
            name,
            dtype: *dtype,
            shape,
            data: bytes,
        })
        .collect();
    let meta: Vec<(&str, &str)> = f
        .meta
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    write_safetensors(path, &items, &meta).unwrap();
}

fn set_meta(f: &mut StFile, key: &str, value: &str) {
    f.meta.iter_mut().find(|(k, _)| k == key).unwrap().1 = value.to_owned();
}

fn meta_of(f: &StFile, key: &str) -> String {
    f.meta.iter().find(|(k, _)| k == key).unwrap().1.clone()
}

/// Load a damaged copy of `src` into `q` under `c` and assert the load was refused and changed
/// NOTHING: parameters, both moments, the step count and the bank (a refused load must not clear
/// a bank a caller is accumulating into).
fn assert_refused(
    q: &mut P,
    src: &Path,
    c: &CheckpointContext,
    what: &str,
    damage: impl FnOnce(&mut StFile),
) {
    let mut f = read_file(src);
    damage(&mut f);
    let dir = Tmp::new();
    write_file(&dir.0.join(STATE_FILE), &f);
    let (st, step, norm) = (
        state(q),
        q.step_count(),
        q.grad_sq_norm().unwrap().to_bits(),
    );
    let r = q.load_state_with(&dir.0, c);
    assert!(r.is_err(), "{what}: the load was accepted");
    assert_eq!(state(q), st, "{what}: a refused load moved state");
    assert_eq!(
        q.step_count(),
        step,
        "{what}: a refused load moved the step"
    );
    assert_eq!(
        q.grad_sq_norm().unwrap().to_bits(),
        norm,
        "{what}: a refused load touched the bank"
    );
}

/// A `q` that is neither the checkpointed state nor empty: a different init, one real step's
/// worth of moments, and a filled bank.
fn victim() -> P {
    let mut q = fresh_seeded(5);
    advance(&mut q, 1);
    acc(&mut q, &Job::nth(3), BankMode::Overwrite).unwrap();
    q
}

#[test]
fn saving_at_the_wrong_step_writes_nothing_and_a_save_is_repeatable() {
    let tmp = Tmp::new();
    let mut p = fresh();
    advance(&mut p, 2);
    for wrong in [0u64, 1, 3, u64::MAX] {
        assert!(
            p.save_state_with(&tmp.0, &ctx(wrong)).is_err(),
            "context step {wrong}"
        );
        assert!(
            !tmp.0.join(STATE_FILE).exists(),
            "context step {wrong}: a file was written"
        );
    }
    assert!(p
        .save_state_with(&tmp.0.join("no-such-dir"), &ctx(2))
        .is_err());
    p.save_state_with(&tmp.0, &ctx(2)).unwrap();
    let first = std::fs::read(tmp.0.join(STATE_FILE)).unwrap();
    p.save_state_with(&tmp.0, &ctx(2)).unwrap();
    assert_eq!(
        first,
        std::fs::read(tmp.0.join(STATE_FILE)).unwrap(),
        "a second save differs"
    );
}

#[test]
fn a_checkpoint_of_another_shape_is_refused_and_moves_nothing() {
    let (tmp, _p, c) = saved(2);
    let others = [
        (
            "hidden",
            ImageMlpSpec {
                hidden: HIDDEN + 1,
                ..spec()
            },
        ),
        (
            "classes",
            ImageMlpSpec {
                classes: CLASSES + 1,
                ..spec()
            },
        ),
        (
            "pixels",
            ImageMlpSpec {
                pixels: PIXELS + 1,
                ..spec()
            },
        ),
    ];
    for (what, s) in others {
        let mut q = ImageMlpProvider::open(cpu(), s).unwrap();
        acc(
            &mut q,
            &Job {
                tokens: vec![1; q_pixels(what)],
                ..Job::ok()
            },
            BankMode::Overwrite,
        )
        .unwrap();
        let (st, norm) = (state(&mut q), q.grad_sq_norm().unwrap().to_bits());
        assert!(q.load_state_with(&tmp.0, &c).is_err(), "{what}");
        assert_eq!(state(&mut q), st, "{what}");
        assert_eq!(q.step_count(), 0, "{what}");
        assert_eq!(q.grad_sq_norm().unwrap().to_bits(), norm, "{what}");
    }
}

fn q_pixels(what: &str) -> usize {
    if what == "pixels" {
        PIXELS + 1
    } else {
        PIXELS
    }
}

/// F16 or BF16 storage is refused (`file.read_f32` requires F32, `image_mlp.rs:581`), whether
/// every tensor or just one, even when the widened values are finite and the shapes right.
#[test]
fn half_precision_tensors_are_refused() {
    let (tmp, _p, c) = saved(2);
    let src = tmp.0.join(STATE_FILE);
    let mut q = victim();
    for (what, dt) in [("all F16", StDtype::F16), ("all BF16", StDtype::BF16)] {
        assert_refused(&mut q, &src, &c, what, |f| {
            f.tensors.iter_mut().for_each(|t| t.1 = dt)
        });
    }
    for (i, dt) in [
        (0, StDtype::F16),
        (2, StDtype::BF16),
        (5, StDtype::BF16),
        (3, StDtype::F16),
    ] {
        assert_refused(&mut q, &src, &c, &format!("tensor {i} as {dt:?}"), |f| {
            f.tensors[i].1 = dt
        });
    }
}

/// The metadata strings are compared to the canonical decimal rendering of the expected value, so
/// any other spelling of the same number is refused: leading zeros, `+`, whitespace, a
/// fractional or exponent form, empty. That is the right behaviour (a checkpoint is a record of
/// exact values, and there is one canonical rendering); the test pins it.
#[test]
fn non_canonical_numbers_in_the_metadata_are_refused() {
    let (tmp, _p, c) = saved(2);
    let src = tmp.0.join(STATE_FILE);
    let mut q = victim();
    let base = read_file(&src);
    for key in ["pixels", "hidden", "classes", "step", "seed", "epoch"] {
        let v = meta_of(&base, key);
        let variants = [
            format!("0{v}"),
            format!("+{v}"),
            format!("-{v}"),
            format!(" {v}"),
            format!("{v} "),
            format!("{v}\n"),
            format!("{v}.0"),
            format!("{v}e0"),
            String::new(),
            "x".to_owned(),
        ];
        for bad in variants {
            assert_refused(&mut q, &src, &c, &format!("{key}={bad:?}"), |f| {
                set_meta(f, key, &bad)
            });
        }
    }
    // Control: the unmodified rewrite loads.
    let dir = Tmp::new();
    write_file(&dir.0.join(STATE_FILE), &base);
    q.load_state_with(&dir.0, &c).unwrap();
}

#[test]
fn missing_extra_and_renamed_metadata_keys_are_refused() {
    let (tmp, _p, c) = saved(2);
    let src = tmp.0.join(STATE_FILE);
    let mut q = victim();
    let keys: Vec<String> = read_file(&src).meta.into_iter().map(|(k, _)| k).collect();
    assert_eq!(keys.len(), 9);
    for key in &keys {
        assert_refused(&mut q, &src, &c, &format!("missing {key}"), |f| {
            f.meta.retain(|(k, _)| k != key)
        });
        assert_refused(&mut q, &src, &c, &format!("renamed {key}"), |f| {
            f.meta.iter_mut().find(|(k, _)| k == key).unwrap().0 = format!("x_{key}");
        });
    }
    assert_refused(&mut q, &src, &c, "extra key", |f| {
        f.meta.push(("extra".into(), "1".into()))
    });
    assert_refused(&mut q, &src, &c, "no metadata", |f| f.meta.clear());
}

/// Every tensor-level defect, each hitting a different point of the read loop so an early
/// `leaves = ...` assignment would be caught: non-finite in the first, a middle and the last
/// tensor read, a transposed w1, a duplicated-by-rename tensor, a missing and an extra one.
#[test]
fn tensor_level_damage_is_refused_and_moves_nothing() {
    let (tmp, _p, c) = saved(2);
    let src = tmp.0.join(STATE_FILE);
    let mut q = victim();
    let names: Vec<String> = read_file(&src).tensors.into_iter().map(|t| t.0).collect();
    assert_eq!(names.len(), 6);
    for name in &names {
        for (label, v) in [
            ("NaN", f32::NAN),
            ("+inf", f32::INFINITY),
            ("-inf", f32::NEG_INFINITY),
        ] {
            assert_refused(&mut q, &src, &c, &format!("{label} in {name}"), |f| {
                let t = f.tensors.iter_mut().find(|t| &t.0 == name).unwrap();
                let last = t.3.len() - 1;
                t.3[last] = v;
            });
        }
        assert_refused(&mut q, &src, &c, &format!("{name} missing"), |f| {
            f.tensors.retain(|t| &t.0 != name)
        });
        assert_refused(&mut q, &src, &c, &format!("{name} renamed"), |f| {
            f.tensors.iter_mut().find(|t| &t.0 == name).unwrap().0 = format!("{name}.x");
        });
    }
    assert_refused(&mut q, &src, &c, "w1 transposed", |f| {
        let t = f
            .tensors
            .iter_mut()
            .find(|t| t.0 == "classifier.w1.weight")
            .unwrap();
        t.2 = vec![PIXELS as u64, HIDDEN as u64];
    });
    assert_refused(&mut q, &src, &c, "w1 flattened", |f| {
        let t = f
            .tensors
            .iter_mut()
            .find(|t| t.0 == "classifier.w1.weight")
            .unwrap();
        t.2 = vec![(PIXELS * HIDDEN) as u64];
    });
    assert_refused(
        &mut q,
        &src,
        &c,
        "adam_m shape differs from its weight",
        |f| {
            let t = f
                .tensors
                .iter_mut()
                .find(|t| t.0 == "classifier.w1.weight.adam_m")
                .unwrap();
            t.2 = vec![PIXELS as u64, HIDDEN as u64];
        },
    );
}

/// Every field of the loop's context, not just the four the existing test varies, including a
/// context whose step agrees with the provider's current step but not the file's.
#[test]
fn a_context_that_is_not_the_files_is_refused() {
    let (tmp, _p, c) = saved(2);
    let mut q = victim();
    let st = state(&mut q);
    for (what, wrong) in [
        (
            "step 0",
            CheckpointContext {
                optimizer_step: 0,
                ..c.clone()
            },
        ),
        (
            "step 1 (the victim's own)",
            CheckpointContext {
                optimizer_step: 1,
                ..c.clone()
            },
        ),
        (
            "step u64::MAX",
            CheckpointContext {
                optimizer_step: u64::MAX,
                ..c.clone()
            },
        ),
        (
            "seed",
            CheckpointContext {
                seed: u64::MAX,
                ..c.clone()
            },
        ),
        (
            "epoch",
            CheckpointContext {
                epoch: u64::MAX,
                ..c.clone()
            },
        ),
        (
            "digest upper case",
            CheckpointContext {
                consumed_digest: c.consumed_digest.to_uppercase(),
                ..c.clone()
            },
        ),
        (
            "digest with a trailing space",
            CheckpointContext {
                consumed_digest: format!("{} ", c.consumed_digest),
                ..c.clone()
            },
        ),
        (
            "digest empty",
            CheckpointContext {
                consumed_digest: String::new(),
                ..c.clone()
            },
        ),
    ] {
        assert!(q.load_state_with(&tmp.0, &wrong).is_err(), "{what}");
        assert_eq!(state(&mut q), st, "{what}");
        assert_eq!(q.step_count(), 1, "{what}");
    }
    assert!(q.load_state_with(&tmp.0.join("missing"), &c).is_err());
    assert!(
        q.load_state_with(&tmp.0.join(STATE_FILE), &c).is_err(),
        "the file given as the directory"
    );
    let dir = Tmp::new();
    std::fs::create_dir(dir.0.join(STATE_FILE)).unwrap();
    assert!(
        q.load_state_with(&dir.0, &c).is_err(),
        "a directory named like the file"
    );
    assert_eq!(state(&mut q), st);
}

/// The digest is an opaque string to the provider: unicode must round-trip byte for byte, and a
/// different Unicode normalisation of the same text is a different string and is refused.
#[test]
fn unicode_digests_round_trip_exactly_and_normalisation_forms_differ() {
    let nfc = "caf\u{e9}-\u{65e5}\u{672c}\u{8a9e}-\u{1F600}";
    let nfd = "cafe\u{301}-\u{65e5}\u{672c}\u{8a9e}-\u{1F600}";
    for digest in [nfc, "", "\"quoted\" \\ back\\slash", "tab\tnewline\n"] {
        let tmp = Tmp::new();
        let mut p = fresh();
        advance(&mut p, 1);
        let c = CheckpointContext {
            consumed_digest: digest.to_owned(),
            ..ctx(1)
        };
        p.save_state_with(&tmp.0, &c).unwrap();
        let mut q = fresh_seeded(5);
        q.load_state_with(&tmp.0, &c).unwrap();
        assert_eq!(state(&mut q), state(&mut p), "digest {digest:?}");
        let f = read_file(&tmp.0.join(STATE_FILE));
        assert_eq!(meta_of(&f, "consumed_digest"), digest);
    }
    let tmp = Tmp::new();
    let mut p = fresh();
    advance(&mut p, 1);
    p.save_state_with(
        &tmp.0,
        &CheckpointContext {
            consumed_digest: nfc.to_owned(),
            ..ctx(1)
        },
    )
    .unwrap();
    let mut q = fresh_seeded(5);
    let st = state(&mut q);
    assert!(q
        .load_state_with(
            &tmp.0,
            &CheckpointContext {
                consumed_digest: nfd.to_owned(),
                ..ctx(1)
            }
        )
        .is_err());
    assert_eq!(state(&mut q), st);
}

/// NUL and other control characters in a metadata value go through the safetensors JSON header:
/// the writer must escape them and the reader unescape them, or refuse at SAVE time. A save that
/// succeeds and a load that then refuses its own file would be a finding.
#[test]
fn a_digest_with_a_nul_either_round_trips_or_is_refused_at_save() {
    let digest = "a\u{0}b\u{1f}c\u{7f}";
    let tmp = Tmp::new();
    let mut p = fresh();
    advance(&mut p, 1);
    let c = CheckpointContext {
        consumed_digest: digest.to_owned(),
        ..ctx(1)
    };
    if p.save_state_with(&tmp.0, &c).is_ok() {
        let mut q = fresh_seeded(5);
        q.load_state_with(&tmp.0, &c).unwrap();
        assert_eq!(state(&mut q), state(&mut p));
    }
}

/// Loading is idempotent, restores an older step over a newer state, empties the bank (a bank
/// summed before the load belongs to the state that was replaced: AdamW would apply it to the
/// loaded parameters), and reproduces special float values bit for bit (-0.0, subnormals).
#[test]
fn load_twice_restores_an_older_state_and_clears_the_bank() {
    let (tmp, mut p, c) = saved(2);

    // Make the saved file hold awkward values by round-tripping them through a rewrite.
    let src = tmp.0.join(STATE_FILE);
    let mut f = read_file(&src);
    f.tensors
        .iter_mut()
        .find(|t| t.0 == "classifier.w1.weight")
        .unwrap()
        .3[0] = -0.0;
    f.tensors
        .iter_mut()
        .find(|t| t.0 == "classifier.w1.weight.adam_v")
        .unwrap()
        .3[1] = 1e-45;
    let special = Tmp::new();
    write_file(&special.0.join(STATE_FILE), &f);

    let mut q = fresh_seeded(5);
    advance(&mut q, 4);
    acc(&mut q, &Job::nth(9), BankMode::Overwrite).unwrap();
    q.load_state_with(&tmp.0, &c).unwrap();
    assert_eq!(
        q.step_count(),
        2,
        "a load must set the step, not keep the newer one"
    );
    assert_eq!(state(&mut q), state(&mut p));
    assert!(
        q.grad_sq_norm().is_err(),
        "the pre-load bank survived the load"
    );
    let (s, d) = per_entry();
    assert!(
        q.adamw_step(&hyper(), 3, &s, &d).is_err(),
        "AdamW applied a pre-load bank"
    );

    q.load_state_with(&tmp.0, &c).unwrap();
    assert_eq!(
        state(&mut q),
        state(&mut p),
        "loading twice changed the state"
    );
    assert_eq!(q.step_count(), 2);

    q.load_state_with(&special.0, &c).unwrap();
    let w1 = q.read_parameters().unwrap().remove(0);
    assert_eq!(w1[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(
        q.read_moments().unwrap()[0].1[1].to_bits(),
        1e-45f32.to_bits()
    );
}

/// Four straight steps equal two steps + save + a fresh provider (other init seed) + load + two
/// more, in every parameter and both moments, bit for bit; and the loaded provider accepts
/// exactly step 3 next.
#[test]
fn load_then_continue_equals_the_uninterrupted_run() {
    let mut straight = fresh();
    advance(&mut straight, 4);

    let (tmp, _first, c) = saved(2);
    let mut resumed = fresh_seeded(99);
    resumed.load_state_with(&tmp.0, &c).unwrap();
    let (s, d) = per_entry();
    for wrong in [0u64, 2, 4] {
        acc(&mut resumed, &Job::nth(2), BankMode::Overwrite).unwrap();
        assert!(
            resumed.adamw_step(&hyper(), wrong, &s, &d).is_err(),
            "step {wrong}"
        );
    }
    advance(&mut resumed, 2);
    assert_eq!(resumed.step_count(), 4);
    assert_eq!(state(&mut resumed), state(&mut straight));
}

/// `save_state_with` of a state that holds a non-finite value must not write a checkpoint that
/// `load_state_with` would then refuse (a checkpoint nothing can resume from). Reached only if a
/// step could produce a non-finite parameter, see the lr finding; vacuous while that is closed.
#[test]
fn a_non_finite_state_is_not_saved_as_a_checkpoint() {
    let mut p = fresh();
    acc(&mut p, &Job::ok(), BankMode::Overwrite).unwrap();
    let (s, d) = per_entry();
    if p.adamw_step(
        &AdamWHyper {
            lr: 1e300,
            ..hyper()
        },
        1,
        &s,
        &d,
    )
    .is_err()
    {
        return;
    }
    let tmp = Tmp::new();
    let finite = p
        .read_parameters()
        .unwrap()
        .iter()
        .flatten()
        .all(|x| x.is_finite());
    if !finite {
        assert!(
            p.save_state_with(&tmp.0, &ctx(1)).is_err(),
            "a non-finite state was saved"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// ClassificationObjective
// ---------------------------------------------------------------------------------------------

fn item(index: u32, fill: u8) -> ImageItem {
    ImageItem {
        index,
        pixels: vec![fill; PIXELS],
        label: (index % 10) as u8,
    }
}

fn batch_of(n: usize) -> ImageBatch {
    ImageBatch {
        index: 4,
        items: (0..n).map(|i| item(i as u32, i as u8)).collect(),
    }
}

#[test]
fn objective_plans_and_closes_batches_of_every_size() {
    let mut o = ClassificationObjective::new();
    assert_eq!(o.run_kind(), "classify");
    assert!(matches!(
        o.plan(&ImageBatch {
            index: 0,
            items: vec![]
        }),
        Err(TrainError::Refused(_))
    ));

    let one = o.plan(&batch_of(1)).unwrap();
    assert_eq!(one.sequences.len(), 1);
    assert_eq!(one.row_scale, 1.0);
    let m = o.close(&[2.5]).unwrap();
    assert_eq!(m.total, 2.5);
    assert!(m.channels.is_empty());

    for n in [3usize, 7, 257] {
        let plan = o.plan(&batch_of(n)).unwrap();
        assert_eq!(plan.sequences.len(), n);
        assert_eq!(plan.row_scale, 1.0 / n as f32);
        assert_eq!(plan.counts.supervised_tokens, n as u64);
        let m = o.close(&vec![1.0; n]).unwrap();
        assert!((m.total - 1.0).abs() < 1e-6, "n {n}: {}", m.total);
    }

    // A plan replaces the previous scale: close after a second plan uses the second.
    o.plan(&batch_of(4)).unwrap();
    o.plan(&batch_of(2)).unwrap();
    assert!((o.close(&[4.0, 4.0]).unwrap().total - 4.0).abs() < 1e-6);
    // A close consumes its plan: closing again is refused.
    assert!(o.close(&[4.0, 4.0]).is_err());
    // NaN and infinite row losses are summed as they come: the loop, not the objective, must
    // refuse a non-finite loss.
    o.plan(&batch_of(2)).unwrap();
    assert!(o.close(&[f64::NAN, 1.0]).unwrap().total.is_nan());
}

/// `close` is handed one loss per planned sequence. A list of the wrong length (a dropped row, an
/// extra row) silently yields a wrong mean (`row_loss_sums.iter().sum() * scale`,
/// `image_mlp.rs:706-711`); and `close` before any `plan` multiplies by a scale of 0 and returns a
/// loss of 0 with no error. Expected to fail.
#[test]
fn finding_close_must_refuse_a_sum_list_that_is_not_the_planned_batch() {
    let mut o = ClassificationObjective::new();
    assert!(
        o.close(&[1.0, 2.0]).is_err(),
        "close before any plan returned a loss"
    );
    o.plan(&batch_of(4)).unwrap();
    for wrong in [vec![], vec![1.0; 3], vec![1.0; 5]] {
        assert!(
            o.close(&wrong).is_err(),
            "{} sums for a batch of 4 were accepted",
            wrong.len()
        );
    }
}

/// The objective does not look inside images: an image of the wrong size, or a label outside the
/// classifier, is planned fine and refused by the provider (`image_mlp.rs:243,264`).
#[test]
fn the_provider_refuses_what_the_objective_plans_blindly() {
    let mut o = ClassificationObjective::new();
    let bad = ImageBatch {
        index: 0,
        items: vec![
            ImageItem {
                index: 0,
                pixels: vec![],
                label: 0,
            },
            ImageItem {
                index: 1,
                pixels: vec![0; PIXELS - 1],
                label: 0,
            },
            ImageItem {
                index: 2,
                pixels: vec![0; PIXELS],
                label: 255,
            },
        ],
    };
    let plan = o.plan(&bad).unwrap();
    let p = &mut fresh();
    for (i, seq) in plan.sequences.iter().enumerate() {
        let job = SequenceJob {
            tokens: &seq.tokens,
            rows: RowTargets {
                positions: &seq.positions,
                targets: &seq.targets,
                scale: plan.row_scale,
            },
            hidden_at: &seq.hidden_at,
        };
        let err = p
            .accumulate(&job, BankMode::Overwrite, None)
            .expect_err("image {i}");
        assert!(is_contract(&err), "image {i}: {err}");
    }
}

#[test]
fn the_consumed_digest_separates_batches_and_is_order_and_size_sensitive() {
    let digest = |b: &ImageBatch| {
        let mut p = ConsumedPrefix::new();
        b.fold_into(&mut p).unwrap();
        p.hexdigest()
    };
    let a = ImageBatch {
        index: 0,
        items: vec![item(1, 0), item(2, 0)],
    };
    let b = ImageBatch {
        index: 0,
        items: vec![item(1, 0)],
    };
    let c = ImageBatch {
        index: 0,
        items: vec![item(2, 0), item(1, 0)],
    };
    let e = ImageBatch {
        index: 0,
        items: vec![],
    };
    let big = ImageBatch {
        index: 0,
        items: vec![item(u32::MAX, 0)],
    };
    let all = [digest(&a), digest(&b), digest(&c), digest(&e), digest(&big)];
    for i in 0..all.len() {
        for j in i + 1..all.len() {
            assert_ne!(all[i], all[j], "batches {i} and {j} collide");
        }
    }
    // Pixels are not part of the digest, only the dataset index: same index, other pixels, same
    // digest (the index identifies the image).
    let d = ImageBatch {
        index: 0,
        items: vec![item(1, 99), item(2, 77)],
    };
    assert_eq!(digest(&a), digest(&d));
    // Two folds are not one fold of the concatenation.
    let mut two = ConsumedPrefix::new();
    b.fold_into(&mut two).unwrap();
    ImageBatch {
        index: 0,
        items: vec![item(2, 0)],
    }
    .fold_into(&mut two)
    .unwrap();
    assert_ne!(two.hexdigest(), digest(&a));
}

// ---------------------------------------------------------------------------------------------
// EpochBatches
// ---------------------------------------------------------------------------------------------

fn collect_all(set: &ImageSet, cfg: ImageSamplerConfig, epoch: u64) -> Vec<ImageBatch> {
    EpochBatches::new(set, cfg, epoch)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

/// `batch == 0` must be a typed refusal. Expected to fail with a panic: `EpochBatches::new`
/// divides `set.len() / cfg.batch` (or `div_ceil`) at `image_mlp.rs:746-750`, BEFORE
/// `ImageBatchSampler::new` gets to refuse a zero batch.
#[test]
fn finding_a_zero_batch_is_refused_not_a_division_by_zero() {
    let set = image_set();
    for drop_last in [true, false] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            matches!(
                EpochBatches::new(&set, sampler_config(0, drop_last), 0),
                Err(TrainError::Refused(_))
            )
        }));
        assert_eq!(
            r.ok(),
            Some(true),
            "drop_last {drop_last}: a zero batch panicked or was accepted"
        );
    }
}

#[test]
fn batches_larger_than_the_set_and_the_drop_last_matrix() {
    let set = image_set();
    // drop_last with batch > N yields no batch: the sampler refuses it.
    for batch in [N + 1, 1 << 20, usize::MAX] {
        assert!(
            matches!(
                EpochBatches::new(&set, sampler_config(batch, true), 0),
                Err(TrainError::Refused(_))
            ),
            "batch {batch} drop_last"
        );
    }
    // Without drop_last one short batch holds the whole set, in sampler order.
    for batch in [N + 1, 1 << 20, 1 << 40] {
        let all = collect_all(&set, sampler_config(batch, false), 0);
        assert_eq!(all.len(), 1, "batch {batch}");
        assert_eq!(all[0].items.len(), N);
        let mut seen: Vec<u32> = all[0].items.iter().map(|i| i.index).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..N as u32).collect::<Vec<_>>());
    }
    // A batch whose pixel count overflows usize is refused by the sampler, not a panic.
    assert!(EpochBatches::new(&set, sampler_config(usize::MAX, false), 0).is_err());

    // batch == N: exactly one batch either way.
    for drop_last in [true, false] {
        assert_eq!(collect_all(&set, sampler_config(N, drop_last), 0).len(), 1);
    }
    // batch 1: N batches of one, indices 0..N.
    let ones = collect_all(&set, sampler_config(1, true), 0);
    assert_eq!(ones.len(), N);
    assert!(ones
        .iter()
        .enumerate()
        .all(|(i, b)| b.index == i as u64 && b.items.len() == 1));
    // batch 10 over 96: 9 full batches dropping the last 6, or 10 batches ending in a short one.
    let dropped = collect_all(&set, sampler_config(10, true), 0);
    assert_eq!(dropped.len(), 9);
    assert!(dropped.iter().all(|b| b.items.len() == 10));
    let kept = collect_all(&set, sampler_config(10, false), 0);
    assert_eq!(kept.len(), 10);
    assert_eq!(kept[9].items.len(), 6);
    assert_eq!(kept[9].index, 9);
    // The dropped run is a prefix of the kept one.
    assert_eq!(dropped[..], kept[..9]);
}

#[test]
fn epochs_are_deterministic_distinct_and_u64_wide() {
    let set = image_set();
    let cfg = sampler_config(8, true);
    let e0 = collect_all(&set, cfg, 0);
    assert_eq!(
        e0,
        collect_all(&set, cfg, 0),
        "an epoch is not reproducible"
    );
    assert_ne!(
        e0,
        collect_all(&set, cfg, 1),
        "epochs 0 and 1 share an order"
    );
    for epoch in [u64::from(u32::MAX), u64::from(u32::MAX) + 1, u64::MAX] {
        let all = collect_all(&set, cfg, epoch);
        assert_eq!(all.len(), N / 8, "epoch {epoch}");
        let mut seen = vec![0u32; N];
        for b in &all {
            for i in &b.items {
                seen[i.index as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "epoch {epoch} does not cover the set once"
        );
    }
    // The iterator is exhausted for good.
    let mut it = EpochBatches::new(&set, cfg, 0).unwrap();
    assert_eq!(it.by_ref().count(), N / 8);
    assert!(it.next().is_none());
    assert!(it.next().is_none());
}

#[test]
fn augmentation_and_a_mismatched_normalization_are_refused() {
    let set = image_set();
    for augment in [
        Augment::CIFAR10,
        Augment {
            crop_pad: 1,
            flip: false,
        },
        Augment {
            crop_pad: 0,
            flip: true,
        },
    ] {
        let cfg = ImageSamplerConfig {
            augment,
            ..sampler_config(8, true)
        };
        assert!(
            matches!(EpochBatches::new(&set, cfg, 0), Err(TrainError::Refused(_))),
            "{augment:?}"
        );
    }
    let rgb = ImageSamplerConfig {
        normalization: Normalization::CIFAR10,
        ..sampler_config(8, true)
    };
    assert!(matches!(
        EpochBatches::new(&set, rgb, 0),
        Err(TrainError::Refused(_))
    ));
}

/// The sampler's own normalization is not what the provider normalises with (the provider uses
/// its spec's mean and std, EpochBatches copies raw u8). Two different `ImageSamplerConfig`
/// normalizations of the same channel count therefore yield identical batches: the field is
/// inert. Pinned as a characterization, not a defect.
#[test]
fn the_sampler_normalization_does_not_change_the_batches() {
    let set = image_set();
    let a = collect_all(&set, sampler_config(8, true), 0);
    let other = ImageSamplerConfig {
        normalization: Normalization::new(&[0.5], &[0.25]).unwrap(),
        ..sampler_config(8, true)
    };
    assert_eq!(a, collect_all(&set, other, 0));
}
