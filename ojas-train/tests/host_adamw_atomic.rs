//! ft-1185100b: a host AdamW step that one entry refuses must leave NO entry stepped.
//!
//! `adamw_update` writes nothing when it refuses (ft-9168a8e8), but its callers step several
//! entries in turn. The contract pinned here, per caller:
//!
//! - [`ToyProvider::adamw_step`] is atomic across entries: it refuses, and moves no value, no
//!   moment and no step counter, whichever entry the refusal is at.
//! - The training loop is atomic across the provider and the host entries: a host entry's
//!   refusal is found before the provider steps, so the provider, every host entry, every host
//!   step count and the optimizer step stay where the last good step left them.

use ojas_train::lappi::mock::ToyProvider;
use ojas_train::lappi::recipe::{BETA1, EPS, OptimizerRecipe, WEIGHT_DECAY};
use ojas_train::lappi::run_control::{ConsumedPrefix, MonotonicClock, WallClockCap};
use ojas_train::lappi::schedule::LrSchedule;
use ojas_train::lappi::step::{AdamWHyper, ParamSpec, StepError, StepProvider};
use ojas_train::lappi::trainer::{
    BatchCounts, ConsumedBatch, Hooks, HostParams, MicroBatchPlan, MicroLoss, Objective,
    SequenceWork, TrainConfig, TrainError, train,
};

const VOCAB: usize = 8;
const HIDDEN: usize = 4;
const LAYERS: usize = 2;
const LR: f64 = 1e-2;
const BETA2: f64 = 0.999;

fn bits(x: &[Vec<f32>]) -> Vec<Vec<u32>> {
    x.iter()
        .map(|v| v.iter().map(|f| f.to_bits()).collect())
        .collect()
}

fn toy() -> ToyProvider {
    ToyProvider::new(VOCAB, HIDDEN, LAYERS, 42).expect("toy provider")
}

fn hyper() -> AdamWHyper {
    AdamWHyper {
        lr: LR,
        beta1: BETA1,
        beta2: BETA2,
        eps: EPS,
        grad_scale: 1.0,
    }
}

/// A provider whose bank holds a small finite gradient everywhere.
fn with_bank(mut p: ToyProvider) -> ToyProvider {
    for (i, g) in p.grads_mut().iter_mut().enumerate() {
        for (k, x) in g.iter_mut().enumerate() {
            *x = 0.05 + 0.01 * ((i + k) % 7) as f32;
        }
    }
    p
}

fn ones(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

fn decays(n: usize) -> Vec<f64> {
    vec![WEIGHT_DECAY; n]
}

/// Plants a refusal at `entry`: 1e21 is finite, but the stored second moment `w2 * g * g`
/// overflows f32.
fn poison(p: &mut ToyProvider, entry: usize) {
    p.grads_mut()[entry][0] = 1e21;
}

fn assert_untouched(p: &ToyProvider, before: &ToyProvider, why: &str) {
    assert_eq!(bits(p.values()), bits(before.values()), "values: {why}");
    for (a, b) in p.moments().iter().zip(before.moments()) {
        assert_eq!(a, b, "moments: {why}");
    }
    assert_eq!(p.step_count(), before.step_count(), "step: {why}");
}

#[test]
fn a_toy_step_refused_at_any_entry_moves_nothing() {
    let n = toy().parameters().len();
    for entry in [0, n / 2, n - 1] {
        let before = with_bank(toy());
        let mut p = with_bank(toy());
        poison(&mut p, entry);
        let e = p
            .adamw_step(&hyper(), 1, &ones(n), &decays(n))
            .expect_err("a non-finite stored moment must be refused");
        assert!(matches!(e, StepError::Contract(_)), "{e}");
        assert_untouched(&p, &before, &format!("refusal at entry {entry}"));
    }
}

#[test]
fn a_toy_step_that_was_refused_can_be_retried_as_the_same_step() {
    let n = toy().parameters().len();
    let mut p = with_bank(toy());
    let good = p.grads()[2][0];
    poison(&mut p, 2);
    p.adamw_step(&hyper(), 1, &ones(n), &decays(n))
        .expect_err("refused");
    // The bank is repaired; the provider must take step 1 as if the refusal never happened.
    p.grads_mut()[2][0] = good;
    p.adamw_step(&hyper(), 1, &ones(n), &decays(n))
        .expect("retry");
    let mut fresh = with_bank(toy());
    fresh
        .adamw_step(&hyper(), 1, &ones(n), &decays(n))
        .expect("fresh");
    assert_eq!(p.step_count(), 1);
    assert_eq!(bits(p.values()), bits(fresh.values()));
    for (a, b) in p.moments().iter().zip(fresh.moments()) {
        assert_eq!(a, b);
    }
}

#[test]
fn a_toy_step_with_a_zero_lr_scale_entry_refuses_whole() {
    let n = toy().parameters().len();
    let before = with_bank(toy());
    let mut p = with_bank(toy());
    let mut scale = ones(n);
    scale[n / 2] = 0.0;
    p.adamw_step(&hyper(), 1, &scale, &decays(n))
        .expect_err("a zero learning rate is refused by the update");
    assert_untouched(&p, &before, "lr_scale 0 at a middle entry");
}

#[test]
fn a_toy_refusal_ahead_of_an_empty_entry_moves_nothing() {
    let make = || {
        with_bank(
            toy()
                .with_trainable_entry("empty.weight", &[0])
                .expect("empty entry"),
        )
    };
    let before = make();
    let mut p = make();
    let n = p.parameters().len();
    poison(&mut p, 0);
    p.adamw_step(&hyper(), 1, &ones(n), &decays(n))
        .expect_err("refused");
    assert_untouched(&p, &before, "refusal before an empty trailing entry");
}

// ---------------------------------------------------------------------------------------------
// The loop: provider + host entries.
// ---------------------------------------------------------------------------------------------

struct Batch(u64);

impl ConsumedBatch for Batch {
    fn index(&self) -> u64 {
        self.0
    }

    fn fold_into(&self, prefix: &mut ConsumedPrefix) -> Result<(), TrainError> {
        prefix.fold(&[&self.0.to_be_bytes()]);
        Ok(())
    }
}

/// Next-token loss over a fixed sequence, with three host tensors whose gradients are set at
/// `close`. `poison` is `(batch index, host entry)`: that entry's gradient is 1e21 there.
struct HostObjective {
    values: Vec<Vec<f32>>,
    grads: Vec<Option<Vec<f32>>>,
    poison: Option<(u64, usize)>,
    batch: u64,
}

impl HostObjective {
    fn new(poison: Option<(u64, usize)>) -> Self {
        Self {
            values: vec![vec![0.5; 3], vec![-0.25; 4], vec![0.125; 2]],
            grads: vec![None, None, None],
            poison,
            batch: 0,
        }
    }
}

impl HostParams for HostObjective {
    fn entries(&self) -> Vec<ParamSpec> {
        self.values
            .iter()
            .enumerate()
            .map(|(i, v)| ParamSpec::new(format!("host.{i}"), &[v.len()]))
            .collect()
    }

    fn zero_grads(&mut self) {
        self.grads.iter_mut().for_each(|g| *g = None);
    }

    fn grads(&self) -> Vec<Option<&[f32]>> {
        self.grads.iter().map(|g| g.as_deref()).collect()
    }

    fn values(&self) -> Vec<&[f32]> {
        self.values.iter().map(Vec::as_slice).collect()
    }

    fn values_and_grads(&mut self) -> Vec<(&mut [f32], Option<&[f32]>)> {
        self.values
            .iter_mut()
            .zip(self.grads.iter())
            .map(|(v, g)| (v.as_mut_slice(), g.as_deref()))
            .collect()
    }
}

impl Objective for HostObjective {
    type Batch = Batch;

    fn run_kind(&self) -> &'static str {
        "ft"
    }

    fn plan(&mut self, batch: &Batch) -> Result<MicroBatchPlan, TrainError> {
        self.batch = batch.0;
        let tokens: Vec<u32> = (0..5)
            .map(|k| ((batch.0 * 3 + k * 2) % VOCAB as u64) as u32)
            .collect();
        Ok(MicroBatchPlan {
            sequences: vec![SequenceWork {
                positions: (0..4).collect(),
                targets: tokens[1..].to_vec(),
                tokens,
                hidden_at: Vec::new(),
            }],
            row_scale: 0.25,
            counts: BatchCounts::default(),
        })
    }

    fn hidden_grad(
        &mut self,
        _seq: usize,
        _positions: &[u32],
        _hidden: &[f32],
        _width: usize,
    ) -> Result<Vec<f32>, TrainError> {
        Err(TrainError::Refused("no outside loss".into()))
    }

    fn close(&mut self, row_loss_sums: &[f64]) -> Result<MicroLoss, TrainError> {
        for (i, g) in self.grads.iter_mut().enumerate() {
            let n = self.values[i].len();
            let mut grad = vec![0.05 + 0.01 * i as f32; n];
            if self.poison == Some((self.batch, i)) {
                grad[0] = 1e21;
            }
            *g = Some(grad);
        }
        Ok(MicroLoss {
            total: row_loss_sums.iter().sum::<f64>() * 0.25,
            channels: Vec::new(),
        })
    }

    fn host(&mut self) -> Option<&mut dyn HostParams> {
        Some(self)
    }

    fn host_ref(&self) -> Option<&dyn HostParams> {
        Some(self)
    }
}

fn config(steps: u64) -> TrainConfig {
    TrainConfig {
        schedule: LrSchedule::new(LR, steps, 1, LR / 10.0).expect("schedule"),
        cap: WallClockCap::new(3600.0).expect("cap"),
        grad_accum: 1,
        optimizer: OptimizerRecipe {
            lr: LR,
            beta2: BETA2,
            lower_layers: None,
        },
        // Large enough that the clip does not scale the planted 1e21 gradient back into range.
        max_grad_norm: 1e30,
        epoch: 0,
        epochs: 1,
        seed: 7,
        checkpoint: None,
        eta: None,
    }
}

/// Runs `n_batches` batches of a `steps`-step schedule. Returns the loop's result, the provider
/// and the objective (its host values).
fn run(
    poison: Option<(u64, usize)>,
    steps: u64,
    n_batches: u64,
) -> (Result<(), TrainError>, ToyProvider, HostObjective) {
    let mut provider = toy();
    let mut objective = HostObjective::new(poison);
    let r = train(
        &mut provider,
        &mut objective,
        (0..n_batches).map(|i| Ok::<_, TrainError>(Batch(i))),
        &config(steps),
        &MonotonicClock::new(),
        None,
        &mut Hooks::default(),
    )
    .map(|_| ());
    (r, provider, objective)
}

#[test]
fn a_host_entry_refusal_leaves_the_provider_and_every_host_entry_at_the_last_good_step() {
    // The reference is the same schedule stopped after one good step by running out of data.
    let (ok, ref_provider, ref_host) = run(None, 2, 1);
    ok.expect("one good step");
    assert_eq!(ref_provider.step_count(), 1);
    // The second step's host gradient is poisoned at the first, a middle and the last entry.
    for entry in 0..3 {
        let (r, provider, host) = run(Some((1, entry)), 2, 2);
        let e = r.expect_err("the second step must be refused");
        assert!(
            e.to_string().contains("non-finite"),
            "entry {entry}: unexpected refusal {e}"
        );
        assert_eq!(
            provider.step_count(),
            1,
            "entry {entry}: the provider took the refused step"
        );
        assert_eq!(
            bits(provider.values()),
            bits(ref_provider.values()),
            "entry {entry}: provider values moved on the refused step"
        );
        for (a, b) in provider.moments().iter().zip(ref_provider.moments()) {
            assert_eq!(a, b, "entry {entry}: provider moments moved");
        }
        assert_eq!(
            bits(&host.values),
            bits(&ref_host.values),
            "entry {entry}: a host entry moved on the refused step"
        );
    }
}

#[test]
fn a_host_entry_refusal_on_the_first_step_moves_nothing() {
    let (r, provider, host) = run(Some((0, 1)), 2, 2);
    r.expect_err("refused");
    assert_eq!(provider.step_count(), 0);
    assert_eq!(bits(provider.values()), bits(toy().values()));
    assert_eq!(bits(&host.values), bits(&HostObjective::new(None).values));
}
