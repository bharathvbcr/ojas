//! Constant, linear and step decays beside Lappi's cosine: the multiplier pinned at step 0, at
//! the warmup end and at the last step, the refusals, and the JSON round trip.
//!
//! Every value is exact in f64: peak 1.0, floor 0.0, warmup 2, total 10 (span 8), so the
//! multiplier is the rate and the tests compare with `assert_eq!`.

use std::path::PathBuf;

use ojas_io::json::parse_json;
use ojas_train::lappi::mock::ToyProvider;
use ojas_train::lappi::pyjson::{CANONICAL, dumps};
use ojas_train::lappi::recipe::OptimizerRecipe;
use ojas_train::lappi::run_control::{ConsumedPrefix, MonotonicClock, WallClockCap};
use ojas_train::lappi::schedule::{Decay, LrSchedule, MAX_LOSS_POINTS, ScheduleError};
use ojas_train::lappi::trainer::{
    BatchCounts, CheckpointPolicy, ConsumedBatch, Hooks, HostParams, MicroBatchPlan, MicroLoss,
    Objective, ResumeState, SequenceWork, TrainConfig, TrainError, train,
};

const STEP_4_HALF: Decay = Decay::Step {
    every: 4,
    gamma: 0.5,
};

fn kinds() -> [Decay; 4] {
    [Decay::Cosine, Decay::Constant, Decay::Linear, STEP_4_HALF]
}

fn sched(decay: Decay) -> LrSchedule {
    LrSchedule::with_decay(1.0, 10, 2, 0.0, decay).unwrap()
}

#[test]
fn step_zero_is_one_over_warmup_and_the_last_warmup_step_is_the_peak() {
    for d in kinds() {
        let s = sched(d);
        assert_eq!(s.lr_at(0).unwrap(), 0.5, "{d:?}: step 0 is 1/warmup");
        assert_eq!(s.lr_at(1).unwrap(), 1.0, "{d:?}: step warmup-1 is the peak");
    }
    let s = LrSchedule::with_decay(1.0, 12, 4, 0.0, Decay::Linear).unwrap();
    assert_eq!(s.lr_at(0).unwrap(), 0.25);
    assert_eq!(s.lr_at(3).unwrap(), 1.0);
}

#[test]
fn step_zero_without_warmup_is_the_decay_at_zero_which_is_one() {
    for d in kinds() {
        let s = LrSchedule::with_decay(1.0, 10, 0, 0.0, d).unwrap();
        assert_eq!(s.lr_at(0).unwrap(), 1.0, "{d:?}");
    }
}

#[test]
fn the_warmup_end_multiplier_is_exactly_one_for_every_kind() {
    for d in kinds() {
        assert_eq!(sched(d).lr_at(2).unwrap(), 1.0, "{d:?}");
    }
}

#[test]
fn the_last_step_multiplier_per_kind() {
    assert_eq!(sched(Decay::Constant).lr_at(9).unwrap(), 1.0);
    assert_eq!(sched(Decay::Linear).lr_at(9).unwrap(), 0.125);
    assert_eq!(sched(STEP_4_HALF).lr_at(9).unwrap(), 0.5);
    let cosine = sched(Decay::Cosine).lr_at(9).unwrap();
    assert!(cosine > 0.0 && cosine < 0.05, "cosine's last step {cosine}");
}

#[test]
fn step_decay_halves_every_interval_from_the_warmup_end() {
    let s = sched(STEP_4_HALF);
    let want = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5];
    for (i, w) in want.iter().enumerate() {
        assert_eq!(s.lr_at(2 + i as u64).unwrap(), *w, "step {}", 2 + i);
    }
}

#[test]
fn the_floor_lifts_every_decay_the_same_way() {
    let s = LrSchedule::with_decay(1.0, 10, 2, 0.25, STEP_4_HALF).unwrap();
    assert_eq!(s.lr_at(9).unwrap(), 0.25 + 0.75 * 0.5);
    let s = LrSchedule::with_decay(1.0, 10, 2, 0.25, Decay::Linear).unwrap();
    assert_eq!(s.lr_at(9).unwrap(), 0.25 + 0.75 * 0.125);
}

#[test]
fn every_kind_refuses_the_step_at_total_and_beyond() {
    for d in kinds() {
        let s = sched(d);
        assert!(s.lr_at(9).is_ok());
        assert_eq!(
            s.lr_at(10),
            Err(ScheduleError::PastEnd {
                step: 10,
                total: 10
            }),
            "{d:?}"
        );
        assert!(s.lr_at(11).is_err());
    }
}

#[test]
fn every_kind_keeps_the_validation_of_new() {
    for d in kinds() {
        assert!(LrSchedule::with_decay(0.0, 10, 1, 0.0, d).is_err());
        assert!(LrSchedule::with_decay(f64::NAN, 10, 1, 0.0, d).is_err());
        assert!(LrSchedule::with_decay(1.0, 10, 1, -1.0, d).is_err());
        assert!(
            LrSchedule::with_decay(1.0, 10, 1, 2.0, d).is_err(),
            "decays upwards"
        );
        assert!(LrSchedule::with_decay(1.0, 0, 0, 0.0, d).is_err());
        assert!(LrSchedule::with_decay(1.0, 10, 10, 0.0, d).is_err());
        assert!(LrSchedule::with_decay(1.0, MAX_LOSS_POINTS + 1, 1, 0.0, d).is_err());
    }
}

#[test]
fn a_step_decay_refuses_a_zero_interval_and_a_gamma_outside_zero_to_one() {
    let step = |every, gamma| LrSchedule::with_decay(1.0, 10, 2, 0.0, Decay::Step { every, gamma });
    assert!(step(0, 0.5).is_err(), "every 0");
    assert!(step(4, 0.0).is_err(), "gamma 0");
    assert!(step(4, -0.5).is_err(), "negative gamma");
    assert!(step(4, 1.5).is_err(), "gamma above 1");
    assert!(step(4, f64::NAN).is_err(), "NaN gamma");
    assert!(step(4, f64::INFINITY).is_err(), "infinite gamma");
    assert!(step(4, 1.0).is_ok(), "gamma 1 is the constant schedule");
}

#[test]
fn new_is_cosine_and_the_cosine_json_is_the_four_key_form() {
    let a = LrSchedule::new(1.0, 10, 2, 0.0).unwrap();
    let b = LrSchedule::with_decay(1.0, 10, 2, 0.0, Decay::Cosine).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.decay(), Decay::Cosine);
    for step in 0..10 {
        assert_eq!(
            a.lr_at(step).unwrap().to_bits(),
            b.lr_at(step).unwrap().to_bits()
        );
    }
    let json = a.to_json().unwrap();
    let keys: Vec<&str> = json
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(keys, ["peak_lr", "total_steps", "warmup_steps", "min_lr"]);
    assert_eq!(
        dumps(&json, CANONICAL).unwrap(),
        r#"{"min_lr":0.0,"peak_lr":1.0,"total_steps":10,"warmup_steps":2}"#
    );
}

#[test]
fn every_kind_round_trips_through_json() {
    for d in kinds() {
        let s = sched(d);
        let text = dumps(&s.to_json().unwrap(), CANONICAL).unwrap();
        let back = parse_json(&text).unwrap();
        assert_eq!(Decay::from_json(&back).unwrap(), d, "{text}");
    }
    let step = dumps(&sched(STEP_4_HALF).to_json().unwrap(), CANONICAL).unwrap();
    assert!(
        step.contains(r#""decay":{"every":4,"gamma":0.5,"kind":"step"}"#),
        "{step}"
    );
}

#[test]
fn a_decay_that_is_unknown_or_ill_typed_is_refused() {
    let read = |text: &str| Decay::from_json(&parse_json(text).unwrap());
    assert_eq!(
        read(r#"{"peak_lr": 1.0}"#),
        Ok(Decay::Cosine),
        "absent is cosine"
    );
    assert!(
        read(r#"{"decay": {"kind": "cosine"}}"#).is_err(),
        "cosine is never written"
    );
    assert!(read(r#"{"decay": {"kind": "wsd"}}"#).is_err());
    assert!(read(r#"{"decay": {"kind": 3}}"#).is_err());
    assert!(read(r#"{"decay": {}}"#).is_err());
    assert!(read(r#"{"decay": "linear"}"#).is_err());
    assert!(
        read(r#"{"decay": {"kind": "step", "gamma": 0.5}}"#).is_err(),
        "no every"
    );
    assert!(
        read(r#"{"decay": {"kind": "step", "every": 4}}"#).is_err(),
        "no gamma"
    );
    assert!(read(r#"{"decay": {"kind": "step", "every": "4", "gamma": 0.5}}"#).is_err());
    assert!(read(r#"{"decay": {"kind": "step", "every": -1, "gamma": 0.5}}"#).is_err());
    assert!(read(r#"{"decay": {"kind": "step", "every": 4, "gamma": "x"}}"#).is_err());
    assert!(matches!(
        read(r#"{"decay": {"kind": "constant"}}"#),
        Ok(Decay::Constant)
    ));
}

#[test]
fn edge_schedules() {
    // total 1, warmup 0: one step, at the peak.
    for d in kinds() {
        let s = LrSchedule::with_decay(1.0, 1, 0, 0.0, d).unwrap();
        assert_eq!(s.lr_at(0).unwrap(), 1.0, "{d:?}");
        assert!(s.lr_at(1).is_err());
    }
    // span 1: the single decay step is at x = 0.
    for d in kinds() {
        let s = LrSchedule::with_decay(1.0, 10, 9, 0.0, d).unwrap();
        assert_eq!(s.lr_at(9).unwrap(), 1.0, "{d:?}");
    }
    // An interval longer than the span never fires.
    let s = LrSchedule::with_decay(
        1.0,
        10,
        2,
        0.0,
        Decay::Step {
            every: 100,
            gamma: 0.5,
        },
    )
    .unwrap();
    for step in 2..10 {
        assert_eq!(s.lr_at(step).unwrap(), 1.0);
    }
    // gamma 1 is the constant schedule, step for step.
    let g1 = sched(Decay::Step {
        every: 3,
        gamma: 1.0,
    });
    let c = sched(Decay::Constant);
    for step in 0..10 {
        assert_eq!(g1.lr_at(step).unwrap(), c.lr_at(step).unwrap());
    }
    // min_lr == peak: flat after the warmup, whatever the decay.
    for d in kinds() {
        let s = LrSchedule::with_decay(1.0, 10, 0, 1.0, d).unwrap();
        for step in 0..10 {
            assert_eq!(s.lr_at(step).unwrap(), 1.0, "{d:?} step {step}");
        }
    }
}

#[test]
fn the_longest_admitted_step_schedule_evaluates() {
    // MAX_LOSS_POINTS keeps k far below i32::MAX for every schedule the constructor admits, so
    // the checked i32 conversion in lr_at is unreachable through the public API; the largest k
    // it does admit evaluates without panicking.
    let s = LrSchedule::with_decay(
        1.0,
        MAX_LOSS_POINTS,
        0,
        0.0,
        Decay::Step {
            every: 1,
            gamma: 1.0,
        },
    )
    .unwrap();
    assert_eq!(s.lr_at(MAX_LOSS_POINTS - 1).unwrap(), 1.0);
}

// ---- the checkpoint round trip, through the real writer and ResumeState::read ----

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

struct NextToken;

impl Objective for NextToken {
    type Batch = Batch;

    fn run_kind(&self) -> &'static str {
        "ft"
    }

    fn plan(&mut self, batch: &Batch) -> Result<MicroBatchPlan, TrainError> {
        let tokens: Vec<u32> = (0..5).map(|k| ((batch.0 * 3 + k * 2) % 8) as u32).collect();
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
        Ok(MicroLoss {
            total: row_loss_sums.iter().sum::<f64>() * 0.25,
            channels: Vec::new(),
        })
    }

    fn host(&mut self) -> Option<&mut dyn HostParams> {
        None
    }

    fn host_ref(&self) -> Option<&dyn HostParams> {
        None
    }
}

#[test]
fn a_checkpoint_records_and_reads_back_each_decay() {
    let root: PathBuf = std::env::temp_dir().join(format!("ojas-schedules-{}", std::process::id()));
    if root.exists() {
        std::fs::remove_dir_all(&root).expect("clear scratch");
    }
    for (i, d) in kinds().into_iter().enumerate() {
        let schedule = LrSchedule::with_decay(1e-2, 4, 1, 1e-3, d).unwrap();
        let cfg = TrainConfig {
            schedule,
            cap: WallClockCap::new(3600.0).unwrap(),
            grad_accum: 1,
            optimizer: OptimizerRecipe {
                lr: 1e-2,
                beta2: 0.999,
                lower_layers: None,
            },
            max_grad_norm: 1.0,
            epoch: 0,
            epochs: 1,
            seed: 7,
            checkpoint: Some(CheckpointPolicy {
                dir: root.join(format!("run{i}")),
                every: 2,
            }),
            eta: None,
        };
        let mut provider = ToyProvider::new(8, 4, 2, 42).unwrap();
        let result = train(
            &mut provider,
            &mut NextToken,
            (0..2).map(|n| Ok::<_, TrainError>(Batch(n))),
            &cfg,
            &MonotonicClock::new(),
            None,
            &mut Hooks::default(),
        )
        .expect("train");
        let state = ResumeState::read(&result.final_checkpoint.expect("checkpoint")).unwrap();
        assert_eq!(state.schedule, schedule, "{d:?}");
        assert_eq!(state.schedule.decay(), d);
    }
    std::fs::remove_dir_all(&root).expect("remove scratch");
}
