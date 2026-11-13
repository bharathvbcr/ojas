//! `Trainer` on the CPU and on a resident test double: G8 (optimizer phase
//! equals `HybridOptimizer::step` bit for bit), G10 (one readback per step),
//! accumulation, the error contract of `docs/framework-design.md` §3 under
//! injected faults, and a short training curve.

mod common;

use common::{bits, snapshot, token_bin, Fault, OptCall, Probe, Resident, Snapshot, TempBin};
use ojas_core::{
    Autocast, AutocastMode, Backend, BackendCore, Budget, DataCursor, Numerics, OjasError, Tensor,
};
use ojas_cpu::{
    CosineSchedule, CpuBackend, HybridOptimizer, HybridParam, LrSchedule, OptimGroup, WsdSchedule,
};
use ojas_data::Batch;
use ojas_model::{
    init_params, param_table, ModelSpec, MomentsRef, NonFinitePolicy, TrainConfig, TrainState,
    Trainer, NANOLAB_ADAM_LR, NANOLAB_MATRIX_LR,
};

const SEQ: usize = 32;
const BATCH: usize = 2;

fn exact(cap: u64) -> CpuBackend {
    CpuBackend::new(Budget::new(cap)).with_numerics(Numerics::Exact)
}

fn config(accum: usize) -> TrainConfig {
    let schedule = LrSchedule::Cosine(CosineSchedule::new(2, 40).unwrap());
    TrainConfig::nanolab(BATCH, SEQ, accum, 1337, schedule)
}

fn host_params(seed: u64) -> Vec<Tensor> {
    init_params(&ModelSpec::tiny(), seed, &Budget::new(1 << 30)).unwrap()
}

fn trainer<B: Backend>(backend: B, cfg: TrainConfig) -> (TempBin, Trainer<B>) {
    let (tmp, bin) = token_bin(20_000, 256);
    let t = Trainer::new(backend, ModelSpec::tiny(), &host_params(5), bin, cfg).unwrap();
    (tmp, t)
}

fn batch(salt: u32) -> Batch {
    let x: Vec<u32> = (0..BATCH * SEQ)
        .map(|i| (i as u32 * 29 + salt * 13 + 1) % 256)
        .collect();
    let y: Vec<u32> = x.iter().map(|v| (v * 5 + 3) % 256).collect();
    Batch {
        x,
        y,
        batch: BATCH,
        seq_len: SEQ,
    }
}

// ---------------------------------------------------------------------- G8

#[test]
fn g8_optimizer_phase_equals_hybrid_optimizer_bitwise_over_five_steps() {
    let probe = Probe::new(exact(1 << 30)).recording();
    let (_tmp, mut t) = trainer(probe, config(2));
    let table = param_table(&ModelSpec::tiny()).unwrap();
    let init = host_params(5);
    let hybrid_params = table
        .iter()
        .zip(&init)
        .filter(|(info, _)| info.trains)
        .map(|(info, value)| {
            let lr = match info.group {
                OptimGroup::MuonMatrix => NANOLAB_MATRIX_LR,
                _ => NANOLAB_ADAM_LR,
            };
            HybridParam::new(info.group, lr, &info.shape, value.to_f32_vec().unwrap()).unwrap()
        })
        .collect();
    let mut hybrid = HybridOptimizer::new(hybrid_params).unwrap();
    let trained: Vec<_> = table.iter().filter(|i| i.trains).collect();
    let frozen = "blocks.0.mixer.vr_lambda";
    let frozen_before = bits(t.backend(), t.param(frozen).unwrap());
    let schedule = CosineSchedule::new(2, 40).unwrap();
    for s in 0..5u64 {
        let report = t.step().unwrap();
        assert_eq!(report.step, s + 1);
        // A committed step moves the cursor past its K * B windows.
        let want = DataCursor {
            shard: 0,
            token_index: (s + 1) * 2 * BATCH as u64,
        };
        assert_eq!(t.cursor(), want);
        // The multiplier is the pre-increment step's.
        assert_eq!(report.lr_multiplier, schedule.multiplier(s).unwrap());
        let calls = t.backend().take_calls();
        // One optimizer call per trainable parameter; the frozen one has none.
        assert_eq!(calls.len(), trained.len());
        for (hp, call) in hybrid.params.iter_mut().zip(calls) {
            hp.grad = match (hp.group, call) {
                (OptimGroup::MuonMatrix, OptCall::Muon { grad, .. }) => grad,
                (
                    OptimGroup::AdamEmbedding | OptimGroup::AdamVector,
                    OptCall::AdamW { grad, step, .. },
                ) => {
                    assert_eq!(step, s, "AdamW gets the pre-increment step");
                    grad
                }
                (group, call) => panic!("{group:?} got {call:?}"),
            };
        }
        hybrid.step(report.lr_multiplier).unwrap();
        for (info, hp) in trained.iter().zip(&hybrid.params) {
            let b = t.backend();
            let name = &info.name;
            assert_eq!(
                bits(b, t.param(name).unwrap()),
                common::f32_bits(&hp.param),
                "{name} step {s}"
            );
            match t.moments(name).unwrap() {
                MomentsRef::Muon { momentum } => {
                    assert_eq!(bits(b, momentum), common::f32_bits(&hp.moment1), "{name}")
                }
                MomentsRef::AdamW { m, v } => {
                    assert_eq!(bits(b, m), common::f32_bits(&hp.moment1), "{name}");
                    assert_eq!(bits(b, v), common::f32_bits(&hp.moment2), "{name}");
                }
                MomentsRef::Frozen => panic!("{name} is trainable"),
            }
        }
        assert_eq!(bits(t.backend(), t.param(frozen).unwrap()), frozen_before);
        assert!(matches!(t.moments(frozen), Some(MomentsRef::Frozen)));
    }
    assert_eq!(hybrid.step_index, 5);
    assert_eq!(t.step_count(), 5);
}

// ------------------------------------------------------------ accumulation

/// With clipping off (coefficient 1) and a power-of-two K, the K-step
/// gradient is the mean of the single-batch gradients bit for bit: scaling
/// by `1/2` before the sum is exact.
#[test]
fn two_micro_batches_give_the_mean_gradient_and_mean_loss() {
    let cfg = TrainConfig {
        grad_clip: 1e30,
        ..config(1)
    };
    let run = |batches: &[Batch]| {
        let (_tmp, mut t) = trainer(Probe::new(exact(1 << 30)).recording(), cfg);
        let report = t.step_tokens(batches).unwrap();
        let grads: Vec<Vec<f32>> = t
            .backend()
            .take_calls()
            .into_iter()
            .map(|c| match c {
                OptCall::Muon { grad, .. } | OptCall::AdamW { grad, .. } => grad,
            })
            .collect();
        (report, grads)
    };
    let (b1, b2) = (batch(1), batch(2));
    let (r1, g1) = run(std::slice::from_ref(&b1));
    let (r2, g2) = run(std::slice::from_ref(&b2));
    let (r, g) = run(&[b1, b2]);
    assert_eq!(r.tokens, 2 * (BATCH * SEQ) as u64);
    assert_eq!(r.loss.to_bits(), ((r1.loss + r2.loss) / 2.0).to_bits());
    for ((a, b), mean) in g1.iter().zip(&g2).zip(&g) {
        let want: Vec<u32> = a
            .iter()
            .zip(b)
            .map(|(x, y)| ((x + y) * 0.5).to_bits())
            .collect();
        assert_eq!(common::f32_bits(mean), want);
    }
}

// --------------------------------------------------------------------- G10

#[test]
fn g10_one_readback_per_step_on_a_resident_backend_and_cpu_bits() {
    let (_tmp_d, mut dev) = trainer(Resident::new(1 << 31), config(2));
    let (_tmp_c, mut cpu) = trainer(exact(1 << 30), config(2));
    for _ in 0..4 {
        let before = dev.backend().budget().device_readbacks();
        let report = dev.step().unwrap();
        let after = dev.backend().budget().device_readbacks();
        assert_eq!(after.0 - before.0, 1, "readbacks in one step");
        assert_eq!(after.1 - before.1, 4, "the one readback is the f32 loss");
        // The device path (seed broadcast on the device, resident params)
        // gives the CPU path's bits.
        let host = cpu.step().unwrap();
        assert_eq!(report.loss.to_bits(), host.loss.to_bits());
        assert_eq!(report.grad_norm.to_bits(), host.grad_norm.to_bits());
    }
    assert_eq!(snapshot(&dev), snapshot(&cpu));
}

// ------------------------------------------------------- failures, step 1-4

fn assert_untouched<B: Backend>(t: &Trainer<B>, before: &Snapshot, err: &OjasError) {
    assert_eq!(&snapshot(t), before, "state changed after {err:?}");
    assert_eq!(t.state(), TrainState::Ready, "poisoned by {err:?}");
}

#[test]
fn an_all_ignored_batch_is_non_finite_and_changes_nothing() {
    let cfg = TrainConfig {
        ignore_index: Some(7),
        ..config(1)
    };
    let (_tmp, mut t) = trainer(exact(1 << 30), cfg);
    let before = snapshot(&t);
    let mut bad = batch(1);
    bad.y.iter_mut().for_each(|v| *v = 7);
    let err = t.step_tokens(&[bad]).unwrap_err();
    assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
    assert_untouched(&t, &before, &err);
    t.step_tokens(&[batch(1)]).unwrap();
    assert_eq!(t.step_count(), 1);
}

#[test]
fn injected_faults_before_the_optimizer_change_nothing() {
    let cases: [(&str, usize, Fault); 6] = [
        ("linear_forward", 5, Fault::NanOutput),
        ("linear_cross_entropy_mean", 1, Fault::Backend),
        ("accumulate_grad", 3, Fault::NonFinite),
        ("embedding_backward", 1, Fault::Capacity),
        ("residual_add_forward", 2, Fault::Backend),
        ("clip_grad_norm", 0, Fault::NonFinite),
    ];
    for (op, nth, fault) in cases {
        let (_tmp, mut t) = trainer(Probe::new(exact(1 << 30)), config(2));
        t.step().unwrap();
        let before = snapshot(&t);
        t.backend().fail(op, nth, fault);
        let err = t.step().unwrap_err();
        assert_untouched(&t, &before, &err);
        t.backend().clear_fault();
        // The same batches run cleanly afterwards.
        t.step().unwrap();
        assert_eq!(t.step_count(), 2, "{op}");
    }
}

#[test]
fn a_real_capacity_refusal_changes_nothing() {
    let cap = 1u64 << 26;
    let (_tmp, mut t) = trainer(exact(cap), config(1));
    let budget = t.backend().budget().clone();
    let live = budget.live_bytes().unwrap();
    // Leave 64 KiB: the parameters and moments fit, a step does not.
    let ballast = budget.try_reserve(cap - live - (64 << 10)).unwrap();
    let before = snapshot(&t);
    let err = t.step().unwrap_err();
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");
    assert_untouched(&t, &before, &err);
    drop(ballast);
    t.step().unwrap();
}

#[test]
fn skip_batch_advances_the_cursor_and_abort_keeps_it() {
    for (policy, moves) in [
        (NonFinitePolicy::Abort, false),
        (NonFinitePolicy::SkipBatch, true),
    ] {
        let cfg = TrainConfig {
            on_nonfinite: policy,
            ..config(2)
        };
        let (_tmp, mut t) = trainer(Probe::new(exact(1 << 30)), cfg);
        let start = t.cursor();
        t.backend().fail("linear_forward", 2, Fault::NanOutput);
        let err = t.step().unwrap_err();
        assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
        assert_eq!(t.step_count(), 0);
        assert_eq!(t.cursor() != start, moves, "{policy:?}");
        if moves {
            assert_eq!(t.cursor().token_index, start.token_index + 2 * BATCH as u64);
        }
        // A fault of another kind never skips.
        t.backend().fail("accumulate_grad", 0, Fault::Backend);
        let here = t.cursor();
        t.step().unwrap_err();
        assert_eq!(t.cursor(), here);
    }
}

#[test]
fn a_schedule_refusal_comes_before_any_work() {
    let cfg = TrainConfig {
        schedule: LrSchedule::Wsd(WsdSchedule::new(1, 3, 0.2).unwrap()),
        ..config(1)
    };
    let (_tmp, mut t) = trainer(Probe::new(exact(1 << 30)), cfg);
    // Steps 0..=3 are inside the 3-step WSD schedule (step == total is its
    // last point); step 4 is past it.
    for _ in 0..4 {
        t.step().unwrap();
    }
    let before = snapshot(&t);
    let forwards = t.backend().count("embedding_forward");
    let err = t.step().unwrap_err();
    assert!(matches!(err, OjasError::OutOfRange { .. }), "{err:?}");
    assert_eq!(
        t.backend().count("embedding_forward"),
        forwards,
        "work ran first"
    );
    assert_untouched(&t, &before, &err);
}

// ------------------------------------------------------ failures, step 5-6

#[test]
fn an_optimizer_or_sync_fault_poisons_and_later_steps_refuse() {
    for (op, nth) in [
        ("muon_ns5_step", 1),
        ("adamw_step", 0),
        ("sync", 0),
        ("download", 0),
    ] {
        let (_tmp, mut t) = trainer(Probe::new(exact(1 << 30)), config(1));
        t.step().unwrap();
        let cursor = t.cursor();
        t.backend().fail(op, nth, Fault::Backend);
        t.step().unwrap_err();
        assert_eq!(t.state(), TrainState::Poisoned, "{op}");
        assert_eq!(t.step_count(), 1, "{op}");
        assert_eq!(t.cursor(), cursor, "{op}");
        t.backend().clear_fault();
        assert!(matches!(t.step(), Err(OjasError::Poisoned)), "{op}");
        assert!(
            matches!(t.step_tokens(&[batch(1)]), Err(OjasError::Poisoned)),
            "{op}"
        );
        assert_eq!(t.step_count(), 1);
    }
}

/// Memory taken between the gradients and the optimizer (another holder of
/// the budget) is refused before the first update: the trainer stays
/// Ready with every parameter and moment unchanged, Save still works, and
/// once the memory is back the next step runs. Before the check, AdamW on
/// `tok_emb` (no scratch) updated and then the first Muon matrix refused
/// its scratch, which poisoned the trainer.
#[test]
fn optimizer_scratch_taken_after_the_gradients_refuses_before_any_update() {
    for room in [0u64, 1, 4096] {
        let (_tmp, mut t) = trainer(Probe::new(exact(1 << 30)), config(1));
        t.step().unwrap();
        let before = snapshot(&t);
        t.backend().squeeze_after_clip(0, room);
        let err = t.step().unwrap_err();
        assert!(
            matches!(err, OjasError::CapacityExceeded { .. }),
            "room {room}: {err:?}"
        );
        assert_untouched(&t, &before, &err);
        // No optimizer call of the refused step ran: only step 1's.
        let adam = adam_calls_per_step(&t);
        let muon = t.params().filter(|(info, _)| info.trains).count() - adam;
        assert_eq!(t.backend().count("adamw_step"), adam, "room {room}");
        assert_eq!(t.backend().count("muon_ns5_step"), muon, "room {room}");
        let dir = std::env::temp_dir().join(format!("ojas-squeeze-{}-{room}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        t.save(&dir)
            .expect("a refused step leaves the trainer saveable");
        std::fs::remove_dir_all(&dir).unwrap();
        t.backend().release();
        t.step().unwrap();
        assert_eq!(t.step_count(), 2);
    }
}

/// AdamW calls of one completed step (the trainer's AdamW slots).
fn adam_calls_per_step<B: Backend>(t: &Trainer<B>) -> usize {
    t.params()
        .filter(|(info, _)| info.trains && info.group != OptimGroup::MuonMatrix)
        .count()
}

/// A capacity sweep from below the trainer's own state to above one
/// step's measured peak. At every cap, either `Trainer::new` refuses with
/// `CapacityExceeded` before allocating, or steps run; a step that cannot
/// fit is refused with `CapacityExceeded` and leaves the trainer Ready and
/// saveable. Capacity never poisons, a step never fails after an earlier
/// one of the same shape succeeded at the same cap, and every cap at or
/// above the measured peak runs every step.
#[test]
fn a_capacity_sweep_refuses_up_front_and_never_poisons() {
    let measure = {
        let (_tmp, mut t) = trainer(exact(1 << 30), config(2));
        let budget = t.backend().budget().clone();
        budget.reset_peak();
        for _ in 0..3 {
            t.step().unwrap();
        }
        budget.peak_bytes()
    };
    let state = {
        let (_tmp, t) = trainer(exact(1 << 30), config(2));
        t.backend().budget().live_bytes().unwrap()
    };
    let (lo, hi) = (state / 2, measure + measure / 8);
    let points = 40u64;
    let (mut refused_open, mut refused_step, mut ran) = (0, 0, 0);
    for i in 0..=points {
        let cap = lo + (hi - lo) * i / points;
        let (_tmp, bin) = token_bin(20_000, 256);
        let opened = Trainer::new(
            exact(cap),
            ModelSpec::tiny(),
            &host_params(5),
            bin,
            config(2),
        );
        let mut t = match opened {
            Ok(t) => t,
            Err(err) => {
                assert!(
                    matches!(err, OjasError::CapacityExceeded { .. }),
                    "cap {cap}: {err:?}"
                );
                assert!(cap < measure, "cap {cap} >= peak {measure} refused at open");
                refused_open += 1;
                continue;
            }
        };
        let mut done = 0;
        for _ in 0..3 {
            let before = snapshot(&t);
            match t.step() {
                Ok(_) => done += 1,
                Err(err) => {
                    assert!(
                        matches!(err, OjasError::CapacityExceeded { .. }),
                        "cap {cap}: {err:?}"
                    );
                    assert_untouched(&t, &before, &err);
                    assert_eq!(done, 0, "cap {cap}: failed after a step of this shape ran");
                    break;
                }
            }
        }
        if done == 0 {
            refused_step += 1;
        } else {
            assert_eq!(done, 3, "cap {cap}");
            ran += 1;
        }
        if cap >= measure {
            assert_eq!(done, 3, "cap {cap} >= measured peak {measure}");
        }
    }
    // The sweep crossed every regime.
    assert!(
        refused_open > 0 && refused_step > 0 && ran > 0,
        "{refused_open} {refused_step} {ran}"
    );
}

// -------------------------------------------------------------- refusals

#[test]
fn construction_and_step_tokens_refuse_bad_input() {
    let (_tmp, bin) = token_bin(20_000, 256);
    let spec = ModelSpec::tiny();
    let mut params = host_params(5);
    params.pop();
    assert!(Trainer::new(exact(1 << 30), spec, &params, bin, config(1)).is_err());
    let (_tmp, bin) = token_bin(20_000, 256);
    let mut params = host_params(5);
    params.swap(1, 2);
    assert!(Trainer::new(exact(1 << 30), spec, &params, bin, config(1)).is_err());
    let (_tmp, bin) = token_bin(20_000, 256);
    let zero = TrainConfig {
        accum: 0,
        ..config(1)
    };
    assert!(Trainer::new(exact(1 << 30), spec, &host_params(5), bin, zero).is_err());
    let (_tmp, bin) = token_bin(20, 256);
    assert!(Trainer::new(exact(1 << 30), spec, &host_params(5), bin, config(1)).is_err());
    // T = 64 fits the bin but not the tiny spec's max_seq of 32.
    let (_tmp, bin) = token_bin(20_000, 256);
    let long = TrainConfig {
        seq_len: 64,
        ..config(1)
    };
    assert!(matches!(
        Trainer::new(exact(1 << 30), spec, &host_params(5), bin, long),
        Err(OjasError::Shape { .. })
    ));
    // Grouped-query attention is a training spec: the trainer opens.
    let (_tmp, bin) = token_bin(20_000, 256);
    let gqa = ModelSpec {
        n_kv_head: 2,
        ..spec
    };
    let gqa_params = init_params(&gqa, 5, &Budget::new(1 << 30)).unwrap();
    assert!(Trainer::new(exact(1 << 30), gqa, &gqa_params, bin, config(1)).is_ok());

    let (_tmp, mut t) = trainer(exact(1 << 30), config(1));
    let before = snapshot(&t);
    assert!(t.step_tokens(&[]).is_err());
    let mut short = batch(1);
    short.seq_len = 16;
    short.x.truncate(BATCH * 16);
    short.y.truncate(BATCH * 16);
    assert!(t.step_tokens(&[short]).is_err());
    let mut ragged = batch(1);
    ragged.y.pop();
    assert!(t.step_tokens(&[ragged]).is_err());
    let mut oov = batch(1);
    oov.x[3] = 256;
    let err = t.step_tokens(&[oov]).unwrap_err();
    assert_untouched(&t, &before, &err);
}

/// `grad_clip` 0 would zero every gradient (torch's `clip_grad_norm_` and
/// `clip_scale` both give a coefficient of 0), so the run would never
/// learn. It is refused with the config, not read as "no clipping".
#[test]
fn a_zero_grad_clip_is_refused_at_validation() {
    for grad_clip in [0.0f32, -0.0] {
        let cfg = TrainConfig {
            grad_clip,
            ..config(1)
        };
        let err = cfg.validate().unwrap_err();
        assert!(
            matches!(&err, OjasError::OutOfRange { detail, .. } if detail.contains("zeroes")),
            "{err:?}"
        );
        let (_tmp, bin) = token_bin(20_000, 256);
        assert!(matches!(
            Trainer::new(exact(1 << 30), ModelSpec::tiny(), &host_params(5), bin, cfg),
            Err(OjasError::OutOfRange { .. })
        ));
    }
    let tiny = TrainConfig {
        grad_clip: f32::MIN_POSITIVE,
        ..config(1)
    };
    tiny.validate().unwrap();
}

#[test]
fn the_callers_parameters_are_never_written() {
    let params = host_params(5);
    let want: Vec<Vec<u32>> = params
        .iter()
        .map(|t| common::f32_bits(&t.to_f32_vec().unwrap()))
        .collect();
    let (_tmp, bin) = token_bin(20_000, 256);
    let mut t = Trainer::new(exact(1 << 30), ModelSpec::tiny(), &params, bin, config(1)).unwrap();
    t.step().unwrap();
    t.step().unwrap();
    for (p, w) in params.iter().zip(&want) {
        assert_eq!(&common::f32_bits(&p.to_f32_vec().unwrap()), w);
    }
}

// ----------------------------------------------------------------- curve

/// The §10 CI shape on the tiny spec: B=2, K=2, T=32, 40 steps, CPU Exact.
/// The synthetic bin has a deterministic next-token rule, so the loss
/// must fall well below `ln 256`.
#[test]
fn forty_steps_cut_the_loss_by_more_than_one_nat() {
    let started = std::time::Instant::now();
    let (_tmp, mut t) = trainer(exact(1 << 30), config(2));
    let mut losses = Vec::new();
    for _ in 0..40 {
        let r = t.step().unwrap();
        assert!(r.loss.is_finite() && r.grad_norm.is_finite());
        losses.push(r.loss);
    }
    let first = losses[0];
    let last = losses[35..].iter().sum::<f32>() / 5.0;
    assert!((first - 256f32.ln()).abs() < 0.05, "step 0 loss {first}");
    assert!(last < 256f32.ln() - 1.0, "losses {losses:?}");
    eprintln!(
        "tiny CPU Exact: 40 steps in {:.2?}, loss {first:.4} -> {last:.4}",
        started.elapsed()
    );
}

// --------------------------------------------------------------- autocast

fn one_batch() -> [Batch; 1] {
    [batch(1)]
}

/// Off never enters a region, so a wrapped backend matches a raw one.
#[test]
fn autocast_off_matches_a_raw_backend_bit_for_bit() {
    let params = host_params(5);
    let cfg = config(1);
    let batches = one_batch();
    let (tmp_raw, bin_raw) = token_bin(20_000, 256);
    let (tmp_wrap, bin_wrap) = token_bin(20_000, 256);
    let mut raw = Trainer::new(exact(1 << 30), ModelSpec::tiny(), &params, bin_raw, cfg).unwrap();
    let mut wrapped = Trainer::new(
        Autocast::new(exact(1 << 30)),
        ModelSpec::tiny(),
        &params,
        bin_wrap,
        cfg,
    )
    .unwrap();
    assert_eq!(raw.run_id(), wrapped.run_id());
    assert!(!cfg.to_json().contains("autocast"));
    for _ in 0..2 {
        let a = raw.step_tokens(&batches).unwrap();
        let b = wrapped.step_tokens(&batches).unwrap();
        assert_eq!(a.loss.to_bits(), b.loss.to_bits());
        assert_eq!(a.grad_norm.to_bits(), b.grad_norm.to_bits());
    }
    assert_eq!(snapshot(&raw), snapshot(&wrapped));
    drop((tmp_raw, tmp_wrap));
}

/// A bf16 region changes at least one parameter and keeps a finite loss.
#[test]
fn autocast_bf16_is_finite_and_moves_a_parameter() {
    let params = host_params(5);
    let batches = one_batch();
    let off = config(1);
    let bf16 = TrainConfig {
        autocast: AutocastMode::Bf16,
        ..off
    };
    assert_ne!(off.to_json(), bf16.to_json());
    let (tmp_off, bin_off) = token_bin(20_000, 256);
    let (tmp_bf, bin_bf) = token_bin(20_000, 256);
    let mut plain = Trainer::new(
        Autocast::new(exact(1 << 30)),
        ModelSpec::tiny(),
        &params,
        bin_off,
        off,
    )
    .unwrap();
    let mut lowered = Trainer::new(
        Autocast::new(exact(1 << 30)),
        ModelSpec::tiny(),
        &params,
        bin_bf,
        bf16,
    )
    .unwrap();
    assert_ne!(plain.run_id(), lowered.run_id());
    let a = plain.step_tokens(&batches).unwrap();
    let b = lowered.step_tokens(&batches).unwrap();
    assert!(b.loss.is_finite() && b.grad_norm.is_finite(), "{b:?}");
    if snapshot(&plain).params == snapshot(&lowered).params {
        panic!(
            "bf16 left every parameter identical to f32; losses {} and {}",
            a.loss, b.loss
        );
    }
    drop((tmp_off, tmp_bf));
}

/// A raw backend has no region. The step refuses and the trainer stays Ready.
#[test]
fn autocast_bf16_on_a_raw_backend_refuses_and_stays_ready() {
    let cfg = TrainConfig {
        autocast: AutocastMode::Bf16,
        ..config(1)
    };
    let (_tmp, mut t) = trainer(exact(1 << 30), cfg);
    let before = snapshot(&t);
    let err = t.step_tokens(&one_batch()).unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::Unsupported {
                op: "autocast_region",
                ..
            }
        ),
        "{err:?}"
    );
    assert_untouched(&t, &before, &err);
}

/// Parameter bits read from the host payload. A poisoned autocast refuses
/// `download`, so the shared `snapshot` helper cannot observe this case.
fn host_words(tensor: &Tensor) -> Vec<u32> {
    tensor
        .to_f32_vec()
        .expect("cpu parameters stay on the host")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

#[allow(clippy::type_complexity)]
fn host_state<B: Backend>(t: &Trainer<B>) -> (Vec<Vec<u32>>, Vec<Vec<Vec<u32>>>, u64, DataCursor) {
    let mut params = Vec::new();
    let mut moments = Vec::new();
    for (info, value) in t.params() {
        params.push(host_words(value));
        moments.push(match t.moments(&info.name).unwrap() {
            MomentsRef::Muon { momentum } => vec![host_words(momentum)],
            MomentsRef::AdamW { m, v } => vec![host_words(m), host_words(v)],
            MomentsRef::Frozen => vec![],
        });
    }
    (params, moments, t.step_count(), t.cursor())
}

/// A poisoned wrapper fails the step and the next one. The trainer stays Ready.
#[test]
fn a_poisoned_autocast_leaves_the_trainer_ready() {
    let (_tmp, mut t) = trainer(Autocast::new(exact(1 << 30)), config(1));
    let before = host_state(&t);
    {
        let backend = t.backend();
        let outer = backend.autocast_region(AutocastMode::Bf16).unwrap();
        let inner = backend.autocast_region(AutocastMode::Bf16).unwrap();
        drop(outer);
        drop(inner);
    }
    let err = t.step_tokens(&one_batch()).unwrap_err();
    assert!(matches!(err, OjasError::Poisoned), "{err:?}");
    assert_eq!(host_state(&t), before, "state changed after {err:?}");
    assert_eq!(t.state(), TrainState::Ready, "poisoned by {err:?}");
    let again = t.step_tokens(&one_batch()).unwrap_err();
    assert!(matches!(again, OjasError::Poisoned), "{again:?}");
    assert_eq!(t.state(), TrainState::Ready);
    assert_eq!(host_state(&t), before);
}

// ------------------------------- ft-9c13f472: token-weighted accumulation

/// A micro-batch of `rows` rows whose targets follow the corpus rule, with
/// ids that depend on `salt` and the row.
fn rows_batch(rows: usize, salt: u32) -> Batch {
    let x: Vec<u32> = (0..rows * SEQ)
        .map(|i| (i as u32 * 29 + salt * 13 + (i / SEQ) as u32 * 7 + 1) % 256)
        .collect();
    let y: Vec<u32> = x.iter().map(|v| (v * 5 + 3) % 256).collect();
    Batch {
        x,
        y,
        batch: rows,
        seq_len: SEQ,
    }
}

/// `a` and `b` as one batch of all their rows.
fn joined(a: &Batch, b: &Batch) -> Batch {
    Batch {
        x: [a.x.as_slice(), b.x.as_slice()].concat(),
        y: [a.y.as_slice(), b.y.as_slice()].concat(),
        batch: a.batch + b.batch,
        seq_len: SEQ,
    }
}

/// One clip-free step on `batches`: the report, and the gradient of every
/// trainable slot as the optimizer received it.
fn recorded_step(cfg: TrainConfig, batches: &[Batch]) -> (ojas_model::StepReport, Vec<Vec<f32>>) {
    let cfg = TrainConfig {
        grad_clip: 1e30,
        ..cfg
    };
    let (_tmp, mut t) = trainer(Probe::new(exact(1 << 30)).recording(), cfg);
    let report = t.step_tokens(batches).unwrap();
    let grads = t
        .backend()
        .take_calls()
        .into_iter()
        .map(|c| match c {
            OptCall::Muon { grad, .. } | OptCall::AdamW { grad, .. } => grad,
        })
        .collect();
    (report, grads)
}

/// Equal up to the rounding of two different reduction orders: every
/// element within `1e-4` of its tensor's largest magnitude. A weighting
/// error moves the gradient by order 1 of it.
fn assert_grads_close(got: &[Vec<f32>], want: &[Vec<f32>]) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.len(), w.len());
        let scale = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for (a, b) in g.iter().zip(w) {
            assert!(
                (a - b).abs() <= 1e-4 * scale + 1e-12,
                "gradient {i}: {a} vs {b} (scale {scale})"
            );
        }
    }
}

fn assert_loss_close(got: f32, want: f32) {
    assert!(
        (got - want).abs() <= 1e-5 * want.abs() + 1e-6,
        "loss {got} vs {want}"
    );
}

/// ft-9c13f472 D2: micro-batches of 8 rows and 1 row weigh each token
/// equally, so the step equals one batch of all 9 rows. Under the
/// per-batch `1/K` weighting the lone row counted as much as the 8.
#[test]
fn uneven_micro_batches_weight_tokens_equally() {
    let (a, b) = (rows_batch(8, 1), rows_batch(1, 2));
    let whole = joined(&a, &b);
    let (r, g) = recorded_step(config(1), &[a, b]);
    let (r_whole, g_whole) = recorded_step(config(1), &[whole]);
    assert_eq!(r.tokens, 9 * SEQ as u64);
    assert_loss_close(r.loss, r_whole.loss);
    assert_grads_close(&g, &g_whole);
    // And not the per-batch mean the old seed formed.
    let (r8, _) = recorded_step(config(1), &[rows_batch(8, 1)]);
    let (r1, _) = recorded_step(config(1), &[rows_batch(1, 2)]);
    let per_batch = (r8.loss + r1.loss) / 2.0;
    assert!(
        (r.loss - per_batch).abs() > 1e-6 || (r8.loss - r1.loss).abs() < 1e-6,
        "the loss is the per-batch mean {per_batch}"
    );
}

/// The same with `ignore_index`: equal rows, unequal valid counts (32 and
/// 64 targets), weights 1/3 and 2/3.
#[test]
fn ignored_targets_weight_by_valid_share() {
    let cfg = TrainConfig {
        ignore_index: Some(7),
        ..config(1)
    };
    let mut a = rows_batch(2, 1);
    a.y[..SEQ].iter_mut().for_each(|v| *v = 7);
    let mut b = rows_batch(2, 2);
    // Targets of `b` that equal the ignored id by the corpus rule would
    // make the counts differ from 32 and 64; the weights are whatever the
    // counts are, and `whole` ignores the same targets.
    b.y.iter_mut().filter(|v| **v == 7).for_each(|v| *v = 8);
    a.y.iter_mut()
        .skip(SEQ)
        .filter(|v| **v == 7)
        .for_each(|v| *v = 8);
    let whole = joined(&a, &b);
    let (r, g) = recorded_step(cfg, &[a, b]);
    let (r_whole, g_whole) = recorded_step(cfg, &[whole]);
    assert_loss_close(r.loss, r_whole.loss);
    assert_grads_close(&g, &g_whole);
}

/// ft-9c13f472 D4: a micro-batch with no valid target has weight 0 and is
/// not run, in any position, so the step is the other batch's bit for bit.
/// Before, it failed the whole step with `NonFinite`.
#[test]
fn an_all_ignored_micro_batch_contributes_nothing() {
    let cfg = TrainConfig {
        ignore_index: Some(7),
        ..config(1)
    };
    let mut ignored = batch(2);
    ignored.y.iter_mut().for_each(|v| *v = 7);
    let mut valid = batch(1);
    valid.y.iter_mut().filter(|v| **v == 7).for_each(|v| *v = 8);
    let (_t0, mut alone) = trainer(Probe::new(exact(1 << 30)), cfg);
    let want = alone.step_tokens(std::slice::from_ref(&valid)).unwrap();
    for order in [
        vec![valid.clone(), ignored.clone()],
        vec![ignored.clone(), valid.clone()],
        vec![ignored.clone(), valid.clone(), ignored.clone()],
    ] {
        let (_t1, mut t) = trainer(Probe::new(exact(1 << 30)), cfg);
        let report = t.step_tokens(&order).unwrap();
        assert_eq!(report.loss.to_bits(), want.loss.to_bits());
        assert_eq!(report.grad_norm.to_bits(), want.grad_norm.to_bits());
        assert_eq!(report.step, 1);
        assert_eq!(snapshot(&t), snapshot(&alone));
        // The ignored batches ran no forward: one batch's worth of embeds.
        assert_eq!(t.backend().count("embedding_forward"), 1);
    }
}

/// A batch that is skipped for weight 0 still has its ids checked, as the
/// embedding would have: the step fails with `OutOfRange` and changes nothing.
#[test]
fn a_skipped_batch_still_refuses_an_out_of_vocab_id() {
    let cfg = TrainConfig {
        ignore_index: Some(7),
        ..config(1)
    };
    let (_tmp, mut t) = trainer(Probe::new(exact(1 << 30)), cfg);
    let before = snapshot(&t);
    let mut ignored = batch(2);
    ignored.y.iter_mut().for_each(|v| *v = 7);
    ignored.x[5] = 256;
    let err = t.step_tokens(&[batch(1), ignored]).unwrap_err();
    assert!(matches!(err, OjasError::OutOfRange { .. }), "{err:?}");
    assert_untouched(&t, &before, &err);
}

/// A step whose every target is ignored stays `NonFinite`, however many
/// micro-batches it has.
#[test]
fn a_step_of_only_ignored_micro_batches_is_non_finite() {
    let cfg = TrainConfig {
        ignore_index: Some(7),
        ..config(1)
    };
    let (_tmp, mut t) = trainer(Probe::new(exact(1 << 30)), cfg);
    let before = snapshot(&t);
    let mut ignored = batch(1);
    ignored.y.iter_mut().for_each(|v| *v = 7);
    let err = t.step_tokens(&[ignored.clone(), ignored]).unwrap_err();
    assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
    assert_untouched(&t, &before, &err);
    assert_eq!(t.backend().count("embedding_forward"), 0);
}

/// An all-ignored step with an out-of-vocabulary id is the `OutOfRange` the
/// embedding raised before the weights existed, not a `NonFinite` that
/// `SkipBatch` would skip (and count) while advancing the cursor.
#[test]
fn an_all_ignored_step_with_an_out_of_vocab_id_is_out_of_range() {
    let cfg = TrainConfig {
        ignore_index: Some(7),
        ..config(1)
    };
    let (_tmp, mut t) = trainer(Probe::new(exact(1 << 30)), cfg);
    let before = snapshot(&t);
    let mut ignored = batch(1);
    ignored.y.fill(7);
    ignored.x[3] = 256;
    let err = t.step_tokens(&[ignored]).unwrap_err();
    assert!(matches!(err, OjasError::OutOfRange { .. }), "{err:?}");
    assert_untouched(&t, &before, &err);
}

// ------------------------------------------ ft-9c13f472: clip and schedule

/// ft-9c13f472 D3b: `f32::MAX` clips nothing, also when the norm is below
/// 1 and `max_norm / norm` overflows `f32`; `inf` is refused by the config
/// with torch's meaning named.
#[test]
fn a_huge_grad_clip_clips_nothing_and_inf_is_refused_naming_torch() {
    // The overflow needs `norm + 1e-6 < 1`. Larger batches average the
    // gradient noise down, so take the smallest norm of a few batches (clip
    // off) and require the precondition: otherwise this test would also pass
    // on the code before the fix.
    let step_on = |grad_clip: f32, b: &Batch| {
        let cfg = TrainConfig {
            grad_clip,
            ..config(1)
        };
        let (_tmp, mut t) = trainer(exact(1 << 30), cfg);
        let report = t.step_tokens(std::slice::from_ref(b)).unwrap();
        (report, snapshot(&t))
    };
    let (norm, small) = [(2usize, 1u32), (8, 1), (32, 1), (32, 2)]
        .iter()
        .map(|&(rows, salt)| rows_batch(rows, salt))
        .map(|b| (step_on(1e30, &b).0.grad_norm, b))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap();
    assert!(
        norm + 1e-6 < 1.0,
        "smallest probed norm {norm} does not reach the f32::MAX / norm overflow"
    );
    // `f32::MAX / (norm + 1e-6)` overflows: coefficient 1, the same step as
    // with the clip off.
    let (report, state) = step_on(f32::MAX, &small);
    let (clip_off, state_off) = step_on(1e30, &small);
    assert_eq!(report.grad_norm.to_bits(), clip_off.grad_norm.to_bits());
    assert_eq!(state, state_off);
    let inf = TrainConfig {
        grad_clip: f32::INFINITY,
        ..config(1)
    };
    match inf.validate().unwrap_err() {
        OjasError::OutOfRange { detail, .. } => {
            assert!(
                detail.contains("torch") && detail.contains("f32::MAX"),
                "{detail}"
            )
        }
        other => panic!("{other:?}"),
    }
}

/// ft-9c13f472 D3a: warmup 0 trains, at multiplier 1.0 from step 0 as
/// nanolab's `_warmup_mult` does.
#[test]
fn warmup_zero_trains_at_the_peak_from_step_zero() {
    let schedule = LrSchedule::Cosine(CosineSchedule::new(0, 40).unwrap());
    let cfg = TrainConfig::nanolab(BATCH, SEQ, 1, 1337, schedule);
    let (_tmp, mut t) = trainer(exact(1 << 30), cfg);
    let first = t.step().unwrap();
    assert_eq!(first.lr_multiplier, 1.0);
    let second = t.step().unwrap();
    // nanolab's own `CosineSchedule` (warmup_steps=0, total 40, lr 1.0) at
    // step 1: 0.9986128001799076.
    assert_eq!(second.lr_multiplier.to_bits(), 0x3feff4a2d4c4784f);
}

// ------------------------------------------ ft-9c13f472 D9: skip counting

/// `MAX_CONSECUTIVE_SKIPS` (trainer.rs, not exported).
const MAX_SKIPS: usize = 8;

fn skip_config() -> TrainConfig {
    TrainConfig {
        on_nonfinite: NonFinitePolicy::SkipBatch,
        ..config(1)
    }
}

/// Fail the first `linear_forward` of the next step with a NaN.
fn poison_next_step(t: &Trainer<Probe<CpuBackend>>) {
    t.backend().fail("linear_forward", 0, Fault::NanOutput);
}

#[test]
fn skip_batch_stops_skipping_after_eight_in_a_row() {
    let (_tmp, mut t) = trainer(Probe::new(exact(1 << 30)), skip_config());
    for n in 1..=MAX_SKIPS {
        let before = t.cursor();
        poison_next_step(&t);
        let err = t.step().unwrap_err();
        assert!(
            matches!(err, OjasError::NonFinite { .. }),
            "skip {n}: {err:?}"
        );
        assert_eq!(
            t.cursor().token_index,
            before.token_index + BATCH as u64,
            "skip {n} advances the cursor"
        );
    }
    // The ninth in a row is the distinct error, with the cursor kept.
    let kept = t.cursor();
    poison_next_step(&t);
    let err = t.step().unwrap_err();
    match &err {
        OjasError::OutOfRange { op, detail } => {
            assert_eq!(*op, "Trainer::step");
            assert!(detail.contains("consecutive non-finite"), "{detail}");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(t.cursor(), kept);
    assert_eq!(t.step_count(), 0);
    assert_eq!(t.state(), TrainState::Ready);
    // A clean step runs (and clears the count: a later fault skips again).
    t.backend().clear_fault();
    t.step().unwrap();
    assert_eq!(t.step_count(), 1);
    poison_next_step(&t);
    let err = t.step().unwrap_err();
    assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
}

#[test]
fn a_committed_step_resets_the_skip_count() {
    let (_tmp, mut t) = trainer(Probe::new(exact(1 << 30)), skip_config());
    for round in 0..3 {
        for n in 0..MAX_SKIPS {
            poison_next_step(&t);
            let err = t.step().unwrap_err();
            assert!(
                matches!(err, OjasError::NonFinite { .. }),
                "round {round} skip {n}: {err:?}"
            );
        }
        t.backend().clear_fault();
        t.step().unwrap();
    }
    assert_eq!(t.step_count(), 3);
}

/// A commit through `step_tokens` ends a run of skips too.
#[test]
fn a_step_tokens_commit_clears_the_skip_count() {
    let (_tmp, mut t) = trainer(Probe::new(exact(1 << 30)), skip_config());
    for _ in 0..MAX_SKIPS {
        poison_next_step(&t);
        assert!(matches!(t.step().unwrap_err(), OjasError::NonFinite { .. }));
    }
    t.backend().clear_fault();
    t.step_tokens(&[batch(1)]).unwrap();
    poison_next_step(&t);
    let err = t.step().unwrap_err();
    assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
}

/// Abort never counts: the cursor stays and the error stays `NonFinite`
/// however often it repeats.
#[test]
fn abort_never_escalates() {
    let (_tmp, mut t) = trainer(Probe::new(exact(1 << 30)), config(1));
    let start = t.cursor();
    for _ in 0..MAX_SKIPS + 3 {
        poison_next_step(&t);
        let err = t.step().unwrap_err();
        assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
        assert_eq!(t.cursor(), start);
    }
}
