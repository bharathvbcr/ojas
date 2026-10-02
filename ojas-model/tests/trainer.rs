//! `Trainer` on the CPU and on a resident test double: G8 (optimizer phase
//! equals `HybridOptimizer::step` bit for bit), G10 (one readback per step),
//! accumulation, the error contract of `docs/framework-design.md` §3 under
//! injected faults, and a short training curve.

mod common;

use common::{bits, snapshot, token_bin, Fault, OptCall, Probe, Resident, Snapshot, TempBin};
use ojas_core::{Backend, Budget, DataCursor, Numerics, OjasError, Tensor};
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
    // Training v1 is multi-head only: a grouped-query spec is refused
    // before its parameters are looked at.
    let (_tmp, bin) = token_bin(20_000, 256);
    let gqa = ModelSpec {
        n_kv_head: 2,
        ..spec
    };
    let gqa_params = init_params(&gqa, 5, &Budget::new(1 << 30)).unwrap();
    assert!(matches!(
        Trainer::new(exact(1 << 30), gqa, &gqa_params, bin, config(1)),
        Err(OjasError::Unsupported { .. })
    ));

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
