//! Adversarial tests for card ft-fe7dcee6 (`Gated` generated from the one
//! method list). Leads: per-op admission pinned independently of
//! `gate.rs`; the commit contract under real cancels mid-step; wrapper
//! stacks.
//!
//! The expectation tables below are written from the pre-change
//! `gate.rs` at 040e1ed (every `enter(op, true)` polled, the four
//! `enter(op, false)` did not), not from `NEVER_POLL`, so a drift in the
//! new constant is caught rather than echoed.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ojas_core::{
    forward_backend_keep, forward_backend_methods, Autocast, AutocastMode, Backend, Budget,
    Numerics, OjasError, Tensor,
};
use ojas_cpu::{CosineSchedule, CpuBackend, LrSchedule};
use ojas_data::TokenBin;
use ojas_model::{init_params, ModelSpec, MomentsRef, TrainConfig, TrainState, Trainer};

use crate::gate::{CancelSlot, Gated};

#[path = "../../ojas-model/tests/reach/mod.rs"]
mod reach;
use reach::{call, op_names, Reach};

/// The ops 040e1ed's `Gated` never polled (`enter(op, false)`).
const OLD_NEVER_POLLED: [&str; 4] = ["download", "adamw_step", "muon_ns5_step", "sync"];

const CANCEL: &str = "cancelled: Explicit";

fn cancelling(polls: &Arc<AtomicUsize>) -> crate::gate::Check {
    let polls = Arc::clone(polls);
    Box::new(move || {
        polls.fetch_add(1, Ordering::SeqCst);
        Err(CANCEL.to_string())
    })
}

fn counting_ok(polls: &Arc<AtomicUsize>) -> crate::gate::Check {
    let polls = Arc::clone(polls);
    Box::new(move || {
        polls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })
}

// ------------------------------------------------------------ lead 1

/// The table covers exactly the trait's ops, so a new op fails here until
/// its admission is pinned.
#[test]
fn the_call_table_covers_every_op() {
    let ops = op_names();
    let reach = Reach::new();
    for name in &ops {
        assert!(call(&reach, name).is_some(), "no call-table arm for {name}");
    }
    let listed: Vec<&str> = ojas_core::BACKEND_METHODS.to_vec();
    for name in &ops {
        assert!(listed.contains(name));
    }
    assert_eq!(
        ops.len(),
        79,
        "op count changed: pin the new op's admission"
    );
}

/// For each op, in three gate states (no check armed, an armed check that
/// passes, an armed check that cancels): is the inner backend reached, how
/// many times is the check polled, and what is the refusal. The expectation
/// is the old hand-written forward's behaviour.
#[test]
fn per_op_admission_matches_the_pre_change_gate() {
    for name in op_names() {
        let never = OLD_NEVER_POLLED.contains(&name);

        // No check armed: every op reaches the inner backend once.
        let slot = Arc::new(CancelSlot::default());
        let g = Gated::new(Reach::new(), Arc::clone(&slot));
        let err = call(&g, name).unwrap().unwrap_err();
        assert!(
            err.to_string().contains(&format!("reached:{name}")),
            "{name} unarmed: {err}"
        );
        assert_eq!(g.inner().count(name), 1, "{name} unarmed");

        // An armed check that passes: one poll for a polling op, none for
        // the four commit ops; the inner backend is reached either way.
        let polls = Arc::new(AtomicUsize::new(0));
        let armed = slot.arm(counting_ok(&polls));
        let err = call(&g, name).unwrap().unwrap_err();
        assert!(
            err.to_string().contains(&format!("reached:{name}")),
            "{name} armed-ok: {err}"
        );
        assert_eq!(g.inner().count(name), 2, "{name} armed-ok");
        assert_eq!(
            polls.load(Ordering::SeqCst),
            usize::from(!never),
            "{name}: polls under a passing check"
        );
        assert_eq!(armed.tripped(), None, "{name}");
        drop(armed);

        // An armed check that cancels.
        let polls = Arc::new(AtomicUsize::new(0));
        let armed = slot.arm(cancelling(&polls));
        let err = call(&g, name).unwrap().unwrap_err();
        let text = err.to_string();
        if never {
            assert!(
                text.contains(&format!("reached:{name}")),
                "{name} must still run: {text}"
            );
            assert_eq!(g.inner().count(name), 3, "{name} cancelled");
            assert_eq!(polls.load(Ordering::SeqCst), 0, "{name} must not poll");
            assert_eq!(armed.tripped(), None, "{name} must not trip the slot");
        } else {
            assert!(
                matches!(err, OjasError::Backend { .. })
                    && text.contains(&format!("{name}: refused after the call was cancelled")),
                "{name} must be refused: {text}"
            );
            assert_eq!(
                g.inner().count(name),
                2,
                "{name} reached the inner backend after cancel"
            );
            assert_eq!(polls.load(Ordering::SeqCst), 1, "{name}");
            assert_eq!(armed.tripped().as_deref(), Some(CANCEL), "{name}");
        }
    }
}

/// Once a poll failed the call stays cancelled: later polling ops are refused
/// without running the check again, the commit ops still run, and the
/// accessors answer. Run over every op in a fixed order.
#[test]
fn a_tripped_gate_refuses_polling_ops_without_repolling_and_keeps_commit_ops_open() {
    let slot = Arc::new(CancelSlot::default());
    let g = Gated::new(Reach::new(), Arc::clone(&slot));
    let polls = Arc::new(AtomicUsize::new(0));
    let armed = slot.arm(cancelling(&polls));
    let mut refused = 0;
    for name in op_names() {
        let r = call(&g, name).unwrap().unwrap_err();
        let was_refused = r.to_string().contains("refused after");
        assert_eq!(
            was_refused,
            !OLD_NEVER_POLLED.contains(&name),
            "{name}: {r}"
        );
        refused += usize::from(was_refused);
    }
    assert_eq!(refused, 75);
    assert_eq!(
        polls.load(Ordering::SeqCst),
        1,
        "the check ran after it had failed"
    );
    // Accessors (`pure`) never poll and never refuse.
    let _ = (g.id(), g.numerics(), g.bf16_operands());
    assert!(g.budget().live_bytes().is_ok());
    assert_eq!(
        g.optimizer_scratch_bytes(ojas_core::OptimizerKind::AdamW, 2, 2)
            .unwrap(),
        None
    );
    assert_eq!(polls.load(Ordering::SeqCst), 1, "an accessor polled");
    assert_eq!(armed.tripped().as_deref(), Some(CANCEL));
    drop(armed);
    // A later call sees nothing of the earlier one.
    let _armed = slot.arm(counting_ok(&Arc::new(AtomicUsize::new(0))));
    let r = call(&g, "linear_forward").unwrap().unwrap_err();
    assert!(r.to_string().contains("reached:linear_forward"));
}

fn injected_marker() -> OjasError {
    OjasError::Backend {
        id: ojas_core::BackendId::Cpu,
        detail: "injected".into(),
    }
}

/// A test fault keyed by op name fires on exactly that op, once, before the
/// inner backend is touched, even for the four ops that never poll and
/// even with a passing check armed.
#[test]
fn an_injected_fault_fires_on_its_own_op_only() {
    for name in op_names() {
        let slot = Arc::new(CancelSlot::default());
        let g = Gated::new(Reach::new(), Arc::clone(&slot));
        slot.inject(
            Box::leak(name.to_string().into_boxed_str()),
            injected_marker,
        );
        let err = call(&g, name).unwrap().unwrap_err();
        assert!(err.to_string().contains("injected"), "{name}: {err}");
        assert_eq!(
            g.inner().total(),
            0,
            "{name}: the inner backend ran under a fault"
        );
        let err = call(&g, name).unwrap().unwrap_err();
        assert!(
            err.to_string().contains(&format!("reached:{name}")),
            "{name} second call: {err}"
        );
    }
}

/// The two renames the card documents: the saving/saved gate ops used to
/// answer to the key of the plain forward/backward. Pin the new behaviour
/// both ways so the rename cannot silently revert or leak.
#[test]
fn the_saving_ops_answer_to_their_own_fault_key_only() {
    let pairs = [
        (
            "per_head_sigmoid_gate_forward",
            "per_head_sigmoid_gate_forward_saving",
        ),
        (
            "per_head_sigmoid_gate_backward",
            "per_head_sigmoid_gate_backward_saved",
        ),
    ];
    for (plain, saving) in pairs {
        let slot = Arc::new(CancelSlot::default());
        let g = Gated::new(Reach::new(), Arc::clone(&slot));
        // The old key no longer fires on the saving op.
        slot.inject(plain, injected_marker);
        let err = call(&g, saving).unwrap().unwrap_err();
        assert!(
            err.to_string().contains(&format!("reached:{saving}")),
            "{saving}: {err}"
        );
        // It still fires on the plain op.
        let err = call(&g, plain).unwrap().unwrap_err();
        assert!(err.to_string().contains("injected"), "{plain}: {err}");
        // The new key fires on the saving op.
        slot.inject(saving, injected_marker);
        let err = call(&g, saving).unwrap().unwrap_err();
        assert!(err.to_string().contains("injected"), "{saving}: {err}");
    }
}

// ------------------------------------------------------------ lead 3

/// Every op reaches a `Reach` backend exactly once through every wrapper
/// order, and not through a trait default (which would reach nothing, or
/// something else). `Autocast` is hand-written by ruling: this asks it too.
#[test]
fn every_op_reaches_the_inner_backend_once_through_every_stack() {
    /// `Autocast` owns regions itself and never forwards `autocast_region`
    /// to its inner backend: by design, so it is the one expected
    /// exception for a stack that contains one.
    fn sweep<W: Backend>(label: &str, stack: &W, reach: &Reach, problems: &mut Vec<String>) {
        let has_autocast = label.contains("Autocast");
        for name in op_names() {
            let before = (reach.count(name), reach.total());
            let result = call(stack, name).unwrap();
            let after = (reach.count(name), reach.total());
            if has_autocast && name == "autocast_region" {
                continue;
            }
            match result {
                Ok(()) => {
                    problems.push(format!("{label}/{name}: returned Ok without the inner op"))
                }
                Err(err) => {
                    if !err.to_string().contains(&format!("reached:{name}")) {
                        problems.push(format!("{label}/{name}: wrong error: {err}"));
                    }
                    if after != (before.0 + 1, before.1 + 1) {
                        problems.push(format!(
                            "{label}/{name}: inner calls {before:?} -> {after:?} (default composed other ops?)"
                        ));
                    }
                }
            }
        }
    }
    let mut problems = Vec::new();
    let reach = Arc::new(Reach::new());
    let slot = || Arc::new(CancelSlot::default());
    sweep("&Arc<R>", &&reach, &reach, &mut problems);
    sweep(
        "Gated<&Arc<R>>",
        &Gated::new(&reach, slot()),
        &reach,
        &mut problems,
    );
    sweep(
        "Gated<Gated<&Arc<R>>>",
        &Gated::new(Gated::new(&reach, slot()), slot()),
        &reach,
        &mut problems,
    );
    sweep(
        "Gated<Autocast<&Arc<R>>>",
        &Gated::new(Autocast::new(&reach), slot()),
        &reach,
        &mut problems,
    );
    sweep(
        "Autocast<Gated<&Arc<R>>>",
        &Autocast::new(Gated::new(&reach, slot())),
        &reach,
        &mut problems,
    );
    sweep(
        "Autocast<Gated<Autocast<&Arc<R>>>>",
        &Autocast::new(Gated::new(Autocast::new(&reach), slot())),
        &reach,
        &mut problems,
    );
    let dynamic: Arc<dyn Backend> = reach.clone();
    sweep(
        "Gated<Arc<dyn Backend>>",
        &Gated::new(dynamic, slot()),
        &reach,
        &mut problems,
    );
    assert!(
        problems.is_empty(),
        "{} problems:\n{}",
        problems.len(),
        problems.join("\n")
    );
}

/// Under a cancelled gate, in either order, a polling op is refused before
/// the inner backend, and a commit op is not.
#[test]
fn a_cancel_under_either_order_refuses_exactly_the_polling_ops() {
    fn sweep<W: Backend>(
        label: &str,
        stack: &W,
        reach: &Reach,
        slot: &Arc<CancelSlot>,
        problems: &mut Vec<String>,
    ) {
        let _armed = slot.arm(cancelling(&Arc::new(AtomicUsize::new(0))));
        for name in op_names() {
            let before = reach.count(name);
            let result = call(stack, name).unwrap();
            let reached = reach.count(name) - before;
            let never = OLD_NEVER_POLLED.contains(&name);
            if reached != usize::from(never) || (result.is_ok() && name != "autocast_region") {
                problems.push(format!(
                    "{label}/{name}: reached={reached} never_polls={never} ok={}",
                    result.is_ok()
                ));
            }
        }
    }
    let mut problems = Vec::new();
    let reach = Arc::new(Reach::new());
    let slot = Arc::new(CancelSlot::default());
    sweep(
        "Gated<Autocast>",
        &Gated::new(Autocast::new(&reach), Arc::clone(&slot)),
        &reach,
        &slot,
        &mut problems,
    );
    sweep(
        "Autocast<Gated>",
        &Autocast::new(Gated::new(&reach, Arc::clone(&slot))),
        &reach,
        &slot,
        &mut problems,
    );
    assert!(
        problems.is_empty(),
        "{} problems:\n{}",
        problems.len(),
        problems.join("\n")
    );
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// A bf16 region's rounding applies through `Gated` in both orders and
/// gives the bits `Autocast<Cpu>` gives, and the test is not vacuous: the
/// unrounded CPU answer differs.
#[test]
fn autocast_rounding_survives_the_gate_in_both_orders() {
    let cpu = Arc::new(CpuBackend::new(Budget::new(1 << 24)).with_numerics(Numerics::Exact));
    let b = Budget::new(1 << 20);
    let x = Tensor::from_f32(
        &[1.0009, 2.0031, -3.0077, 0.1237, 1.3331, 7.0001],
        &[2, 3],
        &b,
    )
    .unwrap();
    let w = Tensor::from_f32(
        &[0.3331, 1.0017, -2.0049, 0.0713, 5.1111, 0.9999],
        &[2, 3],
        &b,
    )
    .unwrap();
    let plain = bits(&cpu.linear_forward(&x, &w).unwrap());

    let direct = Autocast::new(&cpu);
    let _r = direct.autocast_region(AutocastMode::Bf16).unwrap();
    let rounded = bits(&direct.linear_forward(&x, &w).unwrap());
    drop(_r);
    assert_ne!(
        rounded, plain,
        "operands were bf16-exact: the test proves nothing"
    );

    let slot = Arc::new(CancelSlot::default());
    let _armed = slot.arm(counting_ok(&Arc::new(AtomicUsize::new(0))));

    let outer_gate = Gated::new(Autocast::new(&cpu), Arc::clone(&slot));
    let region = outer_gate.autocast_region(AutocastMode::Bf16).unwrap();
    assert_eq!(
        bits(&outer_gate.linear_forward(&x, &w).unwrap()),
        rounded,
        "Gated<Autocast>"
    );
    assert!(outer_gate.inner().autocast_enabled().unwrap());
    drop(region);
    assert!(!outer_gate.inner().autocast_enabled().unwrap());
    assert_eq!(
        bits(&outer_gate.linear_forward(&x, &w).unwrap()),
        plain,
        "region closed"
    );

    let inner_gate = Autocast::new(Gated::new(&cpu, Arc::clone(&slot)));
    let region = inner_gate.autocast_region(AutocastMode::Bf16).unwrap();
    assert_eq!(
        bits(&inner_gate.linear_forward(&x, &w).unwrap()),
        rounded,
        "Autocast<Gated>"
    );
    drop(region);
}

/// A cancel refuses opening a region, leaves no region behind, and closing a
/// region that was already open is not gated.
#[test]
fn a_cancelled_gate_leaks_no_autocast_region() {
    let cpu = Arc::new(CpuBackend::new(Budget::new(1 << 20)));
    let slot = Arc::new(CancelSlot::default());
    let g = Gated::new(Autocast::new(&cpu), Arc::clone(&slot));
    let ok = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&ok);
    let armed = slot.arm(Box::new(move || {
        if flag.load(Ordering::SeqCst) {
            Err(CANCEL.to_string())
        } else {
            Ok(())
        }
    }));
    let region = g.autocast_region(AutocastMode::Bf16).unwrap();
    assert!(g.inner().autocast_enabled().unwrap());
    ok.store(true, Ordering::SeqCst);
    assert!(
        g.autocast_region(AutocastMode::Bf16).is_err(),
        "a nested region opened under a cancel"
    );
    drop(region);
    assert!(
        !g.inner().autocast_enabled().unwrap(),
        "the open region was not closed"
    );
    assert!(g.autocast_region(AutocastMode::Off).is_err());
    assert_eq!(armed.tripped().as_deref(), Some(CANCEL));
}

// ------------------------------------------------------------ lead 2

/// Backend layer that raises `cancel` the moment the trainer's first
/// optimizer/finish op reaches it (the cancel is armed between two ops of
/// the step), and counts commit ops.
struct Tripper<B> {
    inner: B,
    cancel: Arc<AtomicBool>,
    commit_ops: Arc<AtomicUsize>,
    trip_at: usize,
}

impl<B: Backend> Tripper<B> {
    fn admit(&self, op: &'static str) -> Result<(), OjasError> {
        if OLD_NEVER_POLLED.contains(&op) {
            let n = self.commit_ops.fetch_add(1, Ordering::SeqCst);
            if n >= self.trip_at {
                self.cancel.store(true, Ordering::SeqCst);
            }
        }
        Ok(())
    }
}

impl<B: Backend> Backend for Tripper<B> {
    ojas_core::with_backend_methods!(
        forward_backend_methods,
        { self, self.inner, admit, forward_backend_keep }
    );
}

struct TempBin(std::path::PathBuf);
impl Drop for TempBin {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

static BIN: AtomicUsize = AtomicUsize::new(0);

fn token_bin() -> (TempBin, TokenBin) {
    let mut bytes = Vec::new();
    let mut x = 1u32;
    for i in 0..20_000usize {
        x = if i % 97 == 0 {
            (i as u32 / 97 * 31 + 7) % 256
        } else {
            (5 * x + 3) % 256
        };
        bytes.extend_from_slice(&(x as u16).to_le_bytes());
    }
    let n = BIN.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("ojas-w11adv-{}-{n}.bin", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    let bin = TokenBin::open_headerless(&path).unwrap();
    (TempBin(path), bin)
}

fn config(autocast: AutocastMode) -> TrainConfig {
    let schedule = LrSchedule::Cosine(CosineSchedule::new(2, 40).unwrap());
    TrainConfig {
        autocast,
        ..TrainConfig::nanolab(2, 32, 2, 1337, schedule)
    }
}

fn trainer<B: Backend>(backend: B, autocast: AutocastMode) -> (TempBin, Trainer<B>) {
    let (tmp, bin) = token_bin();
    let spec = ModelSpec::tiny();
    let params = init_params(&spec, 5, &Budget::new(1 << 30)).unwrap();
    let t = Trainer::new(backend, spec, &params, bin, config(autocast)).unwrap();
    (tmp, t)
}

#[derive(Debug, PartialEq)]
struct Snap {
    params: Vec<Vec<u32>>,
    moments: Vec<Vec<Vec<u32>>>,
    step: u64,
    cursor: ojas_core::DataCursor,
}

fn snap<B: Backend>(t: &Trainer<B>) -> Snap {
    let mut params = Vec::new();
    let mut moments = Vec::new();
    for (info, v) in t.params() {
        params.push(bits(v));
        moments.push(match t.moments(&info.name).unwrap() {
            MomentsRef::Muon { momentum } => vec![bits(momentum)],
            MomentsRef::AdamW { m, v } => vec![bits(m), bits(v)],
            MomentsRef::Frozen => vec![],
        });
    }
    Snap {
        params,
        moments,
        step: t.step_count(),
        cursor: t.cursor(),
    }
}

fn live_bytes<B: Backend>(t: &Trainer<B>) -> u64 {
    t.backend().budget().live_bytes().unwrap()
}

fn exact() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 30)).with_numerics(Numerics::Exact)
}

/// The reference: one uncancelled step, its snapshot, how many polls it
/// makes, how long it takes, and how many commit ops it runs.
fn reference(autocast: AutocastMode) -> (Snap, usize, Duration, usize) {
    let slot = Arc::new(CancelSlot::default());
    let commits = Arc::new(AtomicUsize::new(0));
    let tripper = Tripper {
        inner: Gated::new(exact(), Arc::clone(&slot)),
        cancel: Arc::new(AtomicBool::new(false)),
        commit_ops: Arc::clone(&commits),
        trip_at: usize::MAX,
    };
    let (_tmp, mut t) = trainer(Autocast::new(tripper), autocast);
    commits.store(0, Ordering::SeqCst);
    let polls = Arc::new(AtomicUsize::new(0));
    let armed = slot.arm(counting_ok(&polls));
    let start = Instant::now();
    t.step().unwrap();
    let took = start.elapsed();
    drop(armed);
    (
        snap(&t),
        polls.load(Ordering::SeqCst),
        took,
        count_commit(&t),
    )
}

fn count_commit<W: Backend>(t: &Trainer<Autocast<Tripper<W>>>) -> usize {
    // `Autocast` has no accessor for its inner; the commit count is the
    // optimizer ops of one step plus sync and download, measured here from
    // the parameter table: every trainable slot is one optimizer op.
    t.params()
        .filter(|(info, _)| !matches!(t.moments(&info.name), Some(MomentsRef::Frozen)))
        .count()
        + 2
}

/// Cancel armed before the Nth commit op of a step (every optimizer op,
/// `sync`, `download`): the step must complete and commit, never be
/// refused or poisoned.
#[test]
fn a_cancel_armed_between_commit_ops_never_refuses_one() {
    for autocast in [AutocastMode::Off, AutocastMode::Bf16] {
        let (want, polls, _, commits) = reference(autocast);
        assert!(polls > 10 && commits > 3, "polls={polls} commits={commits}");
        // The first, a middle, the last optimizer op, sync, download.
        for trip_at in [0, 1, commits / 2, commits - 3, commits - 2, commits - 1] {
            let slot = Arc::new(CancelSlot::default());
            let cancel = Arc::new(AtomicBool::new(false));
            let commits = Arc::new(AtomicUsize::new(0));
            let tripper = Tripper {
                inner: Gated::new(exact(), Arc::clone(&slot)),
                cancel: Arc::clone(&cancel),
                commit_ops: Arc::clone(&commits),
                trip_at,
            };
            let (_tmp, mut t) = trainer(Autocast::new(tripper), autocast);
            commits.store(0, Ordering::SeqCst);
            let flag = Arc::clone(&cancel);
            let armed = slot.arm(Box::new(move || {
                if flag.load(Ordering::SeqCst) {
                    Err(CANCEL.to_string())
                } else {
                    Ok(())
                }
            }));
            let report = t.step();
            assert!(report.is_ok(), "{autocast:?} trip_at={trip_at}: {report:?}");
            assert_eq!(
                t.state(),
                TrainState::Ready,
                "{autocast:?} trip_at={trip_at}"
            );
            assert!(cancel.load(Ordering::SeqCst), "the cancel was never raised");
            assert_eq!(
                armed.tripped(),
                None,
                "a commit op polled and tripped (trip_at={trip_at})"
            );
            drop(armed);
            assert_eq!(
                snap(&t),
                want,
                "{autocast:?} trip_at={trip_at}: committed state differs"
            );
        }
    }
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// A cancel at the poll with index K (swept, seeded) makes the step fail with
/// the cancel and change nothing; the trainer is not poisoned, nothing
/// stays charged, and the next uncancelled step is bit-identical to a
/// trainer that never saw a cancel.
#[test]
fn a_cancel_at_any_poll_changes_nothing_and_the_retry_is_exact() {
    for autocast in [AutocastMode::Off, AutocastMode::Bf16] {
        let (want_after_one, polls, _, _) = reference(autocast);
        let slot = Arc::new(CancelSlot::default());
        let (_tmp, mut t) = trainer(
            Autocast::new(Gated::new(exact(), Arc::clone(&slot))),
            autocast,
        );
        let start = snap(&t);
        let live = live_bytes(&t);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut picks = vec![0, 1, polls / 2, polls - 1];
        picks.extend((0..6).map(|_| (xorshift(&mut seed) as usize) % polls));
        for k in picks {
            let count = Arc::new(AtomicUsize::new(0));
            let c = Arc::clone(&count);
            let armed = slot.arm(Box::new(move || {
                if c.fetch_add(1, Ordering::SeqCst) == k {
                    Err(CANCEL.to_string())
                } else {
                    Ok(())
                }
            }));
            let err = t
                .step()
                .expect_err(&format!("{autocast:?} k={k}: the cancel was swallowed"));
            assert_eq!(armed.tripped().as_deref(), Some(CANCEL), "k={k}: {err}");
            drop(armed);
            assert_eq!(
                t.state(),
                TrainState::Ready,
                "{autocast:?} k={k}: poisoned by a cancel"
            );
            assert_eq!(
                snap(&t),
                start,
                "{autocast:?} k={k}: state changed by a refused step"
            );
            assert_eq!(
                live_bytes(&t),
                live,
                "{autocast:?} k={k}: bytes left charged"
            );
        }
        t.step().unwrap();
        assert_eq!(
            snap(&t),
            want_after_one,
            "{autocast:?}: retry differs from a clean step"
        );
    }
}

/// The same, with a real second thread raising the cancel at a seeded
/// delay (a fraction of one measured step), 10 iterations. Either the
/// step finished and committed everything, or it failed with the cancel
/// and nothing moved; never a poisoned trainer or a partial commit.
#[test]
fn a_cancel_raised_by_another_thread_is_all_or_nothing() {
    let (want_after_one, _, took, _) = reference(AutocastMode::Off);
    let mut seed = 0x1234_5678_9abc_def1u64;
    let (mut ok, mut cancelled) = (0, 0);
    for i in 0..10 {
        let slot = Arc::new(CancelSlot::default());
        let (_tmp, mut t) = trainer(
            Autocast::new(Gated::new(exact(), Arc::clone(&slot))),
            AutocastMode::Off,
        );
        let start = snap(&t);
        let live = live_bytes(&t);
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancel);
        let armed = slot.arm(Box::new(move || {
            if flag.load(Ordering::SeqCst) {
                Err(CANCEL.to_string())
            } else {
                Ok(())
            }
        }));
        let frac = (xorshift(&mut seed) % 1300) as f64 / 1000.0;
        let delay = took.mul_f64(frac);
        let raiser = {
            let cancel = Arc::clone(&cancel);
            std::thread::spawn(move || {
                std::thread::sleep(delay);
                cancel.store(true, Ordering::SeqCst);
            })
        };
        let result = t.step();
        raiser.join().unwrap();
        let tripped = armed.tripped();
        drop(armed);
        match result {
            Ok(_) => {
                ok += 1;
                assert_eq!(
                    tripped, None,
                    "iteration {i}: a successful step recorded a cancel"
                );
                assert_eq!(
                    snap(&t),
                    want_after_one,
                    "iteration {i}: partial or wrong commit"
                );
            }
            Err(err) => {
                cancelled += 1;
                assert_eq!(tripped.as_deref(), Some(CANCEL), "iteration {i}: {err}");
                assert_eq!(t.state(), TrainState::Ready, "iteration {i}: poisoned");
                assert_eq!(
                    snap(&t),
                    start,
                    "iteration {i}: a failed step changed state"
                );
                assert_eq!(live_bytes(&t), live, "iteration {i}: leak");
                t.step().unwrap();
                assert_eq!(snap(&t), want_after_one, "iteration {i}: retry differs");
            }
        }
    }
    eprintln!(
        "threaded cancel: {ok} steps finished, {cancelled} cancelled (delay up to 1.3 x {took:?})"
    );
}
