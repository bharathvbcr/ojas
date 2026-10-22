//! Adversarial tests for `Decay` / `LrSchedule::with_decay` / `Decay::from_json` / the
//! checkpoint's `"decay"` member (A1, ojas-7a lane L1). Public API only.
//!
//! Tests marked `#[ignore = "finding Fn: ..."]` assert the DESIRED behaviour and are expected to
//! FAIL against the current implementation (run with `--ignored`). Every other test is expected
//! to pass; a failure of one is a new finding. Expected outcomes cite the implementation line
//! (ojas-train/src/lappi/schedule.rs `S:`, ojas-train/src/lappi/trainer.rs `T:`,
//! ojas-io/src/json.rs `J:`).

use std::path::{Path, PathBuf};

use ojas_io::json::{JsonValue, parse_json};
use ojas_train::lappi::mock::ToyProvider;
use ojas_train::lappi::pyjson::{CANONICAL, dumps};
use ojas_train::lappi::recipe::OptimizerRecipe;
use ojas_train::lappi::run_control::{ConsumedPrefix, MonotonicClock, WallClockCap};
use ojas_train::lappi::schedule::{Decay, LrSchedule, MAX_LOSS_POINTS, ScheduleError};
use ojas_train::lappi::trainer::{
    BatchCounts, CheckpointPolicy, ConsumedBatch, Hooks, HostParams, MicroBatchPlan, MicroLoss,
    Objective, ResumeState, SequenceWork, TrainConfig, TrainError, train,
};

// ---------------------------------------------------------------- helpers

fn next_below_one() -> f64 {
    f64::from_bits(1.0f64.to_bits() - 1)
}

fn step(every: u64, gamma: f64) -> Decay {
    Decay::Step { every, gamma }
}

fn all_kinds() -> Vec<Decay> {
    vec![Decay::Cosine, Decay::Constant, Decay::Linear, step(2, 0.5)]
}

fn parse(text: &str) -> JsonValue {
    parse_json(text).unwrap_or_else(|e| panic!("test JSON {text:?} did not parse: {e}"))
}

fn read_schedule_decay(text: &str) -> Result<Decay, ScheduleError> {
    Decay::from_json(&parse(text))
}

// ------------------------------------------------- hostile decay JSON (pure)

/// Expected: Err. `kind` is matched with `==` on the exact string (S:`match kind`), so a wrong
/// case or "cosine" (never written, S:`Decay::to_json` returns None for Cosine) is the `other`
/// arm.
#[test]
fn from_json_refuses_wrong_case_spelled_out_cosine_and_unknown_kinds() {
    for kind in [
        "Linear",
        "LINEAR",
        "Constant",
        "Step",
        "STEP",
        "cosine",
        "Cosine",
        "",
        " linear",
        "linear ",
        "linear\u{0}",
        "step\n",
        "wsd",
        "none",
    ] {
        let text = format!(
            r#"{{"decay": {{"kind": {}, "every": 2, "gamma": 0.5}}}}"#,
            dumps(&JsonValue::from(kind), CANONICAL).unwrap()
        );
        assert!(
            read_schedule_decay(&text).is_err(),
            "kind {kind:?} must be refused"
        );
    }
}

/// Expected: Err for every non-object / mis-typed `decay` (S:`d.field_opt(key)` errors on a
/// non-object, mapped to "decay has no kind").
#[test]
fn from_json_refuses_a_decay_that_is_not_an_object() {
    for d in [
        "[]",
        "null",
        "0",
        "1",
        "1.5",
        "true",
        "false",
        r#""linear""#,
        r#""constant""#,
        r#"["kind","linear"]"#,
        r#"[{"kind":"linear"}]"#,
    ] {
        let text = format!(r#"{{"decay": {d}}}"#);
        assert!(
            read_schedule_decay(&text).is_err(),
            "decay {d} must be refused"
        );
    }
}

/// Expected: Err. A missing or non-string `kind` (S:`member("kind")` / `as_str`).
#[test]
fn from_json_refuses_a_missing_or_ill_typed_kind() {
    for d in [
        "{}",
        r#"{"every":2,"gamma":0.5}"#,
        r#"{"kind":null}"#,
        r#"{"kind":1}"#,
        r#"{"kind":["linear"]}"#,
        r#"{"kind":{"k":"linear"}}"#,
        r#"{"kind":true}"#,
    ] {
        let text = format!(r#"{{"decay": {d}}}"#);
        assert!(
            read_schedule_decay(&text).is_err(),
            "decay {d} must be refused"
        );
    }
}

/// Expected: Err for a schedule that is itself not an object (S:`schedule.field_opt` error arm).
#[test]
fn from_json_refuses_a_schedule_that_is_not_an_object() {
    for s in ["null", "[]", "3", r#""x""#, "true"] {
        assert!(Decay::from_json(&parse(s)).is_err(), "schedule {s}");
    }
}

/// Expected: Err. `every` must be a JSON unsigned integer literal (J:`as_u64`): negative, float
/// (even `2.0`, `1e2`), `-0`, string, null, bool, array, and an integer past u64 (the parser
/// itself refuses it, J:`integer ... overflows u64`).
#[test]
fn from_json_refuses_an_every_that_is_not_an_unsigned_integer() {
    for every in [
        "-1",
        "-0",
        "2.0",
        "1e2",
        "1.5",
        r#""2""#,
        r#""""#,
        "null",
        "true",
        "[2]",
        "{}",
        "18446744073709551616",
        "1e309",
    ] {
        let text = format!(r#"{{"decay": {{"kind":"step","every":{every},"gamma":0.5}}}}"#);
        let parsed = parse_json(&text);
        match parsed {
            Err(_) => {}
            Ok(v) => assert!(
                Decay::from_json(&v).is_err(),
                "every {every} must be refused"
            ),
        }
    }
}

/// Expected: Ok for the full u64 range of `every` that is an integer literal: 1, 2^53 + 1
/// (exact as u64 even though not as f64), u64::MAX (J:`JsonNumber::U64`).
#[test]
fn from_json_accepts_every_at_the_u64_extremes() {
    for every in [1u64, (1u64 << 53) + 1, u64::MAX] {
        let text = format!(r#"{{"decay": {{"kind":"step","every":{every},"gamma":0.5}}}}"#);
        assert_eq!(
            read_schedule_decay(&text),
            Ok(step(every, 0.5)),
            "every {every}"
        );
    }
}

/// Expected: Err for a string gamma (including "NaN", "Infinity", "0.5"), null, bool, array,
/// object (J:`as_f64` -> mismatch). `1e309` overflows f64 and the PARSER refuses it
/// (J:`overflows f64`), so the text never reaches from_json.
#[test]
fn from_json_refuses_a_gamma_that_is_not_a_number() {
    for gamma in [
        r#""0.5""#,
        r#""NaN""#,
        r#""nan""#,
        r#""Infinity""#,
        r#""-Infinity""#,
        r#""""#,
        "null",
        "true",
        "[0.5]",
        "{}",
    ] {
        let text = format!(r#"{{"decay": {{"kind":"step","every":2,"gamma":{gamma}}}}}"#);
        assert!(
            read_schedule_decay(&text).is_err(),
            "gamma {gamma} must be refused"
        );
    }
    for gamma in [
        "1e309",
        "-1e309",
        "NaN",
        "Infinity",
        "-Infinity",
        "+0.5",
        ".5",
        "0.5.5",
    ] {
        let text = format!(r#"{{"decay": {{"kind":"step","every":2,"gamma":{gamma}}}}}"#);
        assert!(
            parse_json(&text).is_err(),
            "the parser must refuse gamma {gamma}"
        );
    }
}

/// Expected: gamma `1` (integer literal) is read as 1.0 (J:`as_f64` converts exact integers);
/// the denormal `5e-324` is read exactly; `1e-400` underflows to 0.0 in the parser (J: doc) and
/// `-0.0` / `-0` parse to -0.0. from_json performs no range check (S:`Decay::from_json`), so
/// those are refused by `Decay::validate` (finding F1, fixed).
#[test]
fn from_json_number_forms_of_gamma() {
    let g = |t: &str| {
        read_schedule_decay(&format!(
            r#"{{"decay": {{"kind":"step","every":2,"gamma":{t}}}}}"#
        ))
    };
    assert_eq!(g("1"), Ok(step(2, 1.0)));
    assert_eq!(g("5e-324"), Ok(step(2, 5e-324)));
    assert_eq!(g("2.2250738585072014e-308"), Ok(step(2, f64::MIN_POSITIVE)));
    for t in ["1e-400", "-0.0", "-0"] {
        assert!(g(t).is_err(), "{t} is zero and must be refused");
    }
}

/// Expected: duplicate members are refused by the PARSER (J:247 `duplicate key`), so a hostile
/// file cannot smuggle a second "kind"/"decay"/"every".
#[test]
fn duplicate_keys_never_reach_from_json() {
    for text in [
        r#"{"decay":{"kind":"linear"},"decay":{"kind":"constant"}}"#,
        r#"{"decay":{"kind":"linear","kind":"constant"}}"#,
        r#"{"decay":{"kind":"step","every":1,"every":2,"gamma":0.5}}"#,
        r#"{"decay":{"kind":"step","every":1,"gamma":0.5,"gamma":0.25}}"#,
        r#"{"decay":{"kind":"step","every":1,"gamma":0.5},"decay":{"kind":"linear"}}"#,
    ] {
        assert!(parse_json(text).is_err(), "{text}");
    }
}

/// F2: a member the kind does not have is refused, not read as if absent
/// (S:`Decay::from_json` ends in `deny_unknown_keys`).
#[test]
fn extra_members_in_a_decay_object_are_refused() {
    for text in [
        r#"{"decay":{"kind":"constant","every":0,"gamma":"x"}}"#,
        r#"{"decay":{"kind":"constant","every":2}}"#,
        r#"{"decay":{"kind":"linear","gamma":0.5}}"#,
        r#"{"decay":{"kind":"linear","every":3,"gamma":0.5,"zzz":[]}}"#,
        r#"{"decay":{"kind":"step","every":3,"gamma":0.5,"zzz":null}}"#,
    ] {
        assert!(read_schedule_decay(text).is_err(), "{text}");
    }
}

/// F1: `Decay::from_json` applies `Decay::validate`, the rule `with_decay` applies.
#[test]
fn from_json_applies_the_same_step_validation_as_with_decay() {
    for (every, gamma) in [
        ("0", "0.5"),
        ("2", "0.0"),
        ("2", "-0.0"),
        ("2", "1.5"),
        ("2", "-0.5"),
        ("2", "1e-400"),
    ] {
        let text = format!(r#"{{"decay": {{"kind":"step","every":{every},"gamma":{gamma}}}}}"#);
        assert!(
            read_schedule_decay(&text).is_err(),
            "every {every} gamma {gamma} must not decode to an Ok decay no schedule can hold"
        );
    }
}

/// Expected: an absent member is cosine, `null` is NOT (S:`field_opt` -> Some(Null) ->
/// `d.field_opt("kind")` errors).
#[test]
fn absent_is_cosine_and_null_decay_is_refused() {
    assert_eq!(read_schedule_decay(r#"{"peak_lr":1.0}"#), Ok(Decay::Cosine));
    assert!(read_schedule_decay(r#"{"decay":null}"#).is_err());
}

// ------------------------------------------------------ constructor boundaries

/// Expected: Err for every kind when total/warmup/peak/min are hostile, never a panic
/// (S:with_decay checks, in order).
#[test]
fn constructor_never_panics_on_hostile_numbers() {
    let hostile = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0,
        0.0,
        -1.0,
        5e-324,
        f64::MIN_POSITIVE,
        1.0,
        f64::MAX,
    ];
    let steps = [0u64, 1, 2, MAX_LOSS_POINTS, MAX_LOSS_POINTS + 1, u64::MAX];
    let decays = [
        Decay::Cosine,
        Decay::Constant,
        Decay::Linear,
        step(0, 0.5),
        step(1, 0.5),
        step(u64::MAX, 0.5),
        step(1, f64::NAN),
        step(1, 5e-324),
        step(1, -0.0),
        step(1, 1.0),
        step(1, next_below_one()),
        step(1, 1.0000000000000002),
    ];
    for &p in &hostile {
        for &m in &hostile {
            for &t in &steps {
                for &w in &steps {
                    for d in decays {
                        let _ = LrSchedule::with_decay(p, t, w, m, d);
                    }
                }
            }
        }
    }
}

/// Expected: gamma in (0,1] accepted incl. denormal and 1 - 2^-53; refused: 0, -0.0, >1 by one
/// ulp, NaN, +-inf, negative, subnormal negative (S:`gamma.is_finite() && gamma > 0.0 &&
/// gamma <= 1.0`). every 0 refused, every u64::MAX accepted.
#[test]
fn step_validation_boundaries() {
    let ok = |e, g| LrSchedule::with_decay(1.0, 10, 2, 0.0, step(e, g)).is_ok();
    assert!(ok(1, 5e-324));
    assert!(ok(1, f64::MIN_POSITIVE));
    assert!(ok(1, next_below_one()));
    assert!(ok(1, 1.0));
    assert!(ok(u64::MAX, 0.5));
    assert!(!ok(0, 0.5));
    assert!(!ok(1, 0.0));
    assert!(!ok(1, -0.0));
    assert!(!ok(1, -5e-324));
    assert!(!ok(1, 1.0000000000000002));
    assert!(!ok(1, f64::NAN));
    assert!(!ok(1, f64::INFINITY));
    assert!(!ok(1, f64::NEG_INFINITY));
}

/// Expected for each boundary schedule: values are exact in f64 (peak 1, min 0 or 0.25, powers of
/// two).
#[test]
fn boundary_schedules() {
    // total 1: the single step is step 0, at peak, for every kind (S:lr_at, warmup 0, progress 0).
    for d in all_kinds() {
        let s = LrSchedule::with_decay(1.0, 1, 0, 0.0, d).unwrap();
        assert_eq!(s.lr_at(0).unwrap(), 1.0, "{d:?}");
        assert_eq!(
            s.lr_at(1),
            Err(ScheduleError::PastEnd { step: 1, total: 1 })
        );
        assert_eq!(
            s.lr_at(u64::MAX),
            Err(ScheduleError::PastEnd {
                step: u64::MAX,
                total: 1
            })
        );
    }
    // warmup = total - 1 (span 1): last warmup step is the peak ramp end, the one decay step is
    // x = 0 -> peak, for every kind.
    for d in all_kinds() {
        let s = LrSchedule::with_decay(1.0, 8, 7, 0.0, d).unwrap();
        assert_eq!(s.lr_at(6).unwrap(), 1.0, "{d:?} warmup end");
        assert_eq!(s.lr_at(7).unwrap(), 1.0, "{d:?} span 1");
        assert!(s.lr_at(8).is_err());
    }
    // every 1: gamma^k exactly, k = step - warmup.
    let s = LrSchedule::with_decay(1.0, 10, 2, 0.0, step(1, 0.5)).unwrap();
    for i in 0..8u64 {
        assert_eq!(s.lr_at(2 + i).unwrap(), 0.5f64.powi(i as i32), "k {i}");
    }
    // every > span and every = u64::MAX: never fires (S: integer division gives 0).
    for every in [9u64, 100, u64::MAX] {
        let s = LrSchedule::with_decay(1.0, 10, 2, 0.0, step(every, 0.5)).unwrap();
        for st in 2..10 {
            assert_eq!(s.lr_at(st).unwrap(), 1.0, "every {every} step {st}");
        }
    }
    // gamma tiny: the multiplier underflows to 0 and the rate lands exactly on min_lr, never below.
    for gamma in [5e-324, 1e-300] {
        let s = LrSchedule::with_decay(1.0, 10, 2, 0.25, step(1, gamma)).unwrap();
        assert_eq!(s.lr_at(2).unwrap(), 1.0);
        for st in 4..10 {
            assert_eq!(s.lr_at(st).unwrap(), 0.25, "gamma {gamma} step {st}");
        }
    }
    // min == peak: flat for every kind, including step with a tiny gamma.
    for d in [
        Decay::Cosine,
        Decay::Constant,
        Decay::Linear,
        step(1, 5e-324),
        step(3, 0.5),
    ] {
        let s = LrSchedule::with_decay(1.0, 10, 0, 1.0, d).unwrap();
        for st in 0..10 {
            assert_eq!(s.lr_at(st).unwrap(), 1.0, "{d:?} {st}");
        }
    }
    // Linear, min 0: the last step is peak/span, strictly above 0 (S:Linear = 1 - progress).
    let s = LrSchedule::with_decay(1.0, 8, 0, 0.0, Decay::Linear).unwrap();
    assert_eq!(s.lr_at(7).unwrap(), 0.125);
    assert!(s.lr_at(7).unwrap() > 0.0);
    // The longest admitted step schedule with the smallest interval and a denormal gamma.
    let s = LrSchedule::with_decay(1.0, MAX_LOSS_POINTS, 0, 0.0, step(1, 5e-324)).unwrap();
    assert_eq!(s.lr_at(MAX_LOSS_POINTS - 1).unwrap(), 0.0);
    assert_eq!(s.lr_at(0).unwrap(), 1.0);
}

// ------------------------------------------------------ cosine parity

/// Expected: `LrSchedule::new` and `with_decay(.., Cosine)` are bit-identical, and the four-key
/// JSON text is byte-identical to Python's (sorted keys, canonical separators) for awkward
/// values (S:Decay::Cosine.to_json -> None; `to_json` pushes "decay" only on Some).
#[test]
fn cosine_bits_and_four_key_json_are_unchanged() {
    for (peak, total, warm, min) in [
        (1e-5, 200u64, 10u64, 1e-6),
        (0.1, 7, 0, 0.0),
        (3e-4, 1000, 999, 3e-5),
        (1.0, 1, 0, 1.0),
    ] {
        let a = LrSchedule::new(peak, total, warm, min).unwrap();
        let b = LrSchedule::with_decay(peak, total, warm, min, Decay::Cosine).unwrap();
        assert_eq!(a, b);
        for st in 0..total {
            let want = if st < warm {
                peak * (st + 1) as f64 / warm as f64
            } else {
                let span = total - warm;
                let x = (st - warm) as f64 / span as f64;
                min + (peak - min) * (0.5 * (1.0 + (std::f64::consts::PI * x).cos()))
            };
            assert_eq!(a.lr_at(st).unwrap().to_bits(), want.to_bits(), "step {st}");
            assert_eq!(b.lr_at(st).unwrap().to_bits(), want.to_bits(), "step {st}");
        }
        let text = dumps(&a.to_json().unwrap(), CANONICAL).unwrap();
        assert!(!text.contains("decay"), "{text}");
        assert_eq!(parse(&text).as_object().unwrap().len(), 4);
    }
}

// ------------------------------------------------------ monotonicity properties

fn check_profile(s: &LrSchedule, label: &str) {
    let (peak, min) = (s.peak_lr(), s.min_lr());
    let (total, warm) = (s.total_steps(), s.warmup_steps());
    let mut prev: Option<f64> = None;
    for st in 0..total {
        let lr = s.lr_at(st).unwrap();
        assert!(lr.is_finite(), "{label} step {st}: {lr}");
        // The warmup ramps from peak/warmup, which may be below min_lr; the floor binds after it.
        assert!(
            st < warm || lr >= min,
            "{label} step {st}: {lr} < min {min}"
        );
        assert!(
            lr >= 0.0 && lr <= peak,
            "{label} step {st}: {lr} not in [0, peak {peak}]"
        );
        if let Some(p) = prev {
            if st < warm {
                assert!(
                    lr >= p,
                    "{label} warmup not non-decreasing at {st}: {p} -> {lr}"
                );
            } else if st > warm {
                assert!(lr <= p, "{label} not non-increasing at {st}: {p} -> {lr}");
            }
        }
        prev = Some(lr);
    }
    assert_eq!(
        s.lr_at(warm).unwrap(),
        peak,
        "{label}: the first post-warmup step is the peak"
    );
}

/// Expected: for peak 1 and min in {0, 1/4, 1} (all exact), every kind over the whole grid:
/// warmup non-decreasing, post-warmup non-increasing, lr in [min, peak], first post-warmup step
/// == peak. (S:lr_at; decay multiplier in [0,1].)
#[test]
fn every_kind_is_monotone_and_bounded_over_a_grid() {
    let gammas = [1.0, 0.5, 0.1, 1e-3, 1e-200, 5e-324, next_below_one()];
    let everys = [1u64, 2, 3, 5, 100, u64::MAX];
    let mut cases = 0u64;
    for total in 1..=12u64 {
        for warm in 0..total {
            for min in [0.0, 0.25, 1.0] {
                let mut decays = vec![Decay::Cosine, Decay::Constant, Decay::Linear];
                for &e in &everys {
                    for &g in &gammas {
                        decays.push(step(e, g));
                    }
                }
                for d in decays {
                    let s = LrSchedule::with_decay(1.0, total, warm, min, d).unwrap();
                    check_profile(&s, &format!("total {total} warm {warm} min {min} {d:?}"));
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 10_000, "grid ran {cases} cases");
}

/// Expected: step decay with random gammas in (0, 1) is non-increasing and in [0, 1] (powi is
/// not guaranteed monotone; a failure here is a finding about `f64::powi`, S:Decay::Step arm).
#[test]
fn step_decay_with_random_gammas_is_monotone() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut cases = 0;
    for _ in 0..3000 {
        let gamma = (next() >> 11) as f64 / (1u64 << 53) as f64;
        if gamma == 0.0 {
            continue;
        }
        let every = 1 + next() % 7;
        let total = 2 + next() % 120;
        let warm = next() % total.min(5);
        let s = LrSchedule::with_decay(1.0, total, warm, 0.0, step(every, gamma)).unwrap();
        check_profile(
            &s,
            &format!("gamma {gamma:e} every {every} total {total} warm {warm}"),
        );
        cases += 1;
    }
    assert!(cases > 2900);
}

/// Characterisation (inherited from Python's formula, so also true of cosine): for peak 0.01 and
/// min 0.001 the rate at the peak is 0.010000000000000002, above the peak, for EVERY kind
/// (S:lr_at final line `min + (peak - min) * decay` with decay == 1.0). Verified by hand with
/// Python floats, not by running this test.
#[test]
fn rate_at_the_peak_can_round_one_ulp_above_peak() {
    for d in [Decay::Cosine, Decay::Constant, Decay::Linear, step(1, 0.5)] {
        let s = LrSchedule::with_decay(0.01, 4, 0, 0.001, d).unwrap();
        assert_eq!(s.lr_at(0).unwrap(), 0.010000000000000002, "{d:?}");
        assert!(s.lr_at(0).unwrap() > s.peak_lr(), "{d:?}");
    }
    // The warmup's last step: peak * w / w is two roundings, so peak 0.9, warmup 13 exceeds it.
    let s = LrSchedule::with_decay(0.9, 20, 13, 0.0, Decay::Constant).unwrap();
    assert_eq!(s.lr_at(12).unwrap(), 0.9000000000000001);
}

#[test]
#[ignore = "out of scope: inherited from Lappi LRSchedule parity, filed on ft-b5e64557"]
fn desired_rate_never_exceeds_peak_for_decimal_peaks() {
    let mut bad = Vec::new();
    for peak in [0.01, 0.9, 5e-5, 2e-4, 0.3, 0.7] {
        for min in [0.0, 0.001, 1e-4] {
            if min > peak {
                continue;
            }
            for warm in [0u64, 1, 13] {
                for d in all_kinds() {
                    let s = LrSchedule::with_decay(peak, 30, warm, min, d).unwrap();
                    for st in 0..30 {
                        let lr = s.lr_at(st).unwrap();
                        if lr > peak || lr < min {
                            bad.push((peak, min, warm, format!("{d:?}"), st, lr));
                        }
                    }
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} violations, first {:?}",
        bad.len(),
        bad.first()
    );
}

#[test]
#[ignore = "out of scope: inherited from Lappi LRSchedule parity, filed on ft-b5e64557"]
fn desired_rate_is_finite_for_a_huge_finite_peak() {
    let s = LrSchedule::with_decay(f64::MAX, 4, 2, 0.0, Decay::Constant).unwrap();
    for st in 0..4 {
        assert!(s.lr_at(st).unwrap().is_finite(), "step {st}");
    }
}

// ------------------------------------------------------ JSON round trips

/// Expected: to_json -> dumps -> parse -> from_json is the identity for every kind, including
/// gammas that are not exactly representable in decimal (0.1, 1/3, 2/3, 1-2^-53, MIN_POSITIVE,
/// 5e-324) and every = u64::MAX (S:to_json writes `float(gamma)`; float_repr is shortest
/// round-trip; J parses with std's correctly rounded parser).
#[test]
fn every_decay_round_trips_through_text_bit_for_bit() {
    let mut decays = vec![Decay::Cosine, Decay::Constant, Decay::Linear];
    for gamma in [
        0.1,
        1.0 / 3.0,
        2.0 / 3.0,
        0.7,
        0.3,
        1e-3,
        1e-7,
        next_below_one(),
        f64::MIN_POSITIVE,
        5e-324,
        1.0,
        0.999999999999,
        f64::from_bits(0x4308_3d7d_4ba9_fb7a) / 1e15,
    ] {
        for every in [1u64, 3, (1 << 53) + 1, u64::MAX] {
            decays.push(step(every, gamma));
        }
    }
    for d in decays {
        let s = LrSchedule::with_decay(0.01, 20, 3, 0.001, d).unwrap();
        let text = dumps(&s.to_json().unwrap(), CANONICAL).unwrap();
        let back = Decay::from_json(&parse(&text)).unwrap();
        match (d, back) {
            (
                Decay::Step {
                    every: e1,
                    gamma: g1,
                },
                Decay::Step {
                    every: e2,
                    gamma: g2,
                },
            ) => {
                assert_eq!(e1, e2, "{text}");
                assert_eq!(g1.to_bits(), g2.to_bits(), "{text}");
            }
            _ => assert_eq!(d, back, "{text}"),
        }
        let again = dumps(&s.to_json().unwrap(), CANONICAL).unwrap();
        assert_eq!(text, again, "to_json is deterministic");
    }
}

/// Expected: non-cosine kinds add exactly one member, "decay", after the four Python keys and
/// the text sorts it (CANONICAL sorts keys); constant/linear carry only "kind" (S:Decay::to_json).
#[test]
fn json_shape_of_each_kind() {
    let text = |d| {
        dumps(
            &LrSchedule::with_decay(1.0, 10, 2, 0.0, d)
                .unwrap()
                .to_json()
                .unwrap(),
            CANONICAL,
        )
        .unwrap()
    };
    assert_eq!(
        text(Decay::Constant),
        r#"{"decay":{"kind":"constant"},"min_lr":0.0,"peak_lr":1.0,"total_steps":10,"warmup_steps":2}"#
    );
    assert_eq!(
        text(Decay::Linear),
        r#"{"decay":{"kind":"linear"},"min_lr":0.0,"peak_lr":1.0,"total_steps":10,"warmup_steps":2}"#
    );
    assert_eq!(
        text(step(u64::MAX, 0.1)),
        r#"{"decay":{"every":18446744073709551615,"gamma":0.1,"kind":"step"},"min_lr":0.0,"peak_lr":1.0,"total_steps":10,"warmup_steps":2}"#
    );
}

// ------------------------------------------------------ checkpoint path

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

fn scratch(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("ojas-a1-{tag}-{}", std::process::id()));
    if root.exists() {
        std::fs::remove_dir_all(&root).expect("clear scratch");
    }
    root
}

fn cfg(schedule: LrSchedule, dir: PathBuf) -> TrainConfig {
    TrainConfig {
        schedule,
        cap: WallClockCap::new(3600.0).unwrap(),
        grad_accum: 1,
        optimizer: OptimizerRecipe {
            lr: schedule.peak_lr(),
            beta2: 0.999,
            lower_layers: None,
        },
        max_grad_norm: 1.0,
        epoch: 0,
        epochs: 1,
        seed: 7,
        checkpoint: Some(CheckpointPolicy { dir, every: 2 }),
        eta: None,
    }
}

/// Run 2 of the schedule's 4 steps and return the checkpoint directory.
fn checkpoint_of(schedule: LrSchedule, dir: PathBuf) -> PathBuf {
    let mut provider = ToyProvider::new(8, 4, 2, 42).unwrap();
    let result = train(
        &mut provider,
        &mut NextToken,
        (0..2).map(|n| Ok::<_, TrainError>(Batch(n))),
        &cfg(schedule, dir),
        &MonotonicClock::new(),
        None,
        &mut Hooks::default(),
    )
    .unwrap_or_else(|e| panic!("train: {e}"));
    result.final_checkpoint.expect("a checkpoint")
}

fn base(decay: Decay) -> LrSchedule {
    LrSchedule::with_decay(1e-2, 4, 1, 1e-3, decay).unwrap()
}

/// Expected: ResumeState::read returns a schedule == the written one for kinds and inexact
/// gammas (T:read -> with_decay(.., read_decay)); `Decay` equality is bitwise on gamma through
/// the derived f64 PartialEq, and 0.1 / 1/3 round trip through the shortest repr.
#[test]
fn resume_state_round_trips_each_kind_and_inexact_gamma() {
    let root = scratch("roundtrip");
    let mut decays = vec![
        Decay::Cosine,
        Decay::Constant,
        Decay::Linear,
        step(1, 0.1),
        step(2, 1.0 / 3.0),
        step(u64::MAX, next_below_one()),
        step(1, 5e-324),
        step(3, f64::MIN_POSITIVE),
        step(1, 1.0),
    ];
    decays.dedup();
    for (i, d) in decays.into_iter().enumerate() {
        let s = base(d);
        let dir = checkpoint_of(s, root.join(format!("run{i}")));
        let state = ResumeState::read(&dir).unwrap_or_else(|e| panic!("{d:?}: {e}"));
        assert_eq!(state.schedule, s, "{d:?}");
        match (state.schedule.decay(), d) {
            (Decay::Step { gamma: a, .. }, Decay::Step { gamma: b, .. }) => {
                assert_eq!(a.to_bits(), b.to_bits(), "{d:?}");
            }
            (a, b) => assert_eq!(a, b),
        }
    }
    std::fs::remove_dir_all(&root).ok();
}

/// Replace the `"schedule":{...}` object (flat, as a cosine checkpoint writes it) in
/// `trainer.json` with `schedule_text`. trainer.json carries no digest of itself (T:read checks
/// only provider_files and host_sha256), so the edit is read back as written.
fn tamper(dir: &Path, schedule_text: &str) -> String {
    let path = dir.join("trainer.json");
    let original = std::fs::read_to_string(&path).unwrap();
    let key = "\"schedule\":{";
    let start = original.find(key).expect("schedule member") + "\"schedule\":".len();
    let end = start + original[start..].find('}').expect("end of schedule") + 1;
    let patched = format!(
        "{}{}{}",
        &original[..start],
        schedule_text,
        &original[end..]
    );
    std::fs::write(&path, patched).unwrap();
    original
}

fn sched_with(decay_json: &str) -> String {
    format!(
        r#"{{"min_lr":0.001,"peak_lr":0.01,"total_steps":4,"warmup_steps":1,"decay":{decay_json}}}"#
    )
}

/// Expected: ResumeState::read REFUSES each of these (T:`read_decay` wraps from_json's error in
/// Refused("...: schedule decay: ..."); T:with_decay re-validates every/gamma). Parser-level
/// failures surface as TrainError::Io/Json and also count as refused.
#[test]
fn resume_refuses_hostile_decay_members() {
    let root = scratch("hostile");
    let dir = checkpoint_of(base(Decay::Cosine), root.join("run"));
    let original = std::fs::read_to_string(dir.join("trainer.json")).unwrap();
    let cases: &[&str] = &[
        r#"{"kind":"Linear"}"#,
        r#"{"kind":"cosine"}"#,
        r#"{"kind":""}"#,
        r#"{"kind":null}"#,
        r#"{"kind":7}"#,
        r#"{}"#,
        r#"[]"#,
        r#"null"#,
        r#"3"#,
        r#""linear""#,
        r#"{"kind":"step"}"#,
        r#"{"kind":"step","every":2}"#,
        r#"{"kind":"step","gamma":0.5}"#,
        r#"{"kind":"step","every":0,"gamma":0.5}"#,
        r#"{"kind":"step","every":-1,"gamma":0.5}"#,
        r#"{"kind":"step","every":2.0,"gamma":0.5}"#,
        r#"{"kind":"step","every":1e2,"gamma":0.5}"#,
        r#"{"kind":"step","every":"2","gamma":0.5}"#,
        r#"{"kind":"step","every":18446744073709551616,"gamma":0.5}"#,
        r#"{"kind":"step","every":2,"gamma":"0.5"}"#,
        r#"{"kind":"step","every":2,"gamma":"NaN"}"#,
        r#"{"kind":"step","every":2,"gamma":1e309}"#,
        r#"{"kind":"step","every":2,"gamma":0.0}"#,
        r#"{"kind":"step","every":2,"gamma":-0.0}"#,
        r#"{"kind":"step","every":2,"gamma":0}"#,
        r#"{"kind":"step","every":2,"gamma":1e-400}"#,
        r#"{"kind":"step","every":2,"gamma":-0.5}"#,
        r#"{"kind":"step","every":2,"gamma":1.0000000000000002}"#,
        r#"{"kind":"step","every":2,"gamma":2}"#,
        r#"{"kind":"step","every":2,"gamma":null}"#,
        r#"{"kind":"constant","every":0,"gamma":"x"}"#,
        r#"{"kind":"linear","every":3,"gamma":0.5}"#,
        r#"{"kind":"step","every":2,"gamma":0.5,"zzz":null}"#,
    ];
    for c in cases {
        tamper(&dir, &sched_with(c));
        let r = ResumeState::read(&dir);
        assert!(r.is_err(), "decay {c} must be refused, got Ok");
        std::fs::write(dir.join("trainer.json"), &original).unwrap();
    }
    // Wrong-typed (not merely out-of-range) decays are refused by read_decay itself.
    for c in [
        r#"{"kind":"Linear"}"#,
        r#"[]"#,
        r#"{"kind":"step","every":"2","gamma":0.5}"#,
    ] {
        tamper(&dir, &sched_with(c));
        match ResumeState::read(&dir) {
            Err(TrainError::Refused(m)) => assert!(m.contains("schedule decay"), "{c}: {m}"),
            other => panic!("{c}: expected Refused, got {:?}", other.map(|_| ())),
        }
        std::fs::write(dir.join("trainer.json"), &original).unwrap();
    }
    std::fs::remove_dir_all(&root).ok();
}

/// Expected: these ARE accepted: every = u64::MAX and a denormal gamma; integer gamma 1.
#[test]
fn resume_accepts_the_extremes() {
    let root = scratch("accept");
    let dir = checkpoint_of(base(Decay::Cosine), root.join("run"));
    let original = std::fs::read_to_string(dir.join("trainer.json")).unwrap();
    let cases: &[(&str, Decay)] = &[
        (r#"{"kind":"constant"}"#, Decay::Constant),
        (r#"{"kind":"linear"}"#, Decay::Linear),
        (
            r#"{"kind":"step","every":18446744073709551615,"gamma":5e-324}"#,
            step(u64::MAX, 5e-324),
        ),
        (r#"{"kind":"step","every":1,"gamma":1}"#, step(1, 1.0)),
    ];
    for (c, want) in cases {
        tamper(&dir, &sched_with(c));
        let s = ResumeState::read(&dir).unwrap_or_else(|e| panic!("{c}: {e}"));
        assert_eq!(s.schedule.decay(), *want, "{c}");
        std::fs::write(dir.join("trainer.json"), &original).unwrap();
    }
    std::fs::remove_dir_all(&root).ok();
}

/// Resume `ckpt` under `configured`; the batches after the checkpoint's position.
fn resume_under(ckpt: &Path, configured: LrSchedule, out: PathBuf) -> Result<(), TrainError> {
    let state = ResumeState::read(ckpt)?;
    let mut provider = ToyProvider::new(8, 4, 2, 42).unwrap();
    train(
        &mut provider,
        &mut NextToken,
        (0..4).map(|n| Ok::<_, TrainError>(Batch(n))),
        &cfg(configured, out),
        &MonotonicClock::new(),
        Some(state),
        &mut Hooks::default(),
    )
    .map(|_| ())
}

/// Expected: the resumed schedule is compared with the configured one by derived PartialEq
/// including `decay` (T:581 `r.schedule != cfg.schedule`), so every cross-kind pair, and a step
/// with a different every/gamma, is Refused with the "schedule or grad_accum differs" message;
/// the identical schedule is NOT refused for that reason.
#[test]
fn resume_under_a_different_decay_is_refused_both_directions() {
    let root = scratch("mismatch");
    let kinds = [
        Decay::Cosine,
        Decay::Constant,
        Decay::Linear,
        step(2, 0.5),
        step(1, 0.5),
        step(2, 0.25),
        step(2, next_below_one()),
    ];
    for (i, written) in kinds.iter().enumerate() {
        let ckpt = checkpoint_of(base(*written), root.join(format!("w{i}")));
        for (j, configured) in kinds.iter().enumerate() {
            let r = resume_under(&ckpt, base(*configured), root.join(format!("r{i}-{j}")));
            if i == j {
                if let Err(TrainError::Refused(m)) = &r {
                    assert!(
                        !m.contains("schedule or grad_accum differs"),
                        "{written:?} resumed under itself was refused as a mismatch: {m}"
                    );
                }
            } else {
                match r {
                    Err(TrainError::Refused(m)) => assert!(
                        m.contains("schedule or grad_accum differs"),
                        "{written:?} -> {configured:?}: refused for another reason: {m}"
                    ),
                    Err(e) => panic!("{written:?} -> {configured:?}: wrong error {e}"),
                    Ok(()) => {
                        panic!("{written:?} resumed under {configured:?} was silently adopted")
                    }
                }
            }
        }
    }
    std::fs::remove_dir_all(&root).ok();
}

/// Expected: a checkpoint written BEFORE decay kinds existed (four-key schedule, no "decay")
/// resumes under Cosine and is refused under any other kind (absent == Cosine, S:from_json).
#[test]
fn a_four_key_checkpoint_is_cosine_and_only_cosine() {
    let root = scratch("legacy");
    let ckpt = checkpoint_of(base(Decay::Cosine), root.join("w"));
    let text = std::fs::read_to_string(ckpt.join("trainer.json")).unwrap();
    assert!(
        !text.contains("\"decay\""),
        "a cosine checkpoint carries no decay member"
    );
    for (j, configured) in [Decay::Constant, Decay::Linear, step(2, 0.5)]
        .iter()
        .enumerate()
    {
        match resume_under(&ckpt, base(*configured), root.join(format!("r{j}"))) {
            Err(TrainError::Refused(m)) => assert!(m.contains("differs"), "{m}"),
            Err(e) => panic!("{configured:?}: {e}"),
            Ok(()) => panic!("legacy cosine checkpoint adopted under {configured:?}"),
        }
    }
    std::fs::remove_dir_all(&root).ok();
}
