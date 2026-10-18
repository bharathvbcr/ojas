//! The Metal pre-flight: calibration from the committed LoRA-step rows, the
//! analytic plan, and the dry run. CPU only; the rows file is the real one.

use ojas_core::{Budget, OjasError};
use ojas_cpu::CpuBackend;
use ojas_io::parse_json;
use ojas_model::preflight::{
    dry_run, plan, round_to_bucket, Calibration, Point, PreflightError, PreflightRefusal,
    RefusalReason,
};
use std::path::PathBuf;

const BUCKETS: [u64; 2] = [2_048, 9_638];

fn rows_path() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../bench/results/2026-10-17-lora-step/rows.jsonl"
    ))
}

fn rows_text() -> String {
    let path = rows_path();
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "the calibration rows file is required and could not be read at {}: {e}",
            path.display()
        )
    })
}

/// `(snapshot, tokens, step_peak_bytes)` of every calibrating row.
fn ok_rows() -> Vec<(String, u64, u64)> {
    rows_text()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| parse_json(l).expect("rows.jsonl line parses"))
        .filter(|r| {
            r.get_str("arm").unwrap() == "tape-lora"
                && r.get_str("checkpointing").unwrap() == "on"
                && r.get_str("status").unwrap() == "ok"
        })
        .map(|r| {
            (
                r.get_str("snapshot").unwrap().to_owned(),
                r.get_u64("tokens").unwrap(),
                r.get_u64("step_peak_bytes").unwrap(),
            )
        })
        .collect()
}

fn snapshots() -> Vec<String> {
    let mut all: Vec<String> = ok_rows().into_iter().map(|r| r.0).collect();
    all.sort();
    all.dedup();
    all
}

fn calibrate(snapshot: &str) -> Calibration {
    Calibration::from_rows_path(&rows_path(), snapshot).expect("calibration from the real rows")
}

fn refusal(err: PreflightError) -> PreflightRefusal {
    match err {
        PreflightError::Refused(r) => r,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn calibration_reproduces_each_row_within_one_percent() {
    let rows = ok_rows();
    assert!(
        rows.len() >= 4,
        "expected the four ok rows, got {}",
        rows.len()
    );
    let snaps = snapshots();
    assert_eq!(snaps.len(), 2, "0.8B-Base and 2B-Base");
    for (snapshot, tokens, measured) in rows {
        let fitted = calibrate(&snapshot).peak_bytes(tokens).unwrap();
        let off = (fitted as f64 - measured as f64).abs() / measured as f64;
        assert!(
            off < 0.01,
            "{snapshot} at {tokens}: fitted {fitted} vs measured {measured} ({off})"
        );
    }
}

#[test]
fn the_fit_has_a_per_token_cost_in_the_measured_range() {
    for snapshot in snapshots() {
        let cal = calibrate(&snapshot);
        let (c0, c1) = cal.coefficients();
        assert!(c0 > 0.0, "{snapshot}: fixed cost {c0}");
        assert!(
            (3.0e5..1.2e6).contains(&c1),
            "{snapshot}: {c1} bytes per token is outside what the rows showed"
        );
        assert_eq!(cal.token_range(), (2_048, 9_638));
        assert!(cal.resident_bytes() > 0);
    }
}

#[test]
fn a_bucket_inside_the_rows_is_not_extrapolated_and_beyond_them_is() {
    let snapshot = &snapshots()[0];
    let cal = calibrate(snapshot);
    let inside = plan(&cal, 9_000, &BUCKETS, u64::MAX, 0).unwrap();
    assert_eq!((inside.tokens, inside.bucket), (9_000, 9_638));
    assert!(!inside.extrapolated);
    assert_eq!(inside.estimated_bytes, cal.peak_bytes(9_638).unwrap());

    let beyond = plan(&cal, 16_000, &[16_384], u64::MAX, 0).unwrap();
    assert!(beyond.extrapolated);
    assert!(beyond.estimated_bytes > inside.estimated_bytes);

    let below = plan(&cal, 100, &[512], u64::MAX, 0).unwrap();
    assert!(below.extrapolated);
}

#[test]
fn the_plan_refuses_with_the_shape_when_the_estimate_exceeds_cap_minus_live() {
    let snapshot = &snapshots()[0];
    let cal = calibrate(snapshot);
    let est = cal.peak_bytes(9_638).unwrap();

    assert!(plan(&cal, 9_000, &BUCKETS, est, 0).is_ok());
    let r = refusal(plan(&cal, 9_000, &BUCKETS, est - 1, 0).unwrap_err());
    assert_eq!(
        (
            r.tokens,
            r.bucket,
            r.estimated_bytes,
            r.cap_bytes,
            r.live_bytes
        ),
        (9_000, 9_638, est, est - 1, 0)
    );
    assert_eq!(r.reason, RefusalReason::Estimate);
    assert_eq!(r.measured_peak_bytes, None);

    let live = 1_u64 << 30;
    assert!(plan(&cal, 9_000, &BUCKETS, est + live, live).is_ok());
    let r = refusal(plan(&cal, 9_000, &BUCKETS, est + live - 1, live).unwrap_err());
    assert_eq!((r.cap_bytes, r.live_bytes), (est + live - 1, live));

    let text = r.to_string();
    assert!(text.contains("9000 tokens"), "{text}");
    assert!(text.contains("bucket 9638"), "{text}");
    assert!(text.contains(&est.to_string()), "{text}");
}

#[test]
fn live_bytes_above_the_cap_refuse() {
    let cal = calibrate(&snapshots()[0]);
    let est = cal.peak_bytes(2_048).unwrap();
    let r = refusal(plan(&cal, 2_048, &BUCKETS, est * 2, est * 2 + 1).unwrap_err());
    assert_eq!(r.reason, RefusalReason::Estimate);
}

#[test]
fn a_cap_below_resident_refuses_as_resident() {
    let cal = calibrate(&snapshots()[0]);
    let resident = cal.resident_bytes();
    let r = refusal(plan(&cal, 2_048, &BUCKETS, resident - 1, 0).unwrap_err());
    assert_eq!(r.reason, RefusalReason::ResidentExceedsCap);
    assert_eq!(r.estimated_bytes, resident);
    assert_eq!(r.cap_bytes, resident - 1);
    assert!(r.to_string().contains("resident weights"));
}

#[test]
fn a_snapshot_with_no_rows_refuses_and_never_passes() {
    let err = Calibration::from_rows_path(&rows_path(), "/no/such/snapshot").unwrap_err();
    assert!(
        matches!(&err, PreflightError::NoRows { snapshot } if snapshot == "/no/such/snapshot"),
        "{err:?}"
    );
    let err = Calibration::from_rows_text("", "anything").unwrap_err();
    assert!(matches!(err, PreflightError::NoRows { .. }), "{err:?}");
}

#[test]
fn rows_that_do_not_calibrate_are_skipped() {
    // The refused row (checkpointing off) carries no peak and is not a point.
    let snaps = snapshots();
    let two_b = snaps.iter().find(|s| s.contains("2B")).expect("2B row");
    let cal = calibrate(two_b);
    assert_eq!(cal.token_range(), (2_048, 9_638));
}

#[test]
fn a_missing_or_malformed_rows_file_is_a_calibration_error() {
    let missing = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../bench/results/does-not-exist/rows.jsonl"
    ));
    let err = Calibration::from_rows_path(&missing, "s").unwrap_err();
    assert!(matches!(err, PreflightError::Calibration { .. }), "{err:?}");
    let err = Calibration::from_rows_text("{not json", "s").unwrap_err();
    assert!(matches!(err, PreflightError::Calibration { .. }), "{err:?}");
}

fn pts(list: &[(u64, u64)]) -> Vec<Point> {
    list.iter()
        .map(|&(tokens, peak_bytes)| Point { tokens, peak_bytes })
        .collect()
}

#[test]
fn duplicate_non_monotone_and_single_rows_are_calibration_errors() {
    let dup = Calibration::from_points("s", 1, &pts(&[(10, 100), (10, 200), (20, 300)]));
    assert!(matches!(dup, Err(PreflightError::Calibration { .. })));
    let same_peak = Calibration::from_points("s", 1, &pts(&[(10, 100), (20, 100)]));
    assert!(matches!(same_peak, Err(PreflightError::Calibration { .. })));
    let falling = Calibration::from_points("s", 1, &pts(&[(10, 300), (20, 200)]));
    assert!(matches!(falling, Err(PreflightError::Calibration { .. })));
    let one = Calibration::from_points("s", 1, &pts(&[(10, 300)]));
    assert!(matches!(one, Err(PreflightError::Calibration { .. })));
    let none = Calibration::from_points("s", 1, &[]);
    assert!(matches!(none, Err(PreflightError::Calibration { .. })));
    let negative_intercept = Calibration::from_points("s", 1, &pts(&[(10, 10), (20, 1_000)]));
    assert!(matches!(
        negative_intercept,
        Err(PreflightError::Calibration { .. })
    ));
    let ok = Calibration::from_points("s", 1, &pts(&[(20, 300), (10, 200)])).unwrap();
    assert_eq!(ok.peak_bytes(30).unwrap(), 400);
}

#[test]
fn duplicate_rows_in_a_rows_file_are_a_calibration_error() {
    let row = r#"{"arm":"tape-lora","snapshot":"s","tokens":10,"checkpointing":"on","resident_bytes":5,"step_peak_bytes":100,"status":"ok"}"#;
    let text = format!("{row}\n{row}\n");
    let err = Calibration::from_rows_text(&text, "s").unwrap_err();
    assert!(matches!(err, PreflightError::Calibration { .. }), "{err:?}");
}

#[test]
fn zero_and_negative_tokens_are_errors() {
    let cal = calibrate(&snapshots()[0]);
    for tokens in [0_i64, -1, i64::MIN] {
        assert!(
            matches!(
                plan(&cal, tokens, &BUCKETS, u64::MAX, 0),
                Err(PreflightError::InvalidInput { .. })
            ),
            "plan {tokens}"
        );
        assert!(matches!(
            round_to_bucket(tokens, &BUCKETS),
            Err(PreflightError::InvalidInput { .. })
        ));
        let backend = CpuBackend::new(Budget::new(1 << 20));
        let mut ran = false;
        let out = dry_run(&backend, tokens, |_| {
            ran = true;
            Ok::<(), OjasError>(())
        });
        assert!(matches!(out, Err(PreflightError::InvalidInput { .. })));
        assert!(!ran, "the step must not run for {tokens} tokens");
    }
}

#[test]
fn the_bucket_rounds_up_to_the_runs_widths() {
    let buckets = [4_096, 2_048, 9_638];
    assert_eq!(round_to_bucket(1, &buckets).unwrap(), 2_048);
    assert_eq!(round_to_bucket(2_048, &buckets).unwrap(), 2_048);
    assert_eq!(round_to_bucket(2_049, &buckets).unwrap(), 4_096);
    assert_eq!(round_to_bucket(9_638, &buckets).unwrap(), 9_638);
    assert!(matches!(
        round_to_bucket(9_639, &buckets),
        Err(PreflightError::InvalidInput { .. })
    ));
    assert!(matches!(
        round_to_bucket(10, &[]),
        Err(PreflightError::InvalidInput { .. })
    ));
    assert!(matches!(
        round_to_bucket(10, &[0, 64]),
        Err(PreflightError::InvalidInput { .. })
    ));
}

#[test]
fn a_fitted_peak_beyond_u64_is_an_overflow_error() {
    let cal = Calibration::from_points("s", 1, &pts(&[(1, 1 << 62), (2, 1 << 63)])).unwrap();
    assert_eq!(cal.peak_bytes(2).unwrap(), 1 << 63);
    assert!(matches!(
        cal.peak_bytes(8),
        Err(PreflightError::Overflow { .. })
    ));
    assert!(matches!(
        plan(&cal, 8, &[8], u64::MAX, 0),
        Err(PreflightError::Overflow { .. })
    ));
}

#[test]
fn the_estimate_never_falls_below_a_measured_peak() {
    // Three points off a line: plain least squares passes below (2, 200).
    let points = pts(&[(1, 100), (2, 200), (3, 301), (4, 303)]);
    let cal = Calibration::from_points("s", 1, &points).unwrap();
    for p in &points {
        assert!(
            cal.peak_bytes(p.tokens).unwrap() >= p.peak_bytes,
            "under-estimates {p:?}"
        );
    }
    // Large values where f64 rounding alone can under-estimate.
    let big = pts(&[
        ((1 << 40) + 1, (1 << 62) + 3),
        ((1 << 40) + 7, (1 << 62) + 1_000_001),
        ((1 << 40) + 9, (1 << 62) + 1_000_333),
    ]);
    let cal = Calibration::from_points("s", 1, &big).unwrap();
    for p in &big {
        assert!(cal.peak_bytes(p.tokens).unwrap() >= p.peak_bytes, "{p:?}");
    }
}

#[test]
fn rows_of_another_arm_need_not_carry_this_arms_fields() {
    let other = r#"{"arm":"torch","tokens":10,"status":"ok"}"#;
    let a = r#"{"arm":"tape-lora","snapshot":"s","tokens":10,"checkpointing":"on","resident_bytes":5,"step_peak_bytes":100,"status":"ok"}"#;
    let b = r#"{"arm":"tape-lora","snapshot":"s","tokens":20,"checkpointing":"on","resident_bytes":5,"step_peak_bytes":200,"status":"ok","extra":[1,2]}"#;
    let text = format!("{other}\r\n\r\n{a}\r\n{b}\r\n");
    let cal = Calibration::from_rows_text(&text, "s").unwrap();
    assert_eq!(cal.token_range(), (10, 20));
}

#[test]
fn bom_float_and_wrong_typed_rows_fail_closed() {
    let a = r#"{"arm":"tape-lora","snapshot":"s","tokens":10,"checkpointing":"on","resident_bytes":5,"step_peak_bytes":100,"status":"ok"}"#;
    let bom = format!("\u{feff}{a}\n");
    assert!(matches!(
        Calibration::from_rows_text(&bom, "s"),
        Err(PreflightError::Calibration { .. })
    ));
    let float = a.replace("\"tokens\":10", "\"tokens\":10.0");
    assert!(matches!(
        Calibration::from_rows_text(&float, "s"),
        Err(PreflightError::Calibration { .. })
    ));
    let typed = a.replace("\"step_peak_bytes\":100", "\"step_peak_bytes\":\"100\"");
    assert!(matches!(
        Calibration::from_rows_text(&typed, "s"),
        Err(PreflightError::Calibration { .. })
    ));
    let huge = a.replace("\"step_peak_bytes\":100", "\"step_peak_bytes\":1e999");
    assert!(matches!(
        Calibration::from_rows_text(&huge, "s"),
        Err(PreflightError::Calibration { .. })
    ));
}

#[test]
fn a_panicking_dry_run_releases_the_child_charge() {
    let backend = CpuBackend::new(Budget::new(1_000));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = dry_run(&backend, 2_048, |budget| {
            let _held = budget.try_reserve(400)?;
            if budget.live_bytes()? == 400 {
                panic!("step panics while holding a charge");
            }
            Ok::<(), OjasError>(())
        });
    }));
    assert!(outcome.is_err());
    assert_eq!(backend.budget().live_bytes().unwrap(), 0);
}

#[test]
fn a_charge_on_the_parent_budget_is_invisible_to_the_dry_run() {
    // Documented limit: only the handed child budget is measured.
    let backend = CpuBackend::new(Budget::new(1_000));
    let out = dry_run(&backend, 2_048, |_| {
        let _outside = backend.budget().try_reserve(900)?;
        Ok::<(), OjasError>(())
    })
    .unwrap();
    assert_eq!(out.peak_bytes, 0);
}

#[test]
fn a_dry_run_refusal_with_a_huge_request_saturates() {
    let backend = CpuBackend::new(Budget::new(1_000));
    let out = dry_run(&backend, 2_048, |budget| {
        let _a = budget.try_reserve(10)?;
        let _b = budget.try_reserve(u64::MAX)?;
        Ok::<(), OjasError>(())
    });
    let r = refusal(out.unwrap_err());
    assert_eq!(r.estimated_bytes, u64::MAX);
    assert_eq!(r.measured_peak_bytes, Some(10));
}

#[test]
fn buckets_unsorted_duplicated_and_a_zero_cap_are_handled() {
    assert_eq!(round_to_bucket(5, &[64, 8, 8, 64, 16]).unwrap(), 8);
    let cal = Calibration::from_points("s", 7, &pts(&[(10, 100), (20, 200)])).unwrap();
    let r = refusal(plan(&cal, 10, &[10, 10], 0, 0).unwrap_err());
    assert_eq!(r.reason, RefusalReason::ResidentExceedsCap);
    let r = refusal(plan(&cal, 10, &[10], 1_000, u64::MAX).unwrap_err());
    assert_eq!(r.reason, RefusalReason::Estimate);
}

#[test]
fn a_tiny_budget_forces_the_dry_run_refusal_with_the_shape() {
    let backend = CpuBackend::new(Budget::new(1_000));
    let out = dry_run(&backend, 4_096, |budget| {
        let _a = budget.try_reserve(600)?;
        let _b = budget.try_reserve(600)?;
        Ok::<(), OjasError>(())
    });
    let r = refusal(out.unwrap_err());
    assert_eq!(
        (
            r.tokens,
            r.bucket,
            r.estimated_bytes,
            r.cap_bytes,
            r.live_bytes
        ),
        (4_096, 4_096, 1_200, 1_000, 600)
    );
    assert_eq!(r.reason, RefusalReason::DryRun);
    assert_eq!(r.measured_peak_bytes, Some(600));
    let text = r.to_string();
    assert!(text.contains("4096 tokens"), "{text}");
    assert!(text.contains("measured peak before refusal 600"), "{text}");
    assert_eq!(backend.budget().live_bytes().unwrap(), 0, "released");
}

#[test]
fn a_fitting_dry_run_returns_the_value_and_the_measured_peak() {
    let backend = CpuBackend::new(Budget::new(1_000));
    let out = dry_run(&backend, 2_048, |budget| {
        let held = budget.try_reserve(300)?;
        drop(held);
        let _again = budget.try_reserve(200)?;
        Ok::<u32, OjasError>(7)
    })
    .unwrap();
    assert_eq!(out.value, 7);
    assert_eq!(out.peak_bytes, 300);
    assert_eq!(backend.budget().live_bytes().unwrap(), 0);
}

#[test]
fn the_dry_run_counts_bytes_the_parent_already_holds() {
    let backend = CpuBackend::new(Budget::new(1_000));
    let _held = backend.budget().try_reserve(500).unwrap();
    let out = dry_run(&backend, 2_048, |budget| {
        let _a = budget.try_reserve(600)?;
        Ok::<(), OjasError>(())
    });
    let r = refusal(out.unwrap_err());
    assert_eq!(
        (r.estimated_bytes, r.cap_bytes, r.live_bytes),
        (1_100, 1_000, 500)
    );
    assert_eq!(r.measured_peak_bytes, Some(0), "a refusal raises no peak");
}

#[test]
fn a_non_capacity_step_error_is_not_a_refusal() {
    let backend = CpuBackend::new(Budget::new(1_000));
    let out = dry_run(&backend, 2_048, |_| {
        Err::<(), OjasError>(OjasError::Shape {
            op: "step",
            detail: "bad".to_owned(),
        })
    });
    assert!(matches!(
        out,
        Err(PreflightError::Step(OjasError::Shape { .. }))
    ));
}
