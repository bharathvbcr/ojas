//! `LrSchedule::lr_at` against `LRSchedule.lr_at`, bit for bit, on values
//! `tools/qd_train_oracle_trainer.py` dumped from Python on this host.

mod common;

use qd_train::schedule::LrSchedule;

#[test]
fn every_dumped_rate_is_pythons_to_the_bit() {
    let o = common::trainer_oracle();
    let mut compared = 0usize;
    for case in o["schedule"].as_array().unwrap() {
        let peak = common::fhex(&case["peak_lr"]);
        let floor = common::fhex(&case["min_lr"]);
        let total = case["total_steps"].as_u64().unwrap();
        let warmup = case["warmup_steps"].as_u64().unwrap();
        let s = LrSchedule::new(peak, total, warmup, floor).unwrap();
        let steps = case["steps"].as_array().unwrap();
        let want = common::fhex_list(&case["lr"]);
        assert_eq!(steps.len(), want.len());
        for (step, w) in steps.iter().zip(want) {
            let step = step.as_u64().unwrap();
            let got = s.lr_at(step).unwrap();
            assert_eq!(
                got.to_bits(),
                w.to_bits(),
                "schedule ({peak:e}, {total}, {warmup}, {floor:e}) step {step}: {got:e} vs Python {w:e}"
            );
            compared += 1;
        }
    }
    assert!(compared > 200, "only {compared} rates compared");
}

#[test]
fn real_ft_matches_the_schedules_python_built_for_real_ft_run() {
    // The oracle built these through real_ft_run._control's rule; LrSchedule::real_ft must
    // reproduce the same four fields from (lr, steps) alone.
    let o = common::trainer_oracle();
    let mut matched = 0;
    for case in o["schedule"].as_array().unwrap() {
        let peak = common::fhex(&case["peak_lr"]);
        let total = case["total_steps"].as_u64().unwrap();
        let built = LrSchedule::real_ft(peak, total).unwrap();
        if built.warmup_steps() == case["warmup_steps"].as_u64().unwrap()
            && built.min_lr().to_bits() == common::fhex(&case["min_lr"]).to_bits()
        {
            matched += 1;
        }
    }
    assert_eq!(matched, 8, "the eight real_ft_run schedules in the fixture");
}

/// `--min-lr 0` (v5 `recipe.added[1]`): `LrSchedule::real_ft_with_floor(lr, steps, 0.0)` is
/// `LRSchedule(lr, steps, max(1, steps // 20), 0.0)` bit for bit at every dumped step. Four
/// fixture schedules follow that rule: the three the oracle dumps for it, (1e-5, 20),
/// (1e-5, 200) and F's epoch length (1e-5, 9683), and the older direct (3e-4, 7, 1, 0.0),
/// whose warmup is also `max(1, 7 // 20)`.
#[test]
fn real_ft_with_floor_reproduces_every_floor_zero_schedule_bit_for_bit() {
    let o = common::trainer_oracle();
    let mut matched = Vec::new();
    for case in o["schedule"].as_array().unwrap() {
        let floor = common::fhex(&case["min_lr"]);
        if floor.to_bits() != 0.0f64.to_bits() {
            continue;
        }
        let peak = common::fhex(&case["peak_lr"]);
        let total = case["total_steps"].as_u64().unwrap();
        let built = LrSchedule::real_ft_with_floor(peak, total, 0.0).unwrap();
        if built.warmup_steps() != case["warmup_steps"].as_u64().unwrap() {
            continue;
        }
        let steps = case["steps"].as_array().unwrap();
        let want = common::fhex_list(&case["lr"]);
        for (step, w) in steps.iter().zip(want) {
            let step = step.as_u64().unwrap();
            let got = built.lr_at(step).unwrap();
            assert_eq!(
                got.to_bits(),
                w.to_bits(),
                "real_ft_with_floor({peak:e}, {total}, 0) step {step}: {got:e} vs Python {w:e}"
            );
        }
        matched.push((peak, total));
    }
    assert_eq!(matched.len(), 4, "floor-0 schedules on the real_ft rule: {matched:?}");
    assert!(matched.contains(&(1e-5, 9683)), "F's epoch length at floor 0: {matched:?}");
}

/// The oracle's case for `(peak, total, warmup, min_lr)`, its steps and Python's rates.
fn case(o: &serde_json::Value, peak: f64, total: u64, min_lr: f64) -> (Vec<u64>, Vec<f64>) {
    let c = o["schedule"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| {
            common::fhex(&c["peak_lr"]).to_bits() == peak.to_bits()
                && c["total_steps"].as_u64() == Some(total)
                && common::fhex(&c["min_lr"]).to_bits() == min_lr.to_bits()
        })
        .unwrap_or_else(|| panic!("no oracle case ({peak:e}, {total}, min {min_lr:e})"));
    (
        c["steps"].as_array().unwrap().iter().map(|s| s.as_u64().unwrap()).collect(),
        common::fhex_list(&c["lr"]),
    )
}

/// Rung (d): `LRSchedule(peak_lr=1e-5, total_steps=200, warmup_steps=10, ...)`, bit-equal to
/// Python at every one of its 200 steps. The design (`fable-rung-d-resize.md`) calls it "the
/// recipe's formula at steps=200" and writes `min_lr=1e-6`; the formula
/// (`real_ft_run.py:1833-1834`, `min_lr=lr / 10`) gives 1.0000000000000002e-06, one ulp above
/// 1e-6, and that is what the torch arms (`--max-steps 200`) will run. The trainer uses the
/// formula; the literal is pinned too, and the two are shown to be different schedules.
#[test]
fn rung_d_schedule_is_the_recipes_formula_at_200_steps_bit_equal_at_every_step() {
    let o = common::trainer_oracle();
    let s = LrSchedule::real_ft(1e-5, 200).unwrap();
    assert_eq!(s.warmup_steps(), 10);
    assert_eq!(s.min_lr().to_bits(), (1e-5f64 / 10.0).to_bits());
    assert_ne!(s.min_lr().to_bits(), 1e-6f64.to_bits(), "lr / 10 is not the literal 1e-6");
    let (steps, want) = case(&o, 1e-5, 200, 1e-5 / 10.0);
    assert_eq!(steps, (0..200).collect::<Vec<_>>(), "every step of the schedule is compared");
    for (step, w) in steps.iter().zip(&want) {
        assert_eq!(s.lr_at(*step).unwrap().to_bits(), w.to_bits(), "rung (d) step {step}");
    }
    assert!(s.lr_at(200).is_err(), "step 200 is past the schedule");

    let literal = LrSchedule::new(1e-5, 200, 10, 1e-6).unwrap();
    let (lsteps, lwant) = case(&o, 1e-5, 200, 1e-6);
    let mut differ = 0;
    for (step, w) in lsteps.iter().zip(&lwant) {
        let got = literal.lr_at(*step).unwrap();
        assert_eq!(got.to_bits(), w.to_bits(), "literal schedule step {step}");
        differ += usize::from(got.to_bits() != s.lr_at(*step).unwrap().to_bits());
    }
    eprintln!("rung (d): the literal min_lr=1e-6 schedule differs from the formula's at {differ} of 200 steps");
}
