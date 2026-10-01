//! `LrSchedule::lr_at` against `LRSchedule.lr_at`, bit for bit, on values
//! `tools/qd_train_oracle_trainer.py` dumped from Python on this host.

mod common;

use qd_train::schedule::LrSchedule;

#[test]
fn every_dumped_rate_is_pythons_to_the_bit() {
    let o = common::oracle();
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
    let o = common::oracle();
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
    assert_eq!(matched, 7, "the seven real_ft_run schedules in the fixture");
}
