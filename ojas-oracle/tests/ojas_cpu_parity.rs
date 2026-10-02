//! Parity that can run today: ojas-cpu's schedules and optimizer policy
//! against what the torch fixtures record. (ojas-cpu is this crate's
//! existing dev-dependency.)

use ojas_cpu::{
    optim_group, CosineSchedule, OptimGroup, WsdSchedule, ADAM_HYBRID_WEIGHT_DECAY,
    COSINE_FLOOR_FRAC, MUON_MOMENTUM, MUON_WEIGHT_DECAY, WSD_DECAY_FRAC,
};
use ojas_oracle::golden::{lr_schedules, tiny_trace, Ns5};
use ojas_oracle::parity::LR_MULT_REL_TOL;
use ojas_oracle::spec::{expected_params, tiny_spec, Group};

fn assert_close(got: f64, want: f64, what: &str) {
    let rel = (got - want).abs() / want.abs();
    assert!(
        rel <= LR_MULT_REL_TOL,
        "{what}: ojas {got} vs nanolab {want} (rel {rel:e})"
    );
}

#[test]
fn ojas_cpu_schedules_match_nanolab_multipliers() {
    for case in lr_schedules().unwrap().cases {
        assert_eq!(case.lr_floor_frac, COSINE_FLOOR_FRAC);
        let (w, t) = (case.warmup_steps as u64, case.total_steps as u64);
        for (step, &want) in case.multipliers.iter().enumerate() {
            let got = match case.schedule.as_str() {
                "cosine" => CosineSchedule::new(w, t).unwrap().multiplier(step as u64),
                "wsd" => {
                    assert_eq!(case.wsd_decay_frac, WSD_DECAY_FRAC);
                    WsdSchedule::new(w, t, case.wsd_decay_frac)
                        .unwrap()
                        .multiplier(step as u64)
                }
                other => panic!("unknown schedule {other}"),
            }
            .unwrap();
            assert_close(got, want, &format!("{} step {step}", case.name));
        }
    }
}

#[test]
fn ojas_cpu_cosine_matches_the_trace_schedule() {
    let tr = tiny_trace(Ns5::F32).unwrap();
    let s = &tr.train;
    assert_eq!(s.schedule, "cosine");
    let sched = CosineSchedule::new(s.warmup_steps as u64, s.total_steps as u64).unwrap();
    for (step, &want) in tr.lr_mult.iter().enumerate() {
        assert_close(
            sched.multiplier(step as u64).unwrap(),
            want,
            &format!("step {step}"),
        );
    }
}

#[test]
fn ojas_cpu_optimizer_policy_matches_the_trace_config() {
    let tr = tiny_trace(Ns5::F32).unwrap();
    assert_eq!(tr.train.muon_momentum, MUON_MOMENTUM);
    assert_eq!(tr.train.weight_decay, MUON_WEIGHT_DECAY);
    assert_eq!(ADAM_HYBRID_WEIGHT_DECAY, 0.0);
    // §2's group column (checked against nanolab's _split_params by the
    // exporter) agrees with ojas-cpu's grouping rule.
    for row in expected_params(&tiny_spec()) {
        let ojas = optim_group(row.shape.len(), row.name == "tok_emb.weight");
        let want = match row.group {
            Group::Muon => OptimGroup::MuonMatrix,
            Group::AdamW if row.name == "tok_emb.weight" => OptimGroup::AdamEmbedding,
            Group::AdamW => OptimGroup::AdamVector,
        };
        assert_eq!(ojas, want, "{}", row.name);
    }
}
