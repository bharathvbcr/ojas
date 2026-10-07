//! Parity that can run today: ojas-cpu's schedules and optimizer policy
//! against what the torch fixtures record. (ojas-cpu is this crate's
//! existing dev-dependency.)

use ojas_core::{Backend, Budget, MuonNs5Config, Ns5Precision, Numerics, Tensor};
use ojas_cpu::{
    optim_group, CosineSchedule, CpuBackend, OptimGroup, WsdSchedule, ADAM_HYBRID_WEIGHT_DECAY,
    COSINE_FLOOR_FRAC, MUON_MOMENTUM, MUON_WEIGHT_DECAY, WSD_DECAY_FRAC,
};
use ojas_oracle::golden::{lr_schedules, muon_step_bf16, tiny_trace, MuonStepCase, Ns5};
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

/// One ojas Muon step on a fixture case: the parameter and momentum after it.
fn ojas_muon_step(
    numerics: Numerics,
    case: &MuonStepCase,
    ns5: Ns5Precision,
) -> (Vec<f32>, Vec<f32>) {
    let cpu = CpuBackend::new(Budget::new(1 << 28)).with_numerics(numerics);
    let shape = [case.rows, case.cols];
    let mut p = Tensor::from_f32(&case.p, &shape, cpu.budget()).unwrap();
    let g = Tensor::from_f32(&case.g, &shape, cpu.budget()).unwrap();
    let mut m = Tensor::from_f32(&case.m, &shape, cpu.budget()).unwrap();
    let config = MuonNs5Config {
        lr: case.lr,
        momentum: case.momentum,
        weight_decay: case.weight_decay,
        nesterov: case.nesterov,
        ns5,
    };
    cpu.muon_ns5_step(&mut p, &g, &mut m, config).unwrap();
    (p.to_f32_vec().unwrap(), m.to_f32_vec().unwrap())
}

/// `‖(got − p) − (want − p)‖ / ‖want − p‖`: the error of the step's whole
/// change to the parameter (decay and update), relative to torch's.
fn update_rel(case: &MuonStepCase, got: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for ((&g, &w), &p) in got.iter().zip(&case.p_after).zip(&case.p) {
        let (dg, dw) = (f64::from(g) - f64::from(p), f64::from(w) - f64::from(p));
        num += (dg - dw) * (dg - dw);
        den += dw * dw;
    }
    (num / den).sqrt()
}

/// Stock nanolab's Muon step (`X = G.bfloat16()`) against ojas's
/// [`Ns5Precision::Bf16`] on CPU, under both numerics tiers, on a square, a
/// tall and a wide matrix and a case without decay or Nesterov.
///
/// The momentum buffer is f32 in both and must match bit for bit. Every
/// value of the new parameter must too: on these shapes a bf16 GEMM output
/// rounds the same whatever order its f32 sum is taken in, so the only
/// freedom left is none. The f32 iteration on the same inputs is the
/// contrast: it is off by about 3% of the update, which the bf16 path must
/// not be.
#[test]
fn ojas_cpu_bf16_muon_step_matches_stock_nanolab() {
    let cases = muon_step_bf16().unwrap();
    assert_eq!(cases.len(), 4);
    for numerics in [Numerics::Exact, Numerics::Fast] {
        for (i, case) in cases.iter().enumerate() {
            let what = format!("{numerics:?} case {i} {}x{}", case.rows, case.cols);
            let (p, m) = ojas_muon_step(numerics, case, Ns5Precision::Bf16);
            let m_same = m
                .iter()
                .zip(&case.m_after)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(m_same, "{what}: momentum buffer differs from torch");
            let differ = p
                .iter()
                .zip(&case.p_after)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let rel = update_rel(case, &p);
            let (p32, _) = ojas_muon_step(numerics, case, Ns5Precision::F32);
            let rel32 = update_rel(case, &p32);
            eprintln!(
                "{what}: bf16 NS5 {differ}/{} values differ, update rel {rel:e}; f32 NS5 update rel {rel32:e}",
                p.len()
            );
            assert_eq!(
                differ, 0,
                "{what}: bf16 step is not torch's bits (update rel {rel:e})"
            );
            assert!(
                rel32 > 1e-2,
                "{what}: the f32 contrast is too close ({rel32:e}) to tell"
            );
        }
    }
}
