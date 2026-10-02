//! The parity gates: pinned tolerances, and runners that pass a model
//! reproducing torch and fail one that misses by more than a gate.
//!
//! `Replay` is not a model: it returns the fixtures' own numbers, optionally
//! perturbed, so these tests exercise the gates and nothing else. The model
//! tests run once ojas-model implements `ParityModel` (README).

use ojas_core::OjasError;
use ojas_oracle::golden::{
    tiny_forward, tiny_grads, tiny_trace, GradsAt, HostTensor, Ns5, TensorSet, TokenBatch,
    TrainSetup,
};
use ojas_oracle::parity::{
    check_curve, check_tensors, curve_parity, forward_parity, grads_parity, normwise_rel,
    trace_parity, ParityModel, CURVE_EARLY_ABS_TOL, CURVE_EARLY_STEPS, CURVE_LATE_ABS_TOL,
    CURVE_MIN_DROP_NATS, FORWARD_LOSS_REL_TOL, GRAD_NORMWISE_REL_TOL, LOGITS_NORMWISE_REL_TOL,
    LR_MULT_REL_TOL, TRACE_LOSS_ABS_TOL, TRACE_PARAM_NORMWISE_REL_TOL,
};

#[test]
fn tolerances_are_pinned() {
    assert_eq!(FORWARD_LOSS_REL_TOL, 1e-5);
    assert_eq!(LOGITS_NORMWISE_REL_TOL, 1e-5);
    assert_eq!(GRAD_NORMWISE_REL_TOL, 1e-4);
    assert_eq!(TRACE_LOSS_ABS_TOL, 1e-4);
    assert_eq!(TRACE_PARAM_NORMWISE_REL_TOL, 1e-4);
    assert_eq!((CURVE_EARLY_ABS_TOL, CURVE_EARLY_STEPS), (2e-4, 10));
    assert_eq!(CURVE_LATE_ABS_TOL, 2e-3);
    assert_eq!(CURVE_MIN_DROP_NATS, 1.0);
    assert_eq!(LR_MULT_REL_TOL, 1e-12);
}

#[derive(Default)]
struct Replay {
    loss_scale: f64,
    grad_scale: Option<(String, f32)>,
    grad_extra: Option<(String, f32)>,
    drop_grad: Option<String>,
    loss_shift: Option<(usize, f64)>,
    param_scale: Option<(String, f32)>,
    trace_steps_seen: Vec<usize>,
}

impl Replay {
    fn exact() -> Self {
        Self {
            loss_scale: 1.0,
            ..Self::default()
        }
    }
}

fn scaled(set: &TensorSet, edit: &Option<(String, f32)>) -> TensorSet {
    let tensors = set
        .iter()
        .map(|t| {
            let mut t = t.clone();
            if let Some((name, s)) = edit {
                if &t.name == name {
                    t.data.iter_mut().for_each(|v| *v *= s);
                }
            }
            t
        })
        .collect();
    TensorSet::new(tensors).unwrap()
}

impl ParityModel for Replay {
    fn forward(&mut self, _: &TensorSet, batch: &TokenBatch) -> Result<(f32, Vec<f32>), OjasError> {
        let fx = tiny_forward()?;
        assert_eq!(batch, &fx.batch, "runner passes the fixture batch");
        Ok(((f64::from(fx.loss) * self.loss_scale) as f32, fx.logits))
    }

    fn grads(
        &mut self,
        params: &TensorSet,
        _: &TokenBatch,
        seed: f32,
    ) -> Result<TensorSet, OjasError> {
        assert_eq!(seed, 0.5);
        // Which fixture the runner meant, from the parameters it passed.
        let at = if params
            .get("blocks.0.mixer.o_proj.weight")
            .unwrap()
            .data
            .iter()
            .all(|&v| v == 0.0)
        {
            GradsAt::Init
        } else {
            GradsAt::Step5
        };
        let fx = tiny_grads(at)?;
        let mut out: Vec<HostTensor> = scaled(&fx.grads, &self.grad_scale)
            .iter()
            .filter(|t| Some(&t.name) != self.drop_grad.as_ref())
            .cloned()
            .collect();
        if let Some((name, v)) = &self.grad_extra {
            if let Some(t) = out.iter_mut().find(|t| &t.name == name) {
                t.data[0] += v;
            } else {
                out.push(HostTensor {
                    name: name.clone(),
                    shape: vec![1],
                    data: vec![*v],
                });
            }
        }
        TensorSet::new(out)
    }

    fn train(
        &mut self,
        _: &TensorSet,
        setup: &TrainSetup,
        steps: usize,
    ) -> Result<(Vec<f64>, TensorSet), OjasError> {
        assert_eq!((setup.batch, setup.accum, setup.seq_len), (2, 2, 32));
        self.trace_steps_seen.push(steps);
        let tr = tiny_trace(Ns5::F32)?;
        let mut losses = tr.mean_loss[..steps].to_vec();
        if let Some((i, d)) = self.loss_shift {
            if i < steps {
                losses[i] += d;
            }
        }
        Ok((losses, scaled(&tr.params, &self.param_scale)))
    }
}

#[test]
fn a_model_that_reproduces_torch_passes_every_gate() {
    let mut m = Replay::exact();
    assert_eq!(forward_parity(&mut m).unwrap().worst, 0.0);
    assert_eq!(grads_parity(&mut m, GradsAt::Init).unwrap().worst, 0.0);
    assert_eq!(grads_parity(&mut m, GradsAt::Step5).unwrap().worst, 0.0);
    assert_eq!(trace_parity(&mut m).unwrap().worst, 0.0);
    curve_parity(&mut m).unwrap();
    assert_eq!(m.trace_steps_seen, vec![5, 40]);
}

#[test]
fn forward_gate_is_one_part_in_1e5() {
    let mut m = Replay {
        loss_scale: 1.0 + 0.5e-5,
        ..Replay::exact()
    };
    forward_parity(&mut m).unwrap();
    let mut m = Replay {
        loss_scale: 1.0 + 2e-5,
        ..Replay::exact()
    };
    let err = forward_parity(&mut m).unwrap_err();
    assert_eq!(err.check, "forward");
}

#[test]
fn grads_gate_is_normwise_1e4_and_exact_at_zero() {
    let at5 = |m: &mut Replay| grads_parity(m, GradsAt::Step5);
    let name = "blocks.1.mixer.q_proj.weight".to_string();
    at5(&mut Replay {
        grad_scale: Some((name.clone(), 1.0 + 0.5e-4)),
        ..Replay::exact()
    })
    .unwrap();
    assert!(at5(&mut Replay {
        grad_scale: Some((name.clone(), 1.0 + 2e-4)),
        ..Replay::exact()
    })
    .is_err());
    assert!(at5(&mut Replay {
        drop_grad: Some(name),
        ..Replay::exact()
    })
    .is_err());
    // At the init torch's q_proj gradient is exactly zero: any value fails.
    let q0 = "blocks.0.mixer.q_proj.weight".to_string();
    let mut m = Replay {
        grad_extra: Some((q0, 1e-30)),
        ..Replay::exact()
    };
    assert!(grads_parity(&mut m, GradsAt::Init).is_err());
    // Layer 0's vr_lambda has no torch gradient: zero is allowed, not more.
    let vr0 = "blocks.0.mixer.vr_lambda".to_string();
    grads_parity(
        &mut Replay {
            grad_extra: Some((vr0.clone(), 0.0)),
            ..Replay::exact()
        },
        GradsAt::Init,
    )
    .unwrap();
    assert!(grads_parity(
        &mut Replay {
            grad_extra: Some((vr0, 1e-9)),
            ..Replay::exact()
        },
        GradsAt::Init
    )
    .is_err());
    assert!(grads_parity(
        &mut Replay {
            grad_extra: Some(("bogus".into(), 0.0)),
            ..Replay::exact()
        },
        GradsAt::Init
    )
    .is_err());
}

#[test]
fn trace_gate_is_1e4_on_losses_and_params() {
    trace_parity(&mut Replay {
        loss_shift: Some((2, 0.9e-4)),
        ..Replay::exact()
    })
    .unwrap();
    assert!(trace_parity(&mut Replay {
        loss_shift: Some((2, 1.5e-4)),
        ..Replay::exact()
    })
    .is_err());
    assert!(trace_parity(&mut Replay {
        loss_shift: Some((4, f64::NAN)),
        ..Replay::exact()
    })
    .is_err());
    let p = "blocks.1.ffn.up.weight".to_string();
    trace_parity(&mut Replay {
        param_scale: Some((p.clone(), 1.0 + 0.5e-4)),
        ..Replay::exact()
    })
    .unwrap();
    assert!(trace_parity(&mut Replay {
        param_scale: Some((p, 1.0 + 2e-4)),
        ..Replay::exact()
    })
    .is_err());
}

#[test]
fn curve_gate_follows_section_10() {
    let want = tiny_trace(Ns5::F32).unwrap().mean_loss;
    let shifted = |i: usize, d: f64| {
        let mut v = want.clone();
        v[i] += d;
        v
    };
    check_curve(&shifted(9, 1.9e-4), &want, 256).unwrap();
    assert!(
        check_curve(&shifted(9, 2.1e-4), &want, 256).is_err(),
        "step 10 is early"
    );
    check_curve(&shifted(10, 1.9e-3), &want, 256).unwrap();
    assert!(check_curve(&shifted(10, 2.1e-3), &want, 256).is_err());
    assert!(check_curve(&want[..39], &want, 256).is_err());
    // A curve that ends less than 1 nat below ln V fails even when it matches.
    let flat = vec![(256f64).ln() - 0.5; 40];
    assert!(check_curve(&flat, &flat, 256).is_err());
}

#[test]
fn normwise_and_tensor_checks_fail_closed() {
    assert_eq!(normwise_rel("t", &[0.0, 0.0], &[0.0, 0.0]).unwrap(), 0.0);
    assert_eq!(normwise_rel("t", &[1e-30], &[0.0]).unwrap(), f64::INFINITY);
    assert!(normwise_rel("t", &[f32::NAN], &[1.0]).is_err());
    assert!(normwise_rel("t", &[1.0], &[1.0, 2.0]).is_err());
    assert!(
        (normwise_rel("t", &[3.0, 4.0], &[3.0, 4.5]).unwrap() - 0.5 / (9.0f64 + 20.25).sqrt())
            .abs()
            < 1e-15
    );
    let t = |name: &str, shape: Vec<usize>, data: Vec<f32>| HostTensor {
        name: name.into(),
        shape,
        data,
    };
    let want = TensorSet::new(vec![t("a", vec![2], vec![1.0, 2.0])]).unwrap();
    let reshaped = TensorSet::new(vec![t("a", vec![2, 1], vec![1.0, 2.0])]).unwrap();
    assert!(check_tensors("t", &reshaped, &want, &[], 1e-4).is_err());
    assert!(TensorSet::new(vec![t("a", vec![3], vec![1.0])]).is_err());
    assert!(TensorSet::new(vec![t("a", vec![1], vec![1.0]), t("a", vec![1], vec![1.0])]).is_err());
}
