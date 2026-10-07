//! The ojas-oracle torch fixtures (framework-design.md §9 item 13) against
//! this crate on `CpuBackend` (Exact): forward loss and logits, gradients
//! at the init and at step 5, the 5-step trace and the §10 40-step curve,
//! each against both the f32-NS5 trace and stock nanolab's bf16-NS5 one. The runners in `ojas_oracle::parity` own the fixtures and the
//! tolerances; this file only adapts the model to `ParityModel`.

use ojas_autograd::Tape;
use ojas_core::{Backend, Budget, Ns5Precision, Numerics, OjasError, Tensor};
use ojas_cpu::{
    CosineSchedule, CpuBackend, LrSchedule, WsdSchedule, COSINE_FLOOR_FRAC, MUON_MOMENTUM,
    MUON_WEIGHT_DECAY,
};
use ojas_data::TokenBin;
use ojas_io::SafeTensors;
use ojas_model::{
    bind, forward_logits, forward_loss, load_model, param_table, ActivationCheckpoint::Off, Eval,
    Init, ModelSpec, Rope, StepReport, TrainConfig, Trainer, DEFAULT_CE_CHUNK,
};
use ojas_oracle::golden::{
    tiny_init, tiny_trace, HostTensor, Ns5, TensorSet, TokenBatch, TrainSetup, TINY_DIR,
};
use ojas_oracle::parity::{self, ParityModel};
use ojas_oracle::spec::{expected_params, tiny_spec, Group};

fn exact() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 30)).with_numerics(Numerics::Exact)
}

fn refuse(detail: String) -> OjasError {
    OjasError::Unsupported {
        op: "oracle adapter",
        detail,
    }
}

/// `set`'s tensors in [`param_table`] order, as host tensors.
fn params_of(spec: &ModelSpec, set: &TensorSet, budget: &Budget) -> Result<Vec<Tensor>, OjasError> {
    param_table(spec)?
        .iter()
        .map(|info| {
            let t = set
                .get(&info.name)
                .ok_or_else(|| refuse(format!("fixture has no {}", info.name)))?;
            if t.shape != info.shape {
                return Err(refuse(format!("{}: shape {:?}", info.name, t.shape)));
            }
            Tensor::from_f32(&t.data, &t.shape, budget)
        })
        .collect()
}

fn ids(batch: &TokenBatch, budget: &Budget) -> Result<(Tensor, Tensor), OjasError> {
    let x = Tensor::from_u32(&batch.x, &[batch.batch, batch.seq_len], budget)?;
    let y = Tensor::from_u32(&batch.y, &[batch.batch * batch.seq_len], budget)?;
    Ok((x, y))
}

/// A trainer config from the fixture's nanolab `Config`, refusing any
/// constant this crate's optimizer does not use.
fn train_config(setup: &TrainSetup) -> Result<TrainConfig, OjasError> {
    let same = |what: &str, got: f64, want: f64| {
        if got == want {
            Ok(())
        } else {
            Err(refuse(format!("{what} {got} != {want}")))
        }
    };
    if setup.optimizer != "muon_ns5_adamw" {
        return Err(refuse(format!("optimizer {}", setup.optimizer)));
    }
    same("muon_momentum", setup.muon_momentum, MUON_MOMENTUM)?;
    same("weight_decay", setup.weight_decay, MUON_WEIGHT_DECAY)?;
    same("beta1", setup.beta1, ojas_core::ADAMW_BETA1)?;
    same("beta2", setup.beta2, ojas_core::ADAMW_BETA2)?;
    same("eps", setup.eps, ojas_core::ADAMW_EPS)?;
    same("lr_floor_frac", setup.lr_floor_frac, COSINE_FLOOR_FRAC)?;
    let (warmup, total) = (setup.warmup_steps as u64, setup.total_steps as u64);
    let schedule = match setup.schedule.as_str() {
        "cosine" => LrSchedule::Cosine(CosineSchedule::new(warmup, total)?),
        "wsd" => LrSchedule::Wsd(WsdSchedule::new(warmup, total, setup.wsd_decay_frac)?),
        other => return Err(refuse(format!("schedule {other}"))),
    };
    let mut cfg = TrainConfig::nanolab(
        setup.batch,
        setup.seq_len,
        setup.accum,
        setup.seed,
        schedule,
    );
    cfg.matrix_lr = setup.matrix_lr;
    cfg.adam_lr = setup.lr;
    cfg.muon_ns5 = match setup.ns5 {
        Ns5::F32 => Ns5Precision::F32,
        Ns5::Bf16 => Ns5Precision::Bf16,
    };
    cfg.grad_clip = setup.grad_clip as f32;
    if f64::from(cfg.grad_clip) != setup.grad_clip {
        return Err(refuse(format!(
            "grad_clip {} is not an f32",
            setup.grad_clip
        )));
    }
    Ok(cfg)
}

/// `ParityModel` over `Eval`, `Tape` and `Trainer` on CPU Exact. Keeps the
/// step reports of the last `train` call for the extra checks below.
#[derive(Default)]
struct Cpu {
    reports: Vec<StepReport>,
}

impl ParityModel for Cpu {
    fn forward(
        &mut self,
        params: &TensorSet,
        batch: &TokenBatch,
    ) -> Result<(f32, Vec<f32>), OjasError> {
        let spec = ModelSpec::tiny();
        let cpu = exact();
        let budget = cpu.budget().clone();
        let host = params_of(&spec, params, &budget)?;
        let rope = Rope::new(&spec, batch.seq_len, &budget)?;
        let (x, y) = ids(batch, &budget)?;
        let mut eval = Eval::new(cpu);
        let p = bind(&mut eval, &spec, &host)?;
        let loss = forward_loss(
            &mut eval,
            &spec,
            &p,
            &x,
            &y,
            &rope,
            None,
            DEFAULT_CE_CHUNK,
            Off,
        )?;
        let logits = forward_logits(&mut eval, &spec, &p, &x, &rope)?;
        Ok((loss.to_f32_vec()?[0], logits.to_f32_vec()?))
    }

    fn grads(
        &mut self,
        params: &TensorSet,
        batch: &TokenBatch,
        seed: f32,
    ) -> Result<TensorSet, OjasError> {
        let spec = ModelSpec::tiny();
        let cpu = exact();
        let budget = cpu.budget().clone();
        let host = params_of(&spec, params, &budget)?;
        let rope = Rope::new(&spec, batch.seq_len, &budget)?;
        let (x, y) = ids(batch, &budget)?;
        let mut tape = Tape::new(cpu);
        let p = bind(&mut tape, &spec, &host)?;
        let loss = forward_loss(
            &mut tape,
            &spec,
            &p,
            &x,
            &y,
            &rope,
            None,
            DEFAULT_CE_CHUNK,
            Off,
        )?;
        tape.backward_seeded(loss, seed)?;
        let mut out = Vec::new();
        for (info, var) in param_table(&spec)?.into_iter().zip(p.into_flat()) {
            // Layer 0's vr_lambda has no gradient, as in torch.
            if let Some(g) = tape.take_grad(var) {
                out.push(HostTensor {
                    name: info.name,
                    shape: info.shape,
                    data: g.to_f32_vec()?,
                });
            }
        }
        TensorSet::new(out)
    }

    fn train(
        &mut self,
        params: &TensorSet,
        setup: &TrainSetup,
        steps: usize,
    ) -> Result<(Vec<f64>, TensorSet), OjasError> {
        let spec = ModelSpec::tiny();
        let cfg = train_config(setup)?;
        let host = params_of(&spec, params, &Budget::new(1 << 30))?;
        let bin = TokenBin::open_headerless(&setup.token_bin)
            .map_err(|e| refuse(format!("token bin: {e}")))?;
        let mut t = Trainer::new(exact(), spec, &host, bin, cfg)?;
        self.reports.clear();
        for _ in 0..steps {
            self.reports.push(t.step()?);
        }
        let losses = self.reports.iter().map(|r| f64::from(r.loss)).collect();
        let mut out = Vec::new();
        for (info, value) in t.params() {
            out.push(HostTensor {
                name: info.name.clone(),
                shape: info.shape.clone(),
                data: t.backend().download(value)?.to_f32_vec()?,
            });
        }
        Ok((losses, TensorSet::new(out)?))
    }
}

#[test]
fn oracle_forward_loss_and_logits() {
    let r = parity::forward_parity(&mut Cpu::default()).unwrap();
    eprintln!(
        "forward: loss relative error {:e} (tol {:e})",
        r.worst, r.tolerance
    );
}

#[test]
fn oracle_grads_at_init() {
    let r = parity::grads_parity(&mut Cpu::default(), ojas_oracle::golden::GradsAt::Init).unwrap();
    eprintln!(
        "grads at init: worst normwise {:e} (tol {:e})",
        r.worst, r.tolerance
    );
}

#[test]
fn oracle_grads_at_step5() {
    let r = parity::grads_parity(&mut Cpu::default(), ojas_oracle::golden::GradsAt::Step5).unwrap();
    eprintln!(
        "grads at step 5: worst normwise {:e} (tol {:e})",
        r.worst, r.tolerance
    );
}

#[test]
fn oracle_five_step_trace_with_grad_norms_and_schedule() {
    five_step_trace(Ns5::F32);
}

/// Stock nanolab (`X = G.bfloat16()` in Newton-Schulz) against ojas's bf16
/// NS5. The f32 and bf16 traces differ by up to 1.8e-4 nats over these five
/// steps, so the 1e-4 gate tells the two iterations apart.
#[test]
fn oracle_five_step_trace_bf16_ns5_matches_stock_nanolab() {
    five_step_trace(Ns5::Bf16);
}

fn five_step_trace(ns5: Ns5) {
    let mut m = Cpu::default();
    let r = parity::trace_parity_ns5(&mut m, ns5).unwrap();
    eprintln!(
        "trace {ns5:?}: worst |dloss| {:e} (tol {:e})",
        r.worst, r.tolerance
    );
    // Beyond the runner: the pre-clip grad norm and the LR multiplier of
    // every step, from the same run.
    let fx = tiny_trace(ns5).unwrap();
    for (i, report) in m.reports.iter().enumerate() {
        let want = fx.grad_norm[i];
        let rel = (f64::from(report.grad_norm) - want).abs() / want;
        eprintln!(
            "trace step {}: grad norm {} torch {want} rel {rel:e}",
            i + 1,
            report.grad_norm
        );
        assert!(
            rel <= 1e-4,
            "step {}: grad norm {} vs {want}",
            i + 1,
            report.grad_norm
        );
        let mult = fx.lr_mult[i];
        assert!(
            (report.lr_multiplier - mult).abs() <= 1e-12 * mult.abs(),
            "step {}: lr multiplier {} vs {mult}",
            i + 1,
            report.lr_multiplier
        );
    }
}

#[test]
fn oracle_forty_step_curve() {
    let r = parity::curve_parity(&mut Cpu::default()).unwrap();
    eprintln!("curve: worst |dloss| {:e} (tol {:e})", r.worst, r.tolerance);
}

#[test]
fn oracle_forty_step_curve_bf16_ns5() {
    let r = parity::curve_parity_ns5(&mut Cpu::default(), Ns5::Bf16).unwrap();
    eprintln!(
        "curve bf16 ns5: worst |dloss| {:e} (tol {:e})",
        r.worst, r.tolerance
    );
}

#[test]
fn the_exported_init_loads_with_its_spec() {
    let path = std::path::Path::new(TINY_DIR).join("init.safetensors");
    let file = SafeTensors::open(&path).unwrap();
    let budget = Budget::new(1 << 30);
    let (spec, params) = load_model(&file, &budget).unwrap();
    assert_eq!(spec, ModelSpec::tiny());
    let fixture = tiny_init().unwrap();
    for (info, t) in param_table(&spec).unwrap().iter().zip(&params) {
        let want = &fixture.params.get(&info.name).unwrap().data;
        let got = t.to_f32_vec().unwrap();
        let same = got
            .iter()
            .zip(want)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same && got.len() == want.len(), "{}", info.name);
    }
}

#[test]
fn the_names_table_matches_the_oracles_restatement_of_the_design() {
    let oracle = tiny_spec();
    let ours = ModelSpec::tiny();
    assert_eq!(
        (
            oracle.vocab,
            oracle.n_embd,
            oracle.n_layer,
            oracle.n_head,
            oracle.n_kv_head
        ),
        (
            ours.vocab,
            ours.n_embd,
            ours.n_layer,
            ours.n_head,
            ours.n_kv_head
        )
    );
    assert_eq!(
        (oracle.head_dim, oracle.hidden, oracle.max_seq),
        (ours.head_dim, ours.hidden, ours.max_seq)
    );
    assert_eq!(
        (oracle.rope_base, oracle.rms_eps),
        (ours.rope_base, ours.rms_eps)
    );
    assert_eq!(oracle.tie_embeddings, ours.tie_embeddings);
    let rows = expected_params(&oracle);
    let table = param_table(&ours).unwrap();
    assert_eq!(rows.len(), table.len());
    for row in rows {
        let ours = table.iter().find(|p| p.name == row.name).unwrap();
        assert_eq!(ours.shape, row.shape, "{}", row.name);
        let init_same = match (ours.init, row.init) {
            (Init::Normal { std }, ojas_oracle::spec::Init::Normal(s)) => {
                f64::from(std) == f64::from(s as f32)
            }
            (Init::Zeros, ojas_oracle::spec::Init::Zeros)
            | (Init::Ones, ojas_oracle::spec::Init::Ones) => true,
            _ => false,
        };
        assert!(init_same, "{}: {:?} vs {:?}", row.name, ours.init, row.init);
        let muon = ours.group == ojas_cpu::OptimGroup::MuonMatrix;
        assert_eq!(muon, row.group == Group::Muon, "{}", row.name);
    }
}
