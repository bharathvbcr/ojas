//! Reusable Rust-owned training engine used by the CLI and language bindings.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::checkpoint::{
    load_bf16_shadows, load_optim_state_python_npy, read_training_checkpoint_meta,
    save_training_checkpoint, TrainingCheckpointMeta, CHECKPOINT_VERSION,
};
use crate::init::init_weights_seeded;
use crate::model_bwd::{backward_f32_opts_clip, Grads};
use crate::model_fwd::{forward_f32_uploaded, DualInputBuffers};
#[cfg(test)]
use crate::optim::OPTIM_ATOL;
use crate::optim::{optim_step, zero_grads, ClipMode, LrSchedule, OptimHyperparams, OptimState};
use crate::optimizer_registry::OptimizerKind;
use crate::runtime::{GpuRuntime, PrecisionMode};
use crate::tape::Tape;
use crate::weights::{ModelConfig, Weights};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EngineCreateConfig {
    pub preset: String,
    pub optimizer: OptimizerKind,
    pub seed: u64,
    pub precision: PrecisionModeConfig,
    pub total_steps: usize,
    pub warmdown_steps: usize,
    pub batch: Option<usize>,
    pub seq_len: Option<usize>,
    pub clip_mode: ClipMode,
    pub hyperparams: OptimHyperparams,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionModeConfig {
    Bf16,
    F32,
}

impl Default for EngineCreateConfig {
    fn default() -> Self {
        Self {
            preset: "arch02-128m".into(),
            optimizer: OptimizerKind::MuonNs5Adamw,
            seed: 1337,
            precision: PrecisionModeConfig::Bf16,
            total_steps: 2_000,
            warmdown_steps: 350,
            batch: Some(16),
            seq_len: Some(256),
            clip_mode: ClipMode::Soft,
            hyperparams: OptimHyperparams::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EngineStepMetrics {
    pub completed_step: usize,
    pub loss: f32,
    pub grad_norm: f32,
    pub clip_factor: f32,
    pub lr_multiplier: f32,
    pub dispatches: usize,
}

pub struct TrainingEngine {
    pub rt: Arc<GpuRuntime>,
    pub weights: Weights,
    pub state: OptimState,
    pub schedule: LrSchedule,
    pub seed: u64,
    pub preset: String,
    pub data_cursor_tokens: usize,
    grads: Grads,
    tape: Tape,
    inputs: DualInputBuffers,
}

/// Refuse an id outside `0..vocab`. A negative `i32` must not be cast to a
/// huge `uint` index in a kernel, and it is not clamped into range.
pub fn reject_token_ids(ids: &[i32], vocab: usize, what: &str) -> Result<(), String> {
    if vocab == 0 {
        return Err(format!("{what}: vocabulary is empty"));
    }
    let limit = i32::try_from(vocab).unwrap_or(i32::MAX);
    for (i, &id) in ids.iter().enumerate() {
        if id < 0 || id >= limit {
            return Err(format!("{what}[{i}] = {id} is outside 0..{vocab}"));
        }
    }
    Ok(())
}

impl TrainingEngine {
    pub fn create(mut config: EngineCreateConfig) -> Result<Self, String> {
        if !config.optimizer.native_ready() {
            return Err(format!(
                "optimizer {} is not native-parity-qualified",
                config.optimizer
            ));
        }
        let rt = crate::gpu_runtime()?;
        rt.set_precision(match config.precision {
            PrecisionModeConfig::Bf16 => PrecisionMode::Bf16,
            PrecisionModeConfig::F32 => PrecisionMode::F32,
        });
        rt.set_async_encode(true)?;
        let mut model = ModelConfig::from_preset(&config.preset)?;
        if let Some(batch) = config.batch {
            model.batch = batch;
        }
        if let Some(seq_len) = config.seq_len {
            model.seq_len = seq_len;
        }
        model.validate_metal_shape()?;
        if model.model_dim == 768 && model.num_layers == 24 {
            config.hyperparams.muon_momentum_warmup = 150;
        }
        let weights = init_weights_seeded(&rt, model, config.seed)?;
        Self::assemble(
            rt,
            weights,
            config.optimizer,
            config.hyperparams,
            config.clip_mode,
            LrSchedule::from_warmdown(config.total_steps, config.warmdown_steps),
            config.seed,
            config.preset,
            0,
        )
    }

    fn assemble(
        rt: Arc<GpuRuntime>,
        mut weights: Weights,
        optimizer: OptimizerKind,
        hp: OptimHyperparams,
        clip_mode: ClipMode,
        schedule: LrSchedule,
        seed: u64,
        preset: String,
        data_cursor_tokens: usize,
    ) -> Result<Self, String> {
        if rt.precision() == PrecisionMode::Bf16 {
            weights.ensure_bf16_banks(&rt)?;
        }
        let mut state = OptimState::new_for_kind(&rt, &weights, hp, optimizer)?;
        state.clip_mode = clip_mode;
        let grads = Grads::zeros_like(&rt, &weights)?;
        let tape = Tape::new(weights.cfg.num_layers);
        let bump_bytes = if weights.cfg.model_dim >= 768 {
            512 * 1024 * 1024
        } else if weights.cfg.model_dim >= 256 {
            256 * 1024 * 1024
        } else {
            64 * 1024 * 1024
        };
        rt.ensure_bump(bump_bytes)?;
        let inputs = DualInputBuffers::new(&rt, weights.cfg.batch, weights.cfg.seq_len)?;
        Ok(Self {
            rt,
            weights,
            state,
            schedule,
            seed,
            preset,
            data_cursor_tokens,
            grads,
            tape,
            inputs,
        })
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let meta = read_training_checkpoint_meta(path)?;
        let rt = crate::gpu_runtime()?;
        rt.set_precision(if meta.bf16_precision {
            PrecisionMode::Bf16
        } else {
            PrecisionMode::F32
        });
        rt.set_async_encode(true)?;
        let weights = Weights::load_from_python_npy(&rt, &path.join("weights"), meta.config)?;
        let optimizer: OptimizerKind = meta.optimizer.parse()?;
        let mut engine = Self::assemble(
            rt,
            weights,
            optimizer,
            meta.hyperparams,
            meta.clip_mode,
            meta.schedule,
            meta.seed,
            meta.preset,
            meta.data_cursor_tokens,
        )?;
        if meta.bf16_shadows_saved {
            load_bf16_shadows(&mut engine.weights, &path.join("bf16_shadows"))?;
        }
        load_optim_state_python_npy(&mut engine.state, &path.join("optim"), &path.join("ema"))?;
        if engine.state.step != meta.step {
            return Err(format!(
                "optimizer step {} does not match metadata step {}",
                engine.state.step, meta.step
            ));
        }
        Ok(engine)
    }

    pub fn expected_tokens(&self) -> usize {
        self.weights.cfg.batch * self.weights.cfg.seq_len
    }

    fn validate_tokens(&self, input_ids: &[i32], targets: &[i32]) -> Result<(), String> {
        let expected = self.expected_tokens();
        if input_ids.len() != expected || targets.len() != expected {
            return Err(format!(
                "expected {expected} input and target tokens, got {} and {}",
                input_ids.len(),
                targets.len()
            ));
        }
        let vocab = self.weights.cfg.vocab_size;
        reject_token_ids(input_ids, vocab, "input_ids")?;
        reject_token_ids(targets, vocab, "targets")?;
        Ok(())
    }

    pub fn train_step(
        &mut self,
        input_ids: &[i32],
        targets: &[i32],
    ) -> Result<EngineStepMetrics, String> {
        self.validate_tokens(input_ids, targets)?;
        zero_grads(&self.rt, &self.grads)?;
        let (ids, tgts) = self.inputs.upload(&self.rt, input_ids, targets)?;
        let out = forward_f32_uploaded(&self.rt, &self.weights, ids, tgts, &mut self.tape)?;
        let grad_norm = backward_f32_opts_clip(
            &self.rt,
            &self.weights,
            &self.tape,
            &mut self.grads,
            true,
            Some(&self.state.clip),
            true,
        )?
        .unwrap_or(0.0);
        let lr_multiplier = self.schedule.mul_at(self.state.step);
        optim_step(
            &self.rt,
            &mut self.weights,
            &self.grads,
            &mut self.state,
            true,
            lr_multiplier,
        )?;
        if self.rt.precision() == PrecisionMode::Bf16 {
            self.weights.refresh_bf16_banks(&self.rt)?;
        }
        self.rt.synchronize()?;
        let loss = out.read_loss(&self.rt)?;
        if !loss.is_finite() || !grad_norm.is_finite() {
            return Err(format!(
                "numerical failure at optimizer step {}: loss={loss}, grad_norm={grad_norm}",
                self.state.step
            ));
        }
        self.inputs.mark_synced();
        self.rt.bump_reset()?;
        let dispatches = self.rt.take_dispatch_count();
        self.data_cursor_tokens = self
            .data_cursor_tokens
            .saturating_add(self.expected_tokens().saturating_add(1));
        let clip_factor = if grad_norm > self.state.hp.grad_clip {
            self.state.hp.grad_clip / (grad_norm + 1e-6)
        } else {
            1.0
        };
        Ok(EngineStepMetrics {
            completed_step: self.state.step,
            loss,
            grad_norm,
            clip_factor,
            lr_multiplier,
            dispatches,
        })
    }

    pub fn evaluate(&mut self, input_ids: &[i32], targets: &[i32]) -> Result<f32, String> {
        self.validate_tokens(input_ids, targets)?;
        let (ids, tgts) = self.inputs.upload(&self.rt, input_ids, targets)?;
        let out = forward_f32_uploaded(&self.rt, &self.weights, ids, tgts, &mut self.tape)?;
        self.rt.synchronize()?;
        let loss = out.read_loss(&self.rt)?;
        self.inputs.mark_synced();
        self.rt.bump_reset()?;
        let _ = self.rt.take_dispatch_count();
        Ok(loss)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let meta = TrainingCheckpointMeta {
            version: CHECKPOINT_VERSION,
            step: self.state.step,
            data_cursor_tokens: self.data_cursor_tokens,
            seed: self.seed,
            preset: self.preset.clone(),
            config: self.weights.cfg.clone(),
            parameter_count: self.weights.cfg.count_params(),
            optimizer: self.state.kind.to_string(),
            hyperparams: self.state.hp.clone(),
            clip_mode: self.state.clip_mode,
            schedule: self.schedule,
            bf16_precision: self.rt.precision() == PrecisionMode::Bf16,
            bf16_shadows_saved: self.weights.bf16_banks.is_some(),
        };
        save_training_checkpoint(&self.rt, &self.weights, &self.state, path, &meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn default_config_is_exact_champion_recipe() {
        let cfg = EngineCreateConfig::default();
        assert_eq!(cfg.preset, "arch02-128m");
        assert_eq!(cfg.total_steps, 2_000);
        assert_eq!(cfg.warmdown_steps, 350);
        assert_eq!(cfg.batch, Some(16));
        assert_eq!(cfg.seq_len, Some(256));
        assert_eq!(cfg.hyperparams.ema_decay, 0.997);
    }

    #[test]
    fn negative_and_out_of_vocab_ids_are_refused() {
        let err = super::reject_token_ids(&[-1], 1024, "input_ids").unwrap_err();
        assert!(err.contains("-1"), "{err}");
        let err = super::reject_token_ids(&[1024], 1024, "targets").unwrap_err();
        assert!(err.contains("1024"), "{err}");
        super::reject_token_ids(&[0, 1023], 1024, "input_ids").unwrap();
    }

    #[test]
    fn save_reload_replays_next_step_exactly() {
        let mut cfg = EngineCreateConfig::default();
        cfg.preset = "sota".into();
        cfg.batch = Some(16);
        cfg.seq_len = Some(64);
        cfg.total_steps = 3;
        cfg.warmdown_steps = 0;
        cfg.precision = PrecisionModeConfig::F32;
        let mut uninterrupted = TrainingEngine::create(cfg).expect("create");
        let n = uninterrupted.expected_tokens();
        let input: Vec<i32> = (0..n).map(|i| (i % 1024) as i32).collect();
        let target: Vec<i32> = (0..n).map(|i| ((i + 1) % 1024) as i32).collect();
        uninterrupted.train_step(&input, &target).expect("step one");

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("arch02-resume-{unique}"));
        uninterrupted.save(&root).expect("save");
        let expected = uninterrupted.train_step(&input, &target).expect("step two");
        let expected_qo = uninterrupted.weights.qo_bank.buffer.read_f32();

        let mut resumed = TrainingEngine::load(&root).expect("load");
        let actual = resumed
            .train_step(&input, &target)
            .expect("replayed step two");
        let actual_qo = resumed.weights.qo_bank.buffer.read_f32();
        assert_eq!(actual.completed_step, expected.completed_step);
        let loss_delta = (actual.loss - expected.loss).abs();
        let norm_delta = (actual.grad_norm - expected.grad_norm).abs();
        let weight_delta = actual_qo
            .iter()
            .zip(expected_qo.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, crate::parity::max_finite_error);
        eprintln!(
            "resume replay deltas: loss={loss_delta:e} grad_norm={norm_delta:e} qo={weight_delta:e}"
        );
        // Metal reductions may differ by a few ULP across fresh command queues;
        // the checkpoint contract is deterministic to the optimizer parity
        // tolerance, not bit-identical floating-point reduction order.
        assert!(loss_delta <= 1e-6, "loss delta {loss_delta}");
        assert!(norm_delta <= OPTIM_ATOL, "grad norm delta {norm_delta}");
        assert!(
            weight_delta <= OPTIM_ATOL,
            "master weight delta {weight_delta}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn intentional_nan_is_reported_as_numerical_failure() {
        let mut config = EngineCreateConfig::default();
        config.preset = "sota_toy".into();
        config.batch = Some(1);
        config.seq_len = Some(8);
        config.precision = PrecisionModeConfig::F32;
        let mut engine = TrainingEngine::create(config).expect("engine");
        engine.weights.qo_bank.buffer.contents_f32()[0] = f32::NAN;
        let input = vec![1i32; engine.expected_tokens()];
        let target = vec![2i32; engine.expected_tokens()];
        let error = engine.train_step(&input, &target).unwrap_err();
        assert!(error.contains("numerical failure"), "{error}");
    }
}
