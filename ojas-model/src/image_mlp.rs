//! A toy image classifier as a training loop's [`StepProvider`].
//!
//! [`ImageMlpProvider`] is `linear(W1) -> silu -> linear_cross_entropy(W2)`, no bias, over one
//! flattened image: the smallest model that trains an image through `ojas-train`'s loop on any
//! [`Backend`], so the loop, the gradient bank, AdamW and the checkpoint are exercised by a
//! vision workload before the CNN and ViT land. Everything else is shared: the tape, the
//! [`AdapterBank`], [`adamw_all_or_nothing`] and the loop's own checkpoint policy.
//!
//! # The job contract, read literally
//!
//! - `tokens` are the image's `u8` pixels in CHW order, one token per pixel; the vocabulary is
//!   the input alphabet, so [`StepProvider::vocab_size`] is 256. The pixel count must be the
//!   provider's configured `pixels`.
//! - `rows` is one row: `positions == [0]` and `targets == [label]` with `label < classes`.
//!   Position 0 is the image as a whole (a classifier has no sequence), and the logits are the
//!   fused cross-entropy of `W2 @ silu(W1 @ x)` against the label. Any other position, more
//!   than one row and an out-of-range label are refused with [`StepError::Contract`].
//! - `hidden_at` must be empty. The contract lets an outside loss read the final hidden state
//!   and hand back a gradient, but a classifier has no loss outside the model and no hidden
//!   state that a second consumer would read, so a non-empty `hidden_at` is refused with
//!   [`StepError::Unsupported`] rather than honoured through a callback nothing would pass.
//!   [`StepProvider::hidden_size`] is the width of the hidden layer.
//! - The returned loss is the unscaled cross-entropy; `scale` times its gradient goes to the
//!   bank ([`ojas_autograd::Tape::backward_seeded`]). `scale == 0` returns the value with no
//!   backward and writes zero gradients, so an `Overwrite` still replaces the previous step's.
//!
//! Pixels are normalised on the host to `(p / 255 - mean) / std` per channel, the formula of
//! `ojas_data::ImageBatchSampler`, and bound as a frozen leaf `[1, pixels]`.
//!
//! # Checkpoints
//!
//! [`StepProvider::save_state_with`] writes one safetensors file, [`STATE_FILE`]
//! (`ojas_io::write_safetensors`): per entry the values under the entry's name and both AdamW
//! moments under `<name>.adam_m` and `<name>.adam_v`, all F32, with the format, the layer sizes,
//! the step and the loop's seed, epoch and consumed digest as metadata.
//! [`StepProvider::load_state_with`] reads it with `ojas_io::SafeTensors` and refuses a file
//! with another format, shape, extra or missing tensor, a step, seed, epoch or digest that is
//! not the loop's, and a non-finite value, and changes nothing when it refuses.
//!
//! [`ClassificationObjective`] is the loop's [`Objective`] for it: a batch of `B` images plans
//! `B` single-image sequences at row scale `1 / B`, so the bank holds the batch-mean gradient.
//! [`EpochBatches`] yields an epoch's batches in `ojas-data`'s sampler order.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use ojas_autograd::Tape;
use ojas_core::{AdamWConfig, Backend, CeChunk, DType, OjasError, Tensor};
use ojas_data::{Augment, ImageBatchSampler, ImageSamplerConfig, ImageSet, Normalization};
use ojas_io::{encode_f32_as, write_safetensors, IoError, SafeTensors, StDtype, TensorOut};
use ojas_train::lappi::run_control::ConsumedPrefix;
use ojas_train::lappi::step::{
    check_per_entry, AdamWHyper, BankMode, CheckpointContext, HiddenGrad, ParamSpec, SequenceJob,
    StepError, StepProvider,
};
use ojas_train::lappi::trainer::{
    BatchCounts, ConsumedBatch, HostParams, MicroBatchPlan, MicroLoss, Objective, SequenceWork,
    TrainError,
};

use crate::init::init_param;
use crate::names::Init;
use crate::qwen35::bank::{AdapterBank, BankEntry};
use crate::qwen35::optim::{adamw_all_or_nothing, AdamWEntry, OptimError};
use crate::qwen35::step_provider::{bank_err, core_err};
use crate::DEFAULT_CE_CHUNK;

/// The file [`StepProvider::save_state_with`] writes under the provider directory.
pub const STATE_FILE: &str = "image_mlp.safetensors";

/// One entry's AdamW moments on the host: `(first, second)`.
pub type HostMoments = (Vec<f32>, Vec<f32>);

/// The checkpoint's metadata `format` and `version`.
const FORMAT: &str = "ojas-image-mlp";
const VERSION: &str = "1";

/// The input alphabet: one `u8` pixel is one token.
const PIXEL_VOCAB: usize = 256;

/// The shape, normalisation and init seed of one classifier.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageMlpSpec {
    /// Pixels per image, `channels * height * width`.
    pub pixels: usize,
    pub hidden: usize,
    pub classes: usize,
    /// Per-channel mean and standard deviation, as [`Normalization::new`] takes them. The
    /// channel count divides `pixels`.
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    /// The init seed: each weight draws from its own stream, so a weight depends on this and
    /// its name only.
    pub seed: u64,
}

impl ImageMlpSpec {
    fn validate(&self) -> Result<(), StepError> {
        if self.pixels == 0 || self.hidden == 0 || self.classes == 0 {
            return Err(contract(format!(
                "a classifier needs pixels, hidden and classes above 0, got {}, {}, {}",
                self.pixels, self.hidden, self.classes
            )));
        }
        if self.classes > PIXEL_VOCAB {
            return Err(contract(format!(
                "{} classes: a row target is bounded by the job contract's vocabulary of \
                 {PIXEL_VOCAB} (the input alphabet), so at most {PIXEL_VOCAB} labels can be sent",
                self.classes
            )));
        }
        Normalization::new(&self.mean, &self.std)
            .map_err(|e| contract(format!("normalization: {e}")))?;
        if !self.pixels.is_multiple_of(self.mean.len()) {
            return Err(contract(format!(
                "{} pixels do not split into {} channels",
                self.pixels,
                self.mean.len()
            )));
        }
        if self.lut().iter().flatten().any(|v| !v.is_finite()) {
            return Err(contract(
                "normalization: (p / 255 - mean) / std overflows f32 for some pixel value",
            ));
        }
        Ok(())
    }

    /// Bytes of both weights and both AdamW moments, f32, or a refusal if the count overflows.
    fn state_bytes(&self) -> Result<u64, StepError> {
        let weights = |a: usize, b: usize| a.checked_mul(b);
        weights(self.hidden, self.pixels)
            .zip(weights(self.classes, self.hidden))
            .and_then(|(w1, w2)| w1.checked_add(w2))
            .and_then(|n| u64::try_from(n).ok())
            .and_then(|n| n.checked_mul(3 * 4))
            .ok_or_else(|| {
                contract(format!(
                    "{} x {} x {} classifier: the parameter bytes overflow u64",
                    self.pixels, self.hidden, self.classes
                ))
            })
    }

    /// `lut[c][p] = (p / 255 - mean[c]) / std[c]`, the sampler's table.
    fn lut(&self) -> Vec<[f32; PIXEL_VOCAB]> {
        self.mean
            .iter()
            .zip(&self.std)
            .map(|(&mean, &std)| {
                let mut table = [0.0f32; PIXEL_VOCAB];
                for (p, v) in table.iter_mut().enumerate() {
                    *v = (p as f32 / 255.0 - mean) / std;
                }
                table
            })
            .collect()
    }
}

fn contract(detail: impl Into<String>) -> StepError {
    StepError::Contract(detail.into())
}

fn io_err(e: IoError) -> StepError {
    StepError::Io(e.to_string())
}

/// The loss tensor's first element, after a synchronisation so a deferred device fault surfaces
/// here rather than as a silent NaN.
fn scalar<K: Backend>(backend: &K, t: &Tensor) -> Result<f32, OjasError> {
    backend.sync()?;
    let loss = backend
        .download(t)?
        .to_f32_vec()?
        .first()
        .copied()
        .ok_or_else(|| OjasError::OutOfRange {
            op: "ImageMlpProvider",
            detail: "the loss tensor is empty".into(),
        })?;
    if loss.is_finite() {
        Ok(loss)
    } else {
        Err(OjasError::NonFinite {
            op: "ImageMlpProvider loss",
        })
    }
}

/// An image classifier driven one image per [`StepProvider::accumulate`] call. See the module
/// docs.
pub struct ImageMlpProvider<B: Backend> {
    backend: B,
    spec: ImageMlpSpec,
    lut: Vec<[f32; PIXEL_VOCAB]>,
    table: Vec<ParamSpec>,
    /// `W1 [hidden, pixels]`, then `W2 [classes, hidden]`.
    leaves: Vec<Tensor>,
    moments: Vec<(Tensor, Tensor)>,
    bank: AdapterBank,
    step: u64,
    chunk: CeChunk,
}

impl<B: Backend> ImageMlpProvider<B> {
    /// A classifier with seeded normal init (`std = 1 / sqrt(fan_in)`) and zero moments.
    pub fn open(backend: B, spec: ImageMlpSpec) -> Result<Self, StepError> {
        spec.validate()?;
        // Charge nothing, but refuse before the host vectors are filled if they cannot fit.
        backend
            .budget()
            .check_room(spec.state_bytes()?)
            .map_err(core_err)?;
        let table = vec![
            ParamSpec::new("classifier.w1.weight", &[spec.hidden, spec.pixels]),
            ParamSpec::new("classifier.w2.weight", &[spec.classes, spec.hidden]),
        ];
        let mut leaves = Vec::with_capacity(table.len());
        let mut moments = Vec::with_capacity(table.len());
        for (p, fan_in) in table.iter().zip([spec.pixels, spec.hidden]) {
            let init = Init::Normal {
                std: (1.0 / (fan_in as f64).sqrt()) as f32,
            };
            let values = init_param(&p.name, &p.shape, init, spec.seed).map_err(core_err)?;
            leaves.push(upload(&backend, &values, &p.shape)?);
            let zeros = vec![0.0f32; values.len()];
            moments.push((
                upload(&backend, &zeros, &p.shape)?,
                upload(&backend, &zeros, &p.shape)?,
            ));
        }
        let bank = AdapterBank::new(
            table
                .iter()
                .map(|p| BankEntry {
                    name: p.name.clone(),
                    shape: p.shape.clone(),
                })
                .collect(),
        )
        .map_err(bank_err)?;
        Ok(Self {
            backend,
            lut: spec.lut(),
            spec,
            table,
            leaves,
            moments,
            bank,
            step: 0,
            chunk: DEFAULT_CE_CHUNK,
        })
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Both AdamW moments of every entry, in table order: `(first, second)`.
    pub fn read_moments(&self) -> Result<Vec<HostMoments>, StepError> {
        self.moments
            .iter()
            .map(|(m1, m2)| Ok((self.host(m1)?, self.host(m2)?)))
            .collect()
    }

    fn host(&self, t: &Tensor) -> Result<Vec<f32>, StepError> {
        self.backend
            .download(t)
            .and_then(|h| h.to_f32_vec())
            .map_err(core_err)
    }

    /// Everything about `job` the provider refuses, before any tensor is made.
    fn check(&self, job: &SequenceJob<'_>) -> Result<u32, StepError> {
        job.validate(PIXEL_VOCAB)?;
        if !job.hidden_at.is_empty() {
            return Err(StepError::Unsupported(
                "an image classifier has no loss outside the model: hidden_at must be empty".into(),
            ));
        }
        if job.tokens.len() != self.spec.pixels {
            return Err(contract(format!(
                "{} pixels for a classifier of {}",
                job.tokens.len(),
                self.spec.pixels
            )));
        }
        let rows = &job.rows;
        // The tape seeds the backward with the scale and refuses a subnormal one (it would
        // flush gradients toward zero); refuse it here, typed, before any tensor is made.
        if rows.scale != 0.0 && !rows.scale.is_normal() {
            return Err(contract(format!(
                "row scale {} must be 0 or a positive normal f32",
                rows.scale
            )));
        }
        if rows.positions.len() != 1 {
            return Err(contract(format!(
                "an image has exactly one row, got {}",
                rows.positions.len()
            )));
        }
        if rows.positions[0] != 0 {
            return Err(contract(format!(
                "the row position must be 0 (the image as a whole), got {}",
                rows.positions[0]
            )));
        }
        let label = rows.targets[0];
        if label as usize >= self.spec.classes {
            return Err(contract(format!(
                "label {label} >= {} classes",
                self.spec.classes
            )));
        }
        Ok(label)
    }

    /// The normalised pixels, CHW, as the frozen input `[1, pixels]`.
    fn input(&self, tokens: &[u32]) -> Result<Tensor, OjasError> {
        let plane = self.spec.pixels / self.lut.len();
        let x: Vec<f32> = tokens
            .iter()
            .enumerate()
            .map(|(i, &p)| self.lut[i / plane][p as usize])
            .collect();
        Tensor::from_f32(&x, &[1, self.spec.pixels], self.backend.budget())
    }

    /// One image forward and backward on a fresh tape: the unscaled loss and, for a positive
    /// `scale`, the gradient of `W1` and `W2` times it.
    fn run(&self, tokens: &[u32], label: u32, scale: f32) -> Result<(f64, Vec<Tensor>), OjasError> {
        let budget = self.backend.budget().clone();
        let mut tape = Tape::new(&self.backend);
        let x = tape.frozen_leaf(self.input(tokens)?)?;
        let w1 = tape.leaf(self.leaves[0].clone())?;
        let w2 = tape.leaf(self.leaves[1].clone())?;
        let hidden = tape.linear(x, w1)?;
        let act = tape.silu(hidden)?;
        let target = Tensor::from_u32(&[label], &[1], &budget)?;
        let (value, grads) = if scale > 0.0 {
            let loss = tape.linear_cross_entropy(act, w2, target, None, self.chunk)?;
            let value = scalar(&self.backend, tape.value(loss)?)?;
            tape.backward_seeded(loss, scale)?;
            let mut grads = Vec::with_capacity(2);
            for (v, name) in [(w1, "W1"), (w2, "W2")] {
                grads.push(tape.take_grad(v).ok_or_else(|| OjasError::OutOfRange {
                    op: "ImageMlpProvider",
                    detail: format!("{name} has no gradient after the backward"),
                })?);
            }
            (value, grads)
        } else {
            let loss = tape.linear_cross_entropy_value(act, w2, &target, None, self.chunk)?;
            let value = scalar(&self.backend, &loss)?;
            let grads = self
                .leaves
                .iter()
                .map(|leaf| {
                    let zeros = Tensor::zeros(leaf.shape(), DType::F32, &budget)?;
                    self.backend.upload(&zeros)
                })
                .collect::<Result<Vec<_>, _>>()?;
            (value, grads)
        };
        self.backend.sync()?;
        drop(tape);
        Ok((f64::from(value), grads))
    }

    fn accumulate_into_bank(
        &mut self,
        job: &SequenceJob<'_>,
        mode: BankMode,
    ) -> Result<f64, StepError> {
        let label = self.check(job)?;
        let (value, grads) = self
            .run(job.tokens, label, job.rows.scale)
            .map_err(core_err)?;
        for (i, grad) in grads.into_iter().enumerate() {
            if let Err(e) = self.bank.accumulate(&self.backend, i, mode, grad) {
                // A half-written bank must not look complete.
                self.bank.clear();
                return Err(bank_err(e));
            }
        }
        Ok(value)
    }

    /// The checkpoint's metadata for the state at the loop's `context`.
    fn metadata(&self, context: &CheckpointContext) -> Vec<(&'static str, String)> {
        vec![
            ("format", FORMAT.to_owned()),
            ("version", VERSION.to_owned()),
            ("pixels", self.spec.pixels.to_string()),
            ("hidden", self.spec.hidden.to_string()),
            ("classes", self.spec.classes.to_string()),
            ("step", context.optimizer_step.to_string()),
            ("seed", context.seed.to_string()),
            ("epoch", context.epoch.to_string()),
            ("consumed_digest", context.consumed_digest.clone()),
        ]
    }

    fn device_tensor(&self, values: &[f32], shape: &[usize]) -> Result<Tensor, StepError> {
        upload(&self.backend, values, shape)
    }
}

fn upload<B: Backend>(backend: &B, values: &[f32], shape: &[usize]) -> Result<Tensor, StepError> {
    let host = Tensor::from_f32(values, shape, backend.budget()).map_err(core_err)?;
    backend.upload(&host).map_err(core_err)
}

impl<B: Backend> StepProvider for ImageMlpProvider<B> {
    fn parameters(&self) -> &[ParamSpec] {
        &self.table
    }

    fn hidden_size(&self) -> usize {
        self.spec.hidden
    }

    fn vocab_size(&self) -> usize {
        PIXEL_VOCAB
    }

    fn accumulate(
        &mut self,
        job: &SequenceJob<'_>,
        mode: BankMode,
        _external: Option<&mut HiddenGrad<'_>>,
    ) -> Result<f64, StepError> {
        let result = self.accumulate_into_bank(job, mode);
        if result.is_err() && mode == BankMode::Overwrite {
            // A refused first sequence of a step must not leave the previous sum for the
            // following `Add`s to extend.
            self.bank.clear();
        }
        result
    }

    fn grad_sq_norm(&mut self) -> Result<f64, StepError> {
        self.bank.sq_norm(&self.backend).map_err(bank_err)
    }

    fn supports_lr_scale(&self) -> bool {
        true
    }

    fn adamw_step(
        &mut self,
        hyper: &AdamWHyper,
        step: u64,
        lr_scale: &[f64],
        weight_decay: &[f64],
    ) -> Result<(), StepError> {
        hyper.validate()?;
        check_per_entry(self.leaves.len(), lr_scale, weight_decay)?;
        if step != self.step + 1 {
            return Err(contract(format!(
                "AdamW step {step} asked of a provider that has taken {} steps; the next is {}",
                self.step,
                self.step + 1
            )));
        }
        let mut grads = self.bank.take_grads().map_err(bank_err)?;
        if hyper.grad_scale != 1.0 {
            for g in &mut grads {
                self.backend
                    .scale_grad(g, hyper.grad_scale as f32)
                    .map_err(core_err)?;
            }
        }
        let mut batch: Vec<AdamWEntry<'_>> = self
            .leaves
            .iter_mut()
            .zip(self.moments.iter_mut())
            .zip(&grads)
            .zip(&self.table)
            .enumerate()
            .map(|(i, (((param, (m1, m2)), grad), spec))| AdamWEntry {
                name: spec.name.as_str(),
                param,
                grad,
                moment1: m1,
                moment2: m2,
                config: AdamWConfig {
                    lr: hyper.lr * lr_scale[i],
                    beta1: hyper.beta1,
                    beta2: hyper.beta2,
                    eps: hyper.eps,
                    weight_decay: weight_decay[i],
                },
            })
            .collect();
        adamw_all_or_nothing(&self.backend, &mut batch, self.step, self.backend.budget())
            .map_err(|e: OptimError| StepError::Backend(e.to_string()))?;
        self.step += 1;
        Ok(())
    }

    fn step_count(&self) -> u64 {
        self.step
    }

    fn read_parameters(&mut self) -> Result<Vec<Vec<f32>>, StepError> {
        self.leaves.iter().map(|t| self.host(t)).collect()
    }

    fn describe(&self) -> String {
        format!(
            "ojas image MLP {} -> {} (silu) -> {} classes, no bias, fused cross-entropy, on the \
             {:?} backend",
            self.spec.pixels,
            self.spec.hidden,
            self.spec.classes,
            self.backend.id()
        )
    }

    fn save_state(&mut self, _dir: &Path) -> Result<Vec<PathBuf>, StepError> {
        Err(StepError::Unsupported(
            "the image MLP's checkpoint records the loop's seed, epoch and consumed digest: a \
             CheckpointContext is required, use save_state_with"
                .into(),
        ))
    }

    fn load_state(&mut self, _dir: &Path) -> Result<(), StepError> {
        Err(StepError::Unsupported(
            "the image MLP's resume checks the loop's seed, epoch and consumed digest: a \
             CheckpointContext is required, use load_state_with"
                .into(),
        ))
    }

    fn save_state_with(
        &mut self,
        dir: &Path,
        context: &CheckpointContext,
    ) -> Result<Vec<PathBuf>, StepError> {
        if self.step != context.optimizer_step {
            return Err(contract(format!(
                "the provider is at step {} but the checkpoint context says {}",
                self.step, context.optimizer_step
            )));
        }
        let mut names = Vec::with_capacity(3 * self.leaves.len());
        let mut data = Vec::with_capacity(3 * self.leaves.len());
        let mut shapes = Vec::with_capacity(self.leaves.len());
        let entries = self.table.iter().zip(self.leaves.iter().zip(&self.moments));
        for (spec, (leaf, (m1, m2))) in entries {
            shapes.push(spec.shape.iter().map(|&d| d as u64).collect::<Vec<u64>>());
            for (suffix, t) in [("", leaf), (".adam_m", m1), (".adam_v", m2)] {
                names.push(format!("{}{suffix}", spec.name));
                data.push(encode_f32_as(StDtype::F32, &self.host(t)?).map_err(io_err)?);
            }
        }
        let items: Vec<TensorOut<'_>> = names
            .iter()
            .zip(&data)
            .enumerate()
            .map(|(i, (name, bytes))| TensorOut {
                name,
                dtype: StDtype::F32,
                shape: &shapes[i / 3],
                data: bytes,
            })
            .collect();
        let meta = self.metadata(context);
        let meta: Vec<(&str, &str)> = meta.iter().map(|(k, v)| (*k, v.as_str())).collect();
        let path = dir.join(STATE_FILE);
        write_safetensors(&path, &items, &meta).map_err(io_err)?;
        Ok(vec![path])
    }

    fn load_state_with(
        &mut self,
        dir: &Path,
        context: &CheckpointContext,
    ) -> Result<(), StepError> {
        let file = SafeTensors::open(&dir.join(STATE_FILE)).map_err(io_err)?;
        let want = self.metadata(context);
        let got = file.metadata();
        if got.len() != want.len() {
            return Err(contract(format!(
                "the checkpoint has {} metadata keys, expected {}",
                got.len(),
                want.len()
            )));
        }
        for (key, value) in &want {
            match got.get(*key) {
                Some(v) if v == value => {}
                other => {
                    return Err(contract(format!(
                        "the checkpoint's {key} is {other:?}, this classifier and the loop \
                         expect {value:?}"
                    )))
                }
            }
        }
        let expected: BTreeSet<String> = self
            .table
            .iter()
            .flat_map(|p| {
                [
                    p.name.clone(),
                    format!("{}.adam_m", p.name),
                    format!("{}.adam_v", p.name),
                ]
            })
            .collect();
        let found: BTreeSet<String> = file.names().map(str::to_owned).collect();
        if found != expected {
            return Err(contract(format!(
                "the checkpoint holds tensors {found:?}, expected {expected:?}"
            )));
        }
        let read = |name: &str, shape: &[usize]| -> Result<Tensor, StepError> {
            let (got, values) = file.read_f32(name).map_err(io_err)?;
            let want: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
            if got != want {
                return Err(contract(format!("{name} is {got:?}, expected {want:?}")));
            }
            if values.iter().any(|x| !x.is_finite()) {
                return Err(contract(format!("{name} holds a non-finite value")));
            }
            self.device_tensor(&values, shape)
        };
        let mut leaves = Vec::with_capacity(self.leaves.len());
        let mut moments = Vec::with_capacity(self.leaves.len());
        for spec in &self.table {
            leaves.push(read(&spec.name, &spec.shape)?);
            moments.push((
                read(&format!("{}.adam_m", spec.name), &spec.shape)?,
                read(&format!("{}.adam_v", spec.name), &spec.shape)?,
            ));
        }
        self.leaves = leaves;
        self.moments = moments;
        self.step = context.optimizer_step;
        self.bank.clear();
        Ok(())
    }
}

/// One image of a batch: its index in the dataset, its pixels and its label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageItem {
    pub index: u32,
    pub pixels: Vec<u8>,
    pub label: u8,
}

/// One micro-batch of images, as the loop accounts for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageBatch {
    /// The batch's position in its epoch's order.
    pub index: u64,
    pub items: Vec<ImageItem>,
}

impl ConsumedBatch for ImageBatch {
    fn index(&self) -> u64 {
        self.index
    }

    /// One fold of `[index (8 bytes BE), each image's dataset index (u32 LE)]`.
    fn fold_into(&self, prefix: &mut ConsumedPrefix) -> Result<(), TrainError> {
        let indices: Vec<u8> = self
            .items
            .iter()
            .flat_map(|item| item.index.to_le_bytes())
            .collect();
        prefix.fold(&[&self.index.to_be_bytes(), &indices]);
        Ok(())
    }
}

/// The loop's objective for [`ImageMlpProvider`]: the batch-mean cross-entropy of its images.
#[derive(Debug, Default)]
pub struct ClassificationObjective {
    scale: f32,
    /// The image count of the batch last planned and not yet closed.
    planned: Option<usize>,
}

impl ClassificationObjective {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Objective for ClassificationObjective {
    type Batch = ImageBatch;

    fn run_kind(&self) -> &'static str {
        "classify"
    }

    /// `B` images become `B` sequences at row scale `1 / B`, so the per-image gradients sum to
    /// the mean.
    fn plan(&mut self, batch: &ImageBatch) -> Result<MicroBatchPlan, TrainError> {
        if batch.items.is_empty() {
            return Err(TrainError::Refused(format!(
                "batch {} holds no image",
                batch.index
            )));
        }
        self.scale = 1.0 / batch.items.len() as f32;
        self.planned = Some(batch.items.len());
        let sequences: Vec<SequenceWork> = batch
            .items
            .iter()
            .map(|item| SequenceWork {
                tokens: item.pixels.iter().map(|&p| u32::from(p)).collect(),
                positions: vec![0],
                targets: vec![u32::from(item.label)],
                hidden_at: Vec::new(),
            })
            .collect();
        let supervised = sequences.len() as u64;
        let positions = sequences.iter().map(|s| s.tokens.len() as u64).sum();
        Ok(MicroBatchPlan {
            sequences,
            row_scale: self.scale,
            counts: BatchCounts {
                supervised_tokens: supervised,
                span_rows: 0,
                padded_positions: 0,
                total_positions: positions,
            },
        })
    }

    fn hidden_grad(
        &mut self,
        _seq: usize,
        _positions: &[u32],
        _hidden: &[f32],
        _width: usize,
    ) -> Result<Vec<f32>, TrainError> {
        Err(TrainError::Refused(
            "a classification objective has no outside loss".into(),
        ))
    }

    fn close(&mut self, row_loss_sums: &[f64]) -> Result<MicroLoss, TrainError> {
        let Some(planned) = self.planned.take() else {
            return Err(TrainError::Refused(
                "close without a planned batch to close".into(),
            ));
        };
        if row_loss_sums.len() != planned {
            return Err(TrainError::Refused(format!(
                "{} row-loss sums for a planned batch of {planned} images",
                row_loss_sums.len()
            )));
        }
        Ok(MicroLoss {
            total: row_loss_sums.iter().sum::<f64>() * f64::from(self.scale),
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

/// An epoch's batches in [`ImageBatchSampler`] order, each image copied out as `u8` pixels.
pub struct EpochBatches<'a> {
    set: &'a ImageSet,
    sampler: ImageBatchSampler<'a>,
    epoch: u64,
    batch: usize,
    next: usize,
    batches: usize,
}

impl<'a> EpochBatches<'a> {
    /// `epoch`'s batches under `cfg`'s seed, batch size and `drop_last`. Augmentation is
    /// refused: the classifier's tokens are the stored pixels.
    pub fn new(set: &'a ImageSet, cfg: ImageSamplerConfig, epoch: u64) -> Result<Self, TrainError> {
        if cfg.augment != Augment::NONE {
            return Err(TrainError::Refused(
                "the classifier's tokens are the stored u8 pixels: augmentation is not supported"
                    .into(),
            ));
        }
        if cfg.batch == 0 {
            return Err(TrainError::Refused("batch must be non-zero".into()));
        }
        let batches = if cfg.drop_last {
            set.len() / cfg.batch
        } else {
            set.len().div_ceil(cfg.batch)
        };
        let sampler = ImageBatchSampler::new(set, cfg)
            .map_err(|e| TrainError::Refused(format!("image sampler: {e}")))?;
        Ok(Self {
            set,
            sampler,
            epoch,
            batch: cfg.batch,
            next: 0,
            batches,
        })
    }
}

impl Iterator for EpochBatches<'_> {
    type Item = Result<ImageBatch, TrainError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.batches {
            return None;
        }
        let start = self.next * self.batch;
        let end = (start + self.batch).min(self.set.len());
        let mut items = Vec::with_capacity(end - start);
        for ordinal in start..end {
            let Some(index) = self.sampler.index_at(self.epoch, ordinal) else {
                return Some(Err(TrainError::Refused(format!(
                    "the sampler has no image at ordinal {ordinal} of epoch {}",
                    self.epoch
                ))));
            };
            let Ok(index32) = u32::try_from(index) else {
                return Some(Err(TrainError::Refused(format!(
                    "image index {index} does not fit u32"
                ))));
            };
            items.push(ImageItem {
                index: index32,
                pixels: self.set.image(index).to_vec(),
                label: self.set.label(index),
            });
        }
        let batch = ImageBatch {
            index: self.next as u64,
            items,
        };
        self.next += 1;
        Some(Ok(batch))
    }
}
