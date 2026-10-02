//! One-layer reference train step on the CPU tape.
//!
//! The fixture is d=16, T=4, vocab=32, one head, SwiGLU hidden 64
//! (`int(2/3 * 4 * 16)` rounded up to a multiple of 64). It is not a 124M
//! model. Embeddings stay on AdamW. Matrices go to Muon. Vectors, including
//! `vr_lambda`, stay on AdamW with weight decay 0.
//!
//! A single layer has no earlier value to mix. The blend is the current value
//! with itself, which is the identity for every lambda, so the lambda gradient
//! is exactly 0. The fixture initialises that scalar at 1 so a mistaken Muon
//! assignment (decay 0.1) would move it, and AdamW with a zero gradient leaves
//! the bits.

use ojas_core::{Budget, OjasError, Tensor, RMS_NORM_EPS};
use ojas_cpu::{
    clip_grads, mean_micrograds, optim_group, CosineSchedule, CpuBackend, GradAccumulator,
    HybridOptimizer, HybridParam, OptimGroup,
};

use crate::tape::{Tape, Var};

/// Fixed batch of token ids and next-token targets.
#[derive(Clone, Debug)]
pub struct TokenBatch {
    pub ids: Vec<u32>,
    pub targets: Vec<u32>,
    pub batch: usize,
    pub time: usize,
}

impl TokenBatch {
    pub fn try_new(
        ids: Vec<u32>,
        targets: Vec<u32>,
        batch: usize,
        time: usize,
    ) -> Result<Self, OjasError> {
        let n = batch
            .checked_mul(time)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "tiny_train",
                detail: "batch length overflows".to_string(),
            })?;
        if batch == 0 || time == 0 || ids.len() != n || targets.len() != n {
            return Err(OjasError::Shape {
                op: "tiny_train",
                detail: format!("batch {batch} time {time} does not match ids {}", ids.len()),
            });
        }
        Ok(Self {
            ids,
            targets,
            batch,
            time,
        })
    }
}

/// Seeded one-layer trainer.
pub struct TinyTrain {
    cpu: CpuBackend,
    schedule: CosineSchedule,
    grad_clip: f32,
    d_model: usize,
    vocab: usize,
    time: usize,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    names: Vec<&'static str>,
    opt: HybridOptimizer,
}

impl TinyTrain {
    /// Nanolab peak LRs (matrix 0.025, AdamW `6e-4`), cosine warmup 4 of 20,
    /// grad clip 1. Weight init scale is 0.02 except norms (1), the gate (0),
    /// and `vr_lambda` (1).
    pub fn new(seed: u64) -> Result<Self, OjasError> {
        const D: usize = 16;
        const T: usize = 4;
        const V: usize = 32;
        const HIDDEN: usize = 64;
        let cpu = CpuBackend::new(Budget::new(16 << 20));
        let schedule = CosineSchedule::new(4, 20)?;
        let (rope_cos, rope_sin) = rope_cache(T, D);
        let mut rng = SplitMix64(seed);
        let specs: &[(&str, &[usize], bool, Init)] = &[
            ("tok_emb", &[V, D], true, Init::Normal),
            ("norm1", &[D], false, Init::Ones),
            ("wq", &[D, D], false, Init::Normal),
            ("wk", &[D, D], false, Init::Normal),
            ("wv", &[D, D], false, Init::Normal),
            ("q_norm", &[D], false, Init::Ones),
            ("k_norm", &[D], false, Init::Ones),
            ("gate_w", &[1, D], false, Init::Zeros),
            ("gate_b", &[1], false, Init::Zeros),
            ("vr_lambda", &[1], false, Init::Ones),
            ("wo", &[D, D], false, Init::Normal),
            ("norm2", &[D], false, Init::Ones),
            ("w_gate", &[HIDDEN, D], false, Init::Normal),
            ("w_up", &[HIDDEN, D], false, Init::Normal),
            ("w_down", &[D, HIDDEN], false, Init::Normal),
            ("norm_f", &[D], false, Init::Ones),
        ];
        let mut names = Vec::with_capacity(specs.len());
        let mut params = Vec::with_capacity(specs.len());
        for (name, shape, embedding, init) in specs {
            let values = init.values(shape, &mut rng);
            let group = optim_group(shape.len(), *embedding);
            let lr = match group {
                OptimGroup::MuonMatrix => 0.025,
                OptimGroup::AdamEmbedding | OptimGroup::AdamVector => 6e-4,
            };
            names.push(*name);
            params.push(HybridParam::new(group, lr, shape, values)?);
        }
        Ok(Self {
            cpu,
            schedule,
            grad_clip: 1.0,
            d_model: D,
            vocab: V,
            time: T,
            rope_cos,
            rope_sin,
            names,
            opt: HybridOptimizer::new(params)?,
        })
    }

    pub fn step_index(&self) -> u64 {
        self.opt.step_index
    }

    pub fn names(&self) -> &[&'static str] {
        &self.names
    }

    pub fn group(&self, name: &str) -> Result<OptimGroup, OjasError> {
        Ok(self.slot(name)?.group)
    }

    pub fn param(&self, name: &str) -> Result<&[f32], OjasError> {
        Ok(&self.slot(name)?.param)
    }

    pub fn param_count(&self) -> usize {
        self.opt.params.iter().map(|param| param.param.len()).sum()
    }

    /// Mean cross-entropy of one batch, and one gradient per parameter.
    /// Does not step.
    pub fn backward(&self, batch: &TokenBatch) -> Result<(f32, Vec<Vec<f32>>), OjasError> {
        self.check_batch(batch)?;
        let mut tape = Tape::new(self.cpu.clone());
        let mut leaves = Vec::with_capacity(self.opt.params.len());
        for param in &self.opt.params {
            let tensor = Tensor::from_f32(&param.param, &param.shape, self.cpu.budget())?;
            leaves.push(tape.leaf(tensor)?);
        }
        let ids = Tensor::from_u32(&batch.ids, &[batch.batch, batch.time], self.cpu.budget())?;
        let targets = Tensor::from_u32(
            &batch.targets,
            &[batch.batch, batch.time],
            self.cpu.budget(),
        )?;
        let cos = Tensor::from_f32(
            &self.rope_cos,
            &[self.time, self.d_model],
            self.cpu.budget(),
        )?;
        let sin = Tensor::from_f32(
            &self.rope_sin,
            &[self.time, self.d_model],
            self.cpu.budget(),
        )?;
        let loss = forward_block(
            &mut tape,
            &leaves,
            ids,
            targets,
            cos,
            sin,
            batch.batch,
            self.time,
            self.d_model,
        )?;
        tape.backward(loss)?;
        let loss_v = tape.value(loss)?.to_f32_vec()?;
        if loss_v.len() != 1 || !loss_v[0].is_finite() {
            return Err(OjasError::NonFinite { op: "tiny_train" });
        }
        let mut grads = Vec::with_capacity(leaves.len());
        for (index, leaf) in leaves.iter().enumerate() {
            let width = self.opt.params[index].param.len();
            match tape.grad(*leaf) {
                Some(grad) => {
                    let data = grad.to_f32_vec()?;
                    if data.len() != width || data.iter().any(|value| !value.is_finite()) {
                        return Err(OjasError::NonFinite { op: "tiny_train" });
                    }
                    grads.push(data);
                }
                None => grads.push(vec![0.0; width]),
            }
        }
        Ok((loss_v[0], grads))
    }

    /// Sum micro-batch gradients, divide by K once, clip, then one hybrid step.
    ///
    /// The returned loss is the `f32` mean of the micro-batch mean losses, in
    /// the same order. `batches` empty is accumulation count 0 and does not step.
    pub fn step(&mut self, batches: &[TokenBatch]) -> Result<f32, OjasError> {
        if batches.is_empty() {
            return Err(OjasError::OutOfRange {
                op: "grad_accum",
                detail: "accumulation count 0".to_string(),
            });
        }
        let multiplier = self.schedule.multiplier(self.opt.step_index)?;
        let mut accums = Vec::with_capacity(self.opt.params.len());
        for param in &self.opt.params {
            accums.push(GradAccumulator::new(param.param.len())?);
        }
        let mut loss_sum = 0.0f32;
        for batch in batches {
            let (loss, grads) = self.backward(batch)?;
            loss_sum += loss;
            if !loss_sum.is_finite() {
                return Err(OjasError::NonFinite { op: "tiny_train" });
            }
            for (acc, grad) in accums.iter_mut().zip(grads.iter()) {
                acc.add(grad)?;
            }
        }
        let k = batches.len() as f32;
        let mean_loss = loss_sum / k;
        if !mean_loss.is_finite() {
            return Err(OjasError::NonFinite { op: "tiny_train" });
        }
        let mut reduced = Vec::with_capacity(accums.len());
        for acc in &accums {
            reduced.push(acc.mean()?);
        }
        clip_grads(&mut reduced, self.grad_clip)?;
        for (param, grad) in self.opt.params.iter_mut().zip(reduced) {
            param.grad = grad;
        }
        self.opt.step(multiplier)?;
        Ok(mean_loss)
    }

    fn check_batch(&self, batch: &TokenBatch) -> Result<(), OjasError> {
        if batch.time != self.time {
            return Err(OjasError::Shape {
                op: "tiny_train",
                detail: format!("time {} != {}", batch.time, self.time),
            });
        }
        if batch.ids.iter().any(|id| *id as usize >= self.vocab) {
            return Err(OjasError::OutOfRange {
                op: "tiny_train",
                detail: "token id is outside vocab".to_string(),
            });
        }
        Ok(())
    }

    fn slot(&self, name: &str) -> Result<&HybridParam, OjasError> {
        self.names
            .iter()
            .position(|candidate| *candidate == name)
            .map(|index| &self.opt.params[index])
            .ok_or_else(|| OjasError::OutOfRange {
                op: "tiny_train",
                detail: format!("no parameter {name}"),
            })
    }
}

/// Same reduction the step uses: sum in order, then divide by K once.
pub fn reduce_micrograds(parts: &[&[f32]]) -> Result<Vec<f32>, OjasError> {
    mean_micrograds(parts)
}

#[allow(clippy::too_many_arguments)]
fn forward_block(
    tape: &mut Tape,
    leaves: &[Var],
    ids: Tensor,
    targets: Tensor,
    cos: Tensor,
    sin: Tensor,
    batch: usize,
    time: usize,
    d: usize,
) -> Result<Var, OjasError> {
    let eps = RMS_NORM_EPS;
    let x = tape.embedding(leaves[0], ids)?;
    let h = tape.rms_norm(x, leaves[1], eps)?;
    let q = tape.linear(h, leaves[2])?;
    let q = tape.rms_norm(q, leaves[5], eps)?;
    let k = tape.linear(h, leaves[3])?;
    let k = tape.rms_norm(k, leaves[6], eps)?;
    let v = tape.linear(h, leaves[4])?;
    let q = tape.reshape(q, &[batch, time, 1, d])?;
    let k = tape.reshape(k, &[batch, time, 1, d])?;
    let v = tape.reshape(v, &[batch, time, 1, d])?;
    let q = tape.rope(q, cos.clone(), sin.clone())?;
    let k = tape.rope(k, cos, sin)?;
    // One head: [B, T, 1, D] and [B, 1, T, D] are the same contiguous order.
    let q = tape.reshape(q, &[batch, 1, time, d])?;
    let k = tape.reshape(k, &[batch, 1, time, d])?;
    let v = tape.reshape(v, &[batch, 1, time, d])?;
    let v = tape.value_residual(v, v, leaves[9])?;
    let y = tape.causal_sdpa(q, k, v)?;
    let y = tape.reshape(y, &[batch, time, 1, d])?;
    let y = tape.per_head_gate(h, leaves[7], leaves[8], y)?;
    let y = tape.reshape(y, &[batch, time, d])?;
    let mixed = tape.linear(y, leaves[10])?;
    let x = tape.add(x, mixed)?;
    let n2 = tape.rms_norm(x, leaves[11], eps)?;
    let gate = tape.linear(n2, leaves[12])?;
    let gate = tape.silu(gate)?;
    let up = tape.linear(n2, leaves[13])?;
    let hidden = tape.mul(gate, up)?;
    let ff = tape.linear(hidden, leaves[14])?;
    let x = tape.add(x, ff)?;
    let x = tape.rms_norm(x, leaves[15], eps)?;
    let logits = tape.linear(x, leaves[0])?;
    tape.cross_entropy(logits, targets, None)
}

/// `base ** ((2i) / dim)` inverted, then `cat(freqs, freqs)` along the head axis.
fn rope_cache(time: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut cos = vec![0.0f32; time * dim];
    let mut sin = vec![0.0f32; time * dim];
    for t in 0..time {
        for i in 0..half {
            let inv = 10000.0f64.powf(-((2 * i) as f64) / (dim as f64));
            let freq = (t as f64) * inv;
            let c = freq.cos() as f32;
            let s = freq.sin() as f32;
            cos[t * dim + i] = c;
            cos[t * dim + half + i] = c;
            sin[t * dim + i] = s;
            sin[t * dim + half + i] = s;
        }
    }
    (cos, sin)
}

#[derive(Clone, Copy)]
enum Init {
    Normal,
    Ones,
    Zeros,
}

impl Init {
    fn values(self, shape: &[usize], rng: &mut SplitMix64) -> Vec<f32> {
        let n: usize = shape.iter().product();
        match self {
            Init::Zeros => vec![0.0; n],
            Init::Ones => vec![1.0; n],
            Init::Normal => (0..n).map(|_| 0.02 * rng.unit()).collect(),
        }
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        2.0 * mantissa - 1.0
    }
}
