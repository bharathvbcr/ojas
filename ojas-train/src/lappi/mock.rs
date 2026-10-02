//! A complete tiny model on the CPU, so the loop is exercised end to end with no GPU.
//!
//! It is a real [`StepProvider`], not a stub: a forward, an analytic backward that
//! `tests/trainer_loop.rs` checks against central differences, AdamW through
//! [`crate::adamw::adamw_update`], and a checkpoint that resumes bit for bit. What it does not
//! do is resemble Qwen3.5 beyond the parts the contract touches:
//!
//! ```text
//! h0[p]   = E[tok[p]]                                  embed_tokens.weight [V, H]
//! h{l+1}  = h{l} + tanh(W_l h{l})                      layers.{l}.mlp.weight [H, H] ([out, in])
//! x[p]    = h{L}[p] * (1 + n)                          norm.weight [H], zero-centred like Qwen's
//! logits  = E x[p]                                     the tied head
//! ```
//!
//! Positions do not mix (there is no attention), which the loop does not need: what it needs
//! is per-sequence row losses at chosen positions, gradients of a loss outside the model at
//! chosen hidden states, a bank that sums sequences, a global norm, and a step count.
//! Everything is f32 in a fixed order, so two runs are the same bits.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::adamw::{adamw_update, Moments};
use crate::step::{
    check_per_entry, refuse_lr_scale, AdamWHyper, BankMode, HiddenGrad, ParamSpec, SequenceJob, StepError, StepProvider,
};

/// The toy model.
pub struct ToyProvider {
    vocab: usize,
    hidden: usize,
    layers: usize,
    specs: Vec<ParamSpec>,
    values: Vec<Vec<f32>>,
    grads: Vec<Vec<f32>>,
    moments: Vec<Moments>,
    step: u64,
    lr_scale_supported: bool,
}

/// A deterministic stream for initial weights (splitmix64), so a test needs no RNG crate.
pub struct SplitMix(u64);

impl SplitMix {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-a, a)`.
    pub fn uniform(&mut self, a: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        (2.0 * u - 1.0) * a
    }
}

const EMBED: usize = 0;

/// What the forward keeps for the backward: each layer's input `h_l[p]` (`inputs[l][p]`), its
/// activation `tanh(W_l h_l[p])` (`acts[l][p]`), and the final hidden state `x[p]`.
struct Forward {
    inputs: Vec<Vec<Vec<f32>>>,
    acts: Vec<Vec<Vec<f32>>>,
    x: Vec<Vec<f32>>,
}

impl ToyProvider {
    pub fn new(vocab: usize, hidden: usize, layers: usize, seed: u64) -> Result<Self, StepError> {
        if vocab < 2 || hidden == 0 {
            return Err(StepError::Contract(format!("vocab {vocab} and hidden {hidden} are too small")));
        }
        let mut specs = vec![ParamSpec::new("embed_tokens.weight", &[vocab, hidden])];
        for l in 0..layers {
            specs.push(ParamSpec::new(format!("layers.{l}.mlp.weight"), &[hidden, hidden]));
        }
        specs.push(ParamSpec::new("norm.weight", &[hidden]));
        let mut rng = SplitMix::new(seed);
        let mut values = Vec::with_capacity(specs.len());
        for (i, s) in specs.iter().enumerate() {
            let n = s.numel()?;
            let a = if i == EMBED { 1.0 } else if i == specs.len() - 1 { 0.1 } else { 0.5 / (hidden as f32).sqrt() };
            values.push((0..n).map(|_| rng.uniform(a)).collect());
        }
        let grads = specs.iter().map(|s| s.numel().map(|n| vec![0.0; n])).collect::<Result<_, _>>()?;
        let moments = specs.iter().map(|s| s.numel().map(Moments::zeros)).collect::<Result<_, _>>()?;
        Ok(Self {
            vocab,
            hidden,
            layers,
            specs,
            values,
            grads,
            moments,
            step: 0,
            lr_scale_supported: true,
        })
    }

    /// The same model, refusing any per-entry learning rate other than 1.0, as tessl does
    /// until its `lr_scale` lands.
    pub fn without_lr_scale(mut self) -> Self {
        self.lr_scale_supported = false;
        self
    }

    pub fn values(&self) -> &[Vec<f32>] {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [Vec<f32>] {
        &mut self.values
    }

    /// The bank: the summed gradients since the last [`BankMode::Overwrite`].
    pub fn grads(&self) -> &[Vec<f32>] {
        &self.grads
    }

    /// The bank, writable: lets a test plant a non-finite gradient the way a broken kernel would.
    pub fn grads_mut(&mut self) -> &mut [Vec<f32>] {
        &mut self.grads
    }

    pub fn moments(&self) -> &[Moments] {
        &self.moments
    }

    fn norm(&self) -> &[f32] {
        &self.values[self.specs.len() - 1]
    }

    /// The final hidden state `x` at every position, and every layer's input and activation.
    fn forward(&self, tokens: &[u32]) -> Forward {
        let h = self.hidden;
        let e = &self.values[EMBED];
        // inputs[l][p] = h_l at p; acts[l][p] = tanh(W_l h_l)
        let mut inputs = Vec::with_capacity(self.layers + 1);
        let mut acts = Vec::with_capacity(self.layers);
        let h0: Vec<Vec<f32>> = tokens.iter().map(|&t| e[t as usize * h..(t as usize + 1) * h].to_vec()).collect();
        inputs.push(h0);
        for l in 0..self.layers {
            let w = &self.values[1 + l];
            let cur = &inputs[l];
            let mut a_l = Vec::with_capacity(cur.len());
            let mut next = Vec::with_capacity(cur.len());
            for hp in cur {
                let mut a = vec![0.0f32; h];
                for (o, ao) in a.iter_mut().enumerate() {
                    let mut u = 0.0f32;
                    for i in 0..h {
                        u += w[o * h + i] * hp[i];
                    }
                    *ao = u.tanh();
                }
                next.push(hp.iter().zip(&a).map(|(x, y)| x + y).collect());
                a_l.push(a);
            }
            acts.push(a_l);
            inputs.push(next);
        }
        let n = self.norm();
        let x = inputs[self.layers]
            .iter()
            .map(|hp| hp.iter().zip(n).map(|(v, w)| v * (1.0 + w)).collect())
            .collect();
        Forward { inputs, acts, x }
    }

    /// The final hidden state at every position of `tokens`, row by row: what an outside loss
    /// reads. Exposed for the central-difference check.
    pub fn hidden_states(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        self.forward(tokens).x
    }

    /// The loss the provider reports for `job` with the model as it is: the sum of the row
    /// cross-entropies. Exposed for the central-difference check.
    pub fn row_loss(&self, job: &SequenceJob<'_>) -> Result<f64, StepError> {
        job.validate(self.vocab)?;
        let x = self.forward(job.tokens).x;
        Ok(job
            .rows
            .positions
            .iter()
            .zip(job.rows.targets)
            .map(|(&p, &t)| self.ce(&x[p as usize], t).0)
            .sum())
    }

    /// Cross-entropy of `x` against `target` through the tied head, and the softmax.
    fn ce(&self, x: &[f32], target: u32) -> (f64, Vec<f64>) {
        let h = self.hidden;
        let e = &self.values[EMBED];
        let logits: Vec<f64> = (0..self.vocab)
            .map(|v| e[v * h..(v + 1) * h].iter().zip(x).map(|(a, b)| f64::from(a * b)).sum())
            .collect();
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = logits.iter().map(|l| (l - max).exp()).sum();
        let lse = max + z.ln();
        let sm = logits.iter().map(|l| (l - lse).exp()).collect();
        (lse - logits[target as usize], sm)
    }
}

impl StepProvider for ToyProvider {
    fn parameters(&self) -> &[ParamSpec] {
        &self.specs
    }

    fn hidden_size(&self) -> usize {
        self.hidden
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn accumulate(
        &mut self,
        job: &SequenceJob<'_>,
        mode: BankMode,
        external: Option<&mut HiddenGrad<'_>>,
    ) -> Result<f64, StepError> {
        job.validate(self.vocab)?;
        if !job.hidden_at.is_empty() && external.is_none() {
            return Err(StepError::Contract("hidden positions were requested with no loss to read them".into()));
        }
        let h = self.hidden;
        let Forward { inputs, acts, x } = self.forward(job.tokens);
        let t = job.tokens.len();
        // dx[p]: the gradient at the final hidden state.
        let mut dx = vec![vec![0.0f32; h]; t];
        let mut new_grads: Vec<Vec<f32>> = self.grads.iter().map(|g| vec![0.0; g.len()]).collect();
        let scale = f64::from(job.rows.scale);
        let mut loss = 0.0f64;
        for (&p, &tg) in job.rows.positions.iter().zip(job.rows.targets) {
            let xp = &x[p as usize];
            let (ce, sm) = self.ce(xp, tg);
            loss += ce;
            let e = &self.values[EMBED];
            for v in 0..self.vocab {
                let d = ((sm[v] - if v == tg as usize { 1.0 } else { 0.0 }) * scale) as f32;
                for i in 0..h {
                    dx[p as usize][i] += d * e[v * h + i];
                    new_grads[EMBED][v * h + i] += d * xp[i];
                }
            }
        }
        if let Some(ext) = external
            && !job.hidden_at.is_empty()
        {
            let mut block = Vec::with_capacity(job.hidden_at.len() * h);
            for &p in job.hidden_at {
                block.extend_from_slice(&x[p as usize]);
            }
            let dh = ext(job.hidden_at, &block)?;
            if dh.len() != block.len() {
                return Err(StepError::Contract(format!(
                    "the outside loss returned {} gradient values for {} hidden values",
                    dh.len(),
                    block.len()
                )));
            }
            for (k, &p) in job.hidden_at.iter().enumerate() {
                for i in 0..h {
                    dx[p as usize][i] += dh[k * h + i];
                }
            }
        }
        // Back through the norm, the layers and the gather.
        let last = self.specs.len() - 1;
        let n = self.norm().to_vec();
        let mut dh_l: Vec<Vec<f32>> = Vec::with_capacity(t);
        for p in 0..t {
            let hl = &inputs[self.layers][p];
            for i in 0..h {
                new_grads[last][i] += dx[p][i] * hl[i];
            }
            dh_l.push(dx[p].iter().zip(&n).map(|(d, w)| d * (1.0 + w)).collect());
        }
        for l in (0..self.layers).rev() {
            let w = &self.values[1 + l];
            let mut below = Vec::with_capacity(t);
            for p in 0..t {
                let a = &acts[l][p];
                let hin = &inputs[l][p];
                let du: Vec<f32> = dh_l[p].iter().zip(a).map(|(d, a)| d * (1.0 - a * a)).collect();
                let mut dprev = dh_l[p].clone();
                for o in 0..h {
                    for i in 0..h {
                        new_grads[1 + l][o * h + i] += du[o] * hin[i];
                        dprev[i] += w[o * h + i] * du[o];
                    }
                }
                below.push(dprev);
            }
            dh_l = below;
        }
        for (p, &tok) in job.tokens.iter().enumerate() {
            for i in 0..h {
                new_grads[EMBED][tok as usize * h + i] += dh_l[p][i];
            }
        }
        match mode {
            BankMode::Overwrite => self.grads = new_grads,
            BankMode::Add => {
                for (g, n) in self.grads.iter_mut().zip(new_grads) {
                    for (a, b) in g.iter_mut().zip(n) {
                        *a += b;
                    }
                }
            }
        }
        Ok(loss)
    }

    fn grad_sq_norm(&mut self) -> Result<f64, StepError> {
        Ok(self
            .grads
            .iter()
            .map(|g| g.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>())
            .sum())
    }

    fn supports_lr_scale(&self) -> bool {
        self.lr_scale_supported
    }

    fn adamw_step(&mut self, hyper: &AdamWHyper, step: u64, lr_scale: &[f64], weight_decay: &[f64]) -> Result<(), StepError> {
        hyper.validate()?;
        check_per_entry(self.specs.len(), lr_scale, weight_decay)?;
        if !self.lr_scale_supported {
            refuse_lr_scale(lr_scale, "this toy provider (built without_lr_scale)")?;
        }
        if step != self.step + 1 {
            return Err(StepError::Contract(format!(
                "asked for step {step}, but this provider has taken {}",
                self.step
            )));
        }
        for i in 0..self.specs.len() {
            let lr = hyper.lr * lr_scale[i];
            adamw_update(&mut self.values[i], &self.grads[i], &mut self.moments[i], hyper, lr, weight_decay[i], step)?;
        }
        self.step = step;
        Ok(())
    }

    fn step_count(&self) -> u64 {
        self.step
    }

    fn read_parameters(&mut self) -> Result<Vec<Vec<f32>>, StepError> {
        Ok(self.values.clone())
    }

    fn describe(&self) -> String {
        format!(
            "toy CPU provider (qd_train::mock): vocab {}, hidden {}, {} layer(s), f32, per-entry lr {}",
            self.vocab,
            self.hidden,
            self.layers,
            if self.lr_scale_supported { "yes" } else { "no" }
        )
    }

    fn save_state(&mut self, dir: &Path) -> Result<Vec<PathBuf>, StepError> {
        let path = dir.join("toy-state.f32");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.step.to_le_bytes());
        for ((v, m), s) in self.values.iter().zip(&self.moments).zip(&self.specs) {
            bytes.extend_from_slice(&(s.numel()? as u64).to_le_bytes());
            for x in v.iter().chain(&m.m).chain(&m.v) {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        let mut f = fs::File::options()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| StepError::Io(format!("{}: {e}", path.display())))?;
        f.write_all(&bytes).map_err(|e| StepError::Io(format!("{}: {e}", path.display())))?;
        f.sync_all().map_err(|e| StepError::Io(format!("{}: {e}", path.display())))?;
        Ok(vec![path])
    }

    fn load_state(&mut self, dir: &Path) -> Result<(), StepError> {
        let path = dir.join("toy-state.f32");
        let bytes = fs::read(&path).map_err(|e| StepError::Io(format!("{}: {e}", path.display())))?;
        let bad = || StepError::Contract(format!("{} does not describe this model", path.display()));
        let mut at = 0usize;
        let take8 = |at: &mut usize| -> Result<u64, StepError> {
            let s = bytes.get(*at..*at + 8).ok_or_else(bad)?;
            *at += 8;
            Ok(u64::from_le_bytes(s.try_into().map_err(|_| bad())?))
        };
        let step = take8(&mut at)?;
        let mut values = Vec::new();
        let mut moments = Vec::new();
        for s in &self.specs {
            let n = s.numel()?;
            if take8(&mut at)? != n as u64 {
                return Err(bad());
            }
            let read = |at: &mut usize| -> Result<Vec<f32>, StepError> {
                let s = bytes.get(*at..*at + 4 * n).ok_or_else(bad)?;
                *at += 4 * n;
                Ok(s.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
            };
            values.push(read(&mut at)?);
            moments.push(Moments {
                m: read(&mut at)?,
                v: read(&mut at)?,
            });
        }
        if at != bytes.len() {
            return Err(bad());
        }
        self.values = values;
        self.moments = moments;
        self.step = step;
        Ok(())
    }
}
