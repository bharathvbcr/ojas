//! Decode on any [`Backend`] with a device-resident KV cache
//! (`docs/framework-design.md` §6).
//!
//! Every layer is [`ojas_model::block_with`] on [`ojas_model::Eval`], so the
//! block is the model crate's one definition. Only its attention step is
//! this crate's: for both prefill and decode, write the new positions'
//! post-RoPE keys and blended values with `Backend::kv_cache_write`, then
//! [`Graph::cached_attn`] (`Backend::cached_attention_forward`) with
//! `kv_len = len + Tn`. With `kv_len == Tn` that is causal SDPA
//! (`Backend::cached_attention_forward`'s contract), and it folds in GQA.
//!
//! Logits are the last position's only: its hidden row is gathered on the
//! device (`embedding` over the `[Tn, n_embd]` hidden state), normed and
//! projected, and the `[1, vocab]` row is the one readback per call.

use ojas_core::{Backend, DType, OjasError, Tensor};
use ojas_model::{bind, block_with, Eval, Graph, ModelParams, ModelSpec, Rope};

use crate::decode::{self, Forward};
use crate::gpt::{argmax_token, capacity_refusal, check_params};
use crate::sample::GenerateConfig;

/// A nanolab GPT and its KV cache resident on `B`.
///
/// The cache is one `[1, capacity, n_kv_head, head_dim]` key and value
/// tensor per layer, time-major like the host [`crate::KvCache`].
/// `len` positions are filled; the next token goes at absolute position
/// `len`. A call that would pass `capacity` is
/// [`OjasError::CapacityExceeded`] before anything runs, and a call that
/// fails part-way leaves `len` unchanged: slots at or past `len` are never
/// read, so whatever a failed call wrote there is never seen.
pub struct DeviceDecoder<B: Backend> {
    eval: Eval<B>,
    spec: ModelSpec,
    params: ModelParams<Tensor>,
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    capacity: usize,
    len: usize,
}

const OP: &str = "DeviceDecoder::forward";
const DECODE_OP: &str = "DeviceDecoder::decode";

impl<B: Backend> DeviceDecoder<B> {
    /// Check `params` against `spec` (as [`crate::CpuGpt::new`] does),
    /// upload every parameter to `backend` and allocate a cache of
    /// `capacity` positions, `1..=spec.max_seq`. Grouped-query attention is
    /// supported. Everything is charged to `backend`'s budget.
    pub fn new(
        backend: B,
        spec: &ModelSpec,
        params: &ModelParams<Tensor>,
        capacity: usize,
    ) -> Result<Self, OjasError> {
        const NEW: &str = "DeviceDecoder::new";
        check_params(NEW, spec, params)?;
        if capacity == 0 || capacity > spec.max_seq {
            return Err(OjasError::Shape {
                op: NEW,
                detail: format!("capacity {capacity} outside 1..={}", spec.max_seq),
            });
        }
        let mut eval = Eval::new(backend);
        let params = bind(&mut eval, spec, &params.clone().into_flat())?;
        let shape = [1, capacity, spec.n_kv_head, spec.head_dim];
        let alloc = || -> Result<Tensor, OjasError> {
            let backend = eval.backend();
            let host = Tensor::zeros(&shape, DType::F32, backend.budget())?;
            let resident = backend.upload(&host)?;
            drop(host);
            Ok(resident)
        };
        let mut keys = Vec::with_capacity(spec.n_layer);
        let mut values = Vec::with_capacity(spec.n_layer);
        for _ in 0..spec.n_layer {
            keys.push(alloc()?);
            values.push(alloc()?);
        }
        Ok(Self {
            eval,
            spec: *spec,
            params,
            keys,
            values,
            capacity,
            len: 0,
        })
    }

    pub fn backend(&self) -> &B {
        self.eval.backend()
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// Positions filled.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Positions still free.
    pub fn remaining(&self) -> usize {
        self.capacity - self.len
    }

    /// Forget every position. The cache keeps its memory; the next call
    /// starts at position 0.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    fn check_token(&self, op: &'static str, token: u32) -> Result<(), OjasError> {
        if usize::try_from(token).is_ok_and(|row| row < self.spec.vocab) {
            Ok(())
        } else {
            Err(OjasError::OutOfRange {
                op,
                detail: format!("token id {token} >= vocab {}", self.spec.vocab),
            })
        }
    }

    fn refusal(&self, needed: usize) -> OjasError {
        capacity_refusal(self.spec.kv_width(), self.len, self.capacity, needed)
    }

    /// Forward `tokens` at positions `len..len + tokens.len()` (a prefill
    /// when there are several), append their keys and values, and return
    /// the last position's logits, `[vocab]`. One device readback: that
    /// row. The backend is synced before `len` advances, so a fault a
    /// backend defers to its next sync point is reported here and leaves
    /// `len` unchanged.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, OjasError> {
        if tokens.is_empty() {
            return Err(OjasError::Shape {
                op: OP,
                detail: "no tokens".into(),
            });
        }
        for &id in tokens {
            self.check_token(OP, id)?;
        }
        if tokens.len() > self.remaining() {
            return Err(self.refusal(tokens.len()));
        }
        let (at, tn, d) = (self.len, tokens.len(), self.spec.n_embd);
        let budget = self.eval.backend().budget().clone();
        let rope = Rope::rows(&self.spec, at, tn, &budget)?.upload(self.eval.backend())?;
        let ids = Tensor::from_u32(tokens, &[1, tn], &budget)?;
        let mut x = self.eval.embedding(&self.params.tok_emb, &ids)?;
        let mut v0: Option<Tensor> = None;
        let layers = self
            .params
            .blocks
            .iter()
            .zip(self.keys.iter_mut().zip(self.values.iter_mut()));
        for (p, (keys, values)) in layers {
            // Append this layer's post-RoPE keys and blended values at
            // `at..at + tn`, then attend over every filled position.
            let attend = |g: &mut Eval<B>, q: &Tensor, k: &Tensor, v: &Tensor| {
                g.kv_cache_write(keys, k, at)?;
                g.kv_cache_write(values, v, at)?;
                g.cached_attn(q, keys, values, at + tn)
            };
            let out = block_with(
                &mut self.eval,
                &self.spec,
                p,
                &x,
                v0.as_ref(),
                &rope,
                1,
                attend,
            )?;
            if v0.is_none() {
                v0 = Some(out.raw_v);
            }
            x = out.x;
        }
        let rows = self.eval.reshape(&x, &[tn, d])?;
        let last = if tn == 1 {
            rows
        } else {
            let last = u32::try_from(tn - 1).map_err(|_| OjasError::OutOfRange {
                op: OP,
                detail: format!("position {} exceeds u32", tn - 1),
            })?;
            let index = Tensor::from_u32(&[last], &[1], &budget)?;
            self.eval.embedding(&rows, &index)?
        };
        let h = self
            .eval
            .rms_norm(&last, &self.params.norm_f, self.spec.eps())?;
        let logits = self.eval.linear(&h, &self.params.tok_emb)?;
        let host = self.eval.backend().download(&logits)?;
        self.eval.backend().sync()?;
        let row = host.to_f32_vec()?;
        if row.len() != self.spec.vocab {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!("{} logits for vocab {}", row.len(), self.spec.vocab),
            });
        }
        self.len = at + tn;
        Ok(row)
    }

    /// Greedy continuation through [`argmax_token`]. The prompt is one
    /// prefill call; every emitted token except the last is forwarded, so
    /// `prompt.len() + new_tokens - 1` positions must be free, checked
    /// before any forward.
    pub fn greedy_decode(
        &mut self,
        prompt: &[u32],
        new_tokens: usize,
    ) -> Result<Vec<u32>, OjasError> {
        decode::decode(self, DECODE_OP, prompt, new_tokens, &[], argmax_token)
    }

    /// Sampled continuation of `prompt`, the same loop and sampler as
    /// [`crate::CpuGpt::generate`].
    pub fn generate(
        &mut self,
        prompt: &[u32],
        cfg: &GenerateConfig,
    ) -> Result<Vec<u32>, OjasError> {
        decode::generate(self, DECODE_OP, prompt, cfg)
    }
}

impl<B: Backend> Forward for DeviceDecoder<B> {
    fn room(&self) -> usize {
        self.remaining()
    }

    fn refusal(&self, needed: usize) -> OjasError {
        DeviceDecoder::refusal(self, needed)
    }

    fn check_token(&self, op: &'static str, token: u32) -> Result<(), OjasError> {
        DeviceDecoder::check_token(self, op, token)
    }

    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, OjasError> {
        DeviceDecoder::forward(self, tokens)
    }
}
