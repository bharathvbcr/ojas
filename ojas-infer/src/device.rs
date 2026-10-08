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
//! Logits are the last position's only: its hidden row is a view of the
//! `[Tn, n_embd]` hidden state, normed and projected, and the `[1, vocab]`
//! row is the one readback per call. RoPE rows come from a table for every
//! cache position, uploaded once by `new`. The only per-call upload is the
//! token ids.

use ojas_core::{Backend, DType, OjasError, Tensor};
use ojas_model::{bind, block_with, Eval, Graph, ModelParams, ModelSpec, Rope};

use crate::decode::{self, Forward};
use crate::gpt::{capacity_refusal, check_params, truncate_len};
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
    /// RoPE rows for positions `0..capacity`, resident on `B` since `new`;
    /// each forward takes a view of its positions.
    rope: Rope,
    capacity: usize,
    len: usize,
    traffic: HostTraffic,
    /// The id the last [`DeviceDecoder::forward_greedy`] returned, and its
    /// `[1, 1]` U32 tensor on `B`, for the next call to feed back.
    pending: Option<(u32, Tensor)>,
}

/// Host tensors a [`DeviceDecoder`]'s forwards hand to the backend, and the
/// device tensors they read back, since [`DeviceDecoder::new`] returned.
/// Weight and cache uploads in `new` are not counted; transfers a backend
/// makes inside an op are not seen. Bytes are the tensors' element bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostTraffic {
    pub uploads: u64,
    pub upload_bytes: u64,
    pub readbacks: u64,
    pub readback_bytes: u64,
}

impl HostTraffic {
    fn bytes(t: &Tensor) -> Result<u64, OjasError> {
        let n = t.num_elements()?.saturating_mul(t.dtype().size());
        Ok(u64::try_from(n).unwrap_or(u64::MAX))
    }

    fn upload(&mut self, t: &Tensor) -> Result<(), OjasError> {
        self.uploads = self.uploads.saturating_add(1);
        self.upload_bytes = self.upload_bytes.saturating_add(Self::bytes(t)?);
        Ok(())
    }

    fn readback(&mut self, t: &Tensor) -> Result<(), OjasError> {
        self.readbacks = self.readbacks.saturating_add(1);
        self.readback_bytes = self.readback_bytes.saturating_add(Self::bytes(t)?);
        Ok(())
    }
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
        let rope = {
            let backend = eval.backend();
            Rope::new(spec, capacity, backend.budget())?.upload(backend)?
        };
        Ok(Self {
            eval,
            spec: *spec,
            params,
            keys,
            values,
            rope,
            capacity,
            len: 0,
            traffic: HostTraffic::default(),
            pending: None,
        })
    }

    /// Transfers made by forwards so far, failed calls included.
    pub fn traffic(&self) -> HostTraffic {
        self.traffic
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
        self.pending = None;
    }

    /// Keep positions `0..len` and forget the rest, as
    /// [`crate::KvCache::truncate`]. A `len` above [`Self::len`] is
    /// [`OjasError::OutOfRange`] and changes nothing.
    pub fn truncate(&mut self, len: usize) -> Result<(), OjasError> {
        truncate_len(&mut self.len, len, "DeviceDecoder::truncate")?;
        self.pending = None;
        Ok(())
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
        let logits = self.run(tokens)?;
        self.traffic.readback(&logits)?;
        let host = self.eval.backend().download(&logits)?;
        self.eval.backend().sync()?;
        let row = host.to_f32_vec()?;
        if row.len() != self.spec.vocab {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!("{} logits for vocab {}", row.len(), self.spec.vocab),
            });
        }
        self.len += tokens.len();
        Ok(row)
    }

    /// [`Self::forward`] for greedy decoding: the argmax of the last
    /// position's logits is taken on the device (`Backend::argmax_rows`),
    /// so the one readback is the 4-byte id, not the `[vocab]` row. Ties go
    /// to the lowest id and a non-finite logit is [`OjasError::NonFinite`],
    /// as [`crate::argmax_token`]. The id's device tensor is kept: when the next
    /// call forwards exactly that id, it is fed to the embedding from the
    /// device with no upload. The same `len` contract as [`Self::forward`].
    pub fn forward_greedy(&mut self, tokens: &[u32]) -> Result<u32, OjasError> {
        let logits = self.run(tokens)?;
        let next = self.eval.backend().argmax_rows(&logits)?;
        let host = if next.device().is_some() {
            self.traffic.readback(&next)?;
            self.eval.backend().download(&next)?
        } else {
            next.clone()
        };
        self.eval.backend().sync()?;
        let id = match host.u32_slice()? {
            &[id] => id,
            other => {
                return Err(OjasError::Shape {
                    op: OP,
                    detail: format!("argmax_rows returned {} ids for one row", other.len()),
                })
            }
        };
        self.check_token(OP, id)?;
        self.len += tokens.len();
        self.pending = Some((id, next.reshape(&[1, 1])?));
        Ok(id)
    }

    /// Validate `tokens`, run every block over them (writing their keys and
    /// values at `len..`), and return the last position's `[1, vocab]`
    /// logits on the device. `len` is the caller's to advance.
    fn run(&mut self, tokens: &[u32]) -> Result<Tensor, OjasError> {
        // A device id from the last greedy step serves only the very next
        // call, and only if it forwards exactly that id.
        let pending = self.pending.take();
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
        let rope = self.rope.slice(at, tn)?;
        let ids = match pending {
            Some((id, resident)) if tokens == [id] => resident,
            _ => {
                let budget = self.eval.backend().budget().clone();
                let ids = Tensor::from_u32(tokens, &[1, tn], &budget)?;
                self.traffic.upload(&ids)?;
                ids
            }
        };
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
        // The last position's row, a view of the hidden state: nothing is
        // uploaded to pick it.
        let row_bytes = d * DType::F32.size();
        let last = x.narrow((tn - 1) * row_bytes, &[1, d], &[d, 1])?;
        let h = self
            .eval
            .rms_norm(&last, &self.params.norm_f, self.spec.eps())?;
        self.eval.linear(&h, &self.params.tok_emb)
    }

    /// Greedy continuation. The prompt is one prefill call; every emitted
    /// token except the last is forwarded, so `prompt.len() + new_tokens -
    /// 1` positions must be free, checked before any forward. Each step is
    /// [`Self::forward_greedy`]: a 4-byte readback, and the id is fed back
    /// from the device. The ids equal [`crate::argmax_token`] over
    /// [`Self::forward`]'s logits.
    pub fn greedy_decode(
        &mut self,
        prompt: &[u32],
        new_tokens: usize,
    ) -> Result<Vec<u32>, OjasError> {
        decode::decode(self, DECODE_OP, prompt, new_tokens, &[], None)
    }

    /// Sampled continuation of `prompt`, the same loop and sampler as
    /// [`crate::CpuGpt::generate`].
    ///
    /// The prompt is forwarded from position [`Self::len`]. Each emitted
    /// token except the last is forwarded, so the cache must have room for
    /// `prompt.len() + max_new_tokens - 1` more positions (`prompt.len()`
    /// when `max_new_tokens` is 0), checked before any forward. The last
    /// returned token is not in the cache: to continue later, pass it as
    /// the first token of the next prompt.
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

    fn forward_greedy(&mut self, tokens: &[u32]) -> Result<u32, OjasError> {
        DeviceDecoder::forward_greedy(self, tokens)
    }
}
