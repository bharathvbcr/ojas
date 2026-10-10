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
//! `[Tn, n_embd]` hidden state, normed and projected. What is read back of
//! that `[1, vocab]` row depends on the caller: all of it
//! ([`DeviceDecoder::forward`]), its argmax ([`DeviceDecoder::forward_greedy`],
//! 4 bytes) or its `k` leaders ([`DeviceDecoder::forward_top`], `8k` bytes).
//! RoPE rows come from a table for every cache position, uploaded once by
//! `new`. The only per-call upload is the token ids, and none when the call
//! forwards the one id the previous call picked on the device.

use ojas_core::{Backend, DType, OjasError, Tensor};
use ojas_model::{bind, Eval, ModelParams, ModelSpec, Rope};

use crate::cache::{append, run_pieces, Kv, Ring};
use crate::decode::{self, check_token, Forward};
use crate::gpt::check_params;
use crate::sample::{leading_candidates, ranked_candidates, GenerateConfig};

/// A nanolab GPT and its KV cache resident on `B`.
///
/// The cache is one `[1, slots, n_kv_head, head_dim]` key and value tensor
/// per layer, time-major like the host [`crate::KvCache`]: `slots` is the
/// capacity, or `2W - 1` for a model with a sliding window `W` below it
/// (a ring; see [`Self::slots`]). `len` positions are filled; the next
/// token goes at absolute position `len`. A call that would pass
/// `capacity` is [`OjasError::CapacityExceeded`] before anything runs, and
/// a call that fails part-way leaves `len` unchanged: slots at or past
/// `len` are never read, so whatever a failed call wrote there is never
/// seen. On a ring, a call of more than `W` tokens runs as `W`-token
/// pieces, and a failure leaves `len` after the last piece that completed.
pub struct DeviceDecoder<B: Backend> {
    eval: Eval<B>,
    spec: ModelSpec,
    params: ModelParams<Tensor>,
    kv: Kv,
    /// RoPE rows for positions `0..capacity`, resident on `B` since `new`;
    /// each forward takes a view of its positions.
    rope: Rope,
    traffic: HostTraffic,
    /// The id the last [`DeviceDecoder::forward_greedy`] returned, or the one
    /// leader of the last [`DeviceDecoder::forward_top`], and its `[1, 1]`
    /// U32 tensor on `B`, for the next call to feed back.
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
        let ring = Ring::new(spec.attention_window(), capacity);
        let kv = Kv::new(
            [spec.n_layer, spec.n_kv_head, spec.head_dim],
            ring,
            |shape| {
                let backend = eval.backend();
                let host = Tensor::zeros(shape, DType::F32, backend.budget())?;
                let resident = backend.upload(&host)?;
                drop(host);
                Ok(resident)
            },
        )?;
        let rope = {
            let backend = eval.backend();
            Rope::new(spec, capacity, backend.budget())?.upload(backend)?
        };
        Ok(Self {
            eval,
            spec: *spec,
            params,
            kv,
            rope,
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
        self.kv.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.kv.ring.capacity()
    }

    /// Slots per layer: the capacity, or `2W - 1` for a sliding window `W`
    /// smaller than it, a ring holding the positions the next call needs.
    pub fn slots(&self) -> usize {
        self.kv.ring.slots()
    }

    /// Positions still free.
    pub fn remaining(&self) -> usize {
        self.kv.ring.remaining()
    }

    /// Forget every position. The cache keeps its memory; the next call
    /// starts at position 0.
    pub fn reset(&mut self) {
        self.kv.ring.reset();
        self.pending = None;
    }

    /// Keep positions `0..len` and forget the rest, as
    /// [`crate::KvCache::truncate`]. A `len` above [`Self::len`] is
    /// [`OjasError::OutOfRange`] and changes nothing, and so, on a ring, is
    /// a prefix whose window has been overwritten (the ring holds the last
    /// `slots` positions written).
    pub fn truncate(&mut self, len: usize) -> Result<(), OjasError> {
        self.kv.ring.truncate(len, "DeviceDecoder::truncate")?;
        self.pending = None;
        Ok(())
    }

    fn refusal(&self, needed: usize) -> OjasError {
        self.kv.ring.refusal(self.spec.kv_width(), needed)
    }

    /// Forward `tokens` at positions `len..len + tokens.len()` (a prefill
    /// when there are several), append their keys and values, and return
    /// the last position's logits, `[vocab]`. One device readback: that
    /// row. The backend is synced before `len` advances, so a fault a
    /// backend defers to its next sync point is reported here and leaves
    /// `len` unchanged.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, OjasError> {
        let (logits, last) = self.run(tokens)?;
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
        append(&mut self.kv, last);
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
        let (logits, last) = self.run(tokens)?;
        let next = self.eval.backend().argmax_rows(&logits)?;
        let host = self.read_back(&next)?;
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
        check_token(OP, id, self.spec.vocab)?;
        append(&mut self.kv, last);
        self.pending = Some((id, next.reshape(&[1, 1])?));
        Ok(id)
    }

    /// [`Self::forward`] for sampling that needs only the `k` leaders of the
    /// last position's logits (a `top_k`, or temperature 0 as `k` = 1): they
    /// are selected on the device (`Backend::topk_rows`), so the readback is
    /// `k` values and `k` ids, `8k` bytes, not the `[vocab]` row. Returns
    /// what `ranked_candidates` returns for that row: its finite leaders
    /// ranked by (logit desc, index asc), at most `k` (a `k` past the
    /// vocabulary keeps every finite logit), with the same refusals. `-inf`
    /// leaders (masked logits) are dropped. A backend that refuses `k`
    /// (`OjasError::Unsupported`: Metal and wgpu take `k <= 1024`) has the
    /// row read back and ranked on the host instead. With one leader its id
    /// is kept on the device for the next call, as
    /// [`Self::forward_greedy`]'s is. The same `len` contract as
    /// [`Self::forward`].
    pub fn forward_top(&mut self, tokens: &[u32], k: usize) -> Result<Vec<(u32, f32)>, OjasError> {
        let (logits, last) = self.run(tokens)?;
        // `topk_rows` refuses `k` past its columns; `ranked_candidates`
        // keeps every finite logit then, and so do the row's `vocab` leaders.
        let k = k.min(self.spec.vocab);
        let (leaders, kept) = match self.eval.backend().topk_rows(&logits, k) {
            Ok((values, ids)) => {
                let host_values = self.read_back(&values)?;
                let host_ids = self.read_back(&ids)?;
                self.eval.backend().sync()?;
                let leaders =
                    leading_candidates(&host_values.to_f32_vec()?, host_ids.u32_slice()?)?;
                (leaders, (k == 1).then_some(ids))
            }
            Err(OjasError::Unsupported { .. }) => {
                let row = self.read_back(&logits)?;
                self.eval.backend().sync()?;
                (ranked_candidates(&row.to_f32_vec()?, Some(k))?, None)
            }
            Err(e) => return Err(e),
        };
        for &(id, _) in &leaders {
            check_token(OP, id, self.spec.vocab)?;
        }
        append(&mut self.kv, last);
        if let (Some(ids), &[(id, _)]) = (kept, leaders.as_slice()) {
            self.pending = Some((id, ids.reshape(&[1, 1])?));
        }
        Ok(leaders)
    }

    /// `t` on the host: a device tensor is downloaded and counted as a
    /// readback; a host tensor is shared as it is.
    fn read_back(&mut self, t: &Tensor) -> Result<Tensor, OjasError> {
        if t.device().is_none() {
            return Ok(t.clone());
        }
        self.traffic.readback(t)?;
        self.eval.backend().download(t)
    }

    /// Validate `tokens`, run every block over them (writing their keys and
    /// values at `len..`), and return the last position's `[1, vocab]`
    /// logits on the device with the last piece's length ([`run_pieces`]),
    /// which is the caller's to append.
    fn run(&mut self, tokens: &[u32]) -> Result<(Tensor, usize), OjasError> {
        // A device id from the last greedy step serves only the very next
        // call, and only if it forwards exactly that id.
        let mut pending = self.pending.take();
        if tokens.is_empty() {
            return Err(OjasError::Shape {
                op: OP,
                detail: "no tokens".into(),
            });
        }
        for &id in tokens {
            check_token(OP, id, self.spec.vocab)?;
        }
        if tokens.len() > self.remaining() {
            return Err(self.refusal(tokens.len()));
        }
        let traffic = &mut self.traffic;
        run_pieces(
            &mut self.eval,
            &self.spec,
            &self.params,
            &mut self.kv,
            &self.rope,
            tokens,
            |eval, piece| match pending.take() {
                Some((id, resident)) if piece == [id] => Ok(resident),
                _ => {
                    let budget = eval.backend().budget().clone();
                    let ids = Tensor::from_u32(piece, &[1, piece.len()], &budget)?;
                    traffic.upload(&ids)?;
                    Ok(ids)
                }
            },
        )
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
        decode::decode(
            self,
            DECODE_OP,
            prompt,
            new_tokens,
            &[],
            decode::Pick::Greedy,
        )
    }

    /// Sampled continuation of `prompt`, the same loop and sampler as
    /// [`crate::CpuGpt::generate`]. A setting that needs only the leaders
    /// (temperature 0, or a `top_k`) forwards through [`Self::forward_top`],
    /// so a step reads back `8k` bytes (`k` = 1 at temperature 0); otherwise
    /// through [`Self::forward`] and the `[vocab]` row.
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

    fn vocab(&self) -> usize {
        self.spec.vocab
    }

    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, OjasError> {
        DeviceDecoder::forward(self, tokens)
    }

    fn forward_greedy(&mut self, tokens: &[u32]) -> Result<u32, OjasError> {
        DeviceDecoder::forward_greedy(self, tokens)
    }

    fn forward_top(&mut self, tokens: &[u32], k: usize) -> Result<Vec<(u32, f32)>, OjasError> {
        DeviceDecoder::forward_top(self, tokens, k)
    }
}
