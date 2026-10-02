//! One op vocabulary, two executors (`docs/framework-design.md` §2).
//!
//! [`Graph`] has one method per op the nanolab block uses. `Tape<B>`
//! records each op for training (`V = Var`); [`Eval`] runs it eagerly with
//! no record (`V = Tensor`). Both call the same [`Backend`] method for every
//! op, so under `Numerics::Exact` their forward values are equal bit for bit.

use ojas_autograd::{Tape, Var};
use ojas_core::{Backend, BackendId, CeChunk, OjasError, Tensor};

use crate::spec::ModelSpec;

/// The ops the nanolab block and head need.
///
/// Host tensors passed by reference (`ids`, `cos`, `sin`, `targets`, a KV
/// cache) are made resident on the executor's backend; a tensor already
/// there is shared, not copied.
pub trait Graph {
    /// A value in this graph: a tape variable or a tensor.
    type V: Clone;

    /// The backend every op runs on.
    fn backend_id(&self) -> BackendId;

    /// Refuse a spec this executor cannot run: a tape records training
    /// graphs ([`ModelSpec::validate_for_training`], no grouped-query
    /// attention); [`Eval`] runs any valid spec ([`ModelSpec::validate`]).
    fn check_spec(&self, spec: &ModelSpec) -> Result<(), OjasError>;

    /// Bring a parameter into the graph. On a tape it is a leaf whose
    /// gradient [`Tape::take_grad`] returns.
    fn param(&mut self, value: &Tensor) -> Result<Self::V, OjasError>;

    /// The tensor behind `v`.
    fn tensor<'a>(&'a self, v: &'a Self::V) -> Result<&'a Tensor, OjasError>;

    fn embedding(&mut self, table: &Self::V, ids: &Tensor) -> Result<Self::V, OjasError>;
    fn linear(&mut self, x: &Self::V, w: &Self::V) -> Result<Self::V, OjasError>;
    fn rms_norm(&mut self, x: &Self::V, w: &Self::V, eps: f32) -> Result<Self::V, OjasError>;
    fn rope(&mut self, x: &Self::V, cos: &Tensor, sin: &Tensor) -> Result<Self::V, OjasError>;
    fn permute(&mut self, x: &Self::V, dims: &[usize]) -> Result<Self::V, OjasError>;
    /// Causal SDPA over `[B, H, T, D]`.
    fn sdpa(&mut self, q: &Self::V, k: &Self::V, v: &Self::V) -> Result<Self::V, OjasError>;
    /// [`Backend::cached_attention_forward`]: `q` `[B, Tq, H, D]` against
    /// the first `kv_len` positions of a `[B, Tcap, Hkv, D]` cache. With
    /// `kv_len == Tq` and a cache of exactly `Tq` positions it is causal
    /// attention in time-major layout, grouped-query included.
    fn cached_attn(
        &mut self,
        q: &Self::V,
        k_cache: &Self::V,
        v_cache: &Self::V,
        kv_len: usize,
    ) -> Result<Self::V, OjasError>;
    /// Per-head sigmoid gate of `x` applied to `attn`.
    fn gate(
        &mut self,
        x: &Self::V,
        w: &Self::V,
        b: &Self::V,
        attn: &Self::V,
    ) -> Result<Self::V, OjasError>;
    /// Value residual `(1 - s) v + s v0`, `s = sigmoid(lambda)`.
    fn vres(&mut self, v: &Self::V, v0: &Self::V, lambda: &Self::V) -> Result<Self::V, OjasError>;
    fn silu(&mut self, x: &Self::V) -> Result<Self::V, OjasError>;
    fn mul(&mut self, a: &Self::V, b: &Self::V) -> Result<Self::V, OjasError>;
    fn add(&mut self, x: &Self::V, y: &Self::V) -> Result<Self::V, OjasError>;
    /// The same contiguous values under another shape. No bytes move.
    fn reshape(&mut self, x: &Self::V, shape: &[usize]) -> Result<Self::V, OjasError>;
    /// Mean cross-entropy of `x @ w^T` through the backend's fused
    /// [`Backend::linear_cross_entropy_mean`]. `x` is `[N, d]`, `w` is
    /// `[V, d]`, `targets` is `[N]` `U32`. Returns a rank-0 loss.
    fn lin_ce(
        &mut self,
        x: &Self::V,
        w: &Self::V,
        targets: &Tensor,
        ignore: Option<u32>,
        chunk: CeChunk,
    ) -> Result<Self::V, OjasError>;
}

impl<B: Backend> Graph for Tape<B> {
    type V = Var;

    fn backend_id(&self) -> BackendId {
        self.backend().id()
    }

    fn check_spec(&self, spec: &ModelSpec) -> Result<(), OjasError> {
        spec.validate_for_training()
    }

    fn param(&mut self, value: &Tensor) -> Result<Var, OjasError> {
        self.leaf(value.clone())
    }

    fn tensor<'a>(&'a self, v: &'a Var) -> Result<&'a Tensor, OjasError> {
        self.value(*v)
    }

    fn embedding(&mut self, table: &Var, ids: &Tensor) -> Result<Var, OjasError> {
        Tape::embedding(self, *table, ids.clone())
    }

    fn linear(&mut self, x: &Var, w: &Var) -> Result<Var, OjasError> {
        Tape::linear(self, *x, *w)
    }

    fn rms_norm(&mut self, x: &Var, w: &Var, eps: f32) -> Result<Var, OjasError> {
        Tape::rms_norm(self, *x, *w, eps)
    }

    fn rope(&mut self, x: &Var, cos: &Tensor, sin: &Tensor) -> Result<Var, OjasError> {
        Tape::rope(self, *x, cos.clone(), sin.clone())
    }

    fn permute(&mut self, x: &Var, dims: &[usize]) -> Result<Var, OjasError> {
        Tape::permute(self, *x, dims)
    }

    fn sdpa(&mut self, q: &Var, k: &Var, v: &Var) -> Result<Var, OjasError> {
        Tape::causal_sdpa(self, *q, *k, *v)
    }

    /// Refused: cached attention is an inference op with no backward, and
    /// the tape records training graphs only. The decode path runs on
    /// [`Eval`].
    fn cached_attn(
        &mut self,
        _q: &Var,
        _k_cache: &Var,
        _v_cache: &Var,
        _kv_len: usize,
    ) -> Result<Var, OjasError> {
        Err(OjasError::Unsupported {
            op: "Graph::cached_attn",
            detail: "cached attention has no backward; record decode on Eval, not a Tape"
                .to_string(),
        })
    }

    fn gate(&mut self, x: &Var, w: &Var, b: &Var, attn: &Var) -> Result<Var, OjasError> {
        Tape::per_head_gate(self, *x, *w, *b, *attn)
    }

    fn vres(&mut self, v: &Var, v0: &Var, lambda: &Var) -> Result<Var, OjasError> {
        Tape::value_residual(self, *v, *v0, *lambda)
    }

    fn silu(&mut self, x: &Var) -> Result<Var, OjasError> {
        Tape::silu(self, *x)
    }

    fn mul(&mut self, a: &Var, b: &Var) -> Result<Var, OjasError> {
        Tape::mul(self, *a, *b)
    }

    fn add(&mut self, x: &Var, y: &Var) -> Result<Var, OjasError> {
        Tape::add(self, *x, *y)
    }

    fn reshape(&mut self, x: &Var, shape: &[usize]) -> Result<Var, OjasError> {
        Tape::reshape(self, *x, shape)
    }

    fn lin_ce(
        &mut self,
        x: &Var,
        w: &Var,
        targets: &Tensor,
        ignore: Option<u32>,
        chunk: CeChunk,
    ) -> Result<Var, OjasError> {
        Tape::linear_cross_entropy(self, *x, *w, targets.clone(), ignore, chunk)
    }
}

/// Eager executor: each op runs on `B` at once and nothing is recorded.
pub struct Eval<B: Backend> {
    backend: B,
}

impl<B: Backend> Eval<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// A host tensor uploaded to the backend, or a resident one shared.
    /// The CPU path keeps the tensor exactly as given, as `Tape` does.
    fn place(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        if self.backend.id() == BackendId::Cpu && tensor.device().is_none() {
            Ok(tensor.clone())
        } else {
            self.backend.upload(tensor)
        }
    }

    /// [`Backend::kv_cache_write`]: write `src` `[B, Tn, Hkv, D]` into the
    /// uniquely owned `cache` at time positions `at..at + Tn`. On any error
    /// the cache is unchanged.
    pub fn kv_cache_write(
        &self,
        cache: &mut Tensor,
        src: &Tensor,
        at: usize,
    ) -> Result<(), OjasError> {
        self.backend.kv_cache_write(cache, src, at)
    }
}

impl<B: Backend> Graph for Eval<B> {
    type V = Tensor;

    fn backend_id(&self) -> BackendId {
        self.backend.id()
    }

    fn check_spec(&self, spec: &ModelSpec) -> Result<(), OjasError> {
        spec.validate()
    }

    fn param(&mut self, value: &Tensor) -> Result<Tensor, OjasError> {
        self.place(value)
    }

    fn tensor<'a>(&'a self, v: &'a Tensor) -> Result<&'a Tensor, OjasError> {
        Ok(v)
    }

    fn embedding(&mut self, table: &Tensor, ids: &Tensor) -> Result<Tensor, OjasError> {
        let ids = self.place(ids)?;
        self.backend.embedding_forward(table, &ids)
    }

    fn linear(&mut self, x: &Tensor, w: &Tensor) -> Result<Tensor, OjasError> {
        self.backend.linear_forward(x, w)
    }

    fn rms_norm(&mut self, x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor, OjasError> {
        self.backend.rms_norm_forward(x, w, eps)
    }

    fn rope(&mut self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, OjasError> {
        let cos = self.place(cos)?;
        let sin = self.place(sin)?;
        self.backend.rope_half_split_forward(x, &cos, &sin)
    }

    fn permute(&mut self, x: &Tensor, dims: &[usize]) -> Result<Tensor, OjasError> {
        self.backend.permute(x, dims)
    }

    fn sdpa(&mut self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, OjasError> {
        self.backend.causal_sdpa_forward(q, k, v)
    }

    fn cached_attn(
        &mut self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        kv_len: usize,
    ) -> Result<Tensor, OjasError> {
        let k = self.place(k_cache)?;
        let v = self.place(v_cache)?;
        self.backend.cached_attention_forward(q, &k, &v, kv_len)
    }

    fn gate(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        attn: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.backend.per_head_sigmoid_gate_forward(x, w, b, attn)
    }

    fn vres(&mut self, v: &Tensor, v0: &Tensor, lambda: &Tensor) -> Result<Tensor, OjasError> {
        self.backend.value_residual_blend_forward(v, v0, lambda)
    }

    fn silu(&mut self, x: &Tensor) -> Result<Tensor, OjasError> {
        self.backend.silu_forward(x)
    }

    fn mul(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        self.backend.mul_forward(a, b)
    }

    fn add(&mut self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        self.backend.residual_add_forward(x, y)
    }

    fn reshape(&mut self, x: &Tensor, shape: &[usize]) -> Result<Tensor, OjasError> {
        x.reshape(shape)
    }

    fn lin_ce(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        targets: &Tensor,
        ignore: Option<u32>,
        chunk: CeChunk,
    ) -> Result<Tensor, OjasError> {
        let targets = self.place(targets)?;
        let out = self
            .backend
            .linear_cross_entropy_mean(x, w, &targets, ignore, chunk, false)?;
        if out.loss.num_elements()? != 1 {
            return Err(OjasError::Backend {
                id: self.backend.id(),
                detail: format!(
                    "linear_cross_entropy_mean: loss shape {:?} is not a scalar",
                    out.loss.shape()
                ),
            });
        }
        Ok(out.loss)
    }
}
