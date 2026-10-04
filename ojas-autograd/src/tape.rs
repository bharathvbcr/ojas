//! Reverse-mode tape. Each node stores the forward inputs and calls the
//! matching backward on `B`. The default backend is [`CpuBackend`].
//!
//! On a backend whose id is not [`BackendId::Cpu`], every tensor handed to
//! the tape is uploaded once and every value, seed and gradient stays where
//! that backend computes. Backward reads nothing back to the host.

use ojas_core::{
    inverse_permutation, permute_output_shape, Backend, BackendId, CeChunk, OjasError, Tensor,
};
use ojas_cpu::CpuBackend;

/// Index of a value on a [`Tape`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(pub usize);

#[derive(Clone)]
enum Rec {
    Leaf,
    Embed {
        table: usize,
        ids: Tensor,
    },
    Linear {
        x: usize,
        w: usize,
    },
    Rms {
        x: usize,
        w: usize,
        eps: f32,
    },
    Rope {
        x: usize,
        cos: Tensor,
        sin: Tensor,
    },
    Sdpa {
        q: usize,
        k: usize,
        v: usize,
    },
    Gate {
        x: usize,
        w: usize,
        b: usize,
        attn: usize,
        /// Per-head sigmoid kept by a Fast forward. Exact stores nothing and
        /// recomputes the logits.
        scales: Option<Tensor>,
    },
    Vres {
        v: usize,
        v0: usize,
        lambda: usize,
    },
    Silu {
        x: usize,
    },
    Mul {
        a: usize,
        b: usize,
    },
    Add {
        x: usize,
        y: usize,
    },
    Ce {
        logits: usize,
        targets: Tensor,
        ignore: Option<u32>,
    },
    /// Fused `linear` → `cross_entropy` (`Backend::linear_cross_entropy_mean`).
    /// `grads` holds the seed-1 `(input, weight)` gradients from the forward
    /// call. Backward moves them out, so the leaf gradient it produces is
    /// not shared with this record; a later backward over the same node
    /// recomputes them through the backend from `targets`, `ignore` and
    /// `chunk`.
    LinearCe {
        x: usize,
        w: usize,
        targets: Tensor,
        ignore: Option<u32>,
        chunk: CeChunk,
        grads: Option<(Tensor, Tensor)>,
    },
    /// Views the gradient as `src_shape` on the way back.
    Reshape {
        src: usize,
        src_shape: Vec<usize>,
    },
    /// Permutes the gradient by `inverse` on the way back.
    Permute {
        src: usize,
        inverse: Vec<usize>,
    },
}

/// Records ops and backpropagates into leaf gradients.
///
/// `B` defaults to [`CpuBackend`]. A parallel or GPU backend can be passed
/// instead. Gradients are whatever that backend returns.
pub struct Tape<B: Backend = CpuBackend> {
    backend: B,
    values: Vec<Tensor>,
    grads: Vec<Option<Tensor>>,
    ops: Vec<Rec>,
}

impl<B: Backend> Tape<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            values: Vec::new(),
            grads: Vec::new(),
            ops: Vec::new(),
        }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Drop recorded values, gradients, and ops so the tape can be reused.
    /// The backend stays.
    pub fn clear(&mut self) {
        self.values.clear();
        self.grads.clear();
        self.ops.clear();
        self.values.shrink_to_fit();
        self.grads.shrink_to_fit();
        self.ops.shrink_to_fit();
    }

    /// Record a leaf. A host tensor on a device backend is uploaded first.
    pub fn leaf(&mut self, value: Tensor) -> Result<Var, OjasError> {
        let value = self.place(value)?;
        Ok(self.push(value, Rec::Leaf))
    }

    /// Make `tensor` resident on the backend. The CPU path keeps the tensor
    /// exactly as given.
    fn place(&self, tensor: Tensor) -> Result<Tensor, OjasError> {
        if self.backend.id() == BackendId::Cpu {
            Ok(tensor)
        } else {
            self.backend.upload(&tensor)
        }
    }

    /// `value` repeated over `shape`, contiguous, on the backend. Built on
    /// the host and placed, so nothing is read back.
    fn fill(&self, shape: &[usize], value: f32) -> Result<Tensor, OjasError> {
        let n = shape_product("Tape::fill", shape)?;
        let host = Tensor::from_f32(&vec![value; n], shape, self.backend.budget())?;
        self.place(host)
    }

    fn ones(&self, shape: &[usize]) -> Result<Tensor, OjasError> {
        self.fill(shape, 1.0)
    }

    pub fn value(&self, var: Var) -> Result<&Tensor, OjasError> {
        self.values.get(var.0).ok_or_else(|| OjasError::OutOfRange {
            op: "Tape::value",
            detail: "variable is not on this tape".to_string(),
        })
    }

    pub fn grad(&self, var: Var) -> Option<&Tensor> {
        self.grads.get(var.0).and_then(|slot| slot.as_ref())
    }

    /// Move `var`'s gradient out of the tape without copying it.
    ///
    /// The tape keeps no handle to the returned tensor, so a trainer can
    /// pass it to [`Backend::accumulate_grad`] or an in-place op and then
    /// [`Tape::clear`] the tape. The tensor is uniquely owned as long as the
    /// backend returned a separate allocation for each gradient output, as
    /// `CpuBackend` does. A second call, a call after a failed backward, and
    /// a variable that is not on this tape all return `None`.
    pub fn take_grad(&mut self, var: Var) -> Option<Tensor> {
        self.grads.get_mut(var.0).and_then(Option::take)
    }

    fn index(&self, var: Var, op: &'static str) -> Result<usize, OjasError> {
        if var.0 < self.values.len() {
            Ok(var.0)
        } else {
            Err(OjasError::OutOfRange {
                op,
                detail: "variable is not on this tape".to_string(),
            })
        }
    }

    pub fn embedding(&mut self, table: Var, ids: Tensor) -> Result<Var, OjasError> {
        let table = self.index(table, "Tape::embedding")?;
        let ids = self.place(ids)?;
        let y = self.backend.embedding_forward(&self.values[table], &ids)?;
        Ok(self.push(y, Rec::Embed { table, ids }))
    }

    pub fn linear(&mut self, x: Var, w: Var) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::linear")?;
        let w = self.index(w, "Tape::linear")?;
        let y = self
            .backend
            .linear_forward(&self.values[x], &self.values[w])?;
        Ok(self.push(y, Rec::Linear { x, w }))
    }

    pub fn rms_norm(&mut self, x: Var, w: Var, eps: f32) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::rms_norm")?;
        let w = self.index(w, "Tape::rms_norm")?;
        let y = self
            .backend
            .rms_norm_forward(&self.values[x], &self.values[w], eps)?;
        Ok(self.push(y, Rec::Rms { x, w, eps }))
    }

    pub fn rms_qk_norm(
        &mut self,
        q: Var,
        k: Var,
        q_weight: Var,
        k_weight: Var,
        eps: f32,
    ) -> Result<(Var, Var), OjasError> {
        let qn = self.rms_norm(q, q_weight, eps)?;
        let kn = self.rms_norm(k, k_weight, eps)?;
        Ok((qn, kn))
    }

    pub fn rope(&mut self, x: Var, cos: Tensor, sin: Tensor) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::rope")?;
        let cos = self.place(cos)?;
        let sin = self.place(sin)?;
        let y = self
            .backend
            .rope_half_split_forward(&self.values[x], &cos, &sin)?;
        Ok(self.push(y, Rec::Rope { x, cos, sin }))
    }

    pub fn causal_sdpa(&mut self, q: Var, k: Var, v: Var) -> Result<Var, OjasError> {
        let q = self.index(q, "Tape::causal_sdpa")?;
        let k = self.index(k, "Tape::causal_sdpa")?;
        let v = self.index(v, "Tape::causal_sdpa")?;
        let y =
            self.backend
                .causal_sdpa_forward(&self.values[q], &self.values[k], &self.values[v])?;
        Ok(self.push(y, Rec::Sdpa { q, k, v }))
    }

    pub fn per_head_gate(&mut self, x: Var, w: Var, b: Var, attn: Var) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::per_head_gate")?;
        let w = self.index(w, "Tape::per_head_gate")?;
        let b = self.index(b, "Tape::per_head_gate")?;
        let attn = self.index(attn, "Tape::per_head_gate")?;
        let (y, scales) = self.backend.per_head_sigmoid_gate_forward_saving(
            &self.values[x],
            &self.values[w],
            &self.values[b],
            &self.values[attn],
        )?;
        Ok(self.push(
            y,
            Rec::Gate {
                x,
                w,
                b,
                attn,
                scales,
            },
        ))
    }

    pub fn value_residual(&mut self, v: Var, v0: Var, lambda: Var) -> Result<Var, OjasError> {
        let v = self.index(v, "Tape::value_residual")?;
        let v0 = self.index(v0, "Tape::value_residual")?;
        let lambda = self.index(lambda, "Tape::value_residual")?;
        let y = self.backend.value_residual_blend_forward(
            &self.values[v],
            &self.values[v0],
            &self.values[lambda],
        )?;
        Ok(self.push(y, Rec::Vres { v, v0, lambda }))
    }

    pub fn silu(&mut self, x: Var) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::silu")?;
        let y = self.backend.silu_forward(&self.values[x])?;
        Ok(self.push(y, Rec::Silu { x }))
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Result<Var, OjasError> {
        let a = self.index(a, "Tape::mul")?;
        let b = self.index(b, "Tape::mul")?;
        let y = self.backend.mul_forward(&self.values[a], &self.values[b])?;
        Ok(self.push(y, Rec::Mul { a, b }))
    }

    pub fn add(&mut self, x: Var, y: Var) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::add")?;
        let y = self.index(y, "Tape::add")?;
        let z = self
            .backend
            .residual_add_forward(&self.values[x], &self.values[y])?;
        Ok(self.push(z, Rec::Add { x, y }))
    }

    /// View `x` as `shape`. The element count must match. Rank may change.
    /// `x` must be contiguous; no bytes are copied or read.
    pub fn reshape(&mut self, x: Var, shape: &[usize]) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::reshape")?;
        let src_shape = self.values[x].shape().to_vec();
        let y = self.values[x].reshape(shape)?;
        Ok(self.push(y, Rec::Reshape { src: x, src_shape }))
    }

    /// Reorder the axes of `x`: output axis `i` is input axis `dims[i]`, as
    /// in `torch.permute`. The result is a contiguous copy made by the
    /// backend, which is how `[B, T, H, D]` reaches `[B, H, T, D]` and back.
    /// Backward permutes the gradient by [`inverse_permutation`]`(dims)`.
    /// Invalid `dims` are refused by the backend and nothing is recorded.
    pub fn permute(&mut self, x: Var, dims: &[usize]) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::permute")?;
        // Checked here too, so `inverse_permutation` cannot index out of
        // range whatever the backend validates.
        permute_output_shape("Tape::permute", self.values[x].shape(), dims)?;
        let y = self.backend.permute(&self.values[x], dims)?;
        let inverse = inverse_permutation(dims);
        Ok(self.push(y, Rec::Permute { src: x, inverse }))
    }

    pub fn cross_entropy(
        &mut self,
        logits: Var,
        targets: Tensor,
        ignore: Option<u32>,
    ) -> Result<Var, OjasError> {
        let logits = self.index(logits, "Tape::cross_entropy")?;
        let targets = self.place(targets)?;
        let y = self
            .backend
            .cross_entropy_mean_forward(&self.values[logits], &targets, ignore)?;
        Ok(self.push(
            y,
            Rec::Ce {
                logits,
                targets,
                ignore,
            },
        ))
    }

    /// Mean cross-entropy of `linear(x, w)` through the backend's fused
    /// [`Backend::linear_cross_entropy_mean`], which never holds the full
    /// `[N, V]` logits.
    ///
    /// `x` is `[N, d]`, `w` is `[V, d]` (the tied embedding), `targets` is
    /// `[N]` `U32`. The forward call asks for both gradients, and backward
    /// only scales them by the upstream gradient (the seed of
    /// [`Tape::backward_seeded`] when the loss is the root). A backend that
    /// does not implement the op returns its error, normally
    /// [`OjasError::Unsupported`]; this method does not fall back to the
    /// unfused path, and records nothing on any error.
    pub fn linear_cross_entropy(
        &mut self,
        x: Var,
        w: Var,
        targets: Tensor,
        ignore: Option<u32>,
        chunk: CeChunk,
    ) -> Result<Var, OjasError> {
        const OP: &str = "Tape::linear_cross_entropy";
        let x = self.index(x, OP)?;
        let w = self.index(w, OP)?;
        let targets = self.place(targets)?;
        let (loss, grads) = self.fused_ce(x, w, &targets, ignore, chunk)?;
        Ok(self.push(
            loss,
            Rec::LinearCe {
                x,
                w,
                targets,
                ignore,
                chunk,
                grads: Some(grads),
            },
        ))
    }

    /// One fused call with `want_grad` set, checked against the trait
    /// contract: a scalar loss, and both gradients present and shaped like
    /// their inputs.
    fn fused_ce(
        &self,
        x: usize,
        w: usize,
        targets: &Tensor,
        ignore: Option<u32>,
        chunk: CeChunk,
    ) -> Result<(Tensor, (Tensor, Tensor)), OjasError> {
        let (xv, wv) = (&self.values[x], &self.values[w]);
        let out = self
            .backend
            .linear_cross_entropy_mean(xv, wv, targets, ignore, chunk, true)?;
        let contract = |detail: String| OjasError::Backend {
            id: self.backend.id(),
            detail: format!("linear_cross_entropy_mean: {detail}"),
        };
        let (Some(gx), Some(gw)) = (out.grad_input, out.grad_weight) else {
            return Err(contract(
                "returned no gradient although want_grad was set".to_string(),
            ));
        };
        if gx.shape() != xv.shape() || gw.shape() != wv.shape() {
            return Err(contract(format!(
                "gradients {:?} and {:?} do not match input {:?} and weight {:?}",
                gx.shape(),
                gw.shape(),
                xv.shape(),
                wv.shape()
            )));
        }
        if out.loss.num_elements()? != 1 {
            return Err(contract(format!(
                "loss shape {:?} is not a scalar",
                out.loss.shape()
            )));
        }
        Ok((out.loss, (gx, gw)))
    }

    /// [`Tape::backward_seeded`] with seed 1: the gradient of the sum of
    /// `var`'s elements.
    pub fn backward(&mut self, var: Var) -> Result<(), OjasError> {
        self.backward_seeded(var, 1.0)
    }

    /// Seed `var` with `seed` in every element and walk backward, so each
    /// leaf gradient is `seed` times the gradient of the sum of `var`'s
    /// elements. A training loop that averages K micro-batches passes
    /// `1 / K` (§3 of `docs/framework-design.md`).
    ///
    /// `seed` must be a positive normal `f32`. A NaN or infinite seed is
    /// [`OjasError::NonFinite`]. Zero, a negative value and a subnormal are
    /// [`OjasError::OutOfRange`]: `1 / K` is never any of them, a zero seed
    /// is the usual sign of an integer `1 / K`, and a subnormal seed would
    /// flush gradients toward zero rather than scale them.
    ///
    /// Every call starts from empty gradients, so a repeated call or a call
    /// on an earlier node does not reuse gradients left by a previous walk.
    /// Any error, including a refused seed or a variable that is not on
    /// this tape, leaves no gradients rather than a partial or stale set.
    pub fn backward_seeded(&mut self, var: Var, seed: f32) -> Result<(), OjasError> {
        self.grads.iter_mut().for_each(|slot| *slot = None);
        check_seed(seed)?;
        if var.0 >= self.ops.len() {
            return Err(OjasError::OutOfRange {
                op: "Tape::backward",
                detail: "variable is not on this tape".to_string(),
            });
        }
        let shape = self.values[var.0].shape().to_vec();
        self.grads[var.0] = Some(self.fill(&shape, seed)?);
        for index in (0..=var.0).rev() {
            let Some(grad) = self.grads[index].take() else {
                continue;
            };
            let op = self.op_for_backward(index);
            // Leaves keep the gradient a caller reads. Every other node's
            // gradient has been pushed into its inputs, so it is dropped.
            let retain = matches!(op, Rec::Leaf);
            // Nothing recorded after the root feeds it, so its gradient is
            // still the seed. Only a seed of exactly 1 lets a loss hand its
            // unscaled gradient straight through.
            let unit_root = index == var.0 && seed == 1.0;
            if let Err(err) = self.backward_one(op, &grad, unit_root) {
                self.grads.iter_mut().for_each(|slot| *slot = None);
                return Err(err);
            }
            if retain {
                self.grads[index] = Some(grad);
            }
        }
        Ok(())
    }

    /// The record at `index`, for one backward step. A fused cross-entropy's
    /// stored gradients are moved out rather than shared, so the gradient
    /// it hands to a leaf has no other owner.
    fn op_for_backward(&mut self, index: usize) -> Rec {
        let mut op = self.ops[index].clone();
        if let (Rec::LinearCe { grads: stored, .. }, Rec::LinearCe { grads, .. }) =
            (&mut self.ops[index], &mut op)
        {
            *grads = stored.take();
        }
        op
    }

    fn backward_one(&mut self, op: Rec, grad: &Tensor, unit_root: bool) -> Result<(), OjasError> {
        match op {
            Rec::Leaf => Ok(()),
            Rec::Embed { table, ids } => {
                let gx = self
                    .backend
                    .embedding_backward(&self.values[table], &ids, grad)?;
                self.acc(table, gx)
            }
            Rec::Linear { x, w } => {
                let xv = self.values[x].clone();
                let wv = self.values[w].clone();
                let (gx, gw) = self.backend.linear_backward(&xv, &wv, grad)?;
                self.acc(x, gx)?;
                self.acc(w, gw)
            }
            Rec::Rms { x, w, eps } => {
                let xv = self.values[x].clone();
                let wv = self.values[w].clone();
                let (gx, gw) = self.backend.rms_norm_backward(&xv, &wv, grad, eps)?;
                self.acc(x, gx)?;
                self.acc(w, gw)
            }
            Rec::Rope { x, cos, sin } => {
                let gx = self.backend.rope_half_split_backward(grad, &cos, &sin)?;
                self.acc(x, gx)
            }
            Rec::Sdpa { q, k, v } => {
                let qv = self.values[q].clone();
                let kv = self.values[k].clone();
                let vv = self.values[v].clone();
                let (gq, gk, gv) = self.backend.causal_sdpa_backward(&qv, &kv, &vv, grad)?;
                self.acc(q, gq)?;
                self.acc(k, gk)?;
                self.acc(v, gv)
            }
            Rec::Gate {
                x,
                w,
                b,
                attn,
                scales,
            } => {
                let xv = self.values[x].clone();
                let wv = self.values[w].clone();
                let bv = self.values[b].clone();
                let av = self.values[attn].clone();
                let g = if let Some(scales) = scales {
                    self.backend
                        .per_head_sigmoid_gate_backward_saved(&xv, &wv, &bv, &av, grad, &scales)?
                } else {
                    self.backend
                        .per_head_sigmoid_gate_backward(&xv, &wv, &bv, &av, grad)?
                };
                self.acc(x, g.input)?;
                self.acc(w, g.weight)?;
                self.acc(b, g.bias)?;
                self.acc(attn, g.attn_out)
            }
            Rec::Vres { v, v0, lambda } => {
                let vv = self.values[v].clone();
                let v0v = self.values[v0].clone();
                let lv = self.values[lambda].clone();
                let g = self
                    .backend
                    .value_residual_blend_backward(&vv, &v0v, &lv, grad)?;
                self.acc(v, g.value)?;
                self.acc(v0, g.value0)?;
                self.acc(lambda, g.lambda)
            }
            Rec::Silu { x } => {
                let xv = self.values[x].clone();
                let gx = self.backend.silu_backward(&xv, grad)?;
                self.acc(x, gx)
            }
            Rec::Mul { a, b } => {
                let av = self.values[a].clone();
                let bv = self.values[b].clone();
                let (ga, gb) = self.backend.mul_backward(&av, &bv, grad)?;
                self.acc(a, ga)?;
                self.acc(b, gb)
            }
            Rec::Add { x, y } => {
                let xv = self.values[x].clone();
                let yv = self.values[y].clone();
                let (gx, gy) = self.backend.residual_add_backward(&xv, &yv, grad)?;
                self.acc(x, gx)?;
                self.acc(y, gy)
            }
            Rec::Reshape { src, src_shape } => {
                let gx = grad.reshape(&src_shape)?;
                self.acc(src, gx)
            }
            Rec::Permute { src, inverse } => {
                let gx = self.backend.permute(grad, &inverse)?;
                self.acc(src, gx)
            }
            Rec::Ce {
                logits,
                targets,
                ignore,
            } => {
                require_scalar(grad)?;
                let raw = self.backend.cross_entropy_mean_backward(
                    &self.values[logits],
                    &targets,
                    ignore,
                )?;
                let scaled = self.scale_loss_grad(raw, grad, unit_root)?;
                self.acc(logits, scaled)
            }
            Rec::LinearCe {
                x,
                w,
                targets,
                ignore,
                chunk,
                grads,
            } => {
                require_scalar(grad)?;
                let (gx, gw) = match grads {
                    Some(grads) => grads,
                    None => self.fused_ce(x, w, &targets, ignore, chunk)?.1,
                };
                let gx = self.scale_loss_grad(gx, grad, unit_root)?;
                let gw = self.scale_loss_grad(gw, grad, unit_root)?;
                self.acc(x, gx)?;
                self.acc(w, gw)
            }
        }
    }

    /// A loss's seed-1 gradient `raw` times the scalar upstream gradient
    /// `grad`. `raw` must be owned by the caller alone; at a root seeded with
    /// exactly 1 it is returned as is. Otherwise the CPU path multiplies on
    /// the host, and a device path multiplies by `grad` broadcast on the
    /// device, so nothing is read back. Either way the result is a new
    /// allocation.
    fn scale_loss_grad(
        &self,
        raw: Tensor,
        grad: &Tensor,
        unit_root: bool,
    ) -> Result<Tensor, OjasError> {
        if unit_root {
            Ok(raw)
        } else if self.backend.id() == BackendId::Cpu {
            let seed = scalar_seed(grad)?;
            let mut data = raw.to_f32_vec()?;
            for value in &mut data {
                *value *= seed;
            }
            Tensor::from_f32(&data, raw.shape(), self.backend.budget())
        } else {
            let seed = self.broadcast_scalar(grad, raw.shape())?;
            self.backend.mul_forward(&raw, &seed)
        }
    }

    /// `scalar` repeated over `shape`, built on the backend without reading
    /// `scalar` back: `ones[cols, 1] @ s[1, 1]^T` is a `[cols, 1]` column of
    /// `s`, and `ones[rows, 1] @ column^T` is `[rows, cols]`. Each output is
    /// one product with 1, so it is `s` exactly.
    fn broadcast_scalar(&self, scalar: &Tensor, shape: &[usize]) -> Result<Tensor, OjasError> {
        let n = shape_product("Tape::backward", shape)?;
        let cols = *shape.last().ok_or_else(|| OjasError::Shape {
            op: "Tape::backward",
            detail: "cannot broadcast a scalar over rank 0".to_string(),
        })?;
        let rows = n / cols;
        let s = scalar.reshape(&[1, 1])?;
        let column = self.backend.linear_forward(&self.ones(&[cols, 1])?, &s)?;
        let full = self
            .backend
            .linear_forward(&self.ones(&[rows, 1])?, &column)?;
        full.reshape(shape)
    }

    fn push(&mut self, value: Tensor, op: Rec) -> Var {
        let id = self.values.len();
        self.values.push(value);
        self.grads.push(None);
        self.ops.push(op);
        Var(id)
    }

    fn acc(&mut self, id: usize, grad: Tensor) -> Result<(), OjasError> {
        if let Some(old) = &self.grads[id] {
            let sum = self.backend.residual_add_forward(old, &grad)?;
            self.grads[id] = Some(sum);
        } else {
            self.grads[id] = Some(grad);
        }
        Ok(())
    }
}

/// `tensor` as `shape` over the same storage. Reads no bytes, so a device
/// tensor stays on its device.
fn shape_product(op: &'static str, shape: &[usize]) -> Result<usize, OjasError> {
    if shape.contains(&0) {
        return Err(OjasError::Shape {
            op,
            detail: "empty tensor".to_string(),
        });
    }
    let mut n = 1usize;
    for dim in shape {
        n = n.checked_mul(*dim).ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "shape product overflows".to_string(),
        })?;
    }
    if n == 0 {
        return Err(OjasError::Shape {
            op,
            detail: "empty tensor".to_string(),
        });
    }
    Ok(n)
}

/// The policy in [`Tape::backward_seeded`]: a positive normal `f32`.
fn check_seed(seed: f32) -> Result<(), OjasError> {
    const OP: &str = "Tape::backward_seeded";
    if !seed.is_finite() {
        return Err(OjasError::NonFinite { op: OP });
    }
    if !(seed.is_normal() && seed > 0.0) {
        return Err(OjasError::OutOfRange {
            op: OP,
            detail: format!("seed {seed:e} must be a positive normal f32"),
        });
    }
    Ok(())
}

/// A loss is a scalar, so the gradient flowing into it must be one too.
fn require_scalar(grad: &Tensor) -> Result<(), OjasError> {
    if grad.num_elements()? == 1 {
        Ok(())
    } else {
        Err(OjasError::Shape {
            op: "Tape::backward",
            detail: "cross-entropy seed must be a scalar".to_string(),
        })
    }
}

fn scalar_seed(grad: &Tensor) -> Result<f32, OjasError> {
    let data = grad.to_f32_vec()?;
    if data.len() != 1 {
        return Err(OjasError::Shape {
            op: "Tape::backward",
            detail: "cross-entropy seed must be a scalar".to_string(),
        });
    }
    Ok(data[0])
}
