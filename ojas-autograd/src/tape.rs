//! Reverse-mode tape. Each node stores the forward inputs and calls the
//! matching backward on `B`. The default backend is [`CpuBackend`].
//!
//! On a backend whose id is not [`BackendId::Cpu`], every tensor handed to
//! the tape is uploaded once and every value, seed and gradient stays where
//! that backend computes. Backward reads nothing back to the host.

use ojas_core::{inverse_permutation, permute_output_shape, Backend, BackendId, OjasError, Tensor};
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

    /// Ones in `shape`, contiguous, on the backend.
    fn ones(&self, shape: &[usize]) -> Result<Tensor, OjasError> {
        let n = shape_product("Tape::ones", shape)?;
        let host = Tensor::from_f32(&vec![1.0f32; n], shape, self.backend.budget())?;
        self.place(host)
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
        let y = self.backend.per_head_sigmoid_gate_forward(
            &self.values[x],
            &self.values[w],
            &self.values[b],
            &self.values[attn],
        )?;
        Ok(self.push(y, Rec::Gate { x, w, b, attn }))
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
        let y = reshape_view(&self.values[x], shape)?;
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

    /// Seed `var` with ones (gradient of the sum of its elements) and walk backward.
    ///
    /// Every call starts from empty gradients, so a repeated call or a call on
    /// an earlier node does not reuse gradients left by a previous walk. A
    /// failed walk leaves no gradients rather than a partial set.
    pub fn backward(&mut self, var: Var) -> Result<(), OjasError> {
        if var.0 >= self.ops.len() {
            return Err(OjasError::OutOfRange {
                op: "Tape::backward",
                detail: "variable is not on this tape".to_string(),
            });
        }
        let shape = self.values[var.0].shape().to_vec();
        let ones = self.ones(&shape)?;
        self.grads.iter_mut().for_each(|slot| *slot = None);
        self.grads[var.0] = Some(ones);
        for index in (0..=var.0).rev() {
            let Some(grad) = self.grads[index].take() else {
                continue;
            };
            let op = self.ops[index].clone();
            // Leaves keep the gradient a caller reads. Every other node's
            // gradient has been pushed into its inputs, so it is dropped.
            let retain = matches!(op, Rec::Leaf);
            // Nothing recorded after the root feeds it, so its gradient is
            // still the ones seed.
            let seeded = index == var.0;
            if let Err(err) = self.backward_one(op, &grad, seeded) {
                self.grads.iter_mut().for_each(|slot| *slot = None);
                return Err(err);
            }
            if retain {
                self.grads[index] = Some(grad);
            }
        }
        Ok(())
    }

    fn backward_one(&mut self, op: Rec, grad: &Tensor, seeded: bool) -> Result<(), OjasError> {
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
            Rec::Gate { x, w, b, attn } => {
                let xv = self.values[x].clone();
                let wv = self.values[w].clone();
                let bv = self.values[b].clone();
                let av = self.values[attn].clone();
                let g = self
                    .backend
                    .per_head_sigmoid_gate_backward(&xv, &wv, &bv, &av, grad)?;
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
                let gx = reshape_view(grad, &src_shape)?;
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
                if grad.num_elements()? != 1 {
                    return Err(OjasError::Shape {
                        op: "Tape::backward",
                        detail: "cross-entropy seed must be a scalar".to_string(),
                    });
                }
                let raw = self.backend.cross_entropy_mean_backward(
                    &self.values[logits],
                    &targets,
                    ignore,
                )?;
                let scaled = if self.backend.id() == BackendId::Cpu {
                    let seed = scalar_seed(grad)?;
                    let mut data = raw.to_f32_vec()?;
                    for value in &mut data {
                        *value *= seed;
                    }
                    Tensor::from_f32(&data, raw.shape(), self.backend.budget())?
                } else if seeded {
                    raw
                } else {
                    let seed = self.broadcast_scalar(grad, raw.shape())?;
                    self.backend.mul_forward(&raw, &seed)?
                };
                self.acc(logits, scaled)
            }
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
        let s = reshape_view(scalar, &[1, 1])?;
        let column = self.backend.linear_forward(&self.ones(&[cols, 1])?, &s)?;
        let full = self
            .backend
            .linear_forward(&self.ones(&[rows, 1])?, &column)?;
        reshape_view(&full, shape)
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
fn reshape_view(tensor: &Tensor, shape: &[usize]) -> Result<Tensor, OjasError> {
    let n = tensor.num_elements()?;
    let m = shape_product("Tape::reshape", shape)?;
    if n != m {
        return Err(OjasError::Shape {
            op: "Tape::reshape",
            detail: format!("reshape {n} elements into {shape:?}"),
        });
    }
    if !tensor.is_contiguous()? {
        return Err(OjasError::Shape {
            op: "Tape::reshape",
            detail: "view is not contiguous".to_string(),
        });
    }
    let mut strides = vec![1usize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    tensor.view(shape, &strides, tensor.byte_offset())
}

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
