//! Reverse-mode tape. Each node stores the forward inputs and calls the
//! matching backward on `B`. The default backend is [`CpuBackend`].

use ojas_core::{Backend, OjasError, Tensor};
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
    /// Copies values into `src_shape` on the way back. Forward already copied
    /// into the destination shape, so the bytes are not a view of the source.
    Reshape {
        src: usize,
        src_shape: Vec<usize>,
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

    pub fn leaf(&mut self, value: Tensor) -> Var {
        self.push(value, Rec::Leaf)
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

    /// Copy `x` into `shape`. The element count must match. Rank may change.
    pub fn reshape(&mut self, x: Var, shape: &[usize]) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::reshape")?;
        let src_shape = self.values[x].shape().to_vec();
        let n = self.values[x].num_elements()?;
        let m = shape_product("Tape::reshape", shape)?;
        if n != m {
            return Err(OjasError::Shape {
                op: "Tape::reshape",
                detail: format!("reshape {n} elements into {shape:?}"),
            });
        }
        let data = self.values[x].to_f32_vec()?;
        let y = Tensor::from_f32(&data, shape, self.backend.budget())?;
        Ok(self.push(y, Rec::Reshape { src: x, src_shape }))
    }

    pub fn cross_entropy(
        &mut self,
        logits: Var,
        targets: Tensor,
        ignore: Option<u32>,
    ) -> Result<Var, OjasError> {
        let logits = self.index(logits, "Tape::cross_entropy")?;
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
        let ones = ones_like(&self.backend, &self.values[var.0])?;
        self.grads.iter_mut().for_each(|slot| *slot = None);
        self.grads[var.0] = Some(ones);
        for index in (0..=var.0).rev() {
            let Some(grad) = self.grads[index].clone() else {
                continue;
            };
            let op = self.ops[index].clone();
            if let Err(err) = self.backward_one(op, &grad) {
                self.grads.iter_mut().for_each(|slot| *slot = None);
                return Err(err);
            }
        }
        Ok(())
    }

    fn backward_one(&mut self, op: Rec, grad: &Tensor) -> Result<(), OjasError> {
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
                let data = grad.to_f32_vec()?;
                if data.len() != shape_product("Tape::reshape", &src_shape)? {
                    return Err(OjasError::Shape {
                        op: "Tape::reshape",
                        detail: "reshape grad length does not match the source".to_string(),
                    });
                }
                let gx = Tensor::from_f32(&data, &src_shape, self.backend.budget())?;
                self.acc(src, gx)
            }
            Rec::Ce {
                logits,
                targets,
                ignore,
            } => {
                let seed = scalar_seed(grad)?;
                let raw = self.backend.cross_entropy_mean_backward(
                    &self.values[logits],
                    &targets,
                    ignore,
                )?;
                let mut data = raw.to_f32_vec()?;
                for value in &mut data {
                    *value *= seed;
                }
                let scaled = Tensor::from_f32(&data, raw.shape(), self.backend.budget())?;
                self.acc(logits, scaled)
            }
        }
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

fn ones_like(backend: &impl Backend, tensor: &Tensor) -> Result<Tensor, OjasError> {
    let n = tensor.num_elements()?;
    let data = vec![1.0f32; n];
    Tensor::from_f32(&data, tensor.shape(), backend.budget())
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
