//! Reverse-mode tape. Each node stores the forward inputs and calls the
//! matching backward on `B`. The default backend is [`CpuBackend`].
//!
//! On a backend whose id is not [`BackendId::Cpu`], every tensor handed to
//! the tape is uploaded once and every value, seed and gradient stays where
//! that backend computes. Backward reads nothing back to the host.
//!
//! [`Tape::checkpoint`] keeps a segment's records but not its values, and the
//! backward walk recomputes them from those records (activation
//! checkpointing).

use ojas_core::{
    inverse_permutation, permute_output_shape, Backend, BackendId, CeChunk, GdnInputs, OjasError,
    Tensor,
};
use ojas_cpu::CpuBackend;

/// Index of a value on a [`Tape`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(pub usize);

/// One recorded op: its input indices and whatever its backward needs.
///
/// [`Tape::run_op`] computes a record's value from its inputs, so the same
/// record is both how an op is first recorded and how a checkpointed
/// segment is recomputed. Tensors a forward returns beside its value
/// (`lse`, `scales`, `checkpoints`, `grads`) are `None` until it runs; a
/// checkpointed segment keeps its records with them dropped.
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
    /// `out` is this node: its value is the forward output the backward
    /// reads, next to the row log-sum-exp `lse` the forward returned.
    Sdpa {
        q: usize,
        k: usize,
        v: usize,
        out: usize,
        lse: Option<Tensor>,
        window: Option<usize>,
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
    /// The gated delta rule from a zero initial state. `checkpoints` is the
    /// state every 64 tokens that the forward returned, kept where the
    /// backend computes; the backward recomputes each chunk from it.
    Gdn {
        q: usize,
        k: usize,
        v: usize,
        g: usize,
        beta: usize,
        checkpoints: Option<Tensor>,
    },
    Silu {
        x: usize,
    },
    /// Depthwise causal conv + SiLU from a zero state.
    Conv1d {
        x: usize,
        w: usize,
    },
    /// Gated RMSNorm of `x` with gate `z` and weight `w`.
    GatedRms {
        x: usize,
        z: usize,
        w: usize,
        eps: f32,
    },
    /// Partial RoPE: the leading `cos.shape()[1]` values of each head turn.
    RopePartial {
        x: usize,
        cos: Tensor,
        sin: Tensor,
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
    /// `src` viewed as `shape`; the gradient is viewed as `src_shape` on the
    /// way back.
    Reshape {
        src: usize,
        src_shape: Vec<usize>,
        shape: Vec<usize>,
    },
    /// `src` permuted by `inverse_permutation(inverse)`; the gradient is
    /// permuted by `inverse` on the way back.
    Permute {
        src: usize,
        inverse: Vec<usize>,
    },
    /// Output `part` of checkpointed segment `seg` ([`Tape::checkpoint`]).
    /// Its value is the segment's output; nothing inside the segment is
    /// kept. Part 0 has the lowest index of the segment's outputs, so when
    /// the walk reaches it every output's gradient is final and the segment
    /// is recomputed and backpropagated.
    Checkpoint {
        seg: usize,
        part: usize,
    },
}

impl Rec {
    /// The same record with every node index `i >= from` moved to
    /// `i - from + to`: a segment recorded from `from`, replayed from `to`.
    fn moved(self, from: usize, to: usize) -> Self {
        let m = |i: usize| if i >= from { i - from + to } else { i };
        match self {
            Rec::Leaf => Rec::Leaf,
            Rec::Embed { table, ids } => Rec::Embed {
                table: m(table),
                ids,
            },
            Rec::Linear { x, w } => Rec::Linear { x: m(x), w: m(w) },
            Rec::Rms { x, w, eps } => Rec::Rms {
                x: m(x),
                w: m(w),
                eps,
            },
            Rec::Rope { x, cos, sin } => Rec::Rope { x: m(x), cos, sin },
            Rec::Sdpa {
                q,
                k,
                v,
                out,
                lse,
                window,
            } => Rec::Sdpa {
                q: m(q),
                k: m(k),
                v: m(v),
                out: m(out),
                lse,
                window,
            },
            Rec::Gate {
                x,
                w,
                b,
                attn,
                scales,
            } => Rec::Gate {
                x: m(x),
                w: m(w),
                b: m(b),
                attn: m(attn),
                scales,
            },
            Rec::Vres { v, v0, lambda } => Rec::Vres {
                v: m(v),
                v0: m(v0),
                lambda: m(lambda),
            },
            Rec::Gdn {
                q,
                k,
                v,
                g,
                beta,
                checkpoints,
            } => Rec::Gdn {
                q: m(q),
                k: m(k),
                v: m(v),
                g: m(g),
                beta: m(beta),
                checkpoints,
            },
            Rec::Silu { x } => Rec::Silu { x: m(x) },
            Rec::Conv1d { x, w } => Rec::Conv1d { x: m(x), w: m(w) },
            Rec::GatedRms { x, z, w, eps } => Rec::GatedRms {
                x: m(x),
                z: m(z),
                w: m(w),
                eps,
            },
            Rec::RopePartial { x, cos, sin } => Rec::RopePartial { x: m(x), cos, sin },
            Rec::Mul { a, b } => Rec::Mul { a: m(a), b: m(b) },
            Rec::Add { x, y } => Rec::Add { x: m(x), y: m(y) },
            Rec::Ce {
                logits,
                targets,
                ignore,
            } => Rec::Ce {
                logits: m(logits),
                targets,
                ignore,
            },
            Rec::LinearCe {
                x,
                w,
                targets,
                ignore,
                chunk,
                grads,
            } => Rec::LinearCe {
                x: m(x),
                w: m(w),
                targets,
                ignore,
                chunk,
                grads,
            },
            Rec::Reshape {
                src,
                src_shape,
                shape,
            } => Rec::Reshape {
                src: m(src),
                src_shape,
                shape,
            },
            Rec::Permute { src, inverse } => Rec::Permute {
                src: m(src),
                inverse,
            },
            Rec::Checkpoint { seg, part } => Rec::Checkpoint { seg, part },
        }
    }

    /// The record without the tensors its forward returned beside its value,
    /// which a replay computes again.
    fn stripped(mut self) -> Self {
        match &mut self {
            Rec::Sdpa { lse, .. } => *lse = None,
            Rec::Gate { scales, .. } => *scales = None,
            Rec::Gdn { checkpoints, .. } => *checkpoints = None,
            Rec::LinearCe { grads, .. } => *grads = None,
            _ => {}
        }
        self
    }
}

/// A checkpointed segment: its records as they were made from node `start`
/// on, which of those nodes are its outputs, and the output gradients the
/// backward walk gathers until part 0 replays it.
struct Segment {
    start: usize,
    ops: Vec<Rec>,
    /// Offsets of the outputs from `start`.
    outputs: Vec<usize>,
    /// One slot per output.
    pending: Vec<Option<Tensor>>,
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
    segments: Vec<Segment>,
    /// Set while [`Tape::checkpoint`] records a segment, where a nested
    /// checkpoint is refused.
    in_segment: bool,
}

impl<B: Backend> Tape<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            values: Vec::new(),
            grads: Vec::new(),
            ops: Vec::new(),
            segments: Vec::new(),
            in_segment: false,
        }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Drop recorded values, gradients, ops and checkpointed segments so the
    /// tape can be reused. The backend stays.
    pub fn clear(&mut self) {
        self.values.clear();
        self.grads.clear();
        self.ops.clear();
        self.segments.clear();
        self.values.shrink_to_fit();
        self.grads.shrink_to_fit();
        self.ops.shrink_to_fit();
        self.segments.shrink_to_fit();
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
        self.record(Rec::Embed { table, ids })
    }

    pub fn linear(&mut self, x: Var, w: Var) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::linear")?;
        let w = self.index(w, "Tape::linear")?;
        self.record(Rec::Linear { x, w })
    }

    pub fn rms_norm(&mut self, x: Var, w: Var, eps: f32) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::rms_norm")?;
        let w = self.index(w, "Tape::rms_norm")?;
        self.record(Rec::Rms { x, w, eps })
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
        self.record(Rec::Rope { x, cos, sin })
    }

    /// Causal attention ([`Backend::causal_sdpa_forward`]): `q` `[B, H, T, D]`,
    /// `k` and `v` `[B, Hkv, T, D]`, and an optional sliding `window`. The
    /// row log-sum-exp is kept on the tape for the backward.
    pub fn causal_sdpa(
        &mut self,
        q: Var,
        k: Var,
        v: Var,
        window: Option<usize>,
    ) -> Result<Var, OjasError> {
        let q = self.index(q, "Tape::causal_sdpa")?;
        let k = self.index(k, "Tape::causal_sdpa")?;
        let v = self.index(v, "Tape::causal_sdpa")?;
        self.record(Rec::Sdpa {
            q,
            k,
            v,
            out: self.values.len(),
            lse: None,
            window,
        })
    }

    pub fn per_head_gate(&mut self, x: Var, w: Var, b: Var, attn: Var) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::per_head_gate")?;
        let w = self.index(w, "Tape::per_head_gate")?;
        let b = self.index(b, "Tape::per_head_gate")?;
        let attn = self.index(attn, "Tape::per_head_gate")?;
        self.record(Rec::Gate {
            x,
            w,
            b,
            attn,
            scales: None,
        })
    }

    pub fn value_residual(&mut self, v: Var, v0: Var, lambda: Var) -> Result<Var, OjasError> {
        let v = self.index(v, "Tape::value_residual")?;
        let v0 = self.index(v0, "Tape::value_residual")?;
        let lambda = self.index(lambda, "Tape::value_residual")?;
        self.record(Rec::Vres { v, v0, lambda })
    }

    /// The gated delta rule (`Backend::chunked_gdn_forward`) from a zero
    /// initial state: `q`, `k` `[B, T, H, Dk]`, `v` `[B, T, H, Dv]`, `g` (log
    /// decay) and `beta` `[B, T, H]`. The result is the output `[B, T, H,
    /// Dv]`; the final state is dropped. The record keeps the forward's
    /// checkpoints, `B * H * ceil(T / 64) * Dk * Dv` values, for the
    /// backward.
    pub fn chunked_gdn(
        &mut self,
        q: Var,
        k: Var,
        v: Var,
        g: Var,
        beta: Var,
    ) -> Result<Var, OjasError> {
        const OP: &str = "Tape::chunked_gdn";
        let (q, k, v, g, beta) = (
            self.index(q, OP)?,
            self.index(k, OP)?,
            self.index(v, OP)?,
            self.index(g, OP)?,
            self.index(beta, OP)?,
        );
        self.record(Rec::Gdn {
            q,
            k,
            v,
            g,
            beta,
            checkpoints: None,
        })
    }

    /// Depthwise causal conv over time then SiLU
    /// ([`Backend::causal_conv1d_silu_forward`]): `x` `[B, T, C]`, `w`
    /// `[C, K]`, from a zero state.
    pub fn causal_conv1d_silu(&mut self, x: Var, w: Var) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::causal_conv1d_silu")?;
        let w = self.index(w, "Tape::causal_conv1d_silu")?;
        self.record(Rec::Conv1d { x, w })
    }

    /// Gated RMSNorm ([`Backend::gated_rms_norm_forward`]): `x` normalized
    /// over its last axis, times `w`, times `silu(z)`.
    pub fn gated_rms_norm(&mut self, x: Var, z: Var, w: Var, eps: f32) -> Result<Var, OjasError> {
        const OP: &str = "Tape::gated_rms_norm";
        let (x, z, w) = (self.index(x, OP)?, self.index(z, OP)?, self.index(w, OP)?);
        self.record(Rec::GatedRms { x, z, w, eps })
    }

    /// Partial RoPE ([`Backend::rope_partial_forward`]): `x` `[B, T, H, D]`,
    /// `cos` and `sin` `[T, R]`; the leading `R` of each head turn. Text-only
    /// MRoPE tables come from `ojas_core::mrope_text_tables`.
    pub fn rope_partial(&mut self, x: Var, cos: Tensor, sin: Tensor) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::rope_partial")?;
        let cos = self.place(cos)?;
        let sin = self.place(sin)?;
        self.record(Rec::RopePartial { x, cos, sin })
    }

    pub fn silu(&mut self, x: Var) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::silu")?;
        self.record(Rec::Silu { x })
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Result<Var, OjasError> {
        let a = self.index(a, "Tape::mul")?;
        let b = self.index(b, "Tape::mul")?;
        self.record(Rec::Mul { a, b })
    }

    pub fn add(&mut self, x: Var, y: Var) -> Result<Var, OjasError> {
        let x = self.index(x, "Tape::add")?;
        let y = self.index(y, "Tape::add")?;
        self.record(Rec::Add { x, y })
    }

    /// View `x` as `shape`. The element count must match. Rank may change.
    /// `x` must be contiguous; no bytes are copied or read.
    pub fn reshape(&mut self, x: Var, shape: &[usize]) -> Result<Var, OjasError> {
        let src = self.index(x, "Tape::reshape")?;
        let src_shape = self.values[src].shape().to_vec();
        self.record(Rec::Reshape {
            src,
            src_shape,
            shape: shape.to_vec(),
        })
    }

    /// Reorder the axes of `x`: output axis `i` is input axis `dims[i]`, as
    /// in `torch.permute`. The result is a contiguous copy made by the
    /// backend, which is how `[B, T, H, D]` reaches `[B, H, T, D]` and back.
    /// Backward permutes the gradient by [`inverse_permutation`]`(dims)`.
    /// Invalid `dims` are refused by the backend and nothing is recorded.
    pub fn permute(&mut self, x: Var, dims: &[usize]) -> Result<Var, OjasError> {
        let src = self.index(x, "Tape::permute")?;
        // Checked here too, so `inverse_permutation` cannot index out of
        // range whatever the backend validates.
        permute_output_shape("Tape::permute", self.values[src].shape(), dims)?;
        let inverse = inverse_permutation(dims);
        self.record(Rec::Permute { src, inverse })
    }

    pub fn cross_entropy(
        &mut self,
        logits: Var,
        targets: Tensor,
        ignore: Option<u32>,
    ) -> Result<Var, OjasError> {
        let logits = self.index(logits, "Tape::cross_entropy")?;
        let targets = self.place(targets)?;
        self.record(Rec::Ce {
            logits,
            targets,
            ignore,
        })
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
        self.record(Rec::LinearCe {
            x,
            w,
            targets,
            ignore,
            chunk,
            grads: None,
        })
    }

    /// Record `body`'s ops as one checkpointed segment: the tape keeps the
    /// outputs `body` returns and the records of every op it made, but none
    /// of their other values. When the backward walk reaches the segment it
    /// recomputes those values from the records, backpropagates through
    /// them, and drops them (activation checkpointing).
    ///
    /// The forward then holds a segment's outputs instead of every value in
    /// it, and the backward holds one segment's values at a time; each
    /// segment's forward runs twice. The records keep their inputs' indices
    /// and the small tensors they were given (token ids, RoPE tables,
    /// targets); the tensors a forward returns beside its value
    /// (attention's `lse`, the gate's `scales`, GDN state checkpoints, fused
    /// cross-entropy gradients) are dropped and recomputed.
    ///
    /// The returned variables stand for `body`'s outputs, in order; the
    /// variables `body` made are gone, so use only the returned ones. On a
    /// backend whose ops give the same bits for the same inputs (every
    /// backend under `Numerics::Exact`, and the CPU under either numerics)
    /// the gradients are equal bit for bit to recording `body` directly:
    /// the replay makes the same records in the same order, and each input
    /// receives its contributions in the order the direct walk adds them.
    ///
    /// Refused, with the tape left as it was before the call: an error from
    /// `body`, no outputs, an output `body` did not record or returns twice,
    /// a leaf recorded inside `body` (it could not be recomputed), and a
    /// checkpoint inside `body`.
    pub fn checkpoint<F>(&mut self, body: F) -> Result<Vec<Var>, OjasError>
    where
        F: FnOnce(&mut Self) -> Result<Vec<Var>, OjasError>,
    {
        const OP: &str = "Tape::checkpoint";
        if self.in_segment {
            return Err(OjasError::Unsupported {
                op: OP,
                detail: "a checkpoint inside a checkpointed segment".to_string(),
            });
        }
        let start = self.values.len();
        self.in_segment = true;
        let recorded = body(self);
        self.in_segment = false;
        let outputs = match recorded.and_then(|outs| self.segment_outputs(OP, start, &outs)) {
            Ok(outputs) => outputs,
            Err(err) => {
                self.truncate(start);
                return Err(err);
            }
        };
        let kept: Vec<Tensor> = outputs.iter().map(|&o| self.values[o].clone()).collect();
        let ops = self.ops.drain(start..).map(Rec::stripped).collect();
        self.truncate(start);
        let seg = self.segments.len();
        self.segments.push(Segment {
            start,
            ops,
            outputs: outputs.iter().map(|&o| o - start).collect(),
            pending: vec![None; kept.len()],
        });
        Ok(kept
            .into_iter()
            .enumerate()
            .map(|(part, value)| self.push(value, Rec::Checkpoint { seg, part }))
            .collect())
    }

    /// `outs` as indices, checked: at least one, each recorded at or after
    /// `start`, none twice, and no leaf among the records from `start`.
    fn segment_outputs(
        &self,
        op: &'static str,
        start: usize,
        outs: &[Var],
    ) -> Result<Vec<usize>, OjasError> {
        let refuse = |detail: String| OjasError::Shape { op, detail };
        if outs.is_empty() {
            return Err(refuse(
                "a segment must return at least one output".to_string(),
            ));
        }
        if self.ops[start..].iter().any(|r| matches!(r, Rec::Leaf)) {
            return Err(refuse(
                "a segment recorded a leaf; record it before the checkpoint".to_string(),
            ));
        }
        let mut seen = Vec::with_capacity(outs.len());
        for &out in outs {
            let index = self.index(out, op)?;
            if index < start {
                return Err(refuse(format!(
                    "output {index} was not recorded by the segment"
                )));
            }
            if seen.contains(&index) {
                return Err(refuse(format!("output {index} is returned twice")));
            }
            seen.push(index);
        }
        Ok(seen)
    }

    /// Run `rec`'s forward on the backend and push the value with the
    /// completed record. Nothing is pushed on an error.
    fn record(&mut self, rec: Rec) -> Result<Var, OjasError> {
        let (value, rec) = self.run_op(rec)?;
        Ok(self.push(value, rec))
    }

    /// The forward of one record, whose inputs are already on the tape:
    /// its value, and the record with what that forward returned beside it.
    /// It is the only place an op's forward runs, so a replayed segment
    /// computes exactly what its first recording did.
    fn run_op(&self, rec: Rec) -> Result<(Tensor, Rec), OjasError> {
        let v = &self.values;
        let be = &self.backend;
        Ok(match rec {
            Rec::Leaf | Rec::Checkpoint { .. } => {
                return Err(OjasError::Backend {
                    id: be.id(),
                    detail: "Tape: a leaf or checkpoint output has no forward to run".to_string(),
                })
            }
            Rec::Embed { table, ids } => {
                let y = be.embedding_forward(&v[table], &ids)?;
                (y, Rec::Embed { table, ids })
            }
            Rec::Linear { x, w } => (be.linear_forward(&v[x], &v[w])?, rec),
            Rec::Rms { x, w, eps } => (be.rms_norm_forward(&v[x], &v[w], eps)?, rec),
            Rec::Rope { x, cos, sin } => {
                let y = be.rope_half_split_forward(&v[x], &cos, &sin)?;
                (y, Rec::Rope { x, cos, sin })
            }
            Rec::Sdpa {
                q,
                k,
                v: vi,
                window,
                ..
            } => {
                let (y, lse) = be.causal_sdpa_forward(&v[q], &v[k], &v[vi], window)?;
                let rec = Rec::Sdpa {
                    q,
                    k,
                    v: vi,
                    out: v.len(),
                    lse: Some(lse),
                    window,
                };
                (y, rec)
            }
            Rec::Gate { x, w, b, attn, .. } => {
                let (y, scales) =
                    be.per_head_sigmoid_gate_forward_saving(&v[x], &v[w], &v[b], &v[attn])?;
                let rec = Rec::Gate {
                    x,
                    w,
                    b,
                    attn,
                    scales,
                };
                (y, rec)
            }
            Rec::Vres { v: vi, v0, lambda } => {
                let y = be.value_residual_blend_forward(&v[vi], &v[v0], &v[lambda])?;
                (y, rec)
            }
            Rec::Gdn {
                q,
                k,
                v: vi,
                g,
                beta,
                ..
            } => {
                let out = be.chunked_gdn_forward(GdnInputs {
                    q: &v[q],
                    k: &v[k],
                    v: &v[vi],
                    g: &v[g],
                    beta: &v[beta],
                    initial_state: None,
                })?;
                let rec = Rec::Gdn {
                    q,
                    k,
                    v: vi,
                    g,
                    beta,
                    checkpoints: Some(out.checkpoints),
                };
                (out.output, rec)
            }
            Rec::Silu { x } => (be.silu_forward(&v[x])?, rec),
            Rec::Conv1d { x, w } => (be.causal_conv1d_silu_forward(&v[x], &v[w])?, rec),
            Rec::GatedRms { x, z, w, eps } => {
                (be.gated_rms_norm_forward(&v[x], &v[z], &v[w], eps)?, rec)
            }
            Rec::RopePartial {
                x,
                ref cos,
                ref sin,
            } => (be.rope_partial_forward(&v[x], cos, sin)?, rec),
            Rec::Mul { a, b } => (be.mul_forward(&v[a], &v[b])?, rec),
            Rec::Add { x, y } => (be.residual_add_forward(&v[x], &v[y])?, rec),
            Rec::Ce {
                logits,
                targets,
                ignore,
            } => {
                let y = be.cross_entropy_mean_forward(&v[logits], &targets, ignore)?;
                let rec = Rec::Ce {
                    logits,
                    targets,
                    ignore,
                };
                (y, rec)
            }
            Rec::LinearCe {
                x,
                w,
                targets,
                ignore,
                chunk,
                ..
            } => {
                let (loss, grads) = self.fused_ce(x, w, &targets, ignore, chunk)?;
                let rec = Rec::LinearCe {
                    x,
                    w,
                    targets,
                    ignore,
                    chunk,
                    grads: Some(grads),
                };
                (loss, rec)
            }
            Rec::Reshape { src, ref shape, .. } => (v[src].reshape(shape)?, rec),
            Rec::Permute { src, ref inverse } => {
                (be.permute(&v[src], &inverse_permutation(inverse))?, rec)
            }
        })
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
    /// A checkpointed segment ([`Tape::checkpoint`]) is recomputed when the
    /// walk reaches it and dropped once its inputs have their gradients,
    /// also on an error.
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
        let len = self.values.len();
        // Nothing recorded after the root feeds it, so its gradient is
        // still the seed. Only a seed of exactly 1 lets a loss hand its
        // unscaled gradient straight through.
        let unit_root = (seed == 1.0).then_some(var.0);
        let walked = self.walk(0, var.0, unit_root);
        // A replay that failed part way leaves its nodes past `len`, and
        // gradients pending in segments the walk did not reach.
        self.truncate(len);
        for s in &mut self.segments {
            s.pending.iter_mut().for_each(|g| *g = None);
        }
        if walked.is_err() {
            self.grads.iter_mut().for_each(|slot| *slot = None);
        }
        walked
    }

    /// Backpropagate every node from `top` down to `bottom`, inclusive.
    /// `unit_root` is the node whose gradient is still an exact seed of 1.
    fn walk(
        &mut self,
        bottom: usize,
        top: usize,
        unit_root: Option<usize>,
    ) -> Result<(), OjasError> {
        for index in (bottom..=top).rev() {
            let grad = self.grads[index].take();
            if let Rec::Checkpoint { seg, part } = self.ops[index] {
                // Every consumer of a segment output was recorded after all
                // of the segment's outputs, so this gradient is final.
                self.segments[seg].pending[part] = grad;
                if part == 0 {
                    self.replay(seg)?;
                }
                continue;
            }
            let Some(grad) = grad else {
                continue;
            };
            let op = self.op_for_backward(index);
            // Leaves keep the gradient a caller reads. Every other node's
            // gradient has been pushed into its inputs, so it is dropped.
            let retain = matches!(op, Rec::Leaf);
            self.backward_one(op, &grad, unit_root == Some(index))?;
            if retain {
                self.grads[index] = Some(grad);
            }
        }
        Ok(())
    }

    /// Record segment `seg` again past the end of the tape, seed its outputs
    /// with the gradients the walk gathered, backpropagate it into its
    /// inputs, and drop it. A segment none of whose outputs has a gradient
    /// is not recomputed.
    fn replay(&mut self, seg: usize) -> Result<(), OjasError> {
        let pending: Vec<Option<Tensor>> = self.segments[seg]
            .pending
            .iter_mut()
            .map(Option::take)
            .collect();
        if pending.iter().all(Option::is_none) {
            return Ok(());
        }
        let base = self.values.len();
        let start = self.segments[seg].start;
        for i in 0..self.segments[seg].ops.len() {
            let rec = self.segments[seg].ops[i].clone().moved(start, base);
            self.record(rec)?;
        }
        for (k, grad) in pending.into_iter().enumerate() {
            let out = base + self.segments[seg].outputs[k];
            self.grads[out] = grad;
        }
        let top = self.values.len() - 1;
        self.walk(base, top, None)?;
        self.truncate(base);
        Ok(())
    }

    /// Drop every node from `len` on.
    fn truncate(&mut self, len: usize) {
        self.values.truncate(len);
        self.grads.truncate(len);
        self.ops.truncate(len);
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
            // `walk` replays segments itself and never passes one here.
            Rec::Checkpoint { .. } => Err(self.missing("a checkpoint output")),
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
            Rec::Sdpa {
                q,
                k,
                v,
                out,
                lse,
                window,
            } => {
                let lse = lse.ok_or_else(|| self.missing("attention's lse"))?;
                let (gq, gk, gv) = self.backend.causal_sdpa_backward(
                    &self.values[q],
                    &self.values[k],
                    &self.values[v],
                    &self.values[out],
                    &lse,
                    grad,
                    window,
                )?;
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
            Rec::Gdn {
                q,
                k,
                v,
                g,
                beta,
                checkpoints,
            } => {
                let checkpoints = checkpoints.ok_or_else(|| self.missing("GDN checkpoints"))?;
                let (qv, kv, vv, gv, bv) = (
                    self.values[q].clone(),
                    self.values[k].clone(),
                    self.values[v].clone(),
                    self.values[g].clone(),
                    self.values[beta].clone(),
                );
                let inputs = GdnInputs {
                    q: &qv,
                    k: &kv,
                    v: &vv,
                    g: &gv,
                    beta: &bv,
                    initial_state: None,
                };
                let gr = self
                    .backend
                    .chunked_gdn_backward(inputs, &checkpoints, grad, None)?;
                self.acc(q, gr.q)?;
                self.acc(k, gr.k)?;
                self.acc(v, gr.v)?;
                self.acc(g, gr.g)?;
                self.acc(beta, gr.beta)
            }
            Rec::Silu { x } => {
                let xv = self.values[x].clone();
                let gx = self.backend.silu_backward(&xv, grad)?;
                self.acc(x, gx)
            }
            Rec::Conv1d { x, w } => {
                let (xv, wv) = (self.values[x].clone(), self.values[w].clone());
                let (gx, gw) = self.backend.causal_conv1d_silu_backward(&xv, &wv, grad)?;
                self.acc(x, gx)?;
                self.acc(w, gw)
            }
            Rec::GatedRms { x, z, w, eps } => {
                let (xv, zv, wv) = (
                    self.values[x].clone(),
                    self.values[z].clone(),
                    self.values[w].clone(),
                );
                let g = self
                    .backend
                    .gated_rms_norm_backward(&xv, &zv, &wv, grad, eps)?;
                self.acc(x, g.input)?;
                self.acc(z, g.gate)?;
                self.acc(w, g.weight)
            }
            Rec::RopePartial { x, cos, sin } => {
                let gx = self.backend.rope_partial_backward(grad, &cos, &sin)?;
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
            Rec::Reshape { src, src_shape, .. } => {
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

    /// A record reached the backward without what its forward returned:
    /// a tape bug, never a caller's.
    fn missing(&self, what: &str) -> OjasError {
        OjasError::Backend {
            id: self.backend.id(),
            detail: format!("Tape::backward: {what} is missing from its record"),
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
