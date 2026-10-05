//! Dynamic bf16 autocast for any [`Backend`].
//!
//! A region is a per-thread stack on one [`Autocast`] value, not a process
//! global. Inside a [`AutocastMode::Bf16`] region, matmul-class ops round
//! untagged f32 operands and their activation outputs. The storage compute
//! tag records that the bits are already rounded, so a second cast clones.
//! Norms, embeddings, the loss, and the optimizer stay f32. A device tensor
//! is never downloaded to round it.
//!
//! [`Backend`]: crate::Backend

use crate::backend::{
    AdamWConfig, Backend, BackendId, CeChunk, LinearCe, MuonNs5Config, PerHeadGateGrad,
    ValueResidualGrad,
};
use crate::budget::Budget;
use crate::dtype::DType;
use crate::error::OjasError;
use crate::tensor::{Tensor, COMPUTE_BF16, COMPUTE_F32};
use std::collections::HashMap;
use std::ops::Deref;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, ThreadId};

/// Bit 31 of the depth word. The live frame count is at most 64 × 64.
const POISON_BIT: u32 = 1 << 31;
const MAX_DEPTH: usize = 64;
const MAX_THREADS: usize = 64;

/// Whether the calling thread's innermost region lowers matmul-class ops.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AutocastMode {
    /// Leave every op in f32. This is the mode of an empty stack.
    #[default]
    Off,
    /// Round operands and activation outputs of matmul-class ops to bf16.
    Bf16,
}

/// `bits << 16`. Every bf16 value is an f32.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Round to nearest even on the dropped 16 bits.
///
/// NaN keeps its sign and the top payload bits and sets the quiet bit
/// `0x0040`. The largest non-NaN magnitude, with the sign bit set
/// (`0xFF7FFFFF`), plus the round increment `0x8000` is `0xFF87FFFF` and
/// does not wrap a `u32`.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = 0x7FFF + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

/// Round `value` to bf16 and widen it back to f32. The low 16 bits are zero.
pub fn round_f32_to_bf16(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(value))
}

struct Frame {
    id: u64,
    mode: AutocastMode,
}

struct RegionState {
    poisoned: bool,
    next_id: u64,
    threads: HashMap<ThreadId, Vec<Frame>>,
}

struct Shared {
    depth: AtomicU32,
    state: Mutex<RegionState>,
}

/// Holds the region mutex. A panic while it is held sets the poison bit, so
/// the depth-0 fast path still fails closed.
struct Held<'a> {
    guard: MutexGuard<'a, RegionState>,
    depth: &'a AtomicU32,
    disarm: bool,
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        if !self.disarm {
            self.depth.fetch_or(POISON_BIT, Ordering::Release);
            self.guard.poisoned = true;
        }
    }
}

fn lock(shared: &Shared) -> Result<Held<'_>, OjasError> {
    match shared.state.lock() {
        Ok(guard) => Ok(Held {
            guard,
            depth: &shared.depth,
            disarm: false,
        }),
        Err(poisoned) => {
            let mut guard = poisoned.into_inner();
            guard.poisoned = true;
            shared.depth.fetch_or(POISON_BIT, Ordering::Release);
            Err(OjasError::Poisoned)
        }
    }
}

fn with_state<T>(
    shared: &Shared,
    body: impl FnOnce(&mut RegionState) -> Result<T, OjasError>,
) -> Result<T, OjasError> {
    let mut held = lock(shared)?;
    // `MutexGuard` poisons if it was acquired before a panic and dropped
    // during that panic. Catch the panic, record ours, and unlock while this
    // thread is not unwinding, then resume. A drop that already is unwinding
    // acquired the lock in that state, which does not poison, so it runs the
    // body directly.
    if thread::panicking() {
        let result = body(&mut held.guard);
        held.disarm = true;
        return result;
    }
    match catch_unwind(AssertUnwindSafe(|| body(&mut held.guard))) {
        Ok(result) => {
            held.disarm = true;
            result
        }
        Err(payload) => {
            held.depth.fetch_or(POISON_BIT, Ordering::Release);
            held.guard.poisoned = true;
            held.disarm = true;
            drop(held);
            resume_unwind(payload);
        }
    }
}

fn alloc_frame_id(next: &mut u64) -> Result<u64, OjasError> {
    if *next == u64::MAX {
        return Err(OjasError::OutOfRange {
            op: "autocast_region",
            detail: "frame id would wrap".into(),
        });
    }
    let id = *next;
    *next += 1;
    Ok(id)
}

fn pop(shared: &Shared, thread: ThreadId, id: u64) {
    if shared.depth.load(Ordering::Acquire) & POISON_BIT != 0 {
        return;
    }
    let _ = with_state(shared, |state| {
        if state.poisoned {
            shared.depth.fetch_or(POISON_BIT, Ordering::Release);
            return Ok(());
        }
        let Some(stack) = state.threads.get_mut(&thread) else {
            state.poisoned = true;
            shared.depth.fetch_or(POISON_BIT, Ordering::Release);
            return Ok(());
        };
        if stack.last().is_some_and(|frame| frame.id == id) {
            stack.pop();
            if stack.is_empty() {
                state.threads.remove(&thread);
            }
            shared.depth.fetch_sub(1, Ordering::Release);
        } else {
            state.poisoned = true;
            shared.depth.fetch_or(POISON_BIT, Ordering::Release);
        }
        Ok(())
    });
}

/// Ends one autocast region when dropped. Not `Send`: the frame belongs to
/// the thread that entered it. Dropping out of order poisons the wrapper.
#[must_use = "the guard ends the autocast region when dropped; bind it"]
pub struct AutocastGuard {
    pop: Option<Box<dyn FnOnce()>>,
}

impl AutocastGuard {
    pub(crate) fn noop() -> Self {
        Self { pop: None }
    }
}

impl std::fmt::Debug for AutocastGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutocastGuard")
            .field("active", &self.pop.is_some())
            .finish()
    }
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        if let Some(pop) = self.pop.take() {
            pop();
        }
    }
}

/// Policy wrapper. Delegates every op to `inner`. Not `Clone`: two wrappers
/// do not share a region stack.
pub struct Autocast<B: Backend> {
    inner: B,
    shared: Arc<Shared>,
}

impl<B: Backend> Autocast<B> {
    pub fn new(inner: B) -> Self {
        Self {
            inner,
            shared: Arc::new(Shared {
                depth: AtomicU32::new(0),
                state: Mutex::new(RegionState {
                    poisoned: false,
                    next_id: 0,
                    threads: HashMap::new(),
                }),
            }),
        }
    }

    /// Innermost region on this thread is [`AutocastMode::Bf16`].
    ///
    /// Poison fails closed even when the depth word's low bits are zero.
    pub fn autocast_enabled(&self) -> Result<bool, OjasError> {
        self.region()
    }

    fn region(&self) -> Result<bool, OjasError> {
        let depth = self.shared.depth.load(Ordering::Acquire);
        if depth & POISON_BIT != 0 {
            return Err(OjasError::Poisoned);
        }
        if depth & !POISON_BIT == 0 {
            return Ok(false);
        }
        let thread = thread::current().id();
        with_state(&self.shared, |state| {
            if state.poisoned {
                return Err(OjasError::Poisoned);
            }
            Ok(state
                .threads
                .get(&thread)
                .and_then(|stack| stack.last())
                .is_some_and(|frame| frame.mode == AutocastMode::Bf16))
        })
    }

    fn prep<'a>(&self, tensor: &'a Tensor) -> Result<Operand<'a>, OjasError> {
        if !self.region()? || tensor.compute_tag() == COMPUTE_BF16 {
            return Ok(Operand::Same(tensor));
        }
        Ok(Operand::Rounded(self.round_new(tensor)?))
    }

    fn round_new(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        let rounded = self.inner.cast_bf16(tensor)?;
        rounded.set_compute_tag(COMPUTE_BF16);
        Ok(rounded)
    }

    fn emit(&self, tensor: Tensor) -> Result<Tensor, OjasError> {
        if !self.region()? || tensor.compute_tag() == COMPUTE_BF16 {
            return Ok(tensor);
        }
        self.round_new(&tensor)
    }

    fn promote(&self, on: bool, tagged: bool, tensor: Tensor) -> Result<Tensor, OjasError> {
        if on && tagged {
            self.emit(tensor)
        } else {
            Ok(tensor)
        }
    }

    fn pass(&self) -> Result<(), OjasError> {
        self.region().map(|_| ())
    }

    fn keep_f32(&self, tensor: Tensor) -> Tensor {
        tensor.set_compute_tag(COMPUTE_F32);
        tensor
    }

    fn copy_tag(&self, src: &Tensor, dst: Tensor) -> Tensor {
        dst.set_compute_tag(src.compute_tag());
        dst
    }

    fn wrote(&self, tensor: &Tensor) {
        tensor.set_compute_tag(COMPUTE_F32);
    }
}

enum Operand<'a> {
    Same(&'a Tensor),
    Rounded(Tensor),
}

impl Deref for Operand<'_> {
    type Target = Tensor;
    fn deref(&self) -> &Tensor {
        match self {
            Operand::Same(tensor) => tensor,
            Operand::Rounded(tensor) => tensor,
        }
    }
}

impl<B: Backend> Backend for Autocast<B> {
    fn id(&self) -> BackendId {
        self.inner.id()
    }

    fn budget(&self) -> &Budget {
        self.inner.budget()
    }

    fn numerics(&self) -> crate::backend::Numerics {
        self.inner.numerics()
    }

    fn cast_bf16(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        self.pass()?;
        if tensor.dtype() == DType::F32 && tensor.compute_tag() == COMPUTE_BF16 {
            return Ok(tensor.clone());
        }
        self.round_new(tensor)
    }

    fn autocast_region(&self, mode: AutocastMode) -> Result<AutocastGuard, OjasError> {
        if self.shared.depth.load(Ordering::Acquire) & POISON_BIT != 0 {
            return Err(OjasError::Poisoned);
        }
        let thread = thread::current().id();
        let id = with_state(&self.shared, |state| {
            if state.poisoned {
                return Err(OjasError::Poisoned);
            }
            let new_thread = !state.threads.contains_key(&thread);
            if new_thread && state.threads.len() >= MAX_THREADS {
                return Err(OjasError::OutOfRange {
                    op: "autocast_region",
                    detail: format!("at most {MAX_THREADS} threads may hold an autocast region"),
                });
            }
            let depth_now = state.threads.get(&thread).map(Vec::len).unwrap_or(0);
            if depth_now >= MAX_DEPTH {
                return Err(OjasError::OutOfRange {
                    op: "autocast_region",
                    detail: format!("autocast region depth exceeds {MAX_DEPTH}"),
                });
            }
            // Insert only after the id is allocated, so a refusal cannot
            // leave an empty thread entry or advance `next_id`.
            let id = alloc_frame_id(&mut state.next_id)?;
            state
                .threads
                .entry(thread)
                .or_default()
                .push(Frame { id, mode });
            self.shared.depth.fetch_add(1, Ordering::Release);
            Ok(id)
        })?;
        let shared = Arc::clone(&self.shared);
        Ok(AutocastGuard {
            pop: Some(Box::new(move || pop(&shared, thread, id))),
        })
    }

    fn upload(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        self.pass()?;
        let out = self.inner.upload(tensor)?;
        Ok(self.copy_tag(tensor, out))
    }

    fn download(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        self.pass()?;
        let out = self.inner.download(tensor)?;
        Ok(self.copy_tag(tensor, out))
    }

    fn permute(&self, input: &Tensor, dims: &[usize]) -> Result<Tensor, OjasError> {
        self.pass()?;
        let out = self.inner.permute(input, dims)?;
        Ok(self.copy_tag(input, out))
    }

    fn embedding_forward(&self, table: &Tensor, token_ids: &Tensor) -> Result<Tensor, OjasError> {
        self.pass()?;
        self.inner.embedding_forward(table, token_ids)
    }

    fn embedding_backward(
        &self,
        table: &Tensor,
        token_ids: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.pass()?;
        self.inner.embedding_backward(table, token_ids, grad_output)
    }

    fn linear_forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor, OjasError> {
        let input = self.prep(input)?;
        let weight = self.prep(weight)?;
        let out = self.inner.linear_forward(&input, &weight)?;
        self.emit(out)
    }

    fn linear_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let input = self.prep(input)?;
        let weight = self.prep(weight)?;
        let grad_output = self.prep(grad_output)?;
        let (grad_input, grad_weight) =
            self.inner.linear_backward(&input, &weight, &grad_output)?;
        Ok((self.emit(grad_input)?, self.keep_f32(grad_weight)))
    }

    fn rms_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        self.pass()?;
        self.inner.rms_norm_forward(input, weight, eps)
    }

    fn rms_norm_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.pass()?;
        self.inner
            .rms_norm_backward(input, weight, grad_output, eps)
    }

    fn rope_half_split_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let on = self.region()?;
        let tagged = x.compute_tag() == COMPUTE_BF16;
        let out = self.inner.rope_half_split_forward(x, cos, sin)?;
        self.promote(on, tagged, out)
    }

    fn rope_half_split_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let on = self.region()?;
        let tagged = grad_output.compute_tag() == COMPUTE_BF16;
        let out = self.inner.rope_half_split_backward(grad_output, cos, sin)?;
        self.promote(on, tagged, out)
    }

    fn rms_qk_norm_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.pass()?;
        self.inner
            .rms_qk_norm_forward(q, k, q_weight, k_weight, eps)
    }

    fn rms_qk_norm_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        grad_q: &Tensor,
        grad_k: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), OjasError> {
        self.pass()?;
        self.inner
            .rms_qk_norm_backward(q, k, q_weight, k_weight, grad_q, grad_k, eps)
    }

    fn causal_sdpa_forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, OjasError> {
        let q = self.prep(q)?;
        let k = self.prep(k)?;
        let v = self.prep(v)?;
        let out = self.inner.causal_sdpa_forward(&q, &k, &v)?;
        self.emit(out)
    }

    fn causal_sdpa_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        let q = self.prep(q)?;
        let k = self.prep(k)?;
        let v = self.prep(v)?;
        let grad_output = self.prep(grad_output)?;
        let (gq, gk, gv) = self.inner.causal_sdpa_backward(&q, &k, &v, &grad_output)?;
        Ok((self.emit(gq)?, self.emit(gk)?, self.emit(gv)?))
    }

    fn per_head_sigmoid_gate_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let input = self.prep(input)?;
        let weight = self.prep(weight)?;
        let attn_out = self.prep(attn_out)?;
        let out = self
            .inner
            .per_head_sigmoid_gate_forward(&input, &weight, bias, &attn_out)?;
        self.emit(out)
    }

    fn per_head_sigmoid_gate_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
        grad_output: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        let input = self.prep(input)?;
        let attn_out = self.prep(attn_out)?;
        let grad_output = self.prep(grad_output)?;
        let grad = self.inner.per_head_sigmoid_gate_backward(
            &input,
            weight,
            bias,
            &attn_out,
            &grad_output,
        )?;
        Ok(PerHeadGateGrad {
            input: self.emit(grad.input)?,
            weight: self.keep_f32(grad.weight),
            bias: self.keep_f32(grad.bias),
            attn_out: self.emit(grad.attn_out)?,
        })
    }

    fn per_head_sigmoid_gate_forward_saving(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>), OjasError> {
        let input = self.prep(input)?;
        let weight = self.prep(weight)?;
        let attn_out = self.prep(attn_out)?;
        let (out, scales) = self
            .inner
            .per_head_sigmoid_gate_forward_saving(&input, &weight, bias, &attn_out)?;
        Ok((self.emit(out)?, scales))
    }

    fn per_head_sigmoid_gate_backward_saved(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
        grad_output: &Tensor,
        scales: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        let input = self.prep(input)?;
        let attn_out = self.prep(attn_out)?;
        let grad_output = self.prep(grad_output)?;
        let grad = self.inner.per_head_sigmoid_gate_backward_saved(
            &input,
            weight,
            bias,
            &attn_out,
            &grad_output,
            scales,
        )?;
        Ok(PerHeadGateGrad {
            input: self.emit(grad.input)?,
            weight: self.keep_f32(grad.weight),
            bias: self.keep_f32(grad.bias),
            attn_out: self.emit(grad.attn_out)?,
        })
    }

    fn value_residual_blend_forward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let on = self.region()?;
        let tagged = value.compute_tag() == COMPUTE_BF16
            && value0.compute_tag() == COMPUTE_BF16
            && lambda.compute_tag() == COMPUTE_BF16;
        let out = self
            .inner
            .value_residual_blend_forward(value, value0, lambda)?;
        self.promote(on, tagged, out)
    }

    fn value_residual_blend_backward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
        grad_output: &Tensor,
    ) -> Result<ValueResidualGrad, OjasError> {
        let on = self.region()?;
        let tagged = value.compute_tag() == COMPUTE_BF16
            && value0.compute_tag() == COMPUTE_BF16
            && lambda.compute_tag() == COMPUTE_BF16
            && grad_output.compute_tag() == COMPUTE_BF16;
        let grad = self
            .inner
            .value_residual_blend_backward(value, value0, lambda, grad_output)?;
        Ok(ValueResidualGrad {
            value: self.promote(on, tagged, grad.value)?,
            value0: self.promote(on, tagged, grad.value0)?,
            lambda: self.promote(on, tagged, grad.lambda)?,
        })
    }

    fn silu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        let on = self.region()?;
        let tagged = input.compute_tag() == COMPUTE_BF16;
        let out = self.inner.silu_forward(input)?;
        self.promote(on, tagged, out)
    }

    fn silu_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        let on = self.region()?;
        let tagged =
            input.compute_tag() == COMPUTE_BF16 && grad_output.compute_tag() == COMPUTE_BF16;
        let out = self.inner.silu_backward(input, grad_output)?;
        self.promote(on, tagged, out)
    }

    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        let on = self.region()?;
        let tagged = a.compute_tag() == COMPUTE_BF16 && b.compute_tag() == COMPUTE_BF16;
        let out = self.inner.mul_forward(a, b)?;
        self.promote(on, tagged, out)
    }

    fn mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let on = self.region()?;
        let tagged = a.compute_tag() == COMPUTE_BF16
            && b.compute_tag() == COMPUTE_BF16
            && grad_output.compute_tag() == COMPUTE_BF16;
        let (ga, gb) = self.inner.mul_backward(a, b, grad_output)?;
        Ok((self.promote(on, tagged, ga)?, self.promote(on, tagged, gb)?))
    }

    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        let on = self.region()?;
        let tagged = x.compute_tag() == COMPUTE_BF16 && y.compute_tag() == COMPUTE_BF16;
        let out = self.inner.residual_add_forward(x, y)?;
        self.promote(on, tagged, out)
    }

    fn residual_add_backward(
        &self,
        x: &Tensor,
        y: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let on = self.region()?;
        let tagged = x.compute_tag() == COMPUTE_BF16
            && y.compute_tag() == COMPUTE_BF16
            && grad_output.compute_tag() == COMPUTE_BF16;
        let (gx, gy) = self.inner.residual_add_backward(x, y, grad_output)?;
        Ok((self.promote(on, tagged, gx)?, self.promote(on, tagged, gy)?))
    }

    fn cross_entropy_mean_forward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        self.pass()?;
        self.inner
            .cross_entropy_mean_forward(logits, targets, ignore_index)
    }

    fn cross_entropy_mean_backward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        self.pass()?;
        self.inner
            .cross_entropy_mean_backward(logits, targets, ignore_index)
    }

    fn clip_grad_norm(&self, grads: &mut [Tensor], max_norm: f32) -> Result<f32, OjasError> {
        self.pass()?;
        let norm = self.inner.clip_grad_norm(grads, max_norm)?;
        for grad in grads.iter() {
            self.wrote(grad);
        }
        Ok(norm)
    }

    fn adamw_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        moment1: &mut Tensor,
        moment2: &mut Tensor,
        step: u64,
        config: AdamWConfig,
    ) -> Result<(), OjasError> {
        self.pass()?;
        self.inner
            .adamw_step(param, grad, moment1, moment2, step, config)?;
        self.wrote(param);
        self.wrote(moment1);
        self.wrote(moment2);
        Ok(())
    }

    fn muon_ns5_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        momentum: &mut Tensor,
        config: MuonNs5Config,
    ) -> Result<(), OjasError> {
        self.pass()?;
        self.inner.muon_ns5_step(param, grad, momentum, config)?;
        self.wrote(param);
        self.wrote(momentum);
        Ok(())
    }

    fn optimizer_scratch_bytes(
        &self,
        kind: crate::backend::OptimizerKind,
        rows: usize,
        cols: usize,
    ) -> Result<Option<u64>, OjasError> {
        self.pass()?;
        self.inner.optimizer_scratch_bytes(kind, rows, cols)
    }

    fn sync(&self) -> Result<(), OjasError> {
        self.pass()?;
        self.inner.sync()
    }

    fn accumulate_grad(&self, acc: &mut Tensor, grad: &Tensor) -> Result<(), OjasError> {
        self.pass()?;
        self.inner.accumulate_grad(acc, grad)?;
        self.wrote(acc);
        Ok(())
    }

    fn linear_cross_entropy_mean(
        &self,
        input: &Tensor,
        weight: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
        chunk: CeChunk,
        want_grad: bool,
    ) -> Result<LinearCe, OjasError> {
        let input = self.prep(input)?;
        let weight = self.prep(weight)?;
        let out = self.inner.linear_cross_entropy_mean(
            &input,
            &weight,
            targets,
            ignore_index,
            chunk,
            want_grad,
        )?;
        Ok(LinearCe {
            loss: out.loss,
            grad_input: match out.grad_input {
                Some(grad) => Some(self.emit(grad)?),
                None => None,
            },
            grad_weight: out.grad_weight.map(|grad| self.keep_f32(grad)),
        })
    }

    fn cached_attention_forward(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        kv_len: usize,
    ) -> Result<Tensor, OjasError> {
        let q = self.prep(q)?;
        let k_cache = self.prep(k_cache)?;
        let v_cache = self.prep(v_cache)?;
        let out = self
            .inner
            .cached_attention_forward(&q, &k_cache, &v_cache, kv_len)?;
        self.emit(out)
    }

    fn kv_cache_write(&self, cache: &mut Tensor, src: &Tensor, at: usize) -> Result<(), OjasError> {
        self.pass()?;
        self.inner.kv_cache_write(cache, src, at)?;
        self.wrote(cache);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{permute_output_shape, Numerics};
    use crate::tensor::DeviceBuffer;
    use std::any::Any;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::AtomicU64;
    use std::sync::Barrier;

    const DIRTY: u32 = 0x3f80_0001;
    const ONE: u32 = 0x3f80_0000;

    struct Rec {
        budget: Budget,
        numerics: Numerics,
        casts: AtomicU64,
        linear_calls: AtomicU64,
        saving_calls: AtomicU64,
        saved_calls: AtomicU64,
        grad_bits: AtomicU32,
        seen_input: AtomicU32,
        seen_weight: AtomicU32,
        seen_bias: AtomicU32,
        seen_attn: AtomicU32,
        seen_k: AtomicU32,
        scale_bits: AtomicU32,
    }

    impl Rec {
        fn new(budget: Budget) -> Self {
            Self {
                budget,
                numerics: Numerics::Exact,
                casts: AtomicU64::new(0),
                linear_calls: AtomicU64::new(0),
                saving_calls: AtomicU64::new(0),
                saved_calls: AtomicU64::new(0),
                grad_bits: AtomicU32::new(0),
                seen_input: AtomicU32::new(0),
                seen_weight: AtomicU32::new(0),
                seen_bias: AtomicU32::new(0),
                seen_attn: AtomicU32::new(0),
                seen_k: AtomicU32::new(0),
                scale_bits: AtomicU32::new(0),
            }
        }

        fn with_numerics(mut self, numerics: Numerics) -> Self {
            self.numerics = numerics;
            self
        }
    }

    fn first_bits(tensor: &Tensor) -> u32 {
        tensor
            .f32_slice()
            .ok()
            .and_then(|values| values.first().copied())
            .map(f32::to_bits)
            .unwrap_or(0)
    }

    fn note(slot: &AtomicU32, tensor: &Tensor) {
        slot.store(first_bits(tensor), Ordering::Relaxed);
    }

    fn finite(op: &'static str, tensor: &Tensor) -> Result<(), OjasError> {
        let values = tensor.f32_slice()?;
        if values.iter().any(|value| !value.is_finite()) {
            Err(OjasError::NonFinite { op })
        } else {
            Ok(())
        }
    }

    fn dirty(budget: &Budget, shape: &[usize]) -> Result<Tensor, OjasError> {
        let n = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
        let Some(n) = n else {
            return Err(OjasError::OutOfRange {
                op: "rec",
                detail: "shape product overflow".into(),
            });
        };
        Tensor::from_f32(&vec![f32::from_bits(DIRTY); n], shape, budget)
    }

    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            let z = (-x).exp();
            1.0 / (1.0 + z)
        } else {
            let z = x.exp();
            z / (1.0 + z)
        }
    }

    macro_rules! unsup {
        ($name:ident ($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty) => {
            fn $name(&self, $($arg: $ty),*) -> Result<$ret, OjasError> {
                let _ = ($($arg),*);
                Err(OjasError::Unsupported {
                    op: stringify!($name),
                    detail: "recording backend".into(),
                })
            }
        };
    }

    impl Backend for Rec {
        fn id(&self) -> BackendId {
            BackendId::Cpu
        }

        fn budget(&self) -> &Budget {
            &self.budget
        }

        fn numerics(&self) -> Numerics {
            self.numerics
        }

        fn cast_bf16(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
            const OP: &str = "cast_bf16";
            if tensor.dtype() != DType::F32 {
                return Err(OjasError::Dtype {
                    op: OP,
                    expected: DType::F32,
                    got: tensor.dtype(),
                });
            }
            if tensor.device().is_some() {
                return Err(OjasError::Unsupported {
                    op: OP,
                    detail: "recording backend does not download a device tensor".into(),
                });
            }
            self.casts.fetch_add(1, Ordering::Relaxed);
            let src = tensor.f32_slice()?;
            let mut out = Tensor::zeros(tensor.shape(), DType::F32, self.budget())?;
            {
                let dst = out.f32_slice_mut()?;
                for (dst, src) in dst.iter_mut().zip(src.iter()) {
                    *dst = round_f32_to_bf16(*src);
                }
            }
            Ok(out)
        }

        fn permute(&self, input: &Tensor, dims: &[usize]) -> Result<Tensor, OjasError> {
            const OP: &str = "permute";
            let out_shape = permute_output_shape(OP, input.shape(), dims)?;
            let src = input.f32_slice()?;
            let mut out = Tensor::zeros(&out_shape, DType::F32, self.budget())?;
            let rank = input.shape().len();
            if src.is_empty() {
                return Ok(out);
            }
            let mut in_strides = vec![1usize; rank];
            for axis in (0..rank.saturating_sub(1)).rev() {
                in_strides[axis] = in_strides[axis + 1].saturating_mul(input.shape()[axis + 1]);
            }
            let dst = out.f32_slice_mut()?;
            let mut coord = vec![0usize; rank];
            for dest in dst.iter_mut() {
                let mut src_index = 0usize;
                for axis in 0..rank {
                    src_index += coord[axis] * in_strides[dims[axis]];
                }
                *dest = src[src_index];
                for axis in (0..rank).rev() {
                    coord[axis] += 1;
                    if coord[axis] < out_shape[axis] {
                        break;
                    }
                    coord[axis] = 0;
                }
            }
            Ok(out)
        }

        fn embedding_forward(
            &self,
            table: &Tensor,
            token_ids: &Tensor,
        ) -> Result<Tensor, OjasError> {
            const OP: &str = "embedding_forward";
            if table.dtype() != DType::F32 {
                return Err(OjasError::Dtype {
                    op: OP,
                    expected: DType::F32,
                    got: table.dtype(),
                });
            }
            if token_ids.dtype() != DType::U32 {
                return Err(OjasError::Dtype {
                    op: OP,
                    expected: DType::U32,
                    got: token_ids.dtype(),
                });
            }
            let rows = table.f32_slice()?;
            let ids = token_ids.u32_slice()?;
            let width = *table.shape().last().unwrap_or(&0);
            if width == 0 || rows.len() % width != 0 {
                return Err(OjasError::Shape {
                    op: OP,
                    detail: "embedding table has no rows".into(),
                });
            }
            let vocab = rows.len() / width;
            let mut data = vec![0.0f32; ids.len() * width];
            for (row, &id) in ids.iter().enumerate() {
                let id = id as usize;
                if id >= vocab {
                    return Err(OjasError::OutOfRange {
                        op: OP,
                        detail: format!("token id {id} outside vocab {vocab}"),
                    });
                }
                data[row * width..(row + 1) * width]
                    .copy_from_slice(&rows[id * width..(id + 1) * width]);
            }
            let mut shape = token_ids.shape().to_vec();
            shape.push(width);
            Tensor::from_f32(&data, &shape, self.budget())
        }

        unsup!(embedding_backward(table: &Tensor, token_ids: &Tensor, grad_output: &Tensor) -> Tensor);

        fn linear_forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor, OjasError> {
            const OP: &str = "linear_forward";
            self.linear_calls.fetch_add(1, Ordering::Relaxed);
            note(&self.seen_input, input);
            note(&self.seen_weight, weight);
            finite(OP, input)?;
            finite(OP, weight)?;
            dirty(self.budget(), input.shape())
        }

        fn linear_backward(
            &self,
            input: &Tensor,
            weight: &Tensor,
            grad_output: &Tensor,
        ) -> Result<(Tensor, Tensor), OjasError> {
            const OP: &str = "linear_backward";
            note(&self.seen_input, input);
            finite(OP, input)?;
            finite(OP, weight)?;
            finite(OP, grad_output)?;
            Ok((
                dirty(self.budget(), input.shape())?,
                dirty(self.budget(), weight.shape())?,
            ))
        }

        fn rms_norm_forward(
            &self,
            input: &Tensor,
            weight: &Tensor,
            eps: f32,
        ) -> Result<Tensor, OjasError> {
            const OP: &str = "rms_norm_forward";
            let x = input.f32_slice()?;
            let w = weight.f32_slice()?;
            let cols = w.len();
            if cols == 0 || x.len() % cols != 0 {
                return Err(OjasError::Shape {
                    op: OP,
                    detail: "rms weight does not match the last axis".into(),
                });
            }
            let mut y = vec![0.0f32; x.len()];
            let rows = x.len() / cols;
            for row in 0..rows {
                let sl = &x[row * cols..(row + 1) * cols];
                let mut sum = 0.0f32;
                for value in sl {
                    sum += value * value;
                }
                let inv = 1.0 / (sum / cols as f32 + eps).sqrt();
                for col in 0..cols {
                    y[row * cols + col] = sl[col] * inv * w[col];
                }
            }
            Tensor::from_f32(&y, input.shape(), self.budget())
        }

        unsup!(rms_norm_backward(input: &Tensor, weight: &Tensor, grad_output: &Tensor, eps: f32) -> (Tensor, Tensor));
        unsup!(rms_qk_norm_forward(q: &Tensor, k: &Tensor, q_weight: &Tensor, k_weight: &Tensor, eps: f32) -> (Tensor, Tensor));
        unsup!(rms_qk_norm_backward(
            q: &Tensor,
            k: &Tensor,
            q_weight: &Tensor,
            k_weight: &Tensor,
            grad_q: &Tensor,
            grad_k: &Tensor,
            eps: f32
        ) -> (Tensor, Tensor, Tensor, Tensor));

        fn rope_half_split_forward(
            &self,
            x: &Tensor,
            cos: &Tensor,
            sin: &Tensor,
        ) -> Result<Tensor, OjasError> {
            let y = rope_apply(x, cos, sin, false)?;
            Tensor::from_f32(&y, x.shape(), self.budget())
        }

        fn rope_half_split_backward(
            &self,
            grad_output: &Tensor,
            cos: &Tensor,
            sin: &Tensor,
        ) -> Result<Tensor, OjasError> {
            let y = rope_apply(grad_output, cos, sin, true)?;
            Tensor::from_f32(&y, grad_output.shape(), self.budget())
        }

        fn causal_sdpa_forward(
            &self,
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
        ) -> Result<Tensor, OjasError> {
            const OP: &str = "causal_sdpa_forward";
            note(&self.seen_input, q);
            note(&self.seen_k, k);
            finite(OP, q)?;
            finite(OP, k)?;
            finite(OP, v)?;
            dirty(self.budget(), q.shape())
        }

        fn causal_sdpa_backward(
            &self,
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            grad_output: &Tensor,
        ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
            const OP: &str = "causal_sdpa_backward";
            finite(OP, q)?;
            finite(OP, k)?;
            finite(OP, v)?;
            finite(OP, grad_output)?;
            Ok((
                dirty(self.budget(), q.shape())?,
                dirty(self.budget(), k.shape())?,
                dirty(self.budget(), v.shape())?,
            ))
        }

        fn per_head_sigmoid_gate_forward(
            &self,
            input: &Tensor,
            weight: &Tensor,
            bias: &Tensor,
            attn_out: &Tensor,
        ) -> Result<Tensor, OjasError> {
            const OP: &str = "per_head_sigmoid_gate_forward";
            note(&self.seen_input, input);
            note(&self.seen_weight, weight);
            note(&self.seen_bias, bias);
            note(&self.seen_attn, attn_out);
            finite(OP, input)?;
            finite(OP, weight)?;
            finite(OP, bias)?;
            finite(OP, attn_out)?;
            dirty(self.budget(), attn_out.shape())
        }

        fn per_head_sigmoid_gate_backward(
            &self,
            input: &Tensor,
            weight: &Tensor,
            bias: &Tensor,
            attn_out: &Tensor,
            grad_output: &Tensor,
        ) -> Result<PerHeadGateGrad, OjasError> {
            const OP: &str = "per_head_sigmoid_gate_backward";
            note(&self.seen_input, input);
            note(&self.seen_weight, weight);
            note(&self.seen_bias, bias);
            note(&self.seen_attn, attn_out);
            note(&self.grad_bits, grad_output);
            finite(OP, input)?;
            finite(OP, weight)?;
            finite(OP, bias)?;
            finite(OP, attn_out)?;
            finite(OP, grad_output)?;
            Ok(PerHeadGateGrad {
                input: dirty(self.budget(), input.shape())?,
                weight: dirty(self.budget(), weight.shape())?,
                bias: dirty(self.budget(), bias.shape())?,
                attn_out: dirty(self.budget(), attn_out.shape())?,
            })
        }

        fn per_head_sigmoid_gate_forward_saving(
            &self,
            input: &Tensor,
            weight: &Tensor,
            bias: &Tensor,
            attn_out: &Tensor,
        ) -> Result<(Tensor, Option<Tensor>), OjasError> {
            self.saving_calls.fetch_add(1, Ordering::Relaxed);
            let out = self.per_head_sigmoid_gate_forward(input, weight, bias, attn_out)?;
            let scales = Tensor::from_f32(&[f32::from_bits(DIRTY)], &[1], self.budget())?;
            Ok((out, Some(scales)))
        }

        fn per_head_sigmoid_gate_backward_saved(
            &self,
            input: &Tensor,
            weight: &Tensor,
            bias: &Tensor,
            attn_out: &Tensor,
            grad_output: &Tensor,
            scales: &Tensor,
        ) -> Result<PerHeadGateGrad, OjasError> {
            self.saved_calls.fetch_add(1, Ordering::Relaxed);
            note(&self.scale_bits, scales);
            self.per_head_sigmoid_gate_backward(input, weight, bias, attn_out, grad_output)
        }

        fn value_residual_blend_forward(
            &self,
            value: &Tensor,
            value0: &Tensor,
            lambda: &Tensor,
        ) -> Result<Tensor, OjasError> {
            let v = value.f32_slice()?;
            let v0 = value0.f32_slice()?;
            let lam = lambda.f32_slice()?;
            if v.len() != v0.len() || lam.len() != 1 {
                return Err(OjasError::Shape {
                    op: "value_residual_blend_forward",
                    detail: "value, value0, and scalar lambda".into(),
                });
            }
            let s = sigmoid(lam[0]);
            let y: Vec<f32> = v
                .iter()
                .zip(v0.iter())
                .map(|(value, value0)| (1.0 - s) * value + s * value0)
                .collect();
            Tensor::from_f32(&y, value.shape(), self.budget())
        }

        fn value_residual_blend_backward(
            &self,
            value: &Tensor,
            value0: &Tensor,
            lambda: &Tensor,
            grad_output: &Tensor,
        ) -> Result<ValueResidualGrad, OjasError> {
            let v = value.f32_slice()?;
            let v0 = value0.f32_slice()?;
            let lam = lambda.f32_slice()?;
            let g = grad_output.f32_slice()?;
            if v.len() != v0.len() || v.len() != g.len() || lam.len() != 1 {
                return Err(OjasError::Shape {
                    op: "value_residual_blend_backward",
                    detail: "value, value0, grad, and scalar lambda".into(),
                });
            }
            let s = sigmoid(lam[0]);
            let ds = s * (1.0 - s);
            let gv: Vec<f32> = g.iter().map(|grad| (1.0 - s) * grad).collect();
            let gv0: Vec<f32> = g.iter().map(|grad| s * grad).collect();
            let mut gl = 0.0f32;
            for i in 0..v.len() {
                gl += g[i] * (v0[i] - v[i]) * ds;
            }
            Ok(ValueResidualGrad {
                value: Tensor::from_f32(&gv, value.shape(), self.budget())?,
                value0: Tensor::from_f32(&gv0, value0.shape(), self.budget())?,
                lambda: Tensor::from_f32(&[gl], lambda.shape(), self.budget())?,
            })
        }

        fn silu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
            let x = input.f32_slice()?;
            let y: Vec<f32> = x.iter().map(|value| value * sigmoid(*value)).collect();
            Tensor::from_f32(&y, input.shape(), self.budget())
        }

        fn silu_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
            let x = input.f32_slice()?;
            let g = grad_output.f32_slice()?;
            if x.len() != g.len() {
                return Err(OjasError::Shape {
                    op: "silu_backward",
                    detail: "input and grad differ".into(),
                });
            }
            let y: Vec<f32> = x
                .iter()
                .zip(g.iter())
                .map(|(value, grad)| {
                    let s = sigmoid(*value);
                    grad * s * (1.0 + value * (1.0 - s))
                })
                .collect();
            Tensor::from_f32(&y, input.shape(), self.budget())
        }

        fn mul_forward(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, OjasError> {
            let a = left.f32_slice()?;
            let b = right.f32_slice()?;
            if left.shape() != right.shape() {
                return Err(OjasError::Shape {
                    op: "mul_forward",
                    detail: "operands differ".into(),
                });
            }
            let y: Vec<f32> = a.iter().zip(b.iter()).map(|(a, b)| a * b).collect();
            Tensor::from_f32(&y, left.shape(), self.budget())
        }

        fn mul_backward(
            &self,
            a: &Tensor,
            b: &Tensor,
            grad_output: &Tensor,
        ) -> Result<(Tensor, Tensor), OjasError> {
            let av = a.f32_slice()?;
            let bv = b.f32_slice()?;
            let g = grad_output.f32_slice()?;
            if av.len() != bv.len() || av.len() != g.len() {
                return Err(OjasError::Shape {
                    op: "mul_backward",
                    detail: "operands differ".into(),
                });
            }
            let ga: Vec<f32> = g.iter().zip(bv.iter()).map(|(grad, b)| grad * b).collect();
            let gb: Vec<f32> = g.iter().zip(av.iter()).map(|(grad, a)| grad * a).collect();
            Ok((
                Tensor::from_f32(&ga, a.shape(), self.budget())?,
                Tensor::from_f32(&gb, b.shape(), self.budget())?,
            ))
        }

        fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
            let a = x.f32_slice()?;
            let b = y.f32_slice()?;
            if x.shape() != y.shape() {
                return Err(OjasError::Shape {
                    op: "residual_add_forward",
                    detail: "shapes differ".into(),
                });
            }
            let sum: Vec<f32> = a.iter().zip(b.iter()).map(|(a, b)| a + b).collect();
            Tensor::from_f32(&sum, x.shape(), self.budget())
        }

        fn residual_add_backward(
            &self,
            x: &Tensor,
            y: &Tensor,
            grad_output: &Tensor,
        ) -> Result<(Tensor, Tensor), OjasError> {
            if x.shape() != y.shape() || x.shape() != grad_output.shape() {
                return Err(OjasError::Shape {
                    op: "residual_add_backward",
                    detail: "shapes differ".into(),
                });
            }
            let g = grad_output.f32_slice()?;
            Ok((
                Tensor::from_f32(g, x.shape(), self.budget())?,
                Tensor::from_f32(g, y.shape(), self.budget())?,
            ))
        }

        fn cross_entropy_mean_forward(
            &self,
            logits: &Tensor,
            targets: &Tensor,
            ignore_index: Option<u32>,
        ) -> Result<Tensor, OjasError> {
            const OP: &str = "cross_entropy_mean_forward";
            let logits_v = logits.f32_slice()?;
            let targets_v = targets.u32_slice()?;
            if logits.shape().len() != 2 || targets_v.len() != logits.shape()[0] {
                return Err(OjasError::Shape {
                    op: OP,
                    detail: "logits [N, V] and targets [N]".into(),
                });
            }
            let vocab = logits.shape()[1];
            let mut sum = 0.0f32;
            let mut count = 0u32;
            for (row, &target) in targets_v.iter().enumerate() {
                if ignore_index == Some(target) {
                    continue;
                }
                if target as usize >= vocab {
                    return Err(OjasError::OutOfRange {
                        op: OP,
                        detail: format!("target {target} outside vocab {vocab}"),
                    });
                }
                let sl = &logits_v[row * vocab..(row + 1) * vocab];
                let max = sl.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if !max.is_finite() {
                    return Err(OjasError::NonFinite { op: OP });
                }
                let mut lse = 0.0f32;
                for value in sl {
                    lse += (value - max).exp();
                }
                sum += max + lse.ln() - sl[target as usize];
                count += 1;
            }
            if count == 0 {
                return Err(OjasError::NonFinite { op: OP });
            }
            Tensor::from_f32(&[sum / count as f32], &[1], self.budget())
        }

        unsup!(cross_entropy_mean_backward(logits: &Tensor, targets: &Tensor, ignore_index: Option<u32>) -> Tensor);

        fn clip_grad_norm(&self, grads: &mut [Tensor], _max_norm: f32) -> Result<f32, OjasError> {
            for grad in grads.iter() {
                finite("clip_grad_norm", grad)?;
            }
            Ok(1.0)
        }

        fn adamw_step(
            &self,
            _param: &mut Tensor,
            grad: &Tensor,
            _moment1: &mut Tensor,
            _moment2: &mut Tensor,
            _step: u64,
            _config: AdamWConfig,
        ) -> Result<(), OjasError> {
            finite("adamw_step", grad)?;
            note(&self.grad_bits, grad);
            Ok(())
        }

        fn muon_ns5_step(
            &self,
            _param: &mut Tensor,
            grad: &Tensor,
            _momentum: &mut Tensor,
            _config: MuonNs5Config,
        ) -> Result<(), OjasError> {
            finite("muon_ns5_step", grad)?;
            note(&self.grad_bits, grad);
            Ok(())
        }

        fn linear_cross_entropy_mean(
            &self,
            input: &Tensor,
            weight: &Tensor,
            _targets: &Tensor,
            _ignore_index: Option<u32>,
            _chunk: CeChunk,
            want_grad: bool,
        ) -> Result<LinearCe, OjasError> {
            finite("linear_cross_entropy_mean", input)?;
            finite("linear_cross_entropy_mean", weight)?;
            note(&self.seen_input, input);
            note(&self.seen_weight, weight);
            let loss = Tensor::from_f32(&[f32::from_bits(DIRTY)], &[1], self.budget())?;
            if !want_grad {
                return Ok(LinearCe {
                    loss,
                    grad_input: None,
                    grad_weight: None,
                });
            }
            Ok(LinearCe {
                loss,
                grad_input: Some(dirty(self.budget(), input.shape())?),
                grad_weight: Some(dirty(self.budget(), weight.shape())?),
            })
        }

        fn cached_attention_forward(
            &self,
            q: &Tensor,
            k_cache: &Tensor,
            v_cache: &Tensor,
            _kv_len: usize,
        ) -> Result<Tensor, OjasError> {
            const OP: &str = "cached_attention_forward";
            note(&self.seen_input, q);
            note(&self.seen_k, k_cache);
            note(&self.seen_weight, v_cache);
            finite(OP, q)?;
            finite(OP, k_cache)?;
            finite(OP, v_cache)?;
            dirty(self.budget(), q.shape())
        }

        fn kv_cache_write(
            &self,
            cache: &mut Tensor,
            src: &Tensor,
            _at: usize,
        ) -> Result<(), OjasError> {
            let src = src.f32_slice()?;
            let cache = cache.f32_slice_mut()?;
            if src.len() > cache.len() {
                return Err(OjasError::Shape {
                    op: "kv_cache_write",
                    detail: "src longer than cache".into(),
                });
            }
            cache[..src.len()].copy_from_slice(src);
            Ok(())
        }
    }

    fn rope_apply(
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        backward: bool,
    ) -> Result<Vec<f32>, OjasError> {
        const OP: &str = "rope_half_split_forward";
        let xv = x.f32_slice()?;
        let cv = cos.f32_slice()?;
        let sv = sin.f32_slice()?;
        if xv.len() != cv.len() || xv.len() != sv.len() {
            return Err(OjasError::Shape {
                op: OP,
                detail: "x, cos, and sin differ".into(),
            });
        }
        let d = *x.shape().last().unwrap_or(&0);
        if d == 0 || !d.is_multiple_of(2) || xv.len() % d != 0 {
            return Err(OjasError::Shape {
                op: OP,
                detail: "last axis must be a positive even width".into(),
            });
        }
        let half = d / 2;
        let mut y = vec![0.0f32; xv.len()];
        let rows = xv.len() / d;
        for row in 0..rows {
            for i in 0..half {
                let x1 = xv[row * d + i];
                let x2 = xv[row * d + half + i];
                let c = cv[row * d + i];
                let s = sv[row * d + i];
                if backward {
                    y[row * d + i] = c * x1 + s * x2;
                    y[row * d + half + i] = c * x2 - s * x1;
                } else {
                    y[row * d + i] = x1 * c + (-x2) * s;
                    y[row * d + half + i] = x2 * c + x1 * s;
                }
            }
        }
        Ok(y)
    }

    fn host(budget: &Budget, bits: u32) -> Tensor {
        Tensor::from_f32(&[f32::from_bits(bits)], &[1], budget).unwrap()
    }

    fn tag(tensor: &Tensor) {
        tensor.set_compute_tag(COMPUTE_BF16);
    }

    fn bits_of(tensor: &Tensor) -> u32 {
        tensor.to_f32_vec().unwrap()[0].to_bits()
    }

    fn wrap(cap: u64) -> (Autocast<Rec>, Budget) {
        let budget = Budget::new(cap);
        let autocast = Autocast::new(Rec::new(budget.clone()));
        (autocast, budget)
    }

    #[test]
    fn round_known_vectors_and_every_bf16_pattern() {
        assert_eq!(
            round_f32_to_bf16(f32::from_bits(0x8000_0000)).to_bits(),
            0x8000_0000
        );
        assert_eq!(
            round_f32_to_bf16(f32::from_bits(0x7f80_0001)).to_bits(),
            0x7fc0_0000
        );
        assert_eq!(
            round_f32_to_bf16(f32::from_bits(0x3f80_8000)).to_bits(),
            0x3f80_0000
        );
        assert_eq!(
            round_f32_to_bf16(f32::from_bits(0x3f81_8000)).to_bits(),
            0x3f82_0000
        );
        assert_eq!(round_f32_to_bf16(f32::from_bits(0x0000_0001)).to_bits(), 0);
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7f80);
        assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xff80);
        let quiet = round_f32_to_bf16(f32::from_bits(0x7f80_0001));
        assert_eq!(round_f32_to_bf16(quiet).to_bits(), quiet.to_bits());
        let neg_max = round_f32_to_bf16(f32::from_bits(0xFF7F_FFFF));
        assert_eq!(neg_max.to_bits() & 0xffff, 0);
        for bits in 0..=u16::MAX {
            let value = bf16_to_f32(bits);
            assert_eq!(value.to_bits(), u32::from(bits) << 16);
            let back = f32_to_bf16(value);
            if value.is_nan() {
                assert_eq!(back, bits | 0x0040);
            } else {
                assert_eq!(back, bits);
            }
            let rounded = round_f32_to_bf16(value);
            assert_eq!(rounded.to_bits() & 0xffff, 0);
            assert_eq!(round_f32_to_bf16(rounded).to_bits(), rounded.to_bits());
        }
    }

    #[test]
    fn frame_id_stops_at_u64_max_without_wrapping() {
        let mut next = u64::MAX - 1;
        assert_eq!(alloc_frame_id(&mut next).unwrap(), u64::MAX - 1);
        assert_eq!(next, u64::MAX);
        assert!(matches!(
            alloc_frame_id(&mut next),
            Err(OjasError::OutOfRange {
                op: "autocast_region",
                ..
            })
        ));
        assert_eq!(next, u64::MAX);
    }

    #[test]
    fn rejected_enter_does_not_keep_an_empty_thread_or_advance_the_id() {
        let (autocast, _) = wrap(1 << 12);
        {
            let mut state = autocast.shared.state.lock().unwrap();
            state.next_id = u64::MAX;
        }
        let err = autocast.autocast_region(AutocastMode::Bf16).unwrap_err();
        assert!(matches!(err, OjasError::OutOfRange { .. }));
        let state = autocast.shared.state.lock().unwrap();
        assert!(state.threads.is_empty());
        assert_eq!(state.next_id, u64::MAX);
        assert_eq!(autocast.shared.depth.load(Ordering::Acquire), 0);
    }

    #[test]
    fn regions_are_per_instance_per_thread_and_nest() {
        let (a, _) = wrap(1 << 12);
        let (b, _) = wrap(1 << 12);
        assert!(!a.autocast_enabled().unwrap());
        let outer = a.autocast_region(AutocastMode::Bf16).unwrap();
        assert!(a.autocast_enabled().unwrap());
        assert!(!b.autocast_enabled().unwrap());
        let inner = a.autocast_region(AutocastMode::Off).unwrap();
        assert!(!a.autocast_enabled().unwrap());
        drop(inner);
        assert!(a.autocast_enabled().unwrap());
        let seen = thread::scope(|scope| {
            let a = &a;
            scope
                .spawn(move || a.autocast_enabled().unwrap())
                .join()
                .unwrap()
        });
        assert!(!seen);
        assert!(a.autocast_enabled().unwrap());
        drop(outer);
        assert!(!a.autocast_enabled().unwrap());
    }

    #[test]
    fn guard_is_not_send_and_autocast_is_not_clone() {
        trait AmbiguousIfSend<A> {
            fn some_item() {}
        }
        impl<T: ?Sized> AmbiguousIfSend<()> for T {}
        impl<T: ?Sized + Send> AmbiguousIfSend<u8> for T {}
        let _ = <AutocastGuard as AmbiguousIfSend<_>>::some_item;

        trait AmbiguousIfClone<A> {
            fn some_item() {}
        }
        impl<T: ?Sized> AmbiguousIfClone<()> for T {}
        impl<T: Clone> AmbiguousIfClone<u8> for T {}
        let _ = <Autocast<Rec> as AmbiguousIfClone<_>>::some_item;
    }

    #[test]
    fn unwind_restores_the_outer_region() {
        let (autocast, _) = wrap(1 << 12);
        let _outer = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let caught = catch_unwind(AssertUnwindSafe(|| {
            let _inner = autocast.autocast_region(AutocastMode::Off).unwrap();
            panic!("inner region");
        }));
        assert!(caught.is_err());
        assert!(autocast.autocast_enabled().unwrap());
    }

    #[test]
    fn a_thousand_regions_return_to_off() {
        let (autocast, _) = wrap(1 << 12);
        for _ in 0..1000 {
            let guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
            assert!(autocast.autocast_enabled().unwrap());
            drop(guard);
            assert!(!autocast.autocast_enabled().unwrap());
        }
    }

    #[test]
    fn depth_cap_leaves_the_stack_and_forget_holds_it() {
        let (autocast, _) = wrap(1 << 12);
        let mut guards = Vec::new();
        for _ in 0..MAX_DEPTH {
            guards.push(autocast.autocast_region(AutocastMode::Bf16).unwrap());
        }
        assert!(matches!(
            autocast.autocast_region(AutocastMode::Off),
            Err(OjasError::OutOfRange { .. })
        ));
        assert!(autocast.autocast_enabled().unwrap());
        assert_eq!(
            autocast.shared.depth.load(Ordering::Acquire) & !POISON_BIT,
            MAX_DEPTH as u32
        );
        for guard in guards {
            std::mem::forget(guard);
        }
        assert!(autocast.autocast_enabled().unwrap());
        assert!(matches!(
            autocast.autocast_region(AutocastMode::Bf16),
            Err(OjasError::OutOfRange { .. })
        ));
    }

    #[test]
    fn thread_cap_releases_when_the_guards_drop() {
        let autocast = Arc::new(Autocast::new(Rec::new(Budget::new(1 << 12))));
        let arrived = Arc::new(Barrier::new(MAX_THREADS + 1));
        let release = Arc::new(Barrier::new(MAX_THREADS + 1));
        let mut handles = Vec::new();
        for _ in 0..MAX_THREADS {
            let autocast = Arc::clone(&autocast);
            let arrived = Arc::clone(&arrived);
            let release = Arc::clone(&release);
            handles.push(thread::spawn(move || {
                let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
                arrived.wait();
                release.wait();
            }));
        }
        arrived.wait();
        assert!(matches!(
            autocast.autocast_region(AutocastMode::Bf16),
            Err(OjasError::OutOfRange { .. })
        ));
        release.wait();
        for handle in handles {
            handle.join().unwrap();
        }
        thread::spawn({
            let autocast = Arc::clone(&autocast);
            move || {
                let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
                assert!(autocast.autocast_enabled().unwrap());
            }
        })
        .join()
        .unwrap();
    }

    #[test]
    fn overlapping_linears_do_not_poison() {
        let autocast = Arc::new(Autocast::new(Rec::new(Budget::new(1 << 20))));
        let start = Arc::new(Barrier::new(8));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let autocast = Arc::clone(&autocast);
            let start = Arc::clone(&start);
            handles.push(thread::spawn(move || {
                let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
                start.wait();
                for _ in 0..20 {
                    let x = host(autocast.budget(), ONE);
                    let w = host(autocast.budget(), ONE);
                    autocast.linear_forward(&x, &w).unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        assert!(!autocast.autocast_enabled().unwrap());
        assert_eq!(autocast.inner.linear_calls.load(Ordering::Relaxed), 160);
    }

    #[test]
    fn non_lifo_drop_poisons_every_thread_and_leaves_accessors() {
        let (autocast, _) = wrap(1 << 16);
        let first = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let second = autocast.autocast_region(AutocastMode::Off).unwrap();
        drop(first);
        let depth = autocast.shared.depth.load(Ordering::Acquire);
        assert_ne!(depth & POISON_BIT, 0);
        assert_eq!(depth & !POISON_BIT, 2);
        drop(second);
        assert_eq!(
            autocast.shared.depth.load(Ordering::Acquire) & !POISON_BIT,
            2
        );
        assert!(matches!(
            autocast.autocast_enabled(),
            Err(OjasError::Poisoned)
        ));
        assert_eq!(autocast.id(), BackendId::Cpu);
        assert_eq!(autocast.numerics(), Numerics::Exact);
        let _ = autocast.budget();
        let x = host(autocast.budget(), ONE);
        let err = autocast.linear_forward(&x, &x).unwrap_err();
        assert!(matches!(err, OjasError::Poisoned));
        assert_eq!(autocast.inner.linear_calls.load(Ordering::Relaxed), 0);
        let autocast = Arc::new(autocast);
        let err = thread::spawn({
            let autocast = Arc::clone(&autocast);
            move || autocast.sync().unwrap_err()
        })
        .join()
        .unwrap();
        assert!(matches!(err, OjasError::Poisoned));
    }

    #[test]
    fn panic_inside_held_sets_the_bit_without_poisoning_the_mutex() {
        let (autocast, _) = wrap(1 << 12);
        let shared = Arc::clone(&autocast.shared);
        let caught = catch_unwind(AssertUnwindSafe(|| {
            let _ = with_state(&shared, |_| -> Result<(), OjasError> {
                panic!("held");
            });
        }));
        assert!(caught.is_err());
        assert_ne!(
            autocast.shared.depth.load(Ordering::Acquire) & POISON_BIT,
            0
        );
        let mut held = lock(&autocast.shared).unwrap();
        assert!(held.guard.poisoned);
        held.disarm = true;
        assert!(matches!(
            autocast.autocast_enabled(),
            Err(OjasError::Poisoned)
        ));
    }

    #[test]
    fn panic_while_holding_the_raw_mutex_is_poison() {
        let (autocast, _) = wrap(1 << 12);
        let shared = Arc::clone(&autocast.shared);
        let _ = thread::spawn(move || {
            let _guard = shared.state.lock().unwrap();
            panic!("raw mutex");
        })
        .join();
        assert!(matches!(lock(&autocast.shared), Err(OjasError::Poisoned)));
        assert_ne!(
            autocast.shared.depth.load(Ordering::Acquire) & POISON_BIT,
            0
        );
    }

    #[test]
    fn fast_numerics_pass_through_and_off_linear_is_not_rounded() {
        let budget = Budget::new(1 << 16);
        let autocast = Autocast::new(Rec::new(budget.clone()).with_numerics(Numerics::Fast));
        assert_eq!(autocast.numerics(), Numerics::Fast);
        let x = host(&budget, DIRTY);
        let y = autocast.linear_forward(&x, &x).unwrap();
        assert_eq!(bits_of(&y), DIRTY);
        assert_eq!(y.compute_tag(), COMPUTE_F32);
        assert_eq!(autocast.inner.casts.load(Ordering::Relaxed), 0);
        assert_eq!(autocast.inner.linear_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn bf16_region_rounds_linear_operands_and_the_activation() {
        let (autocast, budget) = wrap(1 << 16);
        let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let x = host(&budget, DIRTY);
        let w = host(&budget, DIRTY);
        let y = autocast.linear_forward(&x, &w).unwrap();
        assert_eq!(autocast.inner.seen_input.load(Ordering::Relaxed), ONE);
        assert_eq!(autocast.inner.seen_weight.load(Ordering::Relaxed), ONE);
        assert_eq!(bits_of(&y), ONE);
        assert_eq!(y.compute_tag(), COMPUTE_BF16);
        assert_eq!(autocast.inner.casts.load(Ordering::Relaxed), 3);
        assert_eq!(bits_of(&x), DIRTY);
    }

    #[test]
    fn a_tagged_operand_is_not_cast_again() {
        let (autocast, budget) = wrap(1 << 16);
        let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let x = host(&budget, ONE);
        tag(&x);
        let w = host(&budget, ONE);
        tag(&w);
        let y = autocast.linear_forward(&x, &w).unwrap();
        assert_eq!(y.compute_tag(), COMPUTE_BF16);
        assert_eq!(autocast.inner.casts.load(Ordering::Relaxed), 1);
        let again = autocast.cast_bf16(&y).unwrap();
        assert_eq!(autocast.inner.casts.load(Ordering::Relaxed), 1);
        assert_eq!(again.compute_tag(), COMPUTE_BF16);
    }

    #[test]
    fn explicit_cast_outside_a_region_rounds_once() {
        let (autocast, budget) = wrap(1 << 16);
        let x = host(&budget, 0x3f81_8000);
        let y = autocast.cast_bf16(&x).unwrap();
        assert_eq!(bits_of(&y), 0x3f82_0000);
        assert_eq!(y.compute_tag(), COMPUTE_BF16);
        assert_eq!(autocast.inner.casts.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn signalling_nan_cast_reaches_linear_as_non_finite() {
        let (autocast, budget) = wrap(1 << 16);
        let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let x = host(&budget, 0x7f80_0001);
        let w = host(&budget, ONE);
        let err = autocast.linear_forward(&x, &w).unwrap_err();
        assert!(matches!(
            err,
            OjasError::NonFinite {
                op: "linear_forward"
            }
        ));
        assert_eq!(autocast.inner.linear_calls.load(Ordering::Relaxed), 1);
        assert!(autocast.inner.casts.load(Ordering::Relaxed) >= 1);
    }

    #[test]
    fn cast_capacity_does_not_call_linear_or_keep_the_charge() {
        let budget = Budget::new(8);
        let autocast = Autocast::new(Rec::new(budget.clone()));
        let x = host(&budget, ONE);
        let w = host(&budget, DIRTY);
        assert_eq!(budget.live_bytes().unwrap(), 8);
        let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let err = autocast.linear_forward(&x, &w).unwrap_err();
        assert!(matches!(
            err,
            OjasError::CapacityExceeded {
                requested: 4,
                cap: 8,
                live: 8
            }
        ));
        assert_eq!(autocast.inner.linear_calls.load(Ordering::Relaxed), 0);
        assert_eq!(budget.live_bytes().unwrap(), 8);
    }

    #[derive(Debug)]
    struct FakeDev {
        backend: BackendId,
        bytes: Vec<u8>,
        reads: AtomicU64,
    }

    impl DeviceBuffer for FakeDev {
        fn backend(&self) -> BackendId {
            self.backend
        }
        fn byte_len(&self) -> usize {
            self.bytes.len()
        }
        fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            let end = offset
                .checked_add(len)
                .ok_or_else(|| OjasError::OutOfRange {
                    op: "read_bytes",
                    detail: "offset overflow".into(),
                })?;
            if end > self.bytes.len() {
                return Err(OjasError::OutOfRange {
                    op: "read_bytes",
                    detail: "read past the buffer".into(),
                });
            }
            Ok(self.bytes[offset..end].to_vec())
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn device_linear_refuses_without_a_readback() {
        let (autocast, budget) = wrap(1 << 16);
        let device = Arc::new(FakeDev {
            backend: BackendId::Metal,
            bytes: vec![0; 4],
            reads: AtomicU64::new(0),
        });
        let x = Tensor::from_device(device.clone(), &[1], DType::F32, &budget).unwrap();
        let w = host(&budget, ONE);
        let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let err = autocast.linear_forward(&x, &w).unwrap_err();
        assert!(matches!(
            err,
            OjasError::Unsupported {
                op: "cast_bf16",
                ..
            }
        ));
        assert_eq!(autocast.inner.linear_calls.load(Ordering::Relaxed), 0);
        assert_eq!(device.reads.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn permute_keeps_a_signalling_nan_and_the_tag() {
        let (autocast, budget) = wrap(1 << 16);
        let values = Tensor::from_f32(&[f32::from_bits(0x7f80_0001), 1.0], &[2], &budget).unwrap();
        tag(&values);
        let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let out = autocast.permute(&values, &[0]).unwrap();
        assert_eq!(out.to_f32_vec().unwrap()[0].to_bits(), 0x7f80_0001);
        assert_eq!(out.compute_tag(), COMPUTE_BF16);
        assert_eq!(autocast.inner.casts.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn master_grads_stay_f32_and_activation_grads_round() {
        let (autocast, budget) = wrap(1 << 16);
        let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let input = host(&budget, DIRTY);
        let weight = host(&budget, DIRTY);
        let grad = host(&budget, DIRTY);
        let (grad_input, grad_weight) = autocast.linear_backward(&input, &weight, &grad).unwrap();
        assert_eq!(bits_of(&grad_input), ONE);
        assert_eq!(grad_input.compute_tag(), COMPUTE_BF16);
        assert_eq!(bits_of(&grad_weight), DIRTY);
        assert_eq!(grad_weight.compute_tag(), COMPUTE_F32);
        assert_eq!(autocast.inner.seen_input.load(Ordering::Relaxed), ONE);
    }

    #[test]
    fn sdpa_and_cached_attention_round_copies_and_leave_the_cache() {
        let (autocast, budget) = wrap(1 << 16);
        let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let q = host(&budget, DIRTY);
        let k = host(&budget, DIRTY);
        let v = host(&budget, DIRTY);
        let out = autocast.causal_sdpa_forward(&q, &k, &v).unwrap();
        assert_eq!(out.compute_tag(), COMPUTE_BF16);
        assert_eq!(bits_of(&out), ONE);
        assert_eq!(autocast.inner.seen_k.load(Ordering::Relaxed), ONE);
        assert_eq!(bits_of(&k), DIRTY);
        let (gq, gk, gv) = autocast.causal_sdpa_backward(&q, &k, &v, &out).unwrap();
        assert_eq!(gq.compute_tag(), COMPUTE_BF16);
        assert_eq!(gk.compute_tag(), COMPUTE_BF16);
        assert_eq!(gv.compute_tag(), COMPUTE_BF16);
        let cached = autocast.cached_attention_forward(&q, &k, &v, 1).unwrap();
        assert_eq!(cached.compute_tag(), COMPUTE_BF16);
        assert_eq!(bits_of(&k), DIRTY);
        assert_eq!(autocast.inner.seen_k.load(Ordering::Relaxed), ONE);
    }

    #[test]
    fn gate_skips_bias_and_keeps_weight_grads_and_scales() {
        let (autocast, budget) = wrap(1 << 20);
        let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let input = host(&budget, DIRTY);
        let weight = host(&budget, DIRTY);
        let bias = host(&budget, DIRTY);
        let attn = host(&budget, DIRTY);
        let (out, scales) = autocast
            .per_head_sigmoid_gate_forward_saving(&input, &weight, &bias, &attn)
            .unwrap();
        assert_eq!(autocast.inner.saving_calls.load(Ordering::Relaxed), 1);
        assert_eq!(autocast.inner.seen_bias.load(Ordering::Relaxed), DIRTY);
        assert_eq!(autocast.inner.seen_input.load(Ordering::Relaxed), ONE);
        assert_eq!(out.compute_tag(), COMPUTE_BF16);
        let scales = scales.unwrap();
        assert_eq!(bits_of(&scales), DIRTY);
        assert_eq!(scales.compute_tag(), COMPUTE_F32);
        let grad = autocast
            .per_head_sigmoid_gate_backward_saved(&input, &weight, &bias, &attn, &out, &scales)
            .unwrap();
        assert_eq!(autocast.inner.saved_calls.load(Ordering::Relaxed), 1);
        assert_eq!(autocast.inner.scale_bits.load(Ordering::Relaxed), DIRTY);
        assert_eq!(grad.input.compute_tag(), COMPUTE_BF16);
        assert_eq!(grad.attn_out.compute_tag(), COMPUTE_BF16);
        assert_eq!(grad.weight.compute_tag(), COMPUTE_F32);
        assert_eq!(bits_of(&grad.weight), DIRTY);
        assert_eq!(grad.bias.compute_tag(), COMPUTE_F32);
        assert_eq!(bits_of(&grad.bias), DIRTY);
    }

    #[test]
    fn promotion_follows_tagged_activations_only() {
        let (autocast, budget) = wrap(1 << 20);
        let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let tagged_one = host(&budget, ONE);
        tag(&tagged_one);
        let dirty = host(&budget, DIRTY);
        let sum = autocast.residual_add_forward(&tagged_one, &dirty).unwrap();
        let raw = f32::from_bits(ONE) + f32::from_bits(DIRTY);
        assert_eq!(bits_of(&sum), raw.to_bits());
        assert_eq!(sum.compute_tag(), COMPUTE_F32);

        let both = host(&budget, ONE);
        tag(&both);
        let other = host(&budget, 0x3f00_0000);
        tag(&other);
        let product = autocast.mul_forward(&both, &other).unwrap();
        assert_eq!(product.compute_tag(), COMPUTE_BF16);
        assert_eq!(bits_of(&product) & 0xffff, 0);

        let x = Tensor::from_f32(
            &[f32::from_bits(DIRTY), f32::from_bits(DIRTY)],
            &[2],
            &budget,
        )
        .unwrap();
        let cos = Tensor::from_f32(&[1.0, 1.0], &[2], &budget).unwrap();
        let sin = Tensor::from_f32(&[0.0, 0.0], &[2], &budget).unwrap();
        tag(&cos);
        tag(&sin);
        let rotated = autocast.rope_half_split_forward(&x, &cos, &sin).unwrap();
        assert_eq!(rotated.to_f32_vec().unwrap()[0].to_bits(), DIRTY);
        assert_eq!(rotated.compute_tag(), COMPUTE_F32);

        let value = host(&budget, ONE);
        let value0 = host(&budget, 0x4000_0000);
        tag(&value);
        tag(&value0);
        let lambda = host(&budget, DIRTY);
        let blended = autocast
            .value_residual_blend_forward(&value, &value0, &lambda)
            .unwrap();
        let s = sigmoid(f32::from_bits(DIRTY));
        let expect = (1.0 - s) * 1.0 + s * 2.0;
        assert_eq!(bits_of(&blended), expect.to_bits());
        assert_eq!(blended.compute_tag(), COMPUTE_F32);
        // The mul above is the only cast: its product is a new untagged tensor.
        assert_eq!(autocast.inner.casts.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn silu_of_a_tagged_input_rounds_and_an_untagged_one_does_not() {
        let (autocast, budget) = wrap(1 << 16);
        let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let tagged = host(&budget, ONE);
        tag(&tagged);
        let out = autocast.silu_forward(&tagged).unwrap();
        let raw = 1.0 * sigmoid(1.0);
        assert_ne!(raw.to_bits() & 0xffff, 0);
        assert_eq!(bits_of(&out), round_f32_to_bf16(raw).to_bits());
        assert_eq!(out.compute_tag(), COMPUTE_BF16);
        let plain = host(&budget, ONE);
        let left = autocast.silu_forward(&plain).unwrap();
        assert_eq!(bits_of(&left), raw.to_bits());
        assert_eq!(left.compute_tag(), COMPUTE_F32);
    }

    #[test]
    fn fp32_ops_do_not_round_and_clear_tags_only_after_success() {
        let (autocast, budget) = wrap(1 << 20);
        let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let table = Tensor::from_f32(&[1.5, 2.5], &[2, 1], &budget).unwrap();
        let ids = Tensor::from_u32(&[1], &[1], &budget).unwrap();
        let embedded = autocast.embedding_forward(&table, &ids).unwrap();
        assert_eq!(embedded.to_f32_vec().unwrap(), vec![2.5]);
        assert_eq!(embedded.compute_tag(), COMPUTE_F32);

        let row = Tensor::from_f32(&[3.0, 4.0], &[2], &budget).unwrap();
        let weight = Tensor::from_f32(&[1.0, 1.0], &[2], &budget).unwrap();
        tag(&row);
        let normed = autocast.rms_norm_forward(&row, &weight, 1e-6).unwrap();
        assert_eq!(normed.compute_tag(), COMPUTE_F32);
        assert_ne!(normed.to_f32_vec().unwrap()[0].to_bits() & 0xffff, 0);

        let logits = Tensor::from_f32(&[0.0, 0.0], &[1, 2], &budget).unwrap();
        let targets = Tensor::from_u32(&[0], &[1], &budget).unwrap();
        let loss = autocast
            .cross_entropy_mean_forward(&logits, &targets, None)
            .unwrap();
        assert!(loss.to_f32_vec().unwrap()[0].is_finite());
        assert_eq!(loss.compute_tag(), COMPUTE_F32);

        let mut acc = host(&budget, ONE);
        tag(&acc);
        let grad = host(&budget, DIRTY);
        tag(&grad);
        let before = autocast.inner.casts.load(Ordering::Relaxed);
        autocast.accumulate_grad(&mut acc, &grad).unwrap();
        let raw = f32::from_bits(ONE) + f32::from_bits(DIRTY);
        assert_eq!(bits_of(&acc), raw.to_bits());
        assert_eq!(acc.compute_tag(), COMPUTE_F32);
        assert_eq!(autocast.inner.casts.load(Ordering::Relaxed), before);
        let wrong = Tensor::from_f32(&[1.0, 2.0], &[2], &budget).unwrap();
        tag(&acc);
        assert!(autocast.accumulate_grad(&mut acc, &wrong).is_err());
        assert_eq!(acc.compute_tag(), COMPUTE_BF16);

        let mut param = host(&budget, ONE);
        let mut moment1 = host(&budget, ONE);
        let mut moment2 = host(&budget, ONE);
        let step_grad = host(&budget, DIRTY);
        tag(&param);
        tag(&moment1);
        tag(&moment2);
        autocast
            .adamw_step(
                &mut param,
                &step_grad,
                &mut moment1,
                &mut moment2,
                0,
                AdamWConfig::nanolab(1e-3, 0.0),
            )
            .unwrap();
        assert_eq!(autocast.inner.grad_bits.load(Ordering::Relaxed), DIRTY);
        assert_eq!(param.compute_tag(), COMPUTE_F32);
        assert_eq!(moment1.compute_tag(), COMPUTE_F32);
        assert_eq!(moment2.compute_tag(), COMPUTE_F32);
        assert_eq!(autocast.inner.casts.load(Ordering::Relaxed), before);

        let mut momentum = host(&budget, ONE);
        tag(&param);
        tag(&momentum);
        autocast
            .muon_ns5_step(
                &mut param,
                &step_grad,
                &mut momentum,
                MuonNs5Config::nanolab_default(),
            )
            .unwrap();
        assert_eq!(autocast.inner.grad_bits.load(Ordering::Relaxed), DIRTY);
        assert_eq!(param.compute_tag(), COMPUTE_F32);
        assert_eq!(momentum.compute_tag(), COMPUTE_F32);

        let nan = host(&budget, 0x7f80_0001);
        tag(&param);
        let err = autocast
            .adamw_step(
                &mut param,
                &nan,
                &mut moment1,
                &mut moment2,
                0,
                AdamWConfig::nanolab(1e-3, 0.0),
            )
            .unwrap_err();
        assert!(matches!(err, OjasError::NonFinite { op: "adamw_step" }));
        assert_eq!(param.compute_tag(), COMPUTE_BF16);

        let mut clipped = host(&budget, DIRTY);
        tag(&clipped);
        let norm = autocast
            .clip_grad_norm(std::slice::from_mut(&mut clipped), 1.0)
            .unwrap();
        assert_eq!(norm, 1.0);
        assert_eq!(bits_of(&clipped), DIRTY);
        assert_eq!(clipped.compute_tag(), COMPUTE_F32);

        let mut cache = host(&budget, ONE);
        tag(&cache);
        let src = host(&budget, DIRTY);
        tag(&src);
        autocast.kv_cache_write(&mut cache, &src, 0).unwrap();
        assert_eq!(bits_of(&cache), DIRTY);
        assert_eq!(cache.compute_tag(), COMPUTE_F32);
        assert_eq!(autocast.inner.casts.load(Ordering::Relaxed), before);
    }

    #[test]
    fn fused_loss_rounds_activation_grad_only() {
        let (autocast, budget) = wrap(1 << 16);
        let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let input = host(&budget, DIRTY);
        let weight = host(&budget, DIRTY);
        let targets = Tensor::from_u32(&[0], &[1], &budget).unwrap();
        let out = autocast
            .linear_cross_entropy_mean(
                &input,
                &weight,
                &targets,
                None,
                CeChunk { rows: 1, cols: 1 },
                true,
            )
            .unwrap();
        assert_eq!(bits_of(&out.loss), DIRTY);
        assert_eq!(out.loss.compute_tag(), COMPUTE_F32);
        let grad_input = out.grad_input.unwrap();
        let grad_weight = out.grad_weight.unwrap();
        assert_eq!(bits_of(&grad_input), ONE);
        assert_eq!(grad_input.compute_tag(), COMPUTE_BF16);
        assert_eq!(bits_of(&grad_weight), DIRTY);
        assert_eq!(grad_weight.compute_tag(), COMPUTE_F32);
        assert_eq!(autocast.inner.seen_input.load(Ordering::Relaxed), ONE);
    }

    #[test]
    fn storage_tag_follows_reshape_and_clears_on_mutation() {
        let budget = Budget::new(1 << 16);
        let tensor = Tensor::from_f32(&[1.0, 2.0], &[2], &budget).unwrap();
        tag(&tensor);
        let viewed = tensor.reshape(&[1, 2]).unwrap();
        assert_eq!(viewed.compute_tag(), COMPUTE_BF16);
        viewed.set_compute_tag(COMPUTE_F32);
        assert_eq!(tensor.compute_tag(), COMPUTE_F32);
        drop(viewed);

        tag(&tensor);
        let mut unique = tensor;
        unique.f32_slice_mut().unwrap()[0] = 3.0;
        assert_eq!(unique.compute_tag(), COMPUTE_F32);

        let shared = host(&budget, ONE);
        tag(&shared);
        let mut alias = shared.clone();
        assert!(alias.f32_slice_mut().is_err());
        assert_eq!(shared.compute_tag(), COMPUTE_BF16);
        assert_eq!(alias.compute_tag(), COMPUTE_BF16);

        let mut host_tensor = host(&budget, ONE);
        tag(&host_tensor);
        assert!(matches!(
            host_tensor.device_buffer_mut(),
            Err(OjasError::Placement { .. })
        ));
        assert_eq!(host_tensor.compute_tag(), COMPUTE_BF16);

        let device = Arc::new(FakeDev {
            backend: BackendId::Wgpu,
            bytes: vec![0; 4],
            reads: AtomicU64::new(0),
        });
        let extra = Arc::clone(&device);
        let mut device_tensor = Tensor::from_device(device, &[1], DType::F32, &budget).unwrap();
        tag(&device_tensor);
        assert!(device_tensor.device_buffer_mut().is_err());
        assert_eq!(device_tensor.compute_tag(), COMPUTE_F32);
        drop(extra);

        let device = Arc::new(FakeDev {
            backend: BackendId::Wgpu,
            bytes: vec![0; 4],
            reads: AtomicU64::new(0),
        });
        let mut device_tensor = Tensor::from_device(device, &[1], DType::F32, &budget).unwrap();
        tag(&device_tensor);
        device_tensor.device_buffer_mut().unwrap();
        assert_eq!(device_tensor.compute_tag(), COMPUTE_F32);
    }

    #[test]
    fn upload_and_download_copy_the_tag_without_rounding() {
        let (autocast, budget) = wrap(1 << 12);
        let _guard = autocast.autocast_region(AutocastMode::Bf16).unwrap();
        let tensor = host(&budget, DIRTY);
        tag(&tensor);
        let up = autocast.upload(&tensor).unwrap();
        assert_eq!(bits_of(&up), DIRTY);
        assert_eq!(up.compute_tag(), COMPUTE_BF16);
        let down = autocast.download(&tensor).unwrap();
        assert_eq!(bits_of(&down), DIRTY);
        assert_eq!(down.compute_tag(), COMPUTE_BF16);
        assert_eq!(autocast.inner.casts.load(Ordering::Relaxed), 0);
    }
}
