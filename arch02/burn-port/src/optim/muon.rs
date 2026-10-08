//! Muon with per-step momentum warmup (0.92 → 0.99 over 1500 steps).
//!
//! Two implementations live here:
//!
//! * [`MuonWarm`] — a `SimpleOptimizer` that Burn drives one parameter at a
//!   time. Correct and simple, but on Apple Silicon it issues ~66 independent
//!   Newton-Schulz loops per step (each 5 iterations × 2 matmuls of tiny
//!   matrices), i.e. hundreds of small kernel launches whose per-dispatch
//!   overhead dominates the actual math. Kept for reference and unit tests.
//!
//! * [`BankedMuon`] — the production path. It mirrors the Python reference's
//!   *parameter banks*: the 66 block matrices are gathered into four
//!   shape-homogeneous stacks and Newton-Schulz runs as **batched** matmuls
//!   over the stack dimension. One NS5 launch set covers 22 (Q+O) or 22 (K+V)
//!   or 11 (up) or 11 (down) matrices at once — ~50× fewer dispatches.
//!
//! Update (both):
//!   buf = m·buf + g ; go = g + m·buf (nesterov) ; u = NS5(go)
//!   p  = p·(1 − lr·wd) − lr·scale·u
//! where scale = sqrt(max(1, out/in)) on the torch [out,in] layout. Our
//! weights are stored [in,out], so scale = sqrt(max(1, cols/rows)); within a
//! bank every matrix shares a shape, hence a single scalar scale per bank.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::module::Param;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{GradientsParams, SimpleOptimizer};
use burn::prelude::*;
use burn::record::Record;
use burn::tensor::backend::AutodiffBackend;

use super::init::newton_schulz5;
use crate::model::Gpt;

// Keller-Jordan quintic coefficients (shared with init::newton_schulz5).
const NS_A: f64 = 3.4445;
const NS_B: f64 = -4.7750;
const NS_C: f64 = 2.0315;

/// Shared, per-step-mutable momentum coefficient.
#[derive(Clone, Debug)]
pub struct MomentumHandle(Arc<AtomicU64>);

impl MomentumHandle {
    pub fn new(initial: f64) -> Self {
        Self(Arc::new(AtomicU64::new(initial.to_bits())))
    }
    pub fn set(&self, m: f64) {
        self.0.store(m.to_bits(), Ordering::Relaxed);
    }
    pub fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }
}

// ===========================================================================
// Batched (banked) Muon — production path
// ===========================================================================

/// Ordered ParamIds of the six Muon-optimized matrices per block. Each vector
/// has `num_layers` entries; index `i` is block `i`.
#[derive(Clone, Debug)]
pub struct MuonBankIds {
    pub q: Vec<burn::module::ParamId>,
    pub out: Vec<burn::module::ParamId>,
    pub k: Vec<burn::module::ParamId>,
    pub v: Vec<burn::module::ParamId>,
    pub up: Vec<burn::module::ParamId>,
    pub down: Vec<burn::module::ParamId>,
}

impl MuonBankIds {
    pub fn from_model<B: AutodiffBackend>(model: &Gpt<B>) -> Self {
        let mut ids = MuonBankIds {
            q: vec![],
            out: vec![],
            k: vec![],
            v: vec![],
            up: vec![],
            down: vec![],
        };
        for b in &model.blocks {
            ids.q.push(b.attn.q_w.id);
            ids.out.push(b.attn.out_w.id);
            ids.k.push(b.attn.k_w.id);
            ids.v.push(b.attn.v_w.id);
            ids.up.push(b.mlp.up_w.id);
            ids.down.push(b.mlp.down_w.id);
        }
        ids
    }
}

/// The four shape-homogeneous gradient banks, stacked once per step and shared
/// by the global grad clip AND the Muon update (the clip scales these stacks
/// in place instead of touching 66 individual tensors).
pub struct BankStacks<NB: Backend> {
    pub qo: Tensor<NB, 3>, // [2L, 512, 512]  (q then out)
    pub kv: Tensor<NB, 3>, // [2L, 512, 256]  (k then v)
    pub up: Tensor<NB, 3>, // [L, 512, 1536]
    pub dn: Tensor<NB, 3>, // [L, 1536, 512]
}

impl<NB: Backend> BankStacks<NB> {
    /// Remove the Muon-group grads from `grads` and stack them by bank.
    pub fn from_grads<B>(grads: &mut GradientsParams, ids: &MuonBankIds) -> Self
    where
        B: AutodiffBackend<InnerBackend = NB>,
    {
        Self {
            qo: stack_grads::<B, NB>(grads, &ids.q, &ids.out),
            kv: stack_grads::<B, NB>(grads, &ids.k, &ids.v),
            up: stack_grads::<B, NB>(grads, &ids.up, &[]),
            dn: stack_grads::<B, NB>(grads, &ids.down, &[]),
        }
    }

    /// Sum of squared elements across all four banks → device scalar [1].
    pub fn sq_norm(&self) -> Tensor<NB, 1> {
        let parts = vec![
            (self.qo.clone() * self.qo.clone()).sum(),
            (self.kv.clone() * self.kv.clone()).sum(),
            (self.up.clone() * self.up.clone()).sum(),
            (self.dn.clone() * self.dn.clone()).sum(),
        ];
        Tensor::cat(parts, 0).sum()
    }

    /// Scale every bank by a device scalar (broadcast, 4 dispatches).
    pub fn scale(&mut self, factor: Tensor<NB, 1>) {
        let f = factor.reshape([1, 1, 1]);
        self.qo = self.qo.clone() * f.clone();
        self.kv = self.kv.clone() * f.clone();
        self.up = self.up.clone() * f.clone();
        self.dn = self.dn.clone() * f;
    }
}

/// Batched Muon. `NB` is the *inner* (non-autodiff) backend; momentum banks
/// live there. Construct with [`BankedMuon::new`] and drive with
/// [`BankedMuon::step`], mirroring `model = opt.step(lr, model, grads)`.
pub struct BankedMuon<NB: Backend> {
    pub momentum: MomentumHandle,
    pub weight_decay: f64,
    pub ns_steps: usize,
    pub eps: f64,
    buf_qo: Option<Tensor<NB, 3>>, // [2L, 512, 512]
    buf_kv: Option<Tensor<NB, 3>>, // [2L, 512, 256]
    buf_up: Option<Tensor<NB, 3>>, // [L, 512, 1536]
    buf_dn: Option<Tensor<NB, 3>>, // [L, 1536, 512]
}

/// Per-bank NS5 timings (ms), only populated when profiling is requested.
#[derive(Default, Debug, Clone)]
pub struct MuonProfile {
    pub qo_ms: f64,
    pub kv_ms: f64,
    pub up_ms: f64,
    pub dn_ms: f64,
}

impl MuonProfile {
    pub fn pairs(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("qo", self.qo_ms),
            ("kv", self.kv_ms),
            ("up", self.up_ms),
            ("dn", self.dn_ms),
        ]
    }
}

impl<NB: Backend> BankedMuon<NB> {
    pub fn new(momentum: MomentumHandle, weight_decay: f64, ns_steps: usize, eps: f64) -> Self {
        Self {
            momentum,
            weight_decay,
            ns_steps,
            eps,
            buf_qo: None,
            buf_kv: None,
            buf_up: None,
            buf_dn: None,
        }
    }

    /// One optimizer step over the four banks (already stacked — see
    /// [`BankStacks::from_grads`]). Returns the updated model.
    pub fn step<B>(
        &mut self,
        lr: f64,
        mut model: Gpt<B>,
        stacks: BankStacks<NB>,
        device: &NB::Device,
        profile: bool,
    ) -> (Gpt<B>, MuonProfile)
    where
        B: AutodiffBackend<InnerBackend = NB>,
    {
        let m = self.momentum.get();
        let mut prof = MuonProfile::default();
        let l = model.blocks.len();

        // scale = sqrt(max(1, cols/rows)) per bank — derived from the actual
        // stack shapes so every model size gets the reference LR scaling.
        let [_, qo_r, qo_c] = stacks.qo.dims();
        let [_, kv_r, kv_c] = stacks.kv.dims();
        let [_, up_r, up_c] = stacks.up.dims();
        let [_, dn_r, dn_c] = stacks.dn.dims();
        let s_qo = bank_scale(qo_r, qo_c);
        let s_kv = bank_scale(kv_r, kv_c);
        let s_up = bank_scale(up_r, up_c);
        let s_dn = bank_scale(dn_r, dn_c);

        let u_qo = self.bank_update(stacks.qo, BankKind::Qo, m, device, profile, &mut prof.qo_ms);
        let u_kv = self.bank_update(stacks.kv, BankKind::Kv, m, device, profile, &mut prof.kv_ms);
        let u_up = self.bank_update(stacks.up, BankKind::Up, m, device, profile, &mut prof.up_ms);
        let u_dn = self.bank_update(stacks.dn, BankKind::Dn, m, device, profile, &mut prof.dn_ms);
        let uq = split_stack::<NB>(u_qo.clone(), 0, l, qo_r, qo_c);
        let uo = split_stack::<NB>(u_qo, l, l, qo_r, qo_c);
        let uk = split_stack::<NB>(u_kv.clone(), 0, l, kv_r, kv_c);
        let uv = split_stack::<NB>(u_kv, l, l, kv_r, kv_c);
        let uu = split_stack::<NB>(u_up, 0, l, up_r, up_c);
        let ud = split_stack::<NB>(u_dn, 0, l, dn_r, dn_c);

        let wd = self.weight_decay;
        let blocks = model
            .blocks
            .into_iter()
            .enumerate()
            .map(|(i, mut blk)| {
                blk.attn.q_w = apply_update::<B>(blk.attn.q_w, uq[i].clone(), lr, wd, s_qo);
                blk.attn.out_w = apply_update::<B>(blk.attn.out_w, uo[i].clone(), lr, wd, s_qo);
                blk.attn.k_w = apply_update::<B>(blk.attn.k_w, uk[i].clone(), lr, wd, s_kv);
                blk.attn.v_w = apply_update::<B>(blk.attn.v_w, uv[i].clone(), lr, wd, s_kv);
                blk.mlp.up_w = apply_update::<B>(blk.mlp.up_w, uu[i].clone(), lr, wd, s_up);
                blk.mlp.down_w = apply_update::<B>(blk.mlp.down_w, ud[i].clone(), lr, wd, s_dn);
                blk
            })
            .collect();
        model.blocks = blocks;
        (model, prof)
    }

    /// Momentum + nesterov + batched NS5 for one bank, updating that bank's
    /// momentum buffer. Optionally times the NS5 (sync-gated).
    fn bank_update(
        &mut self,
        g: Tensor<NB, 3>,
        kind: BankKind,
        m: f64,
        device: &NB::Device,
        profile: bool,
        out_ms: &mut f64,
    ) -> Tensor<NB, 3> {
        let prev = match kind {
            BankKind::Qo => self.buf_qo.clone(),
            BankKind::Kv => self.buf_kv.clone(),
            BankKind::Up => self.buf_up.clone(),
            BankKind::Dn => self.buf_dn.clone(),
        };
        // buf = m*buf + g   (buf = g on first step)
        let buf = match prev {
            Some(b) => b * m + g.clone(),
            None => g.clone(),
        };
        // nesterov: go = g + m*buf
        let go = g + buf.clone() * m;
        match kind {
            BankKind::Qo => self.buf_qo = Some(buf),
            BankKind::Kv => self.buf_kv = Some(buf),
            BankKind::Up => self.buf_up = Some(buf),
            BankKind::Dn => self.buf_dn = Some(buf),
        }

        if profile {
            let _ = NB::sync(device);
            let t = Instant::now();
            let u = ns5_batched(go, self.ns_steps, self.eps);
            let _ = NB::sync(device);
            *out_ms = t.elapsed().as_secs_f64() * 1e3;
            u
        } else {
            ns5_batched(go, self.ns_steps, self.eps)
        }
    }
}

#[derive(Clone, Copy)]
enum BankKind {
    Qo,
    Kv,
    Up,
    Dn,
}

fn bank_scale(rows: usize, cols: usize) -> f64 {
    (cols as f64 / rows as f64).max(1.0).sqrt()
}

/// Remove grads for two id lists and stack them into one [n, r, c] tensor.
fn stack_grads<B, NB>(
    grads: &mut GradientsParams,
    first: &[burn::module::ParamId],
    second: &[burn::module::ParamId],
) -> Tensor<NB, 3>
where
    B: AutodiffBackend<InnerBackend = NB>,
    NB: Backend,
{
    let mut v: Vec<Tensor<NB, 2>> = Vec::with_capacity(first.len() + second.len());
    for id in first.iter().chain(second.iter()) {
        let g = grads
            .remove::<NB, 2>(*id)
            .expect("muon grad missing for banked param");
        v.push(g);
    }
    Tensor::stack::<3>(v, 0)
}

/// Slice `count` matrices [r,c] starting at `offset` out of a [n,r,c] stack.
fn split_stack<NB: Backend>(
    x: Tensor<NB, 3>,
    offset: usize,
    count: usize,
    r: usize,
    c: usize,
) -> Vec<Tensor<NB, 2>> {
    (0..count)
        .map(|i| x.clone().narrow(0, offset + i, 1).reshape([r, c]))
        .collect()
}

/// p ← p·(1 − lr·wd) − lr·scale·update, preserving ParamId and grad flag.
fn apply_update<B>(
    p: Param<Tensor<B, 2>>,
    update: Tensor<B::InnerBackend, 2>,
    lr: f64,
    wd: f64,
    scale: f64,
) -> Param<Tensor<B, 2>>
where
    B: AutodiffBackend,
{
    p.map(|t| {
        let req = t.is_require_grad();
        let mut inner = t.inner();
        if wd > 0.0 {
            inner = inner * (1.0 - lr * wd);
        }
        inner = inner - update * (lr * scale);
        let out = Tensor::from_inner(inner);
        if req {
            out.require_grad()
        } else {
            out
        }
    })
}

/// Batched Newton-Schulz quintic over a [n, r, c] stack. Each matrix is
/// normalized by its own Frobenius norm and iterated with batched matmuls.
pub fn ns5_batched<NB: Backend>(x3: Tensor<NB, 3>, steps: usize, eps: f64) -> Tensor<NB, 3> {
    let [_n, r, c] = x3.dims();
    let needs_t = r > c;
    let mut x = if needs_t { x3.swap_dims(1, 2) } else { x3 };

    // Per-matrix Frobenius norm → [n, 1, 1], broadcast-divide.
    let norm = x
        .clone()
        .powf_scalar(2.0)
        .sum_dim(2)
        .sum_dim(1)
        .sqrt()
        .clamp_min(eps);
    x = x / norm;

    for _ in 0..steps {
        let xt = x.clone().swap_dims(1, 2);
        let a = x.clone().matmul(xt); // [n, p, p]
        let b = a.clone() * NS_B + a.clone().matmul(a) * NS_C;
        x = x.clone() * NS_A + b.matmul(x);
    }
    if needs_t {
        x.swap_dims(1, 2)
    } else {
        x
    }
}

// ===========================================================================
// MuonWarm — per-parameter SimpleOptimizer (reference / tests)
// ===========================================================================

#[derive(Clone)]
pub struct MuonWarm {
    pub momentum: MomentumHandle,
    pub weight_decay: f64, // 0.04
    pub ns_steps: usize,   // 5
    pub eps: f64,          // 1e-7
}

/// Momentum buffer state (recordable for checkpointing).
#[derive(Record, Clone)]
pub struct MuonWarmState<B: Backend, const D: usize> {
    pub buf: Tensor<B, D>,
}

impl<B: Backend> SimpleOptimizer<B> for MuonWarm {
    type State<const D: usize> = MuonWarmState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: f64,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        assert!(D == 2, "Muon requires 2D parameters, got {D}D");
        let m = self.momentum.get();

        let buf = match state {
            Some(s) => s.buf * m + grad.clone(),
            None => grad.clone(),
        };
        let go = grad + buf.clone() * m;

        let shape = go.shape();
        let (rows, cols) = (shape[D - 2], shape[D - 1]);
        let go2 = go.reshape([rows, cols]);
        let u = newton_schulz5(go2, self.ns_steps, self.eps);
        let update: Tensor<B, D> = u.reshape(shape);

        let scale = (cols as f64 / rows as f64).max(1.0).sqrt();

        let tensor = if self.weight_decay > 0.0 {
            tensor * (1.0 - lr * self.weight_decay)
        } else {
            tensor
        };
        let tensor = tensor - update * (lr * scale);

        (tensor, Some(MuonWarmState { buf }))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        state.buf = state.buf.to_device(device);
        state
    }
}

impl MuonWarm {
    /// Wrap into an `Optimizer` usable with `optim.step(lr, model, grads)`.
    #[allow(dead_code)]
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(self) -> OptimizerAdaptor<MuonWarm, M, B> {
        OptimizerAdaptor::from(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type B = burn::backend::NdArray;

    #[test]
    fn momentum_handle_roundtrip() {
        let h = MomentumHandle::new(0.92);
        assert_eq!(h.get(), 0.92);
        h.set(0.955);
        assert_eq!(h.get(), 0.955);
    }

    #[test]
    fn batched_ns5_matches_per_matrix() {
        // The batched NS5 over a stack must equal per-matrix NS5 on each slice.
        let device = Default::default();
        let a = Tensor::<B, 2>::random(
            [8, 6],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let b = Tensor::<B, 2>::random(
            [8, 6],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let stacked = Tensor::stack::<3>(vec![a.clone(), b.clone()], 0);
        let batched = ns5_batched(stacked, 5, 1e-7);
        let ra = newton_schulz5(a, 5, 1e-7);
        let rb = newton_schulz5(b, 5, 1e-7);
        let ea = (batched.clone().narrow(0, 0, 1).reshape([8, 6]) - ra)
            .abs()
            .max()
            .into_scalar();
        let eb = (batched.narrow(0, 1, 1).reshape([8, 6]) - rb)
            .abs()
            .max()
            .into_scalar();
        assert!(ea < 1e-5, "qo slice mismatch {ea}");
        assert!(eb < 1e-5, "second slice mismatch {eb}");
    }

    #[test]
    fn step_moves_param_and_keeps_state() {
        let device = Default::default();
        let opt = MuonWarm {
            momentum: MomentumHandle::new(0.95),
            weight_decay: 0.0,
            ns_steps: 5,
            eps: 1e-7,
        };
        let p = Tensor::<B, 2>::ones([4, 4], &device);
        let g = Tensor::<B, 2>::from_floats(
            [
                [0.1, 0.0, 0.0, 0.0],
                [0.0, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 0.4],
            ],
            &device,
        );
        let (p2, st) = SimpleOptimizer::<B>::step(&opt, 0.01, p.clone(), g, None);
        assert!(st.is_some());
        let moved = (p2 - p).abs().max().into_scalar();
        assert!(moved > 0.0, "parameter did not move");
        assert!(moved < 0.05, "step unexpectedly large: {moved}");
    }

    #[test]
    fn weight_decay_shrinks() {
        let device = Default::default();
        let opt = MuonWarm {
            momentum: MomentumHandle::new(0.0),
            weight_decay: 0.04,
            ns_steps: 5,
            eps: 1e-7,
        };
        let p = Tensor::<B, 2>::ones([2, 2], &device);
        let g = Tensor::<B, 2>::zeros([2, 2], &device);
        let (p2, _) = SimpleOptimizer::<B>::step(&opt, 0.5, p, g, None);
        let v = p2.into_data();
        for x in v.as_slice::<f32>().unwrap() {
            assert!((*x - 0.98).abs() < 1e-4, "got {x}");
        }
    }
}
