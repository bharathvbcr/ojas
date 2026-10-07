//! K10 on the device: the chunked cross-entropy walk of [`crate::ce_rows`]
//! over K0's gather and `deliver` and K1's GEMMs, with this lane's two
//! kernels in between. [`ce_rows_mirror`](crate::ce_rows::ce_rows_mirror) is
//! its operation-for-operation host twin on the FFMA engine.
//!
//! [`CeWorkspace`] holds every buffer one call needs, sized by the plan and
//! bounded by [`crate::ce_rows::scratch_elements`]: the gathered rows, the row state
//! `(m, s, tlogit)`, the targets, and per chunk width (the full width, and
//! the tail's when the vocabulary is not a multiple of the chunk) a logits
//! block. K1's GEMMs take views, so the head chunk `W[v0..v0 + w]` is read in
//! place and `dW[v0..v0 + w]` written in place: no weight or `dW` chunk is
//! copied, as the host mirror already did.
//!
//! [`ce_rows`] **waits**: the per-row losses are formed in f64 on the host
//! from the downloaded `(m, s, tlogit)`, as tessl does, and a non-finite row
//! is refused **before** the gradient walk runs.

use cudarc::driver::PushKernelArg;

use crate::buffer::CudaBuffer;
use crate::ce_rows::{losses, CeOutput, CeRowsPlan, CE_LSE_UPDATE, CE_SOFTMAX_GRAD, MODULE};
use crate::error::CudaError;
use crate::gemm::{gemm, Accumulate, Bf16Engine, GemmSpec};
use crate::gemm_plan::{GemmLayout, GemmShape};
use crate::geometry::grid_1d;
use crate::host_ref::Operands;
use crate::k0::{ce_gather_rows, GatherSource};
use crate::runtime::{driver_error, CudaRuntime};
use crate::small_common::{block_grid, to_usize, ROW_THREADS};
use crate::small_common_cuda::{cfg, function};

/// The scratch for one chunk width.
struct ChunkScratch {
    width: u64,
    /// `[n, width]`: logits, then `dlogits` in place.
    logits: CudaBuffer<f32>,
}

/// Every device buffer a [`ce_rows`] call over `plan` uses.
pub struct CeWorkspace {
    plan: CeRowsPlan,
    hr: CudaBuffer<f32>,
    m: CudaBuffer<f32>,
    s: CudaBuffer<f32>,
    t: CudaBuffer<f32>,
    targets: CudaBuffer<u32>,
    sets: Vec<ChunkScratch>,
}

impl CeWorkspace {
    /// Allocate the workspace for `plan` against the runtime's budget.
    pub fn new(rt: &CudaRuntime, plan: CeRowsPlan) -> Result<Self, CudaError> {
        const OP: &str = "ce_rows workspace";
        let n = to_usize(plan.n, OP)?;
        let hid = to_usize(plan.hidden, OP)?;
        let mut sets = Vec::with_capacity(2);
        for w in plan.widths().into_iter().flatten() {
            let wu = to_usize(w, OP)?;
            sets.push(ChunkScratch {
                width: w,
                logits: rt.alloc_zeros(n * wu, "ce_rows logits")?,
            });
        }
        Ok(CeWorkspace {
            plan,
            hr: rt.alloc_zeros(n * hid, "ce_rows gathered rows")?,
            m: rt.alloc_zeros(n, "ce_rows m")?,
            s: rt.alloc_zeros(n, "ce_rows s")?,
            t: rt.alloc_zeros(n, "ce_rows target logit")?,
            targets: rt.alloc_zeros(n, "ce_rows targets")?,
            sets,
        })
    }

    /// The plan this workspace was sized for.
    pub fn plan(&self) -> &CeRowsPlan {
        &self.plan
    }

    /// f32 elements held: [`crate::ce_rows::scratch_elements`] of the plan.
    pub fn elements(&self) -> u64 {
        let set: usize = self.sets.iter().map(|c| c.logits.len()).sum();
        let total = set + self.hr.len() + self.m.len() + self.s.len() + self.t.len();
        u64::try_from(total).unwrap_or(u64::MAX)
    }
}

/// What the gradient walk writes: `dh [n, hidden]` (the supplied rows, in
/// their order; scattering them back is the caller's) and `dW [vocab,
/// hidden]`, both overwritten. `scale` is the upstream gradient.
pub struct CeGrads<'a> {
    /// Upstream gradient of the reduced loss (finite).
    pub scale: f32,
    /// `[n, hidden]`.
    pub dh: &'a mut CudaBuffer<f32>,
    /// `[vocab, hidden]`.
    pub dw: &'a mut CudaBuffer<f32>,
}

/// The cross-entropy of the supplied `(row, target)` pairs over `h` and the
/// tied head `w` (`[vocab, hidden]`), and with `grads` its gradients. GEMMs
/// run at `operands` on `engine` (`Bf16Engine::Ffma` is the bitwise tier of
/// the mirror). Waits once, for the losses.
pub fn ce_rows(
    rt: &CudaRuntime,
    ws: &mut CeWorkspace,
    h: GatherSource<'_>,
    w: &CudaBuffer<f32>,
    (rows, targets): (&[u32], &[u32]),
    (operands, engine): (Operands, Bf16Engine),
    grads: Option<CeGrads<'_>>,
) -> Result<CeOutput, CudaError> {
    const OP: &str = "cross_entropy_rows";
    let plan = ws.plan;
    let h_len = match &h {
        GatherSource::F32(b) => b.len(),
        GatherSource::Bf16(b) => b.len(),
    };
    plan.check_rows(rows, targets)?;
    plan.check_inputs(h_len, w.len())?;
    if let Some(g) = &grads {
        plan.check_grads(g.dh.len(), g.dw.len())?;
        plan.grad_factor(g.scale)?;
    }
    ce_gather_rows(
        rt,
        h,
        rows,
        plan.hidden,
        (plan.h.ld, plan.h.off),
        &mut ws.hr,
    )?;
    rt.write(&mut ws.targets, targets)?;

    // First walk: the online log-sum-exp.
    for (v0, wd) in plan.chunks() {
        load_logits(rt, ws, w, (v0, wd), operands, engine)?;
        lse_update(rt, ws, (v0, wd))?;
    }
    let m = rt.download(&ws.m)?;
    let s = rt.download(&ws.s)?;
    let t = rt.download(&ws.t)?;
    let out = losses(&plan, &m, &s, &t)?;
    let Some(g) = grads else {
        return Ok(out);
    };

    // Second walk: dlogits, dh (+)= dlogits W_c, dW_c = dlogits^T h.
    let factor = plan.grad_factor(g.scale)?;
    let hid = to_usize(plan.hidden, OP)?;
    let n = to_usize(plan.n, OP)?;
    for (v0, wd) in plan.chunks() {
        load_logits(rt, ws, w, (v0, wd), operands, engine)?;
        softmax_grad(rt, ws, (v0, wd), factor)?;
        let wdu = to_usize(wd, OP)?;
        // The head chunk W[v0..v0 + wd] and the dW rows it owns, in place.
        let (off, len) = (to_usize(v0, OP)? * hid, wdu * hid);
        let set = scratch_for(&mut ws.sets, wd)?;
        let acc = if v0 == 0 {
            Accumulate::Overwrite
        } else {
            Accumulate::Add
        };
        let dh_spec = GemmSpec {
            operands,
            layout: GemmLayout::Nn,
            shape: GemmShape::new(n, hid, wdu)?,
            acc,
        };
        gemm(
            rt,
            dh_spec,
            engine,
            set.logits.all(),
            w.view(off, len, OP)?,
            g.dh.all_mut(),
        )?;
        let dw_spec = GemmSpec {
            operands,
            layout: GemmLayout::Tn,
            shape: GemmShape::new(wdu, hid, n)?,
            acc: Accumulate::Overwrite,
        };
        gemm(
            rt,
            dw_spec,
            engine,
            set.logits.all(),
            ws.hr.all(),
            g.dw.view_mut(off, len, OP)?,
        )?;
    }
    Ok(out)
}

fn scratch_for(sets: &mut [ChunkScratch], width: u64) -> Result<&mut ChunkScratch, CudaError> {
    sets.iter_mut().find(|c| c.width == width).ok_or_else(|| {
        CudaError::invalid("cross_entropy_rows", format!("no scratch of width {width}"))
    })
}

/// `logits = hr @ W[v0..v0 + wd]^T`, the head chunk read in place.
fn load_logits(
    rt: &CudaRuntime,
    ws: &mut CeWorkspace,
    w: &CudaBuffer<f32>,
    (v0, wd): (u64, u64),
    operands: Operands,
    engine: Bf16Engine,
) -> Result<(), CudaError> {
    const OP: &str = "cross_entropy_rows";
    let hidden = ws.plan.hidden;
    let (n, hid) = (to_usize(ws.plan.n, OP)?, to_usize(hidden, OP)?);
    let wdu = to_usize(wd, OP)?;
    let wc = w.view(to_usize(v0, OP)? * hid, wdu * hid, OP)?;
    let set = scratch_for(&mut ws.sets, wd)?;
    let spec = GemmSpec {
        operands,
        layout: GemmLayout::Nt,
        shape: GemmShape::new(n, wdu, hid)?,
        acc: Accumulate::Overwrite,
    };
    gemm(rt, spec, engine, ws.hr.all(), wc, set.logits.all_mut())
}

fn lse_update(
    rt: &CudaRuntime,
    ws: &mut CeWorkspace,
    (v0, wd): (u64, u64),
) -> Result<(), CudaError> {
    let n = ws.plan.n;
    let l = block_grid(n, ROW_THREADS, CE_LSE_UPDATE)?;
    let f = function(rt, &MODULE, CE_LSE_UPDATE)?;
    let first: i32 = i32::from(v0 == 0);
    let set = scratch_for(&mut ws.sets, wd)?;
    let ld = wd;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(set.logits.slice())
        .arg(ws.m.slice_mut())
        .arg(ws.s.slice_mut())
        .arg(ws.t.slice_mut())
        .arg(ws.targets.slice())
        .arg(&n)
        .arg(&wd)
        .arg(&ld)
        .arg(&v0)
        .arg(&first);
    // SAFETY: (const float* logits, float* m, float* s, float* tlogit, const
    // unsigned int* targets, ull n_rows, ull width, ull ld, ull v0, int
    // first); CeWorkspace::new sized every buffer from the plan (its fields
    // are private), so logits is [n, wd] dense and m, s, tlogit, targets hold
    // n; targets were checked below vocab on the host (check_rows). One block of
    // ROW_THREADS per row; only thread 0 of block r writes m[r], s[r] and
    // tlogit[r]. The four are distinct allocations.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(CE_LSE_UPDATE, e))?;
    Ok(())
}

fn softmax_grad(
    rt: &CudaRuntime,
    ws: &mut CeWorkspace,
    (v0, wd): (u64, u64),
    factor: f32,
) -> Result<(), CudaError> {
    let n = ws.plan.n;
    let Some(l) = grid_1d(n * wd) else {
        return Err(CudaError::invalid(CE_SOFTMAX_GRAD, "no logits"));
    };
    let f = function(rt, &MODULE, CE_SOFTMAX_GRAD)?;
    let set = scratch_for(&mut ws.sets, wd)?;
    let ld = wd;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(set.logits.slice_mut())
        .arg(ws.m.slice())
        .arg(ws.s.slice())
        .arg(ws.targets.slice())
        .arg(&n)
        .arg(&wd)
        .arg(&ld)
        .arg(&v0)
        .arg(&factor);
    // SAFETY: (float* logits, const float* m, const float* s, const unsigned
    // int* targets, ull n_rows, ull width, ull ld, ull v0, float scale);
    // logits is [n, wd] dense and m, s, targets hold n (CeWorkspace::new
    // sized them from the plan). The grid strides over n * wd with one writer per logit, in
    // place; logits is a distinct allocation from m, s and targets.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(CE_SOFTMAX_GRAD, e))?;
    Ok(())
}
