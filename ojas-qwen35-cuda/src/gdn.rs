//! K2(i) on the device: the published-rule GDN scan, forward and backward,
//! launched through a validated [`GdnPublishedPlan`].
//!
//! The device counterpart of tessl's `gdn_train_forward` / `gdn_train_backward`
//! (`tessl/src/gdn_train.rs:242-414`), for `B >= 1` variable-length sequences
//! in one launch (layout: [`crate::gdn_plan`]). Every call checks every
//! buffer's length against the plan before it queues anything; the backward
//! refuses a workspace or layout built for another plan and an `s0` / `ds0`
//! that do not pair (tessl `:324-342`). Outputs and inputs cannot alias:
//! outputs are `&mut` borrows of whole buffers. Nothing here waits; results
//! are read with [`CudaRuntime::download`].
//!
//! Absent optional buffers (`s0`, `s_fin`, `d_fin`, `ds0`) are passed as null
//! device pointers together with a flags word; the kernels read or write a
//! pointer only when its flag is set (`crate::gdn_kernels`).

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::buffer::CudaBuffer;
use crate::error::CudaError;
use crate::gdn_kernels::{
    FLAG_DFIN, FLAG_DS0, FLAG_S0, FLAG_SFIN, GDN_PUBLISHED, GDN_PUBLISHED_BWD,
    GDN_PUBLISHED_BWD_FINISH, GDN_PUBLISHED_FWD,
};
use crate::gdn_plan::GdnPublishedPlan;
use crate::geometry::Launch;
use crate::kernels::STRICT_SM90;
use crate::runtime::{driver_error, CudaRuntime};

fn cfg(l: Launch) -> LaunchConfig {
    LaunchConfig {
        grid_dim: l.grid,
        block_dim: l.block,
        // The kernels' shared memory is static (at most 5.2 KiB).
        shared_mem_bytes: 0,
    }
}

fn want(op: &str, name: &str, got: usize, need: usize) -> Result<(), CudaError> {
    if got == need {
        Ok(())
    } else {
        Err(CudaError::invalid(
            op,
            format!("{name} has {got} elements, the plan needs {need}"),
        ))
    }
}

fn u32_of(n: usize, what: &str) -> Result<u32, CudaError> {
    u32::try_from(n)
        .map_err(|_| CudaError::invalid("gdn_published", format!("{what} {n} does not fit u32")))
}

/// A plan's sequence offsets on the device. Built once per batch shape and
/// shared by the forward and the backward.
pub struct GdnPublishedLayout {
    plan: GdnPublishedPlan,
    tok_off: CudaBuffer<u32>,
    ck_off: CudaBuffer<u32>,
}

impl GdnPublishedLayout {
    /// Upload `plan`'s token and checkpoint offsets.
    pub fn upload(rt: &CudaRuntime, plan: &GdnPublishedPlan) -> Result<Self, CudaError> {
        Ok(GdnPublishedLayout {
            tok_off: rt.upload(plan.tok_off(), "gdn_published tok_off")?,
            ck_off: rt.upload(plan.ck_off(), "gdn_published ck_off")?,
            plan: plan.clone(),
        })
    }

    /// The plan the offsets belong to.
    pub fn plan(&self) -> &GdnPublishedPlan {
        &self.plan
    }
}

/// The backward's scratch and per-slice partials (tessl's
/// `GdnTrainWorkspace`, `src/gdn_train.rs:100-145`), for one plan.
pub struct GdnPublishedWorkspace {
    plan: GdnPublishedPlan,
    scratch: CudaBuffer<f32>,
    dq_part: CudaBuffer<f32>,
    dk_part: CudaBuffer<f32>,
    dg_part: CudaBuffer<f32>,
    dbeta_part: CudaBuffer<f32>,
}

impl GdnPublishedWorkspace {
    /// Allocate for `plan` ([`GdnPublishedPlan::workspace_bytes`] bytes).
    pub fn new(rt: &CudaRuntime, plan: &GdnPublishedPlan) -> Result<Self, CudaError> {
        Ok(GdnPublishedWorkspace {
            scratch: rt.alloc_zeros(plan.scratch_len(), "gdn_published scratch")?,
            dq_part: rt.alloc_zeros(plan.qk_part_len(), "gdn_published dq_part")?,
            dk_part: rt.alloc_zeros(plan.qk_part_len(), "gdn_published dk_part")?,
            dg_part: rt.alloc_zeros(plan.gate_part_len(), "gdn_published dg_part")?,
            dbeta_part: rt.alloc_zeros(plan.gate_part_len(), "gdn_published dbeta_part")?,
            plan: plan.clone(),
        })
    }

    /// The plan it was sized for.
    pub fn plan(&self) -> &GdnPublishedPlan {
        &self.plan
    }
}

/// The operands both directions read, in the plan's layout.
#[derive(Clone, Copy)]
pub struct GdnPublishedDeviceInputs<'a> {
    /// `[N, H, 128]`.
    pub q: &'a CudaBuffer<f32>,
    /// `[N, H, 128]`.
    pub k: &'a CudaBuffer<f32>,
    /// `[N, H, Dv]`.
    pub v: &'a CudaBuffer<f32>,
    /// The log decay `[N, H]`.
    pub g: &'a CudaBuffer<f32>,
    /// `[N, H]`.
    pub beta: &'a CudaBuffer<f32>,
    /// The initial state `[B, H, 128, Dv]`; zeros when `None`.
    pub s0: Option<&'a CudaBuffer<f32>>,
}

impl GdnPublishedDeviceInputs<'_> {
    fn check(&self, plan: &GdnPublishedPlan, op: &str) -> Result<(), CudaError> {
        want(op, "q", self.q.len(), plan.qk_len())?;
        want(op, "k", self.k.len(), plan.qk_len())?;
        want(op, "v", self.v.len(), plan.v_len())?;
        want(op, "g", self.g.len(), plan.gate_len())?;
        want(op, "beta", self.beta.len(), plan.gate_len())?;
        if let Some(s0) = self.s0 {
            want(op, "s0", s0.len(), plan.state_len())?;
        }
        Ok(())
    }
}

/// `o` `[N, H, Dv]`, the final state into `s_fin` when given, and the
/// checkpoints the backward needs into `ckpt` ([`GdnPublishedPlan::ckpt_len`]).
pub fn gdn_published_forward(
    rt: &CudaRuntime,
    layout: &GdnPublishedLayout,
    x: GdnPublishedDeviceInputs<'_>,
    o: &mut CudaBuffer<f32>,
    s_fin: Option<&mut CudaBuffer<f32>>,
    ckpt: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    const OP: &str = "gdn_published_forward";
    let plan = layout.plan();
    x.check(plan, OP)?;
    want(OP, "o", o.len(), plan.v_len())?;
    want(OP, "ckpt", ckpt.len(), plan.ckpt_len())?;
    if let Some(sf) = &s_fin {
        want(OP, "s_fin", sf.len(), plan.state_len())?;
    }
    let flags =
        if x.s0.is_some() { FLAG_S0 } else { 0 } | if s_fin.is_some() { FLAG_SFIN } else { 0 };
    let (nseq, heads, v_dim) = (
        u32_of(plan.batch(), "sequences")?,
        u32_of(plan.heads(), "heads")?,
        u32_of(plan.v_dim(), "v_dim")?,
    );
    let null: u64 = 0;
    let f = rt.function(&GDN_PUBLISHED, &STRICT_SM90, GDN_PUBLISHED_FWD)?;
    let mut s_fin = s_fin;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(x.q.slice())
        .arg(x.k.slice())
        .arg(x.v.slice())
        .arg(x.g.slice())
        .arg(x.beta.slice());
    match x.s0 {
        Some(s0) => b.arg(s0.slice()),
        None => b.arg(&null),
    };
    b.arg(o.slice_mut());
    match s_fin.as_mut() {
        Some(sf) => b.arg(sf.slice_mut()),
        None => b.arg(&null),
    };
    b.arg(ckpt.slice_mut())
        .arg(layout.tok_off.slice())
        .arg(layout.ck_off.slice())
        .arg(&nseq)
        .arg(&heads)
        .arg(&v_dim)
        .arg(&flags);
    // SAFETY: the arguments match `qd_gdn_published_fwd`'s 15 parameters in
    // order and type (11 device pointers, 4 x unsigned int). Every buffer's
    // length was checked against the plan above, and the plan's offsets
    // (uploaded with it) bound every index the kernel forms: tokens below
    // tok_off[B] = N, checkpoints below ck_off[B], states below B*H*128*Dv.
    // s0 / s_fin are null exactly when FLAG_S0 / FLAG_SFIN is clear, and the
    // kernel touches them only under their flag. The grid is the plan's
    // (Dv/16, B*H) with 128 threads, which the kernel's indexing assumes.
    unsafe { b.launch(cfg(plan.scan_launch())) }.map_err(|e| driver_error(GDN_PUBLISHED_FWD, e))?;
    Ok(())
}

/// Gradients of every input, written in full.
pub struct GdnPublishedDeviceGrads<'a> {
    /// `[N, H, 128]`.
    pub dq: &'a mut CudaBuffer<f32>,
    /// `[N, H, 128]`.
    pub dk: &'a mut CudaBuffer<f32>,
    /// `[N, H, Dv]`.
    pub dv: &'a mut CudaBuffer<f32>,
    /// `[N, H]`.
    pub dg: &'a mut CudaBuffer<f32>,
    /// `[N, H]`.
    pub dbeta: &'a mut CudaBuffer<f32>,
    /// `[B, H, 128, Dv]`; required exactly when the forward had `s0`.
    pub ds0: Option<&'a mut CudaBuffer<f32>>,
}

/// The backward of [`gdn_published_forward`] from `d_o` (and `d_fin`, the
/// final state's gradient, when given), on the same inputs and the
/// checkpoints the forward saved.
#[allow(clippy::too_many_arguments)]
pub fn gdn_published_backward(
    rt: &CudaRuntime,
    layout: &GdnPublishedLayout,
    x: GdnPublishedDeviceInputs<'_>,
    ckpt: &CudaBuffer<f32>,
    d_o: &CudaBuffer<f32>,
    d_fin: Option<&CudaBuffer<f32>>,
    ws: &mut GdnPublishedWorkspace,
    grads: GdnPublishedDeviceGrads<'_>,
) -> Result<(), CudaError> {
    const OP: &str = "gdn_published_backward";
    let plan = layout.plan();
    if ws.plan() != plan {
        return Err(CudaError::invalid(
            OP,
            format!(
                "the workspace is for lens {:?} H={} Dv={}, not lens {:?} H={} Dv={}",
                ws.plan().lens(),
                ws.plan().heads(),
                ws.plan().v_dim(),
                plan.lens(),
                plan.heads(),
                plan.v_dim()
            ),
        ));
    }
    x.check(plan, OP)?;
    want(OP, "ckpt", ckpt.len(), plan.ckpt_len())?;
    want(OP, "d_o", d_o.len(), plan.v_len())?;
    if let Some(df) = d_fin {
        want(OP, "d_fin", df.len(), plan.state_len())?;
    }
    let GdnPublishedDeviceGrads {
        dq,
        dk,
        dv,
        dg,
        dbeta,
        ds0,
    } = grads;
    want(OP, "dq", dq.len(), plan.qk_len())?;
    want(OP, "dk", dk.len(), plan.qk_len())?;
    want(OP, "dv", dv.len(), plan.v_len())?;
    want(OP, "dg", dg.len(), plan.gate_len())?;
    want(OP, "dbeta", dbeta.len(), plan.gate_len())?;
    let mut ds0 = ds0;
    match (x.s0, ds0.as_ref()) {
        (Some(_), Some(d)) => want(OP, "ds0", d.len(), plan.state_len())?,
        (None, None) => {}
        (Some(_), None) => {
            return Err(CudaError::invalid(
                OP,
                "the forward had an initial state; ds0 is required",
            ))
        }
        (None, Some(_)) => {
            return Err(CudaError::invalid(
                OP,
                "ds0 given but the forward had no initial state",
            ))
        }
    }
    let flags =
        if x.s0.is_some() { FLAG_DS0 } else { 0 } | if d_fin.is_some() { FLAG_DFIN } else { 0 };
    let (nseq, heads, v_dim, ns) = (
        u32_of(plan.batch(), "sequences")?,
        u32_of(plan.heads(), "heads")?,
        u32_of(plan.v_dim(), "v_dim")?,
        u32_of(plan.slices(), "slices")?,
    );
    let rows = plan.rows() as u64;
    let null: u64 = 0;

    let f = rt.function(&GDN_PUBLISHED, &STRICT_SM90, GDN_PUBLISHED_BWD)?;
    {
        let mut b = rt.stream().launch_builder(&f);
        b.arg(x.q.slice())
            .arg(x.k.slice())
            .arg(x.v.slice())
            .arg(x.g.slice())
            .arg(x.beta.slice())
            .arg(d_o.slice());
        match d_fin {
            Some(df) => b.arg(df.slice()),
            None => b.arg(&null),
        };
        b.arg(ckpt.slice())
            .arg(ws.scratch.slice_mut())
            .arg(dv.slice_mut())
            .arg(ws.dq_part.slice_mut())
            .arg(ws.dk_part.slice_mut())
            .arg(ws.dg_part.slice_mut())
            .arg(ws.dbeta_part.slice_mut());
        match ds0.as_mut() {
            Some(d) => b.arg(d.slice_mut()),
            None => b.arg(&null),
        };
        b.arg(layout.tok_off.slice())
            .arg(layout.ck_off.slice())
            .arg(&nseq)
            .arg(&heads)
            .arg(&v_dim)
            .arg(&flags)
            .arg(&rows);
        // SAFETY: the arguments match `qd_gdn_published_bwd`'s 22 parameters
        // in order and type (17 device pointers, 4 x unsigned int, one
        // unsigned long long). Lengths were checked against the plan above and
        // the workspace was sized for this plan (refused otherwise), so the
        // scratch holds B*H*(Dv/16) chunks of 64*128*16 and the partials
        // Dv/16 * N*H rows. d_fin / ds0 are null exactly when FLAG_DFIN /
        // FLAG_DS0 is clear and are touched only under their flag. Grid as the
        // forward's.
        unsafe { b.launch(cfg(plan.scan_launch())) }
            .map_err(|e| driver_error(GDN_PUBLISHED_BWD, e))?;
    }

    let fin = rt.function(&GDN_PUBLISHED, &STRICT_SM90, GDN_PUBLISHED_BWD_FINISH)?;
    let mut b = rt.stream().launch_builder(&fin);
    b.arg(x.q.slice())
        .arg(x.k.slice())
        .arg(ws.dq_part.slice())
        .arg(ws.dk_part.slice())
        .arg(ws.dg_part.slice())
        .arg(ws.dbeta_part.slice())
        .arg(dq.slice_mut())
        .arg(dk.slice_mut())
        .arg(dg.slice_mut())
        .arg(dbeta.slice_mut())
        .arg(&rows)
        .arg(&ns);
    // SAFETY: the arguments match `qd_gdn_published_bwd_finish`'s 12
    // parameters (10 device pointers, unsigned long long rows, unsigned int
    // ns). q, k, dq, dk hold rows*128 elements and dg, dbeta rows; each
    // partial holds ns*rows(*128), as the plan sized them. The block-stride
    // loop visits each row in [0, rows) once with 128 threads per block.
    unsafe { b.launch(cfg(plan.finish_launch())) }
        .map_err(|e| driver_error(GDN_PUBLISHED_BWD_FINISH, e))?;
    Ok(())
}
