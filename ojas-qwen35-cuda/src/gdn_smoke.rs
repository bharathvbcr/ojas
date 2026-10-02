//! K2(i) device checks: what rung 0's successor (`runga`) can run, what the
//! `#[ignore]` device tests share, and the report-only timing Fable's
//! decision 6 needs (`AUDIT/ojas-training-2026-10-01/fable-cuda-asks.md`).
//!
//! - [`gdn_published_checks`]: the module compiles; every smoke case
//!   ([`crate::gdn_host::smoke_cases`]) runs forward and backward on the
//!   device from fresh buffers and equals the host mirror **bit for bit** in
//!   every output, and a second run equals the first. The float64 judgment of
//!   those same cases (1e-4 of max, `gdn_host::published_bounds`) is made on
//!   the host mirror by `tests/device_gdn_published_mirror.rs`, so bit
//!   equality here carries it to the device. The device tests also judge the
//!   device directly against the float64 reference.
//! - [`gdn_published_timing`]: B = 1 launches in series against one batched
//!   launch, report-only, never a gate.
//!
//! Each case runs under [`guarded`], so a panic inside cudarc becomes a
//! `panicked` check and the rest still run.

use cudarc::driver::sys::CUevent_flags;

use crate::buffer::CudaBuffer;
use crate::check::{bitwise_check, diff_bits_f32, Check};
use crate::error::CudaError;
use crate::gdn::{
    gdn_published_backward, gdn_published_forward, GdnPublishedDeviceGrads,
    GdnPublishedDeviceInputs, GdnPublishedLayout, GdnPublishedWorkspace,
};
use crate::gdn_host::{gdn_published_mirror, smoke_cases, GdnPublishedCase, GdnPublishedOutputs};
use crate::gdn_kernels::GDN_PUBLISHED;
use crate::json::{Json, JsonObj};
use crate::kernels::STRICT_SM90;
use crate::runtime::{driver_error, CudaRuntime};
use crate::smoke::guarded;

/// Written to every output before a launch, so an element never written
/// shows (tessl `tests/gdn_train.rs:181-182`).
pub const SENTINEL: f32 = -7.25e27;

/// A case's operands on the device.
pub struct DeviceCase {
    /// The offsets.
    pub layout: GdnPublishedLayout,
    q: CudaBuffer<f32>,
    k: CudaBuffer<f32>,
    v: CudaBuffer<f32>,
    g: CudaBuffer<f32>,
    beta: CudaBuffer<f32>,
    s0: Option<CudaBuffer<f32>>,
    d_o: CudaBuffer<f32>,
    d_fin: Option<CudaBuffer<f32>>,
}

impl DeviceCase {
    /// Upload every operand of `case`.
    pub fn upload(rt: &CudaRuntime, case: &GdnPublishedCase) -> Result<Self, CudaError> {
        Ok(DeviceCase {
            layout: GdnPublishedLayout::upload(rt, &case.plan)?,
            q: rt.upload(&case.q, "gdn q")?,
            k: rt.upload(&case.k, "gdn k")?,
            v: rt.upload(&case.v, "gdn v")?,
            g: rt.upload(&case.g, "gdn g")?,
            beta: rt.upload(&case.beta, "gdn beta")?,
            s0: case
                .s0
                .as_deref()
                .map(|s| rt.upload(s, "gdn s0"))
                .transpose()?,
            d_o: rt.upload(&case.d_o, "gdn d_o")?,
            d_fin: case
                .d_fin
                .as_deref()
                .map(|s| rt.upload(s, "gdn d_fin"))
                .transpose()?,
        })
    }

    /// The forward's operands.
    pub fn inputs(&self) -> GdnPublishedDeviceInputs<'_> {
        GdnPublishedDeviceInputs {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            g: &self.g,
            beta: &self.beta,
            s0: self.s0.as_ref(),
        }
    }
}

fn sentinel(rt: &CudaRuntime, len: usize, label: &str) -> Result<CudaBuffer<f32>, CudaError> {
    rt.upload(&vec![SENTINEL; len], label)
}

/// The forward alone, outputs sentinel-prefilled: `(o, s_fin, ckpt)`, the
/// final state only when `keep_s_fin`.
#[allow(clippy::type_complexity)]
pub fn run_published_forward(
    rt: &CudaRuntime,
    dc: &DeviceCase,
    keep_s_fin: bool,
) -> Result<(Vec<f32>, Option<Vec<f32>>, Vec<f32>), CudaError> {
    let plan = dc.layout.plan();
    let mut o = sentinel(rt, plan.v_len(), "gdn o")?;
    let mut ckpt = sentinel(rt, plan.ckpt_len(), "gdn ckpt")?;
    let mut s_fin = if keep_s_fin {
        Some(sentinel(rt, plan.state_len(), "gdn s_fin")?)
    } else {
        None
    };
    gdn_published_forward(
        rt,
        &dc.layout,
        dc.inputs(),
        &mut o,
        s_fin.as_mut(),
        &mut ckpt,
    )?;
    let s_fin = s_fin.as_ref().map(|s| rt.download(s)).transpose()?;
    Ok((rt.download(&o)?, s_fin, rt.download(&ckpt)?))
}

/// Forward (final state kept) and backward from fresh, sentinel-prefilled
/// buffers; every output downloaded.
pub fn run_published_case(
    rt: &CudaRuntime,
    dc: &DeviceCase,
) -> Result<GdnPublishedOutputs, CudaError> {
    let plan = dc.layout.plan().clone();
    let mut o = sentinel(rt, plan.v_len(), "gdn o")?;
    let mut s_fin = sentinel(rt, plan.state_len(), "gdn s_fin")?;
    let mut ckpt = sentinel(rt, plan.ckpt_len(), "gdn ckpt")?;
    gdn_published_forward(
        rt,
        &dc.layout,
        dc.inputs(),
        &mut o,
        Some(&mut s_fin),
        &mut ckpt,
    )?;
    let mut ws = GdnPublishedWorkspace::new(rt, &plan)?;
    let mut dq = sentinel(rt, plan.qk_len(), "gdn dq")?;
    let mut dk = sentinel(rt, plan.qk_len(), "gdn dk")?;
    let mut dv = sentinel(rt, plan.v_len(), "gdn dv")?;
    let mut dg = sentinel(rt, plan.gate_len(), "gdn dg")?;
    let mut dbeta = sentinel(rt, plan.gate_len(), "gdn dbeta")?;
    let mut ds0 = match dc.s0 {
        Some(_) => Some(sentinel(rt, plan.state_len(), "gdn ds0")?),
        None => None,
    };
    gdn_published_backward(
        rt,
        &dc.layout,
        dc.inputs(),
        &ckpt,
        &dc.d_o,
        dc.d_fin.as_ref(),
        &mut ws,
        GdnPublishedDeviceGrads {
            dq: &mut dq,
            dk: &mut dk,
            dv: &mut dv,
            dg: &mut dg,
            dbeta: &mut dbeta,
            ds0: ds0.as_mut(),
        },
    )?;
    Ok(GdnPublishedOutputs {
        o: rt.download(&o)?,
        s_fin: rt.download(&s_fin)?,
        ckpt: rt.download(&ckpt)?,
        dq: rt.download(&dq)?,
        dk: rt.download(&dk)?,
        dv: rt.download(&dv)?,
        dg: rt.download(&dg)?,
        dbeta: rt.download(&dbeta)?,
        ds0: ds0.as_ref().map(|d| rt.download(d)).transpose()?,
    })
}

/// One bitwise check per output tensor, plus one for a missing or extra `ds0`.
pub fn bitwise_outputs(
    name: &str,
    got: &GdnPublishedOutputs,
    want: &GdnPublishedOutputs,
) -> Vec<Check> {
    let mut out = Vec::new();
    let (g, w) = (got.named(), want.named());
    if g.len() != w.len() {
        out.push(Check::fail(
            &format!("{name}.tensors"),
            format!(
                "{} tensors against {}: ds0 present on one side only",
                g.len(),
                w.len()
            ),
        ));
    }
    for ((tn, gt), (_, wt)) in g.iter().zip(&w) {
        out.push(bitwise_check(
            &format!("{name}.{tn}"),
            diff_bits_f32(gt, wt),
            wt.len(),
        ));
    }
    out
}

/// A report-safe name for a case.
pub fn slug(label: &str) -> String {
    label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

/// The module compiles; every smoke case equals the host mirror bit for bit,
/// and a second run equals the first.
pub fn gdn_published_checks(rt: &CudaRuntime) -> Vec<Check> {
    let mut out = Vec::new();
    let name = format!("nvrtc.{}", GDN_PUBLISHED.name);
    out.extend(guarded(&name, || {
        for entry in GDN_PUBLISHED.entries {
            if let Err(e) = rt.function(&GDN_PUBLISHED, &STRICT_SM90, entry) {
                return vec![Check::from_error(&name, &e)];
            }
        }
        vec![Check::pass(
            &name,
            format!(
                "{} entries compiled and loaded",
                GDN_PUBLISHED.entries.len()
            ),
        )]
    }));
    let cases = match smoke_cases() {
        Ok(c) => c,
        Err(e) => {
            out.push(Check::from_error("gdn_published.cases", &e));
            return out;
        }
    };
    for case in &cases {
        let name = format!("gdn.{}", slug(&case.label));
        out.extend(guarded(&name, || {
            let want = match gdn_published_mirror(case) {
                Ok(m) => m,
                Err(e) => return vec![Check::from_error(&format!("{name}.mirror"), &e)],
            };
            let run = || DeviceCase::upload(rt, case).and_then(|dc| run_published_case(rt, &dc));
            let first = match run() {
                Ok(v) => v,
                Err(e) => return vec![Check::from_error(&name, &e)],
            };
            let second = match run() {
                Ok(v) => v,
                Err(e) => return vec![Check::from_error(&format!("{name}.repeat"), &e)],
            };
            let mut checks = bitwise_outputs(&format!("{name}.vs_mirror"), &first, &want);
            checks.extend(bitwise_outputs(&format!("{name}.repeat"), &second, &first));
            checks
        }));
    }
    out
}

/// GDN layers in Qwen3.5-2B (`cuda-backend-scoping.md` §3.2: "x 18 GDN
/// layers").
pub const QWEN35_2B_GDN_LAYERS: usize = 18;

/// PyTorch's whole training step on the GH200 at shape B (4 x <= 8,441
/// tokens), seconds: 1.20-1.33 s with flash and no mask, 1.93 s on default
/// kernels; Fable's trigger quotes 1.2-1.9 s
/// (`AUDIT/det-attention-backward-scoping-2026-10-01.md:14,44-46`,
/// `fable-cuda-asks.md` decision 6). Reported beside the projection; never
/// compared in code.
pub const PYTORCH_STEP_S: (f64, f64) = (1.2, 1.9);

/// Report-only timing of one GDN layer at one batch shape.
#[derive(Clone, Debug, PartialEq)]
pub struct GdnPublishedTiming {
    /// Sequence lengths.
    pub lens: Vec<usize>,
    /// Value heads.
    pub heads: usize,
    /// Value head dim.
    pub v_dim: usize,
    /// Timed repetitions (min taken).
    pub reps: usize,
    /// One batched forward launch, min over reps, ms.
    pub batched_fwd_ms: f64,
    /// One batched backward (scan + finish), min over reps, ms.
    pub batched_bwd_ms: f64,
    /// The sequences' `B = 1` forwards back to back, sum of per-sequence mins, ms.
    pub serial_fwd_ms: f64,
    /// The sequences' `B = 1` backwards, sum of per-sequence mins, ms.
    pub serial_bwd_ms: f64,
    /// The batched o, dq, dg equal the `B = 1` results bit for bit.
    pub batched_equals_serial_bitwise: bool,
}

impl GdnPublishedTiming {
    /// Seconds of GDN per training step over Qwen3.5-2B's 18 GDN layers, each
    /// running the forward twice (the step rebuilds a layer's forward before
    /// its backward, `tessl/src/qwen35_train.rs:28-33`) and the backward once:
    /// `(serial, batched)`.
    pub fn projected_step_s(&self) -> (f64, f64) {
        let l = QWEN35_2B_GDN_LAYERS as f64;
        (
            l * (2.0 * self.serial_fwd_ms + self.serial_bwd_ms) / 1e3,
            l * (2.0 * self.batched_fwd_ms + self.batched_bwd_ms) / 1e3,
        )
    }

    /// The report object.
    pub fn to_json(&self) -> Json {
        let (serial, batched) = self.projected_step_s();
        JsonObj::new()
            .with("report_only", true)
            .with("lens", Json::Arr(self.lens.iter().map(|&t| Json::from(t)).collect()))
            .with("heads", self.heads)
            .with("v_dim", self.v_dim)
            .with("reps", self.reps)
            .with("batched_fwd_ms", self.batched_fwd_ms)
            .with("batched_bwd_ms", self.batched_bwd_ms)
            .with("serial_fwd_ms", self.serial_fwd_ms)
            .with("serial_bwd_ms", self.serial_bwd_ms)
            .with("batched_equals_serial_bitwise", self.batched_equals_serial_bitwise)
            .with("gdn_layers", QWEN35_2B_GDN_LAYERS)
            .with("projected_step_gdn_s_serial", serial)
            .with("projected_step_gdn_s_batched", batched)
            .with("pytorch_whole_step_s_low", PYTORCH_STEP_S.0)
            .with("pytorch_whole_step_s_high", PYTORCH_STEP_S.1)
            .with(
                "trigger",
                "(ii) is built iff K2(i) alone exceeds PyTorch's whole step after batching (fable-cuda-asks.md decision 6); the reading is the lead's",
            )
            .into()
    }
}

/// Everything one timed configuration holds on the device.
struct Timed {
    dc: DeviceCase,
    ws: GdnPublishedWorkspace,
    o: CudaBuffer<f32>,
    ckpt: CudaBuffer<f32>,
    dq: CudaBuffer<f32>,
    dk: CudaBuffer<f32>,
    dv: CudaBuffer<f32>,
    dg: CudaBuffer<f32>,
    dbeta: CudaBuffer<f32>,
}

impl Timed {
    fn new(rt: &CudaRuntime, case: &GdnPublishedCase) -> Result<Self, CudaError> {
        let p = &case.plan;
        Ok(Timed {
            dc: DeviceCase::upload(rt, case)?,
            ws: GdnPublishedWorkspace::new(rt, p)?,
            o: rt.alloc_zeros(p.v_len(), "timing o")?,
            ckpt: rt.alloc_zeros(p.ckpt_len(), "timing ckpt")?,
            dq: rt.alloc_zeros(p.qk_len(), "timing dq")?,
            dk: rt.alloc_zeros(p.qk_len(), "timing dk")?,
            dv: rt.alloc_zeros(p.v_len(), "timing dv")?,
            dg: rt.alloc_zeros(p.gate_len(), "timing dg")?,
            dbeta: rt.alloc_zeros(p.gate_len(), "timing dbeta")?,
        })
    }

    fn fwd(&mut self, rt: &CudaRuntime) -> Result<(), CudaError> {
        gdn_published_forward(
            rt,
            &self.dc.layout,
            self.dc.inputs(),
            &mut self.o,
            None,
            &mut self.ckpt,
        )
    }

    fn bwd(&mut self, rt: &CudaRuntime) -> Result<(), CudaError> {
        gdn_published_backward(
            rt,
            &self.dc.layout,
            self.dc.inputs(),
            &self.ckpt,
            &self.dc.d_o,
            None,
            &mut self.ws,
            GdnPublishedDeviceGrads {
                dq: &mut self.dq,
                dk: &mut self.dk,
                dv: &mut self.dv,
                dg: &mut self.dg,
                dbeta: &mut self.dbeta,
                ds0: None,
            },
        )
    }
}

/// Milliseconds the queued work of `f` takes on the stream, by CUDA events
/// read only after the runtime's bounded sync.
fn timed_ms(
    rt: &CudaRuntime,
    f: &mut dyn FnMut() -> Result<(), CudaError>,
) -> Result<f64, CudaError> {
    let ev = || {
        rt.context()
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| driver_error("timing: cuEventCreate", e))
    };
    let (start, end) = (ev()?, ev()?);
    start
        .record(rt.stream())
        .map_err(|e| driver_error("timing: cuEventRecord", e))?;
    f()?;
    end.record(rt.stream())
        .map_err(|e| driver_error("timing: cuEventRecord", e))?;
    rt.sync("gdn_published timing")?;
    let ms = start
        .elapsed_ms(&end)
        .map_err(|e| driver_error("timing: cuEventElapsedTime", e))?;
    Ok(f64::from(ms))
}

/// Time one GDN layer's forward and backward on `lens` (tessl's operand
/// distributions, no initial state): one batched launch against the
/// sequences' `B = 1` launches, interleaved, `reps` times each after a warm-up
/// (which also compiles), min taken. The batched and `B = 1` outputs are
/// compared bit for bit on the way. Report-only: nothing here is a gate.
pub fn gdn_published_timing(
    rt: &CudaRuntime,
    lens: &[usize],
    heads: usize,
    v_dim: usize,
    reps: usize,
) -> Result<GdnPublishedTiming, CudaError> {
    if reps == 0 || reps > 20 {
        return Err(CudaError::invalid(
            "gdn_published_timing",
            format!("reps must be in 1..=20, got {reps}"),
        ));
    }
    let case = GdnPublishedCase::varlen(lens, heads, v_dim, 4242, false, false)?;
    let plan = case.plan.clone();
    let mut batched = Timed::new(rt, &case)?;
    let mut serial = (0..plan.batch())
        .map(|b| case.sequence(b).and_then(|c| Timed::new(rt, &c)))
        .collect::<Result<Vec<_>, _>>()?;

    // Warm-up: compiles the module and touches every buffer.
    batched.fwd(rt)?;
    batched.bwd(rt)?;
    for s in serial.iter_mut() {
        s.fwd(rt)?;
        s.bwd(rt)?;
    }
    rt.sync("gdn_published timing warm-up")?;

    let mut best = [f64::INFINITY; 2];
    let mut best_serial = vec![[f64::INFINITY; 2]; serial.len()];
    for _ in 0..reps {
        best[0] = best[0].min(timed_ms(rt, &mut || batched.fwd(rt))?);
        for (s, b) in serial.iter_mut().zip(best_serial.iter_mut()) {
            b[0] = b[0].min(timed_ms(rt, &mut || s.fwd(rt))?);
        }
        best[1] = best[1].min(timed_ms(rt, &mut || batched.bwd(rt))?);
        for (s, b) in serial.iter_mut().zip(best_serial.iter_mut()) {
            b[1] = b[1].min(timed_ms(rt, &mut || s.bwd(rt))?);
        }
    }

    let mut equal = true;
    fn pick<'a>(t: &'a Timed, tensor: &str) -> &'a CudaBuffer<f32> {
        match tensor {
            "o" => &t.o,
            "dq" => &t.dq,
            _ => &t.dg,
        }
    }
    for (tensor, width) in [("o", v_dim), ("dq", 128usize), ("dg", 1usize)] {
        let all = rt.download(pick(&batched, tensor))?;
        let parts = plan.split_tokens(&all, width)?;
        for (b, s) in serial.iter().enumerate() {
            let one = rt.download(pick(s, tensor))?;
            equal &= diff_bits_f32(&one, parts[b]).mismatches == 0;
        }
    }
    Ok(GdnPublishedTiming {
        lens: lens.to_vec(),
        heads,
        v_dim,
        reps,
        batched_fwd_ms: best[0],
        batched_bwd_ms: best[1],
        serial_fwd_ms: best_serial.iter().map(|b| b[0]).sum(),
        serial_bwd_ms: best_serial.iter().map(|b| b[1]).sum(),
        batched_equals_serial_bitwise: equal,
    })
}
