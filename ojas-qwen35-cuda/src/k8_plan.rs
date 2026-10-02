//! K8 plans and bitwise host references: SwiGLU forward (f32 or bf16 out),
//! its backward, and the exact-f32 residual add (`cuda-backend-scoping.md`
//! §3, the K8 row: elementwise, exact f32, host bitwise).
//!
//! The semantics are tessl's, operand for operand:
//! - `out = silu(gate) * up` (`tessl/kernels/qwen35_mlp.metal:23-47`); the
//!   bf16 variant rounds that product once, to nearest even;
//! - `dgate = dy * up * silu'(gate)`, left to right, and `dup = dy *
//!   silu(gate)` (`tessl/kernels/qwen35_bwd.metal:232-259`);
//! - `resid += y` (`tessl/kernels/qwen35_mlp.metal:56-70`).
//!
//! Every operand is a column window ([`ColWindow`]) of a row-major matrix, so
//! a fused `[gate | up]` projection feeds the forward in place and its
//! gradient takes `dgate` and `dup` as two windows of one buffer. SiLU and its
//! derivative are [`crate::k8_act`]'s, the crate's one copy.
//!
//! **NaN.** tessl writes whatever NaN the hardware makes. Here every output
//! goes through `canon_nan`, on the device and in these references, so a NaN
//! compares bit for bit whichever operation produced it (an `inf * 0` makes a
//! different NaN on the Mac than on sm_90). Finite results are untouched.

use crate::bf16::f32_to_bf16_bits;
use crate::error::CudaError;
use crate::k0_plan::ColWindow;
use crate::k8_act::{canon_nan_f32, silu_f32, silu_grad_f32};

/// `(ld, off, buffer length)` of one operand.
pub type WindowSpec = (u64, u64, usize);

fn total(op: &str, rows: u64, width: u64) -> Result<u64, CudaError> {
    rows.checked_mul(width)
        .ok_or_else(|| CudaError::invalid(op, format!("{rows} x {width} overflows u64")))
}

fn check_len(op: &str, name: &str, w: &ColWindow, len: usize) -> Result<(), CudaError> {
    if w.end > u64::try_from(len).unwrap_or(u64::MAX) {
        return Err(CudaError::invalid(
            op,
            format!(
                "{name}: a {len}-element buffer for a window ending at {}",
                w.end
            ),
        ));
    }
    Ok(())
}

/// One element of the forward.
pub fn swiglu_elem(g: f32, u: f32) -> f32 {
    canon_nan_f32(silu_f32(g) * u)
}

/// One element of the backward: `(dgate, dup)`.
pub fn swiglu_bwd_elem(g: f32, u: f32, d: f32) -> (f32, f32) {
    (
        canon_nan_f32((d * u) * silu_grad_f32(g)),
        canon_nan_f32(d * silu_f32(g)),
    )
}

/// One element of the residual add.
pub fn residual_add_elem(resid: f32, y: f32) -> f32 {
    canon_nan_f32(resid + y)
}

/// `out[r, :] = silu(gate[r, :]) * up[r, :]` over `rows x width` windows.
/// `gate` and `up` may be windows of one buffer; `out` is another buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwigluPlan {
    pub rows: u64,
    pub width: u64,
    pub gate: ColWindow,
    pub up: ColWindow,
    pub out: ColWindow,
    /// `rows * width`.
    pub total: u64,
}

impl SwigluPlan {
    pub fn new(
        rows: u64,
        width: u64,
        gate: WindowSpec,
        up: WindowSpec,
        out: WindowSpec,
    ) -> Result<Self, CudaError> {
        const OP: &str = "swiglu";
        Ok(SwigluPlan {
            rows,
            width,
            gate: ColWindow::new(OP, "gate", rows, width, gate)?,
            up: ColWindow::new(OP, "up", rows, width, up)?,
            out: ColWindow::new(OP, "out", rows, width, out)?,
            total: total(OP, rows, width)?,
        })
    }

    fn each(&self, mut f: impl FnMut(usize, usize, usize)) {
        for r in 0..self.rows {
            for c in 0..self.width {
                f(
                    self.gate.at(r, c) as usize,
                    self.up.at(r, c) as usize,
                    self.out.at(r, c) as usize,
                );
            }
        }
    }

    /// The f32 forward on the host, bit for bit the device's.
    pub fn host_f32(&self, gate: &[f32], up: &[f32], out: &mut [f32]) -> Result<(), CudaError> {
        self.check(gate.len(), up.len(), out.len())?;
        self.each(|g, u, o| out[o] = swiglu_elem(gate[g], up[u]));
        Ok(())
    }

    /// The bf16-out forward on the host: the f32 product rounded once.
    pub fn host_bf16(&self, gate: &[f32], up: &[f32], out: &mut [u16]) -> Result<(), CudaError> {
        self.check(gate.len(), up.len(), out.len())?;
        self.each(|g, u, o| out[o] = f32_to_bf16_bits(swiglu_elem(gate[g], up[u])));
        Ok(())
    }

    /// Re-check the plan against buffer lengths.
    pub fn check(&self, gate: usize, up: usize, out: usize) -> Result<(), CudaError> {
        check_len("swiglu", "gate", &self.gate, gate)?;
        check_len("swiglu", "up", &self.up, up)?;
        check_len("swiglu", "out", &self.out, out)
    }
}

/// Where the backward writes `dgate` and `dup`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BwdOut {
    /// Two buffers.
    Separate,
    /// Two disjoint windows of one buffer (the fused projection's gradient).
    Shared,
}

/// The SwiGLU backward over `rows x width` windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwigluBwdPlan {
    pub rows: u64,
    pub width: u64,
    pub gate: ColWindow,
    pub up: ColWindow,
    pub dy: ColWindow,
    pub dgate: ColWindow,
    pub dup: ColWindow,
    pub out: BwdOut,
    pub total: u64,
}

impl SwigluBwdPlan {
    /// With [`BwdOut::Shared`], `dgate` and `dup` describe windows of one
    /// buffer and must be disjoint: same row stride, column ranges apart
    /// (tessl's `no_overlap`, `qwen35_bwd.rs:271-275`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rows: u64,
        width: u64,
        gate: WindowSpec,
        up: WindowSpec,
        dy: WindowSpec,
        dgate: WindowSpec,
        dup: WindowSpec,
        out: BwdOut,
    ) -> Result<Self, CudaError> {
        const OP: &str = "swiglu_bwd";
        let plan = SwigluBwdPlan {
            rows,
            width,
            gate: ColWindow::new(OP, "gate", rows, width, gate)?,
            up: ColWindow::new(OP, "up", rows, width, up)?,
            dy: ColWindow::new(OP, "dy", rows, width, dy)?,
            dgate: ColWindow::new(OP, "dgate", rows, width, dgate)?,
            dup: ColWindow::new(OP, "dup", rows, width, dup)?,
            out,
            total: total(OP, rows, width)?,
        };
        if out == BwdOut::Shared {
            if dgate.2 != dup.2 {
                return Err(CudaError::invalid(
                    OP,
                    format!(
                        "shared dgate/dup buffer given two lengths, {} and {}",
                        dgate.2, dup.2
                    ),
                ));
            }
            if plan.total > 0 && !plan.dgate.disjoint_from(&plan.dup, width) {
                return Err(CudaError::invalid(
                    OP,
                    format!(
                        "dgate (ld {}, off {}) and dup (ld {}, off {}) may share elements",
                        plan.dgate.ld, plan.dgate.off, plan.dup.ld, plan.dup.off
                    ),
                ));
            }
        }
        Ok(plan)
    }

    fn each(&self, mut f: impl FnMut([usize; 5])) {
        for r in 0..self.rows {
            for c in 0..self.width {
                f([
                    self.gate.at(r, c) as usize,
                    self.up.at(r, c) as usize,
                    self.dy.at(r, c) as usize,
                    self.dgate.at(r, c) as usize,
                    self.dup.at(r, c) as usize,
                ]);
            }
        }
    }

    fn check_inputs(&self, gate: usize, up: usize, dy: usize) -> Result<(), CudaError> {
        check_len("swiglu_bwd", "gate", &self.gate, gate)?;
        check_len("swiglu_bwd", "up", &self.up, up)?;
        check_len("swiglu_bwd", "dy", &self.dy, dy)
    }

    /// Into two buffers ([`BwdOut::Separate`]).
    pub fn host_separate(
        &self,
        (gate, up, dy): (&[f32], &[f32], &[f32]),
        dgate: &mut [f32],
        dup: &mut [f32],
    ) -> Result<(), CudaError> {
        if self.out != BwdOut::Separate {
            return Err(CudaError::invalid(
                "swiglu_bwd",
                "the plan writes one shared buffer",
            ));
        }
        self.check_inputs(gate.len(), up.len(), dy.len())?;
        check_len("swiglu_bwd", "dgate", &self.dgate, dgate.len())?;
        check_len("swiglu_bwd", "dup", &self.dup, dup.len())?;
        self.each(|[g, u, d, dg, du]| {
            let (a, b) = swiglu_bwd_elem(gate[g], up[u], dy[d]);
            dgate[dg] = a;
            dup[du] = b;
        });
        Ok(())
    }

    /// Into two windows of one buffer ([`BwdOut::Shared`]).
    pub fn host_shared(
        &self,
        (gate, up, dy): (&[f32], &[f32], &[f32]),
        dgu: &mut [f32],
    ) -> Result<(), CudaError> {
        if self.out != BwdOut::Shared {
            return Err(CudaError::invalid(
                "swiglu_bwd",
                "the plan writes two buffers",
            ));
        }
        self.check_inputs(gate.len(), up.len(), dy.len())?;
        check_len("swiglu_bwd", "dgate", &self.dgate, dgu.len())?;
        check_len("swiglu_bwd", "dup", &self.dup, dgu.len())?;
        self.each(|[g, u, d, dg, du]| {
            let (a, b) = swiglu_bwd_elem(gate[g], up[u], dy[d]);
            dgu[dg] = a;
            dgu[du] = b;
        });
        Ok(())
    }
}

/// `resid[r, :] += y[r, :]` over `rows x width` windows of two buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidualAddPlan {
    pub rows: u64,
    pub width: u64,
    pub y: ColWindow,
    pub resid: ColWindow,
    pub total: u64,
}

impl ResidualAddPlan {
    pub fn new(rows: u64, width: u64, y: WindowSpec, resid: WindowSpec) -> Result<Self, CudaError> {
        const OP: &str = "residual_add";
        Ok(ResidualAddPlan {
            rows,
            width,
            y: ColWindow::new(OP, "y", rows, width, y)?,
            resid: ColWindow::new(OP, "resid", rows, width, resid)?,
            total: total(OP, rows, width)?,
        })
    }

    pub fn host(&self, y: &[f32], resid: &mut [f32]) -> Result<(), CudaError> {
        check_len("residual_add", "y", &self.y, y.len())?;
        check_len("residual_add", "resid", &self.resid, resid.len())?;
        for r in 0..self.rows {
            for c in 0..self.width {
                let (yi, ri) = (self.y.at(r, c) as usize, self.resid.at(r, c) as usize);
                resid[ri] = residual_add_elem(resid[ri], y[yi]);
            }
        }
        Ok(())
    }
}

/// One activation function's host emulation.
pub type ActFn = fn(f32) -> f32;

/// The activation functions the device sweep checks, in the sweep kernel's
/// output order: `out[k * n + i] = f_k(x[i])`.
pub const ACT_SWEEP: [(&str, ActFn); 7] = [
    ("exp", crate::k8_act::exp_f32),
    ("exp_nonpos", crate::k8_act::exp_nonpos_f32),
    ("log", crate::k8_act::log_f32),
    ("softplus", crate::k8_act::softplus_f32),
    ("sigmoid", crate::k8_act::sigmoid_f32),
    ("silu", crate::k8_act::silu_f32),
    ("silu_grad", crate::k8_act::silu_grad_f32),
];

/// The sweep's host side: every function over `x`, function-major.
pub fn act_sweep_host(x: &[f32]) -> Vec<f32> {
    ACT_SWEEP
        .iter()
        .flat_map(|(_, f)| x.iter().map(move |&v| f(v)))
        .collect()
}

/// One K8 window case: shapes and offsets in the shape of a fused
/// `[gate | up]` projection, with values that cross every SiLU regime and
/// carry the special values.
#[derive(Clone, Debug)]
pub struct K8Case {
    pub label: String,
    pub rows: u64,
    pub width: u64,
    /// Row stride of the fused `[gate | up]` input and of its gradient.
    pub ld: u64,
    pub gate_off: u64,
    pub up_off: u64,
    /// The fused input, `rows * ld`.
    pub fused: Vec<f32>,
    /// `dy`, dense `rows x width`.
    pub dy: Vec<f32>,
    /// The residual, dense `rows x width`, and the `y` added into it.
    pub resid: Vec<f32>,
    pub y: Vec<f32>,
}

/// Special values the K8 inputs carry at fixed positions.
const SPECIALS: [f32; 10] = [
    0.0,
    -0.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::MIN_POSITIVE,
    -1.0e-40,
    88.8,
    -104.5,
    1.0e30,
];

/// K8's cases: a 1x1, a ragged 37x129 window inside a 300-wide fused row,
/// and a 64x6144 slab at Qwen3.5-2B's intermediate width.
pub fn k8_cases() -> Vec<K8Case> {
    let shapes = [
        ("one", 1u64, 1u64, 2u64, 0u64, 1u64),
        ("ragged_37x129", 37, 129, 300, 7, 150),
        ("qwen35_2b_64x6144", 64, 6144, 12288, 0, 6144),
    ];
    shapes
        .iter()
        .enumerate()
        .map(|(i, &(label, rows, width, ld, gate_off, up_off))| {
            let seed = 8000 + 10 * i as u64;
            let mut fused = crate::inputs::splitmix_f32(seed, (rows * ld) as usize, 12.0);
            let dense = (rows * width) as usize;
            let mut dy = crate::inputs::splitmix_f32(seed + 1, dense, 2.0);
            let resid = crate::inputs::splitmix_f32(seed + 2, dense, 4.0);
            let mut y = crate::inputs::splitmix_f32(seed + 3, dense, 4.0);
            for (k, &s) in SPECIALS.iter().enumerate() {
                for v in [&mut fused, &mut dy, &mut y] {
                    let at = (k * 7 + 3) % v.len();
                    v[at] = s;
                }
            }
            K8Case {
                label: label.to_string(),
                rows,
                width,
                ld,
                gate_off,
                up_off,
                fused,
                dy,
                resid,
                y,
            }
        })
        .collect()
}

impl K8Case {
    pub fn swiglu_plan(&self) -> Result<SwigluPlan, CudaError> {
        let n = self.fused.len();
        SwigluPlan::new(
            self.rows,
            self.width,
            (self.ld, self.gate_off, n),
            (self.ld, self.up_off, n),
            (self.width, 0, (self.rows * self.width) as usize),
        )
    }

    pub fn bwd_plan(&self, out: BwdOut) -> Result<SwigluBwdPlan, CudaError> {
        let n = self.fused.len();
        let dense = (self.rows * self.width) as usize;
        let (dgate, dup) = match out {
            BwdOut::Shared => ((self.ld, self.gate_off, n), (self.ld, self.up_off, n)),
            BwdOut::Separate => ((self.width, 0, dense), (self.width, 0, dense)),
        };
        SwigluBwdPlan::new(
            self.rows,
            self.width,
            (self.ld, self.gate_off, n),
            (self.ld, self.up_off, n),
            (self.width, 0, dense),
            dgate,
            dup,
            out,
        )
    }

    pub fn residual_plan(&self) -> Result<ResidualAddPlan, CudaError> {
        let dense = (self.rows * self.width) as usize;
        ResidualAddPlan::new(
            self.rows,
            self.width,
            (self.width, 0, dense),
            (self.width, 0, dense),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_elements_are_tessls_operand_order() {
        let (g, u, d) = (0.75f32, -1.25f32, 0.3f32);
        assert_eq!(swiglu_elem(g, u).to_bits(), (silu_f32(g) * u).to_bits());
        let (a, b) = swiglu_bwd_elem(g, u, d);
        assert_eq!(a.to_bits(), ((d * u) * silu_grad_f32(g)).to_bits());
        assert_eq!(b.to_bits(), (d * silu_f32(g)).to_bits());
        // Left to right is not d * (u * s): find an input where they differ,
        // so the order is pinned by a value, not only by the text.
        let differs = (1..4000).any(|i| {
            let x = i as f32 * 0.0137 - 20.0;
            let (u, d) = (1.0 + x * 0.31, 0.7 - x * 0.05);
            ((d * u) * silu_grad_f32(x)).to_bits() != (d * (u * silu_grad_f32(x))).to_bits()
        });
        assert!(differs);
        assert_eq!(residual_add_elem(1.5, 2.25), 3.75);
    }

    #[test]
    fn every_nan_output_is_canonical() {
        let canon = crate::k8_act::CANONICAL_NAN_BITS;
        assert_eq!(swiglu_elem(f32::NAN, 1.0).to_bits(), canon);
        // inf * 0: a fresh NaN, whose bits differ between hosts.
        assert_eq!(swiglu_elem(f32::INFINITY, 0.0).to_bits(), canon);
        assert_eq!(
            residual_add_elem(f32::INFINITY, f32::NEG_INFINITY).to_bits(),
            canon
        );
        let (a, b) = swiglu_bwd_elem(1.0, f32::INFINITY, 0.0);
        assert_eq!((a.to_bits(), b), (canon, 0.0));
        assert_eq!(f32_to_bf16_bits(swiglu_elem(f32::NAN, 1.0)), 0x7fff);
    }

    #[test]
    fn plans_refuse_bad_windows_and_overlapping_shared_outputs() {
        assert!(
            SwigluPlan::new(2, 3, (4, 2, 8), (4, 0, 8), (3, 0, 6)).is_err(),
            "row overrun"
        );
        assert!(
            SwigluPlan::new(2, 3, (4, 1, 7), (4, 0, 8), (3, 0, 6)).is_err(),
            "past end"
        );
        assert!(SwigluPlan::new(2, 3, (4, 1, 8), (4, 0, 8), (3, 0, 6)).is_ok());
        let w = (6, 0, 12);
        let ok = SwigluBwdPlan::new(
            2,
            3,
            w,
            (6, 3, 12),
            w,
            (6, 0, 12),
            (6, 3, 12),
            BwdOut::Shared,
        );
        assert!(ok.is_ok());
        let overlap = SwigluBwdPlan::new(2, 3, w, w, w, (6, 0, 12), (6, 2, 12), BwdOut::Shared);
        assert!(overlap.is_err());
        let strides = SwigluBwdPlan::new(2, 3, w, w, w, (6, 0, 12), (4, 3, 12), BwdOut::Shared);
        assert!(
            strides.is_err(),
            "different strides are refused as unprovable"
        );
        assert!(ResidualAddPlan::new(u64::MAX, 2, (2, 0, 4), (2, 0, 4)).is_err());
    }

    #[test]
    fn the_cases_build_valid_plans_and_their_references_run() {
        for c in k8_cases() {
            let p = c.swiglu_plan().unwrap();
            let mut out = vec![0.0f32; (c.rows * c.width) as usize];
            p.host_f32(&c.fused, &c.fused, &mut out).unwrap();
            let mut out16 = vec![0u16; out.len()];
            p.host_bf16(&c.fused, &c.fused, &mut out16).unwrap();
            for (a, b) in out.iter().zip(&out16) {
                assert_eq!(f32_to_bf16_bits(*a), *b);
            }
            let shared = c.bwd_plan(BwdOut::Shared).unwrap();
            let sep = c.bwd_plan(BwdOut::Separate).unwrap();
            let mut dgu = vec![0.0f32; c.fused.len()];
            shared
                .host_shared((&c.fused, &c.fused, &c.dy), &mut dgu)
                .unwrap();
            let (mut dg, mut du) = (vec![0.0f32; out.len()], vec![0.0f32; out.len()]);
            sep.host_separate((&c.fused, &c.fused, &c.dy), &mut dg, &mut du)
                .unwrap();
            for r in 0..c.rows {
                for col in 0..c.width {
                    let i = (r * c.width + col) as usize;
                    assert_eq!(
                        dg[i].to_bits(),
                        dgu[shared.dgate.at(r, col) as usize].to_bits()
                    );
                    assert_eq!(
                        du[i].to_bits(),
                        dgu[shared.dup.at(r, col) as usize].to_bits()
                    );
                }
            }
            let mut resid = c.resid.clone();
            c.residual_plan().unwrap().host(&c.y, &mut resid).unwrap();
        }
    }

    #[test]
    fn the_sweep_is_function_major() {
        let x = [0.5f32, -2.0];
        let s = act_sweep_host(&x);
        assert_eq!(s.len(), 14);
        assert_eq!(s[0].to_bits(), crate::k8_act::exp_f32(0.5).to_bits());
        assert_eq!(s[13].to_bits(), silu_grad_f32(-2.0).to_bits());
    }
}
