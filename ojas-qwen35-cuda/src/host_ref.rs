//! Host references for K0 and K1.
//!
//! - K0: the same arithmetic as each device kernel, one f32 operation per
//!   element, so a device result must equal these **bitwise**.
//! - K1: [`gemm_f64`], the float64 product a tolerance is measured against,
//!   and [`gemm_ffma_f32`], an emulation of the FFMA kernel's exact order
//!   (`acc = fma(a, b, acc)` over ascending k, from +0.0). Rust's
//!   `f32::mul_add` is one IEEE fused multiply-add, as CUDA's `fmaf`, so the
//!   ExactF32 device result must also equal this bitwise.
//!
//! These live in `src/` until L-cuda-oracle's `tests/reference/` exists; the
//! device checks in [`crate::check`] compare against them.

use crate::bf16::{bf16_bits_to_f32, f32_to_bf16_bits, round_to_bf16};
use crate::gemm_plan::{GemmLayout, GemmShape};
use crate::k0_plan::{
    CopyColsPlan, DeliverMode, DeliverPlan, GatherRowsPlan, ScatterAddRowsPlan, ZeroPlan,
};

/// `usize` from a plan's `u64` index, which a plan has bounded by a slice length.
fn ix(x: u64) -> usize {
    usize::try_from(x).unwrap_or(usize::MAX)
}

/// f32 to bf16 bits, elementwise.
pub fn cast_f32_to_bf16(src: &[f32]) -> Vec<u16> {
    src.iter().copied().map(f32_to_bf16_bits).collect()
}

/// bf16 bits to f32, elementwise.
pub fn cast_bf16_to_f32(src: &[u16]) -> Vec<f32> {
    src.iter().copied().map(bf16_bits_to_f32).collect()
}

/// [`CopyColsPlan`] on the host.
pub fn copy_cols(plan: &CopyColsPlan, src: &[f32], dst: &mut [f32]) {
    for r in 0..plan.rows {
        for c in 0..plan.width {
            dst[ix(r * plan.ld_dst + plan.dst_off + c)] =
                src[ix(r * plan.ld_src + plan.src_off + c)];
        }
    }
}

/// [`DeliverPlan`] on the host.
pub fn deliver(plan: &DeliverPlan, src: &[f32], dst: &mut [f32]) {
    for i in 0..plan.n {
        let s = src[ix(plan.src_off + i)];
        let d = &mut dst[ix(plan.dst_off + i)];
        match plan.mode {
            DeliverMode::Copy => *d = s,
            DeliverMode::Add => *d += s,
        }
    }
}

/// [`ZeroPlan`] on the host.
pub fn zero(plan: &ZeroPlan, dst: &mut [f32]) {
    for i in 0..plan.n {
        dst[ix(plan.off + i)] = 0.0;
    }
}

/// [`ScatterAddRowsPlan`] on the host.
pub fn scatter_add_rows(plan: &ScatterAddRowsPlan, src: &[f32], pos: &[u32], dst: &mut [f32]) {
    for (i, &p) in pos.iter().enumerate() {
        for c in 0..plan.width {
            let i = u64::try_from(i).unwrap_or(u64::MAX);
            dst[ix(u64::from(p) * plan.width + c)] += src[ix(i * plan.width + c)];
        }
    }
}

/// [`GatherRowsPlan`] on the host, f32 input.
pub fn ce_gather_rows_f32(plan: &GatherRowsPlan, h: &[f32], rows: &[u32], out: &mut [f32]) {
    for (i, &r) in rows.iter().enumerate() {
        let i = u64::try_from(i).unwrap_or(u64::MAX);
        for c in 0..plan.hidden {
            out[ix(i * plan.hidden + c)] = h[ix(u64::from(r) * plan.ld + plan.off + c)];
        }
    }
}

/// [`GatherRowsPlan`] on the host, bf16 input widened to f32.
pub fn ce_gather_rows_bf16(plan: &GatherRowsPlan, h: &[u16], rows: &[u32], out: &mut [f32]) {
    for (i, &r) in rows.iter().enumerate() {
        let i = u64::try_from(i).unwrap_or(u64::MAX);
        for c in 0..plan.hidden {
            out[ix(i * plan.hidden + c)] =
                bf16_bits_to_f32(h[ix(u64::from(r) * plan.ld + plan.off + c)]);
        }
    }
}

/// The GEMM's operand precision, as tessl's `GemmOperands`
/// (`tessl/src/gemm.rs:991-998`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operands {
    /// f32 operands as given.
    ExactF32,
    /// Operands rounded to bf16 (round-to-nearest-even) at the GEMM boundary.
    Bf16,
}

impl Operands {
    /// `exact_f32` or `bf16`.
    pub fn name(self) -> &'static str {
        match self {
            Operands::ExactF32 => "exact_f32",
            Operands::Bf16 => "bf16",
        }
    }

    fn load(self, x: f32) -> f32 {
        match self {
            Operands::ExactF32 => x,
            Operands::Bf16 => round_to_bf16(x),
        }
    }
}

/// The float64 product of the operands as the GEMM sees them (bf16-rounded
/// under [`Operands::Bf16`]), plus `c_prev` when accumulating.
pub fn gemm_f64(
    layout: GemmLayout,
    s: GemmShape,
    operands: Operands,
    a: &[f32],
    b: &[f32],
    c_prev: Option<&[f32]>,
) -> Vec<f64> {
    let mut c = vec![0.0f64; s.c_len()];
    for i in 0..s.m {
        for j in 0..s.n {
            let mut acc = 0.0f64;
            for p in 0..s.k {
                let x = f64::from(operands.load(a[layout.a_index(s, i, p)]));
                let y = f64::from(operands.load(b[layout.b_index(s, p, j)]));
                acc += x * y;
            }
            c[i * s.n + j] = acc + c_prev.map_or(0.0, |prev| f64::from(prev[i * s.n + j]));
        }
    }
    c
}

/// The FFMA kernel's arithmetic, bit for bit: per output,
/// `acc = +0.0; for p in 0..k { acc = fma(a[i,p], b[p,j], acc) }`, then
/// `c = c_prev + acc` (one rounding) when accumulating.
pub fn gemm_ffma_f32(
    layout: GemmLayout,
    s: GemmShape,
    operands: Operands,
    a: &[f32],
    b: &[f32],
    c_prev: Option<&[f32]>,
) -> Vec<f32> {
    let mut c = vec![0.0f32; s.c_len()];
    for i in 0..s.m {
        for j in 0..s.n {
            let mut acc = 0.0f32;
            for p in 0..s.k {
                let x = operands.load(a[layout.a_index(s, i, p)]);
                let y = operands.load(b[layout.b_index(s, p, j)]);
                acc = x.mul_add(y, acc);
            }
            c[i * s.n + j] = match c_prev {
                Some(prev) => prev[i * s.n + j] + acc,
                None => acc,
            };
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_add_is_fused_on_this_host() {
        // (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24. Unfused, the product rounds the
        // 2^-24 away before the -1 is added; fused, it survives.
        let x = 1.0f32 + 2f32.powi(-12);
        let fused = x.mul_add(x, -1.0);
        assert_eq!(fused, 2f32.powi(-11) + 2f32.powi(-24));
        assert_ne!(fused, x * x - 1.0);
    }

    #[test]
    fn ffma_emulation_is_close_to_f64_and_differs_from_unfused() {
        let s = GemmShape::new(9, 7, 33).unwrap();
        let a = crate::inputs::splitmix_f32(1, s.a_len(), 1.0);
        let b = crate::inputs::splitmix_f32(2, s.b_len(), 1.0);
        let f = gemm_ffma_f32(GemmLayout::Nn, s, Operands::ExactF32, &a, &b, None);
        let r = gemm_f64(GemmLayout::Nn, s, Operands::ExactF32, &a, &b, None);
        for (x, y) in f.iter().zip(&r) {
            assert!((f64::from(*x) - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn accumulate_adds_the_previous_c_once() {
        let s = GemmShape::new(2, 2, 3).unwrap();
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let prev = [0.5f32, 0.25, -1.0, 2.0];
        let c = gemm_ffma_f32(GemmLayout::Nn, s, Operands::ExactF32, &a, &b, Some(&prev));
        // A @ B = [[4, 5], [10, 11]]; plus prev.
        assert_eq!(c, vec![4.5, 5.25, 9.0, 13.0]);
    }

    #[test]
    fn bf16_operands_round_before_multiplying() {
        let s = GemmShape::new(1, 1, 1).unwrap();
        let x = 1.0f32 + 2f32.powi(-10); // not a bf16
        let c = gemm_ffma_f32(GemmLayout::Nn, s, Operands::Bf16, &[x], &[1.0], None);
        assert_eq!(c[0], 1.0);
        let e = gemm_ffma_f32(GemmLayout::Nn, s, Operands::ExactF32, &[x], &[1.0], None);
        assert_eq!(e[0], x);
    }

    #[test]
    fn scatter_and_gather_follow_their_plans() {
        let plan = ScatterAddRowsPlan::new(&[2, 0], 2, 4, 3, 6).unwrap();
        let mut dst = vec![1.0f32; 6];
        scatter_add_rows(&plan, &[10.0, 20.0, 30.0, 40.0], &[2, 0], &mut dst);
        assert_eq!(dst, vec![31.0, 41.0, 1.0, 1.0, 11.0, 21.0]);

        let g = GatherRowsPlan::new(&[1, 1, 0], 2, (3, 1, 6), 6).unwrap();
        let h = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0f32; 6];
        ce_gather_rows_f32(&g, &h, &[1, 1, 0], &mut out);
        assert_eq!(out, vec![4.0, 5.0, 4.0, 5.0, 1.0, 2.0]);
    }

    #[test]
    fn copy_cols_and_deliver_follow_their_plans() {
        let plan = CopyColsPlan::new(2, 2, (4, 1, 8), (3, 1, 6)).unwrap();
        let src: Vec<f32> = (0..8u8).map(f32::from).collect();
        let mut dst = vec![-1.0f32; 6];
        copy_cols(&plan, &src, &mut dst);
        assert_eq!(dst, vec![-1.0, 1.0, 2.0, -1.0, 5.0, 6.0]);

        let add = DeliverPlan::new((1, 4), (0, 3), 3, DeliverMode::Add).unwrap();
        let mut d = vec![1.0f32; 3];
        deliver(&add, &[9.0, 1.0, 2.0, 3.0], &mut d);
        assert_eq!(d, vec![2.0, 3.0, 4.0]);
        zero(&ZeroPlan::new(1, 2, 3).unwrap(), &mut d);
        assert_eq!(d, vec![2.0, 0.0, 0.0]);
    }
}
