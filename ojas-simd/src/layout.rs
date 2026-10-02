//! Shape and stride validation. Nothing in this crate reads a buffer until a
//! [`Problem`] has been built from it.

use crate::{Operand, SimdError};

/// A GEMM whose every addressed element is in bounds.
///
/// Built only by [`Problem::validate`]. Any `row < rows`, `col < cols` index
/// `row*rs + col*cs` of an operand is at most its validated extent minus one,
/// so it neither overflows nor leaves the slice.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Problem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a_rs: usize,
    pub a_cs: usize,
    pub b_rs: usize,
    pub b_cs: usize,
    pub c_rs: usize,
    pub accumulate: bool,
}

/// `(rows-1)*rs + (cols-1)*cs + 1`, or 0 when the operand is empty.
fn extent(
    rows: usize,
    cols: usize,
    rs: usize,
    cs: usize,
    operand: Operand,
) -> Result<usize, SimdError> {
    if rows == 0 || cols == 0 {
        return Ok(0);
    }
    (rows - 1)
        .checked_mul(rs)
        .and_then(|r| (cols - 1).checked_mul(cs).and_then(|c| r.checked_add(c)))
        .and_then(|last| last.checked_add(1))
        .ok_or(SimdError::ExtentOverflow { operand })
}

fn check_len(required: usize, len: usize, operand: Operand) -> Result<(), SimdError> {
    if len < required {
        Err(SimdError::BufferTooShort {
            operand,
            required,
            len,
        })
    } else {
        Ok(())
    }
}

impl Problem {
    #[allow(clippy::too_many_arguments)]
    pub fn validate(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        a_rs: usize,
        a_cs: usize,
        b: &[f32],
        b_rs: usize,
        b_cs: usize,
        c: &[f32],
        c_rs: usize,
        accumulate: bool,
    ) -> Result<Problem, SimdError> {
        // An operand that is never read (m or n is 0) is not checked.
        let touches_inputs = m > 0 && n > 0;
        if touches_inputs {
            check_len(extent(m, k, a_rs, a_cs, Operand::A)?, a.len(), Operand::A)?;
            check_len(extent(k, n, b_rs, b_cs, Operand::B)?, b.len(), Operand::B)?;
        }
        check_len(extent(m, n, c_rs, 1, Operand::C)?, c.len(), Operand::C)?;
        if m > 1 && n > 0 && c_rs < n {
            return Err(SimdError::OverlappingOutputRows { n, c_rs });
        }
        Ok(Problem {
            m,
            n,
            k,
            a_rs,
            a_cs,
            b_rs,
            b_cs,
            c_rs,
            accumulate,
        })
    }
}

/// How one operand is addressed by row-major CBLAS.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlasOperand {
    /// `CblasTrans`: the operand is stored as its transpose, row-major.
    pub trans: bool,
    pub ld: i32,
}

/// A validated `cblas_sgemm` call in row-major order.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlasCall {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub a: BlasOperand,
    pub b: BlasOperand,
    pub ldc: i32,
    pub beta: f32,
}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
fn to_int(v: usize) -> Result<i32, SimdError> {
    i32::try_from(v).map_err(|_| SimdError::DimensionTooLarge { value: v })
}

/// Row-major CBLAS addressing for a `rows × cols` operand with strides
/// `(rs, cs)`. A stride along an axis of length 1 is never multiplied by a
/// nonzero index, so it is replaced by whichever value makes a layout legal.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
fn blas_operand(
    rows: usize,
    cols: usize,
    mut rs: usize,
    mut cs: usize,
    operand: Operand,
) -> Result<BlasOperand, SimdError> {
    if rows <= 1 {
        rs = if cs == 1 { cols.max(1) } else { 1 };
    }
    if cols <= 1 {
        cs = if rs == 1 { rows.max(1) } else { 1 };
    }
    if cs == 1 && rs >= cols.max(1) {
        Ok(BlasOperand {
            trans: false,
            ld: to_int(rs)?,
        })
    } else if rs == 1 && cs >= rows.max(1) {
        Ok(BlasOperand {
            trans: true,
            ld: to_int(cs)?,
        })
    } else {
        Err(SimdError::UnsupportedLayout { operand })
    }
}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
impl BlasCall {
    pub fn from_problem(p: &Problem) -> Result<BlasCall, SimdError> {
        let ldc = if p.m <= 1 { p.n.max(1) } else { p.c_rs };
        Ok(BlasCall {
            m: to_int(p.m)?,
            n: to_int(p.n)?,
            k: to_int(p.k)?,
            a: blas_operand(p.m, p.k, p.a_rs, p.a_cs, Operand::A)?,
            b: blas_operand(p.k, p.n, p.b_rs, p.b_cs, Operand::B)?,
            ldc: to_int(ldc.max(1))?,
            beta: if p.accumulate { 1.0 } else { 0.0 },
        })
    }
}
