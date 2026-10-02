//! GEMM shapes, layouts, and the row-major to column-major mapping for cuBLAS.
//!
//! All matrices here are row-major, as in tessl (`GemmOperands::{nn,tn,nt}`,
//! `tessl/src/gemm.rs:1012-1047`):
//! - `nn`: `C[m,n] = A[m,k] @ B[k,n]`;
//! - `tn`: `C[m,n] = A^T @ B`, with A stored `[k, m]`;
//! - `nt`: `C[m,n] = A @ B^T`, with B stored `[n, k]`.
//!
//! cuBLAS is column-major. A row-major `X[r, c]` with row stride `c` is the
//! column-major `X^T`, so `C = op(A) op(B)` in row-major is
//! `C^T = op(B)^T op(A)^T` in column-major: cuBLAS's first operand is B,
//! its second is A, and its output is C^T (`m' = n`, `n' = m`). [`cublas_args`]
//! computes the transposes and leading dimensions; its test checks them
//! against a host emulation of BLAS's column-major definition.

use crate::error::CudaError;

/// Which operand is stored transposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmLayout {
    /// `C = A @ B`.
    Nn,
    /// `C = A^T @ B`, A stored `[k, m]`.
    Tn,
    /// `C = A @ B^T`, B stored `[n, k]`.
    Nt,
}

impl GemmLayout {
    /// Every layout, for tests and the rung-0 sweep.
    pub const ALL: [GemmLayout; 3] = [GemmLayout::Nn, GemmLayout::Tn, GemmLayout::Nt];

    /// `nn`, `tn` or `nt`.
    pub fn name(self) -> &'static str {
        match self {
            GemmLayout::Nn => "nn",
            GemmLayout::Tn => "tn",
            GemmLayout::Nt => "nt",
        }
    }

    /// The layout code the FFMA kernel template takes (0, 1, 2).
    pub fn code(self) -> u32 {
        match self {
            GemmLayout::Nn => 0,
            GemmLayout::Tn => 1,
            GemmLayout::Nt => 2,
        }
    }

    /// Row-major index of logical `A[i, p]` in A's storage.
    pub fn a_index(self, s: GemmShape, i: usize, p: usize) -> usize {
        match self {
            GemmLayout::Tn => p * s.m + i,
            GemmLayout::Nn | GemmLayout::Nt => i * s.k + p,
        }
    }

    /// Row-major index of logical `B[p, j]` in B's storage.
    pub fn b_index(self, s: GemmShape, p: usize, j: usize) -> usize {
        match self {
            GemmLayout::Nt => j * s.k + p,
            GemmLayout::Nn | GemmLayout::Tn => p * s.n + j,
        }
    }
}

/// A validated `m x n x k` GEMM shape: every side at least 1, every side and
/// leading dimension within cuBLAS's `int`, every operand length within `usize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemmShape {
    /// Rows of C.
    pub m: usize,
    /// Columns of C.
    pub n: usize,
    /// The contracted dimension.
    pub k: usize,
}

impl GemmShape {
    /// Validate a shape. A zero side is refused rather than treated as a no-op:
    /// callers with nothing to multiply skip the call.
    pub fn new(m: usize, n: usize, k: usize) -> Result<Self, CudaError> {
        for (name, v) in [("m", m), ("n", n), ("k", k)] {
            if v == 0 {
                return Err(CudaError::invalid("GemmShape", format!("{name} is 0")));
            }
            if i32::try_from(v).is_err() {
                return Err(CudaError::invalid(
                    "GemmShape",
                    format!("{name} = {v} does not fit cuBLAS's int"),
                ));
            }
        }
        for (name, a, b) in [("m*k", m, k), ("k*n", k, n), ("m*n", m, n)] {
            if a.checked_mul(b).is_none() {
                return Err(CudaError::invalid(
                    "GemmShape",
                    format!("{name} = {a}*{b} overflows usize"),
                ));
            }
        }
        Ok(GemmShape { m, n, k })
    }

    /// Elements of A's storage.
    pub fn a_len(self) -> usize {
        self.m * self.k
    }

    /// Elements of B's storage.
    pub fn b_len(self) -> usize {
        self.k * self.n
    }

    /// Elements of C.
    pub fn c_len(self) -> usize {
        self.m * self.n
    }

    /// Check operand lengths before any device work.
    pub fn check_lens(self, a: usize, b: usize, c: usize, op: &str) -> Result<(), CudaError> {
        for (name, got, want) in [
            ("A", a, self.a_len()),
            ("B", b, self.b_len()),
            ("C", c, self.c_len()),
        ] {
            if got != want {
                return Err(CudaError::invalid(
                    op,
                    format!(
                        "{name} has {got} elements; a {}x{}x{} GEMM needs {want}",
                        self.m, self.n, self.k
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// A BLAS transpose flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trans {
    /// `CUBLAS_OP_N`.
    N,
    /// `CUBLAS_OP_T`.
    T,
}

/// The arguments of one column-major `cublasGemmEx` that computes a row-major
/// GEMM. cuBLAS's first operand is the row-major **B**, its second is **A**.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CublasArgs {
    /// `transa`: applied to the first operand (B).
    pub trans_first: Trans,
    /// `transb`: applied to the second operand (A).
    pub trans_second: Trans,
    /// cuBLAS `m` (rows of the column-major output) = row-major `n`.
    pub m: i32,
    /// cuBLAS `n` = row-major `m`.
    pub n: i32,
    /// cuBLAS `k` = row-major `k`.
    pub k: i32,
    /// `lda`: leading dimension of the first operand (B).
    pub ld_first: i32,
    /// `ldb`: leading dimension of the second operand (A).
    pub ld_second: i32,
    /// `ldc` = row-major `n`.
    pub ldc: i32,
}

/// The cuBLAS call for a row-major GEMM of `layout` and `shape`.
pub fn cublas_args(layout: GemmLayout, shape: GemmShape) -> Result<CublasArgs, CudaError> {
    let int = |v: usize, what: &str| {
        i32::try_from(v).map_err(|_| {
            CudaError::invalid(
                "cublas_args",
                format!("{what} = {v} does not fit cuBLAS's int"),
            )
        })
    };
    let (m, n, k) = (int(shape.m, "m")?, int(shape.n, "n")?, int(shape.k, "k")?);
    // Row-major storage widths, which are the column-major leading dimensions.
    let (trans_first, ld_first, trans_second, ld_second) = match layout {
        // B [k,n] row-major is B^T column-major ([n,k], ld n): use as is.
        // A [m,k] row-major is A^T column-major ([k,m], ld k): use as is.
        GemmLayout::Nn => (Trans::N, n, Trans::N, k),
        // A stored [k,m] row-major is logical A column-major ([m,k], ld m): transpose it.
        GemmLayout::Tn => (Trans::N, n, Trans::T, m),
        // B stored [n,k] row-major is logical B column-major ([k,n], ld k): transpose it.
        GemmLayout::Nt => (Trans::T, k, Trans::N, k),
    };
    let args = CublasArgs {
        trans_first,
        trans_second,
        m: n,
        n: m,
        k,
        ld_first,
        ld_second,
        ldc: n,
    };
    check_leading_dims(args)?;
    Ok(args)
}

/// BLAS's own leading-dimension rule: `lda >= max(1, rows of A as stored)`.
fn check_leading_dims(a: CublasArgs) -> Result<(), CudaError> {
    let first_rows = match a.trans_first {
        Trans::N => a.m,
        Trans::T => a.k,
    };
    let second_rows = match a.trans_second {
        Trans::N => a.k,
        Trans::T => a.n,
    };
    for (name, ld, rows) in [
        ("lda", a.ld_first, first_rows),
        ("ldb", a.ld_second, second_rows),
        ("ldc", a.ldc, a.m),
    ] {
        if ld < rows.max(1) {
            return Err(CudaError::invalid(
                "cublas_args",
                format!("{name} = {ld} is below the {rows} stored rows"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BLAS's definition, column-major:
    /// `C[i + j*ldc] = sum_p op(A)[i,p] * op(B)[p,j]`, where
    /// `op(X)[r,c] = X[r + c*ld]` for N and `X[c + r*ld]` for T.
    fn colmajor_gemm(a: CublasArgs, first: &[f64], second: &[f64]) -> Vec<f64> {
        let (m, n, k) = (a.m as usize, a.n as usize, a.k as usize);
        let at = |x: &[f64], t: Trans, ld: usize, r: usize, c: usize| match t {
            Trans::N => x[r + c * ld],
            Trans::T => x[c + r * ld],
        };
        let mut c = vec![0.0f64; a.ldc as usize * n];
        for j in 0..n {
            for i in 0..m {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += at(first, a.trans_first, a.ld_first as usize, i, p)
                        * at(second, a.trans_second, a.ld_second as usize, p, j);
                }
                c[i + j * a.ldc as usize] = acc;
            }
        }
        c
    }

    fn rowmajor_gemm(layout: GemmLayout, s: GemmShape, a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0f64; s.c_len()];
        for i in 0..s.m {
            for j in 0..s.n {
                let mut acc = 0.0;
                for p in 0..s.k {
                    acc += a[layout.a_index(s, i, p)] * b[layout.b_index(s, p, j)];
                }
                c[i * s.n + j] = acc;
            }
        }
        c
    }

    #[test]
    fn the_cublas_mapping_computes_the_row_major_product_for_every_layout() {
        for layout in GemmLayout::ALL {
            for (m, n, k) in [(1, 1, 1), (2, 3, 4), (5, 2, 7), (13, 7, 3), (3, 11, 1)] {
                let s = GemmShape::new(m, n, k).unwrap();
                // Small integers: every sum is exact in f64, so equality is exact.
                let a: Vec<f64> = (0..s.a_len())
                    .map(|i| ((i * 7 + 3) % 11) as f64 - 5.0)
                    .collect();
                let b: Vec<f64> = (0..s.b_len())
                    .map(|i| ((i * 5 + 1) % 13) as f64 - 6.0)
                    .collect();
                let args = cublas_args(layout, s).unwrap();
                let got = colmajor_gemm(args, &b, &a);
                let want = rowmajor_gemm(layout, s, &a, &b);
                assert_eq!(got, want, "{} {m}x{n}x{k}: {args:?}", layout.name());
            }
        }
    }

    #[test]
    fn wrong_mappings_are_caught() {
        // The check above is not satisfied by any mapping: passing A first
        // (BLAS's natural order) or flipping a transpose gives another product.
        // A square shape keeps every wrong variant in bounds.
        let s = GemmShape::new(3, 3, 3).unwrap();
        let a: Vec<f64> = (0..9).map(|i| f64::from(i) + 1.0).collect();
        let b: Vec<f64> = (0..9).map(|i| f64::from(i * 3) - 2.0).collect();
        for layout in GemmLayout::ALL {
            let args = cublas_args(layout, s).unwrap();
            let want = rowmajor_gemm(layout, s, &a, &b);
            assert_ne!(colmajor_gemm(args, &a, &b), want, "{}", layout.name());
            let flip = |t| match t {
                Trans::N => Trans::T,
                Trans::T => Trans::N,
            };
            let flipped = CublasArgs {
                trans_second: flip(args.trans_second),
                ..args
            };
            assert_ne!(colmajor_gemm(flipped, &b, &a), want, "{}", layout.name());
        }
    }

    #[test]
    fn shapes_refuse_zero_sides_and_out_of_int_sides() {
        assert!(GemmShape::new(0, 1, 1).is_err());
        assert!(GemmShape::new(1, 0, 1).is_err());
        assert!(GemmShape::new(1, 1, 0).is_err());
        assert!(GemmShape::new(1usize << 31, 1, 1).is_err());
        let s = GemmShape::new(130, 70, 260).unwrap();
        assert!(s.check_lens(130 * 260, 260 * 70, 130 * 70, "t").is_ok());
        assert!(s
            .check_lens(130 * 260, 260 * 70, 130 * 70 - 1, "t")
            .is_err());
    }

    #[test]
    fn index_helpers_match_the_storage_layouts() {
        let s = GemmShape::new(3, 4, 5).unwrap();
        assert_eq!(GemmLayout::Nn.a_index(s, 2, 4), 2 * 5 + 4);
        assert_eq!(GemmLayout::Tn.a_index(s, 2, 4), 4 * 3 + 2);
        assert_eq!(GemmLayout::Nn.b_index(s, 4, 3), 4 * 4 + 3);
        assert_eq!(GemmLayout::Nt.b_index(s, 4, 3), 3 * 5 + 4);
    }
}
