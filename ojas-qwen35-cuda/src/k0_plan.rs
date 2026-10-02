//! Validated plans for the K0 plumbing ops.
//!
//! Each plan is built from its arguments and the lengths of the buffers it
//! will touch, with every index computed in checked `u64`. A plan that exists
//! cannot read or write outside those lengths, so the host reference and the
//! device launch both take a plan and neither re-validates. The semantics are
//! tessl's (`tessl/kernels/qwen35_bwd.metal:674-718`,
//! `tessl/kernels/cross_entropy.metal:28-42`, `tessl/src/qwen35_train.rs:227-243`).

use crate::error::CudaError;

fn to_u64(x: usize) -> u64 {
    // usize is at most 64 bits on every target this crate builds for.
    u64::try_from(x).unwrap_or(u64::MAX)
}

fn mul(a: u64, b: u64, op: &str) -> Result<u64, CudaError> {
    a.checked_mul(b)
        .ok_or_else(|| CudaError::invalid(op, format!("{a} * {b} overflows u64")))
}

fn add(a: u64, b: u64, op: &str) -> Result<u64, CudaError> {
    a.checked_add(b)
        .ok_or_else(|| CudaError::invalid(op, format!("{a} + {b} overflows u64")))
}

/// One past the last element a `[rows, ld]` row-major window of `width`
/// columns at column `off` touches; 0 when the window is empty.
fn window_end(rows: u64, ld: u64, off: u64, width: u64, op: &str) -> Result<u64, CudaError> {
    if rows == 0 || width == 0 {
        return Ok(0);
    }
    add(add(mul(rows - 1, ld, op)?, off, op)?, width, op)
}

/// One validated column window of a row-major matrix held in a buffer:
/// `rows` rows `ld` elements apart, `width` columns from column `off`. It
/// lies inside one row (`off + width <= ld`: a window crossing into the next
/// row is a layout bug) and inside the buffer's `len` elements. K0's
/// `copy_cols` and every K8 window are one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColWindow {
    /// Row stride, in elements.
    pub ld: u64,
    /// First column.
    pub off: u64,
    /// One past the last element touched; 0 for an empty window.
    pub end: u64,
}

impl ColWindow {
    /// Validate `name`'s window `(ld, off)` over a `len`-element buffer.
    pub fn new(
        op: &str,
        name: &str,
        rows: u64,
        width: u64,
        (ld, off, len): (u64, u64, usize),
    ) -> Result<Self, CudaError> {
        if add(off, width, op)? > ld {
            return Err(CudaError::invalid(
                op,
                format!(
                    "{name} window {off}..{} is wider than its row stride {ld}",
                    off + width
                ),
            ));
        }
        let end = window_end(rows, ld, off, width, op)?;
        if end > to_u64(len) {
            return Err(CudaError::invalid(
                op,
                format!("{name} window ends at {end}, past its {len} elements"),
            ));
        }
        Ok(ColWindow { ld, off, end })
    }

    /// The buffer index of row `r`, column `c` (both inside the window).
    pub fn at(&self, r: u64, c: u64) -> u64 {
        r * self.ld + self.off + c
    }

    /// Whether two `width`-wide windows of one buffer share no element. Only
    /// windows with the same row stride are compared (disjoint column
    /// ranges); any other pair is refused as unprovable.
    pub fn disjoint_from(&self, other: &ColWindow, width: u64) -> bool {
        self.ld == other.ld && (self.off + width <= other.off || other.off + width <= self.off)
    }
}

/// `dst[r, dst_off + c] = src[r, src_off + c]` for `r < rows`, `c < width`:
/// a column window of one row-major matrix into another's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyColsPlan {
    /// Rows copied.
    pub rows: u64,
    /// Columns copied per row.
    pub width: u64,
    /// Row stride of `src`, in elements.
    pub ld_src: u64,
    /// First copied column of `src`.
    pub src_off: u64,
    /// Row stride of `dst`, in elements.
    pub ld_dst: u64,
    /// First written column of `dst`.
    pub dst_off: u64,
    /// `rows * width`: the elements the kernel visits.
    pub total: u64,
}

impl CopyColsPlan {
    /// Validate the window against both buffers. Each window must lie inside
    /// one row (`off + width <= ld`): one crossing into the next row is a
    /// layout bug, not a copy.
    pub fn new(
        rows: u64,
        width: u64,
        (ld_src, src_off, src_len): (u64, u64, usize),
        (ld_dst, dst_off, dst_len): (u64, u64, usize),
    ) -> Result<Self, CudaError> {
        const OP: &str = "copy_cols";
        ColWindow::new(OP, "src", rows, width, (ld_src, src_off, src_len))?;
        ColWindow::new(OP, "dst", rows, width, (ld_dst, dst_off, dst_len))?;
        Ok(CopyColsPlan {
            rows,
            width,
            ld_src,
            src_off,
            ld_dst,
            dst_off,
            total: mul(rows, width, OP)?,
        })
    }
}

/// Whether [`DeliverPlan`] overwrites or adds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeliverMode {
    /// `dst = src`.
    Copy,
    /// `dst += src`, one f32 rounding per element.
    Add,
}

/// `dst[dst_off + i] (=|+=) src[src_off + i]` for `i < n`: tessl's `deliver`
/// of a gradient part into a bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeliverPlan {
    /// First element read.
    pub src_off: u64,
    /// First element written.
    pub dst_off: u64,
    /// Elements.
    pub n: u64,
    /// Copy or add.
    pub mode: DeliverMode,
}

impl DeliverPlan {
    /// Validate both ranges.
    pub fn new(
        (src_off, src_len): (u64, usize),
        (dst_off, dst_len): (u64, usize),
        n: u64,
        mode: DeliverMode,
    ) -> Result<Self, CudaError> {
        const OP: &str = "deliver";
        if add(src_off, n, OP)? > to_u64(src_len) {
            return Err(CudaError::invalid(
                OP,
                format!("src range {src_off}+{n} is past its {src_len} elements"),
            ));
        }
        if add(dst_off, n, OP)? > to_u64(dst_len) {
            return Err(CudaError::invalid(
                OP,
                format!("dst range {dst_off}+{n} is past its {dst_len} elements"),
            ));
        }
        Ok(DeliverPlan {
            src_off,
            dst_off,
            n,
            mode,
        })
    }
}

/// `dst[off + i] = 0.0` for `i < n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroPlan {
    /// First element zeroed.
    pub off: u64,
    /// Elements.
    pub n: u64,
}

impl ZeroPlan {
    /// Validate the range.
    pub fn new(off: u64, n: u64, len: usize) -> Result<Self, CudaError> {
        if add(off, n, "zero")? > to_u64(len) {
            return Err(CudaError::invalid(
                "zero",
                format!("range {off}+{n} is past its {len} elements"),
            ));
        }
        Ok(ZeroPlan { off, n })
    }
}

/// `dst[pos[i] * width + c] += src[i * width + c]`: dense `[n, width]` rows
/// added into rows `pos` of a dense `[dst_rows, width]` matrix.
///
/// Ownership by row: positions must be distinct, so no two threads add into
/// one element and no float atomics are needed. A repeated position is
/// refused, naming both indices (tessl checks the same on its host,
/// `qwen35_train.rs:584-592`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScatterAddRowsPlan {
    /// Rows of `src`, and the length of `pos`.
    pub n: u64,
    /// Columns.
    pub width: u64,
    /// Rows of `dst`.
    pub dst_rows: u64,
    /// `n * width`: the elements the kernel visits.
    pub total: u64,
}

impl ScatterAddRowsPlan {
    /// Validate shapes, ranges and distinctness.
    pub fn new(
        pos: &[u32],
        width: u64,
        src_len: usize,
        dst_rows: u64,
        dst_len: usize,
    ) -> Result<Self, CudaError> {
        const OP: &str = "scatter_add_rows";
        let n = to_u64(pos.len());
        let total = mul(n, width, OP)?;
        if total != to_u64(src_len) {
            return Err(CudaError::invalid(
                OP,
                format!("src has {src_len} elements, expected {n} rows x {width}"),
            ));
        }
        if mul(dst_rows, width, OP)? != to_u64(dst_len) {
            return Err(CudaError::invalid(
                OP,
                format!("dst has {dst_len} elements, expected {dst_rows} rows x {width}"),
            ));
        }
        if let Some((i, &p)) = pos
            .iter()
            .enumerate()
            .find(|(_, &p)| u64::from(p) >= dst_rows)
        {
            return Err(CudaError::invalid(
                OP,
                format!("pos[{i}] = {p} is not below the {dst_rows} destination rows"),
            ));
        }
        let mut order: Vec<(u32, usize)> = pos.iter().copied().zip(0..).collect();
        order.sort_unstable();
        if let Some(w) = order.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(CudaError::invalid(
                OP,
                format!(
                    "row {} appears at pos[{}] and pos[{}]; rows must be distinct (one owner per row)",
                    w[0].0, w[0].1, w[1].1
                ),
            ));
        }
        Ok(ScatterAddRowsPlan {
            n,
            width,
            dst_rows,
            total,
        })
    }
}

/// `out[i, c] = h[rows[i] * ld + off + c]` (widened to f32 for bf16 input):
/// the supervised rows of the hidden states, before cross-entropy.
/// Repeated rows are allowed: this only reads them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatherRowsPlan {
    /// Rows gathered.
    pub n_rows: u64,
    /// Columns per row.
    pub hidden: u64,
    /// Row stride of `h`.
    pub ld: u64,
    /// First column read in each row of `h`.
    pub off: u64,
    /// `n_rows * hidden`.
    pub total: u64,
}

impl GatherRowsPlan {
    /// Validate shapes and that every gathered row lies inside `h`.
    pub fn new(
        rows: &[u32],
        hidden: u64,
        (ld, off, h_len): (u64, u64, usize),
        out_len: usize,
    ) -> Result<Self, CudaError> {
        const OP: &str = "ce_gather_rows";
        if add(off, hidden, OP)? > ld {
            return Err(CudaError::invalid(
                OP,
                format!("columns {off}..{} exceed the row stride {ld}", off + hidden),
            ));
        }
        let n_rows = to_u64(rows.len());
        let total = mul(n_rows, hidden, OP)?;
        if total != to_u64(out_len) {
            return Err(CudaError::invalid(
                OP,
                format!("out has {out_len} elements, expected {n_rows} x {hidden}"),
            ));
        }
        for (i, &r) in rows.iter().enumerate() {
            let end = add(add(mul(u64::from(r), ld, OP)?, off, OP)?, hidden, OP)?;
            if hidden > 0 && end > to_u64(h_len) {
                return Err(CudaError::invalid(
                    OP,
                    format!("rows[{i}] = {r} reads up to {end}, past h's {h_len} elements"),
                ));
            }
        }
        Ok(GatherRowsPlan {
            n_rows,
            hidden,
            ld,
            off,
            total,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_cols_accepts_an_exact_fit_and_refuses_one_past() {
        // 3 rows of ld 10, window 4..10 in src; ld 6, window 0..6 in dst.
        let ok = CopyColsPlan::new(3, 6, (10, 4, 30), (6, 0, 18)).unwrap();
        assert_eq!(ok.total, 18);
        assert!(CopyColsPlan::new(3, 6, (10, 4, 29), (6, 0, 18)).is_err());
        assert!(CopyColsPlan::new(3, 6, (10, 4, 30), (6, 0, 17)).is_err());
        // A window that crosses into the next row.
        assert!(CopyColsPlan::new(3, 7, (10, 4, 40), (7, 0, 21)).is_err());
        // Empty windows are allowed and visit nothing.
        assert_eq!(
            CopyColsPlan::new(0, 6, (10, 4, 0), (6, 0, 0))
                .unwrap()
                .total,
            0
        );
    }

    #[test]
    fn copy_cols_refuses_overflowing_strides() {
        assert!(CopyColsPlan::new(u64::MAX, 1, (u64::MAX, 0, 10), (1, 0, 10)).is_err());
    }

    #[test]
    fn deliver_and_zero_check_their_ranges() {
        assert!(DeliverPlan::new((2, 10), (0, 8), 8, DeliverMode::Add).is_ok());
        assert!(DeliverPlan::new((3, 10), (0, 8), 8, DeliverMode::Add).is_err());
        assert!(DeliverPlan::new((0, 10), (1, 8), 8, DeliverMode::Copy).is_err());
        assert!(DeliverPlan::new((u64::MAX, 10), (0, 8), 2, DeliverMode::Copy).is_err());
        assert!(ZeroPlan::new(4, 6, 10).is_ok());
        assert!(ZeroPlan::new(4, 7, 10).is_err());
    }

    #[test]
    fn scatter_refuses_a_duplicate_row_naming_both_indices() {
        let err = ScatterAddRowsPlan::new(&[5, 1, 5], 2, 6, 8, 16).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("row 5 appears at pos[0] and pos[2]"),
            "{text}"
        );
    }

    #[test]
    fn scatter_refuses_out_of_range_rows_and_shape_mismatches() {
        assert!(ScatterAddRowsPlan::new(&[0, 8], 2, 4, 8, 16).is_err());
        assert!(ScatterAddRowsPlan::new(&[0, 7], 2, 5, 8, 16).is_err());
        assert!(ScatterAddRowsPlan::new(&[0, 7], 2, 4, 8, 15).is_err());
        let ok = ScatterAddRowsPlan::new(&[7, 0, 3], 2, 6, 8, 16).unwrap();
        assert_eq!((ok.n, ok.total), (3, 6));
        assert_eq!(ScatterAddRowsPlan::new(&[], 2, 0, 8, 16).unwrap().total, 0);
    }

    #[test]
    fn gather_allows_repeats_and_refuses_rows_past_h() {
        // h is [4, ld 5], columns 1..4.
        let ok = GatherRowsPlan::new(&[3, 0, 3], 3, (5, 1, 20), 9).unwrap();
        assert_eq!(ok.total, 9);
        assert!(GatherRowsPlan::new(&[4], 3, (5, 1, 20), 3).is_err());
        assert!(GatherRowsPlan::new(&[0], 5, (5, 1, 20), 5).is_err());
        assert!(GatherRowsPlan::new(&[0], 3, (5, 1, 20), 4).is_err());
    }
}
