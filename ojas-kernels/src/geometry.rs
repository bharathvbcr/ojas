//! Overflow-checked launch geometry.
//!
//! The host grid is capped by device limits. The kernel uses a grid-stride
//! loop, so an index is `start + stride * step` in `u64` and never wraps a
//! `u32` addition of the element count.

use ojas_core::OjasError;

/// Device limits supplied by the caller. This crate does not query a runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_grid: u32,
}

/// A 1D grid-stride launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub blocks: u32,
    pub threads: u32,
    /// `blocks * threads`, the stride of the loop.
    pub stride: u32,
}

pub fn grid_1d(n: u64, threads: u32, limits: Limits) -> Result<Grid, OjasError> {
    if threads == 0 || limits.max_grid == 0 {
        return Err(OjasError::OutOfRange {
            op: "grid_1d",
            detail: "thread or grid limit is 0".to_string(),
        });
    }
    if n == 0 {
        return Ok(Grid {
            blocks: 0,
            threads,
            stride: 0,
        });
    }
    let threads_u = u64::from(threads);
    let need = n.div_ceil(threads_u);
    let max = u64::from(limits.max_grid);
    let blocks = u32::try_from(need.min(max)).map_err(|_| OjasError::OutOfRange {
        op: "grid_1d",
        detail: "block count does not fit in u32".to_string(),
    })?;
    let stride = blocks
        .checked_mul(threads)
        .ok_or_else(|| OjasError::OutOfRange {
            op: "grid_1d",
            detail: "grid stride overflows u32".to_string(),
        })?;
    if u64::from(stride) == 0 {
        return Err(OjasError::OutOfRange {
            op: "grid_1d",
            detail: "grid stride is 0".to_string(),
        });
    }
    Ok(Grid {
        blocks,
        threads,
        stride,
    })
}

/// Every element index a grid-stride loop would visit, in `u64`.
pub fn cover_1d(n: u64, grid: Grid) -> Result<Vec<u64>, OjasError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let stride = u64::from(grid.stride);
    let mut seen = Vec::new();
    let mut start = 0u64;
    while start < stride && start < n {
        let mut index = start;
        while index < n {
            seen.push(index);
            index = index
                .checked_add(stride)
                .ok_or_else(|| OjasError::OutOfRange {
                    op: "cover_1d",
                    detail: "index overflows u64".to_string(),
                })?;
        }
        start += 1;
    }
    seen.sort_unstable();
    Ok(seen)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u32_boundary_grid_does_not_exceed_the_device_limit() {
        let limits = Limits { max_grid: 65_535 };
        let grid = grid_1d(u64::from(u32::MAX), 256, limits).unwrap();
        assert!(grid.blocks <= limits.max_grid);
        assert!(grid.stride > 0);
        let n = 10_000u64;
        let strided = grid_1d(n, 64, Limits { max_grid: 3 }).unwrap();
        assert!(strided.blocks <= 3);
        let covered = cover_1d(n, strided).unwrap();
        assert_eq!(covered.len() as u64, n);
        assert_eq!(covered.first().copied(), Some(0));
        assert_eq!(covered.last().copied(), Some(n - 1));
    }

    #[test]
    fn zero_and_small_counts() {
        let limits = Limits { max_grid: 8 };
        let empty = grid_1d(0, 32, limits).unwrap();
        assert_eq!(empty.blocks, 0);
        assert!(grid_1d(10, 0, limits).is_err());
        assert!(grid_1d(10, 32, Limits { max_grid: 0 }).is_err());
        let small = grid_1d(10, 32, limits).unwrap();
        assert_eq!(small.blocks, 1);
        assert_eq!(cover_1d(10, small).unwrap(), (0..10).collect::<Vec<_>>());
    }
}
