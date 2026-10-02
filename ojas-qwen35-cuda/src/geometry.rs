//! Launch geometry. Every grid here is a function of the problem shape only:
//! never of the device's SM count (`cuda-backend-scoping.md` §3, determinism
//! rule), so the same shape launches the same grid on every sm_90 part.
//!
//! Elementwise kernels grid-stride over a 64-bit index, so any block count
//! covers every element exactly once and the value written to an element
//! never depends on the grid. The GEMM grid maps one 16x16 thread tile to one
//! 16x16 output tile, and each output's k-order is fixed by the kernel, so
//! the grid does not change its bits either.

use crate::error::CudaError;

/// Threads per block of every elementwise (K0) kernel.
pub const THREADS_1D: u32 = 256;

/// Cap on an elementwise grid. A fixed number, not a device query: the kernels
/// grid-stride past it.
pub const MAX_BLOCKS_1D: u32 = 65_535;

/// Edge of the FFMA GEMM's square output tile and thread block.
pub const GEMM_TILE: u32 = 16;

/// `gridDim.y` limit on every CUDA architecture this crate targets.
pub const MAX_GRID_Y: u32 = 65_535;

/// `gridDim.x` limit on compute capability 3.0 and later.
pub const MAX_GRID_X: u32 = 2_147_483_647;

/// A kernel launch shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    /// Blocks in x, y, z.
    pub grid: (u32, u32, u32),
    /// Threads per block in x, y, z.
    pub block: (u32, u32, u32),
}

/// The grid-stride launch for `n` elements; `None` when there is nothing to do.
pub fn grid_1d(n: u64) -> Option<Launch> {
    if n == 0 {
        return None;
    }
    let need = n.div_ceil(u64::from(THREADS_1D));
    let blocks = u32::try_from(need.min(u64::from(MAX_BLOCKS_1D))).unwrap_or(MAX_BLOCKS_1D);
    Some(Launch {
        grid: (blocks, 1, 1),
        block: (THREADS_1D, 1, 1),
    })
}

/// The FFMA GEMM's launch for an `m x n` output: x over columns, y over rows.
pub fn gemm_grid(m: usize, n: usize) -> Result<Launch, CudaError> {
    let tile = GEMM_TILE as usize;
    let gx = n.div_ceil(tile);
    let gy = m.div_ceil(tile);
    if gx == 0 || gy == 0 {
        return Err(CudaError::invalid(
            "gemm_grid",
            format!("an {m}x{n} output has no tiles"),
        ));
    }
    let gx = u32::try_from(gx)
        .ok()
        .filter(|&g| g <= MAX_GRID_X)
        .ok_or_else(|| {
            CudaError::invalid(
                "gemm_grid",
                format!("{n} columns need more than {MAX_GRID_X} blocks"),
            )
        })?;
    let gy = u32::try_from(gy)
        .ok()
        .filter(|&g| g <= MAX_GRID_Y)
        .ok_or_else(|| {
            CudaError::invalid(
                "gemm_grid",
                format!(
                    "{m} rows need {} blocks in y, past the limit {MAX_GRID_Y}",
                    m.div_ceil(tile)
                ),
            )
        })?;
    Ok(Launch {
        grid: (gx, gy, 1),
        block: (GEMM_TILE, GEMM_TILE, 1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulate the kernels' grid-stride loop and count visits per index.
    fn visits(n: u64, launch: Launch) -> Vec<u32> {
        let stride = u64::from(launch.grid.0) * u64::from(launch.block.0);
        let mut seen = vec![0u32; usize::try_from(n).unwrap()];
        for block in 0..u64::from(launch.grid.0) {
            for thread in 0..u64::from(launch.block.0) {
                let mut i = block * u64::from(launch.block.0) + thread;
                while i < n {
                    seen[usize::try_from(i).unwrap()] += 1;
                    i += stride;
                }
            }
        }
        seen
    }

    #[test]
    fn empty_work_launches_nothing() {
        assert_eq!(grid_1d(0), None);
    }

    #[test]
    fn grid_stride_visits_every_element_exactly_once() {
        for n in [1u64, 255, 256, 257, 1000, 65_535 * 256 + 3] {
            let launch = grid_1d(n).unwrap();
            assert!(launch.grid.0 <= MAX_BLOCKS_1D);
            if n < 2_000_000 {
                assert!(visits(n, launch).iter().all(|&v| v == 1), "n = {n}");
            }
        }
    }

    #[test]
    fn huge_elementwise_work_is_capped_not_overflowed() {
        let launch = grid_1d(u64::MAX).unwrap();
        assert_eq!(launch.grid.0, MAX_BLOCKS_1D);
    }

    #[test]
    fn gemm_grid_tiles_the_output_and_refuses_too_many_rows() {
        let l = gemm_grid(130, 70).unwrap();
        assert_eq!(l.grid, (5, 9, 1));
        assert_eq!(l.block, (16, 16, 1));
        // 248,320-row vocabulary GEMMs fit.
        assert!(gemm_grid(248_320, 2048).is_ok());
        assert!(gemm_grid(16 * 65_535 + 1, 1).is_err());
        assert!(gemm_grid(0, 4).is_err());
    }
}
