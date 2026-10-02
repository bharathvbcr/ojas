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
    // `stride = blocks * threads` has to fit in `u32`. A product that would
    // overflow shrinks the block count; the kernel grid-strides over the rest.
    let max_fit = u64::from(u32::MAX / threads);
    let blocks_u = need.min(max).min(max_fit);
    if blocks_u == 0 {
        return Err(OjasError::OutOfRange {
            op: "grid_1d",
            detail: "block count is 0".to_string(),
        });
    }
    let blocks = u32::try_from(blocks_u).map_err(|_| OjasError::OutOfRange {
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

/// Fold `groups` workgroups into `(x, y)` with each dimension at most
/// `max_per_dim`. The kernel recovers the flat id as `y * x_count + x` and
/// guards the tail. Too many groups is [`OjasError::OutOfRange`]; nothing is
/// clamped.
pub fn fold_grid(groups: u64, max_per_dim: u32) -> Result<(u32, u32), OjasError> {
    if max_per_dim == 0 {
        return Err(OjasError::OutOfRange {
            op: "fold_grid",
            detail: "device reports a workgroup-per-dimension limit of 0".to_string(),
        });
    }
    let max = u64::from(max_per_dim);
    if groups <= max {
        return Ok((groups as u32, 1));
    }
    let rows = groups.div_ceil(max);
    if rows > max {
        return Err(OjasError::OutOfRange {
            op: "fold_grid",
            detail: format!(
                "{groups} workgroups need {rows} rows, past the device limit {max_per_dim}"
            ),
        });
    }
    Ok((max_per_dim, rows as u32))
}

/// Edge of the square output tile one GEMM workgroup owns.
pub const GEMM_TILE: u32 = 64;

/// Edge of the register-blocked tile (`gemm_*_big`), used when both output
/// sides are at least this long; below that most of its lanes would idle.
pub const GEMM_BIG_TILE: u32 = 128;

/// The output tile for an `m x n` product: [`GEMM_BIG_TILE`] when both sides
/// reach it, else [`GEMM_TILE`].
pub fn gemm_tile(m: usize, n: usize) -> u32 {
    let big = GEMM_BIG_TILE as usize;
    if m >= big && n >= big {
        GEMM_BIG_TILE
    } else {
        GEMM_TILE
    }
}

/// `(x, y)` workgroups for an `m x n` output in tiles of [`gemm_tile`]. A
/// grid past the per-dimension limit is refused, not folded: the GEMM kernel
/// reads its tile from the ids.
pub fn gemm_grid(m: usize, n: usize, max_per_dim: u32) -> Result<(u32, u32), OjasError> {
    let tile = gemm_tile(m, n) as usize;
    let gx = n.div_ceil(tile);
    let gy = m.div_ceil(tile);
    if gx == 0 || gy == 0 || gx > max_per_dim as usize || gy > max_per_dim as usize {
        return Err(OjasError::OutOfRange {
            op: "gemm_grid",
            detail: format!("gemm {m}x{n} needs a {gx}x{gy} grid; limit {max_per_dim} per axis"),
        });
    }
    Ok((gx as u32, gy as u32))
}

/// Largest head dimension the WGSL attention template accepts.
pub const ATTENTION_MAX_HEAD_DIM: u32 = 128;

/// Threads that share one row in the attention kernels.
pub const ATTENTION_PARTS: u32 = 4;

/// Lanes of one cached-attention split workgroup; a split's key count is a
/// multiple of this.
pub const CACHED_ATTENTION_LANES: usize = 64;

/// Most keys one cached-attention split holds (its shared score buffer).
pub const CACHED_ATTENTION_MAX_SPLIT: usize = 1024;

/// Workgroups the split pass aims for, so a lone decode query still keeps
/// the GPU busy.
pub const CACHED_ATTENTION_TARGET_GROUPS: usize = 512;

/// `(split length, split count)` for `rows` query rows (`B * Tq * H`)
/// against `kv_len` keys. Splits hold a multiple of
/// [`CACHED_ATTENTION_LANES`] keys, at most [`CACHED_ATTENTION_MAX_SPLIT`];
/// there are enough of them to reach about
/// [`CACHED_ATTENTION_TARGET_GROUPS`] workgroups when `rows` alone does not,
/// and every split starts below `kv_len`.
pub fn cached_attention_splits(rows: usize, kv_len: usize) -> Result<(usize, usize), OjasError> {
    if rows == 0 || kv_len == 0 {
        return Err(OjasError::OutOfRange {
            op: "cached_attention_splits",
            detail: format!("{rows} rows over {kv_len} keys"),
        });
    }
    let lanes = CACHED_ATTENTION_LANES;
    let most = kv_len.div_ceil(lanes);
    let least = kv_len.div_ceil(CACHED_ATTENTION_MAX_SPLIT);
    let want = CACHED_ATTENTION_TARGET_GROUPS.div_ceil(rows);
    let splits = want.clamp(least, most);
    let split = kv_len.div_ceil(splits).div_ceil(lanes) * lanes;
    Ok((split, kv_len.div_ceil(split)))
}

/// Shared-memory plan for the tiled attention kernels.
///
/// `padded_dim` is the head dimension rounded up to 16, 32, 64 or 128; one
/// compiled module serves every real D that pads to it. The forward (and
/// the backward's stats pass) runs `fwd_rows` query rows per workgroup over
/// key blocks of `fwd_keys`; the dQ and dK/dV passes use square blocks of
/// `bwd_block`. Each workgroup has `rows * ATTENTION_PARTS` threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionTiles {
    pub padded_dim: u32,
    pub fwd_rows: u32,
    pub fwd_keys: u32,
    pub bwd_block: u32,
    pub shared_bytes: u32,
}

impl AttentionTiles {
    /// Workgroup memory of the forward and stats kernels.
    pub fn fwd_bytes(padded_dim: u32, rows: u32, keys: u32) -> u32 {
        let sp = padded_dim / 4 + 1;
        (rows + keys) * sp * 16 + rows * (keys + 1) * 4 + 2 * rows * ATTENTION_PARTS * 4
    }

    /// Workgroup memory of the dK/dV kernel, the larger backward kernel.
    pub fn bwd_bytes(padded_dim: u32, block: u32) -> u32 {
        let sp = padded_dim / 4 + 1;
        3 * block * sp * 16 + 2 * block * (block + 1) * 4 + 2 * block * 4
    }
}

/// The padded width for `head_dim`: the smallest of 16, 32, 64, 128 that
/// holds it.
fn attention_padded_dim(head_dim: u32) -> u32 {
    let mut dp = 16u32;
    while dp < head_dim {
        dp *= 2;
    }
    dp
}

/// Pick the largest blocks whose workgroup memory fits `max_shared_bytes`:
/// forward (rows, keys) from (32, 32) down to (8, 4), backward blocks from
/// 32 down to 4. A head dimension of 0 is [`OjasError::OutOfRange`]; one
/// above [`ATTENTION_MAX_HEAD_DIM`] is [`OjasError::UnsupportedHeadDim`]; a
/// device whose shared memory cannot hold the smallest blocks is
/// [`OjasError::CapacityExceeded`].
pub fn attention_tiles(head_dim: u32, max_shared_bytes: u32) -> Result<AttentionTiles, OjasError> {
    if head_dim == 0 {
        return Err(OjasError::OutOfRange {
            op: "attention_tiles",
            detail: "head_dim is 0".to_string(),
        });
    }
    if head_dim > ATTENTION_MAX_HEAD_DIM {
        return Err(OjasError::UnsupportedHeadDim {
            head_dim,
            limit: ATTENTION_MAX_HEAD_DIM,
        });
    }
    let dp = attention_padded_dim(head_dim);
    let refuse = |requested: u32| OjasError::CapacityExceeded {
        requested: u64::from(requested),
        cap: u64::from(max_shared_bytes),
        live: 0,
    };
    const FWD: [(u32, u32); 6] = [(32, 32), (32, 16), (16, 16), (16, 8), (8, 8), (8, 4)];
    let (fwd_rows, fwd_keys) = FWD
        .into_iter()
        .find(|&(r, k)| AttentionTiles::fwd_bytes(dp, r, k) <= max_shared_bytes)
        .ok_or_else(|| refuse(AttentionTiles::fwd_bytes(dp, 8, 4)))?;
    let bwd_block = [32u32, 16, 8, 4]
        .into_iter()
        .find(|&b| AttentionTiles::bwd_bytes(dp, b) <= max_shared_bytes)
        .ok_or_else(|| refuse(AttentionTiles::bwd_bytes(dp, 4)))?;
    Ok(AttentionTiles {
        padded_dim: dp,
        fwd_rows,
        fwd_keys,
        bwd_block,
        shared_bytes: AttentionTiles::fwd_bytes(dp, fwd_rows, fwd_keys)
            .max(AttentionTiles::bwd_bytes(dp, bwd_block)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_attention_splits_cover_every_key_once() {
        for rows in [1usize, 12, 48, 511, 512, 4096, 1 << 20] {
            for kv_len in [1usize, 63, 64, 65, 1023, 1024, 1025, 4096, 100_000] {
                let (split, splits) = cached_attention_splits(rows, kv_len).unwrap();
                assert!(
                    split.is_multiple_of(CACHED_ATTENTION_LANES),
                    "{rows} {kv_len}"
                );
                assert!(
                    split <= CACHED_ATTENTION_MAX_SPLIT,
                    "{rows} {kv_len}: {split}"
                );
                assert!(split * splits >= kv_len, "{rows} {kv_len}: keys left over");
                assert!(
                    (splits - 1) * split < kv_len,
                    "{rows} {kv_len}: an empty split"
                );
            }
        }
        // A lone decode query is split; a wide prefill is not.
        assert_eq!(cached_attention_splits(12, 1024).unwrap(), (64, 16));
        assert_eq!(cached_attention_splits(4096, 1024).unwrap(), (1024, 1));
        assert!(cached_attention_splits(0, 4).is_err());
        assert!(cached_attention_splits(4, 0).is_err());
    }

    #[test]
    fn fold_grid_covers_every_group_within_the_cap() {
        assert_eq!(fold_grid(1, 65_535).unwrap(), (1, 1));
        assert_eq!(fold_grid(65_535, 65_535).unwrap(), (65_535, 1));
        assert_eq!(fold_grid(65_536, 65_535).unwrap(), (65_535, 2));
        assert_eq!(fold_grid(4, 2).unwrap(), (2, 2));
        assert!(fold_grid(5, 2).is_err());
        assert!(fold_grid(1, 0).is_err());
        for groups in [1u64, 7, 65_535, 65_536, 1 << 24, 65_535 * 65_535] {
            let (x, y) = fold_grid(groups, 65_535).unwrap();
            assert!(u64::from(x) * u64::from(y) >= groups, "{groups}");
            assert!(u64::from(x) * u64::from(y - 1) < groups, "{groups}");
        }
        assert!(fold_grid(65_535 * 65_535 + 1, 65_535).is_err());
    }

    #[test]
    fn gemm_grid_rounds_up_and_refuses_past_the_limit() {
        assert_eq!(gemm_grid(1, 1, 65_535).unwrap(), (1, 1));
        assert_eq!(gemm_grid(64, 65, 65_535).unwrap(), (2, 1));
        assert_eq!(gemm_grid(129, 128, 65_535).unwrap(), (1, 2));
        assert_eq!(gemm_grid(127, 300, 65_535).unwrap(), (5, 2));
        assert_eq!(gemm_tile(128, 127), GEMM_TILE);
        assert_eq!(gemm_tile(128, 128), GEMM_BIG_TILE);
        assert!(gemm_grid(0, 4, 65_535).is_err());
        assert!(gemm_grid(64 * 3 + 1, 1, 3).is_err());
    }

    #[test]
    fn attention_tiles_fit_shared_memory() {
        let t = attention_tiles(64, 32 * 1024).unwrap();
        assert_eq!(
            (t.padded_dim, t.fwd_rows, t.fwd_keys, t.bwd_block),
            (64, 32, 32, 16)
        );
        assert!(t.shared_bytes <= 32 * 1024);
        let t = attention_tiles(128, 32 * 1024).unwrap();
        assert_eq!(
            (t.padded_dim, t.fwd_rows, t.fwd_keys, t.bwd_block),
            (128, 32, 16, 16)
        );
        assert!(t.shared_bytes <= 32 * 1024);
        // The WebGPU minimum of 16 KiB still fits every head dim.
        for d in [1u32, 16, 17, 33, 64, 100, 128] {
            let t = attention_tiles(d, 16 * 1024).unwrap();
            assert!(t.shared_bytes <= 16 * 1024, "{d}: {t:?}");
            assert!(
                t.padded_dim >= d && t.padded_dim.is_multiple_of(16),
                "{d}: {t:?}"
            );
            assert!(t.fwd_keys >= ATTENTION_PARTS && t.bwd_block >= ATTENTION_PARTS);
        }
        assert_eq!(attention_tiles(1, 1 << 20).unwrap().padded_dim, 16);
        assert_eq!(attention_tiles(65, 1 << 20).unwrap().padded_dim, 128);
        assert!(matches!(
            attention_tiles(0, 1 << 20),
            Err(OjasError::OutOfRange { .. })
        ));
        assert!(matches!(
            attention_tiles(129, 1 << 20),
            Err(OjasError::UnsupportedHeadDim {
                head_dim: 129,
                limit: 128
            })
        ));
        assert!(matches!(
            attention_tiles(64, 100),
            Err(OjasError::CapacityExceeded { .. })
        ));
    }

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
    fn stride_overflow_shrinks_the_block_count() {
        // Pre-fix: need * threads did not fit in u32, and grid_1d returned
        // OutOfRange ("grid stride overflows u32") instead of shrinking blocks.
        let threads = 1024u32;
        let need = u64::from(u32::MAX / threads) + 1;
        let n = need * u64::from(threads);
        let grid = grid_1d(n, threads, Limits { max_grid: u32::MAX })
            .expect("a grid-stride launch must shrink blocks instead of refusing");
        assert!(grid.blocks <= u32::MAX / threads);
        assert!(u64::from(grid.blocks) < need);
        let stride = grid
            .blocks
            .checked_mul(threads)
            .expect("stride fits in u32");
        assert_eq!(grid.stride, stride);
        assert!(stride > 0);
        let step = u64::from(stride);
        assert!(
            step < n,
            "a smaller stride is only valid because the kernel strides"
        );
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
