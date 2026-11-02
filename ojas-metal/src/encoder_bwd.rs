//! Launch plans of the encoder backward kernels (`kernels/encoder_bwd.metal`):
//! LayerNorm, channel LayerNorm, softmax, the broadcast gradients and the
//! exact GELU's derivative. Host-side and free of Metal types, so the command
//! protocol (`link::Cmd`) can name them on every target.
//!
//! Every plan is a function of the call's extents alone. That is what makes a
//! result repeat bit for bit: the number of splits a column sum is cut into,
//! and the threads a row sum meets in, never depend on the machine or the run.

use ojas_core::{BroadcastDims, MAX_BROADCAST_RANK};

use crate::link::{metal_err, Res};

/// Every kernel `kernels/encoder_bwd.metal` defines; the device thread
/// builds each pipeline at start-up.
pub(crate) const KERNELS: &[&str] = &[
    "ojas_ln_bwd_rows",
    "ojas_ln_bwd_cols_part",
    "ojas_cols_fin4",
    "ojas_cln_bwd_pos",
    "ojas_cln_bwd_ch_part",
    "ojas_gelu_erf_bwd",
    "ojas_softmax_bwd_rows",
    "ojas_bcast_grad_part",
    "ojas_bcast_grad_fin",
];

/// Threads of a threadgroup that reduces one row of `dim` values: the
/// smallest power of two that covers the row, between one simdgroup and
/// [`MAX_ROW_THREADS`]. A power of two is what the kernels' fixed reduction
/// tree needs.
pub(crate) fn row_threads(dim: u32) -> u32 {
    dim.max(1)
        .checked_next_power_of_two()
        .unwrap_or(MAX_ROW_THREADS)
        .clamp(32, MAX_ROW_THREADS)
}

/// `EB_MAX_THREADS` in the Metal source.
pub(crate) const MAX_ROW_THREADS: u32 = 256;

/// Rows a column-sum split takes at least, and most splits one sum is cut
/// into (a LayerNorm's `dw` and `db`).
const ROWS_PER_SPLIT: u32 = 16;
/// Positions a channel-sum split takes at least (a channel LayerNorm's
/// `dw[c]` and `db[c]`, which also spread each split over a threadgroup).
const POSITIONS_PER_SPLIT: u32 = 2048;
const MAX_SPLITS: u32 = 64;

/// `(splits, chunk)` for `items` terms of a sum cut into pieces of at least
/// `per_split`: every split is non-empty, and `splits * chunk` stays within
/// 32 bits for the kernels' index arithmetic.
fn split_plan(items: u32, per_split: u32, max: u32) -> (u32, u32) {
    let wanted = items.div_ceil(per_split.max(1)).clamp(1, max);
    let chunk = items.div_ceil(wanted).max(1);
    (items.div_ceil(chunk).max(1), chunk)
}

/// A LayerNorm backward over `rows` rows of `dim`: the column pass cuts the
/// rows into `splits` runs of `chunk`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LnBwdGeom {
    pub rows: u32,
    pub dim: u32,
    pub splits: u32,
    pub chunk: u32,
}

impl LnBwdGeom {
    pub(crate) fn new(rows: u32, dim: u32) -> Self {
        let (splits, chunk) = split_plan(rows, ROWS_PER_SPLIT, MAX_SPLITS);
        Self {
            rows,
            dim,
            splits,
            chunk,
        }
    }

    /// f32 values of scratch: the per-row `(mean.hi, mean.lo, rstd, pad)` and
    /// the column pass's partials (four values per split and column).
    pub(crate) fn scratch(&self) -> Option<usize> {
        let stats = (self.rows as usize).checked_mul(4)?;
        let parts = (self.splits as usize)
            .checked_mul(self.dim as usize)?
            .checked_mul(4)?;
        stats.checked_add(parts)
    }
}

/// A channel LayerNorm backward over `[batch, channels, spatial]`:
/// `positions = batch * spatial`, and the channel pass cuts them into
/// `splits` runs of `chunk`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ClnBwdGeom {
    pub positions: u32,
    pub channels: u32,
    pub spatial: u32,
    pub splits: u32,
    pub chunk: u32,
}

impl ClnBwdGeom {
    pub(crate) fn new(positions: u32, channels: u32, spatial: u32) -> Self {
        let (splits, chunk) = split_plan(positions, POSITIONS_PER_SPLIT, MAX_SPLITS);
        Self {
            positions,
            channels,
            spatial,
            splits,
            chunk,
        }
    }

    pub(crate) fn scratch(&self) -> Option<usize> {
        let stats = (self.positions as usize).checked_mul(4)?;
        let parts = (self.splits as usize)
            .checked_mul(self.channels as usize)?
            .checked_mul(4)?;
        stats.checked_add(parts)
    }
}

/// Words of a [`BcastGradGeom`] as the kernel reads them.
pub(crate) const BCAST_GEOM_WORDS: usize = 56;
/// Most terms one thread of the broadcast-gradient part kernel adds before
/// another split is worth a thread, and the most partials the sum keeps.
const TERMS_PER_SPLIT: u32 = 32;
const TARGET_THREADS: u64 = 1 << 16;
const MAX_BCAST_SPLITS: u32 = 1024;
const MAX_PARTIALS: u64 = 1 << 22;

/// One operand's gradient of a broadcast add or multiply: `elems` operand
/// elements, each the sum of `reduce` output terms, cut into `splits` runs of
/// `chunk` terms.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BcastGradGeom {
    pub elems: u32,
    pub reduce: u32,
    pub splits: u32,
    pub chunk: u32,
    /// The kernel's `geom` words.
    pub words: [u32; BCAST_GEOM_WORDS],
}

impl BcastGradGeom {
    /// The gradient of operand `a` (`of_a`) or `b` of a broadcast with `dims`,
    /// for a multiply (`mul`) or an add. The operand's axes are the output
    /// axes where its stride is not 0 (kept, one gradient element per
    /// coordinate); the others longer than 1 are summed over (reduced).
    pub(crate) fn new(op: &'static str, dims: &BroadcastDims, of_a: bool, mul: bool) -> Res<Self> {
        let (own, mate) = if of_a {
            (&dims.a_strides, &dims.b_strides)
        } else {
            (&dims.b_strides, &dims.a_strides)
        };
        let rank = dims.out_shape.len();
        if rank > MAX_BROADCAST_RANK {
            return Err(metal_err(format!("{op}: broadcast rank {rank} too large")));
        }
        let mut out_strides = vec![0usize; rank];
        let mut run = 1usize;
        for axis in (0..rank).rev() {
            out_strides[axis] = run;
            run = run
                .checked_mul(dims.out_shape[axis])
                .ok_or_else(|| metal_err(format!("{op}: extent overflows")))?;
        }
        let mut words = [0u32; BCAST_GEOM_WORDS];
        let word = |v: usize| -> Res<u32> {
            u32::try_from(v).map_err(|_| ojas_core::OjasError::Unsupported {
                op,
                detail: format!("{v} exceeds the kernels' 32-bit indexing"),
            })
        };
        let (mut kept, mut reduced) = (0usize, 0usize);
        let (mut elems, mut reduce) = (1u64, 1u64);
        for (axis, &extent) in dims.out_shape.iter().enumerate() {
            if own[axis] != 0 {
                words[8 + kept] = word(extent)?;
                words[16 + kept] = word(out_strides[axis])?;
                words[24 + kept] = word(mate[axis])?;
                kept += 1;
                elems = elems.saturating_mul(extent as u64);
            } else if extent > 1 {
                words[32 + reduced] = word(extent)?;
                words[40 + reduced] = word(out_strides[axis])?;
                words[48 + reduced] = word(mate[axis])?;
                reduced += 1;
                reduce = reduce.saturating_mul(extent as u64);
            }
        }
        let elems = u32::try_from(elems).map_err(|_| metal_err(format!("{op}: gradient size")))?;
        let reduce = u32::try_from(reduce).map_err(|_| metal_err(format!("{op}: reduction")))?;
        // Enough threads to fill the device, never so many partials that the
        // scratch rivals the gradient, and every split non-empty.
        let by_terms = reduce.div_ceil(TERMS_PER_SPLIT);
        let by_threads =
            u32::try_from(TARGET_THREADS.div_ceil(u64::from(elems).max(1))).unwrap_or(u32::MAX);
        let by_scratch = u32::try_from(MAX_PARTIALS / u64::from(elems).max(1)).unwrap_or(u32::MAX);
        let wanted = by_terms
            .min(by_threads)
            .min(by_scratch)
            .clamp(1, MAX_BCAST_SPLITS);
        let chunk = reduce.div_ceil(wanted).max(1);
        let splits = reduce.div_ceil(chunk).max(1);
        words[0] = elems;
        words[1] = reduce;
        words[2] = splits;
        words[3] = chunk;
        words[4] = word(kept)?;
        words[5] = word(reduced)?;
        words[6] = u32::from(mul);
        Ok(Self {
            elems,
            reduce,
            splits,
            chunk,
            words,
        })
    }

    /// f32 values of partial-sum scratch (two per split and element); one
    /// split writes the gradient directly.
    pub(crate) fn scratch(&self) -> Option<usize> {
        if self.splits <= 1 {
            return Some(0);
        }
        (self.splits as usize)
            .checked_mul(self.elems as usize)?
            .checked_mul(2)
    }

    /// Whether every index the kernel forms stays inside `g_len` output
    /// gradient elements and (for a multiply) `other_len` elements of the
    /// other operand: the device thread's own check of what the host planned,
    /// made before it binds anything.
    pub(crate) fn fits(&self, g_len: usize, other_len: Option<usize>) -> bool {
        let w = &self.words;
        let (kept, reduced) = (w[4] as usize, w[5] as usize);
        if kept > 8 || reduced > 8 || self.splits == 0 || self.chunk == 0 {
            return false;
        }
        let mut elems = 1u64;
        let mut reach = [0u64; 2];
        for a in 0..kept {
            elems = elems.saturating_mul(u64::from(w[8 + a]));
            reach[0] += u64::from(w[8 + a].saturating_sub(1)) * u64::from(w[16 + a]);
            reach[1] += u64::from(w[8 + a].saturating_sub(1)) * u64::from(w[24 + a]);
        }
        let mut reduce = 1u64;
        for a in 0..reduced {
            reduce = reduce.saturating_mul(u64::from(w[32 + a]));
            reach[0] += u64::from(w[32 + a].saturating_sub(1)) * u64::from(w[40 + a]);
            reach[1] += u64::from(w[32 + a].saturating_sub(1)) * u64::from(w[48 + a]);
        }
        elems == u64::from(self.elems)
            && reduce == u64::from(self.reduce)
            && elems.saturating_mul(reduce) == g_len as u64
            && reach[0] < g_len as u64
            && other_len.is_none_or(|n| reach[1] < n as u64)
            && u64::from(self.splits) * u64::from(self.chunk) >= reduce
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_threads_is_a_power_of_two_between_a_simdgroup_and_the_cap() {
        assert_eq!(row_threads(1), 32);
        assert_eq!(row_threads(32), 32);
        assert_eq!(row_threads(33), 64);
        assert_eq!(row_threads(196), 256);
        assert_eq!(row_threads(1 << 30), 256);
        assert_eq!(row_threads(u32::MAX), 256);
    }

    #[test]
    fn a_split_plan_has_no_empty_split_and_covers_every_item() {
        for items in [1u32, 2, 15, 16, 17, 1000, 5184, 82_944, 4_000_000] {
            for per in [16u32, 2048] {
                let (splits, chunk) = split_plan(items, per, MAX_SPLITS);
                assert!((1..=MAX_SPLITS).contains(&splits), "{items} {per}");
                assert!(u64::from(splits) * u64::from(chunk) >= u64::from(items));
                assert!(u64::from(splits - 1) * u64::from(chunk) < u64::from(items));
            }
        }
    }

    #[test]
    fn a_bias_gradient_reduces_every_row_and_keeps_each_column() {
        let dims = ojas_core::broadcast_shapes("test", &[5184, 4096], &[4096]).unwrap();
        let g = BcastGradGeom::new("test", &dims, false, false).unwrap();
        assert_eq!((g.elems, g.reduce), (4096, 5184));
        assert!(g.splits > 1 && g.splits <= MAX_BCAST_SPLITS);
        assert_eq!(g.words[4], 1, "one kept axis");
        assert_eq!(g.words[5], 1, "one reduced axis");
        // The stretched operand's own gradient reduces nothing but copies.
        let a = BcastGradGeom::new("test", &dims, true, false).unwrap();
        assert_eq!((a.elems, a.reduce, a.splits), (5184 * 4096, 1, 1));
        assert_eq!(a.scratch(), Some(0));
    }
}
