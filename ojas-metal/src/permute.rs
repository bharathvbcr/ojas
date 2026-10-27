//! Which kernel a `permute` runs on, decided on the host from the shape and
//! the input strides alone.
//!
//! `ojas_permute` (the generic kernel) gives every output element its own
//! thread, which splits the element index over every axis with an integer
//! divide per axis. Most of SpikeTrack's permutes move long contiguous runs,
//! though: a head swap `[B, T, H, D]` to `[B, H, T, D]` copies rows of `D`
//! floats, and the window partition and merge copy rows of a whole window
//! line. Those run as vectorised row copies, with the divides paid once per
//! run (`ojas_permute_runs`). A permute whose output axis that is contiguous
//! in the input is not the last one is a batched transpose, run through a
//! shared-memory tile (`ojas_permute_tile`). Everything else, and any
//! permute the vector kernels cannot read aligned, stays on the generic
//! kernel.
//!
//! The plan looks at *canonical* axes: extent-1 axes dropped, and two
//! neighbouring output axes merged when the input stride of the first is the
//! second's stride times its extent (they are one longer contiguous axis in
//! both tensors). Merging is what turns the window permute
//! `[1,3,24,3,24,1024] -> [1,3,3,24,24,1024]` into rows of 24 x 1024 floats.

use ojas_core::MAX_PERMUTE_RANK;

/// Floats per vector load of the row copy (one `uint4`).
pub(crate) const VECTOR: u32 = 4;

/// `uint4` words a thread of `ojas_permute_runs` copies, consecutive in the
/// row: the integer divides that find the row are paid once per this many
/// 16-byte words instead of once per word.
pub(crate) const WORDS_PER_THREAD: u32 = 4;

/// A run is copied as rows only if it is a whole number of thread spans, so
/// no thread's words straddle two rows.
pub(crate) const RUN_QUANTUM: u32 = VECTOR * WORDS_PER_THREAD;

/// The shortest contiguous run, in floats, that is copied as rows: one thread
/// span. Below it a run is a few words and the row arithmetic is not
/// amortised over enough of them.
pub(crate) const MIN_RUN: u32 = RUN_QUANTUM;

/// Bytes a `uint4` access must be aligned to.
pub(crate) const VECTOR_BYTES: usize = 16;

/// Side of the transpose tile (`PERMUTE_TILE` in the kernel).
pub(crate) const TILE: u32 = 32;

/// The smallest extent of either transposed axis at which a tile is worth a
/// dispatch: under it most of a 32 x 32 tile is padding.
pub(crate) const MIN_TILE_EXTENT: u32 = 8;

/// Axes of a canonical permute: extents and the input stride of each, in
/// output order, outermost first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Axes {
    pub len: usize,
    pub extent: [u32; MAX_PERMUTE_RANK],
    pub stride: [u32; MAX_PERMUTE_RANK],
}

impl Axes {
    const fn new() -> Self {
        Axes {
            len: 0,
            extent: [1; MAX_PERMUTE_RANK],
            stride: [0; MAX_PERMUTE_RANK],
        }
    }

    fn push(&mut self, extent: u32, stride: u32) {
        self.extent[self.len] = extent;
        self.stride[self.len] = stride;
        self.len += 1;
    }

    /// The axes before the last `drop`, as their own `Axes`.
    fn head(&self, drop: usize) -> Axes {
        let mut out = Axes::new();
        for a in 0..self.len.saturating_sub(drop) {
            out.push(self.extent[a], self.stride[a]);
        }
        out
    }

    /// The product of the extents.
    pub fn count(&self) -> u64 {
        self.extent[..self.len]
            .iter()
            .map(|&e| u64::from(e))
            .product()
    }
}

/// The kernel to run and what it needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Plan {
    /// `ojas_permute`: one thread per element.
    Generic,
    /// `ojas_permute_runs`: rows of `run` contiguous floats on both sides,
    /// `outer` the axes that pick the row.
    Runs { run: u32, outer: Axes },
    /// `ojas_permute_tile`: planes of `rows x cols` (output axes `b`, then
    /// `a`), where `b` is contiguous in the input and `a` is `col_stride`
    /// apart; `outer` picks the plane.
    Transpose {
        rows: u32,
        cols: u32,
        col_stride: u32,
        outer: Axes,
    },
}

impl Plan {
    /// The pattern's name, as the tests and the bench state it.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Plan::Generic => "generic",
            Plan::Runs { .. } => "runs",
            Plan::Transpose { .. } => "transpose",
        }
    }
}

/// Extent-1 axes dropped and mergeable neighbours merged. An all-ones shape
/// gives no axes.
pub(crate) fn canonical(rank: usize, oshape: &[u32], istride: &[u32]) -> Axes {
    let mut out = Axes::new();
    for a in 0..rank.min(MAX_PERMUTE_RANK) {
        let (extent, stride) = (oshape[a], istride[a]);
        if extent == 1 {
            continue;
        }
        if out.len > 0 {
            let top = out.len - 1;
            let merged = u64::from(out.extent[top]) * u64::from(extent);
            if u64::from(out.stride[top]) == u64::from(stride) * u64::from(extent)
                && merged <= u64::from(u32::MAX)
            {
                out.extent[top] = merged as u32;
                out.stride[top] = stride;
                continue;
            }
        }
        out.push(extent, stride);
    }
    out
}

/// The plan for a permute with output shape `oshape` and input strides
/// `istride` (entries past `rank` are ignored), reading the input at byte
/// offset `x_off` and writing the output at `y_off`.
pub(crate) fn plan(
    rank: usize,
    oshape: &[u32],
    istride: &[u32],
    x_off: usize,
    y_off: usize,
) -> Plan {
    if rank == 0 || rank > MAX_PERMUTE_RANK || oshape.len() < rank || istride.len() < rank {
        return Plan::Generic;
    }
    let axes = canonical(rank, oshape, istride);
    let Some(last) = axes.len.checked_sub(1) else {
        return Plan::Generic;
    };
    if axes.stride[last] == 1 {
        let run = axes.extent[last];
        let outer = axes.head(1);
        let aligned = x_off.is_multiple_of(VECTOR_BYTES)
            && y_off.is_multiple_of(VECTOR_BYTES)
            && outer.stride[..outer.len]
                .iter()
                .all(|s| s.is_multiple_of(VECTOR));
        if run.is_multiple_of(RUN_QUANTUM) && run >= MIN_RUN && aligned {
            return Plan::Runs { run, outer };
        }
        return Plan::Generic;
    }
    if last >= 1 && axes.stride[last - 1] == 1 {
        let (rows, cols) = (axes.extent[last - 1], axes.extent[last]);
        if rows >= MIN_TILE_EXTENT && cols >= MIN_TILE_EXTENT {
            return Plan::Transpose {
                rows,
                cols,
                col_stride: axes.stride[last],
                outer: axes.head(2),
            };
        }
    }
    Plan::Generic
}

/// Threadgroups of `ojas_permute_tile` over `planes` planes of `rows x cols`
/// words, in one grid dimension. For `n` output words that is at most
/// `n / 64` (tiles of 32 x 32 over extents of at least [`MIN_TILE_EXTENT`]),
/// so below the `n / 32` threadgroups a one-dimensional elementwise dispatch
/// of the same tensor uses (`dispatch_1d`), whose grid dimension tessl caps at
/// `u32::MAX`. Apple's documentation states no lower limit per grid
/// dimension, and Ojas dispatches 1-D grids of 1.3e8 groups already.
pub(crate) fn transpose_groups(rows: u32, cols: u32, planes: u64) -> u64 {
    u64::from(rows.div_ceil(TILE)) * u64::from(cols.div_ceil(TILE)) * planes
}

/// Output words the plan's kernel writes, `None` for the generic kernel
/// (which has a thread per output word and no tail to leave).
pub(crate) fn covers(plan: &Plan) -> Option<u64> {
    match plan {
        Plan::Generic => None,
        Plan::Runs { run, outer } => Some(u64::from(*run) * outer.count()),
        Plan::Transpose {
            rows, cols, outer, ..
        } => Some(u64::from(*rows) * u64::from(*cols) * outer.count()),
    }
}

/// `Ok` when the plan's kernel writes exactly `out_words` words: the row copy
/// has no thread for a partial span and the transpose tiles whole planes, so
/// a plan built from a shape other than the output's would leave words
/// unwritten or write past them.
pub(crate) fn check_covers(plan: &Plan, out_words: usize) -> Result<(), String> {
    match covers(plan) {
        Some(words) if words != out_words as u64 => Err(format!(
            "the {} plan covers {words} words but the output holds {out_words}",
            plan.name()
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Row-major strides of `shape`.
    fn strides(shape: &[usize]) -> Vec<u32> {
        let mut s = vec![1u32; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * shape[i + 1] as u32;
        }
        s
    }

    /// The plan of `permute(shape, dims)` at aligned offsets.
    fn of(shape: &[usize], dims: &[usize]) -> Plan {
        let st = strides(shape);
        let oshape: Vec<u32> = dims.iter().map(|&a| shape[a] as u32).collect();
        let istride: Vec<u32> = dims.iter().map(|&a| st[a]).collect();
        plan(dims.len(), &oshape, &istride, 0, 0)
    }

    #[test]
    fn head_swaps_are_row_copies_of_the_head_dim() {
        for (shape, run) in [
            (vec![9, 576, 16, 64], 64),
            (vec![1, 5184, 16, 64], 64),
            (vec![1, 16, 5184, 64], 64),
            (vec![1, 5184, 8, 32], 32),
            (vec![1, 10400, 8, 32], 32),
            (vec![1, 32, 16, 64], 64),
        ] {
            match of(&shape, &[0, 2, 1, 3]) {
                Plan::Runs { run: r, outer } => {
                    assert_eq!(r, run, "{shape:?}");
                    // [B, T, H] outer axes, the size-1 batch dropped.
                    assert!(outer.len <= 3, "{shape:?}: {outer:?}");
                }
                other => panic!("{shape:?}: {other:?}"),
            }
        }
    }

    #[test]
    fn the_window_permute_merges_to_rows_of_whole_window_lines() {
        for shape in [[1, 3, 24, 3, 24, 1024], [1, 3, 3, 24, 24, 1024]] {
            match of(&shape, &[0, 1, 3, 2, 4, 5]) {
                Plan::Runs { run, outer } => {
                    assert_eq!(run, 24 * 1024, "{shape:?}");
                    assert_eq!(outer.len, 3, "{shape:?}: {outer:?}");
                }
                other => panic!("{shape:?}: {other:?}"),
            }
        }
    }

    #[test]
    fn nhwc_nchw_and_token_moves_are_transposes() {
        for (shape, dims, rows, cols) in [
            (vec![1, 1024, 72, 72], vec![0, 2, 3, 1], 5184, 1024),
            (vec![1, 72, 72, 1024], vec![0, 3, 1, 2], 1024, 5184),
            (vec![1, 256, 5184], vec![0, 2, 1], 5184, 256),
            (vec![1, 5184, 256], vec![0, 2, 1], 256, 5184),
            (vec![256, 82944], vec![1, 0], 82944, 256),
            (vec![32, 82944], vec![1, 0], 82944, 32),
            (vec![4, 64, 100], vec![0, 2, 1], 100, 64),
        ] {
            match of(&shape, &dims) {
                Plan::Transpose {
                    rows: r,
                    cols: c,
                    outer,
                    ..
                } => {
                    assert_eq!((r, c), (rows, cols), "{shape:?} {dims:?}");
                    let planes: usize = shape.iter().product::<usize>() / (rows * cols) as usize;
                    assert_eq!(outer.count(), planes as u64, "{shape:?} {dims:?}");
                }
                other => panic!("{shape:?} {dims:?}: {other:?}"),
            }
        }
    }

    #[test]
    fn everything_else_is_generic() {
        // Short runs, runs that are not whole vectors, tiny transposes, the
        // identity of a scalar, and a rank past the kernels'.
        for (shape, dims) in [
            (vec![2, 3, 4, 4], vec![0, 2, 1, 3]),    // run 4 < MIN_RUN
            (vec![2, 3, 4, 12], vec![0, 2, 1, 3]),   // run 12, not a thread span
            (vec![2, 3, 4, 8], vec![0, 2, 1, 3]),    // run 8 < MIN_RUN
            (vec![3, 255, 2, 33], vec![0, 2, 1, 3]), // run 33
            (vec![5, 3, 7], vec![2, 1, 0]),          // no contiguous axis moves
            (vec![4, 3], vec![1, 0]),                // tile extents 3 and 4
            (vec![1, 1, 1], vec![2, 0, 1]),          // all ones
        ] {
            assert_eq!(of(&shape, &dims), Plan::Generic, "{shape:?} {dims:?}");
        }
        assert_eq!(plan(0, &[], &[], 0, 0), Plan::Generic);
        assert_eq!(plan(9, &[1; 9], &[1; 9], 0, 0), Plan::Generic);
    }

    #[test]
    fn an_unaligned_view_or_stride_keeps_the_generic_kernel() {
        let shape = [9usize, 576, 16, 64];
        let st = strides(&shape);
        let (oshape, istride): (Vec<u32>, Vec<u32>) = [0usize, 2, 1, 3]
            .iter()
            .map(|&a| (shape[a] as u32, st[a]))
            .unzip();
        assert!(matches!(
            plan(4, &oshape, &istride, 16, 32),
            Plan::Runs { .. }
        ));
        // A view 24 bytes into its buffer is not 16-byte aligned.
        assert_eq!(plan(4, &oshape, &istride, 24, 0), Plan::Generic);
        assert_eq!(plan(4, &oshape, &istride, 0, 8), Plan::Generic);
        // An outer stride that is not a whole vector (a slice of a wider
        // tensor): the same run of 64 at strides 66, 6 * 66 ...
        let mut odd = istride.clone();
        odd[0] += 2;
        assert_eq!(plan(4, &oshape, &odd, 0, 0), Plan::Generic);
    }

    #[test]
    fn merging_follows_the_strides_not_the_axis_order() {
        // [2, 3, 4] by [0, 1, 2] is one axis of 24.
        let a = canonical(3, &[2, 3, 4], &[12, 4, 1]);
        assert_eq!((a.len, a.extent[0], a.stride[0]), (1, 24, 1));
        // [2, 3, 4] by [1, 0, 2]: nothing merges, since the 3 comes first and
        // its stride (4) is not the 2's stride (12) times the 2.
        let b = canonical(3, &[3, 2, 4], &[4, 12, 1]);
        assert_eq!(b.len, 3);
        // Size-1 axes vanish and the neighbours across them merge.
        let c = canonical(4, &[2, 1, 3, 4], &[12, 99, 4, 1]);
        assert_eq!((c.len, c.extent[0], c.stride[0]), (1, 24, 1));
    }

    /// What reaches which kernel among the `ojas_core::spatial` permutes:
    /// the window split and merge run as row copies when the window line
    /// (`window * C` floats) is a whole number of 16-float spans, whatever C
    /// is on its own; pixel shuffle and unshuffle stay generic below r = 8,
    /// where a transpose tile would be mostly padding.
    #[test]
    fn spatial_permutes_reach_the_kernels_the_docs_name() {
        // window_partition [B, H, W, C] -> [1, 3, w, 3, w, C] by [0,1,3,2,4,5].
        for (window, c, runs) in [
            (24usize, 1024usize, true),
            (24, 16, true),
            (24, 2, true),  // 48 floats: whole spans though C is not a multiple of 16
            (14, 8, true),  // 112 floats
            (24, 3, false), // 72 floats
            (24, 1, false), // 24 floats
            (7, 16, true),  // 112 floats
            (7, 4, false),  // 28 floats
        ] {
            let shape = [1, 3, window, 3, window, c];
            let want = if runs { "runs" } else { "generic" };
            assert_eq!(of(&shape, &[0, 1, 3, 2, 4, 5]).name(), want, "{shape:?}");
        }
        // pixel_shuffle [N, C, r, r, H, W] by [0,1,4,2,5,3]; unshuffle
        // [N, C, H, r, W, r] by [0,1,3,5,2,4].
        for r in [2usize, 4] {
            assert_eq!(
                of(&[1, 4, r, r, 64, 64], &[0, 1, 4, 2, 5, 3]),
                Plan::Generic,
                "pixel_shuffle r = {r}"
            );
            assert_eq!(
                of(&[1, 4, 64, r, 64, r], &[0, 1, 3, 5, 2, 4]),
                Plan::Generic,
                "pixel_unshuffle r = {r}"
            );
        }
    }

    /// A plan is only run over an output it tiles exactly.
    #[test]
    fn a_plan_that_does_not_tile_its_output_is_refused() {
        let runs = of(&[9, 576, 16, 64], &[0, 2, 1, 3]);
        let words = 9 * 576 * 16 * 64;
        assert_eq!(covers(&runs), Some(words as u64));
        assert!(check_covers(&runs, words).is_ok());
        assert!(check_covers(&runs, words - 16).is_err());
        assert!(check_covers(&runs, words + 16).is_err());
        let tile = of(&[2, 40, 36], &[0, 2, 1]);
        assert_eq!(covers(&tile), Some(2 * 40 * 36));
        assert!(check_covers(&tile, 2 * 40 * 36).is_ok());
        assert!(check_covers(&tile, 2 * 40 * 36 - 1).is_err());
        // The generic kernel has a thread per output word: nothing to refuse.
        assert!(check_covers(&Plan::Generic, 7).is_ok());
    }

    /// The transpose grid never exceeds a 1-D elementwise grid over the same
    /// tensor, even at the 32-bit limit on the output size and the smallest
    /// tile extents the planner accepts.
    #[test]
    fn the_transpose_grid_stays_below_an_elementwise_grid() {
        let n = u64::from(u32::MAX);
        for (rows, cols) in [
            (MIN_TILE_EXTENT, MIN_TILE_EXTENT),
            (MIN_TILE_EXTENT, 4096),
            (4096, MIN_TILE_EXTENT),
            (82944, 256),
            (33, 33),
        ] {
            let planes = n / (u64::from(rows) * u64::from(cols));
            let groups = transpose_groups(rows, cols, planes);
            assert!(groups <= n / 32, "{rows} x {cols}: {groups} groups");
            assert!(groups <= u64::from(u32::MAX));
        }
    }
}
