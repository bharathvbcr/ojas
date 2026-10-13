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
/// guards the tail: the grid is a rectangle of `x * y >= groups`, so the
/// workgroups past `groups` are spare, and a kernel must `return` for them
/// before it reads or writes any buffer (an out-of-range write is clamped
/// onto the last real entry under wgpu's Restrict policy, and races with its
/// owner). Every WGSL entry point is classified in `KERNEL_GRIDS` (test-only), and a
/// unit test of this module fails on a folded-grid kernel that is missing
/// there or whose guard does not come before its first buffer access. Too
/// many groups is [`OjasError::OutOfRange`]; nothing is clamped.
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

/// How one WGSL entry point's dispatch grid relates to its real work.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridKind {
    /// Dispatched on a [`fold_grid`] grid (the backend's `groups()` and
    /// `lanes()`): the kernel recovers a flat id with `flat_group` and the
    /// string names the real bound that id is guarded against.
    Folded(&'static str),
    /// A 2-D grid whose axes are the real extents, each checked against the
    /// per-axis limit and never folded; the string says why no workgroup is
    /// spare or how a rounded-up tile is guarded.
    Exact(&'static str),
    /// One workgroup.
    Single(&'static str),
}

/// Every WGSL `@compute` entry point and its grid. A unit test of this
/// module reads the module sources and fails when an entry point is absent
/// here, when a row names no entry point, when a kernel that reads the flat
/// group id is not [`GridKind::Folded`], or when a folded kernel's first
/// `if (... >= ...) { return; }` guard does not precede its first access to a
/// storage buffer (the fault words included), directly or through a helper.
#[cfg(test)]
pub const KERNEL_GRIDS: &[(&str, GridKind)] = &[
    // pointwise.wgsl: one lane per element.
    (
        "round_bf16",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    ("silu_fwd", GridKind::Folded("lane i >= word 0 (elements)")),
    ("silu_bwd", GridKind::Folded("lane i >= word 0 (elements)")),
    ("mul_fwd", GridKind::Folded("lane i >= word 0 (elements)")),
    ("mul_bwd", GridKind::Folded("lane i >= word 0 (elements)")),
    ("add_fwd", GridKind::Folded("lane i >= word 0 (elements)")),
    (
        "add_inplace",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    (
        "check_finite",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    ("scale", GridKind::Folded("lane i >= word 0 (elements)")),
    ("vr_fwd", GridKind::Folded("lane i >= word 0 (elements)")),
    ("vr_bwd", GridKind::Folded("lane i >= word 0 (elements)")),
    ("gate_fwd", GridKind::Folded("lane i >= word 0 (elements)")),
    (
        "gate_fwd_save",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    (
        "gate_bwd_saved",
        GridKind::Folded("lane rh >= word 0 (rows * heads)"),
    ),
    (
        "gate_bwd",
        GridKind::Folded("lane rh >= word 0 (rows * heads)"),
    ),
    ("rope", GridKind::Folded("lane i >= word 0 (rotated pairs)")),
    // loss.wgsl: one lane per element or one workgroup per row.
    (
        "embed_fwd",
        GridKind::Folded("lane i >= word 0 (tokens * dim)"),
    ),
    (
        "embed_bwd",
        GridKind::Folded("lane i >= word 0 (distinct ids * dim)"),
    ),
    ("ce_fwd", GridKind::Folded("workgroup row >= word 0 (rows)")),
    ("ce_bwd", GridKind::Folded("workgroup row >= word 0 (rows)")),
    (
        "lce_stats",
        GridKind::Folded("workgroup row >= word 0 (tile rows)"),
    ),
    ("lce_finish", GridKind::Folded("lane g >= word 0 (rows)")),
    (
        "lce_grad",
        GridKind::Folded("lane i >= word 0 (tile elements)"),
    ),
    // layout.wgsl.
    ("permute", GridKind::Folded("lane i >= word 0 (elements)")),
    ("kv_write", GridKind::Folded("lane i >= word 0 (elements)")),
    (
        "widen_bf16",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    (
        "narrow_bf16",
        GridKind::Folded("lane 2 * j >= word 0 (elements)"),
    ),
    // norm.wgsl: `256 / width` rows per workgroup.
    (
        "rms_fwd",
        GridKind::Folded("first row of the workgroup >= word 0 (rows)"),
    ),
    (
        "rms_bwd",
        GridKind::Folded("first row of the workgroup >= word 0 (rows)"),
    ),
    // optim.wgsl: one lane per element.
    (
        "adam_check",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    (
        "adam_apply",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    (
        "muon_momentum",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    (
        "muon_normalize",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    (
        "muon_normalize_bf16",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    (
        "muon_round",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    (
        "muon_lincomb_bf16",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    (
        "muon_lincomb",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    (
        "muon_apply",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    (
        "muon_commit",
        GridKind::Folded("lane i >= word 0 (elements)"),
    ),
    // reduce.wgsl.
    (
        "sum_partial",
        GridKind::Folded("group >= ceil(word 0 / CHUNK) (chunks)"),
    ),
    ("col_finish", GridKind::Folded("lane c >= word 1 (columns)")),
    (
        "argmax_rows",
        GridKind::Folded("workgroup row >= word 0 (rows)"),
    ),
    (
        "topk_rows",
        GridKind::Folded("workgroup row >= word 0 (rows)"),
    ),
    ("sum_finish", GridKind::Single("one workgroup, (1, 1, 1)")),
    (
        "col_partial",
        GridKind::Exact("grid (cols / 256, row chunks) are the real axes; lane c >= cols returns"),
    ),
    // clip.wgsl.
    (
        "norm_partial",
        GridKind::Folded("chunk c >= word 3 (the dispatch's chunk end)"),
    ),
    (
        "scale_multi",
        GridKind::Folded("chunk c >= word 4 (the dispatch's chunk end)"),
    ),
    ("norm_finish", GridKind::Single("one workgroup, (1, 1, 1)")),
    // attention.wgsl, cached_attention.wgsl.
    (
        "attn_bwd_dr",
        GridKind::Folded("lane row >= word 2 (query rows)"),
    ),
    (
        "attn_fwd",
        GridKind::Exact(
            "grid (ceil(time / rows), bh), each axis checked; query qi >= time writes nothing",
        ),
    ),
    (
        "attn_bwd_dq",
        GridKind::Exact(
            "grid (ceil(time / rows), bh), each axis checked; query rows past time are masked",
        ),
    ),
    (
        "attn_bwd_dkv",
        GridKind::Exact(
            "grid (ceil(time / rows), bh / rep), each axis checked; key rows past time are masked",
        ),
    ),
    (
        "cattn_split",
        GridKind::Exact(
            "grid (splits, row-heads in dispatches of at most max, batch), each axis exact",
        ),
    ),
    (
        "cattn_merge",
        GridKind::Exact("grid (row-heads in dispatches of at most max, batch), each axis exact"),
    ),
    // gemm.wgsl: gemm_grid is (ceil(n / tile), ceil(m / tile)), each axis
    // checked; edge tiles guard their rows and columns.
    (
        "gemm_nt",
        GridKind::Exact("gemm_grid axes are the real tile counts"),
    ),
    (
        "gemm_nn",
        GridKind::Exact("gemm_grid axes are the real tile counts"),
    ),
    (
        "gemm_tn",
        GridKind::Exact("gemm_grid axes are the real tile counts"),
    ),
    (
        "gemm_nt_big",
        GridKind::Exact("gemm_grid axes are the real tile counts"),
    ),
    (
        "gemm_nn_big",
        GridKind::Exact("gemm_grid axes are the real tile counts"),
    ),
    (
        "gemm_tn_big",
        GridKind::Exact("gemm_grid axes are the real tile counts"),
    ),
    // fault.wgsl.
    ("fault_hold", GridKind::Single("one workgroup, (1, 1, 1)")),
    (
        "fault_release",
        GridKind::Single("one workgroup, (1, 1, 1)"),
    ),
];

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
///
/// 256 is the widest head that still fits a WebGPU workgroup: the merge
/// kernel is one invocation per dimension, and the guaranteed maximum
/// invocations per workgroup is 256. A padded width of 512 does not fit the
/// smallest tiles in 16 KiB of shared memory either.
pub const ATTENTION_MAX_HEAD_DIM: u32 = 256;

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
/// `padded_dim` is the head dimension rounded up to 16, 32, 64, 128 or 256; one
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

/// The padded width for `head_dim`: the smallest of 16, 32, 64, 128, 256 that
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
    use crate::source::{wgsl_module, WgslModule};
    use std::collections::{BTreeMap, BTreeSet};

    struct Func {
        name: String,
        body: String,
        entry: bool,
    }

    /// Source with `//` comments removed.
    fn uncommented(src: &str) -> String {
        src.lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn is_ident(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_'
    }

    /// Byte offsets where `needle` occurs as a whole identifier prefix, that
    /// is, not preceded by an identifier character.
    fn occurrences(hay: &str, needle: &str) -> Vec<usize> {
        hay.match_indices(needle)
            .map(|(at, _)| at)
            .filter(|&at| !hay[..at].chars().next_back().is_some_and(is_ident))
            .collect()
    }

    /// The text from `open` (an opening delimiter) to its match.
    fn balanced(src: &str, open: usize) -> &str {
        let (o, c) = match src.as_bytes()[open] {
            b'(' => (b'(', b')'),
            _ => (b'{', b'}'),
        };
        let mut depth = 0usize;
        for (k, b) in src.bytes().enumerate().skip(open) {
            if b == o {
                depth += 1;
            } else if b == c {
                depth -= 1;
                if depth == 0 {
                    return &src[open..=k];
                }
            }
        }
        panic!("unbalanced delimiter at {open}");
    }

    fn functions(src: &str) -> Vec<Func> {
        let src = uncommented(src);
        let mut out = Vec::new();
        for at in occurrences(&src, "fn ") {
            if !src[..at].ends_with('\n') {
                continue;
            }
            let name: String = src[at + 3..].chars().take_while(|&c| is_ident(c)).collect();
            let params = at + 3 + src[at + 3..].find('(').expect("fn parameters");
            let after = params + balanced(&src, params).len();
            let open = after + src[after..].find('{').expect("fn body");
            let body = balanced(&src, open).to_string();
            let entry = src[..at]
                .lines()
                .next_back()
                .is_some_and(|l| l.trim_start().starts_with("@compute"));
            out.push(Func { name, body, entry });
        }
        out
    }

    fn storage_names(src: &str) -> Vec<String> {
        uncommented(src)
            .lines()
            .filter_map(|l| {
                let rest = l.split("var<storage").nth(1)?;
                let rest = rest.split('>').nth(1)?;
                let name: String = rest.trim().chars().take_while(|&c| is_ident(c)).collect();
                Some(name)
            })
            .collect()
    }

    /// Offset of the first access to a storage buffer in `body`: an index of
    /// one, its address taken, or a call to a helper that does either.
    fn first_touch(body: &str, buffers: &[String], touching: &BTreeSet<String>) -> Option<usize> {
        let mut at = Vec::new();
        for b in buffers {
            at.extend(occurrences(body, &format!("{b}[")));
            at.extend(occurrences(body, &format!("&{b}")).iter().map(|&k| k + 1));
        }
        for h in touching {
            at.extend(occurrences(body, &format!("{h}(")));
        }
        at.into_iter().min()
    }

    /// For every entry point in `src`: whether it derives a folded flat group
    /// id, and, when it does, `Ok` if its first `if (... >= ...) { return; }`
    /// precedes its first storage access, else why not.
    fn audit(src: &str) -> BTreeMap<String, (bool, Result<(), String>)> {
        let funcs = functions(src);
        let buffers = storage_names(src);
        let helpers: Vec<&Func> = funcs.iter().filter(|f| !f.entry).collect();
        let mut touching = BTreeSet::new();
        let mut flat: BTreeSet<String> = BTreeSet::new();
        loop {
            let mut grew = false;
            for h in &helpers {
                if !touching.contains(&h.name)
                    && first_touch(&h.body, &buffers, &touching).is_some()
                {
                    grew |= touching.insert(h.name.clone());
                }
                let reads_flat = !occurrences(&h.body, "flat_group(").is_empty()
                    || flat
                        .iter()
                        .any(|f| !occurrences(&h.body, &format!("{f}(")).is_empty());
                if reads_flat && !flat.contains(&h.name) {
                    grew |= flat.insert(h.name.clone());
                }
            }
            if !grew {
                break;
            }
        }
        flat.insert("flat_group".to_string());
        funcs
            .iter()
            .filter(|f| f.entry)
            .map(|f| {
                let folded = flat
                    .iter()
                    .any(|h| !occurrences(&f.body, &format!("{h}(")).is_empty());
                let guard = if folded {
                    guard_before_access(&f.body, &buffers, &touching)
                } else {
                    Ok(())
                };
                (f.name.clone(), (folded, guard))
            })
            .collect()
    }

    fn guard_before_access(
        body: &str,
        buffers: &[String],
        touching: &BTreeSet<String>,
    ) -> Result<(), String> {
        let Some(ret) = body.find("return;") else {
            return Err("no early return: the tail is not guarded".to_string());
        };
        let Some(start) = body[..ret].rfind("if (") else {
            return Err("the first return is not under an if".to_string());
        };
        if !body[start..ret].contains(">=") {
            return Err("the first return is not a `>=` bound".to_string());
        }
        match first_touch(body, buffers, touching) {
            Some(touch) if touch < ret => Err(format!(
                "a storage buffer is accessed before the tail guard: `{}`",
                body[touch..].chars().take(24).collect::<String>().trim()
            )),
            _ => Ok(()),
        }
    }

    fn every_module() -> Vec<(String, String)> {
        let tiles = attention_tiles(64, 32 * 1024).unwrap();
        [
            WgslModule::Gemm,
            WgslModule::Pointwise,
            WgslModule::Reduce,
            WgslModule::Norm,
            WgslModule::Loss,
            WgslModule::Optim,
            WgslModule::Layout,
            WgslModule::Fault,
            WgslModule::Attention(tiles),
            WgslModule::CachedAttention,
            WgslModule::Clip(1),
        ]
        .into_iter()
        .map(|m| (format!("{m:?}"), wgsl_module(m).unwrap()))
        .collect()
    }

    /// The audit of the real kernels: every entry point is classified in
    /// `KERNEL_GRIDS`, every kernel that reads a folded flat id is `Folded`
    /// and returns for its spare workgroups before it touches a buffer.
    #[test]
    fn every_folded_grid_kernel_guards_its_tail_before_any_buffer_access() {
        let mut found = BTreeMap::new();
        for (module, src) in every_module() {
            for (entry, audited) in audit(&src) {
                assert!(
                    found
                        .insert(entry.clone(), (module.clone(), audited))
                        .is_none(),
                    "entry point {entry} is defined twice"
                );
            }
        }
        let mut table = BTreeMap::new();
        for (entry, kind) in KERNEL_GRIDS {
            assert!(
                table.insert(*entry, *kind).is_none(),
                "{entry} is listed twice"
            );
        }
        let mut problems = Vec::new();
        for (entry, (module, (folded, guard))) in &found {
            match table.get(entry.as_str()) {
                None => problems.push(format!("{module}::{entry} is not in KERNEL_GRIDS")),
                Some(GridKind::Folded(_)) if !folded => problems.push(format!(
                    "{module}::{entry} is marked Folded but reads no flat group id"
                )),
                Some(GridKind::Exact(_) | GridKind::Single(_)) if *folded => {
                    problems.push(format!(
                    "{module}::{entry} reads the flat group id of a folded grid but is not Folded"
                ))
                }
                Some(GridKind::Folded(_)) => {
                    if let Err(why) = guard {
                        problems.push(format!("{module}::{entry}: {why}"));
                    }
                }
                Some(_) => {}
            }
        }
        for entry in table.keys() {
            if !found.contains_key(*entry) {
                problems.push(format!(
                    "KERNEL_GRIDS lists {entry}, which no module defines"
                ));
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    /// The scanner itself: a guarded kernel passes, and an unguarded one, a
    /// guard behind a write, a guard behind a helper that reads a buffer, and
    /// a tail test that is not a bound each fail.
    #[test]
    fn the_tail_audit_catches_what_it_is_for() {
        let head = "var<storage, read_write> y0: array<f32>;\n\
                    fn flat_group(a: u32) -> u32 { return a; }\n\
                    fn peek(i: u32) -> f32 { return y0[i]; }\n";
        let kernel = |body: &str| {
            let src = format!("{head}@compute\nfn k() {{\n{body}\n}}\n");
            audit(&src).remove("k").unwrap()
        };
        let ok = kernel("let i = flat_group(1u); if (i >= 4u) { return; } y0[i] = 1.0;");
        assert_eq!(ok, (true, Ok(())));
        for (what, body) in [
            ("none", "let i = flat_group(1u); y0[i] = 1.0;"),
            (
                "late",
                "let i = flat_group(1u); y0[i] = 1.0; if (i >= 4u) { return; }",
            ),
            (
                "helper",
                "let i = flat_group(1u); let v = peek(i); if (i >= 4u) { return; }",
            ),
            (
                "not a bound",
                "let i = flat_group(1u); if (i == 9u) { return; } y0[i] = 1.0;",
            ),
        ] {
            let (folded, guard) = kernel(body);
            assert!(folded && guard.is_err(), "{what}: {guard:?}");
        }
        // A kernel that never reads a flat id is not folded.
        assert_eq!(kernel("y0[0] = 1.0;"), (false, Ok(())));
    }

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
        // The WebGPU minimum of 16 KiB still fits every head dim, including
        // the widths that pad to 256 (those take the smallest tiles).
        for d in [1u32, 16, 17, 33, 64, 100, 128, 129, 192, 200, 256] {
            let t = attention_tiles(d, 16 * 1024).unwrap();
            assert!(t.shared_bytes <= 16 * 1024, "{d}: {t:?}");
            assert!(
                t.padded_dim >= d && t.padded_dim.is_multiple_of(16),
                "{d}: {t:?}"
            );
            assert!(t.fwd_keys >= ATTENTION_PARTS && t.bwd_block >= ATTENTION_PARTS);
        }
        let wide32 = attention_tiles(256, 32 * 1024).unwrap();
        assert_eq!(
            (
                wide32.padded_dim,
                wide32.fwd_rows,
                wide32.fwd_keys,
                wide32.bwd_block
            ),
            (256, 16, 8, 8)
        );
        assert!(wide32.shared_bytes <= 32 * 1024);
        let wide16 = attention_tiles(256, 16 * 1024).unwrap();
        assert_eq!(
            (
                wide16.padded_dim,
                wide16.fwd_rows,
                wide16.fwd_keys,
                wide16.bwd_block
            ),
            (256, 8, 4, 4)
        );
        assert!(wide16.shared_bytes <= 16 * 1024);
        assert_eq!(attention_tiles(1, 1 << 20).unwrap().padded_dim, 16);
        assert_eq!(attention_tiles(65, 1 << 20).unwrap().padded_dim, 128);
        assert_eq!(attention_tiles(129, 1 << 20).unwrap().padded_dim, 256);
        assert!(matches!(
            attention_tiles(0, 1 << 20),
            Err(OjasError::OutOfRange { .. })
        ));
        assert!(matches!(
            attention_tiles(257, 1 << 20),
            Err(OjasError::UnsupportedHeadDim {
                head_dim: 257,
                limit: 256
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
