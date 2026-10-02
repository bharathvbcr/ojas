//! K2(i): the plan for the GDN training scan at the **published** rule, on the
//! host: shapes, variable sequence lengths, the checkpoint layout, buffer
//! lengths, launch geometry, and the refusals for what the kernels cannot run.
//!
//! The kernels (`crate::gdn_kernels`) port tessl's token-sequential scan
//! (`tessl/kernels/gdn_train.metal`, host side `tessl/src/gdn_train.rs`).
//! tessl runs one sequence per launch (`tessl/src/qwen35_train.rs:841-846`,
//! `batch: 1`). This plan batches `B >= 1` sequences of **unpadded, variable
//! lengths** into one launch.
//!
//! # Layout: the concatenation of the `B = 1` layouts
//!
//! Every tensor is the concatenation, in sequence order, of the tensors a
//! `B = 1` call on each sequence would take:
//!
//! - tokens are packed: `q`, `k`, `dq`, `dk` `[N, H, 128]`; `v`, `o`, `d_o`,
//!   `dv` `[N, H, Dv]`; `g`, `beta`, `dg`, `dbeta` `[N, H]`, with
//!   `N = sum(T_b)` and sequence `b` at token rows
//!   `tok_off[b] .. tok_off[b + 1]`;
//! - states: `s0`, `s_fin`, `d_fin`, `ds0` `[B, H, 128, Dv]`;
//! - checkpoints: sequence `b`'s block `[H, NC_b, 128, Dv]`, `NC_b =
//!   ceil(T_b / 64)`, starting at checkpoint `ck_off[b] = sum_{b' < b} NC_b'`
//!   (times `H * 128 * Dv` elements).
//!
//! When every length is `T`, this is exactly tessl's dense layout:
//! `[B, T, H, D]` tokens and `[B, H, NC, 128, Dv]` checkpoints
//! (`tessl/src/gdn_train.rs:24-25,88-97`), which is the layout of
//! L-cuda-oracle's float64 reference (`tests/reference/gdn_published.rs`). A
//! unit test below asserts the equivalence index by index.
//!
//! # What is refused (fail closed, before any device work)
//!
//! tessl's limits (`tessl/src/gdn_train.rs:56-73,324-342`): a zero batch,
//! sequence or head count; a value dim that is not a non-zero multiple of 16;
//! more than `u32::MAX` `(token, head)` rows; a buffer wider than
//! `i64::MAX / 4` elements. Added for CUDA: a key head dim other than 128 (the
//! kernels are compiled for it), more than `u32::MAX` packed tokens or
//! checkpoints (the offsets are `u32` on the device), and `B * H` past
//! `gridDim.y`'s 65,535.
//!
//! # Precondition on the data: `g <= 0`
//!
//! `g` is a log decay (in the model, `g = -exp(A_log) softplus(a + dt_bias)`).
//! The kernels' `exp(g)` is the crate's `qd_exp_nonpos` (`crate::k8_act`),
//! which returns NaN for a positive argument, `+inf` included, so a positive
//! or NaN `g` makes every output it reaches NaN rather than a silently wrong
//! number. The plan sees shapes, not data, so this is not checked here.
//!
//! # Geometry
//!
//! A function of the shape only, never of the SM count
//! (`cuda-backend-scoping.md` §3, determinism rule):
//! - the scan kernels (forward, backward): one 128-thread block per
//!   (16-column value slice, sequence × head): grid `(Dv / 16, B * H)`;
//! - the backward's finish: one 128-thread block per `(token, head)` row,
//!   block-striding over rows with a fixed cap of 65,535 blocks.

use crate::error::CudaError;
use crate::geometry::{Launch, MAX_BLOCKS_1D, MAX_GRID_Y};

/// Key head dim the kernels are compiled for (tessl `GDN_TRAIN_DK`,
/// `src/gdn_train.rs:39`).
pub const GDN_DK: usize = 128;
/// Value columns per block; `v_dim` must be a non-zero multiple (tessl
/// `GDN_TRAIN_BV`, `src/gdn_train.rs:41`).
pub const GDN_BV: usize = 16;
/// Tokens between saved states (tessl `GDN_TRAIN_CKPT`, `src/gdn_train.rs:43`).
pub const GDN_CKPT: usize = 64;
/// Threads per block: one per key row (tessl `THREADS`, `src/gdn_train.rs:44`).
pub const GDN_THREADS: u32 = 128;
/// transformers' `l2norm` epsilon (`gdn_train.metal:39`).
pub const GDN_L2_EPS: f32 = 1e-6;

const OP: &str = "gdn_published_plan";

/// A validated problem: sequence lengths, heads and value dim, with every
/// derived offset and length computed by checked arithmetic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GdnPublishedPlan {
    lens: Vec<usize>,
    heads: usize,
    v_dim: usize,
    tok_off: Vec<u32>,
    ck_off: Vec<u32>,
    tokens: usize,
    checkpoints: usize,
    sizes: Sizes,
}

/// Element counts of every buffer, computed once with checked arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sizes {
    qk: usize,
    v: usize,
    gate: usize,
    state: usize,
    ckpt: usize,
    scratch: usize,
    qk_part: usize,
    gate_part: usize,
}

fn mul(a: usize, b: usize, what: &str) -> Result<usize, CudaError> {
    a.checked_mul(b)
        .ok_or_else(|| CudaError::invalid(OP, format!("{what} overflows usize ({a} x {b})")))
}

impl GdnPublishedPlan {
    /// A batch of sequences of the given lengths. `key_dim` must be 128.
    pub fn new(
        lens: &[usize],
        heads: usize,
        key_dim: usize,
        v_dim: usize,
    ) -> Result<Self, CudaError> {
        if lens.is_empty() {
            return Err(CudaError::invalid(
                OP,
                "batch, seq and heads must be non-zero: no sequences",
            ));
        }
        if let Some(b) = lens.iter().position(|&t| t == 0) {
            return Err(CudaError::invalid(
                OP,
                format!("batch, seq and heads must be non-zero: sequence {b} has length 0"),
            ));
        }
        if heads == 0 {
            return Err(CudaError::invalid(
                OP,
                "batch, seq and heads must be non-zero: 0 heads",
            ));
        }
        if key_dim != GDN_DK {
            return Err(CudaError::invalid(
                OP,
                format!(
                    "key_dim must be {GDN_DK} (the kernels are compiled for it), got {key_dim}"
                ),
            ));
        }
        if v_dim == 0 || !v_dim.is_multiple_of(GDN_BV) {
            return Err(CudaError::invalid(
                OP,
                format!("v_dim must be a non-zero multiple of {GDN_BV}, got {v_dim}"),
            ));
        }
        let batch = lens.len();
        let blocks_y = mul(batch, heads, "sequences x heads")?;
        if blocks_y > MAX_GRID_Y as usize {
            return Err(CudaError::invalid(
                OP,
                format!("{batch} sequences x {heads} heads = {blocks_y} blocks in y, past gridDim.y's {MAX_GRID_Y}"),
            ));
        }

        let mut tok_off = Vec::with_capacity(batch + 1);
        let mut ck_off = Vec::with_capacity(batch + 1);
        let (mut tokens, mut checkpoints) = (0usize, 0usize);
        for &t in lens {
            tok_off.push(to_u32(tokens, "packed tokens")?);
            ck_off.push(to_u32(checkpoints, "packed checkpoints")?);
            tokens = tokens
                .checked_add(t)
                .ok_or_else(|| CudaError::invalid(OP, "the total token count overflows usize"))?;
            checkpoints += t.div_ceil(GDN_CKPT);
        }
        tok_off.push(to_u32(tokens, "packed tokens")?);
        ck_off.push(to_u32(checkpoints, "packed checkpoints")?);

        let rows = mul(tokens, heads, "(token, head) rows")?;
        if rows > u32::MAX as usize {
            return Err(CudaError::invalid(
                OP,
                format!(
                    "{rows} (token, head) rows is too large: the limit is {}",
                    u32::MAX
                ),
            ));
        }
        let slices = v_dim / GDN_BV;
        let per_state = GDN_DK * v_dim;
        let sizes = Sizes {
            qk: mul(rows, GDN_DK, "q/k")?,
            v: mul(rows, v_dim, "v/o")?,
            gate: rows,
            state: mul(blocks_y, per_state, "states")?,
            ckpt: mul(
                mul(checkpoints, heads, "checkpoints")?,
                per_state,
                "checkpoints",
            )?,
            scratch: mul(
                mul(blocks_y, slices, "scratch")?,
                GDN_CKPT * GDN_DK * GDN_BV,
                "scratch",
            )?,
            qk_part: mul(
                mul(slices, rows, "dq/dk partials")?,
                GDN_DK,
                "dq/dk partials",
            )?,
            gate_part: mul(slices, rows, "dg/dbeta partials")?,
        };
        let widest = [
            sizes.qk,
            sizes.v,
            sizes.state,
            sizes.ckpt,
            sizes.scratch,
            sizes.qk_part,
            sizes.gate_part,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        if widest as u64 > (i64::MAX as u64) / 4 {
            return Err(CudaError::invalid(
                OP,
                format!("the shape is too large: a buffer of {widest} f32 elements"),
            ));
        }
        Ok(GdnPublishedPlan {
            lens: lens.to_vec(),
            heads,
            v_dim,
            tok_off,
            ck_off,
            tokens,
            checkpoints,
            sizes,
        })
    }

    /// `batch` sequences of one length `seq`: tessl's dense `[B, T, H, D]` case.
    pub fn dense(
        batch: usize,
        seq: usize,
        heads: usize,
        key_dim: usize,
        v_dim: usize,
    ) -> Result<Self, CudaError> {
        if batch == 0 {
            return Err(CudaError::invalid(
                OP,
                "batch, seq and heads must be non-zero: batch 0",
            ));
        }
        Self::new(&vec![seq; batch], heads, key_dim, v_dim)
    }

    /// The plan of sequence `b` alone (`B = 1`).
    pub fn sequence(&self, b: usize) -> Result<Self, CudaError> {
        let t = *self.lens.get(b).ok_or_else(|| {
            CudaError::invalid(OP, format!("sequence {b} of {}", self.lens.len()))
        })?;
        Self::new(&[t], self.heads, GDN_DK, self.v_dim)
    }

    /// Sequences.
    pub fn batch(&self) -> usize {
        self.lens.len()
    }
    /// Each sequence's length.
    pub fn lens(&self) -> &[usize] {
        &self.lens
    }
    /// Value heads (after any grouping repeat; the kernels have no grouping).
    pub fn heads(&self) -> usize {
        self.heads
    }
    /// Value head dim.
    pub fn v_dim(&self) -> usize {
        self.v_dim
    }
    /// `v_dim / 16`: the value slices, one block each.
    pub fn slices(&self) -> usize {
        self.v_dim / GDN_BV
    }
    /// Packed tokens `N`.
    pub fn tokens(&self) -> usize {
        self.tokens
    }
    /// `N * H`: the `(token, head)` rows.
    pub fn rows(&self) -> usize {
        self.sizes.gate
    }
    /// Checkpoints per head over all sequences, `sum(NC_b)`.
    pub fn checkpoints(&self) -> usize {
        self.checkpoints
    }
    /// `ceil(T_b / 64)`.
    pub fn seq_checkpoints(&self, b: usize) -> usize {
        self.lens[b].div_ceil(GDN_CKPT)
    }
    /// Token offsets, `B + 1` entries; uploaded for the kernels.
    pub fn tok_off(&self) -> &[u32] {
        &self.tok_off
    }
    /// Checkpoint offsets, `B + 1` entries; uploaded for the kernels.
    pub fn ck_off(&self) -> &[u32] {
        &self.ck_off
    }
    /// `Some(T)` when every sequence has length `T` (tessl's dense case).
    pub fn dense_seq(&self) -> Option<usize> {
        let first = self.lens[0];
        self.lens.iter().all(|&t| t == first).then_some(first)
    }

    /// Elements of `q`, `k`, `dq`, `dk`: `N * H * 128`.
    pub fn qk_len(&self) -> usize {
        self.sizes.qk
    }
    /// Elements of `v`, `o`, `d_o`, `dv`: `N * H * Dv`.
    pub fn v_len(&self) -> usize {
        self.sizes.v
    }
    /// Elements of `g`, `beta`, `dg`, `dbeta`: `N * H`.
    pub fn gate_len(&self) -> usize {
        self.sizes.gate
    }
    /// Elements of `s0`, `s_fin`, `d_fin`, `ds0`: `B * H * 128 * Dv`.
    pub fn state_len(&self) -> usize {
        self.sizes.state
    }
    /// Elements of the checkpoints: `sum(NC_b) * H * 128 * Dv`.
    pub fn ckpt_len(&self) -> usize {
        self.sizes.ckpt
    }
    /// Backward scratch: one chunk of states per block,
    /// `B * H * slices * 64 * 128 * 16` (tessl `src/gdn_train.rs:123`).
    pub fn scratch_len(&self) -> usize {
        self.sizes.scratch
    }
    /// `dq_part` and `dk_part` each: `slices * N * H * 128`.
    pub fn qk_part_len(&self) -> usize {
        self.sizes.qk_part
    }
    /// `dg_part` and `dbeta_part` each: `slices * N * H`.
    pub fn gate_part_len(&self) -> usize {
        self.sizes.gate_part
    }

    /// Device bytes of the backward's workspace (scratch, the four partials,
    /// and the two offset arrays), as tessl's `bytes_for`
    /// (`src/gdn_train.rs:132-140`) plus the offsets.
    pub fn workspace_bytes(&self) -> u64 {
        let f32s = self.sizes.scratch as u64
            + 2 * self.sizes.qk_part as u64
            + 2 * self.sizes.gate_part as u64;
        4 * f32s + 4 * 2 * (self.batch() as u64 + 1)
    }

    /// Device bytes of every operand a forward plus backward touches outside
    /// the workspace: q, k, v, g, beta, o, ckpt, d_o, the five gradients,
    /// and the four state tensors when present.
    pub fn operand_bytes(&self, with_states: bool) -> u64 {
        let s = &self.sizes;
        let mut f32s = 4 * s.qk as u64 + 4 * s.v as u64 + 4 * s.gate as u64 + s.ckpt as u64;
        if with_states {
            f32s += 4 * s.state as u64;
        }
        4 * f32s
    }

    /// Packed token row of token `t` of sequence `b`.
    pub fn token_row(&self, b: usize, t: usize) -> usize {
        assert!(
            t < self.lens[b],
            "token {t} of a {}-token sequence",
            self.lens[b]
        );
        self.tok_off[b] as usize + t
    }

    /// The `(token, head)` row index of `g`, `beta` (and, times the width,
    /// of `q`, `k`, `v`).
    pub fn row(&self, b: usize, t: usize, h: usize) -> usize {
        assert!(h < self.heads, "head {h} of {}", self.heads);
        self.token_row(b, t) * self.heads + h
    }

    /// Element offset of `state[b, h, 0, 0]` in a `[B, H, 128, Dv]` tensor.
    pub fn state_offset(&self, b: usize, h: usize) -> usize {
        assert!(b < self.batch() && h < self.heads);
        (b * self.heads + h) * GDN_DK * self.v_dim
    }

    /// Element offset of checkpoint `c` (the state entering token `64 c`) of
    /// head `h` of sequence `b`, row 0, column 0. The kernels compute the same
    /// expression, `((ck_off[b] * H + h * NC_b + c) * 128 + i) * Dv + j`.
    pub fn ckpt_offset(&self, b: usize, h: usize, c: usize) -> usize {
        let nc = self.seq_checkpoints(b);
        assert!(h < self.heads && c < nc, "checkpoint {c} of {nc}, head {h}");
        (self.ck_off[b] as usize * self.heads + h * nc + c) * GDN_DK * self.v_dim
    }

    /// The scan kernels' launch: one 128-thread block per (value slice,
    /// sequence × head).
    pub fn scan_launch(&self) -> Launch {
        Launch {
            grid: (self.slices() as u32, (self.batch() * self.heads) as u32, 1),
            block: (GDN_THREADS, 1, 1),
        }
    }

    /// The finish kernel's launch: one 128-thread block per row, capped at a
    /// fixed 65,535 blocks that stride over the rest.
    pub fn finish_launch(&self) -> Launch {
        let blocks = self.rows().min(MAX_BLOCKS_1D as usize) as u32;
        Launch {
            grid: (blocks, 1, 1),
            block: (GDN_THREADS, 1, 1),
        }
    }

    /// Split a packed tensor of `width` elements per `(token, head)` row into
    /// each sequence's `B = 1` tensor.
    pub fn split_tokens<'a, T>(
        &self,
        packed: &'a [T],
        width: usize,
    ) -> Result<Vec<&'a [T]>, CudaError> {
        let per_token = mul(self.heads, width, "row width")?;
        if packed.len() != mul(self.tokens, per_token, "packed length")? {
            return Err(CudaError::invalid(
                OP,
                format!(
                    "{} packed elements for {} tokens x {per_token}",
                    packed.len(),
                    self.tokens
                ),
            ));
        }
        Ok(self
            .tok_off
            .windows(2)
            .map(|w| &packed[w[0] as usize * per_token..w[1] as usize * per_token])
            .collect())
    }

    /// Split a checkpoint tensor into each sequence's `[1, H, NC_b, 128, Dv]`.
    pub fn split_ckpt<'a, T>(&self, ckpt: &'a [T]) -> Result<Vec<&'a [T]>, CudaError> {
        if ckpt.len() != self.ckpt_len() {
            return Err(CudaError::invalid(
                OP,
                format!(
                    "{} checkpoint elements, the plan has {}",
                    ckpt.len(),
                    self.ckpt_len()
                ),
            ));
        }
        let per = self.heads * GDN_DK * self.v_dim;
        Ok(self
            .ck_off
            .windows(2)
            .map(|w| &ckpt[w[0] as usize * per..w[1] as usize * per])
            .collect())
    }

    /// Split a `[B, H, 128, Dv]` state tensor into each sequence's.
    pub fn split_states<'a, T>(&self, states: &'a [T]) -> Result<Vec<&'a [T]>, CudaError> {
        if states.len() != self.state_len() {
            return Err(CudaError::invalid(
                OP,
                format!(
                    "{} state elements, the plan has {}",
                    states.len(),
                    self.state_len()
                ),
            ));
        }
        Ok(states
            .chunks_exact(self.heads * GDN_DK * self.v_dim)
            .collect())
    }
}

fn to_u32(n: usize, what: &str) -> Result<u32, CudaError> {
    u32::try_from(n).map_err(|_| {
        CudaError::invalid(
            OP,
            format!("{n} {what} is too large: the device offsets are u32"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(r: Result<GdnPublishedPlan, CudaError>, needle: &str) {
        let e = r.expect_err(needle).to_string();
        assert!(e.contains(needle), "{e:?} lacks {needle:?}");
    }

    #[test]
    fn published_plan_refuses_what_the_kernels_cannot_run() {
        refused(GdnPublishedPlan::new(&[], 2, 128, 32), "non-zero");
        refused(
            GdnPublishedPlan::new(&[5, 0, 3], 2, 128, 32),
            "sequence 1 has length 0",
        );
        refused(GdnPublishedPlan::new(&[5], 0, 128, 32), "non-zero");
        refused(GdnPublishedPlan::dense(0, 5, 2, 128, 32), "non-zero");
        refused(GdnPublishedPlan::new(&[5], 2, 128, 24), "multiple of 16");
        refused(GdnPublishedPlan::new(&[5], 2, 128, 0), "multiple of 16");
        refused(
            GdnPublishedPlan::new(&[5], 2, 64, 32),
            "key_dim must be 128",
        );
        refused(
            GdnPublishedPlan::new(&[5], 2, 256, 32),
            "key_dim must be 128",
        );
        refused(GdnPublishedPlan::new(&[1; 2], 32_768, 128, 16), "gridDim.y");
        GdnPublishedPlan::new(&[1; 2], 32_767, 128, 16).expect("65,534 blocks in y fit");
        // More than u32::MAX (token, head) rows (tessl's `rows > u32::MAX`).
        refused(
            GdnPublishedPlan::new(&[1 << 31, 1 << 31], 2, 128, 16),
            "u32",
        );
        refused(
            GdnPublishedPlan::new(&[1 << 30], 8, 128, 16),
            "rows is too large",
        );
        // A buffer wider than i64::MAX / 4 elements.
        // dq_part = (Dv / 16) * rows * 128 = 2^23 * 4,278,190,080 * 128 elements.
        refused(
            GdnPublishedPlan::new(&[1 << 24], 255, 128, 1 << 27),
            "too large",
        );
    }

    #[test]
    fn published_plan_offsets_and_lengths_for_variable_lengths() {
        let p = GdnPublishedPlan::new(&[1, 63, 64, 65, 130], 3, 128, 32).unwrap();
        assert_eq!(p.batch(), 5);
        assert_eq!(p.tok_off(), &[0, 1, 64, 128, 193, 323]);
        // NC_b = 1, 1, 1, 2, 3.
        assert_eq!(p.ck_off(), &[0, 1, 2, 3, 5, 8]);
        assert_eq!(p.tokens(), 323);
        assert_eq!(p.rows(), 969);
        assert_eq!(p.checkpoints(), 8);
        assert_eq!(p.slices(), 2);
        assert_eq!(p.qk_len(), 969 * 128);
        assert_eq!(p.v_len(), 969 * 32);
        assert_eq!(p.gate_len(), 969);
        assert_eq!(p.state_len(), 5 * 3 * 128 * 32);
        assert_eq!(p.ckpt_len(), 8 * 3 * 128 * 32);
        assert_eq!(p.scratch_len(), 5 * 3 * 2 * 64 * 128 * 16);
        assert_eq!(p.qk_part_len(), 2 * 969 * 128);
        assert_eq!(p.gate_part_len(), 2 * 969);
        assert_eq!(p.dense_seq(), None);
        assert_eq!(p.scan_launch().grid, (2, 15, 1));
        assert_eq!(p.scan_launch().block, (128, 1, 1));
        assert_eq!(p.finish_launch().grid, (969, 1, 1));
    }

    /// Equal lengths give tessl's dense `[B, T, H, D]` and `[B, H, NC, 128, Dv]`
    /// layouts index by index, and tessl's workspace size
    /// (`tessl/tests/gdn_train.rs:497-500`) plus the offsets.
    #[test]
    fn published_plan_with_equal_lengths_is_tessls_dense_layout() {
        let (b, t, h, dv) = (2, 130, 3, 32);
        let p = GdnPublishedPlan::dense(b, t, h, 128, dv).unwrap();
        assert_eq!(p.dense_seq(), Some(t));
        let nc = t.div_ceil(64);
        for bi in 0..b {
            for ti in 0..t {
                for hi in 0..h {
                    assert_eq!(p.row(bi, ti, hi), (bi * t + ti) * h + hi);
                }
            }
            for hi in 0..h {
                assert_eq!(p.state_offset(bi, hi), (bi * h + hi) * 128 * dv);
                for c in 0..nc {
                    assert_eq!(
                        p.ckpt_offset(bi, hi, c),
                        ((bi * h + hi) * nc + c) * 128 * dv
                    );
                }
            }
        }
        assert_eq!(p.ckpt_len(), b * h * nc * 128 * dv);
        let tessl = GdnPublishedPlan::dense(1, 5, 2, 128, 32).unwrap();
        assert_eq!(
            tessl.workspace_bytes(),
            4 * (2 * 2 * 64 * 128 * 16 + 2 * 2 * 10 * 128 + 2 * 2 * 10) + 4 * 2 * 2
        );
    }

    /// Every packed tensor is the concatenation of the sequences' `B = 1`
    /// tensors: splitting by the plan gives each sequence's own plan's sizes.
    #[test]
    fn published_plan_packed_tensors_split_into_each_sequences_b1_tensors() {
        let p = GdnPublishedPlan::new(&[3, 64, 65], 2, 128, 16).unwrap();
        let q: Vec<usize> = (0..p.qk_len()).collect();
        let ck: Vec<usize> = (0..p.ckpt_len()).collect();
        let st: Vec<usize> = (0..p.state_len()).collect();
        let (qs, cs, ss) = (
            p.split_tokens(&q, GDN_DK).unwrap(),
            p.split_ckpt(&ck).unwrap(),
            p.split_states(&st).unwrap(),
        );
        let mut at = (0, 0, 0);
        for b in 0..p.batch() {
            let one = p.sequence(b).unwrap();
            assert_eq!(qs[b].len(), one.qk_len());
            assert_eq!(cs[b].len(), one.ckpt_len());
            assert_eq!(ss[b].len(), one.state_len());
            assert_eq!(qs[b][0], at.0);
            assert_eq!(cs[b][0], at.1);
            assert_eq!(ss[b][0], at.2);
            assert_eq!(cs[b][0], p.ckpt_offset(b, 0, 0));
            assert_eq!(qs[b][0], p.row(b, 0, 0) * GDN_DK);
            at = (
                at.0 + one.qk_len(),
                at.1 + one.ckpt_len(),
                at.2 + one.state_len(),
            );
        }
        assert_eq!(at, (p.qk_len(), p.ckpt_len(), p.state_len()));
        assert!(p.split_tokens(&q[1..], GDN_DK).is_err());
        assert!(p.split_ckpt(&ck[1..]).is_err());
        assert!(p.split_states(&st[1..]).is_err());
        assert!(p.sequence(3).is_err());
    }

    #[test]
    fn published_plan_finish_grid_is_capped_not_sized_by_the_device() {
        // 4 x 8192 tokens x 16 heads: 524,288 rows, capped at 65,535 blocks.
        let p = GdnPublishedPlan::dense(4, 8192, 16, 128, 128).unwrap();
        assert_eq!(p.finish_launch().grid, (65_535, 1, 1));
        assert_eq!(p.scan_launch().grid, (8, 64, 1));
        // The working set the timing run needs, for its budget: 7.25 GB.
        let total = p.workspace_bytes() + p.operand_bytes(false);
        assert!(total > 7_000_000_000 && total < 7_500_000_000, "{total}");
    }
}
