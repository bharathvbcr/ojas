//! **K9: the token embedding**, forward and backward (`cuda-backend-scoping.md`
//! §3 K9): plans, CUDA-C source and host mirrors. The device launches are
//! [`crate::embed_cuda`].
//!
//! - **Forward** `out[r, :] = table[ids[r], :]`: a gather, copied as bits
//!   (`unsigned int` loads and stores, as tessl's `qwen35_embed_rows_f32`,
//!   `tessl/kernels/qwen35_score.metal:186-203`), so it is bit-exact whatever
//!   the values, NaN payloads included.
//! - **Backward** `dw[ids[r], :] += dh[r, :]` for every row, **added** to what
//!   `dw` holds: with Qwen3.5's tied head, `dw` is the head's weight gradient,
//!   which the cross-entropy ([`crate::ce_rows`]) writes first (tessl
//!   `embed_rows_bwd`, `tessl/src/qwen35_bwd.rs:620-631`).
//!
//! **Deterministic backward** (§3 K9: "sort the ids once on the host, then one
//! block per distinct id sums its rows in position order"). Training ids are
//! known on the host, so [`EmbedRuns::new`] checks them (`< vocab`) and groups
//! the rows by id: run `u` covers `pos[run_start[u] .. run_start[u + 1])`, rows
//! ascending, all with id `uniq[u]`. One thread per (run, column) sums its run
//! in that order from `+0.0` and adds the sum once; no two runs share an id, so
//! there are no atomics and the bits are the same on every run (tessl
//! `qwen35_embed_rows_bwd_f32`, `qwen35_bwd.metal:584-615`).
//!
//! Both kernels use only loads, stores and `+`, so the host mirrors here are
//! **bitwise** mirrors.

use crate::error::CudaError;
use crate::kernels::KernelModule;
use crate::small_common::{exact_len, mul, nonzero, to_u64};

/// Entry: the forward gather.
pub const EMBED_FWD: &str = "qd_embed_rows_f32";
/// Entry: the backward, added into `dw`.
pub const EMBED_BWD: &str = "qd_embed_rows_bwd_f32";

/// The K9 NVRTC module.
pub const MODULE: KernelModule = KernelModule {
    name: "k9_embed",
    source: SOURCE,
    entries: &[EMBED_FWD, EMBED_BWD],
};

const SOURCE: &str = concat!(
    crate::device_prelude!(),
    crate::small_common::small_prelude!(),
    r#"
// out[r, c] = table[ids[r], c], as bits. ids are checked on the host.
extern "C" __global__ void qd_embed_rows_f32(
    const unsigned int* ids, const unsigned int* table, unsigned int* out,
    unsigned long long n, unsigned long long hidden)
{
    const unsigned long long total = n * hidden;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long r = i / hidden;
        const unsigned long long c = i - r * hidden;
        out[i] = table[(unsigned long long)ids[r] * hidden + c];
    }
}

// dw[uniq[u], c] += sum over k in [run_start[u], run_start[u + 1]) of
// dh[pos[k], c], k ascending. Runs have distinct ids: one writer per element.
extern "C" __global__ void qd_embed_rows_bwd_f32(
    const float* dh, const unsigned int* pos, const unsigned int* run_start,
    const unsigned int* uniq, float* dw,
    unsigned long long n_runs, unsigned long long hidden)
{
    const unsigned long long total = n_runs * hidden;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long u = i / hidden;
        const unsigned long long c = i - u * hidden;
        float s = 0.0f;
        for (unsigned int k = run_start[u]; k < run_start[u + 1]; ++k) {
            s = s + dh[(unsigned long long)pos[k] * hidden + c];
        }
        const unsigned long long at = (unsigned long long)uniq[u] * hidden + c;
        dw[at] = dw[at] + s;
    }
}
"#
);

/// An embedding of `n` ids into a `[vocab, hidden]` table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbedPlan {
    /// Ids (rows of `out` and `dh`).
    pub n: u64,
    /// Rows of the table (248,320 at Qwen3.5-2B).
    pub vocab: u64,
    /// Columns (2048 at Qwen3.5-2B).
    pub hidden: u64,
}

impl EmbedPlan {
    /// Validate the shape. Ids are `u32`, so `vocab` is at most 2^32 and `n`
    /// below 2^32 (run starts are `u32` offsets into the rows).
    pub fn new(n: u64, vocab: u64, hidden: u64) -> Result<Self, CudaError> {
        const OP: &str = "embed_rows";
        nonzero(OP, "n", n)?;
        nonzero(OP, "vocab", vocab)?;
        nonzero(OP, "hidden", hidden)?;
        if vocab > 1u64 << 32 {
            return Err(CudaError::invalid(
                OP,
                format!("vocab {vocab} exceeds u32 ids"),
            ));
        }
        if n >= 1u64 << 32 {
            return Err(CudaError::invalid(
                OP,
                format!("{n} rows exceed u32 run offsets"),
            ));
        }
        mul(n, hidden, OP)?;
        mul(vocab, hidden, OP)?;
        Ok(EmbedPlan { n, vocab, hidden })
    }

    /// Check `ids` against the plan: one per row, each below `vocab`.
    pub fn check_ids(&self, ids: &[u32]) -> Result<(), CudaError> {
        const OP: &str = "embed_rows";
        exact_len(OP, "ids", ids.len(), self.n)?;
        if let Some((r, &id)) = ids
            .iter()
            .enumerate()
            .find(|(_, &id)| u64::from(id) >= self.vocab)
        {
            return Err(CudaError::invalid(
                OP,
                format!("ids[{r}] = {id} is not below vocab {}", self.vocab),
            ));
        }
        Ok(())
    }

    /// Check the forward's buffers.
    pub fn check_fwd(&self, ids: &[u32], table: usize, out: usize) -> Result<(), CudaError> {
        const OP: &str = "embed_rows";
        self.check_ids(ids)?;
        exact_len(OP, "table", table, self.vocab * self.hidden)?;
        exact_len(OP, "out", out, self.n * self.hidden)
    }

    /// Check the backward's buffers.
    pub fn check_bwd(&self, ids: &[u32], dh: usize, dw: usize) -> Result<(), CudaError> {
        const OP: &str = "embed_rows_bwd";
        self.check_ids(ids)?;
        exact_len(OP, "dh", dh, self.n * self.hidden)?;
        exact_len(OP, "dw", dw, self.vocab * self.hidden)
    }
}

/// The rows grouped by id for the backward.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbedRuns {
    /// Row indices sorted by `(id, row)`.
    pub pos: Vec<u32>,
    /// `run_start[u] .. run_start[u + 1]` is run `u`'s slice of `pos`; one
    /// more entry than there are runs.
    pub run_start: Vec<u32>,
    /// Run `u`'s id; strictly increasing, so no two runs share an id.
    pub uniq: Vec<u32>,
}

impl EmbedRuns {
    /// Group `ids` (checked against `plan`) into runs.
    pub fn new(plan: &EmbedPlan, ids: &[u32]) -> Result<Self, CudaError> {
        plan.check_ids(ids)?;
        let n = u32::try_from(ids.len())
            .map_err(|_| CudaError::invalid("embed_rows_bwd", "more than u32::MAX rows"))?;
        let mut pos: Vec<u32> = (0..n).collect();
        pos.sort_by_key(|&r| (ids[r as usize], r));
        let mut run_start = Vec::new();
        let mut uniq: Vec<u32> = Vec::new();
        for (i, &r) in pos.iter().enumerate() {
            let id = ids[r as usize];
            if uniq.last() != Some(&id) {
                uniq.push(id);
                run_start.push(u32::try_from(i).unwrap_or(u32::MAX));
            }
        }
        run_start.push(n);
        Ok(EmbedRuns {
            pos,
            run_start,
            uniq,
        })
    }

    /// Runs (distinct ids).
    pub fn n_runs(&self) -> u64 {
        to_u64(self.uniq.len())
    }
}

/// `qd_embed_rows_f32` on the host (bit copies).
pub fn embed_rows_fwd_mirror(
    plan: &EmbedPlan,
    ids: &[u32],
    table: &[f32],
) -> Result<Vec<f32>, CudaError> {
    let hidden = usize::try_from(plan.hidden).unwrap_or(usize::MAX);
    let n = ids.len();
    plan.check_fwd(ids, table.len(), n.saturating_mul(hidden))?;
    let mut out = Vec::with_capacity(n * hidden);
    for &id in ids {
        let at = id as usize * hidden;
        out.extend(
            table[at..at + hidden]
                .iter()
                .map(|v| f32::from_bits(v.to_bits())),
        );
    }
    Ok(out)
}

/// `qd_embed_rows_bwd_f32` on the host: adds into `dw`.
pub fn embed_rows_bwd_mirror(
    plan: &EmbedPlan,
    ids: &[u32],
    dh: &[f32],
    dw: &mut [f32],
) -> Result<(), CudaError> {
    plan.check_bwd(ids, dh.len(), dw.len())?;
    let runs = EmbedRuns::new(plan, ids)?;
    let hidden = usize::try_from(plan.hidden).unwrap_or(usize::MAX);
    for (u, &id) in runs.uniq.iter().enumerate() {
        let rows = &runs.pos[runs.run_start[u] as usize..runs.run_start[u + 1] as usize];
        for c in 0..hidden {
            let s = rows
                .iter()
                .fold(0.0f32, |s, &r| s + dh[r as usize * hidden + c]);
            let at = id as usize * hidden + c;
            dw[at] += s;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_group_rows_by_id_in_position_order() {
        let plan = EmbedPlan::new(7, 10, 2).unwrap();
        let ids = [4u32, 1, 4, 9, 1, 4, 0];
        let runs = EmbedRuns::new(&plan, &ids).unwrap();
        assert_eq!(runs.uniq, vec![0, 1, 4, 9]);
        assert_eq!(runs.run_start, vec![0, 1, 3, 6, 7]);
        assert_eq!(runs.pos, vec![6, 1, 4, 0, 2, 5, 3]);
        assert_eq!(runs.n_runs(), 4);
    }

    #[test]
    fn ids_past_the_vocabulary_and_short_buffers_are_refused() {
        let plan = EmbedPlan::new(3, 10, 2).unwrap();
        let err = plan.check_ids(&[0, 10, 3]).unwrap_err();
        assert!(err.to_string().contains("ids[1] = 10"), "{err}");
        assert!(plan.check_ids(&[0, 1]).is_err());
        assert!(plan.check_fwd(&[0, 1, 2], 20, 5).is_err());
        assert!(plan.check_bwd(&[0, 1, 2], 6, 19).is_err());
        assert!(EmbedPlan::new(0, 10, 2).is_err());
        assert!(EmbedPlan::new(1, (1 << 32) + 1, 2).is_err());
    }

    #[test]
    fn the_backward_adds_each_ids_rows_once_in_row_order() {
        let plan = EmbedPlan::new(4, 3, 1).unwrap();
        let ids = [2u32, 0, 2, 2];
        // Large-then-small in row order: (1e8 + 1) - 1e8 + 1 in f32 is 1, not 2.
        let dh = [1.0e8f32, 5.0, 1.0, -1.0e8];
        let mut dw = vec![0.25f32; 3];
        embed_rows_bwd_mirror(&plan, &ids, &dh, &mut dw).unwrap();
        let s = 0.0f32 + 1.0e8 + 1.0 - 1.0e8;
        assert_eq!(dw, vec![0.25 + 5.0, 0.25, 0.25 + s]);
        let fwd = embed_rows_fwd_mirror(
            &plan,
            &[1, 1, 0, 2],
            &[7.0, f32::from_bits(0x7fa0_0001), 9.0],
        )
        .unwrap();
        assert_eq!(
            fwd[0].to_bits(),
            0x7fa0_0001,
            "a NaN payload is copied as bits"
        );
    }
}
