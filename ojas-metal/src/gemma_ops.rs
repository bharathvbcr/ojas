//! Launch plans of the EmbeddingGemma 2 encoder ops that are not attention
//! (`kernels/gemma_ops.metal`): the GELU tanh forward and backward, the
//! length-masked mean over a sequence and the L2 normalize. Host-side and
//! free of Metal types, so the command protocol (`link::Cmd`) can name them
//! on every target.
//!
//! What runs where:
//!
//! - `gelu_tanh_forward`: `ojas_gelu_tanh_fwd`, which calls tessl's own
//!   `tessl_gelu_pytorch_tanh` (`act.h`, included, not copied). tessl has
//!   only the fused `mlp_gelu_tanh` (`gelu(gate) * up`), which has no status
//!   words and needs an `up` operand; this is the same formula as one
//!   element per thread with ojas's finite checks.
//! - `gelu_tanh_backward`: `ojas_gelu_tanh_bwd`. tessl has no GELU
//!   derivative. It uses the forward's clamped inner term for both parts.
//! - `seq_mean_forward`: `ojas_segments_from_lens` builds the
//!   `(b * T, b * T + min(len, T))` row ranges from the device length
//!   buffer (no readback), then tessl's `segment_mean_rows`.
//! - `l2_normalize_forward`: a copy of the operand, then tessl's in-place
//!   `l2_normalize_rows`, whose epsilon is fixed at `1e-12` (torch's
//!   `F.normalize` default). Any other eps is refused, never ignored.

/// Every kernel `kernels/gemma_ops.metal` defines; the device thread builds
/// each pipeline at start-up.
pub(crate) const KERNELS: &[&str] = &[
    "ojas_gelu_tanh_fwd",
    "ojas_gelu_tanh_bwd",
    "ojas_segments_from_lens",
];

/// The epsilon of tessl's `l2_normalize_rows` kernel (`max(||x||, 1e-12)`).
pub(crate) const L2_EPS: f32 = 1e-12;

/// A masked mean of `batch` sequences of `seq` rows of `dim` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SeqMeanGeom {
    pub batch: u32,
    pub seq: u32,
    pub dim: u32,
}

impl SeqMeanGeom {
    /// Rows of the flattened `[batch * seq, dim]` operand, or `None` when
    /// they do not fit the kernels' 32-bit row indices.
    pub(crate) fn rows(&self) -> Option<u32> {
        self.batch.checked_mul(self.seq)
    }

    /// Values of the output, `[batch, dim]`.
    pub(crate) fn out_elems(&self) -> Option<usize> {
        (self.batch as usize).checked_mul(self.dim as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flattened_row_count_refuses_to_wrap() {
        let g = SeqMeanGeom {
            batch: 3,
            seq: 7,
            dim: 5,
        };
        assert_eq!(g.rows(), Some(21));
        assert_eq!(g.out_elems(), Some(15));
        let big = SeqMeanGeom {
            batch: 1 << 16,
            seq: 1 << 16,
            dim: 1,
        };
        assert_eq!(big.rows(), None, "2^32 rows do not fit a u32 index");
    }

    #[test]
    fn the_l2_epsilon_is_the_one_tessl_hardcodes() {
        // `l2_normalize_rows_f32`: `max(sqrt(total), 1e-12f)`.
        assert_eq!(L2_EPS.to_bits(), 1e-12f32.to_bits());
    }
}
