//! The one owner of the tessl layout numbers the host side sizes scratch by.
//!
//! `backend.rs` and `link.rs` charge a call's scratch to the budget and refuse
//! an unsupported shape before any device call, so they need tessl's block
//! sizes, widths and word counts, yet they must also build where tessl does
//! not (the stub off macOS, or without the `metal` feature). This module is the
//! single place that bridges the two:
//!
//! - With tessl (macOS and the `metal` feature) every number is **tessl's own
//!   `pub const`**, re-exported or converted here. Nothing is copied, so a
//!   tessl change reaches the host-side sizing at the next build.
//! - Without tessl the same names hold the literals of [`literal`]. They feed
//!   only code that cannot run there (`MetalBackend::new` returns
//!   `Unsupported` first), and no build that can reach a device ever reads
//!   them. The `literal` module is also compiled into the macOS test build,
//!   where `tests::stub_literals_equal_tessls_constants` pins every literal to
//!   tessl's constant, so the stub's numbers cannot drift from tessl's.
//!
//! The Metal source (`kernels/ojas_backend.metal`) cannot import a Rust
//! constant; `source_pins` pins the three `#define`s that repeat one.

/// The numbers of a build without tessl: one copy, pinned to tessl's constants
/// by `tests::stub_literals_equal_tessls_constants`.
#[cfg(any(test, not(all(target_os = "macos", feature = "metal"))))]
mod literal {
    use std::ops::RangeInclusive;

    pub(crate) const SQ_PIECE: usize = 4096;
    pub(crate) const SQ_OUT_WORDS: usize = 6;
    pub(crate) const CONV1D_WIDTHS: RangeInclusive<usize> = 2..=8;
    pub(crate) const CONV1D_BWD_ROWS: usize = 256;
    pub(crate) const GATED_RMS_BWD_MAX_DIM: usize = 512;
    pub(crate) const GATED_RMS_BWD_ROWS: usize = 64;
}

#[cfg(all(target_os = "macos", feature = "metal"))]
mod tessl_owned {
    use std::ops::RangeInclusive;

    // Elements of one `sq_norm` piece (also of one `ojas_scale_multi`
    // threadgroup, `NORM_CHUNK` in the Metal source), and its result words.
    pub(crate) use tessl::sq_norm::{SQ_OUT_WORDS, SQ_PIECE};

    /// Widths tessl's `qwen35_conv1d_silu*` kernels take.
    pub(crate) const CONV1D_WIDTHS: RangeInclusive<usize> = CONV1D_MIN_WIDTH..=CONV1D_MAX_WIDTH;
    const CONV1D_MIN_WIDTH: usize = *tessl::qwen35::CONV_KERNEL_WIDTHS.start() as usize;
    const CONV1D_MAX_WIDTH: usize = *tessl::qwen35::CONV_KERNEL_WIDTHS.end() as usize;
    /// Flattened rows per weight-gradient block of tessl's conv backward.
    pub(crate) const CONV1D_BWD_ROWS: usize = tessl::qwen35_bwd::CONV_ROWS_PER_BLOCK as usize;
    /// Widest row tessl's gated-norm backward takes.
    pub(crate) const GATED_RMS_BWD_MAX_DIM: usize =
        tessl::qwen35_bwd::GATED_RMS_BWD_MAX_DIM as usize;
    /// Rows per weight-gradient block of tessl's gated-norm backward.
    pub(crate) const GATED_RMS_BWD_ROWS: usize = tessl::qwen35_bwd::GATED_UNITS_PER_BLOCK as usize;
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use tessl_owned::{
    CONV1D_BWD_ROWS, CONV1D_WIDTHS, GATED_RMS_BWD_MAX_DIM, GATED_RMS_BWD_ROWS, SQ_OUT_WORDS,
    SQ_PIECE,
};

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub(crate) use literal::{
    CONV1D_BWD_ROWS, CONV1D_WIDTHS, GATED_RMS_BWD_MAX_DIM, GATED_RMS_BWD_ROWS, SQ_OUT_WORDS,
    SQ_PIECE,
};

/// The Metal source repeats three of these numbers as `#define`s. A Rust
/// constant cannot reach it, so each line is looked up in the source text: a
/// change on either side fails here, not as a wrong reduction on a device.
#[cfg(test)]
mod source_pins {
    use super::SQ_PIECE;
    use crate::backend::GROUP_NORM_THREADS;
    use crate::link::RMS_W_CHUNK;

    const SOURCE: &str = include_str!("../kernels/ojas_backend.metal");

    #[test]
    fn metal_source_defines_equal_the_rust_constants() {
        let norm_chunk = SQ_PIECE;
        let rms_w_chunk = RMS_W_CHUNK;
        let gn_threads = GROUP_NORM_THREADS;
        for line in [
            format!("#define NORM_CHUNK {norm_chunk}u"),
            format!("#define RMS_W_CHUNK {rms_w_chunk}u"),
            format!("#define GN_THREADS {gn_threads}u"),
        ] {
            assert!(
                SOURCE.contains(&line),
                "ojas_backend.metal no longer has `{line}`"
            );
        }
    }
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;
    use crate::backend::{conv1d_part_elems, gated_rms_part_elems};
    use ojas_core::{Conv1dDims, RmsDims};

    /// Only the macOS-with-tessl build reaches a device, and there every
    /// number is tessl's constant. These two keep the conversion honest: the
    /// width range keeps its ends, and the two block sizes stay non-zero (a
    /// zero would make `div_ceil` panic in the sizing code).
    #[test]
    fn tessl_numbers_survive_the_conversion() {
        assert_eq!(*CONV1D_WIDTHS.start(), 2);
        assert_eq!(
            *CONV1D_WIDTHS.end(),
            tessl::qwen35_bwd::CONV_MAX_KW as usize
        );
        assert!(CONV1D_BWD_ROWS > 0 && GATED_RMS_BWD_ROWS > 0);
        assert_eq!(SQ_PIECE, tessl::sq_norm::SQ_PIECE);
        assert_eq!(SQ_OUT_WORDS, tessl::sq_norm::SQ_OUT_WORDS);
    }

    /// The literals a build without tessl uses are tessl's numbers today. They
    /// can never be read where a device exists, so this is their only check.
    #[test]
    fn stub_literals_equal_tessls_constants() {
        assert_eq!(literal::SQ_PIECE, SQ_PIECE);
        assert_eq!(literal::SQ_OUT_WORDS, SQ_OUT_WORDS);
        assert_eq!(literal::CONV1D_WIDTHS, CONV1D_WIDTHS);
        assert_eq!(literal::CONV1D_BWD_ROWS, CONV1D_BWD_ROWS);
        assert_eq!(literal::GATED_RMS_BWD_MAX_DIM, GATED_RMS_BWD_MAX_DIM);
        assert_eq!(literal::GATED_RMS_BWD_ROWS, GATED_RMS_BWD_ROWS);
    }

    /// The host-side scratch formulas equal tessl's own `*_part_len`, which
    /// the device checks the buffer against: a drift would be a refusal
    /// mismatch, so it fails here, before any device is opened.
    #[test]
    fn host_scratch_formulas_equal_tessls_part_lens() {
        for (batch, time, channels, width) in [
            (1usize, 1usize, 1usize, 2usize),
            (1, 255, 12, 4),
            (1, 256, 12, 4),
            (2, 129, 7, 8),
            (3, 1000, 64, 3),
        ] {
            let d = Conv1dDims {
                batch,
                time,
                channels,
                width,
            };
            assert_eq!(
                conv1d_part_elems("t", &d).unwrap(),
                tessl::qwen35_bwd::conv1d_silu_bwd_part_len(
                    batch as u32,
                    time as u32,
                    channels as u32,
                    width as u32
                ),
                "conv {d:?}"
            );
        }
        for (rows, dim) in [(1usize, 1usize), (63, 48), (64, 48), (65, 48), (1000, 512)] {
            let d = RmsDims { rows, dim };
            assert_eq!(
                gated_rms_part_elems("t", &d).unwrap(),
                tessl::qwen35_bwd::gated_rms_norm_bwd_part_len(rows as u32, 1, dim as u32),
                "gated rms {d:?}"
            );
        }
    }
}
