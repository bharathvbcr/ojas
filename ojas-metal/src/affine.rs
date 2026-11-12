//! The map between ojas's affine code width and tessl's.
//!
//! ojas and tessl each define a two-variant `AffineBits` (`Q4`, `Q8`):
//! `ojas_core::AffineBits` owns the host format and `tessl::dequant::AffineBits` names the kernel
//! entry point. This crate is the only one that sees both, so the mapping lives here, as two
//! exhaustive `match`es with no wildcard arm.
//!
//! Where each guard fires:
//!
//! - A width added to `ojas_core::AffineBits` stops every macOS build of ojas-metal with
//!   `E0004`, including CI's macOS `--no-run` build.
//! - A width added to `tessl::dequant::AffineBits` stops the **next** build of ojas-metal
//!   against that tessl. tessl depends on ojas-core and ojas-io only, not on ojas-metal, so
//!   tessl's own verify cannot catch it; this file is where it surfaces.
//!
//! [`checked_tessl_bits`] is what the device thread calls before it dispatches: it also checks
//! that the round trip returns the same width, that both `bits()` agree, and that tessl accepts
//! ojas's group size, so a drift that keeps the variant names but changes a meaning is refused
//! rather than dispatched. Converging to one type (tessl re-exporting `ojas_core::AffineBits`) is
//! a tessl change and is not needed for this guarantee.

use ojas_core::{AffineBits, AFFINE_GROUP_SIZE};
use tessl::dequant::{AffineBits as TesslBits, DEQUANT_AFFINE_GROUP_SIZES};

/// tessl's width for an ojas width. Exhaustive: a new ojas width does not compile.
pub(crate) fn tessl_bits(bits: AffineBits) -> TesslBits {
    match bits {
        AffineBits::Q4 => TesslBits::Q4,
        AffineBits::Q8 => TesslBits::Q8,
    }
}

/// ojas's width for a tessl width. Exhaustive: a new tessl width does not compile.
pub(crate) fn ojas_bits(bits: TesslBits) -> AffineBits {
    match bits {
        TesslBits::Q4 => AffineBits::Q4,
        TesslBits::Q8 => AffineBits::Q8,
    }
}

/// [`tessl_bits`], after checking that the mapping still means what it says: the round trip is
/// the identity, the two `bits()` agree, and tessl accepts ojas's group size.
pub(crate) fn checked_tessl_bits(bits: AffineBits) -> Result<TesslBits, String> {
    let mapped = tessl_bits(bits);
    if ojas_bits(mapped) != bits {
        return Err(format!(
            "dequant_affine_bf16: ojas {bits:?} maps to tessl {mapped:?}, which maps back to {:?}",
            ojas_bits(mapped)
        ));
    }
    if mapped.bits() != bits.bits() {
        return Err(format!(
            "dequant_affine_bf16: ojas {bits:?} is {} bits, tessl {mapped:?} is {}",
            bits.bits(),
            mapped.bits()
        ));
    }
    if !DEQUANT_AFFINE_GROUP_SIZES.contains(&(AFFINE_GROUP_SIZE as u32)) {
        return Err(format!(
            "dequant_affine_bf16: tessl no longer accepts group size {AFFINE_GROUP_SIZE} \
             (it takes {DEQUANT_AFFINE_GROUP_SIZES:?})"
        ));
    }
    Ok(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_bits_map_round_trips_and_keeps_the_width() {
        for bits in [AffineBits::Q4, AffineBits::Q8] {
            let t = tessl_bits(bits);
            assert_eq!(ojas_bits(t), bits);
            assert_eq!(t.bits(), bits.bits());
            assert_eq!(checked_tessl_bits(bits).unwrap(), t);
        }
        for t in [TesslBits::Q4, TesslBits::Q8] {
            assert_eq!(tessl_bits(ojas_bits(t)), t);
        }
        assert_ne!(tessl_bits(AffineBits::Q4), tessl_bits(AffineBits::Q8));
    }

    #[test]
    fn tessl_accepts_the_ojas_group_size() {
        assert!(DEQUANT_AFFINE_GROUP_SIZES.contains(&(AFFINE_GROUP_SIZE as u32)));
    }

    #[test]
    fn the_kernel_entry_names_the_width_it_is_for() {
        assert!(tessl_bits(AffineBits::Q4).entry().contains("q4"));
        assert!(tessl_bits(AffineBits::Q8).entry().contains("q8"));
    }
}
