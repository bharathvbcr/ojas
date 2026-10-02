//! Named sizes and the one shape-product implementation.
//!
//! Other crates still spell some of these numbers locally. This module is
//! the owner inside `ojas-core`. Call sites outside this crate are left
//! alone while those crates are being edited elsewhere.

use crate::OjasError;

/// 1024 bytes.
pub const KIBIBYTE: u64 = 1024;

/// 1024 KiB.
pub const MEBIBYTE: u64 = 1024 * KIBIBYTE;

/// 1024 MiB. The capi per-step budget and the checkpoint read cap use this
/// magnitude; those crates still write the literal themselves.
pub const GIBIBYTE: u64 = 1024 * MEBIBYTE;

/// Fixed ceiling for CPU thread advice.
///
/// `ResourcePlan` clamps a known CPU count to this. Refusing
/// `CpuBackend::with_threads` above it is a later change; the constant
/// lives here so that ceiling and the advice share one number.
pub const CPU_THREAD_CEILING: u32 = 1024;

/// Product of `shape`, with a zero axis short-circuiting to 0.
///
/// A rank-0 shape is 1. Overflow is [`OjasError::OutOfRange`] and does not
/// wrap. A zero dimension is 0 even when another dimension would overflow.
pub fn shape_product(shape: &[usize]) -> Result<usize, OjasError> {
    if shape.contains(&0) {
        return Ok(0);
    }
    let mut n = 1usize;
    for &dim in shape {
        n = n.checked_mul(dim).ok_or_else(|| OjasError::OutOfRange {
            op: "shape_product",
            detail: "shape product overflows".to_string(),
        })?;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_matches_the_scalar_and_zero_rules() {
        assert_eq!(shape_product(&[]).unwrap(), 1);
        assert_eq!(shape_product(&[0, usize::MAX]).unwrap(), 0);
        assert_eq!(shape_product(&[2, 3, 4]).unwrap(), 24);
        assert!(matches!(
            shape_product(&[usize::MAX, 2]),
            Err(OjasError::OutOfRange {
                op: "shape_product",
                ..
            })
        ));
    }

    #[test]
    fn named_sizes_are_powers_of_two() {
        assert_eq!(KIBIBYTE, 1 << 10);
        assert_eq!(MEBIBYTE, 1 << 20);
        assert_eq!(GIBIBYTE, 1 << 30);
        assert_eq!(CPU_THREAD_CEILING, 1024);
    }
}
