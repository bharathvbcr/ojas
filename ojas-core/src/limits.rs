//! Named sizes and the shape-product implementations.
//!
//! Other crates still spell some of these numbers locally. This module is
//! the owner inside `ojas-core`. [`checked_product`] is also the one op-named
//! product the CPU, wgpu and Metal backends call; the zero-refusing
//! variants in `ojas-cpu/src/train.rs` and `ojas-autograd/src/tape.rs` keep
//! their own error contracts and are not folded here.

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
    checked_product("shape_product", shape)
}

/// Product of `dims` for the op `op`, with no zero short-circuit: a zero
/// anywhere does not hide an overflow among the other factors.
///
/// A rank-0 list is 1. Overflow is [`OjasError::OutOfRange`] carrying `op`
/// and never wraps. This is the one op-named checked product the CPU, wgpu
/// and Metal backends validate sizes with, so a given overflow reads the
/// same on every backend.
pub fn checked_product(op: &'static str, dims: &[usize]) -> Result<usize, OjasError> {
    dims.iter().try_fold(1usize, |n, &dim| {
        n.checked_mul(dim).ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "shape product overflows".to_string(),
        })
    })
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

    /// The backends' size checks call this with their own op name; a zero
    /// factor must not mask an overflow the way `shape_product`'s does.
    #[test]
    fn checked_product_names_the_op_and_has_no_zero_short_circuit() {
        assert_eq!(checked_product("op", &[]).unwrap(), 1);
        assert_eq!(checked_product("op", &[2, 3, 4]).unwrap(), 24);
        assert_eq!(checked_product("op", &[0, usize::MAX]).unwrap(), 0);
        for dims in [&[usize::MAX, 2][..], &[usize::MAX, 2, 0][..]] {
            match checked_product("some_op", dims) {
                Err(OjasError::OutOfRange { op, detail }) => {
                    assert_eq!(op, "some_op", "{dims:?}");
                    assert_eq!(detail, "shape product overflows", "{dims:?}");
                }
                other => panic!("{dims:?}: {other:?}"),
            }
        }
        // shape_product short-circuits the same list to 0.
        assert_eq!(shape_product(&[usize::MAX, 2, 0]).unwrap(), 0);
    }

    #[test]
    fn named_sizes_are_powers_of_two() {
        assert_eq!(KIBIBYTE, 1 << 10);
        assert_eq!(MEBIBYTE, 1 << 20);
        assert_eq!(GIBIBYTE, 1 << 30);
        assert_eq!(CPU_THREAD_CEILING, 1024);
    }
}
