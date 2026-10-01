/// Element type stored in a [`crate::Tensor`].
///
/// v1 parameter storage is `F32`. See `docs/dtype-policy.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    Bf16,
    F16,
    U32,
}

impl DType {
    /// Width of one element, in bytes.
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::U32 => 4,
            DType::Bf16 | DType::F16 => 2,
        }
    }

    /// On-disk tag used by checkpoint v1. The values are part of that schema.
    pub fn tag(self) -> u32 {
        match self {
            DType::F32 => 0,
            DType::Bf16 => 1,
            DType::F16 => 2,
            DType::U32 => 3,
        }
    }

    /// Inverse of [`DType::tag`]. An unknown tag is [`crate::OjasError::OutOfRange`].
    pub fn from_tag(tag: u32) -> Result<Self, crate::OjasError> {
        match tag {
            0 => Ok(DType::F32),
            1 => Ok(DType::Bf16),
            2 => Ok(DType::F16),
            3 => Ok(DType::U32),
            _ => Err(crate::OjasError::OutOfRange {
                op: "DType::from_tag",
                detail: format!("unknown dtype tag {tag}"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_round_trip_and_unknown_tags_are_refused() {
        for dtype in [DType::F32, DType::Bf16, DType::F16, DType::U32] {
            assert_eq!(DType::from_tag(dtype.tag()).unwrap(), dtype);
        }
        for tag in [4u32, 5, 255, 1 << 16, u32::MAX] {
            assert!(DType::from_tag(tag).is_err(), "{tag}");
        }
    }
}
