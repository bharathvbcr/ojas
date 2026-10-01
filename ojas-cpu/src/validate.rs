use ojas_core::{Budget, DType, OjasError, Reservation, Tensor};

pub(crate) struct F32V {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

pub(crate) fn shape(op: &'static str, detail: impl Into<String>) -> OjasError {
    OjasError::Shape {
        op,
        detail: detail.into(),
    }
}

pub(crate) fn nonfinite(op: &'static str) -> OjasError {
    OjasError::NonFinite { op }
}

pub(crate) fn f32_in(op: &'static str, t: &Tensor) -> Result<F32V, OjasError> {
    if t.dtype() != DType::F32 {
        return Err(OjasError::Dtype {
            op,
            expected: DType::F32,
            got: t.dtype(),
        });
    }
    if !t.is_contiguous()? {
        return Err(shape(op, "non-contiguous view is not supported"));
    }
    let n = t.num_elements()?;
    if n == 0 || t.shape().contains(&0) {
        return Err(shape(op, "empty tensor"));
    }
    let data = t.to_f32_vec()?;
    if data.iter().any(|value| !value.is_finite()) {
        return Err(nonfinite(op));
    }
    Ok(F32V {
        data,
        shape: t.shape().to_vec(),
    })
}

pub(crate) struct U32V {
    pub data: Vec<u32>,
    pub shape: Vec<usize>,
}

pub(crate) fn u32_in(op: &'static str, t: &Tensor) -> Result<U32V, OjasError> {
    if t.dtype() != DType::U32 {
        return Err(OjasError::Dtype {
            op,
            expected: DType::U32,
            got: t.dtype(),
        });
    }
    if !t.is_contiguous()? {
        return Err(shape(op, "non-contiguous view is not supported"));
    }
    let n = t.num_elements()?;
    if n == 0 || t.shape().contains(&0) {
        return Err(shape(op, "empty tensor"));
    }
    let data = t.to_u32_vec()?;
    Ok(U32V {
        data,
        shape: t.shape().to_vec(),
    })
}

pub(crate) fn same_shape(op: &'static str, a: &[usize], b: &[usize]) -> Result<(), OjasError> {
    if a == b {
        Ok(())
    } else {
        Err(shape(op, format!("shape {a:?} does not match {b:?}")))
    }
}

pub(crate) fn product(op: &'static str, dims: &[usize]) -> Result<usize, OjasError> {
    let mut n = 1usize;
    for &dim in dims {
        n = n.checked_mul(dim).ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "shape product overflows".to_string(),
        })?;
    }
    Ok(n)
}

pub(crate) fn flat(
    op: &'static str,
    row: usize,
    col: usize,
    cols: usize,
) -> Result<usize, OjasError> {
    row.checked_mul(cols)
        .and_then(|base| base.checked_add(col))
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "element index overflows".to_string(),
        })
}

pub(crate) fn payload_bytes(op: &'static str, elements: usize) -> Result<u64, OjasError> {
    let bytes = elements
        .checked_mul(DType::F32.size())
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "byte length overflows".to_string(),
        })?;
    u64::try_from(bytes).map_err(|_| OjasError::OutOfRange {
        op,
        detail: "byte length does not fit in u64".to_string(),
    })
}

/// Charge `bytes` against the budget for the duration of the guard.
///
/// In-place steps do not allocate a second tensor. They still reserve this
/// many bytes and release them before returning, so a cap with no headroom
/// refuses the step instead of writing. The numeric scratch is a temporary
/// `Vec` on the process heap.
pub(crate) fn headroom(
    op: &'static str,
    budget: &Budget,
    bytes: u64,
) -> Result<Reservation, OjasError> {
    let _ = op;
    budget.try_reserve(bytes)
}

/// Refuse an output of `elements` f32 values that the budget cannot hold,
/// before its scratch `Vec` is allocated and filled. Releases immediately;
/// [`alloc_f32`] charges the real tensor.
pub(crate) fn room_for(
    op: &'static str,
    budget: &Budget,
    elements: usize,
) -> Result<(), OjasError> {
    drop(budget.try_reserve(payload_bytes(op, elements)?)?);
    Ok(())
}

pub(crate) fn alloc_f32(
    op: &'static str,
    budget: &Budget,
    data: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    if data.iter().any(|value| !value.is_finite()) {
        return Err(nonfinite(op));
    }
    Tensor::from_f32(data, shape, budget)
}

pub(crate) fn get(op: &'static str, data: &[f32], index: usize) -> Result<f32, OjasError> {
    data.get(index)
        .copied()
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: format!("index {index} exceeds length {}", data.len()),
        })
}
