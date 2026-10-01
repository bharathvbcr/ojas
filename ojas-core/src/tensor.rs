use crate::budget::{Budget, Reservation};
use crate::dtype::DType;
use crate::OjasError;
use std::sync::Arc;

/// Owned bytes plus view metadata.
///
/// `byte_offset` is the start of element 0 inside the allocation. A view that
/// stores the parent buffer and then assumes offset 0 writes at the start of
/// that allocation. [`Tensor::narrow`] adds to the current offset.
/// [`Tensor::view`] takes an absolute offset and does not inherit this one.
///
/// Non-contiguous layouts are legal metadata. Kernel crates may still refuse
/// them with [`OjasError::Shape`] until they implement the strides.
#[derive(Clone, Debug)]
pub struct Tensor {
    storage: Arc<Storage>,
    shape: Box<[usize]>,
    /// Element strides, one per axis. Non-negative only.
    strides: Box<[usize]>,
    dtype: DType,
    byte_offset: usize,
}

#[derive(Debug)]
struct Storage {
    bytes: Vec<u8>,
    _reservation: Reservation,
}

impl Tensor {
    /// Zero-filled contiguous allocation. `byte_offset` is 0.
    pub fn zeros(shape: &[usize], dtype: DType, budget: &Budget) -> Result<Self, OjasError> {
        let strides = contiguous_strides(shape)?;
        let nbytes = contiguous_nbytes(shape, dtype)?;
        let nbytes_u64 = u64_len(nbytes, "Tensor::zeros")?;
        let reservation = budget.try_reserve(nbytes_u64)?;
        let (bytes, reservation) = allocate_zeroed(nbytes, nbytes_u64, reservation, budget)?;
        let tensor = Self {
            storage: Arc::new(Storage {
                bytes,
                _reservation: reservation,
            }),
            shape: shape.to_vec().into_boxed_slice(),
            strides: strides.into_boxed_slice(),
            dtype,
            byte_offset: 0,
        };
        tensor.check_window()?;
        Ok(tensor)
    }

    /// Copy `data` into a new contiguous `F32` allocation.
    pub fn from_f32(data: &[f32], shape: &[usize], budget: &Budget) -> Result<Self, OjasError> {
        let n = num_elements(shape)?;
        if data.len() != n {
            return Err(OjasError::Shape {
                op: "Tensor::from_f32",
                detail: format!("data len {} != shape product {n}", data.len()),
            });
        }
        let mut tensor = Self::zeros(shape, DType::F32, budget)?;
        let bytes = tensor.contiguous_bytes_mut()?;
        let mut chunks = bytes.chunks_exact_mut(4);
        for value in data {
            let slot = chunks.next().ok_or_else(|| OjasError::Shape {
                op: "Tensor::from_f32",
                detail: "f32 window shorter than data".to_string(),
            })?;
            slot.copy_from_slice(&value.to_ne_bytes());
        }
        if !chunks.into_remainder().is_empty() {
            return Err(OjasError::Shape {
                op: "Tensor::from_f32",
                detail: "f32 window is not a multiple of 4 bytes".to_string(),
            });
        }
        Ok(tensor)
    }

    /// Copy `data` into a new contiguous `U32` allocation.
    pub fn from_u32(data: &[u32], shape: &[usize], budget: &Budget) -> Result<Self, OjasError> {
        let n = num_elements(shape)?;
        if data.len() != n {
            return Err(OjasError::Shape {
                op: "Tensor::from_u32",
                detail: format!("data len {} != shape product {n}", data.len()),
            });
        }
        let mut tensor = Self::zeros(shape, DType::U32, budget)?;
        let bytes = tensor.contiguous_bytes_mut()?;
        let mut chunks = bytes.chunks_exact_mut(4);
        for value in data {
            let slot = chunks.next().ok_or_else(|| OjasError::Shape {
                op: "Tensor::from_u32",
                detail: "u32 window shorter than data".to_string(),
            })?;
            slot.copy_from_slice(&value.to_ne_bytes());
        }
        if !chunks.into_remainder().is_empty() {
            return Err(OjasError::Shape {
                op: "Tensor::from_u32",
                detail: "u32 window is not a multiple of 4 bytes".to_string(),
            });
        }
        Ok(tensor)
    }

    /// Contiguous `F32` elements in row-major order.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, OjasError> {
        if self.dtype != DType::F32 {
            return Err(OjasError::Dtype {
                op: "Tensor::to_f32_vec",
                expected: DType::F32,
                got: self.dtype,
            });
        }
        decode_f32(self.contiguous_bytes()?)
    }

    /// Contiguous `U32` elements in row-major order.
    pub fn to_u32_vec(&self) -> Result<Vec<u32>, OjasError> {
        if self.dtype != DType::U32 {
            return Err(OjasError::Dtype {
                op: "Tensor::to_u32_vec",
                expected: DType::U32,
                got: self.dtype,
            });
        }
        decode_u32(self.contiguous_bytes()?)
    }

    /// Replace contiguous `F32` elements. The allocation must be uniquely owned.
    pub fn write_f32(&mut self, values: &[f32]) -> Result<(), OjasError> {
        if self.dtype != DType::F32 {
            return Err(OjasError::Dtype {
                op: "Tensor::write_f32",
                expected: DType::F32,
                got: self.dtype,
            });
        }
        let n = self.num_elements()?;
        if values.len() != n {
            return Err(OjasError::Shape {
                op: "Tensor::write_f32",
                detail: format!("data len {} != shape product {n}", values.len()),
            });
        }
        let bytes = self.contiguous_bytes_mut()?;
        let mut chunks = bytes.chunks_exact_mut(4);
        for value in values {
            let slot = chunks.next().ok_or_else(|| OjasError::Shape {
                op: "Tensor::write_f32",
                detail: "f32 window shorter than data".to_string(),
            })?;
            slot.copy_from_slice(&value.to_ne_bytes());
        }
        if !chunks.into_remainder().is_empty() {
            return Err(OjasError::Shape {
                op: "Tensor::write_f32",
                detail: "f32 window is not a multiple of 4 bytes".to_string(),
            });
        }
        Ok(())
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub fn storage_len(&self) -> usize {
        self.storage.bytes.len()
    }

    pub fn num_elements(&self) -> Result<usize, OjasError> {
        num_elements(&self.shape)
    }

    pub fn is_contiguous(&self) -> Result<bool, OjasError> {
        let expect = contiguous_strides(&self.shape)?;
        Ok(self.strides.as_ref() == expect.as_slice())
    }

    /// Absolute byte offset of `index` (row-major index into the view, not into the allocation).
    pub fn element_byte_offset(&self, index: &[usize]) -> Result<usize, OjasError> {
        if index.len() != self.shape.len() {
            return Err(OjasError::Shape {
                op: "Tensor::element_byte_offset",
                detail: format!(
                    "index rank {} != shape rank {}",
                    index.len(),
                    self.shape.len()
                ),
            });
        }
        let mut elem = 0usize;
        for (axis, (&idx, &dim)) in index.iter().zip(self.shape.iter()).enumerate() {
            if idx >= dim {
                return Err(OjasError::OutOfRange {
                    op: "Tensor::element_byte_offset",
                    detail: format!("index[{axis}]={idx} >= dim {dim}"),
                });
            }
            let stride = *self.strides.get(axis).ok_or_else(|| OjasError::Shape {
                op: "Tensor::element_byte_offset",
                detail: format!("missing stride for axis {axis}"),
            })?;
            let term = idx
                .checked_mul(stride)
                .ok_or_else(|| OjasError::OutOfRange {
                    op: "Tensor::element_byte_offset",
                    detail: format!("index[{axis}] * stride overflows"),
                })?;
            elem = elem
                .checked_add(term)
                .ok_or_else(|| OjasError::OutOfRange {
                    op: "Tensor::element_byte_offset",
                    detail: "element index overflows".to_string(),
                })?;
        }
        let bytes = elem
            .checked_mul(self.dtype.size())
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Tensor::element_byte_offset",
                detail: "element byte offset overflows".to_string(),
            })?;
        self.byte_offset
            .checked_add(bytes)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Tensor::element_byte_offset",
                detail: "absolute byte offset overflows".to_string(),
            })
    }

    /// Contiguous element window, starting at [`Tensor::byte_offset`].
    pub fn contiguous_bytes(&self) -> Result<&[u8], OjasError> {
        if !self.is_contiguous()? {
            return Err(OjasError::Shape {
                op: "Tensor::contiguous_bytes",
                detail: "view is not contiguous".to_string(),
            });
        }
        let nbytes = contiguous_nbytes(&self.shape, self.dtype)?;
        let end = self
            .byte_offset
            .checked_add(nbytes)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Tensor::contiguous_bytes",
                detail: "window end overflows".to_string(),
            })?;
        self.storage
            .bytes
            .get(self.byte_offset..end)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Tensor::contiguous_bytes",
                detail: format!("window {end} exceeds storage {}", self.storage.bytes.len()),
            })
    }

    fn contiguous_bytes_mut(&mut self) -> Result<&mut [u8], OjasError> {
        if !self.is_contiguous()? {
            return Err(OjasError::Shape {
                op: "Tensor::contiguous_bytes_mut",
                detail: "view is not contiguous".to_string(),
            });
        }
        let nbytes = contiguous_nbytes(&self.shape, self.dtype)?;
        let start = self.byte_offset;
        let end = start
            .checked_add(nbytes)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Tensor::contiguous_bytes_mut",
                detail: "window end overflows".to_string(),
            })?;
        let len = self.storage.bytes.len();
        let storage = Arc::get_mut(&mut self.storage).ok_or_else(|| OjasError::Shape {
            op: "Tensor::contiguous_bytes_mut",
            detail: "contiguous write requires a uniquely owned allocation".to_string(),
        })?;
        storage
            .bytes
            .get_mut(start..end)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Tensor::contiguous_bytes_mut",
                detail: format!("window {end} exceeds storage {len}"),
            })
    }

    /// New metadata over the same allocation.
    ///
    /// `byte_offset` is absolute. It does not add to [`Tensor::byte_offset`].
    /// It must be a multiple of the dtype size, or the call is [`OjasError::OutOfRange`].
    pub fn view(
        &self,
        shape: &[usize],
        strides: &[usize],
        byte_offset: usize,
    ) -> Result<Self, OjasError> {
        if shape.len() != strides.len() {
            return Err(OjasError::Shape {
                op: "Tensor::view",
                detail: format!("rank {} != stride rank {}", shape.len(), strides.len()),
            });
        }
        let viewed = Self {
            storage: Arc::clone(&self.storage),
            shape: shape.to_vec().into_boxed_slice(),
            strides: strides.to_vec().into_boxed_slice(),
            dtype: self.dtype,
            byte_offset,
        };
        viewed.check_window()?;
        Ok(viewed)
    }

    /// Same as [`Tensor::view`], but `extra_byte_offset` is added to the current offset.
    pub fn narrow(
        &self,
        extra_byte_offset: usize,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self, OjasError> {
        let byte_offset = self
            .byte_offset
            .checked_add(extra_byte_offset)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Tensor::narrow",
                detail: "byte offset overflow".to_string(),
            })?;
        self.view(shape, strides, byte_offset)
    }

    fn check_window(&self) -> Result<(), OjasError> {
        window_fits(
            self.storage.bytes.len(),
            self.dtype,
            &self.shape,
            &self.strides,
            self.byte_offset,
        )
    }
}

fn allocate_zeroed(
    nbytes: usize,
    nbytes_u64: u64,
    reservation: Reservation,
    budget: &Budget,
) -> Result<(Vec<u8>, Reservation), OjasError> {
    let mut bytes = Vec::new();
    if bytes.try_reserve_exact(nbytes).is_err() {
        // Release first so `live` does not count the refused request.
        drop(reservation);
        return Err(OjasError::CapacityExceeded {
            requested: nbytes_u64,
            cap: budget.cap_bytes(),
            live: budget.live_bytes()?,
        });
    }
    bytes.resize(nbytes, 0);
    Ok((bytes, reservation))
}

fn window_fits(
    storage_len: usize,
    dtype: DType,
    shape: &[usize],
    strides: &[usize],
    byte_offset: usize,
) -> Result<(), OjasError> {
    if shape.len() != strides.len() {
        return Err(OjasError::Shape {
            op: "Tensor::window",
            detail: format!("rank {} != stride rank {}", shape.len(), strides.len()),
        });
    }
    if byte_offset > storage_len {
        return Err(OjasError::OutOfRange {
            op: "Tensor::window",
            detail: format!("byte_offset {byte_offset} > storage {storage_len}"),
        });
    }
    if byte_offset % dtype.size() != 0 {
        return Err(OjasError::OutOfRange {
            op: "Tensor::window",
            detail: format!(
                "byte_offset {byte_offset} is not a multiple of {dtype:?} size {}",
                dtype.size()
            ),
        });
    }
    if shape.contains(&0) {
        return Ok(());
    }
    let mut max_elem = 0usize;
    for (axis, (&dim, &stride)) in shape.iter().zip(strides.iter()).enumerate() {
        let last = dim - 1;
        let term = last
            .checked_mul(stride)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Tensor::window",
                detail: format!("axis {axis} extent overflows"),
            })?;
        max_elem = max_elem
            .checked_add(term)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Tensor::window",
                detail: "view extent overflows".to_string(),
            })?;
    }
    let elem = dtype.size();
    let span = max_elem
        .checked_add(1)
        .and_then(|n| n.checked_mul(elem))
        .ok_or_else(|| OjasError::OutOfRange {
            op: "Tensor::window",
            detail: "view byte span overflows".to_string(),
        })?;
    let end = byte_offset
        .checked_add(span)
        .ok_or_else(|| OjasError::OutOfRange {
            op: "Tensor::window",
            detail: "view end overflows".to_string(),
        })?;
    if end > storage_len {
        return Err(OjasError::OutOfRange {
            op: "Tensor::window",
            detail: format!("view end {end} > storage {storage_len}"),
        });
    }
    Ok(())
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Result<Vec<usize>, OjasError> {
    let mut strides = vec![0; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        // The product over every axis is not a stride; do not refuse it here.
        if i == 0 {
            break;
        }
        acc = acc
            .checked_mul(shape[i])
            .ok_or_else(|| OjasError::OutOfRange {
                op: "contiguous_strides",
                detail: "shape product overflows".to_string(),
            })?;
    }
    Ok(strides)
}

fn num_elements(shape: &[usize]) -> Result<usize, OjasError> {
    if shape.contains(&0) {
        return Ok(0);
    }
    let mut n = 1usize;
    for &dim in shape {
        n = n.checked_mul(dim).ok_or_else(|| OjasError::OutOfRange {
            op: "num_elements",
            detail: "shape product overflows".to_string(),
        })?;
    }
    Ok(n)
}

fn contiguous_nbytes(shape: &[usize], dtype: DType) -> Result<usize, OjasError> {
    num_elements(shape)?
        .checked_mul(dtype.size())
        .ok_or_else(|| OjasError::OutOfRange {
            op: "contiguous_nbytes",
            detail: "byte length overflows".to_string(),
        })
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, OjasError> {
    if bytes.len() % 4 != 0 {
        return Err(OjasError::Shape {
            op: "Tensor::to_f32_vec",
            detail: "f32 window is not a multiple of 4 bytes".to_string(),
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out.push(f32::from_ne_bytes(arr));
    }
    Ok(out)
}

fn decode_u32(bytes: &[u8]) -> Result<Vec<u32>, OjasError> {
    if bytes.len() % 4 != 0 {
        return Err(OjasError::Shape {
            op: "Tensor::to_u32_vec",
            detail: "u32 window is not a multiple of 4 bytes".to_string(),
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out.push(u32::from_ne_bytes(arr));
    }
    Ok(out)
}

fn u64_len(nbytes: usize, op: &'static str) -> Result<u64, OjasError> {
    u64::try_from(nbytes).map_err(|_| OjasError::OutOfRange {
        op,
        detail: "byte length does not fit in u64".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn f32_tensor(n: usize, budget: &Budget) -> Tensor {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Tensor::from_f32(&data, &[n], budget).unwrap()
    }

    #[test]
    fn misaligned_byte_offset_is_refused() {
        let budget = Budget::new(1 << 20);
        let t = f32_tensor(4, &budget);
        for offset in [1usize, 2, 3, 5, 7] {
            let err = t.view(&[1], &[1], offset).unwrap_err();
            assert!(
                matches!(err, OjasError::OutOfRange { .. }),
                "{offset}: {err}"
            );
            let err = t.narrow(offset, &[1], &[1]).unwrap_err();
            assert!(
                matches!(err, OjasError::OutOfRange { .. }),
                "{offset}: {err}"
            );
        }
        let aligned = t.narrow(4, &[3], &[1]).unwrap();
        assert_eq!(aligned.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0]);
        let empty_at_end = t.view(&[0], &[1], 16).unwrap();
        assert_eq!(empty_at_end.to_f32_vec().unwrap(), Vec::<f32>::new());
        assert!(t.view(&[0], &[1], 14).is_err());
    }

    #[test]
    fn is_contiguous_does_not_overflow_on_unused_outer_product() {
        let budget = Budget::new(1 << 20);
        let t = f32_tensor(2, &budget);
        let broadcast = t.view(&[usize::MAX, 2], &[0, 1], 0).unwrap();
        assert!(!broadcast.is_contiguous().unwrap());
        let err = broadcast.contiguous_bytes().unwrap_err();
        assert!(matches!(err, OjasError::Shape { .. }), "{err}");
        assert_eq!(
            broadcast.element_byte_offset(&[usize::MAX - 1, 1]).unwrap(),
            4
        );
    }

    #[test]
    fn zero_dim_num_elements_is_zero_even_with_huge_dims() {
        let budget = Budget::new(1 << 20);
        let t = f32_tensor(2, &budget);
        let empty = t.view(&[usize::MAX, usize::MAX, 0], &[0, 0, 1], 0).unwrap();
        assert_eq!(empty.num_elements().unwrap(), 0);
        assert!(empty.is_contiguous().unwrap());
        assert_eq!(empty.to_f32_vec().unwrap(), Vec::<f32>::new());
        let err = Tensor::zeros(&[0, usize::MAX, usize::MAX], DType::F32, &budget).unwrap_err();
        assert!(matches!(err, OjasError::OutOfRange { .. }), "{err}");
        let zeros = Tensor::zeros(&[usize::MAX, 0, usize::MAX], DType::F32, &budget).unwrap();
        assert_eq!(zeros.strides(), &[0, usize::MAX, 1]);
        let zeros = Tensor::zeros(&[usize::MAX, usize::MAX, 0], DType::F32, &budget).unwrap();
        assert_eq!(zeros.num_elements().unwrap(), 0);
        assert_eq!(zeros.storage_len(), 0);
        assert_eq!(budget.live_bytes().unwrap(), 8);
    }

    #[test]
    fn allocation_failure_reports_live_before_request() {
        let budget = Budget::new(u64::MAX);
        let _held = budget.try_reserve(16).unwrap();
        let elems = isize::MAX as usize / 4 + 1;
        let err = Tensor::zeros(&[elems], DType::F32, &budget).unwrap_err();
        match err {
            OjasError::CapacityExceeded {
                requested,
                cap,
                live,
            } => {
                assert_eq!(requested, (elems * 4) as u64);
                assert_eq!(cap, u64::MAX);
                assert_eq!(live, 16);
            }
            other => panic!("unexpected {other}"),
        }
        assert_eq!(budget.live_bytes().unwrap(), 16);
    }

    #[test]
    fn overflowing_shapes_are_errors_not_panics() {
        let budget = Budget::new(u64::MAX);
        let shapes: [&[usize]; 5] = [
            &[usize::MAX, 2],
            &[usize::MAX / 2 + 1, 2],
            &[usize::MAX / 4 + 1],
            &[1 << 32, 1 << 32],
            &[usize::MAX, usize::MAX, usize::MAX],
        ];
        for shape in shapes {
            for dtype in [DType::F32, DType::Bf16, DType::F16, DType::U32] {
                let err = Tensor::zeros(shape, dtype, &budget).unwrap_err();
                assert!(
                    matches!(
                        err,
                        OjasError::OutOfRange { .. } | OjasError::CapacityExceeded { .. }
                    ),
                    "{shape:?} {dtype:?}: {err}"
                );
            }
        }
        assert_eq!(budget.live_bytes().unwrap(), 0);
        let t = f32_tensor(4, &budget);
        assert!(t.view(&[2], &[usize::MAX], 0).is_err());
        assert!(t.view(&[usize::MAX], &[1], 0).is_err());
        assert!(t.view(&[1], &[1], usize::MAX).is_err());
        assert!(t.narrow(usize::MAX, &[1], &[1]).is_err());
        assert!(t
            .narrow(4, &[1], &[1])
            .unwrap()
            .narrow(usize::MAX - 3, &[1], &[1])
            .is_err());
        assert!(t.view(&[2, 2], &[1], 0).is_err());
        assert!(t.element_byte_offset(&[4]).is_err());
        assert!(t.element_byte_offset(&[0, 0]).is_err());
        assert!(t.element_byte_offset(&[usize::MAX]).is_err());
    }

    #[test]
    fn scalar_and_empty_round_trip() {
        let budget = Budget::new(64);
        let scalar = Tensor::from_f32(&[f32::NAN], &[], &budget).unwrap();
        assert_eq!(scalar.num_elements().unwrap(), 1);
        assert!(scalar.to_f32_vec().unwrap()[0].is_nan());
        let empty = Tensor::from_u32(&[], &[3, 0], &budget).unwrap();
        assert_eq!(empty.to_u32_vec().unwrap(), Vec::<u32>::new());
        assert!(Tensor::from_f32(&[1.0], &[0], &budget).is_err());
        assert!(Tensor::from_u32(&[1, 2], &[3], &budget).is_err());
        let specials = [
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            f32::MIN_POSITIVE / 2.0,
        ];
        let t = Tensor::from_f32(&specials, &[4], &budget).unwrap();
        let back = t.to_f32_vec().unwrap();
        for (a, b) in specials.iter().zip(back.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert!(t.to_u32_vec().is_err());
    }

    #[test]
    fn write_requires_unique_owner_and_matching_len() {
        let budget = Budget::new(1 << 10);
        let mut t = f32_tensor(3, &budget);
        assert!(t.write_f32(&[1.0, 2.0]).is_err());
        let shared = t.clone();
        assert!(t.write_f32(&[1.0, 2.0, 3.0]).is_err());
        drop(shared);
        t.write_f32(&[7.0, 8.0, 9.0]).unwrap();
        assert_eq!(t.to_f32_vec().unwrap(), vec![7.0, 8.0, 9.0]);
        let mut u = Tensor::from_u32(&[1], &[1], &budget).unwrap();
        assert!(u.write_f32(&[1.0]).is_err());
    }

    #[test]
    fn budget_is_released_when_last_view_drops() {
        let budget = Budget::new(64);
        let t = f32_tensor(4, &budget);
        let view = t.narrow(8, &[2], &[1]).unwrap();
        drop(t);
        assert_eq!(budget.live_bytes().unwrap(), 16);
        assert_eq!(view.to_f32_vec().unwrap(), vec![2.0, 3.0]);
        drop(view);
        assert_eq!(budget.live_bytes().unwrap(), 0);
        assert!(Tensor::zeros(&[17], DType::F32, &budget).is_err());
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    fn reference_fits(
        storage_len: usize,
        size: usize,
        shape: &[usize],
        strides: &[usize],
        off: usize,
    ) -> bool {
        if off % size != 0 || off > storage_len {
            return false;
        }
        if shape.contains(&0) {
            return true;
        }
        let mut max_elem: u128 = 0;
        for (&d, &s) in shape.iter().zip(strides) {
            match (d as u128 - 1)
                .checked_mul(s as u128)
                .and_then(|t| max_elem.checked_add(t))
            {
                Some(m) => max_elem = m,
                None => return false,
            }
        }
        (max_elem + 1)
            .checked_mul(size as u128)
            .and_then(|span| span.checked_add(off as u128))
            .is_some_and(|end| end <= storage_len as u128)
    }

    #[test]
    fn randomized_views_match_reference_bounds() {
        let budget = Budget::new(1 << 20);
        let base = f32_tensor(64, &budget);
        let mut rng = SplitMix64(0x0BAD_5EED);
        let edge = [
            0usize,
            1,
            2,
            3,
            63,
            64,
            65,
            usize::MAX / 4,
            usize::MAX - 1,
            usize::MAX,
        ];
        for _ in 0..20_000 {
            let rank = rng.below(4) as usize;
            let pick = |rng: &mut SplitMix64| {
                if rng.below(4) == 0 {
                    edge[rng.below(edge.len() as u64) as usize]
                } else {
                    rng.below(9) as usize
                }
            };
            let shape: Vec<usize> = (0..rank).map(|_| pick(&mut rng)).collect();
            let strides: Vec<usize> = (0..rank).map(|_| pick(&mut rng)).collect();
            let off = if rng.below(2) == 0 {
                pick(&mut rng).wrapping_mul(4)
            } else {
                pick(&mut rng)
            };
            let expect = reference_fits(256, 4, &shape, &strides, off);
            let got = base.view(&shape, &strides, off);
            assert_eq!(got.is_ok(), expect, "{shape:?} {strides:?} {off}");
            if let Ok(v) = got {
                if v.num_elements().unwrap_or(0) > 0 {
                    let last: Vec<usize> = shape.iter().map(|d| d - 1).collect();
                    let at = v.element_byte_offset(&last).unwrap();
                    assert!(at + 4 <= v.storage_len());
                }
                if v.is_contiguous().unwrap_or(false) {
                    let n = v.to_f32_vec().unwrap().len();
                    assert_eq!(n, v.num_elements().unwrap());
                }
            }
        }
    }
}
