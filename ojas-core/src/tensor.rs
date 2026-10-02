use crate::backend::BackendId;
use crate::budget::{Budget, Reservation, Scratch};
use crate::dtype::DType;
use crate::limits::shape_product;
use crate::OjasError;
use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Memory owned by a device backend.
///
/// A backend wraps its own buffer type (an `MTLBuffer`, a `wgpu::Buffer`)
/// and hands it to [`Tensor::from_device`]. Host accessors such as
/// [`Tensor::contiguous_bytes`] refuse a device tensor with
/// [`OjasError::Placement`]; the only way back to host bytes is
/// [`Tensor::to_host`], which calls [`DeviceBuffer::read_bytes`] and is
/// counted in [`device_readbacks`].
pub trait DeviceBuffer: Any + Send + Sync + fmt::Debug {
    /// Backend that owns the memory.
    fn backend(&self) -> BackendId;

    /// Allocation size in bytes. Views are checked against this.
    fn byte_len(&self) -> usize;

    /// Copy `len` bytes starting at `offset` to the host. The range is
    /// already checked against [`DeviceBuffer::byte_len`]. Pending device
    /// work that writes this buffer must finish before the copy.
    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError>;

    /// Downcast hook so a backend can recover its concrete buffer type.
    fn as_any(&self) -> &dyn Any;
}

static READBACKS: AtomicU64 = AtomicU64::new(0);
static READBACK_BYTES: AtomicU64 = AtomicU64::new(0);

/// Process-wide `(calls, bytes)` of device-to-host copies made by
/// [`Tensor::to_host`]. Monotonic. A difference taken around code under test
/// also counts every other thread's readbacks, so it is only exact when
/// nothing else in the process reads back meanwhile (one test per binary, or
/// `--test-threads=1`). To count one caller's readbacks, use
/// [`crate::Budget::device_readbacks`] on the budget that caller's backend
/// charges.
pub fn device_readbacks() -> (u64, u64) {
    (
        READBACKS.load(Ordering::Relaxed),
        READBACK_BYTES.load(Ordering::Relaxed),
    )
}

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
    payload: Payload,
    _reservation: Reservation,
}

#[derive(Debug)]
enum Payload {
    Host(Vec<u8>),
    Device(Arc<dyn DeviceBuffer>),
}

impl Storage {
    fn len(&self) -> usize {
        match &self.payload {
            Payload::Host(bytes) => bytes.len(),
            Payload::Device(buf) => buf.byte_len(),
        }
    }

    fn host(&self, op: &'static str) -> Result<&[u8], OjasError> {
        match &self.payload {
            Payload::Host(bytes) => Ok(bytes),
            Payload::Device(buf) => Err(OjasError::Placement {
                op,
                expected: None,
                found: Some(buf.backend()),
            }),
        }
    }
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
                payload: Payload::Host(bytes),
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

    /// A new contiguous host tensor whose bytes `fill` writes, as
    /// little-endian elements, straight into the tensor's own allocation.
    ///
    /// This is the one-copy path for a loader that reads a tensor from a file
    /// (for example `ojas_io::SafeTensors::read_into`): no intermediate
    /// `Vec` exists beside the tensor. `fill` receives exactly the tensor's
    /// byte length, zeroed. On a big-endian target the elements are swapped
    /// to native order after `fill` returns. An error from `fill` is returned
    /// unchanged and the allocation, with its budget charge, is dropped.
    pub fn from_le_fill<E: From<OjasError>>(
        shape: &[usize],
        dtype: DType,
        budget: &Budget,
        fill: impl FnOnce(&mut [u8]) -> Result<(), E>,
    ) -> Result<Self, E> {
        let mut tensor = Self::zeros(shape, dtype, budget)?;
        let bytes = tensor.contiguous_bytes_mut()?;
        fill(bytes)?;
        if cfg!(target_endian = "big") {
            for element in bytes.chunks_exact_mut(dtype.size()) {
                element.reverse();
            }
        }
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
        write_ne_f32(bytes, data, "Tensor::from_f32")?;
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
        write_ne_u32(bytes, data, "Tensor::from_u32")?;
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
        let bytes = self.writable_f32_bytes(values.len())?;
        write_ne_f32(bytes, values, "Tensor::write_f32")?;
        Ok(())
    }

    /// Whether [`Tensor::write_f32`] of `len` values would be accepted, without
    /// writing. It runs the same checks on the same path: `F32`, `len` equal to
    /// the element count, a contiguous host view, and a uniquely owned
    /// allocation.
    ///
    /// A caller that must update several tensors all-or-nothing checks every
    /// target first. Because this takes `&mut self`, no new handle to the
    /// allocation can appear between the check and the write.
    pub fn ensure_writable_f32(&mut self, len: usize) -> Result<(), OjasError> {
        self.writable_f32_bytes(len).map(|_| ())
    }

    fn writable_f32_bytes(&mut self, len: usize) -> Result<&mut [u8], OjasError> {
        if self.dtype != DType::F32 {
            return Err(OjasError::Dtype {
                op: "Tensor::write_f32",
                expected: DType::F32,
                got: self.dtype,
            });
        }
        let n = self.num_elements()?;
        if len != n {
            return Err(OjasError::Shape {
                op: "Tensor::write_f32",
                detail: format!("data len {len} != shape product {n}"),
            });
        }
        self.contiguous_bytes_mut()
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
        self.storage.len()
    }

    /// Wrap a fresh device allocation. The whole allocation
    /// (`buffer.byte_len()`) is charged to `budget` until the last view
    /// drops. The view is contiguous from byte 0 and must fit in the
    /// allocation.
    ///
    /// Every call charges. Wrapping one buffer twice (two clones of the same
    /// `Arc`) charges it twice. Share a wrapped buffer with [`Tensor::clone`]
    /// or [`Tensor::view`]; wrap bytes that are already reserved with
    /// [`Tensor::from_device_reserved`].
    pub fn from_device(
        buffer: Arc<dyn DeviceBuffer>,
        shape: &[usize],
        dtype: DType,
        budget: &Budget,
    ) -> Result<Self, OjasError> {
        let nbytes_u64 = u64_len(buffer.byte_len(), "Tensor::from_device")?;
        let reservation = budget.try_reserve(nbytes_u64)?;
        Self::from_device_reserved(buffer, shape, dtype, reservation)
    }

    /// Wrap device memory whose bytes the caller already reserved, for
    /// example before allocating on the device. Takes ownership of
    /// `reservation` and does not charge again; the reservation is released
    /// when the last view drops, or immediately if this call fails.
    ///
    /// The reservation must be exactly `buffer.byte_len()` bytes, or the
    /// call is [`OjasError::OutOfRange`]. This is the constructor to re-wrap
    /// a buffer without the second charge that [`Tensor::from_device`] makes.
    pub fn from_device_reserved(
        buffer: Arc<dyn DeviceBuffer>,
        shape: &[usize],
        dtype: DType,
        reservation: Reservation,
    ) -> Result<Self, OjasError> {
        let nbytes_u64 = u64_len(buffer.byte_len(), "Tensor::from_device_reserved")?;
        if reservation.bytes() != nbytes_u64 {
            return Err(OjasError::OutOfRange {
                op: "Tensor::from_device_reserved",
                detail: format!(
                    "reservation of {} bytes != device allocation {nbytes_u64}",
                    reservation.bytes()
                ),
            });
        }
        let strides = contiguous_strides(shape)?;
        let tensor = Self {
            storage: Arc::new(Storage {
                payload: Payload::Device(buffer),
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

    /// Host tensor that takes the bytes and the reservation from `scratch`.
    ///
    /// The scratch is not copied and its budget is not charged again. The
    /// reservation moves into this tensor and is released when the last view
    /// drops. A length that is not the contiguous size of `shape` is
    /// [`OjasError::Shape`]; the scratch is dropped and its charge released.
    /// This constructor always builds host memory. Device buffers stay on
    /// [`Tensor::from_device`] and [`Tensor::from_device_reserved`].
    pub fn from_scratch(
        scratch: Scratch<u8>,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Self, OjasError> {
        let expected = contiguous_nbytes(shape, dtype)?;
        let (data, reservation) = scratch.into_raw();
        if data.len() != expected {
            let got = data.len();
            drop(data);
            drop(reservation);
            return Err(OjasError::Shape {
                op: "Tensor::from_scratch",
                detail: format!("scratch len {got} != contiguous bytes {expected}"),
            });
        }
        let nbytes_u64 = u64_len(expected, "Tensor::from_scratch")?;
        if reservation.bytes() != nbytes_u64 {
            let held = reservation.bytes();
            drop(data);
            drop(reservation);
            return Err(OjasError::OutOfRange {
                op: "Tensor::from_scratch",
                detail: format!("reservation of {held} bytes != scratch length {nbytes_u64}"),
            });
        }
        let strides = contiguous_strides(shape)?;
        let tensor = Self {
            storage: Arc::new(Storage {
                payload: Payload::Host(data),
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

    /// Backend that owns the memory, or `None` for host memory.
    pub fn device(&self) -> Option<BackendId> {
        match &self.storage.payload {
            Payload::Host(_) => None,
            Payload::Device(buf) => Some(buf.backend()),
        }
    }

    /// The device allocation behind this view, or `None` for host memory.
    pub fn device_buffer(&self) -> Option<&Arc<dyn DeviceBuffer>> {
        match &self.storage.payload {
            Payload::Host(_) => None,
            Payload::Device(buf) => Some(buf),
        }
    }

    /// Exclusive access to the device allocation, for an in-place update.
    ///
    /// Succeeds only when this tensor is the sole owner of its storage and
    /// the storage holds the only `Arc` to the device buffer, so no clone,
    /// view or outside handle can observe the write. A host tensor is
    /// [`OjasError::Placement`]; a shared device tensor is
    /// [`OjasError::Shape`] naming the sharing. Downcast with
    /// [`DeviceBuffer::as_any`].
    pub fn device_buffer_mut(&mut self) -> Result<&mut dyn DeviceBuffer, OjasError> {
        const OP: &str = "Tensor::device_buffer_mut";
        if let Payload::Host(_) = &self.storage.payload {
            return Err(OjasError::Placement {
                op: OP,
                expected: None,
                found: None,
            });
        }
        let storage = Arc::get_mut(&mut self.storage).ok_or_else(|| OjasError::Shape {
            op: OP,
            detail: "device storage is shared with another tensor view".to_string(),
        })?;
        match &mut storage.payload {
            Payload::Host(_) => Err(OjasError::Placement {
                op: OP,
                expected: None,
                found: None,
            }),
            Payload::Device(buf) => match Arc::get_mut(buf) {
                Some(buf) => Ok(buf),
                None => Err(OjasError::Shape {
                    op: OP,
                    detail: "device buffer is shared with another Arc<dyn DeviceBuffer>"
                        .to_string(),
                }),
            },
        }
    }

    /// Host copy with the same shape. A host tensor is returned as a cheap
    /// clone that shares storage. A device tensor is read back and counted
    /// in [`device_readbacks`]. A contiguous view copies only its window; a
    /// strided one copies the whole allocation and keeps its strides.
    pub fn to_host(&self, budget: &Budget) -> Result<Self, OjasError> {
        let buf = match &self.storage.payload {
            Payload::Host(_) => return Ok(self.clone()),
            Payload::Device(buf) => buf,
        };
        let (offset, len, strides, byte_offset) = if self.is_contiguous()? {
            let len = contiguous_nbytes(&self.shape, self.dtype)?;
            let strides = self.strides.to_vec();
            (self.byte_offset, len, strides, 0)
        } else {
            (0, buf.byte_len(), self.strides.to_vec(), self.byte_offset)
        };
        let end = offset
            .checked_add(len)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Tensor::to_host",
                detail: "window end overflows".to_string(),
            })?;
        if end > buf.byte_len() {
            return Err(OjasError::OutOfRange {
                op: "Tensor::to_host",
                detail: format!("window {end} exceeds device storage {}", buf.byte_len()),
            });
        }
        let reservation = budget.try_reserve(u64_len(len, "Tensor::to_host")?)?;
        let bytes = buf.read_bytes(offset, len)?;
        if bytes.len() != len {
            return Err(OjasError::Backend {
                id: buf.backend(),
                detail: format!("read_bytes returned {} bytes, asked {len}", bytes.len()),
            });
        }
        READBACKS.fetch_add(1, Ordering::Relaxed);
        READBACK_BYTES.fetch_add(len as u64, Ordering::Relaxed);
        budget.record_readback(len as u64);
        let host = Self {
            storage: Arc::new(Storage {
                payload: Payload::Host(bytes),
                _reservation: reservation,
            }),
            shape: self.shape.clone(),
            strides: strides.into_boxed_slice(),
            dtype: self.dtype,
            byte_offset,
        };
        host.check_window()?;
        Ok(host)
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
        // Placement before layout: a strided device view reports where it lives.
        let bytes = self.storage.host("Tensor::contiguous_bytes")?;
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
        bytes
            .get(self.byte_offset..end)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Tensor::contiguous_bytes",
                detail: format!("window {end} exceeds storage {}", bytes.len()),
            })
    }

    fn contiguous_bytes_mut(&mut self) -> Result<&mut [u8], OjasError> {
        // Placement before layout and ownership: a strided or shared device
        // tensor reports where it lives, not that it is strided or shared.
        self.storage.host("Tensor::contiguous_bytes_mut")?;
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
        let len = self.storage.len();
        let storage = Arc::get_mut(&mut self.storage).ok_or_else(|| OjasError::Shape {
            op: "Tensor::contiguous_bytes_mut",
            detail: "contiguous write requires a uniquely owned allocation".to_string(),
        })?;
        let bytes = match &mut storage.payload {
            Payload::Host(bytes) => bytes,
            Payload::Device(buf) => {
                return Err(OjasError::Placement {
                    op: "Tensor::contiguous_bytes_mut",
                    expected: None,
                    found: Some(buf.backend()),
                })
            }
        };
        bytes
            .get_mut(start..end)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Tensor::contiguous_bytes_mut",
                detail: format!("window {end} exceeds storage {len}"),
            })
    }

    /// The same elements, in the same row-major order, viewed as `shape`.
    ///
    /// No copy: the result shares this tensor's storage and byte offset, and
    /// has contiguous strides. A non-contiguous view, or a `shape` whose
    /// element count differs, is [`OjasError::Shape`]; a `shape` whose
    /// element count overflows is [`OjasError::OutOfRange`].
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, OjasError> {
        const OP: &str = "Tensor::reshape";
        let want = num_elements(shape)?;
        let have = self.num_elements()?;
        if have != want {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!(
                    "cannot view {:?} ({have} elements) as {shape:?}",
                    self.shape
                ),
            });
        }
        if !self.is_contiguous()? {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!("view {:?} is not contiguous", self.shape),
            });
        }
        self.view(shape, &contiguous_strides(shape)?, self.byte_offset)
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
            self.storage.len(),
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
    if !byte_offset.is_multiple_of(dtype.size()) {
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
    shape_product(shape)
}

fn contiguous_nbytes(shape: &[usize], dtype: DType) -> Result<usize, OjasError> {
    num_elements(shape)?
        .checked_mul(dtype.size())
        .ok_or_else(|| OjasError::OutOfRange {
            op: "contiguous_nbytes",
            detail: "byte length overflows".to_string(),
        })
}

fn write_ne_f32(bytes: &mut [u8], values: &[f32], op: &'static str) -> Result<(), OjasError> {
    write_ne_words(bytes, values, op, |slot, value| {
        *slot = value.to_ne_bytes();
    })
}

fn write_ne_u32(bytes: &mut [u8], values: &[u32], op: &'static str) -> Result<(), OjasError> {
    write_ne_words(bytes, values, op, |slot, value| {
        *slot = value.to_ne_bytes();
    })
}

fn write_ne_words<T>(
    bytes: &mut [u8],
    values: &[T],
    op: &'static str,
    write: impl Fn(&mut [u8; 4], &T),
) -> Result<(), OjasError> {
    let (chunks, rest) = bytes.as_chunks_mut::<4>();
    if !rest.is_empty() {
        return Err(OjasError::Shape {
            op,
            detail: "window is not a multiple of 4 bytes".to_string(),
        });
    }
    if chunks.len() < values.len() {
        return Err(OjasError::Shape {
            op,
            detail: "window shorter than data".to_string(),
        });
    }
    for (slot, value) in chunks.iter_mut().zip(values) {
        write(slot, value);
    }
    Ok(())
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, OjasError> {
    decode_ne_words(
        bytes,
        "Tensor::to_f32_vec",
        "f32 window is not a multiple of 4 bytes",
        f32::from_ne_bytes,
    )
}

fn decode_u32(bytes: &[u8]) -> Result<Vec<u32>, OjasError> {
    decode_ne_words(
        bytes,
        "Tensor::to_u32_vec",
        "u32 window is not a multiple of 4 bytes",
        u32::from_ne_bytes,
    )
}

/// Every CPU op copies its inputs out through here, so this must stay a
/// straight copy. `collect` over a mapped slice iterator knows its exact
/// length and writes without a per-element capacity check, so LLVM emits a
/// vector copy loop. A `with_capacity` + `push` loop keeps that check in the
/// loop body and copies one element at a time; a zero-filled `vec!` plus a
/// zip pays an extra zeroing pass. `tests/bench_decode.rs` measures it.
fn decode_ne_words<T>(
    bytes: &[u8],
    op: &'static str,
    detail: &'static str,
    decode: impl Fn([u8; 4]) -> T,
) -> Result<Vec<T>, OjasError> {
    let (chunks, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(OjasError::Shape {
            op,
            detail: detail.to_string(),
        });
    }
    Ok(chunks.iter().map(|chunk| decode(*chunk)).collect())
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

    /// The public accessors only ever pass a window of `n * 4` bytes, so the
    /// ragged-window refusal is reachable only through the private decoders.
    #[test]
    fn decoders_refuse_a_ragged_window_with_their_exact_message() {
        for len in [1usize, 2, 3, 5, 6, 7, 13] {
            let bytes = vec![0xA5u8; len];
            match decode_f32(&bytes).unwrap_err() {
                OjasError::Shape { op, detail } => {
                    assert_eq!(op, "Tensor::to_f32_vec");
                    assert_eq!(detail, "f32 window is not a multiple of 4 bytes");
                }
                other => panic!("len {len}: unexpected {other}"),
            }
            match decode_u32(&bytes).unwrap_err() {
                OjasError::Shape { op, detail } => {
                    assert_eq!(op, "Tensor::to_u32_vec");
                    assert_eq!(detail, "u32 window is not a multiple of 4 bytes");
                }
                other => panic!("len {len}: unexpected {other}"),
            }
        }
        assert!(decode_f32(&[]).unwrap().is_empty());
        assert!(decode_u32(&[]).unwrap().is_empty());
        let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(
            decode_u32(&bytes).unwrap(),
            [
                u32::from_ne_bytes([1, 2, 3, 4]),
                u32::from_ne_bytes([5, 6, 7, 8])
            ]
        );
        let got: Vec<u32> = decode_f32(&bytes)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(got, decode_u32(&bytes).unwrap());
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
    fn from_le_fill_writes_in_place_and_drops_on_error() {
        let budget = Budget::new(1 << 10);
        let values = [1.5f32, -0.0, f32::MIN_POSITIVE, 3.0e38];
        let le: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut seen = 0usize;
        let t = Tensor::from_le_fill(&[2, 2], DType::F32, &budget, |bytes| {
            seen = bytes.len();
            assert!(bytes.iter().all(|&b| b == 0), "fill buffer not zeroed");
            bytes.copy_from_slice(&le);
            Ok::<(), OjasError>(())
        })
        .unwrap();
        assert_eq!(seen, 16);
        let got: Vec<u32> = t
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let want: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, want);
        // Exactly the tensor's bytes are charged: no second copy exists.
        assert_eq!(budget.live_bytes().unwrap(), 16);

        let ids = Tensor::from_le_fill(&[2], DType::U32, &budget, |bytes| {
            bytes.copy_from_slice(&[7, 0, 0, 0, 0, 1, 0, 0]);
            Ok::<(), OjasError>(())
        })
        .unwrap();
        assert_eq!(ids.to_u32_vec().unwrap(), [7, 256]);
        drop((t, ids));
        assert_eq!(budget.live_bytes().unwrap(), 0);

        #[derive(Debug, PartialEq)]
        enum Load {
            Core,
            Short,
        }
        impl From<OjasError> for Load {
            fn from(_: OjasError) -> Self {
                Load::Core
            }
        }
        let failed = Tensor::from_le_fill(&[4], DType::F32, &budget, |_| Err(Load::Short));
        assert_eq!(failed.unwrap_err(), Load::Short);
        assert_eq!(
            budget.live_bytes().unwrap(),
            0,
            "a failed fill kept its charge"
        );
        // Over budget: the core error converts, and fill never runs.
        let over = Tensor::from_le_fill(&[1 << 20], DType::F32, &budget, |_| -> Result<(), Load> {
            panic!("fill ran without a reservation")
        });
        assert_eq!(over.unwrap_err(), Load::Core);
    }

    #[test]
    fn reshape_is_a_shared_contiguous_view_or_a_refusal() {
        let budget = Budget::new(1 << 10);
        let values: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::from_f32(&values, &[2, 3, 4], &budget).unwrap();
        let live = budget.live_bytes().unwrap();

        let r = t.reshape(&[6, 4]).unwrap();
        assert_eq!(r.shape(), &[6, 4]);
        assert_eq!(r.strides(), &[4, 1]);
        assert_eq!(r.to_f32_vec().unwrap(), values);
        assert_eq!(budget.live_bytes().unwrap(), live, "reshape copied");
        assert_eq!(t.reshape(&[24]).unwrap().to_f32_vec().unwrap(), values);
        assert_eq!(t.reshape(&[1, 24, 1]).unwrap().strides(), &[24, 1, 1]);

        // A contiguous window keeps its byte offset.
        let row = t.narrow(4 * 12, &[3, 4], &[4, 1]).unwrap();
        let flat = row.reshape(&[12]).unwrap();
        assert_eq!(flat.byte_offset(), 48);
        assert_eq!(flat.to_f32_vec().unwrap(), values[12..].to_vec());

        let refused = |r: Result<Tensor, OjasError>, want: &str| match r {
            Err(OjasError::Shape { op, detail }) => {
                assert_eq!(op, "Tensor::reshape");
                assert!(detail.contains(want), "{detail}");
            }
            other => panic!("expected Shape ({want}), got {other:?}"),
        };
        refused(t.reshape(&[5, 5]), "elements");
        refused(t.reshape(&[]), "elements");
        refused(t.reshape(&[2, 0, 4]), "elements");
        // A transposed view of the same storage is not contiguous.
        let transposed = t.view(&[4, 6], &[1, 4], 0).unwrap();
        refused(transposed.reshape(&[24]), "not contiguous");
        assert!(matches!(
            t.reshape(&[usize::MAX, 2]),
            Err(OjasError::OutOfRange { .. })
        ));
    }

    #[test]
    fn ensure_writable_agrees_with_write_and_never_writes() {
        let budget = Budget::new(1 << 10);
        let mut t = f32_tensor(3, &budget);
        let before = t.to_f32_vec().unwrap();
        // Wrong length, shared allocation, narrowed view of a live parent,
        // and a non-F32 tensor: each is refused by both calls.
        assert!(t.ensure_writable_f32(2).is_err());
        let shared = t.clone();
        assert!(t.ensure_writable_f32(3).is_err());
        assert!(t.write_f32(&[0.0; 3]).is_err());
        drop(shared);
        let parent = f32_tensor(4, &budget);
        let mut view = parent.narrow(4, &[2], &[1]).unwrap();
        assert!(view.ensure_writable_f32(2).is_err());
        assert!(view.write_f32(&[0.0; 2]).is_err());
        let mut u = Tensor::from_u32(&[1], &[1], &budget).unwrap();
        assert!(u.ensure_writable_f32(1).is_err());
        // Accepted: and the check alone changes nothing.
        t.ensure_writable_f32(3).unwrap();
        assert_eq!(t.to_f32_vec().unwrap(), before);
        t.write_f32(&[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(t.to_f32_vec().unwrap(), vec![4.0, 5.0, 6.0]);
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
        if !off.is_multiple_of(size) || off > storage_len {
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

    #[derive(Debug)]
    struct FakeDevice {
        bytes: Vec<u8>,
        short_read: bool,
        reads: AtomicU64,
    }

    impl DeviceBuffer for FakeDevice {
        fn backend(&self) -> BackendId {
            BackendId::Wgpu
        }
        fn byte_len(&self) -> usize {
            self.bytes.len()
        }
        fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            let mut out = self.bytes[offset..offset + len].to_vec();
            if self.short_read {
                out.pop();
            }
            Ok(out)
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn fake(values: &[f32], short_read: bool) -> Arc<FakeDevice> {
        let bytes = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        Arc::new(FakeDevice {
            bytes,
            short_read,
            reads: AtomicU64::new(0),
        })
    }

    #[test]
    fn device_tensor_refuses_every_host_accessor() {
        let budget = Budget::new(1 << 10);
        let mut t =
            Tensor::from_device(fake(&[1.0, 2.0], false), &[2], DType::F32, &budget).unwrap();
        assert_eq!(t.device(), Some(BackendId::Wgpu));
        let placement = |err: OjasError| {
            matches!(
                err,
                OjasError::Placement {
                    expected: None,
                    found: Some(BackendId::Wgpu),
                    ..
                }
            )
        };
        assert!(placement(t.contiguous_bytes().unwrap_err()));
        assert!(placement(t.to_f32_vec().unwrap_err()));
        assert!(placement(t.write_f32(&[3.0, 4.0]).unwrap_err()));
        // A shared device tensor still reports placement, not sharing.
        let alias = t.clone();
        assert!(placement(t.write_f32(&[3.0, 4.0]).unwrap_err()));
        drop(alias);
    }

    #[test]
    fn to_host_round_trips_contiguous_and_narrowed_views() {
        let budget = Budget::new(1 << 10);
        let buf = fake(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], false);
        let t = Tensor::from_device(buf.clone(), &[2, 3], DType::F32, &budget).unwrap();
        assert_eq!(
            t.to_host(&budget).unwrap().to_f32_vec().unwrap(),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
        let row1 = t.narrow(12, &[3], &[1]).unwrap();
        let host = row1.to_host(&budget).unwrap();
        assert_eq!(host.device(), None);
        assert_eq!(host.byte_offset(), 0);
        assert_eq!(
            host.storage_len(),
            12,
            "contiguous view copies its window only"
        );
        assert_eq!(host.to_f32_vec().unwrap(), [3.0, 4.0, 5.0]);
        let col = t.view(&[2], &[3], 4).unwrap();
        let host = col.to_host(&budget).unwrap();
        assert_eq!(host.strides(), [3]);
        assert_eq!(host.element_byte_offset(&[1]).unwrap(), 16);
        assert_eq!(buf.reads.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn device_memory_is_charged_and_released() {
        let budget = Budget::new(24);
        let t = Tensor::from_device(fake(&[0.0; 6], false), &[6], DType::F32, &budget).unwrap();
        assert_eq!(budget.live_bytes().unwrap(), 24);
        assert!(matches!(
            Tensor::from_device(fake(&[0.0], false), &[1], DType::F32, &budget),
            Err(OjasError::CapacityExceeded { .. })
        ));
        // A readback is a host allocation and must fit too.
        assert!(matches!(
            t.to_host(&budget),
            Err(OjasError::CapacityExceeded { .. })
        ));
        drop(t);
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    #[test]
    fn device_view_larger_than_allocation_is_refused() {
        let budget = Budget::new(1 << 10);
        assert!(matches!(
            Tensor::from_device(fake(&[0.0; 3], false), &[4], DType::F32, &budget),
            Err(OjasError::OutOfRange { .. })
        ));
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    #[test]
    fn short_device_read_is_a_backend_error_not_a_torn_tensor() {
        let budget = Budget::new(1 << 10);
        let t = Tensor::from_device(fake(&[1.0, 2.0], true), &[2], DType::F32, &budget).unwrap();
        assert!(matches!(
            t.to_host(&budget),
            Err(OjasError::Backend {
                id: BackendId::Wgpu,
                ..
            })
        ));
    }

    #[test]
    fn budget_readbacks_count_only_their_own_tree_across_threads() {
        // Two sibling budgets under one root, read back from eight threads at
        // once. Each sibling sees exactly its own copies; the root sees both;
        // a host clone and a failed short read are not counted.
        let root = Budget::new(1 << 20);
        let a = root.child(1 << 19);
        let b = root.child(1 << 19);
        let host = f32_tensor(4, &a);
        host.to_host(&a).unwrap();
        let bad = Tensor::from_device(fake(&[1.0, 2.0], true), &[2], DType::F32, &b).unwrap();
        assert!(bad.to_host(&b).is_err());
        assert_eq!(a.device_readbacks(), (0, 0));
        assert_eq!(b.device_readbacks(), (0, 0));

        const PER_THREAD: u64 = 25;
        std::thread::scope(|s| {
            for i in 0..8 {
                let target = if i % 2 == 0 { a.clone() } else { b.clone() };
                s.spawn(move || {
                    let t = Tensor::from_device(
                        fake(&[1.0, 2.0, 3.0], false),
                        &[3],
                        DType::F32,
                        &target,
                    )
                    .unwrap();
                    for _ in 0..PER_THREAD {
                        drop(t.to_host(&target).unwrap());
                    }
                });
            }
        });
        let each = 4 * PER_THREAD;
        assert_eq!(a.device_readbacks(), (each, each * 12));
        assert_eq!(b.device_readbacks(), (each, each * 12));
        assert_eq!(root.device_readbacks(), (2 * each, 2 * each * 12));
    }

    fn shared_error(err: OjasError) -> bool {
        matches!(err, OjasError::Shape { ref detail, .. } if detail.contains("shared"))
    }

    #[test]
    fn device_buffer_mut_requires_sole_ownership_of_storage_and_buffer() {
        let budget = Budget::new(1 << 10);
        let mut t =
            Tensor::from_device(fake(&[1.0, 2.0], false), &[2], DType::F32, &budget).unwrap();
        let alias = t.clone();
        assert!(shared_error(t.device_buffer_mut().unwrap_err()));
        let view = alias.narrow(4, &[1], &[1]).unwrap();
        drop(alias);
        assert!(shared_error(t.device_buffer_mut().unwrap_err()));
        drop(view);
        let buf = t.device_buffer_mut().unwrap();
        assert_eq!(buf.backend(), BackendId::Wgpu);
        assert!(buf.as_any().downcast_ref::<FakeDevice>().is_some());

        let outside: Arc<dyn DeviceBuffer> = fake(&[1.0], false);
        let held = Arc::clone(&outside);
        let mut t = Tensor::from_device(outside, &[1], DType::F32, &budget).unwrap();
        assert!(shared_error(t.device_buffer_mut().unwrap_err()));
        drop(held);
        assert_eq!(t.device_buffer_mut().unwrap().backend(), BackendId::Wgpu);

        let mut host = f32_tensor(2, &budget);
        assert!(matches!(
            host.device_buffer_mut().unwrap_err(),
            OjasError::Placement { found: None, .. }
        ));
    }

    #[test]
    fn from_device_charges_per_call_and_reserved_charges_once() {
        let budget = Budget::new(1 << 10);
        let buf: Arc<dyn DeviceBuffer> = fake(&[0.0; 4], false);
        let a = Tensor::from_device(Arc::clone(&buf), &[4], DType::F32, &budget).unwrap();
        let b = Tensor::from_device(Arc::clone(&buf), &[4], DType::F32, &budget).unwrap();
        assert_eq!(budget.live_bytes().unwrap(), 32, "two wraps charge twice");
        drop((a, b));
        assert_eq!(budget.live_bytes().unwrap(), 0);

        let reservation = budget.try_reserve(16).unwrap();
        assert_eq!(budget.live_bytes().unwrap(), 16);
        let t =
            Tensor::from_device_reserved(Arc::clone(&buf), &[4], DType::F32, reservation).unwrap();
        assert_eq!(budget.live_bytes().unwrap(), 16, "no second charge");
        drop(t);
        assert_eq!(budget.live_bytes().unwrap(), 0);

        // The view does not fit; the reservation is released, not leaked.
        let reservation = budget.try_reserve(16).unwrap();
        assert!(matches!(
            Tensor::from_device_reserved(Arc::clone(&buf), &[5], DType::F32, reservation),
            Err(OjasError::OutOfRange { .. })
        ));
        assert_eq!(budget.live_bytes().unwrap(), 0);

        // A reservation that does not match the allocation cannot under-charge it.
        let reservation = budget.try_reserve(4).unwrap();
        assert!(matches!(
            Tensor::from_device_reserved(buf, &[1], DType::F32, reservation),
            Err(OjasError::OutOfRange { .. })
        ));
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    #[test]
    fn strided_device_view_reports_placement_not_layout() {
        let budget = Budget::new(1 << 10);
        let t = Tensor::from_device(fake(&[0.0; 6], false), &[2, 3], DType::F32, &budget).unwrap();
        let mut col = t.view(&[2], &[3], 4).unwrap();
        assert!(!col.is_contiguous().unwrap());
        let placement = |err: OjasError| {
            matches!(
                err,
                OjasError::Placement {
                    found: Some(BackendId::Wgpu),
                    ..
                }
            )
        };
        assert!(placement(col.to_f32_vec().unwrap_err()));
        assert!(placement(col.contiguous_bytes().unwrap_err()));
        assert!(placement(col.write_f32(&[1.0, 2.0]).unwrap_err()));
        let ids = Tensor::from_device(fake(&[0.0; 4], false), &[2, 2], DType::U32, &budget)
            .unwrap()
            .view(&[2], &[2], 0)
            .unwrap();
        assert!(placement(ids.to_u32_vec().unwrap_err()));
    }

    #[test]
    fn host_to_host_is_a_shared_clone_and_not_a_readback() {
        let budget = Budget::new(1 << 10);
        let t = f32_tensor(4, &budget);
        let h = t.to_host(&budget).unwrap();
        assert_eq!(budget.live_bytes().unwrap(), 16, "no second allocation");
        assert_eq!(h.to_f32_vec().unwrap(), t.to_f32_vec().unwrap());
    }

    #[test]
    fn from_scratch_keeps_one_charge_and_stays_on_the_host() {
        use crate::Scratch;
        let budget = Budget::new(64);
        let mut scratch = Scratch::<u8>::try_alloc(8, &budget).unwrap();
        scratch
            .as_mut_slice()
            .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(budget.live_bytes().unwrap(), 8);
        let tensor = Tensor::from_scratch(scratch, &[2], DType::F32).unwrap();
        assert_eq!(
            budget.live_bytes().unwrap(),
            8,
            "the reservation moved, it was not charged again"
        );
        assert_eq!(tensor.device(), None);
        assert_eq!(
            tensor.to_f32_vec().unwrap(),
            vec![
                f32::from_ne_bytes([1, 2, 3, 4]),
                f32::from_ne_bytes([5, 6, 7, 8]),
            ]
        );
        let bad = Scratch::<u8>::try_alloc(4, &budget).unwrap();
        assert!(Tensor::from_scratch(bad, &[2], DType::F32).is_err());
        drop(tensor);
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }
}
