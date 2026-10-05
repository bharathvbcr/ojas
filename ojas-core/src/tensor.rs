use crate::backend::BackendId;
use crate::budget::{Budget, Reservation, Scratch};
use crate::dtype::DType;
use crate::limits::shape_product;
use crate::OjasError;
use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

/// Memory owned by a device backend.
///
/// A backend wraps its own buffer type (an `MTLBuffer`, a `wgpu::Buffer`)
/// and hands it to [`Tensor::from_device`]. Host accessors such as
/// [`Tensor::f32_slice`] refuse a device tensor with
/// [`OjasError::Placement`]; the only way back to host memory is
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

/// An element type a host tensor can be built from with
/// [`Tensor::from_scratch`]: `f32` ([`DType::F32`]) and `u32`
/// ([`DType::U32`]). Sealed.
pub trait HostElement: Copy + Default + Send + Sync + 'static + sealed::Sealed {
    /// The dtype of a tensor that holds this element type.
    const DTYPE: DType;
}

impl HostElement for f32 {
    const DTYPE: DType = DType::F32;
}

impl HostElement for u32 {
    const DTYPE: DType = DType::U32;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
}

/// Largest buffer [`Tensor::from_le_reader`] hands its `read` callback.
///
/// It is also the only transient a load adds beside the tensor, so it is
/// kept far below any large parameter: 64 KiB, the piece a checkpoint save
/// writes (`ojas-model`'s `LE_CHUNK`). A 1.5 GB checkpoint takes about
/// 24k reads.
pub const LE_READ_CHUNK_BYTES: usize = 64 << 10;

/// Largest piece [`Tensor::to_host`] reads from a device at once.
///
/// Each piece is one [`DeviceBuffer::read_bytes`] call, charged while it is
/// decoded. Smaller pieces lower the peak a readback adds to the window;
/// larger ones make fewer calls, each of which may wait on the device. A
/// MiB keeps a checkpoint save of a 150 MB tensor to about 150 calls.
pub const READBACK_CHUNK_BYTES: usize = 1 << 20;

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

/// Owned elements plus view metadata.
///
/// `byte_offset` is the start of element 0 inside the allocation, in bytes.
/// It is always a multiple of the dtype size. A view that stores the parent
/// buffer and then assumes offset 0 writes at the start of that allocation.
/// [`Tensor::narrow`] adds to the current offset. [`Tensor::view`] takes an
/// absolute offset and does not inherit this one.
///
/// Host memory is typed: an `F32` tensor holds a `Vec<f32>`, a `U32` tensor a
/// `Vec<u32>`, and `Bf16` and `F16` tensors a `Vec<u16>` of raw bits. A view
/// never changes the dtype, so the element type is fixed when the storage is
/// created.
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

/// `payload` is declared before `_reservation`. Fields drop in order, so the
/// memory is freed and then the charge comes off the budget.
#[derive(Debug)]
struct Storage {
    payload: Payload,
    /// What a scan of the whole host allocation found: [`FINITE_UNKNOWN`],
    /// [`FINITE_YES`] or [`FINITE_NO`]. Every storage starts unknown; only
    /// [`Tensor::all_finite_cached`] over the whole allocation sets it, and
    /// [`Tensor::host_data_mut`], the one path to mutable host elements,
    /// resets it.
    finite: AtomicU8,
    /// Compute tag for dynamic bf16 autocast. [`COMPUTE_F32`] is an ordinary
    /// f32 value. [`COMPUTE_BF16`] means every element is a bf16 value widened
    /// back to f32 (low 16 bits zero). Views share this atomic. An in-place
    /// write clears it: the new bytes need not be bf16-exact.
    compute: AtomicU8,
    _reservation: Reservation,
}

const FINITE_UNKNOWN: u8 = 0;
const FINITE_YES: u8 = 1;
const FINITE_NO: u8 = 2;

/// Untagged f32. New storage starts here.
pub(crate) const COMPUTE_F32: u8 = 0;
/// Bits are bf16-rounded. Set only by the autocast wrapper after a cast.
pub(crate) const COMPUTE_BF16: u8 = 1;

/// Magnitude bits of an f32 (the sign cleared).
const MAGNITUDE: u32 = 0x7fff_ffff;
/// An f32 is NaN or infinite exactly when its magnitude bits are at least
/// this: all eight exponent bits set.
const NON_FINITE: u32 = 0x7f80_0000;
/// Values per block of [`f32_all_finite`]. A block is folded to its largest
/// magnitude without an early exit, which vectorizes to one mask and one
/// unsigned max per four values; a block with a bad value ends the scan.
const SCAN_BLOCK: usize = 1024;

/// No NaN or infinity in `data`, scanned serially in blocks of 1024 values.
///
/// This is the finiteness test a scan passed to
/// [`Tensor::all_finite_cached`] should apply, on the whole window or on
/// each piece of it when a backend splits the scan across threads.
pub fn f32_all_finite(data: &[f32]) -> bool {
    let top = |chunk: &[f32]| {
        chunk
            .iter()
            .fold(0u32, |top, value| top.max(value.to_bits() & MAGNITUDE))
    };
    let (blocks, rest) = data.as_chunks::<SCAN_BLOCK>();
    blocks.iter().all(|block| top(block) < NON_FINITE) && top(rest) < NON_FINITE
}

#[derive(Debug)]
enum Payload {
    Host(HostData),
    Device(Arc<dyn DeviceBuffer>),
}

/// Host elements. Which variant a storage holds follows from its dtype.
#[derive(Debug)]
enum HostData {
    F32(Vec<f32>),
    U32(Vec<u32>),
    /// `Bf16` and `F16` bits; the tensor's dtype says which.
    Half(Vec<u16>),
}

impl HostData {
    fn byte_len(&self) -> usize {
        match self {
            HostData::F32(v) => v.len() * 4,
            HostData::U32(v) => v.len() * 4,
            HostData::Half(v) => v.len() * 2,
        }
    }

    /// `n` zero elements of `dtype`, or `None` when the allocator refuses.
    fn try_zeroed(dtype: DType, n: usize) -> Option<Self> {
        Some(match dtype {
            DType::F32 => HostData::F32(try_zeroed_vec(n)?),
            DType::U32 => HostData::U32(try_zeroed_vec(n)?),
            DType::Bf16 | DType::F16 => HostData::Half(try_zeroed_vec(n)?),
        })
    }

    /// Decode `bytes` into the elements starting at element `start`.
    /// `bytes.len()` is a whole number of elements that fits.
    fn decode_into(&mut self, start: usize, bytes: &[u8], endian: Endian) {
        match self {
            HostData::F32(v) => {
                let (words, _) = bytes.as_chunks::<4>();
                for (slot, word) in v[start..start + words.len()].iter_mut().zip(words) {
                    *slot = f32::from_bits(endian.u32(*word));
                }
            }
            HostData::U32(v) => {
                let (words, _) = bytes.as_chunks::<4>();
                for (slot, word) in v[start..start + words.len()].iter_mut().zip(words) {
                    *slot = endian.u32(*word);
                }
            }
            HostData::Half(v) => {
                let (halves, _) = bytes.as_chunks::<2>();
                for (slot, half) in v[start..start + halves.len()].iter_mut().zip(halves) {
                    *slot = endian.u16(*half);
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Endian {
    Native,
    Little,
}

impl Endian {
    fn u32(self, b: [u8; 4]) -> u32 {
        match self {
            Endian::Native => u32::from_ne_bytes(b),
            Endian::Little => u32::from_le_bytes(b),
        }
    }

    fn u16(self, b: [u8; 2]) -> u16 {
        match self {
            Endian::Native => u16::from_ne_bytes(b),
            Endian::Little => u16::from_le_bytes(b),
        }
    }
}

fn try_zeroed_vec<T: Copy + Default>(n: usize) -> Option<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).ok()?;
    v.resize(n, T::default());
    Some(v)
}

impl Storage {
    fn new(payload: Payload, reservation: Reservation) -> Self {
        Self {
            payload,
            finite: AtomicU8::new(FINITE_UNKNOWN),
            compute: AtomicU8::new(COMPUTE_F32),
            _reservation: reservation,
        }
    }

    /// Size in bytes.
    fn len(&self) -> usize {
        match &self.payload {
            Payload::Host(data) => data.byte_len(),
            Payload::Device(buf) => buf.byte_len(),
        }
    }
}

/// A contiguous host window: element `start`, `len` elements.
struct Window {
    start: usize,
    len: usize,
}

impl Tensor {
    /// Zero-filled contiguous allocation. `byte_offset` is 0.
    pub fn zeros(shape: &[usize], dtype: DType, budget: &Budget) -> Result<Self, OjasError> {
        let strides = contiguous_strides(shape)?;
        let nbytes = contiguous_nbytes(shape, dtype)?;
        let nbytes_u64 = u64_len(nbytes, "Tensor::zeros")?;
        let reservation = budget.try_reserve(nbytes_u64)?;
        let Some(data) = HostData::try_zeroed(dtype, nbytes / dtype.size()) else {
            // Release first so `live` does not count the refused request.
            drop(reservation);
            return Err(OjasError::CapacityExceeded {
                requested: nbytes_u64,
                cap: budget.cap_bytes(),
                live: budget.live_bytes()?,
            });
        };
        let tensor = Self {
            storage: Arc::new(Storage::new(Payload::Host(data), reservation)),
            shape: shape.to_vec().into_boxed_slice(),
            strides: strides.into_boxed_slice(),
            dtype,
            byte_offset: 0,
        };
        tensor.check_window()?;
        Ok(tensor)
    }

    /// A new contiguous host tensor whose elements `read` supplies as
    /// little-endian bytes.
    ///
    /// This is the path for a loader that reads a tensor from a file (for
    /// example `ojas_io::SafeTensors::read_into`). `read(byte_offset, chunk)`
    /// fills `chunk` with the tensor's bytes starting at `byte_offset`; each
    /// chunk is at most [`LE_READ_CHUNK_BYTES`] and a whole number of
    /// elements, and the chunks cover the tensor in order. The chunk buffer
    /// is charged to `budget` beside the tensor and released before this
    /// returns. Decoding is little-endian on every target. An error from
    /// `read` is returned unchanged, and the tensor and the chunk, with their
    /// charges, are dropped. A tensor with no elements never calls `read`.
    pub fn from_le_reader<E: From<OjasError>>(
        shape: &[usize],
        dtype: DType,
        budget: &Budget,
        mut read: impl FnMut(u64, &mut [u8]) -> Result<(), E>,
    ) -> Result<Self, E> {
        const OP: &str = "Tensor::from_le_reader";
        let mut tensor = Self::zeros(shape, dtype, budget)?;
        let total = contiguous_nbytes(shape, dtype)?;
        let chunk_len = total.min(LE_READ_CHUNK_BYTES);
        let mut chunk = Scratch::<u8>::try_alloc(chunk_len, budget)?;
        let data = tensor.host_data_mut(OP)?;
        let size = dtype.size();
        let mut offset = 0usize;
        while offset < total {
            let n = (total - offset).min(chunk_len);
            let buf = &mut chunk.as_mut_slice()[..n];
            buf.fill(0);
            read(u64_len(offset, OP)?, buf)?;
            data.decode_into(offset / size, buf, Endian::Little);
            offset += n;
        }
        drop(chunk);
        Ok(tensor)
    }

    /// Copy `data` into a new contiguous `F32` allocation.
    pub fn from_f32(data: &[f32], shape: &[usize], budget: &Budget) -> Result<Self, OjasError> {
        const OP: &str = "Tensor::from_f32";
        let n = num_elements(shape)?;
        if data.len() != n {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!("data len {} != shape product {n}", data.len()),
            });
        }
        let mut tensor = Self::zeros(shape, DType::F32, budget)?;
        match tensor.host_data_mut(OP)? {
            HostData::F32(v) => v.copy_from_slice(data),
            _ => return Err(storage_mismatch(OP, DType::F32)),
        }
        Ok(tensor)
    }

    /// Copy `data` into a new contiguous `U32` allocation.
    pub fn from_u32(data: &[u32], shape: &[usize], budget: &Budget) -> Result<Self, OjasError> {
        const OP: &str = "Tensor::from_u32";
        let n = num_elements(shape)?;
        if data.len() != n {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!("data len {} != shape product {n}", data.len()),
            });
        }
        let mut tensor = Self::zeros(shape, DType::U32, budget)?;
        match tensor.host_data_mut(OP)? {
            HostData::U32(v) => v.copy_from_slice(data),
            _ => return Err(storage_mismatch(OP, DType::U32)),
        }
        Ok(tensor)
    }

    /// The contiguous `F32` window, borrowed. Refusals, in order: dtype,
    /// placement, layout, window bounds.
    pub fn f32_slice(&self) -> Result<&[f32], OjasError> {
        self.f32_window("Tensor::f32_slice")
    }

    /// Whether the contiguous `F32` window holds no NaN or infinity, with
    /// the answer for the whole allocation cached on its storage.
    ///
    /// `scan` is called on the window at most once and must return whether
    /// every value it is given is finite. An error it returns is passed on
    /// and caches nothing. It is skipped when a scan of the whole allocation
    /// already found every value finite, which answers for every view of it.
    /// A window that is the whole allocation records its result, so a later
    /// call on it, or on any view of the same storage after a finite result,
    /// does not scan. A smaller window over a storage not known finite is
    /// scanned and records nothing, since values outside it say nothing
    /// about it.
    ///
    /// The result of `scan` is trusted, so it must check every element, with
    /// [`f32_all_finite`] on the whole window or on each piece of it. A scan
    /// that reports `true` for a window holding a NaN or an infinity turns
    /// off NaN refusal for that storage, in every op and every view, until
    /// the storage is next written. Debug builds re-check a `true` result for
    /// the whole allocation and panic on a lie.
    ///
    /// Every mutable access to host elements ([`Tensor::f32_slice_mut`],
    /// [`Tensor::write_f32`]) needs sole ownership of the storage and resets
    /// the cached answer, so a value written after a finite result is
    /// scanned on the next call. Refusals, in order, as for
    /// [`Tensor::f32_slice`].
    pub fn all_finite_cached(
        &self,
        scan: impl FnOnce(&[f32]) -> Result<bool, OjasError>,
    ) -> Result<bool, OjasError> {
        const OP: &str = "Tensor::all_finite_cached";
        let window = self.f32_window(OP)?;
        let whole = match self.host_data(OP)? {
            HostData::F32(v) => window.len() == v.len(),
            _ => return Err(storage_mismatch(OP, self.dtype)),
        };
        match self.storage.finite.load(Ordering::Acquire) {
            FINITE_YES => return Ok(true),
            FINITE_NO if whole => return Ok(false),
            _ => {}
        }
        let finite = scan(window)?;
        if whole {
            debug_assert!(
                !finite || window.iter().all(|x| x.is_finite()),
                "{OP}: scan reported a non-finite window as finite"
            );
            let state = if finite { FINITE_YES } else { FINITE_NO };
            self.storage.finite.store(state, Ordering::Release);
        }
        Ok(finite)
    }

    /// The contiguous `U32` window, borrowed. Refusals as for
    /// [`Tensor::f32_slice`].
    pub fn u32_slice(&self) -> Result<&[u32], OjasError> {
        self.u32_window("Tensor::u32_slice")
    }

    /// The contiguous `F32` window, mutable. Refusals, in order: dtype,
    /// placement, layout, window overflow, a shared allocation, window
    /// bounds. The allocation must be uniquely owned: no clone or view of
    /// this tensor may be alive.
    pub fn f32_slice_mut(&mut self) -> Result<&mut [f32], OjasError> {
        const OP: &str = "Tensor::f32_slice_mut";
        self.expect_dtype(OP, DType::F32)?;
        self.f32_window_mut(OP)
    }

    /// Contiguous `F32` elements in row-major order.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, OjasError> {
        Ok(self.f32_window("Tensor::to_f32_vec")?.to_vec())
    }

    /// Contiguous `U32` elements in row-major order.
    pub fn to_u32_vec(&self) -> Result<Vec<u32>, OjasError> {
        Ok(self.u32_window("Tensor::to_u32_vec")?.to_vec())
    }

    /// Encode the contiguous window, native-endian, into `dst`. Any dtype.
    ///
    /// `dst.len()` must be the window's byte length (element count times
    /// `dtype.size()`), or the call is [`OjasError::Shape`]. The caller
    /// owns `dst`, so the caller decides whether that memory is charged.
    /// Refusals, in order: placement, layout, window bounds, `dst` length.
    pub fn write_ne_bytes(&self, dst: &mut [u8]) -> Result<(), OjasError> {
        const OP: &str = "Tensor::write_ne_bytes";
        let w = self.host_window(OP)?;
        let want = w.len * self.dtype.size();
        if dst.len() != want {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!("destination of {} bytes != window {want}", dst.len()),
            });
        }
        match self.host_data(OP)? {
            HostData::F32(v) => {
                let (slots, _) = dst.as_chunks_mut::<4>();
                for (slot, x) in slots.iter_mut().zip(&v[w.start..w.start + w.len]) {
                    *slot = x.to_ne_bytes();
                }
            }
            HostData::U32(v) => {
                let (slots, _) = dst.as_chunks_mut::<4>();
                for (slot, x) in slots.iter_mut().zip(&v[w.start..w.start + w.len]) {
                    *slot = x.to_ne_bytes();
                }
            }
            HostData::Half(v) => {
                let (slots, _) = dst.as_chunks_mut::<2>();
                for (slot, x) in slots.iter_mut().zip(&v[w.start..w.start + w.len]) {
                    *slot = x.to_ne_bytes();
                }
            }
        }
        Ok(())
    }

    /// The contiguous window encoded native-endian, in a new vector. Any
    /// dtype. Refusals as for [`Tensor::write_ne_bytes`], all before
    /// anything is allocated; an allocation the allocator refuses is
    /// [`OjasError::OutOfRange`]. The vector is not charged to any budget: this is for a caller
    /// that hands the bytes on at once, such as a device upload that has
    /// already reserved the device copy.
    pub fn to_ne_bytes(&self) -> Result<Vec<u8>, OjasError> {
        const OP: &str = "Tensor::to_ne_bytes";
        let w = self.host_window(OP)?;
        let len = w.len * self.dtype.size();
        let mut out = Vec::new();
        if out.try_reserve_exact(len).is_err() {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("host allocation of {len} bytes failed"),
            });
        }
        out.resize(len, 0);
        self.write_ne_bytes(&mut out)?;
        Ok(out)
    }

    /// Replace contiguous `F32` elements. The allocation must be uniquely owned.
    pub fn write_f32(&mut self, values: &[f32]) -> Result<(), OjasError> {
        self.writable_f32(values.len())?.copy_from_slice(values);
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
        self.writable_f32(len).map(|_| ())
    }

    /// The window [`Tensor::write_f32`] writes: dtype, then `len`, then the
    /// checks of [`Tensor::f32_slice_mut`]. The returned slice has exactly
    /// `len` elements.
    fn writable_f32(&mut self, len: usize) -> Result<&mut [f32], OjasError> {
        const OP: &str = "Tensor::write_f32";
        self.expect_dtype(OP, DType::F32)?;
        let n = self.num_elements()?;
        if len != n {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!("data len {len} != shape product {n}"),
            });
        }
        self.f32_window_mut(OP)
    }

    fn expect_dtype(&self, op: &'static str, expected: DType) -> Result<(), OjasError> {
        if self.dtype != expected {
            return Err(OjasError::Dtype {
                op,
                expected,
                got: self.dtype,
            });
        }
        Ok(())
    }

    fn f32_window(&self, op: &'static str) -> Result<&[f32], OjasError> {
        self.expect_dtype(op, DType::F32)?;
        let w = self.host_window(op)?;
        match self.host_data(op)? {
            HostData::F32(v) => Ok(&v[w.start..w.start + w.len]),
            _ => Err(storage_mismatch(op, self.dtype)),
        }
    }

    fn u32_window(&self, op: &'static str) -> Result<&[u32], OjasError> {
        self.expect_dtype(op, DType::U32)?;
        let w = self.host_window(op)?;
        match self.host_data(op)? {
            HostData::U32(v) => Ok(&v[w.start..w.start + w.len]),
            _ => Err(storage_mismatch(op, self.dtype)),
        }
    }

    /// The dtype is already checked. Placement, layout, window overflow,
    /// sole ownership, then bounds.
    fn f32_window_mut(&mut self, op: &'static str) -> Result<&mut [f32], OjasError> {
        let w = self.host_window(op)?;
        let dtype = self.dtype;
        match self.host_data_mut(op)? {
            HostData::F32(v) => Ok(&mut v[w.start..w.start + w.len]),
            _ => Err(storage_mismatch(op, dtype)),
        }
    }

    /// The contiguous host window, in elements. Placement before layout: a
    /// strided device view reports where it lives.
    fn host_window(&self, op: &'static str) -> Result<Window, OjasError> {
        if let Payload::Device(buf) = &self.storage.payload {
            return Err(OjasError::Placement {
                op,
                expected: None,
                found: Some(buf.backend()),
            });
        }
        if !self.is_contiguous()? {
            return Err(OjasError::Shape {
                op,
                detail: "view is not contiguous".to_string(),
            });
        }
        let nbytes = contiguous_nbytes(&self.shape, self.dtype)?;
        let end = self
            .byte_offset
            .checked_add(nbytes)
            .ok_or_else(|| OjasError::OutOfRange {
                op,
                detail: "window end overflows".to_string(),
            })?;
        let len = self.storage.len();
        if end > len {
            return Err(OjasError::OutOfRange {
                op,
                detail: format!("window {end} exceeds storage {len}"),
            });
        }
        let size = self.dtype.size();
        Ok(Window {
            start: self.byte_offset / size,
            len: nbytes / size,
        })
    }

    fn host_data(&self, op: &'static str) -> Result<&HostData, OjasError> {
        match &self.storage.payload {
            Payload::Host(data) => Ok(data),
            Payload::Device(buf) => Err(OjasError::Placement {
                op,
                expected: None,
                found: Some(buf.backend()),
            }),
        }
    }

    /// The whole host allocation, mutable, when this tensor solely owns it.
    ///
    /// The caller may write any element, so the cached result of
    /// [`Tensor::all_finite_cached`] is reset here, once, for every mutator.
    /// Sole ownership means no other handle can be reading the flag.
    fn host_data_mut(&mut self, op: &'static str) -> Result<&mut HostData, OjasError> {
        let storage = Arc::get_mut(&mut self.storage).ok_or_else(|| OjasError::Shape {
            op,
            detail: "contiguous write requires a uniquely owned allocation".to_string(),
        })?;
        *storage.finite.get_mut() = FINITE_UNKNOWN;
        *storage.compute.get_mut() = COMPUTE_F32;
        match &mut storage.payload {
            Payload::Host(data) => Ok(data),
            Payload::Device(buf) => Err(OjasError::Placement {
                op,
                expected: None,
                found: Some(buf.backend()),
            }),
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// [`COMPUTE_F32`] or [`COMPUTE_BF16`]. Views of one allocation agree.
    pub(crate) fn compute_tag(&self) -> u8 {
        self.storage.compute.load(Ordering::Acquire)
    }

    /// Record whether these bits are bf16-rounded. Shared by every view.
    pub(crate) fn set_compute_tag(&self, tag: u8) {
        self.storage.compute.store(tag, Ordering::Release);
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

    /// Size of the whole allocation in bytes.
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
            storage: Arc::new(Storage::new(Payload::Device(buffer), reservation)),
            shape: shape.to_vec().into_boxed_slice(),
            strides: strides.into_boxed_slice(),
            dtype,
            byte_offset: 0,
        };
        tensor.check_window()?;
        Ok(tensor)
    }

    /// Host tensor that takes the elements and the reservation from
    /// `scratch`. The dtype is `T`'s ([`HostElement::DTYPE`]).
    ///
    /// The scratch is not copied and its budget is not charged again. The
    /// reservation moves into this tensor and is released when the last view
    /// drops. A length (in elements) that is not the element count of `shape`
    /// is [`OjasError::Shape`]; the scratch is dropped and its charge
    /// released. This constructor always builds host memory. Device buffers
    /// stay on [`Tensor::from_device`] and [`Tensor::from_device_reserved`].
    pub fn from_scratch<T: HostElement>(
        scratch: Scratch<T>,
        shape: &[usize],
    ) -> Result<Self, OjasError> {
        const OP: &str = "Tensor::from_scratch";
        let dtype = T::DTYPE;
        let expected = num_elements(shape)?;
        let nbytes = contiguous_nbytes(shape, dtype)?;
        let (data, reservation) = scratch.into_raw();
        if data.len() != expected {
            let got = data.len();
            drop(data);
            drop(reservation);
            return Err(OjasError::Shape {
                op: OP,
                detail: format!("scratch len {got} != shape product {expected}"),
            });
        }
        let nbytes_u64 = u64_len(nbytes, OP)?;
        if reservation.bytes() != nbytes_u64 {
            let held = reservation.bytes();
            drop(data);
            drop(reservation);
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("reservation of {held} bytes != scratch bytes {nbytes_u64}"),
            });
        }
        // `T` is sealed to f32 and u32; the downcast moves the vector, it
        // does not copy its elements.
        let any: Box<dyn Any> = Box::new(data);
        let host = match any.downcast::<Vec<f32>>() {
            Ok(v) => HostData::F32(*v),
            Err(any) => match any.downcast::<Vec<u32>>() {
                Ok(v) => HostData::U32(*v),
                Err(_) => return Err(storage_mismatch(OP, dtype)),
            },
        };
        let strides = contiguous_strides(shape)?;
        let tensor = Self {
            storage: Arc::new(Storage::new(Payload::Host(host), reservation)),
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
        // An in-place device write can destroy bf16-exactness. Clear the tag
        // before the caller receives the buffer, including when the buffer
        // Arc is shared and the call then refuses: a cleared tag only causes
        // a later cast, and that cast is idempotent.
        *storage.compute.get_mut() = COMPUTE_F32;
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
    ///
    /// The window is read with [`DeviceBuffer::read_bytes`] in pieces of at
    /// most [`READBACK_CHUNK_BYTES`], each decoded straight into the typed
    /// result. `budget` is charged for the result and for one piece; the
    /// piece's charge is released before this returns, so the peak is the
    /// window plus one piece, not twice the window. A short read is
    /// [`OjasError::Backend`] and nothing is kept.
    pub fn to_host(&self, budget: &Budget) -> Result<Self, OjasError> {
        const OP: &str = "Tensor::to_host";
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
                op: OP,
                detail: "window end overflows".to_string(),
            })?;
        if end > buf.byte_len() {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("window {end} exceeds device storage {}", buf.byte_len()),
            });
        }
        if !len.is_multiple_of(self.dtype.size()) {
            return Err(OjasError::Backend {
                id: buf.backend(),
                detail: format!(
                    "device allocation of {len} bytes is not a whole number of {:?} elements",
                    self.dtype
                ),
            });
        }
        let len_u64 = u64_len(len, OP)?;
        let size = self.dtype.size();
        let reservation = budget.try_reserve(len_u64)?;
        let chunk_len = len.min(READBACK_CHUNK_BYTES);
        let transient = budget.try_reserve(u64_len(chunk_len, OP)?)?;
        let Some(mut data) = HostData::try_zeroed(self.dtype, len / size) else {
            drop(transient);
            drop(reservation);
            return Err(OjasError::CapacityExceeded {
                requested: len_u64,
                cap: budget.cap_bytes(),
                live: budget.live_bytes()?,
            });
        };
        let mut done = 0usize;
        while done < len {
            let n = (len - done).min(chunk_len);
            let bytes = buf.read_bytes(offset + done, n)?;
            if bytes.len() != n {
                return Err(OjasError::Backend {
                    id: buf.backend(),
                    detail: format!(
                        "read_bytes returned {} bytes, asked {n} at offset {}",
                        bytes.len(),
                        offset + done
                    ),
                });
            }
            data.decode_into(done / size, &bytes, Endian::Native);
            done += n;
        }
        drop(transient);
        READBACKS.fetch_add(1, Ordering::Relaxed);
        READBACK_BYTES.fetch_add(len_u64, Ordering::Relaxed);
        budget.record_readback(len_u64);
        let host = Self {
            storage: Arc::new(Storage::new(Payload::Host(data), reservation)),
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

/// A host storage whose element type is not the tensor's dtype. Every
/// constructor picks the variant from the dtype and a view never changes
/// it, so this is unreachable; it is reported, not a panic.
fn storage_mismatch(op: &'static str, dtype: DType) -> OjasError {
    OjasError::Shape {
        op,
        detail: format!("host storage does not hold {dtype:?} elements"),
    }
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
        let err = broadcast.f32_slice().unwrap_err();
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
        assert!(placement(t.f32_slice().unwrap_err()));
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
        assert!(placement(col.f32_slice().unwrap_err()));
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

    #[test]
    fn from_le_reader_decodes_little_endian_and_drops_on_error() {
        let budget = Budget::new(1 << 10);
        let values = [1.5f32, -0.0, f32::MIN_POSITIVE, 3.0e38];
        let le: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut calls = Vec::new();
        let t = Tensor::from_le_reader(&[2, 2], DType::F32, &budget, |offset, chunk| {
            calls.push((offset, chunk.len()));
            assert!(chunk.iter().all(|&b| b == 0), "chunk not zeroed");
            let at = offset as usize;
            chunk.copy_from_slice(&le[at..at + chunk.len()]);
            Ok::<(), OjasError>(())
        })
        .unwrap();
        assert_eq!(calls, [(0, 16)]);
        let got: Vec<u32> = t.f32_slice().unwrap().iter().map(|v| v.to_bits()).collect();
        let want: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, want);
        // The chunk was released: only the tensor stays charged.
        assert_eq!(budget.live_bytes().unwrap(), 16);

        let ids = Tensor::from_le_reader(&[2], DType::U32, &budget, |_, chunk| {
            chunk.copy_from_slice(&[7, 0, 0, 0, 0, 1, 0, 0]);
            Ok::<(), OjasError>(())
        })
        .unwrap();
        assert_eq!(ids.u32_slice().unwrap(), [7, 256]);
        let half = Tensor::from_le_reader(&[2], DType::Bf16, &budget, |_, chunk| {
            chunk.copy_from_slice(&[0x80, 0x3f, 0x00, 0xc0]);
            Ok::<(), OjasError>(())
        })
        .unwrap();
        let mut ne = [0u8; 4];
        half.write_ne_bytes(&mut ne).unwrap();
        assert_eq!(
            [
                u16::from_ne_bytes([ne[0], ne[1]]),
                u16::from_ne_bytes([ne[2], ne[3]])
            ],
            [0x3f80, 0xc000]
        );
        drop((t, ids, half));
        assert_eq!(budget.live_bytes().unwrap(), 0);

        let failed = Tensor::from_le_reader(&[4], DType::F32, &budget, |_, _| Err(Load::Short));
        assert_eq!(failed.unwrap_err(), Load::Short);
        assert_eq!(
            budget.live_bytes().unwrap(),
            0,
            "a failed read kept a charge"
        );
        // Over budget: the core error converts, and read never runs.
        let over = Tensor::from_le_reader(
            &[1 << 20],
            DType::F32,
            &budget,
            |_, _| -> Result<(), Load> { panic!("read ran without a reservation") },
        );
        assert_eq!(over.unwrap_err(), Load::Core);
        // The tensor fits but its chunk does not: refused before any read.
        let tight = Budget::new(16 + 15);
        let no_chunk =
            Tensor::from_le_reader(&[4], DType::F32, &tight, |_, _| -> Result<(), Load> {
                panic!("read ran without a charged chunk")
            });
        assert_eq!(no_chunk.unwrap_err(), Load::Core);
        assert_eq!(tight.live_bytes().unwrap(), 0);
        // An empty tensor never calls read.
        let empty =
            Tensor::from_le_reader(&[3, 0], DType::F32, &budget, |_, _| -> Result<(), Load> {
                panic!("read ran for an empty tensor")
            });
        assert_eq!(empty.unwrap().num_elements().unwrap(), 0);
    }

    #[test]
    fn from_le_reader_chunks_cover_the_tensor_and_a_late_short_read_drops_both_charges() {
        let n = LE_READ_CHUNK_BYTES / 4 * 2 + 3;
        let total = n * 4;
        let budget = Budget::new((total + LE_READ_CHUNK_BYTES) as u64);
        let mut calls = Vec::new();
        let t = Tensor::from_le_reader(&[n], DType::F32, &budget, |offset, chunk| {
            calls.push((offset, chunk.len()));
            for (i, word) in chunk.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                *word = ((offset as usize / 4 + i) as f32).to_le_bytes();
            }
            Ok::<(), OjasError>(())
        })
        .unwrap();
        let c = LE_READ_CHUNK_BYTES as u64;
        assert_eq!(
            calls,
            [
                (0, LE_READ_CHUNK_BYTES),
                (c, LE_READ_CHUNK_BYTES),
                (2 * c, 12)
            ]
        );
        let v = t.f32_slice().unwrap();
        assert!(v.iter().enumerate().all(|(i, &x)| x == i as f32));
        assert_eq!(budget.live_bytes().unwrap(), total as u64);
        drop(t);

        let late = Tensor::from_le_reader(&[n], DType::F32, &budget, |offset, _| {
            if offset > 0 {
                Err(Load::Short)
            } else {
                Ok(())
            }
        });
        assert_eq!(late.unwrap_err(), Load::Short);
        assert_eq!(
            budget.live_bytes().unwrap(),
            0,
            "tensor or chunk charge leaked"
        );
    }

    #[test]
    fn typed_slices_follow_views_and_refuse_the_wrong_dtype() {
        let budget = Budget::new(1 << 10);
        let t = f32_tensor(6, &budget);
        assert_eq!(t.f32_slice().unwrap(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let tail = t.narrow(8, &[2, 2], &[2, 1]).unwrap();
        assert_eq!(tail.f32_slice().unwrap(), [2.0, 3.0, 4.0, 5.0]);
        let mid = t.view(&[3], &[1], 4).unwrap();
        assert_eq!(mid.f32_slice().unwrap(), [1.0, 2.0, 3.0]);
        let strided = t.view(&[3], &[2], 0).unwrap();
        assert!(matches!(
            strided.f32_slice().unwrap_err(),
            OjasError::Shape { op: "Tensor::f32_slice", ref detail } if detail == "view is not contiguous"
        ));

        let ids = Tensor::from_u32(&[4, 5, 6], &[3], &budget).unwrap();
        assert_eq!(
            ids.narrow(4, &[2], &[1]).unwrap().u32_slice().unwrap(),
            [5, 6]
        );
        assert!(matches!(
            ids.f32_slice().unwrap_err(),
            OjasError::Dtype {
                op: "Tensor::f32_slice",
                expected: DType::F32,
                got: DType::U32
            }
        ));
        assert!(matches!(
            t.u32_slice().unwrap_err(),
            OjasError::Dtype {
                op: "Tensor::u32_slice",
                expected: DType::U32,
                got: DType::F32
            }
        ));
        let half = Tensor::zeros(&[2], DType::F16, &budget).unwrap();
        assert!(matches!(
            half.f32_slice().unwrap_err(),
            OjasError::Dtype { .. }
        ));
        assert!(matches!(
            half.u32_slice().unwrap_err(),
            OjasError::Dtype { .. }
        ));
        assert_eq!(half.storage_len(), 4);
    }

    #[test]
    fn f32_slice_mut_needs_sole_ownership_and_writes_only_its_window() {
        let budget = Budget::new(1 << 10);
        let mut t = f32_tensor(4, &budget);
        let alias = t.clone();
        assert!(matches!(
            t.f32_slice_mut().unwrap_err(),
            OjasError::Shape { op: "Tensor::f32_slice_mut", ref detail } if detail.contains("uniquely owned")
        ));
        drop(alias);
        t.f32_slice_mut().unwrap()[1] = 9.0;
        assert_eq!(t.to_f32_vec().unwrap(), [0.0, 9.0, 2.0, 3.0]);

        let mut window = t.narrow(8, &[2], &[1]).unwrap();
        drop(t);
        window.f32_slice_mut().unwrap().copy_from_slice(&[7.0, 8.0]);
        assert_eq!(window.f32_slice().unwrap(), [7.0, 8.0]);
        assert_eq!(
            window.storage_len(),
            16,
            "the window still owns the whole allocation"
        );

        let mut ids = Tensor::from_u32(&[1], &[1], &budget).unwrap();
        assert!(matches!(
            ids.f32_slice_mut().unwrap_err(),
            OjasError::Dtype { .. }
        ));
    }

    #[test]
    fn write_f32_checks_length_before_placement() {
        let budget = Budget::new(1 << 10);
        let mut t =
            Tensor::from_device(fake(&[1.0, 2.0], false), &[2], DType::F32, &budget).unwrap();
        assert!(matches!(
            t.write_f32(&[1.0]).unwrap_err(),
            OjasError::Shape { op: "Tensor::write_f32", ref detail } if detail.contains("data len 1")
        ));
        assert!(matches!(
            t.ensure_writable_f32(3).unwrap_err(),
            OjasError::Shape { .. }
        ));
        assert!(matches!(
            t.write_f32(&[1.0, 2.0]).unwrap_err(),
            OjasError::Placement { .. }
        ));
    }

    #[test]
    fn write_ne_bytes_encodes_every_dtype_and_checks_its_destination() {
        let budget = Budget::new(1 << 10);
        let t = Tensor::from_f32(&[1.5, -2.0, 3.25], &[3], &budget).unwrap();
        let tail = t.narrow(4, &[2], &[1]).unwrap();
        let mut out = [0u8; 8];
        tail.write_ne_bytes(&mut out).unwrap();
        let want: Vec<u8> = [-2.0f32, 3.25]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        assert_eq!(out.as_slice(), want.as_slice());
        let ids = Tensor::from_u32(&[0xDEAD_BEEF], &[1], &budget).unwrap();
        let mut out = [0u8; 4];
        ids.write_ne_bytes(&mut out).unwrap();
        assert_eq!(out, 0xDEAD_BEEFu32.to_ne_bytes());
        let half = Tensor::zeros(&[3], DType::Bf16, &budget).unwrap();
        let mut out = [0xFFu8; 6];
        half.write_ne_bytes(&mut out).unwrap();
        assert_eq!(out, [0; 6]);

        let mut short = [0u8; 7];
        assert!(matches!(
            tail.write_ne_bytes(&mut short).unwrap_err(),
            OjasError::Shape {
                op: "Tensor::write_ne_bytes",
                ..
            }
        ));
        let strided = t.view(&[2], &[2], 0).unwrap();
        assert!(matches!(
            strided.write_ne_bytes(&mut [0u8; 8]).unwrap_err(),
            OjasError::Shape { ref detail, .. } if detail == "view is not contiguous"
        ));
        let dev = Tensor::from_device(fake(&[1.0], false), &[1], DType::F32, &budget).unwrap();
        assert!(matches!(
            dev.write_ne_bytes(&mut [0u8; 4]).unwrap_err(),
            OjasError::Placement {
                op: "Tensor::write_ne_bytes",
                found: Some(BackendId::Wgpu),
                ..
            }
        ));

        // to_ne_bytes: the same encoding, validated before it allocates.
        assert_eq!(tail.to_ne_bytes().unwrap(), want);
        let live = budget.live_bytes().unwrap();
        let _ = half.to_ne_bytes().unwrap();
        assert_eq!(
            budget.live_bytes().unwrap(),
            live,
            "to_ne_bytes is not charged"
        );
        let huge = t.view(&[usize::MAX / 8, 3], &[0, 1], 0).unwrap();
        assert!(matches!(
            huge.to_ne_bytes().unwrap_err(),
            OjasError::Shape { op: "Tensor::to_ne_bytes", ref detail } if detail == "view is not contiguous"
        ));
        assert!(matches!(
            dev.to_ne_bytes().unwrap_err(),
            OjasError::Placement {
                op: "Tensor::to_ne_bytes",
                ..
            }
        ));
    }

    #[test]
    fn to_host_charges_the_readback_bytes_while_it_decodes() {
        // 24 device bytes, a 24-byte result, and the 24 bytes read_bytes
        // returns: 72 fits, 71 does not, and a refusal keeps no charge.
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let budget = Budget::new(72);
        let t = Tensor::from_device(fake(&values, false), &[6], DType::F32, &budget).unwrap();
        let host = t.to_host(&budget).unwrap();
        assert_eq!(host.f32_slice().unwrap(), values);
        assert_eq!(
            budget.live_bytes().unwrap(),
            48,
            "the transient was released"
        );
        drop(host);

        let tight = Budget::new(71);
        let t = Tensor::from_device(fake(&values, false), &[6], DType::F32, &tight).unwrap();
        assert!(matches!(
            t.to_host(&tight),
            Err(OjasError::CapacityExceeded {
                requested: 24,
                live: 48,
                ..
            })
        ));
        assert_eq!(tight.live_bytes().unwrap(), 24);
        assert_eq!(
            tight.device_readbacks(),
            (0, 0),
            "a refused readback is not counted"
        );
    }

    /// A device whose reads past `short_after` bytes come back one byte short.
    #[derive(Debug)]
    struct LateShort {
        bytes: Vec<u8>,
        short_after: usize,
        reads: AtomicU64,
    }

    impl DeviceBuffer for LateShort {
        fn backend(&self) -> BackendId {
            BackendId::Metal
        }
        fn byte_len(&self) -> usize {
            self.bytes.len()
        }
        fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            let mut out = self.bytes[offset..offset + len].to_vec();
            if offset >= self.short_after {
                out.pop();
            }
            Ok(out)
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn to_host_reads_in_pieces_and_peaks_at_one_piece_over_the_window() {
        let n = READBACK_CHUNK_BYTES / 4 * 2 + 5;
        let len = n * 4;
        let values: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let device = |short_after| {
            Arc::new(LateShort {
                bytes: values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                short_after,
                reads: AtomicU64::new(0),
            })
        };
        // Device copy, result, and one piece: exactly enough.
        let need = (2 * len + READBACK_CHUNK_BYTES) as u64;
        let budget = Budget::new(need);
        let buf = device(usize::MAX);
        let t = Tensor::from_device(buf.clone(), &[n], DType::F32, &budget).unwrap();
        let host = t.to_host(&budget).unwrap();
        assert_eq!(
            buf.reads.load(Ordering::Relaxed),
            3,
            "ceil(len / piece) reads"
        );
        assert_eq!(host.f32_slice().unwrap(), values.as_slice());
        assert_eq!(
            budget.live_bytes().unwrap(),
            2 * len as u64,
            "the piece was released"
        );
        assert_eq!(
            budget.device_readbacks(),
            (1, len as u64),
            "one readback, not three"
        );
        drop((t, host));

        // One byte less refuses before reading, and keeps only the device copy.
        let tight = Budget::new(need - 1);
        let buf = device(usize::MAX);
        let t = Tensor::from_device(buf.clone(), &[n], DType::F32, &tight).unwrap();
        assert!(matches!(
            t.to_host(&tight),
            Err(OjasError::CapacityExceeded { .. })
        ));
        assert_eq!(buf.reads.load(Ordering::Relaxed), 0);
        assert_eq!(tight.live_bytes().unwrap(), len as u64);
        drop(t);

        // A short second piece is a backend error; both charges are released
        // and nothing is counted as a readback.
        let budget = Budget::new(need);
        let t =
            Tensor::from_device(device(READBACK_CHUNK_BYTES), &[n], DType::F32, &budget).unwrap();
        assert!(matches!(
            t.to_host(&budget),
            Err(OjasError::Backend { id: BackendId::Metal, ref detail }) if detail.contains("at offset 1048576")
        ));
        assert_eq!(budget.live_bytes().unwrap(), len as u64);
        assert_eq!(budget.device_readbacks(), (0, 0));
    }

    #[test]
    fn half_precision_reads_back_bit_exact() {
        let budget = Budget::new(1 << 10);
        let bits: [u16; 4] = [0x3f80, 0x8000, 0x7f80, 0x0001];
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_ne_bytes()).collect();
        let buf = Arc::new(FakeDevice {
            bytes,
            short_read: false,
            reads: AtomicU64::new(0),
        });
        let t = Tensor::from_device(buf, &[4], DType::Bf16, &budget).unwrap();
        let host = t.to_host(&budget).unwrap();
        assert_eq!(host.storage_len(), 8);
        let mut out = [0u8; 8];
        host.write_ne_bytes(&mut out).unwrap();
        let back: Vec<u16> = out
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_ne_bytes(*c))
            .collect();
        assert_eq!(back, bits);
    }

    #[test]
    fn from_scratch_keeps_one_charge_and_stays_on_the_host() {
        use crate::Scratch;
        let budget = Budget::new(64);
        let mut scratch = Scratch::<f32>::try_alloc(2, &budget).unwrap();
        scratch.as_mut_slice().copy_from_slice(&[1.5, -2.5]);
        assert_eq!(budget.live_bytes().unwrap(), 8);
        let tensor = Tensor::from_scratch(scratch, &[2]).unwrap();
        assert_eq!(
            budget.live_bytes().unwrap(),
            8,
            "the reservation moved, it was not charged again"
        );
        assert_eq!(tensor.device(), None);
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.f32_slice().unwrap(), [1.5, -2.5]);

        let mut ids = Scratch::<u32>::try_alloc(3, &budget).unwrap();
        ids.as_mut_slice().copy_from_slice(&[3, 1, 2]);
        let ids = Tensor::from_scratch(ids, &[3, 1]).unwrap();
        assert_eq!(ids.dtype(), DType::U32);
        assert_eq!(ids.u32_slice().unwrap(), [3, 1, 2]);

        let bad = Scratch::<f32>::try_alloc(1, &budget).unwrap();
        assert!(matches!(
            Tensor::from_scratch(bad, &[2]).unwrap_err(),
            OjasError::Shape { op: "Tensor::from_scratch", ref detail } if detail == "scratch len 1 != shape product 2"
        ));
        drop((tensor, ids));
        assert_eq!(
            budget.live_bytes().unwrap(),
            0,
            "a refused scratch kept its charge"
        );
    }

    /// `t.all_finite_cached` with an honest scan that counts its calls.
    fn finite_counted(t: &Tensor, scans: &std::cell::Cell<usize>) -> bool {
        t.all_finite_cached(|w| {
            scans.set(scans.get() + 1);
            Ok(w.iter().all(|x| x.is_finite()))
        })
        .unwrap()
    }

    #[test]
    fn a_write_after_a_finite_scan_resets_the_cached_result() {
        let budget = Budget::new(1 << 20);
        let scans = std::cell::Cell::new(0);
        let mut t = f32_tensor(8, &budget);
        assert!(finite_counted(&t, &scans));
        assert!(finite_counted(&t, &scans));
        assert_eq!(scans.get(), 1, "a finite whole storage is scanned once");
        t.write_f32(&[1.0; 8]).unwrap();
        assert!(finite_counted(&t, &scans));
        assert_eq!(scans.get(), 2, "write_f32 must reset the cached result");
        t.f32_slice_mut().unwrap()[0] = 2.0;
        assert!(finite_counted(&t, &scans));
        assert_eq!(scans.get(), 3, "f32_slice_mut must reset the cached result");
        // Asking for the mutable window resets it even if nothing is written.
        let _ = t.f32_slice_mut().unwrap();
        assert!(finite_counted(&t, &scans));
        assert_eq!(scans.get(), 4);
    }

    #[test]
    fn a_shared_storage_cannot_be_written_so_its_cached_result_holds() {
        let budget = Budget::new(1 << 20);
        let scans = std::cell::Cell::new(0);
        let mut t = f32_tensor(8, &budget);
        assert!(finite_counted(&t, &scans));
        let clone = t.clone();
        let view = t.view(&[2], &[1], 8).unwrap();
        let reshaped = t.reshape(&[2, 4]).unwrap();
        assert!(matches!(t.f32_slice_mut(), Err(OjasError::Shape { .. })));
        assert!(matches!(
            t.write_f32(&[f32::NAN; 8]),
            Err(OjasError::Shape { .. })
        ));
        for other in [&t, &clone, &view, &reshaped] {
            assert!(finite_counted(other, &scans));
        }
        assert_eq!(scans.get(), 1, "a refused write must not reset the result");
        assert_eq!(t.f32_slice().unwrap()[0], 0.0);
    }

    /// The one test that fails if `host_data_mut` stops resetting the flag:
    /// the NaN is written after a cached finite result and must be found.
    #[test]
    fn a_nan_written_after_a_finite_result_is_caught_on_the_next_call() {
        let budget = Budget::new(1 << 20);
        let scans = std::cell::Cell::new(0);
        let mut t = f32_tensor(8, &budget);
        assert!(finite_counted(&t, &scans));
        t.f32_slice_mut().unwrap()[5] = f32::NAN;
        assert!(!finite_counted(&t, &scans));
        assert_eq!(scans.get(), 2);
        // A whole storage known non-finite answers without a scan, and so do
        // its whole-window views; a fresh view after the write is not stale.
        assert!(!finite_counted(&t, &scans));
        assert!(!finite_counted(&t.reshape(&[4, 2]).unwrap(), &scans));
        assert_eq!(scans.get(), 2);
        t.f32_slice_mut().unwrap()[5] = f32::INFINITY;
        assert!(!finite_counted(&t, &scans));
        t.f32_slice_mut().unwrap()[5] = 5.0;
        assert!(finite_counted(&t, &scans));
        assert_eq!(scans.get(), 4);
    }

    #[test]
    fn a_view_over_a_non_finite_storage_scans_its_own_window() {
        let budget = Budget::new(1 << 20);
        let scans = std::cell::Cell::new(0);
        let t = Tensor::from_f32(&[1.0, f32::NAN, 3.0, 4.0], &[4], &budget).unwrap();
        let tail = t.view(&[2], &[1], 8).unwrap();
        let head = t.view(&[2], &[1], 0).unwrap();
        // Before the storage is known: a window scan records nothing.
        assert!(finite_counted(&tail, &scans));
        assert!(finite_counted(&tail, &scans));
        assert_eq!(scans.get(), 2, "a smaller window caches nothing");
        assert!(!finite_counted(&t, &scans));
        assert_eq!(scans.get(), 3);
        // After: the storage is known non-finite, but each window is
        // still scanned on its own.
        assert!(finite_counted(&tail, &scans));
        assert!(!finite_counted(&head, &scans));
        assert_eq!(scans.get(), 5);
        assert!(!finite_counted(&t, &scans));
        assert_eq!(scans.get(), 5);
    }

    #[test]
    fn a_failed_scan_is_passed_on_and_caches_nothing() {
        let budget = Budget::new(1 << 20);
        let scans = std::cell::Cell::new(0);
        let t = f32_tensor(4, &budget);
        let err = t
            .all_finite_cached(|_| {
                Err(OjasError::Shape {
                    op: "scan",
                    detail: "pool failed".to_string(),
                })
            })
            .unwrap_err();
        assert!(matches!(err, OjasError::Shape { op: "scan", .. }));
        assert!(finite_counted(&t, &scans));
        assert_eq!(scans.get(), 1);
    }

    #[test]
    fn refusals_come_before_the_scan() {
        let budget = Budget::new(1 << 20);
        let called = std::cell::Cell::new(false);
        let scan = |_: &[f32]| {
            called.set(true);
            Ok(true)
        };
        let ids = Tensor::from_u32(&[1, 2], &[2], &budget).unwrap();
        assert!(matches!(
            ids.all_finite_cached(scan),
            Err(OjasError::Dtype {
                op: "Tensor::all_finite_cached",
                ..
            })
        ));
        let t = f32_tensor(4, &budget);
        let strided = t.view(&[2], &[2], 0).unwrap();
        assert!(matches!(
            strided.all_finite_cached(scan),
            Err(OjasError::Shape { .. })
        ));
        assert!(!called.get());
    }

    /// The cache is set by a check, so a scan that calls a NaN finite is a
    /// bug the debug build reports instead of caching.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "scan reported a non-finite window as finite")]
    fn a_scan_that_lies_about_a_whole_storage_panics_in_debug() {
        let budget = Budget::new(1 << 20);
        let t = Tensor::from_f32(&[f32::NAN], &[1], &budget).unwrap();
        let _ = t.all_finite_cached(|_| Ok(true));
    }

    /// Every non-finite class is caught at every position (first, inside a
    /// whole block, in the tail past the last block); every finite edge value
    /// passes.
    #[test]
    fn finite_scans_catch_every_nonfinite_at_every_position() {
        let bad = [
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7f80_0001),
            f32::from_bits(0xffc0_0000),
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let good = [
            0.0,
            -0.0,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            1.0,
        ];
        for len in [1usize, 7, 255, 256, 257, 1023, 1024, 1025, 4100] {
            let mut data: Vec<f32> = (0..len).map(|i| good[i % good.len()]).collect();
            assert!(f32_all_finite(&data), "len {len}");
            for at in [0, len / 2, len - 1] {
                for &value in &bad {
                    let keep = data[at];
                    data[at] = value;
                    assert!(
                        !f32_all_finite(&data),
                        "len {len} at {at} {:#x}",
                        value.to_bits()
                    );
                    data[at] = keep;
                }
            }
        }
        assert!(f32_all_finite(&[]));
    }
}
