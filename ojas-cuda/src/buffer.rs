//! [`CudaBuffer<T>`]: a typed, length-checked device allocation held against
//! the runtime's [`crate::budget::AllocBudget`].
//!
//! Uploads and writes must match the buffer's length exactly; downloads
//! wait with the runtime's bounded sync before and after the copy, because
//! cudarc's device-to-host copy into a `Vec` does not synchronise
//! (`cudarc/src/driver/safe/core.rs:1434-1450`).

use cudarc::driver::{CudaSlice, CudaView, CudaViewMut, DeviceRepr, ValidAsZeroBits};

use crate::budget::Reservation;
use crate::error::CudaError;
use crate::runtime::{driver_error, CudaRuntime};

/// Element types a buffer may hold.
pub trait Element: DeviceRepr + ValidAsZeroBits + Copy + Default + Send + Sync + 'static {
    /// The name reports use.
    const NAME: &'static str;
}

impl Element for f32 {
    const NAME: &'static str = "f32";
}

/// bf16 values, as their 16 bits.
impl Element for u16 {
    const NAME: &'static str = "bf16_bits";
}

impl Element for u32 {
    const NAME: &'static str = "u32";
}

/// A device buffer of `len` elements of `T`.
pub struct CudaBuffer<T: Element> {
    slice: CudaSlice<T>,
    // Released when the buffer drops (after the slice's stream-ordered free).
    _reservation: Reservation,
    label: String,
}

impl<T: Element> std::fmt::Debug for CudaBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CudaBuffer<{}>({}, {} elements)",
            T::NAME,
            self.label,
            self.slice.len()
        )
    }
}

impl<T: Element> CudaBuffer<T> {
    /// Elements.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Never true: zero-length buffers are refused at allocation.
    pub fn is_empty(&self) -> bool {
        self.slice.len() == 0
    }

    /// The label given at allocation.
    pub fn label(&self) -> &str {
        &self.label
    }

    pub(crate) fn slice(&self) -> &CudaSlice<T> {
        &self.slice
    }

    pub(crate) fn slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }

    /// The whole buffer as a [`BufView`].
    pub fn all(&self) -> BufView<'_, T> {
        BufView {
            buf: self,
            off: 0,
            len: self.len(),
        }
    }

    /// Elements `[off, off + len)` as a [`BufView`]: refused when empty or
    /// past the end ([`crate::geometry::view_range`]).
    pub fn view(&self, off: usize, len: usize, op: &str) -> Result<BufView<'_, T>, CudaError> {
        crate::geometry::view_range(self.len(), off, len, op)?;
        Ok(BufView {
            buf: self,
            off,
            len,
        })
    }

    /// The whole buffer as a [`BufViewMut`].
    pub fn all_mut(&mut self) -> BufViewMut<'_, T> {
        let len = self.len();
        BufViewMut {
            buf: self,
            off: 0,
            len,
        }
    }

    /// Elements `[off, off + len)` as a [`BufViewMut`], checked as [`Self::view`].
    pub fn view_mut(
        &mut self,
        off: usize,
        len: usize,
        op: &str,
    ) -> Result<BufViewMut<'_, T>, CudaError> {
        crate::geometry::view_range(self.len(), off, len, op)?;
        Ok(BufViewMut {
            buf: self,
            off,
            len,
        })
    }
}

/// A checked, non-empty element range of a [`CudaBuffer`], read-only. A
/// kernel or cuBLAS gets it as a device pointer `off` elements in, so a block
/// of rows of a larger matrix (K10's head chunk) is an operand without a copy.
#[derive(Clone, Copy)]
pub struct BufView<'a, T: Element> {
    buf: &'a CudaBuffer<T>,
    off: usize,
    len: usize,
}

impl<'a, T: Element> BufView<'a, T> {
    /// Elements in the view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Never true: views are non-empty by construction.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The cudarc view: in range by construction.
    pub(crate) fn device(&self) -> CudaView<'a, T> {
        self.buf.slice().slice(self.off..self.off + self.len)
    }
}

/// A checked, non-empty element range of a [`CudaBuffer`], writable.
pub struct BufViewMut<'a, T: Element> {
    buf: &'a mut CudaBuffer<T>,
    off: usize,
    len: usize,
}

impl<T: Element> BufViewMut<'_, T> {
    /// Elements in the view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Never true: views are non-empty by construction.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The cudarc view: in range by construction.
    pub(crate) fn device(&mut self) -> CudaViewMut<'_, T> {
        self.buf
            .slice_mut()
            .slice_mut(self.off..self.off + self.len)
    }
}

impl CudaRuntime {
    /// A zeroed buffer of `len` elements.
    pub fn alloc_zeros<T: Element>(
        &self,
        len: usize,
        label: &str,
    ) -> Result<CudaBuffer<T>, CudaError> {
        let reservation = self.reserve::<T>(len, label)?;
        let slice = self
            .stream()
            .alloc_zeros::<T>(len)
            .map_err(|e| driver_error(&format!("alloc {label}"), e))?;
        Ok(CudaBuffer {
            slice,
            _reservation: reservation,
            label: label.to_string(),
        })
    }

    /// A buffer holding a copy of `data`.
    pub fn upload<T: Element>(&self, data: &[T], label: &str) -> Result<CudaBuffer<T>, CudaError> {
        let mut buf = self.alloc_zeros::<T>(data.len(), label)?;
        self.write(&mut buf, data)?;
        Ok(buf)
    }

    /// Overwrite `buf` with `data`, which must have exactly `buf.len()` elements.
    pub fn write<T: Element>(&self, buf: &mut CudaBuffer<T>, data: &[T]) -> Result<(), CudaError> {
        if data.len() != buf.len() {
            return Err(CudaError::invalid(
                format!("write {}", buf.label),
                format!(
                    "{} host elements for a {}-element buffer",
                    data.len(),
                    buf.len()
                ),
            ));
        }
        let label = format!("upload {}", buf.label);
        self.stream()
            .memcpy_htod(data, buf.slice_mut())
            .map_err(|e| driver_error(&label, e))
    }

    /// Copy `buf` to the host after all queued work.
    pub fn download<T: Element>(&self, buf: &CudaBuffer<T>) -> Result<Vec<T>, CudaError> {
        let label = format!("download {}", buf.label);
        self.sync(&label)?;
        let mut host = vec![T::default(); buf.len()];
        self.stream()
            .memcpy_dtoh(buf.slice(), &mut host)
            .map_err(|e| driver_error(&label, e))?;
        self.sync(&label)?;
        Ok(host)
    }
}

use ojas_core::{BackendId, DeviceBuffer, OjasError};
use std::any::Any;
use std::sync::Arc;

/// A device-resident buffer backing an [`ojas_core::Tensor`].
pub struct CudaDeviceBuffer {
    pub(crate) slice: CudaSlice<u8>,
    pub(crate) len: usize,
    pub(crate) stream: Arc<cudarc::driver::CudaStream>,
    pub(crate) shadow_u32: Option<Arc<[u32]>>,
}

impl CudaDeviceBuffer {
    /// Return the optional host shadow u32 values if present.
    pub fn shadow_u32(&self) -> Option<&Arc<[u32]>> {
        self.shadow_u32.as_ref()
    }
}

impl std::fmt::Debug for CudaDeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDeviceBuffer({} bytes)", self.len)
    }
}

impl DeviceBuffer for CudaDeviceBuffer {
    fn backend(&self) -> BackendId {
        BackendId::Cuda
    }

    fn byte_len(&self) -> usize {
        self.len
    }

    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        if offset.checked_add(len).is_none_or(|end| end > self.len) {
            return Err(OjasError::OutOfRange {
                op: "CudaDeviceBuffer::read_bytes",
                detail: format!("offset {offset} + len {len} > byte_len {}", self.len),
            });
        }
        let view =
            self.slice
                .try_slice(offset..offset + len)
                .ok_or_else(|| OjasError::OutOfRange {
                    op: "CudaDeviceBuffer::read_bytes",
                    detail: format!("slice out of range: {offset}..{}", offset + len),
                })?;
        let mut out = vec![0u8; len];
        self.stream
            .memcpy_dtoh(&view, &mut out)
            .map_err(|e| OjasError::from(crate::runtime::driver_error("memcpy_dtoh", e)))?;
        self.stream
            .synchronize()
            .map_err(|e| OjasError::from(crate::runtime::driver_error("synchronize", e)))?;
        Ok(out)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
