//! [`CudaBuffer<T>`]: a typed, length-checked device allocation held against
//! the runtime's [`crate::budget::AllocBudget`].
//!
//! Uploads and writes must match the buffer's length exactly; downloads
//! wait with the runtime's bounded sync before and after the copy, because
//! cudarc's device-to-host copy into a `Vec` does not synchronise
//! (`cudarc/src/driver/safe/core.rs:1434-1450`).

use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};

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
