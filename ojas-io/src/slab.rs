//! Reading safetensors tensors straight into caller-provided storage.
//!
//! A [`SafeTensors`] already opens by reading the header only and looks one
//! tensor up by name ([`SafeTensors::info`]). This module adds the other half
//! a loader needs: a destination the caller owns.
//!
//! - [`SafeTensors::read_tensor_into`] fills one `&mut [u8]` of exactly the
//!   tensor's length with a positioned read. Nothing else is read or decoded.
//! - [`TensorSink`] hands the reader one destination per tensor, each at an
//!   alignment the caller names. [`SlabSink`] carves them out of one
//!   `&mut [u8]` the caller already holds, however that slice happens to be
//!   aligned. [`OwnedSlab`] is such a slab allocated against a
//!   [`Budget`], so the charge is made before the allocation and is refused,
//!   never clamped.
//! - [`SafeTensors::read_into_sink`] and [`SafeTensors::read_into_slab`] read
//!   a list of tensors through a sink. The plan is checked in full first
//!   (names, duplicates, alignment, and that the sink can hold every byte),
//!   so a slab that is too small is refused before the first read.
//!
//! The bytes land as stored in the file: little-endian, undecoded. A tensor
//! read through a slab is bit-equal to [`SafeTensors::read_bytes`] of the
//! same tensor. Padding between tensors is left as the caller had it.
//!
//! Nothing here uses `unsafe`: alignment is found with
//! [`pointer::align_offset`] on the slab's own address, and a destination a
//! sink returns is re-checked for length and alignment before it is read into.

use crate::error::IoError;
use crate::safetensors::SafeTensors;
use ojas_core::{Budget, Scratch};
use std::collections::BTreeSet;

/// Largest alignment a slab request may name: 1 GiB. A page is 4 or 16 KiB; a
/// bound keeps `align - 1` slack from being an arbitrary number.
pub const MAX_SLAB_ALIGN: usize = 1 << 30;

fn check_align(align: usize) -> Result<(), IoError> {
    if align == 0 || !align.is_power_of_two() || align > MAX_SLAB_ALIGN {
        return Err(IoError::new(format!(
            "slab alignment {align} is not a power of two in 1..={MAX_SLAB_ALIGN}"
        )));
    }
    Ok(())
}

/// Where the reader puts each tensor.
///
/// A sink is asked once to [`reserve`](Self::reserve) the whole plan, then for
/// one [`slot`](Self::slot) per non-duplicate tensor, in plan order. The reader
/// does not trust a sink: it refuses a slot whose length is not the one asked
/// for or whose address is not aligned.
pub trait TensorSink {
    /// Called once before any tensor is read. `total` is the bytes the plan
    /// needs starting from an `align`-aligned address, padding included.
    /// Refuse here if the sink cannot hold it, so no read is wasted. The
    /// default accepts everything and leaves refusal to [`Self::slot`].
    fn reserve(&mut self, total: u64, align: usize) -> Result<(), IoError> {
        let _ = (total, align);
        Ok(())
    }

    /// Storage for tensor `name`: exactly `len` bytes starting at an address
    /// that is a multiple of `align`. A zero-length tensor gets an empty slice.
    fn slot(&mut self, name: &str, len: usize, align: usize) -> Result<&mut [u8], IoError>;
}

/// One tensor's place in a slab.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlabEntry {
    pub name: String,
    /// Byte offset of the tensor from the start of the slab slice.
    pub offset: usize,
    pub len: usize,
}

/// A [`TensorSink`] that carves aligned slots out of a caller's slice, front
/// to back. The slice need not be aligned: each slot's start is rounded up
/// from the slice's real address, so an unaligned slab loses up to `align - 1`
/// bytes at the front and nothing else.
#[derive(Debug)]
pub struct SlabSink<'a> {
    slab: &'a mut [u8],
    at: usize,
    entries: Vec<SlabEntry>,
}

impl<'a> SlabSink<'a> {
    pub fn new(slab: &'a mut [u8]) -> Self {
        Self {
            slab,
            at: 0,
            entries: Vec::new(),
        }
    }

    /// Bytes of the slab consumed so far, padding included.
    pub fn used(&self) -> usize {
        self.at
    }

    /// The slots handed out so far, in order.
    pub fn entries(&self) -> &[SlabEntry] {
        &self.entries
    }

    pub fn into_entries(self) -> Vec<SlabEntry> {
        self.entries
    }

    /// Padding needed at the cursor to reach an `align`-aligned address.
    fn pad_for(&self, align: usize) -> Result<usize, IoError> {
        let rest = &self.slab[self.at..];
        match rest.as_ptr().align_offset(align) {
            usize::MAX => Err(IoError::new(format!(
                "the slab cannot be aligned to {align} bytes"
            ))),
            pad => Ok(pad),
        }
    }
}

impl TensorSink for SlabSink<'_> {
    fn reserve(&mut self, total: u64, align: usize) -> Result<(), IoError> {
        check_align(align)?;
        let pad = self.pad_for(align)?;
        let rest = self.slab.len() - self.at;
        let need = u64::try_from(pad)
            .ok()
            .and_then(|p| p.checked_add(total))
            .ok_or_else(|| IoError::new("slab size overflows"))?;
        if need > rest as u64 {
            return Err(IoError::new(format!(
                "the plan needs {need} bytes ({pad} to align the slab start to {align}, \
                 {total} of tensors and padding); the slab has {rest}"
            )));
        }
        Ok(())
    }

    fn slot(&mut self, name: &str, len: usize, align: usize) -> Result<&mut [u8], IoError> {
        check_align(align)?;
        if len == 0 {
            self.entries.push(SlabEntry {
                name: name.to_string(),
                offset: self.at,
                len: 0,
            });
            return Ok(&mut []);
        }
        let pad = self.pad_for(align)?;
        let rest = self.slab.len() - self.at;
        let end = pad
            .checked_add(len)
            .filter(|&end| end <= rest)
            .ok_or_else(|| {
                IoError::new(format!(
                    "{name}: {len} bytes (after {pad} of alignment padding) do not fit the {rest} bytes left in the slab"
                ))
            })?;
        let offset = self.at + pad;
        self.entries.push(SlabEntry {
            name: name.to_string(),
            offset,
            len,
        });
        let start = self.at;
        self.at += end;
        Ok(&mut self.slab[start + pad..start + end])
    }
}

/// A slab allocated against a [`Budget`]: `len` bytes at an address that is a
/// multiple of `align`.
///
/// The allocation is `len + align - 1` bytes, because that is what makes an
/// aligned window possible without `unsafe`; the whole allocation is charged,
/// before it is made, and released when the slab drops. The window starts
/// zeroed.
#[derive(Debug)]
pub struct OwnedSlab {
    buf: Scratch<u8>,
    lead: usize,
    len: usize,
}

impl OwnedSlab {
    pub fn new(budget: &Budget, len: usize, align: usize) -> Result<Self, IoError> {
        check_align(align)?;
        let total = len
            .checked_add(align - 1)
            .ok_or_else(|| IoError::new("slab size overflows"))?;
        let buf =
            Scratch::<u8>::try_alloc(total, budget).map_err(|e| IoError::new(e.to_string()))?;
        let lead = buf.as_slice().as_ptr().align_offset(align);
        if lead >= align {
            return Err(IoError::new(format!(
                "the allocation cannot be aligned to {align} bytes"
            )));
        }
        Ok(Self { buf, lead, len })
    }

    /// The aligned window.
    pub fn as_slice(&self) -> &[u8] {
        &self.buf.as_slice()[self.lead..self.lead + self.len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let (lead, len) = (self.lead, self.len);
        &mut self.buf.as_mut_slice()[lead..lead + len]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl SafeTensors<'_> {
    /// Fill `dst` with all of tensor `name`, as stored. `dst.len()` must equal
    /// the tensor's byte length exactly: a short or long destination is
    /// refused, so a caller cannot half-read a tensor by mistake. Only that
    /// tensor's range is read.
    pub fn read_tensor_into(&self, name: &str, dst: &mut [u8]) -> Result<(), IoError> {
        let want = self.tensor_len(name)?;
        if dst.len() as u64 != want {
            return Err(IoError::new(format!(
                "{name}: destination is {} bytes, the tensor is {want}",
                dst.len()
            )));
        }
        self.read_into(name, 0, dst)
    }

    /// Bytes of tensor `name`, from the header alone.
    fn tensor_len(&self, name: &str) -> Result<u64, IoError> {
        let info = self.info(name)?;
        Ok(info.end - info.begin)
    }

    /// Bytes [`Self::read_into_sink`] needs for `names` from an
    /// `align`-aligned start: each tensor starts at a multiple of `align`
    /// (a zero-length tensor takes no padding). A slice of unknown alignment
    /// needs up to `align - 1` more. Refuses unknown or repeated names, a bad
    /// alignment, and a total that overflows `u64`.
    pub fn slab_len(&self, names: &[&str], align: usize) -> Result<u64, IoError> {
        Ok(self.plan(names, align)?.1)
    }

    /// Lengths of `names` in order, and the aligned total.
    fn plan(&self, names: &[&str], align: usize) -> Result<(Vec<usize>, u64), IoError> {
        check_align(align)?;
        let a = align as u64;
        let mut seen = BTreeSet::new();
        let mut lens = Vec::with_capacity(names.len());
        let mut cursor = 0u64;
        for &name in names {
            if !seen.insert(name) {
                return Err(IoError::new(format!("{name:?} is listed twice")));
            }
            let len = self.tensor_len(name)?;
            lens.push(
                usize::try_from(len)
                    .map_err(|_| IoError::new(format!("{name}: tensor too large for this host")))?,
            );
            if len > 0 {
                cursor = cursor
                    .checked_add(a - 1)
                    .map(|c| c & !(a - 1))
                    .and_then(|c| c.checked_add(len))
                    .ok_or_else(|| IoError::new(format!("{name}: slab size overflows u64")))?;
            }
        }
        Ok((lens, cursor))
    }

    /// Read `names`, in order, each into the slot `sink` provides.
    ///
    /// Checked before any byte is read: every name exists and appears once,
    /// `align` is valid, and `sink` accepts the total. A slot of the wrong
    /// length or address is refused before it is written to. If a read fails
    /// part-way the destinations of the tensors already read hold their
    /// bytes and the rest are unspecified.
    pub fn read_into_sink(
        &self,
        names: &[&str],
        align: usize,
        sink: &mut (impl TensorSink + ?Sized),
    ) -> Result<(), IoError> {
        let (lens, total) = self.plan(names, align)?;
        sink.reserve(total, align)?;
        for (&name, &len) in names.iter().zip(&lens) {
            let dst = sink.slot(name, len, align)?;
            if dst.len() != len {
                return Err(IoError::new(format!(
                    "{name}: the sink returned {} bytes for a {len}-byte tensor",
                    dst.len()
                )));
            }
            if len > 0 && dst.as_ptr().align_offset(align) != 0 {
                return Err(IoError::new(format!(
                    "{name}: the sink returned storage not aligned to {align} bytes"
                )));
            }
            self.read_tensor_into(name, dst)?;
        }
        Ok(())
    }

    /// [`Self::read_into_sink`] through a [`SlabSink`] over `slab`, returning
    /// where each tensor landed. `slab` may have any alignment; size it with
    /// [`Self::slab_len`] plus `align - 1`, or use an [`OwnedSlab`] of
    /// exactly [`Self::slab_len`].
    pub fn read_into_slab(
        &self,
        names: &[&str],
        align: usize,
        slab: &mut [u8],
    ) -> Result<Vec<SlabEntry>, IoError> {
        let mut sink = SlabSink::new(slab);
        self.read_into_sink(names, align, &mut sink)?;
        Ok(sink.into_entries())
    }
}
