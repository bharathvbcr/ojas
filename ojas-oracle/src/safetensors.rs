//! The oracle's view of a fixture safetensors file.
//!
//! The format is read by `ojas_io::SafeTensors`: header grammar and schema,
//! tiling of the data buffer, dtype widths and positioned reads all live
//! there. This module adds only the fixture policy: a header of at most
//! [`MAX_HEADER_BYTES`], and tensors of dtype `F32` or `I64` only, which is
//! what the generators write. Errors are `OjasError::OutOfRange`.

use ojas_core::OjasError;
use ojas_io::StDtype as IoDtype;

use crate::{bad, io};

/// Headers above this are refused before they are parsed. The tiny fixtures
/// have headers of a few KiB; the 124M export's is about 25 KiB.
pub const MAX_HEADER_BYTES: usize = 1024 * 1024;

/// The dtypes a fixture may hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StDtype {
    F32,
    I64,
}

impl StDtype {
    pub fn width(self) -> usize {
        match self {
            StDtype::F32 => 4,
            StDtype::I64 => 8,
        }
    }
}

/// One tensor's header entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StEntry {
    pub name: String,
    pub dtype: StDtype,
    pub shape: Vec<usize>,
}

/// A parsed fixture file borrowing its bytes.
#[derive(Debug)]
pub struct SafeTensors<'a> {
    inner: ojas_io::SafeTensors<'a>,
    /// In data order: by `data_offsets`, ties (zero-byte tensors) by name.
    entries: Vec<StEntry>,
}

impl<'a> SafeTensors<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, OjasError> {
        let prefix: [u8; 8] = bytes
            .get(..8)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| bad("safetensors file shorter than its length prefix"))?;
        if u64::from_le_bytes(prefix) > MAX_HEADER_BYTES as u64 {
            return Err(bad("safetensors header exceeds 1 MiB"));
        }
        let inner = ojas_io::SafeTensors::parse(bytes).map_err(io)?;
        let mut placed = Vec::new();
        for name in inner.names() {
            let info = inner.info(name).map_err(io)?;
            let dtype = match info.dtype {
                IoDtype::F32 => StDtype::F32,
                IoDtype::I64 => StDtype::I64,
                other => return Err(bad(&format!("{name}: unsupported fixture dtype {other:?}"))),
            };
            let shape = info
                .shape
                .iter()
                .map(|&d| usize::try_from(d))
                .collect::<Result<_, _>>()
                .map_err(|_| bad(&format!("{name}: shape exceeds usize")))?;
            placed.push((
                (info.begin, info.end),
                StEntry {
                    name: name.to_string(),
                    dtype,
                    shape,
                },
            ));
        }
        placed.sort_by_key(|(span, _)| *span);
        Ok(Self {
            inner,
            entries: placed.into_iter().map(|(_, e)| e).collect(),
        })
    }

    /// Entries in data order.
    pub fn entries(&self) -> &[StEntry] {
        &self.entries
    }

    pub fn entry(&self, name: &str) -> Option<&StEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.inner.metadata().get(key).map(String::as_str)
    }

    /// Metadata keys, sorted.
    pub fn metadata_keys(&self) -> impl Iterator<Item = &str> {
        self.inner.metadata().keys().map(String::as_str)
    }

    pub fn f32s(&self, name: &str) -> Result<Vec<f32>, OjasError> {
        self.inner.read_f32(name).map(|(_, v)| v).map_err(io)
    }

    pub fn i64s(&self, name: &str) -> Result<Vec<i64>, OjasError> {
        self.inner.read_i64(name).map(|(_, v)| v).map_err(io)
    }
}
