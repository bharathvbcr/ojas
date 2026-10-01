//! Strict safetensors reader and writer.
//!
//! A file is an 8-byte little-endian header length `n`, then `n` bytes of
//! UTF-8 JSON, then the data buffer. `n` must be in `1..=`[`MAX_HEADER_BYTES`]
//! (100_000_000, the upstream reference cap). Each tensor is
//! `dtype`, `shape`, and `data_offsets`. The tensors must tile the data
//! buffer with no gap and no overlap. Opening a file reads the header only;
//! [`SafeTensors::read_into`] copies one range with a positioned read.

use crate::error::IoError;
use crate::json::{self, Json};
use crate::replace::replace_file;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Largest header accepted. The format's reference implementation uses
/// 100_000_000, not 100 MiB (104_857_600).
pub const MAX_HEADER_BYTES: u64 = 100_000_000;

/// Dtypes this crate reads and writes. Enough for f32 weights and integer
/// token ids. Other safetensors dtypes are rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StDtype {
    F32,
    I64,
    U16,
}

impl StDtype {
    pub fn size(self) -> u64 {
        match self {
            StDtype::U16 => 2,
            StDtype::F32 => 4,
            StDtype::I64 => 8,
        }
    }

    fn tag(self) -> &'static str {
        match self {
            StDtype::F32 => "F32",
            StDtype::I64 => "I64",
            StDtype::U16 => "U16",
        }
    }

    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "F32" => Ok(StDtype::F32),
            "I64" => Ok(StDtype::I64),
            "U16" => Ok(StDtype::U16),
            other => Err(format!("unsupported dtype {other:?}")),
        }
    }
}

/// One tensor's header entry. Offsets are byte offsets into the data buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: StDtype,
    pub shape: Vec<u64>,
    pub begin: u64,
    pub end: u64,
}

/// One tensor to write. `data.len()` must equal the shape product times the
/// dtype width. Products use checked arithmetic.
pub struct TensorOut<'a> {
    pub name: &'a str,
    pub dtype: StDtype,
    pub shape: &'a [u64],
    pub data: &'a [u8],
}

#[derive(Debug)]
enum Source {
    File {
        file: File,
        data_start: u64,
    },
    /// Data buffer only (the bytes after the header). Used by [`SafeTensors::parse`].
    Memory {
        data: Vec<u8>,
    },
}

/// A validated safetensors header, plus a handle for positioned data reads.
#[derive(Debug)]
pub struct SafeTensors {
    source: Source,
    tensors: BTreeMap<String, TensorInfo>,
    metadata: BTreeMap<String, String>,
}

impl SafeTensors {
    /// Validate `bytes` as a whole file. Does not touch the filesystem.
    pub fn parse(bytes: &[u8]) -> Result<Self, IoError> {
        let (header, data_start, data_len) = split_header(bytes, bytes.len() as u64)?;
        let (tensors, metadata) = parse_header(&header, data_len)?;
        let data = bytes
            .get(data_start as usize..)
            .ok_or_else(|| IoError::new("truncated file"))?
            .to_vec();
        Ok(Self {
            source: Source::Memory { data },
            tensors,
            metadata,
        })
    }

    /// Open `path` and validate its header. Tensor bytes stay on disk.
    pub fn open(path: &Path) -> Result<Self, IoError> {
        let what = path.display().to_string();
        let mut file = File::open(path).map_err(|e| IoError::new(format!("{what}: {e}")))?;
        let file_len = file
            .metadata()
            .map_err(|e| IoError::new(format!("{what}: {e}")))?
            .len();
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|e| IoError::new(format!("{what}: header length: {e}")))?;
        let n = u64::from_le_bytes(prefix);
        check_header_len(n, file_len)?;
        let n_us = usize::try_from(n).map_err(|_| IoError::new("header length exceeds usize"))?;
        let mut header = Vec::new();
        header
            .try_reserve_exact(n_us)
            .map_err(|_| IoError::new(format!("header allocation of {n_us} bytes refused")))?;
        header.resize(n_us, 0);
        file.read_exact(&mut header)
            .map_err(|e| IoError::new(format!("{what}: header: {e}")))?;
        let data_start = 8 + n;
        let data_len = file_len - data_start;
        let (tensors, metadata) = parse_header(&header, data_len)
            .map_err(|e| IoError::new(format!("{what}: {}", e.detail())))?;
        Ok(Self {
            source: Source::File { file, data_start },
            tensors,
            metadata,
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo, IoError> {
        self.tensors
            .get(name)
            .ok_or_else(|| IoError::new(format!("no tensor {name:?}")))
    }

    /// Copy `dst.len()` bytes of tensor `name`, starting `offset` bytes into
    /// that tensor. The read is positioned: the rest of the file is not loaded.
    /// `offset` past the tensor, or a range that would run past it, is an error.
    pub fn read_into(&self, name: &str, offset: u64, dst: &mut [u8]) -> Result<(), IoError> {
        let info = self.info(name)?;
        let span = info.end - info.begin;
        let need = u64::try_from(dst.len())
            .map_err(|_| IoError::new(format!("{name}: read length exceeds u64")))?;
        let end = offset
            .checked_add(need)
            .ok_or_else(|| IoError::new(format!("{name}: read offset overflows")))?;
        if end > span {
            return Err(IoError::new(format!(
                "{name}: read [{offset}, {end}) outside tensor length {span}"
            )));
        }
        if dst.is_empty() {
            return Ok(());
        }
        let at = info
            .begin
            .checked_add(offset)
            .ok_or_else(|| IoError::new(format!("{name}: data offset overflows")))?;
        match &self.source {
            Source::Memory { data } => {
                let start = usize::try_from(at)
                    .map_err(|_| IoError::new(format!("{name}: offset exceeds usize")))?;
                let stop = start + dst.len();
                let src = data
                    .get(start..stop)
                    .ok_or_else(|| IoError::new(format!("{name}: truncated tensor data")))?;
                dst.copy_from_slice(src);
                Ok(())
            }
            Source::File { file, data_start } => {
                let abs = data_start
                    .checked_add(at)
                    .ok_or_else(|| IoError::new(format!("{name}: file offset overflows")))?;
                read_exact_at(file, dst, abs)
                    .map_err(|e| IoError::new(format!("{name}: data: {e}")))
            }
        }
    }

    pub fn read_bytes(&self, name: &str) -> Result<Vec<u8>, IoError> {
        let info = self.info(name)?;
        let len = usize::try_from(info.end - info.begin)
            .map_err(|_| IoError::new(format!("{name}: tensor too large for this host")))?;
        let mut dst = Vec::new();
        dst.try_reserve_exact(len)
            .map_err(|_| IoError::new(format!("{name}: allocation of {len} bytes refused")))?;
        dst.resize(len, 0);
        self.read_into(name, 0, &mut dst)?;
        Ok(dst)
    }

    pub fn read_f32(&self, name: &str) -> Result<(Vec<u64>, Vec<f32>), IoError> {
        self.read_typed(name, StDtype::F32, 4, |c| {
            f32::from_le_bytes([c[0], c[1], c[2], c[3]])
        })
    }

    pub fn read_i64(&self, name: &str) -> Result<(Vec<u64>, Vec<i64>), IoError> {
        self.read_typed(name, StDtype::I64, 8, |c| {
            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
        })
    }

    pub fn read_u16(&self, name: &str) -> Result<(Vec<u64>, Vec<u16>), IoError> {
        self.read_typed(name, StDtype::U16, 2, |c| u16::from_le_bytes([c[0], c[1]]))
    }

    fn read_typed<T, F>(
        &self,
        name: &str,
        dtype: StDtype,
        width: usize,
        decode: F,
    ) -> Result<(Vec<u64>, Vec<T>), IoError>
    where
        F: Fn(&[u8]) -> T,
    {
        let info = self.info(name)?;
        if info.dtype != dtype {
            return Err(IoError::new(format!(
                "{name}: expected {dtype:?}, found {:?}",
                info.dtype
            )));
        }
        let bytes = self.read_bytes(name)?;
        if bytes.len() % width != 0 {
            return Err(IoError::new(format!(
                "{name}: data is not a multiple of {width}"
            )));
        }
        let values = bytes.chunks_exact(width).map(decode).collect();
        Ok((info.shape.clone(), values))
    }
}

/// Write tensors in `items` order. The header is padded with spaces so its
/// length is a multiple of 8. The reader accepts that padding. The target is
/// replaced by rename, so a failed write leaves the previous file intact.
pub fn write_safetensors(
    path: &Path,
    items: &[TensorOut<'_>],
    metadata: &[(&str, &str)],
) -> Result<(), IoError> {
    replace_file(path, &encode_safetensors(items, metadata)?)
}

pub fn encode_safetensors(
    items: &[TensorOut<'_>],
    metadata: &[(&str, &str)],
) -> Result<Vec<u8>, IoError> {
    let mut seen = BTreeMap::<&str, ()>::new();
    let mut cursor = 0u64;
    let mut placed: Vec<(&TensorOut<'_>, u64, u64)> = Vec::with_capacity(items.len());
    for item in items {
        if item.name.is_empty() {
            return Err(IoError::new("tensor name is empty"));
        }
        if item.name == "__metadata__" {
            return Err(IoError::new("tensor name __metadata__ is reserved"));
        }
        if seen.insert(item.name, ()).is_some() {
            return Err(IoError::new(format!("duplicate key {:?}", item.name)));
        }
        let nbytes = byte_len(item.name, item.shape, item.dtype)?;
        let data_len = u64::try_from(item.data.len())
            .map_err(|_| IoError::new(format!("{:?}: data length exceeds u64", item.name)))?;
        if nbytes != data_len {
            return Err(IoError::new(format!(
                "{:?}: data is {data_len} bytes, shape times {:?} is {nbytes}",
                item.name, item.dtype
            )));
        }
        let end = cursor
            .checked_add(nbytes)
            .ok_or_else(|| IoError::new(format!("{:?}: data offsets overflow", item.name)))?;
        placed.push((item, cursor, end));
        cursor = end;
    }
    let mut seen_meta = BTreeMap::<&str, ()>::new();
    for (k, _) in metadata {
        if seen_meta.insert(*k, ()).is_some() {
            return Err(IoError::new(format!("duplicate metadata key {k:?}")));
        }
    }

    let mut header = String::new();
    header.push('{');
    let mut first = true;
    if !metadata.is_empty() {
        header.push_str("\"__metadata__\":{");
        for (i, (k, v)) in metadata.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            push_json_str(&mut header, k);
            header.push(':');
            push_json_str(&mut header, v);
        }
        header.push('}');
        first = false;
    }
    for (item, begin, end) in &placed {
        if !first {
            header.push(',');
        }
        first = false;
        push_json_str(&mut header, item.name);
        header.push_str(":{\"dtype\":");
        push_json_str(&mut header, item.dtype.tag());
        header.push_str(",\"shape\":[");
        for (i, d) in item.shape.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            header.push_str(&d.to_string());
        }
        header.push_str("],\"data_offsets\":[");
        header.push_str(&begin.to_string());
        header.push(',');
        header.push_str(&end.to_string());
        header.push_str("]}");
    }
    header.push('}');
    let mut header_bytes = header.into_bytes();
    let pad = (8 - (header_bytes.len() % 8)) % 8;
    header_bytes.extend(std::iter::repeat_n(b' ', pad));
    let n =
        u64::try_from(header_bytes.len()).map_err(|_| IoError::new("header length exceeds u64"))?;
    if n == 0 || n > MAX_HEADER_BYTES {
        return Err(IoError::new(format!(
            "header length {n} outside 1..={MAX_HEADER_BYTES}"
        )));
    }
    let mut out = Vec::new();
    let total = 8usize
        .checked_add(header_bytes.len())
        .and_then(|v| v.checked_add(usize::try_from(cursor).unwrap_or(usize::MAX)))
        .ok_or_else(|| IoError::new("encoded file length overflows"))?;
    out.try_reserve_exact(total)
        .map_err(|_| IoError::new("refused allocation for safetensors encode"))?;
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&header_bytes);
    for (item, _, _) in &placed {
        out.extend_from_slice(item.data);
    }
    Ok(out)
}

fn byte_len(name: &str, shape: &[u64], dtype: StDtype) -> Result<u64, IoError> {
    let numel = shape
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| IoError::new(format!("{name:?}: shape product overflows")))?;
    numel
        .checked_mul(dtype.size())
        .ok_or_else(|| IoError::new(format!("{name:?}: byte size overflows")))
}

fn push_json_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn split_header(bytes: &[u8], file_len: u64) -> Result<(Vec<u8>, u64, u64), IoError> {
    let prefix = bytes
        .get(..8)
        .ok_or_else(|| IoError::new("truncated file: header length"))?;
    let mut len8 = [0u8; 8];
    len8.copy_from_slice(prefix);
    let n = u64::from_le_bytes(len8);
    check_header_len(n, file_len)?;
    let n_us = usize::try_from(n).map_err(|_| IoError::new("header length exceeds usize"))?;
    let header_end = 8usize
        .checked_add(n_us)
        .ok_or_else(|| IoError::new("header end overflows"))?;
    let header = bytes
        .get(8..header_end)
        .ok_or_else(|| IoError::new("truncated file: header runs past the end"))?
        .to_vec();
    let data_start = 8 + n;
    Ok((header, data_start, file_len - data_start))
}

fn check_header_len(n: u64, file_len: u64) -> Result<(), IoError> {
    if n == 0 || n > MAX_HEADER_BYTES {
        return Err(IoError::new(format!(
            "header length {n} outside 1..={MAX_HEADER_BYTES}"
        )));
    }
    let rest = file_len.saturating_sub(8);
    if n > rest {
        return Err(IoError::new(format!(
            "header length {n} runs past the end of a {file_len}-byte file"
        )));
    }
    Ok(())
}

type Header = (BTreeMap<String, TensorInfo>, BTreeMap<String, String>);

fn parse_header(bytes: &[u8], data_len: u64) -> Result<Header, IoError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| IoError::new(format!("header is not UTF-8: {e}")))?;
    let root = json::parse(text).map_err(IoError::new)?;
    let Json::Object(entries) = root else {
        return Err(IoError::new("header is not a JSON object"));
    };
    let mut tensors = BTreeMap::new();
    let mut metadata = BTreeMap::new();
    for (name, v) in entries {
        if name == "__metadata__" {
            let Json::Object(m) = v else {
                return Err(IoError::new("__metadata__ is not an object"));
            };
            for (k, v) in m {
                let Json::Str(s) = v else {
                    return Err(IoError::new(format!(
                        "__metadata__ value for {k:?} is not a string"
                    )));
                };
                metadata.insert(k, s);
            }
            continue;
        }
        if name.is_empty() {
            return Err(IoError::new("tensor name is empty"));
        }
        let info = tensor_info(&name, v)?;
        tensors.insert(name, info);
    }
    check_tiling(&tensors, data_len)?;
    Ok((tensors, metadata))
}

fn tensor_info(name: &str, v: Json) -> Result<TensorInfo, IoError> {
    let Json::Object(fields) = v else {
        return Err(IoError::new(format!("{name}: entry is not an object")));
    };
    let (mut dtype, mut shape, mut offsets) = (None, None, None);
    for (k, v) in fields {
        match (k.as_str(), v) {
            ("dtype", Json::Str(s)) => {
                dtype = Some(StDtype::parse(&s).map_err(|e| IoError::new(format!("{name}: {e}")))?);
            }
            ("shape", Json::Array(a)) => {
                let mut dims = Vec::with_capacity(a.len());
                for d in a {
                    match d {
                        Json::Uint(n) => dims.push(n),
                        _ => {
                            return Err(IoError::new(format!("{name}: shape holds a non-integer")))
                        }
                    }
                }
                shape = Some(dims);
            }
            ("data_offsets", Json::Array(a)) => match a.as_slice() {
                [Json::Uint(b), Json::Uint(e)] => offsets = Some((*b, *e)),
                _ => {
                    return Err(IoError::new(format!(
                        "{name}: data_offsets is not [begin, end]"
                    )))
                }
            },
            (k, _) => {
                return Err(IoError::new(format!(
                    "{name}: unexpected or mistyped field {k:?}"
                )))
            }
        }
    }
    let (Some(dtype), Some(shape), Some((begin, end))) = (dtype, shape, offsets) else {
        return Err(IoError::new(format!(
            "{name}: needs dtype, shape and data_offsets"
        )));
    };
    Ok(TensorInfo {
        dtype,
        shape,
        begin,
        end,
    })
}

fn check_tiling(tensors: &BTreeMap<String, TensorInfo>, data_len: u64) -> Result<(), IoError> {
    let mut ranges: Vec<(u64, u64, &str)> = Vec::with_capacity(tensors.len());
    for (name, t) in tensors {
        if t.begin > t.end || t.end > data_len {
            return Err(IoError::new(format!(
                "{name}: data_offsets [{}, {}] outside the {data_len}-byte data buffer",
                t.begin, t.end
            )));
        }
        let numel = t
            .shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| IoError::new(format!("{name}: shape product overflows")))?;
        let want = numel
            .checked_mul(t.dtype.size())
            .ok_or_else(|| IoError::new(format!("{name}: byte size overflows")))?;
        let got = t.end - t.begin;
        if got != want {
            return Err(IoError::new(format!(
                "{name}: {got} bytes in data_offsets, but shape {:?} x {:?} is {want}",
                t.shape, t.dtype
            )));
        }
        ranges.push((t.begin, t.end, name));
    }
    ranges.sort_unstable();
    let mut at = 0u64;
    for (b, e, name) in ranges {
        if b != at {
            return Err(IoError::new(format!(
                "{name}: starts at byte {b}, expected {at} (gap or overlap; tensors must tile the data buffer)"
            )));
        }
        at = e;
    }
    if at != data_len {
        return Err(IoError::new(format!(
            "the tensors cover {at} of {data_len} data bytes; the rest is an unindexed gap"
        )));
    }
    Ok(())
}

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut rest = buf;
        let mut at = offset;
        while !rest.is_empty() {
            let n = file.seek_read(rest, at)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "short read",
                ));
            }
            let n_u64 = u64::try_from(n).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "read length exceeds u64")
            })?;
            at = at.checked_add(n_u64).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "file offset overflows")
            })?;
            rest = &mut rest[n..];
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, offset);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "positioned read needs unix or windows",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{tmp, tmp_dir, Mix};

    fn frame(header: &[u8], data: &[u8]) -> Vec<u8> {
        let mut v = (header.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(header);
        v.extend_from_slice(data);
        v
    }

    #[test]
    fn truncated_file_is_an_error() {
        let err = SafeTensors::parse(&[1, 2, 3]).unwrap_err();
        assert!(
            err.detail().contains("truncated") || err.detail().contains("header length"),
            "{err}"
        );
        let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut past = frame(header, &1.0f32.to_le_bytes());
        let claimed = (header.len() as u64) + 50;
        past[..8].copy_from_slice(&claimed.to_le_bytes());
        assert!(SafeTensors::parse(&past)
            .unwrap_err()
            .detail()
            .contains("past the end"));
    }

    #[test]
    fn duplicate_key_is_an_error() {
        let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let err = SafeTensors::parse(&frame(header, &1.0f32.to_le_bytes())).unwrap_err();
        assert!(err.detail().contains("duplicate key"), "{err}");
    }

    #[test]
    fn overlapping_range_is_an_error() {
        let header = br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}"#;
        let err = SafeTensors::parse(&frame(header, &[0u8; 6])).unwrap_err();
        assert!(
            err.detail().contains("overlap") || err.detail().contains("gap"),
            "{err}"
        );
    }

    #[test]
    fn huge_header_length_is_an_error() {
        for n in [MAX_HEADER_BYTES + 1, u64::MAX] {
            let err = SafeTensors::parse(&n.to_le_bytes()).unwrap_err();
            assert!(err.detail().contains("outside 1..="), "{n}: {err}");
        }
        assert_eq!(MAX_HEADER_BYTES, 100_000_000);
    }

    #[test]
    fn shape_product_overflow_is_an_error() {
        let header =
            br#"{"a":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#;
        let err = SafeTensors::parse(&frame(header, &[])).unwrap_err();
        assert!(err.detail().contains("overflow"), "{err}");
    }

    #[test]
    fn round_trip_small_f32_and_read_into_one_range() {
        let values = [1.5f32, -2.25, 3.0];
        let mut data = Vec::new();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let path = tmp("rt");
        write_safetensors(
            &path.0,
            &[TensorOut {
                name: "w",
                dtype: StDtype::F32,
                shape: &[3],
                data: &data,
            }],
            &[("format", "ojas")],
        )
        .unwrap();
        let st = SafeTensors::open(&path.0).unwrap();
        assert_eq!(st.read_f32("w").unwrap(), (vec![3], values.to_vec()));
        assert_eq!(st.metadata()["format"], "ojas");
        let mut one = [0u8; 4];
        st.read_into("w", 4, &mut one).unwrap();
        assert_eq!(f32::from_le_bytes(one), -2.25);
        let file_len = std::fs::metadata(&path.0).unwrap().len();
        assert!(file_len > 12);
        assert!(file_len < 10_000);
    }

    #[test]
    fn round_trip_i64_and_u16() {
        let mut i64_data = Vec::new();
        for v in [0i64, -7, i64::MAX] {
            i64_data.extend_from_slice(&v.to_le_bytes());
        }
        let u16_data = 42u16.to_le_bytes();
        let bytes = encode_safetensors(
            &[
                TensorOut {
                    name: "ids",
                    dtype: StDtype::I64,
                    shape: &[3],
                    data: &i64_data,
                },
                TensorOut {
                    name: "tok",
                    dtype: StDtype::U16,
                    shape: &[1],
                    data: &u16_data,
                },
            ],
            &[],
        )
        .unwrap();
        let st = SafeTensors::parse(&bytes).unwrap();
        assert_eq!(st.read_i64("ids").unwrap().1, vec![0, -7, i64::MAX]);
        assert_eq!(st.read_u16("tok").unwrap().1, vec![42]);
    }

    #[test]
    fn ten_malformed_byte_strings_return_err() {
        let one = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let f32_one = 1.0f32.to_le_bytes();
        let dup = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let overlap = br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}"#;
        let overflow =
            br#"{"a":{"dtype":"U16","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#;
        let mut utf8_header = br#"{"#.to_vec();
        utf8_header.push(0xff);
        utf8_header.extend_from_slice(br#"":1}"#);
        let mut past = frame(one, &f32_one);
        past[..8].copy_from_slice(&((one.len() as u64) + 8).to_le_bytes());
        let cases: [Vec<u8>; 10] = [
            vec![],
            vec![1, 2, 3],
            0u64.to_le_bytes().to_vec(),
            u64::MAX.to_le_bytes().to_vec(),
            frame(&utf8_header, &[]),
            frame(dup, &f32_one),
            frame(overlap, &[0u8; 6]),
            frame(one, &[0u8; 8]),
            frame(overflow, &[]),
            past,
        ];
        assert_eq!(cases.len(), 10);
        for (i, bytes) in cases.iter().enumerate() {
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                SafeTensors::parse(bytes)
            }));
            let parsed = caught.unwrap_or_else(|_| panic!("case {i} panicked"));
            assert!(parsed.is_err(), "case {i} returned Ok");
        }
    }

    fn write_w(path: &Path, values: &[f32]) {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shape = [values.len() as u64];
        write_safetensors(
            path,
            &[TensorOut {
                name: "w",
                dtype: StDtype::F32,
                shape: &shape,
                data: &data,
            }],
            &[],
        )
        .unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rewrite_leaves_an_open_reader_on_the_old_file() {
        let path = tmp("swap");
        write_w(&path.0, &[1.0, 2.0]);
        let old = SafeTensors::open(&path.0).unwrap();
        write_w(&path.0, &[3.0, 4.0]);
        assert_eq!(old.read_f32("w").unwrap().1, vec![1.0, 2.0]);
        let new = SafeTensors::open(&path.0).unwrap();
        assert_eq!(new.read_f32("w").unwrap().1, vec![3.0, 4.0]);
    }

    #[test]
    fn failed_write_leaves_the_target_and_no_temp_file() {
        let dir = tmp_dir("st-fail");
        let target = dir.0.join("model.safetensors");
        write_w(&target, &[1.0]);
        let blocked = dir.0.join("sub");
        std::fs::create_dir(&blocked).unwrap();
        std::fs::write(blocked.join("keep"), b"x").unwrap();
        let data = 1.0f32.to_le_bytes();
        let item = [TensorOut {
            name: "w",
            dtype: StDtype::F32,
            shape: &[1],
            data: &data,
        }];
        assert!(write_safetensors(&blocked, &item, &[]).is_err());
        let bad = [TensorOut {
            name: "w",
            dtype: StDtype::F32,
            shape: &[2],
            data: &data,
        }];
        assert!(write_safetensors(&target, &bad, &[]).is_err());
        let mut names: Vec<_> = std::fs::read_dir(&dir.0)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names, ["model.safetensors", "sub"]);
        assert_eq!(
            SafeTensors::open(&target).unwrap().read_f32("w").unwrap().1,
            vec![1.0]
        );
    }

    #[test]
    fn empty_tensor_name_is_an_error_like_the_writer() {
        let header = br#"{"":{"dtype":"U16","shape":[1],"data_offsets":[0,2]}}"#;
        let err = SafeTensors::parse(&frame(header, &[0, 0])).unwrap_err();
        assert!(err.detail().contains("empty"), "{err}");
    }

    struct Owned {
        name: String,
        dtype: StDtype,
        shape: Vec<u64>,
        data: Vec<u8>,
    }

    type Meta = Vec<(String, String)>;

    const SOUP: [char; 10] = ['a', 'Z', '0', '.', '"', '\\', '\n', '\u{1}', 'é', '😀'];

    fn soup_string(rng: &mut Mix, min: usize) -> String {
        let len = min + rng.below(6);
        (0..len).map(|_| SOUP[rng.below(SOUP.len())]).collect()
    }

    fn random_tensors(rng: &mut Mix) -> (Vec<Owned>, Meta) {
        let mut seen = std::collections::BTreeSet::new();
        let mut out = Vec::new();
        for _ in 0..rng.below(5) {
            let name = soup_string(rng, 1);
            if !seen.insert(name.clone()) {
                continue;
            }
            let dtype = [StDtype::F32, StDtype::I64, StDtype::U16][rng.below(3)];
            let shape: Vec<u64> = (0..rng.below(5))
                .map(|_| match rng.below(5) {
                    0 => 0,
                    _ => 1 + rng.below(4) as u64,
                })
                .collect();
            let n = shape.iter().product::<u64>() * dtype.size();
            let data = (0..n).map(|_| rng.next() as u8).collect();
            out.push(Owned {
                name,
                dtype,
                shape,
                data,
            });
        }
        let mut keys = std::collections::BTreeSet::new();
        let meta = (0..rng.below(3))
            .map(|_| (soup_string(rng, 0), soup_string(rng, 0)))
            .filter(|(k, _)| keys.insert(k.clone()))
            .collect();
        (out, meta)
    }

    fn items(t: &[Owned]) -> Vec<TensorOut<'_>> {
        t.iter()
            .map(|o| TensorOut {
                name: &o.name,
                dtype: o.dtype,
                shape: &o.shape,
                data: &o.data,
            })
            .collect()
    }

    fn meta_refs(meta: &Meta) -> Vec<(&str, &str)> {
        meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
    }

    fn assert_matches(st: &SafeTensors, t: &[Owned], meta: &Meta) {
        assert_eq!(st.names().count(), t.len());
        for o in t {
            let info = st.info(&o.name).unwrap();
            assert_eq!(
                (info.dtype, &info.shape),
                (o.dtype, &o.shape),
                "{:?}",
                o.name
            );
            assert_eq!(st.read_bytes(&o.name).unwrap(), o.data, "{:?}", o.name);
            let numel = o.shape.iter().product::<u64>() as usize;
            let typed = match o.dtype {
                StDtype::F32 => st.read_f32(&o.name).map(|(_, v)| v.len()),
                StDtype::I64 => st.read_i64(&o.name).map(|(_, v)| v.len()),
                StDtype::U16 => st.read_u16(&o.name).map(|(_, v)| v.len()),
            };
            assert_eq!(typed.unwrap(), numel);
            let wrong = match o.dtype {
                StDtype::U16 => st.read_f32(&o.name).map(|_| ()),
                _ => st.read_u16(&o.name).map(|_| ()),
            };
            assert!(wrong.is_err());
        }
        let want: BTreeMap<String, String> = meta.iter().cloned().collect();
        assert_eq!(st.metadata(), &want);
    }

    #[test]
    fn random_shapes_round_trip_through_bytes_and_files() {
        let mut rng = Mix::new(0x0517);
        for i in 0..600 {
            let (t, meta) = random_tensors(&mut rng);
            let bytes = encode_safetensors(&items(&t), &meta_refs(&meta)).unwrap();
            assert_eq!(header_len(&bytes) % 8, 0);
            assert_matches(&SafeTensors::parse(&bytes).unwrap(), &t, &meta);
            if i % 20 == 0 {
                let path = tmp("prop");
                write_safetensors(&path.0, &items(&t), &meta_refs(&meta)).unwrap();
                assert_eq!(std::fs::read(&path.0).unwrap(), bytes);
                assert_matches(&SafeTensors::open(&path.0).unwrap(), &t, &meta);
            }
        }
    }

    fn seed_files() -> Vec<Vec<u8>> {
        let mut rng = Mix::new(3);
        let mut tensors = |shape: &[u64], dtype: StDtype, name: &str| {
            let n = shape.iter().product::<u64>() * dtype.size();
            Owned {
                name: name.to_string(),
                dtype,
                shape: shape.to_vec(),
                data: (0..n).map(|_| rng.next() as u8).collect(),
            }
        };
        let rich = vec![
            tensors(&[2, 3], StDtype::F32, "w"),
            tensors(&[0, 5], StDtype::I64, "ids"),
            tensors(&[3], StDtype::U16, "tok"),
            tensors(&[], StDtype::F32, "s"),
        ];
        let meta = vec![
            ("format".to_string(), "pt".to_string()),
            ("k\"\\\n".to_string(), "v\u{1}é".to_string()),
        ];
        let zero = vec![tensors(&[0], StDtype::U16, "z")];
        let escaped =
            br#"{"\u00e9\ud83d\ude00" : {"dtype":"U16","shape":[1],"data_offsets":[0,2]}}"#;
        vec![
            encode_safetensors(&items(&rich), &meta_refs(&meta)).unwrap(),
            encode_safetensors(&[], &[]).unwrap(),
            encode_safetensors(&items(&zero), &[]).unwrap(),
            frame(escaped, &[7, 0]),
        ]
    }

    fn header_len(v: &[u8]) -> u64 {
        let mut len8 = [0u8; 8];
        len8.copy_from_slice(&v[..8]);
        u64::from_le_bytes(len8)
    }

    /// One random corruption of `base`, using `other` as splice material.
    fn mutate(rng: &mut Mix, base: &[u8], other: &[u8]) -> Vec<u8> {
        const BYTES: &[u8] = b"{}[]\":,\\u0123456789DdEe -.\x00\x7f\xc3\xff";
        const NUMBERS: [&str; 6] = [
            "18446744073709551615",
            "18446744073709551616",
            "99999999999999999999999",
            "0",
            "00",
            "4294967296",
        ];
        let mut v = base.to_vec();
        let op = if v.len() < 8 { 4 } else { rng.below(7) };
        let before = v.len();
        match op {
            0 => {
                for _ in 0..1 + rng.below(4) {
                    let i = rng.below(v.len());
                    v[i] ^= 1 << rng.below(8);
                }
            }
            1 => {
                let i = rng.below(v.len());
                v[i] = rng.next() as u8;
            }
            2 => v.truncate(rng.below(v.len() + 1)),
            3 => {
                let n = header_len(&v);
                let picks = [
                    0,
                    1,
                    n.wrapping_sub(1),
                    n.wrapping_add(1),
                    n.wrapping_add(8),
                    MAX_HEADER_BYTES,
                    MAX_HEADER_BYTES + 1,
                    1 << 32,
                    u64::MAX,
                    rng.next(),
                ];
                v[..8].copy_from_slice(&picks[rng.below(picks.len())].to_le_bytes());
            }
            4 => {
                let a = rng.below(other.len() + 1);
                let b = a + rng.below(other.len() - a + 1);
                let at = rng.below(v.len() + 1);
                let end = at + rng.below(v.len() - at + 1);
                v.splice(at..end, other[a..b].iter().copied());
            }
            5 => {
                let at = 8 + rng.below(v.len() - 7);
                for _ in 0..1 + rng.below(3) {
                    v.insert(at, BYTES[rng.below(BYTES.len())]);
                }
            }
            _ => {
                let stop = usize::try_from(header_len(&v))
                    .ok()
                    .and_then(|n| n.checked_add(8))
                    .map_or(v.len(), |e| e.min(v.len()));
                let digits: Vec<usize> = (8..stop).filter(|&i| v[i].is_ascii_digit()).collect();
                if !digits.is_empty() {
                    let s = digits[rng.below(digits.len())];
                    let mut e = s;
                    while e < v.len() && v[e].is_ascii_digit() {
                        e += 1;
                    }
                    let num = NUMBERS[rng.below(NUMBERS.len())].bytes();
                    v.splice(s..e, num);
                }
            }
        }
        if op >= 5 && rng.below(2) == 0 {
            let n = header_len(&v)
                .wrapping_add(v.len() as u64)
                .wrapping_sub(before as u64);
            v[..8].copy_from_slice(&n.to_le_bytes());
        }
        v
    }

    /// Every tensor of an accepted file reads at its stated size, and the
    /// file re-encodes to one with equal contents.
    fn check_accepted(st: &SafeTensors, input_len: usize) {
        let mut t = Vec::new();
        for name in st.names() {
            let info = st.info(name).unwrap();
            let data = st.read_bytes(name).unwrap();
            assert!(data.len() <= input_len);
            assert_eq!(data.len() as u64, info.end - info.begin);
            t.push(Owned {
                name: name.to_string(),
                dtype: info.dtype,
                shape: info.shape.clone(),
                data,
            });
        }
        let meta: Meta = st
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let again = encode_safetensors(&items(&t), &meta_refs(&meta)).unwrap();
        assert_matches(&SafeTensors::parse(&again).unwrap(), &t, &meta);
    }

    fn parse_checked(bytes: &[u8]) -> bool {
        std::panic::catch_unwind(|| match SafeTensors::parse(bytes) {
            Ok(st) => {
                check_accepted(&st, bytes.len());
                true
            }
            Err(_) => false,
        })
        .unwrap_or_else(|_| panic!("panicked on {bytes:?}"))
    }

    #[test]
    fn mutated_files_never_panic_and_accepted_ones_are_consistent() {
        let seeds = seed_files();
        let mut truncations = 0;
        for s in &seeds {
            assert!(parse_checked(s));
            for n in 0..s.len() {
                parse_checked(&s[..n]);
                truncations += 1;
            }
        }
        let mut rng = Mix::new(0x5AFE);
        let (mut accepted, mut via_file) = (0, 0);
        for i in 0..8000 {
            let base = &seeds[rng.below(seeds.len())];
            let other = &seeds[rng.below(seeds.len())];
            let mut v = mutate(&mut rng, base, other);
            for _ in 0..rng.below(3) {
                v = mutate(&mut rng, &v, other);
            }
            let ok = parse_checked(&v);
            accepted += usize::from(ok);
            if i % 80 == 0 {
                let path = tmp("fuzz");
                std::fs::write(&path.0, &v).unwrap();
                let opened = std::panic::catch_unwind(|| SafeTensors::open(&path.0))
                    .unwrap_or_else(|_| panic!("open panicked on {v:?}"));
                assert_eq!(opened.is_ok(), ok, "open and parse disagree on {v:?}");
                if let Ok(st) = opened {
                    let mem = SafeTensors::parse(&v).unwrap();
                    for name in mem.names() {
                        assert_eq!(st.read_bytes(name).unwrap(), mem.read_bytes(name).unwrap());
                    }
                }
                via_file += 1;
            }
        }
        assert_eq!(truncations, seeds.iter().map(Vec::len).sum::<usize>());
        assert_eq!(via_file, 100);
        assert!(
            accepted > 50 && accepted < 8000,
            "accepted {accepted} of 8000"
        );
    }

    fn assert_parse_err(bytes: &[u8], needle: &str) -> IoError {
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            SafeTensors::parse(bytes)
        }));
        let err = caught
            .unwrap_or_else(|_| panic!("parse panicked on {bytes:?}"))
            .expect_err("expected Err");
        assert!(
            err.detail().contains(needle),
            "expected {needle:?} in {err}"
        );
        err
    }

    #[test]
    fn header_length_that_wraps_when_added_to_eight_is_rejected() {
        // 8 + (u64::MAX - 7) wraps to 0 in wrapping arithmetic. That must not
        // become a zero-length header that parses as the file itself.
        for n in [u64::MAX, u64::MAX - 7, u64::MAX - 8, MAX_HEADER_BYTES + 1] {
            let err = assert_parse_err(&n.to_le_bytes(), "outside 1..=");
            assert!(!err.detail().contains("Ok"), "{n}");
        }
        let at_cap = MAX_HEADER_BYTES.to_le_bytes();
        let err = SafeTensors::parse(&at_cap).unwrap_err();
        assert!(
            !err.detail().contains("outside 1..="),
            "exactly {MAX_HEADER_BYTES} is inside the cap; got {err}"
        );
        assert!(
            err.detail().contains("past the end"),
            "short file claiming the maximum header: {err}"
        );
    }

    #[test]
    fn escaped_duplicate_key_overlap_and_byte_size_overflow() {
        let dup = br#"{"a":{"dtype":"U16","shape":[1],"data_offsets":[0,2]},"\u0061":{"dtype":"U16","shape":[1],"data_offsets":[0,2]}}"#;
        assert_parse_err(&frame(dup, &[1, 0]), "duplicate key");

        let contained = br#"{"outer":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"inner":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        assert_parse_err(&frame(contained, &[0u8; 8]), "overlap");

        let backwards = br#"{"w":{"dtype":"U16","shape":[1],"data_offsets":[5,3]}}"#;
        let err = SafeTensors::parse(&frame(backwards, &[0u8; 8])).unwrap_err();
        assert!(
            err.detail().contains("outside") || err.detail().contains("overlap"),
            "{err}"
        );

        // 2^62 elements times 8-byte I64 overflows the byte length. A wrapping
        // multiply would be 0 and would match data_offsets [0, 0].
        let wrapped = br#"{"a":{"dtype":"I64","shape":[4611686018427387904],"data_offsets":[0,0]}}"#;
        assert_parse_err(&frame(wrapped, &[]), "overflow");
    }

    #[test]
    fn deep_json_non_utf8_header_and_truncated_payload() {
        let mut deep = br#"{"w":"#.to_vec();
        deep.extend(std::iter::repeat_n(b'[', 65));
        deep.extend(std::iter::repeat_n(b']', 65));
        deep.push(b'}');
        let err = assert_parse_err(&frame(&deep, &[]), "64");
        assert!(err.detail().contains("64") || err.detail().contains("nesting"));

        let mut bad = br#"{"w":{"dtype":"U16","shape":[1],"data_offsets":[0,2]}}"#.to_vec();
        bad[3] = 0xff;
        assert_parse_err(&frame(&bad, &[1, 0]), "UTF-8");

        let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        assert_parse_err(&frame(header, &[1, 2]), "outside");

        let path = tmp("trunc-open");
        std::fs::write(&path.0, [1, 2, 3]).unwrap();
        let opened = std::panic::catch_unwind(|| SafeTensors::open(&path.0))
            .expect("open panicked on a 3-byte file");
        let err = opened.expect_err("3-byte file opened");
        assert!(
            err.detail().contains("truncated") || err.detail().contains("header length"),
            "{err}"
        );
    }

    #[test]
    fn read_into_skips_bytes_before_a_nonzero_data_offset() {
        // 'a' occupies the prefix. 'b' starts at byte 4. A reader that ignores
        // begin, or that returns the file header, yields the prefix bytes.
        let header = br#"{"a":{"dtype":"U16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"U16","shape":[2],"data_offsets":[4,8]}}"#;
        let data: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0xAA, 0xBB, 0xCC, 0xDD];
        let bytes = frame(header, &data);
        let st = SafeTensors::parse(&bytes).unwrap();
        let mut mid = [0u8; 2];
        st.read_into("b", 1, &mut mid).unwrap();
        assert_eq!(mid, [0xBB, 0xCC]);
        assert_ne!(mid, [0x22, 0x33]);
        assert_ne!(&bytes[..2], &mid);

        let path = tmp("offset");
        std::fs::write(&path.0, &bytes).unwrap();
        let opened = SafeTensors::open(&path.0).unwrap();
        let mut whole = [0u8; 4];
        opened.read_into("b", 0, &mut whole).unwrap();
        assert_eq!(whole, [0xAA, 0xBB, 0xCC, 0xDD]);
        // Poison the on-disk prefix. 'b' is a positioned read and stays put.
        let mut poisoned = bytes.clone();
        let data_at = 8 + header.len();
        poisoned[data_at..data_at + 4].fill(0xA5);
        std::fs::write(&path.0, &poisoned).unwrap();
        let mut again = [0u8; 4];
        opened.read_into("b", 0, &mut again).unwrap();
        assert_eq!(again, [0xAA, 0xBB, 0xCC, 0xDD], "read returned the prefix");
    }

    #[test]
    fn empty_tensor_with_a_valid_dtype_round_trips() {
        let header = br#"{"e":{"dtype":"F32","shape":[0,7],"data_offsets":[0,0]},"w":{"dtype":"U16","shape":[1],"data_offsets":[0,2]}}"#;
        let bytes = frame(header, &[0xBE, 0xEF]);
        let st = SafeTensors::parse(&bytes).unwrap();
        let info = st.info("e").unwrap();
        assert_eq!(info.dtype, StDtype::F32);
        assert_eq!(info.shape, [0, 7]);
        assert_eq!(info.begin, 0);
        assert_eq!(info.end, 0);
        assert!(st.read_bytes("e").unwrap().is_empty());
        assert_eq!(st.read_f32("e").unwrap(), (vec![0, 7], Vec::<f32>::new()));
        assert!(st.read_into("e", 0, &mut []).is_ok());
        assert!(st.read_into("e", 0, &mut [0]).is_err());
        assert_eq!(st.read_u16("w").unwrap().1, vec![0xEFBE]);

        let scalar_empty = encode_safetensors(
            &[TensorOut {
                name: "z",
                dtype: StDtype::I64,
                shape: &[0],
                data: &[],
            }],
            &[],
        )
        .unwrap();
        let st = SafeTensors::parse(&scalar_empty).unwrap();
        assert_eq!(st.read_i64("z").unwrap(), (vec![0], Vec::<i64>::new()));
    }
}
