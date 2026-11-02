//! A minimal `.npy` reader for the committed goldens: little-endian `<f4`,
//! `<f8` and `<i8`, C order, format versions 1.0 and 2.0. Anything else is
//! refused by name rather than best-effort parsed, so a golden written in an
//! unexpected layout fails the test that loads it instead of feeding it
//! plausible numbers.
//!
//! This is the workspace's one `.npy` reader. It was ojas-cuda's `npy`
//! (written by L-cuda-oracle as `tests/reference/npy.rs`, moved into the
//! library by L-cuda-M1) and moved here unchanged, errors included, so
//! every crate reading goldens shares it; ojas-cuda's tests re-export it.

use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data_f32: Option<Vec<f32>>,
    pub data_i64: Option<Vec<i64>>,
    pub data_f64: Option<Vec<f64>>,
}

impl NpyArray {
    pub fn f32_slice(&self) -> Result<&[f32], String> {
        self.data_f32
            .as_deref()
            .ok_or_else(|| "expected float32 npy".to_string())
    }

    pub fn f64_slice(&self) -> Result<&[f64], String> {
        self.data_f64
            .as_deref()
            .ok_or_else(|| "expected float64 npy".to_string())
    }

    pub fn i64_slice(&self) -> Result<&[i64], String> {
        self.data_i64
            .as_deref()
            .ok_or_else(|| "expected int64 npy".to_string())
    }

    pub fn scalar_f32(&self) -> Result<f32, String> {
        let s = self.f32_slice()?;
        if s.len() != 1 {
            return Err(format!("expected scalar, got {} elems", s.len()));
        }
        Ok(s[0])
    }
}

pub const MAX_NPY_HEADER_BYTES: usize = 1 << 20;

pub fn read_npy(path: &Path) -> Result<NpyArray, String> {
    read_npy_with_limit(path, None)
}

pub fn read_npy_with_limit(path: &Path, limit: Option<u64>) -> Result<NpyArray, String> {
    use std::io::{BufReader, Read};

    let what = path.display();
    let (f, file_len) = crate::open::open_regular_file(path).map_err(|e| format!("{what}: {e}"))?;
    let mut f = BufReader::new(f);

    let mut magic_ver = [0u8; 8];
    f.read_exact(&mut magic_ver)
        .map_err(|e| format!("{what}: read header: {e}"))?;
    const MAGIC: &[u8] = b"\x93NUMPY";
    if &magic_ver[..6] != MAGIC {
        return Err(format!("{what}: not an .npy file (bad magic)"));
    }
    let ver_major = magic_ver[6];
    let ver_minor = magic_ver[7];
    let (header_len, prefix) = match ver_major {
        1 => {
            let mut hl = [0u8; 2];
            f.read_exact(&mut hl)
                .map_err(|e| format!("{what}: read header length: {e}"))?;
            (usize::from(u16::from_le_bytes(hl)), 10u64)
        }
        2 | 3 => {
            let mut hl = [0u8; 4];
            f.read_exact(&mut hl)
                .map_err(|e| format!("{what}: read header length: {e}"))?;
            (u32::from_le_bytes(hl) as usize, 12u64)
        }
        v => {
            return Err(format!(
                "{what}: unsupported npy format version {v}.{ver_minor}"
            ))
        }
    };
    if header_len > MAX_NPY_HEADER_BYTES || header_len as u64 > file_len.saturating_sub(prefix) {
        return Err(format!(
            "{what}: header length {header_len} exceeds the cap {MAX_NPY_HEADER_BYTES} or the file"
        ));
    }
    let mut header_bytes = vec![0u8; header_len];
    f.read_exact(&mut header_bytes)
        .map_err(|e| format!("{what}: read header: {e}"))?;
    let header = std::str::from_utf8(&header_bytes)
        .map_err(|e| format!("{what}: header is not UTF-8: {e}"))?;

    let total = prefix + header_len as u64;
    if !total.is_multiple_of(16) {
        return Err(format!(
            "{what}: header is not padded to a multiple of 16 bytes ({total} bytes before the data)"
        ));
    }
    let Some(body) = header.strip_suffix('\n') else {
        return Err(format!("{what}: header does not end with a newline"));
    };
    if !body.trim_end_matches(' ').ends_with('}') {
        return Err(format!(
            "{what}: header dict is not closed by '}}' followed only by space padding and a newline"
        ));
    }
    if ver_major != 3 && !header.is_ascii() {
        return Err(format!(
            "{what}: a version {ver_major} header must be ASCII"
        ));
    }

    let descr = dict_value(header, "descr").map_err(|e| format!("{what}: {e}"))?;
    let fortran = dict_value(header, "fortran_order").map_err(|e| format!("{what}: {e}"))?;
    let shape_src = dict_value(header, "shape").map_err(|e| format!("{what}: {e}"))?;
    if fortran == "False" {
        // C order accepted
    } else if fortran == "True" {
        return Err(format!("{what}: fortran-order npy not supported"));
    } else {
        return Err(format!(
            "{what}: fortran_order is neither True nor False: {fortran}"
        ));
    }
    let shape = parse_shape(shape_src).map_err(|e| format!("{what}: {e}"))?;
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| format!("{what}: shape overflows usize"))?;
    let item_size = match descr {
        "'<f4'" | "'|f4'" => 4usize,
        "'<f8'" | "'|f8'" | "'<i8'" | "'|i8'" => 8usize,
        other => return Err(format!("{what}: unsupported dtype {other}")),
    };
    let want = (numel as u64)
        .checked_mul(item_size as u64)
        .ok_or_else(|| format!("{what}: payload size overflows"))?;
    let rest = file_len - prefix - header_len as u64;
    if want != rest {
        return Err(format!(
            "{what}: shape {shape:?} needs {want} bytes, but {rest} follow the header"
        ));
    }
    if let Some(lim) = limit {
        if want > lim {
            return Err(format!(
                "{what}: payload budget exceeded: the file declares {want} tensor bytes, above the caller's limit of {lim}"
            ));
        }
    }
    match descr {
        "'<f4'" | "'|f4'" => {
            let mut data = vec![0.0f32; numel];
            let mut chunk = [0u8; 4];
            for x in &mut data {
                f.read_exact(&mut chunk)
                    .map_err(|e| format!("{what}: read payload: {e}"))?;
                *x = f32::from_le_bytes(chunk);
            }
            Ok(NpyArray {
                shape,
                data_f32: Some(data),
                data_i64: None,
                data_f64: None,
            })
        }
        "'<f8'" | "'|f8'" => {
            let mut data = vec![0.0f64; numel];
            let mut chunk = [0u8; 8];
            for x in &mut data {
                f.read_exact(&mut chunk)
                    .map_err(|e| format!("{what}: read payload: {e}"))?;
                *x = f64::from_le_bytes(chunk);
            }
            Ok(NpyArray {
                shape,
                data_f32: None,
                data_i64: None,
                data_f64: Some(data),
            })
        }
        "'<i8'" | "'|i8'" => {
            let mut data = vec![0i64; numel];
            let mut chunk = [0u8; 8];
            for x in &mut data {
                f.read_exact(&mut chunk)
                    .map_err(|e| format!("{what}: read payload: {e}"))?;
                *x = i64::from_le_bytes(chunk);
            }
            Ok(NpyArray {
                shape,
                data_f32: None,
                data_i64: Some(data),
                data_f64: None,
            })
        }
        _ => unreachable!(),
    }
}

/// Transpose last two dims of a row-major array. `shape` is updated in place.
pub fn transpose_last2(data: &mut [f32], shape: &mut [usize]) -> Result<(), String> {
    if shape.len() < 2 {
        return Err("transpose_last2 needs rank >= 2".into());
    }
    let r = shape.len();
    let rows = shape[r - 2];
    let cols = shape[r - 1];
    let numel: usize = shape.iter().product();
    if numel != data.len() {
        return Err(format!(
            "transpose_last2: shape {shape:?} holds {numel} elements, data has {}",
            data.len()
        ));
    }
    let batch: usize = shape[..r - 2].iter().product();
    let mut tmp = vec![0.0f32; data.len()];
    for b in 0..batch {
        let src = &data[b * rows * cols..(b + 1) * rows * cols];
        let dst = &mut tmp[b * rows * cols..(b + 1) * rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
    }
    data.copy_from_slice(&tmp);
    shape[r - 2] = cols;
    shape[r - 1] = rows;
    Ok(())
}

/// Write a C-order float32 `.npy` (v1.0).
pub fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<(), String> {
    let numel: usize = shape.iter().product();
    if data.len() != numel {
        return Err(format!(
            "write_npy shape {:?} expects {} elems, got {}",
            shape,
            numel,
            data.len()
        ));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    let shape_str = if shape.is_empty() {
        "()".to_string()
    } else if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}");
    let mut total = 10 + header.len() + 1;
    let pad = (64 - (total % 64)) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    total = 10 + header.len();
    debug_assert_eq!(total % 64, 0);

    let mut f = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    f.write_all(b"\x93NUMPY")
        .map_err(|e| format!("magic: {e}"))?;
    f.write_all(&[1u8, 0]).map_err(|e| format!("ver: {e}"))?;
    let hlen = u16::try_from(header.len())
        .map_err(|_| format!("write_npy: a {}-byte header needs npy v2", header.len()))?;
    f.write_all(&hlen.to_le_bytes())
        .map_err(|e| format!("hlen: {e}"))?;
    f.write_all(header.as_bytes())
        .map_err(|e| format!("header: {e}"))?;
    let mut buf = Vec::with_capacity(data.len().min(4096) * 4);
    for chunk in data.chunks(4096) {
        buf.clear();
        for &x in chunk {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        f.write_all(&buf).map_err(|e| format!("payload: {e}"))?;
    }
    Ok(())
}

#[allow(dead_code)]
pub fn seek_noop() {}

/// The payload, in the dtype the file declares. An `<f4` file is never widened
/// here and an `<f8` file is never narrowed: callers choose explicitly.
#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Data,
}

impl Npy {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Every element widened to f64. Exact for `<f4` and `<f8`; refused for
    /// `<i8`, whose values are indices, not reals.
    pub fn to_f64(&self) -> Result<Vec<f64>, String> {
        match &self.data {
            Data::F32(v) => Ok(v.iter().map(|&x| f64::from(x)).collect()),
            Data::F64(v) => Ok(v.clone()),
            Data::I64(_) => Err("an <i8 array is not a real-valued tensor".to_string()),
        }
    }

    pub fn f32s(&self) -> Result<&[f32], String> {
        match &self.data {
            Data::F32(v) => Ok(v),
            other => Err(format!("expected <f4, got {}", dtype_name(other))),
        }
    }

    pub fn f64s(&self) -> Result<&[f64], String> {
        match &self.data {
            Data::F64(v) => Ok(v),
            other => Err(format!("expected <f8, got {}", dtype_name(other))),
        }
    }

    pub fn i64s(&self) -> Result<&[i64], String> {
        match &self.data {
            Data::I64(v) => Ok(v),
            other => Err(format!("expected <i8, got {}", dtype_name(other))),
        }
    }
}

fn dtype_name(d: &Data) -> &'static str {
    match d {
        Data::F32(_) => "<f4",
        Data::F64(_) => "<f8",
        Data::I64(_) => "<i8",
    }
}

/// Reads and parses one file. The error names the path.
pub fn read(path: &Path) -> Result<Npy, String> {
    let arr = read_npy_with_limit(path, None)?;
    let data = if let Some(v) = arr.data_f32 {
        Data::F32(v)
    } else if let Some(v) = arr.data_f64 {
        Data::F64(v)
    } else if let Some(v) = arr.data_i64 {
        Data::I64(v)
    } else {
        unreachable!()
    };
    Ok(Npy {
        shape: arr.shape,
        data,
    })
}

/// Parses an in-memory `.npy` image.
pub fn parse(bytes: &[u8]) -> Result<Npy, String> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err("not an .npy file (bad magic)".to_string());
    }
    let (header_len, start) = match bytes[6] {
        1 => (
            usize::from(u16::from_le_bytes([bytes[8], bytes[9]])),
            10usize,
        ),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("truncated v2 header length".to_string());
            }
            let n = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
            (
                usize::try_from(n).map_err(|e| format!("header length: {e}"))?,
                12usize,
            )
        }
        v => {
            let minor = if bytes.len() > 7 { bytes[7] } else { 0 };
            return Err(format!("unsupported npy format version {v}.{minor}"));
        }
    };
    if header_len > MAX_NPY_HEADER_BYTES || header_len > bytes.len().saturating_sub(start) {
        return Err(format!(
            "header length {header_len} exceeds the cap {MAX_NPY_HEADER_BYTES} or the file"
        ));
    }
    let end = start
        .checked_add(header_len)
        .filter(|&e| e <= bytes.len())
        .ok_or("header runs past the end of the file")?;
    let header =
        std::str::from_utf8(&bytes[start..end]).map_err(|e| format!("header is not UTF-8: {e}"))?;
    let total = start as u64 + header_len as u64;
    if !total.is_multiple_of(16) {
        return Err(format!(
            "header is not padded to a multiple of 16 bytes ({total} bytes before the data)"
        ));
    }
    let Some(body) = header.strip_suffix('\n') else {
        return Err("header does not end with a newline".to_string());
    };
    if !body.trim_end_matches(' ').ends_with('}') {
        return Err(
            "header dict is not closed by '}' followed only by space padding and a newline"
                .to_string(),
        );
    }
    if bytes[6] != 3 && !header.is_ascii() {
        return Err(format!("a version {} header must be ASCII", bytes[6]));
    }
    let descr = dict_value(header, "descr")?;
    let fortran = dict_value(header, "fortran_order")?;
    let shape_src = dict_value(header, "shape")?;
    if fortran == "False" {
        // C order accepted
    } else if fortran == "True" {
        return Err("fortran-order npy not supported".into());
    } else {
        return Err(format!(
            "fortran_order is neither True nor False: {fortran}"
        ));
    }
    let shape = parse_shape(shape_src)?;
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or("shape overflows usize")?;
    let payload = &bytes[end..];
    let (item, data) = match descr {
        "'<f4'" => (4usize, Data::F32(Vec::new())),
        "'<f8'" => (8usize, Data::F64(Vec::new())),
        "'<i8'" => (8usize, Data::I64(Vec::new())),
        other => return Err(format!("unsupported dtype {other}")),
    };
    let want = numel
        .checked_mul(item)
        .ok_or("payload size overflows usize")?;
    if payload.len() != want {
        return Err(format!(
            "shape {shape:?} needs {want} bytes, but {} follow the header",
            payload.len()
        ));
    }
    // The payload length was checked against the shape above, so every
    // `as_chunks` remainder is empty.
    let data = match data {
        Data::F32(_) => Data::F32(
            payload
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
        ),
        Data::F64(_) => Data::F64(
            payload
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect(),
        ),
        Data::I64(_) => Data::I64(
            payload
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect(),
        ),
    };
    Ok(Npy { shape, data })
}

/// The raw text of `key`'s value in the header's Python dict literal.
fn dict_value<'a>(header: &'a str, key: &str) -> Result<&'a str, String> {
    let needle = format!("'{key}':");
    let at = header
        .find(&needle)
        .ok_or_else(|| format!("missing {key}"))?;
    let rest = header[at + needle.len()..].trim_start();
    let end = if rest.starts_with('(') {
        rest.find(')').map(|i| i + 1)
    } else if let Some(quoted) = rest.strip_prefix('\'') {
        quoted.find('\'').map(|i| i + 2)
    } else {
        rest.find([',', '}'])
    }
    .ok_or_else(|| format!("unterminated value for '{key}'"))?;
    Ok(rest[..end].trim())
}

fn parse_shape(src: &str) -> Result<Vec<usize>, String> {
    let inner = src
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| format!("shape {src} is not a tuple"))?;
    inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| format!("shape entry {s:?}: {e}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A version-1.0 image, as numpy writes it: the header padded with spaces
    /// and ended by a newline so the payload starts on a 64-byte boundary.
    fn image(descr: &str, fortran: &str, shape: &str, payload: &[u8]) -> Vec<u8> {
        let mut header =
            format!("{{'descr': '{descr}', 'fortran_order': {fortran}, 'shape': {shape}, }}");
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend(u16::try_from(header.len()).unwrap().to_le_bytes());
        out.extend(header.as_bytes());
        out.extend(payload);
        out
    }

    #[test]
    fn reads_f64_f32_and_i64_in_c_order() {
        let p: Vec<u8> = [1.5f64, -2.0, 0.25]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let a = parse(&image("<f8", "False", "(3,)", &p)).unwrap();
        assert_eq!(a.shape, vec![3]);
        assert_eq!(a.f64s().unwrap(), &[1.5, -2.0, 0.25]);
        let p: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let b = parse(&image("<f4", "False", "(2, 2)", &p)).unwrap();
        assert_eq!((b.shape.clone(), b.numel()), (vec![2, 2], 4));
        assert_eq!(b.to_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        let p: Vec<u8> = [7i64, -1].iter().flat_map(|x| x.to_le_bytes()).collect();
        let c = parse(&image("<i8", "False", "(2,)", &p)).unwrap();
        assert_eq!(c.i64s().unwrap(), &[7, -1]);
        assert!(c.to_f64().is_err(), "indices are not reals");
        let s = parse(&image("<f8", "False", "()", &1.0f64.to_le_bytes())).unwrap();
        assert_eq!((s.shape.len(), s.numel()), (0, 1));
    }

    #[test]
    fn refuses_what_it_does_not_read() {
        let p = 1.0f64.to_le_bytes();
        assert!(
            parse(&image("<f8", "True", "(1,)", &p)).is_err(),
            "fortran order"
        );
        assert!(
            parse(&image(">f8", "False", "(1,)", &p)).is_err(),
            "big-endian"
        );
        assert!(
            parse(&image("<f8", "False", "(2,)", &p)).is_err(),
            "short payload"
        );
        assert!(parse(b"\x93NUMPX\x01\x00\x00\x00").is_err(), "magic");
        assert!(
            parse(b"\x93NUMPY\x01\x00\xff\x00{").is_err(),
            "header past the end"
        );
    }
}
