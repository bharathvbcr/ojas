//! A minimal `.npy` reader for the committed goldens: little-endian `<f4`,
//! `<f8` and `<i8`, C order, format versions 1.0 and 2.0. Anything else is
//! refused by name rather than best-effort parsed, so a golden written in an
//! unexpected layout fails the test that loads it instead of feeding it
//! plausible numbers.
//!
//! Written by L-cuda-oracle as `tests/reference/npy.rs`. It moved here
//! unchanged (L-cuda-M1, lead's ruling 2026-10-01) because `runga` and the
//! tiny-fixture loader read `.npy` outside the test harness;
//! `tests/reference/npy.rs` now re-exports this module.

use std::path::Path;

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
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
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
        v => return Err(format!("unsupported .npy major version {v}")),
    };
    let end = start
        .checked_add(header_len)
        .filter(|&e| e <= bytes.len())
        .ok_or("header runs past the end of the file")?;
    let header =
        std::str::from_utf8(&bytes[start..end]).map_err(|e| format!("header is not utf-8: {e}"))?;
    let descr = dict_value(header, "descr")?;
    let fortran = dict_value(header, "fortran_order")?;
    let shape_src = dict_value(header, "shape")?;
    if fortran != "False" {
        return Err(format!(
            "fortran_order {fortran} refused; goldens are C order"
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
            "payload is {} bytes, the header's shape {shape:?} needs {want}",
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
        .ok_or_else(|| format!("header has no '{key}'"))?;
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
