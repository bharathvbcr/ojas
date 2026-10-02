//! Payload codec shared by every opcode.
//!
//! Fixed fields are little-endian and read in order by [`Reader`]. Every
//! length is checked against the bytes left in the payload before anything
//! is allocated, and a payload with bytes left over is refused.
//!
//! Option records ([`Fields`]) are `count: u32`, then `count` entries of
//! `tag: u32, len: u32, value: [u8; len]`. Each opcode names the tags it
//! accepts with their kind; an unknown tag, a repeated tag, a value of the
//! wrong length and a string that is not UTF-8 or holds NUL are refused.
//! `go/ffi.go` writes the same layout and the same tag numbers.

use std::collections::BTreeMap;

/// Most entries one option record may hold. Every record defined here has
/// fewer; the cap bounds the parse, not a feature.
pub const MAX_FIELDS: u32 = 64;

/// Longest string field, the same cap as an inline load path.
pub const MAX_STR: usize = crate::load::INLINE_PATH_MAX;

pub struct Reader<'a> {
    rest: &'a [u8],
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { rest: bytes }
    }

    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.rest.len() < n {
            return Err("shape: truncated payload".to_string());
        }
        let (head, tail) = self.rest.split_at(n);
        self.rest = tail;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let head = self.bytes(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(head);
        Ok(out)
    }

    pub fn u32(&mut self) -> Result<u32, String> {
        self.array().map(u32::from_le_bytes)
    }

    pub fn u64(&mut self) -> Result<u64, String> {
        self.array().map(u64::from_le_bytes)
    }

    /// `n` little-endian values of `N` bytes. The byte count is checked
    /// against the payload before the vector is allocated.
    pub fn values<T, const N: usize>(
        &mut self,
        n: usize,
        decode: fn([u8; N]) -> T,
    ) -> Result<Vec<T>, String> {
        let len = n
            .checked_mul(N)
            .ok_or_else(|| "shape: payload size overflows".to_string())?;
        let head = self.bytes(len)?;
        let (chunks, rest) = head.as_chunks::<N>();
        if !rest.is_empty() {
            return Err("shape: payload is not a whole number of values".to_string());
        }
        Ok(chunks.iter().copied().map(decode).collect())
    }

    /// Every byte left.
    pub fn rest(&mut self) -> &'a [u8] {
        std::mem::take(&mut self.rest)
    }

    /// Every byte left, as UTF-8 text (a path or a tokenizer input).
    pub fn rest_str(&mut self) -> Result<&'a str, String> {
        std::str::from_utf8(self.rest()).map_err(|_| "shape: text is not utf-8".to_string())
    }

    /// Refuse bytes left over.
    pub fn finish(&self) -> Result<(), String> {
        if self.rest.is_empty() {
            Ok(())
        } else {
            Err("shape: trailing bytes".to_string())
        }
    }
}

/// What a tag holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    U32,
    U64,
    F32,
    F64,
    /// UTF-8 without NUL, at most [`MAX_STR`] bytes.
    Str,
    /// Exactly 32 raw bytes.
    Bytes32,
    /// A whole number of little-endian `u32`s.
    U32s,
}

/// A parsed option record: each accepted tag at most once.
pub struct Fields<'a> {
    values: BTreeMap<u32, (&'static str, Kind, &'a [u8])>,
}

impl<'a> Fields<'a> {
    /// Read one option record from `r`. `allowed` lists `(tag, name, kind)`.
    pub fn read(r: &mut Reader<'a>, allowed: &[(u32, &'static str, Kind)]) -> Result<Self, String> {
        let count = r.u32()?;
        if count > MAX_FIELDS {
            return Err(format!(
                "shape: option record has {count} fields; at most {MAX_FIELDS}"
            ));
        }
        let mut values = BTreeMap::new();
        for _ in 0..count {
            let tag = r.u32()?;
            let len = usize::try_from(r.u32()?)
                .map_err(|_| "shape: option length overflows".to_string())?;
            let &(_, name, kind) = allowed
                .iter()
                .find(|(t, _, _)| *t == tag)
                .ok_or_else(|| format!("shape: unknown option field {tag}"))?;
            let value = r.bytes(len)?;
            let ok = match kind {
                Kind::U32 | Kind::F32 => len == 4,
                Kind::U64 | Kind::F64 => len == 8,
                Kind::Bytes32 => len == 32,
                Kind::U32s => len.is_multiple_of(4),
                Kind::Str => {
                    if len > MAX_STR {
                        return Err(format!("shape: {name} exceeds {MAX_STR} bytes"));
                    }
                    let text = std::str::from_utf8(value)
                        .map_err(|_| format!("shape: {name} is not utf-8"))?;
                    !text.contains('\0')
                }
            };
            if !ok {
                return Err(format!("shape: {name} has a malformed {len}-byte value"));
            }
            if values.insert(tag, (name, kind, value)).is_some() {
                return Err(format!("shape: option field {name} repeats"));
            }
        }
        Ok(Self { values })
    }

    fn raw(&self, tag: u32, want: Kind) -> Option<(&'static str, &'a [u8])> {
        self.values
            .get(&tag)
            .filter(|(_, kind, _)| *kind == want)
            .map(|&(name, _, bytes)| (name, bytes))
    }

    fn fixed<const N: usize>(&self, tag: u32, want: Kind) -> Option<[u8; N]> {
        self.raw(tag, want).map(|(_, b)| {
            let mut out = [0u8; N];
            out.copy_from_slice(b);
            out
        })
    }

    pub fn u32(&self, tag: u32) -> Option<u32> {
        self.fixed(tag, Kind::U32).map(u32::from_le_bytes)
    }

    pub fn u64(&self, tag: u32) -> Option<u64> {
        self.fixed(tag, Kind::U64).map(u64::from_le_bytes)
    }

    pub fn f32(&self, tag: u32) -> Option<f32> {
        self.fixed(tag, Kind::F32).map(f32::from_le_bytes)
    }

    pub fn f64(&self, tag: u32) -> Option<f64> {
        self.fixed(tag, Kind::F64).map(f64::from_le_bytes)
    }

    pub fn bytes32(&self, tag: u32) -> Option<[u8; 32]> {
        self.fixed(tag, Kind::Bytes32)
    }

    pub fn str(&self, tag: u32) -> Option<&'a str> {
        // Validated as UTF-8 by `read`.
        self.raw(tag, Kind::Str)
            .and_then(|(_, b)| std::str::from_utf8(b).ok())
    }

    pub fn u32s(&self, tag: u32) -> Option<Vec<u32>> {
        self.raw(tag, Kind::U32s).map(|(_, b)| {
            b.as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(u32::from_le_bytes)
                .collect()
        })
    }
}

/// `value`, or the error that the required field `tag` is missing.
pub fn required<T>(value: Option<T>, tag: u32) -> Result<T, String> {
    value.ok_or_else(|| {
        let (_, name, _) = field(tag, Kind::U32);
        format!("shape: option field {name} is required")
    })
}

/// Option tags. One numbering for every record; `go/ffi.go` repeats it.
pub mod tag {
    pub const PATH: u32 = 1;
    pub const DEVICE: u32 = 2;
    pub const THREADS: u32 = 3;
    pub const BUDGET: u32 = 4;
    pub const NUMERICS: u32 = 5;
    pub const SEED: u32 = 6;

    pub const VOCAB: u32 = 10;
    pub const N_EMBD: u32 = 11;
    pub const N_LAYER: u32 = 12;
    pub const N_HEAD: u32 = 13;
    pub const N_KV_HEAD: u32 = 14;
    pub const HEAD_DIM: u32 = 15;
    pub const HIDDEN: u32 = 16;
    pub const MAX_SEQ: u32 = 17;
    pub const ROPE_BASE: u32 = 18;
    pub const RMS_EPS: u32 = 19;

    pub const TOKEN_BIN: u32 = 20;
    pub const BIN_FORMAT: u32 = 21;
    pub const BATCH: u32 = 22;
    pub const SEQ: u32 = 23;
    pub const ACCUM: u32 = 24;
    pub const DATA_SEED: u32 = 25;
    pub const SCHEDULE: u32 = 26;
    pub const WARMUP: u32 = 27;
    pub const TOTAL: u32 = 28;
    pub const DECAY_FRAC: u32 = 29;
    pub const MATRIX_LR: u32 = 30;
    pub const ADAM_LR: u32 = 31;
    pub const GRAD_CLIP: u32 = 32;
    pub const ON_NONFINITE: u32 = 33;
    pub const TOKENIZER_HASH: u32 = 34;

    pub const VOCAB_JSON: u32 = 40;
    pub const MERGES_TXT: u32 = 41;

    pub const TEMPERATURE: u32 = 50;
    pub const TOP_K: u32 = 51;
    pub const TOP_P: u32 = 52;
    pub const MAX_NEW: u32 = 53;
    pub const STOP: u32 = 54;
    pub const PROMPT: u32 = 55;
}

const TAG_NAMES: &[(u32, &str)] = &[
    (tag::PATH, "path"),
    (tag::DEVICE, "device"),
    (tag::THREADS, "threads"),
    (tag::BUDGET, "budget"),
    (tag::NUMERICS, "numerics"),
    (tag::SEED, "seed"),
    (tag::VOCAB, "vocab"),
    (tag::N_EMBD, "n_embd"),
    (tag::N_LAYER, "n_layer"),
    (tag::N_HEAD, "n_head"),
    (tag::N_KV_HEAD, "n_kv_head"),
    (tag::HEAD_DIM, "head_dim"),
    (tag::HIDDEN, "hidden"),
    (tag::MAX_SEQ, "max_seq"),
    (tag::ROPE_BASE, "rope_base"),
    (tag::RMS_EPS, "rms_eps"),
    (tag::TOKEN_BIN, "token_bin"),
    (tag::BIN_FORMAT, "bin_format"),
    (tag::BATCH, "batch"),
    (tag::SEQ, "seq"),
    (tag::ACCUM, "accum"),
    (tag::DATA_SEED, "data_seed"),
    (tag::SCHEDULE, "schedule"),
    (tag::WARMUP, "warmup"),
    (tag::TOTAL, "total"),
    (tag::DECAY_FRAC, "decay_frac"),
    (tag::MATRIX_LR, "matrix_lr"),
    (tag::ADAM_LR, "adam_lr"),
    (tag::GRAD_CLIP, "grad_clip"),
    (tag::ON_NONFINITE, "on_nonfinite"),
    (tag::TOKENIZER_HASH, "tokenizer_hash"),
    (tag::VOCAB_JSON, "vocab_json"),
    (tag::MERGES_TXT, "merges_txt"),
    (tag::TEMPERATURE, "temperature"),
    (tag::TOP_K, "top_k"),
    (tag::TOP_P, "top_p"),
    (tag::MAX_NEW, "max_new_tokens"),
    (tag::STOP, "stop"),
    (tag::PROMPT, "prompt"),
];

/// The `(tag, name, kind)` of `tag`, for an opcode's allow-list.
pub fn field(tag: u32, kind: Kind) -> (u32, &'static str, Kind) {
    let name = TAG_NAMES
        .iter()
        .find(|(t, _)| *t == tag)
        .map(|(_, n)| *n)
        .unwrap_or("field");
    (tag, name, kind)
}

/// Builds option records; the Rust tests use it to write what Go writes.
#[cfg(test)]
#[derive(Default)]
pub struct Writer {
    count: u32,
    body: Vec<u8>,
}

#[cfg(test)]
impl Writer {
    pub fn raw(mut self, tag: u32, value: &[u8]) -> Self {
        self.count += 1;
        self.body.extend_from_slice(&tag.to_le_bytes());
        self.body
            .extend_from_slice(&(value.len() as u32).to_le_bytes());
        self.body.extend_from_slice(value);
        self
    }

    pub fn u32(self, tag: u32, v: u32) -> Self {
        self.raw(tag, &v.to_le_bytes())
    }

    pub fn u64(self, tag: u32, v: u64) -> Self {
        self.raw(tag, &v.to_le_bytes())
    }

    pub fn f32(self, tag: u32, v: f32) -> Self {
        self.raw(tag, &v.to_le_bytes())
    }

    pub fn f64(self, tag: u32, v: f64) -> Self {
        self.raw(tag, &v.to_le_bytes())
    }

    pub fn str(self, tag: u32, v: &str) -> Self {
        self.raw(tag, v.as_bytes())
    }

    pub fn u32s(self, tag: u32, v: &[u32]) -> Self {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.raw(tag, &bytes)
    }

    pub fn finish(self) -> Vec<u8> {
        let mut out = self.count.to_le_bytes().to_vec();
        out.extend_from_slice(&self.body);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALLOWED: &[(u32, &str, Kind)] = &[
        (tag::PATH, "path", Kind::Str),
        (tag::DEVICE, "device", Kind::U32),
        (tag::STOP, "stop", Kind::U32s),
    ];

    #[test]
    fn a_record_round_trips_and_refuses_unknown_repeated_and_malformed_fields() {
        let ok = Writer::default()
            .str(tag::PATH, "m.safetensors")
            .u32(tag::DEVICE, 3)
            .u32s(tag::STOP, &[7, 9])
            .finish();
        let mut r = Reader::new(&ok);
        let f = Fields::read(&mut r, ALLOWED).unwrap();
        r.finish().unwrap();
        assert_eq!(f.str(tag::PATH), Some("m.safetensors"));
        assert_eq!(f.u32(tag::DEVICE), Some(3));
        assert_eq!(f.u32s(tag::STOP), Some(vec![7, 9]));
        assert_eq!(f.u64(tag::DEVICE), None, "a field is read only as its kind");

        let cases: [(Vec<u8>, &str); 6] = [
            (
                Writer::default().u32(tag::SEED, 1).finish(),
                "unknown option field 6",
            ),
            (
                Writer::default()
                    .u32(tag::DEVICE, 1)
                    .u32(tag::DEVICE, 2)
                    .finish(),
                "repeats",
            ),
            (
                Writer::default().u64(tag::DEVICE, 1).finish(),
                "malformed 8-byte",
            ),
            (
                Writer::default().str(tag::PATH, "a\0b").finish(),
                "malformed",
            ),
            (
                Writer::default().raw(tag::PATH, &[0xff, 0xfe]).finish(),
                "not utf-8",
            ),
            (
                Writer::default().raw(tag::STOP, &[1, 2, 3]).finish(),
                "malformed",
            ),
        ];
        for (bytes, want) in cases {
            let err = Fields::read(&mut Reader::new(&bytes), ALLOWED)
                .err()
                .unwrap_or_else(|| panic!("accepted {bytes:?}"));
            assert!(err.contains(want), "{want}: {err}");
        }
    }

    #[test]
    fn lengths_are_checked_before_anything_is_allocated() {
        // A count at the cap, and a length far past the payload.
        let mut huge = (MAX_FIELDS + 1).to_le_bytes().to_vec();
        let err = Fields::read(&mut Reader::new(&huge), ALLOWED)
            .err()
            .unwrap();
        assert!(err.contains("at most"), "{err}");
        huge = 1u32.to_le_bytes().to_vec();
        huge.extend_from_slice(&tag::STOP.to_le_bytes());
        huge.extend_from_slice(&u32::MAX.to_le_bytes());
        let err = Fields::read(&mut Reader::new(&huge), ALLOWED)
            .err()
            .unwrap();
        assert!(err.contains("truncated"), "{err}");
        let long = "a".repeat(MAX_STR + 1);
        let bytes = Writer::default().str(tag::PATH, &long).finish();
        let err = Fields::read(&mut Reader::new(&bytes), ALLOWED)
            .err()
            .unwrap();
        assert!(err.contains("exceeds"), "{err}");
        let err = Reader::new(&[0u8; 7])
            .values(usize::MAX, u32::from_le_bytes)
            .unwrap_err();
        assert!(err.contains("overflows"), "{err}");
    }
}
