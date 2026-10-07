//! Oracle fixtures: golden values ojas is checked against. CI does not run
//! PyTorch; the torch side lives in `python/` and only generates files.
//!
//! - [`rms_norm_fixture`]: one hand-written fp64 RMSNorm row
//!   (`fixtures/rms_norm_f64.json`).
//! - [`golden`]: the tiny nanolab GPT exported from torch (init, forward,
//!   gradients, 5/40-step training traces, LR schedules) and the ojas
//!   `BatchSampler` start dump the torch side replays
//!   (framework-design.md §9 item 13).
//! - [`parity`]: the tolerances later lanes are held to, and runners that
//!   apply them to any model implementing [`parity::ParityModel`].
//! - [`spec`]: the `ojas.spec` JSON schema and the §2 parameter table.
//! - [`safetensors`]: the fixture policy over `ojas_io::SafeTensors`.
//! - [`gdn`]: the gated delta rule's f64 forward and transformers' published
//!   goldens for it (`Backend::chunked_gdn_forward` / `_backward`).
//!
//! JSON is read with `ojas-io`'s strict RFC 8259 reader ([`ojas_io::json`]),
//! held to the fixture limits (4 MiB, depth 16, a tree budget per input
//! byte). On top of it the fixture policy refuses `null` anywhere and string
//! escapes in fixture files (safetensors headers, whose spec is an escaped
//! string, are read by `ojas_io::SafeTensors`). Numbers read as `f64` only
//! when exact; a count or shape entry must be an exact integer literal.

#![forbid(unsafe_code)]

pub mod gdn;
pub mod golden;
pub mod parity;
pub mod safetensors;
pub mod spec;

use ojas_core::OjasError;
use ojas_io::{IoError, JsonLimits, JsonValue};

const FIXTURE: &str = include_str!("../fixtures/rms_norm_f64.json");

/// Hand-written RMSNorm fp64 row.
#[derive(Clone, Debug, PartialEq)]
pub struct RmsNormFixture {
    pub eps: f64,
    pub input: Vec<f64>,
    pub input_shape: Vec<usize>,
    pub weight: Vec<f64>,
    pub weight_shape: Vec<usize>,
    pub expected: Vec<f64>,
    pub expected_shape: Vec<usize>,
}

/// Load the checked-in RMSNorm fixture.
pub fn rms_norm_fixture() -> Result<RmsNormFixture, OjasError> {
    parse_rms_norm(FIXTURE)
}

/// Parse one RMSNorm fixture document.
pub fn parse_rms_norm(text: &str) -> Result<RmsNormFixture, OjasError> {
    let value = parse_json(text)?;
    let obj = value.object()?;
    let format = field(obj, "format").and_then(Json::text).unwrap_or("");
    if format != "ojas-oracle-fixture-v1" {
        return Err(bad("fixture format is not ojas-oracle-fixture-v1"));
    }
    if field(obj, "op").and_then(Json::text) != Some("rms_norm") {
        return Err(bad("fixture op is not rms_norm"));
    }
    let eps = field(obj, "eps")
        .ok_or_else(|| bad("missing eps"))?
        .number()?;
    if !eps.is_finite() {
        return Err(OjasError::NonFinite {
            op: "oracle_fixture",
        });
    }
    let fixture = RmsNormFixture {
        eps,
        input: field(obj, "input")
            .ok_or_else(|| bad("missing input"))?
            .numbers()?,
        input_shape: field(obj, "input_shape")
            .ok_or_else(|| bad("missing input_shape"))?
            .shape()?,
        weight: field(obj, "weight")
            .ok_or_else(|| bad("missing weight"))?
            .numbers()?,
        weight_shape: field(obj, "weight_shape")
            .ok_or_else(|| bad("missing weight_shape"))?
            .shape()?,
        expected: field(obj, "expected")
            .ok_or_else(|| bad("missing expected"))?
            .numbers()?,
        expected_shape: field(obj, "expected_shape")
            .ok_or_else(|| bad("missing expected_shape"))?
            .shape()?,
    };
    if product(&fixture.input_shape)? != fixture.input.len()
        || product(&fixture.weight_shape)? != fixture.weight.len()
        || product(&fixture.expected_shape)? != fixture.expected.len()
    {
        return Err(bad("fixture length does not match its shape"));
    }
    Ok(fixture)
}

pub(crate) fn product(shape: &[usize]) -> Result<usize, OjasError> {
    let mut n = 1usize;
    for &dim in shape {
        n = n
            .checked_mul(dim)
            .ok_or_else(|| bad("shape product overflows"))?;
    }
    Ok(n)
}

pub(crate) fn bad(detail: &str) -> OjasError {
    OjasError::OutOfRange {
        op: "oracle_fixture",
        detail: detail.to_string(),
    }
}

/// A parsed fixture document: `ojas-io`'s strict reader's tree.
pub(crate) type Json = JsonValue;

/// The oracle's reading of a [`Json`] value: any number reads as `f64`
/// (exactly, or it is refused), and a shape entry is an exact non-negative
/// integer literal, so `256.0` is refused as the model's reader refuses it.
pub(crate) trait JsonExt {
    fn object(&self) -> Result<&[(String, Json)], OjasError>;
    fn items(&self) -> Result<&[Json], OjasError>;
    fn number(&self) -> Result<f64, OjasError>;
    fn numbers(&self) -> Result<Vec<f64>, OjasError>;
    fn shape(&self) -> Result<Vec<usize>, OjasError>;
    fn text(&self) -> Option<&str>;
    fn flag(&self) -> Option<bool>;
}

pub(crate) fn io(e: IoError) -> OjasError {
    bad(e.detail())
}

impl JsonExt for Json {
    fn object(&self) -> Result<&[(String, Json)], OjasError> {
        self.as_object().map_err(io)
    }

    fn items(&self) -> Result<&[Json], OjasError> {
        self.as_array().map_err(io)
    }

    fn number(&self) -> Result<f64, OjasError> {
        self.as_f64().map_err(io)
    }

    fn numbers(&self) -> Result<Vec<f64>, OjasError> {
        self.items()?.iter().map(Json::number).collect()
    }

    fn shape(&self) -> Result<Vec<usize>, OjasError> {
        self.items()?
            .iter()
            .map(|v| {
                let n = v.as_u64().map_err(io)?;
                usize::try_from(n).map_err(|_| bad(&format!("shape entry {n} exceeds usize")))
            })
            .collect()
    }

    fn text(&self) -> Option<&str> {
        self.as_str().ok()
    }

    fn flag(&self) -> Option<bool> {
        self.as_bool().ok()
    }
}

pub(crate) fn field<'a>(obj: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    obj.iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}

/// The fixture format nests two levels; deeper input is refused before the
/// recursive descent can exhaust the stack.
const MAX_DEPTH: usize = 16;

/// Public `parse_rms_norm` takes any string. Four mebibytes is enough for a
/// fixture and smaller than the two-mebibyte depth probe in the tests.
const MAX_FIXTURE_BYTES: usize = 4 * 1024 * 1024;
const FIXTURE_BYTE_FACTOR: usize = 8;
const FIXTURE_BYTE_FLOOR: usize = 64 * 1024;

/// How a fixture document is parsed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JsonOptions {
    /// Tree-size budget per input byte. Dense arrays of one- or two-digit
    /// numbers (token rows) cost about 32 tree bytes per 3 input bytes.
    pub byte_factor: usize,
}

/// The RMSNorm fixture (and anything else hand-written).
pub(crate) const FIXTURE_JSON: JsonOptions = JsonOptions {
    byte_factor: FIXTURE_BYTE_FACTOR,
};
/// Generated golden JSON: token rows need a larger tree budget.
pub(crate) const GOLDEN_JSON: JsonOptions = JsonOptions { byte_factor: 16 };

fn parse_json(text: &str) -> Result<Json, OjasError> {
    parse_json_with(text, FIXTURE_JSON)
}

/// Parse with `ojas-io`'s strict reader, then apply the fixture policy: no
/// `null` anywhere, and no string escapes. A backslash can only appear inside
/// a string in valid JSON, so the escape check is a scan of the input.
/// (safetensors headers, whose metadata values are escaped JSON strings, are
/// read by `ojas_io::SafeTensors` instead.)
pub(crate) fn parse_json_with(text: &str, opts: JsonOptions) -> Result<Json, OjasError> {
    if text.len() > MAX_FIXTURE_BYTES {
        return Err(bad("fixture exceeds 4 MiB"));
    }
    let limits = JsonLimits {
        max_input_bytes: MAX_FIXTURE_BYTES,
        max_depth: MAX_DEPTH,
        tree_bytes_per_input_byte: opts.byte_factor,
        tree_bytes_floor: FIXTURE_BYTE_FLOOR,
        ..JsonLimits::DEFAULT
    };
    let value = ojas_io::parse_json_with(text, &limits).map_err(io)?;
    if text.contains('\\') {
        return Err(bad("fixture strings do not use escapes"));
    }
    refuse_null(&value)?;
    Ok(value)
}

fn refuse_null(v: &Json) -> Result<(), OjasError> {
    match v {
        Json::Null => Err(bad("fixture JSON has no null")),
        Json::Array(items) => items.iter().try_for_each(refuse_null),
        Json::Object(pairs) => pairs.iter().try_for_each(|(_, v)| refuse_null(v)),
        _ => Ok(()),
    }
}
