//! fp64 oracle fixtures.
//!
//! The file format is a small JSON subset (`ojas-oracle-fixture-v1`): one
//! object of strings, numbers, and arrays of numbers. See
//! `fixtures/rms_norm_f64.json`. CI does not run PyTorch. Further torch fp64
//! fixtures are generated offline later; this crate ships one hand-written
//! RMSNorm vector so the loader can be tested.

#![forbid(unsafe_code)]

use ojas_core::OjasError;

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
    let format = field(obj, "format").and_then(Json::as_str).unwrap_or("");
    if format != "ojas-oracle-fixture-v1" {
        return Err(bad("fixture format is not ojas-oracle-fixture-v1"));
    }
    if field(obj, "op").and_then(Json::as_str) != Some("rms_norm") {
        return Err(bad("fixture op is not rms_norm"));
    }
    let eps = field(obj, "eps").ok_or_else(|| bad("missing eps"))?.number()?;
    if !eps.is_finite() {
        return Err(OjasError::NonFinite { op: "oracle_fixture" });
    }
    let fixture = RmsNormFixture {
        eps,
        input: field(obj, "input").ok_or_else(|| bad("missing input"))?.numbers()?,
        input_shape: field(obj, "input_shape")
            .ok_or_else(|| bad("missing input_shape"))?
            .shape()?,
        weight: field(obj, "weight").ok_or_else(|| bad("missing weight"))?.numbers()?,
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

fn product(shape: &[usize]) -> Result<usize, OjasError> {
    let mut n = 1usize;
    for &dim in shape {
        n = n.checked_mul(dim).ok_or_else(|| bad("shape product overflows"))?;
    }
    Ok(n)
}

fn bad(detail: &str) -> OjasError {
    OjasError::OutOfRange {
        op: "oracle_fixture",
        detail: detail.to_string(),
    }
}

#[derive(Clone, Debug)]
enum Json {
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    fn object(&self) -> Result<&[(String, Json)], OjasError> {
        match self {
            Json::Object(pairs) => Ok(pairs),
            _ => Err(bad("expected an object")),
        }
    }

    fn number(&self) -> Result<f64, OjasError> {
        match self {
            Json::Number(value) => Ok(*value),
            _ => Err(bad("expected a number")),
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(text) => Some(text),
            _ => None,
        }
    }

    fn numbers(&self) -> Result<Vec<f64>, OjasError> {
        match self {
            Json::Array(items) => items.iter().map(Json::number).collect(),
            _ => Err(bad("expected an array of numbers")),
        }
    }

    fn shape(&self) -> Result<Vec<usize>, OjasError> {
        let nums = self.numbers()?;
        nums.into_iter()
            .map(|value| {
                if value.fract() != 0.0 || value < 0.0 || value > usize::MAX as f64 {
                    Err(bad("shape entry is not a non-negative integer"))
                } else {
                    Ok(value as usize)
                }
            })
            .collect()
    }
}

fn field<'a>(obj: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    obj.iter().find(|(name, _)| name == key).map(|(_, value)| value)
}

/// The fixture format nests two levels; deeper input is refused before the
/// recursive descent can exhaust the stack.
const MAX_DEPTH: usize = 16;

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

fn parse_json(text: &str) -> Result<Json, OjasError> {
    let mut parser = Parser {
        s: text.as_bytes(),
        i: 0,
        depth: 0,
    };
    let value = parser.value()?;
    parser.skip();
    if parser.i != parser.s.len() {
        return Err(bad("trailing data after fixture JSON"));
    }
    Ok(value)
}

impl<'a> Parser<'a> {
    fn value(&mut self) -> Result<Json, OjasError> {
        self.skip();
        let byte = self.peek().ok_or_else(|| bad("unexpected end of fixture"))?;
        match byte {
            b'{' | b'[' => {
                if self.depth == MAX_DEPTH {
                    return Err(bad("fixture nesting is too deep"));
                }
                self.depth += 1;
                let nested = if byte == b'{' { self.object() } else { self.array() };
                self.depth -= 1;
                nested
            }
            b'"' => Ok(Json::String(self.string()?)),
            b'-' | b'0'..=b'9' => Ok(Json::Number(self.number()?)),
            _ => Err(bad("unexpected fixture character")),
        }
    }

    fn object(&mut self) -> Result<Json, OjasError> {
        self.bump(b'{')?;
        let mut pairs = Vec::new();
        loop {
            self.skip();
            if self.eat(b'}') {
                break;
            }
            if !pairs.is_empty() {
                self.bump(b',')?;
                self.skip();
            }
            let key = self.string()?;
            if pairs.iter().any(|(name, _)| *name == key) {
                return Err(bad("duplicate fixture key"));
            }
            self.skip();
            self.bump(b':')?;
            let value = self.value()?;
            pairs.push((key, value));
        }
        Ok(Json::Object(pairs))
    }

    fn array(&mut self) -> Result<Json, OjasError> {
        self.bump(b'[')?;
        let mut items = Vec::new();
        loop {
            self.skip();
            if self.eat(b']') {
                break;
            }
            if !items.is_empty() {
                self.bump(b',')?;
            }
            items.push(self.value()?);
        }
        Ok(Json::Array(items))
    }

    fn string(&mut self) -> Result<String, OjasError> {
        self.skip();
        self.bump(b'"')?;
        let start = self.i;
        while let Some(byte) = self.peek() {
            if byte == b'"' {
                let text = std::str::from_utf8(&self.s[start..self.i])
                    .map_err(|_| bad("fixture string is not utf-8"))?
                    .to_string();
                self.i += 1;
                return Ok(text);
            }
            if byte == b'\\' {
                return Err(bad("fixture strings do not use escapes"));
            }
            self.i += 1;
        }
        Err(bad("unterminated fixture string"))
    }

    fn number(&mut self) -> Result<f64, OjasError> {
        let start = self.i;
        self.eat(b'-');
        let digits = self.i;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        if self.i == digits {
            return Err(bad("expected a number"));
        }
        if self.eat(b'.') {
            let frac = self.i;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
            if self.i == frac {
                return Err(bad("expected digits after decimal point"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            let exp = self.i;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
            if self.i == exp {
                return Err(bad("expected an exponent"));
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).map_err(|_| bad("bad number"))?;
        let value = text.parse::<f64>().map_err(|_| bad("bad number"))?;
        // `parse` rounds an out-of-range literal to infinity instead of failing.
        if value.is_finite() {
            Ok(value)
        } else {
            Err(bad("number does not fit in f64"))
        }
    }

    fn skip(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn eat(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn bump(&mut self, byte: u8) -> Result<(), OjasError> {
        if self.eat(byte) {
            Ok(())
        } else {
            Err(bad("unexpected fixture character"))
        }
    }
}
