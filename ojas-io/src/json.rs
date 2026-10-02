//! The workspace's strict JSON reader (RFC 8259).
//!
//! Objects, arrays, strings, numbers, `true`, `false` and `null`. Strings take
//! the standard escapes, with `\u` surrogates paired; a lone surrogate, a raw
//! control character or an unknown escape is refused. Numbers follow the RFC
//! grammar exactly: no leading zeros, no `+`, no bare `.`, no `NaN` or
//! `Infinity`. An integer literal (no fraction or exponent) is kept exact as
//! [`JsonNumber::U64`] or, when negative, [`JsonNumber::I64`]; one outside
//! those ranges is refused rather than rounded, a limit RFC 8259 §6 permits.
//! `-0` is [`JsonNumber::F64`]`(-0.0)` so its sign survives. Any other number
//! is parsed by std's correctly rounded `f64` parser; a literal that rounds to
//! infinity is refused, one that underflows becomes `0.0` (or `-0.0`).
//!
//! Refused as well: duplicate object keys (compared after unescaping),
//! anything after the root value, and input past any of [`JsonLimits`]:
//! input length, container depth, value count, and accounted tree bytes. Tree
//! bytes are one [`JsonValue`] per value, one [`String`] header per object
//! key, and the string contents; the budget is a multiple of the input length
//! plus a floor, so a short input of dense small numbers cannot turn into
//! gigabytes of nodes.
//!
//! Callers read the tree through typed accessors ([`JsonValue::as_u64`],
//! [`JsonValue::get_str`], [`JsonValue::deny_unknown_keys`], ...) that refuse
//! a type mismatch instead of coercing: `256.0` is not a `u64`, and
//! `9007199254740993` is not an `f64`.

use crate::error::IoError;
use std::collections::BTreeSet;

/// Default container depth. Wider than the three levels a safetensors header
/// uses; the cap exists to refuse absurd nesting before it costs stack.
pub const DEFAULT_MAX_DEPTH: usize = 64;
/// Hard ceiling on [`JsonLimits::max_depth`]. The parser recurses once per
/// container, so a larger caller-supplied limit is refused, not honoured.
pub const MAX_DEPTH_CEILING: usize = 512;
/// Default input cap, the same as the safetensors header cap
/// ([`crate::MAX_HEADER_BYTES`]).
pub const DEFAULT_MAX_INPUT_BYTES: usize = 100_000_000;
/// Default value cap: none beyond the input itself. Every value takes at
/// least one input byte, so the count never exceeds the input length.
pub const DEFAULT_MAX_NODES: usize = usize::MAX;
/// Default tree bytes allowed per input byte.
pub const DEFAULT_TREE_BYTES_PER_INPUT_BYTE: usize = 8;
/// Default tree bytes allowed regardless of input length, so a small document
/// parses even though a node is larger than one input byte.
pub const DEFAULT_TREE_BYTES_FLOOR: usize = 64 * 1024;

/// Limits one parse is held to. Start from [`JsonLimits::DEFAULT`] and
/// override fields: `JsonLimits { max_depth: 16, ..JsonLimits::DEFAULT }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JsonLimits {
    /// Longest input accepted, in bytes; checked before parsing.
    pub max_input_bytes: usize,
    /// Most containers nested inside one another. The root container counts
    /// as one; at most [`MAX_DEPTH_CEILING`].
    pub max_depth: usize,
    /// Most values in the tree (every object, array, string, number and
    /// literal; object keys are not values).
    pub max_nodes: usize,
    /// Tree byte budget: `tree_bytes_per_input_byte * input length +
    /// tree_bytes_floor`, saturating.
    pub tree_bytes_per_input_byte: usize,
    pub tree_bytes_floor: usize,
}

impl JsonLimits {
    pub const DEFAULT: JsonLimits = JsonLimits {
        max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
        max_depth: DEFAULT_MAX_DEPTH,
        max_nodes: DEFAULT_MAX_NODES,
        tree_bytes_per_input_byte: DEFAULT_TREE_BYTES_PER_INPUT_BYTE,
        tree_bytes_floor: DEFAULT_TREE_BYTES_FLOOR,
    };
}

impl Default for JsonLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A JSON number as written: exact integers, or an `f64`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JsonNumber {
    /// An integer literal `>= 0`.
    U64(u64),
    /// An integer literal `< 0`.
    I64(i64),
    /// A literal with a fraction or exponent, or `-0`.
    F64(f64),
}

/// One parsed value. Object members keep their input order; keys are unique.
#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(JsonNumber),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

/// Parse `text` under [`JsonLimits::DEFAULT`].
pub fn parse_json(text: &str) -> Result<JsonValue, IoError> {
    parse_json_with(text, &JsonLimits::DEFAULT)
}

/// Parse `text` under `limits`.
pub fn parse_json_with(text: &str, limits: &JsonLimits) -> Result<JsonValue, IoError> {
    if limits.max_depth > MAX_DEPTH_CEILING {
        return Err(IoError::new(format!(
            "JSON: max_depth {} is above the ceiling {MAX_DEPTH_CEILING}",
            limits.max_depth
        )));
    }
    if text.len() > limits.max_input_bytes {
        return Err(IoError::new(format!(
            "JSON: input of {} bytes exceeds the {}-byte limit",
            text.len(),
            limits.max_input_bytes
        )));
    }
    let mut p = Parser {
        s: text,
        b: text.as_bytes(),
        i: 0,
        nodes: 0,
        bytes: 0,
        max_depth: limits.max_depth,
        node_cap: limits.max_nodes,
        byte_cap: text
            .len()
            .saturating_mul(limits.tree_bytes_per_input_byte)
            .saturating_add(limits.tree_bytes_floor),
    };
    p.ws();
    let root = p.value(0)?;
    p.ws();
    if p.i != p.b.len() {
        return p.err("trailing bytes after the root value");
    }
    Ok(root)
}

struct Parser<'a> {
    s: &'a str,
    b: &'a [u8],
    i: usize,
    nodes: usize,
    bytes: usize,
    max_depth: usize,
    node_cap: usize,
    byte_cap: usize,
}

impl Parser<'_> {
    fn err<T>(&self, msg: &str) -> Result<T, IoError> {
        Err(IoError::new(format!("JSON: {msg} at byte {}", self.i)))
    }

    fn note(&mut self, n: usize) -> Result<(), IoError> {
        self.bytes = self.bytes.saturating_add(n);
        if self.bytes > self.byte_cap {
            return self.err("byte budget exceeded");
        }
        Ok(())
    }

    fn note_node(&mut self) -> Result<(), IoError> {
        self.nodes = self.nodes.saturating_add(1);
        if self.nodes > self.node_cap {
            return self.err("node budget exceeded");
        }
        self.note(std::mem::size_of::<JsonValue>())
    }

    fn ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.b.get(self.i) {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<(), IoError> {
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            self.err(&format!("expected {:?}", c as char))
        }
    }

    fn literal(&mut self, word: &str, v: JsonValue) -> Result<JsonValue, IoError> {
        if !self.b[self.i..].starts_with(word.as_bytes()) {
            return self.err("unexpected character");
        }
        self.note_node()?;
        self.i += word.len();
        Ok(v)
    }

    fn value(&mut self, depth: usize) -> Result<JsonValue, IoError> {
        match self.b.get(self.i) {
            Some(b'{' | b'[') if depth >= self.max_depth => {
                let msg = format!("nesting deeper than {}", self.max_depth);
                self.err(&msg)
            }
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => {
                let text = self.string()?;
                self.note_node()?;
                Ok(JsonValue::String(text))
            }
            Some(b'-' | b'0'..=b'9') => {
                let n = self.number()?;
                self.note_node()?;
                Ok(JsonValue::Number(n))
            }
            Some(b't') => self.literal("true", JsonValue::Bool(true)),
            Some(b'f') => self.literal("false", JsonValue::Bool(false)),
            Some(b'n') => self.literal("null", JsonValue::Null),
            Some(_) => self.err("unexpected character"),
            None => self.err("unexpected end"),
        }
    }

    fn object(&mut self, depth: usize) -> Result<JsonValue, IoError> {
        self.note_node()?;
        self.eat(b'{')?;
        let mut out = Vec::new();
        let mut seen = BTreeSet::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(JsonValue::Object(out));
        }
        loop {
            self.ws();
            if self.b.get(self.i) != Some(&b'"') {
                return self.err("expected a string key");
            }
            let k = self.string()?;
            // The key is a `String` beside the value node, not a value.
            self.note(std::mem::size_of::<String>())?;
            if !seen.insert(k.clone()) {
                return self.err(&format!("duplicate key {k:?}"));
            }
            self.ws();
            self.eat(b':')?;
            self.ws();
            let v = self.value(depth + 1)?;
            out.push((k, v));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(JsonValue::Object(out));
                }
                _ => return self.err("expected ',' or '}'"),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<JsonValue, IoError> {
        self.note_node()?;
        self.eat(b'[')?;
        let mut out = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(JsonValue::Array(out));
        }
        loop {
            self.ws();
            out.push(self.value(depth + 1)?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(JsonValue::Array(out));
                }
                _ => return self.err("expected ',' or ']'"),
            }
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while let Some(b'0'..=b'9') = self.b.get(self.i) {
            self.i += 1;
        }
        self.i - start
    }

    /// RFC 8259 §6: `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`.
    /// The grammar is checked here; std only converts a slice that passed it.
    fn number(&mut self) -> Result<JsonNumber, IoError> {
        let start = self.i;
        let negative = self.b.get(self.i) == Some(&b'-');
        if negative {
            self.i += 1;
        }
        let int_start = self.i;
        let int_digits = self.digits();
        if int_digits == 0 {
            return self.err("a number needs digits");
        }
        if int_digits > 1 && self.b[int_start] == b'0' {
            self.i = int_start;
            return self.err("number with a leading zero");
        }
        let mut float = false;
        if self.b.get(self.i) == Some(&b'.') {
            self.i += 1;
            if self.digits() == 0 {
                return self.err("a fraction needs digits");
            }
            float = true;
        }
        if let Some(b'e' | b'E') = self.b.get(self.i) {
            self.i += 1;
            if let Some(b'+' | b'-') = self.b.get(self.i) {
                self.i += 1;
            }
            if self.digits() == 0 {
                return self.err("an exponent needs digits");
            }
            float = true;
        }
        let text = &self.s[start..self.i];
        if float {
            let v: f64 = text
                .parse()
                .map_err(|_| IoError::new(format!("JSON: bad number at byte {start}")))?;
            if !v.is_finite() {
                return Err(IoError::new(format!(
                    "JSON: number {text} overflows f64 at byte {start}"
                )));
            }
            return Ok(JsonNumber::F64(v));
        }
        let magnitude: Option<u64> = self.s[int_start..self.i].parse().ok();
        match (negative, magnitude) {
            (false, Some(n)) => Ok(JsonNumber::U64(n)),
            (true, Some(0)) => Ok(JsonNumber::F64(-0.0)),
            (true, Some(n)) if n <= 1 << 63 => Ok(JsonNumber::I64((n as i64).wrapping_neg())),
            (false, None) => Err(IoError::new(format!(
                "JSON: integer {text} overflows u64 at byte {start}"
            ))),
            (true, _) => Err(IoError::new(format!(
                "JSON: integer {text} overflows i64 at byte {start}"
            ))),
        }
    }

    fn string(&mut self) -> Result<String, IoError> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            // Copy the run up to the next quote, backslash or control byte.
            // Those are ASCII, so the run ends on a char boundary of `s`.
            let run = self.i;
            while let Some(&c) = self.b.get(self.i) {
                if c == b'"' || c == b'\\' || c < 0x20 {
                    break;
                }
                self.i += 1;
            }
            if self.i > run {
                self.note(self.i - run)?;
                out.push_str(&self.s[run..self.i]);
            }
            let Some(&c) = self.b.get(self.i) else {
                return self.err("unterminated string");
            };
            if c < 0x20 {
                return self.err("control character in string");
            }
            self.i += 1;
            if c == b'"' {
                return Ok(out);
            }
            let Some(&e) = self.b.get(self.i) else {
                return self.err("unterminated escape");
            };
            self.i += 1;
            let ch = match e {
                b'"' => '"',
                b'\\' => '\\',
                b'/' => '/',
                b'b' => '\u{8}',
                b'f' => '\u{c}',
                b'n' => '\n',
                b'r' => '\r',
                b't' => '\t',
                b'u' => self.unicode_escape()?,
                _ => {
                    self.i -= 1;
                    return self.err("invalid escape");
                }
            };
            self.note(ch.len_utf8())?;
            out.push(ch);
        }
    }

    /// The code point of a `\u` escape whose `\u` was just read, joining a
    /// surrogate pair.
    fn unicode_escape(&mut self) -> Result<char, IoError> {
        let hi = self.hex4()?;
        let cp = if (0xD800..0xDC00).contains(&hi) {
            if self.b.get(self.i..self.i + 2) != Some(b"\\u".as_slice()) {
                return self.err("unpaired high surrogate");
            }
            self.i += 2;
            let lo = self.hex4()?;
            if !(0xDC00..0xE000).contains(&lo) {
                return self.err("invalid low surrogate");
            }
            0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
        } else if (0xDC00..0xE000).contains(&hi) {
            return self.err("unpaired low surrogate");
        } else {
            hi
        };
        char::from_u32(cp).map_or_else(|| self.err("invalid code point"), Ok)
    }

    fn hex4(&mut self) -> Result<u32, IoError> {
        let Some(h) = self.b.get(self.i..self.i.saturating_add(4)) else {
            return self.err("short \\u escape");
        };
        let mut v = 0u32;
        for &c in h {
            let Some(d) = (c as char).to_digit(16) else {
                return self.err("bad \\u escape");
            };
            v = v * 16 + d;
        }
        self.i += 4;
        Ok(v)
    }
}

/// `2^64` and `2^63` as `f64`, both exact.
const TWO_64: f64 = 18_446_744_073_709_551_616.0;
const TWO_63: f64 = 9_223_372_036_854_775_808.0;

impl JsonValue {
    /// What this value is, for error messages: `null`, `a boolean`, `an
    /// unsigned integer`, `a negative integer`, `a float`, `a string`, `an
    /// array` or `an object`.
    pub fn kind(&self) -> &'static str {
        match self {
            JsonValue::Null => "null",
            JsonValue::Bool(_) => "a boolean",
            JsonValue::Number(JsonNumber::U64(_)) => "an unsigned integer",
            JsonValue::Number(JsonNumber::I64(_)) => "a negative integer",
            JsonValue::Number(JsonNumber::F64(_)) => "a float",
            JsonValue::String(_) => "a string",
            JsonValue::Array(_) => "an array",
            JsonValue::Object(_) => "an object",
        }
    }

    fn mismatch<T>(&self, want: &str) -> Result<T, IoError> {
        Err(IoError::new(format!(
            "JSON: expected {want}, found {}",
            self.kind()
        )))
    }

    pub fn is_null(&self) -> bool {
        matches!(self, JsonValue::Null)
    }

    pub fn as_bool(&self) -> Result<bool, IoError> {
        match self {
            JsonValue::Bool(b) => Ok(*b),
            _ => self.mismatch("a boolean"),
        }
    }

    pub fn as_str(&self) -> Result<&str, IoError> {
        match self {
            JsonValue::String(s) => Ok(s),
            _ => self.mismatch("a string"),
        }
    }

    pub fn as_array(&self) -> Result<&[JsonValue], IoError> {
        match self {
            JsonValue::Array(a) => Ok(a),
            _ => self.mismatch("an array"),
        }
    }

    /// Members in input order.
    pub fn as_object(&self) -> Result<&[(String, JsonValue)], IoError> {
        match self {
            JsonValue::Object(o) => Ok(o),
            _ => self.mismatch("an object"),
        }
    }

    /// An integer literal `>= 0`. A float, even `256.0`, is refused, as is
    /// `-0`.
    pub fn as_u64(&self) -> Result<u64, IoError> {
        match self {
            JsonValue::Number(JsonNumber::U64(n)) => Ok(*n),
            _ => self.mismatch("an unsigned integer"),
        }
    }

    /// An integer literal in `i64` range. A float is refused.
    pub fn as_i64(&self) -> Result<i64, IoError> {
        match self {
            JsonValue::Number(JsonNumber::I64(n)) => Ok(*n),
            JsonValue::Number(JsonNumber::U64(n)) => i64::try_from(*n)
                .map_err(|_| IoError::new(format!("JSON: integer {n} overflows i64"))),
            _ => self.mismatch("an integer"),
        }
    }

    /// Any number whose value is exactly an `f64`: a float literal, or an
    /// integer that converts without rounding. `2^53 + 1` is refused.
    pub fn as_f64(&self) -> Result<f64, IoError> {
        let exact = match self {
            JsonValue::Number(JsonNumber::F64(v)) => return Ok(*v),
            JsonValue::Number(JsonNumber::U64(n)) => {
                // `as u64` saturates, so 2^64 must be excluded first.
                let f = *n as f64;
                (f < TWO_64 && f as u64 == *n).then_some(f)
            }
            JsonValue::Number(JsonNumber::I64(n)) => {
                let f = *n as f64;
                ((-TWO_63..TWO_63).contains(&f) && f as i64 == *n).then_some(f)
            }
            _ => return self.mismatch("a number"),
        };
        exact.ok_or_else(|| IoError::new("JSON: integer is not exactly representable as f64"))
    }

    /// Member `key` of this object, or `None` if the object lacks it. Refuses
    /// a value that is not an object.
    pub fn field_opt(&self, key: &str) -> Result<Option<&JsonValue>, IoError> {
        Ok(self
            .as_object()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v))
    }

    /// Member `key` of this object; a missing key is an error.
    pub fn field(&self, key: &str) -> Result<&JsonValue, IoError> {
        self.field_opt(key)?
            .ok_or_else(|| IoError::new(format!("JSON: missing key {key:?}")))
    }

    fn keyed<'a, T>(
        &'a self,
        key: &str,
        read: impl FnOnce(&'a JsonValue) -> Result<T, IoError>,
    ) -> Result<T, IoError> {
        read(self.field(key)?).map_err(|e| IoError::new(format!("{key:?}: {}", e.detail())))
    }

    pub fn get_bool(&self, key: &str) -> Result<bool, IoError> {
        self.keyed(key, JsonValue::as_bool)
    }

    pub fn get_str(&self, key: &str) -> Result<&str, IoError> {
        self.keyed(key, JsonValue::as_str)
    }

    pub fn get_u64(&self, key: &str) -> Result<u64, IoError> {
        self.keyed(key, JsonValue::as_u64)
    }

    pub fn get_i64(&self, key: &str) -> Result<i64, IoError> {
        self.keyed(key, JsonValue::as_i64)
    }

    pub fn get_f64(&self, key: &str) -> Result<f64, IoError> {
        self.keyed(key, JsonValue::as_f64)
    }

    pub fn get_array(&self, key: &str) -> Result<&[JsonValue], IoError> {
        self.keyed(key, JsonValue::as_array)
    }

    /// Member `key`, which must itself be an object.
    pub fn get_object(&self, key: &str) -> Result<&JsonValue, IoError> {
        self.keyed(key, |v| v.as_object().map(|_| v))
    }

    /// Refuse an object holding a key outside `allowed`, naming the first one
    /// in input order. For strict schemas, where a field a newer writer adds
    /// must not be silently ignored by an older reader.
    pub fn deny_unknown_keys(&self, allowed: &[&str]) -> Result<(), IoError> {
        self.deny_unknown_keys_at("", allowed)
    }

    /// [`JsonValue::deny_unknown_keys`] for an object nested at `path`, so
    /// the error names the full key: `path` `"text_config"` and key
    /// `"sliding_window"` report `"text_config.sliding_window"`. An empty
    /// `path` names the key alone.
    pub fn deny_unknown_keys_at(&self, path: &str, allowed: &[&str]) -> Result<(), IoError> {
        match self
            .as_object()?
            .iter()
            .find(|(k, _)| !allowed.contains(&k.as_str()))
        {
            Some((k, _)) if path.is_empty() => {
                Err(IoError::new(format!("JSON: unknown key {k:?}")))
            }
            Some((k, _)) => Err(IoError::new(format!(
                "JSON: unknown key {:?}",
                format!("{path}.{k}")
            ))),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<JsonValue, String> {
        parse_json(text).map_err(|e| e.detail().to_string())
    }

    fn nested(n: usize) -> String {
        let mut s = String::new();
        s.push_str(&"[".repeat(n));
        s.push('1');
        s.push_str(&"]".repeat(n));
        s
    }

    fn u(n: u64) -> JsonValue {
        JsonValue::Number(JsonNumber::U64(n))
    }

    fn i(n: i64) -> JsonValue {
        JsonValue::Number(JsonNumber::I64(n))
    }

    fn f(v: f64) -> JsonValue {
        JsonValue::Number(JsonNumber::F64(v))
    }

    fn s(v: &str) -> JsonValue {
        JsonValue::String(v.to_string())
    }

    #[test]
    fn depth_cap_is_64_by_default() {
        assert!(parse(&nested(64)).is_ok());
        let err = parse(&nested(65)).unwrap_err();
        assert!(err.contains("nesting deeper than 64"), "{err}");
    }

    #[test]
    fn very_deep_nesting_is_an_error_not_a_stack_overflow() {
        for open in ["[", "{\"k\":"] {
            let text = open.repeat(200_000);
            assert!(parse(&text).unwrap_err().contains("64"));
        }
        let ceiling = JsonLimits {
            max_depth: MAX_DEPTH_CEILING,
            ..JsonLimits::DEFAULT
        };
        let text = nested(MAX_DEPTH_CEILING);
        assert!(parse_json_with(&text, &ceiling).is_ok());
        let above = JsonLimits {
            max_depth: MAX_DEPTH_CEILING + 1,
            ..JsonLimits::DEFAULT
        };
        let err = parse_json_with("1", &above).unwrap_err();
        assert!(err.detail().contains("ceiling"), "{err}");
    }

    /// The tree accounting charges `size_of::<JsonValue>()` per value. That
    /// was 32 bytes for the integers-only tree this replaced, so the header
    /// budget accepts exactly what it did.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn a_value_is_32_bytes_as_before() {
        assert_eq!(std::mem::size_of::<JsonValue>(), 32);
        assert_eq!(std::mem::size_of::<String>(), 24);
    }

    #[test]
    fn every_limit_allows_its_boundary_and_refuses_one_past() {
        let zero = JsonLimits {
            tree_bytes_per_input_byte: 0,
            tree_bytes_floor: 0,
            ..JsonLimits::DEFAULT
        };
        // Tree bytes: "1" is one value (32); "[1]" two (64); {"a":1} is two
        // values, a key header and one key byte (32 + 24 + 1 + 32).
        for (text, need) in [("1", 32), ("[1]", 64), (r#"{"a":1}"#, 89), (r#""ab""#, 34)] {
            let at = JsonLimits {
                tree_bytes_floor: need,
                ..zero
            };
            assert!(parse_json_with(text, &at).is_ok(), "{text} at {need}");
            let under = JsonLimits {
                tree_bytes_floor: need - 1,
                ..zero
            };
            let err = parse_json_with(text, &under).unwrap_err();
            assert!(err.detail().contains("byte budget"), "{text}: {err}");
        }
        // The per-input-byte factor: "[1]" is 3 input bytes and needs 64.
        let factor = |k| JsonLimits {
            tree_bytes_per_input_byte: k,
            tree_bytes_floor: 1,
            ..JsonLimits::DEFAULT
        };
        assert!(parse_json_with("[1]", &factor(21)).is_ok());
        assert!(parse_json_with("[1]", &factor(20)).is_err());

        // Input bytes.
        let text = r#"{"k":[1,2]}"#;
        let input = |n| JsonLimits {
            max_input_bytes: n,
            ..JsonLimits::DEFAULT
        };
        assert!(parse_json_with(text, &input(text.len())).is_ok());
        let err = parse_json_with(text, &input(text.len() - 1)).unwrap_err();
        assert!(err.detail().contains("exceeds the 10-byte limit"), "{err}");

        // Depth.
        let depth = |n| JsonLimits {
            max_depth: n,
            ..JsonLimits::DEFAULT
        };
        assert!(parse_json_with(&nested(3), &depth(3)).is_ok());
        assert!(parse_json_with(&nested(3), &depth(2)).is_err());
        assert!(parse_json_with("7", &depth(0)).is_ok());
        assert!(parse_json_with("[]", &depth(0)).is_err());

        // Nodes: the object, the array, and two numbers.
        let nodes = |n| JsonLimits {
            max_nodes: n,
            ..JsonLimits::DEFAULT
        };
        assert!(parse_json_with(text, &nodes(4)).is_ok());
        let err = parse_json_with(text, &nodes(3)).unwrap_err();
        assert!(err.detail().contains("node budget"), "{err}");
    }

    #[test]
    fn dense_array_is_refused_before_gigabyte_amplification() {
        let mut text = String::from("[");
        for i in 0..20_000 {
            if i > 0 {
                text.push(',');
            }
            text.push('1');
        }
        text.push(']');
        let err = parse(&text).unwrap_err();
        assert!(err.contains("byte budget"), "{err}");
    }

    #[test]
    fn conformance_accepts() {
        let ok: Vec<(&str, JsonValue)> = vec![
            (r#""😀""#, s("😀")),
            (r#""é\u0000\/""#, s("é\0/")),
            (r#""\b\f\n\r\t\"\\""#, s("\u{8}\u{c}\n\r\t\"\\")),
            ("\"é字😀\u{7f}\"", s("é字😀\u{7f}")),
            (r#""￿""#, s("\u{FFFF}")),
            (r#""􏿿""#, s("\u{10FFFF}")),
            (r#""𝄞""#, s("𝄞")),
            ("18446744073709551615", u(u64::MAX)),
            ("0", u(0)),
            ("-1", i(-1)),
            ("-9223372036854775808", i(i64::MIN)),
            ("-0", f(-0.0)),
            ("-0.0", f(-0.0)),
            ("1.5", f(1.5)),
            ("256.0", f(256.0)),
            ("1e05", f(1e5)),
            ("1E+2", f(100.0)),
            ("1e-2", f(0.01)),
            ("0.1", f(0.1)),
            ("-12.5e-1", f(-1.25)),
            ("1e-400", f(0.0)),
            ("-1e-400", f(-0.0)),
            ("1.7976931348623157e308", f(f64::MAX)),
            ("5e-324", f(f64::from_bits(1))),
            ("true", JsonValue::Bool(true)),
            ("false", JsonValue::Bool(false)),
            ("null", JsonValue::Null),
            (" \t\r\n{} \n", JsonValue::Object(vec![])),
            (
                "[[],{}]",
                JsonValue::Array(vec![JsonValue::Array(vec![]), JsonValue::Object(vec![])]),
            ),
            (r#"{"":0}"#, JsonValue::Object(vec![(String::new(), u(0))])),
            ("[ 1 , 2 ]", JsonValue::Array(vec![u(1), u(2)])),
            (
                r#"{"b":null,"a":[true,false]}"#,
                JsonValue::Object(vec![
                    ("b".into(), JsonValue::Null),
                    (
                        "a".into(),
                        JsonValue::Array(vec![JsonValue::Bool(true), JsonValue::Bool(false)]),
                    ),
                ]),
            ),
        ];
        for (text, want) in ok {
            let got = parse(text);
            assert_eq!(got, Ok(want.clone()), "{text:?}");
            if let (
                Ok(JsonValue::Number(JsonNumber::F64(a))),
                JsonValue::Number(JsonNumber::F64(b)),
            ) = (&got, &want)
            {
                assert_eq!(a.to_bits(), b.to_bits(), "{text:?} sign or bits");
            }
        }
    }

    #[test]
    fn conformance_refuses() {
        let bad = [
            "",
            " ",
            r#""\ude00""#,
            r#""\ud83d""#,
            r#""\ud83dx""#,
            r#""\ud83dA""#,
            r#""\ud83d\ud83d""#,
            r#""\u12""#,
            r#""\u12G4""#,
            r#""\x""#,
            r#""\U0041""#,
            r#""\'""#,
            "\"abc",
            "\"a\u{1}b\"",
            "\"a\tb\"",
            "\"a\nb\"",
            "\"\\",
            "18446744073709551616",
            "99999999999999999999999999",
            "-9223372036854775809",
            "00",
            "01",
            "-01",
            "1.",
            ".5",
            "-.5",
            "1.e5",
            "+1",
            "-",
            "1e",
            "1e+",
            "1E-",
            "1e400",
            "-1e400",
            "NaN",
            "Infinity",
            "-Infinity",
            "0x10",
            "1_000",
            "tru",
            "True",
            "nul",
            "NULL",
            "truex",
            "{} x",
            "{}{}",
            "1 2",
            r#"{"a":1,}"#,
            "[1,]",
            "[,1]",
            "[1 2]",
            "[",
            "{",
            "]",
            r#"{"a"}"#,
            r#"{"a":}"#,
            r#"{"a" 1}"#,
            "{1:2}",
            "{'a':1}",
            r#"{"a":1,"a":2}"#,
            r#"{"a":1,"a":1}"#,
            "\u{feff}{}",
            "\u{a0}{}",
            "/* c */ 1",
            "[1]\u{0}",
        ];
        for text in bad {
            let r = std::panic::catch_unwind(|| parse(text));
            assert!(
                matches!(r, Ok(Err(_))),
                "{text:?} parsed or panicked: {r:?}"
            );
        }
    }

    #[test]
    fn typed_accessors_refuse_mismatches_and_inexact_values() {
        let v = parse_json(
            r#"{"n":256,"x":256.0,"neg":-3,"z":-0,"s":"t","b":true,"a":[1],"o":{},"nil":null,
               "big":9007199254740993,"edge":9007199254740992,"max":18446744073709551615,
               "min":-9223372036854775808}"#,
        )
        .unwrap();
        assert_eq!(v.get_u64("n").unwrap(), 256);
        assert!(v
            .get_u64("x")
            .unwrap_err()
            .detail()
            .contains("found a float"));
        assert!(v.get_u64("neg").is_err());
        assert!(v.get_u64("z").is_err());
        assert_eq!(v.get_i64("neg").unwrap(), -3);
        assert_eq!(v.get_i64("n").unwrap(), 256);
        assert!(v
            .get_i64("max")
            .unwrap_err()
            .detail()
            .contains("overflows i64"));
        assert!(v.get_i64("x").is_err());
        assert_eq!(v.get_f64("x").unwrap(), 256.0);
        assert_eq!(v.get_f64("n").unwrap(), 256.0);
        assert_eq!(v.get_f64("z").unwrap().to_bits(), (-0.0f64).to_bits());
        assert_eq!(v.get_f64("edge").unwrap(), 9007199254740992.0);
        assert!(v.get_f64("big").unwrap_err().detail().contains("exactly"));
        assert!(v.get_f64("max").is_err(), "u64::MAX rounds to 2^64");
        assert_eq!(v.get_f64("min").unwrap(), -TWO_63);
        assert!(v.get_f64("s").is_err());
        assert_eq!(v.get_str("s").unwrap(), "t");
        assert!(v.get_str("n").is_err());
        assert!(v.get_bool("b").unwrap());
        assert!(v.get_bool("nil").is_err());
        assert_eq!(v.get_array("a").unwrap(), [u(1)]);
        assert!(v.get_object("o").unwrap().as_object().unwrap().is_empty());
        assert!(v.get_object("a").is_err());
        assert!(v.field("nil").unwrap().is_null());
        let err = v.get_u64("missing").unwrap_err();
        assert!(err.detail().contains("missing key \"missing\""), "{err}");
        assert!(v.field_opt("missing").unwrap().is_none());
        let err = v.get_str("n").unwrap_err();
        assert!(
            err.detail().starts_with("\"n\": ") && err.detail().contains("an unsigned integer"),
            "{err}"
        );
        assert!(u(1).field("k").is_err(), "field on a non-object");
        assert!(u(1).as_object().is_err());
        assert!(s("x").as_array().is_err());
    }

    #[test]
    fn deny_unknown_keys_names_the_first_stranger() {
        let v = parse_json(r#"{"a":1,"zz":2,"b":3,"yy":4}"#).unwrap();
        assert!(v.deny_unknown_keys(&["a", "b", "zz", "yy"]).is_ok());
        let err = v.deny_unknown_keys(&["a", "b"]).unwrap_err();
        assert!(err.detail().contains("unknown key \"zz\""), "{err}");
        assert!(u(1).deny_unknown_keys(&[]).is_err());
        assert!(JsonValue::Object(vec![]).deny_unknown_keys(&[]).is_ok());
    }

    #[test]
    fn deny_unknown_keys_at_names_the_full_path() {
        let v = parse_json(r#"{"text_config":{"hidden":1,"sliding_window":2}}"#).unwrap();
        let inner = v.field("text_config").unwrap();
        assert!(inner
            .deny_unknown_keys_at("text_config", &["hidden", "sliding_window"])
            .is_ok());
        let err = inner
            .deny_unknown_keys_at("text_config", &["hidden"])
            .unwrap_err();
        assert!(
            err.detail()
                .contains("unknown key \"text_config.sliding_window\""),
            "{err}"
        );
        // An empty path is the plain form.
        let err = inner.deny_unknown_keys_at("", &["hidden"]).unwrap_err();
        assert!(
            err.detail().contains("unknown key \"sliding_window\""),
            "{err}"
        );
        assert!(u(1).deny_unknown_keys_at("p", &[]).is_err());
    }

    /// Render a value as JSON. Floats use `{:?}`, which keeps `.0`, so a float
    /// does not come back as an integer.
    fn render(v: &JsonValue, out: &mut String) {
        match v {
            JsonValue::Null => out.push_str("null"),
            JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            JsonValue::Number(JsonNumber::U64(n)) => out.push_str(&n.to_string()),
            JsonValue::Number(JsonNumber::I64(n)) => out.push_str(&n.to_string()),
            JsonValue::Number(JsonNumber::F64(x)) => out.push_str(&format!("{x:?}")),
            JsonValue::String(t) => {
                out.push('"');
                for c in t.chars() {
                    match c {
                        '"' | '\\' => {
                            out.push('\\');
                            out.push(c);
                        }
                        c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                        c => out.push(c),
                    }
                }
                out.push('"');
            }
            JsonValue::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    render(item, out);
                }
                out.push(']');
            }
            JsonValue::Object(entries) => {
                out.push('{');
                for (i, (k, item)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    render(&JsonValue::String(k.clone()), out);
                    out.push(':');
                    render(item, out);
                }
                out.push('}');
            }
        }
    }

    fn random_value(rng: &mut crate::test_util::Mix, depth: usize) -> JsonValue {
        let pick = if depth >= 4 {
            rng.below(6)
        } else {
            rng.below(8)
        };
        match pick {
            0 => JsonValue::Null,
            1 => JsonValue::Bool(rng.below(2) == 0),
            2 => u(match rng.below(3) {
                0 => rng.below(10) as u64,
                1 => u64::MAX - rng.below(3) as u64,
                _ => rng.next(),
            }),
            3 => i(match rng.below(3) {
                0 => -1 - rng.below(10) as i64,
                1 => i64::MIN + rng.below(3) as i64,
                _ => (rng.next() as i64) | i64::MIN,
            }),
            4 => {
                let x = f64::from_bits(rng.next());
                f(if x.is_finite() {
                    x
                } else {
                    rng.below(1000) as f64 / 8.0
                })
            }
            5 => {
                const SOUP: [char; 10] =
                    ['a', '"', '\\', '\n', '\u{1}', 'é', '😀', '/', ' ', '\u{7f}'];
                JsonValue::String((0..rng.below(6)).map(|_| SOUP[rng.below(10)]).collect())
            }
            6 => JsonValue::Array(
                (0..rng.below(4))
                    .map(|_| random_value(rng, depth + 1))
                    .collect(),
            ),
            _ => JsonValue::Object(
                (0..rng.below(4))
                    .map(|k| (format!("k{k}"), random_value(rng, depth + 1)))
                    .collect(),
            ),
        }
    }

    #[test]
    fn seeded_values_round_trip_exactly() {
        let mut rng = crate::test_util::Mix::new(0x150A);
        for n in 0..5_000 {
            let v = random_value(&mut rng, 0);
            let mut text = String::new();
            render(&v, &mut text);
            let back = parse(&text).unwrap_or_else(|e| panic!("case {n}: {text:?}: {e}"));
            let (mut a, mut b) = (String::new(), String::new());
            render(&v, &mut a);
            render(&back, &mut b);
            assert_eq!(a, b, "case {n}");
            assert_eq!(back, v, "case {n}: {text:?}");
        }
    }

    #[test]
    fn token_soup_never_panics_and_accepted_values_round_trip() {
        const TOKENS: [&str; 30] = [
            "{",
            "}",
            "[",
            "]",
            ":",
            ",",
            "\"",
            "\"a\"",
            "\"a\":",
            " ",
            "0",
            "7",
            "18446744073709551615",
            "\\",
            "\\u",
            "d83d",
            "\\ude00",
            "\\n",
            "é",
            "\u{1}",
            "-",
            ".",
            "e",
            "{\"k\":[",
            "true",
            "null",
            "1.5",
            "e9",
            "+",
            "fals",
        ];
        let mut rng = crate::test_util::Mix::new(0x150);
        let mut accepted = 0;
        for _ in 0..20_000 {
            let text: String = (0..1 + rng.below(12))
                .map(|_| TOKENS[rng.below(TOKENS.len())])
                .collect();
            let parsed = std::panic::catch_unwind(|| parse(&text))
                .unwrap_or_else(|_| panic!("panicked on {text:?}"));
            if let Ok(v) = parsed {
                let mut again = String::new();
                render(&v, &mut again);
                assert_eq!(parse(&again), Ok(v), "{text:?} -> {again:?}");
                accepted += 1;
            }
        }
        assert!(accepted > 100, "accepted {accepted}");
    }
}
