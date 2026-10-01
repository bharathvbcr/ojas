//! A strict JSON reader for the two Hugging Face files this crate reads:
//! `config.json` and `model.safetensors.index.json`.
//!
//! This is a stopgap with a removal plan. tessl's parser (`tessl/src/json.rs`)
//! reads exactly these files but is `mod json` (crate-private), and ojas-io's
//! is crate-private and has no floats, booleans or `null`, which `config.json`
//! needs. The value shape below mirrors tessl's (`Num { value, uint }`), so the
//! day tessl exposes `pub mod json` (requested), this module is deleted and
//! its two callers switch to it.
//!
//! RFC 8259 only: unique keys, the standard escapes with surrogates paired,
//! no raw control characters, no leading zeros or `+`, nothing after the root
//! value. Depth and input size are capped.

/// Containers deeper than this are refused. `config.json` uses three levels.
const MAX_DEPTH: usize = 32;
/// Inputs larger than this are refused. The 2B's index is 64 KB.
pub(crate) const MAX_INPUT_BYTES: usize = 16 << 20;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Json {
    Object(Vec<(String, Json)>),
    Array(Vec<Json>),
    Str(String),
    /// A number, with its exact value when it is a non-negative integer that
    /// fits `u64`.
    Num {
        value: f64,
        uint: Option<u64>,
    },
    Bool(bool),
    Null,
}

impl Json {
    pub(crate) fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub(crate) fn fields(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(fields) => Some(fields),
            _ => None,
        }
    }

    /// A short description for error messages (never the whole value).
    pub(crate) fn kind(&self) -> String {
        match self {
            Json::Object(_) => "an object".into(),
            Json::Array(a) => format!("an array of {}", a.len()),
            Json::Str(s) => format!("the string {s:?}"),
            Json::Num { value, .. } => format!("the number {value}"),
            Json::Bool(b) => format!("{b}"),
            Json::Null => "null".into(),
        }
    }
}

pub(crate) fn parse(text: &str, what: &str) -> Result<Json, String> {
    if text.len() > MAX_INPUT_BYTES {
        return Err(format!(
            "{what}: {} bytes exceeds the {MAX_INPUT_BYTES}-byte cap",
            text.len()
        ));
    }
    let mut p = Parser { s: text.as_bytes(), i: 0 };
    p.ws();
    let root = p.value(0).map_err(|e| format!("{what}: {e}"))?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("{what}: trailing bytes after the root value at byte {}", p.i));
    }
    Ok(root)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.s.get(self.i) {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn expect(&mut self, lit: &[u8]) -> Result<(), String> {
        if self.s[self.i..].starts_with(lit) {
            self.i += lit.len();
            Ok(())
        } else {
            Err(format!("expected {:?} at byte {}", String::from_utf8_lossy(lit), self.i))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        match self.peek() {
            Some(b'{') => self.object(depth + 1),
            Some(b'[') => self.array(depth + 1),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.expect(b"true").map(|_| Json::Bool(true)),
            Some(b'f') => self.expect(b"false").map(|_| Json::Bool(false)),
            Some(b'n') => self.expect(b"null").map(|_| Json::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(c) => Err(format!("unexpected byte {c:#04x} at {}", self.i)),
            None => Err("unexpected end of input".into()),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return Err(format!("nesting deeper than {MAX_DEPTH}"));
        }
        self.i += 1;
        let mut fields: Vec<(String, Json)> = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Object(fields));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err(format!("expected a key at byte {}", self.i));
            }
            let key = self.string()?;
            if fields.iter().any(|(k, _)| *k == key) {
                return Err(format!("duplicate key {key:?}"));
            }
            self.ws();
            self.expect(b":")?;
            self.ws();
            let v = self.value(depth)?;
            fields.push((key, v));
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Object(fields));
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return Err(format!("nesting deeper than {MAX_DEPTH}"));
        }
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value(depth)?);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let digits = self
            .s
            .get(self.i..self.i + 4)
            .ok_or_else(|| "truncated \\u escape".to_string())?;
        let text = std::str::from_utf8(digits).map_err(|_| "bad \\u escape".to_string())?;
        if !text.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("bad \\u escape {text:?}"));
        }
        self.i += 4;
        u32::from_str_radix(text, 16).map_err(|e| e.to_string())
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let start = self.i;
            while let Some(&c) = self.s.get(self.i) {
                if c == b'"' || c == b'\\' || c < 0x20 {
                    break;
                }
                self.i += 1;
            }
            out.push_str(
                std::str::from_utf8(&self.s[start..self.i]).map_err(|_| format!("invalid UTF-8 near byte {start}"))?,
            );
            match self.peek() {
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let e = self.peek().ok_or("truncated escape")?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&hi) {
                                self.expect(b"\\u").map_err(|_| "unpaired high surrogate".to_string())?;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err("unpaired high surrogate".into());
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return Err("unpaired low surrogate".into());
                            } else {
                                hi
                            };
                            out.push(char::from_u32(code).ok_or("invalid code point")?);
                        }
                        other => return Err(format!("unknown escape \\{}", other as char)),
                    }
                }
                Some(c) => return Err(format!("raw control character {c:#04x} in a string")),
                None => return Err("unterminated string".into()),
            }
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while let Some(b'0'..=b'9') = self.peek() {
            self.i += 1;
        }
        self.i - start
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        let negative = self.peek() == Some(b'-');
        if negative {
            self.i += 1;
        }
        let int_start = self.i;
        let n = self.digits();
        if n == 0 {
            return Err(format!("a number needs digits at byte {int_start}"));
        }
        if n > 1 && self.s[int_start] == b'0' {
            return Err(format!("leading zero at byte {int_start}"));
        }
        let mut integral = true;
        if self.peek() == Some(b'.') {
            self.i += 1;
            if self.digits() == 0 {
                return Err(format!("a fraction needs digits at byte {}", self.i));
            }
            integral = false;
        }
        if let Some(b'e' | b'E') = self.peek() {
            self.i += 1;
            if let Some(b'+' | b'-') = self.peek() {
                self.i += 1;
            }
            if self.digits() == 0 {
                return Err(format!("an exponent needs digits at byte {}", self.i));
            }
            integral = false;
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).map_err(|e| e.to_string())?;
        let value: f64 = text.parse().map_err(|_| format!("bad number {text:?}"))?;
        if !value.is_finite() {
            return Err(format!("number {text:?} is not finite as f64"));
        }
        let uint = if integral && !negative { text.parse::<u64>().ok() } else { None };
        Ok(Json::Num { value, uint })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_what_config_json_uses() {
        let j = parse(
            r#"{"a": 1e-06, "b": 10000000, "c": [11, 11, 10], "d": true, "e": null, "f": "sé\n", "g": -0.5}"#,
            "t",
        )
        .unwrap();
        assert_eq!(j.get("a"), Some(&Json::Num { value: 1e-6, uint: None }));
        assert_eq!(
            j.get("b"),
            Some(&Json::Num {
                value: 1e7,
                uint: Some(10_000_000)
            })
        );
        assert_eq!(j.get("d"), Some(&Json::Bool(true)));
        assert_eq!(j.get("e"), Some(&Json::Null));
        assert_eq!(j.get("f"), Some(&Json::Str("sé\n".into())));
        assert_eq!(j.get("g"), Some(&Json::Num { value: -0.5, uint: None }));
    }

    #[test]
    fn refuses_what_rfc_8259_refuses() {
        for bad in [
            r#"{"a": 1, "a": 2}"#,
            r#"{"a": 01}"#,
            r#"{"a": +1}"#,
            r#"{"a": 1.}"#,
            r#"{"a": .5}"#,
            r#"{"a": NaN}"#,
            r#"{"a": 1e999}"#,
            r#"{"a": "\ud800"}"#,
            r#"{"a": "\udc00"}"#,
            "{\"a\": \"\u{1}\"}",
            r#"{"a": 1} x"#,
            r#"{"a": tru}"#,
            r#"{"a": [1,]}"#,
            r#"{"a": 1,}"#,
            "",
        ] {
            assert!(parse(bad, "t").is_err(), "{bad:?} accepted");
        }
        let deep = "[".repeat(MAX_DEPTH + 1) + &"]".repeat(MAX_DEPTH + 1);
        assert!(parse(&deep, "t").unwrap_err().contains("nesting"));
        let ok = "[".repeat(MAX_DEPTH) + &"]".repeat(MAX_DEPTH);
        assert!(parse(&ok, "t").is_ok());
    }
}
