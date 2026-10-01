//! Strict JSON for safetensors headers.
//!
//! Objects, arrays, strings, and non-negative integers only. Duplicate keys,
//! trailing junk, and more than [`MAX_DEPTH`] nested containers are errors.
//! The grammar follows the safetensors subset (no floats, `true`, `false`, or
//! `null`). The depth cap here is 64, which is the ojas limit for absurd
//! nesting. It is wider than the three levels the header format itself uses.

use std::collections::BTreeSet;

pub(crate) const MAX_DEPTH: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Json {
    Object(Vec<(String, Json)>),
    Array(Vec<Json>),
    Str(String),
    /// Non-negative integer that fits in `u64`.
    Uint(u64),
}

pub(crate) fn parse(text: &str) -> Result<Json, String> {
    let mut p = Parser {
        s: text.as_bytes(),
        i: 0,
    };
    p.ws();
    let root = p.value(0)?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("trailing bytes after the root value at {}", p.i));
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

    fn err<T>(&self, msg: &str) -> Result<T, String> {
        Err(format!("header JSON: {msg} at byte {}", self.i))
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            self.err(&format!("expected {:?}", c as char))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        match self.s.get(self.i) {
            Some(b'{' | b'[') if depth >= MAX_DEPTH => self.err("nesting deeper than 64"),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'0'..=b'9') => self.number(),
            Some(_) => self.err(
                "unsupported JSON value (only objects, arrays, strings and non-negative integers)",
            ),
            None => self.err("unexpected end"),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, String> {
        self.eat(b'{')?;
        let mut out = Vec::new();
        let mut seen = BTreeSet::new();
        self.ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Object(out));
        }
        loop {
            self.ws();
            if self.s.get(self.i) != Some(&b'"') {
                return self.err("expected a string key");
            }
            let k = self.string()?;
            if !seen.insert(k.clone()) {
                return self.err(&format!("duplicate key {k:?}"));
            }
            self.ws();
            self.eat(b':')?;
            self.ws();
            let v = self.value(depth + 1)?;
            out.push((k, v));
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Object(out));
                }
                _ => return self.err("expected ',' or '}'"),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, String> {
        self.eat(b'[')?;
        let mut out = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Array(out));
        }
        loop {
            self.ws();
            out.push(self.value(depth + 1)?);
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Array(out));
                }
                _ => return self.err("expected ',' or ']'"),
            }
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        let mut uint: Option<u64> = Some(0);
        let mut digits = 0usize;
        while let Some(&c @ b'0'..=b'9') = self.s.get(self.i) {
            uint = uint
                .and_then(|v| v.checked_mul(10))
                .and_then(|v| v.checked_add(u64::from(c - b'0')));
            self.i += 1;
            digits += 1;
        }
        if digits == 0 {
            return self.err("a number needs digits");
        }
        if digits > 1 && self.s[start] == b'0' {
            return self.err("integer with a leading zero");
        }
        if matches!(self.s.get(self.i), Some(b'.' | b'e' | b'E')) {
            return self.err("non-integer number");
        }
        match uint {
            Some(n) => Ok(Json::Uint(n)),
            None => Err(format!(
                "header JSON: integer overflows u64 at byte {start}"
            )),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let Some(&c) = self.s.get(self.i) else {
                return self.err("unterminated string");
            };
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let Some(&e) = self.s.get(self.i) else {
                        return self.err("unterminated escape");
                    };
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
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if self.slice_at(2) != Some(b"\\u".as_slice()) {
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
                            match char::from_u32(cp) {
                                Some(ch) => out.push(ch),
                                None => return self.err("invalid code point"),
                            }
                        }
                        _ => return self.err("invalid escape"),
                    }
                }
                0x00..=0x1f => return self.err("control character in string"),
                _ => {
                    let start = self.i - 1;
                    let len = match c {
                        0x00..=0x7f => 1,
                        0xc0..=0xdf => 2,
                        0xe0..=0xef => 3,
                        _ => 4,
                    };
                    let end = start
                        .checked_add(len)
                        .ok_or_else(|| format!("header JSON: bad UTF-8 at byte {start}"))?;
                    let chunk = self
                        .s
                        .get(start..end)
                        .ok_or_else(|| format!("header JSON: bad UTF-8 at byte {start}"))?;
                    let chunk = std::str::from_utf8(chunk)
                        .map_err(|_| format!("header JSON: bad UTF-8 at byte {start}"))?;
                    out.push_str(chunk);
                    self.i = end;
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self
            .slice_at(4)
            .ok_or_else(|| format!("header JSON: short \\u escape at byte {}", self.i))?;
        let mut v = 0u32;
        for &c in h {
            let d = (c as char)
                .to_digit(16)
                .ok_or_else(|| format!("header JSON: bad \\u escape at byte {}", self.i))?;
            v = v * 16 + d;
        }
        self.i += 4;
        Ok(v)
    }

    fn slice_at(&self, len: usize) -> Option<&[u8]> {
        let end = self.i.checked_add(len)?;
        self.s.get(self.i..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nested(n: usize) -> String {
        let mut s = String::new();
        s.push_str(&"[".repeat(n));
        s.push('1');
        s.push_str(&"]".repeat(n));
        s
    }

    #[test]
    fn depth_cap_is_64() {
        assert!(parse(&nested(64)).is_ok());
        let err = parse(&nested(65)).unwrap_err();
        assert!(err.contains("64"), "{err}");
    }

    #[test]
    fn duplicate_keys_and_non_integers_fail() {
        assert!(parse(r#"{"a":1,"a":2}"#)
            .unwrap_err()
            .contains("duplicate key"));
        assert!(parse("1.5").unwrap_err().contains("non-integer"));
        assert!(parse("-1").unwrap_err().contains("unsupported JSON value"));
        assert!(parse("01").unwrap_err().contains("leading zero"));
    }

    fn s(v: &str) -> Json {
        Json::Str(v.to_string())
    }

    #[test]
    fn escapes_numbers_and_framing_edge_cases() {
        let ok: [(&str, Json); 12] = [
            (r#""\ud83d\ude00""#, s("😀")),
            (r#""\u00E9\u0000\/""#, s("é\0/")),
            (r#""\b\f\n\r\t\"\\""#, s("\u{8}\u{c}\n\r\t\"\\")),
            ("\"é字😀\u{7f}\"", s("é字😀\u{7f}")),
            ("18446744073709551615", Json::Uint(u64::MAX)),
            ("0", Json::Uint(0)),
            (" \t\r\n{} \n", Json::Object(vec![])),
            (
                "[[],{}]",
                Json::Array(vec![Json::Array(vec![]), Json::Object(vec![])]),
            ),
            (
                r#"{"":0}"#,
                Json::Object(vec![(String::new(), Json::Uint(0))]),
            ),
            (r#""\uFFFF""#, s("\u{FFFF}")),
            (r#""\udbff\udfff""#, s("\u{10FFFF}")),
            ("[ 1 , 2 ]", Json::Array(vec![Json::Uint(1), Json::Uint(2)])),
        ];
        for (text, want) in ok {
            assert_eq!(parse(text), Ok(want), "{text:?}");
        }
        let bad = [
            "",
            " ",
            r#""\ude00""#,
            r#""\ud83d""#,
            r#""\ud83dx""#,
            r#""\ud83d\u0041""#,
            r#""\ud83d\ud83d""#,
            r#""\u12""#,
            r#""\u12G4""#,
            r#""\x""#,
            "\"abc",
            "\"a\u{1}b\"",
            "\"\\",
            "18446744073709551616",
            "99999999999999999999999999",
            "00",
            "1e5",
            "1.",
            "-0",
            "+1",
            "{} x",
            "{}{}",
            r#"{"a":1,}"#,
            "[1,]",
            "[1 2]",
            "[",
            "{",
            r#"{"a"}"#,
            r#"{"a":}"#,
            "{1:2}",
            r#"{"a":1,"\u0061":2}"#,
            "true",
            "null",
            "\u{feff}{}",
            "\u{a0}{}",
        ];
        for text in bad {
            assert!(parse(text).is_err(), "{text:?} parsed");
        }
    }

    #[test]
    fn very_deep_nesting_is_an_error_not_a_stack_overflow() {
        for open in ["[", "{\"k\":"] {
            let text = open.repeat(200_000);
            assert!(parse(&text).unwrap_err().contains("64"));
        }
    }

    fn render(v: &Json, out: &mut String) {
        match v {
            Json::Uint(n) => out.push_str(&n.to_string()),
            Json::Str(t) => {
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
            Json::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    render(item, out);
                }
                out.push(']');
            }
            Json::Object(entries) => {
                out.push('{');
                for (i, (k, item)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    render(&Json::Str(k.clone()), out);
                    out.push(':');
                    render(item, out);
                }
                out.push('}');
            }
        }
    }

    #[test]
    fn token_soup_never_panics_and_accepted_values_round_trip() {
        const TOKENS: [&str; 24] = [
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
