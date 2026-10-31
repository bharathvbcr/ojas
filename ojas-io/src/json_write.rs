//! The workspace's compact JSON writer, over the reader's [`JsonValue`].
//!
//! It moved here from ojas-cuda's `json` (the rung-0 report writer) so the
//! reader and the writer share one tree type. Objects keep insertion order, so
//! a report's text is a function of its content. JSON has no NaN or infinity:
//! a non-finite float is written as the string `"NaN"`, `"inf"` or `"-inf"`,
//! never as a number, so a reader cannot mistake it for a measured value. The
//! parser never produces a non-finite [`JsonNumber::F64`]; only a tree built
//! by hand can hold one.
//!
//! This is not Python's `json.dumps`: finite floats use Rust's shortest
//! round-trip `Debug` form (`1e-7`, `1e16`), where Python writes `1e-07` and
//! `1e+16`. `ojas-train`'s `pyjson` is the Python-exact writer.

use crate::json::{JsonNumber, JsonValue};

/// An object under construction; becomes a [`JsonValue::Object`] through `From`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JsonObj(Vec<(String, JsonValue)>);

impl JsonObj {
    /// An empty object.
    pub fn new() -> Self {
        JsonObj(Vec::new())
    }

    /// Append `key: value`. A repeated key is kept, not merged.
    pub fn with(mut self, key: &str, value: impl Into<JsonValue>) -> Self {
        self.0.push((key.to_string(), value.into()));
        self
    }

    /// Append `key: value` in place.
    pub fn push(&mut self, key: &str, value: impl Into<JsonValue>) {
        self.0.push((key.to_string(), value.into()));
    }
}

impl From<JsonObj> for JsonValue {
    fn from(o: JsonObj) -> Self {
        JsonValue::Object(o.0)
    }
}

impl JsonValue {
    /// The compact text: no spaces, members in their stored order.
    pub fn render(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            JsonValue::Null => out.push_str("null"),
            JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            JsonValue::Number(JsonNumber::U64(u)) => out.push_str(&u.to_string()),
            JsonValue::Number(JsonNumber::I64(i)) => out.push_str(&i.to_string()),
            JsonValue::Number(JsonNumber::F64(x)) => {
                if x.is_nan() {
                    out.push_str("\"NaN\"");
                } else if x.is_infinite() {
                    out.push_str(if *x > 0.0 { "\"inf\"" } else { "\"-inf\"" });
                } else {
                    // Debug is Rust's shortest round-trip form ("0.5", "1e-7"),
                    // which is valid JSON for every finite value.
                    out.push_str(&format!("{x:?}"));
                }
            }
            JsonValue::String(s) => write_str(s, out),
            JsonValue::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            JsonValue::Object(fields) => {
                out.push('{');
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_str(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

/// `s` as a JSON string literal, quotes included: `"`, `\` and the control
/// characters are escaped, everything else (non-ASCII too) is written as is.
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    write_str(s, &mut out);
    out
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
}

impl From<i32> for JsonValue {
    fn from(i: i32) -> Self {
        JsonValue::from(i64::from(i))
    }
}

impl From<f64> for JsonValue {
    /// Any `f64`, a non-finite one included: [`JsonValue::render`] writes it as a string.
    fn from(x: f64) -> Self {
        JsonValue::Number(JsonNumber::F64(x))
    }
}

impl From<f32> for JsonValue {
    fn from(x: f32) -> Self {
        JsonValue::Number(JsonNumber::F64(f64::from(x)))
    }
}

impl<T: Into<JsonValue>> From<Option<T>> for JsonValue {
    /// `None` is `null`.
    fn from(o: Option<T>) -> Self {
        o.map_or(JsonValue::Null, Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse_json;

    #[test]
    fn objects_keep_insertion_order_and_nest() {
        let j = JsonObj::new()
            .with("b", 1u32)
            .with("a", vec!["x", "y"])
            .with(
                "c",
                JsonObj::new()
                    .with("ok", true)
                    .with("none", Option::<u32>::None),
            );
        assert_eq!(
            JsonValue::from(j).render(),
            r#"{"b":1,"a":["x","y"],"c":{"ok":true,"none":null}}"#
        );
    }

    #[test]
    fn strings_are_escaped() {
        let j = JsonValue::from("a\"b\\c\nd\u{1}é");
        assert_eq!(j.render(), "\"a\\\"b\\\\c\\nd\\u0001é\"");
        assert_eq!(quote("a\"b\\c\nd\u{1}é"), j.render());
        assert_eq!(quote(""), "\"\"");
    }

    #[test]
    fn a_quoted_string_reads_back_as_the_same_string() {
        for s in [
            "",
            "plain",
            "q\"b",
            "back\\slash",
            "nl\nt\tcr\r",
            "\u{1}\u{1f}",
            "é😀",
        ] {
            assert_eq!(
                parse_json(&quote(s)).unwrap(),
                JsonValue::String(s.to_string()),
                "{s:?}"
            );
        }
    }

    #[test]
    fn non_finite_floats_are_strings_never_numbers() {
        assert_eq!(JsonValue::from(f64::NAN).render(), "\"NaN\"");
        assert_eq!(JsonValue::from(f64::INFINITY).render(), "\"inf\"");
        assert_eq!(JsonValue::from(f64::NEG_INFINITY).render(), "\"-inf\"");
        assert_eq!(JsonValue::from(0.5f64).render(), "0.5");
        assert_eq!(JsonValue::from(1e-7f64).render(), "1e-7");
        assert_eq!(JsonValue::from(-2.0f32).render(), "-2.0");
        assert_eq!(JsonValue::from(-3i64).render(), "-3");
        // A NaN inside a container is still a string, never a bare token.
        assert_eq!(
            JsonValue::from(vec![f64::NAN, 1.0]).render(),
            "[\"NaN\",1.0]"
        );
    }

    #[test]
    fn integers_are_written_exactly_at_their_extremes() {
        assert_eq!(JsonValue::from(u64::MAX).render(), "18446744073709551615");
        assert_eq!(JsonValue::from(i64::MIN).render(), "-9223372036854775808");
        assert_eq!(JsonValue::from(i32::MIN).render(), "-2147483648");
        assert_eq!(JsonValue::from(0i32).render(), "0");
        assert_eq!(JsonValue::from(7usize).render(), "7");
    }

    #[test]
    fn signed_zero_and_whole_floats_keep_their_point() {
        assert_eq!(JsonValue::from(-0.0f64).render(), "-0.0");
        assert_eq!(JsonValue::from(1.0f64).render(), "1.0");
        assert_eq!(JsonValue::from(0.1f64).render(), "0.1");
    }

    #[test]
    fn every_finite_float_reads_back_as_the_same_f64() {
        for x in [
            0.0f64,
            -0.0,
            1.0,
            -1.5,
            0.1,
            1e-7,
            1e16,
            1e300,
            -1e-300,
            5e-324,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
            f64::EPSILON,
        ] {
            let text = JsonValue::from(x).render();
            match parse_json(&text).unwrap() {
                JsonValue::Number(JsonNumber::F64(y)) => {
                    assert_eq!(y.to_bits(), x.to_bits(), "{x:e} wrote {text}")
                }
                other => panic!("{x:e} wrote {text}, which reads as {other:?}"),
            }
        }
    }

    #[test]
    fn the_writer_reproduces_compact_text_the_reader_parsed() {
        let text = r#"{"a":[1,-2,0.5,"x\ny",null,true,false],"b":{"c":18446744073709551615,"d":-9223372036854775807}}"#;
        assert_eq!(parse_json(text).unwrap().render(), text);
    }
}
