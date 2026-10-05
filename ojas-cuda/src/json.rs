//! A minimal JSON writer for the rung-0 report (no serde: no new dependency).
//!
//! Objects keep insertion order, so a report's text is a function of its
//! content. JSON has no NaN or infinity: a non-finite float is written as
//! the string `"NaN"`, `"inf"` or `"-inf"`, never as a number, so a reader
//! cannot mistake it for a measured value.

/// A JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` / `false`.
    Bool(bool),
    /// A signed integer.
    Int(i64),
    /// An unsigned integer.
    UInt(u64),
    /// A float; non-finite values are written as strings.
    Float(f64),
    /// A string.
    Str(String),
    /// An array.
    Arr(Vec<Json>),
    /// An object, in insertion order.
    Obj(Vec<(String, Json)>),
}

/// An object under construction; becomes a [`Json::Obj`] through `From`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JsonObj(Vec<(String, Json)>);

impl JsonObj {
    /// An empty object.
    pub fn new() -> Self {
        JsonObj(Vec::new())
    }

    /// Append `key: value`.
    pub fn with(mut self, key: &str, value: impl Into<Json>) -> Self {
        self.0.push((key.to_string(), value.into()));
        self
    }

    /// Append `key: value` in place.
    pub fn push(&mut self, key: &str, value: impl Into<Json>) {
        self.0.push((key.to_string(), value.into()));
    }
}

impl From<JsonObj> for Json {
    fn from(o: JsonObj) -> Self {
        Json::Obj(o.0)
    }
}

impl Json {
    /// The compact text.
    pub fn render(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => out.push_str(&i.to_string()),
            Json::UInt(u) => out.push_str(&u.to_string()),
            Json::Float(x) => {
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
            Json::Str(s) => write_str(s, out),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Json::Obj(fields) => {
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

impl From<bool> for Json {
    fn from(b: bool) -> Self {
        Json::Bool(b)
    }
}
impl From<i32> for Json {
    fn from(i: i32) -> Self {
        Json::Int(i64::from(i))
    }
}
impl From<i64> for Json {
    fn from(i: i64) -> Self {
        Json::Int(i)
    }
}
impl From<u32> for Json {
    fn from(u: u32) -> Self {
        Json::UInt(u64::from(u))
    }
}
impl From<u64> for Json {
    fn from(u: u64) -> Self {
        Json::UInt(u)
    }
}
impl From<usize> for Json {
    fn from(u: usize) -> Self {
        Json::UInt(u64::try_from(u).unwrap_or(u64::MAX))
    }
}
impl From<f64> for Json {
    fn from(x: f64) -> Self {
        Json::Float(x)
    }
}
impl From<f32> for Json {
    fn from(x: f32) -> Self {
        Json::Float(f64::from(x))
    }
}
impl From<&str> for Json {
    fn from(s: &str) -> Self {
        Json::Str(s.to_string())
    }
}
impl From<String> for Json {
    fn from(s: String) -> Self {
        Json::Str(s)
    }
}
impl<T: Into<Json>> From<Option<T>> for Json {
    fn from(o: Option<T>) -> Self {
        o.map_or(Json::Null, Into::into)
    }
}
impl<T: Into<Json>> From<Vec<T>> for Json {
    fn from(v: Vec<T>) -> Self {
        Json::Arr(v.into_iter().map(Into::into).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            Json::from(j).render(),
            r#"{"b":1,"a":["x","y"],"c":{"ok":true,"none":null}}"#
        );
    }

    #[test]
    fn strings_are_escaped() {
        let j = Json::from("a\"b\\c\nd\u{1}é");
        assert_eq!(j.render(), "\"a\\\"b\\\\c\\nd\\u0001é\"");
    }

    #[test]
    fn non_finite_floats_are_strings_never_numbers() {
        assert_eq!(Json::from(f64::NAN).render(), "\"NaN\"");
        assert_eq!(Json::from(f64::INFINITY).render(), "\"inf\"");
        assert_eq!(Json::from(f64::NEG_INFINITY).render(), "\"-inf\"");
        assert_eq!(Json::from(0.5f64).render(), "0.5");
        assert_eq!(Json::from(1e-7f64).render(), "1e-7");
        assert_eq!(Json::from(-2.0f32).render(), "-2.0");
        assert_eq!(Json::Int(-3).render(), "-3");
    }
}
