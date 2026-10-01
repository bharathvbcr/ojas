//! Python's `json.dumps`, byte for byte, over the subset of JSON the shard contract hashes.
//!
//! Two hashes this crate must re-derive are sha256 over text that Python's `json` module
//! wrote: `data_snapshot_hash` is over `qd_data.schema.canonical_json(manifest.hashed_body())`
//! (`sort_keys=True, separators=(",", ":"), ensure_ascii=False`), and `ShardHeader.shard_hash`
//! folds in `json.dumps(..., sort_keys=True)` (default separators `", "`/`": "`,
//! `ensure_ascii=True`). A re-derivation that formats one float or one escape differently
//! refuses a genuine artifact as tampered, so the formatting is ported rather than
//! approximated with `serde_json::to_string`, which differs on both counts (`1e-5` against
//! Python's `1e-05`; no `\u` escaping of non-ASCII).
//!
//! **What cannot round-trip, and is refused rather than guessed:** an integer outside
//! `i64`/`u64` (serde_json without `arbitrary_precision` reads it as a float, Python keeps
//! it exact) and a non-finite float (Python writes `NaN`; serde_json cannot hold one). Both
//! are absent from every hashed body this crate reads; meeting one is an error, never a
//! silently different digest.

use serde_json::Value;
use thiserror::Error;

/// How `json.dumps` was called.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DumpOptions {
    /// `sort_keys=True`.
    pub sort_keys: bool,
    /// The first element of `separators`.
    pub item_separator: &'static str,
    /// The second element of `separators`.
    pub key_separator: &'static str,
    /// `ensure_ascii`.
    pub ensure_ascii: bool,
}

/// `qd_data.schema.canonical_json`: `sort_keys=True, separators=(",", ":"), ensure_ascii=False`.
pub const CANONICAL: DumpOptions = DumpOptions {
    sort_keys: true,
    item_separator: ",",
    key_separator: ":",
    ensure_ascii: false,
};

/// `json.dumps(obj, sort_keys=True)`: default separators and `ensure_ascii=True`.
pub const SORTED_DEFAULT: DumpOptions = DumpOptions {
    sort_keys: true,
    item_separator: ", ",
    key_separator: ": ",
    ensure_ascii: true,
};

/// A value Python's `json.dumps` would format in a way this port cannot reproduce exactly.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PyJsonError {
    /// A number serde_json holds as neither `i64`, `u64` nor a finite `f64`.
    #[error("number {0} has no exact Python json.dumps rendering in this port")]
    UnrepresentableNumber(String),
}

/// `json.dumps(value, ...)` with the given options.
pub fn dumps(value: &Value, opts: DumpOptions) -> Result<String, PyJsonError> {
    let mut out = String::new();
    write_value(&mut out, value, opts)?;
    Ok(out)
}

fn write_value(out: &mut String, value: &Value, opts: DumpOptions) -> Result<(), PyJsonError> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                match n.as_f64() {
                    Some(f) if f.is_finite() => out.push_str(&float_repr(f)),
                    _ => return Err(PyJsonError::UnrepresentableNumber(n.to_string())),
                }
            }
        }
        Value::String(s) => write_string(out, s, opts.ensure_ascii),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(opts.item_separator);
                }
                write_value(out, item, opts)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            // Sorted here rather than trusting the map's iteration order: serde_json's `Map`
            // is a BTreeMap unless some crate in the build enables `preserve_order`, and
            // feature unification makes that a property of the whole workspace, not of this
            // crate. Python sorts str keys by code point; Rust's `str` order is UTF-8 byte
            // order, which is the same order.
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if opts.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (i, (k, v)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(opts.item_separator);
                }
                write_string(out, k, opts.ensure_ascii);
                out.push_str(opts.key_separator);
                write_value(out, v, opts)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// `json.encoder.py_encode_basestring` (`ensure_ascii=False`) or
/// `py_encode_basestring_ascii` (`ensure_ascii=True`).
fn write_string(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => push_u_escape(out, c as u32),
            // ESCAPE_ASCII is `([\\"]|[^\ -~])`: everything outside space..tilde, which takes
            // DEL (0x7f) along with every non-ASCII code point.
            c if ensure_ascii && !(' '..='~').contains(&c) => {
                let cp = c as u32;
                if cp < 0x1_0000 {
                    push_u_escape(out, cp);
                } else {
                    let v = cp - 0x1_0000;
                    push_u_escape(out, 0xd800 | ((v >> 10) & 0x3ff));
                    push_u_escape(out, 0xdc00 | (v & 0x3ff));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn push_u_escape(out: &mut String, unit: u32) {
    // Python formats with '\\u{0:04x}': lowercase hex, four digits.
    out.push_str(&format!("\\u{unit:04x}"));
}

/// `float.__repr__` for a finite float.
///
/// Python's `repr` (`format_float_short`, mode `'r'`) writes the shortest digit string that
/// round-trips -- which Rust's `{:e}` also produces -- and then chooses the layout: fixed
/// notation when the decimal exponent `e` of the leading digit satisfies `-4 <= e < 16`,
/// scientific otherwise, with a signed exponent of at least two digits (`1e-05`, `1e+16`).
/// A fixed-notation integer keeps a trailing `.0`.
pub fn float_repr(x: f64) -> String {
    debug_assert!(x.is_finite(), "float_repr called on a non-finite value");
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".to_owned()
        } else {
            "0.0".to_owned()
        };
    }
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp_text) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i32 = exp_text.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let point = exp + 1;
        let n = digits.len() as i32;
        let body = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point >= n {
            format!("{}{}.0", digits, "0".repeat((point - n) as usize))
        } else {
            let (int_part, frac_part) = digits.split_at(point as usize);
            format!("{int_part}.{frac_part}")
        };
        format!("{sign}{body}")
    } else {
        let (first, rest) = digits.split_at(1);
        let mantissa = if rest.is_empty() {
            first.to_owned()
        } else {
            format!("{first}.{rest}")
        };
        let exp_sign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exp_sign}{:02}", exp.unsigned_abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_take_pythons_layout_in_both_regimes() {
        let cases: [(f64, &str); 16] = [
            (0.8, "0.8"),
            (0.05, "0.05"),
            (1.0, "1.0"),
            (100000.0, "100000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e16, "1.2345e+16"),
            (1e100, "1e+100"),
            (-2.5, "-2.5"),
            (-0.0, "-0.0"),
            (0.1 + 0.2, "0.30000000000000004"),
            (123456789.123, "123456789.123"),
            (5e-324, "5e-324"),
        ];
        for (x, want) in cases {
            assert_eq!(float_repr(x), want, "repr({x:e})");
        }
    }

    #[test]
    fn canonical_is_compact_sorted_and_keeps_non_ascii() {
        let v = json!({"b": [1, 2.5, "é\n"], "a": {"z": null, "y": true}});
        assert_eq!(
            dumps(&v, CANONICAL).unwrap(),
            "{\"a\":{\"y\":true,\"z\":null},\"b\":[1,2.5,\"é\\n\"]}"
        );
    }

    #[test]
    fn sorted_default_spaces_and_escapes_non_ascii_with_surrogates() {
        let v = json!({"k": "é\u{7f}😀", "a": []});
        assert_eq!(
            dumps(&v, SORTED_DEFAULT).unwrap(),
            "{\"a\": [], \"k\": \"\\u00e9\\u007f\\ud83d\\ude00\"}"
        );
    }

    #[test]
    fn control_characters_use_lowercase_four_digit_escapes() {
        let v = json!("\u{01}\u{1f}\u{08}\u{0c}\t\r\"\\");
        assert_eq!(
            dumps(&v, CANONICAL).unwrap(),
            "\"\\u0001\\u001f\\b\\f\\t\\r\\\"\\\\\""
        );
    }
}
