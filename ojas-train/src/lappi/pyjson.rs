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
//!
//! **What the trainer half writes through it.** Three more hashes are taken over text this
//! crate writes and Python reads back: `LossLog.digest` (`python/qd_train/run_control.py`,
//! sha256 over `json.dumps(points, sort_keys=True, separators=(",", ":"))` of points whose
//! loss is `float.hex()`), `Protocol.recipe_hash` (`tools/real_ft_run.py`, the same dumps of
//! the recipe) -- both [`CANONICAL_ASCII`] -- and a ledger line (`python/qd_train/ledger.py`,
//! [`CANONICAL`]), whose bytes the next row's `prev_row_hash` hashes. [`obj`] builds an object
//! that refuses a repeated key, [`float`] a number that refuses a non-finite value, and
//! [`float_hex`] / [`float_fromhex`] carry floats as bits. `tests/pyjson_oracle.rs` checks
//! every one against what `tools/qd_train_oracle_trainer.py` dumped from Python itself.

use serde_json::{Map, Number, Value};
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

/// `json.dumps(obj, sort_keys=True, separators=(",", ":"))`: compact, sorted, and the
/// default `ensure_ascii=True`. What `recipe_hash` and `LossLog.digest` hash.
pub const CANONICAL_ASCII: DumpOptions = DumpOptions {
    sort_keys: true,
    item_separator: ",",
    key_separator: ":",
    ensure_ascii: true,
};

/// A value Python's `json.dumps` would format in a way this port cannot reproduce exactly.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PyJsonError {
    /// A number serde_json holds as neither `i64`, `u64` nor a finite `f64`.
    #[error("number {0} has no exact Python json.dumps rendering in this port")]
    UnrepresentableNumber(String),
    /// A float that is not finite: Python would write a bare `NaN`/`Infinity`, which is not
    /// JSON, and `float.hex` would write `nan`/`inf`.
    #[error("{0} is not finite: Python would write it as a bare NaN/Infinity, which is not JSON")]
    NonFinite(String),
    /// An object built with the same key twice: the later value would win silently.
    #[error("key {0:?} given twice")]
    DuplicateKey(String),
    /// A string that is not the 13-hex-digit form `float.hex()` writes.
    #[error("{0:?} is not a float.hex() string")]
    NotFloatHex(String),
}

/// An object from `(key, value)` pairs, refusing a repeated key rather than letting the later
/// value win silently.
pub fn obj<K: Into<String>>(pairs: impl IntoIterator<Item = (K, Value)>) -> Result<Value, PyJsonError> {
    let mut out = Map::new();
    for (k, v) in pairs {
        let k = k.into();
        if out.contains_key(&k) {
            return Err(PyJsonError::DuplicateKey(k));
        }
        out.insert(k, v);
    }
    Ok(Value::Object(out))
}

/// A float as a JSON number Python reads back as a `float` (serde_json keeps it a float even
/// when it is integral, so `1.0` is written `1.0`, not `1`). Refused when not finite.
pub fn float(x: f64) -> Result<Value, PyJsonError> {
    Number::from_f64(x)
        .map(Value::Number)
        .ok_or_else(|| PyJsonError::NonFinite(format!("{x}")))
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
///
/// **Ties.** When two digit strings of the shortest length both round-trip, CPython's dtoa
/// (mode 0) takes the one nearest the exact binary value, ties to even; Rust's `{:e}` may take
/// the other. 852875973836655.25 is such a tie: `{:e}` gives `...655.3`, `repr` `...655.2`.
/// The correctly rounded string of that length (`{:.*e}`) is dtoa's answer whenever it
/// round-trips, so it is preferred when it does.
pub fn float_repr(x: f64) -> String {
    debug_assert!(x.is_finite(), "float_repr called on a non-finite value");
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".to_owned()
        } else {
            "0.0".to_owned()
        };
    }
    let shortest = format!("{:e}", x.abs());
    let n_digits = shortest
        .split_once('e')
        .map_or(0, |(m, _)| m.chars().filter(char::is_ascii_digit).count());
    let rounded = format!("{:.*e}", n_digits.saturating_sub(1), x.abs());
    let sci = if rounded.parse::<f64>().map(f64::to_bits) == Ok(x.abs().to_bits()) {
        rounded
    } else {
        shortest
    };
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

/// `float.hex(x)` for a finite float: `0x1.<13 hex digits>p<signed exponent>` for a normal
/// number, `0x0.<13>p-1022` for a subnormal, `0x0.0p+0` for zero, with a leading `-` for a
/// negative sign (CPython `float_hex`, `TOHEX_NBITS = 53`).
pub fn float_hex(x: f64) -> Result<String, PyJsonError> {
    if !x.is_finite() {
        return Err(PyJsonError::NonFinite(format!("{x}")));
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return Ok(format!("{sign}0x0.0p+0"));
    }
    let bits = x.to_bits();
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let mant = bits & ((1u64 << 52) - 1);
    Ok(if exp_bits == 0 {
        format!("{sign}0x0.{mant:013x}p-1022")
    } else {
        format!("{sign}0x1.{mant:013x}p{:+}", exp_bits - 1023)
    })
}

/// The inverse of [`float_hex`], for exactly the strings it (and Python's `float.hex`)
/// writes: fixtures carry floats this way so a comparison is of bits, not of renderings.
pub fn float_fromhex(s: &str) -> Result<f64, PyJsonError> {
    let bad = || PyJsonError::NotFloatHex(s.to_owned());
    let (neg, body) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    let body = body.strip_prefix("0x").ok_or_else(bad)?;
    let (mant, exp) = body.split_once('p').ok_or_else(bad)?;
    let (lead, frac) = mant.split_once('.').ok_or_else(bad)?;
    let exp: i32 = exp.parse().map_err(|_| bad())?;
    let value = if lead == "0" && frac == "0" && exp == 0 {
        0.0
    } else {
        if frac.len() != 13 {
            return Err(bad());
        }
        let m = u64::from_str_radix(frac, 16).map_err(|_| bad())?;
        match lead {
            "1" => {
                let e = u64::try_from(exp + 1023).map_err(|_| bad())?;
                if e == 0 || e >= 0x7ff {
                    return Err(bad());
                }
                f64::from_bits((e << 52) | m)
            }
            "0" if exp == -1022 => f64::from_bits(m),
            _ => return Err(bad()),
        }
    };
    Ok(if neg { -value } else { value })
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
    fn a_shortest_digit_tie_is_broken_as_cpythons_dtoa_breaks_it() {
        // 852875973836655.25 is exactly halfway between two 16-digit strings that both
        // round-trip; CPython's dtoa takes the even one (...655.2), Rust's `{:e}` the other.
        // Found by tools/qd_train_oracle_trainer.py's float sweep.
        let x = f64::from_bits(0x4308_3d7d_4ba9_fb7a);
        assert_eq!(x * 4.0, 3_411_503_895_346_621.0, "x is exactly 852875973836655.25");
        assert_eq!(float_repr(x), "852875973836655.2");
        assert_eq!(float_repr(-x), "-852875973836655.2");
    }

    #[test]
    fn hex_is_cpythons_and_round_trips() {
        let cases = [
            (1.0, "0x1.0000000000000p+0"),
            (3.0, "0x1.8000000000000p+1"),
            (0.0, "0x0.0p+0"),
            (-0.0, "-0x0.0p+0"),
            (5e-324, "0x0.0000000000001p-1022"),
            (-0.5, "-0x1.0000000000000p-1"),
            (f64::MAX, "0x1.fffffffffffffp+1023"),
        ];
        for (x, want) in cases {
            let got = float_hex(x).unwrap();
            assert_eq!(got, want);
            assert_eq!(float_fromhex(&got).unwrap().to_bits(), x.to_bits());
        }
        assert!(float_fromhex("0x1.8p+1").is_err(), "only the 13-digit form Python writes");
        assert!(float_hex(f64::NAN).is_err());
    }

    #[test]
    fn canonical_ascii_escapes_and_obj_refuses_a_repeated_key() {
        let v = obj([
            ("b", float(1e-5).unwrap()),
            ("a", json!([1, null, true])),
            ("é", json!("tab\there \"q\" \u{1}")),
            ("f", float(1.0).unwrap()),
        ])
        .unwrap();
        assert_eq!(
            dumps(&v, CANONICAL).unwrap(),
            "{\"a\":[1,null,true],\"b\":1e-05,\"f\":1.0,\"é\":\"tab\\there \\\"q\\\" \\u0001\"}"
        );
        assert_eq!(
            dumps(&v, CANONICAL_ASCII).unwrap(),
            "{\"a\":[1,null,true],\"b\":1e-05,\"f\":1.0,\"\\u00e9\":\"tab\\there \\\"q\\\" \\u0001\"}"
        );
        assert!(obj([("k", Value::Null), ("k", Value::Null)]).is_err());
        assert!(float(f64::NAN).is_err() && float(f64::INFINITY).is_err());
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
