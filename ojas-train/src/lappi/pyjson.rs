//! The exact bytes Python writes, where a hash is taken over them.
//!
//! Three things the Rust trainer writes are hashed by Python code that will read them back:
//!
//! * `LossLog.digest` (`python/qd_train/run_control.py:756`) is sha256 over
//!   `json.dumps([...], sort_keys=True, separators=(",", ":"))` of points whose loss is
//!   `float.hex()`;
//! * `Protocol.recipe_hash` (`tools/real_ft_run.py:1892`) is sha256 over the same
//!   `json.dumps` of the recipe, whose floats are `repr` (`1e-05`, `1.0`);
//! * a ledger line (`python/qd_train/ledger.py:475`, `_canonical`) is the same with
//!   `ensure_ascii=False`, and the next row's `prev_row_hash` is sha256 over its bytes.
//!
//! `serde_json` writes `1e-5` and `1.0` as `1e-5` and `1.0` but `1e16` as `1e16`, sorts keys
//! only when no crate in the build enables `preserve_order`, and has no `float.hex`. So the
//! three are written here, by hand, against Python's rules, and `tests/pyjson_oracle.rs` checks
//! them byte for byte against what `tools/qd_train_oracle_trainer.py` dumped from Python itself.
//!
//! Non-finite floats are refused: Python's `json.dumps` would write `NaN`, which is not JSON,
//! and a ledger row that carries one cannot be read by a strict reader.

use std::collections::BTreeMap;

/// A JSON value as Python's `json` module writes it. Keys are kept sorted by construction
/// (`BTreeMap` orders `String`s by their UTF-8 bytes, which is code-point order, which is
/// what `sort_keys=True` sorts by).
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

/// Why a value could not be written as Python would write it.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct PyJsonError(pub String);

impl Json {
    /// An object from `(key, value)` pairs. A repeated key is refused rather than letting the
    /// later one win silently.
    pub fn obj<K: Into<String>>(pairs: impl IntoIterator<Item = (K, Json)>) -> Result<Json, PyJsonError> {
        let mut out = BTreeMap::new();
        for (k, v) in pairs {
            let k = k.into();
            if out.insert(k.clone(), v).is_some() {
                return Err(PyJsonError(format!("key {k:?} given twice")));
            }
        }
        Ok(Json::Obj(out))
    }

    pub fn str(s: impl Into<String>) -> Json {
        Json::Str(s.into())
    }

    /// `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=...)`.
    pub fn dumps(&self, ensure_ascii: bool) -> Result<String, PyJsonError> {
        let mut out = String::new();
        self.write(&mut out, ensure_ascii)?;
        Ok(out)
    }

    fn write(&self, out: &mut String, ensure_ascii: bool) -> Result<(), PyJsonError> {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => out.push_str(&i.to_string()),
            Json::Float(x) => out.push_str(&float_repr(*x)?),
            Json::Str(s) => write_str(out, s, ensure_ascii),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out, ensure_ascii)?;
                }
                out.push(']');
            }
            Json::Obj(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_str(out, k, ensure_ascii);
                    out.push(':');
                    v.write(out, ensure_ascii)?;
                }
                out.push('}');
            }
        }
        Ok(())
    }
}

/// Python's string escaping (`json.encoder`'s `ESCAPE` / `ESCAPE_ASCII` tables): `"` and `\`,
/// the five short escapes, every other control character as `\u00XX`; with `ensure_ascii`,
/// every non-ASCII code point as `\uXXXX` (astral ones as a surrogate pair).
fn write_str(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `repr(x)` for a finite float, as CPython writes it (`float_repr_style == "short"`).
///
/// The digits are the shortest that round-trip, which Rust's `{:e}` also produces; the
/// layout is CPython's `format_float_short` in mode `r`: scientific when the decimal point
/// position `decpt` is `<= -4` or `> 16`, with a sign and at least two exponent digits;
/// fixed otherwise, with `.0` when there are no fractional digits.
pub fn float_repr(x: f64) -> Result<String, PyJsonError> {
    if !x.is_finite() {
        return Err(PyJsonError(format!(
            "{x} is not finite; Python would write it as a bare NaN/Infinity, which is not JSON"
        )));
    }
    if x == 0.0 {
        return Ok(if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string());
    }
    let sign = if x < 0.0 { "-" } else { "" };
    // Rust's `{:e}` gives a shortest round-tripping digit string, but when two strings of that
    // length both round-trip it may not pick the one CPython's dtoa (mode 0) picks: the one
    // nearest the exact value, ties to even. 852875973836655.25 is such a tie: `{:e}` writes
    // ...655.3, `repr` ...655.2. The correctly rounded string of the same length is dtoa's
    // answer whenever it round-trips, so it is taken when it does.
    let shortest = format!("{:e}", x.abs());
    let n = shortest.split_once('e').map_or(0, |(m, _)| m.chars().filter(char::is_ascii_digit).count());
    let exact = format!("{:.*e}", n.saturating_sub(1), x.abs());
    let sci = if exact.parse::<f64>().map(f64::to_bits) == Ok(x.abs().to_bits()) {
        exact
    } else {
        shortest
    };
    let (mantissa, exp) = sci
        .split_once('e')
        .ok_or_else(|| PyJsonError(format!("{sci}: no exponent in Rust's {{:e}}")))?;
    let exp: i32 = exp.parse().map_err(|e| PyJsonError(format!("{sci}: {e}")))?;
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1;
    let n = i32::try_from(digits.len()).map_err(|e| PyJsonError(e.to_string()))?;
    let body = if decpt <= -4 || decpt > 16 {
        let (head, rest) = digits.split_at(1);
        let e = decpt - 1;
        let esign = if e < 0 { '-' } else { '+' };
        if rest.is_empty() {
            format!("{head}e{esign}{:02}", e.abs())
        } else {
            format!("{head}.{rest}e{esign}{:02}", e.abs())
        }
    } else if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat(usize::try_from(-decpt).unwrap_or(0)))
    } else if decpt >= n {
        format!("{digits}{}.0", "0".repeat(usize::try_from(decpt - n).unwrap_or(0)))
    } else {
        let (int, frac) = digits.split_at(usize::try_from(decpt).unwrap_or(0));
        format!("{int}.{frac}")
    };
    Ok(format!("{sign}{body}"))
}

/// `float.hex(x)` for a finite float: `0x1.<13 hex digits>p<signed exponent>` for a normal
/// number, `0x0.<13>p-1022` for a subnormal, `0x0.0p+0` for zero, with a leading `-` for a
/// negative sign (CPython `float_hex`, `TOHEX_NBITS = 53`).
pub fn float_hex(x: f64) -> Result<String, PyJsonError> {
    if !x.is_finite() {
        return Err(PyJsonError(format!("{x} is not finite; float.hex would write inf/nan")));
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return Ok(format!("{sign}0x0.0p+0"));
    }
    let bits = x.to_bits();
    let exp_bits = i32::try_from((bits >> 52) & 0x7ff).map_err(|e| PyJsonError(e.to_string()))?;
    let mant = bits & ((1u64 << 52) - 1);
    Ok(if exp_bits == 0 {
        format!("{sign}0x0.{mant:013x}p-1022")
    } else {
        format!("{sign}0x1.{mant:013x}p{:+}", exp_bits - 1023)
    })
}

/// The inverse of [`float_hex`], for exactly the strings it writes (and Python's `float.hex`
/// writes): fixtures carry floats this way so a comparison is of bits, not of renderings.
pub fn float_fromhex(s: &str) -> Result<f64, PyJsonError> {
    let bad = || PyJsonError(format!("{s:?} is not a float.hex() string"));
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

    #[test]
    fn repr_follows_cpythons_switch_between_fixed_and_scientific() {
        let cases = [
            (1e-5, "1e-05"),
            (1e-4, "0.0001"),
            (0.1, "0.1"),
            (1.0, "1.0"),
            (35403.0, "35403.0"),
            (32400.0, "32400.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e300, "1.5e+300"),
            (-2.5e-7, "-2.5e-07"),
            (0.8353729248046875, "0.8353729248046875"),
            (591.820476162, "591.820476162"),
            (123456789012345678.0, "1.2345678901234568e+17"),
            (5e-324, "5e-324"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
        ];
        for (x, want) in cases {
            assert_eq!(float_repr(x).unwrap(), want, "repr({x:e})");
        }
        assert!(float_repr(f64::NAN).is_err());
        assert!(float_repr(f64::INFINITY).is_err());
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
    fn dumps_sorts_keys_and_escapes_like_python() {
        let v = Json::obj([
            ("b", Json::Float(1e-5)),
            ("a", Json::Arr(vec![Json::Int(1), Json::Null, Json::Bool(true)])),
            ("é", Json::str("tab\there \"q\" \u{1}")),
        ])
        .unwrap();
        assert_eq!(
            v.dumps(false).unwrap(),
            "{\"a\":[1,null,true],\"b\":1e-05,\"é\":\"tab\\there \\\"q\\\" \\u0001\"}"
        );
        assert_eq!(
            v.dumps(true).unwrap(),
            "{\"a\":[1,null,true],\"b\":1e-05,\"\\u00e9\":\"tab\\there \\\"q\\\" \\u0001\"}"
        );
        assert!(Json::obj([("k", Json::Null), ("k", Json::Null)]).is_err());
    }
}
