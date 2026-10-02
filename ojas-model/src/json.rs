//! JSON for the `ojas.spec` and checkpoint config codecs: a string writer,
//! and a flat-object check over the workspace's strict reader
//! ([`ojas_io::parse_json_with`]), which owns the grammar.

use ojas_io::{parse_json_with, IoError, JsonLimits, JsonValue};

/// `s` as a JSON string literal, quotes included.
pub(crate) fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `text` as one flat JSON object, read by the workspace's strict reader
/// ([`ojas_io::parse_json_with`]) under an input cap of `max_bytes`. Every
/// key must be in `keys`; every member must be a string, number or boolean
/// (the shared reader accepts nesting and `null`, a flat schema does not).
/// Duplicate keys, trailing text and malformed numbers are refused by the
/// reader itself. Missing keys and wrong types are refused by the caller's
/// typed `get_*` reads.
pub(crate) fn flat_object(
    text: &str,
    max_bytes: usize,
    keys: &[&str],
) -> Result<JsonValue, IoError> {
    let limits = JsonLimits {
        max_input_bytes: max_bytes,
        ..JsonLimits::DEFAULT
    };
    let root = parse_json_with(text, &limits)?;
    root.deny_unknown_keys(keys)?;
    if let Some((key, value)) = root.as_object()?.iter().find(|(_, v)| {
        matches!(
            v,
            JsonValue::Null | JsonValue::Array(_) | JsonValue::Object(_)
        )
    }) {
        return Err(IoError::new(format!(
            "{key:?} is {}; a flat object holds strings, numbers and booleans",
            value.kind()
        )));
    }
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_strings_parse_back() {
        let bs = char::from(0x5c_u8);
        for s in [
            "",
            "plain",
            "q\"b",
            "back\\slash",
            "nl\nt\tcr\r",
            "\u{1}\u{1f}",
            "é😀",
        ] {
            let text = format!("{{{}:{}}}", quote("k"), quote(s));
            let root = flat_object(&text, 1 << 10, &["k"]).unwrap();
            assert_eq!(root.get_str("k").unwrap(), s, "{s:?}");
        }
        assert!(quote("\u{1}").contains(bs));
    }

    #[test]
    fn a_flat_object_refuses_nesting_null_unknown_keys_and_oversize() {
        let keys = ["a", "b"];
        assert!(flat_object(r#"{"a": 1, "b": "x"}"#, 64, &keys).is_ok());
        assert!(flat_object(r#"{"a": true}"#, 64, &keys).is_ok());
        for bad in [
            r#"{"a": [1]}"#,
            r#"{"a": {"b": 1}}"#,
            r#"{"a": null}"#,
            r#"{"c": 1}"#,
            r#"{"a": 1, "a": 1}"#,
            r#"[1]"#,
            r#""a""#,
            r#"{"a": 1} x"#,
        ] {
            assert!(flat_object(bad, 64, &keys).is_err(), "{bad}");
        }
        let long = format!(r#"{{"a": "{}"}}"#, "x".repeat(64));
        assert!(flat_object(&long, 64, &keys).is_err());
        assert!(flat_object(&long, 128, &keys).is_ok());
    }
}
