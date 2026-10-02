//! The fixture safetensors reader: what it accepts and what it refuses.

mod support;

use ojas_oracle::safetensors::{SafeTensors, StDtype, MAX_HEADER_BYTES};
use support::{build, build_with_header, f32_raw, json_str, Raw};

#[test]
fn round_trips_values_metadata_and_escapes() {
    let meta_value = "{\"a\":\"q\\\\uote\"} tab\t é \u{1F600}";
    let bytes = build(
        &[
            f32_raw("w", &[2, 2], &[1.0, -2.5, 0.0, f32::MIN_POSITIVE]),
            Raw {
                name: "ids".into(),
                dtype: "I64",
                shape: vec![2],
                bytes: [7i64, -3].iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        ],
        &[("k", meta_value)],
    );
    let st = SafeTensors::parse(&bytes).unwrap();
    assert_eq!(
        st.f32s("w").unwrap(),
        vec![1.0, -2.5, 0.0, f32::MIN_POSITIVE]
    );
    assert_eq!(st.i64s("ids").unwrap(), vec![7, -3]);
    assert_eq!(st.entry("w").unwrap().dtype, StDtype::F32);
    assert_eq!(st.metadata("k"), Some(meta_value));
    assert!(st.f32s("ids").is_err(), "dtype is checked on read");
    assert!(st.f32s("missing").is_err());
}

#[test]
fn unicode_escapes_and_surrogate_pairs_decode() {
    let header = r#"{"__metadata__":{"k":"é😀\/\n"}}"#;
    let bytes = build_with_header(header, &[]);
    let st = SafeTensors::parse(&bytes).unwrap();
    assert_eq!(st.metadata("k"), Some("é\u{1F600}/\n"));
    for bad in [
        r#""\ud83d""#,
        r#""\ude00""#,
        r#""\u12""#,
        r#""\x41""#,
        "\"a\u{1}\"",
    ] {
        let header = format!(r#"{{"__metadata__":{{"k":{bad}}}}}"#);
        assert!(
            SafeTensors::parse(&build_with_header(&header, &[])).is_err(),
            "{bad}"
        );
    }
}

fn one(dtype: &str, shape: &str, offsets: &str, data_len: usize) -> Vec<u8> {
    let header =
        format!(r#"{{"t":{{"dtype":"{dtype}","shape":{shape},"data_offsets":{offsets}}}}}"#);
    build_with_header(&header, &vec![0u8; data_len])
}

#[test]
fn malformed_layouts_are_refused() {
    assert!(SafeTensors::parse(&one("F32", "[2]", "[0,8]", 8)).is_ok());
    let cases = [
        ("unsupported dtype", one("BF16", "[2]", "[0,4]", 4)),
        ("size disagrees with shape", one("F32", "[2]", "[0,4]", 4)),
        ("hole before the tensor", one("F32", "[1]", "[4,8]", 8)),
        ("bytes after the last tensor", one("F32", "[1]", "[0,4]", 8)),
        ("offsets past the buffer", one("F32", "[2]", "[0,8]", 4)),
        ("reversed offsets", one("F32", "[0]", "[4,0]", 4)),
        ("three offsets", one("F32", "[1]", "[0,4,4]", 4)),
        ("fractional shape", one("F32", "[1.5]", "[0,4]", 4)),
        ("negative shape", one("F32", "[-1]", "[0,4]", 4)),
        (
            "overflowing shape",
            one("I64", "[4294967296,4294967296]", "[0,8]", 8),
        ),
    ];
    for (what, bytes) in cases {
        assert!(SafeTensors::parse(&bytes).is_err(), "{what}");
    }
    let overlap = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
    assert!(SafeTensors::parse(&build_with_header(overlap, &[0; 8])).is_err());
    let extra_key = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"x":1}}"#;
    assert!(SafeTensors::parse(&build_with_header(extra_key, &[0; 4])).is_err());
    let dup = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
    assert!(SafeTensors::parse(&build_with_header(dup, &[0; 8])).is_err());
    let meta_number = r#"{"__metadata__":{"k":1}}"#;
    assert!(SafeTensors::parse(&build_with_header(meta_number, &[])).is_err());
    let empty_name = r#"{"":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    assert!(SafeTensors::parse(&build_with_header(empty_name, &[0; 4])).is_err());
}

#[test]
fn truncated_and_oversized_headers_are_refused() {
    let good = one("F32", "[1]", "[0,4]", 4);
    for cut in 0..good.len() {
        assert!(SafeTensors::parse(&good[..cut]).is_err(), "cut at {cut}");
    }
    let mut huge = ((MAX_HEADER_BYTES + 1) as u64).to_le_bytes().to_vec();
    huge.extend(vec![b' '; 16]);
    assert!(SafeTensors::parse(&huge).is_err());
    let mut lying = (u64::MAX).to_le_bytes().to_vec();
    lying.extend_from_slice(b"{}");
    assert!(SafeTensors::parse(&lying).is_err());
    assert!(
        SafeTensors::parse(&build_with_header("[]", &[])).is_err(),
        "not an object"
    );
    assert!(
        SafeTensors::parse(&build_with_header("{} x", &[])).is_err(),
        "trailing header bytes"
    );
}

#[test]
fn json_str_helper_escapes_what_the_reader_unescapes() {
    // Guards the helper the other tests rely on.
    assert_eq!(json_str("a\"b\\c\n"), r#""a\"b\\c\u000a""#);
}
