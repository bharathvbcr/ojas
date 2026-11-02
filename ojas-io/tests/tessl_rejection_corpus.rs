//! The rejection corpus tessl's reader held, now held here.
//!
//! ojas-io owns safetensors IO (owner, 2026-11-01), so tessl's reader becomes a
//! thin adapter over this one. Everything tessl's own suite refused has to go
//! on being refused by the reader underneath it, or the dedup quietly widened
//! what a checkpoint may be. This file is that bar: the malformed inputs from
//! `tessl/tests/safetensors.rs`, asserted against ojas-io directly.
//!
//! Each case carries the needle ojas-io's own message must contain. Where that
//! differs from tessl's needle the difference is wording, not a different
//! refusal point; the tessl-side partition is recorded in the commit that
//! reduces tessl's reader to the adapter.
//!
//! One input tessl refuses that this reader takes is a deliberate capability
//! difference, pinned on its own below: sub-byte dtypes. tessl declines them
//! because a whole-byte range cannot be stated for them; ojas-io sizes them
//! from `bits()` and refuses any span that is not bit-exact. tessl keeps the
//! refusal in its own `Dtype`, so this reader supporting them does not widen
//! what tessl accepts.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use ojas_io::SafeTensors;

static SEQ: AtomicUsize = AtomicUsize::new(0);

fn tmp(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "ojasio-parity-{}-{tag}-{n}.safetensors",
        std::process::id()
    ))
}

fn raw(tag: &str, len8: u64, header: &[u8], data: &[u8]) -> PathBuf {
    let p = tmp(tag);
    let mut b = len8.to_le_bytes().to_vec();
    b.extend_from_slice(header);
    b.extend_from_slice(data);
    std::fs::write(&p, b).unwrap();
    p
}

fn file(tag: &str, header: &str, data: &[u8]) -> PathBuf {
    raw(tag, header.len() as u64, header.as_bytes(), data)
}

/// Open `p` both ways and return (accepted, message, agreement note).
///
/// tessl's own `open_err` asserts that the path and in-memory routes refuse the
/// same bytes with the same message, because the two share one validation.
/// That holds here too, except for a file shorter than the 8-byte length field,
/// where each route can only name what it has.
fn probe_path(p: PathBuf) -> (bool, String, String) {
    let opened = SafeTensors::open(&p);
    let bytes = std::fs::read(&p).unwrap();
    let _ = std::fs::remove_file(&p);
    let parsed = SafeTensors::parse(&bytes);
    let note = match (&opened, &parsed) {
        (Ok(_), Ok(_)) => "both-ok".to_string(),
        (Err(a), Err(b)) => {
            // `open` prefixes the path; compare the detail tails.
            let (a, b) = (a.detail().to_string(), b.detail().to_string());
            if a.ends_with(&b) || b.ends_with(&a) || a == b {
                "agree".to_string()
            } else {
                format!("DISAGREE open={a:?} parse={b:?}")
            }
        }
        (Ok(_), Err(b)) => format!("SPLIT open=Ok parse={:?}", b.detail()),
        (Err(a), Ok(_)) => format!("SPLIT open={:?} parse=Ok", a.detail()),
    };
    match opened {
        Ok(_) => (true, String::new(), note),
        Err(e) => (false, e.detail().to_string(), note),
    }
}

/// Assert `p` is refused, that the message contains `needle`, and that both
/// routes agree on it.
fn expect_rejected(tag: &str, needle: &str, p: PathBuf) {
    let (accepted, msg, note) = probe_path(p);
    assert!(!accepted, "{tag}: opened, expected a refusal");
    assert!(
        msg.contains(needle),
        "{tag}: error {msg:?} does not mention {needle:?}"
    );
    assert!(
        !note.starts_with("DISAGREE") && !note.starts_with("SPLIT"),
        "{tag}: {note}"
    );
}

const ONE_F32: &str = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;

/// Headers outside the subset the format uses. Needles are ojas-io's; the
/// comment marks the five where tessl worded the same refusal differently.
#[test]
fn json_outside_the_formats_subset_is_rejected() {
    let d = 1.0f32.to_le_bytes();
    let cases: &[(&str, &str, &str)] = &[
        ("not_obj", r#"[1]"#, "not a JSON object"),
        ("trailing", r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}x"#, "trailing bytes"),
        ("float", r#"{"a":{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}}"#, "non-integer"),
        ("exp", r#"{"a":{"dtype":"F32","shape":[1e0],"data_offsets":[0,4]}}"#, "non-integer"),
        // tessl: "unsupported JSON value"
        ("neg", r#"{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#, "non-integer"),
        ("lead0", r#"{"a":{"dtype":"F32","shape":[01],"data_offsets":[0,4]}}"#, "leading zero"),
        // tessl: "unsupported JSON value"
        ("null", r#"{"a":null}"#, "entry is not an object"),
        // tessl: "unsupported JSON value"
        ("bool", r#"{"a":true}"#, "entry is not an object"),
        ("u64", r#"{"a":{"dtype":"F32","shape":[18446744073709551616],"data_offsets":[0,4]}}"#, "overflows u64"),
        ("dup", r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, "duplicate key"),
        ("dup_field", r#"{"a":{"dtype":"F32","dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, "duplicate key"),
        ("deep", r#"{"a":{"dtype":"F32","shape":[[1]],"data_offsets":[0,4]}}"#, "nesting deeper"),
        ("unterminated", r#"{"a":{"dtype":"F32"#, "unterminated string"),
        ("bad_escape", r#"{"a\q":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, "invalid escape"),
        ("plus_hex", r#"{"a\u+041":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, "bad \\u escape"),
        ("lone_hi", r#"{"a\ud800":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, "unpaired high surrogate"),
        ("lone_lo", r#"{"a\udc00":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, "unpaired low surrogate"),
        ("hi_then_char", r#"{"a\ud800A":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, "unpaired high surrogate"),
        ("ctrl", "{\"a\tb\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", "control character"),
        ("no_colon", r#"{"a" {"dtype":"F32"}}"#, "expected ':'"),
        ("no_comma", r#"{"a":{"dtype":"F32" "shape":[1]}}"#, "expected ',' or '}'"),
        ("arr_comma", r#"{"a":{"dtype":"F32","shape":[1 2],"data_offsets":[0,4]}}"#, "expected ',' or ']'"),
        ("key", r#"{1:2}"#, "expected a string key"),
    ];
    for (tag, h, needle) in cases {
        expect_rejected(tag, needle, file(tag, h, &d));
    }
    // A high surrogate followed by an escape that is not a low surrogate.
    // Built from parts so the second escape stays a six-byte escape here.
    let bs = '\\';
    let bad_lo = format!(
        "{{\"a{bs}ud800{bs}u0041\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}}}"
    );
    assert!(bad_lo.contains("\\u0041"), "{bad_lo}");
    expect_rejected("bad_lo", "invalid low surrogate", file("bad_lo", &bad_lo, &d));

    // Invalid UTF-8 in the header.
    let bad = b"{\"\xff\":1}";
    expect_rejected("utf8", "not UTF-8", raw("utf8", bad.len() as u64, bad, &[]));
}

/// A tensor entry holds exactly `dtype`, `shape` and `data_offsets`.
#[test]
fn entries_must_be_exactly_dtype_shape_and_offsets() {
    let d = 1.0f32.to_le_bytes();
    let cases: &[(&str, &str, &str)] = &[
        ("unknown_field", r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"x":"y"}}"#, "unexpected or mistyped field \"x\""),
        ("missing", r#"{"a":{"dtype":"F32","shape":[1]}}"#, "needs dtype, shape and data_offsets"),
        ("dtype_type", r#"{"a":{"dtype":4,"shape":[1],"data_offsets":[0,4]}}"#, "mistyped field \"dtype\""),
        ("shape_type", r#"{"a":{"dtype":"F32","shape":"1","data_offsets":[0,4]}}"#, "mistyped field \"shape\""),
        ("shape_str", r#"{"a":{"dtype":"F32","shape":["1"],"data_offsets":[0,4]}}"#, "non-integer"),
        ("offs3", r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4,4]}}"#, "not [begin, end]"),
        // tessl: "unsupported dtype"
        ("dtype_name", r#"{"a":{"dtype":"F8","shape":[1],"data_offsets":[0,4]}}"#, "unknown dtype \"F8\""),
        ("entry_type", r#"{"a":"F32"}"#, "not an object"),
        ("meta_type", r#"{"__metadata__":[1]}"#, "__metadata__ is not an object"),
        ("meta_value", r#"{"__metadata__":{"k":1}}"#, "not a string"),
    ];
    for (tag, h, needle) in cases {
        expect_rejected(tag, needle, file(tag, h, &d));
    }
}

/// Offsets must match the shape and tile the data buffer exactly.
#[test]
fn offsets_must_match_the_shape_and_tile_the_data_exactly() {
    let d8 = [0u8; 8];
    let cases: &[(&str, &str, usize, &str)] = &[
        ("reversed", r#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[4,0]}}"#, 4, "outside"),
        ("past", r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#, 4, "outside"),
        ("size", r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#, 4, "shape [2] x F32 is 8"),
        // tessl: "no gap or overlap"
        ("overlap", r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}"#, 6, "gap or overlap"),
        ("gap", r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[2],"data_offsets":[6,8]}}"#, 8, "gap or overlap"),
        ("unindexed", ONE_F32, 8, "cover 4 of 8"),
        // tessl: "element count overflows"
        ("numel", r#"{"a":{"dtype":"U8","shape":[4294967296,4294967296],"data_offsets":[0,0]}}"#, 0, "shape product overflows"),
        ("bytes", r#"{"a":{"dtype":"F64","shape":[2305843009213693952,2],"data_offsets":[0,0]}}"#, 0, "overflows"),
    ];
    for (tag, h, n, needle) in cases {
        expect_rejected(tag, needle, file(tag, h, &d8[..*n]));
    }
}

/// The 8-byte header length is bounded by the file and by the cap.
#[test]
fn header_length_is_bounded_by_the_file_and_the_cap() {
    let h = ONE_F32.as_bytes();
    let d = 1.0f32.to_le_bytes();
    expect_rejected("zero", "outside 1..=", raw("zero", 0, h, &d));
    expect_rejected(
        "cap",
        "outside 1..=",
        raw("cap", ojas_io::MAX_HEADER_BYTES + 1, h, &d),
    );
    expect_rejected("huge", "outside 1..=", raw("huge", u64::MAX, h, &d));
    let past = h.len() as u64 + d.len() as u64 + 1;
    expect_rejected("past_end", "past the end", raw("past_end", past, h, &d));
    // One byte short of the header: the JSON is cut and fails to parse.
    // tessl: "header JSON".
    expect_rejected(
        "cut",
        "expected ',' or '}'",
        raw("cut", h.len() as u64 - 1, h, &d),
    );

    // Shorter than the length field itself. Each route can only name what it
    // has, so this is the one case where the two messages differ; tessl's own
    // `open_err` makes the same exception.
    let p = tmp("short");
    std::fs::write(&p, [1u8, 0, 0]).unwrap();
    let (accepted, msg, _) = probe_path(p);
    assert!(!accepted, "short: opened, expected a refusal");
    assert!(msg.contains("header length"), "short: {msg:?}");
}

/// The one input tessl refuses that this reader takes, and why that is a
/// capability difference rather than a hole.
///
/// `F4` at shape [2] is 2 x 4 bits = exactly one byte, so the span is
/// bit-exact and ojas-io accepts it. A span that is not bit-exact is still
/// refused, which is what makes this safe to support:
/// `ojas-io/tests/unknown_dtype.rs` pins `F4` shape [3] in 2 bytes (12 bits,
/// not a whole number of bytes). tessl's `Dtype` has no F4/F6E2M3/F6E3M2
/// variant and its conversion from `StDtype` errors on all three, so tessl
/// goes on refusing them whatever this reader can do.
#[test]
fn the_only_input_tessl_refused_that_this_reader_takes_is_a_bit_exact_sub_byte_span() {
    // Bit-exact: accepted here.
    let h = r#"{"a":{"dtype":"F4","shape":[2],"data_offsets":[0,1]}}"#;
    let p = file("subbyte_ok", h, &[0]);
    let st = SafeTensors::open(&p).expect("a bit-exact F4 span is readable");
    assert_eq!(st.info("a").unwrap().shape, vec![2]);
    let _ = std::fs::remove_file(&p);

    // Not bit-exact: still refused, for every sub-byte dtype.
    for tag in ["F4", "F6_E2M3", "F6_E3M2"] {
        let h = format!(r#"{{"a":{{"dtype":"{tag}","shape":[3],"data_offsets":[0,2]}}}}"#);
        expect_rejected(tag, "is not a whole number of bytes", file(tag, &h, &[0, 0]));
    }
}
