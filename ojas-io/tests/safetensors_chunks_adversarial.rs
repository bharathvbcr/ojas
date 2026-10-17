//! Adversarial attack on the chunked typed reads and the 0.8.0 dtype tags:
//! hostile headers, chunk boundaries, zero-element tensors, near-miss dtype
//! strings, and the C64 width-8 path through `read_bytes` / `read_into`.

use ojas_io::{encode_safetensors, SafeTensors, StDtype, TensorOut};

/// `READ_CHUNK_BYTES` in `ojas-io/src/safetensors.rs` (private there).
const CHUNK: usize = 128 << 10;

fn frame(header: &str, data: &[u8]) -> Vec<u8> {
    let mut h = header.as_bytes().to_vec();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(data);
    out
}

fn one(dtype: &str, shape: &str, begin: u64, end: u64, data: &[u8]) -> Vec<u8> {
    frame(
        &format!(r#"{{"t":{{"dtype":"{dtype}","shape":{shape},"data_offsets":[{begin},{end}]}}}}"#),
        data,
    )
}

// ---- hostile headers -----------------------------------------------------

#[test]
fn offsets_beyond_the_file_reversed_overlapping_or_gapped_are_refused() {
    // beyond the data buffer
    assert!(SafeTensors::parse(&one("F32", "[2]", 0, 8, &[0; 4])).is_err());
    // begin past end
    assert!(SafeTensors::parse(&one("F32", "[0]", 4, 0, &[0; 8])).is_err());
    // a span ending past u64 range of the buffer
    assert!(SafeTensors::parse(&one("F32", "[1]", u64::MAX - 3, u64::MAX, &[0; 4])).is_err());
    // overlapping
    let overlap = frame(
        r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#,
        &[0; 12],
    );
    assert!(SafeTensors::parse(&overlap).is_err());
    // gap at the start
    assert!(SafeTensors::parse(&one("F32", "[1]", 4, 8, &[0; 8])).is_err());
    // unindexed tail
    assert!(SafeTensors::parse(&one("F32", "[1]", 0, 4, &[0; 8])).is_err());
}

#[test]
fn a_span_that_is_not_a_multiple_of_the_width_is_refused_at_parse() {
    for (dtype, shape, span) in [
        ("F32", "[3]", 11u64),
        ("F32", "[3]", 13),
        ("BF16", "[3]", 5),
        ("C64", "[3]", 23),
        ("C64", "[3]", 25),
        ("I64", "[2]", 15),
    ] {
        let data = vec![0u8; span as usize];
        let err = SafeTensors::parse(&one(dtype, shape, 0, span, &data))
            .expect_err("a span that disagrees with shape x width must not parse");
        assert!(err.to_string().contains("bytes"), "{dtype} {span}: {err}");
    }
}

#[test]
fn a_chunk_boundary_span_off_by_one_byte_is_refused() {
    let n = CHUNK as u64 / 4;
    for span in [CHUNK as u64 - 1, CHUNK as u64 + 1] {
        let data = vec![0u8; span as usize];
        let bytes = one("F32", &format!("[{n}]"), 0, span, &data);
        assert!(SafeTensors::parse(&bytes).is_err(), "span {span}");
    }
    // exactly the chunk parses
    let data = vec![0u8; CHUNK];
    let bytes = one("F32", &format!("[{n}]"), 0, CHUNK as u64, &data);
    assert_eq!(
        SafeTensors::parse(&bytes)
            .unwrap()
            .read_f32("t")
            .unwrap()
            .1
            .len(),
        n as usize
    );
}

#[test]
fn zero_element_tensors_parse_and_read_empty() {
    // shapes [0], [0, N], [N, 0], [0, 0]; offsets are all empty spans.
    let header = r#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"b":{"dtype":"BF16","shape":[0,5],"data_offsets":[0,0]},"c":{"dtype":"F16","shape":[5,0],"data_offsets":[0,0]},"d":{"dtype":"C64","shape":[0,0],"data_offsets":[0,0]},"e":{"dtype":"I64","shape":[0],"data_offsets":[0,0]},"f":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let bytes = frame(header, &1.5f32.to_le_bytes());
    let st = SafeTensors::parse(&bytes).unwrap();
    assert_eq!(st.read_f32("a").unwrap(), (vec![0], vec![]));
    assert_eq!(st.read_bf16_bits("b").unwrap(), (vec![0, 5], vec![]));
    assert_eq!(st.read_f32_widened("c").unwrap(), (vec![5, 0], vec![]));
    assert!(st.read_bytes("d").unwrap().is_empty());
    assert_eq!(st.read_i64("e").unwrap().1, Vec::<i64>::new());
    assert_eq!(st.read_f32("f").unwrap().1, vec![1.5]);
    st.read_into("a", 0, &mut []).unwrap();
    let mut one_byte = [0u8; 1];
    assert!(st.read_into("a", 0, &mut one_byte).is_err());
}

#[test]
fn shape_products_that_overflow_are_refused_without_panic() {
    for (dtype, shape) in [
        ("F32", "[4294967296,4294967296]"),
        ("F32", "[18446744073709551615]"),
        ("F32", "[4611686018427387904]"),
        ("C64", "[2305843009213693952]"),
        ("C64", "[2305843009213693953]"),
        ("F32", "[18446744073709551615,2]"),
        ("BF16", "[9223372036854775808,2]"),
    ] {
        let err = SafeTensors::parse(&one(dtype, shape, 0, 0, &[]))
            .expect_err(&format!("{dtype} {shape} must be refused"));
        let _ = err.to_string();
    }
    // A zero factor makes the product 0 whatever the other dims are.
    let bytes = one("F32", "[0,18446744073709551615]", 0, 0, &[]);
    if let Ok(st) = SafeTensors::parse(&bytes) {
        assert!(st.read_f32("t").unwrap().1.is_empty());
    }
}

#[test]
fn dtype_near_misses_are_refused_and_never_mapped_to_a_neighbour() {
    for tag in [
        "c64",
        "C64 ",
        " C64",
        "C64\\u0000",
        "C 64",
        "C128",
        "f32",
        "F32 ",
        "BF16 ",
        "bf16",
        "F8_E4M3fnuz",
        "F8_E4M3FNUZ ",
        "F8_E5M2FNUz",
        "F8_E4M3FN",
        "F8_E4M3FNUZZ",
        "F7",
        "",
    ] {
        let hdr = format!(r#"{{"t":{{"dtype":"{tag}","shape":[1],"data_offsets":[0,8]}}}}"#);
        let err = SafeTensors::parse(&frame(&hdr, &[0; 8]))
            .expect_err(&format!("tag {tag:?} must be refused"));
        let text = err.to_string();
        assert!(!text.contains("not in the safetensors spec"), "{text}");
        assert!(text.contains("unknown dtype"), "{tag:?}: {text}");
    }
}

#[test]
fn duplicate_tensor_names_are_refused_even_when_escaped() {
    let dup = frame(
        r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"t":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
        &[0; 8],
    );
    assert!(SafeTensors::parse(&dup).is_err());
    let esc = frame(
        r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"t":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
        &[0; 8],
    );
    assert!(SafeTensors::parse(&esc).is_err());
}

#[test]
fn a_header_claiming_f32_over_bf16_sized_bytes_is_refused() {
    // 2 elements of BF16 data under an F32 claim.
    assert!(SafeTensors::parse(&one("F32", "[2]", 0, 4, &[0; 4])).is_err());
    // and the converse: 2 elements claimed BF16, F32-sized span
    assert!(SafeTensors::parse(&one("BF16", "[2]", 0, 8, &[0; 8])).is_err());
}

// ---- chunk boundaries ----------------------------------------------------

fn pattern_f32(i: usize) -> u32 {
    (i as u32).wrapping_mul(2_654_435_761) ^ 0x1234_5678
}

fn file_of(dtype: StDtype, shape: &[u64], data: &[u8]) -> Vec<u8> {
    encode_safetensors(
        &[TensorOut {
            name: "t",
            dtype,
            shape,
            data,
        }],
        &[],
    )
    .unwrap()
}

#[test]
fn f32_reads_are_exact_across_the_chunk_boundary() {
    let per = CHUNK / 4;
    for n in [
        per - 1,
        per,
        per + 1,
        2 * per - 1,
        2 * per,
        2 * per + 1,
        1,
        0,
    ] {
        let mut data = Vec::new();
        for i in 0..n {
            data.extend_from_slice(&pattern_f32(i).to_le_bytes());
        }
        let bytes = file_of(StDtype::F32, &[n as u64], &data);
        let st = SafeTensors::parse(&bytes).unwrap();
        let (shape, got) = st.read_f32("t").unwrap();
        assert_eq!(shape, vec![n as u64]);
        assert_eq!(got.len(), n, "n={n}");
        for (i, v) in got.iter().enumerate() {
            assert_eq!(v.to_bits(), pattern_f32(i), "n={n} i={i}");
        }
    }
}

#[test]
fn half_width_and_i64_reads_are_exact_across_the_chunk_boundary() {
    let half = CHUNK / 2;
    for n in [half - 1, half, half + 1, 2 * half + 1] {
        let bits = |i: usize| (i as u16).wrapping_mul(40503).wrapping_add(7);
        let mut data = Vec::new();
        for i in 0..n {
            data.extend_from_slice(&bits(i).to_le_bytes());
        }
        for dtype in [StDtype::BF16, StDtype::U16] {
            let bytes = file_of(dtype, &[n as u64], &data);
            let st = SafeTensors::parse(&bytes).unwrap();
            let got = if dtype == StDtype::BF16 {
                st.read_bf16_bits("t").unwrap().1
            } else {
                st.read_u16("t").unwrap().1
            };
            assert!(
                got.iter().enumerate().all(|(i, &b)| b == bits(i)),
                "{dtype:?} n={n}"
            );
            assert_eq!(got.len(), n);
        }
        let bytes = file_of(StDtype::F16, &[n as u64], &data);
        let st = SafeTensors::parse(&bytes).unwrap();
        let got = st.read_f32_widened("t").unwrap().1;
        for (i, v) in got.iter().enumerate() {
            let want = ojas_io::f16_to_f32(bits(i));
            assert!(v.to_bits() == want.to_bits(), "f16 n={n} i={i}");
        }
    }
    let eight = CHUNK / 8;
    for n in [eight - 1, eight, eight + 1, 3 * eight + 1] {
        let val = |i: usize| (i as i64).wrapping_mul(0x0101_0101_0101) - 17;
        let mut data = Vec::new();
        for i in 0..n {
            data.extend_from_slice(&val(i).to_le_bytes());
        }
        let bytes = file_of(StDtype::I64, &[n as u64], &data);
        let st = SafeTensors::parse(&bytes).unwrap();
        let got = st.read_i64("t").unwrap().1;
        assert!(
            got.iter().enumerate().all(|(i, &v)| v == val(i)),
            "i64 n={n}"
        );
        assert_eq!(got.len(), n);
    }
}

#[test]
fn c64_bytes_round_trip_at_chunk_boundaries_and_typed_reads_name_it() {
    let eight = CHUNK / 8;
    for n in [eight - 1, eight, eight + 1, 2 * eight, 2 * eight + 1] {
        let data: Vec<u8> = (0..n * 8).map(|i| (i * 7 + i / 251) as u8).collect();
        let bytes = file_of(StDtype::C64, &[n as u64], &data);
        let st = SafeTensors::parse(&bytes).unwrap();
        assert_eq!(st.info("t").unwrap().dtype, StDtype::C64);
        assert_eq!(st.read_bytes("t").unwrap(), data, "n={n}");
        // windows straddling the 128 KiB / 8 byte boundary, one element each
        for at in [CHUNK - 8, CHUNK - 4, CHUNK, CHUNK + 8] {
            if at + 8 <= data.len() {
                let mut w = [0u8; 8];
                st.read_into("t", at as u64, &mut w).unwrap();
                assert_eq!(&w[..], &data[at..at + 8], "n={n} at={at}");
            }
        }
        for err in [
            st.read_f32("t").unwrap_err(),
            st.read_f32_widened("t").unwrap_err(),
            st.read_bf16_bits("t").unwrap_err(),
            st.read_i64("t").unwrap_err(),
            st.read_u16("t").unwrap_err(),
        ] {
            let text = err.to_string();
            assert!(text.contains('t') && text.contains("C64"), "{text}");
        }
    }
}

#[test]
fn a_read_window_one_byte_past_the_tensor_is_refused_at_every_boundary() {
    let n = CHUNK / 4;
    let data = vec![1u8; n * 4];
    let bytes = file_of(StDtype::F32, &[n as u64], &data);
    let st = SafeTensors::parse(&bytes).unwrap();
    let mut w = [0u8; 2];
    assert!(st.read_into("t", (n * 4 - 1) as u64, &mut w).is_err());
    assert!(st.read_into("t", u64::MAX, &mut w).is_err());
    assert!(st.read_into("t", u64::MAX - 1, &mut w).is_err());
    st.read_into("t", (n * 4 - 2) as u64, &mut w).unwrap();
}

#[test]
fn new_tags_round_trip_and_a_short_c64_is_refused_by_the_writer() {
    let f8a = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let c = [9u8; 24];
    let items = [
        TensorOut {
            name: "a",
            dtype: StDtype::F8E4M3Fnuz,
            shape: &[8],
            data: &f8a,
        },
        TensorOut {
            name: "b",
            dtype: StDtype::F8E5M2Fnuz,
            shape: &[2, 4],
            data: &f8a,
        },
        TensorOut {
            name: "c",
            dtype: StDtype::C64,
            shape: &[3],
            data: &c,
        },
    ];
    let bytes = encode_safetensors(&items, &[]).unwrap();
    let st = SafeTensors::parse(&bytes).unwrap();
    assert_eq!(st.read_bytes("a").unwrap(), f8a);
    assert_eq!(st.read_bytes("b").unwrap(), f8a);
    assert_eq!(st.read_bytes("c").unwrap(), c);
    assert!(encode_safetensors(
        &[TensorOut {
            name: "c",
            dtype: StDtype::C64,
            shape: &[3],
            data: &c[..23]
        }],
        &[]
    )
    .is_err());
}
