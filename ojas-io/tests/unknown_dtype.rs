//! One tensor of a dtype this crate does not decode must not reject the file:
//! the others stay readable, and only a typed read of that tensor is refused,
//! naming the tensor and its dtype.

use ojas_io::{encode_safetensors, SafeTensors, StDtype, TensorOut};

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

#[test]
fn bf16_tensors_read_beside_a_u8_tensor() {
    // w: BF16 [2] = 1.0, 2.0 ; mask: U8 [3] ; v: BF16 [1] = -1.0
    let header = r#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"mask":{"dtype":"U8","shape":[3],"data_offsets":[4,7]},"v":{"dtype":"BF16","shape":[1],"data_offsets":[7,9]}}"#;
    let mut data = Vec::new();
    for v in [1.0f32, 2.0] {
        data.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
    }
    data.extend_from_slice(&[7, 8, 9]);
    data.extend_from_slice(&(((-1.0f32).to_bits() >> 16) as u16).to_le_bytes());
    let bytes = frame(header, &data);

    let st = SafeTensors::parse(&bytes).expect("an unfamiliar dtype must not reject the file");
    assert_eq!(format!("{:?}", st.info("mask").unwrap().dtype), "U8");
    assert_eq!(st.read_f32_widened("w").unwrap().1, vec![1.0, 2.0]);
    assert_eq!(st.read_f32_widened("v").unwrap().1, vec![-1.0]);
    assert_eq!(st.read_bytes("mask").unwrap(), vec![7, 8, 9]);

    for err in [
        st.read_f32_widened("mask").unwrap_err(),
        st.read_f32("mask").unwrap_err(),
        st.read_bf16_bits("mask").unwrap_err(),
    ] {
        let text = err.to_string();
        assert!(text.contains("mask") && text.contains("U8"), "{text}");
    }
}

#[test]
fn every_spec_dtype_parses_and_a_name_outside_the_spec_still_fails() {
    // The 19 spec tags plus the three safetensors 0.8.0 adds ("F8_E4M3FNUZ",
    // "F8_E5M2FNUZ", "C64"), compared against the 0.8.0 release's adapter tables.
    let spec = [
        ("BOOL", 8u64),
        ("U8", 8),
        ("I8", 8),
        ("F8_E5M2", 8),
        ("F8_E4M3", 8),
        ("F8_E8M0", 8),
        ("I16", 16),
        ("U16", 16),
        ("F16", 16),
        ("BF16", 16),
        ("I32", 32),
        ("U32", 32),
        ("F32", 32),
        ("F64", 64),
        ("I64", 64),
        ("U64", 64),
        ("F4", 4),
        ("F6_E2M3", 6),
        ("F6_E3M2", 6),
        ("F8_E4M3FNUZ", 8),
        ("F8_E5M2FNUZ", 8),
        ("C64", 64),
    ];
    for (tag, bits) in spec {
        // 8 elements: whole bytes for every width, including F4 and F6.
        let n = 8 * bits / 8;
        let header = format!(r#"{{"t":{{"dtype":"{tag}","shape":[8],"data_offsets":[0,{n}]}}}}"#);
        let bytes = frame(&header, &vec![0u8; n as usize]);
        let st = SafeTensors::parse(&bytes).unwrap_or_else(|e| panic!("{tag}: {e}"));
        let info = st.info("t").unwrap();
        assert_eq!(info.end - info.begin, n, "{tag}");
    }
    let bytes = frame(
        r#"{"t":{"dtype":"F7","shape":[1],"data_offsets":[0,1]}}"#,
        &[0],
    );
    let err = SafeTensors::parse(&bytes).unwrap_err().to_string();
    assert!(err.contains("F7"), "{err}");
    assert!(!err.contains("not in the safetensors spec"), "{err}");
    assert!(err.contains("0.8.0"), "{err}");
}

#[test]
fn safetensors_0_8_0_tags_parse_carry_bytes_and_refuse_typed_reads() {
    // a: F8_E4M3FNUZ [8], b: F8_E5M2FNUZ [8], c: C64 [3] = 24 bytes, w: BF16 [2]
    let header = r#"{"a":{"dtype":"F8_E4M3FNUZ","shape":[8],"data_offsets":[0,8]},"b":{"dtype":"F8_E5M2FNUZ","shape":[8],"data_offsets":[8,16]},"c":{"dtype":"C64","shape":[3],"data_offsets":[16,40]},"w":{"dtype":"BF16","shape":[2],"data_offsets":[40,44]}}"#;
    let a: Vec<u8> = (1..=8).collect();
    let b: Vec<u8> = (11..=18).collect();
    let c: Vec<u8> = (21..45).collect();
    let mut data = [a.clone(), b.clone(), c.clone()].concat();
    for v in [1.5f32, -2.0] {
        data.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
    }
    let bytes = frame(header, &data);
    let st = SafeTensors::parse(&bytes).expect("0.8.0 tags must parse");
    assert_eq!(st.read_f32_widened("w").unwrap().1, vec![1.5, -2.0]);
    for (name, want, dtype) in [
        ("a", &a, "F8E4M3Fnuz"),
        ("b", &b, "F8E5M2Fnuz"),
        ("c", &c, "C64"),
    ] {
        assert_eq!(st.read_bytes(name).unwrap(), *want, "{name}");
        assert_eq!(format!("{:?}", st.info(name).unwrap().dtype), dtype);
        for err in [
            st.read_f32(name).unwrap_err(),
            st.read_f32_widened(name).unwrap_err(),
            st.read_bf16_bits(name).unwrap_err(),
        ] {
            let text = err.to_string();
            assert!(text.contains(name) && text.contains(dtype), "{text}");
        }
    }
    assert_eq!(st.info("c").unwrap().end - st.info("c").unwrap().begin, 24);

    // write_safetensors / encode round-trips every new tag byte for byte.
    let items = [
        TensorOut {
            name: "a",
            dtype: StDtype::F8E4M3Fnuz,
            shape: &[8],
            data: &a,
        },
        TensorOut {
            name: "b",
            dtype: StDtype::F8E5M2Fnuz,
            shape: &[8],
            data: &b,
        },
        TensorOut {
            name: "c",
            dtype: StDtype::C64,
            shape: &[3],
            data: &c,
        },
    ];
    let out = encode_safetensors(&items, &[]).unwrap();
    let text = String::from_utf8_lossy(&out);
    for tag in ["F8_E4M3FNUZ", "F8_E5M2FNUZ", "C64"] {
        assert!(text.contains(tag), "{tag}");
    }
    let back = SafeTensors::parse(&out).unwrap();
    for (name, want) in [("a", &a), ("b", &b), ("c", &c)] {
        assert_eq!(back.read_bytes(name).unwrap(), *want);
        assert_eq!(back.info(name).unwrap().dtype, st.info(name).unwrap().dtype);
    }
    let path =
        std::env::temp_dir().join(format!("ojas-io-fnuz-{}.safetensors", std::process::id()));
    ojas_io::write_safetensors(&path, &items, &[]).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), out);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn c64_with_a_short_span_is_refused() {
    // C64 [3] needs 24 bytes; [0,23] is one short.
    let bytes = frame(
        r#"{"c":{"dtype":"C64","shape":[3],"data_offsets":[0,23]}}"#,
        &[0u8; 23],
    );
    assert!(SafeTensors::parse(&bytes).is_err());
}

#[test]
fn sub_byte_tensor_with_a_partial_byte_is_refused() {
    let bytes = frame(
        r#"{"t":{"dtype":"F4","shape":[3],"data_offsets":[0,2]}}"#,
        &[0, 0],
    );
    assert!(SafeTensors::parse(&bytes).is_err());
}
