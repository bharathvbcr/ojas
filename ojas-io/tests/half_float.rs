//! BF16 and F16 safetensors: exact decode, round-to-nearest-even encode.
//!
//! NaN rule under test: a NaN keeps its sign and the payload bits that fit,
//! and the quiet bit is set (bf16 `0x0040`, f16 `0x0200`), so a NaN whose
//! payload lives only in dropped low bits stays NaN instead of becoming
//! infinity. A quiet NaN therefore round-trips bit for bit; a signalling
//! NaN comes back quiet with the same sign and payload.

use ojas_io::{
    bf16_to_f32, encode_f32_as, encode_safetensors, f16_to_f32, f32_to_bf16, f32_to_f16,
    SafeTensors, StDtype, TensorOut,
};

fn is_bf16_nan(b: u16) -> bool {
    b & 0x7F80 == 0x7F80 && b & 0x007F != 0
}

fn is_f16_nan(b: u16) -> bool {
    b & 0x7C00 == 0x7C00 && b & 0x03FF != 0
}

/// Value of an f16 bit pattern, by the IEEE 754 binary16 definition, in f64.
fn f16_value(b: u16) -> f64 {
    let sign = if b & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((b >> 10) & 0x1F);
    let man = f64::from(b & 0x03FF);
    match exp {
        0 => sign * man * 2f64.powi(-24),
        31 if man == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        e => sign * (1.0 + man / 1024.0) * 2f64.powi(e - 15),
    }
}

#[test]
fn every_bf16_pattern_decodes_exactly_and_round_trips() {
    for b in 0..=u16::MAX {
        let v = bf16_to_f32(b);
        assert_eq!(v.to_bits(), u32::from(b) << 16, "bf16 {b:#06x}");
        let back = f32_to_bf16(v);
        if is_bf16_nan(b) {
            assert!(is_bf16_nan(back), "{b:#06x} -> {back:#06x}");
            assert_eq!(
                back,
                b | 0x0040,
                "{b:#06x}: sign and payload kept, quiet set"
            );
        } else {
            assert_eq!(back, b, "bf16 {b:#06x} did not round-trip");
        }
    }
}

#[test]
fn every_f16_pattern_decodes_exactly_and_round_trips() {
    for b in 0..=u16::MAX {
        let v = f16_to_f32(b);
        let want = f16_value(b);
        if is_f16_nan(b) {
            assert!(v.is_nan(), "{b:#06x}");
            assert_eq!(v.is_sign_negative(), b & 0x8000 != 0, "{b:#06x} sign");
            let back = f32_to_f16(v);
            assert!(is_f16_nan(back), "{b:#06x} -> {back:#06x}");
            assert_eq!(
                back,
                b | 0x0200,
                "{b:#06x}: sign and payload kept, quiet set"
            );
        } else {
            assert_eq!(f64::from(v), want, "f16 {b:#06x}");
            assert_eq!(
                v.is_sign_negative(),
                b & 0x8000 != 0,
                "{b:#06x} sign of zero"
            );
            assert_eq!(f32_to_f16(v), b, "f16 {b:#06x} did not round-trip");
        }
    }
}

/// For every pair of adjacent finite non-negative f16 values the exact
/// midpoint is an f32. It must round to the one with an even mantissa, and
/// one f32 ulp either side must round to the nearer one. Negatives mirror.
#[test]
fn f32_to_f16_rounds_every_midpoint_to_even() {
    for lo in 0u16..0x7BFF {
        let hi = lo + 1;
        let (a, b) = (f16_to_f32(lo), f16_to_f32(hi));
        let mid = ((f64::from(a) + f64::from(b)) / 2.0) as f32;
        assert_eq!(
            f64::from(mid),
            (f64::from(a) + f64::from(b)) / 2.0,
            "{lo:#06x}"
        );
        let even = if lo % 2 == 0 { lo } else { hi };
        assert_eq!(f32_to_f16(mid), even, "midpoint above {lo:#06x}");
        assert_eq!(f32_to_f16(-mid), even | 0x8000, "-midpoint above {lo:#06x}");
        let below = f32::from_bits(mid.to_bits() - 1);
        let above = f32::from_bits(mid.to_bits() + 1);
        assert_eq!(f32_to_f16(below), lo, "just below midpoint of {lo:#06x}");
        assert_eq!(f32_to_f16(above), hi, "just above midpoint of {lo:#06x}");
    }
}

#[test]
fn f32_to_bf16_rounds_every_midpoint_to_even() {
    for lo in 0u16..0x7F7F {
        let hi = lo + 1;
        let mid_bits = (u32::from(lo) << 16) | 0x8000;
        let mid = f32::from_bits(mid_bits);
        let even = if lo % 2 == 0 { lo } else { hi };
        assert_eq!(f32_to_bf16(mid), even, "midpoint above {lo:#06x}");
        assert_eq!(f32_to_bf16(-mid), even | 0x8000);
        assert_eq!(f32_to_bf16(f32::from_bits(mid_bits - 1)), lo);
        assert_eq!(f32_to_bf16(f32::from_bits(mid_bits + 1)), hi);
    }
}

#[test]
fn known_values_and_overflow_to_infinity() {
    // f16 limits.
    assert_eq!(f32_to_f16(65504.0), 0x7BFF);
    assert_eq!(f32_to_f16(65519.996), 0x7BFF, "below the tie stays finite");
    assert_eq!(
        f32_to_f16(65520.0),
        0x7C00,
        "tie between max and 2^16 -> inf"
    );
    assert_eq!(f32_to_f16(1.0e9), 0x7C00);
    assert_eq!(f32_to_f16(-1.0e9), 0xFC00);
    assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
    assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xFC00);
    assert_eq!(f32_to_f16(f32::MAX), 0x7C00);
    assert_eq!(f32_to_f16(2f32.powi(-14)), 0x0400, "smallest normal");
    assert_eq!(f32_to_f16(2f32.powi(-24)), 0x0001, "smallest subnormal");
    assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000, "tie with 0 -> even 0");
    assert_eq!(f32_to_f16(-(2f32.powi(-25))), 0x8000);
    assert_eq!(f32_to_f16(1.5 * 2f32.powi(-25)), 0x0001);
    assert_eq!(f32_to_f16(f32::MIN_POSITIVE), 0x0000);
    assert_eq!(f32_to_f16(0.0), 0x0000);
    assert_eq!(f32_to_f16(-0.0), 0x8000);
    assert_eq!(f32_to_f16(1.0), 0x3C00);
    assert_eq!(f32_to_f16(-2.0), 0xC000);
    assert_eq!(f32_to_f16(0.1), 0x2E66);
    assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
    assert!(is_f16_nan(f32_to_f16(f32::NAN)));
    // A NaN whose payload is only in the low 13 bits stays NaN.
    assert!(is_f16_nan(f32_to_f16(f32::from_bits(0x7F80_0001))));
    assert!(is_bf16_nan(f32_to_bf16(f32::from_bits(0x7F80_0001))));

    // bf16 limits: the largest finite f32 rounds up to infinity.
    assert_eq!(f32_to_bf16(1.0), 0x3F80);
    assert_eq!(f32_to_bf16(f32::MAX), 0x7F80);
    assert_eq!(f32_to_bf16(f32::from_bits(0x7F7F_7FFF)), 0x7F7F);
    assert_eq!(
        f32_to_bf16(f32::from_bits(0x7F7F_8000)),
        0x7F80,
        "odd max ties up"
    );
    assert_eq!(f32_to_bf16(-0.0), 0x8000);
    assert_eq!(
        f32_to_bf16(f32::from_bits(0x0000_8000)),
        0x0000,
        "subnormal tie to even"
    );
    assert_eq!(f32_to_bf16(f32::from_bits(0x0001_8000)), 0x0002);
    assert_eq!(f32_to_bf16(0.1), 0x3DCD);
}

#[test]
fn header_dtype_strings_are_exactly_bf16_and_f16() {
    let values = [1.0f32, -2.5, 0.1, 65504.0];
    let bf = encode_f32_as(StDtype::BF16, &values).unwrap();
    let hf = encode_f32_as(StDtype::F16, &values).unwrap();
    let f = encode_f32_as(StDtype::F32, &values).unwrap();
    assert_eq!(bf.len(), 8);
    assert_eq!(hf.len(), 8);
    assert_eq!(f.len(), 16);
    let file = encode_safetensors(
        &[
            TensorOut {
                name: "b",
                dtype: StDtype::BF16,
                shape: &[2, 2],
                data: &bf,
            },
            TensorOut {
                name: "h",
                dtype: StDtype::F16,
                shape: &[4],
                data: &hf,
            },
            TensorOut {
                name: "f",
                dtype: StDtype::F32,
                shape: &[4],
                data: &f,
            },
        ],
        &[],
    )
    .unwrap();
    let n = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&file[8..8 + n]).unwrap();
    assert!(header.contains(r#""b":{"dtype":"BF16","#), "{header}");
    assert!(header.contains(r#""h":{"dtype":"F16","#), "{header}");

    let st = SafeTensors::parse(&file).unwrap();
    assert_eq!(st.info("b").unwrap().dtype, StDtype::BF16);
    assert_eq!(st.info("h").unwrap().dtype, StDtype::F16);
    let (shape, got) = st.read_f32_widened("b").unwrap();
    assert_eq!(shape, vec![2, 2]);
    let want: Vec<f32> = values
        .iter()
        .map(|&v| bf16_to_f32(f32_to_bf16(v)))
        .collect();
    assert_eq!(got, want);
    let (_, got) = st.read_f32_widened("h").unwrap();
    let want: Vec<f32> = values.iter().map(|&v| f16_to_f32(f32_to_f16(v))).collect();
    assert_eq!(got, want);
    assert_eq!(got[3], 65504.0);
    let (_, got) = st.read_f32_widened("f").unwrap();
    assert_eq!(got, values);
    // The strict F32 reader still refuses a half-width tensor.
    assert!(st.read_f32("b").is_err());
    assert!(st.read_f32("h").is_err());

    // Integer dtypes are not float encodings.
    assert!(encode_f32_as(StDtype::I64, &values).is_err());
    assert!(encode_f32_as(StDtype::U16, &values).is_err());

    // Near-miss spellings stay unsupported.
    for tag in ["bf16", "Bf16", "FP16", "f16", "BF16 ", "F8"] {
        let header = format!(r#"{{"w":{{"dtype":"{tag}","shape":[1],"data_offsets":[0,2]}}}}"#);
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0, 0]);
        assert!(SafeTensors::parse(&bytes).is_err(), "{tag:?} accepted");
    }
    // A half-width tensor whose byte range is odd-sized is refused.
    let header = r#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,3]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0, 0, 0]);
    assert!(SafeTensors::parse(&bytes).is_err());
}
