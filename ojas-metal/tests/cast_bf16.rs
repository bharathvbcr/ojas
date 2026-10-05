//! Device bf16 round: bit-exact with the host round, and a NaN is not a fault.

mod common;

use common::{down, host, host_u32, metal, ok, up};
use ojas_core::{round_f32_to_bf16, Backend, OjasError};

fn patterns() -> Vec<f32> {
    let mut out = Vec::with_capacity(65536 * 2 + 8);
    for bits in 0u32..=u16::MAX as u32 {
        out.push(f32::from_bits(bits << 16));
        out.push(f32::from_bits((bits << 16) | 0x0001));
        out.push(f32::from_bits((bits << 16) | 0x7fff));
        out.push(f32::from_bits((bits << 16) | 0x8000));
    }
    out.extend([
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xff80_0001),
        f32::from_bits(0x7fc0_0000),
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8000_0000),
        f32::from_bits(0x7f7f_ffff),
        f32::from_bits(0xff7f_ffff),
    ]);
    out
}

#[test]
fn cast_bf16_matches_the_host_round_and_a_nan_is_not_a_fault() {
    let m = metal();
    let values = patterns();
    let dev = up(&m, &host(&values, &[values.len()]));
    let y = ok("cast", m.cast_bf16(&dev));
    ok("sync", m.sync());
    let got = down(&y);
    assert_eq!(got.len(), values.len());
    for (i, (src, got)) in values.iter().zip(&got).enumerate() {
        let want = round_f32_to_bf16(*src).to_bits();
        assert_eq!(got.to_bits(), want, "index {i} src {:08x}", src.to_bits());
    }
    let host_only = host(&[1.0], &[1]);
    let err = m.cast_bf16(&host_only).unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::Placement {
                op: "cast_bf16",
                ..
            }
        ),
        "{err:?}"
    );
    let ids = up(&m, &host_u32(&[1, 2, 3, 4], &[4]));
    let err = m.cast_bf16(&ids).unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::Dtype {
                op: "cast_bf16",
                ..
            }
        ),
        "{err:?}"
    );
    let empty = ok("empty view", dev.view(&[0, 4], &[4, 1], 0));
    let err = m.cast_bf16(&empty).unwrap_err();
    match err {
        OjasError::Shape { op, detail } => {
            assert_eq!(op, "cast_bf16");
            assert_eq!(detail, "empty tensor");
        }
        other => panic!("expected Shape, got {other:?}"),
    }
    let window = ok("window", dev.view(&[2], &[1], 8));
    let y = ok("offset cast", m.cast_bf16(&window));
    ok("sync window", m.sync());
    let got = down(&y);
    assert_eq!(got.len(), 2);
    for (src, got) in values[2..4].iter().zip(&got) {
        assert_eq!(got.to_bits(), round_f32_to_bf16(*src).to_bits());
    }
}
