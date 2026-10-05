//! Device bf16 round: bit-exact with the host round, and a NaN is not a fault.

mod common;

use common::{down, fresh, gpu, host_u32, up};
use ojas_core::{round_f32_to_bf16, Backend, Budget, DType, OjasError, Tensor};

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

fn host(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(values, shape, &Budget::new(1 << 30)).unwrap()
}

#[test]
fn cast_bf16_matches_the_host_round_and_a_nan_is_not_a_fault() {
    let g = fresh();
    let values = patterns();
    let dev = g.upload(&host(&values, &[values.len()])).unwrap();
    let y = g.cast_bf16(&dev).unwrap();
    g.sync().unwrap();
    let got = g.download(&y).unwrap().to_f32_vec().unwrap();
    assert_eq!(got.len(), values.len());
    for (i, (src, got)) in values.iter().zip(&got).enumerate() {
        let want = round_f32_to_bf16(*src).to_bits();
        assert_eq!(got.to_bits(), want, "index {i} src {:08x}", src.to_bits());
    }
    let err = g.cast_bf16(&host(&[1.0], &[1])).unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::Placement {
                op: "cast_bf16",
                found: None,
                ..
            }
        ),
        "{err:?}"
    );
    let ids = g.upload(&host_u32(&[1, 2, 3, 4], &[4])).unwrap();
    let err = g.cast_bf16(&ids).unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::Dtype {
                op: "cast_bf16",
                expected: DType::F32,
                got: DType::U32,
            }
        ),
        "{err:?}"
    );
    let empty = dev.view(&[0, 4], &[4, 1], 0).unwrap();
    let err = g.cast_bf16(&empty).unwrap_err();
    match err {
        OjasError::Shape { op, detail } => {
            assert_eq!(op, "cast_bf16");
            assert_eq!(detail, "empty tensor");
        }
        other => panic!("expected Shape, got {other:?}"),
    }
    let window = dev.view(&[2], &[1], 8).unwrap();
    let y = g.cast_bf16(&window).unwrap();
    g.sync().unwrap();
    let got = g.download(&y).unwrap().to_f32_vec().unwrap();
    assert_eq!(got.len(), 2);
    for (src, got) in values[2..4].iter().zip(&got) {
        assert_eq!(got.to_bits(), round_f32_to_bf16(*src).to_bits());
    }
    // The shared backend's cast must not raise a fault either.
    let shared = up(&host(&values[..64], &[64]));
    let y = gpu().cast_bf16(&shared).unwrap();
    gpu().sync().unwrap();
    let _ = down(&y);
}
