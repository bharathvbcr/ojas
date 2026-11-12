//! `BackendModule::dequant_affine_bf16` on the CPU: the host implementation
//! (`ojas_core::dequantize_affine_bf16_tensors`, `ojas-core/src/quant.rs`) behind the backend.
//!
//! It must equal mlx's `mx.dequantize(stream=mx.gpu)` goldens and the host dequantiser, be
//! elementwise (no thread count or numerics mode changes a bit), charge its output before it
//! writes, refuse non-finite results under its own name, and refuse operands it cannot read.

// The goldens are read by ojas-core's reader, included by path: no dependency, no Cargo.lock change.
#[path = "../../ojas-core/tests/mlx_affine_common/mod.rs"]
#[allow(dead_code)]
mod mlx_affine_common;

use std::any::Any;
use std::path::Path;
use std::sync::Arc;

use ojas_core::{
    dequantize_affine_bf16, f32_to_bf16, AffineQuantized, AffineSpec, BackendId, BackendModule,
    Budget, DType, DeviceBuffer, Numerics, OjasError, Tensor,
};
use ojas_cpu::CpuBackend;

const OP: &str = "dequant_affine_bf16";

fn golden_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../ojas-core/tests/fixtures/mlx_affine")
}

fn cpu(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 32), threads)
        .unwrap()
        .with_numerics(numerics)
}

struct Parts {
    packed: Tensor,
    scales: Tensor,
    biases: Tensor,
    spec: AffineSpec,
}

fn parts(
    spec: AffineSpec,
    rows: usize,
    groups: usize,
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
) -> Parts {
    let b = Budget::new(1 << 30);
    let words = groups * spec.words_per_group();
    Parts {
        packed: Tensor::from_u32(packed, &[rows, words], &b).unwrap(),
        scales: Tensor::from_bf16_bits(scales, &[rows, groups], &b).unwrap(),
        biases: Tensor::from_bf16_bits(biases, &[rows, groups], &b).unwrap(),
        spec,
    }
}

impl Parts {
    fn run(&self, c: &CpuBackend) -> Result<Tensor, OjasError> {
        c.dequant_affine_bf16(&self.packed, &self.scales, &self.biases, self.spec)
    }

    fn bits(&self, c: &CpuBackend) -> Vec<u16> {
        self.run(c).unwrap().bf16_slice().unwrap().to_vec()
    }
}

fn expect_same(what: &str, got: &[u16], want: &[u16]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(g, w)| g != w) {
        panic!(
            "{what}: first difference at {i}: got {:#06x}, want {:#06x}",
            got[i], want[i]
        );
    }
}

#[test]
fn cpu_op_equals_host_dequantiser_on_every_golden() {
    let c = cpu(1, Numerics::Exact);
    let b = Budget::new(1 << 32);
    let mut cases = mlx_affine_common::CASES.to_vec();
    cases.push("deq_params_w64");
    for case in cases {
        let golden = mlx_affine_common::load_from(&golden_dir(), case);
        for (bits, spec) in [(8u32, AffineSpec::q8()), (4, AffineSpec::q4())] {
            let tag = format!("{case} q{bits}");
            let (wq_shape, wq) = golden.u32s(&format!("q{bits}.wq"));
            let (sb_shape, scales) = golden.bf16(&format!("q{bits}.scales"));
            let (_, biases) = golden.bf16(&format!("q{bits}.biases"));
            let (_, want) = golden.bf16(&format!("q{bits}.deq"));
            let p = parts(spec, wq_shape[0], sb_shape[1], wq, scales, biases);
            let got = p.bits(&c);
            expect_same(&format!("{tag} against mlx"), &got, want);
            // And against the host dequantiser of the validated parts.
            let q = AffineQuantized::from_parts(
                spec,
                wq_shape[0],
                sb_shape[1] * 64,
                p.packed.clone(),
                p.scales.clone(),
                p.biases.clone(),
            )
            .unwrap();
            let host = dequantize_affine_bf16(&q, &b).unwrap();
            expect_same(
                &format!("{tag} against the host"),
                &got,
                host.bf16_slice().unwrap(),
            );
        }
    }
}

/// Operands of a `[3, 128]` q8 matrix (2 groups per row), scale 1 and bias 0, charged to `budget`.
fn charged(budget: &Budget) -> (Tensor, Tensor, Tensor) {
    (
        Tensor::from_u32(&[0; 3 * 32], &[3, 32], budget).unwrap(),
        Tensor::from_bf16_bits(&[0x3F80; 6], &[3, 2], budget).unwrap(),
        Tensor::from_bf16_bits(&[0; 6], &[3, 2], budget).unwrap(),
    )
}

#[test]
fn cpu_op_charges_the_output_and_refuses_one_byte_short_with_nothing_charged() {
    let spec = AffineSpec::q8();
    let probe = Budget::new(1 << 30);
    let _operands = charged(&probe);
    let held = probe.live_bytes().unwrap();
    let out_bytes = (3 * 2 * 64 * 2) as u64;

    // The operands share the output's budget, as a caller's do.
    let short = Budget::new(held + out_bytes - 1);
    let (p, s, z) = charged(&short);
    let c = CpuBackend::new(short.clone());
    match c.dequant_affine_bf16(&p, &s, &z, spec) {
        Err(OjasError::CapacityExceeded { .. }) => {}
        other => panic!("{other:?}"),
    }
    assert_eq!(short.live_bytes().unwrap(), held, "a refusal charged");

    let exact = Budget::new(held + out_bytes);
    let (p, s, z) = charged(&exact);
    let c = CpuBackend::new(exact.clone());
    let out = c.dequant_affine_bf16(&p, &s, &z, spec).unwrap();
    assert_eq!(out.dtype(), DType::Bf16);
    assert_eq!(exact.live_bytes().unwrap(), held + out_bytes);
    drop(out);
    assert_eq!(exact.live_bytes().unwrap(), held);
}

#[test]
fn cpu_op_refuses_nan_scale_inf_scale_at_code_0_and_inf_bias_as_nonfinite_under_its_own_name() {
    let c = cpu(1, Numerics::Exact);
    for (name, scale, bias) in [
        ("NaN scale", 0x7FC0u16, 0u16),
        ("+inf scale at code 0", 0x7F80, 0),
        ("-inf scale at code 0", 0xFF80, 0),
        ("+inf bias", 0x3F80, 0x7F80),
        ("-inf bias", 0x3F80, 0xFF80),
        ("NaN bias", 0x3F80, 0x7FC0),
    ] {
        for spec in [AffineSpec::q8(), AffineSpec::q4()] {
            let p = parts(
                spec,
                1,
                1,
                &vec![0; spec.words_per_group()],
                &[scale],
                &[bias],
            );
            match p.run(&c) {
                Err(OjasError::NonFinite { op }) => assert_eq!(op, OP, "{name}"),
                other => panic!("{name}: expected NonFinite, got {other:?}"),
            }
        }
    }
    // Finite scale and bias whose result overflows bf16: the check is on the output, so it
    // catches this too (the device's fault word flags the same case).
    let p = parts(
        AffineSpec::q8(),
        1,
        1,
        &[u32::MAX; 16],
        &[0x7F7F],
        &[0x7F7F],
    );
    assert!(matches!(p.run(&c), Err(OjasError::NonFinite { op: OP })));
}

/// scale 2^125, code 8, bias -bf16::MAX: the unfused product overflows f32, the fma is finite.
#[test]
fn fused_overflow_edge_is_finite() {
    let c = cpu(1, Numerics::Exact);
    let scale = f32_to_bf16(2f32.powi(125));
    let p = parts(
        AffineSpec::q8(),
        2,
        1,
        &[0x0808_0808; 32],
        &[scale; 2],
        &[0xFF7F; 2],
    );
    assert!(p.bits(&c).iter().all(|&x| x == 0x7B80));
}

/// A subnormal scale, bias or result behaves as a zero, as on the GPU.
#[test]
fn subnormal_scale_and_bias_flush() {
    let c = cpu(1, Numerics::Exact);
    // Subnormal scale and bias, code 255: unflushed the result is about 2.3e-38, flushed it is 0.
    let p = parts(
        AffineSpec::q8(),
        1,
        1,
        &[u32::MAX; 16],
        &[0x0001],
        &[0x0001],
    );
    assert!(p.bits(&c).iter().all(|&x| x == 0));
    // A subnormal result of normal operands: 2^-126 (1 + 2^-7) * 1 - 2^-126 = 2^-133.
    let p = parts(
        AffineSpec::q8(),
        1,
        1,
        &[0x0101_0101; 16],
        &[0x0081],
        &[0x8080],
    );
    assert!(p.bits(&c).iter().all(|&x| x == 0), "{:#06x?}", p.bits(&c));
}

#[test]
fn q4_lanes_decode_in_order() {
    let c = cpu(1, Numerics::Exact);
    let p = parts(AffineSpec::q4(), 1, 1, &[0x7654_3210; 8], &[0x3F80], &[0]);
    let want: Vec<u16> = (0..64).map(|i| f32_to_bf16((i % 8) as f32)).collect();
    expect_same("q4 lanes", &p.bits(&c), &want);
    let p = parts(AffineSpec::q8(), 1, 1, &[0x0403_0201; 16], &[0x3F80], &[0]);
    let want: Vec<u16> = (0..64).map(|i| f32_to_bf16((i % 4 + 1) as f32)).collect();
    expect_same("q8 lanes", &p.bits(&c), &want);
}

#[derive(Debug)]
struct Resident(usize);

impl DeviceBuffer for Resident {
    fn backend(&self) -> BackendId {
        BackendId::Metal
    }
    fn byte_len(&self) -> usize {
        self.0
    }
    fn read_bytes(&self, _offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        Ok(vec![0; len])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn non_contiguous_and_device_operands_are_refused() {
    let c = cpu(1, Numerics::Exact);
    let b = Budget::new(1 << 30);
    let p = parts(AffineSpec::q8(), 2, 2, &[0; 64], &[0x3F80; 4], &[0; 4]);
    // Permuted strides over the same shape: not one run of bytes, so it is not read.
    let strided = p.scales.view(&[2, 2], &[1, 2], 0).unwrap();
    let r = c.dequant_affine_bf16(&p.packed, &strided, &p.biases, p.spec);
    assert!(
        matches!(&r, Err(OjasError::Shape { op: "Tensor::bf16_slice", detail }) if detail == "view is not contiguous"),
        "{r:?}"
    );
    // A device-resident operand is a Placement refusal, never a readback, in each position.
    let reads = b.device_readbacks();
    let dev_packed =
        Tensor::from_device(Arc::new(Resident(256)), &[2, 32], DType::U32, &b).unwrap();
    let dev_scales = Tensor::from_device(Arc::new(Resident(8)), &[2, 2], DType::Bf16, &b).unwrap();
    for (name, packed, scales, biases) in [
        ("packed", &dev_packed, &p.scales, &p.biases),
        ("scales", &p.packed, &dev_scales, &p.biases),
        ("biases", &p.packed, &p.scales, &dev_scales),
    ] {
        let r = c.dequant_affine_bf16(packed, scales, biases, p.spec);
        assert!(
            matches!(r, Err(OjasError::Placement { .. })),
            "{name}: {r:?}"
        );
    }
    assert_eq!(b.device_readbacks(), reads);
}

/// The `!bf16_is_finite` branch of `expand`: an f32-FINITE result that rounds to a bf16 infinity.
/// scale `bf16::MAX` (2^128 - 2^120), code 1, bias 2^119 (0x7B00) gives 2^128 - 2^119, finite in
/// f32 and exactly the midpoint between `bf16::MAX` and 2^128, which round-to-even sends to
/// infinity. Bias 0x7AFF is below the midpoint and stays `bf16::MAX`.
#[test]
fn f32_finite_result_that_rounds_to_a_bf16_infinity_is_non_finite() {
    let c = cpu(1, Numerics::Exact);
    let p = parts(
        AffineSpec::q8(),
        1,
        1,
        &[0x0101_0101; 16],
        &[0x7F7F],
        &[0x7B00],
    );
    match p.run(&c) {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, OP),
        other => panic!("expected NonFinite, got {other:?}"),
    }
    let p = parts(
        AffineSpec::q8(),
        1,
        1,
        &[0x0101_0101; 16],
        &[0x7F7F],
        &[0x7AFF],
    );
    assert!(p.bits(&c).iter().all(|&x| x == 0x7F7F));
}
