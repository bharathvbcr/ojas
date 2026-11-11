//! Refusals, budget behaviour, layout, the `-0.0` decision, the round-trip bound and
//! `AffineQuantized::from_parts` validation for the affine quantiser (`ojas-core/src/quant.rs`).

use ojas_core::{
    bf16_to_f32, dequantize_affine_bf16, f32_to_bf16, quantize_affine, AffineBits, AffineQuantized,
    AffineSpec, Budget, DType, OjasError, Tensor,
};
use std::time::Instant;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Approximately normal (sum of four uniforms, centred).
    fn normal(&mut self) -> f32 {
        (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 3f32.sqrt()
    }
}

fn big() -> Budget {
    Budget::new(1 << 32)
}

fn bf16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|&v| f32_to_bf16(v)).collect()
}

fn bf16_matrix(values: &[f32], rows: usize, cols: usize, b: &Budget) -> Tensor {
    Tensor::from_bf16_bits(&bf16_bits(values), &[rows, cols], b).unwrap()
}

fn one_group(values: &[(usize, f32)], fill: f32) -> Vec<f32> {
    let mut g = vec![fill; 64];
    for &(i, v) in values {
        g[i] = v;
    }
    g
}

fn q8() -> AffineSpec {
    AffineSpec::q8()
}

fn is_non_finite(e: &OjasError) -> bool {
    matches!(e, OjasError::NonFinite { .. })
}

// --- spec ----------------------------------------------------------------------------------

#[test]
fn spec_accepts_only_4_or_8_bits_and_group_64() {
    assert_eq!(AffineSpec::new(8, 64).unwrap(), AffineSpec::q8());
    assert_eq!(AffineSpec::new(4, 64).unwrap(), AffineSpec::q4());
    assert_eq!(AffineSpec::q8().bits(), AffineBits::Q8);
    assert_eq!(AffineSpec::q4().group_size(), 64);
    for bits in [0u32, 1, 2, 3, 5, 6, 7, 16, 32, u32::MAX] {
        assert!(
            matches!(AffineSpec::new(bits, 64), Err(OjasError::OutOfRange { .. })),
            "{bits} bits"
        );
        assert!(matches!(
            AffineBits::from_bits(bits),
            Err(OjasError::OutOfRange { .. })
        ));
    }
    for group in [0usize, 1, 16, 32, 63, 65, 128, usize::MAX] {
        assert!(
            matches!(AffineSpec::new(8, group), Err(OjasError::OutOfRange { .. })),
            "group {group}"
        );
    }
}

// --- quantiser refusals ---------------------------------------------------------------------

#[test]
fn non_finite_bf16_weight_is_refused_wherever_it_sits() {
    let b = big();
    for bad in [0x7FC0u16, 0x7F80, 0xFF80, 0x7F81, 0xFFC1] {
        for at in [0usize, 63, 64, 127] {
            let mut bits = bf16_bits(&vec![0.25f32; 128]);
            bits[at] = bad;
            let w = Tensor::from_bf16_bits(&bits, &[2, 64], &b).unwrap();
            let before = b.live_bytes().unwrap();
            let err = quantize_affine(&w, q8(), &b).unwrap_err();
            assert!(is_non_finite(&err), "{bad:#x} at {at}: {err}");
            assert_eq!(
                b.live_bytes().unwrap(),
                before,
                "{bad:#x} at {at}: nothing is left charged"
            );
        }
    }
}

#[test]
fn non_finite_f32_weight_is_refused() {
    let b = big();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut v = vec![0.5f32; 64];
        v[17] = bad;
        let w = Tensor::from_f32(&v, &[1, 64], &b).unwrap();
        let err = quantize_affine(&w, q8(), &b).unwrap_err();
        assert!(is_non_finite(&err), "{bad}: {err}");
    }
}

#[test]
fn f32_weight_that_is_not_bf16_exact_is_refused() {
    let b = big();
    for v in [0.1f32, 1.000_000_1, f32::from_bits(0x3F80_0001), 1e-45] {
        let w = Tensor::from_f32(&vec![v; 64], &[1, 64], &b).unwrap();
        let err = quantize_affine(&w, q8(), &b).unwrap_err();
        assert!(matches!(err, OjasError::OutOfRange { .. }), "{v}: {err}");
    }
    // The same values rounded to bf16 are accepted.
    let ok = Tensor::from_f32(&vec![bf16_to_f32(f32_to_bf16(0.1)); 64], &[1, 64], &b).unwrap();
    quantize_affine(&ok, q8(), &b).unwrap();
}

#[test]
fn wrong_dtype_rank_and_shape_are_refused() {
    let b = big();
    for dtype in [DType::F16, DType::U32] {
        let w = Tensor::zeros(&[1, 64], dtype, &b).unwrap();
        assert!(matches!(
            quantize_affine(&w, q8(), &b),
            Err(OjasError::Dtype { got, .. }) if got == dtype
        ));
    }
    for shape in [&[64usize][..], &[2, 2, 64][..], &[][..]] {
        let w = Tensor::zeros(shape, DType::Bf16, &b).unwrap();
        assert!(
            matches!(quantize_affine(&w, q8(), &b), Err(OjasError::Shape { .. })),
            "{shape:?}"
        );
    }
    for shape in [[0usize, 64], [3, 0], [0, 0]] {
        let w = Tensor::zeros(&shape, DType::Bf16, &b).unwrap();
        assert!(
            matches!(quantize_affine(&w, q8(), &b), Err(OjasError::Shape { .. })),
            "{shape:?}"
        );
    }
    for cols in [1usize, 63, 65, 100, 127, 130] {
        let w = Tensor::zeros(&[2, cols], DType::Bf16, &b).unwrap();
        assert!(
            matches!(quantize_affine(&w, q8(), &b), Err(OjasError::Shape { .. })),
            "cols {cols}"
        );
    }
}

#[test]
fn a_range_that_overflows_f32_is_refused_and_leaves_nothing_charged() {
    let b = big();
    let max = bf16_to_f32(0x7F7F);
    let mut v = vec![0.0f32; 128];
    // The second group overflows; the first is fine and has been quantised when it is found.
    v[64] = max;
    v[65] = -max;
    let w = bf16_matrix(&v, 1, 128, &b);
    let before = b.live_bytes().unwrap();
    let err = quantize_affine(&w, q8(), &b).unwrap_err();
    assert!(
        matches!(&err, OjasError::OutOfRange { detail, .. } if detail.contains("group 1")),
        "{err}"
    );
    assert_eq!(b.live_bytes().unwrap(), before);
}

#[test]
fn bf16_max_alone_quantises_and_round_trips() {
    // A constant +max or -max group has no range at all (scale = -0.0, codes via the NaN rule)
    // and dequantises back to itself.
    let b = big();
    let max = bf16_to_f32(0x7F7F);
    for v in [max, -max] {
        let w = bf16_matrix(&vec![v; 64], 1, 64, &b);
        let q = quantize_affine(&w, q8(), &b).unwrap();
        let d = dequantize_affine_bf16(&q, &b).unwrap();
        assert_eq!(d.bf16_slice().unwrap(), w.bf16_slice().unwrap(), "{v}");
    }
}

// --- budget ---------------------------------------------------------------------------------

fn quantised_bytes(rows: usize, cols: usize, bits: u64) -> (u64, u64, u64) {
    let packed = (rows * cols) as u64 * bits / 8;
    let sb = (rows * cols / 64) as u64 * 2;
    (packed, sb, packed + 2 * sb)
}

#[test]
fn outputs_are_charged_exactly_and_released_on_drop() {
    let (rows, cols) = (4usize, 256usize);
    for (spec, bits) in [(AffineSpec::q8(), 8u64), (AffineSpec::q4(), 4)] {
        let (_, _, total) = quantised_bytes(rows, cols, bits);
        let input_budget = big();
        let w = bf16_matrix(&vec![0.5f32; rows * cols], rows, cols, &input_budget);
        let b = Budget::new(total);
        let q = quantize_affine(&w, spec, &b).unwrap();
        assert_eq!(b.live_bytes().unwrap(), total);
        assert_eq!(q.nbytes().unwrap() as u64, total);
        drop(q);
        assert_eq!(b.live_bytes().unwrap(), 0);
    }
}

#[test]
fn a_budget_that_cannot_hold_the_outputs_refuses_with_nothing_charged() {
    let (rows, cols) = (4usize, 256usize);
    let input_budget = big();
    let w = bf16_matrix(&vec![0.5f32; rows * cols], rows, cols, &input_budget);
    let (packed, sb, total) = quantised_bytes(rows, cols, 8);
    // Short by one byte, short of everything, short after the packed codes, after the scales.
    for cap in [
        total - 1,
        0,
        packed - 1,
        packed,
        packed + sb,
        packed + sb + 1,
    ] {
        let b = Budget::new(cap);
        let err = quantize_affine(&w, q8(), &b).unwrap_err();
        assert!(
            matches!(err, OjasError::CapacityExceeded { .. }),
            "cap {cap}: {err}"
        );
        assert_eq!(b.live_bytes().unwrap(), 0, "cap {cap}");
    }
}

#[test]
fn dequantise_is_charged_before_it_allocates() {
    let input_budget = big();
    let w = bf16_matrix(&vec![0.5f32; 4 * 256], 4, 256, &input_budget);
    let q = quantize_affine(&w, q8(), &input_budget).unwrap();
    let need = 4 * 256 * 2;
    for cap in [0u64, need - 1] {
        let b = Budget::new(cap);
        let err = dequantize_affine_bf16(&q, &b).unwrap_err();
        assert!(
            matches!(err, OjasError::CapacityExceeded { .. }),
            "cap {cap}: {err}"
        );
        assert_eq!(b.live_bytes().unwrap(), 0);
    }
    let b = Budget::new(need);
    let d = dequantize_affine_bf16(&q, &b).unwrap();
    assert_eq!(b.live_bytes().unwrap(), need);
    drop(d);
    assert_eq!(b.live_bytes().unwrap(), 0);
}

// --- layout and the -0.0 decision ------------------------------------------------------------

#[test]
fn q8_codes_pack_four_per_word_little_endian() {
    // w = 4*i for i < 63, then 255: w_min 0, w_max 255, so scale -1, bias 255, code = 255 - w.
    let b = big();
    let mut v: Vec<f32> = (0..64).map(|i| 4.0 * i as f32).collect();
    v[63] = 255.0;
    let q = quantize_affine(&bf16_matrix(&v, 1, 64, &b), q8(), &b).unwrap();
    assert_eq!(q.scales().bf16_slice().unwrap(), &[f32_to_bf16(-1.0)]);
    assert_eq!(q.biases().bf16_slice().unwrap(), &[f32_to_bf16(255.0)]);
    let code = |i: usize| if i == 63 { 0u32 } else { 255 - 4 * i as u32 };
    let packed = q.packed().u32_slice().unwrap();
    assert_eq!(q.packed().shape(), &[1, 16]);
    for (j, &word) in packed.iter().enumerate() {
        let want = (0..4).fold(0u32, |acc, k| acc | code(4 * j + k) << (8 * k));
        assert_eq!(word, want, "word {j}");
    }
}

#[test]
fn q4_codes_pack_eight_per_word_little_endian() {
    // w = i % 16: w_min 0, w_max 15, so scale -1, bias 15, code = 15 - w.
    let b = big();
    let v: Vec<f32> = (0..64).map(|i| (i % 16) as f32).collect();
    let q = quantize_affine(&bf16_matrix(&v, 1, 64, &b), AffineSpec::q4(), &b).unwrap();
    assert_eq!(q.packed().shape(), &[1, 8]);
    let want = (0..8).fold(0u32, |acc, k| acc | (15 - k as u32) << (4 * k));
    assert_eq!(q.packed().u32_slice().unwrap()[0], want);
    assert_eq!(q.packed().u32_slice().unwrap()[1], {
        (0..8).fold(0u32, |acc, k| acc | (15 - (8 + k as u32)) << (4 * k))
    });
}

#[test]
fn values_under_half_the_scale_floor_give_code_zero_not_a_refusal() {
    // |w| < 5e-8 puts every group in the at_zero branch, where (w - 0) / +-1e-7 rounds to +-0.0.
    // That is the only way a negative pre-clamp value occurs; it is code 0, bias 0.
    let b = big();
    let floor_scale = f32_to_bf16(1e-7);
    for (hi, lo, scale_bits) in [
        (3e-8f32, -4e-8f32, floor_scale),    // |w_min| larger: scale +1e-7
        (4e-8, -3e-8, floor_scale | 0x8000), // |w_max| larger: scale -1e-7, positive w gives -0.0
    ] {
        let mut v = vec![0.0f32; 64];
        v[0] = hi;
        v[1] = lo;
        v[2] = 1.0e-8;
        v[3] = -1.0e-8;
        let q = quantize_affine(&bf16_matrix(&v, 1, 64, &b), q8(), &b).unwrap();
        assert!(
            q.packed().u32_slice().unwrap().iter().all(|&w| w == 0),
            "{hi} {lo}"
        );
        assert_eq!(q.biases().bf16_slice().unwrap(), &[0]);
        assert_eq!(q.scales().bf16_slice().unwrap(), &[scale_bits]);
        let d = dequantize_affine_bf16(&q, &b).unwrap();
        assert!(d.bf16_slice().unwrap().iter().all(|&x| x == 0));
    }
}

#[test]
fn a_negative_subnormal_beside_zeros_gives_a_negative_scale() {
    // The Apple GPU flushes the subnormal: |w_min| == |w_max| == 0, so side is false.
    let b = big();
    let sub = f32::from_bits(0x8001_0000); // bf16 0x8001, the smallest negative bf16 subnormal
    let v = one_group(&[(9, sub)], 0.0);
    let w = Tensor::from_bf16_bits(&bf16_bits(&v), &[1, 64], &b).unwrap();
    assert_eq!(w.bf16_slice().unwrap()[9], 0x8001);
    let q = quantize_affine(&w, q8(), &b).unwrap();
    assert_eq!(q.scales().bf16_slice().unwrap(), &[f32_to_bf16(-1e-7)]);
    assert_eq!(q.biases().bf16_slice().unwrap(), &[0]);
}

// --- determinism and the round-trip bound -----------------------------------------------------

#[test]
fn quantise_and_dequantise_are_deterministic() {
    let b = big();
    let mut rng = SplitMix64(11);
    let v: Vec<f32> = (0..8 * 512).map(|_| rng.normal() * 0.02).collect();
    let w = bf16_matrix(&v, 8, 512, &b);
    let a = quantize_affine(&w, q8(), &b).unwrap();
    let c = quantize_affine(&w, q8(), &b).unwrap();
    assert_eq!(
        a.packed().u32_slice().unwrap(),
        c.packed().u32_slice().unwrap()
    );
    assert_eq!(
        a.scales().bf16_slice().unwrap(),
        c.scales().bf16_slice().unwrap()
    );
    let d1 = dequantize_affine_bf16(&a, &b).unwrap();
    let d2 = dequantize_affine_bf16(&a, &b).unwrap();
    assert_eq!(d1.bf16_slice().unwrap(), d2.bf16_slice().unwrap());
}

#[test]
fn round_trip_error_stays_within_one_scale_plus_bf16_rounding() {
    let b = big();
    for seed in 0..48u64 {
        let mut rng = SplitMix64(seed.wrapping_mul(7919).wrapping_add(1));
        let rows = 1 + (rng.next() % 4) as usize;
        let cols = 64 * (1 + (rng.next() % 6) as usize);
        let magnitude = 10f32.powf(rng.unit() * 8.0 - 5.0);
        let offset = if seed.is_multiple_of(3) {
            magnitude * 3.0
        } else {
            0.0
        };
        let v: Vec<f32> = (0..rows * cols)
            .map(|_| rng.normal() * magnitude + offset)
            .collect();
        let w = bf16_matrix(&v, rows, cols, &b);
        let wv = w.bf16_slice().unwrap();
        for spec in [AffineSpec::q8(), AffineSpec::q4()] {
            let levels = spec.bits().max_code() as f32;
            let q = quantize_affine(&w, spec, &b).unwrap();
            let d = dequantize_affine_bf16(&q, &b).unwrap();
            let (dv, sv, bv) = (
                d.bf16_slice().unwrap(),
                q.scales().bf16_slice().unwrap(),
                q.biases().bf16_slice().unwrap(),
            );
            for (i, (&got, &orig)) in dv.iter().zip(wv).enumerate() {
                let g = i / 64;
                let (s, bias) = (bf16_to_f32(sv[g]).abs(), bf16_to_f32(bv[g]).abs());
                let (got, orig) = (bf16_to_f32(got), bf16_to_f32(orig));
                // One code step: mlx rounds the edge code q0, which can put the far end of the
                // range at n_bins + 1 before the clamp drops it by one (q4 hits this on 25 of these
                // 48 seeds; the goldens show mlx does the same). Plus the bf16 scale's relative
                // error (2^-9) times the largest code, and the bias's and the result's bf16 rounding.
                let bound = s * (1.0 + levels / 512.0) + (bias + orig.abs()) / 128.0;
                assert!(
                    (got - orig).abs() <= bound,
                    "seed {seed} {:?} element {i}: |{got} - {orig}| > {bound}",
                    spec.bits()
                );
            }
        }
    }
}

// --- from_parts -----------------------------------------------------------------------------

struct Parts {
    packed: Tensor,
    scales: Tensor,
    biases: Tensor,
}

/// Valid q8 parts for a `[2, 128]` matrix.
fn parts(b: &Budget) -> Parts {
    Parts {
        packed: Tensor::zeros(&[2, 32], DType::U32, b).unwrap(),
        scales: Tensor::from_bf16_bits(&[0x3F80; 4], &[2, 2], b).unwrap(),
        biases: Tensor::from_bf16_bits(&[0; 4], &[2, 2], b).unwrap(),
    }
}

fn build(p: Parts, rows: usize, cols: usize) -> Result<AffineQuantized, OjasError> {
    AffineQuantized::from_parts(q8(), rows, cols, p.packed, p.scales, p.biases)
}

#[test]
fn from_parts_accepts_valid_parts_and_signed_zero() {
    let b = big();
    let q = build(parts(&b), 2, 128).unwrap();
    assert_eq!((q.rows(), q.cols(), q.spec()), (2, 128, q8()));
    assert_eq!(q.nbytes().unwrap(), 2 * 32 * 4 + 2 * 4 * 2);
    let mut p = parts(&b);
    p.scales = Tensor::from_bf16_bits(&[0x8000, 0, 0x8000, 0], &[2, 2], &b).unwrap();
    build(p, 2, 128).unwrap();
    let (packed, scales, biases) = build(parts(&b), 2, 128).unwrap().into_parts();
    assert_eq!(
        (packed.dtype(), scales.dtype(), biases.dtype()),
        (DType::U32, DType::Bf16, DType::Bf16)
    );
}

#[test]
fn from_parts_refuses_every_malformed_part() {
    let b = big();
    let z = |shape: &[usize], dtype| Tensor::zeros(shape, dtype, &b).unwrap();

    // dtype
    let mut p = parts(&b);
    p.packed = z(&[2, 32], DType::F32);
    assert!(matches!(build(p, 2, 128), Err(OjasError::Dtype { .. })));
    let mut p = parts(&b);
    p.scales = z(&[2, 2], DType::F32);
    assert!(matches!(build(p, 2, 128), Err(OjasError::Dtype { .. })));
    let mut p = parts(&b);
    p.biases = z(&[2, 2], DType::F16);
    assert!(matches!(build(p, 2, 128), Err(OjasError::Dtype { .. })));
    let mut p = parts(&b);
    p.packed = z(&[2, 32], DType::Bf16);
    assert!(matches!(build(p, 2, 128), Err(OjasError::Dtype { .. })));

    // shape
    for shape in [
        &[2usize, 31][..],
        &[2, 33],
        &[32, 2],
        &[64],
        &[1, 2, 32],
        &[2, 64],
    ] {
        let mut p = parts(&b);
        p.packed = z(shape, DType::U32);
        assert!(
            matches!(build(p, 2, 128), Err(OjasError::Shape { .. })),
            "packed {shape:?}"
        );
    }
    for shape in [&[2usize, 1][..], &[2, 3], &[4], &[1, 2], &[2, 2, 1]] {
        let mut p = parts(&b);
        p.scales = z(shape, DType::Bf16);
        assert!(
            matches!(build(p, 2, 128), Err(OjasError::Shape { .. })),
            "scales {shape:?}"
        );
        let mut p = parts(&b);
        p.biases = z(shape, DType::Bf16);
        assert!(
            matches!(build(p, 2, 128), Err(OjasError::Shape { .. })),
            "biases {shape:?}"
        );
    }

    // rows / cols
    for (rows, cols) in [
        (0usize, 128usize),
        (2, 0),
        (2, 100),
        (2, 127),
        (3, 128),
        (2, 192),
    ] {
        assert!(
            matches!(build(parts(&b), rows, cols), Err(OjasError::Shape { .. })),
            "{rows}x{cols}"
        );
    }
    // overflowing geometry is a refusal, not a panic
    assert!(build(parts(&b), usize::MAX, 128).is_err());
    assert!(build(parts(&b), 2, usize::MAX - 63).is_err());

    // a 4-bit spec with 8-bit packed shape
    let p = parts(&b);
    assert!(
        AffineQuantized::from_parts(AffineSpec::q4(), 2, 128, p.packed, p.scales, p.biases)
            .is_err()
    );
}

#[test]
fn from_parts_refuses_non_finite_scales_and_biases() {
    let b = big();
    for bad in [0x7F80u16, 0xFF80, 0x7FC0, 0xFFC1, 0x7F81] {
        for at in [0usize, 3] {
            let mut bits = [0x3F80u16; 4];
            bits[at] = bad;
            let mut p = parts(&b);
            p.scales = Tensor::from_bf16_bits(&bits, &[2, 2], &b).unwrap();
            assert!(
                is_non_finite(&build(p, 2, 128).unwrap_err()),
                "scale {bad:#x}@{at}"
            );
            let mut p = parts(&b);
            p.biases = Tensor::from_bf16_bits(&bits, &[2, 2], &b).unwrap();
            assert!(
                is_non_finite(&build(p, 2, 128).unwrap_err()),
                "bias {bad:#x}@{at}"
            );
        }
    }
}

#[test]
fn dequantise_is_fused_at_the_overflow_edge() {
    // scale 2^125, code 8, bias -bf16::MAX: 2^125 * 8 = 2^128 overflows f32 unfused, the fma is
    // finite. mlx (mx.dequantize, stream=mx.gpu, mlx 0.31.2) returns the finite bf16 0x7B80.
    let b = big();
    let mut p = parts(&b);
    p.packed = Tensor::from_u32(&[0x0808_0808; 64], &[2, 32], &b).unwrap();
    p.scales = Tensor::from_bf16_bits(&[f32_to_bf16(2f32.powi(125)); 4], &[2, 2], &b).unwrap();
    p.biases = Tensor::from_bf16_bits(&[0xFF7F; 4], &[2, 2], &b).unwrap();
    let q = build(p, 2, 128).unwrap();
    let d = dequantize_affine_bf16(&q, &b).unwrap();
    assert!(d.bf16_slice().unwrap().iter().all(|&x| x == 0x7B80));
}

#[test]
fn dequantise_refuses_parts_whose_result_overflows_bf16() {
    let b = big();
    let mut p = parts(&b);
    p.packed = Tensor::from_u32(&[u32::MAX; 64], &[2, 32], &b).unwrap();
    p.scales = Tensor::from_bf16_bits(&[0x7F7F; 4], &[2, 2], &b).unwrap();
    p.biases = Tensor::from_bf16_bits(&[0x7F7F; 4], &[2, 2], &b).unwrap();
    let q = build(p, 2, 128).unwrap();
    let before = b.live_bytes().unwrap();
    let err = dequantize_affine_bf16(&q, &b).unwrap_err();
    assert!(is_non_finite(&err), "{err}");
    assert_eq!(b.live_bytes().unwrap(), before);
}

// --- throughput (a measurement, not a gate) -------------------------------------------------------

#[test]
#[ignore = "throughput measurement: run with --ignored --nocapture; prints elements/s"]
fn throughput_on_the_2b_linear_shapes() {
    let b = Budget::new(1 << 34);
    let mut rng = SplitMix64(5);
    for (rows, cols) in [(6144usize, 2048usize), (2048, 6144)] {
        let v: Vec<f32> = (0..rows * cols).map(|_| rng.normal() * 0.02).collect();
        let w = bf16_matrix(&v, rows, cols, &b);
        let elems = (rows * cols) as f64;
        let (mut qbest, mut dbest) = (f64::MAX, f64::MAX);
        for _ in 0..5 {
            let t = Instant::now();
            let q = quantize_affine(&w, q8(), &b).unwrap();
            qbest = qbest.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            let d = dequantize_affine_bf16(&q, &b).unwrap();
            dbest = dbest.min(t.elapsed().as_secs_f64());
            assert_eq!(d.shape(), &[rows, cols]);
        }
        println!(
            "[{rows}, {cols}] q8 quantise {:.1} Melem/s ({:.1} ms), dequantise {:.1} Melem/s ({:.1} ms) (min of 5)",
            elems / qbest / 1e6,
            qbest * 1e3,
            elems / dbest / 1e6,
            dbest * 1e3
        );
    }
}
