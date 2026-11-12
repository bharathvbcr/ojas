//! `BackendModule::dequant_affine_bf16` on Metal (tessl's `dequant_affine_{q4,q8}_bf16`): bit-equal
//! to the CPU op and to mlx's goldens, the deferred fault word reported as `NonFinite` under the
//! op's own name, and the refusals. Run with `--test-threads=1`.
//!
//! The CPU op is the reference (`ojas_cpu` runs `ojas_core::dequantize_affine_bf16_tensors`, which
//! the mlx goldens pin on the host).

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;
// The goldens are read by ojas-core's reader, included by path: no dependency, no Cargo.lock change.
#[path = "../../ojas-core/tests/mlx_affine_common/mod.rs"]
#[allow(dead_code)]
mod mlx_affine_common;

use std::path::Path;

use common::{metal, GIB};
use ojas_core::{
    dequant_affine_bf16_dims, f32_to_bf16, quantize_affine, AffineSpec, BackendCore, BackendFloat,
    BackendModule, Budget, DType, OjasError, Tensor,
};
use ojas_cpu::CpuBackend;
use ojas_metal::{MetalBackend, WaitCounts};

const OP: &str = "dequant_affine_bf16";
const RING: usize = 64;

fn golden_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../ojas-core/tests/fixtures/mlx_affine")
}

fn host() -> Budget {
    Budget::new(8 * GIB)
}

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(8 * GIB))
}

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

/// Three host tensors: `packed` `U32 [rows, words]`, `scales`, `biases` `Bf16 [rows, groups]`.
struct Parts {
    packed: Tensor,
    scales: Tensor,
    biases: Tensor,
    spec: AffineSpec,
}

fn host_parts(
    spec: AffineSpec,
    rows: usize,
    groups: usize,
    p: &[u32],
    s: &[u16],
    b: &[u16],
) -> Parts {
    let words = groups * spec.words_per_group();
    let h = host();
    Parts {
        packed: Tensor::from_u32(p, &[rows, words], &h).unwrap(),
        scales: Tensor::from_bf16_bits(s, &[rows, groups], &h).unwrap(),
        biases: Tensor::from_bf16_bits(b, &[rows, groups], &h).unwrap(),
        spec,
    }
}

impl Parts {
    fn up(&self, m: &MetalBackend) -> Parts {
        Parts {
            packed: m.upload(&self.packed).unwrap(),
            scales: m.upload(&self.scales).unwrap(),
            biases: m.upload(&self.biases).unwrap(),
            spec: self.spec,
        }
    }

    fn cpu(&self, c: &CpuBackend) -> Vec<u16> {
        let out = c
            .dequant_affine_bf16(&self.packed, &self.scales, &self.biases, self.spec)
            .unwrap();
        out.bf16_slice().unwrap().to_vec()
    }

    fn run(&self, m: &MetalBackend) -> Result<Tensor, OjasError> {
        m.dequant_affine_bf16(&self.packed, &self.scales, &self.biases, self.spec)
    }
}

fn bits_of(t: &Tensor) -> Vec<u16> {
    assert_eq!(t.dtype(), DType::Bf16);
    t.to_host(&host()).unwrap().bf16_slice().unwrap().to_vec()
}

fn expect_same(what: &str, got: &[u16], want: &[u16]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(g, w)| g != w) {
        let n = got.iter().zip(want).filter(|(g, w)| g != w).count();
        panic!(
            "{what}: {n} of {} differ, first at {i}: got {:#06x}, want {:#06x}",
            got.len(),
            got[i],
            want[i]
        );
    }
}

/// A sync must report `NonFinite` naming `op`, once.
fn assert_fault(m: &MetalBackend, what: &str, op: &str) {
    match m.sync() {
        Err(OjasError::NonFinite { op: got }) => assert_eq!(got, op, "{what}"),
        other => panic!("{what}: expected NonFinite {{ {op} }}, got {other:?}"),
    }
    m.sync()
        .unwrap_or_else(|e| panic!("{what}: a fault is reported once: {e:?}"));
}

// --- parity ------------------------------------------------------------------------------------

/// The five distinct `[out, in]` shapes of Qwen3.5-2B's wrapped linears
/// (`mlx_affine/manifest.json`, `qwen35_2b_base.distinct_shapes_out_in`).
const SHAPES_2B: [(usize, usize); 5] = [
    (512, 2048),
    (2048, 2048),
    (2048, 6144),
    (4096, 2048),
    (6144, 2048),
];

fn quantized_parts(spec: AffineSpec, rows: usize, cols: usize, seed: u64) -> Parts {
    let mut rng = SplitMix64(seed);
    let w: Vec<u16> = (0..rows * cols)
        .map(|_| f32_to_bf16(rng.normal() * 0.05))
        .collect();
    let weight = Tensor::from_bf16_bits(&w, &[rows, cols], &host()).unwrap();
    let q = quantize_affine(&weight, spec, &host()).unwrap();
    let (packed, scales, biases) = q.into_parts();
    Parts {
        packed,
        scales,
        biases,
        spec,
    }
}

/// Every bit of every packed word random, so every code value occurs; finite scales and biases
/// of magnitude under 1, so every result is finite.
fn random_code_parts(spec: AffineSpec, rows: usize, cols: usize, seed: u64) -> Parts {
    let mut rng = SplitMix64(seed);
    let groups = cols / 64;
    let p: Vec<u32> = (0..rows * groups * spec.words_per_group())
        .map(|_| rng.next() as u32)
        .collect();
    let s: Vec<u16> = (0..rows * groups)
        .map(|_| f32_to_bf16((rng.unit() - 0.5) * 2.0))
        .collect();
    let b: Vec<u16> = (0..rows * groups)
        .map(|_| f32_to_bf16((rng.unit() - 0.5) * 2.0))
        .collect();
    host_parts(spec, rows, groups, &p, &s, &b)
}

#[test]
fn metal_equals_cpu_bit_for_bit_on_the_five_2b_shapes() {
    let m = metal();
    let c = cpu();
    for (i, &(rows, cols)) in SHAPES_2B.iter().enumerate() {
        for spec in [AffineSpec::q8(), AffineSpec::q4()] {
            for (kind, parts) in [
                (
                    "quantised",
                    quantized_parts(spec, rows, cols, 100 + i as u64),
                ),
                (
                    "random codes",
                    random_code_parts(spec, rows, cols, 200 + i as u64),
                ),
            ] {
                let tag = format!("[{rows}, {cols}] {:?} {kind}", spec.bits());
                let want = parts.cpu(&c);
                let dev = parts.up(&m);
                let out = dev.run(&m).unwrap_or_else(|e| panic!("{tag}: {e:?}"));
                assert_eq!(out.shape(), [rows, cols], "{tag}");
                m.sync().unwrap_or_else(|e| panic!("{tag}: sync {e:?}"));
                expect_same(&tag, &bits_of(&out), &want);
            }
        }
    }
}

#[test]
fn metal_equals_the_mlx_goldens_and_cpu() {
    let m = metal();
    let c = cpu();
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
            let parts = host_parts(spec, wq_shape[0], sb_shape[1], wq, scales, biases);
            expect_same(&format!("{tag} cpu"), &parts.cpu(&c), want);
            let out = parts
                .up(&m)
                .run(&m)
                .unwrap_or_else(|e| panic!("{tag}: {e:?}"));
            m.sync().unwrap_or_else(|e| panic!("{tag}: sync {e:?}"));
            expect_same(&format!("{tag} metal"), &bits_of(&out), want);
        }
    }
}

/// scale 2^125, code 8, bias -bf16::MAX: the unfused product overflows f32, the fma is finite.
#[test]
fn fused_overflow_edge_matches_cpu() {
    let m = metal();
    let c = cpu();
    let s = vec![f32_to_bf16(2f32.powi(125)); 2];
    let parts = host_parts(AffineSpec::q8(), 2, 1, &[0x0808_0808; 32], &s, &[0xFF7F; 2]);
    let want = parts.cpu(&c);
    assert!(
        want.iter().all(|&x| x == 0x7B80),
        "the CPU reference moved: {want:#x?}"
    );
    let out = parts.up(&m).run(&m).unwrap();
    m.sync().expect("the fused result is finite, so no fault");
    expect_same("fused edge", &bits_of(&out), &want);
}

#[test]
fn q4_lanes_decode_in_order() {
    let m = metal();
    let parts = host_parts(AffineSpec::q4(), 1, 1, &[0x7654_3210; 8], &[0x3F80], &[0]);
    let out = parts.up(&m).run(&m).unwrap();
    m.sync().unwrap();
    let want: Vec<u16> = (0..64).map(|i| f32_to_bf16((i % 8) as f32)).collect();
    expect_same("q4 lanes", &bits_of(&out), &want);
    // q8: four lanes per word, 0x04030201 gives 1, 2, 3, 4.
    let parts = host_parts(AffineSpec::q8(), 1, 1, &[0x0403_0201; 16], &[0x3F80], &[0]);
    let out = parts.up(&m).run(&m).unwrap();
    m.sync().unwrap();
    let want: Vec<u16> = (0..64).map(|i| f32_to_bf16((i % 4 + 1) as f32)).collect();
    expect_same("q8 lanes", &bits_of(&out), &want);
}

// --- faults ------------------------------------------------------------------------------------

/// Scale `bf16::MAX`, code 255, bias `bf16::MAX`: finite operands, an overflowing result.
fn overflowing(spec: AffineSpec) -> Parts {
    host_parts(
        spec,
        1,
        1,
        &vec![u32::MAX; spec.words_per_group()],
        &[0x7F7F],
        &[0x7F7F],
    )
}

fn good(spec: AffineSpec) -> Parts {
    host_parts(
        spec,
        1,
        1,
        &vec![0; spec.words_per_group()],
        &[0x3F80],
        &[0x3F00],
    )
}

#[test]
fn nonfinite_result_is_nonfinite_at_sync_under_dequant_affine_bf16() {
    for spec in [AffineSpec::q8(), AffineSpec::q4()] {
        let m = metal();
        let dev = overflowing(spec).up(&m);
        let out = dev
            .run(&m)
            .expect("a device-detected fault does not fail its op");
        assert_eq!(out.shape(), [1, 64]);
        // `download` is a sync point: it reports the pending fault.
        match m.download(&out) {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, OP, "{spec:?}"),
            other => panic!("{spec:?}: expected NonFinite, got {other:?}"),
        }
        m.sync().expect("reported once");
    }
}

/// Fast math must not elide the check: a NaN scale, an infinite scale at code 0 (`inf * 0` is NaN)
/// and an infinite bias each fault.
#[test]
fn nan_scale_inf_scale_at_code_0_and_inf_bias_fault() {
    let m = metal();
    for (name, scale, bias) in [
        ("NaN scale", 0x7FC0u16, 0u16),
        ("+inf scale at code 0", 0x7F80, 0),
        ("-inf scale at code 0", 0xFF80, 0),
        ("+inf bias", 0x3F80, 0x7F80),
        ("-inf bias", 0x3F80, 0xFF80),
        ("NaN bias", 0x3F80, 0x7FC0),
    ] {
        for spec in [AffineSpec::q8(), AffineSpec::q4()] {
            let p = host_parts(
                spec,
                1,
                1,
                &vec![0; spec.words_per_group()],
                &[scale],
                &[bias],
            );
            p.up(&m)
                .run(&m)
                .unwrap_or_else(|e| panic!("{name}: the op must record, not refuse: {e:?}"));
            assert_fault(&m, name, OP);
        }
    }
}

#[test]
fn fault_is_attributed_in_recording_order() {
    // An earlier faulting op is reported first, whichever op it is.
    let m = metal();
    let nan = common::up(&m, &common::host(&[1.0, f32::NAN], &[2]));
    m.silu_forward(&nan).expect("silu records");
    overflowing(AffineSpec::q8()).up(&m).run(&m).unwrap();
    assert_fault(&m, "silu first", "silu_forward");

    let m = metal();
    let nan = common::up(&m, &common::host(&[1.0, f32::NAN], &[2]));
    overflowing(AffineSpec::q8()).up(&m).run(&m).unwrap();
    m.silu_forward(&nan).expect("silu records");
    assert_fault(&m, "dequant first", OP);
}

#[test]
fn a_faulted_word_is_zeroed_before_reuse() {
    let m = metal();
    let (bad, ok) = (
        overflowing(AffineSpec::q8()).up(&m),
        good(AffineSpec::q8()).up(&m),
    );
    let mut outs = Vec::new();
    // 130 dequants, the bad one early, no scan between: two forced settles free every word.
    for i in 0..130 {
        let p = if i == 3 { &bad } else { &ok };
        outs.push(p.run(&m).unwrap());
    }
    assert_fault(&m, "one bad dequant among 130", OP);
    // The words the bad one raised were zeroed on the way back: 130 clean ones report nothing,
    // across the same forced settles.
    outs.clear();
    for _ in 0..130 {
        outs.push(ok.run(&m).unwrap());
    }
    m.sync()
        .expect("a stale fault word reported a clean dequant");
}

fn ring_waits(m: &MetalBackend, before: &WaitCounts) -> u64 {
    m.wait_counts().since(before).fault_ring
}

#[test]
fn sixty_five_pending_dequants_force_a_settle_and_all_complete() {
    let m = metal();
    let c = cpu();
    let p = good(AffineSpec::q8());
    let want = p.cpu(&c);
    let dev = p.up(&m);
    m.sync().unwrap();
    let before = m.wait_counts();
    let mut outs = Vec::new();
    for _ in 0..RING {
        outs.push(dev.run(&m).unwrap());
    }
    assert_eq!(ring_waits(&m, &before), 0, "{RING} dequants fit the ring");
    outs.push(dev.run(&m).unwrap());
    assert_eq!(ring_waits(&m, &before), 1, "the 65th forces one settle");
    m.sync().unwrap();
    for (i, o) in outs.iter().enumerate() {
        expect_same(&format!("dequant {i}"), &bits_of(o), &want);
    }
}

/// The dequant that forces the settle takes its word only after the settle, so its own fault is
/// not drained and lost with the others.
#[test]
fn the_dequant_that_forces_the_settle_can_itself_fault() {
    let m = metal();
    let (bad, ok) = (
        overflowing(AffineSpec::q8()).up(&m),
        good(AffineSpec::q8()).up(&m),
    );
    m.sync().unwrap();
    let before = m.wait_counts();
    let mut outs: Vec<Tensor> = (0..RING).map(|_| ok.run(&m).unwrap()).collect();
    outs.push(bad.run(&m).unwrap());
    assert_eq!(ring_waits(&m, &before), 1, "the 65th forced the settle");
    assert_fault(&m, "the 65th of 65", OP);
}

// --- refusals ----------------------------------------------------------------------------------

#[test]
fn a_host_operand_is_placement() {
    let m = metal();
    let p = good(AffineSpec::q8());
    let dev = p.up(&m);
    for (name, packed, scales, biases) in [
        ("packed", &p.packed, &dev.scales, &dev.biases),
        ("scales", &dev.packed, &p.scales, &dev.biases),
        ("biases", &dev.packed, &dev.scales, &p.biases),
    ] {
        let r = m.dequant_affine_bf16(packed, scales, biases, p.spec);
        assert!(
            matches!(r, Err(OjasError::Placement { op, .. }) if op == OP),
            "{name}: {r:?}"
        );
    }
    m.sync().unwrap();
}

#[test]
fn a_foreign_metal_device_is_refused() {
    let a = metal();
    let b = metal();
    let p = good(AffineSpec::q8());
    let on_a = p.up(&a);
    let r = on_a.run(&b);
    assert!(
        matches!(&r, Err(OjasError::Backend { detail, .. }) if detail.contains("different MetalBackend")),
        "{r:?}"
    );
}

/// tessl binds its operands at offset 0: a window into a larger buffer is refused, not copied
/// through a float kernel.
#[test]
fn an_offset_window_is_unsupported() {
    let m = metal();
    let h = host();
    // Two rows uploaded as one buffer; the second row is a window at byte offset 64.
    let big = m
        .upload(&Tensor::from_u32(&[0; 32], &[2, 16], &h).unwrap())
        .unwrap();
    let row = big.narrow(16 * 4, &[1, 16], &[16, 1]).unwrap();
    assert_eq!(row.byte_offset(), 64);
    let ok = good(AffineSpec::q8()).up(&m);
    let r = m.dequant_affine_bf16(&row, &ok.scales, &ok.biases, AffineSpec::q8());
    match r {
        Err(OjasError::Unsupported { op, detail }) => {
            assert_eq!(op, OP);
            assert!(
                detail.contains("offset 0") && detail.contains("byte 64"),
                "{detail}"
            );
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
    let big_scales = m
        .upload(&Tensor::from_bf16_bits(&[0x3F80; 2], &[2, 1], &h).unwrap())
        .unwrap();
    let window = big_scales.narrow(2, &[1, 1], &[1, 1]).unwrap();
    let r = m.dequant_affine_bf16(&ok.packed, &window, &ok.biases, AffineSpec::q8());
    assert!(matches!(r, Err(OjasError::Unsupported { .. })), "{r:?}");
}

#[test]
fn a_budget_one_byte_short_is_capacity_exceeded_with_nothing_charged() {
    let pool = metal();
    let (rows, groups) = (3usize, 2usize);
    let spec = AffineSpec::q8();
    let n = rows * groups;
    let p = host_parts(
        spec,
        rows,
        groups,
        &vec![0; rows * groups * spec.words_per_group()],
        &vec![0x3F80; n],
        &vec![0; n],
    );
    let dev = p.up(&pool);
    let out_bytes = (rows * groups * 64 * 2) as u64;
    let short = pool.with_budget(Budget::new(out_bytes - 1));
    let r = dev.run(&short);
    assert!(
        matches!(r, Err(OjasError::CapacityExceeded { .. })),
        "{r:?}"
    );
    assert_eq!(short.budget().live_bytes().unwrap(), 0, "a refusal charged");
    short.sync().unwrap();
    let exact = pool.with_budget(Budget::new(out_bytes));
    let out = dev.run(&exact).unwrap();
    assert_eq!(exact.budget().live_bytes().unwrap(), out_bytes);
    drop(out);
    exact.sync().unwrap();
    assert_eq!(exact.budget().live_bytes().unwrap(), 0);
}

#[test]
fn a_malformed_triple_returns_the_validators_error() {
    let m = metal();
    let h = host();
    let spec = AffineSpec::q8();
    let up = |t: Tensor| m.upload(&t).unwrap();
    let packed = up(Tensor::from_u32(&[0; 30], &[2, 15], &h).unwrap());
    let scales = up(Tensor::from_bf16_bits(&[0x3F80; 4], &[2, 2], &h).unwrap());
    let biases = up(Tensor::from_bf16_bits(&[0; 4], &[2, 2], &h).unwrap());
    let want = dequant_affine_bf16_dims(&packed, &scales, &biases, spec).unwrap_err();
    let got = m
        .dequant_affine_bf16(&packed, &scales, &biases, spec)
        .unwrap_err();
    assert_eq!(format!("{got:?}"), format!("{want:?}"));
    // A q4 triple presented as q8 is that same refusal, naming the width its words imply.
    let q4 = good(AffineSpec::q4()).up(&m);
    let r = m.dequant_affine_bf16(&q4.packed, &q4.scales, &q4.biases, AffineSpec::q8());
    assert!(
        matches!(&r, Err(OjasError::Shape { detail, .. }) if detail.contains("implies 4-bit")),
        "{r:?}"
    );
    m.sync().unwrap();
}

/// The CPU op and this one refuse the same malformed call with the same error.
#[test]
fn cpu_and_metal_agree_on_every_refusal_of_the_validator() {
    let m = metal();
    let c = cpu();
    let h = host();
    let wrong = Tensor::from_f32(&[0.0; 32], &[2, 16], &h).unwrap();
    let scales = Tensor::from_bf16_bits(&[0x3F80; 4], &[2, 2], &h).unwrap();
    let on = |t: &Tensor| m.upload(t).unwrap();
    let want = c
        .dequant_affine_bf16(&wrong, &scales, &scales, AffineSpec::q8())
        .unwrap_err();
    let got = m
        .dequant_affine_bf16(&on(&wrong), &on(&scales), &on(&scales), AffineSpec::q8())
        .unwrap_err();
    assert_eq!(format!("{got:?}"), format!("{want:?}"));
}
