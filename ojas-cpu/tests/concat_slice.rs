//! `Backend::concat` and `Backend::slice` on the CPU.
//!
//! - Values are held to a coordinate-by-coordinate reference written here,
//!   bit for bit, on inputs of random finite bit patterns (every exponent,
//!   subnormals, both zeros), at every axis of ranks 1 to 4.
//! - Slicing what a concat joined gives the parts back, bit for bit, at the
//!   shapes SpikeTrack's memory assembly uses.
//! - Every refusal, a non-finite input (refused as `permute` refuses it, not
//!   passed through), a view and an over-budget output use the variants the
//!   neighbouring ops use, and a refused call leaves the budget as it was.
//! - Under bf16 autocast neither op rounds or computes; the compute tag
//!   follows the data.

use ojas_core::{Autocast, AutocastMode, Backend, Budget, DType, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_capacity, assert_nonfinite, assert_range, assert_shape, bits, SplitMix64};

fn backend(threads: usize) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 32), threads)
        .unwrap()
        .with_numerics(Numerics::Exact)
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

/// The finite values worth moving: both zeros, the smallest subnormal and
/// normal, and the extremes.
const SPECIAL: [f32; 8] = [
    0.0,
    -0.0,
    f32::MIN_POSITIVE,
    -f32::MIN_POSITIVE,
    1e-45,
    -1e-45,
    f32::MAX,
    f32::MIN,
];

/// `n` finite values: the specials first, then random bit patterns with any
/// exponent that is not all ones.
fn finite_bits(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
    let mut out: Vec<f32> = SPECIAL.iter().copied().take(n).collect();
    while out.len() < n {
        let f = f32::from_bits(rng.next_u64() as u32);
        if f.is_finite() {
            out.push(f);
        }
    }
    out
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Row-major coordinates of flat index `i` in `shape`.
fn coords(mut i: usize, shape: &[usize]) -> Vec<usize> {
    let mut c = vec![0; shape.len()];
    for a in (0..shape.len()).rev() {
        c[a] = i % shape[a];
        i /= shape[a];
    }
    c
}

fn flat(c: &[usize], shape: &[usize]) -> usize {
    c.iter().zip(shape).fold(0, |acc, (&x, &d)| acc * d + x)
}

/// `torch.cat` by coordinates: output element `c` comes from the part that
/// holds `c[axis]`, at the local coordinate.
fn ref_concat(parts: &[(Vec<f32>, Vec<usize>)], axis: usize) -> (Vec<f32>, Vec<usize>) {
    let mut out_shape = parts[0].1.clone();
    out_shape[axis] = parts.iter().map(|(_, s)| s[axis]).sum();
    let out = (0..numel(&out_shape))
        .map(|i| {
            let mut c = coords(i, &out_shape);
            let mut at = c[axis];
            for (data, shape) in parts {
                if at < shape[axis] {
                    c[axis] = at;
                    return data[flat(&c, shape)];
                }
                at -= shape[axis];
            }
            unreachable!("axis coordinate past the parts")
        })
        .collect();
    (out, out_shape)
}

/// `narrow(axis, start, len)` by coordinates.
fn ref_slice(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    start: usize,
    len: usize,
) -> (Vec<f32>, Vec<usize>) {
    let mut out_shape = shape.to_vec();
    out_shape[axis] = len;
    let out = (0..numel(&out_shape))
        .map(|i| {
            let mut c = coords(i, &out_shape);
            c[axis] += start;
            data[flat(&c, shape)]
        })
        .collect();
    (out, out_shape)
}

#[test]
fn concat_matches_the_coordinate_reference_bit_for_bit_on_every_axis() {
    let cpu = backend(2);
    let mut rng = SplitMix64(1);
    // (shape of input 0, extents of each input along the axis, per axis).
    let cases: [&[usize]; 5] = [&[7], &[3, 5], &[2, 3, 4], &[2, 1, 3, 5], &[1, 1, 1]];
    for shape in cases {
        for axis in 0..shape.len() {
            for lens in [&[1usize][..], &[2, 1], &[3, 1, 4], &[1, 1, 1, 1, 2]] {
                let parts: Vec<(Vec<f32>, Vec<usize>)> = lens
                    .iter()
                    .map(|&l| {
                        let mut s = shape.to_vec();
                        s[axis] = l;
                        (finite_bits(&mut rng, numel(&s)), s)
                    })
                    .collect();
                let tensors: Vec<Tensor> = parts.iter().map(|(d, s)| t(d, s)).collect();
                let refs: Vec<&Tensor> = tensors.iter().collect();
                let got = cpu.concat(&refs, axis).unwrap();
                let (want, want_shape) = ref_concat(&parts, axis);
                assert_eq!(
                    got.shape(),
                    want_shape.as_slice(),
                    "{shape:?} axis {axis} {lens:?}"
                );
                assert_eq!(got.dtype(), DType::F32);
                assert_eq!(
                    bits(&v(&got)),
                    bits(&want),
                    "{shape:?} axis {axis} {lens:?}"
                );
            }
        }
    }
}

#[test]
fn slice_matches_the_coordinate_reference_for_every_window() {
    let cpu = backend(2);
    let mut rng = SplitMix64(2);
    for shape in [&[7usize][..], &[3, 5], &[2, 3, 4], &[2, 1, 3, 2]] {
        let data = finite_bits(&mut rng, numel(shape));
        let x = t(&data, shape);
        for axis in 0..shape.len() {
            for start in 0..shape[axis] {
                for len in 1..=shape[axis] - start {
                    let got = cpu.slice(&x, axis, start, len).unwrap();
                    let (want, want_shape) = ref_slice(&data, shape, axis, start, len);
                    let at = format!("{shape:?} axis {axis} {start}+{len}");
                    assert_eq!(got.shape(), want_shape.as_slice(), "{at}");
                    assert_eq!(bits(&v(&got)), bits(&want), "{at}");
                }
            }
        }
    }
}

#[test]
fn the_specials_survive_both_ops_with_their_sign_and_payload() {
    let cpu = backend(1);
    let x = t(&SPECIAL, &[2, 4]);
    let joined = cpu.concat(&[&x, &x], 1).unwrap();
    let want: Vec<u32> = (0..2)
        .flat_map(|r| {
            (0..2).flat_map(move |_| SPECIAL[r * 4..r * 4 + 4].iter().map(|f| f.to_bits()))
        })
        .collect();
    assert_eq!(bits(&v(&joined)), want);
    let cut = cpu.slice(&joined, 1, 3, 2).unwrap();
    assert_eq!(
        bits(&v(&cut)),
        [SPECIAL[3], SPECIAL[0], SPECIAL[7], SPECIAL[4]].map(f32::to_bits)
    );
    // -0.0 stays -0.0 and 0.0 stays 0.0: the bits say so.
    assert_eq!(
        v(&cpu.slice(&x, 1, 0, 2).unwrap())[1].to_bits(),
        (-0.0f32).to_bits()
    );
}

#[test]
fn one_input_is_copied_into_a_new_tensor_and_the_output_is_charged() {
    let cpu = backend(1);
    let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let before = cpu.budget().live_bytes().unwrap();
    let y = cpu.concat(&[&x], 0).unwrap();
    assert_eq!(v(&y), v(&x));
    assert_eq!(cpu.budget().live_bytes().unwrap(), before + 16);
    let z = cpu.slice(&x, 0, 0, 2).unwrap();
    assert_eq!(v(&z), v(&x));
    assert_eq!(cpu.budget().live_bytes().unwrap(), before + 32);
    drop((y, z));
    assert_eq!(cpu.budget().live_bytes().unwrap(), before);
}

#[test]
fn memory_assembly_blocks_join_on_the_token_axis_and_the_tail_slices_back() {
    // SpikeTrack's per-frame memory: F frames of [B, n, 256], P pointer
    // blocks of [B, 16, 256] and a zero block, joined along the token axis,
    // then the last P * 16 tokens read back.
    let cpu = backend(4);
    let mut rng = SplitMix64(3);
    let (b, dim, frame_tokens, p) = (2usize, 256usize, 1296usize, 3usize);
    let frames: Vec<Vec<f32>> = (0..3)
        .map(|_| finite_bits(&mut rng, b * frame_tokens * dim))
        .collect();
    let pointers: Vec<Vec<f32>> = (0..p)
        .map(|_| finite_bits(&mut rng, b * 16 * dim))
        .collect();
    let zeros = vec![0.0f32; b * 16 * dim];
    let mut blocks: Vec<Tensor> = frames
        .iter()
        .map(|d| t(d, &[b, frame_tokens, dim]))
        .collect();
    blocks.extend(pointers.iter().map(|d| t(d, &[b, 16, dim])));
    blocks.push(t(&zeros, &[b, 16, dim]));
    let refs: Vec<&Tensor> = blocks.iter().collect();
    let memory = cpu.concat(&refs, 1).unwrap();
    let tokens = 3 * frame_tokens + p * 16 + 16;
    assert_eq!(memory.shape(), [b, tokens, dim]);
    // Every part reads back from where the join put it.
    let mut at = 0;
    for block in &blocks {
        let n = block.shape()[1];
        let back = cpu.slice(&memory, 1, at, n).unwrap();
        assert_eq!(bits(&v(&back)), bits(&v(block)), "block at token {at}");
        at += n;
    }
    // The tail: the pointer blocks and the zero block, as in the model.
    let tail = cpu
        .slice(&memory, 1, tokens - (p * 16 + 16), p * 16 + 16)
        .unwrap();
    let want = cpu
        .concat(&blocks[3..].iter().collect::<Vec<_>>(), 1)
        .unwrap();
    assert_eq!(bits(&v(&tail)), bits(&v(&want)));
}

#[test]
fn slicing_pieces_and_joining_them_again_is_the_identity() {
    let cpu = backend(3);
    let mut rng = SplitMix64(4);
    let shape = [3usize, 11, 5, 2];
    let data = finite_bits(&mut rng, numel(&shape));
    let x = t(&data, &shape);
    for axis in 0..4 {
        let n = shape[axis];
        for _ in 0..20 {
            // Random cut points make 1 to 4 pieces.
            let mut cuts: Vec<usize> = (0..rng.below(4))
                .map(|_| 1 + rng.below(n.max(2) - 1))
                .collect();
            cuts.retain(|&c| c < n);
            cuts.sort_unstable();
            cuts.dedup();
            let mut bounds = vec![0];
            bounds.extend(cuts);
            bounds.push(n);
            let pieces: Vec<Tensor> = bounds
                .windows(2)
                .map(|w| cpu.slice(&x, axis, w[0], w[1] - w[0]).unwrap())
                .collect();
            let refs: Vec<&Tensor> = pieces.iter().collect();
            let joined = cpu.concat(&refs, axis).unwrap();
            assert_eq!(joined.shape(), shape.as_slice());
            assert_eq!(bits(&v(&joined)), bits(&data), "axis {axis} {bounds:?}");
        }
    }
}

#[test]
fn a_large_join_matches_its_parts_and_not_the_thread_count() {
    let mut rng = SplitMix64(5);
    let a_shape = [2usize, 700, 640];
    let b_shape = [2usize, 301, 640];
    let a = finite_bits(&mut rng, numel(&a_shape));
    let b = finite_bits(&mut rng, numel(&b_shape));
    let (at, bt) = (t(&a, &a_shape), t(&b, &b_shape));
    let base = backend(1).concat(&[&at, &bt], 1).unwrap();
    assert_eq!(base.shape(), [2, 1001, 640]);
    for threads in [2, 7, 18] {
        let got = backend(threads).concat(&[&at, &bt], 1).unwrap();
        assert_eq!(bits(&v(&got)), bits(&v(&base)), "x{threads}");
    }
    let cpu = backend(2);
    assert_eq!(bits(&v(&cpu.slice(&base, 1, 0, 700).unwrap())), bits(&a));
    assert_eq!(bits(&v(&cpu.slice(&base, 1, 700, 301).unwrap())), bits(&b));
}

#[test]
fn concat_refuses_what_does_not_fit_with_the_neighbours_variants() {
    let cpu = backend(1);
    let a = t(&[0.5; 24], &[2, 3, 4]);
    let b = t(&[0.5; 40], &[2, 5, 4]);
    let before = cpu.budget().live_bytes().unwrap();
    assert!(cpu.concat(&[&a, &b], 1).is_ok());
    // No inputs, a rank that differs, an axis out of range, an extent off
    // the axis that differs.
    assert_shape(cpu.concat(&[], 0));
    assert_shape(cpu.concat(&[&a, &t(&[0.5; 6], &[2, 3])], 1));
    assert_shape(cpu.concat(&[&t(&[0.5; 6], &[2, 3]), &a], 1));
    assert_shape(cpu.concat(&[&a, &b], 3));
    assert_shape(cpu.concat(&[&a, &b], usize::MAX));
    assert_shape(cpu.concat(&[&t(&[0.5], &[]), &t(&[0.5], &[])], 0));
    assert_shape(cpu.concat(&[&a, &b], 0));
    assert_shape(cpu.concat(&[&a, &b], 2));
    // An empty extent anywhere, and a dtype that is not F32, in any position.
    assert_shape(cpu.concat(&[&a, &t(&[], &[2, 0, 4])], 1));
    assert_shape(cpu.concat(&[&t(&[], &[2, 0, 4]), &a], 1));
    let ids = Tensor::from_u32(&[1; 24], &[2, 3, 4], &Budget::new(1 << 12)).unwrap();
    let half = Tensor::zeros(&[2, 3, 4], DType::Bf16, &Budget::new(1 << 12)).unwrap();
    for odd in [&ids, &half] {
        for refs in [[&a, odd], [odd, &a]] {
            match cpu.concat(&refs, 1) {
                Err(OjasError::Dtype { op, expected, got }) => {
                    assert_eq!((op, expected, got), ("concat", DType::F32, odd.dtype()))
                }
                other => panic!("expected Dtype, got {other:?}"),
            }
        }
    }
    // A view is not read in storage order.
    let base = t(&[0.5; 24], &[2, 4, 3]);
    let swapped = base.view(&[2, 3, 4], &[12, 1, 3], 0).unwrap();
    assert_shape(cpu.concat(&[&a, &swapped], 1));
    assert_shape(cpu.concat(&[&swapped, &a], 1));
    assert_eq!(cpu.budget().live_bytes().unwrap(), before);
}

#[test]
fn shapes_are_refused_before_values_are_read() {
    let cpu = backend(1);
    let mut poisoned = vec![0.5f32; 24];
    poisoned[5] = f32::NAN;
    let bad = t(&poisoned, &[2, 3, 4]);
    let ok = t(&[0.5; 24], &[2, 3, 4]);
    assert_shape(cpu.concat(&[&bad, &ok], 3));
    assert_shape(cpu.concat(&[&bad, &t(&[0.5; 8], &[2, 4])], 1));
    assert_shape(cpu.concat(&[&bad, &t(&[0.5; 36], &[3, 3, 4])], 1));
    assert_shape(cpu.slice(&bad, 3, 0, 1));
    assert_range(cpu.slice(&bad, 1, 2, 2));
    assert_shape(cpu.slice(&bad, 1, 0, 0));
    assert_nonfinite(cpu.concat(&[&bad, &ok], 1));
    assert_nonfinite(cpu.slice(&bad, 1, 0, 1));
}

#[test]
fn a_non_finite_value_in_any_input_is_refused_not_passed_through() {
    let cpu = backend(2);
    let ok = t(&[0.5; 12], &[3, 4]);
    // A signalling NaN with a payload, a quiet one, both infinities.
    for bad in [
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xffc0_1234),
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        for at in [0, 5, 11] {
            let mut data = vec![0.5f32; 12];
            data[at] = bad;
            let poisoned = t(&data, &[3, 4]);
            assert_nonfinite(cpu.concat(&[&poisoned], 0));
            assert_nonfinite(cpu.concat(&[&ok, &poisoned], 1));
            assert_nonfinite(cpu.concat(&[&poisoned, &ok, &ok], 0));
            // The whole input is scanned, not only the window the slice reads.
            assert_nonfinite(cpu.slice(&poisoned, 0, 1, 1));
        }
    }
}

#[test]
fn slice_refuses_what_does_not_fit_with_the_neighbours_variants() {
    let cpu = backend(1);
    let x = t(&[0.5; 48], &[2, 6, 4]);
    // The axis, an empty slice, and rows past the axis or overflowing.
    assert_shape(cpu.slice(&x, 3, 0, 1));
    assert_shape(cpu.slice(&x, usize::MAX, 0, 1));
    assert_shape(cpu.slice(&t(&[0.5], &[]), 0, 0, 1));
    assert_shape(cpu.slice(&x, 1, 0, 0));
    assert_shape(cpu.slice(&x, 1, 99, 0));
    assert_range(cpu.slice(&x, 1, 6, 1));
    assert_range(cpu.slice(&x, 1, 5, 2));
    assert_range(cpu.slice(&x, 1, 0, 7));
    assert_range(cpu.slice(&x, 1, usize::MAX, 1));
    assert_range(cpu.slice(&x, 1, 1, usize::MAX));
    assert_range(cpu.slice(&x, 1, usize::MAX, usize::MAX));
    // The last row and the whole axis are legal.
    assert_eq!(cpu.slice(&x, 1, 5, 1).unwrap().shape(), [2, 1, 4]);
    assert_eq!(cpu.slice(&x, 1, 0, 6).unwrap().shape(), [2, 6, 4]);
    // Empty input, wrong dtype, a view.
    assert_shape(cpu.slice(&t(&[], &[2, 0, 4]), 1, 0, 1));
    let ids = Tensor::from_u32(&[1; 48], &[2, 6, 4], &Budget::new(1 << 12)).unwrap();
    match cpu.slice(&ids, 1, 0, 1) {
        Err(OjasError::Dtype { op, .. }) => assert_eq!(op, "slice"),
        other => panic!("expected Dtype, got {other:?}"),
    }
    let base = t(&[0.5; 48], &[2, 4, 6]);
    assert_shape(cpu.slice(&base.view(&[2, 6, 4], &[24, 1, 6], 0).unwrap(), 1, 0, 1));
}

#[test]
fn an_over_budget_output_is_capacity_exceeded_and_releases_its_charge() {
    let cpu = CpuBackend::new(Budget::new(4096));
    let a = t(&[0.5; 600], &[2, 3, 100]);
    let before = cpu.budget().live_bytes().unwrap();
    // 4800 bytes of output against a 4096 byte cap.
    assert_capacity(cpu.concat(&[&a, &a], 0));
    assert_capacity(cpu.concat(&[&a, &a], 1));
    assert_eq!(cpu.budget().live_bytes().unwrap(), before);
    // A slice whose 8192 bytes of output do not fit.
    let big = t(&[0.5; 2048], &[2, 4, 256]);
    assert_capacity(cpu.slice(&big, 1, 0, 4));
    assert_eq!(cpu.budget().live_bytes().unwrap(), before);
}

fn is_bf16_exact(values: &[f32]) -> bool {
    values
        .iter()
        .all(|&x| ojas_core::round_f32_to_bf16(x).to_bits() == x.to_bits())
}

#[test]
fn a_bf16_region_moves_data_without_rounding_and_the_tag_follows_it() {
    let wrapped = Autocast::new(backend(1));
    let eye = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let _region = wrapped.autocast_region(AutocastMode::Bf16).unwrap();
    let tagged = wrapped
        .linear_forward(&t(&[-1.5, 2.25], &[1, 2]), &eye)
        .unwrap();
    assert!(tagged.bf16_exact());
    // Every input tagged: the output is tagged and holds the same values.
    let both = wrapped.concat(&[&tagged, &tagged], 1).unwrap();
    assert!(both.bf16_exact());
    assert_eq!(v(&both), [-1.5, 2.25, -1.5, 2.25]);
    // One untagged input carrying a value no bf16 holds: not rounded, not tagged.
    let fine = f32::from_bits(0x3f80_0001);
    assert!(!is_bf16_exact(&[fine]));
    let plain = t(&[fine, -1.0], &[1, 2]);
    let mixed = wrapped.concat(&[&tagged, &plain], 1).unwrap();
    assert!(!mixed.bf16_exact());
    assert_eq!(bits(&v(&mixed)), bits(&[-1.5, 2.25, fine, -1.0]));
    // A slice follows its input's tag and values.
    let cut = wrapped.slice(&both, 1, 1, 2).unwrap();
    assert!(cut.bf16_exact());
    assert_eq!(v(&cut), [2.25, -1.5]);
    let cut = wrapped.slice(&plain, 1, 0, 1).unwrap();
    assert!(!cut.bf16_exact());
    assert_eq!(bits(&v(&cut)), bits(&[fine]));
    // Refusals pass straight through the wrapper.
    assert_shape(wrapped.concat(&[], 0));
    assert_range(wrapped.slice(&plain, 1, 2, 1));
}
