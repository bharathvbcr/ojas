//! `MetalBackend::permute`: a device-resident, bit-exact axis reorder that
//! is validated by `ojas_core::permute_output_shape` and charged to the
//! budget like every other output.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{
    inverse_permutation, Backend, BackendId, Budget, DType, OjasError, Tensor, MAX_PERMUTE_RANK,
};
use ojas_metal::MetalBackend;

/// Row-major strides of `shape`, in elements.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// Host reference: output axis `a` is input axis `dims[a]`.
fn reference(x: &[u32], shape: &[usize], dims: &[usize]) -> Vec<u32> {
    let out_shape: Vec<usize> = dims.iter().map(|&a| shape[a]).collect();
    let in_strides = strides(shape);
    let n: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(n);
    for flat in 0..n {
        let mut rem = flat;
        let mut src = 0usize;
        for a in (0..out_shape.len()).rev() {
            let c = rem % out_shape[a];
            rem /= out_shape[a];
            src += c * in_strides[dims[a]];
        }
        out.push(x[src]);
    }
    out
}

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for p in permutations(n - 1) {
        for pos in 0..=p.len() {
            let mut q = p.clone();
            q.insert(pos, n - 1);
            out.push(q);
        }
    }
    out
}

fn is_shape(r: &Result<Tensor, OjasError>) -> bool {
    matches!(r, Err(OjasError::Shape { op: "permute", .. }))
}

#[test]
fn every_rank4_permutation_matches_the_host_reference_bit_for_bit() {
    let m = metal();
    // Odd, distinct extents so a swapped pair of axes cannot pass.
    let shape = [2usize, 3, 5, 7];
    let x = rand(&shape, 11, 1.0);
    let xb: Vec<u32> = ok("x", x.to_f32_vec()).iter().map(|v| v.to_bits()).collect();
    let dx = up(&m, &x);
    let perms = permutations(4);
    assert_eq!(perms.len(), 24);
    for dims in &perms {
        let y = ok("permute", m.permute(&dx, dims));
        let want_shape: Vec<usize> = dims.iter().map(|&a| shape[a]).collect();
        assert_eq!(y.shape(), want_shape.as_slice(), "{dims:?} shape");
        assert_eq!(y.device(), Some(BackendId::Metal), "{dims:?} residency");
        assert_eq!(bits(&y), reference(&xb, &shape, dims), "{dims:?} values");
        // The inverse permutation restores the input exactly.
        let back = ok("inverse", m.permute(&y, &inverse_permutation(dims)));
        assert_eq!(back.shape(), &shape, "{dims:?} inverse shape");
        assert_eq!(bits(&back), xb, "{dims:?} inverse values");
    }
}

#[test]
fn bthd_to_bhtd_round_trips_at_attention_shapes() {
    let m = metal();
    for (i, shape) in [[2usize, 17, 3, 16], [1, 1, 1, 64], [3, 255, 2, 33]]
        .iter()
        .enumerate()
    {
        let x = rand(shape, 20 + i as u64, 1.0);
        let xb: Vec<u32> = ok("x", x.to_f32_vec()).iter().map(|v| v.to_bits()).collect();
        let dx = up(&m, &x);
        let bhtd = ok("to bhtd", m.permute(&dx, &[0, 2, 1, 3]));
        assert_eq!(bhtd.shape(), &[shape[0], shape[2], shape[1], shape[3]]);
        assert_eq!(bits(&bhtd), reference(&xb, shape, &[0, 2, 1, 3]));
        let back = ok("to bthd", m.permute(&bhtd, &[0, 2, 1, 3]));
        assert_eq!(bits(&back), xb, "{shape:?}");
    }
}

#[test]
fn values_move_without_arithmetic() {
    let m = metal();
    let specials = [
        (-0.0f32).to_bits(),
        0.0f32.to_bits(),
        1,           // smallest subnormal
        0x807F_FFFF, // largest negative subnormal
        f32::MAX.to_bits(),
        f32::MIN.to_bits(),
        f32::MIN_POSITIVE.to_bits(),
        1.5f32.to_bits(),
        0x3F80_0001, // 1 + one ulp
        0xBEAA_AAAB, // -1/3
        0x0000_0100,
        0x7F7F_FFFE,
    ];
    let data: Vec<f32> = specials.iter().map(|&b| f32::from_bits(b)).collect();
    let x = host(&data, &[3, 4]);
    let y = ok("permute", m.permute(&up(&m, &x), &[1, 0]));
    assert_eq!(y.shape(), &[4, 3]);
    assert_eq!(bits(&y), reference(&specials, &[3, 4], &[1, 0]));
}

/// The CPU reference refuses a NaN or infinity in a permute input, as it
/// does for every op; Metal does the same rather than copying it.
#[test]
fn non_finite_inputs_are_refused_like_the_cpu_reference() {
    let (m, c) = (metal(), cpu());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7FC0_1234)] {
        let mut data = values(24, 4, 1.0);
        data[17] = bad;
        let x = host(&data, &[2, 3, 4]);
        let want = c.permute(&x, &[2, 0, 1]);
        let got = m.permute(&up(&m, &x), &[2, 0, 1]);
        assert!(
            matches!(want, Err(OjasError::NonFinite { op: "permute" })),
            "cpu {bad}: {want:?}"
        );
        assert!(
            matches!(got, Err(OjasError::NonFinite { op: "permute" })),
            "metal {bad}: {got:?}"
        );
    }
}

#[test]
fn every_rank4_permutation_matches_the_cpu_backend() {
    let (m, c) = (metal(), cpu());
    let shape = [3usize, 1, 4, 5];
    let x = rand(&shape, 12, 1.0);
    let dx = up(&m, &x);
    for dims in &permutations(4) {
        let want = ok("cpu", c.permute(&x, dims));
        let got = ok("metal", m.permute(&dx, dims));
        assert_eq!(got.shape(), want.shape(), "{dims:?}");
        let want_bits: Vec<u32> = ok("want", want.to_f32_vec())
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(bits(&got), want_bits, "{dims:?}");
    }
}

#[test]
fn identity_rank0_rank1_and_max_rank_are_fresh_copies() {
    let m = metal();
    let x = rand(&[4, 5], 1, 1.0);
    let dx = up(&m, &x);
    let id = ok("identity", m.permute(&dx, &[0, 1]));
    assert_eq!(bits(&id), bits(&dx));
    let (a, b) = (
        dx.device_buffer().expect("device"),
        id.device_buffer().expect("device"),
    );
    assert!(!std::sync::Arc::ptr_eq(a, b), "identity must be a new buffer");

    let s = host(&[2.5], &[]);
    let ds = ok("rank0", m.permute(&up(&m, &s), &[]));
    assert_eq!(ds.shape(), &[] as &[usize]);
    assert_eq!(down(&ds), vec![2.5]);

    let v = rand(&[9], 2, 1.0);
    let dv = ok("rank1", m.permute(&up(&m, &v), &[0]));
    assert_eq!(bits(&dv), bits(&up(&m, &v)));

    let shape = [2usize, 1, 3, 1, 2, 2, 1, 3];
    assert_eq!(shape.len(), MAX_PERMUTE_RANK);
    let dims = [7usize, 0, 6, 2, 5, 1, 4, 3];
    let x8 = rand(&shape, 3, 1.0);
    let xb: Vec<u32> = ok("x8", x8.to_f32_vec()).iter().map(|v| v.to_bits()).collect();
    let y8 = ok("rank8", m.permute(&up(&m, &x8), &dims));
    assert_eq!(bits(&y8), reference(&xb, &shape, &dims));
}

#[test]
fn a_contiguous_view_at_a_byte_offset_reads_its_own_window() {
    let m = metal();
    let x = rand(&[4, 6], 5, 1.0);
    let dx = up(&m, &x);
    // Rows 1..3, as a [2, 3, 2] view starting 6 elements in.
    let view = ok("view", dx.view(&[2, 3, 2], &[6, 2, 1], 6 * 4));
    let xb: Vec<u32> = ok("x", x.to_f32_vec()).iter().map(|v| v.to_bits()).collect();
    let window = &xb[6..18];
    let y = ok("permute", m.permute(&view, &[2, 0, 1]));
    assert_eq!(bits(&y), reference(window, &[2, 3, 2], &[2, 0, 1]));
}

#[test]
fn malformed_permutations_are_shape_errors() {
    let m = metal();
    let dx = up(&m, &rand(&[2, 3, 4], 1, 1.0));
    assert!(is_shape(&m.permute(&dx, &[0, 1])), "too few axes");
    assert!(is_shape(&m.permute(&dx, &[0, 1, 2, 3])), "too many axes");
    assert!(is_shape(&m.permute(&dx, &[0, 1, 3])), "axis out of range");
    assert!(is_shape(&m.permute(&dx, &[0, 1, 1])), "repeated axis");
    let nine = up(&m, &rand(&[1; MAX_PERMUTE_RANK + 1], 2, 1.0));
    let dims: Vec<usize> = (0..=MAX_PERMUTE_RANK).collect();
    assert!(is_shape(&m.permute(&nine, &dims)), "rank above the maximum");
}

#[test]
fn placement_dtype_layout_and_empty_inputs_are_refused() {
    let m = metal();
    let x = rand(&[2, 3], 1, 1.0);
    assert!(
        matches!(
            m.permute(&x, &[1, 0]),
            Err(OjasError::Placement {
                expected: Some(BackendId::Metal),
                ..
            })
        ),
        "a host tensor is not silently uploaded"
    );
    let ids = up(&m, &host_u32(&[1, 2, 3, 4, 5, 6], &[2, 3]));
    assert!(
        matches!(
            m.permute(&ids, &[1, 0]),
            Err(OjasError::Dtype {
                expected: DType::F32,
                got: DType::U32,
                ..
            })
        ),
        "only f32 is permuted"
    );
    let dx = up(&m, &x);
    let transposed = ok("strided view", dx.view(&[3, 2], &[1, 3], 0));
    assert!(
        is_shape(&m.permute(&transposed, &[1, 0])),
        "a non-contiguous view is refused like every other op"
    );
    // Metal cannot hold an empty tensor (upload refuses one), so the only
    // reachable zero-size input is a zero-extent view; it is refused.
    let empty = ok("empty view", dx.view(&[0, 3], &[3, 1], 0));
    assert!(is_shape(&m.permute(&empty, &[1, 0])), "empty input");
    let other = metal();
    assert!(
        m.permute(&up(&other, &x), &[1, 0]).is_err(),
        "another backend's buffer is refused"
    );
}

#[test]
fn the_output_is_charged_to_the_budget() {
    let small = ok("new", MetalBackend::new(Budget::new(20 * 1024)));
    let xs = up(&small, &rand(&[64, 64], 1, 1.0));
    assert_eq!(ok("live", small.budget().live_bytes()), 64 * 64 * 4);
    let r = small.permute(&xs, &[1, 0]);
    assert!(
        matches!(
            r,
            Err(OjasError::CapacityExceeded {
                requested: 16384,
                cap: 20480,
                live: 16384
            })
        ),
        "{r:?}"
    );
    assert_eq!(ok("live", small.budget().live_bytes()), 64 * 64 * 4);
    let ys = up(&small, &rand(&[16, 8], 2, 1.0));
    let live = ok("live", small.budget().live_bytes());
    let yt = ok("permute", small.permute(&ys, &[1, 0]));
    assert_eq!(ok("live", small.budget().live_bytes()), live + 16 * 8 * 4);
    drop(yt);
    assert_eq!(ok("live", small.budget().live_bytes()), live);
}
