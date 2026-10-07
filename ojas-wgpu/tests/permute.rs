//! `WgpuBackend::permute` against a host reference: every rank-4 order,
//! the attention layout change and its inverse, odd and unit extents, a lane
//! count past one row of workgroups, views, and the refusals.
//!
//! A permute moves words and does no arithmetic, so every comparison is on
//! the bits, including NaN payloads, `-0.0` and subnormals.

mod common;

use common::*;
use ojas_core::{inverse_permutation, Backend, BackendId, Budget, DType, OjasError, Tensor};
use ojas_wgpu::WgpuBackend;

/// Bit patterns arithmetic would disturb: quiet and signalling NaN payloads,
/// both zeros, subnormals and the extremes.
const AWKWARD: [u32; 8] = [
    0x7fc0_1234,
    0xff80_0001,
    0x8000_0000,
    0x0000_0000,
    0x0000_0001,
    0x807f_ffff,
    0x7f7f_ffff,
    0xff80_0000,
];

fn bits(seed: u64, n: usize) -> Vec<u32> {
    let mut out: Vec<u32> = data(seed, n).iter().map(|v| v.to_bits()).collect();
    for (i, slot) in out.iter_mut().enumerate().step_by(7) {
        *slot = AWKWARD[(i / 7) % AWKWARD.len()];
    }
    out
}

fn host_bits(values: &[u32], shape: &[usize]) -> Tensor {
    let f: Vec<f32> = values.iter().map(|b| f32::from_bits(*b)).collect();
    Tensor::from_f32(&f, shape, host_budget()).unwrap()
}

/// Row-major `out[i] = in[at(i)]` with output axis `a` = input axis `dims[a]`.
fn reference(values: &[u32], shape: &[usize], dims: &[usize]) -> Vec<u32> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for a in (0..rank.saturating_sub(1)).rev() {
        strides[a] = strides[a + 1] * shape[a + 1];
    }
    let out_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
    let n: usize = shape.iter().product();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut rem = i;
        let mut off = 0usize;
        for a in (0..rank).rev() {
            let c = rem % out_shape[a];
            rem /= out_shape[a];
            off += c * strides[dims[a]];
        }
        out.push(values[off]);
    }
    out
}

fn down_bits(g: &WgpuBackend, t: &Tensor) -> Vec<u32> {
    assert_eq!(t.device(), Some(BackendId::Wgpu), "permute left the device");
    g.download(t)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn nonfinite(values: &[u32]) -> bool {
    values.iter().any(|b| b & 0x7f80_0000 == 0x7f80_0000)
}

/// Sync, expecting a deferred `permute` fault exactly when the input held a
/// non-finite value (the backend's contract for every op).
fn settle(g: &WgpuBackend, values: &[u32]) {
    match (nonfinite(values), g.sync()) {
        (true, Err(OjasError::NonFinite { op: "permute" })) | (false, Ok(())) => {}
        (expect, got) => panic!("non-finite input {expect}: sync gave {got:?}"),
    }
}

fn check(g: &WgpuBackend, seed: u64, shape: &[usize], dims: &[usize]) {
    let n = shape.iter().product();
    let values = bits(seed, n);
    let x = g.upload(&host_bits(&values, shape)).unwrap();
    let before = readbacks(g);
    let y = g.permute(&x, dims).unwrap();
    assert_eq!(
        readbacks(g),
        before,
        "permute {shape:?} by {dims:?} read a tensor back"
    );
    let want_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
    assert_eq!(y.shape(), want_shape.as_slice(), "{shape:?} by {dims:?}");
    settle(g, &values);
    let got = down_bits(g, &y);
    // Positive control: the download above is counted on this budget.
    assert_eq!(readbacks(g).0, before.0 + 1, "counter did not move");
    let want = reference(&values, shape, dims);
    if let Some(i) = (0..n).find(|&i| got[i] != want[i]) {
        panic!(
            "{shape:?} by {dims:?}: element {i} is {:#010x}, want {:#010x}",
            got[i], want[i]
        );
    }
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for rest in permutations(n - 1) {
        for at in 0..=rest.len() {
            let mut p = rest.clone();
            p.insert(at, n - 1);
            out.push(p);
        }
    }
    out
}

#[test]
fn every_rank_four_order_moves_bits_exactly() {
    let g = &fresh();
    let orders = permutations(4);
    assert_eq!(orders.len(), 24);
    for (i, dims) in orders.iter().enumerate() {
        check(g, 3000 + i as u64, &[2, 3, 5, 7], dims);
        check(g, 3100 + i as u64, &[3, 1, 4, 65], dims);
    }
    g.sync().expect("every fault was reported by its own check");
}

#[test]
fn a_non_finite_input_is_moved_and_reported_as_permute() {
    // Pre-fix the move raised nothing, so NaN passed through as a clean
    // result. The contract is the backend's: Ok now, NonFinite naming
    // `permute` at the next sync point, bits still moved unchanged.
    let g = &fresh();
    g.sync().unwrap();
    let mut values = bits(3050, 12);
    values.iter_mut().for_each(|b| *b &= 0x3fff_ffff);
    values[5] = 0x7fc0_0abc;
    let x = g.upload(&host_bits(&values, &[3, 4])).unwrap();
    let y = g.permute(&x, &[1, 0]).unwrap();
    match g.download(&y) {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "permute"),
        other => panic!("NaN input passed as a clean result: {other:?}"),
    }
    assert_eq!(down_bits(g, &y), reference(&values, &[3, 4], &[1, 0]));
    // A finite input reports nothing.
    let clean: Vec<u32> = values.iter().map(|b| b & 0x3fff_ffff).collect();
    let x = g.upload(&host_bits(&clean, &[3, 4])).unwrap();
    let _ = g.permute(&x, &[1, 0]).unwrap();
    g.sync().expect("finite input");
}

#[test]
fn attention_layout_round_trips() {
    let g = &fresh();
    // [B, T, H, D] -> [B, H, T, D], the RoPE-to-SDPA hand-off, and back.
    let shape = [2usize, 67, 6, 64];
    let dims = [0usize, 2, 1, 3];
    check(g, 3200, &shape, &dims);
    let n = shape.iter().product();
    let values = bits(3201, n);
    let x = g.upload(&host_bits(&values, &shape)).unwrap();
    let y = g.permute(&x, &dims).unwrap();
    let back = g.permute(&y, &inverse_permutation(&dims)).unwrap();
    assert_eq!(back.shape(), shape.as_slice());
    settle(g, &values);
    assert_eq!(down_bits(g, &back), values);
    g.sync().unwrap();
}

#[test]
fn odd_unit_and_high_rank_shapes_match_the_reference() {
    let g = &fresh();
    check(g, 3300, &[1, 1, 1, 1], &[3, 2, 1, 0]);
    check(g, 3301, &[7, 1, 13, 3], &[2, 0, 3, 1]);
    check(g, 3302, &[17], &[0]);
    check(g, 3303, &[33, 65], &[1, 0]);
    check(g, 3304, &[63, 64, 65], &[2, 0, 1]);
    check(
        g,
        3305,
        &[2, 1, 3, 1, 2, 1, 2, 3],
        &[7, 0, 5, 2, 6, 1, 4, 3],
    );
    check(g, 3306, &[], &[]);
    g.sync().unwrap();
}

#[test]
fn lane_count_past_one_row_of_workgroups_matches_the_reference() {
    // 65_535 workgroups of 256 lanes is 16_776_960; this is past it.
    let g = &fresh();
    check(g, 3400, &[4100, 4097], &[1, 0]);
    g.sync().unwrap();
}

#[test]
fn an_offset_view_is_compacted_first() {
    let g = &fresh();
    let values = bits(3500, 4 + 6 * 5);
    let whole = g.upload(&host_bits(&values, &[34])).unwrap();
    let view = whole.view(&[6, 5], &[5, 1], 16).unwrap();
    let y = g.permute(&view, &[1, 0]).unwrap();
    settle(g, &values[4..]);
    assert_eq!(down_bits(g, &y), reference(&values[4..], &[6, 5], &[1, 0]));
}

#[test]
fn the_output_is_charged_to_the_budget() {
    let budget = Budget::new(1 << 20);
    let g = WgpuBackend::open(budget.clone()).unwrap();
    let start = budget.live_bytes().unwrap();
    let x = g.upload(&host(3600, &[100, 30])).unwrap();
    assert_eq!(budget.live_bytes().unwrap(), start + 12_000);
    let y = g.permute(&x, &[1, 0]).unwrap();
    // The 16-byte geometry table is scratch, charged until its submission.
    assert_eq!(budget.live_bytes().unwrap(), start + 24_016);
    g.sync().unwrap();
    assert_eq!(budget.live_bytes().unwrap(), start + 24_000);
    drop((x, y));
    assert_eq!(budget.live_bytes().unwrap(), start);
    // An output past the budget is refused before anything is allocated.
    let x = g.upload(&host(3601, &[150_000])).unwrap();
    let live = budget.live_bytes().unwrap();
    match g.permute(&x, &[0]) {
        Err(OjasError::CapacityExceeded { .. }) => {}
        other => panic!("expected CapacityExceeded, got {other:?}"),
    }
    assert_eq!(budget.live_bytes().unwrap(), live);
}

#[test]
fn malformed_permutations_are_refused() {
    let g = &fresh();
    let x = g.upload(&host(3700, &[2, 3, 4])).unwrap();
    for dims in [&[0usize, 1][..], &[0, 1, 2, 3], &[0, 1, 1], &[0, 1, 3], &[]] {
        match g.permute(&x, dims) {
            Err(OjasError::Shape { op: "permute", .. }) => {}
            other => panic!("{dims:?}: expected a permute Shape error, got {other:?}"),
        }
    }
    // Rank past ojas_core::MAX_PERMUTE_RANK.
    let deep = g.upload(&host(3701, &[1; 9])).unwrap();
    assert!(matches!(
        g.permute(&deep, &[8, 7, 6, 5, 4, 3, 2, 1, 0]),
        Err(OjasError::Shape { .. })
    ));
    // A host tensor is a placement error; there is no host fallback.
    assert!(matches!(
        g.permute(&host(3702, &[2, 3]), &[1, 0]),
        Err(OjasError::Placement { found: None, .. })
    ));
    // U32 ids carry a host shadow a device permute cannot follow.
    let ids = g.upload(&host_u32(&[1, 2, 3, 4, 5, 6], &[2, 3])).unwrap();
    assert!(matches!(
        g.permute(&ids, &[1, 0]),
        Err(OjasError::Dtype {
            expected: DType::F32,
            ..
        })
    ));
}

/// Edge cases against the CPU, error for error and bit for bit: rank 0, unit
/// extents, the rank cap and one past it, a zero axis (D17: refused by
/// `ojas_core::permute_dims` on every backend), a u32 input and malformed
/// `dims`. An operand with a zero axis cannot be uploaded, so it is passed
/// as the host tensor and as a zero-extent device view; both must give the
/// CPU's error, with nothing charged.
#[test]
fn edge_cases_match_the_cpu_error_for_error_and_bit_for_bit() {
    let g = own();
    let c = cpu();
    let eight = [1usize, 2, 1, 3, 1, 2, 1, 2];
    let reversed: Vec<usize> = (0..8).rev().collect();
    let nine: Vec<usize> = (0..=ojas_core::MAX_PERMUTE_RANK).collect();
    let cases: Vec<(Tensor, Vec<usize>)> = vec![
        (host(3800, &[]), vec![]),
        (host(3801, &[1]), vec![0]),
        (host(3802, &[1, 1, 1]), vec![2, 0, 1]),
        (host(3803, &eight), reversed),
        (host(3804, &[1; 9]), nine),
        (host(3805, &[2, 0, 3]), vec![2, 0, 1]),
        (host(3806, &[0]), vec![0]),
        (host_u32(&[0; 6], &[2, 3]), vec![1, 0]),
        (host(3807, &[2, 3]), vec![0]),
        (host(3808, &[2, 3]), vec![1, 1]),
        (host(3809, &[2, 3]), vec![0, 2]),
    ];
    let host_bits_of = |t: &Tensor| -> Vec<u32> {
        t.to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    for (x, dims) in &cases {
        let what = format!("{:?} by {dims:?}", x.shape());
        let want = c
            .permute(x, dims)
            .map(|y| (y.shape().to_vec(), host_bits_of(&y)));
        let mut operands = Vec::new();
        if x.shape().contains(&0) {
            operands.push(x.clone());
            let full: Vec<usize> = x.shape().iter().map(|&d| d.max(1)).collect();
            let mut strides = vec![1usize; x.shape().len()];
            for a in (0..x.shape().len().saturating_sub(1)).rev() {
                strides[a] = strides[a + 1] * x.shape()[a + 1];
            }
            let dev = up(&host(3810, &full));
            operands.push(dev.view(x.shape(), &strides, 0).unwrap());
        } else {
            operands.push(up(x));
        }
        for operand in operands {
            let live = g.budget().live_bytes().unwrap();
            let got = g
                .permute(&operand, dims)
                .map(|y| (y.shape().to_vec(), down_bits(&g, &y)));
            assert_eq!(format!("{got:?}"), format!("{want:?}"), "{what}");
            assert_eq!(g.budget().live_bytes().unwrap(), live, "{what}");
            g.sync().unwrap();
        }
    }
}
