//! `concat`, `slice` and `fill` on Metal against the CPU backend: bit copies
//! (`ojas_concat_part`, `ojas_slice`) and a device-side fill
//! (`ojas_fill_words`), at every axis of ranks 1 to 4, odd extents, views at
//! unaligned offsets, SpikeTrack's memory assembly, every refusal as the
//! validator refuses it, non-finite faults under the op name, the budget
//! (the output is the only charge), and the dispatch counts.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{concat_dims, fill_dims, slice_dims, Backend, Budget, DType, OjasError, Tensor};
use ojas_metal::MetalBackend;

fn bytes(t: &Tensor) -> Vec<u8> {
    ok("bytes", t.to_ne_bytes())
}

/// A device tensor holds exactly the host reference's bytes.
fn same_bytes(m: &MetalBackend, what: &str, got: &Tensor, want: &Tensor) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    assert_eq!(got.dtype(), want.dtype(), "{what}: dtype");
    assert_eq!(
        got.device(),
        Some(ojas_core::BackendId::Metal),
        "{what}: residency"
    );
    let host = ok("download", m.download(got));
    assert_eq!(bytes(&host), bytes(want), "{what}: bytes");
}

fn dispatches_of(m: &MetalBackend, f: &dyn Fn()) -> u64 {
    ok("sync", m.sync());
    let before = m.dispatches();
    f();
    ok("sync", m.sync());
    m.dispatches() - before
}

#[test]
fn concat_matches_cpu_bitwise_at_every_axis_and_rank() {
    let (m, c) = (metal(), cpu());
    let shapes: [&[usize]; 4] = [&[7], &[3, 5], &[2, 3, 4], &[2, 3, 4, 5]];
    for shape in shapes {
        for axis in 0..shape.len() {
            // Three inputs of differing extent along the axis.
            let parts: Vec<Tensor> = [1usize, 2, 5]
                .iter()
                .enumerate()
                .map(|(i, &len)| {
                    let mut s = shape.to_vec();
                    s[axis] = len;
                    rand(&s, 300 + 10 * axis as u64 + i as u64, 5.0)
                })
                .collect();
            let refs: Vec<&Tensor> = parts.iter().collect();
            let want = ok("cpu", c.concat(&refs, axis));
            let dev: Vec<Tensor> = parts.iter().map(|t| up(&m, t)).collect();
            let drefs: Vec<&Tensor> = dev.iter().collect();
            let got = ok("metal", m.concat(&drefs, axis));
            ok("sync", m.sync());
            same_bits(&format!("{shape:?} axis {axis}"), &got, &want);
        }
    }
}

#[test]
fn concat_of_one_input_of_the_same_buffer_twice_and_of_offset_views() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[2, 3, 4], 320, 3.0);
    let dx = up(&m, &x);
    let want = ok("cpu one", c.concat(&[&x], 1));
    let got = ok("metal one", m.concat(&[&dx], 1));
    ok("sync", m.sync());
    same_bits("one input", &got, &want);
    let want = ok("cpu twice", c.concat(&[&x, &x], 2));
    let got = ok("metal twice", m.concat(&[&dx, &dx], 2));
    ok("sync", m.sync());
    same_bits("the same buffer twice", &got, &want);
    let y = rand(&[2, 5, 4], 321, 3.0);
    let want = ok("cpu views", c.concat(&[&x, &y], 1));
    let got = ok(
        "metal views",
        m.concat(
            &[&offset_view(&m, &x, 3, 322), &offset_view(&m, &y, 5, 323)],
            1,
        ),
    );
    ok("sync", m.sync());
    same_bits("offset views", &got, &want);
}

/// SpikeTrack's memory assembly: frames of `[1, 8, 5184, 32]` keys joined
/// along the key axis.
#[test]
fn memory_assembly_matches_cpu() {
    let (m, c) = (metal(), cpu());
    let frames: Vec<Tensor> = (0..3)
        .map(|i| rand(&[1, 8, 5184, 32], 330 + i, 2.0))
        .collect();
    let refs: Vec<&Tensor> = frames.iter().collect();
    let want = ok("cpu", c.concat(&refs, 2));
    let dev: Vec<Tensor> = frames.iter().map(|t| up(&m, t)).collect();
    let drefs: Vec<&Tensor> = dev.iter().collect();
    let got = ok("metal", m.concat(&drefs, 2));
    ok("sync", m.sync());
    same_bits("3 frames x [1,8,5184,32] along 2", &got, &want);
}

#[test]
fn slice_matches_cpu_bitwise_for_every_axis_start_and_length() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[3, 4, 5, 6], 340, 4.0);
    let dx = up(&m, &x);
    for axis in 0..4 {
        let extent = x.shape()[axis];
        for start in [0usize, 1, extent - 1] {
            for len in [1usize, 2, extent - start] {
                if start + len > extent {
                    continue;
                }
                let want = ok("cpu", c.slice(&x, axis, start, len));
                let got = ok("metal", m.slice(&dx, axis, start, len));
                ok("sync", m.sync());
                same_bits(&format!("axis {axis} start {start} len {len}"), &got, &want);
            }
        }
    }
    // A view at an unaligned offset, and a slice of a slice.
    let want = ok("cpu", c.slice(&x, 1, 1, 2));
    let got = ok("metal", m.slice(&offset_view(&m, &x, 3, 341), 1, 1, 2));
    ok("sync", m.sync());
    same_bits("offset view", &got, &want);
    let inner = ok("slice", m.slice(&dx, 3, 1, 4));
    let got = ok("slice of slice", m.slice(&inner, 2, 2, 3));
    let want = ok("cpu", c.slice(&ok("cpu", c.slice(&x, 3, 1, 4)), 2, 2, 3));
    ok("sync", m.sync());
    same_bits("slice of a slice", &got, &want);
}

#[test]
fn fill_matches_cpu_in_every_dtype() {
    let (m, c) = (metal(), cpu());
    let cases: Vec<(DType, f32)> = vec![
        (DType::F32, 1.5),
        (DType::F32, -0.0),
        (DType::F32, f32::MIN_POSITIVE),
        (DType::F32, f32::from_bits(1)),
        (DType::F32, f32::MAX),
        (DType::Bf16, 1.5),
        (DType::Bf16, -0.0),
        (DType::Bf16, -3.0),
        (DType::U32, 7.0),
        (DType::U32, 0.0),
        (DType::U32, -0.0),
        (DType::U32, 4_294_967_040.0),
    ];
    // Odd and even element counts, rank 0 and a SAM-sized tensor.
    for (dtype, value) in cases {
        for shape in [
            vec![],
            vec![1usize],
            vec![3],
            vec![2, 5],
            vec![1, 8, 33, 33],
        ] {
            let want = ok("cpu", c.fill(&shape, dtype, value));
            let got = ok("metal", m.fill(&shape, dtype, value));
            ok("sync", m.sync());
            same_bytes(&m, &format!("{dtype:?} {value} {shape:?}"), &got, &want);
        }
    }
}

#[test]
fn a_malformed_call_is_refused_as_the_validator_refuses_it() {
    let m = metal();
    let x = rand(&[2, 3, 4], 350, 1.0);
    let y = rand(&[2, 5, 4], 351, 1.0);
    let concat_cases: Vec<(&str, Vec<Tensor>, usize)> = vec![
        ("no inputs", vec![], 0),
        ("axis past the rank", vec![x.clone()], 3),
        ("rank mismatch", vec![x.clone(), rand(&[2, 3], 1, 1.0)], 1),
        (
            "extent off the axis",
            vec![x.clone(), rand(&[2, 3, 5], 2, 1.0)],
            1,
        ),
        (
            "u32 input",
            vec![x.clone(), host_u32(&[0; 24], &[2, 3, 4])],
            1,
        ),
        ("empty input", vec![x.clone(), host(&[], &[2, 0, 4])], 1),
    ];
    for (what, inputs, axis) in concat_cases {
        let refs: Vec<&Tensor> = inputs.iter().collect();
        let want = concat_dims(&refs, axis).expect_err(what);
        let err = m.concat(&refs, axis).expect_err(what);
        assert_eq!(format!("{err:?}"), format!("{want:?}"), "{what} (host)");
        let usable = inputs
            .iter()
            .all(|t| t.dtype() == DType::F32 && !t.shape().contains(&0));
        if usable {
            let dev: Vec<Tensor> = inputs.iter().map(|t| up(&m, t)).collect();
            let drefs: Vec<&Tensor> = dev.iter().collect();
            let err = m.concat(&drefs, axis).expect_err(what);
            assert_eq!(format!("{err:?}"), format!("{want:?}"), "{what} (device)");
        }
        ok("sync", m.sync());
    }
    // A valid call on host operands: placement, after the shape.
    let r = m.concat(&[&x, &y], 1);
    assert!(
        matches!(r, Err(OjasError::Placement { op: "concat", .. })),
        "{r:?}"
    );

    let slice_cases: Vec<(&str, Tensor, usize, usize, usize)> = vec![
        ("axis past the rank", x.clone(), 3, 0, 1),
        ("len 0", x.clone(), 1, 0, 0),
        ("past the extent", x.clone(), 1, 2, 2),
        ("overflow", x.clone(), 1, usize::MAX, 2),
        ("u32 input", host_u32(&[0; 24], &[2, 3, 4]), 1, 0, 1),
        ("empty input", host(&[], &[2, 0, 4]), 1, 0, 1),
    ];
    for (what, input, axis, start, len) in slice_cases {
        let want = slice_dims(&input, axis, start, len).expect_err(what);
        let err = m.slice(&input, axis, start, len).expect_err(what);
        assert_eq!(format!("{err:?}"), format!("{want:?}"), "{what} (host)");
        if input.dtype() == DType::F32 && !input.shape().contains(&0) {
            let err = m.slice(&up(&m, &input), axis, start, len).expect_err(what);
            assert_eq!(format!("{err:?}"), format!("{want:?}"), "{what} (device)");
        }
        ok("sync", m.sync());
    }
    let r = m.slice(&x, 1, 0, 1);
    assert!(
        matches!(r, Err(OjasError::Placement { op: "slice", .. })),
        "{r:?}"
    );

    let fill_cases: Vec<(&str, Vec<usize>, DType, f32)> = vec![
        ("zero extent", vec![2, 0], DType::F32, 1.0),
        ("f16", vec![2], DType::F16, 1.0),
        ("NaN", vec![2], DType::F32, f32::NAN),
        ("infinity", vec![2], DType::F32, f32::INFINITY),
        ("bf16 inexact", vec![2], DType::Bf16, 0.1),
        ("u32 negative", vec![2], DType::U32, -1.0),
        ("u32 fraction", vec![2], DType::U32, 1.5),
        ("u32 too large", vec![2], DType::U32, 4_294_967_296.0),
        ("shape overflow", vec![usize::MAX, 2], DType::F32, 1.0),
    ];
    for (what, shape, dtype, value) in fill_cases {
        let want = fill_dims(&shape, dtype, value).expect_err(what);
        let err = m.fill(&shape, dtype, value).expect_err(what);
        assert_eq!(format!("{err:?}"), format!("{want:?}"), "{what}");
        ok("sync", m.sync());
    }
}

#[test]
fn a_non_finite_element_is_a_deferred_fault_under_the_op_name() {
    let m = metal();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut data = values(24, 360, 1.0);
        data[17] = bad;
        let x = up(&m, &host(&data, &[2, 3, 4]));
        let y = up(&m, &rand(&[2, 3, 4], 361, 1.0));
        let r = m.concat(&[&y, &x], 1);
        deferred(&m, &format!("concat {bad}"), r, "concat");
        let r = m.concat(&[&x, &y], 0);
        deferred(&m, &format!("concat first {bad}"), r, "concat");
        let r = m.slice(&x, 1, 1, 2);
        deferred(&m, &format!("slice {bad}"), r, "slice");
    }
    // The CPU refuses a non-finite input wherever it sits, copied or not:
    // so does Metal, for a value outside the slice.
    let mut data = values(24, 362, 1.0);
    data[0] = f32::NAN;
    let x = up(&m, &host(&data, &[2, 3, 4]));
    let r = m.slice(&x, 1, 1, 1);
    deferred(&m, "NaN outside the slice", r, "slice");
}

/// The output is the only charge: a budget that holds the inputs and the
/// output runs, one value short is refused, and nothing is left charged.
#[test]
fn the_output_is_the_only_charge() {
    let a = rand(&[4, 3, 8], 370, 1.0);
    let b = rand(&[4, 5, 8], 371, 1.0);
    let inputs = (4 * 3 * 8 + 4 * 5 * 8) * 4;
    let out_concat = 4 * 8 * 8 * 4;
    let out_slice = 4 * 2 * 8 * 4;
    let out_fill = 4 * 8 * 8 * 4;
    for (what, run_input, output) in [
        ("concat", inputs, out_concat),
        ("slice", 4 * 3 * 8 * 4, out_slice),
    ] {
        for (room, runs) in [(run_input + output - 4, false), (run_input + output, true)] {
            let tight = MetalBackend::new(Budget::new(room as u64))
                .unwrap_or_else(|e| panic!("Metal device unavailable: {e:?}"));
            let da = up(&tight, &a);
            // A slice runs on one input; only a concat uploads the other.
            let db = (what == "concat").then(|| up(&tight, &b));
            let baseline = ok("live", tight.budget().live_bytes());
            let r = match &db {
                Some(db) => tight.concat(&[&da, db], 1),
                None => tight.slice(&da, 1, 0, 2),
            };
            ok("sync", tight.sync());
            if runs {
                drop(ok(what, r));
            } else {
                assert!(
                    matches!(r, Err(OjasError::CapacityExceeded { .. })),
                    "{what}, one value short: {r:?}"
                );
            }
            assert_eq!(ok("live", tight.budget().live_bytes()), baseline, "{what}");
        }
    }
    for (room, runs) in [(out_fill - 4, false), (out_fill, true)] {
        let tight = MetalBackend::new(Budget::new(room as u64))
            .unwrap_or_else(|e| panic!("Metal device unavailable: {e:?}"));
        let r = tight.fill(&[4, 8, 8], DType::F32, 2.5);
        if runs {
            drop(ok("fill", r));
        } else {
            assert!(
                matches!(r, Err(OjasError::CapacityExceeded { .. })),
                "fill, one value short: {r:?}"
            );
        }
        ok("sync", tight.sync());
        assert_eq!(ok("live", tight.budget().live_bytes()), 0, "fill");
    }
}

/// One dispatch per input for a concat (its kernel scans as it copies), a scan
/// and a kernel for a slice (the whole input is scanned, not only the rows
/// copied) and one for a fill (nothing is uploaded).
#[test]
fn dispatch_counts_are_a_kernel_per_concat_input_a_scan_and_kernel_per_slice_and_one_per_fill() {
    let m = metal();
    let parts: Vec<Tensor> = (0..3)
        .map(|i| up(&m, &rand(&[2, 4, 6], 380 + i, 1.0)))
        .collect();
    let refs: Vec<&Tensor> = parts.iter().collect();
    let c3 = dispatches_of(&m, &|| drop(ok("concat", m.concat(&refs, 1))));
    let c1 = dispatches_of(&m, &|| drop(ok("concat", m.concat(&refs[..1], 1))));
    let s = dispatches_of(&m, &|| drop(ok("slice", m.slice(&parts[0], 2, 1, 3))));
    let f = dispatches_of(&m, &|| {
        drop(ok("fill", m.fill(&[2, 4, 6], DType::F32, 1.0)))
    });
    assert_eq!((c3, c1, s, f), (3, 1, 2, 1));
}
