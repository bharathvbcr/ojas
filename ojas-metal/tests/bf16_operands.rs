//! Bf16 operands of the matmul-class ops on Metal (`Backend::bf16_operands`):
//! a `Bf16` operand gives the bits its widened `F32` copy gives, a resident
//! `Bf16` weight costs 2 bytes per element, and the widened copy is charged
//! only while the op runs. Device `to_bf16` / `to_f32` are bit-exact with
//! ojas-core's host conversions.
#![cfg(target_os = "macos")]

mod common;

use common::{down, host, metal, ok, rand, up, values, GIB};
use ojas_core::{bf16_to_f32, f32_to_bf16, Backend, Budget, DType, OjasError, Tensor};
use ojas_metal::MetalBackend;

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

/// `t`'s bf16 bits, read back from the device.
fn half_bits(t: &Tensor) -> Vec<u16> {
    let h = ok("download", t.to_host(&Budget::new(GIB)));
    ok("bits", h.bf16_slice()).to_vec()
}

/// A host `Bf16` tensor holding `data` rounded, and the `F32` tensor of the
/// same (bf16-exact) values.
fn pair(data: &[f32], shape: &[usize]) -> (Tensor, Tensor) {
    let b: Vec<u16> = data.iter().map(|&v| f32_to_bf16(v)).collect();
    let w: Vec<f32> = b.iter().map(|&h| bf16_to_f32(h)).collect();
    let half = ok(
        "bf16 host",
        Tensor::from_bf16_bits(&b, shape, &Budget::new(GIB)),
    );
    (half, host(&w, shape))
}

fn live(m: &MetalBackend) -> u64 {
    ok("live", m.budget().live_bytes())
}

fn patterns() -> Vec<f32> {
    let mut out = Vec::with_capacity(65536 * 3 + 8);
    for h in 0u32..=u16::MAX as u32 {
        out.push(f32::from_bits(h << 16));
        out.push(f32::from_bits((h << 16) | 0x7fff));
        out.push(f32::from_bits((h << 16) | 0x8000));
    }
    out.extend([
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xff80_0001),
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8000_0000),
        f32::from_bits(0x7f7f_ffff),
        f32::from_bits(0xff7f_ffff),
        f32::from_bits(0x7f7f_8000),
    ]);
    out
}

#[test]
fn metal_takes_bf16_operands() {
    assert!(metal().bf16_operands());
}

/// The backend-neutral contract (`ojas_oracle::bf16_operands`): every F32 /
/// Bf16 mix of the matmul-class ops and the tied table gives the all-F32
/// bits (packed and at an odd element offset), every other op refuses Bf16
/// with `Dtype`, a Bf16
/// NaN or infinity is `NonFinite`, and the device casts give the host
/// conversions' bits. The counts are the CPU's.
#[test]
fn metal_keeps_the_bf16_operand_contract() {
    let r = ojas_oracle::bf16_operands::bf16_operand_contract(&metal())
        .unwrap_or_else(|f| panic!("{f}"));
    eprintln!("metal: {r:?}");
    // The tied table: one Bf16 slot (2 layouts) in embedding_forward,
    // embedding_backward and 8 fused cross-entropy calls (2 chunks, each
    // asking for each of the four `CeGrads` combinations).
    assert!(r.table_lane);
    assert_eq!(r.mixed_calls, 2 * (2 * 10 + 2 * 38) + 2 * 10);
    assert_eq!(r.refusals, 2 + 14);
    assert_eq!(r.non_finite, 2 * 5 + 2 * 8 + 3);
}

#[test]
fn device_conversions_match_the_host_bit_for_bit() {
    let m = metal();
    let values = patterns();
    let n = values.len();
    let dev = up(&m, &host(&values, &[n]));
    let half = ok("to_bf16", m.to_bf16(&dev));
    assert_eq!(half.dtype(), DType::Bf16);
    assert_eq!(half.device(), Some(ojas_core::BackendId::Metal));
    let back = ok("to_f32", m.to_f32(&half));
    ok("sync", m.sync());
    let want: Vec<u16> = values.iter().map(|&v| f32_to_bf16(v)).collect();
    assert_eq!(half_bits(&half), want, "to_bf16 bits");
    let widened: Vec<u32> = want.iter().map(|&h| bf16_to_f32(h).to_bits()).collect();
    assert_eq!(bits(&back), widened, "to_f32 bits");

    // An odd element offset (2 bytes past a word) and an odd count.
    let window = ok("window", half.view(&[5], &[1], 2 * 3));
    let got = ok("to_f32 window", m.to_f32(&window));
    ok("sync window", m.sync());
    assert_eq!(bits(&got), widened[3..8].to_vec());

    // A host-built Bf16 tensor with an odd count round-trips through upload.
    let odd: Vec<u16> = want[..7].to_vec();
    let h = ok("odd", Tensor::from_bf16_bits(&odd, &[7], &Budget::new(GIB)));
    let d = up(&m, &h);
    assert_eq!(half_bits(&d), odd);
    let got = ok("to_f32 odd", m.to_f32(&d));
    ok("sync odd", m.sync());
    assert_eq!(bits(&got), widened[..7].to_vec());

    // Same dtype in: a shared clone, nothing new charged.
    let before = live(&m);
    let same = ok("to_f32 of f32", m.to_f32(&dev));
    let same_half = ok("to_bf16 of bf16", m.to_bf16(&half));
    assert_eq!(live(&m), before);
    drop((same, same_half));

    // Refusals: a host tensor is placement, a U32 one dtype.
    let err = m.to_f32(&h).unwrap_err();
    assert!(
        matches!(err, OjasError::Placement { op: "to_f32", .. }),
        "{err:?}"
    );
    let err = m.to_bf16(&host(&[1.0], &[1])).unwrap_err();
    assert!(
        matches!(err, OjasError::Placement { op: "to_bf16", .. }),
        "{err:?}"
    );
    let ids = up(&m, &common::host_u32(&[1, 2], &[2]));
    let err = m.to_f32(&ids).unwrap_err();
    assert!(
        matches!(err, OjasError::Dtype { op: "to_f32", .. }),
        "{err:?}"
    );
}

#[test]
fn a_resident_bf16_weight_costs_two_bytes_and_its_widened_copy_is_released() {
    let m = metal();
    let (out_f, in_f) = (96usize, 64usize);
    let (half, _) = pair(&values(out_f * in_f, 3, 0.5), &[out_f, in_f]);
    let base = live(&m);
    let w = up(&m, &half);
    assert_eq!(live(&m) - base, (out_f * in_f * 2) as u64, "resident bytes");
    let x = up(&m, &rand(&[8, in_f], 4, 1.0));
    let after_inputs = live(&m);
    let y = ok("linear bf16", m.linear_forward(&x, &w));
    ok("sync", m.sync());
    // Only the output stays charged: the widened weight was dropped.
    assert_eq!(live(&m) - after_inputs, (8 * out_f * 4) as u64);
    drop(y);
    assert_eq!(live(&m), after_inputs);

    // The widened copy is charged while the op runs: a budget with room
    // for the weight, the input and the output, but not the copy, refuses
    // the op.
    let room = (out_f * in_f * 2 + 8 * in_f * 4 + 8 * out_f * 4) as u64;
    let small = m.with_budget(Budget::new(room));
    let w3 = ok("upload small", small.upload(&half));
    let x3 = ok("x small", small.upload(&rand(&[8, in_f], 6, 1.0)));
    let err = small.linear_forward(&x3, &w3).unwrap_err();
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");
}

#[test]
fn linear_with_bf16_operands_gives_the_widened_f32_bits() {
    let m = metal();
    for &(rows, kin, nout) in &[(1usize, 7usize, 5usize), (13, 64, 96), (64, 256, 48)] {
        let (wh, wf) = pair(&values(nout * kin, 11, 0.25), &[nout, kin]);
        let (xh, xf) = pair(&values(rows * kin, 12, 1.0), &[rows, kin]);
        let (gh, gf) = pair(&values(rows * nout, 13, 1.0), &[rows, nout]);
        let (wb, wf) = (up(&m, &wh), up(&m, &wf));
        let (xb, xf) = (up(&m, &xh), up(&m, &xf));
        let (gb, gf) = (up(&m, &gh), up(&m, &gf));

        let want = ok("f32 fwd", m.linear_forward(&xf, &wf));
        let (want_gx, want_gw) = ok("f32 bwd", m.linear_backward(&xf, &wf, &gf));
        // A bf16 weight with f32 activations: the frozen-base case.
        let y = ok("fwd bf16 w", m.linear_forward(&xf, &wb));
        let (gx, gw) = ok("bwd bf16 w", m.linear_backward(&xf, &wb, &gf));
        // Every operand bf16.
        let y_all = ok("fwd all bf16", m.linear_forward(&xb, &wb));
        let (gx_all, gw_all) = ok("bwd all bf16", m.linear_backward(&xb, &wb, &gb));
        ok("sync", m.sync());
        let shape = format!("[{rows},{kin}]x[{nout},{kin}]");
        for (what, got, want) in [
            ("y", &y, &want),
            ("gx", &gx, &want_gx),
            ("gw", &gw, &want_gw),
            ("y all", &y_all, &want),
            ("gx all", &gx_all, &want_gx),
            ("gw all", &gw_all, &want_gw),
        ] {
            assert_eq!(got.dtype(), DType::F32, "{shape} {what}");
            assert_eq!(bits(got), bits(want), "{shape} {what}");
        }
    }
}

#[test]
fn causal_sdpa_with_bf16_operands_gives_the_widened_f32_bits() {
    let m = metal();
    let (b, h, kvh, t, d) = (1usize, 4usize, 2usize, 37usize, 64usize);
    let q = pair(&values(b * h * t * d, 21, 1.0), &[b, h, t, d]);
    let k = pair(&values(b * kvh * t * d, 22, 1.0), &[b, kvh, t, d]);
    let v = pair(&values(b * kvh * t * d, 23, 1.0), &[b, kvh, t, d]);
    let gy = pair(&values(b * h * t * d, 24, 1.0), &[b, h, t, d]);
    let [qb, kb, vb, gb] = [&q, &k, &v, &gy].map(|p| up(&m, &p.0));
    let [qf, kf, vf, gf] = [&q, &k, &v, &gy].map(|p| up(&m, &p.1));
    for window in [None, Some(9)] {
        let (want_y, want_lse) = ok("f32 fwd", m.causal_sdpa_forward(&qf, &kf, &vf, window));
        let (y, lse) = ok("bf16 fwd", m.causal_sdpa_forward(&qb, &kb, &vb, window));
        ok("sync fwd", m.sync());
        assert_eq!(bits(&y), bits(&want_y), "y {window:?}");
        assert_eq!(bits(&lse), bits(&want_lse), "lse {window:?}");

        // The forward output rounded to bf16, so both arms read the same
        // values: the f32 arm its widened copy, the bf16 arm the bf16 bits.
        let yb = ok("y to bf16", m.to_bf16(&want_y));
        let yf = ok("y widened", m.to_f32(&yb));
        let want = ok(
            "f32 bwd",
            m.causal_sdpa_backward(&qf, &kf, &vf, &yf, &want_lse, &gf, window),
        );
        let got = ok(
            "bf16 bwd",
            m.causal_sdpa_backward(&qb, &kb, &vb, &yb, &want_lse, &gb, window),
        );
        ok("sync bwd", m.sync());
        for (what, g, w) in [
            ("dq", &got.0, &want.0),
            ("dk", &got.1, &want.1),
            ("dv", &got.2, &want.2),
        ] {
            assert_eq!(g.dtype(), DType::F32);
            assert_eq!(bits(g), bits(w), "{what} {window:?}");
        }

        // The log-sum-exp stays F32.
        let lse_b = ok("lse to bf16", m.to_bf16(&want_lse));
        let err = m
            .causal_sdpa_backward(&qb, &kb, &vb, &yb, &lse_b, &gb, window)
            .unwrap_err();
        assert!(matches!(err, OjasError::Dtype { .. }), "{err:?}");
    }
}

#[test]
fn a_bf16_operand_that_rounded_to_infinity_is_a_non_finite_fault() {
    let m = metal();
    // f32::MAX rounds to +inf in bf16.
    let mut w = values(6 * 4, 31, 0.5);
    w[5] = f32::MAX;
    let wb = ok("to_bf16", m.to_bf16(&up(&m, &host(&w, &[6, 4]))));
    let x = up(&m, &rand(&[3, 4], 32, 1.0));
    let _y = ok("recorded", m.linear_forward(&x, &wb));
    let err = m.sync().unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::NonFinite {
                op: "linear_forward"
            }
        ),
        "{err:?}"
    );
}

/// The tied table resident as `Bf16`: the gather and the fused head read it
/// on the device without an `f32` copy of the table. The bits are the `F32`
/// table's (one vocabulary tile and several), and under a budget with room
/// for the 2-byte table, the inputs, the outputs and the call's scratch but
/// not a widened table, every op still runs: every device buffer a call
/// makes is charged, so a whole-table copy would be refused.
#[test]
fn a_bf16_tied_table_is_read_in_place_by_the_gather_and_the_fused_head() {
    let m = metal();
    let (vocab, dim, n) = (6000usize, 64usize, 24usize);
    let (half, full) = pair(&values(vocab * dim, 31, 0.5), &[vocab, dim]);
    let table_f32 = (vocab * dim * 4) as u64;
    let base = live(&m);
    let (th, tf) = (up(&m, &half), up(&m, &full));
    assert_eq!(live(&m) - base, table_f32 + table_f32 / 2, "resident bytes");
    let ids_host: Vec<u32> = (0..n as u32)
        .map(|i| (i * 977 + 5) % vocab as u32)
        .collect();
    let ids = up(&m, &common::host_u32(&ids_host, &[n]));
    let x = up(&m, &rand(&[n, dim], 32, 1.0));
    let g = up(&m, &rand(&[n, dim], 33, 1.0));

    let embed = |t: &Tensor| bits(&ok("embed", m.embedding_forward(t, &ids)));
    assert_eq!(embed(&th), embed(&tf), "gather");
    let embed_bwd = |t: &Tensor| bits(&ok("embed bwd", m.embedding_backward(t, &ids, &g)));
    assert_eq!(embed_bwd(&th), embed_bwd(&tf), "embedding backward");
    for chunk in [
        ojas_core::CeChunk {
            rows: 7,
            cols: 1024,
        },
        ojas_core::CeChunk {
            rows: n,
            cols: vocab,
        },
    ] {
        let lce = |t: &Tensor| -> Vec<Vec<u32>> {
            let r = ok(
                "lce",
                m.linear_cross_entropy_mean(&x, t, &ids, None, chunk, ojas_core::CeGrads::BOTH),
            );
            let mut out = vec![bits(&r.loss)];
            out.extend(r.grad_input.iter().map(bits));
            out.extend(r.grad_weight.iter().map(bits));
            out
        };
        assert_eq!(lce(&th), lce(&tf), "fused head {chunk:?}");
    }
    ok("sync", m.sync());

    // A budget for the 2-byte table, the inputs, the fused head's outputs
    // and scratch at 1024-column tiles, and less than a widened table.
    let chunk = ojas_core::CeChunk {
        rows: 8,
        cols: 1024,
    };
    let inputs = table_f32 / 2 + (n * 4 + 2 * n * dim * 4) as u64;
    let outputs = 4 + (n * dim * 4) as u64 + table_f32;
    let scratch = 4 * (8 * 1024 + 3 * 8 + n + 2 + 8 * dim + 2 * 1024 * dim) as u64;
    let room = inputs + outputs + scratch;
    assert!(
        room < table_f32 / 2 + outputs + table_f32,
        "the room must not fit a copy"
    );
    let small = m.with_budget(Budget::new(room));
    let th = ok("upload small", small.upload(&half));
    let ids = ok(
        "ids small",
        small.upload(&common::host_u32(&ids_host, &[n])),
    );
    let x = ok("x small", small.upload(&rand(&[n, dim], 32, 1.0)));
    let g = ok("g small", small.upload(&rand(&[n, dim], 33, 1.0)));
    let r = ok(
        "lce small",
        small.linear_cross_entropy_mean(&x, &th, &ids, None, chunk, ojas_core::CeGrads::BOTH),
    );
    ok("sync", small.sync());
    drop(r);
    drop(ok("embed small", small.embedding_forward(&th, &ids)));
    drop(ok(
        "embed bwd small",
        small.embedding_backward(&th, &ids, &g),
    ));
    ok("sync", small.sync());
}
