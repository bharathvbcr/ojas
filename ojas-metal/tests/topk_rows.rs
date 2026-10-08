//! `ojas_topk_rows`: each row's `k` leaders by value descending under
//! `f32::total_cmp`, then column ascending, against a full sort of the row
//! by the same order; the ids stay on the device and feed
//! `embedding_forward` with no host copy; NaN and `+inf` are deferred faults.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::{host, metal, ok, up};
use ojas_core::{Backend, DType, OjasError, Tensor};
use ojas_metal::{MetalBackend, METAL_TOPK_MAX_K};

/// Qwen's vocabulary.
const COLS: usize = 248_320;
const ROWS: usize = 4;
const KS: [usize; 3] = [1, 7, 50];

fn next(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Four rows at Qwen's vocabulary:
/// 0. 512 levels, so hundreds of exact ties at every level; `-inf` every
///    97th column.
/// 1. every value at most `-1.0` (hundreds of ties at `-1.0`) except
///    interleaved `+0.0` and `-0.0` columns, and `-inf` every 101st column.
/// 2. three finite values (two equal), the rest `-inf`.
/// 3. 20,000 levels, so the 50 leaders span several tied levels.
fn logits() -> Vec<f32> {
    let mut s = 9u64;
    let mut x = vec![0.0f32; ROWS * COLS];
    for (c, v) in x[..COLS].iter_mut().enumerate() {
        *v = if c % 97 == 3 {
            f32::NEG_INFINITY
        } else {
            ((next(&mut s) >> 40) % 512) as f32 * 0.5 - 100.0
        };
    }
    for (c, v) in x[COLS..2 * COLS].iter_mut().enumerate() {
        *v = if c % 101 == 4 {
            f32::NEG_INFINITY
        } else {
            -(((next(&mut s) >> 40) % 512) as f32 * 0.5) - 1.0
        };
    }
    for c in [5, 100, 7_000, 248_000] {
        x[COLS + c] = -0.0;
    }
    for c in [6, 99, 7_001, 200_000] {
        x[COLS + c] = 0.0;
    }
    x[2 * COLS..3 * COLS].fill(f32::NEG_INFINITY);
    x[2 * COLS + 300] = 2.5;
    x[2 * COLS + 10] = 2.5;
    x[2 * COLS + COLS - 1] = -7.0;
    for v in &mut x[3 * COLS..] {
        *v = ((next(&mut s) >> 40) % 20_000) as f32;
    }
    x
}

/// Every row's columns fully sorted by the contract's order.
fn reference(x: &[f32], cols: usize) -> Vec<Vec<u32>> {
    x.chunks(cols)
        .map(|row| {
            let mut order: Vec<u32> = (0..cols as u32).collect();
            order.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]).then(a.cmp(&b)));
            order
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Run `topk_rows` on the device and compare with the reference: shapes,
/// dtypes, residency, value bits and ids.
#[track_caller]
fn check(m: &MetalBackend, x: &[f32], cols: usize, ks: &[usize]) -> Vec<(Tensor, Tensor)> {
    let rows = x.len() / cols;
    let order = reference(x, cols);
    let dev = up(m, &host(x, &[rows, cols]));
    let mut out = Vec::new();
    for &k in ks {
        let (values, ids) = ok("topk_rows", m.topk_rows(&dev, k));
        assert_eq!(values.shape(), [rows, k]);
        assert_eq!(ids.shape(), [rows, k]);
        assert_eq!((values.dtype(), ids.dtype()), (DType::F32, DType::U32));
        assert!(values.device().is_some() && ids.device().is_some());
        let got_v = ok("download", m.download(&values)).to_f32_vec().unwrap();
        let got_i = ok("download", m.download(&ids));
        ok("sync", m.sync());
        let got_i = got_i.u32_slice().unwrap();
        for (r, order) in order.iter().enumerate() {
            let want_i = &order[..k];
            let want_v: Vec<f32> = want_i.iter().map(|&c| x[r * cols + c as usize]).collect();
            assert_eq!(&got_i[r * k..(r + 1) * k], want_i, "k={k} row {r} ids");
            assert_eq!(
                bits(&got_v[r * k..(r + 1) * k]),
                bits(&want_v),
                "k={k} row {r} values"
            );
        }
        out.push((values, ids));
    }
    out
}

#[test]
fn topk_rows_matches_a_full_sort_at_qwen_vocab() {
    let m = metal();
    let x = logits();
    let out = check(&m, &x, COLS, &KS);
    // Spelled out at k = 50: +0.0 columns before -0.0 columns, each in
    // column order; a row with fewer finite values than k ends in -inf in
    // column order.
    let (values, ids) = &out[2];
    let values = m.download(values).unwrap().to_f32_vec().unwrap();
    let ids = m.download(ids).unwrap();
    m.sync().unwrap();
    let ids = ids.u32_slice().unwrap();
    assert_eq!(ids[50..58], [6, 99, 7_001, 200_000, 5, 100, 7_000, 248_000]);
    assert_eq!(bits(&values[50..54]), [0u32; 4]);
    assert_eq!(bits(&values[54..58]), [0x8000_0000u32; 4]);
    assert_eq!(ids[100..107], [10, 300, COLS as u32 - 1, 0, 1, 2, 3]);
    assert_eq!(values[100..103], [2.5, 2.5, -7.0]);
    assert!(values[103..150].iter().all(|&v| v == f32::NEG_INFINITY));
    // The 47th -inf column: 0..=47 without the finite column 10.
    assert_eq!(ids[149], 47);
}

/// `k == cols`, a single column, and more rows than one threadgroup lane
/// count with fewer columns than lanes.
#[test]
fn topk_rows_small_shapes() {
    let m = metal();
    let x = [
        3.0,
        -0.0,
        f32::NEG_INFINITY,
        0.0,
        3.0,
        -2.0,
        -2.0,
        1.0,
        0.0,
        -0.0,
    ];
    check(&m, &x, 5, &[1, 3, 5]);
    check(&m, &[f32::NEG_INFINITY, -1.0], 1, &[1]);
    let mut s = 3u64;
    let many: Vec<f32> = (0..300 * 9)
        .map(|_| ((next(&mut s) >> 40) % 4) as f32 - 1.5)
        .collect();
    check(&m, &many, 9, &[1, 4, 9]);
}

/// The ids index an embedding table straight from the device (no host copy
/// of them exists); a table smaller than their bound is refused.
#[test]
fn topk_rows_ids_feed_embedding_from_the_device() {
    const DIM: usize = 4;
    let m = metal();
    let x = logits();
    let dev = up(&m, &host(&x, &[ROWS, COLS]));
    let (_, ids) = ok("topk_rows", m.topk_rows(&dev, 7));
    let table: Vec<f32> = (0..COLS * DIM).map(|i| (i / DIM) as f32).collect();
    let rows = ok(
        "embedding",
        m.embedding_forward(&up(&m, &host(&table, &[COLS, DIM])), &ids),
    );
    assert_eq!(rows.shape(), [ROWS, 7, DIM]);
    let rows = m.download(&rows).unwrap().to_f32_vec().unwrap();
    let want = m.download(&ids).unwrap();
    m.sync().unwrap();
    for (i, &id) in want.u32_slice().unwrap().iter().enumerate() {
        assert_eq!(rows[i * DIM..(i + 1) * DIM], [id as f32; DIM], "slot {i}");
    }
    let small = up(&m, &host(&table[..(COLS - 1) * DIM], &[COLS - 1, DIM]));
    let err = m.embedding_forward(&small, &ids).unwrap_err();
    assert!(matches!(err, OjasError::OutOfRange { .. }), "{err:?}");
}

/// NaN (either sign) and `+inf` in any row are `NonFinite` by the next sync,
/// even where they would not be among the leaders.
#[test]
fn topk_rows_nonfinite_is_a_deferred_fault() {
    let mut x = logits();
    for bad in [f32::NAN, -f32::NAN, f32::INFINITY] {
        // A fresh backend: a deferred fault may leave the last one poisoned.
        let m = metal();
        let keep = x[3 * COLS + 17];
        x[3 * COLS + 17] = bad;
        let dev = up(&m, &host(&x, &[ROWS, COLS]));
        let err = m
            .topk_rows(&dev, 7)
            .and_then(|(values, _)| m.download(&values))
            .and_then(|_| m.sync())
            .unwrap_err();
        assert!(
            matches!(err, OjasError::NonFinite { op: "topk_rows" }),
            "{bad}: {err:?}"
        );
        x[3 * COLS + 17] = keep;
    }
}

/// `k` outside `1..=cols` is `OutOfRange`, `k` past the kernel's bound is
/// `Unsupported`, both before any device work.
#[test]
fn topk_rows_refusals() {
    let m = metal();
    let x = up(&m, &host(&[1.0; 6], &[2, 3]));
    for k in [0, 4, usize::MAX] {
        let err = m.topk_rows(&x, k).unwrap_err();
        assert!(
            matches!(
                err,
                OjasError::OutOfRange {
                    op: "topk_rows",
                    ..
                }
            ),
            "k={k}: {err:?}"
        );
    }
    let wide = up(
        &m,
        &host(&vec![1.0; METAL_TOPK_MAX_K + 1], &[1, METAL_TOPK_MAX_K + 1]),
    );
    let err = m.topk_rows(&wide, METAL_TOPK_MAX_K + 1).unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::Unsupported {
                op: "topk_rows",
                ..
            }
        ),
        "{err:?}"
    );
    let (values, ids) = ok("topk at the bound", m.topk_rows(&wide, METAL_TOPK_MAX_K));
    let got = m.download(&ids).unwrap();
    let values = m.download(&values).unwrap().to_f32_vec().unwrap();
    m.sync().unwrap();
    let want: Vec<u32> = (0..METAL_TOPK_MAX_K as u32).collect();
    assert_eq!(got.u32_slice().unwrap(), want.as_slice());
    assert!(values.iter().all(|&v| v == 1.0));
}
