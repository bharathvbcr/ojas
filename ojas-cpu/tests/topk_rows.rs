//! The trait's host `topk_rows` (CpuBackend keeps the default): each row's
//! `k` leaders by value descending under `f32::total_cmp`, then column
//! ascending, against a full sort of the row by the same order.

use ojas_core::{Backend, Budget, DType, OjasError, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{bits, SplitMix64};

/// Qwen's vocabulary.
const COLS: usize = 248_320;
const ROWS: usize = 4;
const KS: [usize; 3] = [1, 7, 50];

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 30))
}

/// Four rows at Qwen's vocabulary:
/// 0. 512 levels, so hundreds of exact ties at every level; `-inf` every
///    97th column.
/// 1. every value at most `-1.0` (hundreds of ties at `-1.0`) except
///    interleaved `+0.0` and `-0.0` columns, and `-inf` every 101st column.
/// 2. three finite values (two equal), the rest `-inf`.
/// 3. 20,000 levels, so the 50 leaders span several tied levels.
fn logits() -> Vec<f32> {
    let mut r = SplitMix64(9);
    let mut x = vec![0.0f32; ROWS * COLS];
    for (c, v) in x[..COLS].iter_mut().enumerate() {
        *v = if c % 97 == 3 {
            f32::NEG_INFINITY
        } else {
            ((r.next_u64() >> 40) % 512) as f32 * 0.5 - 100.0
        };
    }
    for (c, v) in x[COLS..2 * COLS].iter_mut().enumerate() {
        *v = if c % 101 == 4 {
            f32::NEG_INFINITY
        } else {
            -(((r.next_u64() >> 40) % 512) as f32 * 0.5) - 1.0
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
        *v = ((r.next_u64() >> 40) % 20_000) as f32;
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

#[track_caller]
fn same_leaders(name: &str, x: &[f32], cols: usize, k: usize, values: &Tensor, ids: &Tensor) {
    let rows = x.len() / cols;
    assert_eq!(values.shape(), [rows, k], "{name} k={k}");
    assert_eq!(ids.shape(), [rows, k], "{name} k={k}");
    assert_eq!(values.dtype(), DType::F32);
    assert_eq!(ids.dtype(), DType::U32);
    let got_v = values.to_f32_vec().unwrap();
    let got_i = ids.u32_slice().unwrap();
    for (r, order) in reference(x, cols).iter().enumerate() {
        let want_i = &order[..k];
        let want_v: Vec<f32> = want_i.iter().map(|&c| x[r * cols + c as usize]).collect();
        assert_eq!(
            &got_i[r * k..(r + 1) * k],
            want_i,
            "{name} k={k} row {r} ids"
        );
        assert_eq!(
            bits(&got_v[r * k..(r + 1) * k]),
            bits(&want_v),
            "{name} k={k} row {r} values"
        );
    }
}

#[test]
fn topk_rows_default_matches_a_full_sort_at_qwen_vocab() {
    let be = cpu();
    let host = logits();
    let x = Tensor::from_f32(&host, &[ROWS, COLS], be.budget()).unwrap();
    for k in KS {
        let (values, ids) = be.topk_rows(&x, k).unwrap();
        assert!(values.device().is_none() && ids.device().is_none());
        same_leaders("cpu", &host, COLS, k, &values, &ids);
    }
    // Spelled out: +0.0 columns before -0.0 columns, each in column order;
    // a row with fewer finite values than k ends in -inf in column order.
    let (values, ids) = be.topk_rows(&x, 50).unwrap();
    let ids = ids.u32_slice().unwrap();
    let values = values.to_f32_vec().unwrap();
    assert_eq!(ids[50..58], [6, 99, 7_001, 200_000, 5, 100, 7_000, 248_000]);
    assert_eq!(bits(&values[50..58])[..4], [0u32; 4]);
    assert_eq!(bits(&values[50..58])[4..], [0x8000_0000u32; 4]);
    assert_eq!(values[58], -1.0);
    assert_eq!(ids[100..107], [10, 300, COLS as u32 - 1, 0, 1, 2, 3]);
    assert_eq!(values[100..103], [2.5, 2.5, -7.0]);
    assert!(values[103..150].iter().all(|&v| v == f32::NEG_INFINITY));
    // The 47th -inf column: 0..=47 without the finite column 10.
    assert_eq!(ids[149], 47);
}

/// `k == cols` (no selection step, only the sort) and a single column.
#[test]
fn topk_rows_default_takes_k_equal_to_cols() {
    let be = cpu();
    let host = [
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
    let x = Tensor::from_f32(&host, &[2, 5], be.budget()).unwrap();
    let (values, ids) = be.topk_rows(&x, 5).unwrap();
    same_leaders("cpu", &host, 5, 5, &values, &ids);
    assert_eq!(ids.u32_slice().unwrap(), [0, 4, 3, 1, 2, 2, 3, 4, 0, 1]);
    let one = Tensor::from_f32(&[f32::NEG_INFINITY], &[1, 1], be.budget()).unwrap();
    let (values, ids) = be.topk_rows(&one, 1).unwrap();
    assert_eq!(values.to_f32_vec().unwrap(), [f32::NEG_INFINITY]);
    assert_eq!(ids.u32_slice().unwrap(), [0]);
}

/// NaN (either sign) and `+inf` anywhere in any row are `NonFinite`, even
/// where they would not be among the leaders; `k` outside `1..=cols`, a
/// non-F32 or non-rank-2 operand are refused; a refusal charges nothing.
#[test]
fn topk_rows_default_refuses_nonfinite_and_bad_k() {
    let be = cpu();
    let mut host = logits();
    for bad in [f32::NAN, -f32::NAN, f32::INFINITY] {
        let keep = host[3 * COLS + 17];
        host[3 * COLS + 17] = bad;
        let x = Tensor::from_f32(&host, &[ROWS, COLS], &Budget::new(1 << 30)).unwrap();
        let err = be.topk_rows(&x, 7).unwrap_err();
        assert!(
            matches!(err, OjasError::NonFinite { op: "topk_rows" }),
            "{bad}: {err:?}"
        );
        host[3 * COLS + 17] = keep;
    }
    let x = Tensor::from_f32(&[1.0; 6], &[2, 3], &Budget::new(1 << 20)).unwrap();
    for k in [0, 4, usize::MAX] {
        let err = be.topk_rows(&x, k).unwrap_err();
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
    let flat = Tensor::from_f32(&[1.0; 6], &[6], &Budget::new(1 << 20)).unwrap();
    assert!(matches!(
        be.topk_rows(&flat, 1).unwrap_err(),
        OjasError::Shape {
            op: "topk_rows",
            ..
        }
    ));
    let ids = Tensor::from_u32(&[1; 6], &[2, 3], &Budget::new(1 << 20)).unwrap();
    assert!(matches!(
        be.topk_rows(&ids, 1).unwrap_err(),
        OjasError::Dtype {
            op: "topk_rows",
            ..
        }
    ));
    assert_eq!(be.budget().live_bytes().unwrap(), 0);
}

/// The ids index an embedding table directly.
#[test]
fn topk_rows_default_ids_feed_embedding() {
    const DIM: usize = 3;
    let be = cpu();
    let host = logits();
    let x = Tensor::from_f32(&host, &[ROWS, COLS], be.budget()).unwrap();
    let (_, ids) = be.topk_rows(&x, 7).unwrap();
    let table: Vec<f32> = (0..COLS * DIM).map(|i| (i / DIM) as f32).collect();
    let table = Tensor::from_f32(&table, &[COLS, DIM], be.budget()).unwrap();
    let rows = be.embedding_forward(&table, &ids).unwrap();
    assert_eq!(rows.shape(), [ROWS, 7, DIM]);
    let rows = rows.to_f32_vec().unwrap();
    for (i, &id) in ids.u32_slice().unwrap().iter().enumerate() {
        assert_eq!(rows[i * DIM..(i + 1) * DIM], [id as f32; DIM], "slot {i}");
    }
}
