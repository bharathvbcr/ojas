//! Gather correctness for `CpuBackend::embedding_forward` and the token-order
//! scatter of `embedding_backward`.
//!
//! The reference below is a scalar row copy (forward) and a scalar running
//! sum in token order (backward). It does not call the op. Thread count must
//! not change bits. A duplicate id copies the same row twice on the forward
//! path and adds in token order on the backward path.

use ojas_core::{Backend, Budget, DType, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_capacity, assert_nonfinite, assert_range, assert_shape, bits, SplitMix64};

fn cpu(threads: usize, cap: u64) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(cap), threads).unwrap()
}

/// Inputs live on their own budget, so a failed op leaves the backend at 0.
fn f32_owned(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn u32_owned(data: &[u32], shape: &[usize]) -> Tensor {
    Tensor::from_u32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

/// Output row-major. Independent of `embedding_forward`.
fn forward_reference(table: &[f32], ids: &[u32], dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(ids.len() * dim);
    for &id in ids {
        let start = id as usize * dim;
        out.extend_from_slice(&table[start..start + dim]);
    }
    out
}

/// Table gradient from 0, each token's row added in ascending token order.
fn backward_reference(ids: &[u32], grad: &[f32], vocab: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; vocab * dim];
    for (n, &id) in ids.iter().enumerate() {
        let row = id as usize * dim;
        for c in 0..dim {
            out[row + c] += grad[n * dim + c];
        }
    }
    out
}

fn check_forward(threads: usize, vocab: usize, dim: usize, ids: &[u32], table: &[f32]) {
    let backend = cpu(threads, 1 << 30);
    let tt = f32_owned(table, &[vocab, dim]);
    let it = u32_owned(ids, &[ids.len()]);
    let y = backend
        .embedding_forward(&tt, &it)
        .unwrap_or_else(|err| panic!("threads {threads} vocab {vocab} dim {dim}: {err}"));
    assert_eq!(y.shape(), &[ids.len(), dim]);
    assert!(y.is_contiguous().unwrap());
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(
        bits(&y.to_f32_vec().unwrap()),
        bits(&forward_reference(table, ids, dim)),
        "threads {threads} vocab {vocab} dim {dim} tokens {}",
        ids.len()
    );
    let out_bytes = (ids.len() * dim * 4) as u64;
    assert_eq!(backend.budget().live_bytes().unwrap(), out_bytes);
    drop(y);
    assert_eq!(backend.budget().live_bytes().unwrap(), 0);
}

fn check_backward(threads: usize, vocab: usize, dim: usize, ids: &[u32], grad: &[f32]) {
    let backend = cpu(threads, 1 << 30);
    let table = f32_owned(&vec![0.25; vocab * dim], &[vocab, dim]);
    let it = u32_owned(ids, &[ids.len()]);
    let gt = f32_owned(grad, &[ids.len(), dim]);
    let g = backend
        .embedding_backward(&table, &it, &gt)
        .unwrap_or_else(|err| panic!("bwd threads {threads}: {err}"));
    assert_eq!(
        bits(&g.to_f32_vec().unwrap()),
        bits(&backward_reference(ids, grad, vocab, dim)),
        "bwd threads {threads} vocab {vocab} dim {dim}"
    );
}

#[test]
fn gather_matches_scalar_reference_at_one_and_six_threads() {
    // Odd dims and token counts, a duplicate, and the last vocab row.
    // The wide case is large enough to split across the scoped pool.
    let cases: &[(usize, usize, usize)] = &[
        (4, 1, 1),
        (5, 3, 1),
        (8, 7, 13),
        (9, 17, 3),
        (6, 63, 5),
        // 128 rows of 768 splits into several scoped chunks (min chunk is 42 rows).
        (16, 768, 128),
        (4, 769, 3),
    ];
    for threads in [1usize, 6] {
        for &(vocab, dim, tokens) in cases {
            let mut table =
                SplitMix64(0x0e3b_0000 + (dim as u64) * 17 + tokens as u64).vec(vocab * dim, 2.0);
            table[0] = -0.0;
            if table.len() > 4 {
                let last = table.len() - 1;
                table[1] = f32::from_bits(1);
                table[last] = f32::MIN;
            }
            let mut ids = Vec::with_capacity(tokens);
            for n in 0..tokens {
                let id = if n % 5 == 0 {
                    0
                } else if n % 5 == 1 {
                    (vocab - 1) as u32
                } else {
                    (n % vocab) as u32
                };
                ids.push(id);
            }
            // Two leading copies of the same row, including when tokens == 1.
            if tokens >= 2 {
                ids[1] = ids[0];
            }
            check_forward(threads, vocab, dim, &ids, &table);
        }
    }
}

#[test]
fn duplicate_ids_scatter_in_token_order_at_one_and_six_threads() {
    // (1e20 + -1e20) + 1.0 is 1.0 in token order. A different association
    // of the same three rows is not.
    const VOCAB: usize = 4;
    const DIM: usize = 3;
    let ids = [2u32, 2, 2, 0, VOCAB as u32 - 1];
    let mut grad = vec![0.0f32; ids.len() * DIM];
    // Three visits of row 2, in token order: (1e20 + -1e20) + 1.0.
    grad[0] = 1e20;
    grad[DIM] = -1e20;
    grad[2 * DIM] = 1.0;
    grad[3 * DIM + 1] = -0.0;
    grad[4 * DIM] = f32::from_bits(1);
    for threads in [1usize, 6] {
        check_backward(threads, VOCAB, DIM, &ids, &grad);
    }
    let one = cpu(1, 1 << 20);
    let six = cpu(6, 1 << 20);
    let table = f32_owned(&[0.5; VOCAB * DIM], &[VOCAB, DIM]);
    let it = u32_owned(&ids, &[ids.len()]);
    let gt = f32_owned(&grad, &[ids.len(), DIM]);
    let a = one.embedding_backward(&table, &it, &gt).unwrap();
    let b = six.embedding_backward(&table, &it, &gt).unwrap();
    assert_eq!(
        bits(&a.to_f32_vec().unwrap()),
        bits(&b.to_f32_vec().unwrap())
    );
}

#[test]
fn nan_or_inf_in_an_unread_row_is_refused_and_releases_the_budget() {
    let (vocab, dim) = (8usize, 5usize);
    for threads in [1usize, 6] {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let backend = cpu(threads, 1 << 20);
            let mut table = vec![0.25f32; vocab * dim];
            // Last row, never named by an id.
            table[(vocab - 1) * dim + 2] = bad;
            let tt = f32_owned(&table, &[vocab, dim]);
            let ids = u32_owned(&[0, 1, 0], &[3]);
            assert_eq!(backend.budget().live_bytes().unwrap(), 0);
            assert_nonfinite(backend.embedding_forward(&tt, &ids));
            assert_eq!(backend.budget().live_bytes().unwrap(), 0);
            let grad = f32_owned(&vec![1.0; 3 * dim], &[3, dim]);
            assert_nonfinite(backend.embedding_backward(&tt, &ids, &grad));
            assert_eq!(backend.budget().live_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn out_of_range_id_and_a_tight_budget_release_every_charge() {
    let (vocab, dim) = (6usize, 7usize);
    let table = {
        let mut t = SplitMix64(11).vec(vocab * dim, 1.0);
        t[0] = -0.0;
        t
    };
    for threads in [1usize, 6] {
        let backend = cpu(threads, 1 << 20);
        let tt = f32_owned(&table, &[vocab, dim]);
        let bad = u32_owned(&[0, vocab as u32, 1], &[3]);
        assert_range(backend.embedding_forward(&tt, &bad));
        assert_eq!(backend.budget().live_bytes().unwrap(), 0);

        let ids = u32_owned(&[0, vocab as u32 - 1, 0], &[1, 3]);
        let out_bytes = (3 * dim * 4) as u64;
        let tight = cpu(threads, out_bytes - 1);
        assert_capacity(tight.embedding_forward(&tt, &ids));
        assert_eq!(tight.budget().live_bytes().unwrap(), 0);

        let ok = cpu(threads, out_bytes);
        let y = ok.embedding_forward(&tt, &ids).unwrap();
        assert_eq!(ok.budget().live_bytes().unwrap(), out_bytes);
        assert_eq!(
            bits(&y.to_f32_vec().unwrap()),
            bits(&forward_reference(&table, &[0, vocab as u32 - 1, 0], dim))
        );
    }
}

#[test]
fn empty_table_or_ids_are_an_empty_tensor_refusal() {
    for threads in [1usize, 6] {
        let backend = cpu(threads, 1 << 20);
        let table = f32_owned(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let empty_ids = Tensor::zeros(&[0], DType::U32, &Budget::new(u64::MAX)).unwrap();
        assert_shape(backend.embedding_forward(&table, &empty_ids));
        assert_eq!(backend.budget().live_bytes().unwrap(), 0);
        let empty_table = Tensor::zeros(&[2, 0], DType::F32, &Budget::new(u64::MAX)).unwrap();
        let ids = u32_owned(&[0], &[1]);
        assert_shape(backend.embedding_forward(&empty_table, &ids));
        assert_eq!(backend.budget().live_bytes().unwrap(), 0);
    }
}
