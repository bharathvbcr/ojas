//! The tied-table lane on the CPU (`Backend::bf16_operands`): a `Bf16`
//! embedding table and fused cross-entropy weight are read in place. Each
//! test runs the op on a bf16-exact `F32` table and on its `Bf16` twin and
//! pins three things: the bits are the `F32` table's, the charge has no
//! `f32` copy of the table in it (the gather and the backward charge their
//! output alone; the fused head one widened `[chunk.cols, d]` block on top of
//! its `F32` charge), and a bf16 NaN or infinity is refused like an `f32`
//! one, read or not.

mod common;

use common::{bits, SplitMix64};
use ojas_core::{
    f32_to_bf16, round_f32_to_bf16, Backend, Budget, CeChunk, DType, Numerics, OjasError, Tensor,
};
use ojas_cpu::CpuBackend;

const THREADS: [usize; 3] = [1, 2, 7];

fn backend(threads: usize, numerics: Numerics, cap: u64) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(cap), threads)
        .unwrap()
        .with_numerics(numerics)
}

/// Inputs live on their own budget, so the backend's sees only the op.
fn inputs() -> Budget {
    Budget::new(u64::MAX)
}

fn f(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &inputs()).unwrap()
}

fn u(data: &[u32], shape: &[usize]) -> Tensor {
    Tensor::from_u32(data, shape, &inputs()).unwrap()
}

/// `data` (bf16-exact) as a `Bf16` table, packed or as a contiguous view one
/// element into a larger allocation (a 2-byte offset).
fn half(data: &[f32], shape: &[usize], offset: bool) -> Tensor {
    let mut stored: Vec<u16> = data.iter().map(|&v| f32_to_bf16(v)).collect();
    if !offset {
        return Tensor::from_bf16_bits(&stored, shape, &inputs()).unwrap();
    }
    stored.insert(0, f32_to_bf16(9.0));
    let whole = Tensor::from_bf16_bits(&stored, &[stored.len()], &inputs()).unwrap();
    let strides = [shape[1], 1];
    whole.narrow(2, shape, &strides).unwrap()
}

/// `n` bf16-exact values in `[-scale, scale)`.
fn table_values(rng: &mut SplitMix64, n: usize, scale: f32) -> Vec<f32> {
    rng.vec(n, scale)
        .into_iter()
        .map(round_f32_to_bf16)
        .collect()
}

fn out_bits(t: &Tensor) -> Vec<u32> {
    bits(&t.to_f32_vec().unwrap())
}

#[test]
fn embedding_forward_widens_the_picked_rows_and_charges_only_the_output() {
    let (vocab, dim, tokens) = (3000usize, 96usize, 777usize);
    let mut rng = SplitMix64(0xb1);
    let mut table = table_values(&mut rng, vocab * dim, 1.0);
    table[5] = -0.0;
    table[6] = f32::from_bits(0x0001_0000); // the smallest bf16 subnormal
    let ids: Vec<u32> = (0..tokens).map(|_| rng.below(vocab) as u32).collect();
    let it = u(&ids, &[7, 111]);
    let out_bytes = (tokens * dim * 4) as u64;
    let want = out_bits(
        &backend(1, Numerics::Exact, out_bytes)
            .embedding_forward(&f(&table, &[vocab, dim]), &it)
            .unwrap(),
    );
    for offset in [false, true] {
        let ht = half(&table, &[vocab, dim], offset);
        for threads in THREADS {
            for numerics in [Numerics::Exact, Numerics::Fast] {
                // A budget of exactly the output: a widened table (another
                // 4 * vocab * dim bytes) could not be charged.
                let be = backend(threads, numerics, out_bytes);
                let y = be.embedding_forward(&ht, &it).unwrap();
                assert_eq!((y.shape(), y.dtype()), (&[7, 111, dim][..], DType::F32));
                assert!(out_bits(&y) == want, "offset {offset} threads {threads}");
                assert_eq!(be.budget().peak_bytes(), out_bytes);
                drop(y);
                let tight = backend(threads, numerics, out_bytes - 1);
                assert!(matches!(
                    tight.embedding_forward(&ht, &it),
                    Err(OjasError::CapacityExceeded { .. })
                ));
                assert_eq!(tight.budget().live_bytes().unwrap(), 0);
            }
        }
    }
}

#[test]
fn embedding_backward_of_a_bf16_table_gives_the_f32_tables_gradient() {
    let (vocab, dim, tokens) = (500usize, 48usize, 3000usize);
    let mut rng = SplitMix64(0xb2);
    let table = table_values(&mut rng, vocab * dim, 1.0);
    let ids: Vec<u32> = (0..tokens).map(|_| rng.below(vocab / 10) as u32).collect();
    let (it, grad) = (
        u(&ids, &[tokens]),
        f(&rng.vec(tokens * dim, 1.0), &[tokens, dim]),
    );
    let out_bytes = (vocab * dim * 4) as u64;
    let want = out_bits(
        &backend(1, Numerics::Exact, out_bytes)
            .embedding_backward(&f(&table, &[vocab, dim]), &it, &grad)
            .unwrap(),
    );
    for offset in [false, true] {
        let ht = half(&table, &[vocab, dim], offset);
        for threads in THREADS {
            let be = backend(threads, Numerics::Fast, out_bytes);
            let g = be.embedding_backward(&ht, &it, &grad).unwrap();
            assert_eq!((g.shape(), g.dtype()), (&[vocab, dim][..], DType::F32));
            assert!(out_bits(&g) == want, "offset {offset} threads {threads}");
            assert_eq!(be.budget().peak_bytes(), out_bytes);
        }
    }
}

/// A bf16 NaN or infinity anywhere in the table is refused, read or not, as
/// an `f32` one is; it outranks an out-of-range id and a budget with no
/// room.
#[test]
fn a_bf16_table_refuses_a_non_finite_value_in_an_unread_row() {
    let (vocab, dim) = (4000usize, 64usize);
    let ids = u(&[0, 1, 2], &[3]);
    let bad_ids = u(&[0, vocab as u32, 2], &[3]);
    let grad = f(&[1.0; 3 * 64], &[3, dim]);
    let x = f(&[0.5; 3 * 64], &[3, dim]);
    let chunk = CeChunk { rows: 2, cols: 512 };
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut table = vec![0.25f32; vocab * dim];
        table[(vocab - 1) * dim + 3] = bad;
        let ht = half(&table, &[vocab, dim], false);
        for threads in THREADS {
            let be = backend(threads, Numerics::Fast, 0);
            let nonfinite =
                |r: Result<(), OjasError>| matches!(r, Err(OjasError::NonFinite { .. }));
            assert!(nonfinite(be.embedding_forward(&ht, &ids).map(drop)));
            assert!(nonfinite(be.embedding_forward(&ht, &bad_ids).map(drop)));
            assert!(nonfinite(be.embedding_backward(&ht, &ids, &grad).map(drop)));
            assert!(nonfinite(
                be.linear_cross_entropy_mean(&x, &ht, &ids, None, chunk, true)
                    .map(drop)
            ));
            assert_eq!(be.budget().live_bytes().unwrap(), 0);
        }
    }
}

/// The fused head over a `Bf16` weight gives the `F32` weight's loss and
/// gradients bit for bit, at one vocabulary chunk and at several, and its
/// peak charge is the `F32` call's plus exactly one widened `[cols, d]`
/// block: no `f32` copy of the table, and a budget one byte short of that
/// peak is refused.
#[test]
fn the_fused_head_widens_one_vocabulary_block_at_a_time() {
    let (n, d, v) = (9usize, 64usize, 4096usize);
    let mut rng = SplitMix64(0xb3);
    let x = f(&rng.vec(n * d, 2.0), &[n, d]);
    let table = table_values(&mut rng, v * d, 0.5);
    let targets: Vec<u32> = (0..n).map(|_| rng.below(v) as u32).collect();
    let targets = u(&targets, &[n]);
    let wf = f(&table, &[v, d]);
    for (cols, rows) in [(256usize, 4usize), (v + 1, n), (1000, 9)] {
        let chunk = CeChunk { rows, cols };
        let block = (cols.min(v) * d * 4) as u64;
        for want_grad in [false, true] {
            let reference = backend(1, Numerics::Exact, u64::MAX);
            let want = reference
                .linear_cross_entropy_mean(&x, &wf, &targets, Some(3), chunk, want_grad)
                .unwrap();
            let f32_peak = reference.budget().peak_bytes();
            let mut want_bits = vec![out_bits(&want.loss)];
            want_bits.extend(want.grad_input.iter().map(out_bits));
            want_bits.extend(want.grad_weight.iter().map(out_bits));
            for offset in [false, true] {
                let ht = half(&table, &[v, d], offset);
                for threads in THREADS {
                    let be = backend(threads, Numerics::Exact, u64::MAX);
                    let got = be
                        .linear_cross_entropy_mean(&x, &ht, &targets, Some(3), chunk, want_grad)
                        .unwrap();
                    let mut got_bits = vec![out_bits(&got.loss)];
                    got_bits.extend(got.grad_input.iter().map(out_bits));
                    got_bits.extend(got.grad_weight.iter().map(out_bits));
                    let at = format!("cols {cols} grad {want_grad} offset {offset} t {threads}");
                    assert!(got_bits == want_bits, "{at}");
                    if let Some(gw) = &got.grad_weight {
                        assert_eq!(gw.dtype(), DType::F32, "{at}");
                    }
                    let peak = be.budget().peak_bytes();
                    if threads == 1 {
                        assert_eq!(peak, f32_peak + block, "{at}");
                    }
                    assert!(peak < f32_peak + (v * d * 4) as u64 || cols > v, "{at}");
                    drop(got);
                    let short = backend(threads, Numerics::Exact, peak - 1);
                    assert!(
                        matches!(
                            short.linear_cross_entropy_mean(
                                &x,
                                &ht,
                                &targets,
                                Some(3),
                                chunk,
                                want_grad
                            ),
                            Err(OjasError::CapacityExceeded { .. })
                        ),
                        "{at}"
                    );
                    assert_eq!(short.budget().live_bytes().unwrap(), 0, "{at}");
                }
            }
        }
    }
}

/// The table's other operands stay `F32` (`U32` for ids and targets): a
/// `Bf16` one is `Dtype`, before anything is charged.
#[test]
fn the_other_operands_of_the_table_ops_refuse_bf16() {
    let (vocab, dim) = (16usize, 8usize);
    let mut rng = SplitMix64(0xb4);
    let table = table_values(&mut rng, vocab * dim, 1.0);
    let ht = half(&table, &[vocab, dim], false);
    let ids = u(&[1, 2, 3], &[3]);
    let x = table_values(&mut rng, 3 * dim, 1.0);
    let xb = half(&x, &[3, dim], false);
    let chunk = CeChunk { rows: 2, cols: 4 };
    let be = backend(2, Numerics::Exact, u64::MAX);
    let dtype = |r: Result<(), OjasError>| {
        matches!(
            r,
            Err(OjasError::Dtype {
                got: DType::Bf16,
                ..
            })
        )
    };
    assert!(dtype(be.embedding_backward(&ht, &ids, &xb).map(drop)));
    assert!(dtype(
        be.linear_cross_entropy_mean(&xb, &ht, &ids, None, chunk, true)
            .map(drop)
    ));
    assert!(dtype(
        be.embedding_forward(&ht, &half(&[1.0], &[1, 1], false))
            .map(drop)
    ));
    assert_eq!(be.budget().peak_bytes(), 0);
}
