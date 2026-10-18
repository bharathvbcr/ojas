//! The adapter gradient bank: per-sequence gradients of unequal widths sum
//! to the documented left fold, `Overwrite` resets, and a wrong-shaped,
//! wrong-dtype or unknown-index gradient is an error that leaves the slot
//! as it was. CPU, bit-exact: the bank's sum is a left fold of plain f32
//! adds in arrival order (`(g1 + g2) + g3`), so the expected value is built
//! the same way on the host and compared bit for bit.

use ojas_core::{Budget, DType, Numerics, Tensor};
use ojas_cpu::CpuBackend;
use ojas_model::qwen35::bank::{AdapterBank, BankEntry, BankError, BankMode};

const RANK: usize = 4;
const IN: usize = 6;
const SMALL: usize = 3;

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 28)).with_numerics(Numerics::Exact)
}

fn budget() -> Budget {
    Budget::new(1 << 28)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `dA = sum_t dh_t x_t^T` of a sequence of `width` positions, `[RANK, IN]`
/// row-major, with a fixed loop order: the gradient of a LoRA `A` over a
/// sequence, so sequences of different widths give different sums.
fn sequence_grad(width: usize, salt: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; RANK * IN];
    for t in 0..width {
        for r in 0..RANK {
            let dh = ((t * 5 + r * 3 + salt) % 13) as f32 * 0.173 - 1.01;
            for j in 0..IN {
                let x = ((t * 7 + j * 11 + salt) % 17) as f32 * 0.0931 - 0.77;
                g[r * IN + j] += dh * x;
            }
        }
    }
    g
}

fn small_grad(width: usize) -> Vec<f32> {
    (0..SMALL)
        .map(|i| (width * (i + 1)) as f32 * 0.1 - 0.9)
        .collect()
}

fn entries() -> Vec<BankEntry> {
    vec![
        BankEntry {
            name: "layers.0.q_proj.lora_A.weight".to_string(),
            shape: vec![RANK, IN],
        },
        BankEntry {
            name: "layers.0.q_proj.lora_B.weight".to_string(),
            shape: vec![SMALL],
        },
    ]
}

fn tensor(values: &[f32], shape: &[usize], b: &Budget) -> Tensor {
    Tensor::from_f32(values, shape, b).unwrap()
}

fn left_fold(parts: &[Vec<f32>]) -> Vec<f32> {
    let mut acc = parts[0].clone();
    for p in &parts[1..] {
        for (a, g) in acc.iter_mut().zip(p) {
            *a += *g;
        }
    }
    acc
}

#[test]
fn three_sequences_of_unequal_widths_sum_to_the_left_fold() {
    let (backend, b) = (cpu(), budget());
    let widths = [5usize, 9, 17];
    let a_parts: Vec<Vec<f32>> = widths
        .iter()
        .enumerate()
        .map(|(i, &w)| sequence_grad(w, i))
        .collect();
    let h_parts: Vec<Vec<f32>> = widths.iter().map(|&w| small_grad(w)).collect();
    // The three sequences really differ, so the sum is not one of them.
    assert_ne!(bits(&a_parts[0]), bits(&a_parts[1]));
    assert_ne!(bits(&a_parts[1]), bits(&a_parts[2]));

    let mut bank = AdapterBank::new(entries()).unwrap();
    assert!(!bank.is_complete());
    for (i, (a, h)) in a_parts.iter().zip(&h_parts).enumerate() {
        let mode = if i == 0 {
            BankMode::Overwrite
        } else {
            BankMode::Add
        };
        bank.accumulate(&backend, 0, mode, tensor(a, &[RANK, IN], &b))
            .unwrap();
        bank.accumulate(&backend, 1, mode, tensor(h, &[SMALL], &b))
            .unwrap();
    }
    assert!(bank.is_complete());

    let got_a = bank.grad(0).unwrap().to_f32_vec().unwrap();
    let got_h = bank.grad(1).unwrap().to_f32_vec().unwrap();
    assert_eq!(bits(&got_a), bits(&left_fold(&a_parts)));
    assert_eq!(bits(&got_h), bits(&left_fold(&h_parts)));

    let all = bank.grads().unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].shape(), &[RANK, IN]);
    assert_eq!(all[1].shape(), &[SMALL]);
}

#[test]
fn overwrite_resets_the_slot_and_the_next_add_starts_from_it() {
    let (backend, b) = (cpu(), budget());
    let mut bank = AdapterBank::new(entries()).unwrap();
    let old = sequence_grad(11, 0);
    let fresh = sequence_grad(3, 1);
    let next = sequence_grad(8, 2);
    bank.accumulate(
        &backend,
        0,
        BankMode::Overwrite,
        tensor(&old, &[RANK, IN], &b),
    )
    .unwrap();
    bank.accumulate(&backend, 0, BankMode::Add, tensor(&old, &[RANK, IN], &b))
        .unwrap();

    // A new step: Overwrite drops the two-sequence sum, nothing of it stays.
    bank.accumulate(
        &backend,
        0,
        BankMode::Overwrite,
        tensor(&fresh, &[RANK, IN], &b),
    )
    .unwrap();
    let got = bank.grad(0).unwrap().to_f32_vec().unwrap();
    assert_eq!(bits(&got), bits(&fresh));

    bank.accumulate(&backend, 0, BankMode::Add, tensor(&next, &[RANK, IN], &b))
        .unwrap();
    let got = bank.grad(0).unwrap().to_f32_vec().unwrap();
    assert_eq!(bits(&got), bits(&left_fold(&[fresh, next])));

    bank.clear();
    assert!(!bank.is_complete());
    assert!(matches!(
        bank.grad(0),
        Err(BankError::Unfilled { entry: 0, .. })
    ));
}

#[test]
fn a_wrong_gradient_is_an_error_and_the_slot_is_kept() {
    let (backend, b) = (cpu(), budget());
    let mut bank = AdapterBank::new(entries()).unwrap();
    let good = sequence_grad(6, 0);
    bank.accumulate(
        &backend,
        0,
        BankMode::Overwrite,
        tensor(&good, &[RANK, IN], &b),
    )
    .unwrap();

    // The same element count in another shape is still a shape error: it
    // is never reinterpreted.
    // An `Add` that is refused keeps the sum; a refused `Overwrite` empties
    // the slot instead (`qwen35_adapters_adversarial.rs`).
    let transposed = tensor(&good, &[IN, RANK], &b);
    let err = bank
        .accumulate(&backend, 0, BankMode::Add, transposed)
        .unwrap_err();
    assert!(
        matches!(&err, BankError::Shape { entry: 0, expected, got, .. }
            if expected == &[RANK, IN] && got == &[IN, RANK]),
        "{err}"
    );
    let short = tensor(&good[..RANK * IN - 1], &[RANK * IN - 1], &b);
    assert!(matches!(
        bank.accumulate(&backend, 0, BankMode::Add, short),
        Err(BankError::Shape { .. })
    ));
    let bf16 = Tensor::zeros(&[RANK, IN], DType::Bf16, &b).unwrap();
    let err = bank
        .accumulate(&backend, 0, BankMode::Add, bf16)
        .unwrap_err();
    assert!(
        matches!(
            err,
            BankError::Dtype {
                got: DType::Bf16,
                ..
            }
        ),
        "{err}"
    );

    // Every refusal left the slot as the first accumulate made it.
    let got = bank.grad(0).unwrap().to_f32_vec().unwrap();
    assert_eq!(bits(&got), bits(&good));
}

#[test]
fn index_and_missing_base_are_refused() {
    let (backend, b) = (cpu(), budget());
    let mut bank = AdapterBank::new(entries()).unwrap();
    let g = tensor(&sequence_grad(4, 0), &[RANK, IN], &b);
    assert!(matches!(
        bank.accumulate(&backend, 2, BankMode::Overwrite, g.clone()),
        Err(BankError::NoSuchEntry { entry: 2, len: 2 })
    ));
    // Add with no Overwrite first: refused, and the slot stays empty.
    assert!(matches!(
        bank.accumulate(&backend, 0, BankMode::Add, g),
        Err(BankError::AddWithoutBase { entry: 0, .. })
    ));
    assert!(matches!(
        bank.grads(),
        Err(BankError::Unfilled { entry: 0, .. })
    ));
}

#[test]
fn a_shape_whose_element_count_overflows_is_refused_at_construction() {
    let r = AdapterBank::new(vec![BankEntry {
        name: "huge".to_string(),
        shape: vec![usize::MAX, 2],
    }]);
    assert!(matches!(r, Err(BankError::ShapeOverflow { .. })));
}

fn filled_bank(backend: &CpuBackend, b: &Budget, a: &[f32], h: Option<&[f32]>) -> AdapterBank {
    let mut bank = AdapterBank::new(entries()).unwrap();
    bank.accumulate(backend, 0, BankMode::Overwrite, tensor(a, &[RANK, IN], b))
        .unwrap();
    if let Some(h) = h {
        bank.accumulate(backend, 1, BankMode::Overwrite, tensor(h, &[SMALL], b))
            .unwrap();
    }
    bank
}

#[test]
fn take_grads_moves_every_slot_out_in_entry_order() {
    let (backend, b) = (cpu(), budget());
    let (g0, g1) = (sequence_grad(5, 0), small_grad(9));
    let mut bank = AdapterBank::new(entries()).unwrap();
    bank.accumulate(
        &backend,
        0,
        BankMode::Overwrite,
        tensor(&g0, &[RANK, IN], &b),
    )
    .unwrap();
    bank.accumulate(&backend, 1, BankMode::Overwrite, tensor(&g1, &[SMALL], &b))
        .unwrap();
    let taken = bank.take_grads().unwrap();
    assert_eq!(taken.len(), 2);
    assert_eq!(bits(&taken[0].to_f32_vec().unwrap()), bits(&g0));
    assert_eq!(bits(&taken[1].to_f32_vec().unwrap()), bits(&g1));
    // The bank is empty after the move: a second take is refused at entry 0.
    assert!(!bank.is_complete());
    assert!(matches!(
        bank.take_grads(),
        Err(BankError::Unfilled { entry: 0, .. })
    ));
}

#[test]
fn take_grads_refuses_an_empty_slot_and_moves_nothing() {
    let (backend, b) = (cpu(), budget());
    let g0 = sequence_grad(5, 0);
    let mut bank = filled_bank(&backend, &b, &g0, None);
    assert!(matches!(
        bank.take_grads(),
        Err(BankError::Unfilled { entry: 1, .. })
    ));
    // Entry 0 was not moved by the refused call.
    let kept = bank.grad(0).unwrap().to_f32_vec().unwrap();
    assert_eq!(bits(&kept), bits(&g0));
}

#[test]
fn sq_norm_accumulates_in_f64_over_every_slot() {
    let (backend, b) = (cpu(), budget());
    let a = sequence_grad(13, 2);
    let h = small_grad(7);
    let bank = filled_bank(&backend, &b, &a, Some(&h));
    let mut reference = 0.0f64;
    for &x in a.iter().chain(&h) {
        reference += f64::from(x) * f64::from(x);
    }
    let got = bank.sq_norm(&backend).unwrap();
    assert_eq!(got.to_bits(), reference.to_bits());
}

#[test]
fn sq_norm_refuses_an_empty_slot_and_names_the_non_finite_entry() {
    let (backend, b) = (cpu(), budget());
    let a = sequence_grad(5, 0);
    let bank = filled_bank(&backend, &b, &a, None);
    assert!(matches!(
        bank.sq_norm(&backend),
        Err(BankError::Unfilled { entry: 1, .. })
    ));

    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut poisoned = sequence_grad(5, 1);
        poisoned[RANK * IN - 1] = bad;
        let mut bank = filled_bank(&backend, &b, &a, Some(&[1.0, 2.0, 3.0]));
        assert!(bank.sq_norm(&backend).is_ok());
        bank.accumulate(
            &backend,
            0,
            BankMode::Overwrite,
            tensor(&poisoned, &[RANK, IN], &b),
        )
        .unwrap();
        assert!(matches!(
            bank.sq_norm(&backend),
            Err(BankError::NonFinite { entry: 0, .. })
        ));
        bank.accumulate(
            &backend,
            1,
            BankMode::Overwrite,
            tensor(&[1.0, bad, 3.0], &[SMALL], &b),
        )
        .unwrap();
        bank.accumulate(
            &backend,
            0,
            BankMode::Overwrite,
            tensor(&a, &[RANK, IN], &b),
        )
        .unwrap();
        assert!(matches!(
            bank.sq_norm(&backend),
            Err(BankError::NonFinite { entry: 1, .. })
        ));
    }
}
