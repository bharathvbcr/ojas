//! Adversarial attack on `LoraLinear::merge` after it began writing into a
//! charged tensor: non-finite weights and deltas, bf16 weights, exact charges,
//! and a failed merge leaving the layer as it was.

use ojas_core::{bf16_to_f32, f32_to_bf16, Budget, DType, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_model::lora::{LoraConfig, LoraError, LoraLinear};

const OUT: usize = 6;
const INN: usize = 5;
const RANK: usize = 2;

fn fill(n: usize, k: f32) -> Vec<f32> {
    (0..n).map(|i| ((i % 13) as f32 - 6.0) * k).collect()
}

fn layer(budget: &Budget, w: Tensor, a: &[f32], b: &[f32], alpha: f32) -> LoraLinear {
    LoraLinear::new(
        LoraConfig::new(RANK, alpha, 0.0).unwrap(),
        w,
        Tensor::from_f32(a, &[RANK, INN], budget).unwrap(),
        Tensor::from_f32(b, &[OUT, RANK], budget).unwrap(),
    )
    .unwrap()
}

fn backend(budget: &Budget) -> CpuBackend {
    CpuBackend::new(budget.clone()).with_numerics(Numerics::Exact)
}

fn weight_bits(l: &LoraLinear) -> Vec<u32> {
    match l.weight().dtype() {
        DType::Bf16 => l
            .weight()
            .bf16_slice()
            .unwrap()
            .iter()
            .map(|&b| b as u32)
            .collect(),
        _ => l
            .weight()
            .f32_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect(),
    }
}

fn is_non_finite(e: &LoraError) -> bool {
    matches!(e, LoraError::Core(OjasError::NonFinite { .. }))
}

#[test]
fn a_non_finite_weight_or_delta_is_refused_and_leaves_the_layer_and_budget_alone() {
    let a = fill(RANK * INN, 0.1);
    let b = fill(OUT * RANK, 0.1);
    let w = fill(OUT * INN, 0.05);
    for (what, poke) in [
        ("nan", f32::NAN),
        ("inf", f32::INFINITY),
        ("-inf", f32::NEG_INFINITY),
    ] {
        // bad weight, f32 and bf16
        for dtype in [DType::F32, DType::Bf16] {
            let budget = Budget::new(1 << 20);
            let mut wv = w.clone();
            wv[7] = poke;
            let wt = if dtype == DType::F32 {
                Tensor::from_f32(&wv, &[OUT, INN], &budget).unwrap()
            } else {
                let bits: Vec<u16> = wv.iter().map(|&v| f32_to_bf16(v)).collect();
                Tensor::from_bf16_bits(&bits, &[OUT, INN], &budget).unwrap()
            };
            let mut l = layer(&budget, wt, &a, &b, 4.0);
            let before = weight_bits(&l);
            let live = budget.live_bytes().unwrap();
            let err = l.merge(&backend(&budget)).unwrap_err();
            assert!(is_non_finite(&err), "{what} {dtype:?} weight: {err}");
            assert!(!l.is_merged());
            assert_eq!(weight_bits(&l), before, "{what} {dtype:?}: weight changed");
            assert_eq!(budget.live_bytes().unwrap(), live, "{what} {dtype:?}: leak");
            // the layer is still usable: unmerge is NotMerged, merge errs again
            assert!(matches!(l.unmerge().unwrap_err(), LoraError::NotMerged));
            assert!(is_non_finite(&l.merge(&backend(&budget)).unwrap_err()));
        }
        // bad A or B, finite weight (the product is non-finite)
        for which in ["a", "b"] {
            let budget = Budget::new(1 << 20);
            let (mut av, mut bv) = (a.clone(), b.clone());
            if which == "a" {
                av[3] = poke;
            } else {
                bv[3] = poke;
            }
            let wt = Tensor::from_f32(&w, &[OUT, INN], &budget).unwrap();
            let mut l = layer(&budget, wt, &av, &bv, 4.0);
            let before = weight_bits(&l);
            let live = budget.live_bytes().unwrap();
            let err = l.merge(&backend(&budget)).unwrap_err();
            assert!(is_non_finite(&err), "{what} in {which}: {err}");
            assert_eq!(weight_bits(&l), before);
            assert_eq!(budget.live_bytes().unwrap(), live);
        }
    }
}

#[test]
fn a_scaled_delta_that_overflows_f32_is_refused() {
    let budget = Budget::new(1 << 20);
    let a = vec![1.0e20f32; RANK * INN];
    let b = vec![1.0e20f32; OUT * RANK];
    let wt = Tensor::from_f32(&fill(OUT * INN, 0.05), &[OUT, INN], &budget).unwrap();
    // B A is 1e40 * r: already inf in f32
    let mut l = layer(&budget, wt, &a, &b, 1.0);
    let before = weight_bits(&l);
    assert!(is_non_finite(&l.merge(&backend(&budget)).unwrap_err()));
    assert_eq!(weight_bits(&l), before);
    // finite product, scale overflows the sum
    let a = vec![1.0e10f32; RANK * INN];
    let b = vec![1.0e10f32; OUT * RANK];
    let wt = Tensor::from_f32(&fill(OUT * INN, 0.05), &[OUT, INN], &budget).unwrap();
    let mut l = layer(&budget, wt, &a, &b, f32::MAX);
    assert!(is_non_finite(&l.merge(&backend(&budget)).unwrap_err()));
}

#[test]
fn a_bf16_weight_merges_to_the_same_bits_as_its_widened_f32_copy_and_unmerges_exactly() {
    let a = fill(RANK * INN, 0.37);
    let b = fill(OUT * RANK, 0.29);
    let bits: Vec<u16> = fill(OUT * INN, 0.173)
        .iter()
        .map(|&v| f32_to_bf16(v))
        .collect();
    let widened: Vec<f32> = bits.iter().map(|&b| bf16_to_f32(b)).collect();

    let budget = Budget::new(1 << 20);
    let mut lb = layer(
        &budget,
        Tensor::from_bf16_bits(&bits, &[OUT, INN], &budget).unwrap(),
        &a,
        &b,
        6.0,
    );
    let mut lf = layer(
        &budget,
        Tensor::from_f32(&widened, &[OUT, INN], &budget).unwrap(),
        &a,
        &b,
        6.0,
    );
    let live = budget.live_bytes().unwrap();
    lb.merge(&backend(&budget)).unwrap();
    // the merged weight is f32 and the bf16 original is kept: one f32 weight more
    assert_eq!(lb.weight().dtype(), DType::F32);
    assert_eq!(budget.live_bytes().unwrap(), live + (OUT * INN * 4) as u64);
    lf.merge(&backend(&budget)).unwrap();
    assert_eq!(weight_bits(&lb), weight_bits(&lf));
    lb.unmerge().unwrap();
    assert_eq!(lb.weight().dtype(), DType::Bf16);
    assert_eq!(lb.weight().bf16_slice().unwrap(), &bits[..]);
    // unmerge frees the merged tensor
    assert_eq!(budget.live_bytes().unwrap(), live + (OUT * INN * 4) as u64);
    assert!(matches!(lb.unmerge().unwrap_err(), LoraError::NotMerged));
    lb.merge(&backend(&budget)).unwrap();
    assert!(matches!(
        lb.merge(&backend(&budget)).unwrap_err(),
        LoraError::AlreadyMerged
    ));
}

#[test]
fn a_merge_that_does_not_fit_leaves_weight_and_charge_untouched() {
    let a = fill(RANK * INN, 0.1);
    let b = fill(OUT * RANK, 0.1);
    let probe = Budget::new(1 << 20);
    let wt = Tensor::from_f32(&fill(OUT * INN, 0.05), &[OUT, INN], &probe).unwrap();
    let mut l = layer(&probe, wt, &a, &b, 4.0);
    probe.reset_peak();
    let live = probe.live_bytes().unwrap();
    l.merge(&backend(&probe)).unwrap();
    let need = probe.peak_bytes();
    assert!(need > live);

    // One byte under the merge's own peak must refuse, with nothing leaked.
    let cap = need - 1;
    let budget = Budget::new(cap);
    let wt = Tensor::from_f32(&fill(OUT * INN, 0.05), &[OUT, INN], &budget).unwrap();
    let mut l = layer(&budget, wt, &a, &b, 4.0);
    let before = weight_bits(&l);
    let live = budget.live_bytes().unwrap();
    let err = l.merge(&backend(&budget)).unwrap_err();
    assert!(
        matches!(err, LoraError::Core(OjasError::CapacityExceeded { .. })),
        "{err}"
    );
    assert!(!l.is_merged());
    assert_eq!(weight_bits(&l), before);
    assert_eq!(budget.live_bytes().unwrap(), live);
}

#[test]
fn a_non_float_weight_is_refused_not_misread() {
    let budget = Budget::new(1 << 20);
    let wt = Tensor::from_u32(&[1u32; OUT * INN], &[OUT, INN], &budget).unwrap();
    let mut l = layer(
        &budget,
        wt,
        &fill(RANK * INN, 0.1),
        &fill(OUT * RANK, 0.1),
        4.0,
    );
    let live = budget.live_bytes().unwrap();
    assert!(l.merge(&backend(&budget)).is_err());
    assert!(!l.is_merged());
    assert_eq!(l.weight().dtype(), DType::U32);
    assert_eq!(budget.live_bytes().unwrap(), live);
}
