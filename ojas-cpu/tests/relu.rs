//! `Backend::relu_forward` on the CPU: values, refusals, thread and
//! numerics independence, and its place in the bf16 autocast policy (a
//! pointwise op, like SiLU: rounded and tagged only when its input is
//! bf16-tagged; the norms and softmax stay f32).

use ojas_core::{
    round_f32_to_bf16, Autocast, AutocastMode, Backend, Budget, DType, Numerics, OjasError, Tensor,
};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_capacity, assert_nonfinite, assert_shape, bits, SplitMix64};

fn backend(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 32), threads)
        .unwrap()
        .with_numerics(numerics)
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

#[test]
fn relu_is_max_with_zero_bit_for_bit() {
    let mut rng = SplitMix64(11);
    let mut data = rng.vec(1000, 4.0);
    data.extend([
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MAX,
        f32::MIN,
        1e-30,
        -1e-30,
    ]);
    let n = data.len();
    let x = t(&data, &[n]);
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let y = v(&backend(1, numerics).relu_forward(&x).unwrap());
        assert_eq!(y.len(), n);
        for (&got, &inp) in y.iter().zip(&data) {
            // `x` where x is not negative (a negative zero stays as it is),
            // `+0` otherwise: every output is an input value or zero.
            let want = if inp < 0.0 { 0.0f32 } else { inp };
            assert_eq!(got.to_bits(), want.to_bits(), "relu({inp:e}) {numerics:?}");
        }
    }
}

#[test]
fn relu_keeps_the_shape_and_does_not_depend_on_the_thread_count() {
    let mut rng = SplitMix64(12);
    // Large enough to be cut into several chunks.
    let shape = [6, 64, 40, 40];
    let n: usize = shape.iter().product();
    let x = t(&rng.vec(n, 2.0), &shape);
    let base = backend(1, Numerics::Exact).relu_forward(&x).unwrap();
    assert_eq!(base.shape(), shape.as_slice());
    assert_eq!(base.dtype(), DType::F32);
    assert!(v(&base).iter().any(|&y| y > 0.0) && v(&base).contains(&0.0));
    for threads in [2, 7, 18] {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let got = backend(threads, numerics).relu_forward(&x).unwrap();
            assert_eq!(bits(&v(&got)), bits(&v(&base)), "x{threads} {numerics:?}");
        }
    }
    // A rank-0 tensor is one element.
    let y = backend(1, Numerics::Exact)
        .relu_forward(&t(&[-2.0], &[]))
        .unwrap();
    assert_eq!((y.shape().len(), v(&y)), (0, vec![0.0]));
}

#[test]
fn relu_refuses_what_every_pointwise_op_refuses() {
    let cpu = backend(1, Numerics::Exact);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_nonfinite(cpu.relu_forward(&t(&[1.0, bad], &[2])));
    }
    // An empty tensor, and one that is not F32.
    assert_shape(cpu.relu_forward(&t(&[], &[0, 3])));
    let ids = Tensor::from_u32(&[1, 2], &[2], &Budget::new(1 << 10)).unwrap();
    match cpu.relu_forward(&ids) {
        Err(OjasError::Dtype { .. }) => {}
        other => panic!("expected Dtype, got {other:?}"),
    }
    // A transposed view is not read in storage order.
    let base = t(
        &(0..12).map(|i| i as f32 - 6.0).collect::<Vec<_>>(),
        &[3, 4],
    );
    assert_shape(cpu.relu_forward(&base.view(&[4, 3], &[1, 4], 0).unwrap()));
    // An output over the budget.
    let tight = CpuBackend::with_threads(Budget::new(64), 1).unwrap();
    assert_capacity(tight.relu_forward(&t(&[1.0; 32], &[32])));
}

fn is_bf16_exact(values: &[f32]) -> bool {
    values
        .iter()
        .all(|&x| round_f32_to_bf16(x).to_bits() == x.to_bits())
}

#[test]
fn a_bf16_region_keeps_relu_of_a_tagged_input_tagged_and_leaves_an_untagged_one() {
    let wrapped = Autocast::new(backend(1, Numerics::Exact));
    let eye = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let _region = wrapped.autocast_region(AutocastMode::Bf16).unwrap();
    // A matmul-class output is rounded and tagged.
    let tagged = wrapped
        .linear_forward(&t(&[-1.5, 2.25], &[1, 2]), &eye)
        .unwrap();
    assert!(tagged.bf16_exact());
    let out = wrapped.relu_forward(&tagged).unwrap();
    assert_eq!(v(&out), [0.0, 2.25]);
    assert!(out.bf16_exact(), "relu of a bf16-tagged input is tagged");
    // An untagged input with a value no bf16 holds passes through unrounded,
    // as SiLU's does.
    let fine = f32::from_bits(0x3f80_0001);
    assert!(!is_bf16_exact(&[fine]));
    let plain = t(&[fine, -1.0], &[2]);
    let out = wrapped.relu_forward(&plain).unwrap();
    assert_eq!(v(&out), [fine, 0.0]);
    assert!(!out.bf16_exact());
}

#[test]
fn the_f32_class_ops_keep_their_outputs_f32_inside_a_bf16_region() {
    // torch autocast runs layer_norm and softmax in f32: their outputs are
    // not rounded even when the inputs are bf16-tagged.
    let wrapped = Autocast::new(backend(1, Numerics::Exact));
    let eye = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let _region = wrapped.autocast_region(AutocastMode::Bf16).unwrap();
    let a = wrapped
        .linear_forward(&t(&[1.0, 3.0], &[1, 2]), &eye)
        .unwrap();
    assert!(a.bf16_exact());
    let w = t(&[1.0, 1.0], &[2]);
    let z = t(&[0.0, 0.0], &[2]);
    let ln = wrapped.layer_norm_forward(&a, &w, &z, 1e-5).unwrap();
    assert!(!ln.bf16_exact());
    assert!(!is_bf16_exact(&v(&ln)), "{:?}", v(&ln));
    let sm = wrapped.softmax_forward(&a).unwrap();
    assert!(!sm.bf16_exact());
    assert!(!is_bf16_exact(&v(&sm)), "{:?}", v(&sm));
    let cl = wrapped
        .channel_layer_norm_forward(&a.reshape(&[1, 2, 1]).unwrap(), &w, &z, 1e-5)
        .unwrap();
    assert!(!cl.bf16_exact());
}
