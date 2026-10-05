//! Numeric checks and adversarial inputs for the CPU reference.
//! Empty, NaN, mismatched shape, and a full budget each return a typed error.

use ojas_core::{
    exp_exact, refuse_unsupported_metal_head_dim, AdamWConfig, Backend, BackendId, Budget, DType,
    MuonNs5Config, OjasError, Tensor, METAL_MAX_HEAD_DIM, RMS_NORM_EPS,
};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_capacity, assert_nonfinite, assert_range, assert_shape, f32t, u32t};

fn wide() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 20))
}

fn empty_f32(cpu: &CpuBackend) -> Tensor {
    Tensor::zeros(&[0, 2], DType::F32, cpu.budget()).unwrap()
}

fn empty_u32(cpu: &CpuBackend) -> Tensor {
    Tensor::zeros(&[0], DType::U32, cpu.budget()).unwrap()
}

fn nan_f32(cpu: &CpuBackend) -> Tensor {
    f32t(cpu, &[f32::NAN, 1.0], &[2])
}

fn noncontig(cpu: &CpuBackend) -> Tensor {
    let base = f32t(cpu, &[1.0, 2.0, 3.0, 4.0], &[4]);
    base.view(&[2], &[2], 0).unwrap()
}

#[test]
fn metal_policy_refuses_above_its_limit_and_cpu_does_not_truncate() {
    let metal = |d| refuse_unsupported_metal_head_dim(BackendId::Metal, d);
    let over = METAL_MAX_HEAD_DIM + 1;
    assert!(metal(METAL_MAX_HEAD_DIM).is_ok());
    match metal(over) {
        Err(OjasError::UnsupportedHeadDim { head_dim, limit })
            if head_dim == over && limit == METAL_MAX_HEAD_DIM => {}
        other => panic!("expected UnsupportedHeadDim, got {other:?}"),
    }
    assert_range(metal(0).map(|_| ()));

    // The CPU computes a head dimension Metal refuses, without truncating it.
    let cpu = wide();
    let dim = over as usize;
    let data = vec![0.01; dim];
    let q = f32t(&cpu, &data, &[1, 1, 1, dim]);
    let k = f32t(&cpu, &data, &[1, 1, 1, dim]);
    let values = vec![0.25f32; dim];
    let v = f32t(&cpu, &values, &[1, 1, 1, dim]);
    let y = cpu.causal_sdpa_forward(&q, &k, &v).unwrap();
    assert_eq!(y.shape(), &[1, 1, 1, dim]);
    let got = y.to_f32_vec().unwrap();
    assert_eq!(got.len(), dim);
    for a in &got {
        assert!((a - 0.25).abs() < 1e-5, "{a} != 0.25");
    }
}

#[test]
fn embedding_lookup_scatter_and_adversarial() {
    let cpu = wide();
    let table = f32t(&cpu, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let ids = u32t(&cpu, &[0, 2, 0], &[3]);
    let y = cpu.embedding_forward(&table, &ids).unwrap();
    assert_eq!(y.to_f32_vec().unwrap(), vec![1.0, 2.0, 5.0, 6.0, 1.0, 2.0]);
    let y2 = cpu.embedding_forward(&table, &ids).unwrap();
    assert_eq!(y.to_f32_vec().unwrap(), y2.to_f32_vec().unwrap());

    let gy = f32t(&cpu, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[3, 2]);
    let gt = cpu.embedding_backward(&table, &ids, &gy).unwrap();
    assert_eq!(gt.to_f32_vec().unwrap(), vec![2.0, 2.0, 0.0, 0.0, 1.0, 1.0]);

    assert_range(cpu.embedding_forward(&table, &u32t(&cpu, &[3], &[1])));
    assert_shape(cpu.embedding_forward(&empty_f32(&cpu), &u32t(&cpu, &[0], &[1])));
    assert_shape(cpu.embedding_forward(&table, &empty_u32(&cpu)));
    // A NaN in a well-formed table: a malformed one is a Shape error first
    // (docs/shape-contract.md D3, tests/shape_first_heavy.rs).
    let nan_table = f32t(&cpu, &[f32::NAN, 1.0], &[1, 2]);
    assert_nonfinite(cpu.embedding_forward(&nan_table, &u32t(&cpu, &[0], &[1])));
    assert_shape(cpu.embedding_forward(&noncontig(&cpu), &u32t(&cpu, &[0], &[1])));
    assert_shape(cpu.embedding_forward(&f32t(&cpu, &[1.0, 2.0, 3.0], &[3]), &ids));

    let budget = Budget::new(3 * 2 * 4 + 4);
    let tight = CpuBackend::new(budget.clone());
    let table = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &budget).unwrap();
    let ids = Tensor::from_u32(&[0], &[1], &budget).unwrap();
    assert_capacity(tight.embedding_forward(&table, &ids));
}

#[test]
fn linear_no_bias_layout_and_adversarial() {
    let cpu = wide();
    // y = x @ W^T, W is [out, in].
    let x = f32t(&cpu, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let w = f32t(&cpu, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
    let y = cpu.linear_forward(&x, &w).unwrap();
    assert_eq!(y.shape(), &[2, 2]);
    assert_eq!(y.to_f32_vec().unwrap(), vec![1.0, 2.0, 4.0, 5.0]);
    assert_eq!(
        y.to_f32_vec().unwrap(),
        cpu.linear_forward(&x, &w).unwrap().to_f32_vec().unwrap()
    );

    let gy = f32t(&cpu, &[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let (gx, gw) = cpu.linear_backward(&x, &w, &gy).unwrap();
    assert_eq!(gx.to_f32_vec().unwrap(), vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    assert_eq!(gw.to_f32_vec().unwrap(), vec![5.0, 7.0, 9.0, 5.0, 7.0, 9.0]);

    assert_shape(cpu.linear_forward(&empty_f32(&cpu), &w));
    assert_nonfinite(cpu.linear_forward(&nan_f32(&cpu), &f32t(&cpu, &[1.0, 0.0], &[1, 2])));
    assert_shape(cpu.linear_forward(&x, &f32t(&cpu, &[1.0, 2.0], &[2])));
    assert_shape(cpu.linear_forward(&noncontig(&cpu), &w));

    let budget = Budget::new(8 + 8);
    let tight = CpuBackend::new(budget.clone());
    let x = Tensor::from_f32(&[1.0, 2.0], &[2], &budget).unwrap();
    let w = Tensor::from_f32(&[3.0, 4.0], &[1, 2], &budget).unwrap();
    assert_capacity(tight.linear_forward(&x, &w));
}

#[test]
fn rms_norm_matches_torch_formula_and_adversarial() {
    let cpu = wide();
    let x = f32t(&cpu, &[0.5, -1.25, 0.75, 2.0], &[4]);
    let w = f32t(&cpu, &[1.0, 0.5, -1.0, 1.5], &[4]);
    let y = cpu.rms_norm_forward(&x, &w, RMS_NORM_EPS).unwrap();
    let expected = [
        0.39605889293691887_f64,
        -0.4950736161711486,
        -0.5940883394053783,
        2.376353357621513,
    ];
    for (got, exp) in y.to_f32_vec().unwrap().iter().zip(expected) {
        assert!((f64::from(*got) - exp).abs() < 1e-5, "{got} vs {exp}");
    }
    assert_eq!(
        y.to_f32_vec().unwrap(),
        cpu.rms_norm_forward(&x, &w, RMS_NORM_EPS)
            .unwrap()
            .to_f32_vec()
            .unwrap()
    );

    let gy = f32t(&cpu, &[1.0, 0.0, -1.0, 0.5], &[4]);
    let (gx, gw) = cpu.rms_norm_backward(&x, &w, &gy, RMS_NORM_EPS).unwrap();
    assert_eq!(gx.shape(), &[4]);
    assert_eq!(gw.shape(), &[4]);
    assert!(gx.to_f32_vec().unwrap().iter().all(|v| v.is_finite()));

    assert_shape(cpu.rms_norm_forward(&empty_f32(&cpu), &w, RMS_NORM_EPS));
    assert_nonfinite(cpu.rms_norm_forward(
        &nan_f32(&cpu),
        &f32t(&cpu, &[1.0, 1.0], &[2]),
        RMS_NORM_EPS,
    ));
    assert_shape(cpu.rms_norm_forward(&x, &f32t(&cpu, &[1.0, 1.0], &[2]), RMS_NORM_EPS));
    assert_shape(cpu.rms_norm_forward(
        &noncontig(&cpu),
        &f32t(&cpu, &[1.0, 1.0], &[2]),
        RMS_NORM_EPS,
    ));

    let budget = Budget::new(16 + 16);
    let tight = CpuBackend::new(budget.clone());
    let x = Tensor::from_f32(&[0.5, -1.25, 0.75, 2.0], &[4], &budget).unwrap();
    let w = Tensor::from_f32(&[1.0, 0.5, -1.0, 1.5], &[4], &budget).unwrap();
    assert_capacity(tight.rms_norm_forward(&x, &w, RMS_NORM_EPS));
}

#[test]
fn rope_half_split_matches_nanolab_sign() {
    let cpu = wide();
    let x = f32t(&cpu, &[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);
    let cos = f32t(&cpu, &[1.0, 0.0, 1.0, 0.0], &[1, 1, 1, 4]);
    let sin = f32t(&cpu, &[0.0, 1.0, 0.0, 1.0], &[1, 1, 1, 4]);
    let y = cpu.rope_half_split_forward(&x, &cos, &sin).unwrap();
    assert_eq!(y.to_f32_vec().unwrap(), vec![1.0, -4.0, 3.0, 2.0]);
    let gy = f32t(&cpu, &[1.0, 1.0, 1.0, 1.0], &[1, 1, 1, 4]);
    let gx = cpu.rope_half_split_backward(&gy, &cos, &sin).unwrap();
    assert_eq!(gx.to_f32_vec().unwrap(), vec![1.0, 1.0, 1.0, -1.0]);

    let x4 = f32t(
        &cpu,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 2, 1, 2],
    );
    let cos_td = f32t(&cpu, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let sin_td = f32t(&cpu, &[0.0, 1.0, 1.0, 0.0], &[2, 2]);
    let y4 = cpu.rope_half_split_forward(&x4, &cos_td, &sin_td).unwrap();
    assert_eq!(y4.shape(), &[2, 2, 1, 2]);
    assert!(y4.to_f32_vec().unwrap().iter().all(|v| v.is_finite()));

    assert_shape(cpu.rope_half_split_forward(&empty_f32(&cpu), &cos, &sin));
    assert_nonfinite(cpu.rope_half_split_forward(
        &nan_f32(&cpu),
        &f32t(&cpu, &[1.0, 0.0], &[2]),
        &f32t(&cpu, &[0.0, 1.0], &[2]),
    ));
    assert_shape(cpu.rope_half_split_forward(
        &f32t(&cpu, &[1.0, 2.0, 3.0], &[3]),
        &f32t(&cpu, &[1.0, 0.0, 0.0], &[3]),
        &f32t(&cpu, &[0.0, 1.0, 0.0], &[3]),
    ));
    assert_shape(cpu.rope_half_split_forward(&noncontig(&cpu), &cos, &sin));

    let budget = Budget::new(16);
    let tight = CpuBackend::new(budget.clone());
    let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], &budget).unwrap();
    let cos = Tensor::from_f32(&[1.0, 0.0, 1.0, 0.0], &[4], cpu.budget()).unwrap();
    let sin = Tensor::from_f32(&[0.0, 1.0, 0.0, 1.0], &[4], cpu.budget()).unwrap();
    assert_capacity(tight.rope_half_split_forward(&x, &cos, &sin));
}

#[test]
fn qk_norm_is_two_rms_norms() {
    let cpu = wide();
    let q = f32t(&cpu, &[0.5, -1.25, 0.75, 2.0], &[1, 1, 2, 2]);
    let k = f32t(&cpu, &[0.25, 0.5, -0.5, 1.0], &[1, 1, 2, 2]);
    let qw = f32t(&cpu, &[1.0, 0.5], &[2]);
    let kw = f32t(&cpu, &[1.0, -1.0], &[2]);
    let (qn, kn) = cpu
        .rms_qk_norm_forward(&q, &k, &qw, &kw, RMS_NORM_EPS)
        .unwrap();
    let q_ref = cpu.rms_norm_forward(&q, &qw, RMS_NORM_EPS).unwrap();
    let k_ref = cpu.rms_norm_forward(&k, &kw, RMS_NORM_EPS).unwrap();
    assert_eq!(qn.to_f32_vec().unwrap(), q_ref.to_f32_vec().unwrap());
    assert_eq!(kn.to_f32_vec().unwrap(), k_ref.to_f32_vec().unwrap());

    assert_shape(cpu.rms_qk_norm_forward(&empty_f32(&cpu), &k, &qw, &kw, RMS_NORM_EPS));
    assert_nonfinite(cpu.rms_qk_norm_forward(
        &nan_f32(&cpu),
        &f32t(&cpu, &[1.0, 1.0], &[2]),
        &f32t(&cpu, &[1.0, 1.0], &[2]),
        &kw,
        RMS_NORM_EPS,
    ));
    assert_shape(cpu.rms_qk_norm_forward(&q, &k, &f32t(&cpu, &[1.0], &[1]), &kw, RMS_NORM_EPS));
    assert_shape(cpu.rms_qk_norm_forward(
        &noncontig(&cpu),
        &k,
        &f32t(&cpu, &[1.0, 1.0], &[2]),
        &kw,
        RMS_NORM_EPS,
    ));

    let budget = Budget::new(16);
    let tight = CpuBackend::new(budget.clone());
    let q = Tensor::from_f32(&[0.5, -1.25, 0.75, 2.0], &[4], &budget).unwrap();
    let k = f32t(&cpu, &[0.25, 0.5, -0.5, 1.0], &[4]);
    let qw = f32t(&cpu, &[1.0, 0.5, 1.0, 1.0], &[4]);
    let kw = f32t(&cpu, &[1.0, 1.0, 1.0, 1.0], &[4]);
    assert_capacity(tight.rms_qk_norm_forward(&q, &k, &qw, &kw, RMS_NORM_EPS));
}

#[test]
fn causal_sdpa_scale_mask_and_adversarial() {
    let cpu = wide();
    let q = f32t(&cpu, &[1.0, 2.0], &[1, 1, 2, 1]);
    let k = f32t(&cpu, &[0.0, 1.0], &[1, 1, 2, 1]);
    let v = f32t(&cpu, &[3.0, 4.0], &[1, 1, 2, 1]);
    let y = cpu.causal_sdpa_forward(&q, &k, &v).unwrap();
    let got = y.to_f32_vec().unwrap();
    assert!((got[0] - 3.0).abs() < 1e-6);
    let e2 = (2.0_f64).exp();
    let p0 = 1.0 / (1.0 + e2);
    let p1 = e2 / (1.0 + e2);
    let expect1 = 3.0 * p0 + 4.0 * p1;
    assert!((f64::from(got[1]) - expect1).abs() < 1e-5, "{}", got[1]);
    assert_eq!(
        got,
        cpu.causal_sdpa_forward(&q, &k, &v)
            .unwrap()
            .to_f32_vec()
            .unwrap()
    );

    let gy = f32t(&cpu, &[1.0, -1.0], &[1, 1, 2, 1]);
    let (gq, gk, gv) = cpu.causal_sdpa_backward(&q, &k, &v, &gy).unwrap();
    assert!(gq.to_f32_vec().unwrap().iter().all(|v| v.is_finite()));
    assert!(gk.to_f32_vec().unwrap().iter().all(|v| v.is_finite()));
    assert!(gv.to_f32_vec().unwrap().iter().all(|v| v.is_finite()));

    assert_shape(cpu.causal_sdpa_forward(&empty_f32(&cpu), &k, &v));
    assert_nonfinite(
        cpu.causal_sdpa_forward(
            &nan_f32(&cpu)
                .view(&[1, 1, 2, 1], &[2, 2, 1, 1], 0)
                .unwrap_or(nan_f32(&cpu)),
            &k,
            &v,
        ),
    );
    let qnan = f32t(&cpu, &[f32::NAN, 1.0], &[1, 1, 2, 1]);
    assert_nonfinite(cpu.causal_sdpa_forward(&qnan, &k, &v));
    assert_shape(cpu.causal_sdpa_forward(&q, &f32t(&cpu, &[0.0, 1.0, 0.0], &[1, 1, 3, 1]), &v));
    assert_shape(cpu.causal_sdpa_forward(&noncontig(&cpu), &k, &v));

    // 1x1 and an odd time/head-dim, including a second head. The reduction
    // order is head dimension upward, then causal key index upward.
    let q = f32t(&cpu, &[1.0], &[1, 1, 1, 1]);
    let k = f32t(&cpu, &[2.0], &[1, 1, 1, 1]);
    let v = f32t(&cpu, &[3.0], &[1, 1, 1, 1]);
    assert_bits(
        &cpu.causal_sdpa_forward(&q, &k, &v)
            .unwrap()
            .to_f32_vec()
            .unwrap(),
        &causal_forward_reference(&[1.0], &[2.0], &[3.0], 1, 1, 1, 1),
    );
    let gy = f32t(&cpu, &[4.0], &[1, 1, 1, 1]);
    let (gq, gk, gv) = cpu.causal_sdpa_backward(&q, &k, &v, &gy).unwrap();
    let (eq, ek, ev) = causal_backward_reference(&[1.0], &[2.0], &[3.0], &[4.0], 1, 1, 1, 1);
    assert_eq!(gq.to_f32_vec().unwrap(), eq);
    assert_eq!(gk.to_f32_vec().unwrap(), ek);
    assert_eq!(gv.to_f32_vec().unwrap(), ev);

    let (batch, heads, time, dim) = (2usize, 2usize, 3usize, 5usize);
    let mut data = Vec::new();
    for i in 0..batch * heads * time * dim {
        data.push(((i % 7) as f32 - 3.0) * 0.1);
    }
    let q = f32t(&cpu, &data, &[batch, heads, time, dim]);
    let k = f32t(
        &cpu,
        &data.iter().rev().copied().collect::<Vec<_>>(),
        &[batch, heads, time, dim],
    );
    let v = f32t(
        &cpu,
        &data.iter().map(|x| x * 0.5).collect::<Vec<_>>(),
        &[batch, heads, time, dim],
    );
    assert_bits(
        &cpu.causal_sdpa_forward(&q, &k, &v)
            .unwrap()
            .to_f32_vec()
            .unwrap(),
        &causal_forward_reference(
            q.to_f32_vec().unwrap().as_slice(),
            k.to_f32_vec().unwrap().as_slice(),
            v.to_f32_vec().unwrap().as_slice(),
            batch,
            heads,
            time,
            dim,
        ),
    );
    let gy = f32t(
        &cpu,
        &data.iter().map(|x| -x).collect::<Vec<_>>(),
        &[batch, heads, time, dim],
    );
    let (gq, gk, gv) = cpu.causal_sdpa_backward(&q, &k, &v, &gy).unwrap();
    let (eq, ek, ev) = causal_backward_reference(
        q.to_f32_vec().unwrap().as_slice(),
        k.to_f32_vec().unwrap().as_slice(),
        v.to_f32_vec().unwrap().as_slice(),
        gy.to_f32_vec().unwrap().as_slice(),
        batch,
        heads,
        time,
        dim,
    );
    assert_eq!(gq.to_f32_vec().unwrap(), eq);
    assert_eq!(gk.to_f32_vec().unwrap(), ek);
    assert_eq!(gv.to_f32_vec().unwrap(), ev);

    // T=9 uses an 8-wide key tile plus a remainder, and T=4, D=16 is the tiny step.
    for (time, dim) in [(9usize, 8usize), (4, 16), (32, 64)] {
        let n = time * dim;
        let qv: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
        let kv: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.07).collect();
        let vv: Vec<f32> = (0..n).map(|i| ((i % 3) as f32 - 1.0) * 0.11).collect();
        let q = f32t(&cpu, &qv, &[1, 1, time, dim]);
        let k = f32t(&cpu, &kv, &[1, 1, time, dim]);
        let v = f32t(&cpu, &vv, &[1, 1, time, dim]);
        assert_bits(
            &cpu.causal_sdpa_forward(&q, &k, &v)
                .unwrap()
                .to_f32_vec()
                .unwrap(),
            &causal_forward_reference(&qv, &kv, &vv, 1, 1, time, dim),
        );
    }

    // A large future key must not change position 0, including the T=32 kernel.
    let (time, dim) = (32usize, 64usize);
    let n = time * dim;
    let qv: Vec<f32> = (0..n).map(|i| ((i % 9) as f32 - 4.0) * 0.03).collect();
    let kv: Vec<f32> = (0..n).map(|i| ((i % 4) as f32 - 1.5) * 0.04).collect();
    let vv: Vec<f32> = (0..n).map(|i| 0.2 + (i % dim) as f32 * 0.01).collect();
    let q = f32t(&cpu, &qv, &[1, 1, time, dim]);
    let k = f32t(&cpu, &kv, &[1, 1, time, dim]);
    let v = f32t(&cpu, &vv, &[1, 1, time, dim]);
    let y0 = cpu
        .causal_sdpa_forward(&q, &k, &v)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let mut kv_future = kv.clone();
    let future = 8 * dim;
    kv_future[future] = 40.0;
    let y1 = cpu
        .causal_sdpa_forward(&q, &f32t(&cpu, &kv_future, &[1, 1, time, dim]), &v)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    assert_eq!(
        y0[..dim]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        y1[..dim]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_ne!(&y0[future..future + dim], &y1[future..future + dim]);

    let budget = Budget::new(4 * 3);
    let tight = CpuBackend::new(budget.clone());
    let q = Tensor::from_f32(&[1.0], &[1, 1, 1, 1], &budget).unwrap();
    let k = Tensor::from_f32(&[1.0], &[1, 1, 1, 1], &budget).unwrap();
    let v = Tensor::from_f32(&[1.0], &[1, 1, 1, 1], &budget).unwrap();
    assert_capacity(tight.causal_sdpa_forward(&q, &k, &v));
}

fn assert_bits(got: &[f32], expect: &[f32]) {
    assert_eq!(got.len(), expect.len());
    for (i, (a, b)) in got.iter().zip(expect).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "index {i}: {a} vs {b}");
    }
}

/// Independent causal forward: scale `1/sqrt(dim)`, keys `0..=t`, sums in
/// increasing index order.
fn causal_forward_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    time: usize,
    dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (dim as f32).sqrt();
    let width = time * dim;
    let mut out = vec![0.0f32; batch * heads * width];
    for b in 0..batch {
        for h in 0..heads {
            let base = (b * heads + h) * width;
            for t in 0..time {
                let mut scores = vec![0.0f32; t + 1];
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += q[base + t * dim + d] * k[base + j * dim + d];
                    }
                    let score = dot * scale;
                    scores[j] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }
                let mut sum = 0.0f32;
                let mut probs = vec![0.0f32; t + 1];
                for j in 0..=t {
                    let e = exp_exact(scores[j] - max_score);
                    probs[j] = e;
                    sum += e;
                }
                for p in &mut probs {
                    *p /= sum;
                }
                for d in 0..dim {
                    let mut acc = 0.0f32;
                    for j in 0..=t {
                        acc += probs[j] * v[base + j * dim + d];
                    }
                    out[base + t * dim + d] = acc;
                }
            }
        }
    }
    out
}

/// Independent causal backward: scale `1/sqrt(dim)`, keys `0..=t`, sums in
/// increasing index order. Used to lock the optimized kernel's association.
#[allow(clippy::too_many_arguments)]
fn causal_backward_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    grad_y: &[f32],
    batch: usize,
    heads: usize,
    time: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let scale = 1.0 / (dim as f32).sqrt();
    let width = time * dim;
    let mut grad_q = vec![0.0f32; batch * heads * width];
    let mut grad_k = vec![0.0f32; batch * heads * width];
    let mut grad_v = vec![0.0f32; batch * heads * width];
    for b in 0..batch {
        for h in 0..heads {
            let base = (b * heads + h) * width;
            for t in 0..time {
                let mut scores = vec![0.0f32; t + 1];
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += q[base + t * dim + d] * k[base + j * dim + d];
                    }
                    let score = dot * scale;
                    scores[j] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }
                let mut sum = 0.0f32;
                let mut probs = vec![0.0f32; t + 1];
                for j in 0..=t {
                    let e = exp_exact(scores[j] - max_score);
                    probs[j] = e;
                    sum += e;
                }
                for p in &mut probs {
                    *p /= sum;
                }
                let mut dprobs = vec![0.0f32; t + 1];
                for j in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += grad_y[base + t * dim + d] * v[base + j * dim + d];
                    }
                    dprobs[j] = dot;
                }
                let mut expected = 0.0f32;
                for j in 0..=t {
                    expected += probs[j] * dprobs[j];
                }
                for j in 0..=t {
                    let ds = probs[j] * (dprobs[j] - expected);
                    let coef = scale * ds;
                    for d in 0..dim {
                        let qd = base + t * dim + d;
                        let kd = base + j * dim + d;
                        grad_q[qd] += coef * k[kd];
                        grad_k[kd] += coef * q[qd];
                        grad_v[kd] += probs[j] * grad_y[qd];
                    }
                }
            }
        }
    }
    (grad_q, grad_k, grad_v)
}

#[test]
fn per_head_gate_and_value_residual() {
    let cpu = wide();
    let x = f32t(&cpu, &[1.0, 2.0], &[1, 2]);
    let w = f32t(&cpu, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = f32t(&cpu, &[0.0, 0.0], &[2]);
    let attn = f32t(&cpu, &[1.0, 1.0], &[1, 2, 1]);
    let y = cpu
        .per_head_sigmoid_gate_forward(&x, &w, &b, &attn)
        .unwrap();
    let g0 = 1.0 / (1.0 + (-1.0_f32).exp());
    let g1 = 1.0 / (1.0 + (-2.0_f32).exp());
    let got = y.to_f32_vec().unwrap();
    assert!((got[0] - g0).abs() < 1e-6);
    assert!((got[1] - g1).abs() < 1e-6);

    let lam = f32t(&cpu, &[0.0], &[]);
    let v = f32t(&cpu, &[2.0, 2.0], &[2]);
    let v0 = f32t(&cpu, &[4.0, 6.0], &[2]);
    let blended = cpu.value_residual_blend_forward(&v, &v0, &lam).unwrap();
    let got = blended.to_f32_vec().unwrap();
    assert!((got[0] - 3.0).abs() < 1e-6);
    assert!((got[1] - 4.0).abs() < 1e-6);

    assert_shape(cpu.per_head_sigmoid_gate_forward(&empty_f32(&cpu), &w, &b, &attn));
    // The NaN and non-contiguous inputs are `[1, 2]`, shaped like `x`: a
    // malformed call is Shape before any NaN scan or layout check
    // (docs/shape-contract.md D3).
    let nan_x = f32t(&cpu, &[f32::NAN, 1.0], &[1, 2]);
    assert_nonfinite(cpu.per_head_sigmoid_gate_forward(&nan_x, &w, &b, &attn));
    assert_shape(cpu.per_head_sigmoid_gate_forward(
        &x,
        &f32t(&cpu, &[1.0, 2.0, 3.0], &[3]),
        &b,
        &attn,
    ));
    let strided_x = f32t(&cpu, &[1.0, 2.0, 3.0, 4.0], &[4])
        .view(&[1, 2], &[2, 2], 0)
        .unwrap();
    assert_shape(cpu.per_head_sigmoid_gate_forward(&strided_x, &w, &b, &attn));

    assert_shape(cpu.value_residual_blend_forward(&empty_f32(&cpu), &v0, &lam));
    assert_nonfinite(cpu.value_residual_blend_forward(
        &nan_f32(&cpu),
        &f32t(&cpu, &[1.0, 1.0], &[2]),
        &lam,
    ));
    assert_shape(cpu.value_residual_blend_forward(&v, &f32t(&cpu, &[1.0], &[1]), &lam));
    assert_shape(cpu.value_residual_blend_forward(&noncontig(&cpu), &v0, &lam));

    let budget = Budget::new(8 + 16 + 8 + 8);
    let tight = CpuBackend::new(budget.clone());
    let x = Tensor::from_f32(&[1.0, 2.0], &[1, 2], &budget).unwrap();
    let w = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &budget).unwrap();
    let b = Tensor::from_f32(&[0.0, 0.0], &[2], &budget).unwrap();
    let attn = Tensor::from_f32(&[1.0, 1.0], &[1, 2, 1], &budget).unwrap();
    assert_capacity(tight.per_head_sigmoid_gate_forward(&x, &w, &b, &attn));

    let budget = Budget::new(8 + 8 + 4);
    let tight = CpuBackend::new(budget.clone());
    let v = Tensor::from_f32(&[2.0, 2.0], &[2], &budget).unwrap();
    let v0 = Tensor::from_f32(&[4.0, 6.0], &[2], &budget).unwrap();
    let lam = Tensor::from_f32(&[0.0], &[1], &budget).unwrap();
    assert_capacity(tight.value_residual_blend_forward(&v, &v0, &lam));
}

#[test]
fn silu_mul_residual_and_adversarial() {
    let cpu = wide();
    let x = f32t(&cpu, &[0.0, 1.0, -1.0], &[3]);
    let y = cpu.silu_forward(&x).unwrap();
    let got = y.to_f32_vec().unwrap();
    assert!(got[0].abs() < 1e-6);
    let s = 1.0 / (1.0 + (-1.0_f32).exp());
    assert!((got[1] - s).abs() < 1e-6);
    assert_eq!(got, cpu.silu_forward(&x).unwrap().to_f32_vec().unwrap());

    let a = f32t(&cpu, &[2.0, -3.0], &[2]);
    let b = f32t(&cpu, &[4.0, 0.5], &[2]);
    assert_eq!(
        cpu.mul_forward(&a, &b).unwrap().to_f32_vec().unwrap(),
        vec![8.0, -1.5]
    );
    assert_eq!(
        cpu.residual_add_forward(&a, &b)
            .unwrap()
            .to_f32_vec()
            .unwrap(),
        vec![6.0, -2.5]
    );
    let gy = f32t(&cpu, &[1.0, 1.0], &[2]);
    let (ga, gb) = cpu.mul_backward(&a, &b, &gy).unwrap();
    assert_eq!(ga.to_f32_vec().unwrap(), b.to_f32_vec().unwrap());
    assert_eq!(gb.to_f32_vec().unwrap(), a.to_f32_vec().unwrap());
    let (gx, gy_b) = cpu.residual_add_backward(&a, &b, &gy).unwrap();
    assert_eq!(gx.to_f32_vec().unwrap(), gy.to_f32_vec().unwrap());
    assert_eq!(gy_b.to_f32_vec().unwrap(), gy.to_f32_vec().unwrap());

    assert_shape(cpu.silu_forward(&empty_f32(&cpu)));
    assert_nonfinite(cpu.silu_forward(&nan_f32(&cpu)));
    assert_shape(cpu.silu_forward(&noncontig(&cpu)));
    assert_shape(cpu.mul_forward(&a, &f32t(&cpu, &[1.0], &[1])));
    assert_shape(cpu.residual_add_forward(&a, &empty_f32(&cpu)));
    assert_nonfinite(cpu.residual_add_forward(&nan_f32(&cpu), &f32t(&cpu, &[1.0, 1.0], &[2])));
    assert_shape(cpu.residual_add_forward(&noncontig(&cpu), &a));
    assert_nonfinite(cpu.mul_forward(&nan_f32(&cpu), &f32t(&cpu, &[1.0, 1.0], &[2])));
    assert_shape(cpu.mul_forward(&noncontig(&cpu), &a));

    let budget = Budget::new(8);
    let tight = CpuBackend::new(budget.clone());
    let x = Tensor::from_f32(&[1.0, 2.0], &[2], &budget).unwrap();
    assert_capacity(tight.silu_forward(&x));
    let budget = Budget::new(16);
    let tight = CpuBackend::new(budget.clone());
    let a = Tensor::from_f32(&[1.0, 2.0], &[2], &budget).unwrap();
    let b = Tensor::from_f32(&[3.0, 4.0], &[2], &budget).unwrap();
    assert_capacity(tight.mul_forward(&a, &b));
    assert_capacity(tight.residual_add_forward(&a, &b));
}

#[test]
fn cross_entropy_ignore_is_option_not_a_dummy() {
    let cpu = wide();
    let logits = f32t(&cpu, &[0.0, 0.0, 2.0, 0.0], &[2, 2]);
    let targets = u32t(&cpu, &[0, 1], &[2]);
    let loss = cpu
        .cross_entropy_mean_forward(&logits, &targets, None)
        .unwrap();
    let row0 = (2.0_f64).ln();
    let row1 = -(0.0_f64 - (2.0_f64).exp().ln_1p());
    // log_softmax class 1 of [2, 0]: 0 - log(e^2+e^0) wait logits 2 and 0, target 1 is the 0 logit.
    let lse = 2.0_f64.max(0.0) + ((2.0_f64 - 2.0).exp() + (0.0_f64 - 2.0).exp()).ln();
    let row1_loss = lse - 0.0;
    let expect = (row0 + row1_loss) / 2.0;
    let _ = row1;
    assert!((f64::from(loss.to_f32_vec().unwrap()[0]) - expect).abs() < 1e-5);

    let ignored = u32t(&cpu, &[0, 9], &[2]);
    let loss_i = cpu
        .cross_entropy_mean_forward(&logits, &ignored, Some(9))
        .unwrap();
    assert!((loss_i.to_f32_vec().unwrap()[0] as f64 - row0).abs() < 1e-5);
    let grad = cpu
        .cross_entropy_mean_backward(&logits, &ignored, Some(9))
        .unwrap()
        .to_f32_vec()
        .unwrap();
    assert_eq!(grad[2], 0.0);
    assert_eq!(grad[3], 0.0);

    let none_vs_some = u32t(&cpu, &[0], &[1]);
    let logits1 = f32t(&cpu, &[0.0, 0.0], &[1, 2]);
    let with_none = cpu
        .cross_entropy_mean_forward(&logits1, &none_vs_some, None)
        .unwrap();
    let with_other = cpu
        .cross_entropy_mean_forward(&logits1, &none_vs_some, Some(7))
        .unwrap();
    assert_eq!(
        with_none.to_f32_vec().unwrap(),
        with_other.to_f32_vec().unwrap()
    );
    // Every row matches a valid class used as ignore_index. Torch's mean is
    // NaN; this reference refuses that empty reduction instead of returning 0.
    assert_nonfinite(cpu.cross_entropy_mean_forward(&logits1, &none_vs_some, Some(0)));
    assert_nonfinite(cpu.cross_entropy_mean_backward(&logits1, &none_vs_some, Some(0)));

    assert_range(cpu.cross_entropy_mean_forward(&logits1, &u32t(&cpu, &[2], &[1]), None));
    assert_shape(cpu.cross_entropy_mean_forward(&empty_f32(&cpu), &u32t(&cpu, &[0], &[1]), None));
    // NaN logits shaped to match their targets: malformed ones are a Shape
    // error first (docs/shape-contract.md D3, tests/shape_first_heavy.rs).
    let nan_logits = f32t(&cpu, &[f32::NAN, 1.0], &[1, 2]);
    assert_nonfinite(cpu.cross_entropy_mean_forward(&nan_logits, &u32t(&cpu, &[0], &[1]), None));
    assert_shape(cpu.cross_entropy_mean_forward(&logits, &u32t(&cpu, &[0], &[1]), None));
    assert_shape(cpu.cross_entropy_mean_forward(&noncontig(&cpu), &u32t(&cpu, &[0], &[1]), None));

    let budget = Budget::new(8 + 4);
    let tight = CpuBackend::new(budget.clone());
    let logits = Tensor::from_f32(&[0.0, 0.0], &[1, 2], &budget).unwrap();
    let targets = Tensor::from_u32(&[0], &[1], &budget).unwrap();
    assert_capacity(tight.cross_entropy_mean_forward(&logits, &targets, None));
}

#[test]
fn clip_and_adam_write_rules() {
    let cpu = wide();
    let mut g = f32t(&cpu, &[3.0, 4.0], &[2]);
    let norm = cpu
        .clip_grad_norm(std::slice::from_mut(&mut g), 1.0)
        .unwrap();
    assert!((norm - 5.0).abs() < 1e-6);
    let scale = 1.0f32 / (5.0 + 1e-6);
    let got = g.to_f32_vec().unwrap();
    assert!((got[0] - 3.0 * scale).abs() < 1e-6);
    assert!((got[1] - 4.0 * scale).abs() < 1e-6);

    let mut tiny = f32t(&cpu, &[0.25], &[1]);
    let bits = tiny.to_f32_vec().unwrap();
    let norm = cpu
        .clip_grad_norm(std::slice::from_mut(&mut tiny), 1.0)
        .unwrap();
    assert!(norm < 1.0);
    assert_eq!(tiny.to_f32_vec().unwrap(), bits);

    let mut bad = f32t(&cpu, &[1.0, f32::NAN], &[2]);
    let snapshot = bad.to_f32_vec().unwrap();
    assert_nonfinite(cpu.clip_grad_norm(std::slice::from_mut(&mut bad), 1.0));
    assert_eq!(
        bad.to_f32_vec().unwrap()[0].to_bits(),
        snapshot[0].to_bits()
    );

    assert_shape(cpu.clip_grad_norm(&mut [], 1.0));
    let mut empty = empty_f32(&cpu);
    assert_shape(cpu.clip_grad_norm(std::slice::from_mut(&mut empty), 1.0));
    let mut weird = noncontig(&cpu);
    assert_shape(cpu.clip_grad_norm(std::slice::from_mut(&mut weird), 1.0));

    // The scale is in place and charges nothing: a budget with no room left
    // still clips, with the roomy backend's bits.
    let budget = Budget::new(8);
    let tight = CpuBackend::new(budget.clone());
    let mut g = Tensor::from_f32(&[3.0, 4.0], &[2], &budget).unwrap();
    let mut roomy = f32t(&cpu, &[3.0, 4.0], &[2]);
    let want = cpu
        .clip_grad_norm(std::slice::from_mut(&mut roomy), 1.0)
        .unwrap();
    let got = tight
        .clip_grad_norm(std::slice::from_mut(&mut g), 1.0)
        .unwrap();
    assert_eq!(got.to_bits(), want.to_bits());
    assert_eq!(budget.live_bytes().unwrap(), 8);
    let bits = |t: &Tensor| -> Vec<u32> {
        t.to_f32_vec()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect()
    };
    assert_eq!(bits(&g), bits(&roomy));
    assert_ne!(g.to_f32_vec().unwrap(), vec![3.0, 4.0]);

    let mut p = f32t(&cpu, &[1.25], &[1]);
    let pbits = p.to_f32_vec().unwrap()[0].to_bits();
    let grad = f32t(&cpu, &[0.0], &[1]);
    let mut m1 = f32t(&cpu, &[0.0], &[1]);
    let mut m2 = f32t(&cpu, &[0.0], &[1]);
    let cfg = AdamWConfig::nanolab(0.1, 0.0);
    cpu.adamw_step(&mut p, &grad, &mut m1, &mut m2, 0, cfg)
        .unwrap();
    assert_eq!(p.to_f32_vec().unwrap()[0].to_bits(), pbits);

    let mut p = f32t(&cpu, &[1.0], &[1]);
    let grad = f32t(&cpu, &[1.0], &[1]);
    let mut m1 = f32t(&cpu, &[0.0], &[1]);
    let mut m2 = f32t(&cpu, &[0.0], &[1]);
    cpu.adamw_step(&mut p, &grad, &mut m1, &mut m2, 0, cfg)
        .unwrap();
    let v = 0.05_f64;
    let denom = v.sqrt() / 0.05_f64.sqrt() + 1e-8;
    let expect = 1.0 - 0.1 / denom;
    assert!((f64::from(p.to_f32_vec().unwrap()[0]) - expect).abs() < 1e-6);

    let mut p = f32t(&cpu, &[1.0], &[1]);
    let grad = f32t(&cpu, &[0.1], &[1]);
    let mut m1 = f32t(&cpu, &[0.0], &[1]);
    let mut m2 = f32t(&cpu, &[0.0], &[1]);
    cpu.adamw_step(&mut p, &grad, &mut m1, &mut m2, 1_000_000, cfg)
        .unwrap();
    assert!(p.to_f32_vec().unwrap()[0].is_finite());

    let mut p = f32t(&cpu, &[1.0], &[1]);
    let bits = p.to_f32_vec().unwrap()[0].to_bits();
    let mut m1 = f32t(&cpu, &[0.0], &[1]);
    let mut m2 = f32t(&cpu, &[0.0], &[1]);
    assert!(matches!(
        cpu.adamw_step(&mut p, &grad, &mut m1, &mut m2, u64::MAX, cfg),
        Err(OjasError::OutOfRange { .. })
    ));
    assert_eq!(p.to_f32_vec().unwrap()[0].to_bits(), bits);
    assert_eq!(m1.to_f32_vec().unwrap(), vec![0.0]);

    let mut p = f32t(&cpu, &[1.0], &[1]);
    let bits = p.to_f32_vec().unwrap()[0].to_bits();
    let mut m1 = f32t(&cpu, &[0.0], &[1]);
    let m1b = m1.to_f32_vec().unwrap();
    let mut m2 = f32t(&cpu, &[0.0], &[1]);
    let bad = f32t(&cpu, &[f32::NAN], &[1]);
    assert_nonfinite(cpu.adamw_step(&mut p, &bad, &mut m1, &mut m2, 0, cfg));
    assert_eq!(p.to_f32_vec().unwrap()[0].to_bits(), bits);
    assert_eq!(m1.to_f32_vec().unwrap(), m1b);

    assert_shape(cpu.adamw_step(&mut p, &empty_f32(&cpu), &mut m1, &mut m2, 0, cfg));
    assert_shape(cpu.adamw_step(
        &mut p,
        &f32t(&cpu, &[1.0, 2.0], &[2]),
        &mut m1,
        &mut m2,
        0,
        cfg,
    ));
    let mut weird = noncontig(&cpu);
    assert_shape(cpu.adamw_step(&mut weird, &grad, &mut m1, &mut m2, 0, cfg));

    // The step is in place and allocates nothing: a budget with no room
    // left still runs it, charges nothing, and gives the roomy result.
    let budget = Budget::new(16);
    let tight = CpuBackend::new(budget.clone());
    let mut p = Tensor::from_f32(&[1.0], &[1], &budget).unwrap();
    let g = Tensor::from_f32(&[0.1], &[1], &budget).unwrap();
    let mut m1 = Tensor::from_f32(&[0.0], &[1], &budget).unwrap();
    let mut m2 = Tensor::from_f32(&[0.0], &[1], &budget).unwrap();
    let roomy = CpuBackend::new(Budget::new(1 << 20));
    let (mut rp, mut rm1, mut rm2) = (
        f32t(&roomy, &[1.0], &[1]),
        f32t(&roomy, &[0.0], &[1]),
        f32t(&roomy, &[0.0], &[1]),
    );
    let rg = f32t(&roomy, &[0.1], &[1]);
    roomy
        .adamw_step(&mut rp, &rg, &mut rm1, &mut rm2, 0, cfg)
        .unwrap();
    tight
        .adamw_step(&mut p, &g, &mut m1, &mut m2, 0, cfg)
        .unwrap();
    assert_eq!(budget.live_bytes().unwrap(), 16);
    for (got, want) in [(&p, &rp), (&m1, &rm1), (&m2, &rm2)] {
        let bits = |t: &Tensor| t.to_f32_vec().unwrap()[0].to_bits();
        assert_eq!(bits(got), bits(want));
    }
    assert_ne!(p.to_f32_vec().unwrap(), vec![1.0]);
}

#[test]
fn muon_ns5_f32_reference_and_adversarial() {
    let cpu = wide();
    let mut p = f32t(&cpu, &[0.5], &[1, 1]);
    let bits = p.to_f32_vec().unwrap()[0].to_bits();
    let g = f32t(&cpu, &[0.0], &[1, 1]);
    let mut m = f32t(&cpu, &[0.0], &[1, 1]);
    let cfg = MuonNs5Config {
        lr: 0.025,
        momentum: 0.99,
        weight_decay: 0.0,
        nesterov: true,
    };
    cpu.muon_ns5_step(&mut p, &g, &mut m, cfg).unwrap();
    assert_eq!(p.to_f32_vec().unwrap()[0].to_bits(), bits);

    let mut p = f32t(&cpu, &[0.2, -0.1, 0.0, 0.4], &[2, 2]);
    let g = f32t(&cpu, &[0.3, -0.2, 0.1, 0.05], &[2, 2]);
    let mut m = f32t(&cpu, &[0.0, 0.0, 0.0, 0.0], &[2, 2]);
    let cfg = MuonNs5Config::nanolab_default();
    cpu.muon_ns5_step(&mut p, &g, &mut m, cfg).unwrap();
    assert!(p.to_f32_vec().unwrap().iter().all(|v| v.is_finite()));
    let again_p = p.to_f32_vec().unwrap();
    let again_m = m.to_f32_vec().unwrap();
    let mut p2 = f32t(&cpu, &[0.2, -0.1, 0.0, 0.4], &[2, 2]);
    let mut m2 = f32t(&cpu, &[0.0; 4], &[2, 2]);
    cpu.muon_ns5_step(&mut p2, &g, &mut m2, cfg).unwrap();
    assert_eq!(p2.to_f32_vec().unwrap(), again_p);
    assert_eq!(m2.to_f32_vec().unwrap(), again_m);

    let mut p = f32t(&cpu, &[1.0], &[1, 1]);
    let bits = p.to_f32_vec().unwrap();
    let mut m = f32t(&cpu, &[0.0], &[1, 1]);
    let mbits = m.to_f32_vec().unwrap();
    let bad = f32t(&cpu, &[f32::NAN], &[1, 1]);
    assert_nonfinite(cpu.muon_ns5_step(&mut p, &bad, &mut m, cfg));
    assert_eq!(p.to_f32_vec().unwrap(), bits);
    assert_eq!(m.to_f32_vec().unwrap(), mbits);

    assert_shape(cpu.muon_ns5_step(&mut p, &empty_f32(&cpu), &mut m, cfg));
    assert_shape(cpu.muon_ns5_step(&mut p, &f32t(&cpu, &[1.0, 2.0], &[2, 1]), &mut m, cfg));
    let mut weird = noncontig(&cpu);
    assert_shape(cpu.muon_ns5_step(&mut weird, &g, &mut m, cfg));

    let budget = Budget::new(4 * 3);
    let tight = CpuBackend::new(budget.clone());
    let mut p = Tensor::from_f32(&[0.2], &[1, 1], &budget).unwrap();
    let g = Tensor::from_f32(&[0.1], &[1, 1], &budget).unwrap();
    let mut mom = Tensor::from_f32(&[0.0], &[1, 1], &budget).unwrap();
    let snap = p.to_f32_vec().unwrap();
    assert_capacity(tight.muon_ns5_step(&mut p, &g, &mut mom, cfg));
    assert_eq!(p.to_f32_vec().unwrap(), snap);
}
