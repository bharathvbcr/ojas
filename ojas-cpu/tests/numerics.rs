//! `Numerics::Fast` against `Numerics::Exact`, thread-count invariance of
//! both, and the backend's numerics and placement reporting. On macOS the
//! Fast GEMMs at or above `ojas_cpu::FAST_WHOLE_CALL_MACS` are single Accelerate calls made
//! with the same arguments whatever the pool size, so the Fast thread-count
//! test there checks that dispatch ignores the pool and that Accelerate
//! repeats on this machine, not that Accelerate's order is thread-invariant.

use std::any::Any;
use std::sync::Arc;

use ojas_core::{
    AdamWConfig, Backend, BackendId, Budget, DType, DeviceBuffer, MuonNs5Config, Numerics,
    OjasError, Tensor,
};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_nonfinite, assert_shape, bits, f32t, u32t, SplitMix64};

const THREADS: [usize; 6] = [1, 2, 3, 7, 16, 18];

fn backend(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 31), threads)
        .unwrap()
        .with_numerics(numerics)
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

/// Linear shapes `(rows, in, out)`: 1x1, primes, both sides of the Exact
/// (MC 72, KC 256, NC 256) and Fast (MC 144, KC 512, NC 512) block edges,
/// and one product well above the parallel threshold.
const LINEAR: [(usize, usize, usize); 9] = [
    (1, 1, 1),
    (73, 11, 13),
    (127, 67, 131),
    (72, 256, 256),
    (73, 257, 257),
    (144, 512, 512),
    (145, 513, 513),
    (6, 1000, 16),
    (300, 700, 520),
];

/// SDPA shapes. `T <= 256` stays on the per-row kernel under Fast; the rest
/// take the blocked kernel, including a `T` just past the cutoff and one
/// that is not a multiple of the 64-key block.
const SDPA: [[usize; 4]; 4] = [
    [1, 2, 64, 32],
    [1, 1, 257, 16],
    [2, 2, 300, 64],
    [1, 3, 520, 40],
];

fn outputs(cpu: &CpuBackend) -> Vec<(String, Vec<f32>)> {
    let mut out = Vec::new();
    for (i, &(rows, kin, nout)) in LINEAR.iter().enumerate() {
        let mut rng = SplitMix64(1000 + i as u64);
        let x = f32t(cpu, &rng.vec(rows * kin, 0.5), &[rows, kin]);
        let w = f32t(cpu, &rng.vec(nout * kin, 0.5), &[nout, kin]);
        let gy = f32t(cpu, &rng.vec(rows * nout, 0.5), &[rows, nout]);
        let y = cpu.linear_forward(&x, &w).unwrap();
        let (gx, gw) = cpu.linear_backward(&x, &w, &gy).unwrap();
        let tag = format!("{rows}x{kin}x{nout}");
        out.push((format!("linear_fwd {tag}"), flat(&y)));
        out.push((format!("linear_gx {tag}"), flat(&gx)));
        out.push((format!("linear_gw {tag}"), flat(&gw)));
    }
    for (i, &(rows, cols)) in [(64usize, 48usize), (300, 200), (256, 384)]
        .iter()
        .enumerate()
    {
        let mut rng = SplitMix64(1100 + i as u64);
        let mut p = f32t(cpu, &rng.vec(rows * cols, 0.1), &[rows, cols]);
        let g = f32t(cpu, &rng.vec(rows * cols, 0.1), &[rows, cols]);
        let mut m = f32t(cpu, &rng.vec(rows * cols, 0.01), &[rows, cols]);
        let config = MuonNs5Config {
            weight_decay: 0.01,
            ..MuonNs5Config::nanolab_default()
        };
        cpu.muon_ns5_step(&mut p, &g, &mut m, config).unwrap();
        out.push((format!("muon_p {rows}x{cols}"), flat(&p)));
        out.push((format!("muon_m {rows}x{cols}"), flat(&m)));
    }
    for (i, shape) in SDPA.iter().enumerate() {
        let n: usize = shape.iter().product();
        let mut rng = SplitMix64(1200 + i as u64);
        let q = f32t(cpu, &rng.vec(n, 1.0), shape);
        let k = f32t(cpu, &rng.vec(n, 1.0), shape);
        let v = f32t(cpu, &rng.vec(n, 1.0), shape);
        let gy = f32t(cpu, &rng.vec(n, 1.0), shape);
        let y = cpu
            .causal_sdpa_forward(&q, &k, &v, None)
            .map(|(y, _)| y)
            .unwrap();
        let (gq, gk, gv) = cpu
            .causal_sdpa_backward_recompute(&q, &k, &v, &gy, None)
            .unwrap();
        let tag = format!("{shape:?}");
        out.push((format!("sdpa_fwd {tag}"), flat(&y)));
        out.push((format!("sdpa_gq {tag}"), flat(&gq)));
        out.push((format!("sdpa_gk {tag}"), flat(&gk)));
        out.push((format!("sdpa_gv {tag}"), flat(&gv)));
    }
    {
        let (rows, dim) = (600usize, 256usize);
        let mut rng = SplitMix64(1300);
        let x = f32t(cpu, &rng.vec(rows * dim, 1.0), &[rows, dim]);
        let w: Vec<f32> = rng.vec(dim, 0.1).iter().map(|v| 1.0 + v).collect();
        let w = f32t(cpu, &w, &[dim]);
        let gy = f32t(cpu, &rng.vec(rows * dim, 1.0), &[rows, dim]);
        let y = cpu.rms_norm_forward(&x, &w, 1e-6).unwrap();
        let (gx, gw) = cpu.rms_norm_backward(&x, &w, &gy, 1e-6).unwrap();
        out.push(("rms_fwd".into(), flat(&y)));
        out.push(("rms_gx".into(), flat(&gx)));
        out.push(("rms_gw".into(), flat(&gw)));
    }
    {
        let (b, t, h, d) = (2usize, 256usize, 4usize, 64usize);
        let mut rng = SplitMix64(1400);
        let x = f32t(cpu, &rng.vec(b * t * h * d, 1.0), &[b, t, h, d]);
        let cos = f32t(cpu, &rng.vec(t * d, 1.0), &[t, d]);
        let sin = f32t(cpu, &rng.vec(t * d, 1.0), &[t, d]);
        let y = cpu.rope_half_split_forward(&x, &cos, &sin).unwrap();
        let gx = cpu.rope_half_split_backward(&x, &cos, &sin).unwrap();
        out.push(("rope_fwd".into(), flat(&y)));
        out.push(("rope_bwd".into(), flat(&gx)));
    }
    {
        let (rows, vocab) = (600usize, 500usize);
        let mut rng = SplitMix64(1500);
        let logits = f32t(cpu, &rng.vec(rows * vocab, 4.0), &[rows, vocab]);
        let targets: Vec<u32> = (0..rows)
            .map(|i| {
                if i % 11 == 3 {
                    7
                } else {
                    rng.below(vocab) as u32
                }
            })
            .collect();
        let targets = u32t(cpu, &targets, &[rows]);
        let loss = cpu
            .cross_entropy_mean_forward(&logits, &targets, Some(7))
            .unwrap();
        let grad = cpu
            .cross_entropy_mean_backward(&logits, &targets, Some(7))
            .unwrap();
        out.push(("ce_loss".into(), flat(&loss)));
        out.push(("ce_grad".into(), flat(&grad)));
    }
    {
        let n = 200_003usize;
        let mut rng = SplitMix64(1600);
        let mut p = f32t(cpu, &rng.vec(n, 1.0), &[n]);
        let g = f32t(cpu, &rng.vec(n, 0.1), &[n]);
        let mut m1 = f32t(cpu, &rng.vec(n, 0.01), &[n]);
        let m2v: Vec<f32> = rng.vec(n, 0.01).iter().map(|v| v.abs()).collect();
        let mut m2 = f32t(cpu, &m2v, &[n]);
        cpu.adamw_step(
            &mut p,
            &g,
            &mut m1,
            &mut m2,
            9,
            AdamWConfig::nanolab(3e-3, 0.1),
        )
        .unwrap();
        out.push(("adamw_p".into(), flat(&p)));
        out.push(("adamw_m1".into(), flat(&m1)));
        out.push(("adamw_m2".into(), flat(&m2)));
    }
    out
}

fn assert_same_bits(numerics: Numerics) {
    let reference = outputs(&backend(1, numerics));
    for threads in THREADS {
        let got = outputs(&backend(threads, numerics));
        assert_eq!(got.len(), reference.len());
        for ((name, want), (_, have)) in reference.iter().zip(&got) {
            assert!(
                bits(want) == bits(have),
                "{numerics:?} {name}: bits at {threads} threads differ from 1 thread"
            );
        }
    }
}

#[test]
fn fast_bits_do_not_depend_on_thread_count() {
    assert_same_bits(Numerics::Fast);
}

#[test]
fn exact_bits_do_not_depend_on_thread_count_on_parallel_shapes() {
    assert_same_bits(Numerics::Exact);
}

/// `max |fast - exact| <= tol * max(|exact|)` per output tensor.
#[test]
fn fast_matches_exact_within_stated_tolerance() {
    let exact = outputs(&backend(7, Numerics::Exact));
    let fast = outputs(&backend(7, Numerics::Fast));
    let mut differs = 0;
    for ((name, e), (_, f)) in exact.iter().zip(&fast) {
        assert_eq!(e.len(), f.len(), "{name}");
        let scale = e
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(f32::MIN_POSITIVE);
        let err = e
            .iter()
            .zip(f)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        // Ops without a reduction change are bit-equal; GEMM and attention
        // reassociate and fuse, and Newton-Schulz repeats that five times.
        // Cross-entropy's Fast exponential (crate exp.rs, 1 ulp measured) and
        // lane sums move the loss and gradient by rounding. Measured on
        // aarch64: at most 3.7e-7 for GEMM and attention, 1.7e-7 for the
        // Muon parameter, 6.3e-8 for the cross-entropy gradient. Fast AdamW
        // does its element arithmetic in f32 where Exact uses f64: 1.2e-7 for
        // the parameter, 9.8e-8 and 9.3e-8 for the moments.
        let tol = if name.starts_with("muon_p")
            || name.starts_with("ce_")
            || name.starts_with("adamw_")
        {
            1e-6
        } else if name.starts_with("linear") || name.starts_with("sdpa") {
            2e-6
        } else {
            0.0
        };
        assert!(
            err <= tol * scale,
            "{name}: max |fast - exact| {err:e} > {tol:e} * {scale:e}"
        );
        if err > 0.0 {
            differs += 1;
        }
    }
    // Fast must actually take a different path somewhere.
    assert!(differs > 0, "Fast reproduced every Exact bit");
}

/// Shapes `(rows, in, out)` for the Fast GEMM dispatch tests. 256³ is above
/// the whole-call cutoff (`ojas_cpu::FAST_WHOLE_CALL_MACS`) everywhere; the
/// first two are below it everywhere, crossing MR, NR and KC; the last two
/// are below it only off macOS, where the cutoff is 2²¹.
const DISPATCH: [(usize, usize, usize); 5] = [
    (7, 9, 17),
    (1, 300, 17),
    (256, 256, 256),
    (73, 11, 13),
    (128, 64, 96),
];

/// `linear_forward`, `grad_x`, `grad_w` for one fixture.
fn linear_products(cpu: &CpuBackend, rows: usize, kin: usize, nout: usize) -> [Vec<f32>; 3] {
    let mut rng = SplitMix64(0xd15 + (rows * 31 + kin * 7 + nout) as u64);
    let x = rng.vec(rows * kin, 0.5);
    let w = rng.vec(nout * kin, 0.5);
    let gy = rng.vec(rows * nout, 0.5);
    let (xt, wt) = (f32t(cpu, &x, &[rows, kin]), f32t(cpu, &w, &[nout, kin]));
    let y = cpu.linear_forward(&xt, &wt).unwrap();
    let (gx, gw) = cpu
        .linear_backward(&xt, &wt, &f32t(cpu, &gy, &[rows, nout]))
        .unwrap();
    [flat(&y), flat(&gx), flat(&gw)]
}

/// The same three products as one `mul_add` chain per output, reduction
/// index ascending from `+0.0`: what `tile_fast` and `sgemm_tile` compute.
#[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
fn fma_chain(rows: usize, kin: usize, nout: usize) -> [Vec<f32>; 3] {
    let mut rng = SplitMix64(0xd15 + (rows * 31 + kin * 7 + nout) as u64);
    let x = rng.vec(rows * kin, 0.5);
    let w = rng.vec(nout * kin, 0.5);
    let gy = rng.vec(rows * nout, 0.5);
    let chain = |m: usize, n: usize, k: usize, at: &dyn Fn(usize, usize, usize) -> (f32, f32)| {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let (a, b) = at(i, j, p);
                    acc = a.mul_add(b, acc);
                }
                c[i * n + j] = acc;
            }
        }
        c
    };
    let y = chain(rows, nout, kin, &|i, j, p| (x[i * kin + p], w[j * kin + p]));
    let gx = chain(rows, kin, nout, &|i, q, j| {
        (gy[i * nout + j], w[j * kin + q])
    });
    let gw = chain(nout, kin, rows, &|j, q, i| {
        (gy[i * nout + j], x[i * kin + q])
    });
    [y, gx, gw]
}

const PRODUCTS: [&str; 3] = ["linear_fwd", "linear_gx", "linear_gw"];

/// Fast is finite, within 1e-4 of Exact relative to the largest Exact
/// magnitude, and bit-identical across two calls on one backend. On macOS
/// the 256³ products are single Accelerate calls; no claim is made here
/// about their bits across thread counts.
#[test]
fn fast_gemm_is_finite_close_to_exact_and_repeatable() {
    let exact = backend(7, Numerics::Exact);
    let fast = backend(18, Numerics::Fast);
    for &(rows, kin, nout) in &DISPATCH {
        let want = linear_products(&exact, rows, kin, nout);
        let first = linear_products(&fast, rows, kin, nout);
        let second = linear_products(&fast, rows, kin, nout);
        for (p, name) in PRODUCTS.iter().enumerate() {
            let tag = format!("{name} {rows}x{kin}x{nout}");
            assert!(first[p].iter().all(|v| v.is_finite()), "{tag}: non-finite");
            assert!(
                bits(&first[p]) == bits(&second[p]),
                "{tag}: two runs differ"
            );
            let scale = want[p].iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let err = want[p]
                .iter()
                .zip(&first[p])
                .fold(0.0f32, |m, (e, f)| m.max((e - f).abs()));
            assert!(err <= 1e-4 * scale, "{tag}: {err:e} > 1e-4 * {scale:e}");
        }
    }
    // Newton-Schulz: 256x384 makes every product a whole call.
    let run = |cpu: &CpuBackend| {
        let mut rng = SplitMix64(0x5c4);
        let (rows, cols) = (256usize, 384usize);
        let mut p = f32t(cpu, &rng.vec(rows * cols, 0.1), &[rows, cols]);
        let g = f32t(cpu, &rng.vec(rows * cols, 0.1), &[rows, cols]);
        let mut m = f32t(cpu, &rng.vec(rows * cols, 0.01), &[rows, cols]);
        cpu.muon_ns5_step(&mut p, &g, &mut m, MuonNs5Config::nanolab_default())
            .unwrap();
        flat(&p)
    };
    let (want, first, second) = (run(&exact), run(&fast), run(&fast));
    assert!(first.iter().all(|v| v.is_finite()), "muon: non-finite");
    assert!(bits(&first) == bits(&second), "muon: two runs differ");
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let err = want
        .iter()
        .zip(&first)
        .fold(0.0f32, |m, (e, f)| m.max((e - f).abs()));
    assert!(err <= 1e-4 * scale, "muon: {err:e} > 1e-4 * {scale:e}");
}

/// `tile_fast` (below the cutoff) and `sgemm_tile` (above it, off macOS)
/// both give the ascending FMA chain, at any thread count. Accelerate's
/// order is unspecified, so the 256³ case is not checked on macOS.
#[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
#[test]
fn fast_gemm_off_accelerate_is_the_ascending_fma_chain() {
    for threads in [1usize, 18] {
        let fast = backend(threads, Numerics::Fast);
        let mut checked = 0;
        for &(rows, kin, nout) in &DISPATCH {
            if cfg!(target_os = "macos") && rows * kin * nout >= ojas_cpu::FAST_WHOLE_CALL_MACS {
                continue;
            }
            checked += 1;
            let got = linear_products(&fast, rows, kin, nout);
            let chain = fma_chain(rows, kin, nout);
            for (p, name) in PRODUCTS.iter().enumerate() {
                assert!(
                    bits(&got[p]) == bits(&chain[p]),
                    "{name} {rows}x{kin}x{nout} threads {threads}: not the FMA chain"
                );
            }
        }
        assert!(
            checked >= 2,
            "only {checked} DISPATCH shapes are below the cutoff"
        );
    }
}

#[test]
fn numerics_is_reported_and_survives_clone() {
    let budget = Budget::new(1 << 20);
    // Fast is the default; Exact is opt-in.
    assert_eq!(CpuBackend::new(budget.clone()).numerics(), Numerics::Fast);
    let threaded = CpuBackend::with_threads(budget.clone(), 4).unwrap();
    assert_eq!(threaded.numerics(), Numerics::Fast);
    let exact = threaded.with_numerics(Numerics::Exact);
    assert_eq!(exact.numerics(), Numerics::Exact);
    assert_eq!(exact.clone().numerics(), Numerics::Exact);
    let dyn_backend: &dyn Backend = &exact;
    assert_eq!(dyn_backend.numerics(), Numerics::Exact);
    assert_eq!(
        exact.with_numerics(Numerics::Fast).numerics(),
        Numerics::Fast
    );
}

#[derive(Debug)]
struct OtherDevice(usize);

impl DeviceBuffer for OtherDevice {
    fn backend(&self) -> BackendId {
        BackendId::Wgpu
    }
    fn byte_len(&self) -> usize {
        self.0
    }
    fn read_bytes(&self, _offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        Ok(vec![0; len])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn device(cpu: &CpuBackend, shape: &[usize], dtype: DType) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_device(Arc::new(OtherDevice(n * 4)), shape, dtype, cpu.budget()).unwrap()
}

fn assert_placement<T: std::fmt::Debug>(what: &str, result: Result<T, OjasError>) {
    match result {
        Err(OjasError::Placement {
            found: Some(BackendId::Wgpu),
            ..
        }) => {}
        other => panic!("{what}: expected Placement from Wgpu, got {other:?}"),
    }
}

/// Every op that reads tensor data refuses device memory with
/// `OjasError::Placement`, on the pooled and the Fast path alike.
#[test]
fn device_tensors_are_placement_errors_not_panics() {
    for cpu in [backend(1, Numerics::Exact), backend(18, Numerics::Fast)] {
        let h = |shape: &[usize]| {
            let n: usize = shape.iter().product();
            f32t(&cpu, &vec![0.5; n], shape)
        };
        let d = |shape: &[usize]| device(&cpu, shape, DType::F32);
        let s = [1usize, 2, 300, 16];
        assert_placement("permute", cpu.permute(&d(&[4, 8]), &[1, 0]));
        assert_placement("linear x", cpu.linear_forward(&d(&[4, 8]), &h(&[3, 8])));
        assert_placement("linear w", cpu.linear_forward(&h(&[4, 8]), &d(&[3, 8])));
        assert_placement(
            "linear_backward",
            cpu.linear_backward(&h(&[4, 8]), &h(&[3, 8]), &d(&[4, 3])),
        );
        assert_placement(
            "sdpa",
            cpu.causal_sdpa_forward(&h(&s), &d(&s), &h(&s), None)
                .map(|(y, _)| y),
        );
        assert_placement(
            "sdpa_backward",
            cpu.causal_sdpa_backward_recompute(&h(&s), &h(&s), &h(&s), &d(&s), None),
        );
        assert_placement("rms", cpu.rms_norm_forward(&d(&[4, 8]), &h(&[8]), 1e-6));
        assert_placement(
            "rms_backward",
            cpu.rms_norm_backward(&h(&[4, 8]), &d(&[8]), &h(&[4, 8]), 1e-6),
        );
        assert_placement(
            "rope",
            cpu.rope_half_split_forward(&h(&[4, 8]), &d(&[4, 8]), &h(&[4, 8])),
        );
        let targets = u32t(&cpu, &[1, 2, 3, 0], &[4]);
        assert_placement(
            "cross_entropy",
            cpu.cross_entropy_mean_forward(&d(&[4, 8]), &targets, None),
        );
        let device_targets = device(&cpu, &[4], DType::U32);
        assert_placement(
            "cross_entropy targets",
            cpu.cross_entropy_mean_backward(&h(&[4, 8]), &device_targets, None),
        );
        let (mut p, mut m1, mut m2) = (h(&[8]), h(&[8]), h(&[8]));
        assert_placement(
            "adamw",
            cpu.adamw_step(
                &mut p,
                &d(&[8]),
                &mut m1,
                &mut m2,
                0,
                AdamWConfig::nanolab(1e-3, 0.0),
            ),
        );
        let (mut p, mut m) = (h(&[4, 8]), h(&[4, 8]));
        assert_placement(
            "muon",
            cpu.muon_ns5_step(
                &mut p,
                &d(&[4, 8]),
                &mut m,
                MuonNs5Config::nanolab_default(),
            ),
        );
    }
}

/// 10,000 calls from 16 OS threads on one backend (one shared pool), each
/// large enough to use the pool, every result bit-equal to a serial run.
#[test]
fn ten_thousand_concurrent_calls_share_one_backend() {
    let cpu = backend(18, Numerics::Fast);
    let mut rng = SplitMix64(1700);
    let (rows, kin, nout) = (64usize, 128usize, 96usize);
    let x = f32t(&cpu, &rng.vec(rows * kin, 0.5), &[rows, kin]);
    let w = f32t(&cpu, &rng.vec(nout * kin, 0.5), &[nout, kin]);
    let s = [1usize, 2, 48, 16];
    let n: usize = s.iter().product();
    let q = f32t(&cpu, &rng.vec(n, 1.0), &s);
    let want_y = bits(&flat(
        &backend(1, Numerics::Fast).linear_forward(&x, &w).unwrap(),
    ));
    let want_a = bits(&flat(
        &backend(1, Numerics::Fast)
            .causal_sdpa_forward(&q, &q, &q, None)
            .map(|(y, _)| y)
            .unwrap(),
    ));
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for caller in 0..16 {
            let (cpu, x, w, q) = (&cpu, &x, &w, &q);
            let (want_y, want_a) = (&want_y, &want_a);
            handles.push(
                std::thread::Builder::new()
                    .stack_size(8 << 20)
                    .spawn_scoped(scope, move || {
                        for call in 0..625 {
                            if (caller + call) % 4 == 0 {
                                let a = cpu
                                    .causal_sdpa_forward(q, q, q, None)
                                    .map(|(y, _)| y)
                                    .unwrap();
                                assert!(
                                    bits(&flat(&a)) == *want_a,
                                    "sdpa caller {caller} call {call}"
                                );
                            } else {
                                let y = cpu.linear_forward(x, w).unwrap();
                                assert!(
                                    bits(&flat(&y)) == *want_y,
                                    "linear caller {caller} call {call}"
                                );
                            }
                        }
                    })
                    .expect("spawn"),
            );
        }
        for handle in handles {
            handle.join().expect("join");
        }
    });
}

#[test]
fn nonfinite_and_empty_inputs_are_refused_on_every_path() {
    for numerics in [Numerics::Exact, Numerics::Fast] {
        for threads in [1, 18] {
            let cpu = backend(threads, numerics);
            // Overflow inside the product, not in the inputs.
            let big = f32t(&cpu, &vec![1e30; 300 * 700], &[300, 700]);
            let w = f32t(&cpu, &vec![1e30; 520 * 700], &[520, 700]);
            assert_nonfinite(cpu.linear_forward(&big, &w));
            let mut nan = vec![0.5f32; 64 * 8];
            nan[200] = f32::NAN;
            let nan = f32t(&cpu, &nan, &[64, 8]);
            assert_nonfinite(cpu.linear_forward(&nan, &f32t(&cpu, &[0.5; 24], &[3, 8])));
            // Scores overflow on both attention kernels (T = 64 and T = 300).
            for t in [64usize, 300] {
                let s = [1usize, 2, t, 16];
                let n: usize = s.iter().product();
                let huge = f32t(&cpu, &vec![1e20; n], &s);
                assert_nonfinite(
                    cpu.causal_sdpa_forward(&huge, &huge, &huge, None)
                        .map(|(y, _)| y),
                );
                let ones = f32t(&cpu, &vec![1.0; n], &s);
                assert_nonfinite(
                    cpu.causal_sdpa_backward_recompute(&huge, &huge, &huge, &ones, None),
                );
            }
            let rows = f32t(&cpu, &vec![1e30; 600 * 256], &[600, 256]);
            assert_nonfinite(cpu.rms_norm_forward(&rows, &f32t(&cpu, &[1.0; 256], &[256]), 1e-6));
            // Zero rows never reach a kernel.
            let empty = Tensor::from_f32(&[], &[0, 8], cpu.budget()).unwrap();
            assert_shape(cpu.linear_forward(&empty, &f32t(&cpu, &[0.5; 24], &[3, 8])));
            let empty_t = Tensor::from_f32(&[], &[1, 2, 0, 16], cpu.budget()).unwrap();
            assert_shape(
                cpu.causal_sdpa_forward(&empty_t, &empty_t, &empty_t, None)
                    .map(|(y, _)| y),
            );
        }
    }
}

/// Fast Muon parameter against Exact at 1 and 6 threads, within the 1e-6
/// relative tolerance of `fast_matches_exact_within_stated_tolerance`.
/// Momentum is bit-equal. A NaN gradient is refused before either matrix
/// is written. Tall shapes take the transpose inside Newton-Schulz.
#[test]
fn muon_fast_matches_exact_at_one_and_six_threads_and_refuses_nan() {
    let shapes = [
        (70usize, 40usize),
        (40usize, 70usize),
        (33usize, 17usize),
        (1usize, 1usize),
    ];
    let cfg = MuonNs5Config::nanolab_default();
    let exact = backend(1, Numerics::Exact);
    for &(rows, cols) in &shapes {
        let mut rng = SplitMix64(0x4d30 + (rows * 16 + cols) as u64);
        let n = rows * cols;
        let p0 = rng.vec(n, 0.1);
        let g0 = rng.vec(n, 0.1);
        let m0 = rng.vec(n, 0.01);
        let run = |cpu: &CpuBackend| {
            let mut p = f32t(cpu, &p0, &[rows, cols]);
            let g = f32t(cpu, &g0, &[rows, cols]);
            let mut m = f32t(cpu, &m0, &[rows, cols]);
            cpu.muon_ns5_step(&mut p, &g, &mut m, cfg).unwrap();
            (flat(&p), flat(&m))
        };
        let (want_p, want_m) = run(&exact);
        let scale = want_p
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(f32::MIN_POSITIVE);
        for threads in [1usize, 6] {
            let (got_p, got_m) = run(&backend(threads, Numerics::Fast));
            let err = want_p
                .iter()
                .zip(&got_p)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                got_p.iter().all(|v| v.is_finite()),
                "{rows}x{cols} threads {threads}: non-finite"
            );
            assert!(
                err <= 1e-6 * scale,
                "{rows}x{cols} threads {threads}: max |fast-exact| {err:e} > 1e-6 * {scale:e}"
            );
            assert_eq!(
                bits(&got_m),
                bits(&want_m),
                "{rows}x{cols} threads {threads}: momentum bits"
            );
        }
    }
    for threads in [1usize, 6] {
        let cpu = backend(threads, Numerics::Fast);
        let mut p = f32t(&cpu, &[0.2, -0.1, 0.0, 0.4, 0.1, -0.2, 0.3, 0.05], &[4, 2]);
        let mut m = f32t(
            &cpu,
            &[0.01, 0.0, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02],
            &[4, 2],
        );
        let p_bits = bits(&flat(&p));
        let m_bits = bits(&flat(&m));
        let mut bad = vec![0.1f32; 8];
        bad[3] = f32::NAN;
        let bad = f32t(&cpu, &bad, &[4, 2]);
        assert_nonfinite(cpu.muon_ns5_step(&mut p, &bad, &mut m, cfg));
        assert_eq!(bits(&flat(&p)), p_bits, "threads {threads}: param written");
        assert_eq!(
            bits(&flat(&m)),
            m_bits,
            "threads {threads}: momentum written"
        );
    }
}

/// Nanolab Muon matrices from `bench_ops`. On macOS, at any thread count,
/// `A @ A` and `B @ X` are six row bands, except tall 2048×768, where each
/// is one `cblas_sgemm`. `X @ Xᵀ` is two bands when `k < 2m` (the square)
/// and one band on the tall shapes. Every step must match the single-thread
/// step with max abs 0.
#[test]
fn muon_nanolab_inputs_match_across_one_and_six_threads() {
    fn case_seed(case: &str) -> u64 {
        case.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
    }
    let cfg = MuonNs5Config::nanolab_default();
    let one = backend(1, Numerics::Fast);
    let six = backend(6, Numerics::Fast);
    for (case, rows, cols) in [
        ("muon_768x768", 768usize, 768usize),
        ("muon_2048x768", 2048, 768),
        ("muon_3072x768", 3072, 768),
    ] {
        let mut rng = SplitMix64(case_seed(case));
        let n = rows * cols;
        let p0: Vec<f32> = (0..n).map(|_| 0.035 * rng.unit()).collect();
        let g0: Vec<f32> = (0..n).map(|_| 0.01 * rng.unit()).collect();
        let m0: Vec<f32> = (0..n).map(|_| 0.01 * rng.unit()).collect();
        let run = |cpu: &CpuBackend| {
            let mut p = f32t(cpu, &p0, &[rows, cols]);
            let g = f32t(cpu, &g0, &[rows, cols]);
            let mut m = f32t(cpu, &m0, &[rows, cols]);
            cpu.muon_ns5_step(&mut p, &g, &mut m, cfg).unwrap();
            (flat(&p), flat(&m))
        };
        let (p1, m1) = run(&one);
        let (p6, m6) = run(&six);
        let max = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        };
        assert_eq!(max(&p1, &p6), 0.0, "{case} param");
        assert_eq!(max(&m1, &m6), 0.0, "{case} momentum");
    }
}
