//! Red-team gate for `CpuBackend::linear_forward` / `linear_backward`:
//! numerics contracts, shapes, views, dtypes, placement and non-finite
//! handling. Budget, cancellation and concurrency live in
//! `redteam_linear_budget.rs`.
//!
//! Every Exact expectation is a scalar chain written here: each output starts
//! at `+0.0` and adds `a * b` (two roundings, no `mul_add`) for the reduction
//! index ascending. Fast expectations below 2^21 multiply-adds are the same
//! chain with `mul_add`; at or above it they are a `gamma_k` bound against an
//! f64 reference plus run-to-run repeatability.
//!
//! Expensive variants are `#[ignore]`d; run them with
//! `cargo test -p ojas-cpu --release --test redteam_linear -- --ignored`.

mod common;

use std::any::Any;
use std::sync::Arc;

use common::SplitMix64;
use ojas_core::{
    Backend, BackendId, Budget, DType, DeviceBuffer, Numerics, OjasError, Scratch, Tensor,
};
use ojas_cpu::CpuBackend;

const THREADS: [usize; 8] = [1, 2, 3, 5, 6, 7, 16, 18];
const HUGE: u64 = 1 << 40;
/// Multiply-adds at which a Fast GEMM leaves the packed kernel.
const WHOLE_CALL_MACS: usize = ojas_cpu::FAST_WHOLE_CALL_MACS;

fn inputs_budget() -> Budget {
    Budget::new(u64::MAX)
}

fn cpu(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(HUGE), threads)
        .unwrap()
        .with_numerics(numerics)
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &inputs_budget()).unwrap()
}

/// F32 tensor whose values have exactly `bits`. `f32::from_bits` and a
/// store keep every bit, so signalling NaNs reach the op unquieted; no
/// arithmetic touches them.
fn raw_f32(bits: &[u32], shape: &[usize]) -> Tensor {
    let budget = inputs_budget();
    let mut scratch = Scratch::<f32>::try_alloc(bits.len(), &budget).unwrap();
    for (slot, &word) in scratch.as_mut_slice().iter_mut().zip(bits) {
        *slot = f32::from_bits(word);
    }
    Tensor::from_scratch(scratch, shape).unwrap()
}

fn vals(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

/// First differing index, for a readable failure instead of a 600k-element dump.
fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
        panic!(
            "{what}: first bit mismatch at {i}: got {:?} ({:#010x}) want {:?} ({:#010x})",
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits()
        );
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Chain {
    /// `acc + a * b`, two roundings.
    Exact,
    /// `a.mul_add(b, acc)`, one rounding.
    Fma,
}

/// `C[m, n] = sum_p a(i, p) * b[p * n + j]`, `p` ascending from `+0.0`.
fn chain(
    m: usize,
    k: usize,
    n: usize,
    a: impl Fn(usize, usize) -> f32,
    b: &[f32],
    mode: Chain,
) -> Vec<f32> {
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        let row = &mut c[i * n..(i + 1) * n];
        for p in 0..k {
            let s = a(i, p);
            let brow = &b[p * n..(p + 1) * n];
            match mode {
                Chain::Exact => {
                    for (acc, &bv) in row.iter_mut().zip(brow) {
                        *acc += s * bv;
                    }
                }
                Chain::Fma => {
                    for (acc, &bv) in row.iter_mut().zip(brow) {
                        *acc = s.mul_add(bv, *acc);
                    }
                }
            }
        }
    }
    c
}

/// f64 value and `sum |a||b|` per output, for the `gamma_k` bound.
fn chain_f64(
    m: usize,
    k: usize,
    n: usize,
    a: impl Fn(usize, usize) -> f32,
    b: &[f32],
) -> (Vec<f64>, Vec<f64>) {
    let mut c = vec![0.0f64; m * n];
    let mut abs = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let s = f64::from(a(i, p));
            for j in 0..n {
                let prod = s * f64::from(b[p * n + j]);
                c[i * n + j] += prod;
                abs[i * n + j] += prod.abs();
            }
        }
    }
    (c, abs)
}

fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; a.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

/// Inputs and the three products of one linear layer.
struct Case {
    rows: usize,
    kin: usize,
    nout: usize,
    x: Vec<f32>,
    w: Vec<f32>,
    g: Vec<f32>,
}

struct Products {
    y: Vec<f32>,
    gx: Vec<f32>,
    gw: Vec<f32>,
}

impl Case {
    fn new(rows: usize, kin: usize, nout: usize, rng: &mut SplitMix64, gen: Gen) -> Self {
        Self {
            rows,
            kin,
            nout,
            x: gen.fill(rng, rows * kin),
            w: gen.fill(rng, nout * kin),
            g: gen.fill(rng, rows * nout),
        }
    }

    fn reference(&self, mode: Chain) -> Products {
        let (rows, kin, nout) = (self.rows, self.kin, self.nout);
        let wt = transpose(&self.w, nout, kin);
        let x = &self.x;
        let g = &self.g;
        Products {
            y: chain(rows, kin, nout, |r, i| x[r * kin + i], &wt, mode),
            gx: chain(rows, nout, kin, |r, o| g[r * nout + o], &self.w, mode),
            gw: chain(nout, rows, kin, |o, r| g[r * nout + o], x, mode),
        }
    }

    fn reference_f64(&self) -> [(Vec<f64>, Vec<f64>); 3] {
        let (rows, kin, nout) = (self.rows, self.kin, self.nout);
        let wt = transpose(&self.w, nout, kin);
        let x = &self.x;
        let g = &self.g;
        [
            chain_f64(rows, kin, nout, |r, i| x[r * kin + i], &wt),
            chain_f64(rows, nout, kin, |r, o| g[r * nout + o], &self.w),
            chain_f64(nout, rows, kin, |o, r| g[r * nout + o], x),
        ]
    }

    fn run(&self, cpu: &CpuBackend, x_shape: &[usize]) -> Products {
        let x = t(&self.x, x_shape);
        let w = t(&self.w, &[self.nout, self.kin]);
        let mut y_shape = x_shape[..x_shape.len() - 1].to_vec();
        y_shape.push(self.nout);
        let g = t(&self.g, &y_shape);
        let y = cpu.linear_forward(&x, &w).unwrap();
        assert_eq!(y.shape(), y_shape.as_slice(), "forward output shape");
        let (gx, gw) = cpu.linear_backward(&x, &w, &g).unwrap();
        assert_eq!(gx.shape(), x_shape, "grad_input shape");
        assert_eq!(gw.shape(), &[self.nout, self.kin], "grad_weight shape");
        Products {
            y: vals(&y),
            gx: vals(&gx),
            gw: vals(&gw),
        }
    }

    fn run2d(&self, cpu: &CpuBackend) -> Products {
        self.run(cpu, &[self.rows, self.kin])
    }

    fn macs(&self) -> usize {
        self.rows * self.kin * self.nout
    }
}

fn assert_products(got: &Products, want: &Products, what: &str) {
    assert_bits(&got.y, &want.y, &format!("{what} y"));
    assert_bits(&got.gx, &want.gx, &format!("{what} grad_x"));
    assert_bits(&got.gw, &want.gw, &format!("{what} grad_w"));
}

/// Value distributions. `Nasty` mixes signed zeros, subnormals, powers of two
/// and large magnitudes so any reordering of a sum changes the bits; every
/// value and partial sum stays finite.
#[derive(Clone, Copy, Debug)]
enum Gen {
    Uniform,
    Nasty,
}

impl Gen {
    fn fill(self, rng: &mut SplitMix64, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.one(rng)).collect()
    }

    fn one(self, rng: &mut SplitMix64) -> f32 {
        match self {
            Gen::Uniform => rng.unit(),
            Gen::Nasty => {
                let sign = if rng.next_u64() & 1 == 0 {
                    1.0f32
                } else {
                    -1.0
                };
                match rng.below(10) {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::from_bits((rng.next_u64() as u32 & 0x007f_ffff) | 1) * sign,
                    3 => sign * 2f32.powi(rng.below(41) as i32 - 20),
                    4 | 5 => sign * (1e3 + 1e6 * rng.unit().abs()),
                    _ => rng.unit(),
                }
            }
        }
    }
}

// ---------------------------------------------------------------- Exact

/// Every dimension takes each value either side of MR=6, NR=16, MC=72,
/// KC/NC=256 while the other two stay odd; then multi-tile products whose
/// tiles straddle the same boundaries. Every thread count.
#[test]
fn exact_linear_matches_scalar_chain_on_block_boundary_shapes() {
    let edges = [1usize, 5, 6, 7, 15, 16, 17, 71, 72, 73, 255, 256, 257];
    let mut shapes = Vec::new();
    for &v in &edges {
        shapes.push((v, 67, 45));
        shapes.push((45, v, 67));
        shapes.push((67, 45, v));
    }
    shapes.extend([
        (73, 257, 257),
        (257, 255, 73),
        (72, 256, 256),
        (71, 513, 129),
        (257, 257, 257),
        (145, 300, 511),
        (3, 5000, 17),
    ]);
    let mut rng = SplitMix64(0x7ed7_ea41);
    let backends: Vec<CpuBackend> = THREADS.iter().map(|&n| cpu(n, Numerics::Exact)).collect();
    for &(rows, kin, nout) in &shapes {
        let case = Case::new(rows, kin, nout, &mut rng, Gen::Nasty);
        let want = case.reference(Chain::Exact);
        for be in &backends {
            let got = case.run2d(be);
            assert_products(
                &got,
                &want,
                &format!("exact {rows}x{kin}x{nout} threads {}", be.threads()),
            );
        }
    }
}

fn degenerate_and_model_shapes(threads: &[usize]) {
    let shapes = [
        (512usize, 768usize, 768usize),
        (1, 768, 768),
        (768, 1, 768),
        (1, 300, 513),
        (513, 300, 1),
        (513, 1, 300),
        (1, 1, 1),
        (1, 4096, 1),
    ];
    let mut rng = SplitMix64(0x0de9_e4e7);
    let backends: Vec<CpuBackend> = threads.iter().map(|&n| cpu(n, Numerics::Exact)).collect();
    for &(rows, kin, nout) in &shapes {
        let case = Case::new(rows, kin, nout, &mut rng, Gen::Uniform);
        let want = case.reference(Chain::Exact);
        for be in &backends {
            let got = case.run2d(be);
            assert_products(
                &got,
                &want,
                &format!("exact {rows}x{kin}x{nout} threads {}", be.threads()),
            );
        }
    }
}

/// 512x768x768 (a GPT-2-small layer), rows=1, in=1, out=1, on every
/// thread count (about 0.3 s in release).
#[test]
fn exact_linear_model_and_degenerate_shapes_match_scalar_chain() {
    degenerate_and_model_shapes(&THREADS);
}

/// The references are not vacuous: on the boundary data the Exact chain
/// differs from the FMA chain and from the same sum taken in descending
/// order, so a kernel that fused or reordered would fail the bit checks.
#[test]
fn references_discriminate_fusion_and_reordering() {
    let mut rng = SplitMix64(0xd15c);
    let case = Case::new(73, 257, 257, &mut rng, Gen::Nasty);
    let exact = case.reference(Chain::Exact);
    let fma = case.reference(Chain::Fma);
    let differs = |a: &[f32], b: &[f32]| a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits());
    assert!(
        differs(&exact.y, &fma.y) && differs(&exact.gx, &fma.gx) && differs(&exact.gw, &fma.gw)
    );
    let (rows, kin, nout) = (case.rows, case.kin, case.nout);
    let mut descending = vec![0.0f32; rows * nout];
    for r in 0..rows {
        for o in 0..nout {
            let mut acc = 0.0f32;
            for i in (0..kin).rev() {
                acc += case.x[r * kin + i] * case.w[o * kin + i];
            }
            descending[r * nout + o] = acc;
        }
    }
    assert!(differs(&exact.y, &descending));
    assert_products(
        &case.run2d(&cpu(7, Numerics::Exact)),
        &exact,
        "discriminating case",
    );
}

/// `[b, t, in]` and `[a, b, t, in]` give the same bits as the flattened 2-D
/// call, and a rank-1 `[in]` input is one row with output `[out]`.
#[test]
fn exact_rank_3_4_and_1_inputs_flatten_to_rows() {
    let mut rng = SplitMix64(0x4a4b_3c3d);
    for &threads in &[1usize, 7] {
        let be = cpu(threads, Numerics::Exact);
        let case = Case::new(2 * 3 * 37, 45, 29, &mut rng, Gen::Nasty);
        let want = case.reference(Chain::Exact);
        for shape in [vec![2 * 3 * 37, 45], vec![6, 37, 45], vec![2, 3, 37, 45]] {
            let got = case.run(&be, &shape);
            assert_products(
                &got,
                &want,
                &format!("rank {} threads {threads}", shape.len()),
            );
        }
        let one = Case::new(1, 45, 29, &mut rng, Gen::Nasty);
        let got = one.run(&be, &[45]);
        assert_products(&got, &one.reference(Chain::Exact), "rank-1 input");
    }
}

/// Signed zeros sum to `+0.0` from the `+0.0` start, subnormal products are
/// kept (no flush to zero), and subnormal inputs pass through, on every
/// thread count and on a multi-tile shape.
#[test]
fn exact_signed_zero_and_subnormals_match_scalar_chain() {
    for &(rows, kin, nout) in &[(7usize, 9usize, 17usize), (73, 257, 257)] {
        let n = |a, b| a * b;
        let tiny = 1e-20f32;
        let sub = f32::from_bits(0x0000_0400); // subnormal
        let mut rng = SplitMix64(0x5ab0);
        let pick = |rng: &mut SplitMix64| match rng.below(6) {
            0 => -0.0f32,
            1 => 0.0,
            2 => tiny,
            3 => -tiny,
            4 => sub,
            _ => -sub * 3.0,
        };
        let case = Case {
            rows,
            kin,
            nout,
            x: (0..n(rows, kin)).map(|_| pick(&mut rng)).collect(),
            w: (0..n(nout, kin)).map(|_| pick(&mut rng)).collect(),
            g: (0..n(rows, nout)).map(|_| pick(&mut rng)).collect(),
        };
        let want = case.reference(Chain::Exact);
        assert!(
            want.y.iter().any(|v| v.is_subnormal()),
            "fixture must produce subnormal outputs"
        );
        for &threads in &THREADS {
            let got = case.run2d(&cpu(threads, Numerics::Exact));
            assert_products(
                &got,
                &want,
                &format!("subnormal {rows}x{kin}x{nout} t{threads}"),
            );
        }
        // All -0.0 input: every product is -0.0 or +0.0, every output +0.0.
        let negz = Case {
            rows,
            kin,
            nout,
            x: vec![-0.0; rows * kin],
            w: vec![1.5; nout * kin],
            g: vec![-0.0; rows * nout],
        };
        let got = negz.run2d(&cpu(7, Numerics::Exact));
        assert!(
            got.y.iter().all(|v| v.to_bits() == 0),
            "-0.0 inputs give +0.0"
        );
        assert!(got.gx.iter().all(|v| v.to_bits() == 0));
        assert!(got.gw.iter().all(|v| v.to_bits() == 0));
    }
}

/// Exact sums ascending: `MAX + MAX` overflows before `-MAX` arrives, while
/// `MAX + -MAX + MAX` stays finite. A reordered or pairwise sum flips both.
#[test]
fn exact_overflow_depends_on_ascending_order() {
    let m = f32::MAX;
    for &threads in &[1usize, 7] {
        let be = cpu(threads, Numerics::Exact);
        let w = t(&[1.0, 1.0, 1.0], &[1, 3]);
        let over = be.linear_forward(&t(&[m, m, -m], &[1, 3]), &w);
        assert_nonfinite(over, &format!("MAX + MAX - MAX threads {threads}"));
        assert_eq!(be.budget().live_bytes().unwrap(), 0);
        let fine = be.linear_forward(&t(&[m, -m, m], &[1, 3]), &w).unwrap();
        assert_eq!(vals(&fine), vec![m]);
    }
}

// ---------------------------------------------------------------- Property sweep

fn sweep(seed: u64, cases: usize) {
    let mut rng = SplitMix64(seed);
    let backends: Vec<CpuBackend> = THREADS.iter().map(|&n| cpu(n, Numerics::Exact)).collect();
    let fast: Vec<CpuBackend> = backends
        .iter()
        .map(|b| b.clone().with_numerics(Numerics::Fast))
        .collect();
    for index in 0..cases {
        let rows = 1 + rng.below(140);
        let kin = 1 + rng.below(300);
        let nout = 1 + rng.below(140);
        let gen = if rng.below(2) == 0 {
            Gen::Uniform
        } else {
            Gen::Nasty
        };
        let case = Case::new(rows, kin, nout, &mut rng, gen);
        let be = &backends[rng.below(THREADS.len())];
        let what = format!(
            "seed {seed:#x} case {index}: {rows}x{kin}x{nout} {gen:?} threads {} \
             (rerun: REDTEAM_SEED={seed:#x})",
            be.threads()
        );
        let x_shape = if rows.is_multiple_of(3) && rng.below(2) == 0 {
            vec![3, rows / 3, kin]
        } else {
            vec![rows, kin]
        };
        assert_products(
            &case.run(be, &x_shape),
            &case.reference(Chain::Exact),
            &what,
        );
        if cfg!(any(target_arch = "aarch64", target_feature = "fma"))
            && case.macs() < WHOLE_CALL_MACS
        {
            let fb = &fast[rng.below(THREADS.len())];
            let got = case.run(fb, &x_shape);
            assert_products(&got, &case.reference(Chain::Fma), &format!("fast {what}"));
        }
    }
}

fn sweep_seed() -> u64 {
    match std::env::var("REDTEAM_SEED") {
        Ok(text) => {
            let text = text.trim_start_matches("0x");
            u64::from_str_radix(text, 16).expect("REDTEAM_SEED is hex")
        }
        Err(_) => 0x0ba5_e5ee_d000_0001,
    }
}

/// Random shapes, value mixes, ranks and thread counts. The failing seed and
/// case are in the panic message; `REDTEAM_SEED=0x... cargo test` replays.
#[test]
fn randomized_linear_sweep_matches_scalar_chains() {
    sweep(sweep_seed(), 80);
}

#[test]
#[ignore = "slow: cargo test -p ojas-cpu --release --test redteam_linear -- --ignored"]
fn randomized_linear_sweep_long() {
    for s in 0..16u64 {
        sweep(sweep_seed() ^ (s.wrapping_mul(0x9e37_79b9_7f4a_7c15)), 200);
    }
}

// ---------------------------------------------------------------- Fast

/// Below the whole-call cutoff a Fast linear is the ascending `mul_add`
/// chain, bit for bit, on every thread count. The shapes that are not below
/// the cutoff on this platform are skipped; at least four always run.
#[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
#[test]
fn fast_below_cutoff_is_the_ascending_fma_chain_on_every_thread_count() {
    let mut rng = SplitMix64(0xfa57);
    let shapes = [
        (127usize, 128usize, 128usize), // 2_080_768: one tile short of 2^21
        (7, 9, 17),
        (73, 7, 13),
        (1, 300, 17),
        (64, 1, 64),
        (73, 11, 13),
        (1, 768, 768),
        (768, 1, 768),
        (100, 100, 100),
        (145, 97, 147),
    ];
    let below: Vec<_> = shapes
        .into_iter()
        .filter(|&(rows, kin, nout)| rows * kin * nout < WHOLE_CALL_MACS)
        .collect();
    assert!(below.len() >= 4, "{below:?}");
    let backends: Vec<CpuBackend> = THREADS.iter().map(|&n| cpu(n, Numerics::Fast)).collect();
    for (rows, kin, nout) in below {
        let case = Case::new(rows, kin, nout, &mut rng, Gen::Nasty);
        let want = case.reference(Chain::Fma);
        for be in &backends {
            let got = case.run2d(be);
            assert_products(
                &got,
                &want,
                &format!("fast {rows}x{kin}x{nout} threads {}", be.threads()),
            );
        }
    }
}

/// `|y - y64| <= gamma_k * sum |a||b|`, `gamma_k = k u / (1 - k u)`.
fn assert_gamma(got: &[f32], want: &(Vec<f64>, Vec<f64>), k: usize, what: &str) {
    let u = f64::from(f32::EPSILON) / 2.0;
    let ku = k as f64 * u;
    let gamma = ku / (1.0 - ku);
    for (i, (&g, (&v, &a))) in got.iter().zip(want.0.iter().zip(&want.1)).enumerate() {
        assert!(g.is_finite(), "{what}: non-finite at {i}");
        let err = (f64::from(g) - v).abs();
        // The f64 reference has its own k * 2^-53 relative error.
        let bound = gamma * a + a * k as f64 * f64::EPSILON + f64::from(f32::from_bits(1));
        assert!(
            err <= bound,
            "{what}: |{g} - {v}| = {err} > gamma_{k} bound {bound} at {i}"
        );
    }
}

fn fast_whole_call(threads: &[usize]) {
    let mut rng = SplitMix64(0xacce1);
    for &(rows, kin, nout) in &[
        (128usize, 128usize, 128usize),
        (512, 768, 768),
        (300, 129, 257),
    ] {
        let case = Case::new(rows, kin, nout, &mut rng, Gen::Uniform);
        assert!(case.macs() >= WHOLE_CALL_MACS);
        let [ry, rgx, rgw] = case.reference_f64();
        #[cfg(not(target_os = "macos"))]
        let fma = case.reference(Chain::Fma);
        for &n in threads {
            let be = cpu(n, Numerics::Fast);
            let first = case.run2d(&be);
            let what = format!("fast whole {rows}x{kin}x{nout} threads {n}");
            assert_gamma(&first.y, &ry, kin, &format!("{what} y"));
            assert_gamma(&first.gx, &rgx, nout, &format!("{what} grad_x"));
            assert_gamma(&first.gw, &rgw, rows, &format!("{what} grad_w"));
            for _ in 0..2 {
                assert_products(&case.run2d(&be), &first, &format!("{what} repeat"));
            }
            // Off macOS a whole call is sgemm_tile: the FMA chain exactly.
            #[cfg(not(target_os = "macos"))]
            assert_products(&first, &fma, &format!("{what} fma chain"));
        }
    }
}

/// At or above 2^21 multiply-adds Fast is one Accelerate call on macOS:
/// finite, within `gamma_k` of f64, and repeatable on one backend, on every
/// thread count (about 0.5 s in release).
#[test]
fn fast_whole_call_is_repeatable_and_within_gamma_k() {
    fast_whole_call(&THREADS);
}

/// Exact also meets the `gamma_k` bound (guards the reference itself).
#[test]
fn exact_is_within_gamma_k_of_f64() {
    let mut rng = SplitMix64(0x9a33a);
    let case = Case::new(97, 513, 131, &mut rng, Gen::Uniform);
    let got = case.run2d(&cpu(5, Numerics::Exact));
    let [ry, rgx, rgw] = case.reference_f64();
    assert_gamma(&got.y, &ry, 513, "exact y");
    assert_gamma(&got.gx, &rgx, 131, "exact grad_x");
    assert_gamma(&got.gw, &rgw, 97, "exact grad_w");
}

// ---------------------------------------------------------------- Non-finite

const POISON: [u32; 8] = [
    0x7fc0_0000, // quiet NaN
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7f80_0001, // signalling NaN, smallest payload
    0xff80_0001, // negative signalling NaN
    0x7fbf_ffff, // signalling NaN, largest payload
    0x7fc1_2345, // quiet NaN with payload
    0xffff_ffff, // all ones
];

fn finite_bits(n: usize, rng: &mut SplitMix64) -> Vec<u32> {
    (0..n).map(|_| rng.unit().to_bits()).collect()
}

fn positions(n: usize) -> Vec<usize> {
    let mut p = vec![0, 1, n / 2, n - 2, n - 1, 255, 256, 257];
    p.retain(|&i| i < n);
    p.sort_unstable();
    p.dedup();
    p
}

fn assert_nonfinite<T>(r: Result<T, OjasError>, what: &str) {
    match r {
        Err(OjasError::NonFinite { .. }) => {}
        Err(err) => panic!("{what}: expected NonFinite, got {err:?}"),
        Ok(_) => panic!("{what}: expected NonFinite, got Ok"),
    }
}

/// A NaN, infinity or signalling NaN anywhere in any operand is `NonFinite`
/// on a backend whose cap is 0 bytes, so nothing was charged first. Covers
/// the first/middle/last element and both sides of the 1024-byte scan block.
#[test]
fn nonfinite_in_any_operand_is_refused_before_any_charge() {
    let (rows, kin, nout) = (20usize, 15usize, 18usize); // x 300, w 270, g 360 elements
    let mut rng = SplitMix64(0x0bad);
    let x = finite_bits(rows * kin, &mut rng);
    let w = finite_bits(nout * kin, &mut rng);
    let g = finite_bits(rows * nout, &mut rng);
    for &threads in &[1usize, 7] {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let be = CpuBackend::with_threads(Budget::new(0), threads)
                .unwrap()
                .with_numerics(numerics);
            for operand in 0..3 {
                let base = [&x, &w, &g][operand];
                for at in positions(base.len()) {
                    for &poison in &POISON {
                        let mut bad = base.clone();
                        bad[at] = poison;
                        let pick = |i: usize, src: &Vec<u32>| {
                            if i == operand {
                                bad.clone()
                            } else {
                                src.clone()
                            }
                        };
                        let xt = raw_f32(&pick(0, &x), &[rows, kin]);
                        let wt = raw_f32(&pick(1, &w), &[nout, kin]);
                        let gt = raw_f32(&pick(2, &g), &[rows, nout]);
                        let what = format!(
                            "operand {operand} at {at} = {poison:#010x} threads {threads} {numerics:?}"
                        );
                        if operand < 2 {
                            assert_nonfinite(be.linear_forward(&xt, &wt), &format!("fwd {what}"));
                        }
                        assert_nonfinite(be.linear_backward(&xt, &wt, &gt), &format!("bwd {what}"));
                        assert_eq!(be.budget().live_bytes().unwrap(), 0, "{what}");
                    }
                }
            }
        }
    }
}

/// A poisoned element at the very end of a 1M-element operand.
#[test]
fn nonfinite_at_the_end_of_a_large_operand() {
    let (rows, kin, nout) = (1024usize, 1023usize, 3usize);
    let mut rng = SplitMix64(0xe0d);
    let mut x = finite_bits(rows * kin, &mut rng);
    *x.last_mut().unwrap() = 0x7f80_0001;
    let w = finite_bits(nout * kin, &mut rng);
    let be = CpuBackend::with_threads(Budget::new(0), 7).unwrap();
    assert_nonfinite(
        be.linear_forward(&raw_f32(&x, &[rows, kin]), &raw_f32(&w, &[nout, kin])),
        "last element sNaN",
    );
}

/// Finite inputs whose sum overflows are `NonFinite`, and the budget is
/// back where it started; in backward one gradient is finite and the other
/// overflows, in each order, so an already-built gradient must be released.
#[test]
fn finite_inputs_that_overflow_are_nonfinite_and_release_the_budget() {
    for &threads in &[1usize, 7] {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let be = cpu(threads, numerics);
            let what = format!("threads {threads} {numerics:?}");
            // 2e38 + 2e38 overflows in the sum, not in a product.
            let r = be.linear_forward(&t(&[2e19, 2e19], &[1, 2]), &t(&[1e19, 1e19], &[1, 2]));
            assert_nonfinite(r, &format!("sum overflow {what}"));
            // One product overflows.
            let r = be.linear_forward(&t(&[1e20; 4], &[2, 2]), &t(&[1e20; 2], &[1, 2]));
            assert_nonfinite(r, &format!("product overflow {what}"));
            assert_eq!(be.budget().live_bytes().unwrap(), 0, "{what}");
            // grad_x finite (1e9 * 1e-30), grad_w overflows (1e9 * 1e30 * 2).
            let x = t(&[1e30, 1e30], &[2, 1]);
            let w = t(&[1e-30], &[1, 1]);
            let g = t(&[1e9, 1e9], &[2, 1]);
            assert_nonfinite(
                be.linear_backward(&x, &w, &g),
                &format!("grad_w overflow {what}"),
            );
            assert_eq!(
                be.budget().live_bytes().unwrap(),
                0,
                "grad_w overflow {what}"
            );
            // grad_x overflows, grad_w finite.
            let x = t(&[1e-30, 1e-30], &[2, 1]);
            let w = t(&[1e30], &[1, 1]);
            assert_nonfinite(
                be.linear_backward(&x, &w, &g),
                &format!("grad_x overflow {what}"),
            );
            assert_eq!(
                be.budget().live_bytes().unwrap(),
                0,
                "grad_x overflow {what}"
            );
            // Near the top of the range but finite: must succeed.
            let y = be
                .linear_forward(&t(&[f32::MAX, 1.7e38], &[2, 1]), &t(&[1.0], &[1, 1]))
                .unwrap();
            assert_eq!(vals(&y), vec![f32::MAX, 1.7e38], "{what}");
            drop(y);
            assert_eq!(be.budget().live_bytes().unwrap(), 0, "{what}");
        }
    }
}

// ---------------------------------------------------------------- Shapes, views, dtypes, placement

fn assert_shape_err<T>(r: Result<T, OjasError>, what: &str) {
    match r {
        Err(OjasError::Shape { .. }) => {}
        Err(err) => panic!("{what}: expected Shape, got {err:?}"),
        Ok(_) => panic!("{what}: expected Shape, got Ok"),
    }
}

/// Empty axes are `Shape` with nothing charged (cap 0).
#[test]
fn zero_dims_are_shape_errors_before_any_charge() {
    let be = CpuBackend::with_threads(Budget::new(0), 7).unwrap();
    let ok_x = t(&[1.0; 12], &[3, 4]);
    let ok_w = t(&[1.0; 8], &[2, 4]);
    let ok_g = t(&[1.0; 6], &[3, 2]);
    for (x_shape, w_shape) in [
        (vec![0usize, 4usize], vec![2usize, 4usize]),
        (vec![3, 0], vec![2, 0]),
        (vec![3, 4], vec![0, 4]),
        (vec![0, 0], vec![0, 0]),
        (vec![2, 0, 4], vec![2, 4]),
    ] {
        let x = if x_shape.contains(&0) {
            t(&[], &x_shape)
        } else {
            ok_x.clone()
        };
        let w = if w_shape.contains(&0) {
            t(&[], &w_shape)
        } else {
            ok_w.clone()
        };
        let what = format!("x {x_shape:?} w {w_shape:?}");
        assert_shape_err(be.linear_forward(&x, &w), &format!("fwd {what}"));
        assert_shape_err(be.linear_backward(&x, &w, &ok_g), &format!("bwd {what}"));
        assert_eq!(be.budget().live_bytes().unwrap(), 0);
    }
    assert_shape_err(
        be.linear_backward(&ok_x, &ok_w, &t(&[], &[3, 0])),
        "empty grad",
    );
}

/// Shape mismatches found after the input copies are charged still leave
/// the budget where it was.
#[test]
fn mismatched_shapes_are_shape_errors_and_release_the_budget() {
    for &threads in &[1usize, 7] {
        let be = cpu(threads, Numerics::Exact);
        let x = t(&[1.0; 12], &[3, 4]);
        let w = t(&[1.0; 8], &[2, 4]);
        let g = t(&[1.0; 6], &[3, 2]);
        let cases: Vec<(&str, Result<(), OjasError>)> = vec![
            (
                "inner dim",
                be.linear_forward(&x, &t(&[1.0; 10], &[2, 5])).map(drop),
            ),
            (
                "weight rank 1",
                be.linear_forward(&x, &t(&[1.0; 4], &[4])).map(drop),
            ),
            (
                "weight rank 3",
                be.linear_forward(&x, &t(&[1.0; 8], &[1, 2, 4])).map(drop),
            ),
            (
                "weight rank 0",
                be.linear_forward(&x, &t(&[1.0], &[])).map(drop),
            ),
            (
                "input rank 0",
                be.linear_forward(&t(&[1.0], &[]), &w).map(drop),
            ),
            (
                "bwd inner dim",
                be.linear_backward(&x, &t(&[1.0; 10], &[2, 5]), &g)
                    .map(drop),
            ),
            (
                "grad wrong out",
                be.linear_backward(&x, &w, &t(&[1.0; 9], &[3, 3])).map(drop),
            ),
            (
                "grad transposed",
                be.linear_backward(&x, &w, &t(&[1.0; 6], &[2, 3])).map(drop),
            ),
            (
                "grad flat",
                be.linear_backward(&x, &w, &t(&[1.0; 6], &[6])).map(drop),
            ),
            (
                "grad rank 3",
                be.linear_backward(&x, &w, &t(&[1.0; 6], &[1, 3, 2]))
                    .map(drop),
            ),
        ];
        for (what, r) in cases {
            assert_shape_err(r, what);
            assert_eq!(be.budget().live_bytes().unwrap(), 0, "{what}");
        }
    }
}

/// Transposed, broadcast (stride 0) and padded-row views are refused as
/// `Shape` before any charge; contiguous views at a byte offset are read
/// from the window, and operands may share one allocation.
#[test]
fn views_noncontiguous_refused_offset_windows_read_correctly() {
    let refuse = CpuBackend::with_threads(Budget::new(0), 7).unwrap();
    let base = t(
        &(0..64).map(|i| i as f32 * 0.25).collect::<Vec<_>>(),
        &[8, 8],
    );
    let x = base.narrow(0, &[3, 4], &[4, 1]).unwrap();
    let transposed = base.view(&[4, 3], &[1, 4], 0).unwrap();
    let broadcast = base.view(&[3, 4], &[0, 1], 0).unwrap();
    let padded = base.view(&[3, 4], &[8, 1], 0).unwrap();
    for (what, bad) in [
        ("transposed", &transposed),
        ("broadcast", &broadcast),
        ("padded", &padded),
    ] {
        assert_shape_err(
            refuse.linear_forward(bad, &x.view(&[4, 4], &[4, 1], 0).unwrap()),
            what,
        );
        assert_shape_err(refuse.linear_forward(&x, bad), what);
        assert_eq!(refuse.budget().live_bytes().unwrap(), 0);
    }
    // x, w and g are all windows into one allocation, at offsets that are
    // not 16-byte aligned.
    let mut rng = SplitMix64(0x0ff5);
    let (rows, kin, nout) = (37usize, 29usize, 41usize);
    let case = Case::new(rows, kin, nout, &mut rng, Gen::Nasty);
    let xo = 3usize;
    let wo = xo + rows * kin + 5;
    let go = wo + nout * kin + 2;
    let mut storage = vec![f32::from_bits(0x7fc0_0000); go + rows * nout + 7];
    storage[xo..xo + rows * kin].copy_from_slice(&case.x);
    storage[wo..wo + nout * kin].copy_from_slice(&case.w);
    storage[go..go + rows * nout].copy_from_slice(&case.g);
    // NaN padding outside the windows: reading past a window is NonFinite.
    let all = t(&storage, &[storage.len()]);
    let xv = all.view(&[rows, kin], &[kin, 1], xo * 4).unwrap();
    let wv = all.view(&[nout, kin], &[kin, 1], wo * 4).unwrap();
    let gv = all.narrow(go * 4, &[rows, nout], &[nout, 1]).unwrap();
    let want = case.reference(Chain::Exact);
    for &threads in &[1usize, 7, 18] {
        let be = cpu(threads, Numerics::Exact);
        let y = be.linear_forward(&xv, &wv).unwrap();
        assert_bits(&vals(&y), &want.y, "offset view y");
        let (gx, gw) = be.linear_backward(&xv, &wv, &gv).unwrap();
        assert_bits(&vals(&gx), &want.gx, "offset view grad_x");
        assert_bits(&vals(&gw), &want.gw, "offset view grad_w");
    }
    // The same tensor as input and grad (kin == nout).
    let sq = t(&case.x[..rows * kin], &[rows, kin]);
    let ww = t(&case.x[..kin * kin], &[kin, kin]);
    let be = cpu(7, Numerics::Exact);
    let (gx, gw) = be.linear_backward(&sq, &ww, &sq).unwrap();
    let alias = Case {
        rows,
        kin,
        nout: kin,
        x: case.x[..rows * kin].to_vec(),
        w: case.x[..kin * kin].to_vec(),
        g: case.x[..rows * kin].to_vec(),
    };
    let want = alias.reference(Chain::Exact);
    assert_bits(&vals(&gx), &want.gx, "aliased grad_x");
    assert_bits(&vals(&gw), &want.gw, "aliased grad_w");
}

/// U32 in any operand is `Dtype` with nothing charged.
#[test]
fn u32_operands_are_dtype_errors_before_any_charge() {
    let be = CpuBackend::with_threads(Budget::new(0), 7).unwrap();
    let ib = inputs_budget();
    let x = t(&[1.0; 12], &[3, 4]);
    let w = t(&[1.0; 8], &[2, 4]);
    let g = t(&[1.0; 6], &[3, 2]);
    let ux = Tensor::from_u32(&[1; 12], &[3, 4], &ib).unwrap();
    let uw = Tensor::from_u32(&[1; 8], &[2, 4], &ib).unwrap();
    let ug = Tensor::from_u32(&[1; 6], &[3, 2], &ib).unwrap();
    let dtype = |r: Result<(), OjasError>, what: &str| match r {
        Err(OjasError::Dtype {
            expected: DType::F32,
            got: DType::U32,
            ..
        }) => {}
        other => panic!("{what}: expected Dtype, got {other:?}"),
    };
    dtype(be.linear_forward(&x, &uw).map(drop), "u32 weight");
    dtype(be.linear_forward(&ux, &w).map(drop), "u32 input");
    dtype(be.linear_backward(&x, &uw, &g).map(drop), "bwd u32 weight");
    dtype(be.linear_backward(&ux, &w, &g).map(drop), "bwd u32 input");
    dtype(be.linear_backward(&x, &w, &ug).map(drop), "bwd u32 grad");
    assert_eq!(be.budget().live_bytes().unwrap(), 0);
}

#[derive(Debug)]
struct FakeDevice(usize);

impl DeviceBuffer for FakeDevice {
    fn backend(&self) -> BackendId {
        BackendId::Metal
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

/// Device-placed operands are `Placement` with nothing charged and no
/// device readback.
#[test]
fn device_operands_are_placement_errors_before_any_charge() {
    let be = CpuBackend::with_threads(Budget::new(0), 7).unwrap();
    let ib = inputs_budget();
    let dev = |shape: &[usize]| {
        let n: usize = shape.iter().product();
        Tensor::from_device(Arc::new(FakeDevice(n * 4)), shape, DType::F32, &ib).unwrap()
    };
    let x = t(&[1.0; 12], &[3, 4]);
    let w = t(&[1.0; 8], &[2, 4]);
    let g = t(&[1.0; 6], &[3, 2]);
    let before = (be.budget().device_readbacks(), ib.device_readbacks());
    let placed = |r: Result<(), OjasError>, what: &str| match r {
        Err(OjasError::Placement {
            found: Some(BackendId::Metal),
            ..
        }) => {}
        other => panic!("{what}: expected Placement, got {other:?}"),
    };
    placed(
        be.linear_forward(&dev(&[3, 4]), &w).map(drop),
        "device input",
    );
    placed(
        be.linear_forward(&x, &dev(&[2, 4])).map(drop),
        "device weight",
    );
    placed(
        be.linear_backward(&x, &w, &dev(&[3, 2])).map(drop),
        "device grad",
    );
    placed(
        be.linear_backward(&x, &dev(&[2, 4]), &g).map(drop),
        "bwd device weight",
    );
    placed(
        be.linear_backward(&dev(&[3, 4]), &w, &g).map(drop),
        "bwd device input",
    );
    assert_eq!(
        (be.budget().device_readbacks(), ib.device_readbacks()),
        before,
        "no implicit readback into the backend or input budget"
    );
    assert_eq!(be.budget().live_bytes().unwrap(), 0);
}
