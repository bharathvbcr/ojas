//! The arithmetic a `Numerics::Fast` linear product runs, by build target and
//! size, pinned through the public ops (`linear_forward` and both products of
//! `linear_backward`) at every thread count.
//!
//! `FAST_WHOLE_CALL_MACS` is the product size (`m * k * n` multiply-adds)
//! where a Fast product leaves the packed kernel for one whole `ojas-simd`
//! call. Each output is one reduction in ascending index order, and which
//! reduction depends on the target:
//!
//! | Fast product | aarch64 | x86_64 built with `fma` | default x86_64 build |
//! | :--- | :--- | :--- | :--- |
//! | below the cutoff | FMA chain | FMA chain | `acc + a * b`, the Exact chain |
//! | at or above, off macOS | FMA chain | FMA chain | FMA chain |
//! | at or above, macOS | Accelerate (unspecified order) | same | same |
//!
//! The packed kernel, `gemv_into`, the lane dots and the vector exponential
//! reach their multiply-add through one `fma` that is a fused multiply-add
//! only where the build enables it (`target_arch = "aarch64"` or
//! `target_feature = "fma"`): without it `mul_add` would be a libm call per
//! lane, so a default x86_64 build keeps a multiply and an add. The whole
//! `ojas-simd` call detects AVX2 and FMA at run time and, where the CPU has no
//! FMA, runs a software `fmaf` with the same bits. So on a default x86_64 build
//! a Fast product changes arithmetic at the cutoff, from the unfused chain
//! below it to the fused chain above it, and a Fast product's bits there are
//! not the bits of the same product on aarch64 below the cutoff. Routing the
//! packed kernel, `gemv_into` and the lane dots through the runtime-detected
//! FMA would make the rows agree and move the default x86_64 Fast bits; that
//! is not done, and these tests are what such a change must update.
//!
//! Each test computes the expected chain here, per output, and compares bits
//! (off macOS, and below the cutoff everywhere). Above the cutoff on macOS the
//! order is Accelerate's own, so the product is held to the error bound any
//! summation order meets.

mod common;

use common::{bits, f32t, SplitMix64};
use ojas_core::{Backend, Budget, Numerics};
use ojas_cpu::{CpuBackend, FAST_WHOLE_CALL_MACS};

/// Whether this build's packed kernel, `gemv_into` and lane dots fuse the
/// multiply-add (`ojas-cpu`'s `fma`).
const FMA_COMPILED: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

const THREADS: [usize; 6] = [1, 2, 3, 7, 16, 18];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Chain {
    Fma,
    Plain,
}

/// One ascending reduction from `+0.0`.
fn reduce(chain: Chain, terms: impl Iterator<Item = (f32, f32)>) -> f32 {
    terms.fold(0.0f32, |acc, (a, b)| match chain {
        Chain::Fma => a.mul_add(b, acc),
        Chain::Plain => acc + a * b,
    })
}

/// `x [m, k] · w [n, k]ᵀ`, `g [m, n] · w [n, k]` and `g [m, n]ᵀ · x [m, k]`
/// under one chain, as `linear_forward` and `linear_backward` compute them.
struct Products {
    y: Vec<f32>,
    gx: Vec<f32>,
    gw: Vec<f32>,
}

struct Operands {
    m: usize,
    k: usize,
    n: usize,
    x: Vec<f32>,
    w: Vec<f32>,
    g: Vec<f32>,
}

impl Operands {
    fn new(m: usize, k: usize, n: usize, seed: u64) -> Self {
        let mut rng = SplitMix64(seed);
        Operands {
            m,
            k,
            n,
            x: rng.vec(m * k, 0.5),
            w: rng.vec(n * k, 0.5),
            g: rng.vec(m * n, 0.5),
        }
    }

    fn products(&self, chain: Chain) -> Products {
        let Operands { m, k, n, x, w, g } = self;
        let (m, k, n) = (*m, *k, *n);
        let mut y = Vec::with_capacity(m * n);
        let mut gx = Vec::with_capacity(m * k);
        let mut gw = Vec::with_capacity(n * k);
        for i in 0..m {
            for j in 0..n {
                y.push(reduce(chain, (0..k).map(|p| (x[i * k + p], w[j * k + p]))));
            }
        }
        for i in 0..m {
            for p in 0..k {
                gx.push(reduce(chain, (0..n).map(|j| (g[i * n + j], w[j * k + p]))));
            }
        }
        for j in 0..n {
            for p in 0..k {
                gw.push(reduce(chain, (0..m).map(|i| (g[i * n + j], x[i * k + p]))));
            }
        }
        Products { y, gx, gw }
    }
}

/// Asserts every output of one product is within `(reduction length + 2) *
/// 2^-24 * sum |a * b|` of its `f64` sum, the bound any summation order
/// meets. `terms(output)` are that output's operand pairs.
fn close<I: Iterator<Item = (f32, f32)>>(
    got: &[f32],
    len: usize,
    outputs: usize,
    terms: impl Fn(usize) -> I,
    label: &str,
) {
    assert_eq!(got.len(), outputs, "{label}");
    let unit = f64::from(f32::EPSILON) / 2.0;
    for (at, value) in got.iter().enumerate() {
        let (mut sum, mut size) = (0.0f64, 0.0f64);
        for (a, b) in terms(at) {
            let t = f64::from(a) * f64::from(b);
            sum += t;
            size += t.abs();
        }
        let tol = (len as f64 + 2.0) * unit * size;
        let err = (f64::from(*value) - sum).abs();
        assert!(
            err <= tol,
            "{label}: [{at}] = {value:e}, f64 {sum:e}, tolerance {tol:e}"
        );
    }
}

/// Shapes `(m, k, n)`: one row (the `gemv_into` path), small ones across the
/// register tile and cache blocks, `127x127x129` (below the off-macOS cutoff,
/// above the macOS one), the fewest rows at `k = n = 64` that sit just below
/// and at the cutoff, and one row at the cutoff.
fn shapes() -> Vec<(usize, usize, usize)> {
    let rows = FAST_WHOLE_CALL_MACS.div_ceil(64 * 64);
    vec![
        (1, 300, 17),
        (7, 9, 17),
        (73, 7, 13),
        (127, 127, 129),
        (rows - 1, 64, 64),
        (rows, 64, 64),
        (1, 128, FAST_WHOLE_CALL_MACS.div_ceil(128)),
    ]
}

fn fast(threads: usize) -> CpuBackend {
    let be = CpuBackend::with_threads(Budget::new(1 << 30), threads).unwrap();
    assert_eq!(be.numerics(), Numerics::Fast);
    be
}

fn run(be: &CpuBackend, ops: &Operands) -> Products {
    let Operands { m, k, n, x, w, g } = ops;
    let x = f32t(be, x, &[*m, *k]);
    let w = f32t(be, w, &[*n, *k]);
    let g = f32t(be, g, &[*m, *n]);
    let y = be.linear_forward(&x, &w).unwrap();
    let (gx, gw) = be.linear_backward(&x, &w, &g).unwrap();
    Products {
        y: y.to_f32_vec().unwrap(),
        gx: gx.to_f32_vec().unwrap(),
        gw: gw.to_f32_vec().unwrap(),
    }
}

/// The chain a Fast product of `macs` multiply-adds runs, where the order is
/// fixed: `None` for a macOS whole call, which is Accelerate's.
fn expected_chain(macs: usize) -> Option<Chain> {
    if macs < FAST_WHOLE_CALL_MACS {
        Some(if FMA_COMPILED {
            Chain::Fma
        } else {
            Chain::Plain
        })
    } else if cfg!(target_os = "macos") {
        None
    } else {
        Some(Chain::Fma)
    }
}

#[test]
fn the_cutoff_shapes_sit_on_both_sides_of_it() {
    let (mut below, mut above) = (0, 0);
    for (m, k, n) in shapes() {
        if m * k * n < FAST_WHOLE_CALL_MACS {
            below += 1;
        } else {
            above += 1;
        }
    }
    assert!(below >= 4 && above >= 2, "{below} below, {above} above");
}

/// Every Fast product, at every thread count, is the chain this target runs
/// at its size: bit for bit where the order is fixed.
#[test]
fn fast_products_are_this_targets_chain_at_every_thread_count() {
    for (m, k, n) in shapes() {
        let ops = Operands::new(m, k, n, 0xfa57_f3a0 + (m * 31 + k * 7 + n) as u64);
        let (fma, plain) = (ops.products(Chain::Fma), ops.products(Chain::Plain));
        // The operands tell the two chains apart, or the test would pass
        // whichever one a build ran.
        assert_ne!(bits(&fma.y), bits(&plain.y), "{m}x{k}x{n} y");
        assert_ne!(bits(&fma.gx), bits(&plain.gx), "{m}x{k}x{n} gx");
        let chain = expected_chain(m * k * n);
        for threads in THREADS {
            let got = run(&fast(threads), &ops);
            let label = format!("{m}x{k}x{n} t{threads}");
            match chain {
                Some(chain) => {
                    let want = ops.products(chain);
                    assert_eq!(bits(&got.y), bits(&want.y), "{label} y");
                    assert_eq!(bits(&got.gx), bits(&want.gx), "{label} gx");
                    assert_eq!(bits(&got.gw), bits(&want.gw), "{label} gw");
                }
                None => {
                    let Operands { x, w, g, .. } = &ops;
                    close(
                        &got.y,
                        k,
                        m * n,
                        |at| (0..k).map(move |p| (x[at / n * k + p], w[at % n * k + p])),
                        &format!("{label} y"),
                    );
                    close(
                        &got.gx,
                        n,
                        m * k,
                        |at| (0..n).map(move |j| (g[at / k * n + j], w[j * k + at % k])),
                        &format!("{label} gx"),
                    );
                    close(
                        &got.gw,
                        m,
                        n * k,
                        |at| (0..m).map(move |i| (g[i * n + at / k], x[i * k + at % k])),
                        &format!("{label} gw"),
                    );
                }
            }
        }
    }
}

/// A Fast product below the cutoff changes no bit with the thread count, and
/// Exact is the plain chain everywhere: the two tiers meet exactly where the
/// build does not fuse.
#[test]
fn below_the_cutoff_fast_meets_exact_only_where_the_build_does_not_fuse() {
    let (m, k, n) = (7, 9, 17);
    let ops = Operands::new(m, k, n, 0xfa57_0001);
    let exact = CpuBackend::with_threads(Budget::new(1 << 30), 3)
        .unwrap()
        .with_numerics(Numerics::Exact);
    let want = ops.products(Chain::Plain);
    let got = run(&exact, &ops);
    assert_eq!(bits(&got.y), bits(&want.y));
    assert_eq!(bits(&got.gx), bits(&want.gx));
    assert_eq!(bits(&got.gw), bits(&want.gw));
    let fast_out = run(&fast(3), &ops);
    assert_eq!(
        bits(&fast_out.y) == bits(&got.y),
        !FMA_COMPILED,
        "Fast and Exact agree below the cutoff exactly when the build does not fuse"
    );
}
