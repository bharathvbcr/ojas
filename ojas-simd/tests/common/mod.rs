//! Shared test helpers: a seeded RNG, strided operand builders and an f64
//! reference with the `γ_k` forward-error bound.

#![allow(dead_code)]

use ojas_simd::Backend;

/// splitmix64; deterministic for a given seed.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Uniform in `[-1, 1)`.
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    pub fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

/// How an operand is laid out in its buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    RowMajor,
    /// Column-major, i.e. the transpose stored row-major.
    ColMajor,
    /// Row-major with `pad` unused elements after each row.
    PaddedRow(usize),
    /// Column-major with `pad` unused elements after each column.
    PaddedCol(usize),
    /// Neither stride is 1: `cs = 2`, `rs = 2*cols + 3`.
    Strided,
    /// Row stride 0: every row is the same row.
    BroadcastRows,
}

pub const KINDS: [Kind; 6] = [
    Kind::RowMajor,
    Kind::ColMajor,
    Kind::PaddedRow(3),
    Kind::PaddedCol(5),
    Kind::Strided,
    Kind::BroadcastRows,
];

/// A strided operand. Unaddressed slots hold NaN, so a kernel that reads the
/// wrong slot produces NaN and fails the reference check.
#[derive(Debug, Clone)]
pub struct Operand {
    pub buf: Vec<f32>,
    pub rs: usize,
    pub cs: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Operand {
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.buf[r * self.rs + c * self.cs]
    }
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.buf[r * self.rs + c * self.cs] = v;
    }
}

pub fn operand(rng: &mut Rng, rows: usize, cols: usize, kind: Kind) -> Operand {
    let (rs, cs) = match kind {
        Kind::RowMajor => (cols, 1),
        Kind::ColMajor => (1, rows),
        Kind::PaddedRow(p) => (cols + p, 1),
        Kind::PaddedCol(p) => (1, rows + p),
        Kind::Strided => (2 * cols + 3, 2),
        Kind::BroadcastRows => (0, 1),
    };
    let len = if rows == 0 || cols == 0 {
        0
    } else {
        (rows - 1) * rs + (cols - 1) * cs + 1
    };
    let mut op = Operand {
        buf: vec![f32::NAN; len],
        rs,
        cs,
        rows,
        cols,
    };
    for r in 0..rows {
        for c in 0..cols {
            op.set(r, c, rng.unit());
        }
    }
    op
}

/// A `m × n` output with row stride `n + pad`. Valid slots are random and
/// padding slots hold a sentinel that must never change.
pub const SENTINEL: f32 = -12345.678;

pub fn output(rng: &mut Rng, m: usize, n: usize, pad: usize) -> (Vec<f32>, usize) {
    let c_rs = n + pad;
    let len = if m == 0 || n == 0 {
        0
    } else {
        (m - 1) * c_rs + n
    };
    let mut c = vec![SENTINEL; len];
    for i in 0..m {
        for j in 0..n {
            c[i * c_rs + j] = rng.unit();
        }
    }
    (c, c_rs)
}

pub fn available_backends() -> Vec<Backend> {
    Backend::ALL
        .into_iter()
        .filter(|b| b.is_available())
        .collect()
}

pub struct Case<'a> {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a: &'a Operand,
    pub b: &'a Operand,
    pub c0: &'a [f32],
    pub c_rs: usize,
    pub accumulate: bool,
}

impl Case<'_> {
    pub fn run(&self, backend: Backend) -> Vec<f32> {
        let mut c = self.c0.to_vec();
        ojas_simd::sgemm_tile_with(
            backend,
            self.m,
            self.n,
            self.k,
            &self.a.buf,
            self.a.rs,
            self.a.cs,
            &self.b.buf,
            self.b.rs,
            self.b.cs,
            &mut c,
            self.c_rs,
            self.accumulate,
        )
        .unwrap_or_else(|e| panic!("{backend:?} refused a valid call: {e}"));
        c
    }

    /// Relative tolerance: `|got - ref| <= TOL_FACTOR * (k+1) * 2^-24 *
    /// (Σ_p |A[i,p]·B[p,j]| + |C0|)`, the forward-error bound `γ_{k+1}` of an
    /// `f32` FMA chain against an f64 reference, scaled by `factor`.
    pub fn check(&self, got: &[f32], factor: f64, what: &str) {
        let u = 2f64.powi(-24);
        assert_eq!(got.len(), self.c0.len());
        for i in 0..self.m {
            for j in 0..self.n {
                let idx = i * self.c_rs + j;
                let c0 = if self.accumulate {
                    self.c0[idx] as f64
                } else {
                    0.0
                };
                let mut sum = c0;
                let mut mag = c0.abs();
                for p in 0..self.k {
                    let t = self.a.get(i, p) as f64 * self.b.get(p, j) as f64;
                    sum += t;
                    mag += t.abs();
                }
                let g = got[idx] as f64;
                if sum.is_nan() {
                    assert!(g.is_nan(), "{what}: C[{i},{j}] = {g}, expected NaN");
                } else if sum.is_infinite() {
                    assert_eq!(g, sum, "{what}: C[{i},{j}]");
                } else {
                    let tol = factor * (self.k as f64 + 1.0) * u * mag + 1e-38;
                    assert!(
                        (g - sum).abs() <= tol,
                        "{what}: C[{i},{j}] = {g}, reference {sum}, |err| {} > tol {tol} \
                         (m={} n={} k={} acc={})",
                        (g - sum).abs(),
                        self.m,
                        self.n,
                        self.k,
                        self.accumulate
                    );
                }
            }
        }
        // Slots between rows of C are never written.
        for (idx, (&g, &o)) in got.iter().zip(self.c0).enumerate() {
            if self.n > 0 && idx % self.c_rs >= self.n {
                assert_eq!(
                    g.to_bits(),
                    o.to_bits(),
                    "{what}: padding slot {idx} written"
                );
            }
        }
    }

    /// Runs every available backend, checks each against the reference and
    /// checks that they agree bit for bit. Returns the shared result.
    pub fn run_all(&self, what: &str) -> Vec<f32> {
        let backends = available_backends();
        let first = self.run(backends[0]);
        self.check(&first, 1.0, &format!("{what} [{:?}]", backends[0]));
        for &bk in &backends[1..] {
            let got = self.run(bk);
            self.check(&got, 1.0, &format!("{what} [{bk:?}]"));
            assert_bits_eq(
                &first,
                &got,
                &format!("{what}: {:?} vs {bk:?}", backends[0]),
            );
        }
        first
    }
}

/// Bit equality, treating any two NaNs as equal (payloads are not specified).
pub fn assert_bits_eq(x: &[f32], y: &[f32], what: &str) {
    assert_eq!(x.len(), y.len(), "{what}: length");
    for (i, (a, b)) in x.iter().zip(y).enumerate() {
        let same = a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        assert!(
            same,
            "{what}: slot {i} differs: {a:e} ({:#x}) vs {b:e} ({:#x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}
