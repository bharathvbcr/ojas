//! The crate's one device `exp`, sigmoid and SiLU, with a bit-identical host
//! emulation of each.
//!
//! **Why not CUDA's `expf`.** libdevice's `expf` is documented to 2 ulp and its
//! algorithm is not published, so no host code reproduces its bits, and they
//! may change with the NVRTC version. K8 is specified as exact f32 with a
//! bitwise host reference (`cuda-backend-scoping.md` §3, the K8 row). So `exp`
//! here is built from operations whose IEEE result is fixed: `fmaf`, `rintf`,
//! exact power-of-two scaling, and every other float multiply, add, subtract
//! and divide written as `__fmul_rn` / `__fadd_rn` / `__fsub_rn` /
//! `__fdiv_rn`, which CUDA never contracts into an FMA whatever `--fmad`
//! says. So the prelude's bits do not depend on `--fmad`; they do need
//! `--ftz=false` (subnormal results), which [`crate::kernels::STRICT_SM90`]
//! sets. Rust's `f32` arithmetic,
//! `f32::mul_add` (fused; `host_ref`'s test `mul_add_is_fused_on_this_host`)
//! and `f32::round_ties_even` are the same operations, so [`exp_f32`] and the
//! device `qd_exp` return the same bits for every input. The device tests and
//! `runga` check that over a sweep of inputs.
//!
//! **One copy for the crate.** K3 (`beta = sigmoid(b)`, the softplus `exp`),
//! K4 (the conv's SiLU), K7 (the gated norm's SiLU), K2 (the decay
//! `exp(g)`) and K8 (SwiGLU) all need these. tessl keeps one copy
//! (`tessl/kernels/qwen35_act.h`) so they cannot drift apart; this is that
//! copy here. A kernel module splices it with `concat!(act_prelude!(), ...)`,
//! or prepends [`ACT_PRELUDE`]. It is guarded, so splicing it twice into one
//! module compiles once.
//!
//! # The API (stable; other lanes depend on these names)
//!
//! Every function takes and returns one `float` (`f32`), except `qd_pow2i`.
//!
//! | Device (CUDA-C) | Host (Rust) | What |
//! | --- | --- | --- |
//! | `qd_exp(x)` | [`exp_f32`] | `e^x` for every f32: `+inf` above 88.8, `+0` below -104 |
//! | `qd_exp_nonpos(x)` | [`exp_nonpos_f32`] | `qd_exp(x)` for `x <= 0`; the canonical NaN for `x > 0` (`+inf` included), by the lead's ruling, so a positive argument reaching a decay is loud |
//! | `qd_log(y)` | [`log_f32`] | `ln y`: `-inf` at 0, `+inf` at `+inf`, NaN below 0 |
//! | `qd_softplus(x)` | [`softplus_f32`] | tessl's `qwen35_softplus` (`qwen35_act.h`): `x` above 20, a Horner `log1p(e^x)` below -3, else `log(1 + e^x)` |
//! | `qd_sigmoid(x)` | [`sigmoid_f32`] | tessl's overflow-free `1 / (1 + e^-x)` |
//! | `qd_silu(x)` | [`silu_f32`] | `x * sigmoid(x)` |
//! | `qd_silu_grad(x)` | [`silu_grad_f32`] | `s * (1 + x * (1 - s))`, `s = sigmoid(x)` |
//! | `qd_canon_nan(y)` | [`canon_nan_f32`] | any NaN becomes [`CANONICAL_NAN_BITS`] |
//! | `qd_pow2i(n)` (int) | (private) | `2^n` for `n` in `[-126, 127]`, exactly |
//!
//! The prelude defines one macro, the include guard `QD_ACT_PRELUDE`, and no
//! other `#define`. It does not define `QD_GRID_STRIDE` or anything from
//! [`crate::kernels`]' prelude.
//!
//! `qd_log` (needed by softplus; tessl uses `precise::log`, and libdevice's
//! `logf` is not host-reproducible either): `y = m 2^k` with `m` in
//! `[sqrt(1/2), sqrt(2))` by exact bit manipulation (a subnormal `y` is first
//! scaled by `2^23`), `f = m - 1` (exact, Sterbenz), `s = f / (2 + f)`,
//! `ln m = 2s + 2s z (1/3 + z/5 + z^2/7 + z^3/9 + z^4/11)` with `z = s^2` in
//! fused Horner steps, then `+ k ln2` as `LN2_LO` then `LN2_HI`, each fused.
//!
//! # The algorithm
//!
//! `qd_exp(x)`:
//! 1. NaN gives the canonical NaN; `x > 88.8` gives `+inf`; `x < -104` gives
//!    `+0` (`e^-104 < 2^-150`, half the smallest subnormal).
//! 2. `k = rint(x * log2(e))`, ties to even.
//! 3. `r = x - k ln2` by Cody-Waite in two fused steps, with
//!    `ln2 = LN2_HI + LN2_LO`. So `|r| <= ~0.347`.
//! 4. `e^r` by the degree-7 Taylor polynomial in Horner form, every step an
//!    `fmaf`. The first dropped term is `r^8/8! < 5.3e-9`, under 0.1 ulp.
//! 5. Scale by `2^k` as `(p * 2^k1) * 2^k2`, with `k1 = k / 2` truncated and
//!    `k2 = k - k1`. Both halves lie in `[-75, 64]`, so both factors are normal.
//!    The first product is exact, and the second rounds once, into the
//!    subnormal range or to `inf` where the true value lies there.
//!
//! The measured accuracy against `f64::exp` is in this module's tests.
//!
//! Sigmoid is tessl's form (`qwen35_act.h`): `e = exp(-|x|)`,
//! `r = 1 / (1 + e)`, then `x >= 0 ? r : e * r`. It never forms `e^|x|`.
//!
//! NaN results are canonicalised to `0x7fffffff`, the value PTX arithmetic
//! returns. The host's own NaN bits differ by platform (x86 gives `0xffc00000`
//! for an invalid operation, aarch64 `0x7fc00000`), so without this a NaN
//! output could not be compared bitwise.

/// The NaN every function here returns, as bits.
pub const CANONICAL_NAN_BITS: u32 = 0x7fff_ffff;

/// Above this, `exp` is `+inf` (`88.8f32`).
pub const EXP_HI_BITS: u32 = 0x42b1_999a;
/// Below this, `exp` is `+0` (`-104f32`).
pub const EXP_LO_BITS: u32 = 0xc2d0_0000;
/// `log2(e)` rounded to f32.
pub const LOG2E_BITS: u32 = 0x3fb8_aa3b;
/// The high part of `ln 2`: `0.693145751953125`, exact with 17 bits.
pub const LN2_HI_BITS: u32 = 0x3f31_7200;
/// `ln 2 - LN2_HI` rounded to f32.
pub const LN2_LO_BITS: u32 = 0x35bf_be8e;
/// `1/n!` rounded to f32, `n = 0..=7` (Horner runs from `n = 7` down).
pub const TAYLOR_BITS: [u32; 8] = [
    0x3f80_0000,
    0x3f80_0000,
    0x3f00_0000,
    0x3e2a_aaab,
    0x3d2a_aaab,
    0x3c08_8889,
    0x3ab6_0b61,
    0x3950_0d01,
];
/// `sqrt(2)` rounded to f32: `ln`'s mantissa is reduced below it.
pub const SQRT2_BITS: u32 = 0x3fb5_04f3;
/// `1/(2j+1)` rounded to f32, `j = 1..=5`: `ln`'s `atanh` series.
pub const ATANH_BITS: [u32; 5] = [
    0x3eaa_aaab,
    0x3e4c_cccd,
    0x3e12_4925,
    0x3de3_8e39,
    0x3dba_2e8c,
];
/// tessl's softplus Horner coefficients, `p` from `-1/8` down to `1`
/// (`qwen35_act.h`): `-1/8, 1/7, -1/6, 1/5, -1/4, 1/3, -1/2, 1`.
pub const SOFTPLUS_BITS: [u32; 8] = [
    0xbe00_0000,
    0x3e12_4925,
    0xbe2a_aaab,
    0x3e4c_cccd,
    0xbe80_0000,
    0x3eaa_aaab,
    0xbf00_0000,
    0x3f80_0000,
];
/// Softplus is the identity above this (`20f32`, torch's threshold).
pub const SOFTPLUS_HI_BITS: u32 = 0x41a0_0000;
/// Below this softplus uses the series (`-3f32`, tessl's switch).
pub const SOFTPLUS_SERIES_BITS: u32 = 0xc040_0000;

/// Every hex literal the device text contains, for the test that keeps the
/// two sides' constants equal: the constants above, plus `+inf`, `-inf`,
/// the smallest normal, `2^23`, `2`, and the two bit masks `ln` uses.
pub const DEVICE_LITERALS: [u32; 7] = [
    0x7f80_0000,
    0xff80_0000,
    0x0080_0000,
    0x4b00_0000,
    0x4000_0000,
    0x007f_ffff,
    0xff,
];

/// The device source, as a literal for `concat!`. The constants are written
/// as bit patterns, so the device parses nothing: `ACT_CONSTANTS_MATCH_HOST`
/// in this module's tests reads them back out of this text.
#[macro_export]
macro_rules! act_prelude {
    () => {
        r#"
#ifndef QD_ACT_PRELUDE
#define QD_ACT_PRELUDE
// ojas-qwen35-cuda src/k8_act.rs: the crate's one exp / sigmoid / SiLU.
// Bit-identical to the host functions there; see that module for why.

__device__ __forceinline__ float qd_canon_nan(float y) {
    return (y != y) ? __uint_as_float(0x7fffffffu) : y;
}

// 2^n for n in [-126, 127], exactly.
__device__ __forceinline__ float qd_pow2i(int n) {
    return __uint_as_float((unsigned int)(n + 127) << 23);
}

__device__ __forceinline__ float qd_exp(float x) {
    if (x != x) return __uint_as_float(0x7fffffffu);
    if (x > __uint_as_float(0x42b1999au)) return __uint_as_float(0x7f800000u);
    if (x < __uint_as_float(0xc2d00000u)) return 0.0f;
    const float kf = rintf(__fmul_rn(x, __uint_as_float(0x3fb8aa3bu)));
    float r = fmaf(-kf, __uint_as_float(0x3f317200u), x);
    r = fmaf(-kf, __uint_as_float(0x35bfbe8eu), r);
    float p = __uint_as_float(0x39500d01u);
    p = fmaf(p, r, __uint_as_float(0x3ab60b61u));
    p = fmaf(p, r, __uint_as_float(0x3c088889u));
    p = fmaf(p, r, __uint_as_float(0x3d2aaaabu));
    p = fmaf(p, r, __uint_as_float(0x3e2aaaabu));
    p = fmaf(p, r, __uint_as_float(0x3f000000u));
    p = fmaf(p, r, __uint_as_float(0x3f800000u));
    p = fmaf(p, r, __uint_as_float(0x3f800000u));
    const int k = (int)kf;
    const int k1 = k / 2;
    const int k2 = k - k1;
    return __fmul_rn(__fmul_rn(p, qd_pow2i(k1)), qd_pow2i(k2));
}

// For x <= 0 only: a positive argument (+inf included) is the canonical NaN.
__device__ __forceinline__ float qd_exp_nonpos(float x) {
    if (x > 0.0f) return __uint_as_float(0x7fffffffu);
    return qd_exp(x);
}

__device__ __forceinline__ float qd_log(float y) {
    if (y != y || y < 0.0f) return __uint_as_float(0x7fffffffu);
    if (y == 0.0f) return __uint_as_float(0xff800000u);
    if (y == __uint_as_float(0x7f800000u)) return y;
    int k = 0;
    if (y < __uint_as_float(0x00800000u)) {
        y = __fmul_rn(y, __uint_as_float(0x4b000000u));
        k = -23;
    }
    const unsigned int bits = __float_as_uint(y);
    k += (int)((bits >> 23) & 0xffu) - 127;
    float m = __uint_as_float((bits & 0x007fffffu) | 0x3f800000u);
    if (m > __uint_as_float(0x3fb504f3u)) {
        m = __fmul_rn(m, __uint_as_float(0x3f000000u));
        k += 1;
    }
    const float f = __fsub_rn(m, 1.0f);
    const float s = __fdiv_rn(f, __fadd_rn(__uint_as_float(0x40000000u), f));
    const float z = __fmul_rn(s, s);
    float q = __uint_as_float(0x3dba2e8cu);
    q = fmaf(q, z, __uint_as_float(0x3de38e39u));
    q = fmaf(q, z, __uint_as_float(0x3e124925u));
    q = fmaf(q, z, __uint_as_float(0x3e4ccccdu));
    q = fmaf(q, z, __uint_as_float(0x3eaaaaabu));
    const float s2 = __fmul_rn(s, __uint_as_float(0x40000000u));
    const float lm = fmaf(__fmul_rn(s2, z), q, s2);
    const float kf = (float)k;
    return fmaf(kf, __uint_as_float(0x3f317200u), fmaf(kf, __uint_as_float(0x35bfbe8eu), lm));
}

// tessl/kernels/qwen35_act.h qwen35_softplus: torch's F.softplus at beta 1,
// threshold 20. Below -3, log1p(e) by Horner, e * (1 - e/2 + ... - e^7/8),
// each step an unfused p * e + c as tessl writes it.
__device__ __forceinline__ float qd_softplus(float x) {
    if (x > __uint_as_float(0x41a00000u)) return x;
    const float e = qd_exp(x);
    if (x < __uint_as_float(0xc0400000u)) {
        float p = __uint_as_float(0xbe000000u);
        p = __fadd_rn(__fmul_rn(p, e), __uint_as_float(0x3e124925u));
        p = __fadd_rn(__fmul_rn(p, e), __uint_as_float(0xbe2aaaabu));
        p = __fadd_rn(__fmul_rn(p, e), __uint_as_float(0x3e4ccccdu));
        p = __fadd_rn(__fmul_rn(p, e), __uint_as_float(0xbe800000u));
        p = __fadd_rn(__fmul_rn(p, e), __uint_as_float(0x3eaaaaabu));
        p = __fadd_rn(__fmul_rn(p, e), __uint_as_float(0xbf000000u));
        p = __fadd_rn(__fmul_rn(p, e), __uint_as_float(0x3f800000u));
        return __fmul_rn(e, p);
    }
    return qd_log(__fadd_rn(1.0f, e));
}

// tessl/kernels/qwen35_act.h: never forms e^|x|.
__device__ __forceinline__ float qd_sigmoid(float x) {
    if (x != x) return __uint_as_float(0x7fffffffu);
    const float e = qd_exp(-fabsf(x));
    const float r = __fdiv_rn(1.0f, __fadd_rn(1.0f, e));
    return (x >= 0.0f) ? r : __fmul_rn(e, r);
}

__device__ __forceinline__ float qd_silu(float x) {
    return qd_canon_nan(__fmul_rn(x, qd_sigmoid(x)));
}

// tessl/kernels/qwen35_bwd.metal:31-35.
__device__ __forceinline__ float qd_silu_grad(float x) {
    const float s = qd_sigmoid(x);
    return qd_canon_nan(__fmul_rn(s, __fadd_rn(1.0f, __fmul_rn(x, __fsub_rn(1.0f, s)))));
}
#endif
"#
    };
}

/// [`act_prelude!`] as a string.
pub const ACT_PRELUDE: &str = act_prelude!();

/// Any NaN as [`CANONICAL_NAN_BITS`]; every other value unchanged.
pub fn canon_nan_f32(y: f32) -> f32 {
    if y.is_nan() {
        f32::from_bits(CANONICAL_NAN_BITS)
    } else {
        y
    }
}

/// `2^n`, exactly, for `n` in `[-126, 127]`.
fn pow2i(n: i32) -> f32 {
    debug_assert!((-126..=127).contains(&n), "pow2i({n})");
    // n + 127 is in [1, 254] for the documented range, so the cast is exact.
    f32::from_bits(((n + 127) as u32) << 23)
}

/// `e^x`, bit-identical to the device `qd_exp`.
pub fn exp_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits(CANONICAL_NAN_BITS);
    }
    if x > f32::from_bits(EXP_HI_BITS) {
        return f32::INFINITY;
    }
    if x < f32::from_bits(EXP_LO_BITS) {
        return 0.0;
    }
    let kf = (x * f32::from_bits(LOG2E_BITS)).round_ties_even();
    let r = (-kf).mul_add(f32::from_bits(LN2_HI_BITS), x);
    let r = (-kf).mul_add(f32::from_bits(LN2_LO_BITS), r);
    let mut p = f32::from_bits(TAYLOR_BITS[7]);
    for &c in TAYLOR_BITS[..7].iter().rev() {
        p = p.mul_add(r, f32::from_bits(c));
    }
    // kf is an integer in [-150, 128] here: x in [-104, 88.8] times log2(e).
    let k = kf as i32;
    let k1 = k / 2;
    let k2 = k - k1;
    (p * pow2i(k1)) * pow2i(k2)
}

/// [`exp_f32`] for `x <= 0`; the canonical NaN for `x > 0`, `+inf` included
/// (device `qd_exp_nonpos`). `-0.0` is not positive: it gives 1.
pub fn exp_nonpos_f32(x: f32) -> f32 {
    if x > 0.0 {
        return f32::from_bits(CANONICAL_NAN_BITS);
    }
    exp_f32(x)
}

/// `ln y`, bit-identical to the device `qd_log`: NaN for NaN or `y < 0`,
/// `-inf` at `+-0`, `+inf` at `+inf`.
pub fn log_f32(y: f32) -> f32 {
    if y.is_nan() || y < 0.0 {
        return f32::from_bits(CANONICAL_NAN_BITS);
    }
    if y == 0.0 {
        return f32::NEG_INFINITY;
    }
    if y == f32::INFINITY {
        return y;
    }
    let mut y = y;
    let mut k: i32 = 0;
    if y < f32::MIN_POSITIVE {
        y *= f32::from_bits(0x4b00_0000);
        k = -23;
    }
    let bits = y.to_bits();
    // The biased exponent is 1..=254 here: y is a positive normal.
    k += ((bits >> 23) & 0xff) as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if m > f32::from_bits(SQRT2_BITS) {
        m *= 0.5;
        k += 1;
    }
    let f = m - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let mut q = f32::from_bits(ATANH_BITS[4]);
    for &c in ATANH_BITS[..4].iter().rev() {
        q = q.mul_add(z, f32::from_bits(c));
    }
    let s2 = s * 2.0;
    let lm = (s2 * z).mul_add(q, s2);
    // k is in [-149, 128]: exact as f32.
    let kf = k as f32;
    kf.mul_add(
        f32::from_bits(LN2_HI_BITS),
        kf.mul_add(f32::from_bits(LN2_LO_BITS), lm),
    )
}

/// tessl's `qwen35_softplus` (`kernels/qwen35_act.h`), bit-identical to the
/// device `qd_softplus`: `x` above 20; below -3, `e * p` with `p` tessl's
/// Horner series in unfused steps; else `ln(1 + e)`, `e = e^x`.
pub fn softplus_f32(x: f32) -> f32 {
    if x > f32::from_bits(SOFTPLUS_HI_BITS) {
        return x;
    }
    let e = exp_f32(x);
    if x < f32::from_bits(SOFTPLUS_SERIES_BITS) {
        let mut p = f32::from_bits(SOFTPLUS_BITS[0]);
        for &c in &SOFTPLUS_BITS[1..] {
            p = p * e + f32::from_bits(c);
        }
        return e * p;
    }
    log_f32(1.0 + e)
}

/// `1 / (1 + e^-x)`, tessl's overflow-free form, bit-identical to
/// `qd_sigmoid`.
pub fn sigmoid_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::from_bits(CANONICAL_NAN_BITS);
    }
    let e = exp_f32(-x.abs());
    let r = 1.0 / (1.0 + e);
    if x >= 0.0 {
        r
    } else {
        e * r
    }
}

/// `x * sigmoid(x)`, bit-identical to `qd_silu`.
pub fn silu_f32(x: f32) -> f32 {
    canon_nan_f32(x * sigmoid_f32(x))
}

/// `s * (1 + x * (1 - s))` with `s = sigmoid(x)`: SiLU's derivative,
/// bit-identical to `qd_silu_grad`.
pub fn silu_grad_f32(x: f32) -> f32 {
    let s = sigmoid_f32(x);
    canon_nan_f32(s * (1.0 + x * (1.0 - s)))
}

/// Inputs every bitwise device check of these functions sweeps: the special
/// values, both range edges and their neighbours, the subnormal-output
/// region, and `n` values spread over `[-110, 95]` by a fixed splitmix
/// stream ([`crate::inputs::splitmix_bits`]).
pub fn act_sweep_inputs(n: usize) -> Vec<f32> {
    let edges = [
        0.0f32,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0xffc0_0000),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::MAX,
        f32::MIN,
        f32::from_bits(EXP_HI_BITS),
        f32::from_bits(EXP_HI_BITS + 1),
        f32::from_bits(EXP_HI_BITS - 1),
        f32::from_bits(EXP_LO_BITS),
        f32::from_bits(EXP_LO_BITS + 1),
        f32::from_bits(EXP_LO_BITS - 1),
        88.72283,
        88.72284,
        -87.33654,
        -87.33655,
        -103.27893,
        -103.97208,
        // +-ln(2)/2: the reduced argument's range edge.
        -std::f32::consts::LN_2 / 2.0,
        std::f32::consts::LN_2 / 2.0,
        1.0,
        -1.0,
        120.0,
        -120.0,
        -89.0,
        20.0,
        -20.0,
    ];
    let mut out: Vec<f32> = edges.to_vec();
    out.extend(
        crate::inputs::splitmix_bits(0xac7, n)
            .into_iter()
            .map(|b| -110.0 + 205.0 * (f64::from(b >> 8) / f64::from(1u32 << 24)) as f32),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `|got - want|` in units of the f32 spacing at `want`.
    fn ulps(got: f32, want: f64) -> f64 {
        let w = want as f32;
        let spacing = if w == 0.0 || !w.is_normal() {
            f64::from(f32::from_bits(1)) // the subnormal spacing
        } else {
            let next = f32::from_bits(w.abs().to_bits() + 1);
            f64::from(next) - f64::from(w.abs())
        };
        (f64::from(got) - want).abs() / spacing
    }

    #[test]
    fn act_constants_match_host() {
        // Every hex constant the device text names is one of the host's, and
        // every host constant appears in the text.
        let text = ACT_PRELUDE;
        let mut host: Vec<u32> = vec![
            CANONICAL_NAN_BITS,
            EXP_HI_BITS,
            EXP_LO_BITS,
            LOG2E_BITS,
            LN2_HI_BITS,
            LN2_LO_BITS,
            SQRT2_BITS,
            SOFTPLUS_HI_BITS,
            SOFTPLUS_SERIES_BITS,
        ];
        host.extend(TAYLOR_BITS);
        host.extend(ATANH_BITS);
        host.extend(SOFTPLUS_BITS);
        host.extend(DEVICE_LITERALS);
        for bits in &host {
            let (padded, short) = (format!("0x{bits:08x}u"), format!("0x{bits:x}u"));
            assert!(
                text.contains(&padded) || text.contains(&short),
                "{padded} missing from the device prelude"
            );
        }
        for (at, _) in text.match_indices("0x") {
            let hex: String = text[at + 2..]
                .chars()
                .take_while(char::is_ascii_hexdigit)
                .collect();
            let v = u32::from_str_radix(&hex, 16).expect("hex constant");
            assert!(
                host.contains(&v),
                "device constant 0x{hex} has no host twin"
            );
        }
    }

    #[test]
    fn the_constants_are_what_they_claim() {
        assert_eq!(f32::from_bits(LOG2E_BITS), std::f64::consts::LOG2_E as f32);
        let hi = f64::from(f32::from_bits(LN2_HI_BITS));
        let lo = f64::from(f32::from_bits(LN2_LO_BITS));
        assert_eq!(hi, 0.693_145_751_953_125);
        assert_eq!(lo as f32, (std::f64::consts::LN_2 - hi) as f32);
        let mut fact = 1.0f64;
        for (n, &bits) in TAYLOR_BITS.iter().enumerate() {
            if n > 0 {
                fact *= n as f64;
            }
            assert_eq!(f32::from_bits(bits), (1.0 / fact) as f32, "1/{n}!");
        }
        for (j, &bits) in ATANH_BITS.iter().enumerate() {
            let d = 2.0 * (j as f64 + 1.0) + 1.0;
            assert_eq!(f32::from_bits(bits), (1.0 / d) as f32, "1/{d}");
        }
        let series = [-8.0f64, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0];
        for (&bits, &d) in SOFTPLUS_BITS.iter().zip(&series) {
            assert_eq!(f32::from_bits(bits), (1.0 / d) as f32, "1/{d}");
            // tessl writes them as f32 divisions; the same values.
            assert_eq!(f32::from_bits(bits), 1.0f32 / d as f32, "1/{d} in f32");
        }
        assert_eq!(f32::from_bits(SQRT2_BITS), std::f64::consts::SQRT_2 as f32);
        assert_eq!(f32::from_bits(SOFTPLUS_HI_BITS), 20.0);
        assert_eq!(f32::from_bits(SOFTPLUS_SERIES_BITS), -3.0);
        assert_eq!(f32::from_bits(EXP_HI_BITS), 88.8);
        assert_eq!(f32::from_bits(EXP_LO_BITS), -104.0);
        // e^-104 is below half the smallest subnormal, so 0 is its rounding.
        assert!((-104.0f64).exp() < f64::from(f32::from_bits(1)) / 2.0);
    }

    #[test]
    fn exp_is_within_two_ulp_of_f64_over_its_whole_range() {
        let mut worst = (0.0f64, 0.0f32);
        // Every 97th f32 from -104 to 88.8, by bit pattern.
        let mut checked = 0usize;
        for bits in (0u32..=u32::MAX).step_by(97) {
            let x = f32::from_bits(bits);
            if !(x.is_finite()
                && x >= f32::from_bits(EXP_LO_BITS)
                && x <= f32::from_bits(EXP_HI_BITS))
            {
                continue;
            }
            let want = f64::from(x).exp();
            if want > f64::from(f32::MAX) {
                assert_eq!(exp_f32(x), f32::INFINITY, "x = {x:e}");
                continue;
            }
            let u = ulps(exp_f32(x), want);
            checked += 1;
            if u > worst.0 {
                worst = (u, x);
            }
        }
        eprintln!(
            "exp_f32: worst {:.3} ulp at x = {:e} over {checked} inputs",
            worst.0, worst.1
        );
        assert!(checked > 10_000_000, "only {checked} inputs checked");
        assert!(worst.0 <= 2.0, "worst {} ulp at {:e}", worst.0, worst.1);
    }

    #[test]
    fn exp_special_values_and_range_edges() {
        assert_eq!(exp_f32(f32::NAN).to_bits(), CANONICAL_NAN_BITS);
        assert_eq!(
            exp_f32(f32::from_bits(0xffc0_0001)).to_bits(),
            CANONICAL_NAN_BITS
        );
        assert_eq!(exp_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp_f32(f32::NEG_INFINITY).to_bits(), 0);
        assert_eq!(exp_f32(0.0), 1.0);
        assert_eq!(exp_f32(-0.0), 1.0);
        assert_eq!(exp_f32(1.0), std::f32::consts::E);
        assert_eq!(exp_f32(-104.5).to_bits(), 0);
        assert_eq!(exp_f32(89.0), f32::INFINITY);
        // The subnormal range is reached, not flushed.
        let sub = exp_f32(-100.0);
        assert!(sub > 0.0 && !sub.is_normal(), "{sub:e}");
        assert!((f64::from(sub) - (-100.0f64).exp()).abs() <= f64::from(f32::from_bits(1)));
        // The largest finite result and the overflow edge.
        assert!(exp_f32(88.72283).is_finite());
        assert_eq!(exp_f32(88.72284), f32::INFINITY);
    }

    #[test]
    fn exp_nonpos_is_exp_at_or_below_zero_and_nan_above() {
        for x in act_sweep_inputs(100_000) {
            let got = exp_nonpos_f32(x);
            if x > 0.0 {
                assert_eq!(got.to_bits(), CANONICAL_NAN_BITS, "x = {x:e}");
            } else {
                assert_eq!(got.to_bits(), exp_f32(x).to_bits(), "x = {x:e}");
            }
        }
        assert_eq!(exp_nonpos_f32(f32::INFINITY).to_bits(), CANONICAL_NAN_BITS);
        assert_eq!(
            exp_nonpos_f32(f32::from_bits(1)).to_bits(),
            CANONICAL_NAN_BITS
        );
        assert_eq!(exp_nonpos_f32(0.0), 1.0);
        assert_eq!(exp_nonpos_f32(-0.0), 1.0);
        assert_eq!(exp_nonpos_f32(f32::NEG_INFINITY).to_bits(), 0);
        assert_eq!(exp_nonpos_f32(f32::NAN).to_bits(), CANONICAL_NAN_BITS);
    }

    #[test]
    fn exp_on_the_nonpositive_range_measured_for_the_gdn_decay() {
        // Every f32 in [-104, 0], by bit pattern: the domain of exp(g), g <= 0.
        let mut worst = (0.0f64, 0.0f32);
        let (lo, hi) = (0x8000_0000u32, EXP_LO_BITS);
        let mut n = 0usize;
        for bits in (lo..=hi).step_by(7) {
            let x = f32::from_bits(bits);
            let u = ulps(exp_nonpos_f32(x), f64::from(x).exp());
            n += 1;
            if u > worst.0 {
                worst = (u, x);
            }
        }
        eprintln!(
            "exp_nonpos_f32 on [-104, 0]: worst {:.3} ulp at x = {:e} over {n} inputs (every 7th f32)",
            worst.0, worst.1
        );
        assert!(worst.0 <= 2.0, "worst {} ulp at {:e}", worst.0, worst.1);
    }

    #[test]
    fn log_is_within_two_ulp_of_f64_and_handles_the_edges() {
        let mut worst = (0.0f64, 0.0f32);
        let mut n = 0usize;
        // Every 101st positive finite f32, subnormals included.
        for bits in (1u32..0x7f80_0000).step_by(101) {
            let y = f32::from_bits(bits);
            let want = f64::from(y).ln();
            let u = ulps(log_f32(y), want);
            n += 1;
            if u > worst.0 {
                worst = (u, y);
            }
        }
        eprintln!(
            "log_f32: worst {:.3} ulp at y = {:e} over {n} inputs",
            worst.0, worst.1
        );
        assert!(worst.0 <= 2.0, "worst {} ulp at {:e}", worst.0, worst.1);
        assert_eq!(log_f32(1.0).to_bits(), 0);
        assert_eq!(log_f32(0.0), f32::NEG_INFINITY);
        assert_eq!(log_f32(-0.0), f32::NEG_INFINITY);
        assert_eq!(log_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(log_f32(-1.0).to_bits(), CANONICAL_NAN_BITS);
        assert_eq!(log_f32(f32::NAN).to_bits(), CANONICAL_NAN_BITS);
        assert_eq!(log_f32(std::f32::consts::E), 1.0);
    }

    #[test]
    fn softplus_is_tessls_and_within_bounds_of_f64() {
        // torch's softplus in f64: x above 20, else log1p(exp(x)).
        let want = |x: f64| if x > 20.0 { x } else { x.exp().ln_1p() };
        let mut worst = 0.0f64;
        let mut at = 0.0f32;
        for x in act_sweep_inputs(300_000) {
            if !x.is_finite() {
                continue;
            }
            let w = want(f64::from(x));
            // Relative to |w|, or to f32's smallest normal where w is below it:
            // there f32 keeps an absolute 2^-149 grid, not a relative one, and
            // below -103.97 the true value rounds to 0.
            let floor = f64::from(f32::MIN_POSITIVE);
            let e = (f64::from(softplus_f32(x)) - w).abs() / w.abs().max(floor);
            if e > worst {
                worst = e;
                at = x;
            }
        }
        eprintln!("softplus_f32: worst {worst:.3e} relative at x = {at:e}");
        // tessl's own bound for the form (qwen35_act.h): 1.2e-6 relative from
        // rounding 1 + e above -3, half an ulp of log; the series is exact to
        // f32 below -3. Asserted at 2e-6 to admit this exp's and log's ulps.
        assert!(worst < 2e-6, "{worst:e} at {at:e}");
        assert_eq!(softplus_f32(25.0), 25.0);
        assert_eq!(softplus_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(softplus_f32(f32::NEG_INFINITY).to_bits(), 0);
        assert_eq!(softplus_f32(f32::NAN).to_bits(), CANONICAL_NAN_BITS);
        assert!(softplus_f32(-120.0) >= 0.0);
    }

    #[test]
    fn sigmoid_silu_and_its_gradient_against_f64() {
        let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
        let mut worst = [0.0f64; 3];
        for x in act_sweep_inputs(200_000) {
            if !x.is_finite() {
                continue;
            }
            let xd = f64::from(x);
            let s = sig(xd);
            let pairs = [
                (sigmoid_f32(x), s),
                (silu_f32(x), xd * s),
                (silu_grad_f32(x), s * (1.0 + xd * (1.0 - s))),
            ];
            for (k, (got, want)) in pairs.into_iter().enumerate() {
                // Relative to max(|want|, 1): the outputs are O(1) or O(x).
                let e = (f64::from(got) - want).abs() / want.abs().max(1.0);
                worst[k] = worst[k].max(e);
            }
        }
        eprintln!(
            "sigmoid {:.3e}, silu {:.3e}, silu_grad {:.3e} (relative to max(|ref|, 1))",
            worst[0], worst[1], worst[2]
        );
        for (k, w) in worst.iter().enumerate() {
            assert!(*w < 1e-6, "function {k}: {w:e}");
        }
    }

    #[test]
    fn sigmoid_and_silu_special_values() {
        assert_eq!(sigmoid_f32(0.0), 0.5);
        assert_eq!(sigmoid_f32(-0.0), 0.5);
        assert_eq!(sigmoid_f32(f32::INFINITY), 1.0);
        assert_eq!(sigmoid_f32(f32::NEG_INFINITY).to_bits(), 0);
        assert_eq!(sigmoid_f32(120.0), 1.0);
        assert_eq!(sigmoid_f32(-120.0).to_bits(), 0);
        assert_eq!(sigmoid_f32(f32::NAN).to_bits(), CANONICAL_NAN_BITS);
        assert_eq!(silu_f32(f32::INFINITY), f32::INFINITY);
        // -inf * sigmoid(-inf) is -inf * 0, NaN as in torch; canonical here.
        assert_eq!(silu_f32(f32::NEG_INFINITY).to_bits(), CANONICAL_NAN_BITS);
        assert_eq!(silu_f32(-120.0).to_bits(), 0x8000_0000, "-0");
        assert_eq!(silu_grad_f32(0.0), 0.5);
        assert_eq!(silu_grad_f32(f32::NAN).to_bits(), CANONICAL_NAN_BITS);
        assert_eq!(
            canon_nan_f32(f32::from_bits(0xffc0_0000)).to_bits(),
            CANONICAL_NAN_BITS
        );
        assert_eq!(canon_nan_f32(-2.5), -2.5);
    }

    #[test]
    fn the_prelude_is_guarded_and_self_contained() {
        assert!(ACT_PRELUDE.contains("#ifndef QD_ACT_PRELUDE"));
        assert!(ACT_PRELUDE.trim_end().ends_with("#endif"));
        assert!(!ACT_PRELUDE.contains("#include"));
        assert!(!ACT_PRELUDE.contains('\0'));
        assert!(!ACT_PRELUDE.contains("atomic"));
        // Only exact or explicitly-fused operations: no libdevice exp/log.
        for banned in ["expf(", "__expf", "exp2f", "logf", "__fdividef"] {
            assert!(!ACT_PRELUDE.contains(banned), "{banned}");
        }
        for f in [
            "qd_exp(",
            "qd_exp_nonpos(",
            "qd_log(",
            "qd_softplus(",
            "qd_sigmoid(",
            "qd_silu(",
            "qd_silu_grad(",
            "qd_canon_nan(",
        ] {
            assert!(
                ACT_PRELUDE.contains(&format!("float {f}float")),
                "{f} is not defined"
            );
        }
    }

    #[test]
    fn the_sweep_covers_the_edges_and_is_reproducible() {
        let a = act_sweep_inputs(1000);
        let b = act_sweep_inputs(1000);
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert!(a.iter().any(|x| x.is_nan()));
        assert!(a.iter().any(|&x| x < -104.0) && a.iter().any(|&x| x > 88.8));
    }
}
