// Kernels behind `MetalBackend` for the ops tessl has no kernel for.
//
// Every kernel takes element counts and guards its own indices. The host
// refuses any tensor above u32::MAX elements before it binds one, so `uint`
// indices cannot wrap; products that can pass u32 are formed in `ulong`.
//
// Status words, one `atomic_uint` slot of ST_WORDS per op in the device
// thread's status slab, zeroed by the host after a waited commit:
//   0  an input is non-finite        1  an output or intermediate is non-finite
// The rest of the slot is unused. Token ids are range-checked, and the
// cross-entropy valid-row count formed, on the host from its copy of the ids;
// ids `ojas_argmax_rows` and `ojas_topk_rows` produced are below their column
// count by construction, which the host checks against the embedding's
// vocabulary instead.
// The elementwise kernels (silu, mul, add, value-residual) set words 0 and 1
// from the elements they read and write; every other op runs
// `ojas_check_finite` over its inputs and outputs.
//
// A kernel that only reads status words written by an earlier dispatch binds
// them as plain `device const uint*`: an atomic load per thread of one
// address cost about 3 ms per pass over 38.6M elements, against about 0 for
// a plain load. The
// Dispatch->Dispatch barrier after every dispatch makes the earlier writes
// visible.
//
// Compiled with -ffp-contract=off and -fmetal-math-mode=safe; transcendental
// calls are `precise::`.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define ST_IN 0u
#define ST_OUT 1u
#define ST_WORDS 8u

/// Sets `st[word]` when any of `x[0..n)` is non-finite. The host dispatches
/// fewer threads than elements (`CHECK_PER_THREAD` in `device.rs`); each
/// thread walks the window with the grid's stride, so neighbouring lanes
/// read neighbouring floats on every step, and stores at most once. One float
/// per thread ran a read-only pass at 151 GB/s, no faster than a read and a
/// write (`metal_bench kernels`). Any thread count covers the window exactly:
/// thread `i` visits `i, i + stride, ...`, and the walk stops before `j`
/// could pass `n` and wrap.
kernel void ojas_check_finite(
    device const float *x [[buffer(0)]],
    device atomic_uint *st [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &word [[buffer(3)]],
    uint i [[thread_position_in_grid]],
    uint stride [[threads_per_grid]])
{
    if (i >= n || word >= ST_WORDS) return;
    bool bad = false;
    for (uint j = i;; j += stride) {
        bad = bad || !isfinite(x[j]);
        if (n - j <= stride) break;
    }
    if (bad) {
        atomic_store_explicit(&st[word], 1u, memory_order_relaxed);
    }
}

/// `ojas_check_finite` of a bf16 window: `n` elements from element `first`
/// of `x` (a 2-byte element offset under the 4-byte bind alignment). A bf16
/// is NaN or infinite when its eight exponent bits are all set, so the scan
/// reads the bits and widens nothing.
kernel void ojas_check_finite_bf16(
    device const ushort *x [[buffer(0)]],
    device atomic_uint *st [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &word [[buffer(3)]],
    constant uint &first [[buffer(4)]],
    uint i [[thread_position_in_grid]],
    uint stride [[threads_per_grid]])
{
    if (i >= n || word >= ST_WORDS) return;
    bool bad = false;
    for (uint j = i;; j += stride) {
        bad = bad || (x[first + j] & 0x7f80u) == 0x7f80u;
        if (n - j <= stride) break;
    }
    if (bad) {
        atomic_store_explicit(&st[word], 1u, memory_order_relaxed);
    }
}

/// An upload carried in the command: `words` are the host's bytes in the
/// constant arena, copied bit for bit (a NaN is data, not a fault).
kernel void ojas_upload_words(
    device uint *x [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    constant uint *words [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    x[i] = words[i];
}

// ------------------------------------------------------------ pointwise ---

/// The CPU reference's branch: no `exp` of a large positive argument.
inline float ojas_sigmoid_ref(float x)
{
    if (x >= 0.0f) {
        const float z = precise::exp(-x);
        return precise::divide(1.0f, 1.0f + z);
    }
    const float z = precise::exp(x);
    return precise::divide(z, 1.0f + z);
}

// The elementwise kernels below check their own operands: `st` is the op's
// status slot, and a thread sets ST_IN when an input element it read is
// non-finite and ST_OUT when an element it wrote is. That is what separate
// `ojas_check_finite` passes over each input and output reported, without
// the extra reads; the passes were 62% of these ops' GPU time
// (`metal_bench kernels`).

inline void ojas_flag(device atomic_uint *st, uint word)
{
    atomic_store_explicit(&st[word], 1u, memory_order_relaxed);
}

// Round one f32 to bf16 (nearest even) and widen it back. NaN keeps its sign
// and the top payload bits and sets the quiet bit (ojas_core::f32_to_bf16).
inline float ojas_bf16(float v)
{
    const uint bits = as_type<uint>(v);
    const uint mag = bits & 0x7fffffffu;
    uint outb;
    if (mag > 0x7f800000u) {
        outb = ((bits >> 16u) | 0x0040u) << 16u;
    } else {
        // Largest non-NaN magnitude, sign bit set, plus 0x8000 does not wrap.
        const uint round = 0x7fffu + ((bits >> 16u) & 1u);
        outb = ((bits + round) >> 16u) << 16u;
    }
    return as_type<float>(outb);
}

// Round each f32 to bf16 and widen it back. This kernel takes no status word:
// a NaN is a defined rounding result, not a fault. `x` and `y` may be the
// same buffer.
kernel void ojas_round_bf16(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    y[i] = ojas_bf16(x[i]);
}

// Widen each bf16 to f32 exactly (`bits << 16`), from element `first` of
// `x` (a 2-byte element offset under the 4-byte bind alignment). No status
// word: a NaN or infinity is carried, and the matmul-class op that reads the
// widened copy refuses it.
kernel void ojas_bf16_to_f32(
    device const ushort *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &first [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    y[i] = as_type<float>(uint(x[first + i]) << 16u);
}

// Round each f32 to bf16 (nearest even, the NaN rule of `ojas_bf16`) and
// store its 16 bits. No status word, as `ojas_round_bf16`.
kernel void ojas_f32_to_bf16(
    device const float *x [[buffer(0)]],
    device ushort *y [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    y[i] = ushort(as_type<uint>(ojas_bf16(x[i])) >> 16u);
}

kernel void ojas_silu_fwd(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    device atomic_uint *st [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float v = x[i];
    const float r = v * ojas_sigmoid_ref(v);
    y[i] = r;
    if (!isfinite(v)) ojas_flag(st, ST_IN);
    if (!isfinite(r)) ojas_flag(st, ST_OUT);
}

kernel void ojas_silu_bwd(
    device const float *x [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    device atomic_uint *st [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float v = x[i];
    const float gy = g[i];
    const float s = ojas_sigmoid_ref(v);
    const float r = gy * s * (1.0f + v * (1.0f - s));
    out[i] = r;
    if (!isfinite(v) || !isfinite(gy)) ojas_flag(st, ST_IN);
    if (!isfinite(r)) ojas_flag(st, ST_OUT);
}

// `gy * s(x) * s(-x)`: `s * (1 - s)` without the cancellation above x ~ 17,
// where `s` rounds to 1 in f32 (the CPU's expression).
kernel void ojas_sigmoid_fwd(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    device atomic_uint *st [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float v = x[i];
    const float r = ojas_sigmoid_ref(v);
    y[i] = r;
    if (!isfinite(v)) ojas_flag(st, ST_IN);
    if (!isfinite(r)) ojas_flag(st, ST_OUT);
}

/// `max(x, 0)` as torch's `F.relu`: `x` where `x >= 0` (a negative zero stays
/// as it is) and `+0` where `x < 0`, so no value is rounded. The sign is read
/// from the bits, not by a float compare: the GPU's `v < 0.0f` treats a
/// negative subnormal as zero and would pass it through, where the CPU
/// returns `+0`. A NaN stays as it is. A NaN or infinity in `x` flags the
/// input word; the output holds only inputs and zero, so it has no flag of
/// its own.
kernel void ojas_relu_fwd(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    device atomic_uint *st [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float v = x[i];
    const uint bits = as_type<uint>(v);
    const uint magnitude = bits & 0x7fffffffu;
    const bool negative = (bits >> 31) != 0u && magnitude != 0u && magnitude <= 0x7f800000u;
    y[i] = as_type<float>(negative ? 0u : bits);
    if (!isfinite(v)) ojas_flag(st, ST_IN);
}

kernel void ojas_sigmoid_bwd(
    device const float *x [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    device atomic_uint *st [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float v = x[i];
    const float gy = g[i];
    const float r = gy * ojas_sigmoid_ref(v) * ojas_sigmoid_ref(-v);
    out[i] = r;
    if (!isfinite(v) || !isfinite(gy)) ojas_flag(st, ST_IN);
    if (!isfinite(r)) ojas_flag(st, ST_OUT);
}

/// torch's `F.softplus` at its defaults (beta 1, threshold 20). tessl's
/// `qwen35_softplus` (`kernels/qwen35_act.h`), copied: MSL has no `log1p`,
/// and `log(1 + e)` loses most of the value for x in [-15, -8], which
/// `a + dt_bias` reaches routinely, so below -3 it takes the series
/// `e - e^2/2 + ... - e^8/8`, exact to f32 there.
inline float ojas_softplus(float x)
{
    if (x > 20.0f) return x;
    const float e = precise::exp(x);
    if (x < -3.0f) {
        float p = -1.0f / 8.0f;
        p = p * e + 1.0f / 7.0f;
        p = p * e - 1.0f / 6.0f;
        p = p * e + 1.0f / 5.0f;
        p = p * e - 1.0f / 4.0f;
        p = p * e + 1.0f / 3.0f;
        p = p * e - 1.0f / 2.0f;
        p = p * e + 1.0f;
        return e * p;
    }
    return precise::log(1.0f + e);
}

/// The gated delta rule's log decay `g = -exp(a_log[h]) * softplus(a +
/// dt_bias[h])` over `a` `[rows, heads]`, one thread per element.
kernel void ojas_gdn_decay_fwd(
    device const float *a [[buffer(0)]],
    device const float *a_log [[buffer(1)]],
    device const float *dt_bias [[buffer(2)]],
    device float *g [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &heads [[buffer(5)]],
    device atomic_uint *st [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const uint h = i % heads;
    const float av = a[i], lv = a_log[h], dv = dt_bias[h];
    const float r = -precise::exp(lv) * ojas_softplus(av + dv);
    g[i] = r;
    if (!isfinite(av) || !isfinite(lv) || !isfinite(dv)) ojas_flag(st, ST_IN);
    if (!isfinite(r)) ojas_flag(st, ST_OUT);
}

/// `da = gy * -exp(a_log) * softplus'(a + dt_bias)` (1 above 20, the
/// sigmoid below), one thread per element.
kernel void ojas_gdn_decay_bwd(
    device const float *a [[buffer(0)]],
    device const float *a_log [[buffer(1)]],
    device const float *dt_bias [[buffer(2)]],
    device const float *gy [[buffer(3)]],
    device float *da [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant uint &heads [[buffer(6)]],
    device atomic_uint *st [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const uint h = i % heads;
    const float av = a[i], lv = a_log[h], dv = dt_bias[h], gv = gy[i];
    const float x = av + dv;
    const float slope = x > 20.0f ? 1.0f : ojas_sigmoid_ref(x);
    const float r = gv * -precise::exp(lv) * slope;
    da[i] = r;
    if (!isfinite(av) || !isfinite(lv) || !isfinite(dv) || !isfinite(gv)) ojas_flag(st, ST_IN);
    if (!isfinite(r)) ojas_flag(st, ST_OUT);
}

/// `da_log[h]` sums `gy * g` and `ddt_bias[h]` sums `da` over the rows, one
/// thread per head walking rows in ascending order (the CPU's order), so
/// the sums do not depend on the launch.
kernel void ojas_gdn_decay_bwd_sum(
    device const float *a [[buffer(0)]],
    device const float *a_log [[buffer(1)]],
    device const float *dt_bias [[buffer(2)]],
    device const float *gy [[buffer(3)]],
    device const float *da [[buffer(4)]],
    device float *dlog [[buffer(5)]],
    device float *ddt [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &heads [[buffer(8)]],
    device atomic_uint *st [[buffer(9)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= heads) return;
    const float rate = -precise::exp(a_log[h]);
    const float dv = dt_bias[h];
    float acc_log = 0.0f, acc_dt = 0.0f;
    for (uint r = 0; r < rows; ++r) {
        const ulong i = (ulong)r * heads + h;
        acc_log += gy[i] * (rate * ojas_softplus(a[i] + dv));
        acc_dt += da[i];
    }
    dlog[h] = acc_log;
    ddt[h] = acc_dt;
    if (!isfinite(acc_log) || !isfinite(acc_dt)) ojas_flag(st, ST_OUT);
}

kernel void ojas_mul_fwd(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    device atomic_uint *st [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float av = a[i];
    const float bv = b[i];
    const float r = av * bv;
    y[i] = r;
    if (!isfinite(av) || !isfinite(bv)) ojas_flag(st, ST_IN);
    if (!isfinite(r)) ojas_flag(st, ST_OUT);
}

kernel void ojas_mul_bwd(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device const float *g [[buffer(2)]],
    device float *ga [[buffer(3)]],
    device float *gb [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    device atomic_uint *st [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float av = a[i];
    const float bv = b[i];
    const float gy = g[i];
    const float ra = gy * bv;
    const float rb = gy * av;
    ga[i] = ra;
    gb[i] = rb;
    if (!isfinite(av) || !isfinite(bv) || !isfinite(gy)) ojas_flag(st, ST_IN);
    if (!isfinite(ra) || !isfinite(rb)) ojas_flag(st, ST_OUT);
}

/// out = x + y: one IEEE add per element, the same as copying `x` and adding
/// `y` into it in place.
kernel void ojas_add_fwd(
    device const float *x [[buffer(0)]],
    device const float *y [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    device atomic_uint *st [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float xv = x[i];
    const float yv = y[i];
    const float r = xv + yv;
    out[i] = r;
    if (!isfinite(xv) || !isfinite(yv)) ojas_flag(st, ST_IN);
    if (!isfinite(r)) ojas_flag(st, ST_OUT);
}

/// gx = gy_out = g. `x` and `y` are read only for their checks: the backward
/// of an add refuses a non-finite forward input like every other op.
kernel void ojas_add_bwd(
    device const float *x [[buffer(0)]],
    device const float *y [[buffer(1)]],
    device const float *g [[buffer(2)]],
    device float *gx [[buffer(3)]],
    device float *gy_out [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    device atomic_uint *st [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float gv = g[i];
    gx[i] = gv;
    gy_out[i] = gv;
    if (!isfinite(x[i]) || !isfinite(y[i]) || !isfinite(gv)) ojas_flag(st, ST_IN);
}

/// out = alpha * x + beta * y. `out` may be `x` or `y`: each thread reads its
/// element before it writes it.
kernel void ojas_axpby(
    device const float *x [[buffer(0)]],
    device const float *y [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant float &alpha [[buffer(4)]],
    constant float &beta [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float a = alpha * x[i];
    const float b = beta * y[i];
    out[i] = a + b;
}

/// out = fma(beta, y, alpha * x): torch's `x_scaled.add(y, alpha=beta)`, one
/// fused multiply-add after the rounded `alpha * x`. Muon's Nesterov blend
/// (alpha 1) and its parameter update (alpha the decay).
kernel void ojas_axpby_fma(
    device const float *x [[buffer(0)]],
    device const float *y [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant float &alpha [[buffer(4)]],
    constant float &beta [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    out[i] = fma(beta, y[i], alpha * x[i]);
}

/// out = r(r(alpha * x) + r(beta * y)), r rounding to bf16: torch's eager
/// `alpha * x + beta * y` on bf16 tensors, each op rounded. Muon's bf16
/// Newton-Schulz (Ns5Precision::Bf16).
kernel void ojas_axpby_bf16(
    device const float *x [[buffer(0)]],
    device const float *y [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant float &alpha [[buffer(4)]],
    constant float &beta [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float a = ojas_bf16(alpha * x[i]);
    const float b = ojas_bf16(beta * y[i]);
    out[i] = ojas_bf16(a + b);
}

/// out = x / denom[slot], a correctly rounded division (not a reciprocal
/// multiply). The divisor stays on the device.
kernel void ojas_div_scalar(
    device const float *x [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    device const float *denom [[buffer(3)]],
    constant uint &slot [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    out[i] = precise::divide(x[i], denom[slot]);
}

/// Newton-Schulz normalizer from stats = [max |x|, sum (x / max |x|)^2]:
/// stats[2] = max |x| * sqrt(sum) + eps. A non-finite or zero result sets
/// ST_OUT.
kernel void ojas_ns_denom(
    device float *stats [[buffer(0)]],
    device atomic_uint *st [[buffer(1)]],
    constant float &eps [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i != 0u) return;
    const float norm = stats[0] * precise::sqrt(stats[1]);
    const float denom = norm + eps;
    if (!isfinite(denom) || denom == 0.0f) {
        atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
    }
    stats[2] = denom;
}

/// ojas_ns_denom for a bf16 iterate: the norm is rounded to bf16, and so is
/// norm + eps, as torch's bf16 `X.norm() + eps` is.
kernel void ojas_ns_denom_bf16(
    device float *stats [[buffer(0)]],
    device atomic_uint *st [[buffer(1)]],
    constant float &eps [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i != 0u) return;
    const float norm = ojas_bf16(stats[0] * precise::sqrt(stats[1]));
    const float denom = ojas_bf16(norm + eps);
    if (!isfinite(denom) || denom == 0.0f) {
        atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
    }
    stats[2] = denom;
}

// -------------------------------------------------------------- RMSNorm ---

// One simdgroup per row (eight rows per 256-thread threadgroup), so a 64-wide
// QK-norm row keeps every lane busy; each lane strides the row by 32 and the
// row reduction is `simd_sum`, whose order is fixed, so results repeat.

#define RMS_ROWS_PER_TG 8u

/// 1 / sqrt(ss / dim + eps), or ST_OUT and 0 when that is not a finite
/// positive value: the rstd rule of the CPU reference.
inline float rms_rstd_of(float ss, uint dim, float eps, device atomic_uint *st)
{
    const float denom = precise::divide(ss, (float)dim) + eps;
    if (!(isfinite(denom) && denom > 0.0f)) {
        atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
        return 0.0f;
    }
    return precise::divide(1.0f, precise::sqrt(denom));
}

/// y = x * rstd * w, rstd computed in the same pass.
kernel void ojas_rms_fwd(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    device atomic_uint *st [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint r = tg * RMS_ROWS_PER_TG + sg;
    if (r >= rows) return;
    const ulong base = (ulong)r * dim;
    float s = 0.0f;
    for (uint c = lane; c < dim; c += 32u) {
        const float v = x[base + c];
        s += v * v;
    }
    const float rs = rms_rstd_of(simd_sum(s), dim, eps, st);
    for (uint c = lane; c < dim; c += 32u) {
        y[base + c] = x[base + c] * rs * w[c];
    }
}

/// gx = (gy w - xhat * mean) * rstd, with rstd recomputed here and kept in
/// `rstd` for the weight gradient.
kernel void ojas_rms_bwd_rows(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *gy [[buffer(2)]],
    device float *rstd [[buffer(3)]],
    device float *gx [[buffer(4)]],
    device atomic_uint *st [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &dim [[buffer(7)]],
    constant float &eps [[buffer(8)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint r = tg * RMS_ROWS_PER_TG + sg;
    if (r >= rows) return;
    const ulong base = (ulong)r * dim;
    float ss = 0.0f;
    for (uint c = lane; c < dim; c += 32u) {
        const float v = x[base + c];
        ss += v * v;
    }
    const float rs = rms_rstd_of(simd_sum(ss), dim, eps, st);
    if (lane == 0u) rstd[r] = rs;
    float s = 0.0f;
    for (uint c = lane; c < dim; c += 32u) {
        const float dxhat = gy[base + c] * w[c];
        const float xhat = x[base + c] * rs;
        s += dxhat * xhat;
    }
    const float mean = simd_sum(s) * precise::divide(1.0f, (float)dim);
    for (uint c = lane; c < dim; c += 32u) {
        const float dxhat = gy[base + c] * w[c];
        const float xhat = x[base + c] * rs;
        gx[base + c] = (dxhat - xhat * mean) * rs;
    }
}

#define RMS_W_CHUNK 64u

/// Stage 1 of gw: part[k * dim + c] = sum over rows [k * CHUNK, ...) of
/// gy * (x * rstd), rows ascending. Grid: x = column, y = chunk.
kernel void ojas_rms_bwd_w_part(
    device const float *x [[buffer(0)]],
    device const float *gy [[buffer(1)]],
    device const float *rstd [[buffer(2)]],
    device float *part [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const uint k = gid.y;
    if (c >= dim) return;
    const uint r0 = k * RMS_W_CHUNK;
    if (r0 >= rows) return;
    const uint r1 = min(r0 + RMS_W_CHUNK, rows);
    float s = 0.0f;
    for (uint r = r0; r < r1; ++r) {
        const ulong i = (ulong)r * dim + c;
        s += gy[i] * (x[i] * rstd[r]);
    }
    part[(ulong)k * dim + c] = s;
}

/// Stage 2 of gw: gw[c] = sum over chunks of part, chunks ascending.
kernel void ojas_rms_bwd_w_sum(
    device const float *part [[buffer(0)]],
    device float *gw [[buffer(1)]],
    constant uint &chunks [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= dim) return;
    float s = 0.0f;
    for (uint k = 0u; k < chunks; ++k) s += part[(ulong)k * dim + c];
    gw[c] = s;
}

/// y = (1 - s) * v + s * v0 with s = sigmoid(lambda[0]).
/// Checks `v`, `v0` and `y` in-kernel (see the pointwise section); `lam` is
/// one value and keeps its own `ojas_check_finite` pass.
kernel void ojas_vres_fwd(
    device const float *v [[buffer(0)]],
    device const float *v0 [[buffer(1)]],
    device const float *lam [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    device atomic_uint *st [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float s = ojas_sigmoid_ref(lam[0]);
    const float a = v[i];
    const float b = v0[i];
    const float r = (1.0f - s) * a + s * b;
    y[i] = r;
    if (!isfinite(a) || !isfinite(b)) ojas_flag(st, ST_IN);
    if (!isfinite(r)) ojas_flag(st, ST_OUT);
}

/// gv = (1 - s) g, gv0 = s g, and the lambda term t = (v0 - v) g, which a
/// reduction sums. Checks `v`, `v0`, `g`, `gv` and `gv0` in-kernel; `t` is an
/// intermediate whose sum, the lambda gradient, is checked after the
/// reduction.
kernel void ojas_vres_bwd(
    device const float *v [[buffer(0)]],
    device const float *v0 [[buffer(1)]],
    device const float *lam [[buffer(2)]],
    device const float *g [[buffer(3)]],
    device float *gv [[buffer(4)]],
    device float *gv0 [[buffer(5)]],
    device float *t [[buffer(6)]],
    constant uint &n [[buffer(7)]],
    device atomic_uint *st [[buffer(8)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float s = ojas_sigmoid_ref(lam[0]);
    const float a = v[i];
    const float b = v0[i];
    const float gy = g[i];
    const float ra = (1.0f - s) * gy;
    const float rb = s * gy;
    gv[i] = ra;
    gv0[i] = rb;
    t[i] = (b - a) * gy;
    if (!isfinite(a) || !isfinite(b) || !isfinite(gy)) ojas_flag(st, ST_IN);
    if (!isfinite(ra) || !isfinite(rb)) ojas_flag(st, ST_OUT);
}

kernel void ojas_vres_lambda(
    device const float *sum [[buffer(0)]],
    device const float *lam [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i != 0u) return;
    const float s = ojas_sigmoid_ref(lam[0]);
    out[0] = sum[0] * s * (1.0f - s);
}

// ----------------------------------------------------------------- RoPE ---

/// Half-split RoPE on the leading `rot` values of each row of `dim` (pairs
/// `p`, `p + rot/2`); the other `dim - rot` are copied. `rot == dim` rotates
/// the whole row. `mode` 0: cos/sin have the shape of x (and `rot == dim`).
/// `mode` 1: x is [batch, time, heads, dim] and cos/sin are [time, rot].
/// Forward:  y1 = x1 c1 + (-x2) s1,  y2 = x2 c2 + x1 s2.
/// Backward: g1' = g1 c1 + g2 s2,    g2' = -g1 s1 + g2 c2.
/// `mode` 2 is the interleaved layout (SAM2 and SAM3, `rot == dim`): x is
/// [batch, time, heads, dim], cos/sin are [time, dim/2], and pair `i` is
/// the adjacent columns `2i, 2i+1`.
/// Forward:  y0 = a c - b s,  y1 = a s + b c  (a, b = x[2i], x[2i+1]).
/// Backward: the same with s negated (the inverse rotation).
/// Grid: x = unit in [0, rot/2 + dim - rot), y = row. Unit u < rot/2 rotates
/// pair u; a later unit copies column rot + (u - rot/2).
kernel void ojas_rope(
    device const float *x [[buffer(0)]],
    device const float *cs [[buffer(1)]],
    device const float *sn [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &mode [[buffer(6)]],
    constant uint &time [[buffer(7)]],
    constant uint &heads [[buffer(8)]],
    constant uint &backward [[buffer(9)]],
    constant uint &rot [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint half_rot = rot / 2u;
    const uint col = gid.x;
    const uint row = gid.y;
    if (rot > dim || col >= half_rot + (dim - rot) || row >= rows) return;
    if (mode >= 1u && (heads == 0u || time == 0u)) return;
    const ulong base = (ulong)row * dim;
    if (col >= half_rot) {
        const ulong c = base + rot + (col - half_rot);
        y[c] = x[c];
        return;
    }
    if (mode == 2u) {
        const ulong tbase = (ulong)((row / heads) % time) * half_rot;
        const float xa = x[base + 2u * col];
        const float xb = x[base + 2u * col + 1u];
        const float tc = cs[tbase + col];
        const float ts = backward == 0u ? sn[tbase + col] : -sn[tbase + col];
        y[base + 2u * col] = xa * tc - xb * ts;
        y[base + 2u * col + 1u] = xa * ts + xb * tc;
        return;
    }
    const ulong cbase = mode == 0u ? base : (ulong)((row / heads) % time) * rot;
    const float a = x[base + col];
    const float b = x[base + col + half_rot];
    const float c1 = cs[cbase + col];
    const float s1 = sn[cbase + col];
    const float c2 = cs[cbase + col + half_rot];
    const float s2 = sn[cbase + col + half_rot];
    if (backward == 0u) {
        y[base + col] = a * c1 + (-b) * s1;
        y[base + col + half_rot] = b * c2 + a * s2;
    } else {
        y[base + col] = a * c1 + b * s2;
        y[base + col + half_rot] = -a * s1 + b * c2;
    }
}

// ------------------------------------------------------------ embedding ---

/// Grid: x = column, y = token. An out-of-range id writes 0; the host has
/// already refused it, so this only keeps the read in bounds.
kernel void ojas_embed_fwd(
    device const float *table [[buffer(0)]],
    device const uint *ids [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant uint &vocab [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim || gid.y >= n) return;
    const uint id = ids[gid.y];
    const ulong o = (ulong)gid.y * dim + gid.x;
    out[o] = id < vocab ? table[(ulong)id * dim + gid.x] : 0.0f;
}

/// `ojas_embed_fwd` of a bf16 table, from element `first` of `table` (a
/// 2-byte element offset under the 4-byte bind alignment): each picked
/// value is widened exactly (`bits << 16`) as it is gathered, so the output
/// holds an f32 table's bits and no f32 copy of the table exists.
kernel void ojas_embed_fwd_bf16(
    device const ushort *table [[buffer(0)]],
    device const uint *ids [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant uint &vocab [[buffer(5)]],
    constant uint &first [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim || gid.y >= n) return;
    const uint id = ids[gid.y];
    const ulong o = (ulong)gid.y * dim + gid.x;
    out[o] = id < vocab
        ? as_type<float>(uint(table[(ulong)first + (ulong)id * dim + gid.x]) << 16u)
        : 0.0f;
}

#define ARGMAX_TG 256u
#define ARGMAX_NONE 0xFFFFFFFFu

/// `out[r]` = the column of row `r`'s largest value, ties to the lowest
/// column, as `ojas_infer::argmax_token`; every column is `< cols`. One
/// threadgroup per row: each lane keeps its best over a strided walk (in
/// rising column order, so a strict `>` keeps the lowest tie), then a tree
/// over the lanes keeps the larger value, or the lower column on a tie. A
/// non-finite value sets status word 0 (the host reports it at the next
/// sync) and does not take part.
kernel void ojas_argmax_rows(
    device const float *x [[buffer(0)]],
    device uint *out [[buffer(1)]],
    device atomic_uint *st [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint r [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float bv[ARGMAX_TG];
    threadgroup uint bi[ARGMAX_TG];
    if (r >= rows) return;
    const ulong base = (ulong)r * cols;
    float best = 0.0f;
    uint idx = ARGMAX_NONE;
    bool bad = false;
    for (uint c = lane; c < cols; c += ARGMAX_TG) {
        const float v = x[base + c];
        if (!isfinite(v)) {
            bad = true;
        } else if (idx == ARGMAX_NONE || v > best) {
            best = v;
            idx = c;
        }
    }
    if (bad) {
        atomic_store_explicit(&st[ST_IN], 1u, memory_order_relaxed);
    }
    bv[lane] = best;
    bi[lane] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ARGMAX_TG / 2u; s > 0u; s >>= 1u) {
        if (lane < s) {
            const float ov = bv[lane + s];
            const uint oi = bi[lane + s];
            const uint mi = bi[lane];
            const bool take = oi != ARGMAX_NONE
                && (mi == ARGMAX_NONE || ov > bv[lane] || (ov == bv[lane] && oi < mi));
            if (take) {
                bv[lane] = ov;
                bi[lane] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        // A row with no finite value has already set word 0; 0 keeps the
        // output a valid column.
        out[r] = bi[0] == ARGMAX_NONE ? 0u : bi[0];
    }
}

/// An f32's rank key: unsigned order of keys is `f32::total_cmp` order with
/// `-0.0` read as `+0.0`, so the two tie (the lower column wins) and `-inf`
/// ranks below every finite value.
static inline uint ojas_rank_key(float v) {
    const uint raw = as_type<uint>(v);
    const uint b = raw == 0x80000000u ? 0u : raw;
    return b ^ ((b >> 31) != 0u ? 0xFFFFFFFFu : 0x80000000u);
}

/// Row `r`'s `k` leaders ranked by `ojas_core::rank_logits`: value
/// descending with `+0.0` and `-0.0` one value, then column ascending.
/// `values[r * k + j]` and `ids[r * k + j]` are the `j`-th leader; every id
/// is `< cols`. One threadgroup per row and `k` rounds: round `j` takes,
/// over the columns ranked strictly after round `j - 1`'s pick (a smaller
/// key, or the same key at a higher column), the largest key, then the
/// lowest column: a strided walk per lane (rising columns, so a strict `>`
/// keeps the lowest column of equal keys), then a tree over the lanes. The
/// first round's walk sets status word 0 for a NaN or `+inf` (the host
/// reports it at the next sync); `-inf` is a value. Distinct (key, column)
/// pairs leave `cols - j` candidates in round `j`, so every round picks.
kernel void ojas_topk_rows(
    device const float *x [[buffer(0)]],
    device float *values [[buffer(1)]],
    device uint *ids [[buffer(2)]],
    device atomic_uint *st [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &cols [[buffer(5)]],
    constant uint &k [[buffer(6)]],
    uint r [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup uint bk[ARGMAX_TG];
    threadgroup uint bi[ARGMAX_TG];
    if (r >= rows) return;
    const ulong base = (ulong)r * cols;
    const ulong obase = (ulong)r * k;
    uint pk = 0u;
    uint pi = ARGMAX_NONE;
    for (uint j = 0u; j < k; ++j) {
        uint best_k = 0u;
        uint best_i = ARGMAX_NONE;
        bool bad = false;
        for (uint c = lane; c < cols; c += ARGMAX_TG) {
            const float v = x[base + c];
            const uint key = ojas_rank_key(v);
            if (j == 0u && (isnan(v) || v == INFINITY)) {
                bad = true;
            }
            const bool after = j == 0u || key < pk || (key == pk && c > pi);
            if (after && (best_i == ARGMAX_NONE || key > best_k)) {
                best_k = key;
                best_i = c;
            }
        }
        if (bad) {
            atomic_store_explicit(&st[ST_IN], 1u, memory_order_relaxed);
        }
        bk[lane] = best_k;
        bi[lane] = best_i;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = ARGMAX_TG / 2u; s > 0u; s >>= 1u) {
            if (lane < s) {
                const uint ok = bk[lane + s];
                const uint oi = bi[lane + s];
                const uint mi = bi[lane];
                const bool take = oi != ARGMAX_NONE
                    && (mi == ARGMAX_NONE || ok > bk[lane] || (ok == bk[lane] && oi < mi));
                if (take) {
                    bk[lane] = ok;
                    bi[lane] = oi;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        pk = bk[0];
        pi = bi[0];
        if (lane == 0u) {
            // Unreachable for 1 <= k <= cols; 0 keeps the id a valid column.
            const uint col = pi == ARGMAX_NONE ? 0u : pi;
            values[obase + j] = x[base + col];
            ids[obase + j] = col;
        }
        // Every lane has read the pick before the next round overwrites it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/// out[v, d] = sum of grad[pos[k], d] over v's tokens, in ascending token
/// order from 0. Grid: x = column, y = vocabulary row.
kernel void ojas_embed_gather(
    device const float *grad [[buffer(0)]],
    device const uint *starts [[buffer(1)]],
    device const uint *counts [[buffer(2)]],
    device const uint *pos [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant uint &vocab [[buffer(5)]],
    constant uint &dim [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dim || gid.y >= vocab) return;
    const uint start = starts[gid.y];
    const uint count = counts[gid.y];
    float s = 0.0f;
    for (uint k = 0u; k < count; ++k) {
        s += grad[(ulong)pos[start + k] * dim + gid.x];
    }
    out[(ulong)gid.y * dim + gid.x] = s;
}

// ------------------------------------------------------------ reduction ---

#define RED_TG 256u

/// mode 0: sum. mode 1: max |x|. mode 2: sum (x / scale[0])^2, 0 when the
/// scale is 0. Group g reduces [g * chunk, min((g + 1) * chunk, n)).
kernel void ojas_reduce_partial(
    device const float *x [[buffer(0)]],
    device float *part [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &chunk [[buffer(3)]],
    constant uint &mode [[buffer(4)]],
    device const float *scale [[buffer(5)]],
    uint g [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float sh[RED_TG];
    const ulong lo = (ulong)g * chunk;
    const ulong hi = min(lo + chunk, (ulong)n);
    const float sc = mode == 2u ? scale[0] : 1.0f;
    float s = 0.0f;
    for (ulong i = lo + lane; i < hi; i += RED_TG) {
        const float v = x[i];
        if (mode == 1u) {
            s = max(s, fabs(v));
        } else if (mode == 2u) {
            const float t = sc > 0.0f ? precise::divide(v, sc) : 0.0f;
            s += t * t;
        } else {
            s += v;
        }
    }
    sh[lane] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = RED_TG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) {
            sh[lane] = mode == 1u ? max(sh[lane], sh[lane + off]) : sh[lane] + sh[lane + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) part[g] = sh[0];
}

/// One threadgroup combines `count` partials into out[slot]. mode 1 takes
/// the max; every other mode sums.
kernel void ojas_reduce_final(
    device const float *part [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &mode [[buffer(3)]],
    constant uint &slot [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float sh[RED_TG];
    float s = 0.0f;
    for (uint i = lane; i < count; i += RED_TG) {
        s = mode == 1u ? max(s, part[i]) : s + part[i];
    }
    sh[lane] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = RED_TG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) {
            sh[lane] = mode == 1u ? max(sh[lane], sh[lane + off]) : sh[lane] + sh[lane + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) out[slot] = sh[0];
}

// ------------------------------------------------- multi-tensor clip ---
//
// `clip_grad_norm`'s scale over many gradients in few dispatches (its norm
// is tessl's `sq_norm`). One dispatch binds up to NORM_SLOTS gradients at
// buffers 0..23; `tbl` holds each slot's element count and the
// dispatch-local index of its first chunk, so threadgroup g finds its slot
// and its NORM_CHUNK-element chunk without a per-tensor launch. The slot
// choice is uniform across a threadgroup.

#define NORM_SLOTS 24u
#define NORM_CHUNK 4096u
#define NORM_TG 256u
#define NORM_PER (NORM_CHUNK / NORM_TG)
#define NORM_PICK(s) (s == 0u ? g0 : s == 1u ? g1 : s == 2u ? g2 : s == 3u ? g3 : s == 4u ? g4 : s == 5u ? g5 : s == 6u ? g6 : s == 7u ? g7 : s == 8u ? g8 : s == 9u ? g9 : s == 10u ? g10 : s == 11u ? g11 : s == 12u ? g12 : s == 13u ? g13 : s == 14u ? g14 : s == 15u ? g15 : s == 16u ? g16 : s == 17u ? g17 : s == 18u ? g18 : s == 19u ? g19 : s == 20u ? g20 : s == 21u ? g21 : s == 22u ? g22 : g23)

/// The slot threadgroup `g` belongs to: the last whose first chunk is at
/// or before `g`.
inline uint norm_slot(constant uint *tbl, uint used, uint g)
{
    uint s = 0u;
    for (uint k = 1u; k < used; ++k) {
        if (g >= tbl[2u * k + 1u]) s = k;
    }
    return s;
}

/// Every gradient of the dispatch times `scale`, in place; a non-finite
/// product sets st[ST_OUT].
kernel void ojas_scale_multi(
    device float *g0 [[buffer(0)]],
    device float *g1 [[buffer(1)]],
    device float *g2 [[buffer(2)]],
    device float *g3 [[buffer(3)]],
    device float *g4 [[buffer(4)]],
    device float *g5 [[buffer(5)]],
    device float *g6 [[buffer(6)]],
    device float *g7 [[buffer(7)]],
    device float *g8 [[buffer(8)]],
    device float *g9 [[buffer(9)]],
    device float *g10 [[buffer(10)]],
    device float *g11 [[buffer(11)]],
    device float *g12 [[buffer(12)]],
    device float *g13 [[buffer(13)]],
    device float *g14 [[buffer(14)]],
    device float *g15 [[buffer(15)]],
    device float *g16 [[buffer(16)]],
    device float *g17 [[buffer(17)]],
    device float *g18 [[buffer(18)]],
    device float *g19 [[buffer(19)]],
    device float *g20 [[buffer(20)]],
    device float *g21 [[buffer(21)]],
    device float *g22 [[buffer(22)]],
    device float *g23 [[buffer(23)]],
    device atomic_uint *st [[buffer(25)]],
    constant uint *tbl [[buffer(26)]],
    constant uint &used [[buffer(27)]],
    constant float &scale [[buffer(28)]],
    uint g [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint s = norm_slot(tbl, used, g);
    const uint n = tbl[2u * s];
    const ulong lo = (ulong)(g - tbl[2u * s + 1u]) * NORM_CHUNK;
    device float *x = NORM_PICK(s);
    bool bad = false;
    for (uint j = 0u; j < NORM_PER; ++j) {
        const ulong i = lo + (ulong)(j * NORM_TG + lane);
        if (i < n) {
            const float r = x[i] * scale;
            bad = bad || !isfinite(r);
            x[i] = r;
        }
    }
    if (bad) atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
}

// -------------------------------------------------------- cross-entropy ---

#define CE_TG 256u

/// (m, s) of a log-sum-exp: s = sum exp(x - m). Merging with an empty side
/// (m = -inf, s = 0) leaves the other side unchanged.
inline float2 ce_merge(float2 a, float2 b)
{
    if (a.x == -INFINITY) return b;
    if (b.x == -INFINITY) return a;
    const float m = max(a.x, b.x);
    return float2(m, a.y * precise::exp(a.x - m) + b.y * precise::exp(b.x - m));
}

/// Cross-entropy in two passes over each row instead of five, with the
/// finiteness checks folded in rather than run as separate passes:
/// 1. every row, ignored or not, is read once: a NaN or infinity sets ST_IN,
///    and each lane keeps an online (max, sum of exp) that the threadgroup
///    merges; the loss row is max + log(sum) - logit[target];
/// 2. with `write_grad`, the gradient row (softmax / n_valid, minus
///    1 / n_valid at the target, the same arithmetic as before) is written,
///    and a value that is not finite sets ST_OUT.
/// An ignored or out-of-range row writes loss 0 and a zero gradient row.
/// n_valid, the rows neither ignored nor out of range, is counted on the
/// host from its copy of the targets.
kernel void ojas_ce_fused(
    device const float *logits [[buffer(0)]],
    device const uint *targets [[buffer(1)]],
    device atomic_uint *st [[buffer(2)]],
    device float *loss_rows [[buffer(3)]],
    device float *grad [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &vocab [[buffer(6)]],
    constant uint &has_ignore [[buffer(7)]],
    constant uint &ignore [[buffer(8)]],
    constant uint &write_grad [[buffer(9)]],
    constant uint &n_valid [[buffer(10)]],
    uint r [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float2 sh[CE_TG];
    if (r >= rows) return;
    const uint t = targets[r];
    const bool skip = (has_ignore != 0u && t == ignore) || t >= vocab;
    const ulong base = (ulong)r * vocab;
    bool bad = false;
    float m = -INFINITY;
    float s = 0.0f;
    for (uint c = lane; c < vocab; c += CE_TG) {
        const float x = logits[base + c];
        if (!isfinite(x)) {
            bad = true;
            continue;
        }
        if (x > m) {
            s = s * precise::exp(m - x) + 1.0f;
            m = x;
        } else {
            s += precise::exp(x - m);
        }
    }
    if (bad) atomic_store_explicit(&st[ST_IN], 1u, memory_order_relaxed);
    if (skip) {
        if (lane == 0u) loss_rows[r] = 0.0f;
        if (write_grad != 0u) {
            for (uint c = lane; c < vocab; c += CE_TG) grad[base + c] = 0.0f;
        }
        return;
    }
    sh[lane] = float2(m, s);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = CE_TG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) sh[lane] = ce_merge(sh[lane], sh[lane + off]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mx = sh[0].x;
    const float sum = sh[0].y;
    if (lane == 0u) loss_rows[r] = mx + precise::log(sum) - logits[base + t];
    if (write_grad == 0u) return;
    const float denom = (float)n_valid;
    const float at_target = precise::divide(1.0f, denom);
    bool bad_out = false;
    for (uint c = lane; c < vocab; c += CE_TG) {
        const float p = precise::divide(precise::exp(logits[base + c] - mx), sum);
        float g = precise::divide(p, denom);
        if (c == t) g -= at_target;
        bad_out = bad_out || !isfinite(g);
        grad[base + c] = g;
    }
    if (bad_out) atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
}

// ------------------------------------------- linear cross-entropy (T3) ---
//
// The logits of `input @ weight^T` exist one [rr, cc] tile at a time (tessl
// GEMMs write the tile); these kernels turn tiles into the loss and the
// gradient of the logits with ojas_ce_fused's arithmetic. Per tile row the
// running (max, sum of exp) is merged across vocabulary tiles with ce_merge;
// `stats` holds (max, sum, target logit) per row of the current row tile.

/// Per tile row (one threadgroup): online (max, sum exp) of the tile row,
/// merged into the row's running pair (assigned on the first column tile),
/// and the target's logit when the target falls in this tile. Every logit is
/// checked, ignored rows included; a non-finite one sets ST_IN, because the
/// composition's linear_forward refuses it before cross-entropy runs.
kernel void ojas_lce_stats(
    device const float *tile [[buffer(0)]],
    device const uint *targets [[buffer(1)]],
    device atomic_uint *st [[buffer(2)]],
    device float *stats [[buffer(3)]],
    constant uint &rr [[buffer(4)]],
    constant uint &cc [[buffer(5)]],
    constant uint &col0 [[buffer(6)]],
    constant uint &first [[buffer(7)]],
    uint r [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float2 sh[CE_TG];
    if (r >= rr) return;
    const ulong base = (ulong)r * cc;
    bool bad = false;
    float m = -INFINITY;
    float s = 0.0f;
    for (uint c = lane; c < cc; c += CE_TG) {
        const float x = tile[base + c];
        if (!isfinite(x)) {
            bad = true;
            continue;
        }
        if (x > m) {
            s = s * precise::exp(m - x) + 1.0f;
            m = x;
        } else {
            s += precise::exp(x - m);
        }
    }
    if (bad) atomic_store_explicit(&st[ST_IN], 1u, memory_order_relaxed);
    sh[lane] = float2(m, s);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = CE_TG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) sh[lane] = ce_merge(sh[lane], sh[lane + off]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane != 0u) return;
    const float2 run = first != 0u ? float2(-INFINITY, 0.0f) : float2(stats[3u * r], stats[3u * r + 1u]);
    const float2 merged = ce_merge(run, sh[0]);
    stats[3u * r] = merged.x;
    stats[3u * r + 1u] = merged.y;
    const uint t = targets[r];
    if (t >= col0 && t - col0 < cc) stats[3u * r + 2u] = tile[base + (t - col0)];
}

/// loss_rows[r] = max + log(sum) - target logit, or 0 for an ignored or
/// out-of-range target (the host has refused the latter).
kernel void ojas_lce_loss(
    device const float *stats [[buffer(0)]],
    device const uint *targets [[buffer(1)]],
    device float *loss_rows [[buffer(2)]],
    constant uint &rr [[buffer(3)]],
    constant uint &vocab [[buffer(4)]],
    constant uint &has_ignore [[buffer(5)]],
    constant uint &ignore [[buffer(6)]],
    uint r [[thread_position_in_grid]])
{
    if (r >= rr) return;
    const uint t = targets[r];
    const bool skip = (has_ignore != 0u && t == ignore) || t >= vocab;
    loss_rows[r] = skip
        ? 0.0f
        : stats[3u * r] + precise::log(stats[3u * r + 1u]) - stats[3u * r + 2u];
}

/// The logits tile becomes the gradient tile in place: softmax / n_valid,
/// minus 1 / n_valid at the target, zero for a skipped row; the arithmetic
/// of ojas_ce_fused. A non-finite value sets ST_OUT. x is the column, y the
/// row.
kernel void ojas_lce_grad(
    device float *tile [[buffer(0)]],
    device const uint *targets [[buffer(1)]],
    device atomic_uint *st [[buffer(2)]],
    device const float *stats [[buffer(3)]],
    constant uint &rr [[buffer(4)]],
    constant uint &cc [[buffer(5)]],
    constant uint &col0 [[buffer(6)]],
    constant uint &vocab [[buffer(7)]],
    constant uint &has_ignore [[buffer(8)]],
    constant uint &ignore [[buffer(9)]],
    constant uint &n_valid [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= cc || gid.y >= rr) return;
    const uint r = gid.y;
    const uint t = targets[r];
    const ulong at = (ulong)r * cc + gid.x;
    if ((has_ignore != 0u && t == ignore) || t >= vocab) {
        tile[at] = 0.0f;
        return;
    }
    const float denom = (float)n_valid;
    const float p = precise::divide(precise::exp(tile[at] - stats[3u * r]), stats[3u * r + 1u]);
    float g = precise::divide(p, denom);
    if (col0 + gid.x == t) g -= precise::divide(1.0f, denom);
    if (!isfinite(g)) atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
    tile[at] = g;
}

/// dst = src when `assign`, else dst += src: a gradient tile's first
/// contribution, then the rest.
kernel void ojas_add_into(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &assign [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    dst[i] = assign != 0u ? src[i] : dst[i] + src[i];
}

kernel void ojas_ce_mean(
    device const float *sum [[buffer(0)]],
    constant uint &n_valid [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i != 0u) return;
    out[0] = precise::divide(sum[0], (float)n_valid);
}

// --------------------------------------------------------------- AdamW ---
//
// torch.optim.AdamW's single-tensor step, in tessl `qwen35_adamw_f32`'s f32
// arithmetic and order (decoupled decay, torch's lerp for the first moment,
// `sqrt(v) / sqrt(bc2) + eps`, then `p += -step_size * m / denom`). The host
// forms the scalars in f64 from `ojas_core::check_adamw`.
//
// Transactional without copies: `ojas_adamw_check` computes every element
// and only flags status (ST_IN for a non-finite input, ST_OUT for a
// non-finite p, m or v); `ojas_adamw_apply`, dispatched after it in the same
// command buffer, recomputes and writes in place only if no status word is
// set. The two kernels run the same inline arithmetic on the same inputs,
// so what is written is what was checked.

struct OjasAdamW {
    float decay_mul;       // 1 - lr * weight_decay
    float lerp_w;          // 1 - beta1
    float beta2;
    float one_minus_beta2;
    float step_size;       // lr / (1 - beta1^step)
    float bc2_sqrt;        // sqrt(1 - beta2^step)
    float eps;
};

inline void ojas_adamw_elem(
    float p, float g, float m, float v, constant OjasAdamW &a,
    thread float &p_out, thread float &m_out, thread float &v_out)
{
    float w = p * a.decay_mul;
    const float diff = g - m;
    const float mi = a.lerp_w < 0.5f ? m + a.lerp_w * diff : g - diff * (1.0f - a.lerp_w);
    const float vi = v * a.beta2 + (a.one_minus_beta2 * g) * g;
    const float denom = precise::divide(precise::sqrt(vi), a.bc2_sqrt) + a.eps;
    w = w + (-a.step_size) * precise::divide(mi, denom);
    p_out = w;
    m_out = mi;
    v_out = vi;
}

kernel void ojas_adamw_check(
    device const float *p [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device const float *m [[buffer(2)]],
    device const float *v [[buffer(3)]],
    device atomic_uint *st [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant OjasAdamW &a [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float pi = p[i], gi = g[i], mi = m[i], vi = v[i];
    if (!(isfinite(pi) && isfinite(gi) && isfinite(mi) && isfinite(vi))) {
        atomic_store_explicit(&st[ST_IN], 1u, memory_order_relaxed);
        return;
    }
    float po, mo, vo;
    ojas_adamw_elem(pi, gi, mi, vi, a, po, mo, vo);
    if (!(isfinite(po) && isfinite(mo) && isfinite(vo))) {
        atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
    }
}

kernel void ojas_adamw_apply(
    device float *p [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device float *m [[buffer(2)]],
    device float *v [[buffer(3)]],
    device const uint *st [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant OjasAdamW &a [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (st[ST_IN] != 0u || st[ST_OUT] != 0u) {
        return;
    }
    float po, mo, vo;
    ojas_adamw_elem(p[i], g[i], m[i], v[i], a, po, mo, vo);
    p[i] = po;
    m[i] = mo;
    v[i] = vo;
}

// ---------------------------------------------------- accumulate_grad ---
//
// acc += grad in place, transactional as AdamW is: the check kernel decides
// (ST_IN for a non-finite input, ST_OUT for a non-finite sum) and the apply
// kernel, after it in the same command buffer, writes the identical f32 sum
// only if no status word is set.

kernel void ojas_acc_check(
    device const float *acc [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device atomic_uint *st [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float a = acc[i], b = g[i];
    if (!(isfinite(a) && isfinite(b))) {
        atomic_store_explicit(&st[ST_IN], 1u, memory_order_relaxed);
        return;
    }
    if (!isfinite(a + b)) {
        atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
    }
}

kernel void ojas_acc_apply(
    device float *acc [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device const uint *st [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (st[ST_IN] != 0u || st[ST_OUT] != 0u) {
        return;
    }
    acc[i] = acc[i] + g[i];
}

/// dst = src, only if no input or output status word is set: the commit of
/// a step whose checks ran earlier in the same command buffer.
kernel void ojas_copy_if_clean(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    device const uint *st [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (st[ST_IN] != 0u || st[ST_OUT] != 0u) {
        return;
    }
    dst[i] = src[i];
}

// ------------------------------------------------------------ KV cache ---
//
// The cache is time-major [B, Tcap, Hkv, D], as ojas-infer's host cache is.

/// cache[b, (at + t) % Tcap, :, :] = src[b, t, :, :] (the cache is a ring),
/// written only if no status word is set (the source's finite check runs
/// before it in the same command buffer). `span` is Tn * Hkv * D, `cap_span`
/// Tcap * Hkv * D, `row` Hkv * D, and `at_slot` at % Tcap.
kernel void ojas_kv_write(
    device const float *src [[buffer(0)]],
    device float *cache [[buffer(1)]],
    device const uint *st [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &span [[buffer(4)]],
    constant uint &cap_span [[buffer(5)]],
    constant uint &at_slot [[buffer(6)]],
    constant uint &row [[buffer(7)]],
    constant uint &cap [[buffer(8)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (st[ST_IN] != 0u || st[ST_OUT] != 0u) {
        return;
    }
    const uint b = i / span;
    const uint r = i - b * span;
    const uint t = r / row;
    const uint slot = (at_slot + t) % cap;
    cache[(ulong)b * cap_span + (ulong)slot * row + (r - t * row)] = src[i];
}

#define CA_SG 32u
#define CA_THREADS (CA_SG * 32u)
#define CA_MAX_D 256u
// A lane holds dims lane, lane+32, ... (eight slots cover 256). The
// simdgroup output is staged 128 dims at a time so `so` stays 16 KiB.
#define CA_PER_LANE 8u
#define CA_CHUNK 128u

/// Causal attention of Tq queries against the kv_len positions written to a
/// ring cache (position j in slot j % cap), grouped-query: head h reads KV
/// head h / (H / Hkv). One threadgroup per (query, head, split) and batch,
/// x = (i * H + h) * splits + s; query i sits at position pos = kv_len - Tq
/// + i and reads positions [lo, pos], lo = pos + 1 - window with a window
/// (window > 0) and 0 without.
///
/// Splits cover [origin, kv_len), origin being query 0's lo. Split s walks
/// keys [max(origin + s * chunk, lo), min(origin + (s + 1) * chunk, pos + 1)). Each of
/// its 32 simdgroups walks keys j0 + sg, j0 + sg + 32, ... with an online
/// softmax; a lane holds dims lane, lane + 32, ... of its partial output,
/// and the score is a `simd_sum`, whose order is fixed. The simdgroups'
/// (max, sum, output) are merged in index order, so results repeat bit for
/// bit. With splits = 1 (chunk >= kv_len) the threadgroup writes the
/// output; otherwise it writes (max, sum, unnormalized output) to `part` at
/// ((row * splits + s) * (d + 2)), row = (b * Tq + i) * H + h, and
/// `ojas_cached_attn_merge` combines the splits in index order. A split
/// with no key writes max = -inf and adds nothing.
///
/// Finite checks are folded in: every q element is read (once per split),
/// and every cache element some query reads is checked as it is read, so a
/// NaN or infinity there sets ST_IN. Slots no query reads are not checked. A non-finite output (scores that
/// overflow) sets ST_OUT, here or in the merge.
kernel void ojas_cached_attn(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *out [[buffer(3)]],
    device atomic_uint *st [[buffer(4)]],
    constant uint &tq [[buffer(5)]],
    constant uint &heads [[buffer(6)]],
    constant uint &kv_heads [[buffer(7)]],
    constant uint &d [[buffer(8)]],
    constant uint &cap [[buffer(9)]],
    constant uint &kv_len [[buffer(10)]],
    constant float &scale [[buffer(11)]],
    constant uint &splits [[buffer(12)]],
    constant uint &chunk [[buffer(13)]],
    device float *part [[buffer(14)]],
    constant uint &window [[buffer(15)]],
    constant uint &origin [[buffer(16)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float qs[CA_MAX_D];
    threadgroup float sm[CA_SG];
    threadgroup float sl[CA_SG];
    threadgroup float so[CA_SG][CA_CHUNK];
    const uint row = tg.x / splits;
    const uint split = tg.x - row * splits;
    const uint i = row / heads;
    const uint hh = row - i * heads;
    const uint b = tg.y;
    if (i >= tq) return;
    if (d > CA_MAX_D) {
        // The host refuses this first; if it ever arrives, the output is
        // unwritten, so the op must fail rather than return it.
        if (tid == 0u) atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
        return;
    }
    const uint kh = hh / (heads / kv_heads);
    const uint pos = kv_len - tq + i;
    const ulong qbase = (((ulong)b * tq + i) * heads + hh) * d;
    bool bad = false;
    for (uint c = tid; c < d; c += CA_THREADS) {
        const float x = q[qbase + c];
        bad = bad || !isfinite(x);
        qs[c] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY;
    float l = 0.0f;
    float o[CA_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint lo = (window > 0u && pos + 1u > window) ? pos + 1u - window : 0u;
    const uint j0 = max(origin + split * chunk, lo);
    const uint j1 = min(origin + split * chunk + chunk, pos + 1u);
    for (uint j = j0 + sg; j < j1; j += CA_SG) {
        const ulong kb = (((ulong)b * cap + j % cap) * kv_heads + kh) * d;
        float part = 0.0f;
        for (uint t = 0u; t < CA_PER_LANE; ++t) {
            const uint c = lane + 32u * t;
            if (c < d) {
                const float kx = k[kb + c];
                bad = bad || !isfinite(kx);
                part += qs[c] * kx;
            }
        }
        const float s = simd_sum(part) * scale;
        const float mn = max(m, s);
        const float corr = precise::exp(m - mn);
        const float p = precise::exp(s - mn);
        l = l * corr + p;
        for (uint t = 0u; t < CA_PER_LANE; ++t) {
            const uint c = lane + 32u * t;
            if (c < d) {
                const float vx = v[kb + c];
                bad = bad || !isfinite(vx);
                o[t] = o[t] * corr + p * vx;
            }
        }
        m = mn;
    }
    if (bad) atomic_store_explicit(&st[ST_IN], 1u, memory_order_relaxed);
    if (lane == 0u) {
        sm[sg] = m;
        sl[sg] = l;
    }
    // Every thread hits both barriers of every chunk. `d` is uniform, so
    // the loop count is too. There is no return after the first barrier.
    for (uint base = 0u; base < d; base += CA_CHUNK) {
        const uint end = min(base + CA_CHUNK, d);
        for (uint t = 0u; t < CA_PER_LANE; ++t) {
            const uint c = lane + 32u * t;
            if (base <= c && c < end) so[sg][c - base] = o[t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (base <= tid && tid < end) {
            float mx = -INFINITY;
            for (uint s = 0u; s < CA_SG; ++s) mx = max(mx, sm[s]);
            float total = 0.0f;
            float acc = 0.0f;
            for (uint s = 0u; s < CA_SG; ++s) {
                // A simdgroup with no key (pos < s) has m = -inf and adds nothing.
                if (sm[s] == -INFINITY) continue;
                const float w = precise::exp(sm[s] - mx);
                total += sl[s] * w;
                acc += so[s][tid - base] * w;
            }
            if (splits > 1u) {
                const ulong pb = ((((ulong)b * tq + i) * heads + hh) * splits + split) * (d + 2u);
                if (tid == 0u) {
                    part[pb] = mx;
                    part[pb + 1u] = total;
                }
                part[pb + 2u + tid] = acc;
            } else {
                const float y = precise::divide(acc, total);
                if (!isfinite(y)) atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
                out[qbase + tid] = y;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/// Combine `ojas_cached_attn`'s splits: one threadgroup per (query, head) and
/// batch, x = i * H + h, one thread per output dim. Splits merge in index
/// order, so results repeat bit for bit. A non-finite output sets ST_OUT.
kernel void ojas_cached_attn_merge(
    device const float *part [[buffer(0)]],
    device float *out [[buffer(1)]],
    device atomic_uint *st [[buffer(2)]],
    constant uint &d [[buffer(3)]],
    constant uint &splits [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 ntg [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (tid >= d) return;
    const ulong row = (ulong)tg.y * ntg.x + tg.x;
    const ulong pb = row * splits * (d + 2u);
    float mx = -INFINITY;
    for (uint s = 0u; s < splits; ++s) mx = max(mx, part[pb + s * (d + 2u)]);
    float total = 0.0f;
    float acc = 0.0f;
    for (uint s = 0u; s < splits; ++s) {
        const ulong at = pb + s * (d + 2u);
        // A split with no key (or a row with none at all) adds nothing.
        if (part[at] == -INFINITY) continue;
        const float w = precise::exp(part[at] - mx);
        total += part[at + 1u] * w;
        acc += part[at + 2u + tid] * w;
    }
    const float y = precise::divide(acc, total);
    if (!isfinite(y)) atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
    out[row * d + tid] = y;
}

// ------------------------------------------------------------- permute ---
//
// y = permute(x).contiguous(). One thread per output element: its row-major
// coordinates over `oshape` pick the input element through `istride`, the
// input stride of the axis each output axis came from. Values move as bits
// (`uint`), so NaN payloads, signed zeros and subnormals are unchanged.
// `rank` <= 8; every offset is below `n` <= u32::MAX.

#define PERMUTE_MAX_RANK 8u

kernel void ojas_permute(
    device const uint *x [[buffer(0)]],
    device uint *y [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &rank [[buffer(3)]],
    constant uint *oshape [[buffer(4)]],
    constant uint *istride [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n || rank > PERMUTE_MAX_RANK) return;
    uint rem = i;
    uint src = 0u;
    for (uint a = rank; a > 0u; --a) {
        const uint extent = oshape[a - 1u];
        src += (rem % extent) * istride[a - 1u];
        rem /= extent;
    }
    y[i] = x[src];
}

// Two kernels for the permutes `src/permute.rs` recognises. Both move words
// as `uint`s (a `uint4` is four of them), so every bit pattern is copied
// unchanged, as in `ojas_permute`; the host has checked the alignment and the
// strides they need.

/// Rows of `run4` `uint4` words that are contiguous in both tensors: a thread
/// copies `PERMUTE_WORDS_PER_THREAD` consecutive words of one row (`run4` is a
/// multiple of it), so the divide that finds the row and the walk over the
/// `outer` axes that pick its input offset (strides in `uint4` units) are
/// paid once per that many words. `total4` is `n / 4`, a multiple of it.
#define PERMUTE_WORDS_PER_THREAD 4u

/// `check_word` value that turns the finite test off (the caller has scanned
/// the input): every word of a `permute`'s input is read exactly once, so the
/// kernels test what they read and flag `check_word` of `st`, which is what
/// the separate `ojas_check_finite` pass would have flagged.
#define PERMUTE_NO_CHECK 0xFFFFFFFFu

/// Whether any of four words is the bit pattern of a NaN or infinity.
inline bool ojas_has_nonfinite(uint4 v)
{
    return any((v & uint4(0x7f800000u)) == uint4(0x7f800000u));
}

kernel void ojas_permute_runs(
    device const uint4 *x [[buffer(0)]],
    device uint4 *y [[buffer(1)]],
    constant uint &total4 [[buffer(2)]],
    constant uint &run4 [[buffer(3)]],
    constant uint &outer [[buffer(4)]],
    constant uint *oshape [[buffer(5)]],
    constant uint *istride4 [[buffer(6)]],
    device atomic_uint *st [[buffer(7)]],
    constant uint &check_word [[buffer(8)]],
    uint i [[thread_position_in_grid]])
{
    const uint word = i * PERMUTE_WORDS_PER_THREAD;
    if (word >= total4 || outer > PERMUTE_MAX_RANK) return;
    uint row = word / run4;
    uint src = word - row * run4;
    for (uint a = outer; a > 0u; --a) {
        const uint extent = oshape[a - 1u];
        src += (row % extent) * istride4[a - 1u];
        row /= extent;
    }
    bool bad = false;
    for (uint k = 0u; k < PERMUTE_WORDS_PER_THREAD; ++k) {
        const uint4 v = x[src + k];
        bad = bad || ojas_has_nonfinite(v);
        y[word + k] = v;
    }
    if (bad && check_word != PERMUTE_NO_CHECK) ojas_flag(st, check_word);
}

#define PERMUTE_TILE 32u
#define PERMUTE_TILE_THREADS 256u

/// Planes of `rows x cols` words, output `[plane][b][a]` from input
/// `x[plane offset + b + a * col_stride]`: `b` is contiguous in the input and
/// `a` in the output, so a threadgroup reads a 32 x 32 tile along `b` into
/// threadgroup memory and writes it along `a`. Grid: `ceil(rows / 32)`
/// threadgroups in x, `planes * col_tiles` in y (`col_tiles = ceil(cols / 32)`),
/// 256 threads each. The plane's input offset is walked over the `outer`
/// axes that pick it.
kernel void ojas_permute_tile(
    device const uint *x [[buffer(0)]],
    device uint *y [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &col_stride [[buffer(4)]],
    constant uint &col_tiles [[buffer(5)]],
    constant uint &outer [[buffer(6)]],
    constant uint *oshape [[buffer(7)]],
    constant uint *istride [[buffer(8)]],
    device atomic_uint *st [[buffer(9)]],
    constant uint &check_word [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint tile[PERMUTE_TILE][PERMUTE_TILE + 1u];
    bool bad = false;
    if (outer > PERMUTE_MAX_RANK) return;
    const uint b0 = tg.x * PERMUTE_TILE;
    const uint a0 = (tg.y % col_tiles) * PERMUTE_TILE;
    const uint plane = tg.y / col_tiles;
    uint rem = plane;
    uint base = 0u;
    for (uint k = outer; k > 0u; --k) {
        const uint extent = oshape[k - 1u];
        base += (rem % extent) * istride[k - 1u];
        rem /= extent;
    }
    const uint lx = tid % PERMUTE_TILE;
    const uint ly = tid / PERMUTE_TILE;
    for (uint j = 0u; j < PERMUTE_TILE / 8u; ++j) {
        const uint la = ly + 8u * j;
        const uint b = b0 + lx;
        const uint a = a0 + la;
        if (b < rows && a < cols) {
            const uint v = x[base + b + a * col_stride];
            bad = bad || (v & 0x7f800000u) == 0x7f800000u;
            tile[la][lx] = v;
        }
    }
    if (bad && check_word != PERMUTE_NO_CHECK) ojas_flag(st, check_word);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0u; j < PERMUTE_TILE / 8u; ++j) {
        const uint lb = ly + 8u * j;
        const uint a = a0 + lx;
        const uint b = b0 + lb;
        if (b < rows && a < cols) y[(plane * rows + b) * cols + a] = tile[lx][lb];
    }
}

// ---------------------------------------------------------- convolution ---
//
// A convolution as GEMM (`Backend::conv2d_forward`): `ojas_im2col` lays each
// output pixel's window out as a row of `cols`, tessl's GEMM multiplies it
// by the weight, and `ojas_conv_nchw` turns the `[pixels, out]` product into
// NCHW, adding the bias last as the CPU does. Dilation is 1 and there is one
// group; `total` <= u32::MAX.

/// cols[p][k]: p = (n, oy, ox) row-major, k = (c, ky, kx) row-major; the
/// input value at (oy * sh + ky - ph, ox * sw + kx - pw), 0 where that lies
/// outside the image. One thread per element of `cols`.
kernel void ojas_im2col(
    device const float *x [[buffer(0)]],
    device float *cols [[buffer(1)]],
    constant uint &total [[buffer(2)]],
    constant uint &c [[buffer(3)]],
    constant uint &h [[buffer(4)]],
    constant uint &w [[buffer(5)]],
    constant uint &kh [[buffer(6)]],
    constant uint &kw [[buffer(7)]],
    constant uint &sh [[buffer(8)]],
    constant uint &sw [[buffer(9)]],
    constant uint &ph [[buffer(10)]],
    constant uint &pw [[buffer(11)]],
    constant uint &oh [[buffer(12)]],
    constant uint &ow [[buffer(13)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint window = kh * kw;
    const uint width = c * window;
    const uint pixel = i / width;
    const uint k = i % width;
    const uint ox = pixel % ow;
    const uint oy = (pixel / ow) % oh;
    const uint n = pixel / (ow * oh);
    const uint ci = k / window;
    const uint ky = (k % window) / kw;
    const uint kx = k % kw;
    const int iy = (int)(oy * sh + ky) - (int)ph;
    const int ix = (int)(ox * sw + kx) - (int)pw;
    float v = 0.0f;
    if (iy >= 0 && iy < (int)h && ix >= 0 && ix < (int)w) {
        v = x[(((ulong)n * c + ci) * h + (uint)iy) * w + (uint)ix];
    }
    cols[i] = v;
}

/// y[n][o][pixel] = prod[n * pixels + pixel][o] + bias[o], over y of
/// [batch, out, pixels]. `has_bias` 0 leaves the product as it is.
kernel void ojas_conv_nchw(
    device const float *prod [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &total [[buffer(3)]],
    constant uint &out_c [[buffer(4)]],
    constant uint &pixels [[buffer(5)]],
    constant uint &has_bias [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint pixel = i % pixels;
    const uint o = (i / pixels) % out_c;
    const uint n = i / (pixels * out_c);
    const float v = prod[((ulong)n * pixels + pixel) * out_c + o];
    y[i] = has_bias != 0u ? v + bias[o] : v;
}

/// Q of a cross-attention for tessl's `encoder_attn`, which takes equal query
/// and key lengths: x `[B, H, Tq, D]` to y `[B, Tk, H, D]` with rows `t >= Tq`
/// zero. One thread per element of y; values move as bits.
kernel void ojas_pad_q(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &total [[buffer(2)]],
    constant uint &h [[buffer(3)]],
    constant uint &tq [[buffer(4)]],
    constant uint &tk [[buffer(5)]],
    constant uint &d [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint di = i % d;
    const uint hi = (i / d) % h;
    const uint t = (i / (d * h)) % tk;
    const uint b = i / (d * h * tk);
    y[i] = t < tq ? x[(((ulong)b * h + hi) * tq + t) * d + di] : 0.0f;
}

/// Depthwise convolution (`groups == c == o`, one filter per channel), one
/// thread per output element: the taps in ascending `(i, j)` order, a tap on
/// padding skipped, the bias added last, so every value is the CPU's bit for
/// bit. `w` is `[c, kh, kw]`; `has_bias` 0 leaves the sum as it is.
kernel void ojas_dwconv(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant uint &total [[buffer(4)]],
    constant uint &c [[buffer(5)]],
    constant uint &h [[buffer(6)]],
    constant uint &wd [[buffer(7)]],
    constant uint &kh [[buffer(8)]],
    constant uint &kw [[buffer(9)]],
    constant uint &sh [[buffer(10)]],
    constant uint &sw [[buffer(11)]],
    constant uint &ph [[buffer(12)]],
    constant uint &pw [[buffer(13)]],
    constant uint &oh [[buffer(14)]],
    constant uint &ow [[buffer(15)]],
    constant uint &has_bias [[buffer(16)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint ox = i % ow;
    const uint oy = (i / ow) % oh;
    const uint ch = (i / (ow * oh)) % c;
    const uint n = i / (ow * oh * c);
    const ulong plane = ((ulong)n * c + ch) * h * wd;
    float acc = 0.0f;
    for (uint ky = 0u; ky < kh; ++ky) {
        const int iy = (int)(oy * sh + ky) - (int)ph;
        if (iy < 0 || iy >= (int)h) continue;
        for (uint kx = 0u; kx < kw; ++kx) {
            const int ix = (int)(ox * sw + kx) - (int)pw;
            if (ix < 0 || ix >= (int)wd) continue;
            acc += w[((ulong)ch * kh + ky) * kw + kx] * x[plane + (ulong)iy * wd + (uint)ix];
        }
    }
    y[i] = has_bias != 0u ? acc + bias[ch] : acc;
}

/// `conv_transpose2d_forward` with kernel == stride and no padding: the
/// product `prod[(n, y, x)][(o, i, j)]` of the NHWC input with the
/// `[c, o kh kw]` weight is the output pixel (y kh + i, x kw + j) of channel
/// o, since each output pixel has exactly one source pixel; the bias is
/// added last. y is [batch, out, h kh, w kw], one thread per element.
kernel void ojas_convt_nchw(
    device const float *prod [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &total [[buffer(3)]],
    constant uint &out_c [[buffer(4)]],
    constant uint &h [[buffer(5)]],
    constant uint &w [[buffer(6)]],
    constant uint &kh [[buffer(7)]],
    constant uint &kw [[buffer(8)]],
    constant uint &has_bias [[buffer(9)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint width = w * kw;
    const uint height = h * kh;
    const uint ox = i % width;
    const uint oy = (i / width) % height;
    const uint o = (i / (width * height)) % out_c;
    const uint n = i / (width * height * out_c);
    const ulong src_pixel = ((ulong)n * h + oy / kh) * w + ox / kw;
    const float v = prod[src_pixel * out_c * kh * kw + ((ulong)o * kh + oy % kh) * kw + ox % kw];
    y[i] = has_bias != 0u ? v + bias[o] : v;
}

// ------------------------------------------------------------- bilinear ---
//
// `Backend::interpolate_bilinear_forward`, torch's source-index rule in f32
// as `ojas-cpu/src/spatial.rs` forms it: the host computes each axis's
// `scale` with the CPU's own f32 expression, and every thread forms its two
// source indices and weights from it with the same IEEE operations (contract
// off), so each value is the CPU's bit for bit and no axis table exists.

/// One axis of an output index `o`: source indices `lo` and `hi` and the
/// weight `lambda` of `hi`. `size` >= 1.
inline void ojas_bilinear_tap(uint o, uint size, float scale, uint align,
                              thread uint &lo, thread uint &hi, thread float &lambda)
{
    const float src = align != 0u
        ? scale * (float)o
        : max(scale * ((float)o + 0.5f) - 0.5f, 0.0f);
    lo = min((uint)floor(src), size - 1u);
    lambda = clamp(src - (float)lo, 0.0f, 1.0f);
    hi = lo + ((lo + 1u < size) ? 1u : 0u);
}

/// y[plane][oy][ox] = (1 - ly) ((1 - lx) v00 + lx v01) + ly ((1 - lx) v10 + lx v11),
/// one thread per output element. A NaN or infinity in the output flags the
/// output word; the host scans the whole input (a pixel no output samples
/// included).
kernel void ojas_bilinear(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &total [[buffer(2)]],
    constant uint &h [[buffer(3)]],
    constant uint &w [[buffer(4)]],
    constant uint &oh [[buffer(5)]],
    constant uint &ow [[buffer(6)]],
    constant float &scale_y [[buffer(7)]],
    constant float &scale_x [[buffer(8)]],
    constant uint &align [[buffer(9)]],
    device atomic_uint *st [[buffer(10)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint ox = i % ow;
    const uint oy = (i / ow) % oh;
    const ulong plane = (ulong)(i / (ow * oh)) * h * w;
    uint y0, y1, x0, x1;
    float ly, lx;
    ojas_bilinear_tap(oy, h, scale_y, align, y0, y1, ly);
    ojas_bilinear_tap(ox, w, scale_x, align, x0, x1, lx);
    const float w0 = 1.0f - lx;
    const float w1 = lx;
    const float p00 = x[plane + (ulong)y0 * w + x0];
    const float p01 = x[plane + (ulong)y0 * w + x1];
    const float p10 = x[plane + (ulong)y1 * w + x0];
    const float p11 = x[plane + (ulong)y1 * w + x1];
    const float top = w0 * p00 + w1 * p01;
    const float bottom = w0 * p10 + w1 * p11;
    const float r = (1.0f - ly) * top + ly * bottom;
    y[i] = r;
    if (!isfinite(r)) ojas_flag(st, ST_OUT);
}

// ------------------------------------------------------------- group norm ---
//
// `Backend::group_norm_forward`: each (image, group) row is `per_group`
// channels of `spatial` values, contiguous. A row has far more values than one
// threadgroup should walk (0.66 to 2.65 million at SAM's pixel decoder, 8
// rows), so the statistics take two passes over `chunks` threadgroups per row
// and a third pass normalizes:
//
//   ojas_gn_sum    partial sums of each chunk
//   ojas_gn_sq     the row mean from the partial sums, then partial sums of
//                  d = x - mean and of d^2
//   ojas_gn_apply  the residual `r = sum(d) / n` of the first mean, the
//                  variance `(sum(d^2) - r * sum(d)) / n`, then
//                  (((x - mean0) - r) * rsqrt(var + eps)) * weight + bias
//
// The first mean `mean0` carries f32 summation error (a few 1e-6 to 1e-4 on a
// row of 2.65 million values near 1e3, by summation order); `sum(d)` is the
// residual of that error. The
// variance subtracts it, and the output subtracts it after `x - mean0`, which
// is exact for x near the mean, rather than folding it into one f32 mean (whose
// spacing, 6e-5 at 1e3, is itself 0.3% of a spread of 0.017). So the output is
// accurate where the CPU's f64 sums are, at any |mean| / std.
//
// Statistics are f32 (Metal has no f64): a thread sums a strided share of its
// chunk, a threadgroup reduces its threads in a fixed order, and a pass reads
// the chunk partials in ascending order, so a result repeats bit for bit. The
// CPU's are f64 sums rounded once; the two differ by f32 summation error.
// The operands are scanned by `ojas_check_finite` dispatches; a non-finite
// output, or a mean or variance that is not finite, sets ST_OUT. The variance
// is checked as computed, before it is clamped at 0 (a clamp would turn a
// -inf into 0 and every output into a finite, wrong value). The range differs
// from the CPU's, whose f64 sums finish where these do not: a row sum
// overflows f32 past abs(x) of 3.4e38 / n (1.3e32 at 2.65 million values a
// row, more for a shorter one), and `d * d` past abs(x - mean) of 1.8e19.
// Those calls fault here instead of returning the bias or a wrong value.
//
// tessl has no group norm, and `bert::layer_norm` does not fit: it runs one
// threadgroup per row, so SAM's 8 rows would use 8 threadgroups, and its
// affine is one weight per column of the row where GroupNorm's is one per
// channel over blocks of `spatial` values.
#define GN_THREADS 256u

inline float ojas_gn_block_sum(float v, threadgroup float *sh, uint tid, uint sg, uint lane)
{
    v = simd_sum(v);
    if (lane == 0u) sh[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (tid == 0u) {
        for (uint i = 0u; i < GN_THREADS / 32u; ++i) total += sh[i];
    }
    return total;
}

inline float ojas_gn_total(device const float *part, uint row, uint chunks)
{
    float total = 0.0f;
    for (uint c = 0u; c < chunks; ++c) total += part[(ulong)row * chunks + c];
    return total;
}

kernel void ojas_gn_sum(
    device const float *x [[buffer(0)]],
    device float *part [[buffer(1)]],
    constant uint &width [[buffer(2)]],
    constant uint &chunks [[buffer(3)]],
    constant uint &chunk_len [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float sh[GN_THREADS / 32u];
    const ulong base = (ulong)tg.y * width;
    const uint lo = min(width, tg.x * chunk_len);
    const uint hi = (uint)min((ulong)width, (ulong)lo + chunk_len);
    float acc = 0.0f;
    for (ulong i = (ulong)lo + tid; i < hi; i += GN_THREADS) acc += x[base + i];
    const float total = ojas_gn_block_sum(acc, sh, tid, sg, lane);
    if (tid == 0u) part[(ulong)tg.y * chunks + tg.x] = total;
}

kernel void ojas_gn_sq(
    device const float *x [[buffer(0)]],
    device const float *part_sum [[buffer(1)]],
    device float *part_dev [[buffer(2)]],
    device float *part_sq [[buffer(3)]],
    constant uint &width [[buffer(4)]],
    constant uint &chunks [[buffer(5)]],
    constant uint &chunk_len [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float sh_dev[GN_THREADS / 32u];
    threadgroup float sh_sq[GN_THREADS / 32u];
    threadgroup float mean_sh;
    if (tid == 0u) mean_sh = precise::divide(ojas_gn_total(part_sum, tg.y, chunks), (float)width);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = mean_sh;
    const ulong base = (ulong)tg.y * width;
    const uint lo = min(width, tg.x * chunk_len);
    const uint hi = (uint)min((ulong)width, (ulong)lo + chunk_len);
    float acc_dev = 0.0f;
    float acc_sq = 0.0f;
    for (ulong i = (ulong)lo + tid; i < hi; i += GN_THREADS) {
        const float d = x[base + i] - mean;
        acc_dev += d;
        acc_sq += d * d;
    }
    const float total_dev = ojas_gn_block_sum(acc_dev, sh_dev, tid, sg, lane);
    const float total_sq = ojas_gn_block_sum(acc_sq, sh_sq, tid, sg, lane);
    if (tid == 0u) {
        part_dev[(ulong)tg.y * chunks + tg.x] = total_dev;
        part_sq[(ulong)tg.y * chunks + tg.x] = total_sq;
    }
}

kernel void ojas_gn_apply(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device const float *part_sum [[buffer(3)]],
    device const float *part_dev [[buffer(4)]],
    device const float *part_sq [[buffer(5)]],
    device float *y [[buffer(6)]],
    device atomic_uint *st [[buffer(7)]],
    constant uint &width [[buffer(8)]],
    constant uint &chunks [[buffer(9)]],
    constant uint &chunk_len [[buffer(10)]],
    constant uint &spatial [[buffer(11)]],
    constant uint &per_group [[buffer(12)]],
    constant uint &groups [[buffer(13)]],
    constant float &eps [[buffer(14)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float stats[3];
    if (tid == 0u) {
        const float n = (float)width;
        const float mean0 = precise::divide(ojas_gn_total(part_sum, tg.y, chunks), n);
        const float sum_d = ojas_gn_total(part_dev, tg.y, chunks);
        const float sum_sq = ojas_gn_total(part_sq, tg.y, chunks);
        const float resid = precise::divide(sum_d, n);
        // sum(d)^2 / n is resid * sum(d): the same value, overflowing much later.
        const float var_raw = precise::divide(sum_sq - resid * sum_d, n);
        const float var = max(var_raw, 0.0f);
        stats[0] = mean0;
        stats[1] = precise::divide(1.0f, precise::sqrt(var + eps));
        stats[2] = resid;
        // A mean or variance that overflowed would turn every output into the
        // bias, finite and unflagged: refuse it. The variance is tested before
        // the clamp, which would hide a -inf and make a NaN unspecified.
        if (!isfinite(mean0 + resid) || !isfinite(var_raw)) ojas_flag(st, ST_OUT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean0 = stats[0];
    const float rstd = stats[1];
    const float resid = stats[2];
    const ulong base = (ulong)tg.y * width;
    const uint first = (tg.y % groups) * per_group;
    const uint lo = min(width, tg.x * chunk_len);
    const uint hi = (uint)min((ulong)width, (ulong)lo + chunk_len);
    bool bad = false;
    for (ulong i = (ulong)lo + tid; i < hi; i += GN_THREADS) {
        const uint ch = first + (uint)(i / spatial);
        const float r = (((x[base + i] - mean0) - resid) * rstd) * weight[ch] + bias[ch];
        y[base + i] = r;
        bad = bad || !isfinite(r);
    }
    if (bad) ojas_flag(st, ST_OUT);
}

// ---------------------------------------------- concat, slice and fill ---
//
// `Backend::concat`, `slice` and `fill`: values move as bits, no arithmetic.
// A copy kernel reads each element as a word and writes it to its place. The
// CPU refuses a non-finite input as NonFinite wherever it sits, so a concat
// kernel sets ST_IN on every element it copies (it copies the whole input),
// and a slice, which copies only part, has its whole input scanned by an
// `ojas_check_finite` dispatch first: the output holds only copied finite
// bits, so it needs no check.
//
//   ojas_concat_part  one input of a concat: element i of the input (rows of
//                     `row_in = len_i * inner`) lands in rows of `row_out =
//                     axis_total * inner` at column `offset = start_i * inner`
//   ojas_slice        element i of the output (rows of `row_out = len * inner`)
//                     reads the input's rows of `row_in = axis_len * inner` at
//                     `offset = start * inner`
//   ojas_fill_words   every word of the output is `word` (an f32 or u32 value,
//                     or a bf16 pattern in both halves)

kernel void ojas_concat_part(
    device const uint *x [[buffer(0)]],
    device uint *y [[buffer(1)]],
    constant uint &total [[buffer(2)]],
    constant uint &row_in [[buffer(3)]],
    constant uint &row_out [[buffer(4)]],
    constant uint &offset [[buffer(5)]],
    device atomic_uint *st [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint o = i / row_in;
    const uint r = i - o * row_in;
    const uint v = x[i];
    y[(ulong)o * row_out + offset + r] = v;
    if (!isfinite(as_type<float>(v))) ojas_flag(st, ST_IN);
}

kernel void ojas_slice(
    device const uint *x [[buffer(0)]],
    device uint *y [[buffer(1)]],
    constant uint &total [[buffer(2)]],
    constant uint &row_out [[buffer(3)]],
    constant uint &row_in [[buffer(4)]],
    constant uint &offset [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint o = i / row_out;
    const uint r = i - o * row_out;
    y[i] = x[(ulong)o * row_in + offset + r];
}

kernel void ojas_fill_words(
    device uint *y [[buffer(0)]],
    constant uint &words [[buffer(1)]],
    constant uint &word [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i < words) y[i] = word;
}

// ---------------------------------------- broadcast, scale and softmax ---
//
// Forward ops of the encoder family (see `Backend::broadcast_add_forward`,
// `scale_forward`, `softmax_forward`). Each sets status words 0 and 1 from
// the elements it reads and writes, like the other elementwise kernels.

#define BCAST_MAX_RANK 8u

/// y = a + b (`is_mul` 0) or a * b over the numpy broadcast of a and b: one
/// thread per output element, `oshape` the output shape and `astride` /
/// `bstride` each operand's element stride per output axis (0 where it is
/// stretched). One IEEE operation per element.
kernel void ojas_bcast_binary(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &rank [[buffer(4)]],
    constant uint &is_mul [[buffer(5)]],
    constant uint *oshape [[buffer(6)]],
    constant uint *astride [[buffer(7)]],
    constant uint *bstride [[buffer(8)]],
    device atomic_uint *st [[buffer(9)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n || rank > BCAST_MAX_RANK) return;
    uint rem = i;
    uint ai = 0u;
    uint bi = 0u;
    for (uint axis = rank; axis > 0u; --axis) {
        const uint extent = oshape[axis - 1u];
        const uint idx = rem % extent;
        rem /= extent;
        ai += idx * astride[axis - 1u];
        bi += idx * bstride[axis - 1u];
    }
    const float av = a[ai];
    const float bv = b[bi];
    const float r = (is_mul != 0u) ? (av * bv) : (av + bv);
    y[i] = r;
    if (!isfinite(av) || !isfinite(bv)) ojas_flag(st, ST_IN);
    if (!isfinite(r)) ojas_flag(st, ST_OUT);
}

/// y = x * scale.
kernel void ojas_scale_fwd(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant float &scale [[buffer(3)]],
    device atomic_uint *st [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float v = x[i];
    const float r = v * scale;
    y[i] = r;
    if (!isfinite(v)) ojas_flag(st, ST_IN);
    if (!isfinite(r)) ojas_flag(st, ST_OUT);
}

/// Softmax over each row of `dim`: e^(x - max) / sum, the sum ascending from
/// column 0 on the row's one thread, as the CPU's Exact path sums it.
kernel void ojas_softmax_rows(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    device atomic_uint *st [[buffer(4)]],
    uint r [[thread_position_in_grid]])
{
    if (r >= rows || dim == 0u) return;
    device const float *xr = x + (ulong)r * dim;
    device float *yr = y + (ulong)r * dim;
    float mx = xr[0];
    for (uint j = 1u; j < dim; ++j) {
        mx = max(mx, xr[j]);
    }
    float sum = 0.0f;
    for (uint j = 0u; j < dim; ++j) {
        const float v = xr[j];
        if (!isfinite(v)) ojas_flag(st, ST_IN);
        const float e = precise::exp(v - mx);
        yr[j] = e;
        sum += e;
    }
    for (uint j = 0u; j < dim; ++j) {
        const float p = yr[j] / sum;
        yr[j] = p;
        if (!isfinite(p)) ojas_flag(st, ST_OUT);
    }
}

// ---------------------------------------------------- causal attention ---
//
// FlashAttention-2, forward and backward, on the TensorOps matrix units,
// after tessl's `qwen35_attn_tiled.metal` and `qwen35_attn_bwd.metal` (head
// dim 256, [B, T, H, D]), here for ojas's [B, H, T, D] planes and head dims
// up to DM. Nothing T x T is stored. The backward rebuilds 32 by 32 score
// blocks. The forward rebuilds 32 by 64: a wider key tile, so its softmax
// and output rescale run half as often per query row.
//
//   S   = scale * Q Kᵀ        (query t sees keys t - W < j <= t; W = 0: 0..=t)
//   O   = softmax(S) V, online over key blocks,
//   lse = log sum_j exp(S_ij)                                    ojas_attn_fwd
//   Dr  = rowsum(dO ∘ O)                                     ojas_attn_bwd_dr
//   P   = exp(S - lse),  dP = dO Vᵀ,  dS = scale * P ∘ (dP - Dr)
//   dQ  = dS K                                                  ojas_attn_bwd_dq
//   dK  = dSᵀ Q,  dV = Pᵀ dO                                    ojas_attn_bwd_dkv
//
// The backward takes the forward's O and lse, so no pass rebuilds the row
// statistics. Each output row is written once by the threadgroup that owns
// it, with no atomics, so results repeat bit for bit.
//
// Grouped-query attention is native: query plane `bh` (b * H + h) reads KV
// plane `bh / rep` (b * Hkv + h / rep), with `rep = H / Hkv`. The dK/dV
// threadgroups own a KV plane and walk its `rep` query planes in increasing
// order, accumulating into the same tensors. Nothing is expanded.
//
// A sliding window `W > 0` keeps keys t - W < j <= t; key blocks wholly
// before a query block's window, and query blocks wholly after a key
// block's, are never visited.
//
// Q, K, V, dO are MPP tensors over the plane with extents (D, T): the
// matrix units read nothing past row T or column D. The score-side block is
// what is zeroed, not those operands: the probability P (forward, dV) and the
// score gradient dS (dQ, dK) of every entry outside the window or past T is
// set to exactly 0 before it multiplies V, K, Q or dO. A V row outside a
// query's window but inside a tile the kernel visits is still multiplied (as
// 0 * V), so a non-finite one reaches that query's O as 0 * NaN, and the
// host's output check reports it as a NonFinite fault, the one a V row inside
// the window gets. The backward checks every operand up front. DM is the
// compiled reduction width; D <= DM is the real head dimension. One (b, h)
// plane must fit i32 extents; the host checks.

using namespace mpp::tensor_ops;

#define ATT_BQ 32
#define ATT_BK 32
// Forward only. A 64-wide key tile halves how often the softmax and the
// output rescale run. The backward stays at 32: the wider tile was slower
// there on the bench shape.
#define ATT_FWD_BK 64
#define ATT_NSG 4
#define ATT_THREADS (ATT_NSG * 32)
// Four consecutive threads share a query row and own ATT_FWD_CPR scores
// each in the forward.
#define ATT_TPR 4
#define ATT_FWD_CPR (ATT_FWD_BK / ATT_TPR)

/// Key `j` is in query `i`'s window.
inline bool attn_live(uint i, uint j, uint W)
{
    return j <= i && (W == 0u || j + W > i);
}

/// The first `bk`-aligned key block any query from `q0` on can see.
inline uint attn_first_key(uint q0, uint W, uint bk)
{
    return (W == 0u || q0 + 1u <= W) ? 0u : ((q0 + 1u - W) / bk) * bk;
}

/// Dr for one query row per simdgroup: `rowsum(dO ∘ O)`, lanes striding the
/// head dimension and `simd_sum` combining them.
kernel void ojas_attn_bwd_dr(
    device const float *dout [[buffer(0)]], device const float *o [[buffer(1)]],
    device float *dvec [[buffer(2)]], constant uint &rows [[buffer(3)]],
    constant uint &D [[buffer(4)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = tgpig.x * (uint)ATT_NSG + sg;
    if (row >= rows) { return; }
    const ulong at = (ulong)row * D;
    float acc = 0.0f;
    for (uint i = lane; i < D; i += 32u) { acc += dout[at + i] * o[at + i]; }
    acc = simd_sum(acc);
    if (lane == 0u) { dvec[row] = acc; }
}

/// dQ for query rows [q0, q0 + BQ) of query plane `bh`: walk the key blocks
/// of its window, rebuild dS per block in threadgroup memory, accumulate
/// dS K.
template <int DM>
inline void attn_bwd_dq_body(
    device float *Q, device float *K, device float *V, device float *dO,
    device const float *lse, device const float *dvec, device float *dQ,
    uint T, uint D, uint rep, uint W, float scale, uint2 tgpig, uint tid,
    threadgroup float *S, threadgroup float *dP, threadgroup float *lse_row,
    threadgroup float *d_row)
{
    const uint q0 = tgpig.x * (uint)ATT_BQ;
    if (q0 >= T) { return; }
    const uint nq = min((uint)ATT_BQ, T - q0);
    const uint bh = tgpig.y;
    const ulong base = (ulong)bh * T * D;
    const ulong kv_base = (ulong)(bh / rep) * T * D;
    const uint t_end = q0 + nq;

    constexpr auto s_desc = matmul2d_descriptor(
        ATT_BQ, ATT_BK, DM, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto acc_desc = matmul2d_descriptor(
        ATT_BQ, DM, ATT_BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<s_desc, execution_simdgroups<ATT_NSG>> s_op;
    matmul2d<acc_desc, execution_simdgroups<ATT_NSG>> acc_op;

    auto mQ = tensor(Q + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mK = tensor(K + kv_base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mV = tensor(V + kv_base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mdO = tensor(dO + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto tQ = mQ.slice(0, (int)q0);
    auto tdO = mdO.slice(0, (int)q0);
    auto tS = tensor(S, dextents<int, 2>{ATT_BK, ATT_BQ}, array<int, 2>{1, ATT_BK});
    auto tdP = tensor(dP, dextents<int, 2>{ATT_BK, ATT_BQ}, array<int, 2>{1, ATT_BK});

    auto dq = acc_op.template get_destination_cooperative_tensor<
        decltype(tS), decltype(mK.slice(0, 0)), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < dq.get_capacity(); ++i) { dq[i] = 0.0f; }

    if (tid < (uint)ATT_BQ) {
        const bool live = tid < nq;
        lse_row[tid] = live ? lse[(ulong)bh * T + q0 + tid] : 0.0f;
        d_row[tid] = live ? dvec[(ulong)bh * T + q0 + tid] : 0.0f;
    }

    for (uint kb = attn_first_key(q0, W, ATT_BK); kb < t_end; kb += (uint)ATT_BK) {
        auto tK = mK.slice(0, (int)kb);
        auto tV = mV.slice(0, (int)kb);
        auto sT = s_op.template get_destination_cooperative_tensor<
            decltype(tQ), decltype(tK), float>();
        s_op.run(tQ, tK, sT);
        sT.store(tS);
        auto pT = s_op.template get_destination_cooperative_tensor<
            decltype(tdO), decltype(tV), float>();
        s_op.run(tdO, tV, pT);
        pT.store(tdP);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < (uint)(ATT_BQ * ATT_BK); i += ATT_THREADS) {
            const uint rr = i / (uint)ATT_BK, c = i % (uint)ATT_BK;
            const bool live = rr < nq && attn_live(q0 + rr, kb + c, W);
            float ds = 0.0f;
            if (live) {
                const float p = precise::exp(S[i] * scale - lse_row[rr]);
                ds = scale * (p * (dP[i] - d_row[rr]));
            }
            S[i] = ds;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc_op.run(tS, tK, dq);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto mdQ = tensor(dQ + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    dq.store(mdQ.slice(0, (int)q0));
}

/// dK and dV for key rows [k0, k0 + BK) of KV plane `g`: for each of its
/// `rep` query planes in increasing order, walk the query blocks from the
/// one holding k0 through the last query whose window holds a key of the
/// block, rebuild Pᵀ and dSᵀ per block in threadgroup memory, accumulate
/// Pᵀ dO and dSᵀ Q.
template <int DM>
inline void attn_bwd_dkv_body(
    device float *Q, device float *K, device float *V, device float *dO,
    device const float *lse, device const float *dvec, device float *dK, device float *dV,
    uint T, uint D, uint rep, uint W, float scale, uint2 tgpig, uint tid,
    threadgroup float *Pt, threadgroup float *dSt, threadgroup float *lse_col,
    threadgroup float *d_col)
{
    const uint k0 = tgpig.x * (uint)ATT_BK;
    if (k0 >= T) { return; }
    const uint nk = min((uint)ATT_BK, T - k0);
    const uint g = tgpig.y;
    const ulong kv_base = (ulong)g * T * D;
    const uint q_end = (W == 0u) ? T : min(T, k0 + nk - 1u + W);

    constexpr auto s_desc = matmul2d_descriptor(
        ATT_BK, ATT_BQ, DM, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto acc_desc = matmul2d_descriptor(
        ATT_BK, DM, ATT_BQ, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<s_desc, execution_simdgroups<ATT_NSG>> s_op;
    matmul2d<acc_desc, execution_simdgroups<ATT_NSG>> acc_op;

    auto mK = tensor(K + kv_base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mV = tensor(V + kv_base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto tK = mK.slice(0, (int)k0);
    auto tV = mV.slice(0, (int)k0);
    auto tPt = tensor(Pt, dextents<int, 2>{ATT_BQ, ATT_BK}, array<int, 2>{1, ATT_BQ});
    auto tdSt = tensor(dSt, dextents<int, 2>{ATT_BQ, ATT_BK}, array<int, 2>{1, ATT_BQ});

    auto dk = acc_op.template get_destination_cooperative_tensor<
        decltype(tPt), decltype(mK.slice(0, 0)), float>();
    auto dv = acc_op.template get_destination_cooperative_tensor<
        decltype(tPt), decltype(mK.slice(0, 0)), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < dk.get_capacity(); ++i) { dk[i] = 0.0f; dv[i] = 0.0f; }

    for (uint r = 0u; r < rep; ++r) {
        const uint bh = g * rep + r;
        const ulong base = (ulong)bh * T * D;
        auto mQ = tensor(Q + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
        auto mdO = tensor(dO + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
        for (uint qb = (k0 / (uint)ATT_BQ) * (uint)ATT_BQ; qb < q_end; qb += (uint)ATT_BQ) {
            const uint nq = min((uint)ATT_BQ, T - qb);
            if (tid < (uint)ATT_BQ) {
                const bool live = tid < nq;
                lse_col[tid] = live ? lse[(ulong)bh * T + qb + tid] : 0.0f;
                d_col[tid] = live ? dvec[(ulong)bh * T + qb + tid] : 0.0f;
            }
            auto tQ = mQ.slice(0, (int)qb);
            auto tdO = mdO.slice(0, (int)qb);
            auto sT = s_op.template get_destination_cooperative_tensor<
                decltype(tK), decltype(tQ), float>();
            s_op.run(tK, tQ, sT);
            sT.store(tPt);
            auto pT = s_op.template get_destination_cooperative_tensor<
                decltype(tV), decltype(tdO), float>();
            s_op.run(tV, tdO, pT);
            pT.store(tdSt);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < (uint)(ATT_BK * ATT_BQ); i += ATT_THREADS) {
                const uint c = i / (uint)ATT_BQ, rr = i % (uint)ATT_BQ;
                const bool live = c < nk && rr < nq && attn_live(qb + rr, k0 + c, W);
                float p = 0.0f;
                float ds = 0.0f;
                if (live) {
                    p = precise::exp(Pt[i] * scale - lse_col[rr]);
                    ds = scale * (p * (dSt[i] - d_col[rr]));
                }
                Pt[i] = p;
                dSt[i] = ds;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            acc_op.run(tPt, tdO, dv);
            acc_op.run(tdSt, tQ, dk);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    auto mdK = tensor(dK + kv_base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mdV = tensor(dV + kv_base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    dk.store(mdK.slice(0, (int)k0));
    dv.store(mdV.slice(0, (int)k0));
}

/// Forward for query rows [q0, q0 + BQ) of query plane `bh`, after tessl's
/// `qwen35_attn_tiled.metal`: S = Q Kᵀ per key block into threadgroup
/// memory, a per-row online softmax there (four threads per row,
/// `ATT_FWD_CPR` columns each), P overwrites S, and O = O · diag(alpha) +
/// P V accumulates in a cooperative tensor. Key blocks outside the block's
/// windows are not visited. A live score that is not finite sets ST_OUT and
/// counts as masked. Each live row's `m + log l` goes to LSE.
template <int DM>
inline void attn_fwd_tiled_body(
    device float *Q, device float *K, device float *V, device float *O, device float *LSE,
    device atomic_uint *st, uint T, uint D, uint rep, uint W, float scale, uint2 tgpig,
    uint tid, threadgroup float *S, threadgroup float *m_row, threadgroup float *l_row,
    threadgroup float *a_row)
{
    const uint q0 = tgpig.x * (uint)ATT_BQ;
    if (q0 >= T) { return; }
    const uint nq = min((uint)ATT_BQ, T - q0);
    const uint bh = tgpig.y;
    const ulong base = (ulong)bh * T * D;
    const ulong kv_base = (ulong)(bh / rep) * T * D;
    const uint t_end = q0 + nq;

    constexpr auto qk_desc = matmul2d_descriptor(
        ATT_BQ, ATT_FWD_BK, DM, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(
        ATT_BQ, DM, ATT_FWD_BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<ATT_NSG>> qk_op;
    matmul2d<pv_desc, execution_simdgroups<ATT_NSG>> pv_op;

    auto mQ = tensor(Q + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mK = tensor(K + kv_base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mV = tensor(V + kv_base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto tQ = mQ.slice(0, (int)q0);
    auto tP = tensor(S, dextents<int, 2>{ATT_FWD_BK, ATT_BQ}, array<int, 2>{1, ATT_FWD_BK});

    auto oT = pv_op.template get_destination_cooperative_tensor<
        decltype(tP), decltype(mV.slice(0, 0)), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < oT.get_capacity(); ++i) { oT[i] = 0.0f; }

    if (tid < (uint)ATT_BQ) {
        m_row[tid] = -INFINITY;
        l_row[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint r = tid / (uint)ATT_TPR;
    const uint c0 = (tid % (uint)ATT_TPR) * (uint)ATT_FWD_CPR;
    const bool row_live = r < nq;
    const uint qi = q0 + r;
    bool bad = false;

    for (uint kb = attn_first_key(q0, W, ATT_FWD_BK); kb < t_end; kb += (uint)ATT_FWD_BK) {
        auto tK = mK.slice(0, (int)kb);
        auto sT = qk_op.template get_destination_cooperative_tensor<
            decltype(tQ), decltype(tK), float>();
        qk_op.run(tQ, tK, sT);
        sT.store(tP);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float s[ATT_FWD_CPR];
        float mx = -INFINITY;
        for (uint c = 0; c < (uint)ATT_FWD_CPR; ++c) {
            const bool live = row_live && attn_live(qi, kb + c0 + c, W);
            const float v = S[r * ATT_FWD_BK + c0 + c] * scale;
            const bool ok = live && isfinite(v);
            bad = bad || (live && !ok);
            s[c] = ok ? v : -INFINITY;
            mx = max(mx, s[c]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float m_old = m_row[r];
        const float m_new = max(m_old, mx);
        // A row that has seen nothing has a zero accumulator: its rescale is
        // exactly 0, not exp(-inf - -inf).
        const float alpha = (m_old == -INFINITY) ? 0.0f : precise::exp(m_old - m_new);
        float sum = 0.0f;
        for (uint c = 0; c < (uint)ATT_FWD_CPR; ++c) {
            const float p = (s[c] == -INFINITY) ? 0.0f : precise::exp(s[c] - m_new);
            S[r * ATT_FWD_BK + c0 + c] = p;
            sum += p;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((tid % (uint)ATT_TPR) == 0u) {
            l_row[r] = l_row[r] * alpha + sum;
            m_row[r] = m_new;
            a_row[r] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < oT.get_capacity(); ++i) {
            if (oT.is_valid_element(i)) {
                const auto idx = oT.get_multidimensional_index(i);
                oT[i] *= a_row[idx[1]];
            }
        }
        auto tV = mV.slice(0, (int)kb);
        pv_op.run(tP, tV, oT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (bad) atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
    if (row_live && (tid % (uint)ATT_TPR) == 0u) {
        LSE[(ulong)bh * T + qi] = m_row[r] + precise::log(l_row[r]);
    }
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < oT.get_capacity(); ++i) {
        if (oT.is_valid_element(i)) {
            const auto idx = oT.get_multidimensional_index(i);
            const float l = l_row[idx[1]];
            oT[i] *= (l > 0.0f) ? precise::divide(1.0f, l) : 0.0f;
        }
    }
    auto mO = tensor(O + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    oT.store(mO.slice(0, (int)q0));
}

/* MPP's type matching rejects const element types; nothing here writes Q,
   K, V or dO. */
#define ATT_INPUTS                                                            \
    const_cast<device float *>(q), const_cast<device float *>(k),             \
    const_cast<device float *>(v), const_cast<device float *>(dout)

#define OJAS_ATTN_TILED(DM)                                                   \
kernel void ojas_attn_fwd_d##DM(                                              \
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], \
    device const float *v [[buffer(2)]], device float *o [[buffer(3)]],       \
    device float *lse [[buffer(4)]], device atomic_uint *st [[buffer(5)]],    \
    constant uint &T [[buffer(6)]], constant uint &D [[buffer(7)]],           \
    constant uint &rep [[buffer(8)]], constant uint &W [[buffer(9)]],         \
    constant float &scale [[buffer(10)]],                                     \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]])                                 \
{                                                                             \
    threadgroup float S[ATT_BQ * ATT_FWD_BK];                                 \
    threadgroup float m_row[ATT_BQ];                                          \
    threadgroup float l_row[ATT_BQ];                                          \
    threadgroup float a_row[ATT_BQ];                                          \
    attn_fwd_tiled_body<DM>(const_cast<device float *>(q),                    \
                            const_cast<device float *>(k),                    \
                            const_cast<device float *>(v), o, lse, st, T, D,  \
                            rep, W, scale, tgpig, tid, S, m_row, l_row,       \
                            a_row);                                           \
}                                                                             \
kernel void ojas_attn_bwd_dq_d##DM(                                           \
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], \
    device const float *v [[buffer(2)]], device const float *dout [[buffer(3)]], \
    device const float *lse [[buffer(4)]], device const float *dvec [[buffer(5)]], \
    device float *dq [[buffer(6)]], constant uint &T [[buffer(7)]],           \
    constant uint &D [[buffer(8)]], constant uint &rep [[buffer(9)]],         \
    constant uint &W [[buffer(10)]], constant float &scale [[buffer(11)]],    \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]])                                 \
{                                                                             \
    threadgroup float S[ATT_BQ * ATT_BK];                                     \
    threadgroup float dP[ATT_BQ * ATT_BK];                                    \
    threadgroup float r0[ATT_BQ];                                             \
    threadgroup float r1[ATT_BQ];                                             \
    attn_bwd_dq_body<DM>(ATT_INPUTS, lse, dvec, dq, T, D, rep, W, scale,      \
                         tgpig, tid, S, dP, r0, r1);                          \
}                                                                             \
kernel void ojas_attn_bwd_dkv_d##DM(                                          \
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], \
    device const float *v [[buffer(2)]], device const float *dout [[buffer(3)]], \
    device const float *lse [[buffer(4)]], device const float *dvec [[buffer(5)]], \
    device float *dk [[buffer(6)]], device float *dv [[buffer(7)]],           \
    constant uint &T [[buffer(8)]], constant uint &D [[buffer(9)]],           \
    constant uint &rep [[buffer(10)]], constant uint &W [[buffer(11)]],       \
    constant float &scale [[buffer(12)]],                                     \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]])                                 \
{                                                                             \
    threadgroup float Pt[ATT_BQ * ATT_BK];                                    \
    threadgroup float dSt[ATT_BQ * ATT_BK];                                   \
    threadgroup float c0[ATT_BQ];                                             \
    threadgroup float c1[ATT_BQ];                                             \
    attn_bwd_dkv_body<DM>(ATT_INPUTS, lse, dvec, dk, dv, T, D, rep, W, scale, \
                          tgpig, tid, Pt, dSt, c0, c1);                       \
}

OJAS_ATTN_TILED(16)
OJAS_ATTN_TILED(32)
OJAS_ATTN_TILED(64)
OJAS_ATTN_TILED(128)
OJAS_ATTN_TILED(256)

// ------------------------------------------ general attention (forward) ---
//
// The contract of `Backend::attention_forward_saving`: bidirectional or causal
// attention, a query length that differs from the key length, an additive
// score bias and per-batch key lengths. One simdgroup per query row, after
// `ojas_cached_attn`: lane `l` owns head dims l, l + 32, ... of the query and
// of the output, the score is a `simd_sum` (whose order is fixed), and the
// softmax is online over the keys the row sees, so results repeat bit for
// bit. Scalar f32 on purpose: it takes any head dim up to 32 * AG_PER_LANE and
// is the reference a tiled kernel is held to; it is not the fast path.
//
//   S   = scale * Q Kᵀ + bias,   O = softmax(S) V,   lse = log sum_j exp(S_ij)
//
//   flags  bit 0 causal   bit 1 a bias is bound   bit 2 key lengths are bound
//   causal: query i sits at position pos = i + Tk - Tq and sees keys
//           pos - W < j <= pos (W = 0: every earlier key); pos < 0 sees none
//   lens:   keys j < lens[b], clamped to Tk (0 hides every key of that batch)
//   bias:   [bias_b, bias_h, bias_q, Tk], an extent of 1 broadcasts
//
// Grouped-query heads read their KV plane in place: query plane bh = b * H + h
// reads KV plane b * (H / rep) + h / rep. A row that sees no key writes zeros
// and an lse of 0, as the CPU does. A score that is not finite is skipped and
// sets ST_OUT.
#define AG_PER_LANE 8u
#define AG_ROWS 4u

kernel void ojas_attn_gen_fwd(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *o [[buffer(3)]],
    device float *lse [[buffer(4)]],
    device const float *bias [[buffer(5)]],
    device const uint *lens [[buffer(6)]],
    device atomic_uint *st [[buffer(7)]],
    constant uint &Tq [[buffer(8)]],
    constant uint &Tk [[buffer(9)]],
    constant uint &D [[buffer(10)]],
    constant uint &H [[buffer(11)]],
    constant uint &rep [[buffer(12)]],
    constant uint &flags [[buffer(13)]],
    constant uint &W [[buffer(14)]],
    constant uint &bias_b [[buffer(15)]],
    constant uint &bias_h [[buffer(16)]],
    constant uint &bias_q [[buffer(17)]],
    constant float &scale [[buffer(18)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    // The whole simdgroup leaves together: `i` depends on the simdgroup only.
    const uint i = tg.x * AG_ROWS + sg;
    if (i >= Tq) return;
    const uint bh = tg.y;
    const uint b = bh / H;
    const uint h = bh - b * H;
    const ulong q_at = ((ulong)bh * Tq + i) * D;
    const ulong kv_base = ((ulong)b * (H / rep) + h / rep) * Tk * D;

    uint hi = Tk;
    if ((flags & 4u) != 0u) hi = min(hi, lens[b]);
    uint lo = 0u;
    if ((flags & 1u) != 0u) {
        const int pos = (int)i + (int)Tk - (int)Tq;
        if (pos < 0) {
            hi = 0u;
        } else {
            hi = min(hi, (uint)pos + 1u);
            if (W != 0u && (uint)pos + 1u > W) lo = (uint)pos + 1u - W;
        }
    }

    float qv[AG_PER_LANE];
    float acc[AG_PER_LANE];
    for (uint t = 0u; t < AG_PER_LANE; ++t) {
        const uint c = lane + 32u * t;
        qv[t] = (c < D) ? q[q_at + c] : 0.0f;
        acc[t] = 0.0f;
    }
    ulong bias_at = 0;
    if ((flags & 2u) != 0u) {
        const ulong plane = (ulong)(bias_b == 1u ? 0u : b) * bias_h + (bias_h == 1u ? 0u : h);
        bias_at = (plane * bias_q + (bias_q == 1u ? 0u : i)) * Tk;
    }

    float m = -INFINITY;
    float l = 0.0f;
    bool bad = false;
    for (uint j = lo; j < hi; ++j) {
        const ulong kv_at = kv_base + (ulong)j * D;
        float part = 0.0f;
        for (uint t = 0u; t < AG_PER_LANE; ++t) {
            const uint c = lane + 32u * t;
            if (c < D) part += qv[t] * k[kv_at + c];
        }
        float s = simd_sum(part) * scale;
        if ((flags & 2u) != 0u) s += bias[bias_at + j];
        // `s` is the same on every lane, so this branch is uniform.
        if (!isfinite(s)) {
            bad = true;
            continue;
        }
        const float m_new = max(m, s);
        const float alpha = (m == -INFINITY) ? 0.0f : precise::exp(m - m_new);
        const float p = precise::exp(s - m_new);
        l = l * alpha + p;
        for (uint t = 0u; t < AG_PER_LANE; ++t) {
            const uint c = lane + 32u * t;
            if (c < D) acc[t] = acc[t] * alpha + p * v[kv_at + c];
        }
        m = m_new;
    }
    if (bad && lane == 0u) atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
    const float inv = (l > 0.0f) ? precise::divide(1.0f, l) : 0.0f;
    for (uint t = 0u; t < AG_PER_LANE; ++t) {
        const uint c = lane + 32u * t;
        if (c < D) o[q_at + c] = acc[t] * inv;
    }
    if (lane == 0u) lse[(ulong)bh * Tq + i] = (l > 0.0f) ? m + precise::log(l) : 0.0f;
}

// ----------------------------- general attention, split-K tiles (D 16, 32) ---
//
// The same contract as `ojas_attn_gen_fwd` (S = scale Q Kᵀ + bias, softmax
// over the keys a row sees, lse = log sum exp, a row that sees no key writes
// zeros and lse 0, a non-finite score is skipped and sets ST_OUT), without
// causal and window, in two kernels:
//
//   ojas_xattn_part_dD    one thread per query row, XA_ROWS rows to a threadgroup.
//                         K and V go through threadgroup memory XA_TILE keys at a
//                         time, shared by every row, and each thread walks its
//                         tile keeping a running max, sum and output; the keys
//                         are cut into `splits` ranges of `per_split`, one
//                         threadgroup row each, so a short Q over a long K still
//                         fills the GPU. A split writes (m, l, acc[D]) per row.
//   ojas_xattn_combine_dD one thread per row merges its splits by their maxima:
//                         M = max m_s, L = sum l_s e^(m_s - M), O = sum acc_s
//                         e^(m_s - M) / L, lse = M + log L.
//
// The score is a sequential f32 dot over D, not the scalar kernel's lane sums,
// so values agree with it to rounding, not bit for bit. Grid y of the first
// kernel is `bh * splits + split`.
#define XA_ROWS 128u
#define XA_TILE 64u

#define OJAS_XATTN(DD)                                                          \
kernel void ojas_xattn_part_d##DD(                                              \
    device const float *q [[buffer(0)]],                                        \
    device const float *k [[buffer(1)]],                                        \
    device const float *v [[buffer(2)]],                                        \
    device float *part [[buffer(3)]],                                           \
    device const float *bias [[buffer(4)]],                                     \
    device const uint *lens [[buffer(5)]],                                      \
    device atomic_uint *st [[buffer(6)]],                                       \
    constant uint &Tq [[buffer(7)]],                                            \
    constant uint &Tk [[buffer(8)]],                                            \
    constant uint &H [[buffer(9)]],                                             \
    constant uint &rep [[buffer(10)]],                                          \
    constant uint &flags [[buffer(11)]],                                        \
    constant uint &bias_b [[buffer(12)]],                                       \
    constant uint &bias_h [[buffer(13)]],                                       \
    constant uint &bias_q [[buffer(14)]],                                       \
    constant uint &splits [[buffer(15)]],                                       \
    constant uint &per_split [[buffer(16)]],                                    \
    constant uint &planes [[buffer(17)]],                                       \
    constant float &scale [[buffer(18)]],                                       \
    uint2 tg [[threadgroup_position_in_grid]],                                  \
    uint tid [[thread_index_in_threadgroup]])                                   \
{                                                                               \
    threadgroup float ks[XA_TILE * DD];                                         \
    threadgroup float vs[XA_TILE * DD];                                         \
    const uint bh = tg.y / splits;                                              \
    const uint split = tg.y - bh * splits;                                      \
    const uint b = bh / H;                                                      \
    const uint h = bh - b * H;                                                  \
    const uint row = tg.x * XA_ROWS + tid;                                      \
    const bool live = row < Tq;                                                 \
    const ulong kv_base = ((ulong)b * (H / rep) + h / rep) * Tk * DD;           \
    uint hi = Tk;                                                               \
    if ((flags & 4u) != 0u) hi = min(hi, lens[b]);                              \
    const uint lo = min(hi, split * per_split);                                 \
    const uint end = min(hi, lo + per_split);                                   \
    float qv[DD];                                                               \
    float acc[DD];                                                              \
    for (uint c = 0u; c < DD; ++c) {                                            \
        qv[c] = live ? q[((ulong)bh * Tq + row) * DD + c] : 0.0f;               \
        acc[c] = 0.0f;                                                          \
    }                                                                           \
    ulong bias_at = 0;                                                          \
    if ((flags & 2u) != 0u && live) {                                           \
        const ulong plane = (ulong)(bias_b == 1u ? 0u : b) * bias_h             \
            + (bias_h == 1u ? 0u : h);                                          \
        bias_at = (plane * bias_q + (bias_q == 1u ? 0u : row)) * Tk;            \
    }                                                                           \
    float m = -INFINITY;                                                        \
    float l = 0.0f;                                                             \
    bool bad = false;                                                           \
    for (uint base = lo; base < end; base += XA_TILE) {                         \
        const uint n = min(XA_TILE, end - base);                                \
        threadgroup_barrier(mem_flags::mem_threadgroup);                        \
        for (uint e = tid; e < n * DD; e += XA_ROWS) {                          \
            ks[e] = k[kv_base + (ulong)base * DD + e];                          \
            vs[e] = v[kv_base + (ulong)base * DD + e];                          \
        }                                                                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                        \
        if (live) {                                                             \
            for (uint j = 0u; j < n; ++j) {                                     \
                float dot = 0.0f;                                               \
                for (uint c = 0u; c < DD; ++c) dot += qv[c] * ks[j * DD + c];   \
                float sc = dot * scale;                                         \
                if ((flags & 2u) != 0u) sc += bias[bias_at + base + j];         \
                if (!isfinite(sc)) {                                            \
                    bad = true;                                                 \
                    continue;                                                   \
                }                                                               \
                const float m_new = max(m, sc);                                 \
                const float alpha = (m == -INFINITY) ? 0.0f                     \
                                                     : precise::exp(m - m_new); \
                const float p = precise::exp(sc - m_new);                       \
                l = l * alpha + p;                                              \
                for (uint c = 0u; c < DD; ++c)                                  \
                    acc[c] = acc[c] * alpha + p * vs[j * DD + c];               \
                m = m_new;                                                      \
            }                                                                   \
        }                                                                       \
    }                                                                           \
    if (bad) atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);      \
    if (live) {                                                                 \
        const ulong at = (((ulong)split * planes + bh) * Tq + row) * (DD + 2u); \
        part[at] = m;                                                           \
        part[at + 1u] = l;                                                      \
        for (uint c = 0u; c < DD; ++c) part[at + 2u + c] = acc[c];              \
    }                                                                           \
}                                                                               \
kernel void ojas_xattn_combine_d##DD(                                           \
    device const float *part [[buffer(0)]],                                     \
    device float *o [[buffer(1)]],                                              \
    device float *lse [[buffer(2)]],                                            \
    constant uint &rows [[buffer(3)]],                                          \
    constant uint &splits [[buffer(4)]],                                        \
    uint r [[thread_position_in_grid]])                                         \
{                                                                               \
    if (r >= rows) return;                                                      \
    float top = -INFINITY;                                                      \
    for (uint s = 0u; s < splits; ++s)                                          \
        top = max(top, part[((ulong)s * rows + r) * (DD + 2u)]);                \
    float total = 0.0f;                                                         \
    float acc[DD];                                                              \
    for (uint c = 0u; c < DD; ++c) acc[c] = 0.0f;                               \
    if (top != -INFINITY) {                                                     \
        for (uint s = 0u; s < splits; ++s) {                                    \
            const ulong at = ((ulong)s * rows + r) * (DD + 2u);                 \
            const float ms = part[at];                                          \
            if (ms == -INFINITY) continue;                                      \
            const float w = precise::exp(ms - top);                             \
            total += part[at + 1u] * w;                                         \
            for (uint c = 0u; c < DD; ++c) acc[c] += part[at + 2u + c] * w;     \
        }                                                                       \
    }                                                                           \
    const float inv = (total > 0.0f) ? precise::divide(1.0f, total) : 0.0f;     \
    for (uint c = 0u; c < DD; ++c) o[(ulong)r * DD + c] = acc[c] * inv;         \
    lse[r] = (total > 0.0f) ? top + precise::log(total) : 0.0f;                 \
}

OJAS_XATTN(16)
OJAS_XATTN(32)
