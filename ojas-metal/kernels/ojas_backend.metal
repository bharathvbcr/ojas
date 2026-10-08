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
// ids `ojas_argmax_rows` produced are below its column count by construction,
// which the host checks against the embedding's vocabulary instead.
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
    if (mode == 1u && (heads == 0u || time == 0u)) return;
    const ulong base = (ulong)row * dim;
    if (col >= half_rot) {
        const ulong c = base + rot + (col - half_rot);
        y[c] = x[c];
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
// `clip_grad_norm` over many gradients in few dispatches. One dispatch binds
// up to NORM_SLOTS gradients at buffers 0..23; `tbl` holds each slot's
// element count and the dispatch-local index of its first chunk, so
// threadgroup g finds its slot and its NORM_CHUNK-element chunk without a
// per-tensor launch. The slot choice is uniform across a threadgroup.

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

/// One pass over each chunk: its max |g| m and its sum of (g / m)^2 (0 when
/// m is 0), both kept so the finish can rescale every chunk to the global
/// max with no f32 overflow while the norm is finite. A non-finite element
/// sets st[ST_IN] and is left out of both. part[g] = (m, sum).
kernel void ojas_norm_multi(
    device const float *g0 [[buffer(0)]],
    device const float *g1 [[buffer(1)]],
    device const float *g2 [[buffer(2)]],
    device const float *g3 [[buffer(3)]],
    device const float *g4 [[buffer(4)]],
    device const float *g5 [[buffer(5)]],
    device const float *g6 [[buffer(6)]],
    device const float *g7 [[buffer(7)]],
    device const float *g8 [[buffer(8)]],
    device const float *g9 [[buffer(9)]],
    device const float *g10 [[buffer(10)]],
    device const float *g11 [[buffer(11)]],
    device const float *g12 [[buffer(12)]],
    device const float *g13 [[buffer(13)]],
    device const float *g14 [[buffer(14)]],
    device const float *g15 [[buffer(15)]],
    device const float *g16 [[buffer(16)]],
    device const float *g17 [[buffer(17)]],
    device const float *g18 [[buffer(18)]],
    device const float *g19 [[buffer(19)]],
    device const float *g20 [[buffer(20)]],
    device const float *g21 [[buffer(21)]],
    device const float *g22 [[buffer(22)]],
    device const float *g23 [[buffer(23)]],
    device float2 *part [[buffer(24)]],
    device atomic_uint *st [[buffer(25)]],
    constant uint *tbl [[buffer(26)]],
    constant uint &used [[buffer(27)]],
    uint g [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float sh[NORM_TG];
    const uint s = norm_slot(tbl, used, g);
    const uint n = tbl[2u * s];
    const ulong lo = (ulong)(g - tbl[2u * s + 1u]) * NORM_CHUNK;
    device const float *x = NORM_PICK(s);
    float v[NORM_PER];
    float peak = 0.0f;
    bool bad = false;
    for (uint j = 0u; j < NORM_PER; ++j) {
        const ulong i = lo + (ulong)(j * NORM_TG + lane);
        float a = i < n ? x[i] : 0.0f;
        if (!isfinite(a)) {
            bad = true;
            a = 0.0f;
        }
        v[j] = a;
        peak = max(peak, fabs(a));
    }
    if (bad) atomic_store_explicit(&st[ST_IN], 1u, memory_order_relaxed);
    sh[lane] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = NORM_TG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) sh[lane] = max(sh[lane], sh[lane + off]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float m = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    if (m > 0.0f) {
        for (uint j = 0u; j < NORM_PER; ++j) {
            const float t = precise::divide(v[j], m);
            acc += t * t;
        }
    }
    sh[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = NORM_TG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) sh[lane] = sh[lane] + sh[lane + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) part[g] = float2(m, sh[0]);
}

/// One threadgroup folds `count` chunk partials: M = max m, then
/// total = sum sum_c * (m_c / M)^2, out[0] = M * sqrt(total), out[1] = M.
kernel void ojas_norm_finish(
    device const float2 *part [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float sh[NORM_TG];
    float peak = 0.0f;
    for (uint i = lane; i < count; i += NORM_TG) peak = max(peak, part[i].x);
    sh[lane] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = NORM_TG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) sh[lane] = max(sh[lane], sh[lane + off]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float big = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    if (big > 0.0f) {
        for (uint i = lane; i < count; i += NORM_TG) {
            const float r = precise::divide(part[i].x, big);
            acc += part[i].y * (r * r);
        }
    }
    sh[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = NORM_TG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) sh[lane] = sh[lane] + sh[lane + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        out[0] = big * precise::sqrt(sh[0]);
        out[1] = big;
    }
}

/// Every gradient of the dispatch times `scale`, in place; a non-finite
/// product sets st[ST_OUT]. Same slot table as `ojas_norm_multi`.
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

/// cache[b, at + t, :, :] = src[b, t, :, :], written only if no status word is
/// set (the source's finite check runs before it in the same command buffer).
/// `span` is Tn * Hkv * D, `cap_span` Tcap * Hkv * D and `at_off` at * Hkv * D.
kernel void ojas_kv_write(
    device const float *src [[buffer(0)]],
    device float *cache [[buffer(1)]],
    device const uint *st [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &span [[buffer(4)]],
    constant uint &cap_span [[buffer(5)]],
    constant uint &at_off [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (st[ST_IN] != 0u || st[ST_OUT] != 0u) {
        return;
    }
    const uint b = i / span;
    const uint r = i - b * span;
    cache[(ulong)b * cap_span + at_off + r] = src[i];
}

#define CA_SG 32u
#define CA_THREADS (CA_SG * 32u)
#define CA_MAX_D 256u
// A lane holds dims lane, lane+32, ... (eight slots cover 256). The
// simdgroup output is staged 128 dims at a time so `so` stays 16 KiB.
#define CA_PER_LANE 8u
#define CA_CHUNK 128u

/// Causal attention of Tq queries against the first kv_len cache positions,
/// grouped-query: head h reads KV head h / (H / Hkv). One threadgroup per
/// (query, head, split) and batch, x = (i * H + h) * splits + s; query i
/// sits at position kv_len - Tq + i.
///
/// Split s walks keys [s * chunk, min((s + 1) * chunk, pos + 1)). Each of
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
/// and every cache element at a position below kv_len is read by the last
/// query row of each head that maps to its KV head, whose splits cover
/// [0, kv_len), so a NaN or infinity there sets ST_IN. Positions at or past
/// kv_len are not read and not checked. A non-finite output (scores that
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
    const uint j0 = split * chunk;
    const uint j1 = min(j0 + chunk, pos + 1u);
    for (uint j = j0 + sg; j < j1; j += CA_SG) {
        const ulong kb = (((ulong)b * cap + j) * kv_heads + kh) * d;
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
// matrix units read nothing past row T or column D, and every block entry
// outside the window or past T is set to exactly 0 before it is
// multiplied. DM is the compiled reduction width; D <= DM is the real head
// dimension. One (b, h) plane must fit i32 extents; the host checks.

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
