// Kernels behind `MetalBackend` for the ops tessl has no kernel for.
//
// Every kernel takes element counts and guards its own indices. The host
// refuses any tensor above u32::MAX elements before it binds one, so `uint`
// indices cannot wrap; products that can pass u32 are formed in `ulong`.
//
// Status words, one `atomic_uint` array per op:
//   0  an input is non-finite        1  an output or intermediate is non-finite
//   2  an index is out of range      3  smallest out-of-range position
//   4  valid-row count (cross-entropy)
//
// Compiled with -ffp-contract=off and -fmetal-math-mode=safe; transcendental
// calls are `precise::`.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define ST_IN 0u
#define ST_OUT 1u
#define ST_RANGE 2u
#define ST_FIRST 3u
#define ST_COUNT 4u
#define ST_WORDS 8u

kernel void ojas_status_init(
    device atomic_uint *st [[buffer(0)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= ST_WORDS) return;
    atomic_store_explicit(&st[i], i == ST_FIRST ? 0xFFFFFFFFu : 0u, memory_order_relaxed);
}

kernel void ojas_check_finite(
    device const float *x [[buffer(0)]],
    device atomic_uint *st [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &word [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n || word >= ST_WORDS) return;
    if (!isfinite(x[i])) {
        atomic_store_explicit(&st[word], 1u, memory_order_relaxed);
    }
}

/// Ids at or above `limit` set ST_RANGE and lower ST_FIRST. With
/// `count_valid`, every id that is neither ignored nor out of range adds one
/// to ST_COUNT.
kernel void ojas_check_ids(
    device const uint *ids [[buffer(0)]],
    device atomic_uint *st [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &limit [[buffer(3)]],
    constant uint &has_ignore [[buffer(4)]],
    constant uint &ignore [[buffer(5)]],
    constant uint &count_valid [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const uint id = ids[i];
    if (has_ignore != 0u && id == ignore) return;
    if (id >= limit) {
        atomic_store_explicit(&st[ST_RANGE], 1u, memory_order_relaxed);
        atomic_fetch_min_explicit(&st[ST_FIRST], i, memory_order_relaxed);
        return;
    }
    if (count_valid != 0u) {
        atomic_fetch_add_explicit(&st[ST_COUNT], 1u, memory_order_relaxed);
    }
}

kernel void ojas_fill_u32(
    device uint *x [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    constant uint &value [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    x[i] = value;
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

kernel void ojas_silu_fwd(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float v = x[i];
    y[i] = v * ojas_sigmoid_ref(v);
}

kernel void ojas_silu_bwd(
    device const float *x [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float v = x[i];
    const float s = ojas_sigmoid_ref(v);
    out[i] = g[i] * s * (1.0f + v * (1.0f - s));
}

kernel void ojas_mul_fwd(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    y[i] = a[i] * b[i];
}

kernel void ojas_mul_bwd(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device const float *g [[buffer(2)]],
    device float *ga [[buffer(3)]],
    device float *gb [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    ga[i] = g[i] * b[i];
    gb[i] = g[i] * a[i];
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

kernel void ojas_scale(
    device float *x [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    constant float &scale [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    x[i] = x[i] * scale;
}

/// y = (1 - s) * v + s * v0 with s = sigmoid(lambda[0]).
kernel void ojas_vres_fwd(
    device const float *v [[buffer(0)]],
    device const float *v0 [[buffer(1)]],
    device const float *lam [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float s = ojas_sigmoid_ref(lam[0]);
    y[i] = (1.0f - s) * v[i] + s * v0[i];
}

/// gv = (1 - s) g, gv0 = s g, and the lambda term t = (v0 - v) g, which a
/// reduction sums.
kernel void ojas_vres_bwd(
    device const float *v [[buffer(0)]],
    device const float *v0 [[buffer(1)]],
    device const float *lam [[buffer(2)]],
    device const float *g [[buffer(3)]],
    device float *gv [[buffer(4)]],
    device float *gv0 [[buffer(5)]],
    device float *t [[buffer(6)]],
    constant uint &n [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float s = ojas_sigmoid_ref(lam[0]);
    const float gy = g[i];
    gv[i] = (1.0f - s) * gy;
    gv0[i] = s * gy;
    t[i] = (v0[i] - v[i]) * gy;
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

/// Half-split RoPE. `mode` 0: cos/sin have the shape of x. `mode` 1: x is
/// [batch, time, heads, dim] and cos/sin are [time, dim].
/// Forward:  y1 = x1 c1 + (-x2) s1,  y2 = x2 c2 + x1 s2.
/// Backward: g1' = g1 c1 + g2 s2,    g2' = -g1 s1 + g2 c2.
/// Grid: x = column in [0, dim/2), y = row.
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
    uint2 gid [[thread_position_in_grid]])
{
    const uint half_dim = dim / 2u;
    const uint col = gid.x;
    const uint row = gid.y;
    if (col >= half_dim || row >= rows) return;
    if (mode == 1u && (heads == 0u || time == 0u)) return;
    const ulong base = (ulong)row * dim;
    const ulong cbase = mode == 0u ? base : (ulong)((row / heads) % time) * dim;
    const float a = x[base + col];
    const float b = x[base + col + half_dim];
    const float c1 = cs[cbase + col];
    const float s1 = sn[cbase + col];
    const float c2 = cs[cbase + col + half_dim];
    const float s2 = sn[cbase + col + half_dim];
    if (backward == 0u) {
        y[base + col] = a * c1 + (-b) * s1;
        y[base + col + half_dim] = b * c2 + a * s2;
    } else {
        y[base + col] = a * c1 + b * s2;
        y[base + col + half_dim] = -a * s1 + b * c2;
    }
}

// ------------------------------------------------------------ embedding ---

/// Grid: x = column, y = token. An out-of-range id writes 0; ojas_check_ids
/// has already flagged it and the op returns an error.
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

kernel void ojas_embed_count(
    device const uint *ids [[buffer(0)]],
    device atomic_uint *counts [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &vocab [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const uint id = ids[i];
    if (id < vocab) {
        atomic_fetch_add_explicit(&counts[id], 1u, memory_order_relaxed);
    }
}

#define SCAN_TG 1024u

/// Exclusive prefix sum of `counts` into `starts`. One threadgroup of
/// SCAN_TG threads; each owns a contiguous chunk.
kernel void ojas_scan_exclusive(
    device const uint *counts [[buffer(0)]],
    device uint *starts [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup uint sh[SCAN_TG];
    const uint chunk = (n + SCAN_TG - 1u) / SCAN_TG;
    const ulong lo = (ulong)lane * chunk;
    const ulong hi = min(lo + chunk, (ulong)n);
    uint local = 0u;
    for (ulong i = lo; i < hi; ++i) {
        local += counts[i];
    }
    sh[lane] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1u; off < SCAN_TG; off <<= 1u) {
        const uint add = lane >= off ? sh[lane - off] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sh[lane] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint run = sh[lane] - local;
    for (ulong i = lo; i < hi; ++i) {
        starts[i] = run;
        run += counts[i];
    }
}

#define PLACE_TG 256u

/// pos[starts[id] + rank] = i, where rank counts earlier tokens with the same
/// id. Each id's rows therefore sit in ascending token order, which is the
/// order the CPU reference adds them.
kernel void ojas_embed_place(
    device const uint *ids [[buffer(0)]],
    device const uint *starts [[buffer(1)]],
    device uint *pos [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &vocab [[buffer(4)]],
    uint i [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    threadgroup uint tile[PLACE_TG];
    const bool live = i < n;
    const uint id = live ? ids[i] : 0xFFFFFFFFu;
    uint rank = 0u;
    // Earlier tokens only; the last tile is this threadgroup's own.
    const uint group_end = min((group + 1u) * PLACE_TG, n);
    for (uint t0 = 0u; t0 < group_end; t0 += PLACE_TG) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint j = t0 + lane;
        tile[lane] = j < n ? ids[j] : 0xFFFFFFFFu;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live) {
            const uint span = min(PLACE_TG, i > t0 ? i - t0 : 0u);
            for (uint jj = 0u; jj < span; ++jj) {
                rank += tile[jj] == id ? 1u : 0u;
            }
        }
    }
    if (live && id < vocab) {
        pos[starts[id] + rank] = i;
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
/// n_valid is ST_COUNT from ojas_check_ids, dispatched before this kernel.
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
    const float denom = (float)atomic_load_explicit(&st[ST_COUNT], memory_order_relaxed);
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

kernel void ojas_ce_mean(
    device const float *sum [[buffer(0)]],
    device const uint *st [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i != 0u) return;
    out[0] = precise::divide(sum[0], (float)st[ST_COUNT]);
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
    device atomic_uint *st [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant OjasAdamW &a [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (atomic_load_explicit(&st[ST_IN], memory_order_relaxed) != 0u
        || atomic_load_explicit(&st[ST_OUT], memory_order_relaxed) != 0u) {
        return;
    }
    float po, mo, vo;
    ojas_adamw_elem(p[i], g[i], m[i], v[i], a, po, mo, vo);
    p[i] = po;
    m[i] = mo;
    v[i] = vo;
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
// up to DM. Nothing T x T is stored: each threadgroup rebuilds 32 x 32
// blocks of the scores in threadgroup memory.
//
//   S   = scale * Q Kᵀ               (query t sees keys 0..=t)
//   O   = softmax(S) V, online over key blocks                  ojas_attn_fwd
//   lse = log sum_j exp(S_ij),  Dr_i = sum_j P_ij dP_ij       ojas_attn_bwd_stats
//   P   = exp(S - lse),  dP = dO Vᵀ,  dS = scale * P ∘ (dP - Dr)
//   dQ  = dS K                                                  ojas_attn_bwd_dq
//   dK  = dSᵀ Q,  dV = Pᵀ dO                                    ojas_attn_bwd_dkv
//
// Dr is formed from P and dP (the trait passes no forward output O), which
// equals rowsum(dO ∘ O). Each output row is written once by the
// threadgroup that owns it, with no atomics, so results repeat bit for bit.
//
// Q, K, V, dO are MPP tensors over the plane with extents (D, T): the
// matrix units read nothing past row T or column D, and every block entry
// outside the causal triangle or past T is set to exactly 0 before it is
// multiplied. DM is the compiled reduction width; D <= DM is the real head
// dimension. One (b, h) plane must fit i32 extents; the host checks.

using namespace mpp::tensor_ops;

#define ATT_BQ 32
#define ATT_BK 32
#define ATT_NSG 4
#define ATT_THREADS (ATT_NSG * 32)

/// Per query row: lse and Dr by an online pass over the key blocks. Four
/// threads share a row, eight columns each; the row's running max, sum and
/// Dr numerator are reduced across them with shuffles (lanes 4r..4r+3 of one
/// simdgroup). A live score that is not finite sets ST_OUT.
template <int DM>
inline void attn_bwd_stats_body(
    device float *Q, device float *K, device float *V, device float *dO,
    device float *lse, device float *dvec, device atomic_uint *st,
    uint T, uint D, float scale, uint2 tgpig, uint tid,
    threadgroup float *S, threadgroup float *dP)
{
    const uint q0 = tgpig.x * (uint)ATT_BQ;
    if (q0 >= T) { return; }
    const uint nq = min((uint)ATT_BQ, T - q0);
    const uint bh = tgpig.y;
    const ulong base = (ulong)bh * T * D;
    const uint t_end = q0 + nq;

    constexpr auto s_desc = matmul2d_descriptor(
        ATT_BQ, ATT_BK, DM, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<s_desc, execution_simdgroups<ATT_NSG>> s_op;

    auto mQ = tensor(Q + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mK = tensor(K + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mV = tensor(V + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mdO = tensor(dO + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto tQ = mQ.slice(0, (int)q0);
    auto tdO = mdO.slice(0, (int)q0);
    auto tS = tensor(S, dextents<int, 2>{ATT_BK, ATT_BQ}, array<int, 2>{1, ATT_BK});
    auto tdP = tensor(dP, dextents<int, 2>{ATT_BK, ATT_BQ}, array<int, 2>{1, ATT_BK});

    const uint r = tid / 4u;
    const uint c0 = (tid % 4u) * 8u;
    const bool row_live = r < nq;
    const uint qi = q0 + r;
    float m = -INFINITY;
    float l = 0.0f;
    float dacc = 0.0f;
    bool bad = false;

    for (uint kb = 0; kb < t_end; kb += (uint)ATT_BK) {
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

        float s[8];
        float bmax = -INFINITY;
        for (uint c = 0; c < 8u; ++c) {
            const uint j = kb + c0 + c;
            const bool live = row_live && j <= qi;
            const float v = S[r * ATT_BK + c0 + c] * scale;
            const bool ok = live && isfinite(v);
            bad = bad || (live && !ok);
            s[c] = ok ? v : -INFINITY;
            bmax = max(bmax, s[c]);
        }
        bmax = max(bmax, simd_shuffle_xor(bmax, 1));
        bmax = max(bmax, simd_shuffle_xor(bmax, 2));
        const float m_new = max(m, bmax);
        float pl = 0.0f;
        float pd = 0.0f;
        if (m_new != -INFINITY) {
            for (uint c = 0; c < 8u; ++c) {
                if (s[c] != -INFINITY) {
                    const float p = precise::exp(s[c] - m_new);
                    pl += p;
                    pd += p * dP[r * ATT_BK + c0 + c];
                }
            }
        }
        pl += simd_shuffle_xor(pl, 1);
        pl += simd_shuffle_xor(pl, 2);
        pd += simd_shuffle_xor(pd, 1);
        pd += simd_shuffle_xor(pd, 2);
        if (m_new != -INFINITY) {
            const float corr = precise::exp(m - m_new);
            l = l * corr + pl;
            dacc = dacc * corr + pd;
            m = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (bad) atomic_store_explicit(&st[ST_OUT], 1u, memory_order_relaxed);
    if (row_live && (tid % 4u) == 0u) {
        const ulong row = (ulong)bh * T + qi;
        lse[row] = m + precise::log(l);
        dvec[row] = precise::divide(dacc, l);
    }
}

/// dQ for query rows [q0, q0 + BQ): walk the key blocks through the last
/// query, rebuild dS per block in threadgroup memory, accumulate dS K.
template <int DM>
inline void attn_bwd_dq_body(
    device float *Q, device float *K, device float *V, device float *dO,
    device const float *lse, device const float *dvec, device float *dQ,
    uint T, uint D, float scale, uint2 tgpig, uint tid,
    threadgroup float *S, threadgroup float *dP, threadgroup float *lse_row,
    threadgroup float *d_row)
{
    const uint q0 = tgpig.x * (uint)ATT_BQ;
    if (q0 >= T) { return; }
    const uint nq = min((uint)ATT_BQ, T - q0);
    const uint bh = tgpig.y;
    const ulong base = (ulong)bh * T * D;
    const uint t_end = q0 + nq;

    constexpr auto s_desc = matmul2d_descriptor(
        ATT_BQ, ATT_BK, DM, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto acc_desc = matmul2d_descriptor(
        ATT_BQ, DM, ATT_BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<s_desc, execution_simdgroups<ATT_NSG>> s_op;
    matmul2d<acc_desc, execution_simdgroups<ATT_NSG>> acc_op;

    auto mQ = tensor(Q + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mK = tensor(K + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mV = tensor(V + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
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

    for (uint kb = 0; kb < t_end; kb += (uint)ATT_BK) {
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
            const bool live = rr < nq && kb + c <= q0 + rr;
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

/// dK and dV for key rows [k0, k0 + BK): walk the query blocks from the one
/// holding k0 to the end, rebuild Pᵀ and dSᵀ per block in threadgroup
/// memory, accumulate Pᵀ dO and dSᵀ Q.
template <int DM>
inline void attn_bwd_dkv_body(
    device float *Q, device float *K, device float *V, device float *dO,
    device const float *lse, device const float *dvec, device float *dK, device float *dV,
    uint T, uint D, float scale, uint2 tgpig, uint tid,
    threadgroup float *Pt, threadgroup float *dSt, threadgroup float *lse_col,
    threadgroup float *d_col)
{
    const uint k0 = tgpig.x * (uint)ATT_BK;
    if (k0 >= T) { return; }
    const uint nk = min((uint)ATT_BK, T - k0);
    const uint bh = tgpig.y;
    const ulong base = (ulong)bh * T * D;

    constexpr auto s_desc = matmul2d_descriptor(
        ATT_BK, ATT_BQ, DM, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto acc_desc = matmul2d_descriptor(
        ATT_BK, DM, ATT_BQ, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<s_desc, execution_simdgroups<ATT_NSG>> s_op;
    matmul2d<acc_desc, execution_simdgroups<ATT_NSG>> acc_op;

    auto mQ = tensor(Q + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mK = tensor(K + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mV = tensor(V + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mdO = tensor(dO + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto tK = mK.slice(0, (int)k0);
    auto tV = mV.slice(0, (int)k0);
    auto tPt = tensor(Pt, dextents<int, 2>{ATT_BQ, ATT_BK}, array<int, 2>{1, ATT_BQ});
    auto tdSt = tensor(dSt, dextents<int, 2>{ATT_BQ, ATT_BK}, array<int, 2>{1, ATT_BQ});

    auto dk = acc_op.template get_destination_cooperative_tensor<
        decltype(tPt), decltype(mQ.slice(0, 0)), float>();
    auto dv = acc_op.template get_destination_cooperative_tensor<
        decltype(tPt), decltype(mQ.slice(0, 0)), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < dk.get_capacity(); ++i) { dk[i] = 0.0f; dv[i] = 0.0f; }

    for (uint qb = (k0 / (uint)ATT_BQ) * (uint)ATT_BQ; qb < T; qb += (uint)ATT_BQ) {
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
            const bool live = c < nk && rr < nq && k0 + c <= qb + rr;
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
    auto mdK = tensor(dK + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mdV = tensor(dV + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    dk.store(mdK.slice(0, (int)k0));
    dv.store(mdV.slice(0, (int)k0));
}

/// Forward for query rows [q0, q0 + BQ), after tessl's
/// `qwen35_attn_tiled.metal`: S = Q Kᵀ per key block into threadgroup
/// memory, a per-row online softmax there (four threads per row, eight
/// columns each), P overwrites S, and O = O · diag(alpha) + P V accumulates
/// in a cooperative tensor. Key blocks past the last query are not visited.
/// A live score that is not finite sets ST_OUT and counts as masked.
template <int DM>
inline void attn_fwd_tiled_body(
    device float *Q, device float *K, device float *V, device float *O,
    device atomic_uint *st, uint T, uint D, float scale, uint2 tgpig, uint tid,
    threadgroup float *S, threadgroup float *m_row, threadgroup float *l_row,
    threadgroup float *a_row)
{
    const uint q0 = tgpig.x * (uint)ATT_BQ;
    if (q0 >= T) { return; }
    const uint nq = min((uint)ATT_BQ, T - q0);
    const uint bh = tgpig.y;
    const ulong base = (ulong)bh * T * D;
    const uint t_end = q0 + nq;

    constexpr auto qk_desc = matmul2d_descriptor(
        ATT_BQ, ATT_BK, DM, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(
        ATT_BQ, DM, ATT_BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<ATT_NSG>> qk_op;
    matmul2d<pv_desc, execution_simdgroups<ATT_NSG>> pv_op;

    auto mQ = tensor(Q + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mK = tensor(K + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto mV = tensor(V + base, dextents<int, 2>{(int)D, (int)T}, array<int, 2>{1, (int)D});
    auto tQ = mQ.slice(0, (int)q0);
    auto tP = tensor(S, dextents<int, 2>{ATT_BK, ATT_BQ}, array<int, 2>{1, ATT_BK});

    auto oT = pv_op.template get_destination_cooperative_tensor<
        decltype(tP), decltype(mV.slice(0, 0)), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < oT.get_capacity(); ++i) { oT[i] = 0.0f; }

    if (tid < (uint)ATT_BQ) {
        m_row[tid] = -INFINITY;
        l_row[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint r = tid / 4u;
    const uint c0 = (tid % 4u) * 8u;
    const bool row_live = r < nq;
    const uint qi = q0 + r;
    bool bad = false;

    for (uint kb = 0; kb < t_end; kb += (uint)ATT_BK) {
        auto tK = mK.slice(0, (int)kb);
        auto sT = qk_op.template get_destination_cooperative_tensor<
            decltype(tQ), decltype(tK), float>();
        qk_op.run(tQ, tK, sT);
        sT.store(tP);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float s[8];
        float mx = -INFINITY;
        for (uint c = 0; c < 8u; ++c) {
            const bool live = row_live && kb + c0 + c <= qi;
            const float v = S[r * ATT_BK + c0 + c] * scale;
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
        for (uint c = 0; c < 8u; ++c) {
            const float p = (s[c] == -INFINITY) ? 0.0f : precise::exp(s[c] - m_new);
            S[r * ATT_BK + c0 + c] = p;
            sum += p;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((tid % 4u) == 0u) {
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

#define OJAS_ATTN_TILED(DM)                                               \
kernel void ojas_attn_fwd_d##DM(                                        \
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], \
    device const float *v [[buffer(2)]], device float *o [[buffer(3)]],       \
    device atomic_uint *st [[buffer(4)]], constant uint &T [[buffer(5)]],     \
    constant uint &D [[buffer(6)]], constant float &scale [[buffer(7)]],      \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]])                                 \
{                                                                             \
    threadgroup float S[ATT_BQ * ATT_BK];                                     \
    threadgroup float m_row[ATT_BQ];                                          \
    threadgroup float l_row[ATT_BQ];                                          \
    threadgroup float a_row[ATT_BQ];                                          \
    attn_fwd_tiled_body<DM>(const_cast<device float *>(q),                    \
                            const_cast<device float *>(k),                    \
                            const_cast<device float *>(v), o, st, T, D,       \
                            scale, tgpig, tid, S, m_row, l_row, a_row);       \
}                                                                             \
kernel void ojas_attn_bwd_stats_d##DM(                                        \
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], \
    device const float *v [[buffer(2)]], device const float *dout [[buffer(3)]], \
    device float *lse [[buffer(4)]], device float *dvec [[buffer(5)]],        \
    device atomic_uint *st [[buffer(6)]], constant uint &T [[buffer(7)]],     \
    constant uint &D [[buffer(8)]], constant float &scale [[buffer(9)]],      \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]])                                 \
{                                                                             \
    threadgroup float S[ATT_BQ * ATT_BK];                                     \
    threadgroup float dP[ATT_BQ * ATT_BK];                                    \
    attn_bwd_stats_body<DM>(ATT_INPUTS, lse, dvec, st, T, D, scale, tgpig,    \
                            tid, S, dP);                                      \
}                                                                             \
kernel void ojas_attn_bwd_dq_d##DM(                                           \
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], \
    device const float *v [[buffer(2)]], device const float *dout [[buffer(3)]], \
    device const float *lse [[buffer(4)]], device const float *dvec [[buffer(5)]], \
    device float *dq [[buffer(6)]], constant uint &T [[buffer(7)]],           \
    constant uint &D [[buffer(8)]], constant float &scale [[buffer(9)]],      \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]])                                 \
{                                                                             \
    threadgroup float S[ATT_BQ * ATT_BK];                                     \
    threadgroup float dP[ATT_BQ * ATT_BK];                                    \
    threadgroup float r0[ATT_BQ];                                             \
    threadgroup float r1[ATT_BQ];                                             \
    attn_bwd_dq_body<DM>(ATT_INPUTS, lse, dvec, dq, T, D, scale, tgpig, tid,  \
                         S, dP, r0, r1);                                      \
}                                                                             \
kernel void ojas_attn_bwd_dkv_d##DM(                                          \
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], \
    device const float *v [[buffer(2)]], device const float *dout [[buffer(3)]], \
    device const float *lse [[buffer(4)]], device const float *dvec [[buffer(5)]], \
    device float *dk [[buffer(6)]], device float *dv [[buffer(7)]],           \
    constant uint &T [[buffer(8)]], constant uint &D [[buffer(9)]],           \
    constant float &scale [[buffer(10)]],                                     \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]])                                 \
{                                                                             \
    threadgroup float Pt[ATT_BQ * ATT_BK];                                    \
    threadgroup float dSt[ATT_BQ * ATT_BK];                                   \
    threadgroup float c0[ATT_BQ];                                             \
    threadgroup float c1[ATT_BQ];                                             \
    attn_bwd_dkv_body<DM>(ATT_INPUTS, lse, dvec, dk, dv, T, D, scale, tgpig,  \
                          tid, Pt, dSt, c0, c1);                              \
}

OJAS_ATTN_TILED(16)
OJAS_ATTN_TILED(32)
OJAS_ATTN_TILED(64)
OJAS_ATTN_TILED(128)
