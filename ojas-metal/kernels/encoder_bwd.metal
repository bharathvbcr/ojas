// Backward kernels of the encoder ops (LayerNorm, channel LayerNorm, exact
// GELU, softmax, broadcast add and multiply, and the bias gradient of a
// biased linear). tessl has none of these: its `qwen35_bwd` and `bert`
// kernels cover RMSNorm and the forward only.
//
// Reduction contract. The CPU reference sums in f64 and rounds once; Metal has
// no f64, so every sum here is a double-single (hi, lo) pair built from
// error-free transformations: `eb_two_sum` and `eb_two_prod` (the product's
// rounding error is `fma(a, b, -a * b)`, exact). A term is therefore added
// without rounding its product, and the running sum keeps about 46 bits. The
// result is rounded to f32 once, so a sum matches the f64 sum rounded once to
// within 2^-45 * sum(|term|) * n, far below one f32 ulp for the extents used
// (n <= 2^24). This needs the build's `-ffp-contract=off` and
// `-fmetal-math-mode=safe`: nothing may be reassociated or fused behind the
// transformations' back.
//
// Order. Each sum's order depends only on its extents, never on the machine or
// the run, so a result repeats bit for bit:
//   - a row reduction (`ojas_ln_bwd_rows`, `ojas_softmax_bwd_rows`): thread t of
//     T adds elements t, t + T, ... ascending; the T partials meet in a fixed
//     binary tree (T/2 + 0, ...). T is a power of two chosen from the row
//     length alone.
//   - a column or channel reduction: split k adds its contiguous range of rows
//     (or positions) ascending; the splits then add in ascending order. The
//     split count is a function of the extents alone.
//
// Status words follow ojas_backend.metal: ST_IN (an input element is
// non-finite) and ST_OUT (an output element, or a variance that is not finite
// and positive, is). A kernel that reads an operand flags it as it reads.
//
// Compiled with -ffp-contract=off and -fmetal-math-mode=safe; transcendental
// calls are `precise::`.
#include <metal_stdlib>
using namespace metal;

#define EB_ST_IN 0u
#define EB_ST_OUT 1u
#define EB_MAX_THREADS 256u

static inline void eb_flag(device atomic_uint *st, uint word)
{
    atomic_store_explicit(&st[word], 1u, memory_order_relaxed);
}

static inline float eb_bf16(float v)
{
    const uint b = as_type<uint>(v);
    if ((b & 0x7fffffffu) > 0x7f800000u) {
        return as_type<float>((b | 0x00400000u) & 0xffff0000u);
    }
    return as_type<float>((b + (0x7fffu + ((b >> 16) & 1u))) & 0xffff0000u);
}

// ------------------------------------------------- double-single arithmetic ---

static inline float2 eb_two_sum(float a, float b)
{
    const float s = a + b;
    const float bb = s - a;
    return float2(s, (a - (s - bb)) + (b - bb));
}

/// `|a| >= |b|` or `a == 0`.
static inline float2 eb_fast_two_sum(float a, float b)
{
    const float s = a + b;
    return float2(s, b - (s - a));
}

static inline float2 eb_two_prod(float a, float b)
{
    const float p = a * b;
    return float2(p, fma(a, b, -p));
}

static inline float2 eb_add_f(float2 a, float b)
{
    const float2 s = eb_two_sum(a.x, b);
    return eb_fast_two_sum(s.x, s.y + a.y);
}

static inline float2 eb_add(float2 a, float2 b)
{
    const float2 s = eb_two_sum(a.x, b.x);
    const float2 t = eb_two_sum(a.y, b.y);
    const float2 u = eb_fast_two_sum(s.x, s.y + t.x);
    return eb_fast_two_sum(u.x, t.y + u.y);
}

/// `a / d` for a float `d`: the quotient, then the quotient of the exact
/// remainder.
static inline float2 eb_div_f(float2 a, float d)
{
    const float q1 = precise::divide(a.x, d);
    const float r = fma(-q1, d, a.x) + a.y;
    return eb_fast_two_sum(q1, precise::divide(r, d));
}

/// Two double-single sums at once (`.xy` and `.zw`), added in a fixed tree
/// over `T` threads (a power of two, at most EB_MAX_THREADS). The total is on
/// every thread on return, and `sh` is free again.
static inline float4 eb_tg_sum2(float4 v, threadgroup float4 *sh, uint t, uint T)
{
    sh[t] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = T >> 1u; s > 0u; s >>= 1u) {
        if (t < s) {
            const float4 a = sh[t];
            const float4 b = sh[t + s];
            sh[t] = float4(eb_add(a.xy, b.xy), eb_add(a.zw, b.zw));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float4 r = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// -------------------------------------------------------------- LayerNorm ---
//
//   xhat = (x - mean) * rstd,  dxhat = g * w
//   dx   = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
//   dw   = sum_rows g * xhat,  db = sum_rows g
//
// with mean and the biased variance as the CPU forms them: two passes, the
// deviation taken from the double-single mean.

/// One threadgroup per row. Writes dx and the row's (mean.hi, mean.lo, rstd)
/// to `stats[4 r ..]` for the column pass. A variance + eps that is not finite
/// and positive sets ST_OUT and gives the row rstd 0 (so dx 0), as the CPU
/// refuses it.
kernel void ojas_ln_bwd_rows(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *gy [[buffer(2)]],
    device float *gx [[buffer(3)]],
    device float *stats [[buffer(4)]],
    device atomic_uint *st [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &dim [[buffer(7)]],
    constant float &eps [[buffer(8)]],
    constant uint &bf16_acc [[buffer(9)]],
    uint r [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]],
    uint T [[threads_per_threadgroup]])
{
    threadgroup float4 sh[EB_MAX_THREADS];
    if (r >= rows) return;
    const ulong base = (ulong)r * dim;
    const float fd = (float)dim;
    bool bad_in = false;

    float2 s = float2(0.0f);
    for (uint j = t; j < dim; j += T) {
        const float v = x[base + j];
        bad_in = bad_in || !isfinite(v);
        s = eb_add_f(s, v);
    }
    float2 mean;
    float rstd = 0.0f;
    float2 mean_g;
    float2 mean_gx;
    if (bf16_acc != 0u) {
        for (uint j = t; j < dim; j += T) {
            const float g = gy[base + j];
            const float wj = w[j];
            bad_in = bad_in || !isfinite(g) || !isfinite(wj);
        }
        if (t == 0u) {
            float s_bf16 = 0.0f;
            for (uint j = 0u; j < dim; ++j) {
                s_bf16 = eb_bf16(s_bf16 + eb_bf16(x[base + j]));
            }
            const float mean_bf16 = precise::divide(s_bf16, fd);
            float q_bf16 = 0.0f;
            for (uint j = 0u; j < dim; ++j) {
                const float d = eb_bf16(eb_bf16(x[base + j]) - eb_bf16(mean_bf16));
                q_bf16 = eb_bf16(q_bf16 + eb_bf16(d * d));
            }
            const float var_bf16 = precise::divide(q_bf16, fd);
            const float v_bf16 = var_bf16 + eps;
            float rstd_bf16 = 0.0f;
            bool ok_bf16 = isfinite(v_bf16) && v_bf16 > 0.0f;
            if (ok_bf16) {
                rstd_bf16 = precise::divide(1.0f, precise::sqrt(v_bf16));
                ok_bf16 = isfinite(rstd_bf16);
                if (!ok_bf16) rstd_bf16 = 0.0f;
            }
            if (!ok_bf16) eb_flag(st, EB_ST_OUT);
            float sg_bf16 = 0.0f;
            float sgx_bf16 = 0.0f;
            for (uint j = 0u; j < dim; ++j) {
                const float dx = eb_bf16(eb_bf16(gy[base + j]) * eb_bf16(w[j]));
                const float xhat = eb_bf16((x[base + j] - mean_bf16) * rstd_bf16);
                sg_bf16 = eb_bf16(sg_bf16 + dx);
                sgx_bf16 = eb_bf16(sgx_bf16 + eb_bf16(dx * xhat));
            }
            sh[0] = float4(
                mean_bf16,
                rstd_bf16,
                precise::divide(sg_bf16, fd),
                precise::divide(sgx_bf16, fd));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mean = float2(sh[0].x, 0.0f);
        rstd = sh[0].y;
        mean_g = float2(sh[0].z, 0.0f);
        mean_gx = float2(sh[0].w, 0.0f);
    } else {
        mean = eb_div_f(eb_tg_sum2(float4(s, 0.0f, 0.0f), sh, t, T).xy, fd);

        float2 q = float2(0.0f);
        for (uint j = t; j < dim; j += T) {
            const float d = (x[base + j] - mean.x) - mean.y;
            const float2 p = eb_two_prod(d, d);
            q = eb_add_f(eb_add_f(q, p.x), p.y);
        }
        const float2 var = eb_div_f(eb_tg_sum2(float4(q, 0.0f, 0.0f), sh, t, T).xy, fd);
        const float v = eb_add_f(var, eps).x;
        bool ok = isfinite(v) && v > 0.0f;
        if (ok) {
            rstd = precise::divide(1.0f, precise::sqrt(v));
            ok = isfinite(rstd);
            if (!ok) rstd = 0.0f;
        }
        if (!ok && t == 0u) eb_flag(st, EB_ST_OUT);

        float2 sg = float2(0.0f);
        float2 sgx = float2(0.0f);
        for (uint j = t; j < dim; j += T) {
            const float g = gy[base + j];
            const float wj = w[j];
            bad_in = bad_in || !isfinite(g) || !isfinite(wj);
            const float xhat = ((x[base + j] - mean.x) - mean.y) * rstd;
            const float2 dx = eb_two_prod(g, wj);
            sg = eb_add_f(eb_add_f(sg, dx.x), dx.y);
            const float2 pp = eb_two_prod(dx.x, xhat);
            sgx = eb_add_f(eb_add_f(eb_add_f(sgx, pp.x), pp.y), dx.y * xhat);
        }
        const float4 tot = eb_tg_sum2(float4(sg, sgx), sh, t, T);
        mean_g = eb_div_f(tot.xy, fd);
        mean_gx = eb_div_f(tot.zw, fd);
    }

    bool bad_out = false;
    for (uint j = t; j < dim; j += T) {
        const float2 dx = eb_two_prod(gy[base + j], w[j]);
        const float xhat = ((x[base + j] - mean.x) - mean.y) * rstd;
        const float term1 = ((dx.x - mean_g.x) + dx.y) - mean_g.y;
        const float term2 = fma(xhat, mean_gx.x, xhat * mean_gx.y);
        const float o = rstd * (term1 - term2);
        gx[base + j] = o;
        bad_out = bad_out || !isfinite(o);
    }
    if (t == 0u) {
        stats[(ulong)r * 4u + 0u] = mean.x;
        stats[(ulong)r * 4u + 1u] = mean.y;
        stats[(ulong)r * 4u + 2u] = rstd;
    }
    if (bad_in) eb_flag(st, EB_ST_IN);
    if (bad_out) eb_flag(st, EB_ST_OUT);
}

/// Stage 1 of dw and db: for column c and split k, the sums over rows
/// [k * chunk, ...) ascending, as (dw.hi, dw.lo, db.hi, db.lo) at
/// `part[4 (k dim + c) ..]`. Grid: x = column, y = split.
kernel void ojas_ln_bwd_cols_part(
    device const float *x [[buffer(0)]],
    device const float *gy [[buffer(1)]],
    device const float *stats [[buffer(2)]],
    device float *part [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &chunk [[buffer(6)]],
    constant uint &bf16_acc [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const uint k = gid.y;
    if (c >= dim) return;
    const ulong r0 = (ulong)k * chunk;
    const ulong r1 = min(r0 + chunk, (ulong)rows);
    float2 a = float2(0.0f);
    float2 b = float2(0.0f);
    for (ulong r = r0; r < r1; ++r) {
        const ulong i = r * dim + c;
        const float mh = stats[r * 4u + 0u];
        const float ml = stats[r * 4u + 1u];
        const float rstd = stats[r * 4u + 2u];
        const float g = gy[i];
        const float xhat = ((x[i] - mh) - ml) * rstd;
        if (bf16_acc != 0u) {
            a.x = eb_bf16(a.x + eb_bf16(eb_bf16(g) * eb_bf16(xhat)));
            b.x = eb_bf16(b.x + eb_bf16(g));
        } else {
            const float2 p = eb_two_prod(g, xhat);
            a = eb_add_f(eb_add_f(a, p.x), p.y);
            b = eb_add_f(b, g);
        }
    }
    const ulong o = ((ulong)k * dim + c) * 4u;
    part[o + 0u] = a.x;
    part[o + 1u] = a.y;
    part[o + 2u] = b.x;
    part[o + 3u] = b.y;
}

/// Stage 2: out_a[c] and out_b[c] are the splits' two sums added in ascending
/// split order, rounded once. A non-finite result sets ST_OUT.
kernel void ojas_cols_fin4(
    device const float *part [[buffer(0)]],
    device float *out_a [[buffer(1)]],
    device float *out_b [[buffer(2)]],
    device atomic_uint *st [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &splits [[buffer(5)]],
    constant uint &bf16_acc [[buffer(6)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= n) return;
    float2 a = float2(0.0f);
    float2 b = float2(0.0f);
    for (uint k = 0u; k < splits; ++k) {
        const ulong o = ((ulong)k * n + c) * 4u;
        if (bf16_acc != 0u) {
            a.x = eb_bf16(a.x + eb_bf16(part[o + 0u]));
            b.x = eb_bf16(b.x + eb_bf16(part[o + 2u]));
        } else {
            a = eb_add(a, float2(part[o + 0u], part[o + 1u]));
            b = eb_add(b, float2(part[o + 2u], part[o + 3u]));
        }
    }
    out_a[c] = a.x;
    out_b[c] = b.x;
    if (!isfinite(a.x) || !isfinite(b.x)) eb_flag(st, EB_ST_OUT);
}

// ---------------------------------------------------- channel LayerNorm ---
//
// `[batch, channels, spatial]`: a position (b, s) normalizes over its channels,
// strided by `spatial`. One thread per position walks the channels in order,
// so neighbouring threads read neighbouring addresses.

kernel void ojas_cln_bwd_pos(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *gy [[buffer(2)]],
    device float *gx [[buffer(3)]],
    device float *stats [[buffer(4)]],
    device atomic_uint *st [[buffer(5)]],
    constant uint &positions [[buffer(6)]],
    constant uint &channels [[buffer(7)]],
    constant uint &spatial [[buffer(8)]],
    constant float &eps [[buffer(9)]],
    constant uint &bf16_acc [[buffer(10)]],
    uint p [[thread_position_in_grid]])
{
    if (p >= positions) return;
    const uint b = p / spatial;
    const uint s = p - b * spatial;
    const ulong base = (ulong)b * channels * spatial + s;
    const ulong step = spatial;
    const float fc = (float)channels;
    bool bad_in = false;

    float2 mean;
    float rstd = 0.0f;
    bool bad_out = false;
    float2 mean_g;
    float2 mean_gx;

    if (bf16_acc != 0u) {
        float sum_bf16 = 0.0f;
        for (uint c = 0u; c < channels; ++c) {
            const float v = x[base + c * step];
            bad_in = bad_in || !isfinite(v);
            sum_bf16 = eb_bf16(sum_bf16 + eb_bf16(v));
        }
        const float mean_bf16 = precise::divide(sum_bf16, fc);
        float q_bf16 = 0.0f;
        for (uint c = 0u; c < channels; ++c) {
            const float d = eb_bf16(eb_bf16(x[base + c * step]) - eb_bf16(mean_bf16));
            q_bf16 = eb_bf16(q_bf16 + eb_bf16(d * d));
        }
        const float var_bf16 = precise::divide(q_bf16, fc);
        const float v_bf16 = var_bf16 + eps;
        bool ok = isfinite(v_bf16) && v_bf16 > 0.0f;
        if (ok) {
            rstd = precise::divide(1.0f, precise::sqrt(v_bf16));
            ok = isfinite(rstd);
            if (!ok) rstd = 0.0f;
        }
        bad_out = !ok;

        float sg_bf16 = 0.0f;
        float sgx_bf16 = 0.0f;
        for (uint c = 0u; c < channels; ++c) {
            const float g = gy[base + c * step];
            const float wc = w[c];
            bad_in = bad_in || !isfinite(g) || !isfinite(wc);
            const float xhat = eb_bf16((x[base + c * step] - mean_bf16) * rstd);
            const float dx = eb_bf16(eb_bf16(g) * eb_bf16(wc));
            sg_bf16 = eb_bf16(sg_bf16 + dx);
            sgx_bf16 = eb_bf16(sgx_bf16 + eb_bf16(dx * xhat));
        }
        mean = float2(mean_bf16, 0.0f);
        mean_g = float2(precise::divide(sg_bf16, fc), 0.0f);
        mean_gx = float2(precise::divide(sgx_bf16, fc), 0.0f);
    } else {
        float2 sum = float2(0.0f);
        for (uint c = 0u; c < channels; ++c) {
            const float v = x[base + c * step];
            bad_in = bad_in || !isfinite(v);
            sum = eb_add_f(sum, v);
        }
        mean = eb_div_f(sum, fc);
        float2 q = float2(0.0f);
        for (uint c = 0u; c < channels; ++c) {
            const float d = (x[base + c * step] - mean.x) - mean.y;
            const float2 pr = eb_two_prod(d, d);
            q = eb_add_f(eb_add_f(q, pr.x), pr.y);
        }
        const float2 var = eb_div_f(q, fc);
        const float v = eb_add_f(var, eps).x;
        bool ok = isfinite(v) && v > 0.0f;
        if (ok) {
            rstd = precise::divide(1.0f, precise::sqrt(v));
            ok = isfinite(rstd);
            if (!ok) rstd = 0.0f;
        }
        bad_out = !ok;

        float2 sg = float2(0.0f);
        float2 sgx = float2(0.0f);
        for (uint c = 0u; c < channels; ++c) {
            const float g = gy[base + c * step];
            const float wc = w[c];
            bad_in = bad_in || !isfinite(g) || !isfinite(wc);
            const float xhat = ((x[base + c * step] - mean.x) - mean.y) * rstd;
            const float2 dx = eb_two_prod(g, wc);
            sg = eb_add_f(eb_add_f(sg, dx.x), dx.y);
            const float2 pp = eb_two_prod(dx.x, xhat);
            sgx = eb_add_f(eb_add_f(eb_add_f(sgx, pp.x), pp.y), dx.y * xhat);
        }
        mean_g = eb_div_f(sg, fc);
        mean_gx = eb_div_f(sgx, fc);
    }

    for (uint c = 0u; c < channels; ++c) {
        const ulong i = base + c * step;
        const float2 dx = eb_two_prod(gy[i], w[c]);
        const float xhat = ((x[i] - mean.x) - mean.y) * rstd;
        const float term1 = ((dx.x - mean_g.x) + dx.y) - mean_g.y;
        const float term2 = fma(xhat, mean_gx.x, xhat * mean_gx.y);
        const float o = rstd * (term1 - term2);
        gx[i] = o;
        bad_out = bad_out || !isfinite(o);
    }
    stats[(ulong)p * 4u + 0u] = mean.x;
    stats[(ulong)p * 4u + 1u] = mean.y;
    stats[(ulong)p * 4u + 2u] = rstd;
    if (bad_in) eb_flag(st, EB_ST_IN);
    if (bad_out) eb_flag(st, EB_ST_OUT);
}

/// Stage 1 of dw[c] and db[c]: one threadgroup per (channel, split); the
/// split covers positions [k * chunk, ...) (batch-major, as the CPU walks
/// them), thread t adding t, t + T, ... and the threads meeting in the fixed
/// tree. Partials are laid out as `ojas_ln_bwd_cols_part`'s.
kernel void ojas_cln_bwd_ch_part(
    device const float *x [[buffer(0)]],
    device const float *gy [[buffer(1)]],
    device const float *stats [[buffer(2)]],
    device float *part [[buffer(3)]],
    constant uint &positions [[buffer(4)]],
    constant uint &channels [[buffer(5)]],
    constant uint &spatial [[buffer(6)]],
    constant uint &chunk [[buffer(7)]],
    constant uint &bf16_acc [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tsz [[threads_per_threadgroup]])
{
    threadgroup float4 sh[EB_MAX_THREADS];
    const uint t = tid.x;
    const uint T = tsz.x;
    const uint c = tg.x;
    const uint k = tg.y;
    if (c >= channels) return;
    const ulong p0 = (ulong)k * chunk;
    const ulong p1 = min(p0 + chunk, (ulong)positions);
    if (bf16_acc != 0u) {
        if (t == 0u) {
            float sa = 0.0f;
            float sb = 0.0f;
            for (ulong p = p0; p < p1; ++p) {
                const ulong bi = p / spatial;
                const ulong s = p - bi * spatial;
                const ulong i = (bi * channels + c) * spatial + s;
                const float mh = stats[p * 4u + 0u];
                const float ml = stats[p * 4u + 1u];
                const float rstd = stats[p * 4u + 2u];
                const float g = gy[i];
                const float xhat = ((x[i] - mh) - ml) * rstd;
                sa = eb_bf16(sa + eb_bf16(eb_bf16(g) * eb_bf16(xhat)));
                sb = eb_bf16(sb + eb_bf16(g));
            }
            const ulong o = ((ulong)k * channels + c) * 4u;
            part[o + 0u] = sa;
            part[o + 1u] = 0.0f;
            part[o + 2u] = sb;
            part[o + 3u] = 0.0f;
        }
        return;
    }
    float2 a = float2(0.0f);
    float2 b = float2(0.0f);
    for (ulong p = p0 + t; p < p1; p += T) {
        const ulong bi = p / spatial;
        const ulong s = p - bi * spatial;
        const ulong i = (bi * channels + c) * spatial + s;
        const float mh = stats[p * 4u + 0u];
        const float ml = stats[p * 4u + 1u];
        const float rstd = stats[p * 4u + 2u];
        const float g = gy[i];
        const float xhat = ((x[i] - mh) - ml) * rstd;
        const float2 pr = eb_two_prod(g, xhat);
        a = eb_add_f(eb_add_f(a, pr.x), pr.y);
        b = eb_add_f(b, g);
    }
    const float4 tot = eb_tg_sum2(float4(a, b), sh, t, T);
    if (t == 0u) {
        const ulong o = ((ulong)k * channels + c) * 4u;
        part[o + 0u] = tot.x;
        part[o + 1u] = tot.y;
        part[o + 2u] = tot.z;
        part[o + 3u] = tot.w;
    }
}

// ------------------------------------------------------------ GELU (erf) ---

/// erfc(x) to a relative error below 1.2e-7 everywhere: the Chebyshev fit
/// tessl's `bert_gelu_erf` uses for the forward (`bert_erfc` in
/// tessl/kernels/bert.metal), repeated here so the backward differentiates
/// the same function the forward evaluates. Relative accuracy is the point:
/// the left tail of Phi is a tiny number an absolute-error erf gets wrong.
static inline float eb_erfc(float x)
{
    const float z = min(fabs(x), 10.0f);
    const float t = precise::divide(1.0f, 1.0f + 0.5f * z);
    const float poly = -z * z - 1.26551223f
        + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f
        + t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f
        + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
    const float r = t * precise::exp(poly);
    return x >= 0.0f ? r : 2.0f - r;
}

/// dy * gelu'(x) = dy * (Phi(x) + x phi(x)), one element per thread. The
/// density term is formed as ((dy c) e) e x with e = exp(-x^2 / 4) once |x|
/// passes 9, where e^(-x^2 / 2) alone would underflow before a gradient near
/// the f32 maximum could bring the term back (the CPU keeps it; `e` stays a
/// normal number up to |x| = 18.6). The cumulative term is fused with the
/// density term in one rounding.
kernel void ojas_gelu_erf_bwd(
    device const float *x [[buffer(0)]],
    device const float *gy [[buffer(1)]],
    device float *gx [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    device atomic_uint *st [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float xv = x[i];
    const float dy = gy[i];
    const float cdf = 0.5f * eb_erfc(-xv * 0.70710678118654752f);
    float dens;
    if (fabs(xv) <= 9.0f) {
        dens = dy * (xv * 0.39894228040143268f * precise::exp(-0.5f * xv * xv));
    } else {
        const float e = precise::exp(-0.25f * xv * xv);
        dens = (((dy * 0.39894228040143268f) * e) * e) * xv;
    }
    const float o = fma(dy, cdf, dens);
    gx[i] = o;
    if (!isfinite(xv) || !isfinite(dy)) eb_flag(st, EB_ST_IN);
    if (!isfinite(o)) eb_flag(st, EB_ST_OUT);
}

// --------------------------------------------------------------- softmax ---

/// dx = y * (g - sum(g * y)) per row, one threadgroup per row. The dot is a
/// double-single sum of exact products; the value is formed from the exact
/// products too, so it carries one rounding.
kernel void ojas_softmax_bwd_rows(
    device const float *y [[buffer(0)]],
    device const float *gy [[buffer(1)]],
    device float *gx [[buffer(2)]],
    device atomic_uint *st [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &bf16_acc [[buffer(6)]],
    uint r [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]],
    uint T [[threads_per_threadgroup]])
{
    threadgroup float4 sh[EB_MAX_THREADS];
    if (r >= rows) return;
    const ulong base = (ulong)r * dim;
    bool bad_in = false;
    float2 dot = float2(0.0f);
    for (uint j = t; j < dim; j += T) {
        const float yv = y[base + j];
        const float gv = gy[base + j];
        bad_in = bad_in || !isfinite(yv) || !isfinite(gv);
        if (bf16_acc == 0u) {
            const float2 p = eb_two_prod(yv, gv);
            dot = eb_add_f(eb_add_f(dot, p.x), p.y);
        }
    }
    float2 d;
    if (bf16_acc != 0u) {
        if (t == 0u) {
            float s = 0.0f;
            for (uint j = 0u; j < dim; ++j) {
                s = eb_bf16(s + eb_bf16(eb_bf16(y[base + j]) * eb_bf16(gy[base + j])));
            }
            sh[0].x = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        d = float2(sh[0].x, 0.0f);
    } else {
        d = eb_tg_sum2(float4(dot, 0.0f, 0.0f), sh, t, T).xy;
    }
    bool bad_out = false;
    for (uint j = t; j < dim; j += T) {
        const float yv = y[base + j];
        const float2 p = eb_two_prod(yv, gy[base + j]);
        const float2 q = eb_two_prod(yv, d.x);
        const float o = ((p.x - q.x) + (p.y - q.y)) - yv * d.y;
        gx[base + j] = o;
        bad_out = bad_out || !isfinite(o);
    }
    if (bad_in) eb_flag(st, EB_ST_IN);
    if (bad_out) eb_flag(st, EB_ST_OUT);
}

// ------------------------------------------------ broadcast gradients ---
//
// The gradient of one operand of a broadcast add or multiply: the output
// gradient `g` summed over the output axes the operand was stretched along,
// for a multiply each term multiplied by the other operand's element. One
// thread per (operand element, split); the split adds its contiguous range of
// the reduced index space in output order (the first reduced axis slowest),
// which is the CPU's order; the splits then meet in `ojas_bcast_grad_fin`. With
// one split the part kernel writes the gradient itself.
//
// `geom` (uint words): [0] operand elements, [1] reduced extent R, [2] splits,
// [3] indices per split, [4] kept rank, [5] reduced rank, [6] 1 for multiply;
// then, 8 words each, the kept axes' extents, output strides and the other
// operand's strides, and the reduced axes' extents, output strides and the
// other operand's strides, first axis first.

#define BG_KEPT_EXT 8u
#define BG_KEPT_OUT 16u
#define BG_KEPT_MATE 24u
#define BG_RED_EXT 32u
#define BG_RED_OUT 40u
#define BG_RED_MATE 48u

kernel void ojas_bcast_grad_part(
    device const float *g [[buffer(0)]],
    device const float *other [[buffer(1)]],
    device float *out [[buffer(2)]],
    device atomic_uint *st [[buffer(3)]],
    constant uint *geom [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint elems = geom[0];
    const uint total = geom[1];
    const uint splits = geom[2];
    const uint chunk = geom[3];
    const uint kept = geom[4];
    const uint reduced = geom[5];
    const bool mul = geom[6] != 0u;
    const bool bf16_acc = geom[7] != 0u;
    const uint e = gid.x;
    const uint k = gid.y;
    if (e >= elems || k >= splits) return;

    ulong o = 0;
    ulong p = 0;
    uint rest = e;
    for (uint a = kept; a > 0u; --a) {
        const uint ext = geom[BG_KEPT_EXT + a - 1u];
        const uint coord = rest % ext;
        rest /= ext;
        o += (ulong)coord * geom[BG_KEPT_OUT + a - 1u];
        p += (ulong)coord * geom[BG_KEPT_MATE + a - 1u];
    }
    const uint r0 = k * chunk;
    const uint r1 = r0 < total ? min(total - r0, chunk) + r0 : r0;
    uint idx[8];
    uint rem = r0;
    for (uint a = reduced; a > 0u; --a) {
        const uint ext = geom[BG_RED_EXT + a - 1u];
        const uint coord = rem % ext;
        rem /= ext;
        idx[a - 1u] = coord;
        o += (ulong)coord * geom[BG_RED_OUT + a - 1u];
        p += (ulong)coord * geom[BG_RED_MATE + a - 1u];
    }
    float2 acc = float2(0.0f);
    bool bad = false;
    for (uint r = r0; r < r1; ++r) {
        const float gv = g[o];
        bad = bad || !isfinite(gv);
        if (bf16_acc) {
            if (mul) {
                const float ov = other[p];
                bad = bad || !isfinite(ov);
                acc.x = eb_bf16(acc.x + eb_bf16(eb_bf16(gv) * eb_bf16(ov)));
            } else {
                acc.x = eb_bf16(acc.x + eb_bf16(gv));
            }
        } else if (mul) {
            const float ov = other[p];
            bad = bad || !isfinite(ov);
            const float2 pr = eb_two_prod(gv, ov);
            acc = eb_add_f(eb_add_f(acc, pr.x), pr.y);
        } else {
            acc = eb_add_f(acc, gv);
        }
        for (uint a = reduced; a > 0u; --a) {
            const uint ax = a - 1u;
            idx[ax] += 1u;
            o += geom[BG_RED_OUT + ax];
            p += geom[BG_RED_MATE + ax];
            if (idx[ax] < geom[BG_RED_EXT + ax]) break;
            o -= (ulong)idx[ax] * geom[BG_RED_OUT + ax];
            p -= (ulong)idx[ax] * geom[BG_RED_MATE + ax];
            idx[ax] = 0u;
        }
    }
    if (bad) eb_flag(st, EB_ST_IN);
    if (splits == 1u) {
        out[e] = acc.x;
        if (!isfinite(acc.x)) eb_flag(st, EB_ST_OUT);
    } else {
        const ulong at = ((ulong)k * elems + e) * 2u;
        out[at + 0u] = acc.x;
        out[at + 1u] = acc.y;
    }
}

/// The splits' partial sums added in ascending split order, rounded once.
kernel void ojas_bcast_grad_fin(
    device const float *part [[buffer(0)]],
    device float *out [[buffer(1)]],
    device atomic_uint *st [[buffer(2)]],
    constant uint &elems [[buffer(3)]],
    constant uint &splits_raw [[buffer(4)]],
    uint e [[thread_position_in_grid]])
{
    if (e >= elems) return;
    const bool bf16_acc = (splits_raw & 0x80000000u) != 0u;
    const uint splits = splits_raw & 0x7fffffffu;
    float2 acc = float2(0.0f);
    for (uint k = 0u; k < splits; ++k) {
        const ulong at = ((ulong)k * elems + e) * 2u;
        if (bf16_acc) {
            acc.x = eb_bf16(acc.x + eb_bf16(part[at + 0u]));
        } else {
            acc = eb_add(acc, float2(part[at + 0u], part[at + 1u]));
        }
    }
    out[e] = acc.x;
    if (!isfinite(acc.x)) eb_flag(st, EB_ST_OUT);
}
