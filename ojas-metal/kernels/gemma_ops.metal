// EmbeddingGemma 2 encoder ops that tessl has no standalone kernel for.
//
//   ojas_gelu_tanh_fwd       y = gelu_tanh(x)             one element a thread
//   ojas_gelu_tanh_bwd       dx = dy * gelu_tanh'(x)      one element a thread
//   ojas_segments_from_lens  seg[b] = (b T, b T + min(len[b], T))
//   ojas_seq_mean_bwd        dx[b,t,d] = dy[b,d] / min(len[b], T), t < that
//   ojas_l2_normalize_bwd    dx = (dy - y (y . dy)) / n, or dy / eps clamped
//
// Why these exist: tessl's GELU is the fused `mlp_gelu_tanh` (gelu(gate) * up,
// no status words), it has no GELU derivative, and its `segment_mean_rows`
// takes row ranges the host would otherwise have to read the key lengths back
// to build. The attention, the masked mean and the L2 normalize themselves run
// on tessl's kernels (`encoder_attn`, `segment_mean_rows`,
// `l2_normalize_rows`); see `src/gemma_ops.rs`.
//
// The forward is tessl's own `tessl_gelu_pytorch_tanh`, included from its
// `act.h` (build.rs passes tessl's kernels directory with -I), so the formula
// has one owner. It bounds the cubic at |x| <= 20 and its tanh argument at
// |.| <= 10. The backward keeps the same |x| <= 20 bound on the cubic, so it
// differentiates the same function, but forms the derivative from the sigmoid
// rather than from 1 + tanh: see `ojas_gelu_tanh_bwd` for why the tanh form
// cannot carry it in f32 where the argument saturates.
//
// Status words, as `ojas_backend.metal`: 0 an input is non-finite, 1 an
// output is non-finite. Compiled with -ffp-contract=off and
// -fmetal-math-mode=safe; transcendentals are `precise::`.
#include <metal_stdlib>
#include "act.h"
using namespace metal;

#define GEMMA_ST_IN 0u
#define GEMMA_ST_OUT 1u

static inline void gemma_flag(device atomic_uint *st, uint word)
{
    atomic_store_explicit(&st[word], 1u, memory_order_relaxed);
}

kernel void ojas_gelu_tanh_fwd(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    device atomic_uint *st [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float v = x[i];
    const float r = tessl_gelu_pytorch_tanh(v);
    y[i] = r;
    if (!isfinite(v)) gemma_flag(st, GEMMA_ST_IN);
    if (!isfinite(r)) gemma_flag(st, GEMMA_ST_OUT);
}

// d/dx [x s(u)], u = 2 sqrt(2/pi) (x + 0.044715 x^3) and s = sigmoid(u):
//   s + x s(1 - s) u',   u' = 2 sqrt(2/pi) (1 + 3 * 0.044715 x^2)
//
// The identity 1 + tanh(u/2) = 2 s makes this the derivative of the exact
// function the forward computes: the forward's 0.5 x (1 + t) is x s. But it
// is s and s(1 - s) that an f32 kernel must form, not 1 + t and 1 - t^2.
// Both of those cancel as |t| -> 1, and 0.5 x u' scales what is left of them
// by about ten at |x| = 5, so two ulps of tanh became fifteen ulps of answer.
// Measured on an M4: at x = 5 the kernel returned dy bit for bit, where the
// derivative is 1 + 1.5e-6. Only t == 1.0f produces that, because 1 - t*t
// vanishes for no other f32 near one (0x3f7fffff leaves 1.19e-7 and
// 0x3f7ffffe leaves 2.38e-7), so `precise::tanh` returned 0x3f800000 at
// |u/2| = 8.4490986 where the correctly rounded value is 0x3f7ffffe, two ulps
// below. That was 1.788e-6 against the torch golden, 99.4% of the oracle
// gate's budget, and exactly 0 at x = -5 against -1.07e-6.
//
// From e = e^-|u| alone neither s nor s(1 - s) = e / (1 + e)^2 cancels on
// either tail, and the denominator stays in [1, 2] so no intermediate is ever
// infinite -- the shape tessl's `tessl_sigmoid` (`act.h`) uses, and the same
// formula as the CPU reference `ojas_core::gelu_tanh_backward_exact`, which is
// bit-identical to the golden here. The tanh argument needs no clamp now: past
// saturation e underflows to 0 on its own, leaving exactly dy for far positive
// x and a signed zero for far negative x, with no 0 * inf. The cubic keeps the
// forward's |x| <= 20 bound so x^3 cannot overflow, and the bounded x is what
// multiplies the slope, so a huge |x| cannot overflow that product either.
kernel void ojas_gelu_tanh_bwd(
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
    const float xc = clamp(v, -20.0f, 20.0f);
    const float x2 = xc * xc;
    const float x3 = x2 * xc;
    const float u = 1.5957691216057308f * (xc + 0.044715f * x3);
    const float e = precise::exp(-fabs(u));
    const float den = 1.0f + e;
    const float s = u >= 0.0f ? 1.0f / den : e / den;
    const float slope = e / (den * den);
    const float du = 1.5957691216057308f * (1.0f + 3.0f * 0.044715f * x2);
    const float d = s + xc * slope * du;
    const float r = gy * d;
    out[i] = r;
    if (!isfinite(v) || !isfinite(gy)) gemma_flag(st, GEMMA_ST_IN);
    if (!isfinite(r)) gemma_flag(st, GEMMA_ST_OUT);
}

// One thread a batch row. The host guarantees B * T fits a u32. A length past
// T is T (`seq_mean_forward_dims`: no length is out of range).
kernel void ojas_segments_from_lens(
    device const uint *lens [[buffer(0)]],
    device uint *seg [[buffer(1)]],
    constant uint &B [[buffer(2)]],
    constant uint &T [[buffer(3)]],
    uint b [[thread_position_in_grid]])
{
    if (b >= B) return;
    const uint start = b * T;
    seg[2u * b] = start;
    seg[2u * b + 1u] = start + min(lens[b], T);
}

// The gradient of the masked mean, one output element a thread: a gather, so
// no two threads write one value and nothing is summed. With n = min(len[b],
// T), element (b, t, d) is dy[b, d] / n for t < n and +0.0 at or past n,
// because the forward never read those rows. n = 0 leaves the whole block
// +0.0. The host checks the whole gradient for non-finite values before this
// runs (ojas_check_finite, GEMMA_ST_IN), so this flags only its output.
//
// tessl has no backward for `segment_mean_rows`; this is the only kernel that
// spreads a pooled gradient back over the rows it came from.
kernel void ojas_seq_mean_bwd(
    device const float *g [[buffer(0)]],
    device const uint *lens [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &B [[buffer(3)]],
    constant uint &T [[buffer(4)]],
    constant uint &D [[buffer(5)]],
    device atomic_uint *st [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= B * T * D) return;
    const uint d = i % D;
    const uint row = i / D;
    const uint b = row / T;
    const uint t = row - b * T;
    const uint n = min(lens[b], T);
    float r = 0.0f;
    if (t < n) r = g[b * D + d] / (float)n;
    out[i] = r;
    if (!isfinite(r)) gemma_flag(st, GEMMA_ST_OUT);
}

// The gradient of the row-wise L2 normalize, one row a thread: two serial
// passes over the row in f32 (the squared norm, then the dot product), so
// there is no threadgroup reduction and no barrier to get wrong. With
// n = ||x|| and y = x / n:
//
//   n >  eps : dx = (dy - y (y . dy)) / n     the gradient of x / ||x||
//   n <= eps : dx = dy / eps                  the forward was linear there
//
// The clamp branch has no projection term: below eps the forward divided by
// the constant eps, so its Jacobian is I / eps. `!(n > eps)` takes the tie
// and a NaN norm into the clamp branch, matching the CPU's `n <= eps`.
//
// The parallelism is the row count, not the width. Rows are the batch of an
// embedding call, so this is small by design; performance is not what this
// kernel is for, and a threadgroup reduction is the obvious change if a
// profile ever asks for one. The host scans both operands, so this flags
// only its output.
kernel void ojas_l2_normalize_bwd(
    device const float *x [[buffer(0)]],
    device const float *g [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    device atomic_uint *st [[buffer(6)]],
    uint r [[thread_position_in_grid]])
{
    if (r >= rows) return;
    const uint at = r * dim;
    float sum_sq = 0.0f;
    for (uint c = 0u; c < dim; ++c) {
        const float v = x[at + c];
        sum_sq += v * v;
    }
    const float norm = precise::sqrt(sum_sq);
    if (!(norm > eps)) {
        for (uint c = 0u; c < dim; ++c) {
            const float d = g[at + c] / eps;
            out[at + c] = d;
            if (!isfinite(d)) gemma_flag(st, GEMMA_ST_OUT);
        }
        return;
    }
    float dot = 0.0f;
    for (uint c = 0u; c < dim; ++c) dot += (x[at + c] / norm) * g[at + c];
    for (uint c = 0u; c < dim; ++c) {
        const float d = (g[at + c] - (x[at + c] / norm) * dot) / norm;
        out[at + c] = d;
        if (!isfinite(d)) gemma_flag(st, GEMMA_ST_OUT);
    }
}
