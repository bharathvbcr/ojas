// EmbeddingGemma 2 encoder ops that tessl has no standalone kernel for.
//
//   ojas_gelu_tanh_fwd       y = gelu_tanh(x)             one element a thread
//   ojas_gelu_tanh_bwd       dx = dy * gelu_tanh'(x)      one element a thread
//   ojas_segments_from_lens  seg[b] = (b T, b T + min(len[b], T))
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
// has one owner. The cubic is bounded at |x| <= 20 and the tanh argument at
// |.| <= 10 there; the backward uses the same bounded inner term for both of
// its parts, so it differentiates the function the forward computes.
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

// d/dx [0.5 x (1 + t)], t = tanh(inner(x)):
//   0.5 (1 + t) + 0.5 x (1 - t^2) inner'(x)
//   inner'(x) = sqrt(2/pi) (1 + 3 * 0.044715 x^2)
// with the forward's bounded x for the inner term and its bounded tanh
// argument, so past |inner| = 10 (where tanh has rounded to +-1 in f32) the
// second part is exactly 0 and the first is 1 or 0: the derivative of the
// saturated function, not a NaN from 0 * inf.
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
    const float inner = 0.7978845608028654f * (xc + 0.044715f * x3);
    const float t = precise::tanh(clamp(inner, -10.0f, 10.0f));
    const float dinner = 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * x2);
    const float d = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * dinner;
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
