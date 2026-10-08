// Stem + smear + embedding gather backward.
#include <metal_stdlib>
using namespace metal;

kernel void stem_smear_bwd_f32(
    device const float *x_norm [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device const float *dout [[buffer(2)]],
    device float *dx_norm [[buffer(3)]],
    device atomic_float *dgate [[buffer(4)]],
    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &C [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= B * T) return;
    const uint t = gid % T;
    const uint b = gid / T;
    const uint base = gid * C;
    const uint prev_base = (b * T + (t - 1)) * C;
    for (uint c = 0; c < C; ++c) {
        float g = 1.0f / (1.0f + exp(-gate[c]));
        float d = dout[base + c];
        // out = (1-g)*xt + g*xp
        atomic_fetch_add_explicit((device atomic_float *)&dx_norm[base + c], (1.0f - g) * d, memory_order_relaxed);
        if (t > 0) {
            atomic_fetch_add_explicit((device atomic_float *)&dx_norm[prev_base + c], g * d, memory_order_relaxed);
        }
        float xt = x_norm[base + c];
        float xp = (t == 0) ? 0.0f : x_norm[prev_base + c];
        // dL/dg = (xp - xt) * d; dg/dgate = g*(1-g)
        float dg = (xp - xt) * d;
        atomic_fetch_add_explicit(&dgate[c], dg * g * (1.0f - g), memory_order_relaxed);
    }
}

/// Embedding + bigram grads from d(pre_norm).
kernel void stem_embed_bwd_f32(
    device const int *input_ids [[buffer(0)]],
    device const float *bigram_emb [[buffer(1)]],  // [Vb, Db]
    device const float *bigram_proj [[buffer(2)]], // [Db, C]
    device const float *bigram_scale [[buffer(3)]],
    device const float *d_pre [[buffer(4)]],       // [B,T,C]
    device atomic_float *d_tok [[buffer(5)]],      // [V, C]
    device atomic_float *d_bigram_emb [[buffer(6)]],
    device atomic_float *d_bigram_proj [[buffer(7)]],
    device atomic_float *d_bigram_scale [[buffer(8)]],
    constant uint &B [[buffer(9)]],
    constant uint &T [[buffer(10)]],
    constant uint &C [[buffer(11)]],
    constant uint &Vb [[buffer(12)]],
    constant uint &Db [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= B * T) return;
    const uint b = gid / T;
    const uint t = gid % T;
    const int tok = input_ids[gid];
    const float scale = *bigram_scale;
    const int mod = (int)Vb - 1;

    int hash_idx;
    if (t == 0) {
        hash_idx = mod;
    } else {
        const int t_cur = input_ids[gid];
        const int t_prev = input_ids[b * T + (t - 1)];
        const int xored = (36313 * t_cur) ^ (27191 * t_prev);
        int r = xored % mod;
        if (r < 0) r += mod;
        hash_idx = r;
    }

    const uint base = gid * C;
    // d_tok[tok] += d_pre
    for (uint c = 0; c < C; ++c) {
        atomic_fetch_add_explicit(&d_tok[(uint)tok * C + c], d_pre[base + c], memory_order_relaxed);
    }

    // out_c += scale * sum_d emb[d] * proj[d,c]
    // d_proj[d,c] += scale * emb[d] * d_pre[c]
    // d_emb[d] += scale * sum_c proj[d,c] * d_pre[c]
    // d_scale += sum_c hsum_c * d_pre[c]
    const device float *erow = bigram_emb + (uint)hash_idx * Db;
    float dscale_acc = 0.0f;
    for (uint d = 0; d < Db; ++d) {
        float demb = 0.0f;
        float e = erow[d];
        for (uint c = 0; c < C; ++c) {
            float dp = d_pre[base + c];
            float p = bigram_proj[d * C + c];
            atomic_fetch_add_explicit(&d_bigram_proj[d * C + c], scale * e * dp, memory_order_relaxed);
            demb += p * dp;
            dscale_acc += e * p * dp;
        }
        atomic_fetch_add_explicit(&d_bigram_emb[(uint)hash_idx * Db + d], scale * demb, memory_order_relaxed);
    }
    atomic_fetch_add_explicit(d_bigram_scale, dscale_acc, memory_order_relaxed);
}
