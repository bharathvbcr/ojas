// Stem embed gather helpers for GEMM-based bigram path (Phase D).
#include <metal_stdlib>
using namespace metal;

/// Gather token rows into out and bigram embedding rows into bg_rows; also write hash indices.
kernel void stem_gather_f32(
    device const int *input_ids [[buffer(0)]],
    device const float *tok_emb [[buffer(1)]],     // [V, C]
    device const float *bigram_emb [[buffer(2)]],  // [Vb, Db]
    device float *tok_out [[buffer(3)]],           // [BT, C]
    device float *bg_rows [[buffer(4)]],           // [BT, Db]
    device int *hash_idx [[buffer(5)]],            // [BT]
    constant uint &B [[buffer(6)]],
    constant uint &T [[buffer(7)]],
    constant uint &C [[buffer(8)]],
    constant uint &Vb [[buffer(9)]],
    constant uint &Db [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= B * T) return;
    const uint b = gid / T;
    const uint t = gid % T;
    const int tok = input_ids[gid];
    if (tok < 0) {
        hash_idx[gid] = 0;
        for (uint c = 0; c < C; ++c) tok_out[gid * C + c] = 0.0f;
        for (uint d = 0; d < Db; ++d) bg_rows[gid * Db + d] = 0.0f;
        return;
    }
    const int mod = (int)Vb - 1;

    int hidx;
    if (t == 0) {
        hidx = mod;
    } else {
        const int t_cur = input_ids[gid];
        const int t_prev = input_ids[b * T + (t - 1)];
        const int xored = (36313 * t_cur) ^ (27191 * t_prev);
        int r = xored % mod;
        if (r < 0) r += mod;
        hidx = r;
    }
    hash_idx[gid] = hidx;

    const device float *trow = tok_emb + (uint)tok * C;
    device float *tout = tok_out + gid * C;
    for (uint c = 0; c < C; ++c) tout[c] = trow[c];

    const device float *brow = bigram_emb + (uint)hidx * Db;
    device float *bout = bg_rows + gid * Db;
    for (uint d = 0; d < Db; ++d) bout[d] = brow[d];
}

/// tok_out += scale * bg_proj_out  (in-place on tok_out which becomes stem pre-norm).
kernel void stem_axpy_scale_f32(
    device float *tok_out [[buffer(0)]],           // [BT, C]
    device const float *bg_proj [[buffer(1)]],     // [BT, C]
    device const float *bigram_scale [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    tok_out[gid] += (*bigram_scale) * bg_proj[gid];
}

/// Scatter-add dense [BT, C] tok grads into embedding table.
kernel void stem_scatter_tok_f32(
    device const int *input_ids [[buffer(0)]],
    device const float *d_pre [[buffer(1)]],       // [BT, C]
    device atomic_float *d_tok [[buffer(2)]],      // [V, C]
    constant uint &BT [[buffer(3)]],
    constant uint &C [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BT) return;
    const int tok = input_ids[gid];
    if (tok < 0) return;
    const uint base = gid * C;
    for (uint c = 0; c < C; ++c) {
        atomic_fetch_add_explicit(&d_tok[(uint)tok * C + c], d_pre[base + c], memory_order_relaxed);
    }
}

/// d_bg_proj = d_pre * scale; also accumulate d_scale from bg_proj_out · d_pre.
kernel void stem_scale_bwd_f32(
    device const float *d_pre [[buffer(0)]],       // [BT, C]
    device const float *bg_proj_out [[buffer(1)]], // [BT, C] = emb @ proj (pre-scale)
    device const float *bigram_scale [[buffer(2)]],
    device float *d_bg_proj [[buffer(3)]],         // [BT, C]
    device atomic_float *d_scale [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    threadgroup float shared[256];
    float local = 0.0f;
    if (gid < n) {
        float dp = d_pre[gid];
        d_bg_proj[gid] = dp * (*bigram_scale);
        local = bg_proj_out[gid] * dp;
    }
    shared[lid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < tpg; stride <<= 1) {
        uint idx = lid * (stride << 1);
        if (idx + stride < tpg) {
            shared[idx] += shared[idx + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        atomic_fetch_add_explicit(d_scale, shared[0], memory_order_relaxed);
    }
}

/// Scatter-add dense [BT, Db] emb grads using taped hash indices.
kernel void stem_scatter_bigram_f32(
    device const int *hash_idx [[buffer(0)]],
    device const float *d_emb_dense [[buffer(1)]], // [BT, Db]
    device atomic_float *d_bigram_emb [[buffer(2)]],
    constant uint &BT [[buffer(3)]],
    constant uint &Db [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BT) return;
    const int hidx = hash_idx[gid];
    const uint base = gid * Db;
    for (uint d = 0; d < Db; ++d) {
        atomic_fetch_add_explicit(
            &d_bigram_emb[(uint)hidx * Db + d],
            d_emb_dense[base + d],
            memory_order_relaxed);
    }
}
