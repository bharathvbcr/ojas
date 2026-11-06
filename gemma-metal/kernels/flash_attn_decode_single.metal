// Single-query decode attention in one pass when every live key fits one chunk.
//
// Generated from tessl's `DECODE_PARTIAL_KERNEL` (kernels/flash_attn_decode.metal): the same
// partial pass, with the same chunk width, lanes and arithmetic, except that it is dispatched
// with one chunk and writes O itself instead of (m, l, acc) for the reduce pass. Over one
// chunk the reduce computes acc * exp(m - m) * (1 / l) = acc * (1 / l), which is what this
// writes, so the output is bit-identical and one dependent dispatch per layer goes away.
// Regenerate if tessl's partial kernel changes (tests/attn_decode_single.rs compares them).
//
// `VERIFY = 1` instances serve a verify block: grid row `b` is query row `b` of the block, at
// absolute position `q_pos_offset + b`, and every row reads the one K/V buffer. Each row then
// runs exactly the decode instance's arithmetic for its own causal key range.
#include <metal_stdlib>
using namespace metal;

constant uint SIMD_W = 32;

#define DECODE_SINGLE_KERNEL(NAME, D, CH, R, VERIFY)                         \
kernel void NAME(                                                             \
    device const float *Q [[buffer(0)]],                                      \
    device const float *K [[buffer(1)]],                                      \
    device const float *V [[buffer(2)]],                                      \
    device float *O [[buffer(3)]],                                               \
    constant uint &B [[buffer(4)]],                                           \
    constant uint &out_bf16 [[buffer(5)]],                                    \
    device const uint *Tkv_ptr [[buffer(6)]],                                 \
    constant uint &H [[buffer(7)]],                                           \
    constant uint &Hkv [[buffer(8)]],                                         \
    constant uint &window [[buffer(9)]],                                      \
    constant float &scale [[buffer(10)]],                                     \
    device const uint *q_pos_offset_ptr [[buffer(11)]],                       \
    device const uint *kv_pos_offset_ptr [[buffer(12)]],                      \
    constant uint &kv_capacity [[buffer(13)]],                                \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint2 tpitg [[thread_position_in_threadgroup]],                           \
    uint2 tptg [[threads_per_threadgroup]])                                   \
{                                                                             \
    /* Head dims per lane, as float4s -- a lane owns four *consecutive* dims  \
       per step, so one instruction moves 16R bytes across the R lanes of a   \
       key group instead of 4R. D % (4R) == 0 for every instantiation and     \
       `K + kv_base` is D-aligned, so the reads are aligned.                   \
                                                                              \
       This was measured *twice*, with opposite results, and the order        \
       matters. On the one-simdgroup-per-threadgroup kernel it lost 8-14% at  \
       every head dim: decode is latency bound, and 16 narrow loads left more \
       requests outstanding than 4 wide ones. Once the GQA group became       \
       co-resident -- four simdgroups in a threadgroup walking the same K/V   \
       -- the balance inverted, because the outstanding requests are now      \
       supplied by the other simdgroups and what is scarce is L1 bandwidth,   \
       which wide loads use better. It is worth 1.1x at D=512 and neutral     \
       elsewhere. A tuning result is only valid against the kernel it was     \
       measured on. */                                                        \
    constexpr uint DPV = (D) / (4u * (R));                                    \
    constexpr uint KPG = SIMD_W / (R); /* keys in flight per simdgroup */     \
    /* One simdgroup per query head, and the `H/Hkv` heads that share a KV    \
       head are put in the *same* threadgroup so they walk K and V together.   \
       Nothing is staged in threadgroup memory and there is no barrier: being  \
       co-resident is the whole mechanism, because the four heads of a 4:1     \
       group then touch each line while it is still in L1 instead of pulling   \
       it four separate times.                                                 \
                                                                              \
       That re-read was free of DRAM cost -- the same shape with Hkv = H       \
       issues identical bytes and takes 3x longer, which is what says the      \
       cache was absorbing it -- but not free of *cache* cost: `global512_     \
       decode_4k` was pushing 134 MB through the load path for 33.6 MB of      \
       unique K/V, and sat 2.4x above its own DRAM floor while the no-GQA      \
       shape sat on it.                                                        \
                                                                              \
       grid.y enumerates (batch, kv-head); the host sizes the threadgroup at   \
       `H/Hkv` simdgroups and falls back to one when that exceeds the Metal    \
       maximum, which the `sgs == 1` case below reproduces exactly. */         \
    const uint sgs = max(tptg.x / SIMD_W, 1u);                                \
    const uint sg = tpitg.x / SIMD_W;                                         \
    const uint lane = tpitg.x % SIMD_W;                                       \
    const uint grp = lane / (R);       /* which key-group */                  \
    const uint dl = lane % (R);        /* which dim slice */                  \
    /* Clamp mutable device state before it participates in any address. */   \
    const uint Tkv = min(*Tkv_ptr, kv_capacity);                              \
    const uint chunk = tgpig.x;                                               \
    /* grid.y enumerates (batch, head-block); a block is the `sgs` consecutive \
       query heads one threadgroup covers. `sgs == H/Hkv` puts exactly the     \
       heads of one KV group together, `sgs == H` puts every head of a batch   \
       item together -- which also makes the threadgroup's per-key read the    \
       full contiguous `[Hkv][D]` row instead of one strided slice of it --    \
       and `sgs == 1` is the original one-head-per-threadgroup dispatch. The   \
       host picks; all three fall out of the same arithmetic. */               \
    const uint blocks = max(H / sgs, 1u);                                     \
    const uint b = tgpig.y / blocks;                                          \
    const uint hb = tgpig.y % blocks;                                         \
    const uint h = hb * sgs + sg;                                             \
    if (h >= H) { return; }                                                   \
    const ulong bh = (ulong)b * H + h;                                        \
    const uint t_k0 = chunk * (CH);                                           \
    if (t_k0 >= Tkv) { return; }                                              \
    const uint n_k = min((uint)(CH), Tkv - t_k0);                             \
    const uint group = max(H / Hkv, 1u);                                      \
    const uint hkv = h / group;                                               \
    const ulong kv_pos_stride = (ulong)Hkv * (D);                             \
    const ulong kv_head_base = (VERIFY ? 0ul                                  \
        : (ulong)b * kv_capacity * kv_pos_stride) + (ulong)hkv * (D);         \
                                                                              \
    const ulong q_abs = (ulong)(*q_pos_offset_ptr) + (VERIFY ? (ulong)b : 0ul); \
    const ulong kv_pos_offset = (ulong)(*kv_pos_offset_ptr);                  \
    const ulong window_back = (window == 0u) ? 0ul : (ulong)window - 1ul;     \
    const ulong k_lo = (window == 0u || q_abs < window_back)                  \
        ? 0ul                                                                 \
        : q_abs - window_back;                                                 \
                                                                              \
    /* Tq == 1, so the single query row is index 0. */                        \
    const ulong q_off = bh * (D);                                             \
    device const float4 *Q4 = (device const float4 *)(Q + q_off);             \
    float4 q_reg[DPV];                                                        \
    for (uint j = 0; j < DPV; ++j) { q_reg[j] = Q4[dl + j * (R)]; }           \
                                                                              \
    float4 acc[DPV];                                                          \
    for (uint j = 0; j < DPV; ++j) { acc[j] = float4(0.0f); }                 \
    /* -FLT_MAX and `l_i > 0` as the seen flag, as in flash_attn_rows: fast   \
       math may assume no value is infinite. */                               \
    float m_i = -FLT_MAX;                                                     \
    float l_i = 0.0f;                                                         \
                                                                              \
    /* Live key sub-range of this chunk, in local indices. Tq == 1, so the    \
       mask bounds are a single interval every lane agrees on, which lets the \
       loop visit only unmasked keys instead of reading every one and masking \
       after. That distinction is worth 2x on a windowed decode: at           \
       window=1024 over Tkv=4096 three quarters of the keys are masked, and   \
       reading them costs full K and V traffic for nothing.                   \
                                                                              \
       Restoring this is a regression fix -- the `continue` the pre-R kernel   \
       had did the same job, and dropping it to keep the butterfly uniform    \
       cost 2x on `swa128_decode_b8_4k` before the sweep caught it. */        \
    const ulong local_lo = (k_lo > kv_pos_offset) ? k_lo - kv_pos_offset      \
                                                   : 0ul;                      \
    const ulong local_hi = (q_abs >= kv_pos_offset)                           \
        ? q_abs - kv_pos_offset + 1ul                                         \
        : 0ul;                                                                \
    const ulong lo_i = max((ulong)t_k0, local_lo);                            \
    const ulong hi_i = min((ulong)t_k0 + n_k, local_hi);                      \
    device bfloat *Ob = (device bfloat *)O;                                   \
    const ulong o_off = bh * (D);                                             \
    if (lo_i >= hi_i) {                                                       \
        /* Nothing unmasked: zeros, as the reduce pass writes for l == 0. */  \
        for (uint d = lane; d < (D); d += SIMD_W) {                           \
            if (out_bf16 != 0u) { Ob[o_off + d] = bfloat(0.0f); }             \
            else { O[o_off + d] = 0.0f; }                                     \
        }                                                                     \
        return;                                                               \
    }                                                                         \
    const uint live = (uint)(hi_i - lo_i);                                    \
    /* Uniform trip count across the KPG key-groups. The butterfly is a       \
       simdgroup-wide primitive, so a group that exited early would leave the \
       others shuffling against inactive lanes; the tail clamps its read to a \
       live key and masks the score instead. */                               \
    const uint iters = (live + KPG - 1u) / KPG;                               \
    for (uint it = 0; it < iters; ++it) {                                     \
        const uint t = it * KPG + grp;                                        \
        const uint tt = min(t, live - 1u);                                    \
        const uint key = (uint)lo_i + tt;                                     \
        const ulong kv_base = kv_head_base + (ulong)key * kv_pos_stride;      \
        device const float4 *K4 = (device const float4 *)(K + kv_base);       \
        float4 dot4 = float4(0.0f);                                           \
        for (uint j = 0; j < DPV; ++j) {                                      \
            dot4 += q_reg[j] * K4[dl + j * (R)];                              \
        }                                                                     \
        float part = dot4.x + dot4.y + dot4.z + dot4.w;                       \
        for (uint off = (R) / 2u; off > 0u; off >>= 1) {                      \
            part += simd_shuffle_xor(part, off);                              \
        }                                                                     \
        const bool key_live = t < live;                                       \
        const float s = key_live ? part * scale : -FLT_MAX;                   \
        const float m_new = max(m_i, s);                                      \
        const float alpha = (l_i > 0.0f) ? exp(m_i - m_new) : 0.0f;           \
        const float p = key_live ? exp(s - m_new) : 0.0f;                     \
        device const float4 *V4 = (device const float4 *)(V + kv_base);       \
        for (uint j = 0; j < DPV; ++j) {                                      \
            acc[j] = acc[j] * alpha + p * V4[dl + j * (R)];                   \
        }                                                                     \
        l_i = l_i * alpha + p;                                                \
        m_i = m_new;                                                          \
    }                                                                         \
                                                                              \
    /* Combine the KPG key-groups. Lanes holding the same `dl` sit R apart,   \
       so butterflying with offsets R, 2R, ... 16 sums across groups while    \
       keeping each lane on its own dims. At R=32 KPG is 1 and this loop does \
       not execute, which is the un-split kernel exactly. */                  \
    float m_all = m_i;                                                        \
    for (uint off = (R); off < SIMD_W; off <<= 1) {                           \
        m_all = max(m_all, simd_shuffle_xor(m_all, off));                     \
    }                                                                         \
    /* A group that saw no key (l_i == 0) contributes nothing. At least one   \
       group saw a key (the chunk is not empty), so m_all is a real score. */ \
    const float w = (l_i > 0.0f) ? exp(m_i - m_all) : 0.0f;                   \
    float l_all = l_i * w;                                                    \
    for (uint off = (R); off < SIMD_W; off <<= 1) {                           \
        l_all += simd_shuffle_xor(l_all, off);                                \
    }                                                                         \
    for (uint j = 0; j < DPV; ++j) {                                          \
        float4 a = acc[j] * w;                                                \
        for (uint off = (R); off < SIMD_W; off <<= 1) {                       \
            a.x += simd_shuffle_xor(a.x, off);                                \
            a.y += simd_shuffle_xor(a.y, off);                                \
            a.z += simd_shuffle_xor(a.z, off);                                \
            a.w += simd_shuffle_xor(a.w, off);                                \
        }                                                                     \
        acc[j] = a;                                                           \
    }                                                                         \
                                                                              \
    /* Group 0's R lanes cover dims dl + j*R for all j, i.e. every dim, so it \
       alone writes. The layout stays canonical [m, l, acc[D]] whatever R is, \
       which is why the reduce pass needs no R of its own. */                 \
    if (grp != 0u) { return; }                                                \
    /* The reduce pass over one chunk: a = acc * exp(m - m) = acc exactly,    \
       then a * (1 / l). */                                                   \
    const float inv_l = (l_all > 0.0f) ? (1.0f / l_all) : 0.0f;               \
    for (uint j = 0; j < DPV; ++j) {                                          \
        const uint d0 = 4u * (dl + j * (R));                                  \
        const float4 o = acc[j] * inv_l;                                      \
        if (out_bf16 != 0u) {                                                 \
            Ob[o_off + d0 + 0u] = bfloat(o.x);                                \
            Ob[o_off + d0 + 1u] = bfloat(o.y);                                \
            Ob[o_off + d0 + 2u] = bfloat(o.z);                                \
            Ob[o_off + d0 + 3u] = bfloat(o.w);                                \
        } else {                                                              \
            O[o_off + d0 + 0u] = o.x;                                         \
            O[o_off + d0 + 1u] = o.y;                                         \
            O[o_off + d0 + 2u] = o.z;                                         \
            O[o_off + d0 + 3u] = o.w;                                         \
        }                                                                     \
    }                                                                         \
    (void)B;                                                                  \
}

DECODE_SINGLE_KERNEL(flash_attn_decode_single_h256_c128_r16, 256, 128, 16, 0)
DECODE_SINGLE_KERNEL(flash_attn_decode_single_h512_c128_r32, 512, 128, 32, 0)
DECODE_SINGLE_KERNEL(flash_attn_verify_single_h256_c128_r16, 256, 128, 16, 1)
DECODE_SINGLE_KERNEL(flash_attn_verify_single_h512_c128_r32, 512, 128, 32, 1)
