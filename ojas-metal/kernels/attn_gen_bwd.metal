// Entry points of the general attention backward (`Backend::attention_backward`
// on Metal). The bodies, the mask and the arithmetic are in
// `attn_bwd_tiled.h`, which `ojas_backend.metal`'s causal entry points share.
// `Dr = rowsum(dO ∘ O)` is `ojas_attn_bwd_dr` there.
//
//   ojas_attn_gen_bwd_dq_dDM     dQ
//   ojas_attn_gen_bwd_dkv_dDM    dK and dV
//   ojas_attn_gen_bwd_dbias_dDM  the bias gradient, when it is asked for
//
// DM is the compiled reduction width; the real head dim D <= DM. Buffers 0-7
// are the same for the three: Q, K, V, dO, lse, Dr, bias, key lengths. A
// buffer the call does not read (no bias, no key lengths) is still bound.
#include "attn_bwd_tiled.h"

/* MPP's type matching rejects const element types; nothing here writes Q,
   K, V or dO. */
#define AGB_INPUTS                                                            \
    const_cast<device float *>(q), const_cast<device float *>(k),             \
    const_cast<device float *>(v), const_cast<device float *>(dout)

#define OJAS_ATTN_GEN_BWD(DM)                                                 \
kernel void ojas_attn_gen_bwd_dq_d##DM(                                       \
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], \
    device const float *v [[buffer(2)]], device const float *dout [[buffer(3)]], \
    device const float *lse [[buffer(4)]], device const float *dvec [[buffer(5)]], \
    device const float *bias [[buffer(6)]], device const uint *lens [[buffer(7)]], \
    device float *dq [[buffer(8)]],                                           \
    constant uint &Tq [[buffer(9)]], constant uint &Tk [[buffer(10)]],        \
    constant uint &D [[buffer(11)]], constant uint &H [[buffer(12)]],         \
    constant uint &rep [[buffer(13)]], constant uint &flags [[buffer(14)]],   \
    constant uint &W [[buffer(15)]], constant uint &R [[buffer(16)]],         \
    constant uint &bias_b [[buffer(17)]], constant uint &bias_h [[buffer(18)]], \
    constant uint &bias_q [[buffer(19)]], constant float &scale [[buffer(20)]], \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]])                                 \
{                                                                             \
    threadgroup float S[ATT_BQ * ATT_BK];                                     \
    threadgroup float dP[ATT_BQ * ATT_BK];                                    \
    threadgroup float r0[ATT_BQ];                                             \
    threadgroup float r1[ATT_BQ];                                             \
    agb_dq_body<DM>(AGB_INPUTS, lse, dvec, bias, lens, dq,                    \
                    AgbMask{Tq, Tk, W, R, flags}, D, H, rep,                  \
                    AgbBias{bias_b, bias_h, bias_q}, scale, tgpig, tid,       \
                    S, dP, r0, r1);                                           \
}                                                                             \
kernel void ojas_attn_gen_bwd_dkv_d##DM(                                      \
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], \
    device const float *v [[buffer(2)]], device const float *dout [[buffer(3)]], \
    device const float *lse [[buffer(4)]], device const float *dvec [[buffer(5)]], \
    device const float *bias [[buffer(6)]], device const uint *lens [[buffer(7)]], \
    device float *dk [[buffer(8)]], device float *dv [[buffer(9)]],           \
    constant uint &Tq [[buffer(10)]], constant uint &Tk [[buffer(11)]],       \
    constant uint &D [[buffer(12)]], constant uint &H [[buffer(13)]],         \
    constant uint &rep [[buffer(14)]], constant uint &flags [[buffer(15)]],   \
    constant uint &W [[buffer(16)]], constant uint &R [[buffer(17)]],         \
    constant uint &bias_b [[buffer(18)]], constant uint &bias_h [[buffer(19)]], \
    constant uint &bias_q [[buffer(20)]], constant float &scale [[buffer(21)]], \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]])                                 \
{                                                                             \
    threadgroup float Pt[ATT_BQ * ATT_BK];                                    \
    threadgroup float dSt[ATT_BQ * ATT_BK];                                   \
    threadgroup float c0[ATT_BQ];                                             \
    threadgroup float c1[ATT_BQ];                                             \
    agb_dkv_body<DM>(AGB_INPUTS, lse, dvec, bias, lens, dk, dv,               \
                     AgbMask{Tq, Tk, W, R, flags}, D, H, rep,                 \
                     AgbBias{bias_b, bias_h, bias_q}, scale, tgpig, tid,      \
                     Pt, dSt, c0, c1);                                        \
}                                                                             \
kernel void ojas_attn_gen_bwd_dbias_d##DM(                                    \
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], \
    device const float *v [[buffer(2)]], device const float *dout [[buffer(3)]], \
    device const float *lse [[buffer(4)]], device const float *dvec [[buffer(5)]], \
    device const float *bias [[buffer(6)]], device const uint *lens [[buffer(7)]], \
    device float *dbias [[buffer(8)]],                                        \
    constant uint &Tq [[buffer(9)]], constant uint &Tk [[buffer(10)]],        \
    constant uint &D [[buffer(11)]], constant uint &H [[buffer(12)]],         \
    constant uint &rep [[buffer(13)]], constant uint &flags [[buffer(14)]],   \
    constant uint &W [[buffer(15)]], constant uint &R [[buffer(16)]],         \
    constant uint &bias_b [[buffer(17)]], constant uint &bias_h [[buffer(18)]], \
    constant uint &bias_q [[buffer(19)]], constant float &scale [[buffer(20)]], \
    constant uint &B [[buffer(21)]],                                          \
    uint2 tgpig [[threadgroup_position_in_grid]],                             \
    uint tid [[thread_index_in_threadgroup]])                                 \
{                                                                             \
    threadgroup float S[ATT_BQ * ATT_BK];                                     \
    threadgroup float dP[ATT_BQ * ATT_BK];                                    \
    threadgroup float r0[ATT_BQ];                                             \
    threadgroup float r1[ATT_BQ];                                             \
    agb_dbias_body<DM>(AGB_INPUTS, lse, dvec, bias, lens, dbias,              \
                       AgbMask{Tq, Tk, W, R, flags}, D, B, H, rep,            \
                       AgbBias{bias_b, bias_h, bias_q}, scale, tgpig, tid,    \
                       S, dP, r0, r1);                                        \
}

OJAS_ATTN_GEN_BWD(16)
OJAS_ATTN_GEN_BWD(32)
OJAS_ATTN_GEN_BWD(64)
OJAS_ATTN_GEN_BWD(128)
OJAS_ATTN_GEN_BWD(256)
