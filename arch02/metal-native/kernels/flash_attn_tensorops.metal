// TensorOps cooperative-tensor flash attention (WWDC26 session 330 recipe).
//
// Session 330 demonstrates *single-tile* QK^T → softmax → P@V with cooperative
// left-input (macOS 26.3+). That recipe is NOT full multi-block online FA-2:
// there is no published guidance for carrying O/m/l across key blocks, causal
// partial-tile masking inside cooperative score tensors, GQA scatter, or bwd
// with taped LSE. See DECISIONS M8.
//
// Hot path: simdgroup FA-2 in flash_attn_fwd.metal / flash_attn_bwd.metal.
// This file ships (1) the single-tile probe and (2) an experimental multi-block
// online FA fwd that stages O in threadgroup memory between TensorOps tiles.
// The multi-block kernel is compiled + smoke-tested but NOT the training default
// until goldens and Instruments NAX beat simdgroup FA-2.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

/// Single Br×Bc tile attention via TensorOps (D=32). Smoke-tests cooperative
/// left-input layout for our shapes — not used for training.
kernel void flash_attn_tensorops_tile_f32(
    device float *Q [[buffer(0)]],
    device float *K [[buffer(1)]],
    device float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    device float *L [[buffer(4)]],
    constant float &scale [[buffer(5)]])
{
    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int D = 32;

    constexpr auto desc_qk = matmul2d_descriptor(
        Br, Bc, D, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_qk, execution_simdgroup> op_qk;

    auto mQ = tensor(Q, dextents<int, 2>{D, Br}, array<int, 2>{1, D});
    auto mK = tensor(K, dextents<int, 2>{D, Bc}, array<int, 2>{1, D});

    auto sT = op_qk.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
#pragma unroll
    for (ushort i = 0; i < sT.get_capacity(); ++i) {
        if (sT.is_valid_element(i)) sT[i] = 0.0f;
    }
    op_qk.run(mQ, mK, sT);

#pragma unroll
    for (ushort i = 0; i < sT.get_capacity(); ++i) {
        if (sT.is_valid_element(i)) sT[i] *= scale;
    }

    auto rowMax = op_qk.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(sT, rowMax, reduction_operation::max, -INFINITY);

    if (is_iterator_compatible(sT, rowMax)) {
        for (auto it = sT.begin(); it != sT.end(); ++it) {
            if (!it.is_valid_element()) continue;
            auto m_it = rowMax.map_iterator(it);
            *it = metal::exp(*it - *m_it);
        }
    }

    auto rowSum = op_qk.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(sT, rowSum, reduction_operation::sum, 0.0f);

    if (is_iterator_compatible(sT, rowSum)) {
        for (auto it = sT.begin(); it != sT.end(); ++it) {
            if (!it.is_valid_element()) continue;
            auto s_it = rowSum.map_iterator(it);
            *it = *it / (*s_it + 1e-20f);
        }
    }

    for (auto it = rowMax.begin(); it != rowMax.end(); ++it) {
        if (!it.is_valid_element()) continue;
        auto ridx = it.get_multidimensional_index();
        int row = (int)ridx[1];
        float mval = *it;
        float sval = 0.0f;
        for (auto sit = rowSum.begin(); sit != rowSum.end(); ++sit) {
            if (!sit.is_valid_element()) continue;
            auto sidx = sit.get_multidimensional_index();
            if (sidx[1] == ridx[1]) {
                sval = *sit;
                break;
            }
        }
        if (row >= 0 && row < Br) {
            L[row] = mval + metal::log(sval + 1e-20f);
        }
    }

    constexpr auto desc_pv = matmul2d_descriptor(
        Br, D, Bc, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_pv, execution_simdgroup> op_pv;
    auto mV = tensor(V, dextents<int, 2>{D, Bc}, array<int, 2>{1, D});
    auto mO = tensor(O, dextents<int, 2>{D, Br}, array<int, 2>{1, D});
    auto oT = op_pv.get_destination_cooperative_tensor<decltype(mQ), decltype(mV), float>();

    bool compatible = op_pv.is_compatible_as_left_input<float, float, float>(sT);
    if (compatible) {
        auto pIn = op_pv.get_left_input_cooperative_tensor<float, float, float>(sT);
#pragma unroll
        for (ushort i = 0; i < oT.get_capacity(); ++i) {
            if (oT.is_valid_element(i)) oT[i] = 0.0f;
        }
        op_pv.run(pIn, mV, oT);
        oT.store(mO);
    } else {
        threadgroup float p_smem[Br * Bc];
        auto mPtg = tensor(p_smem, dextents<int, 2>{Bc, Br}, array<int, 2>{1, Bc});
        sT.store(mPtg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mP = tensor(p_smem, dextents<int, 2>{Bc, Br}, array<int, 2>{1, Bc});
#pragma unroll
        for (ushort i = 0; i < oT.get_capacity(); ++i) {
            if (oT.is_valid_element(i)) oT[i] = 0.0f;
        }
        op_pv.run(mP, mV, oT);
        oT.store(mO);
    }
}

/// Experimental multi-block causal GQA online FA-2 using TensorOps per tile.
///
/// Blockers vs production hot-path (documented in DECISIONS M8):
/// 1. Session 330 has no multi-block online-softmax recipe — O/m/l carry requires
///    threadgroup staging of O between tiles (partially defeats cooperative-input).
/// 2. Causal masking + ragged last tiles must mutate cooperative score elements
///    before reduce_rows; layout-dependent and brittle at D=32.
/// 3. No TensorOps bwd+LSE analogue — training still needs simdgroup bwd.
/// 4. `is_compatible_as_left_input` often false at D=32 → TG round-trip for P@V.
///
/// Dispatch: threadgroups = (q_blocks, B*H), threads = 32 (one simdgroup).
/// Q/O: [B,T,H,D], K/V: [B,T,Hkv,D], L: [B,H,T]. Requires D==32, Br=Bc=32.
kernel void flash_attn_tensorops_online_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    device float *L [[buffer(4)]],
    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant uint &D [[buffer(9)]],
    constant float &scale [[buffer(10)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_simdgroup]],
    uint sid [[simdgroup_index_in_threadgroup]])
{
    // Only simdgroup 0 participates (one simdgroup per query block).
    if (sid != 0) return;
    if (D != 32u) return;

    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int Dd = 32;

    const uint q_block = tgpig.x;
    const uint bh = tgpig.y;
    const uint h = bh % H;
    const uint b = bh / H;
    const uint group = H / Hkv;
    const uint hkv = h / group;
    const uint t_q0 = q_block * Br;
    if (t_q0 >= T) return;

    const uint t_q_max = min(t_q0 + (uint)Br, T) - 1u;
    const uint n_k_blocks = (t_q_max / Bc) + 1u;

    // Online FA state in threadgroup (row-owned by lane when lid < Br). Q/K/V
    // stay in device memory: tensor views preserve the interleaved head stride,
    // avoid three 4 KiB staging arrays, and let TensorOps handle ragged edge
    // loads. Scores and probabilities have disjoint lifetimes, so one tile is
    // their canonical scratch owner. Besides removing copies and barriers,
    // this keeps the 12.25 KiB static allocation below Shader Validation's
    // instrumented threadgroup-memory ceiling.
    threadgroup float O_acc[Br * Dd];
    threadgroup float m_row[Br];
    threadgroup float l_row[Br];
    threadgroup float SP_tile[Br * Bc];
    threadgroup float O_tile[Br * Dd];

    // Init online state. The strided Q view below is read directly per key
    // block; it remains cache-resident and avoids a threadgroup round-trip.
    for (uint i = lid; i < Br * Dd; i += 32) {
        O_acc[i] = 0.0f;
    }
    if (lid < Br) {
        m_row[lid] = -INFINITY;
        l_row[lid] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto desc_qk = matmul2d_descriptor(
        Br, Bc, Dd, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto desc_pv = matmul2d_descriptor(
        Br, Dd, Bc, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_qk, execution_simdgroup> op_qk;
    matmul2d<desc_pv, execution_simdgroup> op_pv;

    for (uint kb = 0; kb < n_k_blocks; ++kb) {
        const uint t_k0 = kb * Bc;
        const uint n_k = min((uint)Bc, T - t_k0);
        const uint n_q = min((uint)Br, T - t_q0);
        const ulong q_off = ((ulong)b * T * H + (ulong)t_q0 * H + h) * D;
        const ulong kv_off = ((ulong)b * T * Hkv + (ulong)t_k0 * Hkv + hkv) * D;
        auto mQ = tensor(const_cast<device float *>(Q) + q_off,
                         dextents<int, 2>{Dd, (int)n_q},
                         array<int, 2>{1, (int)(H * D)});
        auto mK = tensor(const_cast<device float *>(K) + kv_off,
                         dextents<int, 2>{Dd, (int)n_k},
                         array<int, 2>{1, (int)(Hkv * D)});
        auto sT = op_qk.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
#pragma unroll
        for (ushort i = 0; i < sT.get_capacity(); ++i) {
            if (sT.is_valid_element(i)) sT[i] = 0.0f;
        }
        op_qk.run(mQ, mK, sT);

        // Scale + causal / ragged mask into -inf, then store scores to TG for
        // online FA (cooperative max alone is full-tile softmax, not online).
        auto mS = tensor(SP_tile, dextents<int, 2>{Bc, Br}, array<int, 2>{1, Bc});
        sT.store(mS);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = lid; i < Br * Bc; i += 32) {
            const uint row = i / Bc;
            const uint col = i % Bc;
            const uint t_q = t_q0 + row;
            const uint t_k = t_k0 + col;
            float s = -INFINITY;
            // A ragged TensorOps destination is permitted to leave edge
            // elements unstored. Test bounds before reading the shared tile;
            // masking an already-read uninitialized value is still UB.
            if (t_q < T && col < n_k && t_k <= t_q) {
                s = SP_tile[i] * scale;
            }
            SP_tile[i] = s;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Per-row online update (lane owns one query row).
        if (lid < Br) {
            const uint row = lid;
            const uint t_q = t_q0 + row;
            float m_i = m_row[row];
            float l_i = l_row[row];
            float m_ij = -INFINITY;
            if (t_q < T) {
                for (uint col = 0; col < n_k; ++col) {
                    m_ij = max(m_ij, SP_tile[row * Bc + col]);
                }
            }
            const float m_new = max(m_i, m_ij);
            const float alpha = (m_i > -INFINITY) ? metal::exp(m_i - m_new) : 0.0f;

            float row_sum = 0.0f;
            for (uint col = 0; col < Bc; ++col) {
                float p = 0.0f;
                if (t_q < T && col < n_k) {
                    float s = SP_tile[row * Bc + col];
                    p = (s > -INFINITY) ? metal::exp(s - m_new) : 0.0f;
                }
                SP_tile[row * Bc + col] = p;
                row_sum += p;
            }
            l_i = l_i * alpha + row_sum;

            for (uint d = 0; d < Dd; ++d) {
                O_acc[row * Dd + d] *= alpha;
            }
            m_row[row] = m_new;
            l_row[row] = l_i;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // P @ V via TensorOps (P from TG — cooperative-input often incompatible).
        auto mP = tensor(SP_tile, dextents<int, 2>{Bc, Br}, array<int, 2>{1, Bc});
        auto mV = tensor(const_cast<device float *>(V) + kv_off,
                         dextents<int, 2>{Dd, (int)n_k},
                         array<int, 2>{1, (int)(Hkv * D)});
        auto mOt = tensor(O_tile, dextents<int, 2>{Dd, Br}, array<int, 2>{1, Dd});
        auto oT = op_pv.get_destination_cooperative_tensor<decltype(mP), decltype(mV), float>();
#pragma unroll
        for (ushort i = 0; i < oT.get_capacity(); ++i) {
            if (oT.is_valid_element(i)) oT[i] = 0.0f;
        }
        op_pv.run(mP, mV, oT);
        oT.store(mOt);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = lid; i < Br * Dd; i += 32) {
            O_acc[i] += O_tile[i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize + write O/L.
    if (lid < Br) {
        const uint row = lid;
        const uint t_q = t_q0 + row;
        if (t_q < T) {
            const float inv_l = 1.0f / max(l_row[row], 1e-20f);
            const uint o_off = ((b * T + t_q) * H + h) * D;
            for (uint d = 0; d < Dd; ++d) {
                O[o_off + d] = O_acc[row * Dd + d] * inv_l;
            }
            L[(b * H + h) * T + t_q] = m_row[row] + metal::log(max(l_row[row], 1e-20f));
        }
    }
}
