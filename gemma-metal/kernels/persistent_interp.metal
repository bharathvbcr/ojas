// Persistent-interpreter prototype (mini graph only) — Hazy-style stand-in.
//
// Metal has **no grid-wide forward-progress guarantee**. A literal single
// persistent megakernel can deadlock if consumer TGs spin on atomics while
// producer TGs are not resident. This prototype:
//   - launches a *tiny* TG count (mini dims) that fits residually in practice
//   - walks a host-built instruction stream
//   - uses device atomics for a grid barrier between PRODUCE and CONSUME
//   - caps spin waits and sets a fail flag instead of hanging forever
//
// Metal API note: device-scope atomics only accept `memory_order_relaxed`
// (no acquire/release on device). Barrier signaling is therefore relaxed +
// spin; non-atomic `mid[]` visibility relies on GPU coherence after the
// atomic edge — acceptable for a mini correctness prototype, not a proof
// of production memory ordering.
//
// Doctrine home: edges that need all of a producer's output
// (`gate/up → down`, `FA → o_proj`). **Not** for 31B/E4B default decode.
//
// Program for the gate→down stand-in:
//   PRODUCE_MID → GRID_BARRIER → DOWN_PROJ → HALT
// where mid[i] = gelu_pytorch_tanh(gate[i]) * up[i]
// and   out[r] = sum_i mid[i] * W_down[r, i]   (dense f32, mini only)
//
// Program for the FA→o_proj stand-in (sibling):
//   PRODUCE_CTX → GRID_BARRIER → O_PROJ → HALT
// where ctx[i] = tanh(q[i] * k[i] * scale) * v[i]   (element-local FA mock)
// and   out[r] = sum_i ctx[i] * W_o[r, i]           (dense f32, mini only)
// Not real softmax FA — only the grid-sync dependency shape.

#include <metal_stdlib>
#include "gelu.h"
using namespace metal;

constant uint OP_HALT        = 0u;
constant uint OP_PRODUCE_MID = 1u;
constant uint OP_BARRIER     = 2u;
constant uint OP_DOWN_PROJ   = 3u;

/// Sense-reversing grid barrier via device atomics (relaxed only — MSL limit).
/// `deps[0]` = arrival count; `deps[1]` = generation.
/// Only safe when all `n_tg` threadgroups are (or become) resident.
static inline void grid_barrier(
    device atomic_uint *deps,
    device atomic_uint *fail,
    uint n_tg,
    uint max_spin,
    uint tid,
    threadgroup uint &tg_done)
{
    if (tid == 0u) {
        uint gen = atomic_load_explicit(&deps[1], memory_order_relaxed);
        uint arrived =
            atomic_fetch_add_explicit(&deps[0], 1u, memory_order_relaxed) + 1u;
        if (arrived == n_tg) {
            atomic_store_explicit(&deps[0], 0u, memory_order_relaxed);
            atomic_fetch_add_explicit(&deps[1], 1u, memory_order_relaxed);
            tg_done = 1u;
        } else {
            uint spins = 0u;
            while (atomic_load_explicit(&deps[1], memory_order_relaxed) == gen) {
                spins += 1u;
                if (spins > max_spin) {
                    atomic_store_explicit(fail, 1u, memory_order_relaxed);
                    break;
                }
            }
            tg_done = 1u;
        }
    }
    // Intra-TG wait for tid0; also flush TG-local views of device stores.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    (void)tg_done;
}

kernel void persistent_interp_gate_down(
    device const uint *insns [[buffer(0)]],
    constant uint &n_insns [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    device const float *up [[buffer(3)]],
    device float *mid [[buffer(4)]],
    device const float *w_down [[buffer(5)]],
    device float *out [[buffer(6)]],
    device atomic_uint *deps [[buffer(7)]],
    device atomic_uint *fail [[buffer(8)]],
    constant uint &n_mid [[buffer(9)]],
    constant uint &n_out [[buffer(10)]],
    constant uint &n_tg [[buffer(11)]],
    constant uint &max_spin [[buffer(12)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    if (tgid >= n_tg) {
        return;
    }

    threadgroup uint tg_done;
    uint pc = 0u;
    while (pc < n_insns) {
        uint op = insns[pc];
        pc += 1u;

        if (op == OP_HALT) {
            break;
        }

        if (op == OP_PRODUCE_MID) {
            // Partition mid across TGs; threads stride within the TG slice.
            uint chunk = (n_mid + n_tg - 1u) / n_tg;
            uint begin = tgid * chunk;
            uint end = min(begin + chunk, n_mid);
            for (uint i = begin + tid; i < end; i += tptg) {
                mid[i] = tessl_gelu_pytorch_tanh(gate[i]) * up[i];
            }
            threadgroup_barrier(mem_flags::mem_device);
            continue;
        }

        if (op == OP_BARRIER) {
            grid_barrier(deps, fail, n_tg, max_spin, tid, tg_done);
            // Bail if any TG timed out (fail may be set by a peer).
            if (atomic_load_explicit(fail, memory_order_relaxed) != 0u) {
                break;
            }
            continue;
        }

        if (op == OP_DOWN_PROJ) {
            // Needs *all* of mid — the grid-wide dep the barrier established.
            uint chunk = (n_out + n_tg - 1u) / n_tg;
            uint begin = tgid * chunk;
            uint end = min(begin + chunk, n_out);
            for (uint r = begin + tid; r < end; r += tptg) {
                float acc = 0.0f;
                uint base = r * n_mid;
                for (uint i = 0u; i < n_mid; i++) {
                    acc += mid[i] * w_down[base + i];
                }
                out[r] = acc;
            }
            threadgroup_barrier(mem_flags::mem_device);
            continue;
        }

        // Unknown opcode — treat as halt.
        break;
    }
}

// --- Hot Q4 gate→down (bounded TG, E4B opt-in via GEMMA_METAL_FUSE_GATE_DOWN) ---
//
// Same instruction stream as dense stand-in; PRODUCE/DOWN peel Q4 simd math from
// gemv_q4_mlx_simd_gate_up_gelu / gemv_q4_mlx_simd_add with outer row tiles
// (n_tg ≤ 8, not the full 1280-TG GEMV grid).

constant uint PI_Q4_SIMD_SIZE = 32u;
constant uint PI_Q4_SIMD_ROWS = 4u;
constant uint PI_Q4_SIMD_SG_PER_TG = 2u;
constant uint PI_Q4_SIMD_PACKS = 2u;
constant uint PI_Q4_SIMD_VPT = 8u * PI_Q4_SIMD_PACKS;
constant uint PI_Q4_SIMD_BLOCK = PI_Q4_SIMD_SIZE * PI_Q4_SIMD_VPT;

static inline float pi_load_x16_qdot(device const bfloat *x, thread float *xp) {
    bfloat4 x0 = ((device const bfloat4 *)(x))[0];
    bfloat4 x1 = ((device const bfloat4 *)(x + 4u))[0];
    bfloat4 x2 = ((device const bfloat4 *)(x + 8u))[0];
    bfloat4 x3 = ((device const bfloat4 *)(x + 12u))[0];
    float a0 = float(x0.x), a1 = float(x0.y), a2 = float(x0.z), a3 = float(x0.w);
    float a4 = float(x1.x), a5 = float(x1.y), a6 = float(x1.z), a7 = float(x1.w);
    float a8 = float(x2.x), a9 = float(x2.y), a10 = float(x2.z), a11 = float(x2.w);
    float a12 = float(x3.x), a13 = float(x3.y), a14 = float(x3.z), a15 = float(x3.w);
    float sum = (a0 + a1 + a2 + a3) + (a4 + a5 + a6 + a7)
              + (a8 + a9 + a10 + a11) + (a12 + a13 + a14 + a15);
    xp[0] = a0;             xp[1] = a1 / 16.0f;   xp[2] = a2 / 256.0f;  xp[3] = a3 / 4096.0f;
    xp[4] = a4;             xp[5] = a5 / 16.0f;   xp[6] = a6 / 256.0f;  xp[7] = a7 / 4096.0f;
    xp[8] = a8;             xp[9] = a9 / 16.0f;   xp[10] = a10 / 256.0f; xp[11] = a11 / 4096.0f;
    xp[12] = a12;           xp[13] = a13 / 16.0f; xp[14] = a14 / 256.0f; xp[15] = a15 / 4096.0f;
    return sum;
}

static inline float pi_qdot16(
    device const uchar *w,
    thread const float *xp,
    float scale,
    float bias,
    float xsum)
{
    device const ushort *ws = (device const ushort *)w;
    float accum = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        const ushort ww = ws[i];
        accum += xp[4u * i] * float(ww & 0x000fu)
               + xp[4u * i + 1u] * float(ww & 0x00f0u)
               + xp[4u * i + 2u] * float(ww & 0x0f00u)
               + xp[4u * i + 3u] * float(ww & 0xf000u);
    }
    return scale * accum + xsum * bias;
}

// Single `mid` slab (like gemv_q4_mlx_simd_gate_up_gelu): never bind the same
// allocation as both float* and bfloat* buffer args — MSL assumes non-aliasing.
static inline void pi_q4_produce_tile_rm(
    uint row0,
    uint row_limit,
    device const uchar *gate_packed,
    device const bfloat2 *gate_sb,
    device const uchar *up_packed,
    device const bfloat2 *up_sb,
    device const bfloat *x,
    device float *mid,
    uint cols,
    uint group_size,
    uint mid_as_bf16,
    uint sgid,
    uint lane)
{
    const uint row_base = row0 + sgid * PI_Q4_SIMD_ROWS;
    if (row_base >= row_limit) {
        return;
    }
    const uint gpr = cols / group_size;
    const uint row_bytes = cols >> 1u;
    const uint lane_col0 = lane * PI_Q4_SIMD_VPT;
    const uint sb_k_step = PI_Q4_SIMD_BLOCK / group_size;
    float acc_g[PI_Q4_SIMD_ROWS];
    float acc_u[PI_Q4_SIMD_ROWS];
    for (uint r = 0u; r < PI_Q4_SIMD_ROWS; ++r) {
        acc_g[r] = 0.0f;
        acc_u[r] = 0.0f;
    }
    device const uchar *gws = gate_packed + row_base * row_bytes + (lane_col0 >> 1u);
    device const uchar *uws = up_packed + row_base * row_bytes + (lane_col0 >> 1u);
    device const bfloat2 *gsbr = gate_sb + row_base * gpr + (lane_col0 / group_size);
    device const bfloat2 *usbr = up_sb + row_base * gpr + (lane_col0 / group_size);
    device const bfloat *xr = x + lane_col0;
    for (uint k0 = 0u; k0 < cols; k0 += PI_Q4_SIMD_BLOCK) {
        if (k0 + lane_col0 + PI_Q4_SIMD_VPT <= cols) {
            float xt[16];
            const float xsum = pi_load_x16_qdot(xr, xt);
            for (uint r = 0u; r < PI_Q4_SIMD_ROWS; ++r) {
                const uint row = row_base + r;
                if (row >= row_limit) {
                    break;
                }
                const bfloat2 gv = gsbr[r * gpr];
                const bfloat2 uv = usbr[r * gpr];
                acc_g[r] += pi_qdot16(gws + r * row_bytes, xt, float(gv.x), float(gv.y), xsum);
                acc_u[r] += pi_qdot16(uws + r * row_bytes, xt, float(uv.x), float(uv.y), xsum);
            }
        }
        gws += PI_Q4_SIMD_BLOCK >> 1u;
        uws += PI_Q4_SIMD_BLOCK >> 1u;
        gsbr += sb_k_step;
        usbr += sb_k_step;
        xr += PI_Q4_SIMD_BLOCK;
    }
    for (uint r = 0u; r < PI_Q4_SIMD_ROWS; ++r) {
        const uint row = row_base + r;
        if (row >= row_limit) {
            continue;
        }
        const float gsum = simd_sum(acc_g[r]);
        const float usum = simd_sum(acc_u[r]);
        if (lane == 0u) {
            float v = tessl_gelu_pytorch_tanh(gsum) * usum;
            if (mid_as_bf16 != 0u) {
                ((device bfloat *)mid)[row] = bfloat(v);
            } else {
                mid[row] = v;
            }
        }
    }
}

static inline void pi_q4_down_tile_rm(
    uint row0,
    uint row_limit,
    device const uchar *down_packed,
    device const bfloat2 *down_sb,
    device const bfloat *mid_bf16,
    device float *x_out,
    uint n_mid,
    uint group_size,
    uint sgid,
    uint lane)
{
    const uint row_base = row0 + sgid * PI_Q4_SIMD_ROWS;
    if (row_base >= row_limit) {
        return;
    }
    const uint gpr = n_mid / group_size;
    const uint row_bytes = n_mid >> 1u;
    const uint lane_col0 = lane * PI_Q4_SIMD_VPT;
    const uint sb_k_step = PI_Q4_SIMD_BLOCK / group_size;
    float acc[PI_Q4_SIMD_ROWS];
    for (uint r = 0u; r < PI_Q4_SIMD_ROWS; ++r) {
        acc[r] = 0.0f;
    }
    device const uchar *ws = down_packed + row_base * row_bytes + (lane_col0 >> 1u);
    device const bfloat2 *sbr = down_sb + row_base * gpr + (lane_col0 / group_size);
    device const bfloat *xr = mid_bf16 + lane_col0;
    for (uint k0 = 0u; k0 < n_mid; k0 += PI_Q4_SIMD_BLOCK) {
        if (k0 + lane_col0 + PI_Q4_SIMD_VPT <= n_mid) {
            float xt[16];
            const float xsum = pi_load_x16_qdot(xr, xt);
            for (uint r = 0u; r < PI_Q4_SIMD_ROWS; ++r) {
                const uint row = row_base + r;
                if (row >= row_limit) {
                    break;
                }
                const bfloat2 sbv = sbr[r * gpr];
                acc[r] += pi_qdot16(ws + r * row_bytes, xt, float(sbv.x), float(sbv.y), xsum);
            }
        }
        ws += PI_Q4_SIMD_BLOCK >> 1u;
        sbr += sb_k_step;
        xr += PI_Q4_SIMD_BLOCK;
    }
    for (uint r = 0u; r < PI_Q4_SIMD_ROWS; ++r) {
        const uint row = row_base + r;
        if (row >= row_limit) {
            continue;
        }
        const float sum = simd_sum(acc[r]);
        if (lane == 0u) {
            x_out[row] = sum + x_out[row];
        }
    }
}

kernel void persistent_interp_gate_down_q4(
    device const uint *insns [[buffer(0)]],
    constant uint &n_insns [[buffer(1)]],
    device const uchar *gate_packed [[buffer(2)]],
    device const bfloat2 *gate_sb [[buffer(3)]],
    device const bfloat *gate_biases_unused [[buffer(4)]],
    device const uchar *up_packed [[buffer(5)]],
    device const bfloat2 *up_sb [[buffer(6)]],
    device const bfloat *up_biases_unused [[buffer(7)]],
    device const bfloat *x_bf16 [[buffer(8)]],
    device float *mid [[buffer(9)]],
    device const uchar *down_packed [[buffer(10)]],
    device const bfloat2 *down_sb [[buffer(11)]],
    device const bfloat *down_biases_unused [[buffer(12)]],
    device float *x_out [[buffer(13)]],
    device atomic_uint *deps [[buffer(14)]],
    device atomic_uint *fail [[buffer(15)]],
    constant uint &n_mid [[buffer(16)]],
    constant uint &n_out [[buffer(17)]],
    constant uint &cols [[buffer(18)]],
    constant uint &group_size [[buffer(19)]],
    constant uint &n_tg [[buffer(20)]],
    constant uint &max_spin [[buffer(21)]],
    constant uint &mid_as_bf16 [[buffer(22)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)gate_biases_unused;
    (void)up_biases_unused;
    (void)down_biases_unused;
    if (tgid >= n_tg) {
        return;
    }

    const uint rows_per_tile = PI_Q4_SIMD_SG_PER_TG * PI_Q4_SIMD_ROWS;
    threadgroup uint tg_done;
    uint pc = 0u;
    while (pc < n_insns) {
        uint op = insns[pc];
        pc += 1u;

        if (op == OP_HALT) {
            break;
        }

        if (op == OP_PRODUCE_MID) {
            uint chunk = (n_mid + n_tg - 1u) / n_tg;
            uint begin = tgid * chunk;
            uint end = min(begin + chunk, n_mid);
            for (uint tile = begin; tile < end; tile += rows_per_tile) {
                pi_q4_produce_tile_rm(
                    tile,
                    end,
                    gate_packed,
                    gate_sb,
                    up_packed,
                    up_sb,
                    x_bf16,
                    mid,
                    cols,
                    group_size,
                    mid_as_bf16,
                    sgid,
                    lane);
            }
            threadgroup_barrier(mem_flags::mem_device);
            continue;
        }

        if (op == OP_BARRIER) {
            const uint btid = (sgid == 0u && lane == 0u) ? 0u : 1u;
            grid_barrier(deps, fail, n_tg, max_spin, btid, tg_done);
            if (atomic_load_explicit(fail, memory_order_relaxed) != 0u) {
                break;
            }
            continue;
        }

        if (op == OP_DOWN_PROJ) {
            // Hot path always writes bf16 mid (mid_as_bf16=1); cast like shipping.
            device const bfloat *mid_act = (device const bfloat *)mid;
            uint chunk = (n_out + n_tg - 1u) / n_tg;
            uint begin = tgid * chunk;
            uint end = min(begin + chunk, n_out);
            for (uint tile = begin; tile < end; tile += rows_per_tile) {
                pi_q4_down_tile_rm(
                    tile,
                    end,
                    down_packed,
                    down_sb,
                    mid_act,
                    x_out,
                    n_mid,
                    group_size,
                    sgid,
                    lane);
            }
            threadgroup_barrier(mem_flags::mem_device);
            continue;
        }

        break;
    }
}

// --- FA → o_proj sibling (same barrier doctrine, separate entry) -------------

constant uint OP_FA_HALT        = 0u;
constant uint OP_FA_PRODUCE_CTX = 1u;
constant uint OP_FA_BARRIER     = 2u;
constant uint OP_FA_O_PROJ      = 3u;

kernel void persistent_interp_fa_o_proj(
    device const uint *insns [[buffer(0)]],
    constant uint &n_insns [[buffer(1)]],
    device const float *q [[buffer(2)]],
    device const float *k [[buffer(3)]],
    device const float *v [[buffer(4)]],
    device float *ctx [[buffer(5)]],
    device const float *w_o [[buffer(6)]],
    device float *out [[buffer(7)]],
    device atomic_uint *deps [[buffer(8)]],
    device atomic_uint *fail [[buffer(9)]],
    constant uint &n_ctx [[buffer(10)]],
    constant uint &n_out [[buffer(11)]],
    constant uint &n_tg [[buffer(12)]],
    constant uint &max_spin [[buffer(13)]],
    constant float &scale [[buffer(14)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    if (tgid >= n_tg) {
        return;
    }

    threadgroup uint tg_done;
    uint pc = 0u;
    while (pc < n_insns) {
        uint op = insns[pc];
        pc += 1u;

        if (op == OP_FA_HALT) {
            break;
        }

        if (op == OP_FA_PRODUCE_CTX) {
            // Element-local FA mock — TG-partitioned; not real softmax attention.
            uint chunk = (n_ctx + n_tg - 1u) / n_tg;
            uint begin = tgid * chunk;
            uint end = min(begin + chunk, n_ctx);
            for (uint i = begin + tid; i < end; i += tptg) {
                float s = precise::tanh(clamp(q[i] * k[i] * scale, -10.0f, 10.0f));
                ctx[i] = s * v[i];
            }
            threadgroup_barrier(mem_flags::mem_device);
            continue;
        }

        if (op == OP_FA_BARRIER) {
            grid_barrier(deps, fail, n_tg, max_spin, tid, tg_done);
            if (atomic_load_explicit(fail, memory_order_relaxed) != 0u) {
                break;
            }
            continue;
        }

        if (op == OP_FA_O_PROJ) {
            // Needs *all* of ctx — the grid-wide dep the barrier established.
            uint chunk = (n_out + n_tg - 1u) / n_tg;
            uint begin = tgid * chunk;
            uint end = min(begin + chunk, n_out);
            for (uint r = begin + tid; r < end; r += tptg) {
                float acc = 0.0f;
                uint base = r * n_ctx;
                for (uint i = 0u; i < n_ctx; i++) {
                    acc += ctx[i] * w_o[base + i];
                }
                out[r] = acc;
            }
            threadgroup_barrier(mem_flags::mem_device);
            continue;
        }

        // Unknown opcode — treat as halt.
        break;
    }
}
