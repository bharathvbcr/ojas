// 2-D max pooling, `Backend::max_pool2d_forward` and `max_pool2d_backward`.
//
// tessl has no max-pool kernel (its pooling kernels are mean/segment pools
// for embeddings), so these are ojas's own. They reproduce
// `ojas-cpu/src/spatial.rs` bit for bit:
//
//   * A window's winner is the FIRST input in (kh, kw) order that holds its
//     maximum: `v > best` replaces the winner, so a tie keeps the earlier
//     input (torch's rule, and the CPU's `window_argmax`).
//   * The forward writes the winner's value, so -0.0 vs 0.0 and every other
//     bit pattern is the winner's own.
//   * The backward is a gather, one thread per input element: it visits the
//     windows that cover the input in ascending (oy, ox), finds each
//     window's winner again from `x`, and adds the windows' `dy` for which
//     it is the winner, in that order. No atomics, no two threads write one
//     value, so the sum order is fixed by the loop and equals the CPU's.
//
// Compiled with -fmetal-math-mode=safe and -ffp-contract=off (build.rs), so
// the only float operations are the comparisons and `acc += g`.
//
// The finite scans of the operands run as separate `ojas_check_finite`
// passes on the host side of the op (the CPU refuses a non-finite value
// wherever it sits, including one no window reads). The backward flags its
// own output into the status slot, the word `ST_OUT` of
// `ojas-metal/src/device.rs`.
#include <metal_stdlib>
using namespace metal;

#define POOL_ST_OUT 1u

inline void ojas_pool_flag(device atomic_uint *st, uint word)
{
    atomic_store_explicit(&st[word], 1u, memory_order_relaxed);
}

/// The offset in `plane` (a `[h, w]` image) of the first input of window
/// `(oy, ox)`, in (kh, kw) order, that holds the window's maximum.
inline uint ojas_pool_argmax(device const float *plane, uint w, uint kh, uint kw,
                             uint sh, uint sw, uint oy, uint ox)
{
    uint best = (oy * sh) * w + ox * sw;
    float best_v = plane[best];
    for (uint i = 0u; i < kh; ++i) {
        const uint row = (oy * sh + i) * w + ox * sw;
        for (uint j = 0u; j < kw; ++j) {
            const float v = plane[row + j];
            if (v > best_v) {
                best = row + j;
                best_v = v;
            }
        }
    }
    return best;
}

/// y[plane][oy][ox] = the maximum of the window at (oy, ox). One thread per
/// output element; `total` = planes * oh * ow. The windows lie inside the
/// image (the host checks `(oh - 1) * sh + kh <= h`, and the same on w).
kernel void ojas_max_pool2d_fwd(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &total [[buffer(2)]],
    constant uint &h [[buffer(3)]],
    constant uint &w [[buffer(4)]],
    constant uint &oh [[buffer(5)]],
    constant uint &ow [[buffer(6)]],
    constant uint &kh [[buffer(7)]],
    constant uint &kw [[buffer(8)]],
    constant uint &sh [[buffer(9)]],
    constant uint &sw [[buffer(10)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint ox = i % ow;
    const uint oy = (i / ow) % oh;
    const uint p = i / (ow * oh);
    device const float *plane = x + (ulong)p * h * w;
    y[i] = plane[ojas_pool_argmax(plane, w, kh, kw, sh, sw, oy, ox)];
}

/// The first window index on one axis whose span reaches input index `i`:
/// `ceil(max(i + 1 - k, 0) / s)`.
inline uint ojas_pool_first_window(uint i, uint k, uint s)
{
    const uint lo = (i + 1u > k) ? (i + 1u - k) : 0u;
    return (lo + s - 1u) / s;
}

inline float ojas_pool_bf16(float x)
{
    return float(bfloat(x));
}

/// dx[plane][iy][ix] = the sum, over the windows that cover (iy, ix) in
/// ascending (oy, ox) and whose winner it is, of dy at that window. One
/// thread per input element; `total` = planes * h * w. An input no window
/// covers (the floor-mode remainder) gets +0.0, as the CPU's empty sum does.
/// A non-finite sum flags `ST_OUT`.
kernel void ojas_max_pool2d_bwd(
    device const float *x [[buffer(0)]],
    device const float *dy [[buffer(1)]],
    device float *dx [[buffer(2)]],
    constant uint &total [[buffer(3)]],
    constant uint &h [[buffer(4)]],
    constant uint &w [[buffer(5)]],
    constant uint &oh [[buffer(6)]],
    constant uint &ow [[buffer(7)]],
    constant uint &kh [[buffer(8)]],
    constant uint &kw [[buffer(9)]],
    constant uint &sh [[buffer(10)]],
    constant uint &sw [[buffer(11)]],
    device atomic_uint *st [[buffer(12)]],
    constant uint &bf16_acc [[buffer(13)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint ix = i % w;
    const uint iy = (i / w) % h;
    const uint p = i / (w * h);
    device const float *plane = x + (ulong)p * h * w;
    device const float *grads = dy + (ulong)p * oh * ow;
    const uint at = iy * w + ix;
    const uint y_lo = ojas_pool_first_window(iy, kh, sh);
    const uint y_hi = min(iy / sh + 1u, oh);
    const uint x_lo = ojas_pool_first_window(ix, kw, sw);
    const uint x_hi = min(ix / sw + 1u, ow);
    float acc = 0.0f;
    for (uint oy = y_lo; oy < y_hi; ++oy) {
        for (uint ox = x_lo; ox < x_hi; ++ox) {
            if (ojas_pool_argmax(plane, w, kh, kw, sh, sw, oy, ox) == at) {
                const float g = grads[oy * ow + ox];
                acc = (bf16_acc != 0u) ? ojas_pool_bf16(acc + ojas_pool_bf16(g)) : (acc + g);
            }
        }
    }
    dx[i] = acc;
    if (!isfinite(acc)) ojas_pool_flag(st, POOL_ST_OUT);
}
