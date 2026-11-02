// Kernels behind the convolution backward passes of `MetalBackend`
// (`Backend::conv2d_backward`, `Backend::conv_transpose2d_backward`).
//
// tessl has no col2im, no depthwise convolution backward and no per-channel
// sum (rg over tessl/src and tessl/kernels: the only `conv` kernels are the
// causal conv1d of the Qwen3.5 path), so these are ojas's own. The GEMMs of
// the dense routes are tessl's; these kernels are the gathers around them.
//
// Every kernel is a gather: one thread writes each output value, no atomics,
// and every sum runs in a fixed order, the CPU's, so a result repeats bit for
// bit and the kernels that follow the CPU's order exactly (depthwise,
// col2im, the bias gradient) give the CPU's bits when their inputs are the
// CPU's. Compiled with -ffp-contract=off (build.rs), so `a * b + c` is a
// rounded product and a rounded sum, as on the CPU.
//
// Index arithmetic is `uint` for extents the host has bounded by u32::MAX and
// `ulong` for products that can pass it, as in ojas_backend.metal.
#include <metal_stdlib>
using namespace metal;

inline float ojas_bf16(float v)
{
    const uint b = as_type<uint>(v);
    if ((b & 0x7fffffffu) > 0x7f800000u) {
        return as_type<float>((b | 0x00400000u) & 0xffff0000u);
    }
    return as_type<float>((b + (0x7fffu + ((b >> 16) & 1u))) & 0xffff0000u);
}

/// The adjoint of `ojas_im2col` as a gather: dx[n][ci][iy][ix] is the sum of
/// the column-gradient entries of every tap that read the input value, in
/// ascending (ky, kx), zero where no tap reads it. `dcols` is laid out as
/// `ojas_im2col` lays `cols` out: row (n, oy, ox), column (ci, ky, kx). A tap
/// (ky, kx) of output (oy, ox) read input (oy * sh + ky - ph, ox * sw + kx -
/// pw), so input (iy, ix) is read by tap ky at oy = (iy + ph - ky) / sh when
/// that division is exact and in range. One thread per element of dx.
kernel void ojas_col2im(
    device const float *dcols [[buffer(0)]],
    device float *dx [[buffer(1)]],
    constant uint &total [[buffer(2)]],
    constant uint &c [[buffer(3)]],
    constant uint &h [[buffer(4)]],
    constant uint &w [[buffer(5)]],
    constant uint &kh [[buffer(6)]],
    constant uint &kw [[buffer(7)]],
    constant uint &sh [[buffer(8)]],
    constant uint &sw [[buffer(9)]],
    constant uint &ph [[buffer(10)]],
    constant uint &pw [[buffer(11)]],
    constant uint &oh [[buffer(12)]],
    constant uint &ow [[buffer(13)]],
    constant uint &bf16_accum [[buffer(14)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint ix = i % w;
    const uint iy = (i / w) % h;
    const uint ci = (i / (w * h)) % c;
    const uint n = i / (w * h * c);
    const ulong width = (ulong)c * kh * kw;
    const bool bf16_acc = bf16_accum != 0u;
    float acc = 0.0f;
    for (uint ky = 0u; ky < kh; ++ky) {
        const int ty = (int)(iy + ph) - (int)ky;
        if (ty < 0 || (uint)ty % sh != 0u) continue;
        const uint oy = (uint)ty / sh;
        if (oy >= oh) continue;
        for (uint kx = 0u; kx < kw; ++kx) {
            const int tx = (int)(ix + pw) - (int)kx;
            if (tx < 0 || (uint)tx % sw != 0u) continue;
            const uint ox = (uint)tx / sw;
            if (ox >= ow) continue;
            const ulong row = ((ulong)n * oh + oy) * ow + ox;
            const float v = dcols[row * width + ((ulong)ci * kh + ky) * kw + kx];
            acc = bf16_acc ? ojas_bf16(acc + ojas_bf16(v)) : (acc + v);
        }
    }
    dx[i] = acc;
}

/// The input gradient of a depthwise convolution (`groups == c == o`): dx[n][ch]
/// [iy][ix] is the sum over the taps (ky, kx) that read it, in ascending order,
/// of w[ch][ky][kx] * dy at the output the tap belongs to. The product is
/// rounded before it is added, as the CPU's single-term column gradient is.
/// One thread per element of dx.
kernel void ojas_dwconv_bwd_x(
    device const float *dy [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device float *dx [[buffer(2)]],
    constant uint &total [[buffer(3)]],
    constant uint &c [[buffer(4)]],
    constant uint &h [[buffer(5)]],
    constant uint &wd [[buffer(6)]],
    constant uint &kh [[buffer(7)]],
    constant uint &kw [[buffer(8)]],
    constant uint &sh [[buffer(9)]],
    constant uint &sw [[buffer(10)]],
    constant uint &ph [[buffer(11)]],
    constant uint &pw [[buffer(12)]],
    constant uint &oh [[buffer(13)]],
    constant uint &ow [[buffer(14)]],
    constant uint &bf16_accum [[buffer(15)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint ix = i % wd;
    const uint iy = (i / wd) % h;
    const uint ch = (i / (wd * h)) % c;
    const uint n = i / (wd * h * c);
    const ulong plane = ((ulong)n * c + ch) * oh * ow;
    const bool bf16_acc = bf16_accum != 0u;
    float acc = 0.0f;
    for (uint ky = 0u; ky < kh; ++ky) {
        const int ty = (int)(iy + ph) - (int)ky;
        if (ty < 0 || (uint)ty % sh != 0u) continue;
        const uint oy = (uint)ty / sh;
        if (oy >= oh) continue;
        for (uint kx = 0u; kx < kw; ++kx) {
            const int tx = (int)(ix + pw) - (int)kx;
            if (tx < 0 || (uint)tx % sw != 0u) continue;
            const uint ox = (uint)tx / sw;
            if (ox >= ow) continue;
            const float wv = w[((ulong)ch * kh + ky) * kw + kx];
            const float dyv = dy[plane + (ulong)oy * ow + ox];
            if (bf16_acc) {
                acc = ojas_bf16(acc + ojas_bf16(ojas_bf16(wv) * ojas_bf16(dyv)));
            } else {
                const float product = wv * dyv;
                acc += product;
            }
        }
    }
    dx[i] = acc;
}

/// The weight gradient of a depthwise convolution: dw[ch][ky][kx] is the sum
/// over the images, then the output pixels (oy, ox) in ascending order, of
/// x[n][ch][oy * sh + ky - ph][ox * sw + kx - pw] * dy[n][ch][oy][ox], a tap on
/// padding adding nothing. One thread per element of dw, so a weight's sum is
/// one sequential walk in the CPU's order.
kernel void ojas_dwconv_bwd_w(
    device const float *x [[buffer(0)]],
    device const float *dy [[buffer(1)]],
    device float *dw [[buffer(2)]],
    constant uint &total [[buffer(3)]],
    constant uint &batch [[buffer(4)]],
    constant uint &c [[buffer(5)]],
    constant uint &h [[buffer(6)]],
    constant uint &wd [[buffer(7)]],
    constant uint &kh [[buffer(8)]],
    constant uint &kw [[buffer(9)]],
    constant uint &sh [[buffer(10)]],
    constant uint &sw [[buffer(11)]],
    constant uint &ph [[buffer(12)]],
    constant uint &pw [[buffer(13)]],
    constant uint &oh [[buffer(14)]],
    constant uint &ow [[buffer(15)]],
    constant uint &bf16_accum [[buffer(16)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    const uint kx = i % kw;
    const uint ky = (i / kw) % kh;
    const uint ch = i / (kw * kh);
    const bool bf16_acc = bf16_accum != 0u;
    float acc = 0.0f;
    for (uint n = 0u; n < batch; ++n) {
        const ulong xplane = ((ulong)n * c + ch) * h * wd;
        const ulong yplane = ((ulong)n * c + ch) * oh * ow;
        for (uint oy = 0u; oy < oh; ++oy) {
            const int iy = (int)(oy * sh + ky) - (int)ph;
            if (iy < 0 || iy >= (int)h) continue;
            for (uint ox = 0u; ox < ow; ++ox) {
                const int ix = (int)(ox * sw + kx) - (int)pw;
                if (ix < 0 || ix >= (int)wd) continue;
                const float xv = x[xplane + (ulong)iy * wd + (uint)ix];
                const float dyv = dy[yplane + (ulong)oy * ow + ox];
                if (bf16_acc) {
                    acc = ojas_bf16(acc + ojas_bf16(ojas_bf16(xv) * ojas_bf16(dyv)));
                } else {
                    const float product = xv * dyv;
                    acc += product;
                }
            }
        }
    }
    dw[i] = acc;
}

/// The bias gradient of an NCHW output gradient dy `[batch, channels,
/// pixels]`: db[ch] is the sum of dy[n][ch][..] over the images and then the
/// pixels, both ascending, in f32. One thread per channel, so the sum is the
/// CPU's `channel_sums` bit for bit.
kernel void ojas_chan_sum(
    device const float *dy [[buffer(0)]],
    device float *db [[buffer(1)]],
    constant uint &channels [[buffer(2)]],
    constant uint &batch [[buffer(3)]],
    constant uint &pixels [[buffer(4)]],
    constant uint &bf16_accum [[buffer(5)]],
    uint ch [[thread_position_in_grid]])
{
    if (ch >= channels) return;
    const bool bf16_acc = bf16_accum != 0u;
    float acc = 0.0f;
    for (uint n = 0u; n < batch; ++n) {
        const ulong at = ((ulong)n * channels + ch) * pixels;
        for (uint p = 0u; p < pixels; ++p) {
            const float v = dy[at + p];
            acc = bf16_acc ? ojas_bf16(acc + ojas_bf16(v)) : (acc + v);
        }
    }
    db[ch] = acc;
}
