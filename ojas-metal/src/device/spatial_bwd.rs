//! The convolution backward passes on the device thread
//! (`Backend::conv2d_backward`, `Backend::conv_transpose2d_backward`).
//!
//! A one-group convolution is the forward's GEMM turned around: `ojas_im2col`
//! forms `cols [pixels, c kh kw]`, `gy` is permuted from NCHW to `gy' [pixels,
//! o]`, and the two GEMMs are `linear_backward`'s with `cols` as the input:
//! `dw = gy'^T @ cols` and `dcols = gy' @ w`. `ojas_col2im` gathers `dcols`
//! back onto the image (each input value sums the taps that read it in
//! ascending `(kh, kw)`, the CPU's order), and `ojas_chan_sum` sums `gy` per
//! channel in the CPU's order for the bias. A depthwise convolution is three
//! direct gathers and no GEMM. A transposed convolution with kernel == stride
//! reads `gy` through `ojas_im2col` on the input's grid.
//!
//! Every GEMM is the exact-f32 lane: the CPU backward these are held to is
//! f32, and the lane choice of the forward (bf16 where both operands are
//! bf16-exact) is not carried over. The operands are scanned here
//! (`ST_IN`) and the outputs after (`ST_OUT`), all under `op`.

use tessl::dispatch::set_u32;

use super::{bind, u32_of, Worker, ST_IN, ST_OUT};
use crate::link::{metal_err, Arg, ConvGeom, Lane, Reply, Res};

/// The kernels of `kernels/spatial_bwd.metal`, preloaded with the rest.
pub(super) const KERNELS: &[&str] = &[
    "ojas_col2im",
    "ojas_dwconv_bwd_x",
    "ojas_dwconv_bwd_w",
    "ojas_chan_sum",
];

/// The product of `dims` as an element count, or the op's overflow error.
fn big(op: &'static str, dims: &[u32]) -> Res<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
        .ok_or_else(|| metal_err(format!("{op}: convolution extent overflows")))
}

/// `a * b` as an element count, or the op's overflow error.
fn mul(op: &'static str, a: usize, b: usize) -> Res<usize> {
    a.checked_mul(b)
        .ok_or_else(|| metal_err(format!("{op}: convolution extent overflows")))
}

/// The kernels' geometry words after the leading `total`, in the order
/// `ojas_im2col` and `ojas_col2im` take them: channels, input extents, the
/// kernel, stride, padding, the output extents.
fn window_words(g: &ConvGeom) -> [u32; 11] {
    [
        g.c,
        g.h,
        g.w,
        g.kernel[0],
        g.kernel[1],
        g.stride[0],
        g.stride[1],
        g.padding[0],
        g.padding[1],
        g.oh,
        g.ow,
    ]
}

impl Worker {
    /// `conv2d_backward` with one group, as the module comment lays out.
    /// Replies with `dx`, `dw` and, when `bias` is given, `db`.
    pub(super) fn conv2d_bwd(
        &mut self,
        op: &'static str,
        [x, w, gy]: [Arg; 3],
        bias: Option<Arg>,
        g: ConvGeom,
    ) -> Res<Reply> {
        let xv = self.view(x)?;
        let wv = self.view(w)?;
        let gv = self.view(gy)?;
        let bv = bias.map(|a| self.view(a)).transpose()?;
        let [kh, kw] = g.kernel;
        let pixels = big(op, &[g.oh, g.ow])?;
        let rows = big(op, &[g.batch, g.oh, g.ow])?;
        let width = big(op, &[g.c, kh, kw])?;
        let in_n = big(op, &[g.batch, g.c, g.h, g.w])?;
        let out_n = big(op, &[g.batch, g.o, g.oh, g.ow])?;
        let out_c = g.o as usize;
        let cols_n = mul(op, rows, width)?;
        let w_n = mul(op, out_c, width)?;
        if xv.n != in_n || wv.n != w_n || gv.n != out_n || bv.as_ref().is_some_and(|b| b.n != out_c)
        {
            return Err(metal_err(format!(
                "{op}: buffers do not hold a [{}, {}, {}, {}] input, a [{}, {}, {kh}, {kw}] weight, \
                 a [{}, {}, {}, {}] output gradient and the bias asked for",
                g.batch, g.c, g.h, g.w, g.o, g.c, g.batch, g.o, g.oh, g.ow
            )));
        }
        let (cols_u, in_u, pixels_u) = (u32_of(cols_n)?, u32_of(in_n)?, u32_of(pixels)?);
        u32_of(out_n)?;
        let cols = self.fresh(cols_n)?;
        let gyp = self.fresh(out_n)?;
        let dcols = self.fresh(cols_n)?;
        let gx = self.fresh(in_n)?;
        let gw = self.fresh(w_n)?;
        let gb = bv.as_ref().map(|_| self.fresh(out_c)).transpose()?;
        let st = self.status(op)?;
        self.check(&st, &xv, ST_IN)?;
        self.check_operand(&st, w, &wv, ST_IN)?;
        self.check(&st, &gv, ST_IN)?;
        if let (Some(a), Some(b)) = (bias, bv.as_ref()) {
            self.check_operand(&st, a, b, ST_IN)?;
        }
        let words = window_words(&g);
        self.k1("ojas_im2col", cols.n, |b| {
            bind(b, &xv, 0);
            bind(b, &cols, 1);
            set_u32(b, cols_u, 2);
            for (i, v) in words.into_iter().enumerate() {
                set_u32(b, v, 3 + i);
            }
        })?;
        // gy [n, o, pixels] to gy' [n * pixels, o].
        self.permute_into(
            &gv,
            &gyp,
            3,
            &[g.batch, pixels_u, g.o],
            &[g.o * pixels_u, 1, pixels_u],
            &st,
        )?;
        let gemm = self.gemm_lane(Lane::ExactF32);
        let gy_t = self.tt(&gyp, &[rows, out_c])?;
        gemm.nn(
            &gy_t,
            &self.mat(&wv, out_c, width)?,
            &self.tt(&dcols, &[rows, width])?,
        )
        .map_err(metal_err)?;
        gemm.tn(
            &gy_t,
            &self.tt(&cols, &[rows, width])?,
            &self.tt(&gw, &[out_c, width])?,
        )
        .map_err(metal_err)?;
        self.k1("ojas_col2im", in_n, |b| {
            bind(b, &dcols, 0);
            bind(b, &gx, 1);
            set_u32(b, in_u, 2);
            for (i, v) in words.into_iter().enumerate() {
                set_u32(b, v, 3 + i);
            }
        })?;
        self.bias_grad(&gv, gb.as_ref(), g.batch, g.o, pixels_u)?;
        self.keep_grads(&st, gx, gw, gb)
    }

    /// `conv2d_backward` with `groups == c == o`: `ojas_dwconv_bwd_x`,
    /// `ojas_dwconv_bwd_w` and the bias sum, no scratch.
    pub(super) fn depthwise_conv2d_bwd(
        &mut self,
        op: &'static str,
        [x, w, gy]: [Arg; 3],
        bias: Option<Arg>,
        g: ConvGeom,
    ) -> Res<Reply> {
        let xv = self.view(x)?;
        let wv = self.view(w)?;
        let gv = self.view(gy)?;
        let bv = bias.map(|a| self.view(a)).transpose()?;
        let [kh, kw] = g.kernel;
        let pixels = big(op, &[g.oh, g.ow])?;
        let in_n = big(op, &[g.batch, g.c, g.h, g.w])?;
        let out_n = big(op, &[g.batch, g.o, g.oh, g.ow])?;
        let w_n = big(op, &[g.c, kh, kw])?;
        if g.c != g.o
            || xv.n != in_n
            || wv.n != w_n
            || gv.n != out_n
            || bv.as_ref().is_some_and(|b| b.n != g.o as usize)
        {
            return Err(metal_err(format!(
                "{op}: buffers do not hold a [{}, {}, {}, {}] input, a [{}, 1, {kh}, {kw}] weight, \
                 a [{}, {}, {}, {}] output gradient and the bias asked for",
                g.batch, g.c, g.h, g.w, g.c, g.batch, g.o, g.oh, g.ow
            )));
        }
        let (in_u, w_u, pixels_u) = (u32_of(in_n)?, u32_of(w_n)?, u32_of(pixels)?);
        u32_of(out_n)?;
        let gx = self.fresh(in_n)?;
        let gw = self.fresh(w_n)?;
        let gb = bv.as_ref().map(|_| self.fresh(g.o as usize)).transpose()?;
        let st = self.status(op)?;
        self.check(&st, &xv, ST_IN)?;
        self.check_operand(&st, w, &wv, ST_IN)?;
        self.check(&st, &gv, ST_IN)?;
        if let (Some(a), Some(b)) = (bias, bv.as_ref()) {
            self.check_operand(&st, a, b, ST_IN)?;
        }
        let words = window_words(&g);
        self.k1("ojas_dwconv_bwd_x", in_n, |b| {
            bind(b, &gv, 0);
            bind(b, &wv, 1);
            bind(b, &gx, 2);
            set_u32(b, in_u, 3);
            for (i, v) in words.into_iter().enumerate() {
                set_u32(b, v, 4 + i);
            }
        })?;
        self.k1("ojas_dwconv_bwd_w", w_n, |b| {
            bind(b, &xv, 0);
            bind(b, &gv, 1);
            bind(b, &gw, 2);
            set_u32(b, w_u, 3);
            set_u32(b, g.batch, 4);
            for (i, v) in words.into_iter().enumerate() {
                set_u32(b, v, 5 + i);
            }
        })?;
        self.bias_grad(&gv, gb.as_ref(), g.batch, g.o, pixels_u)?;
        self.keep_grads(&st, gx, gw, gb)
    }

    /// `conv_transpose2d_backward` with kernel == stride and no padding. `g`
    /// is the forward's geometry: `c` input channels over `h x w`, `o` output
    /// channels, the kernel, and `oh = h kh`, `ow = w kw`. Output pixel `(h kh
    /// + i, w kw + j)` of channel `o` has exactly one source, the input pixel
    /// `(h, w)`, so `gy` read as the columns of a convolution with this
    /// stride, `ojas_im2col` on `[o, oh, ow]` giving `[n h w, o kh kw]`, is
    /// the column gradient: `dw = x'^T @ dcols` (`x'` the NHWC input) and
    /// `dx' = dcols @ w^T`, which a permute returns to NCHW.
    pub(super) fn conv_transpose2d_bwd(
        &mut self,
        op: &'static str,
        [x, w, gy]: [Arg; 3],
        bias: Option<Arg>,
        g: ConvGeom,
    ) -> Res<Reply> {
        let xv = self.view(x)?;
        let wv = self.view(w)?;
        let gv = self.view(gy)?;
        let bv = bias.map(|a| self.view(a)).transpose()?;
        let [kh, kw] = g.kernel;
        let rows = big(op, &[g.batch, g.h, g.w])?;
        let okk = big(op, &[g.o, kh, kw])?;
        let in_n = big(op, &[g.batch, g.c, g.h, g.w])?;
        let out_n = big(op, &[g.batch, g.o, g.oh, g.ow])?;
        let c = g.c as usize;
        let cols_n = mul(op, rows, okk)?;
        let w_n = mul(op, c, okk)?;
        let pixels = big(op, &[g.oh, g.ow])?;
        if g.oh as usize != g.h as usize * kh as usize
            || g.ow as usize != g.w as usize * kw as usize
            || g.stride != g.kernel
            || g.padding != [0, 0]
            || xv.n != in_n
            || wv.n != w_n
            || gv.n != out_n
            || bv.as_ref().is_some_and(|b| b.n != g.o as usize)
        {
            return Err(metal_err(format!(
                "{op}: buffers do not hold a [{}, {}, {}, {}] input, a [{}, {}, {kh}, {kw}] weight, \
                 a [{}, {}, {}, {}] output gradient and the bias asked for",
                g.batch, g.c, g.h, g.w, g.c, g.o, g.batch, g.o, g.oh, g.ow
            )));
        }
        let (cols_u, pixels_u) = (u32_of(cols_n)?, u32_of(pixels)?);
        u32_of(in_n)?;
        u32_of(out_n)?;
        let xp = self.fresh(in_n)?;
        let dcols = self.fresh(cols_n)?;
        let dxp = self.fresh(in_n)?;
        let gx = self.fresh(in_n)?;
        let gw = self.fresh(w_n)?;
        let gb = bv.as_ref().map(|_| self.fresh(g.o as usize)).transpose()?;
        let st = self.status(op)?;
        self.check(&st, &xv, ST_IN)?;
        self.check_operand(&st, w, &wv, ST_IN)?;
        self.check(&st, &gv, ST_IN)?;
        if let (Some(a), Some(b)) = (bias, bv.as_ref()) {
            self.check_operand(&st, a, b, ST_IN)?;
        }
        // gy as the input of a convolution of kernel == stride whose output
        // grid is this layer's input.
        let words = [
            g.o,
            g.oh,
            g.ow,
            kh,
            kw,
            g.stride[0],
            g.stride[1],
            0,
            0,
            g.h,
            g.w,
        ];
        self.k1("ojas_im2col", dcols.n, |b| {
            bind(b, &gv, 0);
            bind(b, &dcols, 1);
            set_u32(b, cols_u, 2);
            for (i, v) in words.into_iter().enumerate() {
                set_u32(b, v, 3 + i);
            }
        })?;
        // NCHW -> [n, h, w, c], as the forward permutes its input.
        let (ch, h, wd) = (g.c, g.h, g.w);
        self.permute_into(
            &xv,
            &xp,
            4,
            &[g.batch, h, wd, ch],
            &[ch * h * wd, wd, 1, h * wd],
            &st,
        )?;
        let gemm = self.gemm_lane(Lane::ExactF32);
        let cols_t = self.tt(&dcols, &[rows, okk])?;
        gemm.nt(
            &cols_t,
            &self.mat(&wv, c, okk)?,
            &self.tt(&dxp, &[rows, c])?,
        )
        .map_err(metal_err)?;
        gemm.tn(
            &self.tt(&xp, &[rows, c])?,
            &cols_t,
            &self.tt(&gw, &[c, okk])?,
        )
        .map_err(metal_err)?;
        // [n, h, w, c] back to NCHW.
        self.permute_into(
            &dxp,
            &gx,
            4,
            &[g.batch, ch, h, wd],
            &[h * wd * ch, 1, wd * ch, ch],
            &st,
        )?;
        self.bias_grad(&gv, gb.as_ref(), g.batch, g.o, pixels_u)?;
        self.keep_grads(&st, gx, gw, gb)
    }

    /// `db` as `ojas_chan_sum` of the NCHW `gy` when a bias gradient was
    /// asked for.
    fn bias_grad(
        &self,
        gy: &super::V,
        db: Option<&super::V>,
        batch: u32,
        channels: u32,
        pixels: u32,
    ) -> Res<()> {
        let Some(db) = db else {
            return Ok(());
        };
        self.k1("ojas_chan_sum", channels as usize, |b| {
            bind(b, gy, 0);
            bind(b, db, 1);
            set_u32(b, channels, 2);
            set_u32(b, batch, 3);
            set_u32(b, pixels, 4);
        })
    }

    /// Scan the gradients and hand them out: `dx`, `dw`, then `db` if there
    /// is one.
    fn keep_grads(
        &mut self,
        st: &super::St,
        gx: super::V,
        gw: super::V,
        gb: Option<super::V>,
    ) -> Res<Reply> {
        self.check(st, &gx, ST_OUT)?;
        self.check(st, &gw, ST_OUT)?;
        let mut outs = vec![gx, gw];
        if let Some(gb) = gb {
            self.check(st, &gb, ST_OUT)?;
            outs.push(gb);
        }
        Ok(self.keep(outs))
    }
}
