//! Device side of the encoder backward ops: the commands of
//! `backend/enc_bwd_host.rs` encoded as dispatches of
//! `kernels/encoder_bwd.metal`. Each op takes one status slot; a kernel flags
//! the input it reads and the output it writes, and the fault is reported at
//! the next sync point like every other op's.

use tessl::dispatch::{set_f32, set_u32};

use super::{bind, bind_st, u32_of, Worker, ST_IN};
use crate::encoder_bwd::{row_threads, BcastGradGeom, ClnBwdGeom, LnBwdGeom};
use crate::link::{metal_err, Arg, Reply, Res};

impl Worker {
    /// LayerNorm backward: `ojas_ln_bwd_rows` (dx, and each row's statistics
    /// for the column pass), `ojas_ln_bwd_cols_part` (per-split column sums)
    /// and `ojas_cols_fin4` (dw, db). `b` is read only to be checked.
    pub(super) fn ln_bwd(&mut self, args: [Arg; 4], g: LnBwdGeom, eps: f32) -> Res<Reply> {
        const OP: &str = "layer_norm_backward";
        let [x, w, b, gy] = args;
        let (xv, wv, bv, gv) = (self.view(x)?, self.view(w)?, self.view(b)?, self.view(gy)?);
        let (rows, dim) = (g.rows as usize, g.dim as usize);
        let n = rows
            .checked_mul(dim)
            .ok_or_else(|| metal_err(format!("{OP}: extent overflows")))?;
        if rows == 0
            || dim == 0
            || dim >= 1 << 24
            || g.splits == 0
            || g.chunk == 0
            || u64::from(g.splits) * u64::from(g.chunk) < u64::from(g.rows)
            || xv.n != n
            || gv.n != n
            || wv.n != dim
            || bv.n != dim
        {
            return Err(metal_err(format!(
                "{OP}: buffers do not hold {rows} rows of {dim}"
            )));
        }
        let gx = self.fresh(n)?;
        let gw = self.fresh(dim)?;
        let gb = self.fresh(dim)?;
        let stats = self.fresh(rows * 4)?;
        let part = self.fresh(g.splits as usize * dim * 4)?;
        let st = self.status(OP)?;
        self.check_operand(&st, b, &bv, ST_IN)?;
        let threads = row_threads(g.dim) as usize;
        self.ktg("ojas_ln_bwd_rows", rows, 1, threads, |bd| {
            bind(bd, &xv, 0);
            bind(bd, &wv, 1);
            bind(bd, &gv, 2);
            bind(bd, &gx, 3);
            bind(bd, &stats, 4);
            bind_st(bd, &st, 5);
            set_u32(bd, g.rows, 6);
            set_u32(bd, g.dim, 7);
            set_f32(bd, eps, 8);
        })?;
        self.k2("ojas_ln_bwd_cols_part", dim, g.splits as usize, |bd| {
            bind(bd, &xv, 0);
            bind(bd, &gv, 1);
            bind(bd, &stats, 2);
            bind(bd, &part, 3);
            set_u32(bd, g.rows, 4);
            set_u32(bd, g.dim, 5);
            set_u32(bd, g.chunk, 6);
        })?;
        self.k1("ojas_cols_fin4", dim, |bd| {
            bind(bd, &part, 0);
            bind(bd, &gw, 1);
            bind(bd, &gb, 2);
            bind_st(bd, &st, 3);
            set_u32(bd, g.dim, 4);
            set_u32(bd, g.splits, 5);
        })?;
        Ok(self.keep(vec![gx, gw, gb]))
    }

    /// Channel LayerNorm backward over `[batch, channels, spatial]`:
    /// `ojas_cln_bwd_pos` (one thread per position), `ojas_cln_bwd_ch_part`
    /// (a threadgroup per channel and split) and `ojas_cols_fin4`.
    pub(super) fn cln_bwd(&mut self, args: [Arg; 4], g: ClnBwdGeom, eps: f32) -> Res<Reply> {
        const OP: &str = "channel_layer_norm_backward";
        let [x, w, b, gy] = args;
        let (xv, wv, bv, gv) = (self.view(x)?, self.view(w)?, self.view(b)?, self.view(gy)?);
        let (positions, channels, spatial) = (
            g.positions as usize,
            g.channels as usize,
            g.spatial as usize,
        );
        let n = positions
            .checked_mul(channels)
            .ok_or_else(|| metal_err(format!("{OP}: extent overflows")))?;
        if positions == 0
            || channels == 0
            || channels >= 1 << 24
            || spatial == 0
            || !positions.is_multiple_of(spatial)
            || g.splits == 0
            || g.chunk == 0
            || u64::from(g.splits) * u64::from(g.chunk) < u64::from(g.positions)
            || xv.n != n
            || gv.n != n
            || wv.n != channels
            || bv.n != channels
        {
            return Err(metal_err(format!(
                "{OP}: buffers do not hold {positions} positions of {channels} channels"
            )));
        }
        let gx = self.fresh(n)?;
        let gw = self.fresh(channels)?;
        let gb = self.fresh(channels)?;
        let stats = self.fresh(positions * 4)?;
        let part = self.fresh(g.splits as usize * channels * 4)?;
        let st = self.status(OP)?;
        self.check_operand(&st, b, &bv, ST_IN)?;
        self.k1("ojas_cln_bwd_pos", positions, |bd| {
            bind(bd, &xv, 0);
            bind(bd, &wv, 1);
            bind(bd, &gv, 2);
            bind(bd, &gx, 3);
            bind(bd, &stats, 4);
            bind_st(bd, &st, 5);
            set_u32(bd, g.positions, 6);
            set_u32(bd, g.channels, 7);
            set_u32(bd, g.spatial, 8);
            set_f32(bd, eps, 9);
        })?;
        self.ktg(
            "ojas_cln_bwd_ch_part",
            channels,
            g.splits as usize,
            crate::encoder_bwd::MAX_ROW_THREADS as usize,
            |bd| {
                bind(bd, &xv, 0);
                bind(bd, &gv, 1);
                bind(bd, &stats, 2);
                bind(bd, &part, 3);
                set_u32(bd, g.positions, 4);
                set_u32(bd, g.channels, 5);
                set_u32(bd, g.spatial, 6);
                set_u32(bd, g.chunk, 7);
            },
        )?;
        self.k1("ojas_cols_fin4", channels, |bd| {
            bind(bd, &part, 0);
            bind(bd, &gw, 1);
            bind(bd, &gb, 2);
            bind_st(bd, &st, 3);
            set_u32(bd, g.channels, 4);
            set_u32(bd, g.splits, 5);
        })?;
        Ok(self.keep(vec![gx, gw, gb]))
    }

    /// `gy * gelu'(x)`, one element per thread; the kernel flags both inputs
    /// and the output.
    pub(super) fn gelu_erf_bwd(&mut self, x: Arg, gy: Arg) -> Res<Reply> {
        const OP: &str = "gelu_erf_backward";
        let (xv, gv) = (self.view(x)?, self.view(gy)?);
        if xv.n != gv.n {
            return Err(metal_err(format!("{OP}: gradient does not match input")));
        }
        let n = u32_of(xv.n)?;
        let out = self.fresh(xv.n)?;
        let st = self.status(OP)?;
        self.k1("ojas_gelu_erf_bwd", xv.n, |bd| {
            bind(bd, &xv, 0);
            bind(bd, &gv, 1);
            bind(bd, &out, 2);
            set_u32(bd, n, 3);
            bind_st(bd, &st, 4);
        })?;
        Ok(self.keep(vec![out]))
    }

    /// Softmax backward, a threadgroup per row.
    pub(super) fn softmax_bwd(&mut self, y: Arg, gy: Arg, rows: u32, dim: u32) -> Res<Reply> {
        const OP: &str = "softmax_backward";
        let (yv, gv) = (self.view(y)?, self.view(gy)?);
        let n = (rows as usize)
            .checked_mul(dim as usize)
            .ok_or_else(|| metal_err(format!("{OP}: extent overflows")))?;
        if rows == 0 || dim == 0 || yv.n != n || gv.n != n {
            return Err(metal_err(format!(
                "{OP}: buffers do not hold {rows} rows of {dim}"
            )));
        }
        let out = self.fresh(n)?;
        let st = self.status(OP)?;
        self.ktg(
            "ojas_softmax_bwd_rows",
            rows as usize,
            1,
            row_threads(dim) as usize,
            |bd| {
                bind(bd, &yv, 0);
                bind(bd, &gv, 1);
                bind(bd, &out, 2);
                bind_st(bd, &st, 3);
                set_u32(bd, rows, 4);
                set_u32(bd, dim, 5);
            },
        )?;
        Ok(self.keep(vec![out]))
    }

    /// One operand's broadcast gradient: `ojas_bcast_grad_part` (a thread per
    /// element and split) and, with more than one split, `ojas_bcast_grad_fin`.
    pub(super) fn bcast_grad(
        &mut self,
        op: &'static str,
        g: Arg,
        other: Option<Arg>,
        scan: &[Arg],
        geom: &BcastGradGeom,
    ) -> Res<Reply> {
        let gv = self.view(g)?;
        let ov = other.map(|a| self.view(a)).transpose()?;
        if !geom.fits(gv.n, ov.as_ref().map(|v| v.n)) {
            return Err(metal_err(format!(
                "{op}: broadcast gradient plan does not fit its buffers"
            )));
        }
        let elems = geom.elems as usize;
        let out = self.fresh(elems)?;
        let part = if geom.splits > 1 {
            Some(
                self.fresh(
                    elems
                        .checked_mul(geom.splits as usize * 2)
                        .ok_or_else(|| metal_err(format!("{op}: partials overflow")))?,
                )?,
            )
        } else {
            None
        };
        let st = self.status(op)?;
        for a in scan {
            let v = self.view(*a)?;
            self.check(&st, &v, ST_IN)?;
        }
        let words = geom.bytes();
        self.k2("ojas_bcast_grad_part", elems, geom.splits as usize, |bd| {
            bind(bd, &gv, 0);
            bind(bd, ov.as_ref().unwrap_or(&gv), 1);
            bind(bd, part.as_ref().unwrap_or(&out), 2);
            bind_st(bd, &st, 3);
            bd.bind_bytes(&words, 4);
        })?;
        if let Some(part) = &part {
            self.k1("ojas_bcast_grad_fin", elems, |bd| {
                bind(bd, part, 0);
                bind(bd, &out, 1);
                bind_st(bd, &st, 2);
                set_u32(bd, geom.elems, 3);
                set_u32(bd, geom.splits, 4);
            })?;
        }
        Ok(self.keep(vec![out]))
    }
}
