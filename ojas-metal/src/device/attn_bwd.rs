//! `Backend::attention_backward` on the device thread: the general tiled
//! backward of `attn_gen_bwd.metal` over `attn_bwd_tiled.h`.
//!
//! Bidirectional or causal attention, `Tq != Tk`, an additive score bias with
//! its gradient, per-batch key lengths and grouped-query heads, at every
//! compiled head-dim width. The causal self-attention backward of
//! `causal_sdpa_backward` runs the same bodies (`ojas_attn_bwd_dq_dD`,
//! `ojas_attn_bwd_dkv_dD` in `ojas_backend.metal`), with the mask fixed.

use super::*;
use crate::link::AttnBwdGeom;

/// Flag bits of the kernels' mask (`AGB_*` in `attn_bwd_tiled.h`).
const FLAG_CAUSAL: u32 = 1;
const FLAG_BIAS: u32 = 2;
const FLAG_LENS: u32 = 4;
const FLAG_DEAD_ROWS: u32 = 8;

/// Query rows of one tile (`ATT_BQ`), and key rows (`ATT_BK`).
const TILE_Q: usize = ATTN_ROWS;
const TILE_K: usize = ATTN_ROWS;

impl Worker {
    /// The tiled attention backward, from the forward's output and
    /// log-sum-exp: `dr` forms each query row's `Dr = dO . O`, `dq` walks the
    /// key blocks each query block sees, `dkv` walks (per key block, the
    /// query blocks of every query plane sharing the KV plane), and, when the
    /// bias gradient is asked for, `dbias` gives every bias tile to one
    /// threadgroup that sums the planes the bias broadcasts over in a fixed
    /// order. No T x T buffer, no expanded K or V, no atomics and no wait.
    ///
    /// Every operand is scanned for NaN and infinity first, as the CPU
    /// refuses one wherever it sits, and every gradient after.
    pub(super) fn attn_gen_bwd(
        &mut self,
        op: &'static str,
        [q, k, v, o, lse, gy]: [Arg; 6],
        bias: Option<Arg>,
        lens: Option<Arg>,
        geom: AttnBwdGeom,
    ) -> Res<Reply> {
        let g = geom.attn;
        let width = Self::attn_width(g.d)?;
        if !geom.scale.is_finite() {
            return Err(metal_err(format!("{op}: attention scale is not finite")));
        }
        if g.rep == 0
            || !g.bh.is_multiple_of(g.rep)
            || g.heads == 0
            || !g.bh.is_multiple_of(g.heads)
        {
            return Err(metal_err(format!(
                "{op}: {} query planes do not group by {} heads and {} per KV plane",
                g.bh, g.heads, g.rep
            )));
        }
        if geom.bias_grad && (bias.is_none() || g.bias.is_none()) {
            return Err(metal_err(format!("{op}: a bias gradient needs a bias")));
        }
        if geom.radius != 0 && (g.causal || g.q_len != g.kv_len) {
            return Err(metal_err(format!(
                "{op}: a radius needs bidirectional self-attention, got causal {} Tq {} Tk {}",
                g.causal, g.q_len, g.kv_len
            )));
        }
        if geom.dead_rows && (!g.has_lens || g.q_len != g.kv_len) {
            return Err(metal_err(format!(
                "{op}: dead query rows need key lengths and Tq == Tk, got lens {} Tq {} Tk {}",
                g.has_lens, g.q_len, g.kv_len
            )));
        }
        attn_plane_fits(op, g.q_len, g.d)?;
        attn_plane_fits(op, g.kv_len, g.d)?;

        let (qv, kv, vv) = (self.view(q)?, self.view(k)?, self.view(v)?);
        let (ov, lv, gv) = (self.view(o)?, self.view(lse)?, self.view(gy)?);
        let bv = bias.map(|a| self.view(a)).transpose()?;
        let nv = lens.map(|a| self.view(a)).transpose()?;
        let (q_len, kv_len, d) = (g.q_len as usize, g.kv_len as usize, g.d as usize);
        let planes = g.bh as usize;
        let batches = planes / g.heads as usize;
        let q_n = planes * q_len * d;
        let kv_n = (planes / g.rep as usize) * kv_len * d;
        let rows_n = planes * q_len;
        let bias_n = g
            .bias
            .map(|[b, h, t]| b as usize * h as usize * t as usize * kv_len);
        let fits = qv.n == q_n
            && ov.n == q_n
            && gv.n == q_n
            && kv.n == kv_n
            && vv.n == kv_n
            && lv.n == rows_n
            && bv.as_ref().map(|b| b.n) == bias_n
            && nv.as_ref().map(|l| l.n) == g.has_lens.then_some(batches);
        if !fits {
            return Err(metal_err(format!(
                "{op}: buffers do not hold {planes} query planes of [{q_len}, {d}], {} kv planes \
                 of [{kv_len}, {d}] and the bias and key lengths asked for",
                planes / g.rep as usize
            )));
        }

        let dvec = self.fresh(rows_n)?;
        let dq = self.fresh(q_n)?;
        let dk = self.fresh(kv_n)?;
        let dv = self.fresh(kv_n)?;
        let dbias = match (geom.bias_grad, bias_n) {
            (true, Some(n)) => Some(self.fresh(n)?),
            _ => None,
        };
        let st = self.status(op)?;
        for (a, x) in [
            (q, &qv),
            (k, &kv),
            (v, &vv),
            (o, &ov),
            (lse, &lv),
            (gy, &gv),
        ] {
            self.check_operand(&st, a, x, ST_IN)?;
        }
        if let (Some(a), Some(x)) = (bias, bv.as_ref()) {
            self.check_operand(&st, a, x, ST_IN)?;
        }

        let rows = u32_of(rows_n)?;
        self.ktg(
            "ojas_attn_bwd_dr",
            (rows as usize).div_ceil(ATTN_THREADS / 32),
            1,
            ATTN_THREADS,
            |b| {
                bind(b, &gv, 0);
                bind(b, &ov, 1);
                bind(b, &dvec, 2);
                set_u32(b, rows, 3);
                set_u32(b, g.d, 4);
            },
        )?;

        let flags = (if g.causal { FLAG_CAUSAL } else { 0 })
            | (if g.bias.is_some() { FLAG_BIAS } else { 0 })
            | (if g.has_lens { FLAG_LENS } else { 0 })
            | (if geom.dead_rows { FLAG_DEAD_ROWS } else { 0 });
        let [bias_b, bias_h, bias_q] = g.bias.unwrap_or([1, 1, 1]);
        // A buffer the kernel does not read this call is still bound:
        // Metal's validation wants every declared argument set.
        let (bias_buf, lens_buf) = (bv.as_ref().unwrap_or(&qv), nv.as_ref().unwrap_or(&qv));
        let common = |b: &mut Binder<'_>| {
            bind(b, &qv, 0);
            bind(b, &kv, 1);
            bind(b, &vv, 2);
            bind(b, &gv, 3);
            bind(b, &lv, 4);
            bind(b, &dvec, 5);
            bind(b, bias_buf, 6);
            bind(b, lens_buf, 7);
        };
        // The mask and the geometry, from buffer index `at` on: Tq, Tk, D, H,
        // rep, flags, W, R, and the bias extents and the scale.
        let mask = |b: &mut Binder<'_>, at: usize| {
            set_u32(b, g.q_len, at);
            set_u32(b, g.kv_len, at + 1);
            set_u32(b, g.d, at + 2);
            set_u32(b, g.heads, at + 3);
            set_u32(b, g.rep, at + 4);
            set_u32(b, flags, at + 5);
            set_u32(b, g.window, at + 6);
            set_u32(b, geom.radius, at + 7);
            set_u32(b, bias_b, at + 8);
            set_u32(b, bias_h, at + 9);
            set_u32(b, bias_q, at + 10);
            set_f32(b, geom.scale, at + 11);
        };
        self.ktg(
            &format!("ojas_attn_gen_bwd_dq_d{width}"),
            q_len.div_ceil(TILE_Q),
            planes,
            ATTN_THREADS,
            |b| {
                common(b);
                bind(b, &dq, 8);
                mask(b, 9);
            },
        )?;
        self.ktg(
            &format!("ojas_attn_gen_bwd_dkv_d{width}"),
            kv_len.div_ceil(TILE_K),
            planes / g.rep as usize,
            ATTN_THREADS,
            |b| {
                common(b);
                bind(b, &dk, 8);
                bind(b, &dv, 9);
                mask(b, 10);
            },
        )?;
        if let (Some(dbias), Some([bb, bh, bq])) = (dbias.as_ref(), g.bias) {
            // One tile of the bias per threadgroup, or, for a bias with one
            // query row, 32 of its columns: the group sums every query block.
            let batch_count = u32_of(batches)?;
            let bias_planes = bb as usize * bh as usize;
            let y = if bq == 1 {
                bias_planes
            } else {
                bias_planes * q_len.div_ceil(TILE_Q)
            };
            self.ktg(
                &format!("ojas_attn_gen_bwd_dbias_d{width}"),
                kv_len.div_ceil(TILE_K),
                y,
                ATTN_THREADS,
                |b| {
                    common(b);
                    bind(b, dbias, 8);
                    mask(b, 9);
                    set_u32(b, batch_count, 21);
                },
            )?;
        }
        let mut outs = vec![dq, dk, dv];
        outs.extend(dbias);
        for x in &outs {
            self.check(&st, x, ST_OUT)?;
        }
        Ok(self.keep(outs))
    }
}

#[cfg(test)]
#[path = "attn_bwd_tests.rs"]
mod tests;
