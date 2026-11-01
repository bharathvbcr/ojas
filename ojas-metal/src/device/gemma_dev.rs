//! Device side of the EmbeddingGemma 2 encoder ops that are not attention:
//! the commands of `backend/gemma_host.rs` encoded as dispatches of
//! `kernels/gemma_ops.metal` and of tessl's `segment_mean_rows` and
//! `l2_normalize_rows`. Each op takes one status slot; the GELU kernels flag
//! the input they read and the output they write themselves, and the two
//! tessl-backed ops are scanned here, so a fault is reported at the next sync
//! point like every other op's.

use tessl::dispatch::set_u32;
use tessl::embedgemma2::{l2_normalize_rows, segment_mean_rows};

use super::{bind, bind_st, u32_of, Worker, ST_IN, ST_OUT};
use crate::gemma_ops::SeqMeanGeom;
use crate::link::{metal_err, Arg, Reply, Res};

impl Worker {
    /// `gelu_tanh_forward` (`gy` is `None`) or `gelu_tanh_backward`: one
    /// element a thread; the kernels check what they read and write.
    pub(super) fn gelu_tanh(&mut self, x: Arg, gy: Option<Arg>) -> Res<Reply> {
        let op = if gy.is_some() {
            "gelu_tanh_backward"
        } else {
            "gelu_tanh_forward"
        };
        let xv = self.view(x)?;
        let gv = gy.map(|g| self.view(g)).transpose()?;
        if gv.as_ref().is_some_and(|g| g.n != xv.n) {
            return Err(metal_err(format!(
                "{op}: the gradient does not hold the input's {} elements",
                xv.n
            )));
        }
        let n = u32_of(xv.n)?;
        let y = self.fresh(xv.n)?;
        let st = self.status(op)?;
        match &gv {
            None => self.k1("ojas_gelu_tanh_fwd", xv.n, |b| {
                bind(b, &xv, 0);
                bind(b, &y, 1);
                set_u32(b, n, 2);
                bind_st(b, &st, 3);
            })?,
            Some(g) => self.k1("ojas_gelu_tanh_bwd", xv.n, |b| {
                bind(b, &xv, 0);
                bind(b, g, 1);
                bind(b, &y, 2);
                set_u32(b, n, 3);
                bind_st(b, &st, 4);
            })?,
        }
        Ok(self.keep(vec![y]))
    }

    /// `seq_mean_forward`: `ojas_segments_from_lens` turns the device length
    /// buffer into `[batch, 2]` row ranges (no readback), then tessl's
    /// `segment_mean_rows` averages each range of the flattened `[batch *
    /// seq, dim]` operand. tessl adds a column in row order in `f32` and
    /// divides by the count, and an empty range is zeros, which is the
    /// length-0 rule. It scans nothing, so the operand and the output are
    /// checked here; the lengths are U32.
    pub(super) fn seq_mean(&mut self, x: Arg, lens: Arg, g: SeqMeanGeom) -> Res<Reply> {
        const OP: &str = "seq_mean_forward";
        let [xv] = self.wholes([x])?;
        let [lv] = self.wholes([lens])?;
        let rows = g
            .rows()
            .ok_or_else(|| metal_err(format!("{OP}: {} x {} rows overflow", g.batch, g.seq)))?;
        let want = (rows as usize).checked_mul(g.dim as usize);
        let out_n = g.out_elems();
        if g.batch == 0
            || g.seq == 0
            || g.dim == 0
            || want != Some(xv.n)
            || out_n.is_none()
            || lv.n != g.batch as usize
        {
            return Err(metal_err(format!(
                "{OP}: buffers do not hold x [{}, {}, {}] and {} lengths",
                g.batch, g.seq, g.dim, g.batch
            )));
        }
        let out = self.fresh(out_n.unwrap_or(0))?;
        let seg = self.fresh(2 * g.batch as usize)?;
        let st = self.status(OP)?;
        self.check(&st, &xv, ST_IN)?;
        self.k1("ojas_segments_from_lens", g.batch as usize, |b| {
            bind(b, &lv, 0);
            bind(b, &seg, 1);
            set_u32(b, g.batch, 2);
            set_u32(b, g.seq, 3);
        })?;
        segment_mean_rows(&self.rt, &xv.buf, &seg.buf, &out.buf, g.batch, rows, g.dim)
            .map_err(metal_err)?;
        self.check(&st, &out, ST_OUT)?;
        Ok(self.keep(vec![out]))
    }

    /// `l2_normalize_forward` at tessl's epsilon: a copy of the operand
    /// normalized in place by `l2_normalize_rows` (`x / max(||x||, 1e-12)`,
    /// the squares summed in `f32` in tessl's lane order). It scans nothing,
    /// so the operand and the output are checked here.
    pub(super) fn l2_normalize(&mut self, x: Arg, rows: u32, dim: u32) -> Res<Reply> {
        const OP: &str = "l2_normalize_forward";
        let xv = self.view(x)?;
        let want = (rows as usize).checked_mul(dim as usize);
        if rows == 0 || dim == 0 || want != Some(xv.n) {
            return Err(metal_err(format!(
                "{OP}: buffers do not hold {rows} rows of {dim}"
            )));
        }
        let y = self.fresh(xv.n)?;
        let st = self.status(OP)?;
        self.check(&st, &xv, ST_IN)?;
        self.copy(&xv, &y)?;
        l2_normalize_rows(&self.rt, &y.buf, rows, dim, dim).map_err(metal_err)?;
        self.check(&st, &y, ST_OUT)?;
        Ok(self.keep(vec![y]))
    }
}
