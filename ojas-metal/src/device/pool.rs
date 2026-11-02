//! `max_pool2d_forward` and `max_pool2d_backward` on the device thread:
//! `ojas_max_pool2d_fwd` and `ojas_max_pool2d_bwd` (`kernels/max_pool.metal`).
//!
//! Both scan every operand with `ojas_check_finite` first, as the CPU refuses
//! a non-finite value wherever it sits (a pixel the floor-mode remainder
//! leaves out of every window included). The backward's kernel flags its own
//! output.

use super::{bind, bind_st, set_u32, u32_of, Worker, ST_IN};
use crate::link::{metal_err, Arg, PoolGeom, Reply, Res};

/// The element counts a `PoolGeom` describes, `(input, output)`, after
/// checking that every window lies inside its image. The kernels index in 32
/// bits and trust this check; a geometry that fails it is a bug in the
/// caller, not a user error (the validator has refused those).
fn extents(op: &'static str, g: &PoolGeom) -> Res<(usize, usize)> {
    let [kh, kw] = g.kernel;
    let [sh, sw] = g.stride;
    if [g.planes, g.h, g.w, g.oh, g.ow, kh, kw, sh, sw].contains(&0) {
        return Err(metal_err(format!("{op}: a pooling extent is zero")));
    }
    // The last window starts at (oh - 1) * sh and spans kh rows.
    let reach = |count: u32, stride: u32, span: u32| {
        (u64::from(count) - 1) * u64::from(stride) + u64::from(span)
    };
    if reach(g.oh, sh, kh) > u64::from(g.h) || reach(g.ow, sw, kw) > u64::from(g.w) {
        return Err(metal_err(format!(
            "{op}: the windows do not fit a [{}, {}] image",
            g.h, g.w
        )));
    }
    let count = |rows: u32, cols: u32| -> Res<usize> {
        (g.planes as usize)
            .checked_mul(rows as usize)
            .and_then(|n| n.checked_mul(cols as usize))
            .ok_or_else(|| metal_err(format!("{op}: extent overflows")))
    };
    let (input, output) = (count(g.h, g.w)?, count(g.oh, g.ow)?);
    // The kernels index with 32-bit thread ids.
    u32_of(input)?;
    u32_of(output)?;
    Ok((input, output))
}

impl Worker {
    /// `max_pool2d_forward`: one thread per output element.
    pub(super) fn max_pool2d(&mut self, x: Arg, g: PoolGeom) -> Res<Reply> {
        const OP: &str = "max_pool2d_forward";
        let xv = self.view(x)?;
        let (in_n, out_n) = extents(OP, &g)?;
        if in_n != xv.n {
            return Err(metal_err(format!(
                "{OP}: the buffer does not hold {} planes of [{}, {}]",
                g.planes, g.h, g.w
            )));
        }
        let y = self.fresh(out_n)?;
        let st = self.status(OP)?;
        self.check(&st, &xv, ST_IN)?;
        let total = u32_of(out_n)?;
        self.k1("ojas_max_pool2d_fwd", out_n, |b| {
            bind(b, &xv, 0);
            bind(b, &y, 1);
            set_u32(b, total, 2);
            set_u32(b, g.h, 3);
            set_u32(b, g.w, 4);
            set_u32(b, g.oh, 5);
            set_u32(b, g.ow, 6);
            set_u32(b, g.kernel[0], 7);
            set_u32(b, g.kernel[1], 8);
            set_u32(b, g.stride[0], 9);
            set_u32(b, g.stride[1], 10);
        })?;
        Ok(self.keep(vec![y]))
    }

    /// `max_pool2d_backward`: one thread per input element, a gather over
    /// the windows that cover it.
    pub(super) fn max_pool2d_bwd(&mut self, x: Arg, dy: Arg, g: PoolGeom) -> Res<Reply> {
        const OP: &str = "max_pool2d_backward";
        let (xv, dv) = (self.view(x)?, self.view(dy)?);
        let (in_n, out_n) = extents(OP, &g)?;
        if in_n != xv.n || out_n != dv.n {
            return Err(metal_err(format!(
                "{OP}: the buffers do not hold {} planes of [{}, {}] and their [{}, {}] gradient",
                g.planes, g.h, g.w, g.oh, g.ow
            )));
        }
        let dx = self.fresh(in_n)?;
        let st = self.status(OP)?;
        self.check(&st, &xv, ST_IN)?;
        self.check(&st, &dv, ST_IN)?;
        let total = u32_of(in_n)?;
        let bf16 = u32::from(self.bf16_accum());
        self.k1("ojas_max_pool2d_bwd", in_n, |b| {
            bind(b, &xv, 0);
            bind(b, &dv, 1);
            bind(b, &dx, 2);
            set_u32(b, total, 3);
            set_u32(b, g.h, 4);
            set_u32(b, g.w, 5);
            set_u32(b, g.oh, 6);
            set_u32(b, g.ow, 7);
            set_u32(b, g.kernel[0], 8);
            set_u32(b, g.kernel[1], 9);
            set_u32(b, g.stride[0], 10);
            set_u32(b, g.stride[1], 11);
            bind_st(b, &st, 12);
            set_u32(b, bf16, 13);
        })?;
        Ok(self.keep(vec![dx]))
    }
}
