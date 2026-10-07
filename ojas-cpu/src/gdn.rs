//! The gated delta rule (Gated DeltaNet's mixer), forward and backward, at
//! transformers' training seam `torch_chunk_gated_delta_rule(q, k, v, g,
//! beta, initial_state, use_qk_l2norm_in_kernel=True)`. Per `(b, h)`, with
//! the state `S` `[Dk, Dv]`:
//!
//! ```text
//! q^ = l2norm(q) / sqrt(Dk),  k^ = l2norm(k)       l2norm(x) = x / sqrt(|x|^2 + 1e-6)
//! S^ = exp(g_t) S
//! u  = v_t - S^^T k^
//! S  = S^ + k^ (beta_t u)^T
//! o_t = S^T q^
//! ```
//!
//! This is the arithmetic of tessl's `gdn_train` kernels and of the f64
//! reference in `ojas-oracle` (`gdn`), in the same operation order, with
//! one exception: `l2norm` sums squares and applies its scale in `f64`.
//! An `f32` sum of squares overflows for any element above about 1.8e19,
//! and `x * rsqrt(inf)` then normalizes the row to zeros, a finite and
//! wrong result (transformers' f32 path and tessl's Metal kernel both do
//! this). In `f64` no finite `f32` row overflows, so every finite input
//! normalizes correctly; within range the result differs from the `f32`
//! form only in rounding. The
//! forward saves the state every [`GDN_CHECKPOINT_TOKENS`] tokens; the
//! backward walks the chunks in reverse, recomputes each chunk's states from
//! its checkpoint, then runs the reverse-mode recurrence over it.
//!
//! One arithmetic serves both numerics contracts: every sum ascends its
//! index from `0.0` with a separate multiply and add, the decay is
//! [`exp_exact`], and tasks split whole `(b, h)` heads, so the bits do not
//! depend on [`ojas_core::Numerics`], the thread count or the platform.
//!
//! The operands' per-token layout is `[B, T, H, ...]`, so one head's rows
//! are not contiguous. Each head is computed into a head-major scratch
//! (`[B, H, T, ...]`, charged to the budget) and then moved into the
//! output; the states and checkpoints are head-major already.

use ojas_core::{
    exp_exact, sdpa_scale, Budget, GdnDims, OjasError, GDN_CHECKPOINT_TOKENS, GDN_L2NORM_EPS,
};

use crate::attn::fill_parts;
use crate::pool::Exec;
use crate::validate::{product, room_for, shape};

const CKPT: usize = GDN_CHECKPOINT_TOKENS;

/// The forward's operand values, in place: `q`, `k` `[B, T, H, Dk]`, `v`
/// `[B, T, H, Dv]`, `g`, `beta` `[B, T, H]`, `s0` `[B, H, Dk, Dv]`.
#[derive(Clone, Copy)]
pub(crate) struct Operands<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub g: &'a [f32],
    pub beta: &'a [f32],
    pub s0: Option<&'a [f32]>,
}

/// The backward's outputs: `dq`, `dk`, `dv`, `dg`, `dbeta` in the inputs'
/// layouts, and `ds0` when the forward had an initial state.
pub(crate) struct GradsOut<'a> {
    pub dq: &'a mut [f32],
    pub dk: &'a mut [f32],
    pub dv: &'a mut [f32],
    pub dg: &'a mut [f32],
    pub dbeta: &'a mut [f32],
    pub ds0: Option<&'a mut [f32]>,
}

/// Head geometry shared by both passes.
#[derive(Clone, Copy)]
struct Geom {
    t: usize,
    h: usize,
    dk: usize,
    dv: usize,
    nc: usize,
    per: usize,
    heads: usize,
    scale: f32,
}

impl Geom {
    fn new(op: &'static str, d: &GdnDims) -> Result<Self, OjasError> {
        let dk32 = u32::try_from(d.key_dim)
            .map_err(|_| shape(op, format!("gdn key dim {} does not fit u32", d.key_dim)))?;
        Ok(Self {
            t: d.seq,
            h: d.heads,
            dk: d.key_dim,
            dv: d.value_dim,
            nc: d.checkpoints,
            per: product(op, &[d.key_dim, d.value_dim])?,
            heads: product(op, &[d.batch, d.heads])?,
            scale: sdpa_scale(dk32)?,
        })
    }

    /// The operand row of token `ti` of head `head` (`b * H + h`).
    fn row(&self, head: usize, ti: usize) -> usize {
        let (bi, hi) = (head / self.h, head % self.h);
        (bi * self.t + ti) * self.h + hi
    }

    /// Multiply-adds of one token's step, for the pool's split.
    fn work(&self, passes: usize) -> usize {
        self.heads
            .saturating_mul(self.t)
            .saturating_mul(self.per)
            .saturating_mul(passes)
    }

    /// Tasks that run at once: every head when the pass is split, else one.
    fn inflight(&self, exec: Exec<'_>, work: usize) -> usize {
        if work >= 2 * crate::attn::TASK_WORK {
            self.heads.min(exec.pool.threads()).max(1)
        } else {
            1
        }
    }
}

/// Each `d`-wide row through `l2norm`: the normalized rows, and each
/// row's `1 / sqrt(|x|^2 + eps)` in `f64`. The sum of squares, the scale
/// and the product are `f64` (ascending, no fused multiply-add), so no
/// finite row overflows or loses its scale to an `f32` subnormal.
fn l2norm_rows(x: &[f32], d: usize) -> (Vec<f32>, Vec<f64>) {
    let mut out = Vec::with_capacity(x.len());
    let mut rs = Vec::with_capacity(x.len() / d.max(1));
    for row in x.chunks_exact(d) {
        let mut ss = 0.0f64;
        for &v in row {
            let v = f64::from(v);
            ss += v * v;
        }
        let r = 1.0 / (ss + f64::from(GDN_L2NORM_EPS)).sqrt();
        out.extend(row.iter().map(|&v| (f64::from(v) * r) as f32));
        rs.push(r);
    }
    (out, rs)
}

/// One token on one head's state `st` `[dk, dv]`, in place: decay by `a`,
/// read the decayed state against `k`, and add `k (beta u)^T`. Leaves
/// `u = v - S^^T k` and `delta = beta u`.
fn step(st: &mut [f32], k: &[f32], v: &[f32], a: f32, beta: f32, u: &mut [f32], delta: &mut [f32]) {
    let dv = v.len();
    for s in st.iter_mut() {
        *s *= a;
    }
    read_columns(st, k, u);
    for ((uj, dj), &vj) in u.iter_mut().zip(delta.iter_mut()).zip(v) {
        *uj = vj - *uj;
        *dj = beta * *uj;
    }
    for (row, &ki) in st.chunks_exact_mut(dv).zip(k) {
        for (s, &dj) in row.iter_mut().zip(delta.iter()) {
            *s += ki * dj;
        }
    }
}

/// `out[j] = sum_i st[i, j] * x[i]`, each column summed in increasing `i`
/// from `0.0`.
fn read_columns(st: &[f32], x: &[f32], out: &mut [f32]) {
    let dv = out.len();
    out.fill(0.0);
    for (row, &xi) in st.chunks_exact(dv).zip(x) {
        for (o, &s) in out.iter_mut().zip(row) {
            *o += s * xi;
        }
    }
}

/// `dst[row(head, t)]` from head-major `src[head, t]`, `width` values each.
fn to_token_major(g: &Geom, src: &[f32], dst: &mut [f32], width: usize) {
    for head in 0..g.heads {
        for ti in 0..g.t {
            let row = g.row(head, ti);
            let from = (head * g.t + ti) * width;
            dst[row * width..(row + 1) * width].copy_from_slice(&src[from..from + width]);
        }
    }
}

/// Forward into `[output, final_state, checkpoints]`, which the caller
/// charged and zeroed.
pub(crate) fn forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    d: GdnDims,
    x: Operands<'_>,
    [out, fin, ckpt]: [&mut [f32]; 3],
) -> Result<(), OjasError> {
    let g = Geom::new(op, &d)?;
    let rows = d.rows();
    let work = g.work(3);
    // Normalized q and k, their norms, and the head-major output.
    let scratch = product(op, &[rows, 2 * g.dk + 2 + g.dv])?;
    let per_task = product(op, &[g.inflight(exec, work), g.per + 2 * g.dv])?;
    let _hold = room_for(op, budget, scratch.saturating_add(per_task))?;
    let (mut qh, _) = l2norm_rows(x.q, g.dk);
    for q in qh.iter_mut() {
        *q *= g.scale;
    }
    let (kh, _) = l2norm_rows(x.k, g.dk);
    let mut heads_out = vec![0.0f32; rows * g.dv];
    let cancel = exec.pool.cancel_hook();
    let parts: Vec<_> = heads_out
        .chunks_mut(g.t * g.dv)
        .zip(fin.chunks_mut(g.per))
        .zip(ckpt.chunks_mut(g.nc * g.per))
        .map(|((o, f), c)| (o, f, c))
        .collect();
    fill_parts(exec, work, parts, |head, (oh, fh, ch)| {
        cancel()?;
        let mut st = match x.s0 {
            Some(s0) => s0[head * g.per..(head + 1) * g.per].to_vec(),
            None => vec![0.0f32; g.per],
        };
        let (mut u, mut delta) = (vec![0.0f32; g.dv], vec![0.0f32; g.dv]);
        for ti in 0..g.t {
            if ti % CKPT == 0 {
                let c = (ti / CKPT) * g.per;
                ch[c..c + g.per].copy_from_slice(&st);
            }
            let row = g.row(head, ti);
            step(
                &mut st,
                &kh[row * g.dk..(row + 1) * g.dk],
                &x.v[row * g.dv..(row + 1) * g.dv],
                exp_exact(x.g[row]),
                x.beta[row],
                &mut u,
                &mut delta,
            );
            read_columns(
                &st,
                &qh[row * g.dk..(row + 1) * g.dk],
                &mut oh[ti * g.dv..(ti + 1) * g.dv],
            );
        }
        fh.copy_from_slice(&st);
        Ok(())
    })?;
    to_token_major(&g, &heads_out, out, g.dv);
    Ok(())
}

/// One head's slices of the head-major gradient scratch.
struct HeadGrads<'a> {
    dq: &'a mut [f32],
    dk: &'a mut [f32],
    dv: &'a mut [f32],
    dg: &'a mut [f32],
    dbeta: &'a mut [f32],
    ds0: Option<&'a mut [f32]>,
}

/// Backward into `grads` (charged and zeroed by the caller) from the
/// forward's inputs, its checkpoints, `d_o` `[B, T, H, Dv]` and `dfin`
/// `[B, H, Dk, Dv]` (zeros when `None`).
// The reverse-mode step indexes its vectors as the f64 reference in
// ojas-cuda/tests/reference/gdn_published.rs does, term for term.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    d: GdnDims,
    x: Operands<'_>,
    ckpt: &[f32],
    d_o: &[f32],
    dfin: Option<&[f32]>,
    grads: GradsOut<'_>,
) -> Result<(), OjasError> {
    let g = Geom::new(op, &d)?;
    let rows = d.rows();
    let work = g.work(8);
    // Normalized q and k with their norms, and the head-major gradients.
    let scratch = product(op, &[rows, 4 * g.dk + g.dv + 4])?;
    // A chunk of states, the running state, its gradient, the decayed state
    // and its gradient, and the per-token vectors.
    let task = product(op, &[CKPT + 4, g.per])?
        .saturating_add(4 * g.dv)
        .saturating_add(3 * g.dk);
    let per_task = product(op, &[g.inflight(exec, work), task])?;
    let _hold = room_for(op, budget, scratch.saturating_add(per_task))?;
    let (qn, rq) = l2norm_rows(x.q, g.dk);
    let (kh, rk) = l2norm_rows(x.k, g.dk);
    let mut h_dq = vec![0.0f32; rows * g.dk];
    let mut h_dk = vec![0.0f32; rows * g.dk];
    let mut h_dv = vec![0.0f32; rows * g.dv];
    let mut h_dg = vec![0.0f32; rows];
    let mut h_dbeta = vec![0.0f32; rows];
    let ds0_parts: Vec<Option<&mut [f32]>> = match grads.ds0 {
        Some(ds0) => ds0.chunks_mut(g.per).map(Some).collect(),
        None => (0..g.heads).map(|_| None).collect(),
    };
    let parts: Vec<HeadGrads<'_>> = h_dq
        .chunks_mut(g.t * g.dk)
        .zip(h_dk.chunks_mut(g.t * g.dk))
        .zip(h_dv.chunks_mut(g.t * g.dv))
        .zip(h_dg.chunks_mut(g.t))
        .zip(h_dbeta.chunks_mut(g.t))
        .zip(ds0_parts)
        .map(|(((((dq, dk), dv), dg), dbeta), ds0)| HeadGrads {
            dq,
            dk,
            dv,
            dg,
            dbeta,
            ds0,
        })
        .collect();
    let cancel = exec.pool.cancel_hook();
    let (dk, dv, per) = (g.dk, g.dv, g.per);
    fill_parts(exec, work, parts, |head, out| {
        cancel()?;
        let mut ds = match dfin {
            Some(df) => df[head * per..(head + 1) * per].to_vec(),
            None => vec![0.0f32; per],
        };
        let mut states = vec![0.0f32; CKPT * per];
        let mut st = vec![0.0f32; per];
        let mut sh = vec![0.0f32; per];
        let mut dsh = vec![0.0f32; per];
        let (mut u, mut delta, mut ddelta) = (vec![0.0f32; dv], vec![0.0f32; dv], vec![0.0f32; dv]);
        let (mut dqh, mut dkh) = (vec![0.0f32; dk], vec![0.0f32; dk]);
        for cc in (0..g.nc).rev() {
            let (t0, t1) = (cc * CKPT, g.t.min(cc * CKPT + CKPT));
            // The state entering every token of the chunk, from its checkpoint.
            let c = (head * g.nc + cc) * per;
            st.copy_from_slice(&ckpt[c..c + per]);
            for ti in t0..t1 {
                states[(ti - t0) * per..(ti - t0 + 1) * per].copy_from_slice(&st);
                let row = g.row(head, ti);
                step(
                    &mut st,
                    &kh[row * dk..(row + 1) * dk],
                    &x.v[row * dv..(row + 1) * dv],
                    exp_exact(x.g[row]),
                    x.beta[row],
                    &mut u,
                    &mut delta,
                );
            }
            for ti in (t0..t1).rev() {
                cancel()?;
                let row = g.row(head, ti);
                let qr = &qn[row * dk..(row + 1) * dk];
                let kr = &kh[row * dk..(row + 1) * dk];
                let vr = &x.v[row * dv..(row + 1) * dv];
                let dor = &d_o[row * dv..(row + 1) * dv];
                let a = exp_exact(x.g[row]);
                let beta = x.beta[row];
                let prev = &states[(ti - t0) * per..(ti - t0 + 1) * per];
                for (s, &p) in sh.iter_mut().zip(prev) {
                    *s = p * a;
                }
                // u = v - S^^T k^, delta = beta u, as the forward formed them.
                read_columns(&sh, kr, &mut u);
                for j in 0..dv {
                    u[j] = vr[j] - u[j];
                    delta[j] = beta * u[j];
                }
                // o = S^T q^ with S = S^ + k^ delta^T.
                for i in 0..dk {
                    let srow = &sh[i * dv..(i + 1) * dv];
                    let mut acc = 0.0f32;
                    for j in 0..dv {
                        let s = srow[j] + kr[i] * delta[j];
                        acc += s * dor[j];
                    }
                    dqh[i] = acc;
                    let qi = qr[i] * g.scale;
                    for (dsv, &dj) in ds[i * dv..(i + 1) * dv].iter_mut().zip(dor) {
                        *dsv += qi * dj;
                    }
                }
                // S = S^ + k^ delta^T.
                read_columns(&ds, kr, &mut ddelta);
                for i in 0..dk {
                    let mut acc = 0.0f32;
                    for (&dsv, &dj) in ds[i * dv..(i + 1) * dv].iter().zip(delta.iter()) {
                        acc += dsv * dj;
                    }
                    dkh[i] = acc;
                }
                // delta = beta (v - S^^T k^).
                let mut dbeta = 0.0f32;
                for j in 0..dv {
                    out.dv[ti * dv + j] = beta * ddelta[j];
                    dbeta += ddelta[j] * u[j];
                }
                out.dbeta[ti] = dbeta;
                // dS^ = dS - beta k^ ddelta^T;  dk^ -= beta S^ ddelta.
                for i in 0..dk {
                    let bk = beta * kr[i];
                    let mut acc = dkh[i];
                    for j in 0..dv {
                        let ix = i * dv + j;
                        dsh[ix] = ds[ix] - bk * ddelta[j];
                        acc -= beta * sh[ix] * ddelta[j];
                    }
                    dkh[i] = acc;
                }
                // S^ = exp(g) S_prev: dg = <dS^, S^>, dS_prev = exp(g) dS^.
                let mut dg = 0.0f32;
                for (&dsv, &shv) in dsh.iter().zip(sh.iter()) {
                    dg += dsv * shv;
                }
                out.dg[ti] = dg;
                for (dsv, &dshv) in ds.iter_mut().zip(dsh.iter()) {
                    *dsv = a * dshv;
                }
                // l2norm: y = x r  =>  dx = r (dy - y (y . dy)); q^ = scale y.
                let (rqv, rkv) = (rq[row], rk[row]);
                let mut yq = 0.0f32;
                let mut yk = 0.0f32;
                for i in 0..dk {
                    yq += qr[i] * (dqh[i] * g.scale);
                    yk += kr[i] * dkh[i];
                }
                for i in 0..dk {
                    out.dq[ti * dk + i] = (rqv * f64::from(dqh[i] * g.scale - qr[i] * yq)) as f32;
                    out.dk[ti * dk + i] = (rkv * f64::from(dkh[i] - kr[i] * yk)) as f32;
                }
            }
        }
        if let Some(p) = out.ds0 {
            p.copy_from_slice(&ds);
        }
        Ok(())
    })?;
    to_token_major(&g, &h_dq, grads.dq, g.dk);
    to_token_major(&g, &h_dk, grads.dk, g.dk);
    to_token_major(&g, &h_dv, grads.dv, g.dv);
    to_token_major(&g, &h_dg, grads.dg, 1);
    to_token_major(&g, &h_dbeta, grads.dbeta, 1);
    Ok(())
}
