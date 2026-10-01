//! The device thread: macOS with the `metal` feature.
//!
//! It creates the tessl runtime, owns every buffer by id, and runs one
//! [`Cmd`] at a time. Every op ends with one synchronize and a read of its
//! status words, so a failed check discards the outputs before they get ids.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Arc};
use std::thread;

use ojas_core::{sdpa_scale, OjasError, MUON_NS5_A, MUON_NS5_B, MUON_NS5_C, MUON_NS_EPS};
use tessl::dispatch::{self, set_f32, set_gpu_buf_offset, set_u32, Binder};
use tessl::gemm::{transpose_f32_into, GemmOperands};
use tessl::qwen35::{residual_add, Cols};
use tessl::runtime::GpuRuntime;
use tessl::tensor::{gpu_copy, GpuBuffer, Tensor as TT};
use tessl::DType;

use crate::link::{metal_err, rms_w_chunks, Arg, Cmd, Msg, NewBuf, Reply, Res, RopeMode};

const ST_IN: u32 = 0;
const ST_OUT: u32 = 1;
const ST_RANGE: usize = 2;
const ST_FIRST: usize = 3;
const ST_COUNT: usize = 4;
const ST_WORDS: usize = 8;
const ST_BYTES: usize = ST_WORDS * 4;
/// Tiled attention geometry, as `ATT_THREADS` (four simdgroups) and
/// `ATT_BQ` / `ATT_BK` (rows per threadgroup) in the Metal source.
const ATTN_THREADS: usize = 128;
const ATTN_ROWS: usize = 32;
/// Largest head dim the tiled attention kernels are compiled for.
const ATTN_MAX_HEAD_DIM: u32 = 128;

/// Every kernel `MetalBackend` dispatches from the crate's metallib. Start-up
/// fails if one is missing, not the first op that needs it.
const KERNELS: &[&str] = &[
    "ojas_status_init",
    "ojas_check_finite",
    "ojas_check_ids",
    "ojas_fill_u32",
    "ojas_adamw_check",
    "ojas_adamw_apply",
    "ojas_permute",
    "ojas_silu_fwd",
    "ojas_silu_bwd",
    "ojas_mul_fwd",
    "ojas_mul_bwd",
    "ojas_axpby",
    "ojas_div_scalar",
    "ojas_scale",
    "ojas_ns_denom",
    "ojas_vres_fwd",
    "ojas_vres_bwd",
    "ojas_vres_lambda",
    "ojas_rope",
    "ojas_embed_fwd",
    "ojas_embed_count",
    "ojas_scan_exclusive",
    "ojas_embed_place",
    "ojas_embed_gather",
    "ojas_reduce_partial",
    "ojas_reduce_final",
    "ojas_ce_fused",
    "ojas_ce_mean",
    "ojas_rms_fwd",
    "ojas_rms_bwd_rows",
    "ojas_rms_bwd_w_part",
    "ojas_rms_bwd_w_sum",
    "ojas_attn_fwd_d16",
    "ojas_attn_fwd_d32",
    "ojas_attn_fwd_d64",
    "ojas_attn_fwd_d128",
    "ojas_attn_bwd_stats_d16",
    "ojas_attn_bwd_stats_d32",
    "ojas_attn_bwd_stats_d64",
    "ojas_attn_bwd_dq_d16",
    "ojas_attn_bwd_dq_d32",
    "ojas_attn_bwd_dq_d64",
    "ojas_attn_bwd_dkv_d16",
    "ojas_attn_bwd_dkv_d32",
    "ojas_attn_bwd_dkv_d64",
    "ojas_attn_bwd_stats_d128",
    "ojas_attn_bwd_dq_d128",
    "ojas_attn_bwd_dkv_d128",
    "ojas_per_head_gate_fwd",
    "ojas_per_head_gate_bwd",
    "ojas_per_head_gate_dbias",
];

/// Start the device thread and wait until its runtime is up.
pub(crate) fn spawn() -> Res<(mpsc::Sender<Msg>, String)> {
    spawn_with(Worker::open)
}

/// As [`spawn`], with the opener the tests can fail. A failed or panicked
/// opener is joined before this returns, so the owner thread does not stay
/// in the process. A successful opener is detached: it runs until its channel
/// closes.
fn spawn_with(
    open: impl FnOnce() -> Res<Worker> + Send + 'static,
) -> Res<(mpsc::Sender<Msg>, String)> {
    let (tx, rx) = mpsc::channel::<Msg>();
    let (ready_tx, ready_rx) = mpsc::sync_channel::<Res<String>>(1);
    let handle = thread::Builder::new()
        .name("ojas-metal-device".to_string())
        .spawn(move || {
            let opened = catch_unwind(AssertUnwindSafe(open))
                .unwrap_or_else(|_| Err(metal_err("Metal runtime start-up panicked")));
            let mut worker = match opened {
                Ok(worker) => worker,
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(worker.rt.device_name()));
            drop(ready_tx);
            worker.serve(rx);
        })
        .map_err(|e| metal_err(format!("could not spawn the Metal device thread: {e}")))?;
    match ready_rx.recv() {
        Ok(Ok(name)) => {
            drop(handle);
            Ok((tx, name))
        }
        Ok(Err(err)) => {
            let _ = handle.join();
            Err(err)
        }
        Err(_) => {
            let _ = handle.join();
            Err(metal_err("Metal device thread exited during start-up"))
        }
    }
}

/// A contiguous f32 or u32 window of a buffer.
#[derive(Clone)]
struct V {
    buf: GpuBuffer,
    off: usize,
    n: usize,
}

fn bind(b: &mut Binder<'_>, v: &V, index: usize) {
    set_gpu_buf_offset(b, &v.buf, v.off, index);
}

/// The attention kernels' matrix-unit tensors index one `[T, D]` plane with
/// i32 extents.
fn attn_plane_fits(op: &'static str, t: u32, d: u32) -> Res<()> {
    if u64::from(t) * u64::from(d) > i32::MAX as u64 {
        return Err(OjasError::Unsupported {
            op,
            detail: format!("a [T, D] = [{t}, {d}] plane exceeds i32 indexing"),
        });
    }
    Ok(())
}

/// RMSNorm geometry, as `RMS_ROWS_PER_TG` (one simdgroup per row) and
/// `RMS_W_CHUNK` in the Metal source.
const RMS_THREADS: usize = 256;
const RMS_ROWS_PER_TG: usize = 8;

fn rms_groups(rows: u32) -> usize {
    (rows as usize).div_ceil(RMS_ROWS_PER_TG)
}

fn u32_of(n: usize) -> Res<u32> {
    u32::try_from(n).map_err(|_| OjasError::Unsupported {
        op: "MetalBackend",
        detail: format!("{n} exceeds the kernels' 32-bit indexing"),
    })
}

struct Worker {
    rt: Arc<GpuRuntime>,
    bufs: HashMap<u64, GpuBuffer>,
    next_id: u64,
    poisoned: bool,
}

impl Worker {
    fn open() -> Res<Self> {
        let rt = GpuRuntime::new().map_err(metal_err)?;
        rt.set_async_encode(true).map_err(metal_err)?;
        let lib = include_bytes!(concat!(env!("OUT_DIR"), "/ojas_per_head_gate.metallib"));
        if lib.is_empty() {
            return Err(metal_err("embedded ojas-metal metallib is empty"));
        }
        rt.add_metallib_bytes(lib).map_err(metal_err)?;
        for name in KERNELS {
            rt.pipeline(name).map_err(metal_err)?;
        }
        Ok(Self {
            rt,
            bufs: HashMap::new(),
            next_id: 1,
            poisoned: false,
        })
    }

    fn serve(&mut self, rx: mpsc::Receiver<Msg>) {
        for msg in rx {
            if let Cmd::Free { id } = msg.cmd {
                self.bufs.remove(&id);
                continue;
            }
            let out = if self.poisoned {
                Err(OjasError::Poisoned)
            } else {
                match catch_unwind(AssertUnwindSafe(|| self.run(msg.cmd))) {
                    Ok(out) => out,
                    Err(_) => {
                        self.poisoned = true;
                        Err(metal_err(
                            "the Metal device thread panicked; this backend is poisoned",
                        ))
                    }
                }
            };
            if let Some(reply) = msg.reply {
                let _ = reply.send(out);
            }
        }
    }

    // ------------------------------------------------------------ buffers --

    fn view(&self, a: Arg) -> Res<V> {
        let buf = self
            .bufs
            .get(&a.id)
            .ok_or_else(|| metal_err(format!("unknown device buffer {}", a.id)))?;
        let end =
            a.n.checked_mul(4)
                .and_then(|b| b.checked_add(a.off))
                .ok_or_else(|| metal_err("device window overflows"))?;
        if !a.off.is_multiple_of(4) || end > buf.nbytes() {
            return Err(metal_err(format!(
                "window [{}, {end}) does not fit device buffer {} of {} bytes",
                a.off,
                a.id,
                buf.nbytes()
            )));
        }
        Ok(V {
            buf: buf.clone(),
            off: a.off,
            n: a.n,
        })
    }

    fn alloc(&self, bytes: usize) -> Res<GpuBuffer> {
        self.rt.alloc_buffer(bytes).map_err(|e| {
            if e.contains("exceeds device limit") || e.contains("newBuffer failed") {
                OjasError::CapacityExceeded {
                    requested: bytes as u64,
                    cap: self.rt.memory_info().recommended_working_set,
                    live: 0,
                }
            } else {
                metal_err(e)
            }
        })
    }

    /// `n` four-byte elements, offset 0.
    fn fresh(&self, n: usize) -> Res<V> {
        let bytes = n
            .checked_mul(4)
            .ok_or_else(|| metal_err("allocation size overflows"))?;
        Ok(V {
            buf: self.alloc(bytes.max(4))?,
            off: 0,
            n,
        })
    }

    /// Give `outs` ids. Called only after the op's checks passed.
    fn keep(&mut self, outs: Vec<V>) -> Reply {
        let mut ids = Vec::with_capacity(outs.len());
        for v in outs {
            let id = self.next_id;
            self.next_id += 1;
            ids.push(NewBuf { id, bytes: v.n * 4 });
            self.bufs.insert(id, v.buf);
        }
        Reply::Bufs(ids)
    }

    fn tt(&self, v: &V, shape: &[usize]) -> Res<TT> {
        TT::from_buffer(&self.rt, v.buf.clone(), shape, DType::F32, v.off).map_err(metal_err)
    }

    /// A GEMM operand: tessl needs a 16-byte aligned offset, so a view that is
    /// not aligned is copied first.
    fn mat(&self, v: &V, r: usize, c: usize) -> Res<TT> {
        if v.off.is_multiple_of(16) {
            return self.tt(v, &[r, c]);
        }
        let dst = self.fresh(v.n)?;
        self.copy(v, &dst)?;
        self.tt(&dst, &[r, c])
    }

    fn copy(&self, src: &V, dst: &V) -> Res<()> {
        gpu_copy(&self.tt(src, &[src.n])?, &self.tt(dst, &[dst.n])?).map_err(metal_err)
    }

    // ----------------------------------------------------------- dispatch --

    fn k1(&self, name: &str, n: usize, f: impl FnOnce(&mut Binder<'_>)) -> Res<()> {
        let p = self.rt.pipeline(name).map_err(metal_err)?;
        dispatch::dispatch_1d(&self.rt, &p, n, f).map_err(metal_err)
    }

    fn k2(&self, name: &str, nx: usize, ny: usize, f: impl FnOnce(&mut Binder<'_>)) -> Res<()> {
        let p = self.rt.pipeline(name).map_err(metal_err)?;
        dispatch::dispatch_2d(&self.rt, &p, nx, ny, f).map_err(metal_err)
    }

    fn ktg(
        &self,
        name: &str,
        gx: usize,
        gy: usize,
        threads: usize,
        f: impl FnOnce(&mut Binder<'_>),
    ) -> Res<()> {
        let p = self.rt.pipeline(name).map_err(metal_err)?;
        dispatch::dispatch_2d_tg(&self.rt, &p, gx, gy, threads, f).map_err(metal_err)
    }

    fn status(&self) -> Res<GpuBuffer> {
        let st = self.alloc(ST_BYTES)?;
        self.k1("ojas_status_init", ST_WORDS, |b| {
            set_gpu_buf_offset(b, &st, 0, 0)
        })?;
        Ok(st)
    }

    /// Flag word `word` if any element of `v` is NaN or infinite.
    fn check(&self, st: &GpuBuffer, v: &V, word: u32) -> Res<()> {
        let n = u32_of(v.n)?;
        self.k1("ojas_check_finite", v.n, |b| {
            bind(b, v, 0);
            set_gpu_buf_offset(b, st, 0, 1);
            set_u32(b, n, 2);
            set_u32(b, word, 3);
        })
    }

    fn check_ids(
        &self,
        st: &GpuBuffer,
        ids: &V,
        limit: u32,
        ignore: Option<u32>,
        count_valid: bool,
    ) -> Res<()> {
        let n = u32_of(ids.n)?;
        self.k1("ojas_check_ids", ids.n, |b| {
            bind(b, ids, 0);
            set_gpu_buf_offset(b, st, 0, 1);
            set_u32(b, n, 2);
            set_u32(b, limit, 3);
            set_u32(b, u32::from(ignore.is_some()), 4);
            set_u32(b, ignore.unwrap_or(0), 5);
            set_u32(b, u32::from(count_valid), 6);
        })
    }

    /// Wait for the op and turn its status words into an error. Precedence
    /// follows the CPU reference: a non-finite input, then an index out of
    /// range, then (cross-entropy) no valid row, then a non-finite output.
    fn finish(&self, op: &'static str, st: &GpuBuffer, need_count: bool) -> Res<[u32; ST_WORDS]> {
        self.rt.synchronize().map_err(metal_err)?;
        let mut w = [0u32; ST_WORDS];
        {
            let map = st.try_contents_u32().map_err(metal_err)?;
            w.copy_from_slice(&map[..ST_WORDS]);
        }
        if w[ST_IN as usize] != 0 {
            return Err(OjasError::NonFinite { op });
        }
        if w[ST_RANGE] != 0 {
            return Err(OjasError::OutOfRange {
                op,
                detail: format!("index at position {} is out of range", w[ST_FIRST]),
            });
        }
        if need_count && w[ST_COUNT] == 0 {
            return Err(OjasError::NonFinite { op });
        }
        if w[ST_OUT as usize] != 0 {
            return Err(OjasError::NonFinite { op });
        }
        Ok(w)
    }

    fn axpby(&self, x: &V, y: &V, out: &V, alpha: f32, beta: f32) -> Res<()> {
        let n = u32_of(out.n)?;
        self.k1("ojas_axpby", out.n, |b| {
            bind(b, x, 0);
            bind(b, y, 1);
            bind(b, out, 2);
            set_u32(b, n, 3);
            set_f32(b, alpha, 4);
            set_f32(b, beta, 5);
        })
    }

    /// Reduce `src` into `out[slot]`. mode 0 sum, 1 max |x|, 2 sum of
    /// (x / scale)^2 with the scale read at `scale` (buffer, byte offset).
    fn reduce(
        &self,
        src: &V,
        mode: u32,
        scale: Option<(&GpuBuffer, usize)>,
        out: &GpuBuffer,
        slot: u32,
    ) -> Res<()> {
        let groups = src.n.div_ceil(4096).clamp(1, 1024);
        let chunk = src.n.div_ceil(groups);
        let part = self.fresh(groups)?;
        let n = u32_of(src.n)?;
        let chunk_u = u32_of(chunk)?;
        let groups_u = u32_of(groups)?;
        self.ktg("ojas_reduce_partial", groups, 1, 256, |b| {
            bind(b, src, 0);
            bind(b, &part, 1);
            set_u32(b, n, 2);
            set_u32(b, chunk_u, 3);
            set_u32(b, mode, 4);
            match scale {
                Some((buf, off)) => set_gpu_buf_offset(b, buf, off, 5),
                None => set_gpu_buf_offset(b, out, 0, 5),
            }
        })?;
        self.ktg("ojas_reduce_final", 1, 1, 256, |b| {
            bind(b, &part, 0);
            set_gpu_buf_offset(b, out, 0, 1);
            set_u32(b, groups_u, 2);
            set_u32(b, mode, 3);
            set_u32(b, slot, 4);
        })
    }

    fn run(&mut self, cmd: Cmd) -> Res<Reply> {
        match cmd {
            Cmd::Free { id } => {
                self.bufs.remove(&id);
                Ok(Reply::Done)
            }
            Cmd::Upload { bytes } => self.upload(&bytes),
            #[cfg(test)]
            Cmd::InjectPanic => {
                let _open_work = self.status()?;
                panic!("injected device-thread panic");
            }
            Cmd::Read { id, off, len } => self.read(id, off, len),
            Cmd::Permute {
                x,
                rank,
                oshape,
                istride,
            } => self.permute(x, rank, &oshape, &istride),
            Cmd::Embed {
                table,
                ids,
                vocab,
                dim,
            } => self.embed(table, ids, vocab, dim),
            Cmd::EmbedBwd {
                table,
                ids,
                grad,
                vocab,
                dim,
            } => self.embed_bwd(table, ids, grad, vocab, dim),
            Cmd::Linear {
                x,
                w,
                rows,
                kin,
                nout,
            } => self.linear(x, w, rows, kin, nout),
            Cmd::LinearBwd {
                x,
                w,
                gy,
                rows,
                kin,
                nout,
            } => self.linear_bwd(x, w, gy, rows, kin, nout),
            Cmd::Rms {
                x,
                w,
                rows,
                dim,
                eps,
            } => self.rms(x, w, rows, dim, eps),
            Cmd::RmsBwd {
                x,
                w,
                gy,
                rows,
                dim,
                eps,
            } => self.rms_bwd(x, w, gy, rows, dim, eps),
            Cmd::Rope {
                x,
                cos,
                sin,
                rows,
                dim,
                mode,
                backward,
            } => self.rope(x, cos, sin, rows, dim, mode, backward),
            Cmd::Sdpa { q, k, v, bh, t, d } => self.sdpa(q, k, v, bh, t, d),
            Cmd::SdpaBwd {
                q,
                k,
                v,
                gy,
                bh,
                t,
                d,
            } => self.sdpa_bwd(q, k, v, gy, bh, t, d),
            Cmd::Gate {
                x,
                w,
                b,
                attn,
                rows,
                din,
                heads,
                dh,
            } => self.gate(x, w, b, attn, None, rows, din, heads, dh),
            Cmd::GateBwd {
                x,
                w,
                b,
                attn,
                gy,
                rows,
                din,
                heads,
                dh,
            } => self.gate(x, w, b, attn, Some(gy), rows, din, heads, dh),
            Cmd::Vres { v, v0, lam } => self.vres(v, v0, lam, None),
            Cmd::VresBwd { v, v0, lam, gy } => self.vres(v, v0, lam, Some(gy)),
            Cmd::Silu { x } => self.silu(x, None),
            Cmd::SiluBwd { x, gy } => self.silu(x, Some(gy)),
            Cmd::Mul { a, b } => self.mul(a, b, None),
            Cmd::MulBwd { a, b, gy } => self.mul(a, b, Some(gy)),
            Cmd::Add { x, y } => self.add(x, y),
            Cmd::AddBwd { x, y, gy } => self.add_bwd(x, y, gy),
            Cmd::Ce {
                logits,
                targets,
                rows,
                vocab,
                ignore,
                grad,
            } => self.ce(logits, targets, rows, vocab, ignore, grad),
            Cmd::ClipNorm { grads } => self.clip_norm(&grads),
            Cmd::Scale { grads, scale } => self.scale(&grads, scale),
            Cmd::AdamW {
                p,
                g,
                m,
                v,
                scalars,
            } => self.adamw(p, g, m, v, &scalars),
            Cmd::Muon {
                p,
                g,
                m,
                rows,
                cols,
                momentum,
                nesterov,
                decay,
                alpha,
            } => self.muon(p, g, m, rows, cols, momentum, nesterov, decay, alpha),
        }
    }

    fn upload(&mut self, bytes: &[u8]) -> Res<Reply> {
        let buf = self.alloc(bytes.len())?;
        {
            let mut map = buf.try_contents_u8().map_err(metal_err)?;
            map[..bytes.len()].copy_from_slice(bytes);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.bufs.insert(id, buf);
        Ok(Reply::Bufs(vec![NewBuf {
            id,
            bytes: bytes.len(),
        }]))
    }

    fn read(&self, id: u64, off: usize, len: usize) -> Res<Reply> {
        let buf = self
            .bufs
            .get(&id)
            .ok_or_else(|| metal_err(format!("unknown device buffer {id}")))?;
        let end = off
            .checked_add(len)
            .filter(|end| *end <= buf.nbytes())
            .ok_or_else(|| metal_err("read past the end of a device buffer"))?;
        self.rt.synchronize().map_err(metal_err)?;
        let map = buf.try_contents_u8().map_err(metal_err)?;
        Ok(Reply::Bytes(map[off..end].to_vec()))
    }

    /// A bit copy. A NaN or infinity in the input is refused, as the CPU
    /// reference refuses one; the output holds only copied finite bits, so
    /// it needs no check of its own.
    fn permute(&mut self, x: Arg, rank: u32, oshape: &[u32], istride: &[u32]) -> Res<Reply> {
        const OP: &str = "permute";
        let xv = self.view(x)?;
        let n = u32_of(xv.n)?;
        let y = self.fresh(xv.n)?;
        let shape_bytes: Vec<u8> = oshape.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let stride_bytes: Vec<u8> = istride.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let st = self.status()?;
        self.check(&st, &xv, ST_IN)?;
        self.k1("ojas_permute", xv.n, |b| {
            bind(b, &xv, 0);
            bind(b, &y, 1);
            set_u32(b, n, 2);
            set_u32(b, rank, 3);
            b.bind_bytes(&shape_bytes, 4);
            b.bind_bytes(&stride_bytes, 5);
        })?;
        self.finish(OP, &st, false)?;
        Ok(self.keep(vec![y]))
    }

    fn embed(&mut self, table: Arg, ids: Arg, vocab: u32, dim: u32) -> Res<Reply> {
        const OP: &str = "embedding_forward";
        let t = self.view(table)?;
        let ids = self.view(ids)?;
        let n = u32_of(ids.n)?;
        let out = self.fresh(ids.n * dim as usize)?;
        let st = self.status()?;
        self.check(&st, &t, ST_IN)?;
        self.check_ids(&st, &ids, vocab, None, false)?;
        self.k2("ojas_embed_fwd", dim as usize, ids.n, |b| {
            bind(b, &t, 0);
            bind(b, &ids, 1);
            bind(b, &out, 2);
            set_u32(b, n, 3);
            set_u32(b, dim, 4);
            set_u32(b, vocab, 5);
        })?;
        self.check(&st, &out, ST_OUT)?;
        self.finish(OP, &st, false)?;
        Ok(self.keep(vec![out]))
    }

    /// Deterministic scatter-add: count rows per id, prefix-sum the counts,
    /// place each token in ascending order within its id, then gather. Each
    /// table row is a left-to-right f32 sum from 0, as on the CPU.
    fn embed_bwd(&mut self, table: Arg, ids: Arg, grad: Arg, vocab: u32, dim: u32) -> Res<Reply> {
        const OP: &str = "embedding_backward";
        let t = self.view(table)?;
        let ids = self.view(ids)?;
        let g = self.view(grad)?;
        let n = u32_of(ids.n)?;
        let v = vocab as usize;
        let counts = self.fresh(v)?;
        let starts = self.fresh(v)?;
        let pos = self.fresh(ids.n)?;
        let out = self.fresh(v * dim as usize)?;
        let st = self.status()?;
        self.check(&st, &t, ST_IN)?;
        self.check(&st, &g, ST_IN)?;
        self.check_ids(&st, &ids, vocab, None, false)?;
        self.k1("ojas_fill_u32", v, |b| {
            bind(b, &counts, 0);
            set_u32(b, vocab, 1);
            set_u32(b, 0, 2);
        })?;
        self.k1("ojas_embed_count", ids.n, |b| {
            bind(b, &ids, 0);
            bind(b, &counts, 1);
            set_u32(b, n, 2);
            set_u32(b, vocab, 3);
        })?;
        self.ktg("ojas_scan_exclusive", 1, 1, 1024, |b| {
            bind(b, &counts, 0);
            bind(b, &starts, 1);
            set_u32(b, vocab, 2);
        })?;
        self.ktg("ojas_embed_place", ids.n.div_ceil(256), 1, 256, |b| {
            bind(b, &ids, 0);
            bind(b, &starts, 1);
            bind(b, &pos, 2);
            set_u32(b, n, 3);
            set_u32(b, vocab, 4);
        })?;
        self.k2("ojas_embed_gather", dim as usize, v, |b| {
            bind(b, &g, 0);
            bind(b, &starts, 1);
            bind(b, &counts, 2);
            bind(b, &pos, 3);
            bind(b, &out, 4);
            set_u32(b, vocab, 5);
            set_u32(b, dim, 6);
        })?;
        self.check(&st, &out, ST_OUT)?;
        self.finish(OP, &st, false)?;
        Ok(self.keep(vec![out]))
    }

    fn linear(&mut self, x: Arg, w: Arg, rows: usize, kin: usize, nout: usize) -> Res<Reply> {
        const OP: &str = "linear_forward";
        let xv = self.view(x)?;
        let wv = self.view(w)?;
        let y = self.fresh(rows * nout)?;
        let st = self.status()?;
        self.check(&st, &xv, ST_IN)?;
        self.check(&st, &wv, ST_IN)?;
        GemmOperands::ExactF32
            .nt(
                &self.mat(&xv, rows, kin)?,
                &self.mat(&wv, nout, kin)?,
                &self.tt(&y, &[rows, nout])?,
            )
            .map_err(metal_err)?;
        self.check(&st, &y, ST_OUT)?;
        self.finish(OP, &st, false)?;
        Ok(self.keep(vec![y]))
    }

    fn linear_bwd(
        &mut self,
        x: Arg,
        w: Arg,
        gy: Arg,
        rows: usize,
        kin: usize,
        nout: usize,
    ) -> Res<Reply> {
        const OP: &str = "linear_backward";
        let xv = self.view(x)?;
        let wv = self.view(w)?;
        let gv = self.view(gy)?;
        let gx = self.fresh(rows * kin)?;
        let gw = self.fresh(nout * kin)?;
        let st = self.status()?;
        for v in [&xv, &wv, &gv] {
            self.check(&st, v, ST_IN)?;
        }
        let x_t = self.mat(&xv, rows, kin)?;
        let w_t = self.mat(&wv, nout, kin)?;
        let g_t = self.mat(&gv, rows, nout)?;
        GemmOperands::ExactF32
            .nn(&g_t, &w_t, &self.tt(&gx, &[rows, kin])?)
            .map_err(metal_err)?;
        GemmOperands::ExactF32
            .tn(&g_t, &x_t, &self.tt(&gw, &[nout, kin])?)
            .map_err(metal_err)?;
        self.check(&st, &gx, ST_OUT)?;
        self.check(&st, &gw, ST_OUT)?;
        self.finish(OP, &st, false)?;
        Ok(self.keep(vec![gx, gw]))
    }

    fn rms(&mut self, x: Arg, w: Arg, rows: u32, dim: u32, eps: f32) -> Res<Reply> {
        const OP: &str = "rms_norm_forward";
        let xv = self.view(x)?;
        let wv = self.view(w)?;
        let y = self.fresh(xv.n)?;
        let st = self.status()?;
        self.check(&st, &xv, ST_IN)?;
        self.check(&st, &wv, ST_IN)?;
        self.ktg("ojas_rms_fwd", rms_groups(rows), 1, RMS_THREADS, |b| {
            bind(b, &xv, 0);
            bind(b, &wv, 1);
            bind(b, &y, 2);
            set_gpu_buf_offset(b, &st, 0, 3);
            set_u32(b, rows, 4);
            set_u32(b, dim, 5);
            set_f32(b, eps, 6);
        })?;
        self.check(&st, &y, ST_OUT)?;
        self.finish(OP, &st, false)?;
        Ok(self.keep(vec![y]))
    }

    fn rms_bwd(&mut self, x: Arg, w: Arg, gy: Arg, rows: u32, dim: u32, eps: f32) -> Res<Reply> {
        const OP: &str = "rms_norm_backward";
        let xv = self.view(x)?;
        let wv = self.view(w)?;
        let gv = self.view(gy)?;
        let gx = self.fresh(xv.n)?;
        let gw = self.fresh(dim as usize)?;
        let rstd = self.fresh(rows as usize)?;
        let chunks = rms_w_chunks(rows);
        let part = self.fresh(chunks as usize * dim as usize)?;
        let st = self.status()?;
        for v in [&xv, &wv, &gv] {
            self.check(&st, v, ST_IN)?;
        }
        self.ktg("ojas_rms_bwd_rows", rms_groups(rows), 1, RMS_THREADS, |b| {
            bind(b, &xv, 0);
            bind(b, &wv, 1);
            bind(b, &gv, 2);
            bind(b, &rstd, 3);
            bind(b, &gx, 4);
            set_gpu_buf_offset(b, &st, 0, 5);
            set_u32(b, rows, 6);
            set_u32(b, dim, 7);
            set_f32(b, eps, 8);
        })?;
        self.k2("ojas_rms_bwd_w_part", dim as usize, chunks as usize, |b| {
            bind(b, &xv, 0);
            bind(b, &gv, 1);
            bind(b, &rstd, 2);
            bind(b, &part, 3);
            set_u32(b, rows, 4);
            set_u32(b, dim, 5);
        })?;
        self.k1("ojas_rms_bwd_w_sum", dim as usize, |b| {
            bind(b, &part, 0);
            bind(b, &gw, 1);
            set_u32(b, chunks, 2);
            set_u32(b, dim, 3);
        })?;
        self.check(&st, &gx, ST_OUT)?;
        self.check(&st, &gw, ST_OUT)?;
        self.finish(OP, &st, false)?;
        Ok(self.keep(vec![gx, gw]))
    }

    #[allow(clippy::too_many_arguments)]
    fn rope(
        &mut self,
        x: Arg,
        cos: Arg,
        sin: Arg,
        rows: u32,
        dim: u32,
        mode: RopeMode,
        backward: bool,
    ) -> Res<Reply> {
        let op = if backward {
            "rope_half_split_backward"
        } else {
            "rope_half_split_forward"
        };
        let xv = self.view(x)?;
        let cv = self.view(cos)?;
        let sv = self.view(sin)?;
        let y = self.fresh(xv.n)?;
        let (mode_u, time, heads) = match mode {
            RopeMode::Same => (0u32, 1u32, 1u32),
            RopeMode::TimeDim { time, heads } => (1, time, heads),
        };
        let st = self.status()?;
        for v in [&xv, &cv, &sv] {
            self.check(&st, v, ST_IN)?;
        }
        self.k2("ojas_rope", (dim / 2) as usize, rows as usize, |b| {
            bind(b, &xv, 0);
            bind(b, &cv, 1);
            bind(b, &sv, 2);
            bind(b, &y, 3);
            set_u32(b, rows, 4);
            set_u32(b, dim, 5);
            set_u32(b, mode_u, 6);
            set_u32(b, time, 7);
            set_u32(b, heads, 8);
            set_u32(b, u32::from(backward), 9);
        })?;
        self.check(&st, &y, ST_OUT)?;
        self.finish(op, &st, false)?;
        Ok(self.keep(vec![y]))
    }

    /// Compiled width of the tiled attention kernels (`ojas_attn_*_d*`).
    /// Their threadgroup memory does not grow with the head dim, so they are
    /// built up to 128; `ojas_core::METAL_MAX_HEAD_DIM` may refuse less
    /// before a call gets here.
    fn attn_width(d: u32) -> Res<u32> {
        match d {
            1..=16 => Ok(16),
            17..=32 => Ok(32),
            33..=64 => Ok(64),
            65..=ATTN_MAX_HEAD_DIM => Ok(128),
            _ => Err(OjasError::UnsupportedHeadDim {
                head_dim: d,
                limit: ATTN_MAX_HEAD_DIM,
            }),
        }
    }

    fn sdpa(&mut self, q: Arg, k: Arg, v: Arg, bh: u32, t: u32, d: u32) -> Res<Reply> {
        const OP: &str = "causal_sdpa_forward";
        let width = Self::attn_width(d)?;
        let scale = sdpa_scale(d)?;
        attn_plane_fits(OP, t, d)?;
        let (qv, kv, vv) = (self.view(q)?, self.view(k)?, self.view(v)?);
        let o = self.fresh(qv.n)?;
        let st = self.status()?;
        for x in [&qv, &kv, &vv] {
            self.check(&st, x, ST_IN)?;
        }
        let name = format!("ojas_attn_fwd_d{width}");
        let groups = (t as usize).div_ceil(ATTN_ROWS);
        self.ktg(&name, groups, bh as usize, ATTN_THREADS, |b| {
            bind(b, &qv, 0);
            bind(b, &kv, 1);
            bind(b, &vv, 2);
            bind(b, &o, 3);
            set_gpu_buf_offset(b, &st, 0, 4);
            set_u32(b, t, 5);
            set_u32(b, d, 6);
            set_f32(b, scale, 7);
        })?;
        self.check(&st, &o, ST_OUT)?;
        self.finish(OP, &st, false)?;
        Ok(self.keep(vec![o]))
    }

    /// The tiled backward in `ojas_backend.metal`: `stats` writes each query
    /// row's log-sum-exp and Dr, `dq` walks key blocks per query block, and
    /// `dkv` walks query blocks per key block. Three dispatches, no T x T
    /// buffer, one synchronize in [`Self::finish`].
    #[allow(clippy::too_many_arguments)]
    fn sdpa_bwd(&mut self, q: Arg, k: Arg, v: Arg, gy: Arg, bh: u32, t: u32, d: u32) -> Res<Reply> {
        const OP: &str = "causal_sdpa_backward";
        let width = Self::attn_width(d)?;
        let scale = sdpa_scale(d)?;
        attn_plane_fits(OP, t, d)?;
        let (qv, kv, vv, gv) = (self.view(q)?, self.view(k)?, self.view(v)?, self.view(gy)?);
        let rows = bh as usize * t as usize;
        let lse = self.fresh(rows)?;
        let dvec = self.fresh(rows)?;
        let dq = self.fresh(qv.n)?;
        let dk = self.fresh(qv.n)?;
        let dv = self.fresh(qv.n)?;
        let st = self.status()?;
        for x in [&qv, &kv, &vv, &gv] {
            self.check(&st, x, ST_IN)?;
        }
        let groups = (t as usize).div_ceil(ATTN_ROWS);
        let planes = bh as usize;
        let inputs = |b: &mut Binder<'_>| {
            bind(b, &qv, 0);
            bind(b, &kv, 1);
            bind(b, &vv, 2);
            bind(b, &gv, 3);
        };
        self.ktg(
            &format!("ojas_attn_bwd_stats_d{width}"),
            groups,
            planes,
            ATTN_THREADS,
            |b| {
                inputs(b);
                bind(b, &lse, 4);
                bind(b, &dvec, 5);
                set_gpu_buf_offset(b, &st, 0, 6);
                set_u32(b, t, 7);
                set_u32(b, d, 8);
                set_f32(b, scale, 9);
            },
        )?;
        self.ktg(
            &format!("ojas_attn_bwd_dq_d{width}"),
            groups,
            planes,
            ATTN_THREADS,
            |b| {
                inputs(b);
                bind(b, &lse, 4);
                bind(b, &dvec, 5);
                bind(b, &dq, 6);
                set_u32(b, t, 7);
                set_u32(b, d, 8);
                set_f32(b, scale, 9);
            },
        )?;
        self.ktg(
            &format!("ojas_attn_bwd_dkv_d{width}"),
            groups,
            planes,
            ATTN_THREADS,
            |b| {
                inputs(b);
                bind(b, &lse, 4);
                bind(b, &dvec, 5);
                bind(b, &dk, 6);
                bind(b, &dv, 7);
                set_u32(b, t, 8);
                set_u32(b, d, 9);
                set_f32(b, scale, 10);
            },
        )?;
        for x in [&dq, &dk, &dv] {
            self.check(&st, x, ST_OUT)?;
        }
        self.finish(OP, &st, false)?;
        Ok(self.keep(vec![dq, dk, dv]))
    }

    /// Forward when `gy` is `None`. `pre = x @ W^T` is a tessl GEMM; the
    /// sigmoid, its backward and the bias sum are this crate's gate kernels.
    #[allow(clippy::too_many_arguments)]
    fn gate(
        &mut self,
        x: Arg,
        w: Arg,
        bias: Arg,
        attn: Arg,
        gy: Option<Arg>,
        rows: u32,
        din: u32,
        heads: u32,
        dh: u32,
    ) -> Res<Reply> {
        let op = if gy.is_some() {
            "per_head_sigmoid_gate_backward"
        } else {
            "per_head_sigmoid_gate_forward"
        };
        let (xv, wv, bv, av) = (
            self.view(x)?,
            self.view(w)?,
            self.view(bias)?,
            self.view(attn)?,
        );
        let gv = gy.map(|g| self.view(g)).transpose()?;
        let (r, di, h) = (rows as usize, din as usize, heads as usize);
        let units = u32_of(r * h)?;
        let plane = u32_of(av.n)?;
        let pre = self.fresh(r * h)?;
        let st = self.status()?;
        for v in [&xv, &wv, &bv, &av] {
            self.check(&st, v, ST_IN)?;
        }
        if let Some(g) = &gv {
            self.check(&st, g, ST_IN)?;
        }
        let x_t = self.mat(&xv, r, di)?;
        let w_t = self.mat(&wv, h, di)?;
        GemmOperands::ExactF32
            .nt(&x_t, &w_t, &self.tt(&pre, &[r, h])?)
            .map_err(metal_err)?;
        self.check(&st, &pre, ST_OUT)?;
        let Some(g) = gv else {
            let out = self.fresh(av.n)?;
            self.k2("ojas_per_head_gate_fwd", dh as usize, r * h, |b| {
                bind(b, &av, 0);
                bind(b, &pre, 1);
                bind(b, &bv, 2);
                bind(b, &out, 3);
                set_u32(b, rows, 4);
                set_u32(b, heads, 5);
                set_u32(b, dh, 6);
                set_u32(b, plane, 7);
                set_u32(b, units, 8);
                set_u32(b, heads, 9);
                set_u32(b, plane, 10);
            })?;
            self.check(&st, &out, ST_OUT)?;
            self.finish(op, &st, false)?;
            return Ok(self.keep(vec![out]));
        };
        let d_attn = self.fresh(av.n)?;
        let d_pre = self.fresh(r * h)?;
        let d_bias = self.fresh(h)?;
        let gx = self.fresh(r * di)?;
        let gw = self.fresh(h * di)?;
        self.k1("ojas_per_head_gate_bwd", r * h, |b| {
            bind(b, &av, 0);
            bind(b, &pre, 1);
            bind(b, &bv, 2);
            bind(b, &g, 3);
            bind(b, &d_attn, 4);
            bind(b, &d_pre, 5);
            set_u32(b, rows, 6);
            set_u32(b, heads, 7);
            set_u32(b, dh, 8);
            set_u32(b, plane, 9);
            set_u32(b, units, 10);
            set_u32(b, heads, 11);
        })?;
        self.k1("ojas_per_head_gate_dbias", h, |b| {
            bind(b, &d_pre, 0);
            bind(b, &d_bias, 1);
            set_u32(b, rows, 2);
            set_u32(b, heads, 3);
            set_u32(b, units, 4);
            set_u32(b, heads, 5);
        })?;
        let dp_t = self.tt(&d_pre, &[r, h])?;
        GemmOperands::ExactF32
            .nn(&dp_t, &w_t, &self.tt(&gx, &[r, di])?)
            .map_err(metal_err)?;
        GemmOperands::ExactF32
            .tn(&dp_t, &x_t, &self.tt(&gw, &[h, di])?)
            .map_err(metal_err)?;
        for v in [&gx, &gw, &d_bias, &d_attn] {
            self.check(&st, v, ST_OUT)?;
        }
        self.finish(op, &st, false)?;
        Ok(self.keep(vec![gx, gw, d_bias, d_attn]))
    }

    fn vres(&mut self, v: Arg, v0: Arg, lam: Arg, gy: Option<Arg>) -> Res<Reply> {
        let op = if gy.is_some() {
            "value_residual_blend_backward"
        } else {
            "value_residual_blend_forward"
        };
        let (vv, v0v, lv) = (self.view(v)?, self.view(v0)?, self.view(lam)?);
        let gv = gy.map(|g| self.view(g)).transpose()?;
        let n = u32_of(vv.n)?;
        let st = self.status()?;
        for x in [&vv, &v0v, &lv] {
            self.check(&st, x, ST_IN)?;
        }
        let Some(g) = gv else {
            let y = self.fresh(vv.n)?;
            self.k1("ojas_vres_fwd", vv.n, |b| {
                bind(b, &vv, 0);
                bind(b, &v0v, 1);
                bind(b, &lv, 2);
                bind(b, &y, 3);
                set_u32(b, n, 4);
            })?;
            self.check(&st, &y, ST_OUT)?;
            self.finish(op, &st, false)?;
            return Ok(self.keep(vec![y]));
        };
        self.check(&st, &g, ST_IN)?;
        let gvv = self.fresh(vv.n)?;
        let gv0 = self.fresh(vv.n)?;
        let terms = self.fresh(vv.n)?;
        let sum = self.fresh(1)?;
        let glam = self.fresh(1)?;
        self.k1("ojas_vres_bwd", vv.n, |b| {
            bind(b, &vv, 0);
            bind(b, &v0v, 1);
            bind(b, &lv, 2);
            bind(b, &g, 3);
            bind(b, &gvv, 4);
            bind(b, &gv0, 5);
            bind(b, &terms, 6);
            set_u32(b, n, 7);
        })?;
        self.reduce(&terms, 0, None, &sum.buf, 0)?;
        self.k1("ojas_vres_lambda", 1, |b| {
            bind(b, &sum, 0);
            bind(b, &lv, 1);
            bind(b, &glam, 2);
        })?;
        for x in [&gvv, &gv0, &glam] {
            self.check(&st, x, ST_OUT)?;
        }
        self.finish(op, &st, false)?;
        Ok(self.keep(vec![gvv, gv0, glam]))
    }

    fn silu(&mut self, x: Arg, gy: Option<Arg>) -> Res<Reply> {
        let op = if gy.is_some() {
            "silu_backward"
        } else {
            "silu_forward"
        };
        let xv = self.view(x)?;
        let gv = gy.map(|g| self.view(g)).transpose()?;
        let n = u32_of(xv.n)?;
        let y = self.fresh(xv.n)?;
        let st = self.status()?;
        self.check(&st, &xv, ST_IN)?;
        match &gv {
            None => self.k1("ojas_silu_fwd", xv.n, |b| {
                bind(b, &xv, 0);
                bind(b, &y, 1);
                set_u32(b, n, 2);
            })?,
            Some(g) => {
                self.check(&st, g, ST_IN)?;
                self.k1("ojas_silu_bwd", xv.n, |b| {
                    bind(b, &xv, 0);
                    bind(b, g, 1);
                    bind(b, &y, 2);
                    set_u32(b, n, 3);
                })?
            }
        }
        self.check(&st, &y, ST_OUT)?;
        self.finish(op, &st, false)?;
        Ok(self.keep(vec![y]))
    }

    fn mul(&mut self, a: Arg, b_arg: Arg, gy: Option<Arg>) -> Res<Reply> {
        let op = if gy.is_some() {
            "mul_backward"
        } else {
            "mul_forward"
        };
        let (av, bv) = (self.view(a)?, self.view(b_arg)?);
        let gv = gy.map(|g| self.view(g)).transpose()?;
        let n = u32_of(av.n)?;
        let st = self.status()?;
        self.check(&st, &av, ST_IN)?;
        self.check(&st, &bv, ST_IN)?;
        let Some(g) = gv else {
            let y = self.fresh(av.n)?;
            self.k1("ojas_mul_fwd", av.n, |b| {
                bind(b, &av, 0);
                bind(b, &bv, 1);
                bind(b, &y, 2);
                set_u32(b, n, 3);
            })?;
            self.check(&st, &y, ST_OUT)?;
            self.finish(op, &st, false)?;
            return Ok(self.keep(vec![y]));
        };
        self.check(&st, &g, ST_IN)?;
        let ga = self.fresh(av.n)?;
        let gb = self.fresh(av.n)?;
        self.k1("ojas_mul_bwd", av.n, |b| {
            bind(b, &av, 0);
            bind(b, &bv, 1);
            bind(b, &g, 2);
            bind(b, &ga, 3);
            bind(b, &gb, 4);
            set_u32(b, n, 5);
        })?;
        self.check(&st, &ga, ST_OUT)?;
        self.check(&st, &gb, ST_OUT)?;
        self.finish(op, &st, false)?;
        Ok(self.keep(vec![ga, gb]))
    }

    /// `out = x`, then tessl's exact-f32 `residual_add` adds `y` into it.
    fn add(&mut self, x: Arg, y: Arg) -> Res<Reply> {
        const OP: &str = "residual_add_forward";
        let (xv, yv) = (self.view(x)?, self.view(y)?);
        let n = u32_of(xv.n)?;
        let y_off = u32_of(yv.off / 4)?;
        let ld = y_off
            .checked_add(n)
            .ok_or_else(|| metal_err("residual add window overflows u32"))?;
        let out = self.fresh(xv.n)?;
        let st = self.status()?;
        self.check(&st, &xv, ST_IN)?;
        self.check(&st, &yv, ST_IN)?;
        self.copy(&xv, &out)?;
        residual_add(
            &self.rt,
            Cols {
                buf: &yv.buf,
                ld,
                off: y_off,
            },
            Cols::dense(&out.buf, n),
            1,
            n,
        )
        .map_err(metal_err)?;
        self.check(&st, &out, ST_OUT)?;
        self.finish(OP, &st, false)?;
        Ok(self.keep(vec![out]))
    }

    fn add_bwd(&mut self, x: Arg, y: Arg, gy: Arg) -> Res<Reply> {
        const OP: &str = "residual_add_backward";
        let (xv, yv, gv) = (self.view(x)?, self.view(y)?, self.view(gy)?);
        let gx = self.fresh(gv.n)?;
        let gyy = self.fresh(gv.n)?;
        let st = self.status()?;
        for v in [&xv, &yv, &gv] {
            self.check(&st, v, ST_IN)?;
        }
        self.copy(&gv, &gx)?;
        self.copy(&gv, &gyy)?;
        self.finish(OP, &st, false)?;
        Ok(self.keep(vec![gx, gyy]))
    }

    fn ce(
        &mut self,
        logits: Arg,
        targets: Arg,
        rows: u32,
        vocab: u32,
        ignore: Option<u32>,
        grad: bool,
    ) -> Res<Reply> {
        let op = if grad {
            "cross_entropy_mean_backward"
        } else {
            "cross_entropy_mean_forward"
        };
        let lv = self.view(logits)?;
        let tv = self.view(targets)?;
        let loss_rows = self.fresh(rows as usize)?;
        let gout = if grad { Some(self.fresh(lv.n)?) } else { None };
        let st = self.status()?;
        // The logits' finite check and the gradient's are inside
        // `ojas_ce_fused` (every element of both is still checked).
        self.check_ids(&st, &tv, vocab, ignore, true)?;
        self.ktg("ojas_ce_fused", rows as usize, 1, 256, |b| {
            bind(b, &lv, 0);
            bind(b, &tv, 1);
            set_gpu_buf_offset(b, &st, 0, 2);
            bind(b, &loss_rows, 3);
            bind(b, gout.as_ref().unwrap_or(&loss_rows), 4);
            set_u32(b, rows, 5);
            set_u32(b, vocab, 6);
            set_u32(b, u32::from(ignore.is_some()), 7);
            set_u32(b, ignore.unwrap_or(0), 8);
            set_u32(b, u32::from(grad), 9);
        })?;
        let out = match gout {
            Some(g) => {
                self.finish(op, &st, true)?;
                return Ok(self.keep(vec![g]));
            }
            None => {
                let sum = self.fresh(1)?;
                let loss = self.fresh(1)?;
                self.reduce(&loss_rows, 0, None, &sum.buf, 0)?;
                self.k1("ojas_ce_mean", 1, |b| {
                    bind(b, &sum, 0);
                    set_gpu_buf_offset(b, &st, 0, 1);
                    bind(b, &loss, 2);
                })?;
                loss
            }
        };
        self.check(&st, &out, ST_OUT)?;
        self.finish(op, &st, true)?;
        Ok(self.keep(vec![out]))
    }

    /// Per gradient: max |g| and sum (g / max |g|)^2 on the device; the host
    /// combines them in f64, so an f32 sum of squares cannot overflow.
    fn clip_norm(&mut self, grads: &[Arg]) -> Res<Reply> {
        const OP: &str = "clip_grad_norm";
        let views = grads
            .iter()
            .map(|a| self.view(*a))
            .collect::<Res<Vec<_>>>()?;
        let stats = self.fresh(2 * views.len())?;
        let st = self.status()?;
        for (i, v) in views.iter().enumerate() {
            let slot = u32_of(2 * i)?;
            self.check(&st, v, ST_IN)?;
            self.reduce(v, 1, None, &stats.buf, slot)?;
            self.reduce(v, 2, Some((&stats.buf, 2 * i * 4)), &stats.buf, slot + 1)?;
        }
        self.finish(OP, &st, false)?;
        let map = stats.buf.try_contents_f32().map_err(metal_err)?;
        let mut sum_sq = 0.0f64;
        for i in 0..views.len() {
            let amax = f64::from(map[2 * i]);
            sum_sq += amax * amax * f64::from(map[2 * i + 1]);
        }
        Ok(Reply::Norm(sum_sq.sqrt() as f32))
    }

    fn scale(&mut self, grads: &[Arg], scale: f32) -> Res<Reply> {
        const OP: &str = "clip_grad_norm";
        let views = grads
            .iter()
            .map(|a| self.view(*a))
            .collect::<Res<Vec<_>>>()?;
        let st = self.status()?;
        for v in &views {
            let n = u32_of(v.n)?;
            self.k1("ojas_scale", v.n, |b| {
                bind(b, v, 0);
                set_u32(b, n, 1);
                set_f32(b, scale, 2);
            })?;
            self.check(&st, v, ST_OUT)?;
        }
        self.finish(OP, &st, false)?;
        Ok(Reply::Done)
    }

    /// `ojas_adamw_check` decides the whole step (inputs and every new p, m,
    /// v finite), then `ojas_adamw_apply`, in the same command buffer, writes
    /// in place only if no status word is set. One wait, no copies.
    fn adamw(&mut self, p: Arg, g: Arg, m: Arg, v: Arg, scalars: &[f32; 7]) -> Res<Reply> {
        const OP: &str = "adamw_step";
        let (pv, gv, mv, vv) = (self.view(p)?, self.view(g)?, self.view(m)?, self.view(v)?);
        let n = pv.n;
        let n_u = u32_of(n)?;
        let bytes: Vec<u8> = scalars.iter().flat_map(|x| x.to_ne_bytes()).collect();
        let st = self.status()?;
        for name in ["ojas_adamw_check", "ojas_adamw_apply"] {
            self.k1(name, n, |b| {
                bind(b, &pv, 0);
                bind(b, &gv, 1);
                bind(b, &mv, 2);
                bind(b, &vv, 3);
                set_gpu_buf_offset(b, &st, 0, 4);
                set_u32(b, n_u, 5);
                b.bind_bytes(&bytes, 6);
            })?;
        }
        self.finish(OP, &st, false)?;
        Ok(Reply::Done)
    }

    /// Momentum, Nesterov, five Newton-Schulz iterations with tessl GEMMs,
    /// then the update. Written back only if every stage is finite.
    #[allow(clippy::too_many_arguments)]
    fn muon(
        &mut self,
        p: Arg,
        g: Arg,
        m: Arg,
        rows: usize,
        cols: usize,
        momentum: f32,
        nesterov: bool,
        decay: f32,
        alpha: f32,
    ) -> Res<Reply> {
        const OP: &str = "muon_ns5_step";
        let (pv, gv, mv) = (self.view(p)?, self.view(g)?, self.view(m)?);
        let n = pv.n;
        let transposed = rows > cols;
        let (r, c) = if transposed {
            (cols, rows)
        } else {
            (rows, cols)
        };
        let st = self.status()?;
        for x in [&pv, &gv, &mv] {
            self.check(&st, x, ST_IN)?;
        }
        let buf = self.fresh(n)?;
        self.axpby(&mv, &gv, &buf, momentum, 1.0)?;
        self.check(&st, &buf, ST_OUT)?;
        let update = if nesterov {
            let u = self.fresh(n)?;
            self.axpby(&gv, &buf, &u, 1.0, momentum)?;
            u
        } else {
            buf.clone()
        };
        let x = self.fresh(n)?;
        if transposed {
            transpose_f32_into(&self.tt(&update, &[rows, cols])?, &self.tt(&x, &[r, c])?)
                .map_err(metal_err)?;
        } else {
            self.copy(&update, &x)?;
        }
        let stats = self.fresh(3)?;
        self.reduce(&x, 1, None, &stats.buf, 0)?;
        self.reduce(&x, 2, Some((&stats.buf, 0)), &stats.buf, 1)?;
        self.k1("ojas_ns_denom", 1, |b| {
            bind(b, &stats, 0);
            set_gpu_buf_offset(b, &st, 0, 1);
            set_f32(b, MUON_NS_EPS as f32, 2);
        })?;
        let n_u = u32_of(n)?;
        self.k1("ojas_div_scalar", n, |b| {
            bind(b, &x, 0);
            bind(b, &x, 1);
            set_u32(b, n_u, 2);
            bind(b, &stats, 3);
            set_u32(b, 2, 4);
        })?;
        let a = self.fresh(r * r)?;
        let a2 = self.fresh(r * r)?;
        let bm = self.fresh(r * r)?;
        let bx = self.fresh(n)?;
        let x_t = self.tt(&x, &[r, c])?;
        let a_t = self.tt(&a, &[r, r])?;
        let a2_t = self.tt(&a2, &[r, r])?;
        let bm_t = self.tt(&bm, &[r, r])?;
        let bx_t = self.tt(&bx, &[r, c])?;
        for _ in 0..5 {
            GemmOperands::ExactF32
                .nt(&x_t, &x_t, &a_t)
                .map_err(metal_err)?;
            GemmOperands::ExactF32
                .nn(&a_t, &a_t, &a2_t)
                .map_err(metal_err)?;
            self.axpby(&a, &a2, &bm, MUON_NS5_B as f32, MUON_NS5_C as f32)?;
            GemmOperands::ExactF32
                .nn(&bm_t, &x_t, &bx_t)
                .map_err(metal_err)?;
            self.axpby(&x, &bx, &x, MUON_NS5_A as f32, 1.0)?;
            self.check(&st, &x, ST_OUT)?;
        }
        let ortho = if transposed {
            let o = self.fresh(n)?;
            transpose_f32_into(&x_t, &self.tt(&o, &[rows, cols])?).map_err(metal_err)?;
            o
        } else {
            x
        };
        let new_p = self.fresh(n)?;
        self.axpby(&pv, &ortho, &new_p, decay, alpha)?;
        self.check(&st, &new_p, ST_OUT)?;
        self.finish(OP, &st, false)?;
        self.copy(&new_p, &pv)?;
        self.copy(&buf, &mv)?;
        self.rt.synchronize().map_err(metal_err)?;
        Ok(Reply::Done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ojas_core::{Backend, Budget, Tensor};

    fn pattern(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    fn upload(w: &mut Worker, v: &[f32]) -> Arg {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
        match w.run(Cmd::Upload { bytes }) {
            Ok(Reply::Bufs(b)) if b.len() == 1 => Arg {
                id: b[0].id,
                off: 0,
                n: v.len(),
            },
            other => panic!("upload: {other:?}"),
        }
    }

    fn read(w: &mut Worker, id: u64, n: usize) -> Vec<f32> {
        match w.run(Cmd::Read {
            id,
            off: 0,
            len: n * 4,
        }) {
            Ok(Reply::Bytes(b)) => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_ne_bytes(*c))
                .collect(),
            other => panic!("read: {other:?}"),
        }
    }

    /// Planes past i32 extents are refused before anything is allocated
    /// (too large to build here, so the bound is tested directly).
    #[test]
    fn attention_planes_past_i32_extents_are_unsupported() {
        const OP: &str = "causal_sdpa_forward";
        assert!(attn_plane_fits(OP, 1 << 23, 128).is_ok());
        assert!(attn_plane_fits(OP, (1 << 24) - 1, 128).is_ok());
        assert!(attn_plane_fits(OP, i32::MAX as u32, 1).is_ok());
        for (t, d) in [(1u32 << 24, 128u32), (u32::MAX, 1), (1 << 31, 1), (u32::MAX, 128)] {
            assert!(
                matches!(
                    attn_plane_fits(OP, t, d),
                    Err(OjasError::Unsupported { op: OP, .. })
                ),
                "t {t} d {d}"
            );
        }
    }

    /// The tiled backward at head dims 65..=128, below the trait: the core
    /// limit (`METAL_MAX_HEAD_DIM`) refuses them in `MetalBackend` today, so
    /// this drives the device thread's command directly and compares with the
    /// CPU reference at the parity tests' tolerance.
    #[test]
    fn tiled_backward_matches_cpu_at_head_dims_up_to_128() {
        let mut w = Worker::open().expect("Metal device");
        let cpu = ojas_cpu::CpuBackend::new(Budget::new(8 << 30))
            .with_numerics(ojas_core::Numerics::Exact);
        for (bh, t, d) in [(2usize, 67usize, 128usize), (1, 300, 128), (3, 33, 100)] {
            let n = bh * t * d;
            let shape = [1, bh, t, d];
            let host: Vec<Vec<f32>> = (0..4).map(|i| pattern(n, (t * d) as u64 + i)).collect();
            let ht: Vec<Tensor> = host
                .iter()
                .map(|v| Tensor::from_f32(v, &shape, &Budget::new(1 << 30)).expect("host"))
                .collect();
            let (wq, wk, wv) = cpu
                .causal_sdpa_backward(&ht[0], &ht[1], &ht[2], &ht[3])
                .expect("cpu");
            let args: Vec<Arg> = host.iter().map(|v| upload(&mut w, v)).collect();
            let bufs = match w.run(Cmd::SdpaBwd {
                q: args[0],
                k: args[1],
                v: args[2],
                gy: args[3],
                bh: bh as u32,
                t: t as u32,
                d: d as u32,
            }) {
                Ok(Reply::Bufs(b)) if b.len() == 3 => b,
                other => panic!("sdpa bwd d{d}: {other:?}"),
            };
            let atol = 2e-5 * (t as f32).sqrt();
            for (buf, want, what) in [(&bufs[0], &wq, "dq"), (&bufs[1], &wk, "dk"), (&bufs[2], &wv, "dv")] {
                let got = read(&mut w, buf.id, n);
                let want = want.to_f32_vec().expect("cpu values");
                for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - e).abs() <= atol + 1e-3 * e.abs(),
                        "bh{bh} t{t} d{d} {what}[{i}]: got {g} want {e}"
                    );
                }
            }
        }
        // Above 128 both directions refuse, naming their limit.
        let x = upload(&mut w, &pattern(4 * 144, 1));
        let fwd = w.run(Cmd::Sdpa {
            q: x,
            k: x,
            v: x,
            bh: 1,
            t: 4,
            d: 144,
        });
        let bwd = w.run(Cmd::SdpaBwd {
            q: x,
            k: x,
            v: x,
            gy: x,
            bh: 1,
            t: 4,
            d: 144,
        });
        for r in [fwd, bwd] {
            assert!(
                matches!(
                    r,
                    Err(OjasError::UnsupportedHeadDim {
                        head_dim: 144,
                        limit: 128
                    })
                ),
                "{r:?}"
            );
        }
    }

    /// The forward at head dims 65..=128, below the trait, against the CPU
    /// reference at the parity tests' tolerance.
    #[test]
    fn forward_matches_cpu_at_head_dims_up_to_128() {
        let mut w = Worker::open().expect("Metal device");
        let cpu = ojas_cpu::CpuBackend::new(Budget::new(8 << 30))
            .with_numerics(ojas_core::Numerics::Exact);
        for (bh, t, d) in [(2usize, 67usize, 128usize), (1, 300, 128), (3, 33, 100)] {
            let n = bh * t * d;
            let shape = [1, bh, t, d];
            let host: Vec<Vec<f32>> = (0..3).map(|i| pattern(n, (t * d) as u64 + 7 + i)).collect();
            let ht: Vec<Tensor> = host
                .iter()
                .map(|v| Tensor::from_f32(v, &shape, &Budget::new(1 << 30)).expect("host"))
                .collect();
            let want = cpu
                .causal_sdpa_forward(&ht[0], &ht[1], &ht[2])
                .expect("cpu")
                .to_f32_vec()
                .expect("cpu values");
            let args: Vec<Arg> = host.iter().map(|v| upload(&mut w, v)).collect();
            let out = match w.run(Cmd::Sdpa {
                q: args[0],
                k: args[1],
                v: args[2],
                bh: bh as u32,
                t: t as u32,
                d: d as u32,
            }) {
                Ok(Reply::Bufs(b)) if b.len() == 1 => b,
                other => panic!("sdpa fwd d{d}: {other:?}"),
            };
            let got = read(&mut w, out[0].id, n);
            let atol = 2e-5 * (t as f32).sqrt();
            for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - e).abs() <= atol + 1e-4 * e.abs(),
                    "bh{bh} t{t} d{d} o[{i}]: got {g} want {e}"
                );
            }
        }
    }

    const CHILD_ENV: &str = "OJAS_METAL_REFUSED_OPEN_CHILD";
    const SELF_NAME: &str = "device::tests::refused_opens_do_not_accumulate_owner_threads";

    /// Continuation lines from `ps -M` are threads beyond the process line.
    fn extra_threads() -> usize {
        let out = std::process::Command::new("/bin/ps")
            .args(["-M", "-p", &std::process::id().to_string()])
            .output()
            .expect("ps -M");
        assert!(
            out.status.success(),
            "ps -M failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|line| line.starts_with(' ') || line.starts_with('\t'))
            .count()
    }

    /// A Metal load that gets as far as the device thread and then fails must
    /// not leave that thread behind. On this Mac a real load does not refuse
    /// after a probe, so this injects the open failure the owner thread sees.
    #[test]
    fn refused_opens_do_not_accumulate_owner_threads() {
        if std::env::var_os(CHILD_ENV).is_some() {
            let before = extra_threads();
            for i in 0..8 {
                let err = if i == 3 {
                    spawn_with(|| panic!("injected open panic")).expect_err("panic opener")
                } else {
                    spawn_with(|| Err(metal_err("injected open failure")))
                        .expect_err("failed opener")
                };
                let msg = err.to_string();
                assert!(
                    msg.contains("injected") || msg.contains("panicked"),
                    "{msg}"
                );
            }
            let after = extra_threads();
            assert!(
                after <= before,
                "owner threads grew across 8 refused opens: {before} extra -> {after} extra"
            );
            return;
        }
        let exe = std::env::current_exe().expect("test binary");
        let out = std::process::Command::new(exe)
            .args(["--exact", "--test-threads=1", SELF_NAME])
            .env(CHILD_ENV, "1")
            .output()
            .expect("spawn isolated child");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "refused-open child failed:\n{stdout}\n{stderr}"
        );
        assert!(
            stdout.contains("1 passed") || stderr.contains("1 passed"),
            "child ran no test:\n{stdout}\n{stderr}"
        );
    }
}
