//! Decode rows of the paired benchmark (`bench/README.md`, "Decode rows").
//!
//! nanolab's default GPT ([`ModelSpec::nanolab_124m`]) with weights from the
//! shared generator: a `GEN_PROMPT`-token prompt, then greedy decode of
//! `GEN_NEW` tokens. Compiled by `#[path]` into
//! `ojas-infer/examples/decode_vs_torch.rs`, which picks the lane: CPU
//! (`DeviceDecoder<CpuBackend>`), the CPU host path (`CpuGpt`), Metal or
//! wgpu. `bench/torch_rows.py` holds the torch twin (`_gen`), nanolab's own
//! KV-cached `GPT.forward_hidden_window`.
//!
//! Rows:
//! - `gen_prefill_p{P}`: one prompt forward from an empty cache.
//! - `gen_greedy_p{P}_n{N}`: the prompt forward, then `N - 1` single-token
//!   forwards, each fed the previous argmax (the last emitted token is not
//!   forwarded). Decode time per token is `(greedy - prefill) / (N - 1)`.
//!
//! Both rows are gated on the prompt's last-position logits against torch.
//! The greedy ids are not a gate (a near-tie may resolve differently under
//! another summation order): a `_gen_ids` record says how many of the `N`
//! ids equal torch's. A `_gen_traffic` record gives the host transfers one
//! decode step makes (`DeviceDecoder` lanes).
//!
//! Weights: each parameter from `gen` with seed `fnv1a32(name) % 2^20 +
//! 1000`; names containing `norm` are `1 + U * 2^-3`, other vectors
//! `U * 2^-1`, `tok_emb.weight` `U * 2^-2`, other matrices `U * 2^-4`, where
//! `U` is the generator's `[-1, 1)` stream.

use std::fs;

use ojas_core::{Backend, Budget, OjasError, Tensor};
use ojas_infer::{argmax_token, CpuGpt, DeviceDecoder, HostTraffic, KvCache};
use ojas_model::{param_table, ModelParams, ModelSpec};

use crate::rows::{gen, gen_targets, json_string, Runner, R, TOL};

pub const GEN_PROMPT: usize = 32;
pub const GEN_NEW: usize = 32;
const PROMPT_SEED: u64 = 1701;

/// FNV-1a over the name's bytes; `torch_rows.py::_fnv1a32` is the same.
pub fn fnv1a32(name: &str) -> u64 {
    let mut h: u32 = 0x811C_9DC5;
    for b in name.bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    u64::from(h)
}

/// One parameter's values by the rule in the module docs.
pub fn gen_param(name: &str, shape: &[usize]) -> Vec<f32> {
    let n = shape.iter().product();
    let seed = fnv1a32(name) % (1 << 20) + 1000;
    if name.contains("norm") {
        gen(n, seed, -3).into_iter().map(|v| 1.0 + v).collect()
    } else if shape.len() == 1 {
        gen(n, seed, -1)
    } else if name == "tok_emb.weight" {
        gen(n, seed, -2)
    } else {
        gen(n, seed, -4)
    }
}

fn weights(spec: &ModelSpec, budget: &Budget) -> R<ModelParams<Tensor>> {
    let flat = param_table(spec)?
        .iter()
        .map(|info| Tensor::from_f32(&gen_param(&info.name, &info.shape), &info.shape, budget))
        .collect::<R<Vec<_>>>()?;
    ModelParams::from_flat(spec, flat)
}

fn spec_string(spec: &ModelSpec) -> String {
    format!(
        "gen:nanolab_124m:vocab{}:d{}:L{}:H{}x{}:hidden{}:prompt{GEN_PROMPT}:{PROMPT_SEED}:new{GEN_NEW}:\
         weights=fnv1a32%2^20+1000,norm=1+2^-3,vec=2^-1,tok_emb=2^-2,mat=2^-4",
        spec.vocab, spec.n_embd, spec.n_layer, spec.n_head, spec.head_dim, spec.hidden
    )
}

/// One decoder and its cache, forwarded from an empty cache every call.
pub trait Gen {
    fn prefill(&mut self, prompt: &[u32]) -> R<Vec<f32>>;
    fn greedy(&mut self, prompt: &[u32], n: usize) -> R<Vec<u32>>;
    fn traffic(&self) -> Option<HostTraffic>;
}

impl<B: Backend> Gen for DeviceDecoder<B> {
    fn prefill(&mut self, prompt: &[u32]) -> R<Vec<f32>> {
        self.reset();
        self.forward(prompt)
    }

    fn greedy(&mut self, prompt: &[u32], n: usize) -> R<Vec<u32>> {
        self.reset();
        self.greedy_decode(prompt, n)
    }

    fn traffic(&self) -> Option<HostTraffic> {
        Some(DeviceDecoder::traffic(self))
    }
}

/// `CpuGpt` with a cache it clears before every call.
pub struct HostGen {
    pub model: CpuGpt,
    pub budget: Budget,
    pub capacity: usize,
}

impl HostGen {
    fn cache(&self) -> R<KvCache> {
        KvCache::for_model(&self.model, self.capacity, &self.budget)
    }
}

impl Gen for HostGen {
    fn prefill(&mut self, prompt: &[u32]) -> R<Vec<f32>> {
        let mut cache = self.cache()?;
        self.model.forward_tokens(prompt, &mut cache)
    }

    fn greedy(&mut self, prompt: &[u32], n: usize) -> R<Vec<u32>> {
        let mut cache = self.cache()?;
        self.model.greedy_decode(prompt, &mut cache, n)
    }

    fn traffic(&self) -> Option<HostTraffic> {
        None
    }
}

pub fn spec() -> ModelSpec {
    ModelSpec::nanolab_124m()
}

/// The model's weights on the host, from the generator.
pub fn host_weights<Bk: Backend>(r: &Runner<'_, Bk>) -> R<ModelParams<Tensor>> {
    weights(&spec(), &r.host)
}

fn prompt() -> Vec<u32> {
    gen_targets(GEN_PROMPT, PROMPT_SEED)
}

fn emit_info<Bk: Backend>(r: &mut Runner<'_, Bk>, row: &str, kind: &str, body: &str) {
    let line = format!(
        "{{\"runtime\":{},\"row\":{},\"status\":\"info\",\"for\":{},{body}}}",
        json_string(&r.runtime),
        json_string(kind),
        json_string(row)
    );
    r.emit(line);
}

/// Torch's greedy ids, written by its `ref` pass as `<row>.ids` (u32 LE).
fn torch_ids<Bk: Backend>(r: &Runner<'_, Bk>, row: &str) -> Result<Vec<u32>, String> {
    let path = r.refdir.join(format!("{row}.ids"));
    let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (chunks, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(format!("{}: {} bytes", path.display(), bytes.len()));
    }
    Ok(chunks.iter().map(|c| u32::from_le_bytes(*c)).collect())
}

/// Both decode rows on `g`.
pub fn run<Bk: Backend, G: Gen>(r: &mut Runner<'_, Bk>, g: &mut G) {
    let s = spec();
    let sp = spec_string(&s);
    let ids = prompt();
    let prefill_row = format!("gen_prefill_p{GEN_PROMPT}");
    let greedy_row = format!("gen_greedy_p{GEN_PROMPT}_n{GEN_NEW}");
    let parity = |_: &Runner<'_, Bk>, g: &mut G| -> R<Vec<Vec<f32>>> { Ok(vec![g.prefill(&ids)?]) };
    r.run(&prefill_row, &sp, TOL, g, parity, |_, g| {
        g.prefill(&ids)?;
        Ok(Vec::new())
    });
    r.run(&greedy_row, &sp, TOL, g, parity, |_, g| {
        g.greedy(&ids, GEN_NEW)?;
        Ok(Vec::new())
    });
    if !r.want(&greedy_row) {
        return;
    }
    let body = match g.greedy(&ids, GEN_NEW) {
        Ok(mine) => match torch_ids(r, &greedy_row) {
            Ok(theirs) => {
                let same = mine.iter().zip(&theirs).filter(|(a, b)| a == b).count();
                let prefix = mine.iter().zip(&theirs).take_while(|(a, b)| a == b).count();
                format!(
                    "\"ids_equal\":{same},\"common_prefix\":{prefix},\"of\":{},\"torch_of\":{}",
                    mine.len(),
                    theirs.len()
                )
            }
            Err(e) => format!("\"detail\":{}", json_string(&e)),
        },
        Err(e) => format!("\"detail\":{}", json_string(&format!("{e}"))),
    };
    emit_info(r, &greedy_row, "_gen_ids", &body);
    if let Err(e) = traffic(r, g, &ids, &greedy_row) {
        let body = format!("\"detail\":{}", json_string(&format!("{e}")));
        emit_info(r, &greedy_row, "_gen_traffic", &body);
    }
}

/// Transfers of the prefill and of one decode step, from the decoder's own
/// counters: a prefill, then the greedy run; the difference over its
/// `N - 1` decode forwards.
fn traffic<Bk: Backend, G: Gen>(
    r: &mut Runner<'_, Bk>,
    g: &mut G,
    ids: &[u32],
    row: &str,
) -> R<()> {
    let Some(t0) = g.traffic() else {
        return Ok(());
    };
    let logits = g.prefill(ids)?;
    argmax_token(&logits)?;
    let t1 = g.traffic().unwrap_or_default();
    g.greedy(ids, GEN_NEW)?;
    let t2 = g.traffic().unwrap_or_default();
    let steps = (GEN_NEW - 1) as f64;
    let per = |a: u64, b: u64, c: u64| ((c - b) - (b - a)) as f64 / steps;
    let body = format!(
        "\"prefill_uploads\":{},\"prefill_upload_bytes\":{},\"prefill_readback_bytes\":{},\
         \"step_uploads\":{},\"step_upload_bytes\":{},\"step_readbacks\":{},\"step_readback_bytes\":{}",
        t1.uploads - t0.uploads,
        t1.upload_bytes - t0.upload_bytes,
        t1.readback_bytes - t0.readback_bytes,
        per(t0.uploads, t1.uploads, t2.uploads),
        per(t0.upload_bytes, t1.upload_bytes, t2.upload_bytes),
        per(t0.readbacks, t1.readbacks, t2.readbacks),
        per(t0.readback_bytes, t1.readback_bytes, t2.readback_bytes),
    );
    emit_info(r, row, "_gen_traffic", &body);
    Ok(())
}

pub fn row_names() -> [String; 2] {
    [
        format!("gen_prefill_p{GEN_PROMPT}"),
        format!("gen_greedy_p{GEN_PROMPT}_n{GEN_NEW}"),
    ]
}

/// A `DeviceDecoder` lane on `be`.
pub fn run_device<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    let names = row_names();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    r.group(&refs, |r| {
        let w = host_weights(r)?;
        let mut dec = DeviceDecoder::new(r.be, &spec(), &w, GEN_PROMPT + GEN_NEW)?;
        drop(w);
        run(r, &mut dec);
        Ok(())
    });
    if r.want(&names[1]) {
        if let Err(e) = readback(r, &names[1]) {
            let body = format!("\"detail\":{}", json_string(&format!("{e}")));
            emit_info(r, &names[1], "_gen_readback", &body);
        }
    }
}

/// What reading one `[1, vocab]` logit row back costs on this backend, min
/// of [`READBACK_ITERS`]: `Tensor::to_host` (the device read, then the
/// decode of its bytes into a typed host tensor: the transient double copy
/// of `docs/typed-storage-plan.md`) against `DeviceBuffer::read_bytes`
/// alone. The difference is what a read straight into a caller's buffer
/// would save. A host backend has no device buffer and reports nothing.
fn readback<Bk: Backend>(r: &mut Runner<'_, Bk>, row: &str) -> R<()> {
    const READBACK_ITERS: usize = 50;
    let vocab = spec().vocab;
    let host = Tensor::from_f32(&gen(vocab, 77, -1), &[1, vocab], &r.host)?;
    let dev = r.be.upload(&host)?;
    r.be.sync()?;
    let Some(buf) = dev.device_buffer() else {
        return Ok(());
    };
    let bytes = vocab * 4;
    let min_us = |f: &mut dyn FnMut() -> R<()>| -> R<f64> {
        let mut best = f64::INFINITY;
        for _ in 0..READBACK_ITERS {
            let t = std::time::Instant::now();
            f()?;
            best = best.min(t.elapsed().as_secs_f64() * 1e6);
        }
        Ok(best)
    };
    let to_host = min_us(&mut || dev.to_host(&r.host).map(drop))?;
    let raw = min_us(&mut || buf.read_bytes(0, bytes).map(drop))?;
    let body = format!(
        "\"bytes\":{bytes},\"to_host_us\":{to_host:.2},\"read_bytes_us\":{raw:.2},\
         \"decode_pass_us\":{:.2},\"iters\":{READBACK_ITERS}",
        to_host - raw
    );
    emit_info(r, row, "_gen_readback", &body);
    Ok(())
}

/// The `CpuGpt` lane.
pub fn run_host<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    let names = row_names();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    r.group(&refs, |r| {
        let w = host_weights(r)?;
        let model = CpuGpt::new(&spec(), &w)?;
        let mut g = HostGen {
            model,
            budget: r.host.clone(),
            capacity: GEN_PROMPT + GEN_NEW,
        };
        run(r, &mut g);
        Ok::<(), OjasError>(())
    });
}
