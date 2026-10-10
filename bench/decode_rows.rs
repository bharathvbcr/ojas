//! Decode rows of the paired benchmark (`bench/README.md`, "Decode rows").
//!
//! nanolab's default GPT ([`ModelSpec::nanolab_124m`]) with weights from the
//! shared generator: a `GEN_PROMPT`-token prompt, then `GEN_NEW` new tokens.
//! Compiled by `#[path]` into `ojas-infer/examples/decode_vs_torch.rs`,
//! which picks the lane: CPU (`DeviceDecoder<CpuBackend>`), the CPU host
//! path (`CpuGpt`), Metal or wgpu. `bench/torch_rows.py` holds the torch
//! twin (`_gen`), nanolab's own KV-cached `GPT.forward_hidden_window`.
//!
//! Rows:
//! - `gen_prefill_p{P}`: one prompt forward from an empty cache.
//! - `gen_greedy_p{P}_n{N}`: the prompt forward, then `N - 1` single-token
//!   forwards, each fed the previous argmax (the last emitted token is not
//!   forwarded). Decode time per token is `(greedy - prefill) / (N - 1)`.
//! - `gen_topk{K}_p{P}_n{N}`: the same forwards through `generate`, each id
//!   drawn at temperature 1 from the top `K` logits (seed [`SAMPLE_SEED`]):
//!   the sampled path, which needs only the `K` leaders of each row.
//! - `gen_gqa{G}_prefill_p{P}` and `gen_gqa{G}_greedy_p{P}_n{N}`: the first
//!   two on the same model with `G` KV heads (grouped-query attention), its
//!   weights from the same rule.
//!
//! Every row is gated on its model's prompt last-position logits against
//! torch. The ids are not a gate (a near-tie may resolve differently under
//! another summation order): a `_gen_ids` record says how many of a greedy
//! row's `N` ids equal torch's (sampled ids come from another generator and
//! are not compared). A `_gen_traffic` record gives the host transfers one
//! decode step of a row makes (`DeviceDecoder` lanes).
//!
//! Weights: each parameter from `gen` with seed `fnv1a32(name) % 2^20 +
//! 1000`; names containing `norm` are `1 + U * 2^-3`, other vectors
//! `U * 2^-1`, `tok_emb.weight` `U * 2^-2`, other matrices `U * 2^-4`, where
//! `U` is the generator's `[-1, 1)` stream.

use std::fs;

use ojas_core::{Backend, Budget, OjasError, Tensor};
use ojas_infer::{CpuGpt, DeviceDecoder, GenerateConfig, HostTraffic, KvCache, SamplingConfig};
use ojas_model::{param_table, ModelParams, ModelSpec};

use crate::rows::{gen, gen_targets, json_string, Runner, R, TOL};

pub const GEN_PROMPT: usize = 32;
pub const GEN_NEW: usize = 32;
const PROMPT_SEED: u64 = 1701;
/// The sampled row keeps this many leaders, at temperature 1.
pub const GEN_TOP_K: usize = 50;
/// The sampled row's `GenerateConfig::seed`.
pub const SAMPLE_SEED: u64 = 1702;
/// KV heads of the grouped-query rows' model (12 query heads).
pub const GQA_KV_HEADS: usize = 4;

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

/// The rows' spec line; `torch_rows.py::_gen` writes the same. A grouped
/// model adds its KV heads.
fn spec_string(spec: &ModelSpec) -> String {
    let kv = if spec.n_kv_head == spec.n_head {
        String::new()
    } else {
        format!(":kv{}", spec.n_kv_head)
    };
    format!(
        "gen:nanolab_124m:vocab{}:d{}:L{}:H{}x{}{kv}:hidden{}:prompt{GEN_PROMPT}:{PROMPT_SEED}:new{GEN_NEW}:\
         weights=fnv1a32%2^20+1000,norm=1+2^-3,vec=2^-1,tok_emb=2^-2,mat=2^-4",
        spec.vocab, spec.n_embd, spec.n_layer, spec.n_head, spec.head_dim, spec.hidden
    )
}

/// The sampled row's settings: temperature 1, the top [`GEN_TOP_K`].
pub fn sampling() -> GenerateConfig {
    GenerateConfig {
        sampling: SamplingConfig {
            temperature: 1.0,
            top_k: Some(GEN_TOP_K),
            top_p: None,
        },
        seed: SAMPLE_SEED,
        max_new_tokens: GEN_NEW,
        stop_tokens: Vec::new(),
    }
}

/// One decoder and its cache, forwarded from an empty cache every call.
pub trait Gen {
    fn prefill(&mut self, prompt: &[u32]) -> R<Vec<f32>>;
    fn greedy(&mut self, prompt: &[u32], n: usize) -> R<Vec<u32>>;
    fn sample(&mut self, prompt: &[u32], cfg: &GenerateConfig) -> R<Vec<u32>>;
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

    fn sample(&mut self, prompt: &[u32], cfg: &GenerateConfig) -> R<Vec<u32>> {
        self.reset();
        self.generate(prompt, cfg)
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

    fn sample(&mut self, prompt: &[u32], cfg: &GenerateConfig) -> R<Vec<u32>> {
        let mut cache = self.cache()?;
        self.model.generate(prompt, &mut cache, cfg)
    }

    fn traffic(&self) -> Option<HostTraffic> {
        None
    }
}

/// A model the decode rows run, and the names of its rows.
pub struct Model {
    pub spec: ModelSpec,
    pub prefill: String,
    pub greedy: String,
    /// The sampled row; nanolab's default model only.
    pub topk: Option<String>,
}

impl Model {
    pub fn names(&self) -> Vec<String> {
        [Some(&self.prefill), Some(&self.greedy), self.topk.as_ref()]
            .into_iter()
            .flatten()
            .cloned()
            .collect()
    }
}

/// nanolab's default GPT, then the same with [`GQA_KV_HEADS`] KV heads.
pub fn models() -> [Model; 2] {
    let base = ModelSpec::nanolab_124m();
    let gqa = ModelSpec {
        n_kv_head: GQA_KV_HEADS,
        ..base
    };
    [
        Model {
            spec: base,
            prefill: format!("gen_prefill_p{GEN_PROMPT}"),
            greedy: format!("gen_greedy_p{GEN_PROMPT}_n{GEN_NEW}"),
            topk: Some(format!("gen_topk{GEN_TOP_K}_p{GEN_PROMPT}_n{GEN_NEW}")),
        },
        Model {
            spec: gqa,
            prefill: format!("gen_gqa{GQA_KV_HEADS}_prefill_p{GEN_PROMPT}"),
            greedy: format!("gen_gqa{GQA_KV_HEADS}_greedy_p{GEN_PROMPT}_n{GEN_NEW}"),
            topk: None,
        },
    ]
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

/// Every row of `m` on `g`, then its `_gen_ids` and `_gen_traffic` records.
pub fn run<Bk: Backend, G: Gen>(r: &mut Runner<'_, Bk>, g: &mut G, m: &Model) {
    let sp = spec_string(&m.spec);
    let ids = prompt();
    let cfg = sampling();
    let parity = |_: &Runner<'_, Bk>, g: &mut G| -> R<Vec<Vec<f32>>> { Ok(vec![g.prefill(&ids)?]) };
    r.run(&m.prefill, &sp, TOL, g, parity, |_, g| {
        g.prefill(&ids)?;
        Ok(Vec::new())
    });
    r.run(&m.greedy, &sp, TOL, g, parity, |_, g| {
        g.greedy(&ids, GEN_NEW)?;
        Ok(Vec::new())
    });
    if let Some(topk) = &m.topk {
        r.run(topk, &sp, TOL, g, parity, |_, g| {
            g.sample(&ids, &cfg)?;
            Ok(Vec::new())
        });
    }
    if r.want(&m.greedy) {
        let body = match g.greedy(&ids, GEN_NEW) {
            Ok(mine) => match torch_ids(r, &m.greedy) {
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
        emit_info(r, &m.greedy, "_gen_ids", &body);
        traffic_record(r, g, &m.greedy, |g, n| g.greedy(&ids, n));
    }
    if let Some(topk) = m.topk.as_ref().filter(|t| r.want(t)) {
        traffic_record(r, g, topk, |g, n| {
            let cfg = GenerateConfig {
                max_new_tokens: n,
                ..cfg.clone()
            };
            g.sample(&ids, &cfg)
        });
    }
}

/// [`traffic`], or the error it met, as a `_gen_traffic` record for `row`.
fn traffic_record<Bk: Backend, G: Gen>(
    r: &mut Runner<'_, Bk>,
    g: &mut G,
    row: &str,
    decode: impl FnMut(&mut G, usize) -> R<Vec<u32>>,
) {
    if let Err(e) = traffic(r, g, row, decode) {
        let body = format!("\"detail\":{}", json_string(&format!("{e}")));
        emit_info(r, row, "_gen_traffic", &body);
    }
}

/// Transfers of the prompt forward and of one decode step, from the
/// decoder's own counters. `decode(g, n)` is the row's decode of `n` new
/// tokens: with `n` = 1 it is the prompt forward alone (picking as the row
/// picks, so a greedy prompt forward reads back an id, not the row), and
/// with `GEN_NEW` it adds `GEN_NEW - 1` decode forwards; a step is the
/// difference over those.
fn traffic<Bk: Backend, G: Gen>(
    r: &mut Runner<'_, Bk>,
    g: &mut G,
    row: &str,
    mut decode: impl FnMut(&mut G, usize) -> R<Vec<u32>>,
) -> R<()> {
    let Some(t0) = g.traffic() else {
        return Ok(());
    };
    decode(g, 1)?;
    let t1 = g.traffic().unwrap_or_default();
    decode(g, GEN_NEW)?;
    let t2 = g.traffic().unwrap_or_default();
    let steps = (GEN_NEW - 1) as f64;
    // Counters only grow, and the long run makes every forward the short
    // one makes, so a step is never negative; if it were, that is an error.
    let per = |a: u64, b: u64, c: u64| -> R<f64> {
        (c - b)
            .checked_sub(b - a)
            .map(|d| d as f64 / steps)
            .ok_or_else(|| OjasError::Shape {
                op: "decode traffic",
                detail: format!("{GEN_NEW} tokens moved less than 1 ({} < {})", c - b, b - a),
            })
    };
    let body = format!(
        "\"prefill_uploads\":{},\"prefill_upload_bytes\":{},\"prefill_readback_bytes\":{},\
         \"step_uploads\":{},\"step_upload_bytes\":{},\"step_readbacks\":{},\"step_readback_bytes\":{}",
        t1.uploads - t0.uploads,
        t1.upload_bytes - t0.upload_bytes,
        t1.readback_bytes - t0.readback_bytes,
        per(t0.uploads, t1.uploads, t2.uploads)?,
        per(t0.upload_bytes, t1.upload_bytes, t2.upload_bytes)?,
        per(t0.readbacks, t1.readbacks, t2.readbacks)?,
        per(t0.readback_bytes, t1.readback_bytes, t2.readback_bytes)?,
    );
    emit_info(r, row, "_gen_traffic", &body);
    Ok(())
}

/// A `DeviceDecoder` lane on `be`.
pub fn run_device<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    let models = models();
    for m in &models {
        let names = m.names();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        r.group(&refs, |r| {
            let w = weights(&m.spec, &r.host)?;
            let mut dec = DeviceDecoder::new(r.be, &m.spec, &w, GEN_PROMPT + GEN_NEW)?;
            drop(w);
            run(r, &mut dec, m);
            Ok(())
        });
    }
    let greedy = &models[0].greedy;
    if r.want(greedy) {
        if let Err(e) = readback(r, greedy) {
            let body = format!("\"detail\":{}", json_string(&format!("{e}")));
            emit_info(r, greedy, "_gen_readback", &body);
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
    let vocab = ModelSpec::nanolab_124m().vocab;
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
    for m in &models() {
        let names = m.names();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        r.group(&refs, |r| {
            let w = weights(&m.spec, &r.host)?;
            let model = CpuGpt::new(&m.spec, &w)?;
            let mut g = HostGen {
                model,
                budget: r.host.clone(),
                capacity: GEN_PROMPT + GEN_NEW,
            };
            run(r, &mut g, m);
            Ok::<(), OjasError>(())
        });
    }
}
