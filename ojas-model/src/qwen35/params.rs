//! The Qwen3.5 text tower's parameters, and the map between them and the
//! Hugging Face tensors.
//!
//! Three fused Hugging Face tensors are held as parts, so the graph needs no
//! slice op and every part is an ordinary linear or conv weight:
//!
//! - `linear_attn.in_proj_qkv.weight` `[2 Hk Dk + Hv Dv, hidden]` is the row
//!   blocks `[q | k | v]` (transformers splits its output in that order).
//!   A linear's output rows are independent, so the three linears are the
//!   fused one exactly and its gradient is theirs stacked.
//! - `linear_attn.conv1d.weight` `[C, 1, K]` is squeezed to `[C, K]` and cut
//!   into the same three channel blocks. The conv is depthwise, so each
//!   channel's output reads only its own taps: three convs are the fused one
//!   exactly.
//! - `self_attn.q_proj.weight` `[H 2D, hidden]` holds, per head `h`, the
//!   query rows `h 2D .. h 2D + D` and then the output gate rows (transformers
//!   views its output `[B, T, H, 2D]` and chunks the last axis). The parts
//!   are the query rows of every head and the gate rows of every head.
//!
//! [`fuse_grads`] is the inverse, for comparing gradients with transformers
//! and for writing them under Hugging Face names.

use std::convert::Infallible;

use ojas_core::{
    f32_to_bf16, shape_product, Backend, Budget, DType, OjasError, Tensor, LE_READ_CHUNK_BYTES,
};
use ojas_data::CounterRng;
use ojas_io::{bf16_to_f32, f16_to_f32, SafeTensors, StDtype};

use crate::init::fnv1a;
use crate::lora::{LoraConfig, LoraSite};

use crate::qwen35::spec::{Qwen35Mixer, Qwen35Spec};

/// A gated delta net layer's mixer.
#[derive(Clone, Debug)]
pub struct GdnParams<V> {
    /// The `q`, `k` and `v` row blocks of `in_proj_qkv`.
    pub wq: V,
    pub wk: V,
    pub wv: V,
    /// The matching channel blocks of `conv1d.weight`, each `[C, K]`.
    pub conv_q: V,
    pub conv_k: V,
    pub conv_v: V,
    /// `in_proj_z`, the gated norm's gate.
    pub wz: V,
    /// `in_proj_a`, the log decay's input.
    pub wa: V,
    /// `in_proj_b`, `beta`'s logit.
    pub wb: V,
    pub a_log: V,
    pub dt_bias: V,
    /// The gated norm's weight `[Dv]`, a plain scale.
    pub norm: V,
    pub out: V,
}

/// A gated attention layer's mixer.
#[derive(Clone, Debug)]
pub struct AttnParams<V> {
    /// The query rows of `q_proj`, `[H D, hidden]`.
    pub wq: V,
    /// The output gate rows of `q_proj`, `[H D, hidden]`.
    pub wgate: V,
    pub wk: V,
    pub wv: V,
    /// Per-head RMSNorm weights `[D]`, applied as `1 + w`.
    pub q_norm: V,
    pub k_norm: V,
    pub wo: V,
}

#[derive(Clone, Debug)]
pub enum MixerParams<V> {
    GatedDeltaNet(GdnParams<V>),
    Attention(AttnParams<V>),
}

#[derive(Clone, Debug)]
pub struct LayerParams<V> {
    /// `input_layernorm`, applied as `1 + w`.
    pub input_norm: V,
    pub mixer: MixerParams<V>,
    /// `post_attention_layernorm`, applied as `1 + w`.
    pub post_norm: V,
    pub gate: V,
    pub up: V,
    pub down: V,
}

/// The tower: tied embedding, layers, final norm (applied as `1 + w`).
#[derive(Clone, Debug)]
pub struct Qwen35Params<V> {
    pub embed: V,
    pub layers: Vec<LayerParams<V>>,
    pub norm: V,
}

impl<V> Qwen35Params<V> {
    /// `f` of every parameter, in one fixed order (the order of
    /// [`Qwen35Params::values`]).
    pub fn try_map<'a, W, E>(
        &'a self,
        f: &mut impl FnMut(&'a V) -> Result<W, E>,
    ) -> Result<Qwen35Params<W>, E> {
        let embed = f(&self.embed)?;
        let mut layers = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            let input_norm = f(&l.input_norm)?;
            let mixer = match &l.mixer {
                MixerParams::GatedDeltaNet(m) => MixerParams::GatedDeltaNet(GdnParams {
                    wq: f(&m.wq)?,
                    wk: f(&m.wk)?,
                    wv: f(&m.wv)?,
                    conv_q: f(&m.conv_q)?,
                    conv_k: f(&m.conv_k)?,
                    conv_v: f(&m.conv_v)?,
                    wz: f(&m.wz)?,
                    wa: f(&m.wa)?,
                    wb: f(&m.wb)?,
                    a_log: f(&m.a_log)?,
                    dt_bias: f(&m.dt_bias)?,
                    norm: f(&m.norm)?,
                    out: f(&m.out)?,
                }),
                MixerParams::Attention(m) => MixerParams::Attention(AttnParams {
                    wq: f(&m.wq)?,
                    wgate: f(&m.wgate)?,
                    wk: f(&m.wk)?,
                    wv: f(&m.wv)?,
                    q_norm: f(&m.q_norm)?,
                    k_norm: f(&m.k_norm)?,
                    wo: f(&m.wo)?,
                }),
            };
            layers.push(LayerParams {
                input_norm,
                mixer,
                post_norm: f(&l.post_norm)?,
                gate: f(&l.gate)?,
                up: f(&l.up)?,
                down: f(&l.down)?,
            });
        }
        Ok(Qwen35Params {
            embed,
            layers,
            norm: f(&self.norm)?,
        })
    }

    /// Every parameter, in [`Qwen35Params::try_map`]'s order.
    pub fn values(&self) -> Vec<&V> {
        let mut out = Vec::new();
        let Ok(_) = self.try_map(&mut |v| {
            out.push(v);
            Ok::<(), Infallible>(())
        });
        out
    }
}

/// How [`load_hf`] stores a tensor: the tied embedding follows its `table`
/// storage, the weights a `linear` reads follow `linears`, and everything
/// else (norms, convolutions, `A_log`, `dt_bias`) is `F32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HfKind {
    Table,
    Linear,
    Plain,
}

/// One Hugging Face tensor of the tower: its name, stored shape and kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub kind: HfKind,
}

/// Every Hugging Face tensor the tower reads under `prefix` (`"model."` for
/// a text-only checkpoint, `"model.language_model."` under a `text_config`),
/// in [`Qwen35Params::try_map`]'s order of the fused tensors.
pub fn hf_tensors(spec: &Qwen35Spec, prefix: &str) -> Vec<HfTensor> {
    let t = |kind: HfKind, name: String, shape: &[usize]| HfTensor {
        name,
        shape: shape.to_vec(),
        kind,
    };
    let (d, f) = (spec.hidden, spec.intermediate);
    let mut out = vec![t(
        HfKind::Table,
        format!("{prefix}embed_tokens.weight"),
        &[spec.vocab, d],
    )];
    for (i, kind) in spec.layers.iter().enumerate() {
        let l = format!("{prefix}layers.{i}.");
        out.push(t(HfKind::Plain, format!("{l}input_layernorm.weight"), &[d]));
        match kind {
            Qwen35Mixer::GatedDeltaNet => {
                let [cq, ck, cv] = spec.gdn_qkv_channels();
                let c = cq + ck + cv;
                let (h, dv) = (spec.gdn_heads, spec.gdn_value_dim);
                let m = format!("{l}linear_attn.");
                out.push(t(HfKind::Linear, format!("{m}in_proj_qkv.weight"), &[c, d]));
                out.push(t(
                    HfKind::Plain,
                    format!("{m}conv1d.weight"),
                    &[c, 1, spec.conv_width],
                ));
                out.push(t(
                    HfKind::Linear,
                    format!("{m}in_proj_z.weight"),
                    &[h * dv, d],
                ));
                out.push(t(HfKind::Linear, format!("{m}in_proj_a.weight"), &[h, d]));
                out.push(t(HfKind::Linear, format!("{m}in_proj_b.weight"), &[h, d]));
                out.push(t(HfKind::Plain, format!("{m}A_log"), &[h]));
                out.push(t(HfKind::Plain, format!("{m}dt_bias"), &[h]));
                out.push(t(HfKind::Plain, format!("{m}norm.weight"), &[dv]));
                out.push(t(
                    HfKind::Linear,
                    format!("{m}out_proj.weight"),
                    &[d, h * dv],
                ));
            }
            Qwen35Mixer::Attention => {
                let (hq, hkv, hd) = (spec.q_heads, spec.kv_heads, spec.head_dim);
                let m = format!("{l}self_attn.");
                out.push(t(
                    HfKind::Linear,
                    format!("{m}q_proj.weight"),
                    &[hq * 2 * hd, d],
                ));
                out.push(t(
                    HfKind::Linear,
                    format!("{m}k_proj.weight"),
                    &[hkv * hd, d],
                ));
                out.push(t(
                    HfKind::Linear,
                    format!("{m}v_proj.weight"),
                    &[hkv * hd, d],
                ));
                out.push(t(HfKind::Plain, format!("{m}q_norm.weight"), &[hd]));
                out.push(t(HfKind::Plain, format!("{m}k_norm.weight"), &[hd]));
                out.push(t(
                    HfKind::Linear,
                    format!("{m}o_proj.weight"),
                    &[d, hq * hd],
                ));
            }
        }
        out.push(t(
            HfKind::Plain,
            format!("{l}post_attention_layernorm.weight"),
            &[d],
        ));
        out.push(t(
            HfKind::Linear,
            format!("{l}mlp.gate_proj.weight"),
            &[f, d],
        ));
        out.push(t(HfKind::Linear, format!("{l}mlp.up_proj.weight"), &[f, d]));
        out.push(t(
            HfKind::Linear,
            format!("{l}mlp.down_proj.weight"),
            &[d, f],
        ));
    }
    out.push(t(HfKind::Plain, format!("{prefix}norm.weight"), &[d]));
    out
}

fn refuse(op: &'static str, detail: String) -> OjasError {
    OjasError::Shape { op, detail }
}

/// `rows` rows of `cols` values cut into consecutive blocks of `counts`
/// rows.
fn split_rows<T: Copy>(v: &[T], cols: usize, counts: &[usize]) -> Vec<Vec<T>> {
    let mut at = 0;
    counts
        .iter()
        .map(|&n| {
            let part = v[at * cols..(at + n) * cols].to_vec();
            at += n;
            part
        })
        .collect()
}

/// `q_proj`'s rows (per head: `d` query rows, then `d` gate rows) as
/// `(query, gate)`, each `[heads d, cols]`.
fn deinterleave<T: Copy>(v: &[T], heads: usize, d: usize, cols: usize) -> (Vec<T>, Vec<T>) {
    let block = d * cols;
    let (mut q, mut g) = (
        Vec::with_capacity(heads * block),
        Vec::with_capacity(heads * block),
    );
    for h in 0..heads {
        let at = h * 2 * block;
        q.extend_from_slice(&v[at..at + block]);
        g.extend_from_slice(&v[at + block..at + 2 * block]);
    }
    (q, g)
}

/// [`deinterleave`]'s inverse.
fn interleave(q: &[f32], g: &[f32], heads: usize, d: usize, cols: usize) -> Vec<f32> {
    let block = d * cols;
    let mut out = Vec::with_capacity(2 * heads * block);
    for h in 0..heads {
        out.extend_from_slice(&q[h * block..(h + 1) * block]);
        out.extend_from_slice(&g[h * block..(h + 1) * block]);
    }
    out
}

/// Largest decode buffer [`load_hf`] reuses within one tensor's read. It is
/// uncharged: the budget check covers the tensors, and the chunk is transient.
const READ_CHUNK_BYTES: usize = LE_READ_CHUNK_BYTES;

/// Where a tensor's values land in the output tensor, as one run of `len`
/// elements starting `first` elements into the stored tensor.
type Run = (usize, usize);

/// One output tensor's values, decoded from `runs` of the stored tensor `name`
/// in order, `READ_CHUNK_BYTES` at a time, straight into storage of
/// `storage`. A BF16 checkpoint read as `Bf16` is copied as stored; any other
/// source is widened to f32 (exactly, for bf16 and f16) and, for `Bf16`
/// storage, rounded once to nearest even. No tensor-sized buffer besides the
/// output exists.
fn load_runs(
    file: &SafeTensors<'_>,
    name: &str,
    storage: DType,
    shape: &[usize],
    runs: &[Run],
    budget: &Budget,
) -> Result<Tensor, OjasError> {
    const OP: &str = "qwen35::load_hf";
    let stored = file
        .info(name)
        .map_err(|e| refuse(OP, format!("{name}: {e}")))?
        .dtype;
    let width = match stored {
        StDtype::F32 => 4,
        StDtype::BF16 | StDtype::F16 => 2,
        other => {
            return Err(refuse(
                OP,
                format!("{name}: {other:?} is not a floating-point tensor this crate decodes"),
            ))
        }
    };
    let mut out = Tensor::zeros(shape, storage, budget)?;
    // The decode buffer is on the heap, sized to the tensor when it is
    // smaller than a chunk: a worker thread's stack may be 512 KiB.
    let want: usize = runs.iter().map(|&(_, len)| len).sum::<usize>() * width;
    let chunk_len = want.min(READ_CHUNK_BYTES);
    let mut chunk = Vec::new();
    chunk
        .try_reserve_exact(chunk_len)
        .map_err(|_| OjasError::CapacityExceeded {
            requested: chunk_len as u64,
            cap: budget.cap_bytes(),
            live: budget.live_bytes().unwrap_or(0),
        })?;
    chunk.resize(chunk_len, 0u8);
    let per_chunk = chunk_len / width;
    let mut at = 0usize;
    for &(first, len) in runs {
        let mut done = 0usize;
        while done < len {
            let n = (len - done).min(per_chunk);
            let bytes = &mut chunk[..n * width];
            let offset = u64::try_from((first + done) * width)
                .map_err(|_| refuse(OP, format!("{name}: offset exceeds u64")))?;
            file.read_into(name, offset, bytes)
                .map_err(|e| refuse(OP, format!("{name}: {e}")))?;
            let to = at + done..at + done + n;
            match storage {
                DType::F32 => {
                    let dst = &mut out.f32_slice_mut()?[to];
                    for (slot, b) in dst.iter_mut().zip(bytes.chunks_exact(width)) {
                        *slot = match stored {
                            StDtype::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                            StDtype::BF16 => bf16_to_f32(u16::from_le_bytes([b[0], b[1]])),
                            _ => f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
                        };
                    }
                }
                _ => {
                    let dst = &mut out.bf16_slice_mut()?[to];
                    for (slot, b) in dst.iter_mut().zip(bytes.chunks_exact(width)) {
                        *slot = match stored {
                            StDtype::F32 => {
                                f32_to_bf16(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            }
                            StDtype::BF16 => u16::from_le_bytes([b[0], b[1]]),
                            _ => f32_to_bf16(f16_to_f32(u16::from_le_bytes([b[0], b[1]]))),
                        };
                    }
                }
            }
            done += n;
        }
        at += len;
    }
    Ok(out)
}

/// The tower's parameters from a Hugging Face checkpoint, split as the
/// module docs say.
///
/// `linears` is the storage of every weight a `linear` reads (the gated
/// delta net and attention projections and the MLP), and `table` that of the
/// tied embedding: `F32`, widened exactly from bf16 or f16, or `Bf16`, a
/// frozen base held at 2 bytes per element. A `Bf16` weight is read from a
/// BF16 checkpoint as stored, never widened, and from an F32 or F16 one
/// rounded once to nearest even. A `Bf16` table feeds the gather and the
/// fused LM head, which read it in place
/// ([`ojas_core::Backend::bf16_operands`]'s tied-table lane: 1.02 GB instead
/// of 2.03 GB at 2B). Every norm weight, the convolutions, `A_log` and
/// `dt_bias` are `F32` either way: their ops do not take bf16 operands. Any
/// other `linears` or `table` dtype is [`OjasError::Dtype`].
///
/// Every name and shape of [`hf_tensors`] is checked from the header before
/// any tensor is read; a missing tensor or another shape is refused by name.
/// Tensors outside the tower (the vision tower, the MTP block) are not read.
pub fn load_hf(
    spec: &Qwen35Spec,
    file: &SafeTensors<'_>,
    prefix: &str,
    linears: DType,
    table: DType,
    budget: &Budget,
) -> Result<Qwen35Params<Tensor>, OjasError> {
    const OP: &str = "qwen35::load_hf";
    for storage in [linears, table] {
        if !matches!(storage, DType::F32 | DType::Bf16) {
            return Err(OjasError::Dtype {
                op: OP,
                expected: DType::Bf16,
                got: storage,
            });
        }
    }
    spec.validate()?;
    let names = hf_tensors(spec, prefix);
    let storage_of = |kind: HfKind| match kind {
        HfKind::Table => table,
        HfKind::Linear => linears,
        HfKind::Plain => DType::F32,
    };
    let mut total = 0u64;
    for want in &names {
        let info = file
            .info(&want.name)
            .map_err(|e| refuse(OP, format!("{}: {e}", want.name)))?;
        let got: Vec<usize> = info.shape.iter().map(|&n| n as usize).collect();
        if got != want.shape {
            return Err(refuse(
                OP,
                format!("{}: shape {got:?}, expected {:?}", want.name, want.shape),
            ));
        }
        if !matches!(info.dtype, StDtype::F32 | StDtype::BF16 | StDtype::F16) {
            return Err(refuse(
                OP,
                format!(
                    "{}: {:?} is not a floating-point tensor this crate decodes",
                    want.name, info.dtype
                ),
            ));
        }
        let bytes = u64::try_from(shape_product(&want.shape)?)
            .ok()
            .and_then(|n| n.checked_mul(storage_of(want.kind).size() as u64))
            .ok_or_else(|| refuse(OP, format!("{}: byte size overflows", want.name)))?;
        total = total
            .checked_add(bytes)
            .ok_or_else(|| refuse(OP, "total parameter bytes overflow".to_string()))?;
    }
    // Every tensor is decoded straight into its charged output, so the
    // charged total is the whole footprint: refuse now, before a byte is read.
    budget.check_room(total)?;
    let run = |name: &str, shape: &[usize], runs: &[Run]| {
        let kind = names.iter().find(|t| t.name == name).map(|t| t.kind);
        let storage = storage_of(kind.unwrap_or(HfKind::Plain));
        load_runs(file, name, storage, shape, runs, budget)
    };
    let whole = |name: String, shape: &[usize]| -> Result<Tensor, OjasError> {
        run(&name, shape, &[(0, shape_product(shape)?)])
    };
    let d = spec.hidden;
    let f = spec.intermediate;
    let embed = whole(names[0].name.clone(), &[spec.vocab, d])?;
    let mut layers = Vec::with_capacity(spec.layers.len());
    for (i, kind) in spec.layers.iter().enumerate() {
        let l = format!("{prefix}layers.{i}.");
        let input_norm = whole(format!("{l}input_layernorm.weight"), &[d])?;
        let mixer = match kind {
            Qwen35Mixer::GatedDeltaNet => {
                let m = format!("{l}linear_attn.");
                let chans = spec.gdn_qkv_channels();
                let (h, dv, k) = (spec.gdn_heads, spec.gdn_value_dim, spec.conv_width);
                let w = format!("{m}in_proj_qkv.weight");
                let c = format!("{m}conv1d.weight");
                // The fused weight's row blocks are runs of the stored tensor.
                let block = |cols: usize, i: usize| -> Run {
                    let before: usize = chans[..i].iter().sum();
                    (before * cols, chans[i] * cols)
                };
                MixerParams::GatedDeltaNet(GdnParams {
                    wq: run(&w, &[chans[0], d], &[block(d, 0)])?,
                    wk: run(&w, &[chans[1], d], &[block(d, 1)])?,
                    wv: run(&w, &[chans[2], d], &[block(d, 2)])?,
                    conv_q: run(&c, &[chans[0], k], &[block(k, 0)])?,
                    conv_k: run(&c, &[chans[1], k], &[block(k, 1)])?,
                    conv_v: run(&c, &[chans[2], k], &[block(k, 2)])?,
                    wz: whole(format!("{m}in_proj_z.weight"), &[h * dv, d])?,
                    wa: whole(format!("{m}in_proj_a.weight"), &[h, d])?,
                    wb: whole(format!("{m}in_proj_b.weight"), &[h, d])?,
                    a_log: whole(format!("{m}A_log"), &[h])?,
                    dt_bias: whole(format!("{m}dt_bias"), &[h])?,
                    norm: whole(format!("{m}norm.weight"), &[dv])?,
                    out: whole(format!("{m}out_proj.weight"), &[d, h * dv])?,
                })
            }
            Qwen35Mixer::Attention => {
                let m = format!("{l}self_attn.");
                let (hq, hkv, hd) = (spec.q_heads, spec.kv_heads, spec.head_dim);
                // Per head, `hd` query rows then `hd` gate rows.
                let q_proj = format!("{m}q_proj.weight");
                let rows = hd * d;
                let (query, gate): (Vec<Run>, Vec<Run>) = (0..hq)
                    .map(|h| ((h * 2 * rows, rows), (h * 2 * rows + rows, rows)))
                    .unzip();
                MixerParams::Attention(AttnParams {
                    wq: run(&q_proj, &[hq * hd, d], &query)?,
                    wgate: run(&q_proj, &[hq * hd, d], &gate)?,
                    wk: whole(format!("{m}k_proj.weight"), &[hkv * hd, d])?,
                    wv: whole(format!("{m}v_proj.weight"), &[hkv * hd, d])?,
                    q_norm: whole(format!("{m}q_norm.weight"), &[hd])?,
                    k_norm: whole(format!("{m}k_norm.weight"), &[hd])?,
                    wo: whole(format!("{m}o_proj.weight"), &[d, hq * hd])?,
                })
            }
        };
        layers.push(LayerParams {
            input_norm,
            mixer,
            post_norm: whole(format!("{l}post_attention_layernorm.weight"), &[d])?,
            gate: whole(format!("{l}mlp.gate_proj.weight"), &[f, d])?,
            up: whole(format!("{l}mlp.up_proj.weight"), &[f, d])?,
            down: whole(format!("{l}mlp.down_proj.weight"), &[d, f])?,
        });
    }
    let norm = whole(format!("{prefix}norm.weight"), &[d])?;
    Ok(Qwen35Params {
        embed,
        layers,
        norm,
    })
}

/// One tensor under its Hugging Face name, on the host.
#[derive(Clone, Debug)]
pub struct HfValues {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// `grads` (one tensor per parameter, wherever it lives) read to the host
/// and fused back into the Hugging Face tensors of [`hf_tensors`], in that
/// order: the inverse of [`load_hf`]'s split.
pub fn fuse_grads(
    spec: &Qwen35Spec,
    grads: &Qwen35Params<Tensor>,
    prefix: &str,
    budget: &Budget,
) -> Result<Vec<HfValues>, OjasError> {
    const OP: &str = "qwen35::fuse_grads";
    if grads.layers.len() != spec.layers.len() {
        return Err(refuse(
            OP,
            format!(
                "{} layers for a spec of {}",
                grads.layers.len(),
                spec.layers.len()
            ),
        ));
    }
    let host = |t: &Tensor| t.to_host(budget)?.to_f32_vec();
    let mut values: Vec<Vec<f32>> = Vec::new();
    values.push(host(&grads.embed)?);
    for (l, kind) in grads.layers.iter().zip(&spec.layers) {
        values.push(host(&l.input_norm)?);
        match (&l.mixer, kind) {
            (MixerParams::GatedDeltaNet(m), Qwen35Mixer::GatedDeltaNet) => {
                values.push([host(&m.wq)?, host(&m.wk)?, host(&m.wv)?].concat());
                values.push([host(&m.conv_q)?, host(&m.conv_k)?, host(&m.conv_v)?].concat());
                for t in [&m.wz, &m.wa, &m.wb, &m.a_log, &m.dt_bias, &m.norm, &m.out] {
                    values.push(host(t)?);
                }
            }
            (MixerParams::Attention(m), Qwen35Mixer::Attention) => {
                let (hq, hd) = (spec.q_heads, spec.head_dim);
                values.push(interleave(
                    &host(&m.wq)?,
                    &host(&m.wgate)?,
                    hq,
                    hd,
                    spec.hidden,
                ));
                for t in [&m.wk, &m.wv, &m.q_norm, &m.k_norm, &m.wo] {
                    values.push(host(t)?);
                }
            }
            _ => return Err(refuse(OP, "a layer's mixer is not the spec's".to_string())),
        }
        for t in [&l.post_norm, &l.gate, &l.up, &l.down] {
            values.push(host(t)?);
        }
    }
    values.push(host(&grads.norm)?);
    let table = hf_tensors(spec, prefix);
    if table.len() != values.len() {
        return Err(refuse(
            OP,
            format!("{} tensors for a table of {}", values.len(), table.len()),
        ));
    }
    table
        .into_iter()
        .zip(values)
        .map(|(t, values)| {
            let n: usize = t.shape.iter().product();
            if values.len() != n {
                return Err(refuse(
                    OP,
                    format!(
                        "{}: {} values for shape {:?}",
                        t.name,
                        values.len(),
                        t.shape
                    ),
                ));
            }
            Ok(HfValues {
                name: t.name,
                shape: t.shape,
                values,
            })
        })
        .collect()
}

/// The ten linears the LoRA recipe adapts, per layer kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoraKind {
    GdnQkv,
    GdnZ,
    GdnOut,
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    MlpGate,
    MlpUp,
    MlpDown,
}

/// Every [`LoraKind`], in the canonical order of [`Qwen35Lora::sites`]
/// within a layer's mixer, then the MLP.
pub const LORA_KINDS: [LoraKind; 10] = [
    LoraKind::GdnQkv,
    LoraKind::GdnZ,
    LoraKind::GdnOut,
    LoraKind::AttnQ,
    LoraKind::AttnK,
    LoraKind::AttnV,
    LoraKind::AttnO,
    LoraKind::MlpGate,
    LoraKind::MlpUp,
    LoraKind::MlpDown,
];

impl LoraKind {
    /// The Hugging Face module path under a layer, as the checkpoint's
    /// tensor index spells it (`<suffix>.weight` is the tensor).
    pub fn suffix(self) -> &'static str {
        match self {
            Self::GdnQkv => "linear_attn.in_proj_qkv",
            Self::GdnZ => "linear_attn.in_proj_z",
            Self::GdnOut => "linear_attn.out_proj",
            Self::AttnQ => "self_attn.q_proj",
            Self::AttnK => "self_attn.k_proj",
            Self::AttnV => "self_attn.v_proj",
            Self::AttnO => "self_attn.o_proj",
            Self::MlpGate => "mlp.gate_proj",
            Self::MlpUp => "mlp.up_proj",
            Self::MlpDown => "mlp.down_proj",
        }
    }

    pub fn from_suffix(suffix: &str) -> Option<Self> {
        LORA_KINDS.into_iter().find(|k| k.suffix() == suffix)
    }

    fn mixer(self) -> Option<Qwen35Mixer> {
        match self {
            Self::GdnQkv | Self::GdnZ | Self::GdnOut => Some(Qwen35Mixer::GatedDeltaNet),
            Self::AttnQ | Self::AttnK | Self::AttnV | Self::AttnO => Some(Qwen35Mixer::Attention),
            _ => None,
        }
    }
}

/// One adapted linear: where it is and the shape of its Hugging Face
/// tensor, which is one fused weight that the model holds as `out_parts`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoraTarget {
    /// The Hugging Face tensor, `<prefix>layers.<i>.<suffix>.weight`.
    pub name: String,
    pub layer: usize,
    pub kind: LoraKind,
    pub in_features: usize,
    /// Output rows of each part the model holds: `[q, k, v]` of
    /// `in_proj_qkv`, `[query, gate]` of `q_proj`, else one.
    pub out_parts: Vec<usize>,
}

impl LoraTarget {
    /// The adapter's module name, the tensor name without `.weight`.
    pub fn module(&self) -> &str {
        self.name.strip_suffix(".weight").unwrap_or(&self.name)
    }

    /// The fused `B` `[out, r]` from its parts, in the fused tensor's row
    /// order (`q_proj` interleaves query and gate rows per head).
    pub fn fuse_b(&self, spec: &Qwen35Spec, parts: &[Vec<f32>], rank: usize) -> Vec<f32> {
        match self.kind {
            LoraKind::AttnQ => interleave(&parts[0], &parts[1], spec.q_heads, spec.head_dim, rank),
            _ => parts.concat(),
        }
    }

    /// [`LoraTarget::fuse_b`]'s inverse.
    pub fn split_b(&self, spec: &Qwen35Spec, fused: &[f32], rank: usize) -> Vec<Vec<f32>> {
        match self.kind {
            LoraKind::AttnQ => {
                let (q, g) = deinterleave(fused, spec.q_heads, spec.head_dim, rank);
                vec![q, g]
            }
            _ => split_rows(fused, rank, &self.out_parts),
        }
    }

    /// This target's merged export weight: the base tensor [`LoraTarget::name`]
    /// of `file` plus `config`'s scaling times the fused `B A` of `site`,
    /// summed in f32 and rounded once to nearest even to a bf16 tensor in the
    /// Hugging Face layout (`[out, in]`, `in_proj_qkv` as `[q | k | v]`,
    /// `q_proj` with query and gate rows interleaved per head). `B` is
    /// [`LoraTarget::fuse_b`] of the site's parts, exactly the inverse of the
    /// split the adapter trains under. See [`crate::lora::merged_weight_bf16`]
    /// for the arithmetic, the refusals and the charge.
    ///
    /// The base is read once into a charged tensor: as stored from a BF16
    /// file, widened exactly from an F32 or F16 one, so the sum has one
    /// rounding whatever the file's dtype (for the quantised arm the file's
    /// bf16 weight is `dequant(Q(W))`). That tensor and the fused `B` are
    /// freed before this returns. The host copies of the `B` parts are
    /// `out * r` f32, uncharged, and gone before the base is read.
    pub fn merged_bf16<B: Backend>(
        &self,
        spec: &Qwen35Spec,
        site: &LoraSite<Tensor>,
        config: &LoraConfig,
        file: &SafeTensors<'_>,
        backend: &B,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "qwen35::merged_bf16";
        let rank = config.rank();
        let out: usize = self.out_parts.iter().sum();
        if site.b.len() != self.out_parts.len() {
            return Err(refuse(
                OP,
                format!(
                    "{}: {} B parts for {} output parts",
                    self.name,
                    site.b.len(),
                    self.out_parts.len()
                ),
            ));
        }
        for (b, &rows) in site.b.iter().zip(&self.out_parts) {
            if b.shape() != [rows, rank] {
                return Err(refuse(
                    OP,
                    format!(
                        "{}: B part {:?}, expected [{rows}, {rank}]",
                        self.name,
                        b.shape()
                    ),
                ));
            }
        }
        let query_rows = spec.q_heads * spec.head_dim;
        if self.kind == LoraKind::AttnQ && self.out_parts.iter().any(|&rows| rows != query_rows) {
            return Err(refuse(
                OP,
                format!(
                    "{}: parts {:?} are not the spec's {query_rows} rows each",
                    self.name, self.out_parts
                ),
            ));
        }
        let info = file
            .info(&self.name)
            .map_err(|e| refuse(OP, format!("{}: {e}", self.name)))?;
        let got: Vec<usize> = info.shape.iter().map(|&n| n as usize).collect();
        if got != [out, self.in_features] {
            return Err(refuse(
                OP,
                format!(
                    "{}: shape {got:?}, expected [{out}, {}]",
                    self.name, self.in_features
                ),
            ));
        }
        let storage = match info.dtype {
            StDtype::BF16 => DType::Bf16,
            _ => DType::F32,
        };
        let budget = backend.budget().clone();
        let fused_b = {
            let parts = site
                .b
                .iter()
                .map(Tensor::to_f32_vec)
                .collect::<Result<Vec<_>, _>>()?;
            Tensor::from_f32(&self.fuse_b(spec, &parts, rank), &[out, rank], &budget)?
        };
        let weight = load_runs(
            file,
            &self.name,
            storage,
            &[out, self.in_features],
            &[(0, out * self.in_features)],
            &budget,
        )?;
        Ok(crate::lora::merged_weight_bf16(
            config, &weight, &site.a, &fused_b, backend,
        )?)
    }
}

/// Every [`LoraTarget`] of `spec` for `kinds`, layer by layer in the order
/// of [`Qwen35Lora::sites`], checked against the checkpoint's tensor index:
/// each name must be present with the shape the spec gives. A selected
/// target the index lacks, or has with another shape, is refused by name;
/// nothing is read from tensor data.
pub fn lora_targets(
    spec: &Qwen35Spec,
    file: &SafeTensors<'_>,
    prefix: &str,
    kinds: &[LoraKind],
) -> Result<Vec<LoraTarget>, OjasError> {
    const OP: &str = "qwen35::lora_targets";
    spec.validate()?;
    if kinds.is_empty() {
        return Err(refuse(OP, "no LoRA target selected".to_string()));
    }
    let (d, f) = (spec.hidden, spec.intermediate);
    let [cq, ck, cv] = spec.gdn_qkv_channels();
    let (hq, hkv, hd) = (spec.q_heads, spec.kv_heads, spec.head_dim);
    let hv = spec.gdn_heads * spec.gdn_value_dim;
    let mut out = Vec::new();
    for (layer, mixer) in spec.layers.iter().enumerate() {
        for kind in LORA_KINDS {
            if !kinds.contains(&kind) || kind.mixer().is_some_and(|m| m != *mixer) {
                continue;
            }
            let (inn, parts): (usize, Vec<usize>) = match kind {
                LoraKind::GdnQkv => (d, vec![cq, ck, cv]),
                LoraKind::GdnZ => (d, vec![hv]),
                LoraKind::GdnOut => (hv, vec![d]),
                LoraKind::AttnQ => (d, vec![hq * hd, hq * hd]),
                LoraKind::AttnK | LoraKind::AttnV => (d, vec![hkv * hd]),
                LoraKind::AttnO => (hq * hd, vec![d]),
                LoraKind::MlpGate | LoraKind::MlpUp => (d, vec![f]),
                LoraKind::MlpDown => (f, vec![d]),
            };
            let name = format!("{prefix}layers.{layer}.{}.weight", kind.suffix());
            let info = file.info(&name).map_err(|e| {
                refuse(
                    OP,
                    format!("LoRA target {name} is not in the checkpoint: {e}"),
                )
            })?;
            if !matches!(
                info.dtype,
                ojas_io::StDtype::F32 | ojas_io::StDtype::BF16 | ojas_io::StDtype::F16
            ) {
                return Err(refuse(
                    OP,
                    format!(
                        "LoRA target {name} is stored as {:?}, not a float weight the loader reads",
                        info.dtype
                    ),
                ));
            }
            let got: Vec<usize> = info.shape.iter().map(|&n| n as usize).collect();
            let want = [parts.iter().sum::<usize>(), inn];
            if got != want {
                return Err(refuse(
                    OP,
                    format!("LoRA target {name}: shape {got:?}, expected {want:?}"),
                ));
            }
            out.push(LoraTarget {
                name,
                layer,
                kind,
                in_features: inn,
                out_parts: parts,
            });
        }
    }
    Ok(out)
}

/// One slot per adapted linear, `None` where a linear is not adapted. `S`
/// is whatever sits in a slot: a [`crate::lora::LoraSite`] of host tensors
/// or of graph values, or one mask per site.
#[derive(Clone, Debug)]
pub struct Qwen35Lora<S> {
    pub layers: Vec<LayerLora<S>>,
}

#[derive(Clone, Debug)]
pub struct LayerLora<S> {
    pub mixer: MixerLora<S>,
    pub gate: Option<S>,
    pub up: Option<S>,
    pub down: Option<S>,
}

#[derive(Clone, Debug)]
pub enum MixerLora<S> {
    GatedDeltaNet {
        qkv: Option<S>,
        z: Option<S>,
        out: Option<S>,
    },
    Attention {
        q: Option<S>,
        k: Option<S>,
        v: Option<S>,
        o: Option<S>,
    },
}

impl<S> Qwen35Lora<S> {
    /// Every slot empty, for `spec`.
    pub fn empty(spec: &Qwen35Spec) -> Self {
        let layers = spec
            .layers
            .iter()
            .map(|m| LayerLora {
                mixer: match m {
                    Qwen35Mixer::GatedDeltaNet => MixerLora::GatedDeltaNet {
                        qkv: None,
                        z: None,
                        out: None,
                    },
                    Qwen35Mixer::Attention => MixerLora::Attention {
                        q: None,
                        k: None,
                        v: None,
                        o: None,
                    },
                },
                gate: None,
                up: None,
                down: None,
            })
            .collect();
        Self { layers }
    }

    /// The slot of `kind` in `layer`, or `None` if that layer has no such
    /// linear.
    pub fn slot_mut(&mut self, layer: usize, kind: LoraKind) -> Option<&mut Option<S>> {
        let l = self.layers.get_mut(layer)?;
        Some(match (&mut l.mixer, kind) {
            (MixerLora::GatedDeltaNet { qkv, .. }, LoraKind::GdnQkv) => qkv,
            (MixerLora::GatedDeltaNet { z, .. }, LoraKind::GdnZ) => z,
            (MixerLora::GatedDeltaNet { out, .. }, LoraKind::GdnOut) => out,
            (MixerLora::Attention { q, .. }, LoraKind::AttnQ) => q,
            (MixerLora::Attention { k, .. }, LoraKind::AttnK) => k,
            (MixerLora::Attention { v, .. }, LoraKind::AttnV) => v,
            (MixerLora::Attention { o, .. }, LoraKind::AttnO) => o,
            (_, LoraKind::MlpGate) => &mut l.gate,
            (_, LoraKind::MlpUp) => &mut l.up,
            (_, LoraKind::MlpDown) => &mut l.down,
            _ => return None,
        })
    }

    /// [`Qwen35Lora::slot_mut`] by shared reference.
    pub fn slot(&self, layer: usize, kind: LoraKind) -> Option<&Option<S>> {
        let l = self.layers.get(layer)?;
        Some(match (&l.mixer, kind) {
            (MixerLora::GatedDeltaNet { qkv, .. }, LoraKind::GdnQkv) => qkv,
            (MixerLora::GatedDeltaNet { z, .. }, LoraKind::GdnZ) => z,
            (MixerLora::GatedDeltaNet { out, .. }, LoraKind::GdnOut) => out,
            (MixerLora::Attention { q, .. }, LoraKind::AttnQ) => q,
            (MixerLora::Attention { k, .. }, LoraKind::AttnK) => k,
            (MixerLora::Attention { v, .. }, LoraKind::AttnV) => v,
            (MixerLora::Attention { o, .. }, LoraKind::AttnO) => o,
            (_, LoraKind::MlpGate) => &l.gate,
            (_, LoraKind::MlpUp) => &l.up,
            (_, LoraKind::MlpDown) => &l.down,
            _ => return None,
        })
    }

    /// Every filled slot, layer by layer: a mixer's slots in
    /// [`LORA_KINDS`] order, then gate, up, down. This is the order of
    /// [`lora_targets`].
    pub fn sites(&self) -> Vec<&S> {
        let mut out = Vec::new();
        for l in &self.layers {
            match &l.mixer {
                MixerLora::GatedDeltaNet { qkv, z, out: o } => out.extend([qkv, z, o]),
                MixerLora::Attention { q, k, v, o } => out.extend([q, k, v, o]),
            }
            out.extend([&l.gate, &l.up, &l.down]);
        }
        out.into_iter().flatten().collect()
    }

    /// `f` of every filled slot, in [`Qwen35Lora::sites`] order.
    pub fn try_map<T, E>(
        &self,
        f: &mut impl FnMut(&S) -> Result<T, E>,
    ) -> Result<Qwen35Lora<T>, E> {
        let mut m = |s: &Option<S>| s.as_ref().map(&mut *f).transpose();
        let mut layers = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            let mixer = match &l.mixer {
                MixerLora::GatedDeltaNet { qkv, z, out } => MixerLora::GatedDeltaNet {
                    qkv: m(qkv)?,
                    z: m(z)?,
                    out: m(out)?,
                },
                MixerLora::Attention { q, k, v, o } => MixerLora::Attention {
                    q: m(q)?,
                    k: m(k)?,
                    v: m(v)?,
                    o: m(o)?,
                },
            };
            layers.push(LayerLora {
                mixer,
                gate: m(&l.gate)?,
                up: m(&l.up)?,
                down: m(&l.down)?,
            });
        }
        Ok(Qwen35Lora { layers })
    }
}

impl Qwen35Lora<LoraSite<Tensor>> {
    /// A fresh adapter over `targets`: `A` from a [`CounterRng`] seeded
    /// `seed ^ fnv1a(module)` (a site's values depend on the seed and its
    /// name only), every `B` zero.
    pub fn init(
        spec: &Qwen35Spec,
        targets: &[LoraTarget],
        config: &LoraConfig,
        seed: u64,
        budget: &Budget,
    ) -> Result<Self, OjasError> {
        let mut lora = Self::empty(spec);
        for t in targets {
            let mut rng = CounterRng::new(seed ^ fnv1a(t.module().as_bytes()));
            let site = LoraSite::init(config, t.in_features, &t.out_parts, &mut rng, budget)?;
            let slot = lora
                .slot_mut(t.layer, t.kind)
                .ok_or_else(|| refuse("qwen35::lora_init", format!("{}: no such slot", t.name)))?;
            *slot = Some(site);
        }
        Ok(lora)
    }

    /// [`LoraTarget::merged_bf16`] for every target, in `targets` order, each
    /// handed to `sink` under its Hugging Face name and then dropped by the
    /// sink or kept by it: a sink that writes and drops holds one merged
    /// weight at a time. Nothing in `self` or `file` changes.
    ///
    /// Refused before anything is read: a layer count other than the spec's,
    /// a duplicate target, a target whose slot is empty, and an adapter in a
    /// slot no target names (it would be silently left out of the export). A
    /// failure, or an error from `sink`, stops the run; tensors already
    /// handed to `sink` stay with it.
    pub fn merge_bf16<B: Backend>(
        &self,
        spec: &Qwen35Spec,
        targets: &[LoraTarget],
        config: &LoraConfig,
        file: &SafeTensors<'_>,
        backend: &B,
        sink: &mut impl FnMut(&str, Tensor) -> Result<(), OjasError>,
    ) -> Result<(), OjasError> {
        const OP: &str = "qwen35::merge_bf16";
        if self.layers.len() != spec.layers.len() {
            return Err(refuse(
                OP,
                format!(
                    "{} adapter layers for a spec of {}",
                    self.layers.len(),
                    spec.layers.len()
                ),
            ));
        }
        let mut sites = Vec::with_capacity(targets.len());
        for (i, t) in targets.iter().enumerate() {
            if targets[..i].iter().any(|u| u.name == t.name) {
                return Err(refuse(OP, format!("{} is listed twice", t.name)));
            }
            let site = self
                .slot(t.layer, t.kind)
                .and_then(Option::as_ref)
                .ok_or_else(|| refuse(OP, format!("{}: its adapter slot is empty", t.name)))?;
            sites.push(site);
        }
        if self.sites().len() != targets.len() {
            return Err(refuse(
                OP,
                format!(
                    "{} adapters for {} targets: an adapter would be left out",
                    self.sites().len(),
                    targets.len()
                ),
            ));
        }
        for (t, site) in targets.iter().zip(sites) {
            let merged = t.merged_bf16(spec, site, config, file, backend)?;
            sink(&t.name, merged)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleave_inverts_deinterleave() {
        let (heads, d, cols) = (3, 2, 5);
        let v: Vec<f32> = (0..heads * 2 * d * cols).map(|i| i as f32).collect();
        let (q, g) = deinterleave(&v, heads, d, cols);
        // Head 1's query rows start at row 2 d of the fused tensor.
        assert_eq!(q[d * cols], (2 * d * cols) as f32);
        assert_eq!(g[0], (d * cols) as f32);
        assert_eq!(interleave(&q, &g, heads, d, cols), v);
    }

    #[test]
    fn split_rows_cuts_consecutive_blocks() {
        let v: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let parts = split_rows(&v, 2, &[1, 3, 2]);
        assert_eq!(
            parts,
            vec![
                vec![0., 1.],
                vec![2., 3., 4., 5., 6., 7.],
                vec![8., 9., 10., 11.]
            ]
        );
    }
}
