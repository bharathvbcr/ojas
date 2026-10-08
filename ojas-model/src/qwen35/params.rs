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

use ojas_core::{Budget, OjasError, Tensor};
use ojas_io::SafeTensors;

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

/// One Hugging Face tensor of the tower: its name and stored shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfTensor {
    pub name: String,
    pub shape: Vec<usize>,
}

/// Every Hugging Face tensor the tower reads under `prefix` (`"model."` for
/// a text-only checkpoint, `"model.language_model."` under a `text_config`),
/// in [`Qwen35Params::try_map`]'s order of the fused tensors.
pub fn hf_tensors(spec: &Qwen35Spec, prefix: &str) -> Vec<HfTensor> {
    let t = |name: String, shape: &[usize]| HfTensor {
        name,
        shape: shape.to_vec(),
    };
    let (d, f) = (spec.hidden, spec.intermediate);
    let mut out = vec![t(format!("{prefix}embed_tokens.weight"), &[spec.vocab, d])];
    for (i, kind) in spec.layers.iter().enumerate() {
        let l = format!("{prefix}layers.{i}.");
        out.push(t(format!("{l}input_layernorm.weight"), &[d]));
        match kind {
            Qwen35Mixer::GatedDeltaNet => {
                let [cq, ck, cv] = spec.gdn_qkv_channels();
                let c = cq + ck + cv;
                let (h, dv) = (spec.gdn_heads, spec.gdn_value_dim);
                let m = format!("{l}linear_attn.");
                out.push(t(format!("{m}in_proj_qkv.weight"), &[c, d]));
                out.push(t(format!("{m}conv1d.weight"), &[c, 1, spec.conv_width]));
                out.push(t(format!("{m}in_proj_z.weight"), &[h * dv, d]));
                out.push(t(format!("{m}in_proj_a.weight"), &[h, d]));
                out.push(t(format!("{m}in_proj_b.weight"), &[h, d]));
                out.push(t(format!("{m}A_log"), &[h]));
                out.push(t(format!("{m}dt_bias"), &[h]));
                out.push(t(format!("{m}norm.weight"), &[dv]));
                out.push(t(format!("{m}out_proj.weight"), &[d, h * dv]));
            }
            Qwen35Mixer::Attention => {
                let (hq, hkv, hd) = (spec.q_heads, spec.kv_heads, spec.head_dim);
                let m = format!("{l}self_attn.");
                out.push(t(format!("{m}q_proj.weight"), &[hq * 2 * hd, d]));
                out.push(t(format!("{m}k_proj.weight"), &[hkv * hd, d]));
                out.push(t(format!("{m}v_proj.weight"), &[hkv * hd, d]));
                out.push(t(format!("{m}q_norm.weight"), &[hd]));
                out.push(t(format!("{m}k_norm.weight"), &[hd]));
                out.push(t(format!("{m}o_proj.weight"), &[d, hq * hd]));
            }
        }
        out.push(t(format!("{l}post_attention_layernorm.weight"), &[d]));
        out.push(t(format!("{l}mlp.gate_proj.weight"), &[f, d]));
        out.push(t(format!("{l}mlp.up_proj.weight"), &[f, d]));
        out.push(t(format!("{l}mlp.down_proj.weight"), &[d, f]));
    }
    out.push(t(format!("{prefix}norm.weight"), &[d]));
    out
}

fn refuse(op: &'static str, detail: String) -> OjasError {
    OjasError::Shape { op, detail }
}

/// `rows` rows of `cols` values cut into consecutive blocks of `counts`
/// rows.
fn split_rows(v: &[f32], cols: usize, counts: &[usize]) -> Vec<Vec<f32>> {
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
fn deinterleave(v: &[f32], heads: usize, d: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
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

/// The tower's parameters from a Hugging Face checkpoint, each widened to
/// f32 (bf16 and f16 exactly) and split as the module docs say.
///
/// Every name and shape of [`hf_tensors`] is checked from the header before
/// any tensor is read; a missing tensor or another shape is refused by name.
/// Tensors outside the tower (the vision tower, the MTP block) are not read.
pub fn load_hf(
    spec: &Qwen35Spec,
    file: &SafeTensors<'_>,
    prefix: &str,
    budget: &Budget,
) -> Result<Qwen35Params<Tensor>, OjasError> {
    const OP: &str = "qwen35::load_hf";
    spec.validate()?;
    let table = hf_tensors(spec, prefix);
    for want in &table {
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
    }
    let read = |name: &str| -> Result<Vec<f32>, OjasError> {
        file.read_f32_widened(name)
            .map(|(_, v)| v)
            .map_err(|e| refuse(OP, format!("{name}: {e}")))
    };
    let new = |v: &[f32], shape: &[usize]| Tensor::from_f32(v, shape, budget);
    let d = spec.hidden;
    let f = spec.intermediate;
    let embed = new(&read(&table[0].name)?, &[spec.vocab, d])?;
    let mut layers = Vec::with_capacity(spec.layers.len());
    for (i, kind) in spec.layers.iter().enumerate() {
        let l = format!("{prefix}layers.{i}.");
        let input_norm = new(&read(&format!("{l}input_layernorm.weight"))?, &[d])?;
        let mixer = match kind {
            Qwen35Mixer::GatedDeltaNet => {
                let m = format!("{l}linear_attn.");
                let chans = spec.gdn_qkv_channels();
                let (h, dv, k) = (spec.gdn_heads, spec.gdn_value_dim, spec.conv_width);
                let w = split_rows(&read(&format!("{m}in_proj_qkv.weight"))?, d, &chans);
                let c = split_rows(&read(&format!("{m}conv1d.weight"))?, k, &chans);
                MixerParams::GatedDeltaNet(GdnParams {
                    wq: new(&w[0], &[chans[0], d])?,
                    wk: new(&w[1], &[chans[1], d])?,
                    wv: new(&w[2], &[chans[2], d])?,
                    conv_q: new(&c[0], &[chans[0], k])?,
                    conv_k: new(&c[1], &[chans[1], k])?,
                    conv_v: new(&c[2], &[chans[2], k])?,
                    wz: new(&read(&format!("{m}in_proj_z.weight"))?, &[h * dv, d])?,
                    wa: new(&read(&format!("{m}in_proj_a.weight"))?, &[h, d])?,
                    wb: new(&read(&format!("{m}in_proj_b.weight"))?, &[h, d])?,
                    a_log: new(&read(&format!("{m}A_log"))?, &[h])?,
                    dt_bias: new(&read(&format!("{m}dt_bias"))?, &[h])?,
                    norm: new(&read(&format!("{m}norm.weight"))?, &[dv])?,
                    out: new(&read(&format!("{m}out_proj.weight"))?, &[d, h * dv])?,
                })
            }
            Qwen35Mixer::Attention => {
                let m = format!("{l}self_attn.");
                let (hq, hkv, hd) = (spec.q_heads, spec.kv_heads, spec.head_dim);
                let (q, g) = deinterleave(&read(&format!("{m}q_proj.weight"))?, hq, hd, d);
                MixerParams::Attention(AttnParams {
                    wq: new(&q, &[hq * hd, d])?,
                    wgate: new(&g, &[hq * hd, d])?,
                    wk: new(&read(&format!("{m}k_proj.weight"))?, &[hkv * hd, d])?,
                    wv: new(&read(&format!("{m}v_proj.weight"))?, &[hkv * hd, d])?,
                    q_norm: new(&read(&format!("{m}q_norm.weight"))?, &[hd])?,
                    k_norm: new(&read(&format!("{m}k_norm.weight"))?, &[hd])?,
                    wo: new(&read(&format!("{m}o_proj.weight"))?, &[d, hq * hd])?,
                })
            }
        };
        layers.push(LayerParams {
            input_norm,
            mixer,
            post_norm: new(&read(&format!("{l}post_attention_layernorm.weight"))?, &[d])?,
            gate: new(&read(&format!("{l}mlp.gate_proj.weight"))?, &[f, d])?,
            up: new(&read(&format!("{l}mlp.up_proj.weight"))?, &[f, d])?,
            down: new(&read(&format!("{l}mlp.down_proj.weight"))?, &[d, f])?,
        });
    }
    let norm = new(&read(&format!("{prefix}norm.weight"))?, &[d])?;
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
