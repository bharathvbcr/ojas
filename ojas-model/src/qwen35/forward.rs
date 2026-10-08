//! The Qwen3.5 text tower over [`Graph`], to the fused tied-head loss.
//!
//! Each layer follows transformers' `Qwen3_5DecoderLayer`:
//! `x += mixer(norm1(x))`, `x += mlp(norm2(x))`, with every `Qwen3_5RMSNorm`
//! (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, the
//! final `norm`) scaling by `1 + w`. That sum is formed on the graph once per
//! weight, before any layer runs ([`Graph::add`] of the weight and a ones
//! constant), so its gradient reaches `w` unchanged and the parameter keeps
//! Hugging Face's value, weight decay and name.
//!
//! The gated delta net (`Qwen3_5GatedDeltaNet`): `q`, `k`, `v` projections
//! each through the depthwise causal conv + SiLU; `beta = sigmoid(b)`, `g =
//! -exp(A_log) softplus(a + dt_bias)`; the gated delta rule (which takes the
//! `l2norm` of `q` and `k` and scales `q` by `Dk^-1/2` itself); the gated
//! RMSNorm with gate `z` (a plain weight); `out_proj`.
//!
//! Gated attention (`Qwen3_5Attention`): query and gate rows of `q_proj`; the
//! per-head `1 + w` RMSNorm of `q` and `k`; partial RoPE on the leading
//! `rotary_dim` of each head (text-only MRoPE is plain partial RoPE); causal
//! grouped-query SDPA; `attn * sigmoid(gate)`; `o_proj`.
//!
//! The MLP is SwiGLU `down(silu(gate(h)) * up(h))`.

use ojas_core::{mrope_text_tables, Backend, Budget, CeChunk, MropeSection, OjasError, Tensor};

use crate::block::ActivationCheckpoint;
use crate::graph::Graph;
use crate::qwen35::params::{AttnParams, GdnParams, LayerParams, MixerParams, Qwen35Params};
use crate::qwen35::spec::Qwen35Spec;

/// `[B, T, H, D]` to `[B, H, T, D]`; it is its own inverse.
const SWAP_TIME_HEADS: [usize; 4] = [0, 2, 1, 3];

/// The host constants a forward reads: the partial RoPE tables for
/// positions `0..len` and the ones the `1 + w` norms add.
#[derive(Clone, Debug)]
pub struct Qwen35Tables {
    /// `[len, rotary_dim]`.
    pub cos: Tensor,
    pub sin: Tensor,
    /// `[hidden]` and `[head_dim]` ones.
    pub ones_hidden: Tensor,
    pub ones_head: Tensor,
}

impl Qwen35Tables {
    /// Tables for positions `0..len`. Text-only MRoPE collapses to plain
    /// partial RoPE ([`mrope_text_tables`]), whatever the section, so the
    /// table is built with every frequency on the time stream.
    pub fn new(spec: &Qwen35Spec, len: usize, budget: &Budget) -> Result<Self, OjasError> {
        spec.validate()?;
        let plain = MropeSection {
            section: [spec.rotary_dim / 2, 0, 0],
            interleaved: false,
        };
        let (cos, sin) =
            mrope_text_tables(0, len, plain, spec.rotary_dim, spec.rope_theta, budget)?;
        Ok(Self {
            cos,
            sin,
            ones_hidden: Tensor::from_f32(&vec![1.0; spec.hidden], &[spec.hidden], budget)?,
            ones_head: Tensor::from_f32(&vec![1.0; spec.head_dim], &[spec.head_dim], budget)?,
        })
    }

    /// The same tables resident on `backend`, uploaded once.
    pub fn upload<B: Backend + ?Sized>(&self, backend: &B) -> Result<Self, OjasError> {
        Ok(Self {
            cos: backend.upload(&self.cos)?,
            sin: backend.upload(&self.sin)?,
            ones_hidden: backend.upload(&self.ones_hidden)?,
            ones_head: backend.upload(&self.ones_head)?,
        })
    }

    fn len(&self) -> usize {
        self.cos.shape()[0]
    }
}

/// Every `1 + w` norm weight of a layer, formed on the graph.
struct LayerNorms<V> {
    input: V,
    post: V,
    /// `(q_norm, k_norm)` of an attention layer.
    qk: Option<(V, V)>,
}

/// `params` brought into `g` (on a tape, each is a leaf whose gradient
/// [`ojas_autograd::Tape::grad`] returns), in [`Qwen35Params::try_map`]'s
/// order.
pub fn bind<G: Graph>(
    g: &mut G,
    params: &Qwen35Params<Tensor>,
) -> Result<Qwen35Params<G::V>, OjasError> {
    params.try_map(&mut |t| g.param(t))
}

fn shape(op: &'static str, detail: String) -> OjasError {
    OjasError::Shape { op, detail }
}

/// Mean next-token cross-entropy through the fused tied head: a rank-0 loss.
/// `ids` and `targets` are `U32` `[B, T]` with `T` at most the tables'
/// length; transformers' `labels=ids` is `ids[.., ..T - 1]` against
/// `ids[.., 1..]`. `activations` makes each layer a checkpointed segment
/// ([`ActivationCheckpoint::Blocks`]); it changes memory, not values.
#[allow(clippy::too_many_arguments)]
pub fn forward_loss<G: Graph>(
    g: &mut G,
    spec: &Qwen35Spec,
    params: &Qwen35Params<G::V>,
    ids: &Tensor,
    targets: &Tensor,
    tables: &Qwen35Tables,
    chunk: CeChunk,
    activations: ActivationCheckpoint,
) -> Result<G::V, OjasError> {
    const OP: &str = "qwen35::forward_loss";
    spec.validate()?;
    let &[batch, seq] = ids.shape() else {
        return Err(shape(
            OP,
            format!("ids {:?} are not [batch, time]", ids.shape()),
        ));
    };
    if targets.shape() != ids.shape() {
        return Err(shape(
            OP,
            format!("targets {:?} for ids {:?}", targets.shape(), ids.shape()),
        ));
    }
    if seq == 0 || seq > tables.len() {
        return Err(shape(
            OP,
            format!("{seq} positions for tables of {}", tables.len()),
        ));
    }
    if params.layers.len() != spec.layers.len() {
        return Err(shape(
            OP,
            format!(
                "{} layers for a spec of {}",
                params.layers.len(),
                spec.layers.len()
            ),
        ));
    }
    // Rows `0..seq` of the tables: a view, wherever they live.
    let r = spec.rotary_dim;
    let rows = |t: &Tensor| t.narrow(0, &[seq, r], &[r, 1]);
    let (cos, sin) = (rows(&tables.cos)?, rows(&tables.sin)?);
    // Every `1 + w` before any layer: a checkpointed segment refuses a leaf
    // recorded inside it, and the ones are leaves on a tape.
    let one_d = g.param(&tables.ones_hidden)?;
    let one_h = g.param(&tables.ones_head)?;
    let mut norms = Vec::with_capacity(params.layers.len());
    for l in &params.layers {
        let qk = match &l.mixer {
            MixerParams::Attention(m) => {
                Some((g.add(&m.q_norm, &one_h)?, g.add(&m.k_norm, &one_h)?))
            }
            MixerParams::GatedDeltaNet(_) => None,
        };
        norms.push(LayerNorms {
            input: g.add(&l.input_norm, &one_d)?,
            post: g.add(&l.post_norm, &one_d)?,
            qk,
        });
    }
    let final_norm = g.add(&params.norm, &one_d)?;

    let mut x = g.embedding(&params.embed, ids)?;
    let dims = Dims {
        batch,
        seq,
        cos: &cos,
        sin: &sin,
    };
    for (l, n) in params.layers.iter().zip(&norms) {
        x = match activations {
            ActivationCheckpoint::Off => layer(g, spec, l, n, &x, &dims)?,
            ActivationCheckpoint::Blocks => {
                let mut outs =
                    g.checkpoint(|g| layer(g, spec, l, n, &x, &dims).map(|y| vec![y]))?;
                outs.pop()
                    .ok_or_else(|| shape(OP, "a checkpointed layer returned no output".into()))?
            }
        };
    }
    let h = g.rms_norm(&x, &final_norm, spec.eps)?;
    let rows = batch * seq;
    let h = g.reshape(&h, &[rows, spec.hidden])?;
    let targets = targets.reshape(&[rows])?;
    g.lin_ce(&h, &params.embed, &targets, None, chunk)
}

struct Dims<'a> {
    batch: usize,
    seq: usize,
    cos: &'a Tensor,
    sin: &'a Tensor,
}

fn layer<G: Graph>(
    g: &mut G,
    spec: &Qwen35Spec,
    l: &LayerParams<G::V>,
    n: &LayerNorms<G::V>,
    x: &G::V,
    d: &Dims<'_>,
) -> Result<G::V, OjasError> {
    let h = g.rms_norm(x, &n.input, spec.eps)?;
    let mixed = match (&l.mixer, &n.qk) {
        (MixerParams::GatedDeltaNet(m), None) => gated_delta_net(g, spec, m, &h, d)?,
        (MixerParams::Attention(m), Some((qn, kn))) => attention(g, spec, m, qn, kn, &h, d)?,
        _ => {
            return Err(shape(
                "qwen35::layer",
                "a layer's norms do not match its mixer".into(),
            ))
        }
    };
    let x = g.add(x, &mixed)?;
    let h = g.rms_norm(&x, &n.post, spec.eps)?;
    let gate = g.linear(&h, &l.gate)?;
    let gate = g.silu(&gate)?;
    let up = g.linear(&h, &l.up)?;
    let act = g.mul(&gate, &up)?;
    let down = g.linear(&act, &l.down)?;
    g.add(&x, &down)
}

fn gated_delta_net<G: Graph>(
    g: &mut G,
    spec: &Qwen35Spec,
    m: &GdnParams<G::V>,
    h: &G::V,
    d: &Dims<'_>,
) -> Result<G::V, OjasError> {
    let (b, t, heads) = (d.batch, d.seq, spec.gdn_heads);
    let (dk, dv) = (spec.gdn_key_dim, spec.gdn_value_dim);
    let mut branch = |w: &G::V, conv: &G::V, dim: usize| -> Result<G::V, OjasError> {
        let y = g.linear(h, w)?;
        let y = g.conv1d_silu(&y, conv)?;
        g.reshape(&y, &[b, t, heads, dim])
    };
    let q = branch(&m.wq, &m.conv_q, dk)?;
    let k = branch(&m.wk, &m.conv_k, dk)?;
    let v = branch(&m.wv, &m.conv_v, dv)?;
    let z = g.linear(h, &m.wz)?;
    let z = g.reshape(&z, &[b, t, heads, dv])?;
    let beta = g.linear(h, &m.wb)?;
    let beta = g.sigmoid(&beta)?;
    let a = g.linear(h, &m.wa)?;
    let decay = g.gdn_log_decay(&a, &m.a_log, &m.dt_bias)?;
    let o = g.gdn(&q, &k, &v, &decay, &beta)?;
    let o = g.gated_rms_norm(&o, &z, &m.norm, spec.eps)?;
    let o = g.reshape(&o, &[b, t, heads * dv])?;
    g.linear(&o, &m.out)
}

fn attention<G: Graph>(
    g: &mut G,
    spec: &Qwen35Spec,
    m: &AttnParams<G::V>,
    q_norm: &G::V,
    k_norm: &G::V,
    h: &G::V,
    d: &Dims<'_>,
) -> Result<G::V, OjasError> {
    let (b, t, hd) = (d.batch, d.seq, spec.head_dim);
    let mut head = |w: &G::V, heads: usize, norm: Option<&G::V>| -> Result<G::V, OjasError> {
        let y = g.linear(h, w)?;
        let mut y = g.reshape(&y, &[b, t, heads, hd])?;
        if let Some(norm) = norm {
            y = g.rms_norm(&y, norm, spec.eps)?;
            y = g.rope_partial(&y, d.cos, d.sin)?;
        }
        g.permute(&y, &SWAP_TIME_HEADS)
    };
    let q = head(&m.wq, spec.q_heads, Some(q_norm))?;
    let k = head(&m.wk, spec.kv_heads, Some(k_norm))?;
    let v = head(&m.wv, spec.kv_heads, None)?;
    let a = g.sdpa(&q, &k, &v)?;
    let a = g.permute(&a, &SWAP_TIME_HEADS)?;
    let a = g.reshape(&a, &[b, t, spec.q_heads * hd])?;
    let gate = g.linear(h, &m.wgate)?;
    let gate = g.sigmoid(&gate)?;
    let a = g.mul(&a, &gate)?;
    g.linear(&a, &m.wo)
}
