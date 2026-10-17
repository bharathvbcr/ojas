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
use crate::lora::{
    dropout_mask, lora_down, lora_up, scale_constant, scaled_b, Entropy, LoraConfig, LoraSite,
};
use crate::qwen35::params::{
    AttnParams, GdnParams, LayerLora, LayerParams, MixerLora, MixerParams, Qwen35Lora, Qwen35Params,
};
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

/// [`bind`] with every parameter frozen ([`Graph::frozen`]): no gradient
/// is kept for it. This is the base model of a LoRA run.
pub fn bind_frozen<G: Graph>(
    g: &mut G,
    params: &Qwen35Params<Tensor>,
) -> Result<Qwen35Params<G::V>, OjasError> {
    params.try_map(&mut |t| g.frozen(t))
}

/// What the forward needs to add the LoRA branches: the bound sites, the
/// scaling, the frozen constants that apply it, and the dropout masks.
///
/// Everything here is recorded before any layer runs, because a
/// checkpointed segment refuses a leaf recorded inside it; a replayed layer
/// then reads the same masks the first forward did.
pub struct LoraForward<V> {
    sites: Qwen35Lora<LoraSite<V>>,
    scale: f32,
    /// One `scale` constant per distinct `B` shape.
    consts: Vec<(Vec<usize>, V)>,
    /// One mask per site when dropout is on.
    masks: Option<Qwen35Lora<V>>,
}

impl<V: Clone> LoraForward<V> {
    /// `sites` are the adapters' `A` and `B` as graph values (leaves for a
    /// training tape). With `config.dropout() > 0` a mask `[batch, seq, in]`
    /// is drawn per site from `rng`, in [`Qwen35Lora::sites`] order, so a
    /// fixed seed fixes every step's masks.
    #[allow(clippy::too_many_arguments)]
    pub fn new<G: Graph<V = V>>(
        g: &mut G,
        sites: Qwen35Lora<LoraSite<V>>,
        config: &LoraConfig,
        batch: usize,
        seq: usize,
        rng: &mut impl Entropy,
        budget: &Budget,
    ) -> Result<Self, OjasError> {
        let scale = config.scaling();
        let mut consts: Vec<(Vec<usize>, V)> = Vec::new();
        if scale != 1.0 {
            for site in sites.sites() {
                for b in &site.b {
                    let shape = g.tensor(b)?.shape().to_vec();
                    if !consts.iter().any(|(s, _)| *s == shape) {
                        let c = scale_constant(g, &shape, scale, budget)?;
                        consts.push((shape, c));
                    }
                }
            }
        }
        let masks = if config.dropout() > 0.0 {
            let mut one = |site: &LoraSite<V>| -> Result<V, OjasError> {
                let inn = g.tensor(&site.a)?.shape()[1];
                let m = dropout_mask(config.dropout(), &[batch, seq, inn], rng, budget)?;
                g.frozen(&m)
            };
            Some(sites.try_map(&mut one)?)
        } else {
            None
        };
        Ok(Self {
            sites,
            scale,
            consts,
            masks,
        })
    }

    /// The bound sites, e.g. to read each leaf's gradient.
    pub fn sites(&self) -> &Qwen35Lora<LoraSite<V>> {
        &self.sites
    }
}

/// One layer's view of a [`LoraForward`].
struct LayerLoraCtx<'a, V> {
    scale: f32,
    consts: &'a [(Vec<usize>, V)],
    sites: &'a LayerLora<LoraSite<V>>,
    masks: Option<&'a LayerLora<V>>,
}

/// A site's `x A^T`, once, for every part that shares it.
fn site_down<G: Graph>(
    g: &mut G,
    site: Option<&LoraSite<G::V>>,
    mask: Option<&G::V>,
    x: &G::V,
) -> Result<Option<G::V>, OjasError> {
    site.map(|s| lora_down(g, x, &s.a, mask)).transpose()
}

/// `base + (x A^T) B_part_eff^T`, or `base` where there is no adapter.
fn add_lora<G: Graph>(
    g: &mut G,
    ctx: Option<&LayerLoraCtx<'_, G::V>>,
    site: Option<&LoraSite<G::V>>,
    down: Option<&G::V>,
    part: usize,
    base: G::V,
) -> Result<G::V, OjasError> {
    let (Some(ctx), Some(site), Some(down)) = (ctx, site, down) else {
        return Ok(base);
    };
    let b = site
        .b
        .get(part)
        .ok_or_else(|| shape("qwen35::lora", format!("a site has no B part {part}")))?;
    let like = if ctx.scale == 1.0 {
        None
    } else {
        let want = g.tensor(b)?.shape().to_vec();
        ctx.consts.iter().find(|(s, _)| *s == want).map(|(_, c)| c)
    };
    let b_eff = scaled_b(g, b, ctx.scale, like)?;
    let delta = lora_up(g, down, &b_eff)?;
    g.add(&base, &delta)
}

fn shape(op: &'static str, detail: String) -> OjasError {
    OjasError::Shape { op, detail }
}

/// Mean next-token cross-entropy through the fused tied head: a rank-0 loss.
/// `ids` and `targets` are `U32` `[B, T]` with `T` at most the tables'
/// length; transformers' `labels=ids` is `ids[.., ..T - 1]` against
/// `ids[.., 1..]`. `activations` makes each layer a checkpointed segment
/// ([`ActivationCheckpoint::Blocks`]); it changes memory, not values.
///
/// A target equal to `ignore` (transformers' `ignore_index`, `-100` as a
/// `u32` is `4294967196`) contributes no loss and no gradient and is left out
/// of the mean's divisor, as in the GPT [`crate::forward_loss`]; `None`
/// scores every position. A batch with every target ignored is refused by the
/// fused CE, not scored as zero.
#[allow(clippy::too_many_arguments)]
pub fn forward_loss<G: Graph>(
    g: &mut G,
    spec: &Qwen35Spec,
    params: &Qwen35Params<G::V>,
    ids: &Tensor,
    targets: &Tensor,
    tables: &Qwen35Tables,
    ignore: Option<u32>,
    chunk: CeChunk,
    activations: ActivationCheckpoint,
) -> Result<G::V, OjasError> {
    forward_loss_lora(
        g,
        spec,
        params,
        None,
        ids,
        targets,
        tables,
        ignore,
        chunk,
        activations,
    )
}

/// [`forward_loss`] with LoRA branches on the linears `lora` has sites for
/// (`None` is [`forward_loss`] exactly). The base `params` are used as
/// bound: bind them with [`bind_frozen`] to train the adapters alone.
///
/// The tower to the final norm is [`forward_hidden_lora`]; this function
/// adds only the `targets` check and the fused tied-head cross-entropy over
/// every position, so its values and gradients are those of the single
/// function it was split from.
#[allow(clippy::too_many_arguments)]
pub fn forward_loss_lora<G: Graph>(
    g: &mut G,
    spec: &Qwen35Spec,
    params: &Qwen35Params<G::V>,
    lora: Option<&LoraForward<G::V>>,
    ids: &Tensor,
    targets: &Tensor,
    tables: &Qwen35Tables,
    ignore: Option<u32>,
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
    let h = tower(g, OP, spec, params, lora, ids, tables, activations)?;
    let rows = batch * seq;
    let h = g.reshape(&h, &[rows, spec.hidden])?;
    let targets = targets.reshape(&[rows])?;
    g.lin_ce(&h, &params.embed, &targets, ignore, chunk)
}

/// The tower to the final norm: transformers' `Qwen3_5TextModel` output,
/// `last_hidden_state`, `[B, T, hidden]`, before the tied head. `ids` is
/// `U32` `[B, T]` with `T` at most the tables' length. A host-side head
/// (an answer head, a span pointer) reads rows of it with [`gather_rows`]
/// and trains through them on a tape; [`forward_loss`] is this plus the
/// fused tied-head cross-entropy.
///
/// Every layer is causal, so row `b`'s positions `0..n` depend on `ids[b,
/// ..n]` alone: finite tokens after them, such as right padding, change
/// nothing the row reads. That holds for finite padding only: a non-finite
/// value in a padding token's embedding is refused as `NonFinite`, naming
/// the op. The CPU backend refuses it at the op; a GPU backend records the
/// fault and reports it at `Backend::sync` or `Backend::download`, so read
/// results of a GPU tape through those (raw `Tensor::to_host` is not a fault
/// point on Metal and would hand the caller NaN with no error).
/// [`RightPadded`] is the input form that makes only right padding
/// representable.
pub fn forward_hidden<G: Graph>(
    g: &mut G,
    spec: &Qwen35Spec,
    params: &Qwen35Params<G::V>,
    ids: &Tensor,
    tables: &Qwen35Tables,
    activations: ActivationCheckpoint,
) -> Result<G::V, OjasError> {
    forward_hidden_lora(g, spec, params, None, ids, tables, activations)
}

/// [`forward_hidden`] with LoRA branches (`None` is [`forward_hidden`]
/// exactly), under the same contract as [`forward_loss_lora`].
pub fn forward_hidden_lora<G: Graph>(
    g: &mut G,
    spec: &Qwen35Spec,
    params: &Qwen35Params<G::V>,
    lora: Option<&LoraForward<G::V>>,
    ids: &Tensor,
    tables: &Qwen35Tables,
    activations: ActivationCheckpoint,
) -> Result<G::V, OjasError> {
    tower(
        g,
        "qwen35::forward_hidden",
        spec,
        params,
        lora,
        ids,
        tables,
        activations,
    )
}

/// The one implementation behind [`forward_hidden_lora`] and
/// [`forward_loss_lora`]; `op` names the public entry in its errors.
#[allow(clippy::too_many_arguments)]
fn tower<G: Graph>(
    g: &mut G,
    op: &'static str,
    spec: &Qwen35Spec,
    params: &Qwen35Params<G::V>,
    lora: Option<&LoraForward<G::V>>,
    ids: &Tensor,
    tables: &Qwen35Tables,
    activations: ActivationCheckpoint,
) -> Result<G::V, OjasError> {
    spec.validate()?;
    let &[batch, seq] = ids.shape() else {
        return Err(shape(
            op,
            format!("ids {:?} are not [batch, time]", ids.shape()),
        ));
    };
    if seq == 0 || seq > tables.len() {
        return Err(shape(
            op,
            format!("{seq} positions for tables of {}", tables.len()),
        ));
    }
    if params.layers.len() != spec.layers.len() {
        return Err(shape(
            op,
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
    let one_d = g.frozen(&tables.ones_hidden)?;
    let one_h = g.frozen(&tables.ones_head)?;
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
    if let Some(lora) = lora {
        if lora.sites.layers.len() != params.layers.len() {
            return Err(shape(
                op,
                format!(
                    "{} LoRA layers for a spec of {}",
                    lora.sites.layers.len(),
                    params.layers.len()
                ),
            ));
        }
    }
    for (i, (l, n)) in params.layers.iter().zip(&norms).enumerate() {
        let ctx = lora.map(|f| LayerLoraCtx {
            scale: f.scale,
            consts: &f.consts,
            sites: &f.sites.layers[i],
            masks: f.masks.as_ref().map(|m| &m.layers[i]),
        });
        x = match activations {
            ActivationCheckpoint::Off => layer(g, spec, l, n, ctx.as_ref(), &x, &dims)?,
            ActivationCheckpoint::Blocks => {
                let mut outs = g.checkpoint(|g| {
                    layer(g, spec, l, n, ctx.as_ref(), &x, &dims).map(|y| vec![y])
                })?;
                outs.pop()
                    .ok_or_else(|| shape(op, "a checkpointed layer returned no output".into()))?
            }
        };
    }
    g.rms_norm(&x, &final_norm, spec.eps)
}

/// The most rows a flat `[B * T]` index can address: `embedding` takes `u32`
/// ids, and a flat index is `b * T + position`.
const MAX_FLAT_ROWS: u64 = u32::MAX as u64;

/// How long each row of a right-padded batch really is, out of `seq`
/// columns. Only right padding is representable: row `b` holds its `lengths[b]
/// >= 1` tokens in columns `0..lengths[b]` and padding after them, which the
/// causal tower cannot carry into the row's own positions as long as the
/// padding is finite ([`forward_hidden`] says what non-finite padding does). Left padding shifts a row's tokens to other
/// positions, and a hole or a packed row lets one segment read another, so
/// [`RightPadded::from_attention_mask`] refuses both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RightPadded {
    lengths: Vec<usize>,
    seq: usize,
}

impl RightPadded {
    const OP: &'static str = "qwen35::RightPadded";

    /// A batch of `lengths.len()` rows of `seq` columns. Every length is in
    /// `1..=seq`, the batch is not empty, and `B * T` fits the `u32` flat row
    /// index [`gather_rows`] uses; that is checked before anything is
    /// allocated for it.
    pub fn new(lengths: Vec<usize>, seq: usize) -> Result<Self, OjasError> {
        Self::check_extent(lengths.len(), seq)?;
        if let Some((b, &n)) = lengths.iter().enumerate().find(|(_, &n)| n == 0 || n > seq) {
            return Err(shape(
                Self::OP,
                format!("row {b} has length {n}, not in 1..={seq}"),
            ));
        }
        Ok(Self { lengths, seq })
    }

    /// `batch` rows with no padding. The extent is checked before the
    /// lengths are allocated.
    pub fn full(batch: usize, seq: usize) -> Result<Self, OjasError> {
        Self::check_extent(batch, seq)?;
        Self::new(vec![seq; batch], seq)
    }

    /// `batch` and `seq` are both positive and `batch * seq` fits the `u32`
    /// flat row index [`gather_rows`] uses. Every constructor calls this
    /// before it allocates anything sized by `batch`.
    fn check_extent(batch: usize, seq: usize) -> Result<(), OjasError> {
        if batch == 0 || seq == 0 {
            return Err(shape(Self::OP, format!("{batch} rows of {seq} columns")));
        }
        if (batch as u64)
            .checked_mul(seq as u64)
            .is_none_or(|n| n > MAX_FLAT_ROWS)
        {
            return Err(shape(
                Self::OP,
                format!(
                    "{batch} rows of {seq} columns exceed the {MAX_FLAT_ROWS} rows a u32 index addresses"
                ),
            ));
        }
        Ok(())
    }

    /// The lengths of Hugging Face's `attention_mask`, `[batch, seq]` row
    /// major, `1` for a token and `0` for padding. Refused: a value other
    /// than 0 or 1; a row whose first entry is 0 and that has a 1 later (left
    /// padding); a 0 followed by a 1 after the row's first tokens (a hole, or
    /// packed rows); a row of all zeros.
    pub fn from_attention_mask(mask: &[i64], batch: usize, seq: usize) -> Result<Self, OjasError> {
        if batch.checked_mul(seq) != Some(mask.len()) {
            return Err(shape(
                Self::OP,
                format!("{} mask values for {batch} rows of {seq}", mask.len()),
            ));
        }
        Self::check_extent(batch, seq)?;
        if let Some(v) = mask.iter().find(|&&v| v != 0 && v != 1) {
            return Err(shape(
                Self::OP,
                format!("attention_mask holds {v}, not 0 or 1"),
            ));
        }
        let mut lengths = Vec::with_capacity(batch);
        for (b, row) in mask.chunks_exact(seq.max(1)).enumerate() {
            let n = row.iter().take_while(|&&v| v == 1).count();
            let rest_has_token = row[n..].contains(&1);
            if n == 0 && !rest_has_token {
                return Err(shape(Self::OP, format!("row {b} is all padding")));
            }
            if n == 0 {
                return Err(shape(
                    Self::OP,
                    format!("row {b} is left padded: only right padding is supported"),
                ));
            }
            if rest_has_token {
                return Err(shape(
                    Self::OP,
                    format!("row {b} has a token after padding (a hole or a packed row)"),
                ));
            }
            lengths.push(n);
        }
        Self::new(lengths, seq)
    }

    pub fn batch(&self) -> usize {
        self.lengths.len()
    }

    pub fn seq(&self) -> usize {
        self.seq
    }

    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    /// The flat index `b * seq + positions[b]` into `[B * T]` rows, one per
    /// example. Refused: not exactly one position per row, and a position at
    /// or beyond its row's length, which would read padding.
    pub fn flat_ids(&self, positions: &[usize]) -> Result<Vec<u32>, OjasError> {
        if positions.len() != self.lengths.len() {
            return Err(shape(
                "qwen35::gather_rows",
                format!(
                    "{} positions for {} rows",
                    positions.len(),
                    self.lengths.len()
                ),
            ));
        }
        positions
            .iter()
            .zip(&self.lengths)
            .enumerate()
            .map(|(b, (&p, &n))| {
                if p >= n {
                    return Err(shape(
                        "qwen35::gather_rows",
                        format!("row {b}: position {p} is not inside its {n} tokens"),
                    ));
                }
                u32::try_from(b * self.seq + p).map_err(|_| {
                    shape(
                        "qwen35::gather_rows",
                        format!("row {b}: flat index overflows u32"),
                    )
                })
            })
            .collect()
    }
}

/// Row `positions[b]` of example `b` of `h`, `[B, T, D]` to `[B, D]`: how a
/// host-side head reads the hidden state ([`forward_hidden`]) at an answer
/// position per example, with a gradient. The rows are gathered with
/// [`Graph::embedding`] over `h` reshaped to `[B * T, D]` as a computed
/// table: the backward is its scatter-add, so the gradient lands on the
/// gathered rows of `h` and is exactly zero everywhere else. `rows` fixes `B`
/// and `T` (they must match `h`) and bounds every position by its row's true
/// length.
pub fn gather_rows<G: Graph>(
    g: &mut G,
    h: &G::V,
    rows: &RightPadded,
    positions: &[usize],
    budget: &Budget,
) -> Result<G::V, OjasError> {
    const OP: &str = "qwen35::gather_rows";
    let want = [rows.batch(), rows.seq()];
    let hidden = match g.tensor(h)?.shape() {
        &[b, t, d] if [b, t] == want => d,
        other => {
            return Err(shape(
                OP,
                format!("h {other:?} is not [{}, {}, hidden]", want[0], want[1]),
            ))
        }
    };
    let flat = rows.flat_ids(positions)?;
    let ids = Tensor::from_u32(&flat, &[flat.len()], budget)?;
    let table = g.reshape(h, &[want[0] * want[1], hidden])?;
    g.embedding(&table, &ids)
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
    lc: Option<&LayerLoraCtx<'_, G::V>>,
    x: &G::V,
    d: &Dims<'_>,
) -> Result<G::V, OjasError> {
    let h = g.rms_norm(x, &n.input, spec.eps)?;
    let mixed = match (&l.mixer, &n.qk) {
        (MixerParams::GatedDeltaNet(m), None) => gated_delta_net(g, spec, m, lc, &h, d)?,
        (MixerParams::Attention(m), Some((qn, kn))) => attention(g, spec, m, lc, qn, kn, &h, d)?,
        _ => {
            return Err(shape(
                "qwen35::layer",
                "a layer's norms do not match its mixer".into(),
            ))
        }
    };
    let x = g.add(x, &mixed)?;
    let h = g.rms_norm(&x, &n.post, spec.eps)?;
    let mlp = |g: &mut G,
               input: &G::V,
               w: &G::V,
               site: &Option<LoraSite<G::V>>,
               mask: Option<&G::V>|
     -> Result<G::V, OjasError> {
        let base = g.linear(input, w)?;
        let down = site_down(g, site.as_ref(), mask, input)?;
        add_lora(g, lc, site.as_ref(), down.as_ref(), 0, base)
    };
    let masks = lc.and_then(|c| c.masks);
    let gate = mlp(
        g,
        &h,
        &l.gate,
        lc.map_or(&None, |c| &c.sites.gate),
        masks.and_then(|m| m.gate.as_ref()),
    )?;
    let gate = g.silu(&gate)?;
    let up = mlp(
        g,
        &h,
        &l.up,
        lc.map_or(&None, |c| &c.sites.up),
        masks.and_then(|m| m.up.as_ref()),
    )?;
    let act = g.mul(&gate, &up)?;
    let down = mlp(
        g,
        &act,
        &l.down,
        lc.map_or(&None, |c| &c.sites.down),
        masks.and_then(|m| m.down.as_ref()),
    )?;
    g.add(&x, &down)
}

fn gated_delta_net<G: Graph>(
    g: &mut G,
    spec: &Qwen35Spec,
    m: &GdnParams<G::V>,
    lc: Option<&LayerLoraCtx<'_, G::V>>,
    h: &G::V,
    d: &Dims<'_>,
) -> Result<G::V, OjasError> {
    let (b, t, heads) = (d.batch, d.seq, spec.gdn_heads);
    let (dk, dv) = (spec.gdn_key_dim, spec.gdn_value_dim);
    let (qkv, z_site, out_site) = match lc.map(|c| (&c.sites.mixer, c.masks.map(|m| &m.mixer))) {
        Some((MixerLora::GatedDeltaNet { qkv, z, out }, _)) => {
            (qkv.as_ref(), z.as_ref(), out.as_ref())
        }
        _ => (None, None, None),
    };
    let (qkv_mask, z_mask, out_mask) = match lc.and_then(|c| c.masks.map(|m| &m.mixer)) {
        Some(MixerLora::GatedDeltaNet { qkv, z, out }) => (qkv.as_ref(), z.as_ref(), out.as_ref()),
        _ => (None, None, None),
    };
    // One `x A^T` for the three parts of `in_proj_qkv`.
    let qkv_down = site_down(g, qkv, qkv_mask, h)?;
    let mut branch = |w: &G::V, part: usize, conv: &G::V, dim: usize| -> Result<G::V, OjasError> {
        let y = g.linear(h, w)?;
        let y = add_lora(g, lc, qkv, qkv_down.as_ref(), part, y)?;
        let y = g.conv1d_silu(&y, conv)?;
        g.reshape(&y, &[b, t, heads, dim])
    };
    let q = branch(&m.wq, 0, &m.conv_q, dk)?;
    let k = branch(&m.wk, 1, &m.conv_k, dk)?;
    let v = branch(&m.wv, 2, &m.conv_v, dv)?;
    let z = g.linear(h, &m.wz)?;
    let z_down = site_down(g, z_site, z_mask, h)?;
    let z = add_lora(g, lc, z_site, z_down.as_ref(), 0, z)?;
    let z = g.reshape(&z, &[b, t, heads, dv])?;
    let beta = g.linear(h, &m.wb)?;
    let beta = g.sigmoid(&beta)?;
    let a = g.linear(h, &m.wa)?;
    let decay = g.gdn_log_decay(&a, &m.a_log, &m.dt_bias)?;
    let o = g.gdn(&q, &k, &v, &decay, &beta)?;
    let o = g.gated_rms_norm(&o, &z, &m.norm, spec.eps)?;
    let o = g.reshape(&o, &[b, t, heads * dv])?;
    let y = g.linear(&o, &m.out)?;
    let out_down = site_down(g, out_site, out_mask, &o)?;
    add_lora(g, lc, out_site, out_down.as_ref(), 0, y)
}

#[allow(clippy::too_many_arguments)]
fn attention<G: Graph>(
    g: &mut G,
    spec: &Qwen35Spec,
    m: &AttnParams<G::V>,
    lc: Option<&LayerLoraCtx<'_, G::V>>,
    q_norm: &G::V,
    k_norm: &G::V,
    h: &G::V,
    d: &Dims<'_>,
) -> Result<G::V, OjasError> {
    let (b, t, hd) = (d.batch, d.seq, spec.head_dim);
    let (qs, ks, vs, os) = match lc.map(|c| &c.sites.mixer) {
        Some(MixerLora::Attention { q, k, v, o }) => {
            (q.as_ref(), k.as_ref(), v.as_ref(), o.as_ref())
        }
        _ => (None, None, None, None),
    };
    let (qm, km, vm, om) = match lc.and_then(|c| c.masks.map(|m| &m.mixer)) {
        Some(MixerLora::Attention { q, k, v, o }) => {
            (q.as_ref(), k.as_ref(), v.as_ref(), o.as_ref())
        }
        _ => (None, None, None, None),
    };
    // `q_proj` is one fused weight: its adapter's `A` is shared by the
    // query rows (part 0) and the output gate rows (part 1).
    let q_down = site_down(g, qs, qm, h)?;
    let k_down = site_down(g, ks, km, h)?;
    let v_down = site_down(g, vs, vm, h)?;
    let mut head = |w: &G::V,
                    site: Option<&LoraSite<G::V>>,
                    down: Option<&G::V>,
                    part: usize,
                    heads: usize,
                    norm: Option<&G::V>|
     -> Result<G::V, OjasError> {
        let y = g.linear(h, w)?;
        let y = add_lora(g, lc, site, down, part, y)?;
        let mut y = g.reshape(&y, &[b, t, heads, hd])?;
        if let Some(norm) = norm {
            y = g.rms_norm(&y, norm, spec.eps)?;
            y = g.rope_partial(&y, d.cos, d.sin)?;
        }
        g.permute(&y, &SWAP_TIME_HEADS)
    };
    let q = head(&m.wq, qs, q_down.as_ref(), 0, spec.q_heads, Some(q_norm))?;
    let k = head(&m.wk, ks, k_down.as_ref(), 0, spec.kv_heads, Some(k_norm))?;
    let v = head(&m.wv, vs, v_down.as_ref(), 0, spec.kv_heads, None)?;
    let a = g.sdpa(&q, &k, &v, None)?;
    let a = g.permute(&a, &SWAP_TIME_HEADS)?;
    let a = g.reshape(&a, &[b, t, spec.q_heads * hd])?;
    let gate = g.linear(h, &m.wgate)?;
    let gate = add_lora(g, lc, qs, q_down.as_ref(), 1, gate)?;
    let gate = g.sigmoid(&gate)?;
    let a = g.mul(&a, &gate)?;
    let y = g.linear(&a, &m.wo)?;
    let o_down = site_down(g, os, om, &a)?;
    add_lora(g, lc, os, o_down.as_ref(), 0, y)
}
