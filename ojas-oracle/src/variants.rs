//! Torch goldens for the attention variants the tiny MHA fixtures do not
//! cover: grouped-query attention and sliding-window attention.
//!
//! `fixtures/variants/{gqa,swa}.safetensors`, written by
//! `python/variant_fixture.py` (nanolab's own model, fp32, CPU, one thread):
//! the parameters, one micro-batch, the forward loss and logits, and the
//! gradients of the loss. [`crate::parity::variant_forward_parity`] and
//! [`crate::parity::variant_grads_parity`] hold a [`crate::parity::ParityModel`]
//! to them. Each fixture carries its own spec (`ojas-spec-v1`, or
//! `ojas-spec-v2` with `window`), so the model under test is told which
//! shape to run, not left to assume the tiny one.

use ojas_core::OjasError;

use crate::bad;
use crate::golden::{
    check_config_matches_spec, check_names_and_shapes, f32_exact, token_ids, HostTensor,
    Provenance, TensorSet, TokenBatch,
};
use crate::safetensors::{SafeTensors, StDtype};
use crate::spec::{parse_spec, ModelSpec};

const GQA: &[u8] = include_bytes!("../fixtures/variants/gqa.safetensors");
const SWA: &[u8] = include_bytes!("../fixtures/variants/swa.safetensors");

/// The sliding window of the [`Variant::Swa`] fixture.
pub const SWA_WINDOW: usize = 8;

/// One attention variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// 4 query heads over 2 KV heads, full causal attention.
    Gqa,
    /// MHA with a sliding window of [`SWA_WINDOW`] keys on every layer.
    Swa,
}

impl Variant {
    pub const ALL: [Variant; 2] = [Variant::Gqa, Variant::Swa];

    /// The `variant` tag the generator writes into the provenance.
    pub fn name(self) -> &'static str {
        match self {
            Variant::Gqa => "gqa",
            Variant::Swa => "swa",
        }
    }
}

/// A variant's forward and backward from torch.
#[derive(Clone, Debug)]
pub struct VariantFixture {
    pub variant: Variant,
    pub spec: ModelSpec,
    /// The parameters, nanolab names and §2 shapes for `spec`.
    pub params: TensorSet,
    pub batch: TokenBatch,
    /// nanolab's mean cross-entropy and the logits `[batch, seq_len, vocab]`.
    pub loss: f32,
    pub logits: Vec<f32>,
    /// d(loss · `loss_seed`)/dθ; a parameter torch left without a gradient
    /// (layer 0's `vr_lambda`) is absent and listed in `no_grad`.
    pub grads: TensorSet,
    pub loss_seed: f32,
    pub no_grad: Vec<String>,
    pub provenance: Provenance,
}

/// The checked-in fixture of `variant`.
pub fn variant(variant: Variant) -> Result<VariantFixture, OjasError> {
    let bytes = match variant {
        Variant::Gqa => GQA,
        Variant::Swa => SWA,
    };
    load_variant(bytes, variant)
}

/// Parse a variant file. Refuses a file whose tag, spec or tensors disagree
/// with `want`: the spec must be the tiny model with that variant's one
/// change, every tensor must be one of `param/`, `grad/`, `x`, `y`,
/// `logits`, names and shapes must be §2's for the spec, and a parameter has
/// a gradient unless the file lists it in `no_grad`.
pub fn load_variant(bytes: &[u8], want: Variant) -> Result<VariantFixture, OjasError> {
    let st = SafeTensors::parse(bytes)?;
    let spec = parse_spec(
        st.metadata("ojas.spec")
            .ok_or_else(|| bad("missing __metadata__[\"ojas.spec\"]"))?,
    )?;
    let provenance = Provenance::of(&st)?;
    let meta = &provenance.meta;
    if provenance.kind()? != "variant" || meta.str("variant")? != want.name() {
        return Err(bad(&format!("not the {} variant fixture", want.name())));
    }
    check_config_matches_spec(&provenance.config()?, &spec)?;
    let tiny = crate::spec::tiny_spec();
    let expected = match want {
        Variant::Gqa => ModelSpec {
            n_kv_head: 2,
            ..tiny
        },
        Variant::Swa => ModelSpec {
            window: Some(SWA_WINDOW),
            ..tiny
        },
    };
    if spec != expected {
        return Err(bad(&format!(
            "{} spec is {spec:?}, not the tiny model with its one change",
            want.name()
        )));
    }

    let mut params = Vec::new();
    let mut grads = Vec::new();
    for e in st.entries() {
        let tensor = |name: &str| -> Result<HostTensor, OjasError> {
            if e.dtype != StDtype::F32 {
                return Err(bad(&format!("{}: expected F32", e.name)));
            }
            Ok(HostTensor {
                name: name.to_string(),
                shape: e.shape.clone(),
                data: st.f32s(&e.name)?,
            })
        };
        if let Some(name) = e.name.strip_prefix("param/") {
            params.push(tensor(name)?);
        } else if let Some(name) = e.name.strip_prefix("grad/") {
            grads.push(tensor(name)?);
        } else if !matches!(e.name.as_str(), "x" | "y" | "logits") {
            return Err(bad(&format!("unexpected tensor {}", e.name)));
        }
    }
    let params = TensorSet::new(params)?;
    let grads = TensorSet::new(grads)?;
    check_names_and_shapes(&params, &spec)?;
    let no_grad = meta.strings("no_grad")?;
    for name in params.names() {
        match (grads.get(name), no_grad.iter().any(|n| n == name)) {
            (Some(g), false) if Some(&g.shape) == params.get(name).map(|p| &p.shape) => {}
            (None, true) => {}
            _ => {
                return Err(bad(&format!(
                    "{name}: gradient presence or shape disagrees with no_grad"
                )))
            }
        }
    }
    if grads.len() + no_grad.len() != params.len() {
        return Err(bad(
            "variant fixture holds gradients outside the parameters",
        ));
    }
    if let Some(g) = grads.iter().find(|g| g.data.iter().all(|&v| v == 0.0)) {
        return Err(bad(&format!(
            "{}: an all-zero gradient checks nothing",
            g.name
        )));
    }

    let (b, t) = match st.entry("x").map(|e| e.shape.as_slice()) {
        Some(&[b, t]) => (b, t),
        _ => return Err(bad("x is not [batch, seq_len]")),
    };
    if st.entry("y").map(|e| e.shape.clone()) != Some(vec![b, t])
        || st.entry("logits").map(|e| e.shape.clone()) != Some(vec![b, t, spec.vocab])
    {
        return Err(bad("variant tensors disagree on [batch, seq_len, vocab]"));
    }
    if t != spec.max_seq {
        return Err(bad("seq_len is not the spec's max_seq"));
    }
    let batch = TokenBatch {
        x: token_ids(st.i64s("x")?, spec.vocab)?,
        y: token_ids(st.i64s("y")?, spec.vocab)?,
        batch: b,
        seq_len: t,
    };
    let loss = f32_exact(meta, "loss")?;
    if u64::from(loss.to_bits()) as f64 != meta.num("loss_f32_bits")? {
        return Err(bad("loss and loss_f32_bits disagree"));
    }
    Ok(VariantFixture {
        variant: want,
        spec,
        params,
        batch,
        loss,
        logits: st.f32s("logits")?,
        grads,
        loss_seed: f32_exact(meta, "loss_seed")?,
        no_grad,
        provenance,
    })
}
