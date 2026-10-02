//! The `ojas.spec` JSON (`ojas-spec-v1`) and the framework-design.md §2
//! parameter table it implies.
//!
//! `python/export_init.py` writes the spec into `__metadata__["ojas.spec"]`
//! of every exported `model.safetensors`; the schema is documented in
//! `ojas-oracle/README.md`. [`expected_params`] restates §2's table
//! (names, shapes, init, optimizer group) independently of torch, so the
//! fixtures are checked against the design and not only against themselves.

use ojas_core::OjasError;

use crate::{bad, field, parse_json_with, Json, JsonExt, GOLDEN_JSON};

pub const SPEC_FORMAT: &str = "ojas-spec-v1";
pub const SPEC_ARCH: &str = "nanolab-gpt";

/// One parsed spec. Field names follow ojas-infer's `GptConfig`.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSpec {
    pub vocab: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub hidden: usize,
    pub max_seq: usize,
    pub rope_base: f64,
    pub rms_eps: f64,
    pub tie_embeddings: bool,
    pub qk_norm: bool,
    pub gated_attention: bool,
    pub value_residual: bool,
}

/// The fixture model: 2 layers, d=64, 4 heads of 16, SwiGLU 192, V=256, T=32.
pub fn tiny_spec() -> ModelSpec {
    ModelSpec {
        vocab: 256,
        n_embd: 64,
        n_layer: 2,
        n_head: 4,
        n_kv_head: 4,
        head_dim: 16,
        hidden: 192,
        max_seq: 32,
        rope_base: 10000.0,
        rms_eps: 1e-6,
        tie_embeddings: true,
        qk_norm: true,
        gated_attention: true,
        value_residual: true,
    }
}

/// Parse and validate a spec. Unknown keys are refused, so a newer writer
/// cannot add a field an older reader would silently ignore.
pub fn parse_spec(text: &str) -> Result<ModelSpec, OjasError> {
    const KEYS: [&str; 16] = [
        "format",
        "arch",
        "vocab",
        "n_embd",
        "n_layer",
        "n_head",
        "n_kv_head",
        "head_dim",
        "hidden",
        "max_seq",
        "rope_base",
        "rms_eps",
        "tie_embeddings",
        "qk_norm",
        "gated_attention",
        "value_residual",
    ];
    let root = parse_json_with(text, GOLDEN_JSON)?;
    let obj = root.object()?;
    if let Some((key, _)) = obj.iter().find(|(k, _)| !KEYS.contains(&k.as_str())) {
        return Err(bad(&format!("spec has unknown key {key:?}")));
    }
    let text_field = |key: &str| field(obj, key).and_then(Json::text);
    if text_field("format") != Some(SPEC_FORMAT) {
        return Err(bad("spec format is not ojas-spec-v1"));
    }
    if text_field("arch") != Some(SPEC_ARCH) {
        return Err(bad("spec arch is not nanolab-gpt"));
    }
    let count = |key: &str| -> Result<usize, OjasError> {
        let v = field(obj, key)
            .ok_or_else(|| bad(&format!("spec missing {key}")))?
            .as_u64()
            .map_err(|e| bad(&format!("spec {key}: {}", e.detail())))?;
        if !(1..=1u64 << 32).contains(&v) {
            return Err(bad(&format!("spec {key} = {v} is not a positive integer")));
        }
        usize::try_from(v).map_err(|_| bad(&format!("spec {key} = {v} exceeds usize")))
    };
    let real = |key: &str| -> Result<f64, OjasError> {
        let v = field(obj, key)
            .ok_or_else(|| bad(&format!("spec missing {key}")))?
            .number()?;
        if !(v.is_finite() && v > 0.0) {
            return Err(bad(&format!("spec {key} = {v} is not positive and finite")));
        }
        Ok(v)
    };
    let flag = |key: &str| -> Result<bool, OjasError> {
        field(obj, key)
            .and_then(Json::flag)
            .ok_or_else(|| bad(&format!("spec {key} is not a boolean")))
    };
    let spec = ModelSpec {
        vocab: count("vocab")?,
        n_embd: count("n_embd")?,
        n_layer: count("n_layer")?,
        n_head: count("n_head")?,
        n_kv_head: count("n_kv_head")?,
        head_dim: count("head_dim")?,
        hidden: count("hidden")?,
        max_seq: count("max_seq")?,
        rope_base: real("rope_base")?,
        rms_eps: real("rms_eps")?,
        tie_embeddings: flag("tie_embeddings")?,
        qk_norm: flag("qk_norm")?,
        gated_attention: flag("gated_attention")?,
        value_residual: flag("value_residual")?,
    };
    if !spec.n_head.is_multiple_of(spec.n_kv_head) {
        return Err(bad("spec n_head is not a multiple of n_kv_head"));
    }
    if !spec.head_dim.is_multiple_of(2) {
        return Err(bad("spec head_dim is odd; RoPE rotates pairs"));
    }
    Ok(spec)
}

/// Initialisation of one parameter (nanolab, framework-design.md §2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Init {
    /// N(0, std)
    Normal(f64),
    Zeros,
    Ones,
}

/// Optimizer group (nanolab `_split_params`: 2-D non-embedding weights go to
/// Muon, everything else to AdamW).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    Muon,
    AdamW,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParamRow {
    pub name: String,
    pub shape: Vec<usize>,
    pub init: Init,
    pub group: Group,
}

/// §2's table for `spec`, in nanolab `state_dict` order with `lm_head`
/// omitted (it is tied to `tok_emb.weight` and not stored).
pub fn expected_params(spec: &ModelSpec) -> Vec<ParamRow> {
    let (d, h, hd, f, v) = (
        spec.n_embd,
        spec.n_head,
        spec.head_dim,
        spec.hidden,
        spec.vocab,
    );
    let kv = spec.n_kv_head * hd;
    let row = |name: String, shape: Vec<usize>, init, group| ParamRow {
        name,
        shape,
        init,
        group,
    };
    let normal = Init::Normal(0.02);
    let mut rows = vec![row(
        "tok_emb.weight".into(),
        vec![v, d],
        normal,
        Group::AdamW,
    )];
    for i in 0..spec.n_layer {
        let p = |s: &str| format!("blocks.{i}.{s}");
        rows.extend([
            row(p("norm1.weight"), vec![d], Init::Ones, Group::AdamW),
            row(
                p("mixer.q_proj.weight"),
                vec![h * hd, d],
                normal,
                Group::Muon,
            ),
            row(p("mixer.k_proj.weight"), vec![kv, d], normal, Group::Muon),
            row(p("mixer.v_proj.weight"), vec![kv, d], normal, Group::Muon),
            row(
                p("mixer.o_proj.weight"),
                vec![d, h * hd],
                Init::Zeros,
                Group::Muon,
            ),
            row(p("mixer.q_norm.weight"), vec![hd], Init::Ones, Group::AdamW),
            row(p("mixer.k_norm.weight"), vec![hd], Init::Ones, Group::AdamW),
            row(p("mixer.gate.weight"), vec![h, d], normal, Group::Muon),
            row(p("mixer.gate.bias"), vec![h], Init::Zeros, Group::AdamW),
            row(p("mixer.vr_lambda"), vec![1], Init::Zeros, Group::AdamW),
            row(p("norm2.weight"), vec![d], Init::Ones, Group::AdamW),
            row(p("ffn.gate.weight"), vec![f, d], normal, Group::Muon),
            row(p("ffn.up.weight"), vec![f, d], normal, Group::Muon),
            row(p("ffn.down.weight"), vec![d, f], Init::Zeros, Group::Muon),
        ]);
    }
    rows.push(row(
        "norm_f.weight".into(),
        vec![d],
        Init::Ones,
        Group::AdamW,
    ));
    rows
}
