//! The tiny-fixture loader: tessl's `tests/fixtures/qwen35_train/`, a small
//! random `Qwen3_5ForCausalLM` of the 2B's shape family (one GDN layer, one
//! attention layer, tied embeddings), with its token ids, its float64 loss
//! and every parameter's float32 gradient from transformers' `loss.backward()`
//! (`tessl/tests/qwen35_train.rs:1-11`, written by
//! `tessl/tools/qwen35_ref/make_train_fixture.py tiny`).
//!
//! The copy here is `tests/fixtures/qwen35_train_published/`, byte-identical
//! and pinned (`tests/fixture_pins.rs`). Its GDN gradients come from
//! transformers' own Qwen3.5 module, which is the published operator; the
//! name says so (Lappi rule 9). That the generator runs transformers' module
//! is read from tessl's test docs, not re-run here
//! (`GAP-L-CUDA-M1-TINY-FIXTURE-RULE-INFERRED-2026-10-01`).
//!
//! [`TinyFixture::embedded`] parses the copy compiled into the binary, so
//! `runga` carries it to the box; [`TinyFixture::from_dir`] reads a
//! directory, each file opened with `ojas_io::open_nofollow` and bounded in
//! size. Both go through the same parser:
//! - `config.json` through `ojas_io::parse_json_with` under explicit limits,
//!   with every key named (an unknown key is refused: the fixture is pinned,
//!   so a new key means a different fixture);
//! - `model.safetensors` through `ojas_io::SafeTensors`, every tensor BF16;
//! - the `.npy` files through [`crate::npy`]: `ids` `<i8`, `loss` `<f8`,
//!   one `<f4` `grad.<parameter>` per parameter, of the parameter's shape.
//!
//! With `cuda`, [`TinyFixture::upload`] puts all of it in device buffers.

use std::io::Read;
use std::path::Path;

use ojas_io::{open_nofollow, parse_json_with, JsonLimits, JsonValue, SafeTensors, StDtype};

use crate::error::CudaError;
use crate::npy;

/// The files, compiled in.
pub mod files {
    macro_rules! fixture {
        ($f:literal) => {
            include_bytes!(concat!("../tests/fixtures/qwen35_train_published/", $f))
        };
    }
    /// The pins `tests/fixture_pins.rs` checks the copy against.
    pub const SHA256SUMS: &str =
        include_str!("../tests/fixtures/qwen35_train_published_SHA256SUMS");
    pub const CONFIG: &[u8] = fixture!("config.json");
    pub const MODEL: &[u8] = fixture!("model.safetensors");
    pub const IDS: &[u8] = fixture!("ids.npy");
    pub const LOSS: &[u8] = fixture!("loss.npy");
    /// `(parameter name, grad.<name>.npy bytes)`.
    pub const GRADS: [(&str, &[u8]); 27] = [
        (
            "model.embed_tokens.weight",
            fixture!("grad.model.embed_tokens.weight.npy"),
        ),
        (
            "model.layers.0.input_layernorm.weight",
            fixture!("grad.model.layers.0.input_layernorm.weight.npy"),
        ),
        (
            "model.layers.0.linear_attn.A_log",
            fixture!("grad.model.layers.0.linear_attn.A_log.npy"),
        ),
        (
            "model.layers.0.linear_attn.conv1d.weight",
            fixture!("grad.model.layers.0.linear_attn.conv1d.weight.npy"),
        ),
        (
            "model.layers.0.linear_attn.dt_bias",
            fixture!("grad.model.layers.0.linear_attn.dt_bias.npy"),
        ),
        (
            "model.layers.0.linear_attn.in_proj_a.weight",
            fixture!("grad.model.layers.0.linear_attn.in_proj_a.weight.npy"),
        ),
        (
            "model.layers.0.linear_attn.in_proj_b.weight",
            fixture!("grad.model.layers.0.linear_attn.in_proj_b.weight.npy"),
        ),
        (
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            fixture!("grad.model.layers.0.linear_attn.in_proj_qkv.weight.npy"),
        ),
        (
            "model.layers.0.linear_attn.in_proj_z.weight",
            fixture!("grad.model.layers.0.linear_attn.in_proj_z.weight.npy"),
        ),
        (
            "model.layers.0.linear_attn.norm.weight",
            fixture!("grad.model.layers.0.linear_attn.norm.weight.npy"),
        ),
        (
            "model.layers.0.linear_attn.out_proj.weight",
            fixture!("grad.model.layers.0.linear_attn.out_proj.weight.npy"),
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            fixture!("grad.model.layers.0.mlp.down_proj.weight.npy"),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            fixture!("grad.model.layers.0.mlp.gate_proj.weight.npy"),
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            fixture!("grad.model.layers.0.mlp.up_proj.weight.npy"),
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            fixture!("grad.model.layers.0.post_attention_layernorm.weight.npy"),
        ),
        (
            "model.layers.1.input_layernorm.weight",
            fixture!("grad.model.layers.1.input_layernorm.weight.npy"),
        ),
        (
            "model.layers.1.mlp.down_proj.weight",
            fixture!("grad.model.layers.1.mlp.down_proj.weight.npy"),
        ),
        (
            "model.layers.1.mlp.gate_proj.weight",
            fixture!("grad.model.layers.1.mlp.gate_proj.weight.npy"),
        ),
        (
            "model.layers.1.mlp.up_proj.weight",
            fixture!("grad.model.layers.1.mlp.up_proj.weight.npy"),
        ),
        (
            "model.layers.1.post_attention_layernorm.weight",
            fixture!("grad.model.layers.1.post_attention_layernorm.weight.npy"),
        ),
        (
            "model.layers.1.self_attn.k_norm.weight",
            fixture!("grad.model.layers.1.self_attn.k_norm.weight.npy"),
        ),
        (
            "model.layers.1.self_attn.k_proj.weight",
            fixture!("grad.model.layers.1.self_attn.k_proj.weight.npy"),
        ),
        (
            "model.layers.1.self_attn.o_proj.weight",
            fixture!("grad.model.layers.1.self_attn.o_proj.weight.npy"),
        ),
        (
            "model.layers.1.self_attn.q_norm.weight",
            fixture!("grad.model.layers.1.self_attn.q_norm.weight.npy"),
        ),
        (
            "model.layers.1.self_attn.q_proj.weight",
            fixture!("grad.model.layers.1.self_attn.q_proj.weight.npy"),
        ),
        (
            "model.layers.1.self_attn.v_proj.weight",
            fixture!("grad.model.layers.1.self_attn.v_proj.weight.npy"),
        ),
        ("model.norm.weight", fixture!("grad.model.norm.weight.npy")),
    ];
}

/// Largest file [`TinyFixture::from_dir`] reads; the fixture's largest is
/// `model.safetensors` at 458,588 bytes.
pub const MAX_FILE_BYTES: u64 = 16 << 20;
/// Most tensors the model file may hold.
pub const MAX_TENSORS: usize = 4096;

const CONFIG_LIMITS: JsonLimits = JsonLimits {
    max_input_bytes: 16 << 10,
    max_depth: 4,
    max_nodes: 256,
    ..JsonLimits::DEFAULT
};

/// Every key the fixture's `config.json` holds.
const CONFIG_KEYS: [&str; 30] = [
    "attention_bias",
    "attention_dropout",
    "bos_token_id",
    "eos_token_id",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "layer_types",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "max_position_embeddings",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "partial_rotary_factor",
    "rms_norm_eps",
    "rope_parameters",
    "tie_word_embeddings",
    "transformers_version",
    "use_cache",
    "vocab_size",
    // Not in this fixture; accepted so a regenerated one that adds them parses.
    "dtype",
    "torch_dtype",
];

/// A decoder layer's mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// GDN (`linear_attention`), at the published rule.
    LinearAttention,
    /// Gated full attention.
    FullAttention,
}

/// What the loader reads from `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct TinyConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub layers: Vec<LayerKind>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub linear_key_heads: usize,
    pub linear_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub conv_kernel: usize,
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
}

/// One parameter as the model file stores it: bf16 bits, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: String,
    pub shape: Vec<usize>,
    pub bf16: Vec<u16>,
    /// transformers' float32 gradient of the loss, same shape.
    pub grad: Vec<f32>,
}

/// The whole fixture.
#[derive(Clone, Debug, PartialEq)]
pub struct TinyFixture {
    pub config: TinyConfig,
    /// In the model file's (sorted) order.
    pub params: Vec<Param>,
    pub ids: Vec<u32>,
    pub loss: f64,
}

fn bad(what: impl Into<String>) -> CudaError {
    CudaError::invalid("tiny_fixture_published", what)
}

fn io(e: ojas_io::IoError) -> CudaError {
    bad(e.detail().to_string())
}

fn usize_field(v: &JsonValue, key: &str) -> Result<usize, CudaError> {
    let n = v.get_u64(key).map_err(io)?;
    usize::try_from(n).map_err(|e| bad(format!("{key} {n}: {e}")))
}

/// Parse `config.json`.
pub fn parse_config(bytes: &[u8]) -> Result<TinyConfig, CudaError> {
    let text = std::str::from_utf8(bytes).map_err(|e| bad(format!("config.json: {e}")))?;
    let v = parse_json_with(text, &CONFIG_LIMITS).map_err(io)?;
    v.deny_unknown_keys(&CONFIG_KEYS).map_err(io)?;
    if v.get_str("model_type").map_err(io)? != "qwen3_5_text" {
        return Err(bad("model_type is not qwen3_5_text"));
    }
    if v.get_str("hidden_act").map_err(io)? != "silu" {
        return Err(bad("hidden_act is not silu"));
    }
    let layers = v
        .get_array("layer_types")
        .map_err(io)?
        .iter()
        .map(|t| match t.as_str().map_err(io)? {
            "linear_attention" => Ok(LayerKind::LinearAttention),
            "full_attention" => Ok(LayerKind::FullAttention),
            other => Err(bad(format!("layer type {other:?}"))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if layers.len() != usize_field(&v, "num_hidden_layers")? {
        return Err(bad("layer_types disagrees with num_hidden_layers"));
    }
    let rope = v.get_object("rope_parameters").map_err(io)?;
    if rope.get_str("rope_type").map_err(io)? != "default" {
        return Err(bad("rope_type is not default"));
    }
    let prf = v.get_f64("partial_rotary_factor").map_err(io)?;
    if rope.get_f64("partial_rotary_factor").map_err(io)?.to_bits() != prf.to_bits() {
        return Err(bad("two partial_rotary_factor values"));
    }
    Ok(TinyConfig {
        hidden: usize_field(&v, "hidden_size")?,
        intermediate: usize_field(&v, "intermediate_size")?,
        vocab: usize_field(&v, "vocab_size")?,
        layers,
        heads: usize_field(&v, "num_attention_heads")?,
        kv_heads: usize_field(&v, "num_key_value_heads")?,
        head_dim: usize_field(&v, "head_dim")?,
        linear_key_heads: usize_field(&v, "linear_num_key_heads")?,
        linear_value_heads: usize_field(&v, "linear_num_value_heads")?,
        linear_key_head_dim: usize_field(&v, "linear_key_head_dim")?,
        linear_value_head_dim: usize_field(&v, "linear_value_head_dim")?,
        conv_kernel: usize_field(&v, "linear_conv_kernel_dim")?,
        partial_rotary_factor: prf,
        rope_theta: rope.get_f64("rope_theta").map_err(io)?,
        rms_norm_eps: v.get_f64("rms_norm_eps").map_err(io)?,
        tie_word_embeddings: v.get_bool("tie_word_embeddings").map_err(io)?,
    })
}

/// The bytes of every file, however they were obtained.
pub struct FixtureBytes<'a> {
    pub config: &'a [u8],
    pub model: &'a [u8],
    pub ids: &'a [u8],
    pub loss: &'a [u8],
    /// `(parameter name, grad npy bytes)`.
    pub grads: Vec<(String, &'a [u8])>,
}

impl TinyFixture {
    /// The copy compiled into this binary.
    pub fn embedded() -> Result<Self, CudaError> {
        Self::parse(&FixtureBytes {
            config: files::CONFIG,
            model: files::MODEL,
            ids: files::IDS,
            loss: files::LOSS,
            grads: files::GRADS
                .iter()
                .map(|(n, b)| (n.to_string(), *b))
                .collect(),
        })
    }

    /// Read `dir`: each file opened without following a final symlink, and
    /// at most [`MAX_FILE_BYTES`]. The gradient files are those the model
    /// file names (`grad.<parameter>.npy`).
    pub fn from_dir(dir: &Path) -> Result<Self, CudaError> {
        let read = |name: &str| -> Result<Vec<u8>, CudaError> {
            let path = dir.join(name);
            let file = open_nofollow(&path).map_err(io)?;
            let len = file
                .metadata()
                .map_err(|e| bad(format!("{}: {e}", path.display())))?
                .len();
            if len > MAX_FILE_BYTES {
                return Err(bad(format!(
                    "{}: {len} bytes, over {MAX_FILE_BYTES}",
                    path.display()
                )));
            }
            let mut out = Vec::new();
            file.take(MAX_FILE_BYTES + 1)
                .read_to_end(&mut out)
                .map_err(|e| bad(format!("{}: {e}", path.display())))?;
            if out.len() as u64 > MAX_FILE_BYTES {
                return Err(bad(format!(
                    "{}: grew past {MAX_FILE_BYTES} bytes",
                    path.display()
                )));
            }
            Ok(out)
        };
        let config = read("config.json")?;
        let model = read("model.safetensors")?;
        let ids = read("ids.npy")?;
        let loss = read("loss.npy")?;
        let names: Vec<String> = SafeTensors::parse(&model)
            .map_err(io)?
            .names()
            .map(str::to_string)
            .collect();
        if names.len() > MAX_TENSORS {
            return Err(bad(format!("{} tensors, over {MAX_TENSORS}", names.len())));
        }
        let grads: Vec<(String, Vec<u8>)> = names
            .iter()
            .map(|n| Ok((n.clone(), read(&format!("grad.{n}.npy"))?)))
            .collect::<Result<_, CudaError>>()?;
        Self::parse(&FixtureBytes {
            config: &config,
            model: &model,
            ids: &ids,
            loss: &loss,
            grads: grads
                .iter()
                .map(|(n, b)| (n.clone(), b.as_slice()))
                .collect(),
        })
    }

    /// Parse and cross-check every file.
    pub fn parse(b: &FixtureBytes<'_>) -> Result<Self, CudaError> {
        let config = parse_config(b.config)?;
        let st = SafeTensors::parse(b.model).map_err(io)?;
        let names: Vec<String> = st.names().map(str::to_string).collect();
        if names.is_empty() || names.len() > MAX_TENSORS {
            return Err(bad(format!("{} tensors in the model file", names.len())));
        }
        if config.tie_word_embeddings && names.iter().any(|n| n == "lm_head.weight") {
            return Err(bad("tied embeddings, yet the file holds lm_head.weight"));
        }
        let mut grads: std::collections::BTreeMap<&str, &[u8]> = std::collections::BTreeMap::new();
        for (n, bytes) in &b.grads {
            if grads.insert(n.as_str(), bytes).is_some() {
                return Err(bad(format!("grad.{n} given twice")));
            }
        }
        if grads.len() != names.len() || names.iter().any(|n| !grads.contains_key(n.as_str())) {
            return Err(bad(format!(
                "{} gradients for {} parameters, or a name that does not match",
                grads.len(),
                names.len()
            )));
        }
        let mut params = Vec::with_capacity(names.len());
        for name in &names {
            let info = st.info(name).map_err(io)?;
            if info.dtype != StDtype::BF16 {
                return Err(bad(format!(
                    "{name}: {:?}, the fixture stores BF16",
                    info.dtype
                )));
            }
            let shape = info.shape.clone();
            // The reader checked the byte range against shape x 2 at parse.
            let bf16: Vec<u16> = st
                .read_bytes(name)
                .map_err(io)?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect();
            let shape: Vec<usize> = shape
                .iter()
                .map(|&d| usize::try_from(d).map_err(|e| bad(format!("{name}: {e}"))))
                .collect::<Result<_, _>>()?;
            let numel = shape
                .iter()
                .try_fold(1usize, |a, &d| a.checked_mul(d))
                .ok_or_else(|| bad(format!("{name}: shape {shape:?} overflows")))?;
            if bf16.len() != numel {
                return Err(bad(format!(
                    "{name}: {} values for shape {shape:?}",
                    bf16.len()
                )));
            }
            let g =
                npy::parse(grads[name.as_str()]).map_err(|e| bad(format!("grad.{name}: {e}")))?;
            if g.shape != shape {
                return Err(bad(format!(
                    "grad.{name} has shape {:?}, the parameter {shape:?}",
                    g.shape
                )));
            }
            let grad = g
                .f32s()
                .map_err(|e| bad(format!("grad.{name}: {e}")))?
                .to_vec();
            if let Some(k) = grad.iter().position(|x| !x.is_finite()) {
                return Err(bad(format!("grad.{name}[{k}] is not finite")));
            }
            params.push(Param {
                name: name.clone(),
                shape,
                bf16,
                grad,
            });
        }
        let ids_a = npy::parse(b.ids).map_err(|e| bad(format!("ids: {e}")))?;
        if ids_a.shape.len() != 1 {
            return Err(bad(format!("ids has shape {:?}, want [T]", ids_a.shape)));
        }
        let ids = ids_a
            .i64s()
            .map_err(|e| bad(format!("ids: {e}")))?
            .iter()
            .map(|&t| {
                u32::try_from(t)
                    .ok()
                    .filter(|&t| (t as usize) < config.vocab)
                    .ok_or_else(|| {
                        bad(format!(
                            "token id {t} outside the {}-token vocabulary",
                            config.vocab
                        ))
                    })
            })
            .collect::<Result<Vec<u32>, _>>()?;
        if ids.len() < 2 {
            return Err(bad("fewer than two token ids: no next-token loss"));
        }
        let loss_a = npy::parse(b.loss).map_err(|e| bad(format!("loss: {e}")))?;
        let loss = match loss_a.f64s().map_err(|e| bad(format!("loss: {e}")))? {
            [l] if l.is_finite() => *l,
            other => return Err(bad(format!("loss {other:?}: want one finite value"))),
        };
        Ok(TinyFixture {
            config,
            params,
            ids,
            loss,
        })
    }

    /// The parameter called `name`.
    pub fn param(&self, name: &str) -> Option<&Param> {
        self.params.iter().find(|p| p.name == name)
    }
}

#[cfg(feature = "cuda")]
mod device {
    use super::{Param, TinyFixture};
    use crate::buffer::CudaBuffer;
    use crate::error::CudaError;
    use crate::runtime::CudaRuntime;

    /// One parameter on the device: its bf16 bits and its gradient.
    pub struct DeviceParam {
        pub name: String,
        pub shape: Vec<usize>,
        pub bf16: CudaBuffer<u16>,
        pub grad: CudaBuffer<f32>,
    }

    /// The fixture in device buffers.
    pub struct DeviceTinyFixture {
        pub params: Vec<DeviceParam>,
        pub ids: CudaBuffer<u32>,
    }

    impl TinyFixture {
        /// Upload every parameter, gradient and the token ids. Every buffer
        /// is held against the runtime's allocation budget.
        pub fn upload(&self, rt: &CudaRuntime) -> Result<DeviceTinyFixture, CudaError> {
            let params = self
                .params
                .iter()
                .map(|p: &Param| {
                    Ok(DeviceParam {
                        name: p.name.clone(),
                        shape: p.shape.clone(),
                        bf16: rt.upload(&p.bf16, &format!("tiny {}", p.name))?,
                        grad: rt.upload(&p.grad, &format!("tiny grad.{}", p.name))?,
                    })
                })
                .collect::<Result<Vec<_>, CudaError>>()?;
            Ok(DeviceTinyFixture {
                params,
                ids: rt.upload(&self.ids, "tiny ids")?,
            })
        }
    }
}

#[cfg(feature = "cuda")]
pub use device::{DeviceParam, DeviceTinyFixture};

/// The loader on the device, for `runga`: parse the embedded fixture, upload
/// it, read every buffer back and compare bit for bit.
#[cfg(feature = "cuda")]
pub fn loader_checks(rt: &crate::runtime::CudaRuntime) -> Vec<crate::check::Check> {
    use crate::check::{bitwise_check, diff_bits_bf16, diff_bits_f32, Check};
    const N: &str = "loader.tiny_fixture_published";
    let f = match TinyFixture::embedded() {
        Ok(f) => f,
        Err(e) => return vec![Check::from_error(&format!("{N}.parse"), &e)],
    };
    let dev = match f.upload(rt) {
        Ok(d) => d,
        Err(e) => return vec![Check::from_error(&format!("{N}.upload"), &e)],
    };
    let mut out = vec![Check::pass(
        &format!("{N}.parse"),
        format!(
            "{} parameters, {} token ids, loss {}",
            f.params.len(),
            f.ids.len(),
            f.loss
        ),
    )];
    let (mut want16, mut got16, mut wantg, mut gotg) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (p, d) in f.params.iter().zip(&dev.params) {
        if p.name != d.name || p.shape != d.shape {
            out.push(Check::fail(
                &format!("{N}.order"),
                format!("{} vs {}", p.name, d.name),
            ));
            return out;
        }
        match (rt.download(&d.bf16), rt.download(&d.grad)) {
            (Ok(a), Ok(b)) => {
                got16.extend(a);
                gotg.extend(b);
            }
            (Err(e), _) | (_, Err(e)) => {
                out.push(Check::from_error(&format!("{N}.download"), &e));
                return out;
            }
        }
        want16.extend_from_slice(&p.bf16);
        wantg.extend_from_slice(&p.grad);
    }
    out.push(bitwise_check(
        &format!("{N}.params_bf16"),
        diff_bits_bf16(&got16, &want16),
        want16.len(),
    ));
    out.push(bitwise_check(
        &format!("{N}.grads_f32"),
        diff_bits_f32(&gotg, &wantg),
        wantg.len(),
    ));
    out.push(match rt.download(&dev.ids) {
        Ok(ids) if ids == f.ids => Check::pass(&format!("{N}.ids"), format!("{} ids", ids.len())),
        Ok(_) => Check::fail(&format!("{N}.ids"), "token ids differ after the round trip"),
        Err(e) => Check::from_error(&format!("{N}.ids"), &e),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_train_published")
    }

    #[test]
    fn the_embedded_fixture_parses_into_the_2bs_shape_family() {
        let f = TinyFixture::embedded().unwrap();
        let c = &f.config;
        assert_eq!((c.hidden, c.intermediate, c.vocab), (64, 128, 64));
        assert_eq!(
            c.layers,
            [LayerKind::LinearAttention, LayerKind::FullAttention]
        );
        assert_eq!((c.heads, c.kv_heads, c.head_dim), (2, 1, 256));
        assert_eq!((c.linear_key_head_dim, c.linear_value_head_dim), (128, 128));
        assert_eq!(
            (c.conv_kernel, c.partial_rotary_factor, c.rope_theta),
            (4, 0.25, 10000.0)
        );
        assert!(c.tie_word_embeddings);
        assert_eq!(f.params.len(), 27);
        assert_eq!(f.ids.len(), 70);
        assert!(f.loss.is_finite() && f.loss > 0.0);
        let qkv = f
            .param("model.layers.0.linear_attn.in_proj_qkv.weight")
            .unwrap();
        assert_eq!(qkv.shape, [384, 64]);
        assert_eq!((qkv.bf16.len(), qkv.grad.len()), (384 * 64, 384 * 64));
        let embed = f.param("model.embed_tokens.weight").unwrap();
        assert_eq!(embed.shape, [64, 64]);
    }

    #[test]
    fn reading_the_directory_gives_the_embedded_fixture() {
        let a = TinyFixture::from_dir(&dir()).unwrap();
        assert_eq!(a, TinyFixture::embedded().unwrap());
    }

    #[test]
    fn a_tampered_fixture_is_refused() {
        let ok = || FixtureBytes {
            config: files::CONFIG,
            model: files::MODEL,
            ids: files::IDS,
            loss: files::LOSS,
            grads: files::GRADS
                .iter()
                .map(|(n, b)| (n.to_string(), *b))
                .collect(),
        };
        assert!(TinyFixture::parse(&ok()).is_ok());
        // A gradient missing, or one parameter's gradient under another's name.
        let mut b = ok();
        b.grads.pop();
        assert!(TinyFixture::parse(&b).is_err());
        let mut b = ok();
        b.grads[1].1 = files::GRADS[0].1;
        assert!(
            TinyFixture::parse(&b).is_err(),
            "a gradient of the wrong shape"
        );
        // An unknown config key; a truncated model file; ids as the loss.
        let cfg = String::from_utf8(files::CONFIG.to_vec()).unwrap().replacen(
            "\"use_cache\"",
            "\"use_kache\"",
            1,
        );
        let mut b = ok();
        b.config = cfg.as_bytes();
        assert!(TinyFixture::parse(&b).is_err());
        let mut b = ok();
        b.model = &files::MODEL[..files::MODEL.len() - 1];
        assert!(TinyFixture::parse(&b).is_err());
        let mut b = ok();
        b.loss = files::IDS;
        assert!(TinyFixture::parse(&b).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_file_is_refused() {
        let scratch = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("tiny-fixture-symlink-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).unwrap();
        for e in std::fs::read_dir(dir()).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), scratch.join(e.file_name())).unwrap();
        }
        assert!(TinyFixture::from_dir(&scratch).is_ok());
        let loss = scratch.join("loss.npy");
        std::fs::remove_file(&loss).unwrap();
        std::os::unix::fs::symlink(dir().join("loss.npy"), &loss).unwrap();
        let err = TinyFixture::from_dir(&scratch).unwrap_err().to_string();
        assert!(err.contains("symbolic link"), "{err}");
        std::fs::remove_dir_all(&scratch).unwrap();
    }
}
