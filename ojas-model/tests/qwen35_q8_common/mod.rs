//! Shared pieces of the resident q8 tests (`qwen35_q8_tape.rs`, `qwen35_q8_tape_metal.rs`,
//! `qwen35_q8_refusals.rs`): the tiny packed base (`tests/fixtures/qwen35_tiny_q8`, mlx 0.31.2
//! `mx.quantize(.., group_size=64, bits=8, stream=mx.gpu)`) and its fake-quant twin
//! (`qwen35_tiny_fq8`, `mx.dequantize` of the same call), a loader for each, the 16 wrapped parts
//! by name, and a way to write a damaged copy of the q8 file.
//!
//! The including test file declares `mod qwen35_common;` and `mod qwen35_q8_common;` beside each
//! other; this module reads the first through `crate::`.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ojas_autograd::Tape;
use ojas_core::{Backend, DType, PlacedAffine, Tensor};
use ojas_data::CounterRng;
use ojas_io::sha256::file_sha256_hex;
use ojas_io::{parse_json, write_safetensors, JsonValue, SafeTensors, StDtype, TensorOut};
use ojas_model::qwen35::adapters::{AdapterError, BaseIdentity, QuantStorage, QuantizerRecord};
use ojas_model::qwen35::{
    bind_frozen_base, forward_loss_lora_base, load_hf, load_hf_q8_pinned, AttnParams, FrozenWeight,
    GdnParams, LayerParams, LoraForward, MixerParams, Qwen35Params,
};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};

use crate::common::Probe;
use crate::qwen35_common::lora::{fixture_sites, FQ8};
use crate::qwen35_common::{budget, spec, IGNORE, PREFIX};

pub const Q8_DIR: &str = "qwen35_tiny_q8";
pub const FQ8_DIR: &str = "qwen35_tiny_fq8";
pub const MODEL_FILE: &str = "model.safetensors";

/// The wrapped weights the tiny tower has: layer 0 (gated delta net) `wq wk wv wz out` and the
/// MLP's three, layer 1 (attention) `wq wgate wk wv wo` and the MLP's three.
pub const WRAPPED_PARTS: usize = 16;

fn fixture(dir: &str, file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(dir)
        .join(file)
}

pub fn q8_path() -> PathBuf {
    fixture(Q8_DIR, MODEL_FILE)
}

pub fn fq8_path() -> PathBuf {
    fixture(FQ8_DIR, MODEL_FILE)
}

pub fn q8_file() -> SafeTensors<'static> {
    SafeTensors::open(&q8_path()).unwrap()
}

/// The text of `dir`'s `config.json`.
pub fn config_text(dir: &str) -> String {
    let path = fixture(dir, "config.json");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The q8 base's `config.json`, parsed: its `quantization` block says 8 bits, group size 64.
pub fn q8_config() -> JsonValue {
    parse_json(&config_text(Q8_DIR)).unwrap()
}

/// The fake-quant base's `config.json`: it carries no `quantization` block.
pub fn fq8_config() -> JsonValue {
    parse_json(&config_text(FQ8_DIR)).unwrap()
}

/// The record a resident q8 run pins its base with.
pub fn q8_record() -> QuantizerRecord {
    QuantizerRecord {
        format: "mlx-affine".to_string(),
        group_size: "64".to_string(),
        mlx_version: "0.31.2".to_string(),
        storage: QuantStorage::ResidentPacked,
        bits: Some(8),
    }
}

/// The base file at `path` as an identity: its own name, its sha256 and `record`.
pub fn identity_of(path: &Path, record: Option<QuantizerRecord>) -> BaseIdentity {
    BaseIdentity {
        file_name: path.file_name().unwrap().to_str().unwrap().to_string(),
        sha256: file_sha256_hex(path).unwrap(),
        quantizer: record,
    }
}

pub fn q8_identity() -> BaseIdentity {
    identity_of(&q8_path(), Some(q8_record()))
}

/// `load_hf_q8_pinned` of `path` under `identity` and `config`, dense tensors read as the fq8 arm
/// reads them (`Bf16` linears, `F32` table), the packed weights placed on `backend`.
pub fn load_q8_from<B: Backend + ?Sized>(
    path: &Path,
    identity: &BaseIdentity,
    config: &JsonValue,
    backend: &B,
) -> Result<Qwen35Params<Tensor, FrozenWeight<Tensor>>, AdapterError> {
    load_hf_q8_pinned(
        &spec(),
        path,
        identity,
        config,
        PREFIX,
        DType::Bf16,
        DType::F32,
        &budget(),
        backend,
    )
}

/// The tiny q8 base, placed on `backend`.
pub fn load_q8<B: Backend + ?Sized>(backend: &B) -> Qwen35Params<Tensor, FrozenWeight<Tensor>> {
    load_q8_from(&q8_path(), &q8_identity(), &q8_config(), backend).unwrap()
}

/// The fake-quant twin: every wrapped weight is `mx.dequantize` of the q8 base's, held dense in
/// bf16, and everything else is read the way the q8 loader reads it.
pub fn fq8_host() -> Qwen35Params<Tensor> {
    load_hf(
        &spec(),
        &SafeTensors::open(&fq8_path()).unwrap(),
        PREFIX,
        DType::Bf16,
        DType::F32,
        &budget(),
    )
    .unwrap()
}

/// The wrapped parts of a tower by name, in the tower's order.
pub fn wrapped_fields<V, W>(p: &Qwen35Params<V, W>) -> Vec<(String, &W)> {
    let mut out = Vec::new();
    for (i, l) in p.layers.iter().enumerate() {
        match &l.mixer {
            MixerParams::GatedDeltaNet(m) => {
                out.push((format!("layer{i}.gdn.wq"), &m.wq));
                out.push((format!("layer{i}.gdn.wk"), &m.wk));
                out.push((format!("layer{i}.gdn.wv"), &m.wv));
                out.push((format!("layer{i}.gdn.wz"), &m.wz));
                out.push((format!("layer{i}.gdn.out"), &m.out));
            }
            MixerParams::Attention(m) => {
                out.push((format!("layer{i}.attn.wq"), &m.wq));
                out.push((format!("layer{i}.attn.wgate"), &m.wgate));
                out.push((format!("layer{i}.attn.wk"), &m.wk));
                out.push((format!("layer{i}.attn.wv"), &m.wv));
                out.push((format!("layer{i}.attn.wo"), &m.wo));
            }
        }
        out.push((format!("layer{i}.mlp.gate"), &l.gate));
        out.push((format!("layer{i}.mlp.up"), &l.up));
        out.push((format!("layer{i}.mlp.down"), &l.down));
    }
    out
}

/// `f` of every wrapped weight, every other parameter cloned.
pub fn map_wrapped<V: Clone, W, X>(
    p: &Qwen35Params<V, W>,
    f: &mut impl FnMut(&W) -> X,
) -> Qwen35Params<V, X> {
    let layers = p
        .layers
        .iter()
        .map(|l| LayerParams {
            input_norm: l.input_norm.clone(),
            mixer: match &l.mixer {
                MixerParams::GatedDeltaNet(m) => MixerParams::GatedDeltaNet(GdnParams {
                    wq: f(&m.wq),
                    wk: f(&m.wk),
                    wv: f(&m.wv),
                    conv_q: m.conv_q.clone(),
                    conv_k: m.conv_k.clone(),
                    conv_v: m.conv_v.clone(),
                    wz: f(&m.wz),
                    wa: m.wa.clone(),
                    wb: m.wb.clone(),
                    a_log: m.a_log.clone(),
                    dt_bias: m.dt_bias.clone(),
                    norm: m.norm.clone(),
                    out: f(&m.out),
                }),
                MixerParams::Attention(m) => MixerParams::Attention(AttnParams {
                    wq: f(&m.wq),
                    wgate: f(&m.wgate),
                    wk: f(&m.wk),
                    wv: f(&m.wv),
                    q_norm: m.q_norm.clone(),
                    k_norm: m.k_norm.clone(),
                    wo: f(&m.wo),
                }),
            },
            post_norm: l.post_norm.clone(),
            gate: f(&l.gate),
            up: f(&l.up),
            down: f(&l.down),
        })
        .collect();
    Qwen35Params {
        embed: p.embed.clone(),
        layers,
        norm: p.norm.clone(),
    }
}

/// The bf16 `[rows, cols]` dequantisation of `q` on `backend`, where `backend` computes.
pub fn dequant<B: Backend + ?Sized>(backend: &B, q: &PlacedAffine) -> Tensor {
    backend
        .dequant_affine_bf16(q.packed(), q.scales(), q.biases(), q.spec())
        .unwrap()
}

/// The bits of a bf16 tensor read to the host through the backend's own download.
pub fn host_bits<B: Backend + ?Sized>(backend: &B, t: &Tensor) -> Vec<u16> {
    backend.download(t).unwrap().bf16_slice().unwrap().to_vec()
}

/// The "dequantise at bind" implementation this card replaces: every packed weight of `base`
/// becomes a dense bf16 `FrozenWeight::Dense` now, so a step dequantises nothing.
pub fn densified<B: Backend + ?Sized>(
    backend: &B,
    base: &Qwen35Params<Tensor, FrozenWeight<Tensor>>,
) -> Qwen35Params<Tensor, FrozenWeight<Tensor>> {
    map_wrapped(base, &mut |w| match w {
        FrozenWeight::Dense(t) => FrozenWeight::Dense(t.clone()),
        FrozenWeight::Quant(q) => FrozenWeight::Dense(dequant(backend, q)),
    })
}

static DIRS: AtomicU64 = AtomicU64::new(0);

/// A fresh scratch directory under the system temp dir, removed on drop.
pub struct Tmp(pub PathBuf);

impl Tmp {
    pub fn new(tag: &str) -> Self {
        let n = DIRS.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("ojas-q8-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }

    /// `<dir>/model.safetensors`, the name the q8 identity carries.
    pub fn model(&self) -> PathBuf {
        self.0.join(MODEL_FILE)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// What [`rewrite`] does with one tensor of the source file.
pub enum Edit {
    Keep,
    Drop,
    Replace(StDtype, Vec<u64>, Vec<u8>),
}

/// A tensor [`rewrite`] adds: name, dtype, shape and bytes.
pub type Extra = (String, StDtype, Vec<u64>, Vec<u8>);

/// A copy of `src` at `out` where `edit(name, dtype, shape)` says what each tensor becomes, plus
/// the tensors of `extra`.
pub fn rewrite(
    src: &SafeTensors<'_>,
    out: &Path,
    edit: &mut dyn FnMut(&str, StDtype, &[u64]) -> Edit,
    extra: Vec<Extra>,
) {
    let mut names: Vec<String> = src.names().map(str::to_string).collect();
    names.sort();
    let mut kept: Vec<(String, StDtype, Vec<u64>, Vec<u8>)> = Vec::new();
    for name in names {
        let (info, bytes) = src.read_raw(&name).unwrap();
        match edit(&name, info.dtype, &info.shape) {
            Edit::Keep => kept.push((name, info.dtype, info.shape.clone(), bytes)),
            Edit::Drop => {}
            Edit::Replace(dtype, shape, data) => kept.push((name, dtype, shape, data)),
        }
    }
    kept.extend(extra);
    let items: Vec<TensorOut<'_>> = kept
        .iter()
        .map(|(name, dtype, shape, data)| TensorOut {
            name,
            dtype: *dtype,
            shape,
            data,
        })
        .collect();
    write_safetensors(out, &items, &[]).unwrap();
}

/// The op name every backend reports for a dequantisation.
pub const DEQUANT: &str = "dequant_affine_bf16";

/// The dequantisations counted after the bind, after the forward and after the backward.
pub struct Counted {
    pub after_bind: usize,
    pub after_forward: usize,
    pub after_backward: usize,
}

/// One step over `base` on a counting backend wrapped round `inner`. With `o1` the base is first
/// made dense, as the "dequantise at bind" implementation would (the control that proves the
/// counts can tell). `base` must be placed where `inner` computes.
pub fn counted_step<B: Backend>(
    inner: B,
    base: &Qwen35Params<Tensor, FrozenWeight<Tensor>>,
    activations: ActivationCheckpoint,
    o1: bool,
) -> Counted {
    let probe = Probe::new(inner);
    let fx = FQ8;
    let (d, sites) = (fx.data(), fixture_sites(&fx.file(), true));
    let dense;
    let used = if o1 {
        dense = densified(&probe, base);
        &dense
    } else {
        base
    };
    let mut tape = Tape::new(&probe);
    let params = bind_frozen_base(&mut tape, used).unwrap();
    let after_bind = probe.count(DEQUANT);
    let bound = sites
        .try_map(&mut |site| site.try_map(&mut |t| tape.leaf(t.clone())))
        .unwrap();
    let mut rng = CounterRng::new(0);
    let [b, t] = [d.ids.shape()[0], d.ids.shape()[1]];
    let fwd = LoraForward::new(&mut tape, bound, &fx.config(), b, t, &mut rng, &budget()).unwrap();
    let loss = forward_loss_lora_base(
        &mut tape,
        &spec(),
        &params,
        Some(&fwd),
        &d.ids,
        &d.targets,
        &d.tables,
        Some(IGNORE),
        DEFAULT_CE_CHUNK,
        activations,
    )
    .unwrap();
    let after_forward = probe.count(DEQUANT);
    tape.backward(loss).unwrap();
    Counted {
        after_bind,
        after_forward,
        after_backward: probe.count(DEQUANT),
    }
}

/// Bytes of the dense (host) tensors of a base: everything but the packed weights.
pub fn dense_host_bytes(base: &Qwen35Params<Tensor, FrozenWeight<Tensor>>) -> u64 {
    let mut total = 0u64;
    base.try_map_base(
        &mut |t| {
            total += (t.num_elements().unwrap() * t.dtype().size()) as u64;
            Ok::<_, ()>(())
        },
        &mut |q| Ok(q.clone()),
    )
    .unwrap();
    total
}

/// Bytes the packed weights of a base hold.
pub fn packed_bytes(base: &Qwen35Params<Tensor, FrozenWeight<Tensor>>) -> u64 {
    wrapped_fields(base)
        .into_iter()
        .map(|(_, w)| match w {
            FrozenWeight::Quant(q) => q.nbytes().unwrap() as u64,
            FrozenWeight::Dense(t) => (t.num_elements().unwrap() * t.dtype().size()) as u64,
        })
        .sum()
}
