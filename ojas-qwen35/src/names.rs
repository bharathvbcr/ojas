//! The text tower's tensors as transformers names and shapes them, and the
//! header pre-flight a snapshot passes before tessl reads a byte of it.
//!
//! [`tower_tensors`] is written in the order of tessl's
//! `Qwen35Model::parameter_table` (`tessl/src/qwen35_params.rs`, `slots`).
//! That order is only a pre-flight convenience here: once a model is open,
//! every per-entry vector is rebuilt against tessl's live table by name, so a
//! drift between the two cannot put a weight decay on the wrong parameter
//! (and `Qwen35Step::open` refuses a live table that differs from this one).

use std::collections::BTreeMap;
use std::path::Path;

use ojas_io::{SafeTensors, StDtype};

use crate::config::{LayerKind, Qwen35TextConfig};
use crate::error::{Qwen35Error, Result};

/// One parameter: transformers' name below the tower prefix
/// (`layers.3.mlp.gate_proj.weight`) and transformers' shape (`[out, in]`
/// for a linear layer).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

impl TensorSpec {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// The layer index, for a `layers.{l}.` tensor.
    pub fn layer(&self) -> Option<u32> {
        let rest = self.name.strip_prefix("layers.")?;
        rest.split('.').next()?.parse().ok()
    }
}

fn spec(name: String, shape: &[usize]) -> TensorSpec {
    TensorSpec {
        name,
        shape: shape.to_vec(),
    }
}

/// Every text-tower parameter, in tessl's parameter-table order.
pub fn tower_tensors(cfg: &Qwen35TextConfig) -> Vec<TensorSpec> {
    let (h, inter, vocab) = (
        cfg.hidden as usize,
        cfg.intermediate as usize,
        cfg.vocab as usize,
    );
    let key_dim = (cfg.gdn_key_heads * cfg.gdn_key_dim) as usize;
    let value_dim = (cfg.gdn_value_heads * cfg.gdn_value_dim) as usize;
    let conv_dim = 2 * key_dim + value_dim;
    let vh = cfg.gdn_value_heads as usize;
    let d = cfg.head_dim as usize;
    let (q, kv) = (cfg.q_heads as usize, cfg.kv_heads as usize);

    let mut out = vec![spec("embed_tokens.weight".into(), &[vocab, h])];
    for (l, kind) in cfg.layers.iter().enumerate() {
        let p = |s: &str| format!("layers.{l}.{s}");
        match kind {
            LayerKind::LinearAttention => {
                out.push(spec(p("linear_attn.in_proj_qkv.weight"), &[conv_dim, h]));
                out.push(spec(p("linear_attn.in_proj_z.weight"), &[value_dim, h]));
                out.push(spec(p("linear_attn.in_proj_b.weight"), &[vh, h]));
                out.push(spec(p("linear_attn.in_proj_a.weight"), &[vh, h]));
                out.push(spec(p("linear_attn.out_proj.weight"), &[h, value_dim]));
                out.push(spec(
                    p("linear_attn.conv1d.weight"),
                    &[conv_dim, 1, cfg.conv_kernel as usize],
                ));
                out.push(spec(p("linear_attn.A_log"), &[vh]));
                out.push(spec(p("linear_attn.dt_bias"), &[vh]));
                out.push(spec(
                    p("linear_attn.norm.weight"),
                    &[cfg.gdn_value_dim as usize],
                ));
            }
            LayerKind::FullAttention => {
                // q_proj holds the query and its output gate per head.
                out.push(spec(p("self_attn.q_proj.weight"), &[2 * q * d, h]));
                out.push(spec(p("self_attn.k_proj.weight"), &[kv * d, h]));
                out.push(spec(p("self_attn.v_proj.weight"), &[kv * d, h]));
                out.push(spec(p("self_attn.o_proj.weight"), &[h, q * d]));
                out.push(spec(p("self_attn.q_norm.weight"), &[d]));
                out.push(spec(p("self_attn.k_norm.weight"), &[d]));
            }
        }
        out.push(spec(p("mlp.gate_proj.weight"), &[inter, h]));
        out.push(spec(p("mlp.up_proj.weight"), &[inter, h]));
        out.push(spec(p("mlp.down_proj.weight"), &[h, inter]));
        out.push(spec(p("input_layernorm.weight"), &[h]));
        out.push(spec(p("post_attention_layernorm.weight"), &[h]));
    }
    out.push(spec("norm.weight".into(), &[h]));
    out
}

/// What a weights file holds besides the tower.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeaderReport {
    /// Tower tensors found, each with its expected shape and a float dtype.
    pub tower_tensors: usize,
    /// Tensors outside the tower prefix, by their first two name segments
    /// (`model.visual`, `mtp.layers`), with counts. Not loaded.
    pub not_loaded: BTreeMap<String, usize>,
}

/// Check `names` (with each tensor's dtype and shape, from a header) against
/// the tower `cfg` describes. Every tower tensor must be present under
/// `cfg.tower_prefix` with its shape and a float dtype (F32, BF16 or F16:
/// tessl widens each to f32 exactly). A name under the prefix that is not a
/// tower tensor is refused (it would be a parameter the step silently does
/// not train), and so is an `lm_head.weight` anywhere (the config ties it).
pub fn check_header<'a>(
    cfg: &Qwen35TextConfig,
    entries: impl IntoIterator<Item = (&'a str, StDtype, &'a [u64])>,
    what: &str,
) -> Result<HeaderReport> {
    let expected: BTreeMap<String, Vec<usize>> = tower_tensors(cfg)
        .into_iter()
        .map(|t| (format!("{}{}", cfg.tower_prefix, t.name), t.shape))
        .collect();
    let mut found = 0usize;
    let mut not_loaded: BTreeMap<String, usize> = BTreeMap::new();
    let mut seen = std::collections::BTreeSet::new();
    for (name, dtype, shape) in entries {
        if name == "lm_head.weight" || name.ends_with(".lm_head.weight") {
            return Err(Qwen35Error::Config(format!(
                "{what}: {name} is present, but the config ties the LM head to the embedding"
            )));
        }
        match expected.get(name) {
            Some(want) => {
                let got: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                if &got != want {
                    return Err(Qwen35Error::Config(format!(
                        "{what}: {name} has shape {got:?}, the config says {want:?}"
                    )));
                }
                if !matches!(dtype, StDtype::F32 | StDtype::BF16 | StDtype::F16) {
                    return Err(Qwen35Error::Config(format!(
                        "{what}: {name} is {dtype:?}, not a float the loader widens to f32"
                    )));
                }
                seen.insert(name.to_string());
                found += 1;
            }
            None if name.starts_with(&cfg.tower_prefix) => {
                return Err(Qwen35Error::Config(format!(
                    "{what}: {name} is under the tower prefix {:?} but is not a parameter of the tower \
                     the config describes; refusing a tensor the step would silently not train",
                    cfg.tower_prefix
                )));
            }
            None => {
                let group: Vec<&str> = name.splitn(3, '.').take(2).collect();
                *not_loaded.entry(group.join(".")).or_insert(0) += 1;
            }
        }
    }
    let missing: Vec<&String> = expected.keys().filter(|k| !seen.contains(*k)).collect();
    if !missing.is_empty() {
        return Err(Qwen35Error::Config(format!(
            "{what}: {} tower tensors are missing, first {:?}",
            missing.len(),
            missing.iter().take(5).collect::<Vec<_>>()
        )));
    }
    Ok(HeaderReport {
        tower_tensors: found,
        not_loaded,
    })
}

/// [`check_header`] on a checkpoint: a `.safetensors` file, or a
/// `model.safetensors.index.json` whose every shard is read (header only,
/// ojas-io) and checked together. A name in two shards is refused.
pub fn check_weights_file(cfg: &Qwen35TextConfig, path: &Path) -> Result<HeaderReport> {
    let shards = if path.extension().is_some_and(|e| e == "json") {
        shard_paths(cfg, path)?
    } else {
        vec![path.to_path_buf()]
    };
    let mut infos: Vec<(String, StDtype, Vec<u64>)> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for shard in &shards {
        let st = SafeTensors::open(shard)
            .map_err(|e| Qwen35Error::Io(format!("{}: {e}", shard.display())))?;
        for n in st.names() {
            if !seen.insert(n.to_string()) {
                return Err(Qwen35Error::Config(format!(
                    "{}: tensor {n} is in more than one shard",
                    path.display()
                )));
            }
            let i = st.info(n).map_err(|e| Qwen35Error::Io(e.to_string()))?;
            infos.push((n.to_string(), i.dtype, i.shape.clone()));
        }
    }
    check_header(
        cfg,
        infos.iter().map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
        &path.display().to_string(),
    )
}

/// A snapshot resolved to the config and the checkpoint the provider loads,
/// validated on the CPU: the config parsed and refused where it must be, and
/// every shard's header matched against the tower. Nothing here touches the
/// GPU.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub config_path: std::path::PathBuf,
    /// What `tessl::safetensors::CheckpointView::open` takes: a single
    /// `.safetensors` file, or a snapshot's `model.safetensors.index.json`.
    pub weights_path: std::path::PathBuf,
    pub config: Qwen35TextConfig,
    pub header: HeaderReport,
}

impl Snapshot {
    /// A Hugging Face snapshot directory: `config.json`, and either
    /// `model.safetensors.index.json` (any number of shards, symlinked ones
    /// included) or `model.safetensors`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let config_path = dir.join("config.json");
        let config = Qwen35TextConfig::from_file(&config_path)?;
        let index = dir.join("model.safetensors.index.json");
        let weights_path = if index.is_file() {
            index
        } else {
            let single = dir.join("model.safetensors");
            if !single.is_file() {
                return Err(Qwen35Error::Io(format!(
                    "{}: neither model.safetensors.index.json nor model.safetensors",
                    dir.display()
                )));
            }
            single
        };
        Self::from_parts(config_path, weights_path, config)
    }

    /// An explicit config and a weights file (a `.safetensors` file or an index).
    pub fn from_files(config_json: &Path, weights: &Path) -> Result<Self> {
        let config = Qwen35TextConfig::from_file(config_json)?;
        Self::from_parts(config_json.to_path_buf(), weights.to_path_buf(), config)
    }

    fn from_parts(
        config_path: std::path::PathBuf,
        weights_path: std::path::PathBuf,
        config: Qwen35TextConfig,
    ) -> Result<Self> {
        let header = check_weights_file(&config, &weights_path)?;
        Ok(Self {
            config_path,
            weights_path,
            config,
            header,
        })
    }
}

/// The shard files an index's `weight_map` names, as paths beside the index.
/// Each must be a bare file name (the same lexical rule tessl applies), and
/// the map must hold at least one tensor under the tower prefix.
fn shard_paths(cfg: &Qwen35TextConfig, index: &Path) -> Result<Vec<std::path::PathBuf>> {
    let text = std::fs::read_to_string(index)
        .map_err(|e| Qwen35Error::Io(format!("{}: {e}", index.display())))?;
    let what = index.display().to_string();
    let root = crate::config::parse_hf_json(&text, &what).map_err(Qwen35Error::Config)?;
    let map = root
        .field_opt("weight_map")
        .ok()
        .flatten()
        .and_then(|m| m.as_object().ok())
        .ok_or_else(|| Qwen35Error::Config(format!("{what}: no weight_map object")))?;
    let mut files = std::collections::BTreeSet::new();
    let mut tower = false;
    for (name, file) in map {
        tower |= name.starts_with(&cfg.tower_prefix);
        match file {
            ojas_io::JsonValue::String(f) => {
                if f.contains('/') || f.contains('\\') || f == ".." || f.is_empty() {
                    return Err(Qwen35Error::Config(format!(
                        "{what}: {name} maps to {f:?}, not a file name"
                    )));
                }
                files.insert(f.clone());
            }
            v => {
                return Err(Qwen35Error::Config(format!(
                    "{what}: {name} maps to {}",
                    crate::config::shown(v)
                )))
            }
        }
    }
    if !tower {
        return Err(Qwen35Error::Config(format!(
            "{what}: no tensor under {:?}",
            cfg.tower_prefix
        )));
    }
    let dir = index.parent().unwrap_or_else(|| Path::new("."));
    Ok(files.into_iter().map(|f| dir.join(f)).collect())
}
