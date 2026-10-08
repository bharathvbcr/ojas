//! HF safetensors → Q4/Q8 affine Hot banks (bf16 optional debug).
//!
//! Loads text-backbone tensors, splits PLE per layer (Metal 4 GiB), and keeps
//! host-side banks ready for Metal Hot upload (Phase 2+ encode path).

use crate::config::{Gemma4Config, Gemma4TextConfig};
use crate::diag::{self, Stage};
use crate::error::{Error, Result};
use crate::kv::KvLayout;
use crate::ple::{
    build_ple_banks_from_packed_bf16, PleBanks, PlePackLayout, METAL_MAX_BUFFER_BYTES,
};
use crate::quant::{
    bf16_bits_to_f32, quant_matrix_from_mlx_q4, quantize_affine_bf16_bits, quantize_affine_f32,
    QuantMatrix, QuantScheme,
};
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Named host weight after quant.
#[derive(Clone, Debug)]
pub struct NamedMatrix {
    pub name: String,
    pub matrix: QuantMatrix,
}

/// Host-side model banks (zero Metal alloc until upload).
#[derive(Clone, Debug)]
pub struct HostWeightBanks {
    pub config: Gemma4TextConfig,
    pub scheme: QuantScheme,
    /// Non-PLE tensors (embed, layers, norms, lm_head if untied).
    pub matrices: Vec<NamedMatrix>,
    /// Per-layer PLE (empty for 31B).
    pub ple: Option<PleBanks>,
    pub kv_layout: KvLayout,
}

impl HostWeightBanks {
    pub fn total_hot_bytes(&self) -> usize {
        let m: usize = self.matrices.iter().map(|m| m.matrix.nbytes_hot()).sum();
        let p = self.ple.as_ref().map(|p| p.total_hot_bytes()).unwrap_or(0);
        m + p
    }

    pub fn validate_metal_limits(&self) -> Result<()> {
        for m in &self.matrices {
            if m.matrix.nbytes_hot() > METAL_MAX_BUFFER_BYTES {
                return Err(Error::Weights(format!(
                    "tensor '{}' ({} bytes) exceeds Metal 4GiB limit — split required",
                    m.name,
                    m.matrix.nbytes_hot()
                )));
            }
        }
        if let Some(ple) = &self.ple {
            ple.validate_metal_limit()?;
        }
        Ok(())
    }

    pub fn find(&self, name: &str) -> Option<&QuantMatrix> {
        self.matrices
            .iter()
            .find(|m| m.name == name)
            .map(|m| &m.matrix)
    }

    pub fn require(&self, name: &str) -> Result<&QuantMatrix> {
        self.find(name)
            .ok_or_else(|| Error::Weights(format!("missing weight '{name}'")))
    }
}

/// Load options.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    pub scheme: QuantScheme,
    pub max_seq: usize,
    pub ple_layout: PlePackLayout,
    /// If true, skip tensors whose names contain vision/audio prefixes.
    pub text_only: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            scheme: QuantScheme::q4_default(),
            max_seq: 4096,
            ple_layout: PlePackLayout::InterleavedPerToken,
            text_only: true,
        }
    }
}

/// Load from a HuggingFace model directory (`config.json` + `*.safetensors`).
///
/// Auto-detects MLX affine Q4 (`weight` U32 + `scales`/`biases`) vs bf16/f32 HF.
pub fn load_from_hf_dir(dir: impl AsRef<Path>, opts: LoadOptions) -> Result<HostWeightBanks> {
    let dir = dir.as_ref();
    let stage = Stage::begin("weights", format!("load_from_hf_dir {}", dir.display()));
    diag::log(
        "weights",
        format_args!(
            "opts scheme={:?} max_seq={} text_only={} ple_layout={:?}",
            opts.scheme, opts.max_seq, opts.text_only, opts.ple_layout
        ),
    );
    let cfg = match Gemma4Config::from_path(dir.join("config.json")) {
        Ok(c) => c,
        Err(e) => {
            diag::err("weights", "config.json", &e);
            stage.fail(&e);
            return Err(e);
        }
    };
    let text = cfg.text().clone();
    diag::log(
        "weights",
        format_args!(
            "config layers={} hidden={} vocab={} intermediate={} ple={} k_eq_v={}",
            text.num_hidden_layers,
            text.hidden_size,
            text.vocab_size,
            text.intermediate_size,
            text.has_ple(),
            text.attention_k_eq_v
        ),
    );
    let files = list_safetensor_files(dir)?;
    if files.is_empty() {
        let e = Error::Weights(format!("no .safetensors in {}", dir.display()));
        diag::err("weights", "shard list empty", &e);
        stage.fail(&e);
        return Err(e);
    }
    let mut total_bytes = 0u64;
    for f in &files {
        let sz = fs::metadata(f).map(|m| m.len()).unwrap_or(0);
        total_bytes += sz;
        diag::log(
            "weights",
            format_args!(
                "shard {} size={}",
                f.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
                diag::fmt_bytes(sz)
            ),
        );
    }
    diag::log(
        "weights",
        format_args!(
            "found {} safetensor file(s), total={}",
            files.len(),
            diag::fmt_bytes(total_bytes)
        ),
    );
    // Prefer a single shard / primary file; MLX E4B ships one `model.safetensors`.
    let primary = files
        .iter()
        .find(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|s| s == "model.safetensors" || s.ends_with(".safetensors"))
                .unwrap_or(false)
        })
        .unwrap_or(&files[0]);

    // Prefer config.json quantization hint (avoids double-reading 5GB safetensors).
    let mlx_q4 = dir
        .join("config.json")
        .exists()
        .then(|| fs::read_to_string(dir.join("config.json")).ok())
        .flatten()
        .and_then(|s| {
            let v: serde_json::Value = serde_json::from_str(&s).ok()?;
            let q = v.get("quantization").or_else(|| v.get("quantization_config"))?;
            let bits = q.get("bits")?.as_u64()?;
            let mode = q.get("mode").and_then(|m| m.as_str()).unwrap_or("affine");
            Some(bits == 4 && mode == "affine")
        })
        .unwrap_or(false);
    diag::log(
        "weights",
        format_args!(
            "quant detection: config_mlx_q4={mlx_q4} primary={}",
            primary.display()
        ),
    );

    if mlx_q4 || is_mlx_affine_file(primary)? {
        // 31B ships sharded `model-0000N-of-00004.safetensors` — load every shard.
        let mlx_files: Vec<PathBuf> = files
            .iter()
            .filter(|p| {
                is_mlx_affine_file(p).unwrap_or(false)
                    || p.file_name()
                        .and_then(|s| s.to_str())
                        .map(|s| s.ends_with(".safetensors"))
                        .unwrap_or(false)
            })
            .cloned()
            .collect();
        let load_files = if mlx_files.is_empty() {
            vec![primary.clone()]
        } else {
            mlx_files
        };
        match load_mlx_affine_q4_shards(dir, &load_files, text, opts) {
            Ok(b) => {
                stage.ok();
                Ok(b)
            }
            Err(e) => {
                stage.fail(&e);
                Err(e)
            }
        }
    } else {
        let mut tensors: HashMap<String, TensorCpu> = HashMap::new();
        for f in &files {
            load_safetensors_file(f, &mut tensors)?;
        }
        match build_banks_from_tensors(text, tensors, opts) {
            Ok(b) => {
                stage.ok();
                Ok(b)
            }
            Err(e) => {
                stage.fail(&e);
                Err(e)
            }
        }
    }
}

/// Resolve HF hub snapshot dir for `mlx-community/gemma-4-e4b-it-4bit` if cached.
pub fn resolve_default_e4b_mlx_cache() -> Option<PathBuf> {
    resolve_hub_snapshot("models--mlx-community--gemma-4-e4b-it-4bit", |snap| {
        snap.join("model.safetensors").exists() && snap.join("config.json").exists()
    })
}

/// Resolve HF hub snapshot for `mlx-community/gemma-4-31b-it-4bit` (sharded).
pub fn resolve_default_31b_mlx_cache() -> Option<PathBuf> {
    resolve_hub_snapshot("models--mlx-community--gemma-4-31b-it-4bit", |snap| {
        snap.join("config.json").exists()
            && (snap.join("model.safetensors").exists()
                || snap.join("model-00001-of-00004.safetensors").exists())
    })
}

/// Resolve HF hub snapshot for `z-lab/gemma-4-31B-it-DFlash` if cached.
pub fn resolve_default_dflash_draft_cache() -> Option<PathBuf> {
    resolve_hub_snapshot("models--z-lab--gemma-4-31B-it-DFlash", |snap| {
        snap.join("model.safetensors").exists() && snap.join("config.json").exists()
    })
}

/// Resolve HF hub snapshot for `google/gemma-4-E4B-it-assistant`.
pub fn resolve_default_e4b_assistant_cache() -> Option<PathBuf> {
    resolve_hub_snapshot("models--google--gemma-4-E4B-it-assistant", |snap| {
        snap.join("model.safetensors").exists() && snap.join("config.json").exists()
    })
}

fn resolve_hub_snapshot(
    model_dir: &str,
    ok: impl Fn(&Path) -> bool,
) -> Option<PathBuf> {
    let hub = match dirs_hub_cache() {
        Some(h) => h,
        None => {
            diag::log(
                "cache",
                format_args!("no HF hub cache dir (HOME/HF_HOME unset) for {model_dir}"),
            );
            return None;
        }
    };
    let root = hub.join(model_dir);
    let snaps = root.join("snapshots");
    diag::log(
        "cache",
        format_args!("resolve {model_dir} under {}", snaps.display()),
    );
    let rd = match fs::read_dir(&snaps) {
        Ok(rd) => rd,
        Err(e) => {
            diag::log(
                "cache",
                format_args!("snapshots missing for {model_dir}: {e} (path={})", snaps.display()),
            );
            return None;
        }
    };
    let mut dirs: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    let snap = match dirs.into_iter().next_back() {
        Some(s) => s,
        None => {
            diag::log("cache", format_args!("empty snapshots for {model_dir}"));
            return None;
        }
    };
    // Surface incomplete downloads: list *.safetensors sizes under the snapshot.
    if let Ok(entries) = fs::read_dir(&snap) {
        let mut n = 0u32;
        let mut bytes = 0u64;
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                n += 1;
                bytes += e.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
        diag::log(
            "cache",
            format_args!(
                "snapshot {} has {n} .safetensors totaling {}",
                snap.display(),
                diag::fmt_bytes(bytes)
            ),
        );
    }
    if ok(&snap) {
        diag::log("cache", format_args!("OK → {}", snap.display()));
        Some(snap)
    } else {
        diag::log(
            "cache",
            format_args!(
                "REJECT incomplete/missing files under {} (config or weight shard check failed)",
                snap.display()
            ),
        );
        None
    }
}

fn dirs_hub_cache() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(p).join("hub"));
    }
    if let Ok(p) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".cache/huggingface/hub"))
}

fn is_mlx_affine_file(path: &Path) -> Result<bool> {
    // Header-only probe: first 8 bytes = header len, then JSON header.
    let mut f = fs::File::open(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    use std::io::Read;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)
        .map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    if header_len > 64 * 1024 * 1024 {
        return Ok(false);
    }
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header)
        .map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    let v: serde_json::Value = serde_json::from_slice(&header)
        .map_err(|e| Error::Safetensors(format!("header json: {e}")))?;
    let obj = v.as_object().ok_or_else(|| Error::Safetensors("header not object".into()))?;
    for (name, meta) in obj {
        if name == "__metadata__" || !name.contains("language_model") {
            continue;
        }
        if !name.ends_with(".weight") {
            continue;
        }
        if meta.get("dtype").and_then(|d| d.as_str()) == Some("U32") {
            let scales = name.trim_end_matches("weight").to_string() + "scales";
            if obj.contains_key(&scales) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Load MLX-community affine Q4 text backbone into Hot-ready banks.
///
/// Skips vision/audio. PLE table is **not** split into Hot banks yet (Metal 4GB /
/// graph gap) — decode still runs full 42-layer QKV/MLP/lm_head path without PLE.
pub fn load_mlx_affine_q4(
    dir: &Path,
    safetensors_path: &Path,
    text: Gemma4TextConfig,
    opts: LoadOptions,
) -> Result<HostWeightBanks> {
    load_mlx_affine_q4_shards(dir, &[safetensors_path.to_path_buf()], text, opts)
}

/// Multi-shard MLX Q4 load (E4B single file or 31B 0000N-of-00004).
pub fn load_mlx_affine_q4_shards(
    dir: &Path,
    safetensors_paths: &[PathBuf],
    mut text: Gemma4TextConfig,
    opts: LoadOptions,
) -> Result<HostWeightBanks> {
    text.validate()?;
    let kv_layout = KvLayout::from_config(&text, opts.max_seq)?;
    let group_size = match opts.scheme {
        QuantScheme::Q4Mlx { group_size } => group_size,
        QuantScheme::Q4 { group_size } => group_size,
        _ => 64,
    };
    let scheme = QuantScheme::Q4Mlx { group_size };
    diag::log(
        "weights",
        format_args!(
            "MLX affine Q4 load: {} shard(s) group_size={group_size} dir={}",
            safetensors_paths.len(),
            dir.display()
        ),
    );

    let mut matrices = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut keys_seen_total = 0usize;
    let t_all = Instant::now();
    let mut ple_packed: Option<(Vec<u32>, Vec<f32>, Vec<f32>, Vec<usize>)> = None; // w,s,b,shape

    for (si, safetensors_path) in safetensors_paths.iter().enumerate() {
        let t_shard = Instant::now();
        let meta_len = fs::metadata(safetensors_path)
            .map(|m| m.len())
            .unwrap_or(0);
        diag::log(
            "weights",
            format_args!(
                "open shard [{}/{}] {} ({})",
                si + 1,
                safetensors_paths.len(),
                safetensors_path.display(),
                diag::fmt_bytes(meta_len)
            ),
        );
        let bytes = fs::read(safetensors_path).map_err(|e| {
            let err = Error::Io(format!("{}: {e}", safetensors_path.display()));
            diag::err("weights", "shard read", &err);
            err
        })?;
        diag::log(
            "weights",
            format_args!(
                "read {} bytes in {:.1} ms",
                diag::fmt_bytes(bytes.len() as u64),
                t_shard.elapsed().as_secs_f64() * 1e3
            ),
        );
        let st = SafeTensors::deserialize(&bytes).map_err(|e| {
            let err = Error::Safetensors(format!("{}: {e}", safetensors_path.display()));
            diag::err("weights", "safetensors deserialize", &err);
            err
        })?;
        let key_count = st.names().len();
        keys_seen_total += key_count;
        diag::log("weights", format_args!("shard header keys={key_count}"));
        let mut shard_mats = 0usize;

        for name in st.names() {
            if opts.text_only && is_non_text_key(name) {
                continue;
            }
            // E4B MLX uses `language_model.*`; 31B dense may be bare `model.*`.
            let is_text = name.contains("language_model")
                || name.starts_with("model.layers")
                || name.starts_with("model.embed")
                || name.starts_with("model.norm")
                || name == "lm_head.weight"
                || name.starts_with("embed_tokens")
                || name.starts_with("layers.")
                || name == "norm.weight";
            if !is_text {
                continue;
            }
            // MLX shards name most tensors `*.weight`, but Gemma4 also stores
            // per-layer `layer_scalar` (bf16 rank-1). Skipping non-`.weight`
            // previously left scalars at 1.0 and poisoned DFlash captures.
            let is_layer_scalar = name.ends_with(".layer_scalar") || name.ends_with("layer_scalar");
            if !name.ends_with(".weight") && !is_layer_scalar {
                continue;
            }
            if name.contains("embed_tokens_per_layer") {
                let t = st
                    .tensor(name)
                    .map_err(|e| Error::Safetensors(e.to_string()))?;
                if name.ends_with(".weight") && t.dtype() == safetensors::Dtype::U32 {
                    let scales_name = name.trim_end_matches("weight").to_string() + "scales";
                    let biases_name = name.trim_end_matches("weight").to_string() + "biases";
                    let w_u32 = tensor_as_u32(&t)?;
                    let scales = lookup_f32_tensor_across(
                        &st,
                        safetensors_path,
                        safetensors_paths,
                        &scales_name,
                    )?;
                    let biases = lookup_f32_tensor_across(
                        &st,
                        safetensors_path,
                        safetensors_paths,
                        &biases_name,
                    )?;
                    let shape = t.shape().to_vec();
                    diag::log(
                        "weights",
                        format_args!("captured MLX Q4 PLE packed shape={shape:?}"),
                    );
                    ple_packed = Some((w_u32, scales, biases, shape));
                }
                continue;
            }
            if name.contains("per_layer_input_gate") || name.contains("per_layer_projection") {
                continue;
            }
            let t = st
                .tensor(name)
                .map_err(|e| Error::Safetensors(e.to_string()))?;
            let nk = normalize_key(name);
            if seen.contains(&nk) {
                continue;
            }
            if should_skip_tensor(&nk) {
                continue;
            }
            if let Some(layer) = parse_layer_index(&nk) {
                if kv_layout.is_consumer(layer)
                    && (nk.contains("k_proj") || nk.contains("v_proj"))
                {
                    continue;
                }
            }

            match t.dtype() {
                safetensors::Dtype::U32 => {
                    let scales_name = name.trim_end_matches("weight").to_string() + "scales";
                    let biases_name = name.trim_end_matches("weight").to_string() + "biases";
                    // 31B shards occasionally place scales/biases in a neighboring
                    // file (e.g. layer 16 down_proj.biases in 00002 while weight is in 00001).
                    let scales = lookup_f32_tensor_across(
                        &st,
                        safetensors_path,
                        safetensors_paths,
                        &scales_name,
                    )
                    .map_err(|e| {
                        diag::err_msg(
                            "weights",
                            &format!("MLX Q4 missing scales for {name}"),
                            &e,
                        );
                        Error::Weights(format!("MLX Q4 missing scales for {name}: {e}"))
                    })?;
                    let biases = lookup_f32_tensor_across(
                        &st,
                        safetensors_path,
                        safetensors_paths,
                        &biases_name,
                    )
                    .map_err(|e| {
                        diag::err_msg(
                            "weights",
                            &format!("MLX Q4 missing biases for {name}"),
                            &e,
                        );
                        Error::Weights(format!("MLX Q4 missing biases for {name}: {e}"))
                    })?;
                    let shape = t.shape();
                    if shape.len() != 2 {
                        return Err(Error::Weights(format!(
                            "MLX weight {name} shape {:?} not rank-2",
                            shape
                        )));
                    }
                    let rows = shape[0];
                    let packs = shape[1];
                    let cols = packs * 8;
                    let w_u32 = tensor_as_u32(&t)?;
                    let matrix =
                        quant_matrix_from_mlx_q4(rows, cols, group_size, &w_u32, &scales, &biases)?;
                    diag::log(
                        "weights",
                        format_args!(
                            "tensor {nk} dtype=U32 shape=[{rows},{cols}] hot={}",
                            diag::fmt_bytes(matrix.nbytes_hot() as u64)
                        ),
                    );
                    seen.insert(nk.clone());
                    matrices.push(NamedMatrix { name: nk, matrix });
                    shard_mats += 1;
                }
                safetensors::Dtype::BF16 | safetensors::Dtype::F16 | safetensors::Dtype::F32 => {
                    let cpu = tensor_view_to_cpu(&t)?;
                    let matrix = tensor_to_quant_matrix(&cpu, QuantScheme::Bf16)?;
                    diag::log(
                        "weights",
                        format_args!(
                            "tensor {nk} dtype={:?} shape={:?} hot={}",
                            t.dtype(),
                            t.shape(),
                            diag::fmt_bytes(matrix.nbytes_hot() as u64)
                        ),
                    );
                    seen.insert(nk.clone());
                    matrices.push(NamedMatrix { name: nk, matrix });
                    shard_mats += 1;
                }
                other => {
                    let e = Error::Weights(format!(
                        "unsupported dtype {other:?} for {name}"
                    ));
                    diag::err("weights", "tensor dtype", &e);
                    return Err(e);
                }
            }
        }
        diag::log(
            "weights",
            format_args!(
                "shard [{}/{}] done: kept {shard_mats} matrices in {:.1} ms",
                si + 1,
                safetensors_paths.len(),
                t_shard.elapsed().as_secs_f64() * 1e3
            ),
        );
    }

    let _ = dir;
    let ple = if let Some((w_u32, scales, biases, shape)) = ple_packed {
        if shape.len() == 2 {
            let rows = shape[0];
            let packs_per_row = shape[1];
            let cols = packs_per_row * 8; // 8 nibbles / u32
            let m = quant_matrix_from_mlx_q4(rows, cols, group_size, &w_u32, &scales, &biases)?;
            // Store as a single layer-0 bank; upload path treats Q4Mlx PLE specially.
            Some(PleBanks {
                layers: vec![crate::ple::PleLayerBank {
                    layer: 0,
                    vocab: rows,
                    dim: cols, // logical L*ple_dim for Q4 path
                    matrix: m,
                }],
            })
        } else {
            diag::log(
                "weights",
                format_args!("PLE packed unexpected shape {shape:?} — skipping"),
            );
            None
        }
    } else {
        None
    };
    let banks = HostWeightBanks {
        config: text,
        scheme,
        matrices,
        ple,
        kv_layout,
    };
    banks.validate_metal_limits().map_err(|e| {
        diag::err("weights", "Metal 4GiB validate", &e);
        e
    })?;
    diag::log(
        "weights",
        format_args!(
            "MLX load complete: matrices={} header_keys≈{keys_seen_total} hot={} Elapsed_load={:.1}s (PLE Q4 Hot={})",
            banks.matrices.len(),
            diag::fmt_bytes(banks.total_hot_bytes() as u64),
            t_all.elapsed().as_secs_f64(),
            banks.ple.is_some()
        ),
    );
    Ok(banks)
}

/// Resolve an MLX Q4 companion tensor (`scales` / `biases`), searching other
/// safetensor shards when the tensor lives across a shard boundary.
fn lookup_f32_tensor_across(
    primary: &SafeTensors<'_>,
    primary_path: &Path,
    all_paths: &[PathBuf],
    tensor_name: &str,
) -> Result<Vec<f32>> {
    if let Ok(t) = primary.tensor(tensor_name) {
        return tensor_as_f32_flat(&t);
    }
    diag::log(
        "weights",
        format_args!(
            "cross-shard lookup '{tensor_name}' (missing in {})",
            primary_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
        ),
    );
    for path in all_paths {
        if path == primary_path {
            continue;
        }
        let bytes = fs::read(path)
            .map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        let st = SafeTensors::deserialize(&bytes)
            .map_err(|e| Error::Safetensors(format!("{}: {e}", path.display())))?;
        if let Ok(t) = st.tensor(tensor_name) {
            diag::log(
                "weights",
                format_args!(
                    "cross-shard hit '{tensor_name}' in {}",
                    path.file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("?")
                ),
            );
            return tensor_as_f32_flat(&t);
        }
    }
    Err(Error::Weights(format!("tensor not found: {tensor_name}")))
}

fn tensor_as_u32(t: &safetensors::tensor::TensorView<'_>) -> Result<Vec<u32>> {
    let data = t.data();
    if data.len() % 4 != 0 {
        return Err(Error::Weights("U32 tensor byte len not multiple of 4".into()));
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn tensor_as_f32_flat(t: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    let n: usize = t.shape().iter().product();
    match t.dtype() {
        safetensors::Dtype::F32 => {
            let data = t.data();
            let mut out = Vec::with_capacity(n);
            for chunk in data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        safetensors::Dtype::BF16 => {
            let data = t.data();
            let mut out = Vec::with_capacity(n);
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(bf16_bits_to_f32(bits));
            }
            Ok(out)
        }
        safetensors::Dtype::F16 => {
            let data = t.data();
            let mut out = Vec::with_capacity(n);
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f16_bits_to_f32(bits));
            }
            Ok(out)
        }
        other => Err(Error::Weights(format!(
            "expected f32/bf16/f16 scales, got {other:?}"
        ))),
    }
}

fn tensor_view_to_cpu(t: &safetensors::tensor::TensorView<'_>) -> Result<TensorCpu> {
    let dtype = match t.dtype() {
        safetensors::Dtype::BF16 => CpuDType::BF16,
        safetensors::Dtype::F16 => CpuDType::F16,
        safetensors::Dtype::F32 => CpuDType::F32,
        other => {
            return Err(Error::Weights(format!(
                "unsupported dtype {other:?} for cpu view"
            )));
        }
    };
    Ok(TensorCpu {
        shape: t.shape().to_vec(),
        dtype,
        data: t.data().to_vec(),
    })
}

/// Build banks from an in-memory tensor map (unit tests / custom loaders).
pub fn build_banks_from_tensors(
    text: Gemma4TextConfig,
    mut tensors: HashMap<String, TensorCpu>,
    opts: LoadOptions,
) -> Result<HostWeightBanks> {
    let kv_layout = KvLayout::from_config(&text, opts.max_seq)?;

    // Normalize keys: strip common HF prefixes.
    let mut normalized = HashMap::new();
    for (k, v) in tensors.drain() {
        if opts.text_only && is_non_text_key(&k) {
            continue;
        }
        let nk = normalize_key(&k);
        normalized.insert(nk, v);
    }

    let mut matrices = Vec::new();
    let mut ple = None;

    // PLE packed table
    if text.has_ple() {
        let key_candidates = [
            "embed_tokens_per_layer.weight",
            "model.embed_tokens_per_layer.weight",
        ];
        let (key, tensor) = take_first(&mut normalized, &key_candidates).ok_or_else(|| {
            Error::Weights(
                "PLE enabled but embed_tokens_per_layer.weight not found in safetensors".into(),
            )
        })?;
        diag::log(
            "weights",
            format_args!(
                "PLE packed key={key} shape={:?} dtype={:?} nbytes={}",
                tensor.shape,
                tensor.dtype,
                diag::fmt_bytes(tensor.data.len() as u64)
            ),
        );
        let bits = tensor.as_bf16_bits()?;
        let banks = build_ple_banks_from_packed_bf16(&text, &bits, opts.ple_layout, opts.scheme)?;
        diag::log(
            "weights",
            format_args!(
                "PLE split → {} layer banks, total_hot={}",
                banks.num_layers(),
                diag::fmt_bytes(banks.total_hot_bytes() as u64)
            ),
        );
        let _ = key;
        ple = Some(banks);
    }

    // Remaining tensors → quantize
    for (name, tensor) in normalized {
        if should_skip_tensor(&name) {
            continue;
        }
        // Consumer layers have no k/v projections — skip if present erroneously.
        if let Some(layer) = parse_layer_index(&name) {
            if kv_layout.is_consumer(layer)
                && (name.contains("k_proj") || name.contains("v_proj"))
            {
                continue;
            }
            // 31B global k_eq_v: no v_proj on global layers
            if text.attention_k_eq_v
                && name.contains("v_proj")
                && text.layer_type(layer).map(|t| t.is_global()).unwrap_or(false)
            {
                continue;
            }
        }
        let matrix = tensor_to_quant_matrix(&tensor, opts.scheme)?;
        if matrix.nbytes_hot() > METAL_MAX_BUFFER_BYTES {
            return Err(Error::Weights(format!(
                "'{name}' exceeds Metal 4GiB after quant — needs sharding"
            )));
        }
        matrices.push(NamedMatrix { name, matrix });
    }

    let banks = HostWeightBanks {
        config: text,
        scheme: opts.scheme,
        matrices,
        ple,
        kv_layout,
    };
    banks.validate_metal_limits()?;
    Ok(banks)
}

/// CPU tensor view from safetensors.
#[derive(Clone, Debug)]
pub struct TensorCpu {
    pub shape: Vec<usize>,
    pub dtype: CpuDType,
    pub data: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuDType {
    F32,
    BF16,
    F16,
}

impl TensorCpu {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn as_bf16_bits(&self) -> Result<Vec<u16>> {
        match self.dtype {
            CpuDType::BF16 => {
                let n = self.numel();
                if self.data.len() < n * 2 {
                    return Err(Error::Weights("bf16 buffer short".into()));
                }
                let mut out = vec![0u16; n];
                for (i, slot) in out.iter_mut().enumerate() {
                    let b = &self.data[i * 2..i * 2 + 2];
                    *slot = u16::from_le_bytes([b[0], b[1]]);
                }
                Ok(out)
            }
            CpuDType::F32 => {
                let f = self.as_f32()?;
                Ok(f.iter().copied().map(crate::quant::f32_to_bf16_bits).collect())
            }
            CpuDType::F16 => {
                // Interpret IEEE f16 → f32 → bf16 bits (debug path).
                let n = self.numel();
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let b = u16::from_le_bytes([self.data[i * 2], self.data[i * 2 + 1]]);
                    out.push(crate::quant::f32_to_bf16_bits(f16_bits_to_f32(b)));
                }
                Ok(out)
            }
        }
    }

    pub fn as_f32(&self) -> Result<Vec<f32>> {
        match self.dtype {
            CpuDType::F32 => {
                let n = self.numel();
                let mut out = vec![0f32; n];
                for (i, slot) in out.iter_mut().enumerate() {
                    let b = &self.data[i * 4..i * 4 + 4];
                    *slot = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
                Ok(out)
            }
            CpuDType::BF16 => Ok(self
                .as_bf16_bits()?
                .into_iter()
                .map(bf16_bits_to_f32)
                .collect()),
            CpuDType::F16 => {
                let n = self.numel();
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let b = u16::from_le_bytes([self.data[i * 2], self.data[i * 2 + 1]]);
                    out.push(f16_bits_to_f32(b));
                }
                Ok(out)
            }
        }
    }
}

fn tensor_to_quant_matrix(t: &TensorCpu, scheme: QuantScheme) -> Result<QuantMatrix> {
    if t.shape.len() != 2 {
        // Norms / 1D: store as [n, 1] bf16/f32 quant
        if t.shape.len() == 1 {
            let n = t.shape[0];
            return match scheme {
                QuantScheme::Bf16 => {
                    let bits = t.as_bf16_bits()?;
                    quantize_affine_bf16_bits(n, 1, &bits, QuantScheme::Bf16)
                }
                other => {
                    let f = t.as_f32()?;
                    // group_size must divide cols=1 — use Bf16 for 1D or replicate
                    if matches!(
                        other,
                        QuantScheme::Q4 { .. } | QuantScheme::Q4Mlx { .. } | QuantScheme::Q8 { .. }
                    ) {
                        quantize_affine_f32(n, 1, &f, QuantScheme::Bf16)
                    } else {
                        quantize_affine_f32(n, 1, &f, other)
                    }
                }
            };
        }
        return Err(Error::Weights(format!(
            "expected rank 1/2 tensor, got shape {:?}",
            t.shape
        )));
    }
    let rows = t.shape[0];
    let cols = t.shape[1];
    match scheme {
        QuantScheme::Bf16 => {
            let bits = t.as_bf16_bits()?;
            quantize_affine_bf16_bits(rows, cols, &bits, QuantScheme::Bf16)
        }
        other => {
            // If cols not divisible by group size, fall back to bf16 for that tensor.
            if let Some(gs) = other.group_size() {
                if cols % gs != 0 {
                    let bits = t.as_bf16_bits()?;
                    return quantize_affine_bf16_bits(rows, cols, &bits, QuantScheme::Bf16);
                }
            }
            let f = t.as_f32()?;
            quantize_affine_f32(rows, cols, &f, other)
        }
    }
}

fn list_safetensor_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let entries = fs::read_dir(dir).map_err(|e| Error::Io(e.to_string()))?;
    for e in entries {
        let e = e.map_err(|e| Error::Io(e.to_string()))?;
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

fn load_safetensors_file(path: &Path, out: &mut HashMap<String, TensorCpu>) -> Result<()> {
    let t0 = Instant::now();
    let bytes = fs::read(path).map_err(|e| {
        let err = Error::Io(format!("{}: {e}", path.display()));
        diag::err("weights", "safetensors read", &err);
        err
    })?;
    diag::log(
        "weights",
        format_args!(
            "load_safetensors_file {} bytes={} in {:.1} ms",
            path.display(),
            diag::fmt_bytes(bytes.len() as u64),
            t0.elapsed().as_secs_f64() * 1e3
        ),
    );
    let st = SafeTensors::deserialize(&bytes)
        .map_err(|e| Error::Safetensors(format!("{}: {e}", path.display())))?;
    let before = out.len();
    for name in st.names() {
        let t = st
            .tensor(name)
            .map_err(|e| Error::Safetensors(e.to_string()))?;
        let dtype = match t.dtype() {
            safetensors::Dtype::BF16 => CpuDType::BF16,
            safetensors::Dtype::F16 => CpuDType::F16,
            safetensors::Dtype::F32 => CpuDType::F32,
            other => {
                return Err(Error::Weights(format!(
                    "unsupported dtype {other:?} for {name}"
                )));
            }
        };
        let shape: Vec<usize> = t.shape().to_vec();
        let data = t.data().to_vec();
        out.insert(
            name.to_string(),
            TensorCpu { shape, dtype, data },
        );
    }
    diag::log(
        "weights",
        format_args!(
            "deserialized {} tensors (map size {}→{})",
            st.names().len(),
            before,
            out.len()
        ),
    );
    Ok(())
}

fn normalize_key(k: &str) -> String {
    let mut s = k.to_string();
    for prefix in [
        "model.language_model.model.",
        "model.language_model.",
        "language_model.model.",
        "language_model.",
        "model.",
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest.to_string();
            break;
        }
    }
    s
}

fn is_non_text_key(k: &str) -> bool {
    let lk = k.to_lowercase();
    lk.contains("vision")
        || lk.contains("audio")
        || lk.contains("multi_modal")
        || lk.contains("image")
            && (lk.contains("tower") || lk.contains("projector") || lk.contains("encoder"))
}

fn should_skip_tensor(name: &str) -> bool {
    name.contains("inv_freq") // RoPE computed, not loaded
}

fn take_first(
    map: &mut HashMap<String, TensorCpu>,
    keys: &[&str],
) -> Option<(String, TensorCpu)> {
    for k in keys {
        if let Some(v) = map.remove(*k) {
            return Some(((*k).to_string(), v));
        }
    }
    None
}

fn parse_layer_index(name: &str) -> Option<usize> {
    // layers.{i}.
    let marker = "layers.";
    let idx = name.find(marker)?;
    let rest = &name[idx + marker.len()..];
    let end = rest.find('.')?;
    rest[..end].parse().ok()
}

/// Minimal IEEE-754 half → f32.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let (f_exp, f_frac) = if exp == 0 {
        if frac == 0 {
            (0u32, 0u32)
        } else {
            // subnormal
            let mut f = frac;
            let mut e = 127 - 15 + 1;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            (e, f << 13)
        }
    } else if exp == 31 {
        (255, frac << 13)
    } else {
        (exp + 127 - 15, frac << 13)
    };
    f32::from_bits((sign << 31) | (f_exp << 23) | f_frac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Gemma4TextConfig;

    #[test]
    fn normalize_strips_prefixes() {
        assert_eq!(
            normalize_key("model.language_model.layers.0.self_attn.q_proj.weight"),
            "layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn synthetic_load_with_ple_split() {
        let mut cfg = Gemma4TextConfig::e4b_preset();
        cfg.vocab_size_per_layer_input = 64;
        cfg.num_hidden_layers = 6;
        cfg.hidden_size_per_layer_input = 32;
        cfg.num_kv_shared_layers = 2;
        cfg.layer_types = cfg.layer_types[..6].to_vec();

        let vocab = cfg.vocab_size_per_layer_input;
        let l = cfg.num_hidden_layers;
        let d = cfg.hidden_size_per_layer_input;
        let packed_n = vocab * l * d;
        let mut packed = vec![0u8; packed_n * 2];
        for i in 0..packed_n {
            let b = (i as u16).to_le_bytes();
            packed[i * 2] = b[0];
            packed[i * 2 + 1] = b[1];
        }

        let mut tensors = HashMap::new();
        tensors.insert(
            "model.language_model.embed_tokens_per_layer.weight".into(),
            TensorCpu {
                shape: vec![vocab, l * d],
                dtype: CpuDType::BF16,
                data: packed,
            },
        );
        // Tiny stand-in embed [32, 32] (not full vocab×hidden — loader path smoke only).
        let rows = 32usize;
        let cols = 32usize;
        let data: Vec<u8> = (0..rows * cols * 2)
            .map(|i| (i % 251) as u8)
            .collect();
        tensors.insert(
            "embed_tokens.weight".into(),
            TensorCpu {
                shape: vec![rows, cols],
                dtype: CpuDType::BF16,
                data,
            },
        );

        let banks = build_banks_from_tensors(
            cfg,
            tensors,
            LoadOptions {
                scheme: QuantScheme::q4_default(),
                max_seq: 512,
                ple_layout: PlePackLayout::InterleavedPerToken,
                text_only: true,
            },
        )
        .unwrap();
        assert!(banks.ple.is_some());
        assert_eq!(banks.ple.as_ref().unwrap().num_layers(), 6);
        assert_eq!(banks.kv_layout.first_kv_shared, 4); // 6-2
        banks.validate_metal_limits().unwrap();
    }
}
