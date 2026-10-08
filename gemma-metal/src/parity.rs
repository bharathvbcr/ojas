//! Phase 3 parity harness — layer activation compare vs HF/MLX / host reference.
//!
//! Full-weight downloads are optional. Synthetic / shape tests exercise the
//! compare plumbing; unit tests cover KV-share, PLE scales, p-RoPE, softcap,
//! and gelu_tanh. JSON dumps load when available from HF/MLX sidecars.

use crate::config::Gemma4TextConfig;
use crate::error::{Error, Result};

/// Reference backend for activation dumps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefBackend {
    /// HuggingFace transformers (Python dump or safetensors).
    HuggingFace,
    /// MLX / mlx-lm.
    Mlx,
    /// Host-side synthetic reference (unit tests).
    Synthetic,
}

/// One named activation tensor for compare.
#[derive(Clone, Debug)]
pub struct ActivationDump {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl ActivationDump {
    pub fn numel(&self) -> usize {
        self.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
            .expect("activation shape overflow")
    }

    pub fn validate(&self) -> Result<()> {
        let count = self.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| Error::Config(format!("activation '{}': shape overflow", self.name)))?;
        if count == 0 || self.data.len() != count || self.data.iter().any(|x| !x.is_finite()) {
            return Err(Error::Config(format!(
                "activation '{}': expected {count} finite values, got {}; empty evidence is invalid",
                self.name, self.data.len()
            )));
        }
        Ok(())
    }
}

/// Compare stats between candidate and reference.
#[derive(Clone, Debug)]
pub struct CompareReport {
    pub name: String,
    pub max_abs: f32,
    pub mean_abs: f32,
    pub cosine: f32,
}

impl CompareReport {
    pub fn pass(&self, max_abs_tol: f32, cosine_tol: f32) -> bool {
        max_abs_tol.is_finite() && max_abs_tol >= 0.0
            && cosine_tol.is_finite() && (-1.0..=1.0).contains(&cosine_tol)
            && self.max_abs.is_finite() && self.mean_abs.is_finite() && self.cosine.is_finite()
            && self.max_abs <= max_abs_tol && self.cosine >= cosine_tol
    }
}

/// Elementwise compare (same shape required).
pub fn compare_activations(cand: &ActivationDump, reference: &ActivationDump) -> Result<CompareReport> {
    cand.validate()?;
    reference.validate()?;
    if cand.shape != reference.shape {
        return Err(Error::Config(format!(
            "shape mismatch for '{}': {:?} vs {:?}",
            cand.name, cand.shape, reference.shape
        )));
    }
    let n = cand.data.len();
    let mut max_abs = 0f64;
    let mut sum_abs = 0f64;
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (&a, &b) in cand.data.iter().zip(reference.data.iter()) {
        let (a, b) = (f64::from(a), f64::from(b));
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    let cosine = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else if max_abs == 0.0 {
        // Identical zero vectors (or empty) — treat as perfect match.
        1.0
    } else {
        0.0
    };
    Ok(CompareReport {
        name: cand.name.clone(),
        max_abs: max_abs as f32,
        mean_abs: (sum_abs / n as f64) as f32,
        cosine: cosine.clamp(-1.0, 1.0) as f32,
    })
}

/// Layer-by-layer hooks for E4B parity.
pub fn e4b_layer_hook_names(cfg: &Gemma4TextConfig) -> Vec<String> {
    let mut names = vec![
        "embed".into(),
        "final_norm".into(),
        "logits".into(),
    ];
    for layer in 0..cfg.num_hidden_layers {
        names.push(format!("layer{layer}.attn_out"));
        names.push(format!("layer{layer}.mlp_out"));
        if cfg.has_ple() {
            names.push(format!("layer{layer}.ple"));
        }
    }
    names
}

/// Synthetic identity dump for shape plumbing tests.
pub fn synthetic_dump(name: &str, shape: &[usize], fill: f32) -> ActivationDump {
    let n: usize = shape.iter().product();
    ActivationDump {
        name: name.into(),
        shape: shape.to_vec(),
        data: vec![fill; n],
    }
}

/// Load a reference dump from JSON:
/// `{ "name": "...", "shape": [...], "data": [...] }`.
pub fn load_dump_json(s: &str) -> Result<ActivationDump> {
    #[derive(serde::Deserialize)]
    struct Raw {
        name: String,
        shape: Vec<usize>,
        data: Vec<f32>,
    }
    let raw: Raw = serde_json::from_str(s)?;
    let dump = ActivationDump {
        name: raw.name,
        shape: raw.shape,
        data: raw.data,
    };
    dump.validate()?;
    Ok(dump)
}

/// Load a directory of `*.json` activation dumps (HF/MLX export sidecars).
pub fn load_dump_dir(dir: impl AsRef<std::path::Path>) -> Result<Vec<ActivationDump>> {
    let dir = dir.as_ref();
    let mut out = Vec::new();
    let rd = std::fs::read_dir(dir).map_err(|e| Error::Io(format!("dump dir: {e}")))?;
    let mut paths: Vec<_> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    paths.sort();
    for p in paths {
        let s = std::fs::read_to_string(&p)
            .map_err(|e| Error::Io(format!("read {}: {e}", p.display())))?;
        out.push(load_dump_json(&s)?);
    }
    Ok(out)
}

/// PLE scale factors used by the forward path.
pub fn ple_scales(ple_dim: usize) -> (f32, f32) {
    let lookup = (ple_dim as f32).sqrt();
    let combine = std::f32::consts::FRAC_1_SQRT_2;
    (lookup, combine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::{
        apply_rope, gelu_pytorch_tanh, greedy_decode_host, host_forward_prefill,
        proportional_rope_dim, softcap_f32, SyntheticE4bGraph,
    };
    use crate::kv::{KvLayout, KvRole, SharedKvId};

    #[test]
    fn compare_identical() {
        let a = synthetic_dump("x", &[2, 4], 1.5);
        let b = synthetic_dump("x", &[2, 4], 1.5);
        let r = compare_activations(&a, &b).unwrap();
        assert!(r.pass(1e-6, 0.999));
        assert_eq!(r.max_abs, 0.0);
    }

    #[test]
    fn compare_detects_diff() {
        let a = synthetic_dump("x", &[4], 1.0);
        let mut b = synthetic_dump("x", &[4], 1.0);
        b.data[2] = 2.0;
        let r = compare_activations(&a, &b).unwrap();
        assert!(r.max_abs > 0.9);
        assert!(!r.pass(1e-3, 0.9999));
    }

    #[test]
    fn e4b_hooks_include_ple() {
        let cfg = Gemma4TextConfig::e4b_preset();
        let names = e4b_layer_hook_names(&cfg);
        assert!(names.iter().any(|n| n.contains("ple")));
        assert!(names.len() > 42);
    }

    #[test]
    fn load_json_roundtrip() {
        let s = r#"{"name":"logits","shape":[2],"data":[0.1,0.2]}"#;
        let d = load_dump_json(s).unwrap();
        assert_eq!(d.name, "logits");
        assert_eq!(d.data, vec![0.1, 0.2]);
    }

    #[test]
    fn kv_share_aliasing_unit() {
        let cfg = Gemma4TextConfig::e4b_preset();
        let layout = KvLayout::from_config(&cfg, 128).unwrap();
        assert_eq!(layout.first_kv_shared, 24);
        let c = layout.layer(24).unwrap();
        assert!(matches!(
            c.role,
            KvRole::Consumer {
                shared: SharedKvId::SlidingFull
            }
        ));
        // Consumers never get producer slots.
        for i in layout.first_kv_shared..cfg.num_hidden_layers {
            assert!(layout.is_consumer(i));
            assert!(matches!(layout.layer(i).unwrap().role, KvRole::Consumer { .. }));
        }
    }

    #[test]
    fn ple_scales_unit() {
        let (lookup, combine) = ple_scales(256);
        assert!((lookup - 16.0).abs() < 1e-5);
        assert!((combine - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn p_rope_unit() {
        let d = 512usize;
        let rotary = proportional_rope_dim(d, 0.25);
        assert_eq!(rotary, 128);
        let mut x = vec![0f32; d];
        x[0] = 1.0;
        x[1] = 0.0;
        apply_rope(&mut x, d, rotary, 3, 1_000_000.0);
        // Beyond rotary_dim unchanged (zero-pad / pass-through).
        assert_eq!(x[128], 0.0);
        assert!(x[0] != 1.0 || x[1] != 0.0); // rotated
    }

    #[test]
    fn softcap_unit() {
        let y = softcap_f32(30.0, 30.0);
        assert!((y - 30.0 * 1.0f32.tanh()).abs() < 1e-5);
    }

    #[test]
    fn gelu_tanh_unit() {
        assert_eq!(gelu_pytorch_tanh(0.0), 0.0);
        let y = gelu_pytorch_tanh(1.0);
        // PyTorch gelu-tanh approx at 1.0 ≈ 0.841
        assert!((y - 0.8413).abs() < 1e-3);
    }

    #[test]
    fn synthetic_layer_compare_host() {
        let model = SyntheticE4bGraph::mini_parity().unwrap();
        let tokens = [2u32, 5, 9, 1];
        let (_logits, _tok, dumps) = host_forward_prefill(&model, &tokens).unwrap();
        for d in &dumps.dumps {
            let r = compare_activations(d, d).unwrap();
            assert!(r.pass(1e-6, 0.999), "{} max_abs={} cosine={}", r.name, r.max_abs, r.cosine);
        }
        assert!(dumps.get("layer0.attn_out").is_some());
        assert!(dumps.get("layer2.attn_out").is_some()); // consumer layer
        assert!(dumps.get("logits").is_some());
    }

    #[test]
    fn greedy_decode_smoke_synthetic() {
        let model = SyntheticE4bGraph::mini_parity().unwrap();
        let out = greedy_decode_host(&model, &[4u32, 5], 3).unwrap();
        assert!(out.len() >= 2);
        assert!(out.len() <= 5);
    }

    #[test]
    fn load_dump_dir_optional() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("logits.json");
        std::fs::write(&p, r#"{"name":"logits","shape":[2],"data":[1.0,2.0]}"#).unwrap();
        let dumps = load_dump_dir(dir.path()).unwrap();
        assert_eq!(dumps.len(), 1);
        assert_eq!(dumps[0].name, "logits");
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;

    #[test]
    fn invalid_evidence_cannot_pass() {
        for data in [vec![f32::NAN], vec![f32::INFINITY], vec![]] {
            let dump = ActivationDump { name: "invalid".into(), shape: vec![data.len()], data };
            let result = compare_activations(&dump, &dump);
            assert!(result.is_err(), "non-finite or empty evidence accepted");
        }
        let dump = ActivationDump { name: "overflow".into(),
            shape: vec![usize::MAX, 2], data: vec![] };
        assert!(dump.validate().is_err());
        let report = CompareReport { name: "bad".into(), max_abs: 0.0, mean_abs: 0.0, cosine: 1.0 };
        assert!(!report.pass(f32::INFINITY, 0.9));
    }

    #[test]
    fn large_finite_identical_evidence_passes() {
        let dump = ActivationDump { name: "finite".into(),
            shape: vec![2], data: vec![f32::MAX, -f32::MAX] };
        assert!(compare_activations(&dump,&dump).unwrap().pass(0.0,0.999));
    }
}
