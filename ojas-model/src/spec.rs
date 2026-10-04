//! Model shape, and its `ojas.spec` JSON (`ojas-spec-v1`).
//!
//! Field names and the JSON schema are the ones `ojas-oracle/README.md`
//! defines ("The `ojas.spec` JSON"), which follow `ojas-infer`'s
//! `GptConfig`. The JSON is one flat object with exactly these keys:
//! `format` (`"ojas-spec-v1"`), `arch` (`"nanolab-gpt"`), `vocab`,
//! `n_embd`, `n_layer`, `n_head`, `n_kv_head`, `head_dim`, `hidden`,
//! `max_seq` (integers), `rope_base`, `rms_eps` (numbers),
//! `tie_embeddings`, `qk_norm`, `gated_attention`, `value_residual`
//! (booleans). A missing key, an unknown key, a duplicate key, a value of
//! the wrong type, or trailing text is refused; nothing is defaulted.

use ojas_core::{OjasError, RMS_NORM_EPS};
use ojas_io::IoError;

use crate::json::flat_object;

/// The safetensors `__metadata__` key that holds the spec JSON.
pub const SPEC_METADATA_KEY: &str = "ojas.spec";
/// The `format` value this crate reads and writes.
pub const SPEC_FORMAT: &str = "ojas-spec-v1";
/// The `arch` value this crate reads and writes.
pub const SPEC_ARCH: &str = "nanolab-gpt";
/// Longest spec JSON accepted.
const MAX_SPEC_BYTES: usize = 64 << 10;

/// Most transformer blocks a spec may declare. `param_table` builds about
/// 14 named rows per block before any budget is consulted, so a spec read
/// from a file or the wire must not set the size of that table freely. 4096
/// is far above any trained transformer (GPT-3 has 96 blocks).
pub const MAX_LAYERS: usize = 4096;

/// Shape of a nanolab GPT (nanolab `Config` names in the comments).
///
/// `head_dim` is its own field, as in nanolab: `n_head * head_dim` need not
/// equal `n_embd`. `n_kv_head == n_head` is plain multi-head attention.
/// QK-norm, the per-head gate and the value residual are always on: the
/// JSON carries them as booleans and a reader refuses `false`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelSpec {
    /// `vocab_size`.
    pub vocab: usize,
    /// `d_model`.
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// SwiGLU width, nanolab `_swiglu_hidden(d_model)`.
    pub hidden: usize,
    /// `block_size`: the longest sequence the model is run on.
    pub max_seq: usize,
    pub rope_base: f64,
    /// RMSNorm epsilon of every norm (`mixers.py:50`, 1e-6). The kernels
    /// take it as `f32` ([`Self::eps`]).
    pub rms_eps: f64,
    /// `tie_embeddings`: the head is `tok_emb.weight`.
    pub tie_embeddings: bool,
}

impl ModelSpec {
    /// nanolab's defaults: 12 layers, d 768, 12 x 64 heads, SwiGLU 2048,
    /// vocab 50304, block 1024, RoPE base 10000, eps 1e-6, tied embedding.
    pub fn nanolab_124m() -> Self {
        Self {
            vocab: 50304,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            n_kv_head: 12,
            head_dim: 64,
            hidden: swiglu_hidden(768),
            max_seq: 1024,
            rope_base: 10000.0,
            rms_eps: 1e-6,
            tie_embeddings: true,
        }
    }

    /// The CI spec of `docs/framework-design.md` §10 and of the
    /// `ojas-oracle` tiny fixtures: 2 layers, d 64, 4 x 16 heads, SwiGLU
    /// 192, vocab 256, block 32.
    pub fn tiny() -> Self {
        Self {
            vocab: 256,
            n_embd: 64,
            n_layer: 2,
            n_head: 4,
            n_kv_head: 4,
            head_dim: 16,
            hidden: swiglu_hidden(64),
            max_seq: 32,
            rope_base: 10000.0,
            rms_eps: 1e-6,
            tie_embeddings: true,
        }
    }

    /// The RMSNorm epsilon as the kernels take it. For `rms_eps = 1e-6`
    /// this is `ojas_core::RMS_NORM_EPS`.
    pub fn eps(&self) -> f32 {
        self.rms_eps as f32
    }

    /// `n_head * head_dim`, the width of `q_proj`'s output. Call on a spec
    /// that passed [`Self::validate`], which checks the product fits.
    pub fn q_width(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// `n_kv_head * head_dim`, the width of `k_proj` and `v_proj`.
    pub fn kv_width(&self) -> usize {
        self.n_kv_head * self.head_dim
    }

    /// Refuse a spec no executor here can run.
    ///
    /// Every size must be non-zero, `n_layer` at most [`MAX_LAYERS`],
    /// `head_dim` even (half-split RoPE),
    /// `n_head` a multiple of `n_kv_head`, products must fit `usize`, vocab
    /// and `head_dim` must fit `u32`, `rope_base` and `rms_eps` must be
    /// positive and finite (`rms_eps` also as `f32`). An untied head is
    /// [`OjasError::Unsupported`]: the names table has no `lm_head.weight`.
    pub fn validate(&self) -> Result<(), OjasError> {
        const OP: &str = "ModelSpec::validate";
        let sizes = [
            ("vocab", self.vocab),
            ("n_embd", self.n_embd),
            ("n_layer", self.n_layer),
            ("n_head", self.n_head),
            ("n_kv_head", self.n_kv_head),
            ("head_dim", self.head_dim),
            ("hidden", self.hidden),
            ("max_seq", self.max_seq),
        ];
        for (name, value) in sizes {
            if value == 0 {
                return Err(OjasError::OutOfRange {
                    op: OP,
                    detail: format!("{name} is 0"),
                });
            }
        }
        if self.n_layer > MAX_LAYERS {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("n_layer {} exceeds {MAX_LAYERS}", self.n_layer),
            });
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!(
                    "half-split RoPE needs an even head_dim, got {}",
                    self.head_dim
                ),
            });
        }
        if !self.n_head.is_multiple_of(self.n_kv_head) {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!(
                    "n_head {} is not a multiple of n_kv_head {}",
                    self.n_head, self.n_kv_head
                ),
            });
        }
        if u32::try_from(self.vocab).is_err() || u32::try_from(self.head_dim).is_err() {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: "vocab and head_dim must fit u32".to_string(),
            });
        }
        let overflow = || OjasError::OutOfRange {
            op: OP,
            detail: "a parameter size overflows usize".to_string(),
        };
        let q_width = self
            .n_head
            .checked_mul(self.head_dim)
            .ok_or_else(overflow)?;
        for rows in [q_width, self.vocab, self.hidden] {
            rows.checked_mul(self.n_embd)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(overflow)?;
        }
        if !(self.rope_base.is_finite() && self.rope_base > 0.0) {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("rope_base {} is not positive and finite", self.rope_base),
            });
        }
        let eps = self.eps();
        if !(self.rms_eps.is_finite() && self.rms_eps > 0.0 && eps.is_finite() && eps > 0.0) {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("rms_eps {} is not positive and finite in f32", self.rms_eps),
            });
        }
        if !self.tie_embeddings {
            return Err(OjasError::Unsupported {
                op: OP,
                detail: "an untied lm_head is not supported; nanolab ties it".to_string(),
            });
        }
        Ok(())
    }

    /// [`Self::validate`], then refuse what v1 training cannot run: the
    /// check of the tape and the trainer. [`crate::Eval`] runs any spec
    /// [`Self::validate`] accepts, grouped-query attention included.
    ///
    /// Grouped-query attention is refused with [`OjasError::Unsupported`]:
    /// causal SDPA (the training attention, with a backward) takes `q`, `k`
    /// and `v` of one shape and the trait has no head-repeat op, so v1
    /// training runs multi-head attention only (nanolab's default).
    pub fn validate_for_training(&self) -> Result<(), OjasError> {
        self.validate()?;
        if self.n_kv_head != self.n_head {
            return Err(OjasError::Unsupported {
                op: "ModelSpec::validate_for_training",
                detail: format!(
                    "grouped-query attention (n_kv_head {} != n_head {}) is not supported \
                     by the training block",
                    self.n_kv_head, self.n_head
                ),
            });
        }
        Ok(())
    }

    /// The `ojas-spec-v1` JSON of a valid spec.
    pub fn to_json(&self) -> Result<String, OjasError> {
        self.validate()?;
        Ok(format!(
            "{{\"format\":\"{SPEC_FORMAT}\",\"arch\":\"{SPEC_ARCH}\",\"vocab\":{},\"n_embd\":{},\
             \"n_layer\":{},\"n_head\":{},\"n_kv_head\":{},\"head_dim\":{},\"hidden\":{},\
             \"max_seq\":{},\"rope_base\":{:?},\"rms_eps\":{:?},\"tie_embeddings\":true,\
             \"qk_norm\":true,\"gated_attention\":true,\"value_residual\":true}}",
            self.vocab,
            self.n_embd,
            self.n_layer,
            self.n_head,
            self.n_kv_head,
            self.head_dim,
            self.hidden,
            self.max_seq,
            self.rope_base,
            self.rms_eps,
        ))
    }

    /// Parse and validate `ojas-spec-v1` JSON.
    ///
    /// Integer keys take a JSON integer (digits only, no sign, fraction or
    /// exponent); `rope_base` and `rms_eps` take any JSON number whose value
    /// is exactly an `f64` (an integer past 2^53 that would round is
    /// refused). The JSON grammar is `ojas_io`'s strict reader. The
    /// booleans must be `true`: `tie_embeddings` false is refused by
    /// [`Self::validate`], and a model without QK-norm, the gate or the
    /// value residual is not one this crate implements.
    pub fn from_json(text: &str) -> Result<Self, OjasError> {
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
        let io = |e: IoError| spec_error(e.detail().to_string());
        let root = flat_object(text, MAX_SPEC_BYTES, &KEYS).map_err(io)?;
        let text_of = |key: &str, want: &str| match root.get_str(key).map_err(io)? {
            s if s == want => Ok(()),
            s => Err(spec_error(format!("{key} is {s:?}, expected {want:?}"))),
        };
        let count = |key: &str| {
            let n = root.get_u64(key).map_err(io)?;
            usize::try_from(n).map_err(|_| spec_error(format!("{key} {n} does not fit usize")))
        };
        let real = |key: &str| root.get_f64(key).map_err(io);
        let flag = |key: &str| root.get_bool(key).map_err(io);
        text_of("format", SPEC_FORMAT)?;
        text_of("arch", SPEC_ARCH)?;
        for key in ["qk_norm", "gated_attention", "value_residual"] {
            if !flag(key)? {
                return Err(OjasError::Unsupported {
                    op: "ModelSpec::from_json",
                    detail: format!("{key} false is not a model this crate implements"),
                });
            }
        }
        let spec = Self {
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
        };
        spec.validate()?;
        Ok(spec)
    }
}

/// nanolab `_swiglu_hidden`: `int(2/3 * 4 * d)` rounded up to a multiple of 64.
pub fn swiglu_hidden(n_embd: usize) -> usize {
    let h = (2.0f64 / 3.0 * 4.0 * n_embd as f64) as usize;
    h.div_ceil(64) * 64
}

/// `ojas_core::RMS_NORM_EPS` is the nanolab default this module's presets use.
const _: () = assert!(RMS_NORM_EPS == 1e-6f32);

fn spec_error(detail: String) -> OjasError {
    OjasError::Shape {
        op: "ModelSpec::from_json",
        detail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_match_nanolab_and_the_ci_spec() {
        let big = ModelSpec::nanolab_124m();
        assert_eq!(big.hidden, 2048);
        assert_eq!(big.q_width(), 768);
        assert_eq!(big.eps(), RMS_NORM_EPS);
        assert!(big.validate_for_training().is_ok());
        let tiny = ModelSpec::tiny();
        assert_eq!(tiny.hidden, 192);
        assert_eq!((tiny.n_head, tiny.head_dim, tiny.vocab), (4, 16, 256));
        assert!(tiny.validate_for_training().is_ok());
    }

    #[test]
    fn validation_refuses_each_bad_field() {
        let base = ModelSpec::tiny();
        let cases: Vec<(ModelSpec, &str)> = vec![
            (ModelSpec { n_layer: 0, ..base }, "n_layer"),
            (ModelSpec { vocab: 0, ..base }, "vocab"),
            (ModelSpec { max_seq: 0, ..base }, "max_seq"),
            (
                ModelSpec {
                    head_dim: 15,
                    ..base
                },
                "odd head_dim",
            ),
            (
                ModelSpec {
                    n_kv_head: 3,
                    ..base
                },
                "kv not dividing heads",
            ),
            (
                ModelSpec {
                    rope_base: f64::NAN,
                    ..base
                },
                "rope_base",
            ),
            (
                ModelSpec {
                    rms_eps: 0.0,
                    ..base
                },
                "eps",
            ),
            (
                ModelSpec {
                    rms_eps: 1e-60,
                    ..base
                },
                "eps underflows f32",
            ),
            (
                ModelSpec {
                    rms_eps: f64::INFINITY,
                    ..base
                },
                "eps inf",
            ),
            (
                ModelSpec {
                    vocab: usize::MAX,
                    ..base
                },
                "vocab overflow",
            ),
        ];
        for (spec, what) in cases {
            assert!(spec.validate().is_err(), "{what} accepted");
        }
        let untied = ModelSpec {
            tie_embeddings: false,
            ..base
        };
        assert!(matches!(
            untied.validate(),
            Err(OjasError::Unsupported { .. })
        ));
    }

    #[test]
    fn training_refuses_grouped_query_attention_and_eval_accepts_it() {
        let gqa = ModelSpec {
            n_kv_head: 2,
            ..ModelSpec::tiny()
        };
        assert!(gqa.validate().is_ok());
        assert!(matches!(
            gqa.validate_for_training(),
            Err(OjasError::Unsupported { .. })
        ));
    }

    /// What `ojas-oracle/python/export_init.py` writes (Python `json.dumps`).
    const ORACLE_STYLE: &str = r#"{"format": "ojas-spec-v1", "arch": "nanolab-gpt", "vocab": 256, "n_embd": 64, "n_layer": 2, "n_head": 4, "n_kv_head": 4, "head_dim": 16, "hidden": 192, "max_seq": 32, "rope_base": 10000.0, "rms_eps": 1e-06, "tie_embeddings": true, "qk_norm": true, "gated_attention": true, "value_residual": true}"#;

    #[test]
    fn json_round_trips_and_reads_the_oracle_style() {
        for spec in [ModelSpec::tiny(), ModelSpec::nanolab_124m()] {
            let text = spec.to_json().unwrap();
            assert_eq!(ModelSpec::from_json(&text).unwrap(), spec);
        }
        assert_eq!(
            ModelSpec::from_json(ORACLE_STYLE).unwrap(),
            ModelSpec::tiny()
        );
    }

    #[test]
    fn json_refuses_anything_off_schema() {
        let edits: &[(&str, &str, &str)] = &[
            (
                "unknown key",
                "\"vocab\": 256",
                "\"vocab\": 256, \"dropout\": 0.0",
            ),
            ("missing key", "\"max_seq\": 32, ", ""),
            (
                "duplicate key",
                "\"vocab\": 256",
                "\"vocab\": 256, \"vocab\": 256",
            ),
            ("format", "ojas-spec-v1", "ojas-spec-v2"),
            ("arch", "nanolab-gpt", "llama"),
            ("float count", "\"vocab\": 256", "\"vocab\": 256.0"),
            ("negative count", "\"n_layer\": 2", "\"n_layer\": -2"),
            ("string count", "\"n_layer\": 2", "\"n_layer\": \"2\""),
            ("null", "\"hidden\": 192", "\"hidden\": null"),
            ("nested", "\"hidden\": 192", "\"hidden\": [192]"),
            ("bool as int", "\"qk_norm\": true", "\"qk_norm\": 1"),
            ("no qk norm", "\"qk_norm\": true", "\"qk_norm\": false"),
            (
                "no gate",
                "\"gated_attention\": true",
                "\"gated_attention\": false",
            ),
            (
                "no value residual",
                "\"value_residual\": true",
                "\"value_residual\": false",
            ),
            (
                "untied",
                "\"tie_embeddings\": true",
                "\"tie_embeddings\": false",
            ),
            ("zero eps", "1e-06", "0"),
            ("malformed number", "1e-06", "1e-"),
            ("leading zero", "\"vocab\": 256", "\"vocab\": 0256"),
            ("odd head_dim", "\"head_dim\": 16", "\"head_dim\": 15"),
            ("trailing text", "true}", "true} x"),
        ];
        for (what, from, to) in edits {
            let text = ORACLE_STYLE.replacen(from, to, 1);
            assert_ne!(text, ORACLE_STYLE, "{what}: edit did not apply");
            assert!(ModelSpec::from_json(&text).is_err(), "{what} accepted");
        }
        assert!(ModelSpec::from_json(&" ".repeat(MAX_SPEC_BYTES + 1)).is_err());
    }
}
