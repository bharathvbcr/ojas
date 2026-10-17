//! The GQA and sliding-window goldens (`fixtures/variants/`): integrity of
//! the files, the v2 spec that carries a window, and the gates that hold a
//! model to them (`variant_forward_parity`, `variant_grads_parity`, and the
//! per-backend `Gates`). No model runs here; `ojas-model/tests/oracle_parity.rs`
//! runs the CPU, Metal and wgpu trainers against these files.

use ojas_core::OjasError;
use ojas_oracle::golden::{tiny_forward, TensorSet, TokenBatch, TrainSetup};
use ojas_oracle::parity::{variant_forward_parity, variant_grads_parity, Gates, ParityModel};
use ojas_oracle::spec::{parse_spec, tiny_spec, ModelSpec};
use ojas_oracle::variants::{load_variant, variant, Variant, SWA_WINDOW};

const GQA_BYTES: &[u8] = include_bytes!("../fixtures/variants/gqa.safetensors");
const SWA_BYTES: &[u8] = include_bytes!("../fixtures/variants/swa.safetensors");

#[test]
fn the_fixtures_are_the_tiny_model_with_one_change_each() {
    let gqa = variant(Variant::Gqa).unwrap();
    assert_eq!(
        gqa.spec,
        ModelSpec {
            n_kv_head: 2,
            ..tiny_spec()
        }
    );
    assert!(gqa.spec.n_kv_head < gqa.spec.n_head);
    let swa = variant(Variant::Swa).unwrap();
    assert_eq!(
        swa.spec,
        ModelSpec {
            window: Some(SWA_WINDOW),
            ..tiny_spec()
        }
    );
    // The window bites inside the sequence: a window of 8 over 32 positions.
    assert!(SWA_WINDOW < swa.spec.max_seq);
    for fx in [&gqa, &swa] {
        assert_eq!(
            fx.provenance.generator,
            "ojas-oracle/python/variant_fixture.py"
        );
        assert_eq!(fx.provenance.torch, ojas_oracle::golden::TORCH_VERSION);
        assert_eq!(fx.provenance.kind().unwrap(), "variant");
        assert_eq!((fx.batch.batch, fx.batch.seq_len), (2, 32));
        assert_eq!(fx.logits.len(), 2 * 32 * fx.spec.vocab);
        assert!(fx.loss.is_finite() && fx.logits.iter().all(|v| v.is_finite()));
        // Layer 0's vr_lambda has no gradient in torch; every other
        // gradient is non-zero (the loader refuses an all-zero one), unlike
        // the init fixtures where o_proj and ffn.down start at zero.
        assert_eq!(fx.no_grad, ["blocks.0.mixer.vr_lambda"]);
        assert_eq!(fx.grads.len(), fx.params.len() - 1);
        assert_eq!(fx.loss_seed, 1.0);
    }
    // Same tokens in both: the variants differ in the model, not the batch.
    assert_eq!(gqa.batch, swa.batch);
    // The two files are different computations.
    assert_ne!(gqa.logits, swa.logits);
}

#[test]
fn a_fixture_is_refused_as_the_other_variant_and_when_damaged() {
    assert!(load_variant(GQA_BYTES, Variant::Gqa).is_ok());
    assert!(load_variant(SWA_BYTES, Variant::Swa).is_ok());
    let e = load_variant(GQA_BYTES, Variant::Swa).unwrap_err();
    assert!(e.to_string().contains("not the swa variant"), "{e}");
    let e = load_variant(SWA_BYTES, Variant::Gqa).unwrap_err();
    assert!(e.to_string().contains("not the gqa variant"), "{e}");
    // Truncated anywhere: inside the header, and inside the data.
    for cut in [4, 200, GQA_BYTES.len() / 2, GQA_BYTES.len() - 1] {
        assert!(
            load_variant(&GQA_BYTES[..cut], Variant::Gqa).is_err(),
            "cut {cut}"
        );
    }
    // The tiny init is a valid safetensors file with a spec, but not a variant.
    let init = include_bytes!("../fixtures/tiny/init.safetensors");
    assert!(load_variant(init, Variant::Gqa).is_err());
}

const V1: &str = r#"{"format":"ojas-spec-v1","arch":"nanolab-gpt","vocab":256,"n_embd":64,
    "n_layer":2,"n_head":4,"n_kv_head":4,"head_dim":16,"hidden":192,"max_seq":32,
    "rope_base":10000.0,"rms_eps":1e-06,"tie_embeddings":true,"qk_norm":true,
    "gated_attention":true,"value_residual":true}"#;

#[test]
fn spec_v2_carries_a_window_and_v1_does_not() {
    let v2 = V1.replacen("ojas-spec-v1", "ojas-spec-v2", 1).replacen(
        "\"value_residual\":true",
        "\"value_residual\":true,\"window\":8",
        1,
    );
    assert_eq!(
        parse_spec(&v2).unwrap(),
        ModelSpec {
            window: Some(8),
            ..tiny_spec()
        }
    );
    // v1 is unchanged: the tiny spec, no window.
    assert_eq!(parse_spec(V1).unwrap(), tiny_spec());
    // v2 requires its window; v1 refuses one; a window is at least 1 and an
    // exact integer; unknown keys are still refused.
    let no_window = V1.replacen("ojas-spec-v1", "ojas-spec-v2", 1);
    assert!(parse_spec(&no_window).is_err());
    let v1_with_window = V1.replacen(
        "\"value_residual\":true",
        "\"value_residual\":true,\"window\":8",
        1,
    );
    let e = parse_spec(&v1_with_window).unwrap_err();
    assert!(e.to_string().contains("unknown key"), "{e}");
    for bad in ["0", "8.0", "-1", "\"8\"", "null"] {
        let text = v2.replacen("\"window\":8", &format!("\"window\":{bad}"), 1);
        assert!(parse_spec(&text).is_err(), "accepted window {bad}");
    }
    let unknown = v2.replacen("\"window\":8", "\"window\":8,\"sinks\":4", 1);
    assert!(parse_spec(&unknown).is_err());
}

/// Returns torch's numbers for the variant it was configured for, edited by
/// `loss_scale`, `logit_bump` and `grad_scale`, so each gate can be shown to
/// accept and to refuse.
struct Replay {
    fx: Option<ojas_oracle::variants::VariantFixture>,
    loss_scale: f64,
    logit_bump: f32,
    grad_scale: f32,
    configured: Vec<ModelSpec>,
    /// Without `configure` support the trait's default must refuse variants.
    supports_variants: bool,
}

impl Replay {
    fn new(supports_variants: bool) -> Self {
        Self {
            fx: None,
            loss_scale: 1.0,
            logit_bump: 0.0,
            grad_scale: 1.0,
            configured: Vec::new(),
            supports_variants,
        }
    }
}

impl ParityModel for Replay {
    fn configure(&mut self, spec: &ModelSpec) -> Result<(), OjasError> {
        self.configured.push(spec.clone());
        if !self.supports_variants && *spec != tiny_spec() {
            return Err(OjasError::Unsupported {
                op: "Replay::configure",
                detail: "tiny only".to_string(),
            });
        }
        self.fx = Variant::ALL
            .into_iter()
            .map(|v| variant(v).unwrap())
            .find(|fx| fx.spec == *spec);
        Ok(())
    }

    fn forward(
        &mut self,
        params: &TensorSet,
        batch: &TokenBatch,
    ) -> Result<(f32, Vec<f32>), OjasError> {
        if let Some(fx) = &self.fx {
            assert_eq!(
                params, &fx.params,
                "the runner passes the fixture's parameters"
            );
            assert_eq!(batch, &fx.batch);
            let mut logits = fx.logits.clone();
            logits[0] += self.logit_bump;
            return Ok(((f64::from(fx.loss) * self.loss_scale) as f32, logits));
        }
        let fx = tiny_forward()?;
        let mut logits = fx.logits;
        logits[0] += self.logit_bump;
        Ok(((f64::from(fx.loss) * self.loss_scale) as f32, logits))
    }

    fn grads(
        &mut self,
        params: &TensorSet,
        batch: &TokenBatch,
        seed: f32,
    ) -> Result<TensorSet, OjasError> {
        let fx = self.fx.as_ref().expect("a variant was configured");
        assert_eq!((params, batch, seed), (&fx.params, &fx.batch, fx.loss_seed));
        let tensors = fx
            .grads
            .iter()
            .map(|g| {
                let mut g = g.clone();
                if g.name == "blocks.1.mixer.k_proj.weight" {
                    g.data.iter_mut().for_each(|v| *v *= self.grad_scale);
                }
                g
            })
            .collect();
        TensorSet::new(tensors)
    }

    fn train(
        &mut self,
        _: &TensorSet,
        _: &TrainSetup,
        _: usize,
    ) -> Result<(Vec<f64>, TensorSet), OjasError> {
        unreachable!("the variant gates do not train")
    }
}

#[test]
fn a_model_that_reproduces_torch_passes_the_variant_gates_at_zero() {
    for v in Variant::ALL {
        let fx = variant(v).unwrap();
        let mut m = Replay::new(true);
        let f = variant_forward_parity(&mut m, &fx, &Gates::EXACT).unwrap();
        assert_eq!((f.loss_rel, f.logits_normwise), (0.0, 0.0), "{v:?}");
        let g = variant_grads_parity(&mut m, &fx, &Gates::EXACT).unwrap();
        assert_eq!(g.worst, 0.0, "{v:?}");
        // Each runner told the model the fixture's own spec first.
        assert_eq!(m.configured, vec![fx.spec.clone(), fx.spec.clone()]);
    }
}

#[test]
fn the_variant_gates_refuse_a_wrong_loss_logit_or_gradient() {
    let fx = variant(Variant::Gqa).unwrap();
    let forward = |loss_scale, bump, gates: &Gates| {
        let mut m = Replay::new(true);
        m.loss_scale = loss_scale;
        m.logit_bump = bump;
        variant_forward_parity(&mut m, &fx, gates)
    };
    let tol = Gates::EXACT;
    assert!(forward(1.0 + 0.9 * tol.forward_loss_rel, 0.0, &tol).is_ok());
    assert!(forward(1.0 + 1.1 * tol.forward_loss_rel, 0.0, &tol).is_err());
    assert!(forward(f64::NAN, 0.0, &tol).is_err());
    // One logit moved by a relative 1.1 tol of the whole tensor's norm.
    let norm = fx
        .logits
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt();
    let bump = |k: f64| (k * tol.logits_normwise_rel * norm) as f32;
    assert!(forward(1.0, bump(0.9), &tol).is_ok());
    assert!(forward(1.0, bump(1.1), &tol).is_err());
    // A looser gate, not the default, accepts what the default refuses.
    let loose = Gates {
        logits_normwise_rel: 1e-3,
        ..tol
    };
    assert!(forward(1.0, bump(5.0), &tol).is_err());
    assert!(forward(1.0, bump(5.0), &loose).is_ok());

    let grads = |scale, gates: &Gates| {
        let mut m = Replay::new(true);
        m.grad_scale = scale;
        variant_grads_parity(&mut m, &fx, gates)
    };
    assert!(grads(1.0 + 0.9 * tol.grad_normwise_rel as f32, &tol).is_ok());
    assert!(grads(1.0 + 1.5 * tol.grad_normwise_rel as f32, &tol).is_err());
    assert!(grads(
        1.0 + 1.5 * tol.grad_normwise_rel as f32,
        &Gates {
            grad_normwise_rel: 1e-3,
            ..tol
        }
    )
    .is_ok());
}

#[test]
fn an_adapter_without_variant_support_fails_loudly_not_on_the_tiny_model() {
    // The default `configure` accepts only the tiny spec. A model that never
    // overrode it must not be run on a GQA or windowed fixture as if it
    // were the tiny one.
    struct TinyOnly;
    impl ParityModel for TinyOnly {
        fn forward(&mut self, _: &TensorSet, _: &TokenBatch) -> Result<(f32, Vec<f32>), OjasError> {
            panic!("ran a variant fixture on the tiny model")
        }
        fn grads(&mut self, _: &TensorSet, _: &TokenBatch, _: f32) -> Result<TensorSet, OjasError> {
            panic!("ran a variant fixture on the tiny model")
        }
        fn train(
            &mut self,
            _: &TensorSet,
            _: &TrainSetup,
            _: usize,
        ) -> Result<(Vec<f64>, TensorSet), OjasError> {
            unreachable!()
        }
    }
    for v in Variant::ALL {
        let fx = variant(v).unwrap();
        let e = variant_forward_parity(&mut TinyOnly, &fx, &Gates::EXACT).unwrap_err();
        assert!(e.detail.contains("only the tiny spec"), "{e}");
        let e = variant_grads_parity(&mut TinyOnly, &fx, &Gates::EXACT).unwrap_err();
        assert!(e.detail.contains("only the tiny spec"), "{e}");
    }
    // Replay with variants off refuses the same way, through its own configure.
    let fx = variant(Variant::Swa).unwrap();
    assert!(variant_forward_parity(&mut Replay::new(false), &fx, &Gates::EXACT).is_err());
}

#[test]
fn gates_must_be_positive_and_finite_and_exact_is_the_pinned_constants() {
    use ojas_oracle::parity::*;
    assert_eq!(
        Gates::EXACT,
        Gates {
            forward_loss_rel: FORWARD_LOSS_REL_TOL,
            logits_normwise_rel: LOGITS_NORMWISE_REL_TOL,
            grad_normwise_rel: GRAD_NORMWISE_REL_TOL,
            trace_loss_abs: TRACE_LOSS_ABS_TOL,
            trace_param_normwise_rel: TRACE_PARAM_NORMWISE_REL_TOL,
            trace_bf16_param_normwise_rel: TRACE_BF16_PARAM_NORMWISE_REL_TOL,
            curve_early_abs: CURVE_EARLY_ABS_TOL,
            curve_late_abs: CURVE_LATE_ABS_TOL,
        }
    );
    assert!(Gates::EXACT.validate().is_ok());
    let fx = variant(Variant::Gqa).unwrap();
    for bad in [0.0, -1e-5, f64::NAN, f64::INFINITY] {
        for gates in [
            Gates {
                forward_loss_rel: bad,
                ..Gates::EXACT
            },
            Gates {
                grad_normwise_rel: bad,
                ..Gates::EXACT
            },
            Gates {
                curve_late_abs: bad,
                ..Gates::EXACT
            },
        ] {
            assert!(gates.validate().is_err(), "accepted {gates:?}");
            // Every `_with` runner refuses before it runs the model.
            assert!(variant_forward_parity(&mut Replay::new(true), &fx, &gates).is_err());
            assert!(forward_parity_with(&mut Replay::new(true), &gates).is_err());
        }
    }
    // The plain runner is the EXACT one.
    assert_eq!(forward_parity(&mut Replay::new(true)).unwrap().worst, 0.0);
    assert_eq!(
        forward_parity_with(&mut Replay::new(true), &Gates::EXACT)
            .unwrap()
            .loss_rel,
        0.0
    );
}
