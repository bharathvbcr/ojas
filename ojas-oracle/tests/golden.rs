//! Integrity of the tiny golden fixtures: shapes, §2 names and init, the
//! metadata each file reports, and consistency between files.

mod support;

use ojas_oracle::golden::{
    fnv1a64, load_init, lr_schedules, parse_batch_starts, tiny_batch_starts, tiny_forward,
    tiny_grads, tiny_init, tiny_token_bytes, tiny_tokens, tiny_trace, GradsAt, LogitsChecksums,
    Ns5, GOLDEN_SEED, LR_CASES, TORCH_VERSION,
};
use ojas_oracle::parity::{check_curve, TRACE_LOSS_ABS_TOL};
use ojas_oracle::safetensors::SafeTensors;
use ojas_oracle::spec::{expected_params, parse_spec, tiny_spec, Init};
use support::{build, f32_raw, Raw};

const INIT_BYTES: &[u8] = include_bytes!("../fixtures/tiny/init.safetensors");

fn mean_std(v: &[f32]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let var = v
        .iter()
        .map(|&x| (f64::from(x) - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean, var.sqrt())
}

#[test]
fn init_spec_names_and_provenance() {
    let init = tiny_init().unwrap();
    assert_eq!(init.spec, tiny_spec());
    assert_eq!(
        init.provenance.generator,
        "ojas-oracle/python/export_init.py"
    );
    assert_eq!(init.provenance.torch, TORCH_VERSION);
    assert_eq!(init.provenance.kind().unwrap(), "init");
    assert_eq!(
        init.provenance.meta.count("seed").unwrap() as u64,
        GOLDEN_SEED
    );
    assert_eq!(init.params.len(), expected_params(&init.spec).len());
    assert!(
        init.params.get("lm_head.weight").is_none(),
        "only tok_emb is stored"
    );
    let names: Vec<&str> = init.params.names();
    assert!(names.iter().all(|n| !n.starts_with("_orig_mod.")));
    // The exporter hardcodes nanolab's RMSNorm eps; it must be ojas's.
    assert_eq!(init.spec.rms_eps as f32, ojas_core::RMS_NORM_EPS);
}

#[test]
fn init_values_follow_the_section_2_table() {
    let init = tiny_init().unwrap();
    for row in expected_params(&init.spec) {
        let t = init.params.get(&row.name).unwrap();
        match row.init {
            Init::Zeros => assert!(t.data.iter().all(|&v| v == 0.0), "{} not zero", row.name),
            Init::Ones => assert!(t.data.iter().all(|&v| v == 1.0), "{} not ones", row.name),
            Init::Normal(std) => {
                let (mean, s) = mean_std(&t.data);
                let n = t.data.len() as f64;
                assert!(
                    mean.abs() <= 6.0 * std / n.sqrt(),
                    "{} mean {mean}",
                    row.name
                );
                assert!(
                    (s / std - 1.0).abs() <= 6.0 / (2.0 * n).sqrt(),
                    "{} std {s}",
                    row.name
                );
            }
        }
    }
    // The §2 row that a read of mixers.py alone gets wrong: the gate weight is
    // zeroed there and then re-drawn by GPT._init_weights.
    let gate = init.params.get("blocks.0.mixer.gate.weight").unwrap();
    assert!(gate.data.iter().any(|&v| v != 0.0));
}

#[test]
fn load_init_refuses_renamed_reshaped_and_untied_tensors() {
    let st = SafeTensors::parse(INIT_BYTES).unwrap();
    let spec = st.metadata("ojas.spec").unwrap().to_string();
    let oracle = st.metadata("ojas.oracle").unwrap().to_string();
    let meta = [
        ("ojas.spec", spec.as_str()),
        ("ojas.oracle", oracle.as_str()),
    ];
    let raws = |edit: &dyn Fn(&mut Vec<Raw>)| {
        let mut v: Vec<Raw> = st
            .entries()
            .iter()
            .map(|e| f32_raw(&e.name, &e.shape, &st.f32s(&e.name).unwrap()))
            .collect();
        edit(&mut v);
        build(&v, &meta)
    };
    assert!(load_init(&raws(&|_| {})).is_ok(), "rebuilt file must load");

    let renamed = raws(&|v| v[0].name = "embedding.weight".into());
    assert!(load_init(&renamed).is_err());

    let reshaped = raws(&|v| {
        let i = v
            .iter()
            .position(|r| r.name == "blocks.0.mixer.gate.bias")
            .unwrap();
        v[i].shape = vec![2, 2];
    });
    assert!(load_init(&reshaped).is_err());

    let emb = init_tensor(&st, "tok_emb.weight");
    let tied = raws(&|v| v.push(f32_raw("lm_head.weight", &[256, 64], &emb)));
    let loaded = load_init(&tied).unwrap();
    assert!(loaded.params.get("lm_head.weight").is_none());

    let mut untied_vals = emb.clone();
    untied_vals[17] = f32::from_bits(untied_vals[17].to_bits() ^ 1);
    let untied = raws(&|v| v.push(f32_raw("lm_head.weight", &[256, 64], &untied_vals)));
    assert!(load_init(&untied).is_err());

    let no_spec = build(&[], &[("ojas.oracle", oracle.as_str())]);
    assert!(load_init(&no_spec).is_err());
}

fn init_tensor(st: &SafeTensors<'_>, name: &str) -> Vec<f32> {
    st.f32s(name).unwrap()
}

#[test]
fn forward_batch_is_the_first_dumped_micro_batch() {
    let fx = tiny_forward().unwrap();
    let starts = tiny_batch_starts().unwrap();
    let tokens = tiny_tokens().unwrap();
    assert_eq!(fx.batch, starts.micro_batch(&tokens, 0, 0).unwrap());
    let (b, t) = (fx.batch.batch, fx.batch.seq_len);
    assert_eq!((b, t), (2, 32));
    for r in 0..b {
        assert_eq!(
            fx.batch.x[r * t + 1..(r + 1) * t],
            fx.batch.y[r * t..(r + 1) * t - 1]
        );
    }
    assert_eq!(
        fx.provenance
            .meta
            .object("batch")
            .unwrap()
            .numbers("starts")
            .unwrap(),
        {
            starts
                .micro_starts(0, 0)
                .unwrap()
                .iter()
                .map(|&s| s as f64)
                .collect::<Vec<_>>()
        }
    );
}

#[test]
fn forward_checksums_and_loss_agree_with_the_stored_logits() {
    let fx = tiny_forward().unwrap();
    assert_eq!(fx.logits.len(), 2 * 32 * 256);
    let got = LogitsChecksums::of(&fx.logits);
    assert_eq!(got.count, fx.checksums.count);
    for (name, a, b) in [
        ("sum", got.sum, fx.checksums.sum),
        ("sum_sq", got.sum_sq, fx.checksums.sum_sq),
        ("weighted", got.weighted, fx.checksums.weighted),
    ] {
        assert!(
            (a - b).abs() <= 1e-9 * b.abs().max(1.0),
            "{name}: {a} vs {b}"
        );
    }
    // Mean cross-entropy recomputed in f64 from the stored logits.
    let v = fx.vocab;
    let mut total = 0.0f64;
    for (i, &target) in fx.batch.y.iter().enumerate() {
        let row = &fx.logits[i * v..(i + 1) * v];
        let max = row
            .iter()
            .map(|&x| f64::from(x))
            .fold(f64::NEG_INFINITY, f64::max);
        let lse = max
            + row
                .iter()
                .map(|&x| (f64::from(x) - max).exp())
                .sum::<f64>()
                .ln();
        total += lse - f64::from(row[target as usize]);
    }
    let ce = total / fx.batch.y.len() as f64;
    let rel = (ce - f64::from(fx.loss)).abs() / ce;
    assert!(rel < 1e-6, "stored loss {} vs recomputed {ce}", fx.loss);
    // Zero-initialised blocks: the init is close to uniform over 256 tokens.
    assert!((f64::from(fx.loss) - 256f64.ln()).abs() < 0.1);
}

#[test]
fn init_gradients_are_degenerate_exactly_where_zero_init_says() {
    let g = tiny_grads(GradsAt::Init).unwrap();
    assert_eq!(g.loss_seed, 0.5, "seed is 1/K with K = 2");
    assert_eq!(g.no_grad, vec!["blocks.0.mixer.vr_lambda".to_string()]);
    // o_proj and ffn.down are zero, so nothing upstream of them in a block
    // gets a gradient; only the embedding, the two zero projections and the
    // final norm do.
    let mut nonzero: Vec<&str> = g
        .grads
        .iter()
        .filter(|t| !g.zero_grad.contains(&t.name))
        .map(|t| t.name.as_str())
        .collect();
    nonzero.sort_unstable();
    assert_eq!(
        nonzero,
        vec![
            "blocks.0.ffn.down.weight",
            "blocks.0.mixer.o_proj.weight",
            "blocks.1.ffn.down.weight",
            "blocks.1.mixer.o_proj.weight",
            "norm_f.weight",
            "tok_emb.weight",
        ]
    );
    let fx = tiny_forward().unwrap();
    assert_eq!(
        g.loss.to_bits(),
        fx.loss.to_bits(),
        "same params, same batch"
    );
}

#[test]
fn step5_gradients_exercise_every_parameter() {
    let g = tiny_grads(GradsAt::Step5).unwrap();
    assert!(
        g.zero_grad.is_empty(),
        "zero grads at step 5: {:?}",
        g.zero_grad
    );
    assert_eq!(g.no_grad, vec!["blocks.0.mixer.vr_lambda".to_string()]);
    assert!(g.grads.iter().all(|t| t.data.iter().all(|v| v.is_finite())));
    assert!(g.loss < tiny_forward().unwrap().loss);
}

#[test]
fn traces_record_nanolab_semantics() {
    for ns5 in [Ns5::F32, Ns5::Bf16] {
        let tr = tiny_trace(ns5).unwrap();
        let s = &tr.train;
        assert_eq!((tr.steps, tr.params_after_step), (40, 5));
        assert_eq!(
            (s.seed, s.batch, s.accum, s.seq_len),
            (GOLDEN_SEED, 2, 2, 32)
        );
        assert_eq!((s.warmup_steps, s.total_steps), (4, 40));
        assert_eq!(
            (s.optimizer.as_str(), s.schedule.as_str()),
            ("muon_ns5_adamw", "cosine")
        );
        assert_eq!((s.lr, s.matrix_lr, s.grad_clip), (6e-4, 0.025, 1.0));
        assert_eq!((s.beta1, s.beta2, s.eps), (0.9, 0.95, 1e-8));
        assert!(s.token_bin.ends_with("fixtures/tiny/tokens.bin") && s.token_bin.exists());
        assert_eq!(tr.provenance.meta.count("crosschecked_steps").unwrap(), 5);
        assert!(tr.provenance.meta.count("ns5_calls").unwrap() > 0);
        for step in 0..tr.steps {
            let (a, b) = (tr.micro_loss[2 * step], tr.micro_loss[2 * step + 1]);
            assert_eq!(tr.mean_loss[step], (a + b) / 2.0, "mean of micro-losses");
            // train.py:348 logs the last micro-batch (l/K)*K, which is exact.
            assert_eq!(tr.nanolab_loss[step], b);
            assert!(tr.grad_norm[step] > 0.0);
        }
        // Layer 0 has no earlier values to blend: no gradient, never updated.
        assert_eq!(
            tr.params.get("blocks.0.mixer.vr_lambda").unwrap().data,
            vec![0.0]
        );
        let o = tr.params.get("blocks.0.mixer.o_proj.weight").unwrap();
        assert!(o.data.iter().any(|&v| v != 0.0), "o_proj moved off zero");
    }
    let f32_trace = tiny_trace(Ns5::F32).unwrap();
    let ns5_patch = f32_trace.provenance.meta.str("ns5_patch").unwrap();
    assert!(ns5_patch.contains("G.float()") && ns5_patch.contains("G.bfloat16()"));
}

#[test]
fn f32_curve_meets_section_10_and_bf16_differs_beyond_the_trace_gate() {
    let f = tiny_trace(Ns5::F32).unwrap();
    let b = tiny_trace(Ns5::Bf16).unwrap();
    check_curve(&f.mean_loss, &f.mean_loss, 256).unwrap();
    // Step 1 runs before any optimizer step: identical in both.
    assert_eq!(f.mean_loss[0], b.mean_loss[0]);
    // F8: bf16 NS5 alone moves the 5-step trace past its gate, which is why
    // the oracle patches NS5 to f32.
    let worst = (0..f.params_after_step)
        .map(|s| (f.mean_loss[s] - b.mean_loss[s]).abs())
        .fold(0.0, f64::max);
    assert!(
        worst > TRACE_LOSS_ABS_TOL,
        "f32 vs bf16 NS5 worst |Δ| {worst}"
    );
}

#[test]
fn lr_schedule_fixture_has_every_case() {
    let lr = lr_schedules().unwrap();
    assert_eq!(lr.cases.len(), LR_CASES.len());
    assert_eq!(lr.provenance.generator, "ojas-oracle/python/golden.py");
    for c in &lr.cases {
        assert_eq!(c.multipliers.len(), 20);
        assert_eq!(
            (c.lr_floor_frac, c.wsd_decay_frac, c.peak),
            (0.1, 0.2, 0.025)
        );
        for (s, &m) in c.multipliers.iter().enumerate().take(c.warmup_steps) {
            assert!((m - (s + 1) as f64 / c.warmup_steps as f64).abs() < 1e-15);
        }
    }
}

#[test]
fn batch_starts_match_the_token_bin() {
    let d = tiny_batch_starts().unwrap();
    let tokens = tiny_tokens().unwrap();
    assert_eq!(
        (d.seed, d.seq_len, d.batch, d.accum, d.steps),
        (GOLDEN_SEED, 32, 2, 2, 40)
    );
    assert_eq!(d.bin_header_bytes, 0);
    assert_eq!(d.bin_tokens, tokens.len());
    assert_eq!(d.bin_fnv1a64, fnv1a64(tiny_token_bytes()));
    assert_eq!(d.windows_per_epoch, (tokens.len() - 1) / 32);
    assert_eq!(d.cursor_after, (0, 160));
    let mut seen = d.starts.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), d.starts.len(), "one epoch: no window twice");
    let rows = d.rows.as_ref().unwrap();
    for (i, &s) in d.starts.iter().enumerate() {
        assert_eq!(s % 32, 0);
        let s = s as usize;
        assert_eq!(rows[i * 33..(i + 1) * 33], tokens[s..s + 33], "window {i}");
    }
}

/// The 124M export is a generated artifact (about 0.5 GB), not a fixture.
/// Generate it, then run this with `--ignored`; a missing file fails.
#[test]
#[ignore = "needs generated/124m/model.safetensors: python/export_init.py --out ojas-oracle/generated/124m/model.safetensors"]
fn generated_124m_init_follows_section_2() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/generated/124m/model.safetensors"
    );
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let init = load_init(&bytes).unwrap();
    let s = &init.spec;
    assert_eq!(
        (s.n_layer, s.n_embd, s.n_head, s.head_dim, s.hidden, s.vocab, s.max_seq),
        (12, 768, 12, 64, 2048, 50304, 1024)
    );
    for row in expected_params(s) {
        let t = init.params.get(&row.name).unwrap();
        match row.init {
            Init::Zeros => assert!(t.data.iter().all(|&v| v == 0.0), "{}", row.name),
            Init::Ones => assert!(t.data.iter().all(|&v| v == 1.0), "{}", row.name),
            Init::Normal(std) => {
                let (mean, sd) = mean_std(&t.data);
                let n = t.data.len() as f64;
                assert!(
                    mean.abs() <= 6.0 * std / n.sqrt(),
                    "{} mean {mean}",
                    row.name
                );
                assert!(
                    (sd / std - 1.0).abs() <= 6.0 / (2.0 * n).sqrt(),
                    "{} std {sd}",
                    row.name
                );
            }
        }
    }
}

#[test]
fn spec_parser_refuses_drift() {
    let good = r#"{"format":"ojas-spec-v1","arch":"nanolab-gpt","vocab":256,"n_embd":64,
        "n_layer":2,"n_head":4,"n_kv_head":4,"head_dim":16,"hidden":192,"max_seq":32,
        "rope_base":10000.0,"rms_eps":1e-06,"tie_embeddings":true,"qk_norm":true,
        "gated_attention":true,"value_residual":true}"#;
    assert_eq!(parse_spec(good).unwrap(), tiny_spec());
    for (from, to) in [
        (r#""vocab":256"#, r#""vocab":256.5"#),
        (r#""vocab":256"#, r#""vocab":0"#),
        (r#""qk_norm":true"#, r#""qk_norm":1"#),
        (r#""rms_eps":1e-06"#, r#""rms_eps":-1e-06"#),
        (r#""format":"ojas-spec-v1""#, r#""format":"ojas-spec-v2""#),
        (r#""n_kv_head":4"#, r#""n_kv_head":3"#),
        (r#""head_dim":16"#, r#""head_dim":15"#),
        (r#""hidden":192"#, r#""hidden":192,"mup":false"#),
        (r#""hidden":192,"#, ""),
    ] {
        let bad = good.replacen(from, to, 1);
        assert_ne!(bad, good);
        assert!(parse_spec(&bad).is_err(), "accepted {to:?}");
    }
}

/// Counts and shape entries are exact integer literals, as in the model's
/// spec reader: `256.0` is refused in the spec, in fixture metadata, and in a
/// fixture shape. Python's `json.dumps` writes ints without `.0`, so the
/// checked-in fixtures are unaffected.
#[test]
fn whole_floats_are_not_counts() {
    let spec = r#"{"format":"ojas-spec-v1","arch":"nanolab-gpt","vocab":256,"n_embd":64,
        "n_layer":2,"n_head":4,"n_kv_head":4,"head_dim":16,"hidden":192,"max_seq":32,
        "rope_base":10000.0,"rms_eps":1e-06,"tie_embeddings":true,"qk_norm":true,
        "gated_attention":true,"value_residual":true}"#;
    assert_eq!(parse_spec(spec).unwrap(), tiny_spec());
    // A float rope_base is still a float; an integer one reads exactly.
    let int_rope = spec.replacen("10000.0", "10000", 1);
    assert_eq!(parse_spec(&int_rope).unwrap(), tiny_spec());
    for (from, to) in [
        (r#""vocab":256"#, r#""vocab":256.0"#),
        (r#""n_layer":2"#, r#""n_layer":2e0"#),
        (r#""max_seq":32"#, r#""max_seq":-0"#),
    ] {
        let bad = spec.replacen(from, to, 1);
        assert_ne!(bad, spec);
        let err = parse_spec(&bad).expect_err(to);
        assert!(err.to_string().contains("unsigned integer"), "{to}: {err}");
    }

    let starts = include_str!("../fixtures/tiny/batch_starts.json");
    assert!(parse_batch_starts(starts).is_ok());
    for (from, to) in [
        ("\"seq_len\": 32", "\"seq_len\": 32.0"),
        ("\"steps\": 40", "\"steps\": 4e1"),
        ("\"seed\": 1337", "\"seed\": 1337.0"),
    ] {
        let bad = starts.replacen(from, to, 1);
        assert_ne!(bad, starts, "{from} not found");
        let err = parse_batch_starts(&bad).expect_err(to);
        assert!(err.to_string().contains("unsigned integer"), "{to}: {err}");
    }
    let cursor = starts.find("\"cursor_after\"").unwrap();
    let open = cursor + starts[cursor..].find('[').unwrap();
    let close = open + starts[open..].find(',').unwrap();
    let bad = format!(
        "{}{}.0{}",
        &starts[..open + 1],
        starts[open + 1..close].trim(),
        &starts[close..]
    );
    assert!(parse_batch_starts(&bad).is_err(), "float cursor accepted");

    let rms = r#"{"format": "ojas-oracle-fixture-v1", "op": "rms_norm", "eps": 1e-6,
        "input_shape": [2], "input": [0.5, 2.0], "weight_shape": [2], "weight": [1.0, 1.0],
        "expected_shape": [2], "expected": [1.0, 1.0]}"#;
    assert!(ojas_oracle::parse_rms_norm(rms).is_ok());
    let bad = rms.replacen("\"input_shape\": [2]", "\"input_shape\": [2.0]", 1);
    assert!(
        ojas_oracle::parse_rms_norm(&bad).is_err(),
        "float shape accepted"
    );
}
