//! The Metal backend against SpikeTrack's real SAM 3.1 fixtures
//! (`ojas_oracle::sam_fixtures`), through the same generic functions as
//! `ojas-oracle/tests/sam_fixtures_cpu.rs`. Every test needs the 3.4 GB file
//! named by `OJAS_SAM_FIXTURES`, so they are ignored by default; run them with
//!
//! ```text
//! OJAS_SAM_FIXTURES=.../SpikeTrack/fixtures/backbone/visem_22_f0.safetensors \
//!   cargo test -p ojas-metal --test sam_fixtures_metal -- --ignored --nocapture --test-threads=1
//! ```
//!
//! With the variable unset an ignored test FAILS: an unrun check must never
//! report the same result as a passing one. Each test prints its rows as a
//! Markdown table.

#![cfg(target_os = "macos")]

use ojas_core::Budget;
use ojas_metal::MetalBackend;
use ojas_oracle::sam_fixtures::{
    attention, axial_tables, gelu, group_norm, group_norm_header, layer_norm_affine_recovered,
    rope_interleaved, skipped_ops, Fixture, OpReport, ENV_FIXTURES, ENV_GROUPNORM_FIXTURES,
};

fn fixture() -> Fixture<'static> {
    Fixture::from_env()
        .unwrap_or_else(|e| panic!("{ENV_FIXTURES}: {e}"))
        .unwrap_or_else(|| panic!("{ENV_FIXTURES} is not set: this test cannot run"))
}

fn metal() -> MetalBackend {
    MetalBackend::new(Budget::new(1 << 33))
        .unwrap_or_else(|e| panic!("Metal device unavailable: {e:?}"))
}

fn show(rows: &[OpReport]) {
    println!("{}", OpReport::header());
    for r in rows {
        println!("{r}");
    }
    let failed: Vec<String> = rows
        .iter()
        .filter(|r| !r.pass)
        .map(|r| format!("{} [{}]", r.op, r.detail))
        .collect();
    assert!(failed.is_empty(), "outside tolerance: {failed:#?}");
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES (SpikeTrack fixtures)"]
fn rope_tables_match_the_recorded_freqs_cis() {
    let (fx, be) = (fixture(), metal());
    show(&[
        axial_tables(&be, &fx, "trunk.blocks.0", [24, 24], 1.0).unwrap(),
        // 24/72 as python computes it (a double), rounded once to f32.
        axial_tables(
            &be,
            &fx,
            "trunk.blocks.7",
            [72, 72],
            (24.0f64 / 72.0) as f32,
        )
        .unwrap(),
    ]);
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES (SpikeTrack fixtures)"]
fn rope_interleaved_matches_torch() {
    let (fx, be) = (fixture(), metal());
    let mut rows = Vec::new();
    for block in ["trunk.blocks.0", "trunk.blocks.7"] {
        for stream in ["q", "k"] {
            rows.push(rope_interleaved(&be, &fx, block, stream).unwrap());
        }
    }
    show(&rows);
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES (SpikeTrack fixtures)"]
fn gelu_erf_matches_torch() {
    let (fx, be) = (fixture(), metal());
    show(&[
        gelu(
            &be,
            &fx,
            "trunk.blocks.0.mlp.fc1.out",
            "trunk.blocks.0.mlp.gelu.out",
        )
        .unwrap(),
        gelu(
            &be,
            &fx,
            "trunk.blocks.7.mlp.fc1.out",
            "trunk.blocks.7.mlp.gelu.out",
        )
        .unwrap(),
        gelu(&be, &fx, "neck.convs.0.gelu.in", "neck.convs.0.gelu").unwrap(),
    ]);
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES (SpikeTrack fixtures)"]
fn windowed_attention_matches_torch() {
    let (fx, be) = (fixture(), metal());
    show(&[attention(&be, &fx, "trunk.blocks.0", None).unwrap()]);
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES (SpikeTrack fixtures)"]
fn global_attention_matches_torch() {
    let (fx, be) = (fixture(), metal());
    let started = std::time::Instant::now();
    let row = attention(&be, &fx, "trunk.blocks.7", None).unwrap();
    println!("global attention, all heads: {:?}", started.elapsed());
    show(&[row]);
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES (SpikeTrack fixtures)"]
fn layer_norm_rows_match_torch_with_the_affine_part_recovered() {
    let (fx, be) = (fixture(), metal());
    let mut rows = vec![layer_norm_affine_recovered(
        &be,
        &fx,
        "trunk.ln_pre.out.in",
        "trunk.ln_pre.out",
        1e-5,
    )
    .unwrap()];
    for block in ["trunk.blocks.0", "trunk.blocks.7"] {
        for norm in ["norm1", "norm2"] {
            rows.push(
                layer_norm_affine_recovered(
                    &be,
                    &fx,
                    &format!("{block}.{norm}.in"),
                    &format!("{block}.{norm}"),
                    1e-6,
                )
                .unwrap(),
            );
        }
    }
    show(&rows);
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES (SpikeTrack fixtures)"]
fn the_ops_the_fixture_cannot_check_are_recorded() {
    let fx = fixture();
    let skipped = skipped_ops(&fx);
    for s in &skipped {
        println!("SKIPPED {}: {} ({})", s.op, s.tensors, s.reason);
    }
    assert!(
        !skipped.is_empty(),
        "the fixture now holds weights: add the weight-bearing ops"
    );
}

#[test]
#[ignore = "needs OJAS_SAM_GROUPNORM_FIXTURES (SpikeTrack groupnorm fixture)"]
fn group_norm_matches_torch_at_both_pixel_decoder_sizes() {
    let fx = Fixture::from_env_var(ENV_GROUPNORM_FIXTURES)
        .unwrap_or_else(|e| panic!("{ENV_GROUPNORM_FIXTURES}: {e}"))
        .unwrap_or_else(|| panic!("{ENV_GROUPNORM_FIXTURES} is not set: this test cannot run"));
    // The values the fixture's header must record, asserted rather than
    // taken from it silently: `group_norm` refuses a header that differs.
    let (groups, eps) = group_norm_header(&fx).unwrap();
    assert_eq!((groups, eps), (8, 1e-5));
    let be = metal();
    let rows: Vec<OpReport> = ["gn0", "gn1"]
        .iter()
        .map(|prefix| {
            let started = std::time::Instant::now();
            let row = group_norm(&be, &fx, prefix, groups, eps).unwrap();
            println!("{prefix}: {:?}", started.elapsed());
            row
        })
        .collect();
    show(&rows);
}
