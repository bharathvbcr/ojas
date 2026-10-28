//! Micro-bench of GENERATE's argmax of caller logits, with and without the
//! memory-pressure probe `dispatch` runs before `op_generate`.
//!
//! Ignored. Auto-discovered, so the manifest has no `[[bench]]` entry.
//!
//! ```text
//! cargo test -p ojas-capi --bench generate_pressure -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The two arms are [`ojas_capi::generate_argmax`]. Neither arm looks up a
//! session or parses a wire payload. `probe=1` is the sysctl plus the
//! critical-pressure refusal, then the same argmax.

use std::hint::black_box;
use std::time::Instant;

const VOCAB: usize = 50_304;
const N: usize = 200;

#[test]
#[ignore]
fn op_generate_argmax_with_and_without_the_pressure_probe() {
    let logits: Vec<f32> = (0..VOCAB).map(|i| (i as f32) * 1.0e-4).collect();
    let plain = ojas_capi::generate_argmax(&logits, false).expect("argmax");
    let probed = ojas_capi::generate_argmax(&logits, true);
    match &probed {
        Ok(token) => assert_eq!(*token, plain, "probe arm changed the token"),
        Err(err) => assert!(
            err.contains("critical memory pressure"),
            "probe arm failed for another reason: {err}"
        ),
    }
    println!(
        "OJAS_META op=op_generate vocab={VOCAB} probe_ok={} token={plain}",
        probed.is_ok()
    );
    report("0", &logits);
    if probed.is_ok() {
        report("1", &logits);
    } else {
        report_refusal(&logits);
    }
}

fn report(probe: &str, logits: &[f32]) {
    let with_probe = probe == "1";
    for _ in 0..20 {
        black_box(ojas_capi::generate_argmax(black_box(logits), with_probe).unwrap());
    }
    let mut ns = Vec::with_capacity(N);
    for _ in 0..N {
        let t0 = Instant::now();
        black_box(ojas_capi::generate_argmax(black_box(logits), with_probe).unwrap());
        ns.push(t0.elapsed().as_nanos());
    }
    print_row(probe, &ns);
}

fn report_refusal(logits: &[f32]) {
    for _ in 0..20 {
        black_box(ojas_capi::generate_argmax(black_box(logits), true).unwrap_err());
    }
    let mut ns = Vec::with_capacity(N);
    for _ in 0..N {
        let t0 = Instant::now();
        black_box(ojas_capi::generate_argmax(black_box(logits), true).unwrap_err());
        ns.push(t0.elapsed().as_nanos());
    }
    print_row("1", &ns);
}

fn print_row(probe: &str, ns: &[u128]) {
    let mut ordered = ns.to_vec();
    ordered.sort_unstable();
    let min_ns = ordered[0];
    let max_ns = ordered[ordered.len() - 1];
    println!(
        "OJAS_OP op=op_generate dir=argmax shape=V{VOCAB} threads=1 n={N} min_ms={:.6} med_ms={:.6} max_ms={:.6} spread_ms={:.6} numerics=na probe={probe}",
        min_ns as f64 / 1e6,
        ordered[ordered.len() / 2] as f64 / 1e6,
        max_ns as f64 / 1e6,
        (max_ns - min_ns) as f64 / 1e6
    );
}
