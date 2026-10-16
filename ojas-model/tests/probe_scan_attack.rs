//! Adversarial tests for card ft-fe7dcee6, lead 4: the two `Probe`
//! source-text scans in `probe_forwarding.rs` (copied here verbatim as
//! `scan_written_methods_call_inner` and `scan_excluded_names_are_written`
//! so they can be run on edited text without touching `common/mod.rs`).
//!
//! Also the structural replacement: ask `Probe` behaviourally whether every
//! op reaches the wrapped backend, with a backend that answers each op with
//! its own name.

mod common;
#[path = "reach/mod.rs"]
mod reach;

use std::sync::Arc;

use common::Probe;
use ojas_core::{forward_backend_methods, Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use reach::{call, op_names, Reach};

const SRC: &str = include_str!("common/mod.rs");

/// `every_written_probe_method_calls_the_inner_method`, parameterised on the
/// source text. `Ok(written)` is the scan passing.
fn scan_written_methods_call_inner(src: &str) -> Result<usize, String> {
    let start = src
        .find("impl<B: Backend> Backend for Probe<B> {")
        .ok_or("the Probe impl")?;
    let body = &src[start..];
    let body = &body[..body.find("\n}\n").ok_or("end of the Probe impl")?];
    let mut written = 0;
    for name in ojas_core::BACKEND_METHODS {
        let Some(at) = body.find(&format!("\n    fn {name}(")) else {
            continue;
        };
        written += 1;
        let rest = &body[at + 1..];
        let end = rest[1..].find("\n    fn ").map_or(rest.len(), |n| n + 1);
        let compact: String = rest[..end].split_whitespace().collect();
        if !compact.contains(&format!("self.inner.{name}(")) {
            return Err(format!(
                "Probe::{name} does not call the inner backend's {name}"
            ));
        }
    }
    if written <= 20 {
        return Err(format!("the scan found only {written} written methods"));
    }
    Ok(written)
}

/// `every_name_excluded_from_generation_is_written_by_probe`, parameterised.
fn scan_excluded_names_are_written(src: &str) -> Result<usize, String> {
    let macro_start = src
        .find("macro_rules! forward_unwritten {")
        .ok_or("the forward_unwritten macro")?;
    let macro_body = &src[macro_start..];
    let macro_body = &macro_body[..macro_body.find("\n}\n").ok_or("end of the macro")?];
    let impl_start = src
        .find("impl<B: Backend> Backend for Probe<B> {")
        .ok_or("the Probe impl")?;
    let impl_body = &src[impl_start..];
    let impl_body = &impl_body[..impl_body.find("\n}\n").ok_or("end of the Probe impl")?];
    let excluded: Vec<&str> = macro_body
        .lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix('(')?;
            let name = &rest[..rest.find(',')?];
            (name != "$name:ident").then_some(name)
        })
        .collect();
    if excluded.len() <= 30 {
        return Err(format!(
            "the scan found only {} excluded names",
            excluded.len()
        ));
    }
    for name in &excluded {
        if !ojas_core::BACKEND_METHODS.contains(name) {
            return Err(format!("{name} is excluded but is not a Backend method"));
        }
        if !impl_body.contains(&format!("\n    fn {name}(")) {
            return Err(format!(
                "{name} is excluded from generation but Probe does not write it"
            ));
        }
    }
    Ok(excluded.len())
}

fn both_scans(src: &str) -> (Result<usize, String>, Result<usize, String>) {
    (
        scan_written_methods_call_inner(src),
        scan_excluded_names_are_written(src),
    )
}

/// Sanity: on the unedited file both copies pass, so a later "accepted"
/// result is the scan being fooled, not the copy being broken.
#[test]
fn the_scan_copies_pass_on_the_real_source() {
    let (a, b) = both_scans(SRC);
    assert!(a.is_ok() && b.is_ok(), "{a:?} {b:?}");
}

/// Edit `\n    fn {name}(` inside `Probe`'s impl only (other test backends
/// in the file write the same signature).
fn in_probe_impl(src: &str, name: &str, with: impl Fn(&str) -> String) -> String {
    let start = src
        .find("impl<B: Backend> Backend for Probe<B> {")
        .expect("the Probe impl");
    let needle = format!("\n    fn {name}(");
    let at = start + src[start..].find(&needle).expect("written by Probe");
    format!("{}{}{}", &src[..at], with(name), &src[at + needle.len()..])
}

fn erase_with_cfg(src: &str, name: &str) -> String {
    in_probe_impl(src, name, |n| format!("\n    #[cfg(any())]\n    fn {n}("))
}

/// Mutation 1: `#[cfg(any())]` on a written method. The method is compiled
/// out, `Probe` falls to the trait default for it, and the scans still see
/// `\n    fn silu_forward(` with its `self.inner.silu_forward(` body.
#[test]
fn mutation_cfg_erased_method_is_accepted_by_both_scans() {
    for name in ["silu_forward", "sync"] {
        if !SRC.contains(&format!("\n    fn {name}(")) {
            continue;
        }
        let (a, b) = both_scans(&erase_with_cfg(SRC, name));
        assert!(
            a.is_ok(),
            "{name}: written-method scan rejected the cfg-erased method: {a:?}"
        );
        assert!(
            b.is_ok(),
            "{name}: excluded-name scan rejected the cfg-erased method: {b:?}"
        );
    }
}

/// The desired property of the scans: a method compiled out must be
/// rejected. FAILS today: see the characterization above.
#[test]
#[ignore = "BREAK: the Probe source scans accept a #[cfg(any())]-erased method (text scan cannot see attributes)"]
fn the_scans_reject_a_cfg_erased_method() {
    let mutated = erase_with_cfg(SRC, "silu_forward");
    let (a, b) = both_scans(&mutated);
    assert!(
        a.is_err() || b.is_err(),
        "both scans accepted a method that is compiled out"
    );
}

/// Mutation 2: the call survives only inside a comment. The method answers
/// itself; the scan collapses whitespace but keeps comments.
#[test]
fn mutation_call_only_in_a_comment_is_accepted_by_the_written_scan() {
    let start = SRC.find("impl<B: Backend> Backend for Probe<B> {").unwrap();
    let at = start + SRC[start..].find("\n    fn silu_forward(").unwrap();
    let call = "self.inner.silu_forward(";
    let call_at = at + SRC[at..].find(call).unwrap();
    let mut mutated = String::from(&SRC[..call_at]);
    mutated.push_str("Err(OjasError::Poisoned) /* ");
    mutated.push_str(&SRC[call_at..call_at + call.len()]);
    mutated.push_str(" */; let _ = (");
    mutated.push_str(&SRC[call_at + call.len()..]);
    // Keep the text valid enough for the scan; it never compiles this.
    let a = scan_written_methods_call_inner(&mutated);
    assert!(a.is_ok(), "the scan noticed the comment-only call: {a:?}");
}

/// Mutation 3: `fn name (` (rustfmt would not write it, a hand edit can).
/// The written-method scan skips it silently; the excluded-name scan is what
/// catches it, loudly. Holds.
#[test]
fn mutation_space_before_paren_is_caught_by_the_excluded_scan() {
    let mutated = in_probe_impl(SRC, "silu_forward", |n| format!("\n    fn {n} ("));
    let (a, b) = both_scans(&mutated);
    assert!(a.is_ok(), "{a:?}");
    let err = b.unwrap_err();
    assert!(err.contains("silu_forward"), "{err}");
}

/// A compiled demonstration of the same class: an excluded name whose
/// replacement is `#[cfg]`-erased. The wrapper compiles, and runs the trait
/// default (Unsupported) where the wrapped CPU answers.
#[test]
fn a_cfg_erased_override_silently_runs_the_trait_default() {
    macro_rules! all_but_sigmoid {
        (sigmoid_forward, { $($item:tt)* }) => {};
        ($name:ident, { $($item:tt)* }) => { $($item)* };
    }
    struct Erased<B>(B);
    impl<B: Backend> Backend for Erased<B> {
        ojas_core::with_backend_methods!(
            forward_backend_methods,
            { self, self.0, none, all_but_sigmoid }
        );
        #[cfg(any())]
        fn sigmoid_forward(&self, x: &Tensor) -> Result<Tensor, ojas_core::OjasError> {
            self.0.sigmoid_forward(x)
        }
    }
    let cpu = CpuBackend::new(Budget::new(1 << 20));
    let x = Tensor::from_f32(&[0.0, 1.0], &[2], cpu.budget()).unwrap();
    assert!(cpu.sigmoid_forward(&x).is_ok());
    assert!(
        Erased(&cpu).sigmoid_forward(&x).is_err(),
        "the erased override unexpectedly forwarded"
    );
}

/// The structural fix, as a test: every op reaches the wrapped backend
/// through `Probe` exactly once, and is counted by the probe. A method
/// compiled out, answered locally or forwarded to the wrong op fails
/// here, whatever the source looks like.
#[test]
fn probe_reaches_the_wrapped_backend_once_per_op() {
    let reach = Arc::new(Reach::new());
    let probe = Probe::new(&reach);
    for name in op_names() {
        let before = (reach.count(name), reach.total());
        let err = call(&probe, name)
            .unwrap()
            .expect_err("Reach answers every op with an error");
        assert!(
            err.to_string().contains(&format!("reached:{name}")),
            "Probe/{name} did not reach the inner op: {err}"
        );
        assert_eq!(
            (reach.count(name), reach.total()),
            (before.0 + 1, before.1 + 1),
            "Probe/{name}: wrong inner call count"
        );
        assert_eq!(probe.count(name), 1, "Probe did not count {name}");
    }
}
