//! The TensorOps flash kernel on cross-attention: `ojas_attn_fwd_d32` run with
//! `cross` (`Tq != Tk`, every key visible, no window; the generalisation of the
//! causal forward `attn_fwd_tiled_body`), taken by `attention_forward_saving`
//! and `attention_forward` for head dim 32 with no bias and no key lengths.
//! Every output is held to the CPU reference and to both device references
//! the dispatcher used to pick between (`attention_forward_saving_splitk` and
//! `attention_forward_saving_scalar`) on the same inputs, at stage 1's
//! tolerance: the ragged last 64-key block (`Tk` = 1, a block plus or minus
//! one), `Tq` above and below `Tk`, `Tq` = 1, grouped heads, batches, scores
//! whose maximum moves between key blocks (a dropped rescale shows there), a
//! budget with room for the outputs only, and the contracts around the route
//! (planned kernel named, one dispatch, repeatable bit for bit, a non-finite
//! operand is a deferred fault under the called op).

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{AttentionOpts, Backend, Budget, Tensor};
use ojas_metal::MetalBackend;

fn tol(tk: usize) -> f32 {
    4e-5 * (tk as f32).sqrt().max(1.0)
}

#[derive(Clone, Debug)]
struct Case {
    b: usize,
    h: usize,
    hkv: usize,
    tq: usize,
    tk: usize,
}

impl Case {
    fn new(b: usize, h: usize, hkv: usize, tq: usize, tk: usize) -> Self {
        Case { b, h, hkv, tq, tk }
    }
}

const D: usize = 32;

struct Operands {
    q: Tensor,
    k: Tensor,
    v: Tensor,
}

impl Operands {
    fn host(c: &Case, seed: u64) -> Self {
        Operands {
            q: rand(&[c.b, c.h, c.tq, D], seed, 1.0),
            k: rand(&[c.b, c.hkv, c.tk, D], seed + 1, 1.0),
            v: rand(&[c.b, c.hkv, c.tk, D], seed + 2, 1.0),
        }
    }
    fn with(c: &Case, q: Vec<f32>, k: Vec<f32>, v: Vec<f32>) -> Self {
        Operands {
            q: host(&q, &[c.b, c.h, c.tq, D]),
            k: host(&k, &[c.b, c.hkv, c.tk, D]),
            v: host(&v, &[c.b, c.hkv, c.tk, D]),
        }
    }
    fn up(&self, m: &MetalBackend) -> Operands {
        Operands {
            q: up(m, &self.q),
            k: up(m, &self.k),
            v: up(m, &self.v),
        }
    }
}

fn opts() -> AttentionOpts<'static> {
    AttentionOpts::default()
}

/// Dispatches one call encodes on fresh uploads.
fn dispatches(m: &MetalBackend, host: &Operands, which: &str) -> u64 {
    let o = host.up(m);
    ok("sync", m.sync());
    let before = m.dispatches();
    let out = match which {
        "auto" => m.attention_forward_saving(&o.q, &o.k, &o.v, opts()),
        "splitk" => m.attention_forward_saving_splitk(&o.q, &o.k, &o.v, opts()),
        _ => m.attention_forward_saving_scalar(&o.q, &o.k, &o.v, opts()),
    };
    let out = ok(which, out);
    ok("sync", m.sync());
    drop(out);
    m.dispatches() - before
}

/// The flash kernel agrees with the CPU, with split-K and with the scalar
/// kernel (output and log-sum-exp), repeats bit for bit, is planned as
/// `"flash"` and, when `count`, costs one dispatch as the scalar call does.
fn check(m: &MetalBackend, c: &Case, host: &Operands, count: bool) {
    let cpu = cpu();
    let dev = host.up(m);
    let tag = format!("{c:?}");
    assert_eq!(
        ok(
            &tag,
            m.attention_saving_kernel(&dev.q, &dev.k, &dev.v, opts())
        ),
        "flash",
        "{tag}: planned kernel"
    );
    let (want, want_lse) = ok(
        &tag,
        cpu.attention_forward_saving(&host.q, &host.k, &host.v, opts()),
    );
    let (got, got_lse) = ok(
        &tag,
        m.attention_forward_saving(&dev.q, &dev.k, &dev.v, opts()),
    );
    let (sk, sk_lse) = ok(
        &tag,
        m.attention_forward_saving_splitk(&dev.q, &dev.k, &dev.v, opts()),
    );
    let (sc, sc_lse) = ok(
        &tag,
        m.attention_forward_saving_scalar(&dev.q, &dev.k, &dev.v, opts()),
    );
    ok("sync", m.sync());
    same_tensor(&format!("{tag} output"), &got, &want, tol(c.tk), 1e-4);
    same_tensor(&format!("{tag} lse"), &got_lse, &want_lse, tol(c.tk), 1e-4);
    for (name, o, l) in [("split-K", &sk, &sk_lse), ("scalar", &sc, &sc_lse)] {
        close(
            &format!("{tag} output vs {name}"),
            &down(&got),
            &down(o),
            tol(c.tk),
            1e-4,
        );
        close(
            &format!("{tag} lse vs {name}"),
            &down(&got_lse),
            &down(l),
            tol(c.tk),
            1e-4,
        );
    }
    let (again, again_lse) = ok(
        &tag,
        m.attention_forward_saving(&dev.q, &dev.k, &dev.v, opts()),
    );
    ok("sync", m.sync());
    assert_eq!(down(&again), down(&got), "{tag}: not repeatable");
    assert_eq!(
        down(&again_lse),
        down(&got_lse),
        "{tag}: lse not repeatable"
    );
    if count {
        let (flash, scalar) = (dispatches(m, host, "auto"), dispatches(m, host, "scalar"));
        assert_eq!(flash, scalar, "{tag}: one kernel, as the scalar call");
        assert_eq!(
            dispatches(m, host, "splitk"),
            scalar + 1,
            "{tag}: split-K is partials and combine"
        );
    }
}

#[test]
fn query_and_key_counts_around_the_key_block_and_the_query_block_match() {
    let m = metal();
    // Key blocks are 64 keys, query blocks 32 rows: ragged last blocks on
    // both, a single key, a single query, and each side longer than the other.
    let tqs = [1usize, 7, 31, 32, 33, 100, 129];
    let tks = [1usize, 31, 63, 64, 65, 127, 129, 200];
    let mut seed = 100;
    for &tq in &tqs {
        for &tk in &tks {
            seed += 7;
            let c = Case::new(1, 2, 1, tq, tk);
            check(&m, &c, &Operands::host(&c, seed), tq == 33 && tk == 65);
        }
    }
}

#[test]
fn the_sam_shapes_and_batches_and_grouped_heads_match() {
    let m = metal();
    for (i, c) in [
        Case::new(1, 8, 8, 5184, 5200),
        Case::new(1, 8, 8, 5184, 10400),
        Case::new(1, 8, 8, 201, 5184),
        Case::new(2, 4, 4, 33, 130),
        Case::new(2, 4, 2, 33, 130),
        Case::new(1, 6, 2, 40, 77),
        Case::new(3, 2, 2, 1, 1000),
        Case::new(1, 2, 2, 150, 40),
    ]
    .iter()
    .enumerate()
    {
        check(&m, c, &Operands::host(c, 300 + 5 * i as u64), i % 2 == 0);
    }
}

/// Scores whose maximum moves between key blocks: keys grow along the key
/// axis (the last block holds the maximum), shrink along it (the first
/// does), and a query with large logits throughout. A rescale of the running
/// output that is dropped or applied to the wrong row shows as a large error.
#[test]
fn a_row_maximum_that_moves_between_key_blocks_is_rescaled() {
    let m = metal();
    let c = Case::new(1, 2, 2, 33, 300);
    let base = Operands::host(&c, 700);
    let q = ok("q", base.q.to_f32_vec());
    let k = ok("k", base.k.to_f32_vec());
    let v = ok("v", base.v.to_f32_vec());
    let per_key = |scale: &dyn Fn(usize) -> f32| -> Vec<f32> {
        k.iter()
            .enumerate()
            .map(|(i, x)| x * scale((i / D) % c.tk))
            .collect()
    };
    let loud_q: Vec<f32> = q.iter().map(|x| x * 6.0).collect();
    for (what, qs, ks) in [
        ("growing", q.clone(), per_key(&|j| 1.0 + j as f32 / 20.0)),
        (
            "shrinking",
            q.clone(),
            per_key(&|j| 1.0 + (c.tk - j) as f32 / 20.0),
        ),
        (
            "large logits",
            loud_q.clone(),
            per_key(&|j| 1.0 + (j % 7) as f32),
        ),
        (
            "alternating blocks",
            loud_q,
            per_key(&|j| if (j / 64) % 2 == 0 { 0.2 } else { 3.0 }),
        ),
    ] {
        let case = Operands::with(&c, qs, ks, v.clone());
        eprintln!("{what}");
        check(&m, &c, &case, false);
    }
}

/// The flash kernel needs nothing beyond its two outputs, so a budget with
/// room for the operands and the outputs only still runs it, one dispatch,
/// where split-K's partials would not fit and the scalar kernel would run.
#[test]
fn a_budget_that_holds_only_the_outputs_still_runs_the_flash_kernel() {
    let c = Case::new(1, 2, 2, 40, 3000);
    let host = Operands::host(&c, 800);
    let (want, want_lse) = ok(
        "cpu",
        cpu().attention_forward_saving(&host.q, &host.k, &host.v, opts()),
    );
    let planes = c.b * c.h;
    let q_n = planes * c.tq * D;
    let kv_n = planes * c.tk * D;
    // Operands, output and lse, and 1 KiB over.
    let room = ((q_n + 2 * kv_n + q_n + planes * c.tq) * 4 + 1024) as u64;
    let tight = MetalBackend::new(Budget::new(room))
        .unwrap_or_else(|e| panic!("Metal device unavailable: {e:?}"));
    let dev = host.up(&tight);
    let baseline = ok("live", tight.budget().live_bytes());
    assert_eq!(
        ok(
            "kernel",
            tight.attention_saving_kernel(&dev.q, &dev.k, &dev.v, opts())
        ),
        "flash"
    );
    let count = |scalar: bool| {
        ok("sync", tight.sync());
        let before = tight.dispatches();
        let out = if scalar {
            tight.attention_forward_saving_scalar(&dev.q, &dev.k, &dev.v, opts())
        } else {
            tight.attention_forward_saving(&dev.q, &dev.k, &dev.v, opts())
        };
        let out = ok("call", out);
        ok("sync", tight.sync());
        drop(out);
        tight.dispatches() - before
    };
    assert_eq!(
        count(false),
        count(true),
        "flash is one kernel: no partials, no combine"
    );
    let (got, got_lse) = ok(
        "tight",
        tight.attention_forward_saving(&dev.q, &dev.k, &dev.v, opts()),
    );
    ok("sync", tight.sync());
    same_tensor("tight output", &got, &want, tol(c.tk), 1e-4);
    same_tensor("tight lse", &got_lse, &want_lse, tol(c.tk), 1e-4);
    drop((got, got_lse));
    assert_eq!(ok("live", tight.budget().live_bytes()), baseline);
}

/// `attention_forward` (the inference form) takes the flash kernel too where
/// the pad route does not (`Tk` beyond 2.25 `Tq`), and agrees with the CPU.
#[test]
fn attention_forward_beyond_the_pad_ratio_runs_the_flash_kernel() {
    let m = metal();
    let c = Case::new(1, 8, 8, 5184, 15584);
    let host = Operands::host(&c, 900);
    let dev = host.up(&m);
    let want = ok(
        "cpu",
        cpu().attention_forward(&host.q, &host.k, &host.v, opts()),
    );
    ok("sync", m.sync());
    let before = m.dispatches();
    let got = ok("metal", m.attention_forward(&dev.q, &dev.k, &dev.v, opts()));
    ok("sync", m.sync());
    let n = m.dispatches() - before;
    // Three operand scans, the kernel and two output checks: not the pad
    // route's nine.
    assert!(n < 9, "{n} dispatches: the pad route ran");
    same_tensor("15584", &got, &want, tol(c.tk), 1e-4);
}

/// A NaN or infinity anywhere in an operand, including the ragged last key
/// and the last query row, is a deferred fault under the name that was called.
#[test]
fn a_non_finite_operand_faults_under_the_name_that_was_called() {
    let m = metal();
    let c = Case::new(1, 2, 2, 33, 65);
    for (what, at) in [
        ("q last row", 0usize),
        ("k last key", 1),
        ("v last key", 2),
        ("v first", 3),
    ] {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let host_ops = Operands::host(&c, 950);
            let mut q = ok("q", host_ops.q.to_f32_vec());
            let mut k = ok("k", host_ops.k.to_f32_vec());
            let mut v = ok("v", host_ops.v.to_f32_vec());
            match at {
                0 => *q.last_mut().expect("q") = bad,
                1 => *k.last_mut().expect("k") = bad,
                2 => *v.last_mut().expect("v") = bad,
                _ => v[0] = bad,
            }
            let ops = Operands::with(&c, q, k, v).up(&m);
            let r = m.attention_forward_saving(&ops.q, &ops.k, &ops.v, opts());
            let _kept = deferred(&m, &format!("{what} {bad}"), r, "attention_forward_saving");
        }
    }
    // The inference form names itself.
    let ops = Operands::host(&c, 951);
    let mut v = ok("v", ops.v.to_f32_vec());
    v[7] = f32::NAN;
    let ops = Operands::with(
        &c,
        ok("q", ops.q.to_f32_vec()),
        ok("k", ops.k.to_f32_vec()),
        v,
    )
    .up(&m);
    let r = m.attention_forward(&ops.q, &ops.k, &ops.v, opts());
    let _kept = deferred(&m, "attention_forward", r, "attention_forward");
}

/// A call the flash route does not cover stays on its kernel: a bias, key
/// lengths and head dim 16 are split-K, causal and head dim 64 the scalar one.
#[test]
fn the_calls_the_route_does_not_cover_are_planned_elsewhere() {
    let m = metal();
    let c = Case::new(1, 2, 2, 40, 90);
    let host_ops = Operands::host(&c, 970);
    let dev = host_ops.up(&m);
    let bias = up(&m, &rand(&[1, 2, 40, 90], 971, 1.0));
    let lens = up(&m, &host_u32(&[50], &[1]));
    let kernel = |o: AttentionOpts<'_>| {
        ok(
            "kernel",
            m.attention_saving_kernel(&dev.q, &dev.k, &dev.v, o),
        )
    };
    assert_eq!(kernel(opts()), "flash");
    assert_eq!(
        kernel(AttentionOpts {
            bias: Some(&bias),
            ..opts()
        }),
        "splitk"
    );
    assert_eq!(
        kernel(AttentionOpts {
            key_lens: Some(&lens),
            ..opts()
        }),
        "splitk"
    );
    assert_eq!(
        kernel(AttentionOpts {
            causal: true,
            ..opts()
        }),
        "scalar"
    );
    let d16 = Operands {
        q: up(&m, &rand(&[1, 2, 40, 16], 972, 1.0)),
        k: up(&m, &rand(&[1, 2, 90, 16], 973, 1.0)),
        v: up(&m, &rand(&[1, 2, 90, 16], 974, 1.0)),
    };
    assert_eq!(
        ok(
            "d16",
            m.attention_saving_kernel(&d16.q, &d16.k, &d16.v, opts())
        ),
        "splitk"
    );
    let d64 = Operands {
        q: up(&m, &rand(&[1, 2, 40, 64], 975, 1.0)),
        k: up(&m, &rand(&[1, 2, 90, 64], 976, 1.0)),
        v: up(&m, &rand(&[1, 2, 90, 64], 977, 1.0)),
    };
    assert_eq!(
        ok(
            "d64",
            m.attention_saving_kernel(&d64.q, &d64.k, &d64.v, opts())
        ),
        "scalar"
    );
}
