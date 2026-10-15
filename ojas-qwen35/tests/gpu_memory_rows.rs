//! Peak device bytes of one Qwen3.5 training step on Metal, one arm per
//! process, for `bench/results/`. `#[ignore]` and named `gpu_*`: run it alone,
//! under the machine's heavy-job lock, one process per arm so each peak is
//! that arm's alone (the device figures are process-wide).
//!
//! ```text
//! QWEN35_MEM_ARM=tape-bf16 QWEN35_MEM_TOKENS=2048 \
//! cargo test -p ojas-qwen35 --release --test gpu_memory_rows \
//!     -- --ignored --test-threads=1 --nocapture gpu_memory_row
//! ```
//!
//! Arms (`QWEN35_MEM_ARM`):
//! - `provider-f32`: this crate's provider at `Precision::F32`,
//!   `Numerics::ExactF32`; `provider-bf16`: `Precision::Bf16`,
//!   `Numerics::Bf16Operands`. One forward, backward and AdamW step.
//! - `tape-f32` / `tape-bf16`: ojas's Tape on `MetalBackend`, linear weights
//!   loaded `F32` or `Bf16` (`load_hf`), every layer a checkpointed segment,
//!   one forward and backward. Every parameter takes a gradient: this is a
//!   full-gradient step, not the LoRA step (the Tape has no frozen leaves
//!   yet), so its peak over-counts a LoRA step's by the weight gradients.
//!
//! Every position is scored (the causal loss as letter rows), micro-batch 1.
//! The snapshot is `QWEN35_MEM_SNAPSHOT`, or the one snapshot of
//! `Qwen/Qwen3.5-2B-Base` in the Hugging Face cache. The row is one line
//! starting `ROW ` (JSON). A refusal at any phase is a row too, naming the
//! phase and the error, and the test then fails.

#![cfg(target_os = "macos")]

mod common;

use std::time::Instant;

use ojas_autograd::Tape;
use ojas_core::{Backend, Budget, DType, Tensor};
use ojas_io::SafeTensors;
use ojas_metal::MetalBackend;
use ojas_model::qwen35::{bind, forward_loss, load_hf, Qwen35Tables};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};
use ojas_qwen35::{
    AdamWHyper, GroupSpec, LrRule, Numerics, OptimizerPlan, Precision, Qwen35Step, Select,
    Sequence, Snapshot, WdRule,
};

use common::{refused, snapshot_dir, Row};

const GIB: u64 = 1 << 30;

/// `n` deterministic ids below 150000 (ordinary text tokens), skipping the
/// vision special tokens the config declares.
fn ids(n: usize, reserved: &[u32]) -> Vec<u32> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let id = (s % 150_000) as u32;
        if !reserved.contains(&id) {
            out.push(id);
        }
    }
    out
}

#[test]
#[ignore]
fn gpu_memory_row() {
    let arm = std::env::var("QWEN35_MEM_ARM").expect("QWEN35_MEM_ARM");
    let tokens: usize = std::env::var("QWEN35_MEM_TOKENS")
        .expect("QWEN35_MEM_TOKENS")
        .parse()
        .expect("QWEN35_MEM_TOKENS");
    let dir = snapshot_dir("QWEN35_MEM_SNAPSHOT");
    let snap = Snapshot::from_dir(&dir).unwrap();
    let cfg = snap.config.clone();
    let reserved: Vec<u32> = cfg.reserved_token_ids.iter().map(|(_, id)| *id).collect();
    let ids = ids(tokens, &reserved);
    let mut row = Row::new();
    row.text("arm", &arm);
    row.text("snapshot", &dir.display().to_string());
    row.num("tokens", tokens as u64);
    let started = Instant::now();
    let mut row = match arm.as_str() {
        "provider-f32" => provider(&snap, &ids, Precision::F32, Numerics::ExactF32, row),
        "provider-bf16" => provider(&snap, &ids, Precision::Bf16, Numerics::Bf16Operands, row),
        "tape-f32" => tape(&snap, &ids, DType::F32, row),
        "tape-bf16" => tape(&snap, &ids, DType::Bf16, row),
        other => panic!("unknown QWEN35_MEM_ARM {other}"),
    };
    row.num("wall_ms", started.elapsed().as_millis() as u64);
    row.text("status", "ok");
    row.print();
}

fn provider(snap: &Snapshot, ids: &[u32], p: Precision, n: Numerics, mut r: Row) -> Row {
    let mut s = match Qwen35Step::open(snap, n, p, Budget::new(16 * GIB)) {
        Ok(s) => s,
        Err(e) => refused(r, "open", e),
    };
    let open = s.device_bytes();
    r.num("resident_bytes", open.allocated);
    r.num("open_peak_bytes", open.peak);
    r.phase("open");
    s.reset_peak_device_bytes();
    let rows: Vec<u32> = (0..ids.len() as u32 - 1).collect();
    let k = rows.len();
    let pending = match s.forward(&Sequence {
        ids,
        letter_rows: &rows,
        letter_targets: &ids[1..],
        letter_scale: 1.0 / k as f32,
        span_positions: &[],
    }) {
        Ok(p) => p,
        Err(e) => refused(r, "forward", e),
    };
    let loss = pending.letter_ce_sum() / k as f64;
    let after_forward = s.device_bytes();
    r.num("after_forward_bytes", after_forward.allocated);
    r.num("forward_peak_bytes", after_forward.peak);
    r.phase("forward");
    if let Err(e) = s.backward(pending, None) {
        refused(r, "backward", e);
    }
    let plan = OptimizerPlan::build(
        s.parameter_table(),
        &GroupSpec {
            lr: vec![LrRule {
                label: "all".into(),
                select: Select::All,
                lr_scale: 1.0,
            }],
            weight_decay: vec![WdRule {
                label: "all".into(),
                select: Select::All,
                weight_decay: 0.0,
            }],
        },
    )
    .unwrap();
    let hyper = AdamWHyper {
        lr: 1e-5,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        grad_scale: 1.0,
    };
    if let Err(e) = s.adamw_step(&hyper, &plan) {
        refused(r, "adamw_step", e);
    }
    let end = s.device_bytes();
    r.num("step_peak_bytes", end.peak);
    r.text("loss", &format!("{loss:.6}"));
    r
}

fn tape(snap: &Snapshot, ids: &[u32], linears: DType, mut r: Row) -> Row {
    let cfg = snap.config.clone();
    let spec = cfg.tape_spec();
    let n = ids.len() - 1;
    let m = match MetalBackend::new(Budget::new(56 * GIB)) {
        Ok(m) => m,
        Err(e) => refused(r, "MetalBackend::new", e),
    };
    let host_budget = Budget::new(24 * GIB);
    let device = {
        let file = SafeTensors::open(&snap.weights_path).unwrap();
        let host = match load_hf(&spec, &file, &cfg.tower_prefix, linears, &host_budget) {
            Ok(h) => h,
            Err(e) => refused(r, "load_hf", e),
        };
        let bytes = |want: DType| -> u64 {
            host.values()
                .iter()
                .filter(|t| t.dtype() == want)
                .map(|t| (t.num_elements().unwrap() * want.size()) as u64)
                .sum()
        };
        r.num("bf16_param_bytes", bytes(DType::Bf16));
        r.num("f32_param_bytes", bytes(DType::F32));
        r.num(
            "tied_embed_f32_bytes",
            (host.embed.num_elements().unwrap() * 4) as u64,
        );
        match host.try_map(&mut |t| m.upload(t)) {
            Ok(d) => d,
            Err(e) => refused(r, "upload", e),
        }
    };
    if let Err(e) = m.sync() {
        refused(r, "sync weights", e);
    }
    let resident = m.memory_and_reset_peak().unwrap();
    r.num("resident_bytes", resident.allocated);
    r.num("budget_resident_bytes", m.budget().live_bytes().unwrap());
    let tables = Qwen35Tables::new(&spec, n, &host_budget)
        .unwrap()
        .upload(&m)
        .unwrap();
    let ids_t = Tensor::from_u32(&ids[..n], &[1, n], &host_budget).unwrap();
    let targets = Tensor::from_u32(&ids[1..], &[1, n], &host_budget).unwrap();
    let mut tape = Tape::new(m.clone());
    let vars = bind(&mut tape, &device).unwrap();
    drop(device);
    let loss = match forward_loss(
        &mut tape,
        &spec,
        &vars,
        &ids_t,
        &targets,
        &tables,
        None,
        DEFAULT_CE_CHUNK,
        ActivationCheckpoint::Blocks,
    ) {
        Ok(l) => l,
        Err(e) => refused(r, "forward", e),
    };
    if let Err(e) = m.sync() {
        refused(r, "sync forward", e);
    }
    let after_forward = m.memory().unwrap();
    if let Err(e) = tape.backward(loss) {
        refused(r, "backward", e);
    }
    if let Err(e) = m.sync() {
        refused(r, "sync backward", e);
    }
    let end = m.memory().unwrap();
    let got = tape
        .value(loss)
        .unwrap()
        .to_host(&host_budget)
        .unwrap()
        .to_f32_vec()
        .unwrap()[0];
    r.num("after_forward_bytes", after_forward.allocated);
    r.num("step_peak_bytes", end.peak_allocated);
    r.num("working_set_bytes", resident.recommended_working_set);
    r.text("loss", &format!("{got:.6}"));
    drop(tape);
    r
}
