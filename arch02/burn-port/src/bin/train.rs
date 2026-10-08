#![recursion_limit = "256"]
//! Full training loop for arch_02 on Apple Silicon (Metal via CubeCL MSL).
//!
//! Usage:
//!   cargo run --release --bin train -- \
//!     --data-dir ../data --token-bytes token_bytes.json \
//!     [--iters 20000] [--warmdown 3500] [--micro-batch 4] [--grad-accum 96] \
//!     [--seq-len 2048] [--eval-every 500] [--log-every 20] \
//!     [--profile-every 0] [--out out] [--bench]
//!
//! Parity vs python: 786432 tokens/step ⇒ micro_batch * grad_accum = 384.
//!
//! Backend: `Autodiff<Metal>` (CubeCL MSL). Build with `--features cpu-smoke`
//! to swap in the NdArray CPU backend for a GPU-less smoke test.

use std::path::PathBuf;

use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;

use arch02_burn::bpb::{eval_sliding, TokenByteLut};
use arch02_burn::config::{ModelConfig, TrainConfig};
use arch02_burn::data::{load_shard, PrefetchLoader};
use arch02_burn::ema::ema_update;
use arch02_burn::log::{device_report, mem_rss_bytes, MetricsLogger, Phase, Profiler, StepMetrics};
use arch02_burn::model::Gpt;
use arch02_burn::optim::clip::{grad_sq_norm, scale_grads, GradGroup};
use arch02_burn::optim::muon::{BankStacks, BankedMuon, MomentumHandle, MuonBankIds};

#[cfg(not(feature = "cpu-smoke"))]
type Inner = burn::backend::Metal;
#[cfg(feature = "cpu-smoke")]
type Inner = burn::backend::NdArray;

type B = burn::backend::Autodiff<Inner>;

// Concrete device type per backend (avoids naming the `Backend::Device`
// associated type, which the bin's import graph resolves awkwardly).
#[cfg(not(feature = "cpu-smoke"))]
type InnerDevice = burn::backend::wgpu::WgpuDevice;
#[cfg(feature = "cpu-smoke")]
type InnerDevice = burn::backend::ndarray::NdArrayDevice;

/// Flat id groups used for gradient splitting + global clip.
struct ParamGroups {
    muon: GradGroup,   // the 6 block matrices (2D) — also fed to BankedMuon
    embed: GradGroup,  // tied token / bigram / VE tables (2D)
    scalar: GradGroup, // everything else (1D scalars + small 2D projs)
    muon_ids: MuonBankIds,
}

fn collect_groups(model: &Gpt<B>) -> ParamGroups {
    let mut muon_2d = Vec::new();
    let mut scalar_1d = Vec::new();
    let mut scalar_2d = Vec::new();

    for b in &model.blocks {
        muon_2d.extend([
            b.attn.q_w.id,
            b.attn.k_w.id,
            b.attn.v_w.id,
            b.attn.out_w.id,
            b.mlp.up_w.id,
            b.mlp.down_w.id,
        ]);
        scalar_1d.extend([b.attn.q_gain.id, b.attn.vr_lambda.id, b.attn_scale.id, b.mlp_scale.id]);
        scalar_2d.push(b.resid_mix.id);
    }
    scalar_2d.extend([model.skip_weights.id, model.bigram_proj.id, model.ve_proj.id]);
    scalar_1d.extend([
        model.smear_gate.id,
        model.bigram_scale.id,
        model.ve_scale.id,
        model.ve_layer_scales.id,
    ]);
    let embed_2d = vec![model.tok_emb.id, model.bigram_embed.id, model.ve_embed.id];

    ParamGroups {
        muon: GradGroup { ids_1d: vec![], ids_2d: muon_2d },
        embed: GradGroup { ids_1d: vec![], ids_2d: embed_2d },
        scalar: GradGroup { ids_1d: scalar_1d, ids_2d: scalar_2d },
        muon_ids: MuonBankIds::from_model(model),
    }
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}
fn has_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

fn find_val_shard(data_dir: &PathBuf) -> Option<Vec<u16>> {
    std::fs::read_dir(data_dir).ok().and_then(|rd| {
        let mut v: Vec<_> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().contains("val"))
            .collect();
        v.sort();
        v.first().cloned()
    })
    .map(|p| load_shard(&p).expect("failed to load val shard"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = PathBuf::from(arg(&args, "--data-dir").expect("--data-dir required"));
    let out_dir = PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "out".into()));
    std::fs::create_dir_all(&out_dir).ok();

    // --preset sprint (default): full 11L/512d sprint config.
    // --preset sota: the 4L/128d `sota` toy preset — the exact config the
    //   recorded 3070 Ti ladder numbers (sliding BPB 1.9902) were measured on.
    let preset = arg(&args, "--preset").unwrap_or_else(|| "sprint".into());
    let (mcfg, mut tcfg) = match preset.as_str() {
        "sprint" => (ModelConfig::default(), TrainConfig::default()),
        "sota" => (ModelConfig::sota_toy(), TrainConfig::sota_toy()),
        other => panic!("unknown --preset {other:?} (use sprint | sota)"),
    };
    if let Some(v) = arg(&args, "--iters") { tcfg.iterations = v.parse().unwrap(); }
    if let Some(v) = arg(&args, "--warmdown") { tcfg.warmdown_iters = v.parse().unwrap(); }
    if let Some(v) = arg(&args, "--micro-batch") { tcfg.micro_batch = v.parse().unwrap(); }
    if let Some(v) = arg(&args, "--grad-accum") { tcfg.grad_accum = v.parse().unwrap(); }
    if let Some(v) = arg(&args, "--seq-len") { tcfg.seq_len = v.parse().unwrap(); }
    if let Some(v) = arg(&args, "--eval-every") { tcfg.eval_every = v.parse().unwrap(); }
    if let Some(v) = arg(&args, "--eval-batch") { tcfg.eval_batch = v.parse().unwrap(); }
    if let Some(v) = arg(&args, "--log-every") { tcfg.log_every = v.parse().unwrap(); }
    if let Some(v) = arg(&args, "--profile-every") { tcfg.profile_every = v.parse().unwrap(); }
    if let Some(v) = arg(&args, "--seed") { tcfg.seed = v.parse().unwrap(); }

    let device: InnerDevice = Default::default();
    B::seed(&device, tcfg.seed);

    println!("{}", device_report::<Inner>(&device));
    println!(
        "arch_02 value-residual GPT [{preset}] | {}L x {}d, {} steps, {}x{} seqs x {} = {} tok/step",
        mcfg.num_layers, mcfg.model_dim, tcfg.iterations, tcfg.micro_batch, tcfg.grad_accum,
        tcfg.seq_len, tcfg.micro_batch * tcfg.grad_accum * tcfg.seq_len,
    );

    if has_flag(&args, "--bench") {
        let bench_seqs = arg(&args, "--bench-seqs")
            .map(|v| v.parse().unwrap())
            .unwrap_or(32);
        run_bench(&mcfg, &tcfg, &device, &data_dir, bench_seqs);
        return;
    }

    let lut = arg(&args, "--token-bytes")
        .map(|p| TokenByteLut::load(&PathBuf::from(p)).expect("failed to load token-bytes LUT"));

    let mut model: Gpt<B> = Gpt::new(&mcfg, &device);
    let mut ema: Gpt<Inner> = model.valid();
    let groups = collect_groups(&model);

    // Optimizers: BankedMuon (batched NS5) for the block matrices; AdamW for
    // embeddings and scalars.
    let momentum = MomentumHandle::new(tcfg.muon_momentum_start);
    let mut banked = BankedMuon::<Inner>::new(momentum.clone(), tcfg.weight_decay, 5, 1e-7);
    let adamw = || {
        AdamWConfig::new()
            .with_beta_1(tcfg.adam_beta1 as f32)
            .with_beta_2(tcfg.adam_beta2 as f32)
            .with_epsilon(tcfg.adam_eps as f32)
            .with_weight_decay(tcfg.weight_decay as f32)
    };
    let mut optim_embed = adamw().init::<B, Gpt<B>>();
    let mut optim_scalar = adamw().init::<B, Gpt<B>>();

    // Prefetching data loader (host prep overlapped with GPU compute).
    let loader = PrefetchLoader::new(
        &data_dir,
        "train",
        tcfg.micro_batch,
        tcfg.seq_len,
        mcfg.bigram_vocab_size,
        tcfg.prefetch_depth,
    )
    .expect("failed to open training shards");
    let val_tokens = find_val_shard(&data_dir);

    let mut logger = MetricsLogger::new(&out_dir);
    let grad_scale = 1.0 / tcfg.grad_accum as f64;
    let toks_per_step = tcfg.micro_batch * tcfg.grad_accum * tcfg.seq_len;
    let mut last_log = std::time::Instant::now();
    let mut steps_since_log = 0usize;

    for step in 0..tcfg.iterations {
        momentum.set(tcfg.muon_momentum(step));
        let mul = tcfg.lr_mul(step);
        let is_log = step % tcfg.log_every == 0;
        let profiled = tcfg.profile_every > 0 && step % tcfg.profile_every == 0
            || step < tcfg.profile_first;

        let mut prof = Profiler::<Inner>::new(device.clone(), profiled);

        let mut acc_muon = GradientsAccumulator::<Gpt<B>>::new();
        let mut acc_embed = GradientsAccumulator::<Gpt<B>>::new();
        let mut acc_scalar = GradientsAccumulator::<Gpt<B>>::new();
        let mut loss_acc: Option<Tensor<Inner, 1>> = None;

        for _ in 0..tcfg.grad_accum {
            prof.enter(Phase::DataPrep);
            let hb = loader.next();

            prof.enter(Phase::Upload);
            let shape = [tcfg.micro_batch, tcfg.seq_len];
            let xt = Tensor::<B, 1, Int>::from_ints(hb.x.as_slice(), &device).reshape(shape);
            let yt = Tensor::<B, 1, Int>::from_ints(hb.y.as_slice(), &device).reshape(shape);
            let bt = Tensor::<B, 1, Int>::from_ints(hb.bigram.as_slice(), &device).reshape(shape);

            prof.enter(Phase::Forward);
            let loss = model.loss(xt, bt, yt);
            // Accumulate loss on-device; single readback at log cadence.
            let li = loss.clone().inner();
            loss_acc = Some(match loss_acc {
                Some(a) => a + li,
                None => li,
            });

            prof.enter(Phase::Backward);
            let mut grads = (loss * grad_scale).backward();

            prof.enter(Phase::GradSplit);
            let g_muon = GradientsParams::from_params(&mut grads, &model, &groups.muon.ids_2d);
            let g_embed = GradientsParams::from_params(&mut grads, &model, &groups.embed.ids_2d);
            let g_rest = GradientsParams::from_grads(grads, &model);
            acc_muon.accumulate(&model, g_muon);
            acc_embed.accumulate(&model, g_embed);
            acc_scalar.accumulate(&model, g_rest);
        }

        let mut g_muon = acc_muon.grads();
        let mut g_embed = acc_embed.grads();
        let mut g_scalar = acc_scalar.grads();

        prof.enter(Phase::Clip);
        // Global grad clip, fully on-device. The Muon grads are stacked into
        // their four banks FIRST, so both the norm reduction and the clip
        // scaling touch 4 big tensors instead of 66 small ones (the per-tensor
        // version cost ~1.5 s/step in phase profiles on Metal).
        let mut stacks = BankStacks::from_grads::<B>(&mut g_muon, &groups.muon_ids);
        let muon_sq = stacks.sq_norm();
        let embed_sq = grad_sq_norm::<Inner>(&g_embed, &groups.embed);
        let scalar_sq = grad_sq_norm::<Inner>(&g_scalar, &groups.scalar);
        let total_sq = Tensor::cat(vec![muon_sq.clone(), embed_sq.clone(), scalar_sq.clone()], 0).sum();
        let global_norm = total_sq.sqrt().into_scalar().elem::<f64>(); // sole readback
        let mut clip_factor = 1.0;
        if global_norm > tcfg.grad_clip && global_norm > 0.0 {
            clip_factor = tcfg.grad_clip / global_norm;
            let f = Tensor::<Inner, 1>::from_floats([clip_factor as f32], &device);
            stacks.scale(f.clone());
            scale_grads::<Inner>(&mut g_embed, &groups.embed, f.clone());
            scale_grads::<Inner>(&mut g_scalar, &groups.scalar, f);
        }
        let group_norms = if is_log {
            Some((
                muon_sq.sqrt().into_scalar().elem::<f64>(),
                embed_sq.sqrt().into_scalar().elem::<f64>(),
                scalar_sq.sqrt().into_scalar().elem::<f64>(),
            ))
        } else {
            None
        };

        prof.enter(Phase::AdamW);
        model = optim_embed.step(tcfg.tied_embed_lr * mul, model, g_embed);
        model = optim_scalar.step(tcfg.scalar_lr * mul, model, g_scalar);

        prof.enter(Phase::Muon);
        let (m2, muon_prof) =
            banked.step(tcfg.matrix_lr * mul, model, stacks, &device, profiled);
        model = m2;

        prof.enter(Phase::Ema);
        let valid_model = model.valid();
        ema_update(&mut ema, &valid_model, tcfg.ema_decay);
        prof.finish();

        steps_since_log += 1;

        if is_log {
            let loss_val = loss_acc
                .map(|t| t.into_scalar().elem::<f64>() / (tcfg.grad_accum as f64))
                .unwrap_or(f64::NAN);
            let elapsed = last_log.elapsed().as_secs_f64();
            let step_ms = elapsed * 1e3 / steps_since_log as f64;
            let tps = toks_per_step as f64 * steps_since_log as f64 / elapsed.max(1e-9);
            let gn = group_norms;
            let m = StepMetrics {
                step,
                loss: loss_val,
                grad_norm_global: global_norm,
                grad_norm_muon: gn.map(|v| v.0).unwrap_or(0.0),
                grad_norm_embed: gn.map(|v| v.1).unwrap_or(0.0),
                grad_norm_scalar: gn.map(|v| v.2).unwrap_or(0.0),
                clip_factor,
                lr_mul: mul,
                momentum: momentum.get(),
                tokens_per_s: tps,
                step_ms,
                profiled,
                phase_ms: prof.pairs(),
                muon_ns5_ms: if profiled { muon_prof.pairs() } else { vec![] },
                rss_mb: mem_rss_bytes() as f64 / (1024.0 * 1024.0),
            };
            logger.console(&m);
            logger.log(&m);
            last_log = std::time::Instant::now();
            steps_since_log = 0;
        }

        if (step + 1) % tcfg.eval_every == 0 {
            if let (Some(lut), Some(val)) = (&lut, &val_tokens) {
                let n = if tcfg.val_cap > 0 { val.len().min(tcfg.val_cap + 1) } else { val.len() };
                let t_eval = std::time::Instant::now();
                let bpb = eval_sliding(
                    &valid_model, &val[..n], tcfg.seq_len, tcfg.eval_stride, lut,
                    mcfg.bigram_vocab_size, tcfg.eval_batch, &device,
                );
                println!(
                    "step {:5} | val BPB (current weights) = {bpb:.4} | eval {:.1}s over {} tok",
                    step + 1, t_eval.elapsed().as_secs_f64(), n,
                );
            }
        }
    }

    println!("saving checkpoints to {out_dir:?}");
    model
        .clone()
        .save_file(out_dir.join("model_final"), &CompactRecorder::new())
        .expect("save model");
    ema.clone()
        .save_file(out_dir.join("model_ema"), &CompactRecorder::new())
        .expect("save ema");

    if let (Some(lut), Some(val)) = (&lut, &val_tokens) {
        let n = if tcfg.val_cap > 0 { val.len().min(tcfg.val_cap + 1) } else { val.len() };
        println!("running sliding-window eval on EMA weights ({n} tok)...");
        let bpb = eval_sliding(
            &ema, &val[..n], tcfg.seq_len, tcfg.eval_stride, lut,
            mcfg.bigram_vocab_size, tcfg.eval_batch, &device,
        );
        println!("FINAL val BPB (EMA) = {bpb:.4}");
    }
}

/// Throughput sweep: for each micro_batch in {4,8,16,32} (grad_accum chosen to
/// keep `bench_seqs` seqs/step), run a few timed steps and report tok/s +
/// ms/step so the M5 default can be picked empirically. `bench_seqs` is kept
/// small (default 32) so a sweep finishes in minutes; scale tok/s linearly to
/// the parity config (384 seqs/step) — optimizer cost is per-step constant.
fn run_bench(
    mcfg: &ModelConfig,
    tcfg: &TrainConfig,
    device: &InnerDevice,
    data_dir: &PathBuf,
    bench_seqs: usize,
) {
    println!("== throughput bench (forward+backward+opt), {bench_seqs} seqs/step ==");
    let target_seqs = bench_seqs;
    let warmup = 1usize;
    let timed = 3usize;

    for &mb in &[4usize, 8, 16, 32] {
        let ga = (target_seqs / mb).max(1);
        let mut model: Gpt<B> = Gpt::new(mcfg, device);
        let groups = collect_groups(&model);
        let momentum = MomentumHandle::new(tcfg.muon_momentum_start);
        let mut banked = BankedMuon::<Inner>::new(momentum.clone(), tcfg.weight_decay, 5, 1e-7);
        let adamw = || {
            AdamWConfig::new()
                .with_beta_1(tcfg.adam_beta1 as f32)
                .with_beta_2(tcfg.adam_beta2 as f32)
                .with_epsilon(tcfg.adam_eps as f32)
                .with_weight_decay(tcfg.weight_decay as f32)
        };
        let mut optim_embed = adamw().init::<B, Gpt<B>>();
        let mut optim_scalar = adamw().init::<B, Gpt<B>>();
        let loader = match PrefetchLoader::new(
            data_dir, "train", mb, tcfg.seq_len, mcfg.bigram_vocab_size, tcfg.prefetch_depth,
        ) {
            Ok(l) => l,
            Err(e) => {
                println!("  mb={mb:>2}: skipped ({e})");
                continue;
            }
        };
        let grad_scale = 1.0 / ga as f64;
        let toks = mb * ga * tcfg.seq_len;

        let mut t_start = std::time::Instant::now();
        for step in 0..(warmup + timed) {
            if step == warmup {
                let _ = Inner::sync(device);
                t_start = std::time::Instant::now();
            }
            let mut acc_muon = GradientsAccumulator::<Gpt<B>>::new();
            let mut acc_embed = GradientsAccumulator::<Gpt<B>>::new();
            let mut acc_scalar = GradientsAccumulator::<Gpt<B>>::new();
            for _ in 0..ga {
                let hb = loader.next();
                let shape = [mb, tcfg.seq_len];
                let xt = Tensor::<B, 1, Int>::from_ints(hb.x.as_slice(), device).reshape(shape);
                let yt = Tensor::<B, 1, Int>::from_ints(hb.y.as_slice(), device).reshape(shape);
                let bt = Tensor::<B, 1, Int>::from_ints(hb.bigram.as_slice(), device).reshape(shape);
                let loss = model.loss(xt, bt, yt);
                let mut grads = (loss * grad_scale).backward();
                let g_muon = GradientsParams::from_params(&mut grads, &model, &groups.muon.ids_2d);
                let g_embed = GradientsParams::from_params(&mut grads, &model, &groups.embed.ids_2d);
                let g_rest = GradientsParams::from_grads(grads, &model);
                acc_muon.accumulate(&model, g_muon);
                acc_embed.accumulate(&model, g_embed);
                acc_scalar.accumulate(&model, g_rest);
            }
            let mut g_muon = acc_muon.grads();
            let g_embed = acc_embed.grads();
            let g_scalar = acc_scalar.grads();
            let stacks = BankStacks::from_grads::<B>(&mut g_muon, &groups.muon_ids);
            model = optim_embed.step(tcfg.matrix_lr, model, g_embed);
            model = optim_scalar.step(tcfg.scalar_lr, model, g_scalar);
            let (m2, _) = banked.step(tcfg.matrix_lr, model, stacks, device, false);
            model = m2;
        }
        let _ = Inner::sync(device);
        let elapsed = t_start.elapsed().as_secs_f64();
        let ms = elapsed * 1e3 / timed as f64;
        let tps = toks as f64 * timed as f64 / elapsed;
        let rss = mem_rss_bytes() as f64 / (1024.0 * 1024.0);
        println!(
            "  mb={mb:>2} x ga={ga:>3} = {:>4} seqs | {ms:>8.1} ms/step | {tps:>10.0} tok/s | rss {rss:.0} MB",
            mb * ga
        );
    }
}
