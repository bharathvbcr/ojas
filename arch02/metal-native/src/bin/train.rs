//! Phase 4 training harness: FineWeb sota / medium_16m run, --tok-mult bench, bf16/async.
//!
//! Example (0.78M sota toy):
//!   cargo run --release --bin train -- \
//!     --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \
//!     --token-bytes ../burn-port/token_bytes.json \
//!     --out out/sota_seed1337 --iters 3000 --seed 1337
//!
//! Example (~16.4M Soft path):
//!   METAL_NATIVE_FA_TILED=1 cargo run --release --bin train -- \
//!     --preset 16m --bench --bench-steps 20 --f32 --clip-soft \
//!     --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \
//!     --out out/bench_16m_fa_tiled

use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tessl_arch02::bpb::{eval_sliding, TokenByteLut};
use tessl_arch02::checkpoint::{
    collect_divergence_norms_device, load_optim_state_python_npy, read_training_checkpoint_meta,
    save_optim_state_python_npy, save_training_checkpoint, save_weights_python_npy,
    TrainingCheckpointMeta, CHECKPOINT_VERSION,
};
use tessl_arch02::data::{load_shard, PrefetchLoader};
use tessl_arch02::init::{fineweb_token_skip, init_weights_seeded};
use tessl_arch02::log::{
    mem_current_physical_mb, mem_rss_mb, MetricsLogger, Phase, Profiler, StepMetrics,
};
use tessl_arch02::model_bwd::{backward_f32_opts_clip, Grads};
use tessl_arch02::model_fwd::{forward_f32_uploaded, DualInputBuffers};
use tessl_arch02::optim::{
    copy_ema_into_weights, optim_step, zero_grads, ClipMode, LrSchedule, OptimHyperparams,
    OptimState,
};
use tessl_arch02::parity::golden_dir;
use tessl_arch02::research::{capture_weight_snapshot, collect_research_telemetry};
use tessl_arch02::runtime::PrecisionMode;
use tessl_arch02::tape::Tape;
use tessl_arch02::weights::{ModelConfig, Weights};
use tessl_arch02::OptimizerKind;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn has_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

fn find_val_shard(data_dir: &Path) -> Option<Vec<u16>> {
    std::fs::read_dir(data_dir)
        .ok()
        .and_then(|rd| {
            let mut v: Vec<_> = rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.to_string_lossy().contains("val"))
                .collect();
            v.sort();
            v.first().cloned()
        })
        .and_then(|p| load_shard(&p).ok())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();

    let data_dir = arg(&args, "--data-dir").map(PathBuf::from);
    let out_dir = PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "out/metal_native".into()));
    let golden = env::var("GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| golden_dir());

    let resume_root = arg(&args, "--resume").map(PathBuf::from);
    let resume_meta = resume_root
        .as_ref()
        .map(|p| read_training_checkpoint_meta(p))
        .transpose()?;
    let start_step: usize = resume_meta
        .as_ref()
        .map(|m| m.step)
        .or_else(|| arg(&args, "--start-step").and_then(|s| s.parse().ok()))
        .unwrap_or(0);
    let requested_total_steps: Option<usize> = arg(&args, "--total-steps")
        .or_else(|| arg(&args, "--total-iters"))
        .and_then(|s| s.parse().ok());
    let iters: usize = if let Some(total) = requested_total_steps {
        total.checked_sub(start_step).ok_or_else(|| {
            format!("--total-steps {total} is before checkpoint/start step {start_step}")
        })?
    } else {
        arg(&args, "--iters")
            .and_then(|s| s.parse().ok())
            .unwrap_or(3000)
    };
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .or_else(|| resume_meta.as_ref().map(|m| m.seed))
        .unwrap_or(1337);
    let tok_mult: usize = arg(&args, "--tok-mult")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    let log_every: usize = arg(&args, "--log-every")
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let eval_every: usize = arg(&args, "--eval-every")
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let checkpoint_every: usize = arg(&args, "--checkpoint-every")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    // LR schedule: classic `--warmdown N` = linear 1→0 over final N steps (0 = off).
    // Long Soft (100k): add `--warmdown-start` / `--lr-floor` / `--final-warmdown`.
    let warmdown_iters: usize = arg(&args, "--warmdown")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let warmdown_start: Option<usize> = arg(&args, "--warmdown-start").and_then(|s| s.parse().ok());
    let lr_floor: f32 = arg(&args, "--lr-floor")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let final_warmdown: usize = arg(&args, "--final-warmdown")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    // Dump live weights + optim state after this logged step (0-indexed loop index).
    // Example: `--dump-at 2000` → `out/.../dump_step2000/{weights,optim}/`.
    let dump_at: Option<usize> = arg(&args, "--dump-at").and_then(|s| s.parse().ok());
    // Legacy partial dump inputs remain supported; --resume is the exact path.
    let load_weights = resume_root
        .as_ref()
        .map(|p| p.join("weights"))
        .or_else(|| arg(&args, "--load-weights").map(PathBuf::from));
    let load_optim = arg(&args, "--load-optim").map(PathBuf::from);
    // Total optimizer steps for schedule (defaults to start_step + segment iters).
    let total_iters = requested_total_steps.unwrap_or_else(|| start_step.saturating_add(iters));
    let schedule_overridden = has_flag(&args, "--warmdown")
        || has_flag(&args, "--warmdown-start")
        || has_flag(&args, "--lr-floor")
        || has_flag(&args, "--final-warmdown")
        || requested_total_steps.is_some();
    let lr_sched = if !schedule_overridden {
        resume_meta
            .as_ref()
            .map(|m| m.schedule)
            .unwrap_or(LrSchedule {
                total_iters,
                warmdown_start,
                warmdown_iters,
                lr_floor,
                final_warmdown,
            })
    } else {
        LrSchedule {
            total_iters,
            warmdown_start,
            warmdown_iters,
            lr_floor,
            final_warmdown,
        }
    };
    // When set, attach scalar/bank/Muon-momentum norms on every log step (default: on).
    let div_telemetry = !has_flag(&args, "--no-div-telemetry");
    let research_manifest = arg(&args, "--research-manifest");
    let research_telemetry = has_flag(&args, "--research-telemetry") || research_manifest.is_some();
    // Funnel jobs need the final EMA evaluation but not hundreds of gigabytes of
    // duplicate weight dumps. Champion runs retain the safe default (save).
    let save_final_weights = !has_flag(&args, "--no-final-weight-save");
    let bench_steps: Option<usize> = arg(&args, "--bench")
        .map(|_| {
            arg(&args, "--bench-steps")
                .and_then(|s| s.parse().ok())
                .unwrap_or(20)
        })
        .or_else(|| {
            if has_flag(&args, "--bench") {
                Some(20)
            } else {
                None
            }
        });
    // --bench with no value
    let bench_steps = if has_flag(&args, "--bench") && bench_steps.is_none() {
        Some(20)
    } else {
        bench_steps
    };
    let precision_arg = arg(&args, "--precision");
    let use_f32 = has_flag(&args, "--f32") || precision_arg.as_deref() == Some("f32");
    if let Some(value) = precision_arg.as_deref() {
        if value != "f32" && value != "bf16" {
            return Err(format!("--precision must be bf16 or f32, got {value}"));
        }
    }
    let use_tf32 = has_flag(&args, "--tf32");
    let use_async = !has_flag(&args, "--sync");
    let clip_mode = if has_flag(&args, "--clip-match") {
        ClipMode::Match
    } else if has_flag(&args, "--clip-python") {
        ClipMode::Python
    } else if let Some(meta) = &resume_meta {
        meta.clip_mode
    } else {
        // Default Soft: Muon×√c, AdamW×c — Soft-on-AdamW explodes past ~3.5k.
        ClipMode::Soft
    };
    let flash_tensorops = has_flag(&args, "--flash-tensorops");
    let pool_cache_mb: Option<usize> = arg(&args, "--pool-cache-mb").and_then(|s| s.parse().ok());
    let wired_fraction: Option<f64> = arg(&args, "--wired-fraction").and_then(|s| s.parse().ok());
    // Force golden/weights_init (seed-1337 export) instead of seeded random init.
    // Also honored via METAL_NATIVE_GOLDEN_INIT=1 / GOLDEN_INIT=1 for parity tests.
    let golden_init = has_flag(&args, "--golden-init")
        || matches!(
            env::var("METAL_NATIVE_GOLDEN_INIT")
                .or_else(|_| env::var("GOLDEN_INIT"))
                .ok()
                .as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes")
        );

    let rt = tessl_arch02::gpu_runtime()?;
    if let Some(mb) = pool_cache_mb {
        rt.set_pool_cache_cap_bytes(mb.saturating_mul(1024 * 1024));
    }
    if let Some(f) = wired_fraction {
        rt.set_wired_fraction(f);
    }
    rt.set_precision(if use_f32 {
        PrecisionMode::F32
    } else {
        PrecisionMode::Bf16
    });
    if use_tf32 {
        if !use_f32 {
            eprintln!(
                "warning: --tf32 applies to f32 TensorOps GEMMs; ignored under default bf16 path \
                 (pass --f32 --tf32 to enable relaxed_precision)"
            );
        }
        rt.set_relaxed_precision(true);
    }
    if use_async {
        rt.set_async_encode(true)?;
    }
    if flash_tensorops {
        rt.set_flash_tensorops(true);
    }

    let preset = arg(&args, "--preset")
        .or_else(|| resume_meta.as_ref().map(|m| m.preset.clone()))
        .unwrap_or_else(|| "sota".into());
    let mut cfg = ModelConfig::from_preset(&preset)?;

    if let Some(m) = arg(&args, "--mixer") {
        cfg.mixer = match m.as_str() {
            "mingru" => tessl_arch02::weights::MixerKind::MinGRU,
            "mamba2" => tessl_arch02::weights::MixerKind::Mamba2,
            "attention" => tessl_arch02::weights::MixerKind::Attention,
            _ => return Err(format!("Unknown mixer {}", m)),
        };
    }
    if has_flag(&args, "--value-residual") {
        cfg.value_residual = true;
    }
    if let Some(spec) = arg(&args, "--layer-mixers") {
        if spec.trim().is_empty() {
            cfg.layer_mixers = tessl_arch02::weights::default_hybrid_pattern(cfg.num_layers);
        } else {
            let kinds: Result<Vec<_>, _> = spec
                .split(',')
                .map(tessl_arch02::weights::MixerKind::parse)
                .collect();
            cfg.layer_mixers = tessl_arch02::weights::expand_layer_mixers(&kinds?, cfg.num_layers);
        }
        eprintln!(
            "hybrid layer_mixers (L={}): {:?}",
            cfg.num_layers,
            cfg.resolved_layer_mixers()
        );
    }

    let optimizer_kind: OptimizerKind = arg(&args, "--optimizer")
        .or_else(|| resume_meta.as_ref().map(|m| m.optimizer.clone()))
        .unwrap_or_else(|| OptimizerKind::default().to_string())
        .parse()?;
    if !optimizer_kind.native_ready() {
        return Err(format!(
            "optimizer {optimizer_kind} is registered but failed its native systems/parity gate; no fallback is allowed"
        ));
    }
    if let Some(b) = arg(&args, "--batch").and_then(|s| s.parse().ok()) {
        cfg.batch = b;
    }
    if let Some(t) = arg(&args, "--seq-len").and_then(|s| s.parse().ok()) {
        cfg.seq_len = t;
    }
    cfg.validate_metal_shape()?;
    let base_batch = cfg.batch;
    cfg.batch = base_batch * tok_mult;
    let seq_len = cfg.seq_len;
    let toks_per_step = cfg.batch * seq_len;
    let n_params = cfg.count_params();
    if let Some(meta) = &resume_meta {
        if meta.parameter_count != n_params {
            return Err(format!(
                "checkpoint parameter count {} does not match preset {} ({n_params})",
                meta.parameter_count, preset
            ));
        }
        if meta.config != cfg {
            return Err(format!(
                "exact resume requires the checkpoint shape unchanged; checkpoint={:?}, requested={:?}",
                meta.config, cfg
            ));
        }
    }

    let mem = rt.memory_info();
    eprintln!(
        "metal-native | device={} tensorops={} encode=Metal4 precision={:?} \
         relaxed_f32={} async={}",
        rt.device_name(),
        rt.has_tensorops(),
        rt.precision(),
        rt.relaxed_precision(),
        use_async
    );
    eprintln!(
        "memory | recommended_ws={:.1} GiB system={:.1} GiB wired_budget={:.1} GiB \
         pool_cache_cap={:.1} GiB (raise iogpu.wired_limit_mb via sysctl if needed)",
        mem.recommended_working_set as f64 / (1024.0 * 1024.0 * 1024.0),
        mem.memory_size as f64 / (1024.0 * 1024.0 * 1024.0),
        mem.wired_budget as f64 / (1024.0 * 1024.0 * 1024.0),
        mem.pool_cache_cap as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    eprintln!(
        "preset={preset} | L={} C={} H={}/{} hd={} mlp={} V={} | params={:.3}M ({n_params})",
        cfg.num_layers,
        cfg.model_dim,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.mlp_dim,
        cfg.vocab_size,
        n_params as f64 / 1e6,
    );
    eprintln!("optimizer={optimizer_kind} | native_parity_ready=true");
    eprintln!(
        "shape | B={} (base={base_batch}×tok-mult={tok_mult}) T={seq_len} → {toks_per_step} tok/step | iters={iters}",
        cfg.batch
    );
    {
        let (wd_start, wd_len) = lr_sched.main_window();
        let sched_on = wd_len > 0 || final_warmdown > 0;
        if sched_on {
            eprintln!(
                "schedule | total_iters={total_iters} warmdown_start={wd_start} \
                 warmdown={wd_len} lr_floor={lr_floor} final_warmdown={final_warmdown} \
                 (main linear 1→{lr_floor} over [{wd_start}, {}); hold; final→0)",
                wd_start.saturating_add(wd_len)
            );
            // Last-10% warmdown on ≥100k Soft is too late (failed arm: explode ~21–50k
            // under constant LR; warmdown@90k never reached). Nudge toward WSD recipe.
            if total_iters >= 100_000
                && warmdown_start.is_none()
                && wd_start >= total_iters.saturating_sub(total_iters / 5)
            {
                eprintln!(
                    "warning: warmdown starts at step {wd_start} (≥80% of {total_iters}); \
                     Soft-split 100k needs earlier decay — recommend \
                     --warmdown-start 16000 --warmdown 24000 --lr-floor 0.1 \
                     --final-warmdown 10000 (DECISIONS M11)"
                );
            }
        } else if total_iters >= 100_000 {
            eprintln!(
                "warning: total_iters={total_iters} with no LR warmdown; Soft-split 100k \
                 under constant LR explodes ~21k — recommend --warmdown-start 16000 \
                 --warmdown 24000 --lr-floor 0.1 --final-warmdown 10000 (DECISIONS M11)"
            );
        } else if total_iters >= 20_000 {
            // Keep warmdown default 0 for 3k sota toy; nudge long Soft runs toward the
            // validated 20k recipe (--warmdown 3500). See README / DECISIONS M11.
            eprintln!(
                "warning: total_iters={total_iters} with --warmdown 0; for 20k Soft FA_TILED \
                 recommend --warmdown 3500 (DECISIONS M11)"
            );
        }
    }
    eprintln!(
        "flash: FA-2 tiled online-softmax + L tape (bf16-input/f32-accum under \
         PrecisionMode::Bf16; TensorOps multi-block probe not default — DECISIONS M8){}",
        if tessl_arch02::ab_flags::fa_blocksoft() {
            "; FA_BLOCKSOFT=1 (blockwise rowmax+rescale Soft quality probe)"
        } else {
            ""
        }
    );
    eprintln!("golden={}", golden.display());
    eprintln!(
        "seed={seed} | init={} | data_skip={}",
        if load_weights.is_some() {
            "load-weights"
        } else if golden_init {
            "golden"
        } else {
            "seeded-random"
        },
        if data_dir.is_some() {
            fineweb_token_skip(seed)
        } else {
            0
        }
    );
    if let Some(ref p) = load_weights {
        eprintln!("load-weights={} start-step={start_step}", p.display());
    }
    if let Some(n) = dump_at {
        eprintln!(
            "dump-at={n} → {}/dump_step{n}/{{weights,optim}} (Python .npy / optim_step3 layout)",
            out_dir.display()
        );
    }
    if div_telemetry {
        eprintln!("divergence telemetry: on (scalar/bank/Muon norms every --log-every step)");
    }

    if golden_init && preset != "sota" && preset != "sota_toy" && preset != "toy" {
        return Err(format!(
            "--golden-init only matches sota-toy golden banks; preset={preset} needs seeded init \
             (omit --golden-init) or a matching dump via --load-weights"
        ));
    }
    let mut w = if let Some(ref wp) = load_weights {
        Weights::load_from_python_npy(&rt, wp, cfg)?
    } else if golden_init {
        eprintln!(
            "init: loading golden weights_init (seed-agnostic banks; pass without --golden-init for seeded init)"
        );
        Weights::load_from_golden(&rt, &golden, cfg)?
    } else {
        eprintln!(
            "init: seeded random (orthogonal banks + normal embeds; seed={seed}; not torch-Philox bit-identical)"
        );
        init_weights_seeded(&rt, cfg, seed)?
    };
    if rt.precision() == PrecisionMode::Bf16 {
        w.ensure_bf16_banks(&rt)?;
        eprintln!("bf16: persistent weight banks enabled (qo/kv/mlp/ve/bigram)");
    }
    let mut hp = resume_meta
        .as_ref()
        .map(|m| m.hyperparams.clone())
        .unwrap_or_else(OptimHyperparams::default);
    if resume_meta.is_none() && w.cfg.model_dim == 768 && w.cfg.num_layers == 24 {
        hp.muon_momentum_warmup = 150;
    }
    macro_rules! hp_f32 {
        ($flag:literal, $field:ident) => {
            if let Some(value) = arg(&args, $flag) {
                hp.$field = value
                    .parse::<f32>()
                    .map_err(|_| format!("{} expects a float, got {value}", $flag))?;
            }
        };
    }
    hp_f32!("--matrix-lr", matrix_lr);
    hp_f32!("--embed-lr", tied_embed_lr);
    hp_f32!("--scalar-lr", scalar_lr);
    hp_f32!("--weight-decay", weight_decay);
    hp_f32!("--adam-beta1", adam_beta1);
    hp_f32!("--adam-beta2", adam_beta2);
    hp_f32!("--adam-eps", adam_eps);
    hp_f32!("--muon-momentum-start", muon_momentum_start);
    hp_f32!("--muon-momentum-end", muon_momentum_end);
    hp_f32!("--grad-clip", grad_clip);
    hp_f32!("--ema-decay", ema_decay);
    if let Some(value) = arg(&args, "--muon-momentum-warmup") {
        hp.muon_momentum_warmup = value
            .parse::<usize>()
            .map_err(|_| format!("--muon-momentum-warmup expects an integer, got {value}"))?;
    }
    let mut state = OptimState::new_for_kind(&rt, &w, hp, optimizer_kind)?;
    state.clip_mode = clip_mode;
    eprintln!(
        "clip: {:?} (Match=AdamW+Muon×c; Soft=Muon×√c + AdamW×c; Python=c=1 after grad scale)",
        state.clip_mode
    );
    if let (Some(root), Some(meta)) = (&resume_root, &resume_meta) {
        load_optim_state_python_npy(&mut state, &root.join("optim"), &root.join("ema"))?;
        if state.step != meta.step {
            return Err(format!(
                "checkpoint optimizer step {} != checkpoint metadata step {}",
                state.step, meta.step
            ));
        }
        eprintln!(
            "resume={} | exact state step={} data_cursor_tokens={} (Adam+Muon+EMA)",
            root.display(),
            meta.step,
            meta.data_cursor_tokens
        );
    } else if let Some(ref op) = load_optim {
        state.step = start_step;
        tessl_arch02::load_muon_momentum_python_npy(&rt, &mut state, op)?;
        eprintln!("load-optim={} (legacy Muon-only restore)", op.display());
    } else if start_step > 0 {
        state.step = start_step;
        eprintln!(
            "warning: start-step={start_step} with fresh optimizer/EMA state; use --resume for exact continuation"
        );
    }
    let mut grads = Grads::zeros_like(&rt, &w)?;
    let mut tape = Tape::new(w.cfg.num_layers);
    // Bump grows with model_dim / mlp for transpose leftovers + simdgroup scratch.
    // Keep the bump proportional but bounded; large pool size classes no
    // longer round a 256 MiB request to 512 MiB.
    let bump_bytes = if w.cfg.model_dim >= 768 {
        512 * 1024 * 1024
    } else if w.cfg.model_dim >= 256 {
        256 * 1024 * 1024
    } else {
        64 * 1024 * 1024
    };
    rt.ensure_bump(bump_bytes)?;
    let mut inputs = DualInputBuffers::new(&rt, w.cfg.batch, w.cfg.seq_len)?;

    let synthetic = data_dir.is_none();
    if synthetic {
        eprintln!(
            "WARN: no --data-dir; using synthetic golden batches (BPB blocked without FineWeb)"
        );
    } else {
        eprintln!("data={}", data_dir.as_ref().unwrap().display());
    }

    let base_data_skip = fineweb_token_skip(seed);
    let data_skip = resume_meta
        .as_ref()
        .map(|m| m.data_cursor_tokens)
        .unwrap_or(base_data_skip);
    let loader = if let Some(ref dd) = data_dir {
        if data_skip > 0 {
            eprintln!(
                "data: FineWeb token cursor={data_skip} (seed base={base_data_skip}, resume-aware)"
            );
        }
        Some(PrefetchLoader::new_with_seed(
            dd,
            "train",
            w.cfg.batch,
            w.cfg.seq_len,
            w.cfg.bigram_vocab,
            w.cfg.vocab_size,
            4,
            Some(data_skip),
        )?)
    } else {
        None
    };

    let lut = arg(&args, "--token-bytes").and_then(|p| {
        TokenByteLut::load(&PathBuf::from(p))
            .map_err(|e| {
                eprintln!("warn: token-bytes: {e}");
                e
            })
            .ok()
    });
    let val_tokens = data_dir.as_ref().and_then(|d| find_val_shard(d));

    let mut logger = MetricsLogger::new(&out_dir);
    if let Some(ref manifest) = research_manifest {
        std::fs::create_dir_all(&out_dir)
            .map_err(|e| format!("create research output {}: {e}", out_dir.display()))?;
        std::fs::copy(manifest, out_dir.join("research-manifest.json"))
            .map_err(|e| format!("copy research manifest {manifest}: {e}"))?;
    }
    let steps = bench_steps.unwrap_or(iters);
    let is_bench = bench_steps.is_some();
    if is_bench {
        eprintln!("BENCH mode: {steps} steps (warmup 3 excluded from average)");
    }

    let mut last_log = Instant::now();
    let mut steps_since_log = 0usize;
    let mut bench_ms = Vec::new();

    for step_i in 0..steps {
        let step = start_step + step_i;
        let t0 = Instant::now();
        let profiled = step_i < 2 || (step % 100 == 0 && !is_bench);
        let mut prof = Profiler::new(profiled).with_sync({
            let rt = rt.clone();
            move || {
                let _ = rt.synchronize();
            }
        });

        prof.enter(Phase::DataPrep);
        let (ids, tgts) = if let Some(ref loader) = loader {
            let batch = loader.next()?;
            (batch.x, batch.y)
        } else {
            // Cycle golden synthetic batches.
            let bi = step_i % 3;
            let ids = tessl_arch02::parity::load_input_ids(&golden, bi)?;
            let tgts = tessl_arch02::parity::load_target_ids(&golden, bi)?;
            // Golden is B=16; if tok-mult>1, tile the batch.
            if w.cfg.batch > 16 {
                let reps = w.cfg.batch / 16;
                let mut xi = Vec::with_capacity(ids.len() * reps);
                let mut yi = Vec::with_capacity(tgts.len() * reps);
                for _ in 0..reps {
                    xi.extend_from_slice(&ids);
                    yi.extend_from_slice(&tgts);
                }
                (xi, yi)
            } else {
                (ids, tgts)
            }
        };

        prof.enter(Phase::Upload);
        zero_grads(&rt, &grads)?;
        let (ids_t, tgts_t) = inputs.upload(&rt, &ids, &tgts)?;

        prof.enter(Phase::Forward);
        let out = forward_f32_uploaded(&rt, &w, ids_t, tgts_t, &mut tape)?;

        prof.enter(Phase::Backward);
        let in_div_window = (1800..=2500).contains(&step);
        let read_norm = step % log_every == 0
            || step_i < 3
            || step_i + 1 == steps
            || (div_telemetry && in_div_window && step % 50 == 0)
            || dump_at == Some(step);
        let norm = backward_f32_opts_clip(
            &rt,
            &w,
            &tape,
            &mut grads,
            true,
            Some(&state.clip),
            read_norm,
        )?;

        let will_log = step % log_every == 0
            || step_i + 1 == steps
            || is_bench
            || (div_telemetry && in_div_window && step % 50 == 0);
        let weight_snapshot = if research_telemetry && will_log {
            Some(capture_weight_snapshot(&w)?)
        } else {
            None
        };

        prof.enter(Phase::Optim);
        let lr_mul = lr_sched.mul_at(step);
        optim_step(&rt, &mut w, &grads, &mut state, true, lr_mul)?;
        if rt.precision() == PrecisionMode::Bf16 {
            w.refresh_bf16_banks(&rt)?;
        }
        // synchronize covers M4 SharedEvent + CounterHeap t1 on the training CB.

        steps_since_log += 1;
        let do_log = will_log;
        // Always synchronize after each step. Cross-step async left tape / bump /
        // grad buffers free to race with the next encode and poisoned late-run
        // dynamics (~step 2100). Within-step async batching is unchanged.
        rt.synchronize()?;
        inputs.mark_synced();
        rt.bump_reset()?;
        prof.finish();

        // The loss and global norm are four-byte device reductions already
        // synchronized above. Read them every step so a research arm stops at
        // the first numerical failure rather than continuing to update NaNs
        // until the next log/eval boundary.
        let step_loss = out.read_loss(&rt)?;
        let step_grad_norm = state.clip.norm.contents_f32()[0];
        if !step_loss.is_finite() || !step_grad_norm.is_finite() {
            return Err(format!(
                "numerical failure at step {step}: loss={step_loss} grad_norm={step_grad_norm}"
            ));
        }

        let research = if let Some(snapshot) = weight_snapshot.as_ref() {
            Some(collect_research_telemetry(&rt, snapshot, &w, &grads)?)
        } else {
            None
        };
        let dispatches = rt.take_dispatch_count();

        let step_ms = t0.elapsed().as_secs_f64() * 1e3;
        if is_bench && (step_i >= 3 || steps <= 3) {
            bench_ms.push(step_ms);
        }

        if do_log {
            let elapsed = last_log.elapsed().as_secs_f64();
            let tps = (toks_per_step * steps_since_log) as f64 / elapsed.max(1e-9);
            let logged_norm = norm.unwrap_or(step_grad_norm);
            let clip = if logged_norm > 0.3 {
                0.3 / (logged_norm + 1e-6)
            } else {
                1.0
            };
            let divergence = if div_telemetry {
                Some(collect_divergence_norms_device(&rt, &w, &state)?)
            } else {
                None
            };
            let m = StepMetrics {
                step,
                loss: step_loss as f64,
                grad_norm_global: logged_norm as f64,
                clip_factor: clip as f64,
                lr_mul: lr_mul as f64,
                momentum: state.hp.muon_momentum(step) as f64,
                tokens_per_s: tps,
                step_ms,
                optimizer_ms: prof
                    .pairs()
                    .iter()
                    .find_map(|(name, value)| (*name == "optim").then_some(*value))
                    .unwrap_or(0.0),
                dispatches,
                profiled,
                phase_ms: prof.pairs(),
                rss_mb: mem_rss_mb(),
                current_physical_mb: mem_current_physical_mb(),
                precision: format!("{:?}", rt.precision()),
                tok_mult,
                divergence,
                research,
            };
            logger.log(&m);
            logger.console(&m);
            last_log = Instant::now();
            steps_since_log = 0;
        }

        if let Some(n) = dump_at {
            if step == n {
                let dump_root = out_dir.join(format!("dump_step{n}"));
                let wdir = dump_root.join("weights");
                let odir = dump_root.join("optim");
                eprintln!(
                    "dump-at: saving weights → {} and optim state → {}",
                    wdir.display(),
                    odir.display()
                );
                save_weights_python_npy(&rt, &w, &wdir)?;
                save_optim_state_python_npy(&rt, &w, &state, &odir)?;
                eprintln!(
                    "dump-at: done (optim_step={}, Python weights + optim_step3 layout)",
                    state.step
                );
            }
        }

        if !is_bench && checkpoint_every > 0 && state.step % checkpoint_every == 0 {
            let checkpoint_root = out_dir
                .join("checkpoints")
                .join(format!("step_{:08}", state.step));
            let consumed_since_start = state.step.saturating_sub(start_step);
            let cursor = data_skip.saturating_add(
                consumed_since_start.saturating_mul(toks_per_step.saturating_add(1)),
            );
            let meta = TrainingCheckpointMeta {
                version: CHECKPOINT_VERSION,
                step: state.step,
                data_cursor_tokens: cursor,
                seed,
                preset: preset.clone(),
                config: w.cfg.clone(),
                parameter_count: n_params,
                optimizer: optimizer_kind.to_string(),
                hyperparams: state.hp.clone(),
                clip_mode: state.clip_mode,
                schedule: lr_sched,
                bf16_precision: rt.precision() == PrecisionMode::Bf16,
                bf16_shadows_saved: w.bf16_banks.is_some(),
            };
            eprintln!(
                "checkpoint: saving exact state → {}",
                checkpoint_root.display()
            );
            save_training_checkpoint(&rt, &w, &state, &checkpoint_root, &meta)?;
        }

        if !is_bench && eval_every > 0 && (step + 1) % eval_every == 0 {
            if let (Some(ref lut), Some(ref val)) = (&lut, &val_tokens) {
                // Mid-run eval on live weights; final eval uses EMA (see copy below).
                match eval_sliding(
                    &rt,
                    &w,
                    val,
                    w.cfg.seq_len,
                    64,
                    lut,
                    w.cfg.bigram_vocab,
                    w.cfg.batch,
                    16_384,
                ) {
                    Ok(bpb) => eprintln!("step {step} | val sliding BPB (live) = {bpb:.4}"),
                    Err(e) => eprintln!("step {step} | eval skipped: {e}"),
                }
            }
        }
    }

    if is_bench {
        let avg = bench_ms.iter().sum::<f64>() / bench_ms.len().max(1) as f64;
        let tps = toks_per_step as f64 / (avg / 1e3);
        // The burn-port / 3070 Ti reference numbers are for `--preset sota`
        // (4L x 128d x mlp384) ONLY. Printing them next to an arch02-128m run
        // (24L x 768d x mlp2304, ~100x the parameters) invites an invalid
        // comparison — so gate them on the preset actually being measured.
        let is_sota_shape = w.cfg.num_layers == 4 && w.cfg.model_dim == 128;
        let refs = if is_sota_shape {
            " | vs burn-port ~2900 ms | 3070 Ti (PyTorch bf16) ~650-840 ms"
        } else {
            " | no same-shape CUDA/burn reference on record for this preset"
        };
        eprintln!(
            "BENCH done | tok-mult={tok_mult} B={} | {:.1} ms/step | {:.0} tok/s{refs}",
            w.cfg.batch, avg, tps
        );
        return Ok(());
    }

    // Final EMA sliding BPB
    if let (Some(ref lut), Some(ref val)) = (&lut, &val_tokens) {
        eprintln!("copying EMA → live weights for final eval...");
        copy_ema_into_weights(&rt, &state, &mut w)?;
        // Eval at B=16 for parity even if tok-mult was used for throughput.
        let saved_b = w.cfg.batch;
        w.cfg.batch = 16;
        match eval_sliding(
            &rt,
            &w,
            val,
            w.cfg.seq_len,
            64,
            lut,
            w.cfg.bigram_vocab,
            16,
            16_384,
        ) {
            Ok(bpb) => {
                // 1.9944 is the 3070 Ti number for the **sota ladder config**
                // (4L x 128d). Emitting it beside an arch02-128m BPB implies a
                // cross-scale comparison that was never run. Only attach the
                // reference when the shape matches.
                let is_sota_shape = w.cfg.num_layers == 4 && w.cfg.model_dim == 128;
                if is_sota_shape {
                    eprintln!(
                        "FINAL val BPB (EMA sliding) = {bpb:.4}  (3070 Ti sota-ladder reference 1.9944)"
                    );
                } else {
                    eprintln!(
                        "FINAL val BPB (EMA sliding) = {bpb:.4}  (no same-shape CUDA reference \
                         on record for {}L x {}d — do not compare against the sota 1.9944)",
                        w.cfg.num_layers, w.cfg.model_dim
                    );
                }
                let mut f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(out_dir.join("metrics.jsonl"))
                    .ok();
                if let Some(ref mut f) = f {
                    use std::io::Write;
                    let line = if is_sota_shape {
                        format!(
                            "{{\"final_ema_sliding_bpb\":{bpb:.6},\"reference_3070ti_sota_ladder\":1.9944}}"
                        )
                    } else {
                        format!(
                            "{{\"final_ema_sliding_bpb\":{bpb:.6},\"reference\":null,\
                             \"reference_note\":\"no same-shape CUDA run on record\"}}"
                        )
                    };
                    let _ = writeln!(f, "{line}");
                }
            }
            Err(e) => eprintln!("FINAL eval failed: {e}"),
        }
        w.cfg.batch = saved_b;

        if save_final_weights {
            // Persist EMA (copied into live weights above) for Core ML export.
            let ema_dir = out_dir.join("ema_weights");
            eprintln!("saving EMA weights → {}", ema_dir.display());
            save_weights_python_npy(&rt, &w, &ema_dir)?;
            eprintln!("EMA weights saved (Python .npy layout)");
        } else {
            eprintln!("final EMA weight save disabled for research funnel job");
        }
    } else if synthetic {
        eprintln!(
            "BPB blocked: FineWeb shards / --token-bytes not provided. \
             Pass --data-dir .../fineweb10B_sp1024 --token-bytes ../burn-port/token_bytes.json"
        );
        // Still dump live weights so export tooling has something to load.
        let dump = out_dir.join("live_weights");
        eprintln!("saving live weights → {}", dump.display());
        save_weights_python_npy(&rt, &w, &dump)?;
    }

    Ok(())
}
