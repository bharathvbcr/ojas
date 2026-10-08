# metal-native — Decision Log

Date: 2026-07-11. Decisions for the from-scratch Rust + MSL training stack under
`arch_02_value_resid/metal-native/`. Companion: [`README.md`](README.md)
(phases, parity numbers, benches). Burn-port decisions remain in
[`../DECISIONS.md`](../DECISIONS.md).

Plan (do not edit as status source of truth here): Cursor plan
`from-scratch_metal_training_stack_1a0e481a`.

---

## M1. Leave `burn-port` as reference; do not rewrite or delete it

- **Decision:** Keep [`../burn-port`](../burn-port) untouched as the A/B /
  dynamics reference and the validated parity ledger (RoPE NTK base, QK-norm
  before RoPE, XSA mixed-v, etc.).
- **Why:** metal-native replaces Burn on the *training hot path*; it does not
  replace the knowledge already paid for in burn-port. Cross-stack BPB and
  step-time comparisons need a frozen reference.
- **Overturn if:** burn-port and metal-native diverge on documented math and
  the burn path is proven wrong against Python goldens.

## M2. MPP TensorOps as primary GEMM; simdgroup as fallback

- **Decision:** Primary matmul is `mpp::tensor_ops::matmul2d` (Metal Performance
  Primitives / M5 neural accelerators). Hand-tiled `simdgroup_matrix` remains
  the portable / A/B fallback.
- **Why (Audit 2):** Leaving TensorOps idle would forfeit the largest M5 win
  and reintroduce “competitive GEMM” schedule risk. TensorOps also enables
  cooperative-tensor epilogues later.
- **Requires:** macOS 26+ / Metal toolchain with bf16 TensorOps support.
- **Overturn if:** TensorOps numerics fail golden gates at f32, or Apple
  deprecates the API for training-shaped GEMMs.

## M3. Metal 4 training encode only

- **Decision (overturned again 2026-07-12):** Training encode is **Metal 4
  only**. `GpuRuntime::new` fails clearly if the M4 package cannot initialize.
  Classic `MTLCommandQueue` / `MTLCommandBuffer` training encode, `MetalCommandPath`,
  Binder M3 arm, and `--metal3` / `--metal4` CLI flags are removed.
- **Binder:** `dispatch::Binder` is Metal 4 argument-table + const arena
  (`bind_tensor` / `bind_u32` / `bind_f32` / `dispatch` / `barrier`).
  `GpuRuntime::with_binder` always targets M4. **Audit 4:** one **compute
  encoder** is kept open across `with_binder` calls within a CB (packed
  dispatches + per-dispatch barriers). Telemetry `dispatch_count` still counts
  `with_binder` invocations. Packed ops (split-K, clip, AdamW segment) also call
  `barrier()` explicitly. M4 auto-inserts Dispatch→Dispatch Device barriers after
  each `Binder::dispatch`.
- **Const arena:** ~1 MiB shared staging; consts bump-allocate at distinct
  offsets (snapshot captures GPU addresses, not bytes). Cursor resets only after
  `synchronize` / waiting commit. Mid-step non-waiting commit never reuses
  offsets until GPU catch-up; 512-dispatch cap waits before allocator reuse.
- **Residency (Audit 4 P0):** `register_residency` on every `alloc_buffer`
  (+ const arena at init); `useResidencySet` after every `beginCommandBuffer`.
  Cold temps schedule recycle on last Arc drop; **`removeAllocation` + freelist
  after CB complete**. Hot/Bump kinds are not auto-recycled.
- **Timestamps:** CounterHeap t0/t1 folded into the training M4 CB (not
  stamp-only CBs). `synchronize` SharedEvent-waits that same compute queue.
- **Host-zero hygiene:** Mid-step zeros use GPU `zero_f32` (host
  `GpuBuffer::zero` races an in-flight CB). `GpuBuffer::zero` on a bump view
  would memset the whole slab — do not use it for bump windows.
- **Gates:** f32 goldens pass; `cargo test --release --lib` **31/31**.
  FineWeb/synthetic `--bench --bench-steps 20 --f32` B=16: **58.4** ms/step /
  **70151** tok/s / **276** binders (was 80.5 / 50856 / 278 pre-pack).
- **Overturn if:** a required target device lacks Metal 4 encode APIs, or M4
  encode regresses step-time without a residency / barrier fix.

## M4. Megakernels are Phase 4 goal; per-op kernels stay as fallback

- **Decision:** Land correctness on per-op kernels (Phases 1–2), then fuse
  (e.g. `rms_norm_scale`, block megakernels) to crush dispatch count.
- **Status:** `resid_mix_rms_norm_scale_*`, `residual_scale_add_rms_norm_scale_*`
  (f32 + bf16 stream twins), `skip_resid_mix_rms_norm_scale_f32` (decoder),
  `rms_norm_smear_f32` (stem). ~403 dispatches/step (B=16).
- **Overturn if:** Register pressure makes block fusion slower than many small
  TensorOps dispatches on M5.

## M5. Whole-bank Muon NS5 (4 dispatches)

- **Decision:** Muon NS5 runs as whole-bank kernels over the four shape stacks
  (qo / kv / mlp_up / mlp_down), not per-matrix loops.
- **Why:** Matches Python 3D bank design; ~60+ tiny NS5 dispatches → 4.
- **Parity:** Covered by `optim_step3_parity_vs_goldens` (moments + params).

## M6. Core ML / ANE are deploy-only

- **Decision:** Training stays custom MSL. Core ML `.mlpackage` (+ ANE) is
  Phase 5 inference export only (`scripts/export_coreml.py`).
- **Why:** Core ML cannot express this training loop (Muon, custom bwd,
  on-device clip). `MTL4MachineLearningCommandEncoder` is inference-only.
- **Stateful KV (2026-07-12):** Torch `Arch02KV` + `decode_kv_reference` PASS;
  `--stateful-kv` emits `arch02_sota_decode_step.pt`. Core ML `StateType`
  `.mlpackage` convert still blocked (slice lowering). See
  `out/coreml_export/STATEFUL_KV_CORE_AI.md`.
- **Overturn if:** Apple ships a training story that can express Muon + custom
  attention bwd on ANE/GPU without losing parity.

## M7. f32 goldens; bf16 is the training default *intent*

- **Decision:** Golden exporter and parity gates run in **f32** (atol fwd 1e-5,
  bwd 1e-4). Runtime may set `PrecisionMode::Bf16`, but the hot path must still
  be wired end-to-end before trusting bf16 BPB.
- **Why:** Deterministic gates vs Python; bf16 autocast changes RMSNorm eps and
  matmul accumulation — gate first in f32, then enable bf16 carefully.
- **Phase H status:** `PrecisionMode::Bf16` selects real bf16 TensorOps GEMMs
  (`gemm_train` / `gemm_tn_train` / `gemm_nt_train`: f32 masters; activations /
  weight views cast to bf16 and **reused** across sibling GEMMs — persistent bf16
  operands). GEMM accumulate stays f32. Softcap/CE stay f32. Residual-stream
  megakernels write **bf16 norm outs** (`resid_mix_rms_norm_scale_bf16`,
  `residual_scale_add_rms_norm_scale_bf16`); tape mix/mid stay f32. Flash uses
  **bf16-input / f32-accum** FA-2 with LSE + matching bf16 bwd when TensorOps is
  present (f32 FA-2 under `--f32` / goldens). Optional `--tf32` /
  `set_relaxed_precision` enables `matmul2d_tensorops_f32_relaxed` on the f32
  path — keep off for 1e-5 goldens.
  Do not claim BPB parity for bf16 until an f32 run reproduces ~1.99.

## M11. Default clip = Soft (`Muon×√c`, `AdamW×c`); Match / Python are opt-in

- **Decision:** After on-device `clip_grad_norm_`, **Muon** multiplies step+WD by
  `sqrt(clip_coef)`; **AdamW** (embed + scalars) multiplies by `clip_coef`
  (`ClipMode::Soft`). Best long-horizon Soft recipe on metal-native.
- **Evidence (seed 1337, f32):**
  - Soft-everywhere (`√c` on AdamW+Muon), 3k FA_TILED: EMA **2.0369**, late gnorm ~4
  - Soft-everywhere, 20k FA_TILED: best live **2.0561 @3499** then gnorm→10k+,
    Soft clip≈0, FINAL EMA **2.2575** — **REJECT** past ~3.5k
  - Same Soft 3k dump + continue Soft-everywhere →8k: FINAL **2.25**, explode
  - Same dump + Match →8k: stable, FINAL **1.9653**
  - Soft (Muon√c / AdamW×c) FA_TILED **from-scratch 3k**: EMA **2.0222**, late
    gnorm ~5.7 — **better than Soft-everywhere 2.0369**
  - Soft (Muon√c / AdamW×c) continue Soft 3k dump →8k: FINAL **1.9469**, best live
    **1.9635 @6499** — **KEEP** (beats failed Soft 2.2575 and CUDA ref 1.9944 @8k)
  - Soft (Muon√c / AdamW×c) **from-scratch 20k** FA_TILED (no warmdown): FINAL EMA
    **1.9178** (beats CUDA 1.9944) but late raw gnorm O(100–600), chronic Soft
    clip≈0, `vr_λ` 0.7→2.0, `attn_scale` 11→7 — BPB improves via clip+EMA despite
    rising pre-clip norm
  - Soft-everywhere row FA continue also explodes — FA_TILED not required for fail
  - Soft harden (accum off, no FA_TILED) 3k: EMA **2.0502**, late gnorm ~3.4
  - Match 3k: EMA **2.0649**, late gnorm max ~2.8
  - Python: diverges ~2500 (gnorm→500+)
- **Why Soft split:** Soft √c on AdamW over-updates `vr_λ` / `attn_scale` under
  chronic clip; banks shrink via WD while scales climb → gnorm explode. Soft on
  Muon alone keeps the bank-starvation fix without scalar runaway.
- **`--clip-match` / `--clip-python`:** opt-in. See README BPB table.
- **Default 20k Soft recipe (KEEP):** Soft-split + `METAL_NATIVE_FA_TILED=1` +
  `METAL_NATIVE_GEMM_ACCUM` **off** + `--iters 20000 --warmdown 3500 --f32` +
  FineWeb/`token_bytes`. Seed 1337 (**`--golden-init`**, FineWeb skip=0): FINAL
  EMA **1.8969**, gnorm @19999 **216**. True seed-42 arm (seeded init + FineWeb
  skip; `out/.../seed42_..._warmdown_reseed/`): FINAL EMA **1.8876**, gnorm
  @19999 **60**; 2-seed Soft mean **1.8922** (beats CUDA ~1.990). Prior
  unused-seed “seed 42” was re-run noise (FINAL EMA **1.8925**). CLI `--warmdown`
  default stays **0** (3k Soft unchanged); banner warns when `total_iters≥20000`
  and warmdown=0.
- **`--seed` (wired 2026-07-12):** (1) **weight init** via `init_weights_seeded`
  (orthogonal banks + normal embeds; SplitMix64+QR ≈ torch structure, not Philox
  bit-identical); (2) **FineWeb cursor** via `fineweb_token_skip(seed)` (default
  on; `METAL_NATIVE_DATA_SEED=0` → skip 0 = CUDA sequential-from-0).
  `--golden-init` / `METAL_NATIVE_GOLDEN_INIT=1` loads exported seed-1337 golden
  banks (parity / published Soft arm). CUDA ladder: seed → `torch.manual_seed` →
  init only; FineWeb `TokenStream` is seed-agnostic sequential.
- **Remaining risk @20k:** Soft-split late gnorm climbs with constant LR under
  chronic Soft clip (raw gnorm O(100–600) by 15–20k while BPB still improves).
  **Mitigation (2026-07-12):** `--warmdown 3500` on 20k — linear 1→0 from step
  16500; scales matrix/embed/scalar LRs in `optim_step`. Validated: FINAL EMA
  **1.8969** (vs no-wd **1.9178**), gnorm @19999 **216** (vs **636**), warmdown
  window mean gnorm **210** (vs **303**). Do not re-enable Soft-everywhere.
- **3k vs 20k:** 3k sota toy keeps warmdown=0; quality Soft still wants FA_TILED.
  20k always pass `--warmdown 3500` explicitly (docs + CLI banner).
- **100k Soft (FAILED → fixed + validated 2026-07-13):** `--warmdown 10000`
  (last 10%, start@90k) on Soft-split + FA_TILED + golden-init: best live BPB
  **1.9137 @15999**, then rebound ~1.96–1.97; gnorm ≥1000 from ~21k, mean ~3.4k
  by 45–50k; stopped ~53.7k with `lr_mul=1.0` (warmdown never reached).
  Soft-everywhere-style milder than 20k Soft-everywhere (2.2575) but same stop
  rule. **Root cause:** constant LR past the Soft productive window (~16–20k);
  linear-to-end over last 10% is too late, and linear 1→0 from 20k→100k is too
  shallow by the ~21k cliff (lr still ≳0.9). **Fix:** WSD schedule —
  `--warmdown-start 16000 --warmdown 24000 --lr-floor 0.1 --final-warmdown 10000`
  (constant → linear 1→0.1 by 40k → hold → final 0.1→0 over last 10k). CLI:
  `--warmdown N` unchanged for 20k; new `--warmdown-start` / `--lr-floor` /
  `--final-warmdown`. Published arm: `--golden-init` + `METAL_NATIVE_DATA_SEED=0`.
  **Validated** `out/.../100k_..._wsd/`: FINAL EMA **1.8828**, best live
  **1.8819 @96999**, no BPB rebound through former blow-up (~21–50k); hold-window
  mean gnorm ~919 (failed late-wd mean ~3.4k); gnorm≥2000 only 1.4% of logs /
  no sustained streak; wall **~5.6 h** @ ~60k tok/s (M5 Pro). Occasional
  single-step spikes (max ~6.3k @73k under floor) remain under Soft clip — not
  Soft-everywhere diverge.
- **Remaining Δ vs CUDA @3k (~1.99):** FA numerics; @20k Soft+warmdown and @100k
  WSD already beat CUDA seed-1337 on EMA sliding BPB.

## M8. Flash attention: simdgroup FA-2 production; TensorOps multi-block blocked

- **Prior decision (overturned for Phase A):** Defer TensorOps cooperative-tensor
  flash because GQA causal online softmax did not map cleanly onto early
  TensorOps patterns.
- **Overturn condition partially met:** macOS 26.3+ cooperative-tensor *inputs*
  + WWDC26 session 330 recipe (`matmul2d` → `reduce_rows` / `map_iterator` →
  left-input ·V). That recipe is **single-tile** fused attention, not multi-block
  online FA-2.
- **Current status (macOS 26.5 verified):**
  - **Hot path (step-time):** row-wise FA-2 bwd default @ T=256.
  - **Soft quality opt-in:** `METAL_NATIVE_FA_TILED=1` (BR=BC=32) — Soft EMA
    **2.0369** vs row **2.050** (seed 1337, accum off); ~69 vs ~57 ms/step.
  - **Metallib probes (not training default):**
    - `flash_attn_tensorops_tile_f32` — session 330 single Br×Bc tile
    - `flash_attn_tensorops_online_f32` — experimental multi-block causal GQA
      online FA that still stages S/P/O through threadgroup between TensorOps
      tiles (partially defeats cooperative-input). The 2026-09-04 audit removed
      redundant Q/K/V staging and aliases the disjoint S/P lifetimes, reducing
      explicit static threadgroup storage from 28,928 to 12,544 bytes (-56.6%).
      On the current physical M5 Pro validation run, the compiled pipeline
      reported 25,088 bytes static and 0 bytes dynamic against the 32,768-byte
      device limit, leaving 7,680 bytes headroom. Parity against FA-2 measured
      max-absolute error 1.117587e-8 at T=256 and 7.451e-9 at ragged T=65, with
      zero non-finite outputs.
      The old shader-validation pipeline reported 57,856 bytes against the
      M5 Pro's 32,768-byte limit; the refactor makes that validation lane
      executable while preserving aligned T=256 and ragged T=65 parity. The
      opt-in selector now falls back before this D=32-only shader for all other
      supported head dimensions. These are correctness/resource results, not a
      timing claim: the kernel stays off the hot path until pinned pre/post
      paired timings, sota-shape goldens, TensorOps bwd, and an Instruments NAX
      win.
  - **`--flash-tensorops` Soft A/B (2026-07-12):** EMA **2.0462** but late gnorm
    **~13 @2999** — **REJECT** for Soft ladder; M8 still blocks default.
- **Blockers to wiring TensorOps flash as default:**
  1. Session 330 has no multi-block online-softmax / O-rescale recipe — carrying
     `m,l,O` across key blocks forces TG staging of O (and usually P).
  2. Causal + ragged last tiles require element-wise score masking before
     `reduce_rows`; brittle vs cooperative layouts at D=32.
  3. `is_compatible_as_left_input` often false at D=32 → TG round-trip for P@V
     (tile probe already falls back).
  4. No published TensorOps **bwd + LSE** analogue for GQA; training bwd stays
     simdgroup. Mixing TensorOps fwd with simdgroup bwd risks numerics drift.
  5. Must beat simdgroup FA-2 on Instruments NAX **and** pass ≤1e-5/≤1e-4 goldens
     before flipping the default.
- **Gates:** bwd goldens ≤1e-4, `flash_attn_lse_and_bwd_gate`, late-checkpoint
  FD scaffold (`late_checkpoint_flash_fd_scaffold`).
- **Quality track (Audit 6 Soft-harden + P3):** Soft EMA **2.050** default
  (accum off); **2.037** with FA_TILED; CUDA **~1.994**. Keep **row bwd default**
  for step-time; tiled Soft opt-in; TensorOps flash Soft **REJECT** (late gnorm).
- **Phase G (2026-07-21) — FA blocksoft Soft@3k:** see **M17**. FA-3-class
  blockwise online-softmax probe does **not** close Soft@3k → 1.9944; gap remains
  open (f32 vs CUDA bf16 / non-FA dynamics under current defaults).
- **Overturn further (wire TensorOps flash as default) if:** multi-block
  TensorOps FA (+ bwd) passes the same gates at sota shapes and Instruments NAX
  utilization beats the simdgroup path without TG-dominated epilogues.

## M17. Phase G Soft@3k FA blocksoft — REJECT for Soft BPB (2026-07-21)

- **Code audit (sota Soft quality paths):**
  - **Fwd:** default `flash_attn_fwd_f32` (FA-2 sequential online softmax inside
    each BC=32 tile); opt-in `FA_FWD_FAST` → `*_d32_*`; opt-in `--flash-tensorops`
    → `flash_attn_tensorops_online_f32` (Soft REJECT late gnorm ~13).
  - **Bwd:** default row `*_row_d32_*` (speed); Soft quality
    `METAL_NATIVE_FA_TILED=1` → tiled `flash_attn_bwd_{dq,dkv}_f32` BR=BC=32.
  - **Clip:** Soft-split (`Muon×√c` / `AdamW×c`) KEEP; Soft-everywhere REJECT.
  - **Not defaults:** `GEMM_ACCUM`, Soft-everywhere, `--flash-tensorops`.
- **Probe:** FA-2/FA-3 **blockwise** online softmax — rowmax over the BC tile,
  single rescale of `(m,l,O)`, plus `precise::exp`/`precise::log` + `fma`.
  Kernels: `flash_attn_fwd_blocksoft_{f32,d32_f32}`. Flag:
  `METAL_NATIVE_FA_BLOCKSOFT=1` (default **off**). Gate
  `fa_fwd_blocksoft_close_to_sequential`: max|ΔO|/|ΔLSE| vs sequential ~1e-7.
- **Soft@3k seed 1337 (fair A/B, Soft-split + FA_TILED, no golden-init, current
  Audit 7/8-era defaults):**
  | Arm | FINAL EMA sliding BPB | Artifact |
  |-----|----------------------:|----------|
  | Control (FA_TILED) | **2.1063** | `out/phaseG_ctrl_softsplit_fatiled/` |
  | + FA_BLOCKSOFT | **2.1044** | `out/phaseG_blocksoft_softsplit_fatiled/` |
  | CUDA 3070 Ti sota ladder | **1.9944** | — |
- **Verdict:** **REJECT** FA_BLOCKSOFT as Soft quality recipe / default. Δ ≈
  −0.0019 EMA is noise; does **not** move toward 1.9944. Keep flag + kernels as
  research (correct FA-2 block formulation) — not Soft KEEP.
- **Note vs historical Soft-split 2.0222:** under today's code the same Soft-split
  + FA_TILED recipe measures ~**2.106**, not 2.0222. Phase G does not attribute
  that drift to blocksoft (control matches). Soft@3k gap to CUDA is still open;
  next Soft FA lever is **not** further online-softmax recurrence tweaks of this
  class. Do not re-default Soft-REJECTED paths.

## M9. `--tok-mult` for throughput; sota BPB stays B=16

- **Decision:** Reproduce ladder BPB at B=16 / 4096 tok/step. Use `--tok-mult`
  (B=32/64) only for throughput sweeps and future sprint-scale fills.
- **Why:** 64 GB unified memory underfills at B=16; fair BPB comparison must
  match the 3070 Ti sota shape.

## M10. Optim golden gate vs e2e AdamW sensitivity

- **Decision:** Authoritative optim correctness is **optim-only** parity against
  Python step grads + `optim_step3/` goldens (**45/45**). Do not treat long-horizon
  e2e param equality as a hard gate.
- **Why:** AdamW amplifies tiny residual bwd noise; kernels can be correct while
  multi-step e2e params drift.
- **Related:** Full-run BPB still must eventually match ~1.99; that is a
  numerics/dynamics problem, not a substitute for the 45/45 kernel gate.

## M12. ~16M Soft scale-up (`medium_16m` / `--preset 16m`)

- **Decision (2026-07-13):** First Soft path beyond the 0.78M sota toy is
  **metal-native** `ModelConfig::medium_16m`: L=12 C=384 H=12/6 hd=**32**
  mlp=1152 (3×) V=1024 bigram 512/48 VE@10–11 XSA last-4 → **16,411,948**
  params. Default **B=16 T=256** (4096 tok/step; matches sota toy tokens/step).
  CLI: `--preset 16m` (+ optional `--batch` / `--seq-len`). Soft-split KEEP;
  `METAL_NATIVE_FA_TILED=1`; `METAL_NATIVE_GEMM_ACCUM` off; seeded init (no
  golden banks at this shape).
- **Why this shape:** Stays inside Metal FA (`head_dim≤64`; TensorOps probe
  wants 32), GQA group=2 + mlp=3× like sota/sprint, even L for U-net skips,
  ~16.4M within ±20% of 16M. burn-port sprint (11L/512d/hd=64, ~26.7M) is the
  larger reference but breaks hd=32 Soft FA preference.
- **Step-shape optimise (2026-07-13, M5 Pro 64GB, Soft f32, `--bench-steps 18`):**
  | Config | ms/step | tok/s | RSS | Notes |
  |---|---:|---:|---:|---|
  | FA_TILED B8 T512 | **1259** | 3252 | ~5.5 GB | prior default |
  | row FA B8 T512 | ~1314* | ~3118* | ~5.5 GB | FA_TILED wins ~7% |
  | FA_TILED **B16 T256** | **954** | **4292** | ~5.5 GB | **KEEP default** |
  | FA_TILED B16 T256 +tf32 | 949 | 4314 | ~5.5 GB | within noise; leave off |
  | FA_TILED B16 T512 | 2176 | 3764 | ~9.8 GB | 2× tok/step; not worth RSS |
  | FA_TILED B8 T1024 | ~3454* | ~2372* | ~9.8 GB | reject |
  | FA_TILED B4 T1024 | ~1879* | ~2180* | ~5.5 GB | reject |
  \*last-step proxy when BENCH line went to stderr under `/usr/bin/time`.
  Winner: **FA_TILED + B=16 T=256** (~24% vs B8/T512; GEMM_ACCUM stays off).
  Artifacts: `out/opt16m_ab/`.
- **1k Soft verify (2026-07-13):** FA_TILED + B16/T256 + Soft-split + warmdown=0
  + seed 1337 → `out/sota_f32_clipsoft_16m_seed1337_1k_opt/`. Wall **~16 min**
  (08:43–08:59). Live BPB 2.543→2.200 (@199→999); **FINAL EMA BPB 2.1462**.
  Loss 6.94→3.63; gnorm mostly ~0.75–1.3 with one transient spike **3.91 @900**
  then recovery (no NaN / no Soft-everywhere explode). Steady **~890–950 ms/step**
  (~4400 tok/s), RSS ~5.5 GB — vs pre-opt B8/T512 ~1245 ms.
- **20k recipe sketch (1k clean → go):** Soft-split + FA_TILED +
  `--warmdown 3500 --iters 20000 --f32 --clip-soft --seed 1337` into a distinct
  `out/medium16m_...` / `out/sota_f32_clipsoft_16m_...` dir. Watch for mid-run
  gnorm spikes like @900 on 1k; revisit WSD if late gnorm mirrors the 100k toy
  cliff. Wall estimate @~0.95 s ≈ **~5.3 h / 20k**.
- **Generalization done:** `Tape::new(num_layers)`, bank/init already cfg-driven,
  train bump 256 MiB when C≥256. Sota goldens / parity tests stay on
  `Tape::new_sota()`.
- **Overturn / retune if:** OOM at B=16 T=256 on M5 Pro 64GB; Soft 1k/20k needs
  longer context (then try B8/T512); or BPB dynamics need sprint-like hd=64
  (then revisit FA TG caps / burn-port).

## M13. Exact 128M gate and bank-batched TensorOps Muon (2026-07-14)

- **Exact preset:** `arch02-128m` = L24/C768/H24/Hkv12/D32/MLP2304 and the
  lean Arch02 auxiliaries, exactly **128,367,988** parameters. Default champion
  shape is B16/T256 (4096 tokens/step), bf16 dense operands with f32 masters,
  accumulation, optimizer state, and EMA.
- **Correctness repairs:** target QKV backward no longer truncates C/KV tails;
  exact C=768/KV=384 parity is covered. Version-6 checkpoints contain all masters,
  actual persisted bf16 shadow bits, Adam slots, Muon banks, EMA, schedule, seed, step, and
  data cursor. Save/reload/replay deltas on the GPU test are loss 0, grad norm
  `8.94e-8`, and QO master `2.98e-8` (within `OPTIM_ATOL=1e-4`; fresh Metal
  queues do not promise bit-identical reduction order).
- **Rejected exact-scale path:** one-threadgroup-per-matrix simdgroup NS5 took
  **~4201 ms optimizer / 6813 ms total** on the first B16/T256 step. Enabling
  `METAL_NATIVE_MUON_SG=1` changed nothing because it was already the default.
- **Landed replacement:** bank-batched MPP TensorOps NN/TN/NT contractions. One
  flat grid covers every same-shape matrix; fused kernels surround contractions
  with momentum+Frobenius normalization, polynomial combines, master update,
  weight decay, and EMA. Tall and wide batched NS3 match the host oracle.
- **Measured exact-scale result:** first step **478.8 ms optimizer / 2930.5 ms
  total** (Muon **8.8x** faster); six-step steady state after three warmups
  **2816.2 ms/step, 1454 tok/s**, fixed **1701 dispatches**, RSS settles at
  **16.1 GB**, no NaN/swap/dispatch rollover. First-step loss/gnorm/clip are
  unchanged at **6.9624 / 1.046 / 0.287**.
- **Decision:** memory, numerical, and command-budget smoke gates pass. The
  2,000-step champion remains blocked until the broad optimizer funnel has
  native-parity-qualified contenders and selects a winner; do not infer a
  champion from the Muon systems result alone.

## M14. Native optimizer funnel and exact champion preflight (2026-07-14)

- **Native arms:** parity-qualified Metal implementations now cover 14/16 study
  candidates: Muon NS5/NS3/Polar, NorMuon, Muown, MONA, AdamW, Lion, Cautious
  AdamW/Lion, momentum SGD, Sophia, Schedule-Free AdamW, and Prodigy. Same-shape
  Muon-family matrices use bank-batched TensorOps; full-parameter controls use
  fused device updates and role-specific LR/decay routing.
- **Explicit exclusions:** MiMuon needs exact per-matrix singular-gap routing
  from SVD; SOAP needs periodic symmetric eigendecomposition. Metal/MPS does not
  expose those GPU decompositions. CPU Accelerate would force synchronization,
  so both are systems-gate failures rather than approximate or silent fallbacks.
- **Telemetry:** research log steps snapshot weights on-device and reduce
  gradient/update norms by role, sampled orthogonality error, row drift,
  Frobenius spectral proxy drift, and nonfinite counts to fixed-size host scalar
  summaries. Logs also record optimizer time, total step time, dispatches, RSS,
  and current physical footprint.
- **Funnel:** `../MLSystemsLab/nanolab/native_funnel.py` owns a resumable job ledger and exact
  argv for the five-point 16M LR sweep (100 steps), stable 500, two-seed 1000,
  exact-128M top-four 500, and exact-128M top-two/two-seed 1000 stages. AdamW and
  NS5 remain mandatory anchors. `../MLSystemsLab/research/champion-run.json` stays locked until
  advancement completes.
- **Exact preflight:** checkpoint/replay at 128,367,988 params, B16/T256 passes:
  loss delta `0`, gradient delta `7.15e-7`, sampled master-weight delta
  `2.41e-6`, current physical **13.16 GiB**, **1707 dispatches**, zero swap.
  Fresh Metal queues are replay-equivalent at loss atol `1e-5` and optimizer
  atol `1e-4`, not promised bit-identical reductions.
- **Checkpoint I/O correction:** the first full gate exposed a per-float NPY
  syscall loop (~2 MB/s). Bulk contiguous little-endian payload I/O is now
  required. It reduced the exact gate from a multi-minute incomplete save to a
  complete save/load/replay in seconds.
- **Decision:** implementation and preflight infrastructure are ready, but the
  2,000-step command remains intentionally locked. Run the equal-data funnel,
  then run `exact_gate --optimizer <winner>`; only an unlocked manifest may
  launch the champion.

## M15. Champion completed; Audit 7 backward optimization (2026-07-19)

- **Champion landed.** The equal-data funnel completed through the exact-128M
  two-seed 1000 stage and unlocked **muon_polar_adamw @ matrix-lr 0.05**. The
  2,000-step run (`out/champion_128m_seed1337`, bf16, seed 1337) finished at
  FINAL EMA sliding BPB **2.0158**. The historical CUDA **1.9944** figure is the
  **sota ladder** (4L×128d) only — **not** a same-shape 128M twin; that
  comparison is withdrawn until `logs/cuda128m` lands from
  `scripts/cuda_ref_128m.sh` with disclosed `GRAD_ACCUM_STEPS`. Timing was
  ~2.94–3.07 s/step, 1392 tok/s, 1975 dispatches, 13.5 GiB current physical,
  and zero nonfinite values. It ran **without** `METAL_NATIVE_FA_TILED`.
- **Phase wall:** fwd ~300 ms · **bwd ~2170 ms** · optim ~450 ms. Backward is
  ~74% of the step, so Audit 7 targets backward exclusively.
- **GEMM-side flips (measured, KEEP pending champion rerun):**
  `METAL_NATIVE_BWD_CAST_ONCE=1` casts shared f32 grad operands to bf16 once per
  site instead of once per consuming GEMM (11 sites); `METAL_NATIVE_GEMM_ACCUM_DX=1`
  uses `multiply_accumulate` for **dX** NT accumulates only, leaving dW bank
  accumulation on temp+add — the Audit 6 Soft-regression suspect. Together:
  **~3028 → ~2811 ms/step, 1702 → 1482 dispatches (−7.2% / −220)**, beating the
  full Audit 6 accum reference on both axes.
- **CAST_ONCE parity:** step 0 is bit-identical; steps 1–2 differ by ~3e-6,
  the same order as measured off-vs-off run noise (~4e-6). This is inside the
  M13/M14 replay-equivalence atol of `1e-5`. A bit-identical gate is **not**
  the right criterion here — fresh Metal queues do not promise identical
  reduction order. The harness now gates on the documented atol.
- **The backward is one kernel.** `METAL_NATIVE_BWD_PROFILE=1` (synced
  per-section timing) shows **`fa_dqdkv` ≈ 70%** of backward (~1.1 s), then
  mlp_gemms 11%, qkv_gemms 5.3%, resid_glue ~4.8%, and thirteen other sections
  below 3% each.
- **P2 (`METAL_NATIVE_MLP_COOP_POSTFIX`) is closed, not deferred.** It targets
  the 11% MLP GEMM section, which already runs on TensorOps/NAX, while the 70%
  scalar FA kernel is untouched. Reopen only if FA bwd stops dominating.
- **FA_TILED does not change the FA regime.** Tiled costs the same ~1.1 s as
  the row path at T=256/128M, so BR=BC=32 tiling is not the lever. Its value is
  quality: 500-step champion-config EMA BPB **2.4163 vs 2.4209** row (−0.0046,
  single seed).
- **Two structural defects in the row FA bwd kernels** (fixed in
  `kernels/flash_attn_bwd_fast.metal`, `*_row_d32_*`):
  1. Loop-invariant `Q`/`dO` (dQ) and `K`/`V` (dKV) are re-read from device
     memory every inner iteration. The compiler cannot hoist them because
     `device const float *Q` may alias the `device float *dQ` output.
  2. `d_lim = min(D, 64u)` is a runtime bound, so `thread float dq[64]` is
     dynamically indexed and spills to private memory instead of registers,
     wasting half its slots at head_dim 32.
  `METAL_NATIVE_FA_FAST=1` fixes both with a compile-time `DH = 32` and full
  unroll. Numerics are unchanged (same ops, same order).
- **bf16 FA bwd is a consistency repair, not only bandwidth.** Under
  `PrecisionMode::Bf16` the forward computes attention from bf16 Q/K/V and tapes
  an LSE derived from bf16-rounded scores, while the f32 backward recomputes
  those scores in f32 — `p = exp(score_f32 − L_bf16)` is a real fwd/bwd
  mismatch. `METAL_NATIVE_FA_BF16=1` reads bf16 Q/K/V with f32
  dO/L/Delta/scores/accum/grads, matching `flash_attn_fwd_bf16`. dO stays f32
  because dV accumulates `p * dO` directly.
- **Scope limits:** both FA flags require `head_dim == 32` (host-guarded, warns
  once and falls back otherwise) and are ignored under FA_TILED — the
  specialized kernels are row-path only. A tiled bf16 twin is follow-up work,
  justified only if tiling's quality edge survives a seed repeat.
- **Gates:** `fa_bwd_row_d32_matches_generic` (lib) checks the f32 variants
  against the generic row kernels at ≤1e-5 and the bf16 variants at bf16-class
  tolerance. Harness: `scripts/bench_128m_ab.sh {profile,speed,cast-parity,fa,fa-quality,quality}`.
- **Decision:** defaults stay OFF for every Audit 7 flag until the FA ladder and
  its 500-step quality probe report. The champion rerun then stacks the winning
  set and must clear BPB < 2.0158 to replace `out/champion_128m_seed1337`.

## M16. Audit 7 champion shipped; Audit 8 instrumentation (2026-07-20)

- **Champion replaced.** `out/champion_128m_seed1337_audit7` (CAST_ONCE +
  FA_BF16, seed 1337, 2000 steps): FINAL EMA BPB **2.015576** vs the previous
  champion's **2.015756**, at **2005 ms/step / 2071 tok/s** vs 2895 ms / 1431 —
  **+44.7% throughput**, backward **1964 → 1056 ms (−46%)**, dispatches
  1975 → 1899, current physical flat at ~13.5 GiB, zero nonfinite. The BPB
  delta (−0.00018) is a tie against a measured 500-step seed spread of 0.032.
- **Defaults flipped.** `METAL_NATIVE_BWD_CAST_ONCE`, `METAL_NATIVE_FA_FAST`,
  and `METAL_NATIVE_FA_BF16` now default **ON**; set the env var to `0` to
  restore the old path. Rationale: a validated champion must not depend on
  hand-set environment variables (`champion-run.json` does not carry them).
  bf16 FA bwd remains gated on `PrecisionMode::Bf16` **and** `head_dim == 32`,
  so f32 golden runs keep the exact path.
- **`ACCUM_DX` rejected as default — but for the right reason.** A single-seed
  500-step read suggested +0.0040 BPB. Two seeds refuted it: the sign flips
  (+0.0103 on seed 42, −0.0063 on seed 2026) and the control's own seed spread
  is 0.0319, ~10× the effect. It is **not** a proven regression; it is
  unresolved, worth only ~3% step time post-FA, and inherits the Audit 6
  accumulate-mode risk. Off by default, retained as an opt-in flag.
- **Correction — bf16 FA bwd is not a consistency fix.** An earlier Audit 7
  note claimed it aligned bwd with a bf16 forward. `model_fwd::use_bf16_flash`
  returns `false` unconditionally, so the forward *always* runs f32 flash and
  `flash_attn_fwd_bf16` is unreachable. bf16 backward therefore introduces a
  small precision mismatch rather than removing one. It is justified
  **empirically only** (two seeds + 2000-step champion showed no cost). The
  claim is retracted in `optimization_map.md`, `ab_flags.rs`, and the kernel
  header.
- **`fa_dqdkv` after the fix:** 892 → **80 ms** (11×) on the synced profile;
  total synced backward 1281 → 446 ms. The backward is now GEMM-dominated
  (mlp_gemms 29%, fa_dqdkv 18%, qkv_gemms 15%, resid_glue 14%) and those GEMMs
  run at ~10.7 TFLOP/s, so remaining backward headroom is thin.
- **P2 / MLP coop postfix stays closed.** It targets the 11–29% GEMM sections
  that already run on TensorOps/NAX.
- **New instruments:** `METAL_NATIVE_FWD_PROFILE` and
  `METAL_NATIVE_OPTIM_PROFILE`, both reusing the backward profiler. The
  optimizer (~23% of step) had never been sectioned; M13 measured it only in
  aggregate.
- **Audit 8 forward d32 flash — MEASURED KEEP (2026-07-21).**
  `kernels/flash_attn_fwd_fast.metal` `flash_attn_fwd_d32_{f32,bf16}`:
  flash **264 → 39 ms (−85%)**, step **−14.9%** (`optimization_map` Audit 8b).
  Under Bf16 + `FA_FWD_FAST` this is the first reachable bf16 forward flash.
  Parity: `fa_fwd_d32_matches_generic`. **Audit 9A defaults `FA_FWD_FAST` ON.**
- **`resid_glue` atomics hypothesis — WITHDRAWN as a speed claim.** Rowblock
  (`GLUE_ROWBLOCK`) cut atomics ~128× and **did not move the clock** (glue_off
  1488.5 vs rb16 1481.7 ms/step; rb128 regresses). Do not quote “~50 ms atomics
  headroom.” Re-diagnose or accept ~60 ms floor (Audit 9D). Flag stays opt-in
  research only — **removed from blog WIN**.
- **ACCUM_DX “KEEP −7.2%” ship claim — WITHDRAWN.** Not in WIN; default OFF.
- **FA_TILED not required for 128M quality.** Champions/long20k ran without it.
- **Champion replaced again (Audit 8 WIN).** `out/champion_128m_seed1337_audit8`
  FINAL EMA **2.0107** (~1683 ms/step mid-run); seed2026 **2.0404**; long20k
  **1.8155** @ ~1582 ms/step. WIN profile: bwd ~457 / fwd ~412 / optim ~413 with
  **muon_banks 99.8%** of optim — primary leftover wall (Audit 9B).
- **Harness hazard.** Every `bench_128m_ab.sh` mode began with
  `[ -x "$BIN" ] || build`, which runs a **stale binary** after a source change
  — the same failure the funnel explicitly guards against. One `fwd` run was
  invalidated before it was caught. All modes now call `ensure_build`
  (always-incremental), and the profile greps no longer abort the script under
  `pipefail` when a section is missing.
- **Decision:** ship Audit 7 + Audit 8 FA_FWD_FAST defaults. Remaining ceiling
  is modest (muon + MLP sandwich + glue floor) — not another 2×. Architecture
  truth locked in `optimization_map.md` Audit 9.
