# SAM memory cross-attention: which kernel runs (2026-11-01)

Rules: `rules.md` in this directory, committed as `c47986b` **before** any run.
Harness: `ojas-metal/examples/attention_flash_bench.rs` at `dac0b5b`.

## Outcome in one line

**Q1 and Q3 are settled; Q2 (timing) did not run.** The SAM memory
cross-attention shape takes the **flash** kernel, and a biased call takes
**split-K**, not the scalar kernel — both as predicted in `rules.md`. No
millisecond is reported, because no clean or even lock-eligible window existed.

## Q2 first: the timing run did NOT happen

`mac_heavy.sh` was given a 900 s cap (`MAC_HEAVY_MAX_WAIT_S=900`) so it would
fail fast rather than take its 4 h default. It **never acquired the lock**:

```
21:48:33Z waits without the lock: machine busy (load 18.20, max 12; 1423 processes)
21:53:35Z waits without the lock: machine busy (load 17.87, max 12; 1441 processes)
21:58:36Z waits without the lock: machine busy (load 23.25, max 12; 1435 processes)
22:03:37Z stayed busy for the whole wait: machine busy (load 28.22, max 12;
          1465 processes); 'ojas-sam-xattn-route' did NOT run
```

Exit **76** ("stayed busy for the whole wait"), so the job never ran, and it
never ran unlocked. The load was 18–28 throughout, against the lock's cap of 12
and this harness's own cleanliness bar of 6.0. The machine was busy with other
sessions' `cargo` work (a concurrent `cargo build --release -p devmap-cli` with
20 `rustc` processes was observed).

Per `rules.md`, this is a legitimate outcome and **no stand-in is substituted**.
`2026-10-27-attention-mpp` already recorded the same failure mode on this
machine — six attempts and a three-hour wait at loads of 6.3 to 111, verdict
`repeat` throughout.

**Nothing in this file is a timing or rate claim.** The 42.6 ms / 4.5 TFLOP/s
baseline quoted elsewhere remains SpikeTrack's `--profile ops` figure, measured
in a different harness at unrecorded load, and is **not** reproduced here.

## Q1 — the SAM shape runs the flash kernel

The call is `[b, 8, T, 32]` with `AttentionOpts::default()`
(`SpikeTrack/crates/st-model/src/tracker/memory_attention.rs:20-22, 150,
163-173`): head dim **32**, 8 heads, no bias, no key lengths, non-causal.

Three independent lines of evidence, none of which depends on machine load:

1. **Router source** [VERIFIED by reading]. `flash_cross_covers`
   (`ojas-metal/src/backend.rs:1888`) admits it: d=32 ∈ `FLASH_CROSS_HEAD_DIMS`,
   `!causal`, `window == 0`, `bias.is_none()`, `!has_lens`, both lengths > 0,
   planes within i32. `flash_groups` (`:1878`) = `ceil(5184/32) · bh` =
   162 · 8 = **1296 ≥ FLASH_MIN_GROUPS (384)**. `AttnExtra::scalar_only()`
   (`link.rs:440`) is false — it is only `radius != 0 || dead_rows`, and
   `dead_rows` requires key lengths with `q_len == kv_len`.
2. **A committed test at the identical geometry, re-run today** [VERIFIED by
   running]. `ojas-metal/tests/attention_flash.rs:465`
   `attention_forward_beyond_the_pad_ratio_runs_the_flash_kernel` uses
   `Case::new(1, 8, 8, 5184, 15584)` with `const D = 32` — B=1, H=8, Tq=5184,
   D=32, the SAM geometry — and asserts `Ran::FLASH` **and 6 dispatches**
   ("3 scans, the kernel, 2 output checks"), against split-K's 7. It passed on
   this machine today: `test result: ok. 1 passed; … finished in 12.70s`.
   TensorOps is available here, so it took the `Ran::FLASH` branch and not the
   `Ran::SPLITK` one — `ojas-metal/examples/alloc_bench.rs:500` carries an
   unconditional `assert!(main.has_tensorops())` and the committed
   `bench/results/2026-10-27-alloc/smoke-output.txt` contains the `# backend
   24 GiB: …` line printed immediately after it.
3. **A committed statement of the rule at the SAM key count** [COMMITTED CLAIM].
   `ojas-metal/tests/attention_tiled.rs:574-577`, in the `tk ∈ [5200, 10400,
   15616, 36544]` loop: "`attention_forward` takes the pad route up to
   Tk = 2.25 Tq (11664 keys here); beyond it, at head dim 32 with these 1296
   threadgroups, it runs the flash kernel". 36,320 and 36,400 are both far
   beyond 11,664.

Nothing in the routing depends on `Tk` beyond `kv_len > 0` and the i32 plane
bound (36,400 · 32 = 1.16 M, far inside it), so the step from Tk = 15,584
(asserted, re-run today) to Tk = 36,400 carries.

**Not verified:** the assertion was not re-run at Tk = 36,320 itself; that was
the row the lock refused.

## Q3 — a biased call takes split-K, not the scalar kernel

Prediction confirmed. `ojas-metal/tests/attention_flash.rs:621`
`the_calls_the_route_does_not_cover_are_planned_and_run_elsewhere` asserts it by
**observed kernel identity**, at d=32 with a query length sized to clear the
threadgroup floor (`rows = 32 · ceil(384/8) = 1536`):

| call | planner | kernels encoded |
|---|---|---|
| no bias, no lens | `"flash"` (with TensorOps) | `Ran::FLASH` |
| **with a bias** | **`"splitk"`** | **`Ran::SPLITK`** |
| **with key lengths** | **`"splitk"`** | **`Ran::SPLITK`** |
| causal | `"scalar"` | `Ran::SCALAR` |
| head dim 16 | `"splitk"` | `Ran::SPLITK` |
| head dim 48 | `"scalar"` | — |

Mechanism [VERIFIED by reading]: `xattn_splits`
(`ojas-metal/src/backend.rs:1912`) returns `None` only for causal, windowed,
`d ∉ {16, 32}`, or an empty length — it **never inspects the bias or the key
lengths**. With `scalar_only()` false, a biased d=32 call therefore lands on
split-K.

For the detector row specifically (`[1,8,201,32]` over `[1,8,5184,32]`, bias
`[1,8,201,5184]`): flash is excluded twice — `bias.is_none()` fails *and*
`flash_groups = ceil(201/32)·8 = 56 < 384` — and split-K takes it.

**Not verified:** this test was not re-run today (the lock refused); it is a
committed assertion in the suite. Its shapes are synthetic (Tk = 64), sized
from the floor; the detector row's own shape was not separately asserted.

### A note that matters for the original brief

The same test asserts **key lengths → split-K** at d=32. So even on the brief's
premise that SAM's memory attention sets `key_lens`, the call would have gone to
**split-K, never to the scalar kernel**. The brief's "three gates each send it
to scalar" fails on the head dim *and* on this, independently.

## What this run changes, and what it does not

- **Settled:** the routing. There is no routing gap at the SAM shape, so the
  briefed work — widening flash to cover bias and key lengths at head dim 256 —
  would not touch this call.
- **Untouched:** `ATT_BQ` and every other kernel constant. Raising it was
  considered and **withdrawn**: it lives in the shared
  `ojas-metal/kernels/attn_bwd_tiled.h:62-65` and is compiled into the causal
  flash forward at DM 16–256 (the DM=256 variant is the Qwen3.5 training
  forward) and every general backward kernel, so it cannot be justified by one
  SAM number — least of all a contended one. It also has a correctness trap:
  with `ATT_THREADS = 128` and `r = tid / ATT_TPR` (`ATT_TPR = 4`,
  `ojas_backend.metal:2677`), `r` only reaches 32, so `ATT_BQ = 64` would leave
  query rows 32–63 never max-reduced or accumulated — **silently wrong output**,
  not merely slower — unless `ATT_NSG` moves 4 → 8 with it (and host
  `ATTN_THREADS` / `ATTN_ROWS`, `ojas-metal/src/device.rs:139,144`, used at
  `device.rs:2605-2606` to size the grid).
- **Still open:** whether the flash kernel has headroom at d=32. No rate claim
  is possible from this run, and the ceiling that would bound one — f32 GEMM
  throughput **on the matrix units** — is measured nowhere in this repo. The
  f32 linear control is tessl's exactness-constrained `GemmOperands::ExactF32`
  (`ojas-metal/src/device.rs:2266-2271`), not the `matmul2d` path the flash
  kernel uses, so it cannot bound it.
- **Next lever, if the call later proves bandwidth-bound:** bf16 K/V (ojas
  feature 18, not landed), not the tile size.

## Reproducing

```sh
CARGO_TARGET_DIR=/Users/bharath/Code/research/ojas-xattn-probe-target \
  cargo build -p ojas-metal --release --example attention_flash_bench
```

then, from `/Users/bharath/Code/research/ojas`, under the lock:

```sh
MAC_HEAVY_MAX_WAIT_S=900 bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh \
  ojas-sam-xattn-route /Users/bharath/Code/research/ojas-xattn-probe-target/release/examples/attention_flash_bench 5 25 m=36320
```

The binary must be built from the main checkout: this worktree cannot build,
because ojas's sibling path dependencies (`tessl = ../../tessl`, gusset
`../../../devtools/gusset`) do not resolve from `.claude/worktrees/<lane>/`,
and symlinks cannot fix it (cargo canonicalizes; a sibling crate linked under
the ojas tree is captured into ojas's workspace, and tessl depends back on
`../ojas/ojas-core`). Both checkouts were clean at `dac0b5b`.
