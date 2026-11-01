# SAM memory cross-attention: which kernel runs (2026-11-01)

Rules: `rules.md` in this directory, committed as `c47986b`. **Read "What was
and was not pre-registered" below before treating any prediction here as one.**
Harness: `ojas-metal/examples/attention_flash_bench.rs` at `dac0b5b`.

## Outcome in one line

**The routing has two regimes, and which one SAM is in depends on the number of
memory frames.** At one or two memory frames the call runs tessl's tiled
`encoder_attn` pad route; at three or more it runs the **flash** kernel. The
profiled shape (36,320 keys, seven frames) is in the flash regime. A biased
call takes **split-K**, not the scalar kernel. **Q2 (timing) did not run.**

## What was and was not pre-registered

The actual sequence, local time, 2026-11-01:

| time | event | evidence |
|---|---|---|
| 14:37:39 | the Q1 test run finished (`1 passed … 12.70s`) | `main_flash_test.log` mtime |
| 15:18:14 | `rules.md` committed (`c47986b`) | `git log --date=iso-local` |
| 15:48:33 | bench binary ready | `mac-heavy-wait.log` |
| 15:48–16:03 | `mac_heavy` waited and gave up, exit 76 | `mac-heavy-wait.log` |
| 16:07:28 | this summary first committed (`4d02dc0`) | `git log --date=iso-local` |

So:

- **Q1 was not pre-registered.** Its "prediction" was written forty minutes
  after a run that had already confirmed it. It was pre-stated *knowledge*, not
  a prediction, and it cannot count as evidence that the answer was not chosen
  after the fact. `rules.md`'s own wording ("before the first *timed* run") is
  literally true only because no timed run ever happened. An earlier version of
  this file said `rules.md` was committed "before any run"; that was wrong.
- **Q2 was genuinely pre-registered**, and never ran.
- **Q3 was pre-stated from source**, with no run before or after it.
- **The pre-registration also missed the pad-route regime below.** Q1 asked
  "which kernel does the dispatcher encode" as though there were one answer for
  the SAM shape. There are two, selected by `Tk / Tq`. `rules.md` is left
  unedited — patching a pre-registration after the fact is the failure being
  recorded here, not its repair.

## Q2: the timing run did NOT happen

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

## Q1 — two regimes, selected by the memory frame count

The call is `[b, 8, T, 32]` with `AttentionOpts::default()`
(`SpikeTrack/crates/st-model/src/tracker/memory_attention.rs:20-22, 150,
163-173`): head dim **32**, 8 heads, no bias, no key lengths, non-causal.

### The pad route is tested first

[VERIFIED by reading]. SpikeTrack calls `attention_forward`, the inference
form. That method checks `routes_to_encoder_attn`
(`ojas-metal/src/backend.rs:1706`, predicate at `:1712`) at **`:2669`, before
it reaches `attn_forward` and the flash ladder at all**, and hands the call to
tessl's tiled `encoder_attn` when every term holds. For this call they all do
except the length ratio: head dim 32 is in `ENCODER_ATTN_HEAD_DIMS` (`:1671`),
there is no bias, causal flag, window, radius or key length, and the planes fit.
What decides it is

```rust
dims.kv_len as f64 <= max_ratio * dims.q_len as f64   // ENCODER_ATTN_MAX_PAD_RATIO = 2.25, :1688
```

so with `Tq = 5184` the boundary is **11,664 keys**.

### Which frame counts land where

SAM's key count is `Tk = F · 5184 + P` for `F` memory frames and `P` object-
pointer tokens (`memory_attention.rs:118-127`: `m != mi + p` and
`mi % tokens != 0` are refused, `frames = mi / tokens`, `tokens` = 5184 at
1008×1008). Object pointers arrive 16 tokens at a time — ojas's own comment at
`backend.rs:1684-1686` states the ratio as `F + P · 16 / 5184` — and the
profiled 36,320 is `7 · 5184 + 32`, seven frames and two pointer groups.

| memory frames | `Tk` | `Tk / Tq` | kernel |
|---|---|---|---|
| 1 | 5,184 + P | 1.00 … | tessl `encoder_attn` (pad route) |
| 2 | 10,368 + P | 2.00 … | tessl `encoder_attn` (pad route) |
| 3 | 15,552 + P | 3.00 … | **flash** |
| 7 (profiled) | 36,320 | 7.006 | **flash** |

At `F = 2` the pad route holds for any pointer count up to `P = 1,296` tokens
(81 groups), so every realistic two-frame call takes it. At `F = 3` the ratio
starts at 3.00, already past 2.25, so no three-frame call takes it. There is no
`F` for which the boundary is in doubt — which is what the 2.25 bound was chosen
for (`:1680-1687`).

**One caveat on the pad regime** [VERIFIED by reading]: `encoder_attn_forward`
returns `Ok(None)` when the budget has room for the output but not for the pad
route's scratch (`:1068-1072`), and the caller then falls through. So an `F ≤ 2`
call on a tight budget *does* reach the flash ladder, and `flash_cross_covers`
would admit it. The pad route is the kernel for `F ≤ 2` whenever the budget
allows it, not unconditionally.

### Evidence that the flash regime runs the flash kernel

1. **Router source** [VERIFIED by reading] — the primary evidence.
   `flash_cross_covers` (`ojas-metal/src/backend.rs:1888`) admits the call:
   d=32 ∈ `FLASH_CROSS_HEAD_DIMS`, `!causal`, `window == 0`, `bias.is_none()`,
   `!has_lens`, both lengths > 0, planes within i32. `flash_groups` (`:1878`) =
   `ceil(5184/32) · bh` = 162 · 8 = **1296 ≥ FLASH_MIN_GROUPS (384)**.
   `AttnExtra::scalar_only()` (`link.rs:440`) is false — it is only
   `radius != 0 || dead_rows`, and `dead_rows` requires key lengths with
   `q_len == kv_len`. And the pad route is excluded, because 36,320 > 11,664.
2. **A committed statement of the rule at the SAM key count** [COMMITTED CLAIM].
   `ojas-metal/tests/attention_tiled.rs:574-577`, in the `tk ∈ [5200, 10400,
   15616, 36544]` loop: "`attention_forward` takes the pad route up to
   Tk = 2.25 Tq (11664 keys here); beyond it, at head dim 32 with these 1296
   threadgroups, it runs the flash kernel". That sentence states both regimes,
   and its loop exercises both sides of the boundary.
3. **A committed test at the three-frame shape, re-run today** — the run is
   [VERIFIED by running]; what it proves about the kernel is **[INFERRED]**.
   `ojas-metal/tests/attention_flash.rs:465`
   `attention_forward_beyond_the_pad_ratio_runs_the_flash_kernel` uses
   `Case::new(1, 8, 8, 5184, 15584)` with `const D = 32`. Its `Tk` is
   `3 · 5184 + 32` — exactly the **smallest flash-regime SAM shape**, which is
   why the test's own doc comment says it covers the case "where the pad route
   does not (`Tk` beyond 2.25 `Tq`)". It passed on this machine today:
   `test result: ok. 1 passed; … finished in 12.70s`.
   **It is not evidence of which kernel ran.** Its assertion is inside
   `if m.has_tensorops()` (`:479`): with TensorOps it asserts `Ran::FLASH` and 6
   dispatches, and without it asserts `Ran::SPLITK` and 7 — so the test passes
   on either branch and cannot fail on the question Q1 asks. That this machine
   has TensorOps is itself inferred, from `alloc_bench.rs:500`'s unconditional
   `assert!(main.has_tensorops())` and the committed
   `bench/results/2026-10-27-alloc/smoke-output.txt`, an October 27 artifact —
   not established today. A test that pins the kernel at this shape regardless
   of TensorOps does not exist.

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
`flash_groups = ceil(201/32)·8 = 56 < 384` — and split-K takes it. The pad route
is excluded too: `padded_ok` requires `q_len <= kv_len` *and* the ratio bound,
and 5184 / 201 is 25.8.

**Not verified:** this test was not re-run today (the lock refused); it is a
committed assertion in the suite. Its shapes are synthetic (Tk = 64), sized
from the floor; the detector row's own shape was not separately asserted.

### A note that matters for the original brief

The same test asserts **key lengths → split-K** at d=32. So even on the brief's
premise that SAM's memory attention sets `key_lens`, the call would have gone to
**split-K, never to the scalar kernel**. The brief's "three gates each send it
to scalar" fails on the head dim *and* on this, independently.

## What this run changes, and what it does not

- **Settled:** the routing, and that it has two regimes. There is no routing gap
  at the SAM shape in either of them, so the briefed work — widening flash to
  cover bias and key lengths at head dim 256 — would not touch this call.
- **New:** nobody had noticed the pad-route regime. A SAM tracker running on one
  or two memory frames is not exercising the flash kernel at all, so a flash
  change measured at seven frames says nothing about it, and the reverse.
  Any future attention measurement for SpikeTrack has to name its frame count.
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
`../ojas/ojas-core`).

**Checkout state** [VERIFIED by `stat` and `git status`]. The main checkout was
clean of modifications when the Q1 test ran (14:37:39) and when the binary was
built (15:48:33). By 15:51:45 and 15:52:42 a sibling session had modified
`ojas-oracle/python/gemma_ops_fixture.py` and
`ojas-oracle/fixtures/gemma_ops/golden.txt` — a Python fixture generator and its
golden text, neither compiled into the flash path — so the main checkout was
*not* clean when this summary was first written at 16:07:28, as an earlier
version of this sentence claimed. The Rust edits now present there
(`ojas-metal/src/backend.rs` and others, 16:44 onward) all postdate that commit.
This worktree was clean throughout.
