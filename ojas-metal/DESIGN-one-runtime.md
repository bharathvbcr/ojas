# One tessl runtime per process: design and prototype

Branch `gp/ug-one-runtime`. Line numbers are at the base commit c389342
unless a name is given.

## Problem

Four production `GpuRuntime::new` sites give one process four queues, four
buffer pools, four pipeline caches and four residency sets:

| Site | Owner | Pool cap it sets |
|---|---|---|
| `ojas-metal/src/device.rs:576` (`Worker::open`) | the `MetalBackend` device thread | `min(1 GiB, budget/4)` (`pool_cache_cap`, device.rs:578) |
| `ojas-metal/src/gpu.rs:330` (`Session::open`) | the tiny Metal training step | 64 MiB |
| `ojas-qwen35/src/step.rs:399` (`Qwen35Step::open`) | the Qwen3.5 whole-step provider | tessl default 2 GiB |
| `gemma-metal/src/kernels.rs:816` (`GpuRuntime::new_inference`) | gemma-metal (separate workspace) | tessl default |

They share only the process-wide `MTLDevice.currentAllocatedSize`
(`ojas-metal/src/memory.rs:12-14`), so each runtime's wait and cap logic sees
the others' bytes as noise. `GpuRuntime` is `!Send`/`!Sync`
(`tessl/src/runtime.rs:1023-1030`), so "share one `Arc`" across threads is
not available; ojas-metal keeps the runtime on one device thread and lets
handles cross as `u64` ids over an mpsc link (`link.rs`, `device.rs:553`).

## What the prototype changes (verified to compile: NO, see Status)

The smallest move that puts `Qwen35Step` on the backend's runtime is to send
the provider's code to the device thread instead of sending the runtime out.

1. `ojas-metal/src/link.rs`: `Cmd::Run(RunBox)`, with
   `RunBox(Box<dyn FnOnce(&dyn Any) -> Res<()> + Send>)`. link.rs builds
   without tessl (the Linux stub), so the runtime travels as `&dyn Any`.
2. `ojas-metal/src/device.rs`: `Worker::run_provider`, reached through
   `Worker::command` like every other op, so the existing panic catch, poison
   checks and `lost_if_poisoned` apply. It settles and scans the device and
   reports a pending fault first (as `Cmd::Sync` does), turns `async_encode`
   off for the closure (the state `Qwen35Step` ran in on its own runtime;
   the worker keeps it on), runs the closure with `&self.rt`, then waits idle
   and restores the flag.
3. `ojas-metal/src/backend.rs`: `MetalBackend::with_runtime(f)`, the typed
   wrapper (macOS + `metal` feature only; the stub has no such method).
   `f: FnOnce(&Arc<GpuRuntime>) -> Result<(), OjasError> + Send + 'static`.
   Only `f` crosses threads; whatever it builds from the runtime is built,
   used and dropped on the device thread. Results leave through a channel
   the closure owns.
4. `ojas-qwen35/src/step.rs`: `Qwen35Step::open_on(rt, snapshot, numerics,
   precision, budget)` (and `runtime()`); `open` is now `GpuRuntime::new()`
   plus the shared `open_with`. `open_on` reserves the provider's device
   bytes (16 per parameter, `tower_tensors` shapes) from the one `Budget`
   before the first device byte and holds the `Reservation` for the
   provider's life. `open` charges as before (hidden rows only), so existing
   callers' budgets are unaffected.
5. `ojas-qwen35/tests/gpu_shared_runtime.rs` (4 tests, `#[ignore]` like every
   GPU test in the crate): `Arc::ptr_eq` of the step's runtime with the
   closure's; a standalone `open` has a different runtime; the budget is
   charged while open and released after; a budget one byte short refuses
   with `CapacityExceeded` and leaves nothing charged; parameter bits after
   forward+backward+AdamW on the shared runtime equal those on a standalone
   runtime; the backend still serves ops before and after.

Invariants kept: no feature flag; no `unsafe` (ojas-metal keeps
`forbid(unsafe_code)`: the closure is `Send`, the runtime never is); budgets
refuse before allocating; no silent fallback (a panic in `f` poisons the
backend exactly like any device-thread panic).

## Known limits of the prototype

- **One closure, not a handle.** A `Qwen35Step` cannot leave the device
  thread, so the whole micro-step has to run inside one `with_runtime`. A
  trainer that interleaves host work (the span head) between `forward` and
  `backward` needs the persistent handle below.
- **The worker's memory triggers do not see the provider.** `Book.alloc_since_wait`,
  `mem_cap` and `ws_limit` (device.rs:449-483, 776-802) count the worker's
  own allocations. The provider's 16 B/param is charged to the `Budget`, but
  it is not inside `mem_cap`'s residency arithmetic. Fine for a
  provider that waits for itself (the Qwen35 step does); not for two
  providers allocating concurrently, which cannot happen (one closure runs
  at a time, other ops queue behind it).
- **Pool cap.** The shared pool keeps the worker's `min(1 GiB, budget/4)`
  cache cap, the standalone step had 2 GiB. A freelist hit rate change, not a
  correctness change; measure before moving the cap (an A/B with the method in
  memory `tessl-gemm-ab-method`).
- **`async_encode` is forced off for the closure.** That preserves what the
  step ran under. Whether the step is faster under `async_encode(true)`
  (tessl/src/qwen35_train.rs:2103 describes the batched path) is a separate
  A/B.
- **Charge is a lower bound.** tessl rounds buffers to buckets
  (`runtime.rs:270-277`) and an f32 model keeps a packed head copy; both are
  above 16 B/param.
- **Test isolation.** ojas-metal's own tests hold a process-wide GPU lease
  (`gauge::lease`, device.rs:575, `cfg(test)` and crate-internal); the new
  integration tests do not, so they must run `--test-threads=1` like the
  sibling GPU tests, and are not verified under contention.

## Next steps, in order

1. **Persistent provider handle** (removes the one-closure limit).
   `Worker` gains `providers: HashMap<u64, Box<dyn Any>>` (it lives on the
   device thread, so `!Send` values are fine; ids come from `next_id`).
   ```rust
   impl MetalBackend {
       pub fn spawn_provider<P: 'static>(
           &self,
           make: impl FnOnce(&Arc<GpuRuntime>) -> Result<P, OjasError> + Send + 'static,
       ) -> Result<ProviderHandle<P>, OjasError>;
   }
   pub struct ProviderHandle<P> { id: u64, link: Arc<Link>, _p: PhantomData<fn() -> P> } // Send + Sync
   impl<P: 'static> ProviderHandle<P> {
       pub fn with<R: Send + 'static>(
           &self,
           f: impl FnOnce(&Arc<GpuRuntime>, &mut P) -> R + Send + 'static,
       ) -> Result<R, OjasError>;
   }
   // Drop posts Cmd::DropProvider { id }, like MetalBuffer posting Cmd::Free.
   ```
   `Pending` (holds tessl's `PendingStep`, `!Send`) moves into the same
   registry, and the `Qwen35Step` surface becomes a thin `Send` wrapper whose
   methods are `handle.with(|rt, step| step.forward(..))` returning
   `Send` summaries. Pending/bank identity checks (the card
   `gp-qwen35-host-neutral-crate` criterion 5) land here.
2. **`gpu::Session` (gpu.rs:330) on the worker's runtime.** It already is a
   second host layer over tessl; give `Session::open_on(rt)` and build it
   inside `with_runtime`/a handle. Its 64 MiB pool cap goes away with it.
3. **One charge point.** Replace the worker's `mem_cap`/`freed_since_wait`
   bridge (device.rs:92-116, 449-483, 761-772) with the bucket-accurate
   charge (`tessl::allocated_bytes_for`, runtime.rs:270-277), once tessl
   returns a typed allocation error instead of the strings
   `classify_alloc` parses (device.rs:536-551, pinned by a test that
   `include_str!`s tessl's source).
4. **gemma-metal (kernels.rs:816).** A separate workspace with an absolute
   tessl path and `new_inference()` (no timestamps). Sharing needs the worker
   to expose a flavor (`timestamps` off) or gemma-metal to accept a runtime
   argument; it is the last site and the one with a third-party-crate
   dependency set, so it is its own card.
5. **Process-global tessl state** (ICB/barrier `TriFlags`, `KERNEL_TRACE`:
   decode_icb.rs:28,204, ab_flags.rs:105) is already per-process, so one
   runtime makes it consistent; tests that rely on a runtime per test need
   the lease pattern above.

## Status

Written and self-reviewed twice by reading against the definitions it calls
(imports, signatures, visibility, `Cmd`/`Reply` matches, drop order, rustfmt
clean on every touched file); **not compiled, not run**. The owner deferred
all runs on 2026-10-31. Run, in the GPU window, with one build dir:

```text
bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh ug-one-runtime \
  env CARGO_TARGET_DIR=/Users/bharath/Code/research/ojas-ug-one-runtime-build \
  cargo clippy --manifest-path /Users/bharath/Code/research/ojas-ug-one-runtime/Cargo.toml \
  -p ojas-metal -p ojas-qwen35 --all-targets -- -D warnings
# then, same wrapper and env:
cargo test --manifest-path .../Cargo.toml --release -p ojas-metal --test with_runtime -- --test-threads=2
cargo test --manifest-path .../Cargo.toml --release -p ojas-qwen35 --test gpu_shared_runtime -- --ignored --test-threads=1
cargo test --manifest-path .../Cargo.toml --release -p ojas-qwen35 -- --test-threads=2
cargo test --manifest-path .../Cargo.toml --release -p ojas-metal --lib -- --test-threads=2
```

Tests that pin it: `ojas-metal/tests/with_runtime.rs` (one runtime per
backend across calls and clones, on the device thread; the closure's error
returned; a panic poisons) and `ojas-qwen35/tests/gpu_shared_runtime.rs`
(`Arc::ptr_eq` with the step's runtime; budget charged and released;
refusal one byte short; parameter bits equal to a standalone runtime).
