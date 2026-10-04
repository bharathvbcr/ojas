# Fable ruling: Mac inference optimizations after the 2026-10-03 measurements

Asked by session "Lappi model benchmarking" on the human's request, verbatim: "Ask fable and start
working on those optimizations." Advisor: Fable (claude-fable-5-1), through a read-only lane, with the
benchmark report in this folder plus a follow-up message giving the `sha2` "asm" finding. The ruling
is recorded verbatim from its heading on.

# Fable ruling: Mac inference optimizations after the 2026-10-03 measurements

Read-only lane: nothing written, launched or benchmarked. Builds on the 10-02 ruling; items there not named here still stand.

**Navigation.** DevMap Lappi gen 3111, fresh: `readonly_decode` callers graph-confirmed (`answer_generic` + 4 tests in `tests/readonly_slot_isolation.rs`); `PrefixState::digest` callers returned empty with `walk_incomplete`, and `DecisionBackend::decode_slot` had "no indexed traversal start" — so every other file:line below is **rg/Read-found, not graph-confirmed**. GitPulse Lappi: all facets ok; 2 live Claude sessions; main dirty 15 files (backend.rs, parity.rs, tokenizer.rs, service.rs, serve.rs, Cargo.lock among them); `Cargo.lock` collides across 3 worktrees. GitPulse tessl: collisions facet `ok:false` (2 worktrees **not scanned**, not clean); main dirty 27 tracked paths incl. gemm.rs, runtime.rs, nn.rs, qwen35.rs, flash_attn_rows.metal, qwen35_attn.metal. No ListAgents in this lane.

## (a) Order, go/no-go, and the digest's necessity

| # | Item | Verdict | Cheapest decisive measurement first |
|---|---|---|---|
| 1a | Parallel host SHA-256, v1 bytes identical | **go, now** (zero dependency) | none needed: arithmetic below closes it; the A/B row is the measurement |
| 1b | `sha2` feature `asm` | **go, after the human's yes** (new crate `sha2-asm` 0.6.1) | clean build + `sysctl hw.optional.arm.FEAT_SHA256` + CPU throughput ≥2.5 GB/s |
| 1c | Device digest kernel (v2 format) | **deferred** | revisit only if digests exceed 5% of a decision after 1a+1b |
| 2 | Batched passes / `decode_slots` | go, **after** item 1 and after backend.rs's owner lands | `decision.rs` arm `passes=sequential\|batched`: Δ ms and max \|Δlogprob\| |
| 4 | 8K attention | **measure, do not write a kernel** | tessl's own layer bench at 8K on today's tree vs the 09-28 6,213 tok/s |
| 3 | Short-prompt floor / fusion / int8 | fusion: later, tessl only; **int8: no** | `--decision T=16`: the floor vs 3.76 GB ÷ measured GPU bandwidth |
| 5 | ojas Mac inference | **no** | — |

**Is the per-decode digest necessary?** It is the runtime's falsifiable form of hardening §6 (`docs/hardening.md:126`): `readonly_decode` (`answer.rs:548-588`) hashes the `StateBuffer` before and after every decode and puts the result in the answer as `SlotIsolationCheck::Ran{state_hash}` (answer.rs:573) — the tri-state the whole "a check that could not run never looks like one that passed" rule rests on. On qd-metal the runtime's `StateBuffer` is a 52-byte `StateRecord` (backend.rs:81-114), so the cost is entirely the backend recomputing the device digest after each decode (backend.rs:488, 661-665) [V]. tessl enforces read-only structurally (`StateIn::Snapshot` refused as `state_out`, separate suffix K/V — model.rs:24-32), so the digest catches the accidental write, which is what the contract says it is for.

Line between engineering and human: **how** the hash is computed (threads, hardware instructions) is engineering — the bytes are identical. **What** is hashed (device v2 format, a non-SHA hash), **how often** (once per request, sampled), or **whether** `Ran` may be reported without the comparison are contract changes (hardening §6, `docs/schema-api.md:358-372`, `tests/answering_procedure.rs`, `readonly_slot_isolation.rs`) — the human's. My recommendation: do not ask. After 1a+1b the three digests cost ~5-10 ms of a 409-token decision (below) and there is nothing left to decide.

## (b) Item 1: the measurement and the fix

**The hash is the whole phase [V, computed from the log].** 221.5 MB ÷ 0.59 GB/s = 375 ms vs 382 ms measured per digest at 8K (1145/3); 23.4 MB ÷ 0.59 = 40 ms vs 40.6 at T=131. `decision.rs:418-441` synchronizes after the prefill and reads logits back before each digest, and `map_host` (tessl `tensor.rs:165-193`) is zero-copy, so the 48 × (acquire + `commit_m4`) cost <2%. That retires the open half of gap 929: no split instrumentation is needed. The cause is `sha2` 0.10.9 compiling its aarch64 SHA2-instruction backend only under `feature = "asm"` (`sha2-0.10.9/src/sha256.rs:19-22`) and the workspace declaring `sha2 = "0.10"` bare (`Cargo.toml:36`) [V]: the software compressor on one thread. 0.59 GB/s is exactly that.

**Design constraint for 1a [V].** `acquire_access` is one exclusive `AtomicBool` (tessl `runtime.rs:606-618`): a second live `HostMapping` returns `Err("runtime busy")`, and `GpuRuntime` is `!Send` (runtime.rs:411-418). So: on the Worker thread, map one buffer at a time, copy into an owned scratch (`Vec<u8>` reused across the three digests; `extend`, no zeroing), drop the mapping; then `std::thread::scope` over owned slices, buffers largest-first, fan-out bounded by `available_parallelism` (18 here: 6P+12E [V sysctl]). Copy cost: ~120 GB/s single-thread on this machine (ojas `docs/adaptive-resources.md:87`, by report) → ~2-4 ms at 8K, <0.5 ms at task lengths. The v1 digest bytes and `backend.rs` are unchanged.

**Expected [I].** T=409 (30 MB, largest buffer ~1 MB): per digest 52 → ~6 ms (1a) or ~9 ms (1b alone) or ~2-3 ms (both); decision 286 → ~150 / ~160 / ~140 ms. T=8192 (largest buffer 16.8 MB bounds a thread): 382 → ~40 / ~70 / ~15 ms; decision 3.38 → ~2.4 / ~2.5 / ~2.3 s. 1c would buy a further ~2% — not worth a tessl commit into a tree the ojas session holds.

**Is a faster host path acceptable under the contract?** Yes for 1a and 1b: SHA-256 is SHA-256, `state_hash` and `StateRecord.digest` are byte-identical, the GPU tests (`tests/gpu.rs:55-100,131-180`) are unchanged in meaning. A different hash is not (contract-visible). Parallelism inside one buffer would need a tree format (v2) — do not.

**1b also covers every other `qd_runtime::sha256` caller automatically** with identical output: weight hashing at load (`weights.rs:81`, `model.rs:370-379`, inside the 6-thread `LOAD_PARALLEL_LAYERS` scope) — ~1 s of the 4 s load [I] — and `code_that_ran`, `prompt_digest`, the ledger chain. Oracle: the release's `weight_hash` still equals `expected_identity.weight_hash` (today's log already shows `a68f19bc…`). Caveats: on aarch64 the `asm` cfg selects sha2's own `aarch64.rs`, and `sha2-asm` is pulled only as a dependency — whether it **builds** on aarch64-apple-darwin is [U]; and the `Cargo.toml`/`Cargo.lock` edit must be sequenced behind the session that holds `Cargo.lock` dirty.

**Fail-first.** For a throughput change the fail-first artifact is the interleaved A/B `quick` row with a stated bound (`digest_ms` ≤ 1/4 of serial at every T). The unit test (helper == serial `sha256` on 48 unequal slices incl. empty and 1-byte) and `gpu_digest_paths_are_bit_identical` (sibling of `gpu_embed_paths_are_bit_identical`, gpu.rs:194) pass on both sides by construction — say so, do not dress them as fail-first. Once the row is in, **delete the serial path**; `EmbedPath::Device`, measured identical and no faster (gap 931), should go the same way.

## (c) Ownership and lanes

- **Item 1a** — `crates/qd-runtime/src/lib.rs` (new `sha256_slices_parallel` beside `sha256`, lib.rs:140; canonical owner of hashing), `crates/qd-metal/src/model.rs:166-177`, `src/decision.rs` (arm `digest=serial|parallel`), `tests/gpu.rs`. None is dirty [V]. **backend.rs untouched.** Agree with: the lead; message the 2 live sessions first.
- **Item 1b** — `Cargo.toml:36`, `Cargo.lock` (dirty, 3-worktree collision). Agree with: **human** (new dependency), the Cargo.lock holder.
- **Item 2** — qd-runtime `backend.rs:211-230` (`decode_slots`, default two-call impl), `answer.rs:293-318`, `reference.rs`, `ensemble.rs`, `tests/answering_procedure.rs:558-572` (count **queries**), `tests/readonly_slot_isolation.rs`, `docs/schema-api.md:364-368`; qd-metal `backend.rs` (**another session's — wait for it to land**), `model.rs` (batched `run` exists), `decision.rs`. The isolation check becomes one per batched call covering both queries — write that into the pin. gpu.rs bounds batch-vs-batch-1 at 1e-4 (gpu.rs:98), not 1e-5: the new pin's bound is set from the measured max |Δlogprob|, and any non-zero delta re-runs `qd-metal-parity` with the batched path (thresholds untouched, rule 2). Note `passes 61+61` holds only because permuted options are the same strings; `attn_prefix_rows_varlen` (tessl qwen35.rs:2340) is the general path.
- **Items 3-4** — canonical tessl only (rule 6); the ojas session holds gemm.rs/nn.rs/qwen35.rs/runtime.rs and the attention `.metal` files dirty: nothing lands there without them. Lappi consumes.

## (d) What not to do

- **No int8 weight-only.** It changes numerics (parity re-run), changes the release's `weight_hash` identity and qd-export format (human), and buys ~5-7 ms of a ~17 ms floor that is weight-bandwidth-bound: 3.76 GB at ~240 GB/s ≈ 16 ms [I; bandwidth by report, ojas `docs/metal-deferred-faults.md:248`]. After items 1-2 the floor is ~15% of a task-length decision; measure `--decision T=16` before any of it.
- **No new head-dim-256 attention kernel.** tessl already has `flash_attn_rows`/`attn_prefix_rows` at 256 (qwen35.rs:2144-2305). What exists is a **discrepancy**: 09-28 reported 1,318 ms / 6,213 tok/s at 8K on tessl's layer bench (random weights, by report); today qd-metal's state-kept prefill measures 2,133 ms / 3,797 tok/s on real weights [V]. Different harness, different tree (29 dirty paths). Two runs discriminate: tessl's bench on today's dirty tree (~6.2k → qd-metal's prefill path is the gap; ~3.8-4.1k → a tessl regression, then clean HEAD worktree vs dirty — no `git stash` on the shared checkout). 8K is the needle-suite size, not a product length; task lengths come first.
- **No ojas Mac inference path.** ojas-qwen35 and ojas-qwen35-cuda are training-step providers, ojas-infer is nanolab GPT, ojas-metal is a tessl-hosted backend for ojas's own graphs [V, Cargo.toml descriptions]. qd-metal on tessl is Lappi's one Mac inference engine; a second would be a duplicate owner. ojas's role here is CUDA and training — and its session's custody of tessl's dirty tree.
- No sampling or once-per-request digest; no second-pass-only digest (it would report `Ran` for slot 1 without having checked it).

## (e) First steps for the implementing session

0. In `/Users/bharath/Code/research/Lappi-decision`: `ListAgents`; `SendMessage` to both live sessions claiming `crates/qd-runtime/src/lib.rs`, `crates/qd-metal/src/model.rs`, `crates/qd-metal/src/decision.rs`, `crates/qd-metal/tests/gpu.rs`; append the GAP ids below.
1. Add the helper + unit test in qd-runtime, the `DigestPath` on `Model` with the `digest=` arm in decision.rs, the gpu.rs bit-identity test. CPU: `CARGO_BUILD_JOBS=2 cargo test -p qd-runtime -p qd-metal` (GPU tests stay `--ignored`).
2. Under the lock, one job: `bash tools/mac_heavy.sh lappibench cargo run --release -p qd-metal --bin qd-metal-bench -- --decision T=131,409,770,2048,8192 k=4 --arms digest=serial,digest=parallel --snapshot <p4-v4-avg-masters-v1 path> --ledger ledger/mac-qd-metal-2026-10-03.jsonl` (flag spelling per bench.rs:6-7 and decision.rs:153; the `--snapshot` form is what `--decision` takes, gap 993). Bound: parallel `digest_ms` ≤ serial/4 at every T, arms bit-identical.
3. Ask the human for `sha2` `asm`; on yes, sequence the Cargo edit behind the Cargo.lock holder, re-run step 2 with arms `digest=serial,digest=parallel` on the new build, then delete the serial path.

GAP ids to append: GAP-QDM-DIGEST-SHA2-SOFT-BACKEND-NO-ASM-FEATURE-2026-10-03; GAP-TESSL-HOST-LEASE-EXCLUSIVE-BLOCKS-PARALLEL-MAPPING-2026-10-03; GAP-QDM-8K-PREFILL-3797-VS-REPORTED-6213-TOKS-2026-10-03; GAP-FABLE-INFER2-NAVIGATION-2026-10-03 (digest/decode_slot callers not graph-resolved; tessl collisions facet unscanned).
