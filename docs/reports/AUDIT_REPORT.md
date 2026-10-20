# Deep audit → GitPulse backlog, 2026-10-16

Scope: ojas at local main 040e1ed (sibling tessl read-only). Read-only audit: no source, config or dependency was changed, and no build, test, GPU or device run was made. The only writes are GitPulse board writes (label `audit-2026-10-16`) and this file. Nothing was moved to done; no in_progress card was written; no card was merged or deleted.

Method: 8 Sonnet slice auditors (vision, core-autograd, cpu-simd, metal, wgpu, model, data-io, capi-go; at most 4 at a time) → an Opus verifier per slice (re-read every cited line, callers and upstream guards; verdict per finding; criterion-by-criterion check of review/done cards) → Fable as final advisor (grouping, fold-vs-new, waves, owner decisions). The orchestrator filed what Fable ruled.

User requirement carried into every vision card (2026-10-16): **training runs on the GPU.** Metal (ojas-metal over tessl) is the delivery target and the CPU op is the parity oracle. wgpu may refuse a new op with `Unsupported`, and CUDA must still compile. Every vision card needs:
- a Metal forward and backward kernel;
- a Metal-vs-CPU parity test;
- a measured Metal training step against torch MPS.

## 1. Findings by severity × dimension

These are the findings that survived verification (CONFIRMED or DOWNGRADED), 48 in all. The orchestrator classified each one into a dimension from the verified files.

| dimension | high | medium | low | total | findings |
|---|---|---|---|---|---|
| Capability (missing ops / paths) | 2 | 2 | 1 | 5 | VIS-A conv/pool/BN, VIS-C op set (high); VIS-B bidirectional SDPA, VIS-D image data (medium); VIS-H CE smoothing/weights |
| Correctness / numerics | 0 | 1 | 4 | 5 | DI-1 safetensors dtypes (medium); CS-1 libm in Exact loss, CA-2, CA-3, CA-5 |
| Memory accounting | 0 | 3 | 1 | 4 | CA-1 Budget check_room, DI-4 read_typed copies, MD-1 load_hf unchecked (medium); MD-7 |
| Error boundaries / GPU faults | 0 | 2 | 3 | 5 | M-1 Metal wait timeout, CG-1 Go call aborts process (medium); M-2, W-1, CA-6 |
| Defensive validation | 0 | 0 | 6 | 6 | DI-2, CA-4, M-3, MD-2, MD-5, MD-8 |
| Testability / CI | 0 | 1 | 6 | 7 | CS-3 no Miri/TSan (medium); DI-3, CS-5, CG-2, CG-3, CG-4, CG-5 |
| Concurrency | 0 | 0 | 2 | 2 | CS-4 Handoff Drop, DI-7 replace race |
| Performance / benchmarks | 0 | 0 | 3 | 3 | VIS-E upload drain, VIS-F frozen-input dW, MD-3 sampler sort |
| Modular architecture | 0 | 0 | 3 | 3 | VIS-G trainer bound to GPT spec, W-4 dead code, CG-6 superseded C pair |
| Portability (OS / endianness) | 0 | 0 | 2 | 2 | W-5 native-endian uploads, CG-8 cfg(not(unix)) never compiled |
| Docs vs code | 0 | 0 | 6 | 6 | VIS-I, DI-5, DI-9, CS-2, W-2, M-4 |
| **Total** | **2** | **9** | **37** | **48** | |

Raw findings: 63. Of those, 48 were planned, 8 were duplicates of existing cards (kept as evidence on those cards) and 7 were rejected (section 7). No finding was rated critical after verification.

## 2. Top 10 risks (Fable, ranked)

1. **G-V2 / E1.** Conv on the ExactF32 lane without epilogue fusion may land far below torch MPS. tessl's `gemm_epilogue` refuses exact f32 (tessl/src/gemm.rs:933-941), and ojas-metal Linear uses `GemmOperands::ExactF32`. If conv is too slow, the GPU vision goal needs a numerics-lane change or new tessl kernel work, and no card holds either yet.
2. **Vision chain timing.** G-V1 → G-V2 → G-V3 wait on three in-flight lanes (ft-fe7dcee6, ft-9c4ae3c8, ft-cab0fdf1), plus ft-ab2f1daf and the Python-policy peer for goldens. Any slip pushes the GPU-training goal back by a wave.
3. **G-BUDGET.** In default mode, one session's preflight can poison another session's trainer. `check_room` is a real reserve, parent first (ojas-core/src/budget.rs:150). The fix retires a documented, tested behaviour, so it needs E6 before the first edit.
4. **G-GPUFAULT.** A hung GPU never reaches `DeviceLost`: a 30 s stall repeats forever. A retry outside the repo of an in-place op applies it twice. M-2's Metal behaviour cannot be verified without fault injection.
5. **ft-69ef4ce1** sits in review, but nothing implements its core criteria (bf16 GEMM, bf16 attention). ft-3cc9ed3f's "bf16 lanes" widen to f32, so the board overstates the bf16 tier.
6. **ft-b4f9ce4e.** Local main is 66 commits ahead of origin, and no hosted CI has run on any of the local fixes. `--locked` and SHA pins (F-3902) change what "green" means.
7. **G-IOREAD.** About 2 GB of uncharged host transient per large tensor during the 2B load. This happens under the memguard that the LoRA-2B gate (ft-da2c4ab1) runs under.
8. **G-LIBM.** Exact-tier loss bits differ across hosts while README promises they are identical, and the Linux job skips the digest that would show it.
9. **F-3902 (CS-3).** `unsafe impl Send/Sync` over raw pointers in ojas-simd has no Miri or TSan gate, so a reordered store would ship green.
10. **F-BD63 (CG-1).** A public Go call with valid input (16,385 one-row micro-batches) aborts the whole process under a heap ceiling. G-TAPE CA-4 is the same fail-open class on the Tape: a wrong-shaped backend gradient is kept.

## 3. Cards filed and folded

### New cards (10)

| group | item_id | status | pri | sev | title (short) | wave |
|---|---|---|---|---|---|---|
| G-V4 | ft-f66c67c17b224ac3ac0712475bfda50e | ready | 2 | medium | Image data path: std-only MNIST idx / CIFAR-10 readers, Budget-bounded parsing, seeded f32 batch sampler, crop/flip, end-to-end Metal step | 1 |
| G-V1 | ft-4f11ce5a0bb138afdbd053c4bc5a2c9c | backlog | 1 | high | Vision op set on Backend + Tape with Metal kernels: ReLU/GELU/Tanh, bias-add, LayerNorm, softmax, reduce, slice/concat; dW-only linear backward; MLP step vs torch MPS | 4 |
| G-V2 | ft-96023b34df4725734a47fcd090dc1a8e | backlog | 1 | high | CNN on Metal: conv2d fwd/bwd, max/avg/global pool, BatchNorm; LeNet and ResNet-18-style steps vs torch MPS | 5 |
| G-V3 | ft-7f4f2d8768282c37f6bb9ab13d8da1c4 | backlog | 2 | medium | Bidirectional SDPA for ViT on Metal; ViT-tiny step vs torch MPS | 6 |
| G-BENCH | ft-64601da5134c2587295f380a0a2a405c | backlog | 1 | medium | Vision + sampler benchmark baseline: torch MPS rows (MLP/LeNet/ResNet-18/ViT-tiny, batch 128), vision goldens, top_k=None row, Metal upload-wait count | 3 |
| G-BUDGET | ft-54e8ce27306c13e66bf1241d240f09f9 | backlog | 2 | medium | Budget::check_room is a real parent-first reserve: preflight can refuse and poison another session (blocked on E6; label needs-your-words) | 1 |
| G-TAPE | ft-15ebd6b1a30f56c5cde400b6e79cb23c | backlog | 3 | low | Tape hardening: wrong-shaped gradient kept, Var identity, checkpoint panic safety, replay reads autocast at backward, bf16-rounded non-root seed | 7 |
| G-GPUFAULT | ft-757cb6b2a48cc5defb96da14b9536a31 | backlog | 2 | medium | GPU fault reporting fails open beyond OOM (Metal wait timeout, pending feedback, wgpu error queue, GEMM extent after alloc); repositories: tessl | 3 |
| G-IOREAD | ft-b301b6d677c41c66efee4fefc9d35607 | ready | 2 | medium | Typed safetensors reads hold two whole-tensor copies; Qwen3.5 load_hf never checks the Budget from the header | 2 |
| G-LIBM | ft-562b0add462330d980964e37a20249ba | ready | 2 | low | Exact-tier CE loss and GDN softplus call platform libm; README promises cross-machine bits | 1 |

### Folds into existing cards (10)

None of these folds changed status, priority or severity. Each one added the `audit-2026-10-16` label and resent the card's existing logs verbatim.

| fold | item_id | status | revision | what was added |
|---|---|---|---|---|
| F-2946 | ft-29462f23af79fa6d72825ecb7afd1944 | review (P0) | 6→7 | DI-1: criterion 7 (safetensors 0.8.0 dtypes F8_E4M3FNUZ, F8_E5M2FNUZ, C64); criteria 1, 2, 3, 5 marked MET; verification section |
| F-58E3 | ft-58e316a9bae421a3a2d71c26192aaea8 | backlog | 2→3 | MD-5: criterion 6 (`fuse_b`/`split_b` return Result; overflow and Budget checks) |
| F-B5E6 | ft-b5e64557f88bdb433700db571b7a78ec | ready | 3→4 | VIS-G: criteria 12–13 (generic param-group trainer runs a toy classifier); label vision |
| F-09ED | ft-09edeb325bef2537a7d72e212befdd16 | ready | 2→3 | DI-3: criterion 8 (NaN on the torch side of a parity check fails it) |
| F-9C13 | ft-9c13f47224dbb204fe9534c264500a5b | ready | 2→3 | MD-2, VIS-H: criteria 9–10 (resume finiteness scan; CE label smoothing / class weights); label vision |
| F-6964 | ft-6964cdbd2e41f71a813cbfb0d26d414b | backlog | 5→6 | W-4, CS-4: criteria 16–17 |
| F-B5FB | ft-b5fb19e6cf28f0c702a56bde4b44d198 | backlog | 3→4 | DI-2, DI-7, MD-7, DI-5: criteria 10–14 |
| F-BD63 | ft-bd6373d29fcdf322cda6ec69b02475a9 | backlog | 3→4 | CG-1, CG-6, MD-8: criteria 21–25 |
| F-6BA7 | ft-6ba746b48d26e4e6ae20bd81a6d49111 | ready | 4→5 | VIS-I, CS-2, M-4, W-2, DI-9: criteria 11–16. Criterion 6 (activation claims) is marked MET via ft-4f11ce5a, which corrects those doc rows |
| F-3902 | ft-3902a4b3b8d453cd1bc6abcc6f5aaeb8 | backlog | 4→5 | CS-3, CS-5, CG-2, CG-3, CG-4, CG-5, CG-8, W-5: added block (a)–(h) and criteria 16–23, covering Miri, TSan, `go test -race`, the allocator-api leg, SHA-pinned actions, `--locked`, a Windows check with a big-endian guard, and the ojas-simd accelerate tests |

### Annotated review, done and ready cards (13; no status changed)

| item_id | status | result |
|---|---|---|
| ft-6fb212ce8744aef3bdf4e5c6ff1fdf44 | review | MET 1, 3, 5–9. Criterion 2's code is MET but its 2B A/B exists only in the card log. Criterion 4 is partly met (local `LoraKind::suffix` table, not the shared one). New criterion 10: commit the dW-skip A/B |
| ft-531216ed802203374c90becae0bea3f2 | review (P0) | 1, 3 and 4 partly met. The provider leg of 3 is not met (row 2 `status: stopped`). 2 is MET with a deviation. New criterion 5: amend criterion 1 to the retired scope and fix README drift |
| ft-ffb271da0ac8ca501df59b5ecfa58c1b | review | All 8 MET (criterion 1 as amended 2026-10-11) |
| ft-3cc9ed3f01f08ad71b71f403438c3aa0 | review | 1–3 MET **by widening to a charged f32 copy per call, not a bf16 GEMM**. 4 and 6 MET. 5 partly met. 7 and 8 not met. New criterion 9: document the widening and replace stale line refs |
| ft-69ef4ce12ace9dd29f1a02caeff9cf25 | review | MET 1, 4, 6, 8, 10, 11. **NOT MET 2, 3 (the core bf16 GEMM and attention claim), 7, 9.** 5 cannot be verified here. New criterion 12. See decision E12 |
| ft-71d1d2e4457dab89073a8bda6f151016 | review | 1 and 2 MET (under the Fable deviation). 3 partly met: the A/B timings are only in the card log. Miri/TSan was one manual run; the CI gate for it is F-3902 criteria 16–17 |
| ft-9aaa63495eba176188b25f07d000b438 | review | All 3 MET. The card summary says "not merged", but 1b6748f is an ancestor of HEAD |
| ft-b4f9ce4e61238a5853e72ee85f56fb03 | review (P0) | MET 2, 3, 5, 9, 10. 6 and 7 partly met. **NOT MET 1 (no hosted green run; local main is 66 commits ahead of origin).** 4 and 8 cannot be verified here |
| ft-4b67b8d4d89d5a1b38ee4f1e48000dbc | review | 2 and 4 partly met: residual target-lane citations in non-doc files and `/Users/bharath` paths. A lost oracle (target-lane-framework/wsd_oracle.py) is reported to ft-7a6496c9. See decision E16 |
| ft-748ed8b545133b8290a2c4714916a5cf | done | All 14 MET. Caveats: 6's proof is prose only; 9's error class is delegated to ft-cab0fdf1 |
| ft-24d3795aef35957068687f3cade5bc6b | done | 1 MET. 2 MET at toy scale only. 8's Metal part cannot be verified here. 3–7 were not re-checked |
| ft-d1028a21b1cd1e0a736002e6bed093bb | done | **Criterion 3 ("zero-copy slicing for u16 bins") NOT MET as worded.** The code copies through a scratch buffer. ft-b5fb19e6 criterion 14 delivers the fix |
| ft-2c183b7103c95088ae62ea0da43aebc3 | ready | Code MET but bench not met: 1, 2, 9, 10, 11, 15. NOT MET: 3–8, 13, 14, 16, 18–21. 12 is partly met: wgpu and Metal large uploads are MET, but the Metal inline path is still an uncharged copy bounded by INLINE_UPLOAD_BYTES. 17 is conditional on an uncommitted gate profile. **Criterion 20 is not implemented, contrary to the 2026-10-09 re-audit**, and a correction was added under that paragraph |

Not touched:
- ft-69744e549b30e704ec3fe802634f7819 (review): another process was rewriting it (revision 158→179 within minutes). The verifier found all 4 criteria MET. Caveat: its Metal test returns early without a device (skip-not-fail).
- ft-67025371e041f942ff8379401cbffe75: CUDA, out of scope.

### Report-only notes on in_progress cards (no board write)

- **ft-fe7dcee6 (wrappers):** criterion 2's "hand-maintained parallel lists are gone" is NOT MET for Gated. gate.rs:168-664 is a hand-written list of 60 methods, and gate_adversarial.rs:96-114 only checks that `fn name(` appears. backend.rs:1553-1556 claims Gated is generated. Probe suppresses 43 names (ojas-model/tests/common/mod.rs:720-762). Recommendation: generate Gated with an `admit` hook so the 10+ vision methods do not each need a hand edit. G-V1, G-V2, G-V3, G-TAPE and F-9C13 wait on this lane.
- **ft-a0c303f6:** criteria 1–3 are already on main (1f92659, aaabf6b, 66ef93b), so the in_progress status is stale. Criterion 4 (Metal/CUDA equivalents) was not located.
- **ft-cab0fdf1:** G-GPUFAULT and F-BD63's capi lib.rs edits land after it. The audit found no wait-timeout handling anywhere in ojas.
- **ft-9c4ae3c8:** CA-5. A non-root loss seeded `None` under Autocast scales through `broadcast_scalar`, which rounds the seed to bf16 (bf16(1/3) = 0x3EAB). Seed through `scale_grad` until G-TAPE lands.
- **ft-02256073:** DI-2's `SafeTensors::open` sites inherit the F-B5FB fix; do not add a fifth plain `File::open`.
- **ft-ab2f1daf:** G-BENCH goldens and F-09ED's ojas-model/tests edits wait for this lane.
- **ft-71629756:** CUDA, out of scope. G-GPUFAULT's typed timeout variant should be the one ft-9bed58f0 criterion 11 adopts.

## 4. Verified-stale board claims

- ft-9aaa6349: the summary says "at 1b6748f, not merged", but it is merged.
- ft-a0c303f6: in_progress, but its code is on main.
- ft-69ef4ce1: review, but its core criteria 2 and 3 are not implemented. Its "today" text is also stale in the other direction (1, 4, 10 and 11 have landed).
- ft-2c183b71: the re-audit claimed criterion 20 (upload pops) was covered, and it is not. Line refs in 9, 10, 11 and 20 are stale.
- ft-3cc9ed3f: cited lines (backend.rs:1257, :1283; params.rs:304) are stale.
- ft-531216ed: criterion 1 still lists the save/load row that Fable retired 2026-10-15. ojas-qwen35/README.md:259 says a measured sequence length is unmeasured.
- ft-d1028a21 (done): criterion 3 is not met as worded.
- docs/pytorch-parity-plan.md:47 and docs/op-coverage.md:102 claim ReLU/GELU/Tanh, which exist nowhere in ojas (`rg -uu`). G-V1 criterion 9 fixes this.
- ojas-simd/src/arch.rs:9 gives the MSRV as 1.82, while Cargo.toml says 1.97 (F-3902 criterion 12).

## 5. Build order (Fable's waves)

Rules:
- At most 4 cards per wave.
- Cards in the same wave share no planned files. Directory entries count as every file under them.
- From wave 3 on, at most one of the four hot files per wave: ojas-core/src/backend.rs, ojas-autograd/src/tape.rs, ojas-metal/src/device.rs, ojas-metal/src/backend.rs.

| wave | cards | precondition |
|---|---|---|
| 1 | G-BUDGET (needs E6), G-V4, F-2946, G-LIBM | none |
| 2 | G-IOREAD, F-6BA7 | wave 1. Inside F-6BA7: docs/backends.md:107 and ojas-oracle/README.md wait for worktree ojas-w11-oracle; context.rs:207 waits for ft-cab0; bench/torch_rows.py waits for the Python-policy peer |
| 3 | G-BENCH, G-GPUFAULT, F-09ED | ft-cab0, ft-ab2f1daf, ft-a0c303's close, ojas-w11-oracle, the Python-policy peer (a live session with no board card; see E15) |
| 4 | G-V1, F-BD63, F-B5FB | ft-fe7dcee6, ft-9c4ae3c8, ft-02256073, wave 3 |
| 5 | G-V2, F-3902 | wave 4 |
| 6 | G-V3, F-58E3 | wave 5 |
| 7 | G-TAPE, F-B5E6 | wave 6 |
| 8 | F-9C13 (alone) | wave 7. The owner may swap it with wave 4, which shifts the vision chain by one |
| 9 | F-6964 (alone, last) | its own criterion 3: waits for every lane that edits its files |

Not waved: ft-2c183b71 (annotation only), and the open ready/backlog cards this audit did not touch. Those must be slotted against the same file matrix before they start:
- ft-301ddd4c collides with G-LIBM and with the test.yml waves.
- ft-353fa922 collides with waves 3–6.
- ft-da2c4ab1 collides with wave 3.
- ft-2bbced18 and ft-6a88d96d plan the hot files.

## 6. Owner decisions needed

**Blocking** (work cannot start, or starts wrong, without an answer): E1, E6, E9, E12, E15. Each entry gives Fable's recommendation first.

1. **E1 Metal numerics lane for vision ops (G-V1, G-V2). Recommendation: (a) for v1.** Options:
   - (a) ExactF32 with bias and activation as separate passes;
   - (b) admit the relaxed or bf16 lane so tessl's fused epilogue is reachable (dtype-policy change);
   - (c) ask tessl for an exact-f32 epilogue.

   Measure in G-V2 criterion 6, and revisit only for rows below 0.5× torch.
2. **E2 Conv layout.** NCHW externally (matches torch and the CPU oracle); Metal may pack internally.
3. **E3 U8 DType for image uploads.** Defer: widen on the host and measure upload cost in G-BENCH criterion 5 first.
4. **E4 Image decoder crates (png/jpeg).** None now, since a new dependency needs asking. MNIST idx and CIFAR-10 binary readers are std-only.
5. **E5 Vision kernel home.** A new ojas_vision.metal plus one build.rs line, rather than growing ojas_backend.metal (about 2,300 lines).
6. **E6 G-BUDGET (blocking). Recommendation: (a).** G-BUDGET retires a documented and tested behaviour (budget.rs:126-129, :908-913; adaptive-resources.md:205 C7b). Options:
   - (a) make `check_room` read-only, charge child-first, and change the assertion and the doc;
   - (b) keep the semantics and rely on ft-30a7d34f's prepaid mode.

   After (a), decide whether ft-30a7d34f stays.
7. **E7 Timeout policy (G-GPUFAULT).** Hold the error as a pending device error and report Ok for an in-place op that already committed. After N = 3 consecutive timeouts, report `DeviceLost`. Share the variant with ft-9bed58f0 criterion 11.
8. **E8 wgpu error attribution.** Sticky per context (poison until reopen).
9. **E9 Lockfile and toolchain (F-3902) (blocking). Recommendation: (a), and keep the branch with a leg after the rust-toolchain.toml pin.** Options:
   - (a) `--locked` everywhere, with GUSSET_REF/TESSL_REF pinned to SHAs;
   - (b) stay unlocked, with the reason recorded.

   E9b is a separate choice: keep the allocator-api branch with a `GUSSET_ALLOCATOR_API=1` leg, or delete it. Unpinned stable flips that branch on around 2026-11-09 (inferred). This interacts with ft-b4f9ce4e criterion 1.
10. **E10 SDPA API shape (G-V3).** One mask-mode argument on the existing methods, not three new methods. `slice`/`concat` in G-V1 are recorded copies for v1, not views.
11. **E11 Var identity (G-TAPE criterion 2).** Accessor-only refusal (`clear()` bumps a generation), not a breaking `Var` stamp.

    **E11b:** ft-b4f9ce4e criterion 1 needs the 66 local commits pushed. The audit cannot do that, and pushing is an outward action for you.
12. **E12 ft-69ef4ce1 status (blocking for board accuracy).** Move it out of review to ready with the stale text corrected; the audit does not move cards.
13. **E13 G-V1 scope.** Keep VIS-F (frozen-input dW-only backward) in G-V1.
14. **E14 VIS-E threshold.** File the tessl unwaited-upload API only if the upload wait exceeds 10% of a 4-micro-batch CIFAR step (measured by G-BENCH criterion 5).
15. **E15 Python lane coordination (blocking for waves 3 and 5).** G-BENCH and F-3902 edit *.py under bench/ and ojas-oracle/python while the "Remove or port ojas's Python per the policy" peer holds every *.py. Recommendation: both wait for that lane. torch reference rows and golden generators are policy-permitted oracle Python.

    The peer is a live agent session in this checkout (scratchpad in_flight.md, from the GitPulse/ListAgents pass at audit start), not a board card. No open card with that title was in the 48-card board baseline. Its nearest board owner is F-3902 criterion 4 (ft-3902a4b3), which ports or deletes non-oracle Python. Wave 3 and 5's precondition is therefore "that session has landed or released the *.py files". You decide when that holds; alternatively, file a card for it so the dependency is trackable.
16. **E16 ft-4b67b8d4.** Narrow criteria 2 and 4 to docs/ and links, and hand the residue to ft-7a6496c9. Record the lost wsd_oracle.py there too.
17. **E17 G-LIBM.** Make README.md:99 true by routing CE through `log_sum_exp_exact` with a platform-free `ln1p`, rather than retiring the claim.
18. **E18 Acceptance models for the vision track:**
    - MLP 784-512-10 on MNIST;
    - LeNet-5 on MNIST;
    - ResNet-18-style on CIFAR-10 at 32×32;
    - ViT-tiny (patch 4, D=192, 4 layers) on CIFAR-10.

    All at batch 128, fp32.
19. **Added by the orchestrator: sync tasks/*.md briefs before any `replace:true` import.** The folds and annotations above are board-only, so tasks/*.md briefs now diverge from the board. An import with `replace:true` would reset columns and drop these additions.

## 7. Rejected findings

| finding | reason |
|---|---|
| vision-8 (no no-record executor) | `Eval<B>` exists (ojas-model/src/graph.rs:281, :317); softmax moved into G-V1 |
| V8 (no dropout / eval mode) | A deterministic dropout mask exists (ojas-model/src/lora.rs:228-246); `argmax_rows` runs on device |
| core-autograd-7 (non-canonical size-1 stride) | No in-tree view produces one |
| core-autograd-8 (transient before refusal) | The transient is bounded and comes before a refusal. Kept as a lead for a "host transients before the charge" sweep, which G-IOREAD and F-B5FB criterion 13 partly cover |
| WGPU-3 (workgroup memory read before write) | No kernel reads unwritten `var<workgroup>` memory today |
| WGPU-6 (u32 lane margin) | The bound is a proof (`x*y < 2*groups <= 2^24`); the auditor's margin arithmetic was wrong |
| model-4 (checkpoint "bit for bit") | It is a determinism claim (g9 test), not an integrity claim. Lead for ft-b5fb19e6 criterion 5 |

Duplicates (kept as evidence on the named card, no new work):
- V2-doc → ft-6ba746b4 criterion 6
- vision-10 → ft-fe7dcee6 criterion 2
- METAL-5 → ft-6964cdbd criterion 6
- WGPU-7 → ft-2c183b71
- data-io-6 → ft-bd6373d2 criterion 17
- data-io-8 → ft-b5fb19e6 criterion 1
- capi-go-7 → ft-09edeb32 criteria 1–2
- model-6 → data-io-9

Downgrades by the plan:
- VIS-E: measurement only, with no tessl work until G-BENCH.
- VIS-F: a criterion, not a card.
- METAL-4: doc only; host validators make the kernel guards unreachable.
- CS-2: doc only.
- WGPU-5: a `compile_error!` guard only.
- DI-9: one doc row.
- CA-3: accessor-only.
- MD-3: measure first.
- CG-4: deploy-pages permissions are the minimum Pages needs and are not a defect.
- DI-5: a done-card wording fix.
- M-3: kept as a criterion.

## 8. Decisions made without asking (logged per protocol)

- **Scope exclusions:** ojas-cuda/ojas-hip, arch02/, gemma-metal/ and site/ got no cards; their cards were left alone.
- **reviewed_related:** when add_task refused a new card as possibly duplicating open work, the full list of open item_ids was passed after Fable had read every open card via the baseline. The shallow keyword matches were judged separate work.
- **Logs:** on every overwrite, the card's existing logs were resent verbatim so no history was lost.
- **G-BUDGET:** left in backlog with label needs-your-words pending E6.
- **tessl work:** G-GPUFAULT's tessl-side changes are filed on the ojas board with `repositories: tessl`, not on a tessl board.
- **Board-only folds:** tasks/*.md briefs were not edited because the only allowed writes were board writes and this file. They now diverge (see decision 19).
- **Done cards:** annotated in place (status kept as done). A verification section was added; nothing was reopened.
- **ft-2c183b71's 2026-10-09 re-audit paragraph:** a correction line was appended under it rather than rewriting the old text.
- **ft-69744e54 skipped:** another writer was revising it continuously.

## 9. Gaps and caveats in this audit

- **No build or test was run.** Every MET mark is a source read ("verified in source"). Anything that needs a run, a GPU or CUDA is marked UNVERIFIABLE-HERE.
- **Correction of my own earlier claim:** in the first answer this session I said ReLU and GELU exist in ojas-cpu/src/pointwise.rs. The verifier caught it, and `rg -uu` over ojas-cpu/src and ojas-core/src confirms they exist nowhere in ojas.
- **DevMap gaps:**
  - The index was "ready" but call extraction was partial repo-wide.
  - Seven files parse only partially, including tape.rs, wgpu backend.rs and context.rs, so callers inside their ERROR regions are invisible.
  - Several answers came back `walk_incomplete`.
  - GitPulse insights reported "stored extraction payload is obsolete" while devmap_status reported fresh.
  - DevMap cannot link Metal kernels to Rust callers, because dispatch is by name string.
  - Subagents did not have the DevMap tools and used `rg -uu`, labelled as grep.
- **Concurrency:**
  - 5–6 live agent sessions shared this checkout during the audit.
  - Five other worktrees had uncommitted work (ojas-w11-oom, -oracle, -readout, -wrappers, ojas-w8-ckpt).
  - Card ft-69744e54 was being rewritten continuously.
- **Line numbers** are as read at 040e1ed plus the working tree. Peer lanes may move them.

## 10. GPU training readiness (the user's requirement)

Today a Metal image-classifier step cannot be expressed. Missing pieces:
- conv, pool and norm (G-V2);
- ReLU/GELU, bias-add, LayerNorm, softmax and reduce (G-V1);
- an image reader (G-V4);
- a generic trainer (F-B5E6).

The GPU substrate needed to run these on Metal already exists:
- Metal Tape;
- ExactF32 GEMM;
- Budget governance;
- deferred-fault reporting.

The path is G-V4 → G-BENCH → G-V1 → G-V2 → G-V3, plus F-B5E6. Each model's first Metal training step is measured against torch MPS with the card's own step loop:
- the MLP at wave 4 (G-V1 criterion 7);
- the CNN at wave 5;
- the ViT at wave 6.

The generic trainer (F-B5E6) lands at wave 7. The owner can pull it earlier only by reshuffling waves 4–7, because it shares hot files with the vision chain.


---

# Second audit, same day: slice audit (session 12b75c93), 2026-10-16

Two audits ran against ojas at 040e1ed on 2026-10-16 under the same spec, and neither knew about the other until filing.

- **The audit above is vision-led.** It was filed first, by the "Mac image classification setup" session.
- **This audit went slice by slice over every crate.**

This section is appended to the file; nothing above was edited. Cards from this audit carry the label `audit-slices` as well as `audit-2026-10-16`. Where a finding here duplicated a card from the first audit, it was folded into that card rather than filed again (§A3).

**Scope.**
- No source, config or dependency changes.
- No cargo, GPU, bench or device runs (Mac resource limits).
- The only writes are 25 board writes and this appended section.
- Nothing was moved to done or to any other column. No in_progress card was written. No card was merged or deleted. No priority or severity was lowered.

**Method.**
1. Eight Sonnet slice auditors, run at most 4 at a time:
   - S1 core/autograd/model
   - S2 metal/qwen35
   - S3 wgpu/kernels
   - S4 cpu/simd
   - S5 capi/go
   - S6 data/io/infer
   - S7 device/cuda/hip
   - S8 oracle/CI/tooling
2. Six Opus verifier passes. They re-read every cited line, its callers and upstream guards, gave a verdict per finding, and checked the existing cards criterion by criterion.
3. Fable as the final advisor on grouping, fold-vs-new, waves and owner decisions. Fable was re-consulted after the first audit's cards appeared, and its delta rulings decided every re-homing below.

**Counts.**
- 88 raw findings: S1 11, S2 8, S3 8, S4 10, S5 13, S6 11, S7 11, S8 16.
- 1 new finding raised by the S8 verifier: gotip.sh runs without `-a`.
- 83 survived verification.
- 6 were rejected, merged or got no card (§A6).

## A1. Findings by severity × dimension (83 surviving)

| dimension | high | medium | low | total |
|---|---|---|---|---|
| Modular architecture | 0 | 0 | 6 | 6 |
| Strict typing | 0 | 1 | 4 | 5 |
| Validation at trust boundaries | 0 | 1 | 10 | 11 |
| Async / error boundaries | 0 | 4 | 10 | 14 |
| Deterministic testability | 0 | 7 | 15 | 22 |
| Performance / benchmarks | 0 | 1 | 11 | 12 |
| Correctness across Win/macOS/Linux | 1 | 0 | 8 | 9 |
| Security | 1 | 0 | 3 | 4 |
| **Total** | **2** | **14** | **67** | **83** |

Findings in each cell:
- **Architecture:** S1-08, S3-04, S3-06, S3-07, S4-10, S5-07.
- **Typing:** medium S7-02; low S1-01, S1-03, S5-10, S7-08.
- **Validation:** medium S5-03; low S1-07, S1-09, S2-04, S2-06, S5-04, S5-12, S6-02, S6-03, S6-06, S8-13.
- **Error boundaries:** medium S2-03, S3-01, S4-04, S5-02; low S1-04, S1-11, S2-08, S3-02, S4-05, S5-09, S6-01, S6-07, S6-11, S7-04.
- **Testability:** medium S2-05, S7-01, S7-03, S8-01, S8-02, S8-07, S8-10; low S1-05, S3-03, S3-08, S5-06, S5-13, S7-10, S8-03, S8-04, S8-06, S8-08, S8-09, S8-11, S8-12, S8-15, gotip `-a`.
- **Performance:** medium S4-02; low S1-02, S1-06, S2-07, S4-03, S4-06, S4-07, S4-08, S4-09, S5-08, S6-08, S7-06.
- **Portability:** high S2-01; low S1-10, S2-02, S3-05, S4-01, S5-11, S6-04, S6-05, S7-05.
- **Security:** high S5-01; low S7-07, S8-05, S8-14.

No finding was rated critical after verification. 17 of the 83 turned out to duplicate a first-audit card, some only in part. They are listed in §A6 with the card that owns each one.

## A2. Top 10 risks (Fable, ranked; current homes)

1. **S5-01: data loss.** Go `SaveCheckpoint(ctx, id, "<any existing dir under root>")` renames that directory aside and then `remove_dir_all`s it. No `..` or symlink is needed, and the path ships today. Home: **ft-fe6beb2e** (new, p1 high). The sibling Qwen3.5 provider already refuses an existing directory.
2. **S2-01 (+S2-02): ungated Metal tests break Linux CI.** 22 ojas-metal tests in 5 files have no crate-level cfg and panic on the Linux stub. The Linux `cargo test --workspace` step fails as soon as fmt goes green. Two things compound this:
   - 66 unpushed commits, so none of the CI fixes has run on GitHub.
   - S8-02: GitHub cancels pending main runs by default, so a commit can get no verdict.

   Home: **ft-b4f9ce4e** criteria 11-12.
3. **S3-01: wgpu scratch stacks across ops.** wgpu scratch is held until the next submit. Three back-to-back `muon_ns5_step` calls stack three steps' scratch, and the third call fails *after* params 1-2 were updated in place. That is the half-applied state the Backend contract forbids. Home: **ft-81e49280** (new).
4. **S2-03: no finiteness gate on the Qwen3.5 optimizer.** `Qwen35Step::adamw_step` has no gate of its own. A caller with clipping off writes NaN into every master and moment and gets `Ok` back. Home: **ft-e55afcb5** criterion 9.
5. **S4-02: a serial cutoff regresses reductions.** ft-9aaa6349's own Mitigation A keys the serial cutoff on output bytes. rms_backward grad_w, conv1d/gated-RMS dW, gdn_log_decay backward and fused-CE row reductions then run on one thread. Home: **ft-9aaa6349** criterion 4, which blocks that card's review.
6. **S7-01 / S7-02: CUDA kernels not checked before the GH200.** The Mac clang++ syntax check parses 7 of the 12 NVRTC modules. 37 launch sites carry hand-transcribed parameter lists that no test checks. Both are found only on the GH200. Home: **ft-9bed58f0** criteria 13-14; priority raised p3→p2.
7. **S7-03: CUDA comparators accept bad references.** A NaN or inf reference passes the comparators at about 45 call sites (`peak_check` with an inf peak passes anything). Home: **ft-09edeb32** criterion 9, the CUDA twin of the first audit's oracle criterion 8.
8. **S5-02: SAVE and TOKENIZER ignore cancellation.** After a cancel and then Close, the Go handle is stuck for the life of the process. A cancelled SAVE still finishes the destructive swap from risk 1. Home: **ft-fe6beb2e**.
9. **S8-01 / S8-03 / S8-05: CI gaps.** CI never runs a debug-assertions build. It never links the release `libgusset.a` that the `.pc` ships. It never passes `--locked` while the sibling refs float. Homes: **ft-3902a4b3** criteria 24 and 21.
10. **S1-02 / S6-02 / S5-03: loader host memory is uncharged.** Loaders hold about 2× the largest tensor outside the Budget. Tokenizers are charged nowhere. Homes: **ft-b301b6d6** (the first audit's G-IOREAD, with criteria 6-7 added) and **ft-bd6373d2** criterion 26.

## A3. Cards filed and folded

### New cards (7)

| key | item_id | status | pri | sev | title (short) | findings | wave |
|---|---|---|---|---|---|---|---|
| C1 | ft-fe6beb2e61faa322411ed6f5019ef563 | ready | 1 | high | C ABI SAVE replaces and deletes any existing directory under the model root; SAVE/TOKENIZER ignore cancellation; RESUME skips preflight | S5-01, S5-02, S5-04 | 1 |
| C4 | ft-2dc2dc696fec8a4fbc0503e4057b9762 | ready | 2 | medium | Benchmark baselines for the slice audit's unmeasured perf findings (CPU, capi, CUDA-host rows) | baseline halves of S4-02/03/06/07/08/09, S5-08, S7-06 | 1 |
| C13 | ft-81e49280d7fc01bedc9bc3e52e510f51 | ready | 2 | medium | wgpu scratch lifetime: submit + one retry so optimizer_scratch_bytes stays an upper bound | S3-01 | 6 |
| C20 | ft-a76a0328b7c29ce0045a82fe61d10c33 | backlog | 3 | low | Qwen35Tables carry no spec identity; Tensor's derived Debug prints the whole buffer | S1-01, S1-08 | 6 |
| C22 | ft-fb1b9798083bea2a5c2fec25c984f709 | backlog | 3 | medium | ojas-cpu task panic is Err at some thread counts, unwinds at others, loses its message | S4-04 | 4 |
| C23 | ft-845929b11d93810116e180f416d7957e | backlog | 3 | low | CPU serial passes beside the pool: Muon NS5 sweeps, NaN scans bypassing check_f32s, KV read-slot rescan | S4-06, S4-08, S4-09 | 8 |
| C11 | ft-dcc08479c6f46814965169ca17c57605 | backlog | 3 | low | On-disk training state integrity: streaming per-file digest + Qwen3.5 load_state value checks | S1-09, S2-04 | 10 |

Every new card went through the board's related-tasks refusal. Each listed card was either read in this session or read by Fable in the baseline or delta pass. The card was then filed with `reviewed_related` naming every listed id. The longest list was 32 cards, and the board paged it.

### Folds into existing cards (17)

All folds were overwrite-by-item_id, each done after a fresh `get_task`. Every fold kept the existing title, description, criteria, planned files, logs and repositories, and appended to them.

- Peer criteria were never edited. Where this audit extends one, the extension is appended as `— [audit-slices extension, …]`.
- Where this audit disagrees with the peer's `[MET]` marking, the disagreement is appended as `— [audit-slices … disagrees …]` and explained in the description.

| key | item_id | rev | added | priority/severity change |
|---|---|---|---|---|
| C2 | ft-b4f9ce4e61238a5853e72ee85f56fb03 | 7→8 | crit 11 S2-01/S2-02 (Metal test gates + guard script), 12 S8-02 (per-commit concurrency group), 13 S8-12 (include/path guard); crit 9 disagreement suffix | none (p0 critical) |
| C3 | ft-9aaa63495eba176188b25f07d000b438 | 4→5 | crit 4 S4-02 (review blocker), 5 S4-07; crit 1 disagreement suffix | **raised p3→p1, low→medium** |
| C5 | ft-09edeb325bef2537a7d72e212befdd16 | 3→4 | crit 9 S7-03, 10 S3-08, 11 S5-13 | none |
| C6 | ft-3902a4b3b8d453cd1bc6abcc6f5aaeb8 | 5→6 | new crit 24 (S8-01+S8-03+S5-05). Extensions appended to crits 1 (S8-11), 3 (S8-07), 4 (S8-08, S8-13), 6 (S8-04), 8 (S8-10), 11 (S8-09, gotip `-a`), 13 (S7-07), 20 (S8-14), 22 (S6-05, S5-11). Crit 1 marked PARTIAL, crit 7 marked MET | none |
| C8 | ft-4b67b8d4d89d5a1b38ee4f1e48000dbc | 7→8 | crit 11 S8-06 (stray `enable_cname`, docs on the custom domain), 12 S8-15 (run-file commit ancestry) | none |
| C9 | ft-bd6373d29fcdf322cda6ec69b02475a9 | 4→5 | crits 26-33: S5-03, S6-07, S6-11, S5-06, S5-08, S5-09, S5-10, S5-12; crit 17 S6-03 extension | none |
| C10r | ft-b301b6d677c41c66efee4fefc9d35607 | 1→2 | crit 6 (LoraLinear::merge temporaries), 7 (state.rs:461 and k11 callers bit-identical) | none |
| C12 | ft-e55afcb5d52094ecb9098f46b557a269 | 3→4 | crit 9 S2-03 (adamw finiteness gate), 10 S2-05 (device-free state-machine tests) | none (p1 medium) |
| C14r | ft-757cb6b2a48cc5defb96da14b9536a31 | 1→2 | crit 9 S2-08 (unreported working set), 10 S3-02 (atomic fault word) | none |
| C15 | ft-2c183b7103c95088ae62ea0da43aebc3 | 6→7 | crit 22 S2-07 (U32 uploads > 64 KiB wait on the whole queue) | none |
| C16 | ft-9bed58f086849d18953d833b3d4e1197 | 2→3 | crits 13-17: S7-01, S7-02, S7-04, S7-05 (host half), S7-06; S7-11 as a note | **raised p3→p2** |
| C17 | ft-6964cdbd2e41f71a813cbfb0d26d414b | 6→7 | crit 16 S3-04 extension (dead WGSL `fn scale`), crit 17 S4-05 extension; new crit 18 S3-06, 19 S3-07, 20 S4-10 (scaled_lr) | none |
| C18 | ft-69ef4ce12ace9dd29f1a02caeff9cf25 | 11→12 | crit 13 S1-04 (gate weight bits under Autocast), 14 S1-06 (cast cache) | none |
| C21 | ft-562b0add462330d980964e37a20249ba | 1→2 | crit 6 S4-01 (x86 FMA test half), 7 S4-03 (fused CE ignores Fast) | none |
| C24 | ft-b5fb19e6cf28f0c702a56bde4b44d198 | 4→5 | one clause on crit 12 (S6-01); S6-06/S6-01/S1-07 recorded as duplicates of crits 10-13 | none |
| C26 | ft-754f8eff4c4ebebca8a64ca1f4aa7f77 | 1→2 | crit 9 S7-08 (HIP status literals) | none |
| C27 | ft-7a6496c92f8ab3b00f03d0cee954b8d3 | 1→2 | crit 5 S7-10 (gen_goldens.py dead crate path) | none |

### Verification markings (1 card written, 3 not written)

**Written: ft-67025371** (rev 3→4, review).
- Criteria 1-4 and 6: `[MET — verified in source]`, with evidence.
- Criterion 5: `[NOT MET / CANNOT VERIFY]`. The Mac syntax check covers 7 of 12 modules, and no GH200 re-run is recorded.
- A verification section was added. The status stays review.
- The first audit skipped this card as CUDA (out of its scope).

**Not written:**
- **ft-71d1d2e4.** This audit agrees with the first audit's annotation: criteria 1-2 MET; criterion 3's compile-fail half MET, A/B not verified.
- **ft-a0c303f6** is in_progress, and the user's rule is never to touch in_progress cards. Fable proposed an append-only note; the coordinator overrode it on that rule. The note's content, verbatim in substance:
  - Criterion 2 MET: clip.wgsl:64 and reduce.wgsl:30 guards, landed 1f92659 / aaabf6b / 66ef93b.
  - Criterion 1 PARTIAL. It compares against an f64 reference and against uncapped GPU, not "the CPU exactly", and no green run is recorded.
  - Criterion 3 PARTIAL. The static half is MET (KERNEL_GRIDS, geometry.rs:155+, test :725-776). The dynamic half is NOT MET: only 9-10 of 46 folded kernels run under a capped limit, and 36 run uncapped (scale_multi, round/widen/narrow_bf16, permute, kv_write, embed_fwd/bwd, ce_fwd/bwd, lce_*, gate_*, rope, vr_*, add_inplace, check_finite, adam_*, muon_*, argmax_rows, attn_bwd_dr, col_finish, scale).
  - S3-03: `guard_before_access` (geometry.rs:679-700) accepts any first `return;` inside an `if (… >= …)`. It never compares the bound word, so a guard edited to the wrong word still passes.
  - Criterion 4 NOT MET.
  - The brief says ready; the board says in_progress.
- **ft-3902a4b3, ft-b4f9ce4e and ft-9aaa6349** got their markings inside the folds above.

## A4. Verified-stale claims, and disagreements with the first audit

**Disagreements with the first audit (§3 above).** These are recorded on the cards next to the first audit's text; nothing of theirs was removed.

- **ft-b4f9ce4e criterion 9.**
  - First audit: MET.
  - This audit: **NOT MET.**
  - Evidence: GitHub's concurrency docs (fetched 2026-10-16) say that "any existing pending job or workflow in the same concurrency group will be canceled", regardless of cancel-in-progress. With `group: test-${{ github.ref }}`, a burst of three pushes to main cancels the middle commit's pending run.
  - Fix: crit 12, `group: test-${{ github.event_name == 'pull_request' && github.ref || github.sha }}`.
- **ft-9aaa6349 criterion 1.**
  - First audit: "All 3 MET".
  - This audit: **PARTIAL.** Token-major output is direct only for H==1 (gdn.rs:283-291). The committed A/B shows no win.
  - Separately, Mitigation A causes S4-02 and is now a review blocker (crit 4).
- **ft-a0c303f6.**
  - First audit: criteria 1-3 are on main.
  - This audit: 2 MET; 1 and 3 PARTIAL (above).
- **G-BENCH ft-64601da5 vs Opus-S2 on tessl's unwaited upload.**
  - G-BENCH's body says tessl has no unwaited host upload (tensor.rs:175-196, runtime.rs:873-879).
  - Opus-S2 verified that tessl runtime.rs:1645-1659 `alloc_buffer_from_u32` and :1666-1676 `upload_u32` write on the host without waiting. ojas already uses them mid-batch (device.rs:902-914).
  - Re-read both before filing any tessl work for VIS-E. This is recorded on ft-2c183b71.

**Stale board and brief claims found by this audit:**
- HEAD 040e1ed is **66 commits ahead of origin/main (bc498c4)** and 0 behind. None of ef36f76, 6e7a498, aaabf6b or 6c6eabb has run on GitHub. Every CI "MET" on the board is met in source only.
- ft-29462f23: the brief ends "Board: done" after a recorded run on 2026-10-15; the board says review.
- ft-ffb271da: the brief's Resolution section records all 8 criteria done, with unticked boxes. The board says review.
- Status drift between brief and board:

  | card | brief | board |
  |---|---|---|
  | ft-b4f9ce4e | ready | review |
  | ft-71d1d2e4 | ready | review |
  | ft-9aaa6349 | backlog | review |
  | ft-67025371 | backlog, 6 unchecked boxes | review |
  | ft-a0c303f6 | ready | in_progress |
  | ft-69ef4ce1 | ready | review |

- Stale citations inside briefs:
  - ft-3902a4b3 crit 6 (ci_local.sh:19 / :2-3)
  - ft-301ddd4c tessl-step lines (now test.yml:65-72, 122-129, 186-193, 238-245)
  - ft-67025371 lib.rs:146-430 (lib.rs is 183 lines)
  - ft-16173bf9 still lists gpu_tape_2b.rs
  - docs/cuda-backend-scoping.md:81 still cites `CudaDevice::open`, which cbc4a76 retired. This residual is not on any card; ft-6ba746b4 (docs) is the natural home.
- ft-9c13f47224 criterion 9's scope was narrowed by Fable's C11 ruling. The ojas-model resume NaN scan stays there; C11 takes only the digest.

## A5. Build order (Fable's merged waves)

The first audit's waves are the spine. This audit's cards were placed where their audit-added files do not overlap.

Rules:
- At most 4 cards per wave.
- No shared planned files within a wave. A directory entry counts as every file under it.
- At most one hot file per wave from wave 3 on. ⚠ means the first audit's planned-file list for that vision card was not read; verify it before the wave starts.

| wave | cards | gates |
|---|---|---|
| 1 | ft-fe6beb2e (SAVE), ft-b4f9ce4e (+C2), ft-2dc2dc69 (baselines), G-BUDGET ft-54e8ce27 | G-BUDGET needs E6; push gates ft-b4f9ce4e crits 1/8 |
| 2 | G-V4 ft-f66c67c1 ⚠, F-2946 ft-29462f23, G-LIBM ft-562b0add (+C21), ft-754f8eff (+C26, lib.rs only) | |
| 3 | G-IOREAD ft-b301b6d6 (+C10 residuals), F-6BA7 ft-6ba746b4, ft-69ef4ce1 (+C18, autocast.rs + tests), ft-9bed58f0 (+C16) | C16 only after ft-cab0fdf1 lands, else wave 5; params.rs contingency |
| 4 | G-BENCH ft-64601da5, G-GPUFAULT ft-757cb6b2 (+S2-08, S3-02), F-09ED ft-09edeb32 (+C5), ft-fb1b9798 (C22) | |
| 5 | G-V1 ft-4f11ce5a ⚠, F-BD63 ft-bd6373d2 (+C9), F-B5FB ft-b5fb19e6 (+C24), ft-e55afcb5 (+C12) | C12 after ojas-w8-ckpt lands |
| 6 | G-V2 ft-96023b34 ⚠, F-3902 ft-3902a4b3 (+C6), ft-a76a0328 (C20), ft-81e49280 (C13) | C13 after G-BUDGET and G-GPUFAULT's context.rs edit; C6's S8-10 after ojas-w11-oracle/readout |
| 7 | G-V3 ft-7f4f2d87 ⚠, F-58E3 ft-58e316a9, ft-9aaa6349 (+C3), ft-7a6496c9 (+C27) | |
| 8 | G-TAPE ft-15ebd6b1, F-B5E6 ft-b5e64557 ⚠, ft-845929b1 (C23), ft-2c183b71 (+C15) | C15 holds ojas-core/src/backend.rs; if the hot-file rule counts G-TAPE's tape.rs too, C15 slips to wave 11 |
| 9 | F-9C13 ft-9c13f472 (alone) | |
| 10 | F-6964 ft-6964cdbd (+C17), ft-dcc08479 (C11), ft-4b67b8d4 (+C8) | C8 placed here by the coordinator (Fable's merged list omitted it); its audit-added files are disjoint from F-6964 and C11 |
| 11 | C15 only if needed | |

Not scheduled:
- ft-301ddd4c, ft-ffb271da and ft-6ba746b4 receive no slice-audit criteria.
- The ft-67025371 verification marking is a board-only edit.

**Ordering note.** ft-9bed58f0 (C16, wave 3) and ft-09edeb32 (C5, wave 4) both edit ojas-cuda/src/small_common.rs. They are in different waves, so there is no conflict, but C16 now runs first. C16's description says to sequence after C5's small_common edit; follow the wave order instead.

## A6. Rejected, merged and dropped-as-duplicate

**Rejected or no card**

| finding | reason |
|---|---|
| S6-09 (KvCache not bound to its model) | Only CpuGpt with a caller-owned cache is exposed. DeviceDecoder owns its cache, and capi uses that. A doc note on gpt.rs::check_cache is enough. |
| S6-10 (second JSON grammar in bpe.rs) | The surrogate refusal is deliberate (ft-6964cdbd "checked and dropped"). Only the stale "lockfile this session cannot edit" clause at bpe.rs:654-659 is wrong. Delete it when a card next edits bpe.rs (noted on ft-b5fb19e6). |
| S8-16 (ojas-capi pulls ojas-wgpu unconditionally) | Not a defect. wgpu is a real Go device (go/api.go:92 DeviceWgpu), and ojas-model and ojas-infer depend on ojas-wgpu unconditionally. |
| S7-09 (board drift on ft-67025371) | Not a code finding. Handled by the verification marking. |
| S7-11 (CudaDeviceBuffer Send+Sync) | Unverified because the cudarc sources were not read, and there is no cross-thread consumer. Kept as a note on ft-9bed58f0. |
| S5-05 | Same finding as S8-03; merged into ft-3902a4b3 crit 24. |

**Partial withdrawals inside surviving findings**
- **S1-11:** the inverse_permutation half was dropped, because Tape::permute validates first.
- **S4-02:** the decode-gemv example was withdrawn. Fast 1x2048x6144 runs simd_tiles_into.
- **S8-06:** the proposed sitegen CNAME check was rejected. site/CNAME is inert for Actions deploys.
- **S1-02:** the claim that "gpu_memory_rows under-reports" was withdrawn. Those rows record device and budget memory, not host load peak.
- **S6-08:** the V=248,320 sampler row was withdrawn. ojas-infer has no Qwen path, so G-BENCH's V=50,304 row stands.

**Dropped as duplicates of first-audit cards (Fable delta rulings)**

| finding | covered by |
|---|---|
| S8-05 | ft-3902a4b3 crit 21 |
| S3-04 | ft-6964cdbd crit 16 (dead WGSL `fn scale` appended as an extension) |
| S5-07 | ft-bd6373d2 crit 24 |
| S1-03 | G-TAPE ft-15ebd6b1 crit 2 |
| S1-05 | G-TAPE crit 4 |
| S1-07 | ft-b5fb19e6 crit 13 |
| S1-11 | ft-58e316a9 crit 6 |
| S6-06 | ft-b5fb19e6 crits 10-11 |
| S6-01 | ft-b5fb19e6 crit 12 (one clause appended) |
| S1-02 + S6-02 | G-IOREAD ft-b301b6d6 crits 1-5 (residuals added as 6-7) |
| S6-08 | G-BENCH crit 4 |
| S2-06 | ft-6ba746b4 crit 13 (doc defect). No card carries Opus-S2's device-side relation table, rated low. |
| S3-05 | ft-6ba746b4 crit 14 |
| S1-10 + S6-04 | ft-6ba746b4 crit 15 + ft-3902a4b3 crit 22 |
| S4-01 doc half | ft-6ba746b4 crit 12 |

**Cancelled planned cards**
- C7 (fold into ft-301ddd4c): both of its findings went to ft-3902a4b3.
- C10, C14, C19, C24 and C25: dissolved into the first audit's cards as listed above.

## A7. Decisions made without asking (logged per protocol)

1. **No pause for S5-01.** The data-loss stop condition was weighed. Fable ruled to file at p1 high without pausing, because the fix is unambiguous at the canonical owner and only the predicate's strictness is a product call (OD1, safe default).
2. **Report placement.** AUDIT_REPORT.md already held the first audit's report. This section was appended, and the first audit's text was not edited. Overwriting it would have destroyed a sibling deliverable.
3. **Re-homing instead of duplicating.** After re-pulling the board (64 cards, up from 48), Fable re-ruled every colliding group (§A6). Nine planned cards were dissolved or slimmed.
4. **A second label, `audit-slices`,** separates this audit's cards from the first audit's on the board.
5. **Peer text was never deleted.**
   - Disagreements were appended as suffixes to the criteria and as description sections.
   - Peer "[MET]" prefixes were left as they were.
   - Existing logs and repositories were resent verbatim on every overwrite.
6. **ft-a0c303f6 was not written.** The user's rule is never to touch in_progress cards. This overrode Fable's proposed append-only note; the note is in §A3.
7. **ft-3902a4b3 markings.** Only criterion 1 (PARTIAL) and criterion 7 (MET) got prefixes. The NOT MET list went into the description rather than onto 12 criteria, to keep churn down. The spec mandates marking only met criteria.
8. **C8 wave.** Fable's merged wave list omitted the ft-4b67b8d4 fold. The coordinator placed it in wave 10 by the no-shared-files rule.
9. **`reviewed_related` lists.** Every refusal list was passed in full. Each listed card had been read in this session, by Fable in the delta pass, or in the 48-card baseline.
10. **tasks/*.md left as they are.** Every fold is board-only, so brief/board drift widened on the 17 folded cards, on top of the existing drift in §A4.
11. **Subagent reports.** Subagents cannot write files under the harness policy, so each auditor and verifier returned text, and the coordinator saved it to the session scratchpad.
12. **ghost pid 97075 left alone.** GitPulse's "live session" on the narrow-ojas-cuda worktree is an orphaned `ugrep` (about 34 h old; its cwd directory no longer exists). This session did not start it.

## A8. Owner decisions needed before building

These are in addition to the first audit's E1-E18, several of which this audit's cards depend on (E6, E7, E8, E9, E11, E12, E15).

1. **OD1: C1 predicate (ft-fe6beb2e).** What an existing SAVE target may contain.
   - Strict (default): it must be empty, or hold state.ojck and nothing outside {model.safetensors, optim.safetensors, state.ojck}.
   - Relaxed: extra files are allowed.
2. **OD2: digest implementation for C11 (ft-dcc08479).** A new hash crate (needs dependency approval) or an in-crate CRC32C. Either way, both consumers' formats bump.
3. **OD3: S3-06 (ft-6964cdbd crit 18).** WgpuContext::device()/queue() go from pub to pub(crate). This removes public API, so it gets two signals and then an ask.
4. **OD6: S4-09 (ft-845929b1).** Fuse the KV read-slot pre-scan into attend, or keep a "written slots finite" flag.
5. **OD7: S5-08 (ft-bd6373d2 crit 30).** Drop OP_GENERATE from ALLOCATING_OPS, which changes the ErrPressure contract, or cache the probe for about 100 ms. Decide after the baseline row.
6. **OD8: S4-01 (ft-562b0add crit 6).** Runtime-dispatched FMA on x86 changes Fast bits. Default: document per-target behaviour.
7. **OD9: optional early CI card.** S8-01's debug-assertions leg (ft-3902a4b3 crit 24) could be split out to land before wave 6.
8. **OD10: push the 66 local commits.** origin/main is bc498c4. Until then, every CI "MET" is source-only. This is the same decision as the first audit's E11b.
9. **OD11: re-sync tasks/*.md with the board.** Both audits wrote board-only. Re-sync only after checking drift; an `import replace:true` would reset columns and drop the additions. This is the same as the first audit's decision 19.
10. **OD12: kill the orphaned `ugrep`, pid 97075.** It is GitPulse's phantom session. The user should do it.
11. **ft-9aaa6349 status.** It is in review, but criterion 4 (S4-02) is a blocker. Consider moving it back to ready. The audit does not move cards.

OD4 and OD5 (DX12/Windows) were first-audit decisions too; they now sit inside ft-6ba746b4 criteria 14-15.

## A9. Gaps and caveats in this audit

- **No build, test, GPU, CUDA or bench run was made.** Every MET is "verified in source". Every CI claim is source-only, because the 66 commits are unpushed.
- **CUDA-C edits cannot be checked here.** NVRTC never runs on the Mac, so every GH200 criterion stays open until a recorded run.
- **Vendored sources were not available for several claims.** These stay unverified: HIP numeric status values (S7-08), cudarc pool retention (S7-04) and context binding (S7-11).
- **DevMap gaps.**
  - The index was fresh (generation 7020) but `walk_incomplete` everywhere.
  - 19 partial-parse files, including tape.rs, wgpu backend.rs/context.rs, io checkpoint.rs and device host.rs.
  - Metal kernel→Rust links are invisible.
  - DevMap missed the production caller ojas-qwen35 state.rs:461, which grep found (S6-02).
  - Subagents had no DevMap tools. Every "no caller" claim in this section is `rg -uu` unless stated otherwise.
- **GitPulse gaps.**
  - Insights reported "stored extraction payload is obsolete" while devmap_status reported fresh.
  - The "live session" for ft-67025371's worktree was a ghost: pid 97075, an orphaned `ugrep`.
- **Concurrency.**
  - 4-6 live agent sessions shared this checkout.
  - Five worktrees held uncommitted work: ojas-w11-oom, -oracle, -readout, -wrappers, and ojas-w8-ckpt.
  - A Python-policy session holds *.py files.
  - The first audit filed into the same board while this audit was in Phase 3. Both are now idle. collision_risk reported 0 overlapping files at filing time.
- **Board writes were checked only by the write result.** Each one was confirmed by its own response (ok plus the revision bump). The board was not re-listed afterwards.
