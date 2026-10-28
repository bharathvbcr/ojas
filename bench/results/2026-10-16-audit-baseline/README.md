# 2026-10-16 audit baselines

Rows for GitPulse `ft-2dc2dc696fec8a4fbc0503e4057b9762`. They time the kernels as they are. No kernel in this directory was edited to make a number smaller.

The lines are in `rows.txt` (49 `OJAS_OP` rows, copied from the group logs). `env.md` is the machine, the pin, and the per-group load. This README is how to cite them.

`mac_heavy.sh` was not used. That script lives outside this repo. Runs were serial, one release test process at a time. A group started only after two samples, 30 seconds apart, with load1 under 12, load5 under 18, and no `yes` processes. Every group below met that bar. `quiet=true` for all of them.

These files were untracked when written. They are committed now, in the merge of `gp/w19-audit-baseline` into `gp/78-integrate`, as recorded; none of the numbers was re-measured in that merge.

**Do not read these rows as describing the merged code.** They were measured at `105e203`. The `audit_gated_rms_bwd_qwen` and `audit_fused_ce_*` rows predate the gated-RMSNorm rstd change (`gp/w20-gated-rstd`) and the target-excluded cross-entropy merge, which change the code those rows time. Re-measure before citing them for either.

**There is no noise floor behind the 18-thread conclusions.** Each group started at load1 about 9 to 11 (see `env.md`, not the quiet machine the gate name suggests). CPU rows are `n=3` to `n=5` timed calls, one run per arm, no repeats of the whole process, so a row's spread says how that single run varied and says nothing about run-to-run noise. The statement at the end that some ops are slower at 18 threads than at one rests on that, and is a lead to re-run with interleaved repeats, not a finding.

## How to read a row

`OJAS_OP` keys `op`, `dir`, `shape`, `threads`, `min_ms` are the ones `scripts/benchreport` already splits on. `med_ms`, `max_ms`, `spread_ms`, and `numerics` are extra and ignored by that splitter. `spread_ms` is max minus min over the timed calls. The median is `ns[n/2]` after sorting. Shape spaces are underscores, so the line has no `=` inside a value.

`OJAS_META numerics=Fast` on a CPU log is the shared Fast pool, not the row. The row's own `numerics=` is the backend that ran.

CPU warmup is two calls per direction, not counted in `n` (`ojas-cpu/tests/bench_ops.rs`, the loop `for _ in 0..2` inside `run_time`). capi warms up 20 calls, then n=200. The CUDA key warms up 2 calls, then n=200.

## What a later A/B cites

| Row | Fix that must cite it |
|---|---|
| `audit_rms_bwd_8192x2048`, `audit_rms_bwd_65536x128`, `audit_conv1d_silu_bwd_qwen`, `audit_gated_rms_bwd_qwen` | ft-9aaa6349 criterion 4 (S4-02, serial cutoff on output bytes) |
| `audit_gdn_b1_t2048_h16_d128` forward and backward | ft-9aaa6349 criterion 5 (S4-07) |
| `audit_fused_ce_v50304_{exact,fast}`, `audit_fused_ce_v248320_{exact,fast}` | ft-562b0add criterion 7 (S4-03) |
| `audit_muon_768x768_{exact,fast}`, `audit_muon_6144x2048_{exact,fast}` | ft-845929b11d93810116e180f416d7957e (S4-06) |
| `audit_nan_scan_permute_16m`, `audit_nan_scan_add_bwd_16m` | ft-845929b11d93810116e180f416d7957e (S4-08). The number is the whole op with a cold finite cache, including the transpose or the two-buffer copy. |
| `audit_cached_attn_8k`, `audit_cached_attn_32k` | ft-845929b11d93810116e180f416d7957e (S4-09). OD6 is still open. This row measures the current read-slot rescan. It does not choose the design. |
| `op_generate` `probe=0` and `probe=1` | ft-bd6373d2 criterion 30 (S5-08). The delta is the probe. Neither arm looks up a session or parses a wire payload. |
| `cache_key_new` on `GDN_PUBLISHED` + `STRICT_SM90` | ft-9bed58f0 criterion 17 (S7-06). Host only. `source_len=20879`. The timed call hashes the full published source. |

Threads are 1 and 18 on every CPU row. Gemv is Exact only. GDN and the NaN-scan rows stay on Fast. Muon stays `Ns5Precision::F32` and varies the backend numerics.

## Minima, milliseconds

Copied from `rows.txt`. Spread and median are on that line, not here.

| Op | dir | numerics | 1 thread | 18 threads |
|---|---|---|---:|---:|
| rms bwd 8192×2048 | bwd | Fast | 37.7092 | 18.0839 |
| rms bwd 65536×128 | bwd | Fast | 10.5998 | 35.9080 |
| conv1d-SiLU bwd, `[1,1024,6144]` k=4 | bwd | Fast | 104.9753 | 28.4623 |
| gated RMS bwd, `[16384,128]` | bwd | Fast | 33.8166 | 21.5731 |
| fused CE V=50304 | fwd | Exact | 196.8182 | 29.7573 |
| fused CE V=50304 | bwd | Exact | 579.4393 | 80.1441 |
| fused CE V=50304 | fwd | Fast | 21.9165 | 20.8249 |
| fused CE V=50304 | bwd | Fast | 55.9583 | 44.8025 |
| fused CE V=248320, rows=1, hidden=2048 | fwd | Exact | 81.2454 | 83.7687 |
| fused CE V=248320 | bwd | Exact | 441.7403 | 547.3969 |
| fused CE V=248320 | fwd | Fast | 15.3837 | 16.1680 |
| fused CE V=248320 | bwd | Fast | 87.1641 | 72.9278 |
| gemv 1×2048×6144 | fwd | Exact | 2.2352 | 2.3921 |
| muon 768×768 | step | Exact | 277.2535 | 34.5330 |
| muon 768×768 | step | Fast | 14.0357 | 13.7767 |
| muon 6144×2048 | step | Exact | 11044.0936 | 1216.1124 |
| muon 6144×2048 | step | Fast | 530.7258 | 377.3177 |
| GDN `[1,2048,16,128]` | fwd | Fast | 116.7160 | 18.5485 |
| GDN `[1,2048,16,128]` | bwd | Fast | 1336.8915 | 140.0797 |
| permute 4096×4096, cold cache | fwd | Fast | 18.5658 | 14.3190 |
| add-backward 4096×4096, cold cache | bwd | Fast | 6.4707 | 5.7373 |
| cached attention kv=8192 | fwd | Fast | 8.7718 | 1.1352 |
| cached attention kv=32768 | fwd | Fast | 29.6995 | 5.3896 |

capi, n=200, vocab 50304, token 50303, probe succeeded: probe=0 min 0.093917 ms, probe=1 min 0.095042 ms. CUDA `CacheKey::new` min 0.019958 ms, spread 0.018792 ms, source_len 20879.

At both fused-CE shapes Fast is faster than Exact on the whole op. The 65536×128 RMS backward and the V=248320 Exact backward are slower at 18 threads than at one. Those are the current kernels.
