# group_norm on Metal vs SpikeTrack's CPU round trip (2026-10-27)

Question: does `MetalBackend::group_norm_forward` on device-resident tensors beat what SpikeTrack does today for its pixel decoder's GroupNorm (download the activation, run the CPU group_norm, upload the result), at both SAM shapes? Rule, stated by ojas-78 before the first measurement: Metal group_norm earns its place if (a) beats (c) beyond noise at both shapes.

Method: `ojas-metal/examples/group_norm_bench.rs`, `cargo run --release` under mac_heavy on the M5 Pro. Per arm, 5 warm-ups then 25 runs, `Backend::sync` around the call only; the arms run one after another, not interleaved. Groups 8, eps 1e-5. Raw numbers: `bench-output.txt` (both runs).

- (a) Metal `group_norm_forward` on tensors already on the device.
- (b) `CpuBackend::group_norm_forward` on host tensors.
- (c) Download x from the device, CPU group_norm, upload the result: SpikeTrack's current path.
- First call: the first (a) call on fresh inputs (pipelines are created at backend start, so this is buffer and dispatch setup, not compilation).

## Result (final run, after the large-mean correction in 6a870f9; log a-gn2), min / median ms

| Shape | (a) Metal, resident | (b) CPU | (c) download + CPU + upload | (a) first call | (c) / (a), medians |
|---|---|---|---|---|---|
| [1,256,144,144] | 0.878 / 1.201 | 6.577 / 6.603 | 7.990 / 8.103 | 5.020 | 6.7x |
| [1,256,288,288] | 2.003 / 2.062 | 26.373 / 26.439 | 31.969 / 32.244 | 9.766 | 15.6x |

The rule holds at both shapes. Spread was not recorded (min and median only), so "beyond noise" rests on the size of the gap: (c) exceeds (a) by 6.9 ms and 30.2 ms at the medians, and (c)'s minimum is above (a)'s median by 6.8x and 15.5x. The earlier run (log a-gnb, before the correction, other lanes loading the machine) gave (a) 1.087 and 2.264 ms median against (c) 8.135 and 32.491, the same verdict.

(The lane's commit message quotes 9x and 16x; those are min-to-min ratios, 7.990 / 0.878 and 31.969 / 2.003. The median ratios are in the table.)

Bandwidth: the three passes read x three times and write y once, 85 MB at 144x144 and 340 MB at 288x288; at 288x288 that is about 170 GB/s at the 2.0 ms minimum, so the larger shape is bandwidth-bound and the smaller carries launch cost.

## Limits

- Arms are sequential, not interleaved, and no p10-p90 spread is recorded; a gap of 7x to 15x is far outside any spread these arms showed elsewhere, but it is not measured here.
- The first call (5.0 and 9.8 ms) is paid once per shape; it is still below (c).
- One machine (M5 Pro), shared with other sessions during the first run.
