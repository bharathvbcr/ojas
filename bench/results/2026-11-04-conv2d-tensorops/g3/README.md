# G3 result: the Ojas conv route in SpikeTrack, 1,500 frames (2026-11-04)

Rules: `../g2/RULES.md`, written before the runs, applied to the first 300 frames of each held-out
video. The fp16 rule at this length is HOTA >= base - 0.003 with no video dropping by more than 0.01.
Job: `../g2/job.sh` run with `N=300 OUT=g3`, log `job.log`, binary hash in `provenance.txt` (the
G2 binary). Load average was 3.2-6.2 throughout. The run took 21:58-22:52Z.

| config | HOTA (combined) | dHOTA | worst video | tracks vs base | verdict |
|---|---|---|---|---|---|
| base (im2col) | 0.569515 | | | | reproduces E32's documented 0.5695 (SpikeTrack `bench/results/e32-memenc`) |
| conv32 (`OJAS_METAL_CONV2D_TENSOROPS=f32`) | 0.569486 | -0.000029 | 82: -0.0002 | differ on all 5 | **PASS** |
| conv16 (`OJAS_METAL_CONV2D_TENSOROPS=f16`) | 0.569470 | -0.000045 | 82: -0.0002 (47: +0.0002) | differ on 21, 36, 47, 82 | **PASS** |

Per video, base: 21 0.6695, 22 0.6691, 36 0.5319, 47 0.4838, 82 0.5291. Every video in both arms
is within 0.0002 of base.

**The route ran [V]:** each arm's `run/<id>.json` differs from base byte for byte (`cmp`) on four or
five videos. The changes are small enough that HOTA moves by at most 0.0002 on any video.

**What this decides:** both G2 and G3 pass for both precisions, so the conv route is a candidate
for SpikeTrack's default. It is not one yet, and the per-frame gain is small (2-4% of the push,
`../README.md`), because Core ML runs the necks on the default path.
