# G2 result: the Ojas conv route in SpikeTrack, 32 frames (2026-11-04)

Rules: `RULES.md`, written before the runs. Job: `job.sh`, log `job.log`, binary hash in
`provenance.txt`. Load average 4.1-5.5 throughout.

| config | HOTA (combined) | dHOTA | worst video | tracks vs base | verdict |
|---|---|---|---|---|---|
| base (im2col) | 0.602461 | | | | reproduces the documented 0.6025 |
| conv32 (`OJAS_METAL_CONV2D_TENSOROPS=f32`) | 0.602461 | +0.000000 | 0.000000 on every video | differ on 21, 22, 36, 47 | **PASS** |
| conv16 (`OJAS_METAL_CONV2D_TENSOROPS=f16`) | 0.602461 | +0.000000 | 0.000000 on every video | differ on 21, 36, 47 | **PASS** |

Per video (all three configs): 21 0.6748, 22 0.6801, 36 0.5756, 47 0.5599, 82 0.5544.

**The route ran [V]:** each config's `run/<id>.json` differs byte for byte from base on three or
four videos (`cmp`). The differences are small enough that no detection changed its match, so HOTA
is identical to six decimals.

G3, the 1,500-frame run, is in `../g3/`.
