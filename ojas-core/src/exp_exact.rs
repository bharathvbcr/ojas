//! `e^x` correctly rounded to `f32`, the exponential of
//! [`crate::Numerics::Exact`]. Plain `f64` arithmetic with no fused
//! multiply-add and no libm call, so the bits are the same on every
//! platform; libm `expf` is not (`expf(-2^-25)` is `0x3f800000` on glibc
//! 2.41 and `0x3f7fffff` on macOS 27).

/// Above this `e^x` overflows `f32`, below `EXACT_LO` it rounds to `+0.0`
/// (`e^-104` is below `2^-150`, half the smallest subnormal). Between them
/// the `f64` value rounds itself to `+inf`, a subnormal or `+0.0`.
const EXACT_HI: f32 = 89.0;
const EXACT_LO: f32 = -104.0;
/// `1.5 · 2^52`: adding and subtracting it rounds an `f64` below `2^51` to
/// the nearest integer, ties to even.
const ROUND_F64: f64 = 6_755_399_441_055_744.0;
/// `256 / ln 2`, and `ln 2 / 256` split so `k · LN2_256_HI` is exact for
/// `|k| < 2^17` (36 significant bits).
const INV_LN2_256: f64 = f64::from_bits(0x4077_1547_652b_82fe);
const LN2_256_HI: f64 = f64::from_bits(0x3f66_2e42_fefa_0000);
const LN2_256_LO: f64 = f64::from_bits(0x3cfc_f79a_bc9e_3b3a);
/// `e^r - 1 - r` Taylor coefficients. With `|r| <= ln 2 / 512` the first
/// term left out, `r^5 / 120`, is below `2^-54` relative.
const C2: f64 = 1.0 / 2.0;
const C3: f64 = 1.0 / 6.0;
const C4: f64 = 1.0 / 24.0;

/// `2^(j/256)` as `(hi, lo)`; see the generator note in the file.
const EXP2_J256: [(f64, f64); 256] = include!("exp2_j256.rs");

/// `e^x` correctly rounded to `f32` (round to nearest, ties to even), the
/// same bits on every platform. NaN stays NaN, `+inf` and overflow are
/// `+inf`, `-inf` and underflow are `+0.0`.
///
/// `x = k·ln2/256 + r`, `k` rounded to nearest and `|r| <= ln2/512`, then
/// `e^x = 2^(k >> 8) · 2^((k & 255)/256) · e^r`, all in `f64`, and one
/// rounding to `f32` at the end. The `f64` value is within `2^-50` relative
/// of `e^x` (the largest error measured against `decimal` is `2^-53`). The
/// sweep in the tests covers every `f32` input: none whose `e^x` lies that
/// close to a rounding boundary is rounded the wrong way.
#[inline]
pub fn exp_exact(x: f32) -> f32 {
    exp_exact_wide(x) as f32
}

/// The `f64` value [`exp_exact`] rounds, before the rounding.
#[inline(always)]
fn exp_exact_wide(x: f32) -> f64 {
    if x.is_nan() {
        return f64::from(x);
    }
    if x > EXACT_HI {
        return f64::INFINITY;
    }
    if x < EXACT_LO {
        return 0.0;
    }
    let xd = f64::from(x);
    // |xd · 256/ln 2| < 38500: the shift rounds it exactly to an integer.
    let kf = (xd * INV_LN2_256 + ROUND_F64) - ROUND_F64;
    let k = kf as i64;
    let r = (xd - kf * LN2_256_HI) - kf * LN2_256_LO;
    let (t_hi, t_lo) = EXP2_J256[(k & 255) as usize];
    let p = r + r * r * (C2 + r * (C3 + r * C4));
    let y = t_hi + (t_hi * p + t_lo);
    // k is in [-38410, 32871], so k >> 8 is in [-151, 128]: 2^(k >> 8) is a
    // normal f64 and the product is exact.
    y * f64::from_bits((((k >> 8) + 1023) as u64) << 52)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The relative bound on `exp_exact_wide`'s error the sweep assumes.
    const WIDE_ERR: f64 = 1.0 / (1u64 << 50) as f64;

    /// The `f32` rounding boundary nearest `y`: the midpoint between `y as
    /// f32` and its neighbour on `y`'s side. Midpoints of adjacent `f32`s
    /// are exact in `f64`.
    fn boundary(y: f64) -> f64 {
        let f = y as f32;
        if f == f32::INFINITY {
            // Between f32::MAX and 2^128.
            return f64::from(f32::MAX) + f64::from_bits(((103 + 1023) as u64) << 52);
        }
        let fd = f64::from(f);
        let next = if y > fd {
            f32::from_bits(f.to_bits() + 1)
        } else if f == 0.0 {
            return f64::from(f32::from_bits(1)) / 2.0;
        } else {
            f32::from_bits(f.to_bits() - 1)
        };
        (fd + f64::from(next)) / 2.0
    }

    fn pins() -> Vec<(u32, u32)> {
        include_str!("exp_exact_pins.txt")
            .lines()
            .filter(|l| !l.starts_with('#'))
            .map(|l| {
                let (x, y) = l.split_once(' ').unwrap();
                (
                    u32::from_str_radix(x, 16).unwrap(),
                    u32::from_str_radix(y, 16).unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn matches_the_pinned_correctly_rounded_values() {
        let pins = pins();
        assert!(pins.len() > 2000, "{} pins", pins.len());
        for (x, want) in pins {
            let got = exp_exact(f32::from_bits(x)).to_bits();
            assert_eq!(got, want, "exp_exact({x:08x}) = {got:08x}, want {want:08x}");
        }
    }

    #[test]
    fn special_values_and_the_range_edges() {
        assert!(exp_exact(f32::NAN).is_nan());
        assert_eq!(exp_exact(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp_exact(f32::NEG_INFINITY).to_bits(), 0);
        assert_eq!(exp_exact(0.0), 1.0);
        assert_eq!(exp_exact(-0.0), 1.0);
        assert_eq!(exp_exact(f32::MAX), f32::INFINITY);
        assert_eq!(exp_exact(f32::MIN).to_bits(), 0);
        // 0x42b17217 is the last finite result, 0xc2cff1b4 the last non-zero
        // one (the smallest subnormal).
        assert_eq!(
            exp_exact(f32::from_bits(0x42b1_7217)),
            f32::from_bits(0x7f7f_ff84)
        );
        assert_eq!(exp_exact(f32::from_bits(0x42b1_7218)), f32::INFINITY);
        assert_eq!(exp_exact(f32::from_bits(0xc2cf_f1b4)).to_bits(), 1);
        assert_eq!(exp_exact(f32::from_bits(0xc2cf_f1b5)).to_bits(), 0);
    }

    /// Every `f32` in `[EXACT_LO, EXACT_HI]`, about 2.2e9 inputs. Prints
    /// each input whose `f64` value lies within `WIDE_ERR` of a rounding
    /// boundary (`amb`), each of which must be pinned, and fails on any
    /// result that differs from libm's `f64` `exp` rounded to `f32` outside
    /// that window (libm's `f64` error is below `2^-52`, so outside the
    /// window it rounds correctly).
    ///
    /// ```text
    /// cargo test -p ojas-core --release --lib exp_exact::tests::sweep -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn sweep() {
        let ranges = [
            (0x0000_0000u32, EXACT_HI.to_bits()),
            (0x8000_0000u32, EXACT_LO.to_bits()),
        ];
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        let mut lines = Vec::new();
        for (lo, hi) in ranges {
            let n = u64::from(hi - lo) + 1;
            let per = n.div_ceil(threads as u64);
            let parts: Vec<Vec<String>> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..threads as u64)
                    .map(|t| {
                        s.spawn(move || {
                            let mut out = Vec::new();
                            let start = u64::from(lo) + t * per;
                            let end = (start + per).min(u64::from(hi) + 1);
                            for b in start..end {
                                let x = f32::from_bits(b as u32);
                                let y = exp_exact_wide(x);
                                let got = y as f32;
                                let edge = boundary(y);
                                if (y - edge).abs() <= WIDE_ERR * y {
                                    out.push(format!("amb {:08x} {:08x}", b, got.to_bits()));
                                    continue;
                                }
                                let libm = f64::from(x).exp() as f32;
                                if libm.to_bits() != got.to_bits() {
                                    out.push(format!(
                                        "dis {:08x} {:08x} {:08x}",
                                        b,
                                        got.to_bits(),
                                        libm.to_bits()
                                    ));
                                }
                            }
                            out
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            lines.extend(parts.into_iter().flatten());
        }
        for line in &lines {
            println!("{line}");
        }
        let dis = lines.iter().filter(|l| l.starts_with("dis")).count();
        println!("total {} amb {} dis {dis}", lines.len(), lines.len() - dis);
        assert_eq!(dis, 0, "results that differ from libm outside the window");
        let pins = pins();
        for line in lines.iter().filter(|l| l.starts_with("amb")) {
            let mut parts = line.split(' ').skip(1);
            let x = u32::from_str_radix(parts.next().unwrap(), 16).unwrap();
            let got = u32::from_str_radix(parts.next().unwrap(), 16).unwrap();
            let want = pins.iter().find(|p| p.0 == x).map(|p| p.1);
            assert_eq!(
                want,
                Some(got),
                "unpinned or wrong near-boundary input {x:08x}"
            );
        }
    }

    /// Per-call time of libm `f32::exp` and `exp_exact` over `[-20, 0]`,
    /// alternated five times (2026-10-05, M5 Pro: about 1.33 and 1.92 ns).
    ///
    /// ```text
    /// cargo test -p ojas-core --release --lib exp_exact::tests::per_call_cost -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn per_call_cost() {
        let xs: Vec<f32> = (0..1 << 20)
            .map(|i| -(i as f32) * (20.0 / (1 << 20) as f32))
            .collect();
        for round in 0..5 {
            for (name, f) in [("libm", f32::exp as fn(f32) -> f32), ("exact", exp_exact)] {
                let start = std::time::Instant::now();
                let mut acc = 0.0f32;
                for _ in 0..20 {
                    for &x in &xs {
                        acc += f(std::hint::black_box(x));
                    }
                }
                let ns = start.elapsed().as_secs_f64() * 1e9 / (20.0 * xs.len() as f64);
                println!("cost {round} {name} {ns:.2} ns {acc}");
            }
        }
    }

    /// Prints the inputs `exp_exact_pins.txt` was made from, with this
    /// crate's outputs, for the decimal oracle to check.
    #[test]
    #[ignore]
    fn print_pin_candidates() {
        let mut xs: Vec<u32> = Vec::new();
        for (lo, hi) in [
            (0u32, EXACT_HI.to_bits()),
            (0x8000_0000, EXACT_LO.to_bits()),
        ] {
            xs.extend((lo..=hi).step_by(1 << 20));
        }
        // The last finite and first infinite result, and the last non-zero
        // and first zero result.
        let up = (0x42b1_0000u32..0x42b3_0000)
            .find(|&b| exp_exact(f32::from_bits(b)) == f32::INFINITY)
            .unwrap();
        let down = (0xc2cf_0000u32..0xc2d1_0000)
            .find(|&b| exp_exact(f32::from_bits(b)) == 0.0)
            .unwrap();
        xs.extend(up - 3..up + 3);
        xs.extend(down - 3..down + 3);
        for b in xs {
            println!("p {b:08x} {:08x}", exp_exact(f32::from_bits(b)).to_bits());
        }
    }
}
