//! RoPE tables for [`crate::Backend::rope_partial_forward`], including
//! Qwen3.5's multimodal RoPE (MRoPE).
//!
//! MRoPE gives each of the `R / 2` rotary frequencies one of three position
//! streams (time, height, width), chosen by `mrope_section`. Text-only input
//! copies the token position into all three streams, so every frequency's
//! angle is `position * inv_freq` whichever stream it reads, and MRoPE
//! collapses to the plain partial RoPE of [`mrope_text_tables`]. The
//! collapse is a property of the tables, so the backends need one op.

use crate::{Budget, DType, OjasError, Tensor};

const OP: &str = "mrope_tables";

/// How MRoPE assigns frequencies to streams. `section` counts the
/// frequencies of time, height and width and sums to `rotary_dim / 2`.
/// Interleaved (Qwen3.5, `mrope_interleaved`): frequency `i` reads height
/// when `i % 3 == 1` and `i < 3 * section[1]`, width when `i % 3 == 2` and
/// `i < 3 * section[2]`, and time otherwise. Not interleaved (Qwen2-VL):
/// the first `section[0]` read time, the next `section[1]` height, the rest
/// width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MropeSection {
    pub section: [usize; 3],
    pub interleaved: bool,
}

impl MropeSection {
    /// The stream (0 time, 1 height, 2 width) frequency `i` reads.
    pub fn stream(&self, i: usize) -> usize {
        let [t, h, _] = self.section;
        if self.interleaved {
            match i % 3 {
                1 if i < 3 * self.section[1] => 1,
                2 if i < 3 * self.section[2] => 2,
                _ => 0,
            }
        } else if i < t {
            0
        } else if i < t + h {
            1
        } else {
            2
        }
    }
}

/// `cos` and `sin`, each `[T, rotary_dim]`, for position streams
/// `positions` (time, height, width; `T` each) under `section`.
///
/// Frequency `i` of `R / 2` is `inv_freq_i = 1 / theta^(2i / R)` and its
/// angle at row `t` is `positions[stream(i)][t] * inv_freq_i`, both formed
/// in f32 as transformers forms them (`inv_freq` and `inv_freq @
/// position_ids` are float32; the power comes from the platform's `powf`).
/// Only the cosine and sine are taken in f64 and rounded once. Columns `i`
/// and `i + R / 2` hold the same angle (`cat(freqs, freqs)`), the
/// half-split pairing the op rotates.
///
/// Refused: `rotary_dim` zero or odd, `theta` not finite and above 1, a
/// section that does not sum to `rotary_dim / 2`, streams of unequal
/// length or of length 0.
pub fn mrope_tables(
    positions: [&[u32]; 3],
    section: MropeSection,
    rotary_dim: usize,
    theta: f64,
    budget: &Budget,
) -> Result<(Tensor, Tensor), OjasError> {
    let range = |detail: String| OjasError::OutOfRange { op: OP, detail };
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(range(format!(
            "rotary_dim {rotary_dim} must be even and non-zero"
        )));
    }
    if !(theta.is_finite() && theta > 1.0) {
        return Err(range(format!("theta {theta} must be finite and above 1")));
    }
    let half = rotary_dim / 2;
    let sum = section
        .section
        .iter()
        .try_fold(0usize, |a, &s| a.checked_add(s));
    if sum != Some(half) {
        return Err(range(format!(
            "mrope_section {:?} must sum to rotary_dim / 2 = {half}",
            section.section
        )));
    }
    let len = positions[0].len();
    if len == 0 || positions.iter().any(|p| p.len() != len) {
        return Err(OjasError::Shape {
            op: OP,
            detail: format!(
                "position streams of lengths {:?} must be equal and non-empty",
                positions.map(<[u32]>::len)
            ),
        });
    }
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / rotary_dim as f32))
        .collect();
    let mut cos = Tensor::zeros(&[len, rotary_dim], DType::F32, budget)?;
    let mut sin = Tensor::zeros(&[len, rotary_dim], DType::F32, budget)?;
    {
        let (c, s) = (cos.f32_slice_mut()?, sin.f32_slice_mut()?);
        let rows = c
            .chunks_exact_mut(rotary_dim)
            .zip(s.chunks_exact_mut(rotary_dim));
        for (t, (c_row, s_row)) in rows.enumerate() {
            for (i, &f) in inv.iter().enumerate() {
                let angle = f64::from(positions[section.stream(i)][t] as f32 * f);
                let (sv, cv) = angle.sin_cos();
                c_row[i] = cv as f32;
                c_row[i + half] = cv as f32;
                s_row[i] = sv as f32;
                s_row[i + half] = sv as f32;
            }
        }
    }
    Ok((cos, sin))
}

/// [`mrope_tables`] for text-only input at positions `start..start + len`:
/// the same positions in all three streams, which is the plain partial
/// RoPE table whatever `section` says.
pub fn mrope_text_tables(
    start: u32,
    len: usize,
    section: MropeSection,
    rotary_dim: usize,
    theta: f64,
    budget: &Budget,
) -> Result<(Tensor, Tensor), OjasError> {
    let end = u32::try_from(len)
        .ok()
        .and_then(|n| start.checked_add(n))
        .ok_or_else(|| OjasError::OutOfRange {
            op: OP,
            detail: format!("positions {start} + {len} exceed u32"),
        })?;
    let pos: Vec<u32> = (start..end).collect();
    mrope_tables([&pos, &pos, &pos], section, rotary_dim, theta, budget)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(t: &Tensor) -> Vec<u32> {
        t.to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    const QWEN: MropeSection = MropeSection {
        section: [11, 11, 10],
        interleaved: true,
    };

    #[test]
    fn interleaved_and_chunked_assignments() {
        // Qwen3.5: t h w t h w ... for 3 * 10 frequencies, then the 11th
        // height frequency (index 31), and the rest of the indices are time.
        let streams: Vec<usize> = (0..32).map(|i| QWEN.stream(i)).collect();
        for (i, &s) in streams.iter().enumerate() {
            let want = match i % 3 {
                1 if i < 33 => 1,
                2 if i < 30 => 2,
                _ => 0,
            };
            assert_eq!(s, want, "frequency {i}");
        }
        assert_eq!(streams.iter().filter(|&&s| s == 0).count(), 11);
        assert_eq!(streams.iter().filter(|&&s| s == 1).count(), 11);
        assert_eq!(streams.iter().filter(|&&s| s == 2).count(), 10);
        let chunked = MropeSection {
            section: [2, 3, 1],
            interleaved: false,
        };
        let s: Vec<usize> = (0..6).map(|i| chunked.stream(i)).collect();
        assert_eq!(s, [0, 0, 1, 1, 1, 2]);
    }

    /// The collapse: text-only MRoPE tables are the one-stream tables bit
    /// for bit, for either assignment.
    #[test]
    fn text_only_mrope_is_plain_partial_rope() {
        let budget = Budget::new(1 << 24);
        let pos: Vec<u32> = (5..5 + 300).collect();
        let plain_section = MropeSection {
            section: [32, 0, 0],
            interleaved: false,
        };
        let (pc, ps) = mrope_tables([&pos, &pos, &pos], plain_section, 64, 1e7, &budget).unwrap();
        for section in [
            QWEN,
            MropeSection {
                section: [16, 8, 8],
                interleaved: false,
            },
        ] {
            let (c, s) = mrope_text_tables(5, 300, section, 64, 1e7, &budget).unwrap();
            assert_eq!(bits(&c), bits(&pc), "{section:?} cos");
            assert_eq!(bits(&s), bits(&ps), "{section:?} sin");
        }
        // Column i and i + R/2 hold the same angle, and row 0 at position 5
        // of frequency 0 is cos(5), sin(5).
        let c = pc.to_f32_vec().unwrap();
        assert_eq!(c[0].to_bits(), c[32].to_bits());
        assert_eq!(c[0], 5f64.cos() as f32);
    }

    /// Distinct streams (an image patch) do move the frequencies their
    /// section assigns, and only those. A frequency is compared on cos and
    /// sin together: at the low frequencies the angles are near 1e-4, whose
    /// cos rounds to exactly 1.0 in f32 at either position, so only the sin
    /// shows the move.
    #[test]
    fn distinct_streams_change_exactly_their_frequencies() {
        let budget = Budget::new(1 << 24);
        let t: Vec<u32> = vec![7; 4];
        let h: Vec<u32> = vec![2; 4];
        let w: Vec<u32> = vec![3; 4];
        let (c, s) = mrope_tables([&t, &h, &w], QWEN, 64, 1e7, &budget).unwrap();
        let (ct, st) = mrope_tables([&t, &t, &t], QWEN, 64, 1e7, &budget).unwrap();
        let [c, s, ct, st] = [c, s, ct, st].map(|x| x.to_f32_vec().unwrap());
        for i in 0..32 {
            let same = c[i].to_bits() == ct[i].to_bits() && s[i].to_bits() == st[i].to_bits();
            assert_eq!(same, QWEN.stream(i) == 0, "frequency {i}");
        }
    }

    #[test]
    fn refusals() {
        let budget = Budget::new(1 << 24);
        let p = [1u32, 2];
        let ok = mrope_tables([&p, &p, &p], QWEN, 64, 1e7, &budget);
        assert!(ok.is_ok());
        let range = |r: Result<(Tensor, Tensor), OjasError>| {
            assert!(matches!(r, Err(OjasError::OutOfRange { .. })), "{r:?}")
        };
        range(mrope_tables([&p, &p, &p], QWEN, 62, 1e7, &budget));
        range(mrope_tables([&p, &p, &p], QWEN, 63, 1e7, &budget));
        range(mrope_tables([&p, &p, &p], QWEN, 0, 1e7, &budget));
        range(mrope_tables([&p, &p, &p], QWEN, 64, f64::NAN, &budget));
        range(mrope_tables([&p, &p, &p], QWEN, 64, 1.0, &budget));
        range(mrope_text_tables(u32::MAX, 2, QWEN, 64, 1e7, &budget));
        let short = [1u32];
        assert!(matches!(
            mrope_tables([&p, &short, &p], QWEN, 64, 1e7, &budget),
            Err(OjasError::Shape { .. })
        ));
        assert!(matches!(
            mrope_tables([&[], &[], &[]], QWEN, 64, 1e7, &budget),
            Err(OjasError::Shape { .. })
        ));
    }
}
