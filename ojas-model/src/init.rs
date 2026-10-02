//! Fresh nanolab init (`docs/framework-design.md` §2).
//!
//! Each parameter draws from its own [`CounterRng`] seeded with
//! `seed ^ fnv1a(name)`, through Box-Muller, so a parameter's values depend
//! on the seed and its name only, never on the order parameters are built
//! in. Zeros and ones are exact. torch's RNG stream is not reproduced; a
//! parity run loads torch's init from safetensors instead.

use ojas_core::{Budget, OjasError, Tensor};
use ojas_data::CounterRng;

use crate::names::{param_table, Init, ParamInfo};
use crate::spec::ModelSpec;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// 64-bit FNV-1a of `bytes`.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    fnv1a_extend(FNV_OFFSET, bytes)
}

/// FNV-1a continued from `hash` over `bytes`: `fnv1a(a ‖ b)` is
/// `fnv1a_extend(fnv1a(a), b)`.
pub(crate) fn fnv1a_extend(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
    })
}

/// `2^-53`: one unit in the last place of a 53-bit mantissa in `[0, 1)`.
const ULP53: f64 = 1.0 / (1u64 << 53) as f64;

/// Standard normal pairs by Box-Muller over a [`CounterRng`].
struct Normal {
    rng: CounterRng,
    spare: Option<f64>,
}

impl Normal {
    fn new(seed: u64) -> Self {
        Self {
            rng: CounterRng::new(seed),
            spare: None,
        }
    }

    fn next(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        // `u1` is in (0, 1], so `ln(u1)` is finite; `u2` is in [0, 1).
        let u1 = ((self.rng.next_u64() >> 11) + 1) as f64 * ULP53;
        let u2 = (self.rng.next_u64() >> 11) as f64 * ULP53;
        let radius = (-2.0 * u1.ln()).sqrt();
        let (sin, cos) = (std::f64::consts::TAU * u2).sin_cos();
        self.spare = Some(radius * sin);
        radius * cos
    }
}

/// The values of one parameter under `seed`, row-major.
///
/// `Normal { std }` is `std * z` with `z` from Box-Muller in f64, rounded
/// once to f32. `Zeros` is `+0.0` and `Ones` is `1.0`, exactly. A
/// non-positive or non-finite `std` is [`OjasError::OutOfRange`].
pub fn init_values(info: &ParamInfo, seed: u64) -> Result<Vec<f32>, OjasError> {
    const OP: &str = "init_values";
    let n = info
        .shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| OjasError::OutOfRange {
            op: OP,
            detail: format!("{}: element count overflows", info.name),
        })?;
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| OjasError::CapacityExceeded {
            requested: (n as u64).saturating_mul(4),
            cap: 0,
            live: 0,
        })?;
    match info.init {
        Init::Zeros => out.resize(n, 0.0),
        Init::Ones => out.resize(n, 1.0),
        Init::Normal { std } => {
            if !(std.is_finite() && std > 0.0) {
                return Err(OjasError::OutOfRange {
                    op: OP,
                    detail: format!("{}: std {std} is not positive and finite", info.name),
                });
            }
            let mut normal = Normal::new(seed ^ fnv1a(info.name.as_bytes()));
            let std = f64::from(std);
            out.extend((0..n).map(|_| (std * normal.next()) as f32));
        }
    }
    Ok(out)
}

/// Every parameter of `spec` in [`param_table`] order, as host `F32`
/// tensors charged to `budget`.
pub fn init_params(spec: &ModelSpec, seed: u64, budget: &Budget) -> Result<Vec<Tensor>, OjasError> {
    param_table(spec)?
        .iter()
        .map(|info| Tensor::from_f32(&init_values(info, seed)?, &info.shape, budget))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_matches_the_published_vectors() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x8594_4171_f739_67e8);
        assert_eq!(fnv1a_extend(fnv1a(b"foo"), b"bar"), fnv1a(b"foobar"));
    }

    #[test]
    fn a_bad_std_is_refused() {
        for std in [0.0, -0.02, f32::NAN, f32::INFINITY] {
            let info = ParamInfo {
                name: "w".into(),
                shape: vec![4],
                init: Init::Normal { std },
                group: ojas_cpu::OptimGroup::AdamVector,
                trains: true,
            };
            assert!(init_values(&info, 1).is_err(), "std {std} accepted");
        }
    }
}
